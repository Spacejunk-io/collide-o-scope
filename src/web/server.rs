//! Axum HTTP + WebSocket server for the control panel.
//!
//! Two listeners share one router: plain HTTP on the base port (desktop
//! convenience) and HTTPS with a self-signed certificate on base+1 — the
//! QR code points at the HTTPS URL because iOS only exposes motion sensors
//! to secure contexts. The certificate (SANs: localhost + the LAN IP) is
//! persisted under %LOCALAPPDATA%, so a phone that accepted it once stays
//! trusted across restarts; it regenerates only when the LAN IP changes.
//!
//! Access control: every client must present the per-session token — normally
//! through the app-opened URL or QR code — after which a strict cookie keeps
//! it authenticated. WebSockets and mutation POSTs also require an exact
//! same-origin Origin header. Unknown and cross-origin clients get 403.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{ConnectInfo, DefaultBodyLimit, Path, Query, Request, State, WebSocketUpgrade};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures::{SinkExt, StreamExt};
use tokio::io::AsyncWriteExt;
use tokio::sync::broadcast::error::RecvError;

use super::state::{
    CaptureTargetSnapshot, CompositionRootSnapshot, CreativeImageSourceSnapshot,
    CreativeImageTapSnapshot, CreativeScopeSnapshot, EnqueueOutcome, ImageInputSnapshot,
    MotionScopeSnapshot, PresetTargetSnapshot, RerollScope, SymmetryRouteSnapshot, WebAction,
    WebState, MAX_SCENE_NAME_BYTES,
};
use super::static_files;

const AUTH_COOKIE: &str = "cos_key";
const MAX_WS_MESSAGE_BYTES: usize = 16 * 1024;
const MAX_LOGGED_MESSAGE_CHARS: usize = 256;
const MAX_ACTION_VALUE_DEPTH: usize = 8;
const MAX_ACTION_VALUE_NODES: usize = 512;
const MAX_ACTION_VALUE_STRING_BYTES: usize = 2048;
const MAX_JS_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
/// Audio is decoded to at most ten minutes of canonical mono PCM. Bound the
/// upload itself as well so a malformed or accidental multi-gigabyte file
/// cannot consume library storage before FFmpeg gets a chance to reject it.
const MAX_AUDIO_UPLOAD_BYTES: u64 = 512 * 1024 * 1024;
// Leave room for the atomic reservation prefix and the largest numbered
// collision suffix while staying below Windows' 255 UTF-16-code-unit
// component limit.
const MAX_LIBRARY_FILENAME_UTF16: usize = 220;

fn exceeds_upload_limit(extension: &str, bytes: u64) -> bool {
    crate::audio::is_supported_audio_extension(extension) && bytes > MAX_AUDIO_UPLOAD_BYTES
}

/// Start the web server on a background thread. Returns the local URL.
pub fn spawn(state: Arc<WebState>, port: u16) -> String {
    let local_url = format!("http://127.0.0.1:{port}/?key={}", state.access_token);
    state.mark_control_server_starting(local_url.clone());
    let https_port = port + 1;
    let lan_ip = detect_lan_ip();

    // Self-signed TLS: load or mint a certificate covering the LAN IP.
    let tls = load_or_create_tls(lan_ip);
    if let Err(ref e) = tls {
        log::warn!("TLS unavailable, phone sensors will need HTTP: {e}");
    }

    // The QR points at HTTPS (secure context → iOS sensors work).
    let lan_url = match (lan_ip, tls.is_ok()) {
        (Some(ip), true) => format!("https://{ip}:{https_port}/?key={}", state.access_token),
        (Some(ip), false) => format!("http://{ip}:{port}/?key={}", state.access_token),
        _ => local_url.clone(),
    };
    log::info!("Remote control URL: {lan_url}");
    if let Ok(mut slot) = state.lan_url.write() {
        *slot = lan_url;
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");

    std::thread::spawn(move || {
        rt.block_on(async move {
            let app = control_router(state.clone());

            // HTTPS listener (phones — secure context for sensors).
            if let Ok((cert_chain, key_der)) = tls {
                let https_app = app.clone();
                tokio::spawn(async move {
                    match axum_server::tls_rustls::RustlsConfig::from_der(cert_chain, key_der).await
                    {
                        Ok(config) => {
                            let addr = SocketAddr::from(([0, 0, 0, 0], https_port));
                            log::info!("HTTPS control panel listening on {addr}");
                            if let Err(e) = axum_server::bind_rustls(addr, config)
                                .serve(
                                    https_app.into_make_service_with_connect_info::<SocketAddr>(),
                                )
                                .await
                            {
                                log::warn!("HTTPS server error: {e}");
                            }
                        }
                        Err(e) => log::warn!("TLS config rejected: {e}"),
                    }
                });
            }

            // HTTP listener (desktop convenience, unchanged behavior).
            let addr = format!("0.0.0.0:{port}");
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    state.mark_control_server_unavailable(format!("cannot bind port {port}: {e}"));
                    log::error!(
                        "Cannot bind port {port} ({e}) — is another collide-o-scope \
                         already running? This instance continues without a control panel."
                    );
                    return;
                }
            };
            state.mark_control_server_listening();
            log::info!("Web control panel listening on {addr}");
            if let Err(error) = axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            {
                state.mark_control_server_unavailable(format!("control server stopped: {error}"));
                log::error!("Web control panel stopped: {error}");
            }
        });
    });

    local_url
}

/// Build the exact production router independently of listener ownership.
/// Keeping this seam small lets the transport test bind an ephemeral port and
/// exercise the real authentication, WebSocket upgrade, and action ingress.
fn control_router(state: Arc<WebState>) -> Router {
    Router::new()
        .route("/ws", get(ws_handler))
        .route("/thumb/:filename", get(thumb_handler))
        .route("/preview/:filename/:index", get(preview_handler))
        .route("/qr.svg", get(qr_handler))
        .route(
            "/controller-profile",
            post(controller_profile_handler).layer(DefaultBodyLimit::max(
                crate::controller_profile::CONTROLLER_PROFILE_ACTION_MAX_BYTES,
            )),
        )
        .route(
            "/upload",
            post(upload_handler).layer(DefaultBodyLimit::max(8 * 1024 * 1024 * 1024)),
        )
        .route("/delete", post(delete_handler))
        .fallback(get(static_files::serve))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .with_state(state)
}

/// Load the persisted self-signed certificate, or mint a fresh one when
/// absent or when the LAN IP is no longer among its subject names.
/// Returns (DER cert chain, DER private key).
fn load_or_create_tls(lan_ip: Option<IpAddr>) -> Result<(Vec<Vec<u8>>, Vec<u8>), String> {
    let dir = tls_dir()?;
    let cert_path = dir.join("cert.der");
    let key_path = dir.join("key.der");
    let sans_path = dir.join("sans.txt");

    let mut sans = vec!["localhost".to_string(), "127.0.0.1".to_string()];
    if let Some(ip) = lan_ip {
        sans.push(ip.to_string());
    }

    // Reuse the stored cert if it already covers every name we need —
    // regenerating would invalidate the trust a phone already granted.
    if cert_path.exists() && key_path.exists() {
        if let Ok(stored) = std::fs::read_to_string(&sans_path) {
            let stored: Vec<&str> = stored.lines().collect();
            if sans.iter().all(|s| stored.contains(&s.as_str())) {
                let cert = std::fs::read(&cert_path).map_err(|e| e.to_string())?;
                let key = std::fs::read(&key_path).map_err(|e| e.to_string())?;
                log::info!("TLS: using persisted certificate ({})", dir.display());
                return Ok((vec![cert], key));
            }
        }
        log::info!("TLS: LAN address changed; regenerating certificate");
    }

    let certified = rcgen::generate_simple_self_signed(sans.clone())
        .map_err(|e| format!("certificate generation: {e}"))?;
    let cert_der = certified.cert.der().to_vec();
    let key_der = certified.key_pair.serialize_der();

    std::fs::write(&cert_path, &cert_der).map_err(|e| format!("write cert: {e}"))?;
    std::fs::write(&key_path, &key_der).map_err(|e| format!("write key: {e}"))?;
    std::fs::write(&sans_path, sans.join("\n")).map_err(|e| format!("write sans: {e}"))?;
    log::info!("TLS: generated self-signed certificate ({})", dir.display());

    Ok((vec![cert_der], key_der))
}

/// Certificate storage: %LOCALAPPDATA%\collide-o-scope\tls (or ./.tls).
fn tls_dir() -> Result<PathBuf, String> {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .map(|p| p.join("collide-o-scope").join("tls"))
        .unwrap_or_else(|| PathBuf::from(".tls"));
    std::fs::create_dir_all(&base).map_err(|e| format!("tls dir: {e}"))?;
    Ok(base)
}

/// The machine's outbound-facing LAN address. The UDP "connect" sends no
/// packets — it only asks the OS which interface routes outward.
fn detect_lan_ip() -> Option<IpAddr> {
    let sock = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    let ip = sock.local_addr().ok()?.ip();
    if ip.is_loopback() || ip.is_unspecified() {
        None
    } else {
        Some(ip)
    }
}

/// Browser mutation routes must come from the exact host serving the panel.
/// `Origin` is intentionally required for WebSockets and POSTs: ordinary page
/// navigation has no Origin header, while browser script mutations always do.
fn same_origin(headers: &HeaderMap) -> bool {
    let Some(host) = headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let Some(origin) = headers
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let origin = origin.to_ascii_lowercase();
    let host = host.to_ascii_lowercase();
    origin == format!("http://{host}") || origin == format!("https://{host}")
}

fn requires_same_origin(method: &Method, path: &str) -> bool {
    *method == Method::POST || path == "/ws"
}

fn forbidden(message: &'static str) -> Response {
    (
        StatusCode::FORBIDDEN,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        message,
    )
        .into_response()
}

/// Gate every client, including loopback, with a high-entropy session token.
/// The tokenized startup/QR navigation mints a strict session cookie; all
/// browser mutations additionally require an exact same-origin Origin header.
async fn auth(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<Arc<WebState>>,
    req: Request,
    next: Next,
) -> Response {
    let token = &state.access_token;

    let cookie_ok = req
        .headers()
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .map(|cookies| {
            cookies.split(';').any(|c| {
                c.trim()
                    .strip_prefix(AUTH_COOKIE)
                    .and_then(|rest| rest.strip_prefix('='))
                    .is_some_and(|v| v == token)
            })
        })
        .unwrap_or(false);

    let query_ok = req
        .uri()
        .query()
        .map(|q| {
            q.split('&')
                .any(|p| p.strip_prefix("key=").is_some_and(|v| v == token))
        })
        .unwrap_or(false);

    if !cookie_ok && !query_ok {
        log::warn!("Rejected unauthenticated control client: {}", addr.ip());
        return forbidden(
            "<h3>collide-o-scope</h3><p>Access denied. Open the control panel from the app or scan its QR code.</p>",
        );
    }

    if requires_same_origin(req.method(), req.uri().path()) && !same_origin(req.headers()) {
        log::warn!("Rejected cross-origin control mutation from {}", addr.ip());
        return forbidden("<h3>collide-o-scope</h3><p>Cross-origin control request denied.</p>");
    }

    let mut response = next.run(req).await;
    if query_ok && !cookie_ok {
        if let Ok(cookie) = header::HeaderValue::from_str(&format!(
            "{AUTH_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"
        )) {
            response.headers_mut().append(header::SET_COOKIE, cookie);
        }
    }
    response
}

/// QR code (SVG) of the remote URL, rendered on demand.
async fn qr_handler(State(state): State<Arc<WebState>>) -> impl IntoResponse {
    let url = state.lan_url.read().map(|s| s.clone()).unwrap_or_default();

    match qrcode::QrCode::new(url.as_bytes()) {
        Ok(code) => {
            let svg = code
                .render::<qrcode::render::svg::Color>()
                .min_dimensions(220, 220)
                .quiet_zone(true)
                .build();
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "image/svg+xml")],
                svg,
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("QR generation failed: {e}"),
        )
            .into_response(),
    }
}

async fn controller_profile_handler(
    State(state): State<Arc<WebState>>,
    body: axum::body::Bytes,
) -> Response {
    let request = match crate::controller_profile::ControllerProfileAction::from_json_bytes(&body) {
        Ok(request) => request,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("controller profile request rejected: {error}"),
            )
                .into_response();
        }
    };
    match request {
        crate::controller_profile::ControllerProfileAction::Export {} => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/json; charset=utf-8"),
                (
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=\"controller_profile.json\"",
                ),
            ],
            state.controller_profile_export(),
        )
            .into_response(),
        request @ crate::controller_profile::ControllerProfileAction::Import { .. } => {
            let action = WebAction::ControllerProfile { request };
            if !valid_action(&action, 0) {
                return (
                    StatusCode::BAD_REQUEST,
                    "invalid controller profile document",
                )
                    .into_response();
            }
            match state.enqueue_action(action).await {
                EnqueueOutcome::Added | EnqueueOutcome::Coalesced => {
                    (StatusCode::ACCEPTED, "controller profile import queued").into_response()
                }
                EnqueueOutcome::Dropped => (
                    StatusCode::TOO_MANY_REQUESTS,
                    "controller action queue is full",
                )
                    .into_response(),
            }
        }
    }
}

#[derive(serde::Deserialize)]
struct UploadQuery {
    name: String,
}

#[derive(Debug, PartialEq, Eq)]
struct ValidatedLibraryFilename {
    name: String,
    stem: String,
    extension: String,
}

/// Accept one portable Windows filename, exactly as supplied by the client.
///
/// In particular, do not reduce an untrusted path to its final component:
/// accepting `../clip.mp4` as `clip.mp4` hides a malformed request and makes
/// the boundary dependent on platform path-parsing rules. Windows device
/// names and alternate-data-stream syntax are rejected before any join.
fn validate_library_filename(input: &str) -> Option<ValidatedLibraryFilename> {
    if input.is_empty()
        || input.starts_with('.')
        || input.ends_with([' ', '.'])
        || input.encode_utf16().count() > MAX_LIBRARY_FILENAME_UTF16
        || input.chars().any(|ch| {
            ch.is_control() || matches!(ch, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*')
        })
    {
        return None;
    }

    let (stem, extension) = input.rsplit_once('.')?;
    if stem.is_empty()
        || !(crate::layers::is_supported_visual_extension(extension)
            || crate::audio::is_supported_audio_extension(extension))
    {
        return None;
    }

    // Win32 treats these DOS device identifiers as reserved even when an
    // extension (or another extension) follows. Superscript 1/2/3 are also
    // recognized as COM/LPT digits by Windows.
    let device_stem = input
        .split('.')
        .next()
        .unwrap_or_default()
        .trim_end_matches([' ', '.'])
        .to_ascii_uppercase();
    let numbered_device = device_stem
        .strip_prefix("COM")
        .or_else(|| device_stem.strip_prefix("LPT"))
        .is_some_and(|suffix| {
            matches!(
                suffix,
                "1" | "2"
                    | "3"
                    | "4"
                    | "5"
                    | "6"
                    | "7"
                    | "8"
                    | "9"
                    | "\u{00b9}"
                    | "\u{00b2}"
                    | "\u{00b3}"
            )
        });
    if matches!(
        device_stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$" | "CLOCK$"
    ) || numbered_device
    {
        return None;
    }

    Some(ValidatedLibraryFilename {
        name: input.to_string(),
        stem: stem.to_string(),
        extension: extension.to_ascii_lowercase(),
    })
}

async fn reserve_upload_destination(
    folder: &std::path::Path,
    original_name: &str,
    stem: &str,
    ext: &str,
) -> Result<(String, PathBuf, PathBuf), String> {
    for counter in 0_u32.. {
        let candidate = if counter == 0 {
            original_name.to_string()
        } else {
            format!("{stem} ({counter}).{ext}")
        };
        let final_path = folder.join(&candidate);
        if final_path.exists() {
            continue;
        }
        let reservation_path = folder.join(format!(".upload-reserve-{candidate}"));
        match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&reservation_path)
            .await
        {
            Ok(file) => {
                drop(file);
                // Cover files created by another process between our first
                // existence check and reservation.
                if final_path.exists() {
                    let _ = tokio::fs::remove_file(&reservation_path).await;
                    continue;
                }
                return Ok((candidate, final_path, reservation_path));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("reserve destination: {error}")),
        }
    }
    unreachable!("u32 upload suffix space exhausted")
}

async fn create_unique_upload_temp(
    folder: &std::path::Path,
) -> Result<(PathBuf, tokio::fs::File), String> {
    for _ in 0..8 {
        let mut random = [0_u8; 16];
        getrandom::fill(&mut random).map_err(|error| format!("upload entropy: {error}"))?;
        let suffix: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
        let path = folder.join(format!(".upload-{suffix}.part"));
        match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .await
        {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("create upload temp: {error}")),
        }
    }
    Err("could not allocate a unique upload temp file".to_string())
}

/// Streamed clip upload into the library folder. The body is written in
/// chunks to a temp file (never buffered whole in memory), renamed into
/// place on success, and the render thread is asked to rescan. Names are
/// accepted only as a single Windows-safe component carrying a known visual
/// or audio
/// extension; collisions get a numbered suffix rather than overwriting.
async fn upload_handler(
    State(state): State<Arc<WebState>>,
    Query(query): Query<UploadQuery>,
    headers: HeaderMap,
    body: axum::body::Body,
) -> Response {
    let Some(validated) = validate_library_filename(&query.name) else {
        return (StatusCode::BAD_REQUEST, "unsupported or unsafe filename").into_response();
    };
    let ValidatedLibraryFilename {
        name,
        stem,
        extension: ext,
    } = validated;

    let declared_length = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    if declared_length.is_some_and(|length| exceeds_upload_limit(&ext, length)) {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            "audio upload exceeds the 512 MiB limit",
        )
            .into_response();
    }

    let Some(folder) = state.library_folder.read().ok().and_then(|f| f.clone()) else {
        return (
            StatusCode::CONFLICT,
            "no library folder open — load a folder in the app first",
        )
            .into_response();
    };

    // Atomically reserve a collision-free destination. This prevents two
    // simultaneous same-name uploads from sharing either output or temp data.
    let (final_name, final_path, reservation_path) =
        match reserve_upload_destination(&folder, &name, &stem, &ext).await {
            Ok(reservation) => reservation,
            Err(error) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, error).into_response();
            }
        };
    let (temp_path, mut file) = match create_unique_upload_temp(&folder).await {
        Ok(temp) => temp,
        Err(error) => {
            let _ = tokio::fs::remove_file(&reservation_path).await;
            return (StatusCode::INTERNAL_SERVER_ERROR, error).into_response();
        }
    };

    let mut stream = body.into_data_stream();
    let mut written: u64 = 0;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                if exceeds_upload_limit(&ext, written.saturating_add(bytes.len() as u64)) {
                    drop(file);
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    let _ = tokio::fs::remove_file(&reservation_path).await;
                    return (
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "audio upload exceeds the 512 MiB limit",
                    )
                        .into_response();
                }
                if let Err(e) = file.write_all(&bytes).await {
                    drop(file);
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    let _ = tokio::fs::remove_file(&reservation_path).await;
                    return (StatusCode::INTERNAL_SERVER_ERROR, format!("write: {e}"))
                        .into_response();
                }
                written += bytes.len() as u64;
            }
            Err(e) => {
                drop(file);
                let _ = tokio::fs::remove_file(&temp_path).await;
                let _ = tokio::fs::remove_file(&reservation_path).await;
                return (StatusCode::BAD_REQUEST, format!("stream: {e}")).into_response();
            }
        }
    }
    if file.flush().await.is_err() || written == 0 {
        drop(file);
        let _ = tokio::fs::remove_file(&temp_path).await;
        let _ = tokio::fs::remove_file(&reservation_path).await;
        return (StatusCode::BAD_REQUEST, "empty upload").into_response();
    }
    drop(file);

    if let Err(e) = tokio::fs::rename(&temp_path, &final_path).await {
        let _ = tokio::fs::remove_file(&temp_path).await;
        let _ = tokio::fs::remove_file(&reservation_path).await;
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("rename: {e}")).into_response();
    }
    let _ = tokio::fs::remove_file(&reservation_path).await;

    log::info!("Uploaded clip: {final_name} ({written} bytes)");
    let _ = state.enqueue_action(WebAction::RescanLibrary).await;

    (StatusCode::OK, final_name).into_response()
}

#[derive(serde::Deserialize)]
struct DeleteQuery {
    name: String,
}

/// Remove a clip from the library — to the OS Recycle Bin, never a hard
/// delete, so a mid-set mistake is recoverable. A clip currently loaded
/// in a layer has its file held open by the decoder; that delete fails
/// with a clear message instead of corrupting playback.
async fn delete_handler(
    State(state): State<Arc<WebState>>,
    Query(query): Query<DeleteQuery>,
) -> Response {
    let Some(validated) = validate_library_filename(&query.name) else {
        return (StatusCode::BAD_REQUEST, "not a library clip").into_response();
    };
    let name = validated.name;

    let Some(folder) = state.library_folder.read().ok().and_then(|f| f.clone()) else {
        return (StatusCode::CONFLICT, "no library folder open").into_response();
    };
    let path = folder.join(&name);
    if !path.is_file() {
        return (StatusCode::NOT_FOUND, "clip not found").into_response();
    }

    match tokio::task::spawn_blocking(move || trash::delete(&path)).await {
        Ok(Ok(())) => {
            state.remove_library_media_cache_entry(&name);
            log::info!("Clip moved to Recycle Bin: {name}");
            let _ = state.enqueue_action(WebAction::RescanLibrary).await;
            (StatusCode::OK, name).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::CONFLICT,
            format!("cannot remove (in use by a layer?): {e}"),
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response(),
    }
}

async fn thumb_handler(
    Path(filename): Path<String>,
    State(state): State<Arc<WebState>>,
) -> impl IntoResponse {
    if let Ok(cache) = state.thumbnails.read() {
        if let Some(jpeg) = cache.get(&filename) {
            return (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "image/jpeg")],
                jpeg.clone(),
            )
                .into_response();
        }
    }
    StatusCode::NOT_FOUND.into_response()
}

async fn preview_handler(
    Path((filename, index)): Path<(String, usize)>,
    State(state): State<Arc<WebState>>,
) -> impl IntoResponse {
    if let Ok(cache) = state.preview_frames.read() {
        if let Some(frames) = cache.get(&filename) {
            if let Some(jpeg) = frames.get(index) {
                return (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "image/jpeg")],
                    jpeg.clone(),
                )
                    .into_response();
            }
        }
    }
    StatusCode::NOT_FOUND.into_response()
}

fn safe_log_excerpt(text: &str) -> String {
    let mut excerpt = String::new();
    let mut truncated = false;
    for (index, character) in text.chars().enumerate() {
        if index >= MAX_LOGGED_MESSAGE_CHARS {
            truncated = true;
            break;
        }
        excerpt.push(if character.is_control() {
            '\u{fffd}'
        } else {
            character
        });
    }
    if truncated {
        excerpt.push('\u{2026}');
    }
    excerpt
}

fn valid_identifier(value: &str, max_len: usize) -> bool {
    !value.is_empty() && value.len() <= max_len && !value.chars().any(char::is_control)
}

fn valid_optional_layer_id(layer_id: &Option<String>) -> bool {
    layer_id
        .as_deref()
        .is_none_or(|value| valid_identifier(value, 128))
}

fn valid_optional_stable_id(stable_id: &Option<String>) -> bool {
    stable_id.as_deref().is_none_or(|value| {
        !value.is_empty()
            && value.bytes().all(|byte| byte.is_ascii_digit())
            && value.parse::<u64>().is_ok_and(|id| id != 0)
    })
}

fn valid_required_stable_id(stable_id: &str) -> bool {
    !stable_id.is_empty()
        && stable_id.bytes().all(|byte| byte.is_ascii_digit())
        && stable_id.parse::<u64>().is_ok_and(|id| id != 0)
}

fn valid_capture_target(target: &CaptureTargetSnapshot) -> bool {
    match target {
        CaptureTargetSnapshot::Program => true,
        CaptureTargetSnapshot::Layer { layer_id } => valid_required_stable_id(layer_id),
        CaptureTargetSnapshot::Group { group_id } => valid_required_stable_id(group_id),
    }
}

fn valid_output_endpoint_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn valid_composition_revision(revision: u64) -> bool {
    revision != 0
}

fn valid_creative_scope(scope: &CreativeScopeSnapshot) -> bool {
    match scope {
        CreativeScopeSnapshot::Master => true,
        CreativeScopeSnapshot::Layer { layer_id } => valid_required_stable_id(layer_id),
        CreativeScopeSnapshot::Group { group_id } => valid_required_stable_id(group_id),
    }
}

fn valid_preset_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= crate::preset::PRESET_MAX_NAME_BYTES
        && name.trim() == name
        && !name.chars().any(char::is_control)
}

fn valid_preset_target(target: &PresetTargetSnapshot) -> bool {
    match target {
        PresetTargetSnapshot::Master
        | PresetTargetSnapshot::ControllerProfile
        | PresetTargetSnapshot::StageMap => true,
        PresetTargetSnapshot::Layer { layer_id } => valid_required_stable_id(layer_id),
        PresetTargetSnapshot::Group { group_id } => valid_required_stable_id(group_id),
    }
}

fn valid_preset_capture_target(
    kind: crate::preset::PresetKind,
    target: &PresetTargetSnapshot,
) -> bool {
    valid_preset_target(target)
        && match kind {
            crate::preset::PresetKind::Transform | crate::preset::PresetKind::Rack => matches!(
                target,
                PresetTargetSnapshot::Master
                    | PresetTargetSnapshot::Layer { .. }
                    | PresetTargetSnapshot::Group { .. }
            ),
            crate::preset::PresetKind::Matte => matches!(
                target,
                PresetTargetSnapshot::Layer { .. } | PresetTargetSnapshot::Group { .. }
            ),
            crate::preset::PresetKind::Group => {
                matches!(target, PresetTargetSnapshot::Group { .. })
            }
            crate::preset::PresetKind::ControllerProfile => {
                matches!(target, PresetTargetSnapshot::ControllerProfile)
            }
            crate::preset::PresetKind::StageMap => {
                matches!(target, PresetTargetSnapshot::StageMap)
            }
        }
}

fn valid_node_kind(kind: &str, insertable: bool) -> Option<crate::visual_rack::NodeKindTag> {
    let descriptor = crate::visual_rack::NODE_KIND_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.key == kind)?;
    if insertable
        && matches!(
            descriptor.tag,
            crate::visual_rack::NodeKindTag::LegacyCanonical
                | crate::visual_rack::NodeKindTag::LegacyTemporal
        )
    {
        return None;
    }
    Some(descriptor.tag)
}

fn valid_node_param_value(kind: &str, param: &str, value: &serde_json::Value) -> bool {
    let Some(kind) = valid_node_kind(kind, false) else {
        return false;
    };
    // Legacy markers are immutable execution boundaries. Their enabled/wet/
    // blend values are fixed too, even though authored nodes expose those
    // scalar fields for every ordinary kind.
    if matches!(
        kind,
        crate::visual_rack::NodeKindTag::LegacyCanonical
            | crate::visual_rack::NodeKindTag::LegacyTemporal
    ) {
        return false;
    }
    match param {
        "enabled" => return value.is_boolean(),
        "wet" => return number_in(value, 0.0, 1.0),
        "blend" => {
            return matches!(
                value.as_str(),
                Some(
                    "normal"
                        | "screen"
                        | "multiply"
                        | "difference"
                        | "add"
                        | "subtract"
                        | "darken"
                        | "lighten"
                        | "overlay"
                        | "soft_light"
                        | "hard_light"
                        | "exclusion"
                        | "dodge"
                        | "burn"
                        | "alpha_cut"
                )
            );
        }
        // These are topology/routing fields and use barrier actions.
        "variant"
        | "image_tap"
        | "image_channel"
        | "image_invert"
        | "donor_tap"
        | "structure_tap"
        | "detail_tap"
        | "symmetry_donor0_tap"
        | "symmetry_donor1_tap"
        | "symmetry_motion0_donor"
        | "symmetry_motion1_donor" => {
            return false;
        }
        _ => {}
    }
    let Some(descriptor) = crate::visual_rack::NODE_PARAM_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.kind == kind && descriptor.key == param)
    else {
        return false;
    };
    use crate::visual_rack::NodeParamType;
    match descriptor.value_type {
        NodeParamType::Float => descriptor
            .range
            .is_some_and(|range| number_in(value, f64::from(range[0]), f64::from(range[1]))),
        NodeParamType::Vec2 => descriptor.range.is_some_and(|range| {
            value.as_array().is_some_and(|values| {
                values.len() == 2
                    && values
                        .iter()
                        .all(|value| number_in(value, f64::from(range[0]), f64::from(range[1])))
            })
        }),
        NodeParamType::Color => value.as_array().is_some_and(|values| {
            values.len() == 3 && values.iter().all(|value| number_in(value, 0.0, 1.0))
        }),
        NodeParamType::Bool => value.is_boolean(),
        NodeParamType::Unsigned => value
            .as_u64()
            .is_some_and(|value| value <= u64::from(u32::MAX)),
        NodeParamType::Enum => match param {
            "fit_mode" => matches!(value.as_str(), Some("stretch" | "fit" | "fill" | "native")),
            "edge_mode" => matches!(
                value.as_str(),
                Some("transparent" | "clamp" | "repeat" | "mirror")
            ),
            "sampling" => matches!(value.as_str(), Some("linear" | "nearest")),
            "mode" => matches!(
                value.as_str(),
                Some("keep_bright" | "keep_dark" | "remove_color" | "keep_color")
            ),
            "algorithm" => matches!(
                value.as_str(),
                Some("gaussian" | "perlin" | "salt_pepper" | "blue")
            ),
            // Displace's boundary law. Without this arm the shipped panel's
            // Boundary select is dropped here, before dispatch ever sees it.
            //
            // A discrete law that is declared `NodeParamType::Enum` but absent
            // from this allowlist is silently dropped at ingress while the
            // panel control still renders, so every closed vocabulary must
            // list its exact snake_case tokens here. Displace's boundary was
            // exactly that defect: `set_runtime_node_param` accepted the value
            // while `valid_action` had already dropped it.
            "boundary" => matches!(
                value.as_str(),
                Some("transparent" | "mirror" | "wrap" | "hold")
            ),
            // The Symmetry Field's two discrete authored laws. Both vocabularies
            // are closed and append-only; a neighbouring token from the other
            // one must be refused.
            "symmetry_mode" => matches!(
                value.as_str(),
                Some(
                    "cyclic"
                        | "dihedral"
                        | "planar_p1"
                        | "planar_pm"
                        | "planar_p2"
                        | "planar_pmm"
                        | "log_spiral"
                        | "orbit"
                )
            ),
            "symmetry_boundary" => matches!(
                value.as_str(),
                Some("transparent" | "mirror" | "wrap" | "hold" | "cellular_reentry")
            ),
            "block" => matches!(
                value.as_str(),
                Some("four" | "eight" | "sixteen" | "thirty_two" | "sixty_four")
            ),
            "quantization" => matches!(value.as_str(), Some("off" | "coarse" | "medium" | "fine")),
            _ => false,
        },
        // Routes are stable authored topology and are never edited through the
        // ordinary coalescible parameter action.
        NodeParamType::ImageTap | NodeParamType::MotionDonor => false,
    }
}

/// A group rack node may never read its own group's output on the current
/// frame. Shared by every ordered route action so the three call sites cannot
/// drift apart.
fn creative_self_group_current_frame(
    scope: &CreativeScopeSnapshot,
    route: &CreativeImageTapSnapshot,
) -> bool {
    match (scope, &route.input) {
        (
            CreativeScopeSnapshot::Group { group_id },
            CreativeImageSourceSnapshot::GroupOutput { group_id: producer },
        ) => group_id == producer && route.timing == crate::visual_rack::EdgeTiming::CurrentFrame,
        _ => false,
    }
}

fn valid_creative_route(route: &CreativeImageTapSnapshot) -> bool {
    match &route.input {
        CreativeImageSourceSnapshot::SelectedLayer { layer_id, .. } => {
            valid_required_stable_id(layer_id)
        }
        CreativeImageSourceSnapshot::MissingSelectedLayer { .. }
        | CreativeImageSourceSnapshot::MissingGroupOutput { .. } => false,
        CreativeImageSourceSnapshot::GroupOutput { group_id } => valid_required_stable_id(group_id),
        // The etched gesture field is a master-scope singleton with no ID and
        // no position, and it takes part in no scope ordering, so both timings
        // are authorable. A vocabulary missing from this allowlist is silently
        // dropped at ingress, so the value belongs here as well as in the panel.
        CreativeImageSourceSnapshot::OneBelow
        | CreativeImageSourceSnapshot::AllBelow
        | CreativeImageSourceSnapshot::GestureCanvas
        // The programme tap is the same singleton shape and is N-1 by
        // construction, so both timings are authorable here too.
        | CreativeImageSourceSnapshot::ProgramTap => true,
        CreativeImageSourceSnapshot::CleanProgram => {
            route.timing == crate::visual_rack::EdgeTiming::PreviousFrame
        }
    }
}

/// Shared prefilter for an ordered rack-node route action that carries no
/// channel or invert. A group's rack may not read that same group's output on
/// the current frame; the identical rejection lives in `SetVisualNodeRoute`.
fn valid_creative_node_route(
    scope: &CreativeScopeSnapshot,
    node_id: &str,
    route: &CreativeImageTapSnapshot,
    composition_revision: u64,
) -> bool {
    let self_group = match (scope, &route.input) {
        (
            CreativeScopeSnapshot::Group { group_id },
            CreativeImageSourceSnapshot::GroupOutput { group_id: producer },
        ) => group_id == producer && route.timing == crate::visual_rack::EdgeTiming::CurrentFrame,
        _ => false,
    };
    valid_creative_scope(scope)
        && valid_required_stable_id(node_id)
        && !self_group
        && valid_creative_route(route)
        && valid_composition_revision(composition_revision)
}

fn valid_member_ids(ids: &[String]) -> bool {
    ids.len() <= crate::composition::MAX_COMPOSITION_LAYERS
        && ids.iter().all(|id| valid_required_stable_id(id))
        && ids.iter().collect::<std::collections::HashSet<_>>().len() == ids.len()
}

fn valid_root_item(item: &CompositionRootSnapshot) -> bool {
    match item {
        CompositionRootSnapshot::Layer { layer_id, bus } => {
            valid_required_stable_id(layer_id) && matches!(bus.as_str(), "program" | "a" | "b")
        }
        CompositionRootSnapshot::Group { group_id } => valid_required_stable_id(group_id),
    }
}

fn valid_group_param(param: &str, value: &serde_json::Value) -> bool {
    match param {
        "name" => value.as_str().is_some_and(|name| {
            name.len() <= crate::composition::MAX_GROUP_NAME_BYTES
                && name.trim() == name
                && !name.chars().any(char::is_control)
        }),
        "opacity" => number_in(value, 0.0, 1.0),
        "solo" | "bypass" => value.is_boolean(),
        "bus" => matches!(value.as_str(), Some("program" | "a" | "b")),
        param => valid_transform_edit(param, value),
    }
}

fn valid_group_matte_param(param: &str, value: &serde_json::Value) -> bool {
    match param {
        "amount" | "threshold" => number_in(value, 0.0, 1.0),
        "softness" => number_in(value, 0.0, 0.5),
        _ => false,
    }
}

fn valid_scene_name(name: &str) -> bool {
    name.len() <= MAX_SCENE_NAME_BYTES && name.trim() == name && !name.chars().any(char::is_control)
}

fn number_in(value: &serde_json::Value, min: f64, max: f64) -> bool {
    value
        .as_f64()
        .is_some_and(|number| number.is_finite() && (min..=max).contains(&number))
}

fn integer_in(value: &serde_json::Value, min: u64, max: u64) -> bool {
    value
        .as_u64()
        .is_some_and(|number| (min..=max).contains(&number))
}

/// Closed M3/legacy temporal authoring vocabulary. Invalid enums, unknown
/// fields, non-finite numbers, and out-of-range integers are rejected before
/// they can occupy the bounded render queue.
/// Closed vocabulary for the three authored gesture-canvas controls. There is
/// deliberately no key here that could reach the recorded track.
fn valid_gesture_canvas_edit(param: &str, value: &serde_json::Value) -> bool {
    match param {
        "radius" | "strength" | "retention" => number_in(value, 0.0, 1.0),
        _ => false,
    }
}

fn valid_temporal_edit(param: &str, value: &serde_json::Value) -> bool {
    match param {
        "feedback" => number_in(value, 0.0, 0.95),
        "fb_zoom" => number_in(value, 0.9, 1.1),
        "fb_rotate" => number_in(value, -5.0, 5.0),
        "fb_offset_x" | "fb_offset_y" => number_in(value, -0.5, 0.5),
        "fb_hue_rotate" => number_in(value, -180.0, 180.0),
        "fb_saturation" | "fb_gain_r" | "fb_gain_g" | "fb_gain_b" | "fb_sharpen" => {
            number_in(value, 0.0, 2.0)
        }
        "fb_chroma_displace" => number_in(value, 0.0, 0.05),
        "fb_blur" | "fb_pivot" | "fb_threshold" | "fb_noise" => number_in(value, 0.0, 1.0),
        "fb_drive" => number_in(value, 0.25, 4.0),
        "fb_shape" => matches!(value.as_str(), Some("clamp" | "soft" | "wrap" | "fold")),
        "fb_edge" => matches!(
            value.as_str(),
            Some("transparent" | "mirror" | "wrap" | "hold")
        ),
        "fb_reflect_x" | "fb_reflect_y" | "fb_servo" | "fb_servo_defeated" => value.is_boolean(),
        // The B4 display stage's twenty wire params. Seventeen continuous,
        // three discrete closed vocabularies.
        "disp_il_amount" | "disp_il_twitter" | "disp_il_judder" | "disp_phos_r" | "disp_phos_g"
        | "disp_phos_b" | "disp_scanlines" | "disp_beam_shape" | "disp_mask_strength"
        | "disp_mask_dark" | "disp_bloom" | "disp_bloom_radius" | "disp_halation"
        | "disp_defocus" | "disp_sag" => number_in(value, 0.0, 1.0),
        "disp_phosphor" => number_in(value, 0.0, 0.95),
        "disp_beam_width" => number_in(value, 0.1, 3.0),
        "disp_il_mode" => matches!(value.as_str(), Some("weave" | "bob" | "blend")),
        "disp_model" => matches!(
            value.as_str(),
            Some(
                "flat"
                    | "aperture_grille"
                    | "slot_mask"
                    | "shadow_mask"
                    | "lcd_stripe"
                    | "mono"
                    | "green_screen"
            )
        ),
        "disp_il_order" => value.is_boolean(),
        // The B8 master melting edge's six wire params, all continuous.
        "melt_amount" | "melt_width" => number_in(value, 0.0, 2.0),
        "melt_hold" => number_in(value, 0.0, 1.5),
        "melt_swirl" => number_in(value, -1.0, 1.0),
        "melt_chroma" | "melt_creep" => number_in(value, 0.0, 1.0),
        // The B5 codec mosh's eight continuous wire params plus its one
        // discrete recycle law.
        "mosh_amount"
        | "mosh_key_removal"
        | "mosh_hold"
        | "mosh_drop"
        | "mosh_shuffle"
        | "mosh_rate"
        | "mosh_bitrate_starve"
        | "mosh_resync" => number_in(value, 0.0, 1.0),
        "mosh_recycle" => value.is_boolean(),
        "slitscan" | "slit_axis" => number_in(value, 0.0, 1.0),
        "slit_angle" | "loom_angle" => number_in(value, -180.0, 180.0),
        "slit_map" => matches!(
            value.as_str(),
            Some("ramp" | "brightness" | "radial" | "tbc_ramp" | "sweep")
        ),
        "slit_interp" => value.is_boolean(),
        "key_mode" => integer_in(value, 0, 4),
        "key_threshold" | "loom_amount" | "loom_depth" | "atlas_amount" | "atlas_collision"
        | "garden_amount" | "garden_threshold" | "garden_decay" => number_in(value, 0.0, 1.0),
        "key_softness" | "garden_softness" => number_in(value, 0.0, 0.5),
        "key_history" => integer_in(value, 1, 23),
        "loom_topology" => matches!(
            value.as_str(),
            Some("linear" | "radial" | "spiral" | "contour" | "folded" | "kaleidoscopic")
        ),
        "loom_interpolation" => matches!(value.as_str(), Some("floor" | "linear")),
        "loom_phase" => number_in(value, -1_000.0, 1_000.0),
        "loom_scale" => number_in(value, 0.01, 100.0),
        "loom_folds" => integer_in(value, 1, 16),
        "loom_quantization" => integer_in(value, 0, 24),
        "atlas_seed" | "score_seed" | "garden_max_hold_ticks" => {
            integer_in(value, 0, u64::from(u32::MAX))
        }
        "atlas_territories" => integer_in(value, 1, 64),
        "garden_gate" => matches!(
            value.as_str(),
            Some(
                "temporal_delta"
                    | "luma"
                    | "chroma"
                    | "cellular_ridge"
                    | "audio_energy"
                    | "audio_onset"
                    | "matte"
                    | "motion"
            )
        ),
        "score_enabled" => value.is_boolean(),
        "score_state_count" => integer_in(value, 2, 16),
        "score_trigger" => matches!(
            value.as_str(),
            Some("boundary" | "downbeat" | "audio_onset" | "manual")
        ),
        "score_loop_driver" => value
            .as_str()
            .is_some_and(|driver| driver == "none" || valid_required_stable_id(driver)),
        "reset_loop_boundary" | "reset_downbeat" => {
            matches!(value.as_str(), Some("none" | "score" | "memory" | "all"))
        }
        _ => false,
    }
}

fn valid_motion_scope(scope: &MotionScopeSnapshot) -> bool {
    match scope {
        MotionScopeSnapshot::Master => true,
        MotionScopeSnapshot::Layer { layer_id } => valid_required_stable_id(layer_id),
    }
}

/// Closed M4 vocabulary. Algorithm provenance and donor topology have no
/// scalar ingress path; the latter uses its revision-protected barrier action.
fn valid_motion_edit(scope: &MotionScopeSnapshot, param: &str, value: &serde_json::Value) -> bool {
    if !valid_motion_scope(scope) {
        return false;
    }
    let common = match param {
        "field_source" => matches!(
            value.as_str(),
            Some(
                "auto"
                    | "codec_vectors"
                    | "lattice"
                    | "procedural_curl"
                    | "procedural_radial"
                    | "procedural_spiral"
                    | "procedural_contour"
                    | "procedural_chroma"
                    | "procedural_weave"
            )
        ),
        "field_scale" => number_in(value, 0.0, 1.0),
        "field_rate" => number_in(value, -2.0, 2.0),
        "stretch" | "edge_repel" | "vector_trash" => number_in(value, 0.0, 1.0),
        "trash_block_size" => number_in(value, 2.0, 256.0),
        "lattice_quality" => matches!(value.as_str(), Some("draft" | "live" | "high")),
        "shutter_angle" => number_in(value, 0.0, 360.0),
        "shutter_phase" => number_in(value, -1.0, 1.0),
        "shutter_curvature" => number_in(value, -2.0, 2.0),
        "shutter_chromatic_lag" => number_in(value, 0.0, 1.0),
        "shutter_quality" => {
            matches!(value.as_str(), Some("sharp" | "draft" | "live" | "high"))
        }
        _ => false,
    };
    if common {
        return true;
    }
    matches!(scope, MotionScopeSnapshot::Layer { .. })
        && match param {
            "transplant_amount" | "confidence_threshold" | "refresh" | "decay" | "occlusion" => {
                number_in(value, 0.0, 1.0)
            }
            "confidence_softness" => number_in(value, 0.0, 0.5),
            "carrier" => matches!(
                value.as_str(),
                Some("transparent" | "black" | "first_source_frame")
            ),
            // Field Collider values. The two donors are topology and travel on
            // the separate revision-protected barrier; these three are ordinary
            // coalescible values. Layer scope only, because a collider needs a
            // Faraday carrier and the master owns none.
            "collider_enabled" => value.is_boolean(),
            "collider_mode" => matches!(
                value.as_str(),
                Some("sum" | "difference" | "curl" | "projection" | "collision_boundary")
            ),
            "collider_boundary" => matches!(
                value.as_str(),
                Some("transparent" | "mirror" | "wrap" | "hold")
            ),
            _ => false,
        }
}

/// Closed scalar-edit vocabulary for prepared clip transport. Structured
/// config replacement is deliberately absent: each action has one semantic
/// destination and therefore a safe queue coalescing key.
fn valid_clip_transport_edit(param: &str, value: &serde_json::Value) -> bool {
    match param {
        "direction" => matches!(value.as_str(), Some("forward" | "reverse")),
        "end_behavior" => matches!(
            value.as_str(),
            Some("loop" | "ping_pong" | "one_shot" | "hold")
        ),
        "trigger_mode" => matches!(value.as_str(), Some("immediate" | "next_beat" | "next_bar")),
        "in_point" | "out_point" => number_in(value, 0.0, 1.0),
        "rate" => number_in(value, 0.0, 16.0),
        "sample_fps" => value.is_null() || number_in(value, 0.25, 480.0),
        "beat_grid_enabled" | "sync_to_program" | "beat_loop_enabled" => value.is_boolean(),
        "clip_bpm" => number_in(value, 1.0, 999.0),
        "length_beats" => value.is_null() || number_in(value, 1.0 / 64.0, 65_536.0),
        "beats_per_bar" => integer_in(value, 1, 32),
        "beat_loop_start" => number_in(value, 0.0, 65_536.0),
        "beat_loop_length" => number_in(value, 1.0 / 64.0, 64.0),
        _ => false,
    }
}

fn valid_matte_edit(param: &str, value: &serde_json::Value) -> bool {
    match param {
        "enabled" | "invert" => value.is_boolean(),
        "channel" => matches!(
            value.as_str(),
            Some("alpha" | "luma" | "red" | "green" | "blue")
        ),
        "amount" | "threshold" | "softness" => number_in(value, 0.0, 1.0),
        _ => false,
    }
}

fn valid_image_input(input: &ImageInputSnapshot) -> bool {
    match input {
        ImageInputSnapshot::SelectedLayer { layer_id, .. } => valid_required_stable_id(layer_id),
        ImageInputSnapshot::OneBelow
        | ImageInputSnapshot::AllBelow
        | ImageInputSnapshot::CleanProgram
        | ImageInputSnapshot::ProgramHistory => true,
        ImageInputSnapshot::GroupOutput { group_id } => valid_required_stable_id(group_id),
        // Restore-only diagnostics.
        ImageInputSnapshot::MissingSelectedLayer { .. }
        | ImageInputSnapshot::MissingGroupOutput { .. } => false,
    }
}

fn is_layer_routing_target(value: &serde_json::Value) -> bool {
    let Some(target) = value.as_str() else {
        return false;
    };
    let target = crate::modulation::canonical_target(target);
    let Some(rest) = target.strip_prefix("layer") else {
        return false;
    };
    let Some((number, _suffix)) = rest.split_once('_') else {
        return false;
    };
    number.parse::<usize>().is_ok_and(|layer| layer != 0)
        && crate::modulation::is_valid_target(target.as_ref())
}

fn valid_routing_edit(param: &str, value: &serde_json::Value) -> bool {
    match param {
        "source" => value
            .as_str()
            .and_then(crate::modulation::ModSource::try_from_str)
            .is_some(),
        "target" => value
            .as_str()
            .is_some_and(crate::modulation::is_valid_target),
        "depth" => number_in(value, -1.0, 1.0),
        "curve" => matches!(
            value.as_str(),
            Some("linear" | "exp" | "log" | "s_curve" | "steps")
        ),
        "curve_amount" => number_in(value, -2.0, 2.0),
        "attack" | "release" => number_in(value, 0.0, 10.0),
        _ => false,
    }
}

fn valid_json_value(value: &serde_json::Value) -> bool {
    fn visit(value: &serde_json::Value, depth: usize, remaining: &mut usize) -> bool {
        if depth > MAX_ACTION_VALUE_DEPTH || *remaining == 0 {
            return false;
        }
        *remaining -= 1;
        match value {
            serde_json::Value::Null | serde_json::Value::Bool(_) => true,
            serde_json::Value::Number(number) => number
                .as_f64()
                .is_some_and(|number| number.is_finite() && number.abs() <= f64::from(f32::MAX)),
            serde_json::Value::String(value) => {
                value.len() <= MAX_ACTION_VALUE_STRING_BYTES && !value.chars().any(char::is_control)
            }
            serde_json::Value::Array(values) => values
                .iter()
                .all(|value| visit(value, depth + 1, remaining)),
            serde_json::Value::Object(values) => values.iter().all(|(key, value)| {
                valid_identifier(key, 128) && visit(value, depth + 1, remaining)
            }),
        }
    }

    let mut remaining = MAX_ACTION_VALUE_NODES;
    visit(value, 0, &mut remaining)
}

fn valid_f32(value: f32) -> bool {
    value.is_finite()
}

/// Typed-field counterpart of `number_in`: finite and inside an inclusive
/// range, for actions whose values arrive as `f32` rather than JSON numbers.
fn f32_in(value: f32, min: f32, max: f32) -> bool {
    value.is_finite() && (min..=max).contains(&value)
}

fn valid_f64_for_f32(value: f64) -> bool {
    value.is_finite() && value.abs() <= f64::from(f32::MAX)
}

fn transform_number(value: &serde_json::Value) -> Option<f32> {
    value
        .as_f64()
        .filter(|number| number.is_finite() && number.abs() <= f64::from(f32::MAX))
        .map(|number| number as f32)
        .filter(|number| number.is_finite())
}

/// Transform ingress is intentionally stricter than the older generic effect
/// protocol: unknown fields and values outside the authored domain are
/// rejected instead of entering the render queue as silent no-ops.
fn valid_transform_edit(param: &str, value: &serde_json::Value) -> bool {
    let bounded = |min: f32, max: f32| {
        transform_number(value).is_some_and(|number| (min..=max).contains(&number))
    };
    match param {
        "position_x" | "position_y" => {
            bounded(crate::spatial::POSITION_MIN, crate::spatial::POSITION_MAX)
        }
        "scale_x" | "scale_y" => bounded(crate::spatial::SCALE_MIN, crate::spatial::SCALE_MAX),
        "anchor_x" | "anchor_y" => bounded(crate::spatial::ANCHOR_MIN, crate::spatial::ANCHOR_MAX),
        "rotation_deg" | "skew_axis_deg" => bounded(-180.0, 180.0),
        "skew_deg" => bounded(
            -crate::spatial::SKEW_LIMIT_DEGREES,
            crate::spatial::SKEW_LIMIT_DEGREES,
        ),
        // The shared SpatialTransform sanitizer enforces the paired crop
        // extent after this one-field edit is applied to current state.
        "crop_left" | "crop_top" | "crop_right" | "crop_bottom" => {
            bounded(0.0, crate::spatial::CROP_MAX)
        }
        "fit" => matches!(value.as_str(), Some("stretch" | "fit" | "fill" | "native")),
        "edge" => matches!(
            value.as_str(),
            Some("transparent" | "clamp" | "repeat" | "mirror")
        ),
        "sampling" => matches!(value.as_str(), Some("linear" | "nearest")),
        _ => false,
    }
}

fn valid_complete_transform(transform: &crate::spatial::SpatialTransform) -> bool {
    *transform == transform.sanitized()
}

fn valid_action(action: &WebAction, depth: usize) -> bool {
    if depth > 2 {
        return false;
    }
    match action {
        WebAction::Quantized { inner } => {
            !matches!(
                inner.as_ref(),
                WebAction::SetMediaSafetyMode { .. }
                    | WebAction::SetNewLayerFit { .. }
                    | WebAction::SetProxySettings { .. }
                    | WebAction::ClearTemporalMemory
                    | WebAction::TriggerCollisionScore
                    | WebAction::TriggerRefreshGarden
                    | WebAction::SetRefreshGardenMatteRoute { .. }
                    | WebAction::SetRefreshGardenMotionRoute { .. }
                    | WebAction::ClearTemporalEventTrack
                    // A gesture is authored against the frame it was drawn in.
                    // Holding a sample or an arm/disarm edge for the next
                    // downbeat would rewrite the recorded stream.
                    | WebAction::GestureSample { .. }
                    // A B10 bend edge is authored against the frame it was
                    // played in; latching one to a downbeat would move an
                    // envelope trigger the operator timed by hand.
                    | WebAction::BendPad { .. }
                    | WebAction::SetGestureRecording { .. }
                    // The B9 transports are the same class of barrier: an
                    // arm/disarm edge held for a downbeat would attach a take
                    // decision to the wrong stretch of program time.
                    | WebAction::SetPerformanceRecording { .. }
                    | WebAction::SetPerformancePlayback { .. }
                    | WebAction::ClearPerformanceTake
                    | WebAction::SetMotionDonor { .. }
                    | WebAction::SetMotionColliderInput { .. }
                    | WebAction::ClearMotionMemory
                    | WebAction::SetVisualNodeDisplaceRoute { .. }
                    | WebAction::SetVisualNodeSymmetryRoute { .. }
                    // Every ordered, revision-protected route action belongs
                    // here as a class, not one kind at a time. Residual's is
                    // the same barrier and is documented as never latched, but
                    // neither branch could list it: the rule was authored where
                    // Residual did not exist, and Residual was authored where
                    // the rule did not. Today `quantized_action_key` answers
                    // None for all three, so an admitted wrapper would execute
                    // immediately rather than defer — which is exactly why the
                    // refusal has to live here, at the gate, instead of resting
                    // on a downstream lookup that a later latching change could
                    // extend without noticing this omission.
                    | WebAction::SetVisualNodeResidualRoute { .. }
                    | WebAction::BeginHistoryGesture { .. }
                    | WebAction::EndHistoryGesture { .. }
                    | WebAction::CancelHistoryGesture { .. }
                    | WebAction::UndoManual
                    | WebAction::RedoManual
                    | WebAction::CaptureScopedPreset { .. }
                    | WebAction::ApplyScopedPreset { .. }
                    | WebAction::DeleteScopedPreset { .. }
                    | WebAction::RestoreRecoveryJournal
                    | WebAction::DiscardRecoveryJournal
                    | WebAction::ControllerProfile { .. }
            ) && valid_action(inner, depth + 1)
        }
        WebAction::ControllerProfile { request } => match request {
            crate::controller_profile::ControllerProfileAction::Import { document } => {
                document.to_json_bytes().is_ok()
            }
            crate::controller_profile::ControllerProfileAction::Export {} => true,
        },
        // Same predicate the engine's authoring door applies; refusing an
        // invalid fixed frame rate here keeps the queue free of tuples the
        // engine would only refuse at the handler.
        WebAction::SetProxySettings {
            scale,
            frame_rate,
            include_audio,
        } => crate::proxy::ProxySettings::authored(*scale, *frame_rate, *include_audio).is_ok(),
        WebAction::SetVisualNodeParam {
            scope,
            node_id,
            node_kind,
            param,
            value,
            composition_revision,
        } => {
            valid_creative_scope(scope)
                && valid_required_stable_id(node_id)
                && valid_composition_revision(*composition_revision)
                && valid_node_param_value(node_kind, param, value)
        }
        WebAction::InsertVisualNode {
            scope,
            index,
            node_kind,
            composition_revision,
        } => {
            valid_creative_scope(scope)
                && *index < crate::visual_rack::MAX_NODES_PER_RACK
                && valid_node_kind(node_kind, true).is_some()
                && valid_composition_revision(*composition_revision)
        }
        WebAction::RemoveVisualNode {
            scope,
            node_id,
            composition_revision,
        }
        | WebAction::MoveVisualNode {
            scope,
            node_id,
            composition_revision,
            ..
        } => {
            valid_creative_scope(scope)
                && valid_required_stable_id(node_id)
                && valid_composition_revision(*composition_revision)
                && match action {
                    WebAction::MoveVisualNode { to, .. } => {
                        *to < crate::visual_rack::MAX_NODES_PER_RACK
                    }
                    _ => true,
                }
        }
        WebAction::SetVisualNodeMaskVariant {
            scope,
            node_id,
            variant,
            composition_revision,
        } => {
            valid_creative_scope(scope)
                && valid_required_stable_id(node_id)
                && matches!(variant.as_str(), "rectangle" | "ellipse" | "image")
                && valid_composition_revision(*composition_revision)
        }
        WebAction::SetVisualNodeRoute {
            scope,
            node_id,
            route,
            channel,
            composition_revision,
            ..
        } => {
            valid_creative_scope(scope)
                && valid_required_stable_id(node_id)
                && !creative_self_group_current_frame(scope, route)
                && valid_creative_route(route)
                && matches!(
                    channel.as_str(),
                    "alpha" | "luma" | "red" | "green" | "blue"
                )
                && valid_composition_revision(*composition_revision)
        }
        // Displace, Residual, and Symmetry all rewrite the image dependency
        // graph, so they take the same decimal-ID, non-zero-revision,
        // tombstone, and self-group current-frame prefilters as
        // `SetVisualNodeRoute` instead of falling through to the permissive
        // tail.
        WebAction::SetVisualNodeSymmetryRoute {
            scope,
            node_id,
            route,
            composition_revision,
        } => {
            let slot_valid = match route {
                SymmetryRouteSnapshot::Image { index, .. } => {
                    usize::from(*index) < crate::symmetry::SYMMETRY_IMAGE_SLOTS
                }
                SymmetryRouteSnapshot::Motion { index, .. } => {
                    usize::from(*index) < crate::symmetry::SYMMETRY_MOTION_SLOTS
                }
            };
            let payload_valid = match route {
                SymmetryRouteSnapshot::Image { route, .. } => {
                    !creative_self_group_current_frame(scope, route) && valid_creative_route(route)
                }
                // A motion route names a stable layer or clears the slot. It
                // never carries a tombstone, a stage, or a timing.
                SymmetryRouteSnapshot::Motion { layer_id, .. } => {
                    layer_id.as_deref().is_none_or(valid_required_stable_id)
                }
            };
            valid_creative_scope(scope)
                && valid_required_stable_id(node_id)
                && slot_valid
                && payload_valid
                && valid_composition_revision(*composition_revision)
        }
        // The named single- and two-input node routes are ordered topology
        // barriers with no channel or invert of their own. They still need
        // every prefilter `SetVisualNodeRoute` applies, including the
        // group's-own-output current-frame rejection, so a hostile message
        // never reaches the bounded queue in the first place.
        WebAction::SetVisualNodeDisplaceRoute {
            scope,
            node_id,
            route,
            composition_revision,
        } => valid_creative_node_route(scope, node_id, route, *composition_revision),
        WebAction::SetVisualNodeResidualRoute {
            scope,
            node_id,
            route,
            composition_revision,
            // Every `ResidualRouteSlotSnapshot` token is a real authored slot;
            // an unknown token is already a deserialization rejection.
            slot: _,
        } => valid_creative_node_route(scope, node_id, route, *composition_revision),
        WebAction::SetCompositionGroupMatteRoute {
            group_id,
            route,
            channel,
            composition_revision,
            ..
        } => {
            valid_required_stable_id(group_id)
                && route.as_ref().is_none_or(valid_creative_route)
                && matches!(
                    channel.as_str(),
                    "alpha" | "luma" | "red" | "green" | "blue"
                )
                && valid_composition_revision(*composition_revision)
        }
        WebAction::SetCompositionGroupMatteParam {
            group_id,
            param,
            value,
            composition_revision,
        } => {
            valid_required_stable_id(group_id)
                && valid_group_matte_param(param, value)
                && valid_composition_revision(*composition_revision)
        }
        WebAction::SetCompositionGroupParam {
            group_id,
            param,
            value,
            composition_revision,
        } => {
            valid_required_stable_id(group_id)
                && valid_group_param(param, value)
                && valid_composition_revision(*composition_revision)
        }
        WebAction::CreateCompositionGroup {
            name,
            member_layer_ids,
            composition_revision,
            ..
        } => {
            name.len() <= crate::composition::MAX_GROUP_NAME_BYTES
                && name.trim() == name
                && !name.chars().any(char::is_control)
                && valid_member_ids(member_layer_ids)
                && valid_composition_revision(*composition_revision)
        }
        WebAction::RemoveCompositionGroup {
            group_id,
            composition_revision,
        } => {
            valid_required_stable_id(group_id) && valid_composition_revision(*composition_revision)
        }
        WebAction::SetCompositionGroupMembers {
            group_id,
            member_layer_ids,
            composition_revision,
        } => {
            valid_required_stable_id(group_id)
                && valid_member_ids(member_layer_ids)
                && valid_composition_revision(*composition_revision)
        }
        WebAction::MoveCompositionRootItem {
            item,
            composition_revision,
            ..
        } => valid_root_item(item) && valid_composition_revision(*composition_revision),
        WebAction::SetCompositionBusCrossfade { value } => {
            value.is_finite() && (0.0..=1.0).contains(value)
        }
        WebAction::SetCompositionBusMixParam { param, value } => {
            crate::mixing_boundary::BusMixerEdit::parse(param, value).is_some()
        }
        WebAction::SetCompositionLayerBus {
            layer_id,
            bus,
            composition_revision,
        } => {
            valid_required_stable_id(layer_id)
                && matches!(bus.as_str(), "program" | "a" | "b")
                && valid_composition_revision(*composition_revision)
        }
        WebAction::AddLayer { filename } => valid_identifier(filename, 1024),
        WebAction::LoadClipIntoSlot {
            layer_id, filename, ..
        } => valid_required_stable_id(layer_id) && valid_identifier(filename, 1024),
        WebAction::RemoveClipSlot { layer_id, .. }
        | WebAction::ActivateClipSlot { layer_id, .. }
        | WebAction::SetClipCue { layer_id, .. }
        | WebAction::RemoveClipCue { layer_id, .. }
        | WebAction::TriggerClipCue { layer_id, .. }
        | WebAction::SeekClipSlot { layer_id, .. }
        | WebAction::SeekClipSlotTimecode { layer_id, .. } => valid_required_stable_id(layer_id),
        WebAction::SetClipTransport {
            layer_id,
            param,
            value,
            ..
        } => {
            valid_required_stable_id(layer_id)
                && valid_identifier(param, 64)
                && valid_clip_transport_edit(param, value)
        }
        WebAction::CaptureScene { name, .. } => valid_scene_name(name),
        WebAction::PrepareScene { .. }
        | WebAction::RemoveScene { .. }
        | WebAction::TriggerScene { .. } => true,
        WebAction::AddSpoutLayer { sender } => valid_identifier(sender, 255),
        WebAction::SetLayerParam {
            layer_id,
            param,
            value,
            ..
        } => {
            valid_optional_layer_id(layer_id)
                && valid_identifier(param, 64)
                && valid_json_value(value)
                && (param != "blend_mode"
                    || value
                        .as_str()
                        .and_then(crate::layers::BlendMode::from_key)
                        .is_some())
        }
        WebAction::SetLayerEffect {
            layer_id,
            param,
            value,
            ..
        } => {
            valid_optional_layer_id(layer_id)
                && valid_identifier(param, 64)
                && valid_json_value(value)
        }
        WebAction::AddPatternLayer | WebAction::AddTextLayer => true,
        // The single shared parse tables answer both this gate and the
        // engine applier, so the accepted and applied vocabularies are
        // structurally one — the B8 BusMixerEdit law.
        WebAction::SetLayerPattern {
            layer_id,
            param,
            value,
            ..
        } => {
            valid_optional_layer_id(layer_id)
                && valid_identifier(param, 64)
                && crate::pattern_synth::PatternSynthEdit::parse(param, value).is_some()
        }
        WebAction::SetLayerText {
            layer_id,
            param,
            value,
            ..
        } => {
            valid_optional_layer_id(layer_id)
                && param.len() <= 64
                && crate::text_page::TextPageEdit::parse(param, value).is_some()
        }
        WebAction::SetLayerTransform {
            layer_id,
            param,
            value,
            ..
        } => {
            layer_id.is_some()
                && valid_optional_stable_id(layer_id)
                && valid_identifier(param, 64)
                && valid_transform_edit(param, value)
        }
        WebAction::ResetLayerTransform { layer_id, .. } => {
            layer_id.is_some() && valid_optional_stable_id(layer_id)
        }
        WebAction::ApplyLayerTransform {
            layer_id,
            transform,
            ..
        } => {
            layer_id.is_some()
                && valid_optional_stable_id(layer_id)
                && valid_complete_transform(transform)
        }
        WebAction::RemoveLayer { layer_id, .. }
        | WebAction::ResetLayerFx { layer_id, .. }
        | WebAction::SetLayerVisibility { layer_id, .. }
        | WebAction::SetLayerPaused { layer_id, .. }
        | WebAction::MoveLayer { layer_id, .. } => valid_optional_layer_id(layer_id),
        WebAction::SetLayerRerollOnLoop { layer_id, .. } => valid_optional_stable_id(layer_id),
        WebAction::SetLayerMatteParam {
            layer_id,
            param,
            value,
            composition_revision,
        } => {
            valid_required_stable_id(layer_id)
                && valid_identifier(param, 64)
                && valid_matte_edit(param, value)
                && composition_revision.is_none_or(|revision| revision != 0)
        }
        WebAction::SetLayerMatteInput {
            layer_id,
            input,
            composition_revision,
        } => {
            valid_required_stable_id(layer_id)
                && valid_image_input(input)
                && composition_revision.is_none_or(|revision| revision != 0)
        }
        WebAction::Reroll {
            scope,
            index,
            layer_id,
            group_id,
            stack_revision,
            amount,
            ..
        } => {
            amount.is_finite()
                && (0.0..=2.0).contains(amount)
                && match scope {
                    RerollScope::Master => {
                        index.is_none()
                            && layer_id.is_none()
                            && group_id.is_none()
                            && stack_revision.is_none()
                    }
                    RerollScope::Layer => {
                        index.is_some()
                            && layer_id.is_some()
                            && group_id.is_none()
                            && valid_optional_stable_id(layer_id)
                            && stack_revision.is_none()
                    }
                    RerollScope::Group => {
                        index.is_none()
                            && layer_id.is_none()
                            && group_id.as_deref().is_some_and(valid_required_stable_id)
                            && stack_revision.is_none()
                    }
                    RerollScope::All => {
                        index.is_none()
                            && layer_id.is_none()
                            && group_id.is_none()
                            && stack_revision.is_some_and(|revision| revision != 0)
                    }
                }
        }
        WebAction::SetParam { param, value }
        | WebAction::SetNtscParam { param, value }
        | WebAction::SetAudio { param, value }
        | WebAction::SetMidi { param, value } => {
            valid_identifier(param, 64) && valid_json_value(value)
        }
        WebAction::SetTemporal { param, value } => {
            valid_identifier(param, 64) && valid_temporal_edit(param, value)
        }
        WebAction::ClearTemporalMemory
        | WebAction::TriggerCollisionScore
        | WebAction::TriggerRefreshGarden
        | WebAction::ClearTemporalEventTrack => true,
        WebAction::SetRefreshGardenMatteRoute {
            layer_id,
            layer_stack_revision,
            ..
        }
        | WebAction::SetRefreshGardenMotionRoute {
            layer_id,
            layer_stack_revision,
        } => layer_id.as_deref().is_none_or(valid_required_stable_id) && *layer_stack_revision != 0,
        WebAction::SetMotion {
            scope,
            param,
            value,
        } => valid_identifier(param, 64) && valid_motion_edit(scope, param, value),
        WebAction::SetMotionDonor {
            layer_id,
            donor_layer_id,
            layer_stack_revision,
        } => {
            valid_required_stable_id(layer_id)
                && donor_layer_id
                    .as_deref()
                    .is_none_or(valid_required_stable_id)
                && donor_layer_id.as_deref() != Some(layer_id.as_str())
                && *layer_stack_revision != 0
        }
        WebAction::SetMotionColliderInput {
            layer_id,
            donor_layer_id,
            layer_stack_revision,
            ..
        } => {
            // Deliberately no self-donation refusal here, unlike the Faraday
            // donor above: a collider input MAY name its own recipient. The one
            // aliasing law — A and B may never name the same layer — depends on
            // the partner slot's current value and is therefore answered by the
            // engine, which is the only side that knows it. `input` needs no
            // check: an unknown token never deserializes into the closed enum.
            valid_required_stable_id(layer_id)
                && donor_layer_id
                    .as_deref()
                    .is_none_or(valid_required_stable_id)
                && *layer_stack_revision != 0
        }
        WebAction::ClearMotionMemory => true,
        WebAction::SetMasterTransform { param, value } => {
            valid_identifier(param, 64) && valid_transform_edit(param, value)
        }
        WebAction::ApplyMasterTransform { transform } => valid_complete_transform(transform),
        WebAction::SetLfo { param, value, .. } => {
            valid_identifier(param, 64) && valid_json_value(value)
        }
        // B10: the envelope vocabulary is closed at the gate exactly as the
        // engine applier closes it, so the queue never carries a tuple the
        // engine would only refuse later.
        WebAction::SetEnvelope {
            index,
            param,
            value,
        } => {
            *index < crate::modulation::NUM_ENVELOPES
                && match param.as_str() {
                    "attack" | "decay" => value.as_f64().is_some_and(f64::is_finite),
                    "trigger" => value.as_str().is_some_and(|token| {
                        crate::modulation::EnvelopeTrigger::try_from_str(token).is_some()
                    }),
                    "mode" => value.as_str().is_some_and(|token| {
                        crate::modulation::EnvelopeMode::try_from_str(token).is_some()
                    }),
                    _ => false,
                }
        }
        WebAction::SetMacro { index, value } => {
            *index < crate::modulation::NUM_MACROS && valid_f32(*value)
        }
        WebAction::SetModSeed { .. } => true,
        WebAction::BendPad { index, .. } => *index < crate::modulation::NUM_BENDS,
        WebAction::SetRouting {
            route_id,
            target_layer_id,
            layer_stack_revision,
            param,
            value,
            ..
        } => {
            valid_optional_stable_id(route_id)
                && valid_optional_stable_id(target_layer_id)
                && layer_stack_revision.is_none_or(|revision| revision != 0)
                && match target_layer_id {
                    Some(_) => param == "target" && is_layer_routing_target(value),
                    None => layer_stack_revision.is_none() && valid_routing_edit(param, value),
                }
                && valid_identifier(param, 64)
                && valid_json_value(value)
        }
        WebAction::RemoveRouting { route_id, .. } => valid_optional_stable_id(route_id),
        WebAction::SetGyroConfig {
            axis, param, value, ..
        }
        | WebAction::SetPadConfig {
            axis, param, value, ..
        } => valid_identifier(axis, 16) && valid_identifier(param, 64) && valid_json_value(value),
        WebAction::SetBpm { value } | WebAction::SetMorph { value } => valid_f32(*value),
        WebAction::Gyro { alpha, beta, gamma } => {
            valid_f32(*alpha) && valid_f32(*beta) && valid_f32(*gamma)
        }
        WebAction::Pad { x, y, .. } => valid_f32(*x) && valid_f32(*y),
        // Every gesture field is closed and finite-bounded before it can
        // occupy the render queue. `phase` and `mode` are typed engine enums,
        // so an unknown token already failed deserialization; the numeric
        // fields are checked here against the same ranges the quantizer uses,
        // and the stroke identity is checked against the same constant the
        // ingest validator enforces, so the queue never carries a sample the
        // adapter would refuse.
        WebAction::GestureSample {
            stroke,
            x,
            y,
            pressure,
            direction_x,
            direction_y,
            ..
        } => {
            usize::from(*stroke) < crate::gesture::MAX_ACTIVE_STROKES
                && f32_in(*x, 0.0, 1.0)
                && f32_in(*y, 0.0, 1.0)
                && f32_in(*pressure, 0.0, 1.0)
                && f32_in(*direction_x, -1.0, 1.0)
                && f32_in(*direction_y, -1.0, 1.0)
        }
        // The recording barrier is revision-protected at ingress as well as at
        // dispatch, so an arm decision aimed at a replaced program never
        // occupies a queue slot it will only be refused from later.
        WebAction::SetGestureRecording {
            layer_stack_revision,
            ..
        } => *layer_stack_revision != 0,
        // The B9 transports carry the same ingress revision protection.
        WebAction::SetPerformanceRecording {
            layer_stack_revision,
            ..
        }
        | WebAction::SetPerformancePlayback {
            layer_stack_revision,
            ..
        } => *layer_stack_revision != 0,
        WebAction::ClearPerformanceTake => true,
        WebAction::SetGestureCanvas { param, value } => {
            valid_identifier(param, 64) && valid_gesture_canvas_edit(param, value)
        }
        WebAction::MorphGlide {
            target,
            duration_beats,
        } => valid_f32(*target) && valid_f64_for_f32(*duration_beats),
        WebAction::MorphCapture {
            slot,
            stack_revision,
            composition_revision,
        } => {
            matches!(slot.as_str(), "a" | "b")
                && stack_revision.is_none_or(|revision| revision != 0)
                && composition_revision.is_none_or(|revision| revision != 0)
        }
        WebAction::SetMorphLaw { law } => matches!(law.as_str(), "linear" | "equal_power"),
        WebAction::ResetGroup { group } => valid_identifier(group, 32),
        WebAction::StartProgramRecording { .. }
        | WebAction::FinishProgramRecording
        | WebAction::CancelProgramRecording
        | WebAction::SetStageHealthHud { .. } => true,
        WebAction::CaptureStill { target, .. } => valid_capture_target(target),
        WebAction::StartResample {
            target,
            destination_layer_id,
            ..
        } => valid_capture_target(target) && valid_required_stable_id(destination_layer_id),
        WebAction::SetStageTestCard {
            mode,
            output_endpoint_id,
        } => match mode {
            crate::stage_map::TestCardMode::Off => output_endpoint_id.is_none(),
            crate::stage_map::TestCardMode::SmpteBars | crate::stage_map::TestCardMode::Grid => {
                output_endpoint_id
                    .as_deref()
                    .is_some_and(valid_output_endpoint_id)
            }
        },
        WebAction::SetOutputIdentification {
            enabled,
            output_endpoint_id,
        } => {
            if *enabled {
                output_endpoint_id
                    .as_deref()
                    .is_some_and(valid_output_endpoint_id)
            } else {
                output_endpoint_id.is_none()
            }
        }
        WebAction::BeginHistoryGesture { gesture_id }
        | WebAction::EndHistoryGesture { gesture_id }
        | WebAction::CancelHistoryGesture { gesture_id } => {
            (1..=MAX_JS_SAFE_INTEGER).contains(gesture_id)
        }
        WebAction::UndoManual
        | WebAction::RedoManual
        | WebAction::RestoreRecoveryJournal
        | WebAction::DiscardRecoveryJournal => true,
        WebAction::CaptureScopedPreset {
            name,
            kind,
            target,
            preset_revision,
            layer_stack_revision,
            composition_revision,
        } => {
            valid_preset_name(name)
                && valid_preset_capture_target(*kind, target)
                && *preset_revision != 0
                && *layer_stack_revision != 0
                && valid_composition_revision(*composition_revision)
        }
        WebAction::ApplyScopedPreset {
            preset_id,
            target,
            preset_revision,
            layer_stack_revision,
            composition_revision,
        } => {
            valid_required_stable_id(preset_id)
                && valid_preset_target(target)
                && *preset_revision != 0
                && *layer_stack_revision != 0
                && valid_composition_revision(*composition_revision)
        }
        WebAction::DeleteScopedPreset {
            preset_id,
            preset_revision,
        } => valid_required_stable_id(preset_id) && *preset_revision != 0,
        WebAction::StartExport {
            width,
            height,
            fps,
            duration_secs,
            shutter_samples,
            audio_layer_id,
            ..
        } => {
            *width > 0
                && *height > 0
                && *width <= 8192
                && *height <= 8192
                && u64::from(*width)
                    .checked_mul(u64::from(*height))
                    .is_some_and(|pixels| pixels <= crate::media_safety::SAFE_MEDIA_MAX_PIXELS)
                && (1..=240).contains(fps)
                && duration_secs.is_finite()
                && *duration_secs > 0.0
                && *duration_secs <= 3600.0
                && shutter_samples.is_valid()
                && valid_optional_layer_id(audio_layer_id)
        }
        _ => true,
    }
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<WebState>>) -> impl IntoResponse {
    ws.max_message_size(MAX_WS_MESSAGE_BYTES)
        .max_frame_size(MAX_WS_MESSAGE_BYTES)
        .on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<WebState>) {
    let client_id = state.allocate_client_id();
    let (mut sender, mut receiver) = socket.split();

    // Send current state on connect
    let current = state.app.read().await;
    let init_msg = serde_json::to_string(&*current).unwrap();
    drop(current);
    let _ = sender.send(Message::Text(init_msg)).await;

    // Subscribe to broadcast updates (state JSON)
    let mut rx = state.tx.subscribe();
    let send_state = state.clone();

    // Forward broadcasts to this client
    let mut send_task = tokio::spawn(async move {
        loop {
            let msg = match rx.recv().await {
                Ok(msg) => msg,
                Err(RecvError::Lagged(_)) => {
                    // A temporarily slow socket does not need every stale
                    // 30 Hz snapshot. Jump to the live edge and send one
                    // fresh state instead of disconnecting/reconnecting.
                    let current = send_state.app.read().await;
                    let fresh = match serde_json::to_string(&*current) {
                        Ok(fresh) => fresh,
                        Err(error) => {
                            log::warn!("Failed to serialize lag recovery state: {error}");
                            continue;
                        }
                    };
                    drop(current);
                    rx = rx.resubscribe();
                    fresh
                }
                Err(RecvError::Closed) => break,
            };
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Receive actions from this client
    let state_clone = state.clone();
    // If a tab disappears mid-drag, publish an ordered End on disconnect so
    // Main records the final authored value (or an exact no-op) instead of
    // remaining permanently blocked by an orphaned gesture.
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                // Try to parse as a WebAction
                match serde_json::from_str::<WebAction>(&text) {
                    Ok(action) if valid_action(&action, 0) => match action {
                        WebAction::GyroStream { enabled } => {
                            state_clone.set_gyro_stream(client_id, enabled);
                        }
                        action @ WebAction::Gyro { .. } => {
                            let outcome = state_clone.enqueue_action(action).await;
                            if outcome != EnqueueOutcome::Dropped {
                                state_clone.note_gyro_sample(client_id);
                            }
                        }
                        action @ WebAction::BeginHistoryGesture { gesture_id } => {
                            if !state_clone
                                .begin_browser_history_gesture(client_id, gesture_id)
                                .await
                            {
                                log::warn!("Rejected nested browser history gesture");
                                continue;
                            }
                            if state_clone.enqueue_action(action).await == EnqueueOutcome::Dropped {
                                state_clone
                                    .finish_browser_history_gesture(client_id, gesture_id)
                                    .await;
                            }
                        }
                        action @ (WebAction::EndHistoryGesture { gesture_id }
                        | WebAction::CancelHistoryGesture { gesture_id }) => {
                            let cancel = matches!(&action, WebAction::CancelHistoryGesture { .. });
                            if !state_clone
                                .may_finish_browser_history_gesture(client_id, gesture_id, cancel)
                                .await
                            {
                                log::warn!(
                                    "Rejected mismatched or dirty-cancel browser history boundary"
                                );
                                continue;
                            }
                            if state_clone.enqueue_action(action).await != EnqueueOutcome::Dropped {
                                state_clone
                                    .finish_browser_history_gesture(client_id, gesture_id)
                                    .await;
                            }
                        }
                        action => {
                            if !state_clone
                                .admit_browser_action_during_gesture(client_id, &action)
                                .await
                            {
                                log::warn!(
                                    "Rejected cross-controller or cross-destination history action"
                                );
                                continue;
                            }
                            let _ = state_clone.enqueue_action(action).await;
                        }
                    },
                    Ok(_) => log::warn!("Rejected invalid WebAction payload"),
                    Err(error) => log::warn!(
                        "Failed to parse WebAction: {error} - excerpt: {}",
                        safe_log_excerpt(&text)
                    ),
                }
            }
        }
    });

    let sender_finished_first = tokio::select! {
        _ = &mut send_task => true,
        _ = &mut recv_task => false,
    };
    if sender_finished_first {
        recv_task.abort();
        let _ = recv_task.await;
    } else {
        send_task.abort();
        let _ = send_task.await;
    }
    // WebState queues End(old) while it still owns the gesture lock. Only
    // after that ordered barrier exists may another client acquire Begin(new).
    let _ = state.disconnect_browser_history_gesture(client_id).await;
    state.disconnect_gyro_client(client_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::Message as ClientMessage;

    #[tokio::test]
    async fn controller_profile_endpoint_is_bounded_pathless_and_exports_latest_publication() {
        let state = WebState::new().expect("test access token");
        let published = crate::controller_profile::ControllerProfileDocument {
            name: "Published browser profile".to_string(),
            ..Default::default()
        };
        state.publish_controller_profile_export(&published).unwrap();
        let export = crate::controller_profile::ControllerProfileAction::Export {}
            .to_json_bytes()
            .unwrap();
        let response =
            controller_profile_handler(State(state.clone()), axum::body::Bytes::from(export)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(
            response.into_body(),
            crate::controller_profile::CONTROLLER_PROFILE_MAX_BYTES,
        )
        .await
        .unwrap();
        assert_eq!(
            crate::controller_profile::ControllerProfileDocument::from_json_bytes(&body).unwrap(),
            published
        );

        let imported = crate::controller_profile::ControllerProfileDocument {
            name: "Queued browser import".to_string(),
            ..Default::default()
        };
        let request = crate::controller_profile::ControllerProfileAction::Import {
            document: imported.clone(),
        }
        .to_json_bytes()
        .unwrap();
        let response = controller_profile_handler(State(state.clone()), request.into()).await;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let mut actions = state.actions.lock().await;
        assert!(matches!(
            actions.pop(),
            Some(WebAction::ControllerProfile {
                request: crate::controller_profile::ControllerProfileAction::Import { document }
            }) if document == imported
        ));
    }

    #[tokio::test]
    async fn authenticated_websocket_round_trip_dispatches_and_returns_authoritative_state() {
        let state = WebState::new().expect("test access token");
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server_state = state.clone();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                control_router(server_state).into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });

        let mut request = format!("ws://{address}/ws?key={}", state.access_token)
            .into_client_request()
            .unwrap();
        request.headers_mut().insert(
            header::ORIGIN,
            HeaderValue::from_str(&format!("http://{address}")).unwrap(),
        );
        let (mut socket, response) = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            tokio_tungstenite::connect_async(request),
        )
        .await
        .expect("WebSocket upgrade timed out")
        .expect("authenticated same-origin WebSocket upgrade");
        assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);

        let initial = tokio::time::timeout(std::time::Duration::from_secs(2), socket.next())
            .await
            .expect("initial state timed out")
            .expect("server closed before initial state")
            .expect("initial WebSocket frame");
        let ClientMessage::Text(initial) = initial else {
            panic!("expected initial text state");
        };
        let snapshot: crate::web::state::AppSnapshot = serde_json::from_str(&initial).unwrap();
        assert_eq!(snapshot.msg_type, "state");

        for payload in [
            r#"{"action":"reset_fx"}"#,
            r#"{"action":"reset_visual_program"}"#,
            r#"{"action":"set_routing","index":0,"param":"target","value":"layer17_opacity"}"#,
            r#"{"action":"set_media_safety_mode","mode":"expert"}"#,
            r#"{"action":"set_param","param":"brightness","value":0.375}"#,
        ] {
            socket
                .send(ClientMessage::Text(payload.to_string()))
                .await
                .unwrap();
        }

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if state.actions.lock().await.len() == 5 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("actions did not cross the real WebSocket ingress");

        let mut queued = state.actions.lock().await;
        assert!(matches!(queued[0], WebAction::ResetFx));
        assert!(matches!(queued[1], WebAction::ResetVisualProgram));
        assert!(matches!(
            &queued[2],
            WebAction::SetRouting { value, .. } if value == "layer17_opacity"
        ));
        assert!(matches!(
            queued[3],
            WebAction::SetMediaSafetyMode {
                mode: crate::media_safety::MediaSafetyMode::Expert
            }
        ));
        assert!(matches!(
            &queued[4],
            WebAction::SetParam { param, value }
                if param == "brightness" && value.as_f64() == Some(0.375)
        ));
        let actions = std::mem::take(&mut *queued);
        drop(queued);

        // Close the loop exercised by a real controller: socket ingress is
        // drained by the application, dispatched through the production
        // action handler, published, broadcast, and observed back on the same
        // authenticated WebSocket as authoritative state.
        let mut app = crate::App::new(None, None, state.clone());
        for action in actions {
            app.handle_web_action(action);
        }
        app.push_web_state();

        let returned = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                match socket.next().await {
                    Some(Ok(ClientMessage::Text(message))) => break message,
                    Some(Ok(_)) => continue,
                    Some(Err(error)) => panic!("WebSocket failed before returned state: {error}"),
                    None => panic!("server closed before returned state"),
                }
            }
        })
        .await
        .expect("authoritative returned state timed out");
        let returned: crate::web::state::AppSnapshot = serde_json::from_str(&returned).unwrap();
        assert_eq!(returned.effects.brightness, 0.375);
        assert_eq!(
            returned.media_safety.mode,
            crate::media_safety::MediaSafetyMode::Expert
        );

        socket
            .send(ClientMessage::Text(
                r#"{"action":"begin_history_gesture","gesture_id":77}"#.to_string(),
            ))
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if state.actions.lock().await.len() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("history Begin did not cross WebSocket ingress");
        socket
            .send(ClientMessage::Text(
                r#"{"action":"set_param","param":"brightness","value":0.625}"#.to_string(),
            ))
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if state.actions.lock().await.len() == 2 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("history value did not cross WebSocket ingress");
        // A hostile dirty Cancel is rejected before queueing and ownership is
        // retained. Disconnect must therefore emit End, never strand Main's
        // matching transaction.
        socket
            .send(ClientMessage::Text(
                r#"{"action":"cancel_history_gesture","gesture_id":77}"#.to_string(),
            ))
            .await
            .unwrap();
        tokio::task::yield_now().await;
        assert_eq!(state.actions.lock().await.len(), 2);
        socket.close(None).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if state.actions.lock().await.len() == 3 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("disconnect did not close the browser history gesture");
        let queued = state.actions.lock().await;
        assert!(matches!(
            queued.as_slice(),
            [
                WebAction::BeginHistoryGesture { gesture_id: 77 },
                WebAction::SetParam { param, value },
                WebAction::EndHistoryGesture { gesture_id: 77 }
            ] if param == "brightness" && value.as_f64() == Some(0.625)
        ));
        drop(queued);
        server.abort();
        let _ = server.await;
    }

    #[test]
    fn library_filename_validator_accepts_only_exact_safe_media_components() {
        let valid = validate_library_filename("set.final.MP4").unwrap();
        assert_eq!(valid.name, "set.final.MP4");
        assert_eq!(valid.stem, "set.final");
        assert_eq!(valid.extension, "mp4");
        assert!(validate_library_filename("telescope-\u{00e9}t\u{00e9}.mov").is_some());
        assert!(validate_library_filename("COM10.mp4").is_some());
        assert!(validate_library_filename("LPT0.webm").is_some());
        for name in [
            "frame.png",
            "frame.JPEG",
            "frame.webp",
            "score.wav",
            "score.MP3",
            "score.flac",
            "score.ogg",
            "score.opus",
            "score.m4a",
            "score.aac",
        ] {
            assert!(validate_library_filename(name).is_some(), "rejected {name}");
        }

        for invalid in [
            "",
            ".hidden.mp4",
            ".",
            "..",
            "../clip.mp4",
            "..\\clip.mp4",
            "folder/clip.mp4",
            "folder\\clip.mp4",
            "C:\\clip.mp4",
            "clip:stream.mp4",
            "clip.mp4 ",
            "clip.mp4.",
            "clip.txt",
            "clip\n.mp4",
            "clip\u{0000}.mp4",
        ] {
            assert!(
                validate_library_filename(invalid).is_none(),
                "accepted unsafe name {invalid:?}"
            );
        }

        for invalid_character in ['<', '>', ':', '"', '/', '\\', '|', '?', '*'] {
            let name = format!("clip{invalid_character}name.mp4");
            assert!(
                validate_library_filename(&name).is_none(),
                "accepted Windows-invalid character {invalid_character:?}"
            );
        }
    }

    #[test]
    fn library_filename_validator_rejects_windows_device_names() {
        for device in [
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "CONIN$",
            "CONOUT$",
            "CLOCK$",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
            "COM\u{00b9}",
            "COM\u{00b2}",
            "COM\u{00b3}",
            "LPT\u{00b9}",
            "LPT\u{00b2}",
            "LPT\u{00b3}",
        ] {
            for name in [format!("{device}.mp4"), format!("{device}.backup.MOV")] {
                assert!(
                    validate_library_filename(&name).is_none(),
                    "accepted reserved device name {name:?}"
                );
            }
        }
        assert!(validate_library_filename("con .mp4").is_none());
        assert!(validate_library_filename("PrN.mkv").is_none());
    }

    #[test]
    fn library_filename_limit_leaves_room_for_atomic_reservation_suffixes() {
        let maximum = format!("{}.mp4", "a".repeat(MAX_LIBRARY_FILENAME_UTF16 - 4));
        assert_eq!(maximum.encode_utf16().count(), MAX_LIBRARY_FILENAME_UTF16);
        assert!(validate_library_filename(&maximum).is_some());

        let excessive = format!("a{maximum}");
        assert!(validate_library_filename(&excessive).is_none());

        let maximum_stem = maximum.strip_suffix(".mp4").unwrap();
        let longest_reservation = format!(".upload-reserve-{maximum_stem} (4294967295).mp4");
        assert!(longest_reservation.encode_utf16().count() <= 255);

        let emoji_maximum = format!("{}.mp4", "\u{1f52d}".repeat(108));
        assert_eq!(
            emoji_maximum.encode_utf16().count(),
            MAX_LIBRARY_FILENAME_UTF16
        );
        assert!(validate_library_filename(&emoji_maximum).is_some());
        assert!(validate_library_filename(&format!("\u{1f52d}{emoji_maximum}")).is_none());
    }

    #[test]
    fn audio_upload_limit_matches_the_picker_contract() {
        assert_eq!(MAX_AUDIO_UPLOAD_BYTES, 536_870_912);
        for extension in ["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac"] {
            assert!(crate::audio::is_supported_audio_extension(extension));
            assert!(!exceeds_upload_limit(extension, MAX_AUDIO_UPLOAD_BYTES));
            assert!(exceeds_upload_limit(extension, MAX_AUDIO_UPLOAD_BYTES + 1));
        }
        assert!(!exceeds_upload_limit("mp4", u64::MAX));
    }

    #[test]
    fn origin_must_match_host_exactly() {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_static("127.0.0.1:3030"));
        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("http://127.0.0.1:3030"),
        );
        assert!(same_origin(&headers));
        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("http://evil.example"),
        );
        assert!(!same_origin(&headers));
        headers.remove(header::ORIGIN);
        assert!(!same_origin(&headers));
    }

    #[test]
    fn malformed_log_excerpt_is_bounded_and_control_safe() {
        let raw = format!("bad\r\n{}", "x".repeat(1000));
        let excerpt = safe_log_excerpt(&raw);
        assert!(!excerpt.contains('\r'));
        assert!(!excerpt.contains('\n'));
        assert!(excerpt.chars().count() <= MAX_LOGGED_MESSAGE_CHARS + 1);
        assert!(excerpt.ends_with('\u{2026}'));
    }

    #[test]
    fn invalid_exports_and_recursive_wrappers_are_rejected() {
        let huge = WebAction::StartExport {
            width: 8192,
            height: 8192,
            fps: 240,
            duration_secs: 1.0,
            ntsc_quality: crate::ntsc::NtscExportQuality::LiveParity,
            shutter_samples: crate::render_export::ExportShutterSamples::Authored,
            audio_layer: None,
            audio_layer_id: None,
        };
        assert!(!valid_action(&huge, 0));
        let zero_duration = WebAction::StartExport {
            width: 1920,
            height: 1080,
            fps: 60,
            duration_secs: 0.0,
            ntsc_quality: crate::ntsc::NtscExportQuality::LiveParity,
            shutter_samples: crate::render_export::ExportShutterSamples::Authored,
            audio_layer: None,
            audio_layer_id: None,
        };
        assert!(!valid_action(&zero_duration, 0));
        let uhd_area = WebAction::StartExport {
            width: 3840,
            height: 2160,
            fps: 60,
            duration_secs: 1.0,
            ntsc_quality: crate::ntsc::NtscExportQuality::LiveParity,
            shutter_samples: crate::render_export::ExportShutterSamples::Samples16,
            audio_layer: None,
            audio_layer_id: None,
        };
        assert!(valid_action(&uhd_area, 0));
        let above_uhd_area = WebAction::StartExport {
            width: 4096,
            height: 2160,
            fps: 60,
            duration_secs: 1.0,
            ntsc_quality: crate::ntsc::NtscExportQuality::LiveParity,
            shutter_samples: crate::render_export::ExportShutterSamples::Authored,
            audio_layer: None,
            audio_layer_id: None,
        };
        assert!(!valid_action(&above_uhd_area, 0));
        let nested = WebAction::Quantized {
            inner: Box::new(WebAction::Quantized {
                inner: Box::new(WebAction::Quantized {
                    inner: Box::new(WebAction::CancelExport),
                }),
            }),
        };
        assert!(!valid_action(&nested, 0));

        let direct_media_mode = WebAction::SetMediaSafetyMode {
            mode: crate::media_safety::MediaSafetyMode::Safe,
        };
        assert!(valid_action(&direct_media_mode, 0));
        assert!(!valid_action(
            &WebAction::Quantized {
                inner: Box::new(direct_media_mode),
            },
            0,
        ));

        let direct_fit = WebAction::SetNewLayerFit {
            fit: crate::spatial::FitMode::Native,
        };
        assert!(valid_action(&direct_fit, 0));
        assert!(!valid_action(
            &WebAction::Quantized {
                inner: Box::new(direct_fit),
            },
            0,
        ));

        // Proxy settings are a host policy, not a creative control: valid
        // directly, refused inside a Quantized wrapper, and an invalid fixed
        // frame rate is refused at the gate with the engine's own predicate.
        let direct_proxy_settings = WebAction::SetProxySettings {
            scale: crate::proxy::ProxyScale::Quarter,
            frame_rate: crate::proxy::ProxyFrameRate::Fixed {
                numerator: 30,
                denominator: 1,
            },
            include_audio: false,
        };
        assert!(valid_action(&direct_proxy_settings, 0));
        assert!(!valid_action(
            &WebAction::Quantized {
                inner: Box::new(direct_proxy_settings),
            },
            0,
        ));
        assert!(!valid_action(
            &WebAction::SetProxySettings {
                scale: crate::proxy::ProxyScale::Half,
                frame_rate: crate::proxy::ProxyFrameRate::Fixed {
                    numerator: 0,
                    denominator: 1,
                },
                include_audio: true,
            },
            0,
        ));
        assert!(!valid_action(
            &WebAction::SetProxySettings {
                scale: crate::proxy::ProxyScale::Half,
                frame_rate: crate::proxy::ProxyFrameRate::Fixed {
                    numerator: 241,
                    denominator: 1,
                },
                include_audio: true,
            },
            0,
        ));
    }

    #[test]
    fn routing_target_identity_validation_rejects_malformed_or_mismatched_ids() {
        let valid = WebAction::SetRouting {
            index: 0,
            route_id: Some("7".into()),
            target_layer_id: Some("22".into()),
            layer_stack_revision: Some(9),
            param: "target".into(),
            value: serde_json::json!("layer2_opacity"),
        };
        assert!(valid_action(&valid, 0));

        for target in [
            "layer17_opacity".to_string(),
            "layer4096_opacity".to_string(),
            format!("layer{}_opacity", usize::MAX),
        ] {
            let mut dynamic = valid.clone();
            if let WebAction::SetRouting { value, .. } = &mut dynamic {
                *value = serde_json::json!(target);
            }
            assert!(valid_action(&dynamic, 0));
        }
        for target in ["layer0_opacity", "layer184467440737095516160_opacity"] {
            let mut invalid = valid.clone();
            if let WebAction::SetRouting { value, .. } = &mut invalid {
                *value = serde_json::json!(target);
            }
            assert!(!valid_action(&invalid, 0));
        }

        let malformed_route = WebAction::SetRouting {
            index: 0,
            route_id: Some("route-7".into()),
            target_layer_id: Some("22".into()),
            layer_stack_revision: Some(9),
            param: "target".into(),
            value: serde_json::json!("layer2_opacity"),
        };
        assert!(!valid_action(&malformed_route, 0));

        let malformed = WebAction::SetRouting {
            index: 0,
            route_id: Some("7".into()),
            target_layer_id: Some("layer-22".into()),
            layer_stack_revision: Some(9),
            param: "target".into(),
            value: serde_json::json!("layer2_opacity"),
        };
        assert!(!valid_action(&malformed, 0));

        let master_with_layer_identity = WebAction::SetRouting {
            index: 0,
            route_id: Some("7".into()),
            target_layer_id: Some("22".into()),
            layer_stack_revision: Some(9),
            param: "target".into(),
            value: serde_json::json!("brightness"),
        };
        assert!(!valid_action(&master_with_layer_identity, 0));

        let identity_on_depth = WebAction::SetRouting {
            index: 0,
            route_id: Some("7".into()),
            target_layer_id: Some("22".into()),
            layer_stack_revision: Some(9),
            param: "depth".into(),
            value: serde_json::json!(0.5),
        };
        assert!(!valid_action(&identity_on_depth, 0));

        let legacy_master = WebAction::SetRouting {
            index: 0,
            route_id: None,
            target_layer_id: None,
            layer_stack_revision: None,
            param: "target".into(),
            value: serde_json::json!("brightness"),
        };
        assert!(valid_action(&legacy_master, 0));

        for target in [
            "group/9/matte.amount",
            "group/9/matte.threshold",
            "group/9/matte.softness",
            "composition/bus_crossfade",
        ] {
            let mut stable = legacy_master.clone();
            if let WebAction::SetRouting { value, .. } = &mut stable {
                *value = serde_json::json!(target);
            }
            assert!(valid_action(&stable, 0), "rejected stable target {target}");
        }
        for target in [
            "group/0/matte.amount",
            "group/9/matte.channel",
            "group/9/matte.softness.extra",
            "composition/bus",
            "composition/9/bus_crossfade",
        ] {
            let mut invalid = legacy_master.clone();
            if let WebAction::SetRouting { value, .. } = &mut invalid {
                *value = serde_json::json!(target);
            }
            assert!(
                !valid_action(&invalid, 0),
                "accepted invalid target {target}"
            );
        }

        for (param, value, expected) in [
            ("depth", serde_json::json!(1.0), true),
            ("depth", serde_json::json!(1.01), false),
            ("curve", serde_json::json!("s_curve"), true),
            ("curve", serde_json::json!("unknown"), false),
            ("attack", serde_json::json!(10.0), true),
            ("release", serde_json::json!(-0.01), false),
        ] {
            let mut edit = legacy_master.clone();
            if let WebAction::SetRouting {
                param: candidate,
                value: candidate_value,
                ..
            } = &mut edit
            {
                *candidate = param.into();
                *candidate_value = value;
            }
            assert_eq!(valid_action(&edit, 0), expected, "routing edit {param}");
        }
    }

    #[test]
    fn reroll_scope_identity_revision_amount_and_loop_actions_are_strictly_validated() {
        fn reroll(
            scope: RerollScope,
            index: Option<usize>,
            layer_id: Option<&str>,
            stack_revision: Option<u64>,
            amount: f32,
        ) -> WebAction {
            WebAction::Reroll {
                scope,
                index,
                layer_id: layer_id.map(str::to_owned),
                group_id: None,
                stack_revision,
                seed: None,
                mode: crate::web::state::RerollMode::Pattern,
                amount,
                include_grain_controls: false,
                include_transform: false,
                include_rack_controls: false,
                include_group_controls: false,
            }
        }

        assert!(valid_action(
            &reroll(RerollScope::Master, None, None, None, 0.0),
            0
        ));
        assert!(valid_action(
            &reroll(RerollScope::Master, None, None, None, 2.0),
            0
        ));
        for invalid in [
            reroll(RerollScope::Master, Some(0), None, None, 0.7),
            reroll(RerollScope::Master, None, Some("22"), None, 0.7),
            reroll(RerollScope::Master, None, None, Some(9), 0.7),
        ] {
            assert!(!valid_action(&invalid, 0));
        }

        assert!(valid_action(
            &reroll(RerollScope::Layer, Some(3), Some("22"), None, 0.7),
            0
        ));
        for invalid in [
            reroll(RerollScope::Layer, None, Some("22"), None, 0.7),
            reroll(RerollScope::Layer, Some(3), None, None, 0.7),
            reroll(RerollScope::Layer, Some(3), Some("layer-22"), None, 0.7),
            reroll(RerollScope::Layer, Some(3), Some("0"), None, 0.7),
            reroll(RerollScope::Layer, Some(3), Some("22"), Some(9), 0.7),
        ] {
            assert!(!valid_action(&invalid, 0));
        }

        assert!(valid_action(
            &reroll(RerollScope::All, None, None, Some(9), 0.7),
            0
        ));
        for invalid in [
            reroll(RerollScope::All, None, None, None, 0.7),
            reroll(RerollScope::All, None, None, Some(0), 0.7),
            reroll(RerollScope::All, Some(0), None, Some(9), 0.7),
            reroll(RerollScope::All, None, Some("22"), Some(9), 0.7),
        ] {
            assert!(!valid_action(&invalid, 0));
        }

        let group = WebAction::Reroll {
            scope: RerollScope::Group,
            index: None,
            layer_id: None,
            group_id: Some("77".into()),
            stack_revision: None,
            seed: Some(4),
            mode: crate::web::state::RerollMode::Variation,
            amount: 0.7,
            include_grain_controls: false,
            include_transform: false,
            include_rack_controls: true,
            include_group_controls: true,
        };
        assert!(valid_action(&group, 0));
        for group_id in [None, Some("0"), Some("group-77")] {
            let mut invalid = group.clone();
            if let WebAction::Reroll {
                group_id: candidate,
                ..
            } = &mut invalid
            {
                *candidate = group_id.map(str::to_owned);
            }
            assert!(!valid_action(&invalid, 0));
        }

        for amount in [-0.001, 2.001, f32::NAN, f32::INFINITY] {
            assert!(!valid_action(
                &reroll(RerollScope::Master, None, None, None, amount),
                0
            ));
        }

        for layer_id in [None, Some("22")] {
            assert!(valid_action(
                &WebAction::SetLayerRerollOnLoop {
                    index: 3,
                    layer_id: layer_id.map(str::to_owned),
                    enabled: true,
                },
                0
            ));
        }
        for layer_id in ["", "0", "layer-22", "22\n"] {
            assert!(!valid_action(
                &WebAction::SetLayerRerollOnLoop {
                    index: 3,
                    layer_id: Some(layer_id.into()),
                    enabled: false,
                },
                0
            ));
        }
    }

    #[test]
    fn creative_ingress_keeps_markers_immutable_and_missing_routes_output_only() {
        for kind in ["legacy_canonical", "legacy_temporal"] {
            for (param, value) in [
                ("enabled", serde_json::json!(false)),
                ("wet", serde_json::json!(0.5)),
                ("blend", serde_json::json!("screen")),
            ] {
                assert!(!valid_node_param_value(kind, param, &value));
            }
        }
        assert!(valid_node_param_value(
            "digital_color",
            "wet",
            &serde_json::json!(0.5)
        ));
        let missing_layer = CreativeImageTapSnapshot {
            input: CreativeImageSourceSnapshot::MissingSelectedLayer {
                saved_position: crate::performance::SavedLayerPosition::new(4).unwrap(),
                stage: crate::image_routing::LayerImageStage::PostLocalEffects,
            },
            timing: crate::visual_rack::EdgeTiming::PreviousFrame,
        };
        let missing_group = CreativeImageTapSnapshot {
            input: CreativeImageSourceSnapshot::MissingGroupOutput {
                group_id: "9".into(),
            },
            timing: crate::visual_rack::EdgeTiming::PreviousFrame,
        };
        assert!(!valid_creative_route(&missing_layer));
        assert!(!valid_creative_route(&missing_group));

        let current_self = WebAction::SetVisualNodeRoute {
            scope: CreativeScopeSnapshot::Group {
                group_id: "9".into(),
            },
            node_id: "3".into(),
            route: CreativeImageTapSnapshot {
                input: CreativeImageSourceSnapshot::GroupOutput {
                    group_id: "9".into(),
                },
                timing: crate::visual_rack::EdgeTiming::CurrentFrame,
            },
            channel: "alpha".into(),
            invert: false,
            composition_revision: 7,
        };
        assert!(!valid_action(&current_self, 0));
        let mut previous_self = current_self;
        if let WebAction::SetVisualNodeRoute { route, .. } = &mut previous_self {
            route.timing = crate::visual_rack::EdgeTiming::PreviousFrame;
        }
        assert!(valid_action(&previous_self, 0));

        for variant in ["rectangle", "ellipse", "image"] {
            assert!(valid_action(
                &WebAction::SetVisualNodeMaskVariant {
                    scope: CreativeScopeSnapshot::Master,
                    node_id: "3".into(),
                    variant: variant.into(),
                    composition_revision: 7,
                },
                0
            ));
        }
        assert!(!valid_action(
            &WebAction::SetVisualNodeMaskVariant {
                scope: CreativeScopeSnapshot::Master,
                node_id: "3".into(),
                variant: "missing".into(),
                composition_revision: 7,
            },
            0
        ));
    }

    /// Closes the `NodeParamType::Enum` allowlist and the ordered route arm of
    /// `valid_action`. A missing enum arm silently drops the panel's select at
    /// ingress with the control still rendering normally; a missing route arm
    /// lets an unvalidated reroute into the bounded queue.
    #[test]
    fn symmetry_ingress_validates_every_slot_and_closes_both_discrete_vocabularies() {
        // Both discrete vocabularies are closed and complete, and neither
        // accepts a token belonging to the other.
        for mode in [
            "cyclic",
            "dihedral",
            "planar_p1",
            "planar_pm",
            "planar_p2",
            "planar_pmm",
            "log_spiral",
            "orbit",
        ] {
            assert!(
                valid_node_param_value("symmetry", "symmetry_mode", &serde_json::json!(mode)),
                "symmetry_mode must accept {mode}"
            );
        }
        for boundary in ["transparent", "mirror", "wrap", "hold", "cellular_reentry"] {
            assert!(
                valid_node_param_value(
                    "symmetry",
                    "symmetry_boundary",
                    &serde_json::json!(boundary)
                ),
                "symmetry_boundary must accept {boundary}"
            );
        }
        assert!(!valid_node_param_value(
            "symmetry",
            "symmetry_mode",
            &serde_json::json!("cellular_reentry")
        ));
        assert!(!valid_node_param_value(
            "symmetry",
            "symmetry_boundary",
            &serde_json::json!("orbit")
        ));
        // The shipped Displace boundary select reaches dispatch too.
        assert!(valid_node_param_value(
            "displace",
            "boundary",
            &serde_json::json!("mirror")
        ));
        assert!(!valid_node_param_value(
            "displace",
            "boundary",
            &serde_json::json!("cellular_reentry")
        ));

        // Continuous controls are bounded by the descriptor registry, the seed
        // is an ordinary u32 value, and every route key is refused outright.
        assert!(valid_node_param_value(
            "symmetry",
            "symmetry_base_folds",
            &serde_json::json!(32.0)
        ));
        assert!(!valid_node_param_value(
            "symmetry",
            "symmetry_base_folds",
            &serde_json::json!(33.0)
        ));
        assert!(valid_node_param_value(
            "symmetry",
            "symmetry_seed",
            &serde_json::json!(4_294_967_295_u64)
        ));
        assert!(valid_node_param_value(
            "symmetry",
            "symmetry_source_donor0",
            &serde_json::json!(true)
        ));
        for key in [
            "symmetry_donor0_tap",
            "symmetry_donor1_tap",
            "symmetry_motion0_donor",
            "symmetry_motion1_donor",
        ] {
            assert!(
                !valid_node_param_value("symmetry", key, &serde_json::json!("one_below")),
                "{key} must never be editable on the coalescible value path"
            );
        }

        let image_route =
            |index: u8, input: CreativeImageSourceSnapshot, timing| SymmetryRouteSnapshot::Image {
                index,
                route: CreativeImageTapSnapshot { input, timing },
            };
        let action = |scope: CreativeScopeSnapshot, node_id: &str, route, revision| {
            WebAction::SetVisualNodeSymmetryRoute {
                scope,
                node_id: node_id.into(),
                route,
                composition_revision: revision,
            }
        };
        let current = crate::visual_rack::EdgeTiming::CurrentFrame;
        let previous = crate::visual_rack::EdgeTiming::PreviousFrame;

        assert!(valid_action(
            &action(
                CreativeScopeSnapshot::Master,
                "3",
                image_route(1, CreativeImageSourceSnapshot::OneBelow, current),
                7
            ),
            0
        ));
        // Slot index is route identity, and there are exactly two of each.
        assert!(!valid_action(
            &action(
                CreativeScopeSnapshot::Master,
                "3",
                image_route(2, CreativeImageSourceSnapshot::OneBelow, current),
                7
            ),
            0
        ));
        assert!(!valid_action(
            &action(
                CreativeScopeSnapshot::Master,
                "3",
                SymmetryRouteSnapshot::Motion {
                    index: 2,
                    layer_id: None
                },
                7
            ),
            0
        ));
        // Zero or non-decimal identifiers and a zero revision are refused.
        assert!(!valid_action(
            &action(
                CreativeScopeSnapshot::Master,
                "0",
                image_route(0, CreativeImageSourceSnapshot::OneBelow, current),
                7
            ),
            0
        ));
        assert!(!valid_action(
            &action(
                CreativeScopeSnapshot::Master,
                "3",
                image_route(0, CreativeImageSourceSnapshot::OneBelow, current),
                0
            ),
            0
        ));
        assert!(!valid_action(
            &action(
                CreativeScopeSnapshot::Master,
                "3",
                SymmetryRouteSnapshot::Motion {
                    index: 0,
                    layer_id: Some("0".into())
                },
                7
            ),
            0
        ));
        // Clearing a motion slot is legal.
        assert!(valid_action(
            &action(
                CreativeScopeSnapshot::Master,
                "3",
                SymmetryRouteSnapshot::Motion {
                    index: 1,
                    layer_id: None
                },
                7
            ),
            0
        ));
        // Both output-only tombstones are refused on ingress.
        for tombstone in [
            CreativeImageSourceSnapshot::MissingSelectedLayer {
                saved_position: crate::performance::SavedLayerPosition::new(4).unwrap(),
                stage: crate::image_routing::LayerImageStage::PostLocalEffects,
            },
            CreativeImageSourceSnapshot::MissingGroupOutput {
                group_id: "9".into(),
            },
        ] {
            assert!(!valid_action(
                &action(
                    CreativeScopeSnapshot::Master,
                    "3",
                    image_route(0, tombstone, previous),
                    7
                ),
                0
            ));
        }
        // A group rack node cannot read its own group's output on this frame,
        // and Clean Program is previous-frame only.
        let group = CreativeScopeSnapshot::Group {
            group_id: "9".into(),
        };
        let own_output = CreativeImageSourceSnapshot::GroupOutput {
            group_id: "9".into(),
        };
        assert!(!valid_action(
            &action(
                group.clone(),
                "3",
                image_route(0, own_output.clone(), current),
                7
            ),
            0
        ));
        assert!(valid_action(
            &action(group, "3", image_route(0, own_output, previous), 7),
            0
        ));
        assert!(!valid_action(
            &action(
                CreativeScopeSnapshot::Master,
                "3",
                image_route(0, CreativeImageSourceSnapshot::CleanProgram, current),
                7
            ),
            0
        ));

        // No ordered route action may be wrapped in a quantized batch. This is
        // a class rule over every revision-protected route barrier in the
        // registry, so all three are asserted together rather than only the
        // two that existed when the rule was first written.
        for inner in [
            WebAction::SetVisualNodeSymmetryRoute {
                scope: CreativeScopeSnapshot::Master,
                node_id: "3".into(),
                route: image_route(0, CreativeImageSourceSnapshot::OneBelow, current),
                composition_revision: 7,
            },
            WebAction::SetVisualNodeDisplaceRoute {
                scope: CreativeScopeSnapshot::Master,
                node_id: "3".into(),
                route: CreativeImageTapSnapshot {
                    input: CreativeImageSourceSnapshot::OneBelow,
                    timing: current,
                },
                composition_revision: 7,
            },
            WebAction::SetVisualNodeResidualRoute {
                scope: CreativeScopeSnapshot::Master,
                node_id: "3".into(),
                slot: crate::web::state::ResidualRouteSlotSnapshot::Detail,
                route: CreativeImageTapSnapshot {
                    input: CreativeImageSourceSnapshot::OneBelow,
                    timing: current,
                },
                composition_revision: 7,
            },
        ] {
            assert!(!valid_action(
                &WebAction::Quantized {
                    inner: Box::new(inner)
                },
                0
            ));
        }
    }

    /// Every `NodeParamType::Enum` needs an explicit token list in
    /// `valid_node_param_value`; a missing arm renders the panel control
    /// normally and then drops the message at the WebSocket gate.
    #[test]
    fn residual_ingress_closes_both_discrete_vocabularies_and_barriers_each_route_slot() {
        use crate::web::state::ResidualRouteSlotSnapshot;

        for token in ["four", "eight", "sixteen", "thirty_two", "sixty_four"] {
            assert!(
                valid_node_param_value("residual", "block", &serde_json::json!(token)),
                "block token {token} must be admitted"
            );
        }
        for token in ["off", "coarse", "medium", "fine"] {
            assert!(
                valid_node_param_value("residual", "quantization", &serde_json::json!(token)),
                "quantization token {token} must be admitted"
            );
        }
        // Neighbouring, cross-vocabulary and mistyped tokens are refused.
        for (param, token) in [
            ("block", "off"),
            ("block", "one_hundred_twenty_eight"),
            ("block", "thirtytwo"),
            ("quantization", "eight"),
            ("quantization", "coarser"),
        ] {
            assert!(
                !valid_node_param_value("residual", param, &serde_json::json!(token)),
                "{param} must refuse {token}"
            );
        }
        assert!(!valid_node_param_value(
            "residual",
            "block",
            &serde_json::json!(1)
        ));
        // The shipped Displace vocabulary is closed the same way.
        for token in ["transparent", "mirror", "wrap", "hold"] {
            assert!(valid_node_param_value(
                "displace",
                "boundary",
                &serde_json::json!(token)
            ));
        }
        assert!(!valid_node_param_value(
            "displace",
            "boundary",
            &serde_json::json!("clamp")
        ));

        // Continuous values are bounded by the descriptor table itself.
        assert!(valid_node_param_value(
            "residual",
            "mix",
            &serde_json::json!(1.0)
        ));
        assert!(!valid_node_param_value(
            "residual",
            "mix",
            &serde_json::json!(1.0001)
        ));
        assert!(valid_node_param_value(
            "residual",
            "detail_gain",
            &serde_json::json!(4.0)
        ));
        assert!(!valid_node_param_value(
            "residual",
            "detail_gain",
            &serde_json::json!(-0.001)
        ));
        assert!(valid_node_param_value(
            "residual",
            "seed",
            &serde_json::json!(u32::MAX)
        ));
        assert!(!valid_node_param_value(
            "residual",
            "seed",
            &serde_json::json!(u64::from(u32::MAX) + 1)
        ));
        // Both routes belong to the ordered action and are refused on the
        // coalescible value path before the descriptor lookup ever runs.
        for param in ["structure_tap", "detail_tap"] {
            assert!(!valid_node_param_value(
                "residual",
                param,
                &serde_json::json!({"input": {"source": "one_below"}, "timing": "current_frame"})
            ));
        }

        let route =
            |input: CreativeImageSourceSnapshot, timing| CreativeImageTapSnapshot { input, timing };
        let action = |scope: CreativeScopeSnapshot,
                      node_id: &str,
                      slot: ResidualRouteSlotSnapshot,
                      route: CreativeImageTapSnapshot,
                      revision: u64| {
            WebAction::SetVisualNodeResidualRoute {
                scope,
                node_id: node_id.into(),
                slot,
                route,
                composition_revision: revision,
            }
        };
        for slot in [
            ResidualRouteSlotSnapshot::Structure,
            ResidualRouteSlotSnapshot::Detail,
        ] {
            assert!(valid_action(
                &action(
                    CreativeScopeSnapshot::Master,
                    "3",
                    slot,
                    route(
                        CreativeImageSourceSnapshot::OneBelow,
                        crate::visual_rack::EdgeTiming::CurrentFrame
                    ),
                    7
                ),
                0
            ));
            // A zero or non-decimal node ID, a zero revision, and either
            // output-only tombstone are all refused before the queue.
            for hostile in ["0", "", "7a", " 7"] {
                assert!(!valid_action(
                    &action(
                        CreativeScopeSnapshot::Master,
                        hostile,
                        slot,
                        route(
                            CreativeImageSourceSnapshot::OneBelow,
                            crate::visual_rack::EdgeTiming::CurrentFrame
                        ),
                        7
                    ),
                    0
                ));
            }
            assert!(!valid_action(
                &action(
                    CreativeScopeSnapshot::Master,
                    "3",
                    slot,
                    route(
                        CreativeImageSourceSnapshot::OneBelow,
                        crate::visual_rack::EdgeTiming::CurrentFrame
                    ),
                    0
                ),
                0
            ));
            assert!(!valid_action(
                &action(
                    CreativeScopeSnapshot::Master,
                    "3",
                    slot,
                    route(
                        CreativeImageSourceSnapshot::MissingSelectedLayer {
                            saved_position: crate::performance::SavedLayerPosition::new(4).unwrap(),
                            stage: crate::image_routing::LayerImageStage::PostLocalEffects,
                        },
                        crate::visual_rack::EdgeTiming::PreviousFrame
                    ),
                    7
                ),
                0
            ));
            assert!(!valid_action(
                &action(
                    CreativeScopeSnapshot::Master,
                    "3",
                    slot,
                    route(
                        CreativeImageSourceSnapshot::MissingGroupOutput {
                            group_id: "9".into()
                        },
                        crate::visual_rack::EdgeTiming::PreviousFrame
                    ),
                    7
                ),
                0
            ));
            // Clean Program is previous-frame only, per slot.
            assert!(!valid_action(
                &action(
                    CreativeScopeSnapshot::Master,
                    "3",
                    slot,
                    route(
                        CreativeImageSourceSnapshot::CleanProgram,
                        crate::visual_rack::EdgeTiming::CurrentFrame
                    ),
                    7
                ),
                0
            ));
            // A group's rack may not read its own output on the current frame,
            // but the identical route at N-1 is an admitted feedback edge.
            let self_group = CreativeScopeSnapshot::Group {
                group_id: "9".into(),
            };
            assert!(!valid_action(
                &action(
                    self_group.clone(),
                    "3",
                    slot,
                    route(
                        CreativeImageSourceSnapshot::GroupOutput {
                            group_id: "9".into()
                        },
                        crate::visual_rack::EdgeTiming::CurrentFrame
                    ),
                    7
                ),
                0
            ));
            assert!(valid_action(
                &action(
                    self_group,
                    "3",
                    slot,
                    route(
                        CreativeImageSourceSnapshot::GroupOutput {
                            group_id: "9".into()
                        },
                        crate::visual_rack::EdgeTiming::PreviousFrame
                    ),
                    7
                ),
                0
            ));
        }

        // The Displace route action now runs through the same prefilters
        // instead of falling through the permissive default.
        assert!(!valid_action(
            &WebAction::SetVisualNodeDisplaceRoute {
                scope: CreativeScopeSnapshot::Master,
                node_id: "0".into(),
                route: route(
                    CreativeImageSourceSnapshot::OneBelow,
                    crate::visual_rack::EdgeTiming::CurrentFrame
                ),
                composition_revision: 7,
            },
            0
        ));
        assert!(valid_action(
            &WebAction::SetVisualNodeDisplaceRoute {
                scope: CreativeScopeSnapshot::Master,
                node_id: "3".into(),
                route: route(
                    CreativeImageSourceSnapshot::OneBelow,
                    crate::visual_rack::EdgeTiming::CurrentFrame
                ),
                composition_revision: 7,
            },
            0
        ));
    }

    /// A route vocabulary the panel can send but the server-side allowlist does
    /// not name is silently dropped at ingress rather than refused, so the two
    /// sides are checked together here.
    #[test]
    fn the_gesture_canvas_route_is_accepted_at_ingress_at_both_timings() {
        for timing in [
            crate::visual_rack::EdgeTiming::CurrentFrame,
            crate::visual_rack::EdgeTiming::PreviousFrame,
        ] {
            let route = CreativeImageTapSnapshot {
                input: CreativeImageSourceSnapshot::GestureCanvas,
                timing,
            };
            assert!(
                valid_creative_route(&route),
                "the canvas is a positionless singleton and is authorable at {timing:?}"
            );
            assert!(valid_action(
                &WebAction::SetVisualNodeRoute {
                    scope: CreativeScopeSnapshot::Group {
                        group_id: "9".into(),
                    },
                    node_id: "3".into(),
                    route: route.clone(),
                    channel: "alpha".into(),
                    invert: false,
                    composition_revision: 7,
                },
                0
            ));
            assert!(valid_action(
                &WebAction::SetCompositionGroupMatteRoute {
                    group_id: "9".into(),
                    route: Some(route.clone()),
                    channel: "luma".into(),
                    invert: false,
                    composition_revision: 7,
                },
                0
            ));

            // It resolves without an ID, a saved position, or a group lookup,
            // and the resolvers below would happily answer for anything.
            let resolved = route
                .to_runtime(|_| crate::performance::SavedLayerPosition::new(4), |_| true)
                .expect("the singleton resolves to itself");
            assert_eq!(
                resolved.source,
                crate::visual_rack::ResolvedImageSource::GestureCanvas
            );
            assert_eq!(resolved.timing, timing);
            assert_eq!(CreativeImageTapSnapshot::from_runtime(resolved), route);
        }

        // The wire vocabulary stays closed: a near-miss token is refused rather
        // than defaulted onto another producer.
        assert!(serde_json::from_str::<CreativeImageSourceSnapshot>(
            r#"{"source":"gesture_field"}"#
        )
        .is_err());
        assert_eq!(
            serde_json::from_str::<CreativeImageSourceSnapshot>(r#"{"source":"gesture_canvas"}"#)
                .unwrap(),
            CreativeImageSourceSnapshot::GestureCanvas
        );

        // Both panel route editors offer the token and map it back to the same
        // wire value; a vocabulary present on only one side is the exact defect
        // this test exists to catch.
        let js = include_str!("../../static/app.js");
        assert!(js.contains("['gesture_canvas', 'Gesture canvas (etched field)']"));
        assert!(js.contains("case 'gesture_canvas': return 'gesture_canvas';"));
        assert!(js.contains("input = { source: 'gesture_canvas' };"));
    }

    #[test]
    fn the_program_tap_route_is_accepted_at_ingress_at_both_timings() {
        for timing in [
            crate::visual_rack::EdgeTiming::CurrentFrame,
            crate::visual_rack::EdgeTiming::PreviousFrame,
        ] {
            let route = CreativeImageTapSnapshot {
                input: CreativeImageSourceSnapshot::ProgramTap,
                timing,
            };
            assert!(
                valid_creative_route(&route),
                "the tap is a positionless singleton and is authorable at {timing:?}"
            );
            assert!(valid_action(
                &WebAction::SetVisualNodeRoute {
                    scope: CreativeScopeSnapshot::Group {
                        group_id: "9".into(),
                    },
                    node_id: "3".into(),
                    route: route.clone(),
                    channel: "alpha".into(),
                    invert: false,
                    composition_revision: 7,
                },
                0
            ));
            assert!(valid_action(
                &WebAction::SetCompositionGroupMatteRoute {
                    group_id: "9".into(),
                    route: Some(route.clone()),
                    channel: "luma".into(),
                    invert: false,
                    composition_revision: 7,
                },
                0
            ));

            // It resolves without an ID, a saved position, or a group lookup,
            // and the resolvers below would happily answer for anything.
            let resolved = route
                .to_runtime(|_| crate::performance::SavedLayerPosition::new(4), |_| true)
                .expect("the singleton resolves to itself");
            assert_eq!(
                resolved.source,
                crate::visual_rack::ResolvedImageSource::ProgramTap
            );
            assert_eq!(resolved.timing, timing);
            assert_eq!(CreativeImageTapSnapshot::from_runtime(resolved), route);
        }

        // The wire vocabulary stays closed: a near-miss token is refused rather
        // than defaulted onto another producer.
        assert!(
            serde_json::from_str::<CreativeImageSourceSnapshot>(r#"{"source":"program_out"}"#)
                .is_err()
        );
        assert_eq!(
            serde_json::from_str::<CreativeImageSourceSnapshot>(r#"{"source":"program_tap"}"#)
                .unwrap(),
            CreativeImageSourceSnapshot::ProgramTap
        );

        // Both panel route editors offer the token and map it back to the same
        // wire value; a vocabulary present on only one side is the exact defect
        // this test exists to catch.
        let js = include_str!("../../static/app.js");
        assert!(js.contains("['program_tap', 'Program re-entry (N\u{2212}1 audience)']"));
        assert!(js.contains("case 'program_tap': return 'program_tap';"));
        assert!(js.contains("input = { source: 'program_tap' };"));
    }

    #[test]
    fn discrete_node_enums_are_admitted_and_topology_fields_stay_barriered() {
        // The panel renders Displace's Boundary as an ordinary coalescible
        // parameter edit (static/app.js), so ingress must admit every authored
        // value. A missing arm here silently drops the action before
        // `set_runtime_node_param` ever sees it, leaving the authored law
        // unreachable from a browser.
        for value in ["transparent", "mirror", "wrap", "hold"] {
            assert!(
                valid_node_param_value("displace", "boundary", &serde_json::json!(value)),
                "displace boundary {value} must be admitted at ingress"
            );
        }
        for value in [
            serde_json::json!("clamp"),
            serde_json::json!("Transparent"),
            serde_json::json!(""),
            serde_json::json!(0),
            serde_json::json!(true),
            serde_json::json!(["mirror"]),
        ] {
            assert!(
                !valid_node_param_value("displace", "boundary", &value),
                "{value} is outside the closed boundary vocabulary"
            );
        }
        // The vocabulary stays closed per kind: no other kind declares it.
        assert!(!valid_node_param_value(
            "grain",
            "boundary",
            &serde_json::json!("mirror")
        ));
        // The donor route rewrites the image dependency graph, so it remains a
        // revision-protected barrier action and is refused on this path.
        assert!(!valid_node_param_value(
            "displace",
            "donor_tap",
            &serde_json::json!("one_below")
        ));
        // Both gains stay ordinary bounded floats.
        for param in ["amount_x", "amount_y"] {
            assert!(valid_node_param_value(
                "displace",
                param,
                &serde_json::json!(-1.0)
            ));
            assert!(valid_node_param_value(
                "displace",
                param,
                &serde_json::json!(1.0)
            ));
            assert!(!valid_node_param_value(
                "displace",
                param,
                &serde_json::json!(1.5)
            ));
        }
    }

    #[test]
    fn creative_matte_and_bus_actions_have_closed_validation() {
        let route = CreativeImageTapSnapshot {
            input: CreativeImageSourceSnapshot::OneBelow,
            timing: crate::visual_rack::EdgeTiming::CurrentFrame,
        };
        assert!(valid_action(
            &WebAction::SetCompositionGroupMatteRoute {
                group_id: "5".into(),
                route: Some(route),
                channel: "luma".into(),
                invert: true,
                composition_revision: 2,
            },
            0
        ));
        for (param, value, expected) in [
            ("amount", serde_json::json!(0.75), true),
            ("threshold", serde_json::json!(1.0), true),
            ("softness", serde_json::json!(0.5), true),
            ("softness", serde_json::json!(0.51), false),
            ("enabled", serde_json::json!(true), false),
        ] {
            assert_eq!(
                valid_action(
                    &WebAction::SetCompositionGroupMatteParam {
                        group_id: "5".into(),
                        param: param.into(),
                        value,
                        composition_revision: 2,
                    },
                    0
                ),
                expected
            );
        }
        for value in [0.0, 0.5, 1.0] {
            assert!(valid_action(
                &WebAction::SetCompositionBusCrossfade { value },
                0
            ));
        }
        assert!(!valid_action(
            &WebAction::SetCompositionBusCrossfade { value: 1.001 },
            0
        ));
        assert!(valid_action(
            &WebAction::SetCompositionLayerBus {
                layer_id: "44".into(),
                bus: "b".into(),
                composition_revision: 8,
            },
            0
        ));

        // Syntax ingress does not impose the advanced planner's group-member
        // cap on the dynamic top-level root. Main validates against the actual
        // staged root length without allocating from this index.
        assert!(valid_action(
            &WebAction::CreateCompositionGroup {
                name: "late group".into(),
                member_layer_ids: Vec::new(),
                root_index: 273,
                composition_revision: 8,
            },
            0
        ));
        assert!(valid_action(
            &WebAction::MoveCompositionRootItem {
                item: CompositionRootSnapshot::Layer {
                    layer_id: "274".into(),
                    bus: "program".into(),
                },
                to: 273,
                composition_revision: 8,
            },
            0
        ));
    }

    #[test]
    fn oversized_or_nonfinite_action_numbers_are_rejected() {
        let overflow: WebAction =
            serde_json::from_str(r#"{"action":"set_param","param":"brightness","value":1e308}"#)
                .unwrap();
        assert!(!valid_action(&overflow, 0));

        let nested_overflow: WebAction = serde_json::from_str(
            r#"{"action":"set_audio","param":"band_edges","value":{"edges":[100,1e308]}}"#,
        )
        .unwrap();
        assert!(!valid_action(&nested_overflow, 0));

        assert!(!valid_action(
            &WebAction::SetBpm {
                value: f32::INFINITY,
            },
            0,
        ));
        assert!(!valid_action(
            &WebAction::MorphGlide {
                target: 0.5,
                duration_beats: f64::MAX,
            },
            0,
        ));
    }

    #[test]
    fn temporal_originals_ingress_has_a_closed_bounded_vocabulary() {
        for (param, value) in [
            ("feedback", serde_json::json!(0.95)),
            ("fb_offset_x", serde_json::json!(-0.5)),
            ("fb_offset_y", serde_json::json!(0.5)),
            ("fb_reflect_x", serde_json::json!(true)),
            ("fb_hue_rotate", serde_json::json!(-180.0)),
            ("fb_saturation", serde_json::json!(2.0)),
            ("fb_gain_g", serde_json::json!(0.0)),
            ("fb_chroma_displace", serde_json::json!(0.05)),
            ("fb_blur", serde_json::json!(1.0)),
            ("fb_sharpen", serde_json::json!(2.0)),
            ("fb_shape", serde_json::json!("fold")),
            ("fb_drive", serde_json::json!(4.0)),
            ("fb_pivot", serde_json::json!(0.5)),
            ("fb_threshold", serde_json::json!(1.0)),
            ("fb_noise", serde_json::json!(1.0)),
            ("fb_edge", serde_json::json!("mirror")),
            ("fb_servo", serde_json::json!(true)),
            ("fb_servo_defeated", serde_json::json!(false)),
            ("fb_zoom", serde_json::json!(1.1)),
            ("fb_rotate", serde_json::json!(-5.0)),
            ("slitscan", serde_json::json!(1.0)),
            ("slit_angle", serde_json::json!(-180.0)),
            ("slit_axis", serde_json::json!(1.0)),
            ("slit_map", serde_json::json!("ramp")),
            ("slit_map", serde_json::json!("brightness")),
            ("slit_map", serde_json::json!("radial")),
            ("slit_map", serde_json::json!("tbc_ramp")),
            ("slit_map", serde_json::json!("sweep")),
            ("slit_interp", serde_json::json!(true)),
            ("slit_interp", serde_json::json!(false)),
            ("key_mode", serde_json::json!(4)),
            ("key_threshold", serde_json::json!(1.0)),
            ("key_softness", serde_json::json!(0.5)),
            ("key_history", serde_json::json!(23)),
            ("loom_amount", serde_json::json!(1.0)),
            ("loom_topology", serde_json::json!("kaleidoscopic")),
            ("loom_interpolation", serde_json::json!("linear")),
            ("loom_depth", serde_json::json!(1.0)),
            ("loom_phase", serde_json::json!(-1000.0)),
            ("loom_scale", serde_json::json!(100.0)),
            ("loom_angle", serde_json::json!(180.0)),
            ("loom_folds", serde_json::json!(16)),
            ("loom_quantization", serde_json::json!(24)),
            ("atlas_amount", serde_json::json!(1.0)),
            ("atlas_seed", serde_json::json!(u32::MAX)),
            ("atlas_territories", serde_json::json!(64)),
            ("atlas_collision", serde_json::json!(1.0)),
            ("garden_amount", serde_json::json!(1.0)),
            ("garden_gate", serde_json::json!("audio_onset")),
            ("garden_gate", serde_json::json!("motion")),
            ("garden_threshold", serde_json::json!(1.0)),
            ("garden_softness", serde_json::json!(0.5)),
            ("garden_decay", serde_json::json!(1.0)),
            ("garden_max_hold_ticks", serde_json::json!(u32::MAX)),
            ("score_enabled", serde_json::json!(true)),
            ("score_seed", serde_json::json!(u32::MAX)),
            ("score_state_count", serde_json::json!(16)),
            ("score_trigger", serde_json::json!("manual")),
            ("score_loop_driver", serde_json::json!("91")),
            ("score_loop_driver", serde_json::json!("none")),
            ("reset_loop_boundary", serde_json::json!("memory")),
            ("reset_downbeat", serde_json::json!("all")),
            ("mosh_amount", serde_json::json!(1.0)),
            ("mosh_key_removal", serde_json::json!(0.0)),
            ("mosh_hold", serde_json::json!(0.5)),
            ("mosh_drop", serde_json::json!(1.0)),
            ("mosh_shuffle", serde_json::json!(0.25)),
            ("mosh_rate", serde_json::json!(0.5)),
            ("mosh_bitrate_starve", serde_json::json!(1.0)),
            ("mosh_resync", serde_json::json!(0.3)),
            ("mosh_recycle", serde_json::json!(true)),
            ("mosh_recycle", serde_json::json!(false)),
        ] {
            let action = WebAction::SetTemporal {
                param: param.into(),
                value,
            };
            assert!(valid_action(&action, 0), "rejected valid temporal {param}");
        }

        for (param, value) in [
            ("feedback", serde_json::json!(0.951)),
            ("fb_zoom", serde_json::json!(0.899)),
            ("key_mode", serde_json::json!(4.5)),
            ("key_history", serde_json::json!(0)),
            ("loom_topology", serde_json::json!("hexagonal")),
            ("loom_interpolation", serde_json::json!("cubic")),
            ("loom_phase", serde_json::json!(1000.1)),
            ("loom_scale", serde_json::json!(0.0)),
            ("loom_folds", serde_json::json!(17)),
            ("loom_quantization", serde_json::json!(25)),
            ("atlas_seed", serde_json::json!(u64::from(u32::MAX) + 1)),
            ("atlas_territories", serde_json::json!(0)),
            ("garden_gate", serde_json::json!("flow")),
            ("garden_softness", serde_json::json!(0.501)),
            ("slit_map", serde_json::json!("melt")),
            ("slit_map", serde_json::json!(2)),
            ("slit_interp", serde_json::json!(1)),
            ("score_enabled", serde_json::json!(1)),
            ("score_state_count", serde_json::json!(1)),
            ("score_trigger", serde_json::json!("loop")),
            ("score_loop_driver", serde_json::json!("0")),
            ("score_loop_driver", serde_json::json!("missing:3")),
            ("reset_downbeat", serde_json::json!("carrier")),
            ("mosh_amount", serde_json::json!(1.001)),
            ("mosh_key_removal", serde_json::json!(-0.1)),
            ("mosh_recycle", serde_json::json!(1)),
            ("unknown_original", serde_json::json!(0.5)),
        ] {
            let action = WebAction::SetTemporal {
                param: param.into(),
                value,
            };
            assert!(
                !valid_action(&action, 0),
                "accepted invalid temporal {param}"
            );
        }

        assert!(valid_action(&WebAction::ClearTemporalMemory, 0));
        assert!(valid_action(&WebAction::TriggerCollisionScore, 0));
        assert!(valid_action(&WebAction::TriggerRefreshGarden, 0));
        assert!(valid_action(&WebAction::ClearTemporalEventTrack, 0));
        assert!(!valid_action(
            &WebAction::Quantized {
                inner: Box::new(WebAction::ClearTemporalMemory)
            },
            0
        ));
        assert!(!valid_action(
            &WebAction::Quantized {
                inner: Box::new(WebAction::TriggerCollisionScore)
            },
            0
        ));
        assert!(!valid_action(
            &WebAction::Quantized {
                inner: Box::new(WebAction::TriggerRefreshGarden)
            },
            0
        ));
        assert!(!valid_action(
            &WebAction::Quantized {
                inner: Box::new(WebAction::ClearTemporalEventTrack)
            },
            0
        ));
    }

    #[test]
    fn motion_ingress_is_scope_aware_closed_bounded_and_revision_protected() {
        let master = MotionScopeSnapshot::Master;
        let layer = MotionScopeSnapshot::Layer {
            layer_id: "91".into(),
        };
        for (scope, param, value) in [
            (&master, "field_source", serde_json::json!("codec_vectors")),
            (&master, "lattice_quality", serde_json::json!("high")),
            (&master, "shutter_angle", serde_json::json!(360.0)),
            (&master, "shutter_phase", serde_json::json!(-1.0)),
            (&master, "shutter_curvature", serde_json::json!(2.0)),
            (&master, "shutter_chromatic_lag", serde_json::json!(1.0)),
            (&master, "shutter_quality", serde_json::json!("live")),
            (&layer, "transplant_amount", serde_json::json!(1.0)),
            (&layer, "confidence_threshold", serde_json::json!(1.0)),
            (&layer, "confidence_softness", serde_json::json!(0.5)),
            (&layer, "refresh", serde_json::json!(0.0)),
            (&layer, "decay", serde_json::json!(1.0)),
            (&layer, "occlusion", serde_json::json!(1.0)),
            (&layer, "carrier", serde_json::json!("first_source_frame")),
            (
                &master,
                "field_source",
                serde_json::json!("procedural_curl"),
            ),
            (&master, "field_scale", serde_json::json!(1.0)),
            (&master, "field_rate", serde_json::json!(-2.0)),
            (
                &layer,
                "field_source",
                serde_json::json!("procedural_chroma"),
            ),
            (&layer, "field_scale", serde_json::json!(0.0)),
            (&layer, "field_rate", serde_json::json!(2.0)),
        ] {
            assert!(
                valid_action(
                    &WebAction::SetMotion {
                        scope: scope.clone(),
                        param: param.into(),
                        value,
                    },
                    0
                ),
                "rejected valid Motion edit {param}"
            );
        }
        for (scope, param, value) in [
            (&master, "transplant_amount", serde_json::json!(0.5)),
            (&master, "carrier", serde_json::json!("black")),
            (&master, "algorithm_version", serde_json::json!(1)),
            (&layer, "donor", serde_json::json!("77")),
            (&layer, "shutter_angle", serde_json::json!(360.1)),
            (&layer, "shutter_phase", serde_json::json!(-1.1)),
            (&layer, "confidence_softness", serde_json::json!(0.501)),
            (&layer, "field_source", serde_json::json!("fallback")),
            (&layer, "lattice_quality", serde_json::json!("adaptive")),
            (&layer, "carrier", serde_json::json!("history")),
            (&layer, "shutter_quality", serde_json::json!("auto")),
            (&master, "field_source", serde_json::json!("procedural")),
            (&master, "field_scale", serde_json::json!(1.01)),
            (&layer, "field_rate", serde_json::json!(-2.1)),
            (&layer, "vector_trash", serde_json::json!(1.5)),
            (&master, "trash_block_size", serde_json::json!(1.0)),
        ] {
            assert!(
                !valid_action(
                    &WebAction::SetMotion {
                        scope: scope.clone(),
                        param: param.into(),
                        value,
                    },
                    0
                ),
                "accepted invalid Motion edit {param}"
            );
        }
        assert!(!valid_action(
            &WebAction::SetMotion {
                scope: MotionScopeSnapshot::Layer {
                    layer_id: "0".into()
                },
                param: "shutter_angle".into(),
                value: serde_json::json!(20.0),
            },
            0
        ));

        let donor = WebAction::SetMotionDonor {
            layer_id: "91".into(),
            donor_layer_id: Some("77".into()),
            layer_stack_revision: 4,
        };
        assert!(valid_action(&donor, 0));
        for invalid in [
            WebAction::SetMotionDonor {
                layer_id: "91".into(),
                donor_layer_id: Some("91".into()),
                layer_stack_revision: 4,
            },
            WebAction::SetMotionDonor {
                layer_id: "91".into(),
                donor_layer_id: Some("0".into()),
                layer_stack_revision: 4,
            },
            WebAction::SetMotionDonor {
                layer_id: "91".into(),
                donor_layer_id: None,
                layer_stack_revision: 0,
            },
        ] {
            assert!(!valid_action(&invalid, 0));
        }
        assert!(valid_action(&WebAction::ClearMotionMemory, 0));
        for barrier in [donor, WebAction::ClearMotionMemory] {
            assert!(!valid_action(
                &WebAction::Quantized {
                    inner: Box::new(barrier)
                },
                0
            ));
        }
    }

    #[test]
    fn refresh_garden_routes_require_stable_ids_revisions_and_known_stages() {
        let matte: WebAction = serde_json::from_str(
            r#"{"action":"set_refresh_garden_matte_route","layer_id":"91","stage":"pre_local_effects","layer_stack_revision":4}"#,
        )
        .unwrap();
        let motion: WebAction = serde_json::from_str(
            r#"{"action":"set_refresh_garden_motion_route","layer_id":"77","layer_stack_revision":4}"#,
        )
        .unwrap();
        let clear: WebAction = serde_json::from_str(
            r#"{"action":"set_refresh_garden_motion_route","layer_id":null,"layer_stack_revision":4}"#,
        )
        .unwrap();
        assert!(valid_action(&matte, 0));
        assert!(valid_action(&motion, 0));
        assert!(valid_action(&clear, 0));

        for invalid in [
            WebAction::SetRefreshGardenMatteRoute {
                layer_id: Some("0".into()),
                stage: crate::image_routing::LayerImageStage::PostLocalEffects,
                layer_stack_revision: 4,
            },
            WebAction::SetRefreshGardenMotionRoute {
                layer_id: Some("missing:2".into()),
                layer_stack_revision: 4,
            },
            WebAction::SetRefreshGardenMotionRoute {
                layer_id: None,
                layer_stack_revision: 0,
            },
        ] {
            assert!(!valid_action(&invalid, 0));
        }
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_refresh_garden_matte_route","layer_id":"91","stage":"raw","layer_stack_revision":4}"#
        )
        .is_err());
        for barrier in [matte, motion, clear] {
            assert!(!valid_action(
                &WebAction::Quantized {
                    inner: Box::new(barrier),
                },
                0
            ));
        }
    }

    #[test]
    fn transform_actions_require_strict_identity_fields_and_bounded_complete_state() {
        let layer = |layer_id: Option<&str>, param: &str, value: serde_json::Value| {
            WebAction::SetLayerTransform {
                index: 3,
                layer_id: layer_id.map(str::to_owned),
                param: param.into(),
                value,
            }
        };
        assert!(valid_action(
            &layer(Some("22"), "position_x", serde_json::json!(0.25)),
            0
        ));
        assert!(valid_action(
            &layer(Some("22"), "edge", serde_json::json!("mirror")),
            0
        ));
        for invalid in [
            layer(None, "position_x", serde_json::json!(0.25)),
            layer(Some("0"), "position_x", serde_json::json!(0.25)),
            layer(Some("layer-22"), "position_x", serde_json::json!(0.25)),
            layer(Some("22"), "position_x", serde_json::json!(4.01)),
            layer(Some("22"), "skew_deg", serde_json::json!(89.1)),
            layer(Some("22"), "edge", serde_json::json!("smear")),
            layer(Some("22"), "unknown", serde_json::json!(0.0)),
        ] {
            assert!(
                !valid_action(&invalid, 0),
                "unexpectedly valid: {invalid:?}"
            );
        }

        assert!(!valid_action(
            &WebAction::ResetLayerTransform {
                index: 0,
                layer_id: None,
            },
            0,
        ));
        assert!(valid_action(
            &WebAction::ResetLayerTransform {
                index: 99,
                layer_id: Some("22".into()),
            },
            0,
        ));

        let exact = crate::spatial::SpatialTransform::default();
        assert!(valid_action(
            &WebAction::ApplyMasterTransform { transform: exact },
            0,
        ));
        assert!(valid_action(
            &WebAction::ApplyLayerTransform {
                index: 1,
                layer_id: Some("22".into()),
                transform: exact,
            },
            0,
        ));
        let hostile = crate::spatial::SpatialTransform {
            scale: [17.0, 1.0],
            ..Default::default()
        };
        assert!(!valid_action(
            &WebAction::ApplyMasterTransform { transform: hostile },
            0,
        ));

        assert!(valid_action(
            &WebAction::SetMasterTransform {
                param: "fit".into(),
                value: serde_json::json!("fill"),
            },
            0,
        ));
        assert!(!valid_action(
            &WebAction::SetMasterTransform {
                param: "fit".into(),
                value: serde_json::json!(3),
            },
            0,
        ));
    }

    #[test]
    fn morph_enums_and_capture_revision_are_strictly_validated() {
        assert!(valid_action(
            &WebAction::MorphCapture {
                slot: "a".into(),
                stack_revision: Some(9),
                composition_revision: Some(12),
            },
            0,
        ));
        assert!(!valid_action(
            &WebAction::MorphCapture {
                slot: "typo".into(),
                stack_revision: Some(9),
                composition_revision: Some(12),
            },
            0,
        ));
        assert!(!valid_action(
            &WebAction::MorphCapture {
                slot: "b".into(),
                stack_revision: Some(0),
                composition_revision: Some(12),
            },
            0,
        ));
        assert!(!valid_action(
            &WebAction::MorphCapture {
                slot: "b".into(),
                stack_revision: Some(9),
                composition_revision: Some(0),
            },
            0,
        ));
        assert!(valid_action(
            &WebAction::SetMorphLaw {
                law: "equal_power".into(),
            },
            0,
        ));
        assert!(!valid_action(
            &WebAction::SetMorphLaw {
                law: "equal".into()
            },
            0,
        ));
    }

    #[test]
    fn curated_blend_ingress_accepts_every_exact_key_and_rejects_aliases() {
        for blend_mode in crate::layers::BlendMode::ALL {
            let json = serde_json::json!({
                "action": "set_layer_param",
                "index": 0,
                "layer_id": "17",
                "param": "blend_mode",
                "value": blend_mode.key(),
            });
            let action: WebAction = serde_json::from_value(json).unwrap();
            assert!(
                valid_action(&action, 0),
                "server rejected exact blend key {}",
                blend_mode.key()
            );
        }

        for value in [
            serde_json::json!("soft-light"),
            serde_json::json!("Alpha Cut"),
            serde_json::json!("future_blend"),
            serde_json::json!(14),
            serde_json::Value::Null,
        ] {
            let action = WebAction::SetLayerParam {
                index: 0,
                layer_id: Some("17".into()),
                param: "blend_mode".into(),
                value,
            };
            assert!(!valid_action(&action, 0));
        }
    }

    #[test]
    fn prepared_transport_ingress_is_stable_typed_and_finitely_bounded() {
        let parse = |json: &str| serde_json::from_str::<WebAction>(json).unwrap();
        for valid in [
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"direction","value":"reverse"}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"end_behavior","value":"ping_pong"}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"in_point","value":0.25}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"sample_fps","value":null}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"clip_bpm","value":128.5}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"beats_per_bar","value":7}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"beat_loop_length","value":0.015625}"#,
            r#"{"action":"set_clip_cue","layer_id":"17","slot_id":2,"cue_id":4095,"at":1.0}"#,
            r#"{"action":"trigger_scene","scene_id":1,"trigger_mode":"next_bar"}"#,
        ] {
            assert!(valid_action(&parse(valid), 0), "rejected {valid}");
        }

        for invalid in [
            r#"{"action":"set_clip_transport","layer_id":"0","slot_id":2,"param":"rate","value":1}"#,
            r#"{"action":"set_clip_transport","layer_id":"17x","slot_id":2,"param":"rate","value":1}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"in_point","value":1.01}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"sample_fps","value":0.249}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"clip_bpm","value":1000}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"beats_per_bar","value":4.5}"#,
            r#"{"action":"set_clip_transport","layer_id":"17","slot_id":2,"param":"whole_config","value":{}}"#,
        ] {
            assert!(!valid_action(&parse(invalid), 0), "accepted {invalid}");
        }
    }

    #[test]
    fn scene_authoring_ingress_has_bounded_clean_names_and_typed_ids() {
        let parse = |json: &str| serde_json::from_str::<WebAction>(json).unwrap();
        for valid in [
            r#"{"action":"capture_scene","name":"","trigger_mode":"immediate"}"#,
            r#"{"action":"capture_scene","name":"Act II — mirror study","trigger_mode":"next_bar"}"#,
            r#"{"action":"capture_scene","scene_id":65535,"name":"Recapture","trigger_mode":"next_beat"}"#,
            r#"{"action":"remove_scene","scene_id":65535}"#,
        ] {
            assert!(valid_action(&parse(valid), 0), "rejected {valid}");
        }

        let maximum = "a".repeat(MAX_SCENE_NAME_BYTES);
        assert!(valid_action(
            &WebAction::CaptureScene {
                scene_id: None,
                name: maximum,
                trigger_mode: crate::transport::TriggerMode::Immediate,
            },
            0,
        ));
        for name in [
            "a".repeat(MAX_SCENE_NAME_BYTES + 1),
            "界".repeat((MAX_SCENE_NAME_BYTES / 3) + 1),
            " leading".into(),
            "trailing ".into(),
            "line\nbreak".into(),
        ] {
            assert!(!valid_action(
                &WebAction::CaptureScene {
                    scene_id: None,
                    name,
                    trigger_mode: crate::transport::TriggerMode::Immediate,
                },
                0,
            ));
        }

        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"capture_scene","scene_id":0,"name":"bad"}"#
        )
        .is_err());
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"capture_scene","scene_id":65536,"name":"bad"}"#
        )
        .is_err());
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"capture_scene","name":"bad","trigger_mode":"later"}"#
        )
        .is_err());
        assert!(
            serde_json::from_str::<WebAction>(r#"{"action":"remove_scene","scene_id":0}"#).is_err()
        );
    }

    #[test]
    fn prepared_source_and_matte_ingress_never_falls_back_to_positions() {
        let parse = |json: &str| serde_json::from_str::<WebAction>(json).unwrap();
        for valid in [
            r#"{"action":"load_clip_into_slot","layer_id":"12","filename":"study.mov","activate":true,"trigger_mode":"next_beat"}"#,
            r#"{"action":"activate_clip_slot","layer_id":"12","slot_id":4}"#,
            r#"{"action":"set_layer_matte_param","layer_id":"12","param":"amount","value":0.5}"#,
            r#"{"action":"set_layer_matte_param","layer_id":"12","param":"channel","value":"luma"}"#,
            r#"{"action":"set_layer_matte_input","layer_id":"12","input":{"source":"selected_layer","layer_id":"88","stage":"pre_local_effects"}}"#,
            r#"{"action":"set_layer_matte_input","layer_id":"12","input":{"source":"all_below"}}"#,
            r#"{"action":"set_layer_matte_input","layer_id":"12","input":{"source":"group_output","group_id":"1"}}"#,
            r#"{"action":"set_layer_matte_input","layer_id":"12","input":{"source":"all_below"},"composition_revision":9}"#,
        ] {
            assert!(valid_action(&parse(valid), 0), "rejected {valid}");
        }

        for invalid in [
            r#"{"action":"activate_clip_slot","layer_id":"","slot_id":4}"#,
            r#"{"action":"activate_clip_slot","layer_id":"legacy-index:2","slot_id":4}"#,
            r#"{"action":"set_layer_matte_param","layer_id":"12","param":"softness","value":1.01}"#,
            r#"{"action":"set_layer_matte_input","layer_id":"12","input":{"source":"selected_layer","layer_id":"0"}}"#,
            r#"{"action":"set_layer_matte_input","layer_id":"12","input":{"source":"missing_selected_layer","saved_position":2}}"#,
            r#"{"action":"set_layer_matte_input","layer_id":"12","input":{"source":"group_output","group_id":"0"}}"#,
            r#"{"action":"set_layer_matte_input","layer_id":"12","input":{"source":"missing_group_output","group_id":"1"}}"#,
            r#"{"action":"set_layer_matte_input","layer_id":"12","input":{"source":"all_below"},"composition_revision":0}"#,
            r#"{"action":"set_layer_matte_param","layer_id":"12","param":"enabled","value":true,"composition_revision":0}"#,
        ] {
            assert!(!valid_action(&parse(invalid), 0), "accepted {invalid}");
        }
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_layer_matte_input","layer_id":"12","input":{"source":"group_output","group_id":1}}"#
        )
        .is_err());
    }

    #[test]
    fn recorder_and_stage_tool_ingress_has_no_paths_or_positional_fallbacks() {
        let parse = |json: &str| serde_json::from_str::<WebAction>(json).unwrap();
        for valid in [
            r#"{"action":"start_program_recording","auto_import":true}"#,
            r#"{"action":"finish_program_recording"}"#,
            r#"{"action":"cancel_program_recording"}"#,
            r#"{"action":"capture_still","target":"program","auto_import":false}"#,
            r#"{"action":"capture_still","target":"layer","layer_id":"42"}"#,
            r#"{"action":"start_resample","target":"group","group_id":"7","destination_layer_id":"42","activate":true}"#,
            r#"{"action":"set_stage_health_hud","enabled":true}"#,
            r#"{"action":"set_stage_test_card","mode":"smpte_bars","output_endpoint_id":"display-1"}"#,
            r#"{"action":"set_stage_test_card","mode":"off"}"#,
            r#"{"action":"set_output_identification","enabled":true,"output_endpoint_id":"projector_A"}"#,
            r#"{"action":"set_output_identification","enabled":false}"#,
        ] {
            assert!(valid_action(&parse(valid), 0), "rejected {valid}");
        }

        for invalid in [
            r#"{"action":"capture_still","target":"layer","layer_id":"0"}"#,
            r#"{"action":"capture_still","target":"layer","layer_id":"legacy-index:2"}"#,
            r#"{"action":"start_resample","target":"group","group_id":"7","destination_layer_id":"0"}"#,
            r#"{"action":"set_stage_test_card","mode":"grid"}"#,
            r#"{"action":"set_stage_test_card","mode":"off","output_endpoint_id":"display-1"}"#,
            r#"{"action":"set_stage_test_card","mode":"grid","output_endpoint_id":"../display"}"#,
            r#"{"action":"set_output_identification","enabled":true}"#,
            r#"{"action":"set_output_identification","enabled":false,"output_endpoint_id":"display-1"}"#,
        ] {
            assert!(!valid_action(&parse(invalid), 0), "accepted {invalid}");
        }

        // The tagged action shape has no field through which a browser can
        // nominate an arbitrary host path.
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"start_program_recording","output_path":"C:\\\\escape.mp4"}"#
        )
        .is_ok());
        let value = serde_json::to_value(parse(
            r#"{"action":"start_program_recording","output_path":"ignored"}"#,
        ))
        .unwrap();
        assert!(value.get("output_path").is_none());
    }

    #[test]
    fn manual_history_preset_and_recovery_ingress_is_strict_and_unquantized() {
        let layer = PresetTargetSnapshot::Layer {
            layer_id: "41".into(),
        };
        let group = PresetTargetSnapshot::Group {
            group_id: "7".into(),
        };
        for action in [
            WebAction::BeginHistoryGesture { gesture_id: 1 },
            WebAction::EndHistoryGesture {
                gesture_id: MAX_JS_SAFE_INTEGER,
            },
            WebAction::CancelHistoryGesture { gesture_id: 9 },
            WebAction::UndoManual,
            WebAction::RedoManual,
            WebAction::CaptureScopedPreset {
                name: "Prismatic drift".into(),
                kind: crate::preset::PresetKind::Transform,
                target: PresetTargetSnapshot::Master,
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::CaptureScopedPreset {
                name: "Silhouette".into(),
                kind: crate::preset::PresetKind::Matte,
                target: layer.clone(),
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::CaptureScopedPreset {
                name: "Bus study".into(),
                kind: crate::preset::PresetKind::Group,
                target: group.clone(),
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::CaptureScopedPreset {
                name: "Tour controller".into(),
                kind: crate::preset::PresetKind::ControllerProfile,
                target: PresetTargetSnapshot::ControllerProfile,
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::CaptureScopedPreset {
                name: "Gallery map".into(),
                kind: crate::preset::PresetKind::StageMap,
                target: PresetTargetSnapshot::StageMap,
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::ApplyScopedPreset {
                preset_id: "18446744073709551615".into(),
                target: layer.clone(),
                preset_revision: 4,
                layer_stack_revision: 5,
                composition_revision: 6,
            },
            WebAction::DeleteScopedPreset {
                preset_id: "9".into(),
                preset_revision: 4,
            },
            WebAction::RestoreRecoveryJournal,
            WebAction::DiscardRecoveryJournal,
        ] {
            assert!(valid_action(&action, 0), "rejected {action:?}");
            assert!(!valid_action(
                &WebAction::Quantized {
                    inner: Box::new(action)
                },
                0
            ));
        }

        for action in [
            WebAction::BeginHistoryGesture { gesture_id: 0 },
            WebAction::EndHistoryGesture {
                gesture_id: MAX_JS_SAFE_INTEGER + 1,
            },
            WebAction::CaptureScopedPreset {
                name: " leading".into(),
                kind: crate::preset::PresetKind::Rack,
                target: layer.clone(),
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::CaptureScopedPreset {
                name: "No master matte".into(),
                kind: crate::preset::PresetKind::Matte,
                target: PresetTargetSnapshot::Master,
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::CaptureScopedPreset {
                name: "No layer group".into(),
                kind: crate::preset::PresetKind::Group,
                target: layer.clone(),
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::CaptureScopedPreset {
                name: "Wrong document".into(),
                kind: crate::preset::PresetKind::ControllerProfile,
                target: PresetTargetSnapshot::StageMap,
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::CaptureScopedPreset {
                name: "Stale".into(),
                kind: crate::preset::PresetKind::Transform,
                target: group,
                preset_revision: 0,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::ApplyScopedPreset {
                preset_id: "0".into(),
                target: layer,
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            WebAction::DeleteScopedPreset {
                preset_id: "not-an-id".into(),
                preset_revision: 1,
            },
        ] {
            assert!(!valid_action(&action, 0), "accepted {action:?}");
        }

        let overlong = "x".repeat(crate::preset::PRESET_MAX_NAME_BYTES + 1);
        assert!(!valid_action(
            &WebAction::CaptureScopedPreset {
                name: overlong,
                kind: crate::preset::PresetKind::Rack,
                target: PresetTargetSnapshot::Master,
                preset_revision: 1,
                layer_stack_revision: 2,
                composition_revision: 3,
            },
            0
        ));
    }

    #[tokio::test]
    async fn lagged_broadcast_receiver_can_jump_to_live_edge() {
        let (tx, mut rx) = tokio::sync::broadcast::channel(2);
        for value in 0..3 {
            tx.send(value).unwrap();
        }
        assert!(matches!(rx.recv().await, Err(RecvError::Lagged(_))));
        rx = rx.resubscribe();
        tx.send(3).unwrap();
        assert_eq!(rx.recv().await.unwrap(), 3);
    }
    #[test]
    fn gesture_ingress_bounds_every_field_and_closes_its_phase_and_mode_vocabularies() {
        use crate::gesture::{GestureMode, GesturePhase, MAX_ACTIVE_STROKES};

        let sample = |stroke: u8, x: f32, y: f32, pressure: f32, direction: [f32; 2]| {
            WebAction::GestureSample {
                stroke,
                phase: GesturePhase::Move,
                mode: GestureMode::Push,
                x,
                y,
                pressure,
                direction_x: direction[0],
                direction_y: direction[1],
            }
        };

        // The well-formed sample, and both inclusive extremes of every range.
        assert!(valid_action(&sample(0, 0.5, 0.5, 1.0, [0.0, 0.0]), 0));
        assert!(valid_action(&sample(0, 0.0, 0.0, 0.0, [-1.0, -1.0]), 0));
        assert!(valid_action(
            &sample((MAX_ACTIVE_STROKES - 1) as u8, 1.0, 1.0, 1.0, [1.0, 1.0]),
            0
        ));

        // The stroke identity space is the same constant the ingest validator
        // enforces, so the queue never carries a sample the adapter refuses.
        assert!(!valid_action(
            &sample(MAX_ACTIVE_STROKES as u8, 0.5, 0.5, 1.0, [0.0, 0.0]),
            0
        ));
        assert!(!valid_action(&sample(255, 0.5, 0.5, 1.0, [0.0, 0.0]), 0));

        // Every numeric field is finite and inside its own inclusive range.
        for hostile in [f32::NAN, f32::INFINITY, -0.001, 1.001] {
            assert!(
                !valid_action(&sample(0, hostile, 0.5, 1.0, [0.0, 0.0]), 0),
                "x {hostile}"
            );
            assert!(
                !valid_action(&sample(0, 0.5, hostile, 1.0, [0.0, 0.0]), 0),
                "y {hostile}"
            );
            assert!(
                !valid_action(&sample(0, 0.5, 0.5, hostile, [0.0, 0.0]), 0),
                "pressure {hostile}"
            );
        }
        for hostile in [f32::NAN, f32::NEG_INFINITY, -1.001, 1.001] {
            assert!(
                !valid_action(&sample(0, 0.5, 0.5, 1.0, [hostile, 0.0]), 0),
                "direction_x {hostile}"
            );
            assert!(
                !valid_action(&sample(0, 0.5, 0.5, 1.0, [0.0, hostile]), 0),
                "direction_y {hostile}"
            );
        }

        // The discrete vocabularies are typed engine enums rather than a
        // hand-maintained allowlist, so an unknown token fails at
        // deserialization and can never be silently dropped later at ingress.
        for phase in ["begin", "move", "end"] {
            let text = format!(
                r#"{{"action":"gesture_sample","stroke":0,"phase":"{phase}","mode":"push","x":0.5,"y":0.5}}"#
            );
            let action = serde_json::from_str::<WebAction>(&text).expect("closed phase token");
            assert!(valid_action(&action, 0), "{phase}");
        }
        for mode in ["push", "curl"] {
            let text = format!(
                r#"{{"action":"gesture_sample","stroke":0,"phase":"move","mode":"{mode}","x":0.5,"y":0.5}}"#
            );
            let action = serde_json::from_str::<WebAction>(&text).expect("closed mode token");
            assert!(valid_action(&action, 0), "{mode}");
        }
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"gesture_sample","stroke":0,"phase":"hold","mode":"push","x":0.5,"y":0.5}"#
        )
        .is_err());
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"gesture_sample","stroke":0,"phase":"move","mode":"etch","x":0.5,"y":0.5}"#
        )
        .is_err());

        // A hostile value that arrived as JSON is rejected by the same arm.
        let hostile = serde_json::from_str::<WebAction>(
            r#"{"action":"gesture_sample","stroke":0,"phase":"move","mode":"push","x":9.0,"y":0.5}"#,
        )
        .unwrap();
        assert!(!valid_action(&hostile, 0));

        // Recording control is revision-protected at ingress and is never
        // latchable: a sample must not cross an arm/disarm edge waiting for a
        // downbeat, and an arm decision must not outlive its program.
        assert!(valid_action(
            &WebAction::SetGestureRecording {
                enabled: true,
                layer_stack_revision: 7,
            },
            0
        ));
        assert!(
            !valid_action(
                &WebAction::SetGestureRecording {
                    enabled: true,
                    layer_stack_revision: 0,
                },
                0
            ),
            "a zero revision is never a live program"
        );
        assert!(!valid_action(
            &WebAction::Quantized {
                inner: Box::new(WebAction::SetGestureRecording {
                    enabled: true,
                    layer_stack_revision: 7,
                }),
            },
            0
        ));
    }

    #[test]
    fn b10_source_edits_are_gated_by_the_engines_own_vocabularies() {
        assert!(valid_action(
            &WebAction::SetEnvelope {
                index: 3,
                param: "trigger".to_string(),
                value: serde_json::json!("scene_cut"),
            },
            0
        ));
        assert!(!valid_action(
            &WebAction::SetEnvelope {
                index: 4,
                param: "attack".to_string(),
                value: serde_json::json!(0.1),
            },
            0
        ));
        assert!(!valid_action(
            &WebAction::SetEnvelope {
                index: 0,
                param: "trigger".to_string(),
                value: serde_json::json!("beat3"),
            },
            0
        ));
        assert!(!valid_action(
            &WebAction::SetEnvelope {
                index: 0,
                param: "attack".to_string(),
                value: serde_json::json!(f64::NAN),
            },
            0
        ));
        assert!(valid_action(
            &WebAction::SetMacro {
                index: 0,
                value: 0.5,
            },
            0
        ));
        assert!(!valid_action(
            &WebAction::SetMacro {
                index: 4,
                value: 0.5,
            },
            0
        ));
        assert!(valid_action(&WebAction::SetModSeed { seed: 77 }, 0));
        assert!(valid_action(
            &WebAction::BendPad {
                index: 5,
                held: true,
            },
            0
        ));
        assert!(!valid_action(
            &WebAction::BendPad {
                index: 6,
                held: true,
            },
            0
        ));
        // A bend edge is authored against the frame it was played in.
        assert!(!valid_action(
            &WebAction::Quantized {
                inner: Box::new(WebAction::BendPad {
                    index: 0,
                    held: true,
                }),
            },
            0
        ));
    }

    #[test]
    fn performance_transports_are_revision_guarded_and_never_latchable() {
        // The B9 transports carry the gesture barrier's exact ingress law: a
        // zero revision is never a live program, and a latched arm/disarm
        // edge would attach a take decision to the wrong stretch of program
        // time.
        assert!(valid_action(
            &WebAction::SetPerformanceRecording {
                enabled: true,
                layer_stack_revision: 7,
            },
            0
        ));
        assert!(!valid_action(
            &WebAction::SetPerformanceRecording {
                enabled: true,
                layer_stack_revision: 0,
            },
            0
        ));
        assert!(valid_action(
            &WebAction::SetPerformancePlayback {
                enabled: true,
                loop_playback: true,
                layer_stack_revision: 7,
            },
            0
        ));
        assert!(!valid_action(
            &WebAction::SetPerformancePlayback {
                enabled: true,
                loop_playback: true,
                layer_stack_revision: 0,
            },
            0
        ));
        assert!(valid_action(&WebAction::ClearPerformanceTake, 0));
        for inner in [
            WebAction::SetPerformanceRecording {
                enabled: true,
                layer_stack_revision: 7,
            },
            WebAction::SetPerformancePlayback {
                enabled: true,
                loop_playback: false,
                layer_stack_revision: 7,
            },
            WebAction::ClearPerformanceTake,
        ] {
            assert!(!valid_action(
                &WebAction::Quantized {
                    inner: Box::new(inner),
                },
                0
            ));
        }
    }

    /// The authored canvas vocabulary is closed, finite-bounded, and contains
    /// no key that could reach the recorded track.
    #[test]
    fn gesture_canvas_ingress_is_a_closed_finite_vocabulary_with_no_track_key() {
        for param in ["radius", "strength", "retention"] {
            for accepted in [0.0, 0.5, 1.0] {
                assert!(
                    valid_action(
                        &WebAction::SetGestureCanvas {
                            param: param.into(),
                            value: serde_json::json!(accepted),
                        },
                        0
                    ),
                    "{param} must accept {accepted}"
                );
            }
            for rejected in [
                serde_json::json!(-0.001),
                serde_json::json!(1.001),
                serde_json::json!(f64::NAN),
                serde_json::json!(f64::INFINITY),
                serde_json::json!("0.5"),
                serde_json::json!(null),
            ] {
                assert!(
                    !valid_action(
                        &WebAction::SetGestureCanvas {
                            param: param.into(),
                            value: rejected.clone(),
                        },
                        0
                    ),
                    "{param} must refuse {rejected}"
                );
            }
        }

        // Neither the recording nor any neighbouring spelling is authorable
        // through the value path.
        for hostile in [
            "track",
            "events",
            "checksum",
            "recording",
            "decay",
            "radius_x",
            "",
        ] {
            assert!(
                !valid_action(
                    &WebAction::SetGestureCanvas {
                        param: hostile.into(),
                        value: serde_json::json!(0.5),
                    },
                    0
                ),
                "{hostile} must not be an authorable canvas parameter"
            );
        }

        let decoded = serde_json::from_str::<WebAction>(
            r#"{"action":"set_gesture_canvas","param":"strength","value":0.25}"#,
        )
        .unwrap();
        assert!(valid_action(&decoded, 0));
    }
    #[test]
    fn generator_wire_actions_validate_through_the_shared_parse_tables() {
        // Topology adds are plain immediate actions.
        assert!(valid_action(&WebAction::AddPatternLayer, 0));
        assert!(valid_action(&WebAction::AddTextLayer, 0));

        let pattern = |param: &str, value: serde_json::Value| WebAction::SetLayerPattern {
            index: 0,
            layer_id: Some("7".into()),
            param: param.into(),
            value,
        };
        // Continuous values inside range, closed tokens, and both accepted.
        for (param, value) in [
            ("freq_x", serde_json::json!(0.5)),
            ("rate", serde_json::json!(-0.5)),
            ("symmetry", serde_json::json!(8.0)),
            ("hue_spread", serde_json::json!(1.5)),
            ("shape", serde_json::json!("tunnel")),
            ("wave", serde_json::json!("sample_hold")),
            ("color_mode", serde_json::json!("duotone")),
        ] {
            assert!(valid_action(&pattern(param, value.clone()), 0), "{param}");
        }
        // Out-of-range, non-finite, unknown tokens, and unknown params are
        // rejections at the gate, exactly what the applier refuses.
        for (param, value) in [
            ("freq_x", serde_json::json!(2.0)),
            ("rate", serde_json::json!(f64::NAN)),
            ("shape", serde_json::json!("hypercube")),
            ("wave", serde_json::json!(3)),
            ("voltage", serde_json::json!(0.5)),
        ] {
            assert!(!valid_action(&pattern(param, value.clone()), 0), "{param}");
        }

        let text = |param: &str, value: serde_json::Value| WebAction::SetLayerText {
            index: 0,
            layer_id: Some("7".into()),
            param: param.into(),
            value,
        };
        for (param, value) in [
            (
                "body",
                serde_json::json!(
                    "HELLO
WORLD"
                ),
            ),
            ("font", serde_json::json!("sans")),
            ("shape", serde_json::json!("starburst")),
            ("repeat", serde_json::json!(3)),
            ("shape_count", serde_json::json!(12)),
            ("size", serde_json::json!(0.3)),
            ("rot_degrees", serde_json::json!(-90.0)),
            ("ink_r", serde_json::json!(0.5)),
        ] {
            assert!(valid_action(&text(param, value.clone()), 0), "{param}");
        }
        for (param, value) in [
            ("body", serde_json::json!("x".repeat(5000))),
            ("font", serde_json::json!("papyrus")),
            ("repeat", serde_json::json!(0)),
            ("shape_count", serde_json::json!(99)),
            ("size", serde_json::json!(5.0)),
            ("scroll_x", serde_json::json!(0.5)),
        ] {
            assert!(!valid_action(&text(param, value.clone()), 0), "{param}");
        }

        // The wire spellings are the documented ones.
        let decoded = serde_json::from_str::<WebAction>(
            r#"{"action":"set_layer_pattern","index":0,"layer_id":"3","param":"wavefold","value":0.5}"#,
        )
        .unwrap();
        assert!(valid_action(&decoded, 0));
        let decoded = serde_json::from_str::<WebAction>(
            r#"{"action":"set_layer_text","index":0,"param":"body","value":"PAGE"}"#,
        )
        .unwrap();
        assert!(valid_action(&decoded, 0));
        assert!(serde_json::from_str::<WebAction>(r#"{"action":"add_pattern_layer"}"#).is_ok());
        assert!(serde_json::from_str::<WebAction>(r#"{"action":"add_text_layer"}"#).is_ok());
    }
}
