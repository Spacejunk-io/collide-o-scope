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

use super::state::{EnqueueOutcome, RerollScope, WebAction, WebState};
use super::static_files;

const AUTH_COOKIE: &str = "cos_key";
const MAX_WS_MESSAGE_BYTES: usize = 16 * 1024;
const MAX_LOGGED_MESSAGE_CHARS: usize = 256;
const MAX_ACTION_VALUE_DEPTH: usize = 8;
const MAX_ACTION_VALUE_NODES: usize = 512;
const MAX_ACTION_VALUE_STRING_BYTES: usize = 2048;
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

fn valid_f64_for_f32(value: f64) -> bool {
    value.is_finite() && value.abs() <= f64::from(f32::MAX)
}

fn valid_action(action: &WebAction, depth: usize) -> bool {
    if depth > 2 {
        return false;
    }
    match action {
        WebAction::Quantized { inner } => {
            !matches!(inner.as_ref(), WebAction::SetMediaSafetyMode { .. })
                && valid_action(inner, depth + 1)
        }
        WebAction::AddLayer { filename } => valid_identifier(filename, 1024),
        WebAction::AddSpoutLayer { sender } => valid_identifier(sender, 255),
        WebAction::SetLayerParam {
            layer_id,
            param,
            value,
            ..
        }
        | WebAction::SetLayerEffect {
            layer_id,
            param,
            value,
            ..
        } => {
            valid_optional_layer_id(layer_id)
                && valid_identifier(param, 64)
                && valid_json_value(value)
        }
        WebAction::RemoveLayer { layer_id, .. }
        | WebAction::ResetLayerFx { layer_id, .. }
        | WebAction::SetLayerVisibility { layer_id, .. }
        | WebAction::SetLayerPaused { layer_id, .. }
        | WebAction::MoveLayer { layer_id, .. } => valid_optional_layer_id(layer_id),
        WebAction::SetLayerRerollOnLoop { layer_id, .. } => valid_optional_stable_id(layer_id),
        WebAction::Reroll {
            scope,
            index,
            layer_id,
            stack_revision,
            amount,
            ..
        } => {
            amount.is_finite()
                && (0.0..=2.0).contains(amount)
                && match scope {
                    RerollScope::Master => {
                        index.is_none() && layer_id.is_none() && stack_revision.is_none()
                    }
                    RerollScope::Layer => {
                        index.is_some()
                            && layer_id.is_some()
                            && valid_optional_stable_id(layer_id)
                            && stack_revision.is_none()
                    }
                    RerollScope::All => {
                        index.is_none()
                            && layer_id.is_none()
                            && stack_revision.is_some_and(|revision| revision != 0)
                    }
                }
        }
        WebAction::SetParam { param, value }
        | WebAction::SetNtscParam { param, value }
        | WebAction::SetAudio { param, value }
        | WebAction::SetMidi { param, value }
        | WebAction::SetTemporal { param, value } => {
            valid_identifier(param, 64) && valid_json_value(value)
        }
        WebAction::SetLfo { param, value, .. } => {
            valid_identifier(param, 64) && valid_json_value(value)
        }
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
                    None => layer_stack_revision.is_none(),
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
        WebAction::MorphGlide {
            target,
            duration_beats,
        } => valid_f32(*target) && valid_f64_for_f32(*duration_beats),
        WebAction::MorphCapture {
            slot,
            stack_revision,
        } => {
            matches!(slot.as_str(), "a" | "b")
                && stack_revision.is_none_or(|revision| revision != 0)
        }
        WebAction::SetMorphLaw { law } => matches!(law.as_str(), "linear" | "equal_power"),
        WebAction::ResetGroup { group } => valid_identifier(group, 32),
        WebAction::StartExport {
            width,
            height,
            fps,
            duration_secs,
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
                        action => {
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

    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
    state.disconnect_gyro_client(client_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::Message as ClientMessage;

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

        socket.close(None).await.unwrap();
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
                stack_revision,
                seed: None,
                mode: crate::web::state::RerollMode::Pattern,
                amount,
                include_grain_controls: false,
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
    fn morph_enums_and_capture_revision_are_strictly_validated() {
        assert!(valid_action(
            &WebAction::MorphCapture {
                slot: "a".into(),
                stack_revision: Some(9),
            },
            0,
        ));
        assert!(!valid_action(
            &WebAction::MorphCapture {
                slot: "typo".into(),
                stack_revision: Some(9),
            },
            0,
        ));
        assert!(!valid_action(
            &WebAction::MorphCapture {
                slot: "b".into(),
                stack_revision: Some(0),
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
}
