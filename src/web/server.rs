//! Axum HTTP + WebSocket server for the control panel.
//!
//! Two listeners share one router: plain HTTP on the base port (desktop
//! convenience) and HTTPS with a self-signed certificate on base+1 — the
//! QR code points at the HTTPS URL because iOS only exposes motion sensors
//! to secure contexts. The certificate (SANs: localhost + the LAN IP) is
//! persisted under %LOCALAPPDATA%, so a phone that accepted it once stays
//! trusted across restarts; it regenerates only when the LAN IP changes.
//!
//! Access control: loopback clients pass freely; every other client must
//! present the per-session token — normally by scanning the QR code, whose
//! URL carries `?key=…` — after which a cookie keeps them authenticated
//! for the session. Unknown LAN clients get 403.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{ConnectInfo, DefaultBodyLimit, Path, Query, Request, State, WebSocketUpgrade};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures::{SinkExt, StreamExt};
use tokio::io::AsyncWriteExt;

use super::state::{WebAction, WebState};
use super::static_files;

const AUTH_COOKIE: &str = "cos_key";

/// Start the web server on a background thread. Returns the local URL.
pub fn spawn(state: Arc<WebState>, port: u16) -> String {
    let local_url = format!("http://127.0.0.1:{port}");
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
            let app = Router::new()
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
                .with_state(state);

            // HTTPS listener (phones — secure context for sensors).
            if let Ok((cert_chain, key_der)) = tls {
                let https_app = app.clone();
                tokio::spawn(async move {
                    match axum_server::tls_rustls::RustlsConfig::from_der(cert_chain, key_der)
                        .await
                    {
                        Ok(config) => {
                            let addr = SocketAddr::from(([0, 0, 0, 0], https_port));
                            log::info!("HTTPS control panel listening on {addr}");
                            if let Err(e) = axum_server::bind_rustls(addr, config)
                                .serve(
                                    https_app
                                        .into_make_service_with_connect_info::<SocketAddr>(),
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
                    log::error!(
                        "Cannot bind port {port} ({e}) — is another collide-o-scope \
                         already running? This instance continues without a control panel."
                    );
                    return;
                }
            };
            log::info!("Web control panel listening on {addr}");
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });
    });

    local_url
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

/// Gate: loopback passes; LAN needs the session token via cookie or the
/// `?key=` query param (which then sets the cookie).
async fn auth(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<Arc<WebState>>,
    req: Request,
    next: Next,
) -> Response {
    if addr.ip().is_loopback() {
        return next.run(req).await;
    }

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

    if cookie_ok {
        return next.run(req).await;
    }
    if query_ok {
        let mut response = next.run(req).await;
        if let Ok(cookie) = header::HeaderValue::from_str(&format!(
            "{AUTH_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax"
        )) {
            response.headers_mut().append(header::SET_COOKIE, cookie);
        }
        return response;
    }

    log::warn!("Rejected unauthenticated LAN client: {}", addr.ip());
    (
        StatusCode::FORBIDDEN,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        "<h3>collide-o-scope</h3><p>Access denied. Scan the QR code in the control panel to connect.</p>",
    )
        .into_response()
}

/// QR code (SVG) of the remote URL, rendered on demand.
async fn qr_handler(State(state): State<Arc<WebState>>) -> impl IntoResponse {
    let url = state
        .lan_url
        .read()
        .map(|s| s.clone())
        .unwrap_or_default();

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

/// Streamed clip upload into the library folder. The body is written in
/// chunks to a temp file (never buffered whole in memory), renamed into
/// place on success, and the render thread is asked to rescan. Names are
/// reduced to their final path component and must carry a known video
/// extension; collisions get a numbered suffix rather than overwriting.
async fn upload_handler(
    State(state): State<Arc<WebState>>,
    Query(query): Query<UploadQuery>,
    body: axum::body::Body,
) -> Response {
    // Sanitize: no directory components, no hidden files.
    let name = std::path::Path::new(&query.name)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = std::path::Path::new(&name)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if name.is_empty() || name.starts_with('.')
        || !matches!(ext.as_str(), "mp4" | "webm" | "mov" | "avi" | "mkv")
    {
        return (StatusCode::BAD_REQUEST, "unsupported file type").into_response();
    }

    let Some(folder) = state.library_folder.read().ok().and_then(|f| f.clone()) else {
        return (
            StatusCode::CONFLICT,
            "no library folder open — load a folder in the app first",
        )
            .into_response();
    };

    // Collisions get " (n)" suffixes instead of overwriting.
    let stem = std::path::Path::new(&name)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "clip".to_string());
    let mut final_name = name.clone();
    let mut counter = 1;
    while folder.join(&final_name).exists() {
        final_name = format!("{stem} ({counter}).{ext}");
        counter += 1;
    }
    let final_path = folder.join(&final_name);
    let temp_path = folder.join(format!(".upload-{final_name}.part"));

    let mut file = match tokio::fs::File::create(&temp_path).await {
        Ok(f) => f,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("create: {e}"))
                .into_response()
        }
    };

    let mut stream = body.into_data_stream();
    let mut written: u64 = 0;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                if let Err(e) = file.write_all(&bytes).await {
                    drop(file);
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    return (StatusCode::INTERNAL_SERVER_ERROR, format!("write: {e}"))
                        .into_response();
                }
                written += bytes.len() as u64;
            }
            Err(e) => {
                drop(file);
                let _ = tokio::fs::remove_file(&temp_path).await;
                return (StatusCode::BAD_REQUEST, format!("stream: {e}")).into_response();
            }
        }
    }
    if file.flush().await.is_err() || written == 0 {
        drop(file);
        let _ = tokio::fs::remove_file(&temp_path).await;
        return (StatusCode::BAD_REQUEST, "empty upload").into_response();
    }
    drop(file);

    if let Err(e) = tokio::fs::rename(&temp_path, &final_path).await {
        let _ = tokio::fs::remove_file(&temp_path).await;
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("rename: {e}")).into_response();
    }

    log::info!("Uploaded clip: {final_name} ({written} bytes)");
    state.actions.lock().await.push(WebAction::RescanLibrary);

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
    let name = std::path::Path::new(&query.name)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = std::path::Path::new(&name)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if name.is_empty() || !matches!(ext.as_str(), "mp4" | "webm" | "mov" | "avi" | "mkv") {
        return (StatusCode::BAD_REQUEST, "not a library clip").into_response();
    }

    let Some(folder) = state.library_folder.read().ok().and_then(|f| f.clone()) else {
        return (StatusCode::CONFLICT, "no library folder open").into_response();
    };
    let path = folder.join(&name);
    if !path.is_file() {
        return (StatusCode::NOT_FOUND, "clip not found").into_response();
    }

    match tokio::task::spawn_blocking(move || trash::delete(&path)).await {
        Ok(Ok(())) => {
            if let Ok(mut cache) = state.thumbnails.write() {
                cache.remove(&name);
            }
            if let Ok(mut cache) = state.preview_frames.write() {
                cache.remove(&name);
            }
            log::info!("Clip moved to Recycle Bin: {name}");
            state.actions.lock().await.push(WebAction::RescanLibrary);
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

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<WebState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<WebState>) {
    let (mut sender, mut receiver) = socket.split();

    // Send current state on connect
    let current = state.app.read().await;
    let init_msg = serde_json::to_string(&*current).unwrap();
    drop(current);
    let _ = sender.send(Message::Text(init_msg.into())).await;

    // Subscribe to broadcast updates (state JSON)
    let mut rx = state.tx.subscribe();

    // Forward broadcasts to this client
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg.into())).await.is_err() {
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
                    Ok(action) => {
                        state_clone.actions.lock().await.push(action);
                    }
                    Err(e) => {
                        log::warn!("Failed to parse WebAction: {e} — raw: {text}");
                    }
                }
            }
        }
    });

    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}
