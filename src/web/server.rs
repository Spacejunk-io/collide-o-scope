//! Axum HTTP + WebSocket server for the control panel.
//!
//! Plain HTTP is confined to separately owned IPv4 and IPv6 loopback sockets.
//! LAN control exists only through HTTPS on base+1; TLS failure publishes an
//! honest unavailable state and never creates a plaintext fallback. The
//! certificate/key/SAN identity is one permission-safe atomic generation.
//!
//! Access control: every client must present the per-session token — normally
//! through the app-opened URL or QR code — after which a strict cookie keeps
//! it authenticated. WebSockets and mutation POSTs also require an exact
//! same-origin Origin header. Unknown and cross-origin clients get 403.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener};
use std::path::{Path as FsPath, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{
    ConnectInfo, DefaultBodyLimit, FromRef, Path, Query, Request, State, WebSocketUpgrade,
};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures::{FutureExt, SinkExt, StreamExt};
use sha2::Digest as _;
use tokio::io::AsyncWriteExt;

use crate::durable_file::{self, AdmissionError, AdmissionLimits, PublishMode, StagedPublication};

use super::state::{
    ActionIngressAck, ActionIngressTerminalGuard, CaptureTargetSnapshot, CompositionRootSnapshot,
    ControlAccessUrl, ControlListenerSlot, ControlListenerStatus, ControlServerInfo,
    CreativeImageSourceSnapshot, CreativeImageTapSnapshot, CreativeScopeSnapshot, EnqueueOutcome,
    ImageInputSnapshot, MotionScopeSnapshot, PresetTargetSnapshot, RerollScope,
    SymmetryRouteSnapshot, WebAction, WebState, MAX_SCENE_NAME_BYTES,
};
use super::static_files;
use super::tls_identity::{self, IdentityFaults};

const LOOPBACK_AUTH_COOKIE: &str = "cos_loopback";
const LAN_AUTH_COOKIE: &str = "cos_lan";
const MAX_LISTENER_REASON_CHARS: usize = 256;
const RETIRE_TIMEOUT: Duration = Duration::from_secs(3);
const RESTART_BACKOFF: [Duration; 3] = [
    Duration::ZERO,
    Duration::from_millis(75),
    Duration::from_millis(250),
];
const MAX_WS_MESSAGE_BYTES: usize = super::action_wire::MAX_WEB_ACTION_BYTES;
const MAX_LOGGED_MESSAGE_CHARS: usize = 256;
const MAX_ACTION_VALUE_DEPTH: usize = 8;
const MAX_ACTION_VALUE_NODES: usize = 512;
const MAX_ACTION_VALUE_STRING_BYTES: usize = 2048;
const MAX_JS_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
/// Audio is decoded to at most ten minutes of canonical mono PCM. Bound the
/// upload itself as well so a malformed or accidental multi-gigabyte file
/// cannot consume library storage before FFmpeg gets a chance to reject it.
const MAX_AUDIO_UPLOAD_BYTES: u64 = 512 * 1024 * 1024;
const MAX_VISUAL_UPLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_UPLOAD_AGGREGATE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MIN_UPLOAD_DISK_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;
const MAX_CONCURRENT_UPLOADS: usize = 2;
const UPLOAD_CHUNK_IDLE_TIMEOUT: Duration = Duration::from_secs(15);
const UPLOAD_ABSOLUTE_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const MAX_UPLOAD_DESTINATION_ATTEMPTS: u32 = 4_096;
const MAX_UPLOAD_ORPHAN_SCAN_ENTRIES: usize = 8_192;
// Leave room for the atomic reservation prefix and the largest numbered
// collision suffix while staying below Windows' 255 UTF-16-code-unit
// component limit.
const MAX_LIBRARY_FILENAME_UTF16: usize = 220;
const MAX_CONCURRENT_LIBRARY_PAGE_SEARCHES: usize = 4;

fn library_page_gate() -> Arc<tokio::sync::Semaphore> {
    static GATE: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    GATE.get_or_init(|| {
        Arc::new(tokio::sync::Semaphore::new(
            MAX_CONCURRENT_LIBRARY_PAGE_SEARCHES,
        ))
    })
    .clone()
}

fn exceeds_upload_limit(extension: &str, bytes: u64) -> bool {
    bytes > upload_limit_for_extension(extension)
}

fn upload_limit_for_extension(extension: &str) -> u64 {
    if crate::audio::is_supported_audio_extension(extension) {
        MAX_AUDIO_UPLOAD_BYTES
    } else {
        MAX_VISUAL_UPLOAD_BYTES
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ListenerRole {
    LoopbackIpv4,
    LoopbackIpv6,
    LanTls,
}

impl ListenerRole {
    const fn slot(self) -> ControlListenerSlot {
        match self {
            Self::LoopbackIpv4 => ControlListenerSlot::LoopbackIpv4,
            Self::LoopbackIpv6 => ControlListenerSlot::LoopbackIpv6,
            Self::LanTls => ControlListenerSlot::LanTls,
        }
    }

    const fn is_loopback(self) -> bool {
        matches!(self, Self::LoopbackIpv4 | Self::LoopbackIpv6)
    }

    const fn cookie_name(self) -> &'static str {
        if self.is_loopback() {
            LOOPBACK_AUTH_COOKIE
        } else {
            LAN_AUTH_COOKIE
        }
    }

    const fn cookie_is_secure(self) -> bool {
        matches!(self, Self::LanTls)
    }

    const fn label(self) -> &'static str {
        match self {
            Self::LoopbackIpv4 => "loopback IPv4 HTTP",
            Self::LoopbackIpv6 => "loopback IPv6 HTTP",
            Self::LanTls => "LAN HTTPS",
        }
    }
}

struct SessionSecret(String);

impl std::fmt::Debug for SessionSecret {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SessionSecret(<redacted>)")
    }
}

#[derive(Debug)]
struct ControlSession {
    token: SessionSecret,
    fingerprint: String,
}

impl ControlSession {
    fn random() -> Result<Arc<Self>, String> {
        let mut token = [0_u8; 16];
        getrandom::fill(&mut token)
            .map_err(|error| format!("OS entropy unavailable for control session: {error}"))?;
        Self::from_seed(hex_lower(&token))
    }

    fn from_seed(token: String) -> Result<Arc<Self>, String> {
        if token.len() != 32 || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(
                "control session token must be exactly 128 bits of hexadecimal".to_string(),
            );
        }
        let token = token.to_ascii_lowercase();
        let fingerprint = hex_lower(&sha2::Sha256::digest(token.as_bytes()))[..12].to_string();
        Ok(Arc::new(Self {
            token: SessionSecret(token),
            fingerprint,
        }))
    }

    fn access_url(&self, scheme: &str, address: SocketAddr) -> ControlAccessUrl {
        ControlAccessUrl::new(format!("{scheme}://{address}/?key={}", self.token.0))
    }
}

#[derive(Clone)]
struct ControlRouterState {
    web: Arc<WebState>,
    session: Arc<ControlSession>,
    role: ListenerRole,
}

impl FromRef<ControlRouterState> for Arc<WebState> {
    fn from_ref(state: &ControlRouterState) -> Self {
        state.web.clone()
    }
}

#[derive(Clone, Copy, Default)]
struct StartFaults {
    identity: IdentityFaults,
    tls_config: bool,
    crash_role: Option<ListenerRole>,
}

struct StartConfig {
    base_port: u16,
    lan_ip: Option<IpAddr>,
    identity_dir: PathBuf,
    token_seed: Option<String>,
    faults: StartFaults,
}

impl StartConfig {
    fn production(base_port: u16) -> Self {
        Self {
            base_port,
            lan_ip: detect_lan_ip(),
            identity_dir: tls_identity::default_identity_dir(),
            token_seed: None,
            faults: StartFaults::default(),
        }
    }
}

enum PreparedListener {
    Plain {
        role: ListenerRole,
        listener: TcpListener,
        router: Router,
        handle: axum_server::Handle,
    },
    Tls {
        listener: TcpListener,
        router: Router,
        handle: axum_server::Handle,
        config: axum_server::tls_rustls::RustlsConfig,
    },
}

/// Owns the runtime thread, all listener shutdown handles, the generation's
/// secret session, and the final join/retirement witness.
pub struct ControlServerHandle {
    generation: u64,
    base_port: u16,
    state: Arc<WebState>,
    session: Arc<ControlSession>,
    local_url: Option<ControlAccessUrl>,
    listeners: Vec<axum_server::Handle>,
    stopping: Arc<AtomicBool>,
    finished: mpsc::Receiver<()>,
    thread: Option<std::thread::JoinHandle<()>>,
    loopback_bound: bool,
}

impl std::fmt::Debug for ControlServerHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ControlServerHandle")
            .field("generation", &self.generation)
            .field("base_port", &self.base_port)
            .field("session_fingerprint", &self.session.fingerprint)
            .field("local_url", &self.local_url)
            .field("listener_count", &self.listeners.len())
            .field("loopback_bound", &self.loopback_bound)
            .finish()
    }
}

impl ControlServerHandle {
    pub fn local_url(&self) -> Option<&str> {
        self.local_url
            .as_ref()
            .map(ControlAccessUrl::expose_to_local_ui)
    }

    pub fn session_fingerprint(&self) -> &str {
        &self.session.fingerprint
    }

    pub fn has_loopback_listener(&self) -> bool {
        self.loopback_bound
    }

    pub fn retire(&mut self) -> Result<(), String> {
        let Some(_) = self.thread.as_ref() else {
            self.state.mark_control_server_stopped(self.generation);
            return Ok(());
        };
        self.state.mark_control_server_stopping(self.generation);
        self.stopping.store(true, Ordering::Release);
        for listener in &self.listeners {
            listener.shutdown();
        }
        match self.finished.recv_timeout(RETIRE_TIMEOUT) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(format!(
                    "control server generation {} did not retire within {} ms",
                    self.generation,
                    RETIRE_TIMEOUT.as_millis()
                ));
            }
        }
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| {
                "control server runtime thread panicked during retirement".to_string()
            })?;
        }
        self.state.mark_control_server_stopped(self.generation);
        Ok(())
    }

    #[cfg(test)]
    fn token_for_test(&self) -> &str {
        &self.session.token.0
    }
}

impl Drop for ControlServerHandle {
    fn drop(&mut self) {
        if let Err(error) = self.retire() {
            log::error!("{error}; waiting for owned control runtime to finish");
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

/// Start one server generation. Partial listener success is returned as a
/// handle and represented in [`ControlServerInfo`]; only entropy/runtime setup
/// failures prevent creation of the owned handle itself.
pub fn spawn(state: Arc<WebState>, port: u16) -> Result<ControlServerHandle, String> {
    start(state, StartConfig::production(port))
}

/// Retire/rebind with a finite three-attempt backoff. A failed old retirement
/// returns before any new bind, so two generations never race for the ports.
pub fn restart_with_backoff(
    state: Arc<WebState>,
    port: u16,
) -> Result<ControlServerHandle, String> {
    let mut last: Option<ControlServerHandle> = None;
    for delay in RESTART_BACKOFF {
        if let Some(mut prior) = last.take() {
            prior.retire()?;
        }
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }
        let handle = spawn(state.clone(), port)?;
        if handle.has_loopback_listener() {
            return Ok(handle);
        }
        last = Some(handle);
    }
    Ok(last.expect("the bounded restart table has at least one attempt"))
}

/// Hidden packet-test fixture: the caller supplies an isolated identity root
/// and a deterministic non-production token, but receives no token in output.
#[cfg(debug_assertions)]
pub(crate) fn spawn_fixture(
    state: Arc<WebState>,
    port: u16,
    identity_dir: &FsPath,
    token_seed: String,
) -> Result<ControlServerHandle, String> {
    start(
        state,
        StartConfig {
            base_port: port,
            lan_ip: detect_lan_ip(),
            identity_dir: identity_dir.to_path_buf(),
            token_seed: Some(token_seed),
            faults: StartFaults::default(),
        },
    )
}

fn start(state: Arc<WebState>, config: StartConfig) -> Result<ControlServerHandle, String> {
    let https_port = config
        .base_port
        .checked_add(1)
        .ok_or_else(|| "control server base port cannot be 65535".to_string())?;
    let session = match config.token_seed {
        Some(token) => ControlSession::from_seed(token)?,
        None => ControlSession::random()?,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|error| format!("control server runtime creation failed: {error}"))?;
    let generation = state.begin_control_server_generation();
    state.publish_control_server_info(ControlServerInfo {
        generation,
        loopback_ipv4: ControlListenerStatus::Starting,
        loopback_ipv6: ControlListenerStatus::Starting,
        lan_tls: ControlListenerStatus::Starting,
        loopback_ipv4_url: None,
        loopback_ipv6_url: None,
        lan_url: None,
        session_fingerprint: session.fingerprint.clone(),
    });

    let mut info = ControlServerInfo {
        generation,
        session_fingerprint: session.fingerprint.clone(),
        ..ControlServerInfo::default()
    };
    let mut prepared = Vec::new();
    let mut shutdown_handles = Vec::new();

    let ipv4_address = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), config.base_port);
    match bind(ipv4_address) {
        Ok((listener, address)) => {
            info.loopback_ipv4 = ControlListenerStatus::Listening { address };
            info.loopback_ipv4_url = Some(session.access_url("http", address));
            let handle = axum_server::Handle::new();
            shutdown_handles.push(handle.clone());
            prepared.push(PreparedListener::Plain {
                role: ListenerRole::LoopbackIpv4,
                listener,
                router: control_router(state.clone(), session.clone(), ListenerRole::LoopbackIpv4),
                handle,
            });
        }
        Err(error) => {
            info.loopback_ipv4 = unavailable(format!("cannot bind loopback IPv4: {error}"));
        }
    }

    let ipv6_address = SocketAddr::new(Ipv6Addr::LOCALHOST.into(), config.base_port);
    match bind(ipv6_address) {
        Ok((listener, address)) => {
            info.loopback_ipv6 = ControlListenerStatus::Listening { address };
            info.loopback_ipv6_url = Some(session.access_url("http", address));
            let handle = axum_server::Handle::new();
            shutdown_handles.push(handle.clone());
            prepared.push(PreparedListener::Plain {
                role: ListenerRole::LoopbackIpv6,
                listener,
                router: control_router(state.clone(), session.clone(), ListenerRole::LoopbackIpv6),
                handle,
            });
        }
        Err(error) => {
            info.loopback_ipv6 = unavailable(format!("cannot bind loopback IPv6: {error}"));
        }
    }

    match prepare_tls(config.lan_ip, &config.identity_dir, config.faults, &runtime) {
        Ok((lan_ip, tls_config)) => {
            let requested = SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), https_port);
            match bind(requested) {
                Ok((listener, bound_address)) => {
                    info.lan_tls = ControlListenerStatus::Listening {
                        address: bound_address,
                    };
                    let advertised = SocketAddr::new(lan_ip, bound_address.port());
                    info.lan_url = Some(session.access_url("https", advertised));
                    let handle = axum_server::Handle::new();
                    shutdown_handles.push(handle.clone());
                    prepared.push(PreparedListener::Tls {
                        listener,
                        router: control_router(
                            state.clone(),
                            session.clone(),
                            ListenerRole::LanTls,
                        ),
                        handle,
                        config: tls_config,
                    });
                }
                Err(error) => {
                    info.lan_tls = unavailable(format!("cannot bind LAN HTTPS: {error}"));
                }
            }
        }
        Err(error) => info.lan_tls = unavailable(error),
    }

    let loopback_bound = matches!(info.loopback_ipv4, ControlListenerStatus::Listening { .. })
        || matches!(info.loopback_ipv6, ControlListenerStatus::Listening { .. });
    let local_url = info.local_url().cloned();
    state.publish_control_server_info(info.clone());

    for (role, status) in [
        (ListenerRole::LoopbackIpv4, &info.loopback_ipv4),
        (ListenerRole::LoopbackIpv6, &info.loopback_ipv6),
        (ListenerRole::LanTls, &info.lan_tls),
    ] {
        match status {
            ControlListenerStatus::Listening { address } => log::info!(
                "{} listening on {address} (session {})",
                role.label(),
                session.fingerprint
            ),
            ControlListenerStatus::Unavailable { reason } => {
                log::warn!("{} unavailable: {reason}", role.label())
            }
            ControlListenerStatus::Starting | ControlListenerStatus::Stopped => {}
        }
    }

    let stopping = Arc::new(AtomicBool::new(false));
    let thread_stopping = stopping.clone();
    let thread_state = state.clone();
    let crash_role = config.faults.crash_role;
    let active_slots = prepared
        .iter()
        .map(|listener| match listener {
            PreparedListener::Plain { role, .. } => role.slot(),
            PreparedListener::Tls { .. } => ControlListenerSlot::LanTls,
        })
        .collect::<Vec<_>>();
    let (finished_tx, finished_rx) = mpsc::channel();
    let thread = match std::thread::Builder::new()
        .name(format!("cos-control-{generation}"))
        .spawn(move || {
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.block_on(run_listeners(
                    prepared,
                    thread_state.clone(),
                    generation,
                    thread_stopping.clone(),
                    crash_role,
                ));
            }));
            if run.is_err() && !thread_stopping.load(Ordering::Acquire) {
                for slot in active_slots {
                    thread_state.update_control_listener(
                        generation,
                        slot,
                        unavailable("control listener runtime crashed"),
                    );
                }
            }
            let _ = finished_tx.send(());
        }) {
        Ok(thread) => thread,
        Err(error) => {
            for slot in [
                ControlListenerSlot::LoopbackIpv4,
                ControlListenerSlot::LoopbackIpv6,
                ControlListenerSlot::LanTls,
            ] {
                state.update_control_listener(
                    generation,
                    slot,
                    unavailable("control server thread creation failed"),
                );
            }
            return Err(format!("control server thread creation failed: {error}"));
        }
    };

    Ok(ControlServerHandle {
        generation,
        base_port: config.base_port,
        state,
        session,
        local_url,
        listeners: shutdown_handles,
        stopping,
        finished: finished_rx,
        thread: Some(thread),
        loopback_bound,
    })
}

async fn run_listeners(
    listeners: Vec<PreparedListener>,
    state: Arc<WebState>,
    generation: u64,
    stopping: Arc<AtomicBool>,
    crash_role: Option<ListenerRole>,
) {
    // No listener accepts a body until one bounded startup pass has removed
    // staging/reservation artifacts left by a prior crashed process.
    if let Some(folder) = state
        .library_folder
        .read()
        .ok()
        .and_then(|folder| folder.clone())
    {
        let cleanup_generation = state.library_generation();
        match tokio::task::spawn_blocking(move || {
            durable_file::cleanup_orphans(
                &folder,
                &[".upload-stage-", ".upload-reserve-"],
                MAX_UPLOAD_ORPHAN_SCAN_ENTRIES,
            )
        })
        .await
        {
            Ok(Ok(removed)) => {
                state
                    .upload_admission()
                    .mark_cleanup_complete(cleanup_generation);
                if removed > 0 {
                    log::info!("Removed {removed} orphaned browser-upload staging files");
                }
            }
            Ok(Err(error)) => log::warn!("Browser-upload orphan cleanup failed: {error}"),
            Err(error) => log::warn!("Browser-upload orphan cleanup worker failed: {error}"),
        }
    }

    let mut tasks = tokio::task::JoinSet::new();
    for listener in listeners {
        match listener {
            PreparedListener::Plain {
                role,
                listener,
                router,
                handle,
            } => {
                tasks.spawn(async move {
                    let future = async move {
                        if crash_role == Some(role) {
                            panic!("injected listener task crash");
                        }
                        axum_server::from_tcp(listener)
                            .handle(handle)
                            .serve(router.into_make_service_with_connect_info::<SocketAddr>())
                            .await
                    };
                    (
                        role,
                        std::panic::AssertUnwindSafe(future).catch_unwind().await,
                    )
                });
            }
            PreparedListener::Tls {
                listener,
                router,
                handle,
                config,
            } => {
                tasks.spawn(async move {
                    let future = async move {
                        if crash_role == Some(ListenerRole::LanTls) {
                            panic!("injected listener task crash");
                        }
                        axum_server::from_tcp_rustls(listener, config)
                            .handle(handle)
                            .serve(router.into_make_service_with_connect_info::<SocketAddr>())
                            .await
                    };
                    (
                        ListenerRole::LanTls,
                        std::panic::AssertUnwindSafe(future).catch_unwind().await,
                    )
                });
            }
        }
    }

    while let Some(joined) = tasks.join_next().await {
        if stopping.load(Ordering::Acquire) {
            continue;
        }
        let (role, outcome) = match joined {
            Ok(outcome) => outcome,
            Err(error) => {
                log::error!("Control listener join failed: {error}");
                continue;
            }
        };
        let reason = match outcome {
            Ok(Ok(())) => "listener task stopped unexpectedly".to_string(),
            Ok(Err(error)) => format!("listener task failed: {error}"),
            Err(_) => "listener task crashed".to_string(),
        };
        state.update_control_listener(generation, role.slot(), unavailable(reason));
    }
}

fn bind(address: SocketAddr) -> Result<(TcpListener, SocketAddr), String> {
    let listener = TcpListener::bind(address).map_err(|error| error.to_string())?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("set nonblocking: {error}"))?;
    let bound = listener
        .local_addr()
        .map_err(|error| format!("inspect bound address: {error}"))?;
    Ok((listener, bound))
}

fn prepare_tls(
    lan_ip: Option<IpAddr>,
    identity_dir: &FsPath,
    faults: StartFaults,
    runtime: &tokio::runtime::Runtime,
) -> Result<(IpAddr, axum_server::tls_rustls::RustlsConfig), String> {
    let lan_ip = lan_ip.ok_or_else(|| "no routable LAN address was detected".to_string())?;
    let required_sans = vec![
        "localhost".to_string(),
        Ipv4Addr::LOCALHOST.to_string(),
        Ipv6Addr::LOCALHOST.to_string(),
        lan_ip.to_string(),
    ];
    let identity = tls_identity::load_or_create(identity_dir, &required_sans, faults.identity)?;
    if faults.tls_config {
        return Err("injected TLS configuration fault".to_string());
    }
    let config = rustls_config_from_der(runtime, identity.cert_chain, identity.key_der)?;
    Ok((lan_ip, config))
}

/// `RustlsConfig::from_der` uses `tokio::spawn_blocking`, so polling it without
/// an entered Tokio runtime panics. Run the validation on one short, joined
/// scope thread entered into the runtime that the owned server will later use.
/// The extra thread also makes this synchronous API safe when a caller already
/// happens to be inside a different Tokio runtime.
fn rustls_config_from_der(
    runtime: &tokio::runtime::Runtime,
    cert_chain: Vec<Vec<u8>>,
    key_der: Vec<u8>,
) -> Result<axum_server::tls_rustls::RustlsConfig, String> {
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("cos-tls-config".to_string())
            .spawn_scoped(scope, move || {
                runtime.block_on(axum_server::tls_rustls::RustlsConfig::from_der(
                    cert_chain, key_der,
                ))
            })
            .map_err(|error| format!("TLS validation worker creation failed: {error}"))?;
        worker
            .join()
            .map_err(|_| "TLS validation worker panicked".to_string())?
            .map_err(|error| format!("TLS certificate/private-key validation failed: {error}"))
    })
}

fn unavailable(reason: impl AsRef<str>) -> ControlListenerStatus {
    ControlListenerStatus::Unavailable {
        reason: bounded_reason(reason.as_ref()),
    }
}

fn bounded_reason(reason: &str) -> String {
    let mut bounded = String::with_capacity(reason.len().min(MAX_LISTENER_REASON_CHARS));
    for (index, character) in reason.chars().enumerate() {
        if index >= MAX_LISTENER_REASON_CHARS {
            bounded.push('\u{2026}');
            break;
        }
        bounded.push(if character.is_control() {
            ' '
        } else {
            character
        });
    }
    bounded
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

/// Build the exact production router independently of listener ownership.
/// Keeping this seam small lets the transport test bind an ephemeral port and
/// exercise the real authentication, WebSocket upgrade, and action ingress.
fn control_router(
    state: Arc<WebState>,
    session: Arc<ControlSession>,
    role: ListenerRole,
) -> Router {
    let router_state = ControlRouterState {
        web: state,
        session,
        role,
    };
    Router::new()
        .route("/ws", get(ws_handler))
        .route("/thumb/:filename", get(thumb_handler))
        .route("/preview/:filename/:index", get(preview_handler))
        .route("/library-index", get(library_index_handler))
        .route("/qr.svg", get(qr_handler))
        .route(
            "/controller-profile",
            post(controller_profile_handler).layer(DefaultBodyLimit::max(
                crate::controller_profile::CONTROLLER_PROFILE_ACTION_MAX_BYTES,
            )),
        )
        .route(
            "/upload",
            post(upload_handler).layer(DefaultBodyLimit::max(MAX_VISUAL_UPLOAD_BYTES as usize)),
        )
        .route("/delete", post(delete_handler))
        .fallback(get(static_files::serve))
        .layer(middleware::from_fn_with_state(router_state.clone(), auth))
        // Added last so even authentication refusals receive the no-store and
        // browser-hardening headers.
        .layer(middleware::from_fn(security_headers))
        .with_state(router_state)
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
fn same_origin(headers: &HeaderMap, role: ListenerRole) -> bool {
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
    let scheme = if role.cookie_is_secure() {
        "https"
    } else {
        "http"
    };
    origin == format!("{scheme}://{host}")
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
    State(state): State<ControlRouterState>,
    req: Request,
    next: Next,
) -> Response {
    if state.role.is_loopback() && !addr.ip().is_loopback() {
        log::warn!(
            "Rejected non-loopback client on {} listener: {}",
            state.role.label(),
            addr.ip()
        );
        return forbidden(
            "<h3>collide-o-scope</h3><p>This plaintext listener accepts loopback clients only.</p>",
        );
    }
    let token = &state.session.token.0;
    let cookie_name = state.role.cookie_name();

    let cookie_ok = req
        .headers()
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .map(|cookies| {
            cookies.split(';').any(|c| {
                c.trim()
                    .strip_prefix(cookie_name)
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

    if requires_same_origin(req.method(), req.uri().path())
        && !same_origin(req.headers(), state.role)
    {
        log::warn!("Rejected cross-origin control mutation from {}", addr.ip());
        return forbidden("<h3>collide-o-scope</h3><p>Cross-origin control request denied.</p>");
    }

    let mut response = next.run(req).await;
    if query_ok && !cookie_ok {
        let secure = if state.role.cookie_is_secure() {
            "; Secure"
        } else {
            ""
        };
        if let Ok(cookie) = HeaderValue::from_str(&format!(
            "{cookie_name}={token}; Path=/; HttpOnly; SameSite=Strict{secure}"
        )) {
            response.headers_mut().append(header::SET_COOKIE, cookie);
        }
    }
    response
}

async fn security_headers(req: Request, next: Next) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();
    headers.insert(
        HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static(
            "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; media-src 'none'; worker-src 'none'",
        ),
    );
    headers.insert(
        HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY"),
    );
    headers.insert(
        HeaderName::from_static("referrer-policy"),
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    headers.insert(header::PRAGMA, HeaderValue::from_static("no-cache"));
    response
}

/// QR code (SVG) of the remote URL, rendered on demand.
async fn qr_handler(State(state): State<Arc<WebState>>) -> Response {
    let info = state.control_server_info();
    let Some(url) = info.lan_url.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "LAN HTTPS control is unavailable; no QR code was published.",
        )
            .into_response();
    };
    let url = url.expose_to_local_ui();

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
    let request = match crate::controller_profile::ControllerProfileAction::decode_json_bytes(&body)
    {
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
            let envelope = state.envelope_web_action(action);
            if !valid_action(envelope.payload(), 0) {
                let acknowledgement = state.terminal_action_identity_with_ack(
                    envelope.identity(),
                    crate::action_correlation::ActionDisposition::Refused,
                );
                return (StatusCode::BAD_REQUEST, axum::Json(acknowledgement)).into_response();
            }
            let (outcome, acknowledgement) =
                state.enqueue_enveloped_action_with_ack(envelope).await;
            let status = match outcome {
                EnqueueOutcome::Added | EnqueueOutcome::Coalesced => StatusCode::ACCEPTED,
                EnqueueOutcome::Dropped => StatusCode::TOO_MANY_REQUESTS,
            };
            (status, axum::Json(acknowledgement)).into_response()
        }
    }
}

#[derive(serde::Deserialize)]
struct UploadQuery {
    name: String,
}

#[derive(serde::Serialize)]
struct LibraryMutationAck {
    #[serde(flatten)]
    action: ActionIngressAck,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn library_mutation_response(
    status: StatusCode,
    action: ActionIngressAck,
    name: Option<String>,
    error: Option<String>,
) -> Response {
    (
        status,
        axum::Json(LibraryMutationAck {
            action,
            name,
            error,
        }),
    )
        .into_response()
}

/// Exact-once terminal guard for authenticated durable library mutations. A
/// handler cancellation or future early-return after identity minting is an
/// explicit Refused action rather than an orphaned engine sequence.
struct LibraryMutationTerminalGuard {
    state: Arc<WebState>,
    identity: Option<crate::action_correlation::ActionIdentity>,
}

impl LibraryMutationTerminalGuard {
    fn browser(state: Arc<WebState>) -> Self {
        let identity = state
            .action_sequencer()
            .envelope(crate::action_correlation::ActionSourceClass::Browser, ())
            .identity();
        Self {
            state,
            identity: Some(identity),
        }
    }

    fn finish(
        mut self,
        disposition: crate::action_correlation::ActionDisposition,
    ) -> ActionIngressAck {
        let identity = self
            .identity
            .take()
            .expect("library mutation finishes once");
        self.state
            .terminal_action_identity_with_ack(identity, disposition)
    }
}

impl Drop for LibraryMutationTerminalGuard {
    fn drop(&mut self) {
        if let Some(identity) = self.identity.take() {
            self.state.record_terminal_action_identity(
                identity,
                crate::action_correlation::ActionDisposition::Refused,
            );
        }
    }
}

fn finish_blocking_library_mutation<T, E>(
    mutation: LibraryMutationTerminalGuard,
    operation: impl FnOnce() -> Result<T, E>,
) -> (Result<T, E>, ActionIngressAck) {
    let result = operation();
    let disposition = if result.is_ok() {
        crate::action_correlation::ActionDisposition::Coalesced
    } else {
        crate::action_correlation::ActionDisposition::Refused
    };
    let acknowledgement = mutation.finish(disposition);
    (result, acknowledgement)
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

fn reserve_upload_destination(
    folder: &std::path::Path,
    original_name: &str,
    stem: &str,
    ext: &str,
) -> Result<(String, PathBuf, RemovePathOnDrop), String> {
    for counter in 0..MAX_UPLOAD_DESTINATION_ATTEMPTS {
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
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&reservation_path)
        {
            Ok(file) => {
                drop(file);
                let reservation = RemovePathOnDrop::new(reservation_path);
                // Cover files created by another process between our first
                // existence check and reservation.
                if final_path.exists() {
                    drop(reservation);
                    continue;
                }
                return Ok((candidate, final_path, reservation));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("reserve destination: {error}")),
        }
    }
    Err(format!(
        "could not reserve a destination after {MAX_UPLOAD_DESTINATION_ATTEMPTS} names"
    ))
}

#[derive(Debug)]
struct RemovePathOnDrop {
    path: Option<PathBuf>,
}

impl RemovePathOnDrop {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn remove(mut self) -> std::io::Result<()> {
        let Some(path) = self.path.take() else {
            return Ok(());
        };
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

impl Drop for RemovePathOnDrop {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Field order is deliberate: if a detached blocking probe finishes after
/// its HTTP future was cancelled, the open file closes before the publication
/// guard retries staging cleanup.
struct SyncedUploadStaging {
    file: std::fs::File,
    publication: StagedPublication,
}

/// Whether a completed upload body is short of what the client declared.
///
/// A partial body that ends cleanly is otherwise indistinguishable from a
/// whole one: it is renamed into the library, counted as a clip, and given a
/// thumbnail attempt, so the operator sees a real-looking entry that no
/// decoder can open. `Content-Length` is already parsed to enforce the upper
/// bound, and comparing it against what actually arrived costs nothing.
///
/// A client that declares no length gets no verdict here -- there is nothing
/// to compare against, and refusing every chunked upload would be worse than
/// the defect.
fn upload_is_truncated(declared_length: Option<u64>, written: u64) -> bool {
    declared_length.is_some_and(|declared| written < declared)
}

#[derive(Debug, Clone, Copy)]
struct UploadDeadline {
    started: Instant,
    absolute: Duration,
    idle: Duration,
}

impl UploadDeadline {
    fn production() -> Self {
        Self {
            started: Instant::now(),
            absolute: UPLOAD_ABSOLUTE_TIMEOUT,
            idle: UPLOAD_CHUNK_IDLE_TIMEOUT,
        }
    }

    fn remaining_at(self, now: Instant) -> Option<Duration> {
        self.absolute
            .checked_sub(now.checked_duration_since(self.started)?)
    }

    fn remaining(self) -> Option<Duration> {
        self.remaining_at(Instant::now())
    }

    fn next_chunk_wait(self) -> Option<Duration> {
        self.next_chunk_wait_at(Instant::now())
    }

    fn next_chunk_wait_at(self, now: Instant) -> Option<Duration> {
        self.remaining_at(now)
            .map(|remaining| remaining.min(self.idle))
    }
}

fn upload_is_current(state: &WebState, server_generation: u64, library_generation: u64) -> bool {
    state.control_server_generation_accepts_upload(server_generation)
        && state.library_generation_is_current(library_generation)
}

fn cancelled_upload_response() -> Response {
    (
        StatusCode::CONFLICT,
        "upload cancelled because the server or library changed",
    )
        .into_response()
}

fn upload_timeout_response(kind: &str) -> Response {
    (
        StatusCode::REQUEST_TIMEOUT,
        format!("upload {kind} deadline expired"),
    )
        .into_response()
}

fn probe_uploaded_media(
    staging: &FsPath,
    extension: &str,
    media_policy: &crate::media_safety::MediaSafetyPolicy,
) -> Result<(), String> {
    if crate::audio::is_supported_audio_extension(extension) {
        crate::audio::clip::AudioClip::open(staging).map(|_| ())
    } else if crate::layers::STILL_IMAGE_EXTENSIONS
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    {
        crate::video::probe_still_image_dimensions_with_media_policy(
            staging,
            media_policy,
            crate::media_safety::MediaDeviceLimits::none(),
        )
        .map(|_| ())
    } else {
        crate::video::VideoDecoder::probe_dimensions_with_media_policy(
            &staging.to_string_lossy(),
            media_policy,
            crate::media_safety::MediaDeviceLimits::none(),
        )
        .map(|_| ())
    }
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
    let deadline = UploadDeadline::production();
    let Some(validated) = validate_library_filename(&query.name) else {
        return (StatusCode::BAD_REQUEST, "unsupported or unsafe filename").into_response();
    };
    let ValidatedLibraryFilename {
        name,
        stem,
        extension: ext,
    } = validated;

    let declared_length = match headers.get(header::CONTENT_LENGTH) {
        Some(value) => match value
            .to_str()
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
        {
            Some(length) => Some(length),
            None => return (StatusCode::BAD_REQUEST, "invalid Content-Length").into_response(),
        },
        None => None,
    };
    if declared_length.is_some_and(|length| exceeds_upload_limit(&ext, length)) {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "upload exceeds the {}-byte limit for .{ext}",
                upload_limit_for_extension(&ext)
            ),
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
    let server_generation = state.control_server_generation();
    let library_generation = state.library_generation();
    if !upload_is_current(&state, server_generation, library_generation) {
        return cancelled_upload_response();
    }

    let admission = state.upload_admission();
    match admission.begin_cleanup(library_generation) {
        Ok(None) => {}
        Ok(Some(cleanup_lease)) => {
            let cleanup_folder = folder.clone();
            let cleanup_job = tokio::task::spawn_blocking(move || {
                let result = durable_file::cleanup_orphans(
                    &cleanup_folder,
                    &[".upload-stage-", ".upload-reserve-"],
                    MAX_UPLOAD_ORPHAN_SCAN_ENTRIES,
                );
                if result.is_ok() {
                    cleanup_lease.complete();
                }
                result
            });
            let Some(remaining) = deadline.remaining() else {
                return upload_timeout_response("absolute");
            };
            match tokio::time::timeout(remaining, cleanup_job).await {
                Ok(Ok(Ok(_))) => {}
                Ok(Ok(Err(error))) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("upload orphan cleanup failed: {error}"),
                    )
                        .into_response();
                }
                Ok(Err(error)) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("upload orphan cleanup worker failed: {error}"),
                    )
                        .into_response();
                }
                Err(_) => return upload_timeout_response("absolute"),
            }
        }
        Err(AdmissionError::CleanupBusy) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                "library upload cleanup or prior-generation cancellation is still in progress",
            )
                .into_response();
        }
        Err(_) => unreachable!("begin_cleanup only returns CleanupBusy"),
    }

    // A chunked request reserves its complete per-file ceiling up front. That
    // makes aggregate and disk admission truthful even before the first byte
    // arrives; the lease releases automatically on every disconnect/error.
    let reserved_bytes = declared_length
        .unwrap_or_else(|| upload_limit_for_extension(&ext))
        .max(1);
    let disk_folder = folder.clone();
    let disk_job = tokio::task::spawn_blocking(move || durable_file::available_space(&disk_folder));
    let Some(remaining) = deadline.remaining() else {
        return upload_timeout_response("absolute");
    };
    let available_bytes = match tokio::time::timeout(remaining, disk_job).await {
        Ok(Ok(Ok(bytes))) => bytes,
        Ok(Ok(Err(error))) => {
            return (
                StatusCode::INSUFFICIENT_STORAGE,
                format!("cannot inspect library disk headroom: {error}"),
            )
                .into_response();
        }
        Ok(Err(error)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("disk-headroom worker failed: {error}"),
            )
                .into_response();
        }
        Err(_) => return upload_timeout_response("absolute"),
    };
    let _lease = match admission.try_reserve(
        reserved_bytes,
        available_bytes,
        AdmissionLimits {
            max_concurrent: MAX_CONCURRENT_UPLOADS,
            max_reserved_bytes: MAX_UPLOAD_AGGREGATE_BYTES,
            min_free_after_reservations: MIN_UPLOAD_DISK_HEADROOM_BYTES,
        },
    ) {
        Ok(lease) => lease,
        Err(AdmissionError::Concurrency) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                "at most two browser uploads may run concurrently",
            )
                .into_response();
        }
        Err(AdmissionError::AggregateBytes) => {
            return (
                StatusCode::INSUFFICIENT_STORAGE,
                "aggregate upload-byte reservations are full",
            )
                .into_response();
        }
        Err(AdmissionError::DiskHeadroom) => {
            return (
                StatusCode::INSUFFICIENT_STORAGE,
                "upload would consume the required library disk headroom",
            )
                .into_response();
        }
        Err(AdmissionError::CleanupBusy) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                "library upload cleanup is still in progress",
            )
                .into_response();
        }
    };

    // Atomically reserve a collision-free destination. This prevents two
    // simultaneous same-name uploads from sharing either output or temp data.
    let Some(wait) = deadline.next_chunk_wait() else {
        return upload_timeout_response("absolute");
    };
    let reservation_folder = folder.clone();
    let reservation_name = name.clone();
    let reservation_stem = stem.clone();
    let reservation_extension = ext.clone();
    let reservation_job = tokio::task::spawn_blocking(move || {
        reserve_upload_destination(
            &reservation_folder,
            &reservation_name,
            &reservation_stem,
            &reservation_extension,
        )
    });
    let (final_name, final_path, reservation) =
        match tokio::time::timeout(wait, reservation_job).await {
            Ok(Ok(Ok(reservation))) => reservation,
            Ok(Ok(Err(error))) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, error).into_response();
            }
            Ok(Err(error)) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("upload destination worker failed: {error}"),
                )
                    .into_response();
            }
            Err(_) => return upload_timeout_response("idle"),
        };
    let staging_destination = final_path.clone();
    let staging_job = tokio::task::spawn_blocking(move || {
        StagedPublication::create(&staging_destination, "upload-stage")
            .map(|(publication, file)| SyncedUploadStaging { file, publication })
    });
    let Some(remaining) = deadline.remaining() else {
        return upload_timeout_response("absolute");
    };
    let staging = match tokio::time::timeout(remaining, staging_job).await {
        Ok(Ok(Ok(staging))) => staging,
        Ok(Ok(Err(error))) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("create upload staging file: {error}"),
            )
                .into_response();
        }
        Ok(Err(error)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("upload staging worker failed: {error}"),
            )
                .into_response();
        }
        Err(_) => return upload_timeout_response("absolute"),
    };
    let SyncedUploadStaging {
        file: std_file,
        publication,
    } = staging;
    let mut file = tokio::fs::File::from_std(std_file);

    let mut stream = body.into_data_stream();
    let mut written: u64 = 0;
    loop {
        if !upload_is_current(&state, server_generation, library_generation) {
            return cancelled_upload_response();
        }
        let Some(absolute_remaining) = deadline.remaining() else {
            return upload_timeout_response("absolute");
        };
        let wait = absolute_remaining.min(deadline.idle);
        let chunk = match tokio::time::timeout(wait, stream.next()).await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(_) if absolute_remaining <= deadline.idle => {
                return upload_timeout_response("absolute");
            }
            Err(_) => return upload_timeout_response("idle"),
        };
        match chunk {
            Ok(bytes) => {
                let Some(next_written) = written.checked_add(bytes.len() as u64) else {
                    return (
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "upload byte count overflowed",
                    )
                        .into_response();
                };
                if exceeds_upload_limit(&ext, next_written)
                    || declared_length.is_some_and(|declared| next_written > declared)
                {
                    return (
                        StatusCode::PAYLOAD_TOO_LARGE,
                        format!(
                            "upload exceeds its declared or {}-byte format limit",
                            upload_limit_for_extension(&ext)
                        ),
                    )
                        .into_response();
                }
                let Some(absolute_remaining) = deadline.remaining() else {
                    return upload_timeout_response("absolute");
                };
                let wait = absolute_remaining.min(deadline.idle);
                match tokio::time::timeout(wait, file.write_all(&bytes)).await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("write upload staging file: {error}"),
                        )
                            .into_response();
                    }
                    Err(_) if absolute_remaining <= deadline.idle => {
                        return upload_timeout_response("absolute");
                    }
                    Err(_) => return upload_timeout_response("idle"),
                }
                written = next_written;
            }
            Err(error) => {
                return (StatusCode::BAD_REQUEST, format!("upload stream: {error}"))
                    .into_response();
            }
        }
    }
    if written == 0 {
        return (StatusCode::BAD_REQUEST, "empty upload").into_response();
    }
    if upload_is_truncated(declared_length, written) {
        let declared = declared_length.unwrap_or_default();
        log::warn!(
            "Rejected truncated upload {name}: {written} of {declared} declared bytes arrived"
        );
        return (
            StatusCode::BAD_REQUEST,
            format!("truncated upload: {written} of {declared} declared bytes arrived"),
        )
            .into_response();
    }
    if !upload_is_current(&state, server_generation, library_generation) {
        return cancelled_upload_response();
    }
    let Some(absolute_remaining) = deadline.remaining() else {
        return upload_timeout_response("absolute");
    };
    let wait = absolute_remaining.min(deadline.idle);
    match tokio::time::timeout(wait, file.flush()).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("flush upload staging file: {error}"),
            )
                .into_response();
        }
        Err(_) if absolute_remaining <= deadline.idle => {
            return upload_timeout_response("absolute");
        }
        Err(_) => return upload_timeout_response("idle"),
    }

    let Some(remaining) = deadline.remaining() else {
        return upload_timeout_response("absolute");
    };
    let std_file = match tokio::time::timeout(remaining, file.into_std()).await {
        Ok(file) => file,
        Err(_) => return upload_timeout_response("absolute"),
    };
    let probe_extension = ext.clone();
    let media_policy = state.upload_media_safety_policy();
    let probe_job = tokio::task::spawn_blocking(move || -> Result<SyncedUploadStaging, String> {
        let staging = SyncedUploadStaging {
            file: std_file,
            publication,
        };
        staging
            .file
            .sync_all()
            .map_err(|error| format!("sync upload staging file: {error}"))?;
        probe_uploaded_media(
            staging.publication.staging_path(),
            &probe_extension,
            &media_policy,
        )?;
        Ok(staging)
    });
    let Some(remaining) = deadline.remaining() else {
        return upload_timeout_response("absolute");
    };
    let staging = match tokio::time::timeout(remaining, probe_job).await {
        Ok(Ok(Ok(staging))) => staging,
        Ok(Ok(Err(error))) => return (StatusCode::UNSUPPORTED_MEDIA_TYPE, error).into_response(),
        Ok(Err(error)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("media-safety probe worker failed: {error}"),
            )
                .into_response();
        }
        Err(_) => return upload_timeout_response("absolute"),
    };
    let SyncedUploadStaging {
        file: std_file,
        publication,
    } = staging;

    // The bounded body and media probe have now admitted one typed durable
    // library mutation. Mint before the first final-path publication; staging
    // transport failures above this point have no application action identity.
    let mutation = LibraryMutationTerminalGuard::browser(state.clone());
    let commit_state = state.clone();
    let commit_job = tokio::task::spawn_blocking(move || {
        finish_blocking_library_mutation(mutation, || {
            commit_state.with_upload_publication_gate(|| {
                if deadline.remaining().is_none() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "upload absolute deadline expired before publication",
                    ));
                }
                if !upload_is_current(&commit_state, server_generation, library_generation) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "server or library generation changed before publication",
                    ));
                }
                publication.commit_presynced(std_file, PublishMode::NoReplace)
            })
        })
    });
    let (commit, acknowledgement) = match commit_job.await {
        Ok(result) => result,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "upload publication worker failed after its identity was terminalized: {error}"
                ),
            )
                .into_response();
        }
    };
    if let Err(error) = commit {
        let (status, message) = match error.kind() {
            std::io::ErrorKind::AlreadyExists => (
                StatusCode::CONFLICT,
                "the final upload name was claimed before publication".to_string(),
            ),
            std::io::ErrorKind::Interrupted => (
                StatusCode::CONFLICT,
                "upload cancelled because the server or library changed".to_string(),
            ),
            std::io::ErrorKind::TimedOut => (
                StatusCode::REQUEST_TIMEOUT,
                "upload absolute deadline expired".to_string(),
            ),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("publish upload: {error}"),
            ),
        };
        return library_mutation_response(status, acknowledgement, None, Some(message));
    }
    // The blocking owner terminalized the exact durable result before cleanup
    // or the independent rescan enqueue, neither of which may relabel it.
    match tokio::task::spawn_blocking(move || reservation.remove()).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => log::warn!("Upload reservation cleanup failed: {error}"),
        Err(error) => log::warn!("Upload reservation cleanup worker failed: {error}"),
    }

    log::info!("Uploaded clip: {final_name} ({written} bytes)");
    if state.library_generation_is_current(library_generation) {
        let _ = state.enqueue_action(WebAction::RescanLibrary).await;
    }

    library_mutation_response(StatusCode::OK, acknowledgement, Some(final_name), None)
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

    let mutation = LibraryMutationTerminalGuard::browser(state.clone());
    match tokio::task::spawn_blocking(move || {
        finish_blocking_library_mutation(mutation, || trash::delete(&path))
    })
    .await
    {
        Ok((Ok(()), acknowledgement)) => {
            state.remove_library_media_cache_entry(&name);
            log::info!("Clip moved to Recycle Bin: {name}");
            let _ = state.enqueue_action(WebAction::RescanLibrary).await;
            library_mutation_response(StatusCode::OK, acknowledgement, Some(name), None)
        }
        Ok((Err(error), acknowledgement)) => library_mutation_response(
            StatusCode::CONFLICT,
            acknowledgement,
            None,
            Some(format!("cannot remove (in use by a layer?): {error}")),
        ),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("delete worker failed after its identity was terminalized: {error}"),
        )
            .into_response(),
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

#[derive(serde::Deserialize)]
struct LibraryIndexQuery {
    revision: Option<u64>,
    offset: Option<u32>,
    limit: Option<usize>,
    query: Option<String>,
    kind: Option<String>,
}

/// Authenticated, revision-checked and explicitly bounded library paging.
/// Searching a large (but capped) immutable index is kept off the async server
/// executor so neither Main nor unrelated control requests inherit the scan.
async fn library_index_handler(
    Query(query): Query<LibraryIndexQuery>,
    State(state): State<Arc<WebState>>,
) -> Response {
    let kind = match query.kind.as_deref().unwrap_or("visual") {
        "all" => None,
        value => match crate::library_index::LibraryEntryKind::parse(value) {
            Some(kind) => Some(kind),
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    "library kind must be visual, audio, or all",
                )
                    .into_response();
            }
        },
    };
    let request = match crate::library_index::LibraryPageRequest::new(
        query.revision,
        query.offset.unwrap_or(0),
        query
            .limit
            .unwrap_or(crate::library_index::LIBRARY_INDEX_DEFAULT_PAGE_SIZE),
        query.query.as_deref(),
        kind,
    ) {
        Ok(request) => request,
        Err(error) => return (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
    };
    let permit = match library_page_gate().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                "library page/search capacity is busy",
            )
                .into_response();
        }
    };
    let index = state.library_index();
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        index.page(request)
    })
    .await
    {
        Ok(Ok(page)) => match serde_json::to_vec(&page) {
            Ok(body) => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json; charset=utf-8")],
                body,
            )
                .into_response(),
            Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        Ok(Err(error @ crate::library_index::LibraryPageError::RevisionMismatch { .. })) => {
            (StatusCode::CONFLICT, error.to_string()).into_response()
        }
        Ok(Err(error)) => (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
        Err(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "library page worker unavailable",
        )
            .into_response(),
    }
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
            // The B6 Filter Avalanche's predictor law: closed and
            // append-only, snake_case tokens exactly as the enum serializes.
            "avalanche_axis" => matches!(value.as_str(), Some("sub" | "up" | "average")),
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
        // The B5 codec mosh's eleven continuous wire params plus its one
        // discrete recycle law.
        "mosh_amount"
        | "mosh_key_removal"
        | "mosh_hold"
        | "mosh_drop"
        | "mosh_shuffle"
        | "mosh_rate"
        | "mosh_bitrate_starve"
        | "mosh_resync"
        | "mosh_wipe"
        | "mosh_smear"
        | "mosh_trail" => number_in(value, 0.0, 1.0),
        "mosh_recycle" => value.is_boolean(),
        // The B14 sync latch's four continuous wire params plus the switch.
        "sync_amount" | "sync_rate" | "sync_spread" => number_in(value, 0.0, 1.0),
        "sync_bias" => number_in(value, -1.0, 1.0),
        "sync_latched" => value.is_boolean(),
        "slitscan" | "slit_axis" => number_in(value, 0.0, 1.0),
        "slit_angle" | "loom_angle" => number_in(value, -180.0, 180.0),
        "slit_map" => matches!(
            value.as_str(),
            Some("ramp" | "brightness" | "radial" | "tbc_ramp" | "sweep")
        ),
        "slit_interp" => value.is_boolean(),
        "key_mode" => integer_in(value, 0, 4),
        "key_threshold"
        | "long_exposure_amount"
        | "loom_amount"
        | "loom_depth"
        | "atlas_amount"
        | "atlas_collision"
        | "garden_amount"
        | "garden_threshold"
        | "garden_decay" => number_in(value, 0.0, 1.0),
        "key_softness" | "garden_softness" => number_in(value, 0.0, 0.5),
        "key_history" => integer_in(value, 1, 23),
        "long_exposure_frames" => integer_in(value, 2, 24),
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
        WebAction::SetOutputDisplay { display_id, .. } => {
            display_id.is_empty()
                || (display_id.len() <= 64
                    && display_id
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'))
        }
        WebAction::SetSpoutResolution { resolution } => {
            crate::spout_out::SpoutResolutionMode::from_key(resolution).is_some()
        }
        WebAction::SetLayerParam {
            layer_id,
            param,
            value,
            ..
        } => {
            valid_optional_layer_id(layer_id)
                && valid_identifier(param, 64)
                && valid_json_value(value)
                && match param.as_str() {
                    "blend_mode" => value
                        .as_str()
                        .and_then(crate::layers::BlendMode::from_key)
                        .is_some(),
                    // P4c delivery policy: a closed token vocabulary; an
                    // unknown token is a gate refusal, never a default.
                    "delivery" => value
                        .as_str()
                        .and_then(crate::video::PlanarDeliveryPolicy::from_key)
                        .is_some(),
                    "mosh_send" => number_in(value, 0.0, 1.0),
                    _ => true,
                }
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
        WebAction::SetLfo {
            index,
            param,
            value,
        } => {
            *index < crate::modulation::NUM_LFOS
                && match param.as_str() {
                    "shape" => value.as_str().is_some_and(|shape| {
                        matches!(
                            shape,
                            "sine" | "triangle" | "saw" | "square" | "sample_hold"
                        )
                    }),
                    // The engine retains the legacy clamp/wrap behavior for
                    // finite controller values; non-finite numbers never
                    // enter the queue.
                    "beats" | "phase" => value.as_f64().is_some_and(f64::is_finite),
                    "seed" => value
                        .as_u64()
                        .and_then(|seed| u32::try_from(seed).ok())
                        .is_some(),
                    _ => false,
                }
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
        // The bank's two write actions carry the same revision barriers a
        // Morph capture does, and address a fixed eight slots. An out-of-range
        // slot is refused rather than clamped onto a neighbour.
        WebAction::SnapshotBankSave {
            slot,
            stack_revision,
            composition_revision,
        }
        | WebAction::SnapshotBankRecall {
            slot,
            stack_revision,
            composition_revision,
        } => {
            *slot < crate::morph::SNAPSHOT_BANK_SLOTS
                && stack_revision.is_none_or(|revision| revision != 0)
                && composition_revision.is_none_or(|revision| revision != 0)
        }
        WebAction::SnapshotBankClear { slot } => *slot < crate::morph::SNAPSHOT_BANK_SLOTS,
        WebAction::SetSnapshotBankGlide { beats } => {
            beats.is_finite()
                && *beats >= 0.0
                && *beats <= crate::morph::SNAPSHOT_BANK_MAX_GLIDE_BEATS
        }
        WebAction::SetMorphLaw { law } => matches!(law.as_str(), "linear" | "equal_power"),
        WebAction::ResetGroup { group } => valid_identifier(group, 32),
        WebAction::StartProgramRecording { .. }
        | WebAction::FinishProgramRecording
        | WebAction::CancelProgramRecording
        | WebAction::SetStageHealthHud { .. }
        | WebAction::SetMonitorBay { .. }
        | WebAction::MonitorWatch { .. } => true,
        WebAction::SetMonitorProbe { probe } => {
            crate::monitor_bay::MonitorProbe::try_from_str(probe).is_some()
        }
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
    // Subscribe first so Main can observe a real receiver, then request one
    // fresh complete generation. This avoids maintaining or serializing a
    // 30 Hz snapshot while no browser exists.
    let mut rx = state.tx.subscribe();
    let observed_generation = state.request_snapshot();
    let fresh = tokio::time::timeout(Duration::from_millis(250), async {
        loop {
            if state.full_snapshot_generation() > observed_generation {
                let full = state.last_full_snapshot();
                if !full.is_empty() {
                    // Consume whatever newest-only value accompanied the full
                    // publication. It may already be a later live-domain
                    // sample; skipping that one is safer than applying any
                    // delta before this socket has received its full base.
                    let _ = rx.borrow_and_update();
                    return Some(full);
                }
            }
            if rx.changed().await.is_err() {
                return None;
            }
        }
    })
    .await
    .ok()
    .flatten();
    let init_msg = if let Some(message) = fresh {
        message
    } else {
        // A stopped/blocked renderer still gets one bounded startup snapshot;
        // ordinary production connects are fulfilled by Main on its next
        // accepted frame and never take this fallback.
        let current = state.app.read().await;
        Arc::new(serde_json::to_string(&*current).unwrap_or_else(|_| "{}".to_string()))
    };
    if sender
        .send(Message::Text(init_msg.as_ref().clone()))
        .await
        .is_err()
    {
        return;
    }
    // Action dispositions are a separate bounded reliable stream. They are
    // never coalesced with newest-only state and therefore cannot disappear
    // merely because telemetry publishes again.
    let (ack_tx, mut ack_rx) = tokio::sync::mpsc::channel::<String>(64);

    // Forward newest-only state generations to this client. `watch` retains
    // exactly one payload; a slow socket skips obsolete intermediates.
    let mut send_task = tokio::spawn(async move {
        loop {
            let message = tokio::select! {
                changed = rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    rx.borrow_and_update().as_ref().clone()
                }
                ack = ack_rx.recv() => {
                    let Some(ack) = ack else {
                        break;
                    };
                    ack
                }
            };
            if sender.send(Message::Text(message)).await.is_err() {
                break;
            }
        }
    });

    // Receive actions from this client
    let state_clone = state.clone();
    // If a tab disappears mid-drag, publish an ordered End on disconnect so
    // Main records the final authored value (or an exact no-op) instead of
    // remaining permanently blocked by an orphaned gesture.
    let socket_ack_tx = ack_tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                // Try to parse as a WebAction
                match super::action_wire::parse_web_action::<WebAction>(text.as_bytes()) {
                    Ok(action) => {
                        // Mint at the typed parse edge, before semantic/gesture
                        // validation or any socket-local mutation. Every valid
                        // WebAction therefore gets one engine sequence even
                        // when it never enters Main's bounded queue.
                        let envelope = state_clone.envelope_web_action(action);
                        let terminal =
                            state_clone.action_ingress_terminal_guard(envelope.identity());
                        if !valid_action(envelope.payload(), 0) {
                            send_guarded_socket_refusal(
                                &socket_ack_tx,
                                terminal,
                                "invalid_payload",
                            )
                            .await;
                            log::warn!("Rejected invalid WebAction payload");
                            continue;
                        }
                        match envelope.payload() {
                            WebAction::GyroStream { enabled } => {
                                let identity = envelope.identity();
                                let changed = state_clone.set_gyro_stream(client_id, *enabled);
                                debug_assert_eq!(terminal.identity(), identity);
                                let acknowledgement = terminal.terminalize(if changed {
                                    crate::action_correlation::ActionDisposition::Coalesced
                                } else {
                                    crate::action_correlation::ActionDisposition::Refused
                                });
                                send_socket_ack(&socket_ack_tx, acknowledgement).await;
                            }
                            WebAction::MonitorWatch { enabled } => {
                                let identity = envelope.identity();
                                let changed = state_clone.set_monitor_watch(client_id, *enabled);
                                debug_assert_eq!(terminal.identity(), identity);
                                let acknowledgement = terminal.terminalize(if changed {
                                    crate::action_correlation::ActionDisposition::Coalesced
                                } else {
                                    crate::action_correlation::ActionDisposition::Refused
                                });
                                send_socket_ack(&socket_ack_tx, acknowledgement).await;
                            }
                            WebAction::Gyro { .. } => {
                                let outcome = enqueue_socket_action(
                                    &state_clone,
                                    &socket_ack_tx,
                                    envelope,
                                    terminal,
                                )
                                .await;
                                if outcome != EnqueueOutcome::Dropped {
                                    state_clone.note_gyro_sample(client_id);
                                }
                            }
                            WebAction::BeginHistoryGesture { gesture_id } => {
                                let gesture_id = *gesture_id;
                                match state_clone
                                    .enqueue_browser_history_begin_with_ack(
                                        client_id, gesture_id, envelope, terminal,
                                    )
                                    .await
                                {
                                    Ok((_, acknowledgement)) => {
                                        send_socket_ack(&socket_ack_tx, acknowledgement).await;
                                    }
                                    Err(acknowledgement) => {
                                        send_socket_ack_with_reason(
                                            &socket_ack_tx,
                                            acknowledgement,
                                            "gesture_boundary",
                                        )
                                        .await;
                                        log::warn!("Rejected nested browser history gesture");
                                    }
                                }
                            }
                            WebAction::EndHistoryGesture { gesture_id }
                            | WebAction::CancelHistoryGesture { gesture_id } => {
                                let gesture_id = *gesture_id;
                                let cancel = matches!(
                                    envelope.payload(),
                                    WebAction::CancelHistoryGesture { .. }
                                );
                                match state_clone
                                    .enqueue_browser_history_finish_with_ack(
                                        client_id, gesture_id, cancel, envelope, terminal,
                                    )
                                    .await
                                {
                                    Ok((_, acknowledgement)) => {
                                        send_socket_ack(&socket_ack_tx, acknowledgement).await;
                                    }
                                    Err(acknowledgement) => {
                                        send_socket_ack_with_reason(
                                            &socket_ack_tx,
                                            acknowledgement,
                                            "gesture_boundary",
                                        )
                                        .await;
                                        log::warn!(
                                            "Rejected mismatched or dirty-cancel browser history boundary"
                                        );
                                    }
                                }
                            }
                            _ => {
                                match state_clone
                                    .enqueue_browser_action_during_gesture_with_ack(
                                        client_id, envelope, terminal,
                                    )
                                    .await
                                {
                                    Ok((_, acknowledgement)) => {
                                        send_socket_ack(&socket_ack_tx, acknowledgement).await;
                                    }
                                    Err(acknowledgement) => {
                                        send_socket_ack_with_reason(
                                            &socket_ack_tx,
                                            acknowledgement,
                                            "gesture_ownership",
                                        )
                                        .await;
                                        log::warn!(
                                            "Rejected cross-controller or cross-destination history action"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(error) => {
                        send_uncorrelated_socket_refusal(&socket_ack_tx, "invalid_json").await;
                        log::warn!(
                            "Failed to parse WebAction: {error} - excerpt: {}",
                            safe_log_excerpt(&text)
                        );
                    }
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
    state.disconnect_monitor_client(client_id);
}

async fn enqueue_socket_action(
    state: &Arc<WebState>,
    acknowledgements: &tokio::sync::mpsc::Sender<String>,
    action: crate::action_correlation::ActionEnvelope<WebAction>,
    terminal: ActionIngressTerminalGuard,
) -> EnqueueOutcome {
    let (outcome, acknowledgement) = state
        .enqueue_guarded_action_with_ack(action, terminal)
        .await;
    send_socket_ack(acknowledgements, acknowledgement).await;
    outcome
}

async fn send_socket_ack(
    acknowledgements: &tokio::sync::mpsc::Sender<String>,
    acknowledgement: super::state::ActionIngressAck,
) {
    if let Ok(message) = serde_json::to_string(&acknowledgement) {
        let _ = acknowledgements.send(message).await;
    }
}

#[cfg(test)]
async fn send_correlated_socket_refusal(
    state: &Arc<WebState>,
    acknowledgements: &tokio::sync::mpsc::Sender<String>,
    identity: crate::action_correlation::ActionIdentity,
    reason_code: &'static str,
) {
    let guard = state.action_ingress_terminal_guard(identity);
    send_guarded_socket_refusal(acknowledgements, guard, reason_code).await;
}

async fn send_guarded_socket_refusal(
    acknowledgements: &tokio::sync::mpsc::Sender<String>,
    terminal: ActionIngressTerminalGuard,
    reason_code: &'static str,
) {
    let acknowledgement =
        terminal.terminalize(crate::action_correlation::ActionDisposition::Refused);
    send_socket_ack_with_reason(acknowledgements, acknowledgement, reason_code).await;
}

async fn send_socket_ack_with_reason(
    acknowledgements: &tokio::sync::mpsc::Sender<String>,
    acknowledgement: ActionIngressAck,
    reason_code: &'static str,
) {
    let message = serde_json::json!({
        "type": acknowledgement.kind,
        "sequence": acknowledgement.sequence,
        "disposition": acknowledgement.disposition,
        "reason_code": reason_code,
    })
    .to_string();
    let _ = acknowledgements.send(message).await;
}

async fn send_uncorrelated_socket_refusal(
    acknowledgements: &tokio::sync::mpsc::Sender<String>,
    reason_code: &'static str,
) {
    let message = serde_json::json!({
        // Malformed JSON has no typed ActionEnvelope and therefore cannot
        // truthfully carry the non-zero engine ActionSequence vocabulary.
        "type": "transport_refusal",
        "reason_code": reason_code,
    })
    .to_string();
    let _ = acknowledgements.send(message).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use axum::http::HeaderValue;
    use tokio::io::AsyncReadExt as _;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::Message as ClientMessage;

    const FIRST_TEST_TOKEN: &str = "11111111111111111111111111111111";
    const SECOND_TEST_TOKEN: &str = "22222222222222222222222222222222";

    fn control_test_dir(label: &str) -> PathBuf {
        let mut random = [0_u8; 8];
        getrandom::fill(&mut random).unwrap();
        std::env::temp_dir().join(format!(
            "collide-o-scope-control-{label}-{}-{}",
            std::process::id(),
            hex_lower(&random)
        ))
    }

    fn available_port_pair() -> u16 {
        for _ in 0..128 {
            let base = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
            let port = base.local_addr().unwrap().port();
            if port == u16::MAX {
                continue;
            }
            let lan = TcpListener::bind((Ipv4Addr::UNSPECIFIED, port + 1));
            if let Ok(lan) = lan {
                drop(lan);
                drop(base);
                return port;
            }
        }
        panic!("could not reserve an adjacent control-server port pair");
    }

    /// [`available_port_pair`] can only reserve ports by binding and then
    /// dropping its probe listeners, so until the control server rebinds them
    /// any concurrent test (or unrelated process) can steal either port.
    /// Scenarios whose truth depends on a reserved port staying free retry
    /// with a fresh pair when — and only when — the observed failure is that
    /// stolen-port bind error; every other outcome is asserted unchanged.
    const STOLEN_PORT_RETRIES: usize = 8;

    fn reserved_port_was_stolen(status: &ControlListenerStatus, bind_error_marker: &str) -> bool {
        matches!(
            status,
            ControlListenerStatus::Unavailable { reason } if reason.contains(bind_error_marker)
        )
    }

    fn test_start_config(
        base_port: u16,
        identity_dir: &FsPath,
        token: &str,
        faults: StartFaults,
    ) -> StartConfig {
        StartConfig {
            base_port,
            lan_ip: Some(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 10))),
            identity_dir: identity_dir.to_path_buf(),
            token_seed: Some(token.to_string()),
            faults,
        }
    }

    async fn start_test_router(
        state: Arc<WebState>,
        session: Arc<ControlSession>,
        role: ListenerRole,
    ) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                control_router(state, session, role)
                    .into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });
        (address, server)
    }

    async fn raw_http_get(address: SocketAddr, target: &str, extra_headers: &str) -> String {
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let request = format!(
            "GET {target} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n{extra_headers}\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut response))
            .await
            .expect("HTTP response timed out")
            .unwrap();
        String::from_utf8(response).unwrap()
    }

    fn response_headers(response: &str) -> &str {
        response.split_once("\r\n\r\n").unwrap().0
    }

    fn response_cookie(response: &str, name: &str) -> String {
        response_headers(response)
            .lines()
            .find_map(|line| {
                let (header, value) = line.split_once(':')?;
                (header.eq_ignore_ascii_case("set-cookie")
                    && value.trim_start().starts_with(&format!("{name}=")))
                .then(|| value.trim().split(';').next().unwrap().to_string())
            })
            .expect("authenticated query did not mint its listener cookie")
    }

    #[test]
    fn a_short_upload_body_is_refused_rather_than_published_as_a_clip() {
        // The reproduction: a browser declared a whole file and delivered 170
        // bytes. The body ended cleanly, so the upload was renamed into the
        // library, counted as a new clip, and handed to the thumbnail
        // preflight, which is where it finally failed with "moov atom not
        // found" -- long after it had become a library entry no decoder could
        // open.
        assert!(upload_is_truncated(Some(71_407_803), 170));
        assert!(upload_is_truncated(Some(2), 1));

        // Exactly what was declared is whole.
        assert!(!upload_is_truncated(Some(71_407_803), 71_407_803));
        assert!(!upload_is_truncated(Some(0), 0));

        // More than declared is somebody else's problem: the size ceiling
        // already refused an over-long body, and calling a longer-than-
        // declared upload "truncated" would be a lie.
        assert!(!upload_is_truncated(Some(10), 11));

        // A client that declares nothing gets no verdict. Refusing every
        // chunked upload would be worse than the defect.
        assert!(!upload_is_truncated(None, 0));
        assert!(!upload_is_truncated(None, 170));
    }

    #[test]
    fn upload_deadline_has_independent_idle_and_absolute_bounds() {
        let started = Instant::now();
        let deadline = UploadDeadline {
            started,
            absolute: Duration::from_secs(10),
            idle: Duration::from_secs(3),
        };
        assert_eq!(
            deadline.remaining_at(started + Duration::from_secs(4)),
            Some(Duration::from_secs(6))
        );
        assert_eq!(
            deadline
                .next_chunk_wait_at(started + Duration::from_secs(4))
                .unwrap(),
            Duration::from_secs(3)
        );
        assert_eq!(
            deadline.remaining_at(started + Duration::from_secs(10)),
            Some(Duration::ZERO)
        );
        assert_eq!(
            deadline.remaining_at(started + Duration::from_secs(11)),
            None
        );
    }

    #[tokio::test]
    async fn valid_still_upload_uses_probe_and_durable_no_replace_publication() {
        use image::ImageEncoder as _;

        let folder = control_test_dir("durable-upload");
        fs::create_dir(&folder).unwrap();
        fs::write(folder.join(".upload-stage-crash.part"), b"partial").unwrap();
        fs::write(folder.join(".upload-reserve-crash.png"), b"").unwrap();
        let state = WebState::new().unwrap();
        *state.library_folder.write().unwrap() = Some(folder.clone());
        let library_generation = state.begin_library_generation();
        let server_generation = state.begin_control_server_generation();
        assert!(upload_is_current(
            &state,
            server_generation,
            library_generation
        ));

        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&[0, 0, 0, 0], 1, 1, image::ExtendedColorType::Rgba8)
            .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_LENGTH,
            HeaderValue::from_str(&png.len().to_string()).unwrap(),
        );
        let response = upload_handler(
            State(state.clone()),
            Query(UploadQuery {
                name: "atomic.png".to_string(),
            }),
            headers,
            axum::body::Body::from(png.clone()),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 2048)
            .await
            .unwrap();
        let acknowledgement: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(acknowledgement["type"], "action_ack");
        assert_eq!(acknowledgement["sequence"], 1);
        assert_eq!(acknowledgement["disposition"], "coalesced");
        assert_eq!(acknowledgement["name"], "atomic.png");
        assert_eq!(fs::read(folder.join("atomic.png")).unwrap(), png);
        assert!(fs::read_dir(&folder).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".upload-")
        }));
        let mut actions = state.actions.lock().await;
        assert!(matches!(
            actions.pop().map(|action| action.into_payload()),
            Some(WebAction::RescanLibrary)
        ));
        drop(actions);
        let receipts = state.action_receipts_for_test();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].sequence.get(), 1);
        assert_eq!(
            receipts[0].disposition,
            crate::action_correlation::ActionDisposition::Coalesced,
        );
        fs::remove_dir_all(folder).unwrap();
    }

    #[tokio::test]
    async fn durable_library_mutation_identity_is_exact_on_error_and_full_queue() {
        use crate::action_correlation::{ActionDisposition, ActionSourceClass};

        let state = WebState::new().unwrap();
        {
            let mut actions = state.actions.lock().await;
            for _ in 0..super::super::state::MAX_PENDING_ACTIONS {
                actions.push(state.action_sequencer().envelope(
                    ActionSourceClass::Browser,
                    WebAction::RestoreRecoveryJournal,
                ));
            }
        }

        let admitted = LibraryMutationTerminalGuard::browser(state.clone());
        let acknowledgement = admitted.finish(ActionDisposition::Coalesced);
        assert!(acknowledgement.sequence > super::super::state::MAX_PENDING_ACTIONS as u64);
        assert_eq!(
            acknowledgement.disposition,
            super::super::state::ActionIngressDisposition::Coalesced,
        );

        let refused_sequence = acknowledgement.sequence + 1;
        drop(LibraryMutationTerminalGuard::browser(state.clone()));
        let receipts = state.action_receipts_for_test();
        assert_eq!(receipts.len(), 2);
        assert_eq!(receipts[0].sequence.get(), acknowledgement.sequence);
        assert_eq!(receipts[0].disposition, ActionDisposition::Coalesced);
        assert_eq!(receipts[1].sequence.get(), refused_sequence);
        assert_eq!(receipts[1].disposition, ActionDisposition::Refused);
        assert_eq!(
            state.actions.lock().await.len(),
            super::super::state::MAX_PENDING_ACTIONS
        );

        // Once spawn_blocking starts, aborting/dropping the async JoinHandle
        // does not stop the irreversible worker. The identity therefore lives
        // with that worker and reflects its durable result, not handler life.
        let detached_state = WebState::new().unwrap();
        let mutation = LibraryMutationTerminalGuard::browser(detached_state.clone());
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker = tokio::task::spawn_blocking(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            finish_blocking_library_mutation(mutation, || Ok::<_, ()>(()))
        });
        started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("blocking mutation did not start");
        worker.abort();
        drop(worker);
        release_tx.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let receipts = detached_state.action_receipts_for_test();
            if !receipts.is_empty() {
                assert_eq!(receipts.len(), 1);
                assert_eq!(receipts[0].sequence.get(), 1);
                assert_eq!(receipts[0].disposition, ActionDisposition::Coalesced);
                break;
            }
            assert!(
                Instant::now() < deadline,
                "detached mutation lost its identity"
            );
            tokio::task::yield_now().await;
        }
    }

    #[cfg(target_os = "windows")]
    #[tokio::test]
    async fn delete_endpoint_correlates_the_actual_recycle_bin_mutation() {
        let folder = control_test_dir("correlated-delete");
        fs::create_dir(&folder).unwrap();
        let path = folder.join("delete-me.png");
        fs::write(&path, b"temporary delete fixture").unwrap();
        let state = WebState::new().unwrap();
        *state.library_folder.write().unwrap() = Some(folder.clone());

        let response = delete_handler(
            State(state.clone()),
            Query(DeleteQuery {
                name: "delete-me.png".into(),
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 2048)
            .await
            .unwrap();
        let acknowledgement: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(acknowledgement["type"], "action_ack");
        assert_eq!(acknowledgement["sequence"], 1);
        assert_eq!(acknowledgement["disposition"], "coalesced");
        assert_eq!(acknowledgement["name"], "delete-me.png");
        assert!(!path.exists());
        let receipts = state.action_receipts_for_test();
        assert_eq!(receipts.len(), 1);
        assert_eq!(
            receipts[0].disposition,
            crate::action_correlation::ActionDisposition::Coalesced,
        );
        fs::remove_dir_all(folder).unwrap();
    }

    #[tokio::test]
    async fn disconnect_and_failed_media_probe_leave_no_final_or_staging_file() {
        let folder = control_test_dir("cancelled-upload");
        fs::create_dir(&folder).unwrap();
        let state = WebState::new().unwrap();
        *state.library_folder.write().unwrap() = Some(folder.clone());
        state.begin_library_generation();
        state.begin_control_server_generation();

        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_LENGTH, HeaderValue::from_static("8"));
        let disconnected = axum::body::Body::from_stream(futures::stream::iter([
            Ok::<_, std::io::Error>(axum::body::Bytes::from_static(b"part")),
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "injected disconnect",
            )),
        ]));
        let response = upload_handler(
            State(state.clone()),
            Query(UploadQuery {
                name: "disconnected.png".to_string(),
            }),
            headers,
            disconnected,
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(!folder.join("disconnected.png").exists());

        let bytes = b"not a real PNG";
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_LENGTH,
            HeaderValue::from_str(&bytes.len().to_string()).unwrap(),
        );
        let response = upload_handler(
            State(state.clone()),
            Query(UploadQuery {
                name: "hostile.png".to_string(),
            }),
            headers,
            axum::body::Body::from(axum::body::Bytes::from_static(bytes)),
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert!(!folder.join("hostile.png").exists());
        assert!(state.actions.lock().await.is_empty());
        assert!(fs::read_dir(&folder).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".upload-")
        }));
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn server_stop_and_library_generation_are_upload_cancellation_barriers() {
        let state = WebState::new().unwrap();
        let library_generation = state.begin_library_generation();
        let server_generation = state.begin_control_server_generation();
        assert!(upload_is_current(
            &state,
            server_generation,
            library_generation
        ));
        state.mark_control_server_stopping(server_generation);
        assert!(!upload_is_current(
            &state,
            server_generation,
            library_generation
        ));

        let next_server = state.begin_control_server_generation();
        let next_library = state.begin_library_generation();
        assert!(upload_is_current(&state, next_server, next_library));
        state.begin_library_generation();
        assert!(!upload_is_current(&state, next_server, next_library));
    }

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

        // A malformed transport payload has no typed action and therefore
        // consumes neither an engine sequence nor a correlation receipt.
        let malformed =
            controller_profile_handler(State(state.clone()), axum::body::Bytes::from_static(b"{"))
                .await;
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
        assert!(state.action_receipts_for_test().is_empty());

        // Structural decoding succeeds, then semantic validation refuses the
        // action under the first real engine-owned identity.
        let semantic_invalid = crate::controller_profile::ControllerProfileAction::Import {
            document: crate::controller_profile::ControllerProfileDocument {
                version: 0,
                ..Default::default()
            },
        };
        let semantic_invalid = serde_json::to_vec(&semantic_invalid).unwrap();
        let semantic_invalid = controller_profile_handler(
            State(state.clone()),
            axum::body::Bytes::from(semantic_invalid),
        )
        .await;
        assert_eq!(semantic_invalid.status(), StatusCode::BAD_REQUEST);
        let semantic_ack = axum::body::to_bytes(semantic_invalid.into_body(), 1024)
            .await
            .unwrap();
        let semantic_ack: serde_json::Value = serde_json::from_slice(&semantic_ack).unwrap();
        assert_eq!(semantic_ack["type"], "action_ack");
        assert_eq!(semantic_ack["sequence"], 1);
        assert_eq!(semantic_ack["disposition"], "refused");
        let receipts = state.action_receipts_for_test();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].sequence.get(), 1);
        assert_eq!(
            receipts[0].disposition,
            crate::action_correlation::ActionDisposition::Refused,
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
        let acknowledgement = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let acknowledgement: serde_json::Value = serde_json::from_slice(&acknowledgement).unwrap();
        assert_eq!(acknowledgement["type"], "action_ack");
        assert_eq!(acknowledgement["sequence"], 2);
        assert_eq!(acknowledgement["disposition"], "queued");
        let mut actions = state.actions.lock().await;
        let action = actions.pop().expect("correlated controller import");
        assert_eq!(action.sequence().get(), 2);
        assert!(matches!(
            action.into_payload(),
            WebAction::ControllerProfile {
                request: crate::controller_profile::ControllerProfileAction::Import { document }
            } if document == imported
        ));
    }

    #[tokio::test]
    async fn socket_local_meta_and_semantic_refusals_ack_their_real_engine_sequence() {
        use crate::action_correlation::{ActionDisposition, ActionSourceClass};

        let state = WebState::new().expect("test access token");
        let (ack_tx, mut ack_rx) = tokio::sync::mpsc::channel(8);

        let gyro = state.envelope_web_action(WebAction::GyroStream { enabled: true });
        let gyro_identity = gyro.identity();
        assert_eq!(gyro_identity.source(), ActionSourceClass::Phone);
        assert!(state.set_gyro_stream(7, true));
        let acknowledgement =
            state.terminal_action_identity_with_ack(gyro_identity, ActionDisposition::Coalesced);
        send_socket_ack(&ack_tx, acknowledgement).await;
        let gyro_ack = ack_rx.recv().await.expect("gyro acknowledgement");
        let gyro_ack: serde_json::Value = serde_json::from_str(&gyro_ack).unwrap();
        assert_eq!(gyro_ack["sequence"], 1);
        assert_eq!(gyro_ack["disposition"], "coalesced");
        assert!(gyro_ack.get("enabled").is_none());

        let watch = state.envelope_web_action(WebAction::MonitorWatch { enabled: true });
        let watch_identity = watch.identity();
        assert_eq!(watch_identity.source(), ActionSourceClass::Browser);
        assert!(state.set_monitor_watch(7, true));
        let acknowledgement =
            state.terminal_action_identity_with_ack(watch_identity, ActionDisposition::Coalesced);
        send_socket_ack(&ack_tx, acknowledgement).await;
        let watch_ack = ack_rx.recv().await.expect("watch acknowledgement");
        let watch_ack: serde_json::Value = serde_json::from_str(&watch_ack).unwrap();
        assert_eq!(watch_ack["sequence"], 2);
        assert_eq!(watch_ack["disposition"], "coalesced");

        let invalid = state.envelope_web_action(WebAction::SetProxySettings {
            scale: crate::proxy::ProxyScale::Half,
            frame_rate: crate::proxy::ProxyFrameRate::Fixed {
                numerator: 0,
                denominator: 1,
            },
            include_audio: true,
        });
        assert!(!valid_action(invalid.payload(), 0));
        send_correlated_socket_refusal(&state, &ack_tx, invalid.identity(), "invalid_payload")
            .await;
        let invalid_ack = ack_rx.recv().await.expect("invalid acknowledgement");
        let invalid_ack: serde_json::Value = serde_json::from_str(&invalid_ack).unwrap();
        assert_eq!(invalid_ack["sequence"], 3);
        assert_eq!(invalid_ack["disposition"], "refused");
        assert_eq!(invalid_ack["reason_code"], "invalid_payload");
        assert!(invalid_ack.get("param").is_none());

        let receipts = state.action_receipts_for_test();
        assert_eq!(receipts.len(), 3);
        assert_eq!(receipts[0].disposition, ActionDisposition::Coalesced);
        assert_eq!(receipts[1].disposition, ActionDisposition::Coalesced);
        assert_eq!(receipts[2].disposition, ActionDisposition::Refused);
    }

    #[tokio::test]
    async fn authenticated_websocket_round_trip_dispatches_and_returns_authoritative_state() {
        let state = WebState::new().expect("test access token");
        let session = ControlSession::random().expect("test access token");
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server_state = state.clone();
        let server_session = session.clone();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                control_router(server_state, server_session, ListenerRole::LoopbackIpv4)
                    .into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });

        let mut request = format!("ws://{address}/ws?key={}", session.token.0)
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

        socket
            .send(ClientMessage::Text("{not-json".to_string()))
            .await
            .unwrap();
        let transport_refusal =
            tokio::time::timeout(std::time::Duration::from_secs(2), socket.next())
                .await
                .expect("malformed transport refusal timed out")
                .expect("server closed before malformed transport refusal")
                .expect("malformed transport refusal frame");
        let ClientMessage::Text(transport_refusal) = transport_refusal else {
            panic!("expected malformed transport-refusal text");
        };
        let transport_refusal: serde_json::Value =
            serde_json::from_str(&transport_refusal).unwrap();
        assert_eq!(transport_refusal["type"], "transport_refusal");
        assert_eq!(transport_refusal["reason_code"], "invalid_json");
        assert!(transport_refusal.get("sequence").is_none());
        assert!(state.action_receipts_for_test().is_empty());

        socket
            .send(ClientMessage::Text(
                r#"{"action":"set_proxy_settings","scale":"half","frame_rate":{"fixed":{"numerator":0,"denominator":1}},"include_audio":true}"#
                    .to_string(),
            ))
            .await
            .unwrap();
        let semantic_refusal =
            tokio::time::timeout(std::time::Duration::from_secs(2), socket.next())
                .await
                .expect("semantic action refusal timed out")
                .expect("server closed before semantic action refusal")
                .expect("semantic action refusal frame");
        let ClientMessage::Text(semantic_refusal) = semantic_refusal else {
            panic!("expected semantic action-refusal text");
        };
        let semantic_refusal: serde_json::Value = serde_json::from_str(&semantic_refusal).unwrap();
        assert_eq!(semantic_refusal["type"], "action_ack");
        assert_eq!(semantic_refusal["sequence"], 1);
        assert_eq!(semantic_refusal["disposition"], "refused");
        assert_eq!(semantic_refusal["reason_code"], "invalid_payload");
        let receipts = state.action_receipts_for_test();
        assert_eq!(receipts.len(), 1);
        assert_eq!(
            receipts[0].disposition,
            crate::action_correlation::ActionDisposition::Refused,
        );

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
        assert!(matches!(queued[0].payload(), WebAction::ResetFx));
        assert!(matches!(queued[1].payload(), WebAction::ResetVisualProgram));
        assert!(matches!(
            queued[2].payload(),
            WebAction::SetRouting { value, .. } if value == "layer17_opacity"
        ));
        assert!(matches!(
            queued[3].payload(),
            WebAction::SetMediaSafetyMode {
                mode: crate::media_safety::MediaSafetyMode::Expert
            }
        ));
        assert!(matches!(
            queued[4].payload(),
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
            app.handle_web_action(action.into_payload());
        }
        app.push_web_state();

        let returned = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                match socket.next().await {
                    Some(Ok(ClientMessage::Text(message))) => {
                        let value: serde_json::Value = serde_json::from_str(&message).unwrap();
                        if value.get("type").and_then(serde_json::Value::as_str) == Some("state") {
                            break message;
                        }
                    }
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
        assert_eq!(queued.len(), 3);
        assert!(matches!(
            queued[0].payload(),
            WebAction::BeginHistoryGesture { gesture_id: 77 }
        ));
        assert!(matches!(
            queued[1].payload(),
            WebAction::SetParam { param, value }
                if param == "brightness" && value.as_f64() == Some(0.625)
        ));
        assert!(matches!(
            queued[2].payload(),
            WebAction::EndHistoryGesture { gesture_id: 77 }
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
        assert_eq!(MAX_VISUAL_UPLOAD_BYTES, 2_147_483_648);
        assert!(!exceeds_upload_limit("mp4", MAX_VISUAL_UPLOAD_BYTES));
        assert!(exceeds_upload_limit("mp4", MAX_VISUAL_UPLOAD_BYTES + 1));
    }

    #[test]
    fn origin_must_match_host_exactly() {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_static("127.0.0.1:3030"));
        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("http://127.0.0.1:3030"),
        );
        assert!(same_origin(&headers, ListenerRole::LoopbackIpv4));
        assert!(!same_origin(&headers, ListenerRole::LanTls));
        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("https://127.0.0.1:3030"),
        );
        assert!(same_origin(&headers, ListenerRole::LanTls));
        assert!(!same_origin(&headers, ListenerRole::LoopbackIpv4));
        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("http://evil.example"),
        );
        assert!(!same_origin(&headers, ListenerRole::LoopbackIpv4));
        headers.remove(header::ORIGIN);
        assert!(!same_origin(&headers, ListenerRole::LoopbackIpv4));
    }

    #[tokio::test]
    async fn listener_cookie_policies_and_hardening_headers_are_exact() {
        for (role, cookie_name, secure) in [
            (ListenerRole::LoopbackIpv4, LOOPBACK_AUTH_COOKIE, false),
            (ListenerRole::LanTls, LAN_AUTH_COOKIE, true),
        ] {
            let state = WebState::new().unwrap();
            let session = ControlSession::from_seed(FIRST_TEST_TOKEN.to_string()).unwrap();
            let (address, server) = start_test_router(state, session, role).await;
            let response =
                raw_http_get(address, &format!("/missing?key={FIRST_TEST_TOKEN}"), "").await;
            let headers = response_headers(&response).to_ascii_lowercase();
            assert!(headers.starts_with("http/1.1 404"));
            assert!(headers.contains(&format!("set-cookie: {cookie_name}={FIRST_TEST_TOKEN}")));
            assert!(headers.contains("; httponly; samesite=strict"));
            assert_eq!(headers.contains("; secure"), secure);
            assert!(headers.contains("content-security-policy: default-src 'self'"));
            assert!(headers.contains("connect-src 'self'"));
            assert!(!headers.contains("connect-src 'self' ws:"));
            assert!(headers.contains("x-content-type-options: nosniff"));
            assert!(headers.contains("x-frame-options: deny"));
            assert!(headers.contains("referrer-policy: no-referrer"));
            assert!(headers.contains("cache-control: no-store, max-age=0"));
            assert!(headers.contains("pragma: no-cache"));
            server.abort();
            let _ = server.await;
        }

        // The outer layer applies the same no-store/browser policy to an
        // authentication refusal, so an intermediary cannot retain either
        // the denial page or a later token-bearing navigation.
        let state = WebState::new().unwrap();
        let session = ControlSession::from_seed(FIRST_TEST_TOKEN.to_string()).unwrap();
        let (address, server) = start_test_router(state, session, ListenerRole::LoopbackIpv4).await;
        let denied = raw_http_get(address, "/missing", "").await;
        let denied_headers = response_headers(&denied).to_ascii_lowercase();
        assert!(denied_headers.starts_with("http/1.1 403"));
        assert!(denied_headers.contains("content-security-policy:"));
        assert!(denied_headers.contains("cache-control: no-store, max-age=0"));
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn library_index_pages_require_auth_and_reject_unbounded_or_stale_requests() {
        let state = WebState::new().unwrap();
        let session = ControlSession::from_seed(FIRST_TEST_TOKEN.to_string()).unwrap();
        let (address, server) = start_test_router(state, session, ListenerRole::LoopbackIpv4).await;

        let denied = raw_http_get(address, "/library-index?limit=1", "").await;
        assert!(response_headers(&denied).starts_with("HTTP/1.1 403"));

        let accepted = raw_http_get(
            address,
            &format!("/library-index?key={FIRST_TEST_TOKEN}&kind=all&limit=1"),
            "",
        )
        .await;
        assert!(response_headers(&accepted).starts_with("HTTP/1.1 200"));
        let body = accepted.split_once("\r\n\r\n").unwrap().1;
        let page: crate::library_index::LibraryPageSnapshot =
            serde_json::from_str(body).expect("bounded library page JSON");
        assert_eq!(page.limit, 1);
        assert!(page.entries.is_empty());

        let too_large = raw_http_get(
            address,
            &format!(
                "/library-index?key={FIRST_TEST_TOKEN}&limit={}",
                crate::library_index::LIBRARY_INDEX_MAX_PAGE_SIZE + 1
            ),
            "",
        )
        .await;
        assert!(response_headers(&too_large).starts_with("HTTP/1.1 400"));

        let stale = raw_http_get(
            address,
            &format!("/library-index?key={FIRST_TEST_TOKEN}&revision=1&limit=1"),
            "",
        )
        .await;
        assert!(response_headers(&stale).starts_with("HTTP/1.1 409"));

        server.abort();
        let _ = server.await;
    }

    #[test]
    fn tls_fault_and_independent_port_occupation_publish_exact_listener_truth() {
        let mut attempts = 0;
        loop {
            attempts += 1;
            let base_port = available_port_pair();
            let identity_dir = control_test_dir("tls-fault");
            let state = WebState::new().unwrap();
            let mut handle = start(
                state.clone(),
                test_start_config(
                    base_port,
                    &identity_dir,
                    FIRST_TEST_TOKEN,
                    StartFaults {
                        tls_config: true,
                        ..StartFaults::default()
                    },
                ),
            )
            .unwrap();
            let info = state.control_server_info();
            if reserved_port_was_stolen(&info.loopback_ipv4, "cannot bind loopback IPv4") {
                handle.retire().unwrap();
                fs::remove_dir_all(&identity_dir).unwrap();
                assert!(
                    attempts < STOLEN_PORT_RETRIES,
                    "reserved loopback port was stolen {attempts} consecutive times"
                );
                continue;
            }
            assert!(matches!(
                info.loopback_ipv4,
                ControlListenerStatus::Listening { address }
                    if address == SocketAddr::new(Ipv4Addr::LOCALHOST.into(), base_port)
            ));
            assert!(matches!(
                info.lan_tls,
                ControlListenerStatus::Unavailable { ref reason }
                    if reason.contains("injected TLS configuration fault")
            ));
            assert!(info.lan_url.is_none());
            handle.retire().unwrap();
            fs::remove_dir_all(&identity_dir).unwrap();
            break;
        }

        let mut attempts = 0;
        loop {
            attempts += 1;
            let base_port = available_port_pair();
            let Ok(occupied_lan) = TcpListener::bind((Ipv4Addr::UNSPECIFIED, base_port + 1)) else {
                assert!(
                    attempts < STOLEN_PORT_RETRIES,
                    "reserved LAN port was stolen {attempts} consecutive times"
                );
                continue;
            };
            let identity_dir = control_test_dir("occupied-lan");
            let state = WebState::new().unwrap();
            let mut handle = start(
                state.clone(),
                test_start_config(
                    base_port,
                    &identity_dir,
                    FIRST_TEST_TOKEN,
                    StartFaults::default(),
                ),
            )
            .unwrap();
            let info = state.control_server_info();
            if reserved_port_was_stolen(&info.loopback_ipv4, "cannot bind loopback IPv4") {
                handle.retire().unwrap();
                drop(occupied_lan);
                fs::remove_dir_all(&identity_dir).unwrap();
                assert!(
                    attempts < STOLEN_PORT_RETRIES,
                    "reserved loopback port was stolen {attempts} consecutive times"
                );
                continue;
            }
            assert!(matches!(
                info.loopback_ipv4,
                ControlListenerStatus::Listening { .. }
            ));
            assert!(matches!(
                info.lan_tls,
                ControlListenerStatus::Unavailable { ref reason }
                    if reason.contains("cannot bind LAN HTTPS")
            ));
            assert!(info.lan_url.is_none());
            handle.retire().unwrap();
            drop(occupied_lan);
            fs::remove_dir_all(&identity_dir).unwrap();
            break;
        }

        let mut attempts = 0;
        loop {
            attempts += 1;
            let base_port = available_port_pair();
            let Ok(occupied_loopback) = TcpListener::bind((Ipv4Addr::LOCALHOST, base_port)) else {
                assert!(
                    attempts < STOLEN_PORT_RETRIES,
                    "reserved loopback port was stolen {attempts} consecutive times"
                );
                continue;
            };
            let identity_dir = control_test_dir("occupied-loopback");
            let state = WebState::new().unwrap();
            let mut handle = start(
                state.clone(),
                test_start_config(
                    base_port,
                    &identity_dir,
                    FIRST_TEST_TOKEN,
                    StartFaults::default(),
                ),
            )
            .unwrap();
            let info = state.control_server_info();
            if reserved_port_was_stolen(&info.lan_tls, "cannot bind LAN HTTPS") {
                handle.retire().unwrap();
                drop(occupied_loopback);
                fs::remove_dir_all(&identity_dir).unwrap();
                assert!(
                    attempts < STOLEN_PORT_RETRIES,
                    "reserved LAN port was stolen {attempts} consecutive times"
                );
                continue;
            }
            assert!(matches!(
                info.loopback_ipv4,
                ControlListenerStatus::Unavailable { ref reason }
                    if reason.contains("cannot bind loopback IPv4")
            ));
            assert!(matches!(
                info.lan_tls,
                ControlListenerStatus::Listening { .. }
            ));
            assert!(info.lan_url.is_some());
            handle.retire().unwrap();
            drop(occupied_loopback);
            fs::remove_dir_all(&identity_dir).unwrap();
            break;
        }
    }

    #[test]
    fn one_listener_task_crash_does_not_mask_or_retire_the_other_roles() {
        let base_port = available_port_pair();
        let identity_dir = control_test_dir("task-crash");
        let state = WebState::new().unwrap();
        let mut handle = start(
            state.clone(),
            test_start_config(
                base_port,
                &identity_dir,
                FIRST_TEST_TOKEN,
                StartFaults {
                    crash_role: Some(ListenerRole::LoopbackIpv4),
                    ..StartFaults::default()
                },
            ),
        )
        .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let info = loop {
            let info = state.control_server_info();
            if matches!(
                info.loopback_ipv4,
                ControlListenerStatus::Unavailable { .. }
            ) {
                break info;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "injected listener crash did not publish before its deadline"
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        assert!(matches!(
            info.loopback_ipv4,
            ControlListenerStatus::Unavailable { ref reason }
                if reason == "listener task crashed"
        ));
        assert!(matches!(
            info.lan_tls,
            ControlListenerStatus::Listening { .. }
        ));
        assert!(info.lan_url.is_some());
        handle.retire().unwrap();
        fs::remove_dir_all(&identity_dir).unwrap();
    }

    #[tokio::test]
    async fn restart_rebinds_fixed_ports_rotates_secrets_and_rejects_old_cookie() {
        let base_port = available_port_pair();
        let identity_dir = control_test_dir("restart");
        let state = WebState::new().unwrap();
        let mut first = start(
            state.clone(),
            test_start_config(
                base_port,
                &identity_dir,
                FIRST_TEST_TOKEN,
                StartFaults::default(),
            ),
        )
        .unwrap();
        let first_response = raw_http_get(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), base_port),
            &format!("/missing?key={FIRST_TEST_TOKEN}"),
            "",
        )
        .await;
        assert!(response_headers(&first_response).starts_with("HTTP/1.1 404"));
        let old_cookie = response_cookie(&first_response, LOOPBACK_AUTH_COOKIE);
        assert!(!format!("{first:?}").contains(FIRST_TEST_TOKEN));
        assert!(!format!("{:?}", state.control_server_info()).contains(FIRST_TEST_TOKEN));
        first.retire().unwrap();
        assert!(first.thread.is_none());

        let mut second = start(
            state.clone(),
            test_start_config(
                base_port,
                &identity_dir,
                SECOND_TEST_TOKEN,
                StartFaults::default(),
            ),
        )
        .unwrap();
        assert_ne!(first.token_for_test(), second.token_for_test());
        let old_cookie_response = raw_http_get(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), base_port),
            "/missing",
            &format!("Cookie: {old_cookie}\r\n"),
        )
        .await;
        assert!(response_headers(&old_cookie_response).starts_with("HTTP/1.1 403"));
        let fresh_response = raw_http_get(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), base_port),
            &format!("/missing?key={SECOND_TEST_TOKEN}"),
            "",
        )
        .await;
        assert!(response_headers(&fresh_response).starts_with("HTTP/1.1 404"));
        assert!(!format!("{second:?}").contains(SECOND_TEST_TOKEN));
        second.retire().unwrap();
        assert!(second.thread.is_none());

        // Retirement joined every owned task before returning: both fixed
        // roles can be rebound immediately, without a sleep or orphan race.
        let rebound_loopback = TcpListener::bind((Ipv4Addr::LOCALHOST, base_port)).unwrap();
        let rebound_lan = TcpListener::bind((Ipv4Addr::UNSPECIFIED, base_port + 1)).unwrap();
        drop(rebound_loopback);
        drop(rebound_lan);
        fs::remove_dir_all(&identity_dir).unwrap();
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
            alpha: None,
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
            alpha: None,
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
            alpha: None,
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
            alpha: None,
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
                keep_source: false,
                keep_modulation: false,
                keep_output_chain: false,
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
            keep_source: false,
            keep_modulation: false,
            keep_output_chain: false,
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
            ("long_exposure_amount", serde_json::json!(0.0)),
            ("long_exposure_amount", serde_json::json!(1.0)),
            ("long_exposure_frames", serde_json::json!(2)),
            ("long_exposure_frames", serde_json::json!(24)),
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
            ("mosh_wipe", serde_json::json!(0.75)),
            ("mosh_smear", serde_json::json!(1.0)),
            ("mosh_trail", serde_json::json!(0.4)),
            ("mosh_recycle", serde_json::json!(true)),
            ("mosh_recycle", serde_json::json!(false)),
            ("sync_amount", serde_json::json!(1.0)),
            ("sync_rate", serde_json::json!(0.0)),
            ("sync_spread", serde_json::json!(0.5)),
            ("sync_bias", serde_json::json!(-1.0)),
            ("sync_bias", serde_json::json!(1.0)),
            ("sync_latched", serde_json::json!(true)),
            ("sync_latched", serde_json::json!(false)),
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
            ("long_exposure_amount", serde_json::json!(-0.001)),
            ("long_exposure_amount", serde_json::json!(1.001)),
            ("long_exposure_frames", serde_json::json!(1)),
            ("long_exposure_frames", serde_json::json!(25)),
            ("long_exposure_frames", serde_json::json!(2.5)),
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
            ("mosh_wipe", serde_json::json!(1.001)),
            ("mosh_smear", serde_json::json!(-0.001)),
            ("mosh_trail", serde_json::json!(false)),
            ("mosh_recycle", serde_json::json!(1)),
            ("sync_amount", serde_json::json!(1.001)),
            ("sync_rate", serde_json::json!(-0.1)),
            ("sync_bias", serde_json::json!(-1.001)),
            ("sync_latched", serde_json::json!(1)),
            ("sync_offsets", serde_json::json!(0.5)),
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
    fn layer_delivery_ingress_accepts_the_closed_vocabulary_and_rejects_aliases() {
        let action = |value| WebAction::SetLayerParam {
            index: 0,
            layer_id: Some("17".into()),
            param: "delivery".into(),
            value,
        };
        for token in ["legacy_rgba", "metadata_managed"] {
            assert!(
                valid_action(&action(serde_json::json!(token)), 0),
                "server rejected exact delivery token {token}"
            );
        }
        for value in [
            serde_json::json!("planar"),
            serde_json::json!("LegacyRgba"),
            serde_json::json!("metadata-managed"),
            serde_json::json!(1),
            serde_json::Value::Bool(true),
            serde_json::Value::Null,
        ] {
            assert!(!valid_action(&action(value), 0));
        }
    }

    #[test]
    fn layer_mosh_send_ingress_requires_a_unit_interval_number() {
        let action = |value| WebAction::SetLayerParam {
            index: 0,
            layer_id: Some("17".into()),
            param: "mosh_send".into(),
            value,
        };
        for value in [
            serde_json::json!(0),
            serde_json::json!(0.375),
            serde_json::json!(1),
        ] {
            assert!(valid_action(&action(value), 0));
        }
        for value in [
            serde_json::json!(-0.001),
            serde_json::json!(1.001),
            serde_json::json!("0.5"),
            serde_json::Value::Bool(true),
            serde_json::Value::Null,
        ] {
            assert!(!valid_action(&action(value), 0));
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
    async fn newest_only_state_receiver_observes_one_latest_generation() {
        let (tx, mut rx) = tokio::sync::watch::channel(Arc::new(0_u32));
        for value in 1..=3 {
            tx.send_replace(Arc::new(value));
        }
        rx.changed().await.unwrap();
        assert_eq!(**rx.borrow_and_update(), 3);
        assert!(!rx.has_changed().unwrap());
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
    fn lfo_ingress_exposes_exactly_eight_closed_lanes() {
        let edit = |index, param: &str, value| WebAction::SetLfo {
            index,
            param: param.to_string(),
            value,
        };

        assert!(valid_action(
            &edit(7, "shape", serde_json::json!("sample_hold")),
            0
        ));
        assert!(valid_action(
            &edit(7, "seed", serde_json::json!(u32::MAX)),
            0
        ));
        assert!(valid_action(&edit(0, "beats", serde_json::json!(128.0)), 0));
        assert!(valid_action(&edit(0, "phase", serde_json::json!(-2.25)), 0));

        assert!(!valid_action(
            &edit(8, "shape", serde_json::json!("sine")),
            0
        ));
        assert!(!valid_action(
            &edit(0, "shape", serde_json::json!("noise")),
            0
        ));
        assert!(!valid_action(
            &edit(0, "enabled", serde_json::json!(true)),
            0
        ));
        assert!(!valid_action(
            &edit(0, "seed", serde_json::json!(u64::from(u32::MAX) + 1)),
            0
        ));
        assert!(!valid_action(&edit(0, "beats", serde_json::json!("4")), 0));
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

    #[test]
    fn fullscreen_display_ids_are_bounded_opaque_session_tokens() {
        let parse = |display_id: &str| WebAction::SetOutputDisplay {
            display_id: display_id.to_string(),
            inventory_generation: None,
        };
        assert!(valid_action(&parse(""), 0), "empty selects Automatic");
        assert!(valid_action(&parse("display-0123456789abcdef-2"), 0));
        let overlong = "x".repeat(65);
        for invalid in ["display one", "display/one", "display_0123", &overlong] {
            assert!(!valid_action(&parse(invalid), 0), "accepted {invalid:?}");
        }

        for resolution in ["native", "1080p"] {
            assert!(valid_action(
                &WebAction::SetSpoutResolution {
                    resolution: resolution.to_string(),
                },
                0
            ));
        }
        for resolution in ["1920x1080", "4k", "Native", ""] {
            assert!(!valid_action(
                &WebAction::SetSpoutResolution {
                    resolution: resolution.to_string(),
                },
                0
            ));
        }
    }
}
