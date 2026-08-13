//! Web UI experiment for collide-o-scope
//!
//! Run: cargo run (from experiments/web-ui/)
//! Then open: http://localhost:3030
//!
//! Demonstrates a WebSocket-based control panel. The Rust side holds
//! the effect state and broadcasts updates; the web frontend sends
//! parameter changes as JSON messages.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};
use tower_http::services::ServeDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EffectParams {
    pixelate: f32,
    rgb_split: f32,
    hue_shift: f32,
    saturation: f32,
    brightness: f32,
    contrast: f32,
    posterize: f32,
    invert: bool,
    grain_intensity: f32,
    grain_size: f32,
    grain_algo: u32,
    color_grain: bool,
    vignette: f32,
    color_drift: f32,
    breathe_scale: f32,
    breathe_rotation: f32,
    breathe_position: f32,
}

impl Default for EffectParams {
    fn default() -> Self {
        Self {
            pixelate: 1.0,
            rgb_split: 0.0,
            hue_shift: 0.0,
            saturation: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            posterize: 0.0,
            invert: false,
            grain_intensity: 0.0,
            grain_size: 1.0,
            grain_algo: 0,
            color_grain: false,
            vignette: 0.0,
            color_drift: 0.0,
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
        }
    }
}

/// A single parameter update from the web UI
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParamUpdate {
    param: String,
    value: serde_json::Value,
}

struct AppState {
    effects: RwLock<EffectParams>,
    tx: broadcast::Sender<String>,
}

#[tokio::main]
async fn main() {
    let (tx, _) = broadcast::channel::<String>(64);

    let state = Arc::new(AppState {
        effects: RwLock::new(EffectParams::default()),
        tx,
    });

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new("static"))
        .with_state(state);

    let addr = "127.0.0.1:3030";
    println!("Control panel: http://{addr}");
    println!("Open in browser — use devtools to inspect/iterate on UI");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    // Send current state on connect
    let current = state.effects.read().await;
    let init_msg = serde_json::to_string(&*current).unwrap();
    drop(current);
    let _ = sender.send(Message::Text(init_msg.into())).await;

    // Subscribe to broadcast updates
    let mut rx = state.tx.subscribe();

    // Spawn task to forward broadcasts to this client
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Receive updates from this client
    let state_clone = state.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                if let Ok(update) = serde_json::from_str::<ParamUpdate>(&text) {
                    apply_update(&state_clone, &update).await;
                    // Broadcast full state to all clients
                    let current = state_clone.effects.read().await;
                    let state_json = serde_json::to_string(&*current).unwrap();
                    let _ = state_clone.tx.send(state_json);
                }
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}

async fn apply_update(state: &AppState, update: &ParamUpdate) {
    let mut effects = state.effects.write().await;
    let v = &update.value;

    match update.param.as_str() {
        "pixelate" => if let Some(n) = v.as_f64() { effects.pixelate = n as f32; },
        "rgb_split" => if let Some(n) = v.as_f64() { effects.rgb_split = n as f32; },
        "hue_shift" => if let Some(n) = v.as_f64() { effects.hue_shift = n as f32; },
        "saturation" => if let Some(n) = v.as_f64() { effects.saturation = n as f32; },
        "brightness" => if let Some(n) = v.as_f64() { effects.brightness = n as f32; },
        "contrast" => if let Some(n) = v.as_f64() { effects.contrast = n as f32; },
        "posterize" => if let Some(n) = v.as_f64() { effects.posterize = n as f32; },
        "invert" => if let Some(b) = v.as_bool() { effects.invert = b; },
        "grain_intensity" => if let Some(n) = v.as_f64() { effects.grain_intensity = n as f32; },
        "grain_size" => if let Some(n) = v.as_f64() { effects.grain_size = n as f32; },
        "grain_algo" => if let Some(n) = v.as_u64() { effects.grain_algo = n as u32; },
        "color_grain" => if let Some(b) = v.as_bool() { effects.color_grain = b; },
        "vignette" => if let Some(n) = v.as_f64() { effects.vignette = n as f32; },
        "color_drift" => if let Some(n) = v.as_f64() { effects.color_drift = n as f32; },
        "breathe_scale" => if let Some(n) = v.as_f64() { effects.breathe_scale = n as f32; },
        "breathe_rotation" => if let Some(n) = v.as_f64() { effects.breathe_rotation = n as f32; },
        "breathe_position" => if let Some(n) = v.as_f64() { effects.breathe_position = n as f32; },
        _ => {}
    }

    // Print state to terminal (simulates the render engine receiving updates)
    println!("  {} = {}", update.param, update.value);
}
