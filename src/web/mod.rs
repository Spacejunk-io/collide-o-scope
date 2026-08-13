//! Embedded web control panel server.
//!
//! Spawns an axum HTTP + WebSocket server on a background tokio runtime.
//! The web frontend sends parameter changes as JSON; the render loop
//! reads shared state each frame.

pub mod server;
pub mod state;
pub mod static_files;
