//! Spout output: share the final composite with other visual software
//! (OBS, Resolume, MadMapper) as a named Spout2 sender.
//!
//! A worker thread owns the DX11 sender (Spout objects are thread-affine)
//! and receives frames through a bounded channel — the render loop's
//! `try_submit` drops a frame rather than ever blocking, the same contract
//! as the NTSC worker. Frames arrive from the existing async readback
//! pipeline, so enabling Spout costs one GPU readback that NTSC users are
//! already paying.
//!
//! Windows-only; on other platforms the module presents the same API and
//! reports itself unavailable.

use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

pub const SENDER_NAME: &str = "collide-o-scope";

#[derive(Debug, Clone, Default)]
pub struct SpoutStatus {
    /// True once the sender is registered and delivering frames.
    pub active: bool,
    pub error: String,
}

pub struct SpoutOut {
    tx: Option<SyncSender<(Vec<u8>, u32, u32)>>,
    status: Arc<Mutex<SpoutStatus>>,
}

impl SpoutOut {
    pub fn new() -> Self {
        Self {
            tx: None,
            status: Arc::new(Mutex::new(SpoutStatus::default())),
        }
    }

    pub fn is_running(&self) -> bool {
        self.tx.is_some()
    }

    pub fn status(&self) -> SpoutStatus {
        self.status.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Submit a frame. Drops it (returning false) if the worker is behind.
    pub fn try_submit(&mut self, pixels: Vec<u8>, width: u32, height: u32) -> bool {
        let Some(tx) = &self.tx else { return false };
        match tx.try_send((pixels, width, height)) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => {
                // Worker died (error already recorded); drop the sender side.
                self.tx = None;
                false
            }
        }
    }

    #[cfg(windows)]
    pub fn start(&mut self) {
        if self.tx.is_some() {
            return;
        }
        if let Ok(mut s) = self.status.lock() {
            *s = SpoutStatus::default();
        }

        let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<u8>, u32, u32)>(1);
        let status = self.status.clone();

        let spawned = std::thread::Builder::new()
            .name("spout-out".into())
            .spawn(move || {
                let mut sender = match spout2::dx::Sender::new(SENDER_NAME) {
                    Ok(s) => s,
                    Err(e) => {
                        if let Ok(mut st) = status.lock() {
                            st.error = format!("spout init: {e}");
                        }
                        return;
                    }
                };
                // Our pixels are RGBA; tell receivers so via DXGI format.
                const DXGI_FORMAT_R8G8B8A8_UNORM: u32 = 28;
                sender.set_format(DXGI_FORMAT_R8G8B8A8_UNORM);

                while let Ok((pixels, w, h)) = rx.recv() {
                    match sender.send_image(&pixels, w, h) {
                        Ok(()) => {
                            if let Ok(mut st) = status.lock() {
                                if !st.active {
                                    log::info!("Spout sender '{SENDER_NAME}' active ({w}x{h})");
                                }
                                st.active = true;
                            }
                        }
                        Err(e) => {
                            if let Ok(mut st) = status.lock() {
                                st.active = false;
                                st.error = format!("spout send: {e}");
                            }
                            return;
                        }
                    }
                }
            });

        match spawned {
            Ok(_) => self.tx = Some(tx),
            Err(e) => {
                if let Ok(mut s) = self.status.lock() {
                    s.error = format!("spout thread: {e}");
                }
            }
        }
    }

    #[cfg(not(windows))]
    pub fn start(&mut self) {
        if let Ok(mut s) = self.status.lock() {
            s.error = "Spout is Windows-only (use Syphon on macOS)".to_string();
        }
    }

    pub fn stop(&mut self) {
        self.tx = None;
        if let Ok(mut s) = self.status.lock() {
            s.active = false;
        }
    }
}
