//! Threaded wrapper around VideoDecoder.
//!
//! Decoding (ffmpeg + YUV→RGBA scaling) runs on a dedicated thread per layer,
//! feeding a small bounded channel. The render thread only ever does a
//! non-blocking try_recv, so decode hitches (loop-point reopen, slow codecs)
//! never stall the GPU frame. The bounded channel provides natural
//! backpressure: when the render loop stops consuming (pause, slow fps),
//! the decode thread blocks on send and goes idle.

use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};

use super::VideoDecoder;

/// Frames buffered ahead of the render loop. Small to keep seek/param
/// latency low while still absorbing decode jitter.
const CHANNEL_DEPTH: usize = 2;

pub struct DecodedFrame {
    pub rgba: Vec<u8>,
    /// Loop progress 0.0..1.0 at the time this frame was decoded.
    pub progress: f32,
}

pub struct ThreadedDecoder {
    rx: Receiver<DecodedFrame>,
    pub width: u32,
    pub height: u32,
    /// Progress of the most recently consumed frame.
    progress: f32,
    /// True once the decode thread has exited (unrecoverable error).
    finished: bool,
}

impl ThreadedDecoder {
    /// Spawn the decode thread. The decoder is opened *inside* the thread
    /// (ffmpeg contexts hold raw pointers and aren't Send); the open result
    /// and dimensions come back through a one-shot channel, so errors still
    /// surface synchronously to the caller.
    pub fn open(path: &str) -> Result<Self, String> {
        let (tx, rx): (SyncSender<DecodedFrame>, Receiver<DecodedFrame>) =
            std::sync::mpsc::sync_channel(CHANNEL_DEPTH);
        let (meta_tx, meta_rx) = std::sync::mpsc::channel::<Result<(u32, u32), String>>();

        let thread_name = format!("decode-{}", short_name(path));
        let path_owned = path.to_string();
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || match VideoDecoder::open(&path_owned) {
                Ok(decoder) => {
                    let _ = meta_tx.send(Ok((decoder.width, decoder.height)));
                    decode_loop(decoder, tx);
                }
                Err(e) => {
                    let _ = meta_tx.send(Err(e));
                }
            })
            .map_err(|e| format!("Failed to spawn decode thread: {e}"))?;

        let (width, height) = meta_rx
            .recv()
            .map_err(|_| "Decode thread died during open".to_string())??;

        Ok(Self {
            rx,
            width,
            height,
            progress: 0.0,
            finished: false,
        })
    }

    /// Non-blocking: returns the next decoded frame if one is ready.
    pub fn try_next_frame(&mut self) -> Option<Vec<u8>> {
        match self.rx.try_recv() {
            Ok(frame) => {
                self.progress = frame.progress;
                Some(frame.rgba)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.finished = true;
                None
            }
        }
    }

    /// Loop progress of the most recently consumed frame, 0.0..1.0.
    pub fn progress(&self) -> f32 {
        self.progress
    }
}

fn decode_loop(mut decoder: VideoDecoder, tx: SyncSender<DecodedFrame>) {
    loop {
        let Some(rgba) = decoder.next_frame() else {
            // Unrecoverable decode error; drop the sender so the layer
            // sees Disconnected instead of waiting forever.
            return;
        };
        let frame = DecodedFrame {
            rgba,
            progress: decoder.progress(),
        };
        // Blocks when the channel is full — this paces the thread to the
        // render loop's consumption rate. Err means the layer was removed.
        if tx.send(frame).is_err() {
            return;
        }
    }
}

fn short_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().chars().take(12).collect())
        .unwrap_or_else(|| "video".to_string())
}
