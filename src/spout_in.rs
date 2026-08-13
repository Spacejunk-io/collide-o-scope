//! Spout input: receive an RGBA frame from another Windows visual application.
//!
//! The DirectX receiver lives entirely on a worker thread because Spout objects
//! are thread-affine. The worker publishes through a one-frame overwrite slot:
//! if the render loop falls behind, intermediate frames are discarded and the
//! next [`SpoutIn::try_recv`] returns the newest complete frame. Both publishing
//! and receiving use `Mutex::try_lock`, so neither side waits on the other.
//!
//! Windows-only; other platforms expose the same API and report Spout as
//! unavailable when started.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver as StopReceiver, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex, TryLockError};
use std::thread::JoinHandle;
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_millis(16);
const INITIAL_RETRY_BACKOFF: Duration = Duration::from_millis(250);
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(5);
const MAX_SENDER_NAME_BYTES: usize = 255;

/// A complete RGBA8 frame received from a Spout sender.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpoutFrame {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Increases for every frame successfully published by this `SpoutIn`.
    pub sequence: u64,
}

/// A cheap snapshot suitable for display in the control panel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpoutStatus {
    /// True while connected to the requested sender.
    pub active: bool,
    /// Requested sender name while waiting, or the actual connected name.
    pub sender_name: String,
    pub width: u32,
    pub height: u32,
    /// Sequence of the latest frame published to the render loop.
    pub sequence: u64,
    /// Most recent receiver/thread error. Cleared after a successful receive.
    pub error: String,
}

/// Nonblocking bridge from a named Spout sender to the render loop.
pub struct SpoutIn {
    sender_name: String,
    stop_tx: Option<SyncSender<()>>,
    worker: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
    next_sequence: Arc<AtomicU64>,
    latest: Arc<Mutex<Option<SpoutFrame>>>,
    status: Arc<Mutex<SpoutStatus>>,
}

impl SpoutIn {
    /// Create a stopped receiver. Control characters (including embedded NULs)
    /// are removed and names are truncated to Spout's 255-byte safe limit.
    pub fn new(name: impl AsRef<str>) -> Self {
        let sender_name = sanitize_sender_name(name.as_ref());
        let error = if sender_name.is_empty() {
            "Spout sender name is empty after sanitization".to_string()
        } else {
            String::new()
        };
        let status = SpoutStatus {
            sender_name: sender_name.clone(),
            error,
            ..SpoutStatus::default()
        };

        Self {
            sender_name,
            stop_tx: None,
            worker: None,
            running: Arc::new(AtomicBool::new(false)),
            next_sequence: Arc::new(AtomicU64::new(0)),
            latest: Arc::new(Mutex::new(None)),
            status: Arc::new(Mutex::new(status)),
        }
    }

    /// True while the receiver worker exists. This is distinct from
    /// [`SpoutStatus::active`], which means the requested sender is connected.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    pub fn status(&self) -> SpoutStatus {
        lock_recover(&self.status).clone()
    }

    /// Take the newest complete frame without waiting. Returns `None` if there
    /// is no new frame or the worker is replacing the slot at this instant.
    pub fn try_recv(&self) -> Option<SpoutFrame> {
        take_latest(&self.latest)
    }

    #[cfg(windows)]
    pub fn start(&mut self) {
        if self.is_running() {
            return;
        }

        // A worker that ended because of an error is already finished here in
        // normal operation. Reaping it keeps repeated start/stop cycles tidy.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.stop_tx = None;

        if self.sender_name.is_empty() {
            update_status(&self.status, |status| {
                status.active = false;
                status.error = "Spout sender name is empty after sanitization".to_string();
            });
            return;
        }

        clear_latest(&self.latest);
        update_status(&self.status, |status| {
            status.active = false;
            status.sender_name.clone_from(&self.sender_name);
            status.width = 0;
            status.height = 0;
            status.sequence = self.next_sequence.load(Ordering::Relaxed);
            status.error.clear();
        });

        let (stop_tx, stop_rx) = std::sync::mpsc::sync_channel(1);
        let sender_name = self.sender_name.clone();
        let running = Arc::clone(&self.running);
        let next_sequence = Arc::clone(&self.next_sequence);
        let latest = Arc::clone(&self.latest);
        let status = Arc::clone(&self.status);
        running.store(true, Ordering::Release);

        let spawned = std::thread::Builder::new()
            .name("spout-in".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    receive_worker(
                        &sender_name,
                        &stop_rx,
                        &running,
                        &next_sequence,
                        &latest,
                        &status,
                    );
                }));

                if result.is_err() {
                    update_status(&status, |current| {
                        current.active = false;
                        current.error = "Spout receive worker panicked".to_string();
                    });
                }
                running.store(false, Ordering::Release);
                update_status(&status, |current| current.active = false);
            });

        match spawned {
            Ok(worker) => {
                self.stop_tx = Some(stop_tx);
                self.worker = Some(worker);
            }
            Err(error) => {
                self.running.store(false, Ordering::Release);
                update_status(&self.status, |status| {
                    status.active = false;
                    status.error = format!("spout receive thread: {error}");
                });
            }
        }
    }

    #[cfg(not(windows))]
    pub fn start(&mut self) {
        update_status(&self.status, |status| {
            status.active = false;
            status.error = "Spout input is Windows-only".to_string();
        });
    }

    /// Request worker shutdown and wait for its short, interruptible polling
    /// interval to finish. This is a lifecycle operation; frame consumption is
    /// always performed through the nonblocking [`Self::try_recv`].
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(stop_tx) = self.stop_tx.take() {
            let _ = stop_tx.try_send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        clear_latest(&self.latest);
        update_status(&self.status, |status| {
            status.active = false;
            status.width = 0;
            status.height = 0;
        });
    }
}

impl Drop for SpoutIn {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(windows)]
fn receive_worker(
    sender_name: &str,
    stop_rx: &StopReceiver<()>,
    running: &AtomicBool,
    next_sequence: &AtomicU64,
    latest: &Mutex<Option<SpoutFrame>>,
    status: &Mutex<SpoutStatus>,
) {
    let mut retry_backoff = INITIAL_RETRY_BACKOFF;

    while running.load(Ordering::Acquire) {
        let mut receiver = match spout2::dx::Receiver::new(Some(sender_name)) {
            Ok(receiver) => receiver,
            Err(error) => {
                set_worker_error(status, format!("spout receive init: {error}"));
                if wait_or_stopped(stop_rx, running, retry_backoff) {
                    break;
                }
                retry_backoff = next_backoff(retry_backoff);
                continue;
            }
        };

        // The first receive establishes the real sender dimensions.
        let (mut width, mut height) = (16u32, 16u32);
        let mut pixels = vec![0u8; 16 * 16 * 4];
        let retry_error = loop {
            if should_stop(stop_rx, running) {
                return;
            }

            let connected = match receiver.receive_image(&mut pixels, width, height, false, false) {
                Ok(connected) => connected,
                Err(error) => break format!("spout receive image: {error}"),
            };

            if receiver.is_updated() {
                let (new_width, new_height) = receiver.sender_size();
                if new_width == 0 || new_height == 0 {
                    update_status(status, |current| {
                        current.active = false;
                        current.width = 0;
                        current.height = 0;
                    });
                } else {
                    pixels = match allocate_frame(new_width, new_height) {
                        Ok(pixels) => pixels,
                        Err(error) => break error,
                    };
                    width = new_width;
                    height = new_height;
                    update_status(status, |current| {
                        current.active = connected;
                        current.sender_name = receiver.sender_name();
                        current.width = width;
                        current.height = height;
                        if connected {
                            current.error.clear();
                        }
                    });
                }
            } else if connected {
                retry_backoff = INITIAL_RETRY_BACKOFF;
                let actual_name = receiver.sender_name();
                update_status(status, |current| {
                    current.active = true;
                    current.sender_name = actual_name;
                    current.width = width;
                    current.height = height;
                    current.error.clear();
                });

                if receiver.is_frame_new() {
                    if let Some(sequence) =
                        publish_latest(latest, &pixels, width, height, next_sequence)
                    {
                        update_status(status, |current| current.sequence = sequence);
                    }
                }
            } else {
                update_status(status, |current| {
                    current.active = false;
                    current.width = 0;
                    current.height = 0;
                });
            }

            if wait_or_stopped(stop_rx, running, POLL_INTERVAL) {
                return;
            }
        };

        set_worker_error(status, retry_error);
        if wait_or_stopped(stop_rx, running, retry_backoff) {
            break;
        }
        retry_backoff = next_backoff(retry_backoff);
    }
}

#[cfg(windows)]
fn set_worker_error(status: &Mutex<SpoutStatus>, error: String) {
    update_status(status, |current| {
        current.active = false;
        current.width = 0;
        current.height = 0;
        current.error = error;
    });
}

fn should_stop(stop_rx: &StopReceiver<()>, running: &AtomicBool) -> bool {
    if !running.load(Ordering::Acquire) {
        return true;
    }
    match stop_rx.try_recv() {
        Ok(()) | Err(TryRecvError::Disconnected) => true,
        Err(TryRecvError::Empty) => false,
    }
}

fn wait_or_stopped(stop_rx: &StopReceiver<()>, running: &AtomicBool, duration: Duration) -> bool {
    if !running.load(Ordering::Acquire) {
        return true;
    }
    match stop_rx.recv_timeout(duration) {
        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => true,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => !running.load(Ordering::Acquire),
    }
}

fn next_backoff(current: Duration) -> Duration {
    current.saturating_mul(2).min(MAX_RETRY_BACKOFF)
}

fn sanitize_sender_name(input: &str) -> String {
    let sanitized: String = input.chars().filter(|ch| !ch.is_control()).collect();
    let sanitized = sanitized.trim();
    if sanitized.len() <= MAX_SENDER_NAME_BYTES {
        return sanitized.to_string();
    }

    let mut end = MAX_SENDER_NAME_BYTES;
    while !sanitized.is_char_boundary(end) {
        end -= 1;
    }
    sanitized[..end].trim_end().to_string()
}

fn frame_byte_len(width: u32, height: u32) -> Result<usize, String> {
    crate::video::decoder::validate_media_dimensions(width, height, None)
        .map_err(|error| format!("invalid Spout sender: {error}"))?
        .try_into()
        .map_err(|_| format!("Spout sender RGBA size does not fit memory: {width}x{height}"))
}

fn allocate_frame(width: u32, height: u32) -> Result<Vec<u8>, String> {
    let len = frame_byte_len(width, height)?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(len)
        .map_err(|error| format!("could not allocate Spout frame {width}x{height}: {error}"))?;
    pixels.resize(len, 0);
    Ok(pixels)
}

fn publish_latest(
    latest: &Mutex<Option<SpoutFrame>>,
    pixels: &[u8],
    width: u32,
    height: u32,
    next_sequence: &AtomicU64,
) -> Option<u64> {
    let expected_len = frame_byte_len(width, height).ok()?;
    if pixels.len() != expected_len {
        return None;
    }

    let mut slot = match latest.try_lock() {
        Ok(slot) => slot,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => return None,
    };
    let sequence = increment_sequence(next_sequence);

    if let Some(frame) = slot
        .as_mut()
        .filter(|frame| frame.pixels.len() == pixels.len())
    {
        frame.pixels.copy_from_slice(pixels);
        frame.width = width;
        frame.height = height;
        frame.sequence = sequence;
    } else {
        *slot = Some(SpoutFrame {
            pixels: pixels.to_vec(),
            width,
            height,
            sequence,
        });
    }
    Some(sequence)
}

fn increment_sequence(counter: &AtomicU64) -> u64 {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_add(1);
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return next,
            Err(observed) => current = observed,
        }
    }
}

fn take_latest(latest: &Mutex<Option<SpoutFrame>>) -> Option<SpoutFrame> {
    match latest.try_lock() {
        Ok(mut slot) => slot.take(),
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner().take(),
        Err(TryLockError::WouldBlock) => None,
    }
}

fn clear_latest(latest: &Mutex<Option<SpoutFrame>>) {
    *lock_recover(latest) = None;
}

fn update_status(status: &Mutex<SpoutStatus>, update: impl FnOnce(&mut SpoutStatus)) {
    update(&mut lock_recover(status));
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sender_name_is_safe_for_spout_c_strings() {
        assert_eq!(sanitize_sender_name("  Camera\0\nOne  "), "CameraOne");
        assert_eq!(sanitize_sender_name("\t\0\r"), "");

        let long = format!("{}x", "é".repeat(200));
        let sanitized = sanitize_sender_name(&long);
        assert!(sanitized.len() <= MAX_SENDER_NAME_BYTES);
        assert!(sanitized.is_char_boundary(sanitized.len()));
    }

    #[test]
    fn frame_lengths_are_checked_and_bounded() {
        assert_eq!(frame_byte_len(1920, 1080), Ok(1920 * 1080 * 4));
        assert_eq!(frame_byte_len(3840, 2160), Ok(3840 * 2160 * 4));
        assert!(frame_byte_len(4096, 2160).is_err());
        assert!(frame_byte_len(0, 1080).is_err());
        assert!(frame_byte_len(u32::MAX, u32::MAX).is_err());
        assert!(frame_byte_len(16_384, 16_384).is_err());
    }

    #[test]
    fn newest_frame_overwrites_an_unconsumed_frame() {
        let latest = Mutex::new(None);
        let sequence = AtomicU64::new(0);
        assert_eq!(publish_latest(&latest, &[1; 8], 2, 1, &sequence), Some(1));
        assert_eq!(publish_latest(&latest, &[7; 8], 2, 1, &sequence), Some(2));

        let frame = take_latest(&latest).expect("latest frame");
        assert_eq!(frame.pixels, vec![7; 8]);
        assert_eq!(frame.sequence, 2);
        assert!(take_latest(&latest).is_none());
    }

    #[test]
    fn publishing_and_receiving_drop_instead_of_waiting() {
        let latest = Mutex::new(None);
        let sequence = AtomicU64::new(0);
        let guard = latest.lock().expect("lock test slot");

        assert_eq!(publish_latest(&latest, &[1; 4], 1, 1, &sequence), None);
        assert_eq!(take_latest(&latest), None);
        assert_eq!(sequence.load(Ordering::Relaxed), 0);

        drop(guard);
        assert_eq!(publish_latest(&latest, &[1; 4], 1, 1, &sequence), Some(1));
    }

    #[test]
    fn retry_backoff_stops_growing_at_the_bound() {
        let mut delay = INITIAL_RETRY_BACKOFF;
        for _ in 0..20 {
            delay = next_backoff(delay);
        }
        assert_eq!(delay, MAX_RETRY_BACKOFF);
    }
}
