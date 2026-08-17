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

use crate::media_safety::{
    MediaDeviceLimits, MediaReservation, MediaSafetyPolicy, MediaSourceKind,
};

const POLL_INTERVAL: Duration = Duration::from_millis(16);
const INITIAL_RETRY_BACKOFF: Duration = Duration::from_millis(250);
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(5);
const MAX_SENDER_NAME_BYTES: usize = 255;
// Spout's default shared-texture format. `receive_image` returns the texture's
// native four-byte channel order, while our wgpu layer texture is RGBA8.
const DXGI_FORMAT_B8G8R8A8_UNORM: u32 = 87;

#[cfg(windows)]
struct ReceiveWorkerContext<'a> {
    running: &'a AtomicBool,
    next_sequence: &'a AtomicU64,
    latest: &'a Mutex<Option<SpoutFrame>>,
    status: &'a Mutex<SpoutStatus>,
    media_policy: &'a MediaSafetyPolicy,
    device_limits: MediaDeviceLimits,
}

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
    media_policy: MediaSafetyPolicy,
    device_limits: MediaDeviceLimits,
}

impl SpoutIn {
    /// Create a stopped receiver. Control characters (including embedded NULs)
    /// are removed and names are truncated to Spout's 255-byte safe limit.
    #[allow(dead_code)] // Compatibility Safe wrapper for pre-policy callers.
    pub fn new(name: impl AsRef<str>) -> Self {
        Self::new_with_media_policy(name, MediaSafetyPolicy::safe(), MediaDeviceLimits::none())
    }

    /// Create a stopped receiver under an explicit process-local policy. The
    /// worker reserves every above-UHD receive buffer before asking Spout to
    /// fill it, and retains that reservation while the sender size is live.
    pub fn new_with_media_policy(
        name: impl AsRef<str>,
        media_policy: MediaSafetyPolicy,
        device_limits: MediaDeviceLimits,
    ) -> Self {
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
            media_policy,
            device_limits,
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
        let media_policy = self.media_policy.clone();
        let device_limits = self.device_limits;
        running.store(true, Ordering::Release);

        let spawned = std::thread::Builder::new()
            .name("spout-in".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let context = ReceiveWorkerContext {
                        running: &running,
                        next_sequence: &next_sequence,
                        latest: &latest,
                        status: &status,
                        media_policy: &media_policy,
                        device_limits,
                    };
                    receive_worker(&sender_name, &stop_rx, &context);
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
    context: &ReceiveWorkerContext<'_>,
) {
    let running = context.running;
    let next_sequence = context.next_sequence;
    let latest = context.latest;
    let status = context.status;
    let media_policy = context.media_policy;
    let device_limits = context.device_limits;
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
        let (mut pixels, mut _media_reservation) =
            match allocate_frame_with_media_policy(16, 16, media_policy, device_limits) {
                Ok(allocation) => allocation,
                Err(error) => {
                    set_worker_error(status, error);
                    return;
                }
            };
        let mut approved_len = pixels.len();
        let mut sender_format = 0;
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
                sender_format = receiver.sender_format();
                if new_width == 0 || new_height == 0 {
                    update_status(status, |current| {
                        current.active = false;
                        current.width = 0;
                        current.height = 0;
                    });
                } else {
                    let (new_pixels, new_reservation) = match allocate_frame_with_media_policy(
                        new_width,
                        new_height,
                        media_policy,
                        device_limits,
                    ) {
                        Ok(allocation) => allocation,
                        Err(error) => break error,
                    };
                    pixels = new_pixels;
                    _media_reservation = new_reservation;
                    approved_len = pixels.len();
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
                    match publish_latest_approved(
                        latest,
                        &pixels,
                        width,
                        height,
                        approved_len,
                        sender_format,
                        next_sequence,
                    ) {
                        Ok(Some(sequence)) => {
                            update_status(status, |current| current.sequence = sequence);
                        }
                        Ok(None) => {}
                        Err(error) => break error,
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

#[cfg(test)]
fn frame_byte_len(width: u32, height: u32) -> Result<usize, String> {
    crate::video::decoder::validate_media_dimensions(width, height, None)
        .map_err(|error| format!("invalid Spout sender: {error}"))?
        .try_into()
        .map_err(|_| format!("Spout sender RGBA size does not fit memory: {width}x{height}"))
}

#[cfg(test)]
fn frame_byte_len_with_media_policy(
    width: u32,
    height: u32,
    media_policy: &MediaSafetyPolicy,
    device_limits: MediaDeviceLimits,
) -> Result<usize, String> {
    media_policy
        .plan(MediaSourceKind::Spout, width, height, device_limits)
        .map_err(|error| format!("invalid Spout sender: {error}"))?
        .rgba_bytes
        .try_into()
        .map_err(|_| format!("Spout sender RGBA size does not fit memory: {width}x{height}"))
}

fn allocate_frame_with_media_policy(
    width: u32,
    height: u32,
    media_policy: &MediaSafetyPolicy,
    device_limits: MediaDeviceLimits,
) -> Result<(Vec<u8>, MediaReservation), String> {
    let reservation = media_policy
        .reserve_source(MediaSourceKind::Spout, width, height, device_limits)
        .map_err(|error| format!("invalid Spout sender: {error}"))?;
    let len = usize::try_from(reservation.plan().rgba_bytes)
        .map_err(|_| format!("Spout sender RGBA size does not fit memory: {width}x{height}"))?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(len)
        .map_err(|error| format!("could not allocate Spout frame {width}x{height}: {error}"))?;
    pixels.resize(len, 0);
    Ok((pixels, reservation))
}

#[cfg(test)]
fn publish_latest(
    latest: &Mutex<Option<SpoutFrame>>,
    pixels: &[u8],
    width: u32,
    height: u32,
    sender_format: u32,
    next_sequence: &AtomicU64,
) -> Option<u64> {
    let expected_len = frame_byte_len(width, height).ok()?;
    publish_latest_approved(
        latest,
        pixels,
        width,
        height,
        expected_len,
        sender_format,
        next_sequence,
    )
    .ok()
    .flatten()
}

fn publish_latest_approved(
    latest: &Mutex<Option<SpoutFrame>>,
    pixels: &[u8],
    width: u32,
    height: u32,
    expected_len: usize,
    sender_format: u32,
    next_sequence: &AtomicU64,
) -> Result<Option<u64>, String> {
    if pixels.len() != expected_len {
        return Err(format!(
            "Spout sender {}x{} supplied {} bytes; expected {expected_len}",
            width,
            height,
            pixels.len()
        ));
    }

    let mut slot = match latest.try_lock() {
        Ok(slot) => slot,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => return Ok(None),
    };
    let sequence = increment_sequence(next_sequence);

    if let Some(frame) = slot
        .as_mut()
        .filter(|frame| frame.pixels.len() == pixels.len())
    {
        copy_received_pixels_to_rgba(&mut frame.pixels, pixels, sender_format);
        frame.width = width;
        frame.height = height;
        frame.sequence = sequence;
    } else {
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(pixels.len()).map_err(|error| {
            format!(
                "could not allocate Spout publish frame {}x{} ({} bytes): {error}",
                width,
                height,
                pixels.len()
            )
        })?;
        rgba.resize(pixels.len(), 0);
        copy_received_pixels_to_rgba(&mut rgba, pixels, sender_format);
        *slot = Some(SpoutFrame {
            pixels: rgba,
            width,
            height,
            sequence,
        });
    }
    Ok(Some(sequence))
}

/// Copy a native Spout pixel buffer into the RGBA8 layout used by wgpu.
///
/// Format 87 is Spout's documented default and is byte-ordered BGRA. Known
/// RGBA formats and unrecognized format codes are copied unchanged: guessing
/// at an unknown layout would be more destructive than preserving the SDK's
/// legacy byte-for-byte behavior. Conversion happens during the mandatory
/// copy into the one-frame slot, so it adds neither a second full-frame pass
/// nor another allocation in steady state.
fn copy_received_pixels_to_rgba(destination: &mut [u8], source: &[u8], sender_format: u32) {
    debug_assert_eq!(destination.len(), source.len());
    if sender_format != DXGI_FORMAT_B8G8R8A8_UNORM {
        destination.copy_from_slice(source);
        return;
    }

    for (rgba, bgra) in destination.chunks_exact_mut(4).zip(source.chunks_exact(4)) {
        rgba[0] = bgra[2];
        rgba[1] = bgra[1];
        rgba[2] = bgra[0];
        rgba[3] = bgra[3];
    }
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
    fn expert_spout_length_still_obeys_policy_and_device_limits() {
        let policy = MediaSafetyPolicy::for_test(
            crate::media_safety::MediaSafetyMode::Expert,
            2 * 1024 * 1024 * 1024,
        );
        let device = MediaDeviceLimits::new(8_192, 512 * 1024 * 1024);
        assert_eq!(
            frame_byte_len_with_media_policy(7_680, 4_320, &policy, device),
            Ok(7_680 * 4_320 * 4)
        );
        assert!(frame_byte_len_with_media_policy(8_193, 4_320, &policy, device).is_err());
        assert!(frame_byte_len_with_media_policy(
            7_680,
            4_320,
            &policy,
            MediaDeviceLimits::new(8_192, 64 * 1024 * 1024),
        )
        .is_err());
    }

    #[test]
    fn newest_frame_overwrites_an_unconsumed_frame() {
        let latest = Mutex::new(None);
        let sequence = AtomicU64::new(0);
        assert_eq!(
            publish_latest(&latest, &[1; 8], 2, 1, 28, &sequence),
            Some(1)
        );
        assert_eq!(
            publish_latest(&latest, &[7; 8], 2, 1, 28, &sequence),
            Some(2)
        );

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

        assert_eq!(publish_latest(&latest, &[1; 4], 1, 1, 28, &sequence), None);
        assert_eq!(take_latest(&latest), None);
        assert_eq!(sequence.load(Ordering::Relaxed), 0);

        drop(guard);
        assert_eq!(
            publish_latest(&latest, &[1; 4], 1, 1, 28, &sequence),
            Some(1)
        );
    }

    #[test]
    fn bgra_sender_pixels_are_normalized_to_rgba() {
        let source = [10, 20, 30, 40, 1, 2, 3, 4];
        let mut destination = [0; 8];

        copy_received_pixels_to_rgba(&mut destination, &source, DXGI_FORMAT_B8G8R8A8_UNORM);

        assert_eq!(destination, [30, 20, 10, 40, 3, 2, 1, 4]);
    }

    #[test]
    fn rgba_and_unknown_sender_formats_remain_byte_exact() {
        let source = [10, 20, 30, 40, 1, 2, 3, 4];
        for format in [28, 29, 0, u32::MAX] {
            let mut destination = [0; 8];
            copy_received_pixels_to_rgba(&mut destination, &source, format);
            assert_eq!(destination, source, "format {format}");
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires the standard Spout Sender demo to be running"]
    fn live_demo_sender_delivers_a_coloured_frame() {
        let mut input = SpoutIn::new("Spout Sender");
        input.start();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut received = None;

        while std::time::Instant::now() < deadline {
            if let Some(frame) = input.try_recv() {
                let has_colour = frame
                    .pixels
                    .chunks_exact(4)
                    .any(|pixel| pixel[..3].iter().any(|channel| *channel != 0));
                if has_colour {
                    received = Some((input.status(), frame));
                    break;
                }
            }
            std::thread::sleep(POLL_INTERVAL);
        }

        input.stop();
        let (status, frame) = received.expect("a non-black frame from 'Spout Sender'");
        assert!(status.active);
        assert_eq!(status.sender_name, "Spout Sender");
        assert_eq!((frame.width, frame.height), (status.width, status.height));
        assert!(frame.sequence > 0);
        eprintln!(
            "received '{}' at {}x{}, frame {}",
            status.sender_name, frame.width, frame.height, frame.sequence
        );
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
