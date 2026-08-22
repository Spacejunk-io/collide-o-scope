//! Spout output: share the final composite with other visual software
//! (OBS, Resolume, MadMapper) as a named Spout2 sender.
//!
//! A worker thread owns the DX11 sender (Spout objects are thread-affine).
//! The render loop writes to a one-frame mailbox: ordinary frames replace an
//! older pending frame rather than blocking. A blackout request replaces any
//! pending colour frame and briefly takes the send barrier, so no pre-cut send
//! can complete after the cut request returns.
//!
//! Windows-only; on other platforms the module presents the same API and
//! reports itself unavailable.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, TryLockError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(any(windows, test))]
pub const SENDER_NAME: &str = "collide-o-scope";
const RETRY_BACKOFF: Duration = Duration::from_secs(5);

#[cfg(any(windows, test))]
pub const SPOUT_1080P_WIDTH: u32 = 1920;
#[cfg(any(windows, test))]
pub const SPOUT_1080P_HEIGHT: u32 = 1080;

/// Resolution advertised by the Spout sender.
///
/// The keys are a stable host-control boundary. `Native` preserves the live
/// renderer's dimensions; `Fixed1920x1080` scales only on the sender worker so
/// the render thread keeps its existing nonblocking, source-sized contract.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum SpoutResolutionMode {
    Native = 0,
    #[default]
    Fixed1920x1080 = 1,
}

impl SpoutResolutionMode {
    pub const fn key(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Fixed1920x1080 => "1080p",
        }
    }

    pub const fn from_key(key: &str) -> Option<Self> {
        match key.as_bytes() {
            b"native" => Some(Self::Native),
            b"1080p" => Some(Self::Fixed1920x1080),
            _ => None,
        }
    }

    #[cfg(any(windows, test))]
    pub const fn output_dimensions(self, native_width: u32, native_height: u32) -> (u32, u32) {
        match self {
            Self::Native => (native_width, native_height),
            Self::Fixed1920x1080 => (SPOUT_1080P_WIDTH, SPOUT_1080P_HEIGHT),
        }
    }

    const fn from_atomic(value: u8) -> Self {
        match value {
            0 => Self::Native,
            _ => Self::Fixed1920x1080,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SpoutStatus {
    /// True once the sender is registered and delivering frames.
    pub active: bool,
    pub error: String,
    /// Identifies the currently authoritative worker run. Status updates from
    /// an older, stopped worker are ignored.
    pub generation: u64,
    /// Requested sender-resolution policy. The delivered dimensions remain
    /// zero until the current worker completes a successful Spout send.
    pub resolution_mode: SpoutResolutionMode,
    /// Exact dimensions of the most recently delivered frame in this worker
    /// generation, after any worker-side scaling.
    pub delivered_width: u32,
    pub delivered_height: u32,
    delivered_black_epoch: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameKind {
    Regular,
    Black,
}

#[derive(Debug)]
struct QueuedFrame {
    #[cfg(any(windows, test))]
    pixels: Vec<u8>,
    #[cfg(any(windows, test))]
    width: u32,
    #[cfg(any(windows, test))]
    height: u32,
    epoch: u64,
    kind: FrameKind,
}

#[cfg(any(windows, test))]
#[derive(Debug, PartialEq, Eq)]
struct PreparedSenderFrame {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

#[cfg(any(windows, test))]
fn rgba8_byte_len(width: u32, height: u32) -> Result<usize, String> {
    if width == 0 || height == 0 {
        return Err(format!("invalid zero-sized RGBA frame {width}x{height}"));
    }
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| format!("RGBA frame dimensions overflow host memory: {width}x{height}"))
}

/// Consume and, when requested, scale one accepted mailbox frame. This runs
/// exclusively on the Spout worker. Main's render/readback thread never pays
/// for the 1080p allocation or filter pass.
#[cfg(any(windows, test))]
fn prepare_sender_frame(
    frame: &mut QueuedFrame,
    mode: SpoutResolutionMode,
) -> Result<PreparedSenderFrame, String> {
    let expected = rgba8_byte_len(frame.width, frame.height)?;
    if frame.pixels.len() != expected {
        return Err(format!(
            "invalid RGBA frame buffer for {}x{}: expected {expected} bytes, got {}",
            frame.width,
            frame.height,
            frame.pixels.len()
        ));
    }

    let (width, height) = mode.output_dimensions(frame.width, frame.height);
    if (width, height) == (frame.width, frame.height) {
        return Ok(PreparedSenderFrame {
            pixels: std::mem::take(&mut frame.pixels),
            width,
            height,
        });
    }

    let output_len = rgba8_byte_len(width, height)?;
    if frame.kind == FrameKind::Black {
        // Preserve an exact emergency black without spending worker time on a
        // convolution whose mathematical result is already known.
        drop(std::mem::take(&mut frame.pixels));
        return Ok(PreparedSenderFrame {
            pixels: vec![0; output_len],
            width,
            height,
        });
    }

    let source =
        image::RgbaImage::from_raw(frame.width, frame.height, std::mem::take(&mut frame.pixels))
            .ok_or_else(|| {
                format!(
                    "could not construct RGBA source image for {}x{}",
                    frame.width, frame.height
                )
            })?;
    let scaled = image::imageops::resize(
        &source,
        width,
        height,
        image::imageops::FilterType::Triangle,
    );
    debug_assert_eq!(scaled.as_raw().len(), output_len);
    Ok(PreparedSenderFrame {
        pixels: scaled.into_raw(),
        width,
        height,
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct OutputIntent {
    epoch: u64,
    blackout: bool,
}

#[derive(Debug, Default)]
struct MailboxState {
    pending: Option<QueuedFrame>,
    intent: OutputIntent,
    shutdown: bool,
}

/// A latest-frame mailbox with a barrier reserved for safety-critical cuts.
///
/// Ordinary producers only use `try_lock`, so the render loop never waits for
/// the worker. `queue_black` may wait for the one send already in progress;
/// that exceptional barrier makes the cut authoritative when it returns.
#[derive(Debug, Default)]
struct FrameMailbox {
    state: Mutex<MailboxState>,
    available: Condvar,
    send_barrier: Mutex<()>,
}

impl FrameMailbox {
    fn try_queue_regular(&self, frame: QueuedFrame) -> bool {
        debug_assert_eq!(frame.kind, FrameKind::Regular);
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return false,
        };
        if state.shutdown
            || frame.epoch < state.intent.epoch
            || (frame.epoch == state.intent.epoch && state.intent.blackout)
        {
            return false;
        }

        if frame.epoch > state.intent.epoch {
            state.intent = OutputIntent {
                epoch: frame.epoch,
                blackout: false,
            };
        }
        // Latest-frame semantics: overwrite a pending regular frame instead
        // of queuing latency. A valid regular frame from a newer epoch also
        // releases an older blackout intent.
        state.pending = Some(frame);
        self.available.notify_one();
        true
    }

    fn queue_black(&self, width: u32, height: u32, epoch: u64) -> bool {
        let Some(byte_len) = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
        else {
            return false;
        };
        #[cfg(not(any(windows, test)))]
        let _ = byte_len;

        // This is deliberately the sole blocking submit path. Waiting for the
        // current send establishes a cut barrier; the worker revalidates its
        // dequeued frame while holding the same barrier before every send.
        let _barrier = lock_unpoison(&self.send_barrier);
        let mut state = lock_unpoison(&self.state);
        if state.shutdown || epoch < state.intent.epoch {
            return false;
        }
        if state.intent
            == (OutputIntent {
                epoch,
                blackout: true,
            })
        {
            // Already pending or delivered for this epoch. The barrier means
            // an in-flight black send has completed before this return.
            return true;
        }

        state.intent = OutputIntent {
            epoch,
            blackout: true,
        };
        state.pending = Some(QueuedFrame {
            #[cfg(any(windows, test))]
            pixels: vec![0; byte_len],
            #[cfg(any(windows, test))]
            width,
            #[cfg(any(windows, test))]
            height,
            epoch,
            kind: FrameKind::Black,
        });
        self.available.notify_one();
        true
    }

    /// Advance the authoritative colour generation without sending a frame.
    ///
    /// This is the non-black counterpart to `queue_black`: a paused renderer
    /// may need to retain the receiver's last texture while invalidating every
    /// pending or already-dequeued frame from an older visual generation. The
    /// send barrier guarantees that, once this returns, no old frame can be
    /// submitted to Spout. A separately tagged readback may then publish the
    /// exact held audience pixels under `epoch`.
    fn hold_colour_epoch(&self, epoch: u64) -> bool {
        let _barrier = lock_unpoison(&self.send_barrier);
        let mut state = lock_unpoison(&self.state);
        if state.shutdown || epoch < state.intent.epoch {
            return false;
        }
        state.intent = OutputIntent {
            epoch,
            blackout: false,
        };
        state.pending = None;
        true
    }

    #[cfg(any(windows, test))]
    fn recv(&self) -> Option<QueuedFrame> {
        let mut state = lock_unpoison(&self.state);
        loop {
            if state.shutdown {
                return None;
            }
            if let Some(frame) = state.pending.take() {
                return Some(frame);
            }
            state = self
                .available
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
    }

    /// Called by the worker while holding `send_barrier`, immediately before
    /// touching Spout. This invalidates a colour frame dequeued just before a
    /// blackout request replaced the mailbox contents.
    #[cfg(any(windows, test))]
    fn is_current(&self, frame: &QueuedFrame) -> bool {
        let state = lock_unpoison(&self.state);
        if state.shutdown || frame.epoch != state.intent.epoch {
            return false;
        }
        matches!(frame.kind, FrameKind::Black) == state.intent.blackout
    }

    fn shutdown(&self) {
        let mut state = lock_unpoison(&self.state);
        state.shutdown = true;
        state.pending = None;
        self.available.notify_all();
    }
}

struct WorkerRun {
    mailbox: Arc<FrameMailbox>,
    join: JoinHandle<()>,
}

pub struct SpoutOut {
    worker: Option<WorkerRun>,
    status: Arc<Mutex<SpoutStatus>>,
    resolution_mode: Arc<AtomicU8>,
    generation: u64,
    retry_after: Option<Instant>,
}

impl SpoutOut {
    pub fn new() -> Self {
        Self::with_resolution_mode(SpoutResolutionMode::default())
    }

    pub fn with_resolution_mode(resolution_mode: SpoutResolutionMode) -> Self {
        Self {
            worker: None,
            status: Arc::new(Mutex::new(SpoutStatus {
                resolution_mode,
                ..SpoutStatus::default()
            })),
            resolution_mode: Arc::new(AtomicU8::new(resolution_mode as u8)),
            generation: 0,
            retry_after: None,
        }
    }

    pub fn resolution_mode(&self) -> SpoutResolutionMode {
        SpoutResolutionMode::from_atomic(self.resolution_mode.load(Ordering::Relaxed))
    }

    /// Change the worker-side output policy. An active worker adopts it on its
    /// next mailbox frame; the last delivered dimensions remain authoritative
    /// in [`SpoutStatus`] until that send succeeds.
    pub fn set_resolution_mode(&self, resolution_mode: SpoutResolutionMode) {
        self.resolution_mode
            .store(resolution_mode as u8, Ordering::Relaxed);
        lock_unpoison(&self.status).resolution_mode = resolution_mode;
    }

    pub fn is_running(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(|worker| !worker.join.is_finished())
    }

    pub fn status(&self) -> SpoutStatus {
        lock_unpoison(&self.status).clone()
    }

    pub fn black_delivered(&self, epoch: u64) -> bool {
        let status = lock_unpoison(&self.status);
        status.generation == self.generation && status.delivered_black_epoch == Some(epoch)
    }

    /// Submit an ordinary frame for a visual generation. This is always
    /// nonblocking and replaces an older pending ordinary frame.
    pub fn try_submit(&mut self, pixels: Vec<u8>, width: u32, height: u32, epoch: u64) -> bool {
        self.reap_finished_worker();
        let Some(worker) = &self.worker else {
            return false;
        };
        #[cfg(not(any(windows, test)))]
        let _ = (&pixels, width, height);
        worker.mailbox.try_queue_regular(QueuedFrame {
            #[cfg(any(windows, test))]
            pixels,
            #[cfg(any(windows, test))]
            width,
            #[cfg(any(windows, test))]
            height,
            epoch,
            kind: FrameKind::Regular,
        })
    }

    /// Establish an authoritative blackout generation.
    ///
    /// Unlike normal frame submission, this transition may wait for the one
    /// Spout send already in progress. It replaces pending colour immediately,
    /// and no colour frame from this or an older epoch can be sent afterward.
    pub fn cut_to_black(&mut self, width: u32, height: u32, epoch: u64) -> bool {
        self.reap_finished_worker();
        let Some(worker) = &self.worker else {
            return false;
        };
        worker.mailbox.queue_black(width, height, epoch)
    }

    /// Hold the receiver's last delivered texture while advancing the
    /// authoritative colour generation. This transition may briefly wait for
    /// the one send already in progress, exactly like blackout, so no queued
    /// frame from an older epoch can escape after it returns.
    pub fn hold_colour_epoch(&mut self, epoch: u64) -> bool {
        self.reap_finished_worker();
        let Some(worker) = &self.worker else {
            return false;
        };
        worker.mailbox.hold_colour_epoch(epoch)
    }

    #[cfg(windows)]
    pub fn start(&mut self) {
        self.reap_finished_worker();
        if self.worker.is_some() {
            return;
        }
        if self
            .retry_after
            .is_some_and(|deadline| Instant::now() < deadline)
        {
            return;
        }

        let generation = self.next_generation();
        let resolution_mode = self.resolution_mode();
        {
            let mut status = lock_unpoison(&self.status);
            *status = SpoutStatus {
                generation,
                resolution_mode,
                ..SpoutStatus::default()
            };
        }

        let mailbox = Arc::new(FrameMailbox::default());
        let worker_mailbox = mailbox.clone();
        let status = self.status.clone();
        let worker_resolution_mode = self.resolution_mode.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("spout-out-{generation}"))
            .spawn(move || {
                let mut sender = match spout2::dx::Sender::new(SENDER_NAME) {
                    Ok(sender) => sender,
                    Err(error) => {
                        update_status(&status, generation, |current| {
                            current.error = format!("spout init: {error}");
                        });
                        return;
                    }
                };
                // Our pixels are RGBA; tell receivers so via DXGI format.
                const DXGI_FORMAT_R8G8B8A8_UNORM: u32 = 28;
                sender.set_format(DXGI_FORMAT_R8G8B8A8_UNORM);

                while let Some(mut frame) = worker_mailbox.recv() {
                    // Avoid work already known to be stale, then scale outside
                    // the barrier. A blackout can therefore supersede a colour
                    // frame while its worker-only resize is in progress.
                    if !worker_mailbox.is_current(&frame) {
                        continue;
                    }
                    let resolution_mode = SpoutResolutionMode::from_atomic(
                        worker_resolution_mode.load(Ordering::Relaxed),
                    );
                    let prepared = match prepare_sender_frame(&mut frame, resolution_mode) {
                        Ok(prepared) => prepared,
                        Err(error) => {
                            update_status(&status, generation, |current| {
                                current.active = false;
                                current.error = format!("spout scale: {error}");
                                current.delivered_black_epoch = None;
                            });
                            continue;
                        }
                    };

                    // Hold through the final validity check and the external
                    // send. `cut_to_black` uses this as its one-time barrier.
                    let _barrier = lock_unpoison(&worker_mailbox.send_barrier);
                    if !worker_mailbox.is_current(&frame) {
                        continue;
                    }
                    match sender.send_image(&prepared.pixels, prepared.width, prepared.height) {
                        Ok(()) => {
                            record_delivery(
                                &status,
                                generation,
                                prepared.width,
                                prepared.height,
                                frame.kind,
                                frame.epoch,
                            );
                        }
                        Err(error) => {
                            update_status(&status, generation, |current| {
                                current.active = false;
                                current.error = format!("spout send: {error}");
                                current.delivered_black_epoch = None;
                            });
                            return;
                        }
                    }
                }
            });

        match spawned {
            Ok(join) => {
                self.worker = Some(WorkerRun { mailbox, join });
                self.retry_after = None;
            }
            Err(error) => {
                update_status(&self.status, generation, |status| {
                    status.error = format!("spout thread: {error}");
                });
                self.retry_after = Some(Instant::now() + RETRY_BACKOFF);
            }
        }
    }

    #[cfg(not(windows))]
    pub fn start(&mut self) {
        let generation = self.next_generation();
        let resolution_mode = self.resolution_mode();
        let mut status = lock_unpoison(&self.status);
        *status = SpoutStatus {
            error: "Spout is Windows-only; Syphon output is not implemented on macOS".to_string(),
            generation,
            resolution_mode,
            ..SpoutStatus::default()
        };
    }

    /// Stop synchronously so a prior generation cannot publish frames or
    /// status after a subsequent restart. This can wait for a single Spout
    /// send, but it is only used on disable/drop, never for steady-state frames.
    pub fn stop(&mut self) {
        let generation = self.next_generation();
        let resolution_mode = self.resolution_mode();
        // Invalidate status first. A send already in progress may finish while
        // we wait for the worker, but its generation-guarded update is ignored.
        {
            let mut status = lock_unpoison(&self.status);
            *status = SpoutStatus {
                generation,
                resolution_mode,
                ..SpoutStatus::default()
            };
        }
        if let Some(worker) = self.worker.take() {
            worker.mailbox.shutdown();
            if worker.join.join().is_err() {
                log::warn!("Spout output worker panicked during shutdown");
            }
        }
        self.retry_after = None;
    }

    fn next_generation(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1).max(1);
        self.generation
    }

    fn reap_finished_worker(&mut self) {
        let finished = self
            .worker
            .as_ref()
            .is_some_and(|worker| worker.join.is_finished());
        if !finished {
            return;
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join.join();
        }
        self.retry_after = Some(Instant::now() + RETRY_BACKOFF);
    }
}

impl Drop for SpoutOut {
    fn drop(&mut self) {
        self.stop();
    }
}

fn lock_unpoison<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(any(windows, test))]
fn update_status(
    status: &Mutex<SpoutStatus>,
    generation: u64,
    update: impl FnOnce(&mut SpoutStatus),
) {
    let mut status = lock_unpoison(status);
    if status.generation == generation {
        update(&mut status);
    }
}

#[cfg(any(windows, test))]
fn record_delivery(
    status: &Mutex<SpoutStatus>,
    generation: u64,
    width: u32,
    height: u32,
    kind: FrameKind,
    epoch: u64,
) {
    update_status(status, generation, |current| {
        if !current.active || current.delivered_width != width || current.delivered_height != height
        {
            log::info!("Spout sender '{SENDER_NAME}' active ({width}x{height})");
        }
        current.active = true;
        current.error.clear();
        current.delivered_width = width;
        current.delivered_height = height;
        current.delivered_black_epoch = match kind {
            FrameKind::Black => Some(epoch),
            FrameKind::Regular => None,
        };
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{channel, TryRecvError};
    use std::time::Duration;

    fn regular(epoch: u64, value: u8) -> QueuedFrame {
        QueuedFrame {
            pixels: vec![value; 4],
            width: 1,
            height: 1,
            epoch,
            kind: FrameKind::Regular,
        }
    }

    #[test]
    fn resolution_mode_keys_are_strict_stable_and_default_to_1080p() {
        assert_eq!(
            SpoutResolutionMode::default(),
            SpoutResolutionMode::Fixed1920x1080
        );
        for mode in [
            SpoutResolutionMode::Native,
            SpoutResolutionMode::Fixed1920x1080,
        ] {
            assert_eq!(SpoutResolutionMode::from_key(mode.key()), Some(mode));
        }
        assert_eq!(SpoutResolutionMode::Native.key(), "native");
        assert_eq!(SpoutResolutionMode::Fixed1920x1080.key(), "1080p");
        assert_eq!(SpoutResolutionMode::from_key("1920x1080"), None);
        assert_eq!(SpoutResolutionMode::from_key("Native"), None);

        let output = SpoutOut::new();
        assert_eq!(
            output.resolution_mode(),
            SpoutResolutionMode::Fixed1920x1080
        );
        assert_eq!(
            output.status().resolution_mode,
            SpoutResolutionMode::Fixed1920x1080
        );
        assert_eq!(
            SpoutResolutionMode::Native.output_dimensions(1280, 720),
            (1280, 720)
        );
        assert_eq!(
            SpoutResolutionMode::Fixed1920x1080.output_dimensions(640, 480),
            (SPOUT_1080P_WIDTH, SPOUT_1080P_HEIGHT)
        );
    }

    #[test]
    fn native_mode_preserves_dimensions_and_pixels_without_scaling() {
        let pixels = vec![
            1, 2, 3, 4, // pixel 0
            5, 6, 7, 8, // pixel 1
        ];
        let mut frame = QueuedFrame {
            pixels: pixels.clone(),
            width: 2,
            height: 1,
            epoch: 17,
            kind: FrameKind::Regular,
        };
        let prepared = prepare_sender_frame(&mut frame, SpoutResolutionMode::Native).unwrap();
        assert_eq!((prepared.width, prepared.height), (2, 1));
        assert_eq!(prepared.pixels, pixels);
        assert!(frame.pixels.is_empty());
        assert_eq!(frame.epoch, 17);
        assert_eq!(frame.kind, FrameKind::Regular);
    }

    #[test]
    fn fixed_mode_scales_regular_and_black_frames_to_exact_1080p() {
        let colour = [0x12, 0x34, 0x56, 0x78];
        let mut regular_frame = QueuedFrame {
            pixels: colour.to_vec(),
            width: 1,
            height: 1,
            epoch: 23,
            kind: FrameKind::Regular,
        };
        let regular =
            prepare_sender_frame(&mut regular_frame, SpoutResolutionMode::Fixed1920x1080).unwrap();
        assert_eq!(
            (regular.width, regular.height),
            (SPOUT_1080P_WIDTH, SPOUT_1080P_HEIGHT)
        );
        assert_eq!(
            regular.pixels.len(),
            SPOUT_1080P_WIDTH as usize * SPOUT_1080P_HEIGHT as usize * 4
        );
        assert!(regular.pixels.chunks_exact(4).all(|pixel| pixel == colour));

        let mailbox = FrameMailbox::default();
        assert!(mailbox.queue_black(2, 2, 24));
        let mut black_frame = mailbox.recv().expect("black frame queued");
        let black =
            prepare_sender_frame(&mut black_frame, SpoutResolutionMode::Fixed1920x1080).unwrap();
        assert_eq!(
            (black.width, black.height),
            (SPOUT_1080P_WIDTH, SPOUT_1080P_HEIGHT)
        );
        assert_eq!(black.pixels.len(), regular.pixels.len());
        assert!(black.pixels.iter().all(|byte| *byte == 0));
        assert_eq!(black_frame.epoch, 24);
        assert_eq!(black_frame.kind, FrameKind::Black);
        assert!(mailbox.is_current(&black_frame));
    }

    #[test]
    fn malformed_frames_are_refused_before_worker_scaling() {
        let mut malformed = QueuedFrame {
            pixels: vec![0; 15],
            width: 2,
            height: 2,
            epoch: 1,
            kind: FrameKind::Regular,
        };
        let error =
            prepare_sender_frame(&mut malformed, SpoutResolutionMode::Fixed1920x1080).unwrap_err();
        assert!(error.contains("expected 16 bytes, got 15"));
        assert_eq!(malformed.pixels.len(), 15);

        malformed.width = 0;
        let error = prepare_sender_frame(&mut malformed, SpoutResolutionMode::Native).unwrap_err();
        assert!(error.contains("zero-sized"));
    }

    #[test]
    fn requested_mode_and_last_delivered_dimensions_are_separate_status_facts() {
        let output = SpoutOut::with_resolution_mode(SpoutResolutionMode::Native);
        output.set_resolution_mode(SpoutResolutionMode::Fixed1920x1080);
        let requested = output.status();
        assert_eq!(
            requested.resolution_mode,
            SpoutResolutionMode::Fixed1920x1080
        );
        assert_eq!(
            (requested.delivered_width, requested.delivered_height),
            (0, 0)
        );

        let generation = requested.generation;
        record_delivery(
            &output.status,
            generation,
            SPOUT_1080P_WIDTH,
            SPOUT_1080P_HEIGHT,
            FrameKind::Regular,
            7,
        );
        let delivered = output.status();
        assert!(delivered.active);
        assert_eq!(
            (delivered.delivered_width, delivered.delivered_height),
            (SPOUT_1080P_WIDTH, SPOUT_1080P_HEIGHT)
        );
        assert_eq!(delivered.delivered_black_epoch, None);

        record_delivery(
            &output.status,
            generation,
            SPOUT_1080P_WIDTH,
            SPOUT_1080P_HEIGHT,
            FrameKind::Black,
            8,
        );
        assert_eq!(output.status().delivered_black_epoch, Some(8));
        record_delivery(
            &output.status,
            generation.wrapping_add(1),
            640,
            480,
            FrameKind::Regular,
            9,
        );
        let after_stale = output.status();
        assert_eq!(
            (after_stale.delivered_width, after_stale.delivered_height),
            (SPOUT_1080P_WIDTH, SPOUT_1080P_HEIGHT)
        );
        assert_eq!(after_stale.delivered_black_epoch, Some(8));
    }

    #[test]
    fn blackout_replaces_pending_colour_and_rejects_same_epoch_colour() {
        let mailbox = FrameMailbox::default();
        assert!(mailbox.try_queue_regular(regular(7, 0x7f)));
        assert!(mailbox.queue_black(1, 1, 8));
        assert!(!mailbox.try_queue_regular(regular(8, 0xff)));

        let frame = mailbox.recv().expect("black frame queued");
        assert_eq!(frame.kind, FrameKind::Black);
        assert_eq!(frame.epoch, 8);
        assert_eq!((frame.width, frame.height), (1, 1));
        assert!(frame.pixels.iter().all(|byte| *byte == 0));
        assert!(mailbox.is_current(&frame));
    }

    #[test]
    fn dequeued_colour_is_invalidated_before_send_by_new_blackout() {
        let mailbox = FrameMailbox::default();
        assert!(mailbox.try_queue_regular(regular(3, 0xaa)));
        let stale = mailbox.recv().expect("colour frame queued");

        assert!(mailbox.queue_black(1, 1, 4));
        assert!(!mailbox.is_current(&stale));
        let black = mailbox.recv().expect("replacement black queued");
        assert_eq!(black.kind, FrameKind::Black);
        assert!(mailbox.is_current(&black));
    }

    #[test]
    fn held_colour_epoch_drops_pending_and_invalidates_dequeued_old_frames() {
        let mailbox = FrameMailbox::default();
        assert!(mailbox.try_queue_regular(regular(6, 0x66)));
        let dequeued = mailbox.recv().expect("old colour frame queued");
        assert!(mailbox.is_current(&dequeued));

        assert!(mailbox.try_queue_regular(regular(6, 0x77)));
        assert!(mailbox.hold_colour_epoch(7));
        assert!(!mailbox.is_current(&dequeued));

        let state = lock_unpoison(&mailbox.state);
        assert_eq!(state.intent.epoch, 7);
        assert!(!state.intent.blackout);
        assert!(state.pending.is_none());
        drop(state);

        assert!(!mailbox.try_queue_regular(regular(6, 0x88)));
        assert!(mailbox.try_queue_regular(regular(7, 0x99)));
        let held = mailbox.recv().expect("held-generation frame queued");
        assert_eq!(held.epoch, 7);
        assert!(mailbox.is_current(&held));
    }

    #[test]
    fn held_colour_epoch_cannot_move_generation_backwards() {
        let mailbox = FrameMailbox::default();
        assert!(mailbox.queue_black(1, 1, 9));
        assert!(!mailbox.hold_colour_epoch(8));
        let state = lock_unpoison(&mailbox.state);
        assert_eq!(state.intent.epoch, 9);
        assert!(state.intent.blackout);
        assert!(matches!(
            state.pending.as_ref().map(|frame| frame.kind),
            Some(FrameKind::Black)
        ));
    }

    #[test]
    fn newer_regular_epoch_releases_blackout() {
        let mailbox = FrameMailbox::default();
        assert!(mailbox.queue_black(1, 1, 4));
        assert!(mailbox.try_queue_regular(regular(5, 0x33)));
        let frame = mailbox.recv().expect("new generation frame queued");
        assert_eq!(frame.kind, FrameKind::Regular);
        assert_eq!(frame.epoch, 5);
        assert!(mailbox.is_current(&frame));
    }

    #[test]
    fn regular_arriving_during_scaling_cannot_starve_delivery_and_remains_latest_pending() {
        let mailbox = FrameMailbox::default();
        assert!(mailbox.try_queue_regular(regular(5, 0x11)));
        let in_flight = mailbox.recv().expect("first colour frame queued");
        assert!(mailbox.is_current(&in_flight));

        assert!(mailbox.try_queue_regular(regular(5, 0x22)));
        assert!(
            mailbox.is_current(&in_flight),
            "a producer faster than worker scaling must not starve every send"
        );
        let latest = mailbox.recv().expect("replacement colour frame queued");
        assert_eq!(latest.pixels, vec![0x22; 4]);
        assert!(mailbox.is_current(&latest));
    }

    #[test]
    fn blackout_waits_for_the_in_flight_send_barrier() {
        let mailbox = Arc::new(FrameMailbox::default());
        let barrier = lock_unpoison(&mailbox.send_barrier);
        let (started_tx, started_rx) = channel();
        let (done_tx, done_rx) = channel();
        let worker_mailbox = mailbox.clone();
        let join = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let queued = worker_mailbox.queue_black(1, 1, 9);
            done_tx.send(queued).unwrap();
        });

        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(done_rx.try_recv(), Err(TryRecvError::Empty));
        drop(barrier);
        assert!(done_rx.recv_timeout(Duration::from_secs(1)).unwrap());
        join.join().unwrap();
    }

    #[test]
    fn stale_worker_cannot_mutate_new_generation_status() {
        let status = Mutex::new(SpoutStatus {
            generation: 12,
            ..SpoutStatus::default()
        });
        update_status(&status, 11, |status| {
            status.active = true;
            status.error = "stale".to_string();
        });
        let current = lock_unpoison(&status).clone();
        assert!(!current.active);
        assert!(current.error.is_empty());

        update_status(&status, 12, |status| status.active = true);
        assert!(lock_unpoison(&status).active);
    }

    #[test]
    fn stop_joins_worker_and_invalidates_its_status_generation() {
        let mut output = SpoutOut::new();
        output.generation = 41;
        *lock_unpoison(&output.status) = SpoutStatus {
            generation: 41,
            active: true,
            ..SpoutStatus::default()
        };

        let mailbox = Arc::new(FrameMailbox::default());
        let worker_mailbox = mailbox.clone();
        let worker_status = output.status.clone();
        let (started_tx, started_rx) = channel();
        let (finished_tx, finished_rx) = channel();
        let join = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            assert!(worker_mailbox.recv().is_none());
            // Deliberately imitate a late old-worker status write.
            update_status(&worker_status, 41, |status| status.active = true);
            finished_tx.send(()).unwrap();
        });
        output.worker = Some(WorkerRun { mailbox, join });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        output.stop();

        assert_eq!(finished_rx.try_recv(), Ok(()), "stop must join the worker");
        assert!(output.worker.is_none());
        let status = output.status();
        assert_eq!(status.generation, 42);
        assert!(!status.active);
        assert!(status.error.is_empty());
    }
}
