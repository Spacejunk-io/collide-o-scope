//! Generation-safe, request-driven wrapper around [`VideoDecoder`].
//!
//! Commands occupy one overwrite slot. Absolute source-time selection always
//! supersedes queued or in-flight work. The completed-frame mailbox is also
//! latest-only.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::media_safety::{
    MediaAllocationPlan, MediaDeviceLimits, MediaSafetyPolicy, MediaSourceKind,
    PERFORMANCE_MAX_PREPARED_SOURCES, PERFORMANCE_STAGING_WORKERS,
};

#[cfg(test)]
use super::decoder::validate_media_dimensions;
use super::decoder::validate_media_dimensions_with_policy as plan_media_dimensions;
use super::decoder::KeyframeIndexBuildRequest;
use super::indexed::KeyframeIndex;
use super::{CodecMotionFrame, DecodeWorkError, DecodedVideoFrame, FrameMetadata, VideoDecoder};

const DECODER_OPEN_TIMEOUT: Duration = Duration::from_secs(7);
pub const DECODER_TELEMETRY_WINDOW_SAMPLES: usize = 64;

#[derive(Debug, Clone)]
struct DurationWindow {
    nanoseconds: [u64; DECODER_TELEMETRY_WINDOW_SAMPLES],
    next: usize,
    len: usize,
    total_samples: u64,
    last_nanoseconds: Option<u64>,
    peak_nanoseconds: Option<u64>,
}

impl Default for DurationWindow {
    fn default() -> Self {
        Self {
            nanoseconds: [0; DECODER_TELEMETRY_WINDOW_SAMPLES],
            next: 0,
            len: 0,
            total_samples: 0,
            last_nanoseconds: None,
            peak_nanoseconds: None,
        }
    }
}

impl DurationWindow {
    fn record(&mut self, duration: Duration) {
        let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.nanoseconds[self.next] = nanoseconds;
        self.next = (self.next + 1) % DECODER_TELEMETRY_WINDOW_SAMPLES;
        self.len = self
            .len
            .saturating_add(1)
            .min(DECODER_TELEMETRY_WINDOW_SAMPLES);
        self.total_samples = self.total_samples.saturating_add(1);
        self.last_nanoseconds = Some(nanoseconds);
        self.peak_nanoseconds = Some(
            self.peak_nanoseconds
                .map_or(nanoseconds, |peak| peak.max(nanoseconds)),
        );
    }

    fn last(&self) -> Option<Duration> {
        self.last_nanoseconds.map(Duration::from_nanos)
    }

    fn peak(&self) -> Option<Duration> {
        self.peak_nanoseconds.map(Duration::from_nanos)
    }

    fn p95(&self) -> Option<Duration> {
        if self.len == 0 {
            return None;
        }
        let mut sorted = self.nanoseconds;
        sorted[..self.len].sort_unstable();
        let nearest_rank = (95 * self.len).div_ceil(100);
        Some(Duration::from_nanos(sorted[nearest_rank - 1]))
    }
}

struct KeyframeIndexJob {
    request: KeyframeIndexBuildRequest,
    result_tx: SyncSender<Result<KeyframeIndex, String>>,
}

fn keyframe_index_pool() -> Result<&'static SyncSender<KeyframeIndexJob>, String> {
    static POOL: OnceLock<Result<SyncSender<KeyframeIndexJob>, String>> = OnceLock::new();
    POOL.get_or_init(|| {
        let (job_tx, job_rx) =
            std::sync::mpsc::sync_channel::<KeyframeIndexJob>(PERFORMANCE_MAX_PREPARED_SOURCES);
        let job_rx = Arc::new(Mutex::new(job_rx));
        for worker_index in 0..PERFORMANCE_STAGING_WORKERS {
            let worker_rx = job_rx.clone();
            std::thread::Builder::new()
                .name(format!("keyframe-index-{worker_index}"))
                .spawn(move || loop {
                    let job = {
                        let receiver = lock_recover(&worker_rx);
                        receiver.recv()
                    };
                    let Ok(job) = job else {
                        return;
                    };
                    let _ = job.result_tx.send(job.request.build());
                })
                .map_err(|error| format!("failed to spawn keyframe index worker: {error}"))?;
        }
        Ok(job_tx)
    })
    .as_ref()
    .map_err(Clone::clone)
}

fn submit_keyframe_index(
    request: KeyframeIndexBuildRequest,
) -> Result<Receiver<Result<KeyframeIndex, String>>, String> {
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
    match keyframe_index_pool()?.try_send(KeyframeIndexJob { request, result_tx }) {
        Ok(()) => Ok(result_rx),
        Err(TrySendError::Full(_)) => Err(format!(
            "keyframe index staging queue reached its {}-source bound",
            PERFORMANCE_MAX_PREPARED_SOURCES
        )),
        Err(TrySendError::Disconnected(_)) => {
            Err("keyframe index worker pool disconnected".to_string())
        }
    }
}

fn install_ready_keyframe_index(
    decoder: &std::cell::RefCell<VideoDecoder>,
    pending: &std::cell::RefCell<Option<Receiver<Result<KeyframeIndex, String>>>>,
    path: &str,
) {
    let result = {
        let pending = pending.borrow();
        pending.as_ref().map(Receiver::try_recv)
    };
    match result {
        Some(Ok(Ok(index))) => {
            decoder.borrow_mut().install_keyframe_index(index);
            *pending.borrow_mut() = None;
        }
        Some(Ok(Err(error))) => {
            log::warn!("keyframe index unavailable for {path}: {error}");
            *pending.borrow_mut() = None;
        }
        Some(Err(TryRecvError::Disconnected)) => {
            log::warn!("keyframe index worker disconnected for {path}");
            *pending.borrow_mut() = None;
        }
        Some(Err(TryRecvError::Empty)) | None => {}
    }
}

#[derive(Debug)]
pub struct DecodedFrame {
    pub rgba: Vec<u8>,
    pub codec_motion: Option<CodecMotionFrame>,
    /// Loop progress 0.0..1.0 at the time this frame was decoded.
    pub progress: f32,
    /// Cumulative successful EOF reopens at the time this frame was decoded.
    pub loop_generation: u64,
    pub source_generation: u64,
    pub pts: Option<i64>,
    pub source_seconds: f64,
    pub duration_seconds: f64,
    command_epoch: u64,
}

impl DecodedFrame {
    fn from_video(
        frame: DecodedVideoFrame,
        progress: f32,
        loop_generation: u64,
        command_epoch: u64,
    ) -> Self {
        let DecodedVideoFrame {
            rgba,
            metadata,
            codec_motion,
        } = frame;
        let FrameMetadata {
            source_generation,
            pts,
            source_seconds,
            duration_seconds,
        } = metadata;
        let codec_motion =
            codec_motion.filter(|motion| motion.source_generation == source_generation);
        Self {
            rgba,
            codec_motion,
            progress,
            loop_generation,
            source_generation,
            pts,
            source_seconds,
            duration_seconds,
            command_epoch,
        }
    }

    #[cfg(test)]
    fn synthetic(value: u8, source_generation: u64, loop_generation: u64) -> Self {
        Self {
            rgba: vec![value],
            codec_motion: None,
            progress: f32::from(value) / 255.0,
            loop_generation,
            source_generation,
            pts: Some(i64::from(value)),
            source_seconds: f64::from(value) / 30.0,
            duration_seconds: 10.0,
            command_epoch: 1,
        }
    }
}

/// Newest completed media frame plus source-time and loop metadata.
#[derive(Debug, PartialEq)]
pub struct ReadyFrame {
    pub rgba: Vec<u8>,
    pub codec_motion: Option<CodecMotionFrame>,
    pub loops_advanced: u64,
    pub source_generation: u64,
    pub pts: Option<i64>,
    pub source_seconds: f64,
    pub duration_seconds: f64,
}

impl ReadyFrame {
    pub fn still(rgba: Vec<u8>) -> Self {
        let metadata = FrameMetadata::still();
        Self {
            rgba,
            codec_motion: None,
            loops_advanced: 0,
            source_generation: metadata.source_generation,
            pts: metadata.pts,
            source_seconds: metadata.source_seconds,
            duration_seconds: metadata.duration_seconds,
        }
    }
}

/// Stable decoder state for snapshots and operator-facing health reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderHealth {
    Healthy,
    Failed(String),
}

/// Allocation-free, bounded truth about the newest-only decode pipeline.
/// Ages are sampled by the caller and therefore continue to grow even when
/// no worker activity occurs. Upload duration is CPU wall time around the
/// validated queue upload/poll seam; it is not claimed to be a GPU timestamp.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecoderTelemetry {
    pub last_publish_age: Option<Duration>,
    pub last_consume_age: Option<Duration>,
    pub pending_command_depth: u8,
    pub pending_frames: u8,
    pub pending_frames_peak: u8,
    pub published_frames: u64,
    pub consumed_frames: u64,
    pub command_overwrites: u64,
    pub command_drops: u64,
    pub frame_overwrites: u64,
    pub frame_drops: u64,
    pub decode_samples: u64,
    pub last_decode_duration: Option<Duration>,
    pub peak_decode_duration: Option<Duration>,
    pub decode_p95_duration: Option<Duration>,
    pub frame_age_p95_duration: Option<Duration>,
    pub upload_samples: u64,
    pub last_upload_duration: Option<Duration>,
    pub peak_upload_duration: Option<Duration>,
    pub upload_p95_duration: Option<Duration>,
}

#[derive(Debug)]
struct SharedState {
    latest: Option<DecodedFrame>,
    last_published_at: Option<Instant>,
    frame_overwrites: u64,
    frame_drops: u64,
    published_frames: u64,
    pending_frames_peak: u8,
    decode_durations: DurationWindow,
    frame_age_durations: DurationWindow,
    upload_durations: DurationWindow,
    command_error: Option<(u64, u64, String)>,
    health: DecoderHealth,
    health_revision: u64,
}

impl SharedState {
    fn healthy_with_seed(seed: DecodedFrame) -> Self {
        Self::healthy_with_seed_at(seed, Instant::now())
    }

    fn healthy_with_seed_at(seed: DecodedFrame, published_at: Instant) -> Self {
        Self {
            latest: Some(seed),
            last_published_at: Some(published_at),
            frame_overwrites: 0,
            frame_drops: 0,
            published_frames: 1,
            pending_frames_peak: 1,
            decode_durations: DurationWindow::default(),
            frame_age_durations: DurationWindow::default(),
            upload_durations: DurationWindow::default(),
            command_error: None,
            health: DecoderHealth::Healthy,
            health_revision: 1,
        }
    }

    fn publish(&mut self, frame: DecodedFrame, published_at: Instant, decode_duration: Duration) {
        if self.latest.replace(frame).is_some() {
            self.frame_overwrites = self.frame_overwrites.saturating_add(1);
        }
        self.last_published_at = Some(published_at);
        self.published_frames = self.published_frames.saturating_add(1);
        self.pending_frames_peak = 1;
        self.decode_durations.record(decode_duration);
        self.command_error = None;
    }

    fn record_seed_decode(&mut self, decode_duration: Duration) {
        self.decode_durations.record(decode_duration);
    }

    fn drop_frame(&mut self) {
        self.frame_drops = self.frame_drops.saturating_add(1);
    }

    fn set_failed(&mut self, error: String) {
        if matches!(self.health, DecoderHealth::Healthy) {
            self.health = DecoderHealth::Failed(error);
            self.health_revision = self.health_revision.saturating_add(1);
        }
    }
}

/// Commands accepted by the one-slot decoder mailbox.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecodeCommand {
    Select {
        generation: u64,
        target_seconds: f64,
    },
    Stop,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct QueuedCommand {
    command: DecodeCommand,
    epoch: u64,
}

#[derive(Debug)]
struct CommandMailboxState {
    pending: Option<QueuedCommand>,
    epoch: u64,
    generation: u64,
    stopped: bool,
}

#[derive(Debug)]
struct DecodeCommandMailbox {
    state: Mutex<CommandMailboxState>,
    wake: Condvar,
    current_epoch: AtomicU64,
    current_generation: AtomicU64,
    stopped: AtomicBool,
    command_overwrites: AtomicU64,
    command_drops: AtomicU64,
}

impl DecodeCommandMailbox {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CommandMailboxState {
                pending: None,
                epoch: 1,
                generation: 0,
                stopped: false,
            }),
            wake: Condvar::new(),
            current_epoch: AtomicU64::new(1),
            current_generation: AtomicU64::new(0),
            stopped: AtomicBool::new(false),
            command_overwrites: AtomicU64::new(0),
            command_drops: AtomicU64::new(0),
        })
    }

    fn select(&self, generation: u64, target_seconds: f64) -> Result<bool, String> {
        if !target_seconds.is_finite() {
            return Err(format!(
                "source-time selection must be finite, got {target_seconds}"
            ));
        }
        let mut state = lock_recover(&self.state);
        if state.stopped || generation < state.generation {
            self.command_drops.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let epoch = state
            .epoch
            .checked_add(1)
            .ok_or_else(|| "decoder command epoch exhausted".to_string())?;
        state.epoch = epoch;
        state.generation = generation;
        let replaced = state.pending.replace(QueuedCommand {
            command: DecodeCommand::Select {
                generation,
                target_seconds,
            },
            epoch,
        });
        if replaced.is_some() {
            self.command_overwrites.fetch_add(1, Ordering::Relaxed);
        }
        self.current_generation.store(generation, Ordering::Release);
        self.current_epoch.store(epoch, Ordering::Release);
        drop(state);
        self.wake.notify_one();
        Ok(true)
    }

    fn stop(&self) {
        let mut state = lock_recover(&self.state);
        if state.stopped {
            return;
        }
        state.stopped = true;
        state.epoch = state.epoch.saturating_add(1);
        let epoch = state.epoch;
        let replaced = state.pending.replace(QueuedCommand {
            command: DecodeCommand::Stop,
            epoch,
        });
        if replaced.is_some() {
            self.command_overwrites.fetch_add(1, Ordering::Relaxed);
        }
        self.stopped.store(true, Ordering::Release);
        self.current_epoch.store(epoch, Ordering::Release);
        drop(state);
        self.wake.notify_one();
    }

    fn recv(&self) -> QueuedCommand {
        let mut state = lock_recover(&self.state);
        loop {
            if let Some(command) = state.pending.take() {
                return command;
            }
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn is_current(&self, epoch: u64) -> bool {
        !self.stopped.load(Ordering::Acquire) && self.current_epoch.load(Ordering::Acquire) == epoch
    }

    fn generation(&self) -> u64 {
        self.current_generation.load(Ordering::Acquire)
    }

    fn epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::Acquire)
    }

    fn pending_depth(&self) -> usize {
        usize::from(lock_recover(&self.state).pending.is_some())
    }
}

pub struct ThreadedDecoder {
    mailbox: Arc<DecodeCommandMailbox>,
    cancel: Arc<AtomicBool>,
    shared: Arc<Mutex<SharedState>>,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    #[allow(dead_code)]
    pub duration_seconds: f64,
    media_plan: MediaAllocationPlan,
    progress: f32,
    consumed_loop_generation: u64,
    consumed_source_generation: u64,
    failure_revision_reported: u64,
    last_consumed_at: Option<Instant>,
    consumed_frames: u64,
}

impl ThreadedDecoder {
    #[allow(dead_code)]
    pub fn open_with_texture_limit(path: &str, max_dimension: u32) -> Result<Self, String> {
        Self::open_inner(
            path,
            MediaSafetyPolicy::safe(),
            MediaDeviceLimits::texture_only(max_dimension),
        )
    }

    pub fn open_with_media_policy(
        path: &str,
        media_policy: &MediaSafetyPolicy,
        device_limits: MediaDeviceLimits,
    ) -> Result<Self, String> {
        Self::open_inner(path, media_policy.clone(), device_limits)
    }

    fn open_inner(
        path: &str,
        media_policy: MediaSafetyPolicy,
        device_limits: MediaDeviceLimits,
    ) -> Result<Self, String> {
        let mailbox = DecodeCommandMailbox::new();
        let worker_mailbox = mailbox.clone();
        let (meta_tx, meta_rx) =
            std::sync::mpsc::channel::<Result<(u32, u32, f32, f64, MediaAllocationPlan), String>>();

        let thread_name = format!("decode-{}", short_name(path));
        let path_owned = path.to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let shared = Arc::new(Mutex::new(SharedState {
            latest: None,
            last_published_at: None,
            frame_overwrites: 0,
            frame_drops: 0,
            published_frames: 0,
            pending_frames_peak: 0,
            decode_durations: DurationWindow::default(),
            frame_age_durations: DurationWindow::default(),
            upload_durations: DurationWindow::default(),
            command_error: None,
            health: DecoderHealth::Healthy,
            health_revision: 1,
        }));
        let worker_shared = shared.clone();
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let panic_shared = worker_shared.clone();
                let worker = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut decoder = match VideoDecoder::open_with_cancel_and_media_policy(
                        &path_owned,
                        worker_cancel,
                        &media_policy,
                        device_limits,
                    ) {
                        Ok(decoder) => decoder,
                        Err(error) => {
                            let _ = meta_tx.send(Err(error));
                            return;
                        }
                    };

                    let seed_decode_started = Instant::now();
                    let seed = match decoder.next_timed_frame_result(0) {
                        Ok(frame) => DecodedFrame::from_video(
                            frame,
                            decoder.progress(),
                            decoder.loop_generation(),
                            worker_mailbox.epoch(),
                        ),
                        Err(error) => {
                            let _ = meta_tx.send(Err(format!(
                                "failed to decode initial video frame: {error}"
                            )));
                            return;
                        }
                    };
                    let seed_decode_duration = seed_decode_started.elapsed();
                    let dimensions = (
                        decoder.width,
                        decoder.height,
                        decoder.fps,
                        decoder.duration_seconds(),
                        decoder.media_allocation_plan().clone(),
                    );
                    let mut seeded = SharedState::healthy_with_seed(seed);
                    seeded.record_seed_decode(seed_decode_duration);
                    *lock_state(&worker_shared) = seeded;
                    if meta_tx.send(Ok(dimensions)).is_err() {
                        return;
                    }

                    // A bounded helper pool scans a second demuxer. Ordinary
                    // Select commands continue immediately against the safe
                    // fallback entry and install the result only
                    // when it is already ready.
                    let index_result =
                        match submit_keyframe_index(decoder.keyframe_index_build_request()) {
                            Ok(receiver) => Some(receiver),
                            Err(error) => {
                                log::warn!(
                                    "keyframe index staging skipped for {path_owned}: {error}"
                                );
                                None
                            }
                        };
                    let decoder = std::cell::RefCell::new(decoder);
                    let index_result = std::cell::RefCell::new(index_result);
                    let select_mailbox = worker_mailbox.clone();
                    run_decode_commands(
                        worker_mailbox,
                        worker_shared,
                        |generation, target_seconds, epoch| {
                            install_ready_keyframe_index(&decoder, &index_result, &path_owned);
                            let mut decoder = decoder.borrow_mut();
                            let frame = decoder.seek_decode_interruptible(
                                target_seconds,
                                generation,
                                || select_mailbox.is_current(epoch),
                            )?;
                            Ok(DecodedFrame::from_video(
                                frame,
                                decoder.progress(),
                                decoder.loop_generation(),
                                0,
                            ))
                        },
                    );
                }));

                if worker.is_err() {
                    lock_state(&panic_shared)
                        .set_failed("Decode worker panicked unexpectedly".to_string());
                }
            })
            .map_err(|error| format!("Failed to spawn decode thread: {error}"))?;

        let (width, height, fps, duration_seconds, media_plan) =
            match meta_rx.recv_timeout(DECODER_OPEN_TIMEOUT) {
                Ok(result) => result?,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    cancel.store(true, Ordering::Release);
                    mailbox.stop();
                    return Err(format!(
                        "video decoder open timed out after {} seconds for {path}",
                        DECODER_OPEN_TIMEOUT.as_secs()
                    ));
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(format!("decode thread died while opening {path}"));
                }
            };

        Ok(Self {
            mailbox,
            cancel,
            shared,
            width,
            height,
            fps,
            duration_seconds,
            media_plan,
            progress: 0.0,
            consumed_loop_generation: 0,
            consumed_source_generation: 0,
            failure_revision_reported: 0,
            last_consumed_at: None,
            consumed_frames: 0,
        })
    }

    /// Queue an absolute source-time selection. A generation older than the
    /// newest selection is rejected without disturbing current playback.
    pub fn request_source_time(
        &mut self,
        generation: u64,
        target_seconds: f64,
    ) -> Result<bool, String> {
        if let Some(error) = self.terminal_error_once() {
            return Err(error);
        }
        self.mailbox.select(generation, target_seconds)
    }

    /// Compatibility convenience for callers without their own source
    /// generation counter. The returned generation identifies the request.
    #[allow(dead_code)]
    pub fn request_seek(&mut self, target_seconds: f64) -> Result<u64, String> {
        let generation = self
            .mailbox
            .generation()
            .checked_add(1)
            .ok_or_else(|| "source generation exhausted".to_string())?;
        self.request_source_time(generation, target_seconds)?;
        Ok(generation)
    }

    #[allow(dead_code)]
    pub fn source_generation(&self) -> u64 {
        self.mailbox.generation()
    }

    pub fn try_next_ready_frame_result(&mut self) -> Result<Option<ReadyFrame>, String> {
        self.try_next_ready_frame_result_at(Instant::now())
    }

    fn try_next_ready_frame_result_at(
        &mut self,
        consumed_at: Instant,
    ) -> Result<Option<ReadyFrame>, String> {
        let current_generation = self.mailbox.generation();
        let current_epoch = self.mailbox.epoch();
        let mut state = lock_state(&self.shared);
        if state.latest.as_ref().is_some_and(|frame| {
            frame.source_generation != current_generation || frame.command_epoch != current_epoch
        }) {
            state.latest = None;
            state.drop_frame();
        }
        if state
            .command_error
            .as_ref()
            .is_some_and(|(generation, epoch, _)| {
                *generation != current_generation || *epoch != current_epoch
            })
        {
            state.command_error = None;
        }
        if let Some(frame) = state.latest.take() {
            if let Some(published_at) = state.last_published_at {
                state
                    .frame_age_durations
                    .record(duration_since_or_zero(consumed_at, published_at));
            }
            drop(state);
            self.last_consumed_at = Some(consumed_at);
            self.consumed_frames = self.consumed_frames.saturating_add(1);
            self.progress = frame.progress;
            let loops_advanced = if frame.source_generation != self.consumed_source_generation {
                self.consumed_source_generation = frame.source_generation;
                self.consumed_loop_generation = frame.loop_generation;
                0
            } else {
                let advanced = frame
                    .loop_generation
                    .saturating_sub(self.consumed_loop_generation);
                self.consumed_loop_generation =
                    self.consumed_loop_generation.max(frame.loop_generation);
                advanced
            };
            return Ok(Some(ReadyFrame {
                rgba: frame.rgba,
                codec_motion: frame.codec_motion,
                loops_advanced,
                source_generation: frame.source_generation,
                pts: frame.pts,
                source_seconds: frame.source_seconds,
                duration_seconds: frame.duration_seconds,
            }));
        }
        if let Some((_, _, error)) = state.command_error.take() {
            return Err(error);
        }
        drop(state);
        if let Some(error) = self.terminal_error_once() {
            return Err(error);
        }
        Ok(None)
    }

    #[allow(dead_code)]
    pub fn try_next_frame_result(&mut self) -> Result<Option<Vec<u8>>, String> {
        self.try_next_ready_frame_result()
            .map(|ready| ready.map(|frame| frame.rgba))
    }

    pub fn progress(&self) -> f32 {
        self.progress
    }

    pub fn media_allocation_plan(&self) -> &MediaAllocationPlan {
        &self.media_plan
    }

    pub fn health(&self) -> DecoderHealth {
        lock_state(&self.shared).health.clone()
    }

    /// Snapshot decoder timing and newest-only mailbox loss without allocating
    /// or disturbing either queue.
    pub fn telemetry(&self) -> DecoderTelemetry {
        self.telemetry_at(Instant::now())
    }

    fn telemetry_at(&self, sampled_at: Instant) -> DecoderTelemetry {
        let pending_command_depth = u8::try_from(self.mailbox.pending_depth()).unwrap_or(u8::MAX);
        let command_overwrites = self.mailbox.command_overwrites.load(Ordering::Relaxed);
        let command_drops = self.mailbox.command_drops.load(Ordering::Relaxed);
        let state = lock_state(&self.shared);
        DecoderTelemetry {
            last_publish_age: state
                .last_published_at
                .map(|published_at| duration_since_or_zero(sampled_at, published_at)),
            last_consume_age: self
                .last_consumed_at
                .map(|consumed_at| duration_since_or_zero(sampled_at, consumed_at)),
            pending_command_depth,
            pending_frames: u8::from(state.latest.is_some()),
            pending_frames_peak: state.pending_frames_peak,
            published_frames: state.published_frames,
            consumed_frames: self.consumed_frames,
            command_overwrites,
            command_drops,
            frame_overwrites: state.frame_overwrites,
            frame_drops: state.frame_drops,
            decode_samples: state.decode_durations.total_samples,
            last_decode_duration: state.decode_durations.last(),
            peak_decode_duration: state.decode_durations.peak(),
            decode_p95_duration: state.decode_durations.p95(),
            frame_age_p95_duration: state.frame_age_durations.p95(),
            upload_samples: state.upload_durations.total_samples,
            last_upload_duration: state.upload_durations.last(),
            peak_upload_duration: state.upload_durations.peak(),
            upload_p95_duration: state.upload_durations.p95(),
        }
    }

    /// Record one successful validated CPU-to-GPU upload. The caller measures
    /// the complete queue-write/error-scope wall interval; no staging buffer or
    /// timestamp query is allocated for telemetry.
    pub fn record_upload_duration(&mut self, duration: Duration) {
        lock_state(&self.shared).upload_durations.record(duration);
    }

    fn terminal_error_once(&mut self) -> Option<String> {
        let state = lock_state(&self.shared);
        let DecoderHealth::Failed(error) = &state.health else {
            return None;
        };
        if state.health_revision == self.failure_revision_reported {
            return None;
        }
        self.failure_revision_reported = state.health_revision;
        Some(error.clone())
    }
}

impl Drop for ThreadedDecoder {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
        self.mailbox.stop();
    }
}

#[cfg(test)]
fn validate_decode_dimensions(
    width: u32,
    height: u32,
    max_dimension: Option<u32>,
) -> Result<(), String> {
    validate_media_dimensions(width, height, max_dimension)?;
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn validate_decode_dimensions_with_media_policy(
    width: u32,
    height: u32,
    media_policy: &MediaSafetyPolicy,
    device_limits: MediaDeviceLimits,
) -> Result<MediaAllocationPlan, String> {
    plan_media_dimensions(
        width,
        height,
        MediaSourceKind::Video,
        media_policy,
        device_limits,
    )
}

fn run_decode_commands<S>(
    mailbox: Arc<DecodeCommandMailbox>,
    shared: Arc<Mutex<SharedState>>,
    mut select_frame: S,
) where
    S: FnMut(u64, f64, u64) -> Result<DecodedFrame, DecodeWorkError>,
{
    loop {
        let queued = mailbox.recv();
        if matches!(queued.command, DecodeCommand::Stop) {
            return;
        }
        if !mailbox.is_current(queued.epoch) {
            lock_state(&shared).drop_frame();
            continue;
        }

        let (generation, result, decode_duration) = match queued.command {
            DecodeCommand::Select {
                generation,
                target_seconds,
            } => {
                let decode_started = Instant::now();
                let result = select_frame(generation, target_seconds, queued.epoch);
                (generation, result, decode_started.elapsed())
            }
            DecodeCommand::Stop => unreachable!("stop returned above"),
        };

        if !mailbox.is_current(queued.epoch) {
            lock_state(&shared).drop_frame();
            continue;
        }
        let mut state = lock_state(&shared);
        match result {
            Ok(mut frame) => {
                frame.source_generation = generation;
                if let Some(codec_motion) = frame.codec_motion.as_mut() {
                    codec_motion.retag_source_generation(generation);
                }
                frame.command_epoch = queued.epoch;
                state.publish(frame, Instant::now(), decode_duration);
            }
            Err(DecodeWorkError::Failed(error)) => {
                state.command_error = Some((generation, queued.epoch, error));
            }
            Err(DecodeWorkError::Superseded) => state.drop_frame(),
        }
    }
}

fn duration_since_or_zero(later: Instant, earlier: Instant) -> Duration {
    later.checked_duration_since(earlier).unwrap_or_default()
}

fn lock_state(state: &Arc<Mutex<SharedState>>) -> MutexGuard<'_, SharedState> {
    lock_recover(state)
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn short_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .map(|name| name.to_string_lossy().chars().take(12).collect())
        .unwrap_or_else(|| "video".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    fn synthetic_codec_motion(source_generation: u64, frame_ordinal: u64) -> CodecMotionFrame {
        CodecMotionFrame {
            source_dimensions: [1, 1],
            frame_delta_seconds: 1.0 / 30.0,
            source_generation,
            frame_ordinal,
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            provenance: crate::video::CodecMotionProvenance::FfmpegExportMvs,
            frame_type: crate::video::CodecMotionFrameType::Intra,
            status: crate::video::CodecMotionStatus::Intra,
            vectors: Vec::new(),
        }
    }

    fn synthetic_frame_with_motion(value: u8, source_generation: u64) -> DecodedFrame {
        let mut frame = DecodedFrame::synthetic(value, source_generation, 0);
        frame.codec_motion = Some(synthetic_codec_motion(source_generation, u64::from(value)));
        frame
    }

    fn synthetic_decoder(seed: DecodedFrame) -> ThreadedDecoder {
        synthetic_decoder_at(seed, Instant::now())
    }

    fn synthetic_decoder_at(seed: DecodedFrame, published_at: Instant) -> ThreadedDecoder {
        let mailbox = DecodeCommandMailbox::new();
        ThreadedDecoder {
            mailbox,
            cancel: Arc::new(AtomicBool::new(false)),
            shared: Arc::new(Mutex::new(SharedState::healthy_with_seed_at(
                seed,
                published_at,
            ))),
            width: 1,
            height: 1,
            fps: 30.0,
            duration_seconds: 10.0,
            media_plan: crate::media_safety::validate_safe_dimensions(
                MediaSourceKind::Video,
                1,
                1,
                MediaDeviceLimits::none(),
            )
            .unwrap(),
            progress: 0.0,
            consumed_loop_generation: 0,
            consumed_source_generation: 0,
            failure_revision_reported: 0,
            last_consumed_at: None,
            consumed_frames: 0,
        }
    }

    #[test]
    fn ten_thousand_scrub_requests_keep_depth_one_and_newest_wins() {
        let mailbox = DecodeCommandMailbox::new();
        for generation in 1..=10_000 {
            assert!(mailbox
                .select(generation, generation as f64 / 60.0)
                .unwrap());
            assert_eq!(mailbox.pending_depth(), 1);
        }
        assert_eq!(
            mailbox.recv().command,
            DecodeCommand::Select {
                generation: 10_000,
                target_seconds: 10_000.0 / 60.0,
            }
        );
        assert_eq!(mailbox.command_overwrites.load(Ordering::Relaxed), 9_999);
    }

    #[test]
    fn telemetry_ages_grow_from_exact_publish_and_consume_boundaries() {
        let published_at = Instant::now();
        let mut decoder = synthetic_decoder_at(DecodedFrame::synthetic(1, 0, 0), published_at);

        let before_consume = decoder.telemetry_at(published_at + Duration::from_millis(40));
        assert_eq!(
            before_consume.last_publish_age,
            Some(Duration::from_millis(40))
        );
        assert_eq!(before_consume.last_consume_age, None);
        assert_eq!(before_consume.pending_frames, 1);
        assert_eq!(before_consume.pending_frames_peak, 1);
        assert_eq!(
            (
                before_consume.published_frames,
                before_consume.consumed_frames
            ),
            (1, 0)
        );

        let consumed_at = published_at + Duration::from_millis(60);
        assert!(decoder
            .try_next_ready_frame_result_at(consumed_at)
            .unwrap()
            .is_some());
        let later = decoder.telemetry_at(published_at + Duration::from_millis(95));
        assert_eq!(later.last_publish_age, Some(Duration::from_millis(95)));
        assert_eq!(later.last_consume_age, Some(Duration::from_millis(35)));
        assert_eq!(later.pending_frames, 0);
        assert_eq!((later.published_frames, later.consumed_frames), (1, 1));
        assert_eq!(
            later.frame_age_p95_duration,
            Some(Duration::from_millis(60))
        );
    }

    #[test]
    fn telemetry_reports_pending_overwrite_stale_drop_and_upload_samples() {
        let published_at = Instant::now();
        let mut decoder = synthetic_decoder_at(DecodedFrame::synthetic(1, 0, 0), published_at);
        assert!(decoder.mailbox.select(1, 1.0).unwrap());
        assert!(decoder.mailbox.select(1, 2.0).unwrap());
        assert!(!decoder.mailbox.select(0, 3.0).unwrap());

        let queued = decoder.telemetry_at(published_at);
        assert_eq!(queued.pending_command_depth, 1);
        assert_eq!(queued.pending_frames, 1);
        assert_eq!(queued.command_overwrites, 1);
        assert_eq!(queued.command_drops, 1);
        assert_eq!(queued.frame_drops, 0);

        // The request changed generation/epoch before the seed was consumed,
        // so harvesting rejects that completed product without exposing it.
        assert_eq!(
            decoder.try_next_ready_frame_result_at(published_at),
            Ok(None)
        );
        decoder.record_upload_duration(Duration::from_micros(800));
        decoder.record_upload_duration(Duration::from_micros(250));
        let dropped = decoder.telemetry_at(published_at);
        assert_eq!(dropped.frame_drops, 1);
        assert_eq!((dropped.published_frames, dropped.consumed_frames), (1, 0));
        assert_eq!(dropped.upload_samples, 2);
        assert_eq!(
            dropped.last_upload_duration,
            Some(Duration::from_micros(250))
        );
        assert_eq!(
            dropped.peak_upload_duration,
            Some(Duration::from_micros(800))
        );
        assert_eq!(
            dropped.upload_p95_duration,
            Some(Duration::from_micros(800))
        );
    }

    #[test]
    fn completed_mailbox_overwrite_is_counted_without_growing_depth() {
        let published_at = Instant::now();
        let decoder = synthetic_decoder_at(DecodedFrame::synthetic(1, 0, 0), published_at);
        lock_state(&decoder.shared).publish(
            DecodedFrame::synthetic(2, 0, 0),
            published_at + Duration::from_millis(2),
            Duration::from_micros(400),
        );
        let telemetry = decoder.telemetry_at(published_at + Duration::from_millis(3));
        assert_eq!(telemetry.frame_overwrites, 1);
        assert_eq!(telemetry.pending_frames, 1);
        assert_eq!(telemetry.published_frames, 2);
        assert_eq!(telemetry.decode_samples, 1);
        assert_eq!(
            telemetry.last_decode_duration,
            Some(Duration::from_micros(400))
        );
        assert_eq!(
            telemetry.decode_p95_duration,
            Some(Duration::from_micros(400))
        );
        assert_eq!(telemetry.pending_command_depth, 0);
        assert_eq!(telemetry.last_publish_age, Some(Duration::from_millis(1)));
    }

    #[test]
    fn duration_window_is_fixed_and_uses_nearest_rank_p95() {
        let mut window = DurationWindow::default();
        for microseconds in 1..=DECODER_TELEMETRY_WINDOW_SAMPLES as u64 {
            window.record(Duration::from_micros(microseconds));
        }
        assert_eq!(window.total_samples, 64);
        assert_eq!(window.len, DECODER_TELEMETRY_WINDOW_SAMPLES);
        assert_eq!(window.p95(), Some(Duration::from_micros(61)));
        window.record(Duration::from_micros(65));
        assert_eq!(window.total_samples, 65);
        assert_eq!(window.len, DECODER_TELEMETRY_WINDOW_SAMPLES);
        assert_eq!(window.last(), Some(Duration::from_micros(65)));
        assert_eq!(window.peak(), Some(Duration::from_micros(65)));
        assert_eq!(window.p95(), Some(Duration::from_micros(62)));
    }

    #[test]
    fn epoch_exhaustion_is_recoverable_and_stop_still_invalidates_work() {
        let mailbox = DecodeCommandMailbox::new();
        {
            let mut state = lock_recover(&mailbox.state);
            state.epoch = u64::MAX;
        }
        mailbox.current_epoch.store(u64::MAX, Ordering::Release);
        assert_eq!(
            mailbox.select(1, 0.0).unwrap_err(),
            "decoder command epoch exhausted"
        );
        assert!(mailbox.is_current(u64::MAX));
        mailbox.stop();
        assert!(!mailbox.is_current(u64::MAX));
        assert_eq!(mailbox.pending_depth(), 1);
    }

    #[test]
    fn stale_generation_frame_and_error_cannot_publish() {
        let mailbox = DecodeCommandMailbox::new();
        let shared = Arc::new(Mutex::new(SharedState {
            latest: None,
            last_published_at: None,
            frame_overwrites: 0,
            frame_drops: 0,
            published_frames: 0,
            pending_frames_peak: 0,
            decode_durations: DurationWindow::default(),
            frame_age_durations: DurationWindow::default(),
            upload_durations: DurationWindow::default(),
            command_error: None,
            health: DecoderHealth::Healthy,
            health_revision: 1,
        }));
        let started = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        assert!(mailbox.select(0, 1.0).unwrap());

        let worker_mailbox = mailbox.clone();
        let worker_shared = shared.clone();
        let worker_started = started.clone();
        let worker_release = release.clone();
        let worker = std::thread::spawn(move || {
            run_decode_commands(worker_mailbox, worker_shared, |generation, _, _| {
                if generation == 0 {
                    *lock_recover(&worker_started.0) = true;
                    worker_started.1.notify_one();
                    let mut released = lock_recover(&worker_release.0);
                    while !*released {
                        released = worker_release
                            .1
                            .wait(released)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                    Err(DecodeWorkError::Failed(
                        "stale generation error".to_string(),
                    ))
                } else {
                    Ok(DecodedFrame::synthetic(7, generation, 0))
                }
            });
        });
        let mut did_start = lock_recover(&started.0);
        while !*did_start {
            did_start = started
                .1
                .wait(did_start)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        drop(did_start);
        mailbox.select(1, 2.0).unwrap();
        *lock_recover(&release.0) = true;
        release.1.notify_one();

        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let state = lock_state(&shared);
            if state
                .latest
                .as_ref()
                .is_some_and(|frame| frame.source_generation == 1)
            {
                assert!(state.command_error.is_none());
                break;
            }
            drop(state);
            assert!(Instant::now() < deadline, "new generation did not publish");
            std::thread::yield_now();
        }
        mailbox.stop();
        worker.join().unwrap();
    }

    #[test]
    fn source_swap_discards_stale_motion_and_retags_the_newest_seek_atomically() {
        let mailbox = DecodeCommandMailbox::new();
        let shared = Arc::new(Mutex::new(SharedState {
            latest: None,
            last_published_at: None,
            frame_overwrites: 0,
            frame_drops: 0,
            published_frames: 0,
            pending_frames_peak: 0,
            decode_durations: DurationWindow::default(),
            frame_age_durations: DurationWindow::default(),
            upload_durations: DurationWindow::default(),
            command_error: None,
            health: DecoderHealth::Healthy,
            health_revision: 1,
        }));
        let started = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        assert!(mailbox.select(10, 1.0).unwrap());

        let worker_mailbox = mailbox.clone();
        let worker_shared = shared.clone();
        let worker_started = started.clone();
        let worker_release = release.clone();
        let worker = std::thread::spawn(move || {
            run_decode_commands(worker_mailbox, worker_shared, |generation, _, _| {
                if generation == 10 {
                    *lock_recover(&worker_started.0) = true;
                    worker_started.1.notify_one();
                    let mut released = lock_recover(&worker_release.0);
                    while !*released {
                        released = worker_release
                            .1
                            .wait(released)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                    Ok(synthetic_frame_with_motion(10, generation))
                } else {
                    // Deliberately return mismatched adapter metadata. The
                    // publication transaction must retag both it and pixels.
                    Ok(synthetic_frame_with_motion(20, 0))
                }
            });
        });
        let mut did_start = lock_recover(&started.0);
        while !*did_start {
            did_start = started
                .1
                .wait(did_start)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        drop(did_start);
        assert!(mailbox.select(11, 2.0).unwrap());
        *lock_recover(&release.0) = true;
        release.1.notify_one();

        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let state = lock_state(&shared);
            if let Some(frame) = state
                .latest
                .as_ref()
                .filter(|frame| frame.source_generation == 11)
            {
                assert_eq!(frame.rgba, [20]);
                assert_eq!(frame.codec_motion.as_ref().unwrap().source_generation, 11);
                break;
            }
            drop(state);
            assert!(
                Instant::now() < deadline,
                "new motion frame did not publish"
            );
            std::thread::yield_now();
        }
        mailbox.stop();
        worker.join().unwrap();
    }

    #[test]
    fn decoded_frame_drops_a_codec_product_from_another_generation() {
        let decoded = DecodedVideoFrame {
            rgba: vec![7],
            metadata: FrameMetadata::sanitized(42, Some(7), 0.25, 1.0),
            codec_motion: Some(synthetic_codec_motion(41, 7)),
        };
        let frame = DecodedFrame::from_video(decoded, 0.25, 0, 1);
        assert_eq!(frame.source_generation, 42);
        assert!(frame.codec_motion.is_none());
    }

    #[test]
    fn initial_frame_is_seeded_with_source_metadata() {
        let mut decoder = synthetic_decoder(DecodedFrame::synthetic(4, 0, 0));
        let ready = decoder.try_next_ready_frame_result().unwrap().unwrap();
        assert_eq!(ready.rgba, vec![4]);
        assert_eq!(ready.source_generation, 0);
        assert!(ready.codec_motion.is_none());
        assert_eq!(ready.pts, Some(4));
        assert!(ready.source_seconds > 0.0);
        assert_eq!(decoder.try_next_ready_frame_result().unwrap(), None);
    }

    #[test]
    fn worker_failure_is_one_shot_but_health_remains_stable() {
        let mut decoder = synthetic_decoder(DecodedFrame::synthetic(0, 0, 0));
        decoder.try_next_ready_frame_result().unwrap();
        lock_state(&decoder.shared).set_failed("synthetic decode failure".into());
        assert_eq!(
            decoder.try_next_ready_frame_result().unwrap_err(),
            "synthetic decode failure"
        );
        assert_eq!(decoder.try_next_ready_frame_result().unwrap(), None);
        assert_eq!(
            decoder.health(),
            DecoderHealth::Failed("synthetic decode failure".into())
        );
    }

    #[test]
    fn codec_dimensions_are_checked_before_seed_decode() {
        assert!(validate_decode_dimensions(0, 1080, Some(8192)).is_err());
        assert!(validate_decode_dimensions(1920, 0, Some(8192)).is_err());
        assert!(validate_decode_dimensions(8193, 1080, Some(8192)).is_err());
        assert!(validate_decode_dimensions(1920, 1080, Some(8192)).is_ok());
    }
}
