//! Generation-safe, request-driven wrapper around [`VideoDecoder`].
//!
//! Commands occupy one overwrite slot. A source-generation change (seek, cue,
//! loop/reflection, source replacement, or stop) supersedes queued and
//! in-flight work. A newer continuous selection in the same generation only
//! replaces the queued desire: the running decode may finish and publish
//! before the worker advances directly to the newest target. The
//! completed-frame mailbox is also latest-only.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::media_safety::{
    MediaAllocationPlan, MediaDeviceLimits, MediaSafetyPolicy, MediaSourceKind,
    PERFORMANCE_MAX_PREPARED_SOURCES, PERFORMANCE_STAGING_WORKERS,
};

#[cfg(test)]
use super::decoder::validate_media_dimensions;
use super::decoder::validate_media_dimensions_with_policy as plan_media_dimensions;
use super::decoder::{KeyframeIndexBuildRequest, SeekWorkload};
use super::frame_selection::accepted_frame_remains_selected;
use super::indexed::{KeyframeIndex, MAX_INDEX_BUILD_TIME};
use super::retirement::{admit_decoder_worker, DecoderSourceFingerprint, DecoderWorkerToken};
use super::source_descriptor::{
    SourceColorDescriptor, SourceConversionPolicy, SourceDisplayDescriptor,
};
#[cfg(test)]
use super::CodecMotionFrame;
use super::{
    CodecFrameIdentity, CodecMotionProduct, DecodeWorkError, DecodedImagePayload,
    DecodedVideoFrame, FrameMetadata, VideoDecoder,
};

const DECODER_OPEN_TIMEOUT: Duration = Duration::from_secs(7);
/// First-distant-select admission budget. It matches the packet scanner's work
/// cap but deliberately does not absorb arbitrary helper-queue or input-open
/// delay; failure remains early enough for the callers' five-second deadline
/// to report it rather than starting an unreachable fallback decode.
const DISTANT_SELECT_INDEX_WAIT: Duration = MAX_INDEX_BUILD_TIME;
const DISTANT_SELECT_INDEX_POLL: Duration = Duration::from_millis(2);
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

    fn percentile(&self, percentile: usize) -> Option<Duration> {
        if self.len == 0 {
            return None;
        }
        let mut sorted = self.nanoseconds;
        sorted[..self.len].sort_unstable();
        let nearest_rank = (percentile.clamp(1, 100) * self.len).div_ceil(100);
        Some(Duration::from_nanos(sorted[nearest_rank - 1]))
    }

    fn p50(&self) -> Option<Duration> {
        self.percentile(50)
    }

    fn p95(&self) -> Option<Duration> {
        self.percentile(95)
    }

    fn p99(&self) -> Option<Duration> {
        self.percentile(99)
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

#[derive(Debug, PartialEq, Eq)]
enum KeyframeIndexWaitError {
    Superseded,
    Timeout,
    Build(String),
    Disconnected,
}

/// Wait for one already-submitted index build without weakening newest-only
/// selection. The short receive timeout is also the cancellation polling
/// interval, so a newer command never has to wait for the index deadline.
fn wait_for_keyframe_index<I>(
    receiver: &Receiver<Result<KeyframeIndex, String>>,
    timeout: Duration,
    poll: Duration,
    mut is_current: I,
) -> Result<KeyframeIndex, KeyframeIndexWaitError>
where
    I: FnMut() -> bool,
{
    let started = Instant::now();
    loop {
        if !is_current() {
            return Err(KeyframeIndexWaitError::Superseded);
        }
        let elapsed = started.elapsed();
        if elapsed >= timeout {
            return Err(KeyframeIndexWaitError::Timeout);
        }
        let remaining = timeout.saturating_sub(elapsed);
        match receiver.recv_timeout(poll.min(remaining)) {
            Ok(Ok(index)) => return Ok(index),
            Ok(Err(error)) => return Err(KeyframeIndexWaitError::Build(error)),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err(KeyframeIndexWaitError::Disconnected);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingIndexWait {
    None,
    /// The fallback is valid but too expensive for prepared-source latency.
    CostAcceleration,
    /// The fallback cannot reach the target inside the hard frame cap.
    HardCorrectness,
}

fn pending_index_wait(workload: SeekWorkload, prepared_seed: bool) -> PendingIndexWait {
    match (workload, prepared_seed) {
        (SeekWorkload::Hard, _) => PendingIndexWait::HardCorrectness,
        (SeekWorkload::Costly, true) => PendingIndexWait::CostAcceleration,
        (SeekWorkload::Prompt | SeekWorkload::Costly, _) => PendingIndexWait::None,
    }
}

fn index_failure_allows_bounded_fallback(wait: PendingIndexWait) -> bool {
    matches!(wait, PendingIndexWait::CostAcceleration)
}

fn index_wait_timeout_message(wait: PendingIndexWait, path: &str, target_seconds: f64) -> String {
    match wait {
        PendingIndexWait::CostAcceleration => format!(
            "keyframe index for {path} did not become ready within {:.1}s before costly prepared selection at {target_seconds:.6}s; the reachable bounded fallback was not started after the wait so the caller retains its response deadline",
            DISTANT_SELECT_INDEX_WAIT.as_secs_f64()
        ),
        PendingIndexWait::HardCorrectness => format!(
            "keyframe index for {path} did not become ready within {:.1}s; refusing unreachable distant selection at {target_seconds:.6}s",
            DISTANT_SELECT_INDEX_WAIT.as_secs_f64()
        ),
        PendingIndexWait::None => {
            unreachable!("a non-waiting selection cannot have an index wait timeout")
        }
    }
}

/// Install an index opportunistically for ordinary work. Unreachable work
/// always waits; a prepared saved-playhead selection also waits when its
/// resolution-aware fallback cost is high. Prompt preparation and reachable
/// live selections remain nonblocking.
fn prepare_keyframe_index_for_select<I>(
    decoder: &std::cell::RefCell<VideoDecoder>,
    pending: &std::cell::RefCell<Option<Receiver<Result<KeyframeIndex, String>>>>,
    path: &str,
    target_seconds: f64,
    previous_accepted_identity: Option<CodecFrameIdentity>,
    prepared_seed: bool,
    is_current: I,
) -> Result<(), DecodeWorkError>
where
    I: FnMut() -> bool,
{
    let workload = decoder
        .borrow()
        .selection_workload(target_seconds, previous_accepted_identity);
    let wait = pending_index_wait(workload, prepared_seed);
    if matches!(wait, PendingIndexWait::None) {
        install_ready_keyframe_index(decoder, pending, path);
        return Ok(());
    }

    let result = {
        let pending = pending.borrow();
        let Some(receiver) = pending.as_ref() else {
            if index_failure_allows_bounded_fallback(wait) {
                // Cost-aware waiting is an acceleration policy, not a second
                // correctness cap. If staging was saturated or unavailable,
                // the reachable bounded walk remains a valid fallback.
                return Ok(());
            }
            return Err(DecodeWorkError::Failed(format!(
                "cannot select distant source time {target_seconds:.6}s in {path}: no usable keyframe index is available and the bounded decode walk cannot reach it"
            )));
        };
        wait_for_keyframe_index(
            receiver,
            DISTANT_SELECT_INDEX_WAIT,
            DISTANT_SELECT_INDEX_POLL,
            is_current,
        )
    };
    let index = match result {
        Ok(index) => {
            *pending.borrow_mut() = None;
            index
        }
        Err(KeyframeIndexWaitError::Superseded) => return Err(DecodeWorkError::Superseded),
        Err(KeyframeIndexWaitError::Timeout) => {
            // Retain the receiver: a later selection may adopt the bounded
            // build if it completes after this command's wait budget.
            return Err(DecodeWorkError::Failed(index_wait_timeout_message(
                wait,
                path,
                target_seconds,
            )));
        }
        Err(KeyframeIndexWaitError::Build(error)) => {
            *pending.borrow_mut() = None;
            if index_failure_allows_bounded_fallback(wait) {
                log::warn!(
                    "keyframe index unavailable for costly prepared selection at {target_seconds:.6}s in {path}; using bounded fallback: {error}"
                );
                return Ok(());
            }
            return Err(DecodeWorkError::Failed(format!(
                "keyframe index unavailable for distant selection at {target_seconds:.6}s in {path}: {error}"
            )));
        }
        Err(KeyframeIndexWaitError::Disconnected) => {
            *pending.borrow_mut() = None;
            if index_failure_allows_bounded_fallback(wait) {
                log::warn!(
                    "keyframe index worker disconnected before costly prepared selection at {target_seconds:.6}s in {path}; using reachable bounded fallback"
                );
                return Ok(());
            }
            return Err(DecodeWorkError::Failed(format!(
                "keyframe index worker disconnected before distant selection at {target_seconds:.6}s in {path}"
            )));
        }
    };
    decoder.borrow_mut().install_keyframe_index(index);
    if matches!(
        decoder
            .borrow()
            .selection_workload(target_seconds, previous_accepted_identity),
        SeekWorkload::Hard
    ) {
        return Err(DecodeWorkError::Failed(format!(
            "cannot select distant source time {target_seconds:.6}s in {path}: the indexed preceding keyframe still exceeds the bounded decode walk"
        )));
    }
    Ok(())
}

#[derive(Debug)]
pub struct DecodedFrame {
    pub rgba: DecodedImagePayload,
    pub codec_motion: Option<CodecMotionProduct>,
    /// Loop progress 0.0..1.0 at the time this frame was decoded.
    pub progress: f32,
    /// Cumulative successful EOF reopens at the time this frame was decoded.
    pub loop_generation: u64,
    pub source_generation: u64,
    pub pts: Option<i64>,
    pub codec_identity: Option<CodecFrameIdentity>,
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
            codec_identity,
        } = metadata;
        let codec_identity = codec_identity.filter(|identity| {
            identity.source_generation == source_generation && Some(identity.pts) == pts
        });
        let codec_motion = codec_motion.filter(|motion| {
            motion.source_generation == source_generation
                && (motion.latest().status != super::CodecMotionStatus::Available
                    || motion.exact_identity().is_some_and(|identity| {
                        Some(identity.latest_destination) == codec_identity
                    }))
        });
        Self {
            rgba,
            codec_motion,
            progress,
            loop_generation,
            source_generation,
            pts,
            codec_identity,
            source_seconds,
            duration_seconds,
            command_epoch,
        }
    }

    #[cfg(test)]
    fn synthetic(value: u8, source_generation: u64, loop_generation: u64) -> Self {
        Self {
            rgba: DecodedImagePayload::from_owned_rgba(vec![value]),
            codec_motion: None,
            progress: f32::from(value) / 255.0,
            loop_generation,
            source_generation,
            pts: Some(i64::from(value)),
            codec_identity: Some(CodecFrameIdentity {
                source_generation,
                pts: i64::from(value),
                presentation_ordinal: u64::from(value),
            }),
            source_seconds: f64::from(value) / 30.0,
            duration_seconds: 10.0,
            command_epoch: 1,
        }
    }
}

/// Newest completed media frame plus source-time and loop metadata.
#[derive(Debug, PartialEq)]
pub struct ReadyFrame {
    pub rgba: DecodedImagePayload,
    pub codec_motion: Option<CodecMotionProduct>,
    pub loops_advanced: u64,
    pub source_generation: u64,
    pub pts: Option<i64>,
    pub codec_identity: Option<CodecFrameIdentity>,
    pub source_seconds: f64,
    pub duration_seconds: f64,
}

impl ReadyFrame {
    pub fn still(rgba: Vec<u8>) -> Self {
        let metadata = FrameMetadata::still();
        Self {
            rgba: DecodedImagePayload::from_owned_rgba(rgba),
            codec_motion: None,
            loops_advanced: 0,
            source_generation: metadata.source_generation,
            pts: metadata.pts,
            codec_identity: metadata.codec_identity,
            source_seconds: metadata.source_seconds,
            duration_seconds: metadata.duration_seconds,
        }
    }
}

/// How selecting a seed frame at a requested start position ended without
/// producing one. The caller owns the operator-facing message — it knows the
/// source path and its own deadline policy — so these variants carry only
/// what the caller cannot re-derive.
#[derive(Debug)]
pub enum SeedSelectError {
    /// The caller's currency check answered false mid-wait.
    Superseded,
    /// The decoder opened without publishing a decoded first frame.
    NoSeedFrame,
    /// The decoder reported a hard error.
    Decode(String),
    /// No frame for the requested generation arrived within the deadline.
    Timeout { target_seconds: f64 },
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
    /// Immutable source declarations and the actual conversion path frozen
    /// before the seed picture entered the RGBA mailbox.
    pub source_color: SourceColorDescriptor,
    pub source_display: SourceDisplayDescriptor,
    pub conversion_policy: SourceConversionPolicy,
    pub last_publish_age: Option<Duration>,
    pub last_consume_age: Option<Duration>,
    /// Time since the latest decoded source image was successfully accepted by
    /// the GPU upload seam. Unlike publish-to-consume age, this keeps growing
    /// through a visible source hold.
    pub last_accepted_upload_age: Option<Duration>,
    pub pending_command_depth: u8,
    pub pending_frames: u8,
    pub pending_frames_peak: u8,
    pub published_frames: u64,
    pub consumed_frames: u64,
    pub command_overwrites: u64,
    pub command_drops: u64,
    /// Accepted selections whose generation matched the prior accepted
    /// selection. These are ordinary continuous playback desires, not
    /// cancellation tokens.
    pub continuous_requests: u64,
    /// Accepted selections that advanced the source discontinuity token.
    pub discontinuity_requests: u64,
    /// Same-generation desires replaced before the worker received them.
    pub continuous_overwrites: u64,
    /// Queued work replaced by a different generation before receipt.
    pub discontinuity_overwrites: u64,
    pub frame_overwrites: u64,
    pub frame_drops: u64,
    /// Queued commands retired before decode because their generation was no
    /// longer current.
    pub stale_commands: u64,
    /// Decode attempts cooperatively retired by a discontinuity or Stop.
    pub superseded_decode_attempts: u64,
    /// Completed mailbox frames rejected at consume because their generation
    /// was no longer current.
    pub stale_completed_frames: u64,
    pub failed_decode_attempts: u64,
    /// All attempts, including failures and superseded work. The existing
    /// `decode_*` fields remain successful-publication timings.
    pub decode_attempt_samples: u64,
    pub last_decode_attempt_duration: Option<Duration>,
    pub peak_decode_attempt_duration: Option<Duration>,
    pub decode_attempt_p95_duration: Option<Duration>,
    pub decode_samples: u64,
    pub last_decode_duration: Option<Duration>,
    pub peak_decode_duration: Option<Duration>,
    pub decode_p95_duration: Option<Duration>,
    pub publish_interval_p95_duration: Option<Duration>,
    pub publish_interval_p99_duration: Option<Duration>,
    pub peak_publish_interval_duration: Option<Duration>,
    pub frame_age_p95_duration: Option<Duration>,
    pub upload_samples: u64,
    pub last_upload_duration: Option<Duration>,
    pub peak_upload_duration: Option<Duration>,
    pub upload_p95_duration: Option<Duration>,
    pub accepted_uploads: u64,
    pub accepted_upload_interval_p50_duration: Option<Duration>,
    pub accepted_upload_interval_p95_duration: Option<Duration>,
    pub accepted_upload_interval_p99_duration: Option<Duration>,
    pub peak_accepted_upload_interval_duration: Option<Duration>,
    /// Intervals in which more than one continuous playback request elapsed
    /// between accepted images. Authored holds and discontinuities are not
    /// admitted, so this measures decoder delivery starvation rather than
    /// merely time between uploads.
    pub decoder_delivery_hold_p95_duration: Option<Duration>,
    pub peak_decoder_delivery_hold_duration: Option<Duration>,
}

#[derive(Debug)]
struct SharedState {
    latest: Option<DecodedFrame>,
    last_published_at: Option<Instant>,
    publish_interval_durations: DurationWindow,
    last_accepted_upload_at: Option<Instant>,
    accepted_upload_interval_durations: DurationWindow,
    decoder_delivery_hold_durations: DurationWindow,
    accepted_uploads: u64,
    frame_overwrites: u64,
    frame_drops: u64,
    stale_commands: u64,
    superseded_decode_attempts: u64,
    stale_completed_frames: u64,
    failed_decode_attempts: u64,
    published_frames: u64,
    pending_frames_peak: u8,
    decode_attempt_durations: DurationWindow,
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
            publish_interval_durations: DurationWindow::default(),
            last_accepted_upload_at: None,
            accepted_upload_interval_durations: DurationWindow::default(),
            decoder_delivery_hold_durations: DurationWindow::default(),
            accepted_uploads: 0,
            frame_overwrites: 0,
            frame_drops: 0,
            stale_commands: 0,
            superseded_decode_attempts: 0,
            stale_completed_frames: 0,
            failed_decode_attempts: 0,
            published_frames: 1,
            pending_frames_peak: 1,
            decode_attempt_durations: DurationWindow::default(),
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
        if let Some(previous) = self.last_published_at {
            self.publish_interval_durations
                .record(duration_since_or_zero(published_at, previous));
        }
        self.last_published_at = Some(published_at);
        self.published_frames = self.published_frames.saturating_add(1);
        self.pending_frames_peak = 1;
        self.decode_durations.record(decode_duration);
        self.command_error = None;
    }

    fn record_seed_decode(&mut self, decode_duration: Duration) {
        self.decode_attempt_durations.record(decode_duration);
        self.decode_durations.record(decode_duration);
    }

    fn record_decode_attempt(&mut self, decode_duration: Duration) {
        self.decode_attempt_durations.record(decode_duration);
    }

    fn drop_stale_command(&mut self) {
        self.stale_commands = self.stale_commands.saturating_add(1);
        self.frame_drops = self.frame_drops.saturating_add(1);
    }

    fn drop_superseded_decode_attempt(&mut self) {
        self.superseded_decode_attempts = self.superseded_decode_attempts.saturating_add(1);
        self.frame_drops = self.frame_drops.saturating_add(1);
    }

    fn drop_stale_completed_frame(&mut self) {
        self.stale_completed_frames = self.stale_completed_frames.saturating_add(1);
        self.frame_drops = self.frame_drops.saturating_add(1);
    }

    fn record_failed_decode_attempt(&mut self) {
        self.failed_decode_attempts = self.failed_decode_attempts.saturating_add(1);
    }

    fn record_accepted_upload(
        &mut self,
        accepted_at: Instant,
        continuous_requests_since_previous: u64,
    ) {
        if let Some(previous) = self.last_accepted_upload_at {
            let interval = duration_since_or_zero(accepted_at, previous);
            self.accepted_upload_interval_durations.record(interval);
            if continuous_requests_since_previous > 1 {
                self.decoder_delivery_hold_durations.record(interval);
            }
        }
        self.last_accepted_upload_at = Some(accepted_at);
        self.accepted_uploads = self.accepted_uploads.saturating_add(1);
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
    previous_accepted_codec_identity: Option<CodecFrameIdentity>,
    prepared_seed: bool,
    continuous: bool,
}

#[derive(Debug)]
struct CommandMailboxState {
    pending: Option<QueuedCommand>,
    epoch: u64,
    generation: u64,
    stopped: bool,
}

#[derive(Debug)]
pub(super) struct DecodeCommandMailbox {
    state: Mutex<CommandMailboxState>,
    wake: Condvar,
    current_epoch: AtomicU64,
    current_generation: AtomicU64,
    stopped: AtomicBool,
    command_overwrites: AtomicU64,
    command_drops: AtomicU64,
    continuous_requests: AtomicU64,
    discontinuity_requests: AtomicU64,
    continuous_overwrites: AtomicU64,
    discontinuity_overwrites: AtomicU64,
}

impl DecodeCommandMailbox {
    pub(super) fn new() -> Arc<Self> {
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
            continuous_requests: AtomicU64::new(0),
            discontinuity_requests: AtomicU64::new(0),
            continuous_overwrites: AtomicU64::new(0),
            discontinuity_overwrites: AtomicU64::new(0),
        })
    }

    #[cfg(test)]
    fn select(&self, generation: u64, target_seconds: f64) -> Result<bool, String> {
        self.select_after(generation, target_seconds, None)
    }

    fn select_after(
        &self,
        generation: u64,
        target_seconds: f64,
        previous_accepted_codec_identity: Option<CodecFrameIdentity>,
    ) -> Result<bool, String> {
        self.select_after_with_kind(
            generation,
            target_seconds,
            previous_accepted_codec_identity,
            false,
        )
    }

    fn select_prepared_seed(&self, generation: u64, target_seconds: f64) -> Result<bool, String> {
        self.select_after_with_kind(generation, target_seconds, None, true)
    }

    fn select_after_with_kind(
        &self,
        generation: u64,
        target_seconds: f64,
        previous_accepted_codec_identity: Option<CodecFrameIdentity>,
        prepared_seed: bool,
    ) -> Result<bool, String> {
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
        let continuous = generation == state.generation;
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
            previous_accepted_codec_identity: previous_accepted_codec_identity
                .filter(|identity| identity.source_generation == generation),
            prepared_seed,
            continuous,
        });
        if continuous {
            self.continuous_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.discontinuity_requests.fetch_add(1, Ordering::Relaxed);
        }
        if replaced.is_some() {
            self.command_overwrites.fetch_add(1, Ordering::Relaxed);
            if replaced.is_some_and(|queued| match queued.command {
                DecodeCommand::Select {
                    generation: replaced_generation,
                    ..
                } => replaced_generation == generation,
                DecodeCommand::Stop => false,
            }) {
                self.continuous_overwrites.fetch_add(1, Ordering::Relaxed);
            } else {
                self.discontinuity_overwrites
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        self.current_generation.store(generation, Ordering::Release);
        self.current_epoch.store(epoch, Ordering::Release);
        drop(state);
        self.wake.notify_one();
        Ok(true)
    }

    pub(super) fn stop(&self) {
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
            previous_accepted_codec_identity: None,
            prepared_seed: false,
            continuous: false,
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

    /// A source generation, unlike a command epoch, is a true cancellation
    /// boundary. Continuous playback may advance the epoch every program tick
    /// while a valid same-generation decode is still running.
    fn is_generation_current(&self, generation: u64) -> bool {
        !self.stopped.load(Ordering::Acquire)
            && self.current_generation.load(Ordering::Acquire) == generation
    }

    pub(super) fn generation(&self) -> u64 {
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
    /// The retirement supervisor owns the `JoinHandle`; this token converts
    /// that exact worker to a canceling retiree when the decoder is displaced.
    worker: Option<DecoderWorkerToken>,
    shared: Arc<Mutex<SharedState>>,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    source_color_descriptor: SourceColorDescriptor,
    source_display_descriptor: SourceDisplayDescriptor,
    conversion_policy: SourceConversionPolicy,
    #[allow(dead_code)]
    pub duration_seconds: f64,
    media_plan: MediaAllocationPlan,
    progress: f32,
    consumed_loop_generation: u64,
    consumed_source_generation: u64,
    accepted_source_generation: Option<u64>,
    accepted_source_seconds: Option<f64>,
    accepted_codec_identity: Option<CodecFrameIdentity>,
    last_accepted_continuous_requests: Option<u64>,
    last_consumed_upload_token: Option<ConsumedUploadToken>,
    failure_revision_reported: u64,
    last_consumed_at: Option<Instant>,
    consumed_frames: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConsumedUploadToken {
    source_generation: u64,
    source_seconds_bits: u64,
    codec_identity: Option<CodecFrameIdentity>,
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
        let (meta_tx, meta_rx) = std::sync::mpsc::channel::<
            Result<
                (
                    u32,
                    u32,
                    f32,
                    f64,
                    MediaAllocationPlan,
                    SourceColorDescriptor,
                    SourceDisplayDescriptor,
                    SourceConversionPolicy,
                ),
                String,
            >,
        >();

        let thread_name = format!("decode-{}", short_name(path));
        let path_owned = path.to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let worker = admit_decoder_worker(
            DecoderSourceFingerprint::from_source(path),
            cancel.clone(),
            mailbox.clone(),
        )?;
        let shared = Arc::new(Mutex::new(SharedState {
            latest: None,
            last_published_at: None,
            publish_interval_durations: DurationWindow::default(),
            last_accepted_upload_at: None,
            accepted_upload_interval_durations: DurationWindow::default(),
            decoder_delivery_hold_durations: DurationWindow::default(),
            accepted_uploads: 0,
            frame_overwrites: 0,
            frame_drops: 0,
            stale_commands: 0,
            superseded_decode_attempts: 0,
            stale_completed_frames: 0,
            failed_decode_attempts: 0,
            published_frames: 0,
            pending_frames_peak: 0,
            decode_attempt_durations: DurationWindow::default(),
            decode_durations: DurationWindow::default(),
            frame_age_durations: DurationWindow::default(),
            upload_durations: DurationWindow::default(),
            command_error: None,
            health: DecoderHealth::Healthy,
            health_revision: 1,
        }));
        let worker_shared = shared.clone();
        let worker_handle = match std::thread::Builder::new()
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
                        decoder.source_color_descriptor(),
                        decoder.source_display_descriptor(),
                        decoder.conversion_policy(),
                    );
                    let mut seeded = SharedState::healthy_with_seed(seed);
                    seeded.record_seed_decode(seed_decode_duration);
                    *lock_state(&worker_shared) = seeded;
                    if meta_tx.send(Ok(dimensions)).is_err() {
                        return;
                    }

                    // A bounded helper pool scans a second demuxer. Ordinary
                    // nearby/live Select commands continue immediately;
                    // unreachable work and costly prepared saved playheads
                    // cooperatively wait for their pending index below.
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
                        |generation,
                         target_seconds,
                         previous_accepted_codec_identity,
                         prepared_seed,
                         _epoch,
                         continuous| {
                            prepare_keyframe_index_for_select(
                                &decoder,
                                &index_result,
                                &path_owned,
                                target_seconds,
                                previous_accepted_codec_identity,
                                prepared_seed,
                                || select_mailbox.is_generation_current(generation),
                            )?;
                            let mut decoder = decoder.borrow_mut();
                            let frame = if continuous && !prepared_seed {
                                decoder.seek_decode_after_interruptible_bounded(
                                    target_seconds,
                                    generation,
                                    previous_accepted_codec_identity,
                                    super::decoder::LIVE_FORWARD_PROGRESS_PERIODS,
                                    || select_mailbox.is_generation_current(generation),
                                )?
                            } else {
                                decoder.seek_decode_after_interruptible(
                                    target_seconds,
                                    generation,
                                    previous_accepted_codec_identity,
                                    || select_mailbox.is_generation_current(generation),
                                )?
                            };
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
            }) {
            Ok(handle) => handle,
            Err(error) => {
                worker.abandon();
                return Err(format!("Failed to spawn decode thread: {error}"));
            }
        };
        worker.attach(worker_handle);

        let (
            width,
            height,
            fps,
            duration_seconds,
            media_plan,
            source_color_descriptor,
            source_display_descriptor,
            conversion_policy,
        ) = match meta_rx.recv_timeout(DECODER_OPEN_TIMEOUT) {
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
            worker: Some(worker),
            shared,
            width,
            height,
            fps,
            source_color_descriptor,
            source_display_descriptor,
            conversion_policy,
            duration_seconds,
            media_plan,
            progress: 0.0,
            consumed_loop_generation: 0,
            consumed_source_generation: 0,
            accepted_source_generation: None,
            accepted_source_seconds: None,
            accepted_codec_identity: None,
            last_accepted_continuous_requests: None,
            last_consumed_upload_token: None,
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
        if accepted_frame_remains_selected(
            generation,
            target_seconds,
            self.accepted_source_generation,
            self.accepted_source_seconds,
            self.fps,
        ) {
            // The caller asks whether the selection was accepted, not whether
            // a command was enqueued. Holding the already-uploaded exact
            // selection avoids decoding a lower-rate source at the program
            // rate, which otherwise runs the codec ahead and eventually
            // forces an expensive corrective reverse selection.
            return Ok(true);
        }
        let forward_tolerance = 0.25 / f64::from(self.fps.max(1.0));
        let previous_accepted_codec_identity = (Some(generation)
            == self.accepted_source_generation
            && self
                .accepted_source_seconds
                .is_some_and(|accepted| target_seconds > accepted + forward_tolerance))
        .then_some(self.accepted_codec_identity)
        .flatten();
        self.mailbox
            .select_after(generation, target_seconds, previous_accepted_codec_identity)
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

    /// Harvest the open seed frame and, for a nonzero start position on a
    /// seekable source, replace it with the frame at that position: bump the
    /// source generation, queue the absolute selection, and poll until the
    /// matching frame arrives or `timeout` elapses. This is the single
    /// implementation of the prepared-source seed dance; the performance
    /// preparer and proxy hot adoption both call it so the two cannot drift.
    pub fn select_seed_frame_at(
        &mut self,
        start_position: f64,
        timeout: Duration,
        poll: Duration,
        is_current: &dyn Fn() -> bool,
    ) -> Result<ReadyFrame, SeedSelectError> {
        let seed = self
            .try_next_ready_frame_result()
            .map_err(SeedSelectError::Decode)?
            .ok_or(SeedSelectError::NoSeedFrame)?;
        if start_position <= 0.0 || self.duration_seconds <= 0.0 {
            return Ok(seed);
        }
        let generation = self.source_generation().checked_add(1).ok_or_else(|| {
            SeedSelectError::Decode("prepared video source generation exhausted".to_string())
        })?;
        let target_seconds = start_position * self.duration_seconds;
        self.mailbox
            .select_prepared_seed(generation, target_seconds)
            .map_err(SeedSelectError::Decode)?;
        let started = Instant::now();
        loop {
            if !is_current() {
                return Err(SeedSelectError::Superseded);
            }
            match self
                .try_next_ready_frame_result()
                .map_err(SeedSelectError::Decode)?
            {
                Some(frame) if frame.source_generation == generation => return Ok(frame),
                Some(_) | None => {}
            }
            if started.elapsed() >= timeout {
                return Err(SeedSelectError::Timeout { target_seconds });
            }
            std::thread::sleep(poll);
        }
    }

    pub fn try_next_ready_frame_result(&mut self) -> Result<Option<ReadyFrame>, String> {
        self.try_next_ready_frame_result_at(Instant::now())
    }

    fn try_next_ready_frame_result_at(
        &mut self,
        consumed_at: Instant,
    ) -> Result<Option<ReadyFrame>, String> {
        let current_generation = self.mailbox.generation();
        let mut state = lock_state(&self.shared);
        // Discard on **generation**, not on epoch. The source generation is
        // the discontinuity token — a seek, cue, or loop changes it — and
        // dropping a completion that predates one is the whole point of
        // publishing selections before harvesting. The epoch is a different
        // thing: it advances on *every* command, so guarding on it here
        // discards the completion of the immediately preceding request, which
        // in ordinary playback is every frame the decoder ever finishes. The
        // render thread issues a request and harvests a few microseconds
        // later in the same frame body, so the worker essentially never wins
        // that race; delivery then survives only on the rare rendered frame
        // that is not sample-due, which is why playback ran at roughly
        // `(render_fps - sample_fps) / render_fps` of its authored rate and
        // stopped entirely for a source whose rate equals the render rate.
        // Taking the previous request's completion is exactly the documented
        // latest-only mailbox behaviour, one frame of latency and no backlog.
        if state
            .latest
            .as_ref()
            .is_some_and(|frame| frame.source_generation != current_generation)
        {
            state.latest = None;
            state.drop_stale_completed_frame();
        }
        if state
            .command_error
            .as_ref()
            .is_some_and(|(generation, _, _)| *generation != current_generation)
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
            self.last_consumed_upload_token = Some(ConsumedUploadToken {
                source_generation: frame.source_generation,
                source_seconds_bits: frame.source_seconds.to_bits(),
                codec_identity: frame.codec_identity,
            });
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
                codec_identity: frame.codec_identity,
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
            .map(|ready| ready.map(|frame| frame.rgba.into_vec()))
    }

    pub fn progress(&self) -> f32 {
        self.progress
    }

    pub fn media_allocation_plan(&self) -> &MediaAllocationPlan {
        &self.media_plan
    }

    pub const fn source_color_descriptor(&self) -> SourceColorDescriptor {
        self.source_color_descriptor
    }

    pub const fn source_display_descriptor(&self) -> SourceDisplayDescriptor {
        self.source_display_descriptor
    }

    pub const fn conversion_policy(&self) -> SourceConversionPolicy {
        self.conversion_policy
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
        let continuous_requests = self.mailbox.continuous_requests.load(Ordering::Relaxed);
        let discontinuity_requests = self.mailbox.discontinuity_requests.load(Ordering::Relaxed);
        let continuous_overwrites = self.mailbox.continuous_overwrites.load(Ordering::Relaxed);
        let discontinuity_overwrites = self
            .mailbox
            .discontinuity_overwrites
            .load(Ordering::Relaxed);
        let state = lock_state(&self.shared);
        DecoderTelemetry {
            source_color: self.source_color_descriptor,
            source_display: self.source_display_descriptor,
            conversion_policy: self.conversion_policy,
            last_publish_age: state
                .last_published_at
                .map(|published_at| duration_since_or_zero(sampled_at, published_at)),
            last_consume_age: self
                .last_consumed_at
                .map(|consumed_at| duration_since_or_zero(sampled_at, consumed_at)),
            last_accepted_upload_age: state
                .last_accepted_upload_at
                .map(|accepted_at| duration_since_or_zero(sampled_at, accepted_at)),
            pending_command_depth,
            pending_frames: u8::from(state.latest.is_some()),
            pending_frames_peak: state.pending_frames_peak,
            published_frames: state.published_frames,
            consumed_frames: self.consumed_frames,
            command_overwrites,
            command_drops,
            continuous_requests,
            discontinuity_requests,
            continuous_overwrites,
            discontinuity_overwrites,
            frame_overwrites: state.frame_overwrites,
            frame_drops: state.frame_drops,
            stale_commands: state.stale_commands,
            superseded_decode_attempts: state.superseded_decode_attempts,
            stale_completed_frames: state.stale_completed_frames,
            failed_decode_attempts: state.failed_decode_attempts,
            decode_attempt_samples: state.decode_attempt_durations.total_samples,
            last_decode_attempt_duration: state.decode_attempt_durations.last(),
            peak_decode_attempt_duration: state.decode_attempt_durations.peak(),
            decode_attempt_p95_duration: state.decode_attempt_durations.p95(),
            decode_samples: state.decode_durations.total_samples,
            last_decode_duration: state.decode_durations.last(),
            peak_decode_duration: state.decode_durations.peak(),
            decode_p95_duration: state.decode_durations.p95(),
            publish_interval_p95_duration: state.publish_interval_durations.p95(),
            publish_interval_p99_duration: state.publish_interval_durations.p99(),
            peak_publish_interval_duration: state.publish_interval_durations.peak(),
            frame_age_p95_duration: state.frame_age_durations.p95(),
            upload_samples: state.upload_durations.total_samples,
            last_upload_duration: state.upload_durations.last(),
            peak_upload_duration: state.upload_durations.peak(),
            upload_p95_duration: state.upload_durations.p95(),
            accepted_uploads: state.accepted_uploads,
            accepted_upload_interval_p50_duration: state.accepted_upload_interval_durations.p50(),
            accepted_upload_interval_p95_duration: state.accepted_upload_interval_durations.p95(),
            accepted_upload_interval_p99_duration: state.accepted_upload_interval_durations.p99(),
            peak_accepted_upload_interval_duration: state.accepted_upload_interval_durations.peak(),
            decoder_delivery_hold_p95_duration: state.decoder_delivery_hold_durations.p95(),
            peak_decoder_delivery_hold_duration: state.decoder_delivery_hold_durations.peak(),
        }
    }

    /// Commit one successfully uploaded source image as the predecessor for
    /// skipped-frame motion composition. Merely removing a decoded frame from
    /// the mailbox is not acceptance: a failed GPU upload must leave the prior
    /// predecessor intact. The caller also supplies the complete
    /// queue-write/error-scope wall interval for telemetry; no staging buffer
    /// or timestamp query is allocated.
    pub fn record_accepted_upload(
        &mut self,
        source_generation: u64,
        source_seconds: f64,
        codec_identity: Option<CodecFrameIdentity>,
        duration: Duration,
    ) {
        let upload_matches_consumed_frame = self.last_consumed_upload_token.is_some_and(|token| {
            token.source_generation == source_generation
                && token.source_seconds_bits == source_seconds.to_bits()
                && token.codec_identity == codec_identity
        });
        let mut state = lock_state(&self.shared);
        state.upload_durations.record(duration);
        if source_seconds.is_finite() && source_seconds >= 0.0 && upload_matches_consumed_frame {
            let continuous_requests = self.mailbox.continuous_requests.load(Ordering::Relaxed);
            let continuous_requests_since_previous =
                if self.accepted_source_generation == Some(source_generation) {
                    self.last_accepted_continuous_requests
                        .map_or(0, |previous| continuous_requests.saturating_sub(previous))
                } else {
                    0
                };
            state.record_accepted_upload(Instant::now(), continuous_requests_since_previous);
            drop(state);
            self.accepted_source_generation = Some(source_generation);
            self.accepted_source_seconds = Some(source_seconds);
            self.accepted_codec_identity =
                codec_identity.filter(|identity| identity.source_generation == source_generation);
            self.last_accepted_continuous_requests = Some(continuous_requests);
        } else {
            drop(state);
            self.accepted_source_generation = None;
            self.accepted_source_seconds = None;
            self.accepted_codec_identity = None;
            self.last_accepted_continuous_requests = None;
        }
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
        if let Some(worker) = self.worker.take() {
            worker.retire(self.mailbox.generation());
        }
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
    S: FnMut(
        u64,
        f64,
        Option<CodecFrameIdentity>,
        bool,
        u64,
        bool,
    ) -> Result<DecodedFrame, DecodeWorkError>,
{
    loop {
        let queued = mailbox.recv();
        if matches!(queued.command, DecodeCommand::Stop) {
            return;
        }
        let (generation, target_seconds) = match queued.command {
            DecodeCommand::Select {
                generation,
                target_seconds,
            } => (generation, target_seconds),
            DecodeCommand::Stop => unreachable!("stop returned above"),
        };
        if !mailbox.is_generation_current(generation) {
            lock_state(&shared).drop_stale_command();
            continue;
        }

        let decode_started = Instant::now();
        let result = select_frame(
            generation,
            target_seconds,
            queued.previous_accepted_codec_identity,
            queued.prepared_seed,
            queued.epoch,
            queued.continuous,
        );
        let decode_duration = decode_started.elapsed();

        let generation_is_current = mailbox.is_generation_current(generation);
        let epoch_is_latest = mailbox.is_current(queued.epoch);
        let mut state = lock_state(&shared);
        state.record_decode_attempt(decode_duration);
        if !generation_is_current {
            state.drop_superseded_decode_attempt();
            continue;
        }
        match result {
            Ok(mut frame) => {
                frame.source_generation = generation;
                if let Some(identity) = frame.codec_identity.as_mut() {
                    identity.source_generation = generation;
                }
                frame.codec_identity = frame
                    .codec_identity
                    .filter(|identity| Some(identity.pts) == frame.pts);
                if let Some(codec_motion) = frame.codec_motion.as_mut() {
                    codec_motion.retag_source_generation(generation);
                }
                if frame.codec_motion.as_ref().is_some_and(|motion| {
                    motion.latest().status == super::CodecMotionStatus::Available
                        && !motion.exact_identity().is_some_and(|identity| {
                            Some(identity.latest_destination) == frame.codec_identity
                        })
                }) {
                    frame.codec_motion = None;
                }
                frame.command_epoch = queued.epoch;
                state.publish(frame, Instant::now(), decode_duration);
            }
            Err(DecodeWorkError::Failed(error)) => {
                state.record_failed_decode_attempt();
                // A newer same-generation desire makes this failure obsolete;
                // the worker will immediately try that newest target. Exposing
                // the older error would turn healthy coalescing into a visible
                // false failure.
                if epoch_is_latest {
                    state.command_error = Some((generation, queued.epoch, error));
                }
            }
            Err(DecodeWorkError::Superseded) => state.drop_superseded_decode_attempt(),
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
            past_reference_proof: None,
            vectors: Vec::new(),
        }
    }

    fn synthetic_frame_with_motion(value: u8, source_generation: u64) -> DecodedFrame {
        let mut frame = DecodedFrame::synthetic(value, source_generation, 0);
        frame.codec_motion =
            Some(synthetic_codec_motion(source_generation, u64::from(value)).into());
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
            worker: None,
            shared: Arc::new(Mutex::new(SharedState::healthy_with_seed_at(
                seed,
                published_at,
            ))),
            width: 1,
            height: 1,
            fps: 30.0,
            source_color_descriptor: SourceColorDescriptor::default(),
            source_display_descriptor: SourceDisplayDescriptor::default(),
            conversion_policy: SourceConversionPolicy::legacy_video(),
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
            accepted_source_generation: None,
            accepted_source_seconds: None,
            accepted_codec_identity: None,
            last_accepted_continuous_requests: None,
            last_consumed_upload_token: None,
            failure_revision_reported: 0,
            last_consumed_at: None,
            consumed_frames: 0,
        }
    }

    #[test]
    fn distant_selection_wait_accepts_a_delayed_keyframe_index() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(15));
            sender
                .send(Ok(KeyframeIndex::fallback(0).unwrap()))
                .unwrap();
        });

        let index = wait_for_keyframe_index(
            &receiver,
            Duration::from_millis(250),
            Duration::from_millis(1),
            || true,
        )
        .expect("the delayed index should be installed before distant decode");
        assert_eq!(index.len(), 1);
        worker.join().unwrap();
    }

    #[test]
    fn distant_selection_wait_is_superseded_while_the_index_is_pending() {
        let (_sender, receiver) = std::sync::mpsc::sync_channel::<Result<KeyframeIndex, String>>(1);
        let current = Arc::new(AtomicBool::new(true));
        let superseding = current.clone();
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(15));
            superseding.store(false, Ordering::Release);
        });

        assert!(matches!(
            wait_for_keyframe_index(
                &receiver,
                Duration::from_secs(1),
                Duration::from_millis(1),
                || current.load(Ordering::Acquire),
            ),
            Err(KeyframeIndexWaitError::Superseded)
        ));
        worker.join().unwrap();
    }

    #[test]
    fn bounded_index_timeout_does_not_consume_the_receiver() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        assert!(matches!(
            wait_for_keyframe_index(
                &receiver,
                Duration::from_millis(2),
                Duration::from_millis(1),
                || true,
            ),
            Err(KeyframeIndexWaitError::Timeout)
        ));
        sender
            .send(Ok(KeyframeIndex::fallback(0).unwrap()))
            .unwrap();
        assert_eq!(
            wait_for_keyframe_index(
                &receiver,
                Duration::from_millis(20),
                Duration::from_millis(1),
                || true,
            )
            .unwrap()
            .len(),
            1
        );
    }

    #[test]
    fn pending_index_policy_waits_only_for_costly_preparation_or_hard_correctness() {
        assert_eq!(
            pending_index_wait(SeekWorkload::Prompt, true),
            PendingIndexWait::None
        );
        assert_eq!(
            pending_index_wait(SeekWorkload::Costly, false),
            PendingIndexWait::None,
            "reachable live selections must remain nonblocking"
        );
        assert_eq!(
            pending_index_wait(SeekWorkload::Costly, true),
            PendingIndexWait::CostAcceleration
        );
        assert_eq!(
            pending_index_wait(SeekWorkload::Hard, false),
            PendingIndexWait::HardCorrectness
        );
        assert_eq!(
            pending_index_wait(SeekWorkload::Hard, true),
            PendingIndexWait::HardCorrectness
        );
    }

    #[test]
    fn costly_index_failures_remain_reachable_and_timeout_wording_is_exact() {
        assert!(index_failure_allows_bounded_fallback(
            PendingIndexWait::CostAcceleration
        ));
        assert!(!index_failure_allows_bounded_fallback(
            PendingIndexWait::HardCorrectness
        ));

        let costly = index_wait_timeout_message(
            PendingIndexWait::CostAcceleration,
            "ars-longa.mp4",
            30.029_765,
        );
        assert!(costly.contains("costly prepared selection"));
        assert!(costly.contains("reachable bounded fallback"));
        assert!(!costly.contains("unreachable"));

        let hard =
            index_wait_timeout_message(PendingIndexWait::HardCorrectness, "distant.mp4", 409.0);
        assert!(hard.contains("unreachable distant selection"));
    }

    #[test]
    #[ignore = "requires COLLIDEOSCOPE_DISTANT_VIDEO_FIXTURE pointing to a seekable long video"]
    fn real_immediate_distant_seed_selection_reaches_the_requested_generation() {
        let path = std::env::var("COLLIDEOSCOPE_DISTANT_VIDEO_FIXTURE")
            .expect("set COLLIDEOSCOPE_DISTANT_VIDEO_FIXTURE to a long seekable video");
        let start_position = std::env::var("COLLIDEOSCOPE_DISTANT_VIDEO_POSITION")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(0.596_448_694_580_675_2);
        let mut decoder = ThreadedDecoder::open_with_media_policy(
            &path,
            &MediaSafetyPolicy::safe(),
            MediaDeviceLimits::none(),
        )
        .unwrap();
        let target_seconds = start_position * decoder.duration_seconds;
        let started = Instant::now();
        let frame = decoder
            .select_seed_frame_at(
                start_position,
                Duration::from_secs(5),
                Duration::from_millis(2),
                &|| true,
            )
            .unwrap();

        assert_eq!(frame.source_generation, 1);
        assert!(
            (frame.source_seconds - target_seconds).abs() <= 1.0 / f64::from(decoder.fps.max(1.0)),
            "selected {:.6}s instead of requested {target_seconds:.6}s",
            frame.source_seconds
        );
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    #[ignore = "requires COLLIDEOSCOPE_ARS_LONGA_VIDEO_FIXTURE pointing to the exact 445343423-byte source"]
    fn real_ars_longa_saved_playhead_completes_inside_patch_deadline() {
        let path = std::env::var("COLLIDEOSCOPE_ARS_LONGA_VIDEO_FIXTURE")
            .expect("set COLLIDEOSCOPE_ARS_LONGA_VIDEO_FIXTURE to Ars-Longa's long video");
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            445_343_423,
            "fixture is not Ars-Longa's exact source"
        );
        let start_position = 0.167_920_403_325_936_1;
        let mut decoder = ThreadedDecoder::open_with_media_policy(
            &path,
            &MediaSafetyPolicy::safe(),
            MediaDeviceLimits::none(),
        )
        .unwrap();
        let target_seconds = start_position * decoder.duration_seconds;
        let started = Instant::now();
        let frame = decoder
            .select_seed_frame_at(
                start_position,
                Duration::from_secs(5),
                Duration::from_millis(2),
                &|| true,
            )
            .unwrap();

        assert_eq!(frame.source_generation, 1);
        assert!(
            (frame.source_seconds - target_seconds).abs() <= 1.0 / f64::from(decoder.fps.max(1.0)),
            "selected {:.6}s instead of requested {target_seconds:.6}s",
            frame.source_seconds
        );
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn newest_selection_keeps_its_accepted_motion_predecessor_in_the_same_slot() {
        let mailbox = DecodeCommandMailbox::new();
        let first = CodecFrameIdentity {
            source_generation: 7,
            pts: 45,
            presentation_ordinal: 45,
        };
        let newest = CodecFrameIdentity {
            source_generation: 7,
            pts: 75,
            presentation_ordinal: 75,
        };
        assert!(mailbox.select_after(7, 2.0, Some(first)).unwrap());
        assert!(mailbox.select_after(7, 3.0, Some(newest)).unwrap());
        let queued = mailbox.recv();
        assert_eq!(
            queued.command,
            DecodeCommand::Select {
                generation: 7,
                target_seconds: 3.0,
            }
        );
        assert_eq!(queued.previous_accepted_codec_identity, Some(newest));
        assert!(!queued.prepared_seed);
        assert_eq!(mailbox.command_overwrites.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn an_already_uploaded_lower_rate_frame_satisfies_its_selection_window() {
        let mut decoder = synthetic_decoder(DecodedFrame::synthetic(3, 0, 0));
        decoder.fps = 24.0;
        let seed = decoder
            .try_next_ready_frame_result()
            .unwrap()
            .expect("seed frame");
        decoder.record_accepted_upload(
            seed.source_generation,
            seed.source_seconds,
            seed.codec_identity,
            Duration::ZERO,
        );

        assert!(decoder
            .request_source_time(0, seed.source_seconds + 0.020)
            .unwrap());
        assert_eq!(decoder.mailbox.pending_depth(), 0);

        assert!(decoder
            .request_source_time(0, seed.source_seconds + 0.021)
            .unwrap());
        assert_eq!(decoder.mailbox.pending_depth(), 1);
        assert!(!accepted_frame_remains_selected(
            1,
            seed.source_seconds,
            Some(0),
            Some(seed.source_seconds),
            24.0,
        ));
        assert!(!accepted_frame_remains_selected(
            0,
            seed.source_seconds - 0.030,
            Some(0),
            Some(seed.source_seconds),
            24.0,
        ));
    }

    #[test]
    fn prepared_seed_marker_is_overwritten_with_the_newest_live_selection() {
        let mailbox = DecodeCommandMailbox::new();
        assert!(mailbox.select_prepared_seed(1, 30.0).unwrap());
        assert!(mailbox.select_after(1, 0.5, None).unwrap());
        let queued = mailbox.recv();
        assert_eq!(
            queued.command,
            DecodeCommand::Select {
                generation: 1,
                target_seconds: 0.5,
            }
        );
        assert!(!queued.prepared_seed);
        assert_eq!(mailbox.command_overwrites.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn only_a_successful_upload_advances_the_motion_predecessor() {
        let mut decoder = synthetic_decoder(DecodedFrame::synthetic(3, 0, 0));
        let consumed = decoder
            .try_next_ready_frame_result()
            .unwrap()
            .expect("seed frame");
        assert_eq!(consumed.source_generation, 0);

        assert!(decoder.request_source_time(0, 1.0).unwrap());
        let before_upload = decoder.mailbox.recv();
        assert_eq!(before_upload.previous_accepted_codec_identity, None);

        decoder.record_accepted_upload(
            0,
            consumed.source_seconds,
            consumed.codec_identity,
            Duration::from_micros(10),
        );
        assert!(decoder.request_source_time(0, 2.0).unwrap());
        let after_upload = decoder.mailbox.recv();
        assert_eq!(
            after_upload.previous_accepted_codec_identity,
            consumed.codec_identity
        );
    }

    #[test]
    fn hostile_upload_identity_cannot_become_the_motion_predecessor() {
        let mut decoder = synthetic_decoder(DecodedFrame::synthetic(3, 0, 0));
        let consumed = decoder
            .try_next_ready_frame_result()
            .unwrap()
            .expect("seed frame");
        let mut forged = consumed.codec_identity.unwrap();
        forged.pts += 1;
        forged.presentation_ordinal += 1;

        decoder.record_accepted_upload(
            consumed.source_generation,
            consumed.source_seconds,
            Some(forged),
            Duration::from_micros(10),
        );
        assert!(decoder.request_source_time(0, 1.0).unwrap());
        assert_eq!(
            decoder.mailbox.recv().previous_accepted_codec_identity,
            None,
            "a same-generation token that was not paired with the consumed pixels was accepted"
        );

        decoder.record_accepted_upload(
            consumed.source_generation,
            consumed.source_seconds,
            consumed.codec_identity,
            Duration::from_micros(10),
        );
        assert!(decoder.request_source_time(0, 2.0).unwrap());
        assert_eq!(
            decoder.mailbox.recv().previous_accepted_codec_identity,
            consumed.codec_identity
        );
    }

    #[test]
    fn accepted_identity_free_upload_clears_the_prior_motion_predecessor() {
        let published_at = Instant::now();
        let mut decoder = synthetic_decoder_at(DecodedFrame::synthetic(3, 0, 0), published_at);
        let first = decoder
            .try_next_ready_frame_result_at(published_at)
            .unwrap()
            .unwrap();
        decoder.record_accepted_upload(
            first.source_generation,
            first.source_seconds,
            first.codec_identity,
            Duration::from_micros(10),
        );

        let mut identity_free = DecodedFrame::synthetic(4, 0, 0);
        identity_free.codec_identity = None;
        lock_state(&decoder.shared).publish(
            identity_free,
            published_at + Duration::from_millis(1),
            Duration::from_micros(50),
        );
        let second = decoder
            .try_next_ready_frame_result_at(published_at + Duration::from_millis(2))
            .unwrap()
            .unwrap();
        assert_eq!(second.codec_identity, None);
        decoder.record_accepted_upload(
            second.source_generation,
            second.source_seconds,
            second.codec_identity,
            Duration::from_micros(10),
        );
        assert!(decoder.request_source_time(0, 1.0).unwrap());
        assert_eq!(
            decoder.mailbox.recv().previous_accepted_codec_identity,
            None
        );
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
    fn decoder_delivery_holds_require_multiple_continuous_desires() {
        let started = Instant::now();
        let mut state =
            SharedState::healthy_with_seed_at(DecodedFrame::synthetic(1, 0, 0), started);
        state.record_accepted_upload(started, 0);
        state.record_accepted_upload(started + Duration::from_secs(2), 1);
        assert_eq!(
            state.accepted_upload_interval_durations.peak(),
            Some(Duration::from_secs(2))
        );
        assert_eq!(state.decoder_delivery_hold_durations.peak(), None);

        state.record_accepted_upload(started + Duration::from_millis(2_400), 12);
        assert_eq!(
            state.decoder_delivery_hold_durations.peak(),
            Some(Duration::from_millis(400))
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
        decoder.record_accepted_upload(3, 0.0, None, Duration::from_micros(800));
        decoder.record_accepted_upload(3, 0.0, None, Duration::from_micros(250));
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
    fn the_previous_requests_completion_is_delivered_rather_than_discarded() {
        // The render thread publishes a selection and harvests a few
        // microseconds later in the same frame body, so a worker essentially
        // never finishes inside that window. What it does finish is the
        // *previous* request, and that must reach the GPU: guarding the
        // harvest on the per-command epoch discarded it every single frame and
        // playback survived only on rendered frames that were not sample-due.
        let published_at = Instant::now();
        let mut decoder = synthetic_decoder_at(DecodedFrame::synthetic(1, 0, 0), published_at);
        decoder.mailbox.select_after(0, 0.5, None).unwrap();
        let first_epoch = decoder.mailbox.epoch();
        lock_state(&decoder.shared).publish(
            DecodedFrame {
                command_epoch: first_epoch,
                ..DecodedFrame::synthetic(2, 0, 0)
            },
            published_at,
            Duration::from_micros(400),
        );

        // The next frame's request lands before the harvest, exactly as the
        // live loop orders them.
        decoder.mailbox.select_after(0, 0.6, None).unwrap();
        assert_ne!(decoder.mailbox.epoch(), first_epoch);

        let harvested = decoder
            .try_next_ready_frame_result_at(published_at)
            .unwrap()
            .expect("the previous request's completion must still be delivered");
        assert_eq!(harvested.source_generation, 0);
        assert_eq!(
            harvested.pts,
            Some(2),
            "the newest completion wins the mailbox"
        );
    }

    #[test]
    fn a_completion_from_before_a_discontinuity_is_still_discarded() {
        // The generation is the discontinuity token, and it must keep its
        // exact meaning: a seek, cue, or loop retires everything before it.
        let published_at = Instant::now();
        let mut decoder = synthetic_decoder_at(DecodedFrame::synthetic(1, 0, 0), published_at);
        lock_state(&decoder.shared).publish(
            DecodedFrame::synthetic(2, 0, 0),
            published_at,
            Duration::from_micros(400),
        );
        decoder.mailbox.select_after(7, 0.6, None).unwrap();
        assert_eq!(
            decoder.try_next_ready_frame_result_at(published_at),
            Ok(None),
            "a completion from before the discontinuity must never reach the GPU"
        );
    }

    #[test]
    fn newer_same_generation_desire_does_not_cancel_the_running_decode() {
        let mailbox = DecodeCommandMailbox::new();
        let shared = Arc::new(Mutex::new(SharedState {
            latest: None,
            last_published_at: None,
            publish_interval_durations: DurationWindow::default(),
            last_accepted_upload_at: None,
            accepted_upload_interval_durations: DurationWindow::default(),
            decoder_delivery_hold_durations: DurationWindow::default(),
            accepted_uploads: 0,
            frame_overwrites: 0,
            frame_drops: 0,
            stale_commands: 0,
            superseded_decode_attempts: 0,
            stale_completed_frames: 0,
            failed_decode_attempts: 0,
            published_frames: 0,
            pending_frames_peak: 0,
            decode_attempt_durations: DurationWindow::default(),
            decode_durations: DurationWindow::default(),
            frame_age_durations: DurationWindow::default(),
            upload_durations: DurationWindow::default(),
            command_error: None,
            health: DecoderHealth::Healthy,
            health_revision: 1,
        }));
        let first_started = Arc::new((Mutex::new(false), Condvar::new()));
        let first_release = Arc::new((Mutex::new(false), Condvar::new()));
        let second_started = Arc::new((Mutex::new(false), Condvar::new()));
        let second_release = Arc::new((Mutex::new(false), Condvar::new()));

        assert!(mailbox.select(4, 1.0).unwrap());
        let worker_mailbox = mailbox.clone();
        let worker_shared = shared.clone();
        let worker_first_started = first_started.clone();
        let worker_first_release = first_release.clone();
        let worker_second_started = second_started.clone();
        let worker_second_release = second_release.clone();
        let worker = std::thread::spawn(move || {
            run_decode_commands(
                worker_mailbox,
                worker_shared,
                |generation, target_seconds, _, _, _, _| {
                    let (started, release, value) = if target_seconds < 1.5 {
                        (&worker_first_started, &worker_first_release, 10)
                    } else {
                        (&worker_second_started, &worker_second_release, 20)
                    };
                    *lock_recover(&started.0) = true;
                    started.1.notify_one();
                    let mut released = lock_recover(&release.0);
                    while !*released {
                        released = release
                            .1
                            .wait(released)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                    Ok(DecodedFrame::synthetic(value, generation, 0))
                },
            );
        });

        let mut did_start = lock_recover(&first_started.0);
        while !*did_start {
            did_start = first_started
                .1
                .wait(did_start)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        drop(did_start);

        let first_epoch = mailbox.epoch();
        assert!(mailbox.select(4, 2.0).unwrap());
        assert!(!mailbox.is_current(first_epoch));
        assert!(mailbox.is_generation_current(4));
        *lock_recover(&first_release.0) = true;
        first_release.1.notify_one();

        let mut second_did_start = lock_recover(&second_started.0);
        while !*second_did_start {
            second_did_start = second_started
                .1
                .wait(second_did_start)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        drop(second_did_start);
        {
            let state = lock_state(&shared);
            assert_eq!(
                state.latest.as_ref().map(|frame| frame.rgba.as_slice()),
                Some(&[10][..])
            );
            assert_eq!(state.published_frames, 1);
            assert_eq!(state.frame_drops, 0);
            assert_eq!(state.decode_attempt_durations.total_samples, 1);
        }

        *lock_recover(&second_release.0) = true;
        second_release.1.notify_one();
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let state = lock_state(&shared);
            if state.published_frames == 2 {
                assert_eq!(
                    state.latest.as_ref().map(|frame| frame.rgba.as_slice()),
                    Some(&[20][..])
                );
                assert_eq!(state.frame_drops, 0);
                assert_eq!(state.decode_attempt_durations.total_samples, 2);
                break;
            }
            drop(state);
            assert!(Instant::now() < deadline, "second desire did not publish");
            std::thread::yield_now();
        }
        assert_eq!(mailbox.continuous_requests.load(Ordering::Relaxed), 1);
        assert_eq!(mailbox.discontinuity_requests.load(Ordering::Relaxed), 1);
        mailbox.stop();
        worker.join().unwrap();
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
            publish_interval_durations: DurationWindow::default(),
            last_accepted_upload_at: None,
            accepted_upload_interval_durations: DurationWindow::default(),
            decoder_delivery_hold_durations: DurationWindow::default(),
            accepted_uploads: 0,
            frame_overwrites: 0,
            frame_drops: 0,
            stale_commands: 0,
            superseded_decode_attempts: 0,
            stale_completed_frames: 0,
            failed_decode_attempts: 0,
            published_frames: 0,
            pending_frames_peak: 0,
            decode_attempt_durations: DurationWindow::default(),
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
            run_decode_commands(
                worker_mailbox,
                worker_shared,
                |generation, _, _, _, _, _| {
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
                },
            );
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
            publish_interval_durations: DurationWindow::default(),
            last_accepted_upload_at: None,
            accepted_upload_interval_durations: DurationWindow::default(),
            decoder_delivery_hold_durations: DurationWindow::default(),
            accepted_uploads: 0,
            frame_overwrites: 0,
            frame_drops: 0,
            stale_commands: 0,
            superseded_decode_attempts: 0,
            stale_completed_frames: 0,
            failed_decode_attempts: 0,
            published_frames: 0,
            pending_frames_peak: 0,
            decode_attempt_durations: DurationWindow::default(),
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
            run_decode_commands(
                worker_mailbox,
                worker_shared,
                |generation, _, _, _, _, _| {
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
                },
            );
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
                assert_eq!(frame.rgba.as_slice(), [20]);
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
            rgba: DecodedImagePayload::from_owned_rgba(vec![7]),
            metadata: FrameMetadata::sanitized(42, Some(7), 0.25, 1.0),
            codec_motion: Some(synthetic_codec_motion(41, 7).into()),
        };
        let frame = DecodedFrame::from_video(decoded, 0.25, 0, 1);
        assert_eq!(frame.source_generation, 42);
        assert!(frame.codec_motion.is_none());
    }

    #[test]
    fn forward_mailbox_delivery_preserves_immutable_payload_identity() {
        let payload = DecodedImagePayload::from_owned_rgba(vec![3, 4, 5, 6]);
        let payload_id = payload.identity();
        let decoded = DecodedVideoFrame {
            rgba: payload,
            metadata: FrameMetadata::sanitized(12, Some(90), 3.0, 10.0),
            codec_motion: None,
        };
        let frame = DecodedFrame::from_video(decoded, 0.3, 2, 1);
        assert_eq!(frame.rgba.identity(), payload_id);
        let mut decoder = synthetic_decoder(frame);
        decoder.request_source_time(12, 3.0).unwrap();
        let ready = decoder.try_next_ready_frame_result().unwrap().unwrap();
        assert_eq!(ready.rgba.identity(), payload_id);
        assert_eq!(ready.rgba.as_slice(), &[3, 4, 5, 6]);
        assert_eq!(ready.source_generation, 12);
        assert_eq!(ready.pts, Some(90));
    }

    #[test]
    fn initial_frame_is_seeded_with_source_metadata() {
        let mut decoder = synthetic_decoder(DecodedFrame::synthetic(4, 0, 0));
        let ready = decoder.try_next_ready_frame_result().unwrap().unwrap();
        assert_eq!(ready.rgba.as_slice(), [4]);
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

    /// Real-media delivery receipt for the two corrected Cygnus controls. It
    /// intentionally stays opt-in: the source files are operator-owned and a
    /// useful run consumes real wall time. The fixture drives two independent
    /// workers with the same rational 30 Hz, request-before-harvest order as
    /// the live frame loop, while treating each harvested image as an accepted
    /// upload so codec-motion predecessor semantics remain exercised.
    #[test]
    #[ignore = "requires COLLIDEOSCOPE_DELIVERY_SWAN_FIXTURE and COLLIDEOSCOPE_DELIVERY_NATURE_FIXTURE"]
    fn real_two_source_thirty_hertz_delivery_receipt() {
        let paths = [
            std::env::var("COLLIDEOSCOPE_DELIVERY_SWAN_FIXTURE")
                .expect("set COLLIDEOSCOPE_DELIVERY_SWAN_FIXTURE"),
            std::env::var("COLLIDEOSCOPE_DELIVERY_NATURE_FIXTURE")
                .expect("set COLLIDEOSCOPE_DELIVERY_NATURE_FIXTURE"),
        ];
        let seconds = std::env::var("COLLIDEOSCOPE_DELIVERY_RECEIPT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(60)
            .clamp(1, 300);
        let warmup_seconds = std::env::var("COLLIDEOSCOPE_DELIVERY_WARMUP_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(10)
            .min(60);
        let fps = 30_u64;
        let ticks = seconds.saturating_mul(fps);
        let warmup_ticks = warmup_seconds.saturating_mul(fps);
        let total_ticks = warmup_ticks.saturating_add(ticks);
        let mut decoders = paths
            .iter()
            .map(|path| {
                ThreadedDecoder::open_with_media_policy(
                    path,
                    &MediaSafetyPolicy::safe(),
                    MediaDeviceLimits::none(),
                )
                .unwrap_or_else(|error| panic!("could not open {path}: {error}"))
            })
            .collect::<Vec<_>>();
        for (decoder, path) in decoders.iter().zip(&paths) {
            assert!(
                decoder.duration_seconds > (warmup_seconds + seconds) as f64 + 1.0,
                "{path} is too short for an uninterrupted {warmup_seconds}s warm-up plus {seconds}s receipt"
            );
        }

        let started = Instant::now();
        let mut accepted_counts = vec![0_u64; decoders.len()];
        let mut accepted_identities = vec![std::collections::HashSet::new(); decoders.len()];
        let mut last_accepted_at = vec![started; decoders.len()];
        let mut last_accepted_source_seconds = vec![0.0; decoders.len()];
        let mut peak_holds = vec![Duration::ZERO; decoders.len()];
        let mut telemetry_baselines = vec![DecoderTelemetry::default(); decoders.len()];

        for (index, decoder) in decoders.iter_mut().enumerate() {
            let seed = decoder
                .try_next_ready_frame_result()
                .unwrap_or_else(|error| panic!("seed failed for {}: {error}", paths[index]))
                .unwrap_or_else(|| panic!("no seed frame for {}", paths[index]));
            accepted_identities[index].insert(seed.source_seconds.to_bits());
            decoder.record_accepted_upload(
                seed.source_generation,
                seed.source_seconds,
                seed.codec_identity,
                Duration::ZERO,
            );
            accepted_counts[index] = 1;
            last_accepted_at[index] = Instant::now();
            last_accepted_source_seconds[index] = seed.source_seconds;
        }
        if warmup_ticks == 0 {
            for (index, decoder) in decoders.iter().enumerate() {
                telemetry_baselines[index] = decoder.telemetry();
                accepted_counts[index] = 0;
                accepted_identities[index].clear();
                last_accepted_at[index] = Instant::now();
            }
        }

        for ordinal in 1..=total_ticks {
            let deadline =
                started + Duration::from_nanos(ordinal.saturating_mul(1_000_000_000) / fps);
            loop {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                let remaining = deadline.saturating_duration_since(now);
                if remaining > Duration::from_millis(2) {
                    std::thread::sleep(remaining - Duration::from_millis(1));
                } else {
                    std::thread::yield_now();
                }
            }

            let measuring = ordinal > warmup_ticks;
            for (index, decoder) in decoders.iter_mut().enumerate() {
                let target_seconds = ordinal as f64 / fps as f64;
                decoder
                    .request_source_time(0, target_seconds)
                    .unwrap_or_else(|error| {
                        panic!("selection failed for {}: {error}", paths[index])
                    });
                if let Some(frame) = decoder
                    .try_next_ready_frame_result()
                    .unwrap_or_else(|error| panic!("harvest failed for {}: {error}", paths[index]))
                {
                    let accepted_at = Instant::now();
                    if measuring {
                        let hold = accepted_at.saturating_duration_since(last_accepted_at[index]);
                        peak_holds[index] = peak_holds[index].max(hold);
                        if hold > Duration::from_nanos(4_000_000_000 / fps) {
                            eprintln!(
                                "delivery hold {}: target={target_seconds:.6}s previous_source={:.6}s accepted_source={:.6}s hold_ms={}",
                                paths[index],
                                last_accepted_source_seconds[index],
                                frame.source_seconds,
                                hold.as_millis(),
                            );
                        }
                        accepted_identities[index].insert(frame.source_seconds.to_bits());
                        accepted_counts[index] = accepted_counts[index].saturating_add(1);
                    }
                    last_accepted_at[index] = accepted_at;
                    last_accepted_source_seconds[index] = frame.source_seconds;
                    decoder.record_accepted_upload(
                        frame.source_generation,
                        frame.source_seconds,
                        frame.codec_identity,
                        Duration::ZERO,
                    );
                }
            }
            if ordinal == warmup_ticks {
                for (index, decoder) in decoders.iter().enumerate() {
                    telemetry_baselines[index] = decoder.telemetry();
                    accepted_counts[index] = 0;
                    accepted_identities[index].clear();
                    peak_holds[index] = Duration::ZERO;
                    last_accepted_at[index] = Instant::now();
                }
            }
        }

        let mut failures = Vec::new();
        for (index, decoder) in decoders.iter().enumerate() {
            let telemetry = decoder.telemetry();
            let baseline = telemetry_baselines[index];
            let delivered = accepted_counts[index];
            let distinct = accepted_identities[index].len() as u64;
            let expected_distinct = (seconds as f64 * f64::from(decoder.fps.min(fps as f32)))
                .floor()
                .max(1.0) as u64;
            let continuous_overwrites = telemetry
                .continuous_overwrites
                .saturating_sub(baseline.continuous_overwrites);
            let stale_commands = telemetry
                .stale_commands
                .saturating_sub(baseline.stale_commands);
            let cancelled = telemetry
                .superseded_decode_attempts
                .saturating_sub(baseline.superseded_decode_attempts);
            let failed = telemetry
                .failed_decode_attempts
                .saturating_sub(baseline.failed_decode_attempts);
            eprintln!(
                "delivery receipt {}: warmup_s={} requested={} expected_distinct={} distinct={} accepted={} continuous_overwrites={} stale={} cancelled={} failed={} all_attempt_p95_us={} successful_p95_us={} hold_p95_ms={} hold_p99_ms={} hold_max_ms={} decoder_hold_p95_ms={} decoder_hold_max_ms={} current_hold_ms={}",
                paths[index],
                warmup_seconds,
                ticks,
                expected_distinct,
                distinct,
                delivered,
                continuous_overwrites,
                stale_commands,
                cancelled,
                failed,
                telemetry
                    .decode_attempt_p95_duration
                    .map_or(0, |value| value.as_micros()),
                telemetry
                    .decode_p95_duration
                    .map_or(0, |value| value.as_micros()),
                telemetry
                    .accepted_upload_interval_p95_duration
                    .map_or(0, |value| value.as_millis()),
                telemetry
                    .accepted_upload_interval_p99_duration
                    .map_or(0, |value| value.as_millis()),
                peak_holds[index].as_millis(),
                telemetry
                    .decoder_delivery_hold_p95_duration
                    .map_or(0, |value| value.as_millis()),
                telemetry
                    .peak_decoder_delivery_hold_duration
                    .map_or(0, |value| value.as_millis()),
                Instant::now()
                    .saturating_duration_since(last_accepted_at[index])
                    .as_millis(),
            );
            if stale_commands != 0 || cancelled != 0 || failed != 0 {
                failures.push(format!(
                    "{} reported stale/cancelled/failed {stale_commands}/{cancelled}/{failed}",
                    paths[index]
                ));
            }
            if delivered.saturating_mul(100) < expected_distinct.saturating_mul(95) {
                failures.push(format!(
                    "{} accepted only {delivered}/{expected_distinct} expected due source identities",
                    paths[index]
                ));
            }
            if distinct.saturating_mul(100) < expected_distinct.saturating_mul(95) {
                failures.push(format!(
                    "{} delivered only {distinct}/{expected_distinct} distinct due source identities",
                    paths[index]
                ));
            }
            if peak_holds[index] > Duration::from_nanos(4_000_000_000 / fps) {
                failures.push(format!(
                    "{} held a source image for {:?}",
                    paths[index], peak_holds[index]
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("; "));
    }

    #[test]
    fn codec_dimensions_are_checked_before_seed_decode() {
        assert!(validate_decode_dimensions(0, 1080, Some(8192)).is_err());
        assert!(validate_decode_dimensions(1920, 0, Some(8192)).is_err());
        assert!(validate_decode_dimensions(8193, 1080, Some(8192)).is_err());
        assert!(validate_decode_dimensions(1920, 1080, Some(8192)).is_ok());
    }
}
