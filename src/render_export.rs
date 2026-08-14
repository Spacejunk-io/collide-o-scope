//! Offline high-quality patch renderer.
//!
//! Renders a patch (layer configs + master effects + NTSC) to an MP4 file
//! at configurable resolution and duration using a headless wgpu device
//! and piping raw RGBA frames to ffmpeg.

use std::io::{Read as IoRead, Write as IoWrite};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::effects::EffectUniforms;
use crate::layers::BlendMode;
use crate::ntsc::{
    plan_selective_ntsc, process_selective_ntsc_batch_with_state, reference_frame_for_output,
    NtscFrameMetadata, NtscState, SelectiveNtscBatch, SelectiveNtscGeneration,
    SelectiveNtscLayerDescriptor, SelectiveNtscPlan,
};
use crate::patch::PatchState;
use crate::renderer::state::{
    conditional_layer_slots, master_fx_composition_path, visible_stack_indices,
    MasterFxCompositionPath,
};
use crate::video::decoder::{validate_media_dimensions, MAX_MEDIA_PIXELS};
use crate::video::{DecodedStillImage, VideoDecoder};

/// App shutdown must not wait forever on a wedged graphics backend. By the
/// time this expires ExportJob::cancel has already killed/reaped ffmpeg,
/// removed its partial output, and forbidden any later encoder registration.
const EXPORT_DROP_JOIN_TIMEOUT: Duration = Duration::from_secs(1);
/// Public export presets top out at UHD. Keep arbitrary protocol input from
/// allocating a pathological set of full-frame GPU intermediates.
const MAX_EXPORT_EDGE: u32 = 16_384;
const OUTCOME_RUNNING: u8 = 0;
const OUTCOME_CANCEL_REQUESTED: u8 = 1;
const OUTCOME_SUCCEEDED: u8 = 2;
const OUTCOME_FAILED: u8 = 3;
const OUTCOME_CANCELLED: u8 = 4;

const fn transparent_accumulation_clear() -> wgpu::Color {
    wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    }
}

/// Configuration for an offline render job.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration_secs: f32,
    pub output_path: String,
    /// Optional media file whose first audio stream is muxed into the MP4.
    ///
    /// Audio transport is deliberately independent from visual layer
    /// transport: it starts at source time zero at 1x, ignores layer
    /// pause/speed/modulation/loop state, is trimmed when long, and is padded
    /// with silence when short. This remains deterministic when visual speed
    /// changes over time and cannot be represented by one audio tempo.
    pub audio_path: Option<String>,
}

/// Shared state for progress/cancellation between the render thread and the UI.
pub struct ExportProgress {
    /// 0..10000 representing 0.0%..100.0%
    pub progress: AtomicU32,
    /// Set to true to request cancellation.
    pub cancel: Arc<AtomicBool>,
    /// Set to true once cancellation cleanup has completed.
    pub cancelled: AtomicBool,
    /// Set to true when the job is complete (success or failure).
    pub done: AtomicBool,
    /// Error message if the job failed (empty = success).
    pub error: std::sync::Mutex<String>,
    /// Atomic arbitration between a cancellation request and success/failure.
    /// Public cancellation changes RUNNING -> CANCEL_REQUESTED without locks;
    /// only an owner that has completed external cleanup may publish a
    /// terminal outcome.
    outcome: AtomicU8,
    /// Serializes publication of the user-visible terminal fields.
    terminal: Mutex<()>,
    /// True only after ffmpeg has started and may have created/truncated the
    /// destination. Early cancellation must not delete a pre-existing file.
    output_started: AtomicBool,
    /// The supervisor owns an encoder process or is in its spawn window.
    encoder_active: AtomicBool,
    /// True only when no encoder process can still be alive or appear late.
    encoder_cleanup_complete: AtomicBool,
    /// Set immediately before the worker deliberately closes encoder stdin.
    /// An encoder exit without this flag is treated as an unexpected worker
    /// failure rather than a successful completion.
    encoder_finish_requested: AtomicBool,
    /// Internal worker failure/panic requests encoder shutdown without being
    /// mislabeled as a user cancellation.
    abort: AtomicBool,
    pending_worker_error: Mutex<Option<String>>,
}

impl ExportProgress {
    pub fn new() -> Self {
        Self {
            progress: AtomicU32::new(0),
            cancel: Arc::new(AtomicBool::new(false)),
            cancelled: AtomicBool::new(false),
            done: AtomicBool::new(false),
            error: std::sync::Mutex::new(String::new()),
            outcome: AtomicU8::new(OUTCOME_RUNNING),
            terminal: Mutex::new(()),
            output_started: AtomicBool::new(false),
            encoder_active: AtomicBool::new(false),
            encoder_cleanup_complete: AtomicBool::new(true),
            encoder_finish_requested: AtomicBool::new(false),
            abort: AtomicBool::new(false),
            pending_worker_error: Mutex::new(None),
        }
    }

    pub fn progress_f32(&self) -> f32 {
        self.progress.load(Ordering::Relaxed) as f32 / 10000.0
    }

    fn terminal_guard(&self) -> MutexGuard<'_, ()> {
        self.terminal
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Request cancellation without taking a lock or waiting for the process.
    /// The encoder supervisor observes this flag, kills/reaps its child, and
    /// publishes terminal cancellation only after deleting the partial file.
    fn request_cancel(&self) {
        if self
            .outcome
            .compare_exchange(
                OUTCOME_RUNNING,
                OUTCOME_CANCEL_REQUESTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
            || self.outcome.load(Ordering::Acquire) == OUTCOME_CANCEL_REQUESTED
        {
            self.cancel.store(true, Ordering::Release);
            log::debug!("export cancellation requested");
        }
    }
}

/// Handle to a running export job.
pub struct ExportJob {
    pub progress: Arc<ExportProgress>,
    thread: Option<std::thread::JoinHandle<()>>,
    output_path: String,
}

impl ExportJob {
    /// Start an export job on a background thread.
    #[cfg(test)]
    pub fn start(patch: PatchState, config: ExportConfig, library_folder: &str) -> Self {
        Self::start_inner(patch, config, library_folder, None)
    }

    /// Live exports share the renderer device so completing or cancelling an
    /// export cannot tear down a second backend device underneath the live
    /// presentation loop.
    pub fn start_with_gpu(
        patch: PatchState,
        config: ExportConfig,
        library_folder: &str,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        Self::start_inner(patch, config, library_folder, Some((device, queue)))
    }

    fn start_inner(
        patch: PatchState,
        config: ExportConfig,
        library_folder: &str,
        shared_gpu: Option<(wgpu::Device, wgpu::Queue)>,
    ) -> Self {
        let progress = Arc::new(ExportProgress::new());
        let prog = progress.clone();
        let lib_folder = library_folder.to_string();
        let output_path = config.output_path.clone();

        let thread = std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_export(&patch, &config, &prog, &lib_folder, shared_gpu)
            }));
            let worker_error = match result {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error),
                Err(payload) => {
                    let detail = payload
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("unknown panic");
                    Some(format!("export worker panicked: {detail}"))
                }
            };

            finalize_export_worker(&prog, &config.output_path, worker_error);
        });

        Self {
            progress,
            thread: Some(thread),
            output_path,
        }
    }

    /// Check if the job is done.
    pub fn is_done(&self) -> bool {
        self.progress.done.load(Ordering::Acquire)
    }

    /// Request cancellation. This method is deliberately lock-free and does
    /// not claim terminal completion; the supervisor owns process cleanup.
    pub fn cancel(&self) {
        self.progress.request_cancel();
    }

    /// A terminal job may be replaced once the supervisor has proven that no
    /// encoder process or partial artifact remains. Drop gives the render
    /// worker its bounded grace period; a wedged GPU destructor cannot
    /// permanently disable the Render button.
    pub fn can_replace(&self) -> bool {
        self.is_done()
            && self
                .progress
                .encoder_cleanup_complete
                .load(Ordering::Acquire)
    }
}

impl Drop for ExportJob {
    fn drop(&mut self) {
        // The deadline covers the whole Drop operation, including the cancel
        // request. cancel() is lock-free; any encoder remains owned by its
        // dedicated supervisor even if GPU cleanup outlives this handle.
        let deadline = std::time::Instant::now() + EXPORT_DROP_JOIN_TIMEOUT;
        self.cancel();

        let Some(thread) = self.thread.take() else {
            return;
        };
        if thread.thread().id() == std::thread::current().id() {
            // Safe construction never transfers an ExportJob into its own
            // worker, but avoid a self-join panic if that invariant is ever
            // changed. Cancellation is already visible to the worker.
            log::error!("ExportJob dropped from its own worker; skipping self-join");
            return;
        }

        while !thread.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if !thread.is_finished() {
            // Rust has no safe timed join or thread termination. The render
            // worker owns only its GPU state and a supervisor JoinHandle; the
            // supervisor thread itself remains the sole live Child owner and
            // will kill/reap it before publishing a terminal result.
            log::error!(
                "export worker did not exit within {:?}; detaching isolated GPU cleanup",
                EXPORT_DROP_JOIN_TIMEOUT
            );
            return;
        }

        let worker_panicked = thread.join().is_err();
        if !self.progress.done.load(Ordering::Acquire) {
            finalize_export_worker(
                &self.progress,
                &self.output_path,
                Some(if worker_panicked {
                    "export worker terminated unexpectedly".to_string()
                } else {
                    "export worker exited without publishing terminal state".to_string()
                }),
            );
        }
    }
}

fn finalize_export_worker(
    progress: &ExportProgress,
    output_path: &str,
    worker_error: Option<String>,
) {
    // If the render worker unwound while a supervisor still owns ffmpeg, that
    // supervisor is responsible for reap/removal and terminal publication.
    // Publishing here would make `done` lie about external cleanup.
    if progress.encoder_active.load(Ordering::Acquire)
        && !progress.encoder_cleanup_complete.load(Ordering::Acquire)
    {
        if let Some(error) = worker_error {
            *progress
                .pending_worker_error
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(error);
            progress.abort.store(true, Ordering::Release);
        }
        return;
    }

    // Serialize the final commit with ExportJob::cancel(). Whichever obtains
    // this lock first defines whether cancellation was still accepted while
    // the job was running.
    let _terminal = progress.terminal_guard();
    if progress.done.load(Ordering::Acquire) {
        return;
    }
    let mut decision = progress.outcome.load(Ordering::Acquire);
    if worker_error.is_some() && decision == OUTCOME_RUNNING {
        decision = match progress.outcome.compare_exchange(
            OUTCOME_RUNNING,
            OUTCOME_FAILED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => OUTCOME_FAILED,
            Err(actual) => actual,
        };
    }
    let error = if decision == OUTCOME_CANCEL_REQUESTED {
        let cleanup_error = remove_started_output(progress, Some(output_path));
        progress.cancelled.store(true, Ordering::Relaxed);
        progress.outcome.store(OUTCOME_CANCELLED, Ordering::Release);
        Some(match cleanup_error {
            Some(error) => format!("export cancelled; {error}"),
            None => "export cancelled".to_string(),
        })
    } else if decision == OUTCOME_FAILED {
        let worker_error = worker_error
            .unwrap_or_else(|| "export worker failed without diagnostic detail".to_string());
        // ffmpeg opens/truncates the destination at startup. Any later error
        // or caught panic must therefore remove that incomplete artifact just
        // as cancellation does.
        let cleanup_error = remove_started_output(progress, Some(output_path));
        Some(match cleanup_error {
            Some(error) => format!("{worker_error}; {error}"),
            None => worker_error,
        })
    } else if decision == OUTCOME_RUNNING {
        if progress
            .outcome
            .compare_exchange(
                OUTCOME_RUNNING,
                OUTCOME_SUCCEEDED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            // Cancellation won after the first decision load but before the
            // success commit. Cleanup is already complete on this path.
            let cleanup_error = remove_started_output(progress, Some(output_path));
            progress.cancelled.store(true, Ordering::Relaxed);
            progress.outcome.store(OUTCOME_CANCELLED, Ordering::Release);
            let error = match cleanup_error {
                Some(error) => format!("export cancelled; {error}"),
                None => "export cancelled".to_string(),
            };
            *progress
                .error
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = error;
            progress.done.store(true, Ordering::Release);
            return;
        }
        None
    } else {
        return;
    };
    if let Some(error) = error {
        *progress
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = error;
    }
    // Release pairs with is_done's Acquire, publishing cancelled/error before
    // the UI observes the terminal state.
    progress.done.store(true, Ordering::Release);
}

fn remove_started_output(progress: &ExportProgress, output_path: Option<&str>) -> Option<String> {
    if !progress.output_started.load(Ordering::Acquire) {
        return None;
    }
    output_path.and_then(|path| remove_partial_output(path).err())
}

/// Publish cancellation after process/file cleanup but before the large wgpu
/// object graph local to `run_export` is dropped. This keeps the UI terminal
/// even if a graphics backend takes unusually long to destroy resources.
fn publish_cancelled_terminal(progress: &ExportProgress, output_path: Option<&str>) {
    let _terminal = progress.terminal_guard();
    if progress.done.load(Ordering::Acquire) {
        return;
    }
    if progress.outcome.load(Ordering::Acquire) != OUTCOME_CANCEL_REQUESTED {
        return;
    }
    let cleanup_error = remove_started_output(progress, output_path);
    progress.cancelled.store(true, Ordering::Relaxed);
    progress.outcome.store(OUTCOME_CANCELLED, Ordering::Release);
    *progress
        .error
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = match cleanup_error {
        Some(error) => format!("export cancelled; {error}"),
        None => "export cancelled".to_string(),
    };
    progress.done.store(true, Ordering::Release);
}

fn remove_partial_output(path: &str) -> Result<(), String> {
    for attempt in 0..5 {
        match std::fs::remove_file(path) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) if attempt == 4 => {
                return Err(format!("failed to remove partial output '{path}': {error}"));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    unreachable!()
}

fn check_cancelled(progress: &ExportProgress) -> Result<(), String> {
    if progress.cancel.load(Ordering::Acquire) {
        // All current callers are before ffmpeg startup, so no partial output
        // exists and terminal state can be published immediately.
        publish_cancelled_terminal(progress, None);
        Err("export cancelled".to_string())
    } else {
        Ok(())
    }
}

/// Wait without surrendering cancellation control to a blocking Child::wait.
/// Once cancellation is observed, `kill` is issued and the process is still
/// reaped before this function returns.
fn wait_for_ffmpeg(
    child: &mut Child,
    cancel: &AtomicBool,
    abort: &AtomicBool,
) -> std::io::Result<ExitStatus> {
    let mut kill_sent = false;
    loop {
        if (cancel.load(Ordering::Acquire) || abort.load(Ordering::Acquire)) && !kill_sent {
            match child.kill() {
                Ok(()) => kill_sent = true,
                Err(error) => log::warn!("encoder kill request failed; retrying: {error}"),
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {}
            Err(error) => {
                // Keep ownership and keep polling. Returning would drop an
                // unproven Child and make terminal cleanup claims false.
                log::warn!("encoder status poll failed; retaining child: {error}");
                kill_sent = false;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn force_reap_ffmpeg(child: &mut Child) -> ExitStatus {
    static TRUE_FLAG: AtomicBool = AtomicBool::new(true);
    static FALSE_FLAG: AtomicBool = AtomicBool::new(false);
    // This helper deliberately does not return until try_wait proves exit.
    // It runs only on the supervisor thread, so a pathological OS failure can
    // neither block the UI nor detach the still-owned Child.
    wait_for_ffmpeg(child, &TRUE_FLAG, &FALSE_FLAG)
        .expect("wait_for_ffmpeg retains ownership instead of returning errors")
}

struct EncoderCompletion {
    status: Result<ExitStatus, String>,
    stderr: Result<Vec<u8>, String>,
}

struct EncoderSession {
    stdin: ChildStdin,
    completion: std::sync::mpsc::Receiver<EncoderCompletion>,
    supervisor: std::thread::JoinHandle<()>,
}

/// Start the encoder under a dedicated supervisor. The supervisor is the only
/// owner of `Child`; the render worker receives only stdin and a completion
/// channel. Consequently no cancellation, worker unwind, or bounded Drop can
/// detach an unreaped encoder process.
fn start_encoder_supervisor(
    program: String,
    args: Vec<String>,
    progress: Arc<ExportProgress>,
    output_path: String,
) -> Result<EncoderSession, String> {
    progress.encoder_active.store(true, Ordering::Release);
    progress
        .encoder_cleanup_complete
        .store(false, Ordering::Release);
    progress
        .encoder_finish_requested
        .store(false, Ordering::Release);

    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let (completion_tx, completion_rx) = std::sync::mpsc::sync_channel(1);
    let supervisor_progress = progress.clone();
    let supervisor = match std::thread::Builder::new()
        .name("export-encoder-supervisor".into())
        .spawn(move || {
            let finish_without_child = |error: String| {
                supervisor_progress
                    .encoder_active
                    .store(false, Ordering::Release);
                supervisor_progress
                    .encoder_cleanup_complete
                    .store(true, Ordering::Release);
                if supervisor_progress.cancel.load(Ordering::Acquire) {
                    publish_cancelled_terminal(&supervisor_progress, None);
                } else {
                    finalize_export_worker(&supervisor_progress, &output_path, Some(error.clone()));
                }
                let _ = ready_tx.send(Err(error));
            };

            if supervisor_progress.cancel.load(Ordering::Acquire) {
                finish_without_child("export cancelled before encoder startup".to_string());
                return;
            }

            let mut command = Command::new(&program);
            command
                .args(&args)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
            let mut child = match command.spawn() {
                Ok(child) => child,
                Err(error) => {
                    finish_without_child(format!("Failed to spawn {program}: {error}"));
                    return;
                }
            };
            log::debug!("export encoder supervisor spawned process {}", child.id());
            supervisor_progress
                .output_started
                .store(true, Ordering::Release);

            let Some(stdin) = child.stdin.take() else {
                let _ = force_reap_ffmpeg(&mut child);
                supervisor_progress
                    .encoder_active
                    .store(false, Ordering::Release);
                supervisor_progress
                    .encoder_cleanup_complete
                    .store(true, Ordering::Release);
                let error = "ffmpeg stdin pipe unavailable".to_string();
                finalize_export_worker(&supervisor_progress, &output_path, Some(error.clone()));
                let _ = ready_tx.send(Err(error));
                return;
            };
            let Some(mut stderr) = child.stderr.take() else {
                drop(stdin);
                let _ = force_reap_ffmpeg(&mut child);
                supervisor_progress
                    .encoder_active
                    .store(false, Ordering::Release);
                supervisor_progress
                    .encoder_cleanup_complete
                    .store(true, Ordering::Release);
                let error = "ffmpeg stderr pipe unavailable".to_string();
                finalize_export_worker(&supervisor_progress, &output_path, Some(error.clone()));
                let _ = ready_tx.send(Err(error));
                return;
            };
            let stderr_thread = match std::thread::Builder::new()
                .name("export-ffmpeg-stderr".into())
                .spawn(move || {
                    let mut bytes = Vec::new();
                    let result = stderr.read_to_end(&mut bytes).map(|_| bytes);
                    result.map_err(|error| format!("ffmpeg stderr read: {error}"))
                }) {
                Ok(thread) => thread,
                Err(error) => {
                    drop(stdin);
                    let _ = force_reap_ffmpeg(&mut child);
                    supervisor_progress
                        .encoder_active
                        .store(false, Ordering::Release);
                    supervisor_progress
                        .encoder_cleanup_complete
                        .store(true, Ordering::Release);
                    let error = format!("failed to start ffmpeg stderr reader: {error}");
                    finalize_export_worker(&supervisor_progress, &output_path, Some(error.clone()));
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };

            if ready_tx.send(Ok(stdin)).is_err() {
                supervisor_progress.request_cancel();
            }

            log::debug!("export encoder supervisor waiting for process cleanup");
            let status = wait_for_ffmpeg(
                &mut child,
                &supervisor_progress.cancel,
                &supervisor_progress.abort,
            )
            .map_err(|error| format!("ffmpeg wait: {error}"));
            // `wait_for_ffmpeg` returns only after the child is reaped. The
            // stderr pipe is then closed, so this join cannot await a writer.
            let stderr = stderr_thread
                .join()
                .map_err(|_| "ffmpeg stderr reader panicked".to_string())
                .and_then(|result| result);
            log::debug!("export encoder supervisor reaped process");

            supervisor_progress
                .encoder_active
                .store(false, Ordering::Release);
            supervisor_progress
                .encoder_cleanup_complete
                .store(true, Ordering::Release);

            if supervisor_progress.cancel.load(Ordering::Acquire) {
                publish_cancelled_terminal(&supervisor_progress, Some(&output_path));
                log::debug!("export encoder supervisor published cancelled terminal state");
            } else if supervisor_progress.abort.load(Ordering::Acquire) {
                let error = supervisor_progress
                    .pending_worker_error
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
                    .unwrap_or_else(|| "export worker aborted unexpectedly".to_string());
                finalize_export_worker(&supervisor_progress, &output_path, Some(error));
            } else if !supervisor_progress
                .encoder_finish_requested
                .load(Ordering::Acquire)
            {
                finalize_export_worker(
                    &supervisor_progress,
                    &output_path,
                    Some("export worker ended before closing the encoder cleanly".to_string()),
                );
            }

            let _ = completion_tx.send(EncoderCompletion { status, stderr });
        }) {
        Ok(thread) => thread,
        Err(error) => {
            progress.encoder_active.store(false, Ordering::Release);
            progress
                .encoder_cleanup_complete
                .store(true, Ordering::Release);
            return Err(format!("failed to start encoder supervisor: {error}"));
        }
    };

    match ready_rx.recv() {
        Ok(Ok(stdin)) => Ok(EncoderSession {
            stdin,
            completion: completion_rx,
            supervisor,
        }),
        Ok(Err(error)) => {
            let _ = supervisor.join();
            Err(error)
        }
        Err(_) => {
            let _ = supervisor.join();
            progress.encoder_active.store(false, Ordering::Release);
            progress
                .encoder_cleanup_complete
                .store(true, Ordering::Release);
            Err("encoder supervisor exited before startup completed".to_string())
        }
    }
}

/// Uniforms for the composite shader (must match renderer/state.rs).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    opacity: f32,
    blend_mode: u32,
    _pad: [f32; 2],
}

/// Upload one small export uniform without mapping the new buffer.
///
/// `DeviceExt::create_buffer_init` maps at creation and immediately accesses
/// the mapped range. A queue upload preserves the exact bytes while allowing
/// device-loss validation to remain recoverable instead of panicking in the
/// mapped-range accessor.
fn create_uploaded_uniform<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    value: &T,
) -> wgpu::Buffer {
    let bytes = bytemuck::bytes_of(value);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// Internal layer state for offline rendering.
struct ExportLayer {
    /// Index in the saved patch. Failed-to-open layers must not shift
    /// layer-specific modulation routes onto a different video.
    source_index: usize,
    /// `None` is an explicit deterministic-black placeholder for a live,
    /// missing, or undecodable source that cannot be sampled offline.
    decoder: Option<VideoDecoder>,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    effects: EffectUniforms,
    opacity: f32,
    blend_mode: BlendMode,
    bypass_master_fx: bool,
    speed: f32,
    visible: bool,
    paused: bool,
    fps: f32,
    width: u32,
    height: u32,
}

/// Frame-local, morphable render values detached from decoder/GPU ownership.
/// Keeping these values named prevents positional mistakes as the persisted
/// layer state grows.
#[derive(Clone, Copy)]
struct ExportFrameLayerBase {
    effects: EffectUniforms,
    opacity: f32,
    speed: f32,
    fps: f32,
    blend_mode: BlendMode,
    visible: bool,
    paused: bool,
    bypass_master_fx: bool,
}

impl From<&ExportLayer> for ExportFrameLayerBase {
    fn from(layer: &ExportLayer) -> Self {
        Self {
            effects: layer.effects,
            opacity: layer.opacity,
            speed: layer.speed,
            fps: layer.fps,
            blend_mode: layer.blend_mode,
            visible: layer.visible,
            paused: layer.paused,
            bypass_master_fx: layer.bypass_master_fx,
        }
    }
}

fn configured_blend_mode(value: &str) -> BlendMode {
    match value {
        "screen" => BlendMode::Screen,
        "multiply" => BlendMode::Multiply,
        "difference" => BlendMode::Difference,
        _ => BlendMode::Normal,
    }
}

/// Retain an unavailable source as a one-pixel opaque-black texture. Keeping
/// the layer visible preserves the saved compositing stack (including Normal
/// or Multiply darkening) instead of silently changing the patch by omission.
fn black_placeholder_layer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_index: usize,
    layer_cfg: &crate::patch::LayerConfig,
) -> ExportLayer {
    let width = 1;
    let height = 1;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Export Black Placeholder"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0, 0, 0, 255],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let mut effects = EffectUniforms {
        resolution: [width as f32, height as f32],
        ..Default::default()
    };
    layer_cfg.effects.apply_to_uniforms(&mut effects);
    ExportLayer {
        source_index,
        decoder: None,
        texture,
        texture_view,
        effects,
        opacity: layer_cfg.opacity,
        blend_mode: configured_blend_mode(&layer_cfg.blend_mode),
        bypass_master_fx: layer_cfg.bypass_master_fx,
        speed: layer_cfg.speed,
        visible: layer_cfg.visible,
        paused: true,
        fps: if layer_cfg.fps.is_finite() && layer_cfg.fps > 0.0 {
            layer_cfg.fps.clamp(1.0, 240.0)
        } else {
            30.0
        },
        width,
        height,
    }
}

/// Upload an immutable source exactly once. With no decoder handle, the frame
/// loop can never advance or reopen this source; time-varying effects still
/// evaluate normally against these identical input pixels on every frame.
fn still_export_layer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_index: usize,
    layer_cfg: &crate::patch::LayerConfig,
    decoded: DecodedStillImage,
) -> ExportLayer {
    let width = decoded.width;
    let height = decoded.height;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Export Still Layer"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &decoded.rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let mut effects = EffectUniforms {
        resolution: [width as f32, height as f32],
        ..Default::default()
    };
    layer_cfg.effects.apply_to_uniforms(&mut effects);
    ExportLayer {
        source_index,
        decoder: None,
        texture,
        texture_view,
        effects,
        opacity: layer_cfg.opacity,
        blend_mode: configured_blend_mode(&layer_cfg.blend_mode),
        bypass_master_fx: layer_cfg.bypass_master_fx,
        speed: layer_cfg.speed,
        visible: layer_cfg.visible,
        paused: layer_cfg.paused,
        fps: if layer_cfg.fps.is_finite() && layer_cfg.fps > 0.0 {
            layer_cfg.fps.clamp(1.0, 240.0)
        } else {
            30.0
        },
        width,
        height,
    }
}

fn media_has_audio_stream(path: &str) -> Result<bool, String> {
    ffmpeg_next::init().map_err(|error| format!("failed to initialize media probing: {error}"))?;
    let input = ffmpeg_next::format::input(path)
        .map_err(|error| format!("failed to open selected media: {error}"))?;
    Ok(input
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .is_some())
}

fn validate_export_dimensions(width: u32, height: u32) -> Result<u64, String> {
    if width > MAX_EXPORT_EDGE || height > MAX_EXPORT_EDGE {
        return Err(format!(
            "export dimensions {width}x{height} exceed the {MAX_EXPORT_EDGE}px safety edge limit"
        ));
    }
    validate_media_dimensions(width, height, None)
        .map(|rgba_bytes| rgba_bytes / 4)
        .map_err(|error| {
            format!("invalid export dimensions: {error}; maximum pixels {MAX_MEDIA_PIXELS}")
        })
}

/// Construct the complete ffmpeg argv separately so duration and audio
/// policy can be verified without a GPU or subprocess. Raw video always maps
/// explicitly from input zero. Optional audio starts at zero at 1x, then
/// `apad` + `atrim` yields exactly the requested program duration.
fn build_ffmpeg_args(config: &ExportConfig, audio_path: Option<&str>) -> Vec<String> {
    let size = format!("{}x{}", config.width, config.height);
    let fps = config.fps.to_string();
    let duration = format!("{:.6}", config.duration_secs);
    let mut args = [
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    args.push(size);
    args.extend([
        "-framerate".to_owned(),
        fps,
        "-i".to_owned(),
        "pipe:0".to_owned(),
    ]);
    if let Some(path) = audio_path {
        args.extend(["-i".to_owned(), path.to_owned()]);
    }

    args.extend(
        [
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    if audio_path.is_some() {
        args.extend([
            "-map".to_owned(),
            "1:a:0".to_owned(),
            "-filter:a".to_owned(),
            format!("asetpts=PTS-STARTPTS,apad,atrim=end={duration}"),
            "-c:a".to_owned(),
            "aac".to_owned(),
            "-b:a".to_owned(),
            "192k".to_owned(),
        ]);
    } else {
        args.push("-an".to_owned());
    }
    args.extend([
        "-t".to_owned(),
        duration,
        "-movflags".to_owned(),
        "+faststart".to_owned(),
        config.output_path.clone(),
    ]);
    args
}

fn export_frame_count(fps: u32, duration_secs: f32) -> u64 {
    (fps as f64 * duration_secs as f64).ceil() as u64
}

/// Frame-indexed program transport. A patch saved under global Pause exports
/// its held program state for the complete requested duration; decoder frames,
/// shader time, modulation/audio, morph, temporal history, and NTSC therefore
/// share the same frozen timestamp.
fn export_program_transport(frame_num: u64, frame_interval: f32, paused: bool) -> (f32, f32) {
    if paused {
        (0.0, 0.0)
    } else {
        (
            frame_num as f32 * frame_interval,
            if frame_num == 0 { 0.0 } else { frame_interval },
        )
    }
}

/// Mirror live pause semantics for transient modulation state. Routing caches
/// are runtime state rather than patch data, so a recalled paused patch holds
/// their deterministic reconstructed value (zero) instead of sampling an LFO,
/// pad, or imported-audio source only in the offline renderer.
fn update_export_modulation(
    matrix: &mut crate::modulation::ModMatrix,
    beat: f64,
    delta_seconds: f32,
    paused: bool,
) {
    if paused {
        matrix.reset_update_timing();
    } else {
        matrix.update_at_beat(beat, delta_seconds);
    }
}

/// Live Pause holds the materialized patch bases. Re-sampling an active morph
/// only in export would produce a different held frame, especially when the
/// morph target has transient routing state.
fn export_morph_sample(
    morph: Option<&crate::morph::Morph>,
    beat: f64,
    offset: f32,
    paused: bool,
) -> Option<crate::morph::MorphSample> {
    if paused {
        return None;
    }
    let morph = morph.filter(|morph| morph.active())?;
    morph.sample((morph.position_at_beat(beat) + offset).clamp(0.0, 1.0))
}

/// The export stack has no live layer IDs, but patch source positions are
/// stable for the lifetime of one job. Offset by one so every planned ID is
/// nonzero and can be mapped back without a sentinel collision.
fn export_selective_layer_id(source_index: usize) -> u64 {
    source_index as u64 + 1
}

/// Reproduce the live planner input from frame-local, post-morph/post-Mod
/// values. `layers` remains in UI order (top to bottom); the shared planner is
/// the sole authority that reverses contributing entries into compositor
/// order and forces only the bottom contribution to Normal.
#[allow(clippy::too_many_arguments)]
fn plan_export_selective_ntsc(
    frame_num: u64,
    width: u32,
    height: u32,
    fps: u32,
    paused: bool,
    params: &crate::ntsc::NtscParams,
    layers: &[ExportLayer],
    layer_mods: &[(EffectUniforms, f32)],
    master: &EffectUniforms,
) -> Option<SelectiveNtscPlan> {
    if layers.len() != layer_mods.len() {
        return None;
    }
    plan_selective_ntsc(
        SelectiveNtscGeneration {
            visual_epoch: 1,
            topology_generation: 1,
            width,
            height,
            sample_sequence: frame_num,
        },
        NtscFrameMetadata {
            params: params.clone(),
            reference_frame: export_ntsc_reference_frame(frame_num, fps, paused),
        },
        layers
            .iter()
            .zip(layer_mods)
            .map(|(layer, (effects, opacity))| SelectiveNtscLayerDescriptor {
                layer_id: export_selective_layer_id(layer.source_index),
                visible: layer.visible,
                bypass_master_fx: layer.bypass_master_fx,
                opacity: *opacity,
                blend_mode: layer.blend_mode.as_u32(),
                // Export consumes every generated batch synchronously, so no
                // stale-control comparison is needed. Keep a deterministic
                // value in the shared plan nevertheless.
                transform_fingerprint: export_selective_transform_fingerprint(
                    effects,
                    (!layer.bypass_master_fx).then_some(master),
                ),
            }),
    )
}

fn export_selective_transform_fingerprint(
    effects: &EffectUniforms,
    master: Option<&EffectUniforms>,
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytemuck::bytes_of(effects)
        .iter()
        .chain(master.into_iter().flat_map(bytemuck::bytes_of))
    {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn export_ntsc_reference_frame(frame_num: u64, fps: u32, paused: bool) -> usize {
    if paused {
        0
    } else {
        reference_frame_for_output(frame_num, fps)
    }
}

fn export_selective_topology_signature(plan: &SelectiveNtscPlan) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    };
    mix(plan.layers.len() as u64);
    for layer in &plan.layers {
        mix(layer.layer_id);
        mix(layer.bypass_master_fx as u64);
    }
    hash
}

fn run_export(
    patch: &PatchState,
    config: &ExportConfig,
    progress: &Arc<ExportProgress>,
    library_folder: &str,
    shared_gpu: Option<(wgpu::Device, wgpu::Queue)>,
) -> Result<(), String> {
    check_cancelled(progress)?;
    // Validate protocol-controlled sizes before creating a device or any of
    // the many full-frame GPU textures used by the compositor.
    validate_export_dimensions(config.width, config.height)?;
    if !config.width.is_multiple_of(2) || !config.height.is_multiple_of(2) {
        return Err("export dimensions must be even for yuv420p output".to_string());
    }
    if config.fps == 0 || config.fps > 240 {
        return Err("export FPS must be between 1 and 240".to_string());
    }
    if patch.layers.len() > crate::modulation::MAX_MOD_LAYERS {
        return Err(format!(
            "export patch has {} layers; maximum is {}",
            patch.layers.len(),
            crate::modulation::MAX_MOD_LAYERS
        ));
    }
    if !config.duration_secs.is_finite()
        || config.duration_secs <= 0.0
        || config.duration_secs > 3600.0
    {
        return Err("export duration must be greater than 0 and at most 3600 seconds".to_string());
    }
    // Render enough complete CFR frames to cover the requested duration, then
    // let ffmpeg trim the mux to that exact duration. Rounding could otherwise
    // leave the video up to half a frame short.
    let total_frames = export_frame_count(config.fps, config.duration_secs);
    if total_frames == 0 {
        return Err("export duration is shorter than one frame".to_string());
    }

    // Reuse the live renderer device when available. Standalone audit/tests
    // still create an isolated headless device.
    let (device, queue) = if let Some(gpu) = shared_gpu {
        gpu
    } else {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("No GPU adapter found: {e}"))?;
        check_cancelled(progress)?;

        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Export Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(|e| format!("Failed to create export device: {e}"))?
    };
    check_cancelled(progress)?;

    let w = config.width;
    let h = config.height;
    let max_dimension = device.limits().max_texture_dimension_2d;
    if w > max_dimension || h > max_dimension {
        return Err(format!(
            "export dimensions {w}x{h} exceed this GPU's {max_dimension}px texture limit"
        ));
    }

    // --- Build pipelines (same as renderer/state.rs) ---
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let effects_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Effects Texture BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

    let effects_uniform_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Effects Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Vertex"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fullscreen.wgsl").into()),
    });

    let effects_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Effects Fragment"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/effects.wgsl").into()),
    });

    let effects_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Export Effects PL"),
        bind_group_layouts: &[
            Some(&effects_bind_group_layout),
            Some(&effects_uniform_layout),
        ],
        immediate_size: 0,
    });

    let effects_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Export Effects Pipeline"),
        layout: Some(&effects_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vertex_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &effects_fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // Composite pipeline
    let composite_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Composite BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

    let composite_uniform_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export Composite Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let composite_fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Composite Fragment"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
    });

    let composite_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Export Composite PL"),
            bind_group_layouts: &[
                Some(&composite_bind_group_layout),
                Some(&composite_uniform_layout),
            ],
            immediate_size: 0,
        });

    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Export Composite Pipeline"),
        layout: Some(&composite_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vertex_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &composite_fragment,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // --- Composite textures (same 3-texture scheme as live renderer) ---
    let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::COPY_DST;

    let composite_textures: [wgpu::Texture; 3] = std::array::from_fn(|i| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Export Composite {i}")),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: tex_usage,
            view_formats: &[],
        })
    });

    let composite_views: [wgpu::TextureView; 3] = std::array::from_fn(|i| {
        composite_textures[i].create_view(&wgpu::TextureViewDescriptor::default())
    });
    let (opaque_output_pipeline, opaque_output_bind_group_layout) =
        crate::renderer::state::build_opaque_output_pipeline(&device);
    let opaque_output_bind_group = crate::renderer::state::build_opaque_output_bind_group(
        &device,
        &opaque_output_bind_group_layout,
        &composite_views[0],
        &sampler,
    );

    // --- Open video decoders for each layer ---
    let mut layers: Vec<ExportLayer> = Vec::new();
    for (source_index, layer_cfg) in patch.layers.iter().enumerate() {
        check_cancelled(progress)?;
        if crate::layers::spout_sender_from_source_path(&layer_cfg.source_path).is_some() {
            log::warn!(
                "Export: live Spout layer '{}' is unavailable offline; rendering deterministic black at source index {source_index}",
                layer_cfg.filename
            );
            layers.push(black_placeholder_layer(
                &device,
                &queue,
                source_index,
                layer_cfg,
            ));
            continue;
        }

        let persisted = std::path::PathBuf::from(&layer_cfg.source_path);
        let library_path = std::path::Path::new(library_folder).join(&layer_cfg.filename);
        let path = if !layer_cfg.source_path.is_empty() && persisted.is_file() {
            persisted
        } else {
            library_path
        };
        let path_text = path.to_string_lossy();
        if crate::layers::is_still_image_file(&path) {
            match crate::video::decode_still_image(&path, Some(max_dimension)) {
                Ok(decoded) => {
                    layers.push(still_export_layer(
                        &device,
                        &queue,
                        source_index,
                        layer_cfg,
                        decoded,
                    ));
                    continue;
                }
                Err(error) => {
                    log::warn!(
                        "Export: still layer '{}' could not be opened ({error}); rendering deterministic black at source index {source_index}",
                        layer_cfg.filename
                    );
                    layers.push(black_placeholder_layer(
                        &device,
                        &queue,
                        source_index,
                        layer_cfg,
                    ));
                    continue;
                }
            }
        }
        let mut decoder = match VideoDecoder::open_with_cancel(&path_text, progress.cancel.clone())
        {
            Ok(d) => d,
            Err(e) => {
                log::warn!(
                    "Export: layer '{}' could not be opened ({e}); rendering deterministic black at source index {source_index}",
                    layer_cfg.filename
                );
                layers.push(black_placeholder_layer(
                    &device,
                    &queue,
                    source_index,
                    layer_cfg,
                ));
                continue;
            }
        };

        let lw = decoder.width;
        let lh = decoder.height;
        if let Err(error) =
            crate::layers::validate_source_texture_dimensions(lw, lh, max_dimension, "export video")
        {
            log::warn!(
                "Export: layer '{}' cannot use dimensions {lw}x{lh} ({error}); rendering deterministic black at source index {source_index}",
                layer_cfg.filename,
            );
            layers.push(black_placeholder_layer(
                &device,
                &queue,
                source_index,
                layer_cfg,
            ));
            continue;
        }
        let fps = if layer_cfg.fps.is_finite() && layer_cfg.fps > 0.0 {
            layer_cfg.fps
        } else {
            decoder.fps
        }
        .clamp(1.0, 240.0);

        check_cancelled(progress)?;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Export Layer Tex"),
            size: wgpu::Extent3d {
                width: lw,
                height: lh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Seed every layer texture before frame zero. This makes paused layers
        // show a stable first frame and avoids reading uninitialized GPU memory
        // when the export rate is higher than the source rate.
        let first_frame = match decoder.next_frame_result() {
            Ok(frame) => frame,
            Err(error) => {
                log::warn!(
                    "Export: layer '{}' could not decode its first frame ({error}); rendering deterministic black at source index {source_index}",
                    layer_cfg.filename
                );
                layers.push(black_placeholder_layer(
                    &device,
                    &queue,
                    source_index,
                    layer_cfg,
                ));
                continue;
            }
        };
        check_cancelled(progress)?;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &first_frame,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * lw),
                rows_per_image: Some(lh),
            },
            wgpu::Extent3d {
                width: lw,
                height: lh,
                depth_or_array_layers: 1,
            },
        );

        let mut effects = EffectUniforms {
            resolution: [lw as f32, lh as f32],
            ..Default::default()
        };
        layer_cfg.effects.apply_to_uniforms(&mut effects);

        layers.push(ExportLayer {
            source_index,
            decoder: Some(decoder),
            texture,
            texture_view,
            effects,
            opacity: layer_cfg.opacity,
            blend_mode: configured_blend_mode(&layer_cfg.blend_mode),
            bypass_master_fx: layer_cfg.bypass_master_fx,
            speed: layer_cfg.speed,
            visible: layer_cfg.visible,
            paused: layer_cfg.paused,
            fps,
            width: lw,
            height: lh,
        });
    }

    // --- Master effects ---
    let mut master_effects = EffectUniforms {
        resolution: [w as f32, h as f32],
        ..Default::default()
    };
    patch.master.apply_to_uniforms(&mut master_effects);

    // --- NTSC state ---
    let mut ntsc_state = NtscState::new();
    if let Some(ref ntsc_cfg) = patch.ntsc {
        ntsc_state.params = ntsc_cfg.to_params();
    }
    let base_ntsc = ntsc_state.params.clone();
    // Live global and selective workers own independent ntsc-rs state. Keep
    // those processors distinct offline as well, especially when a morph
    // crosses the discrete bypass boundary during one export.
    let mut selective_ntsc_state = NtscState::new();

    // --- Modulation matrix (deterministic: beat derived from frame index) ---
    // Imported audio is sampled from frame-indexed program time. Live capture,
    // MIDI, and other hardware sources read 0 offline; LFO motion renders
    // identically for the same patch every time.
    let mut mod_matrix = crate::modulation::ModMatrix::new();
    if let Some(ref mod_cfg) = patch.modulation {
        mod_cfg.apply_to_matrix(&mut mod_matrix);
    }
    let analysis_clip = if mod_matrix.audio_enabled
        && mod_matrix.audio_source_kind == crate::modulation::AUDIO_SOURCE_FILE
    {
        let requested = std::path::PathBuf::from(&mod_matrix.audio_clip_path);
        let resolved = if requested.is_file() {
            requested
        } else {
            let basename = requested
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new(&mod_matrix.audio_clip_path));
            std::path::Path::new(library_folder).join(basename)
        };
        if !resolved.is_file() {
            return Err(format!(
                "deterministic audio-analysis clip not found: {}",
                mod_matrix.audio_clip_path
            ));
        }
        Some(crate::audio::AudioClip::open(&resolved).map_err(|error| {
            format!(
                "cannot load deterministic audio-analysis clip {}: {error}",
                resolved.display()
            )
        })?)
    } else {
        None
    };
    let export_morph = patch.morph.clone().map(crate::morph::Morph::from_snapshot);

    // --- Temporal effects (feedback/slit-scan), same pass as live ---
    let base_temporal = patch
        .temporal
        .as_ref()
        .map(|t| t.to_params())
        .unwrap_or_default();
    let (temporal_pipeline, temporal_bgl, temporal_ubl) =
        crate::renderer::state::build_temporal_pipeline(&device);
    let (history_texture, history_view) =
        crate::renderer::state::build_history_texture(&device, w, h);
    let (feedback_texture, feedback_view) =
        crate::renderer::state::build_feedback_texture(&device, w, h);
    let mut temporal_state = crate::renderer::state::TemporalState::default();
    let mut previous_selective_frame: Option<bool> = None;
    let mut previous_selective_topology: Option<u64> = None;

    // --- Readback staging buffer ---
    let bytes_per_row = (w * 4 + 255) & !255;
    let buffer_size = (bytes_per_row * h) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export Readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    // Selective VHS reads one straight-alpha slice per contributing layer.
    // Reuse a single staging allocation sequentially so peak host/GPU memory
    // stays one frame plus the final CPU batch rather than N padded frames.
    // This is allocated lazily on the first selective frame, preserving the
    // established resource profile for every legacy/all-inherited export.
    let mut selective_staging: Option<wgpu::Buffer> = None;

    // --- Spawn ffmpeg ---
    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(&config.output_path).parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create export directory {}: {error}",
                parent.display()
            )
        })?;
    }

    let audio_path = match config.audio_path.as_deref() {
        None => None,
        Some(path) => match media_has_audio_stream(path) {
            Ok(true) => Some(path),
            Ok(false) => {
                return Err(format!(
                    "selected export audio source '{path}' contains no audio stream"
                ));
            }
            Err(error) => {
                return Err(format!(
                    "failed to inspect selected export audio source '{path}': {error}"
                ));
            }
        },
    };
    check_cancelled(progress)?;
    let encoder = start_encoder_supervisor(
        "ffmpeg".to_string(),
        build_ffmpeg_args(config, audio_path),
        progress.clone(),
        config.output_path.clone(),
    )?;
    let EncoderSession {
        stdin: mut ffmpeg_stdin,
        completion: encoder_completion,
        supervisor: encoder_supervisor,
    } = encoder;

    // --- Frame loop ---
    let frame_interval = 1.0 / config.fps as f32;
    let mut write_error = None;

    // Track per-layer frame timing (accumulator-based)
    let mut layer_accumulators: Vec<f32> = vec![0.0; layers.len()];

    for frame_num in 0..total_frames {
        if progress.cancel.load(Ordering::Acquire) {
            break;
        }

        // Update time uniform for effects (breathe, grain seed, etc.)
        let (time, program_dt) =
            export_program_transport(frame_num, frame_interval, patch.master_paused);
        master_effects.time = time;

        // Sample the modulation matrix at this frame's beat position and
        // derive the modulated params for this frame (bases untouched).
        let beat = time as f64 * (mod_matrix.clock.bpm as f64 / 60.0);
        if let Some(clip) = &analysis_clip {
            mod_matrix.audio = clip
                .analyze_at_time(
                    time as f64,
                    mod_matrix.audio_gain,
                    mod_matrix.audio_band_config,
                )
                .levels;
        } else {
            // Live capture and hardware sources are deliberately unavailable
            // offline; their zero value makes exports repeatable.
            mod_matrix.audio = crate::audio::AudioLevels::default();
        }
        update_export_modulation(&mut mod_matrix, beat, program_dt, patch.master_paused);
        let modulation_frame = mod_matrix.frame();
        let mut frame_master = master_effects;
        let mut frame_ntsc = base_ntsc.clone();
        let mut frame_temporal = base_temporal;
        // Keep render parameters detached from decoder/runtime handles. A
        // full morph sample can then drive exactly the same layer world as
        // live rendering without mutating the saved patch bases.
        let mut frame_layer_bases: Vec<ExportFrameLayerBase> =
            layers.iter().map(ExportFrameLayerBase::from).collect();

        // Live Pause holds the already-materialized patch bases and does not
        // re-apply a morph. Mirror that exact held-state contract offline.
        if let Some(sample) = export_morph_sample(
            export_morph.as_ref(),
            beat,
            modulation_frame.morph_offset(),
            patch.master_paused,
        ) {
            sample.master.apply_to(&mut frame_master);
            frame_ntsc = sample.ntsc.to_params();
            frame_temporal = sample.temporal.to_params();
            for sampled in sample.layers {
                if let Some((position, _)) = layers
                    .iter()
                    .enumerate()
                    .find(|(_, layer)| layer.source_index == sampled.position)
                {
                    frame_layer_bases[position].opacity = sampled.opacity;
                    frame_layer_bases[position].speed = sampled.speed;
                    if let Some(fps) = sampled.fps {
                        frame_layer_bases[position].fps = fps;
                    }
                    if let Some(effects) = sampled.effects {
                        effects.apply_to(&mut frame_layer_bases[position].effects);
                    }
                    if let Some(key_threshold) = sampled.key_threshold {
                        frame_layer_bases[position].effects.key_threshold = key_threshold;
                    }
                    if let Some(blend_mode) = sampled.blend_mode {
                        frame_layer_bases[position].blend_mode = blend_mode.to_blend_mode();
                    }
                    if let Some(visible) = sampled.visible {
                        frame_layer_bases[position].visible = visible;
                    }
                    if let Some(paused) = sampled.paused {
                        frame_layer_bases[position].paused = paused;
                    }
                    if let Some(bypass_master_fx) = sampled.bypass_master_fx {
                        frame_layer_bases[position].bypass_master_fx = bypass_master_fx;
                    }
                }
            }
        }
        frame_master.time = time;
        let (mod_master, mod_ntsc, mod_temporal) =
            modulation_frame.modulate(&frame_master, &frame_ntsc, &frame_temporal);

        // Per-layer modulated values for this frame (bases untouched).
        let mut layer_mods: Vec<(EffectUniforms, f32)> = Vec::with_capacity(layers.len());
        let mut frame_speeds = Vec::with_capacity(layers.len());
        let mut frame_fps = Vec::with_capacity(layers.len());
        // Keep placeholder slots in the batch so positional layer targets stay
        // attached to their saved source identity even when media is missing.
        let frame_modulations = modulation_frame.modulate_layers(
            frame_layer_bases
                .iter()
                .map(|base| (&base.effects, base.opacity, base.speed, base.fps)),
        );
        for lm in frame_modulations {
            let mut fx = lm.effects;
            fx.time = time;
            frame_speeds.push(lm.speed);
            frame_fps.push(lm.fps);
            layer_mods.push((fx, lm.opacity));
        }
        for (layer, base) in layers.iter_mut().zip(frame_layer_bases.iter()) {
            layer.blend_mode = base.blend_mode;
            layer.visible = base.visible;
            layer.bypass_master_fx = base.bypass_master_fx;
        }

        // --- GPU render ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Export Frame Encoder"),
        });

        let selective_plan = plan_export_selective_ntsc(
            frame_num,
            w,
            h,
            config.fps,
            patch.master_paused,
            &mod_ntsc,
            &layers,
            &layer_mods,
            &mod_master,
        );
        let selective_frame = selective_plan.is_some();
        let selective_topology = selective_plan
            .as_ref()
            .map(export_selective_topology_signature);
        let selective_edge =
            previous_selective_frame.is_some_and(|previous| previous != selective_frame);
        let selective_topology_changed = selective_frame
            && previous_selective_topology
                .is_some_and(|previous| Some(previous) != selective_topology);
        if selective_edge || selective_topology_changed {
            // Match live `reset_visual_generation`: no pre-switch feedback or
            // slit history may cross a selective-VHS topology/path edge.
            temporal_state = crate::renderer::state::TemporalState::default();
        }
        previous_selective_frame = Some(selective_frame);
        previous_selective_topology = selective_topology;
        if let Some(plan) = selective_plan {
            let selective_staging = selective_staging.get_or_insert_with(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Export Selective NTSC Readback"),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            });
            // Each slice is rendered through local FX and conditional direct
            // master FX in one coherent command stream. Composite/VHS stays
            // on the CPU, so no unprocessed intermediate reaches Temporal.
            let slices = match render_and_readback_selective_ntsc_slices_export(
                &device,
                &queue,
                &mut encoder,
                &layers,
                &layer_mods,
                &mod_master,
                &plan,
                &composite_textures,
                &composite_views,
                &effects_pipeline,
                &effects_bind_group_layout,
                &effects_uniform_layout,
                &sampler,
                selective_staging,
                w,
                h,
                bytes_per_row,
                &progress.cancel,
            ) {
                Ok(slices) => slices,
                Err(error) => {
                    write_error = Some(error);
                    break;
                }
            };
            if progress.cancel.load(Ordering::Acquire) {
                write_error = Some("export cancelled before selective NTSC processing".into());
                break;
            }
            let processed = match process_selective_ntsc_batch_with_state(
                &mut selective_ntsc_state,
                SelectiveNtscBatch { plan, slices },
            ) {
                Ok(processed) => processed,
                Err(error) => {
                    write_error = Some(format!("selective NTSC export failed: {error}"));
                    break;
                }
            };
            if progress.cancel.load(Ordering::Acquire) {
                write_error = Some("export cancelled during selective NTSC processing".into());
                break;
            }
            if let Err(error) = upload_engine_composite_export(
                &queue,
                &composite_textures[0],
                &processed.pixels,
                w,
                h,
            ) {
                write_error = Some(error);
                break;
            }
            // CPU processing and queue upload are complete before Temporal is
            // encoded. A fresh command stream prevents the earlier layer
            // passes from overwriting the returned straight-alpha composite.
            encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Export Selective Post-NTSC Encoder"),
            });
        } else {
            // Byte-for-byte legacy path: VHS-off and all-inherited stacks keep
            // the established direct render -> Temporal -> opaque -> global
            // NTSC order without any selective slice allocation or composite.
            render_layers_and_master_export(
                &device,
                &queue,
                &mut encoder,
                &layers,
                &layer_mods,
                &mod_master,
                &composite_textures,
                &composite_views,
                &effects_pipeline,
                &effects_bind_group_layout,
                &effects_uniform_layout,
                &composite_pipeline,
                &composite_bind_group_layout,
                &composite_uniform_layout,
                &sampler,
                w,
                h,
            );
        }

        // Temporal effects + history recording (identical pass to live)
        crate::renderer::state::encode_temporal_with_dt(
            &device,
            &queue,
            &mut encoder,
            &mod_temporal,
            &temporal_pipeline,
            &temporal_bgl,
            &temporal_ubl,
            &sampler,
            &composite_textures,
            &composite_views,
            &history_texture,
            &history_view,
            &feedback_texture,
            &feedback_view,
            &mut temporal_state,
            program_dt,
            !patch.master_paused,
            w,
            h,
        );

        // Export consumes the same opaque SDR program image as live preview,
        // projector, Spout, and NTSC. Keep key alpha inside the engine and
        // flatten it over black exactly once at this boundary.
        crate::renderer::state::encode_opaque_output(
            &mut encoder,
            &opaque_output_pipeline,
            &opaque_output_bind_group,
            &composite_views[2],
        );

        // Submit GPU work
        queue.submit(std::iter::once(encoder.finish()));

        // --- NTSC using the same half-resolution path as live output ---
        let mut pixels = match readback_pixels(
            &device,
            &queue,
            &composite_textures[2],
            &staging,
            w,
            h,
            bytes_per_row,
            &progress.cancel,
        ) {
            Ok(pixels) => pixels,
            Err(error) => {
                write_error = Some(error);
                break;
            }
        };
        // Selective mode has already applied VHS to inherited slices before
        // Temporal. Every other frame retains the established global
        // post-composite VHS call exactly once.
        if !selective_frame {
            ntsc_state.params = mod_ntsc;
            ntsc_state.apply_at_reference_frame(
                &mut pixels,
                w,
                h,
                export_ntsc_reference_frame(frame_num, config.fps, patch.master_paused),
            );
        }

        // Write to ffmpeg
        if let Err(error) = ffmpeg_stdin.write_all(&pixels) {
            write_error = Some(format!("failed to write video frame to ffmpeg: {error}"));
            break;
        }

        // Advance after rendering so frame zero always contains each source's
        // first decoded frame. The accumulator preserves source cadence when
        // source and export frame rates differ.
        if frame_num + 1 < total_frames {
            for (i, layer) in layers.iter_mut().enumerate() {
                let base = frame_layer_bases[i];
                let fps = frame_fps[i];
                let paused = base.paused;
                if patch.master_paused || paused {
                    // Live transport resets fractional debt while paused, so
                    // a morph-driven pause/unpause must resume from a fresh
                    // cadence boundary offline as well.
                    layer_accumulators[i] = 0.0;
                    continue;
                }
                let speed = frame_speeds[i];
                layer_accumulators[i] += frame_interval * speed;
                let layer_interval = 1.0 / fps;
                while layer_accumulators[i] >= layer_interval {
                    if progress.cancel.load(Ordering::Acquire) {
                        write_error = Some("export cancelled during source decode".to_string());
                        break;
                    }
                    layer_accumulators[i] -= layer_interval;
                    let Some(decoder) = layer.decoder.as_mut() else {
                        continue;
                    };
                    match decoder.next_frame_result() {
                        Ok(rgba) => queue.write_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: &layer.texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            &rgba,
                            wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(4 * layer.width),
                                rows_per_image: Some(layer.height),
                            },
                            wgpu::Extent3d {
                                width: layer.width,
                                height: layer.height,
                                depth_or_array_layers: 1,
                            },
                        ),
                        Err(error) => {
                            write_error = Some(format!(
                                "layer {} decode failed during export: {error}",
                                layer.source_index + 1
                            ));
                            break;
                        }
                    }
                }
                if write_error.is_some() {
                    break;
                }
            }
        }

        if write_error.is_some() {
            break;
        }

        // Update progress
        progress.progress.store(
            ((frame_num + 1) * 10000 / total_frames) as u32,
            Ordering::Relaxed,
        );
    }

    // Tell the supervisor this is an intentional end-of-input before closing
    // stdin. A panic/unwind that drops stdin without this flag is a failure.
    progress
        .encoder_finish_requested
        .store(true, Ordering::Release);
    drop(ffmpeg_stdin);
    let completion = encoder_completion
        .recv()
        .map_err(|_| "encoder supervisor ended without a completion result".to_string())?;
    log::debug!("export worker received encoder completion");
    encoder_supervisor
        .join()
        .map_err(|_| "encoder supervisor panicked".to_string())?;

    // Cancellation owns the terminal state even if ffmpeg happened to fail
    // while its stdin was being closed. Cleanup is complete before observers
    // see `cancelled = true`.
    if progress.cancel.load(Ordering::Acquire) {
        publish_cancelled_terminal(progress, Some(&config.output_path));
        drop(layers);
        drop(staging);
        drop(selective_staging);
        drop(history_view);
        drop(history_texture);
        drop(feedback_view);
        drop(feedback_texture);
        drop(opaque_output_bind_group);
        drop(opaque_output_bind_group_layout);
        drop(opaque_output_pipeline);
        drop(composite_views);
        drop(composite_textures);
        drop(temporal_pipeline);
        drop(temporal_bgl);
        drop(temporal_ubl);
        drop(composite_pipeline);
        drop(composite_pipeline_layout);
        drop(composite_fragment);
        drop(composite_uniform_layout);
        drop(composite_bind_group_layout);
        drop(effects_pipeline);
        drop(effects_pipeline_layout);
        drop(effects_fragment);
        drop(vertex_shader);
        drop(effects_uniform_layout);
        drop(effects_bind_group_layout);
        drop(sampler);
        drop(queue);
        drop(device);
        drop(ntsc_state);
        drop(selective_ntsc_state);
        drop(mod_matrix);
        drop(export_morph);
        drop(layer_accumulators);
        return Err("export cancelled".to_string());
    }

    let status = completion.status.inspect_err(|_| {
        let _ = std::fs::remove_file(&config.output_path);
    })?;
    let stderr_bytes = completion.stderr.inspect_err(|_| {
        let _ = std::fs::remove_file(&config.output_path);
    })?;
    let stderr = String::from_utf8_lossy(&stderr_bytes);
    if let Some(error) = write_error {
        let _ = std::fs::remove_file(&config.output_path);
        let detail = stderr.trim();
        return Err(if detail.is_empty() {
            error
        } else {
            format!("{error}: {}", detail.chars().take(300).collect::<String>())
        });
    }
    if !status.success() {
        let _ = std::fs::remove_file(&config.output_path);
        return Err(format!(
            "ffmpeg failed: {}",
            stderr.chars().take(300).collect::<String>()
        ));
    }

    Ok(())
}

/// Readback composite_textures[0] to CPU as RGBA bytes (no row padding).
// These arguments are the complete inputs to one GPU readback operation.
#[allow(clippy::too_many_arguments)]
fn readback_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    staging: &wgpu::Buffer,
    w: u32,
    h: u32,
    bytes_per_row: u32,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, String> {
    if cancel.load(Ordering::Acquire) {
        staging.destroy();
        return Err("export cancelled before GPU readback".to_string());
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Readback Encoder"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    loop {
        if cancel.load(Ordering::Acquire) {
            // `unmap` explicitly cancels a pending map_async operation (and
            // releases an already-completed mapping). Destroying afterward
            // prevents device/resource teardown from waiting on this staging
            // allocation after run_export has published cancellation.
            staging.unmap();
            staging.destroy();
            return Err("export cancelled during GPU readback".to_string());
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(100)),
        });
        match rx.try_recv() {
            Ok(result) => {
                if let Err(error) = result {
                    staging.unmap();
                    staging.destroy();
                    return Err(format!("GPU readback map failed: {error}"));
                }
                break;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => continue,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                staging.unmap();
                staging.destroy();
                return Err("GPU readback callback disconnected".to_string());
            }
        }
    }

    let data = slice.get_mapped_range();
    let row_bytes = (w * 4) as usize;
    let padded_row = bytes_per_row as usize;
    let mut pixels = Vec::with_capacity(row_bytes * h as usize);
    for row in 0..h as usize {
        let start = row * padded_row;
        pixels.extend_from_slice(&data[start..start + row_bytes]);
    }
    drop(data);
    staging.unmap();

    Ok(pixels)
}

fn upload_engine_composite_export(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<(), String> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "selective NTSC export dimensions overflow".to_string())?;
    if pixels.len() != expected {
        return Err(format!(
            "selective NTSC export composite has {} bytes; expected {expected}",
            pixels.len()
        ));
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

/// Render all planned straight-alpha slices and synchronously collect them in
/// the shared plan's bottom-to-top order. Export is intentionally synchronous:
/// it must write each exact requested frame to ffmpeg, unlike live preview's
/// bounded delayed worker.
#[allow(clippy::too_many_arguments)]
fn render_and_readback_selective_ntsc_slices_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    layer_mods: &[(EffectUniforms, f32)],
    master_uniforms: &EffectUniforms,
    plan: &SelectiveNtscPlan,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    staging: &wgpu::Buffer,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    cancel: &AtomicBool,
) -> Result<Vec<Vec<u8>>, String> {
    if layers.len() != layer_mods.len() {
        return Err("selective NTSC export layer/modulation alignment mismatch".into());
    }
    if plan.generation.width != width || plan.generation.height != height {
        return Err("selective NTSC export plan dimensions changed before rendering".into());
    }

    let master_buffer = create_uploaded_uniform(
        device,
        queue,
        "Export Selective NTSC Master FX Uniforms",
        master_uniforms,
    );
    let master_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Export Selective NTSC Master FX Input"),
        layout: effects_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&composite_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    let master_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Export Selective NTSC Master FX Uniforms BG"),
        layout: effects_uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: master_buffer.as_entire_binding(),
        }],
    });

    let mut slices = Vec::with_capacity(plan.layers.len());
    for planned_layer in &plan.layers {
        if cancel.load(Ordering::Acquire) {
            return Err("export cancelled before selective NTSC slice".to_string());
        }
        let source_index = layers
            .iter()
            .position(|layer| {
                export_selective_layer_id(layer.source_index) == planned_layer.layer_id
            })
            .ok_or_else(|| {
                format!(
                    "selective NTSC export layer {} disappeared before rendering",
                    planned_layer.layer_id
                )
            })?;
        let layer = &layers[source_index];
        if !layer.visible || layer.bypass_master_fx != planned_layer.bypass_master_fx {
            return Err(format!(
                "selective NTSC export layer {} changed before rendering",
                planned_layer.layer_id
            ));
        }

        let frame_fx = layer_mods[source_index].0.for_render_target(width, height);
        let fx_buffer = create_uploaded_uniform(
            device,
            queue,
            "Export Selective NTSC Layer FX Uniforms",
            &frame_fx,
        );
        let layer_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Selective NTSC Layer FX Input"),
            layout: effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let layer_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Selective NTSC Layer FX Uniforms BG"),
            layout: effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fx_buffer.as_entire_binding(),
            }],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Selective NTSC Layer FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[1],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &layer_tex_bg, &[]);
            pass.set_bind_group(1, &layer_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        let output_slot = if planned_layer.bypass_master_fx {
            1
        } else {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Selective NTSC Direct Master FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[2],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &master_tex_bg, &[]);
            pass.set_bind_group(1, &master_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
            2
        };

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[output_slot],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // One coherent slice must complete before the shared staging buffer is
        // reused. This is bounded and deterministic for the offline path.
        let finished = std::mem::replace(
            encoder,
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Export Selective NTSC Slice Encoder"),
            }),
        );
        queue.submit(std::iter::once(finished.finish()));
        slices.push(map_export_readback(
            device,
            staging,
            width,
            height,
            bytes_per_row,
            cancel,
            "selective NTSC slice",
        )?);
    }
    Ok(slices)
}

fn map_export_readback(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    cancel: &AtomicBool,
    label: &str,
) -> Result<Vec<u8>, String> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    loop {
        if cancel.load(Ordering::Acquire) {
            staging.unmap();
            return Err(format!("export cancelled during GPU {label} readback"));
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_millis(100)),
        });
        match rx.try_recv() {
            Ok(Ok(())) => break,
            Ok(Err(error)) => {
                staging.unmap();
                return Err(format!("GPU {label} readback map failed: {error}"));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => continue,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                staging.unmap();
                return Err(format!("GPU {label} readback callback disconnected"));
            }
        }
    }
    let data = slice.get_mapped_range();
    let row_bytes = (width * 4) as usize;
    let mut pixels = Vec::with_capacity(row_bytes * height as usize);
    for row in 0..height as usize {
        let start = row * bytes_per_row as usize;
        pixels.extend_from_slice(&data[start..start + row_bytes]);
    }
    drop(data);
    staging.unmap();
    Ok(pixels)
}

/// Mirror [`crate::renderer::state::Renderer::render_layers_and_master`].
/// The legacy branch deliberately delegates to the two pre-existing export
/// passes unchanged; only a visible bypass enters the conditional path.
#[allow(clippy::too_many_arguments)]
fn render_layers_and_master_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    layer_mods: &[(EffectUniforms, f32)],
    master_uniforms: &EffectUniforms,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) {
    let path = master_fx_composition_path(
        layers
            .iter()
            .zip(layer_mods.iter())
            .map(|(layer, (_, opacity))| (layer.visible, layer.bypass_master_fx, *opacity)),
    );
    match path {
        MasterFxCompositionPath::LegacyPostComposite => {
            render_layers_export(
                device,
                queue,
                encoder,
                layers,
                layer_mods,
                composite_textures,
                composite_views,
                effects_pipeline,
                effects_bind_group_layout,
                effects_uniform_layout,
                composite_pipeline,
                composite_bind_group_layout,
                composite_uniform_layout,
                sampler,
                output_width,
                output_height,
            );
            render_master_effects_export(
                device,
                queue,
                encoder,
                master_uniforms,
                composite_textures,
                composite_views,
                effects_pipeline,
                effects_bind_group_layout,
                effects_uniform_layout,
                sampler,
                output_width,
                output_height,
            );
        }
        MasterFxCompositionPath::ConditionalPerLayer => {
            render_layers_with_conditional_master_export(
                device,
                queue,
                encoder,
                layers,
                layer_mods,
                master_uniforms,
                composite_textures,
                composite_views,
                effects_pipeline,
                effects_bind_group_layout,
                effects_uniform_layout,
                composite_pipeline,
                composite_bind_group_layout,
                composite_uniform_layout,
                sampler,
                output_width,
                output_height,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_layers_with_conditional_master_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    layer_mods: &[(EffectUniforms, f32)],
    master_uniforms: &EffectUniforms,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) {
    let visible_layers: Vec<(&ExportLayer, &(EffectUniforms, f32))> = visible_stack_indices(
        layers
            .iter()
            .zip(layer_mods.iter())
            .map(|(layer, _)| layer.visible),
    )
    .into_iter()
    .map(|index| (&layers[index], &layer_mods[index]))
    .collect();
    debug_assert!(visible_layers
        .iter()
        .any(|(layer, _)| layer.bypass_master_fx));

    let master_buffer = create_uploaded_uniform(
        device,
        queue,
        "Export Conditional Master FX Uniforms",
        master_uniforms,
    );
    let master_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Export Conditional Master FX Input"),
        layout: effects_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&composite_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    let master_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Export Conditional Master FX Uniforms BG"),
        layout: effects_uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: master_buffer.as_entire_binding(),
        }],
    });

    for (stack_index, (layer, (mod_fx, mod_opacity))) in visible_layers.iter().enumerate() {
        let frame_fx = mod_fx.for_render_target(output_width, output_height);
        let layer_buffer = create_uploaded_uniform(
            device,
            queue,
            "Export Conditional Layer FX Uniforms",
            &frame_fx,
        );
        let layer_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Conditional Layer FX Input"),
            layout: effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let layer_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Conditional Layer FX Uniforms BG"),
            layout: effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: layer_buffer.as_entire_binding(),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Conditional Layer FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[1],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &layer_tex_bg, &[]);
            pass.set_bind_group(1, &layer_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        let slots = conditional_layer_slots(layer.bypass_master_fx);
        if let Some(master_output) = slots.master_output {
            debug_assert_eq!(master_output, 2);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Conditional Master FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[master_output],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &master_tex_bg, &[]);
            pass.set_bind_group(1, &master_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if stack_index == 0 {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Conditional Clear Base"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[0],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }

        let overlay_slot = slots.master_output.unwrap_or(1);
        let comp_uniforms = CompositeUniforms {
            opacity: *mod_opacity,
            blend_mode: if stack_index == 0 {
                0
            } else {
                layer.blend_mode.as_u32()
            },
            _pad: [0.0; 2],
        };
        let comp_buffer = create_uploaded_uniform(
            device,
            queue,
            "Export Conditional Composite Uniforms",
            &comp_uniforms,
        );
        let composite_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Conditional Composite Textures BG"),
            layout: composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&composite_views[overlay_slot]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let composite_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Export Conditional Composite Uniform BG"),
            layout: composite_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: comp_buffer.as_entire_binding(),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Conditional Composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[slots.composite_output],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(composite_pipeline);
            pass.set_bind_group(0, &composite_tex_bg, &[]);
            pass.set_bind_group(1, &composite_uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[slots.composite_output],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &composite_textures[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: output_width,
                height: output_height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Render all visible layers composited (mirrors Renderer::render_layers).
// Keep the GPU resources explicit so their render-pass lifetimes remain clear.
#[allow(clippy::too_many_arguments)]
fn render_layers_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    layers: &[ExportLayer],
    layer_mods: &[(EffectUniforms, f32)],
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    composite_pipeline: &wgpu::RenderPipeline,
    composite_bind_group_layout: &wgpu::BindGroupLayout,
    composite_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) {
    let visible_layers: Vec<(&ExportLayer, &(EffectUniforms, f32))> = layers
        .iter()
        .zip(layer_mods.iter())
        .filter(|(l, _)| l.visible)
        .rev()
        .collect();

    if visible_layers.is_empty() {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Export Clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &composite_views[0],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        return;
    }

    for (i, (layer, (mod_fx, mod_opacity))) in visible_layers.iter().enumerate() {
        // Mirror the live renderer: this pass maps source UVs into an
        // output-sized composite, so spatial effects use the export target's
        // resolution and aspect rather than the decoded source's.
        let frame_fx = mod_fx.for_render_target(output_width, output_height);
        let fx_buffer = create_uploaded_uniform(device, queue, "Export Layer FX", &frame_fx);

        let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: effects_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&layer.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: effects_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fx_buffer.as_entire_binding(),
            }],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Layer FX"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[1],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(effects_pipeline);
            pass.set_bind_group(0, &tex_bg, &[]);
            pass.set_bind_group(1, &uniform_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Bottom layer: cleared base + Normal blend with real opacity
        // (mirrors Renderer::render_layers).
        if i == 0 {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Clear Base"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &composite_views[0],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(transparent_accumulation_clear()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }
        {
            let comp_uniforms = CompositeUniforms {
                opacity: *mod_opacity,
                blend_mode: if i == 0 { 0 } else { layer.blend_mode.as_u32() },
                _pad: [0.0; 2],
            };
            let comp_buffer =
                create_uploaded_uniform(device, queue, "Export Composite Uniforms", &comp_uniforms);

            let composite_tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: composite_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&composite_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&composite_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });

            let composite_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: composite_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: comp_buffer.as_entire_binding(),
                }],
            });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Export Composite"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &composite_views[2],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(composite_pipeline);
                pass.set_bind_group(0, &composite_tex_bg, &[]);
                pass.set_bind_group(1, &composite_uniform_bg, &[]);
                pass.draw(0..3, 0..1);
            }

            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[2],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &composite_textures[0],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: output_width,
                    height: output_height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

/// Apply master effects to composite_textures[0] (mirrors Renderer::render_master_effects).
// Keep the GPU resources explicit so their render-pass lifetimes remain clear.
#[allow(clippy::too_many_arguments)]
fn render_master_effects_export(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    master_uniforms: &EffectUniforms,
    composite_textures: &[wgpu::Texture; 3],
    composite_views: &[wgpu::TextureView; 3],
    effects_pipeline: &wgpu::RenderPipeline,
    effects_bind_group_layout: &wgpu::BindGroupLayout,
    effects_uniform_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    output_width: u32,
    output_height: u32,
) {
    let fx_buffer = create_uploaded_uniform(device, queue, "Export Master FX", master_uniforms);

    let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: effects_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&composite_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });

    let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: effects_uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: fx_buffer.as_entire_binding(),
        }],
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Export Master FX"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &composite_views[2],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(effects_pipeline);
        pass.set_bind_group(0, &tex_bg, &[]);
        pass.set_bind_group(1, &uniform_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &composite_textures[2],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &composite_textures[0],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: output_width,
            height: output_height,
            depth_or_array_layers: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(duration_secs: f32) -> ExportConfig {
        ExportConfig {
            width: 1920,
            height: 1080,
            fps: 24,
            duration_secs,
            output_path: "render.mp4".to_owned(),
            audio_path: None,
        }
    }

    #[test]
    fn export_layer_accumulation_starts_transparent() {
        let clear = transparent_accumulation_clear();
        assert_eq!([clear.r, clear.g, clear.b, clear.a], [0.0; 4]);
    }

    #[test]
    fn selective_export_uses_shared_order_opacity_and_reference_clock() {
        let params = crate::ntsc::NtscParams {
            enabled: true,
            ..Default::default()
        };
        let plan = plan_selective_ntsc(
            SelectiveNtscGeneration {
                visual_epoch: 1,
                topology_generation: 1,
                width: 2,
                height: 2,
                sample_sequence: 17,
            },
            NtscFrameMetadata {
                params,
                reference_frame: reference_frame_for_output(17, 60),
            },
            [
                SelectiveNtscLayerDescriptor {
                    layer_id: export_selective_layer_id(0),
                    visible: true,
                    bypass_master_fx: true,
                    opacity: 0.4,
                    blend_mode: BlendMode::Difference.as_u32(),
                    transform_fingerprint: 10,
                },
                SelectiveNtscLayerDescriptor {
                    layer_id: export_selective_layer_id(1),
                    visible: false,
                    bypass_master_fx: false,
                    opacity: 1.0,
                    blend_mode: BlendMode::Screen.as_u32(),
                    transform_fingerprint: 20,
                },
                SelectiveNtscLayerDescriptor {
                    layer_id: export_selective_layer_id(2),
                    visible: true,
                    bypass_master_fx: false,
                    opacity: 0.8,
                    blend_mode: BlendMode::Multiply.as_u32(),
                    transform_fingerprint: 30,
                },
            ],
        )
        .unwrap();

        assert_eq!(plan.generation.sample_sequence, 17);
        assert_eq!(plan.metadata.reference_frame, 8);
        assert_eq!(
            plan.layers
                .iter()
                .map(|layer| layer.layer_id)
                .collect::<Vec<_>>(),
            [3, 1]
        );
        assert_eq!(plan.layers[0].blend_mode, 0);
        assert_eq!(plan.layers[1].blend_mode, BlendMode::Difference.as_u32());
        assert_eq!(plan.layers[0].opacity, 0.8);
        assert_eq!(plan.layers[1].opacity, 0.4);
        assert!(!plan.layers[0].bypass_master_fx);
        assert!(plan.layers[1].bypass_master_fx);

        let signature = export_selective_topology_signature(&plan);
        let mut continuous_change = plan.clone();
        continuous_change.layers[0].opacity = 0.2;
        continuous_change.layers[0].transform_fingerprint ^= 1;
        assert_eq!(
            export_selective_topology_signature(&continuous_change),
            signature,
            "ordinary modulation must not erase Temporal history"
        );
        let mut bypass_change = plan.clone();
        bypass_change.layers[0].bypass_master_fx = true;
        assert_ne!(
            export_selective_topology_signature(&bypass_change),
            signature
        );
    }

    fn value_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.windows(2)
            .find(|window| window[0] == flag)
            .map(|window| window[1].as_str())
    }

    #[test]
    fn mux_audio_is_trimmed_or_silence_padded_to_requested_duration() {
        let config = config(1.25);
        let args = build_ffmpeg_args(&config, Some("source with audio.mp4"));

        assert_eq!(value_after(&args, "-t"), Some("1.250000"));
        assert_eq!(value_after(&args, "-map"), Some("0:v:0"));
        assert!(args.windows(2).any(|pair| pair == ["-map", "1:a:0"]));
        assert_eq!(
            value_after(&args, "-filter:a"),
            Some("asetpts=PTS-STARTPTS,apad,atrim=end=1.250000")
        );
        assert!(!args.iter().any(|arg| arg == "-shortest"));
        assert_eq!(args.last().map(String::as_str), Some("render.mp4"));
    }

    #[test]
    fn silent_export_maps_only_video() {
        let config = config(2.0);
        let args = build_ffmpeg_args(&config, None);

        assert!(args.iter().any(|arg| arg == "-an"));
        assert!(!args.iter().any(|arg| arg == "-filter:a"));
        assert!(!args.iter().any(|arg| arg == "1:a:0"));
        assert!(!args.iter().any(|arg| arg == "-shortest"));
    }

    #[test]
    fn frame_count_covers_fractional_requested_duration() {
        assert_eq!(export_frame_count(30, 1.0), 30);
        assert_eq!(export_frame_count(30, 1.001), 31);
        assert_eq!(export_frame_count(24, 1.25), 30);
    }

    #[test]
    fn master_pause_freezes_every_export_program_frame_at_zero() {
        for frame in [0, 1, 29, 300] {
            assert_eq!(
                export_program_transport(frame, 1.0 / 30.0, true),
                (0.0, 0.0)
            );
        }
        assert_eq!(export_program_transport(0, 1.0 / 30.0, false), (0.0, 0.0));
        let (time, dt) = export_program_transport(30, 1.0 / 30.0, false);
        assert!((time - 1.0).abs() < 1e-6);
        assert!((dt - 1.0 / 30.0).abs() < 1e-6);
        assert_eq!(export_ntsc_reference_frame(300, 60, true), 0);
        assert_eq!(export_ntsc_reference_frame(300, 60, false), 150);
    }

    #[test]
    fn master_pause_does_not_initialize_transient_modulation_only_offline() {
        use crate::modulation::{ModMatrix, ModSource, Routing};

        fn matrix_with_positive_lfo() -> ModMatrix {
            let mut matrix = ModMatrix::new();
            matrix.lfos[0].set_phase(0.25);
            matrix
                .routings
                .push(Routing::new(ModSource::Lfo(0), "brightness", 1.0));
            matrix
        }

        let mut paused = matrix_with_positive_lfo();
        update_export_modulation(&mut paused, 0.0, 0.0, true);
        assert_eq!(paused.routings[0].cached_value(), 0.0);

        let mut running = matrix_with_positive_lfo();
        update_export_modulation(&mut running, 0.0, 0.0, false);
        assert!(running.routings[0].cached_value() > 0.99);
    }

    #[test]
    fn master_pause_holds_materialized_bases_instead_of_resampling_morph() {
        let mut a = crate::morph::MorphSlot::default();
        a.master.brightness = -1.0;
        let mut b = crate::morph::MorphSlot::default();
        b.master.brightness = 1.0;
        let morph = crate::morph::Morph {
            a: Some(a),
            b: Some(b),
            t: 1.0,
            ..Default::default()
        };

        assert!(export_morph_sample(Some(&morph), 0.0, 0.0, true).is_none());
        let running = export_morph_sample(Some(&morph), 0.0, 0.0, false).unwrap();
        assert_eq!(running.master.brightness, 1.0);
    }

    #[test]
    fn export_dimensions_are_bounded_before_gpu_allocation() {
        assert!(validate_export_dimensions(0, 1080).is_err());
        assert!(validate_export_dimensions(1920, 0).is_err());
        assert!(validate_export_dimensions(3840, 2160).is_ok());
        assert!(validate_export_dimensions(4096, 2160).is_err());
        assert!(validate_export_dimensions(MAX_EXPORT_EDGE + 1, 2).is_err());
    }

    #[test]
    fn unreadable_explicit_audio_is_a_probe_error_not_no_stream() {
        let missing = std::env::temp_dir().join(format!(
            "collideoscope-missing-audio-probe-{}-{}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let error = media_has_audio_stream(&missing.to_string_lossy()).unwrap_err();
        assert!(error.contains("failed to open selected media"));
    }

    #[test]
    fn every_saved_blend_mode_has_a_placeholder_equivalent() {
        assert_eq!(configured_blend_mode("normal"), BlendMode::Normal);
        assert_eq!(configured_blend_mode("screen"), BlendMode::Screen);
        assert_eq!(configured_blend_mode("multiply"), BlendMode::Multiply);
        assert_eq!(configured_blend_mode("difference"), BlendMode::Difference);
    }

    #[test]
    fn cancellation_request_is_prompt_and_nonterminal_until_cleanup() {
        let progress = ExportProgress::new();
        progress.encoder_active.store(true, Ordering::Release);
        progress
            .encoder_cleanup_complete
            .store(false, Ordering::Release);
        let started = std::time::Instant::now();
        progress.request_cancel();
        assert!(started.elapsed() < Duration::from_millis(25));
        assert!(progress.cancel.load(Ordering::Acquire));
        assert!(!progress.done.load(Ordering::Acquire));
        assert_eq!(
            progress.outcome.load(Ordering::Acquire),
            OUTCOME_CANCEL_REQUESTED
        );
    }

    #[test]
    fn pre_spawn_cancel_has_no_late_child_or_partial_output() {
        let path = std::env::temp_dir().join(format!(
            "collideoscope-supervisor-prespawn-{}-{}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let progress = Arc::new(ExportProgress::new());
        progress.request_cancel();
        let result = start_encoder_supervisor(
            "definitely-not-a-real-encoder".to_string(),
            Vec::new(),
            progress.clone(),
            path.to_string_lossy().into_owned(),
        );
        assert!(result.is_err());
        assert!(progress.done.load(Ordering::Acquire));
        assert!(progress.cancelled.load(Ordering::Acquire));
        assert!(progress.encoder_cleanup_complete.load(Ordering::Acquire));
        assert!(!path.exists());
    }

    #[test]
    fn repeated_cancel_is_idempotent() {
        let progress = ExportProgress::new();
        progress.request_cancel();
        progress.request_cancel();
        assert_eq!(
            progress.outcome.load(Ordering::Acquire),
            OUTCOME_CANCEL_REQUESTED
        );
    }

    #[test]
    fn concurrent_cancel_and_failure_never_overwrite_an_accepted_cancel() {
        for _ in 0..100 {
            let progress = Arc::new(ExportProgress::new());
            let barrier = Arc::new(std::sync::Barrier::new(3));
            let cancel_progress = progress.clone();
            let cancel_barrier = barrier.clone();
            let cancel = std::thread::spawn(move || {
                cancel_barrier.wait();
                cancel_progress.request_cancel();
            });
            let failure_progress = progress.clone();
            let failure_barrier = barrier.clone();
            let failure = std::thread::spawn(move || {
                failure_barrier.wait();
                finalize_export_worker(
                    &failure_progress,
                    "",
                    Some("synthetic failure".to_string()),
                );
            });
            barrier.wait();
            cancel.join().unwrap();
            failure.join().unwrap();

            assert!(progress.done.load(Ordering::Acquire));
            if progress.cancel.load(Ordering::Acquire) {
                assert!(progress.cancelled.load(Ordering::Acquire));
                assert_eq!(progress.outcome.load(Ordering::Acquire), OUTCOME_CANCELLED);
            } else {
                assert_eq!(progress.outcome.load(Ordering::Acquire), OUTCOME_FAILED);
            }
        }
    }

    #[test]
    fn cleaned_terminal_job_is_replaceable_while_gpu_worker_unwinds() {
        let progress = Arc::new(ExportProgress::new());
        progress.done.store(true, Ordering::Release);
        progress
            .encoder_cleanup_complete
            .store(true, Ordering::Release);
        let (release, blocked) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _ = blocked.recv();
        });
        let mut job = ExportJob {
            progress,
            thread: Some(thread),
            output_path: String::new(),
        };
        assert!(job.can_replace());
        release.send(()).unwrap();
        job.thread.take().unwrap().join().unwrap();
    }

    #[test]
    fn terminal_cancel_follows_partial_removal() {
        let path = std::env::temp_dir().join(format!(
            "collideoscope-cancel-order-{}-{}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"partial").unwrap();
        let progress = ExportProgress::new();
        progress.output_started.store(true, Ordering::Release);
        progress.request_cancel();
        publish_cancelled_terminal(&progress, Some(&path.to_string_lossy()));
        assert!(!path.exists());
        assert!(progress.done.load(Ordering::Acquire));
        assert!(progress.cancelled.load(Ordering::Acquire));
    }

    #[cfg(windows)]
    #[test]
    fn supervisor_owns_child_until_reap_then_publishes_cancel() {
        let path = std::env::temp_dir().join(format!(
            "collideoscope-supervisor-cancel-{}-{}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"partial").unwrap();
        let progress = Arc::new(ExportProgress::new());
        let session = start_encoder_supervisor(
            "powershell.exe".to_string(),
            vec![
                "-NoLogo".to_string(),
                "-NoProfile".to_string(),
                "-NonInteractive".to_string(),
                "-Command".to_string(),
                "Start-Sleep -Seconds 30".to_string(),
            ],
            progress.clone(),
            path.to_string_lossy().into_owned(),
        )
        .unwrap();

        let requested = std::time::Instant::now();
        progress.request_cancel();
        assert!(requested.elapsed() < Duration::from_millis(25));
        drop(session.stdin);
        let completion = session
            .completion
            .recv_timeout(Duration::from_secs(2))
            .expect("supervisor did not report completion");
        session.supervisor.join().unwrap();

        assert!(completion.status.is_ok());
        assert!(progress.encoder_cleanup_complete.load(Ordering::Acquire));
        assert!(!progress.encoder_active.load(Ordering::Acquire));
        assert!(!path.exists());
        assert!(progress.done.load(Ordering::Acquire));
        assert!(progress.cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn drop_deadline_includes_the_cancel_request() {
        let progress = Arc::new(ExportProgress::new());
        let (release, blocked) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _ = blocked.recv();
        });
        let job = ExportJob {
            progress,
            thread: Some(thread),
            output_path: String::new(),
        };
        let started = std::time::Instant::now();
        drop(job);
        assert!(started.elapsed() < Duration::from_millis(1200));
        release.send(()).unwrap();
    }

    /// Exact regression for the former live hang: ffmpeg's slow encoder can
    /// back up the raw-video pipe at 720p60, so cancelling around 3% must
    /// break that write, reap the process, publish `cancelled`, and delete the
    /// partial MP4. Run explicitly on a GPU-equipped host with the audit clips.
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit*.mp4"]
    fn live_720p60_cancellation_is_bounded_and_clean() {
        use crate::patch::{EffectsConfig, LayerConfig, PatchState};

        for filename in ["audit.mp4", "audit_audio.mp4"] {
            assert!(
                std::path::Path::new("videos").join(filename).is_file(),
                "missing videos/{filename}"
            );
        }
        let layers = ["audit.mp4", "audit_audio.mp4"]
            .into_iter()
            .map(|filename| LayerConfig {
                filename: filename.to_string(),
                source_path: String::new(),
                opacity: 1.0,
                blend_mode: "normal".to_string(),
                bypass_master_fx: false,
                speed: 1.0,
                fps: 30.0,
                paused: false,
                visible: true,
                effects: EffectsConfig::default(),
            })
            .collect();
        let patch = PatchState {
            master: EffectsConfig::default(),
            master_paused: false,
            layers,
            ntsc: None,
            modulation: None,
            temporal: None,
            morph: None,
        };
        let output = std::env::temp_dir().join(format!(
            "collideoscope-live-cancel-{}-{}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = ExportConfig {
            width: 1280,
            height: 720,
            fps: 60,
            duration_secs: 30.0,
            output_path: output.to_string_lossy().into_owned(),
            audio_path: None,
        };
        let mut job = ExportJob::start(patch, config, "videos");

        let progress_deadline = std::time::Instant::now() + Duration::from_secs(60);
        while job.progress.progress.load(Ordering::Relaxed) < 300 && !job.is_done() {
            assert!(
                std::time::Instant::now() < progress_deadline,
                "export did not reach 3% within 60 seconds"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!job.is_done(), "export finished before cancellation point");

        let cancel_started = std::time::Instant::now();
        job.cancel();
        while !job.is_done() && cancel_started.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(10));
        }

        assert!(job.is_done(), "cancellation exceeded three seconds");
        assert!(job.progress.cancelled.load(Ordering::Relaxed));
        assert!(!output.exists(), "cancelled export left a partial MP4");
        assert!(job
            .progress
            .encoder_cleanup_complete
            .load(Ordering::Acquire));
        assert!(!job.progress.encoder_active.load(Ordering::Acquire));

        let worker_deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !job.thread.as_ref().unwrap().is_finished()
            && std::time::Instant::now() < worker_deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            job.thread.as_ref().unwrap().is_finished(),
            "export worker remained stuck while destroying cancelled GPU resources"
        );
        job.thread.take().unwrap().join().unwrap();
        eprintln!(
            "720p60 cancellation committed and worker exited in {:?}",
            cancel_started.elapsed()
        );
    }
}

#[cfg(test)]
mod effects_audit {
    use super::*;
    use crate::patch::{EffectsConfig, LayerConfig, NtscConfig, PatchState, TemporalConfig};

    fn base_patch() -> PatchState {
        PatchState {
            master: EffectsConfig::default(),
            master_paused: false,
            layers: vec![LayerConfig {
                filename: "audit.mp4".to_string(),
                source_path: String::new(),
                opacity: 1.0,
                blend_mode: "normal".to_string(),
                bypass_master_fx: false,
                speed: 1.0,
                fps: 30.0,
                paused: false,
                visible: true,
                effects: EffectsConfig::default(),
            }],
            ntsc: Some(NtscConfig::default()),
            modulation: None,
            temporal: Some(TemporalConfig::default()),
            morph: None,
        }
    }

    fn render(label: &str, patch: PatchState) {
        let config = ExportConfig {
            width: 320,
            height: 180,
            fps: 24,
            duration_secs: 1.0,
            output_path: format!("renders/audit_{label}.mp4"),
            audio_path: None,
        };
        let job = ExportJob::start(patch, config, "videos");
        while !job.is_done() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let err = job.progress.error.lock().unwrap().clone();
        assert!(err.is_empty(), "{label}: export failed: {err}");
    }

    /// Renders every effect through the real shader chain into labeled
    /// files under renders/, for objective pixel-level verification
    /// (ffprobe signalstats/entropy). Needs a GPU, ffmpeg on PATH, and
    /// videos/audit.mp4 — run explicitly:
    ///   cargo test --release effects_audit -- --ignored --nocapture
    #[test]
    #[ignore = "requires a GPU, ffmpeg on PATH, and videos/audit.mp4"]
    fn render_selective_vhs_bypass_pipeline() {
        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first"
        );
        let mut patch = base_patch();
        let mut bottom = patch.layers[0].clone();
        bottom.opacity = 0.75;
        bottom.blend_mode = "multiply".into();
        patch.layers.push(bottom);
        patch.layers[0].bypass_master_fx = true;
        patch.layers[0].opacity = 0.6;
        patch.layers[0].blend_mode = "difference".into();
        patch.master.hue_shift = 75.0;
        let ntsc = patch.ntsc.as_mut().unwrap();
        ntsc.enabled = true;
        ntsc.snow_intensity = 0.45;
        ntsc.edge_wave_enabled = true;
        ntsc.edge_wave_intensity = 4.0;
        render("selective_vhs_bypass", patch);
    }

    #[test]
    #[ignore = "renders the full effects audit matrix; run explicitly"]
    fn render_effects_matrix() {
        std::fs::create_dir_all("renders").ok();
        assert!(
            std::path::Path::new("videos/audit.mp4").is_file(),
            "create videos/audit.mp4 first (any short clip)"
        );

        render("baseline", base_patch());

        let mut p = base_patch();
        p.master.brightness = 0.4;
        render("brightness", p);

        let mut p = base_patch();
        p.master.invert = true;
        render("invert", p);

        let mut p = base_patch();
        p.master.posterize = 2.0;
        render("posterize", p);

        let mut p = base_patch();
        p.master.pixelate = 32.0;
        render("pixelate", p);

        let mut p = base_patch();
        p.master.vignette = 1.4;
        render("vignette", p);

        let mut p = base_patch();
        p.master.hue_shift = 120.0;
        render("hue", p);

        let mut p = base_patch();
        p.master.contrast = 0.8;
        render("contrast", p);

        let mut p = base_patch();
        p.master.saturation = -1.0;
        render("saturation", p);

        let mut p = base_patch();
        p.master.grain_intensity = 0.28;
        p.master.color_grain = true;
        render("grain", p);

        let mut p = base_patch();
        p.master.rgb_split = 25.0;
        render("rgbsplit", p);

        let mut p = base_patch();
        p.master.color_drift = 0.02;
        p.master.breathe_scale = 0.05;
        render("drift_breathe", p);

        let mut p = base_patch();
        p.master.cellular_amount = 0.85;
        p.master.cellular_scale = 12.0;
        p.master.cellular_warp = 0.6;
        p.master.cellular_speed = 0.75;
        render("cellular", p);

        let mut p = base_patch();
        p.layers[0].effects.key_mode = 1;
        p.layers[0].effects.key_threshold = 0.55;
        render("key", p);

        let mut p = base_patch();
        p.temporal.as_mut().unwrap().feedback = 0.92;
        p.temporal.as_mut().unwrap().fb_zoom = 1.05;
        render("temporal_fb", p);

        let mut p = base_patch();
        p.temporal.as_mut().unwrap().slitscan = 0.85;
        render("slitscan", p);

        let mut p = base_patch();
        {
            let n = p.ntsc.as_mut().unwrap();
            n.enabled = true;
            n.snow_intensity = 0.9;
            n.tracking_noise_enabled = true;
            n.tracking_noise_snow = 0.7;
        }
        render("ntsc", p);

        let mut p = base_patch();
        p.layers[0].opacity = 0.3;
        render("opacity", p);
    }
}
