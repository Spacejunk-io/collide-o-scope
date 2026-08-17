//! Request-driven threaded wrapper around [`VideoDecoder`].
//!
//! The render thread requests a bounded number of media-frame advances and
//! polls a one-frame overwrite mailbox. The worker decodes every requested
//! advance in order but publishes only the newest RGBA image, so fast
//! transport does not require a queue of full-resolution frames. A seeded
//! first frame is available immediately after `open`, including while a layer
//! is paused.

use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::media_safety::{
    MediaAllocationPlan, MediaDeviceLimits, MediaSafetyPolicy, MediaSourceKind,
};

#[cfg(test)]
use super::decoder::validate_media_dimensions;
use super::decoder::validate_media_dimensions_with_policy as plan_media_dimensions;
use super::VideoDecoder;

/// Maximum media-frame advances accepted in one render-tick request.
pub const MAX_ADVANCE_FRAMES: u32 = 8;
const DECODER_OPEN_TIMEOUT: Duration = Duration::from_secs(7);

#[derive(Debug)]
pub struct DecodedFrame {
    pub rgba: Vec<u8>,
    /// Loop progress 0.0..1.0 at the time this frame was decoded.
    pub progress: f32,
    /// Cumulative successful EOF reopens at the time this frame was decoded.
    pub loop_generation: u64,
}

/// Newest completed media frame plus loop boundaries crossed since the
/// consumer last took a frame.
#[derive(Debug, PartialEq, Eq)]
pub struct ReadyFrame {
    pub rgba: Vec<u8>,
    pub loops_advanced: u64,
}

/// Stable decoder state for snapshots and operator-facing health reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderHealth {
    Healthy,
    Failed(String),
}

#[derive(Debug)]
struct SharedState {
    /// A one-frame overwrite mailbox. Publishing a newer result drops the
    /// unconsumed older image instead of accumulating display latency/memory.
    latest: Option<DecodedFrame>,
    health: DecoderHealth,
    health_revision: u64,
}

impl SharedState {
    fn healthy_with_seed(seed: DecodedFrame) -> Self {
        Self {
            latest: Some(seed),
            health: DecoderHealth::Healthy,
            health_revision: 1,
        }
    }

    fn set_failed(&mut self, error: String) {
        if matches!(self.health, DecoderHealth::Healthy) {
            self.health = DecoderHealth::Failed(error);
            self.health_revision = self.health_revision.saturating_add(1);
        }
    }
}

enum DecodeRequest {
    Advance(u32),
}

pub struct ThreadedDecoder {
    request_tx: SyncSender<DecodeRequest>,
    shared: Arc<Mutex<SharedState>>,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    media_plan: MediaAllocationPlan,
    /// Progress of the most recently consumed frame.
    progress: f32,
    /// Loop generation of the most recently consumed frame. Keeping this at
    /// the consumer boundary preserves loop events when the mailbox drops
    /// intermediate images.
    consumed_loop_generation: u64,
    failure_revision_reported: u64,
}

impl ThreadedDecoder {
    /// Open and validate codec dimensions before decoding/allocating the
    /// seeded RGBA frame. Live layers pass the active device's 2D texture
    /// limit so malformed or unsupported media fails before GPU allocation.
    #[allow(dead_code)]
    pub fn open_with_texture_limit(path: &str, max_dimension: u32) -> Result<Self, String> {
        Self::open_inner(
            path,
            MediaSafetyPolicy::safe(),
            MediaDeviceLimits::texture_only(max_dimension),
        )
    }

    /// Open with a process-local media policy. Device edge and per-buffer
    /// limits are checked before FFmpeg allocates its RGBA scaler output.
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
        let (request_tx, request_rx) = std::sync::mpsc::sync_channel(1);
        let (meta_tx, meta_rx) =
            std::sync::mpsc::channel::<Result<(u32, u32, f32, MediaAllocationPlan), String>>();

        let thread_name = format!("decode-{}", short_name(path));
        let path_owned = path.to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let shared = Arc::new(Mutex::new(SharedState {
            latest: None,
            health: DecoderHealth::Healthy,
            health_revision: 1,
        }));
        let worker_shared = shared.clone();
        let worker_media_policy = media_policy;
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let worker = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut decoder = match VideoDecoder::open_with_cancel_and_media_policy(
                        &path_owned,
                        worker_cancel,
                        &worker_media_policy,
                        device_limits,
                    ) {
                        Ok(decoder) => decoder,
                        Err(error) => {
                            let _ = meta_tx.send(Err(error));
                            return;
                        }
                    };

                    let rgba = match decoder.next_frame_result() {
                        Ok(rgba) => rgba,
                        Err(error) => {
                            let _ = meta_tx.send(Err(format!(
                                "failed to decode initial video frame: {error}"
                            )));
                            return;
                        }
                    };
                    let seed = DecodedFrame {
                        rgba,
                        progress: decoder.progress(),
                        loop_generation: decoder.loop_generation(),
                    };
                    let dimensions = (
                        decoder.width,
                        decoder.height,
                        decoder.fps,
                        decoder.media_allocation_plan().clone(),
                    );
                    *lock_state(&worker_shared) = SharedState::healthy_with_seed(seed);
                    if meta_tx
                        .send(Ok((dimensions.0, dimensions.1, dimensions.2, dimensions.3)))
                        .is_err()
                    {
                        return;
                    }

                    run_decode_requests(request_rx, worker_shared.clone(), || {
                        let rgba = decoder.next_frame_result()?;
                        Ok(DecodedFrame {
                            rgba,
                            progress: decoder.progress(),
                            loop_generation: decoder.loop_generation(),
                        })
                    });
                }));

                if worker.is_err() {
                    lock_state(&worker_shared)
                        .set_failed("Decode worker panicked unexpectedly".to_string());
                }
            })
            .map_err(|error| format!("Failed to spawn decode thread: {error}"))?;

        let (width, height, fps, media_plan) = match meta_rx.recv_timeout(DECODER_OPEN_TIMEOUT) {
            Ok(result) => result?,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                cancel.store(true, std::sync::atomic::Ordering::Release);
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
            request_tx,
            shared,
            width,
            height,
            fps,
            media_plan,
            progress: 0.0,
            consumed_loop_generation: 0,
            failure_revision_reported: 0,
        })
    }

    /// Queue up to eight media-frame advances without blocking. One request
    /// can be executing and one can be queued; a full queue accepts zero so
    /// the caller can retain pacing debt for a later tick.
    pub fn request_frames(&mut self, count: u32) -> Result<u32, String> {
        if count == 0 {
            return Ok(0);
        }
        if let Some(error) = self.terminal_error_once() {
            return Err(error);
        }
        if self.is_finished() {
            return Ok(0);
        }

        let requested = count.min(MAX_ADVANCE_FRAMES);
        match self.request_tx.try_send(DecodeRequest::Advance(requested)) {
            Ok(()) => Ok(requested),
            Err(TrySendError::Full(_)) => Ok(0),
            Err(TrySendError::Disconnected(_)) => {
                lock_state(&self.shared)
                    .set_failed("Decode worker disconnected unexpectedly".to_string());
                match self.terminal_error_once() {
                    Some(error) => Err(error),
                    None => Ok(0),
                }
            }
        }
    }

    /// Non-blocking: take the newest completed frame, with the number of loop
    /// boundaries crossed since the previously consumed frame. Intermediate
    /// images may be overwritten, but their loop boundaries are retained by
    /// the cumulative generation carried on the newest image. A terminal
    /// worker error is returned once; callers can inspect
    /// [`health`](Self::health) at any time for stable state.
    pub fn try_next_ready_frame_result(&mut self) -> Result<Option<ReadyFrame>, String> {
        let frame = lock_state(&self.shared).latest.take();
        if let Some(frame) = frame {
            self.progress = frame.progress;
            let loops_advanced = frame
                .loop_generation
                .saturating_sub(self.consumed_loop_generation);
            self.consumed_loop_generation =
                self.consumed_loop_generation.max(frame.loop_generation);
            return Ok(Some(ReadyFrame {
                rgba: frame.rgba,
                loops_advanced,
            }));
        }
        if let Some(error) = self.terminal_error_once() {
            return Err(error);
        }
        Ok(None)
    }

    /// Compatibility poll for callers that only consume RGBA bytes. This has
    /// the same latest-only and one-shot-error behavior as before, but
    /// intentionally discards loop-boundary metadata.
    #[allow(dead_code)]
    pub fn try_next_frame_result(&mut self) -> Result<Option<Vec<u8>>, String> {
        self.try_next_ready_frame_result()
            .map(|ready| ready.map(|frame| frame.rgba))
    }

    /// Loop progress of the most recently consumed frame, 0.0..1.0.
    pub fn progress(&self) -> f32 {
        self.progress
    }

    pub fn media_allocation_plan(&self) -> &MediaAllocationPlan {
        &self.media_plan
    }

    /// Whether the worker has terminated because of an unrecoverable failure.
    pub fn is_finished(&self) -> bool {
        matches!(self.health(), DecoderHealth::Failed(_))
    }

    /// Stable health snapshot. Unlike the one-shot error returned by frame
    /// polling, a terminal failure remains visible for panel snapshots.
    pub fn health(&self) -> DecoderHealth {
        lock_state(&self.shared).health.clone()
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

fn run_decode_requests<F>(
    request_rx: std::sync::mpsc::Receiver<DecodeRequest>,
    shared: Arc<Mutex<SharedState>>,
    mut next_frame: F,
) where
    F: FnMut() -> Result<DecodedFrame, String>,
{
    while let Ok(DecodeRequest::Advance(count)) = request_rx.recv() {
        let mut newest = None;
        let mut failure = None;
        for _ in 0..count.min(MAX_ADVANCE_FRAMES) {
            match next_frame() {
                Ok(frame) => newest = Some(frame),
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }

        let mut state = lock_state(&shared);
        if let Some(frame) = newest {
            state.latest = Some(frame);
        }
        if let Some(error) = failure {
            state.set_failed(error);
            return;
        }
    }
}

fn lock_state(state: &Arc<Mutex<SharedState>>) -> MutexGuard<'_, SharedState> {
    state
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
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    fn synthetic_decoder(
        seed: DecodedFrame,
    ) -> (ThreadedDecoder, std::sync::mpsc::Receiver<DecodeRequest>) {
        let (request_tx, request_rx) = std::sync::mpsc::sync_channel(1);
        let shared = Arc::new(Mutex::new(SharedState::healthy_with_seed(seed)));
        (
            ThreadedDecoder {
                request_tx,
                shared,
                width: 1,
                height: 1,
                fps: 30.0,
                media_plan: crate::media_safety::validate_safe_dimensions(
                    MediaSourceKind::Video,
                    1,
                    1,
                    MediaDeviceLimits::none(),
                )
                .unwrap(),
                progress: 0.0,
                consumed_loop_generation: 0,
                failure_revision_reported: 0,
            },
            request_rx,
        )
    }

    #[test]
    fn initial_frame_is_seeded_without_an_advance_request() {
        let (mut decoder, _request_rx) = synthetic_decoder(DecodedFrame {
            rgba: vec![1, 2, 3, 4],
            progress: 0.25,
            loop_generation: 0,
        });

        assert_eq!(
            decoder.try_next_frame_result().unwrap(),
            Some(vec![1, 2, 3, 4])
        );
        assert_eq!(decoder.progress(), 0.25);
        assert_eq!(decoder.try_next_frame_result().unwrap(), None);
    }

    #[test]
    fn eight_advances_publish_latest_frame_and_all_loop_boundaries() {
        let (mut decoder, request_rx) = synthetic_decoder(DecodedFrame {
            rgba: vec![0],
            progress: 0.0,
            loop_generation: 0,
        });
        assert_eq!(decoder.try_next_frame_result().unwrap(), Some(vec![0]));

        let shared = decoder.shared.clone();
        let calls = Arc::new(AtomicU32::new(0));
        let worker_calls = calls.clone();
        let worker = std::thread::spawn(move || {
            run_decode_requests(request_rx, shared, || {
                let value = worker_calls.fetch_add(1, Ordering::Relaxed) + 1;
                Ok(DecodedFrame {
                    rgba: vec![value as u8],
                    progress: value as f32 / 8.0,
                    loop_generation: u64::from(value),
                })
            });
        });

        assert_eq!(decoder.request_frames(99).unwrap(), MAX_ADVANCE_FRAMES);
        let deadline = Instant::now() + Duration::from_secs(1);
        let newest = loop {
            if let Some(frame) = decoder.try_next_ready_frame_result().unwrap() {
                break frame;
            }
            assert!(
                Instant::now() < deadline,
                "synthetic decode worker timed out"
            );
            std::thread::yield_now();
        };
        assert_eq!(
            newest,
            ReadyFrame {
                rgba: vec![8],
                loops_advanced: 8,
            }
        );
        assert_eq!(calls.load(Ordering::Relaxed), 8);
        drop(decoder);
        worker.join().unwrap();
    }

    #[test]
    fn worker_failure_is_one_shot_but_health_remains_stable() {
        let (mut decoder, _request_rx) = synthetic_decoder(DecodedFrame {
            rgba: vec![0],
            progress: 0.0,
            loop_generation: 0,
        });
        decoder.try_next_frame_result().unwrap();
        lock_state(&decoder.shared).set_failed("synthetic decode failure".into());

        assert_eq!(
            decoder.try_next_frame_result().unwrap_err(),
            "synthetic decode failure"
        );
        assert_eq!(decoder.try_next_frame_result().unwrap(), None);
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

    #[test]
    fn ready_frame_reports_only_unconsumed_loop_generations() {
        let (mut decoder, _request_rx) = synthetic_decoder(DecodedFrame {
            rgba: vec![1],
            progress: 0.0,
            loop_generation: 3,
        });

        assert_eq!(
            decoder.try_next_ready_frame_result().unwrap(),
            Some(ReadyFrame {
                rgba: vec![1],
                loops_advanced: 3,
            })
        );

        lock_state(&decoder.shared).latest = Some(DecodedFrame {
            rgba: vec![2],
            progress: 0.5,
            loop_generation: 3,
        });
        assert_eq!(
            decoder.try_next_ready_frame_result().unwrap(),
            Some(ReadyFrame {
                rgba: vec![2],
                loops_advanced: 0,
            })
        );

        // A stale generation must neither underflow nor rewind the consumer.
        lock_state(&decoder.shared).latest = Some(DecodedFrame {
            rgba: vec![3],
            progress: 0.75,
            loop_generation: 2,
        });
        assert_eq!(
            decoder.try_next_ready_frame_result().unwrap(),
            Some(ReadyFrame {
                rgba: vec![3],
                loops_advanced: 0,
            })
        );

        lock_state(&decoder.shared).latest = Some(DecodedFrame {
            rgba: vec![4],
            progress: 0.0,
            loop_generation: 5,
        });
        assert_eq!(
            decoder.try_next_ready_frame_result().unwrap(),
            Some(ReadyFrame {
                rgba: vec![4],
                loops_advanced: 2,
            })
        );
    }

    #[test]
    fn real_eof_generations_drive_the_live_reroll_seam_once_each() {
        fn wait_for_batch(decoder: &mut ThreadedDecoder) -> ReadyFrame {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(frame) = decoder.try_next_ready_frame_result().unwrap() {
                    return frame;
                }
                assert!(
                    Instant::now() < deadline,
                    "real decode worker timed out while waiting for a requested batch"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("loop-72f.mp4");
        assert!(
            fixture.is_file(),
            "missing decode fixture: {}",
            fixture.display()
        );

        let mut decoder =
            ThreadedDecoder::open_with_texture_limit(&fixture.to_string_lossy(), 8_192).unwrap();
        let initial = wait_for_batch(&mut decoder);
        assert_eq!(initial.loops_advanced, 0);

        let initial_seed = 0x1020_3040;
        let mut effects = crate::effects::EffectUniforms {
            random_seed: initial_seed,
            ..Default::default()
        };
        crate::randomization::apply_live_loop_reroll(&mut effects, true, initial.loops_advanced);

        let mut reported_generations = 0u64;
        let mut saw_repeated_first_generation = false;
        let mut saw_crossing_in_latest_only_batch = false;

        // Seven real decodes are collapsed into each one-frame publication.
        // The 72-frame fixture makes EOF fall inside a batch rather than on
        // every batch boundary, exercising loop metadata after older RGBA
        // images from that request have been discarded.
        for _ in 0..32 {
            assert_eq!(decoder.request_frames(7).unwrap(), 7);
            let frame = wait_for_batch(&mut decoder);
            let consumed_generation = decoder.consumed_loop_generation;
            let seed_before = effects.random_seed;

            reported_generations = reported_generations.saturating_add(frame.loops_advanced);
            assert_eq!(reported_generations, consumed_generation);
            crate::randomization::apply_live_loop_reroll(&mut effects, true, frame.loops_advanced);

            assert_eq!(
                effects.random_seed,
                crate::randomization::advance_seed(initial_seed, consumed_generation),
                "the live seam must consume each cumulative loop generation exactly once"
            );
            if frame.loops_advanced == 0 {
                assert_eq!(effects.random_seed, seed_before);
                saw_repeated_first_generation |= consumed_generation == 1;
            } else {
                assert_eq!(frame.loops_advanced, 1);
                assert!(decoder.progress() > 0.0);
                saw_crossing_in_latest_only_batch = true;
            }

            if consumed_generation == 2 && saw_repeated_first_generation {
                break;
            }
        }

        assert_eq!(decoder.consumed_loop_generation, 2);
        assert!(saw_repeated_first_generation);
        assert!(saw_crossing_in_latest_only_batch);
        assert_eq!(
            effects.random_seed,
            crate::randomization::advance_seed(initial_seed, 2)
        );
    }
}
