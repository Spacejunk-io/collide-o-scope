//! Optional, nonblocking GPU-stage timestamp collection.
//!
//! The ring owns every query, resolve, and readback resource at renderer
//! construction. A live frame either acquires one idle slot or records a
//! bounded drop; it never waits for the GPU and never allocates a replacement
//! slot. Mapping is attached to the submitted command buffer and harvested on
//! a later nonblocking [`wgpu::Device::poll`].

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

pub const GPU_TIMING_RING_SLOTS: usize = 3;
pub const GPU_TIMING_WINDOW: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GpuStage {
    SourcePrepare = 0,
    CreativeComposition = 1,
    TemporalMotion = 2,
    MoshVhs = 3,
    AudienceResolve = 4,
    Submission = 5,
}

impl GpuStage {
    pub const ALL: [Self; 6] = [
        Self::SourcePrepare,
        Self::CreativeComposition,
        Self::TemporalMotion,
        Self::MoshVhs,
        Self::AudienceResolve,
        Self::Submission,
    ];

    pub const COUNT: usize = Self::ALL.len();

    const fn start_query(self) -> u32 {
        self as u32 * 2
    }

    const fn end_query(self) -> u32 {
        self.start_query() + 1
    }

    const fn marker_bit(self, end: bool) -> u16 {
        1_u16 << (self as u16 * 2 + if end { 1 } else { 0 })
    }
}

const QUERY_COUNT: u32 = GpuStage::COUNT as u32 * 2;
const QUERY_BYTES: u64 = QUERY_COUNT as u64 * wgpu::QUERY_SIZE as u64;
const SLOT_IDLE: u8 = 0;
const SLOT_PENDING: u8 = 1;
const SLOT_READY: u8 = 2;
const SLOT_FAILED: u8 = 3;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuLatencyPercentiles {
    pub p50_us: u32,
    pub p95_us: u32,
    pub p99_us: u32,
    pub samples: u16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuTimingSnapshot {
    pub supported: bool,
    pub source_prepare: GpuLatencyPercentiles,
    pub creative_composition: GpuLatencyPercentiles,
    pub temporal_motion: GpuLatencyPercentiles,
    pub mosh_vhs: GpuLatencyPercentiles,
    pub audience_resolve: GpuLatencyPercentiles,
    pub submission: GpuLatencyPercentiles,
    pub last_submission_generation: u64,
    pub dropped_busy_frames: u64,
    pub map_failures: u64,
}

#[derive(Debug, Clone)]
struct StageWindow {
    values: [u32; GPU_TIMING_WINDOW],
    next: usize,
    count: usize,
}

impl Default for StageWindow {
    fn default() -> Self {
        Self {
            values: [0; GPU_TIMING_WINDOW],
            next: 0,
            count: 0,
        }
    }
}

impl StageWindow {
    fn record(&mut self, micros: u32) {
        self.values[self.next] = micros;
        self.next = (self.next + 1) % GPU_TIMING_WINDOW;
        self.count = (self.count + 1).min(GPU_TIMING_WINDOW);
    }

    fn snapshot(&self) -> GpuLatencyPercentiles {
        let mut sorted = self.values;
        sorted[..self.count].sort_unstable();
        GpuLatencyPercentiles {
            p50_us: percentile(&sorted[..self.count], 50),
            p95_us: percentile(&sorted[..self.count], 95),
            p99_us: percentile(&sorted[..self.count], 99),
            samples: u16::try_from(self.count).unwrap_or(u16::MAX),
        }
    }
}

struct GpuTimingSlot {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    state: Arc<AtomicU8>,
    submission_generation: u64,
}

pub(crate) struct SupportedGpuTiming {
    slots: [GpuTimingSlot; GPU_TIMING_RING_SLOTS],
    next_slot: usize,
    active_slot: Option<usize>,
    active_markers: u16,
    timestamp_period_ns: f64,
    windows: [StageWindow; GpuStage::COUNT],
    last_submission_generation: u64,
    dropped_busy_frames: u64,
    map_failures: u64,
}

/// Unsupported adapters retain an explicit state rather than publishing fake
/// zero-duration samples.
pub enum GpuTiming {
    Unsupported,
    Supported(Box<SupportedGpuTiming>),
}

impl GpuTiming {
    pub fn required_features(adapter_features: wgpu::Features) -> wgpu::Features {
        let required =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if adapter_features.contains(required) {
            required
        } else {
            wgpu::Features::empty()
        }
    }

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let required =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let period = f64::from(queue.get_timestamp_period());
        if !device.features().contains(required) || !period.is_finite() || period <= 0.0 {
            return Self::Unsupported;
        }
        let slots = std::array::from_fn(|index| GpuTimingSlot {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("stage GPU timestamp queries"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERY_COUNT,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stage GPU timestamp resolve"),
                size: QUERY_BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stage GPU timestamp readback"),
                size: QUERY_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            state: Arc::new(AtomicU8::new(SLOT_IDLE)),
            submission_generation: index as u64,
        });
        Self::Supported(Box::new(SupportedGpuTiming {
            slots,
            next_slot: 0,
            active_slot: None,
            active_markers: 0,
            timestamp_period_ns: period,
            windows: std::array::from_fn(|_| StageWindow::default()),
            last_submission_generation: 0,
            dropped_busy_frames: 0,
            map_failures: 0,
        }))
    }

    pub fn begin_frame(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
        submission_generation: u64,
    ) -> bool {
        let Self::Supported(timing) = self else {
            return false;
        };
        debug_assert!(timing.active_slot.is_none());
        for offset in 0..GPU_TIMING_RING_SLOTS {
            let index = (timing.next_slot + offset) % GPU_TIMING_RING_SLOTS;
            if timing.slots[index].state.load(Ordering::Acquire) == SLOT_IDLE {
                timing.slots[index].submission_generation = submission_generation;
                timing.active_slot = Some(index);
                timing.active_markers = 0;
                timing.next_slot = (index + 1) % GPU_TIMING_RING_SLOTS;
                return true;
            }
        }
        timing.dropped_busy_frames = timing.dropped_busy_frames.saturating_add(1);
        false
    }

    pub fn begin_stage(&mut self, encoder: &mut wgpu::CommandEncoder, stage: GpuStage) {
        let Self::Supported(timing) = self else {
            return;
        };
        let Some(index) = timing.active_slot else {
            return;
        };
        let bit = stage.marker_bit(false);
        if timing.active_markers & bit == 0 {
            encoder.write_timestamp(&timing.slots[index].queries, stage.start_query());
            timing.active_markers |= bit;
        }
    }

    pub fn end_stage(&mut self, encoder: &mut wgpu::CommandEncoder, stage: GpuStage) {
        self.begin_stage(encoder, stage);
        let Self::Supported(timing) = self else {
            return;
        };
        let Some(index) = timing.active_slot else {
            return;
        };
        let bit = stage.marker_bit(true);
        if timing.active_markers & bit == 0 {
            encoder.write_timestamp(&timing.slots[index].queries, stage.end_query());
            timing.active_markers |= bit;
        }
    }

    /// Resolve every named span, filling an omitted/inactive stage with two
    /// adjacent timestamps. Such a stage is a real near-zero sample, not an
    /// invented post-hoc zero.
    pub fn finish_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        for stage in GpuStage::ALL {
            self.end_stage(encoder, stage);
        }
        let Self::Supported(timing) = self else {
            return;
        };
        let Some(index) = timing.active_slot.take() else {
            return;
        };
        let slot = &timing.slots[index];
        encoder.resolve_query_set(&slot.queries, 0..QUERY_COUNT, &slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, QUERY_BYTES);
        slot.state.store(SLOT_PENDING, Ordering::Release);
        let state = slot.state.clone();
        encoder.map_buffer_on_submit(&slot.readback, wgpu::MapMode::Read, .., move |result| {
            state.store(
                if result.is_ok() {
                    SLOT_READY
                } else {
                    SLOT_FAILED
                },
                Ordering::Release,
            );
        });
        timing.active_markers = 0;
    }

    /// Release an encoder reservation whose commands will never be submitted.
    /// No result is published because no GPU work can truthfully own it.
    pub fn cancel_frame(&mut self) {
        let Self::Supported(timing) = self else {
            return;
        };
        timing.active_slot = None;
        timing.active_markers = 0;
    }

    /// Harvest any completed slots without waiting. The event loop may call
    /// this once per frame; `Poll` is explicitly the nonblocking wgpu mode.
    pub fn poll(&mut self, device: &wgpu::Device) -> GpuTimingSnapshot {
        let Self::Supported(timing) = self else {
            return GpuTimingSnapshot::default();
        };
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in &timing.slots {
            match slot.state.load(Ordering::Acquire) {
                SLOT_READY => {
                    let bytes = slot.readback.slice(..).get_mapped_range();
                    let mut timestamps = [0_u64; QUERY_COUNT as usize];
                    for (output, encoded) in timestamps.iter_mut().zip(bytes.chunks_exact(8)) {
                        *output = u64::from_le_bytes(encoded.try_into().expect("eight-byte query"));
                    }
                    drop(bytes);
                    slot.readback.unmap();
                    record_timestamps(&mut timing.windows, &timestamps, timing.timestamp_period_ns);
                    timing.last_submission_generation = timing
                        .last_submission_generation
                        .max(slot.submission_generation);
                    slot.state.store(SLOT_IDLE, Ordering::Release);
                }
                SLOT_FAILED => {
                    timing.map_failures = timing.map_failures.saturating_add(1);
                    slot.state.store(SLOT_IDLE, Ordering::Release);
                }
                _ => {}
            }
        }
        snapshot(timing)
    }
}

fn record_timestamps(
    windows: &mut [StageWindow; GpuStage::COUNT],
    timestamps: &[u64; QUERY_COUNT as usize],
    timestamp_period_ns: f64,
) {
    for stage in GpuStage::ALL {
        let start = timestamps[stage.start_query() as usize];
        let end = timestamps[stage.end_query() as usize];
        let micros = (end.saturating_sub(start) as f64 * timestamp_period_ns / 1_000.0)
            .round()
            .clamp(0.0, f64::from(u32::MAX)) as u32;
        windows[stage as usize].record(micros);
    }
}

fn snapshot(timing: &SupportedGpuTiming) -> GpuTimingSnapshot {
    GpuTimingSnapshot {
        supported: true,
        source_prepare: timing.windows[GpuStage::SourcePrepare as usize].snapshot(),
        creative_composition: timing.windows[GpuStage::CreativeComposition as usize].snapshot(),
        temporal_motion: timing.windows[GpuStage::TemporalMotion as usize].snapshot(),
        mosh_vhs: timing.windows[GpuStage::MoshVhs as usize].snapshot(),
        audience_resolve: timing.windows[GpuStage::AudienceResolve as usize].snapshot(),
        submission: timing.windows[GpuStage::Submission as usize].snapshot(),
        last_submission_generation: timing.last_submission_generation,
        dropped_busy_frames: timing.dropped_busy_frames,
        map_failures: timing.map_failures,
    }
}

fn percentile(sorted: &[u32], percentile: usize) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let index = sorted
        .len()
        .saturating_mul(percentile)
        .saturating_add(99)
        .saturating_div(100)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_features_are_optional_and_never_requested_partially() {
        assert!(GpuTiming::required_features(wgpu::Features::empty()).is_empty());
        assert!(GpuTiming::required_features(wgpu::Features::TIMESTAMP_QUERY).is_empty());
        let supported =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        assert_eq!(GpuTiming::required_features(supported), supported);
    }

    #[test]
    fn synthetic_query_results_classify_each_named_stage() {
        let mut windows = std::array::from_fn(|_| StageWindow::default());
        let mut timestamps = [0_u64; QUERY_COUNT as usize];
        for stage in GpuStage::ALL {
            timestamps[stage.start_query() as usize] = u64::from(stage as u8) * 100;
            timestamps[stage.end_query() as usize] =
                u64::from(stage as u8) * 100 + u64::from(stage as u8) + 1;
        }
        record_timestamps(&mut windows, &timestamps, 1_000.0);
        for stage in GpuStage::ALL {
            assert_eq!(
                windows[stage as usize].snapshot().p50_us,
                u32::from(stage as u8) + 1
            );
        }
    }

    #[test]
    fn synthetic_gpu_delay_moves_only_its_named_span() {
        let mut windows = std::array::from_fn(|_| StageWindow::default());
        let mut baseline = [0_u64; QUERY_COUNT as usize];
        for stage in GpuStage::ALL {
            baseline[stage.start_query() as usize] = u64::from(stage as u8) * 2_000;
            baseline[stage.end_query() as usize] = baseline[stage.start_query() as usize] + 10;
        }
        record_timestamps(&mut windows, &baseline, 1_000.0);

        let mut delayed = baseline;
        delayed[GpuStage::TemporalMotion.end_query() as usize] =
            delayed[GpuStage::TemporalMotion.start_query() as usize] + 2_500;
        record_timestamps(&mut windows, &delayed, 1_000.0);

        assert_eq!(
            windows[GpuStage::TemporalMotion as usize].snapshot().p95_us,
            2_500
        );
        for stage in GpuStage::ALL {
            if stage != GpuStage::TemporalMotion {
                assert_eq!(windows[stage as usize].snapshot().p95_us, 10);
            }
        }
    }

    #[test]
    fn stage_windows_are_fixed_newest_only_percentiles() {
        let mut window = StageWindow::default();
        for value in 0..GPU_TIMING_WINDOW * 2 {
            window.record(value as u32);
        }
        assert_eq!(usize::from(window.snapshot().samples), GPU_TIMING_WINDOW);
        assert!(window.snapshot().p50_us >= GPU_TIMING_WINDOW as u32);
    }
}
