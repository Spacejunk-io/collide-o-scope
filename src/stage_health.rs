//! Preview-only stage health and explicitly endpoint-scoped calibration tools.
//!
//! This module has no access to renderer textures or composition executors.
//! The health painter accepts only an `egui::Ui` plus a permit that can be
//! minted only for the editor preview. Test cards and output identification
//! are pure decisions keyed to one exact physical output endpoint.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::image_routing::StableLayerId;
#[cfg(test)]
use crate::stage_map::StageCalibrationDecision;
pub use crate::stage_map::{OutputEndpointId, StageSurface, StageToolState, TestCardMode};

pub const STAGE_HEALTH_FRAME_WINDOW: usize = 512;
pub const STAGE_HEALTH_MAX_LAYERS: usize = 256;
pub const STAGE_HEALTH_MAX_TEXT_BYTES: usize = 256;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// One accepted program-frame deadline from a drift-free rational schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FramePacerTick {
    pub wall_interval: Duration,
    pub schedule_lateness: Duration,
    /// Whole due program ticks intentionally skipped to resume at the newest
    /// deadline. A fractional/ordinary scheduling overrun is not a skip.
    pub skipped_program_ticks: u64,
}

/// Fixed-rate wall scheduler whose deadline `n` is
/// `origin + floor(n * 1 second / fps)`. It never rebases a successful tick to
/// `now`, so normal timer jitter cannot accumulate as permanent phase drift.
#[derive(Debug, Clone)]
pub struct RationalFramePacer {
    fps: u64,
    origin: Instant,
    next_ordinal: u64,
    last_accepted_at: Instant,
}

impl RationalFramePacer {
    pub fn new(origin: Instant, fps: u64) -> Self {
        assert!(fps > 0, "a frame pacer requires a nonzero rate");
        Self {
            fps,
            origin,
            next_ordinal: 1,
            last_accepted_at: origin,
        }
    }

    pub fn reset(&mut self, origin: Instant) {
        self.origin = origin;
        self.next_ordinal = 1;
        self.last_accepted_at = origin;
    }

    #[cfg(test)]
    pub fn last_accepted_at(&self) -> Instant {
        self.last_accepted_at
    }

    pub fn next_deadline(&self) -> Instant {
        let deadline_nanos =
            u128::from(self.next_ordinal).saturating_mul(NANOS_PER_SECOND) / u128::from(self.fps);
        self.origin
            .checked_add(Duration::from_nanos(
                u64::try_from(deadline_nanos).unwrap_or(u64::MAX),
            ))
            .unwrap_or(self.last_accepted_at)
    }

    /// Return the newest due tick, coalescing whole missed periods. The event
    /// thread calls this nonblocking method on redraw; `None` means the next
    /// rational deadline has not arrived.
    pub fn accept_due(&mut self, now: Instant) -> Option<FramePacerTick> {
        let elapsed_nanos = now.saturating_duration_since(self.origin).as_nanos();
        // Largest ordinal whose floored rational deadline is <= elapsed.
        // The +1/-1 form is the exact inverse of floor(n * second / fps).
        let due_ordinal = elapsed_nanos
            .saturating_add(1)
            .saturating_mul(u128::from(self.fps))
            .saturating_sub(1)
            / NANOS_PER_SECOND;
        if due_ordinal < u128::from(self.next_ordinal) {
            return None;
        }
        let due_ordinal = u64::try_from(due_ordinal).unwrap_or(u64::MAX);
        let skipped_program_ticks = due_ordinal.saturating_sub(self.next_ordinal);
        let deadline_nanos =
            u128::from(due_ordinal).saturating_mul(NANOS_PER_SECOND) / u128::from(self.fps);
        let deadline_offset =
            Duration::from_nanos(u64::try_from(deadline_nanos).unwrap_or(u64::MAX));
        let deadline = self.origin.checked_add(deadline_offset).unwrap_or(now);
        let tick = FramePacerTick {
            wall_interval: now.saturating_duration_since(self.last_accepted_at),
            schedule_lateness: now.saturating_duration_since(deadline),
            skipped_program_ticks,
        };
        self.next_ordinal = due_ordinal.saturating_add(1);
        self.last_accepted_at = now;
        Some(tick)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageToolsSnapshot {
    pub health_hud_enabled: bool,
    pub test_card: TestCardMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_card_endpoint_id: Option<String>,
    pub output_identification_enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_identification_endpoint_id: Option<String>,
}

/// Opaque proof that the caller selected the editor preview surface.
pub struct EditorPreviewPermit(());

/// Health owns the editor-only permit; StageMap owns every physical endpoint
/// decision. Generic audience/composite consumers can never mint this token.
pub fn editor_preview_permit(
    tools: &StageToolState,
    surface: &StageSurface,
) -> Option<EditorPreviewPermit> {
    (tools.health_hud_enabled() && matches!(surface, StageSurface::EditorPreview))
        .then_some(EditorPreviewPermit(()))
}

fn stage_tools_snapshot(tools: &StageToolState) -> StageToolsSnapshot {
    StageToolsSnapshot {
        health_hud_enabled: tools.health_hud_enabled(),
        test_card: tools.test_card(),
        test_card_endpoint_id: tools
            .test_card_endpoint()
            .map(|endpoint| endpoint.as_str().to_string()),
        output_identification_enabled: tools.output_identification_enabled(),
        output_identification_endpoint_id: tools
            .output_identification_endpoint()
            .map(|endpoint| endpoint.as_str().to_string()),
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageBudgetSnapshot {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub unit: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageBudgetSetSnapshot {
    pub gpu: StageBudgetSnapshot,
    pub media: StageBudgetSnapshot,
    pub ntsc: StageBudgetSnapshot,
    pub motion: StageBudgetSnapshot,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageOutputSnapshot {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub endpoint_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub identity: String,
    pub width: u32,
    pub height: u32,
    /// Integer millihertz avoids unstable float text and exactly represents
    /// common 59.94/29.97 endpoint modes.
    pub refresh_millihz: u32,
    pub active: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageGpuSnapshot {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub adapter: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub backend: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub device: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerStageHealthSnapshot {
    pub layer_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoded_age_ms: Option<u64>,
    pub pending_frames: u32,
    pub dropped_frames: u64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
    #[serde(default)]
    pub source_color: crate::video::SourceColorDescriptor,
    #[serde(default)]
    pub source_display: crate::video::SourceDisplayDescriptor,
    #[serde(default)]
    pub conversion_policy: crate::video::SourceConversionPolicy,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageLatencyPercentilesSnapshot {
    pub p50_us: u32,
    pub p95_us: u32,
    pub p99_us: u32,
    pub samples: u16,
}

impl From<crate::action_correlation::LatencyPercentiles> for StageLatencyPercentilesSnapshot {
    fn from(value: crate::action_correlation::LatencyPercentiles) -> Self {
        Self {
            p50_us: value.p50_us,
            p95_us: value.p95_us,
            p99_us: value.p99_us,
            samples: value.samples,
        }
    }
}

impl From<crate::renderer::gpu_timing::GpuLatencyPercentiles> for StageLatencyPercentilesSnapshot {
    fn from(value: crate::renderer::gpu_timing::GpuLatencyPercentiles) -> Self {
        Self {
            p50_us: value.p50_us,
            p95_us: value.p95_us,
            p99_us: value.p99_us,
            samples: value.samples,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageActionTimingSnapshot {
    pub ingress_to_apply: StageLatencyPercentilesSnapshot,
    pub apply_to_submit: StageLatencyPercentilesSnapshot,
    pub last_presented_sequence: u64,
    pub last_submission_generation: u64,
    pub pending: u16,
    pub uncorrelated_over_capacity: u64,
}

impl From<crate::action_correlation::ActionTimingSnapshot> for StageActionTimingSnapshot {
    fn from(value: crate::action_correlation::ActionTimingSnapshot) -> Self {
        Self {
            ingress_to_apply: value.ingress_to_apply.into(),
            apply_to_submit: value.apply_to_submit.into(),
            last_presented_sequence: value.last_presented_sequence,
            last_submission_generation: value.last_submission_generation,
            pending: value.pending,
            uncorrelated_over_capacity: value.uncorrelated_over_capacity,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageGpuTimingSnapshot {
    pub supported: bool,
    pub source_prepare: StageLatencyPercentilesSnapshot,
    pub creative_composition: StageLatencyPercentilesSnapshot,
    pub temporal_motion: StageLatencyPercentilesSnapshot,
    pub mosh_vhs: StageLatencyPercentilesSnapshot,
    pub audience_resolve: StageLatencyPercentilesSnapshot,
    pub submission: StageLatencyPercentilesSnapshot,
    pub last_submission_generation: u64,
    pub dropped_busy_frames: u64,
    pub map_failures: u64,
}

impl From<crate::renderer::gpu_timing::GpuTimingSnapshot> for StageGpuTimingSnapshot {
    fn from(value: crate::renderer::gpu_timing::GpuTimingSnapshot) -> Self {
        Self {
            supported: value.supported,
            source_prepare: value.source_prepare.into(),
            creative_composition: value.creative_composition.into(),
            temporal_motion: value.temporal_motion.into(),
            mosh_vhs: value.mosh_vhs.into(),
            audience_resolve: value.audience_resolve.into(),
            submission: value.submission.into(),
            last_submission_generation: value.last_submission_generation,
            dropped_busy_frames: value.dropped_busy_frames,
            map_failures: value.map_failures,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageFlightRecorderSnapshot {
    pub enabled: bool,
    pub queued_events: u64,
    pub dropped_full: u64,
    /// Fixed production policy: active plus two independently durable files.
    pub retained_rotations: u8,
    pub rotation_seconds: u8,
    pub total_byte_cap: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageDecoderRetirementHealth {
    #[default]
    Healthy,
    Saturated,
    Stuck,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageDecoderRetirementSnapshot {
    pub health: StageDecoderRetirementHealth,
    pub active_workers: u16,
    pub retiring_workers: u16,
    pub stuck_workers: u16,
    pub owned_workers: u16,
    pub peak_owned_workers: u16,
    pub peak_retiring_workers: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oldest_retirement_age_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oldest_worker_id: Option<u64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub oldest_source_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oldest_source_generation: Option<u64>,
    pub completed_workers: u64,
    pub panicked_workers: u64,
    pub admission_refusals: u64,
    pub hard_cap: u16,
    pub churn_cap: u16,
    pub accepting_new_workers: bool,
}

impl Default for StageDecoderRetirementSnapshot {
    fn default() -> Self {
        Self {
            health: StageDecoderRetirementHealth::Healthy,
            active_workers: 0,
            retiring_workers: 0,
            stuck_workers: 0,
            owned_workers: 0,
            peak_owned_workers: 0,
            peak_retiring_workers: 0,
            oldest_retirement_age_ms: None,
            oldest_worker_id: None,
            oldest_source_fingerprint: String::new(),
            oldest_source_generation: None,
            completed_workers: 0,
            panicked_workers: 0,
            admission_refusals: 0,
            hard_cap: u16::try_from(crate::video::DECODER_WORKER_HARD_CAP).unwrap_or(u16::MAX),
            churn_cap: u16::try_from(crate::video::DECODER_RETIREMENT_CHURN_CAP)
                .unwrap_or(u16::MAX),
            accepting_new_workers: true,
        }
    }
}

impl From<crate::video::DecoderRetirementSnapshot> for StageDecoderRetirementSnapshot {
    fn from(value: crate::video::DecoderRetirementSnapshot) -> Self {
        let health = match value.health {
            crate::video::DecoderRetirementHealth::Healthy => StageDecoderRetirementHealth::Healthy,
            crate::video::DecoderRetirementHealth::Saturated => {
                StageDecoderRetirementHealth::Saturated
            }
            crate::video::DecoderRetirementHealth::Stuck => StageDecoderRetirementHealth::Stuck,
        };
        Self {
            health,
            active_workers: u16::try_from(value.active_workers).unwrap_or(u16::MAX),
            retiring_workers: u16::try_from(value.retiring_workers).unwrap_or(u16::MAX),
            stuck_workers: u16::try_from(value.stuck_workers).unwrap_or(u16::MAX),
            owned_workers: u16::try_from(value.owned_workers).unwrap_or(u16::MAX),
            peak_owned_workers: u16::try_from(value.peak_owned_workers).unwrap_or(u16::MAX),
            peak_retiring_workers: u16::try_from(value.peak_retiring_workers).unwrap_or(u16::MAX),
            oldest_retirement_age_ms: value
                .oldest_retirement_age
                .map(|age| u64::try_from(age.as_millis()).unwrap_or(u64::MAX)),
            oldest_worker_id: value.oldest_retiree.map(|identity| identity.worker_id),
            oldest_source_fingerprint: value
                .oldest_retiree
                .map_or_else(String::new, |identity| identity.source.short_hex()),
            oldest_source_generation: value
                .oldest_retiree
                .map(|identity| identity.source_generation),
            completed_workers: value.completed_workers,
            panicked_workers: value.panicked_workers,
            admission_refusals: value.admission_refusals,
            hard_cap: u16::try_from(value.hard_cap).unwrap_or(u16::MAX),
            churn_cap: u16::try_from(value.churn_cap).unwrap_or(u16::MAX),
            accepting_new_workers: value.accepting_new_workers,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageHealthSnapshot {
    #[serde(default = "current_build_identity_snapshot")]
    pub build_identity: crate::build_identity::BuildIdentitySnapshot,
    pub fps: f32,
    pub frame_time_p50_us: u32,
    pub frame_time_p95_us: u32,
    pub frame_time_p99_us: u32,
    pub frame_samples: u16,
    #[serde(default)]
    pub schedule_lateness_p50_us: u32,
    #[serde(default)]
    pub schedule_lateness_p95_us: u32,
    #[serde(default)]
    pub schedule_lateness_p99_us: u32,
    #[serde(default)]
    pub skipped_program_ticks: u64,
    /// Backwards-compatible alias for `skipped_program_ticks`. This no longer
    /// increments for a sub-period scheduling overrun.
    pub missed_deadlines: u64,
    pub layers: Vec<LayerStageHealthSnapshot>,
    pub layers_truncated: bool,
    pub output: StageOutputSnapshot,
    pub gpu: StageGpuSnapshot,
    #[serde(default)]
    pub gpu_timing: StageGpuTimingSnapshot,
    #[serde(default)]
    pub action_timing: StageActionTimingSnapshot,
    #[serde(default)]
    pub flight_recorder: StageFlightRecorderSnapshot,
    #[serde(default)]
    pub decoder_retirement: StageDecoderRetirementSnapshot,
    pub budgets: StageBudgetSetSnapshot,
    pub tools: StageToolsSnapshot,
}

impl Default for StageHealthSnapshot {
    fn default() -> Self {
        Self {
            build_identity: current_build_identity_snapshot(),
            fps: 0.0,
            frame_time_p50_us: 0,
            frame_time_p95_us: 0,
            frame_time_p99_us: 0,
            frame_samples: 0,
            schedule_lateness_p50_us: 0,
            schedule_lateness_p95_us: 0,
            schedule_lateness_p99_us: 0,
            skipped_program_ticks: 0,
            missed_deadlines: 0,
            layers: Vec::new(),
            layers_truncated: false,
            output: StageOutputSnapshot::default(),
            gpu: StageGpuSnapshot::default(),
            gpu_timing: StageGpuTimingSnapshot::default(),
            action_timing: StageActionTimingSnapshot::default(),
            flight_recorder: StageFlightRecorderSnapshot::default(),
            decoder_retirement: StageDecoderRetirementSnapshot::default(),
            budgets: StageBudgetSetSnapshot::default(),
            tools: StageToolsSnapshot::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LayerStageHealthInput<'a> {
    pub layer_id: StableLayerId,
    pub name: &'a str,
    pub decoded_age: Option<Duration>,
    pub pending_frames: u32,
    pub dropped_frames: u64,
    pub status: &'a str,
    pub source_color: crate::video::SourceColorDescriptor,
    pub source_display: crate::video::SourceDisplayDescriptor,
    pub conversion_policy: crate::video::SourceConversionPolicy,
}

#[derive(Debug, Clone, Copy)]
pub struct StageOutputInput<'a> {
    /// Typed StageMap identity prevents telemetry from introducing a second,
    /// less strict endpoint namespace.
    pub endpoint_id: &'a OutputEndpointId,
    pub identity: &'a str,
    pub width: u32,
    pub height: u32,
    pub refresh_millihz: u32,
    pub active: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct StageGpuInput<'a> {
    pub adapter: &'a str,
    pub backend: &'a str,
    pub device: &'a str,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StageBudgetInput<'a> {
    pub unit: &'a str,
    pub used: Option<u64>,
    pub limit: Option<u64>,
    pub detail: &'a str,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StageBudgetSetInput<'a> {
    pub gpu: StageBudgetInput<'a>,
    pub media: StageBudgetInput<'a>,
    pub ntsc: StageBudgetInput<'a>,
    pub motion: StageBudgetInput<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct StageHealthPublishInput<'a> {
    pub layers: &'a [LayerStageHealthInput<'a>],
    pub output: StageOutputInput<'a>,
    pub gpu: StageGpuInput<'a>,
    pub gpu_timing: crate::renderer::gpu_timing::GpuTimingSnapshot,
    pub action_timing: crate::action_correlation::ActionTimingSnapshot,
    pub budgets: StageBudgetSetInput<'a>,
    pub tools: &'a StageToolState,
}

pub struct StageHealthMonitor {
    frame_times_us: [u32; STAGE_HEALTH_FRAME_WINDOW],
    schedule_lateness_us: [u32; STAGE_HEALTH_FRAME_WINDOW],
    next: usize,
    count: usize,
    total_frame_time_us: u64,
    missed_deadlines: u64,
}

impl Default for StageHealthMonitor {
    fn default() -> Self {
        Self {
            frame_times_us: [0; STAGE_HEALTH_FRAME_WINDOW],
            schedule_lateness_us: [0; STAGE_HEALTH_FRAME_WINDOW],
            next: 0,
            count: 0,
            total_frame_time_us: 0,
            missed_deadlines: 0,
        }
    }
}

impl StageHealthMonitor {
    /// Fixed-storage sample insertion. No allocation occurs on the frame path.
    pub fn observe_frame(
        &mut self,
        elapsed: Duration,
        schedule_lateness: Duration,
        skipped_program_ticks: u64,
    ) {
        let elapsed_us = elapsed.as_micros().min(u128::from(u32::MAX)) as u32;
        let lateness_us = schedule_lateness.as_micros().min(u128::from(u32::MAX)) as u32;
        if self.count == STAGE_HEALTH_FRAME_WINDOW {
            self.total_frame_time_us = self
                .total_frame_time_us
                .saturating_sub(u64::from(self.frame_times_us[self.next]));
        } else {
            self.count += 1;
        }
        self.frame_times_us[self.next] = elapsed_us;
        self.schedule_lateness_us[self.next] = lateness_us;
        self.total_frame_time_us = self
            .total_frame_time_us
            .saturating_add(u64::from(elapsed_us));
        self.next = (self.next + 1) % STAGE_HEALTH_FRAME_WINDOW;
        self.missed_deadlines = self.missed_deadlines.saturating_add(skipped_program_ticks);
    }

    pub fn snapshot(&self, input: StageHealthPublishInput<'_>) -> StageHealthSnapshot {
        let mut sorted = self.frame_times_us;
        sorted[..self.count].sort_unstable();
        let mut sorted_lateness = self.schedule_lateness_us;
        sorted_lateness[..self.count].sort_unstable();
        let mean = if self.count == 0 {
            0.0
        } else {
            self.total_frame_time_us as f64 / self.count as f64
        };
        let fps = if mean > 0.0 {
            (1_000_000.0 / mean).clamp(0.0, 10_000.0) as f32
        } else {
            0.0
        };
        let layers = input
            .layers
            .iter()
            .take(STAGE_HEALTH_MAX_LAYERS)
            .map(|layer| LayerStageHealthSnapshot {
                layer_id: layer.layer_id.get().to_string(),
                name: bounded_text(layer.name),
                decoded_age_ms: layer
                    .decoded_age
                    .map(|age| age.as_millis().min(u128::from(24_u64 * 60 * 60 * 1_000)) as u64),
                pending_frames: layer.pending_frames.min(1_024),
                dropped_frames: layer.dropped_frames,
                status: bounded_text(layer.status),
                source_color: layer.source_color,
                source_display: layer.source_display,
                conversion_policy: layer.conversion_policy,
            })
            .collect();
        StageHealthSnapshot {
            build_identity: current_build_identity_snapshot(),
            fps,
            frame_time_p50_us: percentile(&sorted[..self.count], 50),
            frame_time_p95_us: percentile(&sorted[..self.count], 95),
            frame_time_p99_us: percentile(&sorted[..self.count], 99),
            frame_samples: self.count.min(usize::from(u16::MAX)) as u16,
            schedule_lateness_p50_us: percentile(&sorted_lateness[..self.count], 50),
            schedule_lateness_p95_us: percentile(&sorted_lateness[..self.count], 95),
            schedule_lateness_p99_us: percentile(&sorted_lateness[..self.count], 99),
            skipped_program_ticks: self.missed_deadlines,
            missed_deadlines: self.missed_deadlines,
            layers,
            layers_truncated: input.layers.len() > STAGE_HEALTH_MAX_LAYERS,
            output: StageOutputSnapshot {
                endpoint_id: input.output.endpoint_id.as_str().to_string(),
                identity: bounded_text(input.output.identity),
                width: input.output.width,
                height: input.output.height,
                refresh_millihz: input.output.refresh_millihz.min(1_000_000),
                active: input.output.active,
            },
            gpu: StageGpuSnapshot {
                adapter: bounded_text(input.gpu.adapter),
                backend: bounded_text(input.gpu.backend),
                device: bounded_text(input.gpu.device),
            },
            gpu_timing: input.gpu_timing.into(),
            action_timing: input.action_timing.into(),
            flight_recorder: StageFlightRecorderSnapshot::default(),
            decoder_retirement: StageDecoderRetirementSnapshot::default(),
            budgets: StageBudgetSetSnapshot {
                gpu: budget_snapshot(input.budgets.gpu),
                media: budget_snapshot(input.budgets.media),
                ntsc: budget_snapshot(input.budgets.ntsc),
                motion: budget_snapshot(input.budgets.motion),
            },
            tools: stage_tools_snapshot(input.tools),
        }
    }
}

fn current_build_identity_snapshot() -> crate::build_identity::BuildIdentitySnapshot {
    crate::build_identity::current().snapshot()
}

fn percentile(sorted: &[u32], percentile: usize) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (percentile.saturating_mul(sorted.len()).saturating_add(99)) / 100;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn budget_snapshot(input: StageBudgetInput<'_>) -> StageBudgetSnapshot {
    StageBudgetSnapshot {
        unit: bounded_text(input.unit),
        used: input.used,
        limit: input.limit,
        detail: bounded_text(input.detail),
    }
}

fn bounded_text(value: &str) -> String {
    if value.len() <= STAGE_HEALTH_MAX_TEXT_BYTES {
        return value.to_string();
    }
    let mut boundary = STAGE_HEALTH_MAX_TEXT_BYTES.saturating_sub(3);
    while boundary > 0 && !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    let mut bounded = value[..boundary].to_string();
    bounded.push_str("...");
    bounded
}

/// Paint only operator telemetry. The type signature cannot receive an
/// audience/composite texture, and the permit cannot be minted for one.
pub fn paint_editor_preview_health(
    ui: &mut egui::Ui,
    _permit: EditorPreviewPermit,
    snapshot: &StageHealthSnapshot,
) {
    egui::Frame::popup(ui.style()).show(ui, |ui| {
        ui.strong(format!(
            "Stage {:.1} fps  |  frame p50 {:.2} / p95 {:.2} / p99 {:.2} ms",
            snapshot.fps,
            f64::from(snapshot.frame_time_p50_us) / 1_000.0,
            f64::from(snapshot.frame_time_p95_us) / 1_000.0,
            f64::from(snapshot.frame_time_p99_us) / 1_000.0
        ));
        ui.label(format!(
            "Skipped ticks {}  |  lateness p50 {:.3} / p95 {:.3} / p99 {:.3} ms  |  {}x{} @ {:.3} Hz  |  {}",
            snapshot.skipped_program_ticks,
            f64::from(snapshot.schedule_lateness_p50_us) / 1_000.0,
            f64::from(snapshot.schedule_lateness_p95_us) / 1_000.0,
            f64::from(snapshot.schedule_lateness_p99_us) / 1_000.0,
            snapshot.output.width,
            snapshot.output.height,
            f64::from(snapshot.output.refresh_millihz) / 1_000.0,
            snapshot.output.identity
        ));
        if snapshot.gpu_timing.supported {
            ui.label(format!(
                "GPU p95 ms source {:.3} | creative {:.3} | temporal {:.3} | Mosh/VHS {:.3} | resolve {:.3} | submit {:.3}",
                f64::from(snapshot.gpu_timing.source_prepare.p95_us) / 1_000.0,
                f64::from(snapshot.gpu_timing.creative_composition.p95_us) / 1_000.0,
                f64::from(snapshot.gpu_timing.temporal_motion.p95_us) / 1_000.0,
                f64::from(snapshot.gpu_timing.mosh_vhs.p95_us) / 1_000.0,
                f64::from(snapshot.gpu_timing.audience_resolve.p95_us) / 1_000.0,
                f64::from(snapshot.gpu_timing.submission.p95_us) / 1_000.0,
            ));
        } else {
            ui.weak("GPU timestamps unsupported on this adapter/backend");
        }
        ui.label(format!(
            "Action p95 ingress→apply {:.3} ms | apply→submit {:.3} ms | sequence {} | generation {} | pending {}",
            f64::from(snapshot.action_timing.ingress_to_apply.p95_us) / 1_000.0,
            f64::from(snapshot.action_timing.apply_to_submit.p95_us) / 1_000.0,
            snapshot.action_timing.last_presented_sequence,
            snapshot.action_timing.last_submission_generation,
            snapshot.action_timing.pending,
        ));
        if snapshot.flight_recorder.enabled {
            ui.label(format!(
                "Private flight recorder {} s × {} | queued {} | pressure drops {}",
                snapshot.flight_recorder.rotation_seconds,
                snapshot.flight_recorder.retained_rotations,
                snapshot.flight_recorder.queued_events,
                snapshot.flight_recorder.dropped_full,
            ));
        } else {
            ui.weak("Private flight recorder unavailable");
        }
        let retire_age = snapshot
            .decoder_retirement
            .oldest_retirement_age_ms
            .map_or_else(|| "n/a".to_owned(), |age| format!("{age} ms"));
        let retire_identity = snapshot
            .decoder_retirement
            .oldest_worker_id
            .map_or_else(|| "none".to_owned(), |worker_id| {
                format!(
                    "worker {worker_id} source {} generation {}",
                    snapshot.decoder_retirement.oldest_source_fingerprint,
                    snapshot
                        .decoder_retirement
                        .oldest_source_generation
                        .unwrap_or_default(),
                )
            });
        let decoder_retirement_label = format!(
            "Decoder workers active/retiring/stuck {}/{}/{} | owned {}/{} | oldest {} ({}) | admission refusals {}",
            snapshot.decoder_retirement.active_workers,
            snapshot.decoder_retirement.retiring_workers,
            snapshot.decoder_retirement.stuck_workers,
            snapshot.decoder_retirement.owned_workers,
            snapshot.decoder_retirement.hard_cap,
            retire_age,
            retire_identity,
            snapshot.decoder_retirement.admission_refusals,
        );
        if matches!(
            snapshot.decoder_retirement.health,
            StageDecoderRetirementHealth::Stuck
        ) {
            ui.colored_label(egui::Color32::RED, decoder_retirement_label);
        } else {
            ui.label(decoder_retirement_label);
        }
        ui.label(format!(
            "GPU {} ({})  |  media {}  |  Mosh send {}  |  motion {}",
            snapshot.gpu.adapter,
            snapshot.gpu.backend,
            budget_label(&snapshot.budgets.media),
            budget_label(&snapshot.budgets.ntsc),
            budget_label(&snapshot.budgets.motion)
        ));
        for layer in &snapshot.layers {
            let age = layer
                .decoded_age_ms
                .map_or_else(|| "n/a".to_string(), |age| format!("{age} ms"));
            ui.label(format!(
                "{}  decoded {age}  pending {}  drops {}  color {:?}/{:?}/{:?}  convert {:?}  {}",
                layer.name,
                layer.pending_frames,
                layer.dropped_frames,
                layer.source_color.matrix.value,
                layer.source_color.range.value,
                layer.source_color.transfer.value,
                layer.conversion_policy.kind,
                layer.status
            ));
        }
    });
}

fn budget_label(budget: &StageBudgetSnapshot) -> String {
    match (budget.used, budget.limit) {
        (Some(used), Some(limit)) => format!("{used}/{limit} {}", budget.unit),
        _ if !budget.detail.is_empty() => budget.detail.clone(),
        _ => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(id: u64) -> LayerStageHealthInput<'static> {
        LayerStageHealthInput {
            layer_id: StableLayerId::new(id).unwrap(),
            name: "plate",
            decoded_age: Some(Duration::from_millis(12)),
            pending_frames: 1,
            dropped_frames: 3,
            status: "healthy",
            source_color: Default::default(),
            source_display: Default::default(),
            conversion_policy: Default::default(),
        }
    }

    fn publish<'a>(
        layers: &'a [LayerStageHealthInput<'a>],
        tools: &'a StageToolState,
        endpoint: &'a OutputEndpointId,
    ) -> StageHealthPublishInput<'a> {
        StageHealthPublishInput {
            layers,
            output: StageOutputInput {
                endpoint_id: endpoint,
                identity: "Projector",
                width: 1_920,
                height: 1_080,
                refresh_millihz: 59_940,
                active: true,
            },
            gpu: StageGpuInput {
                adapter: "Adapter",
                backend: "Vulkan",
                device: "Device",
            },
            gpu_timing: crate::renderer::gpu_timing::GpuTimingSnapshot::default(),
            action_timing: crate::action_correlation::ActionTimingSnapshot::default(),
            budgets: StageBudgetSetInput::default(),
            tools,
        }
    }

    #[test]
    fn fixed_window_reports_exact_percentiles_fps_lateness_and_skips() {
        let mut monitor = StageHealthMonitor::default();
        for millis in 1..=100 {
            monitor.observe_frame(
                Duration::from_millis(millis),
                Duration::from_micros(millis),
                u64::from(millis % 25 == 0),
            );
        }
        let tools = StageToolState::default();
        let endpoint = OutputEndpointId::legacy();
        let layers = [layer(1)];
        let snapshot = monitor.snapshot(publish(&layers, &tools, &endpoint));
        assert_eq!(snapshot.frame_time_p50_us, 50_000);
        assert_eq!(snapshot.frame_time_p95_us, 95_000);
        assert_eq!(snapshot.frame_time_p99_us, 99_000);
        assert!((snapshot.fps - (1_000.0 / 50.5)).abs() < 0.01);
        assert_eq!(snapshot.schedule_lateness_p50_us, 50);
        assert_eq!(snapshot.schedule_lateness_p95_us, 95);
        assert_eq!(snapshot.schedule_lateness_p99_us, 99);
        assert_eq!(snapshot.skipped_program_ticks, 4);
        assert_eq!(snapshot.missed_deadlines, 4);
    }

    #[test]
    fn timing_spine_fields_are_additive_for_legacy_stage_snapshots() {
        let mut legacy = serde_json::to_value(StageHealthSnapshot::default()).unwrap();
        let object = legacy.as_object_mut().unwrap();
        object.remove("gpu_timing");
        object.remove("action_timing");
        object.remove("decoder_retirement");
        let restored: StageHealthSnapshot = serde_json::from_value(legacy).unwrap();
        assert!(!restored.gpu_timing.supported);
        assert_eq!(restored.action_timing.ingress_to_apply.samples, 0);
        assert_eq!(restored.action_timing.apply_to_submit.samples, 0);
        assert_eq!(
            restored.decoder_retirement.health,
            StageDecoderRetirementHealth::Healthy
        );
    }

    #[test]
    fn source_descriptor_fields_are_additive_for_legacy_layer_health() {
        let legacy = serde_json::json!({
            "layer_id": "1",
            "name": "legacy.mov",
            "pending_frames": 0,
            "dropped_frames": 0,
            "status": "ready"
        });
        let restored: LayerStageHealthSnapshot = serde_json::from_value(legacy).unwrap();
        assert_eq!(restored.source_color, Default::default());
        assert_eq!(restored.source_display, Default::default());
        assert_eq!(restored.conversion_policy, Default::default());
    }

    #[test]
    fn rational_pacer_honors_the_exact_thirty_hertz_boundary() {
        let origin = Instant::now();
        let mut pacer = RationalFramePacer::new(origin, 30);

        assert_eq!(
            pacer.accept_due(origin + Duration::from_nanos(33_333_332)),
            None
        );
        let tick = pacer
            .accept_due(origin + Duration::from_nanos(33_333_333))
            .expect("the first rational deadline is due");
        assert_eq!(tick.schedule_lateness, Duration::ZERO);
        assert_eq!(tick.skipped_program_ticks, 0);
        assert_eq!(
            pacer.next_deadline(),
            origin + Duration::from_nanos(66_666_666)
        );
    }

    #[test]
    fn rational_pacer_counts_only_whole_skipped_program_ticks() {
        let origin = Instant::now();
        let mut slightly_late = RationalFramePacer::new(origin, 30);
        let tick = slightly_late
            .accept_due(origin + Duration::from_nanos(33_340_333))
            .expect("a slightly late first tick remains due");
        assert_eq!(tick.schedule_lateness, Duration::from_micros(7));
        assert_eq!(tick.skipped_program_ticks, 0);

        let mut stalled = RationalFramePacer::new(origin, 30);
        let tick = stalled
            .accept_due(origin + Duration::from_millis(100))
            .expect("the newest tick after a stall is due");
        assert_eq!(tick.schedule_lateness, Duration::ZERO);
        assert_eq!(tick.skipped_program_ticks, 2);
    }

    #[test]
    fn rational_pacer_does_not_accumulate_integer_period_drift() {
        let origin = Instant::now();
        let mut pacer = RationalFramePacer::new(origin, 30);
        for ordinal in 1_u64..=300 {
            let deadline_nanos = ordinal.saturating_mul(1_000_000_000) / 30;
            let tick = pacer
                .accept_due(origin + Duration::from_nanos(deadline_nanos))
                .expect("every exact rational deadline is accepted");
            assert_eq!(tick.schedule_lateness, Duration::ZERO);
            assert_eq!(tick.skipped_program_ticks, 0);
        }
    }

    #[test]
    fn hostile_layer_facts_are_bounded_without_unbounded_snapshot_growth() {
        let long = "x".repeat(STAGE_HEALTH_MAX_TEXT_BYTES * 4);
        let layers = (1..=(STAGE_HEALTH_MAX_LAYERS as u64 + 8))
            .map(|id| LayerStageHealthInput {
                layer_id: StableLayerId::new(id).unwrap(),
                name: &long,
                decoded_age: Some(Duration::MAX),
                pending_frames: u32::MAX,
                dropped_frames: u64::MAX,
                status: &long,
                source_color: Default::default(),
                source_display: Default::default(),
                conversion_policy: Default::default(),
            })
            .collect::<Vec<_>>();
        let tools = StageToolState::default();
        let endpoint = OutputEndpointId::legacy();
        let snapshot = StageHealthMonitor::default().snapshot(publish(&layers, &tools, &endpoint));
        assert_eq!(snapshot.layers.len(), STAGE_HEALTH_MAX_LAYERS);
        assert!(snapshot.layers_truncated);
        assert!(snapshot.layers[0].name.len() <= STAGE_HEALTH_MAX_TEXT_BYTES);
        assert_eq!(snapshot.layers[0].pending_frames, 1_024);
        assert_eq!(snapshot.layers[0].decoded_age_ms, Some(86_400_000));
    }

    #[test]
    fn health_hud_cannot_receive_any_non_editor_surface() {
        let mut tools = StageToolState::default();
        tools.set_health_hud(true);
        assert!(editor_preview_permit(&tools, &StageSurface::EditorPreview).is_some());
        for surface in [
            StageSurface::Composite,
            StageSurface::Audience,
            StageSurface::Spout,
            StageSurface::Record,
            StageSurface::Export,
            StageSurface::PhysicalOutput(OutputEndpointId::legacy()),
        ] {
            assert!(editor_preview_permit(&tools, &surface).is_none());
        }
    }

    #[test]
    fn calibration_is_exactly_scoped_to_one_selected_physical_endpoint() {
        let endpoint = OutputEndpointId::parse("projector-a").unwrap();
        let other = OutputEndpointId::parse("projector-b").unwrap();
        let mut tools = StageToolState::default();
        tools
            .set_test_card(TestCardMode::SmpteBars, Some(endpoint.clone()))
            .unwrap();
        tools
            .set_output_identification(true, Some(endpoint.clone()))
            .unwrap();
        let selected = tools.decision_for(&StageSurface::PhysicalOutput(endpoint));
        assert!(selected.substitute_with_test_card);
        assert!(selected.overlay_output_identification);
        assert_eq!(
            tools.decision_for(&StageSurface::PhysicalOutput(other)),
            StageCalibrationDecision::default()
        );
        for surface in [
            StageSurface::EditorPreview,
            StageSurface::Composite,
            StageSurface::Audience,
            StageSurface::Spout,
            StageSurface::Record,
            StageSurface::Export,
        ] {
            assert_eq!(
                tools.decision_for(&surface),
                StageCalibrationDecision::default()
            );
        }
    }

    #[test]
    fn endpoint_ids_reject_hostile_or_ambiguous_ingress() {
        assert!(OutputEndpointId::parse("").is_err());
        assert!(OutputEndpointId::parse("../projector").is_err());
        assert!(OutputEndpointId::parse("projector\nother").is_err());
        assert!(OutputEndpointId::parse("x".repeat(129)).is_err());
        assert!(OutputEndpointId::parse("display-1_A").is_ok());
        let mut tools = StageToolState::default();
        assert!(tools.set_test_card(TestCardMode::Grid, None).is_err());
        assert!(tools.set_output_identification(true, None).is_err());
    }
}
