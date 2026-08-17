//! Preview-only stage health and explicitly endpoint-scoped calibration tools.
//!
//! This module has no access to renderer textures or composition executors.
//! The health painter accepts only an `egui::Ui` plus a permit that can be
//! minted only for the editor preview. Test cards and output identification
//! are pure decisions keyed to one exact physical output endpoint.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::image_routing::StableLayerId;
#[cfg(test)]
use crate::stage_map::StageCalibrationDecision;
pub use crate::stage_map::{OutputEndpointId, StageSurface, StageToolState, TestCardMode};

pub const STAGE_HEALTH_FRAME_WINDOW: usize = 512;
pub const STAGE_HEALTH_MAX_LAYERS: usize = 256;
pub const STAGE_HEALTH_MAX_TEXT_BYTES: usize = 256;

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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageHealthSnapshot {
    pub fps: f32,
    pub frame_time_p50_us: u32,
    pub frame_time_p95_us: u32,
    pub frame_time_p99_us: u32,
    pub frame_samples: u16,
    pub missed_deadlines: u64,
    pub layers: Vec<LayerStageHealthSnapshot>,
    pub layers_truncated: bool,
    pub output: StageOutputSnapshot,
    pub gpu: StageGpuSnapshot,
    pub budgets: StageBudgetSetSnapshot,
    pub tools: StageToolsSnapshot,
}

impl Default for StageHealthSnapshot {
    fn default() -> Self {
        Self {
            fps: 0.0,
            frame_time_p50_us: 0,
            frame_time_p95_us: 0,
            frame_time_p99_us: 0,
            frame_samples: 0,
            missed_deadlines: 0,
            layers: Vec::new(),
            layers_truncated: false,
            output: StageOutputSnapshot::default(),
            gpu: StageGpuSnapshot::default(),
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
    pub budgets: StageBudgetSetInput<'a>,
    pub tools: &'a StageToolState,
}

pub struct StageHealthMonitor {
    frame_times_us: [u32; STAGE_HEALTH_FRAME_WINDOW],
    next: usize,
    count: usize,
    total_frame_time_us: u64,
    missed_deadlines: u64,
}

impl Default for StageHealthMonitor {
    fn default() -> Self {
        Self {
            frame_times_us: [0; STAGE_HEALTH_FRAME_WINDOW],
            next: 0,
            count: 0,
            total_frame_time_us: 0,
            missed_deadlines: 0,
        }
    }
}

impl StageHealthMonitor {
    /// Fixed-storage sample insertion. No allocation occurs on the frame path.
    pub fn observe_frame(&mut self, elapsed: Duration, deadline: Duration) {
        let elapsed_us = elapsed.as_micros().min(u128::from(u32::MAX)) as u32;
        if self.count == STAGE_HEALTH_FRAME_WINDOW {
            self.total_frame_time_us = self
                .total_frame_time_us
                .saturating_sub(u64::from(self.frame_times_us[self.next]));
        } else {
            self.count += 1;
        }
        self.frame_times_us[self.next] = elapsed_us;
        self.total_frame_time_us = self
            .total_frame_time_us
            .saturating_add(u64::from(elapsed_us));
        self.next = (self.next + 1) % STAGE_HEALTH_FRAME_WINDOW;
        if elapsed > deadline {
            self.missed_deadlines = self.missed_deadlines.saturating_add(1);
        }
    }

    pub fn snapshot(&self, input: StageHealthPublishInput<'_>) -> StageHealthSnapshot {
        let mut sorted = self.frame_times_us;
        sorted[..self.count].sort_unstable();
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
            })
            .collect();
        StageHealthSnapshot {
            fps,
            frame_time_p50_us: percentile(&sorted[..self.count], 50),
            frame_time_p95_us: percentile(&sorted[..self.count], 95),
            frame_time_p99_us: percentile(&sorted[..self.count], 99),
            frame_samples: self.count.min(usize::from(u16::MAX)) as u16,
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
            "Missed deadlines {}  |  {}x{} @ {:.3} Hz  |  {}",
            snapshot.missed_deadlines,
            snapshot.output.width,
            snapshot.output.height,
            f64::from(snapshot.output.refresh_millihz) / 1_000.0,
            snapshot.output.identity
        ));
        ui.label(format!(
            "GPU {} ({})  |  media {}  |  NTSC {}  |  motion {}",
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
                "{}  decoded {age}  pending {}  drops {}  {}",
                layer.name, layer.pending_frames, layer.dropped_frames, layer.status
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
            budgets: StageBudgetSetInput::default(),
            tools,
        }
    }

    #[test]
    fn fixed_window_reports_exact_percentiles_fps_and_deadlines() {
        let mut monitor = StageHealthMonitor::default();
        for millis in 1..=100 {
            monitor.observe_frame(Duration::from_millis(millis), Duration::from_millis(16));
        }
        let tools = StageToolState::default();
        let endpoint = OutputEndpointId::legacy();
        let layers = [layer(1)];
        let snapshot = monitor.snapshot(publish(&layers, &tools, &endpoint));
        assert_eq!(snapshot.frame_time_p50_us, 50_000);
        assert_eq!(snapshot.frame_time_p95_us, 95_000);
        assert_eq!(snapshot.frame_time_p99_us, 99_000);
        assert!((snapshot.fps - (1_000.0 / 50.5)).abs() < 0.01);
        assert_eq!(snapshot.missed_deadlines, 84);
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
