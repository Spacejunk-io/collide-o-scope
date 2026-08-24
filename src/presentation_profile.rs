//! Evidence-gated P6 presentation-profile planner.
//!
//! `Stable` is the exact v1.6.0 policy. `LowLatency` cannot be admitted by a
//! label or a wgpu hint alone: a host-keyed physical calibration must satisfy
//! the optical, present-stability, deterministic-state, and p99 gates.

use serde::{Deserialize, Serialize};

pub const REFERENCE_HZ: u32 = 30;
pub const LOW_LATENCY_RENDER_HZ_CAP: u32 = 60;
pub const STABLE_MAXIMUM_FRAME_LATENCY: u32 = 2;
pub const LOW_LATENCY_MAXIMUM_FRAME_LATENCY: u32 = 1;
pub const MINIMUM_TEN_MINUTE_60HZ_PRESENTS: u32 = 36_000;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationProfile {
    #[default]
    Stable,
    LowLatency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentMode {
    Fifo,
    Mailbox,
    Immediate,
    FifoRelaxed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfacePresentationCapabilities {
    pub fifo: bool,
    pub maximum_frame_latency_1: bool,
    pub maximum_frame_latency_2: bool,
    pub refresh_millihz: u32,
    pub vrr_active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentationHostIdentity {
    pub adapter_digest: [u8; 32],
    pub driver_digest: [u8; 32],
    pub surface_digest: [u8; 32],
    pub build_digest: [u8; 32],
    pub patch_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentationCalibration {
    pub identity: PresentationHostIdentity,
    pub optical_trials: u32,
    pub stable_action_to_photon_p95_micros: u64,
    pub low_latency_action_to_photon_p95_micros: u64,
    pub stable_dropped_presents: u32,
    pub low_latency_dropped_presents: u32,
    pub stable_late_presents: u32,
    pub low_latency_late_presents: u32,
    pub low_latency_tearing_observations: u32,
    pub stable_cadence_instability_events: u32,
    pub low_latency_cadence_instability_events: u32,
    pub low_latency_present_samples: u32,
    pub cpu_frame_p99_micros: u64,
    pub cpu_budget_micros: u64,
    pub gpu_frame_p99_micros: u64,
    pub gpu_budget_micros: u64,
    pub accepted_reference_state_hash_equal: bool,
    pub offline_export_hash_equal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresentationPlan {
    pub profile: PresentationProfile,
    pub reference_hz: u32,
    pub maximum_render_hz: u32,
    pub present_mode: PresentMode,
    pub maximum_frame_latency: u32,
    pub temporal_state_on_reference_ticks_only: bool,
    pub high_priority_ingress_wakes: bool,
    pub host_local_only: bool,
}

impl PresentationPlan {
    pub const fn stable() -> Self {
        Self {
            profile: PresentationProfile::Stable,
            reference_hz: REFERENCE_HZ,
            maximum_render_hz: REFERENCE_HZ,
            present_mode: PresentMode::Fifo,
            maximum_frame_latency: STABLE_MAXIMUM_FRAME_LATENCY,
            temporal_state_on_reference_ticks_only: true,
            high_priority_ingress_wakes: false,
            host_local_only: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationRefusal {
    FifoUnsupported,
    StableLatencyHintUnsupported,
    LowLatencyHintUnsupported,
    RefreshBelowCandidate,
    CalibrationMissing,
    CalibrationIdentityMismatch,
    TooFewOpticalTrials,
    PhysicalImprovementBelow25Percent,
    TenMinuteRunMissing,
    DroppedPresentRegression,
    LatePresentRegression,
    TearingObserved,
    CadenceRegression,
    CpuP99Budget,
    GpuP99Budget,
    ReferenceStateMismatch,
    OfflineExportMismatch,
}

pub fn admit_presentation_profile(
    requested: PresentationProfile,
    capabilities: SurfacePresentationCapabilities,
    host: PresentationHostIdentity,
    calibration: Option<PresentationCalibration>,
) -> Result<PresentationPlan, PresentationRefusal> {
    if !capabilities.fifo {
        return Err(PresentationRefusal::FifoUnsupported);
    }
    if requested == PresentationProfile::Stable {
        if !capabilities.maximum_frame_latency_2 {
            return Err(PresentationRefusal::StableLatencyHintUnsupported);
        }
        return Ok(PresentationPlan::stable());
    }
    if !capabilities.maximum_frame_latency_1 {
        return Err(PresentationRefusal::LowLatencyHintUnsupported);
    }
    if capabilities.refresh_millihz < 59_000 {
        return Err(PresentationRefusal::RefreshBelowCandidate);
    }
    let calibration = calibration.ok_or(PresentationRefusal::CalibrationMissing)?;
    if calibration.identity != host {
        return Err(PresentationRefusal::CalibrationIdentityMismatch);
    }
    if calibration.optical_trials < 30 {
        return Err(PresentationRefusal::TooFewOpticalTrials);
    }
    // low <= 75% of stable, written without floating point.
    if u128::from(calibration.low_latency_action_to_photon_p95_micros) * 4
        > u128::from(calibration.stable_action_to_photon_p95_micros) * 3
    {
        return Err(PresentationRefusal::PhysicalImprovementBelow25Percent);
    }
    if calibration.low_latency_present_samples < MINIMUM_TEN_MINUTE_60HZ_PRESENTS {
        return Err(PresentationRefusal::TenMinuteRunMissing);
    }
    if calibration.low_latency_dropped_presents > calibration.stable_dropped_presents {
        return Err(PresentationRefusal::DroppedPresentRegression);
    }
    if calibration.low_latency_late_presents > calibration.stable_late_presents {
        return Err(PresentationRefusal::LatePresentRegression);
    }
    if calibration.low_latency_tearing_observations != 0 {
        return Err(PresentationRefusal::TearingObserved);
    }
    if calibration.low_latency_cadence_instability_events
        > calibration.stable_cadence_instability_events
    {
        return Err(PresentationRefusal::CadenceRegression);
    }
    if calibration.cpu_frame_p99_micros > calibration.cpu_budget_micros {
        return Err(PresentationRefusal::CpuP99Budget);
    }
    if calibration.gpu_frame_p99_micros > calibration.gpu_budget_micros {
        return Err(PresentationRefusal::GpuP99Budget);
    }
    if !calibration.accepted_reference_state_hash_equal {
        return Err(PresentationRefusal::ReferenceStateMismatch);
    }
    if !calibration.offline_export_hash_equal {
        return Err(PresentationRefusal::OfflineExportMismatch);
    }
    Ok(PresentationPlan {
        profile: PresentationProfile::LowLatency,
        reference_hz: REFERENCE_HZ,
        maximum_render_hz: LOW_LATENCY_RENDER_HZ_CAP,
        present_mode: PresentMode::Fifo,
        maximum_frame_latency: LOW_LATENCY_MAXIMUM_FRAME_LATENCY,
        temporal_state_on_reference_ticks_only: true,
        high_priority_ingress_wakes: true,
        host_local_only: true,
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PresentationWake {
    pub advance_reference_state: bool,
    pub render: bool,
    pub applied_non_temporal_revision: Option<u64>,
}

/// Pure scheduler prototype. It proves that a redraw can carry a newly
/// accepted non-temporal revision without advancing the 30 Hz media/history
/// address, and that ingress storms cannot exceed the bounded render cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentationScheduler {
    plan: PresentationPlan,
    next_reference_ns: u64,
    next_render_ns: u64,
    newest_non_temporal_revision: Option<u64>,
}

impl PresentationScheduler {
    pub const fn new(plan: PresentationPlan, start_ns: u64) -> Self {
        Self {
            plan,
            next_reference_ns: start_ns,
            next_render_ns: start_ns,
            newest_non_temporal_revision: None,
        }
    }

    pub fn offer_high_priority_non_temporal_revision(&mut self, revision: u64) {
        if self.plan.high_priority_ingress_wakes {
            self.newest_non_temporal_revision = Some(
                self.newest_non_temporal_revision
                    .map_or(revision, |prior| prior.max(revision)),
            );
        }
    }

    pub fn observe(&mut self, now_ns: u64) -> PresentationWake {
        let reference_due = now_ns >= self.next_reference_ns;
        if reference_due {
            self.next_reference_ns = advance_deadline(
                self.next_reference_ns,
                now_ns,
                period_ns(self.plan.reference_hz),
            );
        }
        let render_due = now_ns >= self.next_render_ns
            && (reference_due
                || self.newest_non_temporal_revision.is_some()
                || self.plan.maximum_render_hz > self.plan.reference_hz);
        let revision = render_due
            .then(|| self.newest_non_temporal_revision.take())
            .flatten();
        if render_due {
            self.next_render_ns = advance_deadline(
                self.next_render_ns,
                now_ns,
                period_ns(self.plan.maximum_render_hz),
            );
        }
        PresentationWake {
            advance_reference_state: reference_due,
            render: render_due,
            applied_non_temporal_revision: revision,
        }
    }
}

const fn period_ns(hz: u32) -> u64 {
    1_000_000_000_u64.div_ceil(hz as u64)
}

fn advance_deadline(mut deadline: u64, now: u64, period: u64) -> u64 {
    while deadline <= now {
        deadline = deadline.saturating_add(period);
        if deadline == u64::MAX {
            break;
        }
    }
    deadline
}

pub const fn offline_export_reference_hz(_host_profile: PresentationProfile) -> u32 {
    REFERENCE_HZ
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(byte: u8) -> PresentationHostIdentity {
        PresentationHostIdentity {
            adapter_digest: [byte; 32],
            driver_digest: [byte; 32],
            surface_digest: [byte; 32],
            build_digest: [byte; 32],
            patch_digest: [byte; 32],
        }
    }

    fn capabilities() -> SurfacePresentationCapabilities {
        SurfacePresentationCapabilities {
            fifo: true,
            maximum_frame_latency_1: true,
            maximum_frame_latency_2: true,
            refresh_millihz: 60_000,
            vrr_active: false,
        }
    }

    fn passing_calibration() -> PresentationCalibration {
        PresentationCalibration {
            identity: host(1),
            optical_trials: 30,
            stable_action_to_photon_p95_micros: 40_000,
            low_latency_action_to_photon_p95_micros: 30_000,
            stable_dropped_presents: 0,
            low_latency_dropped_presents: 0,
            stable_late_presents: 0,
            low_latency_late_presents: 0,
            low_latency_tearing_observations: 0,
            stable_cadence_instability_events: 0,
            low_latency_cadence_instability_events: 0,
            low_latency_present_samples: MINIMUM_TEN_MINUTE_60HZ_PRESENTS,
            cpu_frame_p99_micros: 8_000,
            cpu_budget_micros: 10_000,
            gpu_frame_p99_micros: 12_000,
            gpu_budget_micros: 15_000,
            accepted_reference_state_hash_equal: true,
            offline_export_hash_equal: true,
        }
    }

    #[test]
    fn stable_is_the_exact_frozen_policy_and_export_ignores_host_profile() {
        assert_eq!(
            admit_presentation_profile(PresentationProfile::Stable, capabilities(), host(1), None),
            Ok(PresentationPlan::stable())
        );
        assert_eq!(
            offline_export_reference_hz(PresentationProfile::LowLatency),
            REFERENCE_HZ
        );
    }

    #[test]
    fn low_latency_is_unavailable_without_exact_host_physical_evidence() {
        assert_eq!(
            admit_presentation_profile(
                PresentationProfile::LowLatency,
                capabilities(),
                host(1),
                None
            ),
            Err(PresentationRefusal::CalibrationMissing)
        );
        assert_eq!(
            admit_presentation_profile(
                PresentationProfile::LowLatency,
                capabilities(),
                host(2),
                Some(passing_calibration())
            ),
            Err(PresentationRefusal::CalibrationIdentityMismatch)
        );
    }

    #[test]
    fn every_keep_gate_is_machine_checked() {
        let mut calibration = passing_calibration();
        assert!(admit_presentation_profile(
            PresentationProfile::LowLatency,
            capabilities(),
            host(1),
            Some(calibration)
        )
        .is_ok());
        calibration.low_latency_action_to_photon_p95_micros = 30_001;
        assert_eq!(
            admit_presentation_profile(
                PresentationProfile::LowLatency,
                capabilities(),
                host(1),
                Some(calibration)
            ),
            Err(PresentationRefusal::PhysicalImprovementBelow25Percent)
        );
        calibration = passing_calibration();
        calibration.low_latency_present_samples -= 1;
        assert_eq!(
            admit_presentation_profile(
                PresentationProfile::LowLatency,
                capabilities(),
                host(1),
                Some(calibration)
            ),
            Err(PresentationRefusal::TenMinuteRunMissing)
        );
        calibration = passing_calibration();
        calibration.accepted_reference_state_hash_equal = false;
        assert_eq!(
            admit_presentation_profile(
                PresentationProfile::LowLatency,
                capabilities(),
                host(1),
                Some(calibration)
            ),
            Err(PresentationRefusal::ReferenceStateMismatch)
        );
    }

    #[test]
    fn ingress_storm_coalesces_and_never_invents_reference_ticks() {
        let plan = admit_presentation_profile(
            PresentationProfile::LowLatency,
            capabilities(),
            host(1),
            Some(passing_calibration()),
        )
        .unwrap();
        let mut scheduler = PresentationScheduler::new(plan, 0);
        let first = scheduler.observe(0);
        assert!(first.advance_reference_state && first.render);
        for revision in 1..=10_000 {
            scheduler.offer_high_priority_non_temporal_revision(revision);
        }
        assert_eq!(scheduler.observe(1_000_000), PresentationWake::default());
        let ingress_redraw = scheduler.observe(period_ns(60));
        assert!(ingress_redraw.render);
        assert!(!ingress_redraw.advance_reference_state);
        assert_eq!(ingress_redraw.applied_non_temporal_revision, Some(10_000));
        let reference = scheduler.observe(period_ns(30));
        assert!(reference.render && reference.advance_reference_state);
    }
}
