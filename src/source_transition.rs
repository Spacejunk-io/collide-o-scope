//! Bounded, deterministic planning kernel for prepared-source transitions.
//!
//! This module is deliberately GPU- and decoder-free.  It freezes the
//! additive authored law, all-or-none Scene admission, clock semantics, and
//! resource bounds before the renderer is allowed to adopt a dissolve path.

#![allow(
    dead_code,
    reason = "R1 is an evidence-gated planner/reference prototype, not a live renderer path"
)]

use serde::{Deserialize, Serialize};

pub const MAX_SIMULTANEOUS_DISSOLVES: usize = 2;
pub const MAX_TRANSITION_REFERENCE_TICKS: u32 = 3_600;
pub const MAX_TRANSITION_BEAT_NUMERATOR: u32 = 1_024;
pub const MAX_TRANSITION_BEAT_DENOMINATOR: u32 = 1_024;
pub const DEFAULT_TRANSITION_SCRATCH_CAP_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparedTransitionKind {
    #[default]
    Cut,
    Dissolve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "clock", content = "value")]
pub enum PreparedTransitionDuration {
    ReferenceTicks(u32),
    RationalBeats { numerator: u32, denominator: u32 },
}

impl Default for PreparedTransitionDuration {
    fn default() -> Self {
        Self::ReferenceTicks(0)
    }
}

impl PreparedTransitionDuration {
    pub const fn is_zero(self) -> bool {
        match self {
            Self::ReferenceTicks(ticks) => ticks == 0,
            Self::RationalBeats { numerator, .. } => numerator == 0,
        }
    }

    pub const fn validate(self) -> Result<(), TransitionConfigError> {
        match self {
            Self::ReferenceTicks(ticks) if ticks <= MAX_TRANSITION_REFERENCE_TICKS => Ok(()),
            Self::ReferenceTicks(_) => Err(TransitionConfigError::DurationCap),
            Self::RationalBeats {
                numerator,
                denominator,
            } if numerator <= MAX_TRANSITION_BEAT_NUMERATOR
                && denominator > 0
                && denominator <= MAX_TRANSITION_BEAT_DENOMINATOR =>
            {
                Ok(())
            }
            Self::RationalBeats { denominator: 0, .. } => {
                Err(TransitionConfigError::ZeroBeatDenominator)
            }
            Self::RationalBeats { .. } => Err(TransitionConfigError::DurationCap),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionInterruptionPolicy {
    /// A second activation is refused until the first transition completes.
    /// This is the only initially admitted law; future policies are additive.
    #[default]
    Refuse,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedTransition {
    #[serde(default, skip_serializing_if = "is_cut_kind")]
    pub kind: PreparedTransitionKind,
    #[serde(default, skip_serializing_if = "duration_is_zero")]
    pub duration: PreparedTransitionDuration,
    #[serde(default, skip_serializing_if = "is_refuse_policy")]
    pub interruption: TransitionInterruptionPolicy,
}

const fn is_cut_kind(kind: &PreparedTransitionKind) -> bool {
    matches!(kind, PreparedTransitionKind::Cut)
}

const fn is_refuse_policy(policy: &TransitionInterruptionPolicy) -> bool {
    matches!(policy, TransitionInterruptionPolicy::Refuse)
}

const fn duration_is_zero(duration: &PreparedTransitionDuration) -> bool {
    duration.is_zero()
}

impl PreparedTransition {
    pub const fn exact_cut() -> Self {
        Self {
            kind: PreparedTransitionKind::Cut,
            duration: PreparedTransitionDuration::ReferenceTicks(0),
            interruption: TransitionInterruptionPolicy::Refuse,
        }
    }

    /// Absent, explicit Cut, and zero-duration Dissolve all use the frozen
    /// legacy commit path and allocate no transition resource.
    pub const fn is_exact_cut(self) -> bool {
        matches!(self.kind, PreparedTransitionKind::Cut) || self.duration.is_zero()
    }

    pub const fn validate(self) -> Result<(), TransitionConfigError> {
        self.duration.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionConfigError {
    ZeroBeatDenominator,
    DurationCap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalBeatPosition {
    pub numerator: u64,
    pub denominator: u32,
}

impl RationalBeatPosition {
    pub const fn new(numerator: u64, denominator: u32) -> Option<Self> {
        if denominator == 0 {
            None
        } else {
            Some(Self {
                numerator,
                denominator,
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionClockObservation {
    pub accepted_reference_tick: u64,
    pub accepted_beat: RationalBeatPosition,
    pub program_paused: bool,
    pub media_frozen: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionPhase {
    numerator: u128,
    denominator: u128,
}

impl TransitionPhase {
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    pub const fn numerator(self) -> u128 {
        self.numerator
    }

    pub const fn denominator(self) -> u128 {
        self.denominator
    }

    pub fn as_f32(self) -> f32 {
        (self.numerator as f64 / self.denominator as f64) as f32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionClock {
    duration: PreparedTransitionDuration,
    elapsed_ticks: u64,
    elapsed_beat_numerator: u128,
    elapsed_beat_denominator: u128,
    last: TransitionClockObservation,
}

impl TransitionClock {
    pub fn start(
        transition: PreparedTransition,
        observation: TransitionClockObservation,
    ) -> Result<Self, TransitionConfigError> {
        transition.validate()?;
        Ok(Self {
            duration: transition.duration,
            elapsed_ticks: 0,
            elapsed_beat_numerator: 0,
            elapsed_beat_denominator: 1,
            last: observation,
        })
    }

    /// Pause and Media Freeze both hold phase.  A backward reference/beat
    /// observation reanchors and contributes no negative time.  Blackout is
    /// intentionally absent: it changes publication, not authored time.
    pub fn observe(&mut self, observation: TransitionClockObservation) -> TransitionPhase {
        let held = observation.program_paused || observation.media_frozen;
        if !held {
            self.elapsed_ticks = self.elapsed_ticks.saturating_add(
                observation
                    .accepted_reference_tick
                    .saturating_sub(self.last.accepted_reference_tick),
            );
            if let Some((numerator, denominator)) =
                positive_rational_delta(self.last.accepted_beat, observation.accepted_beat)
            {
                let left = self.elapsed_beat_numerator.saturating_mul(denominator);
                let right = numerator.saturating_mul(self.elapsed_beat_denominator);
                self.elapsed_beat_numerator = left.saturating_add(right);
                self.elapsed_beat_denominator = self
                    .elapsed_beat_denominator
                    .saturating_mul(denominator)
                    .max(1);
            }
        }
        self.last = observation;
        self.phase()
    }

    pub fn phase(self) -> TransitionPhase {
        match self.duration {
            PreparedTransitionDuration::ReferenceTicks(duration) => {
                bounded_phase(u128::from(self.elapsed_ticks), u128::from(duration))
            }
            PreparedTransitionDuration::RationalBeats {
                numerator,
                denominator,
            } => bounded_phase(
                self.elapsed_beat_numerator
                    .saturating_mul(u128::from(denominator)),
                self.elapsed_beat_denominator
                    .saturating_mul(u128::from(numerator)),
            ),
        }
    }
}

fn positive_rational_delta(
    before: RationalBeatPosition,
    after: RationalBeatPosition,
) -> Option<(u128, u128)> {
    let denominator = u128::from(before.denominator).saturating_mul(u128::from(after.denominator));
    let before_scaled = u128::from(before.numerator).saturating_mul(u128::from(after.denominator));
    let after_scaled = u128::from(after.numerator).saturating_mul(u128::from(before.denominator));
    (after_scaled > before_scaled).then_some((after_scaled - before_scaled, denominator))
}

fn bounded_phase(numerator: u128, denominator: u128) -> TransitionPhase {
    if denominator == 0 || numerator >= denominator {
        TransitionPhase::ONE
    } else if numerator == 0 {
        TransitionPhase::ZERO
    } else {
        TransitionPhase {
            numerator,
            denominator,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceUvMapping {
    /// Row-major 3x3 transform from output normalized UV to this source's UV.
    pub output_to_source: [i64; 9],
    /// Fixed-point denominator shared by the matrix.  It cannot be zero.
    pub denominator: u32,
}

impl SourceUvMapping {
    pub const IDENTITY: Self = Self {
        output_to_source: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        denominator: 1,
    };

    pub const fn is_valid(self) -> bool {
        self.denominator != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionActivationRequest {
    pub stable_layer_id: u64,
    pub expected_outgoing_generation: u64,
    pub observed_outgoing_generation: u64,
    pub incoming_generation: u64,
    pub incoming_ready: bool,
    pub outgoing_reserved: bool,
    pub incoming_reserved: bool,
    pub transition: PreparedTransition,
    pub outgoing_mapping: SourceUvMapping,
    pub incoming_mapping: SourceUvMapping,
    pub scratch_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmittedDissolve {
    pub stable_layer_id: u64,
    pub outgoing_generation: u64,
    pub incoming_generation: u64,
    pub outgoing_mapping: SourceUvMapping,
    pub incoming_mapping: SourceUvMapping,
    pub scratch_bytes: u64,
    pub transition: PreparedTransition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedTransitionPlan {
    pub request_count: u32,
    pub cut_count: u32,
    pub dissolve_count: u8,
    pub scratch_bytes: u64,
    pub dissolves: [Option<AdmittedDissolve>; MAX_SIMULTANEOUS_DISSOLVES],
}

impl PreparedTransitionPlan {
    pub const fn exact_cut_only(request_count: u32) -> Self {
        Self {
            request_count,
            cut_count: request_count,
            dissolve_count: 0,
            scratch_bytes: 0,
            dissolves: [None; MAX_SIMULTANEOUS_DISSOLVES],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionAdmissionError {
    InvalidConfig,
    DuplicateLayer,
    IncomingNotReady,
    MissingReservation,
    StaleOutgoing,
    InvalidGeneration,
    InvalidMapping,
    DissolveSlotCap,
    ScratchByteCap,
    TransitionAlreadyActive,
}

/// Pure all-or-none Scene admission.  Failure returns no partial plan and the
/// caller must retain every prior live activation unchanged.
pub fn plan_prepared_transitions(
    requests: &[TransitionActivationRequest],
    active_transition_layers: &[u64],
    scratch_cap_bytes: u64,
) -> Result<PreparedTransitionPlan, TransitionAdmissionError> {
    let mut plan = PreparedTransitionPlan::exact_cut_only(requests.len() as u32);
    for (index, request) in requests.iter().copied().enumerate() {
        if requests[..index]
            .iter()
            .any(|prior| prior.stable_layer_id == request.stable_layer_id)
        {
            return Err(TransitionAdmissionError::DuplicateLayer);
        }
        request
            .transition
            .validate()
            .map_err(|_| TransitionAdmissionError::InvalidConfig)?;
        if request.transition.is_exact_cut() {
            continue;
        }
        if active_transition_layers.contains(&request.stable_layer_id) {
            return Err(TransitionAdmissionError::TransitionAlreadyActive);
        }
        if !request.incoming_ready {
            return Err(TransitionAdmissionError::IncomingNotReady);
        }
        if !request.outgoing_reserved || !request.incoming_reserved {
            return Err(TransitionAdmissionError::MissingReservation);
        }
        if request.expected_outgoing_generation != request.observed_outgoing_generation {
            return Err(TransitionAdmissionError::StaleOutgoing);
        }
        if request.outgoing_mapping.is_valid() && request.incoming_mapping.is_valid() {
            // Both are sampled independently at the named pre-layer-effects seam.
        } else {
            return Err(TransitionAdmissionError::InvalidMapping);
        }
        if request.observed_outgoing_generation == 0 || request.incoming_generation == 0 {
            return Err(TransitionAdmissionError::InvalidGeneration);
        }
        let slot = usize::from(plan.dissolve_count);
        if slot == MAX_SIMULTANEOUS_DISSOLVES {
            return Err(TransitionAdmissionError::DissolveSlotCap);
        }
        plan.scratch_bytes = plan
            .scratch_bytes
            .checked_add(request.scratch_bytes)
            .ok_or(TransitionAdmissionError::ScratchByteCap)?;
        if plan.scratch_bytes > scratch_cap_bytes {
            return Err(TransitionAdmissionError::ScratchByteCap);
        }
        plan.cut_count = plan.cut_count.saturating_sub(1);
        plan.dissolve_count = plan.dissolve_count.saturating_add(1);
        plan.dissolves[slot] = Some(AdmittedDissolve {
            stable_layer_id: request.stable_layer_id,
            outgoing_generation: request.observed_outgoing_generation,
            incoming_generation: request.incoming_generation,
            outgoing_mapping: request.outgoing_mapping,
            incoming_mapping: request.incoming_mapping,
            scratch_bytes: request.scratch_bytes,
            transition: request.transition,
        });
    }
    Ok(plan)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PremultipliedRgba(pub [f32; 4]);

/// CPU reference for the one pre-layer-effects transition blend. Endpoint
/// branches deliberately return the original value bit-for-bit, including
/// hidden RGB carried by a zero-alpha fixture.
pub fn blend_premultiplied_reference(
    outgoing: PremultipliedRgba,
    incoming: PremultipliedRgba,
    phase: TransitionPhase,
) -> PremultipliedRgba {
    if phase == TransitionPhase::ZERO {
        return outgoing;
    }
    if phase == TransitionPhase::ONE {
        return incoming;
    }
    let t = phase.as_f32();
    let mut result = [0.0; 4];
    for (result, (left, right)) in result
        .iter_mut()
        .zip(outgoing.0.into_iter().zip(incoming.0))
    {
        *result = left + (right - left) * t;
    }
    PremultipliedRgba(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tick_observation(tick: u64, beat: u64) -> TransitionClockObservation {
        TransitionClockObservation {
            accepted_reference_tick: tick,
            accepted_beat: RationalBeatPosition::new(beat, 1).unwrap(),
            program_paused: false,
            media_frozen: false,
        }
    }

    fn dissolve(ticks: u32) -> PreparedTransition {
        PreparedTransition {
            kind: PreparedTransitionKind::Dissolve,
            duration: PreparedTransitionDuration::ReferenceTicks(ticks),
            interruption: TransitionInterruptionPolicy::Refuse,
        }
    }

    fn request(layer: u64) -> TransitionActivationRequest {
        TransitionActivationRequest {
            stable_layer_id: layer,
            expected_outgoing_generation: layer,
            observed_outgoing_generation: layer,
            incoming_generation: layer + 10,
            incoming_ready: true,
            outgoing_reserved: true,
            incoming_reserved: true,
            transition: dissolve(30),
            outgoing_mapping: SourceUvMapping::IDENTITY,
            incoming_mapping: SourceUvMapping::IDENTITY,
            scratch_bytes: 4 * 1024 * 1024,
        }
    }

    #[test]
    fn absent_cut_and_zero_duration_are_exact_zero_resource() {
        let yaml = serde_yaml::to_string(&PreparedTransition::default()).unwrap();
        assert_eq!(yaml, "{}\n");
        assert_eq!(
            serde_yaml::from_str::<PreparedTransition>("{}").unwrap(),
            PreparedTransition::exact_cut()
        );
        let mut zero = request(1);
        zero.transition = dissolve(0);
        let plan = plan_prepared_transitions(&[zero], &[], 0).unwrap();
        assert_eq!(plan, PreparedTransitionPlan::exact_cut_only(1));
    }

    #[test]
    fn scene_admission_is_atomic_and_capped_at_two() {
        let two =
            plan_prepared_transitions(&[request(1), request(2)], &[], 8 * 1024 * 1024).unwrap();
        assert_eq!(two.dissolve_count, 2);
        assert_eq!(two.scratch_bytes, 8 * 1024 * 1024);
        assert_eq!(
            plan_prepared_transitions(&[request(1), request(2), request(3)], &[], 12 * 1024 * 1024),
            Err(TransitionAdmissionError::DissolveSlotCap)
        );
        let mut late = request(2);
        late.incoming_ready = false;
        assert_eq!(
            plan_prepared_transitions(&[request(1), late], &[], 8 * 1024 * 1024),
            Err(TransitionAdmissionError::IncomingNotReady)
        );
    }

    #[test]
    fn resource_and_generation_boundaries_refuse_without_partial_plan() {
        let exact = request(1);
        assert!(plan_prepared_transitions(&[exact], &[], exact.scratch_bytes).is_ok());
        assert_eq!(
            plan_prepared_transitions(&[exact], &[], exact.scratch_bytes - 1),
            Err(TransitionAdmissionError::ScratchByteCap)
        );
        let mut stale = request(1);
        stale.observed_outgoing_generation += 1;
        assert_eq!(
            plan_prepared_transitions(&[stale], &[], u64::MAX),
            Err(TransitionAdmissionError::StaleOutgoing)
        );
        assert_eq!(
            plan_prepared_transitions(&[request(1)], &[1], u64::MAX),
            Err(TransitionAdmissionError::TransitionAlreadyActive)
        );
    }

    #[test]
    fn reference_tick_clock_has_exact_endpoints_and_freeze_law() {
        let mut clock = TransitionClock::start(dissolve(4), tick_observation(10, 0)).unwrap();
        assert_eq!(clock.phase(), TransitionPhase::ZERO);
        assert_eq!(clock.observe(tick_observation(12, 0)).as_f32(), 0.5);
        let mut held = tick_observation(14, 0);
        held.media_frozen = true;
        assert_eq!(clock.observe(held).as_f32(), 0.5);
        assert_eq!(clock.observe(tick_observation(16, 0)), TransitionPhase::ONE);
    }

    #[test]
    fn rational_beat_clock_is_independent_of_redraw_count() {
        let transition = PreparedTransition {
            kind: PreparedTransitionKind::Dissolve,
            duration: PreparedTransitionDuration::RationalBeats {
                numerator: 3,
                denominator: 2,
            },
            interruption: TransitionInterruptionPolicy::Refuse,
        };
        let start = TransitionClockObservation {
            accepted_reference_tick: 0,
            accepted_beat: RationalBeatPosition::new(4, 2).unwrap(),
            program_paused: false,
            media_frozen: false,
        };
        let mut clock = TransitionClock::start(transition, start).unwrap();
        let redraw_only = TransitionClockObservation {
            accepted_reference_tick: 0,
            ..start
        };
        assert_eq!(clock.observe(redraw_only), TransitionPhase::ZERO);
        let halfway = TransitionClockObservation {
            accepted_beat: RationalBeatPosition::new(11, 4).unwrap(),
            ..start
        };
        assert_eq!(clock.observe(halfway).as_f32(), 0.5);
        let end = TransitionClockObservation {
            accepted_beat: RationalBeatPosition::new(14, 4).unwrap(),
            ..start
        };
        assert_eq!(clock.observe(end), TransitionPhase::ONE);
    }

    #[test]
    fn premultiplied_reference_preserves_exact_endpoints_and_hidden_rgb() {
        let outgoing = PremultipliedRgba([0.75, 0.25, 0.5, 0.0]);
        let incoming = PremultipliedRgba([0.0, 0.5, 0.25, 1.0]);
        assert_eq!(
            blend_premultiplied_reference(outgoing, incoming, TransitionPhase::ZERO),
            outgoing
        );
        assert_eq!(
            blend_premultiplied_reference(outgoing, incoming, TransitionPhase::ONE),
            incoming
        );
        assert_eq!(
            blend_premultiplied_reference(
                outgoing,
                incoming,
                TransitionPhase {
                    numerator: 1,
                    denominator: 2,
                }
            ),
            PremultipliedRgba([0.375, 0.375, 0.375, 0.5])
        );
    }
}
