//! Typed admission boundary for recordable creative value mutations.
//!
//! Transport adapters may disagree about framing, but they are not allowed to
//! disagree about what a performance take records.  An action first becomes a
//! [`CreativeMutationCandidate`] after parsing, stable-ID resolution, and value
//! normalization have produced the existing v1 performance address/value law.
//! Only the live applier may promote that candidate to an
//! [`AcceptedCreativeMutation`].  The frame gate then stamps the reference tick
//! and delegates unchanged to the v1 take encoder.
//!
//! Origin is deliberately process-local provenance: it is bounded, never
//! serialized, and is ignored by duplicate comparison and replay.  Therefore
//! adding this seam cannot change v1 take bytes or checksums.

use crate::action_correlation::ActionSourceClass;
use crate::performance_track::{PerformanceControl, PerformanceRawValue, PerformanceValueLaw};

/// Machine-readable inventory of every origin admitted by engine ingress.
/// Keep this array and [`origin_policy`] exhaustive when the source enum grows.
#[cfg(test)]
pub const ALL_CREATIVE_MUTATION_ORIGINS: [ActionSourceClass; 7] = [
    ActionSourceClass::Browser,
    ActionSourceClass::Phone,
    ActionSourceClass::Native,
    ActionSourceClass::Midi,
    ActionSourceClass::Osc,
    ActionSourceClass::Automation,
    ActionSourceClass::Replay,
];

/// A bounded process-only provenance tag for accepted live mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreativeMutationOrigin {
    Browser,
    Phone,
    Native,
    Midi,
    Osc,
    Automation,
}

/// Whether an ingress class may contribute to a newly recorded take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreativeMutationOriginPolicy {
    Recordable(CreativeMutationOrigin),
    ReplayExcluded,
}

/// Exhaustive mapping from the engine's ingress vocabulary to the D4 policy.
pub const fn origin_policy(source: ActionSourceClass) -> CreativeMutationOriginPolicy {
    match source {
        ActionSourceClass::Browser => {
            CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Browser)
        }
        ActionSourceClass::Phone => {
            CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Phone)
        }
        ActionSourceClass::Native => {
            CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Native)
        }
        ActionSourceClass::Midi => {
            CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Midi)
        }
        ActionSourceClass::Osc => {
            CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Osc)
        }
        ActionSourceClass::Automation => {
            CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Automation)
        }
        ActionSourceClass::Replay => CreativeMutationOriginPolicy::ReplayExcluded,
    }
}

/// A normalized recordable mutation awaiting the live applier's verdict.
///
/// Construction proves that the raw carrier fits the address law.  It does not
/// claim that stable revision, safety, Morph release, or planner preflight has
/// accepted the live edit; those checks remain owned by the live applier.
#[derive(Debug, Clone)]
pub struct CreativeMutationCandidate {
    origin: CreativeMutationOrigin,
    control: PerformanceControl,
    law: PerformanceValueLaw,
    raw: PerformanceRawValue,
    canonical_value: u16,
}

impl CreativeMutationCandidate {
    pub fn new(
        source: ActionSourceClass,
        control: PerformanceControl,
        law: PerformanceValueLaw,
        raw: PerformanceRawValue,
    ) -> Option<Self> {
        let CreativeMutationOriginPolicy::Recordable(origin) = origin_policy(source) else {
            return None;
        };
        let canonical_value = law.encode(&raw)?;
        Some(Self {
            origin,
            control,
            law,
            raw,
            canonical_value,
        })
    }

    /// Promotion is intentionally possible only at the post-validation seam.
    pub fn accept(self) -> AcceptedCreativeMutation {
        AcceptedCreativeMutation {
            origin: self.origin,
            control: self.control,
            law: self.law,
            raw: self.raw,
            canonical_value: self.canonical_value,
        }
    }
}

/// One live-applied mutation waiting for the accepted-frame tick.
#[derive(Debug, Clone)]
pub struct AcceptedCreativeMutation {
    origin: CreativeMutationOrigin,
    control: PerformanceControl,
    law: PerformanceValueLaw,
    raw: PerformanceRawValue,
    canonical_value: u16,
}

impl AcceptedCreativeMutation {
    #[cfg(test)]
    pub const fn origin(&self) -> CreativeMutationOrigin {
        self.origin
    }

    /// Canonical duplicate identity deliberately omits origin.  Two adapters
    /// delivering the same address/value in one frame represent one edit.
    pub fn same_canonical_event(&self, other: &Self) -> bool {
        self.control == other.control
            && self.law == other.law
            && self.canonical_value == other.canonical_value
    }

    pub fn into_v1_parts(self) -> (PerformanceControl, PerformanceValueLaw, PerformanceRawValue) {
        (self.control, self.law, self.raw)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageAcceptedMutation {
    Inserted,
    Duplicate,
    Full,
}

/// Bounded same-frame insertion.  The first source wins provenance only; all
/// origins encode exactly the same v1 event because duplicate identity uses the
/// canonical address/law/value triple.
pub fn stage_accepted_mutation(
    pending: &mut Vec<AcceptedCreativeMutation>,
    mutation: AcceptedCreativeMutation,
    max_pending: usize,
) -> StageAcceptedMutation {
    // Provenance remains retained on the pending item for bounded diagnostics,
    // but must not participate in equality or the v1 stream.
    let _process_only_origin = mutation.origin;
    if pending
        .iter()
        .any(|existing| existing.same_canonical_event(&mutation))
    {
        return StageAcceptedMutation::Duplicate;
    }
    if pending.len() >= max_pending {
        return StageAcceptedMutation::Full;
    }
    pending.push(mutation);
    StageAcceptedMutation::Inserted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(source: ActionSourceClass, value: f32) -> CreativeMutationCandidate {
        CreativeMutationCandidate::new(
            source,
            PerformanceControl::Master {
                param: "brightness".to_string(),
            },
            PerformanceValueLaw::Unit {
                min: -1.0,
                max: 1.0,
            },
            PerformanceRawValue::Continuous(value),
        )
        .expect("live origin and finite value")
    }

    #[test]
    fn every_ingress_origin_has_an_explicit_recording_policy() {
        let policies = ALL_CREATIVE_MUTATION_ORIGINS.map(origin_policy);
        assert_eq!(
            policies,
            [
                CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Browser),
                CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Phone),
                CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Native),
                CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Midi),
                CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Osc),
                CreativeMutationOriginPolicy::Recordable(CreativeMutationOrigin::Automation),
                CreativeMutationOriginPolicy::ReplayExcluded,
            ]
        );
    }

    #[test]
    fn replay_can_never_be_promoted_to_an_accepted_mutation() {
        assert!(CreativeMutationCandidate::new(
            ActionSourceClass::Replay,
            PerformanceControl::Master {
                param: "brightness".to_string(),
            },
            PerformanceValueLaw::Unit {
                min: -1.0,
                max: 1.0,
            },
            PerformanceRawValue::Continuous(0.25),
        )
        .is_none());
    }

    #[test]
    fn unrepresentable_values_never_reach_the_accepted_type() {
        assert!(CreativeMutationCandidate::new(
            ActionSourceClass::Browser,
            PerformanceControl::Master {
                param: "brightness".to_string(),
            },
            PerformanceValueLaw::Unit {
                min: -1.0,
                max: 1.0,
            },
            PerformanceRawValue::Continuous(f32::NAN),
        )
        .is_none());
        assert!(CreativeMutationCandidate::new(
            ActionSourceClass::Osc,
            PerformanceControl::Temporal {
                param: "slit_map".to_string(),
            },
            PerformanceValueLaw::Discrete {
                vocab: vec!["ramp".to_string(), "sweep".to_string()],
            },
            PerformanceRawValue::Token("outside-vocabulary".to_string()),
        )
        .is_none());
    }

    #[test]
    fn same_canonical_value_from_different_origins_deduplicates() {
        let mut pending = Vec::new();
        let browser = candidate(ActionSourceClass::Browser, 0.25).accept();
        let midi = candidate(ActionSourceClass::Midi, 0.25).accept();
        assert_eq!(
            stage_accepted_mutation(&mut pending, browser, 8),
            StageAcceptedMutation::Inserted
        );
        assert_eq!(
            stage_accepted_mutation(&mut pending, midi, 8),
            StageAcceptedMutation::Duplicate
        );
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].origin(), CreativeMutationOrigin::Browser);
    }

    #[test]
    fn one_over_the_pending_cap_is_visible_and_bounded() {
        let mut pending = Vec::new();
        assert_eq!(
            stage_accepted_mutation(
                &mut pending,
                candidate(ActionSourceClass::Osc, 0.25).accept(),
                1,
            ),
            StageAcceptedMutation::Inserted
        );
        assert_eq!(
            stage_accepted_mutation(
                &mut pending,
                candidate(ActionSourceClass::Automation, 0.5).accept(),
                1,
            ),
            StageAcceptedMutation::Full
        );
        assert_eq!(pending.len(), 1);
    }

    fn v1_take_from_origin(source: ActionSourceClass) -> crate::performance_track::PerformanceTake {
        let mut pending = Vec::new();
        assert_eq!(
            stage_accepted_mutation(&mut pending, candidate(source, 0.25).accept(), 8),
            StageAcceptedMutation::Inserted
        );
        let mut take = crate::performance_track::PerformanceTake::default();
        for mutation in pending {
            let (control, law, raw) = mutation.into_v1_parts();
            take.record_accepted(7, control, law, &raw)
                .expect("frozen v1 address/value");
        }
        take.finalize(12);
        take
    }

    #[test]
    fn every_live_origin_produces_the_same_v1_address_value_tick_and_hash() {
        let baseline = v1_take_from_origin(ActionSourceClass::Browser);
        for source in [
            ActionSourceClass::Phone,
            ActionSourceClass::Native,
            ActionSourceClass::Midi,
            ActionSourceClass::Osc,
            ActionSourceClass::Automation,
        ] {
            let candidate = v1_take_from_origin(source);
            assert_eq!(candidate.addresses(), baseline.addresses());
            assert_eq!(candidate.events(), baseline.events());
            assert_eq!(candidate.canonical_bytes(), baseline.canonical_bytes());
            assert_eq!(candidate.checksum_hex(), baseline.checksum_hex());
        }
        assert_eq!(baseline.events()[0].tick, 7);
    }

    #[test]
    fn accepted_seam_is_byte_identical_to_the_frozen_v1_encoder_call() {
        let through_d4 = v1_take_from_origin(ActionSourceClass::Browser);
        let mut direct_v1 = crate::performance_track::PerformanceTake::default();
        direct_v1
            .record_accepted(
                7,
                PerformanceControl::Master {
                    param: "brightness".to_string(),
                },
                PerformanceValueLaw::Unit {
                    min: -1.0,
                    max: 1.0,
                },
                &PerformanceRawValue::Continuous(0.25),
            )
            .expect("legacy v1 fixture");
        direct_v1.finalize(12);
        assert_eq!(through_d4.canonical_bytes(), direct_v1.canonical_bytes());
        assert_eq!(through_d4.checksum_hex(), direct_v1.checksum_hex());
        assert_eq!(
            through_d4.checksum_hex(),
            "be4bb410f3984214fc13667f4135208d089d14aefbbff2fb2f6e19ff5a0758d6",
            "the frozen v1 brightness/tick fixture must remain byte exact"
        );
    }
}
