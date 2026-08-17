//! Typed image-tap and matte-routing contract.
//!
//! Runtime routes use process-stable live layer IDs. Patch DTOs use bounded
//! saved layer positions and must be explicitly mapped during capture/restore;
//! a process identity is never persisted or trusted as a vector index.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::performance::SavedLayerPosition;
use crate::visual_rack::GroupId;

/// Non-zero process-stable identity of a live layer instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableLayerId(u64);

impl StableLayerId {
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Source stage exposed by a selected live layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerImageStage {
    PreLocalEffects,
    #[default]
    PostLocalEffects,
}

/// Runtime M1 image input. The M1 compositor diagnoses group outputs as
/// unavailable; the M2 composition planner resolves the same stable GroupId.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImageInput {
    SelectedLayer {
        layer_id: StableLayerId,
        stage: LayerImageStage,
    },
    /// Authored donor position that could not be mapped during exact restore.
    /// It stays in the graph so a later repair/capture cannot silently retarget it.
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
        stage: LayerImageStage,
    },
    #[default]
    OneBelow,
    AllBelow,
    CleanProgram,
    ProgramHistory,
    GroupOutput {
        group_id: GroupId,
    },
    /// Stable tombstone retained after explicit group deletion.
    MissingGroupOutput {
        group_id: GroupId,
    },
}

/// Serializable counterpart to [`ImageInput`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum SavedImageInput {
    SelectedLayer {
        layer_position: SavedLayerPosition,
        #[serde(default)]
        stage: LayerImageStage,
    },
    /// A selected donor that was explicitly observed missing before capture.
    ///
    /// This must remain distinct from `SelectedLayer`: after a layer deletion,
    /// the vacated saved position may already belong to another layer. Resolving
    /// that position on restore would silently retarget the authored edge.
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
        #[serde(default)]
        stage: LayerImageStage,
    },
    #[default]
    OneBelow,
    AllBelow,
    CleanProgram,
    ProgramHistory,
    /// Stable group output. Legacy M1 evaluation keeps it transparent, while
    /// the M2 composition planner resolves it in the unified graph.
    GroupOutput {
        group_id: GroupId,
    },
    /// Stable tombstone retained after explicit group deletion.
    MissingGroupOutput {
        group_id: GroupId,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatteChannel {
    #[default]
    Alpha,
    Luma,
    Red,
    Green,
    Blue,
}

/// Frame-local live matte route.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerMatte {
    pub enabled: bool,
    pub input: ImageInput,
    pub channel: MatteChannel,
    pub invert: bool,
    pub amount: f32,
    pub threshold: f32,
    pub softness: f32,
}

impl Default for LayerMatte {
    fn default() -> Self {
        Self {
            enabled: false,
            input: ImageInput::default(),
            channel: MatteChannel::Alpha,
            invert: false,
            amount: 1.0,
            threshold: 0.5,
            softness: 0.1,
        }
    }
}

impl LayerMatte {
    pub fn sanitized(self) -> Self {
        Self {
            enabled: self.enabled,
            input: self.input,
            channel: self.channel,
            invert: self.invert,
            amount: finite_clamp(self.amount, 1.0),
            threshold: finite_clamp(self.threshold, 0.5),
            softness: finite_clamp(self.softness, 0.1),
        }
    }

    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        if self.input == (ImageInput::GroupOutput { group_id: removed }) {
            self.input = ImageInput::MissingGroupOutput { group_id: removed };
        }
    }
}

/// Patch-safe matte route. Disabled is the legacy identity and serializes only
/// when a caller chooses to retain an explicitly authored configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct LayerMatteConfig {
    pub enabled: bool,
    pub input: SavedImageInput,
    pub channel: MatteChannel,
    pub invert: bool,
    pub amount: f32,
    pub threshold: f32,
    pub softness: f32,
}

impl Default for LayerMatteConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            input: SavedImageInput::default(),
            channel: MatteChannel::Alpha,
            invert: false,
            amount: 1.0,
            threshold: 0.5,
            softness: 0.1,
        }
    }
}

impl<'de> Deserialize<'de> for LayerMatteConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(default)]
        struct Raw {
            enabled: bool,
            input: SavedImageInput,
            channel: MatteChannel,
            invert: bool,
            amount: f32,
            threshold: f32,
            softness: f32,
        }

        impl Default for Raw {
            fn default() -> Self {
                let value = LayerMatteConfig::default();
                Self {
                    enabled: value.enabled,
                    input: value.input,
                    channel: value.channel,
                    invert: value.invert,
                    amount: value.amount,
                    threshold: value.threshold,
                    softness: value.softness,
                }
            }
        }

        let raw = Raw::deserialize(deserializer)?;
        Ok(Self {
            enabled: raw.enabled,
            input: raw.input,
            channel: raw.channel,
            invert: raw.invert,
            amount: raw.amount,
            threshold: raw.threshold,
            softness: raw.softness,
        }
        .sanitized())
    }
}

impl LayerMatteConfig {
    pub fn is_legacy_disabled(&self) -> bool {
        *self == Self::default()
    }

    pub fn sanitized(self) -> Self {
        Self {
            enabled: self.enabled,
            input: self.input,
            channel: self.channel,
            invert: self.invert,
            amount: finite_clamp(self.amount, 1.0),
            threshold: finite_clamp(self.threshold, 0.5),
            softness: finite_clamp(self.softness, 0.1),
        }
    }

    #[allow(
        dead_code,
        reason = "saved layer-matte invalidation remains a patch/editor migration API"
    )]
    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        if self.input == (SavedImageInput::GroupOutput { group_id: removed }) {
            self.input = SavedImageInput::MissingGroupOutput { group_id: removed };
        }
    }

    /// Map saved positions to live process identities. Missing inputs are
    /// returned explicitly so callers can use transparent black and surface a
    /// status instead of reusing stale textures.
    pub fn to_runtime(
        self,
        mut layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
    ) -> LayerMatte {
        let value = self.sanitized();
        if !value.enabled {
            return LayerMatte::default();
        }
        let input = match value.input {
            SavedImageInput::SelectedLayer {
                layer_position,
                stage,
            } => layer_at_position(layer_position).map_or(
                ImageInput::MissingSelectedLayer {
                    saved_position: layer_position,
                    stage,
                },
                |layer_id| ImageInput::SelectedLayer { layer_id, stage },
            ),
            SavedImageInput::MissingSelectedLayer {
                saved_position,
                stage,
            } => ImageInput::MissingSelectedLayer {
                saved_position,
                stage,
            },
            SavedImageInput::OneBelow => ImageInput::OneBelow,
            SavedImageInput::AllBelow => ImageInput::AllBelow,
            SavedImageInput::CleanProgram => ImageInput::CleanProgram,
            SavedImageInput::ProgramHistory => ImageInput::ProgramHistory,
            SavedImageInput::GroupOutput { group_id } => ImageInput::GroupOutput { group_id },
            SavedImageInput::MissingGroupOutput { group_id } => {
                ImageInput::MissingGroupOutput { group_id }
            }
        };
        LayerMatte {
            enabled: true,
            input,
            channel: value.channel,
            invert: value.invert,
            amount: value.amount,
            threshold: value.threshold,
            softness: value.softness,
        }
    }

    /// Map a live route back to a saved positional DTO. Group identity is
    /// stable and never projected through a layer position.
    pub fn from_runtime(
        matte: LayerMatte,
        mut position_of_layer: impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> Result<Self, ImageRouteDiagnostic> {
        let matte = matte.sanitized();
        let input = match matte.input {
            ImageInput::SelectedLayer { layer_id, stage } => SavedImageInput::SelectedLayer {
                layer_position: position_of_layer(layer_id).ok_or(
                    ImageRouteDiagnostic::Missing(MissingImageInput::LiveLayer(layer_id)),
                )?,
                stage,
            },
            ImageInput::MissingSelectedLayer {
                saved_position,
                stage,
            } => SavedImageInput::MissingSelectedLayer {
                saved_position,
                stage,
            },
            ImageInput::OneBelow => SavedImageInput::OneBelow,
            ImageInput::AllBelow => SavedImageInput::AllBelow,
            ImageInput::CleanProgram => SavedImageInput::CleanProgram,
            ImageInput::ProgramHistory => SavedImageInput::ProgramHistory,
            ImageInput::GroupOutput { group_id } => SavedImageInput::GroupOutput { group_id },
            ImageInput::MissingGroupOutput { group_id } => {
                SavedImageInput::MissingGroupOutput { group_id }
            }
        };
        Ok(Self {
            enabled: matte.enabled,
            input,
            channel: matte.channel,
            invert: matte.invert,
            amount: matte.amount,
            threshold: matte.threshold,
            softness: matte.softness,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingImageInput {
    SelectedLayer(SavedLayerPosition),
    LiveLayer(StableLayerId),
    OneBelow,
    AllBelow,
    ProgramHistoryUninitialized,
    GroupOutputUnavailable(GroupId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageRouteCycle {
    /// A same-frame program tap would depend on the layer currently being
    /// composited. Use ProgramHistory for the intentional delayed feedback edge.
    CleanProgramSameFrame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageRouteDiagnostic {
    Disabled,
    Ready,
    Missing(MissingImageInput),
    Cycle(ImageRouteCycle),
}

impl fmt::Display for ImageRouteDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => formatter.write_str("matte disabled"),
            Self::Ready => formatter.write_str("image input ready"),
            Self::Missing(MissingImageInput::SelectedLayer(position)) => write!(
                formatter,
                "saved layer position {} is missing; using transparent input",
                position.get()
            ),
            Self::Missing(MissingImageInput::LiveLayer(layer_id)) => write!(
                formatter,
                "live layer {} cannot be mapped; using transparent input",
                layer_id.get()
            ),
            Self::Missing(MissingImageInput::OneBelow) => {
                formatter.write_str("there is no layer below; using transparent input")
            }
            Self::Missing(MissingImageInput::AllBelow) => {
                formatter.write_str("there are no layers below; using transparent input")
            }
            Self::Missing(MissingImageInput::ProgramHistoryUninitialized) => {
                formatter.write_str("program history is not initialized; using transparent input")
            }
            Self::Missing(MissingImageInput::GroupOutputUnavailable(group_id)) => write!(
                formatter,
                "group output {} is unavailable; using transparent input",
                group_id.get()
            ),
            Self::Cycle(ImageRouteCycle::CleanProgramSameFrame) => formatter.write_str(
                "same-frame clean program would form a cycle; use program history instead",
            ),
        }
    }
}

/// Read-only availability facts supplied by the compositor.
pub struct ImageRouteContext<'a> {
    pub available_layers: &'a [StableLayerId],
    pub has_one_below: bool,
    pub program_history_initialized: bool,
}

impl ImageInput {
    pub fn diagnose(self, context: &ImageRouteContext<'_>) -> ImageRouteDiagnostic {
        match self {
            Self::SelectedLayer { layer_id, .. } => {
                if context.available_layers.contains(&layer_id) {
                    ImageRouteDiagnostic::Ready
                } else {
                    ImageRouteDiagnostic::Missing(MissingImageInput::LiveLayer(layer_id))
                }
            }
            Self::MissingSelectedLayer { saved_position, .. } => {
                ImageRouteDiagnostic::Missing(MissingImageInput::SelectedLayer(saved_position))
            }
            Self::OneBelow if !context.has_one_below => {
                ImageRouteDiagnostic::Missing(MissingImageInput::OneBelow)
            }
            Self::AllBelow if !context.has_one_below => {
                ImageRouteDiagnostic::Missing(MissingImageInput::AllBelow)
            }
            Self::ProgramHistory if !context.program_history_initialized => {
                ImageRouteDiagnostic::Missing(MissingImageInput::ProgramHistoryUninitialized)
            }
            Self::GroupOutput { group_id } => {
                ImageRouteDiagnostic::Missing(MissingImageInput::GroupOutputUnavailable(group_id))
            }
            Self::MissingGroupOutput { group_id } => {
                ImageRouteDiagnostic::Missing(MissingImageInput::GroupOutputUnavailable(group_id))
            }
            Self::CleanProgram => {
                ImageRouteDiagnostic::Cycle(ImageRouteCycle::CleanProgramSameFrame)
            }
            Self::OneBelow | Self::AllBelow | Self::ProgramHistory => ImageRouteDiagnostic::Ready,
        }
    }
}

impl LayerMatte {
    pub fn diagnose(self, context: &ImageRouteContext<'_>) -> ImageRouteDiagnostic {
        if self.enabled {
            self.input.diagnose(context)
        } else {
            ImageRouteDiagnostic::Disabled
        }
    }
}

fn finite_clamp(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: u64) -> StableLayerId {
        StableLayerId::new(value).unwrap()
    }

    fn group_id(value: u64) -> GroupId {
        GroupId::new(value).unwrap()
    }

    #[test]
    fn saved_selected_layer_maps_by_position_and_round_trips_all_controls() {
        let saved = LayerMatteConfig {
            enabled: true,
            input: SavedImageInput::SelectedLayer {
                layer_position: SavedLayerPosition::new(2).unwrap(),
                stage: LayerImageStage::PreLocalEffects,
            },
            channel: MatteChannel::Luma,
            invert: true,
            amount: 0.75,
            threshold: 0.3,
            softness: 0.2,
        };
        let runtime = saved.to_runtime(|position| (position.get() == 2).then(|| id(99)));
        assert_eq!(
            runtime.input,
            ImageInput::SelectedLayer {
                layer_id: id(99),
                stage: LayerImageStage::PreLocalEffects
            }
        );
        assert_eq!(
            LayerMatteConfig::from_runtime(runtime, |layer_id| {
                (layer_id == id(99)).then(|| SavedLayerPosition::new(2).unwrap())
            })
            .unwrap(),
            saved
        );
        let yaml = serde_yaml::to_string(&saved).unwrap();
        assert_eq!(
            serde_yaml::from_str::<LayerMatteConfig>(&yaml).unwrap(),
            saved
        );
    }

    #[test]
    fn missing_cycle_and_reserved_routes_are_explicit_and_never_stale() {
        let context = ImageRouteContext {
            available_layers: &[id(4)],
            has_one_below: false,
            program_history_initialized: false,
        };
        assert!(matches!(
            ImageInput::SelectedLayer {
                layer_id: id(5),
                stage: LayerImageStage::PostLocalEffects
            }
            .diagnose(&context),
            ImageRouteDiagnostic::Missing(MissingImageInput::LiveLayer(_))
        ));
        assert_eq!(
            ImageInput::CleanProgram.diagnose(&context),
            ImageRouteDiagnostic::Cycle(ImageRouteCycle::CleanProgramSameFrame)
        );
        assert_eq!(
            ImageInput::ProgramHistory.diagnose(&context),
            ImageRouteDiagnostic::Missing(MissingImageInput::ProgramHistoryUninitialized)
        );
        assert_eq!(
            ImageInput::AllBelow.diagnose(&context),
            ImageRouteDiagnostic::Missing(MissingImageInput::AllBelow)
        );
        assert_eq!(
            ImageInput::GroupOutput {
                group_id: group_id(8),
            }
            .diagnose(&context),
            ImageRouteDiagnostic::Missing(MissingImageInput::GroupOutputUnavailable(group_id(8)))
        );
    }

    #[test]
    fn disabled_is_legacy_identity_and_hostile_floats_sanitize() {
        assert!(LayerMatteConfig::default().is_legacy_disabled());
        let value = LayerMatteConfig {
            enabled: true,
            amount: f32::NAN,
            threshold: f32::INFINITY,
            softness: -3.0,
            ..LayerMatteConfig::default()
        }
        .sanitized();
        assert_eq!(value.amount, 1.0);
        assert_eq!(value.threshold, 0.5);
        assert_eq!(value.softness, 0.0);
    }

    #[test]
    fn every_saved_route_variant_is_serde_supported() {
        let inputs = [
            SavedImageInput::SelectedLayer {
                layer_position: SavedLayerPosition::new(0).unwrap(),
                stage: LayerImageStage::PostLocalEffects,
            },
            SavedImageInput::MissingSelectedLayer {
                saved_position: SavedLayerPosition::new(0).unwrap(),
                stage: LayerImageStage::PreLocalEffects,
            },
            SavedImageInput::OneBelow,
            SavedImageInput::AllBelow,
            SavedImageInput::CleanProgram,
            SavedImageInput::ProgramHistory,
            SavedImageInput::GroupOutput {
                group_id: group_id(12),
            },
            SavedImageInput::MissingGroupOutput {
                group_id: group_id(13),
            },
        ];
        for input in inputs {
            let yaml = serde_yaml::to_string(&input).unwrap();
            assert_eq!(
                serde_yaml::from_str::<SavedImageInput>(&yaml).unwrap(),
                input
            );
        }
        let group = LayerMatteConfig {
            enabled: true,
            input: SavedImageInput::GroupOutput {
                group_id: group_id(12),
            },
            ..LayerMatteConfig::default()
        };
        let runtime = group.to_runtime(|_| None);
        assert_eq!(
            runtime.diagnose(&ImageRouteContext {
                available_layers: &[],
                has_one_below: false,
                program_history_initialized: false,
            }),
            ImageRouteDiagnostic::Missing(MissingImageInput::GroupOutputUnavailable(group_id(12)))
        );
    }

    #[test]
    fn legacy_integer_group_output_deserializes_as_nonzero_u64_identity() {
        let decoded: SavedImageInput =
            serde_json::from_str(r#"{"source":"group_output","group_id":4294967297}"#).unwrap();
        assert_eq!(
            decoded,
            SavedImageInput::GroupOutput {
                group_id: group_id(4_294_967_297),
            }
        );
        assert!(serde_json::from_str::<SavedImageInput>(
            r#"{"source":"group_output","group_id":0}"#
        )
        .is_err());
    }

    #[test]
    fn deleted_group_route_becomes_a_stable_missing_tombstone() {
        let removed = group_id(u64::MAX - 1);
        let mut runtime = LayerMatte {
            enabled: true,
            input: ImageInput::GroupOutput { group_id: removed },
            ..LayerMatte::default()
        };
        runtime.mark_group_output_missing(removed);
        assert_eq!(
            runtime.input,
            ImageInput::MissingGroupOutput { group_id: removed }
        );
        let saved = LayerMatteConfig::from_runtime(runtime, |_| None).unwrap();
        assert_eq!(
            saved.input,
            SavedImageInput::MissingGroupOutput { group_id: removed }
        );
        assert_eq!(
            saved.to_runtime(|_| None).input,
            ImageInput::MissingGroupOutput { group_id: removed }
        );
    }

    #[test]
    fn unresolved_selected_donor_becomes_explicitly_missing_on_restore() {
        let saved = LayerMatteConfig {
            enabled: true,
            input: SavedImageInput::SelectedLayer {
                layer_position: SavedLayerPosition::new(23).unwrap(),
                stage: LayerImageStage::PreLocalEffects,
            },
            ..LayerMatteConfig::default()
        };
        let runtime = saved.to_runtime(|_| None);
        assert_eq!(
            runtime.input,
            ImageInput::MissingSelectedLayer {
                saved_position: SavedLayerPosition::new(23).unwrap(),
                stage: LayerImageStage::PreLocalEffects,
            }
        );
        assert_eq!(
            LayerMatteConfig::from_runtime(runtime, |_| None).unwrap(),
            LayerMatteConfig {
                input: SavedImageInput::MissingSelectedLayer {
                    saved_position: SavedLayerPosition::new(23).unwrap(),
                    stage: LayerImageStage::PreLocalEffects,
                },
                ..saved
            }
        );
    }

    #[test]
    fn deleted_donor_capture_serde_restore_never_retargets_the_vacated_position() {
        let saved_position = SavedLayerPosition::new(0).unwrap();
        let deleted_runtime_edge = LayerMatte {
            enabled: true,
            input: ImageInput::MissingSelectedLayer {
                saved_position,
                stage: LayerImageStage::PostLocalEffects,
            },
            channel: MatteChannel::Luma,
            invert: true,
            amount: 0.75,
            threshold: 0.25,
            softness: 0.2,
        };

        // Capture after deletion must write an explicit missing edge, not the
        // ordinary selected-position representation used by a live donor.
        let captured = LayerMatteConfig::from_runtime(deleted_runtime_edge, |_| {
            panic!("a missing runtime donor must not consult live identity mapping")
        })
        .unwrap();
        assert_eq!(
            captured.input,
            SavedImageInput::MissingSelectedLayer {
                saved_position,
                stage: LayerImageStage::PostLocalEffects,
            }
        );

        let yaml = serde_yaml::to_string(&captured).unwrap();
        assert!(yaml.contains("source: missing_selected_layer"));
        let restored: LayerMatteConfig = serde_yaml::from_str(&yaml).unwrap();

        // Simulate the post-delete stack shift: position zero is occupied by
        // an unrelated live layer. The explicit missing edge must not resolve
        // it and must remain transparent/diagnosable after restore.
        let replacement_id = id(999);
        let runtime = restored.to_runtime(|position| {
            assert_eq!(position, saved_position);
            Some(replacement_id)
        });
        assert_eq!(runtime, deleted_runtime_edge);
        assert!(matches!(
            runtime.diagnose(&ImageRouteContext {
                available_layers: &[replacement_id],
                has_one_below: true,
                program_history_initialized: true,
            }),
            ImageRouteDiagnostic::Missing(MissingImageInput::SelectedLayer(position))
                if position == saved_position
        ));
    }
}
