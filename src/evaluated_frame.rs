//! Immutable, CPU-only evaluation shared by the live and offline renderers.
//!
//! Morphing is intentionally upstream of this module: live materializes its
//! active morph into the authored stack, while export builds equivalent
//! frame-local bases. From that common boundary onward this planner is the
//! sole owner of modulation sampling, render parameters, transport parameters,
//! layer/source identity, and program-wide temporal inputs.

#[path = "evaluated_composition.rs"]
pub(crate) mod evaluated_composition;

use crate::effects::params::TemporalParams;
use crate::effects::EffectUniforms;
use crate::image_routing::{
    ImageInput, ImageRouteContext, ImageRouteDiagnostic, LayerImageStage, LayerMatte, MatteChannel,
    StableLayerId,
};
use crate::layers::BlendMode;
use crate::modulation::ModulationFrame;
use crate::ntsc::NtscParams;
use crate::renderer::compositor::{
    MatteChannelCode, MatteResourceLimits, MatteResourcePlan, ResolvedMatteParams,
    MAX_IMAGE_TAP_BYTES, MAX_MATERIALIZED_IMAGE_TAPS,
};
use crate::spatial::{EffectPassUniforms, SpatialGpuUniforms, SpatialTransform};
use std::collections::BTreeMap;

/// Preallocation is an optimization, never an admission limit. Larger real
/// stacks grow normally, while a pathological iterator hint cannot request an
/// enormous allocation before yielding its first actual layer.
const MAX_LAYER_PREALLOCATION: usize = 256;

/// Identifies the source texture/decoder selected for one evaluated layer.
/// `stable_id` survives UI reorder; `slot` is the frame-local texture tap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceTap {
    pub stable_id: u64,
    pub slot: usize,
    pub size: [u32; 2],
}

impl SourceTap {
    pub fn new(stable_id: u64, slot: usize, width: u32, height: u32) -> Self {
        Self {
            stable_id,
            slot,
            size: [width.max(1), height.max(1)],
        }
    }
}

/// Output facts which are constant during evaluation of one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FramePlanContext {
    pub output_size: [u32; 2],
    pub time_seconds: f32,
    /// Highest engine-minted control sequence applied before this immutable
    /// frame was evaluated. Zero means no correlated action has entered the
    /// frame; offline export intentionally remains zero because it has no live
    /// ingress clock.
    pub highest_applied_action_sequence: u64,
    /// Per-frame Study inputs, sampled from the same immutable frame sample
    /// the modulation matrix consumed (live) or derived from the frame index
    /// (export: audio is zero like every live source, beat comes from the
    /// export beat clock). Neutral zeros for callers with no Study in
    /// flight, so every existing constructor site keeps its exact meaning.
    pub study_audio_bands: [f32; 8],
    pub study_beat_phase: f32,
}

impl FramePlanContext {
    pub fn new(width: u32, height: u32, time_seconds: f32) -> Self {
        Self {
            output_size: [width.max(1), height.max(1)],
            time_seconds: if time_seconds.is_finite() {
                time_seconds.max(0.0)
            } else {
                0.0
            },
            highest_applied_action_sequence: 0,
            study_audio_bands: [0.0; 8],
            study_beat_phase: 0.0,
        }
    }

    pub fn with_highest_applied_action_sequence(mut self, sequence: u64) -> Self {
        self.highest_applied_action_sequence = sequence;
        self
    }

    /// Attach the frame's Study inputs, sanitized exactly as the CPU
    /// reference sanitizes them: non-finite lands on the documented neutral
    /// zero, everything clamps into `0..=1`.
    pub fn with_study_inputs(mut self, audio_bands: [f32; 8], beat_phase: f32) -> Self {
        let sanitize = |value: f32| {
            if value.is_finite() {
                value.clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        self.study_audio_bands = audio_bands.map(sanitize);
        self.study_beat_phase = sanitize(beat_phase);
        self
    }
}

/// Post-morph master bases presented to the shared evaluator.
#[derive(Clone, Copy)]
pub struct MasterFrameInput<'a> {
    pub effects: &'a EffectUniforms,
    pub transform: &'a SpatialTransform,
    pub ntsc: &'a NtscParams,
    pub temporal: &'a TemporalParams,
}

/// Post-morph layer bases presented to the shared evaluator.
#[derive(Clone, Copy)]
pub struct LayerFrameInput<'a> {
    pub source: SourceTap,
    pub effects: &'a EffectUniforms,
    pub transform: &'a SpatialTransform,
    pub opacity: f32,
    pub mosh_send: f32,
    pub speed: f32,
    pub fps: f32,
    pub blend_mode: BlendMode,
    pub visible: bool,
    pub paused: bool,
    pub bypass_master_fx: bool,
    pub bypass_temporal_fx: bool,
    /// The B7 pattern synth's authored base, present only on a pattern
    /// layer. The evaluator resolves this frame's modulated copy so live and
    /// export encode identical pattern uniforms from the plan alone.
    pub pattern: Option<&'a crate::pattern_synth::PatternSynthParams>,
}

/// Non-pixel state resolved for one layer. It stays index-aligned with the
/// exact stored pass returned by [`EvaluatedFramePlan::layer_passes`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluatedLayer {
    pub source: SourceTap,
    /// Resolution-independent evaluated geometry used for semantic
    /// fingerprints and diagnostics. The GPU consumes `spatial` below.
    pub transform: SpatialTransform,
    /// Exact affine/crop/edge payload for this source/output pairing.
    pub spatial: SpatialGpuUniforms,
    /// Final opacity consumed by compositing. Keeping it beside the discrete
    /// metadata lets production renderers consume one plan-owned record
    /// instead of consulting either authored layers or a parallel tuple.
    pub opacity: f32,
    /// Final modulated spatial contribution to the shared Codec Mosh stage.
    pub mosh_send: f32,
    pub speed: f32,
    pub fps: f32,
    pub blend_mode: BlendMode,
    pub visible: bool,
    pub paused: bool,
    pub bypass_master_fx: bool,
    pub bypass_temporal_fx: bool,
    /// This frame's modulated pattern-synth values (base plus offsets,
    /// sanitized), or `None` for every other source kind. Frame-local
    /// evaluated data, never authored state.
    pub pattern: Option<crate::pattern_synth::PatternSynthParams>,
}

/// One unique, output-sized donor image that must be materialized before the
/// layer stack composites. Multiple consumers share the same array layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageTapRequest {
    pub donor_layer_index: usize,
    pub stage: LayerImageStage,
    pub array_layer: u32,
    pub consumers: Vec<usize>,
}

/// Frame-local route after stable-ID lookup and cycle/missing diagnosis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedImageInput {
    Disabled,
    MaterializedTap {
        tap_index: usize,
    },
    AllBelow,
    ProgramHistory,
    /// Missing, reserved, and cyclic inputs are a defined zero field. Their
    /// exact cause remains available in `diagnostic` for operator surfaces.
    Transparent,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluatedLayerMatte {
    pub authored: LayerMatte,
    pub resolved_input: ResolvedImageInput,
    pub params: ResolvedMatteParams,
    pub diagnostic: ImageRouteDiagnostic,
}

impl Default for EvaluatedLayerMatte {
    fn default() -> Self {
        Self {
            authored: LayerMatte::default(),
            resolved_input: ResolvedImageInput::Disabled,
            params: ResolvedMatteParams {
                channel: MatteChannelCode::Alpha,
                invert: false,
                amount: 1.0,
                threshold: 0.5,
                softness: 0.1,
                donor_valid: false,
            },
            diagnostic: ImageRouteDiagnostic::Disabled,
        }
    }
}

/// Immutable image-routing program owned by one evaluated frame.
#[derive(Debug, Clone, Default)]
pub struct EvaluatedImageRouting {
    stable_layers: Vec<(StableLayerId, usize)>,
    mattes: Vec<EvaluatedLayerMatte>,
    taps: Vec<ImageTapRequest>,
    resource_plan: Option<MatteResourcePlan>,
}

impl EvaluatedImageRouting {
    pub fn stable_layers(&self) -> &[(StableLayerId, usize)] {
        &self.stable_layers
    }

    pub fn mattes(&self) -> &[EvaluatedLayerMatte] {
        &self.mattes
    }

    pub fn taps(&self) -> &[ImageTapRequest] {
        &self.taps
    }

    pub fn resource_plan(&self) -> Option<MatteResourcePlan> {
        self.resource_plan
    }

    pub fn is_active(&self) -> bool {
        self.resource_plan.is_some()
    }
}

/// One complete, owned frame program. There is no interior mutability and all
/// accessors return shared references, so consumers cannot make transport,
/// direct rendering, selective rendering, and export observe different
/// evaluations of the same modulation frame.
#[derive(Debug, Clone)]
pub struct EvaluatedFramePlan {
    context: FramePlanContext,
    master_transform: SpatialTransform,
    master_pass: EffectPassUniforms,
    ntsc: NtscParams,
    temporal: TemporalParams,
    layer_passes: Vec<EffectPassUniforms>,
    layer_pre_passes: Vec<EffectPassUniforms>,
    layers: Vec<EvaluatedLayer>,
    image_routing: EvaluatedImageRouting,
}

impl EvaluatedFramePlan {
    /// Resolve a post-morph frame into one immutable program.
    ///
    /// Work and allocation are bounded by the actual iterator length. Authored
    /// routing target indices have already been projected into the bounded
    /// [`ModulationFrame`], so a hostile `layer184467...` target cannot size
    /// either vector here.
    pub fn evaluate<'a>(
        modulation: &ModulationFrame,
        context: FramePlanContext,
        master: MasterFrameInput<'a>,
        layers: impl IntoIterator<Item = LayerFrameInput<'a>>,
    ) -> Self {
        let (mut master_effects, master_transform, ntsc, temporal) = modulation.modulate(
            master.effects,
            master.transform,
            master.ntsc,
            master.temporal,
        );
        master_effects.time = context.time_seconds;
        let master_pass = EffectPassUniforms::for_target(
            master_effects,
            master_transform,
            (context.output_size[0], context.output_size[1]),
            (context.output_size[0], context.output_size[1]),
        );

        let iterator = layers.into_iter();
        let (lower, upper) = iterator.size_hint();
        // Trust an exact upper bound only when it agrees with the lower bound;
        // otherwise reserve the guaranteed minimum and grow normally. This
        // avoids proportional allocation from a dishonest custom iterator.
        let capacity = upper
            .filter(|upper| *upper == lower)
            .unwrap_or(lower)
            .min(MAX_LAYER_PREALLOCATION);
        let mut layer_passes = Vec::with_capacity(capacity);
        let mut layer_pre_passes = Vec::with_capacity(capacity);
        let mut evaluated_layers = Vec::with_capacity(capacity);

        for (index, input) in iterator.enumerate() {
            let modulated = modulation.modulate_layer(
                index,
                input.effects,
                input.transform,
                input.opacity,
                input.mosh_send,
                input.speed,
                input.fps,
            );
            let mut effects = modulated.effects;
            effects.time = context.time_seconds;
            let pass = EffectPassUniforms::for_target(
                effects,
                modulated.transform,
                (input.source.size[0], input.source.size[1]),
                (context.output_size[0], context.output_size[1]),
            );
            let pre_effects = EffectUniforms {
                time: context.time_seconds,
                ..EffectUniforms::default()
            };
            let pre_pass = EffectPassUniforms::for_target(
                pre_effects,
                modulated.transform,
                (input.source.size[0], input.source.size[1]),
                (context.output_size[0], context.output_size[1]),
            );
            layer_passes.push(pass);
            layer_pre_passes.push(pre_pass);
            evaluated_layers.push(EvaluatedLayer {
                source: input.source,
                transform: modulated.transform,
                spatial: pass.spatial,
                opacity: modulated.opacity,
                mosh_send: modulated.mosh_send,
                speed: modulated.speed,
                fps: modulated.fps,
                blend_mode: input.blend_mode,
                visible: input.visible,
                paused: input.paused,
                bypass_master_fx: input.bypass_master_fx,
                bypass_temporal_fx: input.bypass_temporal_fx,
                pattern: input
                    .pattern
                    .map(|base| modulation.modulate_layer_pattern(index, base)),
            });
        }

        debug_assert_eq!(layer_passes.len(), evaluated_layers.len());
        debug_assert_eq!(layer_pre_passes.len(), evaluated_layers.len());
        Self {
            context,
            master_transform,
            master_pass,
            ntsc,
            temporal,
            layer_passes,
            layer_pre_passes,
            layers: evaluated_layers,
            image_routing: EvaluatedImageRouting::default(),
        }
    }

    /// Build the immutable Milestone 2 composition/rack program around this
    /// already evaluated M0/M1 frame. Callers pass frame-local runtime values
    /// after Morph/stable modulation projection; this planner never resamples
    /// mutable authored state.
    pub fn plan_composition(
        &self,
        input: evaluated_composition::CompositionPlanInput<'_>,
    ) -> Result<
        evaluated_composition::EvaluatedCompositionPlan,
        evaluated_composition::CompositionPlanError,
    > {
        evaluated_composition::EvaluatedCompositionPlan::evaluate(self, input)
    }

    /// Attach authored mattes without sampling modulation a second time.
    /// Resolution is transactional: an invalid count, identity, or resource
    /// request leaves the exact bare/legacy plan untouched for safe fallback.
    pub fn attach_image_routing(
        &mut self,
        mattes: impl IntoIterator<Item = LayerMatte>,
        program_history_initialized: bool,
    ) -> Result<(), String> {
        let mattes: Vec<LayerMatte> = mattes.into_iter().collect();
        if mattes.len() != self.layers.len() {
            return Err(format!(
                "image-routing matte count {} does not match evaluated layer count {}",
                mattes.len(),
                self.layers.len()
            ));
        }

        let mut stable_layers = Vec::with_capacity(self.layers.len());
        let mut stable_to_index = BTreeMap::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let Some(stable_id) = StableLayerId::new(layer.source.stable_id) else {
                return Err(format!(
                    "evaluated layer at index {index} has invalid zero stable identity"
                ));
            };
            if let Some(previous) = stable_to_index.insert(stable_id, index) {
                return Err(format!(
                    "evaluated layers {previous} and {index} share stable identity {}",
                    stable_id.get()
                ));
            }
            stable_layers.push((stable_id, index));
        }
        let available_layers: Vec<StableLayerId> =
            stable_layers.iter().map(|(id, _)| *id).collect();

        let mut tap_lookup: BTreeMap<(usize, u8), usize> = BTreeMap::new();
        let mut taps: Vec<ImageTapRequest> = Vec::new();
        let mut evaluated_mattes = Vec::with_capacity(mattes.len());
        let mut any_enabled = false;

        for (target_index, matte) in mattes.into_iter().enumerate() {
            let authored = matte.sanitized();
            let context = ImageRouteContext {
                available_layers: &available_layers,
                has_one_below: target_index + 1 < self.layers.len(),
                program_history_initialized,
            };
            let diagnostic = authored.diagnose(&context);
            any_enabled |= authored.enabled;

            let mut donor_valid = diagnostic == ImageRouteDiagnostic::Ready;
            let resolved_input = if !authored.enabled {
                donor_valid = false;
                ResolvedImageInput::Disabled
            } else if diagnostic != ImageRouteDiagnostic::Ready {
                donor_valid = false;
                ResolvedImageInput::Transparent
            } else {
                match authored.input {
                    ImageInput::SelectedLayer {
                        layer_id, stage, ..
                    } => {
                        let donor_layer_index = stable_to_index[&layer_id];
                        let tap_index = materialized_tap(
                            &mut tap_lookup,
                            &mut taps,
                            donor_layer_index,
                            stage,
                            target_index,
                        )?;
                        ResolvedImageInput::MaterializedTap { tap_index }
                    }
                    ImageInput::OneBelow => {
                        let tap_index = materialized_tap(
                            &mut tap_lookup,
                            &mut taps,
                            target_index + 1,
                            LayerImageStage::PostLocalEffects,
                            target_index,
                        )?;
                        ResolvedImageInput::MaterializedTap { tap_index }
                    }
                    ImageInput::AllBelow => ResolvedImageInput::AllBelow,
                    ImageInput::ProgramHistory => ResolvedImageInput::ProgramHistory,
                    // These variants cannot diagnose Ready in Milestone 1.
                    ImageInput::MissingSelectedLayer { .. }
                    | ImageInput::CleanProgram
                    | ImageInput::GroupOutput { .. }
                    | ImageInput::MissingGroupOutput { .. } => {
                        donor_valid = false;
                        ResolvedImageInput::Transparent
                    }
                }
            };
            evaluated_mattes.push(EvaluatedLayerMatte {
                authored,
                resolved_input,
                params: ResolvedMatteParams {
                    channel: matte_channel_code(authored.channel),
                    invert: authored.invert,
                    amount: authored.amount,
                    threshold: authored.threshold,
                    softness: authored.softness,
                    donor_valid,
                },
                diagnostic,
            });
        }

        let resource_plan = if any_enabled {
            Some(MatteResourcePlan::validate(
                self.context.output_size,
                taps.len(),
                MatteResourceLimits {
                    max_texture_dimension_2d: u32::MAX,
                    max_texture_array_layers: MAX_MATERIALIZED_IMAGE_TAPS,
                    max_sampled_textures_per_shader_stage: 3,
                    max_bytes: MAX_IMAGE_TAP_BYTES,
                },
            )?)
        } else {
            None
        };

        self.image_routing = EvaluatedImageRouting {
            stable_layers,
            mattes: evaluated_mattes,
            taps,
            resource_plan,
        };
        Ok(())
    }

    /// Consuming convenience used by pure tests and offline frame builders.
    #[cfg(test)]
    pub fn with_image_routing(
        mut self,
        mattes: impl IntoIterator<Item = LayerMatte>,
        program_history_initialized: bool,
    ) -> Result<Self, String> {
        self.attach_image_routing(mattes, program_history_initialized)?;
        Ok(self)
    }

    pub fn context(&self) -> FramePlanContext {
        self.context
    }

    pub fn master_transform(&self) -> &SpatialTransform {
        &self.master_transform
    }

    /// Exact uniform block for the shared master shader pass.
    pub fn master_pass_uniforms(&self) -> EffectPassUniforms {
        self.master_pass
    }

    /// Borrow the exact stored master payload. Production renderers use this
    /// accessor so frame encoding cannot accidentally reconstruct the block
    /// from subsequently changed authored state.
    pub fn master_pass(&self) -> &EffectPassUniforms {
        &self.master_pass
    }

    pub fn ntsc(&self) -> &NtscParams {
        &self.ntsc
    }

    pub fn temporal(&self) -> &TemporalParams {
        &self.temporal
    }

    /// Exact uniform block for one shared layer shader pass. Independent live
    /// and export consumers can build their own plans and compare these bytes
    /// without copying evaluation logic into either renderer.
    pub fn layer_pass_uniforms(&self, index: usize) -> Option<EffectPassUniforms> {
        self.layer_passes.get(index).copied()
    }

    /// All exact stored layer payloads in source/UI order, index-aligned with
    /// [`Self::layers`].
    pub fn layer_passes(&self) -> &[EffectPassUniforms] {
        &self.layer_passes
    }

    /// Neutral local-effects pass retaining the exact evaluated spatial map.
    /// This materializes a selected PreLocalEffects tap composition-aligned at
    /// output size without applying the donor's local image effects.
    pub fn layer_pre_passes(&self) -> &[EffectPassUniforms] {
        &self.layer_pre_passes
    }

    pub fn layers(&self) -> &[EvaluatedLayer] {
        &self.layers
    }

    pub fn image_routing(&self) -> &EvaluatedImageRouting {
        &self.image_routing
    }

    /// Advanced composition owns M1 mattes in its unified dependency/resource
    /// ledger. Its embedded base must therefore not retain an independently
    /// allocatable M1 routing plan.
    pub(super) fn clone_without_image_routing(&self) -> Self {
        let mut cloned = self.clone();
        cloned.image_routing = EvaluatedImageRouting::default();
        cloned
    }
}

fn stage_key(stage: LayerImageStage) -> u8 {
    match stage {
        LayerImageStage::PreLocalEffects => 0,
        LayerImageStage::PostLocalEffects => 1,
    }
}

fn materialized_tap(
    lookup: &mut BTreeMap<(usize, u8), usize>,
    taps: &mut Vec<ImageTapRequest>,
    donor_layer_index: usize,
    stage: LayerImageStage,
    consumer: usize,
) -> Result<usize, String> {
    let key = (donor_layer_index, stage_key(stage));
    if let Some(&index) = lookup.get(&key) {
        taps[index].consumers.push(consumer);
        return Ok(index);
    }
    let index = taps.len();
    let array_layer = u32::try_from(index)
        .map_err(|_| "image tap index does not fit a GPU array layer".to_string())?;
    if array_layer >= MAX_MATERIALIZED_IMAGE_TAPS {
        return Err(format!(
            "image routing exceeds the hard limit of {MAX_MATERIALIZED_IMAGE_TAPS} materialized taps"
        ));
    }
    taps.push(ImageTapRequest {
        donor_layer_index,
        stage,
        array_layer,
        consumers: vec![consumer],
    });
    lookup.insert(key, index);
    Ok(index)
}

fn matte_channel_code(channel: MatteChannel) -> MatteChannelCode {
    match channel {
        MatteChannel::Alpha => MatteChannelCode::Alpha,
        MatteChannel::Luma => MatteChannelCode::Luma,
        MatteChannel::Red => MatteChannelCode::Red,
        MatteChannel::Green => MatteChannelCode::Green,
        MatteChannel::Blue => MatteChannelCode::Blue,
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::modulation::{ModMatrix, ModSource, Routing};
    use crate::spatial::{EdgeMode, FitMode};

    #[derive(Clone)]
    struct OwnedLayerBase {
        effects: EffectUniforms,
        transform: SpatialTransform,
        opacity: f32,
        mosh_send: f32,
        speed: f32,
        fps: f32,
        blend_mode: BlendMode,
        visible: bool,
        paused: bool,
        bypass_master_fx: bool,
        bypass_temporal_fx: bool,
        source: SourceTap,
    }

    fn layer_input(base: &OwnedLayerBase) -> LayerFrameInput<'_> {
        LayerFrameInput {
            source: base.source,
            effects: &base.effects,
            transform: &base.transform,
            opacity: base.opacity,
            mosh_send: base.mosh_send,
            speed: base.speed,
            fps: base.fps,
            blend_mode: base.blend_mode,
            visible: base.visible,
            paused: base.paused,
            bypass_master_fx: base.bypass_master_fx,
            bypass_temporal_fx: base.bypass_temporal_fx,
            pattern: None,
        }
    }

    fn fixture_layers(count: usize) -> Vec<OwnedLayerBase> {
        (0..count)
            .map(|index| OwnedLayerBase {
                effects: EffectUniforms {
                    hue_shift: index as f32 * 7.0,
                    cellular_amount: 0.2 + index as f32 * 0.03,
                    ..EffectUniforms::default()
                },
                transform: SpatialTransform {
                    position: [index as f32 * 0.025 - 0.08, index as f32 * -0.015],
                    scale: [1.0 + index as f32 * 0.04, 0.95 + index as f32 * 0.02],
                    rotation_deg: index as f32 * 11.0,
                    skew_deg: index as f32 - 3.0,
                    fit: if index % 2 == 0 {
                        FitMode::Fit
                    } else {
                        FitMode::Fill
                    },
                    edge: EdgeMode::Transparent,
                    ..SpatialTransform::new_layer_default()
                },
                opacity: 0.55 + index as f32 * 0.05,
                mosh_send: 1.0 - index as f32 * 0.05,
                speed: 0.75 + index as f32 * 0.1,
                fps: 24.0 + index as f32,
                blend_mode: match index % 4 {
                    0 => BlendMode::Normal,
                    1 => BlendMode::Screen,
                    2 => BlendMode::Multiply,
                    _ => BlendMode::Difference,
                },
                visible: index != 6,
                paused: index == 7,
                bypass_master_fx: index % 3 == 0,
                bypass_temporal_fx: index % 4 == 0,
                source: SourceTap::new(100 + index as u64, index, 640 + index as u32, 480),
            })
            .collect()
    }

    fn master_input<'a>(
        effects: &'a EffectUniforms,
        transform: &'a SpatialTransform,
        ntsc: &'a NtscParams,
        temporal: &'a TemporalParams,
    ) -> MasterFrameInput<'a> {
        MasterFrameInput {
            effects,
            transform,
            ntsc,
            temporal,
        }
    }

    #[test]
    fn one_plan_owns_aligned_render_transport_and_source_values() {
        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 1.0;
        matrix.routings = vec![
            Routing::new(ModSource::Midi(0), "brightness", 0.25),
            Routing::new(ModSource::Midi(0), "position_x", 0.1),
            Routing::new(ModSource::Midi(0), "layer2_opacity", -0.5),
            Routing::new(ModSource::Midi(0), "layer2_mosh_send", -0.4),
            Routing::new(ModSource::Midi(0), "layer2_speed", 0.2),
            Routing::new(ModSource::Midi(0), "layer2_rotation_deg", 0.1),
        ];
        matrix.update_at_beat(3.25, 1.0 / 60.0);

        let layers = fixture_layers(3);
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams::default();
        let temporal = TemporalParams::default();
        let frame = matrix.frame(layers.len());
        let plan = EvaluatedFramePlan::evaluate(
            &frame,
            FramePlanContext::new(1920, 1080, 4.25),
            master_input(&master_effects, &master_transform, &ntsc, &temporal),
            layers.iter().map(layer_input),
        );

        assert_eq!(plan.context().output_size, [1920, 1080]);
        assert_eq!(plan.layers().len(), 3);
        assert_eq!(plan.layer_passes().len(), 3);
        assert_eq!(plan.layers()[1].source.stable_id, 101);
        assert_eq!(plan.layers()[1].source.slot, 1);
        assert!(plan.layers()[0].bypass_master_fx);
        assert!(plan.layers()[0].bypass_temporal_fx);
        assert!(!plan.layers()[1].bypass_temporal_fx);
        assert!(plan.layers()[1].speed > layers[1].speed);
        assert!(plan.layers()[1].opacity < layers[1].opacity);
        assert!(plan.layers()[1].mosh_send < layers[1].mosh_send);
        assert_ne!(
            plan.layers()[1].transform.rotation_deg,
            layers[1].transform.rotation_deg
        );
        assert_eq!(plan.master_pass().effects.time, 4.25);
        assert_eq!(plan.layer_passes()[0].effects.time, 4.25);
    }

    #[test]
    fn authored_layer_mutation_after_evaluation_cannot_change_effective_render_payload() {
        let matrix = ModMatrix::new();
        let mut layers = fixture_layers(1);
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams::default();
        let temporal = TemporalParams::default();
        let frame = matrix.frame(1);
        let plan = EvaluatedFramePlan::evaluate(
            &frame,
            FramePlanContext::new(1920, 1080, 2.0),
            master_input(&master_effects, &master_transform, &ntsc, &temporal),
            layers.iter().map(layer_input),
        );
        let pass_before = bytemuck::bytes_of(plan.layer_passes().first().unwrap()).to_vec();
        let evaluated_before = plan.layers()[0];

        // These are precisely the authored fields that the historical live
        // renderer reread after evaluation. The plan must remain the complete
        // render authority even if UI/transport state changes afterward.
        layers[0].effects.brightness = 0.91;
        layers[0].transform = SpatialTransform {
            position: [0.75, -0.5],
            rotation_deg: 133.0,
            ..SpatialTransform::new_layer_default()
        };
        layers[0].opacity = 0.01;
        layers[0].mosh_send = 0.0;
        layers[0].blend_mode = BlendMode::Difference;
        layers[0].visible = false;
        layers[0].paused = true;
        layers[0].bypass_master_fx = !layers[0].bypass_master_fx;
        layers[0].bypass_temporal_fx = !layers[0].bypass_temporal_fx;
        layers[0].source = SourceTap::new(999, 42, 32, 16);

        assert_eq!(
            bytemuck::bytes_of(plan.layer_passes().first().unwrap()),
            pass_before.as_slice()
        );
        assert_eq!(plan.layers()[0], evaluated_before);
    }

    #[test]
    fn invalid_context_and_extreme_source_identity_do_not_size_the_plan() {
        let matrix = ModMatrix::new();
        let mut layers = fixture_layers(1);
        layers[0].source = SourceTap::new(u64::MAX, usize::MAX, 0, 0);
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams::default();
        let temporal = TemporalParams::default();
        let frame = matrix.frame(1);
        let plan = EvaluatedFramePlan::evaluate(
            &frame,
            FramePlanContext::new(0, 0, f32::NAN),
            master_input(&master_effects, &master_transform, &ntsc, &temporal),
            layers.iter().map(layer_input),
        );

        assert_eq!(plan.context(), FramePlanContext::new(1, 1, 0.0));
        assert_eq!(plan.layers().len(), 1);
        assert_eq!(plan.layers()[0].source.size, [1, 1]);
        assert_eq!(plan.layers()[0].source.slot, usize::MAX);
        assert_eq!(plan.layers()[0].source.stable_id, u64::MAX);
    }

    #[test]
    fn live_action_sequence_is_additive_and_offline_context_stays_uncorrelated() {
        let offline = FramePlanContext::new(1280, 720, 1.0);
        assert_eq!(offline.highest_applied_action_sequence, 0);
        let live = offline.with_highest_applied_action_sequence(47);
        assert_eq!(live.highest_applied_action_sequence, 47);
        assert_eq!(live.output_size, offline.output_size);
        assert_eq!(live.time_seconds, offline.time_seconds);
    }

    #[test]
    fn deterministic_eight_layer_1080p60_cpu_planning_gate() {
        const FPS: u32 = 60;
        const FRAMES: u32 = FPS * 10;
        // This is deliberately the CPU-planner portion of acceptance gate 11,
        // not a claim about GPU fill rate. Ten simulated seconds must plan in
        // less than ten wall seconds, keeping evaluation below one 60 Hz frame
        // period on average even in an unoptimized build. The complementary
        // opt-in offscreen GPU gate measures actual transformed rendering.
        const DEADLINE: Duration = Duration::from_secs(10);

        let mut matrix = ModMatrix::new();
        matrix.midi[0] = 0.8;
        matrix.routings = (0..8)
            .flat_map(|index| {
                [
                    Routing::new(
                        ModSource::Midi(0),
                        format!("layer{}_rotation_deg", index + 1),
                        0.15,
                    ),
                    Routing::new(
                        ModSource::Midi(0),
                        format!("layer{}_position_x", index + 1),
                        0.08,
                    ),
                ]
            })
            .collect();
        let layers = fixture_layers(8);
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams::default();
        let temporal = TemporalParams::default();
        let started = Instant::now();
        let initial_checksum = 0xcbf2_9ce4_8422_2325u64;
        let mut checksum = initial_checksum;

        for frame_index in 0..FRAMES {
            let time = frame_index as f32 / FPS as f32;
            matrix.update_at_beat(time as f64 * 2.0, 1.0 / FPS as f32);
            let frame = matrix.frame(layers.len());
            let plan = EvaluatedFramePlan::evaluate(
                &frame,
                FramePlanContext::new(1920, 1080, time),
                master_input(&master_effects, &master_transform, &ntsc, &temporal),
                layers.iter().map(layer_input),
            );
            assert_eq!(plan.context().output_size, [1920, 1080]);
            assert_eq!(plan.layers().len(), 8);
            assert_eq!(plan.layer_passes().len(), 8);
            checksum = checksum
                .wrapping_mul(0x100_0000_01b3)
                .wrapping_add(plan.layers()[frame_index as usize % 8].source.stable_id)
                .wrapping_add(u64::from(
                    plan.layers()[0].spatial.inverse_row_0[0].to_bits(),
                ));
            black_box(&plan);
        }

        assert_ne!(
            checksum, initial_checksum,
            "the benchmark loop must consume evaluated data"
        );
        assert!(
            started.elapsed() < DEADLINE,
            "eight-layer 1080p60 CPU planning exceeded {DEADLINE:?} for {FRAMES} frames"
        );
    }

    fn routed_fixture(layers: &[OwnedLayerBase], mattes: Vec<LayerMatte>) -> EvaluatedFramePlan {
        routed_fixture_with_history(layers, mattes, false)
    }

    fn routed_fixture_with_history(
        layers: &[OwnedLayerBase],
        mattes: Vec<LayerMatte>,
        history_ready: bool,
    ) -> EvaluatedFramePlan {
        let matrix = ModMatrix::new();
        let frame = matrix.frame(layers.len());
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams::default();
        let temporal = TemporalParams::default();
        EvaluatedFramePlan::evaluate(
            &frame,
            FramePlanContext::new(320, 180, 1.0),
            master_input(&master_effects, &master_transform, &ntsc, &temporal),
            layers.iter().map(layer_input),
        )
        .with_image_routing(mattes, history_ready)
        .unwrap()
    }

    fn stable_id(value: u64) -> StableLayerId {
        StableLayerId::new(value).unwrap()
    }

    #[test]
    fn image_routing_resolves_stable_ids_and_deduplicates_tap_dependencies() {
        let layers = fixture_layers(3);
        let selected_post = LayerMatte {
            enabled: true,
            input: ImageInput::SelectedLayer {
                layer_id: stable_id(102),
                stage: LayerImageStage::PostLocalEffects,
            },
            channel: MatteChannel::Luma,
            invert: true,
            amount: 0.75,
            threshold: 0.4,
            softness: 0.2,
        };
        let plan = routed_fixture(
            &layers,
            vec![selected_post, LayerMatte::default(), selected_post],
        );

        assert_eq!(
            plan.image_routing().stable_layers(),
            &[
                (stable_id(100), 0),
                (stable_id(101), 1),
                (stable_id(102), 2)
            ]
        );
        assert_eq!(plan.image_routing().taps().len(), 1);
        let tap = &plan.image_routing().taps()[0];
        assert_eq!(tap.donor_layer_index, 2);
        assert_eq!(tap.stage, LayerImageStage::PostLocalEffects);
        assert_eq!(tap.consumers, vec![0, 2]);
        assert!(matches!(
            plan.image_routing().mattes()[0].resolved_input,
            ResolvedImageInput::MaterializedTap { tap_index: 0 }
        ));
        assert_eq!(
            plan.image_routing().mattes()[0].diagnostic,
            ImageRouteDiagnostic::Ready
        );
    }

    #[test]
    fn missing_cycle_bottom_and_uninitialized_history_are_transparent_and_visible() {
        let layers = fixture_layers(4);
        let enabled = |input| LayerMatte {
            enabled: true,
            input,
            ..LayerMatte::default()
        };
        let plan = routed_fixture(
            &layers,
            vec![
                enabled(ImageInput::CleanProgram),
                enabled(ImageInput::ProgramHistory),
                enabled(ImageInput::GroupOutput {
                    group_id: crate::visual_rack::GroupId::new(7).unwrap(),
                }),
                enabled(ImageInput::AllBelow),
            ],
        );
        assert!(plan.image_routing().taps().is_empty());
        for matte in plan.image_routing().mattes() {
            assert_eq!(matte.resolved_input, ResolvedImageInput::Transparent);
            assert_ne!(matte.diagnostic, ImageRouteDiagnostic::Ready);
            assert!(!matte.params.donor_valid);
        }
        assert!(plan.image_routing().mattes()[0]
            .diagnostic
            .to_string()
            .contains("cycle"));
        assert!(plan.image_routing().mattes()[2]
            .diagnostic
            .to_string()
            .contains("unavailable"));
        assert!(plan.image_routing().mattes()[3]
            .diagnostic
            .to_string()
            .contains("no layers below"));

        let ready = routed_fixture_with_history(
            &layers,
            vec![
                enabled(ImageInput::ProgramHistory),
                LayerMatte::default(),
                LayerMatte::default(),
                LayerMatte::default(),
            ],
            true,
        );
        assert_eq!(
            ready.image_routing().mattes()[0].resolved_input,
            ResolvedImageInput::ProgramHistory
        );
        assert_eq!(
            ready.image_routing().mattes()[0].diagnostic,
            ImageRouteDiagnostic::Ready
        );
        assert!(ready.image_routing().mattes()[0].params.donor_valid);
    }

    #[test]
    fn pre_local_tap_is_composition_aligned_but_effect_neutral() {
        let layers = fixture_layers(2);
        let plan = routed_fixture(
            &layers,
            vec![
                LayerMatte {
                    enabled: true,
                    input: ImageInput::SelectedLayer {
                        layer_id: stable_id(101),
                        stage: LayerImageStage::PreLocalEffects,
                    },
                    ..LayerMatte::default()
                },
                LayerMatte::default(),
            ],
        );
        assert_eq!(
            plan.layer_pre_passes()[1].spatial,
            plan.layer_passes()[1].spatial
        );
        assert_eq!(
            plan.layer_pre_passes()[1].effects.hue_shift,
            EffectUniforms::default().hue_shift
        );
        assert_ne!(
            plan.layer_pre_passes()[1].effects.hue_shift,
            plan.layer_passes()[1].effects.hue_shift
        );
    }

    #[test]
    fn routed_plan_rejects_duplicate_stable_identity_and_unbounded_taps() {
        let mut duplicate = fixture_layers(2);
        duplicate[1].source.stable_id = duplicate[0].source.stable_id;
        let mut duplicate_plan = {
            let matrix = ModMatrix::new();
            let frame = matrix.frame(2);
            let master_effects = EffectUniforms::default();
            let master_transform = SpatialTransform::default();
            let ntsc = NtscParams::default();
            let temporal = TemporalParams::default();
            EvaluatedFramePlan::evaluate(
                &frame,
                FramePlanContext::new(320, 180, 1.0),
                master_input(&master_effects, &master_transform, &ntsc, &temporal),
                duplicate.iter().map(layer_input),
            )
        };
        let pass_before = duplicate_plan
            .layer_passes()
            .iter()
            .flat_map(|pass| bytemuck::bytes_of(pass).iter().copied())
            .collect::<Vec<_>>();
        let error = duplicate_plan
            .attach_image_routing(vec![LayerMatte::default(); 2], false)
            .unwrap_err();
        assert!(error.contains("share stable identity"));
        assert!(!duplicate_plan.image_routing().is_active());
        let pass_after = duplicate_plan
            .layer_passes()
            .iter()
            .flat_map(|pass| bytemuck::bytes_of(pass).iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(pass_after, pass_before);

        let layers = fixture_layers(65);
        let mattes = layers
            .iter()
            .map(|layer| LayerMatte {
                enabled: true,
                input: ImageInput::SelectedLayer {
                    layer_id: stable_id(layer.source.stable_id),
                    stage: LayerImageStage::PreLocalEffects,
                },
                ..LayerMatte::default()
            })
            .collect::<Vec<_>>();
        let matrix = ModMatrix::new();
        let frame = matrix.frame(layers.len());
        let master_effects = EffectUniforms::default();
        let master_transform = SpatialTransform::default();
        let ntsc = NtscParams::default();
        let temporal = TemporalParams::default();
        let error = EvaluatedFramePlan::evaluate(
            &frame,
            FramePlanContext::new(320, 180, 1.0),
            master_input(&master_effects, &master_transform, &ntsc, &temporal),
            layers.iter().map(layer_input),
        )
        .with_image_routing(mattes, false)
        .unwrap_err();
        assert!(error.contains("hard limit"));
    }
}
