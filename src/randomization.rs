//! Deterministic primitives shared by offline generation and live rerolls.
//!
//! The live controls deliberately avoid entropy from clocks or the operating
//! system. A performer can therefore replay an exact seed, and an omitted
//! seed advances a stable sequence from the state already stored in a patch.

use crate::composition::RuntimeComposition;
use crate::effects::params::TemporalOriginalsParams;
use crate::effects::EffectUniforms;
use crate::image_routing::StableLayerId;
use crate::motion::MotionParams;
use crate::spatial::{
    SpatialTransform, ANCHOR_MAX, ANCHOR_MIN, CROP_MAX, POSITION_MAX, POSITION_MIN, SCALE_MAX,
    SCALE_MIN, SKEW_LIMIT_DEGREES,
};
use crate::visual_rack::{
    GroupId, NodeId, RuntimeMaskParams, RuntimeVisualNodeKind, RuntimeVisualRack,
};

#[derive(Clone, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
    }

    pub(crate) fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    pub(crate) fn chance(&mut self, probability: f32) -> bool {
        self.unit() < probability.clamp(0.0, 1.0)
    }
}

/// The same 32-bit avalanche used by the GPU shader.
pub(crate) const fn avalanche32(mut value: u32) -> u32 {
    value = (value ^ (value >> 16)).wrapping_mul(0x7feb_352d);
    value = (value ^ (value >> 15)).wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

/// Advance a stored seed without consulting wall time or external entropy.
/// The automatically generated sequence never returns the legacy sentinel 0.
pub(crate) fn next_seed(current: u32) -> u32 {
    let candidate = avalanche32(current.wrapping_add(0x9e37_79b9));
    if candidate == 0 {
        1
    } else {
        candidate
    }
}

pub(crate) fn advance_seed(mut current: u32, count: u64) -> u32 {
    for _ in 0..count {
        current = next_seed(current);
    }
    current
}

/// Consume authoritative decoder loop boundaries at the live render seam.
///
/// `loops_advanced` is already de-duplicated by [`crate::video::ThreadedDecoder`]
/// from its cumulative loop generation, so advancing by the reported count
/// preserves every crossed boundary even when its latest-only mailbox drops
/// intermediate images.
pub(crate) fn apply_live_loop_reroll(
    effects: &mut EffectUniforms,
    reroll_on_loop: bool,
    loops_advanced: u64,
) {
    if reroll_on_loop && loops_advanced > 0 {
        effects.random_seed = advance_seed(effects.random_seed, loops_advanced);
    }
}

/// Derive an independent target stream from a user-supplied base. Explicit
/// zero remains zero so a performer can deliberately restore legacy patterns.
pub(crate) fn stream_seed(base: u32, stream: u64) -> u32 {
    if base == 0 {
        return 0;
    }
    let mut rng = SplitMix64::new(u64::from(base) ^ stream.wrapping_mul(0xd6e8_feb8_6659_fd93));
    let candidate = rng.next_u64() as u32;
    if candidate == 0 {
        1
    } else {
        candidate
    }
}

/// Reflect at both endpoints instead of clamping, avoiding probability mass
/// accumulating at slider limits during repeated generation or rerolls.
pub(crate) fn reflect(mut value: f32, min: f32, max: f32) -> f32 {
    if !value.is_finite() || min >= max {
        return min;
    }
    let span = max - min;
    value = (value - min) % (2.0 * span);
    if value < 0.0 {
        value += 2.0 * span;
    }
    if value > span {
        value = 2.0 * span - value;
    }
    min + value
}

pub(crate) fn wrap(value: f32, min: f32, max: f32) -> f32 {
    let span = max - min;
    if !value.is_finite() || span <= 0.0 {
        return min;
    }
    (value - min).rem_euclid(span) + min
}

const MEAN_REVERSION: f32 = 0.85;

pub(crate) fn mutate_linear(
    anchor: f32,
    value: f32,
    min: f32,
    max: f32,
    scale: f32,
    rng: &mut SplitMix64,
) -> f32 {
    reflect(
        anchor + MEAN_REVERSION * (value - anchor) + rng.signed() * scale,
        min,
        max,
    )
}

pub(crate) fn mutate_log(
    anchor: f32,
    value: f32,
    min: f32,
    max: f32,
    scale: f32,
    rng: &mut SplitMix64,
) -> f32 {
    let anchor = anchor.clamp(min, max).ln();
    let value = value.clamp(min, max).ln();
    reflect(
        anchor + MEAN_REVERSION * (value - anchor) + rng.signed() * scale,
        min.ln(),
        max.ln(),
    )
    .exp()
}

fn circular_delta(value: f32, anchor: f32, period: f32) -> f32 {
    (value - anchor + period * 0.5).rem_euclid(period) - period * 0.5
}

pub(crate) fn mutate_circular(
    anchor: f32,
    value: f32,
    min: f32,
    max: f32,
    scale: f32,
    rng: &mut SplitMix64,
) -> f32 {
    let period = max - min;
    wrap(
        anchor + MEAN_REVERSION * circular_delta(value, anchor, period) + rng.signed() * scale,
        min,
        max,
    )
}

pub(crate) fn mutate_discrete<T: Copy + PartialEq>(
    anchor: T,
    value: T,
    choices: &[T],
    change_probability: f32,
    rng: &mut SplitMix64,
) -> T {
    if value != anchor && rng.chance(0.15) {
        anchor
    } else if !choices.is_empty() && rng.chance(change_probability) {
        choices[(rng.next_u64() % choices.len() as u64) as usize]
    } else {
        value
    }
}

/// Bounded, live-safe visual mutation. Alpha keys, visibility, blend, source
/// identity, and every transport field are intentionally outside this API.
pub(crate) fn mutate_live_effects(
    effects: &mut EffectUniforms,
    amount: f32,
    include_grain_controls: bool,
    seed: u32,
    stream: u64,
) {
    let amount = if amount.is_finite() {
        amount.clamp(0.0, 2.0)
    } else {
        0.0
    };
    if amount == 0.0 {
        return;
    }

    let defaults = EffectUniforms::default();
    let mut rng = SplitMix64::new(u64::from(seed) ^ stream.wrapping_mul(0xa076_1d64_78bd_642f));
    // Shift was added after the original Dice field sequence. A domain-
    // separated stream keeps every pre-existing field byte-for-byte stable
    // for the same seed and stream while making Shift equally replayable.
    let mut shift_rng = SplitMix64::new(
        u64::from(seed) ^ stream.wrapping_mul(0xa076_1d64_78bd_642f) ^ 0x5348_4946_5400_0001,
    );
    effects.pixelate_size = mutate_log(
        defaults.pixelate_size,
        effects.pixelate_size,
        1.0,
        32.0,
        amount * 0.45,
        &mut rng,
    )
    .round();
    effects.downsample = mutate_log(
        defaults.downsample,
        effects.downsample,
        0.05,
        1.0,
        amount * 0.22,
        &mut rng,
    );
    effects.rgb_split = mutate_linear(
        defaults.rgb_split,
        effects.rgb_split,
        0.0,
        30.0,
        amount * 4.0,
        &mut rng,
    );
    effects.hue_shift = mutate_circular(
        defaults.hue_shift,
        effects.hue_shift,
        -180.0,
        180.0,
        amount * 50.0,
        &mut rng,
    );
    effects.saturation = mutate_linear(
        defaults.saturation,
        effects.saturation,
        -1.0,
        1.0,
        amount * 0.3,
        &mut rng,
    );
    effects.brightness = mutate_linear(
        defaults.brightness,
        effects.brightness,
        -1.0,
        1.0,
        amount * 0.25,
        &mut rng,
    );
    effects.contrast = mutate_linear(
        defaults.contrast,
        effects.contrast,
        -1.0,
        1.0,
        amount * 0.3,
        &mut rng,
    );
    effects.posterize = mutate_linear(
        defaults.posterize,
        effects.posterize,
        0.0,
        16.0,
        amount * 2.0,
        &mut rng,
    )
    .round();
    effects.grain_intensity = mutate_linear(
        defaults.grain_intensity,
        effects.grain_intensity,
        0.0,
        0.3,
        amount * 0.06,
        &mut rng,
    );
    effects.grain_size = mutate_linear(
        defaults.grain_size,
        effects.grain_size,
        1.0,
        4.0,
        amount * 0.5,
        &mut rng,
    );
    effects.vignette = mutate_linear(
        defaults.vignette,
        effects.vignette,
        0.0,
        1.5,
        amount * 0.22,
        &mut rng,
    );
    effects.color_drift = mutate_linear(
        defaults.color_drift,
        effects.color_drift,
        0.0,
        0.02,
        amount * 0.004,
        &mut rng,
    );
    effects.breathe_scale = mutate_linear(
        defaults.breathe_scale,
        effects.breathe_scale,
        0.0,
        0.05,
        amount * 0.01,
        &mut rng,
    );
    effects.breathe_rotation = mutate_linear(
        defaults.breathe_rotation,
        effects.breathe_rotation,
        0.0,
        2.0,
        amount * 0.35,
        &mut rng,
    );
    effects.breathe_position = mutate_linear(
        defaults.breathe_position,
        effects.breathe_position,
        0.0,
        0.02,
        amount * 0.004,
        &mut rng,
    );
    effects.cellular_amount = mutate_linear(
        defaults.cellular_amount,
        effects.cellular_amount,
        0.0,
        1.0,
        amount * 0.2,
        &mut rng,
    );
    effects.cellular_scale = mutate_log(
        defaults.cellular_scale,
        effects.cellular_scale,
        2.0,
        32.0,
        amount * 0.28,
        &mut rng,
    );
    effects.cellular_warp = mutate_linear(
        defaults.cellular_warp,
        effects.cellular_warp,
        0.0,
        1.0,
        amount * 0.18,
        &mut rng,
    );
    effects.cellular_speed = mutate_linear(
        defaults.cellular_speed,
        effects.cellular_speed,
        0.0,
        2.0,
        amount * 0.35,
        &mut rng,
    );

    if include_grain_controls {
        effects.grain_algo = mutate_discrete(
            defaults.grain_algo as u32,
            effects.grain_algo.round().clamp(0.0, 3.0) as u32,
            &[0, 1, 2, 3],
            (amount * 0.4).clamp(0.0, 0.8),
            &mut rng,
        ) as f32;
        effects.color_grain = if mutate_discrete(
            defaults.color_grain > 0.5,
            effects.color_grain > 0.5,
            &[false, true],
            (amount * 0.4).clamp(0.0, 0.8),
            &mut rng,
        ) {
            1.0
        } else {
            0.0
        };
    }

    effects.shift_amount = mutate_linear(
        defaults.shift_amount,
        effects.shift_amount,
        0.0,
        1.0,
        amount * 0.25,
        &mut shift_rng,
    );
    effects.shift_block_size = mutate_log(
        defaults.shift_block_size,
        effects.shift_block_size,
        2.0,
        256.0,
        amount * 0.35,
        &mut shift_rng,
    )
    .round();
    effects.shift_density = mutate_linear(
        defaults.shift_density,
        effects.shift_density,
        0.0,
        1.0,
        amount * 0.15,
        &mut shift_rng,
    );
    effects.shift_speed = mutate_linear(
        defaults.shift_speed,
        effects.shift_speed,
        0.0,
        20.0,
        amount * 2.5,
        &mut shift_rng,
    );

    // B13 small effects mutate in their own domain-separated stream so every
    // pre-B13 Dice result stays byte-for-byte stable for the same seed and
    // stream. `negative_mode` is a discrete law and is deliberately not
    // rerolled. The layer call sites clear the three master-only optics after
    // this returns, so their mutation reaches master scope only.
    let mut small_fx_rng = SplitMix64::new(
        u64::from(seed) ^ stream.wrapping_mul(0xa076_1d64_78bd_642f) ^ 0x534d_4c46_5800_0001,
    );
    effects.contour = mutate_linear(
        defaults.contour,
        effects.contour,
        0.0,
        1.0,
        amount * 0.18,
        &mut small_fx_rng,
    );
    effects.contour_bands = mutate_log(
        defaults.contour_bands,
        effects.contour_bands,
        2.0,
        40.0,
        amount * 0.25,
        &mut small_fx_rng,
    )
    .round();
    effects.contour_width = mutate_linear(
        defaults.contour_width,
        effects.contour_width,
        0.2,
        6.0,
        amount * 0.6,
        &mut small_fx_rng,
    );
    effects.contour_hue = mutate_linear(
        defaults.contour_hue,
        effects.contour_hue,
        0.0,
        1.0,
        amount * 0.2,
        &mut small_fx_rng,
    );
    effects.contour_fill = mutate_linear(
        defaults.contour_fill,
        effects.contour_fill,
        0.0,
        1.0,
        amount * 0.15,
        &mut small_fx_rng,
    );
    effects.flatten = mutate_linear(
        defaults.flatten,
        effects.flatten,
        0.0,
        1.0,
        amount * 0.18,
        &mut small_fx_rng,
    );
    effects.flatten_levels = mutate_log(
        defaults.flatten_levels,
        effects.flatten_levels,
        2.0,
        16.0,
        amount * 0.3,
        &mut small_fx_rng,
    )
    .round();
    effects.contour_dither = mutate_linear(
        defaults.contour_dither,
        effects.contour_dither,
        0.0,
        1.0,
        amount * 0.2,
        &mut small_fx_rng,
    );
    effects.solarize = mutate_linear(
        defaults.solarize,
        effects.solarize,
        0.0,
        1.0,
        amount * 0.15,
        &mut small_fx_rng,
    );
    effects.negative = mutate_linear(
        defaults.negative,
        effects.negative,
        0.0,
        1.0,
        amount * 0.12,
        &mut small_fx_rng,
    );
    effects.colourpass = mutate_linear(
        defaults.colourpass,
        effects.colourpass,
        0.0,
        1.0,
        amount * 0.15,
        &mut small_fx_rng,
    );
    effects.colourpass_hue = mutate_circular(
        defaults.colourpass_hue,
        effects.colourpass_hue,
        -180.0,
        180.0,
        amount * 40.0,
        &mut small_fx_rng,
    );
    effects.colourpass_width = mutate_linear(
        defaults.colourpass_width,
        effects.colourpass_width,
        0.0,
        1.0,
        amount * 0.15,
        &mut small_fx_rng,
    );
    effects.edge_amount = mutate_linear(
        defaults.edge_amount,
        effects.edge_amount,
        0.0,
        1.0,
        amount * 0.15,
        &mut small_fx_rng,
    );
    effects.edge_hue = mutate_circular(
        defaults.edge_hue,
        effects.edge_hue,
        -180.0,
        180.0,
        amount * 40.0,
        &mut small_fx_rng,
    );
    effects.emboss = mutate_linear(
        defaults.emboss,
        effects.emboss,
        0.0,
        1.0,
        amount * 0.12,
        &mut small_fx_rng,
    );
    effects.emboss_angle = mutate_circular(
        defaults.emboss_angle,
        effects.emboss_angle,
        -180.0,
        180.0,
        amount * 40.0,
        &mut small_fx_rng,
    );
    effects.halftone = mutate_linear(
        defaults.halftone,
        effects.halftone,
        0.0,
        1.0,
        amount * 0.15,
        &mut small_fx_rng,
    );
    effects.halftone_pitch = mutate_linear(
        defaults.halftone_pitch,
        effects.halftone_pitch,
        0.0,
        1.0,
        amount * 0.2,
        &mut small_fx_rng,
    );
    effects.halftone_angle = mutate_circular(
        defaults.halftone_angle,
        effects.halftone_angle,
        -180.0,
        180.0,
        amount * 40.0,
        &mut small_fx_rng,
    );
    effects.moire = mutate_linear(
        defaults.moire,
        effects.moire,
        0.0,
        1.0,
        amount * 0.12,
        &mut small_fx_rng,
    );
    effects.moire_freq = mutate_linear(
        defaults.moire_freq,
        effects.moire_freq,
        0.0,
        1.0,
        amount * 0.2,
        &mut small_fx_rng,
    );
    effects.row_smear = mutate_linear(
        defaults.row_smear,
        effects.row_smear,
        0.0,
        1.0,
        amount * 0.15,
        &mut small_fx_rng,
    );
    effects.bitcrush = mutate_linear(
        defaults.bitcrush,
        effects.bitcrush,
        0.0,
        1.0,
        amount * 0.12,
        &mut small_fx_rng,
    );
    effects.bitcrush_levels = mutate_log(
        defaults.bitcrush_levels,
        effects.bitcrush_levels,
        2.0,
        16.0,
        amount * 0.3,
        &mut small_fx_rng,
    )
    .round();
    effects.bitcrush_dither = mutate_linear(
        defaults.bitcrush_dither,
        effects.bitcrush_dither,
        0.0,
        1.0,
        amount * 0.2,
        &mut small_fx_rng,
    );
    effects.multi_grid_x = mutate_linear(
        defaults.multi_grid_x,
        effects.multi_grid_x,
        1.0,
        8.0,
        amount * 0.8,
        &mut small_fx_rng,
    )
    .round();
    effects.multi_grid_y = mutate_linear(
        defaults.multi_grid_y,
        effects.multi_grid_y,
        1.0,
        8.0,
        amount * 0.8,
        &mut small_fx_rng,
    )
    .round();
    effects.barrel = mutate_linear(
        defaults.barrel,
        effects.barrel,
        -1.0,
        1.0,
        amount * 0.15,
        &mut small_fx_rng,
    );
    effects.chroma_aberration = mutate_linear(
        defaults.chroma_aberration,
        effects.chroma_aberration,
        0.0,
        1.0,
        amount * 0.15,
        &mut small_fx_rng,
    );
    effects.anamorphic_streak = mutate_linear(
        defaults.anamorphic_streak,
        effects.anamorphic_streak,
        0.0,
        1.0,
        amount * 0.1,
        &mut small_fx_rng,
    );
    // B8 key dressing, appended to the end of the domain-separated stream so
    // every earlier draw stays byte-for-byte stable for the same seed and
    // stream. The border colour is a discrete closed table and is
    // deliberately not rerolled.
    effects.key_border = mutate_linear(
        defaults.key_border,
        effects.key_border,
        0.0,
        1.0,
        amount * 0.12,
        &mut small_fx_rng,
    );
    effects.key_shadow = mutate_linear(
        defaults.key_shadow,
        effects.key_shadow,
        0.0,
        1.0,
        amount * 0.12,
        &mut small_fx_rng,
    );
}

/// Bounded opt-in Dice mutation for authored geometry.
///
/// This uses a domain-separated stream rather than consuming the established
/// effects RNG sequence, preserving every historical Dice result when spatial
/// controls are disabled (and preserving the effects bytes when enabled).
/// Discrete fit/edge/sampling choices remain deliberate authored decisions.
pub(crate) fn mutate_live_transform(
    transform: &mut SpatialTransform,
    amount: f32,
    seed: u32,
    stream: u64,
) {
    let amount = if amount.is_finite() {
        amount.clamp(0.0, 2.0)
    } else {
        0.0
    };
    if amount == 0.0 {
        return;
    }

    let defaults = SpatialTransform::default();
    let mut rng = SplitMix64::new(
        u64::from(seed) ^ stream.wrapping_mul(0xa076_1d64_78bd_642f) ^ 0x5350_4154_4941_4c31,
    );
    let mut clean = transform.sanitized();
    for axis in 0..2 {
        clean.position[axis] = mutate_linear(
            defaults.position[axis],
            clean.position[axis],
            POSITION_MIN,
            POSITION_MAX,
            amount * 0.25,
            &mut rng,
        );
        clean.scale[axis] = mutate_linear(
            defaults.scale[axis],
            clean.scale[axis],
            SCALE_MIN,
            SCALE_MAX,
            amount * 0.35,
            &mut rng,
        );
        clean.anchor[axis] = mutate_linear(
            defaults.anchor[axis],
            clean.anchor[axis],
            ANCHOR_MIN,
            ANCHOR_MAX,
            amount * 0.10,
            &mut rng,
        );
    }
    clean.rotation_deg = mutate_circular(
        defaults.rotation_deg,
        clean.rotation_deg,
        -180.0,
        180.0,
        amount * 30.0,
        &mut rng,
    );
    clean.skew_deg = mutate_linear(
        defaults.skew_deg,
        clean.skew_deg,
        -SKEW_LIMIT_DEGREES,
        SKEW_LIMIT_DEGREES,
        amount * 12.0,
        &mut rng,
    );
    clean.skew_axis_deg = mutate_circular(
        defaults.skew_axis_deg,
        clean.skew_axis_deg,
        -180.0,
        180.0,
        amount * 25.0,
        &mut rng,
    );
    for side in 0..4 {
        clean.crop[side] = mutate_linear(
            defaults.crop[side],
            clean.crop[side],
            0.0,
            CROP_MAX,
            amount * 0.04,
            &mut rng,
        );
    }
    *transform = clean.sanitized();
}

// M3 is appended in independent per-field domains. Adding a later original
// cannot consume entropy from the long-established effect/transform/rack
// sequences, nor can it perturb a sibling original's value.
const DICE_TEMPORAL_LOOM_AMOUNT: u64 = 0x5433_4c4f_4f4d_414d;
const DICE_TEMPORAL_LOOM_DEPTH: u64 = 0x5433_4c4f_4f4d_4450;
const DICE_TEMPORAL_LOOM_PHASE: u64 = 0x5433_4c4f_4f4d_5048;
const DICE_TEMPORAL_LOOM_SCALE: u64 = 0x5433_4c4f_4f4d_5343;
const DICE_TEMPORAL_LOOM_ANGLE: u64 = 0x5433_4c4f_4f4d_414e;
const DICE_TEMPORAL_LOOM_FOLDS: u64 = 0x5433_4c4f_4f4d_464f;
const DICE_TEMPORAL_LOOM_QUANT: u64 = 0x5433_4c4f_4f4d_5155;
const DICE_TEMPORAL_ATLAS_AMOUNT: u64 = 0x5433_4154_4c53_414d;
const DICE_TEMPORAL_ATLAS_TERRITORIES: u64 = 0x5433_4154_4c53_5445;
const DICE_TEMPORAL_ATLAS_COLLISION: u64 = 0x5433_4154_4c53_434f;
const DICE_TEMPORAL_GARDEN_AMOUNT: u64 = 0x5433_4741_5244_414d;
const DICE_TEMPORAL_GARDEN_THRESHOLD: u64 = 0x5433_4741_5244_5448;
const DICE_TEMPORAL_GARDEN_SOFTNESS: u64 = 0x5433_4741_5244_534f;
const DICE_TEMPORAL_GARDEN_DECAY: u64 = 0x5433_4741_5244_4443;
const DICE_TEMPORAL_GARDEN_HOLD: u64 = 0x5433_4741_5244_484f;

fn temporal_rng(seed: u32, stream: u64, domain: u64) -> SplitMix64 {
    SplitMix64::new(u64::from(seed) ^ stream.wrapping_mul(0xa076_1d64_78bd_642f) ^ domain)
}

/// Deterministically vary only bounded numeric M3 controls.
///
/// Topology, interpolation, gate, both public seeds, Score configuration,
/// reset policy, and loop-driver identity are explicit authored laws and are
/// never changed by Dice.
pub(crate) fn mutate_live_temporal_originals(
    originals: &mut TemporalOriginalsParams,
    amount: f32,
    seed: u32,
    stream: u64,
) {
    let amount = clean_amount(amount);
    if amount == 0.0 {
        return;
    }
    let defaults = TemporalOriginalsParams::default();

    macro_rules! linear {
        ($field:expr, $anchor:expr, $min:expr, $max:expr, $scale:expr, $domain:expr) => {{
            let mut rng = temporal_rng(seed, stream, $domain);
            $field = mutate_linear($anchor, $field, $min, $max, amount * $scale, &mut rng);
        }};
    }

    linear!(
        originals.loom.amount,
        defaults.loom.amount,
        0.0,
        1.0,
        0.2,
        DICE_TEMPORAL_LOOM_AMOUNT
    );
    linear!(
        originals.loom.depth,
        defaults.loom.depth,
        0.0,
        1.0,
        0.2,
        DICE_TEMPORAL_LOOM_DEPTH
    );
    linear!(
        originals.loom.phase,
        defaults.loom.phase,
        -1_000.0,
        1_000.0,
        0.25,
        DICE_TEMPORAL_LOOM_PHASE
    );
    {
        let mut rng = temporal_rng(seed, stream, DICE_TEMPORAL_LOOM_SCALE);
        originals.loom.scale = mutate_log(
            defaults.loom.scale,
            originals.loom.scale,
            0.01,
            100.0,
            amount * 0.25,
            &mut rng,
        );
    }
    {
        let mut rng = temporal_rng(seed, stream, DICE_TEMPORAL_LOOM_ANGLE);
        originals.loom.angle = mutate_circular(
            defaults.loom.angle,
            originals.loom.angle,
            -180.0,
            180.0,
            amount * 30.0,
            &mut rng,
        );
    }
    {
        let mut rng = temporal_rng(seed, stream, DICE_TEMPORAL_LOOM_FOLDS);
        originals.loom.folds = mutate_linear(
            f32::from(defaults.loom.folds),
            f32::from(originals.loom.folds),
            1.0,
            16.0,
            amount * 2.0,
            &mut rng,
        )
        .round() as u8;
    }
    {
        let mut rng = temporal_rng(seed, stream, DICE_TEMPORAL_LOOM_QUANT);
        originals.loom.quantization = mutate_linear(
            f32::from(defaults.loom.quantization),
            f32::from(originals.loom.quantization),
            0.0,
            24.0,
            amount * 3.0,
            &mut rng,
        )
        .round() as u8;
    }

    linear!(
        originals.atlas.amount,
        defaults.atlas.amount,
        0.0,
        1.0,
        0.2,
        DICE_TEMPORAL_ATLAS_AMOUNT
    );
    {
        let mut rng = temporal_rng(seed, stream, DICE_TEMPORAL_ATLAS_TERRITORIES);
        originals.atlas.territories = mutate_linear(
            f32::from(defaults.atlas.territories),
            f32::from(originals.atlas.territories),
            1.0,
            64.0,
            amount * 6.0,
            &mut rng,
        )
        .round() as u8;
    }
    linear!(
        originals.atlas.collision,
        defaults.atlas.collision,
        0.0,
        1.0,
        0.2,
        DICE_TEMPORAL_ATLAS_COLLISION
    );
    linear!(
        originals.garden.amount,
        defaults.garden.amount,
        0.0,
        1.0,
        0.2,
        DICE_TEMPORAL_GARDEN_AMOUNT
    );
    linear!(
        originals.garden.threshold,
        defaults.garden.threshold,
        0.0,
        1.0,
        0.15,
        DICE_TEMPORAL_GARDEN_THRESHOLD
    );
    linear!(
        originals.garden.softness,
        defaults.garden.softness,
        0.0,
        0.5,
        0.08,
        DICE_TEMPORAL_GARDEN_SOFTNESS
    );
    linear!(
        originals.garden.decay,
        defaults.garden.decay,
        0.0,
        1.0,
        0.15,
        DICE_TEMPORAL_GARDEN_DECAY
    );
    {
        let mut rng = temporal_rng(seed, stream, DICE_TEMPORAL_GARDEN_HOLD);
        let anchor = f64::from(defaults.garden.max_hold_ticks);
        let value = f64::from(originals.garden.max_hold_ticks);
        let candidate = anchor
            + f64::from(MEAN_REVERSION) * (value - anchor)
            + f64::from(rng.signed()) * f64::from(amount) * 30.0;
        originals.garden.max_hold_ticks = candidate.round().clamp(0.0, f64::from(u32::MAX)) as u32;
    }
}

/// Stable owner identity for M4 Motion Dice. A layer's stream follows its
/// immutable ID across stack moves; master and layer domains never overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiceMotionScope {
    Master,
    Layer(StableLayerId),
}

const DICE_MOTION_DOMAIN: u64 = 0x4d34_4d4f_5449_4f4e;
const DICE_MOTION_AMOUNT: u64 = 0x414d_4f55_4e54_0001;
const DICE_MOTION_THRESHOLD: u64 = 0x5448_5245_5348_0001;
const DICE_MOTION_SOFTNESS: u64 = 0x534f_4654_4e45_5353;
const DICE_MOTION_REFRESH: u64 = 0x5245_4652_4553_4801;
const DICE_MOTION_DECAY: u64 = 0x4445_4341_5900_0001;
const DICE_MOTION_OCCLUSION: u64 = 0x4f43_434c_5553_494f;
const DICE_MOTION_SHUTTER_ANGLE: u64 = 0x5348_5554_414e_474c;
const DICE_MOTION_SHUTTER_PHASE: u64 = 0x5348_5554_5048_4153;
const DICE_MOTION_SHUTTER_CURVE: u64 = 0x5348_5554_4355_5256;
const DICE_MOTION_CHROMATIC_LAG: u64 = 0x4348_524f_4d4c_4147;
// B2 procedural field scalars live in fresh domains so every pre-B2 Dice
// stream stays byte-stable.
const DICE_MOTION_FIELD_SCALE: u64 = 0x4649_454c_4453_434c;
const DICE_MOTION_FIELD_RATE: u64 = 0x4649_454c_4452_4154;
// B2 flow shaping, likewise in fresh domains.
const DICE_MOTION_STRETCH: u64 = 0x5354_5245_5443_4800;
const DICE_MOTION_EDGE_REPEL: u64 = 0x4544_4745_5245_5045;
const DICE_MOTION_VECTOR_TRASH: u64 = 0x5645_4354_5452_4153;
const DICE_MOTION_TRASH_BLOCK: u64 = 0x5452_4153_424c_4f43;

const fn dice_motion_owner_domain(scope: DiceMotionScope) -> u64 {
    match scope {
        DiceMotionScope::Master => 0x4d41_5354_4552_4d34,
        DiceMotionScope::Layer(layer_id) => {
            0x4c41_5945_525f_4d34 ^ layer_id.get().wrapping_mul(0x9e37_79b9_7f4a_7c15)
        }
    }
}

fn motion_rng(seed: u32, stream: u64, scope: DiceMotionScope, field: u64) -> SplitMix64 {
    SplitMix64::new(
        u64::from(seed)
            ^ stream.wrapping_mul(0xa076_1d64_78bd_642f)
            ^ DICE_MOTION_DOMAIN
            ^ dice_motion_owner_domain(scope)
            ^ field,
    )
}

/// Deterministically vary only bounded numeric Motion values. Algorithm
/// version, field/quality tiers, donor identity (including a missing
/// tombstone), carrier policy, and shutter quality never change.
pub(crate) fn mutate_live_motion(
    motion: &mut MotionParams,
    amount: f32,
    seed: u32,
    stream: u64,
    scope: DiceMotionScope,
) {
    let amount = clean_amount(amount);
    if amount == 0.0 {
        return;
    }
    let defaults = MotionParams::default();
    macro_rules! linear {
        ($field:expr, $anchor:expr, $min:expr, $max:expr, $scale:expr, $domain:expr) => {{
            let mut rng = motion_rng(seed, stream, scope, $domain);
            $field = mutate_linear($anchor, $field, $min, $max, amount * $scale, &mut rng);
        }};
    }

    if matches!(scope, DiceMotionScope::Layer(_)) {
        linear!(
            motion.transplant.amount,
            defaults.transplant.amount,
            0.0,
            1.0,
            0.2,
            DICE_MOTION_AMOUNT
        );
        linear!(
            motion.transplant.confidence_threshold,
            defaults.transplant.confidence_threshold,
            0.0,
            1.0,
            0.15,
            DICE_MOTION_THRESHOLD
        );
        linear!(
            motion.transplant.confidence_softness,
            defaults.transplant.confidence_softness,
            0.0,
            0.5,
            0.08,
            DICE_MOTION_SOFTNESS
        );
        linear!(
            motion.transplant.refresh,
            defaults.transplant.refresh,
            0.0,
            1.0,
            0.15,
            DICE_MOTION_REFRESH
        );
        linear!(
            motion.transplant.decay,
            defaults.transplant.decay,
            0.0,
            1.0,
            0.15,
            DICE_MOTION_DECAY
        );
        linear!(
            motion.transplant.occlusion,
            defaults.transplant.occlusion,
            0.0,
            1.0,
            0.15,
            DICE_MOTION_OCCLUSION
        );
    }
    linear!(
        motion.shutter.angle_degrees,
        defaults.shutter.angle_degrees,
        0.0,
        360.0,
        60.0,
        DICE_MOTION_SHUTTER_ANGLE
    );
    linear!(
        motion.shutter.phase,
        defaults.shutter.phase,
        -1.0,
        1.0,
        0.25,
        DICE_MOTION_SHUTTER_PHASE
    );
    linear!(
        motion.shutter.curvature,
        defaults.shutter.curvature,
        -2.0,
        2.0,
        0.5,
        DICE_MOTION_SHUTTER_CURVE
    );
    linear!(
        motion.shutter.chromatic_lag,
        defaults.shutter.chromatic_lag,
        0.0,
        1.0,
        0.15,
        DICE_MOTION_CHROMATIC_LAG
    );
    linear!(
        motion.procedural.scale,
        defaults.procedural.scale,
        0.0,
        1.0,
        0.2,
        DICE_MOTION_FIELD_SCALE
    );
    linear!(
        motion.procedural.rate,
        defaults.procedural.rate,
        -2.0,
        2.0,
        0.5,
        DICE_MOTION_FIELD_RATE
    );
    linear!(
        motion.shaping.stretch,
        defaults.shaping.stretch,
        0.0,
        1.0,
        0.15,
        DICE_MOTION_STRETCH
    );
    linear!(
        motion.shaping.edge_repel,
        defaults.shaping.edge_repel,
        0.0,
        1.0,
        0.15,
        DICE_MOTION_EDGE_REPEL
    );
    linear!(
        motion.shaping.vector_trash,
        defaults.shaping.vector_trash,
        0.0,
        1.0,
        0.1,
        DICE_MOTION_VECTOR_TRASH
    );
    linear!(
        motion.shaping.trash_block_size,
        defaults.shaping.trash_block_size,
        2.0,
        256.0,
        24.0,
        DICE_MOTION_TRASH_BLOCK
    );
    *motion = motion.sanitized();
}

/// Stable owner identity for M2 Dice. It is deliberately independent from
/// rack order and live layer position so unrelated insertions/reorders cannot
/// perturb another node's random stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiceRackScope {
    Master,
    Layer(StableLayerId),
    Group(GroupId),
}

const DICE_RACK_DOMAIN: u64 = 0x5241_434b_5f44_4943;
const DICE_GROUP_DOMAIN: u64 = 0x4752_4f55_505f_4449;
// Appended M2 composition streams. Each value has its own domain so adding a
// control cannot perturb established group/rack Dice output or any sibling.
const DICE_GROUP_MATTE_AMOUNT_DOMAIN: u64 = 0x4d41_5454_455f_414d;
const DICE_GROUP_MATTE_THRESHOLD_DOMAIN: u64 = 0x4d41_5454_455f_5448;
const DICE_GROUP_MATTE_SOFTNESS_DOMAIN: u64 = 0x4d41_5454_455f_534f;
const DICE_BUS_CROSSFADE_DOMAIN: u64 = 0x4255_535f_4352_4f53;
/// B8 bus-mixer values: one shared domain, field-separated by the `owner`
/// index so every value recomputes alone.
const DICE_BUS_MIXER_DOMAIN: u64 = 0x4255_535f_4d49_5845;

pub(crate) const fn dice_rack_owner_domain(scope: DiceRackScope) -> u64 {
    match scope {
        DiceRackScope::Master => 0x4d41_5354_4552_0001,
        DiceRackScope::Layer(layer_id) => {
            0x4c41_5945_5200_0000 ^ layer_id.get().wrapping_mul(0x9e37_79b9_7f4a_7c15)
        }
        DiceRackScope::Group(group_id) => {
            0x4752_4f55_5000_0000 ^ group_id.get().wrapping_mul(0xd6e8_feb8_6659_fd93)
        }
    }
}

pub(crate) const fn dice_node_domain(scope: DiceRackScope, node_id: NodeId) -> u64 {
    dice_rack_owner_domain(scope)
        ^ node_id.get().wrapping_mul(0xa076_1d64_78bd_642f)
        ^ DICE_RACK_DOMAIN
}

fn clean_amount(amount: f32) -> f32 {
    if amount.is_finite() {
        amount.clamp(0.0, 2.0)
    } else {
        0.0
    }
}

fn node_rng(seed: u32, stream: u64, scope: DiceRackScope, node_id: NodeId) -> SplitMix64 {
    SplitMix64::new(
        u64::from(seed)
            ^ stream.wrapping_mul(0x8c6f_7365_7061_7261)
            ^ dice_node_domain(scope, node_id),
    )
}

fn composition_value_rng(seed: u32, stream: u64, owner: u64, domain: u64) -> SplitMix64 {
    SplitMix64::new(
        u64::from(seed)
            ^ stream.wrapping_mul(0x8c6f_7365_7061_7261)
            ^ owner.wrapping_mul(0xd6e8_feb8_6659_fd93)
            ^ domain,
    )
}

/// Mutate only value-owned controls in a runtime rack. Node IDs/order/kinds,
/// enabled state, blends, image routes/channels/invert, and legacy markers are
/// invariant. Each node consumes its own stable domain.
pub(crate) fn mutate_runtime_rack_values(
    rack: &mut RuntimeVisualRack,
    amount: f32,
    seed: u32,
    stream: u64,
    scope: DiceRackScope,
) {
    let amount = clean_amount(amount);
    if amount == 0.0 {
        return;
    }
    let node_ids: Vec<_> = rack.iter().map(|node| node.stable_id).collect();
    for node_id in node_ids {
        let Some(node) = rack.get_mut(node_id) else {
            continue;
        };
        if matches!(
            node.kind,
            RuntimeVisualNodeKind::LegacyCanonical | RuntimeVisualNodeKind::LegacyTemporal
        ) {
            continue;
        }
        let mut rng = node_rng(seed, stream, scope, node_id);
        node.wet = mutate_linear(1.0, node.wet, 0.0, 1.0, amount * 0.25, &mut rng);
        match &mut node.kind {
            RuntimeVisualNodeKind::LegacyCanonical | RuntimeVisualNodeKind::LegacyTemporal => {}
            RuntimeVisualNodeKind::Transform(value) => {
                mutate_live_transform(value, amount, seed, dice_node_domain(scope, node_id));
            }
            RuntimeVisualNodeKind::DigitalColor(value) => {
                let defaults = crate::visual_rack::DigitalColorParams::default();
                value.pixelate_size = mutate_log(
                    defaults.pixelate_size,
                    value.pixelate_size,
                    1.0,
                    32.0,
                    amount * 0.45,
                    &mut rng,
                )
                .round();
                value.rgb_split = mutate_linear(
                    defaults.rgb_split,
                    value.rgb_split,
                    0.0,
                    30.0,
                    amount * 4.0,
                    &mut rng,
                );
                value.downsample = mutate_log(
                    defaults.downsample,
                    value.downsample,
                    0.05,
                    1.0,
                    amount * 0.22,
                    &mut rng,
                );
                value.hue_shift = mutate_circular(
                    defaults.hue_shift,
                    value.hue_shift,
                    -180.0,
                    180.0,
                    amount * 50.0,
                    &mut rng,
                );
                value.saturation = mutate_linear(
                    defaults.saturation,
                    value.saturation,
                    -1.0,
                    1.0,
                    amount * 0.3,
                    &mut rng,
                );
                value.brightness = mutate_linear(
                    defaults.brightness,
                    value.brightness,
                    -1.0,
                    1.0,
                    amount * 0.25,
                    &mut rng,
                );
                value.contrast = mutate_linear(
                    defaults.contrast,
                    value.contrast,
                    -1.0,
                    1.0,
                    amount * 0.3,
                    &mut rng,
                );
                value.posterize = mutate_linear(
                    defaults.posterize,
                    value.posterize,
                    0.0,
                    16.0,
                    amount * 2.0,
                    &mut rng,
                )
                .round();
                value.invert = mutate_linear(
                    defaults.invert,
                    value.invert,
                    0.0,
                    1.0,
                    amount * 0.25,
                    &mut rng,
                );
                value.vignette = mutate_linear(
                    defaults.vignette,
                    value.vignette,
                    0.0,
                    1.5,
                    amount * 0.22,
                    &mut rng,
                );
                value.color_drift = mutate_linear(
                    defaults.color_drift,
                    value.color_drift,
                    0.0,
                    0.02,
                    amount * 0.004,
                    &mut rng,
                );
            }
            RuntimeVisualNodeKind::Key(value) => {
                let defaults = crate::visual_rack::KeyParams::default();
                value.threshold = mutate_linear(
                    defaults.threshold,
                    value.threshold,
                    0.0,
                    1.0,
                    amount * 0.2,
                    &mut rng,
                );
                value.softness = mutate_linear(
                    defaults.softness,
                    value.softness,
                    0.0,
                    0.5,
                    amount * 0.1,
                    &mut rng,
                );
                for component in 0..3 {
                    value.color[component] = mutate_linear(
                        defaults.color[component],
                        value.color[component],
                        0.0,
                        1.0,
                        amount * 0.15,
                        &mut rng,
                    );
                }
                value.tolerance = mutate_linear(
                    defaults.tolerance,
                    value.tolerance,
                    0.0,
                    1.0,
                    amount * 0.15,
                    &mut rng,
                );
            }
            RuntimeVisualNodeKind::Cellular(value) => {
                let defaults = crate::visual_rack::CellularParams::default();
                value.amount = mutate_linear(
                    defaults.amount,
                    value.amount,
                    0.0,
                    1.0,
                    amount * 0.2,
                    &mut rng,
                );
                value.scale = mutate_log(
                    defaults.scale,
                    value.scale,
                    2.0,
                    32.0,
                    amount * 0.28,
                    &mut rng,
                );
                value.warp =
                    mutate_linear(defaults.warp, value.warp, 0.0, 1.0, amount * 0.18, &mut rng);
                value.speed = mutate_linear(
                    defaults.speed,
                    value.speed,
                    0.0,
                    2.0,
                    amount * 0.35,
                    &mut rng,
                );
                value.gap_amount = mutate_linear(
                    defaults.gap_amount,
                    value.gap_amount,
                    0.0,
                    1.0,
                    amount * 0.15,
                    &mut rng,
                );
                value.gap_threshold = mutate_linear(
                    defaults.gap_threshold,
                    value.gap_threshold,
                    0.0,
                    1.0,
                    amount * 0.15,
                    &mut rng,
                );
                value.gap_softness = mutate_linear(
                    defaults.gap_softness,
                    value.gap_softness,
                    0.0,
                    0.5,
                    amount * 0.08,
                    &mut rng,
                );
                if rng.chance((amount * 0.5).clamp(0.0, 1.0)) {
                    value.seed = rng.next_u64() as u32;
                }
            }
            RuntimeVisualNodeKind::Shift(value) => {
                let defaults = crate::visual_rack::ShiftParams::default();
                value.amount = mutate_linear(
                    defaults.amount,
                    value.amount,
                    0.0,
                    1.0,
                    amount * 0.25,
                    &mut rng,
                );
                value.block_size = mutate_log(
                    defaults.block_size,
                    value.block_size,
                    2.0,
                    256.0,
                    amount * 0.35,
                    &mut rng,
                )
                .round();
                value.density = mutate_linear(
                    defaults.density,
                    value.density,
                    0.0,
                    1.0,
                    amount * 0.15,
                    &mut rng,
                );
                value.speed = mutate_linear(
                    defaults.speed,
                    value.speed,
                    0.0,
                    20.0,
                    amount * 2.5,
                    &mut rng,
                );
                if rng.chance((amount * 0.5).clamp(0.0, 1.0)) {
                    value.seed = rng.next_u64() as u32;
                }
            }
            RuntimeVisualNodeKind::Grain(value) => {
                let defaults = crate::visual_rack::GrainParams::default();
                value.intensity = mutate_linear(
                    defaults.intensity,
                    value.intensity,
                    0.0,
                    0.3,
                    amount * 0.06,
                    &mut rng,
                );
                value.size =
                    mutate_linear(defaults.size, value.size, 1.0, 4.0, amount * 0.5, &mut rng);
                if rng.chance((amount * 0.5).clamp(0.0, 1.0)) {
                    value.seed = rng.next_u64() as u32;
                }
            }
            RuntimeVisualNodeKind::Mask(mask) => match mask {
                RuntimeMaskParams::Rectangle(value) => {
                    let defaults = crate::visual_rack::RectangleMask::default();
                    for axis in 0..2 {
                        value.center[axis] = mutate_linear(
                            defaults.center[axis],
                            value.center[axis],
                            -2.0,
                            3.0,
                            amount * 0.15,
                            &mut rng,
                        );
                        value.size[axis] = mutate_linear(
                            defaults.size[axis],
                            value.size[axis],
                            0.0,
                            4.0,
                            amount * 0.3,
                            &mut rng,
                        );
                    }
                    value.rotation_deg = mutate_circular(
                        defaults.rotation_deg,
                        value.rotation_deg,
                        -180.0,
                        180.0,
                        amount * 30.0,
                        &mut rng,
                    );
                    value.feather = mutate_linear(
                        defaults.feather,
                        value.feather,
                        0.0,
                        1.0,
                        amount * 0.15,
                        &mut rng,
                    );
                }
                RuntimeMaskParams::Ellipse(value) => {
                    let defaults = crate::visual_rack::EllipseMask::default();
                    for axis in 0..2 {
                        value.center[axis] = mutate_linear(
                            defaults.center[axis],
                            value.center[axis],
                            -2.0,
                            3.0,
                            amount * 0.15,
                            &mut rng,
                        );
                        value.radii[axis] = mutate_linear(
                            defaults.radii[axis],
                            value.radii[axis],
                            0.0,
                            2.0,
                            amount * 0.2,
                            &mut rng,
                        );
                    }
                    value.rotation_deg = mutate_circular(
                        defaults.rotation_deg,
                        value.rotation_deg,
                        -180.0,
                        180.0,
                        amount * 30.0,
                        &mut rng,
                    );
                    value.feather = mutate_linear(
                        defaults.feather,
                        value.feather,
                        0.0,
                        1.0,
                        amount * 0.15,
                        &mut rng,
                    );
                }
                RuntimeMaskParams::Image(value) => {
                    value.amount =
                        mutate_linear(1.0, value.amount, 0.0, 1.0, amount * 0.2, &mut rng);
                    value.threshold =
                        mutate_linear(0.5, value.threshold, 0.0, 1.0, amount * 0.2, &mut rng);
                    value.softness =
                        mutate_linear(0.1, value.softness, 0.0, 0.5, amount * 0.1, &mut rng);
                }
            },
            RuntimeVisualNodeKind::Displace(value) => {
                // Donor route and boundary law are stable authored topology.
                // Dice may move only the two continuous gains. Each node draws
                // from its own stable domain, so appending this arm cannot
                // perturb any previously authored node's stream.
                value.amount_x =
                    mutate_linear(0.0, value.amount_x, -1.0, 1.0, amount * 0.2, &mut rng);
                value.amount_y =
                    mutate_linear(0.0, value.amount_y, -1.0, 1.0, amount * 0.2, &mut rng);
            }
            RuntimeVisualNodeKind::Symmetry(value) => {
                // The two image routes, the two motion routes, the mode, the
                // boundary, the authored seed, and the six mask bits are stable
                // authored topology; Dice moves only the declared continuous
                // controls and therefore can never reroll the sector table.
                // Each node draws from its own stable domain, so appending this
                // arm cannot perturb any previously authored node's stream.
                value.base_folds =
                    mutate_linear(1.0, value.base_folds, 1.0, 32.0, amount * 2.0, &mut rng);
                value.fold_offset =
                    mutate_linear(0.0, value.fold_offset, -32.0, 32.0, amount * 2.0, &mut rng);
                value.radial_phase_deg = mutate_circular(
                    0.0,
                    value.radial_phase_deg,
                    -180.0,
                    180.0,
                    amount * 30.0,
                    &mut rng,
                );
                value.orbit_phase =
                    mutate_linear(0.0, value.orbit_phase, -1.0, 1.0, amount * 0.2, &mut rng);
                value.planar_axis_deg = mutate_circular(
                    0.0,
                    value.planar_axis_deg,
                    -180.0,
                    180.0,
                    amount * 30.0,
                    &mut rng,
                );
                value.planar_phase =
                    mutate_linear(0.0, value.planar_phase, -4.0, 4.0, amount * 0.4, &mut rng);
                value.cell_skew =
                    mutate_linear(0.0, value.cell_skew, -1.0, 1.0, amount * 0.2, &mut rng);
                value.spiral_scale =
                    mutate_linear(0.0, value.spiral_scale, -1.0, 1.0, amount * 0.2, &mut rng);
                value.orbit_radius =
                    mutate_linear(0.0, value.orbit_radius, 0.0, 1.0, amount * 0.2, &mut rng);
                value.orbit_spin_deg = mutate_circular(
                    0.0,
                    value.orbit_spin_deg,
                    -180.0,
                    180.0,
                    amount * 30.0,
                    &mut rng,
                );
                value.motion_gain =
                    mutate_linear(0.0, value.motion_gain, -1.0, 1.0, amount * 0.2, &mut rng);
                value.hue_span =
                    mutate_linear(0.0, value.hue_span, 0.0, 1.0, amount * 0.2, &mut rng);
                for slot in &mut value.center {
                    *slot = mutate_linear(0.5, *slot, -1.0, 2.0, amount * 0.1, &mut rng);
                }
            }
            RuntimeVisualNodeKind::Residual(value) => {
                // Both routes, the block vocabulary, the quantization law, and
                // the quantization seed are stable authored topology. Dice may
                // move only the two continuous values, and this node draws from
                // its own stable domain so appending the arm cannot perturb any
                // previously authored node's stream.
                value.mix = mutate_linear(0.0, value.mix, 0.0, 1.0, amount * 0.2, &mut rng);
                value.detail_gain =
                    mutate_linear(1.0, value.detail_gain, 0.0, 4.0, amount * 0.2, &mut rng);
            }
            // Dice preserves a Study exactly: the digest names a validated
            // document and there is no continuous value to move. The node's
            // common wet was already diced above like every other node's.
            RuntimeVisualNodeKind::Study(_) => {}
            RuntimeVisualNodeKind::ScanProcessor(value) => {
                // `lines`, `samples_per_line`, and the two reversals are
                // plan-time geometry and discrete laws — stable authored
                // topology for Dice's purposes. Dice moves only the fifteen
                // continuous controls, and this node draws from its own
                // stable domain so appending the arm cannot perturb any
                // previously authored node's stream.
                value.amount = mutate_linear(0.0, value.amount, 0.0, 1.0, amount * 0.2, &mut rng);
                value.ribbon_width =
                    mutate_linear(0.12, value.ribbon_width, 0.0, 1.0, amount * 0.2, &mut rng);
                value.velocity_mix =
                    mutate_linear(0.8, value.velocity_mix, 0.0, 1.0, amount * 0.2, &mut rng);
                value.tilt_x = mutate_linear(0.0, value.tilt_x, -1.0, 1.0, amount * 0.2, &mut rng);
                value.tilt_y = mutate_linear(0.0, value.tilt_y, -1.0, 1.0, amount * 0.2, &mut rng);
                value.perspective =
                    mutate_linear(0.3, value.perspective, 0.0, 1.0, amount * 0.2, &mut rng);
                value.s_curve =
                    mutate_linear(0.0, value.s_curve, -1.0, 1.0, amount * 0.2, &mut rng);
                value.skew = mutate_linear(0.0, value.skew, -1.0, 1.0, amount * 0.2, &mut rng);
                value.collapse =
                    mutate_linear(0.0, value.collapse, 0.0, 1.0, amount * 0.2, &mut rng);
                value.osc_amount =
                    mutate_linear(0.0, value.osc_amount, 0.0, 1.0, amount * 0.2, &mut rng);
                value.osc_freq =
                    mutate_linear(0.25, value.osc_freq, 0.0, 1.0, amount * 0.2, &mut rng);
                value.osc_lock =
                    mutate_linear(1.0, value.osc_lock, 0.0, 1.0, amount * 0.2, &mut rng);
                value.lissajous =
                    mutate_linear(0.0, value.lissajous, 0.0, 1.0, amount * 0.2, &mut rng);
                value.mono = mutate_linear(0.0, value.mono, 0.0, 1.0, amount * 0.2, &mut rng);
                value.hue = mutate_linear(0.0, value.hue, 0.0, 1.0, amount * 0.2, &mut rng);
            }
            RuntimeVisualNodeKind::BlockDct(value) => {
                // B6: five continuous controls, own stable domain — appending
                // this arm cannot perturb any previously authored node's
                // stream.
                value.amount = mutate_linear(0.0, value.amount, 0.0, 1.0, amount * 0.2, &mut rng);
                value.quantize =
                    mutate_linear(0.25, value.quantize, 0.0, 1.0, amount * 0.2, &mut rng);
                value.hf_penalty =
                    mutate_linear(0.5, value.hf_penalty, 0.0, 1.0, amount * 0.2, &mut rng);
                value.chroma_crush =
                    mutate_linear(0.4, value.chroma_crush, 0.0, 1.0, amount * 0.2, &mut rng);
                value.block = mutate_linear(0.35, value.block, 0.0, 1.0, amount * 0.2, &mut rng);
            }
            RuntimeVisualNodeKind::PixelSort(value) => {
                value.amount = mutate_linear(0.0, value.amount, 0.0, 1.0, amount * 0.2, &mut rng);
                value.threshold =
                    mutate_linear(0.45, value.threshold, 0.0, 1.0, amount * 0.2, &mut rng);
            }
            RuntimeVisualNodeKind::Avalanche(value) => {
                // The predictor axis is a discrete law Dice never touches.
                value.amount = mutate_linear(0.0, value.amount, 0.0, 1.0, amount * 0.2, &mut rng);
                value.run = mutate_linear(0.4, value.run, 0.0, 1.0, amount * 0.2, &mut rng);
            }
        }
    }
}

/// Apply optional M2 Dice domains to every group and the composition A/B
/// crossfade. Group topology/performance switches, matte routes/channel/
/// invert, membership, and bus assignment remain unchanged. Callers stage and
/// graph-preflight the returned values before commit; this is significant when
/// a matte amount moves from zero to positive and activates a latent route.
pub(crate) fn mutate_runtime_composition_values(
    composition: &mut RuntimeComposition,
    amount: f32,
    seed: u32,
    stream: u64,
    include_rack_controls: bool,
    include_group_controls: bool,
) {
    if clean_amount(amount) == 0.0 || (!include_rack_controls && !include_group_controls) {
        return;
    }
    let group_ids: Vec<_> = composition.groups().map(|group| group.id).collect();
    for group_id in group_ids {
        let Some(group) = composition.group_mut(group_id) else {
            continue;
        };
        if include_group_controls {
            let mut rng = SplitMix64::new(
                u64::from(seed)
                    ^ stream.wrapping_mul(0x8c6f_7365_7061_7261)
                    ^ group_id.get().wrapping_mul(0xd6e8_feb8_6659_fd93)
                    ^ DICE_GROUP_DOMAIN,
            );
            group.opacity = mutate_linear(1.0, group.opacity, 0.0, 1.0, amount * 0.25, &mut rng);
            mutate_live_transform(
                &mut group.transform,
                amount,
                seed,
                stream ^ group_id.get() ^ DICE_GROUP_DOMAIN,
            );
            if let Some(matte) = &mut group.matte {
                let mut amount_rng = composition_value_rng(
                    seed,
                    stream,
                    group_id.get(),
                    DICE_GROUP_MATTE_AMOUNT_DOMAIN,
                );
                matte.amount =
                    mutate_linear(1.0, matte.amount, 0.0, 1.0, amount * 0.2, &mut amount_rng);
                let mut threshold_rng = composition_value_rng(
                    seed,
                    stream,
                    group_id.get(),
                    DICE_GROUP_MATTE_THRESHOLD_DOMAIN,
                );
                matte.threshold = mutate_linear(
                    0.5,
                    matte.threshold,
                    0.0,
                    1.0,
                    amount * 0.2,
                    &mut threshold_rng,
                );
                let mut softness_rng = composition_value_rng(
                    seed,
                    stream,
                    group_id.get(),
                    DICE_GROUP_MATTE_SOFTNESS_DOMAIN,
                );
                matte.softness = mutate_linear(
                    0.1,
                    matte.softness,
                    0.0,
                    0.5,
                    amount * 0.1,
                    &mut softness_rng,
                );
            }
        }
        if include_rack_controls {
            mutate_runtime_rack_values(
                &mut group.rack,
                amount,
                seed,
                stream,
                DiceRackScope::Group(group_id),
            );
        }
    }
    if include_group_controls {
        let mut bus_rng = composition_value_rng(seed, stream, 0, DICE_BUS_CROSSFADE_DOMAIN);
        composition.set_bus_crossfade(mutate_linear(
            0.5,
            composition.bus_crossfade(),
            0.0,
            1.0,
            clean_amount(amount) * 0.2,
            &mut bus_rng,
        ));
        // B8: the bus mixer's seventeen continuous values, each drawn from
        // its own fresh domain-separated stream, so every pre-B8 field —
        // including the crossfade draw above — stays byte-stable for the
        // same seed and stream. Discrete laws (pattern, invert, rep, border
        // colour, blend mode) are never mutated.
        let mut mixer = composition.mixer();
        let scaled = clean_amount(amount);
        let field = |index: u64, anchor: f32, value: f32, min: f32, max: f32, scale: f32| -> f32 {
            let mut rng = composition_value_rng(seed, stream, index, DICE_BUS_MIXER_DOMAIN);
            mutate_linear(anchor, value, min, max, scaled * scale, &mut rng)
        };
        mixer.mix.soft = field(1, 0.03, mixer.mix.soft, 0.0, 1.0, 0.15);
        mixer.mix.origin_x = field(2, 0.0, mixer.mix.origin_x, -1.0, 1.0, 0.2);
        mixer.mix.origin_y = field(3, 0.0, mixer.mix.origin_y, -1.0, 1.0, 0.2);
        mixer.mix.detail = field(4, 0.3, mixer.mix.detail, 0.0, 1.0, 0.2);
        mixer.mix.border = field(5, 0.0, mixer.mix.border, 0.0, 1.0, 0.15);
        mixer.dirt.dirt = field(6, 0.0, mixer.dirt.dirt, 0.0, 1.0, 0.2);
        mixer.dirt.rate = field(7, 0.3, mixer.dirt.rate, 0.0, 1.0, 0.2);
        mixer.dirt.drop = field(8, 0.5, mixer.dirt.drop, 0.0, 1.0, 0.2);
        mixer.dirt.cut = field(9, 0.4, mixer.dirt.cut, 0.0, 1.0, 0.2);
        mixer.dirt.knock = field(10, 0.5, mixer.dirt.knock, 0.0, 1.0, 0.2);
        mixer.dirt.noise = field(11, 0.35, mixer.dirt.noise, 0.0, 1.0, 0.2);
        mixer.melt.melt = field(12, 0.0, mixer.melt.melt, 0.0, 2.0, 0.3);
        mixer.melt.width = field(13, 0.3, mixer.melt.width, 0.0, 2.0, 0.2);
        mixer.melt.hold = field(14, 0.6, mixer.melt.hold, 0.0, 1.5, 0.15);
        mixer.melt.swirl = field(15, 0.0, mixer.melt.swirl, -1.0, 1.0, 0.3);
        mixer.melt.chroma = field(16, 0.5, mixer.melt.chroma, 0.0, 1.0, 0.2);
        mixer.melt.creep = field(17, 0.35, mixer.melt.creep, 0.0, 1.0, 0.2);
        composition.set_mixer(mixer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_live_bounds(effects: &EffectUniforms) {
        assert!((1.0..=32.0).contains(&effects.pixelate_size));
        assert_eq!(effects.pixelate_size.fract(), 0.0);
        assert!((0.05..=1.0).contains(&effects.downsample));
        assert!((0.0..=30.0).contains(&effects.rgb_split));
        assert!((-180.0..=180.0).contains(&effects.hue_shift));
        assert!((-1.0..=1.0).contains(&effects.saturation));
        assert!((-1.0..=1.0).contains(&effects.brightness));
        assert!((-1.0..=1.0).contains(&effects.contrast));
        assert!((0.0..=16.0).contains(&effects.posterize));
        assert_eq!(effects.posterize.fract(), 0.0);
        assert!((0.0..=0.3).contains(&effects.grain_intensity));
        assert!((1.0..=4.0).contains(&effects.grain_size));
        assert!((0.0..=3.0).contains(&effects.grain_algo));
        assert_eq!(effects.grain_algo.fract(), 0.0);
        assert!(effects.color_grain == 0.0 || effects.color_grain == 1.0);
        assert!((0.0..=0.05).contains(&effects.breathe_scale));
        assert!((0.0..=2.0).contains(&effects.breathe_rotation));
        assert!((0.0..=0.02).contains(&effects.breathe_position));
        assert!((0.0..=1.5).contains(&effects.vignette));
        assert!((0.0..=0.02).contains(&effects.color_drift));
        assert!((0.0..=1.0).contains(&effects.cellular_amount));
        assert!((2.0..=32.0).contains(&effects.cellular_scale));
        assert!((0.0..=1.0).contains(&effects.cellular_warp));
        assert!((0.0..=2.0).contains(&effects.cellular_speed));
        assert!((0.0..=1.0).contains(&effects.shift_amount));
        assert!((2.0..=256.0).contains(&effects.shift_block_size));
        assert_eq!(effects.shift_block_size.fract(), 0.0);
        assert!((0.0..=1.0).contains(&effects.shift_density));
        assert!((0.0..=20.0).contains(&effects.shift_speed));
        // B13 small effects.
        assert!((0.0..=1.0).contains(&effects.contour));
        assert!((2.0..=40.0).contains(&effects.contour_bands));
        assert_eq!(effects.contour_bands.fract(), 0.0);
        assert!((0.2..=6.0).contains(&effects.contour_width));
        assert!((0.0..=1.0).contains(&effects.contour_hue));
        assert!((0.0..=1.0).contains(&effects.contour_fill));
        assert!((0.0..=1.0).contains(&effects.flatten));
        assert!((2.0..=16.0).contains(&effects.flatten_levels));
        assert_eq!(effects.flatten_levels.fract(), 0.0);
        assert!((0.0..=1.0).contains(&effects.contour_dither));
        assert!((0.0..=1.0).contains(&effects.solarize));
        assert!((0.0..=1.0).contains(&effects.negative));
        assert!((0.0..=1.0).contains(&effects.colourpass));
        assert!((-180.0..=180.0).contains(&effects.colourpass_hue));
        assert!((0.0..=1.0).contains(&effects.colourpass_width));
        assert!((0.0..=1.0).contains(&effects.edge_amount));
        assert!((-180.0..=180.0).contains(&effects.edge_hue));
        assert!((0.0..=1.0).contains(&effects.emboss));
        assert!((-180.0..=180.0).contains(&effects.emboss_angle));
        assert!((0.0..=1.0).contains(&effects.halftone));
        assert!((0.0..=1.0).contains(&effects.halftone_pitch));
        assert!((-180.0..=180.0).contains(&effects.halftone_angle));
        assert!((0.0..=1.0).contains(&effects.moire));
        assert!((0.0..=1.0).contains(&effects.moire_freq));
        assert!((0.0..=1.0).contains(&effects.row_smear));
        assert!((0.0..=1.0).contains(&effects.bitcrush));
        assert!((2.0..=16.0).contains(&effects.bitcrush_levels));
        assert_eq!(effects.bitcrush_levels.fract(), 0.0);
        assert!((0.0..=1.0).contains(&effects.bitcrush_dither));
        assert!((1.0..=8.0).contains(&effects.multi_grid_x));
        assert_eq!(effects.multi_grid_x.fract(), 0.0);
        assert!((1.0..=8.0).contains(&effects.multi_grid_y));
        assert_eq!(effects.multi_grid_y.fract(), 0.0);
        assert!((-1.0..=1.0).contains(&effects.barrel));
        assert!((0.0..=1.0).contains(&effects.chroma_aberration));
        assert!((0.0..=1.0).contains(&effects.anamorphic_streak));
    }

    /// The B13 domain-separation golden: this array was measured on the
    /// pre-B13 build (a7c700c) for the identical seed/stream/amount, where
    /// the small-effects stream did not exist. The established Dice fields
    /// must reproduce it byte for byte, because the new controls draw from
    /// their own domain-separated stream rather than consuming the
    /// established sequences.
    #[test]
    fn small_effects_dice_leaves_pre_b13_streams_byte_stable() {
        let mut effects = EffectUniforms::default();
        mutate_live_effects(&mut effects, 2.0, true, 1234, 9);
        assert_eq!(
            [
                effects.pixelate_size,
                effects.hue_shift,
                effects.contrast,
                effects.grain_size,
                effects.cellular_speed,
                effects.shift_amount,
                effects.shift_block_size,
                effects.shift_speed,
            ],
            [
                2.0,
                -82.2763,
                -0.14153367,
                1.9421463,
                0.31503654,
                0.20584321,
                8.0,
                2.1054444,
            ],
            "pre-B13 Dice values must not move"
        );
        // The new controls mutate deterministically in their own stream and
        // the discrete negative mode never rerolls.
        let mut again = EffectUniforms::default();
        mutate_live_effects(&mut again, 2.0, true, 1234, 9);
        assert_eq!(bytemuck::bytes_of(&effects), bytemuck::bytes_of(&again));
        assert_eq!(effects.negative_mode, 0.0);
        assert_live_bounds(&effects);
    }

    fn assert_spatial_bounds(transform: SpatialTransform) {
        let clean = transform.sanitized();
        assert_eq!(transform, clean);
        assert!((POSITION_MIN..=POSITION_MAX).contains(&transform.position[0]));
        assert!((POSITION_MIN..=POSITION_MAX).contains(&transform.position[1]));
        assert!((SCALE_MIN..=SCALE_MAX).contains(&transform.scale[0]));
        assert!((SCALE_MIN..=SCALE_MAX).contains(&transform.scale[1]));
        assert!((ANCHOR_MIN..=ANCHOR_MAX).contains(&transform.anchor[0]));
        assert!((ANCHOR_MIN..=ANCHOR_MAX).contains(&transform.anchor[1]));
        assert!((-180.0..=180.0).contains(&transform.rotation_deg));
        assert!((-SKEW_LIMIT_DEGREES..=SKEW_LIMIT_DEGREES).contains(&transform.skew_deg));
        assert!((-180.0..=180.0).contains(&transform.skew_axis_deg));
        assert!(transform
            .crop
            .iter()
            .all(|value| (0.0..=CROP_MAX).contains(value)));
        assert!(transform.crop[0] + transform.crop[2] <= CROP_MAX + f32::EPSILON);
        assert!(transform.crop[1] + transform.crop[3] <= CROP_MAX + f32::EPSILON);
    }

    #[test]
    fn splitmix64_matches_published_golden_vector() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    }

    #[test]
    fn next_and_stream_seed_semantics_are_stable() {
        let expected = [
            0x01fc_e552,
            0xa011_7be9,
            0x00e1_5a61,
            0xbbca_c71f,
            0x7c11_1966,
            0x68d2_7ac3,
        ];
        let mut seed = 0;
        for expected_seed in expected {
            seed = next_seed(seed);
            assert_eq!(seed, expected_seed);
            assert_ne!(seed, 0);
        }
        assert_eq!(advance_seed(0, 0), 0);
        assert_eq!(advance_seed(0, expected.len() as u64), seed);

        let mut long_sequence_seed = 0;
        for _ in 0..10_000 {
            long_sequence_seed = next_seed(long_sequence_seed);
            assert_ne!(long_sequence_seed, 0);
        }

        assert_eq!(stream_seed(0, 0), 0);
        assert_eq!(stream_seed(0, u64::MAX), 0);
        assert_eq!(stream_seed(42, 0), 0x2feb_6e95);
        assert_eq!(stream_seed(42, 1), 0x4709_e75e);
        assert_eq!(stream_seed(42, 7), 0x8922_4570);
        assert_eq!(stream_seed(42, 7), stream_seed(42, 7));
        assert_ne!(stream_seed(42, 0), stream_seed(42, 1));
        assert_ne!(stream_seed(42, 0), 0);
    }

    #[test]
    fn live_mutation_is_deterministic_bounded_and_control_safe() {
        let mut left = EffectUniforms {
            resolution: [333.0, 222.0],
            invert: 0.75,
            time: 12.5,
            key_mode: 3.0,
            key_threshold: 0.37,
            key_softness: 0.12,
            key_color: [0.2, 0.4, 0.6],
            key_tolerance: 0.24,
            cellular_gap_amount: 0.7,
            cellular_gap_threshold: 0.42,
            cellular_gap_softness: 0.08,
            random_seed: 0xfeed_beef,
            ..Default::default()
        };
        let original = left;
        let mut right = left;

        mutate_live_effects(&mut left, 2.0, true, 1234, 9);
        mutate_live_effects(&mut right, 2.0, true, 1234, 9);
        assert_eq!(bytemuck::bytes_of(&left), bytemuck::bytes_of(&right));
        assert_live_bounds(&left);

        let mut clamped_amount = original;
        mutate_live_effects(&mut clamped_amount, 99.0, true, 1234, 9);
        assert_eq!(
            bytemuck::bytes_of(&left),
            bytemuck::bytes_of(&clamped_amount)
        );

        let mut different_seed = original;
        mutate_live_effects(&mut different_seed, 2.0, true, 1235, 9);
        assert_ne!(
            bytemuck::bytes_of(&left),
            bytemuck::bytes_of(&different_seed)
        );

        let mut different_stream = original;
        mutate_live_effects(&mut different_stream, 2.0, true, 1234, 10);
        assert_ne!(
            bytemuck::bytes_of(&left),
            bytemuck::bytes_of(&different_stream)
        );

        assert_eq!(left.resolution, original.resolution);
        assert_eq!(left.invert, original.invert);
        assert_eq!(left.time, original.time);
        assert_eq!(left.key_mode, original.key_mode);
        assert_eq!(left.key_threshold, original.key_threshold);
        assert_eq!(left.key_softness, original.key_softness);
        assert_eq!(left.key_color, original.key_color);
        assert_eq!(left.key_tolerance, original.key_tolerance);
        assert_eq!(left.cellular_gap_amount, original.cellular_gap_amount);
        assert_eq!(left.cellular_gap_threshold, original.cellular_gap_threshold);
        assert_eq!(left.cellular_gap_softness, original.cellular_gap_softness);
        assert_eq!(left.random_seed, original.random_seed);

        for amount in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut unchanged = original;
            mutate_live_effects(&mut unchanged, amount, true, 1234, 9);
            assert_eq!(
                bytemuck::bytes_of(&unchanged),
                bytemuck::bytes_of(&original)
            );
        }
    }

    #[test]
    fn live_variation_remains_bounded_across_seed_and_stream_space() {
        for seed in [0, 1, 42, u32::MAX] {
            for stream in 0..64 {
                let mut effects = EffectUniforms {
                    pixelate_size: 1_000.0,
                    rgb_split: -100.0,
                    hue_shift: 720.0,
                    saturation: 5.0,
                    brightness: -5.0,
                    contrast: 7.0,
                    posterize: 99.0,
                    downsample: 9.0,
                    grain_intensity: -2.0,
                    grain_size: 40.0,
                    grain_algo: 99.0,
                    color_grain: -4.0,
                    breathe_scale: 2.0,
                    breathe_rotation: -9.0,
                    breathe_position: 1.0,
                    vignette: 8.0,
                    color_drift: -1.0,
                    cellular_amount: 5.0,
                    cellular_scale: 200.0,
                    cellular_warp: -7.0,
                    cellular_speed: 20.0,
                    shift_amount: 8.0,
                    shift_block_size: 2_000.0,
                    shift_density: -9.0,
                    shift_speed: 200.0,
                    ..EffectUniforms::default()
                };
                mutate_live_effects(&mut effects, 2.0, true, seed, stream);
                assert_live_bounds(&effects);
            }
        }
    }

    #[test]
    fn temporal_originals_dice_is_deterministic_bounded_and_control_safe() {
        let mut left = TemporalOriginalsParams::default();
        left.loom.topology = crate::effects::params::TemporalTopology::Kaleidoscopic;
        left.loom.interpolation = crate::effects::params::TemporalInterpolation::Linear;
        left.atlas.seed = 0xdead_beef;
        left.garden.gate = crate::effects::params::RefreshGardenGate::AudioOnset;
        left.garden.matte_route = crate::temporal::RefreshGardenMatteRoute::SelectedLayer {
            layer_id: crate::image_routing::StableLayerId::new(91).unwrap(),
            saved_position: crate::performance::SavedLayerPosition::new(3).unwrap(),
            stage: crate::image_routing::LayerImageStage::PostLocalEffects,
        };
        left.garden.motion_route =
            crate::temporal::RefreshGardenMotionRoute::MissingSelectedLayer {
                saved_position: crate::performance::SavedLayerPosition::new(5).unwrap(),
            };
        left.score.enabled = true;
        left.score.seed = 0x1234_5678;
        left.score.trigger = crate::effects::params::CollisionScoreTrigger::Manual;
        left.reset.loop_boundary = crate::temporal::TemporalEventResetMode::Memory;
        let authored = left;
        let mut right = left;

        mutate_live_temporal_originals(&mut left, 2.0, 77, 9);
        mutate_live_temporal_originals(&mut right, 2.0, 77, 9);
        assert_eq!(left, right);
        assert!((0.0..=1.0).contains(&left.loom.amount));
        assert!((0.0..=1.0).contains(&left.loom.depth));
        assert!((-1_000.0..=1_000.0).contains(&left.loom.phase));
        assert!((0.01..=100.0).contains(&left.loom.scale));
        assert!((-180.0..=180.0).contains(&left.loom.angle));
        assert!((1..=16).contains(&left.loom.folds));
        assert!(left.loom.quantization <= 24);
        assert!((0.0..=1.0).contains(&left.atlas.amount));
        assert!((1..=64).contains(&left.atlas.territories));
        assert!((0.0..=1.0).contains(&left.atlas.collision));
        assert!((0.0..=1.0).contains(&left.garden.amount));
        assert!((0.0..=1.0).contains(&left.garden.threshold));
        assert!((0.0..=0.5).contains(&left.garden.softness));
        assert!((0.0..=1.0).contains(&left.garden.decay));

        assert_eq!(left.loom.topology, authored.loom.topology);
        assert_eq!(left.loom.interpolation, authored.loom.interpolation);
        assert_eq!(left.atlas.seed, authored.atlas.seed);
        assert_eq!(left.garden.gate, authored.garden.gate);
        assert_eq!(left.garden.matte_route, authored.garden.matte_route);
        assert_eq!(left.garden.motion_route, authored.garden.motion_route);
        assert_eq!(left.score, authored.score);
        assert_eq!(left.reset, authored.reset);

        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut unchanged = authored;
            mutate_live_temporal_originals(&mut unchanged, invalid, 77, 9);
            assert_eq!(unchanged, authored);
        }
    }

    #[test]
    fn motion_dice_is_id_stable_bounded_and_preserves_authored_topology() {
        use crate::motion::{
            CurvedShutterParams, CurvedShutterQuality, FaradayParams, MotionCarrier, MotionDonor,
            MotionFieldSource, MotionLatticeQuality, MOTION_ALGORITHM_VERSION,
        };
        use crate::performance::SavedLayerPosition;

        let owner = StableLayerId::new(91).unwrap();
        let other = StableLayerId::new(92).unwrap();
        let donor = StableLayerId::new(77).unwrap();
        let authored = MotionParams {
            field_source: MotionFieldSource::CodecVectors,
            lattice_quality: MotionLatticeQuality::High,
            transplant: FaradayParams {
                donor: MotionDonor::Selected {
                    layer_id: donor,
                    saved_position: SavedLayerPosition::new(2).unwrap(),
                },
                carrier: MotionCarrier::FirstSourceFrame,
                ..FaradayParams::default()
            },
            shutter: CurvedShutterParams {
                quality: CurvedShutterQuality::High,
                ..CurvedShutterParams::default()
            },
            ..MotionParams::default()
        };

        let mut left = authored;
        let mut right = authored;
        mutate_live_motion(
            &mut left,
            2.0,
            0x1234_5678,
            9,
            DiceMotionScope::Layer(owner),
        );
        mutate_live_motion(
            &mut right,
            2.0,
            0x1234_5678,
            9,
            DiceMotionScope::Layer(owner),
        );
        assert_eq!(left, right);
        assert_ne!(left, authored);
        assert_eq!(left.algorithm_version, MOTION_ALGORITHM_VERSION);
        assert_eq!(left.field_source, authored.field_source);
        assert_eq!(left.lattice_quality, authored.lattice_quality);
        assert_eq!(left.transplant.donor, authored.transplant.donor);
        assert_eq!(left.transplant.carrier, authored.transplant.carrier);
        assert_eq!(left.shutter.quality, authored.shutter.quality);
        assert!((0.0..=1.0).contains(&left.transplant.amount));
        assert!((0.0..=1.0).contains(&left.transplant.confidence_threshold));
        assert!((0.0..=0.5).contains(&left.transplant.confidence_softness));
        assert!((0.0..=1.0).contains(&left.transplant.refresh));
        assert!((0.0..=1.0).contains(&left.transplant.decay));
        assert!((0.0..=1.0).contains(&left.transplant.occlusion));
        assert!((0.0..=360.0).contains(&left.shutter.angle_degrees));
        assert!((-1.0..=1.0).contains(&left.shutter.phase));
        assert!((-2.0..=2.0).contains(&left.shutter.curvature));
        assert!((0.0..=1.0).contains(&left.shutter.chromatic_lag));

        let mut reordered = authored;
        mutate_live_motion(
            &mut reordered,
            2.0,
            0x1234_5678,
            9,
            DiceMotionScope::Layer(owner),
        );
        assert_eq!(reordered, left, "stack position is outside the RNG domain");
        let mut different_owner = authored;
        mutate_live_motion(
            &mut different_owner,
            2.0,
            0x1234_5678,
            9,
            DiceMotionScope::Layer(other),
        );
        assert_ne!(different_owner, left);

        let mut master = authored;
        mutate_live_motion(&mut master, 2.0, 0x1234_5678, 9, DiceMotionScope::Master);
        assert_eq!(master.transplant, authored.transplant);
        assert_ne!(master.shutter, authored.shutter);

        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut unchanged = authored;
            mutate_live_motion(&mut unchanged, invalid, 7, 3, DiceMotionScope::Layer(owner));
            assert_eq!(unchanged, authored);
        }
    }

    #[test]
    fn discrete_grain_controls_use_only_the_declared_choice_range() {
        let mut seen_algorithms = [false; 4];
        let mut seen_colors = [false; 2];
        for stream in 0..512 {
            let mut effects = EffectUniforms {
                grain_algo: 99.0,
                color_grain: -4.0,
                ..EffectUniforms::default()
            };
            mutate_live_effects(&mut effects, 2.0, true, 0x1234_5678, stream);

            assert!((0.0..=3.0).contains(&effects.grain_algo));
            assert_eq!(effects.grain_algo.fract(), 0.0);
            let algorithm = effects.grain_algo as usize;
            seen_algorithms[algorithm] = true;

            assert!(effects.color_grain == 0.0 || effects.color_grain == 1.0);
            seen_colors[effects.color_grain as usize] = true;
        }
        assert!(seen_algorithms.into_iter().all(|seen| seen));
        assert!(seen_colors.into_iter().all(|seen| seen));

        let mut excluded = EffectUniforms {
            grain_algo: 2.75,
            color_grain: 0.25,
            ..EffectUniforms::default()
        };
        mutate_live_effects(&mut excluded, 2.0, false, 0x1234_5678, 9);
        assert_eq!(excluded.grain_algo, 2.75);
        assert_eq!(excluded.color_grain, 0.25);
    }

    #[test]
    fn live_spatial_mutation_is_opt_in_deterministic_bounded_and_domain_separated() {
        let original = SpatialTransform::new_layer_default();
        let mut left = original;
        let mut right = original;
        mutate_live_transform(&mut left, 2.0, 1234, 9);
        mutate_live_transform(&mut right, 2.0, 1234, 9);
        assert_eq!(left, right);
        assert_ne!(left, original);
        assert_spatial_bounds(left);
        assert_eq!(left.fit, original.fit);
        assert_eq!(left.edge, original.edge);
        assert_eq!(left.sampling, original.sampling);

        let mut other_stream = original;
        mutate_live_transform(&mut other_stream, 2.0, 1234, 10);
        assert_ne!(left, other_stream);

        let effects = EffectUniforms::default();
        let mut effects_without_spatial = effects;
        mutate_live_effects(&mut effects_without_spatial, 2.0, true, 1234, 9);
        let mut unused_spatial = original;
        mutate_live_transform(&mut unused_spatial, 2.0, 1234, 9);
        let mut effects_with_spatial = effects;
        mutate_live_effects(&mut effects_with_spatial, 2.0, true, 1234, 9);
        assert_eq!(
            bytemuck::bytes_of(&effects_without_spatial),
            bytemuck::bytes_of(&effects_with_spatial),
            "spatial opt-in must not consume the established effects RNG stream"
        );

        for amount in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut unchanged = original;
            mutate_live_transform(&mut unchanged, amount, 1234, 9);
            assert_eq!(unchanged, original);
        }
    }

    #[test]
    fn live_spatial_variation_sanitizes_hostile_state_across_seed_space() {
        for seed in [0, 1, 42, u32::MAX] {
            for stream in 0..64 {
                let mut transform = SpatialTransform {
                    position: [f32::INFINITY, -99.0],
                    scale: [99.0, -99.0],
                    anchor: [f32::NAN, 99.0],
                    rotation_deg: 9_999.0,
                    skew_deg: -9_999.0,
                    skew_axis_deg: f32::NEG_INFINITY,
                    crop: [9.0, 9.0, 9.0, 9.0],
                    ..SpatialTransform::new_layer_default()
                };
                mutate_live_transform(&mut transform, 2.0, seed, stream);
                assert_spatial_bounds(transform);
            }
        }
    }

    #[test]
    fn rack_dice_is_stable_by_owner_and_node_and_preserves_image_topology() {
        use crate::visual_rack::{
            MatteChannel, NodeBlend, ResolvedImageSource, ResolvedImageTap, RuntimeImageMatte,
            RuntimeVisualNode,
        };

        let target_id = NodeId::new(10).unwrap();
        let extra_id = NodeId::new(9).unwrap();
        let target = RuntimeVisualNode::authored(
            target_id,
            RuntimeVisualNodeKind::Shift(crate::visual_rack::ShiftParams::default()),
        );
        let extra = RuntimeVisualNode::authored(
            extra_id,
            RuntimeVisualNodeKind::DigitalColor(crate::visual_rack::DigitalColorParams::default()),
        );
        let mut alone = RuntimeVisualRack::try_from_parts(vec![target], Some(11)).unwrap();
        let mut reordered =
            RuntimeVisualRack::try_from_parts(vec![extra, target], Some(11)).unwrap();
        let layer_id = StableLayerId::new(77).unwrap();
        mutate_runtime_rack_values(
            &mut alone,
            1.5,
            0x1234_5678,
            4,
            DiceRackScope::Layer(layer_id),
        );
        mutate_runtime_rack_values(
            &mut reordered,
            1.5,
            0x1234_5678,
            4,
            DiceRackScope::Layer(layer_id),
        );
        assert_eq!(alone.get(target_id), reordered.get(target_id));

        let group_id = GroupId::new(88).unwrap();
        let route = ResolvedImageTap {
            source: ResolvedImageSource::GroupOutput(group_id),
            timing: crate::visual_rack::EdgeTiming::PreviousFrame,
        };
        let matte = RuntimeImageMatte {
            tap: route,
            channel: MatteChannel::Blue,
            invert: true,
            amount: 0.35,
            threshold: 0.42,
            softness: 0.13,
        };
        let mask_id = NodeId::new(12).unwrap();
        let mut mask_node = RuntimeVisualNode::authored(
            mask_id,
            RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)),
        );
        mask_node.enabled = false;
        mask_node.blend = NodeBlend::Difference;
        let mut rack = RuntimeVisualRack::try_from_parts(vec![mask_node], Some(13)).unwrap();
        let before_cursor = rack.next_node_id_raw();
        mutate_runtime_rack_values(
            &mut rack,
            2.0,
            0xfeed_beef,
            9,
            DiceRackScope::Group(group_id),
        );
        let after = rack.get(mask_id).unwrap();
        assert!(!after.enabled);
        assert_eq!(after.blend, NodeBlend::Difference);
        assert_eq!(rack.next_node_id_raw(), before_cursor);
        let RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(after_matte)) = after.kind else {
            panic!("mask topology changed");
        };
        assert_eq!(after_matte.tap, route);
        assert_eq!(after_matte.channel, MatteChannel::Blue);
        assert!(after_matte.invert);
        assert_ne!(
            [
                after_matte.amount,
                after_matte.threshold,
                after_matte.softness
            ],
            [matte.amount, matte.threshold, matte.softness]
        );
    }

    #[test]
    fn group_dice_changes_only_opted_in_values() {
        use crate::composition::{
            BusAssignment, GroupName, RuntimeGroup, RuntimeGroupMembers, RuntimeRootItem,
        };
        use crate::visual_rack::{
            ImageMatte, MatteChannel, RuntimeImageMatte, RuntimeVisualNode, ShiftParams,
        };

        let group_id = GroupId::new(5).unwrap();
        let node_id = NodeId::new(7).unwrap();
        let rack = RuntimeVisualRack::try_from_parts(
            vec![RuntimeVisualNode::authored(
                node_id,
                RuntimeVisualNodeKind::Shift(ShiftParams::default()),
            )],
            Some(8),
        )
        .unwrap();
        let mut matte =
            RuntimeImageMatte::resolve_routes(ImageMatte::default(), &mut |_| None, &|_| false);
        matte.channel = MatteChannel::Blue;
        matte.invert = true;
        matte.amount = 0.35;
        matte.threshold = 0.42;
        matte.softness = 0.13;
        let group = RuntimeGroup {
            id: group_id,
            name: GroupName::new("dice").unwrap(),
            members: RuntimeGroupMembers::default(),
            opacity: 0.6,
            transform: SpatialTransform::default(),
            rack,
            matte: Some(matte),
            solo: true,
            bypass: true,
            bus: BusAssignment::B,
        };
        let mut composition = RuntimeComposition::try_from_parts(
            vec![group],
            vec![RuntimeRootItem::Group { group_id }],
            Some(6),
            0.35,
        )
        .unwrap();
        let before = composition.clone();
        let mut repeated = composition.clone();
        mutate_runtime_composition_values(&mut composition, 2.0, 44, 3, true, true);
        mutate_runtime_composition_values(&mut repeated, 2.0, 44, 3, true, true);
        assert_eq!(composition, repeated);
        let before_group = before.group(group_id).unwrap();
        let after_group = composition.group(group_id).unwrap();
        assert_eq!(composition.root(), before.root());
        assert_eq!(composition.next_group_id_raw(), before.next_group_id_raw());
        assert_ne!(composition.bus_crossfade(), before.bus_crossfade());
        assert!((0.0..=1.0).contains(&composition.bus_crossfade()));
        assert_eq!(after_group.name, before_group.name);
        assert_eq!(after_group.members, before_group.members);
        assert_eq!(after_group.solo, before_group.solo);
        assert_eq!(after_group.bypass, before_group.bypass);
        assert_eq!(after_group.bus, before_group.bus);
        let before_matte = before_group.matte.unwrap();
        let after_matte = after_group.matte.unwrap();
        assert_eq!(after_matte.tap, before_matte.tap);
        assert_eq!(after_matte.channel, MatteChannel::Blue);
        assert!(after_matte.invert);
        assert_ne!(
            [
                after_matte.amount,
                after_matte.threshold,
                after_matte.softness
            ],
            [
                before_matte.amount,
                before_matte.threshold,
                before_matte.softness
            ]
        );
        assert!((0.0..=1.0).contains(&after_matte.amount));
        assert!((0.0..=1.0).contains(&after_matte.threshold));
        assert!((0.0..=0.5).contains(&after_matte.softness));
        assert!(
            after_group.opacity != before_group.opacity
                || after_group.transform != before_group.transform
        );
        assert_ne!(
            after_group.rack.get(node_id),
            before_group.rack.get(node_id)
        );

        let mut excluded = before.clone();
        mutate_runtime_composition_values(&mut excluded, 2.0, 44, 3, true, false);
        assert_eq!(excluded.bus_crossfade(), before.bus_crossfade());
        let excluded_group = excluded.group(group_id).unwrap();
        assert_eq!(excluded_group.opacity, before_group.opacity);
        assert_eq!(excluded_group.transform, before_group.transform);
        assert_eq!(excluded_group.matte, before_group.matte);
        assert_ne!(
            excluded_group.rack.get(node_id),
            before_group.rack.get(node_id)
        );

        // Appending matte and A/B domains must not consume the established
        // group opacity/transform stream.
        let mut without_new_values = before.clone();
        without_new_values.group_mut(group_id).unwrap().matte = None;
        mutate_runtime_composition_values(&mut without_new_values, 2.0, 44, 3, false, true);
        assert_eq!(
            without_new_values.group(group_id).unwrap().opacity,
            after_group.opacity
        );
        assert_eq!(
            without_new_values.group(group_id).unwrap().transform,
            after_group.transform
        );
    }

    #[test]
    fn dice_moves_displace_gains_only_and_never_its_route_or_boundary() {
        use crate::visual_rack::{
            DisplaceBoundary, EdgeTiming, ResolvedImageSource, ResolvedImageTap,
            RuntimeDisplaceParams,
        };

        let authored = RuntimeDisplaceParams {
            tap: ResolvedImageTap {
                source: ResolvedImageSource::CleanProgram,
                timing: EdgeTiming::PreviousFrame,
            },
            amount_x: 0.4,
            amount_y: -0.4,
            boundary: DisplaceBoundary::Mirror,
        };
        let mut rack = RuntimeVisualRack::empty();
        let node_id = rack
            .push(RuntimeVisualNodeKind::Displace(authored))
            .unwrap();
        let grain_id = rack
            .push(RuntimeVisualNodeKind::Grain(
                crate::visual_rack::GrainParams::default(),
            ))
            .unwrap();
        let baseline = rack.clone();

        let params_of = |rack: &RuntimeVisualRack| match rack.get(node_id).unwrap().kind {
            RuntimeVisualNodeKind::Displace(params) => params,
            _ => panic!("displace node"),
        };
        let grain_of = |rack: &RuntimeVisualRack| match rack.get(grain_id).unwrap().kind {
            RuntimeVisualNodeKind::Grain(params) => params,
            _ => panic!("grain node"),
        };

        let mut diced = rack.clone();
        mutate_runtime_rack_values(&mut diced, 1.0, 7, 11, DiceRackScope::Master);
        let after = params_of(&diced);
        assert_eq!(after.tap, authored.tap, "Dice never reroutes a donor");
        assert_eq!(
            after.boundary, authored.boundary,
            "the boundary law is stable authored topology"
        );
        assert!(
            after.amount_x != authored.amount_x || after.amount_y != authored.amount_y,
            "Dice must actually move at least one gain at full amount"
        );
        assert!((-1.0..=1.0).contains(&after.amount_x));
        assert!((-1.0..=1.0).contains(&after.amount_y));

        // Determinism: the same seed/stream reproduces the same gains.
        let mut repeated = rack.clone();
        mutate_runtime_rack_values(&mut repeated, 1.0, 7, 11, DiceRackScope::Master);
        assert_eq!(params_of(&repeated), after);

        // Zero amount is an exact no-op across the whole rack.
        let mut untouched = rack.clone();
        mutate_runtime_rack_values(&mut untouched, 0.0, 7, 11, DiceRackScope::Master);
        assert_eq!(untouched, baseline);

        // A neighbouring node draws from its own stable domain, so appending
        // Displace cannot perturb an older kind's stream.
        let mut without_displace = RuntimeVisualRack::empty();
        without_displace
            .push(RuntimeVisualNodeKind::Displace(authored))
            .unwrap();
        let solo_grain = without_displace
            .push(RuntimeVisualNodeKind::Grain(
                crate::visual_rack::GrainParams::default(),
            ))
            .unwrap();
        assert_eq!(solo_grain, grain_id);
        mutate_runtime_rack_values(&mut without_displace, 1.0, 7, 11, DiceRackScope::Master);
        assert_eq!(grain_of(&without_displace), grain_of(&diced));
    }

    /// Dice moves only the Symmetry Field's declared continuous controls. The
    /// two image routes, the two motion routes, the mode, the boundary, the
    /// authored seed, and the six mask bits are stable authored topology, so
    /// the 32-record sector table is bit-identical before and after.
    #[test]
    fn dice_moves_symmetry_continuous_values_only_and_never_its_routes_masks_or_seed() {
        use crate::motion::MotionDonor;
        use crate::symmetry::{
            RuntimeSymmetryParams, SymmetryBoundary, SymmetryMode, SymmetryMotionMask,
            SymmetryNodeDomain, SymmetrySourceMask,
        };
        use crate::visual_rack::{
            EdgeTiming, ResolvedImageSource, ResolvedImageTap, RuntimeVisualNodeKind,
            RuntimeVisualRack,
        };

        let saved_position = crate::performance::SavedLayerPosition::new(3).unwrap();
        let authored = RuntimeSymmetryParams {
            mode: SymmetryMode::LogSpiral,
            base_folds: 5.0,
            boundary: SymmetryBoundary::Mirror,
            seed: 909,
            source_mask: SymmetrySourceMask {
                carrier: true,
                donor0: true,
                donor1: true,
                clean_history: false,
            },
            motion_mask: SymmetryMotionMask {
                slot0: false,
                slot1: true,
            },
            donors: [
                ResolvedImageTap {
                    source: ResolvedImageSource::CleanProgram,
                    timing: EdgeTiming::PreviousFrame,
                },
                ResolvedImageTap {
                    source: ResolvedImageSource::AllBelow,
                    timing: EdgeTiming::CurrentFrame,
                },
            ],
            motion: [MotionDonor::None, MotionDonor::Missing { saved_position }],
            ..RuntimeSymmetryParams::default()
        };
        let mut rack = RuntimeVisualRack::empty();
        let node_id = rack
            .push(RuntimeVisualNodeKind::Symmetry(authored))
            .unwrap();
        let domain = SymmetryNodeDomain::new(0x4d41_5354_4552, node_id.get());
        let table = authored.sector_table(domain);

        let scope = DiceRackScope::Master;
        let mut diced = rack.clone();
        mutate_runtime_rack_values(&mut diced, 1.0, 4_242, 11, scope);
        let RuntimeVisualNodeKind::Symmetry(params) = diced.get(node_id).unwrap().kind else {
            panic!("symmetry node")
        };
        assert!(
            params.base_folds != authored.base_folds
                || params.radial_phase_deg != authored.radial_phase_deg
                || params.hue_span != authored.hue_span,
            "at least one continuous control must move at amount 1.0"
        );
        assert!(params.base_folds >= 1.0 && params.base_folds <= 32.0);
        assert!(params.hue_span >= 0.0 && params.hue_span <= 1.0);
        assert!(params.motion_gain >= -1.0 && params.motion_gain <= 1.0);
        assert!(params.center[0] >= -1.0 && params.center[0] <= 2.0);

        assert_eq!(params.donors, authored.donors);
        assert_eq!(params.motion, authored.motion);
        assert_eq!(params.mode, authored.mode);
        assert_eq!(params.boundary, authored.boundary);
        assert_eq!(params.seed, authored.seed);
        assert_eq!(params.source_mask, authored.source_mask);
        assert_eq!(params.motion_mask, authored.motion_mask);
        assert_eq!(params.sector_table(domain), table);

        // The same seed and stream reproduce exactly, and amount zero is an
        // exact whole-rack no-op.
        let mut repeat = rack.clone();
        mutate_runtime_rack_values(&mut repeat, 1.0, 4_242, 11, scope);
        assert_eq!(repeat, diced);
        let mut untouched = rack.clone();
        mutate_runtime_rack_values(&mut untouched, 0.0, 4_242, 11, scope);
        assert_eq!(untouched, rack);

        // A neighbouring node is byte-identical with and without this kind
        // present, because every node draws from its own stable domain.
        let grain = crate::visual_rack::GrainParams::default();
        let mut with_symmetry = RuntimeVisualRack::empty();
        with_symmetry
            .push(RuntimeVisualNodeKind::Symmetry(authored))
            .unwrap();
        let neighbour = with_symmetry
            .push(RuntimeVisualNodeKind::Grain(grain))
            .unwrap();
        let mut without_symmetry = RuntimeVisualRack::empty();
        without_symmetry
            .push(RuntimeVisualNodeKind::Displace(
                crate::visual_rack::RuntimeDisplaceParams::default(),
            ))
            .unwrap();
        let same_slot = without_symmetry
            .push(RuntimeVisualNodeKind::Grain(grain))
            .unwrap();
        assert_eq!(same_slot, neighbour, "the neighbour keeps its stable id");
        mutate_runtime_rack_values(&mut with_symmetry, 1.0, 4_242, 11, scope);
        mutate_runtime_rack_values(&mut without_symmetry, 1.0, 4_242, 11, scope);
        assert_eq!(
            with_symmetry.get(neighbour).unwrap().kind,
            without_symmetry.get(neighbour).unwrap().kind
        );
    }

    /// Dice moves the Scan Processor's fifteen continuous controls only. The
    /// two geometry counts and the two reversals are stable authored
    /// topology for Dice's purposes, the same seed and stream reproduce
    /// exactly, amount zero is a whole-rack no-op, and a neighbouring node is
    /// byte-identical with and without the scan present because every node
    /// draws from its own stable domain.
    #[test]
    fn dice_moves_scan_processor_continuous_values_only_and_never_its_geometry_or_reversals() {
        use crate::scan_processor::ScanProcessorParams;
        use crate::visual_rack::{RuntimeVisualNodeKind, RuntimeVisualRack};

        let authored = ScanProcessorParams {
            amount: 0.4,
            lines: 240,
            samples_per_line: 96,
            reverse_h: true,
            reverse_v: false,
            ..ScanProcessorParams::default()
        };
        let mut rack = RuntimeVisualRack::empty();
        let node_id = rack
            .push(RuntimeVisualNodeKind::ScanProcessor(authored))
            .unwrap();

        let scope = DiceRackScope::Master;
        let mut diced = rack.clone();
        mutate_runtime_rack_values(&mut diced, 1.0, 4_242, 11, scope);
        let RuntimeVisualNodeKind::ScanProcessor(params) = diced.get(node_id).unwrap().kind else {
            panic!("scan processor node")
        };
        assert!(
            params.amount != authored.amount
                || params.s_curve != authored.s_curve
                || params.hue != authored.hue,
            "at least one continuous control must move at amount 1.0"
        );
        assert!(params.amount >= 0.0 && params.amount <= 1.0);
        assert!(params.tilt_x >= -1.0 && params.tilt_x <= 1.0);
        assert!(params.osc_lock >= 0.0 && params.osc_lock <= 1.0);
        assert_eq!(params.lines, authored.lines);
        assert_eq!(params.samples_per_line, authored.samples_per_line);
        assert_eq!(params.reverse_h, authored.reverse_h);
        assert_eq!(params.reverse_v, authored.reverse_v);

        let mut repeat = rack.clone();
        mutate_runtime_rack_values(&mut repeat, 1.0, 4_242, 11, scope);
        assert_eq!(repeat, diced);
        let mut untouched = rack.clone();
        mutate_runtime_rack_values(&mut untouched, 0.0, 4_242, 11, scope);
        assert_eq!(untouched, rack);

        let grain = crate::visual_rack::GrainParams::default();
        let mut with_scan = RuntimeVisualRack::empty();
        with_scan
            .push(RuntimeVisualNodeKind::ScanProcessor(authored))
            .unwrap();
        let neighbour = with_scan.push(RuntimeVisualNodeKind::Grain(grain)).unwrap();
        let mut without_scan = RuntimeVisualRack::empty();
        without_scan
            .push(RuntimeVisualNodeKind::Displace(
                crate::visual_rack::RuntimeDisplaceParams::default(),
            ))
            .unwrap();
        let same_slot = without_scan
            .push(RuntimeVisualNodeKind::Grain(grain))
            .unwrap();
        assert_eq!(same_slot, neighbour, "the neighbour keeps its stable id");
        mutate_runtime_rack_values(&mut with_scan, 1.0, 4_242, 11, scope);
        mutate_runtime_rack_values(&mut without_scan, 1.0, 4_242, 11, scope);
        assert_eq!(
            with_scan.get(neighbour).unwrap().kind,
            without_scan.get(neighbour).unwrap().kind
        );
    }

    #[test]
    fn dice_moves_residual_values_only_and_leaves_older_streams_bit_identical() {
        use crate::visual_rack::{
            EdgeTiming, ResidualBlock, ResidualQuantization, ResolvedImageSource, ResolvedImageTap,
            RuntimeResidualParams,
        };

        let authored = RuntimeResidualParams {
            structure: ResolvedImageTap {
                source: ResolvedImageSource::CleanProgram,
                timing: EdgeTiming::PreviousFrame,
            },
            detail: ResolvedImageTap {
                source: ResolvedImageSource::AllBelow,
                timing: EdgeTiming::CurrentFrame,
            },
            block: ResidualBlock::Sixteen,
            quantization: ResidualQuantization::Medium,
            mix: 0.5,
            detail_gain: 2.0,
            seed: 0x00c0_ffee,
            ..RuntimeResidualParams::default()
        };

        // The Grain and Shift nodes are pushed first so their NodeIds — and
        // therefore their Dice domains — are identical in both racks.
        let build = |with_residual: bool| {
            let mut rack = RuntimeVisualRack::empty();
            let grain_id = rack
                .push(RuntimeVisualNodeKind::Grain(
                    crate::visual_rack::GrainParams::default(),
                ))
                .unwrap();
            let shift_id = rack
                .push(RuntimeVisualNodeKind::Shift(
                    crate::visual_rack::ShiftParams::default(),
                ))
                .unwrap();
            let residual_id = with_residual.then(|| {
                rack.push(RuntimeVisualNodeKind::Residual(authored))
                    .unwrap()
            });
            (rack, grain_id, shift_id, residual_id)
        };
        let (rack, grain_id, shift_id, residual_id) = build(true);
        let residual_id = residual_id.unwrap();
        let baseline = rack.clone();

        let params_of = |rack: &RuntimeVisualRack| match rack.get(residual_id).unwrap().kind {
            RuntimeVisualNodeKind::Residual(params) => params,
            _ => panic!("residual node"),
        };

        let mut diced = rack.clone();
        mutate_runtime_rack_values(&mut diced, 1.0, 7, 11, DiceRackScope::Master);
        let after = params_of(&diced);
        assert_eq!(
            after.routes(),
            authored.routes(),
            "Dice never reroutes either donor"
        );
        assert_eq!(
            (after.block, after.quantization),
            (authored.block, authored.quantization),
            "the block and quantization vocabularies are stable authored topology"
        );
        assert_eq!(
            after.seed, authored.seed,
            "the quantization seed is authored topology, not a diced value"
        );
        assert_eq!(after.algorithm_version, authored.algorithm_version);
        assert!(
            after.mix != authored.mix || after.detail_gain != authored.detail_gain,
            "Dice must actually move at least one value at full amount"
        );
        assert!((0.0..=1.0).contains(&after.mix));
        assert!((0.0..=4.0).contains(&after.detail_gain));

        // Determinism: the same amount/seed/stream/scope reproduces exactly.
        let mut repeated = rack.clone();
        mutate_runtime_rack_values(&mut repeated, 1.0, 7, 11, DiceRackScope::Master);
        assert_eq!(params_of(&repeated), after);

        // Zero amount is an exact no-op across the whole rack.
        let mut untouched = rack.clone();
        mutate_runtime_rack_values(&mut untouched, 0.0, 7, 11, DiceRackScope::Master);
        assert_eq!(untouched, baseline);

        // Domain separation: every pre-existing node's stream is bit-identical
        // whether or not the Residual node is present in the rack.
        let (older, older_grain, older_shift, _) = build(false);
        assert_eq!((older_grain, older_shift), (grain_id, shift_id));
        let mut older_diced = older;
        mutate_runtime_rack_values(&mut older_diced, 1.0, 7, 11, DiceRackScope::Master);
        for node_id in [grain_id, shift_id] {
            assert_eq!(
                older_diced.get(node_id).unwrap(),
                diced.get(node_id).unwrap(),
                "appending a Residual node must not perturb an older Dice stream"
            );
        }
    }
}
