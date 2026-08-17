//! Deterministic, patch-only procedural generation.
//!
//! Generation deliberately produces inspectable YAML, a deterministic
//! manifest, and a bounded source-preflight receipt rather than starting GPU
//! exports. Rendering remains an explicit second step, so a large request
//! cannot monopolize the live renderer and every variant can be curated before
//! expensive media work begins.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::layers::BlendMode;
use crate::media_source::{
    ContentIdentity, FingerprintLimits, FingerprintSession, ResolveContext, ResolvedVisualSource,
    DEFAULT_MAX_FINGERPRINT_BYTES, DEFAULT_MAX_SEARCH_ENTRIES,
};
use crate::patch::{EffectsConfig, MotionConfig, PatchState, TemporalOriginalsConfig};
use crate::randomization::{
    mutate_circular, mutate_discrete, mutate_linear, mutate_log, SplitMix64,
};
#[cfg(test)]
use crate::randomization::{reflect, wrap};
use crate::spatial::{
    SpatialTransform, ANCHOR_MAX, ANCHOR_MIN, CROP_MAX, POSITION_MAX, POSITION_MIN, SCALE_MAX,
    SCALE_MIN, SKEW_LIMIT_DEGREES,
};
use crate::visual_rack::{GroupId, MaskParams, NodeId, VisualNodeKind, VisualRack};

/// v6 records the M3 Temporal Originals generation law; v7 adds M4 Motion in
/// new isolated domains. Manifest readers remain data-driven and accept every
/// earlier version string.
pub const GENERATOR_VERSION: &str = "7";
pub const MAX_GENERATED_COUNT: usize = 256;
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;
pub const PREFLIGHT_SCHEMA_VERSION: u32 = 1;
pub const CANONICAL_IDENTITY_ALGORITHM: &str = "normalized-patch-json-v1+sha256";

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub seed: u64,
    pub count: usize,
    pub temperature: f32,
    pub allow_black_sources: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub schema_version: u32,
    pub generator_version: String,
    pub seed: u64,
    pub index: usize,
    pub temperature: f32,
    pub title: String,
    pub slug: String,
    /// Legacy compatibility field. New lineage authority is `anchor_sha256`.
    #[serde(default)]
    pub anchor_fnv1a64: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub canonical_identity_algorithm: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub anchor_sha256: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub piece_sha256: String,
    #[serde(default)]
    pub identity_complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lineage: Option<String>,
    pub logical_sources: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<ManifestSource>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestSource {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_index: Option<usize>,
    pub logical_name: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_len: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offline_policy: Option<String>,
    #[serde(default)]
    pub verified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreflightLimits {
    pub max_search_entries: usize,
    pub max_fingerprint_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreflightReceipt {
    pub schema_version: u32,
    pub canonical_identity_algorithm: String,
    pub anchor_sha256: String,
    pub piece_sha256: String,
    pub status: String,
    pub claim_scope: String,
    pub pixel_identity_claimed: bool,
    pub source_files: usize,
    pub source_bytes: u64,
    pub limits: PreflightLimits,
    pub sources: Vec<ManifestSource>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SourcePreflightConfig {
    pub anchor_dir: Option<PathBuf>,
    pub library_dir: Option<PathBuf>,
    pub max_fingerprint_bytes: u64,
    pub allow_unverified_sources: bool,
}

impl Default for SourcePreflightConfig {
    fn default() -> Self {
        Self {
            anchor_dir: None,
            library_dir: None,
            max_fingerprint_bytes: DEFAULT_MAX_FINGERPRINT_BYTES,
            allow_unverified_sources: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SourceInventory {
    sources: Vec<ManifestSource>,
    limits: PreflightLimits,
    warnings: Vec<String>,
    identity_complete: bool,
}

#[derive(Clone)]
pub struct GeneratedPiece {
    pub patch: PatchState,
    pub manifest: Manifest,
    pub preflight: PreflightReceipt,
}

fn domain_seed(seed: u64, index: usize, domain: u64) -> u64 {
    seed ^ (index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93) ^ domain
}

fn finite_temperature(value: f32) -> Result<f32, String> {
    if !value.is_finite() || !(0.0..=2.0).contains(&value) {
        return Err("temperature must be finite and between 0 and 2".to_string());
    }
    Ok(value)
}

fn canonical_blend_mode(key: &str) -> BlendMode {
    BlendMode::from_key(key).unwrap_or(BlendMode::Normal)
}

/// Keep blend mutation in the established layer RNG stream: parsing the
/// expanded typed set consumes no entropy and `mutate_discrete` performs the
/// same mean-reversion/chance/selection draws it used for the original four.
fn mutate_blend_mode(
    anchor: &str,
    value: &str,
    change_probability: f32,
    rng: &mut SplitMix64,
) -> BlendMode {
    mutate_discrete(
        canonical_blend_mode(anchor),
        canonical_blend_mode(value),
        &BlendMode::ALL,
        change_probability,
        rng,
    )
}

fn mutate_bool(anchor: bool, value: bool, change_probability: f32, rng: &mut SplitMix64) -> bool {
    mutate_discrete(anchor, value, &[false, true], change_probability, rng)
}

fn mutate_effects(
    anchor: &EffectsConfig,
    e: &mut EffectsConfig,
    temperature: f32,
    rng: &mut SplitMix64,
) {
    if temperature == 0.0 {
        return;
    }
    let t = temperature;
    e.pixelate = mutate_log(anchor.pixelate, e.pixelate, 1.0, 32.0, t * 0.45, rng).round();
    e.rgb_split = mutate_linear(anchor.rgb_split, e.rgb_split, 0.0, 30.0, t * 4.0, rng);
    e.hue_shift = mutate_circular(anchor.hue_shift, e.hue_shift, -180.0, 180.0, t * 50.0, rng);
    e.saturation = mutate_linear(anchor.saturation, e.saturation, -1.0, 1.0, t * 0.3, rng);
    e.brightness = mutate_linear(anchor.brightness, e.brightness, -1.0, 1.0, t * 0.25, rng);
    e.contrast = mutate_linear(anchor.contrast, e.contrast, -1.0, 1.0, t * 0.3, rng);
    e.posterize = mutate_linear(anchor.posterize, e.posterize, 0.0, 16.0, t * 2.0, rng).round();
    e.downsample = mutate_log(anchor.downsample, e.downsample, 0.05, 1.0, t * 0.22, rng);
    e.grain_intensity = mutate_linear(
        anchor.grain_intensity,
        e.grain_intensity,
        0.0,
        0.3,
        t * 0.06,
        rng,
    );
    e.grain_size = mutate_linear(anchor.grain_size, e.grain_size, 1.0, 4.0, t * 0.5, rng);
    e.vignette = mutate_linear(anchor.vignette, e.vignette, 0.0, 1.5, t * 0.22, rng);
    e.color_drift = mutate_linear(anchor.color_drift, e.color_drift, 0.0, 0.02, t * 0.004, rng);
    e.breathe_scale = mutate_linear(
        anchor.breathe_scale,
        e.breathe_scale,
        0.0,
        0.05,
        t * 0.009,
        rng,
    );
    e.breathe_rotation = mutate_linear(
        anchor.breathe_rotation,
        e.breathe_rotation,
        0.0,
        2.0,
        t * 0.32,
        rng,
    );
    e.breathe_position = mutate_linear(
        anchor.breathe_position,
        e.breathe_position,
        0.0,
        0.02,
        t * 0.004,
        rng,
    );
    e.key_threshold = mutate_linear(
        anchor.key_threshold,
        e.key_threshold,
        0.0,
        1.0,
        t * 0.1,
        rng,
    );
    e.key_softness = mutate_linear(anchor.key_softness, e.key_softness, 0.0, 0.5, t * 0.07, rng);
    e.cellular_amount = mutate_linear(
        anchor.cellular_amount,
        e.cellular_amount,
        0.0,
        1.0,
        t * 0.14,
        rng,
    );
    e.cellular_scale = mutate_log(
        anchor.cellular_scale,
        e.cellular_scale,
        2.0,
        32.0,
        t * 0.25,
        rng,
    );
    e.cellular_warp = mutate_linear(
        anchor.cellular_warp,
        e.cellular_warp,
        0.0,
        1.0,
        t * 0.12,
        rng,
    );
    e.cellular_speed = mutate_linear(
        anchor.cellular_speed,
        e.cellular_speed,
        0.0,
        2.0,
        t * 0.25,
        rng,
    );
    e.shift_amount = mutate_linear(anchor.shift_amount, e.shift_amount, 0.0, 1.0, t * 0.14, rng);
    e.shift_block_size = mutate_log(
        anchor.shift_block_size,
        e.shift_block_size,
        2.0,
        256.0,
        t * 0.35,
        rng,
    );
    e.shift_density = mutate_linear(
        anchor.shift_density,
        e.shift_density,
        0.0,
        1.0,
        t * 0.12,
        rng,
    );
    e.shift_speed = mutate_linear(anchor.shift_speed, e.shift_speed, 0.0, 20.0, t * 2.5, rng);

    // Discrete changes are deliberately rare. Zero-valued effects activate
    // sparsely so variation does not become a wall of every effect at once.
    let discrete = (temperature * 0.08).clamp(0.0, 0.16);
    if e.invert != anchor.invert && rng.chance(0.15) {
        e.invert = anchor.invert;
    } else if rng.chance(discrete) {
        e.invert = !e.invert;
    }
    if e.color_grain != anchor.color_grain && rng.chance(0.15) {
        e.color_grain = anchor.color_grain;
    } else if rng.chance(discrete) {
        e.color_grain = !e.color_grain;
    }
    e.grain_algo = mutate_discrete(
        anchor.grain_algo,
        e.grain_algo,
        &[0, 1, 2, 3],
        discrete,
        rng,
    );
    // Key selectors remain an explicit artistic choice. Their continuous
    // controls can still evolve deterministically when a chroma key is active
    // (and remain harmless dormant state while it is off).
    for channel in 0..3 {
        e.key_color[channel] = mutate_linear(
            anchor.key_color[channel],
            e.key_color[channel],
            0.0,
            1.0,
            temperature * 0.08,
            rng,
        );
    }
    e.key_tolerance = mutate_linear(
        anchor.key_tolerance,
        e.key_tolerance,
        0.0,
        1.0,
        temperature * 0.08,
        rng,
    );
}

// Appended M3 streams. Every numeric value is isolated so introducing a new
// control cannot shift an established procedural sequence. Discrete laws,
// public seeds, loop-driver identity, and reset policy never mutate here.
const PROCEDURAL_TEMPORAL_LOOM_AMOUNT: u64 = 0x5033_4c4f_4f4d_414d;
const PROCEDURAL_TEMPORAL_LOOM_DEPTH: u64 = 0x5033_4c4f_4f4d_4450;
const PROCEDURAL_TEMPORAL_LOOM_PHASE: u64 = 0x5033_4c4f_4f4d_5048;
const PROCEDURAL_TEMPORAL_LOOM_SCALE: u64 = 0x5033_4c4f_4f4d_5343;
const PROCEDURAL_TEMPORAL_LOOM_ANGLE: u64 = 0x5033_4c4f_4f4d_414e;
const PROCEDURAL_TEMPORAL_LOOM_FOLDS: u64 = 0x5033_4c4f_4f4d_464f;
const PROCEDURAL_TEMPORAL_LOOM_QUANT: u64 = 0x5033_4c4f_4f4d_5155;
const PROCEDURAL_TEMPORAL_ATLAS_AMOUNT: u64 = 0x5033_4154_4c53_414d;
const PROCEDURAL_TEMPORAL_ATLAS_TERRITORIES: u64 = 0x5033_4154_4c53_5445;
const PROCEDURAL_TEMPORAL_ATLAS_COLLISION: u64 = 0x5033_4154_4c53_434f;
const PROCEDURAL_TEMPORAL_GARDEN_AMOUNT: u64 = 0x5033_4741_5244_414d;
const PROCEDURAL_TEMPORAL_GARDEN_THRESHOLD: u64 = 0x5033_4741_5244_5448;
const PROCEDURAL_TEMPORAL_GARDEN_SOFTNESS: u64 = 0x5033_4741_5244_534f;
const PROCEDURAL_TEMPORAL_GARDEN_DECAY: u64 = 0x5033_4741_5244_4443;
const PROCEDURAL_TEMPORAL_GARDEN_HOLD: u64 = 0x5033_4741_5244_484f;

fn mutate_temporal_originals(
    anchor: &TemporalOriginalsConfig,
    value: &mut TemporalOriginalsConfig,
    temperature: f32,
    seed: u64,
    index: usize,
) {
    if temperature == 0.0 {
        return;
    }
    macro_rules! linear {
        ($field:expr, $anchor:expr, $min:expr, $max:expr, $scale:expr, $domain:expr) => {{
            let mut rng = SplitMix64::new(domain_seed(seed, index, $domain));
            $field = mutate_linear($anchor, $field, $min, $max, temperature * $scale, &mut rng);
        }};
    }

    linear!(
        value.loom.amount,
        anchor.loom.amount,
        0.0,
        1.0,
        0.2,
        PROCEDURAL_TEMPORAL_LOOM_AMOUNT
    );
    linear!(
        value.loom.depth,
        anchor.loom.depth,
        0.0,
        1.0,
        0.2,
        PROCEDURAL_TEMPORAL_LOOM_DEPTH
    );
    linear!(
        value.loom.phase,
        anchor.loom.phase,
        -1_000.0,
        1_000.0,
        0.25,
        PROCEDURAL_TEMPORAL_LOOM_PHASE
    );
    {
        let mut rng = SplitMix64::new(domain_seed(seed, index, PROCEDURAL_TEMPORAL_LOOM_SCALE));
        value.loom.scale = mutate_log(
            anchor.loom.scale,
            value.loom.scale,
            0.01,
            100.0,
            temperature * 0.25,
            &mut rng,
        );
    }
    {
        let mut rng = SplitMix64::new(domain_seed(seed, index, PROCEDURAL_TEMPORAL_LOOM_ANGLE));
        value.loom.angle = mutate_circular(
            anchor.loom.angle,
            value.loom.angle,
            -180.0,
            180.0,
            temperature * 30.0,
            &mut rng,
        );
    }
    {
        let mut rng = SplitMix64::new(domain_seed(seed, index, PROCEDURAL_TEMPORAL_LOOM_FOLDS));
        value.loom.folds = mutate_linear(
            f32::from(anchor.loom.folds),
            f32::from(value.loom.folds),
            1.0,
            16.0,
            temperature * 2.0,
            &mut rng,
        )
        .round() as u8;
    }
    {
        let mut rng = SplitMix64::new(domain_seed(seed, index, PROCEDURAL_TEMPORAL_LOOM_QUANT));
        value.loom.quantization = mutate_linear(
            f32::from(anchor.loom.quantization),
            f32::from(value.loom.quantization),
            0.0,
            24.0,
            temperature * 3.0,
            &mut rng,
        )
        .round() as u8;
    }
    linear!(
        value.atlas.amount,
        anchor.atlas.amount,
        0.0,
        1.0,
        0.2,
        PROCEDURAL_TEMPORAL_ATLAS_AMOUNT
    );
    {
        let mut rng = SplitMix64::new(domain_seed(
            seed,
            index,
            PROCEDURAL_TEMPORAL_ATLAS_TERRITORIES,
        ));
        value.atlas.territories = mutate_linear(
            f32::from(anchor.atlas.territories),
            f32::from(value.atlas.territories),
            1.0,
            64.0,
            temperature * 6.0,
            &mut rng,
        )
        .round() as u8;
    }
    linear!(
        value.atlas.collision,
        anchor.atlas.collision,
        0.0,
        1.0,
        0.2,
        PROCEDURAL_TEMPORAL_ATLAS_COLLISION
    );
    linear!(
        value.garden.amount,
        anchor.garden.amount,
        0.0,
        1.0,
        0.2,
        PROCEDURAL_TEMPORAL_GARDEN_AMOUNT
    );
    linear!(
        value.garden.threshold,
        anchor.garden.threshold,
        0.0,
        1.0,
        0.15,
        PROCEDURAL_TEMPORAL_GARDEN_THRESHOLD
    );
    linear!(
        value.garden.softness,
        anchor.garden.softness,
        0.0,
        0.5,
        0.08,
        PROCEDURAL_TEMPORAL_GARDEN_SOFTNESS
    );
    linear!(
        value.garden.decay,
        anchor.garden.decay,
        0.0,
        1.0,
        0.15,
        PROCEDURAL_TEMPORAL_GARDEN_DECAY
    );
    {
        let mut rng = SplitMix64::new(domain_seed(seed, index, PROCEDURAL_TEMPORAL_GARDEN_HOLD));
        let anchor = f64::from(anchor.garden.max_hold_ticks);
        let current = f64::from(value.garden.max_hold_ticks);
        let candidate = anchor
            + 0.85 * (current - anchor)
            + f64::from(rng.signed()) * f64::from(temperature) * 30.0;
        value.garden.max_hold_ticks = candidate.round().clamp(0.0, f64::from(u32::MAX)) as u32;
    }
}

// M4 numeric values live in field-isolated streams. Master and each persisted
// saved-layer position also own separate domains, so Motion cannot perturb
// any v1-v6 generator sequence or a sibling scope. Every topology/provenance
// field remains exactly authored.
const PROCEDURAL_MOTION_DOMAIN: u64 = 0x5034_4d4f_5449_4f4e;
const PROCEDURAL_MOTION_AMOUNT: u64 = 0x414d_4f55_4e54_0001;
const PROCEDURAL_MOTION_THRESHOLD: u64 = 0x5448_5245_5348_0001;
const PROCEDURAL_MOTION_SOFTNESS: u64 = 0x534f_4654_4e45_5353;
const PROCEDURAL_MOTION_REFRESH: u64 = 0x5245_4652_4553_4801;
const PROCEDURAL_MOTION_DECAY: u64 = 0x4445_4341_5900_0001;
const PROCEDURAL_MOTION_OCCLUSION: u64 = 0x4f43_434c_5553_494f;
const PROCEDURAL_MOTION_SHUTTER_ANGLE: u64 = 0x5348_5554_414e_474c;
const PROCEDURAL_MOTION_SHUTTER_PHASE: u64 = 0x5348_5554_5048_4153;
const PROCEDURAL_MOTION_SHUTTER_CURVE: u64 = 0x5348_5554_4355_5256;
const PROCEDURAL_MOTION_CHROMATIC_LAG: u64 = 0x4348_524f_4d4c_4147;

fn mutate_motion_config(
    anchor: &MotionConfig,
    value: &mut MotionConfig,
    temperature: f32,
    seed: u64,
    index: usize,
    owner_domain: u64,
    include_faraday: bool,
) {
    if temperature == 0.0 {
        return;
    }
    macro_rules! linear {
        ($field:expr, $anchor:expr, $min:expr, $max:expr, $scale:expr, $domain:expr) => {{
            let mut rng = SplitMix64::new(domain_seed(
                seed,
                index,
                PROCEDURAL_MOTION_DOMAIN ^ owner_domain ^ $domain,
            ));
            $field = mutate_linear($anchor, $field, $min, $max, temperature * $scale, &mut rng);
        }};
    }

    if include_faraday {
        linear!(
            value.transplant.amount,
            anchor.transplant.amount,
            0.0,
            1.0,
            0.2,
            PROCEDURAL_MOTION_AMOUNT
        );
        linear!(
            value.transplant.confidence_threshold,
            anchor.transplant.confidence_threshold,
            0.0,
            1.0,
            0.15,
            PROCEDURAL_MOTION_THRESHOLD
        );
        linear!(
            value.transplant.confidence_softness,
            anchor.transplant.confidence_softness,
            0.0,
            0.5,
            0.08,
            PROCEDURAL_MOTION_SOFTNESS
        );
        linear!(
            value.transplant.refresh,
            anchor.transplant.refresh,
            0.0,
            1.0,
            0.15,
            PROCEDURAL_MOTION_REFRESH
        );
        linear!(
            value.transplant.decay,
            anchor.transplant.decay,
            0.0,
            1.0,
            0.15,
            PROCEDURAL_MOTION_DECAY
        );
        linear!(
            value.transplant.occlusion,
            anchor.transplant.occlusion,
            0.0,
            1.0,
            0.15,
            PROCEDURAL_MOTION_OCCLUSION
        );
    }
    linear!(
        value.shutter.angle_degrees,
        anchor.shutter.angle_degrees,
        0.0,
        360.0,
        60.0,
        PROCEDURAL_MOTION_SHUTTER_ANGLE
    );
    linear!(
        value.shutter.phase,
        anchor.shutter.phase,
        -1.0,
        1.0,
        0.25,
        PROCEDURAL_MOTION_SHUTTER_PHASE
    );
    linear!(
        value.shutter.curvature,
        anchor.shutter.curvature,
        -2.0,
        2.0,
        0.5,
        PROCEDURAL_MOTION_SHUTTER_CURVE
    );
    linear!(
        value.shutter.chromatic_lag,
        anchor.shutter.chromatic_lag,
        0.0,
        1.0,
        0.15,
        PROCEDURAL_MOTION_CHROMATIC_LAG
    );
    *value = value.sanitized();
}

fn mutate_transform(
    anchor: &SpatialTransform,
    transform: &mut SpatialTransform,
    temperature: f32,
    rng: &mut SplitMix64,
) {
    if temperature == 0.0 {
        return;
    }
    let anchor = anchor.sanitized();
    let mut value = transform.sanitized();
    for axis in 0..2 {
        value.position[axis] = mutate_linear(
            anchor.position[axis],
            value.position[axis],
            POSITION_MIN,
            POSITION_MAX,
            temperature * 0.25,
            rng,
        );
        value.scale[axis] = mutate_linear(
            anchor.scale[axis],
            value.scale[axis],
            SCALE_MIN,
            SCALE_MAX,
            temperature * 0.35,
            rng,
        );
        value.anchor[axis] = mutate_linear(
            anchor.anchor[axis],
            value.anchor[axis],
            ANCHOR_MIN,
            ANCHOR_MAX,
            temperature * 0.10,
            rng,
        );
    }
    value.rotation_deg = mutate_circular(
        anchor.rotation_deg,
        value.rotation_deg,
        -180.0,
        180.0,
        temperature * 30.0,
        rng,
    );
    value.skew_deg = mutate_linear(
        anchor.skew_deg,
        value.skew_deg,
        -SKEW_LIMIT_DEGREES,
        SKEW_LIMIT_DEGREES,
        temperature * 12.0,
        rng,
    );
    value.skew_axis_deg = mutate_circular(
        anchor.skew_axis_deg,
        value.skew_axis_deg,
        -180.0,
        180.0,
        temperature * 25.0,
        rng,
    );
    for side in 0..4 {
        value.crop[side] = mutate_linear(
            anchor.crop[side],
            value.crop[side],
            0.0,
            CROP_MAX,
            temperature * 0.04,
            rng,
        );
    }
    // Fit/edge/sampling are intentionally not randomized. Sanitize once after
    // all four crop sides so their paired extents remain valid.
    *transform = value.sanitized();
}

/// Saved procedural pieces do not own runtime layer IDs, so rack owners use
/// the stable identity available in the persisted graph: the singleton master
/// domain, saved layer position, or GroupId. NodeId then isolates every node's
/// stream from rack insertion and reordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProceduralRackOwner {
    Master,
    Layer(usize),
    Group(GroupId),
}

const PROCEDURAL_CREATIVE_DOMAIN: u64 = 0x4352_4541_5449_5645;
const PROCEDURAL_GROUP_VALUES_DOMAIN: u64 = 0x4752_4f55_505f_5641;
const PROCEDURAL_GROUP_MATTE_DOMAIN: u64 = 0x4752_4f55_505f_4d41;

const fn procedural_owner_domain(owner: ProceduralRackOwner) -> u64 {
    match owner {
        ProceduralRackOwner::Master => 0x4d41_5354_4552_0005,
        ProceduralRackOwner::Layer(position) => {
            0x4c41_5945_5200_0005 ^ (position as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        }
        ProceduralRackOwner::Group(group_id) => {
            0x4752_4f55_5000_0005 ^ group_id.get().wrapping_mul(0xd6e8_feb8_6659_fd93)
        }
    }
}

fn procedural_node_rng(
    seed: u64,
    index: usize,
    owner: ProceduralRackOwner,
    node_id: NodeId,
) -> SplitMix64 {
    SplitMix64::new(domain_seed(
        seed,
        index,
        PROCEDURAL_CREATIVE_DOMAIN
            ^ procedural_owner_domain(owner)
            ^ node_id.get().wrapping_mul(0xa076_1d64_78bd_642f),
    ))
}

/// Only values that can wake a saved image edge need transactional fallback.
/// Every other procedural rack/group mutation is graph-topology neutral.
#[derive(Debug, Clone, Copy)]
enum CreativeEdgeFallback {
    NodeWet {
        owner: ProceduralRackOwner,
        node_id: NodeId,
        prior: f32,
    },
    ImageMaskAmount {
        owner: ProceduralRackOwner,
        node_id: NodeId,
        prior: f32,
    },
    GroupMatteAmount {
        group_id: GroupId,
        prior: f32,
    },
}

fn saved_rack_mut(patch: &mut PatchState, owner: ProceduralRackOwner) -> Option<&mut VisualRack> {
    match owner {
        ProceduralRackOwner::Master => patch.master_rack.as_mut(),
        ProceduralRackOwner::Layer(position) => patch.layers.get_mut(position)?.rack.as_mut(),
        ProceduralRackOwner::Group(group_id) => {
            Some(&mut patch.composition.as_mut()?.group_mut(group_id)?.rack)
        }
    }
}

impl CreativeEdgeFallback {
    fn restore(self, patch: &mut PatchState) {
        match self {
            Self::NodeWet {
                owner,
                node_id,
                prior,
            } => {
                if let Some(node) =
                    saved_rack_mut(patch, owner).and_then(|rack| rack.get_mut(node_id))
                {
                    node.wet = prior;
                }
            }
            Self::ImageMaskAmount {
                owner,
                node_id,
                prior,
            } => {
                let Some(node) =
                    saved_rack_mut(patch, owner).and_then(|rack| rack.get_mut(node_id))
                else {
                    return;
                };
                if let VisualNodeKind::Mask(MaskParams::Image(matte)) = &mut node.kind {
                    matte.amount = prior;
                }
            }
            Self::GroupMatteAmount { group_id, prior } => {
                let Some(matte) = patch
                    .composition
                    .as_mut()
                    .and_then(|composition| composition.group_mut(group_id))
                    .and_then(|group| group.matte.as_mut())
                else {
                    return;
                };
                matte.amount = prior;
            }
        }
    }
}

fn mutate_saved_rack_values(
    anchor: &VisualRack,
    rack: &mut VisualRack,
    temperature: f32,
    seed: u64,
    index: usize,
    owner: ProceduralRackOwner,
    edge_fallbacks: &mut Vec<CreativeEdgeFallback>,
) {
    if temperature == 0.0 {
        return;
    }
    let node_ids: Vec<_> = rack.iter().map(|node| node.stable_id).collect();
    for node_id in node_ids {
        let Some(anchor_node) = anchor.get(node_id).copied() else {
            continue;
        };
        let Some(node) = rack.get_mut(node_id) else {
            continue;
        };
        if matches!(
            node.kind,
            VisualNodeKind::LegacyCanonical | VisualNodeKind::LegacyTemporal
        ) {
            continue;
        }

        let prior_wet = node.wet;
        let prior_image_amount = match node.kind {
            VisualNodeKind::Mask(MaskParams::Image(matte)) => Some(matte.amount),
            _ => None,
        };
        let mut rng = procedural_node_rng(seed, index, owner, node_id);
        node.wet = mutate_linear(
            anchor_node.wet,
            node.wet,
            0.0,
            1.0,
            temperature * 0.25,
            &mut rng,
        );

        match (anchor_node.kind, &mut node.kind) {
            (VisualNodeKind::Transform(anchor), VisualNodeKind::Transform(value)) => {
                mutate_transform(&anchor, value, temperature, &mut rng);
            }
            (VisualNodeKind::DigitalColor(anchor), VisualNodeKind::DigitalColor(value)) => {
                value.pixelate_size = mutate_log(
                    anchor.pixelate_size,
                    value.pixelate_size,
                    1.0,
                    32.0,
                    temperature * 0.45,
                    &mut rng,
                )
                .round();
                value.rgb_split = mutate_linear(
                    anchor.rgb_split,
                    value.rgb_split,
                    0.0,
                    30.0,
                    temperature * 4.0,
                    &mut rng,
                );
                value.downsample = mutate_log(
                    anchor.downsample,
                    value.downsample,
                    0.05,
                    1.0,
                    temperature * 0.22,
                    &mut rng,
                );
                value.hue_shift = mutate_circular(
                    anchor.hue_shift,
                    value.hue_shift,
                    -180.0,
                    180.0,
                    temperature * 50.0,
                    &mut rng,
                );
                value.saturation = mutate_linear(
                    anchor.saturation,
                    value.saturation,
                    -1.0,
                    1.0,
                    temperature * 0.3,
                    &mut rng,
                );
                value.brightness = mutate_linear(
                    anchor.brightness,
                    value.brightness,
                    -1.0,
                    1.0,
                    temperature * 0.25,
                    &mut rng,
                );
                value.contrast = mutate_linear(
                    anchor.contrast,
                    value.contrast,
                    -1.0,
                    1.0,
                    temperature * 0.3,
                    &mut rng,
                );
                value.posterize = mutate_linear(
                    anchor.posterize,
                    value.posterize,
                    0.0,
                    16.0,
                    temperature * 2.0,
                    &mut rng,
                )
                .round();
                value.invert = mutate_linear(
                    anchor.invert,
                    value.invert,
                    0.0,
                    1.0,
                    temperature * 0.25,
                    &mut rng,
                );
                value.vignette = mutate_linear(
                    anchor.vignette,
                    value.vignette,
                    0.0,
                    1.5,
                    temperature * 0.22,
                    &mut rng,
                );
                value.color_drift = mutate_linear(
                    anchor.color_drift,
                    value.color_drift,
                    0.0,
                    0.02,
                    temperature * 0.004,
                    &mut rng,
                );
            }
            (VisualNodeKind::Key(anchor), VisualNodeKind::Key(value)) => {
                value.threshold = mutate_linear(
                    anchor.threshold,
                    value.threshold,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.softness = mutate_linear(
                    anchor.softness,
                    value.softness,
                    0.0,
                    0.5,
                    temperature * 0.1,
                    &mut rng,
                );
                for component in 0..3 {
                    value.color[component] = mutate_linear(
                        anchor.color[component],
                        value.color[component],
                        0.0,
                        1.0,
                        temperature * 0.15,
                        &mut rng,
                    );
                }
                value.tolerance = mutate_linear(
                    anchor.tolerance,
                    value.tolerance,
                    0.0,
                    1.0,
                    temperature * 0.15,
                    &mut rng,
                );
            }
            (VisualNodeKind::Cellular(anchor), VisualNodeKind::Cellular(value)) => {
                value.amount = mutate_linear(
                    anchor.amount,
                    value.amount,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.scale = mutate_log(
                    anchor.scale,
                    value.scale,
                    2.0,
                    32.0,
                    temperature * 0.28,
                    &mut rng,
                );
                value.warp = mutate_linear(
                    anchor.warp,
                    value.warp,
                    0.0,
                    1.0,
                    temperature * 0.18,
                    &mut rng,
                );
                value.speed = mutate_linear(
                    anchor.speed,
                    value.speed,
                    0.0,
                    2.0,
                    temperature * 0.35,
                    &mut rng,
                );
                value.gap_amount = mutate_linear(
                    anchor.gap_amount,
                    value.gap_amount,
                    0.0,
                    1.0,
                    temperature * 0.15,
                    &mut rng,
                );
                value.gap_threshold = mutate_linear(
                    anchor.gap_threshold,
                    value.gap_threshold,
                    0.0,
                    1.0,
                    temperature * 0.15,
                    &mut rng,
                );
                value.gap_softness = mutate_linear(
                    anchor.gap_softness,
                    value.gap_softness,
                    0.0,
                    0.5,
                    temperature * 0.08,
                    &mut rng,
                );
                if rng.chance((temperature * 0.5).clamp(0.0, 1.0)) {
                    value.seed = rng.next_u64() as u32;
                }
            }
            (VisualNodeKind::Shift(anchor), VisualNodeKind::Shift(value)) => {
                value.amount = mutate_linear(
                    anchor.amount,
                    value.amount,
                    0.0,
                    1.0,
                    temperature * 0.25,
                    &mut rng,
                );
                value.block_size = mutate_log(
                    anchor.block_size,
                    value.block_size,
                    2.0,
                    256.0,
                    temperature * 0.35,
                    &mut rng,
                )
                .round();
                value.density = mutate_linear(
                    anchor.density,
                    value.density,
                    0.0,
                    1.0,
                    temperature * 0.15,
                    &mut rng,
                );
                value.speed = mutate_linear(
                    anchor.speed,
                    value.speed,
                    0.0,
                    20.0,
                    temperature * 2.5,
                    &mut rng,
                );
                if rng.chance((temperature * 0.5).clamp(0.0, 1.0)) {
                    value.seed = rng.next_u64() as u32;
                }
            }
            (VisualNodeKind::Grain(anchor), VisualNodeKind::Grain(value)) => {
                value.intensity = mutate_linear(
                    anchor.intensity,
                    value.intensity,
                    0.0,
                    0.3,
                    temperature * 0.06,
                    &mut rng,
                );
                value.size = mutate_linear(
                    anchor.size,
                    value.size,
                    1.0,
                    4.0,
                    temperature * 0.5,
                    &mut rng,
                );
                if rng.chance((temperature * 0.5).clamp(0.0, 1.0)) {
                    value.seed = rng.next_u64() as u32;
                }
            }
            (VisualNodeKind::Mask(anchor), VisualNodeKind::Mask(value)) => match (anchor, value) {
                (MaskParams::Rectangle(anchor), MaskParams::Rectangle(value)) => {
                    for axis in 0..2 {
                        value.center[axis] = mutate_linear(
                            anchor.center[axis],
                            value.center[axis],
                            -2.0,
                            3.0,
                            temperature * 0.15,
                            &mut rng,
                        );
                        value.size[axis] = mutate_linear(
                            anchor.size[axis],
                            value.size[axis],
                            0.0,
                            4.0,
                            temperature * 0.3,
                            &mut rng,
                        );
                    }
                    value.rotation_deg = mutate_circular(
                        anchor.rotation_deg,
                        value.rotation_deg,
                        -180.0,
                        180.0,
                        temperature * 30.0,
                        &mut rng,
                    );
                    value.feather = mutate_linear(
                        anchor.feather,
                        value.feather,
                        0.0,
                        1.0,
                        temperature * 0.15,
                        &mut rng,
                    );
                }
                (MaskParams::Ellipse(anchor), MaskParams::Ellipse(value)) => {
                    for axis in 0..2 {
                        value.center[axis] = mutate_linear(
                            anchor.center[axis],
                            value.center[axis],
                            -2.0,
                            3.0,
                            temperature * 0.15,
                            &mut rng,
                        );
                        value.radii[axis] = mutate_linear(
                            anchor.radii[axis],
                            value.radii[axis],
                            0.0,
                            2.0,
                            temperature * 0.2,
                            &mut rng,
                        );
                    }
                    value.rotation_deg = mutate_circular(
                        anchor.rotation_deg,
                        value.rotation_deg,
                        -180.0,
                        180.0,
                        temperature * 30.0,
                        &mut rng,
                    );
                    value.feather = mutate_linear(
                        anchor.feather,
                        value.feather,
                        0.0,
                        1.0,
                        temperature * 0.15,
                        &mut rng,
                    );
                }
                (MaskParams::Image(anchor), MaskParams::Image(value)) => {
                    value.amount = mutate_linear(
                        anchor.amount,
                        value.amount,
                        0.0,
                        1.0,
                        temperature * 0.2,
                        &mut rng,
                    );
                    value.threshold = mutate_linear(
                        anchor.threshold,
                        value.threshold,
                        0.0,
                        1.0,
                        temperature * 0.2,
                        &mut rng,
                    );
                    value.softness = mutate_linear(
                        anchor.softness,
                        value.softness,
                        0.0,
                        0.5,
                        temperature * 0.1,
                        &mut rng,
                    );
                }
                _ => {}
            },
            (VisualNodeKind::LegacyCanonical | VisualNodeKind::LegacyTemporal, _)
            | (_, VisualNodeKind::LegacyCanonical | VisualNodeKind::LegacyTemporal) => {}
            // A generated patch never changes rack topology. Be defensive if
            // a future schema-normalization step supplies unlike kinds.
            _ => {}
        }

        if !node.enabled {
            continue;
        }
        let current_image_amount = match node.kind {
            VisualNodeKind::Mask(MaskParams::Image(matte)) => Some(matte.amount),
            _ => None,
        };
        let route_effect_active = current_image_amount.is_some_and(|value| value > 0.0);
        if prior_wet <= 0.0 && node.wet > 0.0 && route_effect_active {
            edge_fallbacks.push(CreativeEdgeFallback::NodeWet {
                owner,
                node_id,
                prior: prior_wet,
            });
        }
        if node.wet > 0.0
            && prior_image_amount.is_some_and(|value| value <= 0.0)
            && current_image_amount.is_some_and(|value| value > 0.0)
        {
            edge_fallbacks.push(CreativeEdgeFallback::ImageMaskAmount {
                owner,
                node_id,
                prior: prior_image_amount.unwrap_or(0.0),
            });
        }
    }
}

fn mutate_saved_composition_values(
    anchor: &crate::composition::CompositionTree,
    composition: &mut crate::composition::CompositionTree,
    temperature: f32,
    seed: u64,
    index: usize,
    edge_fallbacks: &mut Vec<CreativeEdgeFallback>,
) {
    if temperature == 0.0 {
        return;
    }
    let group_ids: Vec<_> = composition.groups().map(|group| group.id).collect();
    for group_id in group_ids {
        let Some(anchor_group) = anchor.group(group_id) else {
            continue;
        };
        let Some(group) = composition.group_mut(group_id) else {
            continue;
        };
        let group_domain = procedural_owner_domain(ProceduralRackOwner::Group(group_id));
        let mut group_rng = SplitMix64::new(domain_seed(
            seed,
            index,
            PROCEDURAL_GROUP_VALUES_DOMAIN ^ group_domain,
        ));
        group.opacity = mutate_linear(
            anchor_group.opacity,
            group.opacity,
            0.0,
            1.0,
            temperature * 0.25,
            &mut group_rng,
        );
        mutate_transform(
            &anchor_group.transform,
            &mut group.transform,
            temperature,
            &mut group_rng,
        );

        if let (Some(anchor_matte), Some(matte)) = (anchor_group.matte, group.matte.as_mut()) {
            let prior_amount = matte.amount;
            let mut matte_rng = SplitMix64::new(domain_seed(
                seed,
                index,
                PROCEDURAL_GROUP_MATTE_DOMAIN ^ group_domain,
            ));
            matte.amount = mutate_linear(
                anchor_matte.amount,
                matte.amount,
                0.0,
                1.0,
                temperature * 0.2,
                &mut matte_rng,
            );
            matte.threshold = mutate_linear(
                anchor_matte.threshold,
                matte.threshold,
                0.0,
                1.0,
                temperature * 0.2,
                &mut matte_rng,
            );
            matte.softness = mutate_linear(
                anchor_matte.softness,
                matte.softness,
                0.0,
                0.5,
                temperature * 0.1,
                &mut matte_rng,
            );
            if !group.bypass && prior_amount <= 0.0 && matte.amount > 0.0 {
                edge_fallbacks.push(CreativeEdgeFallback::GroupMatteAmount {
                    group_id,
                    prior: prior_amount,
                });
            }
        }

        mutate_saved_rack_values(
            &anchor_group.rack,
            &mut group.rack,
            temperature,
            seed,
            index,
            ProceduralRackOwner::Group(group_id),
            edge_fallbacks,
        );
    }
}

fn mutate_saved_creative_values(
    anchor: &PatchState,
    patch: &mut PatchState,
    temperature: f32,
    seed: u64,
    index: usize,
    edge_fallbacks: &mut Vec<CreativeEdgeFallback>,
) {
    if temperature == 0.0 {
        return;
    }
    if let (Some(anchor_rack), Some(rack)) = (&anchor.master_rack, &mut patch.master_rack) {
        mutate_saved_rack_values(
            anchor_rack,
            rack,
            temperature,
            seed,
            index,
            ProceduralRackOwner::Master,
            edge_fallbacks,
        );
    }
    for (position, (anchor_layer, layer)) in anchor.layers.iter().zip(&mut patch.layers).enumerate()
    {
        if let (Some(anchor_rack), Some(rack)) = (&anchor_layer.rack, &mut layer.rack) {
            mutate_saved_rack_values(
                anchor_rack,
                rack,
                temperature,
                seed,
                index,
                ProceduralRackOwner::Layer(position),
                edge_fallbacks,
            );
        }
    }
    if let (Some(anchor_composition), Some(composition)) =
        (&anchor.composition, &mut patch.composition)
    {
        mutate_saved_composition_values(
            anchor_composition,
            composition,
            temperature,
            seed,
            index,
            edge_fallbacks,
        );
    }
}

fn validate_generated_patch(patch: &PatchState) -> Result<(), String> {
    let yaml = serde_yaml::to_string(patch)
        .map_err(|error| format!("serialize generated creative graph: {error}"))?;
    serde_yaml::from_str::<PatchState>(&yaml)
        .map(|_| ())
        .map_err(|error| format!("validate generated creative graph: {error}"))
}

fn retain_valid_creative_edge_values(
    patch: &mut PatchState,
    edge_fallbacks: Vec<CreativeEdgeFallback>,
) -> Result<(), String> {
    if validate_generated_patch(patch).is_ok() {
        return Ok(());
    }
    for fallback in edge_fallbacks {
        fallback.restore(patch);
        if validate_generated_patch(patch).is_ok() {
            return Ok(());
        }
    }
    validate_generated_patch(patch)
}

fn normalized_anchor(anchor: &PatchState) -> Result<PatchState, String> {
    let mut compatible_anchor = anchor.clone();
    for layer in &mut compatible_anchor.layers {
        layer.sync_active_slot_from_legacy_mirrors();
    }
    let yaml =
        serde_yaml::to_string(&compatible_anchor).map_err(|e| format!("serialize anchor: {e}"))?;
    let mut normalized: PatchState =
        serde_yaml::from_str(&yaml).map_err(|e| format!("normalize anchor: {e}"))?;

    let sanitize_effects = |effects: &EffectsConfig| {
        let mut uniforms = crate::effects::EffectUniforms::default();
        effects.apply_to_uniforms(&mut uniforms);
        EffectsConfig::from_uniforms(&uniforms)
    };
    normalized.master = sanitize_effects(&normalized.master);
    normalized.master_transform = normalized.master_transform.sanitized();
    normalized.master_motion = normalized.master_motion.map(MotionConfig::sanitized);
    for layer in &mut normalized.layers {
        layer.opacity = if layer.opacity.is_finite() {
            layer.opacity.clamp(0.0, 1.0)
        } else {
            1.0
        };
        layer.speed = if layer.speed.is_finite() {
            layer.speed.clamp(0.25, 4.0)
        } else {
            1.0
        };
        layer.fps = if layer.fps.is_finite() {
            layer.fps.clamp(1.0, 240.0)
        } else {
            30.0
        };
        layer.blend_mode = canonical_blend_mode(layer.blend_mode.as_str())
            .key()
            .to_string();
        layer.effects = sanitize_effects(&layer.effects);
        layer.transform = layer.transform.sanitized();
        layer.motion = layer.motion.map(MotionConfig::sanitized);
        layer.collapse_to_generated_single_slot();
    }
    // Generated studies deliberately avoid copying performance topology. One
    // selected source per layer remains inspectable; mattes are disabled by
    // the collapse above and atomic scenes are never randomized or emitted.
    normalized.scenes = crate::performance::Scenes::default();
    normalized.ntsc = normalized
        .ntsc
        .as_ref()
        .map(|config| crate::patch::NtscConfig::from_params(&config.to_params()));
    normalized.temporal = normalized.temporal.as_ref().map(|config| {
        let mut sanitized = crate::patch::TemporalConfig::from_params(&config.to_params());
        // Runtime conversion has no live layer IDs and therefore represents
        // both saved conductor states as Missing. Restore the persisted
        // Selected-vs-tombstone distinction, and preserve explicit presence
        // so an authored all-default block remains eligible for generation.
        sanitized.originals = config.originals.map(TemporalOriginalsConfig::sanitized);
        sanitized
    });
    normalized.modulation = normalized.modulation.as_ref().map(|config| {
        let mut matrix = crate::modulation::ModMatrix::new();
        config.apply_to_matrix(&mut matrix);
        let mut clean = crate::patch::ModConfig::from_matrix(&matrix);
        if config
            .routings
            .iter()
            .any(|routing| routing.stable_target.is_some())
        {
            // Stable targets are saved-position topology. The compatibility
            // matrix intentionally cannot resolve them without live IDs, so a
            // v5 procedural normalization retains their ordered typed intent.
            clean.routings = config
                .routings
                .iter()
                .take(crate::modulation::MAX_ROUTINGS)
                .cloned()
                .map(|mut routing| {
                    routing.depth = if routing.depth.is_finite() {
                        routing.depth.clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    routing.curve_amount = if routing.curve_amount.is_finite() {
                        routing.curve_amount.clamp(-2.0, 2.0)
                    } else {
                        0.0
                    };
                    routing.attack = if routing.attack.is_finite() {
                        routing.attack.clamp(0.0, 10.0)
                    } else {
                        0.0
                    };
                    routing.release = if routing.release.is_finite() {
                        routing.release.clamp(0.0, 10.0)
                    } else {
                        0.0
                    };
                    routing
                })
                .collect();
        }
        clean
    });
    normalized.morph = normalized
        .morph
        .take()
        .map(|snapshot| crate::morph::Morph::from_snapshot(snapshot).snapshot_at_beat(0.0));
    Ok(normalized)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn weighted_pick<'a>(choices: &'a [(&'a str, u32)], rng: &mut SplitMix64) -> &'a str {
    let total: u64 = choices.iter().map(|(_, weight)| u64::from(*weight)).sum();
    let mut draw = rng.next_u64() % total.max(1);
    for &(word, weight) in choices {
        if draw < u64::from(weight) {
            return word;
        }
        draw -= u64::from(weight);
    }
    choices.last().map(|(word, _)| *word).unwrap_or("signal")
}

fn title_for(patch: &PatchState, seed: u64) -> String {
    const ANALOG_A: &[(&str, u32)] = &[
        ("tracking", 5),
        ("phosphor", 3),
        ("raster", 4),
        ("chroma", 4),
        ("signal", 5),
    ];
    const ANALOG_B: &[(&str, u32)] = &[
        ("drift", 5),
        ("ghost", 3),
        ("decay", 4),
        ("snow", 3),
        ("smear", 2),
    ];
    const DIGITAL_A: &[(&str, u32)] = &[
        ("pixel", 4),
        ("grid", 4),
        ("threshold", 2),
        ("quantized", 2),
        ("cellular", 5),
    ];
    const DIGITAL_B: &[(&str, u32)] = &[
        ("field", 5),
        ("step", 2),
        ("lattice", 5),
        ("phase", 4),
        ("sweep", 3),
    ];
    let analog_score = patch
        .ntsc
        .as_ref()
        .map(|n| {
            u32::from(n.enabled) * 5
                + u32::from(n.tracking_noise_enabled) * 3
                + u32::from(n.snow_intensity > 0.1) * 2
        })
        .unwrap_or(0)
        + u32::from(patch.master.grain_intensity > 0.08) * 2;
    let digital_score = 1
        + u32::from(patch.master.cellular_amount > 0.15) * 5
        + u32::from(patch.master.pixelate > 2.0) * 2
        + u32::from(patch.master.posterize > 1.0) * 2
        + patch
            .layers
            .iter()
            .map(|layer| u32::from(layer.effects.cellular_amount > 0.15) * 2)
            .sum::<u32>();
    let mut rng = SplitMix64::new(seed);
    let domain_draw = rng.next_u64() % u64::from((analog_score + digital_score).max(1));
    let (first, second) = if domain_draw < u64::from(analog_score) {
        (ANALOG_A, ANALOG_B)
    } else {
        (DIGITAL_A, DIGITAL_B)
    };
    format!(
        "{} {}",
        weighted_pick(first, &mut rng),
        weighted_pick(second, &mut rng)
    )
}

fn slugify(title: &str) -> String {
    let mut slug = String::new();
    let mut dash = false;
    for ch in title.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            dash = false;
        } else if !dash && !slug.is_empty() {
            slug.push('-');
            dash = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        "piece".to_string()
    } else {
        slug
    }
}

fn logical_filename(value: &str) -> String {
    value
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty())
        .filter(|part| *part != "." && *part != "..")
        .unwrap_or("unnamed-media")
        .to_string()
}

fn manifest_file_source(
    role: &str,
    layer_index: Option<usize>,
    logical_name: String,
    identity: Option<ContentIdentity>,
) -> ManifestSource {
    let verified = identity.is_some();
    ManifestSource {
        role: role.to_string(),
        layer_index,
        logical_name,
        kind: "file".to_string(),
        byte_len: identity.as_ref().map(|value| value.byte_len),
        sha256: identity.map(|value| value.sha256),
        offline_policy: None,
        verified,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn unverified_inventory(anchor: &PatchState) -> SourceInventory {
    let mut sources = Vec::new();
    let mut warnings = Vec::new();
    for (index, layer) in anchor.layers.iter().enumerate() {
        if let Some(sender) = layer
            .source_path
            .strip_prefix(crate::layers::SPOUT_SOURCE_PREFIX)
        {
            sources.push(ManifestSource {
                role: "layer".into(),
                layer_index: Some(index),
                logical_name: sender.to_string(),
                kind: "spout".into(),
                byte_len: None,
                sha256: None,
                offline_policy: Some("deterministic_black".into()),
                verified: true,
            });
        } else {
            let logical_name = logical_filename(&layer.filename);
            warnings.push(format!(
                "layer {} source {:?} is not content-verified",
                index + 1,
                logical_name
            ));
            sources.push(manifest_file_source(
                "layer",
                Some(index),
                logical_name,
                None,
            ));
        }
    }
    if let Some(modulation) = anchor.modulation.as_ref() {
        if modulation.audio_source_kind == crate::modulation::AUDIO_SOURCE_FILE
            && !modulation.audio_clip_path.is_empty()
        {
            let logical_name =
                if crate::media_source::parse_content_reference(&modulation.audio_clip_path)
                    .ok()
                    .flatten()
                    .is_some()
                {
                    "analysis-audio".to_string()
                } else {
                    logical_filename(&modulation.audio_clip_path)
                };
            warnings.push(format!(
                "analysis-audio source {logical_name:?} is not content-verified"
            ));
            sources.push(manifest_file_source(
                "analysis_audio",
                None,
                logical_name,
                None,
            ));
        }
    }
    let identity_complete = sources
        .iter()
        .all(|source| source.kind != "file" || source.verified);
    SourceInventory {
        sources,
        limits: PreflightLimits {
            max_search_entries: DEFAULT_MAX_SEARCH_ENTRIES,
            max_fingerprint_bytes: DEFAULT_MAX_FINGERPRINT_BYTES,
        },
        warnings,
        identity_complete,
    }
}

/// Resolve and fingerprint every file source before generation commits any
/// output. Local paths and filesystem metadata remain operational state and are
/// never copied into the returned inventory.
pub fn preflight_sources(
    anchor: &PatchState,
    config: &SourcePreflightConfig,
) -> Result<SourceInventory, String> {
    let limits = FingerprintLimits {
        max_search_entries: DEFAULT_MAX_SEARCH_ENTRIES,
        max_total_bytes: config.max_fingerprint_bytes,
    };
    let mut fingerprints = FingerprintSession::new(limits).map_err(|error| error.to_string())?;
    let context = ResolveContext::new(config.anchor_dir.clone(), config.library_dir.clone());
    let mut sources = Vec::new();
    let mut warnings = Vec::new();
    let mut identity_complete = true;

    for (index, layer) in anchor.layers.iter().enumerate() {
        let resolved = crate::media_source::resolve_visual_source(
            &layer.source_path,
            &layer.filename,
            &context,
            None,
            crate::layers::is_supported_visual_file,
            &mut fingerprints,
        );
        match resolved {
            Ok(ResolvedVisualSource::Spout { sender }) => sources.push(ManifestSource {
                role: "layer".into(),
                layer_index: Some(index),
                logical_name: sender,
                kind: "spout".into(),
                byte_len: None,
                sha256: None,
                offline_policy: Some("deterministic_black".into()),
                verified: true,
            }),
            Ok(ResolvedVisualSource::File(resolved)) => {
                let logical_name = logical_filename(&layer.filename);
                let identity = fingerprints
                    .fingerprint(&resolved.path)
                    .map_err(|error| format!("fingerprint layer {}: {error}", index + 1))?;
                sources.push(manifest_file_source(
                    "layer",
                    Some(index),
                    logical_name,
                    Some(identity),
                ));
            }
            Err(error) if config.allow_unverified_sources => {
                let logical_name = logical_filename(&layer.filename);
                identity_complete = false;
                warnings.push(format!(
                    "layer {} source {:?} is unverified: {}",
                    index + 1,
                    logical_name,
                    privacy_safe_resolution_reason(&error)
                ));
                sources.push(manifest_file_source(
                    "layer",
                    Some(index),
                    logical_name,
                    None,
                ));
            }
            Err(error) => {
                return Err(format!(
                    "resolve layer {} source {:?}: {error}; pass --allow-unverified-sources to preserve only its logical name",
                    index + 1,
                    logical_filename(&layer.filename)
                ));
            }
        }
    }

    if let Some(modulation) = anchor.modulation.as_ref() {
        if modulation.audio_source_kind == crate::modulation::AUDIO_SOURCE_FILE
            && !modulation.audio_clip_path.is_empty()
        {
            let embedded =
                crate::media_source::parse_content_reference(&modulation.audio_clip_path)
                    .map_err(|error| error.to_string())?;
            let logical_name = if embedded.is_some() {
                "analysis-audio".to_string()
            } else {
                logical_filename(&modulation.audio_clip_path)
            };
            let resolved = crate::media_source::resolve_file_source(
                &modulation.audio_clip_path,
                &logical_name,
                &context,
                None,
                |path: &Path| crate::audio::is_supported_audio_file(path),
                &mut fingerprints,
            );
            match resolved {
                Ok(resolved) => {
                    let identity = fingerprints.fingerprint(&resolved.path).map_err(|error| {
                        format!("fingerprint analysis-audio source {logical_name:?}: {error}")
                    })?;
                    sources.push(manifest_file_source(
                        "analysis_audio",
                        None,
                        logical_name,
                        Some(identity),
                    ));
                }
                Err(error) if config.allow_unverified_sources => {
                    identity_complete = false;
                    warnings.push(format!(
                        "analysis-audio source {logical_name:?} is unverified: {}",
                        privacy_safe_resolution_reason(&error)
                    ));
                    sources.push(manifest_file_source(
                        "analysis_audio",
                        None,
                        logical_name,
                        None,
                    ));
                }
                Err(error) => {
                    return Err(format!(
                        "resolve analysis-audio source {logical_name:?}: {error}; pass --allow-unverified-sources to preserve only its logical name"
                    ));
                }
            }
        }
    }

    Ok(SourceInventory {
        sources,
        limits: PreflightLimits {
            max_search_entries: limits.max_search_entries,
            max_fingerprint_bytes: limits.max_total_bytes,
        },
        warnings,
        identity_complete,
    })
}

fn privacy_safe_resolution_reason(error: &crate::media_source::SourceResolveError) -> &'static str {
    use crate::media_source::SourceResolveError;
    match error {
        SourceResolveError::InvalidContentReference(_) => "invalid content reference",
        SourceResolveError::InvalidLimit(_) => "invalid resource limit",
        SourceResolveError::Missing(_) => "not found",
        SourceResolveError::ContentMismatch(_) => "content mismatch",
        SourceResolveError::FingerprintBudget(_) => "fingerprint budget exceeded",
        SourceResolveError::SearchBudget(_) => "search budget exceeded",
        SourceResolveError::ChangedDuringFingerprint(_) => "changed during fingerprinting",
        SourceResolveError::Cancelled => "cancelled",
        SourceResolveError::Io(_) => "filesystem error",
    }
}

fn source_for_layer(inventory: &SourceInventory, layer_index: usize) -> Option<&ManifestSource> {
    inventory
        .sources
        .iter()
        .find(|source| source.role == "layer" && source.layer_index == Some(layer_index))
}

fn analysis_audio_source(inventory: &SourceInventory) -> Option<&ManifestSource> {
    inventory
        .sources
        .iter()
        .find(|source| source.role == "analysis_audio")
}

fn apply_inventory_references(patch: &mut PatchState, inventory: &SourceInventory) {
    for (index, layer) in patch.layers.iter_mut().enumerate() {
        let Some(source) = source_for_layer(inventory, index) else {
            layer.filename = logical_filename(&layer.filename);
            layer.source_path.clear();
            layer.collapse_to_generated_single_slot();
            continue;
        };
        layer.filename = source.logical_name.clone();
        match source.kind.as_str() {
            "spout" => {
                layer.source_path = format!(
                    "{}{}",
                    crate::layers::SPOUT_SOURCE_PREFIX,
                    source.logical_name
                );
            }
            "file" => {
                layer.source_path = source
                    .sha256
                    .as_ref()
                    .zip(source.byte_len)
                    .and_then(|(sha256, byte_len)| {
                        ContentIdentity::new(sha256.clone(), byte_len).ok()
                    })
                    .map(|identity| identity.source_reference())
                    .unwrap_or_default();
            }
            _ => layer.source_path.clear(),
        }
        layer.collapse_to_generated_single_slot();
    }
    if let Some(modulation) = patch.modulation.as_mut() {
        if modulation.audio_source_kind == crate::modulation::AUDIO_SOURCE_FILE
            && !modulation.audio_clip_path.is_empty()
        {
            if let Some(source) = analysis_audio_source(inventory) {
                modulation.audio_clip_path = source
                    .sha256
                    .as_ref()
                    .zip(source.byte_len)
                    .and_then(|(sha256, byte_len)| {
                        ContentIdentity::new(sha256.clone(), byte_len).ok()
                    })
                    .map(|identity| identity.source_reference())
                    .unwrap_or_else(|| source.logical_name.clone());
            } else {
                modulation.audio_clip_path = logical_filename(&modulation.audio_clip_path);
            }
        }
    }
}

fn canonical_patch_bytes(patch: &PatchState) -> Result<Vec<u8>, String> {
    let json =
        serde_json::to_vec(patch).map_err(|error| format!("serialize canonical patch: {error}"))?;
    let mut bytes = b"collide-o-scope/canonical-patch/v1\0".to_vec();
    bytes.extend_from_slice(&json);
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_patch_sha256(patch: &PatchState) -> Result<String, String> {
    canonical_patch_bytes(patch).map(|bytes| sha256_hex(&bytes))
}

fn inventory_summary(inventory: &SourceInventory) -> (usize, u64) {
    let mut unique = std::collections::BTreeSet::new();
    for source in &inventory.sources {
        if let (Some(sha256), Some(byte_len)) = (&source.sha256, source.byte_len) {
            unique.insert((sha256.clone(), byte_len));
        }
    }
    let bytes = unique
        .iter()
        .fold(0u64, |total, (_, byte_len)| total.saturating_add(*byte_len));
    (unique.len(), bytes)
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn generate(
    anchor: &PatchState,
    config: &GenerationConfig,
) -> Result<Vec<GeneratedPiece>, String> {
    let inventory = unverified_inventory(anchor);
    generate_with_inventory(anchor, config, &inventory)
}

pub fn generate_with_inventory(
    anchor: &PatchState,
    config: &GenerationConfig,
    inventory: &SourceInventory,
) -> Result<Vec<GeneratedPiece>, String> {
    if !(1..=MAX_GENERATED_COUNT).contains(&config.count) {
        return Err(format!("count must be between 1 and {MAX_GENERATED_COUNT}"));
    }
    let temperature = finite_temperature(config.temperature)?;
    if anchor
        .morph
        .as_ref()
        .is_some_and(|morph| (morph.a.is_some() && morph.b.is_some()) || morph.glide.is_some())
    {
        return Err(
            "generation from an active A/B morph or glide is ambiguous; settle or clear it first"
                .into(),
        );
    }
    let spout_sources: Vec<String> = anchor
        .layers
        .iter()
        .filter_map(|layer| {
            layer
                .source_path
                .strip_prefix("spout://")
                .map(ToOwned::to_owned)
        })
        .collect();
    if !spout_sources.is_empty() && !config.allow_black_sources {
        return Err("live Spout sources render black offline; pass --allow-black-sources to accept that policy".into());
    }

    let mut normalized = normalized_anchor(anchor)?;
    apply_inventory_references(&mut normalized, inventory);
    let anchor_bytes = canonical_patch_bytes(&normalized)?;
    let anchor_sha256 = sha256_hex(&anchor_bytes);
    let anchor_hash = format!("{:016x}", fnv1a64(&anchor_bytes));
    let logical_sources: Vec<String> = normalized
        .layers
        .iter()
        // Manifests are shareable metadata: retain logical names, never local
        // absolute paths. The patch itself still owns the source resolver.
        .map(|layer| {
            layer
                .source_path
                .strip_prefix("spout://")
                .map(|name| format!("spout://{name}"))
                .unwrap_or_else(|| logical_filename(&layer.filename))
        })
        .collect();

    let mut pieces = Vec::with_capacity(config.count);
    let mut walk = normalized.clone();
    for index in 0..config.count {
        let mut patch = walk.clone();
        let mut master_rng = SplitMix64::new(domain_seed(config.seed, index, 0x4d41_5354_4552));
        mutate_effects(
            &normalized.master,
            &mut patch.master,
            temperature,
            &mut master_rng,
        );
        let mut master_spatial_rng =
            SplitMix64::new(domain_seed(config.seed, index, 0x4d53_5041_544c));
        mutate_transform(
            &normalized.master_transform,
            &mut patch.master_transform,
            temperature,
            &mut master_spatial_rng,
        );
        if let (Some(anchor_motion), Some(motion)) = (
            normalized.master_motion.as_ref(),
            patch.master_motion.as_mut(),
        ) {
            mutate_motion_config(
                anchor_motion,
                motion,
                temperature,
                config.seed,
                index,
                0x4d41_5354_4552_4d34,
                false,
            );
        }
        for (layer_index, (anchor_layer, layer)) in normalized
            .layers
            .iter()
            .zip(patch.layers.iter_mut())
            .enumerate()
        {
            let mut rng = SplitMix64::new(domain_seed(
                config.seed,
                index,
                0x4c41_5945_5200 ^ layer_index as u64,
            ));
            mutate_effects(
                &anchor_layer.effects,
                &mut layer.effects,
                temperature,
                &mut rng,
            );
            let mut spatial_rng = SplitMix64::new(domain_seed(
                config.seed,
                index,
                0x4c53_5041_5400 ^ layer_index as u64,
            ));
            mutate_transform(
                &anchor_layer.transform,
                &mut layer.transform,
                temperature,
                &mut spatial_rng,
            );
            if let (Some(anchor_motion), Some(motion)) =
                (anchor_layer.motion.as_ref(), layer.motion.as_mut())
            {
                mutate_motion_config(
                    anchor_motion,
                    motion,
                    temperature,
                    config.seed,
                    index,
                    0x4c41_5945_525f_4d34
                        ^ (layer_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    true,
                );
            }
            layer.opacity = mutate_linear(
                anchor_layer.opacity,
                layer.opacity,
                0.0,
                1.0,
                temperature * 0.12,
                &mut rng,
            );
            layer.speed = mutate_log(
                anchor_layer.speed,
                layer.speed,
                0.25,
                4.0,
                temperature * 0.18,
                &mut rng,
            );
            layer.fps = mutate_linear(
                anchor_layer.fps,
                layer.fps,
                1.0,
                240.0,
                temperature * 3.0,
                &mut rng,
            )
            .round();
            let blend = mutate_blend_mode(
                anchor_layer.blend_mode.as_str(),
                layer.blend_mode.as_str(),
                temperature * 0.04,
                &mut rng,
            );
            layer.blend_mode = blend.key().to_string();
            layer.collapse_to_generated_single_slot();
        }
        if let (Some(anchor_temporal), Some(temporal)) =
            (normalized.temporal.as_ref(), patch.temporal.as_mut())
        {
            let mut rng = SplitMix64::new(domain_seed(config.seed, index, 0x5445_4d50_4f52));
            temporal.feedback = mutate_linear(
                anchor_temporal.feedback,
                temporal.feedback,
                0.0,
                0.95,
                temperature * 0.08,
                &mut rng,
            );
            temporal.fb_zoom = mutate_linear(
                anchor_temporal.fb_zoom,
                temporal.fb_zoom,
                0.9,
                1.1,
                temperature * 0.025,
                &mut rng,
            );
            temporal.fb_rotate = mutate_linear(
                anchor_temporal.fb_rotate,
                temporal.fb_rotate,
                -5.0,
                5.0,
                temperature * 0.8,
                &mut rng,
            );
            temporal.slitscan = mutate_linear(
                anchor_temporal.slitscan,
                temporal.slitscan,
                0.0,
                1.0,
                temperature * 0.12,
                &mut rng,
            );
            let angle = temporal.slit_angle.unwrap_or(temporal.slit_axis * 90.0);
            let anchor_angle = anchor_temporal
                .slit_angle
                .unwrap_or(anchor_temporal.slit_axis * 90.0);
            temporal.slit_angle = Some(mutate_circular(
                anchor_angle,
                angle,
                -180.0,
                180.0,
                temperature * 25.0,
                &mut rng,
            ));
            temporal.key_threshold = mutate_linear(
                anchor_temporal.key_threshold,
                temporal.key_threshold,
                0.0,
                1.0,
                temperature * 0.08,
                &mut rng,
            );
            temporal.key_softness = mutate_linear(
                anchor_temporal.key_softness,
                temporal.key_softness,
                0.0,
                0.5,
                temperature * 0.04,
                &mut rng,
            );
            temporal.key_history = mutate_linear(
                anchor_temporal.key_history,
                temporal.key_history,
                1.0,
                23.0,
                temperature * 2.0,
                &mut rng,
            )
            .round();
        }
        if let (Some(anchor_originals), Some(originals)) = (
            normalized
                .temporal
                .as_ref()
                .and_then(|temporal| temporal.originals.as_ref()),
            patch
                .temporal
                .as_mut()
                .and_then(|temporal| temporal.originals.as_mut()),
        ) {
            mutate_temporal_originals(anchor_originals, originals, temperature, config.seed, index);
        }
        if let (Some(anchor_modulation), Some(modulation)) =
            (normalized.modulation.as_ref(), patch.modulation.as_mut())
        {
            let mut rng = SplitMix64::new(domain_seed(config.seed, index, 0x4d4f_4455_4c41));
            for (anchor_lfo, lfo) in anchor_modulation.lfos.iter().zip(&mut modulation.lfos) {
                lfo.phase = mutate_circular(
                    anchor_lfo.phase,
                    lfo.phase,
                    0.0,
                    1.0,
                    temperature * 0.15,
                    &mut rng,
                );
            }
            for (anchor_route, route) in anchor_modulation
                .routings
                .iter()
                .zip(&mut modulation.routings)
            {
                route.depth = mutate_linear(
                    anchor_route.depth,
                    route.depth,
                    -1.0,
                    1.0,
                    temperature * 0.12,
                    &mut rng,
                );
            }
        }
        if let (Some(anchor_ntsc), Some(ntsc)) = (normalized.ntsc.as_ref(), patch.ntsc.as_mut()) {
            let mut rng = SplitMix64::new(domain_seed(config.seed, index, 0x4e54_5343_0000));
            ntsc.chroma_loss = mutate_linear(
                anchor_ntsc.chroma_loss,
                ntsc.chroma_loss,
                0.0,
                0.01,
                temperature * 0.0015,
                &mut rng,
            );
            ntsc.snow_intensity = mutate_linear(
                anchor_ntsc.snow_intensity,
                ntsc.snow_intensity,
                0.0,
                1.0,
                temperature * 0.12,
                &mut rng,
            );
            ntsc.tracking_noise_snow = mutate_linear(
                anchor_ntsc.tracking_noise_snow,
                ntsc.tracking_noise_snow,
                0.0,
                1.0,
                temperature * 0.1,
                &mut rng,
            );
            ntsc.composite_noise_intensity = mutate_linear(
                anchor_ntsc.composite_noise_intensity,
                ntsc.composite_noise_intensity,
                0.0,
                0.5,
                temperature * 0.06,
                &mut rng,
            );
            ntsc.luma_smear = mutate_linear(
                anchor_ntsc.luma_smear,
                ntsc.luma_smear,
                0.0,
                1.0,
                temperature * 0.1,
                &mut rng,
            );
            ntsc.enabled = mutate_bool(
                anchor_ntsc.enabled,
                ntsc.enabled,
                temperature * 0.04,
                &mut rng,
            );
            if ntsc.enabled {
                ntsc.tracking_noise_enabled = mutate_bool(
                    anchor_ntsc.tracking_noise_enabled,
                    ntsc.tracking_noise_enabled,
                    temperature * 0.05,
                    &mut rng,
                );
            } else {
                ntsc.tracking_noise_enabled = false;
            }
        }

        // M2 creative values evolve in owner/NodeId-isolated domains. This
        // deliberately consumes none of the v4 streams above, and every
        // topology- or route-owning field remains frozen. A dormant image
        // edge may only wake if the complete saved graph still validates.
        let mut creative_edge_fallbacks = Vec::new();
        mutate_saved_creative_values(
            &normalized,
            &mut patch,
            temperature,
            config.seed,
            index,
            &mut creative_edge_fallbacks,
        );
        retain_valid_creative_edge_values(&mut patch, creative_edge_fallbacks)?;

        let title_seed = domain_seed(config.seed, index, 0x5449_544c_4500);
        let title = title_for(&patch, title_seed);
        let slug = slugify(&title);
        let mut warnings = inventory.warnings.clone();
        warnings.extend(
            spout_sources
                .iter()
                .map(|name| format!("Spout source {name:?} is deterministic black offline")),
        );
        let piece_sha256 = canonical_patch_sha256(&patch)?;
        let manifest = Manifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            generator_version: GENERATOR_VERSION.to_string(),
            seed: config.seed,
            index,
            temperature,
            title,
            slug,
            anchor_fnv1a64: anchor_hash.clone(),
            canonical_identity_algorithm: CANONICAL_IDENTITY_ALGORITHM.to_string(),
            anchor_sha256: anchor_sha256.clone(),
            piece_sha256: piece_sha256.clone(),
            identity_complete: inventory.identity_complete,
            lineage: Some(anchor_sha256.clone()),
            logical_sources: logical_sources.clone(),
            sources: inventory.sources.clone(),
            warnings: warnings.clone(),
        };
        let (source_files, source_bytes) = inventory_summary(inventory);
        let status = if inventory.identity_complete && warnings.is_empty() {
            "ready"
        } else {
            "ready_with_warnings"
        };
        let preflight = PreflightReceipt {
            schema_version: PREFLIGHT_SCHEMA_VERSION,
            canonical_identity_algorithm: CANONICAL_IDENTITY_ALGORITHM.to_string(),
            anchor_sha256: anchor_sha256.clone(),
            piece_sha256,
            status: status.to_string(),
            claim_scope: "canonical_configuration_and_source_bytes".to_string(),
            pixel_identity_claimed: false,
            source_files,
            source_bytes,
            limits: inventory.limits.clone(),
            sources: inventory.sources.clone(),
            warnings,
        };
        walk = patch.clone();
        pieces.push(GeneratedPiece {
            patch,
            manifest,
            preflight,
        });
    }
    Ok(pieces)
}

fn sync_parent(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if let Ok(directory) = fs::File::open(parent) {
            directory.sync_all()?;
        }
    }
    Ok(())
}

#[cfg(windows)]
fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::MoveFileExW;

    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    // Passing no MOVEFILE_REPLACE_EXISTING flag makes the commit atomic and
    // fail-closed when another process wins the destination name.
    let moved = unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), 0) };
    if moved == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "source path contains NUL"))?;
    let destination = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "destination path contains NUL")
    })?;
    let moved = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if moved == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(target_vendor = "apple")]
fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "source path contains NUL"))?;
    let destination = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "destination path contains NUL")
    })?;
    let moved =
        unsafe { libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) };
    if moved == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
fn rename_noreplace(_source: &Path, _destination: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace rename is not implemented on this Unix target",
    ))
}

#[cfg(not(any(windows, unix)))]
fn rename_noreplace(_source: &Path, _destination: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace rename is not implemented on this platform",
    ))
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("write {}: {e}", path.display()))
}

pub fn write_patch_only(
    pieces: &[GeneratedPiece],
    output_dir: &Path,
) -> Result<Vec<PathBuf>, String> {
    fs::create_dir_all(output_dir)
        .map_err(|e| format!("create output directory {}: {e}", output_dir.display()))?;

    struct PreparedPiece {
        final_dir: PathBuf,
        temp_dir: PathBuf,
        yaml: String,
        manifest: Vec<u8>,
        preflight: Vec<u8>,
    }

    // Serialize and preflight the complete invocation before committing the
    // first piece. The on-disk transaction boundary remains one piece, but
    // deterministic name conflicts and serialization errors cannot yield a
    // misleading half-batch.
    let mut prepared = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let suffix = format!(
            "{:08x}",
            piece.manifest.seed as u32 ^ piece.manifest.index as u32
        );
        let name = format!(
            "{:04}-{}-{suffix}",
            piece.manifest.index + 1,
            piece.manifest.slug
        );
        let final_dir = output_dir.join(&name);
        if final_dir.exists()
            || prepared
                .iter()
                .any(|plan: &PreparedPiece| plan.final_dir == final_dir)
        {
            return Err(format!("refusing to overwrite {}", final_dir.display()));
        }
        let temp_dir = output_dir.join(format!(".{name}.tmp-{}", std::process::id()));
        if temp_dir.exists()
            || prepared
                .iter()
                .any(|plan: &PreparedPiece| plan.temp_dir == temp_dir)
        {
            return Err(format!(
                "temporary output already exists: {}",
                temp_dir.display()
            ));
        }
        prepared.push(PreparedPiece {
            final_dir,
            temp_dir,
            yaml: serde_yaml::to_string(&piece.patch)
                .map_err(|e| format!("serialize generated patch: {e}"))?,
            manifest: serde_json::to_vec_pretty(&piece.manifest)
                .map_err(|e| format!("serialize generation manifest: {e}"))?,
            preflight: serde_json::to_vec_pretty(&piece.preflight)
                .map_err(|e| format!("serialize generation preflight: {e}"))?,
        });
    }

    let mut created = Vec::with_capacity(pieces.len());
    for plan in prepared {
        let partial_error = |error: String, committed: &[PathBuf]| {
            if committed.is_empty() {
                error
            } else {
                let paths = committed
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "{error}; {} earlier piece(s) remain committed: {paths}",
                    committed.len()
                )
            }
        };
        if let Err(error) = fs::create_dir(&plan.temp_dir) {
            return Err(partial_error(
                format!(
                    "create temporary directory {}: {error}",
                    plan.temp_dir.display()
                ),
                &created,
            ));
        }
        let staged = write_new_file(&plan.temp_dir.join("patch.yaml"), plan.yaml.as_bytes())
            .and_then(|_| write_new_file(&plan.temp_dir.join("manifest.json"), &plan.manifest))
            .and_then(|_| write_new_file(&plan.temp_dir.join("preflight.json"), &plan.preflight))
            .and_then(|_| {
                sync_parent(&plan.temp_dir.join("patch.yaml"))
                    .map_err(|error| format!("sync staged output: {error}"))
            });
        if let Err(error) = staged {
            let _ = fs::remove_dir_all(&plan.temp_dir);
            return Err(partial_error(error, &created));
        }
        if let Err(error) = rename_noreplace(&plan.temp_dir, &plan.final_dir) {
            let _ = fs::remove_dir_all(&plan.temp_dir);
            return Err(partial_error(
                format!("commit {}: {error}", plan.final_dir.display()),
                &created,
            ));
        }
        created.push(plan.final_dir.clone());
        if let Err(error) = sync_parent(&plan.final_dir) {
            return Err(format!(
                "{} committed, but synchronizing its parent directory failed: {error}; committed pieces: {}",
                plan.final_dir.display(),
                created
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    Ok(created)
}

fn unique_capture_path(dir: &Path) -> Result<(PathBuf, PathBuf), String> {
    fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("system clock before epoch: {e}"))?
        .as_millis();
    for counter in 0..10_000u32 {
        let stem = format!("capture-{millis}-{counter:04}");
        let final_path = dir.join(format!("{stem}.yaml"));
        let temp_path = dir.join(format!(".{stem}.tmp-{}", std::process::id()));
        if !final_path.exists() && !temp_path.exists() {
            return Ok((temp_path, final_path));
        }
    }
    Err("could not allocate a unique capture filename".to_string())
}

pub fn quick_capture(patch: &PatchState, dir: &Path) -> Result<PathBuf, String> {
    let yaml = serde_yaml::to_string(patch).map_err(|e| format!("serialize capture: {e}"))?;
    let (temp_path, final_path) = unique_capture_path(dir)?;
    if let Err(error) = write_new_file(&temp_path, yaml.as_bytes()) {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }
    if final_path.exists() {
        let _ = fs::remove_file(&temp_path);
        return Err(format!("refusing to overwrite {}", final_path.display()));
    }
    rename_noreplace(&temp_path, &final_path).map_err(|e| {
        let _ = fs::remove_file(&temp_path);
        format!("commit capture {}: {e}", final_path.display())
    })?;
    sync_parent(&final_path).map_err(|e| {
        format!(
            "capture committed at {}, but synchronizing its parent directory failed: {e}",
            final_path.display()
        )
    })?;
    Ok(final_path)
}

struct CaptureRequest {
    sequence: u64,
    patch: PatchState,
    directory: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureSubmit {
    Queued,
    Busy,
}

pub struct PatchCollector {
    sender: Option<mpsc::SyncSender<CaptureRequest>>,
    status: Arc<Mutex<String>>,
    phase: Arc<std::sync::atomic::AtomicU8>,
    latest_sequence: Arc<std::sync::atomic::AtomicU64>,
    worker: Option<JoinHandle<()>>,
}

const CAPTURE_IDLE: u8 = 0;
const CAPTURE_SAVING: u8 = 1;
const CAPTURE_SAVED: u8 = 2;
const CAPTURE_ERROR: u8 = 3;

fn publish_capture_result(
    status: &Mutex<String>,
    phase: &std::sync::atomic::AtomicU8,
    latest_sequence: &std::sync::atomic::AtomicU64,
    request_sequence: u64,
    result: Result<PathBuf, String>,
) {
    if let Ok(mut state) = status.lock() {
        // Compare while holding the same lock submission uses to publish
        // Saving. An older completion can therefore never pass the sequence
        // check and then overwrite a newer request after waiting on this mutex.
        if latest_sequence.load(std::sync::atomic::Ordering::Acquire) == request_sequence {
            let (message, terminal_phase) = match result {
                Ok(path) => (
                    format!(
                        "Saved {}",
                        path.file_name()
                            .map(|name| name.to_string_lossy())
                            .unwrap_or_default()
                    ),
                    CAPTURE_SAVED,
                ),
                Err(error) => (format!("Error: {error}"), CAPTURE_ERROR),
            };
            *state = message;
            phase.store(terminal_phase, std::sync::atomic::Ordering::Release);
        }
    }
}

impl PatchCollector {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::sync_channel::<CaptureRequest>(1);
        let status = Arc::new(Mutex::new(String::new()));
        let phase = Arc::new(std::sync::atomic::AtomicU8::new(CAPTURE_IDLE));
        let latest_sequence = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let worker_status = status.clone();
        let worker_phase = phase.clone();
        let worker_latest_sequence = latest_sequence.clone();
        let worker = thread::Builder::new()
            .name("patch-collector".to_string())
            .spawn(move || {
                while let Ok(request) = receiver.recv() {
                    let result = quick_capture(&request.patch, &request.directory);
                    publish_capture_result(
                        &worker_status,
                        &worker_phase,
                        &worker_latest_sequence,
                        request.sequence,
                        result,
                    );
                }
            })
            .expect("patch collector thread should start");
        Self {
            sender: Some(sender),
            status,
            phase,
            latest_sequence,
            worker: Some(worker),
        }
    }

    pub fn try_submit(&self, patch: PatchState, directory: PathBuf) -> CaptureSubmit {
        let Some(sender) = self.sender.as_ref() else {
            return CaptureSubmit::Busy;
        };
        // Serialize phase/sequence publication against terminal publication.
        // `status()` remains lock-free for Saving and never waits, while a
        // contended submit simply reports Busy to the caller.
        let Ok(mut status) = self.status.try_lock() else {
            return CaptureSubmit::Busy;
        };
        let previous_phase = self
            .phase
            .swap(CAPTURE_SAVING, std::sync::atomic::Ordering::AcqRel);
        let sequence = self
            .latest_sequence
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            .wrapping_add(1);
        match sender.try_send(CaptureRequest {
            sequence,
            patch,
            directory,
        }) {
            Ok(()) => CaptureSubmit::Queued,
            Err(mpsc::TrySendError::Full(_)) => {
                self.latest_sequence
                    .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                self.phase
                    .store(previous_phase, std::sync::atomic::Ordering::Release);
                CaptureSubmit::Busy
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                *status = "Error: patch collector stopped".to_string();
                self.phase
                    .store(CAPTURE_ERROR, std::sync::atomic::Ordering::Release);
                CaptureSubmit::Busy
            }
        }
    }

    pub fn status(&self) -> String {
        match self.phase.load(std::sync::atomic::Ordering::Acquire) {
            CAPTURE_IDLE => String::new(),
            CAPTURE_SAVING => "Saving…".to_string(),
            CAPTURE_SAVED => self
                .status
                .try_lock()
                .map(|value| value.clone())
                .unwrap_or_else(|_| "Saving…".to_string()),
            CAPTURE_ERROR => self
                .status
                .try_lock()
                .map(|value| value.clone())
                .unwrap_or_else(|_| "Error: capture status unavailable".to_string()),
            _ => "Error: invalid capture state".to_string(),
        }
    }
}

impl Default for PatchCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PatchCollector {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            // Closing the sender lets the worker drain every accepted request.
            // Give ordinary local storage a bounded grace interval, but never
            // make application shutdown hostage to a stalled filesystem.
            let deadline = Instant::now() + Duration::from_millis(500);
            while !worker.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            if worker.is_finished() {
                let _ = worker.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::{EffectsConfig, LayerConfig};

    fn anchor() -> PatchState {
        let clip_slots = crate::performance::ClipSlots::singleton(
            crate::performance::ClipSlotConfig::from_legacy(
                "clip.mp4".to_string(),
                String::new(),
                1.0,
                30.0,
            ),
        );
        PatchState {
            master: EffectsConfig::default(),
            master_transform: SpatialTransform::default(),
            master_motion: None,
            layers: vec![LayerConfig {
                filename: "clip.mp4".to_string(),
                source_path: String::new(),
                opacity: 1.0,
                blend_mode: "normal".to_string(),
                speed: 1.0,
                fps: 30.0,
                paused: false,
                visible: true,
                bypass_master_fx: false,
                reroll_on_loop: false,
                effects: EffectsConfig::default(),
                transform: SpatialTransform::default(),
                motion: None,
                rack: None,
                clip_slots,
                active_clip_slot: Some(crate::performance::ClipSlotId::LEGACY),
                matte: crate::image_routing::LayerMatteConfig::default(),
            }],
            master_rack: None,
            composition: None,
            visual_schema_version: 0,
            master_paused: false,
            media_frozen: false,
            ntsc: None,
            modulation: None,
            temporal: None,
            morph: None,
            scenes: crate::performance::Scenes::default(),
        }
    }

    fn advanced_anchor() -> PatchState {
        use crate::composition::{
            BusAssignment, CompositionTree, Group, GroupMembers, GroupName, RootItem,
        };
        use crate::performance::SavedLayerPosition;
        use crate::visual_rack::{
            EdgeTiming, GrainParams, GroupId, ImageMatte, MaskParams, MatteChannel, NodeId,
            SavedImageSource, SavedImageTap, ShiftParams, VisualNode, VisualNodeKind, VisualRack,
        };

        let mut patch = anchor();
        patch.master_rack = Some(
            VisualRack::try_from_parts(
                vec![VisualNode::authored(
                    NodeId::new(3).unwrap(),
                    VisualNodeKind::Shift(ShiftParams {
                        amount: 0.4,
                        seed: 91,
                        ..ShiftParams::default()
                    }),
                )],
                Some(4),
            )
            .unwrap(),
        );
        patch.layers[0].rack = Some(
            VisualRack::try_from_parts(
                vec![VisualNode::authored(
                    NodeId::new(11).unwrap(),
                    VisualNodeKind::Grain(GrainParams {
                        intensity: 0.08,
                        size: 1.7,
                        seed: 37,
                        ..GrainParams::default()
                    }),
                )],
                Some(12),
            )
            .unwrap(),
        );

        let position = SavedLayerPosition::new(0).unwrap();
        let group_id = GroupId::new(7).unwrap();
        let group_rack = VisualRack::try_from_parts(
            vec![VisualNode::authored(
                NodeId::new(3).unwrap(),
                VisualNodeKind::Mask(MaskParams::Image(ImageMatte {
                    tap: SavedImageTap {
                        source: SavedImageSource::SelectedLayer {
                            layer_position: position,
                            stage: crate::image_routing::LayerImageStage::PreLocalEffects,
                        },
                        timing: EdgeTiming::CurrentFrame,
                    },
                    channel: MatteChannel::Red,
                    invert: true,
                    amount: 0.6,
                    threshold: 0.45,
                    softness: 0.12,
                })),
            )],
            Some(4),
        )
        .unwrap();
        let group = Group {
            id: group_id,
            name: GroupName::new("source study").unwrap(),
            members: GroupMembers::try_from_vec(vec![position]).unwrap(),
            opacity: 0.72,
            transform: SpatialTransform {
                rotation_deg: 17.0,
                ..SpatialTransform::default()
            },
            rack: group_rack,
            matte: Some(ImageMatte {
                tap: SavedImageTap {
                    source: SavedImageSource::CleanProgram,
                    timing: EdgeTiming::PreviousFrame,
                },
                channel: MatteChannel::Luma,
                invert: false,
                amount: 0.8,
                threshold: 0.3,
                softness: 0.2,
            }),
            solo: false,
            bypass: true,
            bus: BusAssignment::B,
        };
        patch.composition = Some(
            CompositionTree::try_from_parts(
                vec![group],
                vec![RootItem::Group { group_id }],
                Some(8),
                0.27,
            )
            .unwrap(),
        );
        patch.visual_schema_version = 1;
        patch
    }

    #[test]
    fn temporal_originals_generation_is_domain_isolated_and_preserves_authored_laws() {
        let mut authored = TemporalOriginalsConfig::default();
        authored.loom.topology = crate::patch::TemporalTopologyConfig::Folded;
        authored.loom.interpolation = crate::patch::TemporalInterpolationConfig::Linear;
        authored.atlas.seed = 0xdead_beef;
        authored.garden.gate = crate::patch::RefreshGardenGateConfig::Matte;
        authored.garden.matte_route = crate::patch::RefreshGardenMatteRouteConfig::SelectedLayer {
            saved_position: crate::performance::SavedLayerPosition::new(1).unwrap(),
            stage: crate::image_routing::LayerImageStage::PostLocalEffects,
        };
        authored.garden.motion_route =
            crate::patch::RefreshGardenMotionRouteConfig::MissingSelectedLayer {
                saved_position: crate::performance::SavedLayerPosition::new(3).unwrap(),
            };
        authored.score.enabled = true;
        authored.score.seed = 0x1234_5678;
        authored.score.trigger = crate::patch::CollisionScoreTriggerConfig::Manual;
        authored.score.loop_driver = crate::patch::CollisionScoreLoopDriverConfig::SelectedLayer {
            saved_position: crate::performance::SavedLayerPosition::new(0).unwrap(),
        };
        authored.reset.loop_boundary = crate::patch::TemporalEventResetModeConfig::Memory;

        let mut left = authored;
        let mut right = authored;
        mutate_temporal_originals(&authored, &mut left, 2.0, 77, 3);
        mutate_temporal_originals(&authored, &mut right, 2.0, 77, 3);
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

        let mut patch_anchor = anchor();
        patch_anchor.temporal = Some(crate::patch::TemporalConfig {
            originals: Some(authored),
            ..crate::patch::TemporalConfig::default()
        });
        let normalized = normalized_anchor(&patch_anchor).unwrap();
        assert!(matches!(
            normalized
                .temporal
                .unwrap()
                .originals
                .unwrap()
                .score
                .loop_driver,
            crate::patch::CollisionScoreLoopDriverConfig::SelectedLayer { saved_position }
                if saved_position.get() == 0
        ));
    }

    #[test]
    fn motion_generation_is_isolated_bounded_and_preserves_authored_laws() {
        use crate::patch::{
            CurvedShutterConfig, CurvedShutterQualityConfig, FaradayConfig, MotionCarrierConfig,
            MotionDonorConfig, MotionFieldSourceConfig, MotionLatticeQualityConfig,
        };

        let authored = MotionConfig {
            field_source: MotionFieldSourceConfig::CodecVectors,
            lattice_quality: MotionLatticeQualityConfig::High,
            transplant: FaradayConfig {
                donor: MotionDonorConfig::Selected {
                    saved_position: crate::performance::SavedLayerPosition::new(0).unwrap(),
                },
                carrier: MotionCarrierConfig::FirstSourceFrame,
                ..FaradayConfig::default()
            },
            shutter: CurvedShutterConfig {
                quality: CurvedShutterQualityConfig::High,
                ..CurvedShutterConfig::default()
            },
            ..MotionConfig::default()
        };

        let mut left = authored;
        let mut right = authored;
        mutate_motion_config(
            &authored,
            &mut left,
            2.0,
            77,
            3,
            0x4c41_5945_525f_4d34,
            true,
        );
        mutate_motion_config(
            &authored,
            &mut right,
            2.0,
            77,
            3,
            0x4c41_5945_525f_4d34,
            true,
        );
        assert_eq!(left, right);
        assert_ne!(left, authored);
        assert_eq!(left.algorithm_version, authored.algorithm_version);
        assert_eq!(left.field_source, authored.field_source);
        assert_eq!(left.lattice_quality, authored.lattice_quality);
        assert_eq!(left.transplant.donor, authored.transplant.donor);
        assert_eq!(left.transplant.carrier, authored.transplant.carrier);
        assert_eq!(left.shutter.quality, authored.shutter.quality);
        assert_eq!(left, left.sanitized());

        let mut master = authored;
        mutate_motion_config(
            &authored,
            &mut master,
            2.0,
            77,
            3,
            0x4d41_5354_4552_4d34,
            false,
        );
        assert_eq!(master.transplant, authored.transplant);
        assert_ne!(master.shutter, authored.shutter);

        // v6 is the M3-compatible projection: introducing authored M4 blocks
        // cannot consume or reorder any pre-M4 generator stream.
        let mut m3_anchor = anchor();
        m3_anchor.temporal = Some(crate::patch::TemporalConfig {
            originals: Some(TemporalOriginalsConfig::default()),
            ..crate::patch::TemporalConfig::default()
        });
        let mut m4_anchor = m3_anchor.clone();
        m4_anchor.master_motion = Some(authored);
        m4_anchor.layers[0].motion = Some(authored);
        let config = GenerationConfig {
            seed: 0x4d34_4953_4f4c_4154,
            count: 1,
            temperature: 1.5,
            allow_black_sources: false,
        };
        let m3 = generate(&m3_anchor, &config).unwrap().remove(0).patch;
        let mut m4 = generate(&m4_anchor, &config).unwrap().remove(0).patch;
        assert_ne!(m4.master_motion, m4_anchor.master_motion);
        assert_ne!(m4.layers[0].motion, m4_anchor.layers[0].motion);
        m4.master_motion = None;
        m4.layers[0].motion = None;
        assert_eq!(
            serde_yaml::to_string(&m4).unwrap(),
            serde_yaml::to_string(&m3).unwrap(),
            "M4 domains must preserve the projected v6/M3 stream"
        );
    }

    fn assert_rack_topology_and_switches_eq(expected: &VisualRack, actual: &VisualRack) {
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual.next_node_id_raw(), expected.next_node_id_raw());
        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert_eq!(actual.stable_id, expected.stable_id);
            assert_eq!(actual.enabled, expected.enabled);
            assert_eq!(actual.blend, expected.blend);
            assert_eq!(actual.kind.tag(), expected.kind.tag());
            match (expected.kind, actual.kind) {
                (VisualNodeKind::Transform(expected), VisualNodeKind::Transform(actual)) => {
                    assert_eq!(actual.fit, expected.fit);
                    assert_eq!(actual.edge, expected.edge);
                    assert_eq!(actual.sampling, expected.sampling);
                }
                (VisualNodeKind::Key(expected), VisualNodeKind::Key(actual)) => {
                    assert_eq!(actual.mode, expected.mode);
                    assert_eq!(actual.invert, expected.invert);
                }
                (VisualNodeKind::Grain(expected), VisualNodeKind::Grain(actual)) => {
                    assert_eq!(actual.algorithm, expected.algorithm);
                    assert_eq!(actual.color, expected.color);
                }
                (
                    VisualNodeKind::Mask(MaskParams::Rectangle(expected)),
                    VisualNodeKind::Mask(MaskParams::Rectangle(actual)),
                ) => assert_eq!(actual.invert, expected.invert),
                (
                    VisualNodeKind::Mask(MaskParams::Ellipse(expected)),
                    VisualNodeKind::Mask(MaskParams::Ellipse(actual)),
                ) => assert_eq!(actual.invert, expected.invert),
                (
                    VisualNodeKind::Mask(MaskParams::Image(expected)),
                    VisualNodeKind::Mask(MaskParams::Image(actual)),
                ) => {
                    assert_eq!(actual.tap, expected.tap);
                    assert_eq!(actual.channel, expected.channel);
                    assert_eq!(actual.invert, expected.invert);
                }
                _ => {}
            }
        }
    }

    fn assert_creative_topology_and_switches_eq(expected: &PatchState, actual: &PatchState) {
        assert_eq!(actual.visual_schema_version, expected.visual_schema_version);
        match (&expected.master_rack, &actual.master_rack) {
            (Some(expected), Some(actual)) => {
                assert_rack_topology_and_switches_eq(expected, actual)
            }
            (None, None) => {}
            _ => panic!("master-rack presence changed"),
        }
        assert_eq!(actual.layers.len(), expected.layers.len());
        for (expected, actual) in expected.layers.iter().zip(&actual.layers) {
            match (&expected.rack, &actual.rack) {
                (Some(expected), Some(actual)) => {
                    assert_rack_topology_and_switches_eq(expected, actual)
                }
                (None, None) => {}
                _ => panic!("layer-rack presence changed"),
            }
        }

        match (&expected.composition, &actual.composition) {
            (Some(expected), Some(actual)) => {
                assert_eq!(actual.root(), expected.root());
                assert_eq!(actual.next_group_id_raw(), expected.next_group_id_raw());
                assert_eq!(actual.bus_crossfade(), expected.bus_crossfade());
                assert_eq!(actual.groups().len(), expected.groups().len());
                for expected_group in expected.groups() {
                    let actual_group = actual.group(expected_group.id).unwrap();
                    assert_eq!(actual_group.id, expected_group.id);
                    assert_eq!(actual_group.name, expected_group.name);
                    assert_eq!(actual_group.members, expected_group.members);
                    assert_eq!(actual_group.solo, expected_group.solo);
                    assert_eq!(actual_group.bypass, expected_group.bypass);
                    assert_eq!(actual_group.bus, expected_group.bus);
                    assert_rack_topology_and_switches_eq(&expected_group.rack, &actual_group.rack);
                    match (expected_group.matte, actual_group.matte) {
                        (Some(expected), Some(actual)) => {
                            assert_eq!(actual.tap, expected.tap);
                            assert_eq!(actual.channel, expected.channel);
                            assert_eq!(actual.invert, expected.invert);
                        }
                        (None, None) => {}
                        _ => panic!("group-matte presence changed"),
                    }
                }
            }
            (None, None) => {}
            _ => panic!("composition presence changed"),
        }
    }

    #[test]
    fn splitmix64_matches_published_golden_vector() {
        let mut rng = SplitMix64::new(0);
        let values: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
        assert_eq!(
            values,
            [
                0xe220_a839_7b1d_cdaf,
                0x6e78_9e6a_a1b9_65f4,
                0x06c4_5d18_8009_454f,
                0xf88b_b8a8_724c_81ec,
                0x1b39_896a_51a8_749b,
            ]
        );
    }

    #[test]
    fn reflection_and_wrap_have_no_wall_pileup() {
        assert!((reflect(1.2, 0.0, 1.0) - 0.8).abs() < 1e-6);
        assert!((reflect(-0.2, 0.0, 1.0) - 0.2).abs() < 1e-6);
        assert!((wrap(190.0, -180.0, 180.0) + 170.0).abs() < 1e-6);
    }

    #[test]
    fn generation_is_deterministic_and_temperature_zero_is_normalized_identity() {
        let config = GenerationConfig {
            seed: 42,
            count: 2,
            temperature: 0.0,
            allow_black_sources: false,
        };
        let first = generate(&anchor(), &config).unwrap();
        let second = generate(&anchor(), &config).unwrap();
        assert_eq!(first[0].manifest, second[0].manifest);
        assert_eq!(
            serde_yaml::to_string(&first[0].patch).unwrap(),
            serde_yaml::to_string(&second[0].patch).unwrap()
        );
        assert_eq!(first[0].patch.master.cellular_amount, 0.0);
        assert_eq!(first[0].patch.master.shift_amount, 0.0);
        assert_eq!(first[0].patch.master.shift_block_size, 8.0);
        assert_eq!(first[0].patch.master.shift_density, 0.5);
        assert_eq!(first[0].patch.master.shift_speed, 3.0);
        assert_eq!(first[0].patch.master_transform, SpatialTransform::default());
        assert_eq!(first[0].patch.layers[0].filename, "clip.mp4");
        assert_eq!(
            first[0].patch.layers[0].transform,
            SpatialTransform::default()
        );
        assert_eq!(first[0].manifest.generator_version, GENERATOR_VERSION);
        assert!(first[0].manifest.title.contains(' '));
    }

    #[test]
    fn generator_v4_collapses_performance_topology_without_randomizing_routes() {
        let prepared: PatchState = serde_yaml::from_str(
            r#"
master: {}
layers:
  - filename: stale.mov
    effects: {}
    clip_slots:
      - { id: 9, name: A, filename: a.mov, transport: { rate: 0.5 } }
      - { id: 27, name: B, filename: b.mov, transport: { rate: 1.25, sample_fps: 25 } }
    active_clip_slot: 27
    matte:
      enabled: true
      input: { source: selected_layer, layer_position: 0, stage: pre_local_effects }
      channel: luma
      amount: 0.75
scenes:
  - id: 3
    bindings:
      - { layer_position: 0, slot_id: 27 }
"#,
        )
        .unwrap();
        let piece = generate(
            &prepared,
            &GenerationConfig {
                seed: 11,
                count: 1,
                temperature: 1.0,
                allow_black_sources: false,
            },
        )
        .unwrap()
        .remove(0);
        let layer = &piece.patch.layers[0];
        assert_eq!(piece.manifest.generator_version, "7");
        assert_eq!(layer.clip_slots.len(), 1);
        assert_eq!(
            layer.active_clip_slot,
            Some(crate::performance::ClipSlotId::LEGACY)
        );
        assert_eq!(layer.filename, "b.mov", "the selected source is retained");
        assert_eq!(
            layer
                .clip_slots
                .get(crate::performance::ClipSlotId::LEGACY)
                .unwrap()
                .filename,
            "b.mov"
        );
        assert!(layer.matte.is_legacy_disabled());
        assert!(piece.patch.scenes.is_empty());
    }

    #[test]
    fn generator_v5_mutates_creative_values_without_perturbing_topology_or_v4_streams() {
        let legacy = anchor();
        let advanced = advanced_anchor();
        let config = GenerationConfig {
            seed: 0x5635_4352_4541_5449,
            count: 1,
            temperature: 1.25,
            allow_black_sources: false,
        };
        let legacy_piece = generate(&legacy, &config).unwrap().remove(0);
        let advanced_piece = generate(&advanced, &config).unwrap().remove(0);

        assert_eq!(advanced_piece.manifest.generator_version, "7");
        assert_creative_topology_and_switches_eq(&advanced, &advanced_piece.patch);
        assert_ne!(advanced_piece.patch.master_rack, advanced.master_rack);
        assert_ne!(advanced_piece.patch.layers[0].rack, advanced.layers[0].rack);
        let original_group = advanced
            .composition
            .as_ref()
            .unwrap()
            .group(GroupId::new(7).unwrap())
            .unwrap();
        let generated_group = advanced_piece
            .patch
            .composition
            .as_ref()
            .unwrap()
            .group(GroupId::new(7).unwrap())
            .unwrap();
        assert!(
            generated_group.opacity != original_group.opacity
                || generated_group.transform != original_group.transform
                || generated_group.rack != original_group.rack
                || generated_group.matte != original_group.matte,
            "nonzero temperature must evolve safe group values"
        );

        let mut projected = advanced_piece.patch.clone();
        projected.master_rack = None;
        projected.composition = None;
        projected.visual_schema_version = 0;
        for layer in &mut projected.layers {
            layer.rack = None;
        }
        assert_eq!(
            serde_yaml::to_string(&projected).unwrap(),
            serde_yaml::to_string(&legacy_piece.patch).unwrap(),
            "M2 creative topology must not consume or reorder any v4 mutation stream"
        );
    }

    #[test]
    fn generator_v5_temperature_zero_is_exact_creative_identity() {
        let advanced = advanced_anchor();
        let generated = generate(
            &advanced,
            &GenerationConfig {
                seed: 0x5445_4d50_5f5a_4552,
                count: 1,
                temperature: 0.0,
                allow_black_sources: false,
            },
        )
        .unwrap()
        .remove(0)
        .patch;

        assert_eq!(generated.master_rack, advanced.master_rack);
        assert_eq!(generated.layers[0].rack, advanced.layers[0].rack);
        assert_eq!(generated.composition, advanced.composition);
    }

    #[test]
    fn generator_v5_creative_domains_ignore_unrelated_node_and_group_insertions() {
        use crate::composition::GroupName;
        use crate::visual_rack::{GrainParams, ShiftParams};

        let base = advanced_anchor();
        let mut expanded = advanced_anchor();
        expanded
            .master_rack
            .as_mut()
            .unwrap()
            .insert(
                0,
                VisualNodeKind::Grain(GrainParams {
                    intensity: 0.03,
                    ..GrainParams::default()
                }),
            )
            .unwrap();
        expanded.layers[0]
            .rack
            .as_mut()
            .unwrap()
            .insert(
                0,
                VisualNodeKind::Shift(ShiftParams {
                    amount: 0.2,
                    ..ShiftParams::default()
                }),
            )
            .unwrap();
        expanded
            .composition
            .as_mut()
            .unwrap()
            .insert_empty_group(GroupName::new("unrelated").unwrap(), 0)
            .unwrap();

        let config = GenerationConfig {
            seed: 0x5354_4142_4c45_5f49,
            count: 1,
            temperature: 1.6,
            allow_black_sources: false,
        };
        let generated_base = generate(&base, &config).unwrap().remove(0).patch;
        let generated_expanded = generate(&expanded, &config).unwrap().remove(0).patch;

        assert_eq!(
            generated_base
                .master_rack
                .as_ref()
                .unwrap()
                .get(NodeId::new(3).unwrap()),
            generated_expanded
                .master_rack
                .as_ref()
                .unwrap()
                .get(NodeId::new(3).unwrap())
        );
        assert_eq!(
            generated_base.layers[0]
                .rack
                .as_ref()
                .unwrap()
                .get(NodeId::new(11).unwrap()),
            generated_expanded.layers[0]
                .rack
                .as_ref()
                .unwrap()
                .get(NodeId::new(11).unwrap())
        );
        let group_id = GroupId::new(7).unwrap();
        let base_group = generated_base
            .composition
            .as_ref()
            .unwrap()
            .group(group_id)
            .unwrap();
        let expanded_group = generated_expanded
            .composition
            .as_ref()
            .unwrap()
            .group(group_id)
            .unwrap();
        assert_eq!(base_group.opacity, expanded_group.opacity);
        assert_eq!(base_group.transform, expanded_group.transform);
        assert_eq!(base_group.matte, expanded_group.matte);
        assert_eq!(
            base_group.rack.get(NodeId::new(3).unwrap()),
            expanded_group.rack.get(NodeId::new(3).unwrap())
        );
    }

    #[test]
    fn generator_v5_reverts_only_edge_values_needed_to_keep_dormant_cycles_safe() {
        use crate::visual_rack::{
            EdgeTiming, ImageMatte, MatteChannel, SavedImageSource, SavedImageTap,
        };

        let mut source = advanced_anchor();
        let group_id = GroupId::new(7).unwrap();
        let group = source
            .composition
            .as_mut()
            .unwrap()
            .group_mut(group_id)
            .unwrap();
        group.bypass = false;
        group.matte = Some(ImageMatte {
            tap: SavedImageTap {
                source: SavedImageSource::GroupOutput { group_id },
                timing: EdgeTiming::CurrentFrame,
            },
            channel: MatteChannel::Luma,
            invert: false,
            amount: 0.0,
            threshold: 0.27,
            softness: 0.08,
        });
        let first = group.rack.get_mut(NodeId::new(3).unwrap()).unwrap();
        first.kind = VisualNodeKind::Mask(MaskParams::Image(ImageMatte {
            tap: SavedImageTap {
                source: SavedImageSource::GroupOutput { group_id },
                timing: EdgeTiming::CurrentFrame,
            },
            channel: MatteChannel::Red,
            invert: true,
            amount: 0.0,
            threshold: 0.31,
            softness: 0.07,
        }));
        let second_id = group
            .rack
            .push(VisualNodeKind::Mask(MaskParams::Image(ImageMatte {
                tap: SavedImageTap {
                    source: SavedImageSource::GroupOutput { group_id },
                    timing: EdgeTiming::CurrentFrame,
                },
                channel: MatteChannel::Blue,
                invert: false,
                amount: 1.0,
                threshold: 0.63,
                softness: 0.11,
            })))
            .unwrap();
        group.rack.get_mut(second_id).unwrap().wet = 0.0;

        let generated = generate(
            &source,
            &GenerationConfig {
                seed: 0x4359_434c_455f_5341,
                count: 1,
                temperature: 2.0,
                allow_black_sources: false,
            },
        )
        .unwrap()
        .remove(0)
        .patch;
        validate_generated_patch(&generated).unwrap();
        let generated_group = generated
            .composition
            .as_ref()
            .unwrap()
            .group(group_id)
            .unwrap();
        assert_eq!(generated_group.matte.unwrap().amount, 0.0);
        let VisualNodeKind::Mask(MaskParams::Image(first)) = generated_group
            .rack
            .get(NodeId::new(3).unwrap())
            .unwrap()
            .kind
        else {
            panic!("first route changed kind");
        };
        assert_eq!(first.amount, 0.0);
        assert_eq!(generated_group.rack.get(second_id).unwrap().wet, 0.0);
        assert!(
            first.threshold != 0.31 || first.softness != 0.07,
            "safe image-matte values should still evolve"
        );
    }

    #[test]
    fn generator_v5_creative_values_are_deterministic_finite_and_sanitized() {
        let source = advanced_anchor();
        let config = GenerationConfig {
            seed: 0x424f_554e_4445_445f,
            count: 8,
            temperature: 2.0,
            allow_black_sources: false,
        };
        let first = generate(&source, &config).unwrap();
        let second = generate(&source, &config).unwrap();
        assert_eq!(first.len(), second.len());
        for (first, second) in first.iter().zip(&second) {
            let first_yaml = serde_yaml::to_string(&first.patch).unwrap();
            assert_eq!(first_yaml, serde_yaml::to_string(&second.patch).unwrap());
            let restored: PatchState = serde_yaml::from_str(&first_yaml).unwrap();
            assert_eq!(
                first_yaml,
                serde_yaml::to_string(&restored).unwrap(),
                "generated values must already be finite and within persisted bounds"
            );
            assert_creative_topology_and_switches_eq(&source, &first.patch);
        }
    }

    #[test]
    fn temperature_zero_canonicalizes_invalid_runtime_values() {
        let mut patch = anchor();
        patch.master.cellular_amount = f32::NAN;
        patch.master.cellular_scale = f32::INFINITY;
        patch.layers[0].opacity = f32::NEG_INFINITY;
        patch.layers[0].speed = 99.0;
        patch.layers[0].fps = f32::NAN;
        patch.layers[0].blend_mode = "unknown".to_string();
        patch.master_transform.rotation_deg = f32::INFINITY;
        patch.layers[0].transform.crop = [9.0, 9.0, 9.0, 9.0];
        let piece = generate(
            &patch,
            &GenerationConfig {
                seed: 8,
                count: 1,
                temperature: 0.0,
                allow_black_sources: false,
            },
        )
        .unwrap()
        .remove(0);
        assert_eq!(piece.patch.master.cellular_amount, 0.0);
        assert_eq!(piece.patch.master.cellular_scale, 10.0);
        assert_eq!(piece.patch.layers[0].opacity, 1.0);
        assert_eq!(piece.patch.layers[0].speed, 4.0);
        assert_eq!(piece.patch.layers[0].fps, 30.0);
        assert_eq!(piece.patch.layers[0].blend_mode, "normal");
        assert_eq!(piece.patch.master_transform.rotation_deg, 0.0);
        assert_eq!(
            piece.patch.layers[0].transform,
            piece.patch.layers[0].transform.sanitized()
        );
    }

    #[test]
    fn procedural_generation_preserves_and_round_trips_every_curated_blend_key() {
        for blend_mode in BlendMode::ALL {
            let mut patch = anchor();
            patch.layers[0].blend_mode = blend_mode.key().to_string();
            let piece = generate(
                &patch,
                &GenerationConfig {
                    seed: 0x424c_454e_4400 + u64::from(blend_mode.as_u32()),
                    count: 1,
                    temperature: 0.0,
                    allow_black_sources: false,
                },
            )
            .unwrap()
            .remove(0);
            assert_eq!(
                piece.patch.layers[0].blend_mode,
                blend_mode.key(),
                "procedural normalization collapsed {blend_mode:?}"
            );

            let yaml = serde_yaml::to_string(&piece.patch).unwrap();
            let restored: PatchState = serde_yaml::from_str(&yaml).unwrap();
            assert_eq!(restored.layers[0].blend_mode, blend_mode.key());
        }
    }

    #[test]
    fn expanded_blend_mutation_reaches_all_modes_without_advancing_the_legacy_rng_stream() {
        const LEGACY: &[BlendMode] = &[
            BlendMode::Normal,
            BlendMode::Screen,
            BlendMode::Multiply,
            BlendMode::Difference,
        ];
        assert_eq!(&BlendMode::ALL[..LEGACY.len()], LEGACY);

        for seed in [0, 1, 7, 0xaced, u64::MAX] {
            for probability in [0.0, 0.37, 1.0] {
                let mut expanded = SplitMix64::new(seed);
                let _ = mutate_blend_mode("normal", "normal", probability, &mut expanded);

                let mut legacy = SplitMix64::new(seed);
                let _ = mutate_discrete(
                    BlendMode::Normal,
                    BlendMode::Normal,
                    LEGACY,
                    probability,
                    &mut legacy,
                );
                assert_eq!(
                    expanded.next_u64(),
                    legacy.next_u64(),
                    "the expanded choice set must not add or reorder RNG draws"
                );
            }
        }

        let mut seen = [false; BlendMode::ALL.len()];
        for seed in 0..4096 {
            let mut rng = SplitMix64::new(seed);
            let blend = mutate_blend_mode("normal", "normal", 1.0, &mut rng);
            seen[blend.as_u32() as usize] = true;
        }
        assert!(
            seen.into_iter().all(|was_seen| was_seen),
            "the typed mutation choices must expose every curated mode"
        );
    }

    #[test]
    fn sequential_variants_form_a_mean_reverting_walk() {
        let config = GenerationConfig {
            seed: 0x5eed,
            count: 12,
            temperature: 0.7,
            allow_black_sources: false,
        };
        let pieces = generate(&anchor(), &config).unwrap();
        let values: Vec<f32> = pieces
            .iter()
            .map(|piece| piece.patch.layers[0].opacity)
            .collect();
        assert!(values.windows(2).any(|pair| pair[0] != pair[1]));
        assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));
        assert_eq!(pieces[0].manifest.lineage, pieces[11].manifest.lineage);
    }

    #[test]
    fn categorical_mutation_reverts_and_titles_are_deterministic() {
        let mut rng = SplitMix64::new(0xaced);
        let mut value = 3u32;
        for _ in 0..128 {
            value = mutate_discrete(0, value, &[0, 1, 2, 3], 0.0, &mut rng);
        }
        assert_eq!(value, 0);

        let patch = anchor();
        let first = title_for(&patch, 0x1234_5678);
        let second = title_for(&patch, 0x1234_5678);
        assert_eq!(first, second);
        assert_eq!(first.split_ascii_whitespace().count(), 2);
    }

    #[test]
    fn manifests_never_expose_absolute_video_paths() {
        let mut patch = anchor();
        patch.layers[0].filename = r"C:\private\performance\clip.mp4".to_string();
        patch.layers[0].source_path = r"C:\private\performance\clip.mp4".to_string();
        let piece = generate(
            &patch,
            &GenerationConfig {
                seed: 4,
                count: 1,
                temperature: 0.0,
                allow_black_sources: false,
            },
        )
        .unwrap()
        .remove(0);
        assert_eq!(piece.manifest.logical_sources, ["clip.mp4"]);
        let json = serde_json::to_string(&piece.manifest).unwrap();
        assert!(!json.contains("private"));
        assert!(!json.contains("C:"));

        patch.layers[0].filename = "/home/performer/private/other.mov".to_string();
        patch.layers[0].source_path = "/home/performer/private/other.mov".to_string();
        let manifest = generate(
            &patch,
            &GenerationConfig {
                seed: 5,
                count: 1,
                temperature: 0.0,
                allow_black_sources: false,
            },
        )
        .unwrap()
        .remove(0)
        .manifest;
        assert_eq!(manifest.logical_sources, ["other.mov"]);
    }

    #[test]
    fn verified_generation_is_byte_identical_across_roots_and_content_sensitive() {
        let base = std::env::temp_dir().join(format!(
            "collideoscope-procedural-content-id-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let first_root = base.join("first-private-root");
        let second_root = base.join("second-private-root");
        fs::create_dir_all(&first_root).unwrap();
        fs::create_dir_all(&second_root).unwrap();
        let first_media = first_root.join("clip.mp4");
        let second_media = second_root.join("clip.mp4");
        fs::write(&first_media, b"same source bytes").unwrap();
        fs::write(&second_media, b"same source bytes").unwrap();

        let config = GenerationConfig {
            seed: 0x5151,
            count: 1,
            temperature: 0.5,
            allow_black_sources: false,
        };
        let build = |media: &Path, root: &Path| {
            let mut patch = anchor();
            patch.layers[0].source_path = media.to_string_lossy().into_owned();
            let inventory = preflight_sources(
                &patch,
                &SourcePreflightConfig {
                    anchor_dir: Some(root.to_path_buf()),
                    ..SourcePreflightConfig::default()
                },
            )
            .unwrap();
            generate_with_inventory(&patch, &config, &inventory)
                .unwrap()
                .remove(0)
        };
        let first = build(&first_media, &first_root);
        let second = build(&second_media, &second_root);
        let serialize = |piece: &GeneratedPiece| {
            (
                serde_yaml::to_string(&piece.patch).unwrap(),
                serde_json::to_vec_pretty(&piece.manifest).unwrap(),
                serde_json::to_vec_pretty(&piece.preflight).unwrap(),
            )
        };
        assert_eq!(serialize(&first), serialize(&second));
        assert!(first.patch.layers[0]
            .source_path
            .starts_with(crate::media_source::CONTENT_SHA256_PREFIX));
        assert!(first.manifest.identity_complete);
        assert!(!first.preflight.pixel_identity_claimed);
        let artifacts = format!(
            "{}\n{}\n{}",
            serde_yaml::to_string(&first.patch).unwrap(),
            serde_json::to_string(&first.manifest).unwrap(),
            serde_json::to_string(&first.preflight).unwrap()
        );
        assert!(!artifacts.contains(&first_root.to_string_lossy().to_string()));
        assert!(!artifacts.contains(&second_root.to_string_lossy().to_string()));

        fs::write(&second_media, b"changed source bytes").unwrap();
        let changed = build(&second_media, &second_root);
        assert_ne!(first.manifest.anchor_sha256, changed.manifest.anchor_sha256);
        assert_ne!(first.manifest.piece_sha256, changed.manifest.piece_sha256);
        assert_ne!(
            first.preflight.sources[0].sha256,
            changed.preflight.sources[0].sha256
        );
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn version_one_through_six_manifests_deserialize_with_safe_identity_defaults() {
        let legacy = r#"{
            "schema_version": 1,
            "generator_version": "VERSION",
            "seed": 9,
            "index": 0,
            "temperature": 0.5,
            "title": "grid field",
            "slug": "grid-field",
            "anchor_fnv1a64": "0123456789abcdef",
            "lineage": "0123456789abcdef",
            "logical_sources": ["clip.mp4"],
            "warnings": []
        }"#;
        for version in ["1", "2", "3", "4", "5", "6"] {
            let manifest: Manifest =
                serde_json::from_str(&legacy.replace("VERSION", version)).unwrap();
            assert_eq!(manifest.schema_version, 1);
            assert_eq!(manifest.generator_version, version);
            assert!(manifest.anchor_sha256.is_empty());
            assert!(manifest.piece_sha256.is_empty());
            assert!(!manifest.identity_complete);
            assert!(manifest.sources.is_empty());
        }
    }

    #[test]
    fn generated_values_stay_finite_bounded_and_preserve_topology() {
        let config = GenerationConfig {
            seed: 99,
            count: 32,
            temperature: 2.0,
            allow_black_sources: false,
        };
        let mut source = anchor();
        source.layers[0].bypass_master_fx = true;
        source.master_transform.fit = crate::spatial::FitMode::Fill;
        source.master_transform.edge = crate::spatial::EdgeMode::Repeat;
        source.master_transform.sampling = crate::spatial::SamplingMode::Nearest;
        source.layers[0].transform.fit = crate::spatial::FitMode::Native;
        source.layers[0].transform.edge = crate::spatial::EdgeMode::Mirror;
        source.layers[0].transform.sampling = crate::spatial::SamplingMode::Nearest;
        let mut saw_spatial_change = false;
        for piece in generate(&source, &config).unwrap() {
            assert_eq!(piece.patch.layers.len(), 1);
            let layer = &piece.patch.layers[0];
            assert_eq!(layer.filename, "clip.mp4");
            assert!(
                layer.bypass_master_fx,
                "procedural mutation must preserve the layer's routing topology"
            );
            assert!(layer.opacity.is_finite() && (0.0..=1.0).contains(&layer.opacity));
            assert!(layer.speed.is_finite() && (0.25..=4.0).contains(&layer.speed));
            assert!(piece.patch.master.cellular_amount.is_finite());
            assert!((0.0..=1.0).contains(&piece.patch.master.cellular_amount));
            assert!((2.0..=32.0).contains(&piece.patch.master.cellular_scale));
            assert!((0.0..=1.0).contains(&piece.patch.master.shift_amount));
            assert!((2.0..=256.0).contains(&piece.patch.master.shift_block_size));
            assert!((0.0..=1.0).contains(&piece.patch.master.shift_density));
            assert!((0.0..=20.0).contains(&piece.patch.master.shift_speed));
            assert_eq!(
                piece.patch.master_transform,
                piece.patch.master_transform.sanitized()
            );
            assert_eq!(layer.transform, layer.transform.sanitized());
            assert_eq!(
                piece.patch.master_transform.fit,
                crate::spatial::FitMode::Fill
            );
            assert_eq!(
                piece.patch.master_transform.edge,
                crate::spatial::EdgeMode::Repeat
            );
            assert_eq!(
                piece.patch.master_transform.sampling,
                crate::spatial::SamplingMode::Nearest
            );
            assert_eq!(layer.transform.fit, crate::spatial::FitMode::Native);
            assert_eq!(layer.transform.edge, crate::spatial::EdgeMode::Mirror);
            assert_eq!(
                layer.transform.sampling,
                crate::spatial::SamplingMode::Nearest
            );
            saw_spatial_change |= piece.patch.master_transform.position != [0.0, 0.0]
                || layer.transform.position != [0.0, 0.0];
        }
        assert!(
            saw_spatial_change,
            "temperature must participate in spatial generation"
        );
    }

    #[test]
    fn live_sources_and_active_morph_require_explicit_resolution() {
        let mut patch = anchor();
        patch.layers[0].source_path = "spout://camera".to_string();
        let mut config = GenerationConfig {
            seed: 1,
            count: 1,
            temperature: 0.5,
            allow_black_sources: false,
        };
        let error = match generate(&patch, &config) {
            Ok(_) => panic!("Spout generation should require explicit black-source policy"),
            Err(error) => error,
        };
        assert!(error.contains("Spout"));
        config.allow_black_sources = true;
        let generated = generate(&patch, &config).unwrap();
        assert_eq!(generated[0].manifest.warnings.len(), 1);
    }

    #[test]
    fn patch_only_writer_is_atomic_and_refuses_overwrite() {
        let root = std::env::temp_dir().join(format!(
            "collideoscope-procedural-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let pieces = generate(
            &anchor(),
            &GenerationConfig {
                seed: 7,
                count: 1,
                temperature: 0.5,
                allow_black_sources: false,
            },
        )
        .unwrap();
        let created = write_patch_only(&pieces, &root).unwrap();
        assert!(created[0].join("patch.yaml").is_file());
        assert!(created[0].join("manifest.json").is_file());
        assert!(created[0].join("preflight.json").is_file());
        assert_eq!(
            fs::read_dir(&created[0])
                .unwrap()
                .filter_map(Result::ok)
                .count(),
            3
        );
        assert!(write_patch_only(&pieces, &root)
            .unwrap_err()
            .contains("overwrite"));
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().starts_with('.'))
            .collect();
        assert!(leftovers.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_preflight_prevents_a_known_late_conflict_from_becoming_partial() {
        let root = std::env::temp_dir().join(format!(
            "collideoscope-procedural-preflight-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let pieces = generate(
            &anchor(),
            &GenerationConfig {
                seed: 77,
                count: 2,
                temperature: 0.5,
                allow_black_sources: false,
            },
        )
        .unwrap();
        let second = write_patch_only(&pieces[1..], &root).unwrap();
        let error = write_patch_only(&pieces, &root).unwrap_err();
        assert!(error.contains("overwrite"));
        assert_eq!(
            fs::read_dir(&root)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.path().is_dir())
                .count(),
            1
        );
        assert!(second[0].is_dir());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn no_replace_commit_preserves_an_existing_destination() {
        let root = std::env::temp_dir().join(format!(
            "collideoscope-procedural-noreplace-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source");
        let destination = root.join("destination");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&destination).unwrap();
        write_new_file(&source.join("source.txt"), b"source").unwrap();
        write_new_file(&destination.join("destination.txt"), b"destination").unwrap();
        assert!(rename_noreplace(&source, &destination).is_err());
        assert!(source.join("source.txt").is_file());
        assert_eq!(
            fs::read(destination.join("destination.txt")).unwrap(),
            b"destination"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn collector_status_never_blocks_and_drop_gives_queued_work_grace() {
        let root = std::env::temp_dir().join(format!(
            "collideoscope-procedural-collector-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let collector = PatchCollector::new();
        collector
            .phase
            .store(CAPTURE_SAVED, std::sync::atomic::Ordering::Release);
        {
            let _guard = collector.status.lock().unwrap();
            assert_eq!(collector.status(), "Saving…");
        }
        assert_eq!(
            collector.try_submit(anchor(), root.clone()),
            CaptureSubmit::Queued
        );
        drop(collector);
        let captures: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect();
        assert_eq!(captures.len(), 1);
        assert_eq!(
            captures[0].extension().and_then(|ext| ext.to_str()),
            Some("yaml")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stale_capture_completion_cannot_overwrite_newer_saving_status() {
        let status = Mutex::new("Saving…".to_string());
        let phase = std::sync::atomic::AtomicU8::new(CAPTURE_SAVING);
        let latest = std::sync::atomic::AtomicU64::new(2);
        publish_capture_result(
            &status,
            &phase,
            &latest,
            1,
            Ok(PathBuf::from("capture-old.yaml")),
        );
        assert_eq!(*status.lock().unwrap(), "Saving…");
        assert_eq!(
            phase.load(std::sync::atomic::Ordering::Acquire),
            CAPTURE_SAVING
        );
        publish_capture_result(
            &status,
            &phase,
            &latest,
            2,
            Ok(PathBuf::from("capture-new.yaml")),
        );
        assert_eq!(*status.lock().unwrap(), "Saved capture-new.yaml");
        assert_eq!(
            phase.load(std::sync::atomic::Ordering::Acquire),
            CAPTURE_SAVED
        );
    }

    #[test]
    fn blocked_old_completion_rechecks_sequence_after_new_submit_publication() {
        let status = Arc::new(Mutex::new("Saving…".to_string()));
        let phase = Arc::new(std::sync::atomic::AtomicU8::new(CAPTURE_SAVING));
        let latest = Arc::new(std::sync::atomic::AtomicU64::new(1));
        let guard = status.lock().unwrap();
        let worker_status = status.clone();
        let worker_phase = phase.clone();
        let worker_latest = latest.clone();
        let publisher = thread::spawn(move || {
            publish_capture_result(
                &worker_status,
                &worker_phase,
                &worker_latest,
                1,
                Ok(PathBuf::from("capture-old.yaml")),
            );
        });
        latest.store(2, std::sync::atomic::Ordering::Release);
        phase.store(CAPTURE_SAVING, std::sync::atomic::Ordering::Release);
        drop(guard);
        publisher.join().unwrap();
        assert_eq!(
            phase.load(std::sync::atomic::Ordering::Acquire),
            CAPTURE_SAVING
        );
        assert_eq!(*status.lock().unwrap(), "Saving…");
    }
}
