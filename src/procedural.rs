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
use crate::symmetry::SymmetryParams;
use crate::visual_rack::{
    DisplaceParams, GroupId, MaskParams, NodeId, ResidualParams, VisualNodeKind, VisualRack,
};

/// v6 records the M3 Temporal Originals generation law; v7 adds M4 Motion in
/// new isolated domains; v8 adds the B2 procedural field scalars and the four
/// flow-shaping controls in fresh domains; v9 adds the B13 small-effects
/// families in fresh per-scope domains; v10 adds the B4 display-physics
/// continuous values in fresh field-isolated domains — every field mutated by
/// an earlier version keeps its byte-identical stream, but a generated piece
/// now carries the new values, so the version names the difference. Manifest
/// readers remain data-driven and accept every earlier version string.
// "11": the B8 blend audit widened `BlendMode::ALL` from 15 to 25 modes.
// `mutate_blend_mode` draws its selection modulo the choice count, so a
// firing seed can now land on a different mode; the draw count itself is
// unchanged (`expanded_blend_mutation_reaches_all_modes_without_advancing_
// the_legacy_rng_stream`), so every other field in the stream is unmoved.
// "12": B5 adds the eight codec-mosh continuous values in fresh
// field-isolated domains; every earlier stream is byte-stable, but a
// generated piece now carries the new values.
// "13": preflight schema 2 records the final emitted patch's resolved
// per-layer Master and Temporal bypass values after a YAML round trip. The
// PRNG streams and mutation laws are unchanged; the version names the new
// inspectable evidence carried beside every generated piece.
// "14": B5 adds motion wipe, vector smear, and motion-trail controls in
// fresh field-isolated domains. Earlier domains retain their exact streams.
pub const GENERATOR_VERSION: &str = "14";
pub const MAX_GENERATED_COUNT: usize = 256;
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;
pub const PREFLIGHT_SCHEMA_VERSION: u32 = 2;
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

/// Resolved authored bypass values in one final generated layer. Indices are
/// zero-based and follow UI stack order, matching `ManifestSource::layer_index`.
/// These are configuration facts, not a claim that a renderer admitted or
/// activated a particular runtime topology.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayerBypassMeasurement {
    pub layer_index: usize,
    pub bypass_master_fx: bool,
    pub bypass_temporal_fx: bool,
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
    /// Schema 1 receipts omit this field and deserialize to an empty,
    /// explicitly unmeasured set. Schema 2 always serializes one record per
    /// final current-stack layer, including all-false values.
    #[serde(default)]
    pub layer_bypass_states: Vec<LayerBypassMeasurement>,
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

/// Appended B13 stream: the small-effects tranche mutates in its own
/// domain-separated sequence per scope, so every pre-B13 procedural stream
/// stays byte-for-byte stable for the same seed. `negative_mode` is a
/// discrete law and never mutates; the three optics mutate at master scope
/// only, matching their master-only authoring law.
const PROCEDURAL_SMALL_FX_DOMAIN: u64 = 0x4233_534d_464c_5800;

fn mutate_small_effects(
    anchor: &EffectsConfig,
    e: &mut EffectsConfig,
    temperature: f32,
    seed: u64,
    index: usize,
    scope_domain: u64,
    include_optics: bool,
) {
    if temperature == 0.0 {
        return;
    }
    let t = temperature;
    let mut rng = SplitMix64::new(domain_seed(
        seed,
        index,
        PROCEDURAL_SMALL_FX_DOMAIN ^ scope_domain,
    ));
    let rng = &mut rng;
    e.contour = mutate_linear(anchor.contour, e.contour, 0.0, 1.0, t * 0.14, rng);
    e.contour_bands = mutate_log(
        anchor.contour_bands,
        e.contour_bands,
        2.0,
        40.0,
        t * 0.25,
        rng,
    )
    .round();
    e.contour_width = mutate_linear(
        anchor.contour_width,
        e.contour_width,
        0.2,
        6.0,
        t * 0.5,
        rng,
    );
    e.contour_hue = mutate_linear(anchor.contour_hue, e.contour_hue, 0.0, 1.0, t * 0.15, rng);
    e.contour_fill = mutate_linear(anchor.contour_fill, e.contour_fill, 0.0, 1.0, t * 0.12, rng);
    e.flatten = mutate_linear(anchor.flatten, e.flatten, 0.0, 1.0, t * 0.14, rng);
    e.flatten_levels = mutate_log(
        anchor.flatten_levels,
        e.flatten_levels,
        2.0,
        16.0,
        t * 0.25,
        rng,
    )
    .round();
    e.contour_dither = mutate_linear(
        anchor.contour_dither,
        e.contour_dither,
        0.0,
        1.0,
        t * 0.15,
        rng,
    );
    e.solarize = mutate_linear(anchor.solarize, e.solarize, 0.0, 1.0, t * 0.12, rng);
    e.negative = mutate_linear(anchor.negative, e.negative, 0.0, 1.0, t * 0.1, rng);
    e.colourpass = mutate_linear(anchor.colourpass, e.colourpass, 0.0, 1.0, t * 0.12, rng);
    e.colourpass_hue = mutate_circular(
        anchor.colourpass_hue,
        e.colourpass_hue,
        -180.0,
        180.0,
        t * 35.0,
        rng,
    );
    e.colourpass_width = mutate_linear(
        anchor.colourpass_width,
        e.colourpass_width,
        0.0,
        1.0,
        t * 0.12,
        rng,
    );
    e.edge_amount = mutate_linear(anchor.edge_amount, e.edge_amount, 0.0, 1.0, t * 0.12, rng);
    e.edge_hue = mutate_circular(anchor.edge_hue, e.edge_hue, -180.0, 180.0, t * 35.0, rng);
    e.emboss = mutate_linear(anchor.emboss, e.emboss, 0.0, 1.0, t * 0.1, rng);
    e.emboss_angle = mutate_circular(
        anchor.emboss_angle,
        e.emboss_angle,
        -180.0,
        180.0,
        t * 35.0,
        rng,
    );
    e.halftone = mutate_linear(anchor.halftone, e.halftone, 0.0, 1.0, t * 0.12, rng);
    e.halftone_pitch = mutate_linear(
        anchor.halftone_pitch,
        e.halftone_pitch,
        0.0,
        1.0,
        t * 0.15,
        rng,
    );
    e.halftone_angle = mutate_circular(
        anchor.halftone_angle,
        e.halftone_angle,
        -180.0,
        180.0,
        t * 35.0,
        rng,
    );
    e.moire = mutate_linear(anchor.moire, e.moire, 0.0, 1.0, t * 0.1, rng);
    e.moire_freq = mutate_linear(anchor.moire_freq, e.moire_freq, 0.0, 1.0, t * 0.15, rng);
    e.row_smear = mutate_linear(anchor.row_smear, e.row_smear, 0.0, 1.0, t * 0.12, rng);
    e.bitcrush = mutate_linear(anchor.bitcrush, e.bitcrush, 0.0, 1.0, t * 0.1, rng);
    e.bitcrush_levels = mutate_log(
        anchor.bitcrush_levels,
        e.bitcrush_levels,
        2.0,
        16.0,
        t * 0.25,
        rng,
    )
    .round();
    e.bitcrush_dither = mutate_linear(
        anchor.bitcrush_dither,
        e.bitcrush_dither,
        0.0,
        1.0,
        t * 0.15,
        rng,
    );
    e.multi_grid_x =
        mutate_linear(anchor.multi_grid_x, e.multi_grid_x, 1.0, 8.0, t * 0.7, rng).round();
    e.multi_grid_y =
        mutate_linear(anchor.multi_grid_y, e.multi_grid_y, 1.0, 8.0, t * 0.7, rng).round();
    if include_optics {
        e.barrel = mutate_linear(anchor.barrel, e.barrel, -1.0, 1.0, t * 0.12, rng);
        e.chroma_aberration = mutate_linear(
            anchor.chroma_aberration,
            e.chroma_aberration,
            0.0,
            1.0,
            t * 0.12,
            rng,
        );
        e.anamorphic_streak = mutate_linear(
            anchor.anamorphic_streak,
            e.anamorphic_streak,
            0.0,
            1.0,
            t * 0.08,
            rng,
        );
    }
    // B8 key dressing, appended after every established draw so the
    // per-scope stream stays byte-stable; the border colour is a discrete
    // closed table and never rerolls.
    e.key_border = mutate_linear(anchor.key_border, e.key_border, 0.0, 1.0, t * 0.1, rng);
    e.key_shadow = mutate_linear(anchor.key_shadow, e.key_shadow, 0.0, 1.0, t * 0.1, rng);
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
const PROCEDURAL_TEMPORAL_LONG_EXPOSURE_AMOUNT: u64 = 0x4c45_5850_4f53_414d;
const PROCEDURAL_TEMPORAL_LONG_EXPOSURE_FRAMES: u64 = 0x4c45_5850_4f53_4652;

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
    linear!(
        value.long_exposure.amount,
        anchor.long_exposure.amount,
        0.0,
        1.0,
        0.2,
        PROCEDURAL_TEMPORAL_LONG_EXPOSURE_AMOUNT
    );
    {
        let mut rng = SplitMix64::new(domain_seed(
            seed,
            index,
            PROCEDURAL_TEMPORAL_LONG_EXPOSURE_FRAMES,
        ));
        value.long_exposure.shutter_frames = mutate_linear(
            f32::from(anchor.long_exposure.shutter_frames),
            f32::from(value.long_exposure.shutter_frames),
            2.0,
            24.0,
            temperature * 3.0,
            &mut rng,
        )
        .round() as u8;
    }
}

// B3 feedback-rig values live in field-isolated streams like the originals,
// so appending them cannot shift any earlier generator draw.
const PROCEDURAL_TEMPORAL_RIG_DOMAIN: u64 = 0x4233_5249_4700_0001;
const PROCEDURAL_RIG_OFFSET_X: u64 = 0x4f46_4653_4554_5800;
const PROCEDURAL_RIG_OFFSET_Y: u64 = 0x4f46_4653_4554_5900;
const PROCEDURAL_RIG_HUE: u64 = 0x4855_4552_4f54_0001;
const PROCEDURAL_RIG_SATURATION: u64 = 0x5341_5455_5241_0001;
const PROCEDURAL_RIG_GAIN_R: u64 = 0x4741_494e_5200_0001;
const PROCEDURAL_RIG_GAIN_G: u64 = 0x4741_494e_4700_0001;
const PROCEDURAL_RIG_GAIN_B: u64 = 0x4741_494e_4200_0001;
const PROCEDURAL_RIG_CHROMA: u64 = 0x4348_524f_4d41_0001;
const PROCEDURAL_RIG_BLUR: u64 = 0x424c_5552_0000_0001;
const PROCEDURAL_RIG_SHARPEN: u64 = 0x5348_4152_5045_4e00;
const PROCEDURAL_RIG_DRIVE: u64 = 0x4452_4956_4500_0001;
const PROCEDURAL_RIG_PIVOT: u64 = 0x5049_564f_5400_0001;
const PROCEDURAL_RIG_THRESHOLD: u64 = 0x5448_5245_5348_4f4c;
const PROCEDURAL_RIG_NOISE: u64 = 0x4e4f_4953_4500_0001;

/// Vary only the rig's bounded continuous values. Reflections, shape, edge,
/// and the two servo switches are discrete authored state and never change.
fn mutate_temporal_rig(
    anchor: &crate::patch::TemporalRigConfig,
    value: &mut crate::patch::TemporalRigConfig,
    temperature: f32,
    seed: u64,
    index: usize,
) {
    if temperature == 0.0 {
        return;
    }
    macro_rules! linear {
        ($field:ident, $min:expr, $max:expr, $scale:expr, $domain:expr) => {{
            let mut rng = SplitMix64::new(domain_seed(
                seed,
                index,
                PROCEDURAL_TEMPORAL_RIG_DOMAIN ^ $domain,
            ));
            value.$field = mutate_linear(
                anchor.$field,
                value.$field,
                $min,
                $max,
                temperature * $scale,
                &mut rng,
            );
        }};
    }
    linear!(offset_x, -0.5, 0.5, 0.05, PROCEDURAL_RIG_OFFSET_X);
    linear!(offset_y, -0.5, 0.5, 0.05, PROCEDURAL_RIG_OFFSET_Y);
    linear!(hue_rotate, -180.0, 180.0, 12.0, PROCEDURAL_RIG_HUE);
    linear!(saturation, 0.0, 2.0, 0.1, PROCEDURAL_RIG_SATURATION);
    linear!(gain_r, 0.0, 2.0, 0.08, PROCEDURAL_RIG_GAIN_R);
    linear!(gain_g, 0.0, 2.0, 0.08, PROCEDURAL_RIG_GAIN_G);
    linear!(gain_b, 0.0, 2.0, 0.08, PROCEDURAL_RIG_GAIN_B);
    linear!(chroma_displace, 0.0, 0.05, 0.004, PROCEDURAL_RIG_CHROMA);
    linear!(blur, 0.0, 1.0, 0.1, PROCEDURAL_RIG_BLUR);
    linear!(sharpen, 0.0, 2.0, 0.15, PROCEDURAL_RIG_SHARPEN);
    linear!(drive, 0.25, 4.0, 0.2, PROCEDURAL_RIG_DRIVE);
    linear!(pivot, 0.0, 1.0, 0.05, PROCEDURAL_RIG_PIVOT);
    linear!(threshold, 0.0, 1.0, 0.08, PROCEDURAL_RIG_THRESHOLD);
    linear!(noise, 0.0, 1.0, 0.08, PROCEDURAL_RIG_NOISE);
    *value = value.sanitized();
}

// B4 display-physics values live in field-isolated streams like the rig's,
// so appending them cannot shift any earlier generator draw. The interlace
// mode, the field-order fault, and the display model are discrete authored
// laws and never change.
const PROCEDURAL_DISPLAY_DOMAIN: u64 = 0x4234_4449_5350_0001;
/// B8 master melting edge: one domain, field-separated below.
const PROCEDURAL_MELT_DOMAIN: u64 = 0x4238_4d45_4c54_0001;
const PROCEDURAL_MELT_AMOUNT: u64 = 0x4d45_4c54_414d_5401;
const PROCEDURAL_MELT_WIDTH: u64 = 0x4d45_4c54_5749_4401;
const PROCEDURAL_MELT_HOLD: u64 = 0x4d45_4c54_484f_4c01;
const PROCEDURAL_MELT_SWIRL: u64 = 0x4d45_4c54_5357_4901;
const PROCEDURAL_MELT_CHROMA: u64 = 0x4d45_4c54_4348_5201;
const PROCEDURAL_MELT_CREEP: u64 = 0x4d45_4c54_4352_5001;
const PROCEDURAL_DISPLAY_IL_AMOUNT: u64 = 0x494c_414d_4f55_4e54;
const PROCEDURAL_DISPLAY_IL_TWITTER: u64 = 0x494c_5457_4954_5445;
const PROCEDURAL_DISPLAY_IL_JUDDER: u64 = 0x494c_4a55_4444_4552;
const PROCEDURAL_DISPLAY_PHOSPHOR: u64 = 0x5048_4f53_5048_4f52;
const PROCEDURAL_DISPLAY_PHOS_R: u64 = 0x5048_4f53_5200_0001;
const PROCEDURAL_DISPLAY_PHOS_G: u64 = 0x5048_4f53_4700_0001;
const PROCEDURAL_DISPLAY_PHOS_B: u64 = 0x5048_4f53_4200_0001;
const PROCEDURAL_DISPLAY_SCANLINES: u64 = 0x5343_414e_4c49_4e45;
const PROCEDURAL_DISPLAY_BEAM_WIDTH: u64 = 0x4245_414d_5749_4454;
const PROCEDURAL_DISPLAY_BEAM_SHAPE: u64 = 0x4245_414d_5348_4150;
const PROCEDURAL_DISPLAY_MASK_STRENGTH: u64 = 0x4d41_534b_5354_5245;
const PROCEDURAL_DISPLAY_MASK_DARK: u64 = 0x4d41_534b_4441_524b;
const PROCEDURAL_DISPLAY_BLOOM: u64 = 0x424c_4f4f_4d00_0001;
const PROCEDURAL_DISPLAY_BLOOM_RADIUS: u64 = 0x424c_4f4f_4d52_4144;
const PROCEDURAL_DISPLAY_HALATION: u64 = 0x4841_4c41_5449_4f4e;
const PROCEDURAL_DISPLAY_DEFOCUS: u64 = 0x4445_464f_4355_5300;
const PROCEDURAL_DISPLAY_SAG: u64 = 0x4856_5341_4700_0001;
/// B5 codec mosh: one domain, field-separated below. The recycle law is
/// discrete and never rerolls.
const PROCEDURAL_MOSH_DOMAIN: u64 = 0x4235_4d4f_5348_0001;
const PROCEDURAL_MOSH_AMOUNT: u64 = 0x4d4f_5348_414d_5401;
const PROCEDURAL_MOSH_KEY: u64 = 0x4d4f_5348_4b45_5901;
const PROCEDURAL_MOSH_HOLD: u64 = 0x4d4f_5348_484f_4c01;
const PROCEDURAL_MOSH_DROP: u64 = 0x4d4f_5348_4452_5001;
const PROCEDURAL_MOSH_SHUFFLE: u64 = 0x4d4f_5348_5348_5501;
const PROCEDURAL_MOSH_RATE: u64 = 0x4d4f_5348_5241_5401;
const PROCEDURAL_MOSH_BITRATE: u64 = 0x4d4f_5348_4249_5401;
const PROCEDURAL_MOSH_RESYNC: u64 = 0x4d4f_5348_5253_5901;
const PROCEDURAL_MOSH_WIPE: u64 = 0x4d4f_5348_5749_5001;
const PROCEDURAL_MOSH_SMEAR: u64 = 0x4d4f_5348_534d_5201;
const PROCEDURAL_MOSH_TRAIL: u64 = 0x4d4f_5348_5452_4c01;

/// Vary only the display stage's seventeen bounded continuous values.
/// B8 master melting edge: mutate the six continuous values in fresh
/// field-isolated domains. Everything is continuous, so nothing discrete
/// can reroll here.
fn mutate_master_melt(
    anchor: &crate::mixing_boundary::MeltParams,
    value: &mut crate::mixing_boundary::MeltParams,
    temperature: f32,
    seed: u64,
    index: usize,
) {
    if temperature == 0.0 {
        return;
    }
    macro_rules! linear {
        ($field:ident, $min:expr, $max:expr, $scale:expr, $domain:expr) => {{
            let mut rng =
                SplitMix64::new(domain_seed(seed, index, PROCEDURAL_MELT_DOMAIN ^ $domain));
            value.$field = mutate_linear(
                anchor.$field,
                value.$field,
                $min,
                $max,
                temperature * $scale,
                &mut rng,
            );
        }};
    }
    linear!(melt, 0.0, 2.0, 0.15, PROCEDURAL_MELT_AMOUNT);
    linear!(width, 0.0, 2.0, 0.1, PROCEDURAL_MELT_WIDTH);
    linear!(hold, 0.0, 1.5, 0.1, PROCEDURAL_MELT_HOLD);
    linear!(swirl, -1.0, 1.0, 0.15, PROCEDURAL_MELT_SWIRL);
    linear!(chroma, 0.0, 1.0, 0.1, PROCEDURAL_MELT_CHROMA);
    linear!(creep, 0.0, 1.0, 0.1, PROCEDURAL_MELT_CREEP);
    *value = value.sanitized();
}

/// B5 codec mosh: mutate every continuous codec and motion-wake value in fresh
/// field-isolated domains. The recycle law is discrete and never rerolls.
fn mutate_codec_mosh(
    anchor: &crate::codec_mosh::CodecMoshParams,
    value: &mut crate::codec_mosh::CodecMoshParams,
    temperature: f32,
    seed: u64,
    index: usize,
) {
    if temperature == 0.0 {
        return;
    }
    macro_rules! linear {
        ($field:ident, $scale:expr, $domain:expr) => {{
            let mut rng =
                SplitMix64::new(domain_seed(seed, index, PROCEDURAL_MOSH_DOMAIN ^ $domain));
            value.$field = mutate_linear(
                anchor.$field,
                value.$field,
                0.0,
                1.0,
                temperature * $scale,
                &mut rng,
            );
        }};
    }
    linear!(amount, 0.1, PROCEDURAL_MOSH_AMOUNT);
    linear!(key_removal, 0.08, PROCEDURAL_MOSH_KEY);
    linear!(hold, 0.1, PROCEDURAL_MOSH_HOLD);
    linear!(drop, 0.08, PROCEDURAL_MOSH_DROP);
    linear!(shuffle, 0.08, PROCEDURAL_MOSH_SHUFFLE);
    linear!(rate, 0.1, PROCEDURAL_MOSH_RATE);
    linear!(bitrate_starve, 0.1, PROCEDURAL_MOSH_BITRATE);
    linear!(resync, 0.08, PROCEDURAL_MOSH_RESYNC);
    linear!(wipe, 0.1, PROCEDURAL_MOSH_WIPE);
    linear!(smear, 0.1, PROCEDURAL_MOSH_SMEAR);
    linear!(trail, 0.1, PROCEDURAL_MOSH_TRAIL);
    *value = value.sanitized();
}

fn mutate_display_physics(
    anchor: &crate::display_physics::DisplayPhysicsParams,
    value: &mut crate::display_physics::DisplayPhysicsParams,
    temperature: f32,
    seed: u64,
    index: usize,
) {
    if temperature == 0.0 {
        return;
    }
    macro_rules! linear {
        ($field:ident, $min:expr, $max:expr, $scale:expr, $domain:expr) => {{
            let mut rng = SplitMix64::new(domain_seed(
                seed,
                index,
                PROCEDURAL_DISPLAY_DOMAIN ^ $domain,
            ));
            value.$field = mutate_linear(
                anchor.$field,
                value.$field,
                $min,
                $max,
                temperature * $scale,
                &mut rng,
            );
        }};
    }
    linear!(il_amount, 0.0, 1.0, 0.1, PROCEDURAL_DISPLAY_IL_AMOUNT);
    linear!(il_twitter, 0.0, 1.0, 0.1, PROCEDURAL_DISPLAY_IL_TWITTER);
    linear!(il_judder, 0.0, 1.0, 0.1, PROCEDURAL_DISPLAY_IL_JUDDER);
    linear!(phosphor, 0.0, 0.95, 0.1, PROCEDURAL_DISPLAY_PHOSPHOR);
    linear!(phos_r, 0.0, 1.0, 0.05, PROCEDURAL_DISPLAY_PHOS_R);
    linear!(phos_g, 0.0, 1.0, 0.05, PROCEDURAL_DISPLAY_PHOS_G);
    linear!(phos_b, 0.0, 1.0, 0.05, PROCEDURAL_DISPLAY_PHOS_B);
    linear!(scanlines, 0.0, 1.0, 0.1, PROCEDURAL_DISPLAY_SCANLINES);
    linear!(beam_width, 0.1, 3.0, 0.15, PROCEDURAL_DISPLAY_BEAM_WIDTH);
    linear!(beam_shape, 0.0, 1.0, 0.1, PROCEDURAL_DISPLAY_BEAM_SHAPE);
    linear!(
        mask_strength,
        0.0,
        1.0,
        0.1,
        PROCEDURAL_DISPLAY_MASK_STRENGTH
    );
    linear!(mask_dark, 0.0, 1.0, 0.1, PROCEDURAL_DISPLAY_MASK_DARK);
    linear!(bloom, 0.0, 1.0, 0.08, PROCEDURAL_DISPLAY_BLOOM);
    linear!(
        bloom_radius,
        0.0,
        1.0,
        0.08,
        PROCEDURAL_DISPLAY_BLOOM_RADIUS
    );
    linear!(halation, 0.0, 1.0, 0.08, PROCEDURAL_DISPLAY_HALATION);
    linear!(defocus, 0.0, 1.0, 0.08, PROCEDURAL_DISPLAY_DEFOCUS);
    linear!(sag, 0.0, 1.0, 0.08, PROCEDURAL_DISPLAY_SAG);
    *value = value.sanitized();
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
// B2 procedural field scalars: fresh domains, so every v7 stream that never
// carried them stays byte-stable.
const PROCEDURAL_MOTION_FIELD_SCALE: u64 = 0x4649_454c_4453_434c;
const PROCEDURAL_MOTION_FIELD_RATE: u64 = 0x4649_454c_4452_4154;
// B2 flow shaping, likewise in fresh domains.
const PROCEDURAL_MOTION_STRETCH: u64 = 0x5354_5245_5443_4800;
const PROCEDURAL_MOTION_EDGE_REPEL: u64 = 0x4544_4745_5245_5045;
const PROCEDURAL_MOTION_VECTOR_TRASH: u64 = 0x5645_4354_5452_4153;
const PROCEDURAL_MOTION_TRASH_BLOCK: u64 = 0x5452_4153_424c_4f43;

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
    linear!(
        value.procedural.scale,
        anchor.procedural.scale,
        0.0,
        1.0,
        0.2,
        PROCEDURAL_MOTION_FIELD_SCALE
    );
    linear!(
        value.procedural.rate,
        anchor.procedural.rate,
        -2.0,
        2.0,
        0.5,
        PROCEDURAL_MOTION_FIELD_RATE
    );
    linear!(
        value.shaping.stretch,
        anchor.shaping.stretch,
        0.0,
        1.0,
        0.15,
        PROCEDURAL_MOTION_STRETCH
    );
    linear!(
        value.shaping.edge_repel,
        anchor.shaping.edge_repel,
        0.0,
        1.0,
        0.15,
        PROCEDURAL_MOTION_EDGE_REPEL
    );
    linear!(
        value.shaping.vector_trash,
        anchor.shaping.vector_trash,
        0.0,
        1.0,
        0.1,
        PROCEDURAL_MOTION_VECTOR_TRASH
    );
    linear!(
        value.shaping.trash_block_size,
        anchor.shaping.trash_block_size,
        2.0,
        256.0,
        24.0,
        PROCEDURAL_MOTION_TRASH_BLOCK
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
    /// Displace collects its donor only while at least one gain is nonzero, so
    /// leaving the zero pair wakes a saved edge exactly like a mask amount.
    DisplaceAmounts {
        owner: ProceduralRackOwner,
        node_id: NodeId,
        prior_x: f32,
        prior_y: f32,
    },
    /// A Symmetry Field claims a saved image edge only while an armed source
    /// slot survives its own exact-bypass test, so a generated geometry value
    /// that leaves the bypass is the same class of wake as a mask amount. The
    /// whole continuous geometry travels in one variant: reverting half of it
    /// could leave a partially woken graph.
    SymmetryGeometry {
        owner: ProceduralRackOwner,
        node_id: NodeId,
        prior: SymmetryParams,
    },
    /// Residual collects both of its donors only while `mix` is nonzero, so
    /// leaving zero wakes two saved edges at once. `mix` is the whole bypass
    /// authority — `detail_gain` cannot wake anything — so restoring the prior
    /// mix returns both routes to dormant in one step.
    ResidualMix {
        owner: ProceduralRackOwner,
        node_id: NodeId,
        prior_mix: f32,
    },
    GroupMatteAmount {
        group_id: GroupId,
        prior: f32,
    },
}

/// The exact predicate `patch::collect_rack_dependencies` uses for a saved
/// Symmetry Field edge, factored out so generation's wake analysis and the
/// validator can never drift apart. A node whose armed slots are all clear
/// claims nothing however active its geometry becomes.
fn symmetry_claims_saved_image_edge(params: SymmetryParams) -> bool {
    !params.is_exact_bypass()
        && params
            .admitted_donor_taps()
            .iter()
            .any(std::option::Option::is_some)
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
            Self::DisplaceAmounts {
                owner,
                node_id,
                prior_x,
                prior_y,
            } => {
                let Some(node) =
                    saved_rack_mut(patch, owner).and_then(|rack| rack.get_mut(node_id))
                else {
                    return;
                };
                if let VisualNodeKind::Displace(params) = &mut node.kind {
                    params.amount_x = prior_x;
                    params.amount_y = prior_y;
                }
            }
            Self::SymmetryGeometry {
                owner,
                node_id,
                prior,
            } => {
                let Some(node) =
                    saved_rack_mut(patch, owner).and_then(|rack| rack.get_mut(node_id))
                else {
                    return;
                };
                if let VisualNodeKind::Symmetry(params) = &mut node.kind {
                    // Restore exactly what generation mutated. Routes, masks,
                    // seed, mode, and boundary were never touched, so they are
                    // deliberately left as they stand.
                    params.base_folds = prior.base_folds;
                    params.fold_offset = prior.fold_offset;
                    params.radial_phase_deg = prior.radial_phase_deg;
                    params.orbit_phase = prior.orbit_phase;
                    params.planar_axis_deg = prior.planar_axis_deg;
                    params.planar_phase = prior.planar_phase;
                    params.cell_skew = prior.cell_skew;
                    params.spiral_scale = prior.spiral_scale;
                    params.orbit_radius = prior.orbit_radius;
                    params.orbit_spin_deg = prior.orbit_spin_deg;
                    params.center = prior.center;
                    params.motion_gain = prior.motion_gain;
                    params.hue_span = prior.hue_span;
                }
            }
            Self::ResidualMix {
                owner,
                node_id,
                prior_mix,
            } => {
                let Some(node) =
                    saved_rack_mut(patch, owner).and_then(|rack| rack.get_mut(node_id))
                else {
                    return;
                };
                if let VisualNodeKind::Residual(params) = &mut node.kind {
                    params.mix = prior_mix;
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
        let prior_displace = match node.kind {
            VisualNodeKind::Displace(params) => Some(params),
            _ => None,
        };
        let prior_symmetry = match node.kind {
            VisualNodeKind::Symmetry(params) => Some(params),
            _ => None,
        };
        let prior_residual = match node.kind {
            VisualNodeKind::Residual(params) => Some(params),
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
            (VisualNodeKind::Displace(anchor), VisualNodeKind::Displace(value)) => {
                // Donor route and boundary law are stable authored topology and
                // are preserved exactly. Each node draws from its own domain, so
                // this arm cannot perturb any older generated stream.
                value.amount_x = mutate_linear(
                    anchor.amount_x,
                    value.amount_x,
                    -1.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.amount_y = mutate_linear(
                    anchor.amount_y,
                    value.amount_y,
                    -1.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
            }
            (VisualNodeKind::Symmetry(anchor), VisualNodeKind::Symmetry(value)) => {
                // Declared continuous controls only. The four routes, both
                // masks, the authored seed, the mode, and the boundary law are
                // stable authored topology, so generation can never rewrite the
                // 32-record sector table. Each node draws from its own domain,
                // so this arm cannot perturb any older generated stream.
                value.base_folds = mutate_linear(
                    anchor.base_folds,
                    value.base_folds,
                    1.0,
                    32.0,
                    temperature * 2.0,
                    &mut rng,
                );
                value.fold_offset = mutate_linear(
                    anchor.fold_offset,
                    value.fold_offset,
                    -32.0,
                    32.0,
                    temperature * 1.5,
                    &mut rng,
                );
                value.radial_phase_deg = mutate_circular(
                    anchor.radial_phase_deg,
                    value.radial_phase_deg,
                    -180.0,
                    180.0,
                    temperature * 40.0,
                    &mut rng,
                );
                value.orbit_phase = mutate_linear(
                    anchor.orbit_phase,
                    value.orbit_phase,
                    -1.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.planar_axis_deg = mutate_circular(
                    anchor.planar_axis_deg,
                    value.planar_axis_deg,
                    -180.0,
                    180.0,
                    temperature * 40.0,
                    &mut rng,
                );
                value.planar_phase = mutate_linear(
                    anchor.planar_phase,
                    value.planar_phase,
                    -4.0,
                    4.0,
                    temperature * 0.4,
                    &mut rng,
                );
                value.cell_skew = mutate_linear(
                    anchor.cell_skew,
                    value.cell_skew,
                    -1.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.spiral_scale = mutate_linear(
                    anchor.spiral_scale,
                    value.spiral_scale,
                    -1.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.orbit_radius = mutate_linear(
                    anchor.orbit_radius,
                    value.orbit_radius,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.orbit_spin_deg = mutate_circular(
                    anchor.orbit_spin_deg,
                    value.orbit_spin_deg,
                    -180.0,
                    180.0,
                    temperature * 40.0,
                    &mut rng,
                );
                value.motion_gain = mutate_linear(
                    anchor.motion_gain,
                    value.motion_gain,
                    -1.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.hue_span = mutate_linear(
                    anchor.hue_span,
                    value.hue_span,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.center[0] = mutate_linear(
                    anchor.center[0],
                    value.center[0],
                    -1.0,
                    2.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.center[1] = mutate_linear(
                    anchor.center[1],
                    value.center[1],
                    -1.0,
                    2.0,
                    temperature * 0.2,
                    &mut rng,
                );
            }
            (VisualNodeKind::Residual(anchor), VisualNodeKind::Residual(value)) => {
                // Both donor routes, the block vocabulary, the quantization law
                // and the quantization seed are stable authored topology and are
                // preserved exactly. Each node draws from its own domain, so
                // this arm cannot perturb any older generated stream.
                value.mix =
                    mutate_linear(anchor.mix, value.mix, 0.0, 1.0, temperature * 0.2, &mut rng);
                value.detail_gain = mutate_linear(
                    anchor.detail_gain,
                    value.detail_gain,
                    0.0,
                    4.0,
                    temperature * 0.2,
                    &mut rng,
                );
            }
            (VisualNodeKind::LegacyCanonical | VisualNodeKind::LegacyTemporal, _)
            | (_, VisualNodeKind::LegacyCanonical | VisualNodeKind::LegacyTemporal) => {}
            (VisualNodeKind::ScanProcessor(anchor), VisualNodeKind::ScanProcessor(value)) => {
                // The two geometry counts and the two reversals are stable
                // authored topology for generation's purposes and are
                // preserved exactly. The fifteen continuous controls mutate
                // anchor-relatively, and each node draws from its own domain,
                // so this arm cannot perturb any older generated stream.
                value.amount = mutate_linear(
                    anchor.amount,
                    value.amount,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.ribbon_width = mutate_linear(
                    anchor.ribbon_width,
                    value.ribbon_width,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.velocity_mix = mutate_linear(
                    anchor.velocity_mix,
                    value.velocity_mix,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.tilt_x = mutate_linear(
                    anchor.tilt_x,
                    value.tilt_x,
                    -1.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.tilt_y = mutate_linear(
                    anchor.tilt_y,
                    value.tilt_y,
                    -1.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.perspective = mutate_linear(
                    anchor.perspective,
                    value.perspective,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.s_curve = mutate_linear(
                    anchor.s_curve,
                    value.s_curve,
                    -1.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.skew = mutate_linear(
                    anchor.skew,
                    value.skew,
                    -1.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.collapse = mutate_linear(
                    anchor.collapse,
                    value.collapse,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.osc_amount = mutate_linear(
                    anchor.osc_amount,
                    value.osc_amount,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.osc_freq = mutate_linear(
                    anchor.osc_freq,
                    value.osc_freq,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.osc_lock = mutate_linear(
                    anchor.osc_lock,
                    value.osc_lock,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.lissajous = mutate_linear(
                    anchor.lissajous,
                    value.lissajous,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.mono = mutate_linear(
                    anchor.mono,
                    value.mono,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.hue =
                    mutate_linear(anchor.hue, value.hue, 0.0, 1.0, temperature * 0.2, &mut rng);
            }
            // B6: the corruption trio's continuous values mutate
            // anchor-relatively in each node's own domain; the avalanche's
            // predictor axis is a discrete law generation preserves exactly.
            (VisualNodeKind::BlockDct(anchor), VisualNodeKind::BlockDct(value)) => {
                value.amount = mutate_linear(
                    anchor.amount,
                    value.amount,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.quantize = mutate_linear(
                    anchor.quantize,
                    value.quantize,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.hf_penalty = mutate_linear(
                    anchor.hf_penalty,
                    value.hf_penalty,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.chroma_crush = mutate_linear(
                    anchor.chroma_crush,
                    value.chroma_crush,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.block = mutate_linear(
                    anchor.block,
                    value.block,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
            }
            (VisualNodeKind::PixelSort(anchor), VisualNodeKind::PixelSort(value)) => {
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
            }
            (VisualNodeKind::Avalanche(anchor), VisualNodeKind::Avalanche(value)) => {
                value.amount = mutate_linear(
                    anchor.amount,
                    value.amount,
                    0.0,
                    1.0,
                    temperature * 0.2,
                    &mut rng,
                );
                value.run =
                    mutate_linear(anchor.run, value.run, 0.0, 1.0, temperature * 0.2, &mut rng);
            }
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
        let current_displace = match node.kind {
            VisualNodeKind::Displace(params) => Some(params),
            _ => None,
        };
        let current_symmetry = match node.kind {
            VisualNodeKind::Symmetry(params) => Some(params),
            _ => None,
        };
        let current_residual = match node.kind {
            VisualNodeKind::Residual(params) => Some(params),
            _ => None,
        };
        let route_effect_active = current_image_amount.is_some_and(|value| value > 0.0)
            || current_displace.is_some_and(|params| !params.is_exact_bypass())
            || current_residual.is_some_and(|params| !params.is_exact_bypass())
            || current_symmetry.is_some_and(symmetry_claims_saved_image_edge);
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
        if node.wet > 0.0
            && prior_displace.is_some_and(DisplaceParams::is_exact_bypass)
            && current_displace.is_some_and(|params| !params.is_exact_bypass())
        {
            let prior = prior_displace.unwrap_or_default();
            edge_fallbacks.push(CreativeEdgeFallback::DisplaceAmounts {
                owner,
                node_id,
                prior_x: prior.amount_x,
                prior_y: prior.amount_y,
            });
        }
        // A Symmetry Field wake is recorded only when the claim answer itself
        // changes, so a piece is never reverted further than the graph needs.
        if node.wet > 0.0
            && prior_symmetry.is_some_and(|params| !symmetry_claims_saved_image_edge(params))
            && current_symmetry.is_some_and(symmetry_claims_saved_image_edge)
        {
            edge_fallbacks.push(CreativeEdgeFallback::SymmetryGeometry {
                owner,
                node_id,
                prior: prior_symmetry.unwrap_or_default(),
            });
        }
        if node.wet > 0.0
            && prior_residual.is_some_and(ResidualParams::is_exact_bypass)
            && current_residual.is_some_and(|params| !params.is_exact_bypass())
        {
            let prior = prior_residual.unwrap_or_default();
            edge_fallbacks.push(CreativeEdgeFallback::ResidualMix {
                owner,
                node_id,
                prior_mix: prior.mix,
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

fn roundtrip_generated_patch(patch: &PatchState) -> Result<PatchState, String> {
    let yaml = serde_yaml::to_string(patch)
        .map_err(|error| format!("serialize generated creative graph: {error}"))?;
    serde_yaml::from_str::<PatchState>(&yaml)
        .map_err(|error| format!("validate generated creative graph: {error}"))
}

#[cfg(test)]
fn validate_generated_patch(patch: &PatchState) -> Result<(), String> {
    roundtrip_generated_patch(patch).map(drop)
}

fn measure_layer_bypasses(patch: &PatchState) -> Vec<LayerBypassMeasurement> {
    patch
        .layers
        .iter()
        .enumerate()
        .map(|(layer_index, layer)| LayerBypassMeasurement {
            layer_index,
            bypass_master_fx: layer.bypass_master_fx,
            bypass_temporal_fx: layer.bypass_temporal_fx,
        })
        .collect()
}

fn retain_valid_creative_edge_values(
    patch: &mut PatchState,
    edge_fallbacks: Vec<CreativeEdgeFallback>,
) -> Result<PatchState, String> {
    if let Ok(roundtripped) = roundtrip_generated_patch(patch) {
        return Ok(roundtripped);
    }
    for fallback in edge_fallbacks {
        fallback.restore(patch);
        if let Ok(roundtripped) = roundtrip_generated_patch(patch) {
            return Ok(roundtripped);
        }
    }
    roundtrip_generated_patch(patch)
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
            // B7 generators are the first sources with perfect offline
            // reconstruction: the whole identity is the layer's own config,
            // so they are verified with no bytes and no black policy, and
            // they never trip the --allow-black-sources gate.
            Ok(ResolvedVisualSource::PatternSynth) => sources.push(ManifestSource {
                role: "layer".into(),
                layer_index: Some(index),
                logical_name: crate::layers::PATTERN_SOURCE_PATH.to_string(),
                kind: "pattern_synth".into(),
                byte_len: None,
                sha256: None,
                offline_policy: Some("reconstructed".into()),
                verified: true,
            }),
            Ok(ResolvedVisualSource::TextPage) => sources.push(ManifestSource {
                role: "layer".into(),
                layer_index: Some(index),
                logical_name: crate::layers::TEXT_PAGE_SOURCE_PATH.to_string(),
                kind: "text_page".into(),
                byte_len: None,
                sha256: None,
                offline_policy: Some("reconstructed".into()),
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
            // B7 generators keep their sentinels: the identity is the
            // layer's own config, which the piece carries verbatim.
            "pattern_synth" => {
                layer.filename = "Pattern Synth".to_string();
                layer.source_path = crate::layers::PATTERN_SOURCE_PATH.to_string();
            }
            "text_page" => {
                layer.filename = "Text Page".to_string();
                layer.source_path = crate::layers::TEXT_PAGE_SOURCE_PATH.to_string();
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
                .strip_prefix(crate::layers::SPOUT_SOURCE_PREFIX)
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
                .strip_prefix(crate::layers::SPOUT_SOURCE_PREFIX)
                .map(|name| format!("{}{name}", crate::layers::SPOUT_SOURCE_PREFIX))
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
        mutate_small_effects(
            &normalized.master,
            &mut patch.master,
            temperature,
            config.seed,
            index,
            0x4d41_5354_4552,
            true,
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
            mutate_small_effects(
                &anchor_layer.effects,
                &mut layer.effects,
                temperature,
                config.seed,
                index,
                0x4c41_5945_5200 ^ layer_index as u64,
                false,
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
        if let (Some(anchor_temporal), Some(temporal)) =
            (normalized.temporal.as_ref(), patch.temporal.as_mut())
        {
            mutate_temporal_rig(
                &anchor_temporal.rig,
                &mut temporal.rig,
                temperature,
                config.seed,
                index,
            );
            mutate_display_physics(
                &anchor_temporal.display,
                &mut temporal.display,
                temperature,
                config.seed,
                index,
            );
            mutate_master_melt(
                &anchor_temporal.melt,
                &mut temporal.melt,
                temperature,
                config.seed,
                index,
            );
            mutate_codec_mosh(
                &anchor_temporal.mosh,
                &mut temporal.mosh,
                temperature,
                config.seed,
                index,
            );
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
        // Receipts must describe what the emitted YAML means when the program
        // reads it, not merely the pre-serialization Rust value. Keep the
        // final piece, its canonical hash, and its measurements on that one
        // deserialized truth. The validation round trip is also returned here
        // so the serial seam runs once, even when a creative fallback lands.
        patch = retain_valid_creative_edge_values(&mut patch, creative_edge_fallbacks)?;

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
            layer_bypass_states: measure_layer_bypasses(&patch),
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

pub(crate) fn sync_parent(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if let Ok(directory) = fs::File::open(parent) {
            directory.sync_all()?;
        }
    }
    Ok(())
}

#[cfg(windows)]
pub(crate) fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
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
pub(crate) fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
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
pub(crate) fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
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
pub(crate) fn rename_noreplace(_source: &Path, _destination: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace rename is not implemented on this Unix target",
    ))
}

#[cfg(not(any(windows, unix)))]
pub(crate) fn rename_noreplace(_source: &Path, _destination: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace rename is not implemented on this platform",
    ))
}

pub(crate) fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
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

    #[test]
    fn small_effects_generator_mutates_in_a_fresh_domain_and_respects_scope() {
        let anchor = EffectsConfig::default();
        let yaml = |config: &EffectsConfig| serde_yaml::to_string(config).unwrap();

        // Temperature zero is a strict no-op.
        let mut frozen = anchor.clone();
        mutate_small_effects(&anchor, &mut frozen, 0.0, 7, 3, 0x4d41_5354_4552, true);
        assert_eq!(yaml(&frozen), yaml(&anchor));

        // The mutation is deterministic per seed/index/scope and bounded,
        // integer counts round, and the discrete negative mode never moves.
        let mut a = anchor.clone();
        let mut b = anchor.clone();
        mutate_small_effects(&anchor, &mut a, 1.5, 7, 3, 0x4d41_5354_4552, true);
        mutate_small_effects(&anchor, &mut b, 1.5, 7, 3, 0x4d41_5354_4552, true);
        assert_eq!(yaml(&a), yaml(&b));
        assert!((0.0..=1.0).contains(&a.contour));
        assert!((2.0..=40.0).contains(&a.contour_bands));
        assert_eq!(a.contour_bands.fract(), 0.0);
        assert!((2.0..=16.0).contains(&a.bitcrush_levels));
        assert_eq!(a.bitcrush_levels.fract(), 0.0);
        assert!((1.0..=8.0).contains(&a.multi_grid_x));
        assert_eq!(a.multi_grid_x.fract(), 0.0);
        assert!((-180.0..=180.0).contains(&a.colourpass_hue));
        assert!((-1.0..=1.0).contains(&a.barrel));
        assert_eq!(a.negative_mode, 0);

        // A different scope domain draws a different sequence, and the layer
        // form never touches the three master-only optics.
        let mut other_scope = anchor.clone();
        mutate_small_effects(
            &anchor,
            &mut other_scope,
            1.5,
            7,
            3,
            0x4c41_5945_5200,
            false,
        );
        assert_ne!(yaml(&other_scope), yaml(&a));
        assert_eq!(other_scope.barrel, 0.0);
        assert_eq!(other_scope.chroma_aberration, 0.0);
        assert_eq!(other_scope.anamorphic_streak, 0.0);
    }

    #[test]
    fn dice_the_generator_and_modulation_all_preserve_the_collider_block_exactly() {
        use crate::patch::{
            FieldColliderConfig, FieldColliderModeConfig, MotionBoundaryModeConfig, MotionConfig,
            MotionDonorConfig,
        };
        use crate::performance::SavedLayerPosition;

        let authored = FieldColliderConfig {
            enabled: true,
            mode: FieldColliderModeConfig::Curl,
            boundary: MotionBoundaryModeConfig::Mirror,
            input_a: MotionDonorConfig::Selected {
                saved_position: SavedLayerPosition::new(1).unwrap(),
            },
            input_b: MotionDonorConfig::Selected {
                saved_position: SavedLayerPosition::new(3).unwrap(),
            },
            ..FieldColliderConfig::default()
        };
        let anchor = MotionConfig {
            collider: authored,
            ..MotionConfig::default()
        };

        // Version 1 adds no collider-only continuous control, so the bounded
        // generator has nothing to move. Every temperature, every seed, every
        // owner domain, with and without Faraday included, must leave the block
        // bit-identical — including both saved donor positions.
        for temperature in [0.0_f32, 0.25, 0.5, 1.0] {
            for seed in [0_u64, 1, 0xDEAD_BEEF, u64::MAX] {
                for include_faraday in [false, true] {
                    let mut mutated = anchor;
                    mutate_motion_config(
                        &anchor,
                        &mut mutated,
                        temperature,
                        seed,
                        3,
                        0x5151_5151,
                        include_faraday,
                    );
                    assert_eq!(
                        mutated.collider, authored,
                        "the generator moved the collider block at t={temperature} seed={seed}"
                    );
                }
            }
        }

        // Modulation exposes no collider address at all: the block is closed
        // authored topology, and `target_range` must not recognise any of it.
        for name in [
            "collider",
            "collider_mode",
            "collider_enabled",
            "collider_boundary",
            "collider_input_a",
            "collider_input_b",
            "layer1_collider",
            "layer1_collider_mode",
            "layer1_collider_enabled",
            "layer1_collider_boundary",
        ] {
            assert!(
                crate::modulation::target_range(name).is_none(),
                "modulation must expose no address for {name}"
            );
        }
    }
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
                delivery: Default::default(),
                opacity: 1.0,
                mosh_send: 1.0,
                blend_mode: "normal".to_string(),
                speed: 1.0,
                fps: 30.0,
                paused: false,
                visible: true,
                bypass_master_fx: false,
                bypass_temporal_fx: false,
                reroll_on_loop: false,
                effects: EffectsConfig::default(),
                transform: SpatialTransform::default(),
                motion: None,
                rack: None,
                clip_slots,
                active_clip_slot: Some(crate::performance::ClipSlotId::LEGACY),
                matte: crate::image_routing::LayerMatteConfig::default(),
                pattern: None,
                text_page: None,
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
            snapshot_bank: None,
            scenes: crate::performance::Scenes::default(),
            autopilot: crate::performance::AutopilotPlan::default(),
            gesture_track: None,
            gesture_canvas: None,
            studies: Vec::new(),
            performance_take: None,
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
    fn temporal_rig_generation_is_domain_isolated_and_preserves_discrete_laws() {
        use crate::patch::{FeedbackShapeConfig, MotionBoundaryModeConfig, TemporalRigConfig};
        let authored = TemporalRigConfig {
            reflect_x: true,
            shape: FeedbackShapeConfig::Soft,
            edge: MotionBoundaryModeConfig::Wrap,
            servo: true,
            servo_defeated: true,
            drive: 2.0,
            ..TemporalRigConfig::default()
        };
        let mut left = authored;
        let mut right = authored;
        mutate_temporal_rig(&authored, &mut left, 2.0, 77, 3);
        mutate_temporal_rig(&authored, &mut right, 2.0, 77, 3);
        assert_eq!(left, right, "rig mutation replays deterministically");
        assert_ne!(left.drive, authored.drive, "temperature moves the values");
        // Discrete laws never change.
        assert!(left.reflect_x);
        assert_eq!(left.shape, FeedbackShapeConfig::Soft);
        assert_eq!(left.edge, MotionBoundaryModeConfig::Wrap);
        assert!(left.servo && left.servo_defeated);
        // Every mutated value stays inside its authored bound.
        assert!((-0.5..=0.5).contains(&left.offset_x));
        assert!((-180.0..=180.0).contains(&left.hue_rotate));
        assert!((0.0..=2.0).contains(&left.saturation));
        assert!((0.0..=0.05).contains(&left.chroma_displace));
        assert!((0.25..=4.0).contains(&left.drive));
        // Zero temperature is byte-exact.
        let mut untouched = authored;
        mutate_temporal_rig(&authored, &mut untouched, 0.0, 77, 3);
        assert_eq!(untouched, authored);
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
        assert!((0.0..=1.0).contains(&left.long_exposure.amount));
        assert!((2..=24).contains(&left.long_exposure.shutter_frames));
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
    fn display_physics_generation_is_deterministic_bounded_and_preserves_discrete_laws() {
        use crate::display_physics::{DisplayModel, DisplayPhysicsParams, InterlaceMode};
        let authored = DisplayPhysicsParams {
            il_amount: 0.3,
            il_mode: InterlaceMode::Blend,
            il_order: true,
            phosphor: 0.5,
            model: DisplayModel::ApertureGrille,
            scanlines: 0.4,
            ..DisplayPhysicsParams::default()
        };
        let mut left = authored;
        let mut right = authored;
        mutate_display_physics(&authored, &mut left, 2.0, 77, 3);
        mutate_display_physics(&authored, &mut right, 2.0, 77, 3);
        assert_eq!(left, right, "display mutation replays deterministically");
        assert_ne!(
            left.il_amount, authored.il_amount,
            "temperature moves the values"
        );
        // Discrete laws never change.
        assert_eq!(left.il_mode, InterlaceMode::Blend);
        assert!(left.il_order);
        assert_eq!(left.model, DisplayModel::ApertureGrille);
        // Every mutated value stays inside its declared range.
        assert!((0.0..=1.0).contains(&left.il_amount));
        assert!((0.0..=0.95).contains(&left.phosphor));
        assert!((0.1..=3.0).contains(&left.beam_width));
        assert!((0.0..=1.0).contains(&left.sag));
        // Zero temperature is byte-exact.
        let mut untouched = authored;
        mutate_display_physics(&authored, &mut untouched, 0.0, 77, 3);
        assert_eq!(untouched, authored);
    }

    #[test]
    fn the_generator_preserves_the_sync_latch_exactly() {
        use crate::sync_latch::SyncLatchParams;
        // B14 adds no generator-mutated value: the whole block, switch and
        // controls alike, is preserved verbatim. The four calls below are the
        // entire temporal mutation the generation walk performs, so running
        // them and finding `sync` untouched is the preservation proof.
        let authored = crate::patch::TemporalConfig {
            sync: SyncLatchParams {
                amount: 0.6,
                rate: 0.4,
                spread: 0.8,
                bias: -0.3,
                latched: true,
            },
            ..crate::patch::TemporalConfig::default()
        };
        let mut mutated = authored.clone();
        mutate_temporal_rig(&authored.rig, &mut mutated.rig, 2.0, 77, 3);
        mutate_display_physics(&authored.display, &mut mutated.display, 2.0, 77, 3);
        mutate_master_melt(&authored.melt, &mut mutated.melt, 2.0, 77, 3);
        mutate_codec_mosh(&authored.mosh, &mut mutated.mosh, 2.0, 77, 3);
        assert_eq!(
            mutated.sync, authored.sync,
            "the generator must preserve the sync latch exactly"
        );

        // And no mutator for it exists at all, so a later edit cannot quietly
        // start moving it without also moving GENERATOR_VERSION.
        // The needle is built at runtime: a literal would match this very
        // assertion and the audit would report itself.
        let needle = format!("mutate_{}", "sync_latch");
        let source = include_str!("procedural.rs");
        assert!(
            !source.contains(&needle),
            "B14 declares no generator mutation; adding one needs a version bump"
        );
    }

    #[test]
    fn codec_mosh_generation_is_deterministic_bounded_and_preserves_the_recycle_law() {
        use crate::codec_mosh::CodecMoshParams;
        let authored = CodecMoshParams {
            amount: 0.4,
            key_removal: 0.9,
            hold: 0.3,
            resync: 0.2,
            wipe: 0.35,
            smear: 0.55,
            trail: 0.7,
            recycle: true,
            ..CodecMoshParams::default()
        };
        let mut left = authored;
        let mut right = authored;
        mutate_codec_mosh(&authored, &mut left, 2.0, 77, 3);
        mutate_codec_mosh(&authored, &mut right, 2.0, 77, 3);
        assert_eq!(left, right, "mosh mutation replays deterministically");
        assert_ne!(left.amount, authored.amount, "temperature moves the values");
        // The discrete recycle law never changes.
        assert!(left.recycle);
        // Every mutated value stays inside the unit interval.
        for value in [
            left.amount,
            left.key_removal,
            left.hold,
            left.drop,
            left.shuffle,
            left.rate,
            left.bitrate_starve,
            left.resync,
            left.wipe,
            left.smear,
            left.trail,
        ] {
            assert!((0.0..=1.0).contains(&value));
        }
        // Zero temperature is byte-exact.
        let mut untouched = authored;
        mutate_codec_mosh(&authored, &mut untouched, 0.0, 77, 3);
        assert_eq!(untouched, authored);
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
                (VisualNodeKind::Displace(expected), VisualNodeKind::Displace(actual)) => {
                    assert_eq!(actual.tap, expected.tap);
                    assert_eq!(actual.boundary, expected.boundary);
                }
                (VisualNodeKind::Symmetry(expected), VisualNodeKind::Symmetry(actual)) => {
                    assert_eq!(actual.donors, expected.donors);
                    assert_eq!(actual.motion, expected.motion);
                    assert_eq!(actual.source_mask, expected.source_mask);
                    assert_eq!(actual.motion_mask, expected.motion_mask);
                    assert_eq!(actual.seed, expected.seed);
                    assert_eq!(actual.mode, expected.mode);
                    assert_eq!(actual.boundary, expected.boundary);
                }
                (VisualNodeKind::Residual(expected), VisualNodeKind::Residual(actual)) => {
                    assert_eq!(actual.routes(), expected.routes());
                    assert_eq!(actual.block, expected.block);
                    assert_eq!(actual.quantization, expected.quantization);
                    assert_eq!(actual.seed, expected.seed);
                    assert_eq!(actual.algorithm_version, expected.algorithm_version);
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
        assert_eq!(first[0].preflight, second[0].preflight);
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
    fn preflight_v2_measures_resolved_independent_bypasses_from_emitted_yaml() {
        let anchor: PatchState = serde_yaml::from_str(
            r#"
master: {}
layers:
  - filename: temporal-only.mp4
    bypass_master_fx: false
    bypass_temporal_fx: true
    effects: {}
  - filename: master-only.mp4
    bypass_master_fx: true
    bypass_temporal_fx: false
    effects: {}
  - filename: omitted.mp4
    effects: {}
"#,
        )
        .unwrap();
        assert!(!anchor.layers[2].bypass_master_fx);
        assert!(!anchor.layers[2].bypass_temporal_fx);

        let piece = generate(
            &anchor,
            &GenerationConfig {
                seed: 0x4259_5041_5353,
                count: 1,
                temperature: 0.0,
                allow_black_sources: false,
            },
        )
        .unwrap()
        .remove(0);

        assert_eq!(piece.manifest.generator_version, "14");
        assert_eq!(piece.preflight.schema_version, 2);
        assert_eq!(
            piece.preflight.layer_bypass_states,
            [
                LayerBypassMeasurement {
                    layer_index: 0,
                    bypass_master_fx: false,
                    bypass_temporal_fx: true,
                },
                LayerBypassMeasurement {
                    layer_index: 1,
                    bypass_master_fx: true,
                    bypass_temporal_fx: false,
                },
                LayerBypassMeasurement {
                    layer_index: 2,
                    bypass_master_fx: false,
                    bypass_temporal_fx: false,
                },
            ]
        );

        let emitted_yaml = serde_yaml::to_string(&piece.patch).unwrap();
        let emitted: PatchState = serde_yaml::from_str(&emitted_yaml).unwrap();
        assert_eq!(
            measure_layer_bypasses(&emitted),
            piece.preflight.layer_bypass_states,
            "the receipt must be reconstructed from the exact emitted YAML truth"
        );
        let receipt_json = serde_json::to_string(&piece.preflight).unwrap();
        assert!(receipt_json.contains("\"bypass_temporal_fx\":false"));
        assert!(receipt_json.contains("\"bypass_temporal_fx\":true"));

        let mut legacy = serde_json::to_value(&piece.preflight).unwrap();
        legacy["schema_version"] = serde_json::json!(1);
        legacy
            .as_object_mut()
            .unwrap()
            .remove("layer_bypass_states");
        let legacy: PreflightReceipt = serde_json::from_value(legacy).unwrap();
        assert_eq!(legacy.schema_version, 1);
        assert!(legacy.layer_bypass_states.is_empty());
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
        assert_eq!(piece.manifest.generator_version, GENERATOR_VERSION);
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

        assert_eq!(advanced_piece.manifest.generator_version, GENERATOR_VERSION);
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
        source.layers[0].bypass_temporal_fx = true;
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
            assert!(
                layer.bypass_temporal_fx,
                "procedural mutation must preserve independent Temporal bypass authoring"
            );
            assert_eq!(
                piece.preflight.layer_bypass_states,
                [LayerBypassMeasurement {
                    layer_index: 0,
                    bypass_master_fx: true,
                    bypass_temporal_fx: true,
                }],
                "every receipt must report the final preserved routing truth"
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
        let written_patch: PatchState =
            serde_yaml::from_slice(&fs::read(created[0].join("patch.yaml")).unwrap()).unwrap();
        let written_preflight: PreflightReceipt =
            serde_json::from_slice(&fs::read(created[0].join("preflight.json")).unwrap()).unwrap();
        assert_eq!(written_preflight.schema_version, PREFLIGHT_SCHEMA_VERSION);
        assert_eq!(
            written_preflight.layer_bypass_states,
            measure_layer_bypasses(&written_patch)
        );
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

    #[test]
    fn generated_displace_moves_gains_only_and_wakes_its_edge_transactionally() {
        use crate::visual_rack::{
            DisplaceBoundary, DisplaceParams, EdgeTiming, SavedImageSource, SavedImageTap,
        };

        let authored = DisplaceParams {
            tap: SavedImageTap {
                source: SavedImageSource::CleanProgram,
                timing: EdgeTiming::PreviousFrame,
            },
            amount_x: 0.3,
            amount_y: -0.3,
            boundary: DisplaceBoundary::Hold,
        };
        let mut anchor = VisualRack::empty();
        let node_id = anchor.push(VisualNodeKind::Displace(authored)).unwrap();
        let params_of = |rack: &VisualRack| match rack.get(node_id).unwrap().kind {
            VisualNodeKind::Displace(params) => params,
            _ => panic!("displace node"),
        };

        let mut generated = anchor.clone();
        let mut fallbacks = Vec::new();
        mutate_saved_rack_values(
            &anchor,
            &mut generated,
            1.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut fallbacks,
        );
        let after = params_of(&generated);
        assert_eq!(after.tap, authored.tap, "generation never reroutes a donor");
        assert_eq!(after.boundary, authored.boundary);
        assert!(after.amount_x != authored.amount_x || after.amount_y != authored.amount_y);
        assert!((-1.0..=1.0).contains(&after.amount_x));
        assert!((-1.0..=1.0).contains(&after.amount_y));

        // Deterministic: identical seed and index reproduce identical gains.
        let mut repeated = anchor.clone();
        mutate_saved_rack_values(
            &anchor,
            &mut repeated,
            1.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        assert_eq!(params_of(&repeated), after);

        // Zero temperature is an exact no-op.
        let mut untouched = anchor.clone();
        mutate_saved_rack_values(
            &anchor,
            &mut untouched,
            0.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        assert_eq!(params_of(&untouched), authored);

        // A node that starts at zero gain has a dormant edge. Waking it must
        // record a transactional fallback so the caller can restore the
        // dormant state if the woken graph fails preflight.
        let mut dormant_anchor = VisualRack::empty();
        let dormant_id = dormant_anchor
            .push(VisualNodeKind::Displace(DisplaceParams {
                tap: SavedImageTap {
                    source: SavedImageSource::OneBelow,
                    timing: EdgeTiming::CurrentFrame,
                },
                ..DisplaceParams::default()
            }))
            .unwrap();
        let mut woken = dormant_anchor.clone();
        let mut fallbacks = Vec::new();
        mutate_saved_rack_values(
            &dormant_anchor,
            &mut woken,
            1.0,
            0x1234,
            1,
            ProceduralRackOwner::Master,
            &mut fallbacks,
        );
        let VisualNodeKind::Displace(params) = woken.get(dormant_id).unwrap().kind else {
            panic!("displace node")
        };
        if !params.is_exact_bypass() {
            assert!(
                fallbacks.iter().any(|fallback| matches!(
                    fallback,
                    CreativeEdgeFallback::DisplaceAmounts { node_id, prior_x, prior_y, .. }
                        if *node_id == dormant_id && *prior_x == 0.0 && *prior_y == 0.0
                )),
                "waking a dormant Displace edge must be transactionally restorable"
            );
        }
    }

    /// Closes the saved (silent) half of Dice. Without the `prior_symmetry`
    /// capture and the paired mutation arm, generation falls into `_ => {}` and
    /// a Symmetry Field is never varied at all; with the arm, it must still
    /// leave every route, mask, seed, mode, and boundary — and therefore the
    /// whole 32-record sector table — bit-identical.
    #[test]
    fn generated_symmetry_moves_geometry_only_and_never_its_routes_masks_or_seed() {
        use crate::symmetry::{
            SavedMotionDonor, SymmetryBoundary, SymmetryMode, SymmetryMotionMask,
            SymmetryNodeDomain, SymmetryParams, SymmetrySourceMask,
        };
        use crate::visual_rack::{EdgeTiming, SavedImageSource, SavedImageTap};

        let authored = SymmetryParams {
            mode: SymmetryMode::PlanarP2,
            boundary: SymmetryBoundary::CellularReentry,
            base_folds: 6.0,
            radial_phase_deg: 30.0,
            hue_span: 0.4,
            motion_gain: 0.3,
            seed: 8_675,
            source_mask: SymmetrySourceMask {
                carrier: true,
                donor0: true,
                donor1: false,
                clean_history: true,
            },
            motion_mask: SymmetryMotionMask {
                slot0: true,
                slot1: false,
            },
            donors: [
                SavedImageTap {
                    source: SavedImageSource::CleanProgram,
                    timing: EdgeTiming::PreviousFrame,
                },
                SavedImageTap {
                    source: SavedImageSource::AllBelow,
                    timing: EdgeTiming::CurrentFrame,
                },
            ],
            motion: [
                SavedMotionDonor::Selected {
                    saved_position: crate::performance::SavedLayerPosition::new(2).unwrap(),
                },
                SavedMotionDonor::None,
            ],
            ..SymmetryParams::default()
        };
        let mut anchor = VisualRack::empty();
        let node_id = anchor.push(VisualNodeKind::Symmetry(authored)).unwrap();
        let params_of = |rack: &VisualRack| match rack.get(node_id).unwrap().kind {
            VisualNodeKind::Symmetry(params) => params,
            _ => panic!("symmetry node"),
        };
        let domain = SymmetryNodeDomain::new(0x4d41_5354_4552, node_id.get());
        let table = authored.sector_table(domain);

        let mut generated = anchor.clone();
        mutate_saved_rack_values(
            &anchor,
            &mut generated,
            1.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        let after = params_of(&generated);
        assert!(
            after.base_folds != authored.base_folds
                || after.radial_phase_deg != authored.radial_phase_deg
                || after.hue_span != authored.hue_span,
            "at least one declared continuous control must move at temperature 1.0"
        );
        assert!((1.0..=32.0).contains(&after.base_folds));
        assert!((-180.0..=180.0).contains(&after.radial_phase_deg));
        assert!((0.0..=1.0).contains(&after.hue_span));
        assert!((-1.0..=1.0).contains(&after.motion_gain));
        assert!((-1.0..=2.0).contains(&after.center[0]));

        assert_eq!(after.donors, authored.donors, "generation never reroutes");
        assert_eq!(after.motion, authored.motion);
        assert_eq!(after.source_mask, authored.source_mask);
        assert_eq!(after.motion_mask, authored.motion_mask);
        assert_eq!(after.seed, authored.seed);
        assert_eq!(after.mode, authored.mode);
        assert_eq!(after.boundary, authored.boundary);
        assert_eq!(
            after.sector_table(domain),
            table,
            "generation can never rewrite one sector record"
        );

        // Deterministic repeat and an exact zero-temperature no-op.
        let mut repeated = anchor.clone();
        mutate_saved_rack_values(
            &anchor,
            &mut repeated,
            1.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        assert_eq!(params_of(&repeated), after);
        let mut untouched = anchor.clone();
        mutate_saved_rack_values(
            &anchor,
            &mut untouched,
            0.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        assert_eq!(params_of(&untouched), authored);

        // A neighbouring node is byte-identical with and without this kind
        // present, because every node draws from its own stable domain.
        let grain = crate::visual_rack::GrainParams::default();
        let mut with_symmetry = VisualRack::empty();
        with_symmetry
            .push(VisualNodeKind::Symmetry(authored))
            .unwrap();
        let neighbour = with_symmetry.push(VisualNodeKind::Grain(grain)).unwrap();
        let mut without_symmetry = VisualRack::empty();
        without_symmetry
            .push(VisualNodeKind::Displace(DisplaceParams::default()))
            .unwrap();
        let same_slot = without_symmetry.push(VisualNodeKind::Grain(grain)).unwrap();
        assert_eq!(same_slot, neighbour, "the neighbour keeps its stable id");
        let with_anchor = with_symmetry.clone();
        let without_anchor = without_symmetry.clone();
        mutate_saved_rack_values(
            &with_anchor,
            &mut with_symmetry,
            1.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        mutate_saved_rack_values(
            &without_anchor,
            &mut without_symmetry,
            1.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        assert_eq!(
            with_symmetry.get(neighbour).unwrap().kind,
            without_symmetry.get(neighbour).unwrap().kind,
            "an older generated stream stays bit-identical beside a new kind"
        );
    }

    /// Closes `route_effect_active` and the `CreativeEdgeFallback` row. A
    /// Symmetry Field claims a saved edge exactly when the validator says it
    /// does, so waking one by lifting `wet` off zero must be transactionally
    /// restorable, and a carrier-only field must never record a spurious
    /// fallback for geometry that claims nothing.
    #[test]
    fn a_generated_symmetry_wake_is_transactional_and_matches_the_validator_predicate() {
        use crate::symmetry::{SymmetryParams, SymmetrySourceMask};
        use crate::visual_rack::{EdgeTiming, SavedImageSource, SavedImageTap};

        let armed_route = SavedImageTap {
            source: SavedImageSource::OneBelow,
            timing: EdgeTiming::CurrentFrame,
        };
        let armed = SymmetryParams {
            source_mask: SymmetrySourceMask {
                carrier: true,
                donor0: true,
                donor1: false,
                clean_history: false,
            },
            donors: [armed_route, SavedImageTap::default()],
            ..SymmetryParams::default()
        };

        // The claim predicate must agree with `collect_rack_dependencies` for
        // every combination of arming and geometry.
        assert!(
            symmetry_claims_saved_image_edge(armed),
            "an armed slot claims its edge"
        );
        assert!(
            !symmetry_claims_saved_image_edge(SymmetryParams {
                source_mask: SymmetrySourceMask::CARRIER_ONLY,
                ..armed
            }),
            "a carrier-only field claims nothing however it is routed"
        );
        assert!(
            !symmetry_claims_saved_image_edge(SymmetryParams {
                source_mask: SymmetrySourceMask::CARRIER_ONLY,
                hue_span: 1.0,
                base_folds: 9.0,
                ..armed
            }),
            "leaving the exact bypass does not by itself claim an edge"
        );

        // A dormant (zero-wet) node whose wet is lifted wakes a real edge, so a
        // transactional fallback must be recorded.
        let mut dormant = VisualRack::empty();
        let dormant_id = dormant.push(VisualNodeKind::Symmetry(armed)).unwrap();
        dormant.get_mut(dormant_id).unwrap().wet = 0.0;
        let anchor = dormant.clone();
        let mut woken = dormant.clone();
        let mut fallbacks = Vec::new();
        mutate_saved_rack_values(
            &anchor,
            &mut woken,
            1.0,
            0xC0FFEE,
            5,
            ProceduralRackOwner::Master,
            &mut fallbacks,
        );
        assert!(
            woken.get(dormant_id).unwrap().wet > 0.0,
            "this fixture's seed must actually lift wet off zero"
        );
        assert!(
            fallbacks.iter().any(|fallback| matches!(
                fallback,
                CreativeEdgeFallback::NodeWet { node_id, prior, .. }
                    if *node_id == dormant_id && *prior == 0.0
            )),
            "waking a Symmetry Field edge must be transactionally restorable"
        );

        // A carrier-only field is not a route consumer, so lifting its wet must
        // never record a fallback: reverting it would revert more than the
        // graph needs.
        let mut carrier_only = VisualRack::empty();
        let carrier_id = carrier_only
            .push(VisualNodeKind::Symmetry(SymmetryParams {
                source_mask: SymmetrySourceMask::CARRIER_ONLY,
                ..armed
            }))
            .unwrap();
        carrier_only.get_mut(carrier_id).unwrap().wet = 0.0;
        let carrier_anchor = carrier_only.clone();
        let mut carrier_woken = carrier_only.clone();
        let mut carrier_fallbacks = Vec::new();
        mutate_saved_rack_values(
            &carrier_anchor,
            &mut carrier_woken,
            1.0,
            0xC0FFEE,
            5,
            ProceduralRackOwner::Master,
            &mut carrier_fallbacks,
        );
        assert!(
            carrier_fallbacks.is_empty(),
            "a carrier-only Symmetry Field must never record a spurious fallback"
        );

        // The restore itself is exact: it puts back every mutated continuous
        // value and touches nothing else.
        let mut patch = advanced_anchor();
        let mut restored_rack = VisualRack::empty();
        let restore_id = restored_rack
            .push(VisualNodeKind::Symmetry(SymmetryParams {
                base_folds: 12.0,
                hue_span: 0.9,
                center: [0.1, 0.2],
                ..armed
            }))
            .unwrap();
        patch.master_rack = Some(restored_rack);
        CreativeEdgeFallback::SymmetryGeometry {
            owner: ProceduralRackOwner::Master,
            node_id: restore_id,
            prior: armed,
        }
        .restore(&mut patch);
        let VisualNodeKind::Symmetry(params) = patch
            .master_rack
            .as_ref()
            .unwrap()
            .get(restore_id)
            .unwrap()
            .kind
        else {
            panic!("symmetry node")
        };
        assert_eq!(params.base_folds, armed.base_folds);
        assert_eq!(params.hue_span, armed.hue_span);
        assert_eq!(params.center, armed.center);
        assert_eq!(
            params.donors, armed.donors,
            "a restore never rewrites a route"
        );
        assert_eq!(params.source_mask, armed.source_mask);
    }

    #[test]
    fn generated_residual_moves_values_only_and_wakes_both_edges_transactionally() {
        use crate::visual_rack::{
            EdgeTiming, ResidualBlock, ResidualQuantization, SavedImageSource, SavedImageTap,
        };

        let authored = ResidualParams {
            structure: SavedImageTap {
                source: SavedImageSource::CleanProgram,
                timing: EdgeTiming::PreviousFrame,
            },
            detail: SavedImageTap {
                source: SavedImageSource::AllBelow,
                timing: EdgeTiming::CurrentFrame,
            },
            block: ResidualBlock::Sixteen,
            quantization: ResidualQuantization::Medium,
            mix: 0.5,
            detail_gain: 2.0,
            seed: 0x00c0_ffee,
            ..ResidualParams::default()
        };
        let mut anchor_rack = VisualRack::empty();
        let node_id = anchor_rack
            .push(VisualNodeKind::Residual(authored))
            .unwrap();
        let params_of = |rack: &VisualRack| match rack.get(node_id).unwrap().kind {
            VisualNodeKind::Residual(params) => params,
            _ => panic!("residual node"),
        };

        let mut generated = anchor_rack.clone();
        mutate_saved_rack_values(
            &anchor_rack,
            &mut generated,
            1.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        let after = params_of(&generated);
        assert_eq!(
            after.routes(),
            authored.routes(),
            "generation never reroutes either donor"
        );
        assert_eq!(
            (after.block, after.quantization),
            (authored.block, authored.quantization)
        );
        assert_eq!(
            after.seed, authored.seed,
            "the quantization seed is authored topology, not a generated value"
        );
        assert_eq!(after.algorithm_version, authored.algorithm_version);
        assert!(after.mix != authored.mix || after.detail_gain != authored.detail_gain);
        assert!((0.0..=1.0).contains(&after.mix));
        assert!((0.0..=4.0).contains(&after.detail_gain));

        // Deterministic: identical seed and index reproduce identical values.
        let mut repeated = anchor_rack.clone();
        mutate_saved_rack_values(
            &anchor_rack,
            &mut repeated,
            1.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        assert_eq!(params_of(&repeated), after);

        // Zero temperature is an exact no-op.
        let mut untouched = anchor_rack.clone();
        mutate_saved_rack_values(
            &anchor_rack,
            &mut untouched,
            0.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        assert_eq!(params_of(&untouched), authored);

        // A neighbouring node draws from its own procedural domain, so
        // appending this arm cannot perturb an older generated stream. The
        // Grain node keeps the same NodeId in both racks.
        let build = |with_residual: bool| {
            let mut rack = VisualRack::empty();
            let grain_id = rack
                .push(VisualNodeKind::Grain(crate::visual_rack::GrainParams {
                    intensity: 0.08,
                    seed: 37,
                    ..crate::visual_rack::GrainParams::default()
                }))
                .unwrap();
            if with_residual {
                rack.push(VisualNodeKind::Residual(authored)).unwrap();
            }
            (rack, grain_id)
        };
        let (with_anchor, grain_id) = build(true);
        let (without_anchor, older_grain_id) = build(false);
        assert_eq!(older_grain_id, grain_id);
        let mut with_generated = with_anchor.clone();
        let mut without_generated = without_anchor.clone();
        mutate_saved_rack_values(
            &with_anchor,
            &mut with_generated,
            1.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        mutate_saved_rack_values(
            &without_anchor,
            &mut without_generated,
            1.0,
            0x5EED,
            3,
            ProceduralRackOwner::Master,
            &mut Vec::new(),
        );
        assert_eq!(
            with_generated.get(grain_id).unwrap(),
            without_generated.get(grain_id).unwrap(),
            "appending a Residual arm must not perturb an older generated stream"
        );

        // A node at zero mix has two dormant edges. Waking it must record one
        // transactional fallback that restores both slots to dormant at once.
        let mut dormant_anchor = VisualRack::empty();
        let dormant_id = dormant_anchor
            .push(VisualNodeKind::Residual(ResidualParams {
                structure: SavedImageTap {
                    source: SavedImageSource::OneBelow,
                    timing: EdgeTiming::CurrentFrame,
                },
                detail: SavedImageTap {
                    source: SavedImageSource::CleanProgram,
                    timing: EdgeTiming::CurrentFrame,
                },
                ..ResidualParams::default()
            }))
            .unwrap();
        let dormant_params = match dormant_anchor.get(dormant_id).unwrap().kind {
            VisualNodeKind::Residual(params) => params,
            _ => panic!("residual node"),
        };
        assert!(dormant_params.is_exact_bypass());

        let mut wakes = 0_u32;
        for seed in 0..8_u64 {
            let mut woken = dormant_anchor.clone();
            let mut fallbacks = Vec::new();
            mutate_saved_rack_values(
                &dormant_anchor,
                &mut woken,
                1.0,
                seed,
                1,
                ProceduralRackOwner::Master,
                &mut fallbacks,
            );
            let VisualNodeKind::Residual(params) = woken.get(dormant_id).unwrap().kind else {
                panic!("residual node")
            };
            if params.is_exact_bypass() {
                assert!(
                    !fallbacks.iter().any(|fallback| matches!(
                        fallback,
                        CreativeEdgeFallback::ResidualMix { .. }
                    )),
                    "a node that stayed dormant must not claim a woken edge"
                );
                continue;
            }
            wakes += 1;
            assert!(
                fallbacks.iter().any(|fallback| matches!(
                    fallback,
                    CreativeEdgeFallback::ResidualMix { node_id, prior_mix, .. }
                        if *node_id == dormant_id && *prior_mix == 0.0
                )),
                "waking a dormant Residual edge must be transactionally restorable"
            );

            // Replaying the fallback returns both routes to dormant.
            let mut patch = anchor();
            patch.master_rack = Some(woken.clone());
            patch.visual_schema_version = 1;
            for fallback in fallbacks {
                fallback.restore(&mut patch);
            }
            let restored = patch.master_rack.as_ref().unwrap();
            let VisualNodeKind::Residual(params) = restored.get(dormant_id).unwrap().kind else {
                panic!("residual node")
            };
            assert!(
                params.is_exact_bypass(),
                "restoring the prior mix must return both slots to dormant"
            );
            assert_eq!(
                params.routes(),
                dormant_params.routes(),
                "a restore never rewrites a route"
            );
        }
        assert!(
            wakes > 0,
            "at least one generated seed must wake the dormant edge"
        );
    }

    /// Generation and Dice mutate authored *values*. A recorded gesture is
    /// neither: nothing here may invent one, mutate one, or shift the RNG
    /// streams of the racks beside one.
    #[test]
    fn generation_never_invents_or_mutates_a_recorded_gesture_and_older_streams_stay_bit_identical()
    {
        use crate::gesture::{
            normalize_gesture_input, GestureEvent, GestureMode, GestureOrigin, GesturePhase,
            GestureTrack, GestureTrackDocument, RawGestureSample,
        };
        use crate::patch::GestureCanvasConfig;

        let sample = |phase, tick| {
            normalize_gesture_input(
                GestureOrigin::NativePointer,
                RawGestureSample::new(0, phase, GestureMode::Curl, [0.4, 0.6])
                    .with_direction([0.0, 1.0]),
                tick,
            )
            .expect("well-formed sample")
        };
        let mut track = GestureTrack::default();
        for (phase, tick) in [
            (GesturePhase::Begin, 40_u64),
            (GesturePhase::Move, 42),
            (GesturePhase::End, 47),
        ] {
            assert!(matches!(
                track.record_accepted(tick, sample(phase, 0)),
                Ok(true)
            ));
        }
        let document = GestureTrackDocument::capture(&track);
        let digest = track.checksum_hex();
        let canvas = GestureCanvasConfig {
            radius: 0.3,
            strength: 0.6,
            retention: 0.5,
        };

        // A gesture-free anchor never gains a gesture section.
        let anchor = advanced_anchor();
        assert_eq!(anchor.gesture_track, None);
        assert_eq!(anchor.gesture_canvas, None);
        let mut without = anchor.clone();
        let mut without_fallbacks = Vec::new();
        mutate_saved_creative_values(
            &anchor,
            &mut without,
            1.0,
            0x5EED,
            5,
            &mut without_fallbacks,
        );
        assert_eq!(
            without.gesture_track, None,
            "generation invents no recording"
        );
        assert_eq!(without.gesture_canvas, None);

        // The same anchor and seed carrying a recorded gesture must produce a
        // byte-identical creative mutation: gesture state is in no RNG domain,
        // so appending it cannot perturb an older Dice stream.
        let mut with_anchor = anchor.clone();
        with_anchor.gesture_track = Some(document.clone());
        with_anchor.gesture_canvas = Some(canvas);
        let mut with = with_anchor.clone();
        let mut with_fallbacks = Vec::new();
        mutate_saved_creative_values(&with_anchor, &mut with, 1.0, 0x5EED, 5, &mut with_fallbacks);
        assert_eq!(
            with.master_rack, without.master_rack,
            "a recorded gesture must not shift any older Dice stream"
        );
        assert_eq!(with.layers[0].rack, without.layers[0].rack);
        assert_eq!(with.composition, without.composition);
        assert_eq!(with_fallbacks.len(), without_fallbacks.len());

        // The recording itself is carried whole and still verifies.
        assert_eq!(with.gesture_track, Some(document.clone()));
        assert_eq!(with.gesture_canvas, Some(canvas));
        let restored = with.gesture_track.as_ref().unwrap().decode().unwrap();
        assert_eq!(restored.checksum_hex(), digest);
        assert_eq!(restored.events(), track.events());

        // Temperature zero is an exact identity, including the gesture world.
        let mut untouched = with_anchor.clone();
        let mut none = Vec::new();
        mutate_saved_creative_values(&with_anchor, &mut untouched, 0.0, 0x5EED, 5, &mut none);
        assert_eq!(untouched.gesture_track, with_anchor.gesture_track);
        assert_eq!(untouched.gesture_canvas, with_anchor.gesture_canvas);

        // A gesture-free patch also keeps its canonical identity: nothing this
        // tranche added is serialized, so the hash is the pre-gesture hash.
        let hash_before = canonical_patch_sha256(&anchor).unwrap();
        let mut still_free = anchor.clone();
        still_free.gesture_track = None;
        still_free.gesture_canvas = None;
        assert_eq!(canonical_patch_sha256(&still_free).unwrap(), hash_before);
        assert_ne!(canonical_patch_sha256(&with_anchor).unwrap(), hash_before);

        // Live Dice never sees gesture state at all.
        let live = include_str!("randomization.rs");
        assert!(
            !live.contains("gesture"),
            "live Dice must have no gesture awareness whatsoever"
        );
        // Silence the otherwise-unused binding without weakening the claim.
        let _: &[GestureEvent] = track.events();
    }
}
