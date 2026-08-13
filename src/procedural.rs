//! Deterministic, patch-only procedural generation.
//!
//! Version one deliberately generates inspectable YAML and manifests rather
//! than starting GPU exports. Rendering remains an explicit second step, so a
//! large request cannot monopolize the live renderer and every variant can be
//! curated before expensive media work begins.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::patch::{EffectsConfig, PatchState};

pub const GENERATOR_VERSION: &str = "1";
pub const MAX_GENERATED_COUNT: usize = 256;

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
    pub anchor_fnv1a64: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lineage: Option<String>,
    pub logical_sources: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone)]
pub struct GeneratedPiece {
    pub patch: PatchState,
    pub manifest: Manifest,
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
    }

    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    fn chance(&mut self, probability: f32) -> bool {
        self.unit() < probability.clamp(0.0, 1.0)
    }
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

/// Reflect a value at both walls rather than clamping it. Reflection avoids
/// accumulating probability mass at minima and maxima during a long walk.
fn reflect(mut value: f32, min: f32, max: f32) -> f32 {
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

fn wrap(value: f32, min: f32, max: f32) -> f32 {
    let span = max - min;
    if !value.is_finite() || span <= 0.0 {
        return min;
    }
    (value - min).rem_euclid(span) + min
}

const MEAN_REVERSION: f32 = 0.85;

fn mutate_linear(
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

fn mutate_log(
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

fn mutate_circular(
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

fn mutate_discrete<T: Copy + PartialEq>(
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

fn normalized_anchor(anchor: &PatchState) -> Result<PatchState, String> {
    let yaml = serde_yaml::to_string(anchor).map_err(|e| format!("serialize anchor: {e}"))?;
    let mut normalized: PatchState =
        serde_yaml::from_str(&yaml).map_err(|e| format!("normalize anchor: {e}"))?;

    let sanitize_effects = |effects: &EffectsConfig| {
        let mut uniforms = crate::effects::EffectUniforms::default();
        effects.apply_to_uniforms(&mut uniforms);
        EffectsConfig::from_uniforms(&uniforms)
    };
    normalized.master = sanitize_effects(&normalized.master);
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
        layer.blend_mode = match layer.blend_mode.as_str() {
            "screen" | "multiply" | "difference" => layer.blend_mode.clone(),
            _ => "normal".to_string(),
        };
        layer.effects = sanitize_effects(&layer.effects);
    }
    normalized.ntsc = normalized
        .ntsc
        .as_ref()
        .map(|config| crate::patch::NtscConfig::from_params(&config.to_params()));
    normalized.temporal = normalized
        .temporal
        .as_ref()
        .map(|config| crate::patch::TemporalConfig::from_params(&config.to_params()));
    normalized.modulation = normalized.modulation.as_ref().map(|config| {
        let mut matrix = crate::modulation::ModMatrix::new();
        config.apply_to_matrix(&mut matrix);
        crate::patch::ModConfig::from_matrix(&matrix)
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

pub fn generate(
    anchor: &PatchState,
    config: &GenerationConfig,
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

    let normalized = normalized_anchor(anchor)?;
    let anchor_yaml = serde_yaml::to_string(&normalized)
        .map_err(|e| format!("serialize normalized anchor: {e}"))?;
    let anchor_hash = format!("{:016x}", fnv1a64(anchor_yaml.as_bytes()));
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
            const BLENDS: &[&str] = &["normal", "screen", "multiply", "difference"];
            let blend = mutate_discrete(
                anchor_layer.blend_mode.as_str(),
                layer.blend_mode.as_str(),
                BLENDS,
                temperature * 0.04,
                &mut rng,
            );
            layer.blend_mode = blend.to_string();
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

        let title_seed = domain_seed(config.seed, index, 0x5449_544c_4500);
        let title = title_for(&patch, title_seed);
        let slug = slugify(&title);
        let warnings = spout_sources
            .iter()
            .map(|name| format!("Spout source {name:?} is deterministic black offline"))
            .collect();
        let manifest = Manifest {
            schema_version: 1,
            generator_version: GENERATOR_VERSION.to_string(),
            seed: config.seed,
            index,
            temperature,
            title,
            slug,
            anchor_fnv1a64: anchor_hash.clone(),
            lineage: Some(anchor_hash.clone()),
            logical_sources: logical_sources.clone(),
            warnings,
        };
        walk = patch.clone();
        pieces.push(GeneratedPiece { patch, manifest });
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
        PatchState {
            master: EffectsConfig::default(),
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
                effects: EffectsConfig::default(),
            }],
            master_paused: false,
            ntsc: None,
            modulation: None,
            temporal: None,
            morph: None,
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
        assert_eq!(first[0].patch.layers[0].filename, "clip.mp4");
        assert!(first[0].manifest.title.contains(' '));
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
    fn generated_values_stay_finite_bounded_and_preserve_topology() {
        let config = GenerationConfig {
            seed: 99,
            count: 32,
            temperature: 2.0,
            allow_black_sources: false,
        };
        let mut source = anchor();
        source.layers[0].bypass_master_fx = true;
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
        }
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
