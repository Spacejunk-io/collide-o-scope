//! Deterministic primitives shared by offline generation and live rerolls.
//!
//! The live controls deliberately avoid entropy from clocks or the operating
//! system. A performer can therefore replay an exact seed, and an omitted
//! seed advances a stable sequence from the state already stored in a patch.

use crate::effects::EffectUniforms;

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
}
