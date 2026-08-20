//! The B6 Filter Avalanche law — reconstruction-filter corruption that
//! cascades.
//!
//! A PNG row is not stored as pixels, it is stored as a difference against a
//! predictor, and the decoder rebuilds it by accumulating. Corrupt one byte
//! and every pixel after it inherits the error, so a single bad value
//! avalanches to the edge of the picture. Which direction it runs is decided
//! by which predictor the row used: SUB accumulates along the line, UP down
//! the column, AVERAGE diagonally with a soft tail.
//!
//! The per-frame law is derived from BENDR (MIT, © 2026 Steve Blythe) and
//! transcribed with attribution: the three predictor directions, the
//! per-lane corruption gate firing at `amount·0.5`, the bounded
//! gradient-sum accumulation (`span = 2 + run·40`, at most 32 taps,
//! out-of-frame taps masked), the per-lane DC seed, and the `fract` wrap —
//! the byte-level overflow that produces the hard hue flips. Two house
//! hardenings BENDR never claimed:
//!
//! - **Determinism.** BENDR re-rolls its corrupt lanes on wall-clock time at
//!   3 Hz. Here the lane epoch is `floor(frame-plan seconds × 3)` through
//!   the shared integer-avalanche hash, keyed by the node's stable authored
//!   id, so Pause holds the fault stream and live and offline replay it
//!   identically.
//! - **The cascade.** BENDR accumulates gradients of the current frame, so
//!   its corruption never travels. Here the accumulation reads the node's
//!   **own previous output** — one retained surface, advanced at most once
//!   per reference tick (the melt-history rate law, because a per-frame
//!   advance would cascade at different speeds live and offline) — so an
//!   error written last tick is re-inherited this tick and the avalanche
//!   becomes visible motion. Before the first committed history the
//!   accumulation reads the carrier itself, which is exactly BENDR's
//!   shipped single-frame law: a cold node degrades to the transcription,
//!   never to nothing.
//!
//! This module is the independent CPU reference the dedicated pass is
//! checked against, in the `gesture.rs` tradition: no wgpu, clock,
//! filesystem, or UI dependency.

use serde::{Deserialize, Serialize};

use crate::mixing_boundary::lane_unit;

/// Bounded accumulation: at most 32 taps regardless of the span control.
pub const AVALANCHE_MAX_TAPS: u32 = 32;
/// The lane epoch advances at 3 Hz on frame-plan seconds — BENDR's own
/// re-roll rate (`floor(u_time*3)`), made deterministic by clocking it on
/// the shared frame-plan time instead of the wall clock: Pause holds it and
/// export replays it structurally.
pub const AVALANCHE_EPOCH_HZ: f32 = 3.0;

/// Fresh hash-lane domains ("AVL" 1/2). Each draw site owns exactly one
/// constant, the B8 law.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const LANE_AVALANCHE_FIRE: u32 = 0x4156_4c01;
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const LANE_AVALANCHE_DC: u32 = 0x4156_4c02;

/// The predictor vocabulary. Closed and append-only (codes 0–2 permanent);
/// a discrete law with no modulatable address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AvalancheAxis {
    /// SUB: the error accumulates along the scanline, code 0.
    #[default]
    Sub,
    /// UP: down the column, code 1.
    Up,
    /// AVERAGE: diagonally with a soft tail, code 2.
    Average,
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the vocabulary tables the CPU reference and its fixtures consume"
    )
)]
impl AvalancheAxis {
    pub const ALL: [Self; 3] = [Self::Sub, Self::Up, Self::Average];

    pub fn code(&self) -> u32 {
        match self {
            Self::Sub => 0,
            Self::Up => 1,
            Self::Average => 2,
        }
    }

    pub fn key(&self) -> &'static str {
        match self {
            Self::Sub => "sub",
            Self::Up => "up",
            Self::Average => "average",
        }
    }

    /// The accumulation step in texels (x rightward, y downward), scaled
    /// `0.7071` on the diagonal exactly as BENDR ships it.
    // The diagonal is BENDR's literal 0.7071, kept byte for byte with the
    // WGSL mirror rather than rewritten as FRAC_1_SQRT_2.
    #[allow(
        clippy::approx_constant,
        reason = "BENDR's transcribed literal, mirrored exactly in corruption.wgsl"
    )]
    pub fn step_texels(&self) -> [f32; 2] {
        match self {
            Self::Sub => [1.0, 0.0],
            Self::Up => [0.0, 1.0],
            Self::Average => [0.7071, 0.7071],
        }
    }
}

/// Authored Filter Avalanche state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AvalancheParams {
    /// Corruption pressure: the lane-fire probability is `amount·0.5`, the
    /// accumulated error scales by `amount·1.6`, and the per-lane DC by
    /// `amount·0.25`. The wake law: zero is an exact bypass.
    pub amount: f32,
    /// Accumulation span control: `span = 2 + run·40` taps, capped at 32.
    pub run: f32,
    /// The predictor. Discrete authored state.
    pub axis: AvalancheAxis,
}

impl Default for AvalancheParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            run: 0.4,
            axis: AvalancheAxis::Sub,
        }
    }
}

impl AvalancheParams {
    /// Clamp every authored value into its declared range. Hostile
    /// non-finite input takes the neutral default rather than a clamped
    /// extreme.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            amount: finite_clamp(self.amount, defaults.amount, 0.0, 1.0),
            run: finite_clamp(self.run, defaults.run, 0.0, 1.0),
            axis: self.axis,
        }
    }

    /// True when no corruption pressure is authored: the executor encodes
    /// nothing, allocates nothing, and the carrier passes through untouched.
    pub fn is_exact_bypass(self) -> bool {
        self.sanitized().amount == 0.0
    }

    /// The accumulation span in taps: `2 + run·40`, capped structurally.
    pub fn span(self) -> f32 {
        (2.0 + self.sanitized().run * 40.0).min(AVALANCHE_MAX_TAPS as f32)
    }
}

/// The lane epoch for a frame-plan timestamp.
pub fn avalanche_epoch(time_seconds: f32) -> u32 {
    if !time_seconds.is_finite() || time_seconds <= 0.0 {
        return 0;
    }
    (time_seconds * AVALANCHE_EPOCH_HZ).floor() as u32
}

/// Whether a lane's corruption fires this epoch. Deterministic: the shared
/// integer-avalanche hash keyed by the master seed, the lane, and the epoch,
/// in this module's own fire domain.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn lane_fires(seed: u32, lane: u32, epoch: u32, amount: f32) -> bool {
    lane_unit(LANE_AVALANCHE_FIRE, epoch, lane, seed) < amount * 0.5
}

/// The lane's static DC offset in [-1, 1]. Seeded by the lane alone (and the
/// master seed), deliberately epoch-invariant — BENDR's own law.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn lane_dc(seed: u32, lane: u32) -> f32 {
    lane_unit(LANE_AVALANCHE_DC, 0, lane, seed) * 2.0 - 1.0
}

/// The whole law over one straight encoded-RGB frame. `previous` is the
/// node's retained previous output when a committed history exists;
/// `None` reads the carrier itself (the cold-start law). Row-major, y
/// downward; out-of-frame taps are masked, in-frame taps clamp.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn avalanche_reference(
    carrier: &[[f32; 3]],
    previous: Option<&[[f32; 3]]>,
    width: usize,
    height: usize,
    params: AvalancheParams,
    seed: u32,
    time_seconds: f32,
) -> Vec<[f32; 3]> {
    let clean = params.sanitized();
    let mut out = carrier.to_vec();
    if clean.amount == 0.0 || width == 0 || height == 0 || carrier.len() < width * height {
        return out;
    }
    let history = match previous {
        Some(history) if history.len() >= width * height => history,
        _ => carrier,
    };
    let epoch = avalanche_epoch(time_seconds);
    let step = clean.axis.step_texels();
    let span = clean.span();
    for y in 0..height {
        for x in 0..width {
            let lane = match clean.axis {
                AvalancheAxis::Sub => y as u32,
                AvalancheAxis::Up | AvalancheAxis::Average => x as u32,
            };
            if !lane_fires(seed, lane, epoch, clean.amount) {
                continue;
            }
            let mut acc = [0.0f32; 3];
            for i in 1..=AVALANCHE_MAX_TAPS {
                let fi = i as f32;
                if fi > span {
                    break;
                }
                let tx = x as f32 - step[0] * fi;
                let ty = y as f32 - step[1] * fi;
                // Out-of-frame taps contribute nothing but do not stop the
                // walk (BENDR's `inb` mask).
                if tx < 0.0 || ty < 0.0 || tx >= width as f32 || ty >= height as f32 {
                    continue;
                }
                let sample = |fx: f32, fy: f32| -> [f32; 3] {
                    let sx = (fx.max(0.0) as usize).min(width - 1);
                    let sy = (fy.max(0.0) as usize).min(height - 1);
                    history[sy * width + sx]
                };
                let a1 = sample(tx, ty);
                let a2 = sample(tx - step[0], ty - step[1]);
                for channel in 0..3 {
                    acc[channel] += a1[channel] - a2[channel];
                }
            }
            let dc = lane_dc(seed, lane);
            let px = &mut out[y * width + x];
            for channel in 0..3 {
                px[channel] =
                    (px[channel] + acc[channel] * clean.amount * 1.6 + dc * clean.amount * 0.25)
                        .rem_euclid(1.0);
            }
        }
    }
    out
}

fn finite_clamp(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_is_neutral_axis_codes_are_frozen_and_bypass_is_amount_alone() {
        let hostile = AvalancheParams {
            amount: f32::NAN,
            run: -5.0,
            axis: AvalancheAxis::Average,
        };
        let clean = hostile.sanitized();
        assert_eq!(clean.amount, 0.0);
        assert_eq!(clean.run, 0.0);
        assert_eq!(clean.axis, AvalancheAxis::Average);
        assert!(hostile.is_exact_bypass());
        assert!(AvalancheParams::default().is_exact_bypass());
        assert!(!AvalancheParams {
            amount: 0.3,
            ..Default::default()
        }
        .is_exact_bypass());
        assert_eq!(
            AvalancheAxis::ALL,
            [
                AvalancheAxis::Sub,
                AvalancheAxis::Up,
                AvalancheAxis::Average
            ]
        );
        for (index, axis) in AvalancheAxis::ALL.iter().enumerate() {
            assert_eq!(axis.code(), index as u32, "codes are the ALL order");
            assert_eq!(axis.key(), ["sub", "up", "average"][index]);
        }
        assert_eq!(AvalancheAxis::default(), AvalancheAxis::Sub);
    }

    #[test]
    fn span_and_epoch_follow_their_laws() {
        let span = |run: f32| {
            AvalancheParams {
                run,
                ..Default::default()
            }
            .span()
        };
        assert!((span(0.0) - 2.0).abs() < 1e-6);
        assert!((span(0.5) - 22.0).abs() < 1e-6);
        assert!(
            (span(1.0) - 32.0).abs() < 1e-6,
            "the 42-tap request caps at the structural 32"
        );
        assert_eq!(avalanche_epoch(0.0), 0);
        assert_eq!(avalanche_epoch(0.33), 0);
        assert_eq!(avalanche_epoch(0.34), 1);
        assert_eq!(avalanche_epoch(10.0), 30, "3 Hz on frame-plan seconds");
        assert_eq!(avalanche_epoch(f32::NAN), 0, "hostile time is epoch zero");
    }

    #[test]
    fn the_fire_gate_is_deterministic_and_honest_about_its_probability() {
        // Same key, same answer — twice.
        for lane in 0..8u32 {
            assert_eq!(
                lane_fires(7, lane, 3, 0.6),
                lane_fires(7, lane, 3, 0.6),
                "the gate is a pure function"
            );
        }
        // Over many lanes the firing fraction approaches amount/2.
        let fired = (0..4096u32)
            .filter(|lane| lane_fires(11, *lane, 5, 0.8))
            .count();
        let fraction = fired as f64 / 4096.0;
        assert!(
            (fraction - 0.4).abs() < 0.04,
            "firing fraction {fraction:.3} must sit near amount/2"
        );
        // A different epoch re-rolls the lane set; a different seed too.
        let set = |seed: u32, epoch: u32| -> Vec<bool> {
            (0..256u32)
                .map(|lane| lane_fires(seed, lane, epoch, 0.5))
                .collect()
        };
        assert_ne!(set(11, 5), set(11, 6));
        assert_ne!(set(11, 5), set(12, 5));
        // The DC seed is epoch-invariant, bounded, and lane-diverse.
        for lane in 0..32u32 {
            let dc = lane_dc(9, lane);
            assert!((-1.0..=1.0).contains(&dc));
        }
        assert_ne!(lane_dc(9, 1), lane_dc(9, 2));
    }

    #[test]
    fn a_flat_history_wraps_only_the_dc_and_gradients_accumulate() {
        let width = 16usize;
        let height = 16usize;
        let flat = vec![[0.5f32; 3]; width * height];
        let params = AvalancheParams {
            amount: 1.0,
            run: 0.5,
            axis: AvalancheAxis::Sub,
        };
        // Flat history: acc is zero everywhere, so a fired lane moves by
        // exactly its DC term and an unfired lane is untouched.
        let out = avalanche_reference(&flat, None, width, height, params, 3, 0.0);
        for y in 0..height {
            let lane_hit = lane_fires(3, y as u32, 0, 1.0);
            for x in 0..width {
                let value = out[y * width + x][0];
                if lane_hit {
                    let expected = (0.5 + lane_dc(3, y as u32) * 0.25).rem_euclid(1.0);
                    assert!(
                        (value - expected).abs() < 1e-5,
                        "a flat field moves by the DC alone"
                    );
                } else {
                    assert_eq!(value, 0.5, "an unfired lane is bit-clean");
                }
            }
        }
        // A hard vertical edge under SUB: pixels to the right of the edge
        // accumulate the gradient and move further than the DC alone.
        let mut edged = flat.clone();
        for y in 0..height {
            for x in 8..width {
                edged[y * width + x] = [0.9; 3];
            }
        }
        let out = avalanche_reference(&edged, None, width, height, params, 3, 0.0);
        let fired_lane = (0..height)
            .find(|y| lane_fires(3, *y as u32, 0, 1.0))
            .expect("some lane fires at amount 1");
        let with_gradient = out[fired_lane * width + 12][0];
        let dc_only = (0.9 + lane_dc(3, fired_lane as u32) * 0.25).rem_euclid(1.0);
        assert!(
            (with_gradient - dc_only).abs() > 1e-4,
            "a crossed edge must contribute accumulated error"
        );
    }

    #[test]
    fn the_cascade_reads_the_previous_output_and_cold_start_reads_the_carrier() {
        let width = 8usize;
        let height = 8usize;
        let carrier = vec![[0.5f32; 3]; width * height];
        // A previous output carrying a hard edge: the error is inherited
        // from history even though the carrier is flat.
        let mut previous = carrier.clone();
        for y in 0..height {
            for x in 4..width {
                previous[y * width + x] = [0.1; 3];
            }
        }
        let params = AvalancheParams {
            amount: 1.0,
            run: 1.0,
            axis: AvalancheAxis::Sub,
        };
        let warm = avalanche_reference(&carrier, Some(&previous), width, height, params, 5, 0.0);
        let cold = avalanche_reference(&carrier, None, width, height, params, 5, 0.0);
        assert_ne!(
            warm, cold,
            "history participates: the cascade inherits last tick's errors"
        );
        // Cold start with a flat carrier: gradients vanish, DC only — the
        // exact BENDR single-frame law.
        let fired_lane = (0..height)
            .find(|y| lane_fires(5, *y as u32, 0, 1.0))
            .expect("some lane fires");
        let expected = (0.5 + lane_dc(5, fired_lane as u32) * 0.25).rem_euclid(1.0);
        assert!((cold[fired_lane * width + 6][0] - expected).abs() < 1e-5);
        // A short hostile history falls back to the carrier read.
        let short = avalanche_reference(
            &carrier,
            Some(&previous[..4]),
            width,
            height,
            params,
            5,
            0.0,
        );
        assert_eq!(short, cold);
    }
}
