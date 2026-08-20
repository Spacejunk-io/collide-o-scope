//! The B6 Block DCT law — a codec's transform with the quantiser under your
//! hand.
//!
//! The artefacts everybody recognises from a badly compressed picture are
//! not properties of a file format, they are properties of a transform: a
//! block DCT, coefficients quantised, transformed back. Ringing around
//! edges, blocking where the quantiser is coarse, and colour smeared across
//! a block because chroma is carried coarser than luma. The transform is
//! separable — one axis quantised, then the other — which is not identical
//! to a true 2D quantisation, but every artefact it produces is a real one.
//!
//! The law is derived from BENDR (MIT, © 2026 Steve Blythe) and transcribed
//! faithfully with attribution: the DCT-II basis with orthonormal scaling,
//! the quantiser step `(0.004 + q·0.5)·(1 + u·tilt·2)` with round-to-nearest,
//! the coefficient-domain chroma crush `(1 − chroma·0.85)` against Rec.601
//! luma, and the 4–16 block-size map `floor(4 + block·12)`. The arithmetic
//! runs in the **encoded sRGB domain on straight RGB** — the B8 code-byte /
//! B5 real-codec precedent: a codec quantises stored values, so quantising
//! linear light would manufacture different artefacts. This module is the
//! independent CPU reference the GPU passes are checked against, in the
//! `gesture.rs` tradition: no wgpu, clock, filesystem, or UI dependency.
//!
//! One deliberate structural deviation from BENDR's single fused pass per
//! axis: BENDR evaluates all N coefficients per fragment (O(N²) taps per
//! pixel). At full output resolution that is hostile, so each axis splits
//! into a coefficient pass (each texel holds its own block-relative
//! coefficient, N taps) and a reconstruction pass (N coefficient taps),
//! O(N) per pixel per pass with byte-identical mathematics — the same sums,
//! reassociated.

use serde::{Deserialize, Serialize};

/// Block-size bounds: `N = floor(4 + block·12)`, BENDR's own map.
pub const DCT_MIN_BLOCK: u32 = 4;
pub const DCT_MAX_BLOCK: u32 = 16;

/// The quantiser step floor. BENDR's constant: even at quantise zero the
/// round is a real (if invisible at 8 bits) quantisation, so bypass is the
/// amount gate, never a "quantise zero is identity" claim.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const DCT_STEP_FLOOR: f32 = 0.004;

/// Authored Block DCT state. All five controls are continuous and
/// modulatable; the wire keys are `dct_`-prefixed because bare names
/// cross-resolve between kinds under `same_wire_parameter`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BlockDctParams {
    /// Dry/wet toward the quantised reconstruction. The wake law: zero is an
    /// exact bypass.
    pub amount: f32,
    /// Quantiser coarseness.
    pub quantize: f32,
    /// High-frequency penalty: the quantiser gets coarser for higher
    /// coefficients, which is the whole reason compression looks the way it
    /// does.
    pub hf_penalty: f32,
    /// Coefficient-domain chroma crush: chroma is quantised harder than
    /// luma, as it always is.
    pub chroma_crush: f32,
    /// Block-size control, mapped to 4–16 texels.
    pub block: f32,
}

impl Default for BlockDctParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            quantize: 0.25,
            hf_penalty: 0.5,
            chroma_crush: 0.4,
            block: 0.35,
        }
    }
}

impl BlockDctParams {
    /// Clamp every authored value into its declared range. Hostile
    /// non-finite input takes the neutral default rather than a clamped
    /// extreme.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            amount: finite_clamp(self.amount, defaults.amount, 0.0, 1.0),
            quantize: finite_clamp(self.quantize, defaults.quantize, 0.0, 1.0),
            hf_penalty: finite_clamp(self.hf_penalty, defaults.hf_penalty, 0.0, 1.0),
            chroma_crush: finite_clamp(self.chroma_crush, defaults.chroma_crush, 0.0, 1.0),
            block: finite_clamp(self.block, defaults.block, 0.0, 1.0),
        }
    }

    /// True when no wet reconstruction is authored: the planner emits the
    /// step but the executor encodes nothing and the carrier passes through
    /// untouched. Amount alone is the wake law (BENDR's own stage gate);
    /// quantiser and block dressing shape a reconstruction that exists only
    /// once the amount does.
    pub fn is_exact_bypass(self) -> bool {
        self.sanitized().amount == 0.0
    }

    /// The block edge in texels: `floor(4 + block·12)`, clamped structurally.
    pub fn block_edge(self) -> u32 {
        let clean = self.sanitized();
        (((4.0 + clean.block * 12.0).floor()) as u32).clamp(DCT_MIN_BLOCK, DCT_MAX_BLOCK)
    }
}

/// DCT-II orthonormal scale: `sqrt(1/N)` for the DC coefficient, `sqrt(2/N)`
/// otherwise — applied on analysis *and* synthesis, BENDR's own spelling.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn dct_scale(u: usize, n: usize) -> f32 {
    if u == 0 {
        (1.0 / n as f32).sqrt()
    } else {
        (2.0 / n as f32).sqrt()
    }
}

/// Forward DCT-II of one block line: `co_u = scale(u)·Σ s_x·cos((2x+1)uπ/2N)`.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn transform_block(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    (0..n)
        .map(|u| {
            let sum: f32 = samples
                .iter()
                .enumerate()
                .map(|(x, s)| {
                    s * ((2.0 * x as f32 + 1.0) * u as f32 * std::f32::consts::PI
                        / (2.0 * n as f32))
                        .cos()
                })
                .sum();
            sum * dct_scale(u, n)
        })
        .collect()
}

/// Inverse of [`transform_block`]: `s_k = Σ_u scale(u)·co_u·cos((2k+1)uπ/2N)`.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn reconstruct_block(coefficients: &[f32]) -> Vec<f32> {
    let n = coefficients.len();
    (0..n)
        .map(|k| {
            coefficients
                .iter()
                .enumerate()
                .map(|(u, co)| {
                    co * dct_scale(u, n)
                        * ((2.0 * k as f32 + 1.0) * u as f32 * std::f32::consts::PI
                            / (2.0 * n as f32))
                            .cos()
                })
                .sum()
        })
        .collect()
}

/// The quantiser step for coefficient index `u`:
/// `(0.004 + q·0.5)·(1 + u·tilt·2)` — floor at [`DCT_STEP_FLOOR`], linear in
/// the quantise control, HF penalty linear in the coefficient index.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn quantize_step(quantize: f32, hf_penalty: f32, u: usize) -> f32 {
    (DCT_STEP_FLOOR + quantize * 0.5) * (1.0 + u as f32 * hf_penalty * 2.0)
}

/// Quantise one RGB coefficient triple in place: crush coefficient chroma
/// against Rec.601 luma first (BENDR quantises what a codec carries coarser
/// *before* the round), then round-to-nearest on the step.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn quantize_coefficient(co: [f32; 3], step: f32, chroma_crush: f32) -> [f32; 3] {
    let y = 0.299 * co[0] + 0.587 * co[1] + 0.114 * co[2];
    let keep = 1.0 - chroma_crush * 0.85;
    let crushed = [
        y + (co[0] - y) * keep,
        y + (co[1] - y) * keep,
        y + (co[2] - y) * keep,
    ];
    crushed.map(|value| (value / step + 0.5).floor() * step)
}

/// The whole one-axis law over one line of encoded straight RGB: per block,
/// forward transform, quantise, inverse transform. This is the reference the
/// GPU's split coefficient/reconstruction passes must reproduce.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn block_dct_line(line: &[[f32; 3]], params: BlockDctParams) -> Vec<[f32; 3]> {
    let clean = params.sanitized();
    let n = clean.block_edge() as usize;
    let mut out = vec![[0.0f32; 3]; line.len()];
    for (block_index, block) in line.chunks(n).enumerate() {
        let base = block_index * n;
        // Transform each channel, quantise the triples, reconstruct.
        let channels: [Vec<f32>; 3] = [0, 1, 2].map(|channel| {
            transform_block(&block.iter().map(|px| px[channel]).collect::<Vec<_>>())
        });
        let mut quantised: [Vec<f32>; 3] =
            [Vec::new(), Vec::new(), Vec::new()].map(|_: Vec<f32>| vec![0.0; block.len()]);
        for u in 0..block.len() {
            let step = quantize_step(clean.quantize, clean.hf_penalty, u);
            let triple = quantize_coefficient(
                [channels[0][u], channels[1][u], channels[2][u]],
                step,
                clean.chroma_crush,
            );
            for channel in 0..3 {
                quantised[channel][u] = triple[channel];
            }
        }
        let restored: [Vec<f32>; 3] =
            [0, 1, 2].map(|channel| reconstruct_block(&quantised[channel]));
        for k in 0..block.len() {
            out[base + k] = [restored[0][k], restored[1][k], restored[2][k]];
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
    fn sanitize_is_neutral_on_hostile_input_and_bypass_is_amount_alone() {
        let hostile = BlockDctParams {
            amount: f32::NAN,
            quantize: f32::INFINITY,
            hf_penalty: -3.0,
            chroma_crush: 9.0,
            block: f32::NEG_INFINITY,
        };
        let clean = hostile.sanitized();
        assert_eq!(clean.amount, 0.0, "non-finite takes the neutral default");
        assert_eq!(clean.quantize, 0.25);
        assert_eq!(clean.hf_penalty, 0.0, "finite out-of-range clamps");
        assert_eq!(clean.chroma_crush, 1.0);
        assert_eq!(clean.block, 0.35);
        assert!(hostile.is_exact_bypass());
        assert!(BlockDctParams::default().is_exact_bypass());
        assert!(!BlockDctParams {
            amount: 0.4,
            ..Default::default()
        }
        .is_exact_bypass());
        // Quantiser dressing alone wakes nothing.
        assert!(BlockDctParams {
            quantize: 1.0,
            block: 1.0,
            ..Default::default()
        }
        .is_exact_bypass());
    }

    #[test]
    fn block_edge_follows_the_bendr_map() {
        let edge = |block: f32| {
            BlockDctParams {
                block,
                ..Default::default()
            }
            .block_edge()
        };
        assert_eq!(edge(0.0), 4);
        assert_eq!(edge(0.35), 8, "the default is the classic 8-point block");
        assert_eq!(edge(1.0), 16);
        assert_eq!(edge(f32::NAN), 8, "hostile block takes the default edge");
    }

    #[test]
    fn forward_inverse_is_identity_without_the_quantiser() {
        for n in [4usize, 8, 16] {
            let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin() * 0.5 + 0.4).collect();
            let restored = reconstruct_block(&transform_block(&samples));
            for (a, b) in samples.iter().zip(restored.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "DCT round trip must be identity at N={n}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn the_quantiser_step_is_floored_tilted_and_monotonic() {
        assert!((quantize_step(0.0, 0.0, 0) - DCT_STEP_FLOOR).abs() < 1e-7);
        // Coarser with the control, coarser with the coefficient index.
        assert!(quantize_step(0.5, 0.5, 0) > quantize_step(0.1, 0.5, 0));
        assert!(quantize_step(0.5, 0.5, 7) > quantize_step(0.5, 0.5, 1));
        // No penalty means a flat spectrum of steps.
        assert!((quantize_step(0.3, 0.0, 9) - quantize_step(0.3, 0.0, 0)).abs() < 1e-7);
    }

    #[test]
    fn chroma_crush_shrinks_coefficient_chroma_toward_its_luma() {
        let co = [0.8f32, 0.2, 0.1];
        let y = 0.299 * co[0] + 0.587 * co[1] + 0.114 * co[2];
        // A tiny step isolates the crush from the round.
        let crushed = quantize_coefficient(co, 1e-6, 1.0);
        for channel in 0..3 {
            let expected = y + (co[channel] - y) * 0.15;
            assert!(
                (crushed[channel] - expected).abs() < 1e-4,
                "full crush retains exactly 15% of coefficient chroma"
            );
        }
        // Neutral grey coefficients are unmoved by any crush.
        let grey = quantize_coefficient([0.5, 0.5, 0.5], 1e-6, 1.0);
        for channel in grey {
            assert!((channel - 0.5).abs() < 1e-4);
        }
    }

    #[test]
    fn a_constant_block_survives_and_an_edge_rings() {
        // A flat block is pure DC: quantisation moves it at most one step.
        let flat = vec![[0.5f32; 3]; 8];
        let params = BlockDctParams {
            amount: 1.0,
            quantize: 0.3,
            ..Default::default()
        };
        let out = block_dct_line(&flat, params);
        let step = quantize_step(0.3, 0.5, 0);
        for px in &out {
            for channel in px {
                assert!(
                    (channel - 0.5).abs() <= step,
                    "a flat block moves at most one DC step"
                );
            }
        }
        // A hard edge under a coarse quantiser rings: the output leaves the
        // input's [0,1] hull somewhere near the edge.
        let mut edge = vec![[0.05f32; 3]; 8];
        for px in edge.iter_mut().skip(4) {
            *px = [0.95; 3];
        }
        let coarse = BlockDctParams {
            amount: 1.0,
            quantize: 0.8,
            ..Default::default()
        };
        let rung = block_dct_line(&edge, coarse);
        let overshoot = rung
            .iter()
            .flatten()
            .any(|v| *v < 0.05 - 1e-4 || *v > 0.95 + 1e-4);
        assert!(overshoot, "a coarse quantiser must ring around a hard edge");
    }
}
