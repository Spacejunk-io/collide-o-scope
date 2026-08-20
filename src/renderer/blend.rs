//! Canonical layer blend laws shared by CPU reference paths and both GPU
//! compositors. RGB values are linear-light and straight-alpha throughout.

use std::borrow::Cow;

use crate::layers::BlendMode;

const BLEND_KERNEL_WGSL: &str = include_str!("../shaders/blend.wgsl");
const COMPOSITE_BODY_WGSL: &str = include_str!("../shaders/composite.wgsl");
const MATTE_COMPOSITE_BODY_WGSL: &str = include_str!("../shaders/matte_composite.wgsl");
const COMPOSITION_HOST_BODY_WGSL: &str = include_str!("../shaders/composition_host.wgsl");

/// Build the ordinary two-input compositor from the one canonical WGSL blend
/// kernel and its binding/entry-point body.
pub(crate) fn composite_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!("{BLEND_KERNEL_WGSL}\n{COMPOSITE_BODY_WGSL}"))
}

/// Build the routed-matte compositor from the exact same WGSL blend kernel.
pub(crate) fn matte_composite_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!("{BLEND_KERNEL_WGSL}\n{MATTE_COMPOSITE_BODY_WGSL}"))
}

/// Build the composition host (copy/present/bus) from the exact same WGSL
/// blend kernel: the B8 bus blend family is the one kernel, not a copy.
pub(crate) fn composition_host_shader_source() -> Cow<'static, str> {
    Cow::Owned(format!("{BLEND_KERNEL_WGSL}\n{COMPOSITION_HOST_BODY_WGSL}"))
}

fn soft_light(backdrop: f32, source: f32) -> f32 {
    if source <= 0.5 {
        backdrop - (1.0 - 2.0 * source) * backdrop * (1.0 - backdrop)
    } else {
        // W3C/PDF separable Soft Light. The low-backdrop branch is the
        // specified cubic polynomial, not an approximation by sqrt.
        let d = if backdrop <= 0.25 {
            ((16.0 * backdrop - 12.0) * backdrop + 4.0) * backdrop
        } else {
            backdrop.sqrt()
        };
        backdrop + (2.0 * source - 1.0) * (d - backdrop)
    }
}

fn dodge(backdrop: f32, source: f32) -> f32 {
    if backdrop <= 0.0 {
        0.0
    } else if source >= 1.0 {
        1.0
    } else {
        (backdrop / (1.0 - source).max(1.0e-6)).min(1.0)
    }
}

fn burn(backdrop: f32, source: f32) -> f32 {
    if backdrop >= 1.0 {
        1.0
    } else if source <= 0.0 {
        0.0
    } else {
        1.0 - ((1.0 - backdrop) / source.max(1.0e-6)).min(1.0)
    }
}

// The B8 additions (codes 15..=24) are derived from BENDR (MIT, © 2026
// Steve Blythe), rewritten for this kernel's linear-light [0, 1] contract.

/// Vivid Light is the canonical burn/dodge split, composed from the same
/// guarded channels Dodge and Burn already use.
fn vivid_light(backdrop: f32, source: f32) -> f32 {
    if source <= 0.5 {
        burn(backdrop, 2.0 * source)
    } else {
        dodge(backdrop, 2.0 * source - 1.0)
    }
}

/// Bitwise meeting of two 8-bit code values. The codes are the *stored*
/// sRGB bytes — BENDR XORs what the framebuffer holds — so the law
/// round-trips through the transfer curve. Rounding at the code lattice
/// makes the quantization exact for texture-sourced pixels on both the CPU
/// and every GPU backend; a truncating linear-domain quantizer would flip a
/// bit whenever the two decodes disagree by one ulp.
fn blend_linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn blend_srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn bits(backdrop: f32, source: f32, and_mode: bool) -> f32 {
    let ix = (blend_linear_to_srgb(backdrop.clamp(0.0, 1.0)) * 255.0).round() as u32;
    let iy = (blend_linear_to_srgb(source.clamp(0.0, 1.0)) * 255.0).round() as u32;
    let r = if and_mode { ix & iy } else { ix ^ iy };
    blend_srgb_to_linear(r as f32 / 255.0)
}

/// The branchless HSV pair the four component-swap modes share, transcribed
/// scalar-for-scalar from the WGSL kernel including its 1e-10 pole guards.
fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    // step(c.b, c.g) selects between vec4(c.bg, K.wz) and vec4(c.gb, K.xy).
    let (px, py, pz, pw) = if c[1] >= c[2] {
        (c[1], c[2], 0.0_f32, -1.0_f32 / 3.0)
    } else {
        (c[2], c[1], -1.0_f32, 2.0_f32 / 3.0)
    };
    // step(p.x, c.r) selects between vec4(p.xyw, c.r) and vec4(c.r, p.yzx).
    let (qx, qy, qz, qw) = if c[0] >= px {
        (c[0], py, pz, px)
    } else {
        (px, py, pw, c[0])
    };
    let d = qx - qw.min(qy);
    [
        (qz + (qw - qy) / (6.0 * d + 1.0e-10)).abs(),
        d / (qx + 1.0e-10),
        qx,
    ]
}

fn hsv_to_rgb(c: [f32; 3]) -> [f32; 3] {
    let k = [0.0_f32, 2.0 / 3.0, 1.0 / 3.0];
    std::array::from_fn(|channel| {
        let p = ((c[0] + k[channel]).fract() * 6.0 - 3.0).abs();
        c[2] * (1.0 + c[1] * ((p - 1.0).clamp(0.0, 1.0) - 1.0))
    })
}

/// Evaluate one blend kernel in linear light. `AlphaCut` is a Porter-Duff
/// operation rather than a color blend, so its RGB kernel is the unchanged
/// destination; [`composite_straight`] supplies its alpha law. The four HSV
/// component-swap modes are non-separable and take the whole-triple path;
/// every other mode is separable per channel.
pub(crate) fn blend_rgb(mode: BlendMode, backdrop: [f32; 3], source: [f32; 3]) -> [f32; 3] {
    if matches!(
        mode,
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity
    ) {
        let hb = rgb_to_hsv(backdrop.map(|v| v.clamp(0.0, 1.0)));
        let hs = rgb_to_hsv(source.map(|v| v.clamp(0.0, 1.0)));
        let swapped = match mode {
            BlendMode::Hue => [hs[0], hb[1], hb[2]],
            BlendMode::Saturation => [hb[0], hs[1], hb[2]],
            BlendMode::Color => [hs[0], hs[1], hb[2]],
            _ => [hb[0], hb[1], hs[2]],
        };
        return hsv_to_rgb(swapped).map(|v| v.clamp(0.0, 1.0));
    }
    std::array::from_fn(|channel| {
        let b = backdrop[channel].clamp(0.0, 1.0);
        let s = source[channel].clamp(0.0, 1.0);
        let result = match mode {
            BlendMode::Normal => s,
            BlendMode::Screen => 1.0 - (1.0 - b) * (1.0 - s),
            BlendMode::Multiply => b * s,
            BlendMode::Difference => (b - s).abs(),
            BlendMode::Add => b + s,
            BlendMode::Subtract => b - s,
            BlendMode::Darken => b.min(s),
            BlendMode::Lighten => b.max(s),
            BlendMode::Overlay => {
                if b <= 0.5 {
                    2.0 * b * s
                } else {
                    1.0 - 2.0 * (1.0 - b) * (1.0 - s)
                }
            }
            BlendMode::SoftLight => soft_light(b, s),
            BlendMode::HardLight => {
                if s <= 0.5 {
                    2.0 * b * s
                } else {
                    1.0 - 2.0 * (1.0 - b) * (1.0 - s)
                }
            }
            BlendMode::Exclusion => b + s - 2.0 * b * s,
            BlendMode::Dodge => dodge(b, s),
            BlendMode::Burn => burn(b, s),
            BlendMode::AlphaCut => b,
            BlendMode::VividLight => vivid_light(b, s),
            BlendMode::PinLight => {
                if s <= 0.5 {
                    b.min(2.0 * s)
                } else {
                    b.max(2.0 * s - 1.0)
                }
            }
            BlendMode::Divide => b / s.max(1.0e-6),
            // Wrap Add is the analogue overflow: the 1.5 gain makes the sum
            // wrap before the source reaches full scale.
            BlendMode::WrapAdd => (b + s * 1.5).fract(),
            BlendMode::Xor => bits(b, s, false),
            BlendMode::And => bits(b, s, true),
            BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity => {
                unreachable!("non-separable modes take the HSV path")
            }
        };
        result.clamp(0.0, 1.0)
    })
}

/// Resolve one layer in straight-alpha, linear-light source-over. Alpha Cut
/// is explicitly destination-out. Zero-alpha endpoints bypass the generic
/// equation so transparent bottoms, hidden RGB, and fringe colors are exact.
pub(crate) fn composite_straight(
    mode: BlendMode,
    base: [f32; 4],
    overlay: [f32; 4],
    opacity: f32,
) -> [f32; 4] {
    let base_alpha = base[3].clamp(0.0, 1.0);
    let source_alpha = opacity.clamp(0.0, 1.0) * overlay[3].clamp(0.0, 1.0);

    if mode == BlendMode::AlphaCut {
        if source_alpha <= 0.0 {
            return [base[0], base[1], base[2], base_alpha];
        }
        let output_alpha = base_alpha * (1.0 - source_alpha);
        if output_alpha <= 1.0e-6 {
            return [0.0; 4];
        }
        return [base[0], base[1], base[2], output_alpha];
    }
    if source_alpha <= 0.0 {
        return [base[0], base[1], base[2], base_alpha];
    }
    if base_alpha <= 0.0 {
        return [overlay[0], overlay[1], overlay[2], source_alpha];
    }

    let blended = blend_rgb(
        mode,
        [base[0], base[1], base[2]],
        [overlay[0], overlay[1], overlay[2]],
    );
    let output_alpha = source_alpha + base_alpha * (1.0 - source_alpha);
    let output_rgb: [f32; 3] = std::array::from_fn(|channel| {
        let premultiplied = base[channel] * base_alpha * (1.0 - source_alpha)
            + overlay[channel] * source_alpha * (1.0 - base_alpha)
            + blended[channel] * base_alpha * source_alpha;
        premultiplied / output_alpha.max(1.0e-6)
    });
    [output_rgb[0], output_rgb[1], output_rgb[2], output_alpha]
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_RGB: [f32; 3] = [0.2, 0.4, 0.8];
    const SOURCE_RGB: [f32; 3] = [0.7, 0.3, 0.6];

    fn close(actual: [f32; 4], expected: [f32; 4]) {
        for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "channel {channel}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn all_twenty_five_modes_have_frozen_opaque_linear_reference_vectors() {
        // Rows 0..=14 are the pre-B8 vectors, byte for byte: the appended
        // family must never move an existing law.
        let expected = [
            [0.7, 0.3, 0.6, 1.0],
            [0.76, 0.58, 0.92, 1.0],
            [0.14, 0.12, 0.48, 1.0],
            [0.5, 0.1, 0.2, 1.0],
            [0.9, 0.7, 1.0, 1.0],
            [0.0, 0.1, 0.2, 1.0],
            [0.2, 0.3, 0.6, 1.0],
            [0.7, 0.4, 0.8, 1.0],
            [0.28, 0.24, 0.84, 1.0],
            [0.2992, 0.304, 0.818_885_45, 1.0],
            [0.52, 0.24, 0.84, 1.0],
            [0.62, 0.46, 0.44, 1.0],
            [2.0 / 3.0, 4.0 / 7.0, 1.0, 1.0],
            [0.0, 0.0, 2.0 / 3.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 1.0, 1.0],
            [0.4, 0.4, 0.8, 1.0],
            [2.0 / 7.0, 1.0, 1.0, 1.0],
            [0.25, 0.85, 0.7, 1.0],
            [0.381_326_1, 0.049_706_563, 0.025_186_857, 1.0],
            [0.097_587_36, 0.215_860_53, 0.545_724_6, 1.0],
            [0.8, 0.2, 0.65, 1.0],
            [12.0 / 35.0, 52.0 / 105.0, 0.8, 1.0],
            [0.8, 12.0 / 35.0, 24.0 / 35.0, 1.0],
            [0.175, 0.35, 0.7, 1.0],
        ];
        assert_eq!(expected.len(), BlendMode::ALL.len());
        for (mode, expected) in BlendMode::ALL.into_iter().zip(expected) {
            close(
                composite_straight(
                    mode,
                    [BASE_RGB[0], BASE_RGB[1], BASE_RGB[2], 1.0],
                    [SOURCE_RGB[0], SOURCE_RGB[1], SOURCE_RGB[2], 1.0],
                    1.0,
                ),
                expected,
            );
        }
    }

    #[test]
    fn all_twenty_five_modes_have_frozen_half_alpha_linear_reference_vectors() {
        // Rows 0..=14 are the pre-B8 vectors, byte for byte.
        let expected = [
            [0.472_727_27, 0.345_454_54, 0.690_909_1, 0.6875],
            [0.489_090_92, 0.421_818_2, 0.778_181_8, 0.6875],
            [0.32, 0.296_363_65, 0.658_181_85, 0.6875],
            [0.418_181_8, 0.290_909_1, 0.581_818_16, 0.6875],
            [0.527_272_7, 0.454_545_47, 0.8, 0.6875],
            [0.281_818_18, 0.290_909_1, 0.581_818_16, 0.6875],
            [0.336_363_64, 0.345_454_54, 0.690_909_1, 0.6875],
            [0.472_727_27, 0.372_727_27, 0.745_454_55, 0.6875],
            [0.358_181_8, 0.329_090_92, 0.756_363_63, 0.6875],
            [0.363_418_2, 0.346_545_46, 0.750_605_1, 0.6875],
            [0.423_636_38, 0.329_090_92, 0.756_363_63, 0.6875],
            [0.450_909_08, 0.389_090_9, 0.647_272_7, 0.6875],
            [0.463_636_37, 0.419_480_53, 0.8, 0.6875],
            [0.281_818_18, 0.263_636_35, 0.709_090_9, 0.6875],
            [0.2, 0.4, 0.8, 0.3125],
            [0.372_727_27, 0.263_636_38, 0.8, 0.6875],
            [0.390_909_08, 0.372_727_3, 0.745_454_6, 0.6875],
            [0.359_740_26, 0.536_363_66, 0.8, 0.6875],
            [0.35, 0.495_454_58, 0.718_181_85, 0.6875],
            [0.385_816_22, 0.277_192_7, 0.534_141_9, 0.6875],
            [0.308_432_9, 0.322_507_4, 0.676_106_7, 0.6875],
            [0.5, 0.318_181_8, 0.704_545_44, 0.6875],
            [0.375_324_7, 0.398_701_3, 0.745_454_6, 0.6875],
            [0.5, 0.357_142_9, 0.714_285_7, 0.6875],
            [0.329_545_47, 0.359_090_92, 0.718_181_85, 0.6875],
        ];
        assert_eq!(expected.len(), BlendMode::ALL.len());
        for (mode, expected) in BlendMode::ALL.into_iter().zip(expected) {
            close(
                composite_straight(
                    mode,
                    [BASE_RGB[0], BASE_RGB[1], BASE_RGB[2], 0.5],
                    [SOURCE_RGB[0], SOURCE_RGB[1], SOURCE_RGB[2], 0.5],
                    0.75,
                ),
                expected,
            );
        }
    }

    #[test]
    fn curated_blend_reference_matrix_has_a_frozen_signature() {
        let cases = [
            (
                [BASE_RGB[0], BASE_RGB[1], BASE_RGB[2], 1.0],
                [SOURCE_RGB[0], SOURCE_RGB[1], SOURCE_RGB[2], 1.0],
                1.0,
            ),
            ([0.91, 0.17, 0.63, 0.0], [0.12, 0.82, 0.41, 0.5], 1.0),
            (
                [BASE_RGB[0], BASE_RGB[1], BASE_RGB[2], 0.5],
                [SOURCE_RGB[0], SOURCE_RGB[1], SOURCE_RGB[2], 0.5],
                0.75,
            ),
        ];
        let mut signature = 0xcbf2_9ce4_8422_2325_u64;
        for mode in BlendMode::ALL {
            for (base, overlay, opacity) in cases {
                for channel in composite_straight(mode, base, overlay, opacity) {
                    let quantized = (channel.clamp(0.0, 1.0) * 65_535.0).round() as u16;
                    for byte in quantized.to_le_bytes() {
                        signature ^= u64::from(byte);
                        signature = signature.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
            }
        }
        // Re-pinned by the B8 blend audit: the matrix now covers all
        // twenty-five modes. The first fifteen modes' contribution is proven
        // unchanged by the two frozen vector tests above, whose rows 0..=14
        // kept their pre-B8 literals byte for byte.
        assert_eq!(signature, 0xd650_b746_679e_f16e);
    }

    #[test]
    fn transparent_and_zero_source_alpha_endpoints_are_exact_for_every_mode() {
        let transparent_base = [0.91, 0.17, 0.63, 0.0];
        let half_source = [0.12, 0.82, 0.41, 0.5];
        let covered_base = [0.2, 0.4, 0.8, 0.625];
        let zero_source = [0.99, 0.01, 0.77, 0.0];

        for mode in BlendMode::ALL {
            let over_empty = composite_straight(mode, transparent_base, half_source, 1.0);
            if mode == BlendMode::AlphaCut {
                close(over_empty, [0.0; 4]);
            } else {
                close(over_empty, half_source);
            }
            close(
                composite_straight(mode, covered_base, zero_source, 1.0),
                covered_base,
            );
        }
    }

    #[test]
    fn alpha_cut_full_coverage_is_canonical_transparent_black() {
        close(
            composite_straight(
                BlendMode::AlphaCut,
                [0.2, 0.4, 0.8, 1.0],
                [0.7, 0.3, 0.6, 1.0],
                1.0,
            ),
            [0.0; 4],
        );
        close(
            composite_straight(
                BlendMode::AlphaCut,
                [0.2, 0.4, 0.8, 0.625],
                [0.7, 0.3, 0.6, 0.0],
                1.0,
            ),
            [0.2, 0.4, 0.8, 0.625],
        );
    }

    #[test]
    fn soft_light_uses_the_w3c_cubic_and_dodge_burn_are_guarded() {
        let cubic = blend_rgb(BlendMode::SoftLight, [0.1; 3], [1.0; 3]);
        assert!((cubic[0] - 0.296).abs() <= 1.0e-6);
        assert_eq!(blend_rgb(BlendMode::Dodge, [0.0; 3], [1.0; 3]), [0.0; 3]);
        assert_eq!(blend_rgb(BlendMode::Dodge, [0.4; 3], [1.0; 3]), [1.0; 3]);
        assert_eq!(blend_rgb(BlendMode::Burn, [1.0; 3], [0.0; 3]), [1.0; 3]);
        assert_eq!(blend_rgb(BlendMode::Burn, [0.4; 3], [0.0; 3]), [0.0; 3]);
    }

    #[test]
    fn both_shader_bodies_receive_one_identical_blend_kernel() {
        let ordinary = composite_shader_source();
        let matte = matte_composite_shader_source();
        for shader in [&ordinary, &matte] {
            assert_eq!(shader.matches("fn blend_rgb(").count(), 1);
            assert_eq!(shader.matches("fn composite_straight_alpha(").count(), 1);
            assert!(shader.contains("W3C/PDF Soft Light"));
            assert!(shader.contains("case 14u") || shader.contains("BLEND_ALPHA_CUT"));
            // The B8 family rides the same single kernel: the highest
            // appended code and the shared HSV pair appear exactly once.
            assert_eq!(shader.matches("fn hsv_swap_rgb(").count(), 1);
            assert_eq!(shader.matches("fn blend_rgb_to_hsv(").count(), 1);
            assert!(shader.contains("case 21u, 22u, 23u, 24u"));
        }
    }

    #[test]
    fn appended_blend_family_matches_its_analytic_laws() {
        // Vivid Light at exactly half source is the burn of a doubled full
        // source: the identity on the backdrop.
        assert_eq!(
            blend_rgb(BlendMode::VividLight, [0.3; 3], [0.5; 3]),
            [0.3; 3]
        );
        // Above half it is the dodge of the re-centered source.
        let vivid = blend_rgb(BlendMode::VividLight, [0.2; 3], [0.75; 3]);
        assert!((vivid[0] - 0.4).abs() <= 1.0e-6);
        // Pin Light keeps the backdrop between the doubled bounds.
        assert_eq!(blend_rgb(BlendMode::PinLight, [0.3; 3], [0.5; 3]), [0.3; 3]);
        let pin_high = blend_rgb(BlendMode::PinLight, [0.1; 3], [0.9; 3]);
        assert!((pin_high[0] - 0.8).abs() <= 1.0e-6);
        let pin_low = blend_rgb(BlendMode::PinLight, [0.9; 3], [0.1; 3]);
        assert!((pin_low[0] - 0.2).abs() <= 1.0e-6);
        // Divide is guarded at the zero source and clamps at unity.
        assert_eq!(blend_rgb(BlendMode::Divide, [0.5; 3], [0.0; 3]), [1.0; 3]);
        assert_eq!(blend_rgb(BlendMode::Divide, [0.3; 3], [0.6; 3]), [0.5; 3]);
        // Wrap Add is the analogue overflow: full scale wraps to zero, and
        // the 1.5 gain overflows before the source reaches full scale.
        assert_eq!(blend_rgb(BlendMode::WrapAdd, [1.0; 3], [0.0; 3]), [0.0; 3]);
        let wrapped = blend_rgb(BlendMode::WrapAdd, [0.5; 3], [0.5; 3]);
        assert!((wrapped[0] - 0.25).abs() <= 1.0e-6);
        // The bitwise pair operates on the stored 8-bit sRGB code bytes:
        // XOR of a value with itself is zero, AND with itself is the value
        // itself up to one code of quantization.
        assert_eq!(blend_rgb(BlendMode::Xor, [0.7; 3], [0.7; 3]), [0.0; 3]);
        let and_self = blend_rgb(BlendMode::And, [0.7; 3], [0.7; 3]);
        assert!((and_self[0] - 0.7).abs() <= 5.0e-3, "{}", and_self[0]);
        // Luminosity keeps the backdrop's hue and saturation while taking the
        // source's value; a grey source therefore greys nothing but the level.
        let lum = blend_rgb(BlendMode::Luminosity, [0.2, 0.4, 0.8], [0.55; 3]);
        assert!(
            (lum[2] - 0.55).abs() <= 1.0e-5,
            "value channel follows source"
        );
        assert!(
            (lum[0] / lum[2] - 0.2 / 0.8).abs() <= 1.0e-4,
            "hue ratio kept"
        );
        // Hue with a grey backdrop stays grey: no saturation to color.
        let hue = blend_rgb(BlendMode::Hue, [0.5; 3], [0.9, 0.1, 0.2]);
        for channel in hue {
            assert!((channel - 0.5).abs() <= 1.0e-5);
        }
    }
}
