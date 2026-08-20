// Canonical linear-light, straight-alpha layer blend laws. This source is
// prepended verbatim to both composite shader bodies by renderer::blend.

const BLEND_ALPHA_CUT: u32 = 14u;
const BLEND_EPSILON: f32 = 0.000001;

fn soft_light_channel(backdrop: f32, source: f32) -> f32 {
    if source <= 0.5 {
        return backdrop
            - (1.0 - 2.0 * source) * backdrop * (1.0 - backdrop);
    }

    // W3C/PDF Soft Light uses this cubic branch for dark backdrops.
    var d = sqrt(backdrop);
    if backdrop <= 0.25 {
        d = ((16.0 * backdrop - 12.0) * backdrop + 4.0) * backdrop;
    }
    return backdrop + (2.0 * source - 1.0) * (d - backdrop);
}

fn dodge_channel(backdrop: f32, source: f32) -> f32 {
    if backdrop <= 0.0 {
        return 0.0;
    }
    if source >= 1.0 {
        return 1.0;
    }
    return min(1.0, backdrop / max(1.0 - source, BLEND_EPSILON));
}

fn burn_channel(backdrop: f32, source: f32) -> f32 {
    if backdrop >= 1.0 {
        return 1.0;
    }
    if source <= 0.0 {
        return 0.0;
    }
    return 1.0 - min(1.0, (1.0 - backdrop) / max(source, BLEND_EPSILON));
}

// The B8 additions (codes 15..=24) are derived from BENDR (MIT, © 2026
// Steve Blythe), rewritten for this kernel's linear-light [0, 1] contract.

// Vivid Light is the canonical burn/dodge split, composed from the same
// guarded channels Dodge and Burn already use.
fn vivid_light_channel(backdrop: f32, source: f32) -> f32 {
    if source <= 0.5 {
        return burn_channel(backdrop, 2.0 * source);
    }
    return dodge_channel(backdrop, 2.0 * source - 1.0);
}

// Bitwise meeting of two 8-bit code values. The codes are the *stored* sRGB
// bytes — BENDR XORs what the framebuffer holds — so the law round-trips
// through the transfer curve. Rounding at the code lattice makes the
// quantization exact for texture-sourced pixels on every backend; a
// truncating linear-domain quantizer would flip a bit whenever CPU and GPU
// decodes disagree by one ulp.
fn blend_linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        return value * 12.92;
    }
    return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

fn blend_srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn bits_channel(backdrop: f32, source: f32, and_mode: u32) -> f32 {
    let ix = u32(round(blend_linear_to_srgb(clamp(backdrop, 0.0, 1.0)) * 255.0));
    let iy = u32(round(blend_linear_to_srgb(clamp(source, 0.0, 1.0)) * 255.0));
    var r = ix ^ iy;
    if and_mode != 0u {
        r = ix & iy;
    }
    return blend_srgb_to_linear(f32(r) / 255.0);
}

fn bits_rgb(backdrop: vec3f, source: vec3f, and_mode: u32) -> vec3f {
    return vec3f(
        bits_channel(backdrop.r, source.r, and_mode),
        bits_channel(backdrop.g, source.g, and_mode),
        bits_channel(backdrop.b, source.b, and_mode),
    );
}

// The branchless HSV pair the four component-swap modes share. Inputs are
// clamped to [0, 1] before conversion; the 1e-10 guards keep the grey and
// black poles finite.
fn blend_rgb_to_hsv(c: vec3f) -> vec3f {
    let k = vec4f(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4f(c.bg, k.wz), vec4f(c.gb, k.xy), step(c.b, c.g));
    let q = mix(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    return vec3f(
        abs(q.z + (q.w - q.y) / (6.0 * d + 0.0000000001)),
        d / (q.x + 0.0000000001),
        q.x,
    );
}

fn blend_hsv_to_rgb(c: vec3f) -> vec3f {
    let p = abs(fract(c.xxx + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - vec3f(3.0));
    return c.z * mix(vec3f(1.0), clamp(p - vec3f(1.0), vec3f(0.0), vec3f(1.0)), c.y);
}

fn hsv_swap_rgb(mode: u32, backdrop: vec3f, source: vec3f) -> vec3f {
    let hb = blend_rgb_to_hsv(clamp(backdrop, vec3f(0.0), vec3f(1.0)));
    let hs = blend_rgb_to_hsv(clamp(source, vec3f(0.0), vec3f(1.0)));
    var swapped: vec3f;
    switch mode {
        case 21u: { swapped = vec3f(hs.x, hb.y, hb.z); }
        case 22u: { swapped = vec3f(hb.x, hs.y, hb.z); }
        case 23u: { swapped = vec3f(hs.x, hs.y, hb.z); }
        default: { swapped = vec3f(hb.x, hb.y, hs.z); }
    }
    return blend_hsv_to_rgb(swapped);
}

fn blend_rgb(mode: u32, backdrop: vec3f, source: vec3f) -> vec3f {
    var blended: vec3f;
    switch mode {
        case 1u: {
            blended = vec3f(1.0) - (vec3f(1.0) - backdrop) * (vec3f(1.0) - source);
        }
        case 2u: { blended = backdrop * source; }
        case 3u: { blended = abs(backdrop - source); }
        case 4u: { blended = backdrop + source; }
        case 5u: { blended = backdrop - source; }
        case 6u: { blended = min(backdrop, source); }
        case 7u: { blended = max(backdrop, source); }
        case 8u: {
            blended = select(
                vec3f(1.0) - 2.0 * (vec3f(1.0) - backdrop) * (vec3f(1.0) - source),
                2.0 * backdrop * source,
                backdrop <= vec3f(0.5),
            );
        }
        case 9u: {
            blended = vec3f(
                soft_light_channel(backdrop.r, source.r),
                soft_light_channel(backdrop.g, source.g),
                soft_light_channel(backdrop.b, source.b),
            );
        }
        case 10u: {
            blended = select(
                vec3f(1.0) - 2.0 * (vec3f(1.0) - backdrop) * (vec3f(1.0) - source),
                2.0 * backdrop * source,
                source <= vec3f(0.5),
            );
        }
        case 11u: { blended = backdrop + source - 2.0 * backdrop * source; }
        case 12u: {
            blended = vec3f(
                dodge_channel(backdrop.r, source.r),
                dodge_channel(backdrop.g, source.g),
                dodge_channel(backdrop.b, source.b),
            );
        }
        case 13u: {
            blended = vec3f(
                burn_channel(backdrop.r, source.r),
                burn_channel(backdrop.g, source.g),
                burn_channel(backdrop.b, source.b),
            );
        }
        case 15u: {
            blended = vec3f(
                vivid_light_channel(backdrop.r, source.r),
                vivid_light_channel(backdrop.g, source.g),
                vivid_light_channel(backdrop.b, source.b),
            );
        }
        case 16u: {
            let b = clamp(backdrop, vec3f(0.0), vec3f(1.0));
            let s = clamp(source, vec3f(0.0), vec3f(1.0));
            blended = select(
                max(b, 2.0 * s - vec3f(1.0)),
                min(b, 2.0 * s),
                s <= vec3f(0.5),
            );
        }
        case 17u: {
            let b = clamp(backdrop, vec3f(0.0), vec3f(1.0));
            let s = clamp(source, vec3f(0.0), vec3f(1.0));
            blended = b / max(s, vec3f(BLEND_EPSILON));
        }
        case 18u: {
            // Wrap Add is the analogue overflow: the 1.5 gain makes the sum
            // wrap before the source reaches full scale.
            let b = clamp(backdrop, vec3f(0.0), vec3f(1.0));
            let s = clamp(source, vec3f(0.0), vec3f(1.0));
            blended = fract(b + s * 1.5);
        }
        case 19u: { blended = bits_rgb(backdrop, source, 0u); }
        case 20u: { blended = bits_rgb(backdrop, source, 1u); }
        case 21u, 22u, 23u, 24u: { blended = hsv_swap_rgb(mode, backdrop, source); }
        default: { blended = source; }
    }
    return clamp(blended, vec3f(0.0), vec3f(1.0));
}

fn composite_straight_alpha(
    mode: u32,
    base: vec4f,
    overlay: vec4f,
    opacity: f32,
) -> vec4f {
    let base_alpha = clamp(base.a, 0.0, 1.0);
    let source_alpha = clamp(opacity, 0.0, 1.0) * clamp(overlay.a, 0.0, 1.0);

    // Alpha Cut is destination-out. It never creates content over an empty
    // destination and keeps destination RGB straight while reducing coverage.
    if mode == BLEND_ALPHA_CUT {
        if source_alpha <= 0.0 {
            return vec4f(base.rgb, base_alpha);
        }
        let output_alpha = base_alpha * (1.0 - source_alpha);
        if output_alpha <= BLEND_EPSILON {
            return vec4f(0.0);
        }
        return vec4f(base.rgb, output_alpha);
    }

    // Exact endpoints protect hidden straight RGB and eliminate fringes.
    if source_alpha <= 0.0 {
        return vec4f(base.rgb, base_alpha);
    }
    if base_alpha <= 0.0 {
        return vec4f(overlay.rgb, source_alpha);
    }

    let blended = blend_rgb(mode, base.rgb, overlay.rgb);
    let output_alpha = source_alpha + base_alpha * (1.0 - source_alpha);
    let output_premultiplied =
        base.rgb * base_alpha * (1.0 - source_alpha)
        + overlay.rgb * source_alpha * (1.0 - base_alpha)
        + blended * base_alpha * source_alpha;
    return vec4f(
        output_premultiplied / max(output_alpha, BLEND_EPSILON),
        output_alpha,
    );
}
