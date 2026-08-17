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
