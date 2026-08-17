// Routed-matte bindings and entry point. renderer::blend prepends the same
// canonical blend.wgsl kernel used by the ordinary compositor.

struct MatteCompositeUniforms {
    opacity: f32,
    blend_mode: u32,
    channel: u32,       // 0=alpha, 1=luma, 2=red, 3=green, 4=blue
    invert: u32,
    amount: f32,
    threshold: f32,
    softness: f32,
    donor_valid: u32,
};

@group(0) @binding(0) var base_tex: texture_2d<f32>;
@group(0) @binding(1) var overlay_tex: texture_2d<f32>;
@group(0) @binding(2) var donor_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(1) @binding(0) var<uniform> uniforms: MatteCompositeUniforms;

fn matte_field(donor: vec4f) -> f32 {
    switch uniforms.channel {
        case 1u: {
            return dot(donor.rgb, vec3f(0.2126, 0.7152, 0.0722));
        }
        case 2u: { return donor.r; }
        case 3u: { return donor.g; }
        case 4u: { return donor.b; }
        default: { return donor.a; }
    }
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let base = textureSample(base_tex, samp, uv);
    let original_overlay = textureSample(overlay_tex, samp, uv);
    let donor = textureSample(donor_tex, samp, uv);

    var shaped = 0.0;
    if uniforms.donor_valid != 0u {
        var field = clamp(matte_field(donor), 0.0, 1.0);
        if uniforms.invert != 0u {
            field = 1.0 - field;
        }
        if uniforms.softness <= 0.0000001 {
            shaped = select(0.0, 1.0, field >= uniforms.threshold);
        } else {
            let half_width = uniforms.softness * 0.5;
            shaped = smoothstep(
                uniforms.threshold - half_width,
                uniforms.threshold + half_width,
                field,
            );
        }
    } else {
        // A missing/rejected route is a defined zero field. Suppress invert,
        // but keep amount continuous so amount=0 is a true dry bypass.
        shaped = 0.0;
    }
    let admission = mix(1.0, shaped, clamp(uniforms.amount, 0.0, 1.0));
    // RGB remains straight; only alpha is admitted by the matte.
    let overlay = vec4f(original_overlay.rgb, original_overlay.a * admission);

    return composite_straight_alpha(
        uniforms.blend_mode,
        base,
        overlay,
        uniforms.opacity,
    );
}
