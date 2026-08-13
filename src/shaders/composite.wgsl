// Composites an overlay layer onto a base layer using blend modes.
// Each composite pass blends one layer onto the accumulated result.

struct CompositeUniforms {
    opacity: f32,
    blend_mode: u32,  // 0=normal, 1=screen, 2=multiply, 3=difference
    _pad: vec2f,
};

@group(0) @binding(0) var base_tex: texture_2d<f32>;
@group(0) @binding(1) var overlay_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(1) @binding(0) var<uniform> uniforms: CompositeUniforms;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let base = textureSample(base_tex, samp, uv);
    let overlay = textureSample(overlay_tex, samp, uv);

    var blended: vec3f;

    switch uniforms.blend_mode {
        case 1u: {
            // Screen: 1 - (1-base)*(1-overlay)
            blended = vec3f(1.0) - (vec3f(1.0) - base.rgb) * (vec3f(1.0) - overlay.rgb);
        }
        case 2u: {
            // Multiply
            blended = base.rgb * overlay.rgb;
        }
        case 3u: {
            // Difference
            blended = abs(base.rgb - overlay.rgb);
        }
        default: {
            // Normal (source-over)
            blended = overlay.rgb;
        }
    }

    // Mix based on opacity
    let result = mix(base.rgb, blended, uniforms.opacity * overlay.a);
    return vec4f(result, max(base.a, overlay.a * uniforms.opacity));
}
