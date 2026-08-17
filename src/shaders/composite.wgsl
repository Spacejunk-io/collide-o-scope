// Two-input compositor bindings and entry point. renderer::blend prepends the
// canonical blend.wgsl kernel before compiling this body.

struct CompositeUniforms {
    opacity: f32,
    blend_mode: u32,
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
    return composite_straight_alpha(
        uniforms.blend_mode,
        base,
        overlay,
        uniforms.opacity,
    );
}
