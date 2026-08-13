// Plain blit: samples the final composite onto the output window's
// surface. Letterboxing is handled by the viewport; alpha is forced
// opaque so the projector never sees through keyed layers.

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return vec4f(textureSample(tex, samp, uv).rgb, 1.0);
}
