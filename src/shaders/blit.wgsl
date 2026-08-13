// Plain blit: samples the already-opaque audience image onto the output
// window's surface. Letterboxing is handled by the viewport.

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSample(tex, samp, uv);
}
