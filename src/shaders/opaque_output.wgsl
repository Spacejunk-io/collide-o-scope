// Convert the engine's straight-alpha program image into the opaque SDR
// image consumed by preview, projector, Spout, NTSC, and MP4 export.
//
// The source and target are sRGB texture views, so texture sampling decodes
// RGB to linear light and the render target encodes it exactly once. The
// multiplication below is therefore a linear-light composite over black.

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let straight = textureSample(tex, samp, uv);
    let coverage = clamp(straight.a, 0.0, 1.0);
    return vec4f(straight.rgb * coverage, 1.0);
}
