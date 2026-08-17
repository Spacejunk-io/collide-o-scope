// Small format-agnostic host operations for the advanced composition engine.
// Every sampled image is straight-alpha linear RGBA. The bus law converts to
// premultiplied form for interpolation/source-over, then returns straight RGB
// so subsequent rack and matte operations retain the engine-wide contract.

struct BusUniforms {
    crossfade: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

// Static 8x8 Bayer order. Dither belongs only to final Advanced audience
// presentation. Compat8 temporal history/feedback uses the separately
// prepared fs_copy pipeline, so recursive memory receives ordinary format
// conversion and never accumulates this pattern. The final pattern is
// frame-invariant: a fixed input produces byte-identical presentation.
const BAYER_8X8: array<u32, 64> = array<u32, 64>(
     0u, 48u, 12u, 60u,  3u, 51u, 15u, 63u,
    32u, 16u, 44u, 28u, 35u, 19u, 47u, 31u,
     8u, 56u,  4u, 52u, 11u, 59u,  7u, 55u,
    40u, 24u, 36u, 20u, 43u, 27u, 39u, 23u,
     2u, 50u, 14u, 62u,  1u, 49u, 13u, 61u,
    34u, 18u, 46u, 30u, 33u, 17u, 45u, 29u,
    10u, 58u,  6u, 54u,  9u, 57u,  5u, 53u,
    42u, 26u, 38u, 22u, 41u, 25u, 37u, 21u,
);

@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var tex_program: texture_2d<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(1) @binding(0) var<uniform> bus: BusUniforms;

fn premultiply(straight: vec4f) -> vec4f {
    let alpha = clamp(straight.a, 0.0, 1.0);
    return vec4f(straight.rgb * alpha, alpha);
}

fn premultiplied_over(top: vec4f, under: vec4f) -> vec4f {
    return top + under * (1.0 - top.a);
}

fn straight_from_premultiplied(value: vec4f) -> vec4f {
    let alpha = clamp(value.a, 0.0, 1.0);
    if alpha <= 0.000001 {
        return vec4f(0.0);
    }
    return vec4f(value.rgb / alpha, alpha);
}

fn linear_to_srgb(value: vec3f) -> vec3f {
    let low = value * 12.92;
    let high = 1.055 * pow(value, vec3f(1.0 / 2.4)) - 0.055;
    return select(high, low, value <= vec3f(0.0031308));
}

fn srgb_to_linear(value: vec3f) -> vec3f {
    let low = value / 12.92;
    let high = pow((value + 0.055) / 1.055, vec3f(2.4));
    return select(high, low, value <= vec3f(0.04045));
}

fn ordered_compat8_dither(position: vec2f) -> f32 {
    let x = u32(position.x) & 7u;
    let y = u32(position.y) & 7u;
    let rank = BAYER_8X8[y * 8u + x];
    return (f32(rank) + 0.5) / 64.0;
}

@fragment
fn fs_copy(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSample(tex_a, linear_sampler, uv);
}

// This entry point is intentionally valid only for an Rgba8UnormSrgb target.
// The render target applies the final sRGB transfer, so move into encoded
// space for the one-code-value dither and back to linear for the attachment.
// Alpha is coverage, not an sRGB channel; retain it without color dither.
@fragment
fn fs_present(
    @location(0) uv: vec2f,
    @builtin(position) position: vec4f,
) -> @location(0) vec4f {
    let straight = textureSample(tex_a, linear_sampler, uv);
    let linear = clamp(straight.rgb, vec3f(0.0), vec3f(1.0));
    let encoded = linear_to_srgb(linear);
    let scaled = encoded * 255.0;
    let lower = floor(scaled);
    let fraction = fract(scaled);
    let threshold = vec3f(ordered_compat8_dither(position.xy));
    let promoted = select(vec3f(0.0), vec3f(1.0), fraction >= threshold);
    // Select the two adjacent representable codes directly. Returning their
    // exact decoded centers makes the distribution immune to small
    // implementation differences in the attachment's sRGB transfer.
    let dithered = clamp((lower + promoted) / 255.0, vec3f(0.0), vec3f(1.0));
    return vec4f(srgb_to_linear(dithered), clamp(straight.a, 0.0, 1.0));
}

@fragment
fn fs_bus(@location(0) uv: vec2f) -> @location(0) vec4f {
    let a = premultiply(textureSample(tex_a, linear_sampler, uv));
    let b = premultiply(textureSample(tex_b, linear_sampler, uv));
    let program = premultiply(textureSample(tex_program, linear_sampler, uv));
    let ab = mix(a, b, clamp(bus.crossfade, 0.0, 1.0));
    return straight_from_premultiplied(premultiplied_over(program, ab));
}
