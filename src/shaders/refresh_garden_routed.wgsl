// Dedicated post-temporal Refresh Garden pass.
//
// This pass deliberately binds exactly three sampled textures: pre-Garden
// current, the established Compat8 feedback carrier, and one stable routed
// signal (selected-layer matte image or low-resolution R8 motion scalar).

struct RoutedGardenUniforms {
    feedback_values: vec4<f32>, // zoom, rotation degrees, validity, reserved
    garden_values: vec4<f32>,   // amount, threshold, softness, decay
    garden_modes: vec4<u32>,    // observation ticks, packed runtime, gate, reserved
};

@group(0) @binding(0) var current_tex: texture_2d<f32>;
@group(0) @binding(1) var feedback_tex: texture_2d<f32>;
@group(0) @binding(2) var signal_tex: texture_2d<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(1) @binding(0) var<uniform> u: RoutedGardenUniforms;

fn premultiply(straight: vec4<f32>) -> vec4<f32> {
    let alpha = clamp(straight.a, 0.0, 1.0);
    return vec4<f32>(straight.rgb * alpha, alpha);
}

fn straight_from_premultiplied(value: vec4<f32>) -> vec4<f32> {
    let alpha = clamp(value.a, 0.0, 1.0);
    if alpha <= 0.000001 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(value.rgb / alpha, alpha);
}

// Explicit four-load covered-color interpolation is the Advanced alpha law;
// it also makes the physical texture-operation ledger auditable.
fn feedback_premultiplied_linear(uv: vec2<f32>) -> vec4<f32> {
    let dimensions = vec2<i32>(textureDimensions(feedback_tex));
    let coordinate = uv * vec2<f32>(dimensions) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(coordinate));
    let fraction = fract(coordinate);
    let maximum = dimensions - vec2<i32>(1);
    let p00 = clamp(base, vec2<i32>(0), maximum);
    let p10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0), maximum);
    let p01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0), maximum);
    let p11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0), maximum);
    let c00 = premultiply(textureLoad(feedback_tex, p00, 0));
    let c10 = premultiply(textureLoad(feedback_tex, p10, 0));
    let c01 = premultiply(textureLoad(feedback_tex, p01, 0));
    let c11 = premultiply(textureLoad(feedback_tex, p11, 0));
    return mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y);
}

fn opened_gate(signal: f32) -> f32 {
    let threshold = clamp(u.garden_values.y, 0.0, 1.0);
    let softness = clamp(u.garden_values.z, 0.0, 0.5);
    if softness <= 0.000001 {
        return select(0.0, 1.0, signal >= threshold);
    }
    return smoothstep(
        max(threshold - softness, 0.0),
        min(threshold + softness, 1.0),
        clamp(signal, 0.0, 1.0),
    );
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let current = premultiply(textureSampleLevel(current_tex, linear_sampler, uv, 0.0));
    if u.feedback_values.z <= 0.5 {
        return straight_from_premultiplied(current);
    }

    let angle = u.feedback_values.y * 0.017453292519943295;
    let c = cos(angle);
    let s = sin(angle);
    var carrier_uv = uv - vec2<f32>(0.5);
    carrier_uv = vec2<f32>(
        carrier_uv.x * c - carrier_uv.y * s,
        carrier_uv.x * s + carrier_uv.y * c,
    ) / max(u.feedback_values.x, 0.01);
    carrier_uv += vec2<f32>(0.5);
    let inside = all(carrier_uv >= vec2<f32>(0.0))
        && all(carrier_uv <= vec2<f32>(1.0));
    let carrier = feedback_premultiplied_linear(carrier_uv) * select(0.0, 1.0, inside);

    let ticks = u.garden_modes.x;
    if ticks == 0u {
        return straight_from_premultiplied(carrier);
    }

    let routed = textureSampleLevel(signal_tex, linear_sampler, uv, 0.0);
    // Gate 6 is Matte/alpha; Gate 7 is the materialized Motion/red scalar.
    let signal = select(routed.r, routed.a, u.garden_modes.z == 6u);
    let force_refresh = (u.garden_modes.y & (1u << 17u)) != 0u;
    let opened = select(opened_gate(signal), 1.0, force_refresh);
    let admission = clamp(u.garden_values.x * opened, 0.0, 1.0);
    let coefficient = clamp(u.garden_values.w, 0.0, 1.0) * (1.0 - admission);
    let retained = pow(coefficient, f32(ticks));
    var injected = 0.0;
    if admission > 0.0 {
        injected = admission * (1.0 - retained) / max(1.0 - coefficient, 0.000001);
    }
    return straight_from_premultiplied(carrier * retained + current * injected);
}
