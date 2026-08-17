// Standalone Collision Rack node executor.
//
// `renderer::rack` prepends the canonical blend.wgsl source. Rack textures
// contain linear-light, straight-alpha RGBA16Float pixels. Every node first
// produces a straight-alpha result, applies the selected RGB blend kernel
// while retaining that processed alpha, then interpolates dry/result in
// premultiplied space. This prevents a soft Key or Mask from acquiring dark
// fringes. Alpha Cut is deliberately destination-out instead of an RGB blend.

const RACK_PASSTHROUGH: u32 = 0u;
const RACK_TRANSFORM: u32 = 1u;
const RACK_DIGITAL_COLOR: u32 = 2u;
const RACK_KEY: u32 = 3u;
const RACK_CELLULAR: u32 = 4u;
const RACK_SHIFT: u32 = 5u;
const RACK_GRAIN: u32 = 6u;
const RACK_RECTANGLE_MASK: u32 = 7u;
const RACK_ELLIPSE_MASK: u32 = 8u;
const RACK_IMAGE_MASK: u32 = 9u;
const RACK_DISPLACE: u32 = 10u;

// Displace boundary laws. These codes are permanent and append-only; they
// mirror `visual_rack::DisplaceBoundary::code`.
const DISPLACE_TRANSPARENT: u32 = 0u;
const DISPLACE_MIRROR: u32 = 1u;
const DISPLACE_WRAP: u32 = 2u;
const DISPLACE_HOLD: u32 = 3u;

struct RackUniforms {
    // Node kind, blend code, donor-valid flag, deterministic seed.
    node_meta: vec4u,
    // Wet, program time in seconds, output width, output height.
    frame: vec4f,
    p0: vec4f,
    p1: vec4f,
    p2: vec4f,
    p3: vec4f,
    p4: vec4f,
    p5: vec4f,
    p6: vec4f,
    p7: vec4f,
    modes: vec4u,
};

@group(0) @binding(0) var source_tex: texture_2d<f32>;
@group(0) @binding(1) var donor_tex: texture_2d<f32>;
@group(0) @binding(2) var linear_samp: sampler;
@group(0) @binding(3) var nearest_samp: sampler;
@group(1) @binding(0) var<uniform> rack: RackUniforms;

fn straight_from_premultiplied_filter(value: vec4f) -> vec4f {
    let alpha = clamp(value.a, 0.0, 1.0);
    if alpha <= 0.000001 { return vec4f(0.0); }
    return vec4f(value.rgb / alpha, alpha);
}

// Rack processing is Advanced-only. Retain straight-alpha surfaces for the
// shared blend ABI, while making every authored linear resample operate on
// covered color so transparent hidden RGB cannot fringe.
fn source_premultiplied_linear(uv: vec2f) -> vec4f {
    let dimensions = vec2i(textureDimensions(source_tex));
    let coordinate = uv * vec2f(dimensions) - vec2f(0.5);
    let base = vec2i(floor(coordinate));
    let fraction = fract(coordinate);
    let maximum = dimensions - vec2i(1);
    let p00 = clamp(base, vec2i(0), maximum);
    let p10 = clamp(base + vec2i(1, 0), vec2i(0), maximum);
    let p01 = clamp(base + vec2i(0, 1), vec2i(0), maximum);
    let p11 = clamp(base + vec2i(1, 1), vec2i(0), maximum);
    let s00 = textureLoad(source_tex, p00, 0);
    let s10 = textureLoad(source_tex, p10, 0);
    let s01 = textureLoad(source_tex, p01, 0);
    let s11 = textureLoad(source_tex, p11, 0);
    let c00 = vec4f(s00.rgb * clamp(s00.a, 0.0, 1.0), clamp(s00.a, 0.0, 1.0));
    let c10 = vec4f(s10.rgb * clamp(s10.a, 0.0, 1.0), clamp(s10.a, 0.0, 1.0));
    let c01 = vec4f(s01.rgb * clamp(s01.a, 0.0, 1.0), clamp(s01.a, 0.0, 1.0));
    let c11 = vec4f(s11.rgb * clamp(s11.a, 0.0, 1.0), clamp(s11.a, 0.0, 1.0));
    return straight_from_premultiplied_filter(
        mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y),
    );
}

fn source_linear(uv: vec2f) -> vec4f {
    return source_premultiplied_linear(uv);
}

fn source_nearest(uv: vec2f) -> vec4f {
    return textureSampleLevel(source_tex, nearest_samp, uv, 0.0);
}

fn avalanche(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn seed_offset() -> f32 {
    if rack.node_meta.w == 0u {
        return 0.0;
    }
    return f32(avalanche(rack.node_meta.w) & 0x00ffffffu);
}

fn hash21(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn value_noise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2f(1.0, 0.0));
    let c = hash21(i + vec2f(0.0, 1.0));
    let d = hash21(i + vec2f(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn perlin_noise(p_in: vec2f, seed: f32) -> f32 {
    var noise = 0.0;
    var amp = 0.5;
    var p = p_in * 8.0;
    for (var i = 0; i < 4; i++) {
        noise += amp * value_noise(p + seed * 13.37);
        p *= 2.0;
        amp *= 0.5;
    }
    return noise * 2.0 - 1.0;
}

fn gaussian_noise(uv: vec2f, seed: f32) -> f32 {
    let u1 = hash21(uv + seed);
    let u2 = hash21(uv + seed + 1.71);
    return sqrt(-2.0 * log(max(u1, 0.001))) * cos(6.28318 * u2);
}

fn salt_pepper_noise(uv: vec2f, seed: f32, density: f32) -> f32 {
    let value = hash21(uv + seed);
    if value < density * 0.5 { return 1.0; }
    if value > 1.0 - density * 0.5 { return -1.0; }
    return 0.0;
}

fn blue_noise(uv: vec2f, seed: f32) -> f32 {
    let resolution = max(rack.frame.zw, vec2f(1.0));
    let center = gaussian_noise(uv, seed);
    let left = gaussian_noise(uv + vec2f(-1.0 / resolution.x, 0.0), seed);
    let right = gaussian_noise(uv + vec2f(1.0 / resolution.x, 0.0), seed);
    let up = gaussian_noise(uv + vec2f(0.0, 1.0 / resolution.y), seed);
    let down = gaussian_noise(uv + vec2f(0.0, -1.0 / resolution.y), seed);
    return center - 0.25 * (left + right + up + down);
}

fn cellular_hash2(cell: vec2i, epoch: u32) -> vec2f {
    let cx = bitcast<u32>(cell.x);
    let cy = bitcast<u32>(cell.y);
    var seed_input = cx ^ (cy * 0x9e3779b9u) ^ (epoch * 0x85ebca6bu);
    if rack.node_meta.w != 0u {
        seed_input ^= avalanche(rack.node_meta.w);
    }
    let seed = avalanche(seed_input);
    let unit = 1.0 / 16777216.0;
    return vec2f(
        f32(avalanche(seed ^ 0x68bc21ebu) & 0x00ffffffu) * unit,
        f32(avalanche(seed ^ 0x02e5be93u) & 0x00ffffffu) * unit,
    );
}

fn cellular_feature(cell: vec2i, epoch: u32, epoch_mix: f32) -> vec2f {
    let current = vec2f(0.2) + cellular_hash2(cell, epoch) * 0.6;
    let next = vec2f(0.2) + cellular_hash2(cell, epoch + 1u) * 0.6;
    return mix(current, next, epoch_mix);
}

fn cellular_worley(p: vec2f, epoch: u32, epoch_mix: f32) -> vec4f {
    let base = vec2i(floor(p));
    var nearest_sq = 1.0e9;
    var second_sq = 1.0e9;
    var nearest_offset = vec2f(0.0);
    for (var oy: i32 = -1; oy <= 1; oy += 1) {
        for (var ox: i32 = -1; ox <= 1; ox += 1) {
            let cell = base + vec2i(ox, oy);
            let feature = vec2f(cell) + cellular_feature(cell, epoch, epoch_mix);
            let offset = feature - p;
            let distance_sq = dot(offset, offset);
            if distance_sq < nearest_sq {
                second_sq = nearest_sq;
                nearest_sq = distance_sq;
                nearest_offset = offset;
            } else if distance_sq < second_sq {
                second_sq = distance_sq;
            }
        }
    }
    return vec4f(sqrt(nearest_sq), sqrt(second_sq), nearest_offset);
}

fn linear_channel_to_srgb(value: f32) -> f32 {
    let v = clamp(value, 0.0, 1.0);
    return select(1.055 * pow(v, 1.0 / 2.4) - 0.055, 12.92 * v, v <= 0.0031308);
}

fn display_chroma(color: vec3f) -> vec2f {
    let y = dot(color, vec3f(0.2126, 0.7152, 0.0722));
    return vec2f((color.b - y) / 1.8556, (color.r - y) / 1.5748);
}

fn processed_chroma(linear_color: vec3f) -> vec2f {
    return display_chroma(vec3f(
        linear_channel_to_srgb(linear_color.r),
        linear_channel_to_srgb(linear_color.g),
        linear_channel_to_srgb(linear_color.b),
    ));
}

fn rgb_to_hsl(c: vec3f) -> vec3f {
    let max_c = max(max(c.r, c.g), c.b);
    let min_c = min(min(c.r, c.g), c.b);
    let lightness = (max_c + min_c) * 0.5;
    let delta = max_c - min_c;
    if delta < 0.001 { return vec3f(0.0, 0.0, lightness); }
    let saturation = select(
        delta / (max_c + min_c),
        delta / (2.0 - max_c - min_c),
        lightness > 0.5,
    );
    var hue: f32;
    if max_c == c.r {
        hue = (c.g - c.b) / delta + select(0.0, 6.0, c.g < c.b);
    } else if max_c == c.g {
        hue = (c.b - c.r) / delta + 2.0;
    } else {
        hue = (c.r - c.g) / delta + 4.0;
    }
    return vec3f(hue / 6.0, saturation, lightness);
}

fn hue_to_rgb(p: f32, q: f32, initial: f32) -> f32 {
    var t = initial;
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 0.5 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

fn hsl_to_rgb(hsl: vec3f) -> vec3f {
    if hsl.y < 0.001 { return vec3f(hsl.z); }
    let q = select(
        hsl.z + hsl.y - hsl.z * hsl.y,
        hsl.z * (1.0 + hsl.y),
        hsl.z < 0.5,
    );
    let p = 2.0 * hsl.z - q;
    return vec3f(
        hue_to_rgb(p, q, hsl.x + 1.0 / 3.0),
        hue_to_rgb(p, q, hsl.x),
        hue_to_rgb(p, q, hsl.x - 1.0 / 3.0),
    );
}

fn sample_transform(output_uv: vec2f) -> vec4f {
    if rack.modes.z == 0u { return vec4f(0.0); }
    if rack.modes.w == 0u { return source_linear(output_uv); }
    var local_uv = vec2f(
        dot(rack.p0.xy, output_uv) + rack.p0.z,
        dot(rack.p1.xy, output_uv) + rack.p1.z,
    );
    var coverage = 1.0;
    if rack.modes.x == 0u {
        let inside = all(local_uv >= vec2f(0.0)) && all(local_uv <= vec2f(1.0));
        coverage = select(0.0, 1.0, inside);
        local_uv = clamp(local_uv, vec2f(0.0), vec2f(1.0));
    } else if rack.modes.x == 1u {
        local_uv = clamp(local_uv, vec2f(0.0), vec2f(1.0));
    } else if rack.modes.x == 2u {
        local_uv = fract(local_uv);
    } else {
        local_uv = vec2f(1.0) - abs(fract(local_uv * 0.5) * 2.0 - vec2f(1.0));
    }
    let source_uv = rack.p2.xy + local_uv * rack.p2.zw;
    var sampled: vec4f;
    if rack.modes.y == 1u {
        sampled = source_nearest(source_uv);
    } else {
        sampled = source_linear(source_uv);
    }
    return sampled * coverage;
}

fn digital_color(uv: vec2f) -> vec4f {
    let resolution = max(rack.frame.zw, vec2f(1.0));
    var sample_uv = uv;
    if rack.p0.z < 0.99 {
        let virtual_res = max(resolution * rack.p0.z, vec2f(1.0));
        sample_uv = (floor(sample_uv * virtual_res) + 0.5) / virtual_res;
    }
    if rack.p0.x > 1.0 {
        sample_uv = floor(sample_uv * resolution / rack.p0.x) * rack.p0.x / resolution;
    }
    var color: vec4f;
    if rack.p2.z > 0.0 {
        let seed = floor(rack.frame.y * 30.0) + seed_offset();
        let r_shift = (hash21(vec2f(seed, 10.0)) - 0.5) * rack.p2.z;
        let b_shift = (hash21(vec2f(seed, 11.0)) - 0.5) * rack.p2.z;
        let center = source_linear(sample_uv);
        color = vec4f(
            source_linear(vec2f(sample_uv.x + r_shift, sample_uv.y)).r,
            center.g,
            source_linear(vec2f(sample_uv.x + b_shift, sample_uv.y)).b,
            center.a,
        );
    } else if rack.p0.y > 0.0 {
        let offset = rack.p0.y / resolution.x;
        let center = source_linear(sample_uv);
        color = vec4f(
            source_linear(vec2f(sample_uv.x + offset, sample_uv.y)).r,
            center.g,
            source_linear(vec2f(sample_uv.x - offset, sample_uv.y)).b,
            center.a,
        );
    } else {
        color = source_linear(sample_uv);
    }
    var rgb = color.rgb;
    if abs(rack.p0.w) > 0.1 || abs(rack.p1.x) > 0.001 {
        var hsl = rgb_to_hsl(rgb);
        hsl.x = fract(hsl.x + rack.p0.w / 360.0);
        hsl.y = clamp(hsl.y + rack.p1.x, 0.0, 1.0);
        rgb = hsl_to_rgb(hsl);
    }
    if abs(rack.p1.y) > 0.001 { rgb += vec3f(rack.p1.y); }
    if abs(rack.p1.z) > 0.001 {
        rgb = (rgb - 0.5) * (1.0 + rack.p1.z * 2.0) + 0.5;
    }
    if rack.p1.w >= 2.0 { rgb = floor(rgb * rack.p1.w) / (rack.p1.w - 1.0); }
    if rack.p2.x > 0.5 { rgb = vec3f(1.0) - rgb; }
    if rack.p2.y > 0.0 {
        let distance = length(uv - 0.5) * 1.414;
        rgb *= max(1.0 - distance * distance * rack.p2.y, 0.0);
    }
    return vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)), color.a);
}

fn key_node(dry: vec4f) -> vec4f {
    var alpha = dry.a;
    let mode = u32(rack.p0.x);
    let softness = max(rack.p0.z, 0.001);
    if mode <= 1u {
        let luma = dot(dry.rgb, vec3f(0.2126, 0.7152, 0.0722));
        var admission = smoothstep(rack.p0.y - softness, rack.p0.y + softness, luma);
        if mode == 1u { admission = 1.0 - admission; }
        alpha *= admission;
    } else {
        let source_chroma = processed_chroma(dry.rgb);
        let target_chroma = display_chroma(clamp(rack.p1.xyz, vec3f(0.0), vec3f(1.0)));
        let distance = clamp(length(source_chroma - target_chroma) / 1.1913178, 0.0, 1.0);
        var outside = smoothstep(rack.p1.w - softness, rack.p1.w + softness, distance);
        if mode == 3u { outside = 1.0 - outside; }
        alpha *= outside;
    }
    if rack.p0.w > 0.5 { alpha = dry.a - alpha; }
    return vec4f(dry.rgb, clamp(alpha, 0.0, dry.a));
}

fn cellular_node(uv: vec2f) -> vec4f {
    let amount = rack.p0.x;
    let scale = rack.p0.y;
    let aspect = max(rack.frame.z / max(rack.frame.w, 1.0), 0.001);
    let p = vec2f(uv.x * aspect, uv.y) * scale;
    let motion_time = max(rack.frame.y, 0.0) * rack.p0.w;
    let epoch_value = floor(motion_time);
    let phase = fract(motion_time);
    let epoch_mix = phase * phase * (3.0 - 2.0 * phase);
    let field = cellular_worley(p, u32(epoch_value), epoch_mix);
    let ridge = 1.0 - smoothstep(0.0, 0.16, max(field.y - field.x, 0.0));
    var warp_cells = vec2f(0.0);
    if field.x > 0.00001 {
        warp_cells = (field.zw / field.x) * min(field.x, 0.5) * (0.28 * rack.p0.z);
    }
    let displaced = uv + warp_cells / vec2f(scale * aspect, scale) * amount;
    var result = source_linear(clamp(displaced, vec2f(0.0), vec2f(1.0)));
    result = vec4f(result.rgb * (1.0 - ridge * amount * 0.12), result.a);
    if rack.p1.x > 0.0001 && ridge > 0.0 {
        let soft = max(rack.p1.z, 0.0001);
        let ridge_mask = smoothstep(rack.p1.y - soft, rack.p1.y + soft, ridge);
        result.a *= 1.0 - ridge_mask * rack.p1.x;
    }
    return result;
}

fn shift_node(uv: vec2f) -> vec4f {
    let band = floor(uv.y * max(rack.frame.w, 1.0) / rack.p0.y);
    let epoch = floor(max(rack.frame.y, 0.0) * rack.p0.w);
    let seed = seed_offset();
    var sample_uv = uv;
    if hash21(vec2f(band + seed, epoch + 17.0)) < rack.p0.z {
        let direction = hash21(vec2f(band * 1.6180339 + seed, epoch + 73.0)) * 2.0 - 1.0;
        sample_uv.x = fract(sample_uv.x + direction * rack.p0.x * 0.25 + 1.0);
    }
    return source_linear(sample_uv);
}

fn grain_node(uv: vec2f, dry: vec4f) -> vec4f {
    var grain_uv = uv;
    if rack.p0.y > 1.5 {
        let grid = rack.frame.zw / rack.p0.y;
        grain_uv = floor(uv * grid) / grid;
    }
    let seed = floor(rack.frame.y * 30.0) + seed_offset();
    var n1: f32;
    var n2: f32;
    var n3: f32;
    let algorithm = u32(rack.p0.z);
    if algorithm == 1u {
        n1 = perlin_noise(grain_uv * rack.frame.zw / 80.0, seed);
        n2 = perlin_noise(grain_uv * rack.frame.zw / 80.0, seed + 100.0);
        n3 = perlin_noise(grain_uv * rack.frame.zw / 80.0, seed + 200.0);
    } else if algorithm == 2u {
        let density = rack.p0.x * 2.0;
        n1 = salt_pepper_noise(grain_uv * rack.frame.zw, seed, density);
        n2 = salt_pepper_noise(grain_uv * rack.frame.zw, seed + 100.0, density);
        n3 = salt_pepper_noise(grain_uv * rack.frame.zw, seed + 200.0, density);
    } else if algorithm == 3u {
        n1 = blue_noise(grain_uv, seed);
        n2 = blue_noise(grain_uv, seed + 100.0);
        n3 = blue_noise(grain_uv, seed + 200.0);
    } else {
        n1 = gaussian_noise(grain_uv * rack.frame.zw, seed);
        n2 = gaussian_noise(grain_uv * rack.frame.zw, seed + 100.0);
        n3 = gaussian_noise(grain_uv * rack.frame.zw, seed + 200.0);
    }
    let noise = select(vec3f(n1), vec3f(n1, n2, n3), rack.p0.w > 0.5);
    return vec4f(clamp(dry.rgb + noise * rack.p0.x, vec3f(0.0), vec3f(1.0)), dry.a);
}

fn rotate_point(point: vec2f, degrees: f32) -> vec2f {
    let angle = -degrees * 0.017453292519943295;
    let cosine = cos(angle);
    let sine = sin(angle);
    return vec2f(point.x * cosine - point.y * sine, point.x * sine + point.y * cosine);
}

fn shape_mask(uv: vec2f, ellipse: bool) -> f32 {
    let local = rotate_point(uv - rack.p0.xy, rack.p1.x);
    var signed_distance: f32;
    if ellipse {
        let radii = max(rack.p0.zw, vec2f(0.000001));
        signed_distance = (length(local / radii) - 1.0) * min(radii.x, radii.y);
    } else {
        let half_size = rack.p0.zw * 0.5;
        let q = abs(local) - half_size;
        signed_distance = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
    }
    var field = select(
        select(0.0, 1.0, signed_distance <= 0.0),
        1.0 - smoothstep(-rack.p1.y * 0.5, rack.p1.y * 0.5, signed_distance),
        rack.p1.y > 0.0000001,
    );
    if rack.p1.z > 0.5 { field = 1.0 - field; }
    return field;
}

fn donor_field(donor: vec4f) -> f32 {
    switch u32(rack.p0.x) {
        case 1u: { return dot(donor.rgb, vec3f(0.2126, 0.7152, 0.0722)); }
        case 2u: { return donor.r; }
        case 3u: { return donor.g; }
        case 4u: { return donor.b; }
        default: { return donor.a; }
    }
}

fn image_mask(uv: vec2f, dry: vec4f) -> vec4f {
    // The sample is unconditional so derivatives remain uniform. A missing
    // binding is nevertheless a defined zero field, and invert is suppressed.
    let donor = textureSampleLevel(donor_tex, linear_samp, uv, 0.0);
    var shaped = 0.0;
    if rack.node_meta.z != 0u {
        var field = clamp(donor_field(donor), 0.0, 1.0);
        if rack.p0.y > 0.5 { field = 1.0 - field; }
        if rack.p1.x <= 0.0000001 {
            shaped = select(0.0, 1.0, field >= rack.p0.w);
        } else {
            let half_width = rack.p1.x * 0.5;
            shaped = smoothstep(rack.p0.w - half_width, rack.p0.w + half_width, field);
        }
    }
    let admission = mix(1.0, shaped, rack.p0.z);
    return vec4f(dry.rgb, dry.a * admission);
}

// Donor vectors are alpha-covered, so the donor is filtered in premultiplied
// space and the result is deliberately returned premultiplied. The decode then
// consumes premultiplied RG together with the filtered alpha, which makes a
// transparent donor exactly zero no matter what its hidden RGB contains.
fn donor_premultiplied_linear(uv: vec2f) -> vec4f {
    let dimensions = vec2i(textureDimensions(donor_tex));
    let coordinate = uv * vec2f(dimensions) - vec2f(0.5);
    let base = vec2i(floor(coordinate));
    let fraction = fract(coordinate);
    let maximum = dimensions - vec2i(1);
    let p00 = clamp(base, vec2i(0), maximum);
    let p10 = clamp(base + vec2i(1, 0), vec2i(0), maximum);
    let p01 = clamp(base + vec2i(0, 1), vec2i(0), maximum);
    let p11 = clamp(base + vec2i(1, 1), vec2i(0), maximum);
    let s00 = textureLoad(donor_tex, p00, 0);
    let s10 = textureLoad(donor_tex, p10, 0);
    let s01 = textureLoad(donor_tex, p01, 0);
    let s11 = textureLoad(donor_tex, p11, 0);
    let c00 = vec4f(s00.rgb * clamp(s00.a, 0.0, 1.0), clamp(s00.a, 0.0, 1.0));
    let c10 = vec4f(s10.rgb * clamp(s10.a, 0.0, 1.0), clamp(s10.a, 0.0, 1.0));
    let c01 = vec4f(s01.rgb * clamp(s01.a, 0.0, 1.0), clamp(s01.a, 0.0, 1.0));
    let c11 = vec4f(s11.rgb * clamp(s11.a, 0.0, 1.0), clamp(s11.a, 0.0, 1.0));
    return mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y);
}

// Remap a displaced coordinate under the authored boundary law. `xy` is the
// carrier coordinate to sample; `z` is binary coverage, which only Transparent
// ever drops to zero.
fn displace_boundary(uv: vec2f) -> vec3f {
    let law = u32(rack.p0.z);
    if law == DISPLACE_MIRROR {
        return vec3f(vec2f(1.0) - abs(fract(uv * 0.5) * 2.0 - vec2f(1.0)), 1.0);
    }
    if law == DISPLACE_WRAP {
        return vec3f(fract(uv), 1.0);
    }
    if law == DISPLACE_HOLD {
        return vec3f(clamp(uv, vec2f(0.0), vec2f(1.0)), 1.0);
    }
    let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
    return vec3f(clamp(uv, vec2f(0.0), vec2f(1.0)), select(0.0, 1.0, inside));
}

fn displace_node(uv: vec2f) -> vec4f {
    // Sampled unconditionally so the lookup cost matches the declared ledger
    // for every donor state; a missing binding is a defined transparent field.
    let donor = donor_premultiplied_linear(uv);
    var vector = vec2f(0.0);
    if rack.node_meta.z != 0u {
        // Neutral donor encoding is RG = 0.5 at full coverage.
        vector = (donor.rg - vec2f(0.5) * donor.a) * 2.0;
    }
    let mapped = displace_boundary(uv + vector * rack.p0.xy);
    let sampled = source_linear(mapped.xy);
    // Coverage is binary, so this stays an exact keep-or-clear decision.
    return select(vec4f(0.0), sampled, mapped.z > 0.5);
}

fn apply_node_law(dry: vec4f, processed: vec4f) -> vec4f {
    let wet = clamp(rack.frame.x, 0.0, 1.0);
    if wet <= 0.0 { return dry; }
    var result: vec4f;
    if rack.node_meta.y == BLEND_ALPHA_CUT {
        // Explicit destination-out law. Processed alpha is the cutting field;
        // processed RGB never leaks into the retained destination.
        result = vec4f(dry.rgb, dry.a * (1.0 - clamp(processed.a, 0.0, 1.0)));
    } else {
        result = vec4f(
            blend_rgb(rack.node_meta.y, clamp(dry.rgb, vec3f(0.0), vec3f(1.0)),
                clamp(processed.rgb, vec3f(0.0), vec3f(1.0))),
            clamp(processed.a, 0.0, 1.0),
        );
    }
    if wet >= 1.0 { return result; }
    let alpha = mix(clamp(dry.a, 0.0, 1.0), result.a, wet);
    let premultiplied = mix(dry.rgb * clamp(dry.a, 0.0, 1.0), result.rgb * result.a, wet);
    if alpha <= BLEND_EPSILON { return vec4f(0.0); }
    return vec4f(premultiplied / alpha, alpha);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let kind = rack.node_meta.x;
    let dry = source_linear(uv);
    if kind == RACK_PASSTHROUGH || rack.frame.x <= 0.0 { return dry; }
    var processed = dry;
    switch kind {
        case RACK_TRANSFORM: { processed = sample_transform(uv); }
        case RACK_DIGITAL_COLOR: { processed = digital_color(uv); }
        case RACK_KEY: { processed = key_node(dry); }
        case RACK_CELLULAR: { processed = cellular_node(uv); }
        case RACK_SHIFT: { processed = shift_node(uv); }
        case RACK_GRAIN: { processed = grain_node(uv, dry); }
        case RACK_RECTANGLE_MASK: {
            processed = vec4f(dry.rgb, dry.a * shape_mask(uv, false));
        }
        case RACK_ELLIPSE_MASK: {
            processed = vec4f(dry.rgb, dry.a * shape_mask(uv, true));
        }
        case RACK_IMAGE_MASK: { processed = image_mask(uv, dry); }
        case RACK_DISPLACE: { processed = displace_node(uv); }
        default: {}
    }
    // Normal/full-wet uses the exact processed straight-alpha value. Other
    // blends use the canonical shared RGB kernel and retain processed alpha.
    if rack.node_meta.y == 0u && rack.frame.x >= 1.0 { return processed; }
    return apply_node_law(dry, processed);
}
