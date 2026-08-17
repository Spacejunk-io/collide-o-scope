// Combined effects fragment shader

struct Uniforms {
    pixelate_size: f32,
    rgb_split: f32,
    resolution: vec2f,
    hue_shift: f32,
    saturation: f32,
    brightness: f32,
    contrast: f32,
    posterize: f32,
    invert: f32,
    downsample: f32,
    time: f32,
    // Analog: grain
    grain_intensity: f32,
    grain_size: f32,
    grain_algo: f32,
    color_grain: f32,
    // Analog: breathing + vignette
    breathe_scale: f32,
    breathe_rotation: f32,
    breathe_position: f32,
    vignette: f32,
    // Color drift + luma key
    color_drift: f32,
    key_mode: f32,       // 0=off, 1=keep bright, 2=keep dark, 3=remove color, 4=keep color
    key_threshold: f32,
    key_softness: f32,
    // Animated cellular / Worley domain warp
    cellular_amount: f32,
    cellular_scale: f32,
    cellular_warp: f32,
    cellular_speed: f32,
    // Cellular ridge-to-alpha key
    cellular_gap_amount: f32,
    cellular_gap_threshold: f32,
    cellular_gap_softness: f32,
    // Reuses the old 32-bit pad within the original nine-vec4 prefix.
    // Zero is a strict legacy sentinel and therefore contributes no offset.
    random_seed: u32,
    // Chroma-key target is supplied in display/sRGB coordinates.
    key_color: vec3f,
    key_tolerance: f32,
    // Deterministic horizontal block displacement. The amount's zero default
    // takes an exact no-op path, preserving legacy pixels.
    shift_amount: f32,
    shift_block_size: f32,
    shift_density: f32,
    shift_speed: f32,
    // Canonical authored spatial transform. These four vec4 slots are packed
    // immediately after the ten legacy effect slots by EffectPassUniforms.
    spatial_inverse_row_0: vec4f,
    spatial_inverse_row_1: vec4f,
    // Crop origin X/Y followed by crop extent X/Y in source UV space.
    spatial_crop: vec4f,
    // Edge mode, sampling mode, valid flag, spatial-active flag.
    spatial_modes: vec4u,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var nearest_samp: sampler;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;

fn straight_from_premultiplied_filter(value: vec4f) -> vec4f {
    let alpha = clamp(value.a, 0.0, 1.0);
    if alpha <= 0.000001 {
        return vec4f(0.0);
    }
    return vec4f(value.rgb / alpha, alpha);
}

// Advanced authored transforms retain straight-alpha storage, but hardware
// filtering straight RGB would interpolate hidden color before coverage.
// Load four texels, premultiply each independently, then bilinearly resolve
// and return to the storage contract. The legacy inactive branch below stays
// on its frozen textureSample expression byte-for-byte.
fn sample_source_premultiplied_linear(uv: vec2f) -> vec4f {
    let dimensions = vec2i(textureDimensions(tex));
    let coordinate = uv * vec2f(dimensions) - vec2f(0.5);
    let base = vec2i(floor(coordinate));
    let fraction = fract(coordinate);
    let maximum = dimensions - vec2i(1);
    let p00 = clamp(base, vec2i(0), maximum);
    let p10 = clamp(base + vec2i(1, 0), vec2i(0), maximum);
    let p01 = clamp(base + vec2i(0, 1), vec2i(0), maximum);
    let p11 = clamp(base + vec2i(1, 1), vec2i(0), maximum);
    let s00 = textureLoad(tex, p00, 0);
    let s10 = textureLoad(tex, p10, 0);
    let s01 = textureLoad(tex, p01, 0);
    let s11 = textureLoad(tex, p11, 0);
    let c00 = vec4f(s00.rgb * clamp(s00.a, 0.0, 1.0), clamp(s00.a, 0.0, 1.0));
    let c10 = vec4f(s10.rgb * clamp(s10.a, 0.0, 1.0), clamp(s10.a, 0.0, 1.0));
    let c01 = vec4f(s01.rgb * clamp(s01.a, 0.0, 1.0), clamp(s01.a, 0.0, 1.0));
    let c11 = vec4f(s11.rgb * clamp(s11.a, 0.0, 1.0), clamp(s11.a, 0.0, 1.0));
    let covered = mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y);
    return straight_from_premultiplied_filter(covered);
}

// Map a composition-space UV through the canonical inverse transform and the
// cropped source rectangle. The inactive branch deliberately contains the
// exact historical sample expression: legacy patches must not gain a clamp,
// coordinate operation, or filtering change merely because the uniform block
// grew. Transparent edges still execute a texture sample so textureSample is
// never placed behind fragment-varying control flow (which would invalidate
// implicit derivatives on strict WGSL backends).
fn sample_source(output_uv: vec2f) -> vec4f {
    if uniforms.spatial_modes.w == 0u {
        return textureSample(tex, samp, output_uv);
    }
    if uniforms.spatial_modes.z == 0u {
        return vec4f(0.0);
    }

    var local_uv = vec2f(
        dot(uniforms.spatial_inverse_row_0.xy, output_uv)
            + uniforms.spatial_inverse_row_0.z,
        dot(uniforms.spatial_inverse_row_1.xy, output_uv)
            + uniforms.spatial_inverse_row_1.z,
    );
    var coverage = 1.0;
    let edge_mode = uniforms.spatial_modes.x;
    if edge_mode == 0u {
        let inside = all(local_uv >= vec2f(0.0)) && all(local_uv <= vec2f(1.0));
        coverage = select(0.0, 1.0, inside);
        // Keep the speculative sample bounded even when coverage is zero.
        local_uv = clamp(local_uv, vec2f(0.0), vec2f(1.0));
    } else if edge_mode == 1u {
        local_uv = clamp(local_uv, vec2f(0.0), vec2f(1.0));
    } else if edge_mode == 2u {
        local_uv = fract(local_uv);
    } else {
        // Triangle-wave mirror repeat: 0,1,0 over every two unit intervals.
        local_uv = vec2f(1.0) - abs(fract(local_uv * 0.5) * 2.0 - vec2f(1.0));
    }

    let source_uv = uniforms.spatial_crop.xy + local_uv * uniforms.spatial_crop.zw;
    var sampled: vec4f;
    if uniforms.spatial_modes.y == 1u {
        sampled = textureSample(tex, nearest_samp, source_uv);
    } else if uniforms.spatial_modes.w == 1u {
        // Frozen LegacyExact active-spatial law. Advanced upgrades the mode to
        // 2 at its private uniform-upload boundary; legacy live/export keeps
        // the historical hardware straight-alpha sample byte-for-byte.
        sampled = textureSample(tex, samp, source_uv);
    } else {
        sampled = sample_source_premultiplied_linear(source_uv);
    }
    return sampled * coverage;
}

// --- Hash / noise functions (ported from legacy GLSL) ---

fn hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn value_noise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f); // smoothstep
    let a = hash(i);
    let b = hash(i + vec2f(1.0, 0.0));
    let c = hash(i + vec2f(0.0, 1.0));
    let d = hash(i + vec2f(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn perlin_noise(uv: vec2f, seed: f32) -> f32 {
    var noise = 0.0;
    var amp = 0.5;
    var p = uv * 8.0;
    for (var i = 0; i < 4; i++) {
        noise += amp * value_noise(p + seed * 13.37);
        p *= 2.0;
        amp *= 0.5;
    }
    return noise * 2.0 - 1.0;
}

fn gaussian_noise(uv: vec2f, seed: f32) -> f32 {
    let u1 = hash(uv + seed);
    let u2 = hash(uv + seed + 1.71);
    return sqrt(-2.0 * log(max(u1, 0.001))) * cos(6.28318 * u2);
}

fn salt_pepper_noise(uv: vec2f, seed: f32, density: f32) -> f32 {
    let r = hash(uv + seed);
    if r < density * 0.5 { return 1.0; }
    if r > 1.0 - density * 0.5 { return -1.0; }
    return 0.0;
}

fn blue_noise(uv: vec2f, seed: f32) -> f32 {
    let center = gaussian_noise(uv, seed);
    let left = gaussian_noise(uv + vec2f(-1.0 / uniforms.resolution.x, 0.0), seed);
    let right = gaussian_noise(uv + vec2f(1.0 / uniforms.resolution.x, 0.0), seed);
    let up = gaussian_noise(uv + vec2f(0.0, 1.0 / uniforms.resolution.y), seed);
    let down = gaussian_noise(uv + vec2f(0.0, -1.0 / uniforms.resolution.y), seed);
    return center - 0.25 * (left + right + up + down);
}

// Integer avalanche hashing keeps cellular feature points deterministic on
// every GPU backend without depending on large-argument trigonometry.
fn cellular_avalanche(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn pattern_seed_offset() -> f32 {
    if uniforms.random_seed == 0u {
        return 0.0;
    }
    // A 24-bit integer is represented exactly by f32, keeping every backend
    // on the same deterministic path before the legacy float hashes run.
    return f32(cellular_avalanche(uniforms.random_seed) & 0x00ffffffu);
}

fn cellular_hash2(cell: vec2i, epoch: u32) -> vec2f {
    let cx = bitcast<u32>(cell.x);
    let cy = bitcast<u32>(cell.y);
    var seed_input = cx ^ (cy * 0x9e3779b9u) ^ (epoch * 0x85ebca6bu);
    if uniforms.random_seed != 0u {
        seed_input ^= cellular_avalanche(uniforms.random_seed);
    }
    let seed = cellular_avalanche(seed_input);
    let hx = cellular_avalanche(seed ^ 0x68bc21ebu);
    let hy = cellular_avalanche(seed ^ 0x02e5be93u);
    let unit = 1.0 / 16777216.0;
    return vec2f(
        f32(hx & 0x00ffffffu) * unit,
        f32(hy & 0x00ffffffu) * unit,
    );
}

// Each feature remains inside the central 60% of its own cell. That bound is
// what makes a 3x3 neighborhood both sufficient and fixed-cost. Adjacent
// deterministic targets are smoothly interpolated, producing Brownian-like
// motion without discontinuities at epoch boundaries.
fn cellular_feature(cell: vec2i, epoch: u32, epoch_mix: f32) -> vec2f {
    let current = vec2f(0.2) + cellular_hash2(cell, epoch) * 0.6;
    let next = vec2f(0.2) + cellular_hash2(cell, epoch + 1u) * 0.6;
    return mix(current, next, epoch_mix);
}

// Returns nearest distance, second-nearest distance, and the vector from the
// sample to its nearest feature point, all in aspect-correct cell space.
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
    // Only two square roots per pixel, after the nearest candidates are known.
    return vec4f(sqrt(nearest_sq), sqrt(second_sq), nearest_offset);
}

// --- Grain ---
fn get_grain(uv: vec2f) -> vec3f {
    var grain_uv = uv;
    if uniforms.grain_size > 1.5 {
        let grid = uniforms.resolution / uniforms.grain_size;
        grain_uv = floor(uv * grid) / grid;
    }

    let seed = floor(uniforms.time * 30.0) + pattern_seed_offset();
    var n1: f32; var n2: f32; var n3: f32;

    let algo = i32(uniforms.grain_algo);
    if algo == 1 {
        // Perlin
        n1 = perlin_noise(grain_uv * uniforms.resolution / 80.0, seed);
        n2 = perlin_noise(grain_uv * uniforms.resolution / 80.0, seed + 100.0);
        n3 = perlin_noise(grain_uv * uniforms.resolution / 80.0, seed + 200.0);
    } else if algo == 2 {
        // Salt & pepper
        let density = uniforms.grain_intensity * 2.0;
        n1 = salt_pepper_noise(grain_uv * uniforms.resolution, seed, density);
        n2 = salt_pepper_noise(grain_uv * uniforms.resolution, seed + 100.0, density);
        n3 = salt_pepper_noise(grain_uv * uniforms.resolution, seed + 200.0, density);
    } else if algo == 3 {
        // Blue noise
        n1 = blue_noise(grain_uv, seed);
        n2 = blue_noise(grain_uv, seed + 100.0);
        n3 = blue_noise(grain_uv, seed + 200.0);
    } else {
        // Gaussian (default)
        n1 = gaussian_noise(grain_uv * uniforms.resolution, seed);
        n2 = gaussian_noise(grain_uv * uniforms.resolution, seed + 100.0);
        n3 = gaussian_noise(grain_uv * uniforms.resolution, seed + 200.0);
    }

    if uniforms.color_grain < 0.5 {
        return vec3f(n1) * uniforms.grain_intensity;
    } else {
        return vec3f(n1, n2, n3) * uniforms.grain_intensity;
    }
}

// --- Breathing (UV distortion) ---
fn apply_breathing(uv: vec2f) -> vec2f {
    var out_uv = uv;
    let seed = floor(uniforms.time * 30.0) + pattern_seed_offset();

    // Scale breathing (zoom pulsing)
    if uniforms.breathe_scale > 0.0 {
        let scale_offset = (hash(vec2f(seed, 0.0)) - 0.5) * uniforms.breathe_scale;
        let scale = 1.0 + scale_offset;
        out_uv = (out_uv - 0.5) / scale + 0.5;
    }

    // Rotation breathing
    if uniforms.breathe_rotation > 0.0 {
        let angle = (hash(vec2f(seed, 1.0)) - 0.5) * uniforms.breathe_rotation * 0.01745;
        let centered = out_uv - 0.5;
        let c = cos(angle);
        let s = sin(angle);
        out_uv = vec2f(centered.x * c - centered.y * s, centered.x * s + centered.y * c) + 0.5;
    }

    // Position drift
    if uniforms.breathe_position > 0.0 {
        let dx = (hash(vec2f(seed, 2.0)) - 0.5) * uniforms.breathe_position;
        let dy = (hash(vec2f(seed, 3.0)) - 0.5) * uniforms.breathe_position;
        out_uv += vec2f(dx, dy);
    }

    return out_uv;
}

// --- Color space helpers ---

// Effect math runs in linear light because the source view is sRGB. Key-color
// controls, however, use ordinary display/sRGB coordinates. Convert the
// processed pixel back to display space before comparing Rec.709 Cb/Cr so a
// picked green remains a green-screen target independent of luminance.
fn linear_channel_to_srgb(value: f32) -> f32 {
    let v = clamp(value, 0.0, 1.0);
    return select(1.055 * pow(v, 1.0 / 2.4) - 0.055, 12.92 * v, v <= 0.0031308);
}

fn display_chroma(color: vec3f) -> vec2f {
    let y = dot(color, vec3f(0.2126, 0.7152, 0.0722));
    return vec2f((color.b - y) / 1.8556, (color.r - y) / 1.5748);
}

fn processed_chroma(linear_color: vec3f) -> vec2f {
    let display = vec3f(
        linear_channel_to_srgb(linear_color.r),
        linear_channel_to_srgb(linear_color.g),
        linear_channel_to_srgb(linear_color.b),
    );
    return display_chroma(display);
}

// Rec.709 Cb/Cr spans 1.1913178 between the farthest display-RGB cube
// vertices (green and magenta). Normalize by that bound so a UI tolerance of
// one means the complete legal color plane, rather than an arbitrary clipped
// subset. Equal-neutral colors (including black and white) remain distance 0.
const MAX_DISPLAY_CHROMA_DISTANCE: f32 = 1.1913178;

fn rgb_to_hsl(c: vec3f) -> vec3f {
    let max_c = max(max(c.r, c.g), c.b);
    let min_c = min(min(c.r, c.g), c.b);
    let l = (max_c + min_c) * 0.5;
    let delta = max_c - min_c;

    if delta < 0.001 {
        return vec3f(0.0, 0.0, l);
    }

    let s = select(
        delta / (max_c + min_c),
        delta / (2.0 - max_c - min_c),
        l > 0.5
    );

    var h: f32;
    if max_c == c.r {
        h = (c.g - c.b) / delta + select(0.0, 6.0, c.g < c.b);
    } else if max_c == c.g {
        h = (c.b - c.r) / delta + 2.0;
    } else {
        h = (c.r - c.g) / delta + 4.0;
    }
    h /= 6.0;

    return vec3f(h, s, l);
}

fn hue_to_rgb(p: f32, q: f32, t_in: f32) -> f32 {
    var t = t_in;
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 0.5 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

fn hsl_to_rgb(hsl: vec3f) -> vec3f {
    let h = hsl.x;
    let s = hsl.y;
    let l = hsl.z;

    if s < 0.001 {
        return vec3f(l, l, l);
    }

    let q = select(l + s - l * s, l * (1.0 + s), l < 0.5);
    let p = 2.0 * l - q;

    return vec3f(
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    );
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    var sample_uv = uv;
    var cellular_ridge = 0.0;

    // --- Breathing (UV distortion before sampling) ---
    if uniforms.breathe_scale > 0.0 || uniforms.breathe_rotation > 0.0 || uniforms.breathe_position > 0.0 {
        sample_uv = apply_breathing(sample_uv);
    }

    // --- Cellular / Worley domain warp ---
    // The disabled branch avoids all 3x3 distance work for legacy patches.
    if uniforms.cellular_amount > 0.0001 {
        let amount = clamp(uniforms.cellular_amount, 0.0, 1.0);
        let scale = clamp(uniforms.cellular_scale, 2.0, 32.0);
        let warp = clamp(uniforms.cellular_warp, 0.0, 1.0);
        let speed = clamp(uniforms.cellular_speed, 0.0, 2.0);
        let aspect = max(uniforms.resolution.x / max(uniforms.resolution.y, 1.0), 0.001);
        let p = vec2f(sample_uv.x * aspect, sample_uv.y) * scale;
        let motion_time = max(uniforms.time, 0.0) * speed;
        let epoch_value = floor(motion_time);
        let phase = fract(motion_time);
        let epoch_mix = phase * phase * (3.0 - 2.0 * phase);
        let field = cellular_worley(p, u32(epoch_value), epoch_mix);

        // F2-F1 approaches zero at Voronoi boundaries. The displacement is
        // capped at 0.14 cell, then converted back from physical cell space.
        cellular_ridge = 1.0 - smoothstep(0.0, 0.16, max(field.y - field.x, 0.0));
        var warp_cells = vec2f(0.0);
        if field.x > 0.00001 {
            warp_cells = (field.zw / field.x) * min(field.x, 0.5) * (0.28 * warp);
        }
        let warp_uv = warp_cells / vec2f(scale * aspect, scale);
        let warped_uv = sample_uv + warp_uv * amount;
        // Preserve the historical clamp only for the exact legacy bypass.
        // Active transforms delegate every exposed coordinate to their
        // explicit Transparent/Clamp/Repeat/Mirror edge law.
        if uniforms.spatial_modes.w == 0u {
            sample_uv = clamp(warped_uv, vec2f(0.0), vec2f(1.0));
        } else {
            sample_uv = warped_uv;
        }
    }

    // --- Shift (seeded horizontal block displacement) ---
    // Bands are anchored to output pixels and advance in discrete time epochs,
    // so a given seed, time and parameter set always produces the same frame.
    if uniforms.shift_amount > 0.0001 {
        let amount = clamp(uniforms.shift_amount, 0.0, 1.0);
        let block_size = clamp(uniforms.shift_block_size, 2.0, 256.0);
        let density = clamp(uniforms.shift_density, 0.0, 1.0);
        let speed = clamp(uniforms.shift_speed, 0.0, 20.0);
        let band = floor(uv.y * max(uniforms.resolution.y, 1.0) / block_size);
        let epoch = floor(max(uniforms.time, 0.0) * speed);
        let seed = pattern_seed_offset();
        let gate = hash(vec2f(band + seed, epoch + 17.0));
        if gate < density {
            let direction = hash(vec2f(band * 1.6180339 + seed, epoch + 73.0)) * 2.0 - 1.0;
            let offset = direction * amount * 0.25;
            if uniforms.spatial_modes.w == 0u {
                sample_uv.x = fract(sample_uv.x + offset + 1.0);
            } else {
                sample_uv.x += offset;
            }
        }
    }

    // --- Downsample (lossy video look) ---
    if uniforms.downsample < 0.99 {
        let virtual_res = uniforms.resolution * uniforms.downsample;
        sample_uv = (floor(sample_uv * virtual_res) + 0.5) / virtual_res;
    }

    // --- Pixelate ---
    if uniforms.pixelate_size > 1.0 {
        let block = uniforms.pixelate_size;
        let res = uniforms.resolution;
        sample_uv = floor(sample_uv * res / block) * block / res;
    }

    // --- Color drift (per-frame random chromatic aberration) ---
    var color: vec4f;
    if uniforms.color_drift > 0.0 {
        let seed = floor(uniforms.time * 30.0) + pattern_seed_offset();
        let r_shift = (hash(vec2f(seed, 10.0)) - 0.5) * uniforms.color_drift;
        let b_shift = (hash(vec2f(seed, 11.0)) - 0.5) * uniforms.color_drift;
        let center = sample_source(sample_uv);
        let r = sample_source(vec2f(sample_uv.x + r_shift, sample_uv.y)).r;
        let g = center.g;
        let b = sample_source(vec2f(sample_uv.x + b_shift, sample_uv.y)).b;
        let a = center.a;
        color = vec4f(r, g, b, a);
    } else if uniforms.rgb_split > 0.0 {
        // --- Static RGB Split ---
        let offset = uniforms.rgb_split / uniforms.resolution.x;
        let center = sample_source(sample_uv);
        let r = sample_source(vec2f(sample_uv.x + offset, sample_uv.y)).r;
        let g = center.g;
        let b = sample_source(vec2f(sample_uv.x - offset, sample_uv.y)).b;
        let a = center.a;
        color = vec4f(r, g, b, a);
    } else {
        color = sample_source(sample_uv);
    }

    var rgb = color.rgb;

    // --- Color adjustment (hue/saturation) ---
    if abs(uniforms.hue_shift) > 0.1 || abs(uniforms.saturation) > 0.001 {
        var hsl = rgb_to_hsl(rgb);
        hsl.x = fract(hsl.x + uniforms.hue_shift / 360.0);
        hsl.y = clamp(hsl.y + uniforms.saturation, 0.0, 1.0);
        rgb = hsl_to_rgb(hsl);
    }

    // --- Brightness ---
    if abs(uniforms.brightness) > 0.001 {
        rgb = rgb + vec3f(uniforms.brightness);
    }

    // --- Contrast ---
    if abs(uniforms.contrast) > 0.001 {
        let factor = 1.0 + uniforms.contrast * 2.0;
        rgb = (rgb - 0.5) * factor + 0.5;
    }

    // --- Posterize ---
    if uniforms.posterize >= 2.0 {
        let levels = uniforms.posterize;
        rgb = floor(rgb * levels) / (levels - 1.0);
    }

    // --- Invert ---
    if uniforms.invert > 0.5 {
        rgb = vec3f(1.0) - rgb;
    }

    // A restrained dark ridge makes cell boundaries legible without turning
    // the source into a binary line drawing.
    if cellular_ridge > 0.0 {
        rgb *= 1.0 - cellular_ridge * clamp(uniforms.cellular_amount, 0.0, 1.0) * 0.12;
    }

    // --- Vignette ---
    if uniforms.vignette > 0.0 {
        let centered = uv - 0.5; // use original UV for vignette position
        let dist = length(centered) * 1.414; // normalize so corners = 1
        let vig = 1.0 - dist * dist * uniforms.vignette;
        rgb *= max(vig, 0.0);
    }

    // --- Grain (additive, applied last) ---
    if uniforms.grain_intensity > 0.0 {
        let grain = get_grain(uv);
        rgb += grain;
    }

    rgb = clamp(rgb, vec3f(0.0), vec3f(1.0));

    // --- Luma key (per-layer): carve shapes out of the frame by luminance.
    // Alpha carries the mask into the composite pass. Keep RGB straight
    // (un-premultiplied): composite.wgsl multiplies it by overlay alpha when
    // blending. Premultiplying here as well would square the soft key mask
    // and produce dark fringes.
    var alpha = color.a;
    if uniforms.cellular_gap_amount > 0.0001 && cellular_ridge > 0.0 {
        let threshold = clamp(uniforms.cellular_gap_threshold, 0.0, 1.0);
        let softness = max(clamp(uniforms.cellular_gap_softness, 0.0, 0.5), 0.0001);
        let ridge_mask = smoothstep(threshold - softness, threshold + softness, cellular_ridge);
        alpha *= 1.0 - ridge_mask * clamp(uniforms.cellular_gap_amount, 0.0, 1.0);
    }
    if uniforms.key_mode > 0.5 && uniforms.key_mode < 2.5 {
        // Bright/dark luminance keys retain their established behavior.
        // The texture view has already decoded sRGB, so use linear-light
        // Rec.709 luminance rather than legacy gamma-domain Rec.601 weights.
        let luma = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
        let soft = max(clamp(uniforms.key_softness, 0.0, 0.5), 0.001);
        var k = smoothstep(uniforms.key_threshold - soft, uniforms.key_threshold + soft, luma);
        if uniforms.key_mode > 1.5 {
            k = 1.0 - k;
        }
        alpha *= k;
    } else if uniforms.key_mode > 2.5 {
        // Chroma distance ignores luminance, making this a real color key
        // rather than an RGB-radius selector. Mode 3 removes the selected
        // chroma; mode 4 retains it. RGB stays straight/un-premultiplied.
        let source_chroma = processed_chroma(rgb);
        let target_chroma = display_chroma(clamp(uniforms.key_color, vec3f(0.0), vec3f(1.0)));
        let distance = clamp(
            length(source_chroma - target_chroma) / MAX_DISPLAY_CHROMA_DISTANCE,
            0.0,
            1.0,
        );
        let tolerance = clamp(uniforms.key_tolerance, 0.0, 1.0);
        let soft = max(clamp(uniforms.key_softness, 0.0, 0.5), 0.001);
        var outside = smoothstep(tolerance - soft, tolerance + soft, distance);
        if uniforms.key_mode > 3.5 {
            outside = 1.0 - outside;
        }
        alpha *= outside;
    }

    return vec4f(rgb, alpha);
}
