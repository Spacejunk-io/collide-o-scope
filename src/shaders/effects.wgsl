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
    key_mode: f32,       // 0=off, 1=keep bright, 2=keep dark
    key_threshold: f32,
    key_softness: f32,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;

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

// --- Grain ---
fn get_grain(uv: vec2f) -> vec3f {
    var grain_uv = uv;
    if uniforms.grain_size > 1.5 {
        let grid = uniforms.resolution / uniforms.grain_size;
        grain_uv = floor(uv * grid) / grid;
    }

    let seed = floor(uniforms.time * 30.0);
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
    let seed = floor(uniforms.time * 30.0);

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

    // --- Breathing (UV distortion before sampling) ---
    if uniforms.breathe_scale > 0.0 || uniforms.breathe_rotation > 0.0 || uniforms.breathe_position > 0.0 {
        sample_uv = apply_breathing(sample_uv);
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
        let seed = floor(uniforms.time * 30.0);
        let r_shift = (hash(vec2f(seed, 10.0)) - 0.5) * uniforms.color_drift;
        let b_shift = (hash(vec2f(seed, 11.0)) - 0.5) * uniforms.color_drift;
        let r = textureSample(tex, samp, vec2f(sample_uv.x + r_shift, sample_uv.y)).r;
        let g = textureSample(tex, samp, sample_uv).g;
        let b = textureSample(tex, samp, vec2f(sample_uv.x + b_shift, sample_uv.y)).b;
        let a = textureSample(tex, samp, sample_uv).a;
        color = vec4f(r, g, b, a);
    } else if uniforms.rgb_split > 0.0 {
        // --- Static RGB Split ---
        let offset = uniforms.rgb_split / uniforms.resolution.x;
        let r = textureSample(tex, samp, vec2f(sample_uv.x + offset, sample_uv.y)).r;
        let g = textureSample(tex, samp, sample_uv).g;
        let b = textureSample(tex, samp, vec2f(sample_uv.x - offset, sample_uv.y)).b;
        let a = textureSample(tex, samp, sample_uv).a;
        color = vec4f(r, g, b, a);
    } else {
        color = textureSample(tex, samp, sample_uv);
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
    // Alpha carries the mask into the composite pass; rgb is premultiplied
    // so a keyed bottom layer correctly fades to black.
    var alpha = color.a;
    if uniforms.key_mode > 0.5 {
        let luma = dot(rgb, vec3f(0.299, 0.587, 0.114));
        let soft = max(uniforms.key_softness, 0.001);
        var k = smoothstep(uniforms.key_threshold - soft, uniforms.key_threshold + soft, luma);
        if uniforms.key_mode > 1.5 {
            k = 1.0 - k;
        }
        alpha *= k;
        rgb *= k;
    }

    return vec4f(rgb, alpha);
}
