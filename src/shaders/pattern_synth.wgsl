// The B7 pattern-synth source pass: one fullscreen triangle computing the
// layer's whole picture from authored parameters and frame-plan time. No
// texture, no sampler — the picture is computed, which is what makes the
// source perfectly reconstructable offline.
//
// Laws derived from BENDR (MIT, © 2026 Steve Blythe), a browser circuit-bent
// video processor; `pattern_synth.rs` is the CPU reference this shader
// follows expression for expression, keeping BENDR's own numeric literals
// (3.14159, 6.2831853, the 0.003 gates). The computed colour is a
// display-domain value — the bytes a decoded video frame would hold — so the
// fragment returns it through the exact piecewise sRGB decode and the
// Rgba8UnormSrgb target's hardware encode stores those very bytes.

struct PatternUniforms {
    // freq_x, freq_y, phase, rate
    freq_phase: vec4f,
    // cross_mod, wavefold, pulse_width, comparator
    signal: vec4f,
    // comp_threshold, comp_soft, symmetry, zoom
    compare_frame: vec4f,
    // rotate, skew, center_x, center_y
    placement: vec4f,
    // warp, hue, hue_spread, saturation
    color_a: vec4f,
    // brightness, color_bands, time, aspect
    color_b: vec4f,
    // shape code, wave code, colour-mode code, reserved
    modes: vec4u,
    reserved: vec4f,
};

@group(0) @binding(0) var<uniform> uni: PatternUniforms;

const PATTERN_TAU: f32 = 6.2831853;

struct PatternVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_pattern(@builtin(vertex_index) vertex_index: u32) -> PatternVertexOutput {
    var out: PatternVertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// BENDR's h21 screen hash, kept expression for expression.
fn pattern_hash21(p_in: vec2f) -> f32 {
    var p = fract(p_in * vec2f(123.34, 456.21));
    p = p + dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// BENDR's vn value noise: smooth-interpolated h21 at cell corners.
fn pattern_value_noise(p: vec2f) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    let h00 = pattern_hash21(i);
    let h10 = pattern_hash21(i + vec2f(1.0, 0.0));
    let h01 = pattern_hash21(i + vec2f(0.0, 1.0));
    let h11 = pattern_hash21(i + vec2f(1.0, 1.0));
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// BENDR's compact HSV-to-RGB.
fn pattern_hsv(h: f32, s: f32, v: f32) -> vec3f {
    let k = fract(vec3f(h) + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0));
    return v * mix(vec3f(1.0), clamp(abs(k * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0)), s);
}

// The oscillator: one cycle of the selected waveform per unit of phase.
fn pattern_waveform(x_in: f32) -> f32 {
    let x = fract(x_in);
    let wave = uni.modes.y;
    if wave == 0u {
        return 0.5 + 0.5 * sin(x * PATTERN_TAU);
    }
    if wave == 1u {
        return abs(x * 2.0 - 1.0);
    }
    if wave == 2u {
        return x;
    }
    if wave == 3u {
        return step(0.5, x);
    }
    if wave == 4u {
        return step(1.0 - clamp(uni.signal.z, 0.02, 0.98), x);
    }
    return pattern_hash21(vec2f(floor(x * 48.0), 7.31));
}

// The additive Mandelbrot pattern is the literal escape-time law, separate
// from the twelve BENDR shape equations above. A finite pass proves escape;
// a point surviving the horizon is only bounded through this limit.
const MANDELBROT_MAX_ITERATIONS: u32 = 256u;

// Returns (escape-time signal, bounded-through-limit flag). The fixed page is
// centred on -0.5 + 0i with imaginary half-span one and real half-span equal
// to the page aspect, so one source pixel has equal complex scale on both axes.
fn mandelbrot_escape_signal(uv: vec2f, aspect: f32) -> vec2f {
    let c = vec2f(
        -0.5 + (uv.x * 2.0 - 1.0) * aspect,
        1.0 - uv.y * 2.0,
    );
    var z = vec2f(0.0);
    var iteration = 0u;
    loop {
        if iteration >= MANDELBROT_MAX_ITERATIONS {
            break;
        }
        let re_squared = z.x * z.x;
        let im_squared = z.y * z.y;
        let next = vec2f(
            re_squared - im_squared + c.x,
            2.0 * z.x * z.y + c.y,
        );
        z = next;
        iteration = iteration + 1u;
        // Strictly greater than the canonical squared bailout. Magnitude
        // exactly two is not itself an escape (notably at c = -2).
        if z.x * z.x + z.y * z.y > 4.0 {
            let signal = f32(MANDELBROT_MAX_ITERATIONS + 1u - iteration)
                / f32(MANDELBROT_MAX_ITERATIONS);
            return vec2f(signal, 0.0);
        }
    }
    return vec2f(0.0, 1.0);
}

// Deterministic motion shared by the two additive splat generators. The
// centre remains within its lattice cell, keeping the neighbourhood bounded.
fn moving_splat_point(
    offset: vec2f,
    h0: f32,
    h1: f32,
    t: f32,
    phase_radians: f32,
) -> vec2f {
    return offset + 0.5 + 0.32 * vec2f(
        sin(h0 * PATTERN_TAU + phase_radians + t * (0.55 + 0.85 * h1)),
        cos(h1 * PATTERN_TAU - phase_radians + t * (0.50 + 0.75 * h0)),
    );
}

// Two exact orbit endpoints and two interpolated positions make a four-sample
// memory-like trail without framebuffer state. Program time alone therefore
// remains enough to rebuild the exact frame during pause/seek/replay/export,
// while the full-resolution hot loop evaluates half as many sin/cos pairs.
fn memory_splats_signal(
    p: vec2f,
    fx: f32,
    freq_y: f32,
    phase: f32,
    t: f32,
    memory: f32,
) -> f32 {
    let g = p * (2.0 + fx * 0.22);
    let gi = floor(g);
    let gf = fract(g);
    let radius = 0.18 + freq_y * 0.32;
    let lag = 0.10 + memory * 0.65;
    let decay = 0.28 + memory * 0.55;
    let trail_weights = vec4f(1.0, decay, decay * decay, decay * decay * decay);
    let phase_radians = phase * PATTERN_TAU;
    // Complete nearest-cell search: omitted centres start at least 0.68 away,
    // beyond the compact kernel's maximum 0.5 radius. Sixteen samples replace
    // the naive 36-sample loop in this full-resolution hot path.
    var start_x = 0;
    var start_y = 0;
    if gf.x < 0.5 {
        start_x = -1;
    }
    if gf.y < 0.5 {
        start_y = -1;
    }
    var alpha = 0.0;
    for (var yi = 0; yi < 2; yi = yi + 1) {
        for (var xi = 0; xi < 2; xi = xi + 1) {
            let x = start_x + xi;
            let y = start_y + yi;
            let offset = vec2f(f32(x), f32(y));
            let cell = gi + offset;
            let h0 = pattern_hash21(cell);
            let h1 = pattern_hash21(cell + vec2f(5.17, 9.31));
            let head = moving_splat_point(offset, h0, h1, t, phase_radians);
            let tail = moving_splat_point(offset, h0, h1, t - 3.0 * lag, phase_radians);
            for (var age = 0; age < 4; age = age + 1) {
                var point = head;
                if age == 3 {
                    // Keep the reconstructed orbit endpoints bit-exact.
                    point = tail;
                } else if age > 0 {
                    point = mix(head, tail, f32(age) / 3.0);
                }
                let delta = point - gf;
                let distance_squared = dot(delta, delta);
                let inner_radius = radius * 0.35;
                let compact = 1.0 - smoothstep(
                    inner_radius * inner_radius,
                    radius * radius,
                    distance_squared,
                );
                let sample_alpha = clamp(compact * trail_weights[u32(age)], 0.0, 1.0);
                alpha = alpha + (1.0 - alpha) * sample_alpha;
            }
        }
    }
    return clamp(alpha, 0.0, 1.0);
}

// Moving anisotropic Gaussian splats, alpha-unioned into a bounded scalar
// field. Hashed major axes avoid per-pixel orientation trig; the finite support
// is zero before a cell outside the nearest-2x2 search can contribute.
fn gaussian_splats_signal(
    p: vec2f,
    fx: f32,
    freq_y: f32,
    phase: f32,
    t: f32,
    anisotropy: f32,
) -> f32 {
    let g = p * (2.0 + fx * 0.22);
    let gi = floor(g);
    let gf = fract(g);
    let sigma_major = 0.16 + freq_y * 0.24;
    let sigma_minor = sigma_major / (1.0 + anisotropy * 3.0);
    let phase_radians = phase * PATTERN_TAU;
    var start_x = 0;
    var start_y = 0;
    if gf.x < 0.5 {
        start_x = -1;
    }
    if gf.y < 0.5 {
        start_y = -1;
    }
    var alpha = 0.0;
    for (var yi = 0; yi < 2; yi = yi + 1) {
        for (var xi = 0; xi < 2; xi = xi + 1) {
            let x = start_x + xi;
            let y = start_y + yi;
            let offset = vec2f(f32(x), f32(y));
            let cell = gi + offset;
            let h0 = pattern_hash21(cell);
            let h1 = pattern_hash21(cell + vec2f(5.17, 9.31));
            let point = moving_splat_point(offset, h0, h1, t, phase_radians);
            let delta = gf - point;
            let h2 = pattern_hash21(cell + vec2f(11.7, 2.93));
            let h3 = pattern_hash21(cell + vec2f(19.19, 7.73));
            let axis = vec2f(h2 * 2.0 - 1.0, h3 * 2.0 - 1.0);
            let axis_length_squared = dot(axis, axis);
            var cos_angle = 1.0;
            var sin_angle = 0.0;
            if axis_length_squared > 1.0e-6 {
                let inverse_axis_length = inverseSqrt(axis_length_squared);
                cos_angle = axis.x * inverse_axis_length;
                sin_angle = axis.y * inverse_axis_length;
            }
            let local = vec2f(
                delta.x * cos_angle + delta.y * sin_angle,
                -delta.x * sin_angle + delta.y * cos_angle,
            );
            let quadratic = local.x * local.x / (sigma_major * sigma_major)
                + local.y * local.y / (sigma_minor * sigma_minor);
            // Finite Gaussian footprint, zero before any omitted cell's 0.68
            // minimum distance, keeps the nearest-2x2 search seam-free.
            let distance_squared = dot(delta, delta);
            let support = 1.0 - smoothstep(0.56 * 0.56, 0.66 * 0.66, distance_squared);
            let sample_alpha = clamp(exp(-0.5 * quadratic) * support, 0.0, 1.0);
            alpha = alpha + (1.0 - alpha) * sample_alpha;
        }
    }
    return clamp(alpha, 0.0, 1.0);
}

@fragment
fn fs_pattern(in: PatternVertexOutput) -> @location(0) vec4f {
    let t = uni.color_b.z * uni.freq_phase.w;
    // Framing: centre, aspect, zoom, rotate, skew, warp — the reference
    // flips uv into BENDR's bottom-up frame first.
    var p = vec2f(
        in.uv.x - 0.5 - uni.placement.z * 0.5,
        (1.0 - in.uv.y) - 0.5 - uni.placement.w * 0.5,
    );
    p.x = p.x * uni.color_b.w;
    let zm = exp2(uni.compare_frame.w * 2.0);
    p = p * zm;
    let a0 = uni.placement.x * 3.14159;
    let c0 = cos(a0);
    let s0 = sin(a0);
    p = vec2f(p.x * c0 - p.y * s0, p.x * s0 + p.y * c0);
    p.x = p.x + p.y * uni.placement.y * 2.0;
    if uni.color_a.x > 0.003 {
        let w = vec2f(
            pattern_value_noise(p * 3.0 + t * 0.2),
            pattern_value_noise(p * 3.0 + 17.3 - t * 0.15),
        );
        p = p + (w - 0.5) * uni.color_a.x * 1.2;
    }
    let fx = 0.2 + uni.freq_phase.x * uni.freq_phase.x * 40.0;
    let fy = 0.2 + uni.freq_phase.y * uni.freq_phase.y * 40.0;
    let ph = uni.freq_phase.z;
    let fm = uni.signal.x;
    let shape = uni.modes.x;
    let nf = max(1.0, floor(uni.compare_frame.z));
    // Polar math is one of the expensive full-resolution fractions. Only
    // shapes that consume radius/angle pay for it; invalid future codes retain
    // the historical POLYGON fallback and therefore still compute radius.
    var r = 0.0;
    if shape == 1u || shape == 2u || shape == 3u || shape == 5u
        || shape == 6u || shape == 8u || shape == 11u || shape > 14u {
        r = length(p);
    }
    var ang = 0.0;
    if shape == 1u || shape == 2u || shape == 6u || shape == 8u {
        ang = atan2(p.y, p.x) / PATTERN_TAU + 0.5;
    }
    var mandelbrot_interior = false;
    var f = 0.0;
    if shape == 0u {
        // SCAN — two ramps cross-modulating.
        let b = pattern_waveform(p.y * fy + t * 0.7);
        let a = pattern_waveform(p.x * fx + ph + t + b * fm * 3.0);
        f = 0.5 * (a + b);
    } else if shape == 1u {
        // RADIAL
        f = pattern_waveform(r * fx + ph + t + pattern_waveform(ang * nf) * fm * 2.0);
    } else if shape == 2u {
        // SPIRAL
        f = pattern_waveform(r * fx + ang * nf + ph + t);
    } else if shape == 3u {
        // PLASMA
        let f0 = 0.25
            * (pattern_waveform(p.x * fx * 0.5 + t)
                + pattern_waveform(p.y * fy * 0.5 - t * 0.8)
                + pattern_waveform((p.x + p.y) * fx * 0.35 + t * 1.3)
                + pattern_waveform(r * fy * 0.5 - t * 0.6));
        f = fract(f0 * (1.0 + fm * 3.0));
    } else if shape == 4u {
        // LISSAJOUS
        let lx = sin(p.x * fx + t);
        let ly = sin(p.y * fy + t * 1.37 + ph * PATTERN_TAU);
        f = pattern_waveform(lx * ly * (0.5 + fm * 3.0) + ph);
    } else if shape == 5u {
        // RINGS
        f = pattern_waveform(floor(r * fx * 0.5 + t) / max(1.0, nf) + ph);
    } else if shape == 6u {
        // STARBURST
        f = pattern_waveform(ang * nf + ph + t + r * fx * 0.06 * fm * 10.0);
    } else if shape == 7u {
        // GRID
        f = max(
            pattern_waveform(p.x * fx + ph + t),
            pattern_waveform(p.y * fy - t),
        );
    } else if shape == 8u {
        // TUNNEL
        let rr = 0.35 / max(r, 0.02);
        f = 0.5 * (pattern_waveform(rr * fx * 0.25 + t) + pattern_waveform(ang * nf + ph));
    } else if shape == 9u {
        // CELLS — Worley-style moving feature points.
        let g = p * max(1.0, fx * 0.25);
        let gi = floor(g);
        let gf = fract(g);
        var md = 8.0;
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let o = vec2f(f32(x), f32(y));
                let h0 = pattern_hash21(gi + o);
                let h1 = pattern_hash21(gi + o + 3.1);
                let pt = o + 0.5
                    + 0.5 * vec2f(sin(h0 * PATTERN_TAU + t * 2.0), cos(h1 * PATTERN_TAU + t * 1.7));
                md = min(md, length(pt - gf));
            }
        }
        f = pattern_waveform(md * (0.5 + fm * 3.0) + ph);
    } else if shape == 10u {
        // INTERFERENCE — two point sources beating.
        let d1 = length(p - vec2f(0.28, 0.0));
        let d2 = length(p + vec2f(0.28, 0.0));
        f = 0.5 * (pattern_waveform(d1 * fx + t) + pattern_waveform(d2 * fy - t));
    } else if shape == 12u {
        // MANDELBROT — raw page coordinates name c; the BENDR framing domain
        // cannot alter the orbit. Later signal/colour stages may dress the
        // exact integer escape time, while bounded points remain black.
        let sample = mandelbrot_escape_signal(in.uv, uni.color_b.w);
        f = sample.x;
        mandelbrot_interior = sample.y > 0.5;
    } else if shape == 13u {
        // MEMORY SPLATS — four reconstructable motion samples per point.
        f = memory_splats_signal(p, fx, uni.freq_phase.y, ph, t, fm);
    } else if shape == 14u {
        // GAUSSIAN SPLATS — moving anisotropic Gaussian kernels.
        f = gaussian_splats_signal(p, fx, uni.freq_phase.y, ph, t, fm);
    } else {
        // POLYGON
        let aa = atan2(p.y, p.x);
        let seg = PATTERN_TAU / nf;
        let rp = r * cos(aa - seg * floor(aa / seg) - seg * 0.5) / max(cos(seg * 0.5), 0.01);
        f = pattern_waveform(rp * fx * 0.5 + ph + t);
    }
    // Wavefolder: keeps folding the signal back on itself, which is where
    // the hard banded video-synth structure comes from.
    if uni.signal.y > 0.003 {
        let k = 1.0 + uni.signal.y * 7.0;
        f = abs(fract(f * k) * 2.0 - 1.0);
    }
    // Comparator: the hard-edged shape maker.
    if uni.signal.w > 0.003 {
        let sf = max(0.001, uni.compare_frame.y * 0.5);
        f = mix(
            f,
            smoothstep(uni.compare_frame.x - sf, uni.compare_frame.x + sf, f),
            uni.signal.w,
        );
    }
    f = clamp(f, 0.0, 1.0);
    // The colouriser.
    let hue = uni.color_a.y;
    let sp = uni.color_a.z;
    let sat = uni.color_a.w;
    let mode = uni.modes.z;
    var c = vec3f(f);
    if mandelbrot_interior {
        c = vec3f(0.0);
    } else if mode == 1u {
        // RGB PHASE — three channels of the same oscillator, offset.
        c = vec3f(
            pattern_waveform(f + hue),
            pattern_waveform(f + hue + sp * 0.33),
            pattern_waveform(f + hue + sp * 0.66),
        );
    } else if mode == 2u {
        c = pattern_hsv(hue + f * sp, sat, mix(1.0, f, 0.25));
    } else if mode == 3u {
        // DUOTONE
        c = mix(
            pattern_hsv(hue, sat, 1.0),
            pattern_hsv(fract(hue + sp * 0.5), sat, 1.0),
            f,
        );
    } else if mode == 4u {
        // BANDS
        let nb = max(2.0, floor(uni.color_b.y));
        let q = floor(f * nb) / nb;
        c = pattern_hsv(hue + q * sp, sat, 1.0);
    }
    let display = clamp(c * uni.color_b.x, vec3f(0.0), vec3f(1.0));
    // The computed value is display-domain; decode it so the sRGB target's
    // hardware encode stores exactly those bytes.
    return vec4f(
        pattern_srgb_to_linear(display.r),
        pattern_srgb_to_linear(display.g),
        pattern_srgb_to_linear(display.b),
        1.0,
    );
}

fn pattern_srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}
