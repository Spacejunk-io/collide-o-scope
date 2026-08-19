// Additive Temporal Originals path.
//
// The legacy shader remains in temporal.wgsl and is selected whenever every
// original is zero. Keep its frozen 64-byte uniform at binding 0; this path
// adds a separate fixed 128-byte block at binding 1. Collision Atlas is a
// bounded analytical 3x3 Worley search and allocates/samples no extra texture.

struct TemporalUniforms {
    feedback: f32,
    fb_zoom: f32,
    fb_rotate: f32,
    slitscan: f32,
    history_len: f32,
    write_index: f32,
    valid_history: f32,
    feedback_valid: f32,
    slit_direction: vec2f,
    key_reference_layer: f32,
    key_valid: f32,
    key_mode: f32,
    key_threshold: f32,
    key_softness: f32,
    _pad0: f32,
};

// B3 feedback rig, mirrored from temporal.wgsl in lockstep.
struct TemporalRigUniforms {
    values_a: vec4f, // offset_x, offset_y, hue rotate radians, saturation
    values_b: vec4f, // chroma displace, blur, sharpen, drive
    values_c: vec4f, // pivot, threshold, noise, tick mix
    values_d: vec4f, // gain r, gain g, gain b, servo strength
    modes_a: vec4u,  // reflect x, reflect y, shape code, edge code
    modes_b: vec4u,  // noise epoch, rig active, reserved, reserved
};

struct TemporalOriginalsUniforms {
    loom_values: vec4f,    // amount, depth, phase, scale
    loom_geometry: vec4f,  // angle radians, aspect ratio, reserved, reserved
    loom_modes: vec4u,     // topology, interpolation, folds, quantization
    atlas_values: vec4f,   // amount, collision, reserved, reserved
    atlas_modes: vec4u,    // seed, territories, score state count, flags
    score_runtime: vec4u,  // score seed, state index, ordinal low, ordinal high
    garden_values: vec4f,  // amount, threshold, softness, decay
    garden_modes: vec4u,   // gate, max hold ticks, observation ticks, packed runtime
};

@group(0) @binding(0) var current_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var feedback_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> u: TemporalUniforms;
@group(1) @binding(1) var<uniform> originals: TemporalOriginalsUniforms;
@group(1) @binding(2) var<uniform> rig: TemporalRigUniforms;

const TAU: f32 = 6.28318530717958647692;

// B12 fixed law: scanlines per TBC-failure sawtooth period.
const TIME_DISPLACE_TBC_LINES: f32 = 8.0;

fn wrap_layer(idx: f32) -> i32 {
    let n = u.history_len;
    return i32(((idx % n) + n) % n);
}

// ---------------------------------------------------------------------------
// B12 time-displace maps. CPU reference: `temporal::time_displace_coord`,
// followed expression for expression. Map codes are permanent: 0 Ramp,
// 1 Brightness, 2 Radial, 3 TbcRamp, 4 Sweep. Ramp is the exact legacy
// slit-scan coordinate; the map code and interpolation flag ride the two
// reserved loom_geometry lanes and the sweep phase rides the reserved
// atlas_values lane, all zero for a default patch.
// ---------------------------------------------------------------------------

fn time_displace_coord(uv: vec2f, covered_luma_in: f32) -> f32 {
    let map_code = u32(originals.loom_geometry.z);
    if map_code == 1u {
        // The picture times itself: bright things lag dark ones.
        return clamp(covered_luma_in, 0.0, 1.0);
    }
    if map_code == 2u {
        // Time pushed out from the centre, aspect-correct.
        let centered = vec2f((uv.x - 0.5) * originals.loom_geometry.y, uv.y - 0.5);
        return clamp(length(centered) * 1.6, 0.0, 1.0);
    }
    if map_code == 3u {
        // Per-scanline failure ramp: a sawtooth over each 8-line group.
        let height = f32(textureDimensions(current_tex).y);
        let line_phase = clamp(uv.y, 0.0, 1.0) * height / TIME_DISPLACE_TBC_LINES;
        return clamp(unit_fraction(line_phase), 0.0, 1.0);
    }
    if map_code == 4u {
        // Wrapped horizontal ramp travelling on the 30 Hz reference clock.
        return clamp(unit_fraction(uv.x - originals.atlas_values.z), 0.0, 1.0);
    }
    return clamp(dot(uv - 0.5, u.slit_direction) + 0.5, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// B3 feedback-rig helpers. This block is duplicated in lockstep in
// temporal_originals.wgsl, exactly as the frozen feedback expression already
// is. The CPU reference is `temporal::feedback_rig_reference`.
// ---------------------------------------------------------------------------

fn rig_active() -> bool {
    return rig.modes_b.y != 0u;
}

// Reflections and the per-tick offset act on the centred, already
// rotated/zoomed coordinate — the regime no rotation can reach.
fn rig_reflect_offset(p_centered: vec2f) -> vec2f {
    var p = p_centered;
    if rig.modes_a.x != 0u { p.x = -p.x; }
    if rig.modes_a.y != 0u { p.y = -p.y; }
    return p + rig.values_a.xy;
}

fn rig_mirror_unit(value: f32) -> f32 {
    let half = value / 2.0;
    let period = (half - floor(half)) * 2.0;
    return clamp(select(period, 2.0 - period, period > 1.0), 0.0, 1.0);
}

// Resolve the fed-back lookup under the frozen program-wide boundary
// numbering. xy is the resolved coordinate, z the coverage: Transparent is
// the exact historical inside test, and the other three always cover.
fn rig_resolve_edge(p: vec2f) -> vec3f {
    let edge = rig.modes_a.w;
    if edge == 1u {
        return vec3f(rig_mirror_unit(p.x), rig_mirror_unit(p.y), 1.0);
    }
    if edge == 2u {
        return vec3f(clamp(p - floor(p), vec2f(0.0), vec2f(1.0)), 1.0);
    }
    if edge == 3u {
        return vec3f(clamp(p, vec2f(0.0), vec2f(1.0)), 1.0);
    }
    let inside = select(0.0, 1.0,
        p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0);
    return vec3f(clamp(p, vec2f(0.0), vec2f(1.0)), inside);
}

fn rig_avalanche(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn rig_noise_sample(uv: vec2f) -> f32 {
    let px = vec2u(uv * vec2f(textureDimensions(feedback_tex)));
    let seed = rig_avalanche(
        px.x ^ (px.y * 0x9e3779b9u) ^ (rig.modes_b.x * 0x85ebca6bu) ^ 0x42335247u,
    );
    return f32(seed & 0x00ffffffu) / 16777216.0;
}

fn rig_shape_component(x: f32, shape: u32) -> f32 {
    if shape == 1u {
        // Soft: unit slope at the pivot, compressing to +/-0.5.
        return tanh(2.0 * x) * 0.5;
    }
    if shape == 2u {
        // Wrap into [-0.5, 0.5).
        return fract(x + 0.5) - 0.5;
    }
    if shape == 3u {
        // Triangular fold: identity through [-0.5, 0.5], reflecting beyond.
        let t = fract((x + 0.5) / 2.0) * 2.0;
        return select(t - 0.5, 1.5 - t, t > 1.0);
    }
    // Clamp.
    return clamp(x, -0.5, 0.5);
}

// The pointwise colour pipeline over the fed-back sample: gains, hue
// rotation, saturation, waveshaper, threshold decay, loop noise, and the
// deterministic compressive servo. The nonlinear stages mix toward identity
// by the clamped tick fraction so the loop stays rate-independent.
fn rig_grade(color_in: vec3f, frag_uv: vec2f) -> vec3f {
    var color = color_in * rig.values_d.xyz;
    let hue = rig.values_a.z;
    if abs(hue) > 1.0e-6 {
        let y = dot(color, vec3f(0.299, 0.587, 0.114));
        let i = dot(color, vec3f(0.596, -0.274, -0.322));
        let q = dot(color, vec3f(0.211, -0.523, 0.312));
        let c = cos(hue);
        let s = sin(hue);
        let i2 = i * c - q * s;
        let q2 = i * s + q * c;
        color = vec3f(
            y + 0.956 * i2 + 0.621 * q2,
            y - 0.272 * i2 - 0.647 * q2,
            y - 1.106 * i2 + 1.703 * q2,
        );
    }
    let saturation = rig.values_a.w;
    if abs(saturation - 1.0) > 1.0e-6 {
        let luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
        color = mix(vec3f(luma), color, saturation);
    }
    let tick_mix = clamp(rig.values_c.w, 0.0, 1.0);
    let drive = rig.values_b.w;
    let shape = rig.modes_a.z;
    if shape != 0u || abs(drive - 1.0) > 1.0e-6 {
        let pivot = rig.values_c.x;
        let x = (color - vec3f(pivot)) * drive;
        let shaped = vec3f(
            rig_shape_component(x.r, shape),
            rig_shape_component(x.g, shape),
            rig_shape_component(x.b, shape),
        ) + vec3f(pivot);
        color = mix(color, shaped, tick_mix);
    }
    let threshold = rig.values_c.y;
    if threshold > 0.0 {
        let luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
        let gate = smoothstep(threshold - 0.05, threshold + 0.05, luma);
        color = color * mix(1.0, gate, tick_mix);
    }
    let noise = rig.values_c.z;
    if noise > 0.0 {
        color = color + vec3f((rig_noise_sample(frag_uv) * 2.0 - 1.0) * noise * 0.25);
    }
    if rig.values_d.w > 0.0 {
        // Deterministic per-pixel auto-level: compress everything the loop
        // pushed above unity. Defeated, this stage is absent and the loop may
        // run to white or black and stay there.
        let luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
        let compressed = color / (1.0 + max(luma - 1.0, 0.0));
        color = mix(color, compressed, tick_mix * rig.values_d.w);
    }
    return max(color, vec3f(0.0));
}

// Sampling-dependent rig stages for the legacy (straight-alpha textureSample)
// variant: chromatic displacement of the lookup and the blur/sharpen
// activator-inhibitor pair over fixed two-texel cross taps.
fn rig_sample_legacy(p: vec2f) -> vec3f {
    var rgb = textureSample(feedback_tex, samp, p).rgb;
    let chroma = rig.values_b.x;
    if chroma > 0.0 {
        let d = vec2f(chroma, 0.0);
        rgb.r = textureSample(feedback_tex, samp, p + d).r;
        rgb.b = textureSample(feedback_tex, samp, p - d).b;
    }
    let blur = rig.values_b.y;
    let sharpen = rig.values_b.z;
    if blur > 0.0 || sharpen > 0.0 {
        let step = vec2f(2.0) / max(vec2f(textureDimensions(feedback_tex)), vec2f(1.0));
        let ring = textureSample(feedback_tex, samp, p + vec2f(step.x, 0.0)).rgb
            + textureSample(feedback_tex, samp, p - vec2f(step.x, 0.0)).rgb
            + textureSample(feedback_tex, samp, p + vec2f(0.0, step.y)).rgb
            + textureSample(feedback_tex, samp, p - vec2f(0.0, step.y)).rgb;
        let blurred = (rgb + ring) / 5.0;
        rgb = mix(rgb, blurred, clamp(blur, 0.0, 1.0)) + (rgb - blurred) * sharpen;
    }
    return rgb;
}


fn unit_fraction(value: f32) -> f32 {
    return value - floor(value);
}

fn triangle_wave(value: f32) -> f32 {
    return 1.0 - abs(unit_fraction(value) * 2.0 - 1.0);
}

fn quantize_age(age: f32) -> f32 {
    let levels = originals.loom_modes.w;
    let bounded = clamp(age, 0.0, 1.0);
    if levels == 0u {
        return bounded;
    }
    if levels == 1u {
        return 0.0;
    }
    let intervals = f32(levels - 1u);
    return round(bounded * intervals) / intervals;
}

fn loom_age(uv: vec2f) -> f32 {
    let scale = originals.loom_values.w;
    let aspect = originals.loom_geometry.y;
    let angle = originals.loom_geometry.x;
    let c = cos(angle);
    let s = sin(angle);
    let centered = vec2f((uv.x - 0.5) * aspect, uv.y - 0.5) * scale;
    let p = vec2f(
        centered.x * c + centered.y * s,
        -centered.x * s + centered.y * c,
    );
    let radius = length(p);
    let turn = atan2(p.y, p.x) / TAU;
    let phase = originals.loom_values.z;
    let folds = f32(originals.loom_modes.z);
    let topology = originals.loom_modes.x;
    var raw = p.y + 0.5 + phase;
    if topology == 1u {
        raw = radius * 2.0 + phase;
    } else if topology == 2u {
        raw = radius * 2.0 + turn + phase;
    } else if topology == 3u {
        raw = (abs(p.x) + abs(p.y)) * 2.0 + phase;
    } else if topology == 4u {
        raw = triangle_wave((p.y + 0.5) * folds + phase);
    } else if topology == 5u {
        let sector = triangle_wave((turn + 0.5) * folds);
        raw = radius * 2.0 + sector + phase;
    }
    return quantize_age(unit_fraction(raw));
}

fn mix_u32(input: u32) -> u32 {
    var value = input;
    value = value ^ (value >> 16u);
    value = value * 0x7feb352du;
    value = value ^ (value >> 15u);
    value = value * 0x846ca68bu;
    return value ^ (value >> 16u);
}

fn hash_unit(value: u32) -> f32 {
    return f32(value >> 8u) * (1.0 / 16777216.0);
}

fn atlas_seed() -> u32 {
    var seed = originals.atlas_modes.x;
    if (originals.atlas_modes.w & 1u) != 0u {
        let conductor = originals.score_runtime.x
            ^ (originals.score_runtime.y * 0x9e3779b9u)
            ^ originals.score_runtime.z
            ^ originals.score_runtime.w;
        seed = seed ^ mix_u32(conductor);
    }
    return seed;
}

fn atlas_cell_hash(cell: vec2i, seed: u32) -> u32 {
    let combined = (bitcast<u32>(cell.x) * 0x8da6b343u)
        ^ (bitcast<u32>(cell.y) * 0xd8163841u)
        ^ seed;
    return mix_u32(combined);
}

// x = history age, y = analytical cellular ridge. Sharing the bounded 3x3
// search keeps the Atlas+Garden combination deterministic and texture-free.
fn collision_atlas_field(uv: vec2f) -> vec2f {
    let territories = max(f32(originals.atlas_modes.y), 1.0);
    let grid_scale = sqrt(territories);
    let p = vec2f(
        uv.x * originals.loom_geometry.y * grid_scale,
        uv.y * grid_scale,
    );
    let base = vec2i(floor(p));
    let seed = atlas_seed();
    var nearest_distance = 1.0e30;
    var second_distance = 1.0e30;
    var nearest_age = 0.0;
    var second_age = 0.0;
    for (var y_offset = -1; y_offset <= 1; y_offset += 1) {
        for (var x_offset = -1; x_offset <= 1; x_offset += 1) {
            let cell = base + vec2i(x_offset, y_offset);
            let hash = atlas_cell_hash(cell, seed);
            let feature = vec2f(cell) + vec2f(
                hash_unit(hash),
                hash_unit(mix_u32(hash ^ 0x68bc21ebu)),
            );
            let delta = feature - p;
            let distance = dot(delta, delta);
            let age = hash_unit(mix_u32(hash ^ 0x02e5be93u));
            if distance < nearest_distance {
                second_distance = nearest_distance;
                second_age = nearest_age;
                nearest_distance = distance;
                nearest_age = age;
            } else if distance < second_distance {
                second_distance = distance;
                second_age = age;
            }
        }
    }
    let ridge_gap = sqrt(second_distance) - sqrt(nearest_distance);
    let ridge = 1.0 - smoothstep(0.0, 0.35, ridge_gap);
    return vec2f(
        mix(nearest_age, second_age, ridge * originals.atlas_values.y),
        ridge,
    );
}

fn covered_luma(color: vec4f) -> f32 {
    return dot(color.rgb * color.a, vec3f(0.2126, 0.7152, 0.0722));
}

fn covered_chroma(color: vec4f) -> f32 {
    let covered = color.rgb * color.a;
    return max(max(covered.r, covered.g), covered.b)
        - min(min(covered.r, covered.g), covered.b);
}

fn garden_gate(signal: f32) -> f32 {
    let threshold = clamp(originals.garden_values.y, 0.0, 1.0);
    let softness = clamp(originals.garden_values.z, 0.0, 0.5);
    if softness <= 0.000001 {
        return select(0.0, 1.0, signal >= threshold);
    }
    return smoothstep(
        max(threshold - softness, 0.0),
        min(threshold + softness, 1.0),
        clamp(signal, 0.0, 1.0),
    );
}

fn history_age_sample(current: vec4f, uv: vec2f, discrete_age: u32) -> vec4f {
    if discrete_age == 0u {
        return current;
    }
    let layer = wrap_layer(u.write_index - f32(discrete_age));
    return textureSample(history_tex, samp, uv, layer);
}

fn woven_history(current: vec4f, uv: vec2f, normalized_age: f32) -> vec4f {
    let max_depth = max(u.valid_history - 1.0, 0.0);
    let requested = normalized_age * originals.loom_values.y * (u.history_len - 1.0);
    let depth = clamp(requested, 0.0, max_depth);
    let lower_age = u32(floor(depth));
    if originals.loom_modes.y == 0u {
        return history_age_sample(current, uv, lower_age);
    }
    let upper_age = u32(ceil(depth));
    let lower = history_age_sample(current, uv, lower_age);
    let upper = history_age_sample(current, uv, upper_age);
    return mix(lower, upper, fract(depth));
}

fn legacy_originals(uv: vec2f) -> vec4f {
    let current = textureSample(current_tex, samp, uv);
    var color = current;
    let garden_amount = originals.garden_values.x;

    // Frozen legacy slit-scan arithmetic/order, generalized by the B12
    // time-displace map. Ramp with interpolation off routes through
    // `history_age_sample` with the identical layer arithmetic, so the
    // default path stays pixel-exact; interpolation costs at most one extra
    // history load per pixel and the depth still clamps against the
    // valid-history counter exactly as History Key does.
    if u.slitscan > 0.001 && u.valid_history > 0.5 {
        let coord = time_displace_coord(uv, covered_luma(current));
        let max_depth = max(u.valid_history - 1.0, 0.0);
        let requested_depth = coord * u.slitscan * (u.history_len - 1.0);
        let depth = min(requested_depth, max_depth);
        if originals.loom_geometry.w > 0.5 {
            let lower = history_age_sample(current, uv, u32(floor(depth)));
            let upper = history_age_sample(current, uv, u32(ceil(depth)));
            color = mix(lower, upper, fract(depth));
        } else if requested_depth >= 1.0 && max_depth >= 1.0 {
            color = history_age_sample(current, uv, u32(floor(depth)));
        }
    }

    // Topology Loom and Collision Atlas share the clean ring. Age zero is the
    // virtual current image; only positive, validity-bounded ages touch the
    // array, so freshly allocated/unwritten layers are never sampled.
    let loom_amount = originals.loom_values.x;
    let atlas_amount = originals.atlas_values.x;
    let originals_amount = max(loom_amount, atlas_amount);
    var atlas_field = vec2f(0.0);
    if atlas_amount > 0.0 || originals.garden_modes.x == 3u {
        atlas_field = collision_atlas_field(uv);
    }
    if originals_amount > 0.0 && u.valid_history > 0.5 {
        var age = select(clamp(uv.y, 0.0, 1.0), loom_age(uv), loom_amount > 0.0);
        if atlas_amount > 0.0 {
            age = mix(age, atlas_field.x, atlas_amount);
            age = quantize_age(age);
        }
        let woven = woven_history(current, uv, age);
        color = mix(color, woven, originals_amount);
    }

    // One carrier read serves both frozen feedback and Refresh Garden. The
    // legacy feedback transform is also Garden's bounded identity/warp law.
    var previous = vec4f(0.0);
    var previous_inside = 0.0;
    if (garden_amount > 0.0 || (u.feedback > 0.001 && !rig_active()))
        && u.feedback_valid > 0.5 {
        let angle = u.fb_rotate * 0.0174533;
        let c = cos(angle);
        let s = sin(angle);
        var p = uv - 0.5;
        p = vec2f(p.x * c - p.y * s, p.x * s + p.y * c) / max(u.fb_zoom, 0.01);
        p += 0.5;
        previous = textureSample(feedback_tex, samp, p);
        previous_inside = select(0.0, 1.0,
            p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0);
    }

    // Frozen legacy feedback arithmetic/order. An active rig takes its own
    // transformed read so Garden keeps the frozen shared carrier untouched.
    if u.feedback > 0.001 && u.feedback_valid > 0.5 {
        if rig_active() {
            let angle = u.fb_rotate * 0.0174533;
            let c = cos(angle);
            let s = sin(angle);
            var p = uv - 0.5;
            p = vec2f(p.x * c - p.y * s, p.x * s + p.y * c) / max(u.fb_zoom, 0.01);
            let resolved = rig_resolve_edge(rig_reflect_offset(p) + 0.5);
            let prev_rgb = rig_grade(rig_sample_legacy(resolved.xy), uv);
            color = vec4f(max(color.rgb, prev_rgb * u.feedback * resolved.z), color.a);
        } else {
            color = vec4f(max(color.rgb, previous.rgb * u.feedback * previous_inside), color.a);
        }
    }

    // Frozen legacy temporal-key arithmetic/order.
    if u.key_mode > 0.5 && u.key_valid > 0.5 {
        let reference = textureSample(history_tex, samp, uv, i32(u.key_reference_layer));
        let current_covered = current.rgb * current.a;
        let reference_covered = reference.rgb * reference.a;
        var signal = 0.0;
        if u.key_mode < 2.5 {
            signal = length(current_covered - reference_covered) * 0.577350269;
        } else {
            let current_luma = dot(current_covered, vec3f(0.2126, 0.7152, 0.0722));
            let reference_luma = dot(reference_covered, vec3f(0.2126, 0.7152, 0.0722));
            if u.key_mode < 3.5 {
                signal = max(current_luma - reference_luma, 0.0);
            } else {
                signal = max(reference_luma - current_luma, 0.0);
            }
        }
        let threshold = clamp(u.key_threshold, 0.0, 1.0);
        let softness = max(clamp(u.key_softness, 0.0, 0.5), 0.001);
        var mask = smoothstep(threshold - softness, threshold + softness, signal);
        if u.key_mode > 1.5 && u.key_mode < 2.5 {
            mask = 1.0 - mask;
        }
        color.a *= mask;
    }

    // Refresh Garden reuses the existing post-temporal feedback texture as
    // its sole carrier. It observes only fixed reference ticks, and batches
    // identical observations with the closed form of the authored recurrence:
    // memory = memory * decay * (1 - admission) + current * admission.
    if garden_amount > 0.0 && u.feedback_valid > 0.5 {
        let carrier = previous * previous_inside;
        let ticks = originals.garden_modes.z;
        if ticks == 0u {
            color = carrier;
        } else {
            let runtime = originals.garden_modes.w;
            let audio_energy = f32(runtime & 0xffffu) * (1.0 / 65535.0);
            let audio_onset = (runtime & (1u << 16u)) != 0u;
            let force_refresh = (runtime & (1u << 17u)) != 0u;
            let gate_mode = originals.garden_modes.x;
            var signal = length(color.rgb * color.a - carrier.rgb * carrier.a)
                * 0.577350269;
            if gate_mode == 1u {
                signal = covered_luma(color);
            } else if gate_mode == 2u {
                signal = covered_chroma(color);
            } else if gate_mode == 3u {
                signal = atlas_field.y;
            } else if gate_mode == 4u {
                signal = audio_energy;
            } else if gate_mode == 5u {
                signal = select(0.0, 1.0, audio_onset);
            } else if gate_mode == 6u {
                // Preserve the frozen LegacyExact active-Matte law. Advanced
                // routed Matte is evaluated by the dedicated post-temporal
                // pass; changing this branch would alter pre-M6 Compat8 pixels.
                signal = current.a;
            } else if gate_mode == 7u {
                // A routed motion field is bound by the dedicated Garden
                // signal path. Until that binding is admitted, Motion is an
                // honest closed gate rather than falling back to delta.
                signal = 0.0;
            }
            let opened = select(garden_gate(signal), 1.0, force_refresh);
            let admission = clamp(garden_amount * opened, 0.0, 1.0);
            let decay = clamp(originals.garden_values.w, 0.0, 1.0);
            let coefficient = decay * (1.0 - admission);
            let retained = pow(coefficient, f32(ticks));
            var injected = 0.0;
            if admission > 0.0 {
                injected = admission * (1.0 - retained) / max(1.0 - coefficient, 0.000001);
            }
            color = carrier * retained + color * injected;
        }
    }

    return color;
}

fn premultiply_originals(straight: vec4f) -> vec4f {
    let alpha = clamp(straight.a, 0.0, 1.0);
    return vec4f(straight.rgb * alpha, alpha);
}

fn straight_from_originals_premultiplied(value: vec4f) -> vec4f {
    let alpha = clamp(value.a, 0.0, 1.0);
    if alpha <= 0.000001 { return vec4f(0.0); }
    return vec4f(value.rgb / alpha, alpha);
}

fn advanced_feedback_premultiplied_linear(uv: vec2f) -> vec4f {
    let dimensions = vec2i(textureDimensions(feedback_tex));
    let coordinate = uv * vec2f(dimensions) - vec2f(0.5);
    let base = vec2i(floor(coordinate));
    let fraction = fract(coordinate);
    let maximum = dimensions - vec2i(1);
    let p00 = clamp(base, vec2i(0), maximum);
    let p10 = clamp(base + vec2i(1, 0), vec2i(0), maximum);
    let p01 = clamp(base + vec2i(0, 1), vec2i(0), maximum);
    let p11 = clamp(base + vec2i(1, 1), vec2i(0), maximum);
    let c00 = premultiply_originals(textureLoad(feedback_tex, p00, 0));
    let c10 = premultiply_originals(textureLoad(feedback_tex, p10, 0));
    let c01 = premultiply_originals(textureLoad(feedback_tex, p01, 0));
    let c11 = premultiply_originals(textureLoad(feedback_tex, p11, 0));
    return mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y);
}

// The same stages for the advanced (covered premultiplied load) variant.
fn rig_sample_advanced(p: vec2f) -> vec4f {
    var previous = advanced_feedback_premultiplied_linear(p);
    let chroma = rig.values_b.x;
    if chroma > 0.0 {
        let d = vec2f(chroma, 0.0);
        previous.r = advanced_feedback_premultiplied_linear(p + d).r;
        previous.b = advanced_feedback_premultiplied_linear(p - d).b;
    }
    let blur = rig.values_b.y;
    let sharpen = rig.values_b.z;
    if blur > 0.0 || sharpen > 0.0 {
        let step = vec2f(2.0) / max(vec2f(textureDimensions(feedback_tex)), vec2f(1.0));
        let ring = advanced_feedback_premultiplied_linear(p + vec2f(step.x, 0.0)).rgb
            + advanced_feedback_premultiplied_linear(p - vec2f(step.x, 0.0)).rgb
            + advanced_feedback_premultiplied_linear(p + vec2f(0.0, step.y)).rgb
            + advanced_feedback_premultiplied_linear(p - vec2f(0.0, step.y)).rgb;
        let blurred = (previous.rgb + ring) / 5.0;
        previous = vec4f(
            mix(previous.rgb, blurred, clamp(blur, 0.0, 1.0))
                + (previous.rgb - blurred) * sharpen,
            previous.a,
        );
    }
    return previous;
}

fn advanced_history_age_sample(current: vec4f, uv: vec2f, discrete_age: u32) -> vec4f {
    if discrete_age == 0u { return current; }
    let layer = wrap_layer(u.write_index - f32(discrete_age));
    return premultiply_originals(textureSample(history_tex, samp, uv, layer));
}

fn advanced_woven_history(current: vec4f, uv: vec2f, normalized_age: f32) -> vec4f {
    let max_depth = max(u.valid_history - 1.0, 0.0);
    let requested = normalized_age * originals.loom_values.y * (u.history_len - 1.0);
    let depth = clamp(requested, 0.0, max_depth);
    let lower_age = u32(floor(depth));
    if originals.loom_modes.y == 0u {
        return advanced_history_age_sample(current, uv, lower_age);
    }
    let upper_age = u32(ceil(depth));
    let lower = advanced_history_age_sample(current, uv, lower_age);
    let upper = advanced_history_age_sample(current, uv, upper_age);
    return mix(lower, upper, fract(depth));
}

fn advanced_originals(uv: vec2f) -> vec4f {
    let current = premultiply_originals(textureSample(current_tex, samp, uv));
    var color = current;
    let garden_amount = originals.garden_values.x;

    // The premultiplied twin of the legacy block above; `current` is already
    // premultiplied here, so its covered luma is a direct dot product.
    if u.slitscan > 0.001 && u.valid_history > 0.5 {
        let coord = time_displace_coord(uv, dot(current.rgb, vec3f(0.2126, 0.7152, 0.0722)));
        let max_depth = max(u.valid_history - 1.0, 0.0);
        let requested_depth = coord * u.slitscan * (u.history_len - 1.0);
        let depth = min(requested_depth, max_depth);
        if originals.loom_geometry.w > 0.5 {
            let lower = advanced_history_age_sample(current, uv, u32(floor(depth)));
            let upper = advanced_history_age_sample(current, uv, u32(ceil(depth)));
            color = mix(lower, upper, fract(depth));
        } else if requested_depth >= 1.0 && max_depth >= 1.0 {
            color = advanced_history_age_sample(current, uv, u32(floor(depth)));
        }
    }

    let loom_amount = originals.loom_values.x;
    let atlas_amount = originals.atlas_values.x;
    let originals_amount = max(loom_amount, atlas_amount);
    var atlas_field = vec2f(0.0);
    if atlas_amount > 0.0 || originals.garden_modes.x == 3u {
        atlas_field = collision_atlas_field(uv);
    }
    if originals_amount > 0.0 && u.valid_history > 0.5 {
        var age = select(clamp(uv.y, 0.0, 1.0), loom_age(uv), loom_amount > 0.0);
        if atlas_amount > 0.0 {
            age = mix(age, atlas_field.x, atlas_amount);
            age = quantize_age(age);
        }
        let woven = advanced_woven_history(current, uv, age);
        color = mix(color, woven, originals_amount);
    }

    var previous = vec4f(0.0);
    var previous_inside = 0.0;
    if (garden_amount > 0.0 || (u.feedback > 0.001 && !rig_active()))
        && u.feedback_valid > 0.5 {
        let angle = u.fb_rotate * 0.0174533;
        let c = cos(angle);
        let s = sin(angle);
        var p = uv - 0.5;
        p = vec2f(p.x * c - p.y * s, p.x * s + p.y * c) / max(u.fb_zoom, 0.01);
        p += 0.5;
        previous = advanced_feedback_premultiplied_linear(p);
        previous_inside = select(0.0, 1.0,
            p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0);
    }

    if u.feedback > 0.001 && u.feedback_valid > 0.5 {
        if rig_active() {
            let angle = u.fb_rotate * 0.0174533;
            let c = cos(angle);
            let s = sin(angle);
            var p = uv - 0.5;
            p = vec2f(p.x * c - p.y * s, p.x * s + p.y * c) / max(u.fb_zoom, 0.01);
            let resolved = rig_resolve_edge(rig_reflect_offset(p) + 0.5);
            let rigged = rig_sample_advanced(resolved.xy);
            let retention = u.feedback * resolved.z;
            color = vec4f(
                max(color.rgb, rig_grade(rigged.rgb, uv) * retention),
                max(color.a, rigged.a * retention),
            );
        } else {
            let retention = u.feedback * previous_inside;
            color = vec4f(
                max(color.rgb, previous.rgb * retention),
                max(color.a, previous.a * retention),
            );
        }
    }

    if u.key_mode > 0.5 && u.key_valid > 0.5 {
        let reference = premultiply_originals(
            textureSample(history_tex, samp, uv, i32(u.key_reference_layer)),
        );
        var signal = 0.0;
        if u.key_mode < 2.5 {
            signal = length(current.rgb - reference.rgb) * 0.577350269;
        } else {
            let current_luma = dot(current.rgb, vec3f(0.2126, 0.7152, 0.0722));
            let reference_luma = dot(reference.rgb, vec3f(0.2126, 0.7152, 0.0722));
            if u.key_mode < 3.5 {
                signal = max(current_luma - reference_luma, 0.0);
            } else {
                signal = max(reference_luma - current_luma, 0.0);
            }
        }
        let threshold = clamp(u.key_threshold, 0.0, 1.0);
        let softness = max(clamp(u.key_softness, 0.0, 0.5), 0.001);
        var mask = smoothstep(threshold - softness, threshold + softness, signal);
        if u.key_mode > 1.5 && u.key_mode < 2.5 { mask = 1.0 - mask; }
        color *= mask;
    }

    if garden_amount > 0.0 && u.feedback_valid > 0.5 {
        let carrier = previous * previous_inside;
        let ticks = originals.garden_modes.z;
        if ticks == 0u {
            color = carrier;
        } else {
            let runtime = originals.garden_modes.w;
            let audio_energy = f32(runtime & 0xffffu) * (1.0 / 65535.0);
            let audio_onset = (runtime & (1u << 16u)) != 0u;
            let force_refresh = (runtime & (1u << 17u)) != 0u;
            let gate_mode = originals.garden_modes.x;
            var signal = length(color.rgb - carrier.rgb) * 0.577350269;
            if gate_mode == 1u {
                signal = dot(color.rgb, vec3f(0.2126, 0.7152, 0.0722));
            } else if gate_mode == 2u {
                signal = max(max(color.r, color.g), color.b)
                    - min(min(color.r, color.g), color.b);
            } else if gate_mode == 3u {
                signal = atlas_field.y;
            } else if gate_mode == 4u {
                signal = audio_energy;
            } else if gate_mode == 5u {
                signal = select(0.0, 1.0, audio_onset);
            } else if gate_mode == 6u {
                signal = 0.0;
            } else if gate_mode == 7u {
                signal = 0.0;
            }
            let opened = select(garden_gate(signal), 1.0, force_refresh);
            let admission = clamp(garden_amount * opened, 0.0, 1.0);
            let decay = clamp(originals.garden_values.w, 0.0, 1.0);
            let coefficient = decay * (1.0 - admission);
            let retained = pow(coefficient, f32(ticks));
            var injected = 0.0;
            if admission > 0.0 {
                injected = admission * (1.0 - retained) / max(1.0 - coefficient, 0.000001);
            }
            color = carrier * retained + color * injected;
        }
    }

    return straight_from_originals_premultiplied(color);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    if u._pad0 < 0.5 { return legacy_originals(uv); }
    return advanced_originals(uv);
}
