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

const TAU: f32 = 6.28318530717958647692;

fn wrap_layer(idx: f32) -> i32 {
    let n = u.history_len;
    return i32(((idx % n) + n) % n);
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

    // Frozen legacy slit-scan arithmetic/order.
    if u.slitscan > 0.001 && u.valid_history > 0.5 {
        let coord = clamp(dot(uv - 0.5, u.slit_direction) + 0.5, 0.0, 1.0);
        let max_depth = max(u.valid_history - 1.0, 0.0);
        let requested_depth = coord * u.slitscan * (u.history_len - 1.0);
        let depth = min(requested_depth, max_depth);
        if requested_depth >= 1.0 && max_depth >= 1.0 {
            let layer = wrap_layer(u.write_index - floor(depth));
            let hist = textureSample(history_tex, samp, uv, layer);
            color = hist;
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
    if (u.feedback > 0.001 || garden_amount > 0.0) && u.feedback_valid > 0.5 {
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

    // Frozen legacy feedback arithmetic/order.
    if u.feedback > 0.001 && u.feedback_valid > 0.5 {
        color = vec4f(max(color.rgb, previous.rgb * u.feedback * previous_inside), color.a);
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

    if u.slitscan > 0.001 && u.valid_history > 0.5 {
        let coord = clamp(dot(uv - 0.5, u.slit_direction) + 0.5, 0.0, 1.0);
        let max_depth = max(u.valid_history - 1.0, 0.0);
        let requested_depth = coord * u.slitscan * (u.history_len - 1.0);
        let depth = min(requested_depth, max_depth);
        if requested_depth >= 1.0 && max_depth >= 1.0 {
            let layer = wrap_layer(u.write_index - floor(depth));
            color = premultiply_originals(textureSample(history_tex, samp, uv, layer));
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
    if (u.feedback > 0.001 || garden_amount > 0.0) && u.feedback_valid > 0.5 {
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
        let retention = u.feedback * previous_inside;
        color = vec4f(
            max(color.rgb, previous.rgb * retention),
            max(color.a, previous.a * retention),
        );
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
