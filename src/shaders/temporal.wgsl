// Temporal effects: feedback trails and slit-scan.
//
// Two distinct memories serve two distinct needs:
// - history_tex (ring of CLEAN pre-temporal composites): slit-scan reads
//   real past frames here. Recording post-effect output instead would make
//   the effect consume itself — cold black gets recorded, sampled, and
//   re-recorded forever.
// - feedback_tex (last frame's POST-temporal output): trails must compound
//   on themselves, so feedback reads the finished previous frame.
//
// `write_index` is the ring layer holding the CURRENT clean frame; depth n
// of history is at write_index - n (wrapping).

struct TemporalUniforms {
    feedback: f32,     // 0 = off, up to 0.95
    fb_zoom: f32,      // per-frame zoom applied to fed-back image
    fb_rotate: f32,    // per-frame rotation, degrees
    slitscan: f32,     // 0 = off, 1 = deepest reach into history
    history_len: f32,  // ring size
    write_index: f32,  // most recent layer
    valid_history: f32,// number of initialized layers, oldest..current
    feedback_valid: f32,
    slit_direction: vec2f, // aspect-correct and normalized to span the frame
    key_reference_layer: f32,
    key_valid: f32,
    key_mode: f32,     // 0=off, 1=motion, 2=still, 3=brightening, 4=darkening
    key_threshold: f32,
    key_softness: f32,
    _pad0: f32,
};

// B3 feedback rig: everything the loop does to the fed-back sample beyond
// the frozen zoom/rotate/retention trio. The legacy 64-byte uniform stays
// frozen; the rig rides its own fixed binding, and `modes_b.y == 0` keeps
// the historical feedback expression untouched, byte for byte.
struct TemporalRigUniforms {
    values_a: vec4f, // offset_x, offset_y, hue rotate radians, saturation
    values_b: vec4f, // chroma displace, blur, sharpen, drive
    values_c: vec4f, // pivot, threshold, noise, tick mix
    values_d: vec4f, // gain r, gain g, gain b, servo strength
    modes_a: vec4u,  // reflect x, reflect y, shape code, edge code
    modes_b: vec4u,  // noise epoch, rig active, reserved, reserved
};

@group(0) @binding(0) var current_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var feedback_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> u: TemporalUniforms;
@group(1) @binding(2) var<uniform> rig: TemporalRigUniforms;

fn wrap_layer(idx: f32) -> i32 {
    let n = u.history_len;
    return i32(((idx % n) + n) % n);
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


fn legacy_temporal(uv: vec2f) -> vec4f {
    let current = textureSample(current_tex, samp, uv);
    var color = current;

    // --- Slit-scan: each row (or column) samples a different past frame.
    if u.slitscan > 0.001 && u.valid_history > 0.5 {
        let coord = clamp(dot(uv - 0.5, u.slit_direction) + 0.5, 0.0, 1.0);
        // Clamp reach to the portion of the ring that has actually been
        // written. This keeps startup deterministic without sampling the
        // undefined contents of freshly allocated GPU textures.
        let max_depth = max(u.valid_history - 1.0, 0.0);
        let requested_depth = coord * u.slitscan * (u.history_len - 1.0);
        let depth = min(requested_depth, max_depth);
        if requested_depth >= 1.0 && max_depth >= 1.0 {
            // Rows at depth < 1 stay live; deeper rows come from history.
            let layer = wrap_layer(u.write_index - floor(depth));
            let hist = textureSample(history_tex, samp, uv, layer);
            color = hist;
        }
    }

    // --- Feedback: blend last frame's output back in, zoomed and rotated.
    // max() gives decaying light trails rather than muddy averaging.
    if u.feedback > 0.001 && u.feedback_valid > 0.5 {
        let angle = u.fb_rotate * 0.0174533;
        let c = cos(angle);
        let s = sin(angle);
        var p = uv - 0.5;
        p = vec2f(p.x * c - p.y * s, p.x * s + p.y * c) / max(u.fb_zoom, 0.01);
        if rig_active() {
            let resolved = rig_resolve_edge(rig_reflect_offset(p) + 0.5);
            let prev_rgb = rig_grade(rig_sample_legacy(resolved.xy), uv);
            color = vec4f(max(color.rgb, prev_rgb * u.feedback * resolved.z), color.a);
        } else {
            p += 0.5;
            let prev = textureSample(feedback_tex, samp, p);
            let inside = select(0.0, 1.0,
                p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0);
            color = vec4f(max(color.rgb, prev.rgb * u.feedback * inside), color.a);
        }
    }

    // --- Temporal delta key. The CPU resolves the reference layer because
    // the 30 Hz history clock can either write or not write on a display
    // frame. `key_valid` guarantees that this sample was initialized.
    if u.key_mode > 0.5 && u.key_valid > 0.5 {
        let reference = textureSample(history_tex, samp, uv, i32(u.key_reference_layer));
        // Analyze premultiplied color so appearance/disappearance through
        // alpha is motion too, while leaving output RGB straight.
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

    return color;
}

fn premultiply_temporal(straight: vec4f) -> vec4f {
    let alpha = clamp(straight.a, 0.0, 1.0);
    return vec4f(straight.rgb * alpha, alpha);
}

fn straight_from_temporal_premultiplied(value: vec4f) -> vec4f {
    let alpha = clamp(value.a, 0.0, 1.0);
    if alpha <= 0.000001 { return vec4f(0.0); }
    return vec4f(value.rgb / alpha, alpha);
}

// Feedback zoom/rotation is the only spatially transformed lookup in the
// legacy temporal shader. Advanced uses four explicit loads so straight-alpha
// storage is interpolated as covered color; LegacyExact remains in
// `legacy_temporal` with its historical textureSample expression.
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
    let c00 = premultiply_temporal(textureLoad(feedback_tex, p00, 0));
    let c10 = premultiply_temporal(textureLoad(feedback_tex, p10, 0));
    let c01 = premultiply_temporal(textureLoad(feedback_tex, p01, 0));
    let c11 = premultiply_temporal(textureLoad(feedback_tex, p11, 0));
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

fn advanced_temporal(uv: vec2f) -> vec4f {
    let current = premultiply_temporal(textureSample(current_tex, samp, uv));
    var color = current;

    if u.slitscan > 0.001 && u.valid_history > 0.5 {
        let coord = clamp(dot(uv - 0.5, u.slit_direction) + 0.5, 0.0, 1.0);
        let max_depth = max(u.valid_history - 1.0, 0.0);
        let requested_depth = coord * u.slitscan * (u.history_len - 1.0);
        let depth = min(requested_depth, max_depth);
        if requested_depth >= 1.0 && max_depth >= 1.0 {
            let layer = wrap_layer(u.write_index - floor(depth));
            color = premultiply_temporal(textureSample(history_tex, samp, uv, layer));
        }
    }

    if u.feedback > 0.001 && u.feedback_valid > 0.5 {
        let angle = u.fb_rotate * 0.0174533;
        let c = cos(angle);
        let s = sin(angle);
        var p = uv - 0.5;
        p = vec2f(p.x * c - p.y * s, p.x * s + p.y * c) / max(u.fb_zoom, 0.01);
        if rig_active() {
            let resolved = rig_resolve_edge(rig_reflect_offset(p) + 0.5);
            let previous = rig_sample_advanced(resolved.xy);
            let retention = u.feedback * resolved.z;
            color = vec4f(
                max(color.rgb, rig_grade(previous.rgb, uv) * retention),
                max(color.a, previous.a * retention),
            );
        } else {
            p += 0.5;
            let previous = advanced_feedback_premultiplied_linear(p);
            let inside = select(0.0, 1.0,
                p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0);
            let retention = u.feedback * inside;
            color = vec4f(
                max(color.rgb, previous.rgb * retention),
                max(color.a, previous.a * retention),
            );
        }
    }

    if u.key_mode > 0.5 && u.key_valid > 0.5 {
        let reference = premultiply_temporal(
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

    return straight_from_temporal_premultiplied(color);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    if u._pad0 < 0.5 { return legacy_temporal(uv); }
    return advanced_temporal(uv);
}
