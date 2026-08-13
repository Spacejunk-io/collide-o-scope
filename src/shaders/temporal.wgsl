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

@group(0) @binding(0) var current_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var feedback_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> u: TemporalUniforms;

fn wrap_layer(idx: f32) -> i32 {
    let n = u.history_len;
    return i32(((idx % n) + n) % n);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
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
        p += 0.5;
        let prev = textureSample(feedback_tex, samp, p);
        let inside = select(0.0, 1.0,
            p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0);
        color = vec4f(max(color.rgb, prev.rgb * u.feedback * inside), color.a);
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
