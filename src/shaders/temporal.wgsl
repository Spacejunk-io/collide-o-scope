// Temporal effects: feedback trails and slit-scan, fed by a ring buffer
// of past output frames stored as a texture array.
//
// Layer semantics: `write_index` is the layer holding the MOST RECENT
// completed frame (last frame's final output). Deeper history is at
// write_index - n (wrapping).

struct TemporalUniforms {
    feedback: f32,     // 0 = off, up to 0.95
    fb_zoom: f32,      // per-frame zoom applied to fed-back image
    fb_rotate: f32,    // per-frame rotation, degrees
    slitscan: f32,     // 0 = off, 1 = deepest reach into history
    history_len: f32,  // ring size
    write_index: f32,  // most recent layer
    slit_axis: f32,    // 0 = rows (y scans time), 1 = columns (x scans time)
    _pad: f32,
};

@group(0) @binding(0) var current_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
@group(1) @binding(0) var<uniform> u: TemporalUniforms;

fn wrap_layer(idx: f32) -> i32 {
    let n = u.history_len;
    return i32(((idx % n) + n) % n);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    var color = textureSample(current_tex, samp, uv);

    // --- Slit-scan: each row (or column) samples a different past frame.
    if u.slitscan > 0.001 {
        let coord = select(uv.y, uv.x, u.slit_axis > 0.5);
        let depth = coord * u.slitscan * (u.history_len - 1.0);
        if depth >= 1.0 {
            let layer = wrap_layer(u.write_index - floor(depth) + 1.0);
            let hist = textureSample(history_tex, samp, uv, layer);
            // Rows at depth < 1 stay live; deeper rows come from history.
            color = hist;
        }
    }

    // --- Feedback: blend last frame's output back in, zoomed and rotated.
    // max() gives decaying light trails rather than muddy averaging.
    if u.feedback > 0.001 {
        let angle = u.fb_rotate * 0.0174533;
        let c = cos(angle);
        let s = sin(angle);
        var p = uv - 0.5;
        p = vec2f(p.x * c - p.y * s, p.x * s + p.y * c) / max(u.fb_zoom, 0.01);
        p += 0.5;
        let prev = textureSample(history_tex, samp, p, wrap_layer(u.write_index));
        let inside = select(0.0, 1.0,
            p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0);
        color = vec4f(max(color.rgb, prev.rgb * u.feedback * inside), color.a);
    }

    return color;
}
