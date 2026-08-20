// The B1 Scan Processor: the tree's first non-fullscreen-triangle pass.
//
// One instanced triangle-strip ribbon per scanline, no vertex buffers —
// position comes from vertex_index/instance_index (the fullscreen-triangle
// tradition, extended) and the carrier is fetched in the VERTEX stage through
// the explicit-load premultiplied bilinear, sampler-free like every dedicated
// pass. Ribbons accumulate additively into a transient cleared to alpha one
// (contributions carry alpha zero so coverage cannot stack past unity where
// lines bunch), and a fullscreen resolve then applies the engine-wide node
// wet/blend law.
//
// The beam law is derived from BENDR (MIT, © 2026 Steve Blythe): the beamAt
// composition order and the beam-energy law `gain = 2 / speed` are
// transcribed faithfully with attribution. `scan_processor.rs` is the CPU
// reference this shader follows expression for expression.

struct ScanUniforms {
    // amount, ribbon_width, velocity_mix, collapse
    deflect: vec4f,
    // tilt_x, tilt_y, perspective, s_curve
    surface: vec4f,
    // skew, osc_amount, osc_freq, osc_lock
    osc: vec4f,
    // lissajous, mono, hue, time_seconds
    color_time: vec4f,
    // lines, samples_per_line, reverse_h, reverse_v
    raster: vec4f,
    // output width, output height, wet, reserved
    frame: vec4f,
    // blend code, reserved x3
    modes: vec4u,
    reserved: vec4f,
};

@group(0) @binding(0) var scan_carrier: texture_2d<f32>;
@group(0) @binding(1) var scan_accumulator: texture_2d<f32>;
@group(0) @binding(2) var<uniform> scan: ScanUniforms;

const SCAN_PI: f32 = 3.14159265;
const SCAN_LUMA_PIVOT: f32 = 0.35;
const SCAN_DEFLECT_SPAN: f32 = 1.6;
const SCAN_GAIN_MIN: f32 = 0.05;
const SCAN_GAIN_MAX: f32 = 8.0;
const SCAN_SPEED_FLOOR: f32 = 0.02;

// The explicit-load bilinear over covered color — rack_node.wgsl's
// `source_premultiplied_linear` filter without the straight conversion: the
// beam wants covered premultiplied light, so hostile RGB behind zero
// coverage steers and draws nothing by arithmetic.
fn scan_covered_bilinear(uv: vec2f) -> vec4f {
    let dimensions = vec2i(textureDimensions(scan_carrier));
    let coordinate = uv * vec2f(dimensions) - vec2f(0.5);
    let base = vec2i(floor(coordinate));
    let fraction = fract(coordinate);
    let maximum = dimensions - vec2i(1);
    let p00 = clamp(base, vec2i(0), maximum);
    let p10 = clamp(base + vec2i(1, 0), vec2i(0), maximum);
    let p01 = clamp(base + vec2i(0, 1), vec2i(0), maximum);
    let p11 = clamp(base + vec2i(1, 1), vec2i(0), maximum);
    let s00 = textureLoad(scan_carrier, p00, 0);
    let s10 = textureLoad(scan_carrier, p10, 0);
    let s01 = textureLoad(scan_carrier, p01, 0);
    let s11 = textureLoad(scan_carrier, p11, 0);
    let c00 = vec4f(s00.rgb * clamp(s00.a, 0.0, 1.0), clamp(s00.a, 0.0, 1.0));
    let c10 = vec4f(s10.rgb * clamp(s10.a, 0.0, 1.0), clamp(s10.a, 0.0, 1.0));
    let c01 = vec4f(s01.rgb * clamp(s01.a, 0.0, 1.0), clamp(s01.a, 0.0, 1.0));
    let c11 = vec4f(s11.rgb * clamp(s11.a, 0.0, 1.0), clamp(s11.a, 0.0, 1.0));
    return mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y);
}

fn scan_luma(covered_rgb: vec3f) -> f32 {
    return dot(covered_rgb, vec3f(0.2126, 0.7152, 0.0722));
}

// Where the beam is when it is this far along this line, before the ribbon
// is built around it — `scan_processor::beam_position`, expression for
// expression. Everything that bends the raster happens here, so it all
// composes with everything else.
fn scan_beam_position(sx: f32, line_v: f32, luma: f32) -> vec2f {
    var px = sx * 2.0 - 1.0;
    var py = 1.0 - line_v * 2.0;
    // S-curve: the continuous-wind yoke, bending the whole raster.
    px += sin(py * SCAN_PI) * scan.surface.w * 0.4;
    // Skew, which is the same control a bench monitor calls parallelogram.
    px += py * scan.osc.x * 0.5;
    // The deflection oscillators. Locked to a multiple of the field rate the
    // pattern stands still; detuned it crawls — the instrument's gesture.
    if scan.osc.y > 0.0005 {
        let f = floor(scan.osc.z * 12.0 + 0.5)
            + (1.0 - scan.osc.w) * fract(scan.osc.z * 12.0);
        let ph = py * f * SCAN_PI + scan.color_time.w * (1.0 - scan.osc.w) * 2.0;
        px += sin(ph) * scan.osc.y * 0.5;
        if scan.color_time.x > 0.0005 {
            py += sin(px * f * 1.61803 * SCAN_PI
                + scan.color_time.w * (1.0 - scan.osc.w) * 1.7)
                * scan.osc.y * scan.color_time.x * 0.5;
        }
    }
    // Raster collapse: remove the current from one deflection system and the
    // whole frame smears down onto a single line.
    py *= 1.0 - clamp(scan.deflect.w, 0.0, 1.0);
    // And the thing the machine is actually for: luminance into the vertical
    // position control.
    py += (luma - SCAN_LUMA_PIVOT) * scan.deflect.x * SCAN_DEFLECT_SPAN;
    // Tilt turns the deflection into an apparent surface — a photographed 2D
    // deflection, never a scene.
    let cx = cos(scan.surface.x);
    let sx2 = sin(scan.surface.x);
    let cy = cos(scan.surface.y);
    let sy = sin(scan.surface.y);
    let dz = (luma - SCAN_LUMA_PIVOT) * scan.deflect.x * SCAN_DEFLECT_SPAN;
    var q = vec3f(px, py, dz);
    q = vec3f(q.x * cy + q.z * sy, q.y, -q.x * sy + q.z * cy);
    q = vec3f(q.x, q.y * cx - q.z * sx2, q.y * sx2 + q.z * cx);
    let w = max(1.0 + q.z * scan.surface.z * 0.6, 0.15);
    return vec2f(q.x, q.y) / w;
}

// The discrete reversals mirror the *read*, never the drawn address.
fn scan_source_uv(sx: f32, line_v: f32) -> vec2f {
    var u = sx;
    var v = line_v;
    if scan.raster.z > 0.5 { u = 1.0 - u; }
    if scan.raster.w > 0.5 { v = 1.0 - v; }
    return vec2f(u, v);
}

fn scan_beam_at(sx: f32, line_v: f32) -> vec2f {
    let fetched = scan_covered_bilinear(scan_source_uv(sx, line_v));
    return scan_beam_position(sx, line_v, scan_luma(fetched.rgb));
}

// The beam colour law, applied per sample in the vertex stage so the CPU
// reference and the shader observe identical per-vertex values: mono folds
// to Rec.709 luma and colourise repaints from a luma-indexed HSV sweep
// scaled by the luma itself, so black stays black.
fn scan_hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let p = abs(fract(vec3f(h) + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - vec3f(3.0));
    return v * mix(vec3f(1.0), clamp(p - vec3f(1.0), vec3f(0.0), vec3f(1.0)), s);
}

fn scan_colorize(covered_rgb: vec3f) -> vec3f {
    let y = scan_luma(covered_rgb);
    var c = mix(covered_rgb, vec3f(y), scan.color_time.y);
    if scan.color_time.z > 0.002 {
        let swept = scan_hsv_to_rgb(fract(scan.color_time.z + y * 0.35), 0.85, 1.0);
        c = mix(c, swept * y, scan.color_time.z);
    }
    return c;
}

struct ScanVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) col: vec3f,
};

@vertex
fn vs_scan(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> ScanVertexOutput {
    let lines = max(scan.raster.x, 1.0);
    let samples = max(scan.raster.y, 2.0);
    let line_v = (f32(instance_index) + 0.5) / lines;
    let si = f32(vertex_index >> 1u);
    let side = select(-1.0, 1.0, (vertex_index & 1u) == 1u);
    let sx = si / (samples - 1.0);

    let fetched = scan_covered_bilinear(scan_source_uv(sx, line_v));
    let here = scan_beam_position(sx, line_v, scan_luma(fetched.rgb));
    // The tangent gives two things at once: which way to lay the ribbon, and
    // how fast the beam is travelling, which is what sets its brightness.
    let d = 1.0 / (samples - 1.0);
    let ahead = scan_beam_at(min(sx + d, 1.0), line_v);
    let back = scan_beam_at(max(sx - d, 0.0), line_v);
    let tang = ahead - back;
    let speed = max(length(tang) / (2.0 * d), SCAN_SPEED_FLOOR);
    let aspect = scan.frame.x / max(scan.frame.y, 1.0);
    var nrm = vec2f(-tang.y, tang.x * aspect);
    let nrm_len = length(nrm);
    // A degenerate tangent takes the vertical normal rather than dividing by
    // zero — the house non-finite law; no finite path changes.
    if nrm_len <= 0.000001 {
        nrm = vec2f(0.0, 1.0);
    } else {
        nrm = nrm / nrm_len;
    }
    let wclip = (0.7 + scan.deflect.y * 7.0) / max(scan.frame.y, 1.0) * 2.0;

    var out: ScanVertexOutput;
    out.position = vec4f(here + nrm * side * wclip, 0.0, 1.0);
    // A slower beam deposits more energy per unit length. Without this the
    // pass is a displacement map with extra steps: gain one at the nominal
    // two-clip-units-per-sweep speed, brighter where the displacement slows
    // the beam, dimmer where it is thrown across.
    let energetic = clamp(2.0 / speed, SCAN_GAIN_MIN, SCAN_GAIN_MAX);
    let gain = 1.0 + (energetic - 1.0) * scan.deflect.z;
    out.col = scan_colorize(fetched.rgb) * gain;
    return out;
}

@fragment
fn fs_scan(in: ScanVertexOutput) -> @location(0) vec4f {
    // Additive into a target cleared to alpha one: contribute none, or the
    // coverage would stack up past unity wherever lines bunch.
    return vec4f(in.col, 0.0);
}

// --- the fullscreen resolve ------------------------------------------------

struct ScanResolveOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_scan_resolve(@builtin(vertex_index) vertex_index: u32) -> ScanResolveOutput {
    var out: ScanResolveOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// The engine-wide node wet/blend law, identical in shape to
// `study_interpreter.wgsl:study_apply_field_law`; `blend_rgb` comes from the
// one canonical blend kernel this shader is composed with.
fn scan_apply_field_law(dry: vec4f, processed: vec4f) -> vec4f {
    let wet = clamp(scan.frame.z, 0.0, 1.0);
    if wet <= 0.0 { return dry; }
    var result: vec4f;
    if scan.modes.x == BLEND_ALPHA_CUT {
        result = vec4f(dry.rgb, dry.a * (1.0 - clamp(processed.a, 0.0, 1.0)));
    } else {
        result = vec4f(
            blend_rgb(scan.modes.x, clamp(dry.rgb, vec3f(0.0), vec3f(1.0)),
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
fn fs_scan_resolve(in: ScanResolveOutput) -> @location(0) vec4f {
    let pixel = vec2i(in.position.xy);
    let dry = textureLoad(scan_carrier, pixel, 0);
    let accumulated = textureLoad(scan_accumulator, pixel, 0);
    // The pass redraws the whole raster, so it legitimately claims full
    // coverage — the hardware does too. With alpha one the drawn image is
    // straight and premultiplied at once.
    let processed = vec4f(accumulated.rgb, 1.0);
    return scan_apply_field_law(dry, processed);
}
