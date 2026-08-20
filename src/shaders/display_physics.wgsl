// The B4 display-physics stage: the field domain, phosphor persistence, and
// the display model, seated after Temporal and before the opaque resolve.
//
// Three fragment stages over one 128-byte uniform record:
//   fs_display_field    — FS_FIELD's recombination against the retained
//                         previous field (weave/bob/blend, twitter, 3:2).
//   fs_display_model    — the display pass: HV sag geometry, the phosphor
//                         max against the N-1 accumulator, defocus/bloom/
//                         halation over the fixed 12-tap ring, beam-profile
//                         scanlines, the mask families, mono/green tints.
//   fs_display_store    — FS_PHOS's accumulator law:
//                         max(current, previous * decay), decay already
//                         exponentiated per reference tick on the CPU.
//
// Laws derived from BENDR (MIT, © 2026 Steve Blythe), rewritten in linear
// light with Rec.709 luma. `display_physics.rs` is the CPU reference this
// shader follows expression for expression. Sampler-free: every read is a
// textureLoad, and the one filtered read (the sag-warped base) is the
// explicit-load covered bilinear. An ACTIVE stage flattens coverage — a
// screen has no transparency — so its output carries alpha one and the
// downstream opaque resolve becomes the identity on it; the flatten still
// happens exactly once.

struct DisplayUniforms {
    // il_amount, il_twitter, il_judder, phosphor
    fields: vec4f,
    // decay r, decay g, decay b (tick-exponentiated), scanlines
    decay: vec4f,
    // beam_width, beam_shape, mask_strength, mask_dark
    beam: vec4f,
    // bloom, bloom_radius, halation, defocus
    optics: vec4f,
    // sag, output width, output height, reserved
    frame: vec4f,
    // il_mode code, model code, field parity, flag bits
    modes: vec4u,
    reserved0: vec4f,
    reserved1: vec4f,
};

@group(0) @binding(0) var display_input: texture_2d<f32>;
@group(0) @binding(1) var display_aux: texture_2d<f32>;
@group(0) @binding(2) var<uniform> display: DisplayUniforms;

const DISPLAY_FLAG_JUDDER_HOLD: u32 = 1u;
const DISPLAY_FLAG_FIELD_VALID: u32 = 2u;
const DISPLAY_FLAG_PHOSPHOR_VALID: u32 = 4u;

struct DisplayVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_display(@builtin(vertex_index) vertex_index: u32) -> DisplayVertexOutput {
    var out: DisplayVertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn display_luma(rgb: vec3f) -> f32 {
    return dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
}

// Straight-alpha in, covered light out: the screen shows rgb * coverage
// over black, exactly what the opaque resolve computes. Alpha-one input
// (an upstream stage's own output) passes through unchanged.
fn covered_load(tex: texture_2d<f32>, pixel: vec2i) -> vec3f {
    let dimensions = vec2i(textureDimensions(tex));
    let clamped = clamp(pixel, vec2i(0), dimensions - vec2i(1));
    let raw = textureLoad(tex, clamped, 0);
    return raw.rgb * clamp(raw.a, 0.0, 1.0);
}

// The explicit-load covered bilinear for the one warped read.
fn covered_bilinear(tex: texture_2d<f32>, uv: vec2f) -> vec3f {
    let dimensions = vec2i(textureDimensions(tex));
    let coordinate = uv * vec2f(dimensions) - vec2f(0.5);
    let base = vec2i(floor(coordinate));
    let fraction = fract(coordinate);
    let c00 = covered_load(tex, base);
    let c10 = covered_load(tex, base + vec2i(1, 0));
    let c01 = covered_load(tex, base + vec2i(0, 1));
    let c11 = covered_load(tex, base + vec2i(1, 1));
    return mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y);
}

// --- the field domain --------------------------------------------------

@fragment
fn fs_display_field(in: DisplayVertexOutput) -> @location(0) vec4f {
    let pixel = vec2i(in.position.xy);
    let cur = covered_load(display_input, pixel);
    let amount = display.fields.x;
    if amount < 0.003 {
        return vec4f(cur, 1.0);
    }
    let parity = display.modes.z & 1u;
    let line_parity = u32(pixel.y) & 1u;
    let this_field = line_parity == parity;
    let field_valid = (display.modes.w & DISPLAY_FLAG_FIELD_VALID) != 0u;
    var prev = cur;
    if field_valid {
        prev = covered_load(display_aux, pixel);
    }
    var out: vec3f;
    switch display.modes.x {
        case 1u: {
            // BOB: only the current field is real; the gaps are filled from
            // its neighbours, so the picture jitters by half a line.
            if this_field {
                out = cur;
            } else {
                let dy = select(-1, 1, parity == 1u);
                out = covered_load(display_input, pixel + vec2i(0, dy));
            }
        }
        case 2u: {
            // BLEND: no comb, but everything that moves ghosts.
            out = mix(cur, prev, 0.5);
        }
        default: {
            // WEAVE: the two fields simply interleave — what an interlaced
            // signal actually is.
            out = select(prev, cur, this_field);
        }
    }
    // Twitter: high vertical frequency lands on one field only.
    if display.fields.y > 0.003 {
        let up = covered_load(display_input, pixel + vec2i(0, 1));
        let dn = covered_load(display_input, pixel - vec2i(0, 1));
        let hf = cur - (up + dn) * 0.5;
        let on_this = select(-1.0, 1.0, this_field);
        out += hf * on_this * display.fields.y * 1.6;
    }
    // 3:2 pulldown: held film frames lean on the previous field.
    if display.fields.z > 0.003 && (display.modes.w & DISPLAY_FLAG_JUDDER_HOLD) != 0u {
        out = mix(out, prev, display.fields.z * 0.85);
    }
    return vec4f(mix(cur, out, amount), 1.0);
}

// --- the display model --------------------------------------------------

fn mask_at(fc: vec2f, model: u32, mask_dark: f32) -> vec3f {
    let dark = 1.0 - mask_dark * 0.55;
    if model == 1u {
        // Aperture grille: three columns cycle R, G, B.
        let m = fc.x % 3.0;
        return vec3f(
            select(dark, 1.0, m < 1.0),
            select(dark, 1.0, m >= 1.0 && m < 2.0),
            select(dark, 1.0, m >= 2.0),
        );
    }
    if model == 2u {
        // Slot mask: staggered triads with a dark slot bar every sixth row.
        let gy = floor(fc.y / 6.0);
        let off = (gy % 2.0) * 1.5;
        let m = (fc.x + off) % 3.0;
        let v = select(dark, 1.0, fc.y % 6.0 < 5.0);
        return vec3f(
            select(dark, 1.0, m < 1.0),
            select(dark, 1.0, m >= 1.0 && m < 2.0),
            select(dark, 1.0, m >= 2.0),
        ) * v;
    }
    if model == 3u {
        // Shadow mask: the dot triad.
        let q = fc / vec2f(6.0, 6.0);
        let f = fract(q) - vec2f(0.5);
        let r = length(f);
        let tri = (floor(q.x) + floor(q.y) * 2.0) % 3.0;
        let t = vec3f(
            select(dark, 1.0, tri < 0.5),
            select(dark, 1.0, tri >= 0.5 && tri < 1.5),
            select(dark, 1.0, tri >= 1.5),
        );
        return t * (1.0 - smoothstep(0.28, 0.5, r) * mask_dark);
    }
    if model == 4u {
        // LCD stripe.
        let m = fc.x % 2.0;
        let stripe = select(dark, 1.06, m < 1.0);
        return vec3f(1.0 + (stripe - 1.0) * 0.85);
    }
    return vec3f(1.0);
}

@fragment
fn fs_display_model(in: DisplayVertexOutput) -> @location(0) vec4f {
    let resolution = vec2f(display.frame.y, max(display.frame.z, 1.0));
    var p = in.uv * 2.0 - vec2f(1.0);
    // HV sag: the raster grows as the mean picture gets brighter — the EHT
    // supply drooping under beam load. Measured at the picture centre.
    if display.frame.x > 0.003 {
        let centre = covered_load(display_input, vec2i(resolution * 0.5));
        let sag = display.frame.x * 0.035 * (display_luma(centre) + 0.35);
        p *= 1.0 - sag;
    }
    let cuv = clamp(p * 0.5 + vec2f(0.5), vec2f(0.0), vec2f(1.0));
    var c = covered_bilinear(display_input, cuv);
    // Phosphor persistence: the decay was applied in the store, so the
    // displayed trail is the N-1 accumulator as it stands.
    if display.fields.w > 0.003 && (display.modes.w & DISPLAY_FLAG_PHOSPHOR_VALID) != 0u {
        let acc = covered_bilinear(display_aux, cuv);
        c = max(c, acc);
    }
    // Defocus + bloom + halation over the fixed 12-tap gather ring.
    if display.optics.w > 0.003 || display.optics.x > 0.003 {
        let px = vec2f(1.0) / resolution;
        var blur = vec3f(0.0);
        var wsum = 0.0;
        let rad = 1.5 + display.optics.y * 16.0;
        for (var i = 0; i < 12; i++) {
            let a = f32(i) * 0.5236;
            let r = (1.0 + f32(i % 3)) * 0.45;
            let off = vec2f(cos(a), sin(a)) * rad * r * px;
            let tp = cuv + off;
            // A tap past the edge contributes nothing rather than stacking
            // the edge pixel into a bright rim.
            let inb = step(0.0, tp.x) * step(tp.x, 1.0) * step(0.0, tp.y) * step(tp.y, 1.0);
            let w = inb / (1.0 + r * 1.4);
            blur += covered_bilinear(display_input, clamp(tp, vec2f(0.0), vec2f(1.0))) * w;
            wsum += w;
        }
        blur = select(c, blur / wsum, wsum > 0.0001);
        c = mix(c, blur, display.optics.w * 0.85);
        if display.optics.x > 0.003 {
            let hot = max(blur - vec3f(0.42), vec3f(0.0)) * 1.9;
            let tint = mix(vec3f(1.0), vec3f(1.25, 0.62, 0.42), display.optics.z);
            c += hot * display.optics.x * 1.5 * tint;
        }
    }
    // The beam and the mask, gated by a non-Flat model.
    let model = display.modes.y;
    if model > 0u {
        // Beam-profile scanlines: a gaussian beam whose width tracks
        // brightness (the Lottes profile).
        let lines = resolution.y;
        let fy = fract(cuv.y * lines) - 0.5;
        let bright = display_luma(c);
        let w = display.beam.x * (0.35 + 0.65 * mix(1.0, bright, display.beam.y));
        let beam = exp(-(fy * fy) / max(0.005, w * w * 0.22));
        c *= mix(1.0, beam * 1.35, display.decay.w);
        c *= mix(vec3f(1.0), mask_at(in.position.xy, model, display.beam.w), display.beam.z);
        if model >= 5u {
            // Mono folds 85% of the way to luma; green-screen additionally
            // tints the phosphor.
            c = mix(c, vec3f(display_luma(c)), 0.85);
        }
        if model >= 6u {
            c *= vec3f(0.75, 1.0, 0.8);
        }
    }
    return vec4f(c, 1.0);
}

// --- the phosphor store --------------------------------------------------

@fragment
fn fs_display_store(in: DisplayVertexOutput) -> @location(0) vec4f {
    let pixel = vec2i(in.position.xy);
    // The accumulator keeps whichever is brighter: this frame's clean
    // signal (pre-field, per BENDR's store) or the decayed trail.
    let cur = covered_load(display_input, pixel);
    var trail = vec3f(0.0);
    if (display.modes.w & DISPLAY_FLAG_PHOSPHOR_VALID) != 0u {
        trail = covered_load(display_aux, pixel) * display.decay.rgb;
    }
    return vec4f(max(cur, trail), 1.0);
}
