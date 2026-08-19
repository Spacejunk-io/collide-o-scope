// The fixed Study interpreter. This shader is compiled once and never
// generated: a validated, compiled Study arrives as a bounded instruction
// buffer (the Symmetry sector-table precedent), and this fragment stage
// walks it with a typed register file. Shader-source generation stays
// permanently refused by StudyAuthority; growing the instruction set is an
// ABI change under the R3 window, and opcode codes are append-only.
//
// Every semantic here mirrors src/study_eval.rs, the CPU reference this
// interpreter is checked against:
// - the R1 history guard clamps depth to valid_history - 1 with the virtual
//   current image at depth zero, exactly temporal_originals.wgsl's law;
// - the hue functions are rack_node.wgsl's, byte for byte (asserted by
//   source text on the CPU side), with the S10b unorm input clamp applied
//   outside them;
// - the bound law clamps every computed component to the ABI's
//   representable range after every instruction;
// - deterministic-random values were resolved to immediates at compile, so
//   the GPU never hashes and randomness cannot drift between the halves.
//
// Two sampled textures, no sampler: every lookup is a textureLoad.

struct StudyFrameUniforms {
    audio_bands0: vec4f,
    audio_bands1: vec4f,
    beat_phase: f32,
    instruction_count: u32,
    valid_history: u32,
    write_index: u32,
    history_len: u32,
    // Renderer-owned node wet and the frozen NodeBlend code; the engine-wide
    // node law below applies them after the interpreter output.
    wet: f32,
    blend_mode: u32,
    _pad0: u32,
};

// One encoded instruction. words.x carries the opcode in its low 16 bits and
// the auxiliary operand (mix amount / hue turns register, history age, audio
// band) in its high 16; words.y/z/w are dst, src a, src b. The immediate
// carries constants and resolved deterministic-random values.
struct StudyOp {
    words: vec4u,
    immediate: vec4f,
};

const STUDY_GPU_MAX_INSTRUCTIONS: u32 = 256u;
const STUDY_BOUND: f32 = 65504.0;

@group(0) @binding(0) var carrier_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d_array<f32>;
@group(0) @binding(2) var<uniform> frame: StudyFrameUniforms;
@group(0) @binding(3) var<uniform> program: array<StudyOp, STUDY_GPU_MAX_INSTRUCTIONS>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// --- rack_node.wgsl's hue law, byte for byte -------------------------------

fn rgb_to_hsl(c: vec3f) -> vec3f {
    let max_c = max(max(c.r, c.g), c.b);
    let min_c = min(min(c.r, c.g), c.b);
    let lightness = (max_c + min_c) * 0.5;
    let delta = max_c - min_c;
    if delta < 0.001 { return vec3f(0.0, 0.0, lightness); }
    let saturation = select(
        delta / (max_c + min_c),
        delta / (2.0 - max_c - min_c),
        lightness > 0.5,
    );
    var hue: f32;
    if max_c == c.r {
        hue = (c.g - c.b) / delta + select(0.0, 6.0, c.g < c.b);
    } else if max_c == c.g {
        hue = (c.b - c.r) / delta + 2.0;
    } else {
        hue = (c.r - c.g) / delta + 4.0;
    }
    return vec3f(hue / 6.0, saturation, lightness);
}

fn hue_to_rgb(p: f32, q: f32, initial: f32) -> f32 {
    var t = initial;
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 0.5 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

fn hsl_to_rgb(hsl: vec3f) -> vec3f {
    if hsl.y < 0.001 { return vec3f(hsl.z); }
    let q = select(
        hsl.z + hsl.y - hsl.z * hsl.y,
        hsl.z * (1.0 + hsl.y),
        hsl.z < 0.5,
    );
    let p = 2.0 * hsl.z - q;
    return vec3f(
        hue_to_rgb(p, q, hsl.x + 1.0 / 3.0),
        hue_to_rgb(p, q, hsl.x),
        hue_to_rgb(p, q, hsl.x - 1.0 / 3.0),
    );
}

// ---------------------------------------------------------------------------

fn bound(value: vec4f) -> vec4f {
    return clamp(value, vec4f(-STUDY_BOUND), vec4f(STUDY_BOUND));
}

fn audio_band(band: u32) -> f32 {
    // Bands were sanitized on the CPU before upload; this only selects.
    if band < 4u {
        return frame.audio_bands0[band];
    }
    if band < 8u {
        return frame.audio_bands1[band - 4u];
    }
    return 0.0;
}

// The R1 guard: depth clamps to valid_history - 1 (zero when nothing is
// committed) and depth zero is the virtual current image, never a stored
// layer — temporal_originals.wgsl's exact law in integer form.
fn guarded_history(current: vec4f, age: u32, pixel: vec2i) -> vec4f {
    var max_depth = 0u;
    if frame.valid_history > 0u {
        max_depth = frame.valid_history - 1u;
    }
    let effective = min(age, max_depth);
    if effective == 0u {
        return current;
    }
    let len = max(frame.history_len, 1u);
    let layer = (frame.write_index + len - (effective % len)) % len;
    return textureLoad(history_tex, pixel, i32(layer), 0);
}

fn hue_rotate(color: vec4f, turns: f32) -> vec4f {
    // The S10b domain clamp: the HSL round trip is defined on unorm colors,
    // so both evaluators clamp the rgb operand first (alpha passes through).
    var hsl = rgb_to_hsl(clamp(color.rgb, vec3f(0.0), vec3f(1.0)));
    hsl.x = fract(hsl.x + turns);
    let rgb = hsl_to_rgb(hsl);
    return vec4f(rgb, color.a);
}

// The engine-wide node wet/blend law, identical in shape to
// `rack_node.wgsl:apply_node_law` and `symmetry_field.wgsl:
// sym_apply_field_law`; `blend_rgb` comes from the one canonical blend
// kernel this shader is composed with.
fn study_apply_field_law(dry: vec4f, processed: vec4f) -> vec4f {
    let wet = clamp(frame.wet, 0.0, 1.0);
    if wet <= 0.0 { return dry; }
    var result: vec4f;
    if frame.blend_mode == BLEND_ALPHA_CUT {
        result = vec4f(dry.rgb, dry.a * (1.0 - clamp(processed.a, 0.0, 1.0)));
    } else {
        result = vec4f(
            blend_rgb(frame.blend_mode, clamp(dry.rgb, vec3f(0.0), vec3f(1.0)),
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
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let pixel = vec2i(in.position.xy);
    let current = bound(textureLoad(carrier_tex, pixel, 0));

    var registers: array<vec4f, 64>;
    var output = vec4f(0.0);

    let count = min(frame.instruction_count, STUDY_GPU_MAX_INSTRUCTIONS);
    for (var index = 0u; index < count; index++) {
        let op = program[index];
        let opcode = op.words.x & 0xffffu;
        let aux = op.words.x >> 16u;
        let dst = op.words.y;
        let a = op.words.z;
        let b = op.words.w;
        switch opcode {
            case 0u: { // LoadCurrentColor
                registers[dst] = current;
            }
            case 1u: { // LoadHistoryColor { age: aux }
                registers[dst] = bound(guarded_history(current, aux, pixel));
            }
            case 2u: { // LoadMotionVector — ABI 1.0's dead-end lane. No
                // opcode can carry a Vector2 to the output, so the value is
                // unobservable by construction and this pass binds no field.
                registers[dst] = vec4f(0.0);
            }
            case 3u: { // LoadAudioBand { band: aux }
                registers[dst] = vec4f(audio_band(aux));
            }
            case 4u: { // LoadBeatPhase
                registers[dst] = vec4f(frame.beat_phase);
            }
            case 5u, 6u: { // LoadDeterministicRandom (resolved) / ConstantScalar
                registers[dst] = vec4f(op.immediate.x);
            }
            case 7u, 8u: { // ConstantVector2 / ConstantColor
                registers[dst] = op.immediate;
            }
            case 9u: { // Add
                registers[dst] = bound(registers[a] + registers[b]);
            }
            case 10u: { // Subtract
                registers[dst] = bound(registers[a] - registers[b]);
            }
            case 11u: { // Multiply
                registers[dst] = bound(registers[a] * registers[b]);
            }
            case 12u: { // Mix { amount: aux }
                let t = registers[aux].x;
                registers[dst] = bound(registers[a] + (registers[b] - registers[a]) * t);
            }
            case 13u: { // Clamp01
                registers[dst] = clamp(registers[a], vec4f(0.0), vec4f(1.0));
            }
            case 14u: { // HueRotate { turns: aux }
                registers[dst] = bound(hue_rotate(registers[a], registers[aux].x));
            }
            case 15u: { // OutputColor
                output = clamp(registers[a], vec4f(0.0), vec4f(1.0));
            }
            default: {}
        }
    }
    return study_apply_field_law(current, output);
}
