// The B14 sync latch: the tape/NTSC horizontal shear, seated on the shared
// slot-0 seam between the B8 melting edge and the B4 display stage. A sync
// fault happens in the signal; the screen model downstream then shows it.
//
// This pass owns no law of its own. `sync_latch.rs` is the CPU reference: it
// draws the per-tick slips on the shared integer avalanche, folds them into
// the bounded per-line table while the switch is latched, and hands this
// shader nothing but the finished per-line offsets in output UV. All the
// fragment stage does is resample each line by its own offset — which is why
// live and offline are structurally identical, and why the whole latch law is
// testable on the CPU without a GPU.
//
// The horizontal wrap is the tape law: a line that loses sync slides and
// wraps around the frame rather than exposing an edge. `fract` puts the
// coordinate back inside the frame and the sampler repeats on U, so the
// bilinear tap that straddles the seam filters across it instead of clamping.
// Like the melting edge this pass reads through a filtering sampler (the
// opaque-resolve precedent at this same seam) and every tap is level-0, so no
// read requires implicit derivatives. The shear preserves coverage — a
// displaced line carries its own alpha — and the downstream opaque resolve
// still flattens exactly once.

// The per-line offsets ride the same uniform as the header. WGSL requires a
// 16-byte stride for a uniform array, so the tail is declared as vec4 lanes
// over exactly the bytes the host writes as a flat f32 run.
const SYNC_LATCH_VEC4_LANES: u32 = 540u;

struct LatchUniforms {
    resolution: vec2f,
    line_count: u32,
    armed: u32,
    offsets: array<vec4f, SYNC_LATCH_VEC4_LANES>,
};

@group(0) @binding(0) var image_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> latch: LatchUniforms;
@group(0) @binding(2) var latch_sampler: sampler;

struct LatchVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_latch(@builtin(vertex_index) vertex_index: u32) -> LatchVertexOutput {
    var out: LatchVertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// The line this fragment belongs to, clamped into the table the host sized to
// this output. A fragment can never address a line the host did not write.
fn latch_line(uv: vec2f) -> u32 {
    let last = max(latch.line_count, 1u) - 1u;
    let row = floor(uv.y * latch.resolution.y);
    return min(u32(max(row, 0.0)), last);
}

@fragment
fn fs_latch(@location(0) uv: vec2f) -> @location(0) vec4f {
    // The textually explicit default branch: a dormant stage samples the
    // identical coordinate, so slot 0 reaches the display stage untouched.
    // The host additionally skips encoding this pass entirely whenever every
    // offset is zero, so the exact prior path never resamples at all.
    if latch.armed == 0u {
        return textureSampleLevel(image_tex, latch_sampler, uv, 0.0);
    }
    let line = latch_line(uv);
    let offset = latch.offsets[line / 4u][line % 4u];
    let shifted = vec2f(fract(uv.x + offset), uv.y);
    return textureSampleLevel(image_tex, latch_sampler, shifted, 0.0);
}
