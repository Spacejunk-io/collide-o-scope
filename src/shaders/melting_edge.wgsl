// The B8 melting edge over the program's own coverage, seated after
// Temporal (and after any global stage upstream of it) and before the B4
// display stage and the opaque resolve.
//
// Every coverage boundary in the program — static key alpha, cellular gap,
// group matte — lands in the composite's alpha channel by this seam, so one
// mechanism melts them all: the coverage is probed at four points a chosen
// distance out; where they disagree the pixel stands on the edge and the
// disagreement direction is the edge normal. That yields a band of
// controlled width with a normal through it. The image is dragged along the
// normal inside the band, and the stage's own previous output is dissolved
// back in within the band, so the smear stays put and creeps a little
// further out every reference tick instead of washing out.
//
// Law derived from BENDR (MIT, © 2026 Steve Blythe), rewritten for this
// tree; `mixing_boundary.rs` is the CPU reference this shader follows
// expression for expression. The chroma law reconstructs RGB from Y/I/Q and
// therefore uses the coherent 601 YIQ round trip the B3 feedback rig
// carries. Unlike the display stage this pass reads through a filtering
// sampler (the opaque-resolve precedent at this same seam); every tap is
// level-0, so no read requires implicit derivatives. The stage preserves
// coverage — the trail legitimately carries alpha — and the downstream
// opaque resolve still flattens exactly once.

struct MeltUniforms {
    melt: f32,
    width: f32,
    hold: f32,
    swirl: f32,
    chroma: f32,
    creep: f32,
    hist_valid: u32,
    _pad0: f32,
    resolution: vec2f,
    output_aspect: f32,
    _pad1: f32,
};

@group(0) @binding(0) var image_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> melt: MeltUniforms;
@group(0) @binding(3) var melt_sampler: sampler;

struct MeltVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_melt(@builtin(vertex_index) vertex_index: u32) -> MeltVertexOutput {
    var out: MeltVertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn coverage_at(uv: vec2f) -> f32 {
    return clamp(
        textureSampleLevel(image_tex, melt_sampler, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0).a,
        0.0,
        1.0,
    );
}

// Coherent 601 YIQ round trip — the B3 feedback-rig matrices.
fn melt_rgb_to_yiq(rgb: vec3f) -> vec3f {
    return vec3f(
        dot(rgb, vec3f(0.299, 0.587, 0.114)),
        dot(rgb, vec3f(0.596, -0.274, -0.322)),
        dot(rgb, vec3f(0.211, -0.523, 0.312)),
    );
}

fn melt_yiq_to_rgb(yiq: vec3f) -> vec3f {
    return vec3f(
        yiq.x + 0.956 * yiq.y + 0.621 * yiq.z,
        yiq.x - 0.272 * yiq.y - 0.647 * yiq.z,
        yiq.x - 1.106 * yiq.y + 1.703 * yiq.z,
    );
}

@fragment
fn fs_melt(@location(0) uv: vec2f) -> @location(0) vec4f {
    // The four-point probe: X aspect-corrected, the resulting normal
    // deliberately not — the shipped anisotropy is the law.
    let r = 0.004 + melt.width * 0.085;
    let rx = vec2f(r / max(melt.output_aspect, 0.0001), 0.0);
    let ry = vec2f(0.0, r);
    let m_left = coverage_at(uv - rx);
    let m_right = coverage_at(uv + rx);
    let m_down = coverage_at(uv - ry);
    let m_up = coverage_at(uv + ry);
    let low = min(min(m_left, m_right), min(m_down, m_up));
    let high = max(max(m_left, m_right), max(m_down, m_up));
    var band = clamp((high - low) * 1.25, 0.0, 1.0);
    var en = vec2f(0.0);
    let g = vec2f(m_right - m_left, m_up - m_down);
    let g_len = length(g);
    if g_len > 0.00001 {
        en = g / g_len;
        let sa = melt.swirl * 1.5707964;
        en = vec2f(cos(sa) * en.x - sin(sa) * en.y, sin(sa) * en.x + cos(sa) * en.y);
    }
    // Creep pushes the melt onto the uncovered side, so a keyed shape bleeds
    // into the background while the background never eats the shape.
    let centre = coverage_at(uv);
    band *= mix(1.0, 1.0 - centre, melt.creep);

    let bd = en * band * melt.melt * 0.055;
    var color = textureSampleLevel(
        image_tex,
        melt_sampler,
        clamp(uv + bd, vec2f(0.0), vec2f(1.0)),
        0.0,
    );

    if melt.hist_valid != 0u && band > 0.001 && melt.hold > 0.002 {
        let pd = en * (0.0015 + melt.melt * 0.04);
        var held = textureSampleLevel(
            history_tex,
            melt_sampler,
            clamp(uv + pd, vec2f(0.0), vec2f(1.0)),
            0.0,
        );
        if melt.chroma > 0.002 {
            // Colour runs further than luma off the edge: a farther tap
            // donates only its chroma pair.
            let far = textureSampleLevel(
                history_tex,
                melt_sampler,
                clamp(uv + pd * (1.0 + 3.0 * melt.chroma), vec2f(0.0), vec2f(1.0)),
                0.0,
            );
            let near_yiq = melt_rgb_to_yiq(held.rgb);
            let far_yiq = melt_rgb_to_yiq(far.rgb);
            held = vec4f(
                melt_yiq_to_rgb(vec3f(
                    near_yiq.x,
                    mix(near_yiq.yz, far_yiq.yz, melt.chroma),
                )),
                held.a,
            );
        }
        let cap = min(0.94 + max(melt.hold - 1.0, 0.0) * 0.11, 0.995);
        color = mix(color, held, clamp(band * melt.hold, 0.0, cap));
    }
    return color;
}
