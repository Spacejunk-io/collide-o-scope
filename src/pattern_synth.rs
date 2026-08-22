//! The B7 pattern-synth source law — a video synthesizer, not a test card.
//!
//! A pattern layer is the first source whose picture is computed rather than
//! decoded or received: coordinates flow through a framing stage (centre,
//! zoom, rotate, skew, domain warp), a shape stage turns position into a
//! scalar signal, an oscillator shapes that signal through a selected
//! waveform, cross-modulation feeds one axis's oscillator into the other's
//! phase, a wavefolder folds the signal back on itself — the hard banded
//! structure of analogue video synthesis — a comparator hardens edges, and a
//! colouriser maps the final scalar to RGB. Nothing here holds state: the
//! whole picture is a pure function of authored parameters and program time,
//! which is exactly what makes the source perfectly reconstructable offline.
//!
//! The shape, oscillator, cross-modulation, wavefolder, comparator, and
//! colouriser laws are derived from BENDR (MIT, © 2026 Steve Blythe) and
//! transcribed faithfully with attribution; the surrounding machinery is a
//! rewrite (Rust / wgpu 29 / WGSL). This module is the independent CPU
//! reference the GPU pass is checked against, in the `gesture.rs` tradition:
//! no wgpu, clock, filesystem, or UI dependency. The transcription keeps
//! BENDR's own numeric literals (including `3.14159` and `6.2831853`) so the
//! WGSL can mirror this file byte for byte.

/// The fixed pattern page. A computed source has no native resolution, so the
/// tree gives it one — fixed rather than output-sized, because an export at a
/// different output size must render the identical picture. 16:9 mirrors
/// BENDR's own processing frame.
pub const PATTERN_SYNTH_WIDTH: u32 = 1_920;
pub const PATTERN_SYNTH_HEIGHT: u32 = 1_080;

/// BENDR's own circle constant, kept byte-identical in the WGSL.
#[allow(
    clippy::approx_constant,
    reason = "BENDR's literal, mirrored byte for byte in pattern_synth.wgsl"
)]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const PATTERN_TAU: f32 = 6.283_185_3;

/// The shape stage: how position becomes the pre-oscillator signal. Codes are
/// permanent and append-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PatternShape {
    /// Two crossed ramps, one per axis, cross-modulating. The default.
    #[default]
    Scan,
    /// Rings of the oscillator driven by radius, angularly cross-modulated.
    Radial,
    /// The radial law with the angle folded into the phase: arms.
    Spiral,
    /// Four summed slow oscillators re-fractured by cross-modulation.
    Plasma,
    /// The product of two axis sines fed through the oscillator.
    Lissajous,
    /// Quantized radius bands.
    Rings,
    /// Angular spokes, radius-sheared by cross-modulation.
    Starburst,
    /// The maximum of the two axis oscillators: a lattice.
    Grid,
    /// Inverse-radius perspective rings with angular spokes.
    Tunnel,
    /// Worley-style moving feature points through the oscillator.
    Cells,
    /// Two point sources beating against each other.
    Interference,
    /// The oscillator driven by polygonal radius.
    Polygon,
    /// The exact Mandelbrot escape-time law `z <- z^2 + c`, on a fixed
    /// aspect-correct complex-plane page. This additive, non-BENDR shape is
    /// deliberately appended so every established shader code remains frozen.
    Mandelbrot,
    /// Compact moving splats composited with four analytically reconstructed
    /// earlier positions. The trail resembles retained visual memory without
    /// holding framebuffer state, so pause and offline export remain exact.
    MemorySplats,
    /// Moving anisotropic Gaussian kernels, alpha-composited into a bounded
    /// scalar field. This is a native 2D generator rather than a 3D point-cloud
    /// import pipeline.
    GaussianSplats,
}

impl PatternShape {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn gpu_code(self) -> u32 {
        match self {
            Self::Scan => 0,
            Self::Radial => 1,
            Self::Spiral => 2,
            Self::Plasma => 3,
            Self::Lissajous => 4,
            Self::Rings => 5,
            Self::Starburst => 6,
            Self::Grid => 7,
            Self::Tunnel => 8,
            Self::Cells => 9,
            Self::Interference => 10,
            Self::Polygon => 11,
            Self::Mandelbrot => 12,
            Self::MemorySplats => 13,
            Self::GaussianSplats => 14,
        }
    }

    pub const ALL: [Self; 15] = [
        Self::Scan,
        Self::Radial,
        Self::Spiral,
        Self::Plasma,
        Self::Lissajous,
        Self::Rings,
        Self::Starburst,
        Self::Grid,
        Self::Tunnel,
        Self::Cells,
        Self::Interference,
        Self::Polygon,
        Self::Mandelbrot,
        Self::MemorySplats,
        Self::GaussianSplats,
    ];
}

/// The oscillator waveform: one cycle per unit of phase. Codes are permanent
/// and append-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PatternWave {
    #[default]
    Sine,
    Triangle,
    Saw,
    Square,
    /// A square whose duty cycle is the authored pulse width.
    Pulse,
    /// Sample-and-hold: 48 hashed steps per cycle.
    SampleHold,
}

impl PatternWave {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn gpu_code(self) -> u32 {
        match self {
            Self::Sine => 0,
            Self::Triangle => 1,
            Self::Saw => 2,
            Self::Square => 3,
            Self::Pulse => 4,
            Self::SampleHold => 5,
        }
    }

    pub const ALL: [Self; 6] = [
        Self::Sine,
        Self::Triangle,
        Self::Saw,
        Self::Square,
        Self::Pulse,
        Self::SampleHold,
    ];
}

/// The colouriser: how the final scalar becomes RGB. Codes are permanent and
/// append-only. `RgbPhase` is BENDR's own default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PatternColorMode {
    /// The scalar as grey.
    Mono,
    /// Three copies of the oscillator, hue-offset per channel. The default.
    #[default]
    RgbPhase,
    /// The scalar sweeps hue.
    HsvSweep,
    /// A two-colour blend.
    Duotone,
    /// Quantized hue bands.
    Bands,
}

impl PatternColorMode {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn gpu_code(self) -> u32 {
        match self {
            Self::Mono => 0,
            Self::RgbPhase => 1,
            Self::HsvSweep => 2,
            Self::Duotone => 3,
            Self::Bands => 4,
        }
    }

    pub const ALL: [Self; 5] = [
        Self::Mono,
        Self::RgbPhase,
        Self::HsvSweep,
        Self::Duotone,
        Self::Bands,
    ];
}

/// The authored pattern-synth state. Every continuous field is a modulation
/// destination; the three vocabularies are discrete authored laws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatternSynthParams {
    /// Shape stage selector. Discrete law.
    pub shape: PatternShape,
    /// Oscillator waveform. Discrete law.
    pub wave: PatternWave,
    /// Colouriser. Discrete law.
    pub color_mode: PatternColorMode,
    /// Horizontal frequency; the shader squares it into 0.2..40.2 cycles.
    pub freq_x: f32,
    /// Vertical frequency, same law.
    pub freq_y: f32,
    /// Static phase offset into the oscillator.
    pub phase: f32,
    /// Clock rate: the pattern's time is `time * rate`, BENDR's own law, so
    /// a modulated rate rescales phase rather than accumulating.
    pub rate: f32,
    /// Cross-modulation depth between the two axes' oscillators.
    pub cross_mod: f32,
    /// Wavefolder depth.
    pub wavefold: f32,
    /// Pulse waveform duty cycle.
    pub pulse_width: f32,
    /// Comparator mix.
    pub comparator: f32,
    /// Comparator threshold.
    pub comp_threshold: f32,
    /// Comparator softness.
    pub comp_soft: f32,
    /// Fold count for the angular shapes; floored in the signal path.
    pub symmetry: f32,
    /// Zoom, `2^(zoom*2)`.
    pub zoom: f32,
    /// Rotation in half-turns (±1 is ±180°).
    pub rotate: f32,
    /// Axis shear.
    pub skew: f32,
    /// Centre offset, half-frame units.
    pub center_x: f32,
    pub center_y: f32,
    /// Domain warp: the coordinate system itself breathes.
    pub warp: f32,
    /// Base hue in unit turns.
    pub hue: f32,
    /// Hue spread across the colouriser.
    pub hue_spread: f32,
    /// Colouriser saturation.
    pub saturation: f32,
    /// Output brightness.
    pub brightness: f32,
    /// Band count for the `Bands` colouriser; floored in the signal path.
    pub color_bands: f32,
}

impl Default for PatternSynthParams {
    fn default() -> Self {
        Self {
            shape: PatternShape::Scan,
            wave: PatternWave::Sine,
            color_mode: PatternColorMode::RgbPhase,
            freq_x: 0.18,
            freq_y: 0.12,
            phase: 0.0,
            rate: 0.08,
            cross_mod: 0.0,
            wavefold: 0.0,
            pulse_width: 0.5,
            comparator: 0.0,
            comp_threshold: 0.5,
            comp_soft: 0.12,
            symmetry: 4.0,
            zoom: 0.0,
            rotate: 0.0,
            skew: 0.0,
            center_x: 0.0,
            center_y: 0.0,
            warp: 0.0,
            hue: 0.55,
            hue_spread: 1.0,
            saturation: 0.9,
            brightness: 1.0,
            color_bands: 6.0,
        }
    }
}

impl PatternSynthParams {
    /// Clamp every authored value into its declared range. Hostile non-finite
    /// input takes the neutral default rather than a clamped extreme.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            shape: self.shape,
            wave: self.wave,
            color_mode: self.color_mode,
            freq_x: finite_clamp(self.freq_x, defaults.freq_x, 0.0, 1.0),
            freq_y: finite_clamp(self.freq_y, defaults.freq_y, 0.0, 1.0),
            phase: finite_clamp(self.phase, defaults.phase, -1.0, 1.0),
            rate: finite_clamp(self.rate, defaults.rate, -1.0, 1.0),
            cross_mod: finite_clamp(self.cross_mod, defaults.cross_mod, 0.0, 1.0),
            wavefold: finite_clamp(self.wavefold, defaults.wavefold, 0.0, 1.0),
            pulse_width: finite_clamp(self.pulse_width, defaults.pulse_width, 0.0, 1.0),
            comparator: finite_clamp(self.comparator, defaults.comparator, 0.0, 1.0),
            comp_threshold: finite_clamp(self.comp_threshold, defaults.comp_threshold, 0.0, 1.0),
            comp_soft: finite_clamp(self.comp_soft, defaults.comp_soft, 0.0, 1.0),
            symmetry: finite_clamp(self.symmetry, defaults.symmetry, 1.0, 16.0),
            zoom: finite_clamp(self.zoom, defaults.zoom, -1.0, 1.0),
            rotate: finite_clamp(self.rotate, defaults.rotate, -1.0, 1.0),
            skew: finite_clamp(self.skew, defaults.skew, -1.0, 1.0),
            center_x: finite_clamp(self.center_x, defaults.center_x, -1.0, 1.0),
            center_y: finite_clamp(self.center_y, defaults.center_y, -1.0, 1.0),
            warp: finite_clamp(self.warp, defaults.warp, 0.0, 1.0),
            hue: finite_clamp(self.hue, defaults.hue, 0.0, 1.0),
            hue_spread: finite_clamp(self.hue_spread, defaults.hue_spread, 0.0, 2.0),
            saturation: finite_clamp(self.saturation, defaults.saturation, 0.0, 1.0),
            brightness: finite_clamp(self.brightness, defaults.brightness, 0.0, 1.5),
            color_bands: finite_clamp(self.color_bands, defaults.color_bands, 2.0, 16.0),
        }
    }
}

/// GLSL `fract`: always in `[0, 1)`, unlike Rust's sign-preserving `fract`.
#[inline]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
fn glsl_fract(x: f32) -> f32 {
    x - x.floor()
}

#[inline]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
fn glsl_fract2(p: [f32; 2]) -> [f32; 2] {
    [glsl_fract(p[0]), glsl_fract(p[1])]
}

#[inline]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// BENDR's `h21` screen hash, kept expression for expression.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn pattern_hash21(p: [f32; 2]) -> f32 {
    let mut q = glsl_fract2([p[0] * 123.34, p[1] * 456.21]);
    let d = q[0] * (q[0] + 45.32) + q[1] * (q[1] + 45.32);
    q = [q[0] + d, q[1] + d];
    glsl_fract(q[0] * q[1])
}

/// BENDR's `vn` value noise: smooth-interpolated `h21` at cell corners.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn pattern_value_noise(p: [f32; 2]) -> f32 {
    let i = [p[0].floor(), p[1].floor()];
    let mut f = glsl_fract2(p);
    f = [
        f[0] * f[0] * (3.0 - 2.0 * f[0]),
        f[1] * f[1] * (3.0 - 2.0 * f[1]),
    ];
    let h00 = pattern_hash21(i);
    let h10 = pattern_hash21([i[0] + 1.0, i[1]]);
    let h01 = pattern_hash21([i[0], i[1] + 1.0]);
    let h11 = pattern_hash21([i[0] + 1.0, i[1] + 1.0]);
    mix(mix(h00, h10, f[0]), mix(h01, h11, f[0]), f[1])
}

/// BENDR's compact HSV-to-RGB, kept expression for expression.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn pattern_hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let k = [
        glsl_fract(h),
        glsl_fract(h + 2.0 / 3.0),
        glsl_fract(h + 1.0 / 3.0),
    ];
    let mut c = [0.0f32; 3];
    for i in 0..3 {
        let w = ((k[i] * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0);
        c[i] = v * mix(1.0, w, s);
    }
    c
}

/// The oscillator: one cycle of the selected waveform per unit of phase.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn pattern_waveform(wave: PatternWave, pulse_width: f32, x: f32) -> f32 {
    let x = glsl_fract(x);
    match wave {
        PatternWave::Sine => 0.5 + 0.5 * (x * PATTERN_TAU).sin(),
        PatternWave::Triangle => (x * 2.0 - 1.0).abs(),
        PatternWave::Saw => x,
        PatternWave::Square => {
            if x >= 0.5 {
                1.0
            } else {
                0.0
            }
        }
        PatternWave::Pulse => {
            if x >= 1.0 - pulse_width.clamp(0.02, 0.98) {
                1.0
            } else {
                0.0
            }
        }
        PatternWave::SampleHold => pattern_hash21([(x * 48.0).floor(), 7.31]),
    }
}

/// The fixed escape-time horizon for the additive Mandelbrot pattern.
///
/// A finite renderer can prove escape, but a point that survives this many
/// iterations is only "bounded through the limit", never claimed as proven
/// set membership. The WGSL carries the same literal.
pub const MANDELBROT_MAX_ITERATIONS: u32 = 256;

/// Map the pattern page onto a fixed, aspect-correct complex plane. The page
/// centre is `-0.5 + 0i`, its imaginary half-span is one, and its real
/// half-span is `aspect`, so one source pixel has the same complex-plane scale
/// on both axes. `uv` follows the tree's top-left-origin convention.
fn mandelbrot_coordinate(uv: [f32; 2], aspect: f32) -> [f32; 2] {
    [-0.5 + (uv[0] * 2.0 - 1.0) * aspect, 1.0 - uv[1] * 2.0]
}

/// Return the first one-based iteration whose orbit exceeds the canonical
/// radius-two bailout, or `None` when the orbit remains bounded through the
/// fixed horizon. This is the literal Mandelbrot law:
///
/// `z_0 = 0`, `z_(n+1) = z_n^2 + c`, escape when `|z_n|^2 > 4`.
///
/// The strict comparison matters: `c = -2` lives on the boundary and must not
/// be rejected merely because its orbit reaches magnitude exactly two.
pub fn mandelbrot_escape_count(c: [f32; 2]) -> Option<u32> {
    let mut z = [0.0f32, 0.0f32];
    for iteration in 1..=MANDELBROT_MAX_ITERATIONS {
        let re_squared = z[0] * z[0];
        let im_squared = z[1] * z[1];
        let next = [re_squared - im_squared + c[0], 2.0 * z[0] * z[1] + c[1]];
        z = next;
        if z[0] * z[0] + z[1] * z[1] > 4.0 {
            return Some(iteration);
        }
    }
    None
}

/// Convert the exact integer escape time into the pattern synth's scalar
/// colour input. Faster escape is brighter; the last admitted escape remains
/// nonzero so it cannot be confused with a bounded-through-limit point.
fn mandelbrot_escape_signal(uv: [f32; 2], aspect: f32) -> (f32, bool) {
    match mandelbrot_escape_count(mandelbrot_coordinate(uv, aspect)) {
        Some(iteration) => (
            (MANDELBROT_MAX_ITERATIONS + 1 - iteration) as f32 / MANDELBROT_MAX_ITERATIONS as f32,
            false,
        ),
        None => (0.0, true),
    }
}

/// Number of analytically reconstructed positions in the Memory Splats
/// trail. Keeping this fixed makes the source's GPU cost bounded and its CPU
/// reference byte-for-byte reproducible.
const MEMORY_SPLAT_SAMPLES: usize = 4;

/// The deterministic orbit of one splat inside its lattice cell. Hashes are
/// computed once per cell and reused by every Memory sample — important in the
/// full-resolution hot loop. The point remains in the cell's 0.18..0.82 box.
fn moving_splat_point(offset: [f32; 2], h0: f32, h1: f32, t: f32, phase_radians: f32) -> [f32; 2] {
    [
        offset[0] + 0.5 + 0.32 * (h0 * PATTERN_TAU + phase_radians + t * (0.55 + 0.85 * h1)).sin(),
        offset[1] + 0.5 + 0.32 * (h1 * PATTERN_TAU - phase_radians + t * (0.50 + 0.75 * h0)).cos(),
    ]
}

/// A compact splat field with visible motion memory. The head and tail are
/// exact points on the deterministic orbit; two intermediate trail positions
/// are reconstructed along that recent motion segment. No framebuffer state
/// is held, so seek, pause, replay, and offline export name the same image.
fn memory_splats_signal(p: [f32; 2], fx: f32, freq_y: f32, phase: f32, t: f32, memory: f32) -> f32 {
    let cell_scale = 2.0 + fx * 0.22;
    let g = [p[0] * cell_scale, p[1] * cell_scale];
    let gi = [g[0].floor(), g[1].floor()];
    let gf = glsl_fract2(g);
    let radius = 0.18 + freq_y * 0.32;
    let lag = 0.10 + memory * 0.65;
    let decay = 0.28 + memory * 0.55;
    let trail_weights = [1.0, decay, decay * decay, decay * decay * decay];
    let phase_radians = phase * PATTERN_TAU;
    // A splat cannot leave its cell's central 64%, and the compact radius is
    // at most 0.5. The nearest 2x2 cells are therefore complete; every omitted
    // cell starts at least 0.68 away. This cuts the dominant per-pixel loop
    // from 36 kernel samples to 16 without changing the field.
    let start_x = if gf[0] < 0.5 { -1 } else { 0 };
    let start_y = if gf[1] < 0.5 { -1 } else { 0 };
    let mut alpha = 0.0f32;
    for yi in 0..2 {
        for xi in 0..2 {
            let x = start_x + xi;
            let y = start_y + yi;
            let offset = [x as f32, y as f32];
            let cell = [gi[0] + offset[0], gi[1] + offset[1]];
            let h0 = pattern_hash21(cell);
            let h1 = pattern_hash21([cell[0] + 5.17, cell[1] + 9.31]);
            // Two exact orbit evaluations define all four trail positions.
            // The naïve formulation evaluated sin/cos for every age; this
            // interpolation halves the dominant SFU work per Memory layer.
            let head = moving_splat_point(offset, h0, h1, t, phase_radians);
            let tail = moving_splat_point(
                offset,
                h0,
                h1,
                t - (MEMORY_SPLAT_SAMPLES - 1) as f32 * lag,
                phase_radians,
            );
            for (age, trail_weight) in trail_weights.iter().copied().enumerate() {
                let trail_position = age as f32 / (MEMORY_SPLAT_SAMPLES - 1) as f32;
                let point = if age == 0 {
                    head
                } else if age + 1 == MEMORY_SPLAT_SAMPLES {
                    tail
                } else {
                    [
                        mix(head[0], tail[0], trail_position),
                        mix(head[1], tail[1], trail_position),
                    ]
                };
                let delta = [point[0] - gf[0], point[1] - gf[1]];
                let distance_squared = delta[0] * delta[0] + delta[1] * delta[1];
                let inner_radius = radius * 0.35;
                let compact = 1.0
                    - smoothstep(
                        inner_radius * inner_radius,
                        radius * radius,
                        distance_squared,
                    );
                let sample_alpha = (compact * trail_weight).clamp(0.0, 1.0);
                // Front-to-back alpha union: bounded even when many trails
                // overlap, unlike an unconstrained additive sum.
                alpha += (1.0 - alpha) * sample_alpha;
            }
        }
    }
    alpha.clamp(0.0, 1.0)
}

/// A moving anisotropic Gaussian-splat field. `anisotropy` stretches each
/// kernel along its own deterministic hashed major axis while its centre moves
/// on the shared orbit. A compact support window reaches zero before any cell
/// omitted by the nearest-2x2 search, so the bounded neighbourhood cannot
/// introduce lattice seams.
fn gaussian_splats_signal(
    p: [f32; 2],
    fx: f32,
    freq_y: f32,
    phase: f32,
    t: f32,
    anisotropy: f32,
) -> f32 {
    let cell_scale = 2.0 + fx * 0.22;
    let g = [p[0] * cell_scale, p[1] * cell_scale];
    let gi = [g[0].floor(), g[1].floor()];
    let gf = glsl_fract2(g);
    let sigma_major = 0.16 + freq_y * 0.24;
    let axis_ratio = 1.0 + anisotropy * 3.0;
    let sigma_minor = sigma_major / axis_ratio;
    let phase_radians = phase * PATTERN_TAU;
    let start_x = if gf[0] < 0.5 { -1 } else { 0 };
    let start_y = if gf[1] < 0.5 { -1 } else { 0 };
    let mut alpha = 0.0f32;
    for yi in 0..2 {
        for xi in 0..2 {
            let x = start_x + xi;
            let y = start_y + yi;
            let offset = [x as f32, y as f32];
            let cell = [gi[0] + offset[0], gi[1] + offset[1]];
            let h0 = pattern_hash21(cell);
            let h1 = pattern_hash21([cell[0] + 5.17, cell[1] + 9.31]);
            let point = moving_splat_point(offset, h0, h1, t, phase_radians);
            let delta = [gf[0] - point[0], gf[1] - point[1]];
            let h2 = pattern_hash21([cell[0] + 11.7, cell[1] + 2.93]);
            let h3 = pattern_hash21([cell[0] + 19.19, cell[1] + 7.73]);
            let axis = [h2 * 2.0 - 1.0, h3 * 2.0 - 1.0];
            let axis_length_squared = axis[0] * axis[0] + axis[1] * axis[1];
            let (cos_angle, sin_angle) = if axis_length_squared > 1.0e-6 {
                let inverse_axis_length = 1.0 / axis_length_squared.sqrt();
                (axis[0] * inverse_axis_length, axis[1] * inverse_axis_length)
            } else {
                // A hash can land arbitrarily close to the origin. Preserve a
                // unit orthonormal basis instead of flattening that rare splat.
                (1.0, 0.0)
            };
            let local = [
                delta[0] * cos_angle + delta[1] * sin_angle,
                -delta[0] * sin_angle + delta[1] * cos_angle,
            ];
            let quadratic = local[0] * local[0] / (sigma_major * sigma_major)
                + local[1] * local[1] / (sigma_minor * sigma_minor);
            let distance_squared = delta[0] * delta[0] + delta[1] * delta[1];
            // Gaussian rasterizers conventionally bound their footprint. The
            // 0.66 cutoff is below the 0.68 distance to every omitted cell.
            let support = 1.0 - smoothstep(0.56 * 0.56, 0.66 * 0.66, distance_squared);
            let sample_alpha = ((-0.5 * quadratic).exp() * support).clamp(0.0, 1.0);
            alpha += (1.0 - alpha) * sample_alpha;
        }
    }
    alpha.clamp(0.0, 1.0)
}

/// The whole signal path for one pixel: framing, shape, oscillator,
/// wavefolder, comparator, colouriser. `uv` is the tree's top-left-origin
/// texture coordinate; the reference flips it into BENDR's bottom-up frame so
/// the transcription stays literal. `time` is the layer's speed-scaled
/// program time; the `rate` law (`t = time * rate`) is BENDR's own and lives
/// here rather than in the caller. The returned RGB is the *stored* picture —
/// display-domain values, exactly the bytes a decoded video frame would hold.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn pattern_synth_pixel(
    params: &PatternSynthParams,
    uv: [f32; 2],
    aspect: f32,
    time: f32,
) -> [f32; 3] {
    let q = params.sanitized();
    let t = time * q.rate;
    // Framing: centre, aspect, zoom, rotate, skew, warp.
    let mut p = [
        uv[0] - 0.5 - q.center_x * 0.5,
        (1.0 - uv[1]) - 0.5 - q.center_y * 0.5,
    ];
    p[0] *= aspect;
    let zm = (q.zoom * 2.0).exp2();
    p = [p[0] * zm, p[1] * zm];
    #[allow(
        clippy::approx_constant,
        reason = "BENDR's literal, mirrored byte for byte in pattern_synth.wgsl"
    )]
    let a0 = q.rotate * 3.14159;
    let (s0, c0) = a0.sin_cos();
    p = [p[0] * c0 - p[1] * s0, p[0] * s0 + p[1] * c0];
    p[0] += p[1] * q.skew * 2.0;
    if q.warp > 0.003 {
        let w = [
            pattern_value_noise([p[0] * 3.0 + t * 0.2, p[1] * 3.0 + t * 0.2]),
            pattern_value_noise([p[0] * 3.0 + 17.3 - t * 0.15, p[1] * 3.0 + 17.3 - t * 0.15]),
        ];
        p[0] += (w[0] - 0.5) * q.warp * 1.2;
        p[1] += (w[1] - 0.5) * q.warp * 1.2;
    }
    let fx = 0.2 + q.freq_x * q.freq_x * 40.0;
    let fy = 0.2 + q.freq_y * q.freq_y * 40.0;
    let ph = q.phase;
    let nf = q.symmetry.floor().max(1.0);
    // Polar coordinates contain a square root and atan2. Only the named polar
    // shapes pay for them; Cartesian, Mandelbrot, and splat layers skip that
    // full-resolution serial work entirely.
    let r = if matches!(
        q.shape,
        PatternShape::Radial
            | PatternShape::Spiral
            | PatternShape::Plasma
            | PatternShape::Rings
            | PatternShape::Starburst
            | PatternShape::Tunnel
            | PatternShape::Polygon
    ) {
        (p[0] * p[0] + p[1] * p[1]).sqrt()
    } else {
        0.0
    };
    let ang = if matches!(
        q.shape,
        PatternShape::Radial
            | PatternShape::Spiral
            | PatternShape::Starburst
            | PatternShape::Tunnel
    ) {
        p[1].atan2(p[0]) / PATTERN_TAU + 0.5
    } else {
        0.0
    };
    let wv = |x: f32| pattern_waveform(q.wave, q.pulse_width, x);
    // The shape stage.
    let mut mandelbrot_interior = false;
    let mut f = match q.shape {
        PatternShape::Scan => {
            let b = wv(p[1] * fy + t * 0.7);
            let a = wv(p[0] * fx + ph + t + b * q.cross_mod * 3.0);
            0.5 * (a + b)
        }
        PatternShape::Radial => wv(r * fx + ph + t + wv(ang * nf) * q.cross_mod * 2.0),
        PatternShape::Spiral => wv(r * fx + ang * nf + ph + t),
        PatternShape::Plasma => {
            let f0 = 0.25
                * (wv(p[0] * fx * 0.5 + t)
                    + wv(p[1] * fy * 0.5 - t * 0.8)
                    + wv((p[0] + p[1]) * fx * 0.35 + t * 1.3)
                    + wv(r * fy * 0.5 - t * 0.6));
            glsl_fract(f0 * (1.0 + q.cross_mod * 3.0))
        }
        PatternShape::Lissajous => {
            let lx = (p[0] * fx + t).sin();
            let ly = (p[1] * fy + t * 1.37 + ph * PATTERN_TAU).sin();
            wv(lx * ly * (0.5 + q.cross_mod * 3.0) + ph)
        }
        PatternShape::Rings => wv((r * fx * 0.5 + t).floor() / nf.max(1.0) + ph),
        PatternShape::Starburst => wv(ang * nf + ph + t + r * fx * 0.06 * q.cross_mod * 10.0),
        PatternShape::Grid => wv(p[0] * fx + ph + t).max(wv(p[1] * fy - t)),
        PatternShape::Tunnel => {
            let rr = 0.35 / r.max(0.02);
            0.5 * (wv(rr * fx * 0.25 + t) + wv(ang * nf + ph))
        }
        PatternShape::Cells => {
            let cell_scale = (fx * 0.25).max(1.0);
            let g = [p[0] * cell_scale, p[1] * cell_scale];
            let gi = [g[0].floor(), g[1].floor()];
            let gf = glsl_fract2(g);
            let mut md = 8.0f32;
            for y in -1..=1 {
                for x in -1..=1 {
                    let o = [x as f32, y as f32];
                    let h0 = pattern_hash21([gi[0] + o[0], gi[1] + o[1]]);
                    let h1 = pattern_hash21([gi[0] + o[0] + 3.1, gi[1] + o[1] + 3.1]);
                    let pt = [
                        o[0] + 0.5 + 0.5 * (h0 * PATTERN_TAU + t * 2.0).sin(),
                        o[1] + 0.5 + 0.5 * (h1 * PATTERN_TAU + t * 1.7).cos(),
                    ];
                    let d = ((pt[0] - gf[0]).powi(2) + (pt[1] - gf[1]).powi(2)).sqrt();
                    md = md.min(d);
                }
            }
            wv(md * (0.5 + q.cross_mod * 3.0) + ph)
        }
        PatternShape::Interference => {
            let d1 = ((p[0] - 0.28).powi(2) + p[1] * p[1]).sqrt();
            let d2 = ((p[0] + 0.28).powi(2) + p[1] * p[1]).sqrt();
            0.5 * (wv(d1 * fx + t) + wv(d2 * fy - t))
        }
        PatternShape::Polygon => {
            let aa = p[1].atan2(p[0]);
            let seg = PATTERN_TAU / nf;
            let rp = r * (aa.rem_euclid(seg) - seg * 0.5).cos() / (seg * 0.5).cos().max(0.01);
            wv(rp * fx * 0.5 + ph + t)
        }
        PatternShape::Mandelbrot => {
            // This branch deliberately reads the raw page coordinate rather
            // than BENDR's framing domain above. Every pixel therefore names
            // one canonical `c`; the ordinary signal and colour stages may
            // dress the escaped scalar later but never change the orbit.
            let (signal, interior) = mandelbrot_escape_signal(uv, aspect);
            mandelbrot_interior = interior;
            signal
        }
        PatternShape::MemorySplats => memory_splats_signal(p, fx, q.freq_y, ph, t, q.cross_mod),
        PatternShape::GaussianSplats => gaussian_splats_signal(p, fx, q.freq_y, ph, t, q.cross_mod),
    };
    // Wavefolder: keeps folding the signal back on itself, which is where the
    // hard banded video-synth structure comes from.
    if q.wavefold > 0.003 {
        let k = 1.0 + q.wavefold * 7.0;
        f = (glsl_fract(f * k) * 2.0 - 1.0).abs();
    }
    // Comparator: the hard-edged shape maker.
    if q.comparator > 0.003 {
        let sf = (q.comp_soft * 0.5).max(0.001);
        f = mix(
            f,
            smoothstep(q.comp_threshold - sf, q.comp_threshold + sf, f),
            q.comparator,
        );
    }
    let f = f.clamp(0.0, 1.0);
    // The colouriser.
    let sp = q.hue_spread;
    let c = if mandelbrot_interior {
        // Preserve the visible set under every pattern colouriser. Effects
        // downstream may still transform it, as they do for any source.
        [0.0; 3]
    } else {
        match q.color_mode {
            PatternColorMode::Mono => [f, f, f],
            PatternColorMode::RgbPhase => [
                wv(f + q.hue),
                wv(f + q.hue + sp * 0.33),
                wv(f + q.hue + sp * 0.66),
            ],
            PatternColorMode::HsvSweep => {
                pattern_hsv(q.hue + f * sp, q.saturation, mix(1.0, f, 0.25))
            }
            PatternColorMode::Duotone => {
                let a = pattern_hsv(q.hue, q.saturation, 1.0);
                let b = pattern_hsv(glsl_fract(q.hue + sp * 0.5), q.saturation, 1.0);
                [mix(a[0], b[0], f), mix(a[1], b[1], f), mix(a[2], b[2], f)]
            }
            PatternColorMode::Bands => {
                let nb = q.color_bands.floor().max(2.0);
                let qf = (f * nb).floor() / nb;
                pattern_hsv(q.hue + qf * sp, q.saturation, 1.0)
            }
        }
    };
    [
        (c[0] * q.brightness).clamp(0.0, 1.0),
        (c[1] * q.brightness).clamp(0.0, 1.0),
        (c[2] * q.brightness).clamp(0.0, 1.0),
    ]
}

/// The pattern pass uniform: eight vec4 lanes, compile-time asserted, shared
/// verbatim between the live and export encoders.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PatternSynthGpuUniforms {
    /// freq_x, freq_y, phase, rate
    pub freq_phase: [f32; 4],
    /// cross_mod, wavefold, pulse_width, comparator
    pub signal: [f32; 4],
    /// comp_threshold, comp_soft, symmetry, zoom
    pub compare_frame: [f32; 4],
    /// rotate, skew, center_x, center_y
    pub placement: [f32; 4],
    /// warp, hue, hue_spread, saturation
    pub color_a: [f32; 4],
    /// brightness, color_bands, time, aspect
    pub color_b: [f32; 4],
    /// shape code, wave code, colour-mode code, reserved
    pub modes: [u32; 4],
    /// Reserved tail so a later lane never moves the stride.
    pub reserved: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<PatternSynthGpuUniforms>() == 128);

impl PatternSynthGpuUniforms {
    /// Build the uniform from sanitized authored state and the layer's
    /// speed-scaled program time.
    pub fn from_params(params: &PatternSynthParams, time: f32) -> Self {
        let q = params.sanitized();
        let aspect = PATTERN_SYNTH_WIDTH as f32 / PATTERN_SYNTH_HEIGHT as f32;
        Self {
            freq_phase: [q.freq_x, q.freq_y, q.phase, q.rate],
            signal: [q.cross_mod, q.wavefold, q.pulse_width, q.comparator],
            compare_frame: [q.comp_threshold, q.comp_soft, q.symmetry, q.zoom],
            placement: [q.rotate, q.skew, q.center_x, q.center_y],
            color_a: [q.warp, q.hue, q.hue_spread, q.saturation],
            color_b: [
                q.brightness,
                q.color_bands,
                if time.is_finite() { time } else { 0.0 },
                aspect,
            ],
            modes: [
                q.shape.gpu_code(),
                q.wave.gpu_code(),
                q.color_mode.gpu_code(),
                0,
            ],
            reserved: [0.0; 4],
        }
    }
}

fn finite_clamp(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

impl PatternShape {
    /// Wire/patch-adjacent token. Kept beside the codes so no stringify site
    /// can disagree.
    pub const fn key(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::Radial => "radial",
            Self::Spiral => "spiral",
            Self::Plasma => "plasma",
            Self::Lissajous => "lissajous",
            Self::Rings => "rings",
            Self::Starburst => "starburst",
            Self::Grid => "grid",
            Self::Tunnel => "tunnel",
            Self::Cells => "cells",
            Self::Interference => "interference",
            Self::Polygon => "polygon",
            Self::Mandelbrot => "mandelbrot",
            Self::MemorySplats => "memory_splats",
            Self::GaussianSplats => "gaussian_splats",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|shape| shape.key() == key)
    }
}

impl PatternWave {
    pub const fn key(self) -> &'static str {
        match self {
            Self::Sine => "sine",
            Self::Triangle => "triangle",
            Self::Saw => "saw",
            Self::Square => "square",
            Self::Pulse => "pulse",
            Self::SampleHold => "sample_hold",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|wave| wave.key() == key)
    }
}

impl PatternColorMode {
    pub const fn key(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::RgbPhase => "rgb_phase",
            Self::HsvSweep => "hsv_sweep",
            Self::Duotone => "duotone",
            Self::Bands => "bands",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.key() == key)
    }
}

/// One validated wire edit to a pattern layer. The single parse table serves
/// both server-gate validation and the engine applier — the B8
/// `BusMixerEdit` law, so the accepted and applied vocabularies are
/// structurally one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternSynthEdit {
    Shape(PatternShape),
    Wave(PatternWave),
    ColorMode(PatternColorMode),
    Scalar(&'static str, f32),
}

impl PatternSynthEdit {
    /// Parse and range-validate one wire edit. Unknown params, out-of-range
    /// or non-finite numbers, and unknown tokens are rejections, never
    /// silently repaired values.
    pub fn parse(param: &str, value: &serde_json::Value) -> Option<Self> {
        let number = |min: f32, max: f32| -> Option<f32> {
            let number = value.as_f64()? as f32;
            (number.is_finite() && (min..=max).contains(&number)).then_some(number)
        };
        Some(match param {
            "shape" => Self::Shape(PatternShape::from_key(value.as_str()?)?),
            "wave" => Self::Wave(PatternWave::from_key(value.as_str()?)?),
            "color_mode" => Self::ColorMode(PatternColorMode::from_key(value.as_str()?)?),
            "freq_x" => Self::Scalar("freq_x", number(0.0, 1.0)?),
            "freq_y" => Self::Scalar("freq_y", number(0.0, 1.0)?),
            "phase" => Self::Scalar("phase", number(-1.0, 1.0)?),
            "rate" => Self::Scalar("rate", number(-1.0, 1.0)?),
            "cross_mod" => Self::Scalar("cross_mod", number(0.0, 1.0)?),
            "wavefold" => Self::Scalar("wavefold", number(0.0, 1.0)?),
            "pulse_width" => Self::Scalar("pulse_width", number(0.0, 1.0)?),
            "comparator" => Self::Scalar("comparator", number(0.0, 1.0)?),
            "comp_threshold" => Self::Scalar("comp_threshold", number(0.0, 1.0)?),
            "comp_soft" => Self::Scalar("comp_soft", number(0.0, 1.0)?),
            "symmetry" => Self::Scalar("symmetry", number(1.0, 16.0)?),
            "zoom" => Self::Scalar("zoom", number(-1.0, 1.0)?),
            "rotate" => Self::Scalar("rotate", number(-1.0, 1.0)?),
            "skew" => Self::Scalar("skew", number(-1.0, 1.0)?),
            "center_x" => Self::Scalar("center_x", number(-1.0, 1.0)?),
            "center_y" => Self::Scalar("center_y", number(-1.0, 1.0)?),
            "warp" => Self::Scalar("warp", number(0.0, 1.0)?),
            "hue" => Self::Scalar("hue", number(0.0, 1.0)?),
            "hue_spread" => Self::Scalar("hue_spread", number(0.0, 2.0)?),
            "saturation" => Self::Scalar("saturation", number(0.0, 1.0)?),
            "brightness" => Self::Scalar("brightness", number(0.0, 1.5)?),
            "color_bands" => Self::Scalar("color_bands", number(2.0, 16.0)?),
            _ => return None,
        })
    }

    /// Apply this validated edit to authored state, then sanitize once.
    pub fn apply(self, params: &mut PatternSynthParams) {
        match self {
            Self::Shape(shape) => params.shape = shape,
            Self::Wave(wave) => params.wave = wave,
            Self::ColorMode(mode) => params.color_mode = mode,
            Self::Scalar(field, value) => match field {
                "freq_x" => params.freq_x = value,
                "freq_y" => params.freq_y = value,
                "phase" => params.phase = value,
                "rate" => params.rate = value,
                "cross_mod" => params.cross_mod = value,
                "wavefold" => params.wavefold = value,
                "pulse_width" => params.pulse_width = value,
                "comparator" => params.comparator = value,
                "comp_threshold" => params.comp_threshold = value,
                "comp_soft" => params.comp_soft = value,
                "symmetry" => params.symmetry = value,
                "zoom" => params.zoom = value,
                "rotate" => params.rotate = value,
                "skew" => params.skew = value,
                "center_x" => params.center_x = value,
                "center_y" => params.center_y = value,
                "warp" => params.warp = value,
                "hue" => params.hue = value,
                "hue_spread" => params.hue_spread = value,
                "saturation" => params.saturation = value,
                "brightness" => params.brightness = value,
                "color_bands" => params.color_bands = value,
                _ => unreachable!("parse admits only the closed vocabulary"),
            },
        }
        *params = params.sanitized();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASPECT: f32 = PATTERN_SYNTH_WIDTH as f32 / PATTERN_SYNTH_HEIGHT as f32;

    #[test]
    fn defaults_survive_sanitize_unchanged() {
        let d = PatternSynthParams::default();
        assert_eq!(d.sanitized(), d);
        assert_eq!(d.shape, PatternShape::Scan);
        assert_eq!(d.wave, PatternWave::Sine);
        assert_eq!(d.color_mode, PatternColorMode::RgbPhase);
    }

    #[test]
    fn hostile_scalars_take_the_neutral_default_not_a_clamped_extreme() {
        let p = PatternSynthParams {
            freq_x: f32::NAN,
            rate: f32::INFINITY,
            hue_spread: f32::NEG_INFINITY,
            brightness: f32::NAN,
            ..PatternSynthParams::default()
        };
        let q = p.sanitized();
        let d = PatternSynthParams::default();
        assert_eq!(q.freq_x, d.freq_x);
        assert_eq!(q.rate, d.rate);
        assert_eq!(q.hue_spread, d.hue_spread);
        assert_eq!(q.brightness, d.brightness);
        // Finite out-of-range still clamps.
        let p = PatternSynthParams {
            symmetry: 99.0,
            color_bands: -5.0,
            ..PatternSynthParams::default()
        };
        let q = p.sanitized();
        assert_eq!(q.symmetry, 16.0);
        assert_eq!(q.color_bands, 2.0);
    }

    #[test]
    fn shape_wave_color_codes_keep_the_frozen_prefix_and_append_new_generators() {
        let shape_codes: Vec<u32> = PatternShape::ALL.iter().map(|s| s.gpu_code()).collect();
        assert_eq!(shape_codes, (0..15).collect::<Vec<u32>>());
        assert_eq!(PatternShape::Mandelbrot.gpu_code(), 12);
        assert_eq!(PatternShape::MemorySplats.gpu_code(), 13);
        assert_eq!(PatternShape::GaussianSplats.gpu_code(), 14);
        assert_eq!(
            PatternShape::from_key("mandelbrot"),
            Some(PatternShape::Mandelbrot)
        );
        assert_eq!(
            PatternShape::from_key("memory_splats"),
            Some(PatternShape::MemorySplats)
        );
        assert_eq!(
            PatternShape::from_key("gaussian_splats"),
            Some(PatternShape::GaussianSplats)
        );
        let wave_codes: Vec<u32> = PatternWave::ALL.iter().map(|w| w.gpu_code()).collect();
        assert_eq!(wave_codes, (0..6).collect::<Vec<u32>>());
        let color_codes: Vec<u32> = PatternColorMode::ALL.iter().map(|c| c.gpu_code()).collect();
        assert_eq!(color_codes, (0..5).collect::<Vec<u32>>());
    }

    #[test]
    fn mandelbrot_escape_counts_pin_the_exact_z_squared_plus_c_law() {
        // Interior and boundary landmarks survive the complete finite horizon.
        assert_eq!(mandelbrot_escape_count([0.0, 0.0]), None);
        assert_eq!(mandelbrot_escape_count([-1.0, 0.0]), None);
        assert_eq!(mandelbrot_escape_count([-2.0, 0.0]), None);

        // Strict `> 4`, checked after each recurrence: c=1 reaches z=2 on
        // iteration two but escapes only at z=5 on iteration three. c=2
        // similarly reaches the radius-two boundary first and escapes next.
        assert_eq!(mandelbrot_escape_count([1.0, 0.0]), Some(3));
        assert_eq!(mandelbrot_escape_count([2.0, 0.0]), Some(2));
    }

    #[test]
    fn mandelbrot_fixed_page_is_aspect_correct_and_conjugate_symmetric() {
        let centre = mandelbrot_coordinate([0.5, 0.5], ASPECT);
        assert_eq!(centre, [-0.5, 0.0]);

        // One source pixel must traverse the same complex-plane distance on
        // either axis; otherwise a round bulb would be stretched at 16:9.
        let right = mandelbrot_coordinate([0.5 + 1.0 / PATTERN_SYNTH_WIDTH as f32, 0.5], ASPECT);
        let up = mandelbrot_coordinate([0.5, 0.5 - 1.0 / PATTERN_SYNTH_HEIGHT as f32], ASPECT);
        assert!(((right[0] - centre[0]) - (up[1] - centre[1])).abs() < 1.0e-7);

        let upper = mandelbrot_coordinate([0.73, 0.31], ASPECT);
        let lower = mandelbrot_coordinate([0.73, 0.69], ASPECT);
        assert!((upper[0] - lower[0]).abs() < 1.0e-7);
        assert!((upper[1] + lower[1]).abs() < 1.0e-7);
        assert_eq!(
            mandelbrot_escape_count(upper),
            mandelbrot_escape_count(lower)
        );
    }

    #[test]
    fn mandelbrot_interior_stays_black_under_every_pattern_colouriser() {
        for color_mode in PatternColorMode::ALL {
            let params = PatternSynthParams {
                shape: PatternShape::Mandelbrot,
                color_mode,
                wavefold: 1.0,
                comparator: 1.0,
                comp_threshold: 0.0,
                brightness: 1.5,
                ..PatternSynthParams::default()
            };
            assert_eq!(
                pattern_synth_pixel(&params, [0.5, 0.5], ASPECT, 123.0),
                [0.0; 3],
                "{color_mode:?} must not colour a bounded-through-limit point"
            );
        }

        let exterior = PatternSynthParams {
            shape: PatternShape::Mandelbrot,
            color_mode: PatternColorMode::Mono,
            ..PatternSynthParams::default()
        };
        let pixel = pattern_synth_pixel(&exterior, [1.0, 0.5], ASPECT, 0.0);
        assert!(
            pixel[0] > 0.0,
            "an escaped point must carry its escape time"
        );
        assert_eq!(pixel[0], pixel[1]);
        assert_eq!(pixel[1], pixel[2]);
    }

    #[test]
    fn splat_fields_are_bounded_deterministic_moving_and_rate_freezable() {
        for shape in [PatternShape::MemorySplats, PatternShape::GaussianSplats] {
            let params = PatternSynthParams {
                shape,
                color_mode: PatternColorMode::Mono,
                freq_x: 0.42,
                freq_y: 0.58,
                rate: 0.73,
                cross_mod: 0.67,
                ..PatternSynthParams::default()
            };
            let mut lit = false;
            let mut moved = false;
            for y in 0..9 {
                for x in 0..16 {
                    let uv = [(x as f32 + 0.5) / 16.0, (y as f32 + 0.5) / 9.0];
                    let first = pattern_synth_pixel(&params, uv, ASPECT, 1.25);
                    let repeat = pattern_synth_pixel(&params, uv, ASPECT, 1.25);
                    let later = pattern_synth_pixel(&params, uv, ASPECT, 2.75);
                    assert_eq!(first, repeat, "{shape:?} must be deterministic");
                    for channel in first.into_iter().chain(later) {
                        assert!(
                            channel.is_finite() && (0.0..=1.0).contains(&channel),
                            "{shape:?} emitted {channel}"
                        );
                    }
                    lit |= first[0] > 0.01 || later[0] > 0.01;
                    moved |= (first[0] - later[0]).abs() > 1.0e-4;
                }
            }
            assert!(lit, "{shape:?} must produce visible splats");
            assert!(moved, "{shape:?} must move with program time");

            let frozen = PatternSynthParams {
                rate: 0.0,
                ..params
            };
            let uv = [0.371, 0.629];
            assert_eq!(
                pattern_synth_pixel(&frozen, uv, ASPECT, 0.0),
                pattern_synth_pixel(&frozen, uv, ASPECT, 999.0),
                "{shape:?} with zero rate must be time-invariant"
            );
        }
    }

    #[test]
    fn the_oscillator_matches_its_analytic_waveforms() {
        // Sine: peak at quarter phase, trough at three quarters.
        assert!((pattern_waveform(PatternWave::Sine, 0.5, 0.25) - 1.0).abs() < 1e-6);
        assert!(pattern_waveform(PatternWave::Sine, 0.5, 0.75).abs() < 1e-6);
        // Triangle: zero at half phase, one at the wrap.
        assert!((pattern_waveform(PatternWave::Triangle, 0.5, 0.5)).abs() < 1e-6);
        assert!((pattern_waveform(PatternWave::Triangle, 0.5, 0.0) - 1.0).abs() < 1e-6);
        // Saw is the wrapped phase itself, in [0, 1) even for negative input.
        assert!((pattern_waveform(PatternWave::Saw, 0.5, 1.3) - 0.3).abs() < 1e-6);
        assert!((pattern_waveform(PatternWave::Saw, 0.5, -0.25) - 0.75).abs() < 1e-6);
        // Square: half duty regardless of authored pulse width.
        assert_eq!(pattern_waveform(PatternWave::Square, 0.9, 0.49), 0.0);
        assert_eq!(pattern_waveform(PatternWave::Square, 0.9, 0.51), 1.0);
        // Pulse: duty is the authored width, clamped into 0.02..0.98.
        assert_eq!(pattern_waveform(PatternWave::Pulse, 0.25, 0.70), 0.0);
        assert_eq!(pattern_waveform(PatternWave::Pulse, 0.25, 0.80), 1.0);
        assert_eq!(pattern_waveform(PatternWave::Pulse, 2.0, 0.03), 1.0);
        // Sample-and-hold: constant within one of the 48 steps (0.101 and
        // 0.104 both floor to cell 4), deterministic across calls.
        let a = pattern_waveform(PatternWave::SampleHold, 0.5, 0.101);
        let b = pattern_waveform(PatternWave::SampleHold, 0.5, 0.104);
        assert_eq!(a, b);
        assert_eq!(a, pattern_waveform(PatternWave::SampleHold, 0.5, 0.101));
        // And steps differ across a cell boundary.
        let c = pattern_waveform(PatternWave::SampleHold, 0.5, 0.1195);
        assert_ne!(a, c);
    }

    #[test]
    fn radial_without_cross_mod_is_rotationally_symmetric() {
        let p = PatternSynthParams {
            shape: PatternShape::Radial,
            cross_mod: 0.0,
            color_mode: PatternColorMode::Mono,
            ..PatternSynthParams::default()
        };
        // Two points at the same radius from the (default, centred) origin.
        // The reference frame is aspect-scaled, so build the pair in that
        // frame and map back to uv.
        let r = 0.21f32;
        let uv_of = |theta: f32| {
            [
                0.5 + r * theta.cos() / ASPECT,
                1.0 - (0.5 + r * theta.sin()),
            ]
        };
        let a = pattern_synth_pixel(&p, uv_of(0.31), ASPECT, 2.5);
        let b = pattern_synth_pixel(&p, uv_of(2.9), ASPECT, 2.5);
        assert!((a[0] - b[0]).abs() < 1e-4, "{a:?} vs {b:?}");
    }

    #[test]
    fn scan_without_cross_mod_separates_into_its_two_ramps() {
        let p = PatternSynthParams {
            shape: PatternShape::Scan,
            cross_mod: 0.0,
            color_mode: PatternColorMode::Mono,
            ..PatternSynthParams::default()
        };
        let q = p.sanitized();
        let fx = 0.2 + q.freq_x * q.freq_x * 40.0;
        let fy = 0.2 + q.freq_y * q.freq_y * 40.0;
        let time = 1.75f32;
        let t = time * q.rate;
        let uv = [0.37f32, 0.62f32];
        let px = (uv[0] - 0.5) * ASPECT;
        let py = (1.0 - uv[1]) - 0.5;
        let expect = 0.5
            * (pattern_waveform(q.wave, q.pulse_width, px * fx + q.phase + t)
                + pattern_waveform(q.wave, q.pulse_width, py * fy + t * 0.7));
        let got = pattern_synth_pixel(&p, uv, ASPECT, time)[0];
        assert!((got - expect).abs() < 1e-6);
    }

    #[test]
    fn the_comparator_at_full_mix_and_zero_soft_is_a_hard_threshold() {
        let p = PatternSynthParams {
            shape: PatternShape::Scan,
            wave: PatternWave::Saw,
            comparator: 1.0,
            comp_soft: 0.0,
            comp_threshold: 0.5,
            color_mode: PatternColorMode::Mono,
            rate: 0.0,
            ..PatternSynthParams::default()
        };
        // Construct two points whose pre-comparator signal is known exactly:
        // with Scan + Saw + no cross-mod, f = (fract(px*fx) + fract(py*fy))/2.
        let q = p.sanitized();
        let fx = 0.2 + q.freq_x * q.freq_x * 40.0;
        let fy = 0.2 + q.freq_y * q.freq_y * 40.0;
        let uv_for = |a: f32, b: f32| {
            let px = a / fx;
            let py = b / fy;
            [0.5 + px / ASPECT, 0.5 - py]
        };
        // f = 0.2: well below threshold, the hard comparator floors it.
        let low = pattern_synth_pixel(&p, uv_for(0.2, 0.2), ASPECT, 0.0)[0];
        assert!(low < 0.02, "f=0.2 must comparator to black, got {low}");
        // f = 0.8: well above threshold, it saturates.
        let high = pattern_synth_pixel(&p, uv_for(0.9, 0.7), ASPECT, 0.0)[0];
        assert!(high > 0.98, "f=0.8 must comparator to white, got {high}");
    }

    #[test]
    fn the_wavefolder_keeps_the_signal_in_range_and_reaches_the_picture() {
        let folded = PatternSynthParams {
            shape: PatternShape::Radial,
            wavefold: 0.8,
            color_mode: PatternColorMode::Mono,
            ..PatternSynthParams::default()
        };
        let plain = PatternSynthParams {
            wavefold: 0.0,
            ..folded
        };
        let mut differs = false;
        for i in 0..32 {
            let uv = [0.5 + i as f32 / 100.0, 0.5];
            let f = pattern_synth_pixel(&folded, uv, ASPECT, 3.0);
            let g = pattern_synth_pixel(&plain, uv, ASPECT, 3.0);
            for channel in f {
                assert!((0.0..=1.0).contains(&channel));
            }
            if (f[0] - g[0]).abs() > 1e-3 {
                differs = true;
            }
        }
        assert!(differs, "an authored wavefold must change the picture");
    }

    #[test]
    fn colourisers_follow_their_analytic_laws() {
        let mut p = PatternSynthParams {
            shape: PatternShape::Scan,
            rate: 0.0,
            // Mono is grey.
            color_mode: PatternColorMode::Mono,
            ..PatternSynthParams::default()
        };
        let m = pattern_synth_pixel(&p, [0.3, 0.3], ASPECT, 0.0);
        assert_eq!(m[0], m[1]);
        assert_eq!(m[1], m[2]);
        // Bands quantizes hue to at most floor(bands) distinct values along a
        // sweep of f; with saturation zero every band is white.
        p.color_mode = PatternColorMode::Bands;
        p.saturation = 0.0;
        let b = pattern_synth_pixel(&p, [0.3, 0.3], ASPECT, 0.0);
        for channel in b {
            assert!((channel - 1.0).abs() < 1e-6);
        }
        // Duotone at f extremes lands on its two poles.
        let a = pattern_hsv(0.55, 0.9, 1.0);
        let z = pattern_hsv(glsl_fract(0.55 + 0.5), 0.9, 1.0);
        assert_ne!(a, z);
        // Brightness scales and clamps.
        p.color_mode = PatternColorMode::Mono;
        p.brightness = 0.0;
        let dark = pattern_synth_pixel(&p, [0.3, 0.3], ASPECT, 0.0);
        assert_eq!(dark, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn zoom_centre_and_rotation_share_the_authored_frame() {
        // The exact centre is invariant under zoom and rotation when no
        // centre offset is authored: p is the zero vector there either way.
        let mut p = PatternSynthParams {
            shape: PatternShape::Radial,
            color_mode: PatternColorMode::Mono,
            ..PatternSynthParams::default()
        };
        let base = pattern_synth_pixel(&p, [0.5, 0.5], ASPECT, 1.0);
        p.zoom = 0.7;
        p.rotate = 0.4;
        let moved = pattern_synth_pixel(&p, [0.5, 0.5], ASPECT, 1.0);
        assert!((base[0] - moved[0]).abs() < 1e-5);
        // An authored centre offset moves the picture.
        p.center_x = 0.4;
        let offset = pattern_synth_pixel(&p, [0.5, 0.5], ASPECT, 1.0);
        assert!((offset[0] - moved[0]).abs() > 1e-4);
    }

    #[test]
    fn the_uniform_is_exactly_128_bytes_and_carries_the_codes() {
        let u = PatternSynthGpuUniforms::from_params(&PatternSynthParams::default(), 2.0);
        assert_eq!(std::mem::size_of::<PatternSynthGpuUniforms>(), 128);
        assert_eq!(u.modes, [0, 0, 1, 0]);
        assert_eq!(u.color_b[2], 2.0);
        // Hostile time is neutralized rather than uploaded.
        let h = PatternSynthGpuUniforms::from_params(&PatternSynthParams::default(), f32::NAN);
        assert_eq!(h.color_b[2], 0.0);
    }

    #[test]
    fn glsl_fract_is_always_non_negative() {
        assert!((glsl_fract(-0.25) - 0.75).abs() < 1e-6);
        assert!((glsl_fract(1.25) - 0.25).abs() < 1e-6);
        assert_eq!(glsl_fract(0.0), 0.0);
    }
}
