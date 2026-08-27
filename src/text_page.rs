//! The B7 text-page source law — a static typeset page, rastered on the CPU.
//!
//! A text page is a self-contained authored picture: a background, an
//! optional geometric shape fan underneath, and typeset text on top. The page
//! is rastered once per authored change into a bounded RGBA image and then
//! costs exactly what a still costs — no clock, no transport, no per-frame
//! work. The layout laws (size/track/rot/repeat, the shape fan with its
//! `1 - f*0.55` size taper, the row and line pitches) are derived from BENDR
//! (MIT, © 2026 Steve Blythe) and transcribed with attribution; the
//! deliberate house deviation is that BENDR's clocked terms (text scroll,
//! shape spin, shape pulse) are absent, because this tree's law is
//! re-render-only-on-authored-change — movement is authored downstream
//! through the spatial transform, effects, and Motion, which the instrument
//! already provides in abundance.
//!
//! Glyphs come from two bundled licensed faces (Hack, MIT; Ubuntu-Light,
//! UFL) already shipped inside `epaint_default_fonts` — a closed two-face
//! vocabulary rather than BENDR's 33 host-font stacks, so the same authored
//! page rasters byte-identically on every host. This module is pure CPU in
//! the `gesture.rs` tradition: no wgpu, clock, filesystem, or UI dependency.

use ab_glyph::{Font, FontRef, ScaleFont};

use crate::performance::AuthoringValueLaw;

/// The fixed page. An authored page has no native resolution, so the tree
/// gives it one — fixed rather than output-sized, because an export at a
/// different output size must raster the identical picture.
pub const TEXT_PAGE_WIDTH: u32 = 1_920;
pub const TEXT_PAGE_HEIGHT: u32 = 1_080;

/// Authored body cap, checked before any layout work. Sanitize truncates on
/// a character boundary rather than refusing the whole page.
pub const TEXT_PAGE_MAX_BODY_BYTES: usize = 4_096;

/// The outline stroke never dilates further than this radius in pixels, so
/// the morphological band stays bounded regardless of authored width.
const TEXT_PAGE_MAX_STROKE_RADIUS: f32 = 10.0;

const TEXT_PAGE_SIZE_RANGE: [f32; 2] = [0.03, 0.6];
const TEXT_PAGE_TRACK_RANGE: [f32; 2] = [-0.1, 0.5];
const TEXT_PAGE_UNIT_RANGE: [f32; 2] = [0.0, 1.0];
const TEXT_PAGE_ROTATION_RANGE: [f32; 2] = [-180.0, 180.0];
const TEXT_PAGE_OUTLINE_RANGE: [f32; 2] = [0.0, 20.0];
const TEXT_PAGE_SHAPE_SIZE_RANGE: [f32; 2] = [0.02, 1.0];
const TEXT_PAGE_SHAPE_STROKE_RANGE: [f32; 2] = [0.0, 40.0];
const TEXT_PAGE_REPEAT_RANGE: [i64; 2] = [1, 8];
const TEXT_PAGE_SHAPE_COUNT_RANGE: [i64; 2] = [1, 24];

/// Canonical scalar wire paths and bounds. Both the parser and performance
/// recorder oracle consume this table, so a newly appended field cannot be
/// recordable under a range different from the one ingress validates.
const TEXT_PAGE_SCALAR_RANGES: &[(&str, [f32; 2])] = &[
    ("size", TEXT_PAGE_SIZE_RANGE),
    ("track", TEXT_PAGE_TRACK_RANGE),
    ("x", TEXT_PAGE_UNIT_RANGE),
    ("y", TEXT_PAGE_UNIT_RANGE),
    ("rot_degrees", TEXT_PAGE_ROTATION_RANGE),
    ("outline", TEXT_PAGE_OUTLINE_RANGE),
    ("ink_r", TEXT_PAGE_UNIT_RANGE),
    ("ink_g", TEXT_PAGE_UNIT_RANGE),
    ("ink_b", TEXT_PAGE_UNIT_RANGE),
    ("bg_r", TEXT_PAGE_UNIT_RANGE),
    ("bg_g", TEXT_PAGE_UNIT_RANGE),
    ("bg_b", TEXT_PAGE_UNIT_RANGE),
    ("shape_size", TEXT_PAGE_SHAPE_SIZE_RANGE),
    ("shape_x", TEXT_PAGE_UNIT_RANGE),
    ("shape_y", TEXT_PAGE_UNIT_RANGE),
    ("shape_fill_r", TEXT_PAGE_UNIT_RANGE),
    ("shape_fill_g", TEXT_PAGE_UNIT_RANGE),
    ("shape_fill_b", TEXT_PAGE_UNIT_RANGE),
    ("shape_stroke", TEXT_PAGE_SHAPE_STROKE_RANGE),
];

fn text_page_scalar_range(param: &str) -> Option<(&'static str, [f32; 2])> {
    TEXT_PAGE_SCALAR_RANGES
        .iter()
        .find(|(key, _)| *key == param)
        .copied()
}

/// The closed two-face vocabulary. Codes are permanent and append-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TextPageFont {
    /// Hack (MIT), the bundled monospace face.
    #[default]
    Mono,
    /// Ubuntu-Light (UFL), the bundled sans face.
    Sans,
}

impl TextPageFont {
    /// Permanent append-only code; the page has no GPU consumer, so the
    /// table exists to freeze the vocabulary and is pinned by tests.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the frozen vocabulary table the tests pin")
    )]
    pub const fn code(self) -> u32 {
        match self {
            Self::Mono => 0,
            Self::Sans => 1,
        }
    }

    pub const ALL: [Self; 2] = [Self::Mono, Self::Sans];
}

/// The shape fan under the text. Codes are permanent and append-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TextPageShape {
    #[default]
    None,
    Circle,
    Ring,
    Rect,
    Tri,
    Cross,
    Bars,
    Grid,
    Rings,
    Starburst,
}

impl TextPageShape {
    /// Permanent append-only code, pinned by tests like the font table.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the frozen vocabulary table the tests pin")
    )]
    pub const fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Circle => 1,
            Self::Ring => 2,
            Self::Rect => 3,
            Self::Tri => 4,
            Self::Cross => 5,
            Self::Bars => 6,
            Self::Grid => 7,
            Self::Rings => 8,
            Self::Starburst => 9,
        }
    }

    pub const ALL: [Self; 10] = [
        Self::None,
        Self::Circle,
        Self::Ring,
        Self::Rect,
        Self::Tri,
        Self::Cross,
        Self::Bars,
        Self::Grid,
        Self::Rings,
        Self::Starburst,
    ];

    /// BENDR strokes rings and starbursts even when no stroke is authored.
    fn always_stroked(self) -> bool {
        matches!(self, Self::Rings | Self::Starburst)
    }
}

/// Performance-recordable value law for one text-page wire field. Body text
/// remains a counted refusal because one recorder event carries only one
/// scalar or closed-vocabulary token.
pub(crate) fn text_page_value_law(param: &str) -> Option<AuthoringValueLaw> {
    match param {
        "font" => Some(AuthoringValueLaw::Discrete(
            TextPageFont::ALL
                .into_iter()
                .map(TextPageFont::key)
                .collect(),
        )),
        "shape" => Some(AuthoringValueLaw::Discrete(
            TextPageShape::ALL
                .into_iter()
                .map(TextPageShape::key)
                .collect(),
        )),
        "repeat" => Some(AuthoringValueLaw::Stepped(TEXT_PAGE_REPEAT_RANGE)),
        "shape_count" => Some(AuthoringValueLaw::Stepped(TEXT_PAGE_SHAPE_COUNT_RANGE)),
        "body" => None,
        _ => text_page_scalar_range(param).map(|(_, range)| AuthoringValueLaw::Unit(range)),
    }
}

/// The authored text-page state. Everything here is authored topology — no
/// field is a modulation destination, because modulating any of them would
/// force a re-raster per frame and the page's whole law is that it costs
/// what a still costs between edits.
#[derive(Debug, Clone, PartialEq)]
pub struct TextPageParams {
    /// The page text; newline-separated lines. Bounded by
    /// [`TEXT_PAGE_MAX_BODY_BYTES`].
    pub body: String,
    pub font: TextPageFont,
    /// Glyph size as a fraction of page height.
    pub size: f32,
    /// Letter tracking in em units added between glyphs.
    pub track: f32,
    /// Text anchor, page UV.
    pub x: f32,
    pub y: f32,
    /// Text rotation about its anchor, degrees.
    pub rot_degrees: f32,
    /// Repeated text rows.
    pub repeat: u32,
    /// Outline stroke width in pixels; above 0.2 the text draws as a stroke
    /// band instead of a fill, BENDR's own gate.
    pub outline: f32,
    /// Text colour, display-domain unit RGB.
    pub ink: [f32; 3],
    /// Page background colour.
    pub bg: [f32; 3],
    /// The shape fan.
    pub shape: TextPageShape,
    pub shape_count: u32,
    /// Shape size as a fraction of page height.
    pub shape_size: f32,
    /// Fan centre, page UV.
    pub shape_x: f32,
    pub shape_y: f32,
    pub shape_fill: [f32; 3],
    /// Shape stroke width in pixels; above 0.2 shapes stroke instead of fill.
    pub shape_stroke: f32,
}

impl Default for TextPageParams {
    fn default() -> Self {
        Self {
            body: "COLLIDE".to_string(),
            font: TextPageFont::Mono,
            size: 0.2,
            track: 0.0,
            x: 0.5,
            y: 0.5,
            rot_degrees: 0.0,
            repeat: 1,
            outline: 0.0,
            ink: [1.0, 1.0, 1.0],
            bg: [0.0, 0.0, 0.0],
            shape: TextPageShape::None,
            shape_count: 1,
            shape_size: 0.3,
            shape_x: 0.5,
            shape_y: 0.5,
            shape_fill: [1.0, 0.184, 0.627],
            shape_stroke: 0.0,
        }
    }
}

impl TextPageParams {
    /// Clamp every authored value into its declared range. Hostile
    /// non-finite input takes the neutral default rather than a clamped
    /// extreme; an oversized body truncates on a character boundary.
    pub fn sanitized(&self) -> Self {
        let defaults = Self::default();
        let mut body = self.body.clone();
        if body.len() > TEXT_PAGE_MAX_BODY_BYTES {
            let mut cut = TEXT_PAGE_MAX_BODY_BYTES;
            while cut > 0 && !body.is_char_boundary(cut) {
                cut -= 1;
            }
            body.truncate(cut);
        }
        Self {
            body,
            font: self.font,
            size: finite_clamp(
                self.size,
                defaults.size,
                TEXT_PAGE_SIZE_RANGE[0],
                TEXT_PAGE_SIZE_RANGE[1],
            ),
            track: finite_clamp(
                self.track,
                defaults.track,
                TEXT_PAGE_TRACK_RANGE[0],
                TEXT_PAGE_TRACK_RANGE[1],
            ),
            x: finite_clamp(
                self.x,
                defaults.x,
                TEXT_PAGE_UNIT_RANGE[0],
                TEXT_PAGE_UNIT_RANGE[1],
            ),
            y: finite_clamp(
                self.y,
                defaults.y,
                TEXT_PAGE_UNIT_RANGE[0],
                TEXT_PAGE_UNIT_RANGE[1],
            ),
            rot_degrees: finite_clamp(
                self.rot_degrees,
                defaults.rot_degrees,
                TEXT_PAGE_ROTATION_RANGE[0],
                TEXT_PAGE_ROTATION_RANGE[1],
            ),
            repeat: self.repeat.clamp(
                TEXT_PAGE_REPEAT_RANGE[0] as u32,
                TEXT_PAGE_REPEAT_RANGE[1] as u32,
            ),
            outline: finite_clamp(
                self.outline,
                defaults.outline,
                TEXT_PAGE_OUTLINE_RANGE[0],
                TEXT_PAGE_OUTLINE_RANGE[1],
            ),
            ink: unit_rgb(self.ink, defaults.ink),
            bg: unit_rgb(self.bg, defaults.bg),
            shape: self.shape,
            shape_count: self.shape_count.clamp(
                TEXT_PAGE_SHAPE_COUNT_RANGE[0] as u32,
                TEXT_PAGE_SHAPE_COUNT_RANGE[1] as u32,
            ),
            shape_size: finite_clamp(
                self.shape_size,
                defaults.shape_size,
                TEXT_PAGE_SHAPE_SIZE_RANGE[0],
                TEXT_PAGE_SHAPE_SIZE_RANGE[1],
            ),
            shape_x: finite_clamp(
                self.shape_x,
                defaults.shape_x,
                TEXT_PAGE_UNIT_RANGE[0],
                TEXT_PAGE_UNIT_RANGE[1],
            ),
            shape_y: finite_clamp(
                self.shape_y,
                defaults.shape_y,
                TEXT_PAGE_UNIT_RANGE[0],
                TEXT_PAGE_UNIT_RANGE[1],
            ),
            shape_fill: unit_rgb(self.shape_fill, defaults.shape_fill),
            shape_stroke: finite_clamp(
                self.shape_stroke,
                defaults.shape_stroke,
                TEXT_PAGE_SHAPE_STROKE_RANGE[0],
                TEXT_PAGE_SHAPE_STROKE_RANGE[1],
            ),
        }
    }
}

/// The two bundled faces, parsed once from the embedded bytes.
pub struct TextPageFonts {
    mono: FontRef<'static>,
    sans: FontRef<'static>,
}

/// The process-wide face set: parsed once from the embedded bytes, shared by
/// every raster site (live authoring, patch apply, export reconstruction).
pub fn bundled_fonts() -> &'static TextPageFonts {
    static FONTS: std::sync::OnceLock<TextPageFonts> = std::sync::OnceLock::new();
    FONTS.get_or_init(TextPageFonts::load)
}

impl TextPageFonts {
    /// Parse the embedded faces. The bytes are compiled into the binary, so
    /// failure here is a build defect, not a runtime input.
    pub fn load() -> Self {
        Self {
            mono: FontRef::try_from_slice(epaint_default_fonts::HACK_REGULAR)
                .expect("bundled Hack face parses"),
            sans: FontRef::try_from_slice(epaint_default_fonts::UBUNTU_LIGHT)
                .expect("bundled Ubuntu-Light face parses"),
        }
    }

    fn face(&self, font: TextPageFont) -> &FontRef<'static> {
        match font {
            TextPageFont::Mono => &self.mono,
            TextPageFont::Sans => &self.sans,
        }
    }
}

/// Raster the authored page into straight-alpha RGBA8 display-domain bytes,
/// `TEXT_PAGE_WIDTH x TEXT_PAGE_HEIGHT`, alpha always 255 — the page is
/// opaque like BENDR's canvas, and the operator keys it downstream when
/// transparency is wanted. Deterministic: the same authored state rasters
/// the same bytes on every host, because both faces are bundled.
pub fn render_text_page(params: &TextPageParams, fonts: &TextPageFonts) -> Vec<u8> {
    let q = params.sanitized();
    let w = TEXT_PAGE_WIDTH as usize;
    let h = TEXT_PAGE_HEIGHT as usize;
    let bg = rgb_bytes(q.bg);
    let mut page = vec![0u8; w * h * 4];
    for px in page.chunks_exact_mut(4) {
        px[0] = bg[0];
        px[1] = bg[1];
        px[2] = bg[2];
        px[3] = 255;
    }
    draw_shape_fan(&mut page, w, h, &q);
    draw_text(&mut page, w, h, &q, fonts);
    page
}

/// Blend `color` over the page at `coverage`, straight alpha in display
/// bytes — the canvas-compositing law the source transcribes.
fn blend_pixel(
    page: &mut [u8],
    w: usize,
    x: i64,
    y: i64,
    h: usize,
    color: [f32; 3],
    coverage: f32,
) {
    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
        return;
    }
    let c = coverage.clamp(0.0, 1.0);
    if c <= 0.0 {
        return;
    }
    let idx = (y as usize * w + x as usize) * 4;
    for ch in 0..3 {
        let base = page[idx + ch] as f32 / 255.0;
        let out = base + (color[ch] - base) * c;
        page[idx + ch] = (out * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// One-pixel antialiased coverage from a signed distance (negative inside).
fn fill_coverage(d: f32) -> f32 {
    (0.5 - d).clamp(0.0, 1.0)
}

/// Stroke coverage: a band of `width` centred on the zero contour.
fn stroke_coverage(d: f32, width: f32) -> f32 {
    (0.5 - (d.abs() - width * 0.5)).clamp(0.0, 1.0)
}

fn sd_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let pa = [p[0] - a[0], p[1] - a[1]];
    let ba = [b[0] - a[0], b[1] - a[1]];
    let denom = ba[0] * ba[0] + ba[1] * ba[1];
    let t = if denom <= f32::EPSILON {
        0.0
    } else {
        ((pa[0] * ba[0] + pa[1] * ba[1]) / denom).clamp(0.0, 1.0)
    };
    let d = [pa[0] - ba[0] * t, pa[1] - ba[1] * t];
    (d[0] * d[0] + d[1] * d[1]).sqrt()
}

fn sd_box(p: [f32; 2], half: [f32; 2]) -> f32 {
    let q = [p[0].abs() - half[0], p[1].abs() - half[1]];
    let outside = ((q[0].max(0.0)).powi(2) + (q[1].max(0.0)).powi(2)).sqrt();
    outside + q[0].max(q[1]).min(0.0)
}

/// Signed distance for one shape instance in its local (rotated) frame with
/// instance size `sz`. Returns `(distance, force_stroke_width)` where a
/// `Some` width means the shape draws as strokes at that width regardless of
/// the fill/stroke gate (BENDR's rings/starburst rule).
fn shape_distance(shape: TextPageShape, p: [f32; 2], sz: f32) -> f32 {
    match shape {
        TextPageShape::None => f32::MAX,
        TextPageShape::Circle => (p[0] * p[0] + p[1] * p[1]).sqrt() - sz,
        TextPageShape::Ring => {
            // The annulus between the BENDR path's two arcs.
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            (r - (sz * 0.775)).abs() - sz * 0.225
        }
        TextPageShape::Rect => sd_box(p, [sz, sz * 0.62]),
        TextPageShape::Tri => {
            // The BENDR triangle: apex up at -sz, base at +0.5 sz.
            let a = [0.0, -sz];
            let b = [sz * 0.87, sz * 0.5];
            let c = [-sz * 0.87, sz * 0.5];
            let inside = point_in_tri(p, a, b, c);
            let d = sd_segment(p, a, b)
                .min(sd_segment(p, b, c))
                .min(sd_segment(p, c, a));
            if inside {
                -d
            } else {
                d
            }
        }
        TextPageShape::Cross => sd_box(p, [sz, sz * 0.18]).min(sd_box(p, [sz * 0.18, sz])),
        TextPageShape::Bars => {
            // Eight vertical bars: x in [-sz + b*sz/4, -sz + b*sz/4 + sz/8].
            let mut d = f32::MAX;
            for b in 0..8 {
                let cx = -sz + b as f32 * sz / 4.0 + sz / 16.0;
                d = d.min(sd_box([p[0] - cx, p[1]], [sz / 16.0, sz]));
            }
            d
        }
        TextPageShape::Grid => {
            // Nine thin verticals and nine thin horizontals.
            let mut d = f32::MAX;
            for b in -4i32..=4 {
                let c = b as f32 * sz / 4.0;
                d = d.min(sd_box([p[0] - c, p[1]], [sz * 0.02, sz]));
                d = d.min(sd_box([p[0], p[1] - c], [sz, sz * 0.02]));
            }
            d
        }
        TextPageShape::Rings => {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let mut d = f32::MAX;
            for b in 1..=6 {
                d = d.min((r - sz * b as f32 / 6.0).abs());
            }
            d
        }
        TextPageShape::Starburst => {
            let mut d = f32::MAX;
            for b in 0..16 {
                #[allow(
                    clippy::approx_constant,
                    reason = "BENDR indexes rays at b*PI/8; the literal mirrors the transcription"
                )]
                let a = b as f32 * 3.14159 / 8.0;
                d = d.min(sd_segment(p, [0.0, 0.0], [a.cos() * sz, a.sin() * sz]));
            }
            d
        }
    }
}

fn point_in_tri(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let sign = |p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]| {
        (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    };
    let d1 = sign(p, a, b);
    let d2 = sign(p, b, c);
    let d3 = sign(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// The shape fan: `n` instances about the fan centre, each rotated by its
/// fan angle and tapered by `1 - f*0.55`, BENDR's own law with the clocked
/// spin and pulse terms at their zero.
fn draw_shape_fan(page: &mut [u8], w: usize, h: usize, q: &TextPageParams) {
    if q.shape == TextPageShape::None {
        return;
    }
    let n = q.shape_count.max(1);
    let cx = q.shape_x * w as f32;
    let cy = q.shape_y * h as f32;
    let stroked = q.shape_stroke > 0.2 || q.shape.always_stroked();
    let stroke_width = if q.shape_stroke > 0.2 {
        q.shape_stroke.max(1.0)
    } else {
        2.0
    };
    for i in 0..n {
        let f = if n == 1 {
            0.0
        } else {
            i as f32 / (n - 1) as f32
        };
        let angle = f * PAGE_TAU / n as f32;
        let sz = q.shape_size * h as f32 * (1.0 - f * 0.55);
        if sz <= 0.5 {
            continue;
        }
        // Instance bounding box: every shape fits inside radius ~1.2 sz plus
        // the stroke band.
        let reach = sz * 1.25 + stroke_width + 2.0;
        let x0 = ((cx - reach).floor() as i64).max(0);
        let x1 = ((cx + reach).ceil() as i64).min(w as i64 - 1);
        let y0 = ((cy - reach).floor() as i64).max(0);
        let y1 = ((cy + reach).ceil() as i64).min(h as i64 - 1);
        let (sa, ca) = angle.sin_cos();
        for py in y0..=y1 {
            for px in x0..=x1 {
                let dx = px as f32 + 0.5 - cx;
                let dy = py as f32 + 0.5 - cy;
                // Inverse-rotate into the instance frame.
                let local = [dx * ca + dy * sa, -dx * sa + dy * ca];
                let d = shape_distance(q.shape, local, sz);
                let cov = if stroked {
                    stroke_coverage(d, stroke_width)
                } else {
                    fill_coverage(d)
                };
                blend_pixel(page, w, px, py, h, q.shape_fill, cov);
            }
        }
    }
}

#[allow(
    clippy::approx_constant,
    reason = "the same BENDR circle constant the pattern synth keeps"
)]
const PAGE_TAU: f32 = 6.283_185_3;

/// Typeset the body: BENDR's layout law at its clocked terms' zero. Each
/// repeat row rasters into a page-sized coverage scratch and rotates about
/// its own anchor, exactly as the canvas transform composed per row.
fn draw_text(page: &mut [u8], w: usize, h: usize, q: &TextPageParams, fonts: &TextPageFonts) {
    let lines: Vec<&str> = q.body.split('\n').collect();
    if lines.iter().all(|l| l.is_empty()) {
        return;
    }
    let px = (q.size * h as f32).max(4.0);
    let font = fonts.face(q.font);
    let scaled = font.as_scaled(ab_glyph::PxScale::from(px));
    let reps = q.repeat.max(1);
    let row_pitch = px * 1.35 * if reps > 1 { 1.2 } else { 1.0 };
    let anchor_x = fract01(q.x) * w as f32;
    let anchor_base_y = fract01(q.y) * h as f32;
    let stroke = q.outline > 0.2;
    let mut scratch = vec![0f32; w * h];
    for r in 0..reps {
        let anchor_y = anchor_base_y + (r as f32 - (reps - 1) as f32 / 2.0) * row_pitch;
        for v in scratch.iter_mut() {
            *v = 0.0;
        }
        let mut any = false;
        for (li, line) in lines.iter().enumerate() {
            if line.is_empty() {
                continue;
            }
            let yy = (li as f32 - (lines.len() - 1) as f32 / 2.0) * px * 1.15;
            // Middle baseline: the canvas 'middle' alignment approximated on
            // the em box.
            let baseline = anchor_y + yy + (scaled.ascent() + scaled.descent()) * 0.5;
            // Measure: advances plus kerning plus tracking.
            let glyph_ids: Vec<ab_glyph::GlyphId> =
                line.chars().map(|c| scaled.glyph_id(c)).collect();
            let tracking = q.track * px;
            let mut total = 0.0f32;
            for (i, gid) in glyph_ids.iter().enumerate() {
                total += scaled.h_advance(*gid);
                if i + 1 < glyph_ids.len() {
                    total += scaled.kern(*gid, glyph_ids[i + 1]) + tracking;
                }
            }
            let mut pen = anchor_x - total / 2.0;
            for (i, gid) in glyph_ids.iter().enumerate() {
                let glyph = gid.with_scale_and_position(px, ab_glyph::point(pen, baseline));
                if let Some(outline) = font.outline_glyph(glyph) {
                    let bounds = outline.px_bounds();
                    outline.draw(|gx, gy, cov| {
                        let tx = bounds.min.x as i64 + gx as i64;
                        let ty = bounds.min.y as i64 + gy as i64;
                        if tx >= 0 && ty >= 0 && (tx as usize) < w && (ty as usize) < h {
                            let idx = ty as usize * w + tx as usize;
                            scratch[idx] = scratch[idx].max(cov);
                        }
                    });
                    any = true;
                }
                pen += scaled.h_advance(*gid);
                if i + 1 < glyph_ids.len() {
                    pen += scaled.kern(*gid, glyph_ids[i + 1]) + tracking;
                }
            }
        }
        if !any {
            continue;
        }
        if stroke {
            stroke_band_in_place(&mut scratch, w, h, q.outline);
        }
        composite_rotated(
            page,
            w,
            h,
            &scratch,
            [anchor_x, anchor_y],
            q.rot_degrees,
            q.ink,
        );
    }
}

/// Turn a fill-coverage scratch into an outline band of the authored width:
/// dilate by half the width, erode by half the width, keep the difference.
/// A morphological rewrite of the canvas stroke, bounded by
/// [`TEXT_PAGE_MAX_STROKE_RADIUS`].
fn stroke_band_in_place(scratch: &mut [f32], w: usize, h: usize, outline: f32) {
    let radius = (outline * 0.5).clamp(0.5, TEXT_PAGE_MAX_STROKE_RADIUS);
    let r = radius.ceil() as i64;
    let source = scratch.to_vec();
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut dilated = 0.0f32;
            let mut eroded = 1.0f32;
            for dy in -r..=r {
                for dx in -r..=r {
                    if (dx * dx + dy * dy) as f32 > radius * radius {
                        continue;
                    }
                    let sx = x + dx;
                    let sy = y + dy;
                    let v = if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                        0.0
                    } else {
                        source[sy as usize * w + sx as usize]
                    };
                    dilated = dilated.max(v);
                    eroded = eroded.min(v);
                }
            }
            scratch[y as usize * w + x as usize] = (dilated - eroded).clamp(0.0, 1.0);
        }
    }
}

/// Composite a coverage scratch onto the page, rotated about the anchor.
/// Zero rotation takes the exact direct path.
fn composite_rotated(
    page: &mut [u8],
    w: usize,
    h: usize,
    scratch: &[f32],
    anchor: [f32; 2],
    rot_degrees: f32,
    ink: [f32; 3],
) {
    let [anchor_x, anchor_y] = anchor;
    if rot_degrees == 0.0 {
        for y in 0..h {
            for x in 0..w {
                let cov = scratch[y * w + x];
                if cov > 0.0 {
                    blend_pixel(page, w, x as i64, y as i64, h, ink, cov);
                }
            }
        }
        return;
    }
    let theta = rot_degrees.to_radians();
    let (s, c) = theta.sin_cos();
    for y in 0..h {
        for x in 0..w {
            // Inverse-rotate the destination pixel about the anchor and
            // sample the unrotated scratch bilinearly.
            let dx = x as f32 + 0.5 - anchor_x;
            let dy = y as f32 + 0.5 - anchor_y;
            let sx = anchor_x + dx * c + dy * s - 0.5;
            let sy = anchor_y - dx * s + dy * c - 0.5;
            let cov = sample_bilinear(scratch, w, h, sx, sy);
            if cov > 0.0 {
                blend_pixel(page, w, x as i64, y as i64, h, ink, cov);
            }
        }
    }
}

fn sample_bilinear(scratch: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    if x < -1.0 || y < -1.0 || x > w as f32 || y > h as f32 {
        return 0.0;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let read = |ix: i64, iy: i64| -> f32 {
        if ix < 0 || iy < 0 || ix >= w as i64 || iy >= h as i64 {
            0.0
        } else {
            scratch[iy as usize * w + ix as usize]
        }
    };
    let x0 = x0 as i64;
    let y0 = y0 as i64;
    let a = read(x0, y0) * (1.0 - fx) + read(x0 + 1, y0) * fx;
    let b = read(x0, y0 + 1) * (1.0 - fx) + read(x0 + 1, y0 + 1) * fx;
    a * (1.0 - fy) + b * fy
}

fn fract01(x: f32) -> f32 {
    let f = x - x.floor();
    if f.is_finite() {
        f
    } else {
        0.5
    }
}

fn rgb_bytes(rgb: [f32; 3]) -> [u8; 3] {
    [
        (rgb[0] * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        (rgb[1] * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        (rgb[2] * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
    ]
}

fn unit_rgb(rgb: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        out[i] = if rgb[i].is_finite() {
            rgb[i].clamp(0.0, 1.0)
        } else {
            fallback[i]
        };
    }
    out
}

fn finite_clamp(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

impl TextPageFont {
    pub const fn key(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::Sans => "sans",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|font| font.key() == key)
    }
}

impl TextPageShape {
    pub const fn key(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Circle => "circle",
            Self::Ring => "ring",
            Self::Rect => "rect",
            Self::Tri => "tri",
            Self::Cross => "cross",
            Self::Bars => "bars",
            Self::Grid => "grid",
            Self::Rings => "rings",
            Self::Starburst => "starburst",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|shape| shape.key() == key)
    }
}

/// One validated wire edit to a text-page layer — the same single-table law
/// as [`crate::pattern_synth::PatternSynthEdit`].
#[derive(Debug, Clone, PartialEq)]
pub enum TextPageEdit {
    Body(String),
    Font(TextPageFont),
    Shape(TextPageShape),
    Repeat(u32),
    ShapeCount(u32),
    Scalar(&'static str, f32),
}

impl TextPageEdit {
    /// Parse and range-validate one wire edit. An oversized body, an unknown
    /// param, an out-of-range number, and an unknown token are rejections.
    pub fn parse(param: &str, value: &serde_json::Value) -> Option<Self> {
        let number = |min: f32, max: f32| -> Option<f32> {
            let number = value.as_f64()? as f32;
            (number.is_finite() && (min..=max).contains(&number)).then_some(number)
        };
        Some(match param {
            "body" => {
                let body = value.as_str()?;
                if body.len() > TEXT_PAGE_MAX_BODY_BYTES {
                    return None;
                }
                Self::Body(body.to_string())
            }
            "font" => Self::Font(TextPageFont::from_key(value.as_str()?)?),
            "shape" => Self::Shape(TextPageShape::from_key(value.as_str()?)?),
            "repeat" => {
                let repeat = value.as_u64()?;
                if !(TEXT_PAGE_REPEAT_RANGE[0] as u64..=TEXT_PAGE_REPEAT_RANGE[1] as u64)
                    .contains(&repeat)
                {
                    return None;
                }
                Self::Repeat(repeat as u32)
            }
            "shape_count" => {
                let count = value.as_u64()?;
                if !(TEXT_PAGE_SHAPE_COUNT_RANGE[0] as u64..=TEXT_PAGE_SHAPE_COUNT_RANGE[1] as u64)
                    .contains(&count)
                {
                    return None;
                }
                Self::ShapeCount(count as u32)
            }
            _ => {
                let (key, [min, max]) = text_page_scalar_range(param)?;
                Self::Scalar(key, number(min, max)?)
            }
        })
    }

    /// Apply this validated edit onto a copy of the authored state. The
    /// caller hands the result to the layer, whose change detection decides
    /// whether a re-raster is due.
    pub fn apply(&self, params: &mut TextPageParams) {
        match self {
            Self::Body(body) => params.body.clone_from(body),
            Self::Font(font) => params.font = *font,
            Self::Shape(shape) => params.shape = *shape,
            Self::Repeat(repeat) => params.repeat = *repeat,
            Self::ShapeCount(count) => params.shape_count = *count,
            Self::Scalar(field, value) => match *field {
                "size" => params.size = *value,
                "track" => params.track = *value,
                "x" => params.x = *value,
                "y" => params.y = *value,
                "rot_degrees" => params.rot_degrees = *value,
                "outline" => params.outline = *value,
                "ink_r" => params.ink[0] = *value,
                "ink_g" => params.ink[1] = *value,
                "ink_b" => params.ink[2] = *value,
                "bg_r" => params.bg[0] = *value,
                "bg_g" => params.bg[1] = *value,
                "bg_b" => params.bg[2] = *value,
                "shape_size" => params.shape_size = *value,
                "shape_x" => params.shape_x = *value,
                "shape_y" => params.shape_y = *value,
                "shape_fill_r" => params.shape_fill[0] = *value,
                "shape_fill_g" => params.shape_fill[1] = *value,
                "shape_fill_b" => params.shape_fill[2] = *value,
                "shape_stroke" => params.shape_stroke = *value,
                _ => unreachable!("parse admits only the closed vocabulary"),
            },
        }
        *params = params.sanitized();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_laws_share_the_parser_tables() {
        for &(param, range) in TEXT_PAGE_SCALAR_RANGES {
            assert_eq!(
                text_page_value_law(param),
                Some(AuthoringValueLaw::Unit(range)),
                "{param}"
            );
            assert!(TextPageEdit::parse(param, &serde_json::json!(range[0])).is_some());
            assert!(TextPageEdit::parse(param, &serde_json::json!(range[1])).is_some());
            let margin = (range[1] - range[0]).abs().max(1.0) * 0.01;
            assert!(
                TextPageEdit::parse(param, &serde_json::json!(range[0] - margin)).is_none(),
                "{param} below range"
            );
            assert!(
                TextPageEdit::parse(param, &serde_json::json!(range[1] + margin)).is_none(),
                "{param} above range"
            );
        }

        for (param, expected) in [
            (
                "font",
                TextPageFont::ALL
                    .into_iter()
                    .map(TextPageFont::key)
                    .collect::<Vec<_>>(),
            ),
            (
                "shape",
                TextPageShape::ALL
                    .into_iter()
                    .map(TextPageShape::key)
                    .collect::<Vec<_>>(),
            ),
        ] {
            assert_eq!(
                text_page_value_law(param),
                Some(AuthoringValueLaw::Discrete(expected.clone()))
            );
            for token in expected {
                assert!(TextPageEdit::parse(param, &serde_json::json!(token)).is_some());
            }
        }

        for (param, range) in [
            ("repeat", TEXT_PAGE_REPEAT_RANGE),
            ("shape_count", TEXT_PAGE_SHAPE_COUNT_RANGE),
        ] {
            assert_eq!(
                text_page_value_law(param),
                Some(AuthoringValueLaw::Stepped(range))
            );
            assert!(TextPageEdit::parse(param, &serde_json::json!(range[0])).is_some());
            assert!(TextPageEdit::parse(param, &serde_json::json!(range[1])).is_some());
            assert!(TextPageEdit::parse(param, &serde_json::json!(range[0] - 1)).is_none());
            assert!(TextPageEdit::parse(param, &serde_json::json!(range[1] + 1)).is_none());
        }

        assert_eq!(text_page_value_law("body"), None);
        assert_eq!(text_page_value_law("not_a_field"), None);
    }

    fn pixel(page: &[u8], x: usize, y: usize) -> [u8; 4] {
        let idx = (y * TEXT_PAGE_WIDTH as usize + x) * 4;
        [page[idx], page[idx + 1], page[idx + 2], page[idx + 3]]
    }

    #[test]
    fn defaults_survive_sanitize_and_hostile_input_takes_neutral_values() {
        let d = TextPageParams::default();
        assert_eq!(d.sanitized(), d);
        let hostile = TextPageParams {
            size: f32::NAN,
            rot_degrees: f32::INFINITY,
            ink: [f32::NAN, 2.0, -1.0],
            repeat: 999,
            ..TextPageParams::default()
        };
        let q = hostile.sanitized();
        assert_eq!(q.size, d.size);
        assert_eq!(q.rot_degrees, d.rot_degrees);
        assert_eq!(q.ink, [1.0, 1.0, 0.0]);
        assert_eq!(q.repeat, 8);
    }

    #[test]
    fn an_oversized_body_truncates_on_a_character_boundary() {
        // 4095 ASCII bytes then a multi-byte char straddling the cap.
        let p = TextPageParams {
            body: format!("{}é", "a".repeat(TEXT_PAGE_MAX_BODY_BYTES - 1)),
            ..TextPageParams::default()
        };
        let q = p.sanitized();
        assert!(q.body.len() <= TEXT_PAGE_MAX_BODY_BYTES);
        assert!(q.body.is_char_boundary(q.body.len()));
        assert_eq!(q.body, "a".repeat(TEXT_PAGE_MAX_BODY_BYTES - 1));
    }

    #[test]
    fn the_page_is_opaque_and_carries_the_background() {
        let p = TextPageParams {
            body: String::new(),
            bg: [0.2, 0.4, 0.6],
            ..TextPageParams::default()
        };
        let fonts = TextPageFonts::load();
        let page = render_text_page(&p, &fonts);
        assert_eq!(
            page.len(),
            (TEXT_PAGE_WIDTH * TEXT_PAGE_HEIGHT * 4) as usize
        );
        let px = pixel(&page, 10, 10);
        assert_eq!(px, [51, 102, 153, 255]);
    }

    #[test]
    fn text_reaches_the_page_and_rerender_is_deterministic() {
        let p = TextPageParams {
            body: "X".to_string(),
            ink: [1.0, 0.0, 0.0],
            bg: [0.0, 0.0, 0.0],
            ..TextPageParams::default()
        };
        let fonts = TextPageFonts::load();
        let a = render_text_page(&p, &fonts);
        let b = render_text_page(&p, &fonts);
        assert_eq!(a, b, "the same authored page must raster the same bytes");
        // Some pixel near the centre carries ink.
        let cx = TEXT_PAGE_WIDTH as usize / 2;
        let cy = TEXT_PAGE_HEIGHT as usize / 2;
        let mut found = false;
        for y in cy.saturating_sub(120)..cy + 120 {
            for x in cx.saturating_sub(120)..cx + 120 {
                let px = pixel(&a, x, y);
                if px[0] > 128 && px[1] < 64 {
                    found = true;
                }
            }
        }
        assert!(found, "the glyph must land near the anchor");
    }

    #[test]
    fn different_authored_state_rasters_a_different_page() {
        let fonts = TextPageFonts::load();
        let a = render_text_page(&TextPageParams::default(), &fonts);
        let moved = TextPageParams {
            body: "OTHER".to_string(),
            ..TextPageParams::default()
        };
        let b = render_text_page(&moved, &fonts);
        assert_ne!(a, b);
        let rotated = TextPageParams {
            rot_degrees: 45.0,
            ..TextPageParams::default()
        };
        let c = render_text_page(&rotated, &fonts);
        assert_ne!(a, c);
    }

    #[test]
    fn the_two_faces_raster_differently_and_both_parse() {
        let fonts = TextPageFonts::load();
        let mut p = TextPageParams {
            body: "Aa".to_string(),
            ..TextPageParams::default()
        };
        let mono = render_text_page(&p, &fonts);
        p.font = TextPageFont::Sans;
        let sans = render_text_page(&p, &fonts);
        assert_ne!(mono, sans);
    }

    #[test]
    fn the_shape_fan_lands_filled_and_stroked_shapes_where_the_law_says() {
        let fonts = TextPageFonts::load();
        let mut p = TextPageParams {
            body: String::new(),
            bg: [0.0, 0.0, 0.0],
            shape: TextPageShape::Circle,
            shape_fill: [0.0, 1.0, 0.0],
            shape_size: 0.3,
            ..TextPageParams::default()
        };
        let page = render_text_page(&p, &fonts);
        // The fan centre sits inside the filled circle.
        let cx = (p.shape_x * TEXT_PAGE_WIDTH as f32) as usize;
        let cy = (p.shape_y * TEXT_PAGE_HEIGHT as f32) as usize;
        assert_eq!(pixel(&page, cx, cy)[1], 255);
        // Rings always stroke: the centre stays background.
        p.shape = TextPageShape::Rings;
        let rings = render_text_page(&p, &fonts);
        assert_eq!(pixel(&rings, cx, cy)[1], 0);
        // A ring lands at radius sz*b/6.
        let sz = p.shape_size * TEXT_PAGE_HEIGHT as f32;
        let rx = cx + sz as usize;
        assert!(pixel(&rings, rx, cy)[1] > 128);
    }

    #[test]
    fn shape_codes_are_frozen_and_append_only() {
        let codes: Vec<u32> = TextPageShape::ALL.iter().map(|s| s.code()).collect();
        assert_eq!(codes, (0..10).collect::<Vec<u32>>());
        let fonts: Vec<u32> = TextPageFont::ALL.iter().map(|f| f.code()).collect();
        assert_eq!(fonts, vec![0, 1]);
    }

    #[test]
    fn repeat_rows_and_outline_change_the_picture() {
        let fonts = TextPageFonts::load();
        let mut p = TextPageParams {
            body: "ROW".to_string(),
            ..TextPageParams::default()
        };
        let one = render_text_page(&p, &fonts);
        p.repeat = 3;
        let three = render_text_page(&p, &fonts);
        assert_ne!(one, three);
        p.repeat = 1;
        p.outline = 4.0;
        let outlined = render_text_page(&p, &fonts);
        assert_ne!(one, outlined);
    }
}
