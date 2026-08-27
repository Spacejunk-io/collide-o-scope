//! B11 monitoring bay: preview-only waveform, vectorscope, and PROBE
//! instruments over a low-resolution readback of an internal signal.
//!
//! The law is derived from BENDR (MIT, © 2026 Steve Blythe), whose scope dock
//! reads the finished programme back at 128×72 and 10 Hz, only while the
//! scope tab is visible, and plots a Rec.601 luma waveform beside a U/V
//! vectorscope with the six 75% colour-bar targets. This module is the
//! independent CPU reference in the `gesture.rs` tradition: no wgpu, no
//! clock, no filesystem dependency beyond the egui paint helpers at the
//! bottom, which consume only data this module derived.
//!
//! Three laws hold everything together:
//!
//! - **Preview-only.** The instruments render to the editor preview and the
//!   browser panel; no audience surface can receive them. The sealed
//!   [`MonitorBayPermit`] (the transform-gizmo seal, deliberately not
//!   stage_health's weaker file-scope shape) folds the bay toggle,
//!   `native_controls_visible`, and the surface check into one token, so a
//!   painter cannot be called for an audience surface even by mistake.
//! - **Zero cost hidden.** The readback is armed only while the native pane
//!   shows the bay or a fresh browser watcher has declared itself; unarmed,
//!   no pass encodes, no buffer maps, and the snapshot block is the empty
//!   default.
//! - **One law, two dumb presenters.** The instrument bitmaps are computed
//!   here, on the CPU, from the readback grid; the native egui painter and
//!   the panel canvas both draw the identical bytes.

use serde::{Deserialize, Serialize};

/// Monitor readback grid width. BENDR's shipped scope size, inside the
/// tranche's ≤160×90 bound (the B10 precedent of following the shipped law).
pub const MONITOR_BAY_WIDTH: usize = 128;
/// Monitor readback grid height.
pub const MONITOR_BAY_HEIGHT: usize = 72;
/// Cells in one readback grid.
pub const MONITOR_BAY_CELLS: usize = MONITOR_BAY_WIDTH * MONITOR_BAY_HEIGHT;
/// The 10 Hz cadence expressed on the 30 Hz reference grid, exactly the B10
/// video-analysis law: three reference ticks per sample.
pub const MONITOR_BAY_INTERVAL_TICKS: u32 = 3;
pub const MONITOR_BAY_INTERVAL_SECONDS: f64 =
    MONITOR_BAY_INTERVAL_TICKS as f64 / crate::effects::params::TEMPORAL_REFERENCE_FPS as f64;

/// Waveform bitmap width — one column per grid column, so the plot needs no
/// horizontal resampling law.
pub const WAVEFORM_WIDTH: usize = MONITOR_BAY_WIDTH;
/// Waveform bitmap height (luma resolution of the plot).
pub const WAVEFORM_HEIGHT: usize = 64;
/// Vectorscope bitmap edge.
pub const SCOPE_SIZE: usize = 64;
/// Intensity one grid sample deposits in an instrument cell. Additive with
/// saturation, BENDR's accumulation law: coincident samples brighten.
pub const INSTRUMENT_HIT: u8 = 40;
/// BENDR's vectorscope gain: U/V are scaled by 1.4 before plotting, so the
/// 75% bars sit comfortably inside the graticule.
pub const SCOPE_GAIN: f32 = 1.4;

/// Rec.601 luma on the encoded bytes — deliberately the same constants and
/// the same (encoded) space as `VideoAnalysisState::analyze` and BENDR's
/// scope, because the instrument observes the stored picture, not a linear
/// reconstruction of it.
pub fn rec601_luma(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

/// BENDR's vectorscope projection: colour-difference axes scaled to the
/// classic graticule, computed on encoded unit values.
pub fn vectorscope_uv(r: f32, g: f32, b: f32) -> (f32, f32) {
    let y = rec601_luma(r, g, b);
    ((b - y) * 0.565, (r - y) * 0.713)
}

/// One vectorscope graticule target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScopeTarget {
    pub label: &'static str,
    /// Normalized [0,1] position inside the scope bitmap, x rightward and
    /// y downward, derived from the same projection the cloud uses.
    pub x: f32,
    pub y: f32,
}

/// Normalized scope position for a U/V pair: centre 0.5, full radius at half
/// the bitmap, gain applied exactly as to the plotted cloud.
pub fn scope_position(u: f32, v: f32) -> (f32, f32) {
    (0.5 + u * SCOPE_GAIN * 0.5, 0.5 - v * SCOPE_GAIN * 0.5)
}

/// The six 75% colour-bar targets, derived from the same projection rather
/// than restated as constants, so the graticule cannot drift from the cloud.
pub fn scope_targets() -> [ScopeTarget; 6] {
    const BARS: [(&str, f32, f32, f32); 6] = [
        ("R", 0.75, 0.0, 0.0),
        ("G", 0.0, 0.75, 0.0),
        ("B", 0.0, 0.0, 0.75),
        ("YL", 0.75, 0.75, 0.0),
        ("CY", 0.0, 0.75, 0.75),
        ("MG", 0.75, 0.0, 0.75),
    ];
    BARS.map(|(label, r, g, b)| {
        let (u, v) = vectorscope_uv(r, g, b);
        let (x, y) = scope_position(u, v);
        ScopeTarget { label, x, y }
    })
}

/// The two instrument bitmaps computed from one readback grid. Intensity is
/// a plain u8 per cell; presentation (tint, graticule) belongs to the
/// painters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorInstruments {
    /// `WAVEFORM_WIDTH × WAVEFORM_HEIGHT`, row-major, row 0 at luma 1.0.
    pub waveform: Vec<u8>,
    /// `SCOPE_SIZE × SCOPE_SIZE`, row-major, +V upward (row 0 at +V).
    pub vectorscope: Vec<u8>,
}

impl Default for MonitorInstruments {
    fn default() -> Self {
        Self {
            waveform: vec![0; WAVEFORM_WIDTH * WAVEFORM_HEIGHT],
            vectorscope: vec![0; SCOPE_SIZE * SCOPE_SIZE],
        }
    }
}

/// Compute both instruments from one RGBA readback grid
/// (`MONITOR_BAY_WIDTH × MONITOR_BAY_HEIGHT × 4` bytes, stored sRGB codes).
/// A short or empty grid yields dark instruments rather than a panic —
/// hostile input draws nothing.
pub fn reduce_instruments(grid: &[u8]) -> MonitorInstruments {
    let mut out = MonitorInstruments::default();
    if grid.len() < MONITOR_BAY_CELLS * 4 {
        return out;
    }
    for y in 0..MONITOR_BAY_HEIGHT {
        for x in 0..MONITOR_BAY_WIDTH {
            let base = (y * MONITOR_BAY_WIDTH + x) * 4;
            let r = f32::from(grid[base]) / 255.0;
            let g = f32::from(grid[base + 1]) / 255.0;
            let b = f32::from(grid[base + 2]) / 255.0;
            let luma = rec601_luma(r, g, b).clamp(0.0, 1.0);
            let row = ((1.0 - luma) * (WAVEFORM_HEIGHT - 1) as f32).round() as usize;
            let row = row.min(WAVEFORM_HEIGHT - 1);
            let cell = &mut out.waveform[row * WAVEFORM_WIDTH + x];
            *cell = cell.saturating_add(INSTRUMENT_HIT);

            let (u, v) = vectorscope_uv(r, g, b);
            let (fx, fy) = scope_position(u, v);
            let sx = (fx * (SCOPE_SIZE - 1) as f32)
                .round()
                .clamp(0.0, (SCOPE_SIZE - 1) as f32) as usize;
            let sy = (fy * (SCOPE_SIZE - 1) as f32)
                .round()
                .clamp(0.0, (SCOPE_SIZE - 1) as f32) as usize;
            let cell = &mut out.vectorscope[sy * SCOPE_SIZE + sx];
            *cell = cell.saturating_add(INSTRUMENT_HIT);
        }
    }
    out
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn unit_byte(value: f32) -> u8 {
    (finite_or_zero(value).clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Encode one NTSC sync-latch line for the diagnostic probe. The signed
/// displacement is normalized by the engine's permanent bound: positive is
/// red, negative is blue, and an exact zero is opaque black. Hostile
/// non-finite input is the neutral zero state, never a diagnostic colour.
pub fn sync_line_state_color(offset: f32) -> [u8; 4] {
    let bound = crate::sync_latch::SYNC_LATCH_MAX_OFFSET;
    let normalized = finite_or_zero(offset).clamp(-bound, bound) / bound;
    [
        unit_byte(normalized.max(0.0)),
        0,
        unit_byte((-normalized).max(0.0)),
        255,
    ]
}

/// Reduce the live per-line sync-latch table to the monitor bay's fixed
/// 128×72 RGBA oracle. Output row `y` samples input row
/// `floor(y * offsets.len() / 72)` and repeats that colour across the row.
/// A table shorter than the diagnostic height is not materialized enough to
/// represent the probe, so it yields one all-black opaque grid.
pub fn reduce_sync_line_state(offsets: &[f32]) -> Vec<u8> {
    let mut grid = vec![0; MONITOR_BAY_CELLS * 4];
    for alpha in grid.iter_mut().skip(3).step_by(4) {
        *alpha = 255;
    }
    if offsets.len() < MONITOR_BAY_HEIGHT {
        return grid;
    }

    for y in 0..MONITOR_BAY_HEIGHT {
        let source_y = y * offsets.len() / MONITOR_BAY_HEIGHT;
        let color = sync_line_state_color(offsets[source_y]);
        for x in 0..MONITOR_BAY_WIDTH {
            let base = (y * MONITOR_BAY_WIDTH + x) * 4;
            grid[base..base + 4].copy_from_slice(&color);
        }
    }
    grid
}

/// Encode one bus-melt band-mask sample. The retained mask is a unit scalar;
/// its diagnostic is opaque grayscale after finite-or-zero sanitization and
/// clamping to the mask's [0,1] domain.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the CPU diagnostic oracle is exercised by renderer parity fixtures"
    )
)]
pub fn melt_band_color(mask: f32) -> [u8; 4] {
    let gray = unit_byte(mask);
    [gray, gray, gray, 255]
}

/// Encode one admitted master motion-field sample for the diagnostic probe.
/// Red is signed horizontal velocity, green is inverted signed vertical
/// velocity, and blue is confidence gated by visibility. Velocity uses the
/// motion engine's own permanent bound; every lane is finite-or-zero and
/// clamped before byte encoding.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the CPU diagnostic oracle is exercised by renderer parity fixtures"
    )
)]
pub fn motion_field_color(
    velocity_uv_per_second: [f32; 2],
    confidence: f32,
    visibility: f32,
) -> [u8; 4] {
    let vx = finite_or_zero(velocity_uv_per_second[0]);
    let vy = finite_or_zero(velocity_uv_per_second[1]);
    let confidence = finite_or_zero(confidence).clamp(0.0, 1.0);
    let visibility = finite_or_zero(visibility).clamp(0.0, 1.0);
    [
        unit_byte((vx / crate::motion::MOTION_MAX_UV_PER_SECOND + 1.0) * 0.5),
        unit_byte((1.0 - vy / crate::motion::MOTION_MAX_UV_PER_SECOND) * 0.5),
        unit_byte(confidence * visibility),
        255,
    ]
}

/// The PROBE vocabulary: which internal signal feeds the instruments. Closed
/// and append-only (codes are permanent); every member is a retained,
/// renderer-owned image that costs nothing to observe. The named deferred
/// probes — the NTSC per-line state, the melt band mask, a motion-field
/// visualizer — join by appending codes, never by renumbering.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MonitorProbe {
    /// The finished programme picture (pre-blackout slot 2), code 0.
    #[default]
    Program = 0,
    /// The B16 programme re-entry tap (the honest N-1 image), code 1.
    ProgramTap = 1,
    /// The gesture-canvas presented donor (the etch field), code 2.
    GestureCanvas = 2,
    /// The live sync-latch offset table rendered as signed line state, code 3.
    NtscLineState = 3,
    /// The retained bus-melt band mask rendered as grayscale, code 4.
    MeltBandMask = 4,
    /// The master scope's exact admitted motion field, code 5. If that field
    /// is absent or has not materialized, the probe is unavailable; it never
    /// rebinds to a layer field, raw field, or any other nearby producer.
    MotionField = 5,
}

impl MonitorProbe {
    /// Frozen append-only code order. New probes may only be appended.
    pub const ALL: [Self; 6] = [
        Self::Program,
        Self::ProgramTap,
        Self::GestureCanvas,
        Self::NtscLineState,
        Self::MeltBandMask,
        Self::MotionField,
    ];

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "permanent probe codes are asserted by the append-only vocabulary fixture"
        )
    )]
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn key(&self) -> &'static str {
        match self {
            Self::Program => "program",
            Self::ProgramTap => "program_tap",
            Self::GestureCanvas => "gesture_canvas",
            Self::NtscLineState => "ntsc_line_state",
            Self::MeltBandMask => "melt_band_mask",
            Self::MotionField => "motion_field",
        }
    }

    /// The one shared parse table (the B8 `BusMixerEdit` law): the server
    /// gate and the applier both answer from this function, so the accepted
    /// and applied vocabularies are structurally one. An unknown token is a
    /// refusal, never a fallback onto `Program`.
    pub fn try_from_str(token: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|probe| probe.key() == token)
    }
}

/// Standard base64 (RFC 4648, with padding) for the snapshot's bitmap
/// payloads. Hand-rolled deliberately: the instrument bytes are the only
/// consumer, and a direct dependency for twenty lines would churn the lock.
pub fn encode_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// One live modulation source published to the bay.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MonitorSourceSnapshot {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub value: f32,
}

/// The additive snapshot block. Default is the inactive bay: empty payloads,
/// zero cost for every pre-B11 snapshot consumer and for every frame the bay
/// is unarmed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MonitorBaySnapshot {
    /// True only while the bay is armed and publishing samples.
    #[serde(default)]
    pub active: bool,
    /// The operator's native-overlay toggle, mirrored so the panel checkbox
    /// stays truthful even while the bay is unarmed.
    #[serde(default)]
    pub native_overlay: bool,
    /// The authored probe token (always present so the panel select can
    /// reflect the engine even while inactive).
    #[serde(default)]
    pub probe: String,
    /// Empty, or a named condition ("unavailable") when the authored probe's
    /// producer is absent — the instruments then hold and the readback stays
    /// idle. Never a silent rebind onto another producer.
    #[serde(default)]
    pub probe_status: String,
    /// Monotonic sample counter so the panel redraws only on fresh data.
    #[serde(default)]
    pub sample: u64,
    #[serde(default)]
    pub waveform_width: u32,
    #[serde(default)]
    pub waveform_height: u32,
    #[serde(default)]
    pub scope_size: u32,
    /// Base64 of the waveform intensity bitmap; empty while inactive.
    #[serde(default)]
    pub waveform_b64: String,
    /// Base64 of the vectorscope intensity bitmap; empty while inactive.
    #[serde(default)]
    pub scope_b64: String,
    /// The modulation matrix's live source values — the PROBE's CPU half.
    #[serde(default)]
    pub sources: Vec<MonitorSourceSnapshot>,
}

/// Host-side bay state: the latest instruments and a freshness counter. The
/// GPU pool and the arming predicate live with the host; this is only the
/// derived data both presenters read.
#[derive(Debug, Default)]
pub struct MonitorBayState {
    instruments: Option<MonitorInstruments>,
    sample_index: u64,
    dirty: bool,
}

impl MonitorBayState {
    /// Ingest one harvested readback grid.
    pub fn ingest(&mut self, grid: &[u8]) {
        self.instruments = Some(reduce_instruments(grid));
        self.sample_index = self.sample_index.wrapping_add(1);
        self.dirty = true;
    }

    /// Clear on the disarm edge, so a re-arm never resurrects a stale
    /// picture (the B4 wake-clearing instinct).
    pub fn clear(&mut self) {
        if self.instruments.is_some() {
            self.instruments = None;
            self.dirty = true;
        }
    }

    pub fn instruments(&self) -> Option<&MonitorInstruments> {
        self.instruments.as_ref()
    }

    /// True once per fresh sample (or clear); consumed by the native painter
    /// to re-upload its textures only when the data moved.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// Build the snapshot block for an armed bay. The caller supplies the
    /// authored probe, its availability status, and the live source values.
    pub fn snapshot(
        &self,
        probe: MonitorProbe,
        probe_status: &str,
        sources: Vec<MonitorSourceSnapshot>,
    ) -> MonitorBaySnapshot {
        let (waveform_b64, scope_b64) = match &self.instruments {
            Some(instruments) => (
                encode_base64(&instruments.waveform),
                encode_base64(&instruments.vectorscope),
            ),
            None => (String::new(), String::new()),
        };
        MonitorBaySnapshot {
            active: true,
            // The caller owns the toggle fact; the state cannot see it.
            native_overlay: false,
            probe: probe.key().to_string(),
            probe_status: probe_status.to_string(),
            sample: self.sample_index,
            waveform_width: WAVEFORM_WIDTH as u32,
            waveform_height: WAVEFORM_HEIGHT as u32,
            scope_size: SCOPE_SIZE as u32,
            waveform_b64,
            scope_b64,
            sources,
        }
    }
}

/// The sealed preview permit, in the transform-gizmo shape and for the same
/// reason: nesting the seal makes this file's own `mod tests` a *sibling* of
/// the private field, so not even the bay's tests can forge a token, and a
/// source audit pins the single declaration and single construction. The
/// permit folds all three conditions — the operator's bay toggle, the
/// native-controls leakage predicate, and the surface check — because a
/// permit that checked fewer would mint on a surface the check's caller got
/// wrong (the gizmo's single-monitor-audience lesson).
mod permit {
    use crate::stage_map::{native_controls_visible, StageSurface, StageToolState};

    /// Opaque proof that the caller is painting the editor preview with the
    /// bay enabled and the native controls visible.
    pub struct MonitorBayPermit(());

    pub fn monitor_bay_permit(
        tools: &StageToolState,
        surface: &StageSurface,
        output_on_main: bool,
    ) -> Option<MonitorBayPermit> {
        (tools.monitor_bay_enabled()
            && native_controls_visible(output_on_main)
            && matches!(surface, StageSurface::EditorPreview))
        .then_some(MonitorBayPermit(()))
    }
}

pub use permit::{monitor_bay_permit, MonitorBayPermit};

/// Everything the native painter needs, resolved by the host before the egui
/// closure so the closure borrows no `App` state.
pub struct MonitorBayPaintInput<'a> {
    pub waveform: Option<(egui::TextureId, egui::Vec2)>,
    pub vectorscope: Option<(egui::TextureId, egui::Vec2)>,
    pub probe: MonitorProbe,
    pub probe_status: &'a str,
    pub sources: &'a [MonitorSourceSnapshot],
}

/// Tint an intensity bitmap into the classic green-phosphor instrument
/// image. Shared by both native textures so the two instruments read as one
/// bay.
pub fn instrument_color_image(intensity: &[u8], width: usize, height: usize) -> egui::ColorImage {
    let mut image =
        egui::ColorImage::new([width, height], vec![egui::Color32::BLACK; width * height]);
    for (pixel, &value) in image.pixels.iter_mut().zip(intensity.iter()) {
        let v = value;
        *pixel = egui::Color32::from_rgb(v / 5, v, v / 3);
    }
    image
}

/// Paint the bay onto the editor preview. Text and geometry only beyond the
/// two pre-uploaded instrument textures; the signature is the boundary — it
/// cannot receive an audience surface because the permit cannot be minted
/// for one.
pub fn paint_monitor_bay(
    ui: &mut egui::Ui,
    _permit: &MonitorBayPermit,
    input: &MonitorBayPaintInput,
) {
    egui::Frame::popup(ui.style()).show(ui, |ui| {
        ui.strong("MONITOR");
        ui.horizontal(|ui| {
            if let Some((texture, size)) = input.waveform {
                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                ui.painter().image(
                    texture,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                // Graticule: black and white luma lines.
                for luma in [0.0f32, 1.0] {
                    let y = rect.top() + (1.0 - luma) * rect.height();
                    ui.painter().line_segment(
                        [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                        egui::Stroke::new(1.0_f32, egui::Color32::from_gray(90)),
                    );
                }
            }
            if let Some((texture, size)) = input.vectorscope {
                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                ui.painter().image(
                    texture,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
                let centre = rect.center();
                for radius in [0.33f32, 0.66, 1.0] {
                    ui.painter().circle_stroke(
                        centre,
                        radius * rect.width() * 0.5,
                        egui::Stroke::new(1.0_f32, egui::Color32::from_gray(70)),
                    );
                }
                for target in scope_targets() {
                    let pos = egui::pos2(
                        rect.left() + target.x * rect.width(),
                        rect.top() + target.y * rect.height(),
                    );
                    ui.painter().circle_stroke(
                        pos,
                        2.5,
                        egui::Stroke::new(1.0_f32, egui::Color32::from_gray(140)),
                    );
                }
            }
        });
        let status = if input.probe_status.is_empty() {
            String::new()
        } else {
            format!(" — {}", input.probe_status)
        };
        ui.label(format!("PROBE: {}{status}", input.probe.key()));
        if !input.sources.is_empty() {
            egui::Grid::new("monitor-bay-sources")
                .num_columns(6)
                .spacing([10.0, 1.0])
                .show(ui, |ui| {
                    for (index, source) in input.sources.iter().enumerate() {
                        ui.weak(&source.name);
                        ui.monospace(format!("{:+.2}", source.value));
                        if index % 3 == 2 {
                            ui.end_row();
                        }
                    }
                });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage_map::{StageSurface, StageToolState};

    fn grid_of(rgba: [u8; 4]) -> Vec<u8> {
        let mut grid = Vec::with_capacity(MONITOR_BAY_CELLS * 4);
        for _ in 0..MONITOR_BAY_CELLS {
            grid.extend_from_slice(&rgba);
        }
        grid
    }

    #[test]
    fn rec601_luma_matches_the_analysis_law() {
        assert!((rec601_luma(1.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((rec601_luma(1.0, 0.0, 0.0) - 0.299).abs() < 1e-6);
        assert!((rec601_luma(0.0, 1.0, 0.0) - 0.587).abs() < 1e-6);
        assert!((rec601_luma(0.0, 0.0, 1.0) - 0.114).abs() < 1e-6);
    }

    #[test]
    fn vectorscope_projection_matches_the_bendr_axes() {
        // Pure 75% red: y = 0.224250, u = -y*0.565, v = (0.75-y)*0.713.
        let (u, v) = vectorscope_uv(0.75, 0.0, 0.0);
        assert!((u - (-0.224_25 * 0.565)).abs() < 1e-5);
        assert!((v - (0.525_75 * 0.713)).abs() < 1e-5);
        // Neutral grey has no colour difference.
        let (u, v) = vectorscope_uv(0.5, 0.5, 0.5);
        assert!(u.abs() < 1e-6 && v.abs() < 1e-6);
    }

    #[test]
    fn scope_targets_derive_from_the_projection() {
        let targets = scope_targets();
        assert_eq!(targets.len(), 6);
        let red = targets[0];
        assert_eq!(red.label, "R");
        let (u, v) = vectorscope_uv(0.75, 0.0, 0.0);
        let (x, y) = scope_position(u, v);
        assert!((red.x - x).abs() < 1e-6 && (red.y - y).abs() < 1e-6);
        // Complementary bars land mirrored through the centre.
        let cyan = targets[4];
        assert!((red.x + cyan.x - 1.0).abs() < 1e-4);
        assert!((red.y + cyan.y - 1.0).abs() < 1e-4);
        // Every target stays inside the unit bitmap at the shipped gain.
        for target in targets {
            assert!((0.0..=1.0).contains(&target.x));
            assert!((0.0..=1.0).contains(&target.y));
        }
    }

    #[test]
    fn a_flat_grey_field_is_one_saturated_waveform_row_and_one_scope_point() {
        let instruments = reduce_instruments(&grid_of([128, 128, 128, 255]));
        let luma = rec601_luma(128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0);
        let row = ((1.0 - luma) * (WAVEFORM_HEIGHT - 1) as f32).round() as usize;
        for y in 0..WAVEFORM_HEIGHT {
            for x in 0..WAVEFORM_WIDTH {
                let value = instruments.waveform[y * WAVEFORM_WIDTH + x];
                if y == row {
                    // 72 hits per column saturate the additive law.
                    assert_eq!(value, 255, "column {x} must saturate on its luma row");
                } else {
                    assert_eq!(value, 0, "row {y} col {x} must stay dark");
                }
            }
        }
        let lit: Vec<usize> = instruments
            .vectorscope
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(lit.len(), 1, "a neutral field is a single scope point");
        let centre = (SCOPE_SIZE / 2 - 1) * SCOPE_SIZE..=(SCOPE_SIZE / 2 + 1) * SCOPE_SIZE;
        assert!(
            centre.contains(&lit[0]),
            "the neutral point sits at the scope centre"
        );
    }

    #[test]
    fn a_two_tone_column_splits_the_waveform_and_hostile_input_draws_nothing() {
        let mut grid = grid_of([0, 0, 0, 255]);
        // Top half white, bottom half black.
        for y in 0..MONITOR_BAY_HEIGHT / 2 {
            for x in 0..MONITOR_BAY_WIDTH {
                let base = (y * MONITOR_BAY_WIDTH + x) * 4;
                grid[base] = 255;
                grid[base + 1] = 255;
                grid[base + 2] = 255;
            }
        }
        let instruments = reduce_instruments(&grid);
        let top_row = 0usize;
        let bottom_row = WAVEFORM_HEIGHT - 1;
        assert!(instruments.waveform[top_row * WAVEFORM_WIDTH] > 0);
        assert!(instruments.waveform[bottom_row * WAVEFORM_WIDTH] > 0);
        // A short grid yields the dark default instead of panicking.
        let short = reduce_instruments(&[255u8; 16]);
        assert!(short.waveform.iter().all(|v| *v == 0));
        assert!(short.vectorscope.iter().all(|v| *v == 0));
    }

    #[test]
    fn probe_vocabulary_is_closed_and_round_trips() {
        let frozen = [
            (MonitorProbe::Program, 0, "program"),
            (MonitorProbe::ProgramTap, 1, "program_tap"),
            (MonitorProbe::GestureCanvas, 2, "gesture_canvas"),
            (MonitorProbe::NtscLineState, 3, "ntsc_line_state"),
            (MonitorProbe::MeltBandMask, 4, "melt_band_mask"),
            (MonitorProbe::MotionField, 5, "motion_field"),
        ];
        assert_eq!(
            MonitorProbe::ALL.map(|probe| probe.key()),
            frozen.map(|v| v.2)
        );
        for (probe, code, token) in frozen {
            assert_eq!(probe.code(), code);
            assert_eq!(probe.key(), token);
            assert_eq!(MonitorProbe::try_from_str(token), Some(probe));
        }
        for near_miss in [
            "programme",
            "ntsc-line-state",
            "NTSC_line_state",
            "ntsc_line_states",
            "melt_band",
            "melt-band-mask",
            "motion",
            "motion_fields",
            "motion_field ",
            "",
        ] {
            assert_eq!(
                MonitorProbe::try_from_str(near_miss),
                None,
                "near miss must be refused: {near_miss:?}"
            );
        }
        assert_eq!(MonitorProbe::default(), MonitorProbe::Program);
    }

    #[test]
    fn sync_line_state_oracle_is_signed_bounded_opaque_and_row_mapped() {
        use crate::sync_latch::SYNC_LATCH_MAX_OFFSET;

        assert_eq!(sync_line_state_color(0.0), [0, 0, 0, 255]);
        assert_eq!(
            sync_line_state_color(SYNC_LATCH_MAX_OFFSET * 0.5),
            [128, 0, 0, 255]
        );
        assert_eq!(
            sync_line_state_color(-SYNC_LATCH_MAX_OFFSET),
            [0, 0, 255, 255]
        );
        assert_eq!(
            sync_line_state_color(SYNC_LATCH_MAX_OFFSET * 4.0),
            [255, 0, 0, 255]
        );
        assert_eq!(sync_line_state_color(-f32::MAX), [0, 0, 255, 255]);
        for hostile in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(sync_line_state_color(hostile), [0, 0, 0, 255]);
        }

        let short_offsets = vec![SYNC_LATCH_MAX_OFFSET; MONITOR_BAY_HEIGHT - 1];
        for short in [&[][..], short_offsets.as_slice()] {
            let grid = reduce_sync_line_state(short);
            assert_eq!(grid.len(), MONITOR_BAY_CELLS * 4);
            assert!(grid.chunks_exact(4).all(|pixel| pixel == [0, 0, 0, 255]));
        }

        let mut offsets = vec![0.0; MONITOR_BAY_HEIGHT * 2];
        offsets[0] = -SYNC_LATCH_MAX_OFFSET;
        offsets[2] = SYNC_LATCH_MAX_OFFSET * 0.5;
        offsets[(MONITOR_BAY_HEIGHT - 1) * 2] = SYNC_LATCH_MAX_OFFSET;
        let grid = reduce_sync_line_state(&offsets);
        let pixel = |x: usize, y: usize| {
            let base = (y * MONITOR_BAY_WIDTH + x) * 4;
            &grid[base..base + 4]
        };
        assert_eq!(pixel(0, 0), [0, 0, 255, 255]);
        assert_eq!(pixel(MONITOR_BAY_WIDTH - 1, 1), [128, 0, 0, 255]);
        assert_eq!(pixel(0, MONITOR_BAY_HEIGHT - 1), [255, 0, 0, 255]);
    }

    #[test]
    fn melt_and_motion_color_oracles_pin_clamping_axes_and_visibility() {
        use crate::motion::MOTION_MAX_UV_PER_SECOND;

        assert_eq!(melt_band_color(-1.0), [0, 0, 0, 255]);
        assert_eq!(melt_band_color(0.5), [128, 128, 128, 255]);
        assert_eq!(melt_band_color(2.0), [255, 255, 255, 255]);
        assert_eq!(melt_band_color(f32::NAN), [0, 0, 0, 255]);

        assert_eq!(motion_field_color([0.0, 0.0], 0.0, 1.0), [128, 128, 0, 255]);
        assert_eq!(
            motion_field_color(
                [MOTION_MAX_UV_PER_SECOND, MOTION_MAX_UV_PER_SECOND],
                0.5,
                0.5,
            ),
            [255, 0, 64, 255]
        );
        assert_eq!(
            motion_field_color(
                [-MOTION_MAX_UV_PER_SECOND, -MOTION_MAX_UV_PER_SECOND],
                1.0,
                1.0,
            ),
            [0, 255, 255, 255]
        );
        assert_eq!(
            motion_field_color(
                [
                    MOTION_MAX_UV_PER_SECOND * 4.0,
                    -MOTION_MAX_UV_PER_SECOND * 4.0
                ],
                2.0,
                2.0,
            ),
            [255, 255, 255, 255]
        );
        assert_eq!(
            motion_field_color([f32::NAN, f32::INFINITY], f32::NAN, 1.0),
            [128, 128, 0, 255]
        );
    }

    #[test]
    fn base64_matches_the_rfc_vectors() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
        assert_eq!(encode_base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(encode_base64(b"Man"), "TWFu");
    }

    #[test]
    fn state_snapshot_is_inactive_by_default_and_publishes_fresh_samples() {
        let mut state = MonitorBayState::default();
        assert!(state.instruments().is_none());
        assert!(!state.take_dirty());
        let default_snapshot = MonitorBaySnapshot::default();
        assert!(!default_snapshot.active);
        assert!(default_snapshot.waveform_b64.is_empty());

        state.ingest(&grid_of([128, 128, 128, 255]));
        assert!(state.take_dirty());
        assert!(!state.take_dirty(), "dirty is consumed once");
        let snapshot = state.snapshot(MonitorProbe::Program, "", Vec::new());
        assert!(snapshot.active);
        assert_eq!(snapshot.sample, 1);
        assert_eq!(snapshot.probe, "program");
        assert!(!snapshot.waveform_b64.is_empty());
        assert!(!snapshot.scope_b64.is_empty());
        assert_eq!(snapshot.waveform_width, WAVEFORM_WIDTH as u32);

        state.clear();
        assert!(state.take_dirty(), "a clear is a visible change");
        assert!(state.instruments().is_none());
        let cleared = state.snapshot(MonitorProbe::ProgramTap, "unavailable", Vec::new());
        assert!(cleared.waveform_b64.is_empty());
        assert_eq!(cleared.probe_status, "unavailable");
    }

    #[test]
    fn the_permit_is_minted_only_for_a_visible_enabled_editor_preview() {
        let mut tools = StageToolState::default();
        assert!(
            monitor_bay_permit(&tools, &StageSurface::EditorPreview, false).is_none(),
            "the bay toggle gates the permit"
        );
        tools.set_monitor_bay(true);
        assert!(monitor_bay_permit(&tools, &StageSurface::EditorPreview, false).is_some());
        assert!(
            monitor_bay_permit(&tools, &StageSurface::EditorPreview, true).is_none(),
            "a single-monitor audience output refuses the permit"
        );
        let audience_surfaces = [
            StageSurface::Composite,
            StageSurface::Audience,
            StageSurface::Spout,
            StageSurface::Record,
            StageSurface::Export,
            StageSurface::PhysicalOutput(crate::stage_map::OutputEndpointId::legacy()),
        ];
        for surface in audience_surfaces {
            assert!(
                monitor_bay_permit(&tools, &surface, false).is_none(),
                "no audience surface may mint the permit: {surface:?}"
            );
        }
    }

    #[test]
    fn the_bay_permit_has_exactly_one_constructor() {
        let source = include_str!("monitor_bay.rs");
        let (production, _tests) = source
            .split_once("mod tests {")
            .expect("the tests module exists");
        let declaration = "pub struct MonitorBayPermit(());";
        assert_eq!(
            production.matches(declaration).count(),
            1,
            "exactly one permit declaration"
        );
        assert_eq!(
            production.matches("MonitorBayPermit(())").count(),
            2,
            "the declaration plus exactly one mint"
        );
        assert!(
            !production.contains("derive(Clone"),
            "the permit must not be duplicable"
        );
        let seal_start = production
            .find("mod permit {")
            .expect("the sealed submodule exists");
        let seal_end = production
            .find("pub use permit::")
            .expect("the re-export exists");
        let sealed = &production[seal_start..seal_end];
        assert_eq!(
            sealed.matches("MonitorBayPermit(())").count(),
            2,
            "declaration and mint both live inside the seal"
        );
    }
}
