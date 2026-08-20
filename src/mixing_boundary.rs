//! The B8 mixing-boundary law — bus wipes, the dirty-mixer fault stage, and
//! the melting edge.
//!
//! Three mechanisms share this module. The wipe family turns the bus A/B
//! crossfade from a constant into an analytic spatial matte with softness, a
//! movable origin, MULTI tiling, and a coloured border rule. The dirty mixer
//! is a bench mixer that has been dropped: an event clock decides when a
//! firing happens at all, and four fault laws (knock, cut, dropout, noise)
//! decide what a firing does — bit-clean between firings, because the tick
//! index is the only state. The melting edge treats a coverage boundary as a
//! feedback region: the matte is probed at four points a chosen distance
//! out, disagreement is the edge and its direction the normal, the incoming
//! picture is dragged along that normal, and the stage's own previous frame
//! is dissolved back in inside the band so the smear creeps outward instead
//! of washing out.
//!
//! The laws are derived from BENDR (MIT, © 2026 Steve Blythe) and rewritten
//! for this tree (Rust / wgpu / WGSL, linear light, Rec.709 luma where a
//! luma is needed, integer avalanche hashing on the Shift band/epoch/seed
//! precedent instead of BENDR's float hash — so live and export replay the
//! same faults from frame-plan time alone). The melt chroma law reconstructs
//! RGB from Y/I/Q and therefore uses the coherent 601 YIQ round trip the B3
//! feedback rig already carries; every other luma here is Rec.709. This
//! module is the independent CPU reference the GPU stages are checked
//! against, in the `gesture.rs` tradition: no wgpu, clock, filesystem, or UI
//! dependency.

// Production executes these laws through their GPU twins (`fs_bus` in
// `composition_host.wgsl` and `melting_edge.wgsl`); this module's evaluation
// half — the wipe/dirt/melt reference functions and their hash lanes — is
// the independent reference the fixtures check those twins against, and is
// deliberately consumed only by tests, exactly as the gesture-canvas and
// Study CPU references are. The param structs, sanitize laws, wake laws,
// and the shared wire-edit table are live production code.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

use crate::layers::BlendMode;

/// The wipe pattern vocabulary. Codes are permanent and append-only;
/// `Dissolve` is the exact historical constant-crossfade law.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WipePattern {
    #[default]
    Dissolve,
    WipeH,
    WipeV,
    Diagonal,
    Box,
    Circle,
    SplitH,
    SplitV,
    BlindsV,
    BlindsH,
    Clock,
    DiagBars,
    Blocks,
}

impl WipePattern {
    /// Every pattern in its permanent append-only numeric order.
    pub const ALL: [Self; 13] = [
        Self::Dissolve,
        Self::WipeH,
        Self::WipeV,
        Self::Diagonal,
        Self::Box,
        Self::Circle,
        Self::SplitH,
        Self::SplitV,
        Self::BlindsV,
        Self::BlindsH,
        Self::Clock,
        Self::DiagBars,
        Self::Blocks,
    ];

    /// Permanent append-only shader codes.
    pub const fn code(self) -> u32 {
        match self {
            Self::Dissolve => 0,
            Self::WipeH => 1,
            Self::WipeV => 2,
            Self::Diagonal => 3,
            Self::Box => 4,
            Self::Circle => 5,
            Self::SplitH => 6,
            Self::SplitV => 7,
            Self::BlindsV => 8,
            Self::BlindsH => 9,
            Self::Clock => 10,
            Self::DiagBars => 11,
            Self::Blocks => 12,
        }
    }

    /// Stable lowercase wire/patch token.
    pub const fn key(self) -> &'static str {
        match self {
            Self::Dissolve => "dissolve",
            Self::WipeH => "wipe_h",
            Self::WipeV => "wipe_v",
            Self::Diagonal => "diagonal",
            Self::Box => "box",
            Self::Circle => "circle",
            Self::SplitH => "split_h",
            Self::SplitV => "split_v",
            Self::BlindsV => "blinds_v",
            Self::BlindsH => "blinds_h",
            Self::Clock => "clock",
            Self::DiagBars => "diag_bars",
            Self::Blocks => "blocks",
        }
    }

    /// Parse the exact stable token without inventing aliases.
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|pattern| pattern.key() == key)
    }
}

/// The eight back colours a bench mixer offers, in the order they are always
/// listed. A closed table: border and key dressing fills select from it and
/// nothing interpolates between entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackColor {
    #[default]
    White,
    Yellow,
    Cyan,
    Green,
    Magenta,
    Red,
    Blue,
    Black,
}

impl BackColor {
    /// Every colour in its permanent append-only numeric order.
    pub const ALL: [Self; 8] = [
        Self::White,
        Self::Yellow,
        Self::Cyan,
        Self::Green,
        Self::Magenta,
        Self::Red,
        Self::Blue,
        Self::Black,
    ];

    /// Permanent append-only shader codes.
    pub const fn code(self) -> u32 {
        match self {
            Self::White => 0,
            Self::Yellow => 1,
            Self::Cyan => 2,
            Self::Green => 3,
            Self::Magenta => 4,
            Self::Red => 5,
            Self::Blue => 6,
            Self::Black => 7,
        }
    }

    /// Stable lowercase wire/patch token.
    pub const fn key(self) -> &'static str {
        match self {
            Self::White => "white",
            Self::Yellow => "yellow",
            Self::Cyan => "cyan",
            Self::Green => "green",
            Self::Magenta => "magenta",
            Self::Red => "red",
            Self::Blue => "blue",
            Self::Black => "black",
        }
    }

    /// Parse the exact stable token without inventing aliases.
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|color| color.key() == key)
    }

    /// The fill colour in linear light.
    pub const fn rgb(self) -> [f32; 3] {
        match self {
            Self::White => [1.0, 1.0, 1.0],
            Self::Yellow => [1.0, 1.0, 0.0],
            Self::Cyan => [0.0, 1.0, 1.0],
            Self::Green => [0.0, 1.0, 0.0],
            Self::Magenta => [1.0, 0.0, 1.0],
            Self::Red => [1.0, 0.0, 0.0],
            Self::Blue => [0.0, 0.0, 1.0],
            Self::Black => [0.0, 0.0, 0.0],
        }
    }
}

/// Below this an amount-style control is authored off and its stage takes
/// the exact prior path. Matches BENDR's own `> 0.002` gates.
pub const MIX_AMOUNT_EPSILON: f32 = 0.002;

/// The authored bus mix state beyond the crossfade itself: the wipe pattern
/// vocabulary and the blend family at the A/B meet. Frame-local evaluated
/// values — never topology.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BusMixParams {
    pub pattern: WipePattern,
    /// Feathered edge width. The fader remap keeps full A at 0 and full B
    /// at 1 despite the feather.
    pub soft: f32,
    /// Wipe origin, offset in half-frame units on each axis.
    pub origin_x: f32,
    pub origin_y: f32,
    /// Repetition count for blinds/bars/blocks (2 + detail * 14, floored).
    pub detail: f32,
    /// Inverts the field, not the fader.
    pub invert: bool,
    /// MULTI tiling: 1 is the whole frame, 2 is x4, 4 is x16.
    pub rep: u32,
    /// Border rule amount; the fill colour comes from the closed table.
    pub border: f32,
    pub border_color: BackColor,
    /// How A and B meet where both carry coverage. `Normal` is the exact
    /// historical premultiplied crossfade; `AlphaCut` is not authorable at
    /// the bus (a crossfade has no destination to cut).
    pub blend: BlendMode,
}

impl Default for BusMixParams {
    fn default() -> Self {
        Self {
            pattern: WipePattern::Dissolve,
            soft: 0.03,
            origin_x: 0.0,
            origin_y: 0.0,
            detail: 0.3,
            invert: false,
            rep: 1,
            border: 0.0,
            border_color: BackColor::White,
            blend: BlendMode::Normal,
        }
    }
}

impl BusMixParams {
    /// Clamp every authored value into its declared range. Hostile
    /// non-finite input takes the neutral default rather than a clamped
    /// extreme; an `AlphaCut` blend sanitizes to `Normal`.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            pattern: self.pattern,
            soft: finite_clamp(self.soft, defaults.soft, 0.0, 1.0),
            origin_x: finite_clamp(self.origin_x, defaults.origin_x, -1.0, 1.0),
            origin_y: finite_clamp(self.origin_y, defaults.origin_y, -1.0, 1.0),
            detail: finite_clamp(self.detail, defaults.detail, 0.0, 1.0),
            invert: self.invert,
            rep: self.rep.clamp(1, 4),
            border: finite_clamp(self.border, defaults.border, 0.0, 1.0),
            border_color: self.border_color,
            blend: if self.blend == BlendMode::AlphaCut {
                BlendMode::Normal
            } else {
                self.blend
            },
        }
    }

    /// The authored state selects the exact historical bus law: a plain
    /// premultiplied crossfade with no wipe, no border, and Normal meeting.
    /// Softness, origin, detail, invert, rep, and border colour are dressing
    /// on an authored pattern or border and wake nothing alone.
    pub fn is_exact_dissolve(self) -> bool {
        let clean = self.sanitized();
        clean.pattern == WipePattern::Dissolve
            && clean.blend == BlendMode::Normal
            && clean.border <= MIX_AMOUNT_EPSILON
    }
}

/// The dirty-mixer fault stage. `dirt` is the wake control: at zero the
/// stage is bit-clean off; the four flavour amounts are dressing on an
/// armed event clock and wake nothing alone.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DirtParams {
    /// Firing probability per tick (scaled by 0.85 so the clock never
    /// saturates), and the decay-length authority.
    pub dirt: f32,
    /// Event-clock rate: 0.5 + rate * 15 ticks per second.
    pub rate: f32,
    /// Line bands drop through to the other side of the crossbar or to
    /// nothing at all.
    pub drop: f32,
    /// The crossbar thrown to the wrong input for a moment.
    pub cut: f32,
    /// The timebase shoved sideways, crawling back inside the tick.
    pub knock: f32,
    /// The switching transient: band-limited monochrome spray with colour
    /// dropping out.
    pub noise: f32,
}

impl Default for DirtParams {
    fn default() -> Self {
        Self {
            dirt: 0.0,
            rate: 0.3,
            drop: 0.5,
            cut: 0.4,
            knock: 0.5,
            noise: 0.35,
        }
    }
}

impl DirtParams {
    /// Clamp every authored value into its declared range. Hostile
    /// non-finite input takes the neutral default rather than a clamped
    /// extreme.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            dirt: finite_clamp(self.dirt, defaults.dirt, 0.0, 1.0),
            rate: finite_clamp(self.rate, defaults.rate, 0.0, 1.0),
            drop: finite_clamp(self.drop, defaults.drop, 0.0, 1.0),
            cut: finite_clamp(self.cut, defaults.cut, 0.0, 1.0),
            knock: finite_clamp(self.knock, defaults.knock, 0.0, 1.0),
            noise: finite_clamp(self.noise, defaults.noise, 0.0, 1.0),
        }
    }

    /// The whole stage delegates exactly when `dirt` is authored off.
    pub fn is_exact_off(self) -> bool {
        self.sanitized().dirt <= MIX_AMOUNT_EPSILON
    }
}

/// The melting edge. `melt` wakes the band probe and the drag; the history
/// dissolve additionally needs `hold`, and the history surface is only
/// allocated when both are up — BENDR's own "costs nothing off" rule, and
/// this tree's delegation law.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MeltParams {
    /// Drag amount along the edge normal. The wake control.
    pub melt: f32,
    /// Probe distance: r = 0.004 + width * 0.085 in UV.
    pub width: f32,
    /// How much of the stage's own previous frame survives inside the band.
    /// Above 1 the band stops settling and keeps building.
    pub hold: f32,
    /// Rotates the drag from across the edge (0) toward along it (±1).
    pub swirl: f32,
    /// Colour runs further than luma off the edge, the way it does off a
    /// composite edge.
    pub chroma: f32,
    /// Which side of the seam melts: at 1 the shape bleeds into the
    /// background while the background never eats the shape.
    pub creep: f32,
}

impl Default for MeltParams {
    fn default() -> Self {
        Self {
            melt: 0.0,
            width: 0.3,
            hold: 0.6,
            swirl: 0.0,
            chroma: 0.5,
            creep: 0.35,
        }
    }
}

impl MeltParams {
    /// Clamp every authored value into its declared range. Hostile
    /// non-finite input takes the neutral default rather than a clamped
    /// extreme.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            melt: finite_clamp(self.melt, defaults.melt, 0.0, 2.0),
            width: finite_clamp(self.width, defaults.width, 0.0, 2.0),
            hold: finite_clamp(self.hold, defaults.hold, 0.0, 1.5),
            swirl: finite_clamp(self.swirl, defaults.swirl, -1.0, 1.0),
            chroma: finite_clamp(self.chroma, defaults.chroma, 0.0, 1.0),
            creep: finite_clamp(self.creep, defaults.creep, 0.0, 1.0),
        }
    }

    /// The probe and drag are live. Width, swirl, chroma, and creep are
    /// dressing on an authored melt and wake nothing alone.
    pub fn is_active(self) -> bool {
        self.sanitized().melt > MIX_AMOUNT_EPSILON
    }

    /// The history dissolve is live, and therefore the history surface may
    /// exist. Both controls must be up: a melt with no hold displaces
    /// without memory, a hold with no melt has no band to fill.
    pub fn is_armed(self) -> bool {
        let clean = self.sanitized();
        clean.melt > MIX_AMOUNT_EPSILON && clean.hold > MIX_AMOUNT_EPSILON
    }
}

/// The complete authored bus-mixer state the composition tree carries: the
/// wipe/blend mix laws, the dirty-mixer fault stage, and the bus melt. One
/// bundle so the saved tree, the runtime tree, and every frame copy carry
/// exactly the same shape.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BusMixerState {
    pub mix: BusMixParams,
    pub dirt: DirtParams,
    pub melt: MeltParams,
}

impl BusMixerState {
    /// Sanitize every sub-block.
    pub fn sanitized(self) -> Self {
        Self {
            mix: self.mix.sanitized(),
            dirt: self.dirt.sanitized(),
            melt: self.melt.sanitized(),
        }
    }

    /// The exact pre-B8 bus law: a plain premultiplied crossfade with no
    /// wipe, no blend, no faults, and no melt.
    pub fn is_exact_legacy_bus(self) -> bool {
        self.mix.is_exact_dissolve() && self.dirt.is_exact_off() && !self.melt.is_active()
    }
}

/// The master melting-edge stage's GPU uniform. Field order mirrors
/// `MeltUniforms` in `melting_edge.wgsl` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeltGpuUniforms {
    pub melt: f32,
    pub width: f32,
    pub hold: f32,
    pub swirl: f32,
    pub chroma: f32,
    pub creep: f32,
    pub hist_valid: u32,
    pub _pad0: f32,
    pub resolution: [f32; 2],
    pub output_aspect: f32,
    pub _pad1: f32,
}

const _: () = assert!(std::mem::size_of::<MeltGpuUniforms>() == 48);

impl MeltGpuUniforms {
    pub fn from_parts(params: MeltParams, dimensions: [u32; 2], history_valid: bool) -> Self {
        let clean = params.sanitized();
        let width = dimensions[0].max(1) as f32;
        let height = dimensions[1].max(1) as f32;
        Self {
            melt: clean.melt,
            width: clean.width,
            hold: clean.hold,
            swirl: clean.swirl,
            chroma: clean.chroma,
            creep: clean.creep,
            hist_valid: u32::from(history_valid && clean.is_armed()),
            _pad0: 0.0,
            resolution: [width, height],
            output_aspect: width / height,
            _pad1: 0.0,
        }
    }
}

/// One validated bus-mixer wire edit. Both wire validators and the applier
/// share this single parser, so the accepted vocabulary and the applied
/// vocabulary are structurally the same table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BusMixerEdit {
    WipePattern(WipePattern),
    WipeSoft(f32),
    WipeX(f32),
    WipeY(f32),
    WipeDetail(f32),
    WipeInvert(bool),
    WipeRep(u32),
    WipeBorder(f32),
    WipeBorderColor(BackColor),
    Blend(BlendMode),
    Dirt(f32),
    DirtRate(f32),
    DirtDrop(f32),
    DirtCut(f32),
    DirtKnock(f32),
    DirtNoise(f32),
    Melt(f32),
    MeltWidth(f32),
    MeltHold(f32),
    MeltSwirl(f32),
    MeltChroma(f32),
    MeltCreep(f32),
}

impl BusMixerEdit {
    /// Parse and range-validate one wire edit. Unknown params, out-of-range
    /// or non-finite numbers, unknown tokens, and the `alpha_cut` blend (a
    /// crossfade has no destination to cut) are all rejections, never
    /// silently repaired values.
    pub fn parse(param: &str, value: &serde_json::Value) -> Option<Self> {
        let number = |min: f32, max: f32| -> Option<f32> {
            let number = value.as_f64()? as f32;
            (number.is_finite() && (min..=max).contains(&number)).then_some(number)
        };
        Some(match param {
            "wipe_pattern" => Self::WipePattern(WipePattern::from_key(value.as_str()?)?),
            "wipe_soft" => Self::WipeSoft(number(0.0, 1.0)?),
            "wipe_x" => Self::WipeX(number(-1.0, 1.0)?),
            "wipe_y" => Self::WipeY(number(-1.0, 1.0)?),
            "wipe_detail" => Self::WipeDetail(number(0.0, 1.0)?),
            "wipe_invert" => Self::WipeInvert(value.as_bool()?),
            "wipe_rep" => {
                let rep = value.as_u64()?;
                if !(1..=4).contains(&rep) {
                    return None;
                }
                Self::WipeRep(rep as u32)
            }
            "wipe_border" => Self::WipeBorder(number(0.0, 1.0)?),
            "wipe_border_color" => Self::WipeBorderColor(BackColor::from_key(value.as_str()?)?),
            "blend" => {
                let blend = BlendMode::from_key(value.as_str()?)?;
                if blend == BlendMode::AlphaCut {
                    return None;
                }
                Self::Blend(blend)
            }
            "dirt" => Self::Dirt(number(0.0, 1.0)?),
            "dirt_rate" => Self::DirtRate(number(0.0, 1.0)?),
            "dirt_drop" => Self::DirtDrop(number(0.0, 1.0)?),
            "dirt_cut" => Self::DirtCut(number(0.0, 1.0)?),
            "dirt_knock" => Self::DirtKnock(number(0.0, 1.0)?),
            "dirt_noise" => Self::DirtNoise(number(0.0, 1.0)?),
            "melt" => Self::Melt(number(0.0, 2.0)?),
            "melt_width" => Self::MeltWidth(number(0.0, 2.0)?),
            "melt_hold" => Self::MeltHold(number(0.0, 1.5)?),
            "melt_swirl" => Self::MeltSwirl(number(-1.0, 1.0)?),
            "melt_chroma" => Self::MeltChroma(number(0.0, 1.0)?),
            "melt_creep" => Self::MeltCreep(number(0.0, 1.0)?),
            _ => return None,
        })
    }

    /// Apply the validated edit to a mixer state.
    pub fn apply(self, state: &mut BusMixerState) {
        match self {
            Self::WipePattern(pattern) => state.mix.pattern = pattern,
            Self::WipeSoft(value) => state.mix.soft = value,
            Self::WipeX(value) => state.mix.origin_x = value,
            Self::WipeY(value) => state.mix.origin_y = value,
            Self::WipeDetail(value) => state.mix.detail = value,
            Self::WipeInvert(value) => state.mix.invert = value,
            Self::WipeRep(value) => state.mix.rep = value,
            Self::WipeBorder(value) => state.mix.border = value,
            Self::WipeBorderColor(color) => state.mix.border_color = color,
            Self::Blend(blend) => state.mix.blend = blend,
            Self::Dirt(value) => state.dirt.dirt = value,
            Self::DirtRate(value) => state.dirt.rate = value,
            Self::DirtDrop(value) => state.dirt.drop = value,
            Self::DirtCut(value) => state.dirt.cut = value,
            Self::DirtKnock(value) => state.dirt.knock = value,
            Self::DirtNoise(value) => state.dirt.noise = value,
            Self::Melt(value) => state.melt.melt = value,
            Self::MeltWidth(value) => state.melt.width = value,
            Self::MeltHold(value) => state.melt.hold = value,
            Self::MeltSwirl(value) => state.melt.swirl = value,
            Self::MeltChroma(value) => state.melt.chroma = value,
            Self::MeltCreep(value) => state.melt.creep = value,
        }
        *state = state.sanitized();
    }
}

fn finite_clamp(value: f32, neutral: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        neutral
    }
}

// ---------------------------------------------------------------------------
// Deterministic hashing: the Shift band/epoch/seed law on the integer
// avalanche, in fresh per-lane domains. The WGSL mirrors these expressions
// exactly; nothing here consumes sequential RNG state, so any draw
// recomputes alone and one lane can never perturb another.
// ---------------------------------------------------------------------------

/// The shared integer avalanche (`cellular_avalanche` in `effects.wgsl`).
pub fn avalanche(value: u32) -> u32 {
    let mut x = value;
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// Top 24 bits as an exact unit float, the established deterministic path.
pub fn hash_unit(value: u32) -> f32 {
    (value >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// One keyed lane draw. `avalanche(0) == 0`, so a zero seed is naturally the
/// unseeded stream without a branch.
pub fn lane_hash(a: u32, b: u32, lane: u32, seed: u32) -> u32 {
    let mixed = a ^ b.wrapping_mul(0x9e37_79b9) ^ lane ^ avalanche(seed);
    avalanche(avalanche(mixed) ^ 0x68bc_21eb)
}

/// [`lane_hash`] as a unit float.
pub fn lane_unit(a: u32, b: u32, lane: u32, seed: u32) -> f32 {
    hash_unit(lane_hash(a, b, lane, seed))
}

/// Fresh hash-lane domains for the dirty mixer and the Blocks wipe. Each
/// draw site owns exactly one constant.
pub const LANE_WIPE_BLOCKS: u32 = 0x4d58_5701; // "MXW" 1
pub const LANE_DIRT_FIRE: u32 = 0x4d58_4401; // "MXD" 1
pub const LANE_DIRT_KNOCK_FRAME: u32 = 0x4d58_4402;
pub const LANE_DIRT_KNOCK_ROW: u32 = 0x4d58_4403;
pub const LANE_DIRT_KNOCK_VERTICAL: u32 = 0x4d58_4404;
pub const LANE_DIRT_CUT: u32 = 0x4d58_4405;
pub const LANE_DIRT_DROP_HEIGHT: u32 = 0x4d58_4406;
pub const LANE_DIRT_DROP_BAND: u32 = 0x4d58_4407;
pub const LANE_DIRT_DROP_SIDE: u32 = 0x4d58_4408;
pub const LANE_DIRT_DROP_SKEW: u32 = 0x4d58_4409;
pub const LANE_DIRT_DROP_DEAD: u32 = 0x4d58_440a;
pub const LANE_DIRT_DROP_CELL: u32 = 0x4d58_440b;
pub const LANE_DIRT_NOISE: u32 = 0x4d58_440c;

// ---------------------------------------------------------------------------
// The wipe field law.
// ---------------------------------------------------------------------------

/// The analytic wipe field in [0, 1]. `uv` is output UV, `aspect` is
/// width/height. Only `Circle` is aspect-corrected — the shipped law.
/// `Dissolve` has no field; callers use the fader directly.
pub fn wipe_field(uv: [f32; 2], params: BusMixParams, aspect: f32, seed: u32) -> f32 {
    let clean = params.sanitized();
    // MULTI tiles the whole pattern; the origin offset applies inside the
    // tile, so every tile gets the same displaced origin.
    let rep = clean.rep as f32;
    let tu = if clean.rep >= 2 {
        [fract(uv[0] * rep), fract(uv[1] * rep)]
    } else {
        uv
    };
    let off = [clean.origin_x * 0.5, clean.origin_y * 0.5];
    let c = [tu[0] - 0.5 - off[0], tu[1] - 0.5 - off[1]];
    // Normalised against the distance to the farthest corner from wherever
    // the origin has been moved to, so the fader travels evenly to full
    // coverage.
    let far = [off[0].abs() + 0.5, off[1].abs() + 0.5];
    let n = 2.0 + (clean.detail * 14.0).floor();
    match clean.pattern {
        WipePattern::Dissolve => 0.0,
        WipePattern::WipeH => tu[0],
        WipePattern::WipeV => 1.0 - tu[1],
        WipePattern::Diagonal => (tu[0] + (1.0 - tu[1])) * 0.5,
        WipePattern::Box => (c[0].abs() / far[0]).max(c[1].abs() / far[1]),
        WipePattern::Circle => {
            let scaled = [c[0] * aspect, c[1]];
            let far_scaled = [far[0] * aspect, far[1]];
            length(scaled) / length(far_scaled).max(0.0001)
        }
        WipePattern::SplitH => c[0].abs() / far[0],
        WipePattern::SplitV => c[1].abs() / far[1],
        WipePattern::BlindsV => fract(tu[0] * n),
        WipePattern::BlindsH => fract(tu[1] * n),
        WipePattern::Clock => {
            let angle = c[1].atan2(c[0]) / std::f32::consts::TAU + 0.5;
            fract(angle)
        }
        WipePattern::DiagBars => fract((tu[0] + tu[1]) * n * 0.5),
        WipePattern::Blocks => {
            let cell_x = (tu[0] * n * 2.0).floor().max(0.0) as u32;
            let cell_y = (tu[1] * n).floor().max(0.0) as u32;
            lane_unit(cell_x, cell_y, LANE_WIPE_BLOCKS, seed)
        }
    }
}

/// The feathered fader remap: `sw` is the half-width floor, `tt` the
/// remapped fader that still reaches full A at 0 and full B at 1.
pub fn wipe_shaping(soft: f32, fader: f32) -> (f32, f32) {
    let sw = (soft * 0.5).max(0.002);
    let tt = fader * (1.0 + 2.0 * sw) - sw;
    (sw, tt)
}

/// The complete mix matte at one point: field, invert, feathered threshold.
/// `Dissolve` is the constant fader — no field, no gradient, no band.
pub fn mix_matte(uv: [f32; 2], params: BusMixParams, fader: f32, aspect: f32, seed: u32) -> f32 {
    let clean = params.sanitized();
    let t = fader.clamp(0.0, 1.0);
    if clean.pattern == WipePattern::Dissolve {
        return t;
    }
    let mut d = wipe_field(uv, clean, aspect, seed);
    if clean.invert {
        d = 1.0 - d;
    }
    let (sw, tt) = wipe_shaping(clean.soft, t);
    smoothstep(d - sw, d + sw, tt)
}

/// The border rule band: a flat core to 45% of the half-width, feathered to
/// the edge, gated off within 0.4% of either fader end.
pub fn wipe_border_band(d: f32, tt: f32, fader: f32, border: f32) -> f32 {
    if border <= MIX_AMOUNT_EPSILON {
        return 0.0;
    }
    let bw = 0.004 + border * 0.1;
    let profile = 1.0 - smoothstep(bw * 0.45, bw, (d - tt).abs());
    let gate = if (0.004..=0.996).contains(&fader) {
        1.0
    } else {
        0.0
    };
    profile * gate
}

// ---------------------------------------------------------------------------
// The dirty-mixer laws. Everything is a pure function of the tick index,
// the pixel address, the authored amounts, and the master random seed —
// bit-clean between firings because a non-firing tick zeroes the envelope
// exactly and every fault is gated on it.
// ---------------------------------------------------------------------------

/// One evaluated event-clock state: the tick index, the position within the
/// tick, and the decaying envelope (exactly zero on a non-firing tick).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirtClock {
    pub tick: u32,
    pub fraction: f32,
    pub envelope: f32,
}

/// Evaluate the event clock at a program time. The rate law is
/// 0.5 + rate * 15 ticks per second; the envelope decays inside its own
/// tick, fast at low dirt (a flick) and slow at high dirt (the whole tick).
pub fn dirt_clock(params: DirtParams, time_seconds: f32, seed: u32) -> DirtClock {
    let clean = params.sanitized();
    let rate = 0.5 + clean.rate * 15.0;
    let phase = time_seconds.max(0.0) * rate;
    let tick = phase.floor().min(u32::MAX as f32) as u32;
    let fraction = fract(phase);
    if clean.dirt <= MIX_AMOUNT_EPSILON {
        return DirtClock {
            tick,
            fraction,
            envelope: 0.0,
        };
    }
    let fires = lane_unit(tick, 0, LANE_DIRT_FIRE, seed) <= clean.dirt * 0.85;
    let envelope = if fires {
        let decay = 11.0 + (1.6 - 11.0) * clean.dirt;
        (-fraction * decay).exp()
    } else {
        0.0
    };
    DirtClock {
        tick,
        fraction,
        envelope,
    }
}

/// The knock: a horizontal shove that shears down the frame plus a per-line
/// jitter, and a wrapped vertical hop. Returns the displaced UV; below the
/// engagement threshold it returns the input exactly.
pub fn dirt_knock_uv(
    uv: [f32; 2],
    resolution_y: f32,
    clock: DirtClock,
    params: DirtParams,
    seed: u32,
) -> [f32; 2] {
    let clean = params.sanitized();
    let kn = clock.envelope * clean.knock;
    if kn <= 0.0005 {
        return uv;
    }
    let row = (uv[1] * resolution_y.max(1.0)).floor().max(0.0) as u32;
    let mut shove = (lane_unit(clock.tick, 0, LANE_DIRT_KNOCK_FRAME, seed) - 0.5) * 0.16 * kn;
    shove *= 0.4 + 0.6 * (1.0 - uv[1]);
    shove += (lane_unit(row, clock.tick, LANE_DIRT_KNOCK_ROW, seed) - 0.5) * 0.05 * kn;
    let hop = (lane_unit(clock.tick, 0, LANE_DIRT_KNOCK_VERTICAL, seed) - 0.5) * 0.06 * kn;
    [uv[0] + shove, fract(uv[1] + hop)]
}

/// The cut: a firing can throw the crossbar to the wrong input for a
/// moment. Returns the overridden matte.
pub fn dirt_cut_matte(matte: f32, clock: DirtClock, params: DirtParams, seed: u32) -> f32 {
    let clean = params.sanitized();
    if clock.envelope <= 0.001 || clean.cut <= MIX_AMOUNT_EPSILON {
        return matte;
    }
    let want = if lane_unit(clock.tick, 0, LANE_DIRT_CUT, seed) >= 0.5 {
        1.0
    } else {
        0.0
    };
    let strength = (clock.envelope * clean.cut * 1.4).clamp(0.0, 1.0);
    matte + (want - matte) * strength
}

/// One line band's dropout decision. `None` means the band survives; a
/// dropped band names its replacement: the other side of the crossbar
/// (`to_b`), a horizontal skew for the replacement tap, whether it is dead
/// grey hash instead, and the mix-in strength.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DropoutBand {
    pub to_b: bool,
    pub skew: f32,
    pub dead: bool,
    pub strength: f32,
}

/// Evaluate the dropout law for the band containing `y_pixels`. Band height
/// is constant across the whole tick; the probability is modulated by the
/// envelope but the mix-in strength deliberately is not — BENDR's law.
pub fn dirt_dropout(
    y_pixels: f32,
    clock: DirtClock,
    params: DirtParams,
    seed: u32,
) -> Option<DropoutBand> {
    let clean = params.sanitized();
    if clock.envelope <= 0.001 || clean.drop <= MIX_AMOUNT_EPSILON {
        return None;
    }
    let band_height = 2.0 + 26.0 * lane_unit(clock.tick, 0, LANE_DIRT_DROP_HEIGHT, seed);
    let band = (y_pixels.max(0.0) / band_height).floor() as u32;
    let probability = (clean.drop * clock.envelope * 1.3).clamp(0.0, 0.95);
    let draw = lane_unit(band, clock.tick, LANE_DIRT_DROP_BAND, seed);
    if draw < 1.0 - probability {
        return None;
    }
    Some(DropoutBand {
        to_b: lane_unit(band, clock.tick, LANE_DIRT_DROP_SIDE, seed) >= 0.5,
        skew: (lane_unit(band, clock.tick, LANE_DIRT_DROP_SKEW, seed) - 0.5) * 0.09,
        dead: lane_unit(band, clock.tick, LANE_DIRT_DROP_DEAD, seed) >= 0.82,
        strength: (clean.drop * 1.2).clamp(0.0, 1.0),
    })
}

/// The dead-band grey hash value at one pixel: 2.5-pixel horizontal cells,
/// per-line, ceiling one half.
pub fn dirt_dead_value(x_pixels: f32, band: u32, clock: DirtClock, seed: u32) -> f32 {
    let cell = (x_pixels.max(0.0) / 2.5).floor() as u32;
    lane_unit(
        cell,
        band ^ clock.tick.wrapping_mul(0x85eb_ca6b),
        LANE_DIRT_DROP_CELL,
        seed,
    ) * 0.5
}

/// The band index the dropout and dead laws share for one tick.
pub fn dirt_dropout_band_index(y_pixels: f32, clock: DirtClock, seed: u32) -> u32 {
    let band_height = 2.0 + 26.0 * lane_unit(clock.tick, 0, LANE_DIRT_DROP_HEIGHT, seed);
    (y_pixels.max(0.0) / band_height).floor() as u32
}

/// The switching transient at one pixel: a signed monochrome offset from
/// 3x1-pixel cells (video-noise band limiting) and the desaturation factor
/// toward Rec.709 luma. Both scale with the envelope.
pub fn dirt_noise(
    x_pixels: f32,
    y_pixels: f32,
    clock: DirtClock,
    params: DirtParams,
    seed: u32,
) -> (f32, f32) {
    let clean = params.sanitized();
    if clock.envelope <= 0.001 || clean.noise <= MIX_AMOUNT_EPSILON {
        return (0.0, 0.0);
    }
    let cell_x = (x_pixels.max(0.0) / 3.0).floor() as u32;
    let row = y_pixels.max(0.0).floor() as u32;
    let draw = lane_unit(
        cell_x ^ row.wrapping_mul(0x9e37_79b9),
        clock.tick,
        LANE_DIRT_NOISE,
        seed,
    ) - 0.5;
    (
        draw * clean.noise * clock.envelope * 1.6,
        (clean.noise * clock.envelope * 0.5).clamp(0.0, 1.0),
    )
}

// ---------------------------------------------------------------------------
// The melting edge.
// ---------------------------------------------------------------------------

/// The four-point probe radius in UV: r = 0.004 + width * 0.085. The X
/// probe offset is aspect-corrected; the resulting normal deliberately is
/// not — the shipped anisotropy is the law.
pub fn melt_probe_radius(width: f32) -> f32 {
    0.004 + width.clamp(0.0, 2.0) * 0.085
}

/// The band and edge normal from four matte probes (left, right, down, up).
/// Band is disagreement gained 1.25 and clipped; the normal is the raw
/// central difference, normalized, zero when degenerate.
pub fn melt_band_and_normal(probes: [f32; 4], swirl: f32) -> (f32, [f32; 2]) {
    let [left, right, down, up] = probes;
    let low = left.min(right).min(down.min(up));
    let high = left.max(right).max(down.max(up));
    let band = ((high - low) * 1.25).clamp(0.0, 1.0);
    let g = [right - left, up - down];
    let len = length(g);
    if len <= 1.0e-5 {
        return (band, [0.0, 0.0]);
    }
    let mut normal = [g[0] / len, g[1] / len];
    let angle = swirl.clamp(-1.0, 1.0) * std::f32::consts::FRAC_PI_2;
    let (sin, cos) = angle.sin_cos();
    normal = [
        cos * normal[0] - sin * normal[1],
        sin * normal[0] + cos * normal[1],
    ];
    (band, normal)
}

/// Creep pushes the melt onto the outgoing side, so the shape bleeds into
/// the background rather than the background eating into the shape.
pub fn melt_creep_band(band: f32, matte: f32, creep: f32) -> f32 {
    let toward = 1.0 - matte.clamp(0.0, 1.0);
    band * (1.0 + creep.clamp(0.0, 1.0) * (toward - 1.0))
}

/// The incoming-picture drag along the normal.
pub fn melt_drag(normal: [f32; 2], band: f32, melt: f32) -> [f32; 2] {
    [
        normal[0] * band * melt * 0.055,
        normal[1] * band * melt * 0.055,
    ]
}

/// The per-store creep offset for the history tap. Compounded through the
/// history surface, this is what makes the smear creep a little further out
/// every reference tick.
pub fn melt_history_offset(normal: [f32; 2], melt: f32) -> [f32; 2] {
    let push = 0.0015 + melt * 0.04;
    [normal[0] * push, normal[1] * push]
}

/// The ceiling on how much of the last frame can survive. Unity is the old
/// limit and still reads as a trail that settles; the extra travel past it
/// stops the band settling at all.
pub fn melt_hold_cap(hold: f32) -> f32 {
    (0.94 + (hold - 1.0).max(0.0) * 0.11).min(0.995)
}

/// The history dissolve amount inside the band.
pub fn melt_hold_mix(band: f32, hold: f32) -> f32 {
    (band * hold).clamp(0.0, melt_hold_cap(hold))
}

/// Coherent 601 YIQ forward — the B3 feedback-rig round trip, reused
/// because the chroma law reconstructs RGB and a mixed-standard inverse
/// would not.
pub fn rgb_to_yiq(rgb: [f32; 3]) -> [f32; 3] {
    [
        0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2],
        0.596 * rgb[0] - 0.274 * rgb[1] - 0.322 * rgb[2],
        0.211 * rgb[0] - 0.523 * rgb[1] + 0.312 * rgb[2],
    ]
}

/// Coherent 601 YIQ inverse.
pub fn yiq_to_rgb(yiq: [f32; 3]) -> [f32; 3] {
    let [y, i, q] = yiq;
    [
        y + 0.956 * i + 0.621 * q,
        y - 0.272 * i - 0.647 * q,
        y - 1.106 * i + 1.703 * q,
    ]
}

/// The chroma-runs-further law: keep the near tap's luma, crossfade the
/// chroma pair toward the far tap.
pub fn melt_chroma_mix(near: [f32; 3], far: [f32; 3], chroma: f32) -> [f32; 3] {
    let a = rgb_to_yiq(near);
    let b = rgb_to_yiq(far);
    let t = chroma.clamp(0.0, 1.0);
    yiq_to_rgb([a[0], a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t])
}

// ---------------------------------------------------------------------------
// Small shared math, mirrored by the WGSL expression for expression.
// ---------------------------------------------------------------------------

fn fract(value: f32) -> f32 {
    value - value.floor()
}

fn length(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wipe_pattern_codes_and_keys_are_append_only() {
        let keys = [
            "dissolve",
            "wipe_h",
            "wipe_v",
            "diagonal",
            "box",
            "circle",
            "split_h",
            "split_v",
            "blinds_v",
            "blinds_h",
            "clock",
            "diag_bars",
            "blocks",
        ];
        assert_eq!(keys.len(), WipePattern::ALL.len());
        for (code, (pattern, key)) in WipePattern::ALL.into_iter().zip(keys).enumerate() {
            assert_eq!(pattern.code(), code as u32);
            assert_eq!(pattern.key(), key);
            assert_eq!(WipePattern::from_key(key), Some(pattern));
            let json = serde_json::to_string(&pattern).unwrap();
            assert_eq!(json, format!("\"{key}\""));
        }
        assert_eq!(WipePattern::from_key("melt"), None);
        assert_eq!(WipePattern::from_key("slide_left"), None);
    }

    #[test]
    fn back_color_table_is_the_frozen_bench_order() {
        let expected = [
            ("white", [1.0, 1.0, 1.0]),
            ("yellow", [1.0, 1.0, 0.0]),
            ("cyan", [0.0, 1.0, 1.0]),
            ("green", [0.0, 1.0, 0.0]),
            ("magenta", [1.0, 0.0, 1.0]),
            ("red", [1.0, 0.0, 0.0]),
            ("blue", [0.0, 0.0, 1.0]),
            ("black", [0.0, 0.0, 0.0]),
        ];
        assert_eq!(expected.len(), BackColor::ALL.len());
        for (code, (color, (key, rgb))) in BackColor::ALL.into_iter().zip(expected).enumerate() {
            assert_eq!(color.code(), code as u32);
            assert_eq!(color.key(), key);
            assert_eq!(BackColor::from_key(key), Some(color));
            assert_eq!(color.rgb(), rgb);
        }
    }

    #[test]
    fn hostile_params_sanitize_to_neutral_values_not_clamped_extremes() {
        let hostile = BusMixParams {
            soft: f32::NAN,
            origin_x: f32::INFINITY,
            origin_y: f32::NEG_INFINITY,
            detail: f32::NAN,
            rep: 99,
            border: f32::NAN,
            blend: BlendMode::AlphaCut,
            ..BusMixParams::default()
        };
        let clean = hostile.sanitized();
        let defaults = BusMixParams::default();
        assert_eq!(clean.soft, defaults.soft);
        assert_eq!(clean.origin_x, defaults.origin_x, "infinity is non-finite");
        assert_eq!(clean.origin_y, defaults.origin_y);
        assert_eq!(clean.detail, defaults.detail);
        assert_eq!(clean.rep, 4);
        assert_eq!(clean.border, defaults.border);
        assert_eq!(clean.blend, BlendMode::Normal, "AlphaCut is not a bus law");
        // A finite out-of-range value does clamp.
        let overflow = BusMixParams {
            origin_x: 5.0,
            soft: -3.0,
            ..BusMixParams::default()
        }
        .sanitized();
        assert_eq!(overflow.origin_x, 1.0);
        assert_eq!(overflow.soft, 0.0);

        let dirt = DirtParams {
            dirt: f32::NAN,
            rate: f32::INFINITY,
            ..DirtParams::default()
        }
        .sanitized();
        assert_eq!(dirt.dirt, 0.0);
        assert_eq!(
            dirt.rate,
            DirtParams::default().rate,
            "neutral, not extreme"
        );

        let melt = MeltParams {
            melt: f32::NAN,
            width: f32::NEG_INFINITY,
            hold: f32::NAN,
            ..MeltParams::default()
        }
        .sanitized();
        assert_eq!(melt.melt, 0.0, "hostile melt is neutral off, not full");
        assert_eq!(melt.width, MeltParams::default().width);
        assert_eq!(melt.hold, MeltParams::default().hold);
    }

    #[test]
    fn wake_laws_ignore_dressing_controls() {
        // Bus mix: softness/origin/detail/invert/rep are dressing.
        let dressed = BusMixParams {
            soft: 0.9,
            origin_x: 0.7,
            detail: 1.0,
            invert: true,
            rep: 4,
            ..BusMixParams::default()
        };
        assert!(dressed.is_exact_dissolve());
        assert!(!BusMixParams {
            pattern: WipePattern::WipeH,
            ..BusMixParams::default()
        }
        .is_exact_dissolve());
        assert!(!BusMixParams {
            border: 0.5,
            ..BusMixParams::default()
        }
        .is_exact_dissolve());
        assert!(!BusMixParams {
            blend: BlendMode::Screen,
            ..BusMixParams::default()
        }
        .is_exact_dissolve());

        // Dirt: the four flavours wake nothing alone.
        assert!(DirtParams {
            drop: 1.0,
            cut: 1.0,
            knock: 1.0,
            noise: 1.0,
            ..DirtParams::default()
        }
        .is_exact_off());
        assert!(!DirtParams {
            dirt: 0.5,
            ..DirtParams::default()
        }
        .is_exact_off());

        // Melt: width/swirl/chroma/creep are dressing; hold alone has no
        // band to fill; melt alone displaces but arms no history.
        assert!(!MeltParams {
            width: 2.0,
            swirl: 1.0,
            chroma: 1.0,
            creep: 1.0,
            ..MeltParams::default()
        }
        .is_active());
        let melt_only = MeltParams {
            melt: 1.0,
            hold: 0.0,
            ..MeltParams::default()
        };
        assert!(melt_only.is_active());
        assert!(!melt_only.is_armed());
        assert!(MeltParams {
            melt: 1.0,
            ..MeltParams::default()
        }
        .is_armed());
    }

    #[test]
    fn wipe_fields_hit_their_analytic_landmarks() {
        let base = BusMixParams::default();
        let at = |pattern: WipePattern, uv: [f32; 2]| {
            wipe_field(uv, BusMixParams { pattern, ..base }, 16.0 / 9.0, 0)
        };
        // Horizontal wipe is the x coordinate; vertical runs top-down.
        assert_eq!(at(WipePattern::WipeH, [0.25, 0.9]), 0.25);
        assert!((at(WipePattern::WipeV, [0.9, 0.25]) - 0.75).abs() <= 1.0e-6);
        // Diagonal averages the two ramps.
        assert!((at(WipePattern::Diagonal, [1.0, 1.0]) - 0.5).abs() <= 1.0e-6);
        // Box and circle are 0 at the centre and 1 at the far corner.
        assert_eq!(at(WipePattern::Box, [0.5, 0.5]), 0.0);
        assert!((at(WipePattern::Box, [0.0, 0.0]) - 1.0).abs() <= 1.0e-6);
        assert_eq!(at(WipePattern::Circle, [0.5, 0.5]), 0.0);
        assert!((at(WipePattern::Circle, [1.0, 1.0]) - 1.0).abs() <= 1.0e-4);
        // A circle at equal UV distance is aspect-weighted: the horizontal
        // arm reads further along the field than the vertical arm on 16:9.
        let horizontal = at(WipePattern::Circle, [0.75, 0.5]);
        let vertical = at(WipePattern::Circle, [0.5, 0.75]);
        assert!(horizontal > vertical);
        // Splits fold about the centre.
        assert!(
            (at(WipePattern::SplitH, [0.25, 0.5]) - at(WipePattern::SplitH, [0.75, 0.5])).abs()
                <= 1.0e-6
        );
        // Blinds repeat: detail 0.3 gives n = 2 + floor(4.2) = 6 stripes.
        assert!((at(WipePattern::BlindsV, [0.25, 0.5]) - 0.5).abs() <= 1.0e-5);
        // Clock wraps a full turn.
        assert!((at(WipePattern::Clock, [1.0, 0.5]) - 0.5).abs() <= 1.0e-6);
        // Blocks is deterministic per cell and changes with the seed.
        let block_a = at(WipePattern::Blocks, [0.1, 0.1]);
        assert_eq!(block_a, at(WipePattern::Blocks, [0.1, 0.1]));
        let seeded = wipe_field(
            [0.1, 0.1],
            BusMixParams {
                pattern: WipePattern::Blocks,
                ..base
            },
            16.0 / 9.0,
            7,
        );
        assert_ne!(block_a, seeded);
    }

    #[test]
    fn multi_tiling_repeats_the_field_with_the_origin_inside_each_tile() {
        let params = BusMixParams {
            pattern: WipePattern::Box,
            rep: 2,
            origin_x: 0.4,
            ..BusMixParams::default()
        };
        let a = wipe_field([0.1, 0.2], params, 1.0, 0);
        let b = wipe_field([0.6, 0.7], params, 1.0, 0);
        assert!((a - b).abs() <= 1.0e-6, "tiles repeat exactly");
    }

    #[test]
    fn mix_matte_reaches_exact_endpoints_despite_the_feather() {
        for pattern in WipePattern::ALL {
            let params = BusMixParams {
                pattern,
                soft: 0.8,
                ..BusMixParams::default()
            };
            for x in 0..8 {
                for y in 0..8 {
                    let uv = [x as f32 / 7.0, y as f32 / 7.0];
                    assert_eq!(
                        mix_matte(uv, params, 0.0, 16.0 / 9.0, 3),
                        0.0,
                        "{pattern:?} full A at fader 0"
                    );
                    assert_eq!(
                        mix_matte(uv, params, 1.0, 16.0 / 9.0, 3),
                        1.0,
                        "{pattern:?} full B at fader 1"
                    );
                }
            }
        }
    }

    #[test]
    fn dissolve_matte_is_the_fader_everywhere() {
        let params = BusMixParams::default();
        for x in 0..5 {
            let uv = [x as f32 / 4.0, 0.3];
            assert_eq!(mix_matte(uv, params, 0.37, 1.0, 0), 0.37);
        }
    }

    #[test]
    fn border_band_is_flat_cored_feathered_and_end_gated() {
        // On the join the profile is full.
        assert_eq!(wipe_border_band(0.5, 0.5, 0.5, 0.5), 1.0);
        // Inside 45% of the half-width still full; outside the half-width
        // zero.
        let bw = 0.004 + 0.5 * 0.1;
        assert_eq!(wipe_border_band(0.5 + bw * 0.44, 0.5, 0.5, 0.5), 1.0);
        assert_eq!(wipe_border_band(0.5 + bw * 1.01, 0.5, 0.5, 0.5), 0.0);
        // Gated off at the fader ends.
        assert_eq!(wipe_border_band(0.5, 0.5, 0.0, 0.5), 0.0);
        assert_eq!(wipe_border_band(0.5, 0.5, 1.0, 0.5), 0.0);
        // Border off is exactly zero.
        assert_eq!(wipe_border_band(0.5, 0.5, 0.5, 0.0), 0.0);
    }

    #[test]
    fn dirt_clock_is_bit_clean_between_firings_and_off_at_zero() {
        let params = DirtParams {
            dirt: 0.4,
            ..DirtParams::default()
        };
        // Find a non-firing tick: its envelope must be exactly zero.
        let mut saw_firing = false;
        let mut saw_quiet = false;
        for tick in 0..64 {
            let rate = 0.5 + params.rate * 15.0;
            let time = (tick as f32 + 0.25) / rate;
            let clock = dirt_clock(params, time, 0);
            assert_eq!(clock.tick, tick);
            if clock.envelope == 0.0 {
                saw_quiet = true;
            } else {
                saw_firing = true;
                assert!(clock.envelope > 0.0 && clock.envelope <= 1.0);
            }
        }
        assert!(saw_firing, "a 0.4 dirt clock fires within 64 ticks");
        assert!(saw_quiet, "a 0.4 dirt clock stays quiet within 64 ticks");

        // Dirt zero: envelope exactly zero at every time, and every fault
        // law returns its exact identity.
        let off = DirtParams::default();
        let clock = dirt_clock(off, 12.34, 9);
        assert_eq!(clock.envelope, 0.0);
        assert_eq!(dirt_knock_uv([0.3, 0.7], 1080.0, clock, off, 9), [0.3, 0.7]);
        assert_eq!(dirt_cut_matte(0.42, clock, off, 9), 0.42);
        assert_eq!(dirt_dropout(371.0, clock, off, 9), None);
        assert_eq!(dirt_noise(100.0, 200.0, clock, off, 9), (0.0, 0.0));
    }

    #[test]
    fn dirt_envelope_decays_inside_its_own_tick() {
        let params = DirtParams {
            dirt: 1.0,
            rate: 0.0,
            ..DirtParams::default()
        };
        // rate 0 -> 0.5 ticks/second. Find a firing tick.
        let mut firing_tick = None;
        for tick in 0..64u32 {
            if lane_unit(tick, 0, LANE_DIRT_FIRE, 0) <= 0.85 {
                firing_tick = Some(tick);
                break;
            }
        }
        let tick = firing_tick.expect("0.85 probability fires quickly");
        let early = dirt_clock(params, (tick as f32 + 0.1) * 2.0, 0);
        let late = dirt_clock(params, (tick as f32 + 0.9) * 2.0, 0);
        assert!(early.envelope > late.envelope);
        // The analytic decay constant at dirt 1 is 1.6.
        assert!((early.envelope - (-0.1_f32 * 1.6).exp()).abs() <= 1.0e-5);
        assert!((late.envelope - (-0.9_f32 * 1.6).exp()).abs() <= 1.0e-5);
    }

    #[test]
    fn dirt_faults_are_deterministic_per_seed_and_distinct_across_seeds() {
        let params = DirtParams {
            dirt: 1.0,
            ..DirtParams::default()
        };
        let clock = dirt_clock(params, 5.0, 11);
        let again = dirt_clock(params, 5.0, 11);
        assert_eq!(clock, again);
        let knock_a = dirt_knock_uv([0.4, 0.6], 1080.0, clock, params, 11);
        assert_eq!(
            knock_a,
            dirt_knock_uv([0.4, 0.6], 1080.0, clock, params, 11)
        );
        // A different master seed produces a different fault stream
        // somewhere within a few ticks.
        let mut differs = false;
        for tick in 0..32 {
            let time = (tick as f32 + 0.2) / (0.5 + params.rate * 15.0);
            let a = dirt_clock(params, time, 1);
            let b = dirt_clock(params, time, 2);
            if a.envelope != b.envelope {
                differs = true;
                break;
            }
        }
        assert!(differs, "the master seed enters the fire lane");
    }

    #[test]
    fn dropout_probability_gate_is_honest_over_many_bands() {
        let params = DirtParams {
            dirt: 1.0,
            drop: 0.6,
            ..DirtParams::default()
        };
        // A firing clock with a known envelope.
        let mut clock = None;
        for tick in 0..64u32 {
            if lane_unit(tick, 0, LANE_DIRT_FIRE, 0) <= 0.85 {
                let rate = 0.5 + params.rate * 15.0;
                clock = Some(dirt_clock(params, (tick as f32 + 0.05) / rate, 0));
                break;
            }
        }
        let clock = clock.unwrap();
        assert!(clock.envelope > 0.0);
        let probability = (params.drop * clock.envelope * 1.3).clamp(0.0, 0.95);
        let mut dropped = 0u32;
        let total = 4096u32;
        for band in 0..total {
            let band_height = 2.0 + 26.0 * lane_unit(clock.tick, 0, LANE_DIRT_DROP_HEIGHT, 0);
            let y = band as f32 * band_height + 0.5;
            if dirt_dropout(y, clock, params, 0).is_some() {
                dropped += 1;
            }
        }
        let measured = dropped as f32 / total as f32;
        assert!(
            (measured - probability).abs() < 0.03,
            "measured {measured} expected {probability}"
        );
        // Strength is not envelope-scaled: it is the authored amount alone.
        for band in 0..total {
            let band_height = 2.0 + 26.0 * lane_unit(clock.tick, 0, LANE_DIRT_DROP_HEIGHT, 0);
            let y = band as f32 * band_height + 0.5;
            if let Some(dropout) = dirt_dropout(y, clock, params, 0) {
                assert_eq!(dropout.strength, (0.6_f32 * 1.2).clamp(0.0, 1.0));
                assert!(dropout.skew.abs() <= 0.045 + 1.0e-6);
                break;
            }
        }
    }

    #[test]
    fn melt_band_and_normal_recover_an_analytic_vertical_edge() {
        // A vertical matte edge: left probes low, right probes high. The
        // disagreement direction is +X — a horizontal normal.
        let (band, normal) = melt_band_and_normal([0.0, 1.0, 0.5, 0.5], 0.0);
        assert_eq!(band, 1.0, "full disagreement saturates the band");
        assert!((normal[0] - 1.0).abs() <= 1.0e-6);
        assert!(normal[1].abs() <= 1.0e-6);

        // Band width law: disagreement 0.4 gains 1.25 to 0.5.
        let (band, _) = melt_band_and_normal([0.3, 0.7, 0.5, 0.5], 0.0);
        assert!((band - 0.5).abs() <= 1.0e-6);

        // No disagreement: no band, no normal — a plain dissolve has no
        // boundary, so nothing happens.
        let (band, normal) = melt_band_and_normal([0.4, 0.4, 0.4, 0.4], 0.0);
        assert_eq!(band, 0.0);
        assert_eq!(normal, [0.0, 0.0]);

        // Swirl rotates the normal a quarter turn at full: across the edge
        // becomes along it.
        let (_, swirled) = melt_band_and_normal([0.0, 1.0, 0.5, 0.5], 1.0);
        assert!(swirled[0].abs() <= 1.0e-6);
        assert!((swirled[1] - 1.0).abs() <= 1.0e-6);
        let (_, counter) = melt_band_and_normal([0.0, 1.0, 0.5, 0.5], -1.0);
        assert!((counter[1] + 1.0).abs() <= 1.0e-6);
    }

    #[test]
    fn melt_probe_radius_and_drag_follow_the_transcribed_constants() {
        assert!((melt_probe_radius(0.0) - 0.004).abs() <= 1.0e-7);
        assert!((melt_probe_radius(1.0) - 0.089).abs() <= 1.0e-7);
        let drag = melt_drag([1.0, 0.0], 0.5, 2.0);
        assert!((drag[0] - 0.055).abs() <= 1.0e-6);
        assert_eq!(drag[1], 0.0);
        let push = melt_history_offset([0.0, 1.0], 1.0);
        assert!((push[1] - 0.0415).abs() <= 1.0e-6);
    }

    #[test]
    fn melt_creep_selects_the_outgoing_side() {
        // Full creep: the band survives only where the matte is low.
        assert_eq!(melt_creep_band(1.0, 1.0, 1.0), 0.0);
        assert_eq!(melt_creep_band(1.0, 0.0, 1.0), 1.0);
        // No creep: both sides melt.
        assert_eq!(melt_creep_band(1.0, 1.0, 0.0), 1.0);
        // Half creep at the seam midpoint: three quarters.
        assert!((melt_creep_band(1.0, 0.5, 0.5) - 0.75).abs() <= 1.0e-6);
    }

    #[test]
    fn melt_hold_cap_settles_at_unity_and_builds_past_it() {
        assert!((melt_hold_cap(0.6) - 0.94).abs() <= 1.0e-6);
        assert!((melt_hold_cap(1.0) - 0.94).abs() <= 1.0e-6);
        assert!((melt_hold_cap(1.5) - 0.995).abs() <= 1.0e-6);
        // The mix is band-scaled and capped.
        assert!((melt_hold_mix(0.5, 0.6) - 0.3).abs() <= 1.0e-6);
        assert_eq!(melt_hold_mix(1.0, 1.5), 0.995);
    }

    #[test]
    fn melt_chroma_keeps_near_luma_and_walks_the_chroma_pair() {
        let near = [0.8, 0.2, 0.1];
        let far = [0.1, 0.7, 0.9];
        // Zero chroma: the near tap, reconstructed through the round trip.
        let zero = melt_chroma_mix(near, far, 0.0);
        for (channel, (actual, expected)) in zero.into_iter().zip(near).enumerate() {
            assert!(
                (actual - expected).abs() <= 2.0e-3,
                "channel {channel}: {actual} vs {expected}"
            );
        }
        // Full chroma: the far tap's chroma under the near tap's luma.
        let full = melt_chroma_mix(near, far, 1.0);
        let y_full = rgb_to_yiq(full);
        let y_near = rgb_to_yiq(near);
        let y_far = rgb_to_yiq(far);
        assert!((y_full[0] - y_near[0]).abs() <= 2.0e-3, "luma stays near");
        assert!((y_full[1] - y_far[1]).abs() <= 2.0e-3, "I walks to far");
        assert!((y_full[2] - y_far[2]).abs() <= 2.0e-3, "Q walks to far");
    }

    #[test]
    fn yiq_round_trip_is_close_on_the_unit_cube() {
        for r in 0..3 {
            for g in 0..3 {
                for b in 0..3 {
                    let rgb = [r as f32 / 2.0, g as f32 / 2.0, b as f32 / 2.0];
                    let back = yiq_to_rgb(rgb_to_yiq(rgb));
                    for channel in 0..3 {
                        assert!((back[channel] - rgb[channel]).abs() <= 2.0e-3);
                    }
                }
            }
        }
    }

    #[test]
    fn bus_mixer_edits_parse_validate_and_apply_the_closed_vocabulary() {
        use serde_json::json;
        // Every continuous param round-trips through parse + apply.
        let table: [(&str, f64, f64); 17] = [
            ("wipe_soft", 0.4, 2.0),
            ("wipe_x", -0.5, 1.5),
            ("wipe_y", 0.5, -1.5),
            ("wipe_detail", 0.9, -0.1),
            ("wipe_border", 0.3, 1.1),
            ("dirt", 0.7, -0.1),
            ("dirt_rate", 0.2, 1.1),
            ("dirt_drop", 0.9, 1.1),
            ("dirt_cut", 0.1, -0.1),
            ("dirt_knock", 0.8, 1.1),
            ("dirt_noise", 0.6, 1.1),
            ("melt", 1.5, 2.1),
            ("melt_width", 1.9, -0.1),
            ("melt_hold", 1.2, 1.6),
            ("melt_swirl", -0.7, 1.1),
            ("melt_chroma", 0.9, -0.1),
            ("melt_creep", 0.1, 1.1),
        ];
        let mut state = BusMixerState::default();
        for (param, valid, invalid) in table {
            let edit = BusMixerEdit::parse(param, &json!(valid))
                .unwrap_or_else(|| panic!("{param} must accept {valid}"));
            edit.apply(&mut state);
            assert!(
                BusMixerEdit::parse(param, &json!(invalid)).is_none(),
                "{param} must reject {invalid}"
            );
            assert!(
                BusMixerEdit::parse(param, &json!(f64::NAN)).is_none(),
                "{param} must reject non-finite input"
            );
        }
        assert_eq!(state.mix.soft, 0.4);
        assert_eq!(state.melt.hold, 1.2);

        // Discrete vocabularies: closed tokens, typed rejections.
        let mut state = BusMixerState::default();
        BusMixerEdit::parse("wipe_pattern", &json!("clock"))
            .unwrap()
            .apply(&mut state);
        assert_eq!(state.mix.pattern, WipePattern::Clock);
        assert!(BusMixerEdit::parse("wipe_pattern", &json!("slide_left")).is_none());
        BusMixerEdit::parse("wipe_border_color", &json!("magenta"))
            .unwrap()
            .apply(&mut state);
        assert_eq!(state.mix.border_color, BackColor::Magenta);
        assert!(BusMixerEdit::parse("wipe_border_color", &json!("orange")).is_none());
        BusMixerEdit::parse("blend", &json!("vivid_light"))
            .unwrap()
            .apply(&mut state);
        assert_eq!(state.mix.blend, BlendMode::VividLight);
        // A crossfade has no destination to cut.
        assert!(BusMixerEdit::parse("blend", &json!("alpha_cut")).is_none());
        BusMixerEdit::parse("wipe_invert", &json!(true))
            .unwrap()
            .apply(&mut state);
        assert!(state.mix.invert);
        BusMixerEdit::parse("wipe_rep", &json!(4))
            .unwrap()
            .apply(&mut state);
        assert_eq!(state.mix.rep, 4);
        assert!(BusMixerEdit::parse("wipe_rep", &json!(0)).is_none());
        assert!(BusMixerEdit::parse("wipe_rep", &json!(5)).is_none());
        // Unknown params are rejections, never silently ignored values.
        assert!(BusMixerEdit::parse("bus_gain", &json!(0.5)).is_none());
    }

    #[test]
    fn mixer_state_serde_skips_nothing_and_rejects_unknown_fields() {
        // The bundle round-trips whole.
        let authored = BusMixerState {
            mix: BusMixParams {
                pattern: WipePattern::Box,
                border: 0.4,
                border_color: BackColor::Cyan,
                blend: BlendMode::PinLight,
                ..BusMixParams::default()
            },
            dirt: DirtParams {
                dirt: 0.5,
                ..DirtParams::default()
            },
            melt: MeltParams {
                melt: 1.0,
                ..MeltParams::default()
            },
        };
        let yaml = serde_yaml::to_string(&authored).unwrap();
        let restored: BusMixerState = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored, authored);
        // Unknown fields are rejected rather than silently dropped.
        assert!(serde_yaml::from_str::<BusMixerState>("mix:\n  melted: 1.0\n").is_err());
        // The exact-legacy law.
        assert!(BusMixerState::default().is_exact_legacy_bus());
        assert!(!authored.is_exact_legacy_bus());
    }

    #[test]
    fn avalanche_and_lanes_match_the_established_integer_law() {
        // The shared avalanche's fixed point at zero keeps the unseeded
        // stream branch-free.
        assert_eq!(avalanche(0), 0);
        // Distinct lanes decorrelate the same coordinates.
        assert_ne!(
            lane_hash(5, 9, LANE_DIRT_DROP_BAND, 0),
            lane_hash(5, 9, LANE_DIRT_DROP_SIDE, 0)
        );
        // The unit hash is the top 24 bits exactly.
        assert_eq!(hash_unit(0xffff_ffff), 16_777_215.0 / 16_777_216.0);
        assert_eq!(hash_unit(0x0000_00ff), 0.0);
    }
}
