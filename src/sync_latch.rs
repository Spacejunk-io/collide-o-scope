//! The B14 sync-latch law: the second and final failure switch.
//!
//! "A model that always recovers is a model that cannot actually break."
//! `servo_defeated` (B3) proved the shape at the feedback loop; this is the
//! other half. The seat is the tape/NTSC-adjacent horizontal shear: each
//! reference tick some bands of scanlines lose sync and slip sideways.
//! Unlatched, a slip lives exactly as long as its own tick and the picture
//! heals — the B8 knock's law, bit-clean between firings. Latched, every
//! slip is written into a bounded per-line offset table **and stays there**,
//! accumulating, until the operator releases the switch and the whole
//! accumulated displacement unwinds in one step.
//!
//! **Bounded state may latch but never grow.** The table is the entire
//! latched state: one `f32` per output line, hard-capped at
//! [`SYNC_LATCH_MAX_LINES`] lines and [`SYNC_LATCH_MAX_OFFSET`] of
//! displacement per line, so no resource law is threatened and the
//! accumulation cannot run away. Nothing else accumulates.
//!
//! Every draw is the Shift band/epoch/seed law on the shared integer
//! avalanche — [`crate::mixing_boundary::lane_unit`] in fresh per-lane
//! domains, keyed by the master random seed, the stage's own 30 Hz reference
//! ordinal, and the band index. Nothing consumes sequential RNG state, so a
//! tick recomputes alone, live and offline draw the same faults from a
//! common start, and Pause holds the fault stream still because the stage is
//! clocked by the program-advancing delta only.
//!
//! This module is the independent CPU reference in the `gesture.rs`
//! tradition — no `wgpu`, clock, filesystem, or UI dependency —
//! and `sync_latch.wgsl` consumes the table it produces.

use crate::mixing_boundary::lane_unit;

/// The hard cap on the stored table, in lines. A 4K output's line count;
/// the table is sized to the live output and never exceeds this.
pub const SYNC_LATCH_MAX_LINES: usize = 2160;

/// The bound on one line's accumulated displacement, in output UV. Latching
/// accumulates toward this and then stops: the picture can be shredded, but
/// the state cannot grow without limit.
pub const SYNC_LATCH_MAX_OFFSET: f32 = 0.25;

/// The magnitude of a single slip at full `amount`, in output UV.
pub const SYNC_LATCH_SLIP_UV: f32 = 0.02;

/// The wake deadband, the `MIX_AMOUNT_EPSILON` law.
pub const SYNC_LATCH_EPSILON: f32 = 0.002;

/// The most reference ticks one frame may fold, mirroring the 24-tick burst
/// clamp `history_ticks_for_delta` already applies: a long stall must not
/// bill the table for every skipped tick at once.
pub const SYNC_LATCH_MAX_TICK_BURST: u32 = 24;

/// The widest band a slip can cover, in lines.
pub const SYNC_LATCH_MAX_BAND_LINES: u32 = 64;

/// Fresh hash-lane domains — "SYN" 1 and 2. Each draw site owns exactly one
/// constant, so one lane can never perturb another.
pub const LANE_SYNC_FIRE: u32 = 0x5359_4e01;
pub const LANE_SYNC_SLIP: u32 = 0x5359_4e02;

/// The authored sync-latch state. Four continuous values plus the switch.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SyncLatchParams {
    /// Magnitude of a newly drawn slip. Zero is the exact prior path.
    pub amount: f32,
    /// How often a band loses sync, as a per-band firing probability per
    /// reference tick.
    pub rate: f32,
    /// Band height: 0 shears single lines (maximum shred), 1 shears blocks
    /// of [`SYNC_LATCH_MAX_BAND_LINES`] lines (a tape tear).
    pub spread: f32,
    /// Directional bias. Zero draws symmetric slips; ±1 forces every slip to
    /// one side, so a latched table accumulates monotonically toward the cap.
    pub bias: f32,
    /// **The switch.** Off, a slip heals with its own tick. On, every slip is
    /// written into the table and stays until release.
    pub latched: bool,
}

impl Default for SyncLatchParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            rate: 0.35,
            spread: 0.25,
            bias: 0.0,
            latched: false,
        }
    }
}

impl SyncLatchParams {
    /// Clamp every authored value into its declared range. Hostile
    /// non-finite input takes the neutral default rather than a clamped
    /// extreme.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            amount: finite_clamp(self.amount, defaults.amount, 0.0, 1.0),
            rate: finite_clamp(self.rate, defaults.rate, 0.0, 1.0),
            spread: finite_clamp(self.spread, defaults.spread, 0.0, 1.0),
            bias: finite_clamp(self.bias, defaults.bias, -1.0, 1.0),
            latched: self.latched,
        }
    }

    /// Whether new slips are being drawn at all. Both controls must be up: a
    /// magnitude with no rate never fires, and a rate with no magnitude
    /// displaces by nothing — either way the stage stays dormant and slot 0
    /// reaches the display stage untouched.
    ///
    /// The switch deliberately does **not** appear here. Latching an inert
    /// stage accumulates nothing, which is the honest reading of a switch
    /// that only decides whether faults heal.
    pub fn is_active(self) -> bool {
        let clean = self.sanitized();
        clean.amount > SYNC_LATCH_EPSILON && clean.rate > SYNC_LATCH_EPSILON
    }

    /// The exact pre-B14 authored state.
    pub fn is_exact_off(self) -> bool {
        self == Self::default()
    }
}

fn finite_clamp(value: f32, neutral: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        neutral
    }
}

fn fract(value: f32) -> f32 {
    value - value.floor()
}

/// The reference ordinal's low bits, the B3 rig's epoch law: the hash takes
/// a `u32`, and a wrap after 2^32 ticks (over four and a half years at
/// 30 Hz) is not a correctness boundary worth a wider lane.
pub fn tick_key(total_ticks: u64) -> u32 {
    (total_ticks & u64::from(u32::MAX)) as u32
}

/// Band height in lines: 1 at `spread` 0, [`SYNC_LATCH_MAX_BAND_LINES`] at 1.
/// Never zero, so the band index is always well defined.
pub fn band_height(spread: f32) -> u32 {
    let clean = finite_clamp(spread, 0.0, 0.0, 1.0);
    1 + (clean * (SYNC_LATCH_MAX_BAND_LINES - 1) as f32).round() as u32
}

/// Whether one band loses sync on one tick. At full rate half the bands slip
/// every tick; at zero nothing ever fires.
pub fn band_fires(band: u32, tick: u32, rate: f32, seed: u32) -> bool {
    let clean = finite_clamp(rate, 0.0, 0.0, 1.0);
    if clean <= SYNC_LATCH_EPSILON {
        return false;
    }
    lane_unit(band, tick, LANE_SYNC_FIRE, seed) <= clean * 0.5
}

/// One band's slip on one tick, in output UV, already clamped into the
/// stage's bound. `bias` folds the symmetric draw toward one side while
/// keeping its magnitude, so at ±1 every slip carries the same sign.
pub fn band_slip(band: u32, tick: u32, amount: f32, bias: f32, seed: u32) -> f32 {
    let amount = finite_clamp(amount, 0.0, 0.0, 1.0);
    let bias = finite_clamp(bias, 0.0, -1.0, 1.0);
    let centered = (lane_unit(band, tick, LANE_SYNC_SLIP, seed) - 0.5) * 2.0;
    let weight = bias.abs();
    let directed = if bias >= 0.0 {
        centered.abs()
    } else {
        -centered.abs()
    };
    let signed = centered * (1.0 - weight) + directed * weight;
    (signed * amount * SYNC_LATCH_SLIP_UV).clamp(-SYNC_LATCH_MAX_OFFSET, SYNC_LATCH_MAX_OFFSET)
}

/// The horizontal wrap law the shader applies: a sheared line wraps around
/// the frame exactly as a tape losing horizontal sync does. `sync_latch.wgsl`
/// mirrors this expression and its sampler repeats on U, so the bilinear tap
/// that straddles the seam filters across it rather than clamping.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the wrap executes in sync_latch.wgsl; this is the CPU reference the GPU parity fixture measures against"
    )
)]
pub fn sampled_u(u: f32, offset: f32) -> f32 {
    fract(u + offset)
}

/// The bounded per-line offset table — the entire latched state — plus the
/// applied offsets one frame hands to the shader.
///
/// The table is **program memory** on the temporal-ring and bus-melt
/// precedent: blackout darkens the audience without erasing it, and a
/// release resumes from the damage the cut interrupted. A source cut or seek
/// likewise keeps it, because neither starts a new program. Only a patch
/// generation, an Apply Look, a broad revert, and a resize clear it — a new
/// program has no history of shears.
///
/// The table is deliberately **not** persisted in patches. The switch
/// persists; the accumulation is runtime state that regrows deterministically
/// from the seed and the clock, exactly as a temporal ring does, so a patch
/// keeps its bytes and its canonical hash and live and offline agree from any
/// common start.
#[derive(Debug, Clone)]
pub struct SyncLatchState {
    /// The accumulated table. Always all-zero while the switch is off.
    latched: Vec<f32>,
    /// What the shader consumes this frame: the table when latched, the
    /// current tick's transient slips when not.
    applied: Vec<f32>,
    damaged: bool,
}

impl SyncLatchState {
    /// A cleared table sized to an output's line count, capped at
    /// [`SYNC_LATCH_MAX_LINES`] and never empty.
    pub fn new(line_count: u32) -> Self {
        let lines = Self::clamp_lines(line_count);
        Self {
            latched: vec![0.0; lines],
            applied: vec![0.0; lines],
            damaged: false,
        }
    }

    fn clamp_lines(line_count: u32) -> usize {
        (line_count.max(1) as usize).min(SYNC_LATCH_MAX_LINES)
    }

    /// The number of lines the table currently addresses.
    pub fn line_count(&self) -> u32 {
        self.latched.len() as u32
    }

    /// Zero the whole table in one step. This is both the release law — the
    /// accumulated displacement unwinds at once, never decaying out — and
    /// the hard-reset law.
    pub fn clear(&mut self) {
        if self.damaged {
            self.latched.fill(0.0);
            self.damaged = false;
        }
        self.applied.fill(0.0);
    }

    /// Whether the table currently holds any accumulated displacement. An
    /// undamaged, inactive stage encodes nothing.
    pub fn has_damage(&self) -> bool {
        self.damaged
    }

    /// The per-line offsets this frame applies, in output UV.
    pub fn applied(&self) -> &[f32] {
        &self.applied
    }

    /// Advance the stage to `current_tick`, folding the `elapsed` newly
    /// advanced reference ticks, and recompute the applied offsets. Exactly
    /// one call per frame; `elapsed` is zero on a frame inside a tick, which
    /// holds the shear still rather than redrawing it at frame rate.
    pub fn advance(&mut self, params: SyncLatchParams, current_tick: u64, elapsed: u32, seed: u32) {
        let clean = params.sanitized();
        let active = clean.is_active();

        if !clean.latched {
            // Release unwinds the entire accumulated displacement in one
            // step. Expressing it as "unlatched implies an empty table"
            // rather than as a falling-edge handler means the two can never
            // drift apart.
            if self.damaged {
                self.latched.fill(0.0);
                self.damaged = false;
            }
        } else if active && elapsed > 0 {
            let burst = elapsed.min(SYNC_LATCH_MAX_TICK_BURST);
            let first = current_tick.saturating_sub(u64::from(burst) - 1);
            for step in 0..burst {
                self.fold_tick(clean, tick_key(first.saturating_add(u64::from(step))), seed);
            }
            self.damaged = self.latched.iter().any(|offset| *offset != 0.0);
        }

        if clean.latched {
            // A latched tick is already folded, so the table is the whole
            // applied displacement. Note this holds even once `amount` or
            // `rate` falls back to zero: the switch stops new damage, it does
            // not repair the damage already done.
            self.applied.copy_from_slice(&self.latched);
        } else if active {
            let tick = tick_key(current_tick);
            let height = band_height(clean.spread);
            for (line, slot) in self.applied.iter_mut().enumerate() {
                let band = line as u32 / height;
                *slot = if band_fires(band, tick, clean.rate, seed) {
                    band_slip(band, tick, clean.amount, clean.bias, seed)
                } else {
                    0.0
                };
            }
        } else {
            self.applied.fill(0.0);
        }
    }

    fn fold_tick(&mut self, clean: SyncLatchParams, tick: u32, seed: u32) {
        let height = band_height(clean.spread);
        for (line, slot) in self.latched.iter_mut().enumerate() {
            let band = line as u32 / height;
            if band_fires(band, tick, clean.rate, seed) {
                let slip = band_slip(band, tick, clean.amount, clean.bias, seed);
                *slot = (*slot + slip).clamp(-SYNC_LATCH_MAX_OFFSET, SYNC_LATCH_MAX_OFFSET);
            }
        }
    }
}

/// The stage's GPU uniform header. The per-line offsets follow it in the same
/// buffer, written directly from [`SyncLatchState::applied`] so no frame ever
/// materializes the whole 8 KiB record on the stack. Field order mirrors
/// `SyncUniforms` in `sync_latch.wgsl` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SyncLatchGpuHeader {
    pub resolution: [f32; 2],
    pub line_count: u32,
    /// `active` is a WGSL reserved keyword, so the lane is named `armed`
    /// on both sides of the boundary.
    pub armed: u32,
}

const _: () = assert!(std::mem::size_of::<SyncLatchGpuHeader>() == 16);

/// The uniform buffer's exact size: the 16-byte header plus one `f32` per
/// capped line. WGSL sees the tail as `array<vec4<f32>, 540>`, which is the
/// same bytes at the 16-byte stride a uniform array requires.
pub const SYNC_LATCH_UNIFORM_BYTES: u64 =
    16 + (SYNC_LATCH_MAX_LINES as u64) * (std::mem::size_of::<f32>() as u64);

const _: () = assert!(SYNC_LATCH_UNIFORM_BYTES == 8_656);
const _: () = assert!(SYNC_LATCH_MAX_LINES.is_multiple_of(4));

impl SyncLatchGpuHeader {
    pub fn from_parts(dimensions: [u32; 2], line_count: u32, armed: bool) -> Self {
        Self {
            resolution: [dimensions[0].max(1) as f32, dimensions[1].max(1) as f32],
            line_count: line_count.max(1),
            armed: u32::from(armed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u32 = 7;

    fn armed() -> SyncLatchParams {
        SyncLatchParams {
            amount: 1.0,
            rate: 1.0,
            spread: 0.0,
            bias: 0.0,
            latched: false,
        }
    }

    #[test]
    fn the_default_is_the_exact_prior_path() {
        let params = SyncLatchParams::default();
        assert!(params.is_exact_off());
        assert!(!params.is_active());
        assert!(!params.latched);
        assert_eq!(params.amount, 0.0);
    }

    #[test]
    fn hostile_values_take_the_neutral_default_not_a_clamped_extreme() {
        let hostile = SyncLatchParams {
            amount: f32::NAN,
            rate: f32::INFINITY,
            spread: f32::NEG_INFINITY,
            bias: f32::NAN,
            latched: true,
        }
        .sanitized();
        let defaults = SyncLatchParams::default();
        assert_eq!(hostile.amount, defaults.amount);
        assert_eq!(hostile.rate, defaults.rate);
        assert_eq!(hostile.spread, defaults.spread);
        assert_eq!(hostile.bias, defaults.bias);
        // The switch is not a scalar and survives sanitize untouched.
        assert!(hostile.latched);
    }

    #[test]
    fn out_of_range_values_clamp_into_their_declared_ranges() {
        let clamped = SyncLatchParams {
            amount: 9.0,
            rate: -3.0,
            spread: 4.0,
            bias: -8.0,
            latched: false,
        }
        .sanitized();
        assert_eq!(clamped.amount, 1.0);
        assert_eq!(clamped.rate, 0.0);
        assert_eq!(clamped.spread, 1.0);
        assert_eq!(clamped.bias, -1.0);
    }

    #[test]
    fn neither_control_alone_wakes_the_stage() {
        let magnitude_only = SyncLatchParams {
            amount: 1.0,
            rate: 0.0,
            ..SyncLatchParams::default()
        };
        let rate_only = SyncLatchParams {
            amount: 0.0,
            rate: 1.0,
            ..SyncLatchParams::default()
        };
        assert!(!magnitude_only.is_active());
        assert!(!rate_only.is_active());
        assert!(SyncLatchParams {
            amount: 1.0,
            rate: 1.0,
            ..SyncLatchParams::default()
        }
        .is_active());
    }

    #[test]
    fn the_switch_alone_never_wakes_an_inert_stage() {
        let latched_but_inert = SyncLatchParams {
            latched: true,
            ..SyncLatchParams::default()
        };
        assert!(!latched_but_inert.is_active());
        let mut state = SyncLatchState::new(64);
        for tick in 0..200u64 {
            state.advance(latched_but_inert, tick, 1, SEED);
        }
        assert!(!state.has_damage());
        assert!(state.applied().iter().all(|offset| *offset == 0.0));
    }

    #[test]
    fn band_height_spans_one_line_to_the_cap() {
        assert_eq!(band_height(0.0), 1);
        assert_eq!(band_height(1.0), SYNC_LATCH_MAX_BAND_LINES);
        assert_eq!(band_height(f32::NAN), 1);
        for spread in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let height = band_height(spread);
            assert!((1..=SYNC_LATCH_MAX_BAND_LINES).contains(&height));
        }
    }

    #[test]
    fn a_zero_rate_never_fires_and_a_full_rate_fires_about_half_the_bands() {
        for band in 0..500u32 {
            assert!(!band_fires(band, 3, 0.0, SEED));
        }
        let fired = (0..4_000u32)
            .filter(|band| band_fires(*band, 11, 1.0, SEED))
            .count();
        // The draw is a unit hash against rate * 0.5.
        assert!(
            (1_800..=2_200).contains(&fired),
            "full rate fired {fired} of 4000 bands"
        );
    }

    #[test]
    fn bias_forces_the_slip_sign_while_keeping_its_magnitude() {
        for band in 0..512u32 {
            let symmetric = band_slip(band, 5, 1.0, 0.0, SEED);
            let positive = band_slip(band, 5, 1.0, 1.0, SEED);
            let negative = band_slip(band, 5, 1.0, -1.0, SEED);
            assert!(positive >= 0.0, "band {band} slipped {positive} at bias +1");
            assert!(negative <= 0.0, "band {band} slipped {negative} at bias -1");
            assert!((positive - symmetric.abs()).abs() < 1e-6);
            assert!((negative + symmetric.abs()).abs() < 1e-6);
        }
    }

    #[test]
    fn every_slip_stays_inside_the_declared_bound() {
        for band in 0..2_000u32 {
            for tick in 0..8u32 {
                let slip = band_slip(band, tick, 1.0, 0.0, SEED);
                assert!(slip.abs() <= SYNC_LATCH_SLIP_UV + 1e-7);
                assert!(slip.abs() <= SYNC_LATCH_MAX_OFFSET);
            }
        }
    }

    #[test]
    fn unlatched_slips_heal_with_their_own_tick() {
        let params = armed();
        let mut state = SyncLatchState::new(128);
        let mut moved_at_least_once = false;
        for tick in 0..64u64 {
            state.advance(params, tick, 1, SEED);
            // Nothing accumulates, ever.
            assert!(
                !state.has_damage(),
                "tick {tick} accumulated while unlatched"
            );
            if state.applied().iter().any(|offset| *offset != 0.0) {
                moved_at_least_once = true;
            }
        }
        assert!(moved_at_least_once, "an armed unlatched stage never moved");
    }

    #[test]
    fn a_frame_inside_a_tick_holds_the_shear_still() {
        let params = armed();
        let mut first = SyncLatchState::new(128);
        let mut second = SyncLatchState::new(128);
        first.advance(params, 9, 1, SEED);
        let after_advance = first.applied().to_vec();
        // The same tick with no newly elapsed tick — the second frame of a
        // 60 fps pair — must present the identical shear.
        second.advance(params, 9, 1, SEED);
        second.advance(params, 9, 0, SEED);
        assert_eq!(after_advance, second.applied());
    }

    #[test]
    fn latching_accumulates_monotonically_and_stops_at_the_cap() {
        let params = SyncLatchParams {
            amount: 1.0,
            rate: 1.0,
            spread: 0.0,
            bias: 1.0,
            latched: true,
        };
        let mut state = SyncLatchState::new(96);
        let mut previous = vec![0.0_f32; 96];
        for tick in 0..4_000u64 {
            state.advance(params, tick, 1, SEED);
            for (line, offset) in state.applied().iter().enumerate() {
                assert!(
                    *offset >= previous[line] - 1e-7,
                    "line {line} went backwards at tick {tick}"
                );
                assert!(
                    *offset <= SYNC_LATCH_MAX_OFFSET,
                    "line {line} exceeded the cap at tick {tick}"
                );
                previous[line] = *offset;
            }
        }
        assert!(state.has_damage());
        // With a positive bias and thousands of firings every line saturates,
        // which is exactly the bound doing its job.
        assert!(
            state
                .applied()
                .iter()
                .all(|offset| (*offset - SYNC_LATCH_MAX_OFFSET).abs() < 1e-6),
            "a saturating run left a line short of the cap"
        );
    }

    #[test]
    fn release_unwinds_the_whole_table_in_one_step() {
        let latched = SyncLatchParams {
            amount: 1.0,
            rate: 1.0,
            spread: 0.0,
            bias: 1.0,
            latched: true,
        };
        let mut state = SyncLatchState::new(96);
        for tick in 0..200u64 {
            state.advance(latched, tick, 1, SEED);
        }
        assert!(state.has_damage());
        assert!(state.applied().iter().any(|offset| *offset > 0.0));

        // Before release, lines carry accumulated displacement far beyond
        // one slip — that is what "latched" bought.
        assert!(
            state
                .applied()
                .iter()
                .any(|offset| *offset > SYNC_LATCH_SLIP_UV * 4.0),
            "the latched run never accumulated past a few single slips"
        );

        // One frame with the switch off unwinds the whole accumulation at
        // once — not decayed toward home over several frames. The stage is
        // still armed, so it keeps drawing transient faults; what must be
        // gone is every trace of the accumulation, and a transient slip can
        // never exceed one slip's magnitude.
        let released = SyncLatchParams {
            latched: false,
            ..latched
        };
        state.advance(released, 200, 0, SEED);
        assert!(!state.has_damage());
        assert!(
            state
                .applied()
                .iter()
                .all(|offset| offset.abs() <= SYNC_LATCH_SLIP_UV + 1e-7),
            "a released line still carried more than one slip of displacement"
        );

        // Released and silenced, every line is exactly home.
        let silent = SyncLatchParams {
            amount: 0.0,
            latched: false,
            ..latched
        };
        state.advance(silent, 201, 1, SEED);
        assert!(!state.has_damage());
        assert!(state.applied().iter().all(|offset| *offset == 0.0));
    }

    #[test]
    fn damage_survives_a_magnitude_pulled_to_zero_while_latched() {
        let latched = SyncLatchParams {
            amount: 1.0,
            rate: 1.0,
            spread: 0.0,
            bias: 1.0,
            latched: true,
        };
        let mut state = SyncLatchState::new(64);
        for tick in 0..50u64 {
            state.advance(latched, tick, 1, SEED);
        }
        let held = state.applied().to_vec();
        assert!(held.iter().any(|offset| *offset > 0.0));

        // The switch stops new damage; it does not repair what is done.
        let silenced = SyncLatchParams {
            amount: 0.0,
            ..latched
        };
        for tick in 50..120u64 {
            state.advance(silenced, tick, 1, SEED);
        }
        assert_eq!(held, state.applied());
        assert!(state.has_damage());
    }

    #[test]
    fn a_stalled_frame_folds_at_most_the_burst_clamp() {
        let params = SyncLatchParams {
            amount: 1.0,
            rate: 1.0,
            spread: 0.0,
            bias: 1.0,
            latched: true,
        };
        let mut burst = SyncLatchState::new(64);
        let mut stepped = SyncLatchState::new(64);
        // One frame claiming a thousand elapsed ticks folds the last 24.
        burst.advance(params, 1_000, 1_000, SEED);
        for tick in (1_000 - u64::from(SYNC_LATCH_MAX_TICK_BURST) + 1)..=1_000 {
            stepped.advance(params, tick, 1, SEED);
        }
        assert_eq!(burst.applied(), stepped.applied());
    }

    #[test]
    fn the_table_is_deterministic_per_seed_and_distinct_across_seeds() {
        let params = SyncLatchParams {
            amount: 1.0,
            rate: 0.6,
            spread: 0.2,
            bias: 0.4,
            latched: true,
        };
        let run = |seed: u32| {
            let mut state = SyncLatchState::new(120);
            for tick in 0..60u64 {
                state.advance(params, tick, 1, seed);
            }
            state.applied().to_vec()
        };
        assert_eq!(run(11), run(11));
        assert_ne!(run(11), run(12));
    }

    #[test]
    fn the_table_is_bounded_by_the_line_cap() {
        let huge = SyncLatchState::new(100_000);
        assert_eq!(huge.line_count(), SYNC_LATCH_MAX_LINES as u32);
        assert_eq!(huge.applied().len(), SYNC_LATCH_MAX_LINES);
        let zero = SyncLatchState::new(0);
        assert_eq!(zero.line_count(), 1);
        let exact = SyncLatchState::new(SYNC_LATCH_MAX_LINES as u32 + 1);
        assert_eq!(exact.line_count(), SYNC_LATCH_MAX_LINES as u32);
    }

    #[test]
    fn a_hard_clear_zeroes_the_table_without_resizing_it() {
        let params = SyncLatchParams {
            amount: 1.0,
            rate: 1.0,
            spread: 0.0,
            bias: 1.0,
            latched: true,
        };
        let mut state = SyncLatchState::new(80);
        for tick in 0..40u64 {
            state.advance(params, tick, 1, SEED);
        }
        assert!(state.has_damage());
        state.clear();
        assert!(!state.has_damage());
        assert_eq!(state.line_count(), 80);
        assert!(state.applied().iter().all(|offset| *offset == 0.0));
    }

    #[test]
    fn spread_shears_contiguous_blocks_of_lines() {
        let params = SyncLatchParams {
            amount: 1.0,
            rate: 0.5,
            spread: 1.0,
            bias: 0.0,
            latched: false,
        };
        let mut state = SyncLatchState::new(256);
        state.advance(params, 3, 1, SEED);
        let height = band_height(1.0) as usize;
        // Every line inside one band carries the identical offset, which is
        // what makes a tear a tear rather than static.
        for block in 0..(256 / height) {
            let first = state.applied()[block * height];
            for line in 0..height {
                assert_eq!(state.applied()[block * height + line], first);
            }
        }
    }

    #[test]
    fn the_wrap_law_keeps_every_sample_inside_the_frame() {
        for step in 0..=100u32 {
            let u = step as f32 / 100.0;
            for offset in [
                -SYNC_LATCH_MAX_OFFSET,
                -0.02,
                0.0,
                0.02,
                SYNC_LATCH_MAX_OFFSET,
            ] {
                let sampled = sampled_u(u, offset);
                assert!(
                    (0.0..1.0).contains(&sampled),
                    "u {u} + {offset} sampled {sampled}"
                );
            }
        }
        assert!((sampled_u(0.1, -0.25) - 0.85).abs() < 1e-6);
        assert!((sampled_u(0.9, 0.25) - 0.15).abs() < 1e-6);
    }

    #[test]
    fn the_uniform_header_and_buffer_sizes_are_frozen() {
        assert_eq!(std::mem::size_of::<SyncLatchGpuHeader>(), 16);
        assert_eq!(SYNC_LATCH_UNIFORM_BYTES, 8_656);
        let header = SyncLatchGpuHeader::from_parts([1920, 1080], 1080, true);
        assert_eq!(header.resolution, [1920.0, 1080.0]);
        assert_eq!(header.line_count, 1080);
        assert_eq!(header.armed, 1);
        let dormant = SyncLatchGpuHeader::from_parts([0, 0], 0, false);
        assert_eq!(dormant.resolution, [1.0, 1.0]);
        assert_eq!(dormant.line_count, 1);
        assert_eq!(dormant.armed, 0);
    }

    #[test]
    fn the_hash_lanes_are_distinct_domains() {
        assert_ne!(LANE_SYNC_FIRE, LANE_SYNC_SLIP);
        // A fresh domain must not collide with the B8 dirt lanes it borrows
        // its law from.
        for existing in [
            crate::mixing_boundary::LANE_DIRT_FIRE,
            crate::mixing_boundary::LANE_DIRT_KNOCK_FRAME,
            crate::mixing_boundary::LANE_DIRT_KNOCK_ROW,
            crate::mixing_boundary::LANE_WIPE_BLOCKS,
        ] {
            assert_ne!(LANE_SYNC_FIRE, existing);
            assert_ne!(LANE_SYNC_SLIP, existing);
        }
    }
}
