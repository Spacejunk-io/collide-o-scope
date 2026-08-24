//! Evaluation-only D2 photosensitivity-risk advisor contracts.
//!
//! This is a measurement and classification reference, not a medical or
//! regulatory safety system. The prototype is intentionally not wired into
//! the live application until its P1 GPU/readback performance gate and a
//! separate accessibility/legal review exist. It has no API that can mutate,
//! attenuate, replace, or veto program pixels.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

pub const ADVISOR_ALGORITHM_NAME: &str = "cos-photosensitivity-advisor-evaluation";
pub const ADVISOR_ALGORITHM_VERSION: u16 = 1;
pub const ADVISOR_GRID_WIDTH: usize = 64;
pub const ADVISOR_GRID_HEIGHT: usize = 36;
pub const ADVISOR_CELLS: usize = ADVISOR_GRID_WIDTH * ADVISOR_GRID_HEIGHT;
pub const ADVISOR_TAPS_PER_AXIS: usize = 4;
pub const ADVISOR_TAPS_PER_CELL: usize = ADVISOR_TAPS_PER_AXIS * ADVISOR_TAPS_PER_AXIS;
pub const ADVISOR_RING_CAPACITY: usize = 120;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdvisorAlgorithm {
    CosPhotosensitivityEvaluation,
}

pub const ADVISOR_ALGORITHM: AdvisorAlgorithm = AdvisorAlgorithm::CosPhotosensitivityEvaluation;

/// IEC-style sRGB transfer values rounded to linear-light Q0.16. Pinning the
/// complete 8-bit domain avoids platform `powf` differences in the reference.
const SRGB8_TO_LINEAR_Q: [u32; 256] = [
    0, 20, 40, 60, 80, 99, 119, 139, 159, 179, 199, 219, 241, 264, 288, 313, 340, 367, 396, 427,
    458, 491, 526, 562, 599, 637, 677, 718, 761, 805, 851, 898, 947, 997, 1048, 1101, 1156, 1212,
    1270, 1330, 1391, 1453, 1517, 1583, 1651, 1720, 1790, 1863, 1937, 2013, 2090, 2170, 2250, 2333,
    2418, 2504, 2592, 2681, 2773, 2866, 2961, 3058, 3157, 3258, 3360, 3464, 3570, 3678, 3788, 3900,
    4014, 4129, 4247, 4366, 4488, 4611, 4736, 4864, 4993, 5124, 5257, 5392, 5530, 5669, 5810, 5953,
    6099, 6246, 6395, 6547, 6700, 6856, 7014, 7174, 7335, 7500, 7666, 7834, 8004, 8177, 8352, 8528,
    8708, 8889, 9072, 9258, 9445, 9635, 9828, 10022, 10219, 10417, 10619, 10822, 11028, 11235,
    11446, 11658, 11873, 12090, 12309, 12530, 12754, 12980, 13209, 13440, 13673, 13909, 14146,
    14387, 14629, 14874, 15122, 15371, 15623, 15878, 16135, 16394, 16656, 16920, 17187, 17456,
    17727, 18001, 18277, 18556, 18837, 19121, 19407, 19696, 19987, 20281, 20577, 20876, 21177,
    21481, 21787, 22096, 22407, 22721, 23038, 23357, 23678, 24002, 24329, 24658, 24990, 25325,
    25662, 26001, 26344, 26688, 27036, 27386, 27739, 28094, 28452, 28813, 29176, 29542, 29911,
    30282, 30656, 31033, 31412, 31794, 32179, 32567, 32957, 33350, 33745, 34143, 34544, 34948,
    35355, 35764, 36176, 36591, 37008, 37429, 37852, 38278, 38706, 39138, 39572, 40009, 40449,
    40891, 41337, 41785, 42236, 42690, 43147, 43606, 44069, 44534, 45002, 45473, 45947, 46423,
    46903, 47385, 47871, 48359, 48850, 49344, 49841, 50341, 50844, 51349, 51858, 52369, 52884,
    53401, 53921, 54445, 54971, 55500, 56032, 56567, 57105, 57646, 58190, 58737, 59287, 59840,
    60396, 60955, 61517, 62082, 62650, 63221, 63795, 64372, 64952, 65535,
];

/// The only truthful production status until the P1 p95/p99 receipt and the
/// independent review named by RFC D2 exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdvisorAvailability {
    DeferredP1AndReview,
}

pub const ADVISOR_AVAILABILITY: AdvisorAvailability = AdvisorAvailability::DeferredP1AndReview;

/// Operator-supplied evaluation thresholds. There is deliberately no
/// `Default`: this crate does not guess a standards-derived venue policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdvisorPolicy {
    /// Per-cell luma or RGB transition threshold in linear-light Q0.16.
    pub transition_threshold_q: u16,
    /// A cell is red-saturated at or above this linear-light red value.
    pub red_saturation_q: u16,
    /// Required red lead over both green and blue, in Q0.16.
    pub red_dominance_q: u16,
    /// Minimum affected lattice area for one transition event.
    pub min_affected_cells: u16,
    /// Minimum reversing cells for one reversal event.
    pub min_reversal_cells: u16,
    /// Minimum red-transition cells for one red event.
    pub min_red_cells: u16,
    /// Reference-tick window, bounded by the fixed four-second ring.
    pub window_ticks: u16,
    pub attention_transition_events: u16,
    pub elevated_transition_events: u16,
    pub elevated_reversal_events: u16,
    pub elevated_red_events: u16,
    /// Consecutive reference ticks with qualifying affected area. This is a
    /// duration measurement, not a wall-clock timer; missing ticks break it.
    pub elevated_sustained_ticks: u16,
}

impl AdvisorPolicy {
    pub fn validate(self) -> Result<Self, AdvisorPolicyError> {
        if self.transition_threshold_q == 0 {
            return Err(AdvisorPolicyError::ZeroTransitionThreshold);
        }
        for (field, value) in [
            ("min_affected_cells", self.min_affected_cells),
            ("min_reversal_cells", self.min_reversal_cells),
            ("min_red_cells", self.min_red_cells),
        ] {
            if value == 0 || usize::from(value) > ADVISOR_CELLS {
                return Err(AdvisorPolicyError::CellThreshold { field, value });
            }
        }
        if self.window_ticks == 0 || usize::from(self.window_ticks) > ADVISOR_RING_CAPACITY {
            return Err(AdvisorPolicyError::WindowTicks(self.window_ticks));
        }
        for (field, value) in [
            (
                "attention_transition_events",
                self.attention_transition_events,
            ),
            (
                "elevated_transition_events",
                self.elevated_transition_events,
            ),
            ("elevated_reversal_events", self.elevated_reversal_events),
            ("elevated_red_events", self.elevated_red_events),
            ("elevated_sustained_ticks", self.elevated_sustained_ticks),
        ] {
            if value == 0 || value > self.window_ticks {
                return Err(AdvisorPolicyError::EventThreshold { field, value });
            }
        }
        if self.elevated_transition_events < self.attention_transition_events {
            return Err(AdvisorPolicyError::ElevatedBelowAttention);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvisorPolicyError {
    ZeroTransitionThreshold,
    CellThreshold { field: &'static str, value: u16 },
    WindowTicks(u16),
    EventThreshold { field: &'static str, value: u16 },
    ElevatedBelowAttention,
}

impl fmt::Display for AdvisorPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ZeroTransitionThreshold => {
                formatter.write_str("transition threshold must be nonzero")
            }
            Self::CellThreshold { field, value } => write!(
                formatter,
                "{field}={value} is outside the fixed 1..={ADVISOR_CELLS} lattice bound"
            ),
            Self::WindowTicks(value) => write!(
                formatter,
                "window_ticks={value} is outside the fixed 1..={ADVISOR_RING_CAPACITY} bound"
            ),
            Self::EventThreshold { field, value } => write!(
                formatter,
                "{field}={value} is zero or exceeds the configured window"
            ),
            Self::ElevatedBelowAttention => formatter.write_str(
                "elevated transition count cannot be below the attention transition count",
            ),
        }
    }
}

impl std::error::Error for AdvisorPolicyError {}

/// The complete GPU-to-CPU payload. All fields are aggregate integers; no
/// pixels, text, paths, source names, or authored values can enter readback.
#[repr(C)]
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    bytemuck::Pod,
    bytemuck::Zeroable,
)]
#[serde(deny_unknown_fields)]
pub struct CompactTransitionCounters {
    pub sampled_cells: u32,
    pub initialized_cells: u32,
    pub affected_cells: u32,
    pub reversal_cells: u32,
    pub red_transition_cells: u32,
    pub luma_delta_sum_q: u32,
    pub color_delta_sum_q: u32,
    pub reserved: u32,
}

impl CompactTransitionCounters {
    pub const BYTE_LEN: u64 = std::mem::size_of::<Self>() as u64;

    pub fn validate(self) -> Result<Self, AdvisorSampleError> {
        if self.sampled_cells != ADVISOR_CELLS as u32 {
            return Err(AdvisorSampleError::SampledCells(self.sampled_cells));
        }
        if self.initialized_cells > self.sampled_cells
            || self.affected_cells > self.initialized_cells
            || self.reversal_cells > self.affected_cells
            || self.red_transition_cells > self.affected_cells
        {
            return Err(AdvisorSampleError::CounterOrdering);
        }
        let max_sum = self.initialized_cells.saturating_mul(u16::MAX.into());
        if self.luma_delta_sum_q > max_sum || self.color_delta_sum_q > max_sum {
            return Err(AdvisorSampleError::DeltaSum);
        }
        if self.reserved != 0 {
            return Err(AdvisorSampleError::Reserved(self.reserved));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvisorSampleError {
    InvalidPolicy(AdvisorPolicyError),
    EmptyRaster,
    RasterLength,
    RasterOverflow,
    SampledCells(u32),
    CounterOrdering,
    DeltaSum,
    Reserved(u32),
    StaleSequence,
    StaleReferenceTick,
}

impl fmt::Display for AdvisorSampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidPolicy(error) => write!(formatter, "invalid advisor policy: {error}"),
            Self::EmptyRaster => formatter.write_str("advisor raster is empty"),
            Self::RasterLength => formatter.write_str("advisor raster byte length is incomplete"),
            Self::RasterOverflow => formatter.write_str("advisor raster byte length overflowed"),
            Self::SampledCells(value) => write!(
                formatter,
                "advisor sample contains {value} cells; expected {ADVISOR_CELLS}"
            ),
            Self::CounterOrdering => {
                formatter.write_str("advisor aggregate counters violate their subset bounds")
            }
            Self::DeltaSum => formatter.write_str("advisor aggregate delta sum exceeds its bound"),
            Self::Reserved(value) => {
                write!(formatter, "advisor reserved counter is nonzero ({value})")
            }
            Self::StaleSequence => formatter.write_str("advisor sequence is not newer"),
            Self::StaleReferenceTick => formatter.write_str("advisor reference tick is not newer"),
        }
    }
}

impl std::error::Error for AdvisorSampleError {}

#[derive(Debug, Clone, Copy, Default)]
struct ReferenceCellHistory {
    rgb_luma: [u32; 4],
    direction: i32,
    initialized: bool,
}

/// Deterministic CPU reference for the fixed 64×36×16-sample GPU kernel.
/// The input slice is borrowed and never retained or mutated.
pub struct PhotosensitivityCpuReference {
    cells: Vec<ReferenceCellHistory>,
}

impl Default for PhotosensitivityCpuReference {
    fn default() -> Self {
        Self {
            cells: vec![ReferenceCellHistory::default(); ADVISOR_CELLS],
        }
    }
}

impl PhotosensitivityCpuReference {
    pub fn reset(&mut self) {
        self.cells.fill(ReferenceCellHistory::default());
    }

    pub fn analyze_rgba8_srgb(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
        policy: AdvisorPolicy,
    ) -> Result<CompactTransitionCounters, AdvisorSampleError> {
        let policy = policy
            .validate()
            .map_err(AdvisorSampleError::InvalidPolicy)?;
        if width == 0 || height == 0 {
            return Err(AdvisorSampleError::EmptyRaster);
        }
        let required = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(AdvisorSampleError::RasterOverflow)?;
        if pixels.len() < required {
            return Err(AdvisorSampleError::RasterLength);
        }

        let mut counters = CompactTransitionCounters {
            sampled_cells: ADVISOR_CELLS as u32,
            ..CompactTransitionCounters::default()
        };
        for cell_y in 0..ADVISOR_GRID_HEIGHT {
            for cell_x in 0..ADVISOR_GRID_WIDTH {
                let cell_index = cell_y * ADVISOR_GRID_WIDTH + cell_x;
                let current = reference_cell_rgb_luma(pixels, width, height, cell_x, cell_y);
                let history = &mut self.cells[cell_index];
                if !history.initialized {
                    history.rgb_luma = current;
                    history.initialized = true;
                    continue;
                }
                counters.initialized_cells += 1;
                let luma_delta = current[3].abs_diff(history.rgb_luma[3]);
                let color_delta = current[..3]
                    .iter()
                    .zip(&history.rgb_luma[..3])
                    .map(|(current, previous)| current.abs_diff(*previous))
                    .max()
                    .unwrap_or(0);
                counters.luma_delta_sum_q = counters.luma_delta_sum_q.saturating_add(luma_delta);
                counters.color_delta_sum_q = counters.color_delta_sum_q.saturating_add(color_delta);

                let affected =
                    luma_delta.max(color_delta) >= u32::from(policy.transition_threshold_q);
                if affected {
                    counters.affected_cells += 1;
                    if red_saturated(current, policy) || red_saturated(history.rgb_luma, policy) {
                        counters.red_transition_cells += 1;
                    }
                }

                if luma_delta >= u32::from(policy.transition_threshold_q) {
                    let direction = match current[3].cmp(&history.rgb_luma[3]) {
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                        std::cmp::Ordering::Greater => 1,
                    };
                    if history.direction != 0 && direction != 0 && direction != history.direction {
                        counters.reversal_cells += 1;
                    }
                    if direction != 0 {
                        history.direction = direction;
                    }
                }
                history.rgb_luma = current;
            }
        }
        counters.validate()
    }
}

fn red_saturated(rgb_luma: [u32; 4], policy: AdvisorPolicy) -> bool {
    let red = rgb_luma[0];
    let dominance = u32::from(policy.red_dominance_q);
    red >= u32::from(policy.red_saturation_q)
        && red >= rgb_luma[1].saturating_add(dominance)
        && red >= rgb_luma[2].saturating_add(dominance)
}

fn reference_cell_rgb_luma(
    pixels: &[u8],
    width: usize,
    height: usize,
    cell_x: usize,
    cell_y: usize,
) -> [u32; 4] {
    let mut sum = [0_u32; 3];
    for tap_y in 0..ADVISOR_TAPS_PER_AXIS {
        for tap_x in 0..ADVISOR_TAPS_PER_AXIS {
            let x = sample_coordinate(cell_x, tap_x, ADVISOR_GRID_WIDTH, width);
            let y = sample_coordinate(cell_y, tap_y, ADVISOR_GRID_HEIGHT, height);
            let offset = (y * width + x) * 4;
            for channel in 0..3 {
                sum[channel] += srgb_code_to_linear_q(pixels[offset + channel]);
            }
        }
    }
    let rounded_half = (ADVISOR_TAPS_PER_CELL / 2) as u32;
    let divisor = ADVISOR_TAPS_PER_CELL as u32;
    let rgb = sum.map(|value| value.saturating_add(rounded_half) / divisor);
    [rgb[0], rgb[1], rgb[2], luma_q(rgb)]
}

fn sample_coordinate(cell: usize, tap: usize, grid_extent: usize, raster_extent: usize) -> usize {
    let subdivision = cell * ADVISOR_TAPS_PER_AXIS + tap;
    let numerator = (subdivision * 2 + 1) as u64 * raster_extent as u64;
    let denominator = (grid_extent * ADVISOR_TAPS_PER_AXIS * 2) as u64;
    ((numerator / denominator) as usize).min(raster_extent - 1)
}

pub(crate) fn srgb_code_to_linear_q(code: u8) -> u32 {
    SRGB8_TO_LINEAR_Q[usize::from(code)]
}

fn luma_q(rgb: [u32; 3]) -> u32 {
    // Rec.709 coefficients sum to exactly 65,536. Integer arithmetic makes
    // CPU/GPU classification independent of fused floating-point behavior.
    rgb[0]
        .saturating_mul(13_933)
        .saturating_add(rgb[1].saturating_mul(46_871))
        .saturating_add(rgb[2].saturating_mul(4_732))
        .saturating_add(32_768)
        / 65_536
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdvisorLevel {
    Clear,
    Attention,
    Elevated,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdvisorCounters {
    pub admitted_samples: u64,
    pub rejected_malformed_samples: u64,
    pub rejected_stale_samples: u64,
    pub missing_reference_ticks: u64,
    pub clear_results: u64,
    pub attention_results: u64,
    pub elevated_results: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdvisorTelemetry {
    pub algorithm: AdvisorAlgorithm,
    pub algorithm_version: u16,
    pub availability: AdvisorAvailability,
    pub policy: AdvisorPolicy,
    pub level: AdvisorLevel,
    pub last_sequence: u64,
    pub last_reference_tick: u64,
    pub window_observations: u16,
    pub window_transition_events: u16,
    pub window_reversal_events: u16,
    pub window_red_events: u16,
    pub window_longest_sustained_ticks: u16,
    pub latest: CompactTransitionCounters,
    pub counters: AdvisorCounters,
}

#[derive(Debug, Clone, Copy)]
struct AdvisorObservation {
    reference_tick: u64,
    transition: bool,
    reversal: bool,
    red: bool,
}

/// Evaluation-only bounded classifier. Construction is explicit and policy
/// validation is mandatory; nothing creates this type in the live app.
pub struct PhotosensitivityAdvisor {
    policy: AdvisorPolicy,
    ring: VecDeque<AdvisorObservation>,
    last_sequence: Option<u64>,
    last_reference_tick: Option<u64>,
    latest: CompactTransitionCounters,
    counters: AdvisorCounters,
}

impl PhotosensitivityAdvisor {
    pub fn new_evaluation_only(policy: AdvisorPolicy) -> Result<Self, AdvisorPolicyError> {
        Ok(Self {
            policy: policy.validate()?,
            ring: VecDeque::with_capacity(ADVISOR_RING_CAPACITY),
            last_sequence: None,
            last_reference_tick: None,
            latest: CompactTransitionCounters::default(),
            counters: AdvisorCounters::default(),
        })
    }

    pub fn reset_observations(&mut self) {
        self.ring.clear();
        self.last_sequence = None;
        self.last_reference_tick = None;
        self.latest = CompactTransitionCounters::default();
    }

    pub fn observe(
        &mut self,
        sequence: u64,
        reference_tick: u64,
        sample: CompactTransitionCounters,
    ) -> Result<AdvisorTelemetry, AdvisorSampleError> {
        let sample = match sample.validate() {
            Ok(sample) => sample,
            Err(error) => {
                self.counters.rejected_malformed_samples =
                    self.counters.rejected_malformed_samples.saturating_add(1);
                return Err(error);
            }
        };
        if self.last_sequence.is_some_and(|last| sequence <= last) {
            self.counters.rejected_stale_samples =
                self.counters.rejected_stale_samples.saturating_add(1);
            return Err(AdvisorSampleError::StaleSequence);
        }
        if self
            .last_reference_tick
            .is_some_and(|last| reference_tick <= last)
        {
            self.counters.rejected_stale_samples =
                self.counters.rejected_stale_samples.saturating_add(1);
            return Err(AdvisorSampleError::StaleReferenceTick);
        }
        if let Some(last) = self.last_reference_tick {
            self.counters.missing_reference_ticks = self
                .counters
                .missing_reference_ticks
                .saturating_add(reference_tick.saturating_sub(last).saturating_sub(1));
        }

        let observation = AdvisorObservation {
            reference_tick,
            transition: sample.affected_cells >= u32::from(self.policy.min_affected_cells),
            reversal: sample.reversal_cells >= u32::from(self.policy.min_reversal_cells),
            red: sample.red_transition_cells >= u32::from(self.policy.min_red_cells),
        };
        if self.ring.len() == ADVISOR_RING_CAPACITY {
            self.ring.pop_front();
        }
        self.ring.push_back(observation);
        while self.ring.front().is_some_and(|oldest| {
            reference_tick.saturating_sub(oldest.reference_tick)
                >= u64::from(self.policy.window_ticks)
        }) {
            self.ring.pop_front();
        }
        self.last_sequence = Some(sequence);
        self.last_reference_tick = Some(reference_tick);
        self.latest = sample;
        self.counters.admitted_samples = self.counters.admitted_samples.saturating_add(1);

        let telemetry = self.telemetry();
        match telemetry.level {
            AdvisorLevel::Clear => {
                self.counters.clear_results = self.counters.clear_results.saturating_add(1)
            }
            AdvisorLevel::Attention => {
                self.counters.attention_results = self.counters.attention_results.saturating_add(1)
            }
            AdvisorLevel::Elevated => {
                self.counters.elevated_results = self.counters.elevated_results.saturating_add(1)
            }
        }
        Ok(self.telemetry())
    }

    pub fn telemetry(&self) -> AdvisorTelemetry {
        let transition_events = self
            .ring
            .iter()
            .filter(|observation| observation.transition)
            .count() as u16;
        let reversal_events = self
            .ring
            .iter()
            .filter(|observation| observation.reversal)
            .count() as u16;
        let red_events = self
            .ring
            .iter()
            .filter(|observation| observation.red)
            .count() as u16;
        let mut longest_sustained_ticks = 0_u16;
        let mut current_sustained_ticks = 0_u16;
        let mut previous_transition_tick = None;
        for observation in &self.ring {
            if observation.transition {
                current_sustained_ticks = if previous_transition_tick
                    .is_some_and(|previous| observation.reference_tick == previous + 1)
                {
                    current_sustained_ticks.saturating_add(1)
                } else {
                    1
                };
                longest_sustained_ticks = longest_sustained_ticks.max(current_sustained_ticks);
                previous_transition_tick = Some(observation.reference_tick);
            } else {
                current_sustained_ticks = 0;
                previous_transition_tick = None;
            }
        }
        let elevated = transition_events >= self.policy.elevated_transition_events
            && (reversal_events >= self.policy.elevated_reversal_events
                || red_events >= self.policy.elevated_red_events
                || longest_sustained_ticks >= self.policy.elevated_sustained_ticks);
        let attention = transition_events >= self.policy.attention_transition_events;
        AdvisorTelemetry {
            algorithm: ADVISOR_ALGORITHM,
            algorithm_version: ADVISOR_ALGORITHM_VERSION,
            availability: ADVISOR_AVAILABILITY,
            policy: self.policy,
            level: if elevated {
                AdvisorLevel::Elevated
            } else if attention {
                AdvisorLevel::Attention
            } else {
                AdvisorLevel::Clear
            },
            last_sequence: self.last_sequence.unwrap_or(0),
            last_reference_tick: self.last_reference_tick.unwrap_or(0),
            window_observations: self.ring.len() as u16,
            window_transition_events: transition_events,
            window_reversal_events: reversal_events,
            window_red_events: red_events,
            window_longest_sustained_ticks: longest_sustained_ticks,
            latest: self.latest,
            counters: self.counters,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: usize = 256;
    const HEIGHT: usize = 144;

    fn policy() -> AdvisorPolicy {
        AdvisorPolicy {
            // These are synthetic-test thresholds, not a venue or standards
            // recommendation.
            transition_threshold_q: 4_000,
            red_saturation_q: 40_000,
            red_dominance_q: 12_000,
            min_affected_cells: 384,
            min_reversal_cells: 384,
            min_red_cells: 384,
            window_ticks: 120,
            attention_transition_events: 2,
            elevated_transition_events: 4,
            elevated_reversal_events: 2,
            elevated_red_events: 2,
            elevated_sustained_ticks: 4,
        }
        .validate()
        .expect("synthetic evaluation policy")
    }

    fn solid(r: u8, g: u8, b: u8) -> Vec<u8> {
        [r, g, b, 255].repeat(WIDTH * HEIGHT)
    }

    #[test]
    fn d2_pinned_srgb_table_covers_the_complete_monotonic_code_domain() {
        assert_eq!(SRGB8_TO_LINEAR_Q.len(), 256);
        assert_eq!(SRGB8_TO_LINEAR_Q[0], 0);
        assert_eq!(SRGB8_TO_LINEAR_Q[10], 199);
        assert_eq!(SRGB8_TO_LINEAR_Q[64], 3_360);
        assert_eq!(SRGB8_TO_LINEAR_Q[128], 14_146);
        assert_eq!(SRGB8_TO_LINEAR_Q[192], 34_544);
        assert_eq!(SRGB8_TO_LINEAR_Q[255], 65_535);
        assert!(SRGB8_TO_LINEAR_Q
            .windows(2)
            .all(|codes| codes[0] < codes[1]));
    }

    fn patch(
        mut frame: Vec<u8>,
        patch_width: usize,
        patch_height: usize,
        rgba: [u8; 4],
    ) -> Vec<u8> {
        for y in 0..patch_height.min(HEIGHT) {
            for x in 0..patch_width.min(WIDTH) {
                let offset = (y * WIDTH + x) * 4;
                frame[offset..offset + 4].copy_from_slice(&rgba);
            }
        }
        frame
    }

    #[test]
    fn d2_cpu_reference_is_deterministic_borrow_only_and_fixed_work() {
        let pixels = patch(solid(9, 20, 31), 73, 51, [220, 41, 17, 255]);
        let before = pixels.clone();
        let mut first = PhotosensitivityCpuReference::default();
        let mut second = PhotosensitivityCpuReference::default();

        let first_prime = first
            .analyze_rgba8_srgb(&pixels, WIDTH, HEIGHT, policy())
            .expect("prime first reference");
        let second_prime = second
            .analyze_rgba8_srgb(&pixels, WIDTH, HEIGHT, policy())
            .expect("prime second reference");
        assert_eq!(first_prime, second_prime);
        assert_eq!(first_prime.sampled_cells, ADVISOR_CELLS as u32);
        assert_eq!(first_prime.initialized_cells, 0);

        let changed = patch(solid(9, 20, 31), 73, 51, [17, 220, 41, 255]);
        let first_changed = first
            .analyze_rgba8_srgb(&changed, WIDTH, HEIGHT, policy())
            .expect("analyze first reference");
        let second_changed = second
            .analyze_rgba8_srgb(&changed, WIDTH, HEIGHT, policy())
            .expect("analyze second reference");
        assert_eq!(first_changed, second_changed);
        assert_eq!(first_changed.sampled_cells, ADVISOR_CELLS as u32);
        assert_eq!(first_changed.initialized_cells, ADVISOR_CELLS as u32);
        assert_eq!(
            pixels, before,
            "analysis must never mutate or retain pixels"
        );
    }

    #[test]
    fn d2_static_slow_fade_small_area_blackout_and_freeze_stay_bounded() {
        let evaluation_policy = policy();
        let black = solid(0, 0, 0);
        let mut reference = PhotosensitivityCpuReference::default();
        assert_eq!(
            reference
                .analyze_rgba8_srgb(&black, WIDTH, HEIGHT, evaluation_policy)
                .expect("prime")
                .initialized_cells,
            0
        );

        // Static/frozen frames produce no transition, including a paused
        // frame repeated longer than the complete bounded ring.
        for _ in 0..=ADVISOR_RING_CAPACITY {
            let frozen = reference
                .analyze_rgba8_srgb(&black, WIDTH, HEIGHT, evaluation_policy)
                .expect("frozen frame");
            assert_eq!(frozen.affected_cells, 0);
            assert_eq!(frozen.reversal_cells, 0);
        }

        // Small monotonic code-value steps remain below the synthetic policy.
        for value in 1..=8 {
            let slow = reference
                .analyze_rgba8_srgb(
                    &solid(value, value, value),
                    WIDTH,
                    HEIGHT,
                    evaluation_policy,
                )
                .expect("slow fade");
            assert_eq!(slow.affected_cells, 0);
        }

        // A deliberately sub-threshold area is measured but cannot become an
        // event under this policy.
        let small_flash = patch(solid(8, 8, 8), 8, 8, [255, 255, 255, 255]);
        let small = reference
            .analyze_rgba8_srgb(&small_flash, WIDTH, HEIGHT, evaluation_policy)
            .expect("small-area fixture");
        assert!(small.affected_cells < u32::from(evaluation_policy.min_affected_cells));

        // Blackout is observed as a transition, never enacted by the advisor;
        // once frozen at black it immediately becomes a zero-transition frame.
        let blackout = reference
            .analyze_rgba8_srgb(&black, WIDTH, HEIGHT, evaluation_policy)
            .expect("blackout transition");
        assert!(blackout.affected_cells <= ADVISOR_CELLS as u32);
        let held_black = reference
            .analyze_rgba8_srgb(&black, WIDTH, HEIGHT, evaluation_policy)
            .expect("held blackout");
        assert_eq!(held_black.affected_cells, 0);
    }

    #[test]
    fn d2_full_field_reversal_and_saturated_red_are_classified_without_mutation() {
        let evaluation_policy = policy();
        let black = solid(0, 0, 0);
        let white = solid(255, 255, 255);
        let red = solid(255, 0, 0);
        let mut reference = PhotosensitivityCpuReference::default();
        let mut advisor = PhotosensitivityAdvisor::new_evaluation_only(evaluation_policy)
            .expect("evaluation advisor");

        let frames = [&black, &white, &black, &red, &black, &white];
        let mut final_telemetry = None;
        for (index, frame) in frames.into_iter().enumerate() {
            let sample = reference
                .analyze_rgba8_srgb(frame, WIDTH, HEIGHT, evaluation_policy)
                .expect("hostile fixture");
            final_telemetry = Some(
                advisor
                    .observe(index as u64 + 1, index as u64 + 1, sample)
                    .expect("admit hostile fixture"),
            );
        }
        let telemetry = final_telemetry.expect("telemetry");
        assert_eq!(telemetry.level, AdvisorLevel::Elevated);
        assert!(telemetry.window_transition_events >= 4);
        assert!(telemetry.window_reversal_events >= 2);
        assert!(telemetry.window_red_events >= 2);
        assert!(telemetry.window_longest_sustained_ticks >= 4);
        assert_eq!(telemetry.counters.elevated_results, 2);
    }

    #[test]
    fn d2_classifier_rejects_hostile_counters_and_stale_ordering() {
        let evaluation_policy = policy();
        let mut advisor = PhotosensitivityAdvisor::new_evaluation_only(evaluation_policy)
            .expect("evaluation advisor");
        let valid = CompactTransitionCounters {
            sampled_cells: ADVISOR_CELLS as u32,
            initialized_cells: ADVISOR_CELLS as u32,
            ..CompactTransitionCounters::default()
        };
        advisor.observe(1, 1, valid).expect("first sample");
        assert_eq!(
            advisor.observe(1, 2, valid),
            Err(AdvisorSampleError::StaleSequence)
        );
        assert_eq!(
            advisor.observe(2, 1, valid),
            Err(AdvisorSampleError::StaleReferenceTick)
        );
        let malformed = CompactTransitionCounters {
            sampled_cells: ADVISOR_CELLS as u32,
            initialized_cells: 1,
            affected_cells: 2,
            ..CompactTransitionCounters::default()
        };
        assert_eq!(
            advisor.observe(2, 2, malformed),
            Err(AdvisorSampleError::CounterOrdering)
        );
        let telemetry = advisor.telemetry();
        assert_eq!(telemetry.counters.admitted_samples, 1);
        assert_eq!(telemetry.counters.rejected_stale_samples, 2);
        assert_eq!(telemetry.counters.rejected_malformed_samples, 1);
    }

    #[test]
    fn d2_ring_cadence_and_privacy_telemetry_are_bounded() {
        let evaluation_policy = policy();
        let mut advisor = PhotosensitivityAdvisor::new_evaluation_only(evaluation_policy)
            .expect("evaluation advisor");
        let sample = CompactTransitionCounters {
            sampled_cells: ADVISOR_CELLS as u32,
            initialized_cells: ADVISOR_CELLS as u32,
            affected_cells: ADVISOR_CELLS as u32,
            luma_delta_sum_q: ADVISOR_CELLS as u32 * 4_000,
            color_delta_sum_q: ADVISOR_CELLS as u32 * 4_000,
            ..CompactTransitionCounters::default()
        };
        for sequence in 1..=200_u64 {
            advisor
                .observe(sequence, sequence, sample)
                .expect("bounded observation");
        }
        let telemetry = advisor.telemetry();
        assert!(usize::from(telemetry.window_observations) <= ADVISOR_RING_CAPACITY);
        assert_eq!(
            telemetry.window_observations,
            evaluation_policy.window_ticks
        );

        advisor.reset_observations();
        advisor
            .observe(201, 400, sample)
            .expect("first cadence sample");
        advisor
            .observe(202, 403, sample)
            .expect("gapped cadence sample");
        let telemetry = advisor.telemetry();
        assert_eq!(telemetry.counters.missing_reference_ticks, 2);
        assert_eq!(telemetry.window_longest_sustained_ticks, 1);

        let json = serde_json::to_value(telemetry).expect("serialize numeric telemetry");
        let object = json.as_object().expect("telemetry object");
        assert_eq!(
            object.get("algorithm"),
            Some(&serde_json::json!("cos_photosensitivity_evaluation"))
        );
        assert_eq!(object.get("algorithm_version"), Some(&serde_json::json!(1)));
        assert_eq!(
            object.get("availability"),
            Some(&serde_json::json!("deferred_p1_and_review"))
        );
        assert!(
            json.to_string().len() < 1_500,
            "telemetry must stay compact and aggregate-only"
        );
        let mut hostile = serde_json::to_value(sample).expect("aggregate JSON");
        hostile
            .as_object_mut()
            .expect("aggregate object")
            .insert("pixels".to_owned(), serde_json::json!("authored bytes"));
        assert!(
            serde_json::from_value::<CompactTransitionCounters>(hostile).is_err(),
            "aggregate schema must reject fields that could carry authored data"
        );
    }

    #[test]
    fn d2_policy_and_raster_bounds_fail_closed() {
        let mut invalid = policy();
        invalid.transition_threshold_q = 0;
        assert_eq!(
            invalid.validate(),
            Err(AdvisorPolicyError::ZeroTransitionThreshold)
        );
        assert_eq!(
            PhotosensitivityCpuReference::default().analyze_rgba8_srgb(
                &[0, 0, 0, 255],
                1,
                1,
                invalid,
            ),
            Err(AdvisorSampleError::InvalidPolicy(
                AdvisorPolicyError::ZeroTransitionThreshold
            ))
        );
        invalid = policy();
        invalid.window_ticks = ADVISOR_RING_CAPACITY as u16 + 1;
        assert_eq!(
            invalid.validate(),
            Err(AdvisorPolicyError::WindowTicks(
                ADVISOR_RING_CAPACITY as u16 + 1
            ))
        );

        let mut reference = PhotosensitivityCpuReference::default();
        assert_eq!(
            reference.analyze_rgba8_srgb(&[], 0, 1, policy()),
            Err(AdvisorSampleError::EmptyRaster)
        );
        assert_eq!(
            reference.analyze_rgba8_srgb(&[0; 3], 1, 1, policy()),
            Err(AdvisorSampleError::RasterLength)
        );
    }
}
