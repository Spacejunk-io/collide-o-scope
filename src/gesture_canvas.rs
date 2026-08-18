//! Bounded vector canvas etched by the portable gesture event stream.
//!
//! This module owns CPU state and laws only. Like `crate::gesture` it has no
//! `wgpu`, clock, filesystem, or UI dependency: it is handed an already-derived
//! 30 Hz reference-tick address and a slice of recorded events, and it answers
//! with the exact field the shader must reproduce. It is therefore the
//! independent reference the GPU fixtures compare against, written before the
//! WGSL rather than derived from it.
//!
//! The resource contract imitates `crate::motion`: every bound is an
//! independently checked limit with its own typed error, evaluated *before* any
//! allocation, and the byte ledger is reconciled against the resources actually
//! created. The cell layout deliberately breaks format uniformity the same way
//! `renderer/motion.rs` does for small surfaces — a signed `Rg16Float` vector
//! ping-pong pair plus an `Rg8Unorm` gate ping-pong pair, twelve bytes a cell.
//!
//! Two laws carry the whole expressive contract:
//!
//! - `Push` displaces along the stroke direction; `Curl` displaces along its
//!   perpendicular. Both are analytic — a closed-form falloff around the sample
//!   position, never an iterative solve — so live and offline agree exactly.
//! - Overlapping strokes compose in *recorded order*. Each sample blends the
//!   field toward its own etched vector, which does not commute, so reordering
//!   two overlapping strokes is a visible difference rather than a rounding
//!   one. Order is part of the contract.

#![allow(
    dead_code,
    reason = "S3b freezes the vector canvas and its resource ledger before the host, patch, and browser wiring that consume them land"
)]

use std::fmt;

use crate::gesture::{GestureEvent, GestureMode, MAX_GESTURE_DECAY_TICKS};

/// Hard edge bound for either axis of a gesture canvas.
pub const GESTURE_CANVAS_MAX_EDGE: u32 = 2_048;

/// Hard cell bound for one gesture canvas.
pub const GESTURE_CANVAS_MAX_CELLS: u64 = 2_100_000;

/// Frozen per-cell footprint: a signed vector ping-pong pair plus a gate
/// ping-pong pair. It is a declared contract, not a derived convenience — the
/// renderer reconciles its real texture ledger against it.
pub const GESTURE_CANVAS_BYTES_PER_CELL: u64 = 12;

/// Hard byte bound for one canvas's complete working set.
pub const GESTURE_CANVAS_MAX_BYTES: u64 = 32 * 1024 * 1024;

/// At most this many canvases may be live in one composition.
pub const GESTURE_CANVAS_MAX_ACTIVE: u32 = 2;

/// Composition-wide gesture-canvas working-set ceiling.
pub const GESTURE_CANVAS_AGGREGATE_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Every canvas owns one 256-byte uniform slot per encoded pass. The stride is
/// frozen rather than derived from the device so live and export address the
/// same slots on every adapter.
pub const GESTURE_CANVAS_UNIFORM_STRIDE: u64 = 256;

/// Ordered samples one update may carry. A frame that crosses several reference
/// ticks replays every event it crossed, so this is generous; an over-cap
/// update is refused with a typed error rather than silently dropping authored
/// path points.
pub const GESTURE_CANVAS_MAX_SAMPLES_PER_UPDATE: usize = 256;

/// Two `Rg16Float` parities: four bytes each.
const VECTOR_BYTES_PER_CELL: u64 = 8;

/// Two `Rg8Unorm` parities: two bytes each.
const GATE_BYTES_PER_CELL: u64 = 4;

/// One `Rgba16Float` presented donor image: eight bytes a cell.
///
/// This is a *separate* class from the twelve-byte working set above, and it is
/// deliberately not folded into `GESTURE_CANVAS_BYTES_PER_CELL`. The working
/// set is the ping-pong memory the etch passes read and write; the presented
/// image is the single routable donor a composition image tap binds, written
/// once per committed frame from the committed parity. Keeping the two apart
/// means the frozen twelve-byte reconcile still means exactly what it meant,
/// and the presented surface is charged once, under its own name, against the
/// same narrowable ceilings.
pub const GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL: u64 = 8;

/// The declared per-cell footprint and the real format split are the same
/// number by construction; neither can drift without the other.
const _: () = assert!(VECTOR_BYTES_PER_CELL + GATE_BYTES_PER_CELL == GESTURE_CANVAS_BYTES_PER_CELL);

/// Typed failure vocabulary for gesture-canvas admission and update.
///
/// Every limit in the frozen table has its own variant. They are checked
/// independently — one bound being structurally tighter than another at today's
/// cell layout must never be a reason to delete the looser check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureCanvasError {
    InvalidDimensions([u32; 2]),
    CanvasEdge { dimensions: [u32; 2], limit: u32 },
    CellCount { count: u64, limit: u64 },
    BytesPerCell { declared: u64, expected: u64 },
    CanvasBytes { bytes: u64, limit: u64 },
    TooManyCanvases { count: u32, limit: u32 },
    AggregateBytes { bytes: u64, limit: u64 },
    UniformStride { stride: u64, alignment: u32 },
    DecayTickBudget { ticks: u32, limit: u32 },
    TooManySamples { count: usize, limit: usize },
    DeviceTextureDimension { dimensions: [u32; 2], limit: u32 },
    LedgerMismatch { declared: u64, expected: u64 },
    LimitCeiling { requested: u64, ceiling: u64 },
    ArithmeticOverflow,
}

impl fmt::Display for GestureCanvasError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions(dimensions) => write!(
                formatter,
                "gesture canvas dimensions must be non-zero, got {}x{}",
                dimensions[0], dimensions[1]
            ),
            Self::CanvasEdge { dimensions, limit } => write!(
                formatter,
                "gesture canvas {}x{} exceeds the {limit}-cell edge limit",
                dimensions[0], dimensions[1]
            ),
            Self::CellCount { count, limit } => write!(
                formatter,
                "gesture canvas has {count} cells, exceeding {limit}"
            ),
            Self::BytesPerCell { declared, expected } => write!(
                formatter,
                "gesture canvas declares {declared} bytes per cell; the frozen layout is {expected}"
            ),
            Self::CanvasBytes { bytes, limit } => write!(
                formatter,
                "gesture canvas needs {bytes} bytes, exceeding {limit}"
            ),
            Self::TooManyCanvases { count, limit } => write!(
                formatter,
                "gesture plan requests {count} canvases; limit is {limit}"
            ),
            Self::AggregateBytes { bytes, limit } => write!(
                formatter,
                "prepared gesture canvases need {bytes} bytes, exceeding {limit}"
            ),
            Self::UniformStride { stride, alignment } => write!(
                formatter,
                "gesture canvas uniform stride {stride} is not a multiple of the device alignment {alignment}"
            ),
            Self::DecayTickBudget { ticks, limit } => write!(
                formatter,
                "gesture canvas update declares {ticks} decay ticks; the budget is {limit}"
            ),
            Self::TooManySamples { count, limit } => write!(
                formatter,
                "gesture canvas update carries {count} samples; limit is {limit}"
            ),
            Self::DeviceTextureDimension { dimensions, limit } => write!(
                formatter,
                "gesture canvas {}x{} exceeds device texture edge {limit}",
                dimensions[0], dimensions[1]
            ),
            Self::LedgerMismatch { declared, expected } => write!(
                formatter,
                "gesture canvas ledger reports {declared} bytes; the plan admitted {expected}"
            ),
            Self::LimitCeiling { requested, ceiling } => write!(
                formatter,
                "gesture canvas limit {requested} is zero or above the frozen ceiling {ceiling}"
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("gesture canvas resource arithmetic overflow")
            }
        }
    }
}

impl std::error::Error for GestureCanvasError {}

/// Canonical canvas geometry.
///
/// Constructing one is the edge and cell-count admission gate; nothing else in
/// this module accepts a raw width and height, so those two limits cannot be
/// bypassed by a second construction path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GestureCanvasGrid {
    pub width: u32,
    pub height: u32,
    pub cell_count: u64,
}

impl GestureCanvasGrid {
    pub fn new(width: u32, height: u32) -> Result<Self, GestureCanvasError> {
        if width == 0 || height == 0 {
            return Err(GestureCanvasError::InvalidDimensions([width, height]));
        }
        if width > GESTURE_CANVAS_MAX_EDGE || height > GESTURE_CANVAS_MAX_EDGE {
            return Err(GestureCanvasError::CanvasEdge {
                dimensions: [width, height],
                limit: GESTURE_CANVAS_MAX_EDGE,
            });
        }
        let cell_count = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or(GestureCanvasError::ArithmeticOverflow)?;
        if cell_count > GESTURE_CANVAS_MAX_CELLS {
            return Err(GestureCanvasError::CellCount {
                count: cell_count,
                limit: GESTURE_CANVAS_MAX_CELLS,
            });
        }
        Ok(Self {
            width,
            height,
            cell_count,
        })
    }

    pub const fn dimensions(self) -> [u32; 2] {
        [self.width, self.height]
    }

    /// Bytes this grid's complete ping-pong working set occupies.
    pub fn bytes(self) -> Result<u64, GestureCanvasError> {
        self.cell_count
            .checked_mul(GESTURE_CANVAS_BYTES_PER_CELL)
            .ok_or(GestureCanvasError::ArithmeticOverflow)
    }

    /// Bytes this grid's single presented donor image occupies.
    pub fn presented_bytes(self) -> Result<u64, GestureCanvasError> {
        self.cell_count
            .checked_mul(GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL)
            .ok_or(GestureCanvasError::ArithmeticOverflow)
    }

    /// Normalized centre of one cell, in the same space the shader's fullscreen
    /// fragment UV occupies. `y` increases downward, so a canvas coordinate and
    /// a texture coordinate are the same number.
    pub fn cell_position(self, x: u32, y: u32) -> [f32; 2] {
        [
            (x as f32 + 0.5) / self.width as f32,
            (y as f32 + 0.5) / self.height as f32,
        ]
    }

    pub fn index(self, x: u32, y: u32) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        usize::try_from(u64::from(y) * u64::from(self.width) + u64::from(x)).ok()
    }
}

/// Device facts plus the byte ceilings under which a plan is admitted.
///
/// The module ceilings are the frozen table; `bounded` may only narrow them, so
/// an untrusted host can tighten its own budget but never widen it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GestureCanvasLimits {
    pub max_texture_dimension_2d: u32,
    pub min_uniform_buffer_offset_alignment: u32,
    pub max_canvas_bytes: u64,
    pub max_aggregate_bytes: u64,
}

impl GestureCanvasLimits {
    pub const fn device(
        max_texture_dimension_2d: u32,
        min_uniform_buffer_offset_alignment: u32,
    ) -> Self {
        Self {
            max_texture_dimension_2d,
            min_uniform_buffer_offset_alignment,
            max_canvas_bytes: GESTURE_CANVAS_MAX_BYTES,
            max_aggregate_bytes: GESTURE_CANVAS_AGGREGATE_MAX_BYTES,
        }
    }

    /// Narrow the byte ceilings. Zero and any value above the frozen ceiling
    /// are refused, imitating `RecoveryLimits::bounded`.
    pub fn bounded(
        self,
        max_canvas_bytes: u64,
        max_aggregate_bytes: u64,
    ) -> Result<Self, GestureCanvasError> {
        if max_canvas_bytes == 0 || max_canvas_bytes > GESTURE_CANVAS_MAX_BYTES {
            return Err(GestureCanvasError::LimitCeiling {
                requested: max_canvas_bytes,
                ceiling: GESTURE_CANVAS_MAX_BYTES,
            });
        }
        if max_aggregate_bytes == 0 || max_aggregate_bytes > GESTURE_CANVAS_AGGREGATE_MAX_BYTES {
            return Err(GestureCanvasError::LimitCeiling {
                requested: max_aggregate_bytes,
                ceiling: GESTURE_CANVAS_AGGREGATE_MAX_BYTES,
            });
        }
        Ok(Self {
            max_canvas_bytes,
            max_aggregate_bytes,
            ..self
        })
    }
}

/// One canvas an authored composition asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GestureCanvasRequest {
    pub grid: GestureCanvasGrid,
    /// Decay ticks this canvas's update is authored to process. A declared
    /// budget above `MAX_GESTURE_DECAY_TICKS` is refused before allocation; an
    /// *observed* runtime gap clamps instead, because elapsed program time is a
    /// fact rather than an authored number.
    pub decay_ticks: u32,
}

impl GestureCanvasRequest {
    pub const fn new(grid: GestureCanvasGrid) -> Self {
        Self {
            grid,
            decay_ticks: 1,
        }
    }

    pub const fn with_decay_ticks(mut self, decay_ticks: u32) -> Self {
        self.decay_ticks = decay_ticks;
        self
    }
}

/// Byte ledger a renderer reports for the resources it actually created.
///
/// This is the fail-closed reconcile seam: the plan is the canonical
/// accounting, the ledger is what exists, and a disagreement is an error rather
/// than a silently over-allocated composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GestureCanvasLedger {
    pub canvases: u32,
    pub bytes_per_cell: u64,
    pub uniform_stride: u64,
    pub total_bytes: u64,
    /// Per-cell footprint of the presented donor image, reconciled separately
    /// from the working set so a renderer that changed the presented format
    /// fails closed rather than silently binding a different image.
    pub presented_bytes_per_cell: u64,
    pub presented_bytes: u64,
}

/// Canonical admitted resource plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GestureCanvasPlan {
    pub canvases: u32,
    pub cells: u64,
    pub vector_bytes: u64,
    pub gate_bytes: u64,
    pub total_bytes: u64,
    /// Bytes the presented donor images occupy. Charged and limited as its own
    /// class rather than summed into `total_bytes`, so the frozen twelve-byte
    /// working-set reconcile keeps its exact prior meaning.
    pub presented_bytes: u64,
    pub max_canvas_bytes: u64,
    pub max_dimensions: [u32; 2],
    pub decay_ticks: u32,
    pub uniform_stride: u64,
}

impl GestureCanvasPlan {
    /// Admit a composition's canvases, one independent limit at a time and all
    /// of them before a single byte is allocated.
    pub fn preflight(
        requests: &[GestureCanvasRequest],
        limits: GestureCanvasLimits,
    ) -> Result<Self, GestureCanvasError> {
        // The frozen uniform stride must be addressable on this device. It is
        // checked once, up front, because a device that cannot address the
        // stride cannot host any canvas at all.
        let alignment = limits.min_uniform_buffer_offset_alignment.max(1);
        if !GESTURE_CANVAS_UNIFORM_STRIDE.is_multiple_of(u64::from(alignment)) {
            return Err(GestureCanvasError::UniformStride {
                stride: GESTURE_CANVAS_UNIFORM_STRIDE,
                alignment,
            });
        }

        let mut plan = Self {
            uniform_stride: GESTURE_CANVAS_UNIFORM_STRIDE,
            ..Self::default()
        };
        for request in requests {
            if request.decay_ticks > MAX_GESTURE_DECAY_TICKS {
                return Err(GestureCanvasError::DecayTickBudget {
                    ticks: request.decay_ticks,
                    limit: MAX_GESTURE_DECAY_TICKS,
                });
            }
            let dimensions = request.grid.dimensions();
            if dimensions[0] > limits.max_texture_dimension_2d
                || dimensions[1] > limits.max_texture_dimension_2d
            {
                return Err(GestureCanvasError::DeviceTextureDimension {
                    dimensions,
                    limit: limits.max_texture_dimension_2d,
                });
            }

            let vector_bytes = checked_bytes(request.grid.cell_count, VECTOR_BYTES_PER_CELL)?;
            let gate_bytes = checked_bytes(request.grid.cell_count, GATE_BYTES_PER_CELL)?;
            let presented_bytes = checked_bytes(
                request.grid.cell_count,
                GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL,
            )?;
            // The presented donor is an independent class under the same
            // narrowable per-canvas ceiling. It is checked here, before the
            // working set, so a host that narrowed its budget refuses the
            // presented image on its own terms instead of discovering it only
            // through a summed total it cannot attribute.
            if presented_bytes > limits.max_canvas_bytes {
                return Err(GestureCanvasError::CanvasBytes {
                    bytes: presented_bytes,
                    limit: limits.max_canvas_bytes,
                });
            }
            let canvas_bytes = vector_bytes
                .checked_add(gate_bytes)
                .ok_or(GestureCanvasError::ArithmeticOverflow)?;
            if canvas_bytes > limits.max_canvas_bytes {
                return Err(GestureCanvasError::CanvasBytes {
                    bytes: canvas_bytes,
                    limit: limits.max_canvas_bytes,
                });
            }

            plan.canvases = plan
                .canvases
                .checked_add(1)
                .ok_or(GestureCanvasError::ArithmeticOverflow)?;
            if plan.canvases > GESTURE_CANVAS_MAX_ACTIVE {
                return Err(GestureCanvasError::TooManyCanvases {
                    count: plan.canvases,
                    limit: GESTURE_CANVAS_MAX_ACTIVE,
                });
            }
            plan.cells = plan
                .cells
                .checked_add(request.grid.cell_count)
                .ok_or(GestureCanvasError::ArithmeticOverflow)?;
            plan.vector_bytes = plan
                .vector_bytes
                .checked_add(vector_bytes)
                .ok_or(GestureCanvasError::ArithmeticOverflow)?;
            plan.gate_bytes = plan
                .gate_bytes
                .checked_add(gate_bytes)
                .ok_or(GestureCanvasError::ArithmeticOverflow)?;
            plan.presented_bytes = plan
                .presented_bytes
                .checked_add(presented_bytes)
                .ok_or(GestureCanvasError::ArithmeticOverflow)?;
            plan.max_canvas_bytes = plan.max_canvas_bytes.max(canvas_bytes);
            plan.max_dimensions[0] = plan.max_dimensions[0].max(dimensions[0]);
            plan.max_dimensions[1] = plan.max_dimensions[1].max(dimensions[1]);
            plan.decay_ticks = plan.decay_ticks.max(request.decay_ticks);
        }

        if plan.presented_bytes > limits.max_aggregate_bytes {
            return Err(GestureCanvasError::AggregateBytes {
                bytes: plan.presented_bytes,
                limit: limits.max_aggregate_bytes,
            });
        }
        plan.total_bytes = plan
            .vector_bytes
            .checked_add(plan.gate_bytes)
            .ok_or(GestureCanvasError::ArithmeticOverflow)?;
        if plan.total_bytes > limits.max_aggregate_bytes {
            return Err(GestureCanvasError::AggregateBytes {
                bytes: plan.total_bytes,
                limit: limits.max_aggregate_bytes,
            });
        }
        Ok(plan)
    }

    /// Reconcile the resources a renderer actually created against this plan.
    ///
    /// The per-cell layout and the uniform stride are declared contracts rather
    /// than derived numbers, so this is where a renderer that quietly changed a
    /// texture format or an offset alignment fails closed instead of shipping a
    /// canvas the CPU reference no longer describes.
    pub fn reconcile(self, ledger: GestureCanvasLedger) -> Result<(), GestureCanvasError> {
        if ledger.bytes_per_cell != GESTURE_CANVAS_BYTES_PER_CELL {
            return Err(GestureCanvasError::BytesPerCell {
                declared: ledger.bytes_per_cell,
                expected: GESTURE_CANVAS_BYTES_PER_CELL,
            });
        }
        if ledger.uniform_stride != GESTURE_CANVAS_UNIFORM_STRIDE {
            return Err(GestureCanvasError::UniformStride {
                stride: ledger.uniform_stride,
                alignment: 0,
            });
        }
        if ledger.canvases != self.canvases {
            return Err(GestureCanvasError::TooManyCanvases {
                count: ledger.canvases,
                limit: self.canvases,
            });
        }
        if ledger.presented_bytes_per_cell != GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL {
            return Err(GestureCanvasError::BytesPerCell {
                declared: ledger.presented_bytes_per_cell,
                expected: GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL,
            });
        }
        if ledger.total_bytes != self.total_bytes {
            return Err(GestureCanvasError::LedgerMismatch {
                declared: ledger.total_bytes,
                expected: self.total_bytes,
            });
        }
        if ledger.presented_bytes != self.presented_bytes {
            return Err(GestureCanvasError::LedgerMismatch {
                declared: ledger.presented_bytes,
                expected: self.presented_bytes,
            });
        }
        Ok(())
    }
}

fn checked_bytes(count: u64, bytes_per_element: u64) -> Result<u64, GestureCanvasError> {
    count
        .checked_mul(bytes_per_element)
        .ok_or(GestureCanvasError::ArithmeticOverflow)
}

/// Authored, continuously modulatable canvas controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GestureCanvasParams {
    /// Etch radius in normalized canvas space.
    pub radius: f32,
    /// Displacement a full-pressure sample leaves at the stroke centre.
    pub strength: f32,
    /// Fraction of an unheld cell that survives one reference tick.
    pub retention: f32,
}

impl Default for GestureCanvasParams {
    fn default() -> Self {
        Self {
            radius: 0.12,
            strength: 0.5,
            retention: 0.99,
        }
    }
}

impl GestureCanvasParams {
    /// Clamp every authored value. A non-finite input takes the default rather
    /// than a clamped extreme, matching the established Displace law.
    pub fn sanitized(self) -> Self {
        let default = Self::default();
        Self {
            radius: finite_or(self.radius, default.radius).clamp(0.0, 1.0),
            strength: finite_or(self.strength, default.strength).clamp(0.0, 1.0),
            retention: finite_or(self.retention, default.retention).clamp(0.0, 1.0),
        }
    }

    /// A radius or strength of exactly zero cannot move anything, so the whole
    /// etch is an exact bypass and the renderer encodes no etch pass.
    pub fn is_exact_etch_bypass(self) -> bool {
        let sanitized = self.sanitized();
        sanitized.radius <= 0.0 || sanitized.strength <= 0.0
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// One canvas cell.
///
/// `vector` is the signed displacement the consumer applies; `coverage` is how
/// strongly the cell has been etched; `hold` is the per-cell resistance to
/// decay a heavier press leaves behind. The three map exactly onto the frozen
/// twelve-byte layout: `vector` is the `Rg16Float` pair, `coverage` and `hold`
/// are the two `Rg8Unorm` channels.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GestureCell {
    pub vector: [f32; 2],
    pub coverage: f32,
    pub hold: f32,
}

/// Neutral donor encoding for the frozen Displace decode: straight `RG = 0.5`
/// at full coverage means zero displacement.
pub const GESTURE_DONOR_NEUTRAL: f32 = 0.5;

/// Present one cell as the premultiplied RGBA donor image a routed image tap
/// samples.
///
/// The consumer side is already frozen. `displace_node` decodes its donor as
///
/// ```text
/// vector = (premultiplied_rg - 0.5 * alpha) * 2
/// ```
///
/// so the canvas presents straight `RG = vector * 0.5 + 0.5`, premultiplied
/// against the gate's coverage as alpha. Nothing new is invented on the
/// consumer side: no second decode, no node kind, no bind slot.
///
/// Two exactness properties fall out of that arithmetic rather than out of a
/// convention, and both are tested:
///
/// * an un-etched cell presents `(0, 0, 0, 0)` and decodes to exactly zero;
/// * a zero-gate cell decodes to exactly zero *whatever* vector it stores,
///   because the premultiply uses the same alpha the decode subtracts. The
///   established hostile-hidden-RGB law therefore holds here for free.
///
/// Coverage scales the decoded displacement, which is the gate doing its job:
/// a lightly etched cell displaces proportionally less than a fully etched one.
/// Blue is unused and is presented as an explicit zero, never left undefined.
pub fn present_displace_donor(cell: GestureCell) -> [f32; 4] {
    let alpha = finite_or(cell.coverage, 0.0).clamp(0.0, 1.0);
    let straight =
        |value: f32| finite_or(value, 0.0).clamp(-1.0, 1.0) * 0.5 + GESTURE_DONOR_NEUTRAL;
    [
        straight(cell.vector[0]) * alpha,
        straight(cell.vector[1]) * alpha,
        0.0,
        alpha,
    ]
}

/// The exact decode `displace_node` performs, spelled here so the presentation
/// law above can be checked against its real consumer rather than believed.
pub fn decode_displace_donor(premultiplied: [f32; 4]) -> [f32; 2] {
    let alpha = premultiplied[3];
    [
        (premultiplied[0] - GESTURE_DONOR_NEUTRAL * alpha) * 2.0,
        (premultiplied[1] - GESTURE_DONOR_NEUTRAL * alpha) * 2.0,
    ]
}

/// The direction one sample displaces along.
///
/// `Push` follows the stroke; `Curl` follows a quarter turn from it, in the
/// same canvas space the grid uses (x right, y down). Both are exact: nothing
/// here searches, iterates, or consults a neighbour.
pub fn etch_axis(mode: GestureMode, direction: [f32; 2]) -> [f32; 2] {
    match mode {
        GestureMode::Push => direction,
        GestureMode::Curl => [-direction[1], direction[0]],
    }
}

/// Radial falloff around a sample. Exactly one at the centre, exactly zero at
/// and beyond the radius, and flat at the rim so a stroke leaves no visible
/// step. `length` is spelled as the shader spells it so both sides agree.
pub fn etch_falloff(distance: f32, radius: f32) -> f32 {
    if !distance.is_finite() || !radius.is_finite() || radius <= 0.0 || distance >= radius {
        return 0.0;
    }
    let remaining = 1.0 - distance / radius;
    remaining * remaining
}

/// Retention one cell keeps per reference tick.
///
/// `hold` is the resistance a heavier press left behind: a fully held cell
/// retains exactly, an unheld cell retains at the authored rate. Retention is
/// still finite for every cell, because `hold` itself decays at the authored
/// rate — nothing is etched permanently.
pub fn cell_retention_per_tick(hold: f32, retention: f32) -> f32 {
    let hold = finite_or(hold, 0.0).clamp(0.0, 1.0);
    let retention = finite_or(retention, 0.0).clamp(0.0, 1.0);
    retention + (1.0 - retention) * hold
}

/// One decoded sample as the canvas consumes it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GestureEtchSample {
    pub position: [f32; 2],
    pub pressure: f32,
    /// The already-resolved displacement axis, unit length or exactly zero.
    pub axis: [f32; 2],
}

impl GestureEtchSample {
    /// Decode a recorded event. Quantized codes are the only representation the
    /// track holds, so this is the single place the canvas sees floats, and it
    /// derives them from the same accessors offline replay uses.
    pub fn from_event(event: GestureEvent) -> Self {
        Self {
            position: event.position(),
            pressure: event.pressure(),
            axis: etch_axis(event.mode, event.direction()),
        }
    }

    /// A sample with no direction cannot say which way to displace, and a
    /// sample with no pressure cannot displace at all. Either is inert: it is
    /// carried through the plan honestly as a counted sample that changed
    /// nothing, never repaired into an invented direction.
    pub fn is_inert(self) -> bool {
        (self.axis[0] == 0.0 && self.axis[1] == 0.0) || self.pressure <= 0.0
    }
}

/// Bounded, ordered field of cells.
#[derive(Debug, Clone, PartialEq)]
pub struct GestureCanvasField {
    grid: GestureCanvasGrid,
    cells: Vec<GestureCell>,
}

impl GestureCanvasField {
    /// Allocate a cleared field. The grid has already passed the edge and
    /// cell-count gates, so this is the first and only allocation.
    pub fn new(grid: GestureCanvasGrid) -> Result<Self, GestureCanvasError> {
        let cells =
            usize::try_from(grid.cell_count).map_err(|_| GestureCanvasError::ArithmeticOverflow)?;
        Ok(Self {
            grid,
            cells: vec![GestureCell::default(); cells],
        })
    }

    pub const fn grid(&self) -> GestureCanvasGrid {
        self.grid
    }

    pub fn cells(&self) -> &[GestureCell] {
        &self.cells
    }

    pub fn cell(&self, x: u32, y: u32) -> Option<GestureCell> {
        self.grid.index(x, y).map(|index| self.cells[index])
    }

    /// The premultiplied donor texel a routed image tap samples at this cell.
    ///
    /// This is the CPU reference for the presented image, in the same module
    /// that owns the field itself, so a future GPU presenter is compared
    /// against it rather than becoming its own definition.
    pub fn present_displace_donor(&self, x: u32, y: u32) -> Option<[f32; 4]> {
        self.cell(x, y).map(present_displace_donor)
    }

    pub fn clear(&mut self) {
        self.cells.fill(GestureCell::default());
    }

    /// Decay the whole field by a bounded tick count. Retention is applied in
    /// closed form — `retention^ticks`, one operation per cell — so a long gap
    /// never becomes a long loop. The tick count is bounded regardless, because
    /// the budget is the law and the closed form is only the implementation.
    ///
    /// Decay is deliberately *not* composable across a split tick count: the
    /// retention a cell keeps depends on its `hold`, and `hold` itself decays,
    /// so ten ticks at once is not five ticks twice. A sample is indexed by its
    /// reference tick and is therefore grouping-invariant; the decay clock is
    /// indexed by the elapsed gap, so live and offline must derive the same
    /// tick sequence rather than merely the same total. Say this out loud
    /// rather than letting a reader assume an identity that does not hold.
    pub fn decay(&mut self, params: GestureCanvasParams, ticks: u32) {
        if ticks == 0 {
            return;
        }
        let params = params.sanitized();
        let exponent = ticks as f32;
        let hold_retention = params.retention.powf(exponent);
        for cell in &mut self.cells {
            let retention = cell_retention_per_tick(cell.hold, params.retention).powf(exponent);
            cell.vector[0] *= retention;
            cell.vector[1] *= retention;
            cell.coverage *= retention;
            cell.hold *= hold_retention;
        }
    }

    /// Etch one sample into every cell it reaches.
    ///
    /// The vector *blends* toward this sample's target rather than accumulating
    /// onto it. That is what makes composition order-dependent: two overlapping
    /// strokes leave different fields depending on which was drawn first, and a
    /// test proves it.
    pub fn etch(&mut self, sample: GestureEtchSample, params: GestureCanvasParams) {
        if sample.is_inert() {
            return;
        }
        let params = params.sanitized();
        if params.is_exact_etch_bypass() {
            return;
        }
        let pressure = finite_or(sample.pressure, 0.0).clamp(0.0, 1.0);
        let target = [
            sample.axis[0] * params.strength,
            sample.axis[1] * params.strength,
        ];
        for y in 0..self.grid.height {
            for x in 0..self.grid.width {
                let position = self.grid.cell_position(x, y);
                let dx = position[0] - sample.position[0];
                let dy = position[1] - sample.position[1];
                let distance = (dx * dx + dy * dy).sqrt();
                let blend = etch_falloff(distance, params.radius) * pressure;
                if blend <= 0.0 {
                    continue;
                }
                let Some(index) = self.grid.index(x, y) else {
                    continue;
                };
                let cell = &mut self.cells[index];
                cell.vector[0] = cell.vector[0] * (1.0 - blend) + target[0] * blend;
                cell.vector[1] = cell.vector[1] * (1.0 - blend) + target[1] * blend;
                cell.coverage += blend * (1.0 - cell.coverage);
                cell.hold += blend * (pressure - cell.hold);
            }
        }
    }
}

/// Decay ticks one update is allowed to process.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GestureDecayBudget {
    pub ticks: u32,
    /// The observed gap was longer than the budget and was clamped. The
    /// operator sees the clamp; the canvas never silently pretends the gap was
    /// shorter than it was.
    pub clamped: bool,
}

/// Clamp an observed reference-tick gap to the decay budget.
///
/// This imitates `history_ticks_for_delta`'s twenty-four-tick burst clamp: a
/// long gap — a paused program resumed, a slow frame, a seek — settles for the
/// budget rather than billing every tick that passed.
pub const fn decay_ticks_for_gap(gap_ticks: u64) -> GestureDecayBudget {
    if gap_ticks > MAX_GESTURE_DECAY_TICKS as u64 {
        GestureDecayBudget {
            ticks: MAX_GESTURE_DECAY_TICKS,
            clamped: true,
        }
    } else {
        GestureDecayBudget {
            ticks: gap_ticks as u32,
            clamped: false,
        }
    }
}

/// Typed reasons a gesture canvas is reset. Never a boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureCanvasResetCause {
    PatchGeneration,
    ApplyLook,
    SourceCut,
    SourceReplacement,
    Resize,
    BroadRevert,
    ManualClear,
    ExportCancelled,
}

/// Which memories a reset touches.
///
/// `field` is the etched image; `decay_clock` is the last observed reference
/// tick. Splitting them is the whole point: a seek or a source replacement must
/// not erase an operator's authored etch, but it must stop the canvas billing
/// the skipped program time as decay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GestureCanvasResetDomains {
    pub field: bool,
    pub decay_clock: bool,
}

impl GestureCanvasResetDomains {
    pub const NONE: Self = Self {
        field: false,
        decay_clock: false,
    };

    pub const HARD: Self = Self {
        field: true,
        decay_clock: true,
    };

    /// A cut or a source swap rebases the clock and keeps the etch.
    pub const REBASE: Self = Self {
        field: false,
        decay_clock: true,
    };

    pub const fn for_cause(cause: GestureCanvasResetCause) -> Self {
        match cause {
            GestureCanvasResetCause::PatchGeneration
            | GestureCanvasResetCause::ApplyLook
            | GestureCanvasResetCause::Resize
            | GestureCanvasResetCause::BroadRevert
            | GestureCanvasResetCause::ManualClear
            | GestureCanvasResetCause::ExportCancelled => Self::HARD,
            GestureCanvasResetCause::SourceCut | GestureCanvasResetCause::SourceReplacement => {
                Self::REBASE
            }
        }
    }

    pub const fn union(self, other: Self) -> Self {
        Self {
            field: self.field || other.field,
            decay_clock: self.decay_clock || other.decay_clock,
        }
    }
}

/// One frame's authored input.
#[derive(Debug, Clone, Copy)]
pub struct GestureCanvasFrameInput<'a> {
    /// Absolute 30 Hz reference address for this frame, already derived by the
    /// caller from the accepted-frame accumulator live or from the rounded
    /// rational map offline. Wall time never reaches this module.
    pub reference_tick: u64,
    /// Program Freeze holds the canvas: a frozen frame neither decays nor
    /// etches, and neither consumes a reference address.
    pub program_advances: bool,
    pub events: &'a [GestureEvent],
    /// One frame's evaluated canvas controls.
    ///
    /// `None` uses the authored base. `Some` is the modulated copy the one
    /// architectural law requires: a route contributes an offset to a *copy*
    /// of the base state for the current frame and the authored value the
    /// operator can see and save is never touched as a side effect.
    pub evaluated_params: Option<GestureCanvasParams>,
}

/// What one staged frame will do, and what the renderer must encode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GestureCanvasFramePlan {
    pub decay_ticks: u32,
    pub decay_clamped: bool,
    /// Samples that will actually move the field, in recorded order.
    pub applied_samples: u32,
    /// Samples carried and counted but inert — no direction or no pressure.
    pub inert_samples: u32,
    /// Program Freeze held this frame. Nothing changed and no address moved.
    pub held: bool,
}

impl GestureCanvasFramePlan {
    /// A frame that neither decays nor etches. The renderer encodes no pass and
    /// the canvas is byte-identical afterwards.
    pub const fn is_exact_bypass(self) -> bool {
        self.decay_ticks == 0 && self.applied_samples == 0
    }
}

#[derive(Debug, Clone, PartialEq)]
struct GestureCanvasSnapshot {
    field: GestureCanvasField,
    last_tick: Option<u64>,
    generation: u64,
}

/// Transactional canvas state.
///
/// The private staged snapshot is the whole transaction: `stage_frame` asserts
/// none is open and snapshots *before* it changes anything, `commit_staged`
/// drops the snapshot, and `discard_staged` restores it, so a discarded frame
/// leaves no visible change. A reset abandons an open transaction rather than
/// restoring it — a reset is not an undo.
#[derive(Debug, Clone, PartialEq)]
pub struct GestureCanvasState {
    params: GestureCanvasParams,
    field: GestureCanvasField,
    last_tick: Option<u64>,
    generation: u64,
    last_reset: Option<GestureCanvasResetCause>,
    staged: Option<GestureCanvasSnapshot>,
    samples: Vec<GestureEtchSample>,
    /// The plan and the evaluated parameters the open transaction was staged
    /// with. A renderer must encode the frame the CPU reference actually
    /// applied — including this frame's *modulated* radius, strength, and
    /// retention — rather than the authored values a later read of `params()`
    /// would return. Keeping them beside the staged snapshot is what lets live
    /// and offline share one encode call instead of each rebuilding the frame.
    staged_plan: GestureCanvasFramePlan,
    staged_params: GestureCanvasParams,
}

impl GestureCanvasState {
    /// Build a cleared canvas. The sample scratch is sized to the frozen
    /// per-update cap once, here, so a warm frame allocates nothing.
    pub fn new(
        grid: GestureCanvasGrid,
        params: GestureCanvasParams,
    ) -> Result<Self, GestureCanvasError> {
        Ok(Self {
            params: params.sanitized(),
            field: GestureCanvasField::new(grid)?,
            last_tick: None,
            generation: 0,
            last_reset: None,
            staged: None,
            samples: Vec::with_capacity(GESTURE_CANVAS_MAX_SAMPLES_PER_UPDATE),
            staged_plan: GestureCanvasFramePlan::default(),
            staged_params: params.sanitized(),
        })
    }

    pub const fn params(&self) -> GestureCanvasParams {
        self.params
    }

    pub fn set_params(&mut self, params: GestureCanvasParams) {
        self.params = params.sanitized();
    }

    pub const fn field(&self) -> &GestureCanvasField {
        &self.field
    }

    pub const fn grid(&self) -> GestureCanvasGrid {
        self.field.grid
    }

    /// Accepted commits since the last reset. Discard and Program Freeze never
    /// advance it.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn last_tick(&self) -> Option<u64> {
        self.last_tick
    }

    pub const fn last_reset(&self) -> Option<GestureCanvasResetCause> {
        self.last_reset
    }

    pub const fn has_staged_frame(&self) -> bool {
        self.staged.is_some()
    }

    /// The ordered samples the staged frame applied. The renderer encodes one
    /// pass per entry, in this order, so GPU composition order is the recorded
    /// order by construction rather than by convention.
    pub fn staged_samples(&self) -> &[GestureEtchSample] {
        &self.samples
    }

    /// The plan the open transaction produced. Meaningful only while a frame is
    /// staged; a renderer pairs it with `staged_samples` and `staged_params`.
    pub const fn staged_plan(&self) -> GestureCanvasFramePlan {
        self.staged_plan
    }

    /// The evaluated parameters the open transaction was staged with. This is
    /// the modulated copy for the frame, never the authored base.
    pub const fn staged_params(&self) -> GestureCanvasParams {
        self.staged_params
    }

    /// Stage one frame.
    ///
    /// Over-cap sample counts are refused *before* the snapshot is taken, so a
    /// refused frame leaves no open transaction. Everything else is applied to
    /// the snapshot-protected field: decay first, then every sample in recorded
    /// order.
    pub fn stage_frame(
        &mut self,
        input: GestureCanvasFrameInput<'_>,
    ) -> Result<GestureCanvasFramePlan, GestureCanvasError> {
        assert!(
            self.staged.is_none(),
            "gesture canvas frame already staged; commit or discard it first"
        );
        if input.events.len() > GESTURE_CANVAS_MAX_SAMPLES_PER_UPDATE {
            return Err(GestureCanvasError::TooManySamples {
                count: input.events.len(),
                limit: GESTURE_CANVAS_MAX_SAMPLES_PER_UPDATE,
            });
        }

        self.staged = Some(GestureCanvasSnapshot {
            field: self.field.clone(),
            last_tick: self.last_tick,
            generation: self.generation,
        });
        self.samples.clear();
        let params = input
            .evaluated_params
            .map_or(self.params, GestureCanvasParams::sanitized);
        self.staged_params = params;

        if !input.program_advances {
            self.staged_plan = GestureCanvasFramePlan {
                held: true,
                ..GestureCanvasFramePlan::default()
            };
            return Ok(self.staged_plan);
        }

        // A first observation establishes the clock rather than billing every
        // tick since the program began.
        let gap = self
            .last_tick
            .map_or(0, |last| input.reference_tick.saturating_sub(last));
        let budget = decay_ticks_for_gap(gap);
        self.field.decay(params, budget.ticks);

        let mut plan = GestureCanvasFramePlan {
            decay_ticks: budget.ticks,
            decay_clamped: budget.clamped,
            ..GestureCanvasFramePlan::default()
        };
        for event in input.events {
            let sample = GestureEtchSample::from_event(*event);
            if sample.is_inert() || params.is_exact_etch_bypass() {
                plan.inert_samples = plan.inert_samples.saturating_add(1);
                continue;
            }
            self.field.etch(sample, params);
            self.samples.push(sample);
            plan.applied_samples = plan.applied_samples.saturating_add(1);
        }
        self.last_tick = Some(input.reference_tick);
        self.staged_plan = plan;
        Ok(plan)
    }

    /// Accept the staged frame. This is the only thing that advances the
    /// generation.
    pub fn commit_staged(&mut self) {
        if self.staged.take().is_some() {
            self.generation = self.generation.saturating_add(1);
        }
    }

    /// Roll the staged frame back. A discarded frame produces no visible
    /// change: the field, the decay clock, and the generation all return to
    /// exactly what they were before staging.
    pub fn discard_staged(&mut self) {
        if let Some(snapshot) = self.staged.take() {
            self.field = snapshot.field;
            self.last_tick = snapshot.last_tick;
            self.generation = snapshot.generation;
        }
        self.samples.clear();
    }

    pub fn reset_for(&mut self, cause: GestureCanvasResetCause) {
        self.apply_reset_domains(GestureCanvasResetDomains::for_cause(cause));
        self.last_reset = Some(cause);
    }

    /// Apply a reset domain set. An open transaction is *abandoned*, not
    /// restored — a reset that rewound to the pre-frame state would resurrect
    /// exactly the memory the reset exists to clear.
    pub fn apply_reset_domains(&mut self, domains: GestureCanvasResetDomains) {
        if domains == GestureCanvasResetDomains::NONE {
            return;
        }
        self.staged = None;
        self.samples.clear();
        if domains.field {
            self.field.clear();
            self.generation = 0;
        }
        if domains.decay_clock {
            self.last_tick = None;
        }
    }

    /// Resize the canvas. A resize is a hard reset: the etch is authored in
    /// normalized canvas space but its sampled resolution is part of what was
    /// authored, and silently resampling would invent cells nobody etched.
    ///
    /// The new geometry goes through `GestureCanvasGrid::new`, so a resize
    /// cannot smuggle an out-of-range grid past the edge and cell-count gates,
    /// and a refused resize changes nothing at all.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), GestureCanvasError> {
        let field = GestureCanvasField::new(GestureCanvasGrid::new(width, height)?)?;
        self.field = field;
        self.reset_for(GestureCanvasResetCause::Resize);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gesture::{GesturePhase, MAX_ACTIVE_STROKES};

    fn grid(width: u32, height: u32) -> GestureCanvasGrid {
        GestureCanvasGrid::new(width, height).expect("admitted grid")
    }

    fn limits() -> GestureCanvasLimits {
        GestureCanvasLimits::device(GESTURE_CANVAS_MAX_EDGE, 256)
    }

    /// Q16 lattice step. The analytic fixtures below author exact positions so
    /// the closed-form falloff can be checked exactly; the quantized ingest
    /// path is a separately proven law in `crate::gesture`, and the one fixture
    /// that drives it end to end asserts against this step rather than zero.
    const Q16_STEP: f32 = 1.0 / 65_535.0;

    /// An exact, already-decoded sample at full pressure. `direction` must
    /// already be unit length; the axis law is applied here so a fixture reads
    /// as "this stroke, this mode".
    fn sample(position: [f32; 2], direction: [f32; 2], mode: GestureMode) -> GestureEtchSample {
        GestureEtchSample {
            position,
            pressure: 1.0,
            axis: etch_axis(mode, direction),
        }
    }

    fn event(
        stroke: u8,
        mode: GestureMode,
        position: [f32; 2],
        pressure: f32,
        direction: [f32; 2],
    ) -> GestureEvent {
        GestureEvent::quantized(
            stroke,
            GesturePhase::Move,
            mode,
            position,
            pressure,
            direction,
        )
    }

    fn authored_params(radius: f32, strength: f32, retention: f32) -> GestureCanvasParams {
        GestureCanvasParams {
            radius,
            strength,
            retention,
        }
    }

    #[test]
    fn frozen_gesture_canvas_bounds_match_the_specification_table() {
        assert_eq!(GESTURE_CANVAS_MAX_EDGE, 2_048);
        assert_eq!(GESTURE_CANVAS_MAX_CELLS, 2_100_000);
        assert_eq!(GESTURE_CANVAS_BYTES_PER_CELL, 12);
        assert_eq!(GESTURE_CANVAS_MAX_BYTES, 33_554_432);
        assert_eq!(GESTURE_CANVAS_MAX_ACTIVE, 2);
        assert_eq!(GESTURE_CANVAS_AGGREGATE_MAX_BYTES, 67_108_864);
        assert_eq!(GESTURE_CANVAS_UNIFORM_STRIDE, 256);
        assert_eq!(MAX_GESTURE_DECAY_TICKS, 4_096);
        // The twelve bytes are the two ping-pong pairs and nothing else.
        assert_eq!(VECTOR_BYTES_PER_CELL, 8);
        assert_eq!(GATE_BYTES_PER_CELL, 4);
        assert_eq!(
            VECTOR_BYTES_PER_CELL + GATE_BYTES_PER_CELL,
            GESTURE_CANVAS_BYTES_PER_CELL
        );
        // The presented donor is a separate class and stays out of the frozen
        // twelve. One `Rgba16Float` image, written once per committed frame.
        assert_eq!(GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL, 8);
        assert_ne!(
            GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL,
            GESTURE_CANVAS_BYTES_PER_CELL
        );
    }

    #[test]
    fn every_independent_canvas_limit_is_rejected_one_unit_over() {
        // Grid edge.
        assert_eq!(
            GestureCanvasGrid::new(GESTURE_CANVAS_MAX_EDGE + 1, 16),
            Err(GestureCanvasError::CanvasEdge {
                dimensions: [GESTURE_CANVAS_MAX_EDGE + 1, 16],
                limit: GESTURE_CANVAS_MAX_EDGE,
            })
        );
        assert_eq!(
            GestureCanvasGrid::new(16, GESTURE_CANVAS_MAX_EDGE + 1),
            Err(GestureCanvasError::CanvasEdge {
                dimensions: [16, GESTURE_CANVAS_MAX_EDGE + 1],
                limit: GESTURE_CANVAS_MAX_EDGE,
            })
        );
        assert!(GestureCanvasGrid::new(GESTURE_CANVAS_MAX_EDGE, 1).is_ok());

        // Cell count, independently of the edge: both edges are legal here.
        let over_cells = GestureCanvasGrid::new(2_048, 1_026);
        assert_eq!(
            over_cells,
            Err(GestureCanvasError::CellCount {
                count: 2_101_248,
                limit: GESTURE_CANVAS_MAX_CELLS,
            })
        );
        assert!(GestureCanvasGrid::new(2_048, 1_025).is_ok());

        // Zero is refused separately from the ceiling.
        assert_eq!(
            GestureCanvasGrid::new(0, 16),
            Err(GestureCanvasError::InvalidDimensions([0, 16]))
        );

        // Bytes per cell, through the fail-closed ledger reconcile.
        let plan = GestureCanvasPlan::preflight(&[GestureCanvasRequest::new(grid(8, 8))], limits())
            .expect("admitted plan");
        let ledger = GestureCanvasLedger {
            canvases: 1,
            bytes_per_cell: GESTURE_CANVAS_BYTES_PER_CELL,
            uniform_stride: GESTURE_CANVAS_UNIFORM_STRIDE,
            total_bytes: plan.total_bytes,
            presented_bytes_per_cell: GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL,
            presented_bytes: plan.presented_bytes,
        };
        assert_eq!(plan.reconcile(ledger), Ok(()));
        assert_eq!(
            plan.reconcile(GestureCanvasLedger {
                bytes_per_cell: GESTURE_CANVAS_BYTES_PER_CELL - 1,
                ..ledger
            }),
            Err(GestureCanvasError::BytesPerCell {
                declared: 11,
                expected: 12,
            })
        );

        // Bytes per canvas. At the frozen twelve-byte layout the cell cap binds
        // first (2,100,000 x 12 = 25,200,000 < 33,554,432), so the byte cap is
        // proven through the narrowable ceiling it actually governs. Both facts
        // are asserted rather than one being deleted for being subsumed.
        const {
            assert!(
                GESTURE_CANVAS_MAX_CELLS * GESTURE_CANVAS_BYTES_PER_CELL < GESTURE_CANVAS_MAX_BYTES
            );
        }
        let canvas = grid(64, 64);
        let bytes = canvas.bytes().expect("canvas bytes");
        let narrowed = limits()
            .bounded(bytes - 1, GESTURE_CANVAS_AGGREGATE_MAX_BYTES)
            .expect("narrowed canvas ceiling");
        assert_eq!(
            GestureCanvasPlan::preflight(&[GestureCanvasRequest::new(canvas)], narrowed),
            Err(GestureCanvasError::CanvasBytes {
                bytes,
                limit: bytes - 1,
            })
        );
        assert!(GestureCanvasPlan::preflight(
            &[GestureCanvasRequest::new(canvas)],
            limits()
                .bounded(bytes, GESTURE_CANVAS_AGGREGATE_MAX_BYTES)
                .unwrap()
        )
        .is_ok());

        // Active canvases.
        let request = GestureCanvasRequest::new(grid(8, 8));
        assert!(GestureCanvasPlan::preflight(&[request; 2], limits()).is_ok());
        assert_eq!(
            GestureCanvasPlan::preflight(&[request; 3], limits()),
            Err(GestureCanvasError::TooManyCanvases {
                count: 3,
                limit: GESTURE_CANVAS_MAX_ACTIVE,
            })
        );

        // Aggregate bytes, again through the ceiling it governs.
        let pair_bytes = bytes * 2;
        let narrowed = limits()
            .bounded(GESTURE_CANVAS_MAX_BYTES, pair_bytes - 1)
            .expect("narrowed aggregate ceiling");
        assert_eq!(
            GestureCanvasPlan::preflight(&[GestureCanvasRequest::new(canvas); 2], narrowed),
            Err(GestureCanvasError::AggregateBytes {
                bytes: pair_bytes,
                limit: pair_bytes - 1,
            })
        );

        // Uniform stride: a device whose offset alignment does not divide the
        // frozen stride cannot host a canvas at all.
        assert_eq!(
            GestureCanvasPlan::preflight(
                &[request],
                GestureCanvasLimits::device(GESTURE_CANVAS_MAX_EDGE, 512)
            ),
            Err(GestureCanvasError::UniformStride {
                stride: GESTURE_CANVAS_UNIFORM_STRIDE,
                alignment: 512,
            })
        );
        assert!(GestureCanvasPlan::preflight(
            &[request],
            GestureCanvasLimits::device(GESTURE_CANVAS_MAX_EDGE, 64)
        )
        .is_ok());

        // Decay ticks per update.
        assert!(GestureCanvasPlan::preflight(
            &[request.with_decay_ticks(MAX_GESTURE_DECAY_TICKS)],
            limits()
        )
        .is_ok());
        assert_eq!(
            GestureCanvasPlan::preflight(
                &[request.with_decay_ticks(MAX_GESTURE_DECAY_TICKS + 1)],
                limits()
            ),
            Err(GestureCanvasError::DecayTickBudget {
                ticks: MAX_GESTURE_DECAY_TICKS + 1,
                limit: MAX_GESTURE_DECAY_TICKS,
            })
        );

        // A device edge below the module edge is its own independent refusal.
        assert_eq!(
            GestureCanvasPlan::preflight(
                &[GestureCanvasRequest::new(grid(2_048, 8))],
                GestureCanvasLimits::device(1_024, 256)
            ),
            Err(GestureCanvasError::DeviceTextureDimension {
                dimensions: [2_048, 8],
                limit: 1_024,
            })
        );
    }

    #[test]
    fn a_caller_supplied_byte_ceiling_may_only_narrow_and_never_widen() {
        let base = limits();
        assert_eq!(
            base.bounded(
                GESTURE_CANVAS_MAX_BYTES + 1,
                GESTURE_CANVAS_AGGREGATE_MAX_BYTES
            ),
            Err(GestureCanvasError::LimitCeiling {
                requested: GESTURE_CANVAS_MAX_BYTES + 1,
                ceiling: GESTURE_CANVAS_MAX_BYTES,
            })
        );
        assert_eq!(
            base.bounded(
                GESTURE_CANVAS_MAX_BYTES,
                GESTURE_CANVAS_AGGREGATE_MAX_BYTES + 1
            ),
            Err(GestureCanvasError::LimitCeiling {
                requested: GESTURE_CANVAS_AGGREGATE_MAX_BYTES + 1,
                ceiling: GESTURE_CANVAS_AGGREGATE_MAX_BYTES,
            })
        );
        assert_eq!(
            base.bounded(0, GESTURE_CANVAS_AGGREGATE_MAX_BYTES),
            Err(GestureCanvasError::LimitCeiling {
                requested: 0,
                ceiling: GESTURE_CANVAS_MAX_BYTES,
            })
        );
        assert_eq!(
            base.bounded(GESTURE_CANVAS_MAX_BYTES, 0),
            Err(GestureCanvasError::LimitCeiling {
                requested: 0,
                ceiling: GESTURE_CANVAS_AGGREGATE_MAX_BYTES,
            })
        );
        let narrowed = base.bounded(1_024, 2_048).expect("narrowed");
        assert_eq!(narrowed.max_canvas_bytes, 1_024);
        assert_eq!(narrowed.max_aggregate_bytes, 2_048);
        assert_eq!(
            narrowed.max_texture_dimension_2d,
            base.max_texture_dimension_2d
        );
    }

    #[test]
    fn the_admitted_plan_and_the_actual_ledger_reconcile_or_fail_closed() {
        let plan = GestureCanvasPlan::preflight(
            &[
                GestureCanvasRequest::new(grid(32, 16)),
                GestureCanvasRequest::new(grid(8, 8)),
            ],
            limits(),
        )
        .expect("admitted plan");
        assert_eq!(plan.canvases, 2);
        assert_eq!(plan.cells, 32 * 16 + 8 * 8);
        assert_eq!(plan.vector_bytes, plan.cells * VECTOR_BYTES_PER_CELL);
        assert_eq!(plan.gate_bytes, plan.cells * GATE_BYTES_PER_CELL);
        assert_eq!(plan.total_bytes, plan.cells * GESTURE_CANVAS_BYTES_PER_CELL);
        assert_eq!(
            plan.presented_bytes,
            plan.cells * GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL
        );
        assert_eq!(plan.max_dimensions, [32, 16]);
        assert_eq!(plan.uniform_stride, GESTURE_CANVAS_UNIFORM_STRIDE);

        let ledger = GestureCanvasLedger {
            canvases: plan.canvases,
            bytes_per_cell: GESTURE_CANVAS_BYTES_PER_CELL,
            uniform_stride: GESTURE_CANVAS_UNIFORM_STRIDE,
            total_bytes: plan.total_bytes,
            presented_bytes_per_cell: GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL,
            presented_bytes: plan.presented_bytes,
        };
        assert_eq!(plan.reconcile(ledger), Ok(()));
        // The presented donor image is reconciled on its own terms: a changed
        // presented format and a changed presented byte total each fail closed
        // without touching the frozen twelve-byte working-set claim.
        assert_eq!(
            plan.reconcile(GestureCanvasLedger {
                presented_bytes_per_cell: GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL / 2,
                ..ledger
            }),
            Err(GestureCanvasError::BytesPerCell {
                declared: 4,
                expected: GESTURE_CANVAS_PRESENTED_BYTES_PER_CELL,
            })
        );
        assert_eq!(
            plan.reconcile(GestureCanvasLedger {
                presented_bytes: plan.presented_bytes + 8,
                ..ledger
            }),
            Err(GestureCanvasError::LedgerMismatch {
                declared: plan.presented_bytes + 8,
                expected: plan.presented_bytes,
            })
        );
        assert_eq!(
            plan.reconcile(GestureCanvasLedger {
                uniform_stride: 128,
                ..ledger
            }),
            Err(GestureCanvasError::UniformStride {
                stride: 128,
                alignment: 0,
            })
        );
        assert_eq!(
            plan.reconcile(GestureCanvasLedger {
                canvases: 1,
                ..ledger
            }),
            Err(GestureCanvasError::TooManyCanvases { count: 1, limit: 2 })
        );
        assert_eq!(
            plan.reconcile(GestureCanvasLedger {
                total_bytes: plan.total_bytes + 12,
                ..ledger
            }),
            Err(GestureCanvasError::LedgerMismatch {
                declared: plan.total_bytes + 12,
                expected: plan.total_bytes,
            })
        );
        // An empty composition admits nothing at all.
        let empty = GestureCanvasPlan::preflight(&[], limits()).expect("empty plan");
        assert_eq!(empty.canvases, 0);
        assert_eq!(empty.total_bytes, 0);
        assert_eq!(empty.presented_bytes, 0);
    }

    #[test]
    fn push_displaces_along_the_stroke_and_curl_along_its_perpendicular() {
        // The axis law itself, stated analytically for the four cardinal
        // directions plus a diagonal.
        for (direction, push, curl) in [
            ([1.0_f32, 0.0_f32], [1.0_f32, 0.0_f32], [0.0_f32, 1.0_f32]),
            ([0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]),
            ([-1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]),
            ([0.0, -1.0], [0.0, -1.0], [1.0, 0.0]),
        ] {
            assert_eq!(etch_axis(GestureMode::Push, direction), push);
            assert_eq!(etch_axis(GestureMode::Curl, direction), curl);
            // The perpendicular is exactly a quarter turn: same length, zero
            // dot product with the stroke.
            let dot = push[0] * curl[0] + push[1] * curl[1];
            assert!(dot.abs() < 1.0e-6, "curl axis is not perpendicular: {dot}");
        }

        // And the same law observed in the field. One centred sample on a grid
        // whose centre cell sits exactly at the sample position.
        let params = authored_params(0.5, 0.4, 1.0);
        let centre = [0.5_f32, 0.5_f32];
        let mut pushed = GestureCanvasField::new(grid(3, 3)).expect("field");
        pushed.etch(sample(centre, [1.0, 0.0], GestureMode::Push), params);
        let mut curled = GestureCanvasField::new(grid(3, 3)).expect("field");
        curled.etch(sample(centre, [1.0, 0.0], GestureMode::Curl), params);

        // Centre cell: falloff is exactly one, pressure exactly one, so the
        // blend is total and the cell holds the target vector exactly.
        let push_centre = pushed.cell(1, 1).expect("centre");
        assert!((push_centre.vector[0] - 0.4).abs() < 1.0e-6);
        assert!(push_centre.vector[1].abs() < 1.0e-6);
        assert!((push_centre.coverage - 1.0).abs() < 1.0e-6);
        assert!((push_centre.hold - 1.0).abs() < 1.0e-6);

        let curl_centre = curled.cell(1, 1).expect("centre");
        assert!(curl_centre.vector[0].abs() < 1.0e-6);
        assert!((curl_centre.vector[1] - 0.4).abs() < 1.0e-6);

        // Off-centre cell: the analytic falloff is checkable in closed form.
        let position = pushed.grid().cell_position(2, 1);
        let distance =
            ((position[0] - centre[0]).powi(2) + (position[1] - centre[1]).powi(2)).sqrt();
        let blend = etch_falloff(distance, params.radius);
        let expected = 0.4 * blend;
        let off = pushed.cell(2, 1).expect("off centre");
        assert!(
            (off.vector[0] - expected).abs() < 1.0e-6,
            "expected {expected}, got {}",
            off.vector[0]
        );
        assert!((off.coverage - blend).abs() < 1.0e-6);

        // Curl at the same cell is the same magnitude on the other axis.
        let curl_off = curled.cell(2, 1).expect("off centre");
        assert!(curl_off.vector[0].abs() < 1.0e-6);
        assert!((curl_off.vector[1] - expected).abs() < 1.0e-6);
    }

    #[test]
    fn the_analytic_falloff_is_one_at_the_centre_and_exactly_zero_at_the_radius() {
        assert_eq!(etch_falloff(0.0, 0.25), 1.0);
        assert_eq!(etch_falloff(0.25, 0.25), 0.0);
        assert_eq!(etch_falloff(0.5, 0.25), 0.0);
        assert_eq!(etch_falloff(0.125, 0.25), 0.25);
        // Hostile inputs are inert rather than infinite.
        assert_eq!(etch_falloff(f32::NAN, 0.25), 0.0);
        assert_eq!(etch_falloff(0.1, f32::NAN), 0.0);
        assert_eq!(etch_falloff(0.1, 0.0), 0.0);
        assert_eq!(etch_falloff(0.1, -0.5), 0.0);
    }

    #[test]
    fn an_undirected_or_unpressed_sample_is_inert_and_never_invents_a_direction() {
        let undirected = sample([0.5, 0.5], [0.0, 0.0], GestureMode::Push);
        assert!(undirected.is_inert());
        let unpressed =
            GestureEtchSample::from_event(event(0, GestureMode::Push, [0.5, 0.5], 0.0, [1.0, 0.0]));
        assert!(unpressed.is_inert());

        let mut field = GestureCanvasField::new(grid(4, 4)).expect("field");
        let before = field.clone();
        field.etch(undirected, GestureCanvasParams::default());
        field.etch(unpressed, GestureCanvasParams::default());
        assert_eq!(field, before);

        // A zero radius or zero strength is an exact etch bypass too.
        assert!(authored_params(0.0, 0.5, 0.99).is_exact_etch_bypass());
        assert!(authored_params(0.25, 0.0, 0.99).is_exact_etch_bypass());
        assert!(!GestureCanvasParams::default().is_exact_etch_bypass());
        let mut bypassed = GestureCanvasField::new(grid(4, 4)).expect("field");
        let before = bypassed.clone();
        bypassed.etch(
            sample([0.5, 0.5], [1.0, 0.0], GestureMode::Push),
            authored_params(0.0, 0.5, 0.99),
        );
        assert_eq!(bypassed, before);
    }

    #[test]
    fn hostile_authored_canvas_parameters_sanitize_to_the_default_rather_than_an_extreme() {
        let hostile = GestureCanvasParams {
            radius: f32::NAN,
            strength: f32::INFINITY,
            retention: f32::NEG_INFINITY,
        }
        .sanitized();
        assert_eq!(hostile, GestureCanvasParams::default());
        let clamped = authored_params(4.0, -2.0, 7.0).sanitized();
        assert_eq!(clamped.radius, 1.0);
        assert_eq!(clamped.strength, 0.0);
        assert_eq!(clamped.retention, 1.0);
    }

    #[test]
    fn two_overlapping_strokes_compose_in_recorded_order_and_reordering_changes_the_field() {
        let params = authored_params(0.5, 0.6, 1.0);
        let first = sample([0.45, 0.5], [1.0, 0.0], GestureMode::Push);
        let second = sample([0.55, 0.5], [0.0, 1.0], GestureMode::Push);

        let mut forward = GestureCanvasField::new(grid(5, 5)).expect("field");
        forward.etch(first, params);
        forward.etch(second, params);

        let mut reversed = GestureCanvasField::new(grid(5, 5)).expect("field");
        reversed.etch(second, params);
        reversed.etch(first, params);

        assert_ne!(
            forward, reversed,
            "overlapping strokes must not commute; order is part of the contract"
        );

        // The difference is a real one at the overlap, not a rounding artefact.
        let a = forward.cell(2, 2).expect("overlap");
        let b = reversed.cell(2, 2).expect("overlap");
        let separation = (a.vector[0] - b.vector[0]).abs() + (a.vector[1] - b.vector[1]).abs();
        assert!(
            separation > 0.05,
            "reordered overlap differs by only {separation}"
        );

        // Coverage is an over-accumulation and is deliberately order-free, so
        // the vector alone carries the order. Saying so here keeps a later
        // reader from "fixing" one of the two.
        assert!((a.coverage - b.coverage).abs() < 1.0e-6);

        // Two strokes that do not overlap compose identically either way.
        let far_a = sample([0.1, 0.1], [1.0, 0.0], GestureMode::Push);
        let far_b = sample([0.9, 0.9], [0.0, 1.0], GestureMode::Push);
        let tight = params.sanitized();
        let tight = GestureCanvasParams {
            radius: 0.1,
            ..tight
        };
        let mut ab = GestureCanvasField::new(grid(5, 5)).expect("field");
        ab.etch(far_a, tight);
        ab.etch(far_b, tight);
        let mut ba = GestureCanvasField::new(grid(5, 5)).expect("field");
        ba.etch(far_b, tight);
        ba.etch(far_a, tight);
        assert_eq!(ab, ba);
    }

    #[test]
    fn edge_and_corner_cells_etch_from_a_sample_pinned_to_the_canvas_boundary() {
        let params = authored_params(0.5, 0.5, 1.0);
        let canvas = grid(4, 4);

        // A sample at the exact canvas origin. The nearest cell centre is
        // (0.125, 0.125), so the corner is reached and the opposite corner is
        // not — the field neither wraps nor clamps a stroke back inside.
        let mut corner = GestureCanvasField::new(canvas).expect("field");
        corner.etch(sample([0.0, 0.0], [1.0, 0.0], GestureMode::Push), params);
        let near = corner.cell(0, 0).expect("corner");
        let far = corner.cell(3, 3).expect("opposite corner");
        let distance = (0.125_f32 * 0.125 + 0.125 * 0.125).sqrt();
        let expected = 0.5 * etch_falloff(distance, params.radius);
        assert!((near.vector[0] - expected).abs() < 1.0e-6);
        assert_eq!(far, GestureCell::default());

        // Each of the four edges, driven from its own boundary sample.
        for (position, cell) in [
            ([0.5_f32, 0.0_f32], (1_u32, 0_u32)),
            ([0.5, 1.0], (1, 3)),
            ([0.0, 0.5], (0, 1)),
            ([1.0, 0.5], (3, 1)),
        ] {
            let mut field = GestureCanvasField::new(canvas).expect("field");
            field.etch(sample(position, [1.0, 0.0], GestureMode::Push), params);
            let etched = field.cell(cell.0, cell.1).expect("edge cell");
            assert!(
                etched.coverage > 0.0,
                "boundary sample {position:?} never reached cell {cell:?}"
            );
        }

        // A sample outside the unit canvas cannot exist: the quantizer clamps
        // it to the boundary rather than addressing a cell that is not there.
        let outside = GestureEtchSample::from_event(event(
            0,
            GestureMode::Push,
            [4.0, -3.0],
            1.0,
            [1.0, 0.0],
        ));
        assert_eq!(outside.position, [1.0, 0.0]);
    }

    #[test]
    fn a_decoded_event_reaches_the_canvas_on_the_q16_lattice_with_an_exact_unit_axis() {
        let decoded = GestureEtchSample::from_event(event(
            0,
            GestureMode::Curl,
            [0.25, 0.75],
            0.5,
            [0.6, 0.8],
        ));
        let exact = sample([0.25, 0.75], [0.6, 0.8], GestureMode::Curl);
        assert!((decoded.position[0] - exact.position[0]).abs() <= Q16_STEP);
        assert!((decoded.position[1] - exact.position[1]).abs() <= Q16_STEP);
        assert!((decoded.pressure - 0.5).abs() <= Q16_STEP);
        // The stored direction is renormalized on decode, so the canvas always
        // receives a unit axis and never a quantization-shortened one.
        let length = decoded.axis[0].hypot(decoded.axis[1]);
        assert!((length - 1.0).abs() < 1.0e-4, "axis length {length}");
        assert!((decoded.axis[0] - exact.axis[0]).abs() < 1.0e-4);
        assert!((decoded.axis[1] - exact.axis[1]).abs() < 1.0e-4);
    }

    #[test]
    fn decay_is_finite_holds_a_pressed_mark_longer_and_clamps_a_long_gap_to_the_budget() {
        let params = authored_params(0.4, 1.0, 0.9);

        // A light mark and a heavy mark, side by side.
        let mut field = GestureCanvasField::new(grid(2, 1)).expect("field");
        field.etch(
            GestureEtchSample {
                position: field.grid().cell_position(0, 0),
                pressure: 0.25,
                axis: [1.0, 0.0],
            },
            params,
        );
        field.etch(
            GestureEtchSample {
                position: field.grid().cell_position(1, 0),
                pressure: 1.0,
                axis: [1.0, 0.0],
            },
            params,
        );
        let light_before = field.cell(0, 0).expect("light");
        let heavy_before = field.cell(1, 0).expect("heavy");
        assert!(heavy_before.hold > light_before.hold);

        field.decay(params, 10);
        let light = field.cell(0, 0).expect("light");
        let heavy = field.cell(1, 0).expect("heavy");
        let light_kept = light.vector[0] / light_before.vector[0];
        let heavy_kept = heavy.vector[0] / heavy_before.vector[0];
        assert!(
            heavy_kept > light_kept,
            "a pressed mark must resist decay: heavy {heavy_kept} vs light {light_kept}"
        );

        // Retention is finite for every cell: hold decays too, so a held mark
        // eventually releases rather than being etched permanently.
        field.decay(params, MAX_GESTURE_DECAY_TICKS);
        let released = field.cell(1, 0).expect("heavy");
        assert!(released.hold < 1.0e-6, "hold never released: {released:?}");
        assert!(released.coverage < 1.0e-6);
        assert!(released.vector[0].abs() < 1.0e-6);

        // The budget clamp itself. A high retention keeps the difference
        // between the clamped and unclamped tick counts plainly observable.
        let held = authored_params(0.4, 1.0, 0.9995);
        let mut budgeted = GestureCanvasField::new(grid(1, 1)).expect("field");
        budgeted.etch(
            GestureEtchSample {
                position: [0.5, 0.5],
                pressure: 0.0625,
                axis: [1.0, 0.0],
            },
            held,
        );
        let mut unbudgeted = budgeted.clone();
        let gap = decay_ticks_for_gap(u64::from(MAX_GESTURE_DECAY_TICKS) * 4);
        assert_eq!(gap.ticks, MAX_GESTURE_DECAY_TICKS);
        assert!(gap.clamped);
        budgeted.decay(held, gap.ticks);
        unbudgeted.decay(held, MAX_GESTURE_DECAY_TICKS * 4);
        assert!(
            budgeted.cell(0, 0).expect("cell").coverage
                > unbudgeted.cell(0, 0).expect("cell").coverage * 2.0,
            "the tick budget clamp made no observable difference"
        );

        // And the budget itself, at and one past the boundary.
        assert_eq!(
            decay_ticks_for_gap(0),
            GestureDecayBudget {
                ticks: 0,
                clamped: false
            }
        );
        assert_eq!(
            decay_ticks_for_gap(u64::from(MAX_GESTURE_DECAY_TICKS)),
            GestureDecayBudget {
                ticks: MAX_GESTURE_DECAY_TICKS,
                clamped: false
            }
        );
        assert_eq!(
            decay_ticks_for_gap(u64::from(MAX_GESTURE_DECAY_TICKS) + 1),
            GestureDecayBudget {
                ticks: MAX_GESTURE_DECAY_TICKS,
                clamped: true
            }
        );
        assert_eq!(
            decay_ticks_for_gap(u64::MAX),
            GestureDecayBudget {
                ticks: MAX_GESTURE_DECAY_TICKS,
                clamped: true
            }
        );
        // A zero-tick decay is exactly a hold.
        let mut untouched = GestureCanvasField::new(grid(2, 2)).expect("field");
        untouched.etch(sample([0.5, 0.5], [1.0, 0.0], GestureMode::Push), params);
        let before = untouched.clone();
        untouched.decay(params, 0);
        assert_eq!(untouched, before);
    }

    #[test]
    fn a_full_retention_canvas_holds_its_etch_across_the_whole_tick_budget() {
        let held = authored_params(0.4, 1.0, 1.0);
        let mut field = GestureCanvasField::new(grid(2, 2)).expect("field");
        field.etch(sample([0.5, 0.5], [1.0, 0.0], GestureMode::Push), held);
        let before = field.clone();
        field.decay(held, MAX_GESTURE_DECAY_TICKS);
        assert_eq!(field, before);
    }

    #[test]
    fn a_submitted_frame_commits_and_a_discarded_frame_leaves_no_visible_change() {
        let mut state =
            GestureCanvasState::new(grid(4, 4), authored_params(0.4, 0.5, 0.99)).expect("canvas");
        let stroke = [event(0, GestureMode::Push, [0.5, 0.5], 1.0, [1.0, 0.0])];

        let plan = state
            .stage_frame(GestureCanvasFrameInput {
                reference_tick: 0,
                program_advances: true,
                events: &stroke,
                evaluated_params: None,
            })
            .expect("staged");
        assert!(state.has_staged_frame());
        assert_eq!(plan.applied_samples, 1);
        assert_eq!(plan.inert_samples, 0);
        assert_eq!(plan.decay_ticks, 0);
        assert!(!plan.held);
        assert_eq!(state.staged_samples().len(), 1);
        state.commit_staged();
        assert!(!state.has_staged_frame());
        assert_eq!(state.generation(), 1);
        assert_eq!(state.last_tick(), Some(0));
        let committed = state.field().clone();
        assert_ne!(committed, GestureCanvasField::new(grid(4, 4)).expect("f"));

        // A discarded frame rewinds the field, the decay clock, and the
        // generation together.
        let plan = state
            .stage_frame(GestureCanvasFrameInput {
                reference_tick: 30,
                program_advances: true,
                events: &stroke,
                evaluated_params: None,
            })
            .expect("staged");
        assert_eq!(plan.decay_ticks, 30);
        assert!(!plan.decay_clamped);
        assert_ne!(*state.field(), committed);
        state.discard_staged();
        assert_eq!(*state.field(), committed);
        assert_eq!(state.last_tick(), Some(0));
        assert_eq!(state.generation(), 1);
        assert!(state.staged_samples().is_empty());
    }

    #[test]
    #[should_panic(expected = "gesture canvas frame already staged")]
    fn staging_a_second_frame_over_an_open_transaction_panics_rather_than_overwriting_it() {
        let mut state =
            GestureCanvasState::new(grid(2, 2), GestureCanvasParams::default()).expect("canvas");
        let input = GestureCanvasFrameInput {
            reference_tick: 0,
            program_advances: true,
            events: &[],
            evaluated_params: None,
        };
        state.stage_frame(input).expect("first");
        let _ = state.stage_frame(input);
    }

    #[test]
    fn a_frozen_program_holds_the_canvas_and_never_bills_the_frozen_ticks_as_decay() {
        let mut state =
            GestureCanvasState::new(grid(4, 4), authored_params(0.4, 0.5, 0.9)).expect("canvas");
        let stroke = [event(0, GestureMode::Push, [0.5, 0.5], 1.0, [1.0, 0.0])];
        state
            .stage_frame(GestureCanvasFrameInput {
                reference_tick: 0,
                program_advances: true,
                events: &stroke,
                evaluated_params: None,
            })
            .expect("staged");
        state.commit_staged();
        let held = state.field().clone();

        // Nine hundred frozen frames at an advancing host address change
        // nothing and do not move the canvas clock.
        for tick in 1..=900 {
            let plan = state
                .stage_frame(GestureCanvasFrameInput {
                    reference_tick: tick,
                    program_advances: false,
                    events: &stroke,
                    evaluated_params: None,
                })
                .expect("staged");
            assert!(plan.held);
            assert!(plan.is_exact_bypass());
            assert_eq!(plan.decay_ticks, 0);
            assert_eq!(plan.applied_samples, 0);
            state.commit_staged();
        }
        assert_eq!(*state.field(), held);
        assert_eq!(state.last_tick(), Some(0));

        // Resuming bills the real gap once, clamped by the budget, rather than
        // replaying nine hundred ticks of catch-up debt.
        let plan = state
            .stage_frame(GestureCanvasFrameInput {
                reference_tick: 901,
                program_advances: true,
                events: &[],
                evaluated_params: None,
            })
            .expect("staged");
        assert_eq!(plan.decay_ticks, 901);
        assert!(!plan.decay_clamped);
        state.commit_staged();
        assert_eq!(state.last_tick(), Some(901));
    }

    #[test]
    fn identical_grouped_and_ungrouped_reference_tick_replay_produce_the_same_canvas() {
        let events: Vec<GestureEvent> = (0..6)
            .map(|index| {
                event(
                    0,
                    GestureMode::Push,
                    [0.2 + index as f32 * 0.1, 0.5],
                    1.0,
                    [1.0, 0.0],
                )
            })
            .collect();
        let authored = authored_params(0.3, 0.5, 1.0);

        // One frame carrying every sample, versus six frames carrying one each
        // at the same reference address. Decay is held out of the comparison by
        // pinning both to the same tick, which is precisely the grouping law:
        // the address, not the frame boundary, is what a sample is indexed by.
        let mut grouped = GestureCanvasState::new(grid(6, 6), authored).expect("canvas");
        grouped
            .stage_frame(GestureCanvasFrameInput {
                reference_tick: 7,
                program_advances: true,
                events: &events,
                evaluated_params: None,
            })
            .expect("staged");
        grouped.commit_staged();

        let mut ungrouped = GestureCanvasState::new(grid(6, 6), authored).expect("canvas");
        for chunk in events.chunks(1) {
            ungrouped
                .stage_frame(GestureCanvasFrameInput {
                    reference_tick: 7,
                    program_advances: true,
                    events: chunk,
                    evaluated_params: None,
                })
                .expect("staged");
            ungrouped.commit_staged();
        }
        assert_eq!(grouped.field(), ungrouped.field());

        // And in two- and three-sample groupings.
        for size in [2, 3] {
            let mut state = GestureCanvasState::new(grid(6, 6), authored).expect("canvas");
            for chunk in events.chunks(size) {
                state
                    .stage_frame(GestureCanvasFrameInput {
                        reference_tick: 7,
                        program_advances: true,
                        events: chunk,
                        evaluated_params: None,
                    })
                    .expect("staged");
                state.commit_staged();
            }
            assert_eq!(
                grouped.field(),
                state.field(),
                "grouping of {size} changed the canvas"
            );
        }
    }

    #[test]
    fn an_over_cap_update_is_refused_before_a_transaction_is_opened() {
        let mut state =
            GestureCanvasState::new(grid(2, 2), GestureCanvasParams::default()).expect("canvas");
        let flood = vec![
            event(0, GestureMode::Push, [0.5, 0.5], 1.0, [1.0, 0.0]);
            GESTURE_CANVAS_MAX_SAMPLES_PER_UPDATE + 1
        ];
        assert_eq!(
            state.stage_frame(GestureCanvasFrameInput {
                reference_tick: 0,
                program_advances: true,
                events: &flood,
                evaluated_params: None,
            }),
            Err(GestureCanvasError::TooManySamples {
                count: GESTURE_CANVAS_MAX_SAMPLES_PER_UPDATE + 1,
                limit: GESTURE_CANVAS_MAX_SAMPLES_PER_UPDATE,
            })
        );
        assert!(!state.has_staged_frame());
        assert_eq!(state.generation(), 0);
        // Exactly at the cap is admitted.
        assert!(state
            .stage_frame(GestureCanvasFrameInput {
                reference_tick: 0,
                program_advances: true,
                events: &flood[..GESTURE_CANVAS_MAX_SAMPLES_PER_UPDATE],
                evaluated_params: None,
            })
            .is_ok());
    }

    #[test]
    fn every_reset_cause_clears_exactly_its_domains_and_abandons_an_open_transaction() {
        let build = || {
            let mut state = GestureCanvasState::new(grid(4, 4), authored_params(0.4, 0.5, 0.99))
                .expect("canvas");
            state
                .stage_frame(GestureCanvasFrameInput {
                    reference_tick: 12,
                    program_advances: true,
                    events: &[event(0, GestureMode::Push, [0.5, 0.5], 1.0, [1.0, 0.0])],
                    evaluated_params: None,
                })
                .expect("staged");
            state.commit_staged();
            state
        };
        let etched = build();
        let cleared = GestureCanvasField::new(grid(4, 4)).expect("field");
        assert_ne!(*etched.field(), cleared);

        for cause in [
            GestureCanvasResetCause::PatchGeneration,
            GestureCanvasResetCause::ApplyLook,
            GestureCanvasResetCause::Resize,
            GestureCanvasResetCause::BroadRevert,
            GestureCanvasResetCause::ManualClear,
            GestureCanvasResetCause::ExportCancelled,
        ] {
            let mut state = build();
            state.reset_for(cause);
            assert_eq!(*state.field(), cleared, "{cause:?} did not clear the field");
            assert_eq!(state.last_tick(), None);
            assert_eq!(state.generation(), 0);
            assert_eq!(state.last_reset(), Some(cause));
        }

        // A cut or a source swap rebases the decay clock and deliberately keeps
        // the authored etch: a seek is not an erase.
        for cause in [
            GestureCanvasResetCause::SourceCut,
            GestureCanvasResetCause::SourceReplacement,
        ] {
            let mut state = build();
            let before = state.field().clone();
            state.reset_for(cause);
            assert_eq!(*state.field(), before, "{cause:?} erased an authored etch");
            assert_eq!(state.last_tick(), None);
            assert_eq!(state.generation(), 1);
        }

        // A reset abandons an open transaction rather than restoring it.
        let mut state = build();
        state
            .stage_frame(GestureCanvasFrameInput {
                reference_tick: 40,
                program_advances: true,
                events: &[event(0, GestureMode::Curl, [0.25, 0.25], 1.0, [0.0, 1.0])],
                evaluated_params: None,
            })
            .expect("staged");
        assert!(state.has_staged_frame());
        state.reset_for(GestureCanvasResetCause::PatchGeneration);
        assert!(!state.has_staged_frame());
        assert_eq!(*state.field(), cleared);
        state.discard_staged();
        assert_eq!(
            *state.field(),
            cleared,
            "a discard after a reset resurrected abandoned state"
        );

        // The empty domain set is a no-op, and domains compose.
        let mut untouched = build();
        let before = untouched.field().clone();
        untouched.apply_reset_domains(GestureCanvasResetDomains::NONE);
        assert_eq!(*untouched.field(), before);
        assert_eq!(untouched.last_tick(), Some(12));
        assert_eq!(
            GestureCanvasResetDomains::REBASE.union(GestureCanvasResetDomains::HARD),
            GestureCanvasResetDomains::HARD
        );
    }

    #[test]
    fn a_resize_is_a_hard_reset_that_never_resamples_cells_nobody_etched() {
        let mut state =
            GestureCanvasState::new(grid(4, 4), authored_params(0.4, 0.5, 0.99)).expect("canvas");
        state
            .stage_frame(GestureCanvasFrameInput {
                reference_tick: 3,
                program_advances: true,
                events: &[event(0, GestureMode::Push, [0.5, 0.5], 1.0, [1.0, 0.0])],
                evaluated_params: None,
            })
            .expect("staged");
        state.commit_staged();

        state.resize(8, 2).expect("resize");
        assert_eq!(state.grid(), grid(8, 2));
        assert_eq!(state.field().cells().len(), 16);
        assert_eq!(
            *state.field(),
            GestureCanvasField::new(grid(8, 2)).expect("field")
        );
        assert_eq!(state.last_tick(), None);
        assert_eq!(state.generation(), 0);
        assert_eq!(state.last_reset(), Some(GestureCanvasResetCause::Resize));

        // A resize to a refused geometry changes nothing at all.
        let before = state.field().clone();
        assert_eq!(
            state.resize(GESTURE_CANVAS_MAX_EDGE + 1, 1),
            Err(GestureCanvasError::CanvasEdge {
                dimensions: [GESTURE_CANVAS_MAX_EDGE + 1, 1],
                limit: GESTURE_CANVAS_MAX_EDGE,
            })
        );
        assert_eq!(*state.field(), before);
        assert_eq!(state.grid(), grid(8, 2));
    }

    #[test]
    fn sixteen_simultaneous_strokes_etch_sixteen_distinct_marks_in_one_update() {
        // The canvas inherits the track's stroke identity space rather than
        // declaring a second one.
        let events: Vec<GestureEvent> = (0..MAX_ACTIVE_STROKES as u8)
            .map(|stroke| {
                event(
                    stroke,
                    if stroke % 2 == 0 {
                        GestureMode::Push
                    } else {
                        GestureMode::Curl
                    },
                    [(f32::from(stroke) + 0.5) / MAX_ACTIVE_STROKES as f32, 0.5],
                    1.0,
                    [1.0, 0.0],
                )
            })
            .collect();
        let mut state = GestureCanvasState::new(
            grid(MAX_ACTIVE_STROKES as u32, 1),
            authored_params(0.03, 0.5, 1.0),
        )
        .expect("canvas");
        let plan = state
            .stage_frame(GestureCanvasFrameInput {
                reference_tick: 0,
                program_advances: true,
                events: &events,
                evaluated_params: None,
            })
            .expect("staged");
        assert_eq!(plan.applied_samples, MAX_ACTIVE_STROKES as u32);
        state.commit_staged();
        // This fixture drives the full quantized path, so the tolerance is the
        // Q16 lattice carried through the falloff (a position error of one
        // lattice step over a 0.03 radius), not the exact-arithmetic bound the
        // analytic fixtures use.
        let tolerance = 4.0 * Q16_STEP / 0.03;
        for stroke in 0..MAX_ACTIVE_STROKES as u32 {
            let cell = state.field().cell(stroke, 0).expect("cell");
            assert!(
                (cell.coverage - 1.0).abs() < tolerance,
                "stroke {stroke} missing: {cell:?}"
            );
            if stroke % 2 == 0 {
                assert!((cell.vector[0] - 0.5).abs() < tolerance);
                assert!(cell.vector[1].abs() < tolerance);
            } else {
                assert!(cell.vector[0].abs() < tolerance);
                assert!((cell.vector[1] - 0.5).abs() < tolerance);
            }
        }
    }
    /// The whole point of the presented encoding: the canvas becomes a donor
    /// image the *already frozen* Displace decode reads, so no second decode,
    /// node kind, or bind slot is invented on the consumer side.
    #[test]
    fn the_presented_donor_inverts_the_frozen_displace_decode_and_gates_exactly() {
        // An un-etched cell is transparent black and decodes to exact zero.
        let empty = GestureCell::default();
        assert_eq!(present_displace_donor(empty), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            decode_displace_donor(present_displace_donor(empty)),
            [0.0, 0.0]
        );

        // A zero-gate cell decodes to exact zero *whatever* vector it stores.
        // The premultiply uses the same alpha the decode subtracts, so the
        // established hostile-hidden-RGB law holds here without a second rule.
        for hostile in [
            [1.0_f32, -1.0_f32],
            [-0.75, 0.25],
            [f32::MAX, f32::MIN],
            [f32::NAN, f32::INFINITY],
        ] {
            let cell = GestureCell {
                vector: hostile,
                coverage: 0.0,
                hold: 1.0,
            };
            let presented = present_displace_donor(cell);
            assert_eq!(presented, [0.0, 0.0, 0.0, 0.0], "{hostile:?}");
            assert_eq!(decode_displace_donor(presented), [0.0, 0.0], "{hostile:?}");
        }

        // Neutral encoding is straight RG = 0.5 at full coverage.
        let neutral = GestureCell {
            vector: [0.0, 0.0],
            coverage: 1.0,
            hold: 0.0,
        };
        assert_eq!(present_displace_donor(neutral), [0.5, 0.5, 0.0, 1.0]);
        assert_eq!(decode_displace_donor([0.5, 0.5, 0.0, 1.0]), [0.0, 0.0]);

        // A fully covered cell round-trips its authored vector, and a partly
        // covered one displaces proportionally less: the gate does its job
        // through the same arithmetic rather than through a separate control.
        for vector in [[0.5_f32, -0.25_f32], [-1.0, 1.0], [0.125, 0.0]] {
            for coverage in [1.0_f32, 0.5, 0.25] {
                let cell = GestureCell {
                    vector,
                    coverage,
                    hold: 0.0,
                };
                let decoded = decode_displace_donor(present_displace_donor(cell));
                for axis in 0..2 {
                    assert!(
                        (decoded[axis] - vector[axis] * coverage).abs() < 1e-6,
                        "{vector:?} at {coverage}: {decoded:?}"
                    );
                }
            }
        }

        // A non-finite vector takes the neutral fallback rather than a clamped
        // extreme, and an out-of-range one clamps into the encodable band.
        let hostile = GestureCell {
            vector: [f32::NAN, 4.0],
            coverage: 1.0,
            hold: 0.0,
        };
        let decoded = decode_displace_donor(present_displace_donor(hostile));
        assert!(decoded[0].abs() < 1e-6);
        assert!((decoded[1] - 1.0).abs() < 1e-6);

        // The field-level accessor is the same law, addressed by cell.
        let mut field = GestureCanvasField::new(grid(4, 4)).expect("field");
        field.etch(
            GestureEtchSample {
                position: [0.5, 0.5],
                pressure: 1.0,
                axis: [1.0, 0.0],
            },
            GestureCanvasParams::default(),
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    field.present_displace_donor(x, y),
                    Some(present_displace_donor(field.cell(x, y).expect("cell")))
                );
            }
        }
        assert_eq!(field.present_displace_donor(4, 0), None);
    }
}
