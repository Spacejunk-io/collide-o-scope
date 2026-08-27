//! Symmetry Field — the frozen CPU group-closure reference and geometry domain.
//!
//! This module is deliberately independent of any shader, exactly as
//! [`crate::temporal::temporal_loom_age`] is, so unit tests can pin the group
//! topology, the fold semantics, and the boundary laws without requiring a GPU
//! adapter. The dedicated eight-texture pass that will consume this domain
//! reproduces these laws; it does not reinterpret them.
//!
//! # The closed vocabulary
//!
//! Eight modes with permanent append-only codes: cyclic `Cn`, dihedral `Dn`,
//! the four planar groups `p1`/`pm`/`p2`/`pmm`, a bounded log spiral, and
//! orbit. Every one of them is a genuine finite group acting on the plane:
//! applying its generator the full order of the group returns the identity and
//! the composition table closes. [`SymmetryDomain::apply`] is the action and
//! [`SymmetryElement::compose`] is the table; the tests verify the two agree
//! rather than trusting either alone.
//!
//! # The four phase semantics, frozen
//!
//! These four controls are deliberately different from one another and must
//! never be conflated:
//!
//! - **radial phase** rotates the sector **origin**. It conjugates the whole
//!   group frame, so the fundamental wedge turns with it and the folded
//!   coordinate moves.
//! - **orbit phase** rotates sector **classification**. It relabels which
//!   sector record a sample reads and never touches geometry: the folded
//!   coordinate is bit-identical across any orbit phase.
//! - **planar axis** rotates the **lattice basis**, conjugated through
//!   output-aspect space so an authored angle stays physical on a non-square
//!   output.
//! - **planar phase** translates the **primary lattice coordinate by one cell
//!   period** per unit. A phase of exactly one leaves the folded coordinate
//!   unchanged and advances the cell index by exactly one.
//!
//! `orbit_phase` is named for the orbit of the group action and applies to
//! every mode; it is not specific to [`SymmetryMode::Orbit`].
//!
//! # Rounding
//!
//! [`SymmetryParams::effective_folds`] is the sole rounding point for the fold
//! count. It rounds the already-modulated `base_folds + fold_offset` exactly
//! once and then clamps to `1..=32`. Nothing else in this module rounds folds,
//! and a source-text audit in the tests enforces that.
//!
//! # The 32-sector table
//!
//! [`SymmetrySectorTable`] is frozen at [`SYMMETRY_SECTOR_RECORDS`] records.
//! Each record chooses one [`SymmetrySource`], an optional motion slot, one
//! clean-history age inside `0..=SYMMETRY_MAX_HISTORY_AGE`, and one signed hue
//! offset. Every one of those four choices is an independent draw from
//! [`sector_lane_hash`], keyed by exactly `(stable node domain, authored seed,
//! sector index, lane-domain constant)`.
//!
//! **Runtime donor availability is not part of that key.** Losing a selected
//! donor binds the neutral views for the sectors that name it and leaves all 32
//! records bit-identical; it never rerolls a sector. Binding validity travels
//! separately, in [`SymmetryGpuBindings`].
//!
//! # Slots are routes
//!
//! There are exactly [`SYMMETRY_IMAGE_SLOTS`] image routes and
//! [`SYMMETRY_MOTION_SLOTS`] motion routes, and the slot index *is* the route
//! identity. Selected routes capture saved positions and resolve once to
//! runtime stable IDs; a `Missing` route retains its saved position and never
//! rebinds against a replacement that later occupies it.

use serde::{Deserialize, Serialize};

use crate::image_routing::StableLayerId;
use crate::motion::MotionDonor;
use crate::performance::SavedLayerPosition;
use crate::spatial::{
    apply_2x2, conjugate_through_output_aspect, finite_clamp, multiply_2x2, output_aspect_basis,
    rotation_matrix, wrap_degrees,
};
use crate::temporal::TEMPORAL_HISTORY_LEN;
use crate::visual_rack::{GroupId, ResolvedImageTap, SavedImageTap, VisualScopeId};

/// The frozen sector-table width. A sector index is a lookup key into exactly
/// this many records, so the fold count can never exceed it.
pub const SYMMETRY_SECTOR_RECORDS: usize = 32;

/// Exactly two fixed image donor slots and exactly two fixed motion donor
/// slots. The counts are frozen: slot index is route identity, so a variable
/// slot count would make an authored route depend on how many other routes
/// happen to exist.
pub const SYMMETRY_IMAGE_SLOTS: usize = 2;
pub const SYMMETRY_MOTION_SLOTS: usize = 2;

pub const SYMMETRY_MIN_FOLDS: u8 = 1;
pub const SYMMETRY_MAX_FOLDS: u8 = SYMMETRY_SECTOR_RECORDS as u8;

/// The oldest clean-history age a sector record may name. Derived from the
/// committed Compat8 ring rather than restated as a literal: age 0 is the
/// virtual current image and ages `1..=23` address stored layers.
pub const SYMMETRY_MAX_HISTORY_AGE: u32 = TEMPORAL_HISTORY_LEN - 1;

pub const FOLD_OFFSET_LIMIT: f32 = 32.0;
pub const PLANAR_PHASE_LIMIT: f32 = 4.0;
pub const CELL_SKEW_LIMIT: f32 = 1.0;
pub const SPIRAL_SCALE_LIMIT: f32 = 1.0;
pub const ORBIT_RADIUS_LIMIT: f32 = 1.0;
pub const ORBIT_PHASE_LIMIT: f32 = 1.0;
pub const CENTER_MIN: f32 = -1.0;
pub const CENTER_MAX: f32 = 2.0;
/// Signed gain applied to a sector's motion vector. Zero is inert.
pub const MOTION_GAIN_LIMIT: f32 = 1.0;
/// Hue rotation span in turns. A sector's normalized hue offset is scaled by
/// this, so a zero span is exactly no hue rotation whatever the table says.
pub const HUE_SPAN_LIMIT: f32 = 1.0;

/// Permanent hash-domain constants for the sector table. `SYMMETRY` in ASCII.
const SYMMETRY_TABLE_DOMAIN: u64 = 0x5359_4d4d_4554_5259;
const SYMMETRY_OWNER_MIX: u64 = 0x9e37_79b9_7f4a_7c15;
const SYMMETRY_NODE_MIX: u64 = 0xa076_1d64_78bd_642f;
const SYMMETRY_SECTOR_MIX: u64 = 0xd6e8_feb8_6659_fd93;
const SYMMETRY_SEED_MIX: u64 = 0x8c6f_7365_7061_7261;

const TAU: f32 = std::f32::consts::TAU;

/// Log-radius climb contributed by one fold step at full spiral scale.
const SPIRAL_LOG_CLIMB_PER_UNIT: f32 = 0.5;
/// Hard bound on the log-radius period of one full turn. This is what makes
/// the spiral *bounded*: the quotient window can never grow without limit.
const SPIRAL_MAX_LOG_PERIOD: f32 = 6.0;
/// Below this period the spiral quotient would collapse every radius onto a
/// single circle, so the mode degenerates to its pure cyclic geometry instead.
const SPIRAL_MIN_LOG_PERIOD: f32 = 0.25;
/// Radius at which the bounded spiral window starts.
const SPIRAL_ANCHOR_RADIUS: f32 = 0.25;
/// The origin is a fixed point of every rotation and has no defined log radius.
const SPIRAL_MIN_RADIUS: f32 = 1.0e-4;

const MIN_ASPECT: f32 = 1.0 / 100.0;
const MAX_ASPECT: f32 = 100.0;

/// The closed mode vocabulary. Codes are permanent and append-only: a new mode
/// takes the next integer and no existing entry is ever renumbered or reused.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymmetryMode {
    /// Cyclic `Cn`: `n` rotations, no reflection.
    #[default]
    Cyclic,
    /// Dihedral `Dn`: `n` rotations and a mirror, order `2n`.
    Dihedral,
    /// Planar `p1`: translations only.
    PlanarP1,
    /// Planar `pm`: translations and one mirror along the primary axis.
    PlanarPm,
    /// Planar `p2`: translations and a two-fold rotation.
    PlanarP2,
    /// Planar `pmm`: translations and mirrors along both lattice axes.
    PlanarPmm,
    /// Bounded log spiral: `Cn` acting on the log-polar quotient torus.
    LogSpiral,
    /// Orbit: `Cn` presented as `n` satellites on a ring.
    Orbit,
}

impl SymmetryMode {
    /// Permanent append-only shader/persistence code. Never renumber an entry.
    pub const fn code(self) -> u32 {
        match self {
            Self::Cyclic => 0,
            Self::Dihedral => 1,
            Self::PlanarP1 => 2,
            Self::PlanarPm => 3,
            Self::PlanarP2 => 4,
            Self::PlanarPmm => 5,
            Self::LogSpiral => 6,
            Self::Orbit => 7,
        }
    }

    /// Every mode in the closed vocabulary, in code order.
    pub const ALL: [Self; 8] = [
        Self::Cyclic,
        Self::Dihedral,
        Self::PlanarP1,
        Self::PlanarPm,
        Self::PlanarP2,
        Self::PlanarPmm,
        Self::LogSpiral,
        Self::Orbit,
    ];

    pub const fn key(self) -> &'static str {
        match self {
            Self::Cyclic => "cyclic",
            Self::Dihedral => "dihedral",
            Self::PlanarP1 => "planar_p1",
            Self::PlanarPm => "planar_pm",
            Self::PlanarP2 => "planar_p2",
            Self::PlanarPmm => "planar_pmm",
            Self::LogSpiral => "log_spiral",
            Self::Orbit => "orbit",
        }
    }

    /// Order of the rotation part of the point group. The radial family scales
    /// with the authored fold count; the planar groups have the fixed point
    /// groups their names declare.
    pub const fn point_rotations(self, folds: u8) -> u32 {
        match self {
            Self::Cyclic | Self::Dihedral | Self::LogSpiral | Self::Orbit => folds as u32,
            Self::PlanarP1 | Self::PlanarPm => 1,
            Self::PlanarP2 | Self::PlanarPmm => 2,
        }
    }

    /// Whether the point group contains a reflection generator.
    pub const fn has_reflection(self) -> bool {
        match self {
            Self::Dihedral | Self::PlanarPm | Self::PlanarPmm => true,
            Self::Cyclic | Self::PlanarP1 | Self::PlanarP2 | Self::LogSpiral | Self::Orbit => false,
        }
    }

    /// Whether the group carries a translation lattice.
    pub const fn has_lattice(self) -> bool {
        match self {
            Self::PlanarP1 | Self::PlanarPm | Self::PlanarP2 | Self::PlanarPmm => true,
            Self::Cyclic | Self::Dihedral | Self::LogSpiral | Self::Orbit => false,
        }
    }

    /// Order of the point group: rotations times the reflection multiplicity.
    pub const fn point_group_order(self, folds: u8) -> u32 {
        self.point_rotations(folds) * if self.has_reflection() { 2 } else { 1 }
    }

    /// Which walls of the fundamental domain are mirror lines.
    ///
    /// A mirrored wall is continuous: the folded coordinate agrees from both
    /// sides. A wall that is not mirrored is a rotation or translation seam and
    /// jumps by one full domain width. Entry 0 is the primary wall (the
    /// angular wall for the radial family, the primary lattice wall for the
    /// planar groups); entry 1 is the secondary wall.
    ///
    /// This is the declaration that "the sector boundary is continuous exactly
    /// where the group says it is" is measured against.
    pub const fn mirrored_walls(self) -> [bool; 2] {
        match self {
            Self::Dihedral | Self::PlanarPmm => [true, true],
            Self::PlanarPm => [true, false],
            Self::Cyclic | Self::PlanarP1 | Self::PlanarP2 | Self::LogSpiral | Self::Orbit => {
                [false, false]
            }
        }
    }

    /// Only cyclic geometry can claim an exact bypass. Other modes stay active
    /// even where they are numerically inert: over-claiming a bypass is a pixel
    /// bug, while under-claiming one only costs a pass.
    pub const fn is_bypass_capable(self) -> bool {
        matches!(self, Self::Cyclic)
    }
}

/// Boundary law for a folded coordinate that leaves the unit source domain.
///
/// The first four codes are the established `Transparent | Mirror | Wrap |
/// Hold` vocabulary and are byte-compatible with the Displace node's boundary
/// codes; `CellularReentry` appends code 4. `Transparent` is the authored
/// default and the only law that removes coverage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymmetryBoundary {
    #[default]
    Transparent,
    Mirror,
    Wrap,
    Hold,
    /// One deterministic D4 cell transform. A coordinate that leaves the unit
    /// cell re-enters through a square-symmetry copy of it selected by the cell
    /// index alone. This is never recursive sampling: see [`cellular_reentry`].
    CellularReentry,
}

impl SymmetryBoundary {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn code(self) -> u32 {
        match self {
            Self::Transparent => 0,
            Self::Mirror => 1,
            Self::Wrap => 2,
            Self::Hold => 3,
            Self::CellularReentry => 4,
        }
    }

    /// Every boundary in the closed vocabulary, in code order.
    pub const ALL: [Self; 5] = [
        Self::Transparent,
        Self::Mirror,
        Self::Wrap,
        Self::Hold,
        Self::CellularReentry,
    ];

    pub const fn key(self) -> &'static str {
        match self {
            Self::Transparent => "transparent",
            Self::Mirror => "mirror",
            Self::Wrap => "wrap",
            Self::Hold => "hold",
            Self::CellularReentry => "cellular_reentry",
        }
    }

    /// Map a folded coordinate back into the unit source domain and report
    /// coverage. Only `Transparent` can report `false`.
    pub fn resolve(self, uv: [f32; 2]) -> ([f32; 2], bool) {
        let uv = [finite_or(uv[0], 0.0), finite_or(uv[1], 0.0)];
        match self {
            Self::Transparent => {
                let inside = (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]);
                // Keep the speculative coordinate bounded even with no coverage,
                // exactly as `sample_source` does for edge mode 0.
                (clamp_unit(uv), inside)
            }
            Self::Hold => (clamp_unit(uv), true),
            Self::Wrap => ([wrap_unit(uv[0]), wrap_unit(uv[1])], true),
            Self::Mirror => ([mirror_unit(uv[0]), mirror_unit(uv[1])], true),
            Self::CellularReentry => (cellular_reentry(uv), true),
        }
    }
}

/// One element of a symmetry group.
///
/// `rotation` is the index of the rotation generator modulo the point group's
/// rotation order, `reflected` records the reflection generator, and `lattice`
/// is the integer translation carried by the planar groups. The radial family
/// always carries the zero translation, so the same record and the same
/// composition law describe every mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SymmetryElement {
    pub rotation: u32,
    pub reflected: bool,
    pub lattice: [i32; 2],
}

impl SymmetryElement {
    pub const IDENTITY: Self = Self {
        rotation: 0,
        reflected: false,
        lattice: [0, 0],
    };

    pub const fn rotation(rotation: u32) -> Self {
        Self {
            rotation,
            reflected: false,
            lattice: [0, 0],
        }
    }

    /// The element equivalent to applying `other` first and then `self`.
    ///
    /// This is the standard semidirect product: the dihedral law on the point
    /// group, with the point part acting on the incoming translation. The tests
    /// verify it against [`SymmetryDomain::apply`] rather than assuming it.
    pub fn compose(self, other: Self, rotations: u32) -> Self {
        let order = rotations.max(1);
        let left = self.rotation % order;
        let right = other.rotation % order;
        let rotation = if self.reflected {
            (left + order - right) % order
        } else {
            (left + right) % order
        };
        let moved = point_action(left, self.reflected, other.lattice, order);
        Self {
            rotation,
            reflected: self.reflected != other.reflected,
            lattice: [
                self.lattice[0].saturating_add(moved[0]),
                self.lattice[1].saturating_add(moved[1]),
            ],
        }
    }

    /// The unique inverse under [`Self::compose`].
    pub fn inverse(self, rotations: u32) -> Self {
        let order = rotations.max(1);
        let left = self.rotation % order;
        // A reflected point element is its own inverse: `(R^k M)^2 = I`.
        let rotation = if self.reflected {
            left
        } else {
            (order - left) % order
        };
        let negated = [
            self.lattice[0].saturating_neg(),
            self.lattice[1].saturating_neg(),
        ];
        Self {
            rotation,
            reflected: self.reflected,
            lattice: point_action(rotation, self.reflected, negated, order),
        }
    }

    pub const fn is_identity(self) -> bool {
        self.rotation == 0 && !self.reflected && self.lattice[0] == 0 && self.lattice[1] == 0
    }
}

/// The point-group action on an integer lattice translation. Reflection negates
/// the primary coordinate; the planar groups carry at most a two-fold rotation,
/// which negates both. A radial element always carries `[0, 0]`, so the general
/// case below is exact for every mode.
fn point_action(rotation: u32, reflected: bool, translation: [i32; 2], rotations: u32) -> [i32; 2] {
    let mirrored = if reflected {
        [translation[0].saturating_neg(), translation[1]]
    } else {
        translation
    };
    let half_turns = if rotations == 0 {
        0
    } else {
        rotation % rotations
    };
    if rotations == 2 && half_turns == 1 {
        [mirrored[0].saturating_neg(), mirrored[1].saturating_neg()]
    } else {
        mirrored
    }
}

/// Which image one sector record reads. Codes are permanent and append-only.
///
/// `Carrier` is the node's own input and is always bindable. `Donor0`/`Donor1`
/// name the two fixed image slots by slot index; `CleanHistory` names the
/// committed Compat8 clean-history ring at the record's own age.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymmetrySource {
    #[default]
    Carrier,
    Donor0,
    Donor1,
    CleanHistory,
}

impl SymmetrySource {
    /// Permanent append-only shader code. Never renumber an existing entry.
    pub const fn code(self) -> u32 {
        match self {
            Self::Carrier => 0,
            Self::Donor0 => 1,
            Self::Donor1 => 2,
            Self::CleanHistory => 3,
        }
    }

    pub const ALL: [Self; 4] = [
        Self::Carrier,
        Self::Donor0,
        Self::Donor1,
        Self::CleanHistory,
    ];

    /// The image slot a record reads, when it reads one. `Carrier` and
    /// `CleanHistory` own no slot.
    pub const fn image_slot(self) -> Option<u8> {
        match self {
            Self::Donor0 => Some(0),
            Self::Donor1 => Some(1),
            Self::Carrier | Self::CleanHistory => None,
        }
    }
}

/// The authored source mask: which of the four sources the sector table may
/// draw from. It is authored state, never a runtime availability report — a
/// donor that fails to bind at runtime does not change this mask and therefore
/// cannot reroll a single sector record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SymmetrySourceMask {
    pub carrier: bool,
    pub donor0: bool,
    pub donor1: bool,
    pub clean_history: bool,
}

impl Default for SymmetrySourceMask {
    fn default() -> Self {
        Self::CARRIER_ONLY
    }
}

impl SymmetrySourceMask {
    /// The exact default: carrier only.
    pub const CARRIER_ONLY: Self = Self {
        carrier: true,
        donor0: false,
        donor1: false,
        clean_history: false,
    };

    /// An entirely empty mask is not a legal draw domain, so it takes the
    /// neutral carrier-only fallback rather than a clamped extreme.
    pub const fn sanitized(self) -> Self {
        if self.carrier || self.donor0 || self.donor1 || self.clean_history {
            self
        } else {
            Self::CARRIER_ONLY
        }
    }

    /// Permanent append-only bit order, matching [`SymmetrySource::code`].
    pub const fn bits(self) -> u32 {
        let clean = self.sanitized();
        (clean.carrier as u32)
            | ((clean.donor0 as u32) << 1)
            | ((clean.donor1 as u32) << 2)
            | ((clean.clean_history as u32) << 3)
    }

    /// The ordered draw domain. The count is never zero.
    pub const fn eligible(self) -> ([SymmetrySource; 4], usize) {
        let clean = self.sanitized();
        let mut sources = [SymmetrySource::Carrier; 4];
        let mut count = 0;
        if clean.carrier {
            sources[count] = SymmetrySource::Carrier;
            count += 1;
        }
        if clean.donor0 {
            sources[count] = SymmetrySource::Donor0;
            count += 1;
        }
        if clean.donor1 {
            sources[count] = SymmetrySource::Donor1;
            count += 1;
        }
        if clean.clean_history {
            sources[count] = SymmetrySource::CleanHistory;
            count += 1;
        }
        (sources, count)
    }

    pub const fn is_carrier_only(self) -> bool {
        let clean = self.sanitized();
        clean.carrier && !clean.donor0 && !clean.donor1 && !clean.clean_history
    }
}

/// The authored motion mask: which of the two fixed motion slots the sector
/// table may draw from. An empty mask means no sector ever names a motion
/// donor, which is the exact default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SymmetryMotionMask {
    pub slot0: bool,
    pub slot1: bool,
}

impl SymmetryMotionMask {
    pub const fn is_empty(self) -> bool {
        !self.slot0 && !self.slot1
    }

    /// Permanent append-only bit order by slot index.
    pub const fn bits(self) -> u32 {
        (self.slot0 as u32) | ((self.slot1 as u32) << 1)
    }

    /// The ordered draw domain of motion slots. The count may be zero.
    pub const fn eligible(self) -> ([u8; 2], usize) {
        let mut slots = [0_u8; 2];
        let mut count = 0;
        if self.slot0 {
            slots[count] = 0;
            count += 1;
        }
        if self.slot1 {
            slots[count] = 1;
            count += 1;
        }
        (slots, count)
    }
}

/// Saved twin of [`MotionDonor`].
///
/// A live [`MotionDonor::Selected`] carries a process-lifetime
/// [`StableLayerId`] that must never be persisted, so the saved form retains
/// only the position. This mirrors the vocabulary of
/// `crate::patch::MotionDonorConfig`, which performs the same job for the
/// Faraday transplant; the Symmetry node serializes inside `VisualNodeKind`
/// rather than through a patch-side config twin and therefore needs its own
/// saved type at this layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SavedMotionDonor {
    #[default]
    None,
    Selected {
        saved_position: SavedLayerPosition,
    },
    /// Retained after a failed restore or an explicit layer deletion. It never
    /// resolves against a newly inserted layer at the vacated position.
    Missing {
        saved_position: SavedLayerPosition,
    },
}

impl SavedMotionDonor {
    /// Resolve once against the live stack. A saved `Missing` donor stays
    /// missing: it keeps its provenance and never rebinds.
    pub fn to_runtime(
        self,
        layer_at_position: &mut impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
    ) -> MotionDonor {
        match self {
            Self::None => MotionDonor::None,
            Self::Selected { saved_position } => layer_at_position(saved_position).map_or(
                MotionDonor::Missing { saved_position },
                |layer_id| MotionDonor::Selected {
                    layer_id,
                    saved_position,
                },
            ),
            Self::Missing { saved_position } => MotionDonor::Missing { saved_position },
        }
    }

    /// Capture a live route without ever persisting a process identity.
    pub fn from_runtime(
        donor: MotionDonor,
        position_of_layer: &mut impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> Self {
        match donor {
            MotionDonor::None => Self::None,
            MotionDonor::Selected {
                layer_id,
                saved_position,
            } => position_of_layer(layer_id).map_or(
                Self::Missing { saved_position },
                |resolved_position| Self::Selected {
                    saved_position: resolved_position,
                },
            ),
            MotionDonor::Missing { saved_position } => Self::Missing { saved_position },
        }
    }

    pub const fn selected_position(self) -> Option<SavedLayerPosition> {
        match self {
            Self::Selected { saved_position } => Some(saved_position),
            Self::None | Self::Missing { .. } => None,
        }
    }

    /// Preserve a deleted layer identity explicitly so a future layer at the
    /// same position cannot inherit this motion route.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved symmetry invalidation supports patch/editor migrations"
        )
    )]
    pub fn mark_layer_missing(&mut self, removed: SavedLayerPosition) {
        if let Self::Selected { saved_position } = *self {
            if saved_position == removed {
                *self = Self::Missing { saved_position };
            }
        }
    }
}

/// Stable per-node hash domain for the sector table.
///
/// `owner` is the rack owner's stable domain (master, a stable layer id, or a
/// group id) and `node_id` is the node's stable id. Both are authored identity;
/// neither depends on stack position or on what happens to bind at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymmetryNodeDomain(u64);

impl SymmetryNodeDomain {
    pub const fn new(owner: u64, node_id: u64) -> Self {
        Self(
            SYMMETRY_TABLE_DOMAIN
                ^ owner.wrapping_mul(SYMMETRY_OWNER_MIX)
                ^ node_id.wrapping_mul(SYMMETRY_NODE_MIX),
        )
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    /// The canonical domain of one authored node: its owning visual scope plus
    /// its process-stable node identity.
    ///
    /// This is the single derivation every consumer must use — the planner, the
    /// dedicated executor, Dice, and the panel — so that one authored node owns
    /// exactly one sector table wherever it is read from. Moving a node to a
    /// different scope is a different authored identity and legitimately owns a
    /// different table; reordering the stack is not, and never reaches here.
    pub const fn for_scope(scope: VisualScopeId, node_id: u64) -> Self {
        Self::new(symmetry_scope_owner(scope), node_id)
    }
}

/// Permanent per-scope owner keys. Never renumber an existing entry: doing so
/// would silently reroll every saved node's sector table in that scope.
///
/// Only *authored* identity may enter this key. A `GroupId` qualifies: it is
/// serialized in the composition and a resolved group keeps the saved value. A
/// live `StableLayerId` does not. It is a process-lifetime identity that is
/// deliberately never serialized, an export job numbers layers `position + 1`,
/// and a fresh process or a replaced clip mints a different value for the same
/// authored layer. Mixing it in would reroll all 32 records of a saved node on
/// every reload and would make an exported file disagree with the program it
/// was rendered from — the divergence
/// `the_symmetry_sector_table_is_identical_live_and_offline_for_the_same_authored_patch`
/// measures. The layer arm therefore contributes only its scope kind; per-node
/// distinction comes from the node's persisted `stable_id` and its authored
/// `seed`, both of which survive save/load, reorder, and export unchanged.
///
/// The consequence is stated rather than hidden: two Symmetry Fields carrying
/// the same node id in two different layer racks share a table unless they are
/// given different seeds. That is a bounded correlation between two authored
/// nodes; a table that changed under the operator's feet on every reload is a
/// correctness failure, and the two are not comparable.
const fn symmetry_scope_owner(scope: VisualScopeId) -> u64 {
    match scope {
        VisualScopeId::Master => 0x4d41_5354_4552,
        VisualScopeId::Program => 0x0050_524f_4752_414d,
        VisualScopeId::Layer(_) => 0x4c41_5945_5200_0000,
        VisualScopeId::Group(id) => {
            0x4752_4f55_5000_0000 ^ id.get().wrapping_mul(SYMMETRY_SEED_MIX)
        }
    }
}

/// One independent draw inside a sector record. Each lane has its own permanent
/// domain constant, so adding or reordering lanes later cannot shift an
/// existing lane's stream, and changing one authored mask cannot perturb the
/// draws of another lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryLane {
    Source,
    Motion,
    HistoryAge,
    Hue,
}

impl SymmetryLane {
    pub const ALL: [Self; 4] = [Self::Source, Self::Motion, Self::HistoryAge, Self::Hue];

    /// Permanent append-only lane domain. Never renumber an existing entry.
    pub const fn domain(self) -> u64 {
        match self {
            Self::Source => 0x534f_5552_4345_0001,
            Self::Motion => 0x4d4f_5449_4f4e_0002,
            Self::HistoryAge => 0x4147_4520_4c41_0003,
            Self::Hue => 0x4855_4520_4c41_0004,
        }
    }
}

/// The stable counter hash behind every sector record.
///
/// It is keyed by exactly four things: the stable node domain, the authored
/// seed, the sector index, and the lane-domain constant. Runtime donor
/// availability is deliberately absent from that key, which is what makes
/// losing a donor a binding change rather than a reroll. There is no sequential
/// generator state, so any record can be recomputed on its own.
pub const fn sector_lane_hash(
    domain: SymmetryNodeDomain,
    seed: u32,
    sector: u32,
    lane: SymmetryLane,
) -> u64 {
    let mut value = domain.get()
        ^ (seed as u64).wrapping_mul(SYMMETRY_SEED_MIX)
        ^ (sector as u64).wrapping_mul(SYMMETRY_SECTOR_MIX)
        ^ lane.domain();
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// One of the 32 frozen sector records.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetrySectorRecord {
    pub source: SymmetrySource,
    /// The motion slot this sector reads, or `None`. Slot index is route
    /// identity; a `Some(slot)` whose donor fails to bind stays `Some(slot)`
    /// and binds the neutral motion views instead.
    pub motion: Option<u8>,
    /// Clean-history age, always inside `0..=SYMMETRY_MAX_HISTORY_AGE`.
    pub history_age: u32,
    /// Signed normalized hue offset in turns, scaled by the authored hue span.
    pub hue_offset: f32,
}

impl SymmetrySectorRecord {
    /// The neutral record: the carrier, no motion, the virtual current image,
    /// and no hue rotation.
    pub const NEUTRAL: Self = Self {
        source: SymmetrySource::Carrier,
        motion: None,
        history_age: 0,
        hue_offset: 0.0,
    };

    /// The permanent shader encoding of the motion choice. Zero is `None`; a
    /// slot is stored as `slot + 1` so the neutral value is also the zero word.
    pub const fn motion_code(self) -> u32 {
        match self.motion {
            Some(slot) => slot as u32 + 1,
            None => 0,
        }
    }

    pub const fn from_motion_code(code: u32) -> Option<u8> {
        if code == 0 {
            None
        } else {
            Some((code - 1) as u8)
        }
    }
}

/// The frozen 32-record sector table.
///
/// The domain is fixed at [`SYMMETRY_SECTOR_RECORDS`] records regardless of the
/// authored fold count, so a fold change re-reads existing records instead of
/// rewriting the table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetrySectorTable {
    records: [SymmetrySectorRecord; SYMMETRY_SECTOR_RECORDS],
}

impl SymmetrySectorTable {
    /// Generate the complete table.
    ///
    /// Every record is an independent function of the four frozen key parts.
    /// The authored masks select the draw domain; they are authored state, so a
    /// mask edit is an authored change, while a runtime binding failure is not
    /// visible here at all.
    pub fn generate(
        domain: SymmetryNodeDomain,
        seed: u32,
        source_mask: SymmetrySourceMask,
        motion_mask: SymmetryMotionMask,
    ) -> Self {
        let (sources, source_count) = source_mask.eligible();
        let (motion_slots, motion_count) = motion_mask.eligible();
        // Index zero of the motion draw is always "no motion", so an armed mask
        // still yields sectors without a motion donor.
        let motion_options = motion_count + 1;
        let mut records = [SymmetrySectorRecord::NEUTRAL; SYMMETRY_SECTOR_RECORDS];
        for (index, record) in records.iter_mut().enumerate() {
            let sector = index as u32;
            let source_draw = sector_lane_hash(domain, seed, sector, SymmetryLane::Source);
            let motion_draw = sector_lane_hash(domain, seed, sector, SymmetryLane::Motion);
            let age_draw = sector_lane_hash(domain, seed, sector, SymmetryLane::HistoryAge);
            let hue_draw = sector_lane_hash(domain, seed, sector, SymmetryLane::Hue);
            let motion_index = (motion_draw % motion_options as u64) as usize;
            *record = SymmetrySectorRecord {
                source: sources[(source_draw % source_count as u64) as usize],
                motion: if motion_index == 0 {
                    None
                } else {
                    Some(motion_slots[motion_index - 1])
                },
                // The modulus is the ring's own age domain, so a record can
                // never name an age the committed ring cannot answer.
                history_age: (age_draw % u64::from(SYMMETRY_MAX_HISTORY_AGE + 1)) as u32,
                hue_offset: unit_hash(hue_draw).mul_add(2.0, -1.0),
            };
        }
        Self { records }
    }

    /// The table every exactly-default node owns: carrier, no motion, the
    /// virtual current image, and no hue.
    pub const NEUTRAL: Self = Self {
        records: [SymmetrySectorRecord::NEUTRAL; SYMMETRY_SECTOR_RECORDS],
    };

    /// A sector index is always a legal record index because the fold count is
    /// clamped to the table width before classification.
    pub fn record(&self, sector: u32) -> SymmetrySectorRecord {
        self.records[sector as usize % SYMMETRY_SECTOR_RECORDS]
    }

    pub const fn records(&self) -> &[SymmetrySectorRecord; SYMMETRY_SECTOR_RECORDS] {
        &self.records
    }
}

/// Authored state of one Symmetry Field node.
///
/// Every field is continuous and modulatable except `mode`, `boundary`, the
/// masks, the seed, and the four routes, which are stable authored topology.
/// The exact default — cyclic, one fold, carrier-only, no motion, no history,
/// no hue, neutral phase/axis/center — is an exact bypass that delegates before
/// allocating or encoding anything.
///
/// The two image slots and the two motion slots are fixed. **Slot index is
/// route identity**: a consumer addresses `donors[1]` as the second image route
/// whether or not `donors[0]` currently binds, and a tombstoned slot keeps its
/// saved provenance rather than sliding onto another donor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SymmetryParams {
    pub mode: SymmetryMode,
    /// Authored fold count. Continuous so it can be modulated; rounded exactly
    /// once by [`SymmetryParams::effective_folds`].
    pub base_folds: f32,
    /// Signed fold offset summed with `base_folds` before the single rounding.
    pub fold_offset: f32,
    /// Rotates the sector **origin**, in degrees.
    pub radial_phase_deg: f32,
    /// Rotates sector **classification**, in whole relabel cycles.
    pub orbit_phase: f32,
    /// Rotates the **lattice basis**, in degrees.
    pub planar_axis_deg: f32,
    /// Translates the **primary lattice coordinate** by one cell period per
    /// unit.
    pub planar_phase: f32,
    /// Bounded lattice shear. The bound is what keeps the basis invertible.
    pub cell_skew: f32,
    /// Log-radius climb per fold step for the bounded log spiral.
    pub spiral_scale: f32,
    /// Satellite ring radius for orbit mode, in physical units.
    pub orbit_radius: f32,
    /// Satellite frame spin for orbit mode, in degrees.
    pub orbit_spin_deg: f32,
    /// Symmetry center in normalized composition space.
    pub center: [f32; 2],
    pub boundary: SymmetryBoundary,
    /// Signed gain on the motion vector a sector's motion donor supplies.
    pub motion_gain: f32,
    /// Hue rotation span in turns applied to a sector's normalized hue offset.
    pub hue_span: f32,
    /// Authored seed of the 32-record sector table.
    pub seed: u32,
    /// Which sources the sector table may draw from.
    pub source_mask: SymmetrySourceMask,
    /// Which motion slots the sector table may draw from.
    pub motion_mask: SymmetryMotionMask,
    /// The two fixed image slots, addressed by slot index.
    pub donors: [SavedImageTap; SYMMETRY_IMAGE_SLOTS],
    /// The two fixed motion slots, addressed by slot index.
    pub motion: [SavedMotionDonor; SYMMETRY_MOTION_SLOTS],
}

impl Default for SymmetryParams {
    fn default() -> Self {
        Self {
            mode: SymmetryMode::Cyclic,
            base_folds: 1.0,
            fold_offset: 0.0,
            radial_phase_deg: 0.0,
            orbit_phase: 0.0,
            planar_axis_deg: 0.0,
            planar_phase: 0.0,
            cell_skew: 0.0,
            spiral_scale: 0.0,
            orbit_radius: 0.0,
            orbit_spin_deg: 0.0,
            center: [0.5, 0.5],
            boundary: SymmetryBoundary::Transparent,
            motion_gain: 0.0,
            hue_span: 0.0,
            seed: 0,
            source_mask: SymmetrySourceMask::CARRIER_ONLY,
            motion_mask: SymmetryMotionMask {
                slot0: false,
                slot1: false,
            },
            donors: [SavedImageTap {
                source: crate::visual_rack::SavedImageSource::OneBelow,
                timing: crate::visual_rack::EdgeTiming::CurrentFrame,
            }; SYMMETRY_IMAGE_SLOTS],
            motion: [SavedMotionDonor::None; SYMMETRY_MOTION_SLOTS],
        }
    }
}

impl SymmetryParams {
    /// Clamp every authored value into its declared range. Hostile non-finite
    /// input takes the neutral fallback rather than a clamped extreme.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            mode: self.mode,
            base_folds: finite_clamp(
                self.base_folds,
                defaults.base_folds,
                f32::from(SYMMETRY_MIN_FOLDS),
                f32::from(SYMMETRY_MAX_FOLDS),
            ),
            fold_offset: finite_clamp(self.fold_offset, 0.0, -FOLD_OFFSET_LIMIT, FOLD_OFFSET_LIMIT),
            radial_phase_deg: wrap_degrees(self.radial_phase_deg),
            orbit_phase: finite_clamp(self.orbit_phase, 0.0, -ORBIT_PHASE_LIMIT, ORBIT_PHASE_LIMIT),
            planar_axis_deg: wrap_degrees(self.planar_axis_deg),
            planar_phase: finite_clamp(
                self.planar_phase,
                0.0,
                -PLANAR_PHASE_LIMIT,
                PLANAR_PHASE_LIMIT,
            ),
            cell_skew: finite_clamp(self.cell_skew, 0.0, -CELL_SKEW_LIMIT, CELL_SKEW_LIMIT),
            spiral_scale: finite_clamp(
                self.spiral_scale,
                0.0,
                -SPIRAL_SCALE_LIMIT,
                SPIRAL_SCALE_LIMIT,
            ),
            orbit_radius: finite_clamp(self.orbit_radius, 0.0, 0.0, ORBIT_RADIUS_LIMIT),
            orbit_spin_deg: wrap_degrees(self.orbit_spin_deg),
            center: [
                finite_clamp(self.center[0], defaults.center[0], CENTER_MIN, CENTER_MAX),
                finite_clamp(self.center[1], defaults.center[1], CENTER_MIN, CENTER_MAX),
            ],
            boundary: self.boundary,
            motion_gain: finite_clamp(self.motion_gain, 0.0, -MOTION_GAIN_LIMIT, MOTION_GAIN_LIMIT),
            hue_span: finite_clamp(self.hue_span, 0.0, 0.0, HUE_SPAN_LIMIT),
            seed: self.seed,
            source_mask: self.source_mask.sanitized(),
            motion_mask: self.motion_mask,
            donors: self.donors,
            motion: self.motion,
        }
    }

    /// The sole rounding point for the fold count.
    ///
    /// Both inputs arrive already modulated. They are summed first, rounded
    /// exactly once, and only then clamped to `1..=32`. Rounding each input
    /// separately would silently discard fractional modulation, so nothing else
    /// in this module rounds folds.
    pub fn effective_folds(self) -> u8 {
        let base = finite_or(self.base_folds, 1.0);
        let offset = finite_or(self.fold_offset, 0.0);
        let modulated = base + offset;
        // The single rounding. A finite pair can still overflow to infinity, so
        // guard the sum rather than rounding a non-finite value.
        let rounded = if modulated.is_finite() {
            modulated.round()
        } else {
            1.0
        };
        rounded.clamp(f32::from(SYMMETRY_MIN_FOLDS), f32::from(SYMMETRY_MAX_FOLDS)) as u8
    }

    /// True when the fold is provably the identity map **and** every sector
    /// record must read the carrier unaltered, so the planner collects no
    /// resources and the executor encodes no pass. The carrier then passes
    /// through untouched rather than through any sampling path.
    ///
    /// Both halves are required. An identity fold whose table can still read a
    /// donor or the history ring is not a bypass, and a carrier-only table
    /// under an active fold is not one either.
    pub fn is_exact_bypass(self) -> bool {
        let clean = self.sanitized();
        clean.mode.is_bypass_capable()
            && clean.effective_folds() == SYMMETRY_MIN_FOLDS
            && clean.radial_phase_deg == 0.0
            && clean.orbit_phase == 0.0
            && clean.table_is_neutral()
    }

    /// True when the authored masks and hue span force every generated record
    /// to equal [`SymmetrySectorRecord::NEUTRAL`], whatever the seed. This is a
    /// property of the authored state alone, so it is decidable without
    /// generating the table.
    pub fn table_is_neutral(self) -> bool {
        let clean = self.sanitized();
        clean.source_mask.is_carrier_only() && clean.motion_mask.is_empty() && clean.hue_span == 0.0
    }

    /// Generate this node's 32-record sector table. A neutral table short
    /// circuits to the frozen constant so an exact-default node never depends
    /// on its seed.
    pub fn sector_table(self, domain: SymmetryNodeDomain) -> SymmetrySectorTable {
        let clean = self.sanitized();
        if clean.table_is_neutral() {
            return SymmetrySectorTable::NEUTRAL;
        }
        SymmetrySectorTable::generate(domain, clean.seed, clean.source_mask, clean.motion_mask)
    }

    /// The image route at one slot. The slot index is the route identity.
    pub const fn donor_tap(self, slot: u8) -> Option<SavedImageTap> {
        match slot {
            0 => Some(self.donors[0]),
            1 => Some(self.donors[1]),
            _ => None,
        }
    }

    /// The motion route at one slot. The slot index is the route identity.
    pub const fn motion_donor(self, slot: u8) -> Option<SavedMotionDonor> {
        match slot {
            0 => Some(self.motion[0]),
            1 => Some(self.motion[1]),
            _ => None,
        }
    }

    /// The image routes this node actually samples, indexed by slot.
    ///
    /// Admission is answered **per slot**, because slot index is route
    /// identity: a donor whose source-mask bit is clear can never be chosen by
    /// any sector record, so it claims no dependency edge, no tombstone
    /// diagnostic, and no binding slot. That is the same real delegation an
    /// exact bypass performs for the whole node, applied one slot at a time.
    /// Clearing slot 0 never slides slot 1's route down into slot 0.
    ///
    /// The mask is authored topology rather than a frame-local value, so this
    /// answer is stable across modulation, Morph, and Dice.
    pub fn admitted_donor_taps(self) -> [Option<SavedImageTap>; SYMMETRY_IMAGE_SLOTS] {
        let mask = self.source_mask.sanitized();
        [
            mask.donor0.then_some(self.donors[0]),
            mask.donor1.then_some(self.donors[1]),
        ]
    }

    /// The motion routes this node actually consumes, indexed by slot. An
    /// unarmed slot requests no primitive vector/gate field at all.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the live planner walks the runtime twin; the saved twin is its frozen mirror"
        )
    )]
    pub fn admitted_motion_donors(self) -> [Option<SavedMotionDonor>; SYMMETRY_MOTION_SLOTS] {
        [
            self.motion_mask.slot0.then_some(self.motion[0]),
            self.motion_mask.slot1.then_some(self.motion[1]),
        ]
    }

    /// Every saved layer position this node names, image slots first. The
    /// fixed width keeps the rack walkers allocation free.
    pub const fn selected_layer_positions(self) -> [Option<SavedLayerPosition>; 4] {
        [
            self.donors[0].selected_layer_position(),
            self.donors[1].selected_layer_position(),
            self.motion[0].selected_position(),
            self.motion[1].selected_position(),
        ]
    }

    pub const fn referenced_groups(self) -> [Option<GroupId>; SYMMETRY_IMAGE_SLOTS] {
        [
            self.donors[0].referenced_group(),
            self.donors[1].referenced_group(),
        ]
    }

    /// Preserve a deleted group identity explicitly, per image slot, so a
    /// future group at the same root position cannot inherit either route.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved symmetry invalidation supports patch/editor migrations"
        )
    )]
    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        for tap in &mut self.donors {
            if tap.source
                == (crate::visual_rack::SavedImageSource::GroupOutput { group_id: removed })
            {
                tap.source =
                    crate::visual_rack::SavedImageSource::MissingGroupOutput { group_id: removed };
            }
        }
    }

    /// Resolve the authored controls into the concrete geometry domain for one
    /// output size.
    pub fn domain(self, output_dimensions: (u32, u32)) -> SymmetryDomain {
        let clean = self.sanitized();
        let folds = clean.effective_folds();
        let rotations = clean.mode.point_rotations(folds).max(1);
        let output_aspect = output_aspect(output_dimensions);
        let (to_physical, from_physical) = output_aspect_basis(output_aspect);
        let raw_step = clean.spiral_scale * SPIRAL_LOG_CLIMB_PER_UNIT;
        let period =
            (f32::from(folds) * raw_step).clamp(-SPIRAL_MAX_LOG_PERIOD, SPIRAL_MAX_LOG_PERIOD);
        // Derive the step back out of the clamped period so exactly `folds`
        // steps climb exactly one period and the generator closes.
        let spiral_step = if period.abs() >= SPIRAL_MIN_LOG_PERIOD {
            period / f32::from(folds)
        } else {
            0.0
        };
        // A fold count of one leaves the whole plane in a single cell, so the
        // lattice period is the full normalized span.
        let cell_period = 1.0 / f32::from(folds);
        let orbit_offset = fold_relabel_offset(clean.orbit_phase, folds);
        SymmetryDomain {
            mode: clean.mode,
            folds,
            rotations,
            sector_width: TAU / rotations as f32,
            cell_period,
            frame_angle: clean.radial_phase_deg.to_radians(),
            axis_angle: clean.planar_axis_deg.to_radians(),
            planar_phase: clean.planar_phase,
            skew: [[1.0, clean.cell_skew], [0.0, 1.0]],
            skew_inverse: [[1.0, -clean.cell_skew], [0.0, 1.0]],
            spiral_step,
            orbit_radius: clean.orbit_radius,
            orbit_spin: clean.orbit_spin_deg.to_radians(),
            orbit_offset,
            center: clean.center,
            output_aspect,
            to_physical,
            from_physical,
            boundary: clean.boundary,
            exact_bypass: clean.is_exact_bypass(),
        }
    }

    /// Fold one output coordinate. This is the reference the dedicated pass
    /// reproduces.
    pub fn fold(self, uv: [f32; 2], output_dimensions: (u32, u32)) -> SymmetryFold {
        self.domain(output_dimensions).fold(uv)
    }
}

/// Route-resolved Symmetry Field state.
///
/// The strict runtime twin of [`SymmetryParams`]: identical values, with each
/// saved route replaced by its resolved form. Live routing is by stable ID; a
/// saved position survives only inside a resolved route's missing provenance.
/// Both slot arrays keep their fixed width, so slot index remains route
/// identity after resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeSymmetryParams {
    pub mode: SymmetryMode,
    pub base_folds: f32,
    pub fold_offset: f32,
    pub radial_phase_deg: f32,
    pub orbit_phase: f32,
    pub planar_axis_deg: f32,
    pub planar_phase: f32,
    pub cell_skew: f32,
    pub spiral_scale: f32,
    pub orbit_radius: f32,
    pub orbit_spin_deg: f32,
    pub center: [f32; 2],
    pub boundary: SymmetryBoundary,
    pub motion_gain: f32,
    pub hue_span: f32,
    pub seed: u32,
    pub source_mask: SymmetrySourceMask,
    pub motion_mask: SymmetryMotionMask,
    pub donors: [ResolvedImageTap; SYMMETRY_IMAGE_SLOTS],
    pub motion: [MotionDonor; SYMMETRY_MOTION_SLOTS],
}

impl Default for RuntimeSymmetryParams {
    fn default() -> Self {
        // `ResolvedImageTap` has no `Default`; the exact default route is the
        // same deterministic OneBelow/current-frame donor every image-consuming
        // node starts from.
        let saved = SymmetryParams::default();
        Self {
            mode: saved.mode,
            base_folds: saved.base_folds,
            fold_offset: saved.fold_offset,
            radial_phase_deg: saved.radial_phase_deg,
            orbit_phase: saved.orbit_phase,
            planar_axis_deg: saved.planar_axis_deg,
            planar_phase: saved.planar_phase,
            cell_skew: saved.cell_skew,
            spiral_scale: saved.spiral_scale,
            orbit_radius: saved.orbit_radius,
            orbit_spin_deg: saved.orbit_spin_deg,
            center: saved.center,
            boundary: saved.boundary,
            motion_gain: saved.motion_gain,
            hue_span: saved.hue_span,
            seed: saved.seed,
            source_mask: saved.source_mask,
            motion_mask: saved.motion_mask,
            donors: [ResolvedImageTap {
                source: crate::visual_rack::ResolvedImageSource::OneBelow,
                timing: crate::visual_rack::EdgeTiming::CurrentFrame,
            }; SYMMETRY_IMAGE_SLOTS],
            motion: [MotionDonor::None; SYMMETRY_MOTION_SLOTS],
        }
    }
}

impl RuntimeSymmetryParams {
    /// Sanitize the saved value first, then resolve each route exactly once.
    pub fn resolve_routes(
        saved: SymmetryParams,
        layer_at_position: &mut impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        group_exists: &impl Fn(GroupId) -> bool,
    ) -> Self {
        let saved = saved.sanitized();
        let donors = [
            saved.donors[0].to_runtime(&mut *layer_at_position, group_exists),
            saved.donors[1].to_runtime(&mut *layer_at_position, group_exists),
        ];
        let motion = [
            saved.motion[0].to_runtime(layer_at_position),
            saved.motion[1].to_runtime(layer_at_position),
        ];
        Self {
            mode: saved.mode,
            base_folds: saved.base_folds,
            fold_offset: saved.fold_offset,
            radial_phase_deg: saved.radial_phase_deg,
            orbit_phase: saved.orbit_phase,
            planar_axis_deg: saved.planar_axis_deg,
            planar_phase: saved.planar_phase,
            cell_skew: saved.cell_skew,
            spiral_scale: saved.spiral_scale,
            orbit_radius: saved.orbit_radius,
            orbit_spin_deg: saved.orbit_spin_deg,
            center: saved.center,
            boundary: saved.boundary,
            motion_gain: saved.motion_gain,
            hue_span: saved.hue_span,
            seed: saved.seed,
            source_mask: saved.source_mask,
            motion_mask: saved.motion_mask,
            donors,
            motion,
        }
    }

    /// Capture without ever persisting a process identity, then sanitize.
    pub fn capture_routes(
        self,
        position_of_layer: &mut impl FnMut(StableLayerId) -> Option<SavedLayerPosition>,
    ) -> SymmetryParams {
        let donors = [
            self.donors[0].to_saved(&mut *position_of_layer),
            self.donors[1].to_saved(&mut *position_of_layer),
        ];
        let motion = [
            SavedMotionDonor::from_runtime(self.motion[0], position_of_layer),
            SavedMotionDonor::from_runtime(self.motion[1], position_of_layer),
        ];
        SymmetryParams {
            mode: self.mode,
            base_folds: self.base_folds,
            fold_offset: self.fold_offset,
            radial_phase_deg: self.radial_phase_deg,
            orbit_phase: self.orbit_phase,
            planar_axis_deg: self.planar_axis_deg,
            planar_phase: self.planar_phase,
            cell_skew: self.cell_skew,
            spiral_scale: self.spiral_scale,
            orbit_radius: self.orbit_radius,
            orbit_spin_deg: self.orbit_spin_deg,
            center: self.center,
            boundary: self.boundary,
            motion_gain: self.motion_gain,
            hue_span: self.hue_span,
            seed: self.seed,
            source_mask: self.source_mask,
            motion_mask: self.motion_mask,
            donors,
            motion,
        }
        .sanitized()
    }

    /// Mirror of [`SymmetryParams::sanitized`] for the live model. It routes
    /// through the saved sanitizer so the two can never diverge.
    pub fn sanitized(self) -> Self {
        let values = self.values().sanitized();
        Self {
            mode: values.mode,
            base_folds: values.base_folds,
            fold_offset: values.fold_offset,
            radial_phase_deg: values.radial_phase_deg,
            orbit_phase: values.orbit_phase,
            planar_axis_deg: values.planar_axis_deg,
            planar_phase: values.planar_phase,
            cell_skew: values.cell_skew,
            spiral_scale: values.spiral_scale,
            orbit_radius: values.orbit_radius,
            orbit_spin_deg: values.orbit_spin_deg,
            center: values.center,
            boundary: values.boundary,
            motion_gain: values.motion_gain,
            hue_span: values.hue_span,
            seed: values.seed,
            source_mask: values.source_mask,
            motion_mask: values.motion_mask,
            donors: self.donors,
            motion: self.motion,
        }
    }

    /// The route-free authored values, carrying the saved default routes. This
    /// is the single place the runtime twin borrows the saved laws from; it
    /// deliberately does not expose the live routes.
    pub fn values(self) -> SymmetryParams {
        SymmetryParams {
            mode: self.mode,
            base_folds: self.base_folds,
            fold_offset: self.fold_offset,
            radial_phase_deg: self.radial_phase_deg,
            orbit_phase: self.orbit_phase,
            planar_axis_deg: self.planar_axis_deg,
            planar_phase: self.planar_phase,
            cell_skew: self.cell_skew,
            spiral_scale: self.spiral_scale,
            orbit_radius: self.orbit_radius,
            orbit_spin_deg: self.orbit_spin_deg,
            center: self.center,
            boundary: self.boundary,
            motion_gain: self.motion_gain,
            hue_span: self.hue_span,
            seed: self.seed,
            source_mask: self.source_mask,
            motion_mask: self.motion_mask,
            ..SymmetryParams::default()
        }
    }

    /// Mirror of [`SymmetryParams::is_exact_bypass`] for the live model.
    pub fn is_exact_bypass(self) -> bool {
        self.values().is_exact_bypass()
    }

    /// Mirror of [`SymmetryParams::sector_table`] for the live model. Runtime
    /// route availability is deliberately not an argument.
    pub fn sector_table(self, domain: SymmetryNodeDomain) -> SymmetrySectorTable {
        self.values().sector_table(domain)
    }

    pub const fn donor_tap(self, slot: u8) -> Option<ResolvedImageTap> {
        match slot {
            0 => Some(self.donors[0]),
            1 => Some(self.donors[1]),
            _ => None,
        }
    }

    pub const fn motion_donor(self, slot: u8) -> Option<MotionDonor> {
        match slot {
            0 => Some(self.motion[0]),
            1 => Some(self.motion[1]),
            _ => None,
        }
    }

    /// Mirror of [`SymmetryParams::admitted_donor_taps`] for the live model.
    /// The two answers must agree slot for slot, so the saved and live planner
    /// walks admit exactly the same routes.
    pub fn admitted_donor_taps(self) -> [Option<ResolvedImageTap>; SYMMETRY_IMAGE_SLOTS] {
        let mask = self.source_mask.sanitized();
        [
            mask.donor0.then_some(self.donors[0]),
            mask.donor1.then_some(self.donors[1]),
        ]
    }

    /// Mirror of [`SymmetryParams::admitted_motion_donors`] for the live model.
    pub fn admitted_motion_donors(self) -> [Option<MotionDonor>; SYMMETRY_MOTION_SLOTS] {
        [
            self.motion_mask.slot0.then_some(self.motion[0]),
            self.motion_mask.slot1.then_some(self.motion[1]),
        ]
    }

    /// Every live layer this node names, image slots first.
    pub const fn selected_layer_ids(self) -> [Option<StableLayerId>; 4] {
        [
            resolved_selected_layer(self.donors[0]),
            resolved_selected_layer(self.donors[1]),
            motion_selected_layer(self.motion[0]),
            motion_selected_layer(self.motion[1]),
        ]
    }

    pub const fn referenced_groups(self) -> [Option<GroupId>; SYMMETRY_IMAGE_SLOTS] {
        [
            self.donors[0].referenced_group(),
            self.donors[1].referenced_group(),
        ]
    }

    pub fn mark_layer_output_missing(&mut self, removed: StableLayerId) {
        for tap in &mut self.donors {
            tap.mark_layer_missing(removed);
        }
        for donor in &mut self.motion {
            if let MotionDonor::Selected {
                layer_id,
                saved_position,
            } = *donor
            {
                if layer_id == removed {
                    *donor = MotionDonor::Missing { saved_position };
                }
            }
        }
    }

    pub fn mark_group_output_missing(&mut self, removed: GroupId) {
        for tap in &mut self.donors {
            tap.mark_group_missing(removed);
        }
    }
}

const fn resolved_selected_layer(tap: ResolvedImageTap) -> Option<StableLayerId> {
    match tap.source {
        crate::visual_rack::ResolvedImageSource::SelectedLayer { layer_id, .. } => Some(layer_id),
        _ => None,
    }
}

const fn motion_selected_layer(donor: MotionDonor) -> Option<StableLayerId> {
    match donor {
        MotionDonor::Selected { layer_id, .. } => Some(layer_id),
        MotionDonor::None | MotionDonor::Missing { .. } => None,
    }
}

/// One sample carried into the fundamental domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetryFold {
    /// The folded source coordinate in output UV space, after the boundary law.
    pub uv: [f32; 2],
    /// The sector classification: the lookup key into the sector table.
    pub sector: u32,
    /// The group element that carries the fundamental domain onto this sample.
    pub element: SymmetryElement,
    /// False only when a boundary law removed coverage.
    pub covered: bool,
}

impl SymmetryFold {
    /// The exact-bypass result: the carrier coordinate passes through
    /// bit-identical, with no sampling conversion of any kind.
    pub const fn identity(uv: [f32; 2]) -> Self {
        Self {
            uv,
            sector: 0,
            element: SymmetryElement::IDENTITY,
            covered: true,
        }
    }
}

/// Classification of one point in the group's own coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetryClassification {
    /// The group element `e` with `apply(e, local) == point`.
    pub element: SymmetryElement,
    /// The representative inside the fundamental domain.
    pub local: [f32; 2],
    /// Sector classification before the orbit-phase relabel.
    pub raw_sector: u32,
}

/// The resolved geometry of one Symmetry Field at one output size.
///
/// The group acts in its own coordinates: physical, phase-rotated space for the
/// radial family, and dimensionless lattice cells for the planar groups.
/// [`SymmetryDomain::group_coordinates`] and [`SymmetryDomain::output_coordinates`]
/// are the only places output UV enters and leaves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetryDomain {
    pub mode: SymmetryMode,
    pub folds: u8,
    /// Order of the rotation part of the point group.
    pub rotations: u32,
    /// Angular width of one rotation step, radians.
    pub sector_width: f32,
    /// Lattice cell period in physical units.
    pub cell_period: f32,
    /// Radial phase as a frame rotation, radians.
    pub frame_angle: f32,
    /// Planar axis as a lattice basis rotation, radians.
    pub axis_angle: f32,
    /// Planar phase in whole cell periods.
    pub planar_phase: f32,
    pub skew: [[f32; 2]; 2],
    pub skew_inverse: [[f32; 2]; 2],
    /// Log-radius climb per rotation step; zero outside the spiral mode and
    /// inside the spiral's bounded dead zone.
    pub spiral_step: f32,
    pub orbit_radius: f32,
    pub orbit_spin: f32,
    /// Whole sectors the classification is relabelled by.
    pub orbit_offset: u32,
    pub center: [f32; 2],
    pub output_aspect: f32,
    pub to_physical: [[f32; 2]; 2],
    pub from_physical: [[f32; 2]; 2],
    pub boundary: SymmetryBoundary,
    pub exact_bypass: bool,
}

impl SymmetryDomain {
    /// Every element of the point group, in a stable order.
    pub fn point_group(&self) -> Vec<SymmetryElement> {
        let mut elements = Vec::with_capacity(self.mode.point_group_order(self.folds) as usize);
        for reflected in [false, true] {
            if reflected && !self.mode.has_reflection() {
                continue;
            }
            for rotation in 0..self.rotations {
                elements.push(SymmetryElement {
                    rotation,
                    reflected,
                    lattice: [0, 0],
                });
            }
        }
        elements
    }

    /// The primary rotation generator.
    pub const fn generator(&self) -> SymmetryElement {
        SymmetryElement::rotation(1)
    }

    /// Angular extent of the fundamental wedge for the radial family. A
    /// reflection halves the rotation step, because the far half of the step
    /// folds back through the mirror at its midpoint.
    pub fn wedge_angle(&self) -> f32 {
        if self.mode.has_reflection() {
            self.sector_width * 0.5
        } else {
            self.sector_width
        }
    }

    /// Extent of the fundamental domain along each lattice axis, as a fraction
    /// of one cell. A mirrored axis halves; the two-fold rotation of `p2` halves
    /// the secondary axis only, which is why `p2` has no mirror line.
    pub const fn cell_extent(&self) -> [f32; 2] {
        match self.mode {
            SymmetryMode::PlanarPm => [0.5, 1.0],
            SymmetryMode::PlanarP2 => [1.0, 0.5],
            SymmetryMode::PlanarPmm => [0.5, 0.5],
            SymmetryMode::PlanarP1
            | SymmetryMode::Cyclic
            | SymmetryMode::Dihedral
            | SymmetryMode::LogSpiral
            | SymmetryMode::Orbit => [1.0, 1.0],
        }
    }

    /// The reflection generator, when the group has one.
    pub fn reflection_generator(&self) -> Option<SymmetryElement> {
        self.mode.has_reflection().then_some(SymmetryElement {
            rotation: 0,
            reflected: true,
            lattice: [0, 0],
        })
    }

    /// The bounded log-spiral quotient period, when the spiral is active.
    ///
    /// `Some(period)` means the group acts on a log-polar torus: `folds` steps
    /// climb exactly one period, so the generator closes. `None` means the
    /// radius passes through untouched.
    pub fn spiral_period(&self) -> Option<f32> {
        if self.mode != SymmetryMode::LogSpiral || self.spiral_step == 0.0 {
            return None;
        }
        let period = (f32::from(self.folds) * self.spiral_step).abs();
        (period >= SPIRAL_MIN_LOG_PERIOD).then_some(period)
    }

    /// Reduce a point into the domain's own canonical representative set. This
    /// is the identity for every mode except the bounded log spiral, whose
    /// space is a quotient torus.
    pub fn canonical_point(&self, point: [f32; 2]) -> [f32; 2] {
        let Some(period) = self.spiral_period() else {
            return point;
        };
        let radius = point[0].hypot(point[1]);
        if !radius.is_finite() || radius <= SPIRAL_MIN_RADIUS {
            return point;
        }
        rescale_to_log(point, radius, wrap_log(radius.ln(), period))
    }

    /// The group action. `apply(e, local)` carries the fundamental domain onto
    /// the sample that classified as `e`.
    pub fn apply(&self, element: SymmetryElement, point: [f32; 2]) -> [f32; 2] {
        if self.mode.has_lattice() {
            let mirrored = if element.reflected {
                [-point[0], point[1]]
            } else {
                point
            };
            let turned = if self.rotations == 2 && element.rotation % 2 == 1 {
                [-mirrored[0], -mirrored[1]]
            } else {
                mirrored
            };
            return [
                turned[0] + element.lattice[0] as f32,
                turned[1] + element.lattice[1] as f32,
            ];
        }
        let mirrored = if element.reflected {
            [point[0], -point[1]]
        } else {
            point
        };
        let steps = element.rotation % self.rotations.max(1);
        let turned = rotate(mirrored, steps as f32 * self.sector_width);
        self.spiral_climb(turned, steps)
    }

    /// Apply the bounded spiral climb of `steps` rotation generators.
    fn spiral_climb(&self, point: [f32; 2], steps: u32) -> [f32; 2] {
        if steps == 0 {
            // The identity must be exactly the identity, with no round trip
            // through the logarithm.
            return point;
        }
        let Some(period) = self.spiral_period() else {
            return point;
        };
        let radius = point[0].hypot(point[1]);
        if !radius.is_finite() || radius <= SPIRAL_MIN_RADIUS {
            return point;
        }
        let climbed = radius.ln() + steps as f32 * self.spiral_step;
        rescale_to_log(point, radius, wrap_log(climbed, period))
    }

    /// Output UV into the group's own coordinates.
    pub fn group_coordinates(&self, uv: [f32; 2]) -> [f32; 2] {
        let offset = [
            finite_or(uv[0], self.center[0]) - self.center[0],
            finite_or(uv[1], self.center[1]) - self.center[1],
        ];
        let physical = apply_2x2(self.to_physical, offset);
        if self.mode.has_lattice() {
            let unrotated = rotate(physical, -self.axis_angle);
            let unskewed = apply_2x2(self.skew_inverse, unrotated);
            // Cell 0 is centered on the authored center, and the planar phase
            // translates the primary lattice coordinate by whole cell periods.
            return [
                unskewed[0] / self.cell_period + 0.5 + self.planar_phase,
                unskewed[1] / self.cell_period + 0.5,
            ];
        }
        // The radial phase conjugates the whole group frame, which is what
        // rotating the sector origin means.
        rotate(physical, -self.frame_angle)
    }

    /// The group's own coordinates back into output UV.
    ///
    /// Orbit mode's satellite offset and spin are applied here rather than in
    /// the group, because a uniform post-fold frame is a presentation of `Cn`
    /// and not a generator of it.
    pub fn output_coordinates(&self, point: [f32; 2]) -> [f32; 2] {
        let physical = if self.mode.has_lattice() {
            let scaled = [
                (point[0] - 0.5) * self.cell_period,
                (point[1] - 0.5) * self.cell_period,
            ];
            rotate(apply_2x2(self.skew, scaled), self.axis_angle)
        } else {
            let framed = if self.mode == SymmetryMode::Orbit {
                rotate([point[0] - self.orbit_radius, point[1]], self.orbit_spin)
            } else {
                point
            };
            rotate(framed, self.frame_angle)
        };
        let offset = apply_2x2(self.from_physical, physical);
        [offset[0] + self.center[0], offset[1] + self.center[1]]
    }

    /// The group action expressed as a linear map on output UV offsets.
    ///
    /// The radial phase cancels out of a conjugation, so a pure rotation in the
    /// group frame is a pure rotation in physical space. Conjugating it back
    /// through output-aspect space is what keeps an authored angle physical on
    /// a non-square output.
    pub fn rotation_in_uv_space(&self, steps: u32) -> [[f32; 2]; 2] {
        let steps = steps % self.rotations.max(1);
        conjugate_through_output_aspect(
            rotation_matrix(steps as f32 * self.sector_width),
            self.output_aspect,
        )
    }

    /// The lattice basis expressed in output UV space: cell coordinates in,
    /// UV offsets out.
    pub fn lattice_basis_in_uv(&self) -> [[f32; 2]; 2] {
        let physical = multiply_2x2(rotation_matrix(self.axis_angle), self.skew);
        let scaled = multiply_2x2(physical, [[self.cell_period, 0.0], [0.0, self.cell_period]]);
        multiply_2x2(self.from_physical, scaled)
    }

    /// Classify one point in the group's own coordinates.
    pub fn classify(&self, point: [f32; 2]) -> SymmetryClassification {
        match self.mode {
            SymmetryMode::Cyclic
            | SymmetryMode::Dihedral
            | SymmetryMode::LogSpiral
            | SymmetryMode::Orbit => self.classify_radial(point),
            SymmetryMode::PlanarP1
            | SymmetryMode::PlanarPm
            | SymmetryMode::PlanarP2
            | SymmetryMode::PlanarPmm => self.classify_planar(point),
        }
    }

    fn classify_radial(&self, point: [f32; 2]) -> SymmetryClassification {
        let point = self.canonical_point(point);
        let order = self.rotations.max(1);
        let angle = point[1].atan2(point[0]).rem_euclid(TAU);
        let index = (angle / self.sector_width).floor();
        let step = (index as i64).clamp(0, i64::from(order) - 1) as u32;
        let within = angle - step as f32 * self.sector_width;
        let element = if self.mode.has_reflection() && within > self.wedge_angle() {
            SymmetryElement {
                rotation: (step + 1) % order,
                reflected: true,
                lattice: [0, 0],
            }
        } else {
            SymmetryElement::rotation(step)
        };
        SymmetryClassification {
            element,
            local: self.apply(element.inverse(order), point),
            raw_sector: step,
        }
    }

    fn classify_planar(&self, point: [f32; 2]) -> SymmetryClassification {
        let cell = [point[0].floor(), point[1].floor()];
        let fraction = [point[0] - cell[0], point[1] - cell[1]];
        let (local, mirror_u, mirror_v) = match self.mode {
            SymmetryMode::PlanarP1 => (fraction, false, false),
            SymmetryMode::PlanarPm => {
                let (folded, mirrored) = mirror_fold(fraction[0]);
                ([folded, fraction[1]], mirrored, false)
            }
            SymmetryMode::PlanarP2 => {
                // A two-fold rotation folds both axes together about the cell
                // center, so it has no mirror line and its wall is a seam.
                if fraction[1] > 0.5 {
                    ([1.0 - fraction[0], 1.0 - fraction[1]], true, true)
                } else {
                    (fraction, false, false)
                }
            }
            SymmetryMode::PlanarPmm => {
                let (folded_u, mirrored_u) = mirror_fold(fraction[0]);
                let (folded_v, mirrored_v) = mirror_fold(fraction[1]);
                ([folded_u, folded_v], mirrored_u, mirrored_v)
            }
            // A radial mode never reaches the lattice fold. Treating its cell
            // as the trivial one keeps this function total without a wildcard.
            SymmetryMode::Cyclic
            | SymmetryMode::Dihedral
            | SymmetryMode::LogSpiral
            | SymmetryMode::Orbit => (fraction, false, false),
        };
        let (rotation, reflected) = planar_point_element(mirror_u, mirror_v);
        let lattice = [
            cell_index(cell[0]).saturating_add(i32::from(mirror_u)),
            cell_index(cell[1]).saturating_add(i32::from(mirror_v)),
        ];
        let order = self.rotations.max(1);
        SymmetryClassification {
            element: SymmetryElement {
                rotation: rotation % order,
                reflected,
                lattice,
            },
            local,
            // A planar sector varies along both lattice axes so a cell diagonal
            // never repeats a single record across a whole row.
            raw_sector: lattice[0]
                .saturating_add(lattice[1])
                .rem_euclid(i32::from(self.folds).max(1)) as u32,
        }
    }

    /// Fold one output coordinate into the fundamental domain.
    pub fn fold(&self, uv: [f32; 2]) -> SymmetryFold {
        if self.exact_bypass {
            // The exact bypass returns the carrier coordinate verbatim. It must
            // not travel through the aspect round trip or any boundary law.
            return SymmetryFold::identity(uv);
        }
        let classification = self.classify(self.group_coordinates(uv));
        let (folded, covered) = self
            .boundary
            .resolve(self.output_coordinates(classification.local));
        SymmetryFold {
            uv: folded,
            sector: self.sector_of(classification.raw_sector),
            element: classification.element,
            covered,
        }
    }

    /// Apply the orbit-phase relabel. Classification only: the folded
    /// coordinate is never touched by this.
    pub fn sector_of(&self, raw_sector: u32) -> u32 {
        let folds = u32::from(self.folds).max(1);
        (raw_sector % folds + self.orbit_offset) % folds
    }
}

/// Whole-sector relabel offset for the orbit phase.
///
/// This rounds a relabel count, not a fold count: the fold count is already
/// frozen by [`SymmetryParams::effective_folds`] before this runs.
fn fold_relabel_offset(orbit_phase: f32, folds: u8) -> u32 {
    let folds = i32::from(folds).max(1);
    let steps = (finite_or(orbit_phase, 0.0) * folds as f32).floor();
    cell_index(steps).rem_euclid(folds) as u32
}

/// The planar point element for a pair of per-axis mirrors.
///
/// The four planar point groups all sit inside `D2 = {I, Mu, Mv, R180}`, which
/// the dihedral parametrization covers exactly: `Mv = R180 . Mu`.
const fn planar_point_element(mirror_u: bool, mirror_v: bool) -> (u32, bool) {
    (mirror_v as u32, mirror_u != mirror_v)
}

fn mirror_fold(fraction: f32) -> (f32, bool) {
    if fraction > 0.5 {
        (1.0 - fraction, true)
    } else {
        (fraction, false)
    }
}

/// One deterministic D4 cell transform.
///
/// A coordinate outside the unit cell re-enters through a square-symmetry copy
/// of that cell, chosen by the cell index alone. The function is a **total
/// function of one cell coordinate**: it reads the cell index once, selects one
/// of the eight D4 elements, and applies it. There is no self-call, no loop,
/// and no iteration count, so a coordinate arbitrarily far outside the domain
/// costs exactly the same work as one just outside it and always lands inside
/// the unit cell after a single application. This is emphatically not recursive
/// sampling, and the tests assert that structurally rather than by comment.
pub fn cellular_reentry(uv: [f32; 2]) -> [f32; 2] {
    let cell = [uv[0].floor(), uv[1].floor()];
    let fraction = [uv[0] - cell[0], uv[1] - cell[1]];
    apply_d4(d4_cell_element(cell), fraction)
}

/// Select one of the eight D4 elements from a cell index. Two parity bits and
/// one supercell parity bit give a deterministic period-four tiling in each
/// axis.
fn d4_cell_element(cell: [f32; 2]) -> u8 {
    let x = cell_index(cell[0]);
    let y = cell_index(cell[1]);
    let parity_x = x.rem_euclid(2) as u8;
    let parity_y = y.rem_euclid(2) as u8;
    let supercell = x.div_euclid(2).wrapping_add(y.div_euclid(2)).rem_euclid(2) as u8;
    parity_x | (parity_y << 1) | (supercell << 2)
}

/// Apply one D4 element to a unit-cell fraction, about the cell center. Bit 2
/// is the reflection and bits 0..2 are the quarter turn.
fn apply_d4(element: u8, fraction: [f32; 2]) -> [f32; 2] {
    let mut point = [fraction[0] - 0.5, fraction[1] - 0.5];
    if element & 0b100 != 0 {
        point[0] = -point[0];
    }
    point = match element & 0b011 {
        0 => point,
        1 => [-point[1], point[0]],
        2 => [-point[0], -point[1]],
        _ => [point[1], -point[0]],
    };
    [point[0] + 0.5, point[1] + 0.5]
}

fn rotate(vector: [f32; 2], angle: f32) -> [f32; 2] {
    apply_2x2(rotation_matrix(angle), vector)
}

fn rescale_to_log(point: [f32; 2], radius: f32, log_radius: f32) -> [f32; 2] {
    let scale = log_radius.exp() / radius;
    if scale.is_finite() {
        [point[0] * scale, point[1] * scale]
    } else {
        point
    }
}

/// Reduce a log radius into the bounded quotient window.
fn wrap_log(log_radius: f32, period: f32) -> f32 {
    let anchor = SPIRAL_ANCHOR_RADIUS.ln();
    anchor + (log_radius - anchor).rem_euclid(period)
}

/// The WGSL `fract` law, which is floor based and therefore correct for
/// negative input. Rust's `f32::fract` truncates instead and must not be used.
fn wrap_unit(value: f32) -> f32 {
    value - value.floor()
}

/// The triangle-wave mirror repeat used by `sample_source` edge mode 3 and by
/// the Displace node's `MIRROR` boundary.
fn mirror_unit(value: f32) -> f32 {
    1.0 - (wrap_unit(value * 0.5) * 2.0 - 1.0).abs()
}

fn clamp_unit(uv: [f32; 2]) -> [f32; 2] {
    [uv[0].clamp(0.0, 1.0), uv[1].clamp(0.0, 1.0)]
}

fn cell_index(value: f32) -> i32 {
    if value.is_finite() {
        value as i32
    } else {
        0
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn output_aspect(dimensions: (u32, u32)) -> f32 {
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return 1.0;
    }
    let aspect = dimensions.0 as f32 / dimensions.1 as f32;
    if aspect.is_finite() {
        aspect.clamp(MIN_ASPECT, MAX_ASPECT)
    } else {
        1.0
    }
}

/// True when a sector record may name this clean-history age. Age 0 is the
/// virtual current image; `1..=23` address stored ring layers.
pub const fn history_age_is_in_domain(age: u32) -> bool {
    age <= SYMMETRY_MAX_HISTORY_AGE
}

/// Map a lane draw into `[0, 1)` through its top 24 bits, the widest window
/// binary32 represents exactly, so the same draw yields the same float on every
/// host.
fn unit_hash(value: u64) -> f32 {
    const SCALE: f32 = 1.0 / (1_u32 << 24) as f32;
    ((value >> 40) as u32) as f32 * SCALE
}

/// Rows of [`SymmetryGpuUniforms::motion_rows`] owned by one motion slot.
pub const SYMMETRY_MOTION_ROWS_PER_SLOT: usize = 4;

/// Renderer-supplied facts the authored domain cannot know: which routes
/// actually bound this frame, the low-resolution motion grids, and the
/// committed clean-history ring cursor.
///
/// Binding validity enters the uniform record and nothing else. It is never an
/// argument to sector-table generation, so losing a donor changes a binding
/// flag and leaves all 32 records bit-identical.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SymmetryGpuBindings {
    pub donor_valid: [bool; SYMMETRY_IMAGE_SLOTS],
    pub motion_valid: [bool; SYMMETRY_MOTION_SLOTS],
    /// `[width, height]` of each motion slot's `MotionGrid`, zero when absent.
    pub motion_grid: [[u32; 2]; SYMMETRY_MOTION_SLOTS],
    /// `TemporalReadSnapshot::virtual_write`.
    pub history_write_index: u32,
    /// `TemporalReadSnapshot::virtual_valid`; a record whose age exceeds this
    /// must be clamped by the consumer before it touches a ring layer.
    pub history_valid: u32,
}

/// The node controls the dedicated pass observes that are not authored
/// geometry: the wet amount, the blend law, and the program time the frame was
/// sampled at.
///
/// These arrive from the evaluated node, not from the authored parameters, so
/// they are supplied to [`SymmetryGpuUniforms::pack`] separately rather than
/// being smuggled into [`SymmetryParams`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetryFrameUniforms {
    pub wet: f32,
    pub blend_code: u32,
    pub time_seconds: f32,
}

impl Default for SymmetryFrameUniforms {
    /// Fully wet, Normal blend, at time zero.
    fn default() -> Self {
        Self {
            wet: 1.0,
            blend_code: 0,
            time_seconds: 0.0,
        }
    }
}

/// The exact 1,024-byte dynamic-offset uniform record of one Symmetry Field.
///
/// Sixty-four whole 16-byte lanes: four `meta` lanes, four `params` lanes,
/// eight `motion_rows` lanes, the 32 sector records at one lane each, one
/// renderer-owned `frame` lane, one renderer-owned `frame_modes` lane, and 14
/// reserved lanes. The reserved tail exists so the dedicated pass can add
/// further renderer-owned fields without moving the stride or any existing
/// offset, which is exactly what `frame`/`frame_modes` consumed two of.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SymmetryGpuUniforms {
    pub meta: [[u32; 4]; 4],
    pub params: [[f32; 4]; 4],
    pub motion_rows: [[f32; 4]; 8],
    pub sectors: [[u32; 4]; SYMMETRY_SECTOR_RECORDS],
    /// Wet, program time in seconds, output width, output height.
    pub frame: [f32; 4],
    /// Blend code, then three reserved lanes.
    pub frame_modes: [u32; 4],
    pub padding: [[u32; 4]; 14],
}

const _: () = assert!(std::mem::size_of::<SymmetryGpuUniforms>() == 1_024);
const _: () = assert!(std::mem::align_of::<SymmetryGpuUniforms>() == 4);

impl SymmetryGpuUniforms {
    /// The byte length the dynamic-offset arena strides by, before the device's
    /// `min_uniform_buffer_offset_alignment` is applied.
    pub const BYTES: u64 = 1_024;

    /// Pack one node's authored geometry, its sector table, and this frame's
    /// bindings into the frozen record.
    pub fn pack(
        params: SymmetryParams,
        output_dimensions: (u32, u32),
        table: &SymmetrySectorTable,
        bindings: SymmetryGpuBindings,
        frame: SymmetryFrameUniforms,
    ) -> Self {
        let clean = params.sanitized();
        let domain = clean.domain(output_dimensions);
        let mut sectors = [[0_u32; 4]; SYMMETRY_SECTOR_RECORDS];
        for (lane, record) in sectors.iter_mut().zip(table.records()) {
            *lane = [
                record.source.code(),
                record.motion_code(),
                record.history_age,
                record.hue_offset.to_bits(),
            ];
        }
        let mut motion_rows = [[0.0_f32; 4]; 8];
        for slot in 0..SYMMETRY_MOTION_SLOTS {
            let base = slot * SYMMETRY_MOTION_ROWS_PER_SLOT;
            let [width, height] = bindings.motion_grid[slot];
            motion_rows[base] = [
                width as f32,
                height as f32,
                if width == 0 { 0.0 } else { 1.0 / width as f32 },
                if height == 0 {
                    0.0
                } else {
                    1.0 / height as f32
                },
            ];
            motion_rows[base + 1] = [
                f32::from(u8::from(bindings.motion_valid[slot])),
                slot as f32,
                0.0,
                0.0,
            ];
            // Rows two and three of each slot are reserved for the dedicated
            // pass's own sampling and gate terms.
        }
        Self {
            meta: [
                [
                    domain.mode.code(),
                    domain.boundary.code(),
                    u32::from(domain.folds),
                    domain.rotations,
                ],
                [
                    clean.source_mask.bits(),
                    clean.motion_mask.bits(),
                    clean.seed,
                    domain.orbit_offset,
                ],
                [
                    u32::from(bindings.donor_valid[0]),
                    u32::from(bindings.donor_valid[1]),
                    u32::from(bindings.motion_valid[0]),
                    u32::from(bindings.motion_valid[1]),
                ],
                [
                    bindings.history_write_index,
                    bindings.history_valid,
                    SYMMETRY_SECTOR_RECORDS as u32,
                    u32::from(domain.exact_bypass),
                ],
            ],
            params: [
                [
                    domain.center[0],
                    domain.center[1],
                    domain.sector_width,
                    domain.cell_period,
                ],
                [
                    domain.frame_angle,
                    domain.axis_angle,
                    domain.planar_phase,
                    domain.spiral_step,
                ],
                [
                    domain.orbit_radius,
                    domain.orbit_spin,
                    domain.output_aspect,
                    domain.skew[0][1],
                ],
                [clean.motion_gain, clean.hue_span, 0.0, 0.0],
            ],
            motion_rows,
            sectors,
            frame: [
                finite_or(frame.wet, 1.0).clamp(0.0, 1.0),
                finite_or(frame.time_seconds, 0.0).max(0.0),
                output_dimensions.0 as f32,
                output_dimensions.1 as f32,
            ],
            frame_modes: [frame.blend_code, 0, 0, 0],
            padding: [[0; 4]; 14],
        }
    }

    /// Read one packed sector lane back out. The packing is lossless, so this
    /// reconstructs the CPU record exactly.
    pub fn sector(&self, index: usize) -> SymmetrySectorRecord {
        let lane = self.sectors[index % SYMMETRY_SECTOR_RECORDS];
        SymmetrySectorRecord {
            source: match lane[0] {
                1 => SymmetrySource::Donor0,
                2 => SymmetrySource::Donor1,
                3 => SymmetrySource::CleanHistory,
                _ => SymmetrySource::Carrier,
            },
            motion: SymmetrySectorRecord::from_motion_code(lane[1]),
            history_age: lane[2],
            hue_offset: f32::from_bits(lane[3]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQUARE: (u32, u32) = (1024, 1024);
    const WIDE: (u32, u32) = (1920, 1080);

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected} within {tolerance}"
        );
    }

    fn close_point(actual: [f32; 2], expected: [f32; 2], tolerance: f32) {
        close(actual[0], expected[0], tolerance);
        close(actual[1], expected[1], tolerance);
    }

    fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
        (a[0] - b[0]).hypot(a[1] - b[1])
    }

    /// A representative domain per mode, with every mode-specific control armed
    /// so no test accidentally exercises a degenerate configuration.
    fn domain_for(mode: SymmetryMode, folds: f32) -> SymmetryDomain {
        SymmetryParams {
            mode,
            base_folds: folds,
            spiral_scale: 0.8,
            orbit_radius: 0.3,
            orbit_spin_deg: 20.0,
            cell_skew: 0.25,
            planar_axis_deg: 15.0,
            ..SymmetryParams::default()
        }
        .domain(SQUARE)
    }

    /// Canonical probe points inside the domain's own coordinate space.
    fn probes(domain: &SymmetryDomain) -> Vec<[f32; 2]> {
        let raw: [[f32; 2]; 5] = if domain.mode.has_lattice() {
            [
                [0.21, 0.34],
                [-1.4, 2.6],
                [0.5, 0.5],
                [3.25, -0.75],
                [-2.1, -3.4],
            ]
        } else {
            [
                [0.31, 0.12],
                [-0.44, 0.27],
                [0.18, -0.36],
                [-0.22, -0.41],
                [0.47, 0.03],
            ]
        };
        raw.into_iter()
            .map(|point| domain.canonical_point(point))
            .collect()
    }

    /// The point group plus, for the planar groups, representative lattice
    /// translations. The point group alone never exercises the semidirect part
    /// of the composition law.
    fn test_elements(domain: &SymmetryDomain) -> Vec<SymmetryElement> {
        let point_group = domain.point_group();
        let mut elements = point_group.clone();
        if domain.mode.has_lattice() {
            for lattice in [[1, 0], [0, 1], [-2, 1], [3, -2]] {
                for &element in &point_group {
                    elements.push(SymmetryElement { lattice, ..element });
                }
            }
        }
        elements
    }

    fn position(value: u32) -> SavedLayerPosition {
        SavedLayerPosition::new(value).expect("a nonzero saved position")
    }

    fn live(value: u64) -> StableLayerId {
        StableLayerId::new(value).expect("a nonzero live id")
    }

    const NODE_DOMAIN: SymmetryNodeDomain = SymmetryNodeDomain::new(0x4d41_5354_4552, 7);

    /// A node with every mask armed, so the table draws from the full domain.
    fn armed_params() -> SymmetryParams {
        SymmetryParams {
            mode: SymmetryMode::Dihedral,
            base_folds: 6.0,
            hue_span: 0.5,
            motion_gain: 0.75,
            seed: 0x1234_5678,
            source_mask: SymmetrySourceMask {
                carrier: true,
                donor0: true,
                donor1: true,
                clean_history: true,
            },
            motion_mask: SymmetryMotionMask {
                slot0: true,
                slot1: true,
            },
            ..SymmetryParams::default()
        }
    }

    #[test]
    fn the_sector_table_freezes_thirty_two_records_inside_the_source_motion_and_history_domains() {
        assert_eq!(SYMMETRY_IMAGE_SLOTS, 2);
        assert_eq!(SYMMETRY_MOTION_SLOTS, 2);

        // Source codes and lane domains are permanent and append-only.
        for (source, code) in [
            (SymmetrySource::Carrier, 0_u32),
            (SymmetrySource::Donor0, 1),
            (SymmetrySource::Donor1, 2),
            (SymmetrySource::CleanHistory, 3),
        ] {
            assert_eq!(source.code(), code);
        }
        assert_eq!(SymmetrySource::ALL.len(), 4);
        assert_eq!(SymmetrySource::Carrier.image_slot(), None);
        assert_eq!(SymmetrySource::Donor0.image_slot(), Some(0));
        assert_eq!(SymmetrySource::Donor1.image_slot(), Some(1));
        assert_eq!(SymmetrySource::CleanHistory.image_slot(), None);
        let lanes: std::collections::BTreeSet<u64> =
            SymmetryLane::ALL.iter().map(|lane| lane.domain()).collect();
        assert_eq!(lanes.len(), SymmetryLane::ALL.len());

        let params = armed_params();
        let table = params.sector_table(NODE_DOMAIN);
        assert_eq!(table.records().len(), SYMMETRY_SECTOR_RECORDS);

        let mut sources = std::collections::BTreeSet::new();
        let mut motions = std::collections::BTreeSet::new();
        let mut oldest_age = 0;
        for (index, record) in table.records().iter().enumerate() {
            assert_eq!(*record, table.record(index as u32));
            assert!(
                history_age_is_in_domain(record.history_age),
                "sector {index} named age {} outside the committed ring",
                record.history_age
            );
            oldest_age = oldest_age.max(record.history_age);
            assert!(record.hue_offset >= -1.0 && record.hue_offset <= 1.0);
            assert!(matches!(record.motion, None | Some(0) | Some(1)));
            sources.insert(record.source.code());
            motions.insert(record.motion_code());
            assert_eq!(
                SymmetrySectorRecord::from_motion_code(record.motion_code()),
                record.motion
            );
        }
        // A fully armed mask actually reaches every choice, so the domain
        // assertions above are not vacuous.
        assert_eq!(sources, std::collections::BTreeSet::from([0, 1, 2, 3]));
        assert_eq!(motions, std::collections::BTreeSet::from([0, 1, 2]));

        // The ring bound is reached and 24 is rejected.
        assert_eq!(oldest_age, SYMMETRY_MAX_HISTORY_AGE);
        assert!(history_age_is_in_domain(SYMMETRY_MAX_HISTORY_AGE));
        assert!(!history_age_is_in_domain(SYMMETRY_MAX_HISTORY_AGE + 1));
        assert!(!history_age_is_in_domain(24));

        // A narrowed mask narrows the draw domain rather than producing an
        // illegal record, and an empty mask takes the carrier-only fallback.
        let history_only = SymmetryParams {
            source_mask: SymmetrySourceMask {
                carrier: false,
                donor0: false,
                donor1: false,
                clean_history: true,
            },
            motion_mask: SymmetryMotionMask {
                slot0: false,
                slot1: true,
            },
            ..armed_params()
        };
        let narrowed = history_only.sector_table(NODE_DOMAIN);
        for record in narrowed.records() {
            assert_eq!(record.source, SymmetrySource::CleanHistory);
            assert!(matches!(record.motion, None | Some(1)));
        }
        let empty_mask = SymmetrySourceMask {
            carrier: false,
            donor0: false,
            donor1: false,
            clean_history: false,
        };
        assert_eq!(empty_mask.sanitized(), SymmetrySourceMask::CARRIER_ONLY);
        assert_eq!(empty_mask.bits(), 1);
        assert_eq!(empty_mask.eligible().1, 1);
        assert_eq!(
            SymmetrySourceMask {
                carrier: true,
                donor0: true,
                donor1: true,
                clean_history: true,
            }
            .bits(),
            0b1111
        );
        assert_eq!(
            SymmetryMotionMask {
                slot0: false,
                slot1: true,
            }
            .bits(),
            0b10
        );
    }

    /// Runtime donor availability is not part of the sector-table key, so
    /// losing a selected donor leaves every one of the 32 records
    /// bit-identical and changes only the binding.
    #[test]
    fn sector_records_are_bit_identical_when_a_runtime_donor_is_lost() {
        let params = SymmetryParams {
            donors: [
                SavedImageTap {
                    source: crate::visual_rack::SavedImageSource::SelectedLayer {
                        layer_position: position(4),
                        stage: crate::image_routing::LayerImageStage::PostLocalEffects,
                    },
                    timing: crate::visual_rack::EdgeTiming::CurrentFrame,
                },
                SavedImageTap::default(),
            ],
            motion: [
                SavedMotionDonor::Selected {
                    saved_position: position(5),
                },
                SavedMotionDonor::None,
            ],
            ..armed_params()
        };
        let bound = params.sector_table(NODE_DOMAIN);

        // Resolve against a stack that no longer holds either donor.
        let mut missing_lookup = |_: SavedLayerPosition| None;
        let lost = RuntimeSymmetryParams::resolve_routes(params, &mut missing_lookup, &|_| false);
        assert!(matches!(
            lost.donors[0].source,
            crate::visual_rack::ResolvedImageSource::MissingSelectedLayer { .. }
        ));
        assert!(matches!(lost.motion[0], MotionDonor::Missing { .. }));
        assert_eq!(lost.sector_table(NODE_DOMAIN), bound);

        // The captured saved form is likewise a pure binding change.
        let mut no_positions = |_: StableLayerId| None;
        let recaptured = lost.capture_routes(&mut no_positions);
        let after = recaptured.sector_table(NODE_DOMAIN);
        assert_eq!(after, bound);
        for (index, (before, after)) in bound.records().iter().zip(after.records()).enumerate() {
            assert_eq!(before.source, after.source, "sector {index}");
            assert_eq!(before.motion, after.motion, "sector {index}");
            assert_eq!(before.history_age, after.history_age, "sector {index}");
            assert_eq!(
                before.hue_offset.to_bits(),
                after.hue_offset.to_bits(),
                "sector {index} hue offset is not bit identical"
            );
        }

        // In the packed record only the validity lane moves.
        let dimensions = (1920, 1080);
        let all_bound = SymmetryGpuUniforms::pack(
            params,
            dimensions,
            &bound,
            SymmetryGpuBindings {
                donor_valid: [true, true],
                motion_valid: [true, false],
                motion_grid: [[120, 68], [0, 0]],
                history_write_index: 3,
                history_valid: 24,
            },
            SymmetryFrameUniforms::default(),
        );
        let donor_lost = SymmetryGpuUniforms::pack(
            recaptured,
            dimensions,
            &after,
            SymmetryGpuBindings {
                donor_valid: [false, true],
                motion_valid: [false, false],
                motion_grid: [[0, 0], [0, 0]],
                history_write_index: 3,
                history_valid: 24,
            },
            SymmetryFrameUniforms::default(),
        );
        assert_eq!(all_bound.sectors, donor_lost.sectors);
        assert_eq!(all_bound.params, donor_lost.params);
        assert_eq!(all_bound.meta[0], donor_lost.meta[0]);
        assert_eq!(all_bound.meta[1], donor_lost.meta[1]);
        assert_ne!(all_bound.meta[2], donor_lost.meta[2]);
        assert_eq!(donor_lost.meta[2], [0, 1, 0, 0]);

        // Structural, not merely observed: the generator's source text never
        // mentions a runtime route or a binding at all.
        let body = function_body(module_source(), "    pub fn generate(");
        for forbidden in [
            "valid",
            "Resolved",
            "MotionDonor",
            "donors",
            "bindings",
            "Runtime",
        ] {
            assert!(
                !body.contains(forbidden),
                "sector generation must not observe {forbidden}"
            );
        }
    }

    /// The table domain is built from authored identity only.
    ///
    /// A live `StableLayerId` is process lifetime and is never serialized, and
    /// an export job numbers layers `position + 1`, so a domain that consumed
    /// one would reroll a saved node's 32 records on every reload and make the
    /// exported file disagree with the live program. A `GroupId` is authored,
    /// is serialized in the composition, and legitimately does distinguish.
    #[test]
    fn the_table_domain_carries_only_authored_identity_and_never_a_process_lifetime_layer_id() {
        let params = armed_params();
        let node = 7_u64;
        let layer = |id: u64| VisualScopeId::Layer(StableLayerId::new(id).expect("a live id"));
        let group = |id: u64| VisualScopeId::Group(GroupId::new(id).expect("a group id"));

        // Export numbers this layer 1; a live host after ordinary layer churn
        // calls the same authored layer 910. Neither may move the table.
        let export_scope = SymmetryNodeDomain::for_scope(layer(1), node);
        let live_scope = SymmetryNodeDomain::for_scope(layer(910), node);
        assert_eq!(export_scope, live_scope);
        assert_eq!(
            params.sector_table(export_scope),
            params.sector_table(live_scope)
        );

        // The persisted node id and the authored seed are what distinguish two
        // nodes, and they survive save/load, reorder, and export unchanged.
        assert_ne!(
            SymmetryNodeDomain::for_scope(layer(1), node),
            SymmetryNodeDomain::for_scope(layer(1), node + 1)
        );
        assert_ne!(
            params.sector_table(export_scope),
            SymmetryParams {
                seed: params.seed.wrapping_add(1),
                ..params
            }
            .sector_table(export_scope)
        );

        // Scope kinds stay distinct, and an authored group id still separates
        // two groups because it is persisted rather than minted at runtime.
        let master = SymmetryNodeDomain::for_scope(VisualScopeId::Master, node);
        let program = SymmetryNodeDomain::for_scope(VisualScopeId::Program, node);
        let first_group = SymmetryNodeDomain::for_scope(group(1), node);
        let second_group = SymmetryNodeDomain::for_scope(group(2), node);
        for (left, right) in [
            (master, program),
            (master, export_scope),
            (master, first_group),
            (program, export_scope),
            (program, first_group),
            (export_scope, first_group),
            (first_group, second_group),
        ] {
            assert_ne!(left, right);
        }
    }

    /// The table is stable by owner, node, and seed, and no other authored
    /// value can disturb it.
    #[test]
    fn the_sector_table_is_stable_by_owner_node_and_seed_and_reseeds_deterministically() {
        let params = armed_params();
        let table = params.sector_table(NODE_DOMAIN);
        assert_eq!(params.sector_table(NODE_DOMAIN), table);

        let other_node = SymmetryNodeDomain::new(0x4d41_5354_4552, 8);
        let other_owner = SymmetryNodeDomain::new(0x4c41_5945_5200, 7);
        assert_ne!(NODE_DOMAIN.get(), other_node.get());
        assert_ne!(NODE_DOMAIN.get(), other_owner.get());
        assert_ne!(params.sector_table(other_node), table);
        assert_ne!(params.sector_table(other_owner), table);
        assert_ne!(
            SymmetryParams {
                seed: params.seed.wrapping_add(1),
                ..params
            }
            .sector_table(NODE_DOMAIN),
            table
        );

        // Every continuous control is orthogonal to the table: modulating,
        // dicing, or morphing a value can never reroll a sector.
        for moved in [
            SymmetryParams {
                base_folds: 11.0,
                ..params
            },
            SymmetryParams {
                fold_offset: -3.0,
                ..params
            },
            SymmetryParams {
                radial_phase_deg: 42.0,
                ..params
            },
            SymmetryParams {
                orbit_phase: 0.4,
                ..params
            },
            SymmetryParams {
                planar_axis_deg: -70.0,
                ..params
            },
            SymmetryParams {
                cell_skew: 0.6,
                ..params
            },
            SymmetryParams {
                motion_gain: -1.0,
                ..params
            },
            SymmetryParams {
                center: [0.1, 0.9],
                ..params
            },
            SymmetryParams {
                mode: SymmetryMode::LogSpiral,
                ..params
            },
            SymmetryParams {
                boundary: SymmetryBoundary::CellularReentry,
                ..params
            },
        ] {
            assert_eq!(moved.sector_table(NODE_DOMAIN), table);
        }

        // Every lane is independently keyed, so one lane's draw is not a
        // function of another's.
        for sector in 0..SYMMETRY_SECTOR_RECORDS as u32 {
            let draws: std::collections::BTreeSet<u64> = SymmetryLane::ALL
                .iter()
                .map(|lane| sector_lane_hash(NODE_DOMAIN, params.seed, sector, *lane))
                .collect();
            assert_eq!(draws.len(), SymmetryLane::ALL.len(), "sector {sector}");
        }
    }

    /// The exact default owns the frozen neutral table: carrier only, no
    /// motion, the virtual current image, and no hue, whatever the seed.
    #[test]
    fn the_exact_default_owns_the_neutral_carrier_only_table_whatever_its_seed() {
        let default = SymmetryParams::default();
        assert!(default.table_is_neutral());
        assert_eq!(
            default.sector_table(NODE_DOMAIN),
            SymmetrySectorTable::NEUTRAL
        );
        for seed in [0_u32, 1, 0xdead_beef, u32::MAX] {
            let table = SymmetryParams { seed, ..default }.sector_table(NODE_DOMAIN);
            assert_eq!(table, SymmetrySectorTable::NEUTRAL);
            for record in table.records() {
                assert_eq!(*record, SymmetrySectorRecord::NEUTRAL);
                assert_eq!(record.source, SymmetrySource::Carrier);
                assert_eq!(record.motion, None);
                assert_eq!(record.motion_code(), 0);
                assert_eq!(record.history_age, 0);
                assert_eq!(record.hue_offset, 0.0);
            }
        }

        // Arming any one of the four table controls ends the neutral claim.
        assert!(!SymmetryParams {
            hue_span: 0.001,
            ..default
        }
        .table_is_neutral());
        assert!(!SymmetryParams {
            motion_mask: SymmetryMotionMask {
                slot0: true,
                slot1: false
            },
            ..default
        }
        .table_is_neutral());
        assert!(!SymmetryParams {
            source_mask: SymmetrySourceMask {
                donor1: true,
                ..SymmetrySourceMask::CARRIER_ONLY
            },
            ..default
        }
        .table_is_neutral());
    }

    /// Slot index is route identity across resolution and capture, and a
    /// missing route retains its saved position and never rebinds.
    #[test]
    fn the_two_image_slots_and_two_motion_slots_are_addressed_by_slot_index_and_never_rebind() {
        let params = SymmetryParams {
            donors: [
                SavedImageTap {
                    source: crate::visual_rack::SavedImageSource::SelectedLayer {
                        layer_position: position(2),
                        stage: crate::image_routing::LayerImageStage::PreLocalEffects,
                    },
                    timing: crate::visual_rack::EdgeTiming::PreviousFrame,
                },
                SavedImageTap {
                    source: crate::visual_rack::SavedImageSource::SelectedLayer {
                        layer_position: position(3),
                        stage: crate::image_routing::LayerImageStage::PostLocalEffects,
                    },
                    timing: crate::visual_rack::EdgeTiming::CurrentFrame,
                },
            ],
            motion: [
                SavedMotionDonor::Selected {
                    saved_position: position(2),
                },
                SavedMotionDonor::Selected {
                    saved_position: position(9),
                },
            ],
            ..armed_params()
        };
        assert_eq!(
            params.selected_layer_positions(),
            [
                Some(position(2)),
                Some(position(3)),
                Some(position(2)),
                Some(position(9)),
            ]
        );
        assert!(params.donor_tap(2).is_none());
        assert!(params.motion_donor(2).is_none());

        // Only position 2 resolves; slot one of each pair tombstones.
        let mut lookup = |value: SavedLayerPosition| (value.get() == 2).then(|| live(21));
        let runtime = RuntimeSymmetryParams::resolve_routes(params, &mut lookup, &|_| false);
        assert_eq!(
            runtime.donors[0].source,
            crate::visual_rack::ResolvedImageSource::SelectedLayer {
                layer_id: live(21),
                saved_position: position(2),
                stage: crate::image_routing::LayerImageStage::PreLocalEffects,
            }
        );
        assert_eq!(
            runtime.donors[0].timing,
            crate::visual_rack::EdgeTiming::PreviousFrame,
            "each slot keeps its own edge timing"
        );
        assert_eq!(
            runtime.donors[1].source,
            crate::visual_rack::ResolvedImageSource::MissingSelectedLayer {
                saved_position: position(3),
                stage: crate::image_routing::LayerImageStage::PostLocalEffects,
            }
        );
        assert_eq!(
            runtime.motion[0],
            MotionDonor::Selected {
                layer_id: live(21),
                saved_position: position(2),
            }
        );
        assert_eq!(
            runtime.motion[1],
            MotionDonor::Missing {
                saved_position: position(9)
            }
        );
        assert_eq!(
            runtime.selected_layer_ids(),
            [Some(live(21)), None, Some(live(21)), None]
        );
        assert_eq!(runtime.donor_tap(1).unwrap(), runtime.donors[1]);
        assert_eq!(runtime.motion_donor(1).unwrap(), runtime.motion[1]);
        assert!(runtime.donor_tap(2).is_none());
        assert!(runtime.motion_donor(2).is_none());

        // Capture never persists a process identity and keeps the tombstones.
        let mut positions = |id: StableLayerId| (id == live(21)).then(|| position(2));
        let captured = runtime.capture_routes(&mut positions);
        assert_eq!(
            captured.motion[0],
            SavedMotionDonor::Selected {
                saved_position: position(2)
            }
        );
        assert_eq!(
            captured.motion[1],
            SavedMotionDonor::Missing {
                saved_position: position(9)
            }
        );

        // A replacement occupying either vacated position must not rebind.
        let mut everything = |value: SavedLayerPosition| StableLayerId::new(u64::from(value.get()));
        let rebound = RuntimeSymmetryParams::resolve_routes(captured, &mut everything, &|_| true);
        assert!(matches!(
            rebound.donors[1].source,
            crate::visual_rack::ResolvedImageSource::MissingSelectedLayer { .. }
        ));
        assert_eq!(
            rebound.motion[1],
            MotionDonor::Missing {
                saved_position: position(9)
            }
        );

        // A live layer deletion tombstones every slot that named it, and only
        // those slots.
        let mut live_node = RuntimeSymmetryParams::resolve_routes(
            params,
            &mut |value: SavedLayerPosition| StableLayerId::new(u64::from(value.get())),
            &|_| true,
        );
        live_node.mark_layer_output_missing(live(3));
        assert!(matches!(
            live_node.donors[1].source,
            crate::visual_rack::ResolvedImageSource::MissingSelectedLayer { .. }
        ));
        assert!(matches!(live_node.motion[0], MotionDonor::Selected { .. }));
        let mut selected = SavedMotionDonor::Selected {
            saved_position: position(9),
        };
        selected.mark_layer_missing(position(2));
        assert_eq!(selected.selected_position(), Some(position(9)));
        selected.mark_layer_missing(position(9));
        assert_eq!(
            selected,
            SavedMotionDonor::Missing {
                saved_position: position(9)
            }
        );
        assert_eq!(selected.selected_position(), None);
    }

    /// The dynamic-offset record is exactly one kilobyte of whole 16-byte
    /// lanes, and the sector packing is lossless in both directions.
    #[test]
    fn symmetry_gpu_uniforms_are_exactly_one_kilobyte_and_round_trip_every_sector_record() {
        assert_eq!(std::mem::size_of::<SymmetryGpuUniforms>(), 1_024);
        assert_eq!(SymmetryGpuUniforms::BYTES, 1_024);
        assert_eq!(std::mem::offset_of!(SymmetryGpuUniforms, meta), 0);
        assert_eq!(std::mem::offset_of!(SymmetryGpuUniforms, params), 64);
        assert_eq!(std::mem::offset_of!(SymmetryGpuUniforms, motion_rows), 128);
        assert_eq!(std::mem::offset_of!(SymmetryGpuUniforms, sectors), 256);
        assert_eq!(std::mem::offset_of!(SymmetryGpuUniforms, frame), 768);
        assert_eq!(std::mem::offset_of!(SymmetryGpuUniforms, frame_modes), 784);
        assert_eq!(std::mem::offset_of!(SymmetryGpuUniforms, padding), 800);
        assert!(std::mem::size_of::<SymmetryGpuUniforms>().is_multiple_of(16));

        let params = armed_params();
        let table = params.sector_table(NODE_DOMAIN);
        let bindings = SymmetryGpuBindings {
            donor_valid: [true, false],
            motion_valid: [true, true],
            motion_grid: [[120, 68], [240, 135]],
            history_write_index: 5,
            history_valid: 24,
        };
        let packed = SymmetryGpuUniforms::pack(
            params,
            (1920, 1080),
            &table,
            bindings,
            SymmetryFrameUniforms {
                wet: 0.5,
                blend_code: 3,
                time_seconds: 2.25,
            },
        );
        assert_eq!(bytemuck::bytes_of(&packed).len(), 1_024);

        for index in 0..SYMMETRY_SECTOR_RECORDS {
            let record = table.records()[index];
            let decoded = packed.sector(index);
            assert_eq!(decoded.source, record.source, "sector {index}");
            assert_eq!(decoded.motion, record.motion, "sector {index}");
            assert_eq!(decoded.history_age, record.history_age, "sector {index}");
            assert_eq!(
                decoded.hue_offset.to_bits(),
                record.hue_offset.to_bits(),
                "sector {index} hue offset is not lossless"
            );
            assert_eq!(decoded, record);
            assert_eq!(packed.sectors[index][0], record.source.code());
            assert_eq!(packed.sectors[index][1], record.motion_code());
        }

        let domain = params.domain((1920, 1080));
        assert_eq!(
            packed.meta[0],
            [
                domain.mode.code(),
                domain.boundary.code(),
                u32::from(domain.folds),
                domain.rotations,
            ]
        );
        assert_eq!(
            packed.meta[1],
            [
                params.source_mask.bits(),
                params.motion_mask.bits(),
                params.seed,
                domain.orbit_offset,
            ]
        );
        assert_eq!(packed.meta[2], [1, 0, 1, 1]);
        assert_eq!(packed.meta[3], [5, 24, SYMMETRY_SECTOR_RECORDS as u32, 0]);
        assert_eq!(
            packed.params[3],
            [params.motion_gain, params.hue_span, 0.0, 0.0]
        );
        assert_eq!(packed.motion_rows[0][0], 120.0);
        assert_eq!(packed.motion_rows[0][1], 68.0);
        close(packed.motion_rows[0][2], 1.0 / 120.0, 1.0e-7);
        assert_eq!(packed.motion_rows[SYMMETRY_MOTION_ROWS_PER_SLOT][0], 240.0);
        assert_eq!(packed.motion_rows[1], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            packed.motion_rows[SYMMETRY_MOTION_ROWS_PER_SLOT + 1],
            [1.0, 1.0, 0.0, 0.0]
        );
        // The renderer-owned lanes carry the node controls the dedicated pass
        // reads, and the remaining reserved tail is still exactly zero.
        assert_eq!(packed.frame, [0.5, 2.25, 1920.0, 1080.0]);
        assert_eq!(packed.frame_modes, [3, 0, 0, 0]);
        assert_eq!(packed.padding, [[0; 4]; 14], "the reserved tail stays zero");

        // Hostile renderer-owned values sanitize to their neutral fallbacks
        // rather than reaching the shader.
        let hostile = SymmetryGpuUniforms::pack(
            params,
            (1920, 1080),
            &table,
            bindings,
            SymmetryFrameUniforms {
                wet: f32::NAN,
                blend_code: 0,
                time_seconds: f32::NEG_INFINITY,
            },
        );
        assert_eq!(hostile.frame[0], 1.0);
        assert_eq!(hostile.frame[1], 0.0);

        // An absent motion grid never divides by zero.
        let unbound = SymmetryGpuUniforms::pack(
            params,
            (1920, 1080),
            &table,
            SymmetryGpuBindings::default(),
            SymmetryFrameUniforms::default(),
        );
        assert_eq!(unbound.motion_rows[0], [0.0; 4]);
        assert_eq!(unbound.meta[2], [0; 4]);

        // The exact-default node packs the neutral table and declares itself a
        // bypass in the record the pass reads.
        let neutral = SymmetryGpuUniforms::pack(
            SymmetryParams::default(),
            (1920, 1080),
            &SymmetrySectorTable::NEUTRAL,
            SymmetryGpuBindings::default(),
            SymmetryFrameUniforms::default(),
        );
        assert_eq!(neutral.meta[3][3], 1);
        assert_eq!(neutral.sectors, [[0_u32, 0, 0, 0]; SYMMETRY_SECTOR_RECORDS]);
    }

    fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
        let start = source
            .find(signature)
            .unwrap_or_else(|| panic!("{signature} is missing from the module source"));
        let body = &source[start..];
        let open = body.find('\n').expect("a signature line ends in a newline");
        let end = body
            .find("\n}\n")
            .expect("a top level function body ends at column zero");
        &body[open..end]
    }

    fn module_source() -> &'static str {
        let source = include_str!("symmetry.rs");
        let tests = source
            .find("#[cfg(test)]\nmod tests")
            .expect("the test module marker is present");
        &source[..tests]
    }

    #[test]
    fn the_mode_and_boundary_vocabularies_are_closed_and_their_codes_are_append_only() {
        let expected_modes: [(SymmetryMode, u32); 8] = [
            (SymmetryMode::Cyclic, 0),
            (SymmetryMode::Dihedral, 1),
            (SymmetryMode::PlanarP1, 2),
            (SymmetryMode::PlanarPm, 3),
            (SymmetryMode::PlanarP2, 4),
            (SymmetryMode::PlanarPmm, 5),
            (SymmetryMode::LogSpiral, 6),
            (SymmetryMode::Orbit, 7),
        ];
        assert_eq!(SymmetryMode::ALL.len(), expected_modes.len());
        for (index, (mode, code)) in expected_modes.into_iter().enumerate() {
            assert_eq!(mode.code(), code);
            assert_eq!(SymmetryMode::ALL[index], mode);
        }

        let expected_boundaries: [(SymmetryBoundary, u32); 5] = [
            (SymmetryBoundary::Transparent, 0),
            (SymmetryBoundary::Mirror, 1),
            (SymmetryBoundary::Wrap, 2),
            (SymmetryBoundary::Hold, 3),
            (SymmetryBoundary::CellularReentry, 4),
        ];
        assert_eq!(SymmetryBoundary::ALL.len(), expected_boundaries.len());
        for (index, (boundary, code)) in expected_boundaries.into_iter().enumerate() {
            assert_eq!(boundary.code(), code);
            assert_eq!(SymmetryBoundary::ALL[index], boundary);
        }
        assert_eq!(SymmetryMode::default(), SymmetryMode::Cyclic);
        assert_eq!(SymmetryBoundary::default(), SymmetryBoundary::Transparent);
    }

    #[test]
    fn every_mode_closes_its_composition_table_and_returns_to_identity_after_its_full_order() {
        for mode in SymmetryMode::ALL {
            for folds in [1.0_f32, 2.0, 3.0, 6.0, 32.0] {
                let domain = domain_for(mode, folds);
                let elements = domain.point_group();
                assert_eq!(
                    elements.len() as u32,
                    mode.point_group_order(domain.folds),
                    "{mode:?} declares a point group order it does not enumerate"
                );

                for &left in &elements {
                    for &right in &elements {
                        let product = left.compose(right, domain.rotations);
                        assert!(
                            elements.contains(&product),
                            "{mode:?} composition escaped the group: {left:?} . {right:?}"
                        );
                    }
                    let inverse = left.inverse(domain.rotations);
                    assert!(elements.contains(&inverse));
                    assert!(left.compose(inverse, domain.rotations).is_identity());
                    assert!(inverse.compose(left, domain.rotations).is_identity());
                }

                let mut accumulated = SymmetryElement::IDENTITY;
                for step in 1..=domain.rotations {
                    accumulated = accumulated.compose(domain.generator(), domain.rotations);
                    assert_eq!(
                        accumulated.is_identity(),
                        step == domain.rotations,
                        "{mode:?} generator returned to identity after {step} of {} steps",
                        domain.rotations
                    );
                }

                if let Some(reflection) = domain.reflection_generator() {
                    assert!(reflection
                        .compose(reflection, domain.rotations)
                        .is_identity());
                }
            }
        }
    }

    #[test]
    fn the_group_action_agrees_with_the_composition_table_for_every_mode_and_element_pair() {
        for mode in SymmetryMode::ALL {
            let domain = domain_for(mode, 6.0);
            let elements = test_elements(&domain);
            for point in probes(&domain) {
                for &left in &elements {
                    for &right in &elements {
                        let composed = domain.apply(left.compose(right, domain.rotations), point);
                        let sequential = domain.apply(left, domain.apply(right, point));
                        close_point(composed, sequential, 2.0e-3);
                    }
                }
            }
        }
    }

    #[test]
    fn the_translation_lattice_is_a_normal_subgroup_the_planar_point_groups_conjugate_into_itself()
    {
        for mode in SymmetryMode::ALL
            .into_iter()
            .filter(|mode| mode.has_lattice())
        {
            let domain = domain_for(mode, 4.0);
            let order = domain.rotations;
            let translations: [SymmetryElement; 4] =
                [[1, 0], [0, 1], [-2, 1], [3, -2]].map(|lattice| SymmetryElement {
                    lattice,
                    ..SymmetryElement::IDENTITY
                });

            // The translations form an abelian subgroup under composition.
            for left in translations {
                for right in translations {
                    let product = left.compose(right, order);
                    assert!(!product.reflected && product.rotation == 0);
                    assert_eq!(
                        product.lattice,
                        [
                            left.lattice[0] + right.lattice[0],
                            left.lattice[1] + right.lattice[1]
                        ]
                    );
                    assert_eq!(product, right.compose(left, order));
                }
                assert!(left.compose(left.inverse(order), order).is_identity());
                assert!(left.inverse(order).compose(left, order).is_identity());
            }

            // Conjugating a translation by any point element lands back inside
            // the translation subgroup: the lattice is normal.
            for point_element in domain.point_group() {
                for translation in translations {
                    let conjugated = point_element
                        .compose(translation, order)
                        .compose(point_element.inverse(order), order);
                    assert!(
                        !conjugated.reflected && conjugated.rotation == 0,
                        "{mode:?} conjugation left the translation subgroup"
                    );
                    assert_eq!(
                        conjugated.lattice.map(i32::abs).into_iter().sum::<i32>(),
                        translation.lattice.map(i32::abs).into_iter().sum::<i32>(),
                        "{mode:?} conjugation changed the translation length"
                    );
                }
            }
        }
    }

    #[test]
    fn iterating_the_generator_action_returns_every_probe_point_to_itself_for_every_mode() {
        for mode in SymmetryMode::ALL {
            for folds in [2.0_f32, 3.0, 6.0] {
                let domain = domain_for(mode, folds);
                for point in probes(&domain) {
                    let mut walked = point;
                    for _ in 0..domain.rotations {
                        walked = domain.apply(domain.generator(), walked);
                    }
                    close_point(walked, point, 2.0e-3);
                }
            }
        }
    }

    #[test]
    fn classification_recovers_every_sample_through_its_own_group_element_for_every_mode() {
        for mode in SymmetryMode::ALL {
            for folds in [1.0_f32, 2.0, 5.0, 8.0] {
                let domain = domain_for(mode, folds);
                for point in probes(&domain) {
                    let classified = domain.classify(point);
                    let restored = domain.apply(classified.element, classified.local);
                    close_point(restored, point, 2.0e-3);
                    assert!(classified.raw_sector < u32::from(domain.folds));
                    assert!(
                        domain.sector_of(classified.raw_sector) < SYMMETRY_SECTOR_RECORDS as u32
                    );
                }
            }
        }
    }

    #[test]
    fn every_folded_sample_lands_inside_the_declared_fundamental_domain_for_every_mode() {
        for mode in SymmetryMode::ALL {
            let domain = domain_for(mode, 6.0);
            for step in 0..64 {
                let angle = TAU * step as f32 / 64.0;
                let point = [0.37 * angle.cos(), 0.37 * angle.sin()];
                let classified = domain.classify(domain.canonical_point(point));
                if domain.mode.has_lattice() {
                    for (axis, extent) in domain.cell_extent().into_iter().enumerate() {
                        let local = classified.local[axis];
                        assert!(
                            local >= -1.0e-4 && local <= extent + 1.0e-4,
                            "{mode:?} folded outside its declared cell extent on axis {axis}: {local}"
                        );
                    }
                } else {
                    let upper = domain.wedge_angle();
                    let local_angle = classified.local[1]
                        .atan2(classified.local[0])
                        .rem_euclid(TAU);
                    assert!(
                        local_angle <= upper + 1.0e-3 || local_angle >= TAU - 1.0e-3,
                        "{mode:?} folded outside its wedge: {local_angle} > {upper}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_sector_boundary_is_continuous_exactly_where_the_group_says_it_is_for_every_mode() {
        const EPSILON: f32 = 1.0e-4;
        for mode in SymmetryMode::ALL {
            let domain = domain_for(mode, 6.0);
            let walls = mode.mirrored_walls();
            let probe_pairs: [([f32; 2], [f32; 2]); 2] = if mode.has_lattice() {
                [
                    ([-EPSILON, 0.23], [EPSILON, 0.23]),
                    ([0.23, -EPSILON], [0.23, EPSILON]),
                ]
            } else {
                let radius = 0.37;
                let upper = if mode.has_reflection() {
                    domain.sector_width * 0.5
                } else {
                    domain.sector_width
                };
                let polar = |angle: f32| [radius * angle.cos(), radius * angle.sin()];
                [
                    (polar(upper - EPSILON), polar(upper + EPSILON)),
                    (polar(-EPSILON), polar(EPSILON)),
                ]
            };

            for (wall, (below, above)) in probe_pairs.into_iter().enumerate() {
                let low = domain.classify(domain.canonical_point(below)).local;
                let high = domain.classify(domain.canonical_point(above)).local;
                let jump = distance(low, high);
                if walls[wall] {
                    assert!(
                        jump <= 1.0e-2,
                        "{mode:?} declared wall {wall} mirrored but it jumps by {jump}"
                    );
                } else {
                    assert!(
                        jump >= 0.1,
                        "{mode:?} declared wall {wall} seamed but it only moves {jump}"
                    );
                }
            }
        }
    }

    #[test]
    fn effective_folds_rounds_the_modulated_sum_exactly_once_and_then_clamps_to_one_through_thirty_two(
    ) {
        // Rounding each input separately would give four; the modulated sum is
        // 4.8 and rounds to five exactly once.
        let modulated = SymmetryParams {
            base_folds: 2.4,
            fold_offset: 2.4,
            ..SymmetryParams::default()
        };
        assert_eq!(modulated.effective_folds(), 5);
        assert_eq!(
            modulated.base_folds.round() + modulated.fold_offset.round(),
            4.0
        );

        // A fractional offset alone must still move the fold count.
        let nudged = SymmetryParams {
            base_folds: 3.0,
            fold_offset: 0.6,
            ..SymmetryParams::default()
        };
        assert_eq!(nudged.effective_folds(), 4);

        let below = SymmetryParams {
            base_folds: 2.0,
            fold_offset: -40.0,
            ..SymmetryParams::default()
        };
        assert_eq!(below.effective_folds(), SYMMETRY_MIN_FOLDS);
        let above = SymmetryParams {
            base_folds: 30.0,
            fold_offset: 40.0,
            ..SymmetryParams::default()
        };
        assert_eq!(above.effective_folds(), SYMMETRY_MAX_FOLDS);

        for hostile in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let params = SymmetryParams {
                base_folds: hostile,
                fold_offset: hostile,
                ..SymmetryParams::default()
            };
            assert_eq!(params.effective_folds(), 1);
        }
    }

    #[test]
    fn effective_folds_is_the_only_place_the_symmetry_domain_rounds_a_fold_count() {
        let module = module_source();
        assert_eq!(
            module.matches(".round()").count(),
            1,
            "another rounding site appeared outside effective_folds"
        );
        assert!(function_body(module, "pub fn effective_folds(").contains(".round()"));

        // The geometry must consume exactly the count effective_folds froze, so
        // no second rounding can hide inside the domain construction.
        for step in 0..80 {
            let params = SymmetryParams {
                mode: SymmetryMode::Cyclic,
                base_folds: 1.0 + step as f32 * 0.37,
                fold_offset: step as f32 * 0.11,
                ..SymmetryParams::default()
            };
            let folds = params.effective_folds();
            let domain = params.domain(SQUARE);
            assert_eq!(domain.folds, folds);
            close(domain.sector_width, TAU / f32::from(folds), 1.0e-6);
            close(domain.cell_period, 1.0 / f32::from(folds), 1.0e-6);
        }
    }

    #[test]
    fn radial_phase_rotates_the_sector_origin_and_carries_the_folded_coordinate_with_it() {
        let base = SymmetryParams {
            mode: SymmetryMode::Cyclic,
            base_folds: 8.0,
            ..SymmetryParams::default()
        };
        let sector_degrees = 360.0 / 8.0;
        let sample = [0.86, 0.62];

        let unshifted = base.fold(sample, SQUARE);
        let shifted = SymmetryParams {
            radial_phase_deg: sector_degrees,
            ..base
        }
        .fold(sample, SQUARE);

        // Rotating the origin by exactly one sector turns the fundamental wedge
        // by one sector, so the folded coordinate rotates with it.
        let expected = {
            let domain = base.domain(SQUARE);
            let offset = [
                unshifted.uv[0] - domain.center[0],
                unshifted.uv[1] - domain.center[1],
            ];
            let turned = apply_2x2(domain.rotation_in_uv_space(1), offset);
            [turned[0] + domain.center[0], turned[1] + domain.center[1]]
        };
        close_point(shifted.uv, expected, 1.0e-3);
        assert!(
            distance(shifted.uv, unshifted.uv) > 0.05,
            "radial phase must move the folded coordinate, not only relabel it"
        );
        assert_ne!(shifted.element.rotation, unshifted.element.rotation);
    }

    #[test]
    fn orbit_phase_rotates_sector_classification_without_moving_the_folded_coordinate() {
        for mode in SymmetryMode::ALL {
            let base = SymmetryParams {
                mode,
                base_folds: 8.0,
                spiral_scale: 0.8,
                orbit_radius: 0.3,
                cell_skew: 0.25,
                ..SymmetryParams::default()
            };
            let sample = [0.71, 0.29];
            let unshifted = base.fold(sample, WIDE);
            for (phase, steps) in [(0.25_f32, 2_u32), (0.5, 4), (0.875, 7)] {
                let shifted = SymmetryParams {
                    orbit_phase: phase,
                    ..base
                }
                .fold(sample, WIDE);
                // Classification only: the geometry is bit-identical.
                assert_eq!(
                    shifted.uv, unshifted.uv,
                    "{mode:?} orbit phase moved the folded coordinate"
                );
                assert_eq!(shifted.element, unshifted.element);
                assert_eq!(shifted.sector, (unshifted.sector + steps) % 8);
            }
        }
    }

    #[test]
    fn planar_axis_rotates_the_lattice_basis_and_stays_physical_on_a_non_square_output() {
        let upright = SymmetryParams {
            mode: SymmetryMode::PlanarP1,
            base_folds: 4.0,
            ..SymmetryParams::default()
        };
        let turned = SymmetryParams {
            planar_axis_deg: 90.0,
            ..upright
        };

        let upright_basis = upright.domain(WIDE).lattice_basis_in_uv();
        let turned_basis = turned.domain(WIDE).lattice_basis_in_uv();

        // The primary lattice vector is horizontal at zero degrees and vertical
        // at ninety, and the aspect conjugation is what keeps the physical
        // right angle a right angle in UV.
        let aspect = 1920.0 / 1080.0;
        close_point(
            [upright_basis[0][0], upright_basis[1][0]],
            [0.25 / aspect, 0.0],
            1.0e-6,
        );
        close_point(
            [turned_basis[0][0], turned_basis[1][0]],
            [0.0, 0.25],
            1.0e-6,
        );

        // The phase is untouched by the axis: the basis change is a rotation,
        // never a translation.
        assert_eq!(upright.domain(WIDE).planar_phase, 0.0);
        assert_eq!(turned.domain(WIDE).planar_phase, 0.0);

        // A physical square cell must remain a physical square cell.
        let primary = [turned_basis[0][0] * aspect, turned_basis[1][0]];
        let secondary = [turned_basis[0][1] * aspect, turned_basis[1][1]];
        close(primary[0].hypot(primary[1]), 0.25, 1.0e-6);
        close(secondary[0].hypot(secondary[1]), 0.25, 1.0e-6);
        close(
            primary[0] * secondary[0] + primary[1] * secondary[1],
            0.0,
            1.0e-6,
        );
    }

    #[test]
    fn planar_phase_translates_the_primary_lattice_coordinate_by_exactly_one_cell_period() {
        for mode in [
            SymmetryMode::PlanarP1,
            SymmetryMode::PlanarPm,
            SymmetryMode::PlanarP2,
            SymmetryMode::PlanarPmm,
        ] {
            let base = SymmetryParams {
                mode,
                base_folds: 4.0,
                ..SymmetryParams::default()
            };
            let sample = [0.63, 0.41];
            let unshifted = base.fold(sample, SQUARE);
            let whole = SymmetryParams {
                planar_phase: 1.0,
                ..base
            }
            .fold(sample, SQUARE);
            let half = SymmetryParams {
                planar_phase: 0.5,
                ..base
            }
            .fold(sample, SQUARE);

            // One whole cell period leaves the folded coordinate exactly where
            // it was and advances the primary cell index by exactly one.
            close_point(whole.uv, unshifted.uv, 1.0e-5);
            assert_eq!(whole.element.lattice[0], unshifted.element.lattice[0] + 1);
            assert_eq!(whole.element.lattice[1], unshifted.element.lattice[1]);

            // Half a period is a real translation, not a relabel.
            assert!(distance(half.uv, unshifted.uv) > 1.0e-3);

            // The basis is never rotated by a phase.
            assert_eq!(
                base.domain(SQUARE).lattice_basis_in_uv(),
                SymmetryParams {
                    planar_phase: 1.0,
                    ..base
                }
                .domain(SQUARE)
                .lattice_basis_in_uv()
            );
        }
    }

    #[test]
    fn analytic_sector_fixtures_hold_for_the_radial_and_planar_families() {
        // Cyclic C4 on a square output: the point one quarter turn around folds
        // onto the same wedge coordinate through sector one.
        let cyclic = SymmetryParams {
            mode: SymmetryMode::Cyclic,
            base_folds: 4.0,
            ..SymmetryParams::default()
        };
        let east = cyclic.fold([0.9, 0.5], SQUARE);
        close_point(east.uv, [0.9, 0.5], 1.0e-6);
        assert_eq!(east.sector, 0);
        assert_eq!(east.element, SymmetryElement::rotation(0));

        let south = cyclic.fold([0.5, 0.9], SQUARE);
        close_point(south.uv, [0.9, 0.5], 1.0e-5);
        assert_eq!(south.sector, 1);
        assert_eq!(south.element, SymmetryElement::rotation(1));

        // Dihedral D4: a sample past the wedge midpoint mirrors back and the
        // element it reports carries it exactly onto the sample again.
        let dihedral = SymmetryParams {
            mode: SymmetryMode::Dihedral,
            base_folds: 4.0,
            ..SymmetryParams::default()
        };
        let domain = dihedral.domain(SQUARE);
        let angle = 80.0_f32.to_radians();
        let group_point = [0.3 * angle.cos(), 0.3 * angle.sin()];
        let classified = domain.classify(group_point);
        assert!(classified.element.reflected);
        assert_eq!(classified.element.rotation, 1);
        close(
            classified.local[1].atan2(classified.local[0]),
            10.0_f32.to_radians(),
            1.0e-4,
        );
        close_point(
            domain.apply(classified.element, classified.local),
            group_point,
            1.0e-5,
        );

        // Planar p1 with two cells across the normalized span: a sample one
        // whole cell to the right folds back by exactly one cell period.
        let planar = SymmetryParams {
            mode: SymmetryMode::PlanarP1,
            base_folds: 2.0,
            ..SymmetryParams::default()
        };
        let inside = planar.fold([0.6, 0.55], SQUARE);
        close_point(inside.uv, [0.6, 0.55], 1.0e-6);
        assert_eq!(inside.element.lattice, [0, 0]);
        let shifted = planar.fold([0.6 + 0.5, 0.55], SQUARE);
        close_point(shifted.uv, [0.6, 0.55], 1.0e-5);
        assert_eq!(shifted.element.lattice, [1, 0]);
        assert_eq!(shifted.sector, 1);
    }

    #[test]
    fn a_non_square_output_keeps_authored_sector_angles_physical() {
        let params = SymmetryParams {
            mode: SymmetryMode::Cyclic,
            base_folds: 8.0,
            ..SymmetryParams::default()
        };
        let aspect = 1920.0_f32 / 1080.0;
        let radius = 0.3_f32;

        // Two samples straddling the physical forty-five degree wall. Dropping
        // the aspect conjugation would report both as sector one.
        for (degrees, expected_sector) in [(44.0_f32, 0_u32), (46.0, 1)] {
            let angle = degrees.to_radians();
            let uv = [
                0.5 + radius * angle.cos() / aspect,
                0.5 + radius * angle.sin(),
            ];
            let folded = params.fold(uv, WIDE);
            assert_eq!(
                folded.sector, expected_sector,
                "{degrees} degrees physical classified as sector {}",
                folded.sector
            );
        }

        // The same authored angles on a square output agree, which is what
        // makes the wide-output result a statement about aspect and not about
        // the fold.
        for (degrees, expected_sector) in [(44.0_f32, 0_u32), (46.0, 1)] {
            let angle = degrees.to_radians();
            let uv = [0.5 + radius * angle.cos(), 0.5 + radius * angle.sin()];
            assert_eq!(params.fold(uv, SQUARE).sector, expected_sector);
        }
    }

    #[test]
    fn every_boundary_law_has_its_analytic_fixture_and_only_transparent_removes_coverage() {
        let outside = [1.25_f32, 0.5];

        let (transparent, covered) = SymmetryBoundary::Transparent.resolve(outside);
        assert!(!covered);
        close_point(transparent, [1.0, 0.5], 0.0);
        let (_, inside_covered) = SymmetryBoundary::Transparent.resolve([0.25, 0.5]);
        assert!(inside_covered);

        let (hold, covered) = SymmetryBoundary::Hold.resolve(outside);
        assert!(covered);
        close_point(hold, [1.0, 0.5], 0.0);

        let (wrap, covered) = SymmetryBoundary::Wrap.resolve(outside);
        assert!(covered);
        close_point(wrap, [0.25, 0.5], 1.0e-6);
        // The floor based wrap is what makes negative input correct.
        close_point(
            SymmetryBoundary::Wrap.resolve([-0.25, 0.5]).0,
            [0.75, 0.5],
            1.0e-6,
        );

        let (mirror, covered) = SymmetryBoundary::Mirror.resolve(outside);
        assert!(covered);
        close_point(mirror, [0.75, 0.5], 1.0e-6);
        close_point(
            SymmetryBoundary::Mirror.resolve([-0.25, 0.5]).0,
            [0.25, 0.5],
            1.0e-6,
        );

        // Cell [1, 0] selects D4 element one: a single quarter turn about the
        // cell center with no reflection.
        let (cellular, covered) = SymmetryBoundary::CellularReentry.resolve(outside);
        assert!(covered);
        close_point(cellular, [0.5, 0.25], 1.0e-6);

        for boundary in SymmetryBoundary::ALL {
            for hostile in [[f32::NAN, 0.5], [f32::INFINITY, f32::NEG_INFINITY]] {
                let (resolved, _) = boundary.resolve(hostile);
                assert!(resolved[0].is_finite() && resolved[1].is_finite());
            }
        }
    }

    #[test]
    fn cellular_reentry_is_one_total_cell_transform_with_no_self_call_and_no_loop() {
        // Structural: the transform reads one cell coordinate and returns. No
        // self-call, no loop, no iteration count anywhere in the three
        // functions that make it up.
        let module = module_source();
        for signature in [
            "pub fn cellular_reentry(",
            "fn d4_cell_element(",
            "fn apply_d4(",
        ] {
            let body = function_body(module, signature);
            assert!(
                !body.contains("cellular_reentry("),
                "{signature} calls back into the re-entry transform"
            );
            for construct in ["loop", "while ", "for ", ".iter(", "recurs"] {
                assert!(
                    !body.contains(construct),
                    "{signature} contains the iterating construct {construct}"
                );
            }
        }

        // Total: one application always lands inside the unit cell, however far
        // outside the input started, so a second application is never needed.
        for value in [
            -1.0e9_f32, -12345.75, -3.25, -0.001, 0.0, 0.5, 1.25, 7.5, 12345.75, 1.0e9,
        ] {
            let reentered = cellular_reentry([value, -value]);
            assert!(
                (0.0..=1.0).contains(&reentered[0]) && (0.0..=1.0).contains(&reentered[1]),
                "{value} left the unit cell at {reentered:?}"
            );
        }

        // A function of one cell coordinate: the element choice depends on the
        // cell index alone and repeats with a period of four cells in each
        // axis, so nothing carried across cells can influence it.
        let fraction = [0.3_f32, 0.7];
        for x in -2..2 {
            for y in -2..2 {
                let cell = [x as f32, y as f32];
                let shifted_cell = [(x + 4) as f32, (y + 4) as f32];
                assert_eq!(d4_cell_element(cell), d4_cell_element(shifted_cell));
                close_point(
                    cellular_reentry([fraction[0] + cell[0], fraction[1] + cell[1]]),
                    cellular_reentry([
                        fraction[0] + shifted_cell[0],
                        fraction[1] + shifted_cell[1],
                    ]),
                    1.0e-6,
                );
            }
        }
    }

    #[test]
    fn the_cellular_reentry_cell_transform_is_the_eight_element_d4_group() {
        let probe = [0.3_f32, 0.65];
        let mut images = Vec::new();
        for element in 0..8_u8 {
            let image = apply_d4(element, probe);
            assert!(
                !images.contains(&image),
                "D4 element {element} duplicated another element"
            );
            images.push(image);
            // Every element is an isometry about the cell center.
            close(
                (image[0] - 0.5).hypot(image[1] - 0.5),
                (probe[0] - 0.5).hypot(probe[1] - 0.5),
                1.0e-6,
            );
        }
        assert_eq!(images.len(), 8);
        // Closure: composing any two elements reproduces a member of the set.
        for left in 0..8_u8 {
            for right in 0..8_u8 {
                let composed = apply_d4(left, apply_d4(right, probe));
                assert!(
                    images
                        .iter()
                        .any(|image| distance(*image, composed) <= 1.0e-6),
                    "D4 composition {left} . {right} escaped the group"
                );
            }
        }
        // Every cell selects an element that is actually in the table.
        for x in -8..8 {
            for y in -8..8 {
                assert!(d4_cell_element([x as f32, y as f32]) < 8);
            }
        }
    }

    #[test]
    fn the_bounded_log_spiral_closes_on_its_quotient_and_degenerates_instead_of_collapsing() {
        let active = domain_for(SymmetryMode::LogSpiral, 6.0);
        let period = active
            .spiral_period()
            .expect("an armed spiral has a period");
        assert!(period <= SPIRAL_MAX_LOG_PERIOD);
        close(f32::from(active.folds) * active.spiral_step, period, 1.0e-4);

        // Every canonical point sits in one bounded annulus, and canonicalizing
        // twice changes nothing.
        for point in probes(&active) {
            let radius = point[0].hypot(point[1]);
            let anchor = SPIRAL_ANCHOR_RADIUS.ln();
            assert!(radius.ln() >= anchor - 1.0e-3);
            assert!(radius.ln() <= anchor + period + 1.0e-3);
            close_point(active.canonical_point(point), point, 1.0e-4);
        }

        // Below the minimum period the spiral degenerates to pure cyclic
        // geometry rather than collapsing every radius onto one circle.
        let dormant = SymmetryParams {
            mode: SymmetryMode::LogSpiral,
            base_folds: 6.0,
            spiral_scale: 0.01,
            ..SymmetryParams::default()
        }
        .domain(SQUARE);
        assert_eq!(dormant.spiral_period(), None);
        assert_eq!(dormant.spiral_step, 0.0);
        let point = [0.4_f32, -0.1];
        close_point(dormant.canonical_point(point), point, 0.0);
        close_point(
            dormant.apply(dormant.generator(), point),
            rotate(point, dormant.sector_width),
            1.0e-6,
        );
    }

    #[test]
    fn the_exact_default_is_a_bypass_that_passes_its_carrier_coordinate_through_untouched() {
        let params = SymmetryParams::default();
        assert_eq!(params.mode, SymmetryMode::Cyclic);
        assert_eq!(params.effective_folds(), 1);
        assert_eq!(params.center, [0.5, 0.5]);
        assert_eq!(params.boundary, SymmetryBoundary::Transparent);
        assert!(params.is_exact_bypass());
        assert!(params.domain(WIDE).exact_bypass);

        for uv in [[0.0_f32, 0.0], [0.37, 0.91], [1.0, 1.0], [0.5, 0.5]] {
            let folded = params.fold(uv, WIDE);
            // Bit identical, not merely close: the bypass must not travel
            // through the aspect round trip.
            assert_eq!(folded.uv, uv);
            assert_eq!(folded.sector, 0);
            assert_eq!(folded.element, SymmetryElement::IDENTITY);
            assert!(folded.covered);
        }

        // Anything that actually changes the geometry stops being a bypass.
        for active in [
            SymmetryParams {
                base_folds: 2.0,
                ..params
            },
            SymmetryParams {
                fold_offset: 1.0,
                ..params
            },
            SymmetryParams {
                radial_phase_deg: 5.0,
                ..params
            },
            SymmetryParams {
                orbit_phase: 0.5,
                ..params
            },
        ] {
            assert!(!active.is_exact_bypass());
        }
        // A non-cyclic mode is never claimed as a bypass even at one fold.
        for mode in SymmetryMode::ALL {
            let single = SymmetryParams { mode, ..params };
            assert_eq!(single.is_exact_bypass(), mode == SymmetryMode::Cyclic);
        }
    }

    #[test]
    fn hostile_authored_values_sanitize_to_neutral_fallbacks_rather_than_clamped_extremes() {
        let hostile = SymmetryParams {
            mode: SymmetryMode::Orbit,
            base_folds: f32::NAN,
            fold_offset: f32::INFINITY,
            radial_phase_deg: f32::NEG_INFINITY,
            orbit_phase: f32::NAN,
            planar_axis_deg: f32::NAN,
            planar_phase: f32::INFINITY,
            cell_skew: f32::NAN,
            spiral_scale: f32::NEG_INFINITY,
            orbit_radius: f32::NAN,
            orbit_spin_deg: f32::INFINITY,
            center: [f32::NAN, f32::INFINITY],
            boundary: SymmetryBoundary::CellularReentry,
            motion_gain: f32::NAN,
            hue_span: f32::NEG_INFINITY,
            seed: 7,
            // An entirely empty source mask is not a legal draw domain.
            source_mask: SymmetrySourceMask {
                carrier: false,
                donor0: false,
                donor1: false,
                clean_history: false,
            },
            motion_mask: SymmetryMotionMask {
                slot0: true,
                slot1: false,
            },
            donors: SymmetryParams::default().donors,
            motion: SymmetryParams::default().motion,
        };
        let clean = hostile.sanitized();
        assert_eq!(clean.motion_gain, 0.0);
        assert_eq!(clean.hue_span, 0.0);
        assert_eq!(clean.seed, 7);
        assert_eq!(clean.source_mask, SymmetrySourceMask::CARRIER_ONLY);
        assert!(clean.motion_mask.slot0);
        assert_eq!(clean.mode, SymmetryMode::Orbit);
        assert_eq!(clean.boundary, SymmetryBoundary::CellularReentry);
        assert_eq!(clean.base_folds, 1.0);
        assert_eq!(clean.fold_offset, 0.0);
        assert_eq!(clean.radial_phase_deg, 0.0);
        assert_eq!(clean.orbit_phase, 0.0);
        assert_eq!(clean.planar_axis_deg, 0.0);
        assert_eq!(clean.planar_phase, 0.0);
        assert_eq!(clean.cell_skew, 0.0);
        assert_eq!(clean.spiral_scale, 0.0);
        assert_eq!(clean.orbit_radius, 0.0);
        assert_eq!(clean.orbit_spin_deg, 0.0);
        assert_eq!(clean.center, [0.5, 0.5]);
        assert_eq!(clean.sanitized(), clean);

        // Out of range finite values clamp; they do not fall back.
        let extreme = SymmetryParams {
            base_folds: 900.0,
            cell_skew: -4.0,
            orbit_radius: 9.0,
            center: [-40.0, 40.0],
            ..SymmetryParams::default()
        }
        .sanitized();
        assert_eq!(extreme.base_folds, f32::from(SYMMETRY_MAX_FOLDS));
        assert_eq!(extreme.cell_skew, -CELL_SKEW_LIMIT);
        assert_eq!(extreme.orbit_radius, ORBIT_RADIUS_LIMIT);
        assert_eq!(extreme.center, [CENTER_MIN, CENTER_MAX]);

        // A hostile sample never produces a non-finite folded coordinate.
        for mode in SymmetryMode::ALL {
            for boundary in SymmetryBoundary::ALL {
                let params = SymmetryParams {
                    mode,
                    base_folds: 5.0,
                    boundary,
                    ..SymmetryParams::default()
                };
                for uv in [[f32::NAN, 0.5], [f32::INFINITY, f32::NEG_INFINITY]] {
                    let folded = params.fold(uv, WIDE);
                    assert!(folded.uv[0].is_finite() && folded.uv[1].is_finite());
                    assert!(folded.sector < SYMMETRY_SECTOR_RECORDS as u32);
                }
            }
        }
    }

    #[test]
    fn the_sector_table_width_and_history_age_bounds_reuse_the_committed_ring() {
        assert_eq!(SYMMETRY_SECTOR_RECORDS, 32);
        assert_eq!(usize::from(SYMMETRY_MAX_FOLDS), SYMMETRY_SECTOR_RECORDS);
        assert_eq!(SYMMETRY_MAX_HISTORY_AGE, TEMPORAL_HISTORY_LEN - 1);
        assert_eq!(SYMMETRY_MAX_HISTORY_AGE, 23);
        assert!(history_age_is_in_domain(0));
        assert!(history_age_is_in_domain(23));
        assert!(!history_age_is_in_domain(24));
        assert!(!history_age_is_in_domain(TEMPORAL_HISTORY_LEN));

        // A sector index is always a legal record index, at any fold count.
        for folds in 1..=SYMMETRY_MAX_FOLDS {
            let params = SymmetryParams {
                mode: SymmetryMode::Dihedral,
                base_folds: f32::from(folds),
                orbit_phase: 0.9,
                ..SymmetryParams::default()
            };
            for step in 0..37 {
                let angle = TAU * step as f32 / 37.0;
                let uv = [0.5 + 0.3 * angle.cos(), 0.5 + 0.3 * angle.sin()];
                let sector = params.fold(uv, WIDE).sector;
                assert!(sector < u32::from(folds));
                assert!((sector as usize) < SYMMETRY_SECTOR_RECORDS);
            }
        }
    }

    #[test]
    fn symmetry_params_round_trip_through_serde_and_default_every_absent_field() {
        let params = SymmetryParams {
            mode: SymmetryMode::PlanarPmm,
            base_folds: 7.0,
            fold_offset: -1.5,
            radial_phase_deg: 30.0,
            orbit_phase: 0.25,
            planar_axis_deg: -12.0,
            planar_phase: 0.75,
            cell_skew: 0.4,
            spiral_scale: -0.6,
            orbit_radius: 0.2,
            orbit_spin_deg: 15.0,
            center: [0.4, 0.6],
            boundary: SymmetryBoundary::CellularReentry,
            motion_gain: -0.5,
            hue_span: 0.25,
            seed: 4_242,
            source_mask: SymmetrySourceMask {
                carrier: true,
                donor0: false,
                donor1: true,
                clean_history: true,
            },
            motion_mask: SymmetryMotionMask {
                slot0: false,
                slot1: true,
            },
            donors: [
                SavedImageTap {
                    source: crate::visual_rack::SavedImageSource::CleanProgram,
                    timing: crate::visual_rack::EdgeTiming::PreviousFrame,
                },
                SavedImageTap {
                    source: crate::visual_rack::SavedImageSource::AllBelow,
                    timing: crate::visual_rack::EdgeTiming::CurrentFrame,
                },
            ],
            motion: [
                SavedMotionDonor::None,
                SavedMotionDonor::Missing {
                    saved_position: SavedLayerPosition::new(3).expect("a nonzero position"),
                },
            ],
        };
        let encoded = serde_yaml::to_string(&params).expect("params serialize");
        assert!(encoded.contains("planar_pmm"));
        assert!(encoded.contains("cellular_reentry"));
        assert!(encoded.contains("clean_program"));
        assert!(encoded.contains("previous_frame"));
        let decoded: SymmetryParams = serde_yaml::from_str(&encoded).expect("params deserialize");
        assert_eq!(decoded, params);

        let empty: SymmetryParams = serde_yaml::from_str("{}").expect("an empty map deserializes");
        assert_eq!(empty, SymmetryParams::default());
        assert!(empty.is_exact_bypass());

        let partial: SymmetryParams = serde_yaml::from_str("mode: dihedral\nbase_folds: 6.0\n")
            .expect("partial deserializes");
        assert_eq!(partial.mode, SymmetryMode::Dihedral);
        assert_eq!(partial.base_folds, 6.0);
        assert_eq!(partial.center, SymmetryParams::default().center);
        assert_eq!(partial.boundary, SymmetryBoundary::Transparent);
    }

    #[test]
    fn the_saved_and_runtime_slot_admission_answers_agree_slot_for_slot_under_every_mask() {
        // Route admission is answered per slot from the source and motion
        // masks. The saved walk in `patch` and the live walk in the planner use
        // these two functions, so they must agree for every mask, including the
        // hostile all-clear source mask that sanitizes to carrier only.
        let masks = [
            SymmetrySourceMask::CARRIER_ONLY,
            SymmetrySourceMask {
                donor0: true,
                ..SymmetrySourceMask::CARRIER_ONLY
            },
            SymmetrySourceMask {
                donor1: true,
                ..SymmetrySourceMask::CARRIER_ONLY
            },
            SymmetrySourceMask {
                carrier: true,
                donor0: true,
                donor1: true,
                clean_history: true,
            },
            SymmetrySourceMask {
                carrier: false,
                donor0: false,
                donor1: false,
                clean_history: false,
            },
        ];
        let motion_masks = [
            SymmetryMotionMask {
                slot0: false,
                slot1: false,
            },
            SymmetryMotionMask {
                slot0: true,
                slot1: false,
            },
            SymmetryMotionMask {
                slot0: false,
                slot1: true,
            },
            SymmetryMotionMask {
                slot0: true,
                slot1: true,
            },
        ];
        for source_mask in masks {
            for motion_mask in motion_masks {
                let saved = SymmetryParams {
                    source_mask,
                    motion_mask,
                    motion: [
                        SavedMotionDonor::Selected {
                            saved_position: position(2),
                        },
                        SavedMotionDonor::Missing {
                            saved_position: position(5),
                        },
                    ],
                    ..armed_params()
                };
                let mut lookup = |value: SavedLayerPosition| (value.get() == 2).then(|| live(21));
                let runtime = RuntimeSymmetryParams::resolve_routes(saved, &mut lookup, &|_| false);

                let saved_image = saved.admitted_donor_taps().map(|tap| tap.is_some());
                let runtime_image = runtime.admitted_donor_taps().map(|tap| tap.is_some());
                assert_eq!(saved_image, runtime_image);
                let expected_image = [
                    source_mask.sanitized().donor0,
                    source_mask.sanitized().donor1,
                ];
                assert_eq!(saved_image, expected_image);

                let saved_motion = saved.admitted_motion_donors().map(|slot| slot.is_some());
                let runtime_motion = runtime.admitted_motion_donors().map(|slot| slot.is_some());
                assert_eq!(saved_motion, runtime_motion);
                assert_eq!(saved_motion, [motion_mask.slot0, motion_mask.slot1]);

                // A cleared slot never shifts the surviving route down.
                if let Some(tap) = runtime.admitted_donor_taps()[1] {
                    assert_eq!(tap, runtime.donors[1]);
                }
                if let Some(donor) = runtime.admitted_motion_donors()[1] {
                    assert_eq!(donor, runtime.motion[1]);
                    assert!(matches!(donor, MotionDonor::Missing { .. }));
                }
            }
        }

        // An exact-default node is carrier only, so neither slot is admitted
        // whatever route it happens to carry.
        let bypass = SymmetryParams::default();
        assert!(bypass.is_exact_bypass());
        assert_eq!(bypass.admitted_donor_taps(), [None, None]);
        assert_eq!(bypass.admitted_motion_donors(), [None, None]);
    }
}
