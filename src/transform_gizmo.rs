//! Preview-only direct manipulation of the canonical `SpatialTransform`.
//!
//! This module introduces **no** geometry of its own. Every drag resolves to
//! absolute values in the exact `param` vocabulary the browser's
//! `set_layer_transform` / `set_master_transform` / `set_group_transform`
//! actions already carry, and the host feeds them through the one authoring
//! function those actions use.
//! A gizmo that could author something the numeric editor cannot author would
//! be a second authored geometry wearing a handle, which the spatial contract
//! forbids; keeping the vocabulary closed and shared makes that impossible by
//! construction rather than by review.
//!
//! Three laws are worth stating before the code, because each is easy to
//! violate accidentally:
//!
//! 1. **Every drag reads the transform captured at `Begin`.** Hit testing,
//!    handle placement, and every delta are computed from one immutable
//!    snapshot of the authored transform and its derived uniforms. Deriving
//!    them from the *live* transform would let a crop edge chase its own
//!    motion and a scale handle accelerate under the pointer.
//! 2. **Reading never authors.** `SpatialTransform::gpu_uniforms` is pure, and
//!    nothing here mutates a transform outside `GizmoDrag::update` and
//!    `nudge_edits`. Opening, hovering, and hit testing therefore cannot move
//!    a patch off the exact legacy identity, so `spatial_modes.w` stays `0`
//!    for a patch nobody has moved.
//! 3. **A transform that cannot be inverted has no handles.** The render path
//!    fails a singular or non-finite transform to transparent; hit testing
//!    fails it closed, with no identity fallback. Handing back a grabbable
//!    handle over a transform that renders nothing would be a control that
//!    lies about what it will do.
//!
//! The module owns no `wgpu`, clock, or filesystem dependency. Only
//! `paint_transform_gizmo` touches `egui`, so the whole coordinate and drag
//! law is ordinary CPU-testable code.

use crate::image_routing::StableLayerId;
use crate::spatial::{
    apply_2x2, finite_clamp, output_aspect_basis, wrap_degrees, SpatialGpuUniforms,
    SpatialTransform, ANCHOR_MAX, ANCHOR_MIN, CROP_MAX, POSITION_MAX, POSITION_MIN, SCALE_MAX,
    SCALE_MIN,
};
use crate::stage_map::{native_controls_visible, StageSurface};
use crate::visual_rack::GroupId;

/// Divisor applied to every handle's delta while Alt is held.
///
/// Alt is deliberately one uniform law across every handle rather than a
/// different meaning per handle: an operator holding Alt is asking for a finer
/// version of the gesture they are already making, not for a different gesture.
pub const FINE_DRAG_FACTOR: f32 = 0.25;

/// Rotation snap, in degrees, while Shift is held.
pub const ROTATE_SNAP_DEGREES: f32 = 15.0;

/// Keyboard nudge step in composition-output UV.
pub const NUDGE_STEP: f32 = 1.0 / 128.0;

/// Shift nudge step. Coarse, for crossing the frame quickly.
pub const NUDGE_COARSE_STEP: f32 = 1.0 / 16.0;

/// Alt nudge step. Fine, for the last pixel.
pub const NUDGE_FINE_STEP: f32 = 1.0 / 1024.0;

/// Pick radius for a point handle, in composition-output UV.
pub const HANDLE_PICK_RADIUS_UV: f32 = 0.02;

/// Stroke width for every gizmo outline, in egui points.
///
/// Named rather than written inline at the two call sites, and the reason is a
/// compiler law rather than taste. `egui::Stroke::new` takes `impl Into<f32>`,
/// so a bare `1.5` there is inferred through an `f32: From<f64>` fallback that
/// rustc 1.97 reports as `float_literal_f32_fallback` — a warning under check
/// and test, and an error under `-D warnings`. A typed constant fixes the
/// inference at its definition, states the width once, and cannot drift back
/// into an inline literal at a third call site.
const GIZMO_STROKE_WIDTH: f32 = 1.5;

/// Local-space offset of the rotation handle above the footprint's top edge.
pub const ROTATE_HANDLE_LOCAL_OFFSET: f32 = 0.12;

/// Local-space offset of the move handle below the footprint's bottom edge.
///
/// Translation is a point handle rather than the whole footprint body, and that
/// is a boundary rather than a style choice. The preview surface is *already*
/// the gesture-etch canvas: a body-sized target would claim every drag over the
/// image and silently take the etch surface away, because an untransformed
/// source covers the entire composition. The gizmo therefore owns only its own
/// handles, and every other pixel of the preview still belongs to the etch
/// stroke it belonged to before.
///
/// It sits opposite the rotation handle so the two cannot overlap, and clear of
/// the anchor, which by default sits exactly at the footprint centre.
pub const MOVE_HANDLE_LOCAL_OFFSET: f32 = 0.12;

/// Smallest lever arm a scale or rotate drag may be driven from.
///
/// A grab exactly on the pivot carries no direction and no magnitude. Refusing
/// the axis is the honest answer; dividing by it would author an infinity that
/// sanitization would then land on a clamped extreme.
const MIN_PIVOT_DISTANCE: f32 = 1.0e-4;

/// Absolute cap on the edits one drag update or nudge may emit.
///
/// Two scale axes, two position axes, or a symmetric crop pair are the widest
/// cases; the array is sized to the maximum so an update allocates nothing.
pub const MAX_GIZMO_EDITS: usize = 4;

/// The permit, sealed in its own submodule.
///
/// `stage_health::EditorPreviewPermit` proved the shape, but its barrier is
/// module-scoped: that file's own `mod tests` could legally write
/// `EditorPreviewPermit(())`. Nesting the type one level deeper makes this
/// file's `mod tests` a *sibling* of the private field rather than a
/// descendant, so not even the gizmo's own tests can forge a token. The only
/// way to hold one is to have asked [`preview_gizmo_permit`] and been told yes.
mod permit {
    use super::{native_controls_visible, StageSurface};

    /// Opaque proof that the caller is painting the editor preview surface.
    ///
    /// A `bool` checked at each paint is one forgotten `if` away from a gizmo
    /// baked into a recording. A value that only one function can mint cannot
    /// be produced by an audience-facing consumer at all — and every such
    /// consumer would also have to acquire an `egui::Painter`, which none of
    /// them owns.
    pub struct PreviewGizmoPermit(());

    /// Mint the gizmo's paint/hit-test permit.
    ///
    /// Two independent conditions must both hold, and both are read rather
    /// than re-derived. The surface must be the editor preview. And the native
    /// controls must be visible at all, which answers from
    /// [`native_controls_visible`] — the same predicate the RECOVERY strip, the
    /// patch editor, the health HUD, and the gesture surface answer from — so a
    /// single-monitor Output that reuses the main swapchain removes the gizmo
    /// by exactly the mechanism that already removes every other native
    /// control.
    ///
    /// Folding both conditions into the token matters: the surface argument at
    /// a preview call site is naturally a constant, so a permit that checked
    /// only the surface would mint happily on a single-monitor audience output
    /// and leave the whole leakage question to an outer `if`.
    pub fn preview_gizmo_permit(
        surface: &StageSurface,
        output_on_main: bool,
    ) -> Option<PreviewGizmoPermit> {
        (native_controls_visible(output_on_main) && matches!(surface, StageSurface::EditorPreview))
            .then_some(PreviewGizmoPermit(()))
    }
}

pub use permit::{preview_gizmo_permit, PreviewGizmoPermit};

/// The scopes a preview gizmo may address.
///
/// These are exactly the scopes the browser numeric editor can author, through
/// `set_master_transform`, `set_layer_transform`, and `set_group_transform`.
/// A group is identified only by its stable [`GroupId`]: it never carries a
/// member-layer identity, so a host cannot silently turn a missing group into
/// a drag on one of its former members. Scope selection and stale-ID refusal
/// remain host responsibilities; the portable geometry layer only retains the
/// exact identity it was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoScope {
    Master,
    Layer(StableLayerId),
    Group(GroupId),
}

/// Which corner of the footprint a scale drag is driven from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleCorner {
    TopLeft,
    TopRight,
    BottomRight,
    BottomLeft,
}

impl ScaleCorner {
    pub const ALL: [Self; 4] = [
        Self::TopLeft,
        Self::TopRight,
        Self::BottomRight,
        Self::BottomLeft,
    ];

    /// The corner's position in cropped-source local UV.
    pub const fn local(self) -> [f32; 2] {
        match self {
            Self::TopLeft => [0.0, 0.0],
            Self::TopRight => [1.0, 0.0],
            Self::BottomRight => [1.0, 1.0],
            Self::BottomLeft => [0.0, 1.0],
        }
    }
}

/// Which edge of the footprint a crop drag is driven from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropEdge {
    Left,
    Top,
    Right,
    Bottom,
}

impl CropEdge {
    pub const ALL: [Self; 4] = [Self::Left, Self::Top, Self::Right, Self::Bottom];

    /// The edge handle's position in cropped-source local UV.
    pub const fn local(self) -> [f32; 2] {
        match self {
            Self::Left => [0.0, 0.5],
            Self::Top => [0.5, 0.0],
            Self::Right => [1.0, 0.5],
            Self::Bottom => [0.5, 1.0],
        }
    }

    /// Index into the authored `crop` array this edge owns.
    const fn crop_index(self) -> usize {
        match self {
            Self::Left => 0,
            Self::Top => 1,
            Self::Right => 2,
            Self::Bottom => 3,
        }
    }

    /// The edge directly across the footprint, used by the symmetric law.
    const fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Top => Self::Bottom,
            Self::Right => Self::Left,
            Self::Bottom => Self::Top,
        }
    }

    /// Whether this edge trims from the far side of its axis.
    const fn trims_from_far_side(self) -> bool {
        matches!(self, Self::Right | Self::Bottom)
    }

    /// Which local axis this edge moves along.
    const fn axis(self) -> usize {
        match self {
            Self::Left | Self::Right => 0,
            Self::Top | Self::Bottom => 1,
        }
    }
}

/// A grabbable handle. The vocabulary is closed, so an unknown handle is a
/// compile error rather than a positional fallback onto another control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoHandle {
    Translate,
    Scale(ScaleCorner),
    Rotate,
    Anchor,
    Crop(CropEdge),
}

/// Modifier state for one drag or nudge.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GizmoModifiers {
    pub shift: bool,
    pub alt: bool,
}

impl GizmoModifiers {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the neutral modifier set is the baseline every drag-law golden compares against"
        )
    )]
    pub const NONE: Self = Self {
        shift: false,
        alt: false,
    };

    /// Alt is the fine-drag law for every handle.
    fn fine_scale(self) -> f32 {
        if self.alt {
            FINE_DRAG_FACTOR
        } else {
            1.0
        }
    }
}

/// Pointer phase, matching the three facts egui reports per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoPhase {
    Begin,
    Move,
    End,
}

/// One pointer edge observed while building the egui frame.
///
/// It is collected during construction and dispatched afterwards, through the
/// same deterministic local-last boundary the RECOVERY strip and the native
/// gesture surface already use: browser ingress is drained first, then native
/// edges land in the frame they were drawn in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GizmoPointerEvent {
    pub phase: GizmoPhase,
    pub output_uv: [f32; 2],
    pub modifiers: GizmoModifiers,
}

/// Direction of a keyboard nudge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoNudge {
    Left,
    Right,
    Up,
    Down,
}

impl GizmoNudge {
    /// Unit step in composition-output UV. Y grows downward, matching both the
    /// composition's UV convention and the screen the operator is looking at.
    const fn direction(self) -> [f32; 2] {
        match self {
            Self::Left => [-1.0, 0.0],
            Self::Right => [1.0, 0.0],
            Self::Up => [0.0, -1.0],
            Self::Down => [0.0, 1.0],
        }
    }
}

/// The authored fields a gizmo may write.
///
/// Each maps to the exact `param` string the browser transform actions carry,
/// so a gizmo edit and a numeric edit reach the identical authoring function
/// with identical arguments. Fit, edge, and sampling are deliberately absent:
/// they are discrete choices with no continuous handle, and the numeric editor
/// already owns them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoParam {
    PositionX,
    PositionY,
    ScaleX,
    ScaleY,
    AnchorX,
    AnchorY,
    RotationDeg,
    CropLeft,
    CropTop,
    CropRight,
    CropBottom,
}

impl GizmoParam {
    /// The wire/authoring name. This is the shared vocabulary, not a second one.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PositionX => "position_x",
            Self::PositionY => "position_y",
            Self::ScaleX => "scale_x",
            Self::ScaleY => "scale_y",
            Self::AnchorX => "anchor_x",
            Self::AnchorY => "anchor_y",
            Self::RotationDeg => "rotation_deg",
            Self::CropLeft => "crop_left",
            Self::CropTop => "crop_top",
            Self::CropRight => "crop_right",
            Self::CropBottom => "crop_bottom",
        }
    }

    /// The inclusive authored range, taken from the spatial contract's own
    /// constants rather than restated. Rotation wraps instead of clamping and
    /// therefore reports the full circle.
    const fn range(self) -> (f32, f32) {
        match self {
            Self::PositionX | Self::PositionY => (POSITION_MIN, POSITION_MAX),
            Self::ScaleX | Self::ScaleY => (SCALE_MIN, SCALE_MAX),
            Self::AnchorX | Self::AnchorY => (ANCHOR_MIN, ANCHOR_MAX),
            Self::RotationDeg => (-180.0, 180.0),
            Self::CropLeft | Self::CropTop | Self::CropRight | Self::CropBottom => (0.0, CROP_MAX),
        }
    }

    /// Bring a computed value into the authored range.
    ///
    /// A non-finite computation lands on the field's neutral value, never on a
    /// clamped extreme: an infinity is a broken reading, and turning it into
    /// the strongest possible authored value invents intent out of a fault.
    fn sanitize(self, value: f32) -> f32 {
        let (min, max) = self.range();
        match self {
            Self::RotationDeg => wrap_degrees(value),
            Self::ScaleX | Self::ScaleY => finite_clamp(value, 1.0, min, max),
            Self::AnchorX | Self::AnchorY => finite_clamp(value, 0.5, min, max),
            _ => finite_clamp(value, 0.0, min, max),
        }
    }
}

/// One absolute authored value a gizmo asks for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GizmoEdit {
    pub param: GizmoParam,
    pub value: f32,
}

/// A bounded, allocation-free set of edits.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GizmoEdits {
    edits: [Option<GizmoEdit>; MAX_GIZMO_EDITS],
    len: usize,
}

impl GizmoEdits {
    pub const EMPTY: Self = Self {
        edits: [None; MAX_GIZMO_EDITS],
        len: 0,
    };

    /// Record one edit, sanitizing it into the authored range on the way in.
    ///
    /// Overflow past the fixed capacity is impossible for the laws below — the
    /// widest emits two values — but it is dropped rather than panicking, so a
    /// future handle cannot turn a miscount into a crash on the render thread.
    fn push(&mut self, param: GizmoParam, value: f32) {
        if self.len >= MAX_GIZMO_EDITS {
            return;
        }
        self.edits[self.len] = Some(GizmoEdit {
            param,
            value: param.sanitize(value),
        });
        self.len += 1;
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the bounded-capacity law is asserted by the overflow golden"
        )
    )]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = GizmoEdit> + '_ {
        self.edits.iter().take(self.len).filter_map(|edit| *edit)
    }

    /// The value this set carries for one param, if any. Test and host
    /// convenience; the host applies the whole set in order.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "every drag-law golden reads its authored value back through this"
        )
    )]
    pub fn value_of(&self, param: GizmoParam) -> Option<f32> {
        self.iter()
            .find(|edit| edit.param == param)
            .map(|edit| edit.value)
    }
}

/// A rectangle in the host's pointer space.
///
/// Kept as plain numbers so the coordinate law needs no `egui` and can be
/// exercised at any aspect ratio and any DPI in an ordinary CPU test. egui
/// reports both the widget rect and the pointer in the same logical points, so
/// the ratio below is DPI-independent by construction; a test still pins that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreviewPaneRect {
    pub min: [f32; 2],
    pub size: [f32; 2],
}

impl PreviewPaneRect {
    pub const fn new(min: [f32; 2], size: [f32; 2]) -> Self {
        Self { min, size }
    }

    /// Map a pointer position onto composition-output UV.
    ///
    /// Deliberately **unclamped**: a scale or rotate drag that leaves the
    /// letterboxed image must keep tracking the pointer. Clamping here would
    /// freeze the gesture at the pane border and silently cap how far an
    /// operator can push a handle. A degenerate or non-finite pane has no
    /// coordinate space to map into and refuses instead of dividing by zero.
    pub fn output_uv(self, pointer: [f32; 2]) -> Option<[f32; 2]> {
        if !self.size[0].is_finite()
            || !self.size[1].is_finite()
            || self.size[0] <= 0.0
            || self.size[1] <= 0.0
            || !self.min[0].is_finite()
            || !self.min[1].is_finite()
            || !pointer[0].is_finite()
            || !pointer[1].is_finite()
        {
            return None;
        }
        let uv = [
            (pointer[0] - self.min[0]) / self.size[0],
            (pointer[1] - self.min[1]) / self.size[1],
        ];
        (uv[0].is_finite() && uv[1].is_finite()).then_some(uv)
    }

    /// Map composition-output UV back into pointer space, for painting.
    pub fn pane_position(self, output_uv: [f32; 2]) -> Option<[f32; 2]> {
        if !self.size[0].is_finite() || !self.size[1].is_finite() {
            return None;
        }
        let position = [
            self.min[0] + output_uv[0] * self.size[0],
            self.min[1] + output_uv[1] * self.size[1],
        ];
        (position[0].is_finite() && position[1].is_finite()).then_some(position)
    }
}

/// The immutable frame one gizmo interaction is resolved against.
///
/// It bundles the authored transform with the uniforms derived from it by the
/// production `SpatialTransform::gpu_uniforms`, the same call the render path
/// makes. There is no second inverse here: `map_output_to_local` is the
/// canonical one, promoted out of `#[cfg(test)]`, and the forward map is its
/// algebraic inverse rather than an independently authored matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GizmoFrame {
    transform: SpatialTransform,
    uniforms: SpatialGpuUniforms,
    output_aspect: f32,
    /// Forward 2x2, mapping a local offset from the anchor to an output offset.
    forward: [[f32; 2]; 2],
    /// The anchor's position in composition-output UV.
    anchor_output: [f32; 2],
    /// The anchor in cropped-source local UV.
    anchor_local: [f32; 2],
}

impl GizmoFrame {
    /// Derive the interaction frame for one scope.
    ///
    /// `source_dimensions` and `output_dimensions` must be exactly what
    /// `EffectPassUniforms::for_target` passes for the same scope — a layer's
    /// actual source size, or the output size twice for master and group.
    /// Group materialization transforms the already-composited, output-sized
    /// group surface; using any member's source dimensions would therefore
    /// place handles on geometry nobody renders. Anything else would place
    /// handles on a transform nobody renders.
    ///
    /// Returns `None` for the same conditions the shader treats as
    /// unrenderable: a collapsed or singular transform, a zero dimension, or a
    /// non-finite derived matrix. There is no identity fallback.
    pub fn new(
        transform: SpatialTransform,
        source_dimensions: (u32, u32),
        output_dimensions: (u32, u32),
    ) -> Option<Self> {
        if output_dimensions.0 == 0 || output_dimensions.1 == 0 {
            return None;
        }
        let transform = transform.sanitized();
        let uniforms = transform.gpu_uniforms(
            source_dimensions.0,
            source_dimensions.1,
            output_dimensions.0,
            output_dimensions.1,
        );
        if !uniforms.is_spatially_valid() {
            return None;
        }
        let inverse = [
            [uniforms.inverse_row_0[0], uniforms.inverse_row_0[1]],
            [uniforms.inverse_row_1[0], uniforms.inverse_row_1[1]],
        ];
        let determinant = inverse[0][0] * inverse[1][1] - inverse[0][1] * inverse[1][0];
        if !determinant.is_finite() || determinant == 0.0 {
            return None;
        }
        let forward = [
            [inverse[1][1] / determinant, -inverse[0][1] / determinant],
            [-inverse[1][0] / determinant, inverse[0][0] / determinant],
        ];
        if forward.into_iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        // `local = I*(output - t) + a` for the authored anchor `a`, so the
        // anchor sits at output `t` exactly. Reading it back out of the packed
        // translation keeps one derivation rather than restating the target.
        let translation = [uniforms.inverse_row_0[2], uniforms.inverse_row_1[2]];
        let crop = uniforms.crop;
        if crop[2] <= 0.0 || crop[3] <= 0.0 || !crop[2].is_finite() || !crop[3].is_finite() {
            return None;
        }
        let anchor_local = [
            (transform.anchor[0] - crop[0]) / crop[2],
            (transform.anchor[1] - crop[1]) / crop[3],
        ];
        if !anchor_local[0].is_finite() || !anchor_local[1].is_finite() {
            return None;
        }
        // local = I*output + translation, and local == anchor_local at the
        // anchor, so anchor_output = F*(anchor_local - translation).
        let anchor_output = apply_2x2(
            forward,
            [
                anchor_local[0] - translation[0],
                anchor_local[1] - translation[1],
            ],
        );
        if !anchor_output[0].is_finite() || !anchor_output[1].is_finite() {
            return None;
        }
        let output_aspect = output_dimensions.0 as f32 / output_dimensions.1 as f32;
        if !output_aspect.is_finite() || output_aspect <= 0.0 {
            return None;
        }
        Some(Self {
            transform,
            uniforms,
            output_aspect,
            forward,
            anchor_output,
            anchor_local,
        })
    }

    pub const fn transform(&self) -> SpatialTransform {
        self.transform
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the goldens compare this against the canonical inverse directly"
        )
    )]
    pub const fn uniforms(&self) -> SpatialGpuUniforms {
        self.uniforms
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the pivot goldens drive rotation and anchor drags from this point"
        )
    )]
    pub const fn anchor_output(&self) -> [f32; 2] {
        self.anchor_output
    }

    /// Composition-output UV to cropped-source local UV. Fails closed.
    pub fn local_at(&self, output_uv: [f32; 2]) -> Option<[f32; 2]> {
        self.uniforms.hit_test_local(output_uv)
    }

    /// Cropped-source local UV to composition-output UV.
    pub fn output_at(&self, local_uv: [f32; 2]) -> Option<[f32; 2]> {
        let offset = apply_2x2(
            self.forward,
            [
                local_uv[0] - self.anchor_local[0],
                local_uv[1] - self.anchor_local[1],
            ],
        );
        let output = [
            self.anchor_output[0] + offset[0],
            self.anchor_output[1] + offset[1],
        ];
        (output[0].is_finite() && output[1].is_finite()).then_some(output)
    }

    /// The four corners of the source footprint, in composition-output UV.
    pub fn footprint(&self) -> Option<[[f32; 2]; 4]> {
        let mut corners = [[0.0_f32; 2]; 4];
        for (slot, corner) in corners.iter_mut().zip(ScaleCorner::ALL) {
            *slot = self.output_at(corner.local())?;
        }
        Some(corners)
    }

    /// Where one handle sits, in composition-output UV.
    pub fn handle_position(&self, handle: GizmoHandle) -> Option<[f32; 2]> {
        match handle {
            GizmoHandle::Translate => self.output_at([0.5, 1.0 + MOVE_HANDLE_LOCAL_OFFSET]),
            GizmoHandle::Scale(corner) => self.output_at(corner.local()),
            GizmoHandle::Rotate => self.output_at([0.5, -ROTATE_HANDLE_LOCAL_OFFSET]),
            GizmoHandle::Anchor => Some(self.anchor_output),
            GizmoHandle::Crop(edge) => self.output_at(edge.local()),
        }
    }

    /// The handles offered, in the order hit testing considers them.
    ///
    /// Order is priority: the anchor and the rotation ring sit on or near the
    /// footprint's own points, so they must be able to win against the corner
    /// and edge handles beneath them, and the body is last because it is the
    /// largest target.
    fn ordered_handles() -> impl Iterator<Item = GizmoHandle> {
        std::iter::once(GizmoHandle::Anchor)
            .chain(std::iter::once(GizmoHandle::Rotate))
            .chain(std::iter::once(GizmoHandle::Translate))
            .chain(ScaleCorner::ALL.into_iter().map(GizmoHandle::Scale))
            .chain(CropEdge::ALL.into_iter().map(GizmoHandle::Crop))
    }

    /// Pick the handle under a pointer, or `None`.
    ///
    /// A transform that could not be inverted never reaches here: `new`
    /// already refused it, so there is no path by which an unrenderable
    /// transform offers a grabbable control.
    pub fn hit_test(&self, output_uv: [f32; 2]) -> Option<GizmoHandle> {
        if !output_uv[0].is_finite() || !output_uv[1].is_finite() {
            return None;
        }
        let radius_sq = HANDLE_PICK_RADIUS_UV * HANDLE_PICK_RADIUS_UV;
        let mut best: Option<(GizmoHandle, f32)> = None;
        for handle in Self::ordered_handles() {
            let Some(position) = self.handle_position(handle) else {
                continue;
            };
            // Distance is measured in physical space so a point handle is a
            // circle on screen rather than an ellipse on a wide output.
            let (to_physical, _) = output_aspect_basis(self.output_aspect);
            let delta = apply_2x2(
                to_physical,
                [output_uv[0] - position[0], output_uv[1] - position[1]],
            );
            let distance_sq = delta[0] * delta[0] + delta[1] * delta[1];
            if !distance_sq.is_finite() || distance_sq > radius_sq {
                continue;
            }
            if best.is_none_or(|(_, best_distance)| distance_sq < best_distance) {
                best = Some((handle, distance_sq));
            }
        }
        // Deliberately no body fallback: a pointer that is not on a handle
        // belongs to the gesture-etch surface underneath, exactly as it did
        // before the gizmo existed.
        best.map(|(handle, _)| handle)
    }
}

/// What the host should do about a cancel request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GizmoCancel {
    /// Escape before any value moved: restore this transform verbatim and
    /// cancel the open history gesture.
    Restore(SpatialTransform),
    /// Escape after a value already committed. A commit is not a cancel, so the
    /// gesture is closed normally and the host performs an ordinary undo.
    UndoCommitted,
}

/// One open pointer drag.
///
/// Everything it needs is captured at `Begin`: the scope, the handle, the
/// authored transform, the derived frame, and the grab point. Nothing is read
/// from live state afterwards, so a reorder, a removal, or a patch load during
/// the drag cannot retarget it — the host aborts it instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GizmoDrag {
    scope: GizmoScope,
    handle: GizmoHandle,
    frame: GizmoFrame,
    origin_uv: [f32; 2],
    origin_local: [f32; 2],
    committed: bool,
}

impl GizmoDrag {
    /// Begin a drag at a pointer position, if it lands on a handle.
    pub fn begin(
        scope: GizmoScope,
        frame: GizmoFrame,
        output_uv: [f32; 2],
    ) -> Option<(Self, GizmoHandle)> {
        let handle = frame.hit_test(output_uv)?;
        let origin_local = frame.local_at(output_uv)?;
        Some((
            Self {
                scope,
                handle,
                frame,
                origin_uv: output_uv,
                origin_local,
                committed: false,
            },
            handle,
        ))
    }

    pub const fn scope(&self) -> GizmoScope {
        self.scope
    }

    pub const fn handle(&self) -> GizmoHandle {
        self.handle
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the no-retarget golden asserts the captured snapshot is immutable"
        )
    )]
    pub const fn captured(&self) -> SpatialTransform {
        self.frame.transform
    }

    /// Whether any value has been committed by this drag yet.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the Escape law golden distinguishes the two cancel meanings by this"
        )
    )]
    pub const fn has_committed(&self) -> bool {
        self.committed
    }

    /// Note that the host applied a non-empty edit set.
    ///
    /// The host calls this only when an edit actually changed authored state,
    /// which is what makes Escape's two behaviours distinguishable: a drag that
    /// moved nothing is still cancellable.
    pub fn mark_committed(&mut self) {
        self.committed = true;
    }

    /// What Escape means right now.
    pub fn cancel(&self) -> GizmoCancel {
        if self.committed {
            GizmoCancel::UndoCommitted
        } else {
            GizmoCancel::Restore(self.frame.transform)
        }
    }

    /// Resolve a pointer position into absolute authored values.
    ///
    /// Every branch reads `self.frame`, the snapshot taken at `Begin`. That is
    /// what keeps a crop edge from chasing its own motion and a scale handle
    /// from compounding: the drag always answers "where should this be, given
    /// where it started", never "how much has it moved since last frame".
    pub fn update(&self, output_uv: [f32; 2], modifiers: GizmoModifiers) -> GizmoEdits {
        let mut edits = GizmoEdits::EMPTY;
        if !output_uv[0].is_finite() || !output_uv[1].is_finite() {
            return edits;
        }
        match self.handle {
            GizmoHandle::Translate => self.translate(output_uv, modifiers, &mut edits),
            GizmoHandle::Scale(corner) => self.scale(corner, output_uv, modifiers, &mut edits),
            GizmoHandle::Rotate => self.rotate(output_uv, modifiers, &mut edits),
            GizmoHandle::Anchor => self.anchor(output_uv, modifiers, &mut edits),
            GizmoHandle::Crop(edge) => self.crop(edge, output_uv, modifiers, &mut edits),
        }
        edits
    }

    /// Delta from the grab point, with the Alt fine law already applied.
    fn delta(&self, output_uv: [f32; 2], modifiers: GizmoModifiers) -> [f32; 2] {
        let fine = modifiers.fine_scale();
        [
            (output_uv[0] - self.origin_uv[0]) * fine,
            (output_uv[1] - self.origin_uv[1]) * fine,
        ]
    }

    /// The effective pointer once the Alt fine law has scaled the gesture.
    ///
    /// Handles that read an absolute position rather than a displacement — the
    /// anchor and the crop edges — must see the same damped motion, or Alt
    /// would silently do nothing for them.
    fn effective_uv(&self, output_uv: [f32; 2], modifiers: GizmoModifiers) -> [f32; 2] {
        let delta = self.delta(output_uv, modifiers);
        [self.origin_uv[0] + delta[0], self.origin_uv[1] + delta[1]]
    }

    fn translate(&self, output_uv: [f32; 2], modifiers: GizmoModifiers, edits: &mut GizmoEdits) {
        let mut delta = self.delta(output_uv, modifiers);
        if modifiers.shift {
            // Axis lock keeps the dominant component and discards the other,
            // measured in physical space so the choice matches what the
            // operator sees on a non-square output.
            let (to_physical, _) = output_aspect_basis(self.frame.output_aspect);
            let physical = apply_2x2(to_physical, delta);
            if physical[0].abs() >= physical[1].abs() {
                delta[1] = 0.0;
            } else {
                delta[0] = 0.0;
            }
        }
        let base = self.frame.transform.position;
        edits.push(GizmoParam::PositionX, base[0] + delta[0]);
        edits.push(GizmoParam::PositionY, base[1] + delta[1]);
    }

    fn scale(
        &self,
        corner: ScaleCorner,
        output_uv: [f32; 2],
        modifiers: GizmoModifiers,
        edits: &mut GizmoEdits,
    ) {
        let Some(current_local) = self.frame.local_at(self.effective_uv(output_uv, modifiers))
        else {
            return;
        };
        let anchor = self.frame.anchor_local;
        // The lever arm is measured from the grabbed corner, not from wherever
        // the pointer happened to land, so a slightly off-centre grab does not
        // jump the footprint on the first move.
        let grabbed = corner.local();
        let base = self.frame.transform.scale;
        let mut ratio = [1.0_f32; 2];
        let mut drivable = [false; 2];
        for axis in 0..2 {
            let arm = grabbed[axis] - anchor[axis];
            if arm.abs() <= MIN_PIVOT_DISTANCE {
                continue;
            }
            let reached = current_local[axis] - anchor[axis];
            let candidate = reached / arm;
            if !candidate.is_finite() {
                continue;
            }
            ratio[axis] = candidate;
            drivable[axis] = true;
        }
        if modifiers.shift {
            // Uniform scale takes the better-conditioned axis: the one with the
            // longer lever arm carries less relative noise.
            let pick = if (grabbed[0] - anchor[0]).abs() >= (grabbed[1] - anchor[1]).abs() {
                0
            } else {
                1
            };
            if drivable[pick] {
                ratio = [ratio[pick]; 2];
                drivable = [true; 2];
            }
        }
        if drivable[0] {
            edits.push(GizmoParam::ScaleX, base[0] * ratio[0]);
        }
        if drivable[1] {
            edits.push(GizmoParam::ScaleY, base[1] * ratio[1]);
        }
    }

    fn rotate(&self, output_uv: [f32; 2], modifiers: GizmoModifiers, edits: &mut GizmoEdits) {
        let pivot = self.frame.anchor_output;
        // Rotation is authored in physical space and conjugated back by
        // `gpu_uniforms`, so the drag must measure its angle there too, or a
        // quarter turn on a 16:9 output would not author 90 degrees.
        let (to_physical, _) = output_aspect_basis(self.frame.output_aspect);
        let from = apply_2x2(
            to_physical,
            [self.origin_uv[0] - pivot[0], self.origin_uv[1] - pivot[1]],
        );
        let to = apply_2x2(
            to_physical,
            [output_uv[0] - pivot[0], output_uv[1] - pivot[1]],
        );
        if from[0].hypot(from[1]) <= MIN_PIVOT_DISTANCE || to[0].hypot(to[1]) <= MIN_PIVOT_DISTANCE
        {
            return;
        }
        let swept = to[1].atan2(to[0]) - from[1].atan2(from[0]);
        if !swept.is_finite() {
            return;
        }
        let swept_degrees = swept.to_degrees() * modifiers.fine_scale();
        let mut authored = self.frame.transform.rotation_deg + swept_degrees;
        if modifiers.shift {
            // Snap the resulting angle, not the delta: an operator holding
            // Shift wants the transform to sit on a multiple, regardless of
            // where the rotation started.
            authored = (authored / ROTATE_SNAP_DEGREES).round() * ROTATE_SNAP_DEGREES;
        }
        edits.push(GizmoParam::RotationDeg, authored);
    }

    fn anchor(&self, output_uv: [f32; 2], modifiers: GizmoModifiers, edits: &mut GizmoEdits) {
        let Some(mut local) = self.frame.local_at(self.effective_uv(output_uv, modifiers)) else {
            return;
        };
        if modifiers.shift {
            // Axis lock, measured against where the grab started in the same
            // local frame the value is authored in.
            let delta = [
                local[0] - self.origin_local[0],
                local[1] - self.origin_local[1],
            ];
            if delta[0].abs() >= delta[1].abs() {
                local[1] = self.origin_local[1];
            } else {
                local[0] = self.origin_local[0];
            }
        }
        // `anchor` is authored in the *uncropped* source rectangle, while the
        // inverse hands back cropped-normalized local UV. Undoing the crop
        // normalization here is what keeps the handle under the pointer when a
        // crop is active.
        let crop = self.frame.uniforms.crop;
        edits.push(GizmoParam::AnchorX, crop[0] + local[0] * crop[2]);
        edits.push(GizmoParam::AnchorY, crop[1] + local[1] * crop[3]);
    }

    fn crop(
        &self,
        edge: CropEdge,
        output_uv: [f32; 2],
        modifiers: GizmoModifiers,
        edits: &mut GizmoEdits,
    ) {
        let Some(local) = self.frame.local_at(self.effective_uv(output_uv, modifiers)) else {
            return;
        };
        let crop = self.frame.uniforms.crop;
        let axis = edge.axis();
        // Where the pointer sits in the *uncropped* source rectangle.
        let source = crop[axis] + local[axis] * crop[axis + 2];
        if !source.is_finite() {
            return;
        }
        let trim = if edge.trims_from_far_side() {
            1.0 - source
        } else {
            source
        };
        edits.push(crop_param(edge), trim);
        if modifiers.shift {
            // Symmetric crop: the opposite edge takes the same trim, so the
            // retained rectangle stays centred on the source.
            edits.push(crop_param(edge.opposite()), trim);
        }
    }
}

const fn crop_param(edge: CropEdge) -> GizmoParam {
    match edge.crop_index() {
        0 => GizmoParam::CropLeft,
        1 => GizmoParam::CropTop,
        2 => GizmoParam::CropRight,
        _ => GizmoParam::CropBottom,
    }
}

/// Absolute values for one keyboard nudge of a scope's position.
///
/// The step law is fixed and documented: an unmodified arrow moves
/// [`NUDGE_STEP`], Shift moves the coarse step, and Alt moves the fine step.
/// Holding both takes the fine step, because Alt is the fine-detail law
/// everywhere in this module and the more precise request wins.
pub fn nudge_edits(
    transform: SpatialTransform,
    nudge: GizmoNudge,
    modifiers: GizmoModifiers,
) -> GizmoEdits {
    let step = if modifiers.alt {
        NUDGE_FINE_STEP
    } else if modifiers.shift {
        NUDGE_COARSE_STEP
    } else {
        NUDGE_STEP
    };
    let direction = nudge.direction();
    let base = transform.sanitized().position;
    let mut edits = GizmoEdits::EMPTY;
    edits.push(GizmoParam::PositionX, base[0] + direction[0] * step);
    edits.push(GizmoParam::PositionY, base[1] + direction[1] * step);
    edits
}

/// Paint the gizmo over the editor preview.
///
/// The signature is the boundary: it cannot be called without a
/// [`PreviewGizmoPermit`], and that token can only be minted for
/// [`StageSurface::EditorPreview`] while the native controls are visible. No
/// audience, composite, Spout, record, export, or physical StageMap consumer
/// can obtain one, and none of them owns an `egui::Ui` to pass either. The
/// gizmo is drawn into the editor window's own egui layer and never into the
/// composition the renderer hands to those surfaces.
pub fn paint_transform_gizmo(
    painter: &egui::Painter,
    _permit: &PreviewGizmoPermit,
    pane: PreviewPaneRect,
    frame: &GizmoFrame,
    active: Option<GizmoHandle>,
) {
    let stroke_color = egui::Color32::from_rgb(255, 214, 102);
    let accent = egui::Color32::from_rgb(120, 220, 255);
    let Some(corners) = frame.footprint() else {
        return;
    };
    let mut pane_corners = [egui::Pos2::ZERO; 4];
    for (slot, corner) in pane_corners.iter_mut().zip(corners) {
        let Some(position) = pane.pane_position(corner) else {
            return;
        };
        *slot = egui::pos2(position[0], position[1]);
    }
    for index in 0..4 {
        painter.line_segment(
            [pane_corners[index], pane_corners[(index + 1) % 4]],
            egui::Stroke::new(GIZMO_STROKE_WIDTH, stroke_color),
        );
    }
    let point = |handle: GizmoHandle, color: egui::Color32, radius: f32| {
        let Some(uv) = frame.handle_position(handle) else {
            return;
        };
        let Some(position) = pane.pane_position(uv) else {
            return;
        };
        let filled = active == Some(handle);
        let centre = egui::pos2(position[0], position[1]);
        if filled {
            painter.circle_filled(centre, radius, color);
        } else {
            painter.circle_stroke(centre, radius, egui::Stroke::new(GIZMO_STROKE_WIDTH, color));
        }
    };
    for corner in ScaleCorner::ALL {
        point(GizmoHandle::Scale(corner), stroke_color, 4.0);
    }
    point(GizmoHandle::Translate, stroke_color, 5.0);
    for edge in CropEdge::ALL {
        point(GizmoHandle::Crop(edge), accent, 3.0);
    }
    point(GizmoHandle::Rotate, stroke_color, 5.0);
    point(GizmoHandle::Anchor, accent, 5.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{EdgeMode, FitMode};

    const SQUARE: (u32, u32) = (512, 512);
    const WIDE: (u32, u32) = (1920, 1080);
    const SOURCE_43: (u32, u32) = (640, 480);

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected} (tolerance {tolerance})"
        );
    }

    fn frame_for(
        transform: SpatialTransform,
        source: (u32, u32),
        output: (u32, u32),
    ) -> GizmoFrame {
        GizmoFrame::new(transform, source, output).expect("the fixture transform is renderable")
    }

    /// Where a translate drag must be grabbed. Translation is a point handle,
    /// not the footprint body, so the etch surface keeps every other pixel.
    fn move_handle(frame: &GizmoFrame) -> [f32; 2] {
        frame
            .handle_position(GizmoHandle::Translate)
            .expect("the move handle is placed")
    }

    // ---- coordinate law -------------------------------------------------

    #[test]
    fn pane_mapping_is_unclamped_and_dpi_independent() {
        // The same pane at two DPI scales: egui reports the rect and the
        // pointer in the same logical points, so the ratio must not move.
        let low = PreviewPaneRect::new([10.0, 20.0], [400.0, 300.0]);
        let high = PreviewPaneRect::new([20.0, 40.0], [800.0, 600.0]);
        let low_uv = low.output_uv([210.0, 170.0]).unwrap();
        let high_uv = high.output_uv([420.0, 340.0]).unwrap();
        close(low_uv[0], high_uv[0], 1.0e-6);
        close(low_uv[1], high_uv[1], 1.0e-6);
        close(low_uv[0], 0.5, 1.0e-6);
        close(low_uv[1], 0.5, 1.0e-6);

        // A pointer dragged past the letterboxed image keeps tracking rather
        // than freezing at the border.
        let outside = low.output_uv([-190.0, 620.0]).unwrap();
        assert!(outside[0] < 0.0, "left of the pane must stay negative");
        assert!(outside[1] > 1.0, "below the pane must stay above one");

        // Round trip.
        let back = low.pane_position(low_uv).unwrap();
        close(back[0], 210.0, 1.0e-3);
        close(back[1], 170.0, 1.0e-3);
    }

    #[test]
    fn degenerate_and_non_finite_panes_refuse() {
        assert!(PreviewPaneRect::new([0.0, 0.0], [0.0, 100.0])
            .output_uv([1.0, 1.0])
            .is_none());
        assert!(PreviewPaneRect::new([0.0, 0.0], [f32::NAN, 100.0])
            .output_uv([1.0, 1.0])
            .is_none());
        assert!(PreviewPaneRect::new([0.0, 0.0], [100.0, 100.0])
            .output_uv([f32::INFINITY, 1.0])
            .is_none());
    }

    /// One round-trip case: an authored transform and the exact dimension pair
    /// the render path would pass for it.
    type RoundTripCase = (SpatialTransform, (u32, u32), (u32, u32));

    #[test]
    fn local_output_round_trip_holds_across_aspects_and_geometry() {
        let cases: [RoundTripCase; 6] = [
            (SpatialTransform::default(), SQUARE, SQUARE),
            (SpatialTransform::default(), SOURCE_43, WIDE),
            (
                SpatialTransform {
                    fit: FitMode::Fit,
                    rotation_deg: 37.0,
                    ..SpatialTransform::default()
                },
                SOURCE_43,
                WIDE,
            ),
            (
                SpatialTransform {
                    position: [0.2, -0.15],
                    scale: [1.8, 0.6],
                    rotation_deg: -114.0,
                    skew_deg: 22.0,
                    skew_axis_deg: 41.0,
                    ..SpatialTransform::default()
                },
                SOURCE_43,
                WIDE,
            ),
            (
                SpatialTransform {
                    crop: [0.1, 0.2, 0.15, 0.05],
                    scale: [1.3, 1.3],
                    rotation_deg: 90.0,
                    ..SpatialTransform::default()
                },
                SOURCE_43,
                WIDE,
            ),
            (
                SpatialTransform {
                    fit: FitMode::Native,
                    anchor: [0.1, 0.9],
                    rotation_deg: 12.5,
                    ..SpatialTransform::default()
                },
                SOURCE_43,
                (3840, 2160),
            ),
        ];
        let probes = [
            [0.0, 0.0],
            [0.25, 0.75],
            [0.5, 0.5],
            [1.0, 1.0],
            [-0.3, 1.4],
        ];
        for (index, (transform, source, output)) in cases.into_iter().enumerate() {
            let frame = frame_for(transform, source, output);
            for probe in probes {
                let local = frame.local_at(probe).expect("renderable transform maps");
                let back = frame.output_at(local).expect("forward map is finite");
                close(back[0], probe[0], 2.0e-4);
                close(back[1], probe[1], 2.0e-4);

                // ...and the other direction, so neither map is merely the
                // left inverse of the other on a lucky subset.
                let forward = frame.output_at(probe).expect("forward map is finite");
                let round = frame.local_at(forward).expect("inverse map is finite");
                close(round[0], probe[0], 2.0e-4);
                close(round[1], probe[1], 2.0e-4);
                let _ = index;
            }
        }
    }

    #[test]
    fn the_forward_map_agrees_with_the_canonical_inverse() {
        // The forward map must be the algebraic inverse of the shader's own
        // matrix, not a second authored transform that happens to agree today.
        let transform = SpatialTransform {
            position: [0.13, -0.27],
            scale: [1.4, 0.75],
            anchor: [0.2, 0.8],
            rotation_deg: 55.0,
            skew_deg: -18.0,
            skew_axis_deg: 27.0,
            fit: FitMode::Fill,
            ..SpatialTransform::default()
        };
        let frame = frame_for(transform, SOURCE_43, WIDE);
        let uniforms = frame.uniforms();
        for probe in [[0.1, 0.2], [0.9, 0.4], [0.5, 0.95]] {
            let canonical = uniforms.map_output_to_local(probe);
            let ours = frame.local_at(probe).unwrap();
            close(ours[0], canonical[0], 1.0e-6);
            close(ours[1], canonical[1], 1.0e-6);
        }
    }

    #[test]
    fn the_anchor_sits_exactly_where_the_transform_puts_it() {
        let transform = SpatialTransform {
            position: [0.2, -0.1],
            anchor: [0.25, 0.75],
            rotation_deg: 90.0,
            scale: [1.6, 0.8],
            ..SpatialTransform::default()
        };
        let frame = frame_for(transform, SQUARE, SQUARE);
        let anchor_output = frame.anchor_output();
        let local = frame.local_at(anchor_output).unwrap();
        // The anchor's local coordinate is the authored anchor, crop-normalized.
        close(local[0], 0.25, 1.0e-4);
        close(local[1], 0.75, 1.0e-4);
    }

    // ---- fail-closed ----------------------------------------------------

    #[test]
    fn a_collapsed_transform_has_no_frame_and_therefore_no_handles() {
        let collapsed = SpatialTransform {
            scale: [0.0, 1.0],
            ..SpatialTransform::default()
        };
        assert_eq!(
            collapsed.sanitized().gpu_uniforms(512, 512, 512, 512).modes[2],
            0,
            "the fixture must really be the shader's invalid case"
        );
        assert!(
            GizmoFrame::new(collapsed, SQUARE, SQUARE).is_none(),
            "a transform that renders nothing must not offer a grabbable handle"
        );
    }

    #[test]
    fn non_finite_and_zero_dimension_inputs_fail_closed() {
        // Non-finite authored values sanitize to the documented defaults rather
        // than to clamped extremes, so this frame is renderable...
        let hostile = SpatialTransform {
            position: [f32::NAN, f32::INFINITY],
            scale: [f32::NEG_INFINITY, 2.0],
            rotation_deg: f32::NAN,
            ..SpatialTransform::default()
        };
        let frame = frame_for(hostile, SQUARE, SQUARE);
        assert_eq!(frame.transform().position, [0.0, 0.0]);
        assert_eq!(frame.transform().scale, [1.0, 2.0]);
        assert_eq!(frame.transform().rotation_deg, 0.0);

        // ...but a zero output dimension has no composition to map into.
        assert!(GizmoFrame::new(SpatialTransform::default(), SQUARE, (0, 1080)).is_none());
        assert!(GizmoFrame::new(SpatialTransform::default(), SQUARE, (1920, 0)).is_none());
        // A zero source dimension is the shader's invalid case too.
        assert!(GizmoFrame::new(SpatialTransform::default(), (0, 480), WIDE).is_none());

        // A non-finite pointer never picks a handle.
        assert!(frame.hit_test([f32::NAN, 0.5]).is_none());
        assert!(frame.hit_test([0.5, f32::INFINITY]).is_none());
    }

    #[test]
    fn hit_testing_never_authors_anything() {
        // The exact legacy identity must survive being looked at.
        let transform = SpatialTransform::default();
        let frame = frame_for(transform, SOURCE_43, WIDE);
        assert_eq!(
            frame.uniforms().modes[3],
            0,
            "the fixture starts on the exact historical sample"
        );
        for probe in [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.2, 0.8]] {
            let _ = frame.hit_test(probe);
            let _ = frame.local_at(probe);
            let _ = frame.footprint();
        }
        assert_eq!(frame.transform(), transform);
        assert_eq!(
            frame.transform().gpu_uniforms(640, 480, 1920, 1080).modes[3],
            0,
            "opening, hovering, and hit testing must leave spatial_modes.w at zero"
        );
    }

    // ---- handles --------------------------------------------------------

    #[test]
    fn handles_are_picked_by_documented_priority() {
        let frame = frame_for(SpatialTransform::default(), SQUARE, SQUARE);
        // The default anchor sits at the footprint centre.
        assert_eq!(frame.hit_test([0.5, 0.5]), Some(GizmoHandle::Anchor));
        assert_eq!(
            frame.hit_test([0.0, 0.0]),
            Some(GizmoHandle::Scale(ScaleCorner::TopLeft))
        );
        assert_eq!(
            frame.hit_test([1.0, 1.0]),
            Some(GizmoHandle::Scale(ScaleCorner::BottomRight))
        );
        assert_eq!(
            frame.hit_test([0.0, 0.5]),
            Some(GizmoHandle::Crop(CropEdge::Left))
        );
        assert_eq!(
            frame.hit_test([0.5, -ROTATE_HANDLE_LOCAL_OFFSET]),
            Some(GizmoHandle::Rotate)
        );
        assert_eq!(
            frame.hit_test([0.5, 1.0 + MOVE_HANDLE_LOCAL_OFFSET]),
            Some(GizmoHandle::Translate)
        );
        // Well outside the footprint and outside every handle.
        assert_eq!(frame.hit_test([2.5, 2.5]), None);
    }

    /// The boundary that keeps the preview usable for two things at once.
    ///
    /// An untransformed source covers the whole composition, so a body-sized
    /// translate target would have claimed every drag over the image and taken
    /// the gesture-etch surface away entirely. Every pixel that is not on a
    /// handle must report no hit; the host then routes that drag to the etch
    /// stroke it always belonged to.
    #[test]
    fn the_gizmo_claims_only_its_handles_and_leaves_the_rest_of_the_preview_alone() {
        let frame = frame_for(SpatialTransform::default(), SQUARE, SQUARE);
        for probe in [
            [0.3, 0.7],
            [0.7, 0.3],
            [0.2, 0.2],
            [0.8, 0.8],
            [0.5, 0.25],
            [0.5, 0.75],
        ] {
            assert_eq!(
                frame.hit_test(probe),
                None,
                "{probe:?} is inside the footprint but on no handle, so it belongs to the etch surface"
            );
        }
        // ...while every handle still answers.
        for handle in GizmoFrame::ordered_handles() {
            let position = frame.handle_position(handle).expect("handle is placed");
            assert!(
                frame.hit_test(position).is_some(),
                "{handle:?} must remain grabbable"
            );
        }
    }

    #[test]
    fn the_footprint_follows_the_authored_geometry() {
        // A half-scale layer covers the middle half of the composition.
        let frame = frame_for(
            SpatialTransform {
                scale: [0.5, 0.5],
                ..SpatialTransform::default()
            },
            SQUARE,
            SQUARE,
        );
        let corners = frame.footprint().unwrap();
        close(corners[0][0], 0.25, 1.0e-4);
        close(corners[0][1], 0.25, 1.0e-4);
        close(corners[2][0], 0.75, 1.0e-4);
        close(corners[2][1], 0.75, 1.0e-4);
    }

    // ---- drag laws ------------------------------------------------------

    #[test]
    fn translate_moves_position_one_for_one_in_output_uv() {
        let frame = frame_for(SpatialTransform::default(), SQUARE, SQUARE);
        let grab = move_handle(&frame);
        let (drag, handle) = GizmoDrag::begin(GizmoScope::Master, frame, grab).unwrap();
        assert_eq!(handle, GizmoHandle::Translate);
        let edits = drag.update([grab[0] + 0.1, grab[1] - 0.05], GizmoModifiers::NONE);
        close(edits.value_of(GizmoParam::PositionX).unwrap(), 0.1, 1.0e-5);
        close(
            edits.value_of(GizmoParam::PositionY).unwrap(),
            -0.05,
            1.0e-5,
        );
    }

    #[test]
    fn shift_locks_translation_to_the_dominant_physical_axis() {
        let frame = frame_for(SpatialTransform::default(), SOURCE_43, WIDE);
        let grab = move_handle(&frame);
        let (drag, _) = GizmoDrag::begin(GizmoScope::Master, frame, grab).unwrap();
        let shift = GizmoModifiers {
            shift: true,
            alt: false,
        };
        // A 16:9 output makes equal UV deltas unequal on screen: 0.05 of width
        // is nearly twice 0.05 of height, so X must win.
        let edits = drag.update([grab[0] + 0.05, grab[1] + 0.05], shift);
        close(edits.value_of(GizmoParam::PositionX).unwrap(), 0.05, 1.0e-5);
        close(edits.value_of(GizmoParam::PositionY).unwrap(), 0.0, 1.0e-5);

        // A clearly vertical drag locks the other way.
        let edits = drag.update([grab[0] + 0.01, grab[1] + 0.2], shift);
        close(edits.value_of(GizmoParam::PositionX).unwrap(), 0.0, 1.0e-5);
        close(edits.value_of(GizmoParam::PositionY).unwrap(), 0.2, 1.0e-5);
    }

    #[test]
    fn alt_is_the_fine_law_for_every_handle() {
        let frame = frame_for(SpatialTransform::default(), SQUARE, SQUARE);
        let alt = GizmoModifiers {
            shift: false,
            alt: true,
        };
        let grab = move_handle(&frame);
        let (translate, _) = GizmoDrag::begin(GizmoScope::Master, frame, grab).unwrap();
        let edits = translate.update([grab[0] + 0.1, grab[1]], alt);
        close(
            edits.value_of(GizmoParam::PositionX).unwrap(),
            0.1 * FINE_DRAG_FACTOR,
            1.0e-5,
        );

        // The anchor reads an absolute position, so Alt must still damp it.
        let (anchor, handle) = GizmoDrag::begin(GizmoScope::Master, frame, [0.5, 0.5]).unwrap();
        assert_eq!(handle, GizmoHandle::Anchor);
        let coarse = anchor.update([0.9, 0.5], GizmoModifiers::NONE);
        let fine = anchor.update([0.9, 0.5], alt);
        let coarse_x = coarse.value_of(GizmoParam::AnchorX).unwrap();
        let fine_x = fine.value_of(GizmoParam::AnchorX).unwrap();
        close(fine_x - 0.5, (coarse_x - 0.5) * FINE_DRAG_FACTOR, 1.0e-5);
    }

    #[test]
    fn scaling_is_the_ratio_of_lever_arms_about_the_anchor() {
        let frame = frame_for(SpatialTransform::default(), SQUARE, SQUARE);
        // Grab the bottom-right corner and pull it to twice its distance from
        // the centre anchor: both axes must double.
        let (drag, handle) = GizmoDrag::begin(GizmoScope::Master, frame, [1.0, 1.0]).unwrap();
        assert_eq!(handle, GizmoHandle::Scale(ScaleCorner::BottomRight));
        let edits = drag.update([1.5, 1.5], GizmoModifiers::NONE);
        close(edits.value_of(GizmoParam::ScaleX).unwrap(), 2.0, 1.0e-4);
        close(edits.value_of(GizmoParam::ScaleY).unwrap(), 2.0, 1.0e-4);

        // A purely horizontal pull leaves Y alone.
        let edits = drag.update([1.5, 1.0], GizmoModifiers::NONE);
        close(edits.value_of(GizmoParam::ScaleX).unwrap(), 2.0, 1.0e-4);
        close(edits.value_of(GizmoParam::ScaleY).unwrap(), 1.0, 1.0e-4);

        // ...unless Shift asks for a uniform scale.
        let edits = drag.update(
            [1.5, 1.0],
            GizmoModifiers {
                shift: true,
                alt: false,
            },
        );
        close(edits.value_of(GizmoParam::ScaleX).unwrap(), 2.0, 1.0e-4);
        close(edits.value_of(GizmoParam::ScaleY).unwrap(), 2.0, 1.0e-4);
    }

    #[test]
    fn a_scale_grab_on_the_pivot_refuses_that_axis() {
        // Anchor pinned to the corner being grabbed: the lever arm is zero, so
        // there is no ratio to author and the axis must be left alone rather
        // than driven to a clamped extreme by a division.
        let frame = frame_for(
            SpatialTransform {
                anchor: [1.0, 1.0],
                ..SpatialTransform::default()
            },
            SQUARE,
            SQUARE,
        );
        let corner = frame
            .handle_position(GizmoHandle::Scale(ScaleCorner::BottomRight))
            .unwrap();
        let (drag, _) = GizmoDrag::begin(GizmoScope::Master, frame, corner).unwrap();
        let edits = drag.update([corner[0] + 0.2, corner[1] + 0.2], GizmoModifiers::NONE);
        assert!(
            edits.value_of(GizmoParam::ScaleX).is_none(),
            "a zero lever arm must author nothing on that axis"
        );
        assert!(edits.value_of(GizmoParam::ScaleY).is_none());
    }

    #[test]
    fn rotation_is_measured_in_physical_space_and_snaps_under_shift() {
        let frame = frame_for(SpatialTransform::default(), SQUARE, SQUARE);
        let pivot = frame.anchor_output();
        // Grab directly right of the pivot and sweep to directly below it: a
        // quarter turn in screen coordinates.
        let (drag, _) = GizmoDrag::begin(
            GizmoScope::Master,
            frame,
            frame
                .handle_position(GizmoHandle::Rotate)
                .expect("rotate handle exists"),
        )
        .unwrap();
        let radius = pivot[1] - frame.handle_position(GizmoHandle::Rotate).unwrap()[1];
        let quarter = [pivot[0] + radius, pivot[1]];
        let edits = drag.update(quarter, GizmoModifiers::NONE);
        close(
            edits.value_of(GizmoParam::RotationDeg).unwrap(),
            90.0,
            1.0e-3,
        );

        // Shift snaps the authored angle onto the grid.
        let nearly = [
            pivot[0] + radius * 0.9_f32.to_radians().cos(),
            pivot[1] - radius * 0.9_f32.to_radians().sin(),
        ];
        let snapped = drag
            .update(
                nearly,
                GizmoModifiers {
                    shift: true,
                    alt: false,
                },
            )
            .value_of(GizmoParam::RotationDeg)
            .unwrap();
        close(snapped % ROTATE_SNAP_DEGREES, 0.0, 1.0e-3);
    }

    #[test]
    fn rotation_on_a_wide_output_authors_a_true_quarter_turn() {
        // The same sweep on 16:9 must still author 90 degrees, because the
        // angle is measured in physical space exactly as `gpu_uniforms`
        // conjugates it back. A UV-space measurement would report roughly 61
        // degrees here, so this is the test that would catch a missing
        // conjugation.
        let frame = frame_for(SpatialTransform::default(), SOURCE_43, WIDE);
        let pivot = frame.anchor_output();
        let grab = frame
            .handle_position(GizmoHandle::Rotate)
            .expect("the rotate handle is placed");
        let (drag, handle) = GizmoDrag::begin(GizmoScope::Master, frame, grab).unwrap();
        assert_eq!(handle, GizmoHandle::Rotate, "the fixture must grab Rotate");

        // Rotate the grab vector by exactly a quarter turn in physical space,
        // then map it back into output UV — the inverse of what the drag law
        // does internally, so the two must agree on 90 degrees.
        let aspect = WIDE.0 as f32 / WIDE.1 as f32;
        let physical = [(grab[0] - pivot[0]) * aspect, grab[1] - pivot[1]];
        let turned = apply_2x2(
            crate::spatial::rotation_matrix(90.0_f32.to_radians()),
            physical,
        );
        let target = [pivot[0] + turned[0] / aspect, pivot[1] + turned[1]];

        let edits = drag.update(target, GizmoModifiers::NONE);
        close(
            edits.value_of(GizmoParam::RotationDeg).unwrap(),
            90.0,
            1.0e-2,
        );
    }

    #[test]
    fn the_anchor_handle_follows_the_pointer_in_uncropped_source_uv() {
        let frame = frame_for(
            SpatialTransform {
                crop: [0.2, 0.1, 0.1, 0.3],
                ..SpatialTransform::default()
            },
            SQUARE,
            SQUARE,
        );
        let (drag, _) = GizmoDrag::begin(GizmoScope::Master, frame, frame.anchor_output()).unwrap();
        let target = [0.25, 0.75];
        let edits = drag.update(target, GizmoModifiers::NONE);
        let local = frame.local_at(target).unwrap();
        let crop = frame.uniforms().crop;
        close(
            edits.value_of(GizmoParam::AnchorX).unwrap(),
            crop[0] + local[0] * crop[2],
            1.0e-5,
        );
        close(
            edits.value_of(GizmoParam::AnchorY).unwrap(),
            crop[1] + local[1] * crop[3],
            1.0e-5,
        );
    }

    #[test]
    fn crop_edges_author_trims_and_shift_makes_them_symmetric() {
        let frame = frame_for(SpatialTransform::default(), SQUARE, SQUARE);
        let (drag, handle) = GizmoDrag::begin(GizmoScope::Master, frame, [0.0, 0.5]).unwrap();
        assert_eq!(handle, GizmoHandle::Crop(CropEdge::Left));
        let edits = drag.update([0.25, 0.5], GizmoModifiers::NONE);
        close(edits.value_of(GizmoParam::CropLeft).unwrap(), 0.25, 1.0e-5);
        assert!(edits.value_of(GizmoParam::CropRight).is_none());

        let symmetric = drag.update(
            [0.25, 0.5],
            GizmoModifiers {
                shift: true,
                alt: false,
            },
        );
        close(
            symmetric.value_of(GizmoParam::CropLeft).unwrap(),
            0.25,
            1.0e-5,
        );
        close(
            symmetric.value_of(GizmoParam::CropRight).unwrap(),
            0.25,
            1.0e-5,
        );

        // The far-side edge authors the complementary trim.
        let (right, handle) = GizmoDrag::begin(GizmoScope::Master, frame, [1.0, 0.5]).unwrap();
        assert_eq!(handle, GizmoHandle::Crop(CropEdge::Right));
        let edits = right.update([0.8, 0.5], GizmoModifiers::NONE);
        close(edits.value_of(GizmoParam::CropRight).unwrap(), 0.2, 1.0e-5);
    }

    // ---- bounds and sanitization ---------------------------------------

    #[test]
    fn every_authored_value_lands_inside_the_spatial_contract() {
        let frame = frame_for(SpatialTransform::default(), SQUARE, SQUARE);
        let (drag, _) = GizmoDrag::begin(GizmoScope::Master, frame, move_handle(&frame)).unwrap();
        // A drag far outside any plausible pane must still author in range.
        let edits = drag.update([1.0e9, -1.0e9], GizmoModifiers::NONE);
        let x = edits.value_of(GizmoParam::PositionX).unwrap();
        let y = edits.value_of(GizmoParam::PositionY).unwrap();
        assert!((POSITION_MIN..=POSITION_MAX).contains(&x), "{x}");
        assert!((POSITION_MIN..=POSITION_MAX).contains(&y), "{y}");

        // A non-finite pointer authors nothing at all.
        assert!(drag
            .update([f32::NAN, 0.5], GizmoModifiers::NONE)
            .is_empty());
    }

    #[test]
    fn a_non_finite_computation_takes_the_neutral_value_not_a_clamped_extreme() {
        assert_eq!(GizmoParam::ScaleX.sanitize(f32::INFINITY), 1.0);
        assert_eq!(GizmoParam::ScaleY.sanitize(f32::NAN), 1.0);
        assert_eq!(GizmoParam::AnchorX.sanitize(f32::NEG_INFINITY), 0.5);
        assert_eq!(GizmoParam::PositionX.sanitize(f32::NAN), 0.0);
        assert_eq!(GizmoParam::CropLeft.sanitize(f32::INFINITY), 0.0);
        assert_eq!(GizmoParam::RotationDeg.sanitize(f32::NAN), 0.0);
        // A finite out-of-range value still clamps normally.
        assert_eq!(GizmoParam::ScaleX.sanitize(1.0e9), SCALE_MAX);
    }

    #[test]
    fn the_edit_set_is_bounded_and_allocation_free() {
        let mut edits = GizmoEdits::EMPTY;
        for _ in 0..(MAX_GIZMO_EDITS + 3) {
            edits.push(GizmoParam::PositionX, 0.1);
        }
        assert_eq!(edits.len(), MAX_GIZMO_EDITS);
        assert_eq!(edits.iter().count(), MAX_GIZMO_EDITS);
    }

    // ---- cancel law -----------------------------------------------------

    #[test]
    fn escape_cancels_before_a_commit_and_undoes_after_one() {
        let transform = SpatialTransform {
            position: [0.05, 0.05],
            ..SpatialTransform::default()
        };
        let frame = frame_for(transform, SQUARE, SQUARE);
        let (mut drag, _) =
            GizmoDrag::begin(GizmoScope::Master, frame, move_handle(&frame)).unwrap();
        assert!(!drag.has_committed());
        assert_eq!(
            drag.cancel(),
            GizmoCancel::Restore(transform.sanitized()),
            "before the first commit Escape restores the captured transform"
        );
        drag.mark_committed();
        assert!(drag.has_committed());
        assert_eq!(
            drag.cancel(),
            GizmoCancel::UndoCommitted,
            "after a commit Escape is an ordinary undo, not a cancel"
        );
    }

    #[test]
    fn a_drag_captures_its_scope_and_cannot_retarget() {
        let layer = StableLayerId::new(77).unwrap();
        let group = GroupId::new(91).unwrap();
        let frame = frame_for(SpatialTransform::default(), SQUARE, SQUARE);
        let (drag, _) =
            GizmoDrag::begin(GizmoScope::Layer(layer), frame, move_handle(&frame)).unwrap();
        assert_eq!(drag.scope(), GizmoScope::Layer(layer));
        // The captured transform is immutable for the life of the drag, so a
        // later authored change cannot be read back into the drag's own math.
        assert_eq!(drag.captured(), SpatialTransform::default());

        let (group_drag, _) =
            GizmoDrag::begin(GizmoScope::Group(group), frame, move_handle(&frame)).unwrap();
        assert_eq!(group_drag.scope(), GizmoScope::Group(group));
        assert_eq!(group_drag.captured(), SpatialTransform::default());
    }

    #[test]
    fn a_group_frame_uses_the_output_sized_composite_not_member_geometry() {
        let group = GroupId::new(91).unwrap();
        let transform = SpatialTransform {
            position: [0.2, -0.15],
            scale: [0.75, 1.25],
            rotation_deg: 23.0,
            fit: FitMode::Fit,
            ..SpatialTransform::default()
        };
        let output_frame = frame_for(transform, WIDE, WIDE);
        let member_sized_frame = frame_for(transform, SOURCE_43, WIDE);
        let (drag, _) = GizmoDrag::begin(
            GizmoScope::Group(group),
            output_frame,
            move_handle(&output_frame),
        )
        .unwrap();

        assert_eq!(drag.scope(), GizmoScope::Group(group));
        assert_eq!(drag.captured(), transform.sanitized());
        assert_ne!(
            bytemuck::bytes_of(&output_frame.uniforms()),
            bytemuck::bytes_of(&member_sized_frame.uniforms()),
            "a member's source aspect must not masquerade as group geometry"
        );
    }

    // ---- nudges ---------------------------------------------------------

    #[test]
    fn nudges_follow_the_documented_step_law() {
        let transform = SpatialTransform::default();
        let plain = nudge_edits(transform, GizmoNudge::Right, GizmoModifiers::NONE);
        close(
            plain.value_of(GizmoParam::PositionX).unwrap(),
            NUDGE_STEP,
            1.0e-7,
        );
        close(plain.value_of(GizmoParam::PositionY).unwrap(), 0.0, 1.0e-7);

        let coarse = nudge_edits(
            transform,
            GizmoNudge::Down,
            GizmoModifiers {
                shift: true,
                alt: false,
            },
        );
        close(
            coarse.value_of(GizmoParam::PositionY).unwrap(),
            NUDGE_COARSE_STEP,
            1.0e-7,
        );

        let fine = nudge_edits(
            transform,
            GizmoNudge::Up,
            GizmoModifiers {
                shift: false,
                alt: true,
            },
        );
        close(
            fine.value_of(GizmoParam::PositionY).unwrap(),
            -NUDGE_FINE_STEP,
            1.0e-7,
        );

        // Alt wins when both are held: the more precise request is the one the
        // operator is asking for.
        let both = nudge_edits(
            transform,
            GizmoNudge::Left,
            GizmoModifiers {
                shift: true,
                alt: true,
            },
        );
        close(
            both.value_of(GizmoParam::PositionX).unwrap(),
            -NUDGE_FINE_STEP,
            1.0e-7,
        );
    }

    // ---- permit ---------------------------------------------------------

    /// The permit's barrier is structural, and this asserts the structure
    /// rather than the behaviour.
    ///
    /// The type is declared once and constructed once, both inside the sealed
    /// submodule, and it derives nothing — no `Clone`, no `Copy`, no
    /// `Default` — so a held permit cannot be duplicated into a second painter
    /// and an audience-facing consumer cannot conjure one from thin air. A
    /// source audit is the same idiom the Symmetry hue law already uses, and it
    /// fails loudly if a future edit adds a second way in.
    ///
    /// Only the production half of the file is audited, because this test's own
    /// string literals would otherwise count themselves.
    #[test]
    fn the_gizmo_permit_has_exactly_one_constructor() {
        let source = include_str!("transform_gizmo.rs");
        let production = source
            .split_once("mod tests {")
            .expect("the file ends in a test module")
            .0;
        let declaration = format!("pub struct PreviewGizmoPermit({})", "()");
        let construction = format!("PreviewGizmoPermit({})", "()");
        assert_eq!(
            production.matches(declaration.as_str()).count(),
            1,
            "the permit is declared exactly once"
        );
        assert_eq!(
            production.matches(construction.as_str()).count(),
            2,
            "the declaration plus exactly one construction, and nothing else"
        );
        assert!(
            !production.contains("derive(Clone"),
            "the permit must not become duplicable"
        );
        // And both really sit inside the sealed submodule, where this file's
        // own test module is a sibling rather than a descendant.
        let sealed = production
            .split_once("mod permit {")
            .expect("the permit submodule exists")
            .1
            .split_once("pub use permit::")
            .expect("the submodule is re-exported")
            .0;
        assert_eq!(sealed.matches(construction.as_str()).count(), 2);
    }

    #[test]
    fn the_permit_is_minted_only_for_a_visible_editor_preview() {
        assert!(preview_gizmo_permit(&StageSurface::EditorPreview, false).is_some());
        // Single-monitor Output reusing the main swapchain removes it by the
        // same predicate that removes the RECOVERY strip.
        assert!(preview_gizmo_permit(&StageSurface::EditorPreview, true).is_none());
        for surface in [
            StageSurface::Composite,
            StageSurface::Audience,
            StageSurface::Spout,
            StageSurface::Record,
            StageSurface::Export,
            StageSurface::PhysicalOutput(
                crate::stage_map::OutputEndpointId::parse("front-of-house").unwrap(),
            ),
        ] {
            assert!(
                preview_gizmo_permit(&surface, false).is_none(),
                "no audience-facing surface may mint a gizmo permit: {surface:?}"
            );
            assert!(preview_gizmo_permit(&surface, true).is_none());
        }
    }

    #[test]
    fn an_explicit_clamp_edge_survives_a_gizmo_drag() {
        // The gizmo authors continuous geometry only. An explicitly authored
        // edge law is not its business and must come back unchanged.
        let transform = SpatialTransform {
            edge: EdgeMode::Clamp,
            ..SpatialTransform::default()
        };
        let frame = frame_for(transform, SQUARE, SQUARE);
        let (drag, _) = GizmoDrag::begin(GizmoScope::Master, frame, move_handle(&frame)).unwrap();
        let edits = drag.update([0.5, 0.5], GizmoModifiers::NONE);
        assert!(!edits.is_empty());
        for edit in edits.iter() {
            assert!(
                !matches!(
                    edit.param,
                    GizmoParam::CropLeft | GizmoParam::CropTop | GizmoParam::CropRight
                ),
                "a translate drag must not touch crop"
            );
        }
        assert_eq!(frame.transform().edge, EdgeMode::Clamp);
    }
}
