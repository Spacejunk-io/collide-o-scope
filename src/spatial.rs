//! Resolution-independent authored spatial transforms.
//!
//! Runtime state remains separate from the legacy effect uniform block. The
//! GPU receives one packed pass uniform containing both structures so the
//! existing single-pass layer shader can apply the affine transform without
//! allocating another full-frame texture.

use serde::{Deserialize, Serialize};

use crate::effects::EffectUniforms;

pub const POSITION_MIN: f32 = -4.0;
pub const POSITION_MAX: f32 = 4.0;
pub const SCALE_MIN: f32 = -16.0;
pub const SCALE_MAX: f32 = 16.0;
pub const ANCHOR_MIN: f32 = -2.0;
pub const ANCHOR_MAX: f32 = 3.0;
pub const SKEW_LIMIT_DEGREES: f32 = 89.0;
pub const CROP_MAX: f32 = 1.0 - MIN_CROP_EXTENT;

const MIN_CROP_EXTENT: f32 = 1.0 / 4096.0;
const MIN_INVERTIBLE_SCALE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitMode {
    /// Historical behavior: map the complete source UV rectangle to the
    /// complete composition, regardless of aspect ratio.
    #[default]
    Stretch,
    Fit,
    Fill,
    Native,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeMode {
    /// Safe authored default once a transform exposes canvas outside a source.
    #[default]
    Transparent,
    /// Explicitly extend the source's border pixels.
    Clamp,
    Repeat,
    Mirror,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMode {
    #[default]
    Linear,
    Nearest,
}

/// Authored transform in normalized composition/source coordinates.
///
/// Position zero means the composition center. Anchor is expressed in the
/// original source UV rectangle. Changing only the anchor is therefore
/// visually inert: it selects the pivot for later scale, rotation, and skew,
/// rather than acting as an implicit translation.
/// Positive rotation follows screen coordinates and therefore turns clockwise.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpatialTransform {
    pub position: [f32; 2],
    pub scale: [f32; 2],
    pub anchor: [f32; 2],
    pub rotation_deg: f32,
    pub skew_deg: f32,
    pub skew_axis_deg: f32,
    pub fit: FitMode,
    /// Left, top, right, bottom fractions of the uncropped source.
    pub crop: [f32; 4],
    pub edge: EdgeMode,
    pub sampling: SamplingMode,
}

impl Default for SpatialTransform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            scale: [1.0, 1.0],
            anchor: [0.5, 0.5],
            rotation_deg: 0.0,
            skew_deg: 0.0,
            skew_axis_deg: 0.0,
            fit: FitMode::Stretch,
            crop: [0.0; 4],
            // Identity itself takes the exact historical shader bypass. Once
            // any authored field activates spatial sampling, exposed canvas
            // must start transparent unless Clamp was selected explicitly.
            edge: EdgeMode::Transparent,
            sampling: SamplingMode::Linear,
        }
    }
}

impl SpatialTransform {
    /// Creative default for a newly added layer. Missing transform data in an
    /// old patch still uses Default and therefore preserves the exact inactive
    /// historical sample path; once transformed, its exposed edge is clear.
    pub fn new_layer_default() -> Self {
        Self {
            fit: FitMode::Fit,
            edge: EdgeMode::Transparent,
            ..Self::default()
        }
    }

    pub fn sanitized(self) -> Self {
        let mut clean = Self {
            position: [
                finite_clamp(self.position[0], 0.0, POSITION_MIN, POSITION_MAX),
                finite_clamp(self.position[1], 0.0, POSITION_MIN, POSITION_MAX),
            ],
            scale: [
                finite_clamp(self.scale[0], 1.0, SCALE_MIN, SCALE_MAX),
                finite_clamp(self.scale[1], 1.0, SCALE_MIN, SCALE_MAX),
            ],
            anchor: [
                finite_clamp(self.anchor[0], 0.5, ANCHOR_MIN, ANCHOR_MAX),
                finite_clamp(self.anchor[1], 0.5, ANCHOR_MIN, ANCHOR_MAX),
            ],
            rotation_deg: wrap_degrees(self.rotation_deg),
            skew_deg: finite_clamp(self.skew_deg, 0.0, -SKEW_LIMIT_DEGREES, SKEW_LIMIT_DEGREES),
            skew_axis_deg: wrap_degrees(self.skew_axis_deg),
            fit: self.fit,
            crop: [
                finite_clamp(self.crop[0], 0.0, 0.0, 1.0 - MIN_CROP_EXTENT),
                finite_clamp(self.crop[1], 0.0, 0.0, 1.0 - MIN_CROP_EXTENT),
                finite_clamp(self.crop[2], 0.0, 0.0, 1.0 - MIN_CROP_EXTENT),
                finite_clamp(self.crop[3], 0.0, 0.0, 1.0 - MIN_CROP_EXTENT),
            ],
            edge: self.edge,
            sampling: self.sampling,
        };
        clean.crop[2] = clean.crop[2].min(1.0 - MIN_CROP_EXTENT - clean.crop[0]);
        clean.crop[3] = clean.crop[3].min(1.0 - MIN_CROP_EXTENT - clean.crop[1]);
        clean
    }

    pub fn is_legacy_identity(self) -> bool {
        self.sanitized() == Self::default()
    }

    /// Stable, allocation-free signature for render-plan/topology invalidation.
    /// It is not a persisted content identity or a cryptographic digest.
    pub fn fingerprint(self) -> u64 {
        let clean = self.sanitized();
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for value in clean
            .position
            .into_iter()
            .chain(clean.scale)
            .chain(clean.anchor)
            .chain([clean.rotation_deg, clean.skew_deg, clean.skew_axis_deg])
            .chain(clean.crop)
        {
            let bits = if value == 0.0 { 0 } else { value.to_bits() };
            for byte in bits.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        }
        for byte in [
            fit_mode_code(clean.fit),
            edge_mode_code(clean.edge),
            sampling_mode_code(clean.sampling),
        ] {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }

    pub fn interpolate(a: Self, b: Self, weights: [f32; 2], choose_b: bool) -> Self {
        let a = a.sanitized();
        let b = b.sanitized();
        if weights[1] == 0.0 {
            return a;
        }
        if weights[0] == 0.0 {
            return b;
        }
        Self {
            position: [
                blend(a.position[0], b.position[0], weights),
                blend(a.position[1], b.position[1], weights),
            ],
            scale: [
                blend_scale(a.scale[0], b.scale[0], weights),
                blend_scale(a.scale[1], b.scale[1], weights),
            ],
            anchor: [
                blend(a.anchor[0], b.anchor[0], weights),
                blend(a.anchor[1], b.anchor[1], weights),
            ],
            rotation_deg: blend_degrees(a.rotation_deg, b.rotation_deg, weights),
            skew_deg: blend(a.skew_deg, b.skew_deg, weights),
            skew_axis_deg: blend_degrees(a.skew_axis_deg, b.skew_axis_deg, weights),
            fit: if choose_b { b.fit } else { a.fit },
            crop: [
                blend(a.crop[0], b.crop[0], weights),
                blend(a.crop[1], b.crop[1], weights),
                blend(a.crop[2], b.crop[2], weights),
                blend(a.crop[3], b.crop[3], weights),
            ],
            edge: if choose_b { b.edge } else { a.edge },
            sampling: if choose_b { b.sampling } else { a.sampling },
        }
        .sanitized()
    }

    pub fn gpu_uniforms(
        self,
        source_width: u32,
        source_height: u32,
        output_width: u32,
        output_height: u32,
    ) -> SpatialGpuUniforms {
        let clean = self.sanitized();
        let legacy_bypass = clean.is_legacy_identity();
        let crop_width = 1.0 - clean.crop[0] - clean.crop[2];
        let crop_height = 1.0 - clean.crop[1] - clean.crop[3];
        let crop = [clean.crop[0], clean.crop[1], crop_width, crop_height];
        let invalid_dimensions =
            source_width == 0 || source_height == 0 || output_width == 0 || output_height == 0;
        let collapsed = clean.scale[0].abs() < MIN_INVERTIBLE_SCALE
            || clean.scale[1].abs() < MIN_INVERTIBLE_SCALE;
        if invalid_dimensions || collapsed {
            return SpatialGpuUniforms::invalid(crop, clean.edge, clean.sampling);
        }

        let source_aspect =
            (source_width as f32 * crop_width) / (source_height as f32 * crop_height);
        let output_aspect = output_width as f32 / output_height as f32;
        let fit_size = match clean.fit {
            FitMode::Stretch => [1.0, 1.0],
            FitMode::Fit if source_aspect >= output_aspect => [1.0, output_aspect / source_aspect],
            FitMode::Fit => [source_aspect / output_aspect, 1.0],
            FitMode::Fill if source_aspect >= output_aspect => [source_aspect / output_aspect, 1.0],
            FitMode::Fill => [1.0, output_aspect / source_aspect],
            FitMode::Native => [
                source_width as f32 * crop_width / output_width as f32,
                source_height as f32 * crop_height / output_height as f32,
            ],
        };
        let scale = [fit_size[0] * clean.scale[0], fit_size[1] * clean.scale[1]];

        let rotation = rotation_matrix(clean.rotation_deg.to_radians());
        let axis = rotation_matrix(clean.skew_axis_deg.to_radians());
        let inverse_axis = rotation_matrix(-clean.skew_axis_deg.to_radians());
        let shear = [[1.0, clean.skew_deg.to_radians().tan()], [0.0, 1.0]];
        let authored_physical = multiply_2x2(
            rotation,
            multiply_2x2(axis, multiply_2x2(shear, inverse_axis)),
        );
        // UV X and Y are not equal physical distances on a non-square output.
        // Conjugate the authored rotation/shear through pixel-aspect space so
        // 90 degrees remains 90 degrees on screen.
        let authored = conjugate_through_output_aspect(authored_physical, output_aspect);
        let forward = multiply_2x2(authored, [[scale[0], 0.0], [0.0, scale[1]]]);
        let determinant = forward[0][0] * forward[1][1] - forward[0][1] * forward[1][0];
        // Authored near-zero scale was rejected above. Fit/Native may
        // legitimately produce a very small determinant (for example a 4×3
        // pixel source mapped one-for-one into 1920×1080); an absolute affine
        // determinant cutoff would erase that valid footprint. Reject only a
        // truly singular/non-finite matrix, then validate the computed inverse.
        if !determinant.is_finite() || determinant == 0.0 {
            return SpatialGpuUniforms::invalid(crop, clean.edge, clean.sampling);
        }
        let inverse = [
            [forward[1][1] / determinant, -forward[0][1] / determinant],
            [-forward[1][0] / determinant, forward[0][0] / determinant],
        ];
        if inverse
            .into_iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return SpatialGpuUniforms::invalid(crop, clean.edge, clean.sampling);
        }
        let anchor = [
            (clean.anchor[0] - clean.crop[0]) / crop_width,
            (clean.anchor[1] - clean.crop[1]) / crop_height,
        ];
        let target = [
            0.5 + clean.position[0] + fit_size[0] * (anchor[0] - 0.5),
            0.5 + clean.position[1] + fit_size[1] * (anchor[1] - 0.5),
        ];
        let translate = [
            anchor[0] - inverse[0][0] * target[0] - inverse[0][1] * target[1],
            anchor[1] - inverse[1][0] * target[0] - inverse[1][1] * target[1],
        ];
        SpatialGpuUniforms {
            inverse_row_0: [inverse[0][0], inverse[0][1], translate[0], 0.0],
            inverse_row_1: [inverse[1][0], inverse[1][1], translate[1], 0.0],
            crop,
            modes: [
                edge_mode_code(clean.edge),
                sampling_mode_code(clean.sampling),
                1,
                u32::from(!legacy_bypass),
            ],
        }
    }
}

/// Four vec4 slots appended after the ten legacy EffectUniforms slots.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpatialGpuUniforms {
    pub inverse_row_0: [f32; 4],
    pub inverse_row_1: [f32; 4],
    /// Crop origin X/Y followed by crop extent X/Y.
    pub crop: [f32; 4],
    /// Edge mode, sampling mode, valid flag, spatial-active flag.
    pub modes: [u32; 4],
}

impl SpatialGpuUniforms {
    fn invalid(crop: [f32; 4], edge: EdgeMode, sampling: SamplingMode) -> Self {
        Self {
            inverse_row_0: [0.0; 4],
            inverse_row_1: [0.0; 4],
            crop,
            modes: [edge_mode_code(edge), sampling_mode_code(sampling), 0, 1],
        }
    }

    #[cfg(test)]
    fn map_output_to_local(self, output_uv: [f32; 2]) -> [f32; 2] {
        [
            self.inverse_row_0[0] * output_uv[0]
                + self.inverse_row_0[1] * output_uv[1]
                + self.inverse_row_0[2],
            self.inverse_row_1[0] * output_uv[0]
                + self.inverse_row_1[1] * output_uv[1]
                + self.inverse_row_1[2],
        ]
    }
}

/// CPU/WGSL layout shared by every effects pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EffectPassUniforms {
    pub effects: EffectUniforms,
    pub spatial: SpatialGpuUniforms,
}

impl EffectPassUniforms {
    pub fn new(effects: EffectUniforms, spatial: SpatialGpuUniforms) -> Self {
        Self { effects, spatial }
    }

    /// Build the one authoritative CPU/WGSL payload used by live and export.
    pub fn for_target(
        effects: EffectUniforms,
        transform: SpatialTransform,
        source_dimensions: (u32, u32),
        output_dimensions: (u32, u32),
    ) -> Self {
        Self::new(
            effects.for_render_target(output_dimensions.0, output_dimensions.1),
            transform.gpu_uniforms(
                source_dimensions.0,
                source_dimensions.1,
                output_dimensions.0,
                output_dimensions.1,
            ),
        )
    }
}

pub(crate) fn finite_clamp(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

pub(crate) fn wrap_degrees(value: f32) -> f32 {
    if value.is_finite() {
        let wrapped = (value + 180.0).rem_euclid(360.0) - 180.0;
        if wrapped == -180.0 && value.is_sign_positive() {
            180.0
        } else {
            wrapped
        }
    } else {
        0.0
    }
}

fn blend(a: f32, b: f32, weights: [f32; 2]) -> f32 {
    a * weights[0] + b * weights[1]
}

fn normalized_t(weights: [f32; 2]) -> f32 {
    let total = weights[0] + weights[1];
    if total.is_finite() && total.abs() > f32::EPSILON {
        (weights[1] / total).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn blend_scale(a: f32, b: f32, weights: [f32; 2]) -> f32 {
    if a.signum() == b.signum()
        && a.abs() >= MIN_INVERTIBLE_SCALE
        && b.abs() >= MIN_INVERTIBLE_SCALE
    {
        let t = normalized_t(weights);
        a.signum() * (a.abs().ln() * (1.0 - t) + b.abs().ln() * t).exp()
    } else {
        blend(a, b, weights)
    }
}

fn blend_degrees(a: f32, b: f32, weights: [f32; 2]) -> f32 {
    let t = normalized_t(weights);
    let delta = (b - a + 180.0).rem_euclid(360.0) - 180.0;
    wrap_degrees(a + delta * t)
}

pub(crate) fn rotation_matrix(angle: f32) -> [[f32; 2]; 2] {
    let (sin, cos) = angle.sin_cos();
    [[cos, -sin], [sin, cos]]
}

/// Basis change between output UV space and physical (square-pixel) space.
/// Returns `(to_physical, from_physical)`.
///
/// UV X and Y are not equal physical distances on a non-square output. Any
/// authored angle must be expressed in physical space and conjugated back, or a
/// 90 degree turn stops looking like 90 degrees on screen.
pub(crate) fn output_aspect_basis(output_aspect: f32) -> ([[f32; 2]; 2], [[f32; 2]; 2]) {
    (
        [[output_aspect, 0.0], [0.0, 1.0]],
        [[1.0 / output_aspect, 0.0], [0.0, 1.0]],
    )
}

/// Conjugate a linear map authored in physical space into output UV space.
pub(crate) fn conjugate_through_output_aspect(
    authored_physical: [[f32; 2]; 2],
    output_aspect: f32,
) -> [[f32; 2]; 2] {
    let (to_physical, from_physical) = output_aspect_basis(output_aspect);
    multiply_2x2(from_physical, multiply_2x2(authored_physical, to_physical))
}

pub(crate) fn apply_2x2(matrix: [[f32; 2]; 2], vector: [f32; 2]) -> [f32; 2] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

pub(crate) fn multiply_2x2(a: [[f32; 2]; 2], b: [[f32; 2]; 2]) -> [[f32; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

const fn edge_mode_code(mode: EdgeMode) -> u32 {
    match mode {
        EdgeMode::Transparent => 0,
        EdgeMode::Clamp => 1,
        EdgeMode::Repeat => 2,
        EdgeMode::Mirror => 3,
    }
}

const fn fit_mode_code(mode: FitMode) -> u32 {
    match mode {
        FitMode::Stretch => 0,
        FitMode::Fit => 1,
        FitMode::Fill => 2,
        FitMode::Native => 3,
    }
}

const fn sampling_mode_code(mode: SamplingMode) -> u32 {
    match mode {
        SamplingMode::Linear => 0,
        SamplingMode::Nearest => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn gpu_layout_is_wgsl_uniform_aligned() {
        assert_eq!(std::mem::size_of::<SpatialGpuUniforms>(), 64);
        assert_eq!(std::mem::size_of::<EffectPassUniforms>(), 224);
        assert_eq!(std::mem::offset_of!(EffectPassUniforms, spatial), 160);
        assert!(std::mem::size_of::<EffectPassUniforms>().is_multiple_of(16));
    }

    #[test]
    fn legacy_default_is_exact_identity() {
        let gpu = SpatialTransform::default().gpu_uniforms(640, 480, 1920, 1080);
        assert_eq!(SpatialTransform::default().edge, EdgeMode::Transparent);
        assert_eq!(gpu.modes[2], 1);
        assert_eq!(gpu.modes[3], 0);
        assert_eq!(gpu.map_output_to_local([0.0, 0.0]), [0.0, 0.0]);
        assert_eq!(gpu.map_output_to_local([1.0, 1.0]), [1.0, 1.0]);
        assert_eq!(gpu.crop, [0.0, 0.0, 1.0, 1.0]);

        let moved = SpatialTransform {
            position: [0.1, 0.0],
            ..SpatialTransform::default()
        }
        .gpu_uniforms(640, 480, 1920, 1080);
        assert_eq!(moved.modes[0], edge_mode_code(EdgeMode::Transparent));
        assert_eq!(moved.modes[3], 1);

        let explicit_clamp = SpatialTransform {
            edge: EdgeMode::Clamp,
            ..SpatialTransform::default()
        };
        assert!(!explicit_clamp.is_legacy_identity());
        assert_eq!(
            explicit_clamp.gpu_uniforms(640, 480, 1920, 1080).modes[0],
            edge_mode_code(EdgeMode::Clamp)
        );
    }

    #[test]
    fn fit_fill_and_native_preserve_declared_geometry() {
        let fit = SpatialTransform {
            fit: FitMode::Fit,
            ..SpatialTransform::default()
        }
        .gpu_uniforms(640, 480, 1920, 1080);
        close(fit.map_output_to_local([0.125, 0.5])[0], 0.0);
        close(fit.map_output_to_local([0.875, 0.5])[0], 1.0);

        let fill = SpatialTransform {
            fit: FitMode::Fill,
            ..SpatialTransform::default()
        }
        .gpu_uniforms(640, 480, 1920, 1080);
        close(fill.map_output_to_local([0.5, 0.0])[1], 0.125);
        close(fill.map_output_to_local([0.5, 1.0])[1], 0.875);

        let native = SpatialTransform {
            fit: FitMode::Native,
            ..SpatialTransform::default()
        }
        .gpu_uniforms(640, 480, 1920, 1080);
        close(native.map_output_to_local([1.0 / 3.0, 0.5])[0], 0.0);
        close(native.map_output_to_local([2.0 / 3.0, 0.5])[0], 1.0);

        let tiny_native = SpatialTransform {
            fit: FitMode::Native,
            ..SpatialTransform::default()
        }
        .gpu_uniforms(4, 3, 1920, 1080);
        assert_eq!(tiny_native.modes[2], 1);
        assert_eq!(tiny_native.modes[3], 1);
        let tiny_left = tiny_native.map_output_to_local([0.5 - 2.0 / 1920.0, 0.5])[0];
        let tiny_right = tiny_native.map_output_to_local([0.5 + 2.0 / 1920.0, 0.5])[0];
        assert!(tiny_left.abs() <= 2.0e-5, "tiny Native left = {tiny_left}");
        assert!(
            (tiny_right - 1.0).abs() <= 2.0e-5,
            "tiny Native right = {tiny_right}"
        );
    }

    #[test]
    fn anchor_rotation_keeps_anchor_at_authored_position() {
        let transform = SpatialTransform {
            position: [0.2, -0.1],
            anchor: [0.0, 0.0],
            rotation_deg: 90.0,
            ..SpatialTransform::default()
        };
        let gpu = transform.gpu_uniforms(100, 100, 100, 100);
        let local = gpu.map_output_to_local([0.2, -0.1]);
        close(local[0], 0.0);
        close(local[1], 0.0);
    }

    #[test]
    fn changing_only_anchor_does_not_translate_the_source() {
        let gpu = SpatialTransform {
            anchor: [0.15, 0.85],
            ..SpatialTransform::default()
        }
        .gpu_uniforms(640, 480, 1920, 1080);
        close(gpu.map_output_to_local([0.0, 0.0])[0], 0.0);
        close(gpu.map_output_to_local([0.0, 0.0])[1], 0.0);
        close(gpu.map_output_to_local([1.0, 1.0])[0], 1.0);
        close(gpu.map_output_to_local([1.0, 1.0])[1], 1.0);
    }

    #[test]
    fn rotation_is_aspect_correct_in_physical_output_space() {
        let gpu = SpatialTransform {
            fit: FitMode::Fit,
            rotation_deg: 90.0,
            ..SpatialTransform::default()
        }
        .gpu_uniforms(640, 480, 1920, 1080);
        let top_middle = gpu.map_output_to_local([0.78125, 0.5]);
        close(top_middle[0], 0.5);
        close(top_middle[1], 0.0);
        let right_middle = gpu.map_output_to_local([0.5, 7.0 / 6.0]);
        close(right_middle[0], 1.0);
        close(right_middle[1], 0.5);
    }

    #[test]
    fn crop_and_non_finite_values_are_sanitized() {
        let clean = SpatialTransform {
            position: [f32::NAN, f32::INFINITY],
            scale: [f32::NEG_INFINITY, 2.0],
            crop: [0.8, 0.9, 0.8, 0.9],
            ..SpatialTransform::default()
        }
        .sanitized();
        assert_eq!(clean.position, [0.0, 0.0]);
        assert_eq!(clean.scale, [1.0, 2.0]);
        close(clean.crop[0] + clean.crop[2], 1.0 - MIN_CROP_EXTENT);
        close(clean.crop[1] + clean.crop[3], 1.0 - MIN_CROP_EXTENT);
    }

    #[test]
    fn collapsed_scale_is_explicitly_invalid() {
        let gpu = SpatialTransform {
            scale: [0.0, 1.0],
            ..SpatialTransform::default()
        }
        .gpu_uniforms(100, 100, 100, 100);
        assert_eq!(gpu.modes[2], 0);
        assert_eq!(gpu.modes[3], 1);
    }

    #[test]
    fn interpolation_uses_short_arc_and_geometric_scale() {
        let a = SpatialTransform {
            scale: [1.0, 1.0],
            rotation_deg: 170.0,
            ..SpatialTransform::default()
        };
        let b = SpatialTransform {
            scale: [4.0, 4.0],
            rotation_deg: -170.0,
            ..SpatialTransform::default()
        };
        let middle = SpatialTransform::interpolate(a, b, [0.5, 0.5], false);
        close(middle.scale[0], 2.0);
        close(middle.rotation_deg.abs(), 180.0);
    }
}
