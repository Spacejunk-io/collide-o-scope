//! Fixed-resource GPU executor for authored Collision Racks.
//!
//! This module is intentionally independent of the live and offline renderer
//! orchestration. It compiles a route-resolved [`RuntimeVisualRack`] into an
//! immutable sequence, owns exactly two reusable output-sized RGBA16Float
//! surfaces, and creates no GPU object while encoding a warmed frame. Saved
//! positional image routes never enter this layer: image-mask bindings pair a
//! stable [`NodeId`] with the exact [`ResolvedImageTap`] captured in the plan.

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use crate::layers::BlendMode;
#[cfg(test)]
use crate::renderer::blend::blend_rgb;
use crate::spatial::{SpatialGpuUniforms, SpatialTransform};
use crate::visual_rack::{
    node_kind_descriptor, CellularParams, DigitalColorParams, EllipseMask, GrainAlgorithm,
    GrainParams, KeyMode, KeyParams, MatteChannel, NodeBlend, NodeId, NodeKindTag, RectangleMask,
    ResolvedImageTap, RuntimeDisplaceParams, RuntimeImageMatte, RuntimeMaskParams,
    RuntimeRackError, RuntimeVisualNode, RuntimeVisualNodeKind, RuntimeVisualRack, ShiftParams,
    MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK, MAX_NODES_PER_RACK, MAX_SAMPLED_TEXTURES_PER_PASS,
    MAX_TEXTURE_SAMPLES_PER_RACK,
};

pub(crate) const RACK_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[cfg(test)]
const UNIFORM_SLOT_COUNT: usize = MAX_NODES_PER_RACK + 1;
const KIND_PASSTHROUGH: u32 = 0;
const KIND_TRANSFORM: u32 = 1;
const KIND_DIGITAL_COLOR: u32 = 2;
const KIND_KEY: u32 = 3;
const KIND_CELLULAR: u32 = 4;
const KIND_SHIFT: u32 = 5;
const KIND_GRAIN: u32 = 6;
const KIND_RECTANGLE_MASK: u32 = 7;
const KIND_ELLIPSE_MASK: u32 = 8;
const KIND_IMAGE_MASK: u32 = 9;
const KIND_DISPLACE: u32 = 10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RackPassKind {
    Transform(SpatialTransform),
    DigitalColor(DigitalColorParams),
    Key(KeyParams),
    Cellular(CellularParams),
    Shift(ShiftParams),
    Grain(GrainParams),
    RectangleMask(RectangleMask),
    EllipseMask(EllipseMask),
    ImageMask(RuntimeImageMatte),
    Displace(RuntimeDisplaceParams),
}

impl RackPassKind {
    const fn tag(self) -> NodeKindTag {
        match self {
            Self::Transform(_) => NodeKindTag::Transform,
            Self::DigitalColor(_) => NodeKindTag::DigitalColor,
            Self::Key(_) => NodeKindTag::Key,
            Self::Cellular(_) => NodeKindTag::Cellular,
            Self::Shift(_) => NodeKindTag::Shift,
            Self::Grain(_) => NodeKindTag::Grain,
            Self::RectangleMask(_) | Self::EllipseMask(_) | Self::ImageMask(_) => NodeKindTag::Mask,
            Self::Displace(_) => NodeKindTag::Displace,
        }
    }

    pub const fn image_tap(self) -> Option<ResolvedImageTap> {
        match self {
            Self::ImageMask(matte) => Some(matte.tap),
            Self::Displace(displace) => Some(displace.tap),
            _ => None,
        }
    }

    /// Kinds whose authored values make the whole pass a no-op. Only Displace
    /// declares one: zero gain on both axes must delegate to the dry carrier
    /// exactly, without encoding a pass or binding a donor.
    fn is_exact_value_bypass(self) -> bool {
        match self {
            Self::Displace(displace) => displace.is_exact_bypass(),
            _ => false,
        }
    }
}

/// One immutable pass in exact authored order. Disabled and zero-wet nodes
/// remain in the plan so topology/order signatures are observable, but encode
/// skips them without touching a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RackPassDescriptor {
    pub node_id: NodeId,
    pub enabled: bool,
    pub wet: f32,
    pub blend: NodeBlend,
    pub kind: RackPassKind,
}

impl RackPassDescriptor {
    pub fn is_exact_bypass(self) -> bool {
        !self.enabled || self.wet <= 0.0 || self.kind.is_exact_value_bypass()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CollisionRackPlan {
    source_dimensions: [u32; 2],
    output_dimensions: [u32; 2],
    passes: Box<[RackPassDescriptor]>,
    logical_texture_lookups_per_pixel: u32,
    texture_samples_per_pixel: u32,
    max_sampled_textures_in_pass: u32,
    cross_input_taps: u32,
}

impl CollisionRackPlan {
    /// Compile only the route-resolved runtime model. Legacy marker nodes are
    /// owned by the frozen legacy path and are an explicit error here.
    pub fn compile(
        rack: &RuntimeVisualRack,
        source_dimensions: [u32; 2],
        output_dimensions: [u32; 2],
    ) -> Result<Self, RackCompileError> {
        if source_dimensions.contains(&0) || output_dimensions.contains(&0) {
            return Err(RackCompileError::ZeroDimensions {
                source: source_dimensions,
                output: output_dimensions,
            });
        }
        if rack.len() > MAX_NODES_PER_RACK {
            return Err(RackCompileError::TooManyPasses {
                requested: rack.len(),
                limit: MAX_NODES_PER_RACK,
            });
        }
        let declared = rack.resource_budget().map_err(RackCompileError::Rack)?;
        if declared.logical_texture_lookups_per_pixel > MAX_LOGICAL_TEXTURE_LOOKUPS_PER_RACK
            || declared.texture_samples_per_pixel > MAX_TEXTURE_SAMPLES_PER_RACK
            || declared.max_sampled_textures_in_pass > MAX_SAMPLED_TEXTURES_PER_PASS
        {
            return Err(RackCompileError::DeclaredBudgetExceeded {
                samples: declared.texture_samples_per_pixel,
                textures: declared.max_sampled_textures_in_pass,
            });
        }

        let mut ids = BTreeSet::new();
        let mut passes = Vec::with_capacity(rack.len());
        let mut independently_counted_logical_lookups = 0_u32;
        let mut independently_counted_samples = 0_u32;
        let mut independently_counted_textures = 0_u32;
        let mut independently_counted_inputs = 0_u32;
        let mut independently_counted_passes = 0_u32;
        for node in rack.iter() {
            if !ids.insert(node.stable_id) {
                return Err(RackCompileError::DuplicateNodeId(node.stable_id));
            }
            let descriptor = node_kind_descriptor(node.kind.tag()).budget;
            if node.enabled {
                independently_counted_passes = independently_counted_passes
                    .checked_add(u32::from(descriptor.full_frame_passes))
                    .ok_or(RackCompileError::BudgetOverflow)?;
                independently_counted_logical_lookups = independently_counted_logical_lookups
                    .checked_add(u32::from(descriptor.logical_texture_lookups_per_pixel))
                    .ok_or(RackCompileError::BudgetOverflow)?;
                independently_counted_samples = independently_counted_samples
                    .checked_add(u32::from(descriptor.texture_samples_per_pixel))
                    .ok_or(RackCompileError::BudgetOverflow)?;
                independently_counted_textures = independently_counted_textures
                    .max(u32::from(descriptor.sampled_textures_in_pass));
                independently_counted_inputs = independently_counted_inputs
                    .checked_add(u32::from(descriptor.cross_input_taps))
                    .ok_or(RackCompileError::BudgetOverflow)?;
            }
            let pass = compile_pass(*node)?;
            if pass.kind.tag() != node.kind.tag() {
                return Err(RackCompileError::BudgetContractMismatch);
            }
            passes.push(pass);
        }
        if independently_counted_passes != declared.full_frame_passes
            || independently_counted_logical_lookups != declared.logical_texture_lookups_per_pixel
            || independently_counted_samples != declared.texture_samples_per_pixel
            || independently_counted_textures != declared.max_sampled_textures_in_pass
            || independently_counted_inputs != declared.cross_input_taps
        {
            return Err(RackCompileError::BudgetContractMismatch);
        }
        Ok(Self {
            source_dimensions,
            output_dimensions,
            passes: passes.into_boxed_slice(),
            logical_texture_lookups_per_pixel: declared.logical_texture_lookups_per_pixel,
            texture_samples_per_pixel: declared.texture_samples_per_pixel,
            max_sampled_textures_in_pass: declared.max_sampled_textures_in_pass,
            cross_input_taps: declared.cross_input_taps,
        })
    }

    pub fn passes(&self) -> &[RackPassDescriptor] {
        &self.passes
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "compiled-plan dimensions are exposed for rack goldens"
        )
    )]
    pub const fn source_dimensions(&self) -> [u32; 2] {
        self.source_dimensions
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "compiled-plan dimensions are exposed for rack goldens"
        )
    )]
    pub const fn output_dimensions(&self) -> [u32; 2] {
        self.output_dimensions
    }

    #[allow(
        dead_code,
        reason = "declared sample accounting remains part of the rack preflight contract"
    )]
    pub const fn logical_texture_lookups_per_pixel(&self) -> u32 {
        self.logical_texture_lookups_per_pixel
    }

    #[allow(
        dead_code,
        reason = "declared sample accounting remains part of the rack preflight contract"
    )]
    pub const fn texture_samples_per_pixel(&self) -> u32 {
        self.texture_samples_per_pixel
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "compiled texture accounting is exposed for rack goldens"
        )
    )]
    pub const fn max_sampled_textures_in_pass(&self) -> u32 {
        self.max_sampled_textures_in_pass
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "compiled route accounting is exposed for rack goldens"
        )
    )]
    pub const fn cross_input_taps(&self) -> u32 {
        self.cross_input_taps
    }
}

fn compile_pass(node: RuntimeVisualNode) -> Result<RackPassDescriptor, RackCompileError> {
    let kind = match node.kind {
        RuntimeVisualNodeKind::LegacyCanonical | RuntimeVisualNodeKind::LegacyTemporal => {
            return Err(RackCompileError::LegacyNode {
                node_id: node.stable_id,
                tag: node.kind.tag(),
            });
        }
        RuntimeVisualNodeKind::Transform(value) => RackPassKind::Transform(value.sanitized()),
        RuntimeVisualNodeKind::DigitalColor(value) => {
            RackPassKind::DigitalColor(sanitize_digital(value))
        }
        RuntimeVisualNodeKind::Key(value) => RackPassKind::Key(sanitize_key(value)),
        RuntimeVisualNodeKind::Cellular(value) => RackPassKind::Cellular(sanitize_cellular(value)),
        RuntimeVisualNodeKind::Shift(value) => RackPassKind::Shift(sanitize_shift(value)),
        RuntimeVisualNodeKind::Grain(value) => RackPassKind::Grain(sanitize_grain(value)),
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(value)) => {
            RackPassKind::RectangleMask(sanitize_rectangle(value))
        }
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(value)) => {
            RackPassKind::EllipseMask(sanitize_ellipse(value))
        }
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(value)) => {
            RackPassKind::ImageMask(sanitize_runtime_matte(value))
        }
        RuntimeVisualNodeKind::Displace(value) => {
            RackPassKind::Displace(sanitize_runtime_displace(value))
        }
    };
    Ok(RackPassDescriptor {
        node_id: node.stable_id,
        enabled: node.enabled,
        wet: finite_clamp(node.wet, 1.0, 0.0, 1.0),
        blend: node.blend,
        kind,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RackCompileError {
    Rack(RuntimeRackError),
    ZeroDimensions { source: [u32; 2], output: [u32; 2] },
    TooManyPasses { requested: usize, limit: usize },
    DuplicateNodeId(NodeId),
    LegacyNode { node_id: NodeId, tag: NodeKindTag },
    DeclaredBudgetExceeded { samples: u32, textures: u32 },
    BudgetOverflow,
    BudgetContractMismatch,
}

impl fmt::Display for RackCompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rack(error) => write!(formatter, "runtime rack is invalid: {error}"),
            Self::ZeroDimensions { source, output } => write!(
                formatter,
                "rack dimensions must be non-zero (source={}x{}, output={}x{})",
                source[0], source[1], output[0], output[1]
            ),
            Self::TooManyPasses { requested, limit } => {
                write!(formatter, "rack requested {requested} passes; limit is {limit}")
            }
            Self::DuplicateNodeId(id) => {
                write!(formatter, "rack contains duplicate node id {}", id.get())
            }
            Self::LegacyNode { node_id, tag } => write!(
                formatter,
                "{tag:?} node {} belongs to the frozen legacy renderer and cannot run in the advanced rack executor",
                node_id.get()
            ),
            Self::DeclaredBudgetExceeded { samples, textures } => write!(
                formatter,
                "rack declares {samples} samples/pixel and {textures} sampled textures/pass, beyond executor limits"
            ),
            Self::BudgetOverflow => formatter.write_str("rack budget arithmetic overflowed"),
            Self::BudgetContractMismatch => formatter.write_str(
                "rack resource declaration disagrees with the executor's independent pass count",
            ),
        }
    }
}

impl std::error::Error for RackCompileError {}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RackUniforms {
    meta: [u32; 4],
    frame: [f32; 4],
    p0: [f32; 4],
    p1: [f32; 4],
    p2: [f32; 4],
    p3: [f32; 4],
    p4: [f32; 4],
    p5: [f32; 4],
    p6: [f32; 4],
    p7: [f32; 4],
    modes: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<RackUniforms>() == 176);

impl RackUniforms {
    fn passthrough(output: [u32; 2]) -> Self {
        Self {
            meta: [KIND_PASSTHROUGH, 0, 0, 0],
            frame: [1.0, 0.0, output[0] as f32, output[1] as f32],
            p0: [0.0; 4],
            p1: [0.0; 4],
            p2: [0.0; 4],
            p3: [0.0; 4],
            p4: [0.0; 4],
            p5: [0.0; 4],
            p6: [0.0; 4],
            p7: [0.0; 4],
            modes: [0; 4],
        }
    }

    fn for_pass(
        pass: RackPassDescriptor,
        source_dimensions: [u32; 2],
        output_dimensions: [u32; 2],
        time_seconds: f32,
        donor_valid: bool,
    ) -> Self {
        let mut uniforms = Self::passthrough(output_dimensions);
        uniforms.meta[1] = pass.blend.code();
        uniforms.meta[2] = u32::from(donor_valid);
        uniforms.frame[0] = pass.wet;
        uniforms.frame[1] = finite_or(time_seconds, 0.0).max(0.0);
        match pass.kind {
            RackPassKind::Transform(value) => {
                uniforms.meta[0] = KIND_TRANSFORM;
                let spatial = value.gpu_uniforms(
                    source_dimensions[0],
                    source_dimensions[1],
                    output_dimensions[0],
                    output_dimensions[1],
                );
                write_spatial(&mut uniforms, spatial);
            }
            RackPassKind::DigitalColor(value) => {
                uniforms.meta[0] = KIND_DIGITAL_COLOR;
                uniforms.p0 = [
                    value.pixelate_size,
                    value.rgb_split,
                    value.downsample,
                    value.hue_shift,
                ];
                uniforms.p1 = [
                    value.saturation,
                    value.brightness,
                    value.contrast,
                    value.posterize,
                ];
                uniforms.p2 = [value.invert, value.vignette, value.color_drift, 0.0];
            }
            RackPassKind::Key(value) => {
                uniforms.meta[0] = KIND_KEY;
                uniforms.p0 = [
                    key_mode_code(value.mode) as f32,
                    value.threshold,
                    value.softness,
                    f32::from(value.invert),
                ];
                uniforms.p1 = [
                    value.color[0],
                    value.color[1],
                    value.color[2],
                    value.tolerance,
                ];
            }
            RackPassKind::Cellular(value) => {
                uniforms.meta[0] = KIND_CELLULAR;
                uniforms.meta[3] = value.seed;
                uniforms.p0 = [value.amount, value.scale, value.warp, value.speed];
                uniforms.p1 = [
                    value.gap_amount,
                    value.gap_threshold,
                    value.gap_softness,
                    0.0,
                ];
            }
            RackPassKind::Shift(value) => {
                uniforms.meta[0] = KIND_SHIFT;
                uniforms.meta[3] = value.seed;
                uniforms.p0 = [value.amount, value.block_size, value.density, value.speed];
            }
            RackPassKind::Grain(value) => {
                uniforms.meta[0] = KIND_GRAIN;
                uniforms.meta[3] = value.seed;
                uniforms.p0 = [
                    value.intensity,
                    value.size,
                    grain_algorithm_code(value.algorithm) as f32,
                    f32::from(value.color),
                ];
            }
            RackPassKind::RectangleMask(value) => {
                uniforms.meta[0] = KIND_RECTANGLE_MASK;
                uniforms.p0 = [
                    value.center[0],
                    value.center[1],
                    value.size[0],
                    value.size[1],
                ];
                uniforms.p1 = [
                    value.rotation_deg,
                    value.feather,
                    f32::from(value.invert),
                    0.0,
                ];
            }
            RackPassKind::EllipseMask(value) => {
                uniforms.meta[0] = KIND_ELLIPSE_MASK;
                uniforms.p0 = [
                    value.center[0],
                    value.center[1],
                    value.radii[0],
                    value.radii[1],
                ];
                uniforms.p1 = [
                    value.rotation_deg,
                    value.feather,
                    f32::from(value.invert),
                    0.0,
                ];
            }
            RackPassKind::ImageMask(value) => {
                uniforms.meta[0] = KIND_IMAGE_MASK;
                uniforms.p0 = [
                    matte_channel_code(value.channel) as f32,
                    f32::from(value.invert),
                    value.amount,
                    value.threshold,
                ];
                uniforms.p1[0] = value.softness;
            }
            RackPassKind::Displace(value) => {
                uniforms.meta[0] = KIND_DISPLACE;
                uniforms.p0 = [
                    value.amount_x,
                    value.amount_y,
                    value.boundary.code() as f32,
                    0.0,
                ];
            }
        }
        uniforms
    }
}

fn write_spatial(uniforms: &mut RackUniforms, spatial: SpatialGpuUniforms) {
    uniforms.p0 = spatial.inverse_row_0;
    uniforms.p1 = spatial.inverse_row_1;
    uniforms.p2 = spatial.crop;
    uniforms.modes = spatial.modes;
}

#[cfg(test)]
fn node_blend_mode(blend: NodeBlend) -> BlendMode {
    BlendMode::from_u32(blend.code()).expect("NodeBlend and BlendMode use one frozen code table")
}

/// CPU reference for the rack's alpha/blend/wet contract. Inputs and output
/// are straight-alpha; interpolation itself occurs in premultiplied space.
#[cfg(test)]
fn apply_node_law(dry: [f32; 4], processed: [f32; 4], wet: f32, blend: NodeBlend) -> [f32; 4] {
    let wet = finite_clamp(wet, 1.0, 0.0, 1.0);
    if wet <= 0.0 {
        return dry;
    }
    let dry_alpha = finite_clamp(dry[3], 0.0, 0.0, 1.0);
    let processed_alpha = finite_clamp(processed[3], 0.0, 0.0, 1.0);
    let result = if blend == NodeBlend::AlphaCut {
        [dry[0], dry[1], dry[2], dry_alpha * (1.0 - processed_alpha)]
    } else {
        let rgb = blend_rgb(
            node_blend_mode(blend),
            dry[..3].try_into().expect("three channels"),
            processed[..3].try_into().expect("three channels"),
        );
        [rgb[0], rgb[1], rgb[2], processed_alpha]
    };
    if wet >= 1.0 {
        return result;
    }
    let alpha = dry_alpha * (1.0 - wet) + result[3] * wet;
    if alpha <= 1.0e-6 {
        return [0.0; 4];
    }
    let rgb: [f32; 3] = std::array::from_fn(|channel| {
        (dry[channel] * dry_alpha * (1.0 - wet) + result[channel] * result[3] * wet) / alpha
    });
    [rgb[0], rgb[1], rgb[2], alpha]
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RackAllocationSnapshot {
    pub textures: u64,
    pub buffers: u64,
    pub bind_groups: u64,
    pub pipelines: u64,
}

impl RackAllocationSnapshot {
    pub const fn total(self) -> u64 {
        self.textures + self.buffers + self.bind_groups + self.pipelines
    }
}

#[derive(Default)]
struct RackAllocationCounters {
    textures: AtomicU64,
    buffers: AtomicU64,
    bind_groups: AtomicU64,
    pipelines: AtomicU64,
}

impl RackAllocationCounters {
    fn snapshot(&self) -> RackAllocationSnapshot {
        RackAllocationSnapshot {
            textures: self.textures.load(Ordering::Relaxed),
            buffers: self.buffers.load(Ordering::Relaxed),
            bind_groups: self.bind_groups.load(Ordering::Relaxed),
            pipelines: self.pipelines.load(Ordering::Relaxed),
        }
    }
}

struct RackSurface {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// Source texture and dimensions prepared outside frame encode. Creating this
/// binding is counted; reusing it for arbitrarily many frames is allocation
/// free. Its texture view may use any filterable float texture format.
pub(crate) struct RackSourceBinding {
    dimensions: [u32; 2],
    bind_group: wgpu::BindGroup,
}

impl RackSourceBinding {
    #[allow(
        dead_code,
        reason = "prepared binding dimensions are exposed for GPU goldens"
    )]
    pub const fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }
}

pub(crate) struct RackImageInput<'a> {
    pub node_id: NodeId,
    pub tap: ResolvedImageTap,
    pub view: Option<&'a wgpu::TextureView>,
}

struct RackImageBinding {
    node_id: NodeId,
    tap: ResolvedImageTap,
    groups: Option<[wgpu::BindGroup; 2]>,
    valid: bool,
}

/// Bounded route-resolved image-mask bindings, prepared outside encode and
/// searched without allocation by stable node ID plus resolved route value.
pub(crate) struct RackImageBindings {
    entries: Box<[RackImageBinding]>,
}

impl RackImageBindings {
    pub fn empty() -> Self {
        Self {
            entries: Box::new([]),
        }
    }

    #[allow(dead_code, reason = "binding cardinality is exposed for GPU goldens")]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Update only the frame-local readiness bit of an already prepared
    /// donor. The texture binding remains immutable, so cold N-1 histories
    /// can become valid without allocating a replacement bind group.
    pub fn set_valid(&mut self, node_id: NodeId, tap: ResolvedImageTap, valid: bool) -> bool {
        let Ok(index) = self
            .entries
            .binary_search_by_key(&node_id, |entry| entry.node_id)
        else {
            return false;
        };
        let entry = &mut self.entries[index];
        if entry.tap != tap {
            return false;
        }
        entry.valid = valid && entry.groups.is_some();
        true
    }

    fn find(&self, node_id: NodeId, tap: ResolvedImageTap) -> Option<&RackImageBinding> {
        let index = self
            .entries
            .binary_search_by_key(&node_id, |entry| entry.node_id)
            .ok()?;
        let entry = &self.entries[index];
        (entry.tap == tap).then_some(entry)
    }
}

impl Default for RackImageBindings {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RackEncodeReport {
    pub surface_index: usize,
    pub executed_passes: u8,
    pub missing_image_nodes: [Option<NodeId>; MAX_NODES_PER_RACK],
    pub missing_image_count: u8,
}

impl RackEncodeReport {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "missing-route iteration is exposed for GPU goldens"
        )
    )]
    pub fn missing_image_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.missing_image_nodes[..usize::from(self.missing_image_count)]
            .iter()
            .flatten()
            .copied()
    }
}

pub(crate) struct RackOutput<'a> {
    pub texture: &'a wgpu::Texture,
    pub view: &'a wgpu::TextureView,
    #[allow(
        dead_code,
        reason = "output dimensions are exposed for GPU adapter verification"
    )]
    pub dimensions: [u32; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RackGpuError {
    ZeroDimensions([u32; 2]),
    DimensionsExceedDevice {
        requested: [u32; 2],
        limit: u32,
    },
    BufferSizeOverflow,
    ResourceCreation {
        context: &'static str,
        kind: &'static str,
        message: String,
    },
    TooManyImageBindings {
        requested: usize,
        limit: usize,
    },
    DuplicateImageBinding(NodeId),
    PlanOutputMismatch {
        plan: [u32; 2],
        executor: [u32; 2],
    },
    SourceDimensionMismatch {
        plan: [u32; 2],
        binding: [u32; 2],
    },
    UniformSlotRange {
        base: usize,
        required: usize,
        capacity: usize,
    },
}

impl fmt::Display for RackGpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions(dimensions) => write!(
                formatter,
                "rack executor dimensions must be non-zero ({}x{})",
                dimensions[0], dimensions[1]
            ),
            Self::DimensionsExceedDevice { requested, limit } => write!(
                formatter,
                "rack executor {}x{} exceeds the device 2D texture limit of {limit}",
                requested[0], requested[1]
            ),
            Self::BufferSizeOverflow => formatter.write_str("rack uniform buffer size overflowed"),
            Self::ResourceCreation {
                context,
                kind,
                message,
            } => write!(formatter, "{context} failed ({kind}): {message}"),
            Self::TooManyImageBindings { requested, limit } => write!(
                formatter,
                "rack image binding set contains {requested} entries; limit is {limit}"
            ),
            Self::DuplicateImageBinding(id) => {
                write!(formatter, "duplicate image binding for node {}", id.get())
            }
            Self::PlanOutputMismatch { plan, executor } => write!(
                formatter,
                "rack plan output {}x{} does not match executor {}x{}",
                plan[0], plan[1], executor[0], executor[1]
            ),
            Self::SourceDimensionMismatch { plan, binding } => write!(
                formatter,
                "rack source binding {}x{} does not match plan source {}x{}",
                binding[0], binding[1], plan[0], plan[1]
            ),
            Self::UniformSlotRange {
                base,
                required,
                capacity,
            } => write!(
                formatter,
                "rack uniform slots {base}..{} exceed warmed capacity {capacity}",
                base.saturating_add(*required)
            ),
        }
    }
}

impl std::error::Error for RackGpuError {}

/// Fixed-pipeline, fixed-surface executor. Construction is transactional: all
/// handle-returning resource creation is enclosed by validation, internal and
/// OOM scopes, and no handle escapes if a scope reports an error.
pub(crate) struct CollisionRackExecutor {
    dimensions: [u32; 2],
    surfaces: [RackSurface; 2],
    pipeline: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_group: wgpu::BindGroup,
    uniform_stride: u64,
    uniform_slot_capacity: usize,
    missing_groups: [wgpu::BindGroup; 2],
    linear_sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,
    _zero_texture: wgpu::Texture,
    zero_view: wgpu::TextureView,
    allocations: RackAllocationCounters,
}

impl CollisionRackExecutor {
    #[cfg(test)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dimensions: [u32; 2],
    ) -> Result<Self, RackGpuError> {
        Self::new_with_uniform_slots(device, queue, dimensions, UNIFORM_SLOT_COUNT)
    }

    /// Create one fixed executor whose uniform arena can retain multiple rack
    /// invocations in a single unsubmitted command buffer. Callers assign
    /// disjoint bases with [`Self::encode_at`]; the standalone path retains
    /// the original one-rack capacity through [`Self::new`].
    pub fn new_with_uniform_slots(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dimensions: [u32; 2],
        uniform_slot_capacity: usize,
    ) -> Result<Self, RackGpuError> {
        if dimensions.contains(&0) {
            return Err(RackGpuError::ZeroDimensions(dimensions));
        }
        let dimension_limit = device.limits().max_texture_dimension_2d;
        if dimensions[0] > dimension_limit || dimensions[1] > dimension_limit {
            return Err(RackGpuError::DimensionsExceedDevice {
                requested: dimensions,
                limit: dimension_limit,
            });
        }
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment.max(1));
        let uniform_size = std::mem::size_of::<RackUniforms>() as u64;
        let uniform_stride = uniform_size
            .checked_add(alignment - 1)
            .ok_or(RackGpuError::BufferSizeOverflow)?
            / alignment
            * alignment;
        let uniform_slot_capacity = uniform_slot_capacity.max(1);
        let buffer_size = uniform_stride
            .checked_mul(
                u64::try_from(uniform_slot_capacity)
                    .map_err(|_| RackGpuError::BufferSizeOverflow)?,
            )
            .ok_or(RackGpuError::BufferSizeOverflow)?;

        let allocations = RackAllocationCounters::default();
        let resources = create_checked(device, "Collision Rack initialization", || {
            let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Collision Rack linear sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            });
            let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Collision Rack nearest sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            });
            let texture_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Collision Rack texture BGL"),
                    entries: &[
                        sampled_texture_entry(0),
                        sampled_texture_entry(1),
                        sampler_entry(2),
                        sampler_entry(3),
                    ],
                });
            let uniform_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Collision Rack dynamic uniform BGL"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: NonZeroU64::new(uniform_size),
                        },
                        count: None,
                    }],
                });
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Collision Rack warmed uniforms"),
                size: buffer_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Collision Rack warmed uniform BG"),
                layout: &uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(uniform_size),
                    }),
                }],
            });
            let vertex = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Collision Rack vertex shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/rack_fullscreen.wgsl").into(),
                ),
            });
            let fragment_source = format!(
                "{}\n{}",
                include_str!("../shaders/blend.wgsl"),
                include_str!("../shaders/rack_node.wgsl")
            );
            let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Collision Rack node shader"),
                source: wgpu::ShaderSource::Wgsl(fragment_source.into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Collision Rack pipeline layout"),
                bind_group_layouts: &[Some(&texture_layout), Some(&uniform_layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Collision Rack fixed node pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vertex,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: RACK_TEXTURE_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            let surfaces = std::array::from_fn(|index| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(if index == 0 {
                        "Collision Rack ping RGBA16Float"
                    } else {
                        "Collision Rack pong RGBA16Float"
                    }),
                    size: wgpu::Extent3d {
                        width: dimensions[0],
                        height: dimensions[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: RACK_TEXTURE_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                RackSurface { texture, view }
            });
            let zero_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Collision Rack defined-zero donor"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: RACK_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let zero_view = zero_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let missing_groups = std::array::from_fn(|index| {
                create_texture_group(
                    device,
                    &texture_layout,
                    &surfaces[index].view,
                    &zero_view,
                    &linear_sampler,
                    &nearest_sampler,
                    "Collision Rack missing-donor BG",
                )
            });
            RackResources {
                surfaces,
                pipeline,
                texture_layout,
                uniform_buffer,
                uniform_group,
                missing_groups,
                linear_sampler,
                nearest_sampler,
                zero_texture,
                zero_view,
            }
        })?;

        // Explicitly initialize the diagnostic donor to transparent black.
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &resources.zero_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0; 8],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        allocations.textures.store(3, Ordering::Relaxed);
        allocations.buffers.store(1, Ordering::Relaxed);
        allocations.bind_groups.store(3, Ordering::Relaxed);
        allocations.pipelines.store(1, Ordering::Relaxed);
        Ok(Self {
            dimensions,
            surfaces: resources.surfaces,
            pipeline: resources.pipeline,
            texture_layout: resources.texture_layout,
            uniform_buffer: resources.uniform_buffer,
            uniform_group: resources.uniform_group,
            uniform_stride,
            uniform_slot_capacity,
            missing_groups: resources.missing_groups,
            linear_sampler: resources.linear_sampler,
            nearest_sampler: resources.nearest_sampler,
            _zero_texture: resources.zero_texture,
            zero_view: resources.zero_view,
            allocations,
        })
    }

    #[allow(
        dead_code,
        reason = "executor-dimension inspection remains a compatibility seam for GPU adapters"
    )]
    pub const fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }

    pub fn allocation_snapshot(&self) -> RackAllocationSnapshot {
        self.allocations.snapshot()
    }

    pub fn prepare_source(
        &self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
        dimensions: [u32; 2],
    ) -> Result<RackSourceBinding, RackGpuError> {
        if dimensions.contains(&0) {
            return Err(RackGpuError::ZeroDimensions(dimensions));
        }
        let bind_group = create_checked(device, "Collision Rack source binding", || {
            create_texture_group(
                device,
                &self.texture_layout,
                view,
                &self.zero_view,
                &self.linear_sampler,
                &self.nearest_sampler,
                "Collision Rack prepared source BG",
            )
        })?;
        self.allocations.bind_groups.fetch_add(1, Ordering::Relaxed);
        Ok(RackSourceBinding {
            dimensions,
            bind_group,
        })
    }

    pub fn prepare_image_bindings(
        &self,
        device: &wgpu::Device,
        inputs: &[RackImageInput<'_>],
    ) -> Result<RackImageBindings, RackGpuError> {
        if inputs.len() > MAX_NODES_PER_RACK {
            return Err(RackGpuError::TooManyImageBindings {
                requested: inputs.len(),
                limit: MAX_NODES_PER_RACK,
            });
        }
        let mut sorted = inputs.iter().collect::<Vec<_>>();
        sorted.sort_unstable_by_key(|input| input.node_id);
        if let Some(duplicate) = sorted
            .windows(2)
            .find(|pair| pair[0].node_id == pair[1].node_id)
            .map(|pair| pair[0].node_id)
        {
            return Err(RackGpuError::DuplicateImageBinding(duplicate));
        }
        let entries = create_checked(device, "Collision Rack image bindings", || {
            sorted
                .into_iter()
                .map(|input| {
                    let groups = input.view.map(|view| {
                        std::array::from_fn(|index| {
                            create_texture_group(
                                device,
                                &self.texture_layout,
                                &self.surfaces[index].view,
                                view,
                                &self.linear_sampler,
                                &self.nearest_sampler,
                                "Collision Rack routed image-mask BG",
                            )
                        })
                    });
                    RackImageBinding {
                        node_id: input.node_id,
                        tap: input.tap,
                        groups,
                        valid: input.view.is_some(),
                    }
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })?;
        let created_groups = entries
            .iter()
            .filter(|entry| entry.groups.is_some())
            .count()
            * 2;
        self.allocations
            .bind_groups
            .fetch_add(created_groups as u64, Ordering::Relaxed);
        Ok(RackImageBindings { entries })
    }

    /// Encode seed conversion plus every active node in exact authored order.
    /// Uniform writes update preallocated dynamic slots; render passes only
    /// reference warmed pipelines, bind groups, buffers and surfaces.
    #[cfg(test)]
    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &CollisionRackPlan,
        source: &RackSourceBinding,
        images: &RackImageBindings,
        time_seconds: f32,
    ) -> Result<RackEncodeReport, RackGpuError> {
        self.encode_at(queue, encoder, plan, source, images, 0, time_seconds)
    }

    /// Encode at a caller-reserved uniform base. Disjoint bases are required
    /// when several rack segments share this executor before one queue submit;
    /// queue writes otherwise alias and later segments change earlier passes.
    #[allow(
        clippy::too_many_arguments,
        reason = "the allocation-free encode boundary keeps every borrowed GPU input explicit"
    )]
    pub fn encode_at(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &CollisionRackPlan,
        source: &RackSourceBinding,
        images: &RackImageBindings,
        uniform_base_slot: usize,
        time_seconds: f32,
    ) -> Result<RackEncodeReport, RackGpuError> {
        if plan.output_dimensions != self.dimensions {
            return Err(RackGpuError::PlanOutputMismatch {
                plan: plan.output_dimensions,
                executor: self.dimensions,
            });
        }
        if plan.source_dimensions != source.dimensions {
            return Err(RackGpuError::SourceDimensionMismatch {
                plan: plan.source_dimensions,
                binding: source.dimensions,
            });
        }
        let required_slots = plan
            .passes
            .iter()
            .filter(|pass| !pass.is_exact_bypass())
            .count()
            .checked_add(1)
            .ok_or(RackGpuError::BufferSizeOverflow)?;
        if uniform_base_slot
            .checked_add(required_slots)
            .is_none_or(|end| end > self.uniform_slot_capacity)
        {
            return Err(RackGpuError::UniformSlotRange {
                base: uniform_base_slot,
                required: required_slots,
                capacity: self.uniform_slot_capacity,
            });
        }
        let allocations_before = self.allocation_snapshot();
        let seed_offset = self
            .uniform_stride
            .checked_mul(
                u64::try_from(uniform_base_slot).map_err(|_| RackGpuError::BufferSizeOverflow)?,
            )
            .ok_or(RackGpuError::BufferSizeOverflow)?;
        queue.write_buffer(
            &self.uniform_buffer,
            seed_offset,
            bytemuck::bytes_of(&RackUniforms::passthrough(self.dimensions)),
        );
        self.encode_pass(
            encoder,
            "Collision Rack source seed",
            &source.bind_group,
            u32::try_from(seed_offset).map_err(|_| RackGpuError::BufferSizeOverflow)?,
            &self.surfaces[0].view,
        );

        let mut current_surface = 0_usize;
        let mut executed = 0_u8;
        let mut missing_image_nodes = [None; MAX_NODES_PER_RACK];
        let mut missing_count = 0_u8;
        for pass in plan.passes.iter().copied() {
            if pass.is_exact_bypass() {
                continue;
            }
            let target_surface = current_surface ^ 1;
            let mut donor_valid = false;
            let texture_group = match pass.kind.image_tap() {
                Some(tap) => match images.find(pass.node_id, tap) {
                    Some(binding) => match binding.groups.as_ref() {
                        Some(groups) if binding.valid => {
                            donor_valid = true;
                            &groups[current_surface]
                        }
                        Some(_) | None => {
                            missing_image_nodes[usize::from(missing_count)] = Some(pass.node_id);
                            missing_count += 1;
                            &self.missing_groups[current_surface]
                        }
                    },
                    None => {
                        missing_image_nodes[usize::from(missing_count)] = Some(pass.node_id);
                        missing_count += 1;
                        &self.missing_groups[current_surface]
                    }
                },
                None => &self.missing_groups[current_surface],
            };
            let uniform = RackUniforms::for_pass(
                pass,
                plan.source_dimensions,
                plan.output_dimensions,
                time_seconds,
                donor_valid,
            );
            let slot = uniform_base_slot
                .checked_add(usize::from(executed))
                .and_then(|slot| slot.checked_add(1))
                .ok_or(RackGpuError::BufferSizeOverflow)?;
            let offset = self
                .uniform_stride
                .checked_mul(u64::try_from(slot).map_err(|_| RackGpuError::BufferSizeOverflow)?)
                .ok_or(RackGpuError::BufferSizeOverflow)?;
            queue.write_buffer(&self.uniform_buffer, offset, bytemuck::bytes_of(&uniform));
            self.encode_pass(
                encoder,
                "Collision Rack authored node",
                texture_group,
                u32::try_from(offset).map_err(|_| RackGpuError::BufferSizeOverflow)?,
                &self.surfaces[target_surface].view,
            );
            executed += 1;
            current_surface = target_surface;
        }
        debug_assert_eq!(allocations_before, self.allocation_snapshot());
        Ok(RackEncodeReport {
            surface_index: current_surface,
            executed_passes: executed,
            missing_image_nodes,
            missing_image_count: missing_count,
        })
    }

    pub fn output(&self, report: RackEncodeReport) -> RackOutput<'_> {
        let surface = &self.surfaces[report.surface_index];
        RackOutput {
            texture: &surface.texture,
            view: &surface.view,
            dimensions: self.dimensions,
        }
    }

    /// Expose the two executor-owned RGBA16F surfaces as host scratch only at
    /// explicit orchestration boundaries. Callers must not retain either view
    /// across a subsequent rack encode, which deterministically overwrites
    /// both sides of the ping-pong pair.
    pub fn surface(&self, index: usize) -> Option<RackOutput<'_>> {
        let surface = self.surfaces.get(index)?;
        Some(RackOutput {
            texture: &surface.texture,
            view: &surface.view,
            dimensions: self.dimensions,
        })
    }

    fn encode_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &'static str,
        texture_group: &wgpu::BindGroup,
        uniform_offset: u32,
        target: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, texture_group, &[]);
        pass.set_bind_group(1, &self.uniform_group, &[uniform_offset]);
        pass.draw(0..3, 0..1);
    }
}

struct RackResources {
    surfaces: [RackSurface; 2],
    pipeline: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_group: wgpu::BindGroup,
    missing_groups: [wgpu::BindGroup; 2],
    linear_sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,
    zero_texture: wgpu::Texture,
    zero_view: wgpu::TextureView,
}

fn sampled_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn create_texture_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &wgpu::TextureView,
    donor: &wgpu::TextureView,
    linear_sampler: &wgpu::Sampler,
    nearest_sampler: &wgpu::Sampler,
    label: &'static str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(donor),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(linear_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(nearest_sampler),
            },
        ],
    })
}

fn create_checked<T>(
    device: &wgpu::Device,
    context: &'static str,
    create: impl FnOnce() -> T,
) -> Result<T, RackGpuError> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let value = create();
    let errors = [
        ("out of memory", pollster::block_on(out_of_memory.pop())),
        ("internal/backend", pollster::block_on(internal.pop())),
        ("validation", pollster::block_on(validation.pop())),
    ];
    if let Some((kind, error)) = errors
        .into_iter()
        .find_map(|(kind, error)| error.map(|error| (kind, error)))
    {
        return Err(RackGpuError::ResourceCreation {
            context,
            kind,
            message: error.to_string(),
        });
    }
    Ok(value)
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn finite_clamp(value: f32, fallback: f32, minimum: f32, maximum: f32) -> f32 {
    finite_or(value, fallback).clamp(minimum, maximum)
}

fn wrap_degrees(value: f32) -> f32 {
    let value = finite_or(value, 0.0);
    let wrapped = (value + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped == -180.0 && value.is_sign_positive() {
        180.0
    } else {
        wrapped
    }
}

fn sanitize_digital(value: DigitalColorParams) -> DigitalColorParams {
    DigitalColorParams {
        pixelate_size: finite_clamp(value.pixelate_size, 1.0, 1.0, 32.0),
        rgb_split: finite_clamp(value.rgb_split, 0.0, 0.0, 30.0),
        downsample: finite_clamp(value.downsample, 1.0, 0.05, 1.0),
        hue_shift: wrap_degrees(value.hue_shift),
        saturation: finite_clamp(value.saturation, 0.0, -1.0, 1.0),
        brightness: finite_clamp(value.brightness, 0.0, -1.0, 1.0),
        contrast: finite_clamp(value.contrast, 0.0, -1.0, 1.0),
        posterize: finite_clamp(value.posterize, 0.0, 0.0, 16.0),
        invert: finite_clamp(value.invert, 0.0, 0.0, 1.0),
        vignette: finite_clamp(value.vignette, 0.0, 0.0, 1.5),
        color_drift: finite_clamp(value.color_drift, 0.0, 0.0, 0.02),
    }
}

fn sanitize_key(value: KeyParams) -> KeyParams {
    KeyParams {
        mode: value.mode,
        threshold: finite_clamp(value.threshold, 0.5, 0.0, 1.0),
        softness: finite_clamp(value.softness, 0.1, 0.0, 0.5),
        color: value
            .color
            .map(|channel| finite_clamp(channel, 0.0, 0.0, 1.0)),
        tolerance: finite_clamp(value.tolerance, 0.15, 0.0, 1.0),
        invert: value.invert,
    }
}

fn sanitize_cellular(value: CellularParams) -> CellularParams {
    CellularParams {
        amount: finite_clamp(value.amount, 0.0, 0.0, 1.0),
        scale: finite_clamp(value.scale, 10.0, 2.0, 32.0),
        warp: finite_clamp(value.warp, 0.35, 0.0, 1.0),
        speed: finite_clamp(value.speed, 0.25, 0.0, 2.0),
        gap_amount: finite_clamp(value.gap_amount, 0.0, 0.0, 1.0),
        gap_threshold: finite_clamp(value.gap_threshold, 0.65, 0.0, 1.0),
        gap_softness: finite_clamp(value.gap_softness, 0.08, 0.0, 0.5),
        seed: value.seed,
    }
}

fn sanitize_shift(value: ShiftParams) -> ShiftParams {
    ShiftParams {
        amount: finite_clamp(value.amount, 0.0, 0.0, 1.0),
        block_size: finite_clamp(value.block_size, 8.0, 2.0, 256.0),
        density: finite_clamp(value.density, 0.5, 0.0, 1.0),
        speed: finite_clamp(value.speed, 3.0, 0.0, 20.0),
        seed: value.seed,
    }
}

fn sanitize_grain(value: GrainParams) -> GrainParams {
    GrainParams {
        intensity: finite_clamp(value.intensity, 0.0, 0.0, 0.3),
        size: finite_clamp(value.size, 1.0, 1.0, 4.0),
        algorithm: value.algorithm,
        color: value.color,
        seed: value.seed,
    }
}

fn sanitize_rectangle(value: RectangleMask) -> RectangleMask {
    RectangleMask {
        center: value
            .center
            .map(|component| finite_clamp(component, 0.5, -2.0, 3.0)),
        size: value
            .size
            .map(|component| finite_clamp(component, 1.0, 0.0, 4.0)),
        rotation_deg: wrap_degrees(value.rotation_deg),
        feather: finite_clamp(value.feather, 0.0, 0.0, 1.0),
        invert: value.invert,
    }
}

fn sanitize_ellipse(value: EllipseMask) -> EllipseMask {
    EllipseMask {
        center: value
            .center
            .map(|component| finite_clamp(component, 0.5, -2.0, 3.0)),
        radii: value
            .radii
            .map(|component| finite_clamp(component, 0.5, 0.0, 2.0)),
        rotation_deg: wrap_degrees(value.rotation_deg),
        feather: finite_clamp(value.feather, 0.0, 0.0, 1.0),
        invert: value.invert,
    }
}

fn sanitize_runtime_matte(value: RuntimeImageMatte) -> RuntimeImageMatte {
    RuntimeImageMatte {
        tap: value.tap,
        channel: value.channel,
        invert: value.invert,
        amount: finite_clamp(value.amount, 1.0, 0.0, 1.0),
        threshold: finite_clamp(value.threshold, 0.5, 0.0, 1.0),
        softness: finite_clamp(value.softness, 0.1, 0.0, 0.5),
    }
}

fn sanitize_runtime_displace(value: RuntimeDisplaceParams) -> RuntimeDisplaceParams {
    RuntimeDisplaceParams {
        tap: value.tap,
        amount_x: finite_clamp(value.amount_x, 0.0, -1.0, 1.0),
        amount_y: finite_clamp(value.amount_y, 0.0, -1.0, 1.0),
        boundary: value.boundary,
    }
}

const fn key_mode_code(mode: KeyMode) -> u32 {
    match mode {
        KeyMode::KeepBright => 0,
        KeyMode::KeepDark => 1,
        KeyMode::RemoveColor => 2,
        KeyMode::KeepColor => 3,
    }
}

const fn grain_algorithm_code(algorithm: GrainAlgorithm) -> u32 {
    match algorithm {
        GrainAlgorithm::Gaussian => 0,
        GrainAlgorithm::Perlin => 1,
        GrainAlgorithm::SaltPepper => 2,
        GrainAlgorithm::Blue => 3,
    }
}

const fn matte_channel_code(channel: MatteChannel) -> u32 {
    match channel {
        MatteChannel::Alpha => 0,
        MatteChannel::Luma => 1,
        MatteChannel::Red => 2,
        MatteChannel::Green => 3,
        MatteChannel::Blue => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visual_rack::{
        DisplaceBoundary, EdgeTiming, LegacyRackScope, ResolvedImageSource, RuntimeVisualNodeKind,
        PREMULTIPLIED_BILINEAR_TEXTURE_OPS,
    };

    fn node_id(value: u64) -> NodeId {
        NodeId::new(value).unwrap()
    }

    /// Straight-alpha reference image with the exact filtering law used by
    /// `rack_node.wgsl`: clamped `textureLoad` corners, covered in
    /// premultiplied space, bilinearly mixed there.
    struct RefImage {
        dimensions: [u32; 2],
        pixels: Vec<[f32; 4]>,
    }

    impl RefImage {
        fn new(dimensions: [u32; 2], pixels: Vec<[f32; 4]>) -> Self {
            assert_eq!(
                pixels.len(),
                dimensions[0] as usize * dimensions[1] as usize
            );
            Self { dimensions, pixels }
        }

        fn uniform(dimensions: [u32; 2], pixel: [f32; 4]) -> Self {
            let count = dimensions[0] as usize * dimensions[1] as usize;
            Self::new(dimensions, vec![pixel; count])
        }

        fn load(&self, x: i32, y: i32) -> [f32; 4] {
            let x = x.clamp(0, self.dimensions[0] as i32 - 1) as usize;
            let y = y.clamp(0, self.dimensions[1] as i32 - 1) as usize;
            self.pixels[y * self.dimensions[0] as usize + x]
        }

        fn premultiplied_bilinear(&self, uv: [f32; 2]) -> [f32; 4] {
            let coordinate = [
                uv[0] * self.dimensions[0] as f32 - 0.5,
                uv[1] * self.dimensions[1] as f32 - 0.5,
            ];
            let base = [coordinate[0].floor(), coordinate[1].floor()];
            let fraction = [coordinate[0] - base[0], coordinate[1] - base[1]];
            let (bx, by) = (base[0] as i32, base[1] as i32);
            let cover = |pixel: [f32; 4]| {
                let alpha = pixel[3].clamp(0.0, 1.0);
                [pixel[0] * alpha, pixel[1] * alpha, pixel[2] * alpha, alpha]
            };
            let c00 = cover(self.load(bx, by));
            let c10 = cover(self.load(bx + 1, by));
            let c01 = cover(self.load(bx, by + 1));
            let c11 = cover(self.load(bx + 1, by + 1));
            std::array::from_fn(|channel| {
                let top = c00[channel] + (c10[channel] - c00[channel]) * fraction[0];
                let bottom = c01[channel] + (c11[channel] - c01[channel]) * fraction[0];
                top + (bottom - top) * fraction[1]
            })
        }

        fn straight_bilinear(&self, uv: [f32; 2]) -> [f32; 4] {
            let premultiplied = self.premultiplied_bilinear(uv);
            let alpha = premultiplied[3].clamp(0.0, 1.0);
            if alpha <= 0.000_001 {
                return [0.0; 4];
            }
            [
                premultiplied[0] / alpha,
                premultiplied[1] / alpha,
                premultiplied[2] / alpha,
                alpha,
            ]
        }
    }

    fn fract(value: f32) -> f32 {
        value - value.floor()
    }

    /// Alpha-covered donor decode. Neutral encoding is `RG = 0.5` at full
    /// coverage; premultiplied RG against filtered alpha makes a transparent
    /// or missing donor exactly zero whatever its hidden RGB holds.
    fn displace_vector(donor: &RefImage, uv: [f32; 2], donor_valid: bool) -> [f32; 2] {
        if !donor_valid {
            return [0.0, 0.0];
        }
        let sample = donor.premultiplied_bilinear(uv);
        [
            (sample[0] - 0.5 * sample[3]) * 2.0,
            (sample[1] - 0.5 * sample[3]) * 2.0,
        ]
    }

    fn displace_boundary(uv: [f32; 2], boundary: DisplaceBoundary) -> ([f32; 2], bool) {
        let clamped = [uv[0].clamp(0.0, 1.0), uv[1].clamp(0.0, 1.0)];
        match boundary {
            DisplaceBoundary::Mirror => (
                [
                    1.0 - (fract(uv[0] * 0.5) * 2.0 - 1.0).abs(),
                    1.0 - (fract(uv[1] * 0.5) * 2.0 - 1.0).abs(),
                ],
                true,
            ),
            DisplaceBoundary::Wrap => ([fract(uv[0]), fract(uv[1])], true),
            DisplaceBoundary::Hold => (clamped, true),
            DisplaceBoundary::Transparent => {
                let inside = uv[0] >= 0.0 && uv[0] <= 1.0 && uv[1] >= 0.0 && uv[1] <= 1.0;
                (clamped, inside)
            }
        }
    }

    /// CPU reference for the whole Displace node, mirroring `displace_node`.
    fn displace_reference(
        carrier: &RefImage,
        donor: &RefImage,
        uv: [f32; 2],
        params: RuntimeDisplaceParams,
        donor_valid: bool,
    ) -> [f32; 4] {
        let vector = displace_vector(donor, uv, donor_valid);
        let displaced = [
            uv[0] + vector[0] * params.amount_x,
            uv[1] + vector[1] * params.amount_y,
        ];
        let (mapped, covered) = displace_boundary(displaced, params.boundary);
        if !covered {
            return [0.0; 4];
        }
        carrier.straight_bilinear(mapped)
    }

    fn displace(amount_x: f32, amount_y: f32, boundary: DisplaceBoundary) -> RuntimeDisplaceParams {
        RuntimeDisplaceParams {
            amount_x,
            amount_y,
            boundary,
            ..RuntimeDisplaceParams::default()
        }
    }

    fn route(source: ResolvedImageSource) -> ResolvedImageTap {
        ResolvedImageTap {
            source,
            timing: EdgeTiming::CurrentFrame,
        }
    }

    fn plan(rack: &RuntimeVisualRack, dimensions: [u32; 2]) -> CollisionRackPlan {
        CollisionRackPlan::compile(rack, dimensions, dimensions).unwrap()
    }

    #[test]
    fn compile_preserves_authored_order_and_rejects_both_legacy_markers() {
        let mut rack = RuntimeVisualRack::empty();
        let transform = rack
            .push(RuntimeVisualNodeKind::Transform(SpatialTransform::default()))
            .unwrap();
        let key = rack
            .push(RuntimeVisualNodeKind::Key(KeyParams::default()))
            .unwrap();
        let grain = rack
            .push(RuntimeVisualNodeKind::Grain(GrainParams::default()))
            .unwrap();
        rack.move_node(grain, 0, LegacyRackScope::Group).unwrap();

        let compiled = plan(&rack, [64, 48]);
        assert_eq!(
            compiled
                .passes()
                .iter()
                .map(|pass| pass.node_id)
                .collect::<Vec<_>>(),
            vec![grain, transform, key]
        );
        assert_eq!(compiled.output_dimensions(), [64, 48]);
        assert_eq!(compiled.source_dimensions(), [64, 48]);

        for scope in [LegacyRackScope::Layer, LegacyRackScope::Master] {
            let legacy = RuntimeVisualRack::synthetic_legacy(scope);
            let error = CollisionRackPlan::compile(&legacy, [1, 1], [1, 1]).unwrap_err();
            assert!(matches!(error, RackCompileError::LegacyNode { .. }));
        }
    }

    #[test]
    fn node_wet_is_premultiplied_and_zero_wet_is_bit_exact() {
        let dry = [0.8, 0.2, 0.4, 0.75];
        let cut = [0.8, 0.2, 0.4, 0.0];
        assert_eq!(apply_node_law(dry, cut, 0.0, NodeBlend::Screen), dry);

        let half = apply_node_law(dry, cut, 0.5, NodeBlend::Normal);
        assert_eq!(half[0], dry[0]);
        assert_eq!(half[1], dry[1]);
        assert_eq!(half[2], dry[2]);
        assert!((half[3] - 0.375).abs() <= 1.0e-6);

        let alpha_cut = apply_node_law(dry, [1.0, 0.0, 0.0, 0.25], 1.0, NodeBlend::AlphaCut);
        assert_eq!(alpha_cut, [0.8, 0.2, 0.4, 0.5625]);
    }

    #[test]
    fn hostile_frame_local_values_are_sanitized_before_uniform_creation() {
        let mut rack = RuntimeVisualRack::empty();
        let id = rack
            .push(RuntimeVisualNodeKind::DigitalColor(
                DigitalColorParams::default(),
            ))
            .unwrap();
        let node = rack.get_mut(id).unwrap();
        node.wet = f32::NAN;
        node.kind = RuntimeVisualNodeKind::DigitalColor(DigitalColorParams {
            pixelate_size: f32::NEG_INFINITY,
            rgb_split: f32::INFINITY,
            downsample: f32::NAN,
            hue_shift: f32::INFINITY,
            saturation: -99.0,
            brightness: 99.0,
            contrast: f32::NAN,
            posterize: f32::INFINITY,
            invert: f32::NEG_INFINITY,
            vignette: f32::NAN,
            color_drift: 99.0,
        });
        let compiled = plan(&rack, [1920, 1080]);
        let pass = compiled.passes()[0];
        assert_eq!(pass.wet, 1.0);
        let RackPassKind::DigitalColor(clean) = pass.kind else {
            panic!("expected DigitalColor");
        };
        assert_eq!(clean.pixelate_size, 1.0);
        assert_eq!(clean.rgb_split, 0.0);
        assert_eq!(clean.downsample, 1.0);
        assert_eq!(clean.saturation, -1.0);
        assert_eq!(clean.brightness, 1.0);
        assert_eq!(clean.color_drift, 0.02);
        let uniforms = RackUniforms::for_pass(
            pass,
            compiled.source_dimensions(),
            compiled.output_dimensions(),
            f32::NAN,
            false,
        );
        for value in uniforms
            .frame
            .into_iter()
            .chain(uniforms.p0)
            .chain(uniforms.p1)
            .chain(uniforms.p2)
            .chain(uniforms.p3)
            .chain(uniforms.p4)
            .chain(uniforms.p5)
            .chain(uniforms.p6)
            .chain(uniforms.p7)
        {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn image_pass_retains_the_resolved_stable_route() {
        let expected_route = route(ResolvedImageSource::CleanProgram);
        let mut rack = RuntimeVisualRack::empty();
        let id = rack
            .push(RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(
                RuntimeImageMatte {
                    tap: expected_route,
                    channel: MatteChannel::Luma,
                    invert: false,
                    amount: 0.7,
                    threshold: 0.4,
                    softness: 0.2,
                },
            )))
            .unwrap();
        let compiled = plan(&rack, [16, 9]);
        assert_eq!(compiled.passes()[0].node_id, id);
        assert_eq!(compiled.passes()[0].kind.image_tap(), Some(expected_route));
        assert_eq!(compiled.cross_input_taps(), 1);
        assert_eq!(compiled.max_sampled_textures_in_pass(), 2);
    }

    struct GpuHarness {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl GpuHarness {
        fn new() -> Self {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .expect("GPU adapter for Collision Rack test");
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("Collision Rack test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }))
                .expect("GPU device for Collision Rack test");
            Self { device, queue }
        }

        fn texture(
            &self,
            dimensions: [u32; 2],
            pixels: &[[f32; 4]],
            label: &'static str,
        ) -> (wgpu::Texture, wgpu::TextureView, Vec<u8>) {
            assert_eq!(
                pixels.len(),
                dimensions[0] as usize * dimensions[1] as usize
            );
            let mut bytes = Vec::with_capacity(pixels.len() * 8);
            for pixel in pixels {
                for channel in pixel {
                    bytes.extend_from_slice(&f32_to_f16(*channel).to_le_bytes());
                }
            }
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: dimensions[0],
                    height: dimensions[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: RACK_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dimensions[0] * 8),
                    rows_per_image: Some(dimensions[1]),
                },
                wgpu::Extent3d {
                    width: dimensions[0],
                    height: dimensions[1],
                    depth_or_array_layers: 1,
                },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view, bytes)
        }

        fn render(
            &self,
            executor: &CollisionRackExecutor,
            plan: &CollisionRackPlan,
            source: &RackSourceBinding,
            images: &RackImageBindings,
            time: f32,
        ) -> (RackEncodeReport, Vec<u8>) {
            let dimensions = plan.output_dimensions();
            let unpadded_row = dimensions[0] * 8;
            let padded_row = (unpadded_row + 255) & !255;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Collision Rack test readback"),
                size: u64::from(padded_row) * u64::from(dimensions[1]),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Collision Rack test encoder"),
                });
            let report = executor
                .encode(&self.queue, &mut encoder, plan, source, images, time)
                .unwrap();
            let output = executor.output(report);
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: output.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(dimensions[1]),
                    },
                },
                wgpu::Extent3d {
                    width: dimensions[0],
                    height: dimensions[1],
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(encoder.finish()));
            let slice = staging.slice(..);
            let (send, receive) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU wait");
            receive.recv().expect("map callback").expect("map result");
            let mapped = slice.get_mapped_range();
            let mut compact = Vec::with_capacity(unpadded_row as usize * dimensions[1] as usize);
            for row in 0..dimensions[1] as usize {
                let start = row * padded_row as usize;
                compact.extend_from_slice(&mapped[start..start + unpadded_row as usize]);
            }
            drop(mapped);
            staging.unmap();
            (report, compact)
        }
    }

    fn decode_pixels(bytes: &[u8]) -> Vec<[f32; 4]> {
        bytes
            .chunks_exact(8)
            .map(|pixel| {
                std::array::from_fn(|channel| {
                    let offset = channel * 2;
                    f16_to_f32(u16::from_le_bytes([pixel[offset], pixel[offset + 1]]))
                })
            })
            .collect()
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_noncommutative_reorder_and_wet_zero_byte_identity() {
        let gpu = GpuHarness::new();
        let dimensions = [4, 4];
        let pixels = vec![[0.25, 0.25, 0.25, 1.0]; 16];
        let (_texture, view, source_bytes) = gpu.texture(dimensions, &pixels, "rack order source");
        let executor = CollisionRackExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        let source = executor
            .prepare_source(&gpu.device, &view, dimensions)
            .unwrap();

        let digital = DigitalColorParams {
            invert: 1.0,
            ..Default::default()
        };
        let key = KeyParams {
            mode: KeyMode::KeepBright,
            threshold: 0.5,
            softness: 0.0,
            ..Default::default()
        };
        let mut first = RuntimeVisualRack::empty();
        first
            .push(RuntimeVisualNodeKind::DigitalColor(digital))
            .unwrap();
        first.push(RuntimeVisualNodeKind::Key(key)).unwrap();
        let mut second = RuntimeVisualRack::empty();
        second.push(RuntimeVisualNodeKind::Key(key)).unwrap();
        second
            .push(RuntimeVisualNodeKind::DigitalColor(digital))
            .unwrap();
        let (_, a) = gpu.render(
            &executor,
            &plan(&first, dimensions),
            &source,
            &RackImageBindings::empty(),
            1.0,
        );
        let (_, b) = gpu.render(
            &executor,
            &plan(&second, dimensions),
            &source,
            &RackImageBindings::empty(),
            1.0,
        );
        assert_ne!(a, b, "reordering noncommutative nodes must change output");

        let mut bypass = RuntimeVisualRack::empty();
        let id = bypass
            .push(RuntimeVisualNodeKind::Grain(GrainParams {
                intensity: 0.3,
                seed: 77,
                ..Default::default()
            }))
            .unwrap();
        bypass.get_mut(id).unwrap().wet = 0.0;
        let (_, unchanged) = gpu.render(
            &executor,
            &plan(&bypass, dimensions),
            &source,
            &RackImageBindings::empty(),
            2.0,
        );
        assert_eq!(unchanged, source_bytes, "wet=0 must be a byte-exact bypass");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_key_and_masks_keep_straight_rgb_without_fringe_and_shapes_differ() {
        let gpu = GpuHarness::new();
        let dimensions = [8, 8];
        let pixels = vec![[0.25, 0.5, 0.75, 1.0]; 64];
        let (_texture, view, _) = gpu.texture(dimensions, &pixels, "rack mask source");
        let executor = CollisionRackExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        let source = executor
            .prepare_source(&gpu.device, &view, dimensions)
            .unwrap();

        let mut keyed = RuntimeVisualRack::empty();
        keyed
            .push(RuntimeVisualNodeKind::Key(KeyParams {
                mode: KeyMode::KeepBright,
                threshold: 1.0,
                softness: 0.0,
                ..Default::default()
            }))
            .unwrap();
        let (_, key_bytes) = gpu.render(
            &executor,
            &plan(&keyed, dimensions),
            &source,
            &RackImageBindings::empty(),
            0.0,
        );
        let key_pixel = decode_pixels(&key_bytes)[0];
        assert!(key_pixel[3] <= 0.001);
        assert!((key_pixel[0] - 0.25).abs() <= 0.001);
        assert!((key_pixel[1] - 0.5).abs() <= 0.001);
        assert!((key_pixel[2] - 0.75).abs() <= 0.001);

        let mut rectangle = RuntimeVisualRack::empty();
        rectangle
            .push(RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(
                RectangleMask {
                    size: [0.8, 0.8],
                    ..Default::default()
                },
            )))
            .unwrap();
        let mut ellipse = RuntimeVisualRack::empty();
        ellipse
            .push(RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(
                EllipseMask {
                    radii: [0.4, 0.4],
                    ..Default::default()
                },
            )))
            .unwrap();
        let (_, rectangle_bytes) = gpu.render(
            &executor,
            &plan(&rectangle, dimensions),
            &source,
            &RackImageBindings::empty(),
            0.0,
        );
        let (_, ellipse_bytes) = gpu.render(
            &executor,
            &plan(&ellipse, dimensions),
            &source,
            &RackImageBindings::empty(),
            0.0,
        );
        let rectangle_pixels = decode_pixels(&rectangle_bytes);
        let ellipse_pixels = decode_pixels(&ellipse_bytes);
        let corner_inside_rectangle = dimensions[0] as usize + 1;
        assert!(rectangle_pixels[corner_inside_rectangle][3] >= 0.999);
        assert!(ellipse_pixels[corner_inside_rectangle][3] <= 0.001);
        assert!((ellipse_pixels[corner_inside_rectangle][0] - 0.25).abs() <= 0.001);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_image_mask_missing_and_route_mismatch_are_defined_zero_but_valid_donor_passes() {
        let gpu = GpuHarness::new();
        let dimensions = [2, 2];
        let opaque = vec![[0.5, 0.25, 0.75, 1.0]; 4];
        let white = vec![[1.0; 4]; 4];
        let (_source_texture, source_view, _) =
            gpu.texture(dimensions, &opaque, "rack image source");
        let (_donor_texture, donor_view, _) = gpu.texture(dimensions, &white, "rack image donor");
        let executor = CollisionRackExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        let source = executor
            .prepare_source(&gpu.device, &source_view, dimensions)
            .unwrap();
        let expected_route = route(ResolvedImageSource::CleanProgram);
        let mut rack = RuntimeVisualRack::empty();
        let id = rack
            .push(RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(
                RuntimeImageMatte {
                    tap: expected_route,
                    channel: MatteChannel::Alpha,
                    invert: true,
                    amount: 1.0,
                    threshold: 0.5,
                    softness: 0.0,
                },
            )))
            .unwrap();
        let compiled = plan(&rack, dimensions);

        let (missing_report, missing_bytes) = gpu.render(
            &executor,
            &compiled,
            &source,
            &RackImageBindings::empty(),
            0.0,
        );
        assert_eq!(
            missing_report.missing_image_nodes().collect::<Vec<_>>(),
            vec![id]
        );
        assert!(decode_pixels(&missing_bytes)
            .iter()
            .all(|pixel| pixel[3] <= 0.001));

        let mismatched = executor
            .prepare_image_bindings(
                &gpu.device,
                &[RackImageInput {
                    node_id: id,
                    tap: route(ResolvedImageSource::OneBelow),
                    view: Some(&donor_view),
                }],
            )
            .unwrap();
        let (mismatch_report, mismatch_bytes) =
            gpu.render(&executor, &compiled, &source, &mismatched, 0.0);
        assert_eq!(mismatch_report.missing_image_count, 1);
        assert_eq!(mismatch_bytes, missing_bytes);

        let valid = executor
            .prepare_image_bindings(
                &gpu.device,
                &[RackImageInput {
                    node_id: id,
                    tap: expected_route,
                    view: Some(&donor_view),
                }],
            )
            .unwrap();
        let (valid_report, valid_bytes) = gpu.render(&executor, &compiled, &source, &valid, 0.0);
        assert_eq!(valid_report.missing_image_count, 0);
        // Inverted white donor is a valid zero field. Change inversion to
        // prove a valid donor can admit rather than conflating it with missing.
        let RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) =
            &mut rack.get_mut(id).unwrap().kind
        else {
            unreachable!()
        };
        matte.invert = false;
        let admitting = plan(&rack, dimensions);
        let (_, admitted_bytes) = gpu.render(&executor, &admitting, &source, &valid, 0.0);
        assert!(decode_pixels(&admitted_bytes)
            .iter()
            .all(|pixel| pixel[3] >= 0.999));
        assert_ne!(valid_bytes, admitted_bytes);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_repeated_hostile_render_is_deterministic_and_warmed_encode_allocates_nothing() {
        let gpu = GpuHarness::new();
        let dimensions = [8, 8];
        let pixels = (0..64)
            .map(|index| {
                let x = (index % 8) as f32 / 8.0;
                let y = (index / 8) as f32 / 8.0;
                [x, y, 0.5, 1.0]
            })
            .collect::<Vec<_>>();
        let (_texture, view, _) = gpu.texture(dimensions, &pixels, "rack deterministic source");
        let executor = CollisionRackExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        let source = executor
            .prepare_source(&gpu.device, &view, dimensions)
            .unwrap();
        let mut rack = RuntimeVisualRack::empty();
        rack.push(RuntimeVisualNodeKind::Cellular(CellularParams {
            amount: 0.8,
            gap_amount: 0.6,
            seed: u32::MAX,
            ..Default::default()
        }))
        .unwrap();
        rack.push(RuntimeVisualNodeKind::Shift(ShiftParams {
            amount: 1.0,
            density: 1.0,
            seed: 0xdead_beef,
            ..Default::default()
        }))
        .unwrap();
        rack.push(RuntimeVisualNodeKind::Grain(GrainParams {
            intensity: 0.3,
            algorithm: GrainAlgorithm::Blue,
            color: true,
            seed: 123_456,
            ..Default::default()
        }))
        .unwrap();
        let compiled = plan(&rack, dimensions);
        let images = RackImageBindings::empty();
        let warmed = executor.allocation_snapshot();
        assert!(warmed.total() > 0);
        let (_, first) = gpu.render(&executor, &compiled, &source, &images, 12.5);
        assert_eq!(executor.allocation_snapshot(), warmed);
        let (_, second) = gpu.render(&executor, &compiled, &source, &images, 12.5);
        assert_eq!(executor.allocation_snapshot(), warmed);
        assert_eq!(first, second);
        assert!(decode_pixels(&first)
            .into_iter()
            .flatten()
            .all(f32::is_finite));
    }

    fn f32_to_f16(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mantissa = bits & 0x007f_ffff;
        if exponent <= 0 {
            if exponent < -10 {
                return sign;
            }
            let mantissa = mantissa | 0x0080_0000;
            let shift = 14 - exponent;
            return sign | ((mantissa + (1 << (shift - 1))) >> shift) as u16;
        }
        if exponent >= 31 {
            return sign | 0x7c00;
        }
        sign | ((exponent as u16) << 10) | ((mantissa + 0x1000) >> 13) as u16
    }

    fn f16_to_f32(value: u16) -> f32 {
        let sign = u32::from(value & 0x8000) << 16;
        let exponent = (value >> 10) & 0x1f;
        let fraction = value & 0x03ff;
        let bits = if exponent == 0 {
            if fraction == 0 {
                sign
            } else {
                let mut fraction = u32::from(fraction);
                let mut exponent = -14_i32;
                while fraction & 0x0400 == 0 {
                    fraction <<= 1;
                    exponent -= 1;
                }
                fraction &= 0x03ff;
                sign | (((exponent + 127) as u32) << 23) | (fraction << 13)
            }
        } else if exponent == 0x1f {
            sign | 0x7f80_0000 | (u32::from(fraction) << 13)
        } else {
            sign | (u32::from(exponent + 112) << 23) | (u32::from(fraction) << 13)
        };
        f32::from_bits(bits)
    }

    #[test]
    fn half_helpers_cover_the_fixture_domain() {
        for value in [0.0, 0.125, 0.25, 0.5, 0.75, 1.0] {
            assert!((f16_to_f32(f32_to_f16(value)) - value).abs() <= 0.0005);
        }
    }

    #[test]
    fn route_mismatch_cannot_silently_retarget_an_image_binding() {
        let expected = route(ResolvedImageSource::CleanProgram);
        let other = route(ResolvedImageSource::OneBelow);
        let bindings = RackImageBindings {
            entries: vec![RackImageBinding {
                node_id: node_id(3),
                tap: other,
                groups: None,
                valid: false,
            }]
            .into_boxed_slice(),
        };
        assert!(bindings.find(node_id(3), expected).is_none());
        assert!(bindings.find(node_id(3), other).is_some());
    }

    #[test]
    fn displace_ledger_charges_one_pass_three_lookups_and_twelve_explicit_operations() {
        let budget = node_kind_descriptor(NodeKindTag::Displace).budget;
        assert_eq!(budget.full_frame_passes, 1);
        assert_eq!(budget.logical_texture_lookups_per_pixel, 3);
        assert_eq!(budget.texture_samples_per_pixel, 12);
        assert_eq!(budget.sampled_textures_in_pass, 2);
        assert_eq!(budget.cross_input_taps, 1);
        // Three logical bilinear lookups are exactly twelve explicit loads:
        // dry carrier, displaced carrier, donor vector field.
        assert_eq!(
            budget.texture_samples_per_pixel,
            budget.logical_texture_lookups_per_pixel * PREMULTIPLIED_BILINEAR_TEXTURE_OPS
        );

        let mut rack = RuntimeVisualRack::empty();
        let id = rack
            .push(RuntimeVisualNodeKind::Displace(displace(
                0.25,
                -0.5,
                DisplaceBoundary::Wrap,
            )))
            .unwrap();
        let compiled = plan(&rack, [16, 9]);
        assert_eq!(compiled.logical_texture_lookups_per_pixel(), 3);
        assert_eq!(compiled.texture_samples_per_pixel(), 12);
        assert_eq!(compiled.max_sampled_textures_in_pass(), 2);
        assert_eq!(compiled.cross_input_taps(), 1);
        assert!(compiled.max_sampled_textures_in_pass() <= MAX_SAMPLED_TEXTURES_PER_PASS);
        assert_eq!(compiled.passes()[0].node_id, id);
        assert_eq!(
            compiled.passes()[0].kind.image_tap(),
            Some(RuntimeDisplaceParams::default().tap),
            "an active Displace must expose exactly one cross-scope donor tap"
        );
    }

    #[test]
    fn displace_zero_amounts_and_zero_wet_are_exact_bypasses_that_encode_no_pass() {
        let mut rack = RuntimeVisualRack::empty();
        rack.push(RuntimeVisualNodeKind::Displace(
            RuntimeDisplaceParams::default(),
        ))
        .unwrap();
        let compiled = plan(&rack, [8, 8]);
        assert!(
            compiled.passes()[0].is_exact_bypass(),
            "an exact-default Displace must delegate before encoding a pass"
        );

        // Hostile non-finite gains sanitize to zero and stay an exact bypass.
        let mut hostile = RuntimeVisualRack::empty();
        hostile
            .push(RuntimeVisualNodeKind::Displace(displace(
                f32::NAN,
                f32::INFINITY,
                DisplaceBoundary::Wrap,
            )))
            .unwrap();
        assert!(plan(&hostile, [8, 8]).passes()[0].is_exact_bypass());

        // One nonzero axis is enough to make the node live.
        for (x, y) in [(0.001_f32, 0.0_f32), (0.0, -0.001)] {
            let mut live = RuntimeVisualRack::empty();
            live.push(RuntimeVisualNodeKind::Displace(displace(
                x,
                y,
                DisplaceBoundary::Transparent,
            )))
            .unwrap();
            assert!(!plan(&live, [8, 8]).passes()[0].is_exact_bypass());
        }

        // Zero wet remains an exact bypass even with live gains.
        let mut wet_zero = RuntimeVisualRack::empty();
        let id = wet_zero
            .push(RuntimeVisualNodeKind::Displace(displace(
                0.5,
                0.5,
                DisplaceBoundary::Wrap,
            )))
            .unwrap();
        wet_zero.get_mut(id).unwrap().wet = 0.0;
        assert!(plan(&wet_zero, [8, 8]).passes()[0].is_exact_bypass());

        // No other node kind acquires a value-driven bypass from this change.
        let mut others = RuntimeVisualRack::empty();
        others
            .push(RuntimeVisualNodeKind::Grain(GrainParams::default()))
            .unwrap();
        others
            .push(RuntimeVisualNodeKind::Cellular(CellularParams::default()))
            .unwrap();
        others
            .push(RuntimeVisualNodeKind::Shift(ShiftParams::default()))
            .unwrap();
        for pass in plan(&others, [8, 8]).passes() {
            assert!(!pass.is_exact_bypass());
        }
    }

    #[test]
    fn displace_neutral_and_transparent_hostile_donors_produce_exact_zero_displacement() {
        let carrier = RefImage::new(
            [4, 1],
            vec![
                [0.0, 0.0, 0.0, 1.0],
                [0.25, 0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
        );
        let params = displace(1.0, 1.0, DisplaceBoundary::Wrap);
        let probes = [[0.125, 0.5], [0.375, 0.5], [0.625, 0.5], [0.875, 0.5]];

        // Neutral RG = 0.5 at full coverage.
        let neutral = RefImage::uniform([4, 1], [0.5, 0.5, 0.0, 1.0]);
        // Fully transparent, but with maximally hostile hidden RGB.
        let hostile = RefImage::uniform([4, 1], [1.0, 1.0, 1.0, 0.0]);
        // Half coverage with neutral straight RG stays exactly neutral.
        let half_covered = RefImage::uniform([4, 1], [0.5, 0.5, 0.5, 0.5]);

        for uv in probes {
            let identity = carrier.straight_bilinear(uv);
            for donor in [&neutral, &hostile, &half_covered] {
                assert_eq!(displace_vector(donor, uv, true), [0.0, 0.0]);
                assert_eq!(
                    displace_reference(&carrier, donor, uv, params, true),
                    identity
                );
            }
            // A missing binding is the same defined zero field.
            assert_eq!(displace_vector(&hostile, uv, false), [0.0, 0.0]);
            assert_eq!(
                displace_reference(&carrier, &neutral, uv, params, false),
                identity
            );
        }
    }

    #[test]
    fn displace_analytic_axis_fixtures_hold_for_every_boundary() {
        // A 4x1 carrier whose red channel identifies the sampled column.
        let carrier = RefImage::new(
            [4, 1],
            vec![
                [0.0, 0.0, 0.0, 1.0],
                [0.25, 0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
        );
        let plus_x = RefImage::uniform([4, 1], [1.0, 0.5, 0.0, 1.0]);
        let minus_x = RefImage::uniform([4, 1], [0.0, 0.5, 0.0, 1.0]);
        let plus_y = RefImage::uniform([4, 1], [0.5, 1.0, 0.0, 1.0]);
        let minus_y = RefImage::uniform([4, 1], [0.5, 0.0, 0.0, 1.0]);
        let uv = [0.25, 0.5];

        // Full-scale donors decode to exactly ±1 on their own axis only.
        assert_eq!(displace_vector(&plus_x, uv, true), [1.0, 0.0]);
        assert_eq!(displace_vector(&minus_x, uv, true), [-1.0, 0.0]);
        assert_eq!(displace_vector(&plus_y, uv, true), [0.0, 1.0]);
        assert_eq!(displace_vector(&minus_y, uv, true), [0.0, -1.0]);

        // A quarter-scale +X gain moves the sample a quarter of the frame.
        let quarter = displace(0.25, 0.0, DisplaceBoundary::Hold);
        assert_eq!(
            displace_reference(&carrier, &plus_x, uv, quarter, true),
            carrier.straight_bilinear([0.5, 0.5])
        );
        let quarter_negative = displace(0.25, 0.0, DisplaceBoundary::Hold);
        assert_eq!(
            displace_reference(&carrier, &minus_x, uv, quarter_negative, true),
            carrier.straight_bilinear([0.0, 0.5])
        );

        // Full +X gain pushes 0.25 to 1.25, exercising every boundary law.
        let expectations = [
            (DisplaceBoundary::Wrap, Some(0.25_f32)),
            (DisplaceBoundary::Mirror, Some(0.75)),
            (DisplaceBoundary::Hold, Some(1.0)),
            (DisplaceBoundary::Transparent, None),
        ];
        for (boundary, expected_x) in expectations {
            let params = displace(1.0, 0.0, boundary);
            let (mapped, covered) = displace_boundary([1.25, 0.5], boundary);
            let actual = displace_reference(&carrier, &plus_x, uv, params, true);
            match expected_x {
                Some(x) => {
                    assert!(covered, "{boundary:?} must keep coverage");
                    assert!(
                        (mapped[0] - x).abs() <= 1.0e-6,
                        "{boundary:?} mapped 1.25 to {} rather than {x}",
                        mapped[0]
                    );
                    assert_eq!(actual, carrier.straight_bilinear([x, 0.5]));
                }
                None => {
                    assert!(
                        !covered,
                        "Transparent must drop coverage outside the domain"
                    );
                    assert_eq!(actual, [0.0; 4], "Transparent must resolve to nothing");
                }
            }
        }

        // Full -X gain pushes 0.25 to -0.75 through the same laws.
        for (boundary, expected_x) in [
            (DisplaceBoundary::Wrap, Some(0.25_f32)),
            (DisplaceBoundary::Mirror, Some(0.75)),
            (DisplaceBoundary::Hold, Some(0.0)),
            (DisplaceBoundary::Transparent, None),
        ] {
            let (mapped, covered) = displace_boundary([-0.75, 0.5], boundary);
            match expected_x {
                Some(x) => {
                    assert!(covered);
                    assert!(
                        (mapped[0] - x).abs() <= 1.0e-6,
                        "{boundary:?} mapped -0.75 to {}",
                        mapped[0]
                    );
                }
                None => assert!(!covered),
            }
        }
    }

    #[test]
    fn displace_boundary_codes_are_append_only_and_uniforms_carry_them() {
        assert_eq!(DisplaceBoundary::Transparent.code(), 0);
        assert_eq!(DisplaceBoundary::Mirror.code(), 1);
        assert_eq!(DisplaceBoundary::Wrap.code(), 2);
        assert_eq!(DisplaceBoundary::Hold.code(), 3);
        assert_eq!(DisplaceBoundary::default(), DisplaceBoundary::Transparent);

        for boundary in [
            DisplaceBoundary::Transparent,
            DisplaceBoundary::Mirror,
            DisplaceBoundary::Wrap,
            DisplaceBoundary::Hold,
        ] {
            let pass = RackPassDescriptor {
                node_id: node_id(7),
                enabled: true,
                wet: 1.0,
                blend: NodeBlend::Normal,
                kind: RackPassKind::Displace(displace(0.5, -0.25, boundary)),
            };
            let uniforms = RackUniforms::for_pass(pass, [8, 8], [8, 8], 0.0, true);
            assert_eq!(uniforms.meta[0], KIND_DISPLACE);
            assert_eq!(uniforms.meta[2], 1, "donor validity travels in meta.z");
            assert_eq!(uniforms.p0[0], 0.5);
            assert_eq!(uniforms.p0[1], -0.25);
            assert_eq!(uniforms.p0[2], boundary.code() as f32);
            let missing = RackUniforms::for_pass(pass, [8, 8], [8, 8], 0.0, false);
            assert_eq!(missing.meta[2], 0);
        }
    }

    #[test]
    fn displace_pass_sanitizes_hostile_gains_without_touching_route_or_boundary() {
        let tap = route(ResolvedImageSource::CleanProgram);
        let authored = RuntimeDisplaceParams {
            tap,
            amount_x: 9.0,
            amount_y: -9.0,
            boundary: DisplaceBoundary::Mirror,
        };
        let mut rack = RuntimeVisualRack::empty();
        rack.push(RuntimeVisualNodeKind::Displace(authored))
            .unwrap();
        let compiled = plan(&rack, [8, 8]);
        let RackPassKind::Displace(compiled_params) = compiled.passes()[0].kind else {
            panic!("compiled pass must remain a Displace");
        };
        assert_eq!(compiled_params.amount_x, 1.0);
        assert_eq!(compiled_params.amount_y, -1.0);
        assert_eq!(compiled_params.boundary, DisplaceBoundary::Mirror);
        assert_eq!(
            compiled_params.tap, tap,
            "sanitizing never rewrites a route"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_displace_matches_the_cpu_reference_and_zero_gain_is_byte_identical() {
        let gpu = GpuHarness::new();
        let dimensions = [8, 8];
        let carrier_pixels: Vec<[f32; 4]> = (0..64)
            .map(|index| {
                let x = (index % 8) as f32 / 7.0;
                let y = (index / 8) as f32 / 7.0;
                [x, y, 0.5, 1.0]
            })
            .collect();
        let carrier = RefImage::new(dimensions, carrier_pixels.clone());
        let (_carrier_texture, carrier_view, carrier_bytes) =
            gpu.texture(dimensions, &carrier_pixels, "displace carrier");

        // Constant full-scale +X donor at full coverage.
        let donor_pixels = vec![[1.0, 0.5, 0.0, 1.0]; 64];
        let donor = RefImage::new(dimensions, donor_pixels.clone());
        let (_donor_texture, donor_view, _) = gpu.texture(dimensions, &donor_pixels, "donor");

        let executor = CollisionRackExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        let source = executor
            .prepare_source(&gpu.device, &carrier_view, dimensions)
            .unwrap();

        let params = displace(0.25, 0.0, DisplaceBoundary::Wrap);
        let mut rack = RuntimeVisualRack::empty();
        let id = rack.push(RuntimeVisualNodeKind::Displace(params)).unwrap();
        let compiled = plan(&rack, dimensions);
        let bindings = executor
            .prepare_image_bindings(
                &gpu.device,
                &[RackImageInput {
                    node_id: id,
                    tap: params.tap,
                    view: Some(&donor_view),
                }],
            )
            .unwrap();
        let warm = executor.allocation_snapshot();
        let (report, bytes) = gpu.render(&executor, &compiled, &source, &bindings, 0.0);
        assert_eq!(report.executed_passes, 1);
        assert_eq!(report.missing_image_count, 0);
        assert_eq!(
            warm,
            executor.allocation_snapshot(),
            "a warmed Displace encode must allocate nothing"
        );

        let rendered = decode_pixels(&bytes);
        for (index, actual) in rendered.iter().enumerate() {
            let uv = [
                (index % 8) as f32 / 8.0 + 0.5 / 8.0,
                (index / 8) as f32 / 8.0 + 0.5 / 8.0,
            ];
            let expected = displace_reference(&carrier, &donor, uv, params, true);
            for channel in 0..4 {
                assert!(
                    (actual[channel] - expected[channel]).abs() <= 0.01,
                    "pixel {index} channel {channel}: GPU {} vs reference {}",
                    actual[channel],
                    expected[channel]
                );
            }
        }

        // Zero gain must return the carrier bytes untouched.
        let mut bypass = RuntimeVisualRack::empty();
        bypass
            .push(RuntimeVisualNodeKind::Displace(
                RuntimeDisplaceParams::default(),
            ))
            .unwrap();
        let (bypass_report, bypass_bytes) = gpu.render(
            &executor,
            &plan(&bypass, dimensions),
            &source,
            &RackImageBindings::empty(),
            0.0,
        );
        assert_eq!(bypass_report.executed_passes, 0);
        assert_eq!(
            bypass_bytes, carrier_bytes,
            "zero-gain Displace must be a byte-exact bypass"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_displace_transparent_hostile_donor_is_byte_identical_to_the_carrier() {
        let gpu = GpuHarness::new();
        let dimensions = [8, 8];
        let carrier_pixels: Vec<[f32; 4]> = (0..64)
            .map(|index| [(index % 8) as f32 / 7.0, 0.25, 0.75, 1.0])
            .collect();
        let (_carrier_texture, carrier_view, _) =
            gpu.texture(dimensions, &carrier_pixels, "hostile carrier");
        let executor = CollisionRackExecutor::new(&gpu.device, &gpu.queue, dimensions).unwrap();
        let source = executor
            .prepare_source(&gpu.device, &carrier_view, dimensions)
            .unwrap();

        let params = displace(1.0, 1.0, DisplaceBoundary::Transparent);
        let mut rack = RuntimeVisualRack::empty();
        let id = rack.push(RuntimeVisualNodeKind::Displace(params)).unwrap();
        let compiled = plan(&rack, dimensions);

        // Neutral donor: a full-gain node that must still not move a pixel.
        let neutral = vec![[0.5, 0.5, 0.0, 1.0]; 64];
        let (_neutral_texture, neutral_view, _) =
            gpu.texture(dimensions, &neutral, "neutral donor");
        let neutral_bindings = executor
            .prepare_image_bindings(
                &gpu.device,
                &[RackImageInput {
                    node_id: id,
                    tap: params.tap,
                    view: Some(&neutral_view),
                }],
            )
            .unwrap();
        let (_, neutral_bytes) = gpu.render(&executor, &compiled, &source, &neutral_bindings, 0.0);

        // Transparent donor carrying maximally hostile hidden RGB.
        let hostile = vec![[1.0, 1.0, 1.0, 0.0]; 64];
        let (_hostile_texture, hostile_view, _) =
            gpu.texture(dimensions, &hostile, "hostile donor");
        let hostile_bindings = executor
            .prepare_image_bindings(
                &gpu.device,
                &[RackImageInput {
                    node_id: id,
                    tap: params.tap,
                    view: Some(&hostile_view),
                }],
            )
            .unwrap();
        let (_, hostile_bytes) = gpu.render(&executor, &compiled, &source, &hostile_bindings, 0.0);

        assert_eq!(
            hostile_bytes, neutral_bytes,
            "hidden RGB at alpha zero must never reach the vector field"
        );

        // A missing binding is the same defined zero field.
        let (missing_report, missing_bytes) = gpu.render(
            &executor,
            &compiled,
            &source,
            &RackImageBindings::empty(),
            0.0,
        );
        assert_eq!(missing_report.missing_image_count, 1);
        assert_eq!(missing_bytes, neutral_bytes);
    }
}
