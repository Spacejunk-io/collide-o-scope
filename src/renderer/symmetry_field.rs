//! Dedicated eight-texture GPU executor for the Symmetry Field.
//!
//! The fixed Collision Rack pipeline binds exactly two sampled textures behind
//! one frozen 176-byte uniform record. A Symmetry Field needs eight
//! simultaneously sampled textures — a carrier, two donors, the Compat8
//! clean-history D2 array, and a vector/gate pair for each of two motion
//! donors — behind a 1,024-byte record, so it cannot ride that layout and is
//! lifted into its own step by the composition planner.
//!
//! Eight sampled textures in one pass is portable under the production device
//! floor. `docs/evidence/s2-eight-texture-floor-receipt.json` (commit
//! `4866d34`, the
//! receipt's own `claim_first_proven`) created its
//! probe device with the *exact* production `required_limits`
//! (`wgpu::Limits::default()`, `renderer/state.rs:2892`), proved an
//! eight-texture layout is accepted, and proved a seventeen-texture layout is
//! REFUSED on that same device. The refusal is what makes the acceptance
//! generalize: wgpu validates against the requested limits, not the adapter's
//! raw capability, and `request_device` refuses any adapter below the floor.
//! The receipt's own scope limits carry forward verbatim — capability only, not
//! performance, bandwidth or cache behaviour, on one adapter and one backend.
//!
//! Resource contract, exactly:
//!
//! | Item | Exact contract |
//! |---|---:|
//! | Sampled textures per pass | 8 |
//! | Bind groups in the pipeline layout | 3 (image, uniform, motion) |
//! | Render passes per node | 1 |
//! | Worst-case explicit texture operations per pixel | 10 |
//! | Uniform arena per node | 1,024 bytes |
//! | Neutral textures | 3 tiny textures / 4 views |
//! | Full-frame persistent surfaces | 0 |
//!
//! **Three bind groups is a deliberate, documented deviation from the frozen
//! table's "Bind groups: 2".** The motion vector/gate pair moved into a group
//! of its own because a `MotionGpuField` owns a *committed ping/pong parity of
//! its own* (`MotionMemoryStage::render_field_index`), which is a third parity
//! dimension above the carrier parity and the composition's N-1 tap parity.
//! Held in one input group those three would multiply — 4 × 4 = 16 prebuilt
//! groups per node. Split, they add: **4 image groups** (carrier parity × N-1
//! parity) plus **4 motion groups** (the two slots' committed parities, which
//! are deliberately not required to agree) = **8 prebuilt groups per node**,
//! and three groups bound per pass. The sampled-texture count, the pass count
//! and the per-pixel operation ledger are unchanged; only the grouping moved.
//! Without it a fully authored motion route would decode to exactly zero, which
//! is the same failure mode as an isolated shader.
//!
//! This executor owns **no** output-sized surface. The caller supplies both
//! carrier parities at prepare time and the render target at encode time, so a
//! Symmetry Field adds nothing to the composition's warm surface ledger.
//!
//! The clean-history ring is likewise **borrowed, never duplicated**. A new
//! RGBA16F full-frame history ring would cost 398.1 MB (379.7 MiB) at 1080p
//! — `1920 × 1080 × 8 × 24` bytes = 398,131,200, a decimal-vs-binary unit
//! distinction three documents previously blurred; the
//! committed Compat8 ring (`composition_host.rs:1208`, 24 layers of
//! `Rgba8UnormSrgb`, already charged as
//! `ADVANCED_TEMPORAL_COMPAT8_SURFACE_LAYERS = 25`) is bound through its single
//! D2Array *read* view. The 24 single-layer views are render targets owned by
//! the temporal pass and are never handed to this executor.

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::effects::params::TEMPORAL_REFERENCE_FPS;
use crate::evaluated_frame::evaluated_composition::EvaluatedSymmetryFieldPlan;
use crate::renderer::rack::RACK_TEXTURE_FORMAT;
use crate::symmetry::{
    SymmetryFrameUniforms, SymmetryGpuBindings, SymmetryGpuUniforms, SYMMETRY_IMAGE_SLOTS,
    SYMMETRY_MOTION_SLOTS,
};
use crate::visual_rack::{
    node_kind_descriptor, NodeId, NodeKindTag, MAX_NODES_PER_RACK,
    MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS,
};

/// The eight sampled textures the dedicated pipeline declares, across its image
/// and motion groups. This is the count the planner admits against
/// [`MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS`] and against the device's own raw
/// `max_sampled_textures_per_shader_stage`. Splitting the pipeline layout into
/// three groups did not change it: a fragment stage's sampled-texture budget is
/// counted across every bound group, not per group.
pub(crate) const SYMMETRY_FIELD_SAMPLED_TEXTURES: u32 = 8;

/// Sampled textures in the image group: carrier, donor 0, donor 1, and the
/// clean-history D2 array.
pub(crate) const SYMMETRY_FIELD_IMAGE_TEXTURES: u32 = 4;

/// Sampled textures in the motion group: one vector/gate pair per motion slot.
pub(crate) const SYMMETRY_FIELD_MOTION_TEXTURES: u32 = 2 * SYMMETRY_MOTION_SLOTS as u32;

/// The two image bind groups prepared per node, one per carrier parity.
///
/// This is the INNER parity, the one that selects which surface holds the
/// carrier. The composition additionally prepares both committed N-1 read
/// parities above it, exactly as it does for a rack segment, so one authored
/// node costs four image bind groups in a live composition and two per call
/// here. Two per node is the honest count at this boundary; do not report it
/// as one.
pub(crate) const SYMMETRY_FIELD_CARRIER_PARITIES: usize = 2;

/// The committed ping/pong parities one `MotionGpuField` publishes through.
pub(crate) const SYMMETRY_FIELD_MOTION_PARITIES: usize = 2;

/// The motion bind groups prepared per node: every combination of the two
/// slots' committed parities.
///
/// The two slots are two independent fields with independent
/// `MotionMemoryStage::render_field_index` values, so they are deliberately not
/// required to agree; every pair is prebuilt and encode selects one. Four per
/// node, once — not once per N-1 read parity, because the motion group holds no
/// image view and is therefore parity-independent in that dimension.
pub(crate) const SYMMETRY_FIELD_MOTION_GROUPS: usize =
    SYMMETRY_FIELD_MOTION_PARITIES * SYMMETRY_FIELD_MOTION_PARITIES;

/// One reference tick. A sector's motion vector is UV per second, and this is
/// what converts it into a bounded per-frame displacement. It is derived from
/// the clean-history ring's own cadence rather than from wall time or the live
/// frame rate, so live and offline render the same offset from the same
/// authored state.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the reference tick lives in the shader; the constant is its CPU witness"
    )
)]
pub(crate) const SYMMETRY_MOTION_REFERENCE_SECONDS: f32 = 1.0 / TEMPORAL_REFERENCE_FPS;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SymmetryAllocationSnapshot {
    pub textures: u64,
    pub buffers: u64,
    pub bind_groups: u64,
    pub pipelines: u64,
}

#[derive(Default)]
struct SymmetryAllocationCounters {
    textures: AtomicU64,
    buffers: AtomicU64,
    bind_groups: AtomicU64,
    pipelines: AtomicU64,
}

impl SymmetryAllocationCounters {
    fn snapshot(&self) -> SymmetryAllocationSnapshot {
        SymmetryAllocationSnapshot {
            textures: self.textures.load(Ordering::Relaxed),
            buffers: self.buffers.load(Ordering::Relaxed),
            bind_groups: self.bind_groups.load(Ordering::Relaxed),
            pipelines: self.pipelines.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SymmetryFieldGpuError {
    ZeroDimensions([u32; 2]),
    DimensionsExceedDevice {
        requested: [u32; 2],
        limit: u32,
    },
    /// The device this executor was built on cannot bind eight sampled
    /// textures in one fragment stage. Every adapter that satisfies the
    /// production `request_device` can, so this is a refusal, never a fallback.
    SampledTextureFloor {
        required: u32,
        available: u32,
    },
    BufferSizeOverflow,
    ResourceCreation {
        context: &'static str,
        kind: &'static str,
        message: String,
    },
    TooManyNodeBindings {
        requested: usize,
        limit: usize,
    },
    DuplicateNodeBinding(NodeId),
    UnpreparedNode(NodeId),
    UnpreparedMotionNode(NodeId),
    CarrierParity(usize),
    MotionParity {
        slot: usize,
        parity: usize,
    },
    UniformSlotRange {
        slot: usize,
        capacity: usize,
    },
}

impl fmt::Display for SymmetryFieldGpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions(dimensions) => write!(
                formatter,
                "Symmetry Field executor dimensions must be non-zero ({}x{})",
                dimensions[0], dimensions[1]
            ),
            Self::DimensionsExceedDevice { requested, limit } => write!(
                formatter,
                "Symmetry Field executor {}x{} exceeds the device 2D texture limit of {limit}",
                requested[0], requested[1]
            ),
            Self::SampledTextureFloor {
                required,
                available,
            } => write!(
                formatter,
                "the dedicated Symmetry Field pass needs {required} sampled textures per fragment \
                 stage; this device reports {available}"
            ),
            Self::BufferSizeOverflow => {
                formatter.write_str("Symmetry Field uniform buffer size overflowed")
            }
            Self::ResourceCreation {
                context,
                kind,
                message,
            } => write!(formatter, "{context} failed ({kind}): {message}"),
            Self::TooManyNodeBindings { requested, limit } => write!(
                formatter,
                "Symmetry Field binding set contains {requested} nodes; limit is {limit}"
            ),
            Self::DuplicateNodeBinding(id) => write!(
                formatter,
                "duplicate Symmetry Field binding for node {}",
                id.get()
            ),
            Self::UnpreparedNode(id) => write!(
                formatter,
                "Symmetry Field node {} was never prepared",
                id.get()
            ),
            Self::UnpreparedMotionNode(id) => write!(
                formatter,
                "Symmetry Field node {} has no prepared motion bind groups",
                id.get()
            ),
            Self::CarrierParity(parity) => write!(
                formatter,
                "Symmetry Field carrier parity {parity} is outside the prepared pair"
            ),
            Self::MotionParity { slot, parity } => write!(
                formatter,
                "Symmetry Field motion slot {slot} committed parity {parity} is outside the \
                 prepared pair"
            ),
            Self::UniformSlotRange { slot, capacity } => write!(
                formatter,
                "Symmetry Field uniform slot {slot} exceeds the arena capacity of {capacity}"
            ),
        }
    }
}

impl std::error::Error for SymmetryFieldGpuError {}

/// One motion donor's primitive vector/gate ping-pong pair, at its own
/// `MotionGrid` resolution rather than the output resolution.
///
/// **Both committed parities of both textures** are supplied, because the
/// donor's `MotionGpuField` chooses its own `render_field_index` per frame and
/// that choice is not known at prepare time. Prebuilding every pair is what
/// keeps warm encode allocation-free.
///
/// Both textures are required. An incomplete pair is expressed by passing
/// `None` for the whole slot, never by supplying one half, because a vector
/// without its gate would be applied at full confidence.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SymmetryMotionViews<'a> {
    pub vectors: [&'a wgpu::TextureView; SYMMETRY_FIELD_MOTION_PARITIES],
    pub gates: [&'a wgpu::TextureView; SYMMETRY_FIELD_MOTION_PARITIES],
    pub grid: [u32; 2],
}

/// Everything one authored node's IMAGE group routes to, resolved to views by
/// the caller.
///
/// Slot index is route identity all the way down: `donors[1]` is always the
/// second image route even when the first is missing, and clearing slot 0 never
/// slides slot 1's donor into its place.
pub(crate) struct SymmetryFieldInput<'a> {
    pub node_id: NodeId,
    pub donors: [Option<&'a wgpu::TextureView>; SYMMETRY_IMAGE_SLOTS],
    /// The Compat8 clean-history **D2Array read view**, borrowed from
    /// `CompositionHost`. `None` binds the neutral single-layer array.
    pub history: Option<&'a wgpu::TextureView>,
}

/// Everything one authored node's MOTION group routes to.
///
/// Prepared separately from the image group and exactly once per node: the
/// motion group holds no image view, so it is independent of both the carrier
/// parity and the composition's N-1 read parity.
pub(crate) struct SymmetryFieldMotionInput<'a> {
    pub node_id: NodeId,
    pub motion: [Option<SymmetryMotionViews<'a>>; SYMMETRY_MOTION_SLOTS],
}

struct SymmetryFieldBinding {
    node_id: NodeId,
    /// One prepared image bind group per carrier parity. Two per node is the
    /// honest count: warm encode selects `groups[parity]` and creates nothing.
    groups: [wgpu::BindGroup; SYMMETRY_FIELD_CARRIER_PARITIES],
    /// A view was bound for this image slot. Immutable for the topology.
    donor_bound: [bool; SYMMETRY_IMAGE_SLOTS],
    /// The bound view holds this frame's content. Frame-local, flipped by
    /// [`SymmetryFieldBindings::set_donor_ready`] without reallocating, exactly
    /// as a cold N-1 rack donor becomes valid.
    donor_ready: [bool; SYMMETRY_IMAGE_SLOTS],
    history_bound: bool,
}

/// Bounded per-node image bindings, prepared outside encode and searched
/// without allocation by stable node ID.
pub(crate) struct SymmetryFieldBindings {
    entries: Box<[SymmetryFieldBinding]>,
}

impl SymmetryFieldBindings {
    pub fn empty() -> Self {
        Self {
            entries: Box::new([]),
        }
    }

    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "binding cardinality is exposed for GPU goldens")
    )]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Update only one image slot's frame-local readiness bit. The texture
    /// binding stays immutable, and the slot index is the route identity: a
    /// readiness flip on slot 0 can never move slot 1's donor.
    pub fn set_donor_ready(&mut self, node_id: NodeId, slot: usize, ready: bool) -> bool {
        if slot >= SYMMETRY_IMAGE_SLOTS {
            return false;
        }
        let Ok(index) = self
            .entries
            .binary_search_by_key(&node_id, |entry| entry.node_id)
        else {
            return false;
        };
        let entry = &mut self.entries[index];
        entry.donor_ready[slot] = ready && entry.donor_bound[slot];
        true
    }

    fn find(&self, node_id: NodeId) -> Option<&SymmetryFieldBinding> {
        let index = self
            .entries
            .binary_search_by_key(&node_id, |entry| entry.node_id)
            .ok()?;
        Some(&self.entries[index])
    }
}

impl Default for SymmetryFieldBindings {
    fn default() -> Self {
        Self::empty()
    }
}

struct SymmetryFieldMotionBinding {
    node_id: NodeId,
    /// `[slot 0 committed parity][slot 1 committed parity]`. Four per node,
    /// prebuilt once; encode indexes and creates nothing.
    groups: [[wgpu::BindGroup; SYMMETRY_FIELD_MOTION_PARITIES]; SYMMETRY_FIELD_MOTION_PARITIES],
    /// A complete vector/gate pair was bound for this slot. Immutable for the
    /// topology.
    motion_bound: [bool; SYMMETRY_MOTION_SLOTS],
    /// The bound field holds a materialized parity this frame. Frame-local,
    /// flipped by [`SymmetryFieldMotionBindings::set_motion_ready`] without
    /// reallocating, exactly as an image donor's readiness bit is.
    motion_ready: [bool; SYMMETRY_MOTION_SLOTS],
    motion_grid: [[u32; 2]; SYMMETRY_MOTION_SLOTS],
}

/// Bounded per-node motion bindings.
pub(crate) struct SymmetryFieldMotionBindings {
    entries: Box<[SymmetryFieldMotionBinding]>,
}

impl SymmetryFieldMotionBindings {
    pub fn empty() -> Self {
        Self {
            entries: Box::new([]),
        }
    }

    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "binding cardinality is exposed for GPU goldens")
    )]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Update only one motion slot's frame-local readiness bit.
    ///
    /// A donor field whose committed parity is not yet materialized — the first
    /// frame of a lattice, or a held acquisition — is honestly *not ready*, and
    /// the packed record's validity lane closes so the shader decodes exactly
    /// zero displacement. The bind group is untouched.
    pub fn set_motion_ready(&mut self, node_id: NodeId, slot: usize, ready: bool) -> bool {
        if slot >= SYMMETRY_MOTION_SLOTS {
            return false;
        }
        let Ok(index) = self
            .entries
            .binary_search_by_key(&node_id, |entry| entry.node_id)
        else {
            return false;
        };
        let entry = &mut self.entries[index];
        entry.motion_ready[slot] = ready && entry.motion_bound[slot];
        true
    }

    fn find(&self, node_id: NodeId) -> Option<&SymmetryFieldMotionBinding> {
        let index = self
            .entries
            .binary_search_by_key(&node_id, |entry| entry.node_id)
            .ok()?;
        Some(&self.entries[index])
    }
}

impl Default for SymmetryFieldMotionBindings {
    fn default() -> Self {
        Self::empty()
    }
}

/// What one encoded pass actually observed. Counts are per node, so two lost
/// image routes on one node report two missing donors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SymmetryFieldEncodeReport {
    /// True when the uniform declared the exact-bypass identity. The pass is
    /// still encoded — the caller's target is a different surface than the
    /// carrier, so there is nothing to delegate to at this level — but the
    /// shader takes the direct `textureLoad` branch and the result is the
    /// carrier byte for byte.
    pub identity: bool,
    pub missing_donors: u8,
    pub missing_motion: u8,
    pub history_bound: bool,
}

/// The committed clean-history read cursor, as the temporal pass sees it.
///
/// Age 0 is the **virtual current image** and never addresses a stored layer;
/// ages `1..valid` address `(write + 24 - age) % 24`. Both fields come from
/// `temporal::temporal_read_snapshot`, never from raw `TemporalState` fields,
/// so a Symmetry Field and the temporal pass agree about which layer an age
/// names.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SymmetryHistoryCursor {
    pub write_index: u32,
    pub valid: u32,
}

/// Fixed-pipeline, surface-free executor. Construction is transactional: every
/// handle-returning creation is enclosed by validation, internal and OOM
/// scopes, and no handle escapes if a scope reports an error.
pub(crate) struct SymmetryFieldExecutor {
    dimensions: [u32; 2],
    pipeline: wgpu::RenderPipeline,
    image_layout: wgpu::BindGroupLayout,
    motion_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_group: wgpu::BindGroup,
    uniform_stride: u64,
    uniform_slot_capacity: usize,
    _neutral_image: wgpu::Texture,
    neutral_image_view: wgpu::TextureView,
    neutral_history_view: wgpu::TextureView,
    _neutral_vectors: wgpu::Texture,
    neutral_vector_view: wgpu::TextureView,
    _neutral_gates: wgpu::Texture,
    neutral_gate_view: wgpu::TextureView,
    allocations: SymmetryAllocationCounters,
}

impl SymmetryFieldExecutor {
    /// Create the dedicated executor for one output size and a caller-counted
    /// number of uniform slots — one slot per Symmetry Field step the frame can
    /// encode before a single queue submit.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dimensions: [u32; 2],
        uniform_slot_capacity: usize,
    ) -> Result<Self, SymmetryFieldGpuError> {
        if dimensions.contains(&0) {
            return Err(SymmetryFieldGpuError::ZeroDimensions(dimensions));
        }
        let limits = device.limits();
        let dimension_limit = limits.max_texture_dimension_2d;
        if dimensions[0] > dimension_limit || dimensions[1] > dimension_limit {
            return Err(SymmetryFieldGpuError::DimensionsExceedDevice {
                requested: dimensions,
                limit: dimension_limit,
            });
        }
        // Refuse by name rather than degrade. The production device request is
        // `Limits::default()`, whose floor is sixteen, so this can only trip on
        // a device built with a deliberately narrower request.
        if limits.max_sampled_textures_per_shader_stage < SYMMETRY_FIELD_SAMPLED_TEXTURES {
            return Err(SymmetryFieldGpuError::SampledTextureFloor {
                required: SYMMETRY_FIELD_SAMPLED_TEXTURES,
                available: limits.max_sampled_textures_per_shader_stage,
            });
        }
        // The stride is align_up(1024, min_uniform_buffer_offset_alignment).
        // Never hardcode 1024: an adapter reporting a larger alignment strides
        // wider, and the record size stays exactly 1,024 either way.
        let alignment = u64::from(limits.min_uniform_buffer_offset_alignment.max(1));
        let uniform_size = SymmetryGpuUniforms::BYTES;
        let uniform_stride = aligned_uniform_stride(uniform_size, alignment)
            .ok_or(SymmetryFieldGpuError::BufferSizeOverflow)?;
        let uniform_slot_capacity = uniform_slot_capacity.max(1);
        let buffer_size = uniform_stride
            .checked_mul(
                u64::try_from(uniform_slot_capacity)
                    .map_err(|_| SymmetryFieldGpuError::BufferSizeOverflow)?,
            )
            .ok_or(SymmetryFieldGpuError::BufferSizeOverflow)?;
        if buffer_size > limits.max_buffer_size {
            return Err(SymmetryFieldGpuError::BufferSizeOverflow);
        }
        let last_offset = uniform_stride
            .checked_mul(uniform_slot_capacity.saturating_sub(1) as u64)
            .ok_or(SymmetryFieldGpuError::BufferSizeOverflow)?;
        if u32::try_from(last_offset).is_err() {
            return Err(SymmetryFieldGpuError::BufferSizeOverflow);
        }

        let allocations = SymmetryAllocationCounters::default();
        let resources = create_checked(device, "Symmetry Field initialization", || {
            // Group 0 — the image half of the eight: carrier, both donors, and
            // the clean-history D2 array.
            let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Symmetry Field image BGL"),
                entries: &[
                    loaded_texture_entry(0),
                    loaded_texture_entry(1),
                    loaded_texture_entry(2),
                    loaded_array_texture_entry(3),
                ],
            });
            // Group 2 — the motion half. Separate precisely so a field's own
            // committed parity adds to the carrier and N-1 parities instead of
            // multiplying with them.
            let motion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Symmetry Field motion BGL"),
                entries: &[
                    loaded_texture_entry(0),
                    loaded_texture_entry(1),
                    loaded_texture_entry(2),
                    loaded_texture_entry(3),
                ],
            });
            let uniform_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Symmetry Field dynamic uniform BGL"),
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
                label: Some("Symmetry Field warmed uniforms"),
                size: buffer_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Symmetry Field warmed uniform BG"),
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
                label: Some("Symmetry Field vertex shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/rack_fullscreen.wgsl").into(),
                ),
            });
            let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Symmetry Field fragment shader"),
                source: wgpu::ShaderSource::Wgsl(symmetry_field_fragment_source().into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Symmetry Field pipeline layout"),
                bind_group_layouts: &[
                    Some(&image_layout),
                    Some(&uniform_layout),
                    Some(&motion_layout),
                ],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Symmetry Field dedicated pipeline"),
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
            // Three tiny textures and four views. The image texture carries the
            // D2 view every unbound donor takes AND the single-layer D2Array
            // view an unbound clean history takes, which is why four views cost
            // only three allocations.
            let neutral_image = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Symmetry Field defined-zero image"),
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
            let neutral_image_view =
                neutral_image.create_view(&wgpu::TextureViewDescriptor::default());
            let neutral_history_view = neutral_image.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Symmetry Field defined-zero history array"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                array_layer_count: Some(1),
                ..Default::default()
            });
            let neutral_vectors = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Symmetry Field defined-zero motion vectors"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: crate::renderer::motion::MOTION_VECTOR_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let neutral_vector_view =
                neutral_vectors.create_view(&wgpu::TextureViewDescriptor::default());
            let neutral_gates = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Symmetry Field defined-zero motion gates"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: crate::renderer::motion::MOTION_GATE_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let neutral_gate_view =
                neutral_gates.create_view(&wgpu::TextureViewDescriptor::default());
            SymmetryFieldResources {
                pipeline,
                image_layout,
                motion_layout,
                uniform_buffer,
                uniform_group,
                neutral_image,
                neutral_image_view,
                neutral_history_view,
                neutral_vectors,
                neutral_vector_view,
                neutral_gates,
                neutral_gate_view,
            }
        })?;

        // Explicitly initialize every neutral texture. A missing route binds a
        // defined zero, never uninitialized texture memory.
        write_zero_texel(queue, &resources.neutral_image, 8);
        write_zero_texel(queue, &resources.neutral_vectors, 4);
        write_zero_texel(queue, &resources.neutral_gates, 2);

        allocations.textures.store(3, Ordering::Relaxed);
        allocations.buffers.store(1, Ordering::Relaxed);
        allocations.bind_groups.store(1, Ordering::Relaxed);
        allocations.pipelines.store(1, Ordering::Relaxed);

        Ok(Self {
            dimensions,
            pipeline: resources.pipeline,
            image_layout: resources.image_layout,
            motion_layout: resources.motion_layout,
            uniform_buffer: resources.uniform_buffer,
            uniform_group: resources.uniform_group,
            uniform_stride,
            uniform_slot_capacity,
            _neutral_image: resources.neutral_image,
            neutral_image_view: resources.neutral_image_view,
            neutral_history_view: resources.neutral_history_view,
            _neutral_vectors: resources.neutral_vectors,
            neutral_vector_view: resources.neutral_vector_view,
            _neutral_gates: resources.neutral_gates,
            neutral_gate_view: resources.neutral_gate_view,
            allocations,
        })
    }

    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "executor dimensions are exposed for GPU goldens")
    )]
    pub const fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }

    pub fn allocation_snapshot(&self) -> SymmetryAllocationSnapshot {
        self.allocations.snapshot()
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the aligned stride is exposed for GPU arena goldens"
        )
    )]
    pub const fn uniform_stride(&self) -> u64 {
        self.uniform_stride
    }

    /// Build every node's IMAGE bind groups, both carrier parities, once.
    ///
    /// `carriers` is the caller's ping/pong pair; encode selects one by index
    /// and creates nothing. A route that resolved to no view binds the
    /// executor's own defined-zero neutral, so every one of the four image
    /// bindings is always occupied by a real, initialized texture.
    pub fn prepare_bindings(
        &self,
        device: &wgpu::Device,
        carriers: [&wgpu::TextureView; SYMMETRY_FIELD_CARRIER_PARITIES],
        inputs: &[SymmetryFieldInput<'_>],
    ) -> Result<SymmetryFieldBindings, SymmetryFieldGpuError> {
        validate_node_set(inputs.len(), inputs.iter().map(|input| input.node_id))?;
        let mut ordered = inputs.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|input| input.node_id);
        let entries = create_checked(device, "Symmetry Field binding preparation", || {
            ordered
                .iter()
                .map(|input| {
                    let donor_views = [
                        input.donors[0].unwrap_or(&self.neutral_image_view),
                        input.donors[1].unwrap_or(&self.neutral_image_view),
                    ];
                    let history_view = input.history.unwrap_or(&self.neutral_history_view);
                    let groups = std::array::from_fn(|parity| {
                        create_image_group(
                            device,
                            &self.image_layout,
                            carriers[parity],
                            donor_views,
                            history_view,
                        )
                    });
                    let donor_bound = [input.donors[0].is_some(), input.donors[1].is_some()];
                    SymmetryFieldBinding {
                        node_id: input.node_id,
                        groups,
                        donor_bound,
                        donor_ready: donor_bound,
                        history_bound: input.history.is_some(),
                    }
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })?;
        self.allocations.bind_groups.fetch_add(
            entries.len() as u64 * SYMMETRY_FIELD_CARRIER_PARITIES as u64,
            Ordering::Relaxed,
        );
        Ok(SymmetryFieldBindings { entries })
    }

    /// Build every node's MOTION bind groups — all four committed-parity
    /// combinations — once.
    ///
    /// This is the hand-off the frozen contract's "Bind groups: 2" row could
    /// not express. Each motion slot names a different `MotionGpuField`, each
    /// field publishes through its own `render_field_index`, and the two are
    /// deliberately not required to agree, so every `[slot 0][slot 1]` pair is
    /// prebuilt here and selected by index at encode. A slot with no admitted
    /// field, or an incomplete vector/gate pair, binds the executor's
    /// defined-zero neutral views in every combination.
    pub fn prepare_motion_bindings(
        &self,
        device: &wgpu::Device,
        inputs: &[SymmetryFieldMotionInput<'_>],
    ) -> Result<SymmetryFieldMotionBindings, SymmetryFieldGpuError> {
        validate_node_set(inputs.len(), inputs.iter().map(|input| input.node_id))?;
        let mut ordered = inputs.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|input| input.node_id);
        let entries = create_checked(device, "Symmetry Field motion binding preparation", || {
            ordered
                .iter()
                .map(|input| {
                    let groups = std::array::from_fn(|first| {
                        std::array::from_fn(|second| {
                            let parities = [first, second];
                            let views: [(&wgpu::TextureView, &wgpu::TextureView);
                                SYMMETRY_MOTION_SLOTS] =
                                std::array::from_fn(|slot| match input.motion[slot] {
                                    Some(pair) => {
                                        (pair.vectors[parities[slot]], pair.gates[parities[slot]])
                                    }
                                    None => (&self.neutral_vector_view, &self.neutral_gate_view),
                                });
                            create_motion_group(device, &self.motion_layout, views)
                        })
                    });
                    let motion_bound: [bool; SYMMETRY_MOTION_SLOTS] =
                        std::array::from_fn(|slot| input.motion[slot].is_some());
                    SymmetryFieldMotionBinding {
                        node_id: input.node_id,
                        groups,
                        motion_bound,
                        motion_ready: motion_bound,
                        motion_grid: std::array::from_fn(|slot| {
                            input.motion[slot].map_or([0, 0], |pair| pair.grid)
                        }),
                    }
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })?;
        self.allocations.bind_groups.fetch_add(
            entries.len() as u64 * SYMMETRY_FIELD_MOTION_GROUPS as u64,
            Ordering::Relaxed,
        );
        Ok(SymmetryFieldMotionBindings { entries })
    }

    /// A step whose node contributes nothing at all. The caller may skip the
    /// whole step — the target keeps its carrier untouched — but the executor
    /// never relies on that: encoding an inert node is still exactly the
    /// carrier, byte for byte.
    pub fn is_inert(plan: &EvaluatedSymmetryFieldPlan) -> bool {
        !plan.enabled
            || plan.wet.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
            || plan.params.is_exact_bypass()
    }

    /// Encode exactly one full-frame pass into `target`.
    ///
    /// Unlike a Collision Rack node, this step's result lands in a surface that
    /// is not its carrier, so an exact bypass cannot be expressed by omitting
    /// the pass here. The delegation is real one level up — a bypassed node
    /// collects no image tap, requests no motion field, and owns the frozen
    /// neutral sector table — and the pass itself takes the shader's direct
    /// `textureLoad` branch, which reproduces the carrier bit for bit.
    #[allow(
        clippy::too_many_arguments,
        reason = "the allocation-free encode boundary keeps every borrowed GPU input explicit"
    )]
    pub fn encode_at(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        plan: &EvaluatedSymmetryFieldPlan,
        bindings: &SymmetryFieldBindings,
        motion_bindings: &SymmetryFieldMotionBindings,
        carrier_parity: usize,
        motion_parity: [usize; SYMMETRY_MOTION_SLOTS],
        target: &wgpu::TextureView,
        uniform_slot: usize,
        history: SymmetryHistoryCursor,
        time_seconds: f32,
    ) -> Result<SymmetryFieldEncodeReport, SymmetryFieldGpuError> {
        if carrier_parity >= SYMMETRY_FIELD_CARRIER_PARITIES {
            return Err(SymmetryFieldGpuError::CarrierParity(carrier_parity));
        }
        for (slot, parity) in motion_parity.iter().copied().enumerate() {
            if parity >= SYMMETRY_FIELD_MOTION_PARITIES {
                return Err(SymmetryFieldGpuError::MotionParity { slot, parity });
            }
        }
        if uniform_slot >= self.uniform_slot_capacity {
            return Err(SymmetryFieldGpuError::UniformSlotRange {
                slot: uniform_slot,
                capacity: self.uniform_slot_capacity,
            });
        }
        let binding = bindings
            .find(plan.node_id)
            .ok_or(SymmetryFieldGpuError::UnpreparedNode(plan.node_id))?;
        let motion_binding = motion_bindings
            .find(plan.node_id)
            .ok_or(SymmetryFieldGpuError::UnpreparedMotionNode(plan.node_id))?;
        let allocations_before = self.allocation_snapshot();

        let params = plan.params.values();
        let table = plan.params.sector_table(plan.domain);
        let donor_valid: [bool; SYMMETRY_IMAGE_SLOTS] =
            std::array::from_fn(|slot| binding.donor_bound[slot] && binding.donor_ready[slot]);
        let motion_valid: [bool; SYMMETRY_MOTION_SLOTS] = std::array::from_fn(|slot| {
            motion_binding.motion_bound[slot] && motion_binding.motion_ready[slot]
        });
        let gpu_bindings = SymmetryGpuBindings {
            donor_valid,
            motion_valid,
            motion_grid: motion_binding.motion_grid,
            history_write_index: history.write_index,
            history_valid: if binding.history_bound {
                history.valid
            } else {
                // An unbound ring answers no age at all rather than reading the
                // single neutral layer as if it were materialized history.
                0
            },
        };
        let uniforms = SymmetryGpuUniforms::pack(
            params,
            (self.dimensions[0], self.dimensions[1]),
            &table,
            gpu_bindings,
            SymmetryFrameUniforms {
                wet: plan.wet,
                blend_code: plan.blend.code(),
                time_seconds,
            },
        );
        let offset = self
            .uniform_stride
            .checked_mul(
                u64::try_from(uniform_slot)
                    .map_err(|_| SymmetryFieldGpuError::BufferSizeOverflow)?,
            )
            .ok_or(SymmetryFieldGpuError::BufferSizeOverflow)?;
        queue.write_buffer(&self.uniform_buffer, offset, bytemuck::bytes_of(&uniforms));

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Symmetry Field dedicated pass"),
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
            pass.set_bind_group(0, &binding.groups[carrier_parity], &[]);
            pass.set_bind_group(
                1,
                &self.uniform_group,
                &[u32::try_from(offset).map_err(|_| SymmetryFieldGpuError::BufferSizeOverflow)?],
            );
            // The third group is the committed motion parity pair. Every
            // combination was prebuilt, so this is an index, never a build.
            pass.set_bind_group(
                2,
                &motion_binding.groups[motion_parity[0]][motion_parity[1]],
                &[],
            );
            pass.draw(0..3, 0..1);
        }

        debug_assert_eq!(allocations_before, self.allocation_snapshot());
        let armed_sources = plan.params.values().source_mask;
        let missing_donors = u8::from(armed_sources.donor0 && !donor_valid[0])
            + u8::from(armed_sources.donor1 && !donor_valid[1]);
        let armed_motion = plan.params.values().motion_mask;
        let missing_motion = u8::from(armed_motion.slot0 && !motion_valid[0])
            + u8::from(armed_motion.slot1 && !motion_valid[1]);
        Ok(SymmetryFieldEncodeReport {
            identity: uniforms.meta[3][3] != 0
                || plan.wet.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater),
            missing_donors,
            missing_motion,
            history_bound: binding.history_bound,
        })
    }
}

struct SymmetryFieldResources {
    pipeline: wgpu::RenderPipeline,
    image_layout: wgpu::BindGroupLayout,
    motion_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_group: wgpu::BindGroup,
    neutral_image: wgpu::Texture,
    neutral_image_view: wgpu::TextureView,
    neutral_history_view: wgpu::TextureView,
    neutral_vectors: wgpu::Texture,
    neutral_vector_view: wgpu::TextureView,
    neutral_gates: wgpu::Texture,
    neutral_gate_view: wgpu::TextureView,
}

/// The fragment module: the canonical blend kernel prepended to the dedicated
/// shader, exactly as `renderer::rack` builds its own fragment module.
pub(crate) fn symmetry_field_fragment_source() -> String {
    format!(
        "{}\n{}",
        include_str!("../shaders/blend.wgsl"),
        include_str!("../shaders/symmetry_field.wgsl")
    )
}

/// Every sampled entry is declared non-filterable: this pass owns no sampler,
/// so nothing here may depend on a format's filtering capability.
fn loaded_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn loaded_array_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

/// The bounded, duplicate-free node-set law both prepare entry points share.
fn validate_node_set(
    count: usize,
    node_ids: impl Iterator<Item = NodeId>,
) -> Result<(), SymmetryFieldGpuError> {
    if count > MAX_NODES_PER_RACK {
        return Err(SymmetryFieldGpuError::TooManyNodeBindings {
            requested: count,
            limit: MAX_NODES_PER_RACK,
        });
    }
    let mut seen = BTreeSet::new();
    for node_id in node_ids {
        if !seen.insert(node_id) {
            return Err(SymmetryFieldGpuError::DuplicateNodeBinding(node_id));
        }
    }
    Ok(())
}

fn create_image_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    carrier: &wgpu::TextureView,
    donors: [&wgpu::TextureView; SYMMETRY_IMAGE_SLOTS],
    history: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Symmetry Field image BG"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(carrier),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(donors[0]),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(donors[1]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(history),
            },
        ],
    })
}

fn create_motion_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    motion: [(&wgpu::TextureView, &wgpu::TextureView); SYMMETRY_MOTION_SLOTS],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Symmetry Field motion BG"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(motion[0].0),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(motion[0].1),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(motion[1].0),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(motion[1].1),
            },
        ],
    })
}

/// `align_up(item, alignment)`. The record stays exactly 1,024 bytes; only the
/// distance between two slots grows on a device that demands a wider alignment,
/// which is why 1,024 is never written as a stride literal.
fn aligned_uniform_stride(item_size: u64, alignment: u64) -> Option<u64> {
    let alignment = alignment.max(1);
    Some(item_size.checked_add(alignment - 1)? / alignment * alignment)
}

fn write_zero_texel(queue: &wgpu::Queue, texture: &wgpu::Texture, bytes_per_texel: u32) {
    let zero = [0_u8; 8];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &zero[..bytes_per_texel as usize],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_texel),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

fn create_checked<T>(
    device: &wgpu::Device,
    context: &'static str,
    create: impl FnOnce() -> T,
) -> Result<T, SymmetryFieldGpuError> {
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
        return Err(SymmetryFieldGpuError::ResourceCreation {
            context,
            kind,
            message: error.to_string(),
        });
    }
    Ok(value)
}

/// The dedicated pass's declared node budget, read from the frozen registry
/// rather than restated, so this executor and the rack ledger can never
/// disagree about one node's cost.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the declared ledger is asserted by the dedicated-pass fixtures"
    )
)]
pub(crate) fn symmetry_field_declared_textures() -> u32 {
    u32::from(
        node_kind_descriptor(NodeKindTag::Symmetry)
            .budget
            .sampled_textures_in_pass,
    )
}

const _: () = assert!(SYMMETRY_FIELD_SAMPLED_TEXTURES <= MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS);
// Splitting the pipeline layout moved bindings between groups; it did not
// change how many sampled textures the fragment stage occupies.
const _: () = assert!(
    SYMMETRY_FIELD_IMAGE_TEXTURES + SYMMETRY_FIELD_MOTION_TEXTURES
        == SYMMETRY_FIELD_SAMPLED_TEXTURES
);
const _: () = assert!(SYMMETRY_FIELD_MOTION_GROUPS == 4);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::motion::{MOTION_GATE_FORMAT, MOTION_VECTOR_FORMAT};
    use crate::symmetry::{
        RuntimeSymmetryParams, SymmetryBoundary, SymmetryMode, SymmetryMotionMask,
        SymmetryNodeDomain, SymmetryParams, SymmetrySource, SymmetrySourceMask,
        SYMMETRY_SECTOR_RECORDS,
    };
    use crate::visual_rack::{NodeBlend, VisualScopeId};

    const SHADER: &str = include_str!("../shaders/symmetry_field.wgsl");
    const RACK_SHADER: &str = include_str!("../shaders/rack_node.wgsl");

    fn node_id(value: u64) -> NodeId {
        NodeId::new(value).expect("a non-zero node id")
    }

    fn field_plan(params: RuntimeSymmetryParams, wet: f32) -> EvaluatedSymmetryFieldPlan {
        field_plan_in_scope(VisualScopeId::Master, 7, params, wet)
    }

    fn field_plan_in_scope(
        scope: VisualScopeId,
        id: u64,
        params: RuntimeSymmetryParams,
        wet: f32,
    ) -> EvaluatedSymmetryFieldPlan {
        let id = node_id(id);
        EvaluatedSymmetryFieldPlan {
            node_id: id,
            domain: SymmetryNodeDomain::for_scope(scope, id.get()),
            enabled: true,
            wet,
            blend: NodeBlend::Normal,
            params: params.sanitized(),
            motion_field_slots: [None, None],
            resources: Default::default(),
        }
    }

    /// The whole geometry domain of one mode, armed so no test accidentally
    /// exercises a degenerate configuration, and deliberately off-centre so no
    /// sample of an even grid can land exactly on a sector wall.
    fn armed(mode: SymmetryMode) -> RuntimeSymmetryParams {
        RuntimeSymmetryParams {
            mode,
            base_folds: 3.0,
            fold_offset: 0.0,
            radial_phase_deg: 11.0,
            orbit_phase: 0.0,
            planar_axis_deg: 17.0,
            planar_phase: 0.0,
            cell_skew: 0.15,
            spiral_scale: 0.5,
            orbit_radius: 0.12,
            orbit_spin_deg: 9.0,
            center: [0.4137, 0.5279],
            boundary: SymmetryBoundary::Mirror,
            motion_gain: 0.0,
            hue_span: 0.0,
            ..RuntimeSymmetryParams::default()
        }
    }

    // ---------------------------------------------------------------------
    // Source contracts. These are the only enforcement of the eight-binding,
    // no-sampler, ten-operation law short of an adapter.
    // ---------------------------------------------------------------------

    /// Every one of the eight bindings is declared exactly once, split across
    /// exactly three bind groups — image, uniform, motion — and the pass owns
    /// no sampler at all.
    ///
    /// Three groups is the documented deviation from the frozen table's
    /// "Bind groups: 2". The motion pair is separate so a `MotionGpuField`'s
    /// own committed parity ADDS to the carrier and N-1 parities instead of
    /// multiplying with them. The eight-texture count is unchanged: a fragment
    /// stage's sampled-texture budget is counted across every bound group.
    #[test]
    fn the_symmetry_shader_binds_eight_textures_in_three_groups_and_declares_no_sampler() {
        let declarations = [
            "@group(0) @binding(0) var carrier_tex: texture_2d<f32>;",
            "@group(0) @binding(1) var donor0_tex: texture_2d<f32>;",
            "@group(0) @binding(2) var donor1_tex: texture_2d<f32>;",
            "@group(0) @binding(3) var clean_history_tex: texture_2d_array<f32>;",
            "@group(1) @binding(0) var<uniform> field: SymmetryFieldUniforms;",
            "@group(2) @binding(0) var motion0_vector_tex: texture_2d<f32>;",
            "@group(2) @binding(1) var motion0_gate_tex: texture_2d<f32>;",
            "@group(2) @binding(2) var motion1_vector_tex: texture_2d<f32>;",
            "@group(2) @binding(3) var motion1_gate_tex: texture_2d<f32>;",
        ];
        for declaration in declarations {
            assert_eq!(
                SHADER.matches(declaration).count(),
                1,
                "{declaration} must be declared exactly once"
            );
        }
        assert_eq!(
            SHADER.matches("@group(0) @binding(").count(),
            SYMMETRY_FIELD_IMAGE_TEXTURES as usize,
            "group 0 holds exactly the four image textures"
        );
        assert_eq!(
            SHADER.matches("@group(2) @binding(").count(),
            SYMMETRY_FIELD_MOTION_TEXTURES as usize,
            "group 2 holds exactly the four motion textures"
        );
        assert_eq!(
            SHADER.matches("@group(0) @binding(").count()
                + SHADER.matches("@group(2) @binding(").count(),
            SYMMETRY_FIELD_SAMPLED_TEXTURES as usize,
            "the split moved bindings between groups; it did not change how many \
             sampled textures the fragment stage occupies"
        );
        assert!(
            !SHADER.contains("@group(3)") && !SHADER.contains("@group(1) @binding(1)"),
            "the pipeline layout is exactly three bind groups, one of them the uniform"
        );
        // No sampler means no `sampler` type, no sampler binding, and no
        // sampler-taking builtin anywhere in the pass.
        assert!(!SHADER.contains(": sampler"), "the pass owns no sampler");
        assert!(!SHADER.contains("textureSample"), "every lookup is a load");
        assert!(!SHADER.contains("textureGather"));
    }

    /// One logical lookup is four explicit texture operations. The pass makes
    /// exactly four covered bilinears — dry carrier, carrier source, donor 0,
    /// donor 1, clean history — of which at most two run per pixel, plus
    /// exactly one vector and one gate load per motion slot branch.
    #[test]
    fn the_symmetry_shader_loads_four_corners_per_filter_and_one_texel_per_motion_lane() {
        for texture in [
            "carrier_tex",
            "donor0_tex",
            "donor1_tex",
            "clean_history_tex",
        ] {
            let loads = SHADER.matches(&format!("textureLoad({texture},")).count();
            let expected = if texture == "carrier_tex" {
                // Four filter corners plus the one direct texel the identity
                // and wet-zero branches read.
                5
            } else {
                4
            };
            assert_eq!(loads, expected, "{texture} explicit operations");
        }
        for texture in [
            "motion0_vector_tex",
            "motion0_gate_tex",
            "motion1_vector_tex",
            "motion1_gate_tex",
        ] {
            assert_eq!(
                SHADER.matches(&format!("textureLoad({texture},")).count(),
                1,
                "{texture} is one nearest load, never a filter"
            );
        }
    }

    /// The identity and wet-zero branches read the carrier texel directly, and
    /// they do it before anything else in the fragment can filter.
    #[test]
    fn the_identity_and_wet_zero_branches_load_the_carrier_texel_before_any_filter() {
        let body = SHADER
            .split_once("fn fs_main(")
            .expect("the fragment entry point")
            .1;
        let bypass = body
            .find("return sym_carrier_texel(input.position.xy);")
            .expect("the direct carrier branch");
        let filter = body
            .find("let dry = sym_carrier_linear(input.uv);")
            .expect("the covered dry filter");
        assert!(
            bypass < filter,
            "the direct texel branch must precede every filtered lookup"
        );
        assert!(SHADER.contains("if sym_exact_bypass() || field.frame.x <= 0.0 {"));
        let direct = SHADER
            .split_once("fn sym_carrier_texel(")
            .expect("the direct texel helper")
            .1;
        let direct = &direct[..direct.find("\n}\n").expect("a top level body")];
        assert!(
            !direct.contains("mix(") && !direct.contains("sym_straight_from_premultiplied"),
            "the direct texel path must not convert through premultiplied space"
        );
    }

    /// A clean-history age is guarded before it can name a ring layer, and the
    /// guard textually precedes every array load, exactly as the legacy
    /// temporal shader's ordering is asserted.
    #[test]
    fn every_clean_history_age_is_guarded_before_the_ring_layer_is_read() {
        let guard = SHADER
            .find("return age != 0u && age < sym_history_valid();")
            .expect("the materialized-age guard");
        let load = SHADER
            .find("textureLoad(clean_history_tex,")
            .expect("a ring layer load");
        assert!(guard < load, "the guard precedes the first ring load");
        assert!(
            SHADER.contains("fn sym_history_layer(age: u32) -> i32 {"),
            "the age to layer law is one named function"
        );
        // The ring index law is the temporal shader's, and the layer is
        // additionally clamped into the real array.
        assert!(SHADER.contains("sym_rem_euclid(sym_history_write() - f32(age), history_len)"));
        assert!(SHADER.contains("return clamp(i32(raw), 0, i32(history_len) - 1);"));
    }

    /// One hue rotation law, two copies. The Symmetry Field's HSL round trip is
    /// character identical to the Collision Rack's.
    #[test]
    fn the_symmetry_and_rack_shaders_share_one_character_identical_hue_rotation() {
        for signature in ["fn rgb_to_hsl(", "fn hue_to_rgb(", "fn hsl_to_rgb("] {
            let mine = wgsl_function(SHADER, signature);
            let theirs = wgsl_function(RACK_SHADER, signature);
            assert_eq!(mine, theirs, "{signature} must stay one law");
        }
        assert!(SHADER.contains("if turns == 0.0 { return color; }"));
    }

    /// Every mode and boundary code in the shader is the append-only Rust code.
    #[test]
    fn the_symmetry_shader_mirrors_every_mode_boundary_and_source_code() {
        let modes = [
            ("SYM_CYCLIC", SymmetryMode::Cyclic),
            ("SYM_DIHEDRAL", SymmetryMode::Dihedral),
            ("SYM_PLANAR_P1", SymmetryMode::PlanarP1),
            ("SYM_PLANAR_PM", SymmetryMode::PlanarPm),
            ("SYM_PLANAR_P2", SymmetryMode::PlanarP2),
            ("SYM_PLANAR_PMM", SymmetryMode::PlanarPmm),
            ("SYM_LOG_SPIRAL", SymmetryMode::LogSpiral),
            ("SYM_ORBIT", SymmetryMode::Orbit),
        ];
        for (name, mode) in modes {
            assert!(
                SHADER.contains(&format!("const {name}: u32 = {}u;", mode.code())),
                "{name} must carry code {}",
                mode.code()
            );
        }
        let boundaries = [
            ("SYM_BOUNDARY_TRANSPARENT", SymmetryBoundary::Transparent),
            ("SYM_BOUNDARY_MIRROR", SymmetryBoundary::Mirror),
            ("SYM_BOUNDARY_WRAP", SymmetryBoundary::Wrap),
            ("SYM_BOUNDARY_HOLD", SymmetryBoundary::Hold),
            (
                "SYM_BOUNDARY_CELLULAR_REENTRY",
                SymmetryBoundary::CellularReentry,
            ),
        ];
        for (name, boundary) in boundaries {
            assert!(
                SHADER.contains(&format!("const {name}: u32 = {}u;", boundary.code())),
                "{name} must carry code {}",
                boundary.code()
            );
        }
        let sources = [
            ("SYM_SOURCE_CARRIER", SymmetrySource::Carrier),
            ("SYM_SOURCE_DONOR0", SymmetrySource::Donor0),
            ("SYM_SOURCE_DONOR1", SymmetrySource::Donor1),
            ("SYM_SOURCE_CLEAN_HISTORY", SymmetrySource::CleanHistory),
        ];
        for (name, source) in sources {
            assert!(
                SHADER.contains(&format!("const {name}: u32 = {}u;", source.code())),
                "{name} must carry code {}",
                source.code()
            );
        }
        assert!(SHADER.contains(&format!(
            "const SYM_SECTOR_RECORDS: u32 = {SYMMETRY_SECTOR_RECORDS}u;"
        )));
        assert!(SHADER.contains("sectors: array<vec4u, 32>,"));
        assert!(SHADER.contains("reserved: array<vec4u, 14>,"));
    }

    /// The shader's motion offset is a reference tick, never wall clock, and the
    /// constant it uses is the ring's own cadence.
    #[test]
    fn the_symmetry_motion_offset_is_scaled_by_one_reference_tick_and_never_by_wall_time() {
        assert_eq!(SYMMETRY_MOTION_REFERENCE_SECONDS, 1.0 / 30.0);
        let declared = SHADER
            .split_once("const SYM_MOTION_REFERENCE_SECONDS: f32 = ")
            .expect("the reference tick constant")
            .1;
        let declared: f32 = declared[..declared.find(';').expect("a terminated constant")]
            .parse()
            .expect("a float literal");
        assert!(
            (declared - SYMMETRY_MOTION_REFERENCE_SECONDS).abs() <= 1.0e-7,
            "{declared} must be one reference tick"
        );
        assert!(SHADER.contains("* SYM_MOTION_REFERENCE_SECONDS"));
        assert!(
            !SHADER.contains("field.frame.y *"),
            "program time must not scale the motion offset"
        );
    }

    /// The dedicated ledger is the frozen node descriptor, and the executor
    /// declares exactly what the descriptor promises.
    #[test]
    fn the_dedicated_pass_declares_one_pass_four_lookups_ten_operations_and_eight_textures() {
        let budget = node_kind_descriptor(NodeKindTag::Symmetry).budget;
        assert_eq!(budget.full_frame_passes, 1);
        assert_eq!(budget.logical_texture_lookups_per_pixel, 4);
        assert_eq!(budget.texture_samples_per_pixel, 10);
        assert_eq!(budget.sampled_textures_in_pass, 8);
        assert_eq!(budget.cross_input_taps, 2);
        assert_eq!(symmetry_field_declared_textures(), 8);
        assert_eq!(
            SYMMETRY_FIELD_SAMPLED_TEXTURES,
            u32::from(budget.sampled_textures_in_pass)
        );
        // Ten explicit operations is two covered bilinears plus one vector and
        // one gate load, not four bilinears.
        assert_eq!(
            u32::from(budget.texture_samples_per_pixel),
            2 * u32::from(crate::visual_rack::PREMULTIPLIED_BILINEAR_TEXTURE_OPS) + 2
        );
        const {
            assert!(SYMMETRY_FIELD_SAMPLED_TEXTURES <= MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS);
            // The dedicated ceiling is exactly why this pass cannot ride the
            // fixed three-texture Collision Rack layout.
            assert!(
                SYMMETRY_FIELD_SAMPLED_TEXTURES > crate::visual_rack::MAX_SAMPLED_TEXTURES_PER_PASS
            );
        }
    }

    /// The arena strides by the aligned record, never by a hardcoded kilobyte.
    #[test]
    fn the_uniform_arena_strides_by_the_kilobyte_record_aligned_to_the_device() {
        assert_eq!(SymmetryGpuUniforms::BYTES, 1_024);
        for (alignment, expected) in [
            (1_u64, 1_024_u64),
            (64, 1_024),
            (256, 1_024),
            (512, 1_024),
            (1_024, 1_024),
            (2_048, 2_048),
        ] {
            assert_eq!(
                aligned_uniform_stride(SymmetryGpuUniforms::BYTES, alignment),
                Some(expected),
                "alignment {alignment}"
            );
        }
    }

    /// The exact default delegates: the caller may skip the step entirely, and
    /// a live wet of zero or a disabled node delegates the same way.
    #[test]
    fn an_exact_default_a_disabled_node_and_a_wet_of_zero_are_all_inert_steps() {
        let default = field_plan(RuntimeSymmetryParams::default(), 1.0);
        assert!(SymmetryFieldExecutor::is_inert(&default));
        let mut disabled = field_plan(armed(SymmetryMode::Dihedral), 1.0);
        disabled.enabled = false;
        assert!(SymmetryFieldExecutor::is_inert(&disabled));
        assert!(SymmetryFieldExecutor::is_inert(&field_plan(
            armed(SymmetryMode::Dihedral),
            0.0
        )));
        assert!(SymmetryFieldExecutor::is_inert(&field_plan(
            armed(SymmetryMode::Dihedral),
            f32::NAN
        )));
        // An armed node is emphatically not inert.
        assert!(!SymmetryFieldExecutor::is_inert(&field_plan(
            armed(SymmetryMode::Dihedral),
            1.0
        )));
    }

    /// A sector table is a function of authored identity alone, and the packed
    /// record reflects a lost donor only in its validity lane.
    #[test]
    fn losing_a_donor_moves_only_the_validity_lane_of_the_packed_record() {
        let mut params = armed(SymmetryMode::Dihedral);
        params.source_mask = SymmetrySourceMask {
            carrier: true,
            donor0: true,
            donor1: true,
            clean_history: true,
        };
        params.motion_mask = SymmetryMotionMask {
            slot0: true,
            slot1: false,
        };
        let plan = field_plan(params, 1.0);
        let table = plan.params.sector_table(plan.domain);
        let bound = SymmetryGpuUniforms::pack(
            plan.params.values(),
            (64, 64),
            &table,
            SymmetryGpuBindings {
                donor_valid: [true, true],
                motion_valid: [true, false],
                motion_grid: [[8, 8], [0, 0]],
                history_write_index: 5,
                history_valid: 24,
            },
            SymmetryFrameUniforms::default(),
        );
        let lost = SymmetryGpuUniforms::pack(
            plan.params.values(),
            (64, 64),
            &table,
            SymmetryGpuBindings {
                donor_valid: [false, true],
                motion_valid: [false, false],
                motion_grid: [[0, 0], [0, 0]],
                history_write_index: 5,
                history_valid: 24,
            },
            SymmetryFrameUniforms::default(),
        );
        assert_eq!(bound.sectors, lost.sectors);
        assert_eq!(bound.meta[0], lost.meta[0]);
        assert_eq!(bound.meta[1], lost.meta[1]);
        assert_eq!(bound.params, lost.params);
        assert_ne!(bound.meta[2], lost.meta[2]);
    }

    // ---------------------------------------------------------------------
    // The CPU reference. It is the stage-one geometry reference plus the
    // shader's own covered filter, written independently here so a GPU fixture
    // is never compared against a golden of itself.
    // ---------------------------------------------------------------------

    struct RefImage {
        dimensions: [u32; 2],
        pixels: Vec<[f32; 4]>,
    }

    impl RefImage {
        fn load(&self, x: i32, y: i32) -> [f32; 4] {
            let x = x.clamp(0, self.dimensions[0] as i32 - 1) as usize;
            let y = y.clamp(0, self.dimensions[1] as i32 - 1) as usize;
            self.pixels[y * self.dimensions[0] as usize + x]
        }

        /// `rack_node.wgsl`'s law, transplanted: clamped corners, covered in
        /// premultiplied space, mixed there, then converted back to straight.
        fn covered_bilinear(&self, uv: [f32; 2]) -> [f32; 4] {
            let cx = uv[0] * self.dimensions[0] as f32 - 0.5;
            let cy = uv[1] * self.dimensions[1] as f32 - 0.5;
            let bx = cx.floor();
            let by = cy.floor();
            let fx = cx - bx;
            let fy = cy - by;
            let cover = |value: [f32; 4]| {
                let alpha = value[3].clamp(0.0, 1.0);
                [value[0] * alpha, value[1] * alpha, value[2] * alpha, alpha]
            };
            let c00 = cover(self.load(bx as i32, by as i32));
            let c10 = cover(self.load(bx as i32 + 1, by as i32));
            let c01 = cover(self.load(bx as i32, by as i32 + 1));
            let c11 = cover(self.load(bx as i32 + 1, by as i32 + 1));
            let mut mixed = [0.0_f32; 4];
            for channel in 0..4 {
                let top = c00[channel] + (c10[channel] - c00[channel]) * fx;
                let bottom = c01[channel] + (c11[channel] - c01[channel]) * fx;
                mixed[channel] = top + (bottom - top) * fy;
            }
            let alpha = mixed[3].clamp(0.0, 1.0);
            if alpha <= 1.0e-6 {
                return [0.0; 4];
            }
            [mixed[0] / alpha, mixed[1] / alpha, mixed[2] / alpha, alpha]
        }
    }

    /// The UV of one output pixel under the shared fullscreen triangle, whose
    /// origin is top left.
    fn pixel_uv(column: u32, row: u32, dimensions: [u32; 2]) -> [f32; 2] {
        [
            (column as f32 + 0.5) / dimensions[0] as f32,
            (row as f32 + 0.5) / dimensions[1] as f32,
        ]
    }

    /// The expected pixel for a carrier-only, hue-free, motion-free field:
    /// fold, then filter, then let the boundary own coverage.
    fn symmetry_reference(
        carrier: &RefImage,
        params: SymmetryParams,
        uv: [f32; 2],
        dimensions: [u32; 2],
    ) -> [f32; 4] {
        let folded = params.fold(uv, (dimensions[0], dimensions[1]));
        if !folded.covered {
            return [0.0; 4];
        }
        carrier.covered_bilinear(folded.uv)
    }

    /// A pixel is comparable only where the fold is locally stable. A sample
    /// sitting on a sector wall can be classified either way by a one-ulp
    /// difference between two independent implementations, so the fixtures
    /// compare the interior and count how much of the frame that was.
    fn fold_is_locally_stable(params: SymmetryParams, uv: [f32; 2], dimensions: [u32; 2]) -> bool {
        const PROBE: f32 = 2.0e-4;
        let center = params.fold(uv, (dimensions[0], dimensions[1]));
        for offset in [[PROBE, 0.0], [-PROBE, 0.0], [0.0, PROBE], [0.0, -PROBE]] {
            let probed = params.fold(
                [uv[0] + offset[0], uv[1] + offset[1]],
                (dimensions[0], dimensions[1]),
            );
            if probed.sector != center.sector || probed.covered != center.covered {
                return false;
            }
            if (probed.uv[0] - center.uv[0]).abs() > 4.0e-3
                || (probed.uv[1] - center.uv[1]).abs() > 4.0e-3
            {
                return false;
            }
        }
        true
    }

    fn wgsl_function<'a>(source: &'a str, signature: &str) -> &'a str {
        let start = source
            .find(signature)
            .unwrap_or_else(|| panic!("{signature} is missing"));
        let body = &source[start..];
        let end = body
            .find("\n}\n")
            .expect("a top level body ends at column zero");
        &body[..end]
    }

    // ---------------------------------------------------------------------
    // Physical GPU fixtures.
    // ---------------------------------------------------------------------

    /// One node's three-group binding set: the image groups (both carrier
    /// parities), the motion groups (every committed-parity combination), and
    /// the committed parity each motion slot reads this frame.
    struct TestBindings {
        image: SymmetryFieldBindings,
        motion: SymmetryFieldMotionBindings,
        motion_parity: [usize; SYMMETRY_MOTION_SLOTS],
    }

    /// Prepare both halves the way the composition does: the image group per
    /// carrier parity, the motion group once per node.
    fn prepare_test_bindings(
        device: &wgpu::Device,
        executor: &SymmetryFieldExecutor,
        carriers: [&wgpu::TextureView; SYMMETRY_FIELD_CARRIER_PARITIES],
        node_id: NodeId,
        donors: [Option<&wgpu::TextureView>; SYMMETRY_IMAGE_SLOTS],
        history: Option<&wgpu::TextureView>,
        motion: [Option<SymmetryMotionViews<'_>>; SYMMETRY_MOTION_SLOTS],
    ) -> TestBindings {
        TestBindings {
            image: executor
                .prepare_bindings(
                    device,
                    carriers,
                    &[SymmetryFieldInput {
                        node_id,
                        donors,
                        history,
                    }],
                )
                .expect("image bindings"),
            motion: executor
                .prepare_motion_bindings(device, &[SymmetryFieldMotionInput { node_id, motion }])
                .expect("motion bindings"),
            motion_parity: [0, 0],
        }
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
                .expect("GPU adapter for Symmetry Field test");
            // Byte for byte the production request at renderer/state.rs:2892.
            // wgpu validates against the REQUESTED limits, so this is what makes
            // the eight-texture claim a floor claim and not an adapter claim.
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("Symmetry Field test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }))
                .expect("GPU device for Symmetry Field test");
            Self { device, queue }
        }

        fn image(
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
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC,
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

        fn motion_texture(
            &self,
            format: wgpu::TextureFormat,
            grid: [u32; 2],
            texels: &[u8],
            bytes_per_texel: u32,
            label: &'static str,
        ) -> (wgpu::Texture, wgpu::TextureView) {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: grid[0],
                    height: grid[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
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
                texels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(grid[0] * bytes_per_texel),
                    rows_per_image: Some(grid[1]),
                },
                wgpu::Extent3d {
                    width: grid[0],
                    height: grid[1],
                    depth_or_array_layers: 1,
                },
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        }

        #[allow(clippy::too_many_arguments)]
        fn render(
            &self,
            executor: &SymmetryFieldExecutor,
            plan: &EvaluatedSymmetryFieldPlan,
            bindings: &TestBindings,
            carrier_parity: usize,
            target: &wgpu::Texture,
            target_view: &wgpu::TextureView,
            dimensions: [u32; 2],
            history: SymmetryHistoryCursor,
        ) -> (SymmetryFieldEncodeReport, Vec<u8>) {
            let unpadded_row = dimensions[0] * 8;
            let padded_row = (unpadded_row + 255) & !255;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Symmetry Field test readback"),
                size: u64::from(padded_row) * u64::from(dimensions[1]),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Symmetry Field test encoder"),
                });
            let report = executor
                .encode_at(
                    &self.queue,
                    &mut encoder,
                    plan,
                    &bindings.image,
                    &bindings.motion,
                    carrier_parity,
                    bindings.motion_parity,
                    target_view,
                    0,
                    history,
                    0.0,
                )
                .expect("the dedicated pass encodes");
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: target,
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
            let mut compacted = Vec::with_capacity(unpadded_row as usize * dimensions[1] as usize);
            for row in 0..dimensions[1] as usize {
                let start = row * padded_row as usize;
                compacted.extend_from_slice(&mapped[start..start + unpadded_row as usize]);
            }
            drop(mapped);
            staging.unmap();
            (report, compacted)
        }
    }

    fn decode_pixels(bytes: &[u8]) -> Vec<[f32; 4]> {
        bytes
            .chunks_exact(8)
            .map(|chunk| {
                std::array::from_fn(|channel| {
                    f16_to_f32(u16::from_le_bytes([
                        chunk[channel * 2],
                        chunk[channel * 2 + 1],
                    ]))
                })
            })
            .collect()
    }

    /// A carrier the covered filter cannot reproduce.
    ///
    /// Every fourth texel is transparent while still carrying hidden RGB, and
    /// the rest carry fractional alpha, so `straight_from_premultiplied`
    /// collapses the transparent ones to zero and rounds the rest. Only a
    /// direct `textureLoad` of the exact texel can come back byte identical,
    /// which is what makes the identity fixture discriminating rather than
    /// accidentally true for an opaque image at exact pixel centres.
    fn hostile_carrier_pixels(dimensions: [u32; 2]) -> Vec<[f32; 4]> {
        let mut pixels = Vec::with_capacity(dimensions[0] as usize * dimensions[1] as usize);
        for row in 0..dimensions[1] {
            for column in 0..dimensions[0] {
                let index = column + row * dimensions[0];
                let alpha = match index % 4 {
                    0 => 1.0,
                    1 => 0.5,
                    2 => 0.125,
                    _ => 0.0,
                };
                pixels.push([
                    0.9 - 0.05 * (column % 5) as f32,
                    0.3 + 0.07 * (row % 5) as f32,
                    0.7,
                    alpha,
                ]);
            }
        }
        pixels
    }

    /// A deterministic, high-contrast carrier so a wrong fold cannot look right.
    fn carrier_pixels(dimensions: [u32; 2]) -> Vec<[f32; 4]> {
        let mut pixels = Vec::with_capacity(dimensions[0] as usize * dimensions[1] as usize);
        for row in 0..dimensions[1] {
            for column in 0..dimensions[0] {
                let u = (column as f32 + 0.5) / dimensions[0] as f32;
                let v = (row as f32 + 0.5) / dimensions[1] as f32;
                pixels.push([u, v, ((column + row) % 2) as f32 * 0.75, 1.0]);
            }
        }
        pixels
    }

    /// Acceptance item: the exact default renders its carrier byte for byte.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_symmetry_field_default_readback_is_bit_identical_to_its_carrier() {
        let gpu = GpuHarness::new();
        let dimensions = [8, 8];
        let executor =
            SymmetryFieldExecutor::new(&gpu.device, &gpu.queue, dimensions, 1).expect("executor");
        let pixels = hostile_carrier_pixels(dimensions);
        // The covered filter would collapse every transparent texel to zero and
        // round every fractional-alpha one, so this carrier can only survive a
        // direct texel load.
        assert!(pixels.iter().any(|pixel| pixel[3] == 0.0 && pixel[0] > 0.0));
        assert!(pixels.iter().any(|pixel| pixel[3] > 0.0 && pixel[3] < 1.0));
        let (_carrier, carrier_view, carrier_bytes) =
            gpu.image(dimensions, &pixels, "Symmetry Field carrier");
        let (spare, spare_view, _) = gpu.image(dimensions, &pixels, "Symmetry Field spare");
        let (target, target_view, _) = gpu.image(
            dimensions,
            &vec![[0.0; 4]; pixels.len()],
            "Symmetry Field target",
        );
        let _ = (&spare, &spare_view);
        let plan = field_plan(RuntimeSymmetryParams::default(), 1.0);
        assert!(
            SymmetryFieldExecutor::is_inert(&plan),
            "the exact default is a delegation the caller may skip entirely"
        );
        let bindings = prepare_test_bindings(
            &gpu.device,
            &executor,
            [&carrier_view, &spare_view],
            plan.node_id,
            [None, None],
            None,
            [None, None],
        );
        let (report, bytes) = gpu.render(
            &executor,
            &plan,
            &bindings,
            0,
            &target,
            &target_view,
            dimensions,
            SymmetryHistoryCursor::default(),
        );
        assert!(report.identity, "the record declares the identity");
        assert_eq!(
            bytes, carrier_bytes,
            "the identity branch must reproduce the carrier byte for byte"
        );

        // A wet of zero takes the same direct branch on an otherwise fully
        // armed node, and is equally byte identical.
        let wet_zero = field_plan(armed(SymmetryMode::Dihedral), 0.0);
        let wet_bindings = prepare_test_bindings(
            &gpu.device,
            &executor,
            [&carrier_view, &spare_view],
            wet_zero.node_id,
            [None, None],
            None,
            [None, None],
        );
        let (wet_report, wet_bytes) = gpu.render(
            &executor,
            &wet_zero,
            &wet_bindings,
            0,
            &target,
            &target_view,
            dimensions,
            SymmetryHistoryCursor::default(),
        );
        assert!(wet_report.identity);
        assert_eq!(wet_bytes, carrier_bytes);
    }

    /// Acceptance item: the shader reproduces the stage-one CPU group reference
    /// for every mode, within the established binary16 tolerance, wherever the
    /// fold is locally stable.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_symmetry_field_matches_the_cpu_group_reference_for_every_mode() {
        let gpu = GpuHarness::new();
        // Deliberately NON-SQUARE, so the output-aspect conjugation that keeps
        // an authored angle physical is load bearing in this fixture.
        let dimensions = [24, 12];
        let executor =
            SymmetryFieldExecutor::new(&gpu.device, &gpu.queue, dimensions, 1).expect("executor");
        let pixels = carrier_pixels(dimensions);
        let carrier = RefImage {
            dimensions,
            pixels: pixels.clone(),
        };
        let (_carrier_texture, carrier_view, _) =
            gpu.image(dimensions, &pixels, "Symmetry Field carrier");
        let (spare, spare_view, _) = gpu.image(dimensions, &pixels, "Symmetry Field spare");
        let (target, target_view, _) = gpu.image(
            dimensions,
            &vec![[0.0; 4]; pixels.len()],
            "Symmetry Field target",
        );
        let _ = &spare;
        for mode in SymmetryMode::ALL {
            for boundary in SymmetryBoundary::ALL {
                let mut runtime = armed(mode);
                runtime.boundary = boundary;
                let plan = field_plan(runtime, 1.0);
                assert!(
                    !SymmetryFieldExecutor::is_inert(&plan),
                    "{mode:?} with {boundary:?} must be an active pass"
                );
                let params = plan.params.values();
                let bindings = prepare_test_bindings(
                    &gpu.device,
                    &executor,
                    [&carrier_view, &spare_view],
                    plan.node_id,
                    [None, None],
                    None,
                    [None, None],
                );
                let (_, bytes) = gpu.render(
                    &executor,
                    &plan,
                    &bindings,
                    0,
                    &target,
                    &target_view,
                    dimensions,
                    SymmetryHistoryCursor::default(),
                );
                let rendered = decode_pixels(&bytes);
                let mut compared = 0_usize;
                for row in 0..dimensions[1] {
                    for column in 0..dimensions[0] {
                        let uv = pixel_uv(column, row, dimensions);
                        if !fold_is_locally_stable(params, uv, dimensions) {
                            continue;
                        }
                        compared += 1;
                        let expected = symmetry_reference(&carrier, params, uv, dimensions);
                        let actual = rendered[(row * dimensions[0] + column) as usize];
                        for channel in 0..4 {
                            assert!(
                                (actual[channel] - expected[channel]).abs() <= 0.01,
                                "{mode:?}/{boundary:?} pixel ({column},{row}) channel {channel}: \
                                 {} vs {}",
                                actual[channel],
                                expected[channel]
                            );
                        }
                    }
                }
                let total = (dimensions[0] * dimensions[1]) as usize;
                assert!(
                    compared * 2 >= total,
                    "{mode:?}/{boundary:?} compared only {compared} of {total} pixels; the fixture \
                     must exercise the interior, not just a handful of stable samples"
                );
            }
        }
    }

    /// Acceptance item: a missing donor and an incomplete motion pair bind the
    /// neutral views and change nothing at all.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_symmetry_field_missing_donor_and_incomplete_motion_pair_bind_neutral_and_change_nothing()
    {
        let gpu = GpuHarness::new();
        let dimensions = [16, 16];
        let executor =
            SymmetryFieldExecutor::new(&gpu.device, &gpu.queue, dimensions, 1).expect("executor");
        let pixels = carrier_pixels(dimensions);
        let (_carrier_texture, carrier_view, _) =
            gpu.image(dimensions, &pixels, "Symmetry Field carrier");
        let (spare, spare_view, _) = gpu.image(dimensions, &pixels, "Symmetry Field spare");
        let (target, target_view, _) = gpu.image(
            dimensions,
            &vec![[0.0; 4]; pixels.len()],
            "Symmetry Field target",
        );
        let _ = &spare;

        // Every source and both motion slots are armed, so the sector table
        // genuinely names donors, history and motion in some sectors.
        let mut runtime = armed(SymmetryMode::Dihedral);
        runtime.source_mask = SymmetrySourceMask {
            carrier: true,
            donor0: true,
            donor1: true,
            clean_history: true,
        };
        runtime.motion_mask = SymmetryMotionMask {
            slot0: true,
            slot1: true,
        };
        runtime.motion_gain = 1.0;
        let plan = field_plan(runtime, 1.0);
        let table = plan.params.sector_table(plan.domain);
        assert!(
            table
                .records()
                .iter()
                .any(|record| record.source != SymmetrySource::Carrier),
            "the armed table must name at least one non-carrier source"
        );
        assert!(
            table.records().iter().any(|record| record.motion.is_some()),
            "the armed table must name at least one motion slot"
        );

        let carrier_only = |slot_views: [Option<&wgpu::TextureView>; 2],
                            motion: [Option<SymmetryMotionViews<'_>>; 2],
                            history: Option<&wgpu::TextureView>| {
            prepare_test_bindings(
                &gpu.device,
                &executor,
                [&carrier_view, &spare_view],
                plan.node_id,
                slot_views,
                history,
                motion,
            )
        };

        let nothing_bound = carrier_only([None, None], [None, None], None);
        let (missing_report, missing_bytes) = gpu.render(
            &executor,
            &plan,
            &nothing_bound,
            0,
            &target,
            &target_view,
            dimensions,
            SymmetryHistoryCursor::default(),
        );
        assert_eq!(missing_report.missing_donors, 2);
        assert_eq!(missing_report.missing_motion, 2);
        assert!(!missing_report.history_bound);

        // Half a motion pair is not a pair. Supplying only the vectors would be
        // applying an ungated field at full confidence, so the whole slot is
        // expressed as absent and binds the neutral pair.
        let (_vectors, vector_view) = gpu.motion_texture(
            MOTION_VECTOR_FORMAT,
            [4, 4],
            &[0_u8; 4 * 4 * 4],
            4,
            "Symmetry Field test motion vectors",
        );
        let (_gates, gate_view) = gpu.motion_texture(
            MOTION_GATE_FORMAT,
            [4, 4],
            &[0_u8; 4 * 4 * 2],
            2,
            "Symmetry Field test motion gates",
        );
        let zero_motion = carrier_only(
            [None, None],
            [
                Some(SymmetryMotionViews {
                    vectors: [&vector_view, &vector_view],
                    gates: [&gate_view, &gate_view],
                    grid: [4, 4],
                }),
                None,
            ],
            None,
        );
        let (_zero_report, zero_bytes) = gpu.render(
            &executor,
            &plan,
            &zero_motion,
            0,
            &target,
            &target_view,
            dimensions,
            SymmetryHistoryCursor::default(),
        );
        assert_eq!(
            missing_bytes, zero_bytes,
            "a fully closed gate must displace by exactly nothing, byte for byte"
        );

        // The same authored node with a carrier-only table renders identically,
        // because an unbound source resolves to the carrier before the filter.
        let mut carrier_runtime = armed(SymmetryMode::Dihedral);
        carrier_runtime.motion_gain = 1.0;
        let carrier_plan = field_plan(carrier_runtime, 1.0);
        let carrier_bindings = prepare_test_bindings(
            &gpu.device,
            &executor,
            [&carrier_view, &spare_view],
            carrier_plan.node_id,
            [None, None],
            None,
            [None, None],
        );
        let (carrier_report, carrier_bytes) = gpu.render(
            &executor,
            &carrier_plan,
            &carrier_bindings,
            0,
            &target,
            &target_view,
            dimensions,
            SymmetryHistoryCursor::default(),
        );
        assert_eq!(carrier_report.missing_donors, 0);
        assert_eq!(carrier_report.missing_motion, 0);
        assert_eq!(
            missing_bytes, carrier_bytes,
            "losing every donor must land exactly on the carrier-only render"
        );
    }

    /// Acceptance item: live/export equality on real hardware.
    ///
    /// The same authored node is planned twice — once under a live
    /// process-lifetime layer identity (the process-global counter has already
    /// handed out 909 identities) and once under export's `position + 1`
    /// identity — and each payload is packed and rendered independently through
    /// its own bindings. Sharing one plan between the two consumers would make
    /// the proof tautological, so both derive their own
    /// `SymmetryNodeDomain::for_scope`.
    ///
    /// The control at the end renders a genuinely different authored node and
    /// asserts the frame moves, so the equality above cannot pass by the shader
    /// simply ignoring the sector table.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_symmetry_field_live_and_export_layer_identities_render_byte_identical_frames() {
        let gpu = GpuHarness::new();
        let dimensions = [16, 16];
        let executor =
            SymmetryFieldExecutor::new(&gpu.device, &gpu.queue, dimensions, 1).expect("executor");
        let pixels = carrier_pixels(dimensions);
        let (_carrier, carrier_view, _) = gpu.image(dimensions, &pixels, "Symmetry Field carrier");
        let (_spare, spare_view, _) = gpu.image(dimensions, &pixels, "Symmetry Field spare");
        let (target, target_view, _) = gpu.image(
            dimensions,
            &vec![[0.0; 4]; pixels.len()],
            "Symmetry Field target",
        );

        // Armed so the sector table is genuinely load bearing: a non-zero hue
        // span makes every record's hue offset reach the pixels, whatever
        // source that record names.
        let mut runtime = armed(SymmetryMode::Dihedral);
        runtime.source_mask = SymmetrySourceMask {
            carrier: true,
            donor0: true,
            donor1: false,
            clean_history: true,
        };
        runtime.hue_span = 0.75;
        runtime.seed = 0x0BAD_5EED;

        let layer = |id: u64| {
            VisualScopeId::Layer(crate::image_routing::StableLayerId::new(id).expect("a live id"))
        };
        let export_plan = field_plan_in_scope(layer(1), 7, runtime, 1.0);
        let live_plan = field_plan_in_scope(layer(910), 7, runtime, 1.0);
        assert_ne!(
            export_plan.params.sector_table(export_plan.domain),
            crate::symmetry::SymmetrySectorTable::NEUTRAL,
            "an armed node must not fall back to the neutral table"
        );

        let render = |plan: &EvaluatedSymmetryFieldPlan| {
            let bindings = prepare_test_bindings(
                &gpu.device,
                &executor,
                [&carrier_view, &spare_view],
                plan.node_id,
                [Some(&spare_view), None],
                None,
                [None, None],
            );
            gpu.render(
                &executor,
                plan,
                &bindings,
                0,
                &target,
                &target_view,
                dimensions,
                SymmetryHistoryCursor::default(),
            )
            .1
        };

        let export_bytes = render(&export_plan);
        let live_bytes = render(&live_plan);
        assert_eq!(
            live_bytes, export_bytes,
            "the offline render of an authored patch must reproduce the live \
             program byte for byte; a process-lifetime layer ID cannot reach \
             the sector table"
        );

        // Control: a different authored node id is a different table and must
        // move the frame, so the equality above is not vacuous.
        let other_plan = field_plan_in_scope(layer(1), 8, runtime, 1.0);
        assert_ne!(
            other_plan.params.sector_table(other_plan.domain),
            export_plan.params.sector_table(export_plan.domain)
        );
        assert_ne!(
            render(&other_plan),
            export_bytes,
            "the shader must actually read the sector table"
        );
    }

    /// Acceptance item: a warmed encode allocates nothing, and both prepared
    /// carrier parities are real — the same carrier bound at either parity
    /// renders byte-identically.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_symmetry_field_warm_encode_allocates_nothing_and_both_carrier_parities_agree() {
        let gpu = GpuHarness::new();
        let dimensions = [16, 16];
        let executor =
            SymmetryFieldExecutor::new(&gpu.device, &gpu.queue, dimensions, 4).expect("executor");
        let warm = executor.allocation_snapshot();
        assert_eq!(
            warm,
            SymmetryAllocationSnapshot {
                textures: 3,
                buffers: 1,
                bind_groups: 1,
                pipelines: 1,
            },
            "three tiny neutral textures, one arena, one uniform bind group, one pipeline"
        );
        assert!(executor.uniform_stride() >= SymmetryGpuUniforms::BYTES);
        assert_eq!(executor.dimensions(), dimensions);

        let pixels = carrier_pixels(dimensions);
        // Both parities hold the SAME pixels, so a parity that bound the wrong
        // view would still be caught by the byte comparison only if the pass
        // truly reads through the selected group; give them different content.
        let mut swapped = pixels.clone();
        swapped.reverse();
        let (_first, first_view, _) = gpu.image(dimensions, &pixels, "Symmetry Field carrier A");
        let (_second, second_view, _) = gpu.image(dimensions, &swapped, "Symmetry Field carrier B");
        let (target, target_view, _) = gpu.image(
            dimensions,
            &vec![[0.0; 4]; pixels.len()],
            "Symmetry Field target",
        );
        let plan = field_plan(armed(SymmetryMode::PlanarPmm), 1.0);
        let bindings = prepare_test_bindings(
            &gpu.device,
            &executor,
            [&first_view, &second_view],
            plan.node_id,
            [None, None],
            None,
            [None, None],
        );
        let prepared = executor.allocation_snapshot();
        assert_eq!(
            prepared.bind_groups,
            warm.bind_groups
                + SYMMETRY_FIELD_CARRIER_PARITIES as u64
                + SYMMETRY_FIELD_MOTION_GROUPS as u64,
            "two image bind groups per node — one per carrier parity — plus four \
             motion bind groups, one per committed-parity combination"
        );
        assert_eq!(bindings.image.len(), 1);
        assert_eq!(bindings.motion.len(), 1);

        let (_, parity_zero) = gpu.render(
            &executor,
            &plan,
            &bindings,
            0,
            &target,
            &target_view,
            dimensions,
            SymmetryHistoryCursor::default(),
        );
        assert_eq!(executor.allocation_snapshot(), prepared);
        let (_, repeat) = gpu.render(
            &executor,
            &plan,
            &bindings,
            0,
            &target,
            &target_view,
            dimensions,
            SymmetryHistoryCursor::default(),
        );
        assert_eq!(executor.allocation_snapshot(), prepared);
        assert_eq!(parity_zero, repeat, "the encode is deterministic");

        let (_, parity_one) = gpu.render(
            &executor,
            &plan,
            &bindings,
            1,
            &target,
            &target_view,
            dimensions,
            SymmetryHistoryCursor::default(),
        );
        assert_eq!(executor.allocation_snapshot(), prepared);
        assert_ne!(
            parity_zero, parity_one,
            "the second parity is a real, distinct binding, not a duplicate of the first"
        );

        // Rendering the second carrier's content through parity 1 must agree
        // with the CPU reference computed from that same content, which is what
        // proves parity 1 binds the surface it claims to.
        let carrier = RefImage {
            dimensions,
            pixels: swapped,
        };
        let params = plan.params.values();
        let rendered = decode_pixels(&parity_one);
        let mut compared = 0_usize;
        for row in 0..dimensions[1] {
            for column in 0..dimensions[0] {
                let uv = pixel_uv(column, row, dimensions);
                if !fold_is_locally_stable(params, uv, dimensions) {
                    continue;
                }
                compared += 1;
                let expected = symmetry_reference(&carrier, params, uv, dimensions);
                let actual = rendered[(row * dimensions[0] + column) as usize];
                for channel in 0..4 {
                    assert!(
                        (actual[channel] - expected[channel]).abs() <= 0.01,
                        "parity 1 pixel ({column},{row}) channel {channel}: {} vs {}",
                        actual[channel],
                        expected[channel]
                    );
                }
            }
        }
        assert!(compared * 2 >= (dimensions[0] * dimensions[1]) as usize);
    }

    /// The motion hand-off, on real hardware: all four committed-parity
    /// combinations are prebuilt, each selects a genuinely different field, and
    /// the two slots' parities are independent.
    ///
    /// Each slot's field is given a ZERO vector texture at parity 0 and a
    /// non-zero one at parity 1, with both gates fully open. Rendering at
    /// `[0, 0]` must therefore land exactly on the no-motion render, and each of
    /// `[1, 0]`, `[0, 1]` and `[1, 1]` must move the frame differently. That is
    /// only possible if the executor prepared four distinct motion groups and
    /// encode indexes the pair it was handed — which is exactly what the frozen
    /// two-bind-group contract could not express, and why an authored motion
    /// route used to decode to zero.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_symmetry_field_selects_each_committed_motion_parity_independently_per_slot() {
        let gpu = GpuHarness::new();
        let dimensions = [16, 16];
        let grid = [4_u32, 4];
        let executor =
            SymmetryFieldExecutor::new(&gpu.device, &gpu.queue, dimensions, 1).expect("executor");
        let pixels = carrier_pixels(dimensions);
        let (_carrier, carrier_view, _) = gpu.image(dimensions, &pixels, "Symmetry Field carrier");
        let (_spare, spare_view, _) = gpu.image(dimensions, &pixels, "Symmetry Field spare");
        let (target, target_view, _) = gpu.image(
            dimensions,
            &vec![[0.0; 4]; pixels.len()],
            "Symmetry Field target",
        );

        // Rg16Float UV-per-second vectors. One reference tick scales them, so
        // 6.0 UV/s is 0.2 UV of displacement — several pixels at this size, far
        // outside binary16 noise.
        let vector_texels = |x: f32, y: f32| {
            let mut bytes = Vec::with_capacity((grid[0] * grid[1]) as usize * 4);
            for _ in 0..grid[0] * grid[1] {
                bytes.extend_from_slice(&f32_to_f16(x).to_le_bytes());
                bytes.extend_from_slice(&f32_to_f16(y).to_le_bytes());
            }
            bytes
        };
        // Rg8Unorm: x is lattice confidence, y is the validity/occlusion lane.
        // Both wide open, so nothing but the vector decides the offset.
        let open_gate = vec![255_u8; (grid[0] * grid[1]) as usize * 2];

        let (_zero_vectors, zero_vector_view) = gpu.motion_texture(
            MOTION_VECTOR_FORMAT,
            grid,
            &vector_texels(0.0, 0.0),
            4,
            "Symmetry Field parity-0 zero vectors",
        );
        let (_slot0_vectors, slot0_vector_view) = gpu.motion_texture(
            MOTION_VECTOR_FORMAT,
            grid,
            &vector_texels(6.0, 0.0),
            4,
            "Symmetry Field slot 0 parity-1 vectors",
        );
        let (_slot1_vectors, slot1_vector_view) = gpu.motion_texture(
            MOTION_VECTOR_FORMAT,
            grid,
            &vector_texels(0.0, -6.0),
            4,
            "Symmetry Field slot 1 parity-1 vectors",
        );
        let (_gates, gate_view) = gpu.motion_texture(
            MOTION_GATE_FORMAT,
            grid,
            &open_gate,
            2,
            "Symmetry Field open gates",
        );

        let mut runtime = armed(SymmetryMode::Dihedral);
        runtime.motion_mask = SymmetryMotionMask {
            slot0: true,
            slot1: true,
        };
        runtime.motion_gain = 1.0;
        let plan = field_plan(runtime, 1.0);
        let table = plan.params.sector_table(plan.domain);
        for slot in 0..SYMMETRY_MOTION_SLOTS {
            assert!(
                table
                    .records()
                    .iter()
                    .any(|record| record.motion == Some(slot as u8)),
                "the armed table must name motion slot {slot} in at least one sector"
            );
        }

        let mut bindings = prepare_test_bindings(
            &gpu.device,
            &executor,
            [&carrier_view, &spare_view],
            plan.node_id,
            [None, None],
            None,
            [
                Some(SymmetryMotionViews {
                    vectors: [&zero_vector_view, &slot0_vector_view],
                    gates: [&gate_view, &gate_view],
                    grid,
                }),
                Some(SymmetryMotionViews {
                    vectors: [&zero_vector_view, &slot1_vector_view],
                    gates: [&gate_view, &gate_view],
                    grid,
                }),
            ],
        );
        let prepared = executor.allocation_snapshot();

        let render_at = |bindings: &TestBindings| {
            gpu.render(
                &executor,
                &plan,
                bindings,
                0,
                &target,
                &target_view,
                dimensions,
                SymmetryHistoryCursor::default(),
            )
        };

        bindings.motion_parity = [0, 0];
        let (report, both_zero) = render_at(&bindings);
        assert_eq!(report.missing_motion, 0, "both slots bound a complete pair");
        assert_eq!(executor.allocation_snapshot(), prepared);

        let mut moved = Vec::new();
        for parity in [[1, 0], [0, 1], [1, 1]] {
            bindings.motion_parity = parity;
            let (_, bytes) = render_at(&bindings);
            assert_eq!(
                executor.allocation_snapshot(),
                prepared,
                "selecting a committed motion parity must allocate nothing"
            );
            assert_ne!(
                bytes, both_zero,
                "committed parity {parity:?} must reach the pixels"
            );
            moved.push(bytes);
        }
        assert_ne!(moved[0], moved[1], "the two slots move independently");
        assert_ne!(moved[0], moved[2]);
        assert_ne!(moved[1], moved[2]);

        // A field whose committed parity is not materialized yet is an honest
        // zero: the frame-local readiness bit closes the validity lane and the
        // prebuilt group is left exactly where it was.
        bindings.motion_parity = [1, 1];
        for slot in 0..SYMMETRY_MOTION_SLOTS {
            assert!(bindings.motion.set_motion_ready(plan.node_id, slot, false));
        }
        let (unready_report, unready) = render_at(&bindings);
        assert_eq!(
            unready_report.missing_motion, 2,
            "an unmaterialized parity is reported through the stable diagnostic"
        );
        assert_eq!(
            unready, both_zero,
            "an unmaterialized committed parity must displace by exactly nothing"
        );
        assert_eq!(executor.allocation_snapshot(), prepared);
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
}
