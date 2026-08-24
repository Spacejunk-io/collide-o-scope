//! Bounded persistent GPU-object ownership for the live Compat8 renderer.
//!
//! Advanced composition already has a dynamic-offset uniform arena.  This
//! module deliberately follows the same alignment, overflow, and refusal law
//! instead of inventing a second interpretation of wgpu limits.  Counters are
//! cumulative: callers establish a warmed snapshot and compare later
//! snapshots, so lazy admission remains visible while an ordinary frame is
//! required to have a zero delta.

use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GpuObjectConstructionSnapshot {
    pub buffers: u64,
    pub bind_groups: u64,
    pub pipelines: u64,
    pub textures: u64,
    pub samplers: u64,
}

/// Full-frame work in one retained render topology.  Copy bytes describe
/// texel payload (`width * height * 4` for the Exact RGBA8 surfaces), not
/// backend padding or an estimated bandwidth cost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FullFrameWork {
    pub render_passes: u64,
    pub copy_passes: u64,
    pub copy_bytes: u64,
}

#[allow(
    dead_code,
    reason = "optional performance telemetry consumes per-window receipt deltas"
)]
impl FullFrameWork {
    pub(crate) const fn delta_since(self, earlier: Self) -> Self {
        Self {
            render_passes: self.render_passes.saturating_sub(earlier.render_passes),
            copy_passes: self.copy_passes.saturating_sub(earlier.copy_passes),
            copy_bytes: self.copy_bytes.saturating_sub(earlier.copy_bytes),
        }
    }

    pub(crate) const fn saturating_add(self, other: Self) -> Self {
        Self {
            render_passes: self.render_passes.saturating_add(other.render_passes),
            copy_passes: self.copy_passes.saturating_add(other.copy_passes),
            copy_bytes: self.copy_bytes.saturating_add(other.copy_bytes),
        }
    }
}

/// Cumulative P5 receipt for completed ordinary LegacyExact accumulator
/// plans. `planned` is the immutable ping-pong plan, `executed` is what its
/// encoder actually emitted, and `legacy_baseline` is the byte-for-byte
/// predecessor topology (the same render passes plus one full-frame copy
/// after each composite/master transform).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FullFrameWorkSnapshot {
    pub planned: FullFrameWork,
    pub executed: FullFrameWork,
    pub legacy_baseline: FullFrameWork,
}

#[allow(
    dead_code,
    reason = "optional performance telemetry consumes per-window receipt deltas"
)]
impl FullFrameWorkSnapshot {
    pub(crate) const fn delta_since(self, earlier: Self) -> Self {
        Self {
            planned: self.planned.delta_since(earlier.planned),
            executed: self.executed.delta_since(earlier.executed),
            legacy_baseline: self.legacy_baseline.delta_since(earlier.legacy_baseline),
        }
    }
}

#[derive(Default)]
pub(crate) struct FullFrameWorkCounters {
    planned_render_passes: AtomicU64,
    planned_copy_passes: AtomicU64,
    planned_copy_bytes: AtomicU64,
    executed_render_passes: AtomicU64,
    executed_copy_passes: AtomicU64,
    executed_copy_bytes: AtomicU64,
    baseline_render_passes: AtomicU64,
    baseline_copy_passes: AtomicU64,
    baseline_copy_bytes: AtomicU64,
}

impl FullFrameWorkCounters {
    pub(crate) fn record_completed(
        &self,
        planned: FullFrameWork,
        executed: FullFrameWork,
        legacy_baseline: FullFrameWork,
    ) {
        self.planned_render_passes
            .fetch_add(planned.render_passes, Ordering::Relaxed);
        self.planned_copy_passes
            .fetch_add(planned.copy_passes, Ordering::Relaxed);
        self.planned_copy_bytes
            .fetch_add(planned.copy_bytes, Ordering::Relaxed);
        self.executed_render_passes
            .fetch_add(executed.render_passes, Ordering::Relaxed);
        self.executed_copy_passes
            .fetch_add(executed.copy_passes, Ordering::Relaxed);
        self.executed_copy_bytes
            .fetch_add(executed.copy_bytes, Ordering::Relaxed);
        self.baseline_render_passes
            .fetch_add(legacy_baseline.render_passes, Ordering::Relaxed);
        self.baseline_copy_passes
            .fetch_add(legacy_baseline.copy_passes, Ordering::Relaxed);
        self.baseline_copy_bytes
            .fetch_add(legacy_baseline.copy_bytes, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> FullFrameWorkSnapshot {
        FullFrameWorkSnapshot {
            planned: FullFrameWork {
                render_passes: self.planned_render_passes.load(Ordering::Relaxed),
                copy_passes: self.planned_copy_passes.load(Ordering::Relaxed),
                copy_bytes: self.planned_copy_bytes.load(Ordering::Relaxed),
            },
            executed: FullFrameWork {
                render_passes: self.executed_render_passes.load(Ordering::Relaxed),
                copy_passes: self.executed_copy_passes.load(Ordering::Relaxed),
                copy_bytes: self.executed_copy_bytes.load(Ordering::Relaxed),
            },
            legacy_baseline: FullFrameWork {
                render_passes: self.baseline_render_passes.load(Ordering::Relaxed),
                copy_passes: self.baseline_copy_passes.load(Ordering::Relaxed),
                copy_bytes: self.baseline_copy_bytes.load(Ordering::Relaxed),
            },
        }
    }
}

#[allow(
    dead_code,
    reason = "live telemetry consumes these numeric receipts outside focused renderer tests"
)]
impl GpuObjectConstructionSnapshot {
    pub(crate) const fn total(self) -> u64 {
        self.buffers
            .saturating_add(self.bind_groups)
            .saturating_add(self.pipelines)
            .saturating_add(self.textures)
            .saturating_add(self.samplers)
    }

    pub(crate) const fn delta_since(self, earlier: Self) -> Self {
        Self {
            buffers: self.buffers.saturating_sub(earlier.buffers),
            bind_groups: self.bind_groups.saturating_sub(earlier.bind_groups),
            pipelines: self.pipelines.saturating_sub(earlier.pipelines),
            textures: self.textures.saturating_sub(earlier.textures),
            samplers: self.samplers.saturating_sub(earlier.samplers),
        }
    }

    pub(crate) const fn saturating_add(self, other: Self) -> Self {
        Self {
            buffers: self.buffers.saturating_add(other.buffers),
            bind_groups: self.bind_groups.saturating_add(other.bind_groups),
            pipelines: self.pipelines.saturating_add(other.pipelines),
            textures: self.textures.saturating_add(other.textures),
            samplers: self.samplers.saturating_add(other.samplers),
        }
    }
}

#[derive(Default)]
#[allow(
    dead_code,
    reason = "all five audit domains remain represented even when a build has no sampler construction"
)]
pub(crate) struct GpuObjectConstructionCounters {
    buffers: AtomicU64,
    bind_groups: AtomicU64,
    pipelines: AtomicU64,
    textures: AtomicU64,
    samplers: AtomicU64,
}

#[allow(
    dead_code,
    reason = "live telemetry consumes snapshots; optional constructors consume individual counters"
)]
impl GpuObjectConstructionCounters {
    pub(crate) fn snapshot(&self) -> GpuObjectConstructionSnapshot {
        GpuObjectConstructionSnapshot {
            buffers: self.buffers.load(Ordering::Relaxed),
            bind_groups: self.bind_groups.load(Ordering::Relaxed),
            pipelines: self.pipelines.load(Ordering::Relaxed),
            textures: self.textures.load(Ordering::Relaxed),
            samplers: self.samplers.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn buffer(&self) {
        self.buffers.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn bind_group(&self) {
        self.bind_groups.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn pipeline(&self) {
        self.pipelines.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn texture(&self) {
        self.textures.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn sampler(&self) {
        self.samplers.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UniformArenaAdmission {
    pub stride: u64,
    pub byte_len: u64,
    pub capacity: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UniformArenaError {
    ZeroItemSize,
    ZeroCapacity,
    CapacityExceeded { requested: usize, admitted: usize },
    ArithmeticOverflow,
    DeviceBufferLimit { requested: u64, limit: u64 },
    DynamicOffsetLimit { last_offset: u64 },
    SlotExhausted { requested: usize, capacity: usize },
    PayloadSize { expected: u64, actual: u64 },
}

impl fmt::Display for UniformArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ZeroItemSize => formatter.write_str("uniform arena item size is zero"),
            Self::ZeroCapacity => formatter.write_str("uniform arena capacity is zero"),
            Self::CapacityExceeded {
                requested,
                admitted,
            } => write!(
                formatter,
                "uniform arena requests {requested} slots; admitted maximum is {admitted}"
            ),
            Self::ArithmeticOverflow => formatter.write_str("uniform arena byte size overflows"),
            Self::DeviceBufferLimit { requested, limit } => write!(
                formatter,
                "uniform arena requests {requested} bytes; device maximum is {limit}"
            ),
            Self::DynamicOffsetLimit { last_offset } => write!(
                formatter,
                "uniform arena last dynamic offset {last_offset} exceeds u32"
            ),
            Self::SlotExhausted {
                requested,
                capacity,
            } => write!(
                formatter,
                "uniform arena slot {requested} exceeds capacity {capacity}"
            ),
            Self::PayloadSize { expected, actual } => write!(
                formatter,
                "uniform arena payload is {actual} bytes; expected {expected}"
            ),
        }
    }
}

impl std::error::Error for UniformArenaError {}

pub(crate) fn admit_uniform_arena(
    limits: &wgpu::Limits,
    item_size: u64,
    capacity: usize,
    admitted_capacity: usize,
) -> Result<UniformArenaAdmission, UniformArenaError> {
    if item_size == 0 {
        return Err(UniformArenaError::ZeroItemSize);
    }
    if capacity == 0 {
        return Err(UniformArenaError::ZeroCapacity);
    }
    if capacity > admitted_capacity {
        return Err(UniformArenaError::CapacityExceeded {
            requested: capacity,
            admitted: admitted_capacity,
        });
    }
    let alignment = u64::from(limits.min_uniform_buffer_offset_alignment.max(1));
    let stride = item_size
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or(UniformArenaError::ArithmeticOverflow)?;
    let byte_len = stride
        .checked_mul(capacity as u64)
        .ok_or(UniformArenaError::ArithmeticOverflow)?;
    if byte_len > limits.max_buffer_size {
        return Err(UniformArenaError::DeviceBufferLimit {
            requested: byte_len,
            limit: limits.max_buffer_size,
        });
    }
    let last_offset = stride
        .checked_mul(capacity.saturating_sub(1) as u64)
        .ok_or(UniformArenaError::ArithmeticOverflow)?;
    if last_offset > u64::from(u32::MAX) {
        return Err(UniformArenaError::DynamicOffsetLimit { last_offset });
    }
    Ok(UniformArenaAdmission {
        stride,
        byte_len,
        capacity,
    })
}

pub(crate) struct UniformArena {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    admission: UniformArenaAdmission,
    item_size: u64,
}

impl UniformArena {
    pub(crate) fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        item_size: u64,
        capacity: usize,
        admitted_capacity: usize,
        label: &'static str,
        counters: &GpuObjectConstructionCounters,
    ) -> Result<Self, UniformArenaError> {
        let admission =
            admit_uniform_arena(&device.limits(), item_size, capacity, admitted_capacity)?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: admission.byte_len,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        counters.buffer();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: NonZeroU64::new(item_size),
                }),
            }],
        });
        counters.bind_group();
        Ok(Self {
            buffer,
            bind_group,
            admission,
            item_size,
        })
    }

    pub(crate) fn write<T: bytemuck::Pod>(
        &self,
        queue: &wgpu::Queue,
        slot: usize,
        value: &T,
    ) -> Result<u32, UniformArenaError> {
        if slot >= self.admission.capacity {
            return Err(UniformArenaError::SlotExhausted {
                requested: slot,
                capacity: self.admission.capacity,
            });
        }
        let bytes = bytemuck::bytes_of(value);
        if bytes.len() as u64 != self.item_size {
            return Err(UniformArenaError::PayloadSize {
                expected: self.item_size,
                actual: bytes.len() as u64,
            });
        }
        let offset = self.admission.stride * slot as u64;
        queue.write_buffer(&self.buffer, offset, bytes);
        Ok(offset as u32)
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    #[allow(
        dead_code,
        reason = "runtime resource-ledger integration reads the retained physical byte count"
    )]
    pub(crate) const fn byte_len(&self) -> u64 {
        self.admission.byte_len
    }

    #[allow(
        dead_code,
        reason = "arena admission tests inspect the retained slot bound directly"
    )]
    pub(crate) const fn capacity(&self) -> usize {
        self.admission.capacity
    }
}

/// A stable source cache key.  Positional layer indices are deliberately
/// absent: reorder retains the group; backing/view, raster, surface, or
/// renderer generation changes replace it before encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StableTextureViewKey {
    pub stable_source_id: u64,
    pub backing_view_generation: u64,
    pub raster: [u32; 2],
    pub surface_generation: u64,
    pub renderer_generation: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TextureCacheInvalidationSnapshot {
    pub source_removed: u64,
    pub backing_or_view_generation: u64,
    pub raster: u64,
    pub surface: u64,
    pub renderer: u64,
}

struct CachedTextureBindGroup {
    key: StableTextureViewKey,
    bind_group: Arc<wgpu::BindGroup>,
}

pub(crate) struct StableTextureBindGroupCache {
    entries: BTreeMap<u64, CachedTextureBindGroup>,
    capacity: usize,
    invalidations: TextureCacheInvalidationSnapshot,
}

impl StableTextureBindGroupCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            capacity,
            invalidations: TextureCacheInvalidationSnapshot::default(),
        }
    }

    pub(crate) fn retain_stable_ids(&mut self, mut keep: impl FnMut(u64) -> bool) {
        let before = self.entries.len();
        self.entries.retain(|stable_id, _| keep(*stable_id));
        self.invalidations.source_removed = self
            .invalidations
            .source_removed
            .saturating_add((before - self.entries.len()) as u64);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_create_effects(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        linear_sampler: &wgpu::Sampler,
        nearest_sampler: &wgpu::Sampler,
        view: &wgpu::TextureView,
        key: StableTextureViewKey,
        counters: &GpuObjectConstructionCounters,
    ) -> Result<Arc<wgpu::BindGroup>, String> {
        if let Some(cached) = self.entries.get(&key.stable_source_id) {
            if cached.key == key {
                return Ok(Arc::clone(&cached.bind_group));
            }
        } else if self.entries.len() >= self.capacity {
            return Err(format!(
                "stable texture bind-group cache is full ({} admitted entries)",
                self.capacity
            ));
        }

        if let Some(previous) = self.entries.get(&key.stable_source_id) {
            if previous.key.backing_view_generation != key.backing_view_generation {
                self.invalidations.backing_or_view_generation = self
                    .invalidations
                    .backing_or_view_generation
                    .saturating_add(1);
            }
            if previous.key.raster != key.raster {
                self.invalidations.raster = self.invalidations.raster.saturating_add(1);
            }
            if previous.key.surface_generation != key.surface_generation {
                self.invalidations.surface = self.invalidations.surface.saturating_add(1);
            }
            if previous.key.renderer_generation != key.renderer_generation {
                self.invalidations.renderer = self.invalidations.renderer.saturating_add(1);
            }
        }

        let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Stable Exact source effects inputs"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(nearest_sampler),
                },
            ],
        }));
        counters.bind_group();
        self.entries.insert(
            key.stable_source_id,
            CachedTextureBindGroup {
                key,
                bind_group: Arc::clone(&bind_group),
            },
        );
        Ok(bind_group)
    }

    pub(crate) const fn invalidations(&self) -> TextureCacheInvalidationSnapshot {
        self.invalidations
    }

    #[allow(
        dead_code,
        reason = "cache invalidation tests inspect the bounded resident-entry count"
    )]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU adapter for persistent-object tests");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Persistent-object test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("GPU device for persistent-object tests")
    }

    fn limits(alignment: u32, max_buffer_size: u64) -> wgpu::Limits {
        wgpu::Limits {
            min_uniform_buffer_offset_alignment: alignment,
            max_buffer_size,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn alignment_and_physical_bytes_match_the_advanced_arena_law() {
        let admitted = admit_uniform_arena(&limits(256, 4096), 300, 4, 4).unwrap();
        assert_eq!(admitted.stride, 512);
        assert_eq!(admitted.byte_len, 2048);
        assert_eq!(admitted.capacity, 4);
    }

    #[test]
    fn one_slot_over_refuses_before_gpu_construction() {
        assert_eq!(
            admit_uniform_arena(&limits(256, 4096), 16, 5, 4),
            Err(UniformArenaError::CapacityExceeded {
                requested: 5,
                admitted: 4,
            })
        );
    }

    #[test]
    fn one_byte_under_device_limit_refuses_before_gpu_construction() {
        assert_eq!(
            admit_uniform_arena(&limits(256, 1023), 16, 4, 4),
            Err(UniformArenaError::DeviceBufferLimit {
                requested: 1024,
                limit: 1023,
            })
        );
    }

    #[test]
    fn construction_snapshot_delta_keeps_all_five_domains_independent() {
        let earlier = GpuObjectConstructionSnapshot {
            buffers: 3,
            bind_groups: 5,
            pipelines: 7,
            textures: 11,
            samplers: 13,
        };
        let later = GpuObjectConstructionSnapshot {
            buffers: 4,
            bind_groups: 7,
            pipelines: 10,
            textures: 15,
            samplers: 18,
        };
        assert_eq!(
            later.delta_since(earlier),
            GpuObjectConstructionSnapshot {
                buffers: 1,
                bind_groups: 2,
                pipelines: 3,
                textures: 4,
                samplers: 5,
            }
        );
        assert_eq!(later.delta_since(earlier).total(), 15);
    }

    #[test]
    fn p5_full_frame_work_keeps_planned_executed_and_predecessor_receipts_independent() {
        let counters = FullFrameWorkCounters::default();
        let before = counters.snapshot();
        let planned = FullFrameWork {
            render_passes: 9,
            copy_passes: 0,
            copy_bytes: 0,
        };
        let baseline = FullFrameWork {
            render_passes: 9,
            copy_passes: 9,
            copy_bytes: 9 * 1_920 * 1_080 * 4,
        };
        counters.record_completed(planned, planned, baseline);
        counters.record_completed(
            FullFrameWork {
                render_passes: 1,
                ..FullFrameWork::default()
            },
            FullFrameWork {
                render_passes: 1,
                ..FullFrameWork::default()
            },
            FullFrameWork {
                render_passes: 1,
                copy_passes: 1,
                copy_bytes: 4,
            },
        );
        let after = counters.snapshot();
        assert_eq!(
            after,
            FullFrameWorkSnapshot {
                planned: FullFrameWork {
                    render_passes: 10,
                    copy_passes: 0,
                    copy_bytes: 0,
                },
                executed: FullFrameWork {
                    render_passes: 10,
                    copy_passes: 0,
                    copy_bytes: 0,
                },
                legacy_baseline: baseline.saturating_add(FullFrameWork {
                    render_passes: 1,
                    copy_passes: 1,
                    copy_bytes: 4,
                }),
            }
        );
        assert_eq!(after.delta_since(before), after);
    }

    #[test]
    fn stable_texture_key_is_not_positional_and_carries_every_invalidator() {
        let base = StableTextureViewKey {
            stable_source_id: 7,
            backing_view_generation: 11,
            raster: [1920, 1080],
            surface_generation: 13,
            renderer_generation: 17,
        };
        let reordered = base;
        assert_eq!(base, reordered);
        assert_ne!(
            base,
            StableTextureViewKey {
                backing_view_generation: 12,
                ..base
            }
        );
        assert_ne!(
            base,
            StableTextureViewKey {
                raster: [1280, 720],
                ..base
            }
        );
        assert_ne!(
            base,
            StableTextureViewKey {
                surface_generation: 14,
                ..base
            }
        );
        assert_ne!(
            base,
            StableTextureViewKey {
                renderer_generation: 18,
                ..base
            }
        );
    }

    #[test]
    fn warmed_uniform_writes_construct_zero_gpu_objects() {
        let (device, queue) = gpu_device();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Persistent-object test uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            }],
        });
        let counters = GpuObjectConstructionCounters::default();
        let arena = UniformArena::new(
            &device,
            &layout,
            16,
            8,
            8,
            "Persistent-object test arena",
            &counters,
        )
        .unwrap();
        assert_eq!(arena.capacity(), 8);
        let warmed = counters.snapshot();
        assert_eq!(warmed.buffers, 1);
        assert_eq!(warmed.bind_groups, 1);
        for index in 0..10_000_usize {
            let payload = [index as u32, 2, 3, 4];
            let offset = arena.write(&queue, index % 8, &payload).unwrap();
            assert_eq!(u64::from(offset) % arena.admission.stride, 0);
        }
        queue.submit(std::iter::empty());
        assert_eq!(counters.snapshot().delta_since(warmed).total(), 0);
    }

    #[test]
    fn stable_texture_cache_reuses_and_counts_every_invalidator() {
        let (device, _queue) = gpu_device();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Persistent-object texture cache layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Persistent-object texture cache source"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let linear = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let counters = GpuObjectConstructionCounters::default();
        let mut cache = StableTextureBindGroupCache::new(2);
        let base = StableTextureViewKey {
            stable_source_id: 41,
            backing_view_generation: 1,
            raster: [4, 4],
            surface_generation: 1,
            renderer_generation: 1,
        };
        let first = cache
            .get_or_create_effects(&device, &layout, &linear, &nearest, &view, base, &counters)
            .unwrap();
        let warmed = counters.snapshot();
        let reused = cache
            .get_or_create_effects(&device, &layout, &linear, &nearest, &view, base, &counters)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &reused));
        assert_eq!(counters.snapshot().delta_since(warmed).total(), 0);

        for changed in [
            StableTextureViewKey {
                backing_view_generation: 2,
                ..base
            },
            StableTextureViewKey {
                backing_view_generation: 2,
                raster: [8, 4],
                ..base
            },
            StableTextureViewKey {
                backing_view_generation: 2,
                raster: [8, 4],
                surface_generation: 2,
                ..base
            },
            StableTextureViewKey {
                backing_view_generation: 2,
                raster: [8, 4],
                surface_generation: 2,
                renderer_generation: 2,
                ..base
            },
        ] {
            cache
                .get_or_create_effects(
                    &device, &layout, &linear, &nearest, &view, changed, &counters,
                )
                .unwrap();
        }
        let invalidations = cache.invalidations();
        assert_eq!(invalidations.backing_or_view_generation, 1);
        assert_eq!(invalidations.raster, 1);
        assert_eq!(invalidations.surface, 1);
        assert_eq!(invalidations.renderer, 1);
        assert_eq!(cache.len(), 1);
        cache.retain_stable_ids(|_| false);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.invalidations().source_removed, 1);
    }
}
