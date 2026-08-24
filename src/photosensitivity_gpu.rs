//! Isolated GPU reduction/readback contract for the deferred D2 advisor.
//!
//! Construction is explicit, the source binding is read-only and fixed for
//! the stage lifetime, and readback is exactly eight aggregate `u32`s. No live
//! renderer owns this stage while the required P1 p95/p99 performance gate and
//! independent accessibility/legal review are absent.

use crate::photosensitivity_advisor::{
    AdvisorPolicy, AdvisorPolicyError, CompactTransitionCounters, ADVISOR_CELLS,
    ADVISOR_TAPS_PER_CELL,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub const ADVISOR_WORKGROUP_SIZE: u32 = 64;
pub const ADVISOR_WORKGROUPS: u32 = ADVISOR_CELLS as u32 / ADVISOR_WORKGROUP_SIZE;
pub const ADVISOR_TEXTURE_LOADS_PER_SAMPLE: u32 =
    ADVISOR_CELLS as u32 * ADVISOR_TAPS_PER_CELL as u32;
pub const ADVISOR_READBACK_SLOTS: usize = 3;
pub const ADVISOR_HISTORY_CELL_BYTES: u64 = 32;
pub const ADVISOR_HISTORY_BUFFER_BYTES: u64 = ADVISOR_CELLS as u64 * ADVISOR_HISTORY_CELL_BYTES;
pub const ADVISOR_READBACK_POOL_BYTES: u64 =
    CompactTransitionCounters::BYTE_LEN * ADVISOR_READBACK_SLOTS as u64;

const SLOT_IDLE: u8 = 0;
const SLOT_IN_FLIGHT: u8 = 1;
const SLOT_MAPPED: u8 = 2;
const SLOT_FAILED: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuAdvisorKernelContract {
    pub lattice_cells: u32,
    pub taps_per_cell: u32,
    pub texture_loads_per_sample: u32,
    pub workgroup_size: u32,
    pub workgroups_per_sample: u32,
    pub compact_readback_bytes: u32,
    pub readback_slots: u8,
    pub history_buffer_bytes: u64,
    pub readback_pool_bytes: u64,
}

pub const fn gpu_advisor_kernel_contract() -> GpuAdvisorKernelContract {
    GpuAdvisorKernelContract {
        lattice_cells: ADVISOR_CELLS as u32,
        taps_per_cell: ADVISOR_TAPS_PER_CELL as u32,
        texture_loads_per_sample: ADVISOR_TEXTURE_LOADS_PER_SAMPLE,
        workgroup_size: ADVISOR_WORKGROUP_SIZE,
        workgroups_per_sample: ADVISOR_WORKGROUPS,
        compact_readback_bytes: CompactTransitionCounters::BYTE_LEN as u32,
        readback_slots: ADVISOR_READBACK_SLOTS as u8,
        history_buffer_bytes: ADVISOR_HISTORY_BUFFER_BYTES,
        readback_pool_bytes: ADVISOR_READBACK_POOL_BYTES,
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuReductionPolicy {
    transition_threshold_q: u32,
    red_saturation_q: u32,
    red_dominance_q: u32,
    reserved: u32,
}

impl From<AdvisorPolicy> for GpuReductionPolicy {
    fn from(policy: AdvisorPolicy) -> Self {
        Self {
            transition_threshold_q: u32::from(policy.transition_threshold_q),
            red_saturation_q: u32::from(policy.red_saturation_q),
            red_dominance_q: u32::from(policy.red_dominance_q),
            reserved: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuAdvisorRuntimeCounters {
    pub scheduled_samples: u64,
    pub dropped_busy_samples: u64,
    pub completed_samples: u64,
    pub malformed_samples: u64,
    pub map_failures: u64,
    pub history_resets: u64,
    pub reset_busy_rejections: u64,
    pub submitted_workgroups: u64,
    pub submitted_texture_loads: u64,
    pub submitted_readback_bytes: u64,
    pub mapped_readback_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAdvisorError {
    Policy(AdvisorPolicyError),
    UnsupportedSourceFormat(wgpu::TextureFormat),
    DeviceLimit {
        resource: &'static str,
        requested: u64,
        limit: u64,
    },
}

impl fmt::Display for GpuAdvisorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Policy(error) => write!(formatter, "invalid advisor policy: {error}"),
            Self::UnsupportedSourceFormat(format) => write!(
                formatter,
                "advisor evaluation requires an sRGB RGBA/BGRA8 source view, got {format:?}"
            ),
            Self::DeviceLimit {
                resource,
                requested,
                limit,
            } => write!(
                formatter,
                "advisor {resource} requests {requested} bytes, device limit is {limit}"
            ),
        }
    }
}

impl std::error::Error for GpuAdvisorError {}

impl From<AdvisorPolicyError> for GpuAdvisorError {
    fn from(error: AdvisorPolicyError) -> Self {
        Self::Policy(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAdvisorAdmission {
    Scheduled { sequence: u64 },
    Busy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuAdvisorSample {
    pub sequence: u64,
    pub reference_tick: u64,
    pub counters: CompactTransitionCounters,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuAdvisorDropReason {
    MapFailed,
    MalformedAggregate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum GpuAdvisorPoll {
    Sample(GpuAdvisorSample),
    Dropped {
        sequence: u64,
        reference_tick: u64,
        reason: GpuAdvisorDropReason,
    },
}

struct AdvisorReadbackSlot {
    buffer: wgpu::Buffer,
    status: Arc<AtomicU8>,
    sequence: u64,
    reference_tick: u64,
}

/// Evaluation-only fixed-source GPU stage. Recreate this value when the
/// source backing/view generation changes; there is intentionally no rebind
/// method that could accidentally preserve history across identities.
pub struct PhotosensitivityAdvisorGpu {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    history_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    slots: [AdvisorReadbackSlot; ADVISOR_READBACK_SLOTS],
    next_sequence: u64,
    runtime_counters: GpuAdvisorRuntimeCounters,
}

impl PhotosensitivityAdvisorGpu {
    pub fn new_evaluation_only(
        device: &wgpu::Device,
        source_texture: &wgpu::Texture,
        policy: AdvisorPolicy,
    ) -> Result<Self, GpuAdvisorError> {
        let policy = policy.validate()?;
        let source_format = source_texture.format();
        if !matches!(
            source_format,
            wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            return Err(GpuAdvisorError::UnsupportedSourceFormat(source_format));
        }
        let limits = device.limits();
        if ADVISOR_HISTORY_BUFFER_BYTES > limits.max_buffer_size {
            return Err(GpuAdvisorError::DeviceLimit {
                resource: "history buffer",
                requested: ADVISOR_HISTORY_BUFFER_BYTES,
                limit: limits.max_buffer_size,
            });
        }
        if ADVISOR_HISTORY_BUFFER_BYTES > limits.max_storage_buffer_binding_size {
            return Err(GpuAdvisorError::DeviceLimit {
                resource: "history storage binding",
                requested: ADVISOR_HISTORY_BUFFER_BYTES,
                limit: limits.max_storage_buffer_binding_size,
            });
        }

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("D2 advisor evaluation bind layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(ADVISOR_HISTORY_BUFFER_BYTES),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(CompactTransitionCounters::BYTE_LEN),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<GpuReductionPolicy>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("D2 advisor evaluation shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/photosensitivity_advisor.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("D2 advisor evaluation pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("D2 advisor evaluation pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reduce"),
            compilation_options: Default::default(),
            cache: None,
        });

        let history_buffer = create_zeroed_buffer(
            device,
            "D2 advisor evaluation history",
            ADVISOR_HISTORY_BUFFER_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let counter_buffer = create_zeroed_buffer(
            device,
            "D2 advisor evaluation aggregate counters",
            CompactTransitionCounters::BYTE_LEN,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let policy_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("D2 advisor immutable evaluation policy"),
            contents: bytemuck::bytes_of(&GpuReductionPolicy::from(policy)),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let source_view = source_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("D2 advisor evaluation bind group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: history_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: policy_buffer.as_entire_binding(),
                },
            ],
        });
        let slots = std::array::from_fn(|index| AdvisorReadbackSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("D2 advisor compact readback {index}")),
                size: CompactTransitionCounters::BYTE_LEN,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            status: Arc::new(AtomicU8::new(SLOT_IDLE)),
            sequence: 0,
            reference_tick: 0,
        });

        Ok(Self {
            pipeline,
            bind_group,
            history_buffer,
            counter_buffer,
            slots,
            next_sequence: 1,
            runtime_counters: GpuAdvisorRuntimeCounters::default(),
        })
    }

    /// Schedule exactly 2,304 invocations, 36,864 texture loads, and a 32-byte
    /// counter copy. Saturation drops the new observation; it never queues a
    /// backlog or allocates another slot.
    pub fn schedule(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        reference_tick: u64,
    ) -> GpuAdvisorAdmission {
        let Some(index) = self
            .slots
            .iter()
            .position(|slot| slot.status.load(Ordering::Acquire) == SLOT_IDLE)
        else {
            self.runtime_counters.dropped_busy_samples =
                self.runtime_counters.dropped_busy_samples.saturating_add(1);
            return GpuAdvisorAdmission::Busy;
        };

        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("D2 advisor evaluation reduction"),
        });
        encoder.clear_buffer(
            &self.counter_buffer,
            0,
            Some(CompactTransitionCounters::BYTE_LEN),
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("D2 advisor fixed-lattice pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(ADVISOR_WORKGROUPS, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.counter_buffer,
            0,
            &self.slots[index].buffer,
            0,
            CompactTransitionCounters::BYTE_LEN,
        );

        let slot = &mut self.slots[index];
        slot.sequence = sequence;
        slot.reference_tick = reference_tick;
        slot.status.store(SLOT_IN_FLIGHT, Ordering::Release);
        queue.submit(std::iter::once(encoder.finish()));
        let status = Arc::clone(&slot.status);
        slot.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                status.store(
                    if result.is_ok() {
                        SLOT_MAPPED
                    } else {
                        SLOT_FAILED
                    },
                    Ordering::Release,
                );
            });

        self.runtime_counters.scheduled_samples =
            self.runtime_counters.scheduled_samples.saturating_add(1);
        self.runtime_counters.submitted_workgroups = self
            .runtime_counters
            .submitted_workgroups
            .saturating_add(u64::from(ADVISOR_WORKGROUPS));
        self.runtime_counters.submitted_texture_loads = self
            .runtime_counters
            .submitted_texture_loads
            .saturating_add(u64::from(ADVISOR_TEXTURE_LOADS_PER_SAMPLE));
        self.runtime_counters.submitted_readback_bytes = self
            .runtime_counters
            .submitted_readback_bytes
            .saturating_add(CompactTransitionCounters::BYTE_LEN);
        GpuAdvisorAdmission::Scheduled { sequence }
    }

    /// Poll strictly in submission order. The returned data shape cannot hold
    /// pixels or authored strings, and a map failure recycles just one slot.
    pub fn poll(&mut self) -> Option<GpuAdvisorPoll> {
        let index = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.status.load(Ordering::Acquire) != SLOT_IDLE)
            .min_by_key(|(_, slot)| slot.sequence)
            .map(|(index, _)| index)?;
        let status = self.slots[index].status.load(Ordering::Acquire);
        if status == SLOT_IN_FLIGHT {
            return None;
        }
        let sequence = self.slots[index].sequence;
        let reference_tick = self.slots[index].reference_tick;
        if status == SLOT_FAILED {
            self.slots[index].buffer.unmap();
            self.slots[index].status.store(SLOT_IDLE, Ordering::Release);
            self.runtime_counters.map_failures =
                self.runtime_counters.map_failures.saturating_add(1);
            return Some(GpuAdvisorPoll::Dropped {
                sequence,
                reference_tick,
                reason: GpuAdvisorDropReason::MapFailed,
            });
        }
        debug_assert_eq!(status, SLOT_MAPPED);
        let sample = {
            let mapped = self.slots[index].buffer.slice(..).get_mapped_range();
            bytemuck::pod_read_unaligned::<CompactTransitionCounters>(&mapped)
        };
        self.slots[index].buffer.unmap();
        self.slots[index].status.store(SLOT_IDLE, Ordering::Release);
        self.runtime_counters.mapped_readback_bytes = self
            .runtime_counters
            .mapped_readback_bytes
            .saturating_add(CompactTransitionCounters::BYTE_LEN);
        match sample.validate() {
            Ok(counters) => {
                self.runtime_counters.completed_samples =
                    self.runtime_counters.completed_samples.saturating_add(1);
                Some(GpuAdvisorPoll::Sample(GpuAdvisorSample {
                    sequence,
                    reference_tick,
                    counters,
                }))
            }
            Err(_) => {
                self.runtime_counters.malformed_samples =
                    self.runtime_counters.malformed_samples.saturating_add(1);
                Some(GpuAdvisorPoll::Dropped {
                    sequence,
                    reference_tick,
                    reason: GpuAdvisorDropReason::MalformedAggregate,
                })
            }
        }
    }

    /// Clear cell history only when all map slots have been harvested. Queue
    /// ordering then proves the clear precedes the next scheduled reduction.
    pub fn reset_history_if_idle(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        if self
            .slots
            .iter()
            .any(|slot| slot.status.load(Ordering::Acquire) != SLOT_IDLE)
        {
            self.runtime_counters.reset_busy_rejections = self
                .runtime_counters
                .reset_busy_rejections
                .saturating_add(1);
            return false;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("D2 advisor history reset"),
        });
        encoder.clear_buffer(&self.history_buffer, 0, Some(ADVISOR_HISTORY_BUFFER_BYTES));
        encoder.clear_buffer(
            &self.counter_buffer,
            0,
            Some(CompactTransitionCounters::BYTE_LEN),
        );
        queue.submit(std::iter::once(encoder.finish()));
        self.runtime_counters.history_resets =
            self.runtime_counters.history_resets.saturating_add(1);
        true
    }

    pub const fn runtime_counters(&self) -> GpuAdvisorRuntimeCounters {
        self.runtime_counters
    }
}

fn create_zeroed_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        let zeroes = vec![0; usize::try_from(size).expect("bounded advisor buffer size")];
        mapped.copy_from_slice(&zeroes);
    }
    buffer.unmap();
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photosensitivity_advisor::{
        PhotosensitivityCpuReference, ADVISOR_GRID_HEIGHT, ADVISOR_GRID_WIDTH,
    };

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    const WIDTH: u32 = 256;
    const HEIGHT: u32 = 144;

    fn policy() -> AdvisorPolicy {
        AdvisorPolicy {
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

    #[test]
    fn d2_gpu_contract_is_constant_with_raster_and_compact() {
        let contract = gpu_advisor_kernel_contract();
        assert_eq!(contract.lattice_cells, 64 * 36);
        assert_eq!(
            contract.lattice_cells,
            (ADVISOR_GRID_WIDTH * ADVISOR_GRID_HEIGHT) as u32
        );
        assert_eq!(contract.taps_per_cell, 16);
        assert_eq!(contract.texture_loads_per_sample, 64 * 36 * 16);
        assert_eq!(contract.workgroups_per_sample, 36);
        assert_eq!(contract.compact_readback_bytes, 32);
        assert_eq!(contract.readback_slots, 3);
        assert_eq!(contract.history_buffer_bytes, 64 * 36 * 32);
        assert_eq!(contract.readback_pool_bytes, 3 * 32);
        assert_eq!(std::mem::size_of::<GpuReductionPolicy>(), 16);
    }

    #[test]
    fn d2_gpu_shader_has_read_only_pixels_and_aggregate_only_readback() {
        let shader = include_str!("shaders/photosensitivity_advisor.wgsl");
        assert!(shader.contains("var source_texture: texture_2d<f32>"));
        assert!(shader.contains("textureLoad(source_texture"));
        assert!(!shader.contains("texture_storage_"));
        assert!(!shader.contains("textureStore"));
        assert!(!shader.contains("sampler"));
        assert!(shader.contains("var<storage, read_write> counters: CompactCounters"));
        assert!(!shader.contains("array<vec4<f32>, 2304"));
    }

    fn acquire_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("D2 advisor evaluation test"),
            ..Default::default()
        }))
        .ok()
    }

    fn solid(r: u8, g: u8, b: u8) -> Vec<u8> {
        [r, g, b, 255].repeat((WIDTH * HEIGHT) as usize)
    }

    fn upload(queue: &wgpu::Queue, texture: &wgpu::Texture, pixels: &[u8]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 4),
                rows_per_image: Some(HEIGHT),
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
    }

    fn schedule_and_harvest(
        stage: &mut PhotosensitivityAdvisorGpu,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        reference_tick: u64,
    ) -> CompactTransitionCounters {
        assert!(matches!(
            stage.schedule(device, queue, reference_tick),
            GpuAdvisorAdmission::Scheduled { .. }
        ));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        match stage.poll().expect("completed compact readback") {
            GpuAdvisorPoll::Sample(sample) => {
                assert_eq!(sample.reference_tick, reference_tick);
                sample.counters
            }
            dropped => panic!("unexpected advisor drop: {dropped:?}"),
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter; promotion also requires a separate P1 p95/p99 run"]
    fn d2_gpu_flat_hostile_fixtures_match_cpu_and_pool_is_bounded() {
        let Some((device, queue)) = acquire_device() else {
            panic!("no GPU adapter available for the opt-in fixture");
        };
        let source = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("D2 advisor evaluation fixture source"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut stage = PhotosensitivityAdvisorGpu::new_evaluation_only(&device, &source, policy())
            .expect("create evaluation stage");
        let mut cpu = PhotosensitivityCpuReference::default();

        for (tick, pixels) in [
            (1, solid(0, 0, 0)),
            (2, solid(255, 255, 255)),
            (3, solid(0, 0, 0)),
            (4, solid(255, 0, 0)),
            (5, solid(128, 64, 192)),
        ] {
            upload(&queue, &source, &pixels);
            let expected = cpu
                .analyze_rgba8_srgb(&pixels, WIDTH as usize, HEIGHT as usize, policy())
                .expect("CPU reference");
            let actual = schedule_and_harvest(&mut stage, &device, &queue, tick);
            if tick <= 4 {
                assert_eq!(
                    actual, expected,
                    "exact endpoint aggregate mismatch at tick {tick}"
                );
            } else {
                // Hardware sRGB decode may differ from the pinned CPU table by
                // a few Q0.16 units. Classification counts must remain exact;
                // summed magnitudes get an explicit two-unit-per-cell bound.
                assert_eq!(actual.sampled_cells, expected.sampled_cells);
                assert_eq!(actual.initialized_cells, expected.initialized_cells);
                assert_eq!(actual.affected_cells, expected.affected_cells);
                assert_eq!(actual.reversal_cells, expected.reversal_cells);
                assert_eq!(actual.red_transition_cells, expected.red_transition_cells);
                assert_eq!(actual.reserved, expected.reserved);
                let magnitude_tolerance = expected.initialized_cells.saturating_mul(2);
                assert!(
                    actual.luma_delta_sum_q.abs_diff(expected.luma_delta_sum_q)
                        <= magnitude_tolerance
                );
                assert!(
                    actual
                        .color_delta_sum_q
                        .abs_diff(expected.color_delta_sum_q)
                        <= magnitude_tolerance
                );
            }
        }

        // Three slots is a hard bound even if GPU completion races ahead: a
        // completed map remains occupied until the caller explicitly polls.
        for tick in 6..9 {
            assert!(matches!(
                stage.schedule(&device, &queue, tick),
                GpuAdvisorAdmission::Scheduled { .. }
            ));
        }
        assert_eq!(
            stage.schedule(&device, &queue, 9),
            GpuAdvisorAdmission::Busy
        );
        assert!(!stage.reset_history_if_idle(&device, &queue));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU wait");
        for _ in 0..ADVISOR_READBACK_SLOTS {
            assert!(matches!(stage.poll(), Some(GpuAdvisorPoll::Sample(_))));
        }
        assert!(stage.reset_history_if_idle(&device, &queue));

        let counters = stage.runtime_counters();
        assert_eq!(counters.scheduled_samples, 8);
        assert_eq!(counters.dropped_busy_samples, 1);
        assert_eq!(counters.completed_samples, 8);
        assert_eq!(counters.map_failures, 0);
        assert_eq!(counters.reset_busy_rejections, 1);
        assert_eq!(counters.history_resets, 1);
        assert_eq!(
            counters.submitted_workgroups,
            8 * u64::from(ADVISOR_WORKGROUPS)
        );
        assert_eq!(
            counters.submitted_texture_loads,
            8 * u64::from(ADVISOR_TEXTURE_LOADS_PER_SAMPLE)
        );
        assert_eq!(
            counters.mapped_readback_bytes,
            8 * CompactTransitionCounters::BYTE_LEN
        );
    }
}
