//! Data-only R2 planner for at most two independent Codec-Mosh domains.
//!
//! This module deliberately owns no codec, GPU, patch, panel, or worker. It is
//! the bounded structural experiment required before any runtime feature may
//! be proposed. The existing single final-program Codec-Mosh path remains the
//! compatibility path and is not represented as one of these new domains.

use serde::de;
use serde::{Deserialize, Deserializer, Serialize};

pub const MAX_MOSH_DOMAINS: usize = 2;
pub const PROTOTYPE_MAX_EDGE: u32 = 1_280;
pub const PROTOTYPE_MAX_PIXELS: u64 = 1_280 * 720;
pub const PROTOTYPE_MAX_JOB_BYTES: u64 = 8 * 1024 * 1024;
pub const PROTOTYPE_TOTAL_BYTES: u64 = 48 * 1024 * 1024;
pub const PROTOTYPE_MAX_WAKE_CELLS: u32 = 4_096;
pub const PROTOTYPE_MAX_RETAINED_FRAMES: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct MoshDomainId(u64);

impl MoshDomainId {
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for MoshDomainId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| de::Error::custom("Mosh domain identity must be non-zero"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoshDomainOrder {
    /// Tap after the domain's admitted layer/group composition, then compose
    /// the influenced result before the one shared final temporal/VHS finish.
    PostCompositionPreFinalTemporalVhs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DomainSlot {
    Vacant,
    Live(MoshDomainId),
    Tombstone(MoshDomainId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainRegistryError {
    DuplicateIdentity,
    TombstonedIdentity,
    CapacityExhausted,
    UnknownIdentity,
}

/// Two fixed identity slots. Deletion is permanent for the document lifetime:
/// a tombstone can be replayed but never silently retargeted or recycled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoshDomainRegistry {
    slots: [DomainSlot; MAX_MOSH_DOMAINS],
}

impl Default for MoshDomainRegistry {
    fn default() -> Self {
        Self {
            slots: [DomainSlot::Vacant; MAX_MOSH_DOMAINS],
        }
    }
}

impl MoshDomainRegistry {
    pub fn create(&mut self, id: MoshDomainId) -> Result<(), DomainRegistryError> {
        if self.slots.contains(&DomainSlot::Live(id)) {
            return Err(DomainRegistryError::DuplicateIdentity);
        }
        if self.slots.contains(&DomainSlot::Tombstone(id)) {
            return Err(DomainRegistryError::TombstonedIdentity);
        }
        let Some(slot) = self
            .slots
            .iter_mut()
            .find(|slot| matches!(slot, DomainSlot::Vacant))
        else {
            return Err(DomainRegistryError::CapacityExhausted);
        };
        *slot = DomainSlot::Live(id);
        Ok(())
    }

    pub fn delete(&mut self, id: MoshDomainId) -> Result<(), DomainRegistryError> {
        let Some(slot) = self
            .slots
            .iter_mut()
            .find(|slot| **slot == DomainSlot::Live(id))
        else {
            return Err(DomainRegistryError::UnknownIdentity);
        };
        *slot = DomainSlot::Tombstone(id);
        Ok(())
    }

    pub fn contains(&self, id: MoshDomainId) -> bool {
        self.slots.contains(&DomainSlot::Live(id))
    }

    pub fn live_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| matches!(slot, DomainSlot::Live(_)))
            .count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoshDomainRequest {
    pub id: MoshDomainId,
    /// Monotonic domain generation. Zero is never a live generation.
    pub generation: u64,
    pub width: u32,
    pub height: u32,
    pub job_bytes: u64,
    pub wake_cells: u32,
    pub retained_frames: u8,
    /// Path-free structural recipe identity. Mutable histories never share it
    /// as storage; it is comparison/provenance only.
    pub recipe_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoshDomainSend {
    pub stable_item_id: u64,
    pub domain: MoshDomainId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoshDomainLimits {
    pub max_edge: u32,
    pub max_pixels_per_domain: u64,
    pub max_job_bytes_per_domain: u64,
    pub max_total_bytes: u64,
    pub max_wake_cells_per_domain: u32,
    pub max_retained_frames_per_domain: u8,
    pub max_workers: u8,
}

impl MoshDomainLimits {
    pub const fn prototype_720p() -> Self {
        Self {
            max_edge: PROTOTYPE_MAX_EDGE,
            max_pixels_per_domain: PROTOTYPE_MAX_PIXELS,
            max_job_bytes_per_domain: PROTOTYPE_MAX_JOB_BYTES,
            max_total_bytes: PROTOTYPE_TOTAL_BYTES,
            max_wake_cells_per_domain: PROTOTYPE_MAX_WAKE_CELLS,
            max_retained_frames_per_domain: PROTOTYPE_MAX_RETAINED_FRAMES,
            max_workers: MAX_MOSH_DOMAINS as u8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoshDomainPlanError {
    TooManyRequests,
    DuplicateRequest,
    UnknownDomain,
    InvalidRaster,
    RasterCap,
    JobByteCap,
    WakeCellCap,
    RetainedFrameCap,
    TotalByteCap,
    WorkerCap,
    UnknownSendDomain,
    InvalidGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoshDomainPhysicalPlan {
    pub id: MoshDomainId,
    pub generation: u64,
    pub raster_bytes: u64,
    pub job_bytes: u64,
    pub wake_bytes: u64,
    pub retained_frames: u8,
    pub send_count: u64,
    pub recipe_digest: [u8; 32],
    /// A domain-specific mutable-history owner. Different IDs are forbidden
    /// from aliasing this identity even when recipes are byte-identical.
    pub mutable_history_owner: MoshDomainId,
}

impl MoshDomainPhysicalPlan {
    pub const fn total_bytes(self) -> u64 {
        self.raster_bytes
            .saturating_add(self.job_bytes)
            .saturating_add(self.wake_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoshDomainPlan {
    pub order: MoshDomainOrder,
    pub domains: [Option<MoshDomainPhysicalPlan>; MAX_MOSH_DOMAINS],
    pub worker_count: u8,
    pub total_bytes: u64,
}

impl MoshDomainPlan {
    pub const fn zero_domain_exact_legacy() -> Self {
        Self {
            order: MoshDomainOrder::PostCompositionPreFinalTemporalVhs,
            domains: [None; MAX_MOSH_DOMAINS],
            worker_count: 0,
            total_bytes: 0,
        }
    }
}

fn checked_domain_plan(
    request: MoshDomainRequest,
    limits: MoshDomainLimits,
) -> Result<MoshDomainPhysicalPlan, MoshDomainPlanError> {
    if request.width == 0 || request.height == 0 {
        return Err(MoshDomainPlanError::InvalidRaster);
    }
    if request.generation == 0 {
        return Err(MoshDomainPlanError::InvalidGeneration);
    }
    let pixels = u64::from(request.width)
        .checked_mul(u64::from(request.height))
        .ok_or(MoshDomainPlanError::RasterCap)?;
    if request.width > limits.max_edge
        || request.height > limits.max_edge
        || pixels > limits.max_pixels_per_domain
    {
        return Err(MoshDomainPlanError::RasterCap);
    }
    if request.job_bytes > limits.max_job_bytes_per_domain {
        return Err(MoshDomainPlanError::JobByteCap);
    }
    if request.wake_cells > limits.max_wake_cells_per_domain {
        return Err(MoshDomainPlanError::WakeCellCap);
    }
    if request.retained_frames > limits.max_retained_frames_per_domain {
        return Err(MoshDomainPlanError::RetainedFrameCap);
    }
    let raster_bytes = pixels
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_mul(u64::from(request.retained_frames)))
        .ok_or(MoshDomainPlanError::TotalByteCap)?;
    let wake_bytes = u64::from(request.wake_cells)
        .checked_mul(16)
        .ok_or(MoshDomainPlanError::TotalByteCap)?;
    Ok(MoshDomainPhysicalPlan {
        id: request.id,
        generation: request.generation,
        raster_bytes,
        job_bytes: request.job_bytes,
        wake_bytes,
        retained_frames: request.retained_frames,
        send_count: 0,
        recipe_digest: request.recipe_digest,
        mutable_history_owner: request.id,
    })
}

pub fn plan_mosh_domains(
    registry: &MoshDomainRegistry,
    requests: &[MoshDomainRequest],
    sends: impl IntoIterator<Item = MoshDomainSend>,
    limits: MoshDomainLimits,
) -> Result<MoshDomainPlan, MoshDomainPlanError> {
    if requests.len() > MAX_MOSH_DOMAINS {
        return Err(MoshDomainPlanError::TooManyRequests);
    }
    if requests.len() > usize::from(limits.max_workers) {
        return Err(MoshDomainPlanError::WorkerCap);
    }
    let mut plan = MoshDomainPlan::zero_domain_exact_legacy();
    for (index, request) in requests.iter().copied().enumerate() {
        if !registry.contains(request.id) {
            return Err(MoshDomainPlanError::UnknownDomain);
        }
        if requests[..index].iter().any(|prior| prior.id == request.id) {
            return Err(MoshDomainPlanError::DuplicateRequest);
        }
        let domain = checked_domain_plan(request, limits)?;
        plan.total_bytes = plan
            .total_bytes
            .checked_add(domain.total_bytes())
            .ok_or(MoshDomainPlanError::TotalByteCap)?;
        if plan.total_bytes > limits.max_total_bytes {
            return Err(MoshDomainPlanError::TotalByteCap);
        }
        plan.domains[index] = Some(domain);
    }
    plan.worker_count = requests.len() as u8;

    for send in sends {
        let Some(domain) = plan
            .domains
            .iter_mut()
            .flatten()
            .find(|domain| domain.id == send.domain)
        else {
            return Err(MoshDomainPlanError::UnknownSendDomain);
        };
        // Stable item identity is deliberately not retained here. Resource
        // ownership stays constant with layer count; the compiled composition
        // plan owns its ordinary bounded stable-ID routing table.
        let _ = send.stable_item_id;
        domain.send_count = domain.send_count.saturating_add(1);
    }
    Ok(plan)
}

/// A synchronization-kernel model of one domain's newest-only job mailbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatestOnlyDomainMailbox<T> {
    pending: Option<DomainJob<T>>,
    pub offered: u64,
    pub superseded: u64,
    pub stale_refused: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainJob<T> {
    pub domain: MoshDomainId,
    pub generation: u64,
    pub payload: T,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainJobError {
    WrongDomain,
    StaleGeneration,
}

impl<T> Default for LatestOnlyDomainMailbox<T> {
    fn default() -> Self {
        Self {
            pending: None,
            offered: 0,
            superseded: 0,
            stale_refused: 0,
        }
    }
}

impl<T> LatestOnlyDomainMailbox<T> {
    pub fn offer(
        &mut self,
        owner: MoshDomainId,
        live_generation: u64,
        job: DomainJob<T>,
    ) -> Result<(), DomainJobError> {
        if job.domain != owner {
            return Err(DomainJobError::WrongDomain);
        }
        if live_generation == 0 || job.generation != live_generation {
            self.stale_refused = self.stale_refused.saturating_add(1);
            return Err(DomainJobError::StaleGeneration);
        }
        self.offered = self.offered.saturating_add(1);
        if self.pending.replace(job).is_some() {
            self.superseded = self.superseded.saturating_add(1);
        }
        Ok(())
    }

    pub fn take(&mut self) -> Option<DomainJob<T>> {
        self.pending.take()
    }

    pub const fn depth(&self) -> usize {
        if self.pending.is_some() {
            1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: u64) -> MoshDomainId {
        MoshDomainId::new(value).unwrap()
    }

    fn request(id: MoshDomainId) -> MoshDomainRequest {
        MoshDomainRequest {
            id,
            generation: id.get(),
            width: 1_280,
            height: 720,
            job_bytes: 4 * 1024 * 1024,
            wake_cells: 2_048,
            retained_frames: 3,
            recipe_digest: [id.get() as u8; 32],
        }
    }

    #[test]
    fn zero_domains_are_exactly_zero_resource_and_one_domain_allocates_no_second() {
        let registry = MoshDomainRegistry::default();
        let zero = plan_mosh_domains(
            &registry,
            &[],
            std::iter::empty(),
            MoshDomainLimits::prototype_720p(),
        )
        .unwrap();
        assert_eq!(zero, MoshDomainPlan::zero_domain_exact_legacy());

        let mut registry = registry;
        registry.create(id(7)).unwrap();
        let one = plan_mosh_domains(
            &registry,
            &[request(id(7))],
            [MoshDomainSend {
                stable_item_id: 99,
                domain: id(7),
            }],
            MoshDomainLimits::prototype_720p(),
        )
        .unwrap();
        assert_eq!(one.worker_count, 1);
        assert!(one.domains[0].is_some());
        assert!(one.domains[1].is_none());
    }

    #[test]
    fn two_domain_physical_resources_are_constant_with_layer_count() {
        let mut registry = MoshDomainRegistry::default();
        registry.create(id(1)).unwrap();
        registry.create(id(2)).unwrap();
        let requests = [request(id(1)), request(id(2))];
        let one = plan_mosh_domains(
            &registry,
            &requests,
            [MoshDomainSend {
                stable_item_id: 1,
                domain: id(1),
            }],
            MoshDomainLimits::prototype_720p(),
        )
        .unwrap();
        let many = plan_mosh_domains(
            &registry,
            &requests,
            (0..100_000).map(|index| MoshDomainSend {
                stable_item_id: index,
                domain: if index % 2 == 0 { id(1) } else { id(2) },
            }),
            MoshDomainLimits::prototype_720p(),
        )
        .unwrap();
        assert_eq!(one.total_bytes, many.total_bytes);
        assert_eq!(one.worker_count, many.worker_count);
        assert_ne!(
            many.domains[0].unwrap().mutable_history_owner,
            many.domains[1].unwrap().mutable_history_owner
        );
    }

    #[test]
    fn deletion_is_a_tombstone_and_never_retargets_identity() {
        let mut registry = MoshDomainRegistry::default();
        registry.create(id(1)).unwrap();
        registry.delete(id(1)).unwrap();
        assert_eq!(
            registry.create(id(1)),
            Err(DomainRegistryError::TombstonedIdentity)
        );
        registry.create(id(2)).unwrap();
        assert_eq!(registry.live_count(), 1);
        assert_eq!(
            registry.create(id(3)),
            Err(DomainRegistryError::CapacityExhausted)
        );
    }

    #[test]
    fn byte_worker_and_component_caps_refuse_at_the_exact_boundary() {
        let mut registry = MoshDomainRegistry::default();
        registry.create(id(1)).unwrap();
        let mut limits = MoshDomainLimits::prototype_720p();
        let exact = request(id(1));
        let physical = checked_domain_plan(exact, limits).unwrap().total_bytes();
        limits.max_total_bytes = physical;
        assert!(plan_mosh_domains(&registry, &[exact], [], limits).is_ok());
        limits.max_total_bytes = physical - 1;
        assert_eq!(
            plan_mosh_domains(&registry, &[exact], [], limits),
            Err(MoshDomainPlanError::TotalByteCap)
        );

        let mut over = exact;
        over.job_bytes = PROTOTYPE_MAX_JOB_BYTES + 1;
        assert_eq!(
            plan_mosh_domains(&registry, &[over], [], MoshDomainLimits::prototype_720p()),
            Err(MoshDomainPlanError::JobByteCap)
        );
    }

    #[test]
    fn latest_only_mailboxes_and_domain_histories_are_independent() {
        let mut first = LatestOnlyDomainMailbox::default();
        let mut second = LatestOnlyDomainMailbox::default();
        first
            .offer(
                id(1),
                1,
                DomainJob {
                    domain: id(1),
                    generation: 1,
                    payload: 10,
                },
            )
            .unwrap();
        first
            .offer(
                id(1),
                1,
                DomainJob {
                    domain: id(1),
                    generation: 1,
                    payload: 11,
                },
            )
            .unwrap();
        second
            .offer(
                id(2),
                2,
                DomainJob {
                    domain: id(2),
                    generation: 2,
                    payload: 20,
                },
            )
            .unwrap();
        assert_eq!(first.depth(), 1);
        assert_eq!(first.superseded, 1);
        assert_eq!(first.take().unwrap().payload, 11);
        assert_eq!(second.take().unwrap().payload, 20);
        assert_eq!(
            first.offer(
                id(1),
                2,
                DomainJob {
                    domain: id(1),
                    generation: 1,
                    payload: 12,
                },
            ),
            Err(DomainJobError::StaleGeneration)
        );
        assert_eq!(first.stale_refused, 1);
    }

    #[test]
    fn zero_identity_and_zero_generation_are_refused() {
        assert!(serde_yaml::from_str::<MoshDomainId>("0").is_err());
        let mut registry = MoshDomainRegistry::default();
        registry.create(id(1)).unwrap();
        let mut invalid = request(id(1));
        invalid.generation = 0;
        assert_eq!(
            plan_mosh_domains(
                &registry,
                &[invalid],
                [],
                MoshDomainLimits::prototype_720p()
            ),
            Err(MoshDomainPlanError::InvalidGeneration)
        );
    }
}
