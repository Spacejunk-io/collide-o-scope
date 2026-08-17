//! Host-local media allocation policy.
//!
//! Safe mode preserves collide-o-scope's established UHD-area source limit.
//! Expert mode may admit sources up to DCI-8K area, but only after checked
//! dimension/device validation and reservation of a conservative planning
//! estimate derived from physical memory. The policy intentionally has no
//! patch representation: loading untrusted creative state must never loosen a
//! process-local resource boundary.
//!
//! Portable `wgpu` exposes texture-edge and per-buffer limits, but it does not
//! expose a portable live VRAM budget. Callers must therefore retain recoverable
//! GPU error scopes around the eventual allocation; this module never presents
//! its host-memory plan as a measurement of free VRAM.

use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use serde::{Deserialize, Serialize};

pub const SAFE_MEDIA_MAX_PIXELS: u64 = 3_840 * 2_160;
pub const SAFE_MEDIA_MAX_RGBA_BYTES: u64 = SAFE_MEDIA_MAX_PIXELS * 4;
/// DCI 8K (8192x4320) is the absolute aggregate-area ceiling for Expert mode.
pub const EXPERT_MEDIA_MAX_PIXELS: u64 = 8_192 * 4_320;
pub const EXPERT_MEDIA_MAX_RGBA_BYTES: u64 = EXPERT_MEDIA_MAX_PIXELS * 4;
pub const ABSOLUTE_MEDIA_MAX_EDGE: u32 = 16_384;

const EXPERT_MEMORY_FRACTION_DIVISOR: u64 = 8;
const EXPERT_MEMORY_BUDGET_HARD_MAX: u64 = 2 * 1024 * 1024 * 1024;

/// `wgpu` does not currently offer one portable cross-backend VRAM-budget API.
/// Expose that fact for status surfaces instead of manufacturing a number.
pub const PORTABLE_VRAM_BUDGET_AVAILABLE: bool = false;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaSafetyMode {
    #[default]
    Safe,
    Expert,
}

impl MediaSafetyMode {
    const fn as_u8(self) -> u8 {
        match self {
            Self::Safe => 0,
            Self::Expert => 1,
        }
    }

    const fn from_u8(value: u8) -> Self {
        if value == 1 {
            Self::Expert
        } else {
            Self::Safe
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaSourceKind {
    Video,
    Still,
    Spout,
}

impl MediaSourceKind {
    /// Conservative simultaneous CPU/GPU working-set estimate. This is a
    /// planning weight, not a claim about a codec's exact private allocations.
    const fn working_set_multiplier(self) -> u64 {
        match self {
            // Decoder/scaler image, packed mailbox image, transfer lifetime,
            // and the source texture.
            Self::Video => 4,
            // The image crate may retain format-native working storage while
            // producing RGBA, followed by the held image and source texture.
            Self::Still => 6,
            // Native receive image, converted newest-image slot, transfer
            // lifetime, and the source texture.
            Self::Spout => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaDeviceLimits {
    pub max_texture_dimension_2d: Option<u32>,
    pub max_buffer_size: Option<u64>,
}

impl MediaDeviceLimits {
    pub const fn none() -> Self {
        Self {
            max_texture_dimension_2d: None,
            max_buffer_size: None,
        }
    }

    pub const fn texture_only(max_texture_dimension_2d: u32) -> Self {
        Self {
            max_texture_dimension_2d: Some(max_texture_dimension_2d),
            max_buffer_size: None,
        }
    }

    pub const fn new(max_texture_dimension_2d: u32, max_buffer_size: u64) -> Self {
        Self {
            max_texture_dimension_2d: Some(max_texture_dimension_2d),
            max_buffer_size: Some(max_buffer_size),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaLimitReason {
    InvalidDimensions,
    ArithmeticOverflow,
    AbsoluteEdge,
    DeviceLimitsUnavailable,
    DeviceTextureEdge,
    DeviceBuffer,
    SafeArea,
    ExpertArea,
    MemoryBudgetUnavailable,
    AggregateMemoryBudget,
    AllocationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaLimitError {
    pub reason: MediaLimitReason,
    pub width: u32,
    pub height: u32,
    pub message: String,
}

impl MediaLimitError {
    fn new(reason: MediaLimitReason, width: u32, height: u32, message: impl Into<String>) -> Self {
        Self {
            reason,
            width,
            height,
            message: message.into(),
        }
    }
}

impl fmt::Display for MediaLimitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for MediaLimitError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaAllocationPlan {
    pub source_kind: MediaSourceKind,
    pub width: u32,
    pub height: u32,
    pub pixels: u64,
    pub rgba_bytes: u64,
    /// Conservative combined CPU/GPU planning weight for this source.
    pub working_set_bytes: u64,
    /// True only when the source exceeds the exact Safe-mode UHD-area limit.
    pub requires_expert: bool,
}

impl MediaAllocationPlan {
    /// The still decoder's own allocation ceiling. Other simultaneous source
    /// storage is represented separately by `working_set_bytes`.
    pub fn still_decoder_allocation_limit(&self) -> Result<u64, MediaLimitError> {
        self.rgba_bytes.checked_mul(4).ok_or_else(|| {
            MediaLimitError::new(
                MediaLimitReason::ArithmeticOverflow,
                self.width,
                self.height,
                format!(
                    "still-image decoder allocation estimate overflows for {}x{}",
                    self.width, self.height
                ),
            )
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaSafetySnapshotData {
    pub mode: MediaSafetyMode,
    pub safe_max_pixels: u64,
    pub safe_max_rgba_bytes: u64,
    pub expert_max_pixels: u64,
    pub expert_max_rgba_bytes: u64,
    pub absolute_max_edge: u32,
    pub physical_memory_bytes: Option<u64>,
    pub planning_budget_bytes: u64,
    pub reserved_bytes: u64,
    pub available_planning_bytes: u64,
    pub device_max_texture_dimension_2d: Option<u32>,
    pub device_max_buffer_size: Option<u64>,
    /// Always false until `wgpu` grows a portable cross-backend VRAM budget.
    pub portable_vram_budget_available: bool,
}

impl MediaSafetySnapshotData {
    pub fn rationale(&self) -> String {
        let mode = match self.mode {
            MediaSafetyMode::Safe => format!(
                "Safe mode limits each source to {SAFE_MEDIA_MAX_PIXELS} pixels / {SAFE_MEDIA_MAX_RGBA_BYTES} RGBA bytes"
            ),
            MediaSafetyMode::Expert => format!(
                "Expert mode permits at most {EXPERT_MEDIA_MAX_PIXELS} pixels / {EXPERT_MEDIA_MAX_RGBA_BYTES} RGBA bytes and reserves a conservative host-memory working set"
            ),
        };
        let gpu = self.device_max_texture_dimension_2d.map_or_else(
            || "GPU texture edge not detected yet".to_string(),
            |edge| format!("GPU 2D texture edge {edge}px"),
        );
        format!(
            "{mode}; {gpu}; {} of {} planning bytes reserved. Portable VRAM headroom is unavailable and is verified only by recoverable GPU allocation.",
            self.reserved_bytes, self.planning_budget_bytes
        )
    }
}

struct MediaSafetyInner {
    mode: AtomicU8,
    physical_memory_bytes: Option<u64>,
    planning_budget_bytes: u64,
    reserved_bytes: Mutex<u64>,
}

/// Cloneable, process-local policy shared by decoder, Spout, and export
/// workers. It is intentionally not serializable as a whole.
#[derive(Clone)]
pub struct MediaSafetyPolicy {
    inner: Arc<MediaSafetyInner>,
}

impl fmt::Debug for MediaSafetyPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MediaSafetyPolicy")
            .field("snapshot", &self.snapshot(MediaDeviceLimits::none()))
            .finish()
    }
}

impl Default for MediaSafetyPolicy {
    fn default() -> Self {
        Self::new(MediaSafetyMode::Safe)
    }
}

impl MediaSafetyPolicy {
    pub fn new(mode: MediaSafetyMode) -> Self {
        Self::from_physical_memory(mode, detect_physical_memory_bytes())
    }

    pub fn safe() -> Self {
        Self::new(MediaSafetyMode::Safe)
    }

    fn from_physical_memory(mode: MediaSafetyMode, physical_memory_bytes: Option<u64>) -> Self {
        let planning_budget_bytes = physical_memory_bytes
            .map(planning_budget_for_physical_memory)
            .unwrap_or(0);
        Self {
            inner: Arc::new(MediaSafetyInner {
                mode: AtomicU8::new(mode.as_u8()),
                physical_memory_bytes,
                planning_budget_bytes,
                reserved_bytes: Mutex::new(0),
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(mode: MediaSafetyMode, planning_budget_bytes: u64) -> Self {
        Self {
            inner: Arc::new(MediaSafetyInner {
                mode: AtomicU8::new(mode.as_u8()),
                physical_memory_bytes: Some(
                    planning_budget_bytes.saturating_mul(EXPERT_MEMORY_FRACTION_DIVISOR),
                ),
                planning_budget_bytes,
                reserved_bytes: Mutex::new(0),
            }),
        }
    }

    pub fn mode(&self) -> MediaSafetyMode {
        MediaSafetyMode::from_u8(self.inner.mode.load(Ordering::Acquire))
    }

    /// Change the host-local policy for future source allocations. Existing
    /// reservations remain valid so disabling Expert never destroys a live
    /// layer or makes an accepted video fail at its next loop boundary.
    pub fn set_mode(&self, mode: MediaSafetyMode) -> Result<(), MediaLimitError> {
        if mode == MediaSafetyMode::Expert && self.inner.planning_budget_bytes == 0 {
            return Err(MediaLimitError::new(
                MediaLimitReason::MemoryBudgetUnavailable,
                0,
                0,
                "Expert media mode is unavailable because physical memory could not be detected",
            ));
        }
        self.inner.mode.store(mode.as_u8(), Ordering::Release);
        Ok(())
    }

    pub fn snapshot(&self, device_limits: MediaDeviceLimits) -> MediaSafetySnapshotData {
        let reserved_bytes = *lock_recover(&self.inner.reserved_bytes);
        MediaSafetySnapshotData {
            mode: self.mode(),
            safe_max_pixels: SAFE_MEDIA_MAX_PIXELS,
            safe_max_rgba_bytes: SAFE_MEDIA_MAX_RGBA_BYTES,
            expert_max_pixels: EXPERT_MEDIA_MAX_PIXELS,
            expert_max_rgba_bytes: EXPERT_MEDIA_MAX_RGBA_BYTES,
            absolute_max_edge: ABSOLUTE_MEDIA_MAX_EDGE,
            physical_memory_bytes: self.inner.physical_memory_bytes,
            planning_budget_bytes: self.inner.planning_budget_bytes,
            reserved_bytes,
            available_planning_bytes: self
                .inner
                .planning_budget_bytes
                .saturating_sub(reserved_bytes),
            device_max_texture_dimension_2d: device_limits.max_texture_dimension_2d,
            device_max_buffer_size: device_limits.max_buffer_size,
            portable_vram_budget_available: PORTABLE_VRAM_BUDGET_AVAILABLE,
        }
    }

    pub fn plan(
        &self,
        source_kind: MediaSourceKind,
        width: u32,
        height: u32,
        device_limits: MediaDeviceLimits,
    ) -> Result<MediaAllocationPlan, MediaLimitError> {
        validate_dimensions(
            self.mode(),
            source_kind,
            width,
            height,
            device_limits,
            self.inner.planning_budget_bytes,
        )
    }

    fn reserve(&self, plan: MediaAllocationPlan) -> Result<MediaReservation, MediaLimitError> {
        let reservation_bytes = if plan.requires_expert {
            plan.working_set_bytes
        } else {
            0
        };
        if reservation_bytes == 0 {
            return Ok(MediaReservation {
                inner: None,
                reserved_bytes: 0,
                plan,
            });
        }

        let mut reserved = lock_recover(&self.inner.reserved_bytes);
        let requested_total = reserved.checked_add(reservation_bytes).ok_or_else(|| {
            MediaLimitError::new(
                MediaLimitReason::ArithmeticOverflow,
                plan.width,
                plan.height,
                "aggregate media reservation overflows",
            )
        })?;
        if requested_total > self.inner.planning_budget_bytes {
            let available = self.inner.planning_budget_bytes.saturating_sub(*reserved);
            return Err(MediaLimitError::new(
                MediaLimitReason::AggregateMemoryBudget,
                plan.width,
                plan.height,
                format!(
                    "{}x{} {} source needs about {} planning bytes, but only {available} of the {}-byte Expert media budget remain",
                    plan.width,
                    plan.height,
                    source_kind_name(plan.source_kind),
                    plan.working_set_bytes,
                    self.inner.planning_budget_bytes
                ),
            ));
        }
        *reserved = requested_total;
        drop(reserved);

        Ok(MediaReservation {
            inner: Some(self.inner.clone()),
            reserved_bytes: reservation_bytes,
            plan,
        })
    }

    pub fn reserve_source(
        &self,
        source_kind: MediaSourceKind,
        width: u32,
        height: u32,
        device_limits: MediaDeviceLimits,
    ) -> Result<MediaReservation, MediaLimitError> {
        let plan = self.plan(source_kind, width, height, device_limits)?;
        self.reserve(plan)
    }
}

pub struct MediaReservation {
    inner: Option<Arc<MediaSafetyInner>>,
    reserved_bytes: u64,
    plan: MediaAllocationPlan,
}

impl fmt::Debug for MediaReservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MediaReservation")
            .field("reserved_bytes", &self.reserved_bytes)
            .field("plan", &self.plan)
            .finish()
    }
}

impl MediaReservation {
    pub fn plan(&self) -> &MediaAllocationPlan {
        &self.plan
    }

    #[cfg(test)]
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }
}

impl Drop for MediaReservation {
    fn drop(&mut self) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut reserved = lock_recover(&inner.reserved_bytes);
        *reserved = reserved.saturating_sub(self.reserved_bytes);
    }
}

/// Exact legacy Safe-mode validation without constructing a long-lived policy.
pub fn validate_safe_dimensions(
    source_kind: MediaSourceKind,
    width: u32,
    height: u32,
    device_limits: MediaDeviceLimits,
) -> Result<MediaAllocationPlan, MediaLimitError> {
    validate_dimensions(
        MediaSafetyMode::Safe,
        source_kind,
        width,
        height,
        device_limits,
        0,
    )
}

fn validate_dimensions(
    mode: MediaSafetyMode,
    source_kind: MediaSourceKind,
    width: u32,
    height: u32,
    device_limits: MediaDeviceLimits,
    planning_budget_bytes: u64,
) -> Result<MediaAllocationPlan, MediaLimitError> {
    if width == 0 || height == 0 {
        return Err(MediaLimitError::new(
            MediaLimitReason::InvalidDimensions,
            width,
            height,
            format!("media reported invalid dimensions {width}x{height}"),
        ));
    }
    if width > ABSOLUTE_MEDIA_MAX_EDGE || height > ABSOLUTE_MEDIA_MAX_EDGE {
        return Err(MediaLimitError::new(
            MediaLimitReason::AbsoluteEdge,
            width,
            height,
            format!(
                "media dimensions {width}x{height} exceed the {ABSOLUTE_MEDIA_MAX_EDGE}px absolute safety edge"
            ),
        ));
    }
    if let Some(limit) = device_limits.max_texture_dimension_2d {
        if width > limit || height > limit {
            return Err(MediaLimitError::new(
                MediaLimitReason::DeviceTextureEdge,
                width,
                height,
                format!(
                    "media dimensions {width}x{height} exceed this GPU's {limit}px 2D texture limit"
                ),
            ));
        }
    }

    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| {
            MediaLimitError::new(
                MediaLimitReason::ArithmeticOverflow,
                width,
                height,
                format!("media pixel count overflows for {width}x{height}"),
            )
        })?;
    let rgba_bytes = pixels.checked_mul(4).ok_or_else(|| {
        MediaLimitError::new(
            MediaLimitReason::ArithmeticOverflow,
            width,
            height,
            format!("media RGBA byte size overflows for {width}x{height}"),
        )
    })?;

    let requires_expert = pixels > SAFE_MEDIA_MAX_PIXELS;
    if requires_expert && mode == MediaSafetyMode::Safe {
        return Err(MediaLimitError::new(
            MediaLimitReason::SafeArea,
            width,
            height,
            format!(
                "media dimensions {width}x{height} require {pixels} pixels/{rgba_bytes} RGBA bytes; Safe mode permits {SAFE_MEDIA_MAX_PIXELS} pixels/{SAFE_MEDIA_MAX_RGBA_BYTES} bytes (the area of 3840x2160). Enable the host-local Expert media mode to attempt a larger source."
            ),
        ));
    }
    if requires_expert
        && (device_limits.max_texture_dimension_2d.is_none()
            || device_limits.max_buffer_size.is_none())
    {
        return Err(MediaLimitError::new(
            MediaLimitReason::DeviceLimitsUnavailable,
            width,
            height,
            "Expert media source rejected because actual GPU texture-edge and per-buffer limits are not both available",
        ));
    }
    if pixels > EXPERT_MEDIA_MAX_PIXELS || rgba_bytes > EXPERT_MEDIA_MAX_RGBA_BYTES {
        return Err(MediaLimitError::new(
            MediaLimitReason::ExpertArea,
            width,
            height,
            format!(
                "media dimensions {width}x{height} require {pixels} pixels/{rgba_bytes} RGBA bytes; the absolute Expert ceiling is {EXPERT_MEDIA_MAX_PIXELS} pixels/{EXPERT_MEDIA_MAX_RGBA_BYTES} bytes (DCI-8K area)"
            ),
        ));
    }
    // Safe mode did not historically consult a buffer limit, so retain that
    // exact behavior. Expert admission adds the per-buffer constraint because
    // it is intentionally stricter than merely raising the UHD area cap.
    if let (true, Some(limit)) = (requires_expert, device_limits.max_buffer_size) {
        if rgba_bytes > limit {
            return Err(MediaLimitError::new(
                MediaLimitReason::DeviceBuffer,
                width,
                height,
                format!(
                    "media dimensions {width}x{height} need a {rgba_bytes}-byte RGBA buffer, exceeding this device's {limit}-byte per-buffer limit"
                ),
            ));
        }
    }

    let working_set_bytes = rgba_bytes
        .checked_mul(source_kind.working_set_multiplier())
        .ok_or_else(|| {
            MediaLimitError::new(
                MediaLimitReason::ArithmeticOverflow,
                width,
                height,
                format!("media working-set estimate overflows for {width}x{height}"),
            )
        })?;
    if requires_expert {
        if planning_budget_bytes == 0 {
            return Err(MediaLimitError::new(
                MediaLimitReason::MemoryBudgetUnavailable,
                width,
                height,
                "Expert media source rejected because physical memory could not be detected",
            ));
        }
        if working_set_bytes > planning_budget_bytes {
            return Err(MediaLimitError::new(
                MediaLimitReason::AggregateMemoryBudget,
                width,
                height,
                format!(
                    "{width}x{height} {} source needs about {working_set_bytes} planning bytes, exceeding this host's {planning_budget_bytes}-byte Expert media budget",
                    source_kind_name(source_kind)
                ),
            ));
        }
    }

    Ok(MediaAllocationPlan {
        source_kind,
        width,
        height,
        pixels,
        rgba_bytes,
        working_set_bytes,
        requires_expert,
    })
}

const fn source_kind_name(kind: MediaSourceKind) -> &'static str {
    match kind {
        MediaSourceKind::Video => "video",
        MediaSourceKind::Still => "still-image",
        MediaSourceKind::Spout => "Spout",
    }
}

const fn planning_budget_for_physical_memory(physical_memory_bytes: u64) -> u64 {
    let fraction = physical_memory_bytes / EXPERT_MEMORY_FRACTION_DIVISOR;
    if fraction < EXPERT_MEMORY_BUDGET_HARD_MAX {
        fraction
    } else {
        EXPERT_MEMORY_BUDGET_HARD_MAX
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(windows)]
fn detect_physical_memory_bytes() -> Option<u64> {
    use std::mem::size_of;
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>().try_into().ok()?,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    // SAFETY: `status` has the documented size in `dwLength` and remains valid
    // and exclusively borrowed for the duration of the system call.
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.ullTotalPhys)
}

#[cfg(unix)]
fn detect_physical_memory_bytes() -> Option<u64> {
    // SAFETY: `sysconf` has no pointer arguments or retained state. Negative
    // results are the documented unsupported/error signal and are rejected.
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    // SAFETY: same as above.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if pages <= 0 || page_size <= 0 {
        return None;
    }
    u64::try_from(pages)
        .ok()?
        .checked_mul(u64::try_from(page_size).ok()?)
}

#[cfg(not(any(windows, unix)))]
fn detect_physical_memory_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const LARGE_BUDGET: u64 = 2 * 1024 * 1024 * 1024;
    const GENEROUS_DEVICE: MediaDeviceLimits = MediaDeviceLimits {
        max_texture_dimension_2d: Some(16_384),
        max_buffer_size: Some(512 * 1024 * 1024),
    };

    #[test]
    fn safe_mode_is_exact_legacy_uhd_area_policy() {
        let policy = MediaSafetyPolicy::for_test(MediaSafetyMode::Safe, LARGE_BUDGET);
        assert!(policy
            .plan(MediaSourceKind::Video, 3_840, 2_160, GENEROUS_DEVICE)
            .is_ok());
        assert!(policy
            .plan(MediaSourceKind::Video, 8_192, 1, GENEROUS_DEVICE)
            .is_ok());
        let error = policy
            .plan(MediaSourceKind::Video, 3_840, 2_161, GENEROUS_DEVICE)
            .unwrap_err();
        assert_eq!(error.reason, MediaLimitReason::SafeArea);
        assert!(error.to_string().contains("Safe mode"));
    }

    #[test]
    fn expert_accepts_5k_and_both_common_8k_areas_with_budget() {
        let policy = MediaSafetyPolicy::for_test(MediaSafetyMode::Expert, LARGE_BUDGET);
        for (width, height) in [(5_120, 2_880), (7_680, 4_320), (8_192, 4_320)] {
            let plan = policy
                .plan(MediaSourceKind::Video, width, height, GENEROUS_DEVICE)
                .unwrap();
            assert!(plan.requires_expert);
            assert_eq!(plan.rgba_bytes, u64::from(width) * u64::from(height) * 4);
        }
    }

    #[test]
    fn expert_remains_bounded_by_area_edge_buffer_and_memory() {
        let policy = MediaSafetyPolicy::for_test(MediaSafetyMode::Expert, LARGE_BUDGET);
        assert_eq!(
            policy
                .plan(
                    MediaSourceKind::Video,
                    5_120,
                    2_880,
                    MediaDeviceLimits::none(),
                )
                .unwrap_err()
                .reason,
            MediaLimitReason::DeviceLimitsUnavailable
        );
        assert_eq!(
            policy
                .plan(MediaSourceKind::Video, 8_193, 4_320, GENEROUS_DEVICE)
                .unwrap_err()
                .reason,
            MediaLimitReason::ExpertArea
        );
        assert_eq!(
            policy
                .plan(
                    MediaSourceKind::Video,
                    8_192,
                    4_320,
                    MediaDeviceLimits::new(8_000, 512 * 1024 * 1024),
                )
                .unwrap_err()
                .reason,
            MediaLimitReason::DeviceTextureEdge
        );
        assert_eq!(
            policy
                .plan(
                    MediaSourceKind::Video,
                    7_680,
                    4_320,
                    MediaDeviceLimits::new(8_192, 64 * 1024 * 1024),
                )
                .unwrap_err()
                .reason,
            MediaLimitReason::DeviceBuffer
        );

        let small = MediaSafetyPolicy::for_test(MediaSafetyMode::Expert, 256 * 1024 * 1024);
        assert_eq!(
            small
                .plan(MediaSourceKind::Still, 5_120, 2_880, GENEROUS_DEVICE)
                .unwrap_err()
                .reason,
            MediaLimitReason::AggregateMemoryBudget
        );
    }

    #[test]
    fn checked_math_and_absolute_edge_reject_hostile_dimensions() {
        let policy = MediaSafetyPolicy::for_test(MediaSafetyMode::Expert, LARGE_BUDGET);
        assert_eq!(
            policy
                .plan(MediaSourceKind::Video, 0, 1, GENEROUS_DEVICE)
                .unwrap_err()
                .reason,
            MediaLimitReason::InvalidDimensions
        );
        assert_eq!(
            policy
                .plan(MediaSourceKind::Video, u32::MAX, u32::MAX, GENEROUS_DEVICE)
                .unwrap_err()
                .reason,
            MediaLimitReason::AbsoluteEdge
        );

        let synthetic_overflow = MediaAllocationPlan {
            source_kind: MediaSourceKind::Still,
            width: 1,
            height: 1,
            pixels: 1,
            rgba_bytes: u64::MAX,
            working_set_bytes: u64::MAX,
            requires_expert: true,
        };
        assert_eq!(
            synthetic_overflow
                .still_decoder_allocation_limit()
                .unwrap_err()
                .reason,
            MediaLimitReason::ArithmeticOverflow
        );
    }

    #[test]
    fn aggregate_reservation_addition_is_checked() {
        let policy = MediaSafetyPolicy::for_test(MediaSafetyMode::Expert, u64::MAX);
        let first = MediaAllocationPlan {
            source_kind: MediaSourceKind::Video,
            width: 1,
            height: 1,
            pixels: 1,
            rgba_bytes: 4,
            working_set_bytes: u64::MAX - 1,
            requires_expert: true,
        };
        let _reservation = policy.reserve(first).unwrap();
        let second = MediaAllocationPlan {
            source_kind: MediaSourceKind::Video,
            width: 1,
            height: 1,
            pixels: 1,
            rgba_bytes: 4,
            working_set_bytes: 2,
            requires_expert: true,
        };
        assert_eq!(
            policy.reserve(second).unwrap_err().reason,
            MediaLimitReason::ArithmeticOverflow
        );
    }

    #[test]
    fn expert_reservations_are_aggregate_and_release_on_drop() {
        let rgba = 5_120_u64 * 2_880 * 4;
        let one_video = rgba * MediaSourceKind::Video.working_set_multiplier();
        let policy = MediaSafetyPolicy::for_test(MediaSafetyMode::Expert, one_video * 2);
        let plan = policy
            .plan(MediaSourceKind::Video, 5_120, 2_880, GENEROUS_DEVICE)
            .unwrap();

        let first = policy.reserve(plan.clone()).unwrap();
        let second = policy.reserve(plan.clone()).unwrap();
        assert_eq!(
            policy.snapshot(GENEROUS_DEVICE).reserved_bytes,
            one_video * 2
        );
        assert_eq!(
            policy.reserve(plan.clone()).unwrap_err().reason,
            MediaLimitReason::AggregateMemoryBudget
        );
        drop(first);
        let third = policy.reserve(plan).unwrap();
        assert_eq!(third.reserved_bytes(), one_video);
        drop((second, third));
        assert_eq!(policy.snapshot(GENEROUS_DEVICE).reserved_bytes, 0);
    }

    #[test]
    fn disabling_expert_keeps_existing_reservations_but_restores_safe_admission() {
        let policy = MediaSafetyPolicy::for_test(MediaSafetyMode::Expert, LARGE_BUDGET);
        let reservation = policy
            .reserve_source(MediaSourceKind::Video, 5_120, 2_880, GENEROUS_DEVICE)
            .unwrap();
        policy.set_mode(MediaSafetyMode::Safe).unwrap();
        assert!(policy.snapshot(GENEROUS_DEVICE).reserved_bytes > 0);
        assert_eq!(
            policy
                .plan(MediaSourceKind::Video, 5_120, 2_880, GENEROUS_DEVICE)
                .unwrap_err()
                .reason,
            MediaLimitReason::SafeArea
        );
        drop(reservation);
        assert_eq!(policy.snapshot(GENEROUS_DEVICE).reserved_bytes, 0);
    }

    #[test]
    fn snapshot_is_explicit_that_vram_is_not_portably_reported() {
        let policy = MediaSafetyPolicy::for_test(MediaSafetyMode::Safe, LARGE_BUDGET);
        let snapshot = policy.snapshot(GENEROUS_DEVICE);
        assert!(!snapshot.portable_vram_budget_available);
        assert!(snapshot
            .rationale()
            .contains("Portable VRAM headroom is unavailable"));
    }
}
