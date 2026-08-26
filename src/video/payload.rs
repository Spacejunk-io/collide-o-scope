//! Immutable, reference-counted decoded-image ownership.
//!
//! Each physical packed allocation owns exactly one aggregate media-ledger
//! lease. Logical forward/cache/upload/read-only handles only increment owner
//! counters. When the final handle drops, an allocation may return to its
//! decoder's bounded exclusive pool; mutation is therefore impossible while
//! any consumer can still read the pixels.

use std::fmt;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use crate::media_safety::{DecodedImageLease, DecodedImageLedger};

use super::planar::PlanarConversionRecipe;

static NEXT_PAYLOAD_ID: AtomicU64 = AtomicU64::new(1);

/// The bounded decoded-image vocabulary. `PackedRgba8` is the exact legacy
/// delivery; `PlanarYuv420p8` is the P4c admitted planar delivery — tightly
/// packed Y, U, V planes in one allocation, converted on the GPU at the
/// upload seam under the recipe the payload carries. Every byte-consuming
/// site must dispatch on this format; [`DecodedImagePayload::expect_packed_rgba8`]
/// is the fail-closed accessor for paths that can only mean packed pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodedPixelFormat {
    PackedRgba8,
    PlanarYuv420p8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodedRasterLayout {
    pub width: u32,
    pub height: u32,
    /// Luma/packed row bytes. For `PlanarYuv420p8` this is the Y plane's
    /// row width; chroma geometry derives from the frame dimensions.
    pub stride: usize,
    pub format: DecodedPixelFormat,
}

impl DecodedRasterLayout {
    pub fn packed_rgba8(width: u32, height: u32) -> Result<Self, String> {
        let stride = usize::try_from(width)
            .map_err(|_| "decoded image width does not fit this platform".to_string())?
            .checked_mul(4)
            .ok_or_else(|| "decoded image row byte count overflows".to_string())?;
        let _ = stride
            .checked_mul(usize::try_from(height).unwrap_or(usize::MAX))
            .ok_or_else(|| "decoded image byte count overflows".to_string())?;
        Ok(Self {
            width,
            height,
            stride,
            format: DecodedPixelFormat::PackedRgba8,
        })
    }

    pub fn planar_yuv420p8(width: u32, height: u32) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("planar decoded image dimensions cannot be zero".to_string());
        }
        let stride = usize::try_from(width)
            .map_err(|_| "decoded image width does not fit this platform".to_string())?;
        let layout = Self {
            width,
            height,
            stride,
            format: DecodedPixelFormat::PlanarYuv420p8,
        };
        // Validate the complete plane arithmetic once at construction.
        let _ = layout.byte_len()?;
        Ok(layout)
    }

    /// Chroma plane texel dimensions for the 4:2:0 family (ceil halves).
    pub fn chroma_dimensions(self) -> (usize, usize) {
        let half = |value: u32| {
            let value = usize::try_from(value).unwrap_or(usize::MAX);
            value / 2 + value % 2
        };
        (half(self.width), half(self.height))
    }

    pub fn byte_len(self) -> Result<usize, String> {
        let height = usize::try_from(self.height)
            .map_err(|_| "decoded image height does not fit this platform".to_string())?;
        match self.format {
            DecodedPixelFormat::PackedRgba8 => self
                .stride
                .checked_mul(height)
                .ok_or_else(|| "decoded image byte count overflows".to_string()),
            DecodedPixelFormat::PlanarYuv420p8 => {
                let (chroma_width, chroma_height) = self.chroma_dimensions();
                self.stride
                    .checked_mul(height)
                    .and_then(|luma| {
                        chroma_width
                            .checked_mul(chroma_height)
                            .and_then(|chroma| chroma.checked_mul(2))
                            .and_then(|chroma| luma.checked_add(chroma))
                    })
                    .ok_or_else(|| "decoded image byte count overflows".to_string())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum DecodedPayloadOwner {
    Forward = 0,
    ReverseCache = 1,
    Upload = 2,
    ReadOnly = 3,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodedPayloadOwnerSnapshot {
    pub forward: u64,
    pub reverse_cache: u64,
    pub upload: u64,
    pub read_only: u64,
}

struct PoolAllocationLease {
    pool: Weak<RasterPoolInner>,
    bytes: u64,
}

impl Drop for PoolAllocationLease {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            pool.owned_bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

struct RecycledRaster {
    bytes: Vec<u8>,
    _aggregate_lease: Option<DecodedImageLease>,
    _pool_lease: Option<PoolAllocationLease>,
}

#[derive(Default)]
struct RasterPoolState {
    idle: Vec<RecycledRaster>,
}

struct RasterPoolInner {
    layout: DecodedRasterLayout,
    max_idle_slots: usize,
    max_physical_bytes: u64,
    owned_bytes: AtomicU64,
    state: Mutex<RasterPoolState>,
    ledger: Arc<DecodedImageLedger>,
}

/// One decoder's format/stride-specific bounded recycle pool.
#[derive(Clone)]
pub(crate) struct DecodedRasterPool {
    inner: Arc<RasterPoolInner>,
}

impl fmt::Debug for DecodedRasterPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = lock_recover(&self.inner.state);
        formatter
            .debug_struct("DecodedRasterPool")
            .field("layout", &self.inner.layout)
            .field("idle_slots", &state.idle.len())
            .field(
                "owned_bytes",
                &self.inner.owned_bytes.load(Ordering::Acquire),
            )
            .field("max_physical_bytes", &self.inner.max_physical_bytes)
            .finish()
    }
}

impl DecodedRasterPool {
    pub(crate) fn new(
        layout: DecodedRasterLayout,
        max_idle_slots: usize,
        max_physical_bytes: u64,
        ledger: Arc<DecodedImageLedger>,
    ) -> Result<Self, String> {
        let bytes = u64::try_from(layout.byte_len()?)
            .map_err(|_| "decoded image byte count does not fit u64".to_string())?;
        if bytes == 0 {
            return Err("decoded image pool cannot admit an empty raster".to_string());
        }
        if max_physical_bytes < bytes {
            return Err(format!(
                "decoded image pool limit {max_physical_bytes} is smaller than one {bytes}-byte raster"
            ));
        }
        Ok(Self {
            inner: Arc::new(RasterPoolInner {
                layout,
                max_idle_slots,
                max_physical_bytes,
                owned_bytes: AtomicU64::new(0),
                state: Mutex::new(RasterPoolState::default()),
                ledger,
            }),
        })
    }

    /// Repack one FFmpeg plane into an exclusive pooled buffer, then freeze it
    /// behind an immutable Arc. The only pixel copies here are the unavoidable
    /// format/stride materialization; sharing and reverse-cache operations copy
    /// zero pixel bytes.
    pub(crate) fn materialize_plane(
        &self,
        source: &[u8],
        source_stride: usize,
    ) -> Result<DecodedImagePayload, String> {
        let layout = self.inner.layout;
        let output_len = layout.byte_len()?;
        let height = usize::try_from(layout.height)
            .map_err(|_| "decoded image height does not fit this platform".to_string())?;
        let required_source_len = source_stride
            .checked_mul(height)
            .ok_or_else(|| "decoded source plane byte count overflows".to_string())?;
        if source_stride < layout.stride || source.len() < required_source_len {
            return Err(format!(
                "invalid decoded RGBA plane: stride={source_stride}, row_bytes={}, height={height}, data_len={}",
                layout.stride,
                source.len()
            ));
        }

        let mut raster = self.acquire(output_len)?;
        raster.bytes.clear();
        if source_stride == layout.stride {
            raster.bytes.extend_from_slice(&source[..output_len]);
        } else {
            for row in source.chunks_exact(source_stride).take(height) {
                raster.bytes.extend_from_slice(&row[..layout.stride]);
            }
        }
        debug_assert_eq!(raster.bytes.len(), output_len);
        self.inner
            .ledger
            .record_materialization_copy(u64::try_from(output_len).unwrap_or(u64::MAX));
        Ok(DecodedImagePayload::from_recycled(
            raster,
            layout,
            &self.inner,
            DecodedPayloadOwner::Forward,
            None,
        ))
    }

    /// Repack the decoder's three yuv420p planes into one exclusive pooled
    /// allocation (tightly packed Y, then U, then V — the P4c contract
    /// packing), then freeze it with its conversion recipe attached. Same
    /// recycling, ledger, and ownership law as the packed materializer; the
    /// only pixel copies are the stride-dropping row copies.
    pub(crate) fn materialize_yuv420p_planes(
        &self,
        luma: (&[u8], usize),
        chroma_u: (&[u8], usize),
        chroma_v: (&[u8], usize),
        recipe: PlanarConversionRecipe,
    ) -> Result<DecodedImagePayload, String> {
        let layout = self.inner.layout;
        if layout.format != DecodedPixelFormat::PlanarYuv420p8 {
            return Err("this pool does not materialize planar frames".to_string());
        }
        let output_len = layout.byte_len()?;
        let height = usize::try_from(layout.height)
            .map_err(|_| "decoded image height does not fit this platform".to_string())?;
        let (chroma_width, chroma_height) = layout.chroma_dimensions();

        let validate = |label: &str,
                        (data, stride): (&[u8], usize),
                        row_bytes: usize,
                        rows: usize|
         -> Result<(), String> {
            let required = stride
                .checked_mul(rows.saturating_sub(1))
                .and_then(|prefix| prefix.checked_add(row_bytes))
                .ok_or_else(|| format!("decoded {label} plane byte count overflows"))?;
            if stride < row_bytes || data.len() < required {
                return Err(format!(
                    "invalid decoded {label} plane: stride={stride}, row_bytes={row_bytes}, rows={rows}, data_len={}",
                    data.len()
                ));
            }
            Ok(())
        };
        validate("luma", luma, layout.stride, height)?;
        validate("chroma-u", chroma_u, chroma_width, chroma_height)?;
        validate("chroma-v", chroma_v, chroma_width, chroma_height)?;

        let mut raster = self.acquire(output_len)?;
        raster.bytes.clear();
        let mut pack = |(data, stride): (&[u8], usize), row_bytes: usize, rows: usize| {
            if stride == row_bytes {
                raster
                    .bytes
                    .extend_from_slice(&data[..row_bytes.saturating_mul(rows)]);
            } else {
                for row in data.chunks(stride).take(rows) {
                    raster.bytes.extend_from_slice(&row[..row_bytes]);
                }
            }
        };
        pack(luma, layout.stride, height);
        pack(chroma_u, chroma_width, chroma_height);
        pack(chroma_v, chroma_width, chroma_height);
        debug_assert_eq!(raster.bytes.len(), output_len);
        self.inner
            .ledger
            .record_materialization_copy(u64::try_from(output_len).unwrap_or(u64::MAX));
        Ok(DecodedImagePayload::from_recycled(
            raster,
            layout,
            &self.inner,
            DecodedPayloadOwner::Forward,
            Some(recipe),
        ))
    }

    pub(crate) fn ledger(&self) -> Arc<DecodedImageLedger> {
        self.inner.ledger.clone()
    }

    fn acquire(&self, output_len: usize) -> Result<RecycledRaster, String> {
        if let Some(raster) = lock_recover(&self.inner.state).idle.pop() {
            debug_assert!(raster.bytes.capacity() >= output_len);
            self.inner.ledger.record_reuse();
            return Ok(raster);
        }

        let mut buffer = Vec::new();
        if let Err(error) = buffer.try_reserve_exact(output_len) {
            return Err(format!(
                "could not reserve {output_len} bytes for packed decoded image: {error}"
            ));
        }
        // Vec may receive more capacity than requested. Charge the allocation
        // the engine can actually retain, not merely its logical image length.
        let bytes = u64::try_from(buffer.capacity())
            .map_err(|_| "decoded image allocation capacity does not fit u64".to_string())?;
        reserve_pool_bytes(&self.inner, bytes)?;
        let pool_lease = PoolAllocationLease {
            pool: Arc::downgrade(&self.inner),
            bytes,
        };
        let aggregate_lease = self
            .inner
            .ledger
            .try_reserve(bytes)
            .map_err(str::to_string)?;
        self.inner.ledger.record_allocation();
        Ok(RecycledRaster {
            bytes: buffer,
            _aggregate_lease: Some(aggregate_lease),
            _pool_lease: Some(pool_lease),
        })
    }

    #[cfg(test)]
    pub(crate) fn idle_slots(&self) -> usize {
        lock_recover(&self.inner.state).idle.len()
    }
}

fn reserve_pool_bytes(pool: &RasterPoolInner, bytes: u64) -> Result<(), String> {
    let mut current = pool.owned_bytes.load(Ordering::Acquire);
    loop {
        let requested = current
            .checked_add(bytes)
            .ok_or_else(|| "decoded image pool byte accounting overflow".to_string())?;
        if requested > pool.max_physical_bytes {
            return Err("decoded image pool is fully retained by live consumers".to_string());
        }
        match pool.owned_bytes.compare_exchange_weak(
            current,
            requested,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return Ok(()),
            Err(observed) => current = observed,
        }
    }
}

struct DecodedImagePayloadInner {
    identity: u64,
    layout: DecodedRasterLayout,
    raster: Option<RecycledRaster>,
    pool: Weak<RasterPoolInner>,
    owners: [AtomicU64; 4],
    ledger: Option<Arc<DecodedImageLedger>>,
    /// Present exactly on planar payloads: the frame-local conversion law
    /// the upload seam applies. Packed payloads carry `None`.
    conversion_recipe: Option<PlanarConversionRecipe>,
}

impl Drop for DecodedImagePayloadInner {
    fn drop(&mut self) {
        let Some(raster) = self.raster.take() else {
            return;
        };
        let Some(pool) = self.pool.upgrade() else {
            return;
        };
        if pool.layout != self.layout {
            return;
        }
        let mut state = lock_recover(&pool.state);
        if state.idle.len() < pool.max_idle_slots {
            state.idle.push(raster);
        }
    }
}

/// Immutable Arc-owned packed image. Clone is a logical-owner clone, never a
/// pixel clone; callers should use [`Self::share_as`] when the new role is
/// known so diagnostics can explain retention precisely.
pub struct DecodedImagePayload {
    inner: Arc<DecodedImagePayloadInner>,
    owner: DecodedPayloadOwner,
}

impl DecodedImagePayload {
    fn from_recycled(
        raster: RecycledRaster,
        layout: DecodedRasterLayout,
        pool: &Arc<RasterPoolInner>,
        owner: DecodedPayloadOwner,
        conversion_recipe: Option<PlanarConversionRecipe>,
    ) -> Self {
        let ledger = Some(pool.ledger.clone());
        let inner = Arc::new(DecodedImagePayloadInner {
            identity: NEXT_PAYLOAD_ID.fetch_add(1, Ordering::Relaxed).max(1),
            layout,
            raster: Some(raster),
            pool: Arc::downgrade(pool),
            owners: std::array::from_fn(|_| AtomicU64::new(0)),
            ledger,
            conversion_recipe,
        });
        let payload = Self { inner, owner };
        payload.add_owner();
        payload
    }

    /// Compatibility constructor for immutable still/test images that already
    /// own their exact bytes. Decoder allocations use the ledgered pool path.
    pub fn from_owned_rgba(bytes: Vec<u8>) -> Self {
        let stride = bytes.len();
        let inner = Arc::new(DecodedImagePayloadInner {
            identity: NEXT_PAYLOAD_ID.fetch_add(1, Ordering::Relaxed).max(1),
            layout: DecodedRasterLayout {
                width: 0,
                height: u32::from(!bytes.is_empty()),
                stride,
                format: DecodedPixelFormat::PackedRgba8,
            },
            raster: Some(RecycledRaster {
                bytes,
                _aggregate_lease: None,
                _pool_lease: None,
            }),
            pool: Weak::new(),
            owners: std::array::from_fn(|_| AtomicU64::new(0)),
            ledger: None,
            conversion_recipe: None,
        });
        let payload = Self {
            inner,
            owner: DecodedPayloadOwner::Forward,
        };
        payload.add_owner();
        payload
    }

    pub fn share_as(&self, owner: DecodedPayloadOwner) -> Self {
        let shared = Self {
            inner: self.inner.clone(),
            owner,
        };
        shared.add_owner();
        shared
    }

    pub fn identity(&self) -> u64 {
        self.inner.identity
    }

    pub fn layout(&self) -> DecodedRasterLayout {
        self.inner.layout
    }

    /// The frame-local conversion law of a planar payload; `None` for packed.
    pub fn conversion_recipe(&self) -> Option<PlanarConversionRecipe> {
        self.inner.conversion_recipe
    }

    /// Fail-closed accessor for paths whose bytes can only mean packed RGBA.
    /// A planar payload reaching such a path is a routing defect, and this
    /// turns it into a typed refusal instead of silently misread pixels.
    pub fn expect_packed_rgba8(&self) -> Result<&[u8], String> {
        match self.inner.layout.format {
            DecodedPixelFormat::PackedRgba8 => Ok(self.as_slice()),
            DecodedPixelFormat::PlanarYuv420p8 => {
                Err("planar decoded frame reached a packed-RGBA-only consumer".to_string())
            }
        }
    }

    pub fn owner_snapshot(&self) -> DecodedPayloadOwnerSnapshot {
        let load = |index: usize| self.inner.owners[index].load(Ordering::Relaxed);
        DecodedPayloadOwnerSnapshot {
            forward: load(DecodedPayloadOwner::Forward as usize),
            reverse_cache: load(DecodedPayloadOwner::ReverseCache as usize),
            upload: load(DecodedPayloadOwner::Upload as usize),
            read_only: load(DecodedPayloadOwner::ReadOnly as usize),
        }
    }

    pub(crate) fn record_invalidation(&self) {
        if let Some(ledger) = &self.inner.ledger {
            ledger.record_invalidation();
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.inner
            .raster
            .as_ref()
            .map_or(&[], |raster| raster.bytes.as_slice())
    }

    /// Recover an owned Vec when this is the final handle. Shared payloads use
    /// an explicit copy; this compatibility path is never used by reverse
    /// cache insertion/hits or decoded uploads.
    pub fn into_vec(self) -> Vec<u8> {
        if let Some(ledger) = &self.inner.ledger {
            ledger.record_reference_copy(u64::try_from(self.len()).unwrap_or(u64::MAX));
        }
        self.as_slice().to_vec()
    }

    fn add_owner(&self) {
        let index = self.owner as usize;
        self.inner.owners[index].fetch_add(1, Ordering::Relaxed);
        if let Some(ledger) = &self.inner.ledger {
            ledger.add_owner(index);
        }
    }

    fn remove_owner(&self) {
        let index = self.owner as usize;
        let _ =
            self.inner.owners[index].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                Some(value.saturating_sub(1))
            });
        if let Some(ledger) = &self.inner.ledger {
            ledger.remove_owner(index);
        }
    }
}

impl Clone for DecodedImagePayload {
    fn clone(&self) -> Self {
        self.share_as(self.owner)
    }
}

impl Drop for DecodedImagePayload {
    fn drop(&mut self) {
        self.remove_owner();
    }
}

impl Deref for DecodedImagePayload {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[u8]> for DecodedImagePayload {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl fmt::Debug for DecodedImagePayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodedImagePayload")
            .field("identity", &self.identity())
            .field("layout", &self.layout())
            .field("bytes", &self.len())
            .field("owner", &self.owner)
            .field("owners", &self.owner_snapshot())
            .finish()
    }
}

impl PartialEq for DecodedImagePayload {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for DecodedImagePayload {}

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(ledger: Arc<DecodedImageLedger>, max_physical_bytes: u64) -> DecodedRasterPool {
        DecodedRasterPool::new(
            DecodedRasterLayout::packed_rgba8(2, 2).unwrap(),
            2,
            max_physical_bytes,
            ledger,
        )
        .unwrap()
    }

    #[test]
    fn sharing_preserves_identity_and_charges_physical_bytes_once() {
        let ledger = DecodedImageLedger::new(64);
        let pool = pool(ledger.clone(), 64);
        let forward = pool.materialize_plane(&[7; 16], 8).unwrap();
        let cache = forward.share_as(DecodedPayloadOwner::ReverseCache);
        let upload = forward.share_as(DecodedPayloadOwner::Upload);
        let read = forward.share_as(DecodedPayloadOwner::ReadOnly);

        assert_eq!(forward.identity(), cache.identity());
        assert_eq!(forward.identity(), upload.identity());
        assert_eq!(forward.identity(), read.identity());
        assert_eq!(
            forward.owner_snapshot(),
            DecodedPayloadOwnerSnapshot {
                forward: 1,
                reverse_cache: 1,
                upload: 1,
                read_only: 1,
            }
        );
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.physical_bytes, 16);
        assert_eq!(snapshot.allocations, 1);
        assert_eq!(snapshot.reference_copied_bytes, 0);
        assert_eq!(snapshot.forward_owners, 1);
        assert_eq!(snapshot.cache_owners, 1);
        assert_eq!(snapshot.upload_owners, 1);
        assert_eq!(snapshot.readonly_owners, 1);
    }

    #[test]
    fn warm_pool_reuses_without_new_allocation_or_overwriting_live_arc() {
        let ledger = DecodedImageLedger::new(32);
        let pool = pool(ledger.clone(), 32);
        let first = pool.materialize_plane(&[1; 16], 8).unwrap();
        let retained = first.share_as(DecodedPayloadOwner::ReverseCache);
        drop(first);
        let second = pool.materialize_plane(&[2; 16], 8).unwrap();
        assert_eq!(retained.as_slice(), &[1; 16]);
        assert_eq!(second.as_slice(), &[2; 16]);
        assert!(pool.materialize_plane(&[3; 16], 8).is_err());

        drop(retained);
        let allocations = ledger.snapshot().allocations;
        let third = pool.materialize_plane(&[3; 16], 8).unwrap();
        assert_eq!(third.as_slice(), &[3; 16]);
        assert_eq!(ledger.snapshot().allocations, allocations);
        assert_eq!(ledger.snapshot().reuses, 1);
    }

    #[test]
    fn aggregate_ledger_refuses_exactly_one_byte_over_cap() {
        let ledger = DecodedImageLedger::new(15);
        let pool = pool(ledger.clone(), 16);
        let error = pool.materialize_plane(&[0; 16], 8).unwrap_err();
        assert_eq!(error, "aggregate decoded image budget exceeded");
        assert_eq!(ledger.snapshot().physical_bytes, 0);
        assert_eq!(ledger.snapshot().allocations, 0);
    }

    #[test]
    fn source_retirement_releases_pool_and_payload_to_baseline() {
        let ledger = DecodedImageLedger::new(64);
        {
            let pool = pool(ledger.clone(), 64);
            let payload = pool.materialize_plane(&[9; 16], 8).unwrap();
            let cache = payload.share_as(DecodedPayloadOwner::ReverseCache);
            drop(payload);
            drop(cache);
            assert_eq!(pool.idle_slots(), 1);
            assert_eq!(ledger.snapshot().physical_bytes, 16);
        }
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.physical_bytes, 0);
        assert_eq!(snapshot.forward_owners, 0);
        assert_eq!(snapshot.cache_owners, 0);
        assert_eq!(snapshot.upload_owners, 0);
        assert_eq!(snapshot.readonly_owners, 0);
    }

    #[test]
    fn planar_pool_materializes_strided_planes_with_recipe_and_honest_bytes() {
        let recipe = PlanarConversionRecipe {
            bit_depth: 8,
            full_range: false,
            chroma_offset: [0.0, 0.5],
            kr: 0.2126,
            kb: 0.0722,
        };
        let layout = DecodedRasterLayout::planar_yuv420p8(4, 2).unwrap();
        assert_eq!(layout.byte_len().unwrap(), 12);
        assert_eq!(layout.chroma_dimensions(), (2, 1));

        let ledger = DecodedImageLedger::new(64);
        let planar_pool = DecodedRasterPool::new(layout, 2, 64, ledger.clone()).unwrap();
        // Strided planes: padding bytes (99) must not be retained.
        let luma = [1, 2, 3, 4, 99, 99, 5, 6, 7, 8, 99, 99];
        let chroma_u = [110, 111, 99];
        let chroma_v = [120, 121, 99];
        let payload = planar_pool
            .materialize_yuv420p_planes((&luma, 6), (&chroma_u, 3), (&chroma_v, 3), recipe)
            .unwrap();
        assert_eq!(
            payload.as_slice(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 110, 111, 120, 121]
        );
        assert_eq!(payload.layout().format, DecodedPixelFormat::PlanarYuv420p8);
        assert_eq!(payload.conversion_recipe(), Some(recipe));
        // The ledger charges the actual planar bytes — 1.5 per pixel, not 4.
        assert_eq!(ledger.snapshot().physical_bytes, 12);

        // A planar payload can never be misread as packed pixels.
        assert!(payload.expect_packed_rgba8().is_err());
        let clone = payload.clone();
        assert_eq!(payload.identity(), clone.identity());

        // Recycling: dropping the final handle returns the allocation.
        drop(payload);
        drop(clone);
        let allocations = ledger.snapshot().allocations;
        let second = planar_pool
            .materialize_yuv420p_planes((&luma, 6), (&chroma_u, 3), (&chroma_v, 3), recipe)
            .unwrap();
        assert_eq!(ledger.snapshot().allocations, allocations);
        assert_eq!(second.as_slice()[..4], [1, 2, 3, 4]);

        // A short plane is a typed refusal before any allocation.
        assert!(planar_pool
            .materialize_yuv420p_planes((&luma[..7], 6), (&chroma_u, 3), (&chroma_v, 3), recipe)
            .is_err());
        // The packed pool refuses planar materialization outright.
        let packed_pool = pool(DecodedImageLedger::new(64), 64);
        assert!(packed_pool
            .materialize_yuv420p_planes((&luma, 6), (&chroma_u, 3), (&chroma_v, 3), recipe)
            .is_err());
    }

    #[test]
    fn padded_rows_repack_into_same_frozen_legacy_bytes() {
        let ledger = DecodedImageLedger::new(64);
        let pool = pool(ledger.clone(), 64);
        let source = [
            1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 9, 10, 11, 12, 13, 14, 15, 16, 99, 99,
        ];
        let payload = pool.materialize_plane(&source, 10).unwrap();
        assert_eq!(
            payload.as_slice(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        assert_eq!(ledger.snapshot().materialization_copied_bytes, 16);
        assert_eq!(ledger.snapshot().reference_copied_bytes, 0);
    }
}
