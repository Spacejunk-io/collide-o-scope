//! B5 codec mosh: a real encoder and decoder wired back to back in-process,
//! with the bitstream broken between them.
//!
//! Nothing here is a shader imitating a codec — the artefacts are the
//! decoder's own. The stage laws are derived from BENDR (MIT, © 2026 Steve
//! Blythe), whose codec stage (`p42_capture.js`) settles every control's
//! semantics; this module is a rewrite for FFmpeg's library codecs with one
//! deliberate deviation: BENDR's `Math.random()` fault clock becomes the
//! shared deterministic avalanche hash keyed by the master `random_seed` and
//! the stage's 30 Hz reference ordinal, because our export contract demands a
//! replayable fault stream and BENDR explicitly disclaims one (it disables
//! the stage offline outright).
//!
//! The pure half of this module — parameter sanitize, the bitrate and resync
//! laws, the per-chunk fault decisions, and the bounded chunk ring — has no
//! FFmpeg dependency and is the reference the round trip follows. The engine
//! half owns two `ffmpeg-next` codec contexts (`mpeg4`, `threads = 1` — the
//! macroblock artefacts are the right artefacts, and single-threaded encode
//! is what makes two renders on one host byte-identical) plus the two
//! software scalers between RGBA and YUV420P. Cross-machine bit-identity is
//! explicitly not claimed: the encoder's identity is recorded in the export
//! sidecar instead.

use serde::{Deserialize, Serialize};

/// BENDR's own stage gate: below this the stage is a true bypass — no
/// encoder alive, no readback armed, byte-identical to the prior path. The
/// deadband matters because the round trip is lossy even at zero fault
/// pressure; "off" must mean *not run*, never "run and hope it is identity".
pub const MOSH_AMOUNT_DEADBAND: f32 = 0.003;

/// The encode resolution cap, transcribed from BENDR and made orientation
/// independent: the longest edge is at most 640, aspect is preserved until
/// the small-edge floor binds, and both edges are even and at least 64. A
/// width-only cap lets a hostile portrait make the software codec arbitrarily
/// tall, defeating the very latency bound this constant exists to provide.
pub const MOSH_MAX_WIDTH: u32 = 640;
pub const MOSH_MIN_EDGE: u32 = 64;

/// Motion-wake analysis stays on the already-capped codec image. A 16-pixel
/// cell is the native MPEG-4 macroblock scale, so even a square maximum frame
/// owns at most 40 x 40 cells.
pub const MOSH_MOTION_BLOCK_PIXELS: u32 = 16;
pub const MOSH_MOTION_MAX_CELLS: usize = 1_600;
/// Decoder side data is expected from our own bounded MPEG-4 stream, but the
/// broken wire makes every byte hostile. Validate bytes and records before an
/// unaligned read and never allocate in proportion to the side data.
pub const MOSH_MOTION_MAX_SIDE_DATA_BYTES: usize = 2 * 1024 * 1024;
pub const MOSH_MOTION_MAX_RECORDS: usize = MOSH_MOTION_MAX_CELLS * 16;
const MOSH_MOTION_MAX_BLOCK_EXTENT: u16 = 256;
const MOSH_MOTION_COORDINATE_MARGIN: i64 = 256;
/// Mean luma delta maps linearly from a quiet four-code-value floor to a
/// decisive 32-code-value change. Decoder vectors can still wake a flat or
/// iso-luma object, so this is a fallback rather than an optical-flow claim.
const MOSH_MOTION_LUMA_FLOOR: f32 = 4.0;
const MOSH_MOTION_LUMA_CEILING: f32 = 32.0;
/// One decoded displacement can be stretched to twelve times its measured
/// length. The source coordinate is still clamped to the existing wet frame,
/// so the control changes reads, never work count or allocation.
const MOSH_SMEAR_MAX_GAIN: f32 = 12.0;
const MOSH_SMEAR_MAX_CODEC_PIXELS: f32 = 64.0;

/// Chunk ring bounds. BENDR caps the ring at 90 delta chunks (≈3 s at its
/// 30 fps stamp rate); the byte cap is ours, because a bounded surface must
/// be bounded in bytes, not only in entries. Eviction is FIFO on either
/// bound — a ring evicts, it never refuses.
pub const MOSH_RING_MAX_CHUNKS: usize = 90;
pub const MOSH_RING_MAX_BYTES: usize = 8 * 1024 * 1024;
/// Shuffle only fires while the ring holds MORE than this many chunks.
pub const MOSH_RING_SHUFFLE_GATE: usize = 10;
/// Shuffle never re-injects the newest six chunks: the pick is uniform over
/// `0 .. len - 6` exclusive, so the re-injected delta is always visibly out
/// of order rather than a near-duplicate of the current frame.
pub const MOSH_RING_NEWEST_EXCLUDED: usize = 6;

/// The bitrate-starvation map: `4 Mbps × 0.02^q`, so q = 0 is a healthy
/// 4 Mbps, the 0.35 default is ≈ 1 Mbps, and q = 1 is a starved 80 kbps.
pub const MOSH_BASE_BITRATE: f64 = 4_000_000.0;
const MOSH_BITRATE_FLOOR_RATIO: f64 = 0.02;

/// Bounded drain budgets for the encode and decode legs. A moshed bitstream
/// fed back into a decoder is exactly the case that hangs, so every receive
/// loop carries a finite budget in the `MAX_PACKETS_WITHOUT_FRAME` tradition.
pub const MOSH_MAX_PACKETS_PER_FRAME: usize = 16;
pub const MOSH_MAX_DECODE_FRAMES_PER_CHUNK: usize = 8;
/// One frame's total decoder feeds (normal emit + holds + shuffle) can never
/// exceed this, whatever the fault dice say.
pub const MOSH_MAX_EMITS_PER_FRAME: usize = 16;

/// The decoder-resurrection policy, transcribed: a decode fault rebuilds the
/// decoder and forces a bootstrap key; more than six consecutive rebuild
/// cycles gives the stage up with a named note, and more than thirty good
/// frames (31) forgives the count back to one.
pub const MOSH_DECODER_FAIL_LIMIT: u32 = 6;
pub const MOSH_DECODER_GOOD_STREAK: u32 = 30;

/// Fixed domain constant separating the mosh fault stream from every other
/// deterministic hash stream in the program.
const MOSH_HASH_DOMAIN: u32 = 0x434d_5348; // "CMSH"

/// Lane constants for the independent fault dice. Each decision draws its
/// own lane so one control's dice can never perturb another's.
pub const MOSH_LANE_KEY: u32 = 1;
pub const MOSH_LANE_DROP: u32 = 2;
pub const MOSH_LANE_HOLD_GATE: u32 = 3;
pub const MOSH_LANE_HOLD_COUNT: u32 = 4;
pub const MOSH_LANE_SHUFFLE_GATE: u32 = 5;
pub const MOSH_LANE_SHUFFLE_PICK: u32 = 6;

/// The authored stage. BENDR's eight continuous controls and discrete recycle
/// law are followed by three additive collide-o-scope motion-wake controls.
/// Their zero defaults preserve the exact pre-v1.5 codec path: no exported
/// motion-vector side data, luma analysis, wake allocation, displaced read,
/// or changed pixel arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CodecMoshParams {
    /// Dry/wet crossfade of the decoded round trip over the clean audience
    /// image, blended in the stored sRGB bytes exactly as BENDR blends in
    /// its display framebuffer. Below `MOSH_AMOUNT_DEADBAND` the stage is a
    /// true bypass.
    pub amount: f32,
    /// Probability that a keyframe chunk is removed between encoder and
    /// decoder. Deliberately NOT scaled by `rate` — key removal is the
    /// effect, not an event. The first key after any reset always passes
    /// (the decoder needs one whole picture to damage), and a forced resync
    /// key still faces these dice, so `key_removal = 1` never recovers.
    pub key_removal: f32,
    /// Probability (× `rate`) that a delta chunk is re-applied 1–5 extra
    /// times under fresh timestamps: the same motion vectors applied to a
    /// picture they were never measured from.
    pub hold: f32,
    /// Probability (× `rate`) that a delta chunk never reaches the decoder.
    /// A dropped chunk also skips its own hold/shuffle dice, but it still
    /// enters the ring first.
    pub drop: f32,
    /// Probability (× `rate`) that an old delta from the chunk ring is
    /// re-injected after this one, at least six chunks stale.
    pub shuffle: f32,
    /// The event-rate multiplier for hold/drop/shuffle. Key removal is
    /// exempt by law.
    pub rate: f32,
    /// Bitrate starvation: `4 Mbps × 0.02^q` with ±25% reconfigure
    /// hysteresis. Every reconfigure forces a full re-acquire, so sweeping
    /// this control visibly snaps the picture back.
    pub bitrate_starve: f32,
    /// Periodic forced keyframe: period `max(2, round((1−r)·300) + 2)`
    /// encoder-fed frames. At zero no periodic key is ever produced — the
    /// only keys are the forced bootstrap ones.
    pub resync: f32,
    /// How strongly the damaged picture is localized to moving macroblocks.
    /// Zero keeps the historical uniform dry/wet blend; one makes motion and
    /// its retained wake the complete reveal matte.
    pub wipe: f32,
    /// Pull damaged pixels backward along the decoder's forward motion
    /// displacement. The operation is one clamped nearest read fused into the
    /// existing output blend, never a second full-frame pass.
    pub smear: f32,
    /// Per-30-Hz-tick retention of the motion wake. Zero keeps only the
    /// current observation; one holds the wake until the stream is reset.
    pub trail: f32,
    /// Discrete law: CLEAN feeds the encoder the clean audience image, so
    /// the damage never compounds; RECYCLED feeds it the stage's own
    /// previous blended output, so every pass is built on the last one's
    /// wreckage.
    pub recycle: bool,
}

impl Default for CodecMoshParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            key_removal: 0.95,
            hold: 0.25,
            drop: 0.0,
            shuffle: 0.0,
            rate: 0.5,
            bitrate_starve: 0.35,
            resync: 0.0,
            wipe: 0.0,
            smear: 0.0,
            trail: 0.0,
            recycle: false,
        }
    }
}

fn finite_unit_or(value: f32, neutral: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        neutral
    }
}

impl CodecMoshParams {
    /// Clamp every continuous control to the unit interval; a non-finite
    /// input takes the field's neutral default, never a clamped extreme.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let neutral = Self::default();
        Self {
            amount: finite_unit_or(self.amount, neutral.amount),
            key_removal: finite_unit_or(self.key_removal, neutral.key_removal),
            hold: finite_unit_or(self.hold, neutral.hold),
            drop: finite_unit_or(self.drop, neutral.drop),
            shuffle: finite_unit_or(self.shuffle, neutral.shuffle),
            rate: finite_unit_or(self.rate, neutral.rate),
            bitrate_starve: finite_unit_or(self.bitrate_starve, neutral.bitrate_starve),
            resync: finite_unit_or(self.resync, neutral.resync),
            wipe: finite_unit_or(self.wipe, neutral.wipe),
            smear: finite_unit_or(self.smear, neutral.smear),
            trail: finite_unit_or(self.trail, neutral.trail),
            recycle: self.recycle,
        }
    }

    /// The wake law: `amount` alone arms the stage. Every other control
    /// shapes an inactive mechanism and wakes nothing — BENDR's own gate.
    #[must_use]
    pub fn is_active(self) -> bool {
        self.sanitized().amount >= MOSH_AMOUNT_DEADBAND
    }

    /// The additive analysis path is useful only when either spatial control
    /// can make its wake visible. `trail` shapes that wake but deliberately
    /// cannot allocate an invisible history by itself.
    #[must_use]
    fn motion_shaping_active(self) -> bool {
        let clean = self.sanitized();
        clean.wipe >= MOSH_AMOUNT_DEADBAND || clean.smear >= MOSH_AMOUNT_DEADBAND
    }
}

/// The encode resolution for one output size: longest edge at most
/// `MOSH_MAX_WIDTH`, aspect preserved until the small-edge floor binds, both
/// edges even and at least `MOSH_MIN_EDGE`.
#[must_use]
pub fn mosh_dimensions(width: u32, height: u32) -> (u32, u32) {
    let w = width.max(1);
    let h = height.max(1);
    let longest = w.max(h);
    let scale = if longest > MOSH_MAX_WIDTH {
        f64::from(MOSH_MAX_WIDTH) / f64::from(longest)
    } else {
        1.0
    };
    let tw = (f64::from(w) * scale).round().max(1.0) as u32;
    let th = (f64::from(h) * scale).round().max(1.0) as u32;
    ((tw & !1).max(MOSH_MIN_EDGE), (th & !1).max(MOSH_MIN_EDGE))
}

/// The bitrate-starvation map.
#[must_use]
pub fn mosh_target_bitrate(bitrate_starve: f32) -> u32 {
    let q = f64::from(finite_unit_or(bitrate_starve, 0.35));
    (MOSH_BASE_BITRATE * MOSH_BITRATE_FLOOR_RATIO.powf(q)).round() as u32
}

/// The ±25% reconfigure hysteresis: an encoder is only reopened when the
/// target moves more than a quarter of itself away from the current rate.
/// An engine with no configured rate yet always reconfigures.
#[must_use]
pub fn mosh_bitrate_reconfigure_needed(current: Option<u32>, want: u32) -> bool {
    match current {
        None => true,
        Some(current) => {
            let delta = i64::from(want) - i64::from(current);
            delta.unsigned_abs() > u64::from(want) / 4
        }
    }
}

fn mosh_motion_decoder_rebuild_needed(motion_vectors_enabled: bool, motion_shaping: bool) -> bool {
    // Enabling export requires a decoder open. Disabling does not: rebuilding
    // immediately on every knob crossing would thrash the stream/bootstrap.
    // The unused flag becomes free at the next natural bitrate/size reset,
    // while analysis and side-data parsing stop immediately.
    motion_shaping && !motion_vectors_enabled
}

fn mosh_fault_count_after_decode_error(fails: u32, good: u32) -> u32 {
    if good > MOSH_DECODER_GOOD_STREAK {
        1
    } else {
        fails.saturating_add(1)
    }
}

fn discard_recycled_input_on_stream_reset(recycled: &mut Option<Vec<u8>>) {
    *recycled = None;
}

fn mosh_decoder_receive_would_block(error: &ffmpeg::Error) -> bool {
    matches!(
        error,
        ffmpeg::Error::Other {
            errno: ffmpeg::error::EAGAIN,
        }
    )
}

/// The resync period in encoder-fed frames, or `None` for "never recovers".
#[must_use]
pub fn mosh_resync_period(resync: f32) -> Option<u32> {
    let r = finite_unit_or(resync, 0.0);
    if r <= MOSH_AMOUNT_DEADBAND {
        return None;
    }
    Some((((1.0 - f64::from(r)) * 300.0).round() as u32 + 2).max(2))
}

/// Whether this encoder-fed frame is forced to be a keyframe: the bootstrap
/// after any reset, an explicit re-acquire, or the periodic resync clock.
#[must_use]
pub fn mosh_wants_keyframe(seen: bool, need_key: bool, resync: f32, frame_count: u64) -> bool {
    if !seen || need_key {
        return true;
    }
    mosh_resync_period(resync).is_some_and(|period| frame_count.is_multiple_of(u64::from(period)))
}

/// The shared integer avalanche, byte-identical to `effects.wgsl`'s
/// `cellular_avalanche` and `motion.rs`'s `motion_avalanche`, in the mosh
/// stage's own fixed domain.
fn mosh_avalanche(value: u32) -> u32 {
    let mut x = value;
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// One deterministic unit sample for a fault decision: keyed by the stage's
/// reference ordinal, the packet index within the frame, the master
/// `random_seed`, and the decision lane. No wall time, ever.
#[must_use]
pub fn mosh_hash(ordinal: u64, packet_index: u32, seed: u32, lane: u32) -> f32 {
    let low = ordinal as u32;
    let high = (ordinal >> 32) as u32;
    let mixed = mosh_avalanche(
        low ^ high.wrapping_mul(0x9e37_79b9)
            ^ packet_index.wrapping_mul(0x85eb_ca6b)
            ^ seed.wrapping_mul(0x27d4_eb2f)
            ^ lane.wrapping_mul(0x1656_67b1)
            ^ MOSH_HASH_DOMAIN,
    );
    (mixed & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// The fate of one keyframe chunk between encoder and decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoshKeyDecision {
    Emit,
    Remove,
}

/// Decide one keyframe chunk. `bootstrap_pending` is true when no key has
/// reached the decoder since the last reset; the bootstrap always passes.
#[must_use]
pub fn decide_key_chunk(
    params: CodecMoshParams,
    ordinal: u64,
    packet_index: u32,
    seed: u32,
    bootstrap_pending: bool,
) -> MoshKeyDecision {
    if bootstrap_pending {
        return MoshKeyDecision::Emit;
    }
    let params = params.sanitized();
    if mosh_hash(ordinal, packet_index, seed, MOSH_LANE_KEY) < params.key_removal {
        MoshKeyDecision::Remove
    } else {
        MoshKeyDecision::Emit
    }
}

/// The fate of one delta chunk: dropped outright, or emitted with a number
/// of extra hold re-applications and an optional shuffle pick (a unit value
/// the ring maps onto an old chunk).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoshDeltaDecision {
    pub dropped: bool,
    pub extra_repeats: u32,
    pub shuffle_pick: Option<f32>,
}

/// Decide one delta chunk. Drop wins first and suppresses this chunk's own
/// hold/shuffle dice, exactly as BENDR's early return does; the caller must
/// still push the chunk into the ring before consulting this decision.
#[must_use]
pub fn decide_delta_chunk(
    params: CodecMoshParams,
    ordinal: u64,
    packet_index: u32,
    seed: u32,
) -> MoshDeltaDecision {
    let params = params.sanitized();
    let rate = params.rate;
    if mosh_hash(ordinal, packet_index, seed, MOSH_LANE_DROP) < params.drop * rate {
        return MoshDeltaDecision {
            dropped: true,
            extra_repeats: 0,
            shuffle_pick: None,
        };
    }
    let extra_repeats =
        if mosh_hash(ordinal, packet_index, seed, MOSH_LANE_HOLD_GATE) < params.hold * rate {
            1 + (mosh_hash(ordinal, packet_index, seed, MOSH_LANE_HOLD_COUNT) * params.hold * 5.0)
                .floor() as u32
        } else {
            0
        };
    let shuffle_pick = (mosh_hash(ordinal, packet_index, seed, MOSH_LANE_SHUFFLE_GATE)
        < params.shuffle * rate)
        .then(|| mosh_hash(ordinal, packet_index, seed, MOSH_LANE_SHUFFLE_PICK));
    MoshDeltaDecision {
        dropped: false,
        extra_repeats,
        shuffle_pick,
    }
}

/// The bounded delta-chunk ring. Keys never enter it; eviction is FIFO on
/// both the entry cap and the byte cap.
#[derive(Debug, Default)]
pub struct MoshChunkRing {
    entries: std::collections::VecDeque<Vec<u8>>,
    bytes: usize,
}

impl MoshChunkRing {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: Vec<u8>) {
        self.bytes = self.bytes.saturating_add(chunk.len());
        self.entries.push_back(chunk);
        while self.entries.len() > MOSH_RING_MAX_CHUNKS || self.bytes > MOSH_RING_MAX_BYTES {
            match self.entries.pop_front() {
                Some(evicted) => self.bytes = self.bytes.saturating_sub(evicted.len()),
                None => break,
            }
        }
    }

    /// Map a unit pick onto an old chunk: only while the ring holds more
    /// than `MOSH_RING_SHUFFLE_GATE` entries, and never one of the newest
    /// `MOSH_RING_NEWEST_EXCLUDED`.
    #[must_use]
    pub fn pick(&self, unit: f32) -> Option<&[u8]> {
        if self.entries.len() <= MOSH_RING_SHUFFLE_GATE {
            return None;
        }
        let selectable = self.entries.len() - MOSH_RING_NEWEST_EXCLUDED;
        let unit = finite_unit_or(unit, 0.0);
        let index = ((unit * selectable as f32).floor() as usize).min(selectable - 1);
        self.entries.get(index).map(Vec::as_slice)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the bound fixtures assert the ring's exact caps")
    )]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the bound fixtures assert the ring's exact caps")
    )]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the bound fixtures assert the ring's exact caps")
    )]
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Sampled inputs that travel with one frame's pixels — the NTSC metadata
/// law: a delayed job must be processed with the parameters sampled at its
/// own readback, never a newer frame's.
#[derive(Debug, Clone)]
pub struct MoshFrameMetadata {
    pub params: CodecMoshParams,
    /// Explicit capability bit for the internal packed send matte. Only this
    /// mode interprets input alpha as per-pixel mosh influence; ordinary
    /// audience RGBA remains ungated even when its alpha is not opaque.
    pub use_influence_alpha: bool,
    /// The stage's 30 Hz reference ordinal: the same program-clock frame the
    /// NTSC phase uses live, and the frame-index derivation offline. It
    /// drives the fault dice, so Pause holds the fault stream still.
    pub ordinal: u64,
    /// The master `random_seed` sampled with the frame.
    pub seed: u32,
    /// One continuous armed interval. Linked Temporal dry deliberately keeps
    /// the broader visual epoch stable, so this narrower generation prevents
    /// a delayed pre-dry job from resurfacing after re-entry.
    pub generation: u64,
    /// When the live final-program VHS path is active while the mosh is armed,
    /// the worker runs the VHS kernel after Codec Mosh in the same hop — one
    /// admission, one latest-only asynchronous hop, and the exact offline
    /// ordering (mosh, then VHS, on the same pixels). Completion is
    /// host-dependent; queue depth remains bounded. `None` when VHS is disabled
    /// or when an isolated Temporal-bypass route owns the dry overlay.
    pub ntsc: Option<crate::ntsc::NtscFrameMetadata>,
}

/// One processed frame returned by the worker. The caller must discard it
/// when its generation is no longer current.
pub struct MoshProcessedFrame {
    pub pixels: Vec<u8>,
    pub epoch: u64,
    pub generation: u64,
    /// True when this combined worker hop also completed final-program VHS.
    pub ntsc_processed: bool,
}

struct MoshJob {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    metadata: MoshFrameMetadata,
    epoch: u64,
}

// ---------------------------------------------------------------------------
// The engine: two ffmpeg-next codec contexts and the broken wire between them.
// ---------------------------------------------------------------------------

use ffmpeg_next as ffmpeg;

#[derive(Debug, Clone, Copy)]
struct ValidatedMoshMotionVector {
    destination: [i32; 2],
    block: [u16; 2],
    /// Forward displacement from the past reference into this destination,
    /// in codec-image pixels.
    forward: [f32; 2],
    past_reference: bool,
}

fn validate_mosh_motion_vector(
    raw: ffmpeg::ffi::AVMotionVector,
    dimensions: [u32; 2],
) -> Result<ValidatedMoshMotionVector, ()> {
    let block = [u16::from(raw.w), u16::from(raw.h)];
    if block[0] == 0
        || block[1] == 0
        || block[0] > MOSH_MOTION_MAX_BLOCK_EXTENT
        || block[1] > MOSH_MOTION_MAX_BLOCK_EXTENT
        || raw.motion_scale == 0
        || raw.source == 0
    {
        return Err(());
    }
    let coordinate_limit =
        i64::from(dimensions[0].max(dimensions[1])).saturating_add(MOSH_MOTION_COORDINATE_MARGIN);
    if [raw.src_x, raw.src_y, raw.dst_x, raw.dst_y]
        .into_iter()
        .any(|coordinate| i64::from(coordinate).abs() > coordinate_limit)
    {
        return Err(());
    }
    let reference_minus_destination = [
        raw.motion_x as f64 / f64::from(raw.motion_scale),
        raw.motion_y as f64 / f64::from(raw.motion_scale),
    ];
    let displacement_limits = [
        f64::from(dimensions[0].saturating_add(u32::from(MOSH_MOTION_MAX_BLOCK_EXTENT))),
        f64::from(dimensions[1].saturating_add(u32::from(MOSH_MOTION_MAX_BLOCK_EXTENT))),
    ];
    if reference_minus_destination
        .iter()
        .zip(displacement_limits)
        .any(|(value, limit)| !value.is_finite() || value.abs() > limit)
    {
        return Err(());
    }
    let expected_source = [
        f64::from(raw.dst_x) + reference_minus_destination[0],
        f64::from(raw.dst_y) + reference_minus_destination[1],
    ];
    if [raw.src_x, raw.src_y]
        .into_iter()
        .zip(expected_source)
        .any(|(source, expected)| (f64::from(source) - expected).abs() > 1.0)
    {
        return Err(());
    }
    Ok(ValidatedMoshMotionVector {
        destination: [i32::from(raw.dst_x), i32::from(raw.dst_y)],
        block,
        // FFmpeg records reference minus destination. The wake needs the
        // forward, previous-to-current direction, hence the negation.
        forward: [
            -(reference_minus_destination[0] as f32),
            -(reference_minus_destination[1] as f32),
        ],
        past_reference: raw.source < 0,
    })
}

fn read_mosh_motion_vector(bytes: &[u8], index: usize) -> ffmpeg::ffi::AVMotionVector {
    let record_bytes = std::mem::size_of::<ffmpeg::ffi::AVMotionVector>();
    let offset = index * record_bytes;
    // SAFETY: the caller proved exact record-size divisibility and bounded
    // `index` by the derived count. AVFrame side data makes no alignment
    // promise, so an unaligned read is required.
    unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(offset).cast()) }
}

/// Rasterize decoder motion side data directly into the fixed macroblock
/// arrays. Validation is a complete first pass, so a malformed later record
/// cannot leave a partially published field. No record-sized allocation is
/// performed.
fn rasterize_mosh_motion_side_data(
    bytes: &[u8],
    dimensions: [u32; 2],
    grid: [u32; 2],
    wake: &mut [f32],
    vectors: &mut [[f32; 2]],
    vector_area: &mut [u32],
) -> bool {
    let record_bytes = std::mem::size_of::<ffmpeg::ffi::AVMotionVector>();
    if dimensions.contains(&0)
        || grid.contains(&0)
        || record_bytes == 0
        || bytes.len() > MOSH_MOTION_MAX_SIDE_DATA_BYTES
        || !bytes.len().is_multiple_of(record_bytes)
    {
        return false;
    }
    let count = bytes.len() / record_bytes;
    let expected = u64::from(grid[0]).saturating_mul(u64::from(grid[1]));
    // MPEG-4 can subdivide a macroblock, but a sixteen-record allowance per
    // grid cell is already as fine as 4x4 luma blocks. Refuse larger corrupt
    // side-data scans even when the global byte ceiling would admit them.
    let frame_record_limit = usize::try_from(expected)
        .unwrap_or(usize::MAX)
        .saturating_mul(16)
        .min(MOSH_MOTION_MAX_RECORDS);
    if count == 0
        || count > frame_record_limit
        || usize::try_from(expected).ok() != Some(wake.len())
        || vectors.len() != wake.len()
        || vector_area.len() != wake.len()
    {
        return false;
    }
    let mut past_records = 0_usize;
    for index in 0..count {
        let Ok(vector) =
            validate_mosh_motion_vector(read_mosh_motion_vector(bytes, index), dimensions)
        else {
            return false;
        };
        if vector.past_reference {
            past_records += 1;
        }
    }
    if past_records == 0 {
        return false;
    }

    for index in 0..count {
        let vector = validate_mosh_motion_vector(read_mosh_motion_vector(bytes, index), dimensions)
            .expect("the transactional validation pass admitted this record");
        if !vector.past_reference {
            continue;
        }
        let half_width = i64::from(vector.block[0]) / 2;
        let half_height = i64::from(vector.block[1]) / 2;
        let left = i64::from(vector.destination[0]) - half_width;
        let top = i64::from(vector.destination[1]) - half_height;
        let right = left + i64::from(vector.block[0]);
        let bottom = top + i64::from(vector.block[1]);
        let clipped_left = left.clamp(0, i64::from(dimensions[0]));
        let clipped_top = top.clamp(0, i64::from(dimensions[1]));
        let clipped_right = right.clamp(0, i64::from(dimensions[0]));
        let clipped_bottom = bottom.clamp(0, i64::from(dimensions[1]));
        if clipped_left >= clipped_right || clipped_top >= clipped_bottom {
            continue;
        }
        let cell = i64::from(MOSH_MOTION_BLOCK_PIXELS);
        let grid_left = u32::try_from(clipped_left / cell).unwrap_or(0);
        let grid_top = u32::try_from(clipped_top / cell).unwrap_or(0);
        let grid_right = u32::try_from((clipped_right + cell - 1) / cell)
            .unwrap_or(grid[0])
            .min(grid[0]);
        let grid_bottom = u32::try_from((clipped_bottom + cell - 1) / cell)
            .unwrap_or(grid[1])
            .min(grid[1]);
        let area = u32::from(vector.block[0]) * u32::from(vector.block[1]);
        let magnitude = vector.forward[0].hypot(vector.forward[1]);
        // Four codec pixels of displacement are already a decisive moving
        // macroblock. Smaller vectors remain proportionally useful and the
        // luma fallback independently covers residual-only movement.
        let vector_wake = (magnitude / 4.0).clamp(0.0, 1.0);
        for y in grid_top..grid_bottom {
            for x in grid_left..grid_right {
                let slot = usize::try_from(u64::from(y) * u64::from(grid[0]) + u64::from(x))
                    .expect("the 40x40 motion grid fits usize");
                wake[slot] = wake[slot].max(vector_wake);
                let incumbent_magnitude = vectors[slot][0].hypot(vectors[slot][1]);
                // Direction follows the strongest meaningful decoder motion.
                // Block area is only a deterministic tie-breaker; a tiny,
                // nearly static partition must never overwrite a larger
                // vector that actually created the cell's wake.
                let replace_direction = match magnitude.total_cmp(&incumbent_magnitude) {
                    std::cmp::Ordering::Greater => magnitude > 0.0,
                    std::cmp::Ordering::Equal => magnitude > 0.0 && area < vector_area[slot],
                    std::cmp::Ordering::Less => false,
                };
                if replace_direction {
                    vector_area[slot] = area;
                    vectors[slot] = vector.forward;
                }
            }
        }
    }
    true
}

fn motion_luma_wake(mean_absolute_delta: f32) -> f32 {
    ((mean_absolute_delta - MOSH_MOTION_LUMA_FLOOR)
        / (MOSH_MOTION_LUMA_CEILING - MOSH_MOTION_LUMA_FLOOR))
        .clamp(0.0, 1.0)
}

fn motion_trail_decay(trail: f32, mut ticks: u64) -> f32 {
    // Exponentiation by squaring applies every elapsed reference tick in at
    // most 64 multiplies. Capping a pause at 600 ticks makes old trails
    // survive arbitrarily long transport gaps.
    let mut factor = trail.clamp(0.0, 1.0);
    let mut decay = 1.0_f32;
    while ticks > 0 {
        if ticks & 1 != 0 {
            decay *= factor;
        }
        factor *= factor;
        ticks >>= 1;
    }
    decay
}

fn allocate_motion_vec<T: Clone>(length: usize, value: T) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| format!("mosh motion wake could not reserve {length} elements"))?;
    values.resize(length, value);
    Ok(values)
}

/// One low-resolution motion memory. It owns no full-resolution pixels: one
/// packed luma observation plus fixed macroblock arrays are enough to reveal,
/// displace, and retain the real codec result.
struct MoshMotionState {
    width: u32,
    height: u32,
    grid_width: u32,
    grid_height: u32,
    previous_luma: Vec<u8>,
    previous_luma_valid: bool,
    fallback_wake: Vec<f32>,
    decoder_wake: Vec<f32>,
    decoder_vectors: Vec<[f32; 2]>,
    decoder_vector_area: Vec<u32>,
    sampled_influence: Vec<f32>,
    dilated_influence: Vec<f32>,
    retained_influence: Vec<f32>,
    output_trail_influence: Vec<f32>,
    dilated_wake: Vec<f32>,
    dilated_vectors: Vec<[f32; 2]>,
    wake: Vec<f32>,
    trail_vectors: Vec<[f32; 2]>,
    last_ordinal: Option<u64>,
    #[cfg(test)]
    decoder_ingest_count: usize,
}

impl MoshMotionState {
    fn new(width: u32, height: u32) -> Result<Self, String> {
        if width == 0 || height == 0 || width > MOSH_MAX_WIDTH || height > MOSH_MAX_WIDTH {
            return Err(format!(
                "mosh motion dimensions {width}x{height} exceed the capped codec domain"
            ));
        }
        let grid_width = width.div_ceil(MOSH_MOTION_BLOCK_PIXELS);
        let grid_height = height.div_ceil(MOSH_MOTION_BLOCK_PIXELS);
        let count = usize::try_from(u64::from(grid_width) * u64::from(grid_height))
            .map_err(|_| "mosh motion grid overflowed".to_string())?;
        if count == 0 || count > MOSH_MOTION_MAX_CELLS {
            return Err(format!(
                "mosh motion grid {grid_width}x{grid_height} exceeds {MOSH_MOTION_MAX_CELLS} cells"
            ));
        }
        let luma_count = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| "mosh motion luma size overflowed".to_string())?;
        Ok(Self {
            width,
            height,
            grid_width,
            grid_height,
            previous_luma: allocate_motion_vec(luma_count, 0_u8)?,
            previous_luma_valid: false,
            fallback_wake: allocate_motion_vec(count, 0.0_f32)?,
            decoder_wake: allocate_motion_vec(count, 0.0_f32)?,
            decoder_vectors: allocate_motion_vec(count, [0.0_f32; 2])?,
            decoder_vector_area: allocate_motion_vec(count, u32::MAX)?,
            sampled_influence: allocate_motion_vec(count, 0.0_f32)?,
            dilated_influence: allocate_motion_vec(count, 0.0_f32)?,
            retained_influence: allocate_motion_vec(count, 0.0_f32)?,
            output_trail_influence: allocate_motion_vec(count, 0.0_f32)?,
            dilated_wake: allocate_motion_vec(count, 0.0_f32)?,
            dilated_vectors: allocate_motion_vec(count, [0.0_f32; 2])?,
            wake: allocate_motion_vec(count, 0.0_f32)?,
            trail_vectors: allocate_motion_vec(count, [0.0_f32; 2])?,
            last_ordinal: None,
            #[cfg(test)]
            decoder_ingest_count: 0,
        })
    }

    fn clear_history(&mut self) {
        self.previous_luma_valid = false;
        self.fallback_wake.fill(0.0);
        self.decoder_wake.fill(0.0);
        self.decoder_vectors.fill([0.0; 2]);
        self.decoder_vector_area.fill(u32::MAX);
        self.clear_influence_history();
        self.dilated_wake.fill(0.0);
        self.dilated_vectors.fill([0.0; 2]);
        self.wake.fill(0.0);
        self.trail_vectors.fill([0.0; 2]);
        self.last_ordinal = None;
    }

    fn clear_influence_history(&mut self) {
        self.sampled_influence.fill(0.0);
        self.dilated_influence.fill(0.0);
        self.retained_influence.fill(0.0);
        self.output_trail_influence.fill(0.0);
    }

    fn begin_frame(
        &mut self,
        luma: &[u8],
        stride: usize,
        ordinal: u64,
        trail: f32,
    ) -> Result<(), String> {
        let width = self.width as usize;
        let height = self.height as usize;
        let required = height
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(stride))
            .and_then(|offset| offset.checked_add(width))
            .ok_or_else(|| "mosh motion luma extent overflowed".to_string())?;
        if stride < width || luma.len() < required {
            self.clear_history();
            return Err(format!(
                "mosh motion luma plane {} bytes/stride {stride} is short for {}x{}",
                luma.len(),
                self.width,
                self.height
            ));
        }

        self.fallback_wake.fill(0.0);
        self.decoder_wake.fill(0.0);
        self.decoder_vectors.fill([0.0; 2]);
        self.decoder_vector_area.fill(u32::MAX);
        self.sampled_influence.fill(0.0);
        self.dilated_influence.fill(0.0);
        match self.last_ordinal {
            Some(previous) if ordinal < previous => {
                self.wake.fill(0.0);
                self.trail_vectors.fill([0.0; 2]);
                self.retained_influence.fill(0.0);
                // A reverse seek starts a new observation sequence. Comparing
                // its luma against a later frame creates a full-scene false
                // wake even though retained motion was correctly cleared.
                self.previous_luma_valid = false;
            }
            Some(previous) => {
                let ticks = ordinal.saturating_sub(previous);
                if trail <= 0.0 {
                    self.wake.fill(0.0);
                    self.trail_vectors.fill([0.0; 2]);
                    self.retained_influence.fill(0.0);
                } else if ticks > 0 && trail < 1.0 {
                    let decay = motion_trail_decay(trail, ticks);
                    for (wake, influence) in self.wake.iter_mut().zip(&mut self.retained_influence)
                    {
                        *wake *= decay;
                        *influence *= decay;
                    }
                }
            }
            None => {}
        }
        self.last_ordinal = Some(ordinal);
        for index in 0..self.wake.len() {
            self.retained_influence[index] = self.retained_influence[index].min(self.wake[index]);
            self.output_trail_influence[index] = self.retained_influence[index];
        }

        if self.previous_luma_valid {
            for cell_y in 0..self.grid_height {
                let top = cell_y * MOSH_MOTION_BLOCK_PIXELS;
                let bottom = (top + MOSH_MOTION_BLOCK_PIXELS).min(self.height);
                for cell_x in 0..self.grid_width {
                    let left = cell_x * MOSH_MOTION_BLOCK_PIXELS;
                    let right = (left + MOSH_MOTION_BLOCK_PIXELS).min(self.width);
                    let mut delta = 0_u64;
                    let mut samples = 0_u64;
                    for y in top..bottom {
                        let current_row = y as usize * stride;
                        let previous_row = y as usize * width;
                        for x in left..right {
                            delta += u64::from(
                                luma[current_row + x as usize]
                                    .abs_diff(self.previous_luma[previous_row + x as usize]),
                            );
                            samples += 1;
                        }
                    }
                    let mean = if samples == 0 {
                        0.0
                    } else {
                        delta as f32 / samples as f32
                    };
                    let slot = usize::try_from(
                        u64::from(cell_y) * u64::from(self.grid_width) + u64::from(cell_x),
                    )
                    .expect("the bounded motion grid fits usize");
                    self.fallback_wake[slot] = motion_luma_wake(mean);
                }
            }
        }
        for y in 0..height {
            self.previous_luma[y * width..(y + 1) * width]
                .copy_from_slice(&luma[y * stride..y * stride + width]);
        }
        self.previous_luma_valid = true;
        Ok(())
    }

    fn sample_influence_alpha(
        &mut self,
        pixels: &[u8],
        output_width: u32,
        output_height: u32,
    ) -> Result<(), String> {
        let row_bytes = usize::try_from(output_width)
            .ok()
            .and_then(|width| width.checked_mul(4))
            .ok_or_else(|| "mosh influence row overflowed".to_string())?;
        let required = row_bytes
            .checked_mul(output_height as usize)
            .ok_or_else(|| "mosh influence frame overflowed".to_string())?;
        if output_width == 0 || output_height == 0 || pixels.len() < required {
            return Err("mosh influence frame is short".to_string());
        }
        // Four stratified samples per axis make this aggregation a hard
        // <=25,600 alpha-read budget (1,600 cells * 16), independent of 4K/8K
        // output size and without another full-resolution pass/allocation.
        const SAMPLES_PER_AXIS: usize = 4;
        for cell_y in 0..self.grid_height {
            let rows = motion_cell_output_span(cell_y, self.height, output_height);
            let row_samples = rows.len().min(SAMPLES_PER_AXIS);
            for cell_x in 0..self.grid_width {
                let columns = motion_cell_output_span(cell_x, self.width, output_width);
                let column_samples = columns.len().min(SAMPLES_PER_AXIS);
                let mut strongest = 0_u8;
                for sample_y in 0..row_samples {
                    let y = rows.start
                        + ((sample_y * 2 + 1) * rows.len() / (row_samples * 2)).min(rows.len() - 1);
                    for sample_x in 0..column_samples {
                        let x = columns.start
                            + ((sample_x * 2 + 1) * columns.len() / (column_samples * 2))
                                .min(columns.len() - 1);
                        strongest = strongest.max(pixels[y * row_bytes + x * 4 + 3]);
                    }
                }
                let slot = usize::try_from(
                    u64::from(cell_y) * u64::from(self.grid_width) + u64::from(cell_x),
                )
                .expect("the bounded motion grid fits usize");
                self.sampled_influence[slot] = f32::from(strongest) / 255.0;
            }
        }
        Ok(())
    }

    fn ingest_decoder_frame(&mut self, frame: &ffmpeg::util::frame::video::Video) {
        // Only the newest decoded picture supplies direction. Multiple held
        // or shuffled emits can complete in one apply call; retaining a union
        // would pair the final pixels with vectors from pictures no longer
        // presented.
        self.decoder_wake.fill(0.0);
        self.decoder_vectors.fill([0.0; 2]);
        self.decoder_vector_area.fill(u32::MAX);
        #[cfg(test)]
        {
            self.decoder_ingest_count += 1;
        }
        let Some(side_data) = frame.side_data(ffmpeg::util::frame::side_data::Type::MotionVectors)
        else {
            return;
        };
        let _ = rasterize_mosh_motion_side_data(
            side_data.data(),
            [self.width, self.height],
            [self.grid_width, self.grid_height],
            &mut self.decoder_wake,
            &mut self.decoder_vectors,
            &mut self.decoder_vector_area,
        );
    }

    fn finish_frame(&mut self) {
        // One fixed macroblock dilation fills textureless object interiors and
        // bridges a fast edge without making work depend on authored values.
        for y in 0..self.grid_height {
            for x in 0..self.grid_width {
                let mut best_wake = 0.0_f32;
                let mut best_vector = [0.0_f32; 2];
                let mut best_vector_magnitude = 0.0_f32;
                let mut best_influence = 0.0_f32;
                let min_y = y.saturating_sub(1);
                let max_y = (y + 1).min(self.grid_height - 1);
                let min_x = x.saturating_sub(1);
                let max_x = (x + 1).min(self.grid_width - 1);
                for source_y in min_y..=max_y {
                    for source_x in min_x..=max_x {
                        let source = usize::try_from(
                            u64::from(source_y) * u64::from(self.grid_width) + u64::from(source_x),
                        )
                        .expect("the bounded motion grid fits usize");
                        let candidate = self.fallback_wake[source].max(self.decoder_wake[source]);
                        if candidate > best_wake {
                            best_wake = candidate;
                        }
                        best_influence = best_influence
                            .max(self.sampled_influence[source] * candidate.clamp(0.0, 1.0));
                        // Wake and direction are deliberately dilated on
                        // independent scores. A stronger luma-only fallback
                        // reveals motion but cannot erase a neighboring valid
                        // decoder direction needed by the smear.
                        let vector = self.decoder_vectors[source];
                        let vector_magnitude = vector[0].hypot(vector[1]);
                        if vector_magnitude > best_vector_magnitude {
                            best_vector_magnitude = vector_magnitude;
                            best_vector = vector;
                        }
                    }
                }
                let slot =
                    usize::try_from(u64::from(y) * u64::from(self.grid_width) + u64::from(x))
                        .expect("the bounded motion grid fits usize");
                self.dilated_wake[slot] = best_wake;
                self.dilated_vectors[slot] = best_vector;
                self.dilated_influence[slot] = best_influence;
            }
        }
        for index in 0..self.wake.len() {
            let instant = self.dilated_wake[index];
            self.wake[index] = self.wake[index].max(instant).clamp(0.0, 1.0);
            let vector = self.dilated_vectors[index];
            if vector[0] != 0.0 || vector[1] != 0.0 {
                self.trail_vectors[index] = vector;
            }
            self.retained_influence[index] = self.retained_influence[index]
                .max(self.dilated_influence[index])
                .min(self.wake[index]);
        }
    }

    fn sample_cell(&self, cell_x: u32, cell_y: u32) -> (f32, [f32; 2], f32) {
        let cell_x = cell_x.min(self.grid_width - 1);
        let cell_y = cell_y.min(self.grid_height - 1);
        let slot =
            usize::try_from(u64::from(cell_y) * u64::from(self.grid_width) + u64::from(cell_x))
                .expect("the bounded motion grid fits usize");
        (
            self.wake[slot],
            self.trail_vectors[slot],
            self.output_trail_influence[slot],
        )
    }
}

fn motion_local_amount(amount: f32, wipe: f32, wake: f32) -> f32 {
    amount * ((1.0 - wipe) + wipe * wake.clamp(0.0, 1.0))
}

fn motion_cell_output_span(
    cell: u32,
    codec_extent: u32,
    output_extent: u32,
) -> std::ops::Range<usize> {
    let left = cell
        .saturating_mul(MOSH_MOTION_BLOCK_PIXELS)
        .min(codec_extent);
    let right = cell
        .saturating_add(1)
        .saturating_mul(MOSH_MOTION_BLOCK_PIXELS)
        .min(codec_extent);
    let scale_ceil = |coordinate: u32| {
        let numerator = u64::from(coordinate) * u64::from(output_extent);
        usize::try_from(
            numerator.saturating_add(u64::from(codec_extent).saturating_sub(1))
                / u64::from(codec_extent.max(1)),
        )
        .unwrap_or(output_extent as usize)
        .min(output_extent as usize)
    };
    scale_ceil(left)..scale_ceil(right)
}

fn motion_smear_offset(
    forward_codec_pixels: [f32; 2],
    smear: f32,
    output_dimensions: [u32; 2],
    codec_dimensions: [u32; 2],
) -> [i64; 2] {
    let gain = 1.0 + smear.clamp(0.0, 1.0) * (MOSH_SMEAR_MAX_GAIN - 1.0);
    let forward = [
        forward_codec_pixels[0].clamp(-MOSH_SMEAR_MAX_CODEC_PIXELS, MOSH_SMEAR_MAX_CODEC_PIXELS),
        forward_codec_pixels[1].clamp(-MOSH_SMEAR_MAX_CODEC_PIXELS, MOSH_SMEAR_MAX_CODEC_PIXELS),
    ];
    [
        (forward[0] * output_dimensions[0] as f32 / codec_dimensions[0].max(1) as f32 * gain)
            .round() as i64,
        (forward[1] * output_dimensions[1] as f32 / codec_dimensions[1].max(1) as f32 * gain)
            .round() as i64,
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoshEmitOutcome {
    Accepted,
    AcceptedPicture,
    Reset,
}

impl MoshEmitOutcome {
    fn accepted(self) -> bool {
        !matches!(self, Self::Reset)
    }

    fn decoded_picture(self) -> bool {
        matches!(self, Self::AcceptedPicture)
    }

    fn reset(self) -> bool {
        matches!(self, Self::Reset)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MoshBootstrapState {
    seen: bool,
    need_key: bool,
}

fn mosh_bootstrap_after_emit(
    state: MoshBootstrapState,
    key_packet: bool,
    outcome: MoshEmitOutcome,
) -> MoshBootstrapState {
    if outcome.reset() {
        return MoshBootstrapState {
            seen: false,
            need_key: true,
        };
    }
    if key_packet && outcome.accepted() {
        return MoshBootstrapState {
            seen: true,
            need_key: false,
        };
    }
    state
}

fn mosh_decoded_frame_in_fixed_domain(
    format: ffmpeg::format::Pixel,
    dimensions: [u32; 2],
    expected: [u32; 2],
) -> bool {
    format == ffmpeg::format::Pixel::YUV420P
        && dimensions == expected
        && !dimensions.contains(&0)
        && dimensions[0] <= MOSH_MAX_WIDTH
        && dimensions[1] <= MOSH_MAX_WIDTH
}

#[cfg(test)]
fn motion_smear_source(
    coordinate: [u32; 2],
    forward_codec_pixels: [f32; 2],
    smear: f32,
    output_dimensions: [u32; 2],
    codec_dimensions: [u32; 2],
) -> [u32; 2] {
    let offset = motion_smear_offset(
        forward_codec_pixels,
        smear,
        output_dimensions,
        codec_dimensions,
    );
    [
        (i64::from(coordinate[0]) - offset[0])
            .clamp(0, i64::from(output_dimensions[0].saturating_sub(1))) as u32,
        (i64::from(coordinate[1]) - offset[1])
            .clamp(0, i64::from(output_dimensions[1].saturating_sub(1))) as u32,
    ]
}

/// The encoder identity recorded in the export sidecar: the honesty shape of
/// the hw-decode receipt. Per-host repeatability is claimed; cross-machine
/// bit-identity is not, and this is the record of why.
#[must_use]
pub fn mosh_encoder_identity() -> String {
    format!("mpeg4/avcodec-{}", ffmpeg::codec::version())
}

/// The synchronous round-trip engine. Live it is owned by the worker thread;
/// offline it is owned by the export loop and called in place — one kernel,
/// two owners, exactly the `NtscState` relation.
pub struct MoshEngine {
    width: u32,
    height: u32,
    mosh_width: u32,
    mosh_height: u32,
    encoder: ffmpeg::encoder::video::Encoder,
    decoder: ffmpeg::decoder::Video,
    scale_in: ffmpeg::software::scaling::Context,
    scale_out: Option<ffmpeg::software::scaling::Context>,
    current_bitrate: Option<u32>,
    /// True once any key has reached the decoder since the last reset.
    seen: bool,
    /// Force the next encoder-fed frame to be a key and let it pass.
    need_key: bool,
    /// Encoder-fed frame counter: the resync clock and the PTS source.
    frame_count: u64,
    /// Monotonic re-stamp counter for every decoder feed.
    emit_count: i64,
    ring: MoshChunkRing,
    /// The last decoded picture, held across decoder starvation so a dropped
    /// chunk smears rather than flashing dry — the stage's own prior-image
    /// hold.
    last_decoded: Option<ffmpeg::util::frame::video::Video>,
    /// The previous blended full-resolution output, kept only while the
    /// recycle law is authored.
    recycled: Option<Vec<u8>>,
    /// Decoder MV export is lazy: a legacy recipe never asks libavcodec to
    /// publish side data it will not inspect.
    motion_vectors_enabled: bool,
    /// Allocated only while wipe or smear can reach the pixels. It is dropped
    /// as soon as both controls return to their deadband, so an invisible
    /// trail cannot resurface later in the same armed codec interval.
    motion_state: Option<MoshMotionState>,
    fails: u32,
    good: u32,
}

fn open_mosh_encoder(
    mosh_width: u32,
    mosh_height: u32,
    bitrate: u32,
) -> Result<ffmpeg::encoder::video::Encoder, String> {
    let codec = ffmpeg::encoder::find(ffmpeg::codec::Id::MPEG4)
        .ok_or_else(|| "the linked FFmpeg build has no mpeg4 encoder".to_string())?;
    let mut context = ffmpeg::codec::context::Context::new_with_codec(codec);
    // threads = 1 before open: single-threaded encode is the per-host
    // determinism lever, and avcodec_open2 reads the field exactly once.
    context.set_threading(ffmpeg::threading::Config::count(1));
    let mut video = context
        .encoder()
        .video()
        .map_err(|error| format!("mpeg4 encoder context: {error}"))?;
    video.set_width(mosh_width);
    video.set_height(mosh_height);
    video.set_format(ffmpeg::format::Pixel::YUV420P);
    video.set_time_base(ffmpeg::Rational(1, 30));
    video.set_frame_rate(Some(ffmpeg::Rational(30, 1)));
    // Keyframes are forced explicitly per frame. 600 is the mpeg4 encoder's
    // own hard ceiling (it clamps anything larger), so the encoder volunteers
    // a key at most once every 600 fed frames — and a volunteered key is just
    // another key chunk: it faces the removal dice exactly like a forced one,
    // so `key_removal = 1` still never recovers.
    video.set_gop(600);
    video.set_max_b_frames(0);
    video.set_bit_rate(bitrate as usize);
    video
        .open_with(ffmpeg::Dictionary::new())
        .map_err(|error| format!("mpeg4 encoder open: {error}"))
}

fn open_mosh_decoder(export_motion_vectors: bool) -> Result<ffmpeg::decoder::Video, String> {
    let codec = ffmpeg::decoder::find(ffmpeg::codec::Id::MPEG4)
        .ok_or_else(|| "the linked FFmpeg build has no mpeg4 decoder".to_string())?;
    let mut context = ffmpeg::codec::context::Context::new_with_codec(codec);
    context.set_threading(ffmpeg::threading::Config::count(1));
    // The whole point is to watch the decoder cope with a broken bitstream:
    // ask it to hand over concealed/corrupt pictures instead of hiding them.
    // SAFETY: the context owns a live, uniquely borrowed AVCodecContext and
    // both flags are public ABI written before open — the
    // `enable_export_motion_vectors` precedent.
    unsafe {
        (*context.as_mut_ptr()).flags |= ffmpeg::ffi::AV_CODEC_FLAG_OUTPUT_CORRUPT as i32;
        (*context.as_mut_ptr()).flags2 |= ffmpeg::ffi::AV_CODEC_FLAG2_SHOW_ALL;
        if export_motion_vectors {
            (*context.as_mut_ptr()).flags2 |= ffmpeg::ffi::AV_CODEC_FLAG2_EXPORT_MVS;
        }
    }
    context
        .decoder()
        .video()
        .map_err(|error| format!("mpeg4 decoder open: {error}"))
}

fn rgba_scaler(
    from: (u32, u32),
    to: (u32, u32),
) -> Result<ffmpeg::software::scaling::Context, String> {
    ffmpeg::software::scaling::Context::get(
        ffmpeg::format::Pixel::RGBA,
        from.0,
        from.1,
        ffmpeg::format::Pixel::YUV420P,
        to.0,
        to.1,
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )
    .map_err(|error| format!("mosh input scaler: {error}"))
}

impl MoshEngine {
    /// Open the pair for one output size. A missing encoder or decoder is a
    /// typed refusal here, never a panic downstream.
    pub fn open(width: u32, height: u32) -> Result<Self, String> {
        Self::open_configured(width, height, false)
    }

    fn open_configured(
        width: u32,
        height: u32,
        export_motion_vectors: bool,
    ) -> Result<Self, String> {
        ffmpeg::init().map_err(|error| format!("ffmpeg init failed: {error}"))?;
        if width == 0 || height == 0 {
            return Err(format!("mosh dimensions {width}x{height} are empty"));
        }
        let (mosh_width, mosh_height) = mosh_dimensions(width, height);
        let bitrate = mosh_target_bitrate(CodecMoshParams::default().bitrate_starve);
        let encoder = open_mosh_encoder(mosh_width, mosh_height, bitrate)?;
        let decoder = open_mosh_decoder(export_motion_vectors)?;
        let scale_in = rgba_scaler((width, height), (mosh_width, mosh_height))?;
        Ok(Self {
            width,
            height,
            mosh_width,
            mosh_height,
            encoder,
            decoder,
            scale_in,
            scale_out: None,
            current_bitrate: Some(bitrate),
            seen: false,
            need_key: true,
            frame_count: 0,
            emit_count: 0,
            ring: MoshChunkRing::new(),
            last_decoded: None,
            recycled: None,
            motion_vectors_enabled: export_motion_vectors,
            motion_state: None,
            fails: 0,
            good: 0,
        })
    }

    /// The full reset: build/stop/decoder-death semantics. The ring, the
    /// held picture, and the bootstrap flags all go together.
    fn reset_stream(&mut self) {
        self.seen = false;
        self.need_key = true;
        self.ring.clear();
        self.last_decoded = None;
        self.scale_out = None;
        self.recycled = None;
        if let Some(motion) = self.motion_state.as_mut() {
            motion.clear_history();
        }
    }

    fn rebuild_decoder(&mut self) -> Result<(), String> {
        // Commit the logical reset before a fallible host reopen. Even if
        // allocation/open fails, no prior key/ring/picture/recycle owner may
        // survive an outcome already reported as `Reset` to the feed loop.
        self.reset_stream();
        self.decoder = open_mosh_decoder(self.motion_vectors_enabled)?;
        Ok(())
    }

    fn reconfigure_bitrate_stream(
        &mut self,
        bitrate: u32,
        export_motion_vectors: bool,
    ) -> Result<(), String> {
        // Construct both contexts before replacing either: a failed host
        // open leaves the old, internally matched stream pair usable.
        let encoder = open_mosh_encoder(self.mosh_width, self.mosh_height, bitrate)?;
        let decoder = open_mosh_decoder(export_motion_vectors)?;
        self.encoder = encoder;
        self.decoder = decoder;
        self.motion_vectors_enabled = export_motion_vectors;
        self.current_bitrate = Some(bitrate);
        self.reset_stream();
        // This is an authored configuration change, not decoder death. Its
        // clean bootstrap begins a new health interval.
        self.fails = 0;
        self.good = 0;
        Ok(())
    }

    fn record_decode_fault(&mut self) {
        // BENDR's source law is strict: 30 is not forgiven; picture 31 is.
        self.fails = mosh_fault_count_after_decode_error(self.fails, self.good);
        self.good = 0;
    }

    fn recover_decoded_output(&mut self, reason: &str) {
        log::warn!("mosh decoded output rejected; forcing reacquire: {reason}");
        self.record_decode_fault();
        if self.fails <= MOSH_DECODER_FAIL_LIMIT {
            if let Err(error) = self.rebuild_decoder() {
                log::error!("mosh decoder recovery reopen failed: {error}");
            }
        } else {
            // The logical ownership reset is mandatory even when the retry
            // budget is exhausted and no host reopen will be attempted.
            self.reset_stream();
        }
    }

    /// Feed one chunk to the decoder under a fresh monotonic timestamp and
    /// drain whatever pictures fall out, bounded.
    fn emit(&mut self, bytes: &[u8]) -> MoshEmitOutcome {
        let mut packet = ffmpeg::Packet::copy(bytes);
        packet.set_pts(Some(self.emit_count));
        packet.set_dts(Some(self.emit_count));
        self.emit_count += 1;
        if self.decoder.send_packet(&packet).is_err() {
            self.record_decode_fault();
            if self.fails <= MOSH_DECODER_FAIL_LIMIT {
                if let Err(error) = self.rebuild_decoder() {
                    log::error!("mosh decoder rebuild failed: {error}");
                }
            } else {
                self.reset_stream();
            }
            return MoshEmitOutcome::Reset;
        }
        let mut decoded_picture = false;
        for _ in 0..MOSH_MAX_DECODE_FRAMES_PER_CHUNK {
            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {}
                Err(error) if mosh_decoder_receive_would_block(&error) => break,
                Err(error) => {
                    // This engine never sends decoder EOF, so EOF and every
                    // non-EAGAIN fault mean the broken stream escaped the
                    // ordinary drain boundary. Retire all prior ownership and
                    // re-enter through the documented bootstrap policy.
                    self.recover_decoded_output(&format!("decoder receive failed: {error}"));
                    return MoshEmitOutcome::Reset;
                }
            }
            self.good = self.good.saturating_add(1);
            self.last_decoded = Some(decoded);
            decoded_picture = true;
        }
        if decoded_picture {
            MoshEmitOutcome::AcceptedPicture
        } else {
            MoshEmitOutcome::Accepted
        }
    }

    /// One frame's round trip, in place. `pixels` is the tightly packed
    /// RGBA audience image; on return it holds the dry/wet blend. A frame
    /// with no decoded picture yet (or a removed bootstrap) passes the dry
    /// image through unchanged — the engine's own prior-image hold covers
    /// every later starvation.
    #[allow(
        clippy::too_many_arguments,
        reason = "the synchronous kernel keeps dimensions, sampled params/matte mode, and deterministic clock keys explicit at its live/export boundary"
    )]
    pub fn apply(
        &mut self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        params: CodecMoshParams,
        use_influence_alpha: bool,
        ordinal: u64,
        seed: u32,
    ) -> Result<(), String> {
        let params = params.sanitized();
        // Recycled ownership leaves the engine before any fallible frame
        // validation or codec work. Every error path therefore drops it.
        let prior_recycled = if params.recycle {
            self.recycled.take()
        } else {
            self.recycled = None;
            None
        };
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(4))
            .ok_or_else(|| "mosh frame size overflowed".to_string())?;
        if width == 0 || height == 0 || pixels.len() < expected {
            return Err(format!(
                "mosh frame {width}x{height} does not match {} bytes",
                pixels.len()
            ));
        }
        // Move, never clone, the former 4K output. Ownership stays local
        // until this apply succeeds, so any early return/fault drops stale
        // pixels and a RECYCLED frame pays no second full-frame allocation.
        let mut recycled_input = prior_recycled.filter(|prior| prior.len() >= expected);
        let active = params.amount >= MOSH_AMOUNT_DEADBAND;
        let motion_shaping = active && params.motion_shaping_active();
        if !motion_shaping {
            self.motion_state = None;
        }
        if !active {
            for pixel in pixels[..expected].chunks_exact_mut(4) {
                pixel[3] = 255;
            }
            return Ok(());
        }
        if self.fails > MOSH_DECODER_FAIL_LIMIT {
            return Err("mosh decoder unavailable after repeated faults".to_string());
        }
        if width != self.width || height != self.height {
            // A resized programme is a new stream: rebuild the whole pair.
            discard_recycled_input_on_stream_reset(&mut recycled_input);
            *self = Self::open_configured(width, height, motion_shaping)?;
        }

        // Bitrate starvation with the ±25% hysteresis; a reconfigure is a
        // fresh encoder AND decoder. Retaining the old decoder, delta ring,
        // or held/recycled images lets the former stream leak into the new
        // encoder sequence and can strand the decoder before its bootstrap.
        let want = mosh_target_bitrate(params.bitrate_starve);
        let bitrate_changed = mosh_bitrate_reconfigure_needed(self.current_bitrate, want);
        let enable_motion_vectors =
            mosh_motion_decoder_rebuild_needed(self.motion_vectors_enabled, motion_shaping);
        if bitrate_changed {
            // A bitrate change already requires a new decoder, so this
            // natural reset sheds a sticky EXPORT_MVS flag when shaping is
            // currently off at no additional stream-thrash cost.
            self.reconfigure_bitrate_stream(want, motion_shaping)?;
            discard_recycled_input_on_stream_reset(&mut recycled_input);
        } else if enable_motion_vectors {
            // EXPORT_MVS is an open-time decoder flag. Switching it on is a
            // real decoder replacement and therefore a full stream reset.
            let decoder = open_mosh_decoder(true)?;
            self.decoder = decoder;
            self.motion_vectors_enabled = true;
            self.reset_stream();
            discard_recycled_input_on_stream_reset(&mut recycled_input);
            self.fails = 0;
            self.good = 0;
        }

        // Choose the encoder's input: the clean image, or the stage's own
        // previous blended output under the recycle law.
        let encode_source: &[u8] = recycled_input.as_deref().unwrap_or(&pixels[..expected]);

        // RGBA rows into an FFmpeg frame, honoring the aligned stride.
        let mut rgba = ffmpeg::util::frame::video::Video::new(
            ffmpeg::format::Pixel::RGBA,
            self.width,
            self.height,
        );
        {
            let stride = rgba.stride(0);
            let row_bytes = self.width as usize * 4;
            let data = rgba.data_mut(0);
            for row in 0..self.height as usize {
                data[row * stride..row * stride + row_bytes]
                    .copy_from_slice(&encode_source[row * row_bytes..(row + 1) * row_bytes]);
            }
        }
        let mut yuv = ffmpeg::util::frame::video::Video::empty();
        self.scale_in
            .run(&rgba, &mut yuv)
            .map_err(|error| format!("mosh downscale: {error}"))?;
        if motion_shaping {
            let needs_state = self.motion_state.as_ref().is_none_or(|motion| {
                motion.width != self.mosh_width || motion.height != self.mosh_height
            });
            if needs_state {
                self.motion_state = Some(MoshMotionState::new(self.mosh_width, self.mosh_height)?);
            }
            self.motion_state
                .as_mut()
                .expect("motion state was allocated above")
                .begin_frame(yuv.data(0), yuv.stride(0), ordinal, params.trail)?;
            if use_influence_alpha {
                self.motion_state
                    .as_mut()
                    .expect("motion state was allocated above")
                    .sample_influence_alpha(&pixels[..expected], width, height)?;
            } else {
                // Explicit mode is also the ownership boundary: ordinary
                // audience alpha cannot leave an influence memory that later
                // resurfaces when a packed matte is enabled.
                self.motion_state
                    .as_mut()
                    .expect("motion state was allocated above")
                    .clear_influence_history();
            }
        }
        yuv.set_pts(Some(self.frame_count as i64));
        let force_key =
            mosh_wants_keyframe(self.seen, self.need_key, params.resync, self.frame_count);
        yuv.set_kind(if force_key {
            ffmpeg::util::picture::Type::I
        } else {
            ffmpeg::util::picture::Type::None
        });
        self.frame_count += 1;
        self.encoder
            .send_frame(&yuv)
            .map_err(|error| format!("mosh encode: {error}"))?;

        // Drain the encoder and break the wire, chunk by chunk.
        let mut emits = 0_usize;
        let mut decoded_picture = false;
        'packets: for packet_index in 0..MOSH_MAX_PACKETS_PER_FRAME {
            let mut packet = ffmpeg::Packet::empty();
            if self.encoder.receive_packet(&mut packet).is_err() {
                break;
            }
            let Some(bytes) = packet.data().map(<[u8]>::to_vec) else {
                continue;
            };
            let index = packet_index as u32;
            if packet.is_key() {
                match decide_key_chunk(params, ordinal, index, seed, !self.seen || self.need_key) {
                    MoshKeyDecision::Emit => {
                        if emits < MOSH_MAX_EMITS_PER_FRAME {
                            let outcome = self.emit(&bytes);
                            emits += 1;
                            decoded_picture |= outcome.decoded_picture();
                            let bootstrap = mosh_bootstrap_after_emit(
                                MoshBootstrapState {
                                    seen: self.seen,
                                    need_key: self.need_key,
                                },
                                true,
                                outcome,
                            );
                            self.seen = bootstrap.seen;
                            self.need_key = bootstrap.need_key;
                            if outcome.reset() {
                                break 'packets;
                            }
                        }
                    }
                    MoshKeyDecision::Remove => {}
                }
                continue;
            }
            // Deltas enter the ring before their own dice, dropped or not.
            self.ring.push(bytes.clone());
            if !self.seen {
                // BENDR retains pre-bootstrap deltas for later shuffle but
                // never feeds them to a decoder that has no accepted key.
                continue;
            }
            let decision = decide_delta_chunk(params, ordinal, index, seed);
            if decision.dropped {
                continue;
            }
            if emits < MOSH_MAX_EMITS_PER_FRAME {
                let outcome = self.emit(&bytes);
                emits += 1;
                decoded_picture |= outcome.decoded_picture();
                if outcome.reset() {
                    break 'packets;
                }
            }
            for _ in 0..decision.extra_repeats {
                if emits >= MOSH_MAX_EMITS_PER_FRAME {
                    break;
                }
                let outcome = self.emit(&bytes);
                emits += 1;
                decoded_picture |= outcome.decoded_picture();
                if outcome.reset() {
                    break 'packets;
                }
            }
            if let Some(pick) = decision.shuffle_pick {
                if emits < MOSH_MAX_EMITS_PER_FRAME {
                    if let Some(stale) = self.ring.pick(pick).map(<[u8]>::to_vec) {
                        let outcome = self.emit(&stale);
                        emits += 1;
                        decoded_picture |= outcome.decoded_picture();
                        if outcome.reset() {
                            break 'packets;
                        }
                    }
                }
            }
        }
        if self.fails > MOSH_DECODER_FAIL_LIMIT {
            return Err("mosh decoder unavailable after repeated faults".to_string());
        }
        // At most one side-data scan per apply: all decoder feeds first
        // compete for `last_decoded`, then only that newest presented picture
        // supplies vectors. Held pictures are not reparsed on a later frame.
        if decoded_picture {
            let (motion_state, last_decoded) = (&mut self.motion_state, &self.last_decoded);
            if let (Some(motion), Some(decoded)) = (motion_state.as_mut(), last_decoded.as_ref()) {
                motion.ingest_decoder_frame(decoded);
            }
        }
        if let Some(motion) = self.motion_state.as_mut() {
            motion.finish_frame();
        }

        // Blend the newest decoded picture over the dry image. No picture
        // yet means the dry image passes through — the honest bootstrap.
        let Some(decoded) = self.last_decoded.as_ref() else {
            for pixel in pixels[..expected].chunks_exact_mut(4) {
                pixel[3] = 255;
            }
            return Ok(());
        };
        let decoded_format = decoded.format();
        let decoded_dimensions = [decoded.width(), decoded.height()];
        if !mosh_decoded_frame_in_fixed_domain(
            decoded_format,
            decoded_dimensions,
            [self.mosh_width, self.mosh_height],
        ) {
            self.recover_decoded_output("format/dimensions escaped the fixed MPEG-4 domain");
            for pixel in pixels[..expected].chunks_exact_mut(4) {
                pixel[3] = 255;
            }
            return Ok(());
        }
        if self.scale_out.is_none() {
            match ffmpeg::software::scaling::Context::get(
                decoded_format,
                decoded_dimensions[0],
                decoded_dimensions[1],
                ffmpeg::format::Pixel::RGBA,
                self.width,
                self.height,
                ffmpeg::software::scaling::flag::Flags::BILINEAR,
            ) {
                Ok(scale_out) => self.scale_out = Some(scale_out),
                Err(error) => {
                    self.recover_decoded_output(&format!("output scaler construction: {error}"));
                    for pixel in pixels[..expected].chunks_exact_mut(4) {
                        pixel[3] = 255;
                    }
                    return Ok(());
                }
            }
        }
        let scale_out = self.scale_out.as_mut().expect("output scaler built above");
        let mut wet = ffmpeg::util::frame::video::Video::empty();
        let decoded = self
            .last_decoded
            .as_ref()
            .expect("decoded picture survived domain validation");
        if let Err(error) = scale_out.run(decoded, &mut wet) {
            self.recover_decoded_output(&format!("output scaler run: {error}"));
            for pixel in pixels[..expected].chunks_exact_mut(4) {
                pixel[3] = 255;
            }
            return Ok(());
        }
        let stride = wet.stride(0);
        let data = wet.data(0);
        let row_bytes = self.width as usize * 4;
        let amount = params.amount;
        if motion_shaping {
            let motion = self
                .motion_state
                .as_ref()
                .expect("motion shaping keeps its bounded state alive");
            for row in 0..self.height as usize {
                let dry_row = &mut pixels[row * row_bytes..(row + 1) * row_bytes];
                let codec_y = (u64::try_from(row).unwrap_or(u64::MAX) * u64::from(self.mosh_height)
                    / u64::from(self.height))
                .min(u64::from(self.mosh_height - 1)) as u32;
                let cell_y = codec_y / MOSH_MOTION_BLOCK_PIXELS;
                for cell_x in 0..motion.grid_width {
                    let texels = motion_cell_output_span(cell_x, self.mosh_width, self.width);
                    let (wake, forward, trail_influence) = motion.sample_cell(cell_x, cell_y);
                    let cell_amount = motion_local_amount(amount, params.wipe, wake);
                    let smear_mix = params.smear * wake;
                    if smear_mix > 0.0 {
                        let smear_offset = motion_smear_offset(
                            forward,
                            params.smear,
                            [self.width, self.height],
                            [self.mosh_width, self.mosh_height],
                        );
                        let source_row = (row as i64 - smear_offset[1])
                            .clamp(0, i64::from(self.height - 1))
                            as usize;
                        for texel in texels {
                            let base = texel * 4;
                            let influence = if use_influence_alpha {
                                // Current coverage remains exact per pixel. Only
                                // a prior coarse send, already gated by its
                                // retained wake, can admit a vacated motion trail.
                                (f32::from(dry_row[base + 3]) / 255.0).max(trail_influence)
                            } else {
                                1.0
                            };
                            let local_amount = cell_amount * influence;
                            let source_column = (texel as i64 - smear_offset[0])
                                .clamp(0, i64::from(self.width - 1))
                                as usize;
                            let current_wet = row * stride + base;
                            let source_wet = source_row * stride + source_column * 4;
                            for channel in 0..3 {
                                let dry = f32::from(dry_row[base + channel]);
                                let wet_value = f32::from(data[current_wet + channel]);
                                let displaced = f32::from(data[source_wet + channel]);
                                let shaped_wet = wet_value + (displaced - wet_value) * smear_mix;
                                dry_row[base + channel] =
                                    (dry + (shaped_wet - dry) * local_amount).round() as u8;
                            }
                            dry_row[base + 3] = 255;
                        }
                    } else {
                        // Wipe-only recipes and zero-wake cells need no
                        // displaced wet read. At 4K this removes one complete
                        // extra RGB fetch (~23.7 MiB) from the common path.
                        for texel in texels {
                            let base = texel * 4;
                            let influence = if use_influence_alpha {
                                (f32::from(dry_row[base + 3]) / 255.0).max(trail_influence)
                            } else {
                                1.0
                            };
                            let local_amount = cell_amount * influence;
                            let current_wet = row * stride + base;
                            for channel in 0..3 {
                                let dry = f32::from(dry_row[base + channel]);
                                let wet_value = f32::from(data[current_wet + channel]);
                                dry_row[base + channel] =
                                    (dry + (wet_value - dry) * local_amount).round() as u8;
                            }
                            dry_row[base + 3] = 255;
                        }
                    }
                }
            }
        } else if use_influence_alpha {
            // The matte-only route intentionally retains the historical
            // uniform pixel law apart from its explicit per-pixel multiplier.
            for row in 0..self.height as usize {
                let wet_row = &data[row * stride..row * stride + row_bytes];
                let dry_row = &mut pixels[row * row_bytes..(row + 1) * row_bytes];
                for texel in 0..self.width as usize {
                    let base = texel * 4;
                    let local_amount = amount * f32::from(dry_row[base + 3]) / 255.0;
                    for channel in 0..3 {
                        let index = base + channel;
                        let dry = f32::from(dry_row[index]);
                        let wet_value = f32::from(wet_row[index]);
                        dry_row[index] = (dry + (wet_value - dry) * local_amount).round() as u8;
                    }
                    dry_row[base + 3] = 255;
                }
            }
        } else {
            // Literal legacy fast path: with no motion controls and no packed
            // matte, pixel arithmetic and memory traffic are unchanged.
            for row in 0..self.height as usize {
                let wet_row = &data[row * stride..row * stride + row_bytes];
                let dry_row = &mut pixels[row * row_bytes..(row + 1) * row_bytes];
                for texel in 0..self.width as usize {
                    for channel in 0..3 {
                        let index = texel * 4 + channel;
                        let dry = f32::from(dry_row[index]);
                        let wet_value = f32::from(wet_row[index]);
                        dry_row[index] = (dry + (wet_value - dry) * amount).round() as u8;
                    }
                    // The audience image is opaque; the round trip must not
                    // reintroduce coverage.
                    dry_row[texel * 4 + 3] = 255;
                }
            }
        }
        if params.recycle {
            let mut next = recycled_input.unwrap_or_default();
            next.clear();
            next.extend_from_slice(&pixels[..expected]);
            self.recycled = Some(next);
        } else {
            self.recycled = None;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The worker: the global-VHS shape verbatim.
// ---------------------------------------------------------------------------

use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError};

/// At most one job is in flight; while the worker is busy a new frame is
/// simply skipped — healthy bounded backpressure, counted as such.
pub struct MoshWorker {
    job_tx: SyncSender<MoshJob>,
    result_rx: Receiver<Result<MoshProcessedFrame, String>>,
    in_flight: usize,
    failed: bool,
    last_error: String,
}

fn mosh_worker_process(
    engine: &mut Option<MoshEngine>,
    ntsc_state: &mut crate::ntsc::NtscState,
    job: MoshJob,
) -> Result<MoshProcessedFrame, String> {
    let mut pixels = job.pixels;
    let MoshFrameMetadata {
        params,
        use_influence_alpha,
        ordinal,
        seed,
        generation,
        ntsc,
    } = job.metadata;
    if engine
        .as_ref()
        .is_none_or(|engine| engine.width != job.width || engine.height != job.height)
    {
        *engine = Some(MoshEngine::open_configured(
            job.width,
            job.height,
            params.motion_shaping_active(),
        )?);
    }
    let engine = engine.as_mut().expect("mosh engine opened above");
    engine.apply(
        &mut pixels,
        job.width,
        job.height,
        params,
        use_influence_alpha,
        ordinal,
        seed,
    )?;
    // The combined hop keeps the asynchronous boundary and bounded queue
    // unchanged while making VHS the final stylized programme operation.
    let ntsc_processed = ntsc.is_some();
    if let Some(metadata) = ntsc {
        ntsc_state.params = metadata.params;
        if !ntsc_state.apply_at_reference_frame(
            &mut pixels,
            job.width,
            job.height,
            metadata.reference_frame,
        ) {
            return Err(format!(
                "NTSC rejected combined Codec Mosh frame {}x{} (invalid buffer or disabled effect)",
                job.width, job.height
            ));
        }
    }
    Ok(MoshProcessedFrame {
        pixels,
        epoch: job.epoch,
        generation,
        ntsc_processed,
    })
}

impl MoshWorker {
    #[must_use]
    pub fn new() -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel::<MoshJob>(1);
        let (result_tx, result_rx) =
            std::sync::mpsc::sync_channel::<Result<MoshProcessedFrame, String>>(1);
        let spawn = std::thread::Builder::new()
            .name("mosh-worker".to_string())
            .spawn(move || {
                let mut engine: Option<MoshEngine> = None;
                let mut ntsc_state = crate::ntsc::NtscState::new();
                let mut active_generation: Option<(u64, u64)> = None;
                while let Ok(job) = job_rx.recv() {
                    // A re-armed interval starts from a fresh codec/VHS state.
                    // Old work may finish, but its tagged result is rejected by
                    // the host and can never seed the new interval's codec.
                    let job_generation = (job.epoch, job.metadata.generation);
                    if active_generation != Some(job_generation) {
                        engine = None;
                        ntsc_state = crate::ntsc::NtscState::new();
                        active_generation = Some(job_generation);
                    }
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        mosh_worker_process(&mut engine, &mut ntsc_state, job)
                    }))
                    .unwrap_or_else(|_| {
                        Err("mosh worker panicked while processing a frame".to_string())
                    });
                    if result_tx.send(result).is_err() {
                        return;
                    }
                }
            });
        match spawn {
            Ok(_) => Self {
                job_tx,
                result_rx,
                in_flight: 0,
                failed: false,
                last_error: String::new(),
            },
            Err(error) => {
                let message = format!("Failed to spawn mosh worker: {error}");
                log::error!("{message}");
                Self {
                    job_tx,
                    result_rx,
                    in_flight: 0,
                    failed: true,
                    last_error: message,
                }
            }
        }
    }

    /// Admission: a dead worker is `Unavailable` before a busy one is
    /// `Busy`, exactly the NTSC ladder.
    pub fn try_submit_outcome(
        &mut self,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        metadata: MoshFrameMetadata,
        epoch: u64,
    ) -> crate::ntsc::NtscSubmitOutcome {
        if self.failed {
            return crate::ntsc::NtscSubmitOutcome::Unavailable;
        }
        if self.in_flight > 0 {
            return crate::ntsc::NtscSubmitOutcome::Busy;
        }
        let job = MoshJob {
            pixels,
            width,
            height,
            metadata,
            epoch,
        };
        match self.job_tx.try_send(job) {
            Ok(()) => {
                self.in_flight += 1;
                crate::ntsc::NtscSubmitOutcome::Accepted
            }
            Err(TrySendError::Full(_)) => crate::ntsc::NtscSubmitOutcome::Busy,
            Err(TrySendError::Disconnected(_)) => {
                self.mark_failed("mosh worker input disconnected");
                crate::ntsc::NtscSubmitOutcome::Unavailable
            }
        }
    }

    pub fn try_recv(&mut self) -> Option<MoshProcessedFrame> {
        match self.result_rx.try_recv() {
            Ok(Ok(frame)) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                Some(frame)
            }
            Ok(Err(error)) => {
                // The slot is released either way: one bad frame must not
                // wedge the path.
                self.in_flight = self.in_flight.saturating_sub(1);
                log::error!("mosh worker: {error}");
                self.last_error = error;
                None
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.mark_failed("mosh worker output disconnected");
                None
            }
        }
    }

    #[must_use]
    pub fn error(&self) -> &str {
        &self.last_error
    }

    #[must_use]
    pub fn is_busy(&self) -> bool {
        !self.failed && self.in_flight > 0
    }

    fn mark_failed(&mut self, message: &str) {
        self.failed = true;
        self.in_flight = 0;
        if self.last_error.is_empty() {
            self.last_error = message.to_string();
        }
    }
}

impl Default for MoshWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostile_scalars_sanitize_to_the_neutral_default_never_a_clamped_extreme() {
        let hostile = CodecMoshParams {
            amount: f32::NAN,
            key_removal: f32::INFINITY,
            hold: f32::NEG_INFINITY,
            drop: -3.0,
            shuffle: 9.0,
            rate: f32::NAN,
            bitrate_starve: f32::NAN,
            resync: f32::NAN,
            wipe: f32::INFINITY,
            smear: f32::NEG_INFINITY,
            trail: f32::NAN,
            recycle: true,
        };
        let clean = hostile.sanitized();
        let neutral = CodecMoshParams::default();
        assert_eq!(clean.amount, neutral.amount);
        assert_eq!(clean.key_removal, neutral.key_removal);
        assert_eq!(clean.hold, neutral.hold);
        assert_eq!(clean.drop, 0.0, "finite input clamps");
        assert_eq!(clean.shuffle, 1.0, "finite input clamps");
        assert_eq!(clean.rate, neutral.rate);
        assert_eq!(clean.bitrate_starve, neutral.bitrate_starve);
        assert_eq!(clean.resync, neutral.resync);
        assert_eq!(clean.wipe, neutral.wipe);
        assert_eq!(clean.smear, neutral.smear);
        assert_eq!(clean.trail, neutral.trail);
        assert!(clean.recycle, "the discrete law is not a scalar");
    }

    #[test]
    fn motion_controls_default_to_the_literal_legacy_analysis_fast_path() {
        let neutral = CodecMoshParams::default();
        assert_eq!(
            (neutral.wipe, neutral.smear, neutral.trail),
            (0.0, 0.0, 0.0)
        );
        assert!(!neutral.motion_shaping_active());
        assert!(
            !CodecMoshParams {
                trail: 1.0,
                ..neutral
            }
            .motion_shaping_active(),
            "an invisible trail alone allocates no analysis history"
        );
        assert!(CodecMoshParams {
            wipe: MOSH_AMOUNT_DEADBAND,
            ..neutral
        }
        .motion_shaping_active());
        assert!(CodecMoshParams {
            smear: MOSH_AMOUNT_DEADBAND,
            ..neutral
        }
        .motion_shaping_active());
    }

    #[test]
    fn the_wake_law_is_amount_alone_at_bendrs_own_deadband() {
        let mut params = CodecMoshParams {
            key_removal: 1.0,
            hold: 1.0,
            drop: 1.0,
            shuffle: 1.0,
            rate: 1.0,
            bitrate_starve: 1.0,
            resync: 1.0,
            recycle: true,
            ..CodecMoshParams::default()
        };
        assert!(
            !params.is_active(),
            "dressing controls shape an inactive mechanism and wake nothing"
        );
        params.amount = MOSH_AMOUNT_DEADBAND;
        assert!(params.is_active());
        params.amount = MOSH_AMOUNT_DEADBAND * 0.5;
        assert!(!params.is_active());
    }

    #[test]
    fn the_bitrate_map_and_hysteresis_follow_the_transcribed_law() {
        assert_eq!(mosh_target_bitrate(0.0), 4_000_000);
        assert_eq!(mosh_target_bitrate(1.0), 80_000);
        // The 0.35 default lands near one megabit.
        let default = mosh_target_bitrate(0.35);
        assert!((900_000..1_200_000).contains(&default), "{default}");
        // Hysteresis: exactly a quarter away does not reconfigure; one more
        // does. An unconfigured engine always does.
        assert!(mosh_bitrate_reconfigure_needed(None, 1_000_000));
        assert!(!mosh_bitrate_reconfigure_needed(Some(1_250_000), 1_000_000));
        assert!(mosh_bitrate_reconfigure_needed(Some(1_250_001), 1_000_000));
        assert!(!mosh_bitrate_reconfigure_needed(Some(750_000), 1_000_000));
        assert!(mosh_bitrate_reconfigure_needed(Some(749_999), 1_000_000));
    }

    #[test]
    fn decoder_export_transitions_avoid_off_thrashing_and_natural_resets_shed_the_flag() {
        assert!(mosh_motion_decoder_rebuild_needed(false, true));
        assert!(!mosh_motion_decoder_rebuild_needed(true, true));
        assert!(
            !mosh_motion_decoder_rebuild_needed(true, false),
            "an immediate off crossing does not reset the stream"
        );
        assert!(
            !mosh_motion_decoder_rebuild_needed(false, false),
            "a naturally reconfigured decoder receives motion_shaping=false"
        );
        let mut moved_recycle = Some(vec![7_u8; 16]);
        discard_recycled_input_on_stream_reset(&mut moved_recycle);
        assert!(
            moved_recycle.is_none(),
            "a locally moved recycle owner cannot cross any stream reset"
        );
    }

    #[test]
    fn decoder_receive_treats_only_eagain_as_normal_drain_completion() {
        let would_block = ffmpeg::Error::Other {
            errno: ffmpeg::error::EAGAIN,
        };
        assert!(mosh_decoder_receive_would_block(&would_block));
        assert!(
            !mosh_decoder_receive_would_block(&ffmpeg::Error::Eof),
            "this persistent decoder never receives EOF; it must reset instead"
        );
        assert!(!mosh_decoder_receive_would_block(
            &ffmpeg::Error::InvalidData
        ));
    }

    #[test]
    fn bootstrap_emit_outcomes_and_good_streak_match_the_source_state_machine() {
        let bootstrap = MoshBootstrapState {
            seen: false,
            need_key: true,
        };
        assert!(
            !bootstrap.seen,
            "a capped/skipped key leaves bootstrap unchanged"
        );
        let accepted = mosh_bootstrap_after_emit(bootstrap, true, MoshEmitOutcome::AcceptedPicture);
        assert_eq!(
            accepted,
            MoshBootstrapState {
                seen: true,
                need_key: false
            }
        );
        let reset = mosh_bootstrap_after_emit(accepted, false, MoshEmitOutcome::Reset);
        assert_eq!(reset, bootstrap, "any reset forces an honest key reacquire");
        assert!(!bootstrap.seen, "deltas are gated until a key is accepted");

        assert_eq!(
            mosh_fault_count_after_decode_error(4, MOSH_DECODER_GOOD_STREAK),
            5,
            "BENDR's 30-picture boundary remains strict"
        );
        assert_eq!(
            mosh_fault_count_after_decode_error(4, MOSH_DECODER_GOOD_STREAK + 1),
            1,
            "picture 31 forgives the old failure count"
        );
    }

    #[test]
    fn decoded_output_domain_rejects_format_and_dimension_drift_before_swscale() {
        assert!(mosh_decoded_frame_in_fixed_domain(
            ffmpeg::format::Pixel::YUV420P,
            [320, 180],
            [320, 180]
        ));
        assert!(!mosh_decoded_frame_in_fixed_domain(
            ffmpeg::format::Pixel::RGBA,
            [320, 180],
            [320, 180]
        ));
        assert!(!mosh_decoded_frame_in_fixed_domain(
            ffmpeg::format::Pixel::YUV420P,
            [640, 360],
            [320, 180]
        ));
        assert!(!mosh_decoded_frame_in_fixed_domain(
            ffmpeg::format::Pixel::YUV420P,
            [641, 180],
            [641, 180]
        ));
    }

    #[test]
    fn the_resync_period_matches_the_transcribed_table() {
        assert_eq!(mosh_resync_period(0.0), None, "zero never recovers");
        assert_eq!(mosh_resync_period(f32::NAN), None, "non-finite is neutral");
        assert_eq!(mosh_resync_period(0.004), Some(301));
        assert_eq!(mosh_resync_period(0.5), Some(152));
        assert_eq!(mosh_resync_period(1.0), Some(2));
        // The forced clock: bootstrap and explicit re-acquire always key.
        assert!(mosh_wants_keyframe(false, false, 0.0, 7));
        assert!(mosh_wants_keyframe(true, true, 0.0, 7));
        assert!(!mosh_wants_keyframe(true, false, 0.0, 7));
        assert!(mosh_wants_keyframe(true, false, 0.5, 152));
        assert!(!mosh_wants_keyframe(true, false, 0.5, 153));
    }

    #[test]
    fn the_dimension_law_caps_the_longest_edge_at_640_even_and_at_least_64() {
        assert_eq!(mosh_dimensions(1920, 1080), (640, 360));
        assert_eq!(mosh_dimensions(1280, 720), (640, 360));
        assert_eq!(mosh_dimensions(640, 481), (640, 480));
        assert_eq!(mosh_dimensions(320, 240), (320, 240));
        assert_eq!(mosh_dimensions(100, 50), (100, 64), "the floor holds");
        assert_eq!(mosh_dimensions(3841, 2161), (640, 360));
        assert_eq!(mosh_dimensions(1080, 1920), (360, 640));
        assert_eq!(mosh_dimensions(64, 16_384), (64, 640));
        assert_eq!(mosh_dimensions(1, 1), (64, 64));
        for dimensions in [(1920, 1080), (1080, 1920), (65_535, 1), (1, 65_535)] {
            let encoded = mosh_dimensions(dimensions.0, dimensions.1);
            assert!(encoded.0 <= MOSH_MAX_WIDTH && encoded.1 <= MOSH_MAX_WIDTH);
            assert_eq!(encoded.0 % 2, 0);
            assert_eq!(encoded.1 % 2, 0);
        }
    }

    #[test]
    fn luma_motion_wake_is_bounded_and_trail_decays_by_reference_ordinal() {
        assert!(MoshMotionState::new(MOSH_MAX_WIDTH + 1, 64).is_err());
        let mut motion = MoshMotionState::new(32, 16).expect("two bounded macroblocks");
        let quiet = vec![0_u8; 32 * 16];
        motion.begin_frame(&quiet, 32, 0, 0.8).unwrap();
        motion.finish_frame();
        assert!(motion.wake.iter().all(|wake| *wake == 0.0));

        let changed = vec![255_u8; 32 * 16];
        motion.begin_frame(&changed, 32, 1, 0.5).unwrap();
        motion.finish_frame();
        assert!(motion.wake.iter().all(|wake| *wake == 1.0));

        motion.begin_frame(&changed, 32, 2, 0.5).unwrap();
        motion.finish_frame();
        assert!(motion
            .wake
            .iter()
            .all(|wake| (*wake - 0.5).abs() <= f32::EPSILON));

        motion.begin_frame(&changed, 32, 3, 0.0).unwrap();
        motion.finish_frame();
        assert!(
            motion.wake.iter().all(|wake| *wake == 0.0),
            "zero trail forgets a stopped object's wake in one reference tick"
        );
    }

    #[test]
    fn reverse_ordinals_invalidate_luma_and_long_gaps_decay_every_tick() {
        let quiet = vec![0_u8; 32 * 16];
        let changed = vec![255_u8; 32 * 16];
        let mut reversed = MoshMotionState::new(32, 16).unwrap();
        reversed.begin_frame(&quiet, 32, 10, 1.0).unwrap();
        reversed.begin_frame(&changed, 32, 9, 1.0).unwrap();
        reversed.finish_frame();
        assert!(reversed.wake.iter().all(|wake| *wake == 0.0));

        let six_hundred = motion_trail_decay(0.999, 600);
        let thousand = motion_trail_decay(0.999, 1_000);
        assert!(
            thousand < six_hundred,
            "elapsed ticks are never capped at 600"
        );
        let mut gap = MoshMotionState::new(32, 16).unwrap();
        gap.wake.fill(1.0);
        gap.last_ordinal = Some(0);
        gap.previous_luma_valid = true;
        gap.begin_frame(&quiet, 32, 1_000, 0.999).unwrap();
        assert!(gap
            .wake
            .iter()
            .all(|wake| (*wake - thousand).abs() <= f32::EPSILON));
    }

    #[test]
    fn high_send_motion_retains_a_coarse_trail_over_low_send_background() {
        let (width, height) = (64_u32, 16_u32);
        let quiet = vec![0_u8; (width * height) as usize];
        let mut high_object = vec![0_u8; (width * height * 4) as usize];
        for y in 0..height as usize {
            for x in 0..16_usize {
                high_object[(y * width as usize + x) * 4 + 3] = 255;
            }
        }
        let low_background = vec![0_u8; high_object.len()];
        let mut motion = MoshMotionState::new(width, height).unwrap();
        motion.begin_frame(&quiet, width as usize, 0, 0.5).unwrap();
        motion.fallback_wake[0] = 1.0;
        motion
            .sample_influence_alpha(&high_object, width, height)
            .unwrap();
        motion.finish_frame();
        assert_eq!(
            motion.output_trail_influence[0], 0.0,
            "current alpha stays exact"
        );
        assert_eq!(motion.retained_influence[0], 1.0);

        motion.begin_frame(&quiet, width as usize, 1, 0.5).unwrap();
        motion
            .sample_influence_alpha(&low_background, width, height)
            .unwrap();
        motion.finish_frame();
        let (_, _, vacated_trail) = motion.sample_cell(0, 0);
        let (_, _, distant_background) = motion.sample_cell(3, 0);
        assert!((vacated_trail - 0.5).abs() <= f32::EPSILON);
        assert_eq!(distant_background, 0.0);
    }

    #[test]
    fn luma_fallback_can_dominate_wake_without_erasing_decoder_direction() {
        let mut motion = MoshMotionState::new(32, 16).unwrap();
        motion.fallback_wake[0] = 1.0;
        motion.decoder_wake[1] = 0.5;
        motion.decoder_vectors[1] = [2.0, 0.0];
        motion.finish_frame();
        assert_eq!(motion.dilated_wake[0], 1.0);
        assert_eq!(motion.dilated_vectors[0], [2.0, 0.0]);
        assert_eq!(motion.trail_vectors[0], [2.0, 0.0]);
    }

    #[test]
    fn decoder_motion_vectors_are_transactionally_validated_and_rasterized() {
        // SAFETY: AVMotionVector is a plain FFmpeg ABI record. Zero is a
        // valid starting representation and every semantically required
        // field is assigned before validation.
        let mut raw: ffmpeg::ffi::AVMotionVector = unsafe { std::mem::zeroed() };
        raw.source = -1;
        raw.w = 16;
        raw.h = 16;
        raw.src_x = 12;
        raw.src_y = 8;
        raw.dst_x = 16;
        raw.dst_y = 8;
        raw.motion_x = -4;
        raw.motion_y = 0;
        raw.motion_scale = 1;
        // SAFETY: the slice borrows exactly the initialized ABI record's
        // bytes for the duration of this call.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::addr_of!(raw).cast::<u8>(),
                std::mem::size_of_val(&raw),
            )
        };
        let mut wake = [0.0_f32; 2];
        let mut vectors = [[0.0_f32; 2]; 2];
        let mut best_area = [u32::MAX; 2];
        assert!(rasterize_mosh_motion_side_data(
            bytes,
            [32, 16],
            [2, 1],
            &mut wake,
            &mut vectors,
            &mut best_area,
        ));
        assert_eq!(wake, [1.0, 1.0]);
        assert_eq!(vectors, [[4.0, 0.0], [4.0, 0.0]]);

        let mut sentinel_wake = [0.25_f32; 2];
        let mut sentinel_vectors = [[9.0_f32, 8.0]; 2];
        let mut sentinel_areas = [7_u32; 2];
        assert!(!rasterize_mosh_motion_side_data(
            &bytes[..bytes.len() - 1],
            [32, 16],
            [2, 1],
            &mut sentinel_wake,
            &mut sentinel_vectors,
            &mut sentinel_areas,
        ));
        assert_eq!(sentinel_wake, [0.25; 2]);
        assert_eq!(sentinel_vectors, [[9.0, 8.0]; 2]);
        assert_eq!(sentinel_areas, [7; 2]);

        // A complete first record followed by a same-size hostile record is
        // also transactional: validation finishes before either record may
        // publish a cell.
        let mut invalid_later = raw;
        invalid_later.motion_scale = 0;
        let mixed_records = [raw, invalid_later];
        // SAFETY: the byte slice covers exactly the initialized record array.
        let mixed_bytes = unsafe {
            std::slice::from_raw_parts(
                mixed_records.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&mixed_records),
            )
        };
        let mut mixed_wake = [0.25_f32; 2];
        let mut mixed_vectors = [[9.0_f32, 8.0]; 2];
        let mut mixed_areas = [7_u32; 2];
        assert!(!rasterize_mosh_motion_side_data(
            mixed_bytes,
            [32, 16],
            [2, 1],
            &mut mixed_wake,
            &mut mixed_vectors,
            &mut mixed_areas,
        ));
        assert_eq!(mixed_wake, [0.25; 2]);
        assert_eq!(mixed_vectors, [[9.0, 8.0]; 2]);
        assert_eq!(mixed_areas, [7; 2]);

        raw.motion_scale = 0;
        assert!(validate_mosh_motion_vector(raw, [32, 16]).is_err());
    }

    #[test]
    fn conflicting_decoder_vectors_choose_motion_strength_before_block_area() {
        // SAFETY: both records are plain initialized FFmpeg ABI values.
        let mut strong: ffmpeg::ffi::AVMotionVector = unsafe { std::mem::zeroed() };
        strong.source = -1;
        strong.w = 16;
        strong.h = 16;
        strong.dst_x = 16;
        strong.dst_y = 8;
        strong.src_x = 8;
        strong.src_y = 8;
        strong.motion_x = -8;
        strong.motion_scale = 1;
        let mut weak = strong;
        weak.w = 8;
        weak.h = 8;
        weak.src_x = 15;
        weak.motion_x = -1;
        let records = [strong, weak];
        // SAFETY: the byte slice covers exactly the initialized record array.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                records.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&records),
            )
        };
        let mut wake = [0.0_f32; 2];
        let mut vectors = [[0.0_f32; 2]; 2];
        let mut areas = [u32::MAX; 2];
        assert!(rasterize_mosh_motion_side_data(
            bytes,
            [32, 16],
            [2, 1],
            &mut wake,
            &mut vectors,
            &mut areas,
        ));
        assert_eq!(vectors, [[8.0, 0.0], [8.0, 0.0]]);
        assert_eq!(areas, [256, 256]);
    }

    #[test]
    fn wipe_and_smear_math_is_clamped_constant_work_per_output_texel() {
        assert_eq!(motion_local_amount(0.8, 0.0, 0.0), 0.8);
        assert_eq!(motion_local_amount(0.8, 1.0, 0.0), 0.0);
        assert!((motion_local_amount(0.8, 1.0, 0.5) - 0.4).abs() <= f32::EPSILON);
        assert_eq!(
            motion_smear_source([50, 50], [4.0, 0.0], 1.0, [100, 100], [100, 100]),
            [2, 50]
        );
        assert_eq!(
            motion_smear_source([2, 2], [1000.0, 1000.0], 1.0, [100, 100], [100, 100]),
            [0, 0],
            "hostile decoder displacement cannot read outside the wet frame"
        );
        let mut boundary = 0_usize;
        for cell in 0..100_u32.div_ceil(MOSH_MOTION_BLOCK_PIXELS) {
            let span = motion_cell_output_span(cell, 100, 1_919);
            assert_eq!(span.start, boundary, "macroblock spans have no gaps");
            boundary = span.end;
        }
        assert_eq!(boundary, 1_919, "the fused spans cover the full output row");
    }

    #[test]
    fn the_fault_dice_are_deterministic_per_lane_and_never_cross_lanes() {
        let a = mosh_hash(7, 0, 42, MOSH_LANE_DROP);
        assert_eq!(a, mosh_hash(7, 0, 42, MOSH_LANE_DROP));
        assert!((0.0..1.0).contains(&a));
        assert_ne!(a, mosh_hash(7, 0, 42, MOSH_LANE_HOLD_GATE));
        assert_ne!(a, mosh_hash(8, 0, 42, MOSH_LANE_DROP));
        assert_ne!(a, mosh_hash(7, 1, 42, MOSH_LANE_DROP));
        assert_ne!(a, mosh_hash(7, 0, 43, MOSH_LANE_DROP));
    }

    #[test]
    fn the_key_bootstrap_always_passes_and_later_keys_face_the_dice() {
        let params = CodecMoshParams {
            amount: 1.0,
            key_removal: 1.0,
            ..CodecMoshParams::default()
        };
        assert_eq!(
            decide_key_chunk(params, 0, 0, 1, true),
            MoshKeyDecision::Emit,
            "the decoder needs one whole picture to damage"
        );
        assert_eq!(
            decide_key_chunk(params, 0, 0, 1, false),
            MoshKeyDecision::Remove,
            "key_removal = 1 removes every later key, including forced resyncs"
        );
        let never = CodecMoshParams {
            amount: 1.0,
            key_removal: 0.0,
            ..CodecMoshParams::default()
        };
        for ordinal in 0..64 {
            assert_eq!(
                decide_key_chunk(never, ordinal, 0, 1, false),
                MoshKeyDecision::Emit
            );
        }
    }

    #[test]
    fn delta_decisions_honor_the_probability_gates_and_the_drop_early_return() {
        let quiet = CodecMoshParams {
            amount: 1.0,
            hold: 0.0,
            drop: 0.0,
            shuffle: 0.0,
            rate: 1.0,
            ..CodecMoshParams::default()
        };
        for ordinal in 0..64 {
            let decision = decide_delta_chunk(quiet, ordinal, 0, 9);
            assert!(!decision.dropped);
            assert_eq!(decision.extra_repeats, 0);
            assert!(decision.shuffle_pick.is_none());
        }
        let loud = CodecMoshParams {
            amount: 1.0,
            hold: 1.0,
            drop: 1.0,
            shuffle: 1.0,
            rate: 1.0,
            ..CodecMoshParams::default()
        };
        for ordinal in 0..64 {
            let decision = decide_delta_chunk(loud, ordinal, 0, 9);
            assert!(decision.dropped, "drop wins first");
            assert_eq!(decision.extra_repeats, 0, "a dropped chunk rolls no hold");
            assert!(decision.shuffle_pick.is_none(), "nor shuffle");
        }
        let holding = CodecMoshParams {
            amount: 1.0,
            hold: 1.0,
            drop: 0.0,
            shuffle: 0.0,
            rate: 1.0,
            ..CodecMoshParams::default()
        };
        let mut maximum = 0;
        for ordinal in 0..4_096 {
            let decision = decide_delta_chunk(holding, ordinal, 0, 9);
            assert!(!decision.dropped);
            assert!((1..=5).contains(&decision.extra_repeats));
            maximum = maximum.max(decision.extra_repeats);
        }
        assert_eq!(maximum, 5, "the source law reaches its full 1..=5 range");
        // A zero event rate silences hold/drop/shuffle but never key removal.
        let gated = CodecMoshParams {
            amount: 1.0,
            hold: 1.0,
            drop: 1.0,
            shuffle: 1.0,
            rate: 0.0,
            key_removal: 1.0,
            ..CodecMoshParams::default()
        };
        for ordinal in 0..64 {
            let decision = decide_delta_chunk(gated, ordinal, 0, 9);
            assert!(!decision.dropped);
            assert_eq!(decision.extra_repeats, 0);
            assert!(decision.shuffle_pick.is_none());
            assert_eq!(
                decide_key_chunk(gated, ordinal, 0, 9, false),
                MoshKeyDecision::Remove,
                "key removal is deliberately exempt from the event rate"
            );
        }
    }

    #[test]
    fn the_chunk_ring_is_bounded_in_entries_and_bytes_and_shields_the_newest_six() {
        let mut ring = MoshChunkRing::new();
        assert!(ring.pick(0.5).is_none(), "an empty ring never picks");
        for index in 0..MOSH_RING_MAX_CHUNKS + 10 {
            ring.push(vec![index as u8; 8]);
        }
        assert_eq!(ring.len(), MOSH_RING_MAX_CHUNKS, "FIFO entry cap");
        assert_eq!(ring.bytes(), MOSH_RING_MAX_CHUNKS * 8);

        // The byte cap binds independently: one chunk over evicts the oldest.
        let mut fat = MoshChunkRing::new();
        let half = MOSH_RING_MAX_BYTES / 2;
        fat.push(vec![1; half]);
        fat.push(vec![2; half]);
        assert_eq!(fat.len(), 2);
        fat.push(vec![3; 1]);
        assert_eq!(fat.len(), 2, "the oldest half was evicted");
        assert!(fat.bytes() <= MOSH_RING_MAX_BYTES);
        assert_eq!(fat.pick(0.0), None, "under the shuffle gate nothing picks");

        // The pick law: gate at more than ten, and the newest six excluded.
        let mut gated = MoshChunkRing::new();
        for index in 0..=MOSH_RING_SHUFFLE_GATE {
            gated.push(vec![index as u8]);
        }
        assert_eq!(gated.len(), MOSH_RING_SHUFFLE_GATE + 1);
        let newest_selectable = (gated.len() - MOSH_RING_NEWEST_EXCLUDED - 1) as u8;
        assert_eq!(gated.pick(0.0), Some(&[0_u8][..]));
        assert_eq!(gated.pick(0.999_999), Some(&[newest_selectable][..]));

        ring.clear();
        assert!(ring.is_empty());
        assert_eq!(ring.bytes(), 0);
    }

    #[test]
    fn the_encoder_identity_names_the_codec_and_the_linked_avcodec() {
        let identity = mosh_encoder_identity();
        assert!(identity.starts_with("mpeg4/avcodec-"), "{identity}");
    }

    #[test]
    #[ignore = "requires the host FFmpeg swscale library"]
    fn rgba_to_yuv_analysis_ignores_the_explicit_influence_alpha_channel() {
        ffmpeg::init().expect("FFmpeg initializes");
        let (width, height) = (64_u32, 64_u32);
        let make_rgba = |alpha: u8| {
            let mut frame =
                ffmpeg::util::frame::video::Video::new(ffmpeg::format::Pixel::RGBA, width, height);
            let stride = frame.stride(0);
            let data = frame.data_mut(0);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let offset = y * stride + x * 4;
                    data[offset] = (x * 3 + y) as u8;
                    data[offset + 1] = (x + y * 5) as u8;
                    data[offset + 2] = (x * 7 + y * 11) as u8;
                    data[offset + 3] = alpha;
                }
            }
            frame
        };
        let mut scaler = rgba_scaler((width, height), (width, height)).unwrap();
        let mut transparent_yuv = ffmpeg::util::frame::video::Video::empty();
        let mut opaque_yuv = ffmpeg::util::frame::video::Video::empty();
        scaler.run(&make_rgba(0), &mut transparent_yuv).unwrap();
        scaler.run(&make_rgba(255), &mut opaque_yuv).unwrap();
        for plane in 0..3 {
            let plane_width = if plane == 0 { width } else { width / 2 } as usize;
            let plane_height = if plane == 0 { height } else { height / 2 } as usize;
            let transparent_stride = transparent_yuv.stride(plane);
            let opaque_stride = opaque_yuv.stride(plane);
            for row in 0..plane_height {
                assert_eq!(
                    &transparent_yuv.data(plane)
                        [row * transparent_stride..row * transparent_stride + plane_width],
                    &opaque_yuv.data(plane)[row * opaque_stride..row * opaque_stride + plane_width],
                    "YUV plane {plane}, row {row} must be independent of packed matte alpha"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires the host FFmpeg's mpeg4 encoder/decoder pair"]
    fn bitrate_reconfigure_clears_every_old_stream_owner_and_early_returns_are_opaque() {
        let (width, height) = (128_u32, 96_u32);
        let mut engine = MoshEngine::open(width, height).unwrap();
        engine.ring.push(vec![1, 2, 3]);
        engine.last_decoded = Some(ffmpeg::util::frame::video::Video::empty());
        engine.recycled = Some(vec![7; (width * height * 4) as usize]);
        engine.motion_state = Some(MoshMotionState::new(width, height).unwrap());
        engine.motion_state.as_mut().unwrap().wake.fill(1.0);
        engine.seen = true;
        engine.need_key = false;
        engine.fails = 3;
        engine.good = 9;

        let bitrate = mosh_target_bitrate(1.0);
        engine.reconfigure_bitrate_stream(bitrate, true).unwrap();
        assert_eq!(engine.current_bitrate, Some(bitrate));
        assert!(engine.motion_vectors_enabled);
        assert!(engine.ring.is_empty());
        assert!(engine.last_decoded.is_none());
        assert!(engine.recycled.is_none());
        assert!(!engine.seen && engine.need_key);
        assert_eq!((engine.fails, engine.good), (0, 0));
        assert!(engine
            .motion_state
            .as_ref()
            .unwrap()
            .wake
            .iter()
            .all(|wake| *wake == 0.0));

        engine.ring.push(vec![4, 5, 6]);
        engine.last_decoded = Some(ffmpeg::util::frame::video::Video::empty());
        engine.recycled = Some(vec![8; (width * height * 4) as usize]);
        engine.seen = true;
        engine.need_key = false;
        engine.recover_decoded_output("synthetic scaler ownership fixture");
        assert!(engine.ring.is_empty());
        assert!(engine.last_decoded.is_none());
        assert!(engine.recycled.is_none());
        assert!(!engine.seen && engine.need_key);

        engine.recycled = Some(vec![9; (width * height * 4) as usize]);
        let mut inactive = vec![0_u8; (width * height * 4) as usize];
        for (index, pixel) in inactive.chunks_exact_mut(4).enumerate() {
            pixel.copy_from_slice(&[(index % 251) as u8, 17, 29, 0]);
        }
        let dry_rgb: Vec<[u8; 3]> = inactive
            .chunks_exact(4)
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect();
        engine
            .apply(
                &mut inactive,
                width,
                height,
                CodecMoshParams::default(),
                true,
                0,
                1,
            )
            .unwrap();
        assert!(engine.recycled.is_none(), "clean revocation is immediate");
        for (pixel, expected_rgb) in inactive.chunks_exact(4).zip(dry_rgb) {
            assert_eq!(&pixel[..3], &expected_rgb);
            assert_eq!(pixel[3], 255, "every successful early return is opaque");
        }
    }

    #[test]
    #[ignore = "requires the host FFmpeg's mpeg4 encoder/decoder pair"]
    fn active_influence_alpha_gates_rgb_and_flagless_frames_ignore_it() {
        let (width, height) = (128_u32, 96_u32);
        let params = CodecMoshParams {
            amount: 1.0,
            key_removal: 0.0,
            hold: 0.0,
            drop: 0.0,
            shuffle: 0.0,
            rate: 0.0,
            bitrate_starve: 0.8,
            resync: 1.0,
            recycle: false,
            ..CodecMoshParams::default()
        };
        let source_frame = |ordinal: u64| {
            let mut pixels = Vec::with_capacity((width * height * 4) as usize);
            for y in 0..height {
                for x in 0..width {
                    let moving_edge = u8::from((x + ordinal as u32 * 9) % width > width / 2);
                    pixels.extend_from_slice(&[
                        moving_edge
                            .saturating_mul(211)
                            .saturating_add((x % 37) as u8),
                        ((x * 5 + y * 3 + ordinal as u32 * 7) % 251) as u8,
                        240_u8.saturating_sub(moving_edge.saturating_mul(173)),
                        255,
                    ]);
                }
            }
            pixels
        };
        let mut full_send = MoshEngine::open(width, height).unwrap();
        let mut zero_send = MoshEngine::open(width, height).unwrap();
        let mut flagless_arbitrary = MoshEngine::open(width, height).unwrap();
        let mut flagless_opaque = MoshEngine::open(width, height).unwrap();
        let mut wet_reached_rgb = false;

        for ordinal in 0..12_u64 {
            let source = source_frame(ordinal);
            let mut full = source.clone();
            let mut dry = source.clone();
            let mut arbitrary = source.clone();
            let mut reference = source.clone();
            for (index, pixel) in dry.chunks_exact_mut(4).enumerate() {
                pixel[3] = 0;
                arbitrary[index * 4 + 3] = ((index * 73 + ordinal as usize * 19) % 256) as u8;
            }

            full_send
                .apply(&mut full, width, height, params, true, ordinal, 0x15a0)
                .unwrap();
            zero_send
                .apply(&mut dry, width, height, params, true, ordinal, 0x15a0)
                .unwrap();
            flagless_arbitrary
                .apply(
                    &mut arbitrary,
                    width,
                    height,
                    params,
                    false,
                    ordinal,
                    0x15a0,
                )
                .unwrap();
            flagless_opaque
                .apply(
                    &mut reference,
                    width,
                    height,
                    params,
                    false,
                    ordinal,
                    0x15a0,
                )
                .unwrap();

            for (((source, full), dry), (arbitrary, reference)) in source
                .chunks_exact(4)
                .zip(full.chunks_exact(4))
                .zip(dry.chunks_exact(4))
                .zip(arbitrary.chunks_exact(4).zip(reference.chunks_exact(4)))
            {
                assert_eq!(&dry[..3], &source[..3], "alpha zero must stay dry");
                assert_eq!(dry[3], 255);
                assert_eq!(&full[..3], &reference[..3], "alpha 255 is full send");
                assert_eq!(
                    &arbitrary[..3],
                    &reference[..3],
                    "the explicit false capability ignores arbitrary alpha"
                );
                assert_eq!((full[3], arbitrary[3], reference[3]), (255, 255, 255));
                wet_reached_rgb |= full[..3] != source[..3];
            }
        }
        assert!(
            wet_reached_rgb,
            "the active full-send control must demonstrably reach decoded RGB"
        );
    }

    #[test]
    #[ignore = "requires the host FFmpeg's mpeg4 encoder/decoder pair"]
    fn motion_wipe_smear_and_stopped_object_trail_are_deterministic_per_host() {
        let (width, height) = (128_u32, 96_u32);
        let frame = |ordinal: u64| {
            let phase = ordinal.min(6) as u32;
            let object_left = 8 + phase * 10;
            let mut pixels = Vec::with_capacity((width * height * 4) as usize);
            for y in 0..height {
                for x in 0..width {
                    let moving = x >= object_left && x < object_left + 24 && (24..48).contains(&y);
                    let texture = ((x * 13 + y * 7) % 191) as u8;
                    let color = if moving {
                        [245, 32, texture, 255]
                    } else {
                        [texture, texture / 2, 255_u8.saturating_sub(texture), 255]
                    };
                    pixels.extend_from_slice(&color);
                }
            }
            pixels
        };
        let shaped = CodecMoshParams {
            amount: 1.0,
            key_removal: 0.0,
            hold: 0.5,
            drop: 0.0,
            shuffle: 0.0,
            rate: 1.0,
            bitrate_starve: 0.35,
            resync: 0.0,
            wipe: 1.0,
            smear: 0.8,
            trail: 0.85,
            recycle: false,
        };
        let run_shaped = || {
            let mut engine = MoshEngine::open_configured(width, height, true).unwrap();
            let mut outputs = Vec::new();
            let mut saw_decoder_vector = false;
            for ordinal in 0..10_u64 {
                let mut pixels = frame(ordinal);
                engine
                    .apply(&mut pixels, width, height, shaped, false, ordinal, 0x51de)
                    .unwrap();
                let motion = engine.motion_state.as_ref().unwrap();
                saw_decoder_vector |= motion
                    .trail_vectors
                    .iter()
                    .any(|vector| vector[0] != 0.0 || vector[1] != 0.0);
                outputs.push(pixels);
            }
            let retained_wake = engine
                .motion_state
                .as_ref()
                .unwrap()
                .wake
                .iter()
                .copied()
                .fold(0.0_f32, f32::max);
            let ingest_count = engine.motion_state.as_ref().unwrap().decoder_ingest_count;
            assert!(engine.motion_vectors_enabled);
            (outputs, retained_wake, saw_decoder_vector, ingest_count)
        };
        let first = run_shaped();
        let second = run_shaped();
        assert_eq!(first.0, second.0);
        assert!(first.1 > 0.0, "the stopped object retains a decaying wake");
        assert!(first.2, "the MPEG-4 decoder exported a usable local vector");
        assert!(
            (1..=10).contains(&first.3),
            "side data is ingested at most once per apply"
        );

        let classic = CodecMoshParams {
            wipe: 0.0,
            smear: 0.0,
            trail: 0.0,
            ..shaped
        };
        let mut classic_engine = MoshEngine::open(width, height).unwrap();
        let mut classic_outputs = Vec::new();
        for ordinal in 0..10_u64 {
            let mut pixels = frame(ordinal);
            classic_engine
                .apply(&mut pixels, width, height, classic, false, ordinal, 0x51de)
                .unwrap();
            classic_outputs.push(pixels);
        }
        assert!(
            first
                .0
                .iter()
                .zip(classic_outputs)
                .any(|(shaped, classic)| *shaped != classic),
            "motion controls alter the legacy uniform wet blend"
        );
    }

    /// The real round trip on this host: two engines fed the identical frame
    /// sequence must produce byte-identical output (threads = 1 is the
    /// per-host determinism lever; cross-machine identity is deliberately
    /// not claimed), the moshed image must demonstrably differ from the dry
    /// one, and an inactive amount must be a true no-touch bypass.
    #[test]
    #[ignore = "requires the host FFmpeg's mpeg4 encoder/decoder pair"]
    fn mosh_round_trip_is_deterministic_per_host_and_reaches_the_pixels() {
        let (w, h) = (320_u32, 180_u32);
        let frame = |index: u64| -> Vec<u8> {
            let mut pixels = Vec::with_capacity((w * h * 4) as usize);
            for y in 0..h {
                for x in 0..w {
                    // A moving gradient with a hard edge, so deltas carry
                    // real motion for the hold law to misapply.
                    let edge = u8::from((x + index as u32 * 7) % w > w / 2) * 200;
                    pixels.extend_from_slice(&[
                        edge.saturating_add((x % 251) as u8),
                        (y % 251) as u8,
                        255 - edge,
                        255,
                    ]);
                }
            }
            pixels
        };
        let params = CodecMoshParams {
            amount: 1.0,
            key_removal: 0.95,
            hold: 0.6,
            drop: 0.2,
            shuffle: 0.4,
            rate: 1.0,
            bitrate_starve: 0.6,
            resync: 0.0,
            wipe: 0.0,
            smear: 0.0,
            trail: 0.0,
            recycle: false,
        };
        let run = || -> Vec<Vec<u8>> {
            let mut engine = MoshEngine::open(w, h).expect("mpeg4 pair opens");
            (0..12_u64)
                .map(|ordinal| {
                    let mut pixels = frame(ordinal);
                    engine
                        .apply(&mut pixels, w, h, params, false, ordinal, 0x4235)
                        .expect("the round trip stays healthy");
                    pixels
                })
                .collect()
        };
        let first = run();
        let second = run();
        assert_eq!(first, second, "two runs on one host are byte-identical");
        let mut any_differs = false;
        for (ordinal, output) in first.iter().enumerate() {
            any_differs |= *output != frame(ordinal as u64);
        }
        assert!(any_differs, "the mosh demonstrably reaches the pixels");

        // The wake law in pixels: an inactive amount touches nothing, even
        // on an engine that already carries codec state.
        let mut engine = MoshEngine::open(w, h).unwrap();
        let mut warm = frame(0);
        engine
            .apply(&mut warm, w, h, params, false, 0, 0x4235)
            .unwrap();
        let dry = frame(1);
        let mut untouched = dry.clone();
        engine
            .apply(
                &mut untouched,
                w,
                h,
                CodecMoshParams::default(),
                false,
                1,
                0x4235,
            )
            .unwrap();
        assert_eq!(untouched, dry, "amount zero is a true no-touch bypass");
    }

    /// The engine half needs the host FFmpeg's mpeg4 pair, so the full round
    /// trip is opt-in like `effects_audit`; this hosted test proves only the
    /// bounded, typed refusal shape of the worker under backpressure.
    #[test]
    fn the_worker_is_one_in_flight_drop_new_and_releases_the_slot_on_error() {
        let mut worker = MoshWorker::new();
        let metadata = MoshFrameMetadata {
            params: CodecMoshParams {
                amount: 1.0,
                ..CodecMoshParams::default()
            },
            use_influence_alpha: false,
            ordinal: 0,
            seed: 1,
            generation: 1,
            ntsc: None,
        };
        // A hostile job (length mismatch) is a typed processing error: the
        // slot must come back and the worker must stay alive.
        let outcome = worker.try_submit_outcome(vec![0_u8; 16], 64, 64, metadata.clone(), 1);
        assert!(outcome.is_accepted());
        assert_eq!(
            worker.try_submit_outcome(Vec::new(), 64, 64, metadata.clone(), 1),
            crate::ntsc::NtscSubmitOutcome::Busy,
            "one in flight, drop-new-while-busy"
        );
        let mut released = false;
        for _ in 0..500 {
            if worker.try_recv().is_some() {
                panic!("a hostile job must not produce a frame");
            }
            if !worker.is_busy() {
                released = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(released, "a failed job releases the in-flight slot");
        assert!(
            !worker.error().is_empty(),
            "the failure is named, never silent"
        );
    }
}
