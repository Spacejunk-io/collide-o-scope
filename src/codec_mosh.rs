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

/// The encode resolution cap, transcribed from BENDR: at most 640 wide,
/// aspect preserved, both edges even and at least 64. "A software encoder
/// has to finish inside a frame or the queue grows without bound… the
/// artefact is the codec, not the detail."
pub const MOSH_MAX_WIDTH: u32 = 640;
pub const MOSH_MIN_EDGE: u32 = 64;

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
/// cycles gives the stage up with a named note, and a streak of thirty good
/// frames forgives the count back to one.
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

/// The authored stage. Eight continuous controls plus one discrete law, all
/// riding `TemporalParams` on the B3-rig closure pattern. Defaults are
/// BENDR's own; `amount` at zero is the exact prior path.
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
    /// Probability (× `rate`) that a delta chunk is re-applied 1–6 extra
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
            recycle: self.recycle,
        }
    }

    /// The wake law: `amount` alone arms the stage. Every other control
    /// shapes an inactive mechanism and wakes nothing — BENDR's own gate.
    #[must_use]
    pub fn is_active(self) -> bool {
        self.sanitized().amount >= MOSH_AMOUNT_DEADBAND
    }
}

/// The encode resolution for one output size: at most `MOSH_MAX_WIDTH` wide,
/// aspect preserved, both edges even and at least `MOSH_MIN_EDGE`.
#[must_use]
pub fn mosh_dimensions(width: u32, height: u32) -> (u32, u32) {
    let w = width.max(1);
    let h = height.max(1);
    let tw = w.min(MOSH_MAX_WIDTH);
    let th = ((f64::from(tw) * f64::from(h) / f64::from(w)).round() as u32).max(1);
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
    /// When the live global-VHS path is active while the mosh is armed, the
    /// worker runs the VHS kernel first in the same hop — one admission, one
    /// frame of latency, and the exact offline ordering (VHS, then mosh, on
    /// the same pixels). `None` on the disabled and selective paths, whose
    /// pixels are already the finished pre-mosh programme.
    pub ntsc: Option<crate::ntsc::NtscFrameMetadata>,
}

/// One processed frame returned by the worker. The caller must discard it
/// when its generation is no longer current.
pub struct MoshProcessedFrame {
    pub pixels: Vec<u8>,
    pub epoch: u64,
    pub generation: u64,
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

fn open_mosh_decoder() -> Result<ffmpeg::decoder::Video, String> {
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
        ffmpeg::init().map_err(|error| format!("ffmpeg init failed: {error}"))?;
        if width == 0 || height == 0 {
            return Err(format!("mosh dimensions {width}x{height} are empty"));
        }
        let (mosh_width, mosh_height) = mosh_dimensions(width, height);
        let bitrate = mosh_target_bitrate(CodecMoshParams::default().bitrate_starve);
        let encoder = open_mosh_encoder(mosh_width, mosh_height, bitrate)?;
        let decoder = open_mosh_decoder()?;
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
    }

    fn rebuild_decoder(&mut self) -> Result<(), String> {
        self.decoder = open_mosh_decoder()?;
        self.reset_stream();
        Ok(())
    }

    fn record_decode_fault(&mut self) {
        if self.good > MOSH_DECODER_GOOD_STREAK {
            // It was working; treat the fault as a one-off.
            self.fails = 1;
        } else {
            self.fails = self.fails.saturating_add(1);
        }
        self.good = 0;
    }

    /// Feed one chunk to the decoder under a fresh monotonic timestamp and
    /// drain whatever pictures fall out, bounded.
    fn emit(&mut self, bytes: &[u8]) {
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
            }
            return;
        }
        for _ in 0..MOSH_MAX_DECODE_FRAMES_PER_CHUNK {
            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            if self.decoder.receive_frame(&mut decoded).is_err() {
                break;
            }
            self.good = self.good.saturating_add(1);
            self.last_decoded = Some(decoded);
        }
    }

    /// One frame's round trip, in place. `pixels` is the tightly packed
    /// RGBA audience image; on return it holds the dry/wet blend. A frame
    /// with no decoded picture yet (or a removed bootstrap) passes the dry
    /// image through unchanged — the engine's own prior-image hold covers
    /// every later starvation.
    pub fn apply(
        &mut self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        params: CodecMoshParams,
        ordinal: u64,
        seed: u32,
    ) -> Result<(), String> {
        let params = params.sanitized();
        if !params.is_active() {
            return Ok(());
        }
        if self.fails > MOSH_DECODER_FAIL_LIMIT {
            return Err("mosh decoder unavailable after repeated faults".to_string());
        }
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
        if width != self.width || height != self.height {
            // A resized programme is a new stream: rebuild the whole pair.
            *self = Self::open(width, height)?;
        }

        // Bitrate starvation with the ±25% hysteresis; a reconfigure is a
        // fresh encoder and a forced re-acquire.
        let want = mosh_target_bitrate(params.bitrate_starve);
        if mosh_bitrate_reconfigure_needed(self.current_bitrate, want) {
            self.encoder = open_mosh_encoder(self.mosh_width, self.mosh_height, want)?;
            self.current_bitrate = Some(want);
            self.need_key = true;
            self.seen = false;
        }

        // Choose the encoder's input: the clean image, or the stage's own
        // previous blended output under the recycle law.
        let recycled_input = params
            .recycle
            .then(|| self.recycled.clone())
            .flatten()
            .filter(|prior| prior.len() >= expected);
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
        for packet_index in 0..MOSH_MAX_PACKETS_PER_FRAME {
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
                        self.seen = true;
                        self.need_key = false;
                        if emits < MOSH_MAX_EMITS_PER_FRAME {
                            self.emit(&bytes);
                            emits += 1;
                        }
                    }
                    MoshKeyDecision::Remove => {}
                }
                continue;
            }
            // Deltas enter the ring before their own dice, dropped or not.
            self.ring.push(bytes.clone());
            let decision = decide_delta_chunk(params, ordinal, index, seed);
            if decision.dropped {
                continue;
            }
            if emits < MOSH_MAX_EMITS_PER_FRAME {
                self.emit(&bytes);
                emits += 1;
            }
            for _ in 0..decision.extra_repeats {
                if emits >= MOSH_MAX_EMITS_PER_FRAME {
                    break;
                }
                self.emit(&bytes);
                emits += 1;
            }
            if let Some(pick) = decision.shuffle_pick {
                if emits < MOSH_MAX_EMITS_PER_FRAME {
                    if let Some(stale) = self.ring.pick(pick).map(<[u8]>::to_vec) {
                        self.emit(&stale);
                        emits += 1;
                    }
                }
            }
        }
        if self.fails > MOSH_DECODER_FAIL_LIMIT {
            return Err("mosh decoder unavailable after repeated faults".to_string());
        }

        // Blend the newest decoded picture over the dry image. No picture
        // yet means the dry image passes through — the honest bootstrap.
        let Some(decoded) = self.last_decoded.as_ref() else {
            return Ok(());
        };
        if self.scale_out.is_none() {
            self.scale_out = Some(
                ffmpeg::software::scaling::Context::get(
                    decoded.format(),
                    decoded.width(),
                    decoded.height(),
                    ffmpeg::format::Pixel::RGBA,
                    self.width,
                    self.height,
                    ffmpeg::software::scaling::flag::Flags::BILINEAR,
                )
                .map_err(|error| format!("mosh output scaler: {error}"))?,
            );
        }
        let scale_out = self.scale_out.as_mut().expect("output scaler built above");
        let mut wet = ffmpeg::util::frame::video::Video::empty();
        if scale_out.run(decoded, &mut wet).is_err() {
            // A decoded picture the scaler refuses (mid-death dimensions) is
            // a fault, not a frame: rebuild the output leg next time.
            self.scale_out = None;
            return Ok(());
        }
        let stride = wet.stride(0);
        let data = wet.data(0);
        let row_bytes = self.width as usize * 4;
        let amount = params.amount;
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
        if params.recycle {
            self.recycled = Some(pixels[..expected].to_vec());
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
    // The combined hop: on the live global-VHS path the worker runs the VHS
    // kernel first, on the same pixels, so the mosh eats the finished
    // programme in the exact offline order.
    if let Some(metadata) = job.metadata.ntsc {
        ntsc_state.params = metadata.params;
        ntsc_state.apply_at_reference_frame(
            &mut pixels,
            job.width,
            job.height,
            metadata.reference_frame,
        );
    }
    if engine
        .as_ref()
        .is_none_or(|engine| engine.width != job.width || engine.height != job.height)
    {
        *engine = Some(MoshEngine::open(job.width, job.height)?);
    }
    let engine = engine.as_mut().expect("mosh engine opened above");
    engine.apply(
        &mut pixels,
        job.width,
        job.height,
        job.metadata.params,
        job.metadata.ordinal,
        job.metadata.seed,
    )?;
    Ok(MoshProcessedFrame {
        pixels,
        epoch: job.epoch,
        generation: job.metadata.generation,
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
        assert!(clean.recycle, "the discrete law is not a scalar");
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
    fn the_dimension_law_caps_at_640_wide_even_and_at_least_64() {
        assert_eq!(mosh_dimensions(1920, 1080), (640, 360));
        assert_eq!(mosh_dimensions(1280, 720), (640, 360));
        assert_eq!(mosh_dimensions(640, 481), (640, 480));
        assert_eq!(mosh_dimensions(320, 240), (320, 240));
        assert_eq!(mosh_dimensions(100, 50), (100, 64), "the floor holds");
        assert_eq!(mosh_dimensions(3841, 2161), (640, 360));
        assert_eq!(mosh_dimensions(1, 1), (64, 64));
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
        let mut saw_extra = false;
        for ordinal in 0..64 {
            let decision = decide_delta_chunk(holding, ordinal, 0, 9);
            assert!(!decision.dropped);
            assert!((1..=6).contains(&decision.extra_repeats));
            saw_extra |= decision.extra_repeats > 1;
        }
        assert!(saw_extra, "the hold count spans its 1..=6 range");
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
            recycle: false,
        };
        let run = || -> Vec<Vec<u8>> {
            let mut engine = MoshEngine::open(w, h).expect("mpeg4 pair opens");
            (0..12_u64)
                .map(|ordinal| {
                    let mut pixels = frame(ordinal);
                    engine
                        .apply(&mut pixels, w, h, params, ordinal, 0x4235)
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
        engine.apply(&mut warm, w, h, params, 0, 0x4235).unwrap();
        let dry = frame(1);
        let mut untouched = dry.clone();
        engine
            .apply(&mut untouched, w, h, CodecMoshParams::default(), 1, 0x4235)
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
