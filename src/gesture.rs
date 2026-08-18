//! Portable gesture-field event contract shared by live authoring and export.
//!
//! This module owns CPU state and laws only. It deliberately has no `wgpu`,
//! clock, filesystem, or UI dependency: a caller supplies an already-derived
//! 30 Hz reference-tick address and one quantized sample, and the track answers
//! with a bounded, checksummable event stream that replays identically offline.
//! Wall time never enters the track, the checksum, or anything derived from
//! them — the reference tick is the only address a gesture ever has.
//!
//! The substrate is deliberately the M3 explicit-event track: `GestureTrack`
//! reproduces `TemporalEventTrack`'s origin rebasing, non-decreasing tick law,
//! `truncated` flag, and cap-returns-false contract, `GestureReplay`
//! reproduces `TemporalEventReplay`'s monotonic drain cursor, and
//! `GestureEventRecorder` reproduces `TemporalEventRecorder`'s accepted-frame
//! clock, whose tick is derived before this frame's delta is added so the
//! first accepted frame records at tick 0.
//!
//! `normalize_gesture_input` is the single adapter every input surface reaches.
//! A native pointer, a phone panel, a MIDI controller, and an OSC peer all
//! produce byte-identical events for the same logical gesture: the origin is
//! provenance, reaches only the typed refusal, and never the event bits.

#![allow(
    dead_code,
    reason = "S3b freezes the portable gesture event contract and its sidecar document before the canvas that consumes them lands"
)]

use std::fmt;

use serde::{de, Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use crate::effects::params::TEMPORAL_REFERENCE_FPS;

/// Append-only algorithm version for the portable gesture event stream. It
/// appears both in the sidecar document and inside the hashed field stream, so
/// a future encoding can never be mistaken for this one.
pub const GESTURE_ALGORITHM_VERSION: u16 = 1;

/// Gesture events are addressed on the same 30 Hz authoring reference as the
/// temporal history ring. This is that constant, not a second literal: a
/// gesture recorded at 24, 30, or 60 fps must land on the same tick address.
pub const GESTURE_REFERENCE_FPS: f32 = TEMPORAL_REFERENCE_FPS;

/// Hard cap for recorded gesture events retained for deterministic offline
/// replay. Reaching it sets `truncated` and refuses further recording; it never
/// panics and never silently drops the newest sample in place of an older one.
pub const MAX_GESTURE_EVENTS: usize = 4_096;

/// Byte ceiling for a serialized gesture-track document, enforced on both
/// encode and decode.
pub const MAX_GESTURE_SERIALIZED_BYTES: usize = 256 * 1024;

/// Stroke identity space. Because identity is the concurrency bound, at most
/// this many strokes can be open at once.
pub const MAX_ACTIVE_STROKES: usize = 16;

/// Ceiling on decay ticks processed in one canvas update. A longer gap clamps
/// rather than looping unbounded, mirroring the 24-tick history burst clamp.
pub const MAX_GESTURE_DECAY_TICKS: u32 = 4_096;

/// The open-stroke set is carried as a `u16` bitmask, so the identity space and
/// the mask width are the same number by construction.
const _: () = assert!(MAX_ACTIVE_STROKES == u16::BITS as usize);

/// Domain separator for the canonical checksum. The version literal appears
/// here as well as in the hashed `version` field, exactly as the recovery
/// journal repeats its version in both the magic and the domain string.
pub const GESTURE_CHECKSUM_DOMAIN: &[u8] = b"collide-o-scope/gesture-track/v1\0";

/// Fixed width of one encoded event: `tick:u32le`, `stroke:u8`, `phase:u8`,
/// `mode:u8`, `reserved:u8`, `x:u16le`, `y:u16le`, `pressure:u16le`,
/// `dx:i16le`, `dy:i16le`.
pub const GESTURE_EVENT_ENCODED_BYTES: usize = 18;

/// Domain, `version:u16le`, `flags:u16le`, `origin_tick:u64le`, and
/// `event_count:u32le` precede the event stream.
pub const GESTURE_CHECKSUM_HEADER_BYTES: usize = GESTURE_CHECKSUM_DOMAIN.len() + 2 + 2 + 8 + 4;

/// Reserved encoding byte, held at zero so a future per-event field can be
/// appended without moving any established offset.
const GESTURE_EVENT_RESERVED: u8 = 0;

/// Append-only checksum flag bits. The flags are part of the hashed stream, so
/// a truncated or explicitly incomplete recording can never be presented as a
/// complete one under the same digest.
pub const GESTURE_FLAG_TRUNCATED: u16 = 1 << 0;
pub const GESTURE_FLAG_INCOMPLETE: u16 = 1 << 1;

/// Q16 scale: an authored `[0, 1]` value is stored as `round(value * 65_535)`.
const Q16_SCALE: f32 = 65_535.0;

/// Q15 scale: an authored `[-1, 1]` component is stored as
/// `round(value * 32_767)`, leaving `i16::MIN` unreachable by quantization.
const Q15_SCALE: f32 = 32_767.0;

/// Quantize an authored unit value to its Q16 code. A non-finite authored value
/// takes the neutral zero code rather than a clamped extreme, matching the
/// established Displace sanitization law.
pub fn quantize_unit(value: f32) -> u16 {
    if !value.is_finite() {
        return 0;
    }
    (value.clamp(0.0, 1.0) * Q16_SCALE).round() as u16
}

/// Recover the authored unit value from its Q16 code.
pub fn dequantize_unit(raw: u16) -> f32 {
    f32::from(raw) / Q16_SCALE
}

/// Quantize an authored signed component to its Q15 code, with the same
/// neutral-zero fallback for non-finite input.
pub fn quantize_signed(value: f32) -> i16 {
    if !value.is_finite() {
        return 0;
    }
    (value.clamp(-1.0, 1.0) * Q15_SCALE).round() as i16
}

/// Recover the authored signed component from its Q15 code. The clamp only
/// matters for `i16::MIN`, which quantization cannot produce but a hostile
/// document can declare.
pub fn dequantize_signed(raw: i16) -> f32 {
    (f32::from(raw) / Q15_SCALE).clamp(-1.0, 1.0)
}

/// Stroke lifecycle phase. Codes are permanent and append-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GesturePhase {
    #[default]
    Begin,
    Move,
    End,
}

impl GesturePhase {
    /// Permanent append-only wire code. Never renumber an existing entry.
    pub const fn code(self) -> u8 {
        match self {
            Self::Begin => 0,
            Self::Move => 1,
            Self::End => 2,
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Begin),
            1 => Some(Self::Move),
            2 => Some(Self::End),
            _ => None,
        }
    }
}

/// Etching law carried by the stroke. `Push` displaces along the stroke
/// direction; `Curl` displaces along its perpendicular. Codes are permanent and
/// append-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GestureMode {
    #[default]
    Push,
    Curl,
}

impl GestureMode {
    /// Permanent append-only wire code. Never renumber an existing entry.
    pub const fn code(self) -> u8 {
        match self {
            Self::Push => 0,
            Self::Curl => 1,
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Push),
            1 => Some(Self::Curl),
            _ => None,
        }
    }
}

/// One portable, fixed-width gesture event.
///
/// The quantized codes are the only representation. There is no retained float
/// path: an authored sample is quantized once on ingest, so live authoring and
/// offline replay observe identical bits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GestureEvent {
    /// Reference tick relative to the track origin, stamped by the track.
    pub reference_tick: u32,
    pub stroke: u8,
    pub phase: GesturePhase,
    pub mode: GestureMode,
    /// Q16 normalized canvas position.
    pub x: u16,
    pub y: u16,
    /// Q16 normalized pressure.
    pub pressure: u16,
    /// Q15 direction components, renormalized on decode.
    pub direction_x: i16,
    pub direction_y: i16,
}

impl GestureEvent {
    /// Quantize an authored sample. The reference tick stays zero until a track
    /// addresses it: addressing is the track's authority, never the caller's.
    pub fn quantized(
        stroke: u8,
        phase: GesturePhase,
        mode: GestureMode,
        position: [f32; 2],
        pressure: f32,
        direction: [f32; 2],
    ) -> Self {
        Self {
            reference_tick: 0,
            stroke,
            phase,
            mode,
            x: quantize_unit(position[0]),
            y: quantize_unit(position[1]),
            pressure: quantize_unit(pressure),
            direction_x: quantize_signed(direction[0]),
            direction_y: quantize_signed(direction[1]),
        }
    }

    /// Stamp an explicit relative address. Used by the track and by tests that
    /// exercise the validator directly.
    pub const fn at(mut self, reference_tick: u32) -> Self {
        self.reference_tick = reference_tick;
        self
    }

    pub fn position(self) -> [f32; 2] {
        [dequantize_unit(self.x), dequantize_unit(self.y)]
    }

    pub fn pressure(self) -> f32 {
        dequantize_unit(self.pressure)
    }

    /// Decode the direction and renormalize it to unit length. A zero-length
    /// stored direction stays exactly zero rather than becoming an arbitrary
    /// axis, so an unset direction can never etch a stroke.
    pub fn direction(self) -> [f32; 2] {
        let x = dequantize_signed(self.direction_x);
        let y = dequantize_signed(self.direction_y);
        let length = x.mul_add(x, y * y).sqrt();
        if length > 0.0 {
            [x / length, y / length]
        } else {
            [0.0, 0.0]
        }
    }

    fn encode_into(self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.reference_tick.to_le_bytes());
        bytes.push(self.stroke);
        bytes.push(self.phase.code());
        bytes.push(self.mode.code());
        bytes.push(GESTURE_EVENT_RESERVED);
        bytes.extend_from_slice(&self.x.to_le_bytes());
        bytes.extend_from_slice(&self.y.to_le_bytes());
        bytes.extend_from_slice(&self.pressure.to_le_bytes());
        bytes.extend_from_slice(&self.direction_x.to_le_bytes());
        bytes.extend_from_slice(&self.direction_y.to_le_bytes());
    }
}

/// Typed failure vocabulary for gesture ingest, decode, and document handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GestureError {
    StrokeOutOfRange { stroke: u8 },
    TooManyOpenStrokes { stroke: u8 },
    StrokeAlreadyOpen { stroke: u8 },
    StrokeNotOpen { stroke: u8, phase: GesturePhase },
    NonMonotonicTick { previous: u32, observed: u32 },
    TickBeforeOrigin { absolute: u64, origin: u64 },
    TickOutOfRange { relative: u64 },
    TooManyEvents(usize),
    EventCount { declared: u32, observed: usize },
    OriginWithoutEvents(u64),
    UnsupportedVersion(u16),
    ChecksumMismatch { declared: String, computed: String },
    DocumentBytes(usize),
    Serialization(String),
}

impl fmt::Display for GestureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StrokeOutOfRange { stroke } => write!(
                formatter,
                "gesture stroke {stroke} is outside the {MAX_ACTIVE_STROKES}-stroke identity space"
            ),
            Self::TooManyOpenStrokes { stroke } => write!(
                formatter,
                "gesture stroke {stroke} would exceed {MAX_ACTIVE_STROKES} open strokes"
            ),
            Self::StrokeAlreadyOpen { stroke } => {
                write!(formatter, "gesture stroke {stroke} is already open")
            }
            Self::StrokeNotOpen { stroke, phase } => write!(
                formatter,
                "gesture {phase:?} for stroke {stroke} has no open Begin"
            ),
            Self::NonMonotonicTick { previous, observed } => write!(
                formatter,
                "gesture reference tick {observed} precedes the recorded tick {previous}"
            ),
            Self::TickBeforeOrigin { absolute, origin } => write!(
                formatter,
                "gesture reference tick {absolute} precedes the track origin {origin}"
            ),
            Self::TickOutOfRange { relative } => write!(
                formatter,
                "gesture reference tick {relative} exceeds the relative 32-bit window"
            ),
            Self::TooManyEvents(count) => write!(
                formatter,
                "gesture track declares {count} events above the {MAX_GESTURE_EVENTS}-event cap"
            ),
            Self::EventCount { declared, observed } => write!(
                formatter,
                "gesture track declares {declared} events but carries {observed}"
            ),
            Self::OriginWithoutEvents(origin) => write!(
                formatter,
                "empty gesture track declares origin tick {origin}"
            ),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported gesture track version {version}")
            }
            Self::ChecksumMismatch { declared, computed } => write!(
                formatter,
                "gesture track checksum {declared} does not match the computed {computed}"
            ),
            Self::DocumentBytes(bytes) => write!(
                formatter,
                "gesture document of {bytes} bytes is empty or above the {MAX_GESTURE_SERIALIZED_BYTES}-byte cap"
            ),
            Self::Serialization(error) => {
                write!(formatter, "gesture document serialization failed: {error}")
            }
        }
    }
}

impl std::error::Error for GestureError {}

/// The single well-formedness authority for a gesture event stream.
///
/// Live ingest and document decode both drive this one state machine, so a
/// hand-built document can never describe a stream that could not have been
/// recorded. Nothing here repairs: an orphan `Move`/`End` or a duplicate
/// `Begin` is rejected, and a stream whose strokes are still open is valid but
/// explicitly incomplete and is never auto-closed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GestureStreamValidator {
    open_strokes: u16,
    last_tick: Option<u32>,
}

impl GestureStreamValidator {
    pub const fn open_stroke_count(self) -> u32 {
        self.open_strokes.count_ones()
    }

    /// A stream with no open strokes is complete. Completeness is a fact about
    /// the recording, never a repair applied to it.
    pub const fn is_complete(self) -> bool {
        self.open_strokes == 0
    }

    /// Admit one already-addressed event, advancing the open-stroke set. The
    /// caller drives a copy first when a rejection must leave state untouched.
    pub fn accept(&mut self, event: &GestureEvent) -> Result<(), GestureError> {
        if usize::from(event.stroke) >= MAX_ACTIVE_STROKES {
            return Err(GestureError::StrokeOutOfRange {
                stroke: event.stroke,
            });
        }
        if let Some(previous) = self.last_tick {
            if event.reference_tick < previous {
                return Err(GestureError::NonMonotonicTick {
                    previous,
                    observed: event.reference_tick,
                });
            }
        }
        let bit = 1_u16 << event.stroke;
        match event.phase {
            GesturePhase::Begin => {
                if self.open_strokes & bit != 0 {
                    return Err(GestureError::StrokeAlreadyOpen {
                        stroke: event.stroke,
                    });
                }
                // The identity range above is the concurrency bound, so this
                // guard restates the law rather than adding a second one: a
                // future widening of the identity space cannot silently widen
                // how many strokes may be open at once.
                if self.open_stroke_count() as usize >= MAX_ACTIVE_STROKES {
                    return Err(GestureError::TooManyOpenStrokes {
                        stroke: event.stroke,
                    });
                }
                self.open_strokes |= bit;
            }
            GesturePhase::Move => {
                if self.open_strokes & bit == 0 {
                    return Err(GestureError::StrokeNotOpen {
                        stroke: event.stroke,
                        phase: GesturePhase::Move,
                    });
                }
            }
            GesturePhase::End => {
                if self.open_strokes & bit == 0 {
                    return Err(GestureError::StrokeNotOpen {
                        stroke: event.stroke,
                        phase: GesturePhase::End,
                    });
                }
                self.open_strokes &= !bit;
            }
        }
        self.last_tick = Some(event.reference_tick);
        Ok(())
    }
}

/// Portable, bounded gesture track passed by value into an export job or a
/// sidecar document. It contains no wall time, GPU state, source path, or
/// runtime identity.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GestureTrack {
    origin_tick: Option<u64>,
    events: Vec<GestureEvent>,
    truncated: bool,
    validator: GestureStreamValidator,
}

impl GestureTrack {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn events(&self) -> &[GestureEvent] {
        &self.events
    }

    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    /// Absolute host tick the first recorded event was rebased against.
    /// Absolute ticks never leave the recorder in any other form.
    pub const fn origin_tick(&self) -> Option<u64> {
        self.origin_tick
    }

    pub const fn open_stroke_count(&self) -> u32 {
        self.validator.open_stroke_count()
    }

    /// A track whose strokes are all closed. An incomplete track is still
    /// valid: it replays exactly as recorded and is never auto-closed.
    pub const fn is_complete(&self) -> bool {
        self.validator.is_complete()
    }

    /// Record one well-formed event at an absolute 30 Hz tick address.
    ///
    /// `Ok(true)` recorded it; `Ok(false)` means the track had already reached
    /// `MAX_GESTURE_EVENTS`, so `truncated` is now set and the recorded prefix
    /// is authoritative; `Err` means the event was ill-formed and was rejected
    /// rather than repaired, leaving the track byte-identical to before.
    ///
    /// After truncation the dropped prefix is not reconstructed: a later event
    /// for a stroke whose `Begin` never landed is reported as `StrokeNotOpen`.
    /// The `truncated` flag, not a repaired stream, is what the operator sees.
    pub fn record_accepted(
        &mut self,
        absolute_reference_tick: u64,
        event: GestureEvent,
    ) -> Result<bool, GestureError> {
        let origin = self.origin_tick.unwrap_or(absolute_reference_tick);
        let relative =
            absolute_reference_tick
                .checked_sub(origin)
                .ok_or(GestureError::TickBeforeOrigin {
                    absolute: absolute_reference_tick,
                    origin,
                })?;
        let reference_tick =
            u32::try_from(relative).map_err(|_| GestureError::TickOutOfRange { relative })?;
        let stamped = event.at(reference_tick);

        // Validate on a copy so a rejected event cannot half-advance the
        // open-stroke set or the monotonic tick watermark.
        let mut probe = self.validator;
        probe.accept(&stamped)?;

        if self.events.len() >= MAX_GESTURE_EVENTS {
            self.truncated = true;
            return Ok(false);
        }
        self.origin_tick = Some(origin);
        self.events.push(stamped);
        self.validator = probe;
        Ok(true)
    }

    pub fn replay(&self) -> GestureReplay<'_> {
        GestureReplay {
            events: &self.events,
            cursor: 0,
        }
    }

    /// Hashed flag word. Truncation and incompleteness are facts about the
    /// recording, so they are inside the digest rather than beside it.
    pub fn flags(&self) -> u16 {
        let mut flags = 0;
        if self.truncated {
            flags |= GESTURE_FLAG_TRUNCATED;
        }
        if !self.is_complete() {
            flags |= GESTURE_FLAG_INCOMPLETE;
        }
        flags
    }

    /// The canonical, domain-separated, explicit little-endian field stream.
    ///
    /// The event portion carries relative addresses only, so it never depends
    /// on how the host grouped samples into frames. The absolute origin is a
    /// declared hashed field rather than a hidden one: the same gesture made at
    /// two different program positions is honestly two recordings. No wall
    /// time, path, or runtime identity can enter the stream.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            GESTURE_CHECKSUM_HEADER_BYTES + self.events.len() * GESTURE_EVENT_ENCODED_BYTES,
        );
        bytes.extend_from_slice(GESTURE_CHECKSUM_DOMAIN);
        bytes.extend_from_slice(&GESTURE_ALGORITHM_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.flags().to_le_bytes());
        bytes.extend_from_slice(&self.origin_tick.unwrap_or(0).to_le_bytes());
        bytes.extend_from_slice(&self.declared_event_count().to_le_bytes());
        for event in &self.events {
            event.encode_into(&mut bytes);
        }
        bytes
    }

    pub fn checksum(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_bytes()).into()
    }

    pub fn checksum_hex(&self) -> String {
        format!("{:x}", Sha256::digest(self.canonical_bytes()))
    }

    /// Absolute 30 Hz address of the last recorded event, or zero for an empty
    /// track. Absolute ticks never leave the track in any other form, and this
    /// one exists so a restored recording can resume its session clock without
    /// rebasing — a rebase would rewrite the hashed origin of a recording the
    /// operator already holds.
    pub fn last_absolute_tick(&self) -> u64 {
        match (self.origin_tick, self.events.last()) {
            (Some(origin), Some(event)) => origin.saturating_add(u64::from(event.reference_tick)),
            _ => 0,
        }
    }

    /// The event count is bounded by `MAX_GESTURE_EVENTS`, so it always fits
    /// the hashed `u32` field.
    fn declared_event_count(&self) -> u32 {
        u32::try_from(self.events.len()).unwrap_or(u32::MAX)
    }
}

/// Borrow-only monotonic drain cursor over a recorded track.
#[derive(Debug, Clone)]
pub struct GestureReplay<'a> {
    events: &'a [GestureEvent],
    cursor: usize,
}

impl<'a> GestureReplay<'a> {
    /// Consume every event whose relative reference tick is due. A low display
    /// rate may cross several ticks in one frame; every crossed event is
    /// returned in recorded order rather than collapsed. The cursor is
    /// monotonic and never rewinds.
    pub fn events_due(&mut self, reference_tick: u32) -> &'a [GestureEvent] {
        let start = self.cursor;
        while self
            .events
            .get(self.cursor)
            .is_some_and(|event| event.reference_tick <= reference_tick)
        {
            self.cursor += 1;
        }
        &self.events[start..self.cursor]
    }
}

/// Count-capped event sequence. The allocation is bounded before the input is
/// trusted, so a declared-huge sequence is rejected by the cap rather than by
/// allocating first and checking afterwards.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundedGestureEvents(Vec<GestureEvent>);

impl<'de> Deserialize<'de> for BoundedGestureEvents {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Visitor;
        impl<'de> de::Visitor<'de> for Visitor {
            type Value = BoundedGestureEvents;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "at most {MAX_GESTURE_EVENTS} gesture events")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut values =
                    Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX_GESTURE_EVENTS));
                while values.len() < MAX_GESTURE_EVENTS {
                    let Some(value) = sequence.next_element()? else {
                        return Ok(BoundedGestureEvents(values));
                    };
                    values.push(value);
                }
                if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom("too many gesture events"));
                }
                Ok(BoundedGestureEvents(values))
            }
        }
        deserializer.deserialize_seq(Visitor)
    }
}

/// Portable sidecar document for a recorded gesture track.
///
/// The field set is frozen: `version`, `origin_tick`, `event_count`,
/// `truncated`, `checksum`, `events`. Operational paths and filesystem metadata
/// must never enter it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GestureTrackDocument {
    pub version: u16,
    pub origin_tick: u64,
    pub event_count: u32,
    pub truncated: bool,
    pub checksum: String,
    pub events: Vec<GestureEvent>,
}

impl GestureTrackDocument {
    pub fn capture(track: &GestureTrack) -> Self {
        Self {
            version: GESTURE_ALGORITHM_VERSION,
            origin_tick: track.origin_tick.unwrap_or(0),
            event_count: track.declared_event_count(),
            truncated: track.truncated,
            checksum: track.checksum_hex(),
            events: track.events.clone(),
        }
    }

    /// The single acceptance path. It rebuilds the track through the same
    /// validator that governs live ingest and then re-derives and compares the
    /// canonical checksum, so neither an ill-formed stream nor a mismatched
    /// digest can be accepted.
    pub fn decode(&self) -> Result<GestureTrack, GestureError> {
        if self.version != GESTURE_ALGORITHM_VERSION {
            return Err(GestureError::UnsupportedVersion(self.version));
        }
        if self.events.len() > MAX_GESTURE_EVENTS {
            return Err(GestureError::TooManyEvents(self.events.len()));
        }
        if usize::try_from(self.event_count) != Ok(self.events.len()) {
            return Err(GestureError::EventCount {
                declared: self.event_count,
                observed: self.events.len(),
            });
        }
        if self.events.is_empty() && self.origin_tick != 0 {
            return Err(GestureError::OriginWithoutEvents(self.origin_tick));
        }
        let mut validator = GestureStreamValidator::default();
        for event in &self.events {
            validator.accept(event)?;
        }
        let track = GestureTrack {
            origin_tick: (!self.events.is_empty()).then_some(self.origin_tick),
            events: self.events.clone(),
            truncated: self.truncated,
            validator,
        };
        let computed = track.checksum_hex();
        if computed != self.checksum {
            return Err(GestureError::ChecksumMismatch {
                declared: self.checksum.clone(),
                computed,
            });
        }
        Ok(track)
    }

    pub fn validate(&self) -> Result<(), GestureError> {
        self.decode().map(|_| ())
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, GestureError> {
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| GestureError::Serialization(error.to_string()))?;
        validate_document_bytes(bytes.len())?;
        Ok(bytes)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, GestureError> {
        validate_document_bytes(bytes.len())?;
        serde_json::from_slice(bytes)
            .map_err(|error| GestureError::Serialization(error.to_string()))
    }
}

impl<'de> Deserialize<'de> for GestureTrackDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            version: u16,
            origin_tick: u64,
            event_count: u32,
            truncated: bool,
            checksum: String,
            events: BoundedGestureEvents,
        }

        let raw = Raw::deserialize(deserializer)?;
        let document = Self {
            version: raw.version,
            origin_tick: raw.origin_tick,
            event_count: raw.event_count,
            truncated: raw.truncated,
            checksum: raw.checksum,
            events: raw.events.0,
        };
        document.validate().map_err(de::Error::custom)?;
        Ok(document)
    }
}

fn validate_document_bytes(bytes: usize) -> Result<(), GestureError> {
    if bytes == 0 || bytes > MAX_GESTURE_SERIALIZED_BYTES {
        Err(GestureError::DocumentBytes(bytes))
    } else {
        Ok(())
    }
}

/// Surface that authored one gesture sample.
///
/// Provenance only. It never reaches the event bits, so the same logical
/// gesture drawn on a tablet, sent by a phone, played from a MIDI controller,
/// or received over OSC records byte-identical events and one identical
/// checksum. The origin decides only what the *host* does around the event —
/// which status text names it, and (in the manual-history law) whether a
/// completed stroke opens an undo entry at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GestureOrigin {
    NativePointer,
    Phone,
    Midi,
    Osc,
}

impl GestureOrigin {
    /// Every origin, in a fixed order, so a closure test can drive one logical
    /// gesture through all four without a hand-maintained second list.
    pub const ALL: [Self; 4] = [Self::NativePointer, Self::Phone, Self::Midi, Self::Osc];

    pub const fn key(self) -> &'static str {
        match self {
            Self::NativePointer => "native_pointer",
            Self::Phone => "phone",
            Self::Midi => "midi",
            Self::Osc => "osc",
        }
    }

    /// A controller protocol drives gestures as automation. It mirrors
    /// `AutomationOrigin::records_manual_history` returning false: a MIDI or
    /// OSC stroke is performance data and never opens a manual-history entry.
    /// The authored surfaces are the physical pointer and the phone panel.
    pub const fn is_automation(self) -> bool {
        match self {
            Self::NativePointer | Self::Phone => false,
            Self::Midi | Self::Osc => true,
        }
    }
}

impl fmt::Display for GestureOrigin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.key())
    }
}

/// One authored sample as an input surface reports it, before normalization.
///
/// Position and pressure are unit-space floats, direction is an unnormalized
/// vector in the same space. Nothing here is quantized yet: this is the only
/// float-valued gesture type, and `normalize_gesture_input` is the only thing
/// that consumes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawGestureSample {
    pub stroke: u8,
    pub phase: GesturePhase,
    pub mode: GestureMode,
    pub position: [f32; 2],
    pub pressure: f32,
    /// Unnormalized stroke direction. Magnitude is discarded; only the
    /// direction survives quantization.
    pub direction: [f32; 2],
}

impl RawGestureSample {
    /// A sample with full pressure and no direction, the neutral shape every
    /// ingress starts from before it fills in what its surface actually knows.
    pub const fn new(
        stroke: u8,
        phase: GesturePhase,
        mode: GestureMode,
        position: [f32; 2],
    ) -> Self {
        Self {
            stroke,
            phase,
            mode,
            position,
            pressure: 1.0,
            direction: [0.0, 0.0],
        }
    }

    pub const fn with_pressure(mut self, pressure: f32) -> Self {
        self.pressure = pressure;
        self
    }

    pub const fn with_direction(mut self, direction: [f32; 2]) -> Self {
        self.direction = direction;
        self
    }
}

/// Typed refusal vocabulary for the one normalized adapter.
///
/// Every variant carries its origin: an operator needs to know which surface
/// misbehaved. The origin reaches the *error* and never the accepted event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureIngestError {
    StrokeOutOfRange { origin: GestureOrigin, stroke: u8 },
    NonFinitePosition { origin: GestureOrigin },
    NonFinitePressure { origin: GestureOrigin },
}

impl fmt::Display for GestureIngestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StrokeOutOfRange { origin, stroke } => write!(
                formatter,
                "{origin} gesture stroke {stroke} is outside the {MAX_ACTIVE_STROKES}-stroke identity space"
            ),
            Self::NonFinitePosition { origin } => {
                write!(formatter, "{origin} gesture position is not finite")
            }
            Self::NonFinitePressure { origin } => {
                write!(formatter, "{origin} gesture pressure is not finite")
            }
        }
    }
}

impl std::error::Error for GestureIngestError {}

/// Reduce an authored direction vector to unit length without overflowing on a
/// hostile magnitude.
///
/// A non-finite component is treated as unset rather than rejected: zero is
/// already the defined "no direction" encoding and is inert, so a broken
/// direction can only fail to etch, never etch somewhere invented. The
/// magnitude is divided out before the hypotenuse so a pair of huge finite
/// components cannot square to infinity and lose a real direction.
fn unit_direction(direction: [f32; 2]) -> [f32; 2] {
    let x = if direction[0].is_finite() {
        direction[0]
    } else {
        0.0
    };
    let y = if direction[1].is_finite() {
        direction[1]
    } else {
        0.0
    };
    let scale = x.abs().max(y.abs());
    if scale <= 0.0 {
        return [0.0, 0.0];
    }
    let x = x / scale;
    let y = y / scale;
    let length = x.hypot(y);
    if length > 0.0 {
        [x / length, y / length]
    } else {
        [0.0, 0.0]
    }
}

/// The one normalized gesture adapter. There is no second path.
///
/// Every input surface — native pointer or tablet, phone panel, MIDI
/// controller, OSC peer — hands its raw sample here and receives the quantized
/// event, or a typed refusal. `origin` is provenance and is deliberately never
/// read while building the event: it reaches `GestureIngestError` and nothing
/// else, so all four surfaces produce byte-identical events for the same
/// logical gesture.
///
/// What is refused rather than repaired:
/// - a stroke outside the identity space, because clamping it would silently
///   merge two distinct strokes into one;
/// - a non-finite position or pressure, because inventing the canvas origin
///   out of `NaN` would etch a mark the operator never made.
///
/// What is sanitized: the direction vector, which is reduced to unit length
/// and falls back to the inert zero direction. `quantize_unit`'s own
/// neutral-zero fallback stays the last line of defence for any other
/// construction path; this adapter simply never relies on it for a position.
///
/// `tick` is the 30 Hz reference address the caller has already derived. When
/// recording is armed the track re-stamps it relative to the track origin —
/// addressing a *recorded* event is always the track's authority. When
/// recording is not armed the stamped address is all the event ever has, and
/// the event is session-local: it is never implied replayable.
pub fn normalize_gesture_input(
    origin: GestureOrigin,
    raw: RawGestureSample,
    tick: u32,
) -> Result<GestureEvent, GestureIngestError> {
    if usize::from(raw.stroke) >= MAX_ACTIVE_STROKES {
        return Err(GestureIngestError::StrokeOutOfRange {
            origin,
            stroke: raw.stroke,
        });
    }
    if !raw.position[0].is_finite() || !raw.position[1].is_finite() {
        return Err(GestureIngestError::NonFinitePosition { origin });
    }
    if !raw.pressure.is_finite() {
        return Err(GestureIngestError::NonFinitePressure { origin });
    }
    Ok(GestureEvent::quantized(
        raw.stroke,
        raw.phase,
        raw.mode,
        raw.position,
        raw.pressure,
        unit_direction(raw.direction),
    )
    .at(tick))
}

/// One continuous-controller update from a protocol that has no notion of a
/// stroke.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GestureControlInput {
    X(f32),
    Y(f32),
    Pressure(f32),
    /// Contact gate. A rising edge opens the stroke, a falling edge closes it.
    Contact(bool),
}

/// Continuous-controller gesture surface for MIDI and OSC.
///
/// Those protocols deliver independent scalars, never stroke events, so the
/// stroke machine lives here: a rising contact gate emits `Begin`, a falling
/// gate emits `End`, and `sample_motion` emits at most one `Move` per frame
/// carrying the direction actually travelled since the last emitted sample.
/// One `Move` per frame is exactly one 30 Hz reference address, so a controller
/// sweeping at any physical rate records the same addresses as a pointer.
///
/// The surface holds no clock: the host decides when a frame happened.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GestureControlSurface {
    stroke: u8,
    mode: GestureMode,
    position: [f32; 2],
    pressure: f32,
    emitted: Option<[f32; 2]>,
}

impl Default for GestureControlSurface {
    fn default() -> Self {
        Self {
            stroke: 0,
            mode: GestureMode::Push,
            position: [0.5, 0.5],
            pressure: 1.0,
            emitted: None,
        }
    }
}

impl GestureControlSurface {
    /// Address this surface's strokes to one identity so two protocols cannot
    /// interleave into a single stroke.
    pub const fn with_stroke(mut self, stroke: u8) -> Self {
        self.stroke = stroke;
        self
    }

    pub const fn is_open(&self) -> bool {
        self.emitted.is_some()
    }

    pub const fn mode(&self) -> GestureMode {
        self.mode
    }

    pub const fn set_mode(&mut self, mode: GestureMode) {
        self.mode = mode;
    }

    /// Apply one scalar update. Only a contact edge produces a sample; an axis
    /// or pressure update is stored and observed by the next `sample_motion`,
    /// so a controller sending X and Y separately still authors one point.
    pub fn apply(&mut self, input: GestureControlInput) -> Option<RawGestureSample> {
        match input {
            GestureControlInput::X(value) => {
                self.position[0] = value;
                None
            }
            GestureControlInput::Y(value) => {
                self.position[1] = value;
                None
            }
            GestureControlInput::Pressure(value) => {
                self.pressure = value;
                None
            }
            GestureControlInput::Contact(true) => {
                if self.emitted.is_some() {
                    return None;
                }
                self.emitted = Some(self.position);
                Some(self.sample(GesturePhase::Begin, [0.0, 0.0]))
            }
            GestureControlInput::Contact(false) => {
                let previous = self.emitted.take()?;
                let direction = [
                    self.position[0] - previous[0],
                    self.position[1] - previous[1],
                ];
                Some(self.sample(GesturePhase::End, direction))
            }
        }
    }

    /// Emit this frame's `Move`, if the surface is open and actually moved.
    /// A stationary open surface emits nothing rather than a zero-direction
    /// point, so holding a joystick still does not fill the bounded track.
    pub fn sample_motion(&mut self) -> Option<RawGestureSample> {
        let previous = self.emitted?;
        let direction = [
            self.position[0] - previous[0],
            self.position[1] - previous[1],
        ];
        if direction[0] == 0.0 && direction[1] == 0.0 {
            return None;
        }
        self.emitted = Some(self.position);
        Some(self.sample(GesturePhase::Move, direction))
    }

    /// Release an open stroke without a controller edge — a stalled or vanished
    /// protocol stream must not leave a stroke open forever.
    pub fn release(&mut self) -> Option<RawGestureSample> {
        self.apply(GestureControlInput::Contact(false))
    }

    fn sample(&self, phase: GesturePhase, direction: [f32; 2]) -> RawGestureSample {
        RawGestureSample {
            stroke: self.stroke,
            phase,
            mode: self.mode,
            position: self.position,
            pressure: self.pressure,
            direction,
        }
    }
}

/// What one accepted frame's recording attempt did.
///
/// Counts, not booleans, so several samples inside one frame each report their
/// own fate. Every field saturates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GestureRecordOutcome {
    pub recorded: u32,
    /// Well-formed samples refused because the track is already at
    /// `MAX_GESTURE_EVENTS`.
    pub truncated: u32,
    /// Ill-formed samples rejected rather than repaired.
    pub rejected: u32,
}

impl GestureRecordOutcome {
    pub const fn is_empty(self) -> bool {
        self.recorded == 0 && self.truncated == 0 && self.rejected == 0
    }
}

/// Accepted-frame clock for live gesture recording.
///
/// Rejected frames do not advance it and Program Freeze does not call it, so it
/// never accumulates catch-up debt for time the audience never saw. Its only
/// output is a 30 Hz integer address. This is `TemporalEventRecorder`'s exact
/// shape: the tick is derived *before* this frame's delta is added, so the
/// first accepted frame records at tick 0.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GestureEventRecorder {
    track: GestureTrack,
    accepted_seconds: f64,
}

impl GestureEventRecorder {
    /// The 30 Hz address the next accepted frame will occupy.
    pub fn reference_tick(&self) -> u64 {
        (self.accepted_seconds * f64::from(GESTURE_REFERENCE_FPS))
            .round()
            .clamp(0.0, u64::MAX as f64) as u64
    }

    /// The same address narrowed for `normalize_gesture_input`, which stamps a
    /// relative 32-bit address. A session long enough to saturate this holds
    /// the last representable address rather than wrapping into the past.
    pub fn reference_tick_u32(&self) -> u32 {
        u32::try_from(self.reference_tick()).unwrap_or(u32::MAX)
    }

    /// Record one accepted, program-advancing frame's samples at that frame's
    /// address, then advance the accepted clock.
    pub fn record_accepted(
        &mut self,
        delta_seconds: f32,
        events: &[GestureEvent],
    ) -> GestureRecordOutcome {
        let tick = self.reference_tick();
        let mut outcome = GestureRecordOutcome::default();
        for event in events {
            match self.track.record_accepted(tick, *event) {
                Ok(true) => outcome.recorded = outcome.recorded.saturating_add(1),
                Ok(false) => outcome.truncated = outcome.truncated.saturating_add(1),
                Err(_) => outcome.rejected = outcome.rejected.saturating_add(1),
            }
        }
        self.accepted_seconds += f64::from(crate::temporal::sanitize_delta(delta_seconds));
        outcome
    }

    pub const fn track(&self) -> &GestureTrack {
        &self.track
    }

    /// Install a track recovered from a patch and resume this session's clock
    /// at that track's last recorded address.
    ///
    /// The relative event stream and the absolute origin are both preserved
    /// exactly, so a restored track's canonical checksum is byte-identical to
    /// the one that was saved. Resuming at the last recorded address rather
    /// than at zero is what keeps a later live event monotonic *without*
    /// rebasing: a rebase would silently rewrite the hashed origin of a
    /// recording the operator already holds.
    pub fn restore(&mut self, track: GestureTrack) {
        self.accepted_seconds =
            track.last_absolute_tick() as f64 / f64::from(GESTURE_REFERENCE_FPS);
        self.track = track;
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    use serde::de::IntoDeserializer;

    fn begin(stroke: u8) -> GestureEvent {
        GestureEvent::quantized(
            stroke,
            GesturePhase::Begin,
            GestureMode::Push,
            [0.25, 0.75],
            1.0,
            [1.0, 0.0],
        )
    }

    fn moved(stroke: u8) -> GestureEvent {
        GestureEvent::quantized(
            stroke,
            GesturePhase::Move,
            GestureMode::Push,
            [0.5, 0.5],
            0.5,
            [0.0, 1.0],
        )
    }

    fn end(stroke: u8) -> GestureEvent {
        GestureEvent::quantized(
            stroke,
            GesturePhase::End,
            GestureMode::Curl,
            [0.75, 0.25],
            0.0,
            [-1.0, 0.0],
        )
    }

    /// One logical stroke: absolute tick offsets paired with their events.
    fn authored_stroke() -> Vec<(u64, GestureEvent)> {
        vec![
            (0, begin(0)),
            (0, moved(0)),
            (1, moved(0)),
            (7, moved(0)),
            (7, end(0)),
        ]
    }

    fn record_all(base_tick: u64, authored: &[(u64, GestureEvent)]) -> GestureTrack {
        let mut track = GestureTrack::default();
        for (offset, event) in authored {
            assert_eq!(track.record_accepted(base_tick + offset, *event), Ok(true));
        }
        track
    }

    /// Build a track directly from an event list, deliberately bypassing ingest
    /// so an encoding assertion isolates the field stream from the flag word.
    fn encoded_track(events: Vec<GestureEvent>) -> GestureTrack {
        GestureTrack {
            origin_tick: (!events.is_empty()).then_some(0),
            events,
            truncated: false,
            validator: GestureStreamValidator::default(),
        }
    }

    #[test]
    fn gesture_reference_rate_is_the_temporal_reference_and_never_a_second_literal() {
        assert_eq!(GESTURE_REFERENCE_FPS, TEMPORAL_REFERENCE_FPS);
        let source = include_str!("gesture.rs");
        assert!(source.contains("pub const GESTURE_REFERENCE_FPS: f32 = TEMPORAL_REFERENCE_FPS;"));
        // Assembled at compile time so the needle itself never appears in the
        // source text this assertion scans.
        let forbidden = concat!("30", ".0");
        assert!(
            !source.contains(forbidden),
            "gesture.rs must never redefine the 30 Hz reference as its own literal"
        );
    }

    #[test]
    fn frozen_gesture_bounds_and_encoding_widths_match_the_specification() {
        assert_eq!(GESTURE_ALGORITHM_VERSION, 1);
        assert_eq!(MAX_GESTURE_EVENTS, 4_096);
        assert_eq!(MAX_GESTURE_SERIALIZED_BYTES, 256 * 1024);
        assert_eq!(MAX_ACTIVE_STROKES, 16);
        assert_eq!(MAX_GESTURE_DECAY_TICKS, 4_096);
        assert_eq!(
            GESTURE_CHECKSUM_DOMAIN,
            b"collide-o-scope/gesture-track/v1\0"
        );
        assert_eq!(GESTURE_CHECKSUM_DOMAIN.len(), 33);
        assert_eq!(GESTURE_EVENT_ENCODED_BYTES, 18);
        assert_eq!(GESTURE_CHECKSUM_HEADER_BYTES, 49);
        assert_eq!(GESTURE_FLAG_TRUNCATED, 1);
        assert_eq!(GESTURE_FLAG_INCOMPLETE, 2);

        let track = record_all(0, &authored_stroke());
        assert_eq!(
            track.canonical_bytes().len(),
            GESTURE_CHECKSUM_HEADER_BYTES + 5 * GESTURE_EVENT_ENCODED_BYTES
        );
    }

    #[test]
    fn q16_and_q15_round_trip_exactly_at_every_representable_lattice_point() {
        for raw in 0..=u16::MAX {
            assert_eq!(quantize_unit(dequantize_unit(raw)), raw, "Q16 code {raw}");
        }
        for raw in -32_767_i16..=32_767 {
            assert_eq!(
                quantize_signed(dequantize_signed(raw)),
                raw,
                "Q15 code {raw}"
            );
        }
        // i16::MIN is unreachable by quantization but a hostile document may
        // declare it; it decodes to the clamped extreme, never past it.
        assert_eq!(dequantize_signed(i16::MIN), -1.0);
        assert_eq!(quantize_signed(dequantize_signed(i16::MIN)), -32_767);
    }

    #[test]
    fn quantization_pins_the_extremes_and_sanitizes_hostile_authored_values() {
        assert_eq!(quantize_unit(0.0), 0);
        assert_eq!(quantize_unit(1.0), u16::MAX);
        assert_eq!(dequantize_unit(0), 0.0);
        assert_eq!(dequantize_unit(u16::MAX), 1.0);
        assert_eq!(quantize_signed(-1.0), -32_767);
        assert_eq!(quantize_signed(0.0), 0);
        assert_eq!(quantize_signed(1.0), 32_767);
        assert_eq!(dequantize_signed(0), 0.0);

        assert_eq!(quantize_unit(2.5), u16::MAX);
        assert_eq!(quantize_unit(-2.5), 0);
        assert_eq!(quantize_signed(5.0), 32_767);
        assert_eq!(quantize_signed(-5.0), -32_767);

        for hostile in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(quantize_unit(hostile), 0);
            assert_eq!(quantize_signed(hostile), 0);
        }

        let event = GestureEvent::quantized(
            3,
            GesturePhase::Move,
            GestureMode::Curl,
            [f32::NAN, 4.0],
            f32::NEG_INFINITY,
            [f32::INFINITY, -9.0],
        );
        assert_eq!(event.x, 0);
        assert_eq!(event.y, u16::MAX);
        assert_eq!(event.pressure, 0);
        assert_eq!(event.direction_x, 0);
        assert_eq!(event.direction_y, -32_767);
    }

    #[test]
    fn decoded_direction_is_renormalized_and_an_unset_direction_stays_exactly_zero() {
        let diagonal = GestureEvent::quantized(
            0,
            GesturePhase::Move,
            GestureMode::Push,
            [0.5, 0.5],
            1.0,
            [1.0, 1.0],
        );
        let [x, y] = diagonal.direction();
        assert!((x - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5, "{x}");
        assert!((y - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5, "{y}");
        assert!((x * x + y * y - 1.0).abs() < 1e-5);

        let short = GestureEvent {
            direction_x: 1,
            direction_y: 0,
            ..diagonal
        };
        assert_eq!(short.direction(), [1.0, 0.0]);

        let unset = GestureEvent {
            direction_x: 0,
            direction_y: 0,
            ..diagonal
        };
        assert_eq!(unset.direction(), [0.0, 0.0]);

        // The Q16 lattice has no exact 0.5, so position decodes to the nearest
        // representable code and re-quantizes to the same bits.
        assert_eq!(
            diagonal.position(),
            [dequantize_unit(diagonal.x), dequantize_unit(diagonal.y)]
        );
        assert_eq!(quantize_unit(diagonal.position()[1]), diagonal.y);
        assert_eq!(diagonal.pressure(), 1.0);
    }

    #[test]
    fn phase_and_mode_codes_are_append_only_and_reject_unknown_wire_codes() {
        assert_eq!(GesturePhase::Begin.code(), 0);
        assert_eq!(GesturePhase::Move.code(), 1);
        assert_eq!(GesturePhase::End.code(), 2);
        assert_eq!(GestureMode::Push.code(), 0);
        assert_eq!(GestureMode::Curl.code(), 1);

        for phase in [GesturePhase::Begin, GesturePhase::Move, GesturePhase::End] {
            assert_eq!(GesturePhase::from_code(phase.code()), Some(phase));
        }
        for mode in [GestureMode::Push, GestureMode::Curl] {
            assert_eq!(GestureMode::from_code(mode.code()), Some(mode));
        }
        assert_eq!(GesturePhase::from_code(3), None);
        assert_eq!(GestureMode::from_code(2), None);

        assert!(serde_json::from_str::<GesturePhase>("\"hover\"").is_err());
        assert!(serde_json::from_str::<GestureMode>("\"smear\"").is_err());
        assert_eq!(
            serde_json::to_string(&GesturePhase::Move).unwrap(),
            "\"move\""
        );
        assert_eq!(
            serde_json::to_string(&GestureMode::Curl).unwrap(),
            "\"curl\""
        );
    }

    #[test]
    fn gesture_track_rebases_its_origin_and_keeps_relative_ticks_non_decreasing() {
        let track = record_all(1_000, &authored_stroke());
        assert_eq!(track.origin_tick(), Some(1_000));
        assert_eq!(
            track
                .events()
                .iter()
                .map(|event| event.reference_tick)
                .collect::<Vec<_>>(),
            vec![0, 0, 1, 7, 7]
        );
        assert!(track.is_complete());
        assert!(!track.truncated());

        let mut backwards = GestureTrack::default();
        assert_eq!(backwards.record_accepted(100, begin(0)), Ok(true));
        assert_eq!(
            backwards.record_accepted(99, moved(0)),
            Err(GestureError::TickBeforeOrigin {
                absolute: 99,
                origin: 100
            })
        );
        assert_eq!(backwards.events().len(), 1);

        // A rejected event leaves the open-stroke set and the tick watermark
        // exactly as they were.
        assert_eq!(backwards.record_accepted(105, moved(0)), Ok(true));
        assert_eq!(backwards.events()[1].reference_tick, 5);
        assert_eq!(
            backwards.record_accepted(103, moved(0)),
            Err(GestureError::NonMonotonicTick {
                previous: 5,
                observed: 3
            })
        );
        assert_eq!(backwards.events().len(), 2);
    }

    #[test]
    fn a_relative_tick_beyond_the_thirty_two_bit_window_is_rejected_rather_than_wrapped() {
        let mut track = GestureTrack::default();
        assert_eq!(track.record_accepted(0, begin(0)), Ok(true));
        let beyond = u64::from(u32::MAX) + 1;
        assert_eq!(
            track.record_accepted(beyond, moved(0)),
            Err(GestureError::TickOutOfRange { relative: beyond })
        );
        assert_eq!(track.events().len(), 1);
        assert_eq!(
            track.record_accepted(u64::from(u32::MAX), moved(0)),
            Ok(true)
        );
        assert_eq!(track.events()[1].reference_tick, u32::MAX);
    }

    #[test]
    fn gesture_track_returns_false_at_the_four_thousand_ninety_sixth_event_and_flags_truncation() {
        let mut track = GestureTrack::default();
        assert_eq!(track.record_accepted(0, begin(0)), Ok(true));
        for tick in 1..MAX_GESTURE_EVENTS as u64 {
            assert_eq!(track.record_accepted(tick, moved(0)), Ok(true));
        }
        assert_eq!(track.events().len(), MAX_GESTURE_EVENTS);
        assert!(!track.truncated());

        let over_cap = track.clone();
        let mut over_cap = over_cap;
        assert_eq!(
            over_cap.record_accepted(MAX_GESTURE_EVENTS as u64, moved(0)),
            Ok(false)
        );
        assert!(over_cap.truncated());
        assert_eq!(over_cap.events().len(), MAX_GESTURE_EVENTS);
        assert_eq!(over_cap.events(), track.events());
        assert_eq!(
            over_cap.flags(),
            GESTURE_FLAG_TRUNCATED | GESTURE_FLAG_INCOMPLETE
        );

        // A hostile event at the cap is still rejected as ill-formed rather
        // than reported as ordinary truncation.
        let mut hostile = track.clone();
        assert_eq!(
            hostile.record_accepted(MAX_GESTURE_EVENTS as u64, begin(0)),
            Err(GestureError::StrokeAlreadyOpen { stroke: 0 })
        );
        assert!(!hostile.truncated());

        over_cap.clear();
        assert!(over_cap.events().is_empty());
        assert!(!over_cap.truncated());
        assert_eq!(over_cap.origin_tick(), None);
        assert!(over_cap.is_complete());
    }

    #[test]
    fn orphan_move_and_end_events_are_rejected_rather_than_repaired() {
        let mut track = GestureTrack::default();
        assert_eq!(
            track.record_accepted(0, moved(2)),
            Err(GestureError::StrokeNotOpen {
                stroke: 2,
                phase: GesturePhase::Move
            })
        );
        assert_eq!(
            track.record_accepted(0, end(2)),
            Err(GestureError::StrokeNotOpen {
                stroke: 2,
                phase: GesturePhase::End
            })
        );
        assert!(track.events().is_empty());
        assert_eq!(track.origin_tick(), None);

        assert_eq!(track.record_accepted(0, begin(2)), Ok(true));
        assert_eq!(track.record_accepted(1, end(2)), Ok(true));
        assert_eq!(
            track.record_accepted(2, moved(2)),
            Err(GestureError::StrokeNotOpen {
                stroke: 2,
                phase: GesturePhase::Move
            })
        );
        assert_eq!(track.events().len(), 2);
    }

    #[test]
    fn a_second_begin_for_an_open_stroke_is_rejected_and_leaves_the_track_unchanged() {
        let mut track = GestureTrack::default();
        assert_eq!(track.record_accepted(0, begin(5)), Ok(true));
        let before = track.clone();
        assert_eq!(
            track.record_accepted(1, begin(5)),
            Err(GestureError::StrokeAlreadyOpen { stroke: 5 })
        );
        assert_eq!(track, before);
        assert_eq!(track.open_stroke_count(), 1);
    }

    #[test]
    fn sixteen_strokes_may_be_open_at_once_and_a_seventeenth_identity_is_out_of_range() {
        let mut track = GestureTrack::default();
        for stroke in 0..MAX_ACTIVE_STROKES as u8 {
            assert_eq!(track.record_accepted(0, begin(stroke)), Ok(true));
        }
        assert_eq!(track.open_stroke_count(), MAX_ACTIVE_STROKES as u32);
        assert!(!track.is_complete());

        let stroke = MAX_ACTIVE_STROKES as u8;
        assert_eq!(
            track.record_accepted(0, begin(stroke)),
            Err(GestureError::StrokeOutOfRange { stroke })
        );
        assert_eq!(
            track.record_accepted(0, moved(u8::MAX)),
            Err(GestureError::StrokeOutOfRange { stroke: u8::MAX })
        );
        assert_eq!(track.events().len(), MAX_ACTIVE_STROKES);

        for stroke in 0..MAX_ACTIVE_STROKES as u8 {
            assert_eq!(track.record_accepted(1, end(stroke)), Ok(true));
        }
        assert_eq!(track.open_stroke_count(), 0);
        assert!(track.is_complete());
    }

    #[test]
    fn a_track_with_unclosed_strokes_is_valid_explicitly_incomplete_and_never_auto_closed() {
        let mut track = GestureTrack::default();
        assert_eq!(track.record_accepted(0, begin(1)), Ok(true));
        assert_eq!(track.record_accepted(3, moved(1)), Ok(true));
        assert!(!track.is_complete());
        assert_eq!(track.open_stroke_count(), 1);
        assert_eq!(track.flags(), GESTURE_FLAG_INCOMPLETE);
        assert_eq!(track.events().len(), 2);
        assert_eq!(track.events()[1].phase, GesturePhase::Move);

        let document = GestureTrackDocument::capture(&track);
        let decoded = document.decode().expect("incomplete tracks are valid");
        assert_eq!(decoded, track);
        assert!(!decoded.is_complete());
        assert_eq!(decoded.events().len(), 2);
        assert!(decoded
            .events()
            .iter()
            .all(|event| event.phase != GesturePhase::End));
    }

    #[test]
    fn ingest_and_decode_share_one_validator_so_a_hand_built_document_cannot_bypass_it() {
        let track = record_all(0, &authored_stroke());
        let mut orphaned = track.events().to_vec();
        orphaned.remove(0);

        // Re-derive the checksum for the hostile stream so the validator, not
        // a stale digest, is what refuses it.
        let hostile = encoded_track(orphaned.clone());
        let document = GestureTrackDocument {
            version: GESTURE_ALGORITHM_VERSION,
            origin_tick: 0,
            event_count: orphaned.len() as u32,
            truncated: false,
            checksum: hostile.checksum_hex(),
            events: orphaned,
        };
        assert_eq!(
            document.decode(),
            Err(GestureError::StrokeNotOpen {
                stroke: 0,
                phase: GesturePhase::Move
            })
        );

        // The same rejection arrives through serde, before any caller sees a
        // document value at all.
        let json = serde_json::to_vec(&document).unwrap();
        let error = GestureTrackDocument::from_json_bytes(&json).unwrap_err();
        assert!(
            matches!(&error, GestureError::Serialization(message) if message.contains("no open Begin")),
            "{error}"
        );

        let mut non_monotonic = track.events().to_vec();
        non_monotonic[2].reference_tick = 0;
        non_monotonic[3].reference_tick = 0;
        non_monotonic[4].reference_tick = 0;
        non_monotonic[1].reference_tick = 5;
        let rewound = encoded_track(non_monotonic.clone());
        let document = GestureTrackDocument {
            version: GESTURE_ALGORITHM_VERSION,
            origin_tick: 0,
            event_count: non_monotonic.len() as u32,
            truncated: false,
            checksum: rewound.checksum_hex(),
            events: non_monotonic,
        };
        assert_eq!(
            document.decode(),
            Err(GestureError::NonMonotonicTick {
                previous: 5,
                observed: 0
            })
        );
    }

    #[test]
    fn canonical_gesture_checksum_pins_a_domain_separated_little_endian_field_stream() {
        let empty = GestureTrack::default();
        let bytes = empty.canonical_bytes();
        assert_eq!(bytes.len(), GESTURE_CHECKSUM_HEADER_BYTES);
        assert_eq!(
            &bytes[..GESTURE_CHECKSUM_DOMAIN.len()],
            GESTURE_CHECKSUM_DOMAIN
        );
        assert_eq!(&bytes[33..35], &1_u16.to_le_bytes());
        assert_eq!(&bytes[35..37], &0_u16.to_le_bytes());
        assert_eq!(&bytes[37..45], &0_u64.to_le_bytes());
        assert_eq!(&bytes[45..49], &0_u32.to_le_bytes());

        let mut track = GestureTrack::default();
        assert_eq!(track.record_accepted(9, begin(2)), Ok(true));
        assert_eq!(track.record_accepted(12, end(2)), Ok(true));
        let bytes = track.canonical_bytes();
        assert_eq!(
            bytes.len(),
            GESTURE_CHECKSUM_HEADER_BYTES + 2 * GESTURE_EVENT_ENCODED_BYTES
        );
        assert_eq!(&bytes[37..45], &9_u64.to_le_bytes());
        assert_eq!(&bytes[45..49], &2_u32.to_le_bytes());
        let first = &bytes[49..67];
        assert_eq!(&first[0..4], &0_u32.to_le_bytes());
        assert_eq!(first[4], 2);
        assert_eq!(first[5], GesturePhase::Begin.code());
        assert_eq!(first[6], GestureMode::Push.code());
        assert_eq!(first[7], GESTURE_EVENT_RESERVED);
        assert_eq!(&first[8..10], &quantize_unit(0.25).to_le_bytes());
        assert_eq!(&first[10..12], &quantize_unit(0.75).to_le_bytes());
        assert_eq!(&first[12..14], &u16::MAX.to_le_bytes());
        assert_eq!(&first[14..16], &32_767_i16.to_le_bytes());
        assert_eq!(&first[16..18], &0_i16.to_le_bytes());
        let second = &bytes[67..85];
        assert_eq!(&second[0..4], &3_u32.to_le_bytes());
        assert_eq!(second[5], GesturePhase::End.code());
        assert_eq!(second[6], GestureMode::Curl.code());

        assert_eq!(
            track.checksum_hex(),
            "735aeb178d703aa828330c5b8106874cbde810f286aef8718433bd188a7044b4"
        );
        assert_eq!(track.checksum_hex().len(), 64);
        assert_eq!(track.checksum(), <[u8; 32]>::from(Sha256::digest(bytes)));
    }

    #[test]
    fn canonical_checksum_and_replay_are_invariant_to_frame_grouping_and_drain_rate() {
        let authored = authored_stroke();
        let reference = record_all(1_000, &authored);

        // Frame grouping is invisible: the tick address, not the call, is an
        // event's identity. Handing the host's samples over one per frame, in
        // pairs, in threes, or all in one frame yields byte-identical tracks.
        for grouping in [1_usize, 2, 3, authored.len()] {
            let mut track = GestureTrack::default();
            for chunk in authored.chunks(grouping) {
                for (offset, event) in chunk {
                    assert_eq!(track.record_accepted(1_000 + offset, *event), Ok(true));
                }
            }
            assert_eq!(track, reference, "grouping {grouping}");
            assert_eq!(
                track.canonical_bytes(),
                reference.canonical_bytes(),
                "grouping {grouping}"
            );
            assert_eq!(track.checksum_hex(), reference.checksum_hex());
        }

        // Origin rebasing keeps the event stream portable across program
        // positions, while the absolute origin stays a declared hashed field
        // rather than a hidden one.
        for base in [0_u64, 1_000, 4_000, 9_876_543_210] {
            let other = record_all(base, &authored);
            assert_eq!(other.events(), reference.events(), "origin {base}");
            assert_eq!(other.origin_tick(), Some(base));
            assert_eq!(other.flags(), reference.flags());
            assert_eq!(
                &other.canonical_bytes()[GESTURE_CHECKSUM_HEADER_BYTES..],
                &reference.canonical_bytes()[GESTURE_CHECKSUM_HEADER_BYTES..],
                "origin {base}"
            );
            assert_eq!(
                base == 1_000,
                other.checksum_hex() == reference.checksum_hex(),
                "origin {base}"
            );
        }

        // Draining one tick at a time and draining once at the end return the
        // same events in the same recorded order.
        let mut fine = reference.replay();
        let mut fine_events = Vec::new();
        for tick in 0..=7 {
            fine_events.extend_from_slice(fine.events_due(tick));
        }
        let mut coarse = reference.replay();
        assert_eq!(coarse.events_due(u32::MAX), reference.events());
        assert_eq!(fine_events, reference.events());
        assert!(fine.events_due(u32::MAX).is_empty());
        assert!(coarse.events_due(u32::MAX).is_empty());
    }

    #[test]
    fn the_checksum_covers_every_encoded_field_and_both_recording_flags() {
        let base = vec![begin(0).at(4), end(0).at(9)];
        let baseline = encoded_track(base.clone()).checksum_hex();

        let mutations: Vec<(&str, Vec<GestureEvent>)> = vec![
            ("tick", {
                let mut events = base.clone();
                events[1].reference_tick = 10;
                events
            }),
            ("stroke", {
                let mut events = base.clone();
                events[0].stroke = 1;
                events
            }),
            ("phase", {
                let mut events = base.clone();
                events[1].phase = GesturePhase::Move;
                events
            }),
            ("mode", {
                let mut events = base.clone();
                events[0].mode = GestureMode::Curl;
                events
            }),
            ("x", {
                let mut events = base.clone();
                events[0].x = events[0].x.wrapping_add(1);
                events
            }),
            ("y", {
                let mut events = base.clone();
                events[0].y = events[0].y.wrapping_add(1);
                events
            }),
            ("pressure", {
                let mut events = base.clone();
                events[0].pressure = events[0].pressure.wrapping_sub(1);
                events
            }),
            ("direction_x", {
                let mut events = base.clone();
                events[0].direction_x -= 1;
                events
            }),
            ("direction_y", {
                let mut events = base.clone();
                events[0].direction_y += 1;
                events
            }),
            ("order", vec![base[1], base[0]]),
            ("count", vec![base[0]]),
        ];
        for (field, events) in mutations {
            assert_ne!(
                encoded_track(events).checksum_hex(),
                baseline,
                "{field} must move the canonical checksum"
            );
        }

        // The two recording flags are inside the digest, so an incomplete or
        // truncated recording can never share a complete one's checksum.
        let mut incomplete = GestureTrack::default();
        assert_eq!(incomplete.record_accepted(0, begin(0)), Ok(true));
        assert_eq!(incomplete.flags(), GESTURE_FLAG_INCOMPLETE);
        assert_ne!(
            incomplete.checksum_hex(),
            encoded_track(vec![begin(0)]).checksum_hex()
        );

        let mut truncated = encoded_track(vec![begin(0)]);
        let complete_digest = truncated.checksum_hex();
        truncated.truncated = true;
        assert_ne!(truncated.checksum_hex(), complete_digest);
    }

    #[test]
    fn gesture_track_document_round_trips_the_recorded_stream_and_its_checksum() {
        let track = record_all(2_048, &authored_stroke());
        let document = GestureTrackDocument::capture(&track);
        assert_eq!(document.version, GESTURE_ALGORITHM_VERSION);
        assert_eq!(document.origin_tick, 2_048);
        assert_eq!(document.event_count, 5);
        assert!(!document.truncated);
        assert_eq!(document.checksum, track.checksum_hex());

        let bytes = document
            .to_json_bytes()
            .expect("document is within its cap");
        let restored = GestureTrackDocument::from_json_bytes(&bytes).expect("round trip");
        assert_eq!(restored, document);
        let decoded = restored.decode().expect("round trip decodes");
        assert_eq!(decoded, track);
        assert_eq!(decoded.checksum_hex(), track.checksum_hex());
        assert_eq!(decoded.origin_tick(), Some(2_048));

        // The document carries no operational path or filesystem metadata.
        let text = String::from_utf8(bytes).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        let mut keys = parsed
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "checksum",
                "event_count",
                "events",
                "origin_tick",
                "truncated",
                "version"
            ]
        );

        let empty = GestureTrackDocument::capture(&GestureTrack::default());
        assert_eq!(empty.event_count, 0);
        assert_eq!(empty.origin_tick, 0);
        assert_eq!(empty.decode(), Ok(GestureTrack::default()));
    }

    #[test]
    fn unknown_fields_versions_counts_and_checksum_mismatches_fail_closed() {
        let track = record_all(0, &authored_stroke());
        let document = GestureTrackDocument::capture(&track);

        let mut unknown = serde_json::to_value(&document).unwrap();
        unknown["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<GestureTrackDocument>(unknown).is_err());

        let mut unknown_event = serde_json::to_value(&document).unwrap();
        unknown_event["events"][0]["unexpected"] = serde_json::json!(1);
        assert!(serde_json::from_value::<GestureTrackDocument>(unknown_event).is_err());

        let mut missing = serde_json::to_value(&document).unwrap();
        missing["events"][0].as_object_mut().unwrap().remove("mode");
        assert!(serde_json::from_value::<GestureTrackDocument>(missing).is_err());

        let mut wrong_version = document.clone();
        wrong_version.version = 2;
        assert_eq!(
            wrong_version.decode(),
            Err(GestureError::UnsupportedVersion(2))
        );

        let mut wrong_count = document.clone();
        wrong_count.event_count = 4;
        assert_eq!(
            wrong_count.decode(),
            Err(GestureError::EventCount {
                declared: 4,
                observed: 5
            })
        );

        let mut wrong_origin = GestureTrackDocument::capture(&GestureTrack::default());
        wrong_origin.origin_tick = 7;
        assert_eq!(
            wrong_origin.decode(),
            Err(GestureError::OriginWithoutEvents(7))
        );

        let mut wrong_checksum = document.clone();
        wrong_checksum.checksum = "0".repeat(64);
        assert!(matches!(
            wrong_checksum.decode(),
            Err(GestureError::ChecksumMismatch { .. })
        ));

        // A silently edited event is caught even though the stream stays
        // well-formed: the digest is re-derived, never trusted.
        let mut edited = document;
        edited.events[2].x = edited.events[2].x.wrapping_add(1);
        assert!(matches!(
            edited.decode(),
            Err(GestureError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn document_byte_caps_are_enforced_on_both_encode_and_decode() {
        assert_eq!(
            GestureTrackDocument::from_json_bytes(&[]),
            Err(GestureError::DocumentBytes(0))
        );
        let oversized = vec![b' '; MAX_GESTURE_SERIALIZED_BYTES + 1];
        assert_eq!(
            GestureTrackDocument::from_json_bytes(&oversized),
            Err(GestureError::DocumentBytes(
                MAX_GESTURE_SERIALIZED_BYTES + 1
            ))
        );

        // A full-cap track serializes past the byte ceiling, and encode refuses
        // it with the same typed error rather than emitting an over-cap file.
        let mut track = GestureTrack::default();
        assert_eq!(track.record_accepted(0, begin(0)), Ok(true));
        for tick in 1..MAX_GESTURE_EVENTS as u64 {
            assert_eq!(track.record_accepted(tick, moved(0)), Ok(true));
        }
        let document = GestureTrackDocument::capture(&track);
        assert!(matches!(
            document.to_json_bytes(),
            Err(GestureError::DocumentBytes(_))
        ));
    }

    /// A `SeqAccess` that advertises an enormous length and never ends. A
    /// visitor that sized its allocation from the hint, or that drained before
    /// checking, could not survive it.
    struct EventBomb<'a> {
        template: serde_json::Value,
        produced: &'a Cell<usize>,
    }

    impl<'de> de::SeqAccess<'de> for EventBomb<'_> {
        type Error = serde_json::Error;

        fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
        where
            T: de::DeserializeSeed<'de>,
        {
            self.produced.set(self.produced.get() + 1);
            seed.deserialize(self.template.clone().into_deserializer())
                .map(Some)
        }

        fn size_hint(&self) -> Option<usize> {
            Some(usize::MAX)
        }
    }

    struct EventBombDeserializer<'a> {
        template: serde_json::Value,
        produced: &'a Cell<usize>,
    }

    impl<'de> Deserializer<'de> for EventBombDeserializer<'_> {
        type Error = serde_json::Error;

        fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
        where
            V: de::Visitor<'de>,
        {
            Err(de::Error::custom("the bomb only offers a sequence"))
        }

        fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where
            V: de::Visitor<'de>,
        {
            visitor.visit_seq(EventBomb {
                template: self.template,
                produced: self.produced,
            })
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
            bytes byte_buf option unit unit_struct newtype_struct tuple
            tuple_struct map struct enum identifier ignored_any
        }
    }

    #[test]
    fn bounded_event_serde_is_capped_before_allocation_and_never_drains_a_hostile_sequence() {
        let template = serde_json::to_value(begin(0)).unwrap();
        let produced = Cell::new(0_usize);
        let error = BoundedGestureEvents::deserialize(EventBombDeserializer {
            template,
            produced: &produced,
        })
        .expect_err("an unbounded sequence must be refused");
        assert!(
            error.to_string().contains("too many gesture events"),
            "{error}"
        );
        assert_eq!(
            produced.get(),
            MAX_GESTURE_EVENTS + 1,
            "the visitor must stop one element past the cap, never drain the input"
        );

        // The same cap holds for an ordinary over-length JSON array.
        let bomb = serde_json::Value::Array(
            (0..=MAX_GESTURE_EVENTS)
                .map(|_| serde_json::to_value(begin(0)).unwrap())
                .collect(),
        );
        assert!(serde_json::from_value::<BoundedGestureEvents>(bomb).is_err());

        // Exactly the cap is accepted.
        let full = serde_json::Value::Array(
            (0..MAX_GESTURE_EVENTS)
                .map(|_| serde_json::to_value(begin(0)).unwrap())
                .collect(),
        );
        let accepted = serde_json::from_value::<BoundedGestureEvents>(full).unwrap();
        assert_eq!(accepted.0.len(), MAX_GESTURE_EVENTS);
    }
    // ---------------------------------------------------------------------
    // Section 2 — the one normalized event adapter and its four ingresses.
    // ---------------------------------------------------------------------

    /// A straight three-point stroke expressed as raw authored samples. Every
    /// ingress in the tranche reduces to exactly this list, so it is the
    /// fixture the four-origin closure test drives.
    fn straight_stroke() -> Vec<RawGestureSample> {
        vec![
            RawGestureSample::new(0, GesturePhase::Begin, GestureMode::Push, [0.25, 0.5]),
            RawGestureSample::new(0, GesturePhase::Move, GestureMode::Push, [0.5, 0.5])
                .with_direction([0.25, 0.0]),
            RawGestureSample::new(0, GesturePhase::Move, GestureMode::Push, [0.75, 0.5])
                .with_direction([0.25, 0.0]),
            RawGestureSample::new(0, GesturePhase::End, GestureMode::Push, [1.0, 0.5])
                .with_direction([0.25, 0.0]),
        ]
    }

    /// Record one sample per accepted frame at 30 Hz, the way the host records
    /// a stroke drawn over four frames.
    fn record_stroke(origin: GestureOrigin, samples: &[RawGestureSample]) -> GestureEventRecorder {
        let mut recorder = GestureEventRecorder::default();
        for sample in samples {
            let tick = recorder.reference_tick_u32();
            let event = normalize_gesture_input(origin, *sample, tick)
                .expect("the fixture stroke is well formed for every origin");
            let outcome = recorder.record_accepted(1.0 / GESTURE_REFERENCE_FPS, &[event]);
            assert_eq!(outcome.recorded, 1);
        }
        recorder
    }

    #[test]
    fn the_same_logical_gesture_through_all_four_origins_records_byte_identical_tracks_and_checksums(
    ) {
        let samples = straight_stroke();
        let reference = record_stroke(GestureOrigin::NativePointer, &samples);
        let reference_track = reference.track().clone();
        assert_eq!(reference_track.events().len(), samples.len());
        assert!(reference_track.is_complete());

        for origin in GestureOrigin::ALL {
            let recorder = record_stroke(origin, &samples);
            assert_eq!(
                recorder.track(),
                &reference_track,
                "{origin} must record a byte-identical track"
            );
            assert_eq!(
                recorder.track().canonical_bytes(),
                reference_track.canonical_bytes(),
                "{origin} must produce an identical canonical stream"
            );
            assert_eq!(
                recorder.track().checksum(),
                reference_track.checksum(),
                "{origin} must produce an identical checksum"
            );
        }

        // The event bits themselves carry no provenance, so one sample built
        // through four origins is one value, not four.
        let bits = GestureOrigin::ALL
            .into_iter()
            .map(|origin| normalize_gesture_input(origin, samples[1], 7).unwrap())
            .collect::<Vec<_>>();
        assert!(bits.windows(2).all(|pair| pair[0] == pair[1]));

        // Provenance still classifies the surfaces for the history law.
        assert!(!GestureOrigin::NativePointer.is_automation());
        assert!(!GestureOrigin::Phone.is_automation());
        assert!(GestureOrigin::Midi.is_automation());
        assert!(GestureOrigin::Osc.is_automation());
    }

    #[test]
    fn a_controller_surface_reproduces_the_identical_stroke_from_independent_scalar_updates() {
        // MIDI and OSC deliver X and Y as separate scalars with no stroke of
        // their own. The surface reduces them to exactly the authored sample
        // list the pointer surfaces produce.
        let mut surface = GestureControlSurface::default().with_stroke(0);
        let mut produced = Vec::new();

        surface.apply(GestureControlInput::X(0.25));
        surface.apply(GestureControlInput::Y(0.5));
        produced.extend(surface.apply(GestureControlInput::Contact(true)));
        assert!(surface.is_open());

        for x in [0.5_f32, 0.75] {
            // A separate X and Y update inside one frame is still one point.
            surface.apply(GestureControlInput::X(x));
            surface.apply(GestureControlInput::Y(0.5));
            produced.extend(surface.sample_motion());
        }

        surface.apply(GestureControlInput::X(1.0));
        produced.extend(surface.apply(GestureControlInput::Contact(false)));
        assert!(!surface.is_open());

        assert_eq!(produced.len(), 4);
        let expected = straight_stroke();
        for (index, (actual, wanted)) in produced.iter().zip(expected.iter()).enumerate() {
            let actual = normalize_gesture_input(GestureOrigin::Midi, *actual, index as u32)
                .expect("surface samples are well formed");
            let wanted = normalize_gesture_input(GestureOrigin::Phone, *wanted, index as u32)
                .expect("fixture samples are well formed");
            assert_eq!(
                actual, wanted,
                "surface sample {index} must match the fixture"
            );
        }

        // A stationary open surface authors nothing rather than a
        // zero-direction point, so holding a controller still cannot fill the
        // bounded track.
        let mut held = GestureControlSurface::default();
        held.apply(GestureControlInput::Contact(true));
        assert!(held.sample_motion().is_none());
        assert!(held.sample_motion().is_none());

        // A second contact edge on an already-open surface authors nothing,
        // and a release edge on a closed surface authors nothing.
        let mut edges = GestureControlSurface::default();
        assert!(edges.apply(GestureControlInput::Contact(false)).is_none());
        assert!(edges.apply(GestureControlInput::Contact(true)).is_some());
        assert!(edges.apply(GestureControlInput::Contact(true)).is_none());
        assert!(edges.release().is_some());
        assert!(edges.release().is_none());

        // Two protocols hold separate surfaces, so their strokes never merge.
        let midi = GestureControlSurface::default().with_stroke(1);
        let osc = GestureControlSurface::default().with_stroke(2);
        assert_ne!(midi, osc);
    }

    #[test]
    fn the_adapter_refuses_an_out_of_range_stroke_and_a_non_finite_position_rather_than_inventing_one(
    ) {
        for origin in GestureOrigin::ALL {
            let out_of_range = RawGestureSample::new(
                MAX_ACTIVE_STROKES as u8,
                GesturePhase::Begin,
                GestureMode::Push,
                [0.5, 0.5],
            );
            assert_eq!(
                normalize_gesture_input(origin, out_of_range, 0),
                Err(GestureIngestError::StrokeOutOfRange {
                    origin,
                    stroke: MAX_ACTIVE_STROKES as u8,
                }),
                "clamping the identity would merge two distinct strokes"
            );
            let hostile_identity =
                RawGestureSample::new(255, GesturePhase::Begin, GestureMode::Push, [0.5, 0.5]);
            assert_eq!(
                normalize_gesture_input(origin, hostile_identity, 0),
                Err(GestureIngestError::StrokeOutOfRange {
                    origin,
                    stroke: 255
                })
            );

            for position in [
                [f32::NAN, 0.5],
                [0.5, f32::INFINITY],
                [f32::NEG_INFINITY, f32::NAN],
            ] {
                let sample =
                    RawGestureSample::new(0, GesturePhase::Begin, GestureMode::Push, position);
                assert_eq!(
                    normalize_gesture_input(origin, sample, 0),
                    Err(GestureIngestError::NonFinitePosition { origin }),
                    "a NaN coordinate must never become an etch at the canvas origin"
                );
            }

            let hostile_pressure =
                RawGestureSample::new(0, GesturePhase::Begin, GestureMode::Push, [0.5, 0.5])
                    .with_pressure(f32::NAN);
            assert_eq!(
                normalize_gesture_input(origin, hostile_pressure, 0),
                Err(GestureIngestError::NonFinitePressure { origin })
            );

            // Every refusal names its surface; the accepted event never does.
            let message = normalize_gesture_input(origin, out_of_range, 0)
                .unwrap_err()
                .to_string();
            assert!(message.contains(origin.key()), "{message}");
        }

        // An out-of-canvas coordinate is still clamped rather than refused:
        // the identity space and the finiteness of a coordinate are laws, a
        // coordinate slightly past the edge is not.
        let overshoot =
            RawGestureSample::new(0, GesturePhase::Begin, GestureMode::Push, [-3.0, 9.0])
                .with_pressure(4.0);
        let event = normalize_gesture_input(GestureOrigin::Phone, overshoot, 0).unwrap();
        assert_eq!(event.position(), [0.0, 1.0]);
        assert_eq!(event.pressure(), 1.0);
    }

    #[test]
    fn authored_direction_magnitude_is_discarded_and_a_hostile_component_stays_inert() {
        let unit = unit_direction([3.0, 4.0]);
        assert!((unit[0] - 0.6).abs() < 1e-6, "{unit:?}");
        assert!((unit[1] - 0.8).abs() < 1e-6, "{unit:?}");

        // A magnitude large enough to square to infinity must still yield the
        // real direction rather than collapsing to the inert zero.
        let huge = unit_direction([f32::MAX, f32::MAX]);
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        assert!((huge[0] - diagonal).abs() < 1e-6, "{huge:?}");
        assert!((huge[1] - diagonal).abs() < 1e-6, "{huge:?}");

        // A tiny magnitude carries the same direction as a large one.
        assert_eq!(unit_direction([1e-30, 0.0]), [1.0, 0.0]);

        // Non-finite components are treated as unset, never rejected and never
        // invented: zero is already the defined inert direction.
        assert_eq!(unit_direction([f32::NAN, f32::NAN]), [0.0, 0.0]);
        assert_eq!(unit_direction([0.0, 0.0]), [0.0, 0.0]);
        assert_eq!(unit_direction([f32::NAN, -2.0]), [0.0, -1.0]);

        // The adapter routes the direction through that law, so a hostile
        // magnitude cannot reach the quantizer as a clamped diagonal.
        let sample = RawGestureSample::new(0, GesturePhase::Move, GestureMode::Curl, [0.5, 0.5])
            .with_direction([1000.0, 0.0]);
        let event = normalize_gesture_input(GestureOrigin::Osc, sample, 0).unwrap();
        assert_eq!(event.direction(), [1.0, 0.0]);
        assert_eq!(event.direction_y, 0);
        assert_eq!(event.mode, GestureMode::Curl);
    }

    #[test]
    fn the_live_recorder_addresses_the_first_accepted_frame_at_tick_zero_at_24_30_and_60_fps() {
        // The same wall-clock stroke described at three display rates occupies
        // the same 30 Hz addresses, exactly as the temporal event track does:
        // one sample on the first accepted frame and one a second later.
        for fps in [24_u32, 30, 60] {
            let delta = 1.0 / fps as f32;
            let mut recorder = GestureEventRecorder::default();
            let mut ticks = Vec::new();
            for frame in 0..=fps {
                let events = if frame == 0 || frame == fps {
                    let sample = RawGestureSample::new(
                        0,
                        if frame == 0 {
                            GesturePhase::Begin
                        } else {
                            GesturePhase::End
                        },
                        GestureMode::Push,
                        [0.5, 0.5],
                    );
                    ticks.push(recorder.reference_tick());
                    vec![normalize_gesture_input(
                        GestureOrigin::NativePointer,
                        sample,
                        recorder.reference_tick_u32(),
                    )
                    .unwrap()]
                } else {
                    Vec::new()
                };
                recorder.record_accepted(delta, &events);
            }
            assert_eq!(
                ticks,
                vec![0, 30],
                "fps {fps} must address one 30 Hz timeline"
            );
            let recorded = recorder
                .track()
                .events()
                .iter()
                .map(|event| event.reference_tick)
                .collect::<Vec<_>>();
            assert_eq!(recorded, vec![0, 30], "fps {fps}");
        }
    }

    #[test]
    fn live_and_export_gesture_addresses_agree_on_one_thirty_hertz_timeline() {
        // There is exactly one export derivation and the gesture track reuses
        // it, so this compares the live accumulator against the shared rounded
        // rational map rather than against a second copy of the rule.
        for fps in [24_u32, 30, 60] {
            let delta = 1.0 / fps as f32;
            let mut recorder = GestureEventRecorder::default();
            for frame in 0..u64::from(fps) * 2 {
                assert_eq!(
                    recorder.reference_tick(),
                    crate::render_export::export_temporal_reference_tick(frame, fps),
                    "fps {fps} frame {frame}"
                );
                recorder.record_accepted(delta, &[]);
            }
        }
    }

    #[test]
    fn a_paused_program_holds_the_gesture_clock_and_never_accumulates_catch_up_debt() {
        let delta = 1.0 / GESTURE_REFERENCE_FPS;
        let mut recorder = GestureEventRecorder::default();
        for _ in 0..30 {
            recorder.record_accepted(delta, &[]);
        }
        assert_eq!(recorder.reference_tick(), 30);

        // A frozen program does not call the recorder at all: any amount of
        // wall time passes and the address does not move.
        assert_eq!(recorder.reference_tick(), 30);

        // A paused frame that still reaches the recorder carries the program
        // clock's zero delta, which is equally inert. Thirty seconds of them
        // create no catch-up debt.
        for _ in 0..900 {
            recorder.record_accepted(0.0, &[]);
        }
        assert_eq!(recorder.reference_tick(), 30);

        // Resuming continues from the held address rather than catching up.
        recorder.record_accepted(delta, &[]);
        assert_eq!(recorder.reference_tick(), 31);

        // A hostile frame delta takes the shared one-reference-frame fallback
        // and a negative delta is inert; neither rewinds the address.
        recorder.record_accepted(f32::NAN, &[]);
        assert_eq!(recorder.reference_tick(), 32);
        recorder.record_accepted(-5.0, &[]);
        assert_eq!(recorder.reference_tick(), 32);

        // A cleared recorder returns to the origin with an empty track.
        recorder.clear();
        assert_eq!(recorder.reference_tick(), 0);
        assert!(recorder.track().events().is_empty());
    }

    #[test]
    fn one_accepted_frame_reports_recorded_truncated_and_rejected_samples_separately() {
        let mut recorder = GestureEventRecorder::default();
        let open = normalize_gesture_input(
            GestureOrigin::Phone,
            RawGestureSample::new(0, GesturePhase::Begin, GestureMode::Push, [0.5, 0.5]),
            0,
        )
        .unwrap();
        let orphan = normalize_gesture_input(
            GestureOrigin::Phone,
            RawGestureSample::new(3, GesturePhase::Move, GestureMode::Push, [0.5, 0.5]),
            0,
        )
        .unwrap();
        let outcome = recorder.record_accepted(1.0 / GESTURE_REFERENCE_FPS, &[open, orphan]);
        assert_eq!(
            outcome,
            GestureRecordOutcome {
                recorded: 1,
                truncated: 0,
                rejected: 1,
            },
            "an orphan Move is rejected, never repaired into a stroke"
        );
        assert!(!outcome.is_empty());
        assert_eq!(recorder.track().events().len(), 1);
        assert!(!recorder.track().is_complete());
        assert_eq!(recorder.track().open_stroke_count(), 1);

        // Filling the bounded track reports truncation instead, and the flag
        // the operator sees is on the track itself.
        let filler = normalize_gesture_input(
            GestureOrigin::Phone,
            RawGestureSample::new(0, GesturePhase::Move, GestureMode::Push, [0.5, 0.5]),
            0,
        )
        .unwrap();
        let batch = vec![filler; MAX_GESTURE_EVENTS];
        let outcome = recorder.record_accepted(0.0, &batch);
        assert_eq!(outcome.recorded, (MAX_GESTURE_EVENTS - 1) as u32);
        assert_eq!(outcome.truncated, 1);
        assert_eq!(outcome.rejected, 0);
        assert_eq!(recorder.track().events().len(), MAX_GESTURE_EVENTS);
        assert!(recorder.track().truncated());
        assert!(GestureRecordOutcome::default().is_empty());
    }
}
