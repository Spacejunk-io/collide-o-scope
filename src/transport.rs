//! Pure, deterministic source-time selection for prepared clips.
//!
//! The timeline owns no decoder, clock, path, or GPU resource. Live playback
//! and offline export provide the same [`ProgramTransportTick`] values and get
//! the same [`FrameSelection`] values back. Decoder generations and bounded
//! command mailboxes are integration concerns; this module reports every
//! discontinuity and boundary crossing needed to drive them.

use std::fmt;

use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Hard limit for cue storage in one clip slot.
pub const MAX_CUE_POINTS: usize = 64;
/// Largest numeric cue identity accepted on disk or over the control protocol.
pub const MAX_CUE_ID: u16 = 4095;
/// Largest authored SMPTE hour. Keeping this below three digits makes the
/// browser/native wire shape fixed-width and bounds every conversion.
pub const MAX_TIMECODE_HOURS: u8 = 99;

const DEFAULT_CLIP_BPM: f64 = 120.0;
const MIN_CLIP_BPM: f64 = 1.0;
const MAX_CLIP_BPM: f64 = 999.0;
const MIN_RATE: f64 = 0.0;
const MAX_RATE: f64 = 16.0;
const MIN_SAMPLE_FPS: f64 = 0.25;
const MAX_SAMPLE_FPS: f64 = 480.0;
const MIN_BEAT_LENGTH: f64 = 1.0 / 64.0;
const MAX_BEAT_LENGTH: f64 = 65_536.0;
const MAX_BEAT_LOOP_LENGTH: f64 = 64.0;
const MAX_TICK_SECONDS: f64 = 86_400.0;
const MAX_SOURCE_DURATION_SECONDS: f64 = 31_536_000.0;
const MAX_BEAT_DELTA: f64 = 1_000_000.0;
const MAX_NORMALIZED_TRAVEL: f64 = 1_000_000_000_000.0;
const MAX_SOURCE_FRAMES: u64 = u32::MAX as u64;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaybackDirection {
    #[default]
    Forward,
    Reverse,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndBehavior {
    #[default]
    Loop,
    PingPong,
    /// Present the terminal frame on the crossing tick, then become transparent.
    OneShot,
    /// Keep presenting the terminal frame after the boundary is reached.
    Hold,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerMode {
    #[default]
    Immediate,
    NextBeat,
    NextBar,
}

/// Closed source-time rates accepted by the transport console. Fractional
/// NTSC rates retain exact rational arithmetic; drop-frame is a numbering law,
/// not a rounded playback speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimecodeRate {
    Fps24,
    Fps25,
    Fps30,
    Fps50,
    Fps60,
    Ntsc24,
    Ntsc30,
    Ntsc30Drop,
    Ntsc60,
    Ntsc60Drop,
}

impl TimecodeRate {
    /// Display-frame modulus used by the final `frames` field.
    pub const fn nominal_fps(self) -> u8 {
        match self {
            Self::Fps24 | Self::Ntsc24 => 24,
            Self::Fps25 => 25,
            Self::Fps30 | Self::Ntsc30 | Self::Ntsc30Drop => 30,
            Self::Fps50 => 50,
            Self::Fps60 | Self::Ntsc60 | Self::Ntsc60Drop => 60,
        }
    }

    /// Exact source-frame rate as numerator / denominator.
    pub const fn rational(self) -> (u32, u32) {
        match self {
            Self::Fps24 => (24, 1),
            Self::Fps25 => (25, 1),
            Self::Fps30 => (30, 1),
            Self::Fps50 => (50, 1),
            Self::Fps60 => (60, 1),
            Self::Ntsc24 => (24_000, 1_001),
            Self::Ntsc30 | Self::Ntsc30Drop => (30_000, 1_001),
            Self::Ntsc60 | Self::Ntsc60Drop => (60_000, 1_001),
        }
    }

    const fn dropped_labels_per_minute(self) -> u8 {
        match self {
            Self::Ntsc30Drop => 2,
            Self::Ntsc60Drop => 4,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimecodeError {
    Hours,
    Minutes,
    Seconds,
    Frames { requested: u8, nominal_fps: u8 },
    DroppedLabel,
    InvalidDuration,
}

impl fmt::Display for TimecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hours => write!(
                formatter,
                "timecode hours must be no greater than {MAX_TIMECODE_HOURS}"
            ),
            Self::Minutes => formatter.write_str("timecode minutes must be within 0..=59"),
            Self::Seconds => formatter.write_str("timecode seconds must be within 0..=59"),
            Self::Frames {
                requested,
                nominal_fps,
            } => write!(
                formatter,
                "timecode frame {requested} is outside 0..{} for this rate",
                nominal_fps.saturating_sub(1)
            ),
            Self::DroppedLabel => formatter.write_str(
                "drop-frame timecode uses a skipped frame label at this minute boundary",
            ),
            Self::InvalidDuration => formatter.write_str(
                "timecode seek requires a finite active source duration greater than zero",
            ),
        }
    }
}

impl std::error::Error for TimecodeError {}

/// Bounded SMPTE-style source address. The semicolon convention is presented
/// by the UI for drop-frame rates, while the wire format remains a typed map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SourceTimecode {
    pub hours: u8,
    pub minutes: u8,
    pub seconds: u8,
    pub frames: u8,
    pub rate: TimecodeRate,
}

impl SourceTimecode {
    pub fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        rate: TimecodeRate,
    ) -> Result<Self, TimecodeError> {
        if hours > MAX_TIMECODE_HOURS {
            return Err(TimecodeError::Hours);
        }
        if minutes > 59 {
            return Err(TimecodeError::Minutes);
        }
        if seconds > 59 {
            return Err(TimecodeError::Seconds);
        }
        let nominal_fps = rate.nominal_fps();
        if frames >= nominal_fps {
            return Err(TimecodeError::Frames {
                requested: frames,
                nominal_fps,
            });
        }
        let dropped = rate.dropped_labels_per_minute();
        if dropped != 0 && !minutes.is_multiple_of(10) && seconds == 0 && frames < dropped {
            return Err(TimecodeError::DroppedLabel);
        }
        Ok(Self {
            hours,
            minutes,
            seconds,
            frames,
            rate,
        })
    }

    /// Zero-based real source-frame ordinal represented by this label.
    pub fn frame_number(self) -> u64 {
        let nominal = u64::from(self.rate.nominal_fps());
        let total_seconds =
            u64::from(self.hours) * 3_600 + u64::from(self.minutes) * 60 + u64::from(self.seconds);
        let nominal_frames = total_seconds * nominal + u64::from(self.frames);
        let drop = u64::from(self.rate.dropped_labels_per_minute());
        if drop == 0 {
            return nominal_frames;
        }
        let total_minutes = u64::from(self.hours) * 60 + u64::from(self.minutes);
        let dropped = drop * (total_minutes - total_minutes / 10);
        nominal_frames.saturating_sub(dropped)
    }

    pub fn source_seconds(self) -> f64 {
        let (numerator, denominator) = self.rate.rational();
        self.frame_number() as f64 * f64::from(denominator) / f64::from(numerator)
    }

    pub fn normalized_for_duration(
        self,
        source_duration_seconds: f64,
    ) -> Result<NormalizedTime, TimecodeError> {
        if !source_duration_seconds.is_finite() || source_duration_seconds <= 0.0 {
            return Err(TimecodeError::InvalidDuration);
        }
        Ok(NormalizedTime::clamped(
            self.source_seconds() / source_duration_seconds,
        ))
    }
}

impl<'de> Deserialize<'de> for SourceTimecode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            hours: u8,
            minutes: u8,
            seconds: u8,
            frames: u8,
            rate: TimecodeRate,
        }

        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.hours, raw.minutes, raw.seconds, raw.frames, raw.rate)
            .map_err(de::Error::custom)
    }
}

/// A finite, inclusive normalized source position.
///
/// Construction from code can clamp with [`NormalizedTime::clamped`]. Serde
/// rejects non-finite values instead of allowing poisoned timeline state.
#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
pub struct NormalizedTime(f64);

impl NormalizedTime {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);

    pub fn new(value: f64) -> Option<Self> {
        (value.is_finite() && (0.0..=1.0).contains(&value)).then_some(Self(if value == 0.0 {
            0.0
        } else {
            value
        }))
    }

    pub fn clamped(value: f64) -> Self {
        let value = if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self(if value == 0.0 { 0.0 } else { value })
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Debug for NormalizedTime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("NormalizedTime")
            .field(&self.0)
            .finish()
    }
}

impl Serialize for NormalizedTime {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for NormalizedTime {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NormalizedTimeVisitor;

        impl<'de> Visitor<'de> for NormalizedTimeVisitor {
            type Value = NormalizedTime;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a finite normalized number in the inclusive range 0..=1")
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                NormalizedTime::new(value)
                    .ok_or_else(|| E::custom("normalized time must be finite and within 0..=1"))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_f64(value as f64)
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_f64(value as f64)
            }
        }

        deserializer.deserialize_any(NormalizedTimeVisitor)
    }
}

/// Allocation-free cue identity. Labels belong to presentation metadata, not
/// the source-time law.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CueId(u16);

impl CueId {
    pub fn new(value: u16) -> Option<Self> {
        (value <= MAX_CUE_ID).then_some(Self(value))
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

impl Serialize for CueId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u16(self.0)
    }
}

impl<'de> Deserialize<'de> for CueId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u16::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| {
            de::Error::custom(format_args!("cue id must be no greater than {MAX_CUE_ID}"))
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CuePoint {
    pub id: CueId,
    pub at: NormalizedTime,
}

/// Fixed-capacity cue collection. Its serde representation is a normal JSON
/// array, but deserialization stops before reading a 65th entry and duplicate
/// numeric IDs are rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CuePoints {
    entries: [Option<CuePoint>; MAX_CUE_POINTS],
    len: u8,
}

impl Default for CuePoints {
    fn default() -> Self {
        Self {
            entries: [None; MAX_CUE_POINTS],
            len: 0,
        }
    }
}

impl CuePoints {
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &CuePoint> {
        self.entries[..self.len()].iter().map(|entry| {
            entry
                .as_ref()
                .expect("CuePoints maintains a compact initialized prefix")
        })
    }

    pub fn get(&self, id: CueId) -> Option<CuePoint> {
        self.iter().find(|cue| cue.id == id).copied()
    }

    /// Insert a cue, replacing an existing cue with the same ID. Returns false
    /// only when a new ID would exceed the fixed capacity.
    pub fn insert(&mut self, cue: CuePoint) -> bool {
        if let Some(index) = self.entries[..self.len()]
            .iter()
            .position(|entry| entry.is_some_and(|current| current.id == cue.id))
        {
            self.entries[index] = Some(cue);
            return true;
        }
        if self.len() == MAX_CUE_POINTS {
            return false;
        }
        self.entries[self.len()] = Some(cue);
        self.len += 1;
        true
    }

    /// Remove one cue without allocating and retain deterministic authored
    /// order for every remaining cue.
    pub fn remove(&mut self, id: CueId) -> bool {
        let Some(index) = self.entries[..self.len()]
            .iter()
            .position(|entry| entry.is_some_and(|cue| cue.id == id))
        else {
            return false;
        };
        let last = self.len().saturating_sub(1);
        for position in index..last {
            self.entries[position] = self.entries[position + 1];
        }
        self.entries[last] = None;
        self.len = self.len.saturating_sub(1);
        true
    }
}

impl Serialize for CuePoints {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for cue in self.iter() {
            sequence.serialize_element(cue)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for CuePoints {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CuePointsVisitor;

        impl<'de> Visitor<'de> for CuePointsVisitor {
            type Value = CuePoints;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "at most {MAX_CUE_POINTS} unique cue points")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut cues = CuePoints::default();
                while let Some(cue) = sequence.next_element::<CuePoint>()? {
                    if cues.len() == MAX_CUE_POINTS {
                        return Err(de::Error::custom(format_args!(
                            "a clip may contain at most {MAX_CUE_POINTS} cue points"
                        )));
                    }
                    if cues.get(cue.id).is_some() {
                        return Err(de::Error::custom(format_args!(
                            "duplicate cue id {}",
                            cue.id.get()
                        )));
                    }
                    let inserted = cues.insert(cue);
                    debug_assert!(inserted);
                }
                Ok(cues)
            }
        }

        deserializer.deserialize_seq(CuePointsVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ClipBeatGrid {
    /// Source tempo metadata used to derive clip length when `length_beats` is
    /// absent.
    pub bpm: f64,
    /// Exact musical length when known. Otherwise duration * BPM / 60 is used.
    pub length_beats: Option<f64>,
    /// When true, program-beat deltas, rather than seconds, advance the clip.
    pub sync_to_program: bool,
    pub beats_per_bar: u8,
}

impl Default for ClipBeatGrid {
    fn default() -> Self {
        Self {
            bpm: DEFAULT_CLIP_BPM,
            length_beats: None,
            sync_to_program: false,
            beats_per_bar: 4,
        }
    }
}

impl ClipBeatGrid {
    pub fn sanitized(self) -> Self {
        Self {
            bpm: finite_clamp(self.bpm, DEFAULT_CLIP_BPM, MIN_CLIP_BPM, MAX_CLIP_BPM),
            length_beats: self.length_beats.and_then(|beats| {
                beats.is_finite().then_some(beats).and_then(|beats| {
                    (beats > 0.0).then_some(beats.clamp(MIN_BEAT_LENGTH, MAX_BEAT_LENGTH))
                })
            }),
            sync_to_program: self.sync_to_program,
            beats_per_bar: self.beats_per_bar.clamp(1, 32),
        }
    }

    pub fn total_beats(self, source_duration_seconds: f64) -> Option<f64> {
        let grid = self.sanitized();
        grid.length_beats.or_else(|| {
            let duration = sanitize_duration(source_duration_seconds);
            (duration > 0.0)
                .then_some((duration * grid.bpm / 60.0).clamp(MIN_BEAT_LENGTH, MAX_BEAT_LENGTH))
        })
    }
}

impl<'de> Deserialize<'de> for ClipBeatGrid {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(default)]
        struct Raw {
            bpm: f64,
            length_beats: Option<f64>,
            sync_to_program: bool,
            beats_per_bar: u8,
        }

        impl Default for Raw {
            fn default() -> Self {
                let value = ClipBeatGrid::default();
                Self {
                    bpm: value.bpm,
                    length_beats: value.length_beats,
                    sync_to_program: value.sync_to_program,
                    beats_per_bar: value.beats_per_bar,
                }
            }
        }

        let raw = Raw::deserialize(deserializer)?;
        Ok(Self {
            bpm: raw.bpm,
            length_beats: raw.length_beats,
            sync_to_program: raw.sync_to_program,
            beats_per_bar: raw.beats_per_bar,
        }
        .sanitized())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BeatLoop {
    /// Absolute beat offset within the clip beat grid.
    pub start_beat: f64,
    pub length_beats: f64,
}

impl Default for BeatLoop {
    fn default() -> Self {
        Self {
            start_beat: 0.0,
            length_beats: 1.0,
        }
    }
}

impl BeatLoop {
    pub fn sanitized(self) -> Self {
        Self {
            start_beat: finite_clamp(self.start_beat, 0.0, 0.0, MAX_BEAT_LENGTH),
            length_beats: finite_clamp(
                self.length_beats,
                1.0,
                MIN_BEAT_LENGTH,
                MAX_BEAT_LOOP_LENGTH,
            ),
        }
    }
}

impl<'de> Deserialize<'de> for BeatLoop {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(default)]
        struct Raw {
            start_beat: f64,
            length_beats: f64,
        }

        impl Default for Raw {
            fn default() -> Self {
                let value = BeatLoop::default();
                Self {
                    start_beat: value.start_beat,
                    length_beats: value.length_beats,
                }
            }
        }

        let raw = Raw::deserialize(deserializer)?;
        Ok(Self {
            start_beat: raw.start_beat,
            length_beats: raw.length_beats,
        }
        .sanitized())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ClipTransportConfig {
    pub direction: PlaybackDirection,
    pub end_behavior: EndBehavior,
    pub trigger_mode: TriggerMode,
    pub in_point: NormalizedTime,
    pub out_point: NormalizedTime,
    /// Non-negative playback multiplier. Direction is represented separately.
    pub rate: f64,
    /// Optional presentation cadence. Logical transport still advances every
    /// tick, while the selected source frame changes only on cadence events.
    pub sample_fps: Option<f64>,
    pub beat_grid: Option<ClipBeatGrid>,
    pub beat_loop: Option<BeatLoop>,
    pub cues: CuePoints,
}

impl Default for ClipTransportConfig {
    fn default() -> Self {
        Self {
            direction: PlaybackDirection::Forward,
            end_behavior: EndBehavior::Loop,
            trigger_mode: TriggerMode::Immediate,
            in_point: NormalizedTime::ZERO,
            out_point: NormalizedTime::ONE,
            rate: 1.0,
            sample_fps: None,
            beat_grid: None,
            beat_loop: None,
            cues: CuePoints::default(),
        }
    }
}

impl ClipTransportConfig {
    pub fn sanitized(self) -> Self {
        let mut in_point = self.in_point;
        let mut out_point = self.out_point;
        if in_point > out_point {
            std::mem::swap(&mut in_point, &mut out_point);
        }
        Self {
            direction: self.direction,
            end_behavior: self.end_behavior,
            trigger_mode: self.trigger_mode,
            in_point,
            out_point,
            rate: finite_clamp(self.rate, 1.0, MIN_RATE, MAX_RATE),
            sample_fps: self.sample_fps.and_then(|fps| {
                (fps.is_finite() && fps > 0.0).then_some(fps.clamp(MIN_SAMPLE_FPS, MAX_SAMPLE_FPS))
            }),
            beat_grid: self.beat_grid.map(ClipBeatGrid::sanitized),
            beat_loop: self.beat_loop.map(BeatLoop::sanitized),
            cues: self.cues,
        }
    }

    pub fn cue(&self, id: CueId) -> Option<CuePoint> {
        self.cues.get(id)
    }
}

impl<'de> Deserialize<'de> for ClipTransportConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(default)]
        struct Raw {
            direction: PlaybackDirection,
            end_behavior: EndBehavior,
            trigger_mode: TriggerMode,
            in_point: NormalizedTime,
            out_point: NormalizedTime,
            rate: f64,
            sample_fps: Option<f64>,
            beat_grid: Option<ClipBeatGrid>,
            beat_loop: Option<BeatLoop>,
            cues: CuePoints,
        }

        impl Default for Raw {
            fn default() -> Self {
                let value = ClipTransportConfig::default();
                Self {
                    direction: value.direction,
                    end_behavior: value.end_behavior,
                    trigger_mode: value.trigger_mode,
                    in_point: value.in_point,
                    out_point: value.out_point,
                    rate: value.rate,
                    sample_fps: value.sample_fps,
                    beat_grid: value.beat_grid,
                    beat_loop: value.beat_loop,
                    cues: value.cues,
                }
            }
        }

        let raw = Raw::deserialize(deserializer)?;
        Ok(Self {
            direction: raw.direction,
            end_behavior: raw.end_behavior,
            trigger_mode: raw.trigger_mode,
            in_point: raw.in_point,
            out_point: raw.out_point,
            rate: raw.rate,
            sample_fps: raw.sample_fps,
            beat_grid: raw.beat_grid,
            beat_loop: raw.beat_loop,
            cues: raw.cues,
        }
        .sanitized())
    }
}

/// Persistent runtime phase for one active clip slot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ClipTransportState {
    /// Continuously evaluated transport position.
    pub position: NormalizedTime,
    /// Most recently exposed position after optional sample-FPS holding.
    pub presented_position: NormalizedTime,
    /// Current traversal direction; PingPong may differ from authored direction.
    pub direction: PlaybackDirection,
    configured_direction: PlaybackDirection,
    configured_end_behavior: EndBehavior,
    pub completed: bool,
    pub sample_accumulator_seconds: f64,
    pub generation: u64,
    pub last_program_beat: Option<f64>,
    initialized: bool,
}

impl Default for ClipTransportState {
    fn default() -> Self {
        Self {
            position: NormalizedTime::ZERO,
            presented_position: NormalizedTime::ZERO,
            direction: PlaybackDirection::Forward,
            configured_direction: PlaybackDirection::Forward,
            configured_end_behavior: EndBehavior::Loop,
            completed: false,
            sample_accumulator_seconds: 0.0,
            generation: 0,
            last_program_beat: None,
            initialized: false,
        }
    }
}

impl ClipTransportState {
    pub fn at(position: NormalizedTime, direction: PlaybackDirection) -> Self {
        Self {
            position,
            presented_position: position,
            direction,
            configured_direction: direction,
            initialized: true,
            ..Self::default()
        }
    }

    fn sanitized(mut self) -> Self {
        self.sample_accumulator_seconds =
            finite_clamp(self.sample_accumulator_seconds, 0.0, 0.0, MAX_TICK_SECONDS);
        self.last_program_beat = self.last_program_beat.filter(|beat| beat.is_finite());
        self
    }
}

impl<'de> Deserialize<'de> for ClipTransportState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(default)]
        struct Raw {
            position: NormalizedTime,
            presented_position: NormalizedTime,
            direction: PlaybackDirection,
            configured_direction: PlaybackDirection,
            configured_end_behavior: EndBehavior,
            completed: bool,
            sample_accumulator_seconds: f64,
            generation: u64,
            last_program_beat: Option<f64>,
            initialized: bool,
        }

        impl Default for Raw {
            fn default() -> Self {
                let value = ClipTransportState::default();
                Self {
                    position: value.position,
                    presented_position: value.presented_position,
                    direction: value.direction,
                    configured_direction: value.configured_direction,
                    configured_end_behavior: value.configured_end_behavior,
                    completed: value.completed,
                    sample_accumulator_seconds: value.sample_accumulator_seconds,
                    generation: value.generation,
                    last_program_beat: value.last_program_beat,
                    initialized: value.initialized,
                }
            }
        }

        let raw = Raw::deserialize(deserializer)?;
        Ok(Self {
            position: raw.position,
            presented_position: raw.presented_position,
            direction: raw.direction,
            configured_direction: raw.configured_direction,
            configured_end_behavior: raw.configured_end_behavior,
            completed: raw.completed,
            sample_accumulator_seconds: raw.sample_accumulator_seconds,
            generation: raw.generation,
            last_program_beat: raw.last_program_beat,
            initialized: raw.initialized,
        }
        .sanitized())
    }
}

/// One authoritative program-clock observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ProgramTransportTick {
    /// Piece-local elapsed delta. Callers pass zero during Freeze Program.
    pub delta_seconds: f64,
    /// Authoritative quarter-note position, internal or MIDI-derived.
    pub program_beat: f64,
    /// False for Freeze Program (or another program-wide transport block).
    pub program_running: bool,
    /// False for Freeze Media. Effective source motion requires both gates.
    pub media_running: bool,
    /// Set when the program beat was explicitly re-anchored or sought.
    pub program_discontinuity: bool,
    pub source_duration_seconds: f64,
    pub source_frame_count: u64,
    /// Direct normalized seek. When both seek fields are present, this wins.
    pub seek_to: Option<NormalizedTime>,
    /// Numeric cue request. A missing cue is reported and leaves time intact.
    pub cue_id: Option<CueId>,
}

impl Default for ProgramTransportTick {
    fn default() -> Self {
        Self {
            delta_seconds: 0.0,
            program_beat: 0.0,
            program_running: true,
            media_running: true,
            program_discontinuity: false,
            source_duration_seconds: 0.0,
            source_frame_count: 0,
            seek_to: None,
            cue_id: None,
        }
    }
}

impl ProgramTransportTick {
    /// Canonicalize an untrusted clock/decoder observation without allocating.
    /// A non-finite beat becomes an explicit discontinuity at beat zero.
    pub fn sanitized(mut self) -> Self {
        self.delta_seconds = finite_clamp(self.delta_seconds, 0.0, 0.0, MAX_TICK_SECONDS);
        if !self.program_beat.is_finite() {
            self.program_beat = 0.0;
            self.program_discontinuity = true;
        }
        self.source_duration_seconds = sanitize_duration(self.source_duration_seconds);
        self.source_frame_count = self.source_frame_count.min(MAX_SOURCE_FRAMES);
        self
    }
}

impl<'de> Deserialize<'de> for ProgramTransportTick {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(default)]
        struct Raw {
            delta_seconds: f64,
            program_beat: f64,
            program_running: bool,
            media_running: bool,
            program_discontinuity: bool,
            source_duration_seconds: f64,
            source_frame_count: u64,
            seek_to: Option<NormalizedTime>,
            cue_id: Option<CueId>,
        }

        impl Default for Raw {
            fn default() -> Self {
                let value = ProgramTransportTick::default();
                Self {
                    delta_seconds: value.delta_seconds,
                    program_beat: value.program_beat,
                    program_running: value.program_running,
                    media_running: value.media_running,
                    program_discontinuity: value.program_discontinuity,
                    source_duration_seconds: value.source_duration_seconds,
                    source_frame_count: value.source_frame_count,
                    seek_to: value.seek_to,
                    cue_id: value.cue_id,
                }
            }
        }

        let raw = Raw::deserialize(deserializer)?;
        Ok(Self {
            delta_seconds: raw.delta_seconds,
            program_beat: raw.program_beat,
            program_running: raw.program_running,
            media_running: raw.media_running,
            program_discontinuity: raw.program_discontinuity,
            source_duration_seconds: raw.source_duration_seconds,
            source_frame_count: raw.source_frame_count,
            seek_to: raw.seek_to,
            cue_id: raw.cue_id,
        }
        .sanitized())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameSelection {
    /// Source position actually presented this tick.
    pub normalized_time: NormalizedTime,
    /// Continuously evaluated position, which can run ahead of a sampled hold.
    pub logical_time: NormalizedTime,
    pub source_seconds: f64,
    pub frame_index: Option<u64>,
    pub direction: PlaybackDirection,
    pub advancing: bool,
    pub sample_due: bool,
    pub held: bool,
    pub transparent: bool,
    pub completed: bool,
    /// True when a seek, re-anchor, authored-law change, wrap, or reflection
    /// requires stale decoder work to be discarded.
    pub discontinuity: bool,
    /// Number of terminal boundaries reached during this tick, saturated at
    /// `u32::MAX`. This includes exact hits and multiple crossings.
    pub boundary_events: u32,
    /// Monotonic generation after applying this tick's discontinuities.
    pub generation: u64,
    pub cue_miss: bool,
}

/// Stateless source-time evaluator.
#[derive(Debug, Default, Clone, Copy)]
pub struct TransportTimeline;

impl TransportTimeline {
    /// Evaluate one tick without consulting wall time or mutable engine state.
    /// The returned state is the sole input required for the next call.
    pub fn select(
        config: &ClipTransportConfig,
        state: ClipTransportState,
        tick: ProgramTransportTick,
    ) -> (ClipTransportState, FrameSelection) {
        let config = config.sanitized();
        let mut state = state.sanitized();
        let tick = tick.sanitized();
        let duration = tick.source_duration_seconds;
        let frame_count = tick.source_frame_count;
        let delta_seconds = tick.delta_seconds;
        let (range_start, range_end) = effective_range(&config, duration);

        let mut base_discontinuity = tick.program_discontinuity;
        let mut cue_miss = false;

        if !state.initialized {
            let initial = match config.direction {
                PlaybackDirection::Forward => range_start,
                PlaybackDirection::Reverse => range_end,
            };
            state.position = NormalizedTime::clamped(initial);
            state.presented_position = state.position;
            state.direction = config.direction;
            state.configured_direction = config.direction;
            state.configured_end_behavior = config.end_behavior;
            state.completed = false;
            state.initialized = true;
            base_discontinuity = true;
        }

        if state.configured_direction != config.direction
            || state.configured_end_behavior != config.end_behavior
        {
            state.direction = config.direction;
            state.configured_direction = config.direction;
            state.configured_end_behavior = config.end_behavior;
            state.completed = false;
            state.sample_accumulator_seconds = 0.0;
            base_discontinuity = true;
        }

        let clamped_position = state.position.get().clamp(range_start, range_end);
        let clamped_presented = state.presented_position.get().clamp(range_start, range_end);
        if clamped_position.to_bits() != state.position.get().to_bits()
            || clamped_presented.to_bits() != state.presented_position.get().to_bits()
        {
            state.position = NormalizedTime::clamped(clamped_position);
            state.presented_position = NormalizedTime::clamped(clamped_presented);
            state.completed = false;
            base_discontinuity = true;
        }

        let requested_seek = tick
            .seek_to
            .or_else(|| tick.cue_id.and_then(|id| config.cue(id).map(|cue| cue.at)));
        if tick.seek_to.is_none() && tick.cue_id.is_some() && requested_seek.is_none() {
            cue_miss = true;
        }
        if let Some(seek) = requested_seek {
            state.position = NormalizedTime::clamped(seek.get().clamp(range_start, range_end));
            state.presented_position = state.position;
            state.direction = config.direction;
            state.completed = false;
            state.sample_accumulator_seconds = 0.0;
            base_discontinuity = true;
        }

        if config.end_behavior == EndBehavior::PingPong {
            normalize_ping_pong_endpoint_direction(
                &mut state.direction,
                state.position.get(),
                range_start,
                range_end,
            );
        }

        let beat_is_finite = tick.program_beat.is_finite();
        let mut beat_delta = 0.0;
        if beat_is_finite {
            if let Some(previous) = state.last_program_beat {
                let raw_delta = tick.program_beat - previous;
                if raw_delta < 0.0 || !raw_delta.is_finite() {
                    base_discontinuity = true;
                } else if tick.program_running && !tick.program_discontinuity {
                    beat_delta = raw_delta.min(MAX_BEAT_DELTA);
                }
            }
            // Updating while either freeze is active prevents catch-up on resume.
            state.last_program_beat = Some(tick.program_beat);
        } else {
            base_discontinuity = true;
            state.last_program_beat = None;
        }

        let running = tick.program_running
            && tick.media_running
            && !tick.program_discontinuity
            && !state.completed;
        let distance = if running {
            normalized_distance(&config, duration, delta_seconds, beat_delta)
        } else {
            0.0
        };
        let previous_position = state.position;
        let completed_before_advance = state.completed;
        let (position, direction, completed, boundary_events) = advance_position(
            state.position.get(),
            state.direction,
            config.end_behavior,
            range_start,
            range_end,
            distance,
            state.completed,
        );
        state.position = NormalizedTime::clamped(position);
        state.direction = direction;
        state.completed = completed;

        let advancing = distance > 0.0
            && (state.position != previous_position || boundary_events > 0 || state.completed);
        let discontinuity = base_discontinuity || boundary_events > 0;
        let generation_increment =
            u64::from(base_discontinuity).saturating_add(u64::from(boundary_events));
        state.generation = state.generation.saturating_add(generation_increment);

        let mut sample_due = discontinuity;
        match config.sample_fps {
            None => {
                if advancing {
                    sample_due = true;
                }
            }
            Some(sample_fps) if running && config.rate > 0.0 => {
                let period = 1.0 / sample_fps;
                state.sample_accumulator_seconds =
                    (state.sample_accumulator_seconds + delta_seconds).min(MAX_TICK_SECONDS);
                if state.sample_accumulator_seconds + f64::EPSILON >= period {
                    let events = (state.sample_accumulator_seconds / period).floor();
                    state.sample_accumulator_seconds -= events * period;
                    if state.sample_accumulator_seconds < 0.0
                        || state.sample_accumulator_seconds >= period
                    {
                        state.sample_accumulator_seconds =
                            state.sample_accumulator_seconds.rem_euclid(period);
                    }
                    sample_due = true;
                }
            }
            Some(_) => {}
        }
        if sample_due {
            state.presented_position = state.position;
        }

        let transparent = config.end_behavior == EndBehavior::OneShot
            && state.completed
            && completed_before_advance;
        let selected = state.presented_position.get();
        let source_seconds = if duration > 0.0 {
            selected * duration
        } else {
            0.0
        };
        let frame_index = frame_index_at(selected, frame_count);
        let held = !sample_due || !advancing || !running || state.completed;

        (
            state,
            FrameSelection {
                normalized_time: state.presented_position,
                logical_time: state.position,
                source_seconds,
                frame_index,
                direction: state.direction,
                advancing,
                sample_due,
                held,
                transparent,
                completed: state.completed,
                discontinuity,
                boundary_events,
                generation: state.generation,
                cue_miss,
            },
        )
    }
}

fn effective_range(config: &ClipTransportConfig, duration: f64) -> (f64, f64) {
    let base_start = config.in_point.get();
    let base_end = config.out_point.get();
    let Some(beat_loop) = config.beat_loop else {
        return (base_start, base_end);
    };
    let Some(total_beats) = config.beat_grid.and_then(|grid| grid.total_beats(duration)) else {
        return (base_start, base_end);
    };
    let beat_loop = beat_loop.sanitized();
    let loop_start = (beat_loop.start_beat / total_beats).clamp(0.0, 1.0);
    let loop_end = ((beat_loop.start_beat + beat_loop.length_beats) / total_beats).clamp(0.0, 1.0);
    let intersected_start = base_start.max(loop_start);
    let intersected_end = base_end.min(loop_end);
    if intersected_start <= intersected_end {
        (intersected_start, intersected_end)
    } else {
        // An out-of-range beat loop cannot poison or invert the authored range.
        (base_start, base_end)
    }
}

fn normalized_distance(
    config: &ClipTransportConfig,
    duration: f64,
    delta_seconds: f64,
    beat_delta: f64,
) -> f64 {
    let distance = match config.beat_grid {
        Some(grid) if grid.sync_to_program => grid
            .total_beats(duration)
            .map_or(0.0, |beats| beat_delta * config.rate / beats),
        _ if duration > 0.0 => delta_seconds * config.rate / duration,
        _ => 0.0,
    };
    finite_clamp(distance, 0.0, 0.0, MAX_NORMALIZED_TRAVEL)
}

fn normalize_ping_pong_endpoint_direction(
    direction: &mut PlaybackDirection,
    position: f64,
    start: f64,
    end: f64,
) {
    if end <= start {
        return;
    }
    if position == start && *direction == PlaybackDirection::Reverse {
        *direction = PlaybackDirection::Forward;
    } else if position == end && *direction == PlaybackDirection::Forward {
        *direction = PlaybackDirection::Reverse;
    }
}

fn advance_position(
    position: f64,
    direction: PlaybackDirection,
    behavior: EndBehavior,
    start: f64,
    end: f64,
    distance: f64,
    already_completed: bool,
) -> (f64, PlaybackDirection, bool, u32) {
    if distance <= 0.0 || already_completed {
        return (position.clamp(start, end), direction, already_completed, 0);
    }
    let span = end - start;
    if span <= 0.0 {
        let completed = matches!(behavior, EndBehavior::OneShot | EndBehavior::Hold);
        return (start, direction, completed, u32::from(completed));
    }
    let position = position.clamp(start, end);
    match behavior {
        EndBehavior::Loop => match direction {
            PlaybackDirection::Forward => {
                let total = position - start + distance;
                let events = saturated_event_count((total / span).floor());
                let remainder = total.rem_euclid(span);
                (start + remainder, direction, false, events)
            }
            PlaybackDirection::Reverse => {
                let total = end - position + distance;
                let events = saturated_event_count((total / span).floor());
                let remainder = total.rem_euclid(span);
                (end - remainder, direction, false, events)
            }
        },
        EndBehavior::PingPong => {
            let offset = position - start;
            let phase = match direction {
                PlaybackDirection::Forward => offset,
                PlaybackDirection::Reverse => 2.0 * span - offset,
            };
            let next_phase = phase + distance;
            let before_boundary = (phase / span).floor();
            let after_boundary = (next_phase / span).floor();
            let events = saturated_event_count(after_boundary - before_boundary);
            let folded = next_phase.rem_euclid(2.0 * span);
            if folded < span {
                (start + folded, PlaybackDirection::Forward, false, events)
            } else if folded > span {
                (
                    end - (folded - span),
                    PlaybackDirection::Reverse,
                    false,
                    events,
                )
            } else {
                (end, PlaybackDirection::Reverse, false, events)
            }
        }
        EndBehavior::OneShot | EndBehavior::Hold => match direction {
            PlaybackDirection::Forward => {
                let reached = position + distance >= end;
                if reached {
                    (end, direction, true, 1)
                } else {
                    (position + distance, direction, false, 0)
                }
            }
            PlaybackDirection::Reverse => {
                let reached = position - distance <= start;
                if reached {
                    (start, direction, true, 1)
                } else {
                    (position - distance, direction, false, 0)
                }
            }
        },
    }
}

fn frame_index_at(normalized: f64, frame_count: u64) -> Option<u64> {
    if frame_count == 0 {
        return None;
    }
    let scaled = (normalized.clamp(0.0, 1.0) * frame_count as f64).floor();
    Some((scaled as u64).min(frame_count - 1))
}

fn saturated_event_count(events: f64) -> u32 {
    if !events.is_finite() || events <= 0.0 {
        0
    } else if events >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        events as u32
    }
}

fn sanitize_duration(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value.min(MAX_SOURCE_DURATION_SECONDS)
    } else {
        0.0
    }
}

fn finite_clamp(value: f64, fallback: f64, min: f64, max: f64) -> f64 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time(value: f64) -> NormalizedTime {
        NormalizedTime::new(value).unwrap()
    }

    fn cue_id(value: u16) -> CueId {
        CueId::new(value).unwrap()
    }

    fn tick(delta_seconds: f64, duration: f64) -> ProgramTransportTick {
        ProgramTransportTick {
            delta_seconds,
            source_duration_seconds: duration,
            source_frame_count: 100,
            ..ProgramTransportTick::default()
        }
    }

    fn initialized_at(value: f64, direction: PlaybackDirection) -> ClipTransportState {
        ClipTransportState::at(time(value), direction)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "expected {expected:.16}, got {actual:.16}"
        );
    }

    #[test]
    fn enum_wire_names_and_defaults_are_stable() {
        assert_eq!(
            serde_json::to_string(&PlaybackDirection::Reverse).unwrap(),
            r#""reverse""#
        );
        assert_eq!(
            serde_json::to_string(&EndBehavior::PingPong).unwrap(),
            r#""ping_pong""#
        );
        assert_eq!(
            serde_json::to_string(&TriggerMode::NextBar).unwrap(),
            r#""next_bar""#
        );
        assert_eq!(PlaybackDirection::default(), PlaybackDirection::Forward);
        assert_eq!(EndBehavior::default(), EndBehavior::Loop);
        assert_eq!(TriggerMode::default(), TriggerMode::Immediate);
    }

    #[test]
    fn normalized_time_and_cue_ids_reject_hostile_serde() {
        assert!(serde_json::from_str::<NormalizedTime>("-0.01").is_err());
        assert!(serde_json::from_str::<NormalizedTime>("1.01").is_err());
        assert!(serde_yaml::from_str::<NormalizedTime>(".nan").is_err());
        assert!(serde_json::from_str::<CueId>("4096").is_err());
        assert_eq!(NormalizedTime::clamped(f64::NAN), NormalizedTime::ZERO);
        assert_eq!(NormalizedTime::clamped(4.0), NormalizedTime::ONE);
        assert_eq!(NormalizedTime::new(-0.0).unwrap().get().to_bits(), 0);
    }

    #[test]
    fn typed_timecode_uses_exact_rates_and_drop_frame_numbering() {
        let frame = SourceTimecode::new(0, 0, 1, 0, TimecodeRate::Fps24).unwrap();
        assert_eq!(frame.frame_number(), 24);
        assert_eq!(frame.source_seconds(), 1.0);
        assert_eq!(
            frame.normalized_for_duration(4.0).unwrap(),
            NormalizedTime::new(0.25).unwrap()
        );

        let one_hour = SourceTimecode::new(1, 0, 0, 0, TimecodeRate::Ntsc30Drop).unwrap();
        assert_eq!(one_hour.frame_number(), 107_892);
        assert!((one_hour.source_seconds() - 3_599.996_4).abs() < 0.000_001);

        let ten_minutes = SourceTimecode::new(0, 10, 0, 0, TimecodeRate::Ntsc60Drop).unwrap();
        assert_eq!(ten_minutes.frame_number(), 35_964);
        assert!((ten_minutes.source_seconds() - 599.999_4).abs() < 0.000_001);
    }

    #[test]
    fn typed_timecode_rejects_skipped_labels_hostile_fields_and_unknown_keys() {
        assert_eq!(
            SourceTimecode::new(0, 1, 0, 0, TimecodeRate::Ntsc30Drop),
            Err(TimecodeError::DroppedLabel)
        );
        assert!(SourceTimecode::new(0, 1, 0, 2, TimecodeRate::Ntsc30Drop).is_ok());
        assert!(SourceTimecode::new(0, 10, 0, 0, TimecodeRate::Ntsc30Drop).is_ok());
        assert!(SourceTimecode::new(0, 0, 0, 24, TimecodeRate::Fps24).is_err());
        assert!(SourceTimecode::new(100, 0, 0, 0, TimecodeRate::Fps24).is_err());

        for hostile in [
            r#"{"hours":0,"minutes":60,"seconds":0,"frames":0,"rate":"fps24"}"#,
            r#"{"hours":0,"minutes":0,"seconds":0,"frames":0,"rate":"unknown"}"#,
            r#"{"hours":0,"minutes":0,"seconds":0,"frames":0,"rate":"fps24","path":"C:\\\\private.mov"}"#,
        ] {
            assert!(serde_json::from_str::<SourceTimecode>(hostile).is_err());
        }
        assert_eq!(
            SourceTimecode::new(0, 0, 1, 0, TimecodeRate::Fps24)
                .unwrap()
                .normalized_for_duration(0.0),
            Err(TimecodeError::InvalidDuration)
        );
    }

    #[test]
    fn cues_are_fixed_capacity_unique_and_round_trip_without_string_ids() {
        let mut cues = CuePoints::default();
        for id in 0..MAX_CUE_POINTS as u16 {
            assert!(cues.insert(CuePoint {
                id: cue_id(id),
                at: time(f64::from(id) / MAX_CUE_POINTS as f64),
            }));
        }
        assert!(!cues.insert(CuePoint {
            id: cue_id(200),
            at: time(0.5),
        }));
        assert!(cues.insert(CuePoint {
            id: cue_id(7),
            at: time(0.75),
        }));
        assert_eq!(cues.len(), MAX_CUE_POINTS);
        assert_eq!(cues.get(cue_id(7)).unwrap().at, time(0.75));
        assert!(cues.remove(cue_id(7)));
        assert!(cues.get(cue_id(7)).is_none());
        assert_eq!(cues.len(), MAX_CUE_POINTS - 1);
        assert!(!cues.remove(cue_id(7)));

        let encoded = serde_json::to_string(&cues).unwrap();
        let decoded: CuePoints = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, cues);
        assert!(
            serde_json::from_str::<CuePoints>(r#"[{"id":1,"at":0.1},{"id":1,"at":0.2}]"#).is_err()
        );
        assert_eq!(std::mem::size_of::<CueId>(), std::mem::size_of::<u16>());
    }

    #[test]
    fn configuration_deserialization_defaults_and_sanitizes() {
        let defaulted: ClipTransportConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted, ClipTransportConfig::default());

        let sanitized: ClipTransportConfig = serde_yaml::from_str(
            "rate: .nan\nsample_fps: .inf\nin_point: 0.8\nout_point: 0.2\nbeat_grid:\n  bpm: .nan\n  beats_per_bar: 0\nbeat_loop:\n  start_beat: .nan\n  length_beats: .inf\n",
        )
        .unwrap();
        assert_eq!(sanitized.rate, 1.0);
        assert_eq!(sanitized.sample_fps, None);
        assert_eq!(sanitized.in_point, time(0.2));
        assert_eq!(sanitized.out_point, time(0.8));
        assert_eq!(sanitized.beat_grid.unwrap().bpm, DEFAULT_CLIP_BPM);
        assert_eq!(sanitized.beat_grid.unwrap().beats_per_bar, 1);
        assert_eq!(sanitized.beat_loop.unwrap(), BeatLoop::default());

        let tick: ProgramTransportTick = serde_yaml::from_str(
            "delta_seconds: .inf\nprogram_beat: .nan\nsource_duration_seconds: -.inf\nsource_frame_count: 18446744073709551615\n",
        )
        .unwrap();
        assert_eq!(tick.delta_seconds, 0.0);
        assert_eq!(tick.program_beat, 0.0);
        assert!(tick.program_discontinuity);
        assert_eq!(tick.source_duration_seconds, 0.0);
        assert_eq!(tick.source_frame_count, MAX_SOURCE_FRAMES);
    }

    #[test]
    fn loop_boundaries_are_half_open_and_count_exact_multiple_crossings() {
        struct Case {
            direction: PlaybackDirection,
            start: f64,
            distance: f64,
            expected: f64,
            events: u32,
        }
        let cases = [
            Case {
                direction: PlaybackDirection::Forward,
                start: 0.25,
                distance: 0.5,
                expected: 0.75,
                events: 0,
            },
            Case {
                direction: PlaybackDirection::Forward,
                start: 0.25,
                distance: 0.75,
                expected: 0.0,
                events: 1,
            },
            Case {
                direction: PlaybackDirection::Forward,
                start: 0.25,
                distance: 3.75,
                expected: 0.0,
                events: 4,
            },
            Case {
                direction: PlaybackDirection::Reverse,
                start: 0.75,
                distance: 0.75,
                expected: 1.0,
                events: 1,
            },
            Case {
                direction: PlaybackDirection::Reverse,
                start: 0.75,
                distance: 3.75,
                expected: 1.0,
                events: 4,
            },
        ];
        for case in cases {
            let (position, direction, completed, events) = advance_position(
                case.start,
                case.direction,
                EndBehavior::Loop,
                0.0,
                1.0,
                case.distance,
                false,
            );
            assert_close(position, case.expected);
            assert_eq!(direction, case.direction);
            assert!(!completed);
            assert_eq!(events, case.events);
        }
    }

    #[test]
    fn ping_pong_reflects_at_exact_boundaries_and_across_many_crossings() {
        struct Case {
            direction: PlaybackDirection,
            start: f64,
            distance: f64,
            expected: f64,
            expected_direction: PlaybackDirection,
            events: u32,
        }
        let cases = [
            Case {
                direction: PlaybackDirection::Forward,
                start: 0.25,
                distance: 0.75,
                expected: 1.0,
                expected_direction: PlaybackDirection::Reverse,
                events: 1,
            },
            Case {
                direction: PlaybackDirection::Forward,
                start: 0.25,
                distance: 1.0,
                expected: 0.75,
                expected_direction: PlaybackDirection::Reverse,
                events: 1,
            },
            Case {
                direction: PlaybackDirection::Forward,
                start: 0.25,
                distance: 3.75,
                expected: 0.0,
                expected_direction: PlaybackDirection::Forward,
                events: 4,
            },
            Case {
                direction: PlaybackDirection::Reverse,
                start: 0.75,
                distance: 0.75,
                expected: 0.0,
                expected_direction: PlaybackDirection::Forward,
                events: 1,
            },
        ];
        for case in cases {
            let (position, direction, completed, events) = advance_position(
                case.start,
                case.direction,
                EndBehavior::PingPong,
                0.0,
                1.0,
                case.distance,
                false,
            );
            assert_close(position, case.expected);
            assert_eq!(direction, case.expected_direction);
            assert!(!completed);
            assert_eq!(events, case.events);
        }
    }

    #[test]
    fn one_shot_becomes_transparent_after_terminal_while_hold_stays_visible() {
        for (behavior, later_transparent) in
            [(EndBehavior::OneShot, true), (EndBehavior::Hold, false)]
        {
            let config = ClipTransportConfig {
                end_behavior: behavior,
                ..ClipTransportConfig::default()
            };
            let state = initialized_at(0.9, PlaybackDirection::Forward);
            let (state, selected) = TransportTimeline::select(&config, state, tick(0.2, 1.0));
            assert_eq!(state.position, NormalizedTime::ONE);
            assert!(state.completed);
            assert!(selected.completed);
            // The terminal frame is observable once even for OneShot.
            assert!(!selected.transparent);
            assert_eq!(selected.boundary_events, 1);

            let (state_again, selected_again) =
                TransportTimeline::select(&config, state, tick(10.0, 1.0));
            assert_eq!(state_again.position, NormalizedTime::ONE);
            assert_eq!(selected_again.boundary_events, 0);
            assert_eq!(selected_again.transparent, later_transparent);
        }
    }

    #[test]
    fn one_shot_exact_forward_reverse_and_large_overshoots_show_terminal_once() {
        struct Case {
            direction: PlaybackDirection,
            start: f64,
            delta_seconds: f64,
            terminal: NormalizedTime,
        }
        let cases = [
            Case {
                direction: PlaybackDirection::Forward,
                start: 0.75,
                delta_seconds: 0.25,
                terminal: NormalizedTime::ONE,
            },
            Case {
                direction: PlaybackDirection::Reverse,
                start: 0.25,
                delta_seconds: 0.25,
                terminal: NormalizedTime::ZERO,
            },
            Case {
                direction: PlaybackDirection::Forward,
                start: 0.4,
                delta_seconds: 10_000.0,
                terminal: NormalizedTime::ONE,
            },
            Case {
                direction: PlaybackDirection::Reverse,
                start: 0.6,
                delta_seconds: 10_000.0,
                terminal: NormalizedTime::ZERO,
            },
        ];
        for case in cases {
            let config = ClipTransportConfig {
                direction: case.direction,
                end_behavior: EndBehavior::OneShot,
                ..ClipTransportConfig::default()
            };
            let mut state = initialized_at(case.start, case.direction);
            state.configured_end_behavior = EndBehavior::OneShot;
            let (state, terminal) =
                TransportTimeline::select(&config, state, tick(case.delta_seconds, 1.0));
            assert_eq!(terminal.normalized_time, case.terminal);
            assert_eq!(terminal.boundary_events, 1);
            assert!(terminal.completed);
            assert!(!terminal.transparent);

            let (_, after) = TransportTimeline::select(&config, state, tick(0.0, 1.0));
            assert_eq!(after.normalized_time, case.terminal);
            assert_eq!(after.boundary_events, 0);
            assert!(after.completed);
            assert!(after.transparent);
        }
    }

    #[test]
    fn reverse_one_shot_and_in_out_range_use_the_authored_lower_boundary() {
        let config = ClipTransportConfig {
            direction: PlaybackDirection::Reverse,
            end_behavior: EndBehavior::OneShot,
            in_point: time(0.2),
            out_point: time(0.8),
            ..ClipTransportConfig::default()
        };
        let (state, initial) =
            TransportTimeline::select(&config, ClipTransportState::default(), tick(0.0, 1.0));
        assert_eq!(initial.normalized_time, time(0.8));
        assert_eq!(initial.direction, PlaybackDirection::Reverse);
        let (_, terminal) = TransportTimeline::select(&config, state, tick(1.0, 1.0));
        assert_eq!(terminal.normalized_time, time(0.2));
        assert!(!terminal.transparent);
        assert_eq!(terminal.boundary_events, 1);
    }

    #[test]
    fn rate_scales_seconds_and_frame_selection_clamps_the_last_frame() {
        let config = ClipTransportConfig {
            rate: 2.0,
            end_behavior: EndBehavior::Hold,
            ..ClipTransportConfig::default()
        };
        let mut state = initialized_at(0.0, PlaybackDirection::Forward);
        state.configured_end_behavior = EndBehavior::Hold;
        let (state, quarter) = TransportTimeline::select(&config, state, tick(0.5, 4.0));
        assert_eq!(state.position, time(0.25));
        assert_close(quarter.source_seconds, 1.0);
        assert_eq!(quarter.frame_index, Some(25));

        let (_, last) = TransportTimeline::select(&config, state, tick(10.0, 4.0));
        assert_eq!(last.normalized_time, NormalizedTime::ONE);
        assert_eq!(last.frame_index, Some(99));
    }

    #[test]
    fn sample_fps_holds_presentation_but_not_logical_transport() {
        let config = ClipTransportConfig {
            sample_fps: Some(2.0),
            end_behavior: EndBehavior::Hold,
            ..ClipTransportConfig::default()
        };
        let mut state = initialized_at(0.0, PlaybackDirection::Forward);
        state.configured_end_behavior = EndBehavior::Hold;
        let (state, first) = TransportTimeline::select(&config, state, tick(0.2, 1.0));
        assert_eq!(first.logical_time, time(0.2));
        assert_eq!(first.normalized_time, time(0.0));
        assert!(!first.sample_due);
        assert!(first.held);

        let (state, second) = TransportTimeline::select(&config, state, tick(0.3, 1.0));
        assert_eq!(second.logical_time, time(0.5));
        assert_eq!(second.normalized_time, time(0.5));
        assert!(second.sample_due);
        assert_close(state.sample_accumulator_seconds, 0.0);

        let (_, dropped_to_latest) = TransportTimeline::select(&config, state, tick(0.75, 1.0));
        assert_eq!(dropped_to_latest.normalized_time, NormalizedTime::ONE);
        assert_eq!(dropped_to_latest.boundary_events, 1);
    }

    #[test]
    fn zero_rate_is_a_true_hold_and_does_not_emit_sample_cadence_events() {
        let config = ClipTransportConfig {
            rate: 0.0,
            sample_fps: Some(120.0),
            ..ClipTransportConfig::default()
        };
        let state = initialized_at(0.3, PlaybackDirection::Forward);
        let (state, selected) = TransportTimeline::select(&config, state, tick(2.0, 1.0));
        assert_eq!(selected.normalized_time, time(0.3));
        assert!(!selected.advancing);
        assert!(!selected.sample_due);
        assert!(selected.held);
        assert_eq!(state.sample_accumulator_seconds, 0.0);
    }

    #[test]
    fn bpm_sync_and_short_beat_loop_share_the_program_beat() {
        let config = ClipTransportConfig {
            end_behavior: EndBehavior::Loop,
            beat_grid: Some(ClipBeatGrid {
                bpm: 120.0,
                length_beats: Some(8.0),
                sync_to_program: true,
                beats_per_bar: 4,
            }),
            beat_loop: Some(BeatLoop {
                start_beat: 2.0,
                length_beats: 2.0,
            }),
            ..ClipTransportConfig::default()
        };
        let mut state = initialized_at(0.25, PlaybackDirection::Forward);
        state.last_program_beat = Some(10.0);
        let tick = ProgramTransportTick {
            delta_seconds: 100.0,
            program_beat: 11.0,
            source_duration_seconds: 4.0,
            source_frame_count: 80,
            ..ProgramTransportTick::default()
        };
        let (state, selected) = TransportTimeline::select(&config, state, tick);
        // One program beat is 1/8 clip; beat-loop range is normalized .25..=.5.
        assert_eq!(selected.normalized_time, time(0.375));
        assert_eq!(state.position, time(0.375));

        let wrap_tick = ProgramTransportTick {
            program_beat: 12.0,
            source_duration_seconds: 4.0,
            source_frame_count: 80,
            ..ProgramTransportTick::default()
        };
        let (_, wrapped) = TransportTimeline::select(&config, state, wrap_tick);
        assert_eq!(wrapped.normalized_time, time(0.25));
        assert_eq!(wrapped.boundary_events, 1);
    }

    #[test]
    fn direct_seek_wins_over_cue_and_missing_cue_is_non_destructive() {
        let mut config = ClipTransportConfig::default();
        assert!(config.cues.insert(CuePoint {
            id: cue_id(3),
            at: time(0.75),
        }));
        let state = initialized_at(0.1, PlaybackDirection::Forward);
        let (state, direct) = TransportTimeline::select(
            &config,
            state,
            ProgramTransportTick {
                source_duration_seconds: 1.0,
                seek_to: Some(time(0.4)),
                cue_id: Some(cue_id(3)),
                ..ProgramTransportTick::default()
            },
        );
        assert_eq!(direct.normalized_time, time(0.4));
        assert!(direct.discontinuity);
        assert!(!direct.cue_miss);
        assert_eq!(direct.generation, 1);

        let (_, miss) = TransportTimeline::select(
            &config,
            state,
            ProgramTransportTick {
                source_duration_seconds: 1.0,
                cue_id: Some(cue_id(9)),
                ..ProgramTransportTick::default()
            },
        );
        assert_eq!(miss.normalized_time, time(0.4));
        assert!(miss.cue_miss);
        assert!(!miss.discontinuity);
    }

    #[test]
    fn freeze_program_and_freeze_media_do_not_accumulate_time_or_beats() {
        let config = ClipTransportConfig {
            beat_grid: Some(ClipBeatGrid {
                length_beats: Some(4.0),
                sync_to_program: true,
                ..ClipBeatGrid::default()
            }),
            ..ClipTransportConfig::default()
        };
        let mut state = initialized_at(0.25, PlaybackDirection::Forward);
        state.last_program_beat = Some(1.0);

        for (program_running, media_running, beat) in [(false, false, 5.0), (true, false, 9.0)] {
            let (next, selected) = TransportTimeline::select(
                &config,
                state,
                ProgramTransportTick {
                    delta_seconds: 20.0,
                    program_beat: beat,
                    program_running,
                    media_running,
                    source_duration_seconds: 2.0,
                    ..ProgramTransportTick::default()
                },
            );
            assert_eq!(selected.normalized_time, time(0.25));
            assert!(!selected.advancing);
            assert!(selected.held);
            state = next;
        }

        let (_, resumed) = TransportTimeline::select(
            &config,
            state,
            ProgramTransportTick {
                program_beat: 10.0,
                source_duration_seconds: 2.0,
                ..ProgramTransportTick::default()
            },
        );
        assert_eq!(resumed.normalized_time, time(0.5));
    }

    #[test]
    fn clock_reanchors_and_backwards_beats_are_discontinuities_without_motion() {
        let config = ClipTransportConfig {
            beat_grid: Some(ClipBeatGrid {
                length_beats: Some(4.0),
                sync_to_program: true,
                ..ClipBeatGrid::default()
            }),
            ..ClipTransportConfig::default()
        };
        let mut state = initialized_at(0.5, PlaybackDirection::Forward);
        state.last_program_beat = Some(100.0);
        for tick in [
            ProgramTransportTick {
                program_beat: 10.0,
                source_duration_seconds: 1.0,
                ..ProgramTransportTick::default()
            },
            ProgramTransportTick {
                program_beat: 200.0,
                program_discontinuity: true,
                source_duration_seconds: 1.0,
                ..ProgramTransportTick::default()
            },
        ] {
            let (next, selected) = TransportTimeline::select(&config, state, tick);
            assert_eq!(selected.normalized_time, time(0.5));
            assert!(selected.discontinuity);
            assert!(!selected.advancing);
            state = next;
        }
    }

    #[test]
    fn hostile_runtime_inputs_fail_to_a_bounded_safe_selection() {
        let config = ClipTransportConfig {
            rate: f64::INFINITY,
            sample_fps: Some(f64::NAN),
            beat_grid: Some(ClipBeatGrid {
                bpm: f64::NAN,
                length_beats: Some(f64::INFINITY),
                sync_to_program: true,
                beats_per_bar: 0,
            }),
            beat_loop: Some(BeatLoop {
                start_beat: f64::NEG_INFINITY,
                length_beats: f64::NAN,
            }),
            ..ClipTransportConfig::default()
        };
        let state = ClipTransportState {
            sample_accumulator_seconds: f64::NAN,
            last_program_beat: Some(f64::NAN),
            ..ClipTransportState::default()
        };
        let (state, selected) = TransportTimeline::select(
            &config,
            state,
            ProgramTransportTick {
                delta_seconds: f64::INFINITY,
                program_beat: f64::NAN,
                source_duration_seconds: f64::NEG_INFINITY,
                source_frame_count: u64::MAX,
                ..ProgramTransportTick::default()
            },
        );
        assert!(state.position.get().is_finite());
        assert!(state.sample_accumulator_seconds.is_finite());
        assert!(selected.source_seconds.is_finite());
        assert!(selected
            .frame_index
            .is_some_and(|frame| frame < MAX_SOURCE_FRAMES));
        assert!(selected.discontinuity);
    }

    #[test]
    fn table_property_all_modes_remain_in_range_and_generation_is_monotonic() {
        let behaviors = [
            EndBehavior::Loop,
            EndBehavior::PingPong,
            EndBehavior::OneShot,
            EndBehavior::Hold,
        ];
        let directions = [PlaybackDirection::Forward, PlaybackDirection::Reverse];
        let distances = [0.0, 0.000_001, 0.125, 0.5, 1.0, 9.75, 10_000.0];
        for behavior in behaviors {
            for direction in directions {
                for start_index in 0..=20 {
                    for distance in distances {
                        let start = 0.2 + f64::from(start_index) * 0.03;
                        let (position, _, _, events) =
                            advance_position(start, direction, behavior, 0.2, 0.8, distance, false);
                        assert!((0.2..=0.8).contains(&position));
                        if distance == 0.0 {
                            assert_eq!(events, 0);
                        }
                    }
                }
            }
        }

        let config = ClipTransportConfig::default();
        let mut state = ClipTransportState::default();
        let mut generation = 0;
        for index in 0..1_000 {
            let (next, selected) = TransportTimeline::select(
                &config,
                state,
                ProgramTransportTick {
                    delta_seconds: 0.017 + f64::from(index % 7) * 0.001,
                    program_beat: f64::from(index) / 30.0,
                    source_duration_seconds: 0.37,
                    source_frame_count: 11,
                    ..ProgramTransportTick::default()
                },
            );
            assert!(selected.generation >= generation);
            assert!((0.0..=1.0).contains(&selected.normalized_time.get()));
            generation = selected.generation;
            state = next;
        }
    }

    fn run_contract(fps: u32) -> Vec<(ClipTransportState, FrameSelection)> {
        let config = ClipTransportConfig {
            direction: PlaybackDirection::Forward,
            end_behavior: EndBehavior::PingPong,
            rate: 1.375,
            sample_fps: Some(23.976),
            beat_grid: Some(ClipBeatGrid {
                bpm: 128.0,
                length_beats: Some(7.5),
                sync_to_program: true,
                beats_per_bar: 4,
            }),
            beat_loop: Some(BeatLoop {
                start_beat: 0.5,
                length_beats: 4.0,
            }),
            ..ClipTransportConfig::default()
        };
        let mut state = ClipTransportState::default();
        let mut output = Vec::new();
        for frame in 0..fps * 12 {
            let seconds = f64::from(frame) / f64::from(fps);
            let tick = ProgramTransportTick {
                delta_seconds: 1.0 / f64::from(fps),
                program_beat: seconds * 128.0 / 60.0,
                source_duration_seconds: 3.75,
                source_frame_count: 90,
                seek_to: (frame == fps * 5).then_some(time(0.4)),
                ..ProgramTransportTick::default()
            };
            let (next, selected) = TransportTimeline::select(&config, state, tick);
            output.push((next, selected));
            state = next;
        }
        output
    }

    #[test]
    fn live_and_export_sequences_are_bit_identical_at_24_30_and_60_fps() {
        for fps in [24, 30, 60] {
            let live = run_contract(fps);
            let export = run_contract(fps);
            assert_eq!(live, export, "live/export divergence at {fps} fps");
            for ((live_state, live_frame), (export_state, export_frame)) in live.iter().zip(&export)
            {
                assert_eq!(
                    live_state.position.get().to_bits(),
                    export_state.position.get().to_bits()
                );
                assert_eq!(
                    live_frame.source_seconds.to_bits(),
                    export_frame.source_seconds.to_bits()
                );
            }
        }
    }
}
