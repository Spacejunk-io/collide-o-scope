//! Persistent, bounded domain model for prepared clip performance.
//!
//! Runtime staging and decoder ownership deliberately live outside these
//! types.  Patch, web, live, and export code can therefore share stable slot
//! and scene identities without ever treating an untrusted ID as a vector
//! index.

use std::collections::HashSet;
use std::fmt;

use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::transport::{
    ClipTransportConfig, CueId, EndBehavior, NormalizedTime, PlaybackDirection, TriggerMode,
};

/// Hard patch/runtime limit for prepared sources owned by one visual layer.
pub const MAX_CLIP_SLOTS_PER_LAYER: usize = 32;
/// Hard patch/runtime limit for named atomic scene records.
pub const MAX_SCENES: usize = 128;
/// A scene cannot address more layer positions than the engine will plan.
pub const MAX_SCENE_BINDINGS: usize = 256;
/// Hard authored/runtime limit for the Scene-only Autopilot sequence.
pub const MAX_AUTOPILOT_STEPS: usize = 128;
/// Longest dwell accepted for one Autopilot step.
pub const MAX_AUTOPILOT_HOLD_BEATS: u16 = 256;
/// Resolume-style useful default: one bar in the engine's default 4/4 meter.
pub const DEFAULT_AUTOPILOT_HOLD_BEATS: u16 = 4;
/// Largest saved zero-based layer position accepted from untrusted input.
pub const MAX_SAVED_LAYER_POSITION: u32 = 4095;

/// Engine-owned description of one scalar authoring value.
///
/// Performance recording, browser validation, and offline replay translate
/// this neutral metadata into their own wire/storage representations. Keeping
/// the owner vocabulary and range here prevents those adapters from growing a
/// second hand-maintained parameter table.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AuthoringValueLaw {
    Unit([f32; 2]),
    Discrete(Vec<&'static str>),
    Toggle,
    Stepped([i64; 2]),
}

/// Integer beat/bar boundaries crossed by one forward clock observation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BeatCrossings {
    pub beats: u32,
    pub bars: u32,
}

impl BeatCrossings {
    pub const fn crossed_beat(self) -> bool {
        self.beats != 0
    }

    pub const fn crossed_bar(self) -> bool {
        self.bars != 0
    }
}

/// Pure boundary detector shared by immediate/beat/bar scene scheduling.
///
/// The first observation, a frozen observation, a non-finite sample, or any
/// backward clock movement only reanchors. They never release queued work.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BeatBoundaryTracker {
    last_program_beat: Option<f64>,
}

impl BeatBoundaryTracker {
    pub fn observe(&mut self, program_beat: f64, beats_per_bar: u8, frozen: bool) -> BeatCrossings {
        if !program_beat.is_finite() {
            self.last_program_beat = None;
            return BeatCrossings::default();
        }
        let current = program_beat.max(0.0);
        let Some(previous) = self.last_program_beat.replace(current) else {
            return BeatCrossings::default();
        };
        if frozen || current <= previous {
            return BeatCrossings::default();
        }

        let previous_integer = previous.floor();
        let current_integer = current.floor();
        if current_integer <= previous_integer {
            return BeatCrossings::default();
        }
        let beats = saturating_positive_f64_to_u32(current_integer - previous_integer);
        let beats_per_bar = f64::from(beats_per_bar.clamp(1, 32));
        let previous_bar = (previous_integer / beats_per_bar).floor();
        let current_bar = (current_integer / beats_per_bar).floor();
        let bars = saturating_positive_f64_to_u32(current_bar - previous_bar);
        BeatCrossings { beats, bars }
    }

    pub fn reanchor(&mut self, program_beat: f64) {
        self.last_program_beat = program_beat.is_finite().then(|| program_beat.max(0.0));
    }
}

fn saturating_positive_f64_to_u32(value: f64) -> u32 {
    if value >= f64::from(u32::MAX) {
        u32::MAX
    } else if value > 0.0 {
        value as u32
    } else {
        0
    }
}

/// Stable identity within one layer's slot collection. It is never an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClipSlotId(u16);

impl ClipSlotId {
    pub const LEGACY: Self = Self(1);

    pub const fn new(value: u16) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

impl Default for ClipSlotId {
    fn default() -> Self {
        Self::LEGACY
    }
}

impl Serialize for ClipSlotId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u16(self.0)
    }
}

impl<'de> Deserialize<'de> for ClipSlotId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u16::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| de::Error::custom("clip slot id must be non-zero"))
    }
}

/// Stable identity within the patch's scene collection. It is never an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SceneId(u16);

impl SceneId {
    pub const fn new(value: u16) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

impl Serialize for SceneId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u16(self.0)
    }
}

impl<'de> Deserialize<'de> for SceneId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u16::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| de::Error::custom("scene id must be non-zero"))
    }
}

/// Number of accepted media beats for which an Autopilot Scene remains live.
///
/// Keeping the bound in the type makes authored, YAML, web, and runtime paths
/// share the same law. The wire representation remains an ordinary integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AutopilotHoldBeats(u16);

impl AutopilotHoldBeats {
    pub const DEFAULT: Self = Self(DEFAULT_AUTOPILOT_HOLD_BEATS);

    pub const fn new(value: u16) -> Option<Self> {
        if value == 0 || value > MAX_AUTOPILOT_HOLD_BEATS {
            None
        } else {
            Some(Self(value))
        }
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

impl Default for AutopilotHoldBeats {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Serialize for AutopilotHoldBeats {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u16(self.0)
    }
}

impl<'de> Deserialize<'de> for AutopilotHoldBeats {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u16::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| {
            de::Error::custom(format_args!(
                "Autopilot hold_beats must be in 1..={MAX_AUTOPILOT_HOLD_BEATS}"
            ))
        })
    }
}

/// Zero-based position in the saved layer stack.
///
/// This is deliberately distinct from a process-stable live layer ID. Exact
/// restore resolves the position to the newly constructed live identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SavedLayerPosition(u32);

impl SavedLayerPosition {
    pub const fn new(value: u32) -> Option<Self> {
        if value <= MAX_SAVED_LAYER_POSITION {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// Resolve through bounds-checked slice access. Callers never index a
    /// vector with this untrusted value directly.
    pub fn resolve<T>(self, layers: &[T]) -> Option<&T> {
        usize::try_from(self.0)
            .ok()
            .and_then(|position| layers.get(position))
    }
}

impl Serialize for SavedLayerPosition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(self.0)
    }
}

impl<'de> Deserialize<'de> for SavedLayerPosition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| {
            de::Error::custom(format_args!(
                "saved layer position must be no greater than {MAX_SAVED_LAYER_POSITION}"
            ))
        })
    }
}

/// One prepared visual source and its authored source-time law.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipSlotConfig {
    pub id: ClipSlotId,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Human-readable logical filename retained independently of host paths.
    pub filename: String,
    /// Canonical path, `cos-sha256://...` reference, or `spout://...` identity.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_path: String,
    #[serde(default)]
    pub transport: ClipTransportConfig,
    #[serde(default)]
    pub saved_playhead: NormalizedTime,
}

impl ClipSlotConfig {
    /// Exact migration of the former one-source layer transport.
    pub fn from_legacy(filename: String, source_path: String, speed: f32, fps: f32) -> Self {
        let rate = if speed.is_finite() {
            f64::from(speed.clamp(0.25, 4.0))
        } else {
            1.0
        };
        let sample_fps = if fps.is_finite() && fps > 0.0 {
            Some(f64::from(fps.clamp(1.0, 240.0)))
        } else {
            Some(30.0)
        };
        let transport = ClipTransportConfig {
            direction: PlaybackDirection::Forward,
            end_behavior: EndBehavior::Loop,
            rate,
            sample_fps,
            ..ClipTransportConfig::default()
        }
        .sanitized();
        Self {
            id: ClipSlotId::LEGACY,
            name: filename.clone(),
            filename,
            source_path,
            transport,
            saved_playhead: NormalizedTime::ZERO,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformanceCollectionError {
    Empty(&'static str),
    TooMany {
        collection: &'static str,
        limit: usize,
    },
    DuplicateId {
        collection: &'static str,
        id: u16,
    },
    DuplicateLayerPosition(u32),
}

impl fmt::Display for PerformanceCollectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty(collection) => write!(formatter, "{collection} may not be empty"),
            Self::TooMany { collection, limit } => {
                write!(
                    formatter,
                    "{collection} may contain at most {limit} entries"
                )
            }
            Self::DuplicateId { collection, id } => {
                write!(formatter, "duplicate {collection} id {id}")
            }
            Self::DuplicateLayerPosition(position) => {
                write!(formatter, "duplicate scene layer position {position}")
            }
        }
    }
}

impl std::error::Error for PerformanceCollectionError {}

/// Bounded slot sequence with ID lookup rather than ID indexing.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipSlots(Vec<ClipSlotConfig>);

impl ClipSlots {
    pub fn try_from_vec(entries: Vec<ClipSlotConfig>) -> Result<Self, PerformanceCollectionError> {
        if entries.is_empty() {
            return Err(PerformanceCollectionError::Empty("clip slots"));
        }
        if entries.len() > MAX_CLIP_SLOTS_PER_LAYER {
            return Err(PerformanceCollectionError::TooMany {
                collection: "clip slots",
                limit: MAX_CLIP_SLOTS_PER_LAYER,
            });
        }
        let mut ids = HashSet::with_capacity(entries.len());
        for slot in &entries {
            if !ids.insert(slot.id) {
                return Err(PerformanceCollectionError::DuplicateId {
                    collection: "clip slot",
                    id: slot.id.get(),
                });
            }
        }
        Ok(Self(entries))
    }

    pub fn singleton(slot: ClipSlotConfig) -> Self {
        Self(vec![slot])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ClipSlotConfig> {
        self.0.iter()
    }

    pub fn get(&self, id: ClipSlotId) -> Option<&ClipSlotConfig> {
        self.0.iter().find(|slot| slot.id == id)
    }

    pub fn get_mut(&mut self, id: ClipSlotId) -> Option<&mut ClipSlotConfig> {
        self.0.iter_mut().find(|slot| slot.id == id)
    }

    /// Replace a stable slot in place or append it when bounded capacity is
    /// available. Existing order is retained so UI/patch diffs stay stable.
    pub fn upsert(&mut self, slot: ClipSlotConfig) -> Result<(), PerformanceCollectionError> {
        if let Some(existing) = self.get_mut(slot.id) {
            *existing = slot;
            return Ok(());
        }
        if self.0.len() == MAX_CLIP_SLOTS_PER_LAYER {
            return Err(PerformanceCollectionError::TooMany {
                collection: "clip slots",
                limit: MAX_CLIP_SLOTS_PER_LAYER,
            });
        }
        self.0.push(slot);
        Ok(())
    }

    /// Remove a slot while preserving the non-empty Set invariant. `None`
    /// means the ID was absent or it names the sole remaining slot.
    pub fn remove_if_not_last(&mut self, id: ClipSlotId) -> Option<ClipSlotConfig> {
        if self.0.len() <= 1 {
            return None;
        }
        let index = self.0.iter().position(|slot| slot.id == id)?;
        Some(self.0.remove(index))
    }

    pub fn first_id(&self) -> Option<ClipSlotId> {
        self.0.first().map(|slot| slot.id)
    }

    pub fn active_or_first(&self, requested: Option<ClipSlotId>) -> Option<ClipSlotId> {
        requested
            .filter(|id| self.get(*id).is_some())
            .or_else(|| self.first_id())
    }
}

impl Serialize for ClipSlots {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for slot in &self.0 {
            sequence.serialize_element(slot)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for ClipSlots {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ClipSlotsVisitor;

        impl<'de> Visitor<'de> for ClipSlotsVisitor {
            type Value = ClipSlots;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "at most {MAX_CLIP_SLOTS_PER_LAYER} unique clip slots"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut entries = Vec::with_capacity(
                    sequence
                        .size_hint()
                        .unwrap_or(0)
                        .min(MAX_CLIP_SLOTS_PER_LAYER),
                );
                while let Some(slot) = sequence.next_element::<ClipSlotConfig>()? {
                    if entries.len() == MAX_CLIP_SLOTS_PER_LAYER {
                        return Err(de::Error::custom(format_args!(
                            "a layer may contain at most {MAX_CLIP_SLOTS_PER_LAYER} clip slots"
                        )));
                    }
                    entries.push(slot);
                }
                Self::Value::try_from_vec(entries).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_seq(ClipSlotsVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneBinding {
    pub layer_position: SavedLayerPosition,
    pub slot_id: ClipSlotId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cue_id: Option<CueId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneReferenceErrorKind {
    Layer,
    Slot,
    Cue,
}

/// Stable, allocation-free diagnostic for one authored scene binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneReferenceIssue {
    pub scene_id: SceneId,
    pub layer_position: SavedLayerPosition,
    pub slot_id: ClipSlotId,
    pub cue_id: Option<CueId>,
    pub kind: SceneReferenceErrorKind,
}

impl fmt::Display for SceneReferenceIssue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let missing = match self.kind {
            SceneReferenceErrorKind::Layer => "layer",
            SceneReferenceErrorKind::Slot => "slot",
            SceneReferenceErrorKind::Cue => "cue",
        };
        write!(
            formatter,
            "Scene {} references missing {missing} at layer position {} (slot {}",
            self.scene_id.get(),
            self.layer_position.get(),
            self.slot_id.get()
        )?;
        if let Some(cue) = self.cue_id {
            write!(formatter, ", cue {}", cue.get())?;
        }
        formatter.write_str(")")
    }
}

/// Bounded bindings with at most one transaction entry per saved layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SceneBindings(Vec<SceneBinding>);

impl SceneBindings {
    pub fn try_from_vec(entries: Vec<SceneBinding>) -> Result<Self, PerformanceCollectionError> {
        if entries.len() > MAX_SCENE_BINDINGS {
            return Err(PerformanceCollectionError::TooMany {
                collection: "scene bindings",
                limit: MAX_SCENE_BINDINGS,
            });
        }
        let mut positions = HashSet::with_capacity(entries.len());
        for binding in &entries {
            if !positions.insert(binding.layer_position) {
                return Err(PerformanceCollectionError::DuplicateLayerPosition(
                    binding.layer_position.get(),
                ));
            }
        }
        Ok(Self(entries))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &SceneBinding> {
        self.0.iter()
    }
}

impl Serialize for SceneBindings {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for binding in &self.0 {
            sequence.serialize_element(binding)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for SceneBindings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SceneBindingsVisitor;

        impl<'de> Visitor<'de> for SceneBindingsVisitor {
            type Value = SceneBindings;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "at most {MAX_SCENE_BINDINGS} unique saved-layer bindings"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut entries =
                    Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX_SCENE_BINDINGS));
                while let Some(binding) = sequence.next_element::<SceneBinding>()? {
                    if entries.len() == MAX_SCENE_BINDINGS {
                        return Err(de::Error::custom(format_args!(
                            "a scene may contain at most {MAX_SCENE_BINDINGS} bindings"
                        )));
                    }
                    entries.push(binding);
                }
                Self::Value::try_from_vec(entries).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_seq(SceneBindingsVisitor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scene {
    pub id: SceneId,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default)]
    pub trigger_mode: TriggerMode,
    #[serde(default)]
    pub bindings: SceneBindings,
}

/// Bounded scene set with stable-ID lookup rather than numeric indexing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Scenes(Vec<Scene>);

impl Scenes {
    pub fn try_from_vec(entries: Vec<Scene>) -> Result<Self, PerformanceCollectionError> {
        if entries.len() > MAX_SCENES {
            return Err(PerformanceCollectionError::TooMany {
                collection: "scenes",
                limit: MAX_SCENES,
            });
        }
        let mut ids = HashSet::with_capacity(entries.len());
        for scene in &entries {
            if !ids.insert(scene.id) {
                return Err(PerformanceCollectionError::DuplicateId {
                    collection: "scene",
                    id: scene.id.get(),
                });
            }
        }
        Ok(Self(entries))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Scene> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Scene> {
        self.0.iter_mut()
    }

    pub fn get(&self, id: SceneId) -> Option<&Scene> {
        self.0.iter().find(|scene| scene.id == id)
    }

    pub fn get_mut(&mut self, id: SceneId) -> Option<&mut Scene> {
        self.0.iter_mut().find(|scene| scene.id == id)
    }

    /// Replace a stable Scene in place or append within the authored bound.
    pub fn upsert(&mut self, scene: Scene) -> Result<(), PerformanceCollectionError> {
        if let Some(existing) = self.get_mut(scene.id) {
            *existing = scene;
            return Ok(());
        }
        if self.0.len() == MAX_SCENES {
            return Err(PerformanceCollectionError::TooMany {
                collection: "scenes",
                limit: MAX_SCENES,
            });
        }
        self.0.push(scene);
        Ok(())
    }

    pub fn remove(&mut self, id: SceneId) -> Option<Scene> {
        let index = self.0.iter().position(|scene| scene.id == id)?;
        Some(self.0.remove(index))
    }
}

impl Serialize for Scenes {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for scene in &self.0 {
            sequence.serialize_element(scene)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for Scenes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ScenesVisitor;

        impl<'de> Visitor<'de> for ScenesVisitor {
            type Value = Scenes;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "at most {MAX_SCENES} uniquely identified scenes")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut entries =
                    Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX_SCENES));
                while let Some(scene) = sequence.next_element::<Scene>()? {
                    if entries.len() == MAX_SCENES {
                        return Err(de::Error::custom(format_args!(
                            "a patch may contain at most {MAX_SCENES} scenes"
                        )));
                    }
                    entries.push(scene);
                }
                Self::Value::try_from_vec(entries).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_seq(ScenesVisitor)
    }
}

/// End-of-sequence law for the Scene-only beat Autopilot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutopilotRepeat {
    Once,
    #[default]
    Loop,
}

/// One stable Scene recall and its exact accepted-media-beat dwell.
/// Duplicate Scene IDs are deliberately legal and retain authored order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutopilotStep {
    pub scene_id: SceneId,
    #[serde(default)]
    pub hold_beats: AutopilotHoldBeats,
}

/// Bounded authored Autopilot step sequence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AutopilotSteps(Vec<AutopilotStep>);

impl AutopilotSteps {
    pub fn try_from_vec(entries: Vec<AutopilotStep>) -> Result<Self, PerformanceCollectionError> {
        if entries.len() > MAX_AUTOPILOT_STEPS {
            return Err(PerformanceCollectionError::TooMany {
                collection: "Autopilot steps",
                limit: MAX_AUTOPILOT_STEPS,
            });
        }
        Ok(Self(entries))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &AutopilotStep> {
        self.0.iter()
    }

    pub fn get(&self, index: usize) -> Option<&AutopilotStep> {
        self.0.get(index)
    }
}

impl Serialize for AutopilotSteps {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for step in &self.0 {
            sequence.serialize_element(step)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for AutopilotSteps {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct AutopilotStepsVisitor;

        impl<'de> Visitor<'de> for AutopilotStepsVisitor {
            type Value = AutopilotSteps;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "at most {MAX_AUTOPILOT_STEPS} Autopilot steps")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                // Never trust an unbounded sequence hint as an allocation
                // request. The vector cannot grow beyond the authored cap.
                let mut entries =
                    Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX_AUTOPILOT_STEPS));
                while let Some(step) = sequence.next_element::<AutopilotStep>()? {
                    if entries.len() == MAX_AUTOPILOT_STEPS {
                        return Err(de::Error::custom(format_args!(
                            "an Autopilot may contain at most {MAX_AUTOPILOT_STEPS} steps"
                        )));
                    }
                    entries.push(step);
                }
                Self::Value::try_from_vec(entries).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_seq(AutopilotStepsVisitor)
    }
}

/// Stable diagnostic for an authored step whose Scene no longer exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutopilotSceneReferenceIssue {
    pub step_index: usize,
    pub scene_id: SceneId,
}

impl fmt::Display for AutopilotSceneReferenceIssue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Autopilot step {} references missing Scene {}",
            self.step_index + 1,
            self.scene_id.get()
        )
    }
}

impl std::error::Error for AutopilotSceneReferenceIssue {}

/// Persisted Scene-only beat sequence. An empty value is the additive legacy
/// default and is omitted from canonical patch serialization.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutopilotPlan {
    #[serde(default)]
    pub repeat: AutopilotRepeat,
    #[serde(default)]
    pub steps: AutopilotSteps,
}

impl AutopilotPlan {
    #[allow(
        dead_code,
        reason = "retained for typed native editors and pure scheduler fixtures"
    )]
    pub fn try_new(
        repeat: AutopilotRepeat,
        steps: Vec<AutopilotStep>,
    ) -> Result<Self, PerformanceCollectionError> {
        Ok(Self {
            repeat,
            steps: AutopilotSteps::try_from_vec(steps)?,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Resolve stable Scene identities against the currently authored set.
    /// The first broken reference is reported with its authored step index;
    /// duplicate valid references remain legal.
    pub fn validate_scenes(&self, scenes: &Scenes) -> Result<(), AutopilotSceneReferenceIssue> {
        for (step_index, step) in self.steps.iter().enumerate() {
            if scenes.get(step.scene_id).is_none() {
                return Err(AutopilotSceneReferenceIssue {
                    step_index,
                    scene_id: step.scene_id,
                });
            }
        }
        Ok(())
    }
}

// The binary currently has no library root. Keeping the pure state machine
// below this domain module makes it testable now without changing main.rs;
// live integration can refer to `performance::autopilot` directly.
#[path = "autopilot.rs"]
pub mod autopilot;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MAX_CUE_POINTS;

    fn slot(id: u16) -> ClipSlotConfig {
        ClipSlotConfig {
            id: ClipSlotId::new(id).unwrap(),
            name: format!("Slot {id}"),
            filename: format!("clip-{id}.mp4"),
            source_path: format!("cos-sha256://clip-{id}"),
            transport: ClipTransportConfig::default(),
            saved_playhead: NormalizedTime::clamped(0.25),
        }
    }

    #[test]
    fn legacy_slot_is_full_range_forward_loop_with_saved_playhead() {
        let slot = ClipSlotConfig::from_legacy(
            "archive.mov".into(),
            "C:/media/archive.mov".into(),
            1.75,
            24.0,
        );
        assert_eq!(slot.id, ClipSlotId::LEGACY);
        assert_eq!(slot.transport.direction, PlaybackDirection::Forward);
        assert_eq!(slot.transport.end_behavior, EndBehavior::Loop);
        assert_eq!(slot.transport.in_point, NormalizedTime::ZERO);
        assert_eq!(slot.transport.out_point, NormalizedTime::ONE);
        assert_eq!(slot.transport.rate, 1.75);
        assert_eq!(slot.transport.sample_fps, Some(24.0));
        assert_eq!(slot.saved_playhead, NormalizedTime::ZERO);

        let hostile = ClipSlotConfig::from_legacy("hostile.mov".into(), String::new(), 99.0, 999.0);
        assert_eq!(hostile.transport.rate, 4.0);
        assert_eq!(hostile.transport.sample_fps, Some(240.0));
    }

    #[test]
    fn slot_limits_and_duplicate_ids_are_rejected_without_id_indexing() {
        let mut slots = ClipSlots::try_from_vec(vec![slot(31), slot(7)]).unwrap();
        assert_eq!(slots.get(ClipSlotId::new(7).unwrap()).unwrap().id.get(), 7);
        slots.upsert(slot(7)).unwrap();
        slots.upsert(slot(12)).unwrap();
        assert_eq!(slots.len(), 3);
        assert_eq!(
            slots
                .remove_if_not_last(ClipSlotId::new(12).unwrap())
                .unwrap()
                .id
                .get(),
            12
        );
        assert!(slots
            .remove_if_not_last(ClipSlotId::new(99).unwrap())
            .is_none());
        assert!(matches!(
            ClipSlots::try_from_vec(vec![slot(1), slot(1)]),
            Err(PerformanceCollectionError::DuplicateId { .. })
        ));
        assert!(matches!(
            ClipSlots::try_from_vec(Vec::new()),
            Err(PerformanceCollectionError::Empty("clip slots"))
        ));
        assert!(serde_yaml::from_str::<ClipSlots>("[]").is_err());

        let yaml = serde_yaml::to_string(
            &(1..=MAX_CLIP_SLOTS_PER_LAYER + 1)
                .map(|id| slot(u16::try_from(id).unwrap()))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(serde_yaml::from_str::<ClipSlots>(&yaml)
            .unwrap_err()
            .to_string()
            .contains("at most 32"));

        assert_eq!(MAX_CUE_POINTS, 64, "transport owns the cue hard limit");
    }

    #[test]
    fn scenes_bound_counts_ids_positions_and_round_trip() {
        let scene = Scene {
            id: SceneId::new(9).unwrap(),
            name: "Cut Nine".into(),
            trigger_mode: TriggerMode::NextBar,
            bindings: SceneBindings::try_from_vec(vec![SceneBinding {
                layer_position: SavedLayerPosition::new(3).unwrap(),
                slot_id: ClipSlotId::new(7).unwrap(),
                cue_id: CueId::new(5),
            }])
            .unwrap(),
        };
        let mut scenes = Scenes::try_from_vec(vec![scene.clone()]).unwrap();
        let yaml = serde_yaml::to_string(&scenes).unwrap();
        let restored: Scenes = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored, scenes);
        assert_eq!(restored.get(scene.id).unwrap().name, "Cut Nine");
        let replacement = Scene {
            name: "Replacement".into(),
            ..scene.clone()
        };
        scenes.upsert(replacement).unwrap();
        assert_eq!(scenes.get(scene.id).unwrap().name, "Replacement");
        assert_eq!(scenes.remove(scene.id).unwrap().id, scene.id);
        assert!(scenes.is_empty());

        let duplicate = SceneBindings::try_from_vec(vec![
            SceneBinding {
                layer_position: SavedLayerPosition::new(0).unwrap(),
                slot_id: ClipSlotId::LEGACY,
                cue_id: None,
            },
            SceneBinding {
                layer_position: SavedLayerPosition::new(0).unwrap(),
                slot_id: ClipSlotId::new(2).unwrap(),
                cue_id: None,
            },
        ]);
        assert!(matches!(
            duplicate,
            Err(PerformanceCollectionError::DuplicateLayerPosition(0))
        ));

        let too_many = (0..=MAX_SCENES)
            .map(|index| Scene {
                id: SceneId::new(u16::try_from(index + 1).unwrap()).unwrap(),
                name: String::new(),
                trigger_mode: TriggerMode::Immediate,
                bindings: SceneBindings::default(),
            })
            .collect();
        assert!(matches!(
            Scenes::try_from_vec(too_many),
            Err(PerformanceCollectionError::TooMany {
                limit: MAX_SCENES,
                ..
            })
        ));
    }

    #[test]
    fn autopilot_plan_is_bounded_defaults_to_a_bar_loop_and_allows_duplicates() {
        let repeated = AutopilotStep {
            scene_id: SceneId::new(9).unwrap(),
            hold_beats: AutopilotHoldBeats::default(),
        };
        let plan =
            AutopilotPlan::try_new(AutopilotRepeat::default(), vec![repeated, repeated]).unwrap();
        assert_eq!(plan.repeat, AutopilotRepeat::Loop);
        assert_eq!(plan.steps.get(0).unwrap().hold_beats.get(), 4);
        assert_eq!(plan.steps.get(1).unwrap().scene_id, repeated.scene_id);

        let yaml = serde_yaml::to_string(&plan).unwrap();
        let restored: AutopilotPlan = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored, plan);

        let too_many = (0..=MAX_AUTOPILOT_STEPS)
            .map(|index| AutopilotStep {
                scene_id: SceneId::new(u16::try_from(index + 1).unwrap()).unwrap(),
                hold_beats: AutopilotHoldBeats::default(),
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            AutopilotSteps::try_from_vec(too_many.clone()),
            Err(PerformanceCollectionError::TooMany {
                limit: MAX_AUTOPILOT_STEPS,
                ..
            })
        ));
        let hostile_yaml = serde_yaml::to_string(&too_many).unwrap();
        assert!(serde_yaml::from_str::<AutopilotSteps>(&hostile_yaml)
            .unwrap_err()
            .to_string()
            .contains("at most 128"));
    }

    #[test]
    fn autopilot_holds_and_scene_references_fail_closed_without_pruning_intent() {
        assert!(serde_yaml::from_str::<AutopilotHoldBeats>("0").is_err());
        assert!(serde_yaml::from_str::<AutopilotHoldBeats>("257").is_err());
        assert_eq!(
            serde_yaml::from_str::<AutopilotHoldBeats>("256")
                .unwrap()
                .get(),
            256
        );

        let plan: AutopilotPlan = serde_yaml::from_str(
            "repeat: once\nsteps:\n  - scene_id: 7\n  - scene_id: 99\n    hold_beats: 1\n",
        )
        .unwrap();
        assert_eq!(plan.steps.get(0).unwrap().hold_beats.get(), 4);
        let scenes = Scenes::try_from_vec(vec![Scene {
            id: SceneId::new(7).unwrap(),
            name: String::new(),
            trigger_mode: TriggerMode::Immediate,
            bindings: SceneBindings::default(),
        }])
        .unwrap();
        let issue = plan.validate_scenes(&scenes).unwrap_err();
        assert_eq!(issue.step_index, 1);
        assert_eq!(issue.scene_id.get(), 99);
        assert_eq!(plan.len(), 2, "validation never prunes authored steps");
    }

    #[test]
    fn invalid_or_zero_domain_ids_and_positions_fail_deserialization() {
        assert!(serde_yaml::from_str::<ClipSlotId>("0").is_err());
        assert!(serde_yaml::from_str::<SceneId>("0").is_err());
        assert!(serde_yaml::from_str::<SavedLayerPosition>("4096").is_err());
    }

    #[test]
    fn beat_boundaries_release_exact_and_multiple_forward_crossings() {
        let mut tracker = BeatBoundaryTracker::default();
        assert_eq!(tracker.observe(1.25, 4, false), BeatCrossings::default());
        assert_eq!(
            tracker.observe(2.0, 4, false),
            BeatCrossings { beats: 1, bars: 0 }
        );
        assert_eq!(tracker.observe(2.0, 4, false), BeatCrossings::default());
        assert_eq!(
            tracker.observe(9.2, 4, false),
            BeatCrossings { beats: 7, bars: 2 }
        );
    }

    #[test]
    fn frozen_backward_and_non_finite_beats_only_reanchor() {
        let mut tracker = BeatBoundaryTracker::default();
        tracker.observe(3.25, 4, false);
        assert_eq!(tracker.observe(5.25, 4, true), BeatCrossings::default());
        assert_eq!(
            tracker.observe(6.0, 4, false),
            BeatCrossings { beats: 1, bars: 0 }
        );
        assert_eq!(tracker.observe(2.0, 4, false), BeatCrossings::default());
        assert_eq!(
            tracker.observe(3.0, 4, false),
            BeatCrossings { beats: 1, bars: 0 }
        );
        assert_eq!(
            tracker.observe(f64::NAN, 4, false),
            BeatCrossings::default()
        );
        assert_eq!(tracker.observe(8.0, 4, false), BeatCrossings::default());
    }
}
