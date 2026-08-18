//! Versioned, identity-safe scoped creative presets.
//!
//! M5 creative presets intentionally copy values only. They never replace
//! rack node IDs, image donors, group identity/membership, root order, or
//! allocator cursors. Rack and group application first proves compatible
//! topology on a clone, then publishes the complete value set atomically.
//! Controller Profile and Stage Map are separate typed document scopes: those
//! payloads retain their complete independently bounded document semantics.

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

use serde::{de, Deserialize, Deserializer, Serialize};

use crate::composition::{BusAssignment, Group, RuntimeGroup};
use crate::image_routing::{LayerMatte, LayerMatteConfig};
use crate::spatial::SpatialTransform;
use crate::visual_rack::{
    ImageMatte, RouteCaptureError, RuntimeImageMatte, RuntimeVisualRack, VisualRack,
    MAX_NODES_PER_RACK,
};

pub const PRESET_SCHEMA_VERSION: u16 = 1;
pub const PRESET_MAX_COUNT: usize = 128;
pub const PRESET_MAX_LIBRARY_BYTES: usize = 8 * 1024 * 1024;
pub const PRESET_LIBRARY_FILE_NAME: &str = "preset_library.yaml";
/// A StageMap document may itself occupy one MiB; retain bounded room for the
/// preset envelope without narrowing that module's already-approved domain.
pub const PRESET_MAX_ENTRY_BYTES: usize = 2 * 1024 * 1024;
pub const PRESET_MAX_NAME_BYTES: usize = 80;

/// Stable typed-document tags. M5 deliberately does not accept an opaque JSON
/// substitute for either owning module's validated document.
pub const PRESET_EXTENSION_CONTROLLER_PROFILE: &str = "controller_profile";
pub const PRESET_EXTENSION_STAGE_MAP: &str = "stage_map";

#[derive(Debug, Clone, PartialEq)]
pub struct PresetLibraryLoad {
    pub path: PathBuf,
    pub library: PresetLibrary,
    pub status: crate::controller_profile::PersistedDocumentLoadStatus,
}

/// Presets are host/operator state, independent from artistic PatchState,
/// recovery journals, controller profiles, and venue Stage Maps.
pub fn default_preset_library_path() -> PathBuf {
    crate::controller_profile::default_control_state_dir().join(PRESET_LIBRARY_FILE_NAME)
}

#[cfg_attr(
    test,
    allow(
        dead_code,
        reason = "App tests isolate themselves from the operator's default preset file"
    )
)]
pub fn load_default_preset_library() -> PresetLibraryLoad {
    load_preset_library_or_default(&default_preset_library_path())
}

pub fn load_preset_library_or_default(path: &Path) -> PresetLibraryLoad {
    use crate::controller_profile::{
        bounded_status, read_bounded_document, BoundedDocumentReadError,
        PersistedDocumentLoadStatus,
    };

    let (library, status) = match read_bounded_document(path, PRESET_MAX_LIBRARY_BYTES) {
        Ok(Some(bytes)) => match std::str::from_utf8(&bytes)
            .map_err(|error| PresetError::Serialization(error.to_string()))
            .and_then(PresetLibrary::from_yaml)
        {
            Ok(library) => (library, PersistedDocumentLoadStatus::Loaded),
            Err(error) => (
                PresetLibrary::default(),
                PersistedDocumentLoadStatus::DefaultInvalid(bounded_status(error.to_string())),
            ),
        },
        Ok(None) => (
            PresetLibrary::default(),
            PersistedDocumentLoadStatus::DefaultMissing,
        ),
        Err(BoundedDocumentReadError::TooLarge(bytes)) => (
            PresetLibrary::default(),
            PersistedDocumentLoadStatus::DefaultInvalid(bounded_status(format!(
                "document is {bytes} bytes; limit is {PRESET_MAX_LIBRARY_BYTES}"
            ))),
        ),
        Err(BoundedDocumentReadError::Io(error)) => (
            PresetLibrary::default(),
            PersistedDocumentLoadStatus::DefaultIo(bounded_status(error)),
        ),
    };
    PresetLibraryLoad {
        path: path.to_path_buf(),
        library,
        status,
    }
}

pub fn save_preset_library_atomic(library: &PresetLibrary, path: &Path) -> Result<(), PresetError> {
    let yaml = library.to_yaml()?;
    crate::controller_profile::write_atomic_document(path, yaml.as_bytes())
        .map_err(|error| PresetError::Io(error.to_string()))
}

#[cfg_attr(
    test,
    allow(
        dead_code,
        reason = "App tests validate publication without writing the operator's default preset file"
    )
)]
pub fn save_default_preset_library_atomic(library: &PresetLibrary) -> Result<PathBuf, PresetError> {
    let path = default_preset_library_path();
    save_preset_library_atomic(library, &path)?;
    Ok(path)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PresetId(NonZeroU64);

impl PresetId {
    pub const fn new(raw: u64) -> Option<Self> {
        match NonZeroU64::new(raw) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

impl Serialize for PresetId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(self.get())
    }
}

impl<'de> Deserialize<'de> for PresetId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = u64::deserialize(deserializer)?;
        Self::new(raw).ok_or_else(|| de::Error::custom("preset ID must be non-zero"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresetKind {
    Transform,
    Rack,
    Matte,
    Group,
    ControllerProfile,
    StageMap,
}

impl PresetKind {
    pub const fn key(self) -> &'static str {
        match self {
            Self::Transform => "transform",
            Self::Rack => "rack",
            Self::Matte => "matte",
            Self::Group => "group",
            Self::ControllerProfile => PRESET_EXTENSION_CONTROLLER_PROFILE,
            Self::StageMap => PRESET_EXTENSION_STAGE_MAP,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresetError {
    UnsupportedSchemaVersion(u16),
    EmptyName,
    NameTooLong { bytes: usize, limit: usize },
    InvalidName,
    EntryTooLarge { bytes: usize, limit: usize },
    LibraryTooLarge { bytes: usize, limit: usize },
    TooManyPresets { count: usize, limit: usize },
    DuplicateId(PresetId),
    InvalidCreatedOrdinal(u64),
    InvalidNextId { next: u64, greatest: u64 },
    IdExhausted,
    MissingPreset(PresetId),
    NonCanonicalTransform,
    RackTopologySignatureMismatch,
    GroupTopologySignatureMismatch,
    IncompatibleTarget,
    RouteCapture(String),
    Serialization(String),
    Io(String),
}

impl fmt::Display for PresetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(
                    formatter,
                    "unsupported scoped preset schema version {version}"
                )
            }
            Self::EmptyName => formatter.write_str("preset name must not be empty"),
            Self::NameTooLong { bytes, limit } => {
                write!(formatter, "preset name is {bytes} bytes; limit is {limit}")
            }
            Self::InvalidName => {
                formatter.write_str("preset name has surrounding whitespace or control characters")
            }
            Self::EntryTooLarge { bytes, limit } => {
                write!(formatter, "preset is {bytes} bytes; limit is {limit}")
            }
            Self::LibraryTooLarge { bytes, limit } => {
                write!(
                    formatter,
                    "preset library is {bytes} bytes; limit is {limit}"
                )
            }
            Self::TooManyPresets { count, limit } => {
                write!(
                    formatter,
                    "preset library has {count} entries; limit is {limit}"
                )
            }
            Self::DuplicateId(id) => write!(formatter, "duplicate preset ID {}", id.get()),
            Self::InvalidCreatedOrdinal(ordinal) => {
                write!(
                    formatter,
                    "preset creation ordinal {ordinal} must be non-zero"
                )
            }
            Self::InvalidNextId { next, greatest } => write!(
                formatter,
                "next preset ID {next} must advance past observed ID {greatest}"
            ),
            Self::IdExhausted => formatter.write_str("preset identity space is exhausted"),
            Self::MissingPreset(id) => write!(formatter, "preset {} does not exist", id.get()),
            Self::NonCanonicalTransform => {
                formatter.write_str("preset transform is outside the authored bounds")
            }
            Self::RackTopologySignatureMismatch => {
                formatter.write_str("preset rack topology signature is inconsistent")
            }
            Self::GroupTopologySignatureMismatch => {
                formatter.write_str("preset group topology signature is inconsistent")
            }
            Self::IncompatibleTarget => formatter
                .write_str("preset target has incompatible value topology; nothing was applied"),
            Self::RouteCapture(error) => write!(formatter, "capture preset routes: {error}"),
            Self::Serialization(error) => write!(formatter, "serialize preset: {error}"),
            Self::Io(error) => write!(formatter, "persist preset library: {error}"),
        }
    }
}

impl std::error::Error for PresetError {}

fn validate_name(name: &str) -> Result<(), PresetError> {
    if name.is_empty() {
        return Err(PresetError::EmptyName);
    }
    if name.len() > PRESET_MAX_NAME_BYTES {
        return Err(PresetError::NameTooLong {
            bytes: name.len(),
            limit: PRESET_MAX_NAME_BYTES,
        });
    }
    if name.trim() != name || name.chars().any(char::is_control) {
        return Err(PresetError::InvalidName);
    }
    Ok(())
}

fn ensure_canonical_transform(transform: SpatialTransform) -> Result<(), PresetError> {
    if transform == transform.sanitized() {
        Ok(())
    } else {
        Err(PresetError::NonCanonicalTransform)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RackPreset {
    pub topology_signature: u64,
    pub rack: VisualRack,
}

impl RackPreset {
    pub fn from_saved(rack: VisualRack) -> Self {
        Self {
            topology_signature: rack.topology_signature(),
            rack,
        }
    }

    pub fn capture_runtime(
        rack: &RuntimeVisualRack,
        position_of_layer: impl FnMut(
            crate::image_routing::StableLayerId,
        ) -> Option<crate::performance::SavedLayerPosition>,
    ) -> Result<Self, PresetError> {
        let saved = rack
            .capture_routes(position_of_layer)
            .map_err(|error| PresetError::RouteCapture(error.to_string()))?;
        Ok(Self::from_saved(saved))
    }

    pub fn validate(&self) -> Result<(), PresetError> {
        if self.rack.len() > MAX_NODES_PER_RACK
            || self.rack.topology_signature() != self.topology_signature
        {
            return Err(PresetError::RackTopologySignatureMismatch);
        }
        Ok(())
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-rack value law is retained for persistence goldens"
        )
    )]
    pub fn apply_to_saved(&self, target: &mut VisualRack) -> Result<(), PresetError> {
        self.validate()?;
        let mut candidate = target.clone();
        if !crate::morph::apply_rack_values(&self.rack, &mut candidate) {
            return Err(PresetError::IncompatibleTarget);
        }
        *target = candidate;
        Ok(())
    }

    pub fn apply_to_runtime(&self, target: &mut RuntimeVisualRack) -> Result<(), PresetError> {
        self.validate()?;
        let mut candidate = target.clone();
        if !crate::morph::apply_saved_rack_values_to_runtime(&self.rack, &mut candidate) {
            return Err(PresetError::IncompatibleTarget);
        }
        *target = candidate;
        Ok(())
    }
}

impl From<RouteCaptureError> for PresetError {
    fn from(error: RouteCaptureError) -> Self {
        Self::RouteCapture(error.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MattePreset {
    pub enabled: bool,
    pub channel: PresetMatteChannel,
    pub invert: bool,
    pub amount: f32,
    pub threshold: f32,
    pub softness: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresetMatteChannel {
    #[default]
    Alpha,
    Luma,
    Red,
    Green,
    Blue,
}

impl From<crate::image_routing::MatteChannel> for PresetMatteChannel {
    fn from(value: crate::image_routing::MatteChannel) -> Self {
        match value {
            crate::image_routing::MatteChannel::Alpha => Self::Alpha,
            crate::image_routing::MatteChannel::Luma => Self::Luma,
            crate::image_routing::MatteChannel::Red => Self::Red,
            crate::image_routing::MatteChannel::Green => Self::Green,
            crate::image_routing::MatteChannel::Blue => Self::Blue,
        }
    }
}

impl From<PresetMatteChannel> for crate::image_routing::MatteChannel {
    fn from(value: PresetMatteChannel) -> Self {
        match value {
            PresetMatteChannel::Alpha => Self::Alpha,
            PresetMatteChannel::Luma => Self::Luma,
            PresetMatteChannel::Red => Self::Red,
            PresetMatteChannel::Green => Self::Green,
            PresetMatteChannel::Blue => Self::Blue,
        }
    }
}

impl From<crate::visual_rack::MatteChannel> for PresetMatteChannel {
    fn from(value: crate::visual_rack::MatteChannel) -> Self {
        match value {
            crate::visual_rack::MatteChannel::Alpha => Self::Alpha,
            crate::visual_rack::MatteChannel::Luma => Self::Luma,
            crate::visual_rack::MatteChannel::Red => Self::Red,
            crate::visual_rack::MatteChannel::Green => Self::Green,
            crate::visual_rack::MatteChannel::Blue => Self::Blue,
        }
    }
}

impl From<PresetMatteChannel> for crate::visual_rack::MatteChannel {
    fn from(value: PresetMatteChannel) -> Self {
        match value {
            PresetMatteChannel::Alpha => Self::Alpha,
            PresetMatteChannel::Luma => Self::Luma,
            PresetMatteChannel::Red => Self::Red,
            PresetMatteChannel::Green => Self::Green,
            PresetMatteChannel::Blue => Self::Blue,
        }
    }
}

impl Default for MattePreset {
    fn default() -> Self {
        Self {
            enabled: false,
            channel: PresetMatteChannel::Alpha,
            invert: false,
            amount: 1.0,
            threshold: 0.5,
            softness: 0.1,
        }
    }
}

impl MattePreset {
    pub fn from_layer(matte: LayerMatte) -> Self {
        let matte = matte.sanitized();
        Self {
            enabled: matte.enabled,
            channel: matte.channel.into(),
            invert: matte.invert,
            amount: matte.amount,
            threshold: matte.threshold,
            softness: matte.softness,
        }
    }

    pub fn from_runtime_group(matte: RuntimeImageMatte) -> Self {
        Self {
            enabled: true,
            channel: matte.channel.into(),
            invert: matte.invert,
            amount: matte.amount,
            threshold: matte.threshold,
            softness: matte.softness,
        }
        .sanitized()
    }

    pub fn sanitized(self) -> Self {
        let route_placeholder = LayerMatteConfig {
            enabled: self.enabled,
            input: crate::image_routing::SavedImageInput::OneBelow,
            channel: self.channel.into(),
            invert: self.invert,
            amount: self.amount,
            threshold: self.threshold,
            softness: self.softness,
        }
        .sanitized();
        Self {
            enabled: route_placeholder.enabled,
            channel: route_placeholder.channel.into(),
            invert: route_placeholder.invert,
            amount: route_placeholder.amount,
            threshold: route_placeholder.threshold,
            softness: route_placeholder.softness,
        }
    }

    pub fn apply_to_layer(self, target: &mut LayerMatte) {
        let values = self.sanitized();
        target.enabled = values.enabled;
        target.channel = values.channel.into();
        target.invert = values.invert;
        target.amount = values.amount;
        target.threshold = values.threshold;
        target.softness = values.softness;
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-layer matte law is retained for persistence goldens"
        )
    )]
    pub fn apply_to_saved_layer(self, target: &mut LayerMatteConfig) {
        let values = self.sanitized();
        target.enabled = values.enabled;
        target.channel = values.channel.into();
        target.invert = values.invert;
        target.amount = values.amount;
        target.threshold = values.threshold;
        target.softness = values.softness;
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-group matte law is retained for persistence goldens"
        )
    )]
    pub fn apply_to_saved_group(self, target: &mut ImageMatte) -> Result<(), PresetError> {
        let values = self.sanitized();
        if !values.enabled {
            return Err(PresetError::IncompatibleTarget);
        }
        target.channel = values.channel.into();
        target.invert = values.invert;
        target.amount = values.amount;
        target.threshold = values.threshold;
        target.softness = values.softness;
        Ok(())
    }

    pub fn apply_to_runtime_group(self, target: &mut RuntimeImageMatte) -> Result<(), PresetError> {
        let values = self.sanitized();
        if !values.enabled {
            return Err(PresetError::IncompatibleTarget);
        }
        target.channel = values.channel.into();
        target.invert = values.invert;
        target.amount = values.amount;
        target.threshold = values.threshold;
        target.softness = values.softness;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupPreset {
    pub topology_signature: u64,
    pub opacity: f32,
    pub transform: SpatialTransform,
    pub rack: RackPreset,
    pub matte: Option<MattePreset>,
    pub solo: bool,
    pub bypass: bool,
    pub bus: BusAssignment,
}

fn group_topology_signature(rack: u64, matte_present: bool) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in rack
        .to_le_bytes()
        .into_iter()
        .chain([u8::from(matte_present)])
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

impl GroupPreset {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-group capture law is retained for persistence goldens"
        )
    )]
    pub fn from_saved(group: &Group) -> Self {
        let rack = RackPreset::from_saved(group.rack.clone());
        let matte = group.matte.map(|matte| MattePreset {
            enabled: true,
            channel: matte.channel.into(),
            invert: matte.invert,
            amount: matte.amount,
            threshold: matte.threshold,
            softness: matte.softness,
        });
        Self {
            topology_signature: group_topology_signature(rack.topology_signature, matte.is_some()),
            opacity: group.opacity,
            transform: group.transform,
            rack,
            matte,
            solo: group.solo,
            bypass: group.bypass,
            bus: group.bus,
        }
    }

    pub fn capture_runtime(
        group: &RuntimeGroup,
        position_of_layer: impl FnMut(
            crate::image_routing::StableLayerId,
        ) -> Option<crate::performance::SavedLayerPosition>,
    ) -> Result<Self, PresetError> {
        let rack = RackPreset::capture_runtime(&group.rack, position_of_layer)?;
        let matte = group.matte.map(|matte| MattePreset {
            enabled: true,
            channel: matte.channel.into(),
            invert: matte.invert,
            amount: matte.amount,
            threshold: matte.threshold,
            softness: matte.softness,
        });
        Ok(Self {
            topology_signature: group_topology_signature(rack.topology_signature, matte.is_some()),
            opacity: group.opacity,
            transform: group.transform,
            rack,
            matte,
            solo: group.solo,
            bypass: group.bypass,
            bus: group.bus,
        })
    }

    pub fn validate(&self) -> Result<(), PresetError> {
        self.rack.validate()?;
        ensure_canonical_transform(self.transform)?;
        if !self.opacity.is_finite() || !(0.0..=1.0).contains(&self.opacity) {
            return Err(PresetError::IncompatibleTarget);
        }
        if self.topology_signature
            != group_topology_signature(self.rack.topology_signature, self.matte.is_some())
        {
            return Err(PresetError::GroupTopologySignatureMismatch);
        }
        if self
            .matte
            .is_some_and(|matte| !matte.enabled || matte != matte.sanitized())
        {
            return Err(PresetError::IncompatibleTarget);
        }
        Ok(())
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "saved-group apply law is retained for persistence goldens"
        )
    )]
    pub fn apply_to_saved(&self, target: &mut Group) -> Result<(), PresetError> {
        self.validate()?;
        if self.matte.is_some() != target.matte.is_some() {
            return Err(PresetError::IncompatibleTarget);
        }
        let mut candidate = target.clone();
        self.rack.apply_to_saved(&mut candidate.rack)?;
        candidate.opacity = self.opacity;
        candidate.transform = self.transform;
        if let (Some(values), Some(matte)) = (self.matte, &mut candidate.matte) {
            matte.channel = values.channel.into();
            matte.invert = values.invert;
            matte.amount = values.sanitized().amount;
            matte.threshold = values.sanitized().threshold;
            matte.softness = values.sanitized().softness;
        }
        candidate.solo = self.solo;
        candidate.bypass = self.bypass;
        candidate.bus = self.bus;
        *target = candidate;
        Ok(())
    }

    pub fn apply_to_runtime(&self, target: &mut RuntimeGroup) -> Result<(), PresetError> {
        self.validate()?;
        if self.matte.is_some() != target.matte.is_some() {
            return Err(PresetError::IncompatibleTarget);
        }
        let mut candidate = target.clone();
        self.rack.apply_to_runtime(&mut candidate.rack)?;
        candidate.opacity = self.opacity;
        candidate.transform = self.transform;
        if let (Some(values), Some(matte)) = (self.matte, &mut candidate.matte) {
            let values = values.sanitized();
            matte.channel = values.channel.into();
            matte.invert = values.invert;
            matte.amount = values.amount;
            matte.threshold = values.threshold;
            matte.softness = values.softness;
        }
        candidate.solo = self.solo;
        candidate.bypass = self.bypass;
        candidate.bus = self.bus;
        *target = candidate;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "values", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PresetPayload {
    Transform(SpatialTransform),
    Rack(RackPreset),
    Matte(MattePreset),
    Group(GroupPreset),
    /// Exact, independently bounded controller document. Applying this does
    /// not rewrite creative topology; Main resolves its saved layer positions
    /// against the live stable-ID stack before publishing it to MIDI.
    ControllerProfile(crate::controller_profile::ControllerProfileDocument),
    /// Exact, independently bounded venue document. Endpoint and slice IDs
    /// are part of this document's identity and are validated before publish.
    StageMap(crate::stage_map::StageMap),
}

impl PresetPayload {
    pub const fn kind(&self) -> PresetKind {
        match self {
            Self::Transform(_) => PresetKind::Transform,
            Self::Rack(_) => PresetKind::Rack,
            Self::Matte(_) => PresetKind::Matte,
            Self::Group(_) => PresetKind::Group,
            Self::ControllerProfile(_) => PresetKind::ControllerProfile,
            Self::StageMap(_) => PresetKind::StageMap,
        }
    }

    fn validate(&self) -> Result<(), PresetError> {
        match self {
            Self::Transform(transform) => ensure_canonical_transform(*transform),
            Self::Rack(rack) => rack.validate(),
            Self::Matte(values) => {
                if *values == values.sanitized() {
                    Ok(())
                } else {
                    Err(PresetError::IncompatibleTarget)
                }
            }
            Self::Group(group) => group.validate(),
            Self::ControllerProfile(profile) => profile
                .to_json_bytes()
                .map(|_| ())
                .map_err(|error| PresetError::Serialization(error.to_string())),
            Self::StageMap(stage_map) => stage_map
                .to_yaml_bytes()
                .map(|_| ())
                .map_err(|error| PresetError::Serialization(error.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScopedPreset {
    pub schema_version: u16,
    pub id: PresetId,
    pub name: String,
    pub created_ordinal: u64,
    pub payload: PresetPayload,
}

impl ScopedPreset {
    fn validate(&self) -> Result<(), PresetError> {
        if self.schema_version != PRESET_SCHEMA_VERSION {
            return Err(PresetError::UnsupportedSchemaVersion(self.schema_version));
        }
        validate_name(&self.name)?;
        if self.created_ordinal == 0 {
            return Err(PresetError::InvalidCreatedOrdinal(self.created_ordinal));
        }
        self.payload.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| PresetError::Serialization(error.to_string()))?
            .len();
        if bytes > PRESET_MAX_ENTRY_BYTES {
            return Err(PresetError::EntryTooLarge {
                bytes,
                limit: PRESET_MAX_ENTRY_BYTES,
            });
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for ScopedPreset {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            schema_version: u16,
            id: PresetId,
            name: String,
            created_ordinal: u64,
            payload: PresetPayload,
        }

        let raw = Raw::deserialize(deserializer)?;
        let value = Self {
            schema_version: raw.schema_version,
            id: raw.id,
            name: raw.name,
            created_ordinal: raw.created_ordinal,
            payload: raw.payload,
        };
        value.validate().map_err(de::Error::custom)?;
        Ok(value)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PresetLibrary {
    pub schema_version: u16,
    next_preset_id: u64,
    next_created_ordinal: u64,
    presets: Vec<ScopedPreset>,
}

impl Default for PresetLibrary {
    fn default() -> Self {
        Self {
            schema_version: PRESET_SCHEMA_VERSION,
            next_preset_id: 1,
            next_created_ordinal: 1,
            presets: Vec::new(),
        }
    }
}

struct BoundedPresetList(Vec<ScopedPreset>);

impl<'de> Deserialize<'de> for BoundedPresetList {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PresetListVisitor;

        impl<'de> de::Visitor<'de> for PresetListVisitor {
            type Value = BoundedPresetList;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "at most {PRESET_MAX_COUNT} scoped presets")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                if sequence
                    .size_hint()
                    .is_some_and(|count| count > PRESET_MAX_COUNT)
                {
                    return Err(de::Error::custom(format_args!(
                        "preset library exceeds {PRESET_MAX_COUNT} entries"
                    )));
                }
                let mut presets =
                    Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(PRESET_MAX_COUNT));
                while presets.len() < PRESET_MAX_COUNT {
                    let Some(preset) = sequence.next_element()? else {
                        return Ok(BoundedPresetList(presets));
                    };
                    presets.push(preset);
                }
                if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::custom(format_args!(
                        "preset library exceeds {PRESET_MAX_COUNT} entries"
                    )));
                }
                Ok(BoundedPresetList(presets))
            }
        }

        deserializer.deserialize_seq(PresetListVisitor)
    }
}

impl PresetLibrary {
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ScopedPreset> {
        self.presets.iter()
    }

    pub fn get(&self, id: PresetId) -> Option<&ScopedPreset> {
        self.presets.iter().find(|preset| preset.id == id)
    }

    pub fn insert(
        &mut self,
        name: impl Into<String>,
        payload: PresetPayload,
    ) -> Result<PresetId, PresetError> {
        if self.presets.len() == PRESET_MAX_COUNT {
            return Err(PresetError::TooManyPresets {
                count: self.presets.len() + 1,
                limit: PRESET_MAX_COUNT,
            });
        }
        let id = PresetId::new(self.next_preset_id).ok_or(PresetError::IdExhausted)?;
        let ordinal = self.next_created_ordinal;
        if ordinal == 0 {
            return Err(PresetError::IdExhausted);
        }
        let preset = ScopedPreset {
            schema_version: PRESET_SCHEMA_VERSION,
            id,
            name: name.into(),
            created_ordinal: ordinal,
            payload,
        };
        preset.validate()?;
        let mut candidate = self.clone();
        candidate.presets.push(preset);
        candidate.next_preset_id = self.next_preset_id.checked_add(1).unwrap_or(0);
        candidate.next_created_ordinal = self.next_created_ordinal.checked_add(1).unwrap_or(0);
        candidate.validate()?;
        *self = candidate;
        Ok(id)
    }

    pub fn remove(&mut self, id: PresetId) -> Result<ScopedPreset, PresetError> {
        let index = self
            .presets
            .iter()
            .position(|preset| preset.id == id)
            .ok_or(PresetError::MissingPreset(id))?;
        Ok(self.presets.remove(index))
    }

    pub fn to_yaml(&self) -> Result<String, PresetError> {
        self.validate()?;
        let yaml = serde_yaml::to_string(self)
            .map_err(|error| PresetError::Serialization(error.to_string()))?;
        if yaml.len() > PRESET_MAX_LIBRARY_BYTES {
            return Err(PresetError::LibraryTooLarge {
                bytes: yaml.len(),
                limit: PRESET_MAX_LIBRARY_BYTES,
            });
        }
        Ok(yaml)
    }

    pub fn from_yaml(yaml: &str) -> Result<Self, PresetError> {
        if yaml.len() > PRESET_MAX_LIBRARY_BYTES {
            return Err(PresetError::LibraryTooLarge {
                bytes: yaml.len(),
                limit: PRESET_MAX_LIBRARY_BYTES,
            });
        }
        serde_yaml::from_str(yaml).map_err(|error| PresetError::Serialization(error.to_string()))
    }

    fn validate(&self) -> Result<(), PresetError> {
        if self.schema_version != PRESET_SCHEMA_VERSION {
            return Err(PresetError::UnsupportedSchemaVersion(self.schema_version));
        }
        if self.presets.len() > PRESET_MAX_COUNT {
            return Err(PresetError::TooManyPresets {
                count: self.presets.len(),
                limit: PRESET_MAX_COUNT,
            });
        }
        let mut ids = BTreeSet::new();
        let mut greatest = 0;
        let mut greatest_ordinal = 0;
        for preset in &self.presets {
            preset.validate()?;
            if !ids.insert(preset.id) {
                return Err(PresetError::DuplicateId(preset.id));
            }
            greatest = greatest.max(preset.id.get());
            greatest_ordinal = greatest_ordinal.max(preset.created_ordinal);
        }
        if (self.next_preset_id != 0 && self.next_preset_id <= greatest)
            || (self.next_preset_id == 0 && greatest != u64::MAX)
        {
            return Err(PresetError::InvalidNextId {
                next: self.next_preset_id,
                greatest,
            });
        }
        if (self.next_created_ordinal != 0 && self.next_created_ordinal <= greatest_ordinal)
            || (self.next_created_ordinal == 0 && greatest_ordinal != u64::MAX)
        {
            return Err(PresetError::InvalidNextId {
                next: self.next_created_ordinal,
                greatest: greatest_ordinal,
            });
        }
        let bytes = serde_json::to_vec(self)
            .map_err(|error| PresetError::Serialization(error.to_string()))?
            .len();
        if bytes > PRESET_MAX_LIBRARY_BYTES {
            return Err(PresetError::LibraryTooLarge {
                bytes,
                limit: PRESET_MAX_LIBRARY_BYTES,
            });
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for PresetLibrary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            schema_version: u16,
            next_preset_id: u64,
            next_created_ordinal: u64,
            presets: BoundedPresetList,
        }

        let raw = Raw::deserialize(deserializer)?;
        let value = Self {
            schema_version: raw.schema_version,
            next_preset_id: raw.next_preset_id,
            next_created_ordinal: raw.next_created_ordinal,
            presets: raw.presets.0,
        };
        value.validate().map_err(de::Error::custom)?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::image_routing::{
        ImageInput, LayerImageStage, MatteChannel as LayerMatteChannel, SavedImageInput,
        StableLayerId,
    };
    use crate::performance::SavedLayerPosition;
    use crate::visual_rack::{
        DigitalColorParams, EdgeTiming, LegacyRackScope, MatteChannel as RackMatteChannel,
        NodeBlend, ResolvedImageSource, ResolvedImageTap, RuntimeImageMatte, RuntimeMaskParams,
        RuntimeVisualNodeKind,
    };

    static TEST_PATH_SEQUENCE: AtomicU64 = AtomicU64::new(1);

    fn test_library_path(label: &str) -> PathBuf {
        let ordinal = TEST_PATH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "collide-o-scope-preset-library-{label}-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        directory.join(PRESET_LIBRARY_FILE_NAME)
    }

    fn remove_test_library(path: &Path) {
        let _ = fs::remove_file(path);
        if let Some(parent) = path.parent() {
            for entry in fs::read_dir(parent).unwrap() {
                let entry = entry.unwrap();
                assert!(
                    !entry.file_name().to_string_lossy().contains(".tmp-"),
                    "atomic writer left a temporary file"
                );
                fs::remove_file(entry.path()).unwrap();
            }
            fs::remove_dir(parent).unwrap();
        }
    }

    fn layer_id(raw: u64) -> StableLayerId {
        StableLayerId::new(raw).unwrap()
    }

    fn saved_position(raw: u32) -> SavedLayerPosition {
        SavedLayerPosition::new(raw).unwrap()
    }

    #[test]
    fn rack_preset_copies_values_but_preserves_node_and_donor_identity() {
        let donor = layer_id(44);
        let other_donor = layer_id(99);
        let mut source = RuntimeVisualRack::empty();
        let node_id = source
            .push(RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(
                RuntimeImageMatte {
                    tap: ResolvedImageTap {
                        source: ResolvedImageSource::SelectedLayer {
                            layer_id: donor,
                            saved_position: saved_position(3),
                            stage: LayerImageStage::PostLocalEffects,
                        },
                        timing: EdgeTiming::CurrentFrame,
                    },
                    channel: RackMatteChannel::Alpha,
                    invert: false,
                    amount: 0.25,
                    threshold: 0.4,
                    softness: 0.2,
                },
            )))
            .unwrap();
        let preset =
            RackPreset::capture_runtime(&source, |id| (id == donor).then(|| saved_position(3)))
                .unwrap();

        let mut target = source.clone();
        let node = target.get_mut(node_id).unwrap();
        node.enabled = false;
        node.wet = 0.9;
        node.blend = NodeBlend::Screen;
        let RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) = &mut node.kind else {
            panic!("image mask")
        };
        matte.tap.source = ResolvedImageSource::SelectedLayer {
            layer_id: other_donor,
            saved_position: saved_position(8),
            stage: LayerImageStage::PreLocalEffects,
        };
        matte.amount = 0.8;

        preset.apply_to_runtime(&mut target).unwrap();
        let node = target.get(node_id).unwrap();
        assert!(node.enabled);
        let RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(matte)) = node.kind else {
            panic!("image mask")
        };
        assert_eq!(matte.amount, 0.25);
        assert!(matches!(
            matte.tap.source,
            ResolvedImageSource::SelectedLayer {
                layer_id,
                saved_position: provenance,
                stage: LayerImageStage::PreLocalEffects,
            } if layer_id == other_donor && provenance == saved_position(8)
        ));
    }

    /// Presets are node-kind agnostic: both entry points delegate to
    /// `morph::apply_*`, so registering the Symmetry Field there is what closes
    /// this row. A missing morph arm would make every preset apply return
    /// `IncompatibleTarget` for any rack containing the node.
    #[test]
    fn rack_preset_copies_symmetry_values_but_preserves_its_four_routes_and_masks() {
        use crate::symmetry::{
            RuntimeSymmetryParams, SavedMotionDonor, SymmetryBoundary, SymmetryMode,
            SymmetryMotionMask, SymmetrySourceMask,
        };

        let donor = layer_id(44);
        let other_donor = layer_id(99);
        let authored = RuntimeSymmetryParams {
            mode: SymmetryMode::Dihedral,
            boundary: SymmetryBoundary::Wrap,
            base_folds: 9.0,
            hue_span: 0.6,
            seed: 5_150,
            source_mask: SymmetrySourceMask {
                carrier: true,
                donor0: true,
                donor1: false,
                clean_history: true,
            },
            motion_mask: SymmetryMotionMask {
                slot0: true,
                slot1: false,
            },
            donors: [
                ResolvedImageTap {
                    source: ResolvedImageSource::SelectedLayer {
                        layer_id: donor,
                        saved_position: saved_position(3),
                        stage: LayerImageStage::PostLocalEffects,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                ResolvedImageTap {
                    source: ResolvedImageSource::OneBelow,
                    timing: EdgeTiming::CurrentFrame,
                },
            ],
            motion: [
                crate::motion::MotionDonor::Selected {
                    layer_id: donor,
                    saved_position: saved_position(3),
                },
                crate::motion::MotionDonor::None,
            ],
            ..RuntimeSymmetryParams::default()
        };
        let mut source = RuntimeVisualRack::empty();
        let node_id = source
            .push(RuntimeVisualNodeKind::Symmetry(authored))
            .unwrap();
        let preset =
            RackPreset::capture_runtime(&source, |id| (id == donor).then(|| saved_position(3)))
                .unwrap();

        // The target is differently routed, differently armed, and carries
        // different values.
        let live_donor = ResolvedImageTap {
            source: ResolvedImageSource::SelectedLayer {
                layer_id: other_donor,
                saved_position: saved_position(8),
                stage: LayerImageStage::PreLocalEffects,
            },
            timing: EdgeTiming::PreviousFrame,
        };
        let live_mask = SymmetrySourceMask {
            carrier: true,
            donor0: false,
            donor1: true,
            clean_history: false,
        };
        let mut target = source.clone();
        let node = target.get_mut(node_id).unwrap();
        let RuntimeVisualNodeKind::Symmetry(params) = &mut node.kind else {
            panic!("symmetry node")
        };
        params.donors = [live_donor, live_donor];
        params.source_mask = live_mask;
        params.motion = [crate::motion::MotionDonor::None; 2];
        params.motion_mask = SymmetryMotionMask {
            slot0: false,
            slot1: true,
        };
        params.base_folds = 1.0;
        params.hue_span = 0.0;
        params.seed = 1;

        preset.apply_to_runtime(&mut target).unwrap();
        let RuntimeVisualNodeKind::Symmetry(params) = target.get(node_id).unwrap().kind else {
            panic!("symmetry node")
        };
        assert_eq!(params.base_folds, 9.0);
        assert_eq!(params.hue_span, 0.6);
        assert_eq!(params.mode, SymmetryMode::Dihedral);
        assert_eq!(params.boundary, SymmetryBoundary::Wrap);
        assert_eq!(params.seed, 5_150);
        assert_eq!(
            params.donors,
            [live_donor, live_donor],
            "a preset must never retarget a live route slot"
        );
        assert_eq!(
            params.source_mask, live_mask,
            "a preset must never arm a source the operator never armed"
        );
        assert_eq!(
            params.motion_mask,
            SymmetryMotionMask {
                slot0: false,
                slot1: true
            }
        );

        // The saved twin of a motion route persists only its saved position, so
        // a captured preset can never carry a process identity.
        let captured = SavedMotionDonor::from_runtime(authored.motion[0], &mut |id| {
            (id == donor).then(|| saved_position(3))
        });
        assert_eq!(
            captured,
            SavedMotionDonor::Selected {
                saved_position: saved_position(3)
            }
        );
    }

    #[test]
    fn topology_mismatch_rejects_without_partial_apply() {
        let mut saved = VisualRack::synthetic_legacy(LegacyRackScope::Layer);
        saved
            .push(crate::visual_rack::VisualNodeKind::DigitalColor(
                DigitalColorParams {
                    brightness: 0.2,
                    ..DigitalColorParams::default()
                },
            ))
            .unwrap();
        let preset = RackPreset::from_saved(saved);
        let mut target = RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer);
        let before = target.clone();
        assert_eq!(
            preset.apply_to_runtime(&mut target),
            Err(PresetError::IncompatibleTarget)
        );
        assert_eq!(target, before);
    }

    #[test]
    fn matte_preset_preserves_route_while_applying_channel_and_values() {
        let mut matte = LayerMatte {
            enabled: true,
            input: ImageInput::SelectedLayer {
                layer_id: layer_id(55),
                stage: LayerImageStage::PreLocalEffects,
            },
            channel: LayerMatteChannel::Alpha,
            invert: false,
            amount: 1.0,
            threshold: 0.5,
            softness: 0.1,
        };
        let values = MattePreset {
            enabled: true,
            channel: PresetMatteChannel::Luma,
            invert: true,
            amount: 0.2,
            threshold: 0.7,
            softness: 0.3,
        };
        values.apply_to_layer(&mut matte);
        assert!(matches!(
            matte.input,
            ImageInput::SelectedLayer { layer_id: id, stage: LayerImageStage::PreLocalEffects }
                if id == layer_id(55)
        ));
        assert_eq!(matte.channel, LayerMatteChannel::Luma);
        assert!(matte.invert);
        assert_eq!(matte.amount, 0.2);

        let mut saved = LayerMatteConfig {
            enabled: true,
            input: SavedImageInput::SelectedLayer {
                layer_position: saved_position(9),
                stage: LayerImageStage::PostLocalEffects,
            },
            ..LayerMatteConfig::default()
        };
        values.apply_to_saved_layer(&mut saved);
        assert!(matches!(
            saved.input,
            SavedImageInput::SelectedLayer { layer_position, .. }
                if layer_position == saved_position(9)
        ));

        let mut group_matte = ImageMatte {
            tap: crate::visual_rack::SavedImageTap {
                source: crate::visual_rack::SavedImageSource::OneBelow,
                timing: EdgeTiming::PreviousFrame,
            },
            ..ImageMatte::default()
        };
        let group_route = group_matte.tap;
        values.apply_to_saved_group(&mut group_matte).unwrap();
        assert_eq!(group_matte.tap, group_route);
        assert_eq!(group_matte.channel, RackMatteChannel::Luma);
        assert!(group_matte.invert);
    }

    #[test]
    fn group_preset_preserves_group_id_name_members_and_routing_topology() {
        let id = crate::visual_rack::GroupId::new(7).unwrap();
        let members =
            crate::composition::GroupMembers::try_from_vec(vec![saved_position(2)]).unwrap();
        let mut group = Group {
            id,
            name: crate::composition::GroupName::new("Target").unwrap(),
            members,
            opacity: 1.0,
            transform: SpatialTransform::default(),
            rack: VisualRack::empty(),
            matte: None,
            solo: false,
            bypass: false,
            bus: BusAssignment::Program,
        };
        let mut sampled = group.clone();
        sampled.opacity = 0.25;
        sampled.solo = true;
        sampled.bus = BusAssignment::A;
        let preset = GroupPreset::from_saved(&sampled);
        let name = group.name.clone();
        let members = group.members.clone();
        preset.apply_to_saved(&mut group).unwrap();
        assert_eq!(group.id, id);
        assert_eq!(group.name, name);
        assert_eq!(group.members, members);
        assert_eq!(group.opacity, 0.25);
        assert!(group.solo);
        assert_eq!(group.bus, BusAssignment::A);
    }

    #[test]
    fn library_round_trip_is_versioned_bounded_and_monotonic() {
        let mut library = PresetLibrary::default();
        let first = library
            .insert(
                "Position",
                PresetPayload::Transform(SpatialTransform::default()),
            )
            .unwrap();
        let second = library
            .insert("Matte", PresetPayload::Matte(MattePreset::default()))
            .unwrap();
        assert!(second.get() > first.get());
        library.remove(first).unwrap();
        let yaml = library.to_yaml().unwrap();
        let restored = PresetLibrary::from_yaml(&yaml).unwrap();
        assert!(restored.get(first).is_none());
        assert_eq!(restored.get(second).unwrap().name, "Matte");
        let third = {
            let mut restored = restored;
            restored
                .insert(
                    "Third",
                    PresetPayload::Transform(SpatialTransform::default()),
                )
                .unwrap()
        };
        assert!(third.get() > second.get());
    }

    #[test]
    fn preset_library_persistence_defaults_safely_and_replaces_atomically() {
        use crate::controller_profile::PersistedDocumentLoadStatus;

        let path = test_library_path("atomic");
        fs::remove_dir(path.parent().unwrap()).unwrap();
        let missing = load_preset_library_or_default(&path);
        assert_eq!(missing.path, path);
        assert_eq!(missing.library, PresetLibrary::default());
        assert_eq!(missing.status, PersistedDocumentLoadStatus::DefaultMissing);

        let mut first = PresetLibrary::default();
        first
            .insert(
                "Identity",
                PresetPayload::Transform(SpatialTransform::default()),
            )
            .unwrap();
        save_preset_library_atomic(&first, &path).unwrap();
        let loaded = load_preset_library_or_default(&path);
        assert_eq!(loaded.library, first);
        assert_eq!(loaded.status, PersistedDocumentLoadStatus::Loaded);

        let mut replacement = first.clone();
        replacement
            .insert("Matte", PresetPayload::Matte(MattePreset::default()))
            .unwrap();
        save_preset_library_atomic(&replacement, &path).unwrap();
        assert_eq!(
            PresetLibrary::from_yaml(&fs::read_to_string(&path).unwrap()).unwrap(),
            replacement
        );

        let published = fs::read(&path).unwrap();
        let mut invalid = replacement;
        invalid.schema_version = PRESET_SCHEMA_VERSION + 1;
        assert_eq!(
            save_preset_library_atomic(&invalid, &path),
            Err(PresetError::UnsupportedSchemaVersion(
                PRESET_SCHEMA_VERSION + 1
            ))
        );
        assert_eq!(fs::read(&path).unwrap(), published);
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
        remove_test_library(&path);
    }

    #[test]
    fn preset_library_load_rejects_hostile_files_without_rewriting_them() {
        use crate::controller_profile::PersistedDocumentLoadStatus;

        let path = test_library_path("hostile");
        fs::write(&path, vec![b'x'; PRESET_MAX_LIBRARY_BYTES + 1]).unwrap();
        let oversized_bytes = fs::read(&path).unwrap();
        let oversized = load_preset_library_or_default(&path);
        assert_eq!(oversized.library, PresetLibrary::default());
        assert!(matches!(
            oversized.status,
            PersistedDocumentLoadStatus::DefaultInvalid(_)
        ));
        assert_eq!(fs::read(&path).unwrap(), oversized_bytes);

        fs::write(&path, [0xff, 0xfe, 0xfd]).unwrap();
        let invalid_utf8 = fs::read(&path).unwrap();
        let invalid = load_preset_library_or_default(&path);
        assert_eq!(invalid.library, PresetLibrary::default());
        assert!(matches!(
            invalid.status,
            PersistedDocumentLoadStatus::DefaultInvalid(_)
        ));
        assert_eq!(fs::read(&path).unwrap(), invalid_utf8);

        fs::write(
            &path,
            "schema_version: 1\nnext_preset_id: 1\nnext_created_ordinal: 1\npresets: []\nunexpected: true\n",
        )
        .unwrap();
        let unknown_field = fs::read(&path).unwrap();
        let unknown = load_preset_library_or_default(&path);
        assert_eq!(unknown.library, PresetLibrary::default());
        assert!(matches!(
            unknown.status,
            PersistedDocumentLoadStatus::DefaultInvalid(_)
        ));
        assert_eq!(fs::read(&path).unwrap(), unknown_field);
        remove_test_library(&path);
    }

    #[test]
    fn default_preset_library_path_is_separate_platform_state() {
        let path = default_preset_library_path();
        assert_eq!(
            path.parent(),
            Some(crate::controller_profile::default_control_state_dir().as_path())
        );
        assert_eq!(
            path.file_name(),
            Some(std::ffi::OsStr::new(PRESET_LIBRARY_FILE_NAME))
        );
        assert_ne!(
            path,
            crate::controller_profile::default_controller_profile_path()
        );
    }

    #[test]
    fn controller_profile_and_stage_map_presets_are_exact_typed_documents() {
        let profile = crate::controller_profile::ControllerProfileDocument::default();
        let stage_map = crate::stage_map::StageMap::default();
        let mut library = PresetLibrary::default();
        let profile_id = library
            .insert(
                "Controller",
                PresetPayload::ControllerProfile(profile.clone()),
            )
            .unwrap();
        let stage_map_id = library
            .insert("Venue", PresetPayload::StageMap(stage_map.clone()))
            .unwrap();

        let restored = PresetLibrary::from_yaml(&library.to_yaml().unwrap()).unwrap();
        assert!(matches!(
            &restored.get(profile_id).unwrap().payload,
            PresetPayload::ControllerProfile(document) if document == &profile
        ));
        assert!(matches!(
            &restored.get(stage_map_id).unwrap().payload,
            PresetPayload::StageMap(document) if document == &stage_map
        ));
        assert_eq!(PresetKind::ControllerProfile.key(), "controller_profile");
        assert_eq!(PresetKind::StageMap.key(), "stage_map");
    }

    #[test]
    fn hostile_serde_rejects_versions_ids_names_and_noncanonical_values() {
        let yaml = r#"
schema_version: 9
next_preset_id: 2
next_created_ordinal: 2
presets: []
"#;
        assert!(PresetLibrary::from_yaml(yaml).is_err());
        assert!(PresetLibrary::from_yaml(
            "schema_version: 1\nnext_preset_id: 1\nnext_created_ordinal: 1\npresets: []\nopaque: true\n"
        )
        .is_err());
        let mut library = PresetLibrary::default();
        assert!(matches!(
            library.insert(
                " bad ",
                PresetPayload::Transform(SpatialTransform::default())
            ),
            Err(PresetError::InvalidName)
        ));
        let hostile = SpatialTransform {
            position: [f32::NAN, 0.0],
            ..SpatialTransform::default()
        };
        assert!(matches!(
            library.insert("Hostile", PresetPayload::Transform(hostile)),
            Err(PresetError::NonCanonicalTransform)
        ));
        assert!(PresetLibrary::from_yaml(&"x".repeat(PRESET_MAX_LIBRARY_BYTES + 1)).is_err());

        let zero_ordinal = PresetLibrary {
            schema_version: PRESET_SCHEMA_VERSION,
            next_preset_id: 2,
            next_created_ordinal: 1,
            presets: vec![ScopedPreset {
                schema_version: PRESET_SCHEMA_VERSION,
                id: PresetId::new(1).unwrap(),
                name: "Invalid ordinal".into(),
                created_ordinal: 0,
                payload: PresetPayload::Transform(SpatialTransform::default()),
            }],
        };
        let zero_ordinal_yaml = serde_yaml::to_string(&zero_ordinal).unwrap();
        assert!(PresetLibrary::from_yaml(&zero_ordinal_yaml).is_err());

        let payload =
            serde_json::to_value(PresetPayload::Transform(SpatialTransform::default())).unwrap();
        let presets = (1..=PRESET_MAX_COUNT + 1)
            .map(|id| {
                serde_json::json!({
                    "schema_version": PRESET_SCHEMA_VERSION,
                    "id": id,
                    "name": format!("Preset {id}"),
                    "created_ordinal": id,
                    "payload": payload.clone(),
                })
            })
            .collect::<Vec<_>>();
        let too_many = serde_json::json!({
            "schema_version": PRESET_SCHEMA_VERSION,
            "next_preset_id": PRESET_MAX_COUNT + 2,
            "next_created_ordinal": PRESET_MAX_COUNT + 2,
            "presets": presets,
        })
        .to_string();
        assert!(PresetLibrary::from_yaml(&too_many).is_err());
    }
}
