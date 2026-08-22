//! Versioned, bounded controller profiles and pure MIDI message decoding.
//!
//! Profiles are deliberately independent of PatchState. Saved layer targets
//! use patch positions and resolve atomically to runtime stable identities;
//! saved Scene actions use authored [`SceneId`] values directly. The resolved
//! profile then survives ordinary reorder without retargeting. Hardware
//! callbacks never use this module directly: they put bounded raw messages on
//! the MIDI queue and the host decodes them later.

#![allow(
    dead_code,
    reason = "M5 profile/runtime seams are consumed by Main and Web after API freeze"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::image_routing::StableLayerId;
use crate::performance::{SavedLayerPosition, SceneId};
use crate::visual_rack::{GroupId, NodeId};

pub const CONTROLLER_PROFILE_VERSION: u16 = 1;
pub const CONTROLLER_PROFILE_FILE_NAME: &str = "controller_profile.json";
pub const CONTROLLER_PROFILE_MAX_BYTES: usize = 256 * 1024;
pub const CONTROLLER_PROFILE_MAX_NAME_BYTES: usize = 96;
pub const CONTROLLER_DEVICE_NAME_MAX_BYTES: usize = 256;
pub const CONTROLLER_PROFILE_MAX_BINDINGS: usize = 256;
pub const CONTROLLER_PROFILE_MAX_FEEDBACK_BINDINGS: usize = 256;
pub const CONTROLLER_RELATIVE_STEP_MAX: f32 = 1.0;
/// Small envelope allowance above the document cap for the browser-safe
/// tagged import/export request. The profile nested inside remains subject to
/// `CONTROLLER_PROFILE_MAX_BYTES` when it is validated/exported.
pub const CONTROLLER_PROFILE_ACTION_MAX_BYTES: usize = CONTROLLER_PROFILE_MAX_BYTES + 1_024;
pub const PERSISTED_DOCUMENT_STATUS_MAX_BYTES: usize = 512;

static DOCUMENT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistedDocumentLoadStatus {
    Loaded,
    DefaultMissing,
    DefaultInvalid(String),
    DefaultIo(String),
}

impl PersistedDocumentLoadStatus {
    pub const fn used_default(&self) -> bool {
        !matches!(self, Self::Loaded)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ControllerProfileLoad {
    pub path: PathBuf,
    pub document: ControllerProfileDocument,
    pub status: PersistedDocumentLoadStatus,
}

pub fn default_controller_profile_path() -> PathBuf {
    default_control_state_dir().join(CONTROLLER_PROFILE_FILE_NAME)
}

pub fn load_default_controller_profile() -> ControllerProfileLoad {
    load_controller_profile_or_default(&default_controller_profile_path())
}

pub fn load_controller_profile_or_default(path: &Path) -> ControllerProfileLoad {
    let (document, status) = match read_bounded_document(path, CONTROLLER_PROFILE_MAX_BYTES) {
        Ok(Some(bytes)) => match ControllerProfileDocument::from_json_bytes(&bytes) {
            Ok(document) => (document, PersistedDocumentLoadStatus::Loaded),
            Err(error) => (
                ControllerProfileDocument::default(),
                PersistedDocumentLoadStatus::DefaultInvalid(bounded_status(error.to_string())),
            ),
        },
        Ok(None) => (
            ControllerProfileDocument::default(),
            PersistedDocumentLoadStatus::DefaultMissing,
        ),
        Err(BoundedDocumentReadError::TooLarge(bytes)) => (
            ControllerProfileDocument::default(),
            PersistedDocumentLoadStatus::DefaultInvalid(bounded_status(format!(
                "document is {bytes} bytes; limit is {CONTROLLER_PROFILE_MAX_BYTES}"
            ))),
        ),
        Err(BoundedDocumentReadError::Io(error)) => (
            ControllerProfileDocument::default(),
            PersistedDocumentLoadStatus::DefaultIo(bounded_status(error)),
        ),
    };
    ControllerProfileLoad {
        path: path.to_path_buf(),
        document,
        status,
    }
}

pub fn save_controller_profile_atomic(
    document: &ControllerProfileDocument,
    path: &Path,
) -> Result<(), ControllerProfileError> {
    let bytes = document.to_json_bytes()?;
    write_atomic_document(path, &bytes)
        .map_err(|error| ControllerProfileError::Io(bounded_status(error.to_string())))
}

pub fn save_default_controller_profile_atomic(
    document: &ControllerProfileDocument,
) -> Result<PathBuf, ControllerProfileError> {
    let path = default_controller_profile_path();
    save_controller_profile_atomic(document, &path)?;
    Ok(path)
}

/// Read a profile chosen by the native host without falling back to defaults.
///
/// Path authority deliberately stays at the caller: Main may pass the result
/// of a native picker here, while browser actions can carry only the data-only
/// [`ControllerProfileAction`] below.
pub fn read_controller_profile_import(
    path: &Path,
) -> Result<ControllerProfileDocument, ControllerProfileError> {
    match read_bounded_document(path, CONTROLLER_PROFILE_MAX_BYTES) {
        Ok(Some(bytes)) => ControllerProfileDocument::from_json_bytes(&bytes),
        Ok(None) => Err(ControllerProfileError::Io(
            "selected controller profile does not exist".to_string(),
        )),
        Err(BoundedDocumentReadError::TooLarge(bytes)) => Err(
            ControllerProfileError::DocumentBytes(usize::try_from(bytes).unwrap_or(usize::MAX)),
        ),
        Err(BoundedDocumentReadError::Io(error)) => {
            Err(ControllerProfileError::Io(bounded_status(error)))
        }
    }
}

/// The per-user state root. The ladder itself lives in [`crate::host_paths`]
/// so the TLS, proxy-cache, stage, and recovery callers cannot drift apart.
fn default_state_dir_from(
    local_app_data: Option<&OsStr>,
    xdg_state_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> PathBuf {
    crate::host_paths::state_root_from(local_app_data, xdg_state_home, home)
}

pub(crate) fn default_control_state_dir() -> PathBuf {
    let local_app_data = std::env::var_os("LOCALAPPDATA");
    let xdg_state_home = std::env::var_os("XDG_STATE_HOME");
    let home = std::env::var_os("HOME");
    default_state_dir_from(
        local_app_data.as_deref(),
        xdg_state_home.as_deref(),
        home.as_deref(),
    )
}

pub(crate) fn bounded_status(value: String) -> String {
    let mut output = String::new();
    for character in value.chars().filter(|character| !character.is_control()) {
        if output.len().saturating_add(character.len_utf8()) > PERSISTED_DOCUMENT_STATUS_MAX_BYTES {
            break;
        }
        output.push(character);
    }
    output
}

pub(crate) enum BoundedDocumentReadError {
    TooLarge(u64),
    Io(String),
}

pub(crate) fn read_bounded_document(
    path: &Path,
    max_bytes: usize,
) -> Result<Option<Vec<u8>>, BoundedDocumentReadError> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(BoundedDocumentReadError::Io(error.to_string())),
    };
    let advertised = file
        .metadata()
        .map_err(|error| BoundedDocumentReadError::Io(error.to_string()))?
        .len();
    if advertised > max_bytes as u64 {
        return Err(BoundedDocumentReadError::TooLarge(advertised));
    }
    let mut bytes = Vec::with_capacity(advertised as usize);
    file.take(max_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| BoundedDocumentReadError::Io(error.to_string()))?;
    if bytes.len() > max_bytes {
        return Err(BoundedDocumentReadError::TooLarge(bytes.len() as u64));
    }
    Ok(Some(bytes))
}

pub(crate) fn write_atomic_document(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "document path has no file name",
        )
    })?;
    let parent = path
        .parent()
        .filter(|candidate| !candidate.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let mut temporary = None;
    for _ in 0..16 {
        let ordinal = DOCUMENT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = OsString::from(".");
        temporary_name.push(file_name);
        temporary_name.push(format!(".tmp-{}-{ordinal}", std::process::id()));
        let candidate = parent.join(temporary_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    let (temporary_path, mut temporary_file) = temporary.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve a unique temporary document",
        )
    })?;
    let publication = (|| {
        temporary_file.write_all(bytes)?;
        temporary_file.sync_all()?;
        drop(temporary_file);
        atomic_replace(&temporary_path, path)?;
        sync_parent(path)
    })();
    if publication.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    publication
}

fn sync_parent(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if let Ok(directory) = File::open(parent) {
            directory.sync_all()?;
        }
    }
    Ok(())
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    if unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    } == 0
    {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutomationOrigin {
    Midi,
    Osc(SocketAddr),
    HostAutomation,
}

impl AutomationOrigin {
    /// Controller and protocol mutations are automation, never a manual edit
    /// boundary in the authored history journal.
    pub const fn records_manual_history(self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutomationValue {
    Absolute(f32),
    Delta(f32),
    Trigger,
    Gate(bool),
}

impl AutomationValue {
    pub fn finite_bounded(self) -> Option<Self> {
        match self {
            Self::Absolute(value) if value.is_finite() => {
                Some(Self::Absolute(value.clamp(0.0, 1.0)))
            }
            Self::Delta(value) if value.is_finite() => Some(Self::Delta(value.clamp(-1.0, 1.0))),
            Self::Trigger => Some(Self::Trigger),
            Self::Gate(value) => Some(Self::Gate(value)),
            Self::Absolute(_) | Self::Delta(_) => None,
        }
    }
}

/// Closed cross-protocol vocabulary. Main performs the final typed adapter to
/// application actions; neither MIDI profiles nor OSC packets carry arbitrary
/// parameter strings through that boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlParameter {
    Value,
    Amount,
    Wet,
    Bypass,
    Enabled,
    Opacity,
    Speed,
    Rate,
    PositionX,
    PositionY,
    ScaleX,
    ScaleY,
    Rotation,
    Brightness,
    Contrast,
    Saturation,
    Hue,
    Threshold,
    Softness,
    Visibility,
    Paused,
    Solo,
    BusCrossfade,
    ProgramFreeze,
    MediaFreeze,
    Blackout,
    Play,
    SeekNormalized,
    Bpm,
    TapTempo,
    Downbeat,
    ClearMotionMemory,
    ClearTemporalMemory,
    /// Gesture-field etching surface. A controller has no notion of a stroke,
    /// so it drives four independent scalars and the stroke machine lives in
    /// `gesture::GestureControlSurface`. Appended, never renumbered: MIDI
    /// profiles and OSC paths both address these by their snake_case key.
    GestureX,
    GestureY,
    GesturePressure,
    GestureContact,
    /// B10 bend pads. Momentary engine surfaces on the gesture-contact law: a
    /// profile binds a note or button with `Gate` mode and the press/release
    /// edges drive the pad directly, never a `WebAction`. Appended, never
    /// renumbered.
    Bend1,
    Bend2,
    Bend3,
    Bend4,
    Bend5,
    Bend6,
}

impl ControlParameter {
    pub const fn key(self) -> &'static str {
        match self {
            Self::Value => "value",
            Self::Amount => "amount",
            Self::Wet => "wet",
            Self::Bypass => "bypass",
            Self::Enabled => "enabled",
            Self::Opacity => "opacity",
            Self::Speed => "speed",
            Self::Rate => "rate",
            Self::PositionX => "position_x",
            Self::PositionY => "position_y",
            Self::ScaleX => "scale_x",
            Self::ScaleY => "scale_y",
            Self::Rotation => "rotation",
            Self::Brightness => "brightness",
            Self::Contrast => "contrast",
            Self::Saturation => "saturation",
            Self::Hue => "hue",
            Self::Threshold => "threshold",
            Self::Softness => "softness",
            Self::Visibility => "visibility",
            Self::Paused => "paused",
            Self::Solo => "solo",
            Self::BusCrossfade => "bus_crossfade",
            Self::ProgramFreeze => "program_freeze",
            Self::MediaFreeze => "media_freeze",
            Self::Blackout => "blackout",
            Self::Play => "play",
            Self::SeekNormalized => "seek_normalized",
            Self::Bpm => "bpm",
            Self::TapTempo => "tap_tempo",
            Self::Downbeat => "downbeat",
            Self::ClearMotionMemory => "clear_motion_memory",
            Self::ClearTemporalMemory => "clear_temporal_memory",
            Self::GestureX => "gesture_x",
            Self::GestureY => "gesture_y",
            Self::GesturePressure => "gesture_pressure",
            Self::GestureContact => "gesture_contact",
            Self::Bend1 => "bend1",
            Self::Bend2 => "bend2",
            Self::Bend3 => "bend3",
            Self::Bend4 => "bend4",
            Self::Bend5 => "bend5",
            Self::Bend6 => "bend6",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "value" => Self::Value,
            "amount" => Self::Amount,
            "wet" => Self::Wet,
            "bypass" => Self::Bypass,
            "enabled" => Self::Enabled,
            "opacity" => Self::Opacity,
            "speed" => Self::Speed,
            "rate" => Self::Rate,
            "position_x" => Self::PositionX,
            "position_y" => Self::PositionY,
            "scale_x" => Self::ScaleX,
            "scale_y" => Self::ScaleY,
            "rotation" => Self::Rotation,
            "brightness" => Self::Brightness,
            "contrast" => Self::Contrast,
            "saturation" => Self::Saturation,
            "hue" => Self::Hue,
            "threshold" => Self::Threshold,
            "softness" => Self::Softness,
            "visibility" => Self::Visibility,
            "paused" => Self::Paused,
            "solo" => Self::Solo,
            "bus_crossfade" => Self::BusCrossfade,
            "program_freeze" => Self::ProgramFreeze,
            "media_freeze" => Self::MediaFreeze,
            "blackout" => Self::Blackout,
            "play" => Self::Play,
            "seek_normalized" => Self::SeekNormalized,
            "bpm" => Self::Bpm,
            "tap_tempo" => Self::TapTempo,
            "downbeat" => Self::Downbeat,
            "clear_motion_memory" => Self::ClearMotionMemory,
            "clear_temporal_memory" => Self::ClearTemporalMemory,
            "gesture_x" => Self::GestureX,
            "gesture_y" => Self::GestureY,
            "gesture_pressure" => Self::GesturePressure,
            "gesture_contact" => Self::GestureContact,
            "bend1" => Self::Bend1,
            "bend2" => Self::Bend2,
            "bend3" => Self::Bend3,
            "bend4" => Self::Bend4,
            "bend5" => Self::Bend5,
            "bend6" => Self::Bend6,
            _ => return None,
        })
    }

    /// The B10 bend pad this parameter drives, if any.
    pub const fn bend_index(self) -> Option<usize> {
        match self {
            Self::Bend1 => Some(0),
            Self::Bend2 => Some(1),
            Self::Bend3 => Some(2),
            Self::Bend4 => Some(3),
            Self::Bend5 => Some(4),
            Self::Bend6 => Some(5),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeNodeScope {
    Master,
    Layer(StableLayerId),
    Group(GroupId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuntimeControlAddress {
    LegacyMidiSlot(u8),
    Master(ControlParameter),
    Layer {
        layer_id: StableLayerId,
        parameter: ControlParameter,
    },
    Group {
        group_id: GroupId,
        parameter: ControlParameter,
    },
    Node {
        scope: RuntimeNodeScope,
        node_id: NodeId,
        parameter: ControlParameter,
    },
    Transport(ControlParameter),
    /// Prepare one authored Scene by stable ID. This is an action endpoint,
    /// not a scalar parameter; MIDI resolves it to a rising-edge trigger and
    /// OSC treats each asserted message as one pulse.
    ScenePrepare {
        scene_id: SceneId,
    },
    /// Trigger one authored Scene by stable ID using its authored trigger
    /// mode. Keeping the ID in the runtime address prevents reorder from
    /// redirecting a physical control to another Scene.
    SceneTrigger {
        scene_id: SceneId,
    },
}

impl RuntimeControlAddress {
    pub const fn is_scene_action(self) -> bool {
        matches!(self, Self::ScenePrepare { .. } | Self::SceneTrigger { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum SavedNodeScope {
    Master,
    Layer { position: SavedLayerPosition },
    Group { group_id: GroupId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum SavedControlAddress {
    LegacyMidiSlot {
        slot: u8,
    },
    Master {
        parameter: ControlParameter,
    },
    Layer {
        position: SavedLayerPosition,
        parameter: ControlParameter,
    },
    Group {
        group_id: GroupId,
        parameter: ControlParameter,
    },
    Node {
        node_scope: SavedNodeScope,
        node_id: NodeId,
        parameter: ControlParameter,
    },
    Transport {
        parameter: ControlParameter,
    },
    ScenePrepare {
        scene_id: SceneId,
    },
    SceneTrigger {
        scene_id: SceneId,
    },
}

impl SavedControlAddress {
    pub const fn is_scene_action(self) -> bool {
        matches!(self, Self::ScenePrepare { .. } | Self::SceneTrigger { .. })
    }

    fn resolve(
        self,
        layer_at_position: &mut impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        scene_exists: &mut impl FnMut(SceneId) -> bool,
    ) -> Result<RuntimeControlAddress, ControllerProfileError> {
        Ok(match self {
            Self::LegacyMidiSlot { slot } if slot < 4 => {
                RuntimeControlAddress::LegacyMidiSlot(slot)
            }
            Self::LegacyMidiSlot { slot } => {
                return Err(ControllerProfileError::LegacySlot(slot));
            }
            Self::Master { parameter } => RuntimeControlAddress::Master(parameter),
            Self::Layer {
                position,
                parameter,
            } => RuntimeControlAddress::Layer {
                layer_id: layer_at_position(position)
                    .ok_or(ControllerProfileError::MissingLayer(position))?,
                parameter,
            },
            Self::Group {
                group_id,
                parameter,
            } => RuntimeControlAddress::Group {
                group_id,
                parameter,
            },
            Self::Node {
                node_scope,
                node_id,
                parameter,
            } => RuntimeControlAddress::Node {
                scope: match node_scope {
                    SavedNodeScope::Master => RuntimeNodeScope::Master,
                    SavedNodeScope::Layer { position } => RuntimeNodeScope::Layer(
                        layer_at_position(position)
                            .ok_or(ControllerProfileError::MissingLayer(position))?,
                    ),
                    SavedNodeScope::Group { group_id } => RuntimeNodeScope::Group(group_id),
                },
                node_id,
                parameter,
            },
            Self::Transport { parameter } => RuntimeControlAddress::Transport(parameter),
            Self::ScenePrepare { scene_id } if scene_exists(scene_id) => {
                RuntimeControlAddress::ScenePrepare { scene_id }
            }
            Self::SceneTrigger { scene_id } if scene_exists(scene_id) => {
                RuntimeControlAddress::SceneTrigger { scene_id }
            }
            Self::ScenePrepare { scene_id } | Self::SceneTrigger { scene_id } => {
                return Err(ControllerProfileError::MissingScene(scene_id));
            }
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum MidiDeviceSelector {
    #[default]
    FirstAvailable,
    Exact {
        name: String,
    },
}

impl MidiDeviceSelector {
    pub fn matches(&self, name: &str) -> bool {
        match self {
            Self::FirstAvailable => true,
            Self::Exact { name: expected } => expected == name,
        }
    }

    fn validate(&self) -> Result<(), ControllerProfileError> {
        if let Self::Exact { name } = self {
            if name.is_empty() || name.len() > CONTROLLER_DEVICE_NAME_MAX_BYTES {
                return Err(ControllerProfileError::DeviceName);
            }
            if name.chars().any(char::is_control) {
                return Err(ControllerProfileError::DeviceName);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum MidiChannelFilter {
    #[default]
    Omni,
    Exact {
        channel: u8,
    },
}

impl MidiChannelFilter {
    pub const fn matches(self, zero_based_channel: u8) -> bool {
        match self {
            Self::Omni => true,
            Self::Exact { channel } => channel == zero_based_channel + 1,
        }
    }

    fn validate(self) -> Result<(), ControllerProfileError> {
        match self {
            Self::Exact { channel } if !(1..=16).contains(&channel) => {
                Err(ControllerProfileError::Channel(channel))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MidiInputSource {
    ControlChange { controller: u8 },
    Note { note: u8 },
}

impl MidiInputSource {
    fn validate(self) -> Result<(), ControllerProfileError> {
        let value = match self {
            Self::ControlChange { controller } => controller,
            Self::Note { note } => note,
        };
        if value <= 127 {
            Ok(())
        } else {
            Err(ControllerProfileError::MidiData(value))
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MidiValueEncoding {
    #[default]
    Absolute,
    RelativeTwosComplement,
    RelativeBinaryOffset,
    RelativeSignMagnitude,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MidiButtonMode {
    Momentary,
    Toggle,
    Gate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MidiOutputMessage {
    ControlChange { channel: u8, controller: u8 },
    Note { channel: u8, note: u8 },
}

impl MidiOutputMessage {
    pub fn bytes(self, normalized: f32) -> Result<[u8; 3], ControllerProfileError> {
        let value = if normalized.is_finite() {
            (normalized.clamp(0.0, 1.0) * 127.0).round() as u8
        } else {
            return Err(ControllerProfileError::NonFinite);
        };
        let (channel, data, family) = match self {
            Self::ControlChange {
                channel,
                controller,
            } => (channel, controller, 0xb0),
            Self::Note { channel, note } => (channel, note, 0x90),
        };
        if !(1..=16).contains(&channel) {
            return Err(ControllerProfileError::Channel(channel));
        }
        if data > 127 {
            return Err(ControllerProfileError::MidiData(data));
        }
        Ok([family | (channel - 1), data, value])
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ControllerBinding {
    pub id: u16,
    pub source: MidiInputSource,
    pub channel: Option<MidiChannelFilter>,
    pub encoding: MidiValueEncoding,
    pub button_mode: Option<MidiButtonMode>,
    pub press_threshold: u8,
    pub relative_step: f32,
    pub target: SavedControlAddress,
    pub feedback: Option<MidiOutputMessage>,
}

impl Default for ControllerBinding {
    fn default() -> Self {
        Self {
            id: 1,
            source: MidiInputSource::ControlChange { controller: 1 },
            channel: None,
            encoding: MidiValueEncoding::Absolute,
            button_mode: None,
            press_threshold: 64,
            relative_step: 1.0 / 127.0,
            target: SavedControlAddress::LegacyMidiSlot { slot: 0 },
            feedback: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ControllerProfileDocument {
    pub version: u16,
    pub name: String,
    pub input: MidiDeviceSelector,
    pub output: MidiDeviceSelector,
    pub channel: MidiChannelFilter,
    pub bindings: Vec<ControllerBinding>,
}

impl Default for ControllerProfileDocument {
    fn default() -> Self {
        Self::legacy_four_cc()
    }
}

impl ControllerProfileDocument {
    /// Compatibility profile for the historical four modulation slots.
    pub fn legacy_four_cc() -> Self {
        Self {
            version: CONTROLLER_PROFILE_VERSION,
            name: "Legacy four CC".to_string(),
            input: MidiDeviceSelector::FirstAvailable,
            output: MidiDeviceSelector::FirstAvailable,
            channel: MidiChannelFilter::Omni,
            bindings: (0_u8..4)
                .map(|slot| ControllerBinding {
                    id: u16::from(slot) + 1,
                    source: MidiInputSource::ControlChange {
                        controller: slot + 1,
                    },
                    target: SavedControlAddress::LegacyMidiSlot { slot },
                    ..ControllerBinding::default()
                })
                .collect(),
        }
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ControllerProfileError> {
        if bytes.len() > CONTROLLER_PROFILE_MAX_BYTES {
            return Err(ControllerProfileError::DocumentBytes(bytes.len()));
        }
        let document: Self = serde_json::from_slice(bytes)
            .map_err(|error| ControllerProfileError::Json(error.to_string()))?;
        document.validate()?;
        Ok(document)
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, ControllerProfileError> {
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| ControllerProfileError::Json(error.to_string()))?;
        if bytes.len() > CONTROLLER_PROFILE_MAX_BYTES {
            return Err(ControllerProfileError::DocumentBytes(bytes.len()));
        }
        Ok(bytes)
    }

    pub fn validate(&self) -> Result<(), ControllerProfileError> {
        if self.version != CONTROLLER_PROFILE_VERSION {
            return Err(ControllerProfileError::Version(self.version));
        }
        if self.name.is_empty()
            || self.name.len() > CONTROLLER_PROFILE_MAX_NAME_BYTES
            || self.name.chars().any(char::is_control)
        {
            return Err(ControllerProfileError::ProfileName);
        }
        self.input.validate()?;
        self.output.validate()?;
        self.channel.validate()?;
        if self.bindings.len() > CONTROLLER_PROFILE_MAX_BINDINGS {
            return Err(ControllerProfileError::BindingCount(self.bindings.len()));
        }
        let mut ids = BTreeSet::new();
        let mut feedback = 0_usize;
        for binding in &self.bindings {
            if binding.id == 0 || !ids.insert(binding.id) {
                return Err(ControllerProfileError::BindingId(binding.id));
            }
            binding.source.validate()?;
            if !(1..=127).contains(&binding.press_threshold) {
                return Err(ControllerProfileError::PressThreshold(binding.id));
            }
            if let Some(channel) = binding.channel {
                channel.validate()?;
            }
            if !binding.relative_step.is_finite()
                || !(0.0..=CONTROLLER_RELATIVE_STEP_MAX).contains(&binding.relative_step)
                || (binding.encoding != MidiValueEncoding::Absolute && binding.relative_step <= 0.0)
            {
                return Err(ControllerProfileError::RelativeStep(binding.id));
            }
            if matches!(binding.source, MidiInputSource::Note { .. })
                && binding.button_mode.is_none()
            {
                return Err(ControllerProfileError::NoteNeedsButton(binding.id));
            }
            if binding.button_mode.is_some() && binding.encoding != MidiValueEncoding::Absolute {
                return Err(ControllerProfileError::ButtonEncoding(binding.id));
            }
            if binding.target.is_scene_action()
                && binding.button_mode != Some(MidiButtonMode::Momentary)
            {
                return Err(ControllerProfileError::SceneActionNeedsMomentary(
                    binding.id,
                ));
            }
            if let Some(output) = binding.feedback {
                output.bytes(0.0)?;
                feedback += 1;
            }
        }
        if feedback > CONTROLLER_PROFILE_MAX_FEEDBACK_BINDINGS {
            return Err(ControllerProfileError::FeedbackCount(feedback));
        }
        Ok(())
    }

    pub fn resolve(
        &self,
        layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
    ) -> Result<ResolvedControllerProfile, ControllerProfileError> {
        self.resolve_with_scenes(layer_at_position, |_| false)
    }

    /// Resolve portable targets against one immutable live world. The legacy
    /// one-closure [`Self::resolve`] seam remains available for profiles with
    /// no Scene actions; it deliberately treats every Scene as absent.
    pub fn resolve_with_scenes(
        &self,
        mut layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
        mut scene_exists: impl FnMut(SceneId) -> bool,
    ) -> Result<ResolvedControllerProfile, ControllerProfileError> {
        self.validate()?;
        let mut bindings = Vec::with_capacity(self.bindings.len());
        for binding in &self.bindings {
            bindings.push(ResolvedControllerBinding {
                id: binding.id,
                source: binding.source,
                channel: binding.channel.unwrap_or(self.channel),
                encoding: binding.encoding,
                button_mode: binding.button_mode,
                press_threshold: binding.press_threshold,
                relative_step: binding.relative_step,
                target: binding
                    .target
                    .resolve(&mut layer_at_position, &mut scene_exists)?,
                feedback: binding.feedback,
            });
        }
        Ok(ResolvedControllerProfile {
            name: self.name.clone(),
            input: self.input.clone(),
            output: self.output.clone(),
            bindings: bindings.into_boxed_slice(),
        })
    }
}

/// Data-only controller-profile command suitable for a bounded browser action
/// body. There is intentionally no path, URL, device-open, or arbitrary host
/// action variant. Native import/export chooses paths in Main and then uses
/// the same validated document/JSON seams.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControllerProfileAction {
    Import { document: ControllerProfileDocument },
    Export {},
}

impl ControllerProfileAction {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ControllerProfileError> {
        if bytes.len() > CONTROLLER_PROFILE_ACTION_MAX_BYTES {
            return Err(ControllerProfileError::ActionBytes(bytes.len()));
        }
        let action: Self = serde_json::from_slice(bytes)
            .map_err(|error| ControllerProfileError::Json(error.to_string()))?;
        if let Self::Import { document } = &action {
            // Validate the authored values and the independently bounded
            // pretty-JSON representation before Main resolves live IDs.
            let _ = document.to_json_bytes()?;
        }
        Ok(action)
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, ControllerProfileError> {
        if let Self::Import { document } = self {
            let _ = document.to_json_bytes()?;
        }
        let bytes = serde_json::to_vec(self)
            .map_err(|error| ControllerProfileError::Json(error.to_string()))?;
        if bytes.len() > CONTROLLER_PROFILE_ACTION_MAX_BYTES {
            return Err(ControllerProfileError::ActionBytes(bytes.len()));
        }
        Ok(bytes)
    }
}

/// A validated document tied to the exact resolved runtime mapping derived
/// from it. Private fields prevent a caller from accidentally pairing one
/// persisted profile with another profile's live stable identities.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedControllerProfileSwap {
    document: ControllerProfileDocument,
    runtime: ResolvedControllerProfile,
}

impl PreparedControllerProfileSwap {
    pub fn document(&self) -> &ControllerProfileDocument {
        &self.document
    }

    pub fn runtime(&self) -> &ResolvedControllerProfile {
        &self.runtime
    }

    pub fn into_parts(self) -> (ControllerProfileDocument, ResolvedControllerProfile) {
        (self.document, self.runtime)
    }
}

/// Validate and resolve a candidate completely before the host changes its
/// saved document, worker/device selection, or decoder button state. Main can
/// durably publish `prepared.document()`, apply `prepared.runtime()`, and only
/// then replace its live document/revision; every fallible semantic operation
/// has already completed at this seam.
pub fn prepare_controller_profile_swap(
    document: ControllerProfileDocument,
    layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
) -> Result<PreparedControllerProfileSwap, ControllerProfileError> {
    let runtime = document.resolve(layer_at_position)?;
    Ok(PreparedControllerProfileSwap { document, runtime })
}

/// Scene-aware counterpart used by the live host. A missing stable Scene ID
/// rejects the complete candidate before either the saved document or the
/// MIDI decoder can change.
pub fn prepare_controller_profile_swap_with_scenes(
    document: ControllerProfileDocument,
    layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
    scene_exists: impl FnMut(SceneId) -> bool,
) -> Result<PreparedControllerProfileSwap, ControllerProfileError> {
    let runtime = document.resolve_with_scenes(layer_at_position, scene_exists)?;
    Ok(PreparedControllerProfileSwap { document, runtime })
}

/// Shared byte-oriented import seam for a native picker or another bounded
/// host transport. Browser JSON should use [`ControllerProfileAction`] so its
/// lack of path authority is structural rather than a UI convention.
pub fn prepare_controller_profile_json_import(
    bytes: &[u8],
    layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
) -> Result<PreparedControllerProfileSwap, ControllerProfileError> {
    let document = ControllerProfileDocument::from_json_bytes(bytes)?;
    prepare_controller_profile_swap(document, layer_at_position)
}

pub fn prepare_controller_profile_json_import_with_scenes(
    bytes: &[u8],
    layer_at_position: impl FnMut(SavedLayerPosition) -> Option<StableLayerId>,
    scene_exists: impl FnMut(SceneId) -> bool,
) -> Result<PreparedControllerProfileSwap, ControllerProfileError> {
    let document = ControllerProfileDocument::from_json_bytes(bytes)?;
    prepare_controller_profile_swap_with_scenes(document, layer_at_position, scene_exists)
}

/// Export only the portable validated profile document. The destination path
/// is chosen and atomically published by the native host, never by a browser
/// request.
pub fn export_controller_profile_json(
    document: &ControllerProfileDocument,
) -> Result<Vec<u8>, ControllerProfileError> {
    document.to_json_bytes()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedControllerBinding {
    pub id: u16,
    pub source: MidiInputSource,
    pub channel: MidiChannelFilter,
    pub encoding: MidiValueEncoding,
    pub button_mode: Option<MidiButtonMode>,
    pub press_threshold: u8,
    pub relative_step: f32,
    pub target: RuntimeControlAddress,
    pub feedback: Option<MidiOutputMessage>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedControllerProfile {
    pub name: String,
    pub input: MidiDeviceSelector,
    pub output: MidiDeviceSelector,
    pub bindings: Box<[ResolvedControllerBinding]>,
}

impl ResolvedControllerProfile {
    pub fn legacy_four_cc() -> Self {
        ControllerProfileDocument::legacy_four_cc()
            .resolve(|_| None)
            .expect("legacy profile has no saved layer targets")
    }

    pub fn feedback_bytes(
        &self,
        address: RuntimeControlAddress,
        value: f32,
    ) -> impl Iterator<Item = ([u8; 3], u16)> + '_ {
        self.bindings.iter().filter_map(move |binding| {
            (binding.target == address)
                .then_some(binding.feedback?)
                .and_then(|message| message.bytes(value).ok())
                .map(|bytes| (bytes, binding.id))
        })
    }

    pub fn validate(&self) -> Result<(), ControllerProfileError> {
        if self.name.is_empty()
            || self.name.len() > CONTROLLER_PROFILE_MAX_NAME_BYTES
            || self.name.chars().any(char::is_control)
        {
            return Err(ControllerProfileError::ProfileName);
        }
        self.input.validate()?;
        self.output.validate()?;
        if self.bindings.len() > CONTROLLER_PROFILE_MAX_BINDINGS {
            return Err(ControllerProfileError::BindingCount(self.bindings.len()));
        }
        let mut ids = BTreeSet::new();
        let mut feedback = 0_usize;
        for binding in &self.bindings {
            if binding.id == 0 || !ids.insert(binding.id) {
                return Err(ControllerProfileError::BindingId(binding.id));
            }
            binding.source.validate()?;
            binding.channel.validate()?;
            if !(1..=127).contains(&binding.press_threshold) {
                return Err(ControllerProfileError::PressThreshold(binding.id));
            }
            if !binding.relative_step.is_finite()
                || !(0.0..=CONTROLLER_RELATIVE_STEP_MAX).contains(&binding.relative_step)
                || (binding.encoding != MidiValueEncoding::Absolute && binding.relative_step <= 0.0)
            {
                return Err(ControllerProfileError::RelativeStep(binding.id));
            }
            if matches!(binding.source, MidiInputSource::Note { .. })
                && binding.button_mode.is_none()
            {
                return Err(ControllerProfileError::NoteNeedsButton(binding.id));
            }
            if binding.button_mode.is_some() && binding.encoding != MidiValueEncoding::Absolute {
                return Err(ControllerProfileError::ButtonEncoding(binding.id));
            }
            if binding.target.is_scene_action()
                && binding.button_mode != Some(MidiButtonMode::Momentary)
            {
                return Err(ControllerProfileError::SceneActionNeedsMomentary(
                    binding.id,
                ));
            }
            if let Some(output) = binding.feedback {
                output.bytes(0.0)?;
                feedback = feedback.saturating_add(1);
            }
        }
        if feedback > CONTROLLER_PROFILE_MAX_FEEDBACK_BINDINGS {
            return Err(ControllerProfileError::FeedbackCount(feedback));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MidiTransportEvent {
    Start,
    Continue,
    Stop,
    Clock,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControllerEventKind {
    Control {
        binding_id: u16,
        address: RuntimeControlAddress,
        value: AutomationValue,
    },
    Transport(MidiTransportEvent),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControllerEvent {
    pub timestamp_us: u64,
    pub origin: AutomationOrigin,
    pub kind: ControllerEventKind,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ControllerDecodeReport {
    pub matched_bindings: usize,
    pub emitted_events: usize,
    pub dropped_events: usize,
}

/// Exact wire-shape admission shared by the driver callback and the pure
/// decoder. Collide-O-Scope's controller protocol accepts only three-byte
/// Note Off, Note On, and Control Change messages plus the four one-byte
/// realtime/transport messages it consumes. Running status is a stream-level
/// facility and is deliberately not accepted at this complete-message seam.
///
/// Keeping this check ahead of button state is important: masking a set high
/// bit from a malformed channel-data byte could otherwise turn hostile input
/// into a valid control mutation.
pub(crate) fn is_well_formed_controller_midi(message: &[u8]) -> bool {
    let Some(&status) = message.first() else {
        return false;
    };
    match status {
        0xf8 | 0xfa | 0xfb | 0xfc => message.len() == 1,
        value if matches!(value & 0xf0, 0x80 | 0x90 | 0xb0) => {
            message.len() == 3 && message[1] < 0x80 && message[2] < 0x80
        }
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ButtonState {
    pressed: bool,
    toggled: bool,
}

#[derive(Debug, Clone)]
pub struct ControllerDecoder {
    profile: ResolvedControllerProfile,
    buttons: BTreeMap<u16, ButtonState>,
}

impl ControllerDecoder {
    pub fn new(profile: ResolvedControllerProfile) -> Self {
        Self {
            profile,
            buttons: BTreeMap::new(),
        }
    }

    pub fn replace_profile(&mut self, profile: ResolvedControllerProfile) {
        self.profile = profile;
        self.buttons.clear();
    }

    pub fn profile(&self) -> &ResolvedControllerProfile {
        &self.profile
    }

    /// Decode into caller-owned storage. A three-byte message can match more
    /// than one authored binding, but the bounded profile caps that fan-out.
    pub fn decode(&mut self, timestamp_us: u64, message: &[u8], output: &mut Vec<ControllerEvent>) {
        let _ = self.decode_bounded(timestamp_us, message, output, usize::MAX);
    }

    pub fn decode_bounded(
        &mut self,
        timestamp_us: u64,
        message: &[u8],
        output: &mut Vec<ControllerEvent>,
        output_limit: usize,
    ) -> ControllerDecodeReport {
        let mut report = ControllerDecodeReport::default();
        if !is_well_formed_controller_midi(message) {
            return report;
        }
        let Some(&status) = message.first() else {
            return report;
        };
        let transport = match status {
            0xfa => Some(MidiTransportEvent::Start),
            0xfb => Some(MidiTransportEvent::Continue),
            0xfc => Some(MidiTransportEvent::Stop),
            0xf8 => Some(MidiTransportEvent::Clock),
            _ => None,
        };
        if let Some(transport) = transport {
            report.matched_bindings = 1;
            if output.len() < output_limit {
                output.push(ControllerEvent {
                    timestamp_us,
                    origin: AutomationOrigin::Midi,
                    kind: ControllerEventKind::Transport(transport),
                });
                report.emitted_events = 1;
            } else {
                report.dropped_events = 1;
            }
            return report;
        }
        let family = status & 0xf0;
        let channel = status & 0x0f;
        let number = message[1];
        let raw = message[2];

        // Copy one compact binding at a time so mutable button state can be
        // updated without allocating or aliasing the immutable profile.
        for index in 0..self.profile.bindings.len() {
            let binding = self.profile.bindings[index];
            let matches = binding.channel.matches(channel)
                && match binding.source {
                    MidiInputSource::ControlChange { controller } => {
                        family == 0xb0 && controller == number
                    }
                    MidiInputSource::Note { note } => {
                        matches!(family, 0x80 | 0x90) && note == number
                    }
                };
            if !matches {
                continue;
            }
            report.matched_bindings = report.matched_bindings.saturating_add(1);
            let value = if let Some(mode) = binding.button_mode {
                let pressed = match binding.source {
                    MidiInputSource::Note { .. } => family == 0x90 && raw > 0,
                    MidiInputSource::ControlChange { .. } => raw >= binding.press_threshold,
                };
                let state = self.buttons.entry(binding.id).or_default();
                let rising = pressed && !state.pressed;
                let changed = pressed != state.pressed;
                state.pressed = pressed;
                match mode {
                    MidiButtonMode::Momentary if rising => Some(AutomationValue::Trigger),
                    MidiButtonMode::Momentary => None,
                    MidiButtonMode::Toggle if rising => {
                        state.toggled = !state.toggled;
                        Some(AutomationValue::Gate(state.toggled))
                    }
                    MidiButtonMode::Toggle => None,
                    MidiButtonMode::Gate if changed => Some(AutomationValue::Gate(pressed)),
                    MidiButtonMode::Gate => None,
                }
            } else {
                decode_continuous(binding.encoding, raw, binding.relative_step)
            };
            if let Some(value) = value {
                if output.len() < output_limit {
                    output.push(ControllerEvent {
                        timestamp_us,
                        origin: AutomationOrigin::Midi,
                        kind: ControllerEventKind::Control {
                            binding_id: binding.id,
                            address: binding.target,
                            value,
                        },
                    });
                    report.emitted_events = report.emitted_events.saturating_add(1);
                } else {
                    report.dropped_events = report.dropped_events.saturating_add(1);
                }
            }
        }
        report
    }
}

fn decode_continuous(encoding: MidiValueEncoding, raw: u8, step: f32) -> Option<AutomationValue> {
    if encoding == MidiValueEncoding::Absolute {
        return Some(AutomationValue::Absolute(f32::from(raw) / 127.0));
    }
    let delta = match encoding {
        MidiValueEncoding::Absolute => unreachable!(),
        MidiValueEncoding::RelativeTwosComplement => match raw {
            1..=63 => i16::from(raw),
            65..=127 => i16::from(raw) - 128,
            0 | 64 => 0,
            128..=u8::MAX => 0,
        },
        MidiValueEncoding::RelativeBinaryOffset => i16::from(raw) - 64,
        MidiValueEncoding::RelativeSignMagnitude => {
            let magnitude = i16::from(raw & 0x3f);
            if raw & 0x40 == 0 {
                magnitude
            } else {
                -magnitude
            }
        }
    };
    (delta != 0).then_some(AutomationValue::Delta(
        (delta as f32 * step).clamp(-1.0, 1.0),
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControllerProfileError {
    DocumentBytes(usize),
    ActionBytes(usize),
    Json(String),
    Version(u16),
    ProfileName,
    DeviceName,
    BindingCount(usize),
    FeedbackCount(usize),
    BindingId(u16),
    Channel(u8),
    MidiData(u8),
    PressThreshold(u16),
    RelativeStep(u16),
    NoteNeedsButton(u16),
    ButtonEncoding(u16),
    SceneActionNeedsMomentary(u16),
    LegacySlot(u8),
    MissingLayer(SavedLayerPosition),
    MissingScene(SceneId),
    NonFinite,
    Io(String),
}

impl fmt::Display for ControllerProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid controller profile: {self:?}")
    }
}

impl std::error::Error for ControllerProfileError {}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_PATH_SEQUENCE: AtomicU64 = AtomicU64::new(1);

    fn test_document_path(label: &str) -> PathBuf {
        let ordinal = TEST_PATH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "collide-o-scope-controller-profile-{label}-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        directory.join(CONTROLLER_PROFILE_FILE_NAME)
    }

    fn remove_test_document(path: &Path) {
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

    fn position(value: u32) -> SavedLayerPosition {
        SavedLayerPosition::new(value).unwrap()
    }

    fn stable(value: u64) -> StableLayerId {
        StableLayerId::new(value).unwrap()
    }

    fn scene(value: u16) -> SceneId {
        SceneId::new(value).unwrap()
    }

    #[test]
    fn legacy_profile_preserves_the_historical_four_absolute_cc_slots() {
        let profile = ResolvedControllerProfile::legacy_four_cc();
        let mut decoder = ControllerDecoder::new(profile);
        let mut output = Vec::new();
        for (cc, value) in [(1, 0), (2, 32), (3, 64), (4, 127)] {
            decoder.decode(cc.into(), &[0xb7, cc, value], &mut output);
        }
        assert_eq!(output.len(), 4);
        for (slot, event) in output.iter().enumerate() {
            assert!(matches!(
                event.kind,
                ControllerEventKind::Control {
                    address: RuntimeControlAddress::LegacyMidiSlot(candidate),
                    value: AutomationValue::Absolute(_),
                    ..
                } if usize::from(candidate) == slot
            ));
        }
    }

    #[test]
    fn malformed_wire_messages_never_emit_or_mutate_button_state() {
        let binding = ResolvedControllerBinding {
            id: 1,
            source: MidiInputSource::ControlChange { controller: 10 },
            channel: MidiChannelFilter::Omni,
            encoding: MidiValueEncoding::Absolute,
            button_mode: Some(MidiButtonMode::Toggle),
            press_threshold: 64,
            relative_step: 0.1,
            target: RuntimeControlAddress::Master(ControlParameter::Enabled),
            feedback: None,
        };
        let mut decoder = ControllerDecoder::new(ResolvedControllerProfile {
            name: "wire validation".into(),
            input: MidiDeviceSelector::FirstAvailable,
            output: MidiDeviceSelector::FirstAvailable,
            bindings: vec![binding].into_boxed_slice(),
        });
        let malformed: &[&[u8]] = &[
            &[],
            &[0xb0, 10],
            &[0xb0, 10, 127, 0],
            &[0xb0, 0x80 | 10, 127],
            &[0xb0, 10, 0xff],
            &[0x90, 0x80 | 60, 127],
            &[0xf8, 0],
            &[0xfa, 0],
            &[0xfb, 0],
            &[0xfc, 0],
            &[10, 127],
        ];
        let mut events = Vec::new();
        for message in malformed {
            assert!(!is_well_formed_controller_midi(message));
            assert_eq!(
                decoder.decode_bounded(0, message, &mut events, usize::MAX),
                ControllerDecodeReport::default()
            );
        }
        assert!(events.is_empty());
        assert!(decoder.buttons.is_empty());

        decoder.decode(1, &[0xb0, 10, 127], &mut events);
        assert!(matches!(
            events.as_slice(),
            [ControllerEvent {
                kind: ControllerEventKind::Control {
                    value: AutomationValue::Gate(true),
                    ..
                },
                ..
            }]
        ));
    }

    #[test]
    fn hostile_documents_reject_before_or_during_atomic_validation() {
        assert!(matches!(
            ControllerProfileDocument::from_json_bytes(&vec![
                b' ';
                CONTROLLER_PROFILE_MAX_BYTES + 1
            ]),
            Err(ControllerProfileError::DocumentBytes(_))
        ));
        let mut profile = ControllerProfileDocument::legacy_four_cc();
        profile.name = "x".repeat(CONTROLLER_PROFILE_MAX_NAME_BYTES + 1);
        assert_eq!(profile.validate(), Err(ControllerProfileError::ProfileName));
        profile = ControllerProfileDocument::legacy_four_cc();
        profile.bindings = vec![ControllerBinding::default(); CONTROLLER_PROFILE_MAX_BINDINGS + 1];
        assert!(matches!(
            profile.validate(),
            Err(ControllerProfileError::BindingCount(_))
        ));

        profile = ControllerProfileDocument::legacy_four_cc();
        profile.bindings[0].press_threshold = u8::MAX;
        assert_eq!(
            profile.validate(),
            Err(ControllerProfileError::PressThreshold(1))
        );
        profile = ControllerProfileDocument::legacy_four_cc();
        profile.input = MidiDeviceSelector::Exact {
            name: "bad\nport".into(),
        };
        assert_eq!(profile.validate(), Err(ControllerProfileError::DeviceName));
        assert_eq!(
            MidiOutputMessage::ControlChange {
                channel: 1,
                controller: u8::MAX,
            }
            .bytes(0.0),
            Err(ControllerProfileError::MidiData(u8::MAX))
        );
    }

    #[test]
    fn persisted_profile_round_trip_uses_saved_positions_not_runtime_ids() {
        let mut profile = ControllerProfileDocument::legacy_four_cc();
        profile.bindings[0].target = SavedControlAddress::Layer {
            position: position(3),
            parameter: ControlParameter::Opacity,
        };
        profile.bindings[0].feedback = Some(MidiOutputMessage::Note {
            channel: 16,
            note: 72,
        });
        let bytes = profile.to_json_bytes().unwrap();
        let json = std::str::from_utf8(&bytes).unwrap();
        assert!(json.contains("\"position\""));
        let decoded = ControllerProfileDocument::from_json_bytes(&bytes).unwrap();
        assert_eq!(decoded, profile);
        assert_eq!(
            decoded.bindings[0].feedback.unwrap().bytes(0.5).unwrap(),
            [0x9f, 72, 64]
        );
    }

    #[test]
    fn import_export_contract_is_bounded_pathless_and_prepared_as_one_pair() {
        let mut document = ControllerProfileDocument::legacy_four_cc();
        document.name = "Portable controller".into();
        document.bindings[0].target = SavedControlAddress::Layer {
            position: position(2),
            parameter: ControlParameter::Opacity,
        };
        let exported = export_controller_profile_json(&document).unwrap();
        assert!(exported.len() <= CONTROLLER_PROFILE_MAX_BYTES);
        let prepared = prepare_controller_profile_json_import(&exported, |candidate| {
            (candidate == position(2)).then_some(stable(77))
        })
        .unwrap();
        assert_eq!(prepared.document(), &document);
        assert_eq!(
            prepared.runtime().bindings[0].target,
            RuntimeControlAddress::Layer {
                layer_id: stable(77),
                parameter: ControlParameter::Opacity,
            }
        );
        let (prepared_document, prepared_runtime) = prepared.into_parts();
        assert_eq!(prepared_document, document);
        assert_eq!(prepared_runtime.name, "Portable controller");

        let action = ControllerProfileAction::Import {
            document: prepared_document,
        };
        let action_bytes = action.to_json_bytes().unwrap();
        assert_eq!(
            ControllerProfileAction::from_json_bytes(&action_bytes).unwrap(),
            action
        );
        assert!(std::str::from_utf8(&action_bytes)
            .unwrap()
            .find("path")
            .is_none());
        assert!(matches!(
            ControllerProfileAction::from_json_bytes(br#"{"action":"export","path":"C:/host"}"#),
            Err(ControllerProfileError::Json(_))
        ));
        assert!(matches!(
            ControllerProfileAction::from_json_bytes(&vec![
                b' ';
                CONTROLLER_PROFILE_ACTION_MAX_BYTES + 1
            ]),
            Err(ControllerProfileError::ActionBytes(_))
        ));
    }

    #[test]
    fn native_import_reader_has_no_default_fallback_and_obeys_the_byte_cap() {
        let path = test_document_path("native-import");
        let document = ControllerProfileDocument {
            name: "Native import".into(),
            ..ControllerProfileDocument::default()
        };
        fs::write(&path, document.to_json_bytes().unwrap()).unwrap();
        assert_eq!(read_controller_profile_import(&path).unwrap(), document);
        fs::write(&path, vec![b'x'; CONTROLLER_PROFILE_MAX_BYTES + 1]).unwrap();
        assert!(matches!(
            read_controller_profile_import(&path),
            Err(ControllerProfileError::DocumentBytes(_))
        ));
        fs::remove_file(&path).unwrap();
        assert!(matches!(
            read_controller_profile_import(&path),
            Err(ControllerProfileError::Io(_))
        ));
        remove_test_document(&path);
    }

    #[test]
    fn profile_persistence_defaults_safely_and_replaces_atomically() {
        let path = test_document_path("atomic");
        fs::remove_dir(path.parent().unwrap()).unwrap();
        let missing = load_controller_profile_or_default(&path);
        assert_eq!(missing.document, ControllerProfileDocument::default());
        assert_eq!(missing.status, PersistedDocumentLoadStatus::DefaultMissing);
        assert!(missing.status.used_default());

        let first = ControllerProfileDocument {
            name: "First controller".into(),
            ..ControllerProfileDocument::default()
        };
        save_controller_profile_atomic(&first, &path).unwrap();
        let loaded = load_controller_profile_or_default(&path);
        assert_eq!(loaded.document, first);
        assert_eq!(loaded.status, PersistedDocumentLoadStatus::Loaded);
        assert!(!loaded.status.used_default());

        let mut replacement = first.clone();
        replacement.name = "Replacement controller".into();
        replacement.input = MidiDeviceSelector::Exact {
            name: "Exact Input".into(),
        };
        save_controller_profile_atomic(&replacement, &path).unwrap();
        assert_eq!(
            ControllerProfileDocument::from_json_bytes(&fs::read(&path).unwrap()).unwrap(),
            replacement
        );
        let published = fs::read(&path).unwrap();
        let mut invalid = replacement.clone();
        invalid.name = "x".repeat(CONTROLLER_PROFILE_MAX_NAME_BYTES + 1);
        assert_eq!(
            save_controller_profile_atomic(&invalid, &path),
            Err(ControllerProfileError::ProfileName)
        );
        assert_eq!(fs::read(&path).unwrap(), published);
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
        remove_test_document(&path);
    }

    #[test]
    fn profile_persistence_rejects_hostile_files_and_unknown_nested_fields() {
        let path = test_document_path("hostile");
        fs::write(&path, vec![b'x'; CONTROLLER_PROFILE_MAX_BYTES + 1]).unwrap();
        let oversized = load_controller_profile_or_default(&path);
        assert_eq!(oversized.document, ControllerProfileDocument::default());
        assert!(matches!(
            oversized.status,
            PersistedDocumentLoadStatus::DefaultInvalid(_)
        ));

        let profile = ControllerProfileDocument::default();
        let mut top_level = serde_json::to_value(&profile).unwrap();
        top_level["unexpected"] = serde_json::json!(true);
        assert!(matches!(
            ControllerProfileDocument::from_json_bytes(&serde_json::to_vec(&top_level).unwrap()),
            Err(ControllerProfileError::Json(_))
        ));
        let mut nested = serde_json::to_value(&profile).unwrap();
        nested["bindings"][0]["source"]["unexpected"] = serde_json::json!(true);
        assert!(matches!(
            ControllerProfileDocument::from_json_bytes(&serde_json::to_vec(&nested).unwrap()),
            Err(ControllerProfileError::Json(_))
        ));
        let mut binding = serde_json::to_value(&profile).unwrap();
        binding["bindings"][0]["unexpected"] = serde_json::json!(true);
        assert!(matches!(
            ControllerProfileDocument::from_json_bytes(&serde_json::to_vec(&binding).unwrap()),
            Err(ControllerProfileError::Json(_))
        ));
        remove_test_document(&path);
    }

    #[test]
    fn default_profile_path_follows_the_platform_state_precedence() {
        let local = PathBuf::from("local-state");
        let xdg = PathBuf::from("xdg-state");
        let home = PathBuf::from("home");
        assert_eq!(
            default_state_dir_from(
                Some(local.as_os_str()),
                Some(xdg.as_os_str()),
                Some(home.as_os_str())
            ),
            local.join("collide-o-scope")
        );
        assert_eq!(
            default_state_dir_from(None, Some(xdg.as_os_str()), Some(home.as_os_str())),
            xdg.join("collide-o-scope")
        );
        assert_eq!(
            default_state_dir_from(None, None, Some(home.as_os_str())),
            home.join(".local").join("state").join("collide-o-scope")
        );
        assert_eq!(
            default_state_dir_from(None, None, None),
            PathBuf::from(".collide-o-scope")
        );
        assert_eq!(
            default_controller_profile_path().file_name(),
            Some(OsStr::new(CONTROLLER_PROFILE_FILE_NAME))
        );
        let status = bounded_status(format!("{}\nignored", "é".repeat(400)));
        assert!(status.len() <= PERSISTED_DOCUMENT_STATUS_MAX_BYTES);
        assert!(!status.chars().any(char::is_control));
    }

    #[test]
    fn saved_layer_resolution_is_stable_across_later_reorder() {
        let mut profile = ControllerProfileDocument::legacy_four_cc();
        profile.bindings[0].target = SavedControlAddress::Layer {
            position: position(2),
            parameter: ControlParameter::Opacity,
        };
        let resolved = profile
            .resolve(|candidate| (candidate == position(2)).then_some(stable(41)))
            .unwrap();
        assert_eq!(
            resolved.bindings[0].target,
            RuntimeControlAddress::Layer {
                layer_id: stable(41),
                parameter: ControlParameter::Opacity,
            }
        );
        // A later positional mapping now names another layer; the already
        // resolved runtime profile remains attached to stable ID 41.
        let reordered = |_: SavedLayerPosition| Some(stable(99));
        assert_eq!(reordered(position(2)), Some(stable(99)));
        assert_eq!(
            resolved.bindings[0].target,
            RuntimeControlAddress::Layer {
                layer_id: stable(41),
                parameter: ControlParameter::Opacity,
            }
        );
    }

    #[test]
    fn scene_actions_round_trip_in_v1_profiles_and_resolve_only_live_stable_ids() {
        let mut profile = ControllerProfileDocument::legacy_four_cc();
        profile.bindings[0] = ControllerBinding {
            id: 1,
            source: MidiInputSource::Note { note: 60 },
            button_mode: Some(MidiButtonMode::Momentary),
            target: SavedControlAddress::ScenePrepare { scene_id: scene(7) },
            ..ControllerBinding::default()
        };
        profile.bindings[1] = ControllerBinding {
            id: 2,
            source: MidiInputSource::Note { note: 61 },
            button_mode: Some(MidiButtonMode::Momentary),
            target: SavedControlAddress::SceneTrigger { scene_id: scene(7) },
            ..ControllerBinding::default()
        };

        let bytes = profile.to_json_bytes().unwrap();
        assert!(std::str::from_utf8(&bytes)
            .unwrap()
            .contains(r#""scope": "scene_trigger""#));
        let restored = ControllerProfileDocument::from_json_bytes(&bytes).unwrap();
        assert_eq!(restored, profile);
        assert_eq!(
            restored.version, 1,
            "the additive vocabulary keeps v1 readable"
        );

        assert_eq!(
            restored.resolve_with_scenes(|_| None, |_| false),
            Err(ControllerProfileError::MissingScene(scene(7)))
        );
        let resolved = restored
            .resolve_with_scenes(|_| None, |candidate| candidate == scene(7))
            .unwrap();
        assert_eq!(
            resolved.bindings[0].target,
            RuntimeControlAddress::ScenePrepare { scene_id: scene(7) }
        );
        assert_eq!(
            resolved.bindings[1].target,
            RuntimeControlAddress::SceneTrigger { scene_id: scene(7) }
        );
    }

    #[test]
    fn scene_actions_require_momentary_bindings_and_emit_only_on_rising_edges() {
        let mut profile = ControllerProfileDocument::legacy_four_cc();
        profile.bindings[0].target = SavedControlAddress::SceneTrigger { scene_id: scene(9) };
        assert_eq!(
            profile.validate(),
            Err(ControllerProfileError::SceneActionNeedsMomentary(1))
        );

        profile.bindings[0].source = MidiInputSource::Note { note: 64 };
        profile.bindings[0].button_mode = Some(MidiButtonMode::Momentary);
        let resolved = profile
            .resolve_with_scenes(|_| None, |candidate| candidate == scene(9))
            .unwrap();
        let mut decoder = ControllerDecoder::new(resolved);
        let mut events = Vec::new();
        for message in [
            [0x90, 64, 127],
            [0x90, 64, 127],
            [0x80, 64, 0],
            [0x90, 64, 127],
        ] {
            decoder.decode(0, &message, &mut events);
        }
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|event| matches!(
            event.kind,
            ControllerEventKind::Control {
                address: RuntimeControlAddress::SceneTrigger { scene_id },
                value: AutomationValue::Trigger,
                ..
            } if scene_id == scene(9)
        )));
    }

    #[test]
    fn relative_encodings_cover_twos_offset_and_sign_magnitude() {
        assert_eq!(
            decode_continuous(MidiValueEncoding::RelativeTwosComplement, 1, 0.1),
            Some(AutomationValue::Delta(0.1))
        );
        assert_eq!(
            decode_continuous(MidiValueEncoding::RelativeTwosComplement, 127, 0.1),
            Some(AutomationValue::Delta(-0.1))
        );
        assert_eq!(
            decode_continuous(MidiValueEncoding::RelativeBinaryOffset, 65, 0.1),
            Some(AutomationValue::Delta(0.1))
        );
        assert_eq!(
            decode_continuous(MidiValueEncoding::RelativeBinaryOffset, 63, 0.1),
            Some(AutomationValue::Delta(-0.1))
        );
        assert_eq!(
            decode_continuous(MidiValueEncoding::RelativeSignMagnitude, 1, 0.1),
            Some(AutomationValue::Delta(0.1))
        );
        assert_eq!(
            decode_continuous(MidiValueEncoding::RelativeSignMagnitude, 65, 0.1),
            Some(AutomationValue::Delta(-0.1))
        );
    }

    #[test]
    fn note_and_cc_buttons_have_distinct_momentary_toggle_and_gate_laws() {
        let make = |id, controller, mode| ResolvedControllerBinding {
            id,
            source: MidiInputSource::ControlChange { controller },
            channel: MidiChannelFilter::Omni,
            encoding: MidiValueEncoding::Absolute,
            button_mode: Some(mode),
            press_threshold: 64,
            relative_step: 0.1,
            target: RuntimeControlAddress::Master(ControlParameter::Enabled),
            feedback: None,
        };
        let profile = ResolvedControllerProfile {
            name: "buttons".into(),
            input: MidiDeviceSelector::FirstAvailable,
            output: MidiDeviceSelector::FirstAvailable,
            bindings: vec![
                make(1, 10, MidiButtonMode::Momentary),
                make(2, 11, MidiButtonMode::Toggle),
                make(3, 12, MidiButtonMode::Gate),
                ResolvedControllerBinding {
                    id: 4,
                    source: MidiInputSource::Note { note: 60 },
                    channel: MidiChannelFilter::Exact { channel: 2 },
                    encoding: MidiValueEncoding::Absolute,
                    button_mode: Some(MidiButtonMode::Gate),
                    press_threshold: 64,
                    relative_step: 0.1,
                    target: RuntimeControlAddress::Master(ControlParameter::Play),
                    feedback: None,
                },
            ]
            .into_boxed_slice(),
        };
        let mut decoder = ControllerDecoder::new(profile);
        let mut events = Vec::new();
        for message in [
            [0xb0, 10, 127],
            [0xb0, 10, 0],
            [0xb0, 11, 127],
            [0xb0, 11, 0],
            [0xb0, 11, 127],
            [0xb0, 12, 127],
            [0xb0, 12, 0],
            [0x91, 60, 100],
            [0x81, 60, 0],
        ] {
            decoder.decode(0, &message, &mut events);
        }
        let values = events
            .iter()
            .filter_map(|event| match event.kind {
                ControllerEventKind::Control { value, .. } => Some(value),
                ControllerEventKind::Transport(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            vec![
                AutomationValue::Trigger,
                AutomationValue::Gate(true),
                AutomationValue::Gate(false),
                AutomationValue::Gate(true),
                AutomationValue::Gate(false),
                AutomationValue::Gate(true),
                AutomationValue::Gate(false),
            ]
        );
    }

    #[test]
    fn transport_bytes_are_typed_and_never_alias_channel_messages() {
        let mut decoder = ControllerDecoder::new(ResolvedControllerProfile::legacy_four_cc());
        let mut output = Vec::new();
        for byte in [0xfa, 0xfb, 0xfc, 0xf8] {
            decoder.decode(42, &[byte], &mut output);
        }
        assert_eq!(
            output.iter().map(|event| event.kind).collect::<Vec<_>>(),
            vec![
                ControllerEventKind::Transport(MidiTransportEvent::Start),
                ControllerEventKind::Transport(MidiTransportEvent::Continue),
                ControllerEventKind::Transport(MidiTransportEvent::Stop),
                ControllerEventKind::Transport(MidiTransportEvent::Clock),
            ]
        );
        assert!(!AutomationOrigin::Midi.records_manual_history());
        assert!(!AutomationOrigin::Osc("127.0.0.1:9".parse().unwrap()).records_manual_history());
        assert!(!AutomationOrigin::HostAutomation.records_manual_history());
    }
}
