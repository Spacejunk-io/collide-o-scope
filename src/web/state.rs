//! Shared state between the web control panel and the render engine.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{watch, Mutex, RwLock};

use crate::action_correlation::{
    ActionCorrelationMonitor, ActionDisposition, ActionEnvelope, ActionIdentity, ActionSequencer,
    ActionSourceClass, ActionTimingSnapshot,
};
use crate::durable_file::UploadAdmission;
use crate::effects::EffectUniforms;
use crate::image_routing::{LayerImageStage, MatteChannel};
use crate::performance::{
    autopilot::AutopilotState, AutopilotPlan, ClipSlotId, SavedLayerPosition, SceneId,
};
use crate::spatial::{FitMode, SpatialTransform};
use crate::transport::{ClipTransportConfig, CueId, NormalizedTime, SourceTimecode, TriggerMode};
use crate::visual_rack::EdgeTiming;

/// Maximum UTF-8 payload for a browser-authored optional Scene name.
/// Empty is valid and renders as the stable numeric Scene identity.
pub const MAX_SCENE_NAME_BYTES: usize = 128;

/// One listener's lifecycle. Loopback IPv4, loopback IPv6, and LAN TLS each
/// own a separate value so success on one socket can never conceal failure on
/// another.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ControlListenerStatus {
    #[default]
    Stopped,
    Starting,
    Listening {
        address: SocketAddr,
    },
    Unavailable {
        reason: String,
    },
}

/// A bearer-token URL may be exposed only at the two deliberate local UI
/// boundaries: the native Open Panel link and the authenticated QR surface.
/// Debug formatting is always redacted so assertion/panic/log formatting
/// cannot copy the session secret into diagnostics.
#[derive(Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ControlAccessUrl(String);

impl ControlAccessUrl {
    pub(crate) fn new(url: String) -> Self {
        Self(url)
    }

    pub fn expose_to_local_ui(&self) -> &str {
        &self.0
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for ControlAccessUrl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ControlAccessUrl(<redacted>)")
    }
}

/// Atomic native-shell view of all listener roles for one server generation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ControlServerInfo {
    pub generation: u64,
    pub loopback_ipv4: ControlListenerStatus,
    pub loopback_ipv6: ControlListenerStatus,
    pub lan_tls: ControlListenerStatus,
    pub(crate) loopback_ipv4_url: Option<ControlAccessUrl>,
    pub(crate) loopback_ipv6_url: Option<ControlAccessUrl>,
    pub lan_url: Option<ControlAccessUrl>,
    /// First twelve hexadecimal digits of SHA-256(session token). This is a
    /// correlation label, not an authentication credential.
    pub session_fingerprint: String,
}

impl ControlServerInfo {
    pub fn local_url(&self) -> Option<&ControlAccessUrl> {
        self.loopback_ipv4_url
            .as_ref()
            .or(self.loopback_ipv6_url.as_ref())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ControlListenerSlot {
    LoopbackIpv4,
    LoopbackIpv6,
    LanTls,
}

/// Shared state accessible by both the web server and the render loop.
pub struct WebState {
    /// Full app snapshot (pushed from render loop each frame)
    pub app: RwLock<AppSnapshot>,
    /// Broadcast channel for pushing state to all WebSocket clients
    /// Newest-only serialized state. A slow browser observes the latest
    /// coherent generation; obsolete 30 Hz snapshots are never queued.
    pub tx: watch::Sender<Arc<String>>,
    snapshot_requested: AtomicBool,
    full_snapshot_generation: AtomicU64,
    last_full_snapshot: std::sync::RwLock<Arc<String>>,
    serialized_publications: AtomicU64,
    /// Actions queue: browser pushes commands, render loop drains them. The
    /// engine-owned envelope is deliberately internal and non-serialized.
    pub actions: Mutex<Vec<ActionEnvelope<WebAction>>>,
    action_sequencer: Arc<ActionSequencer>,
    action_correlation: std::sync::Mutex<ActionCorrelationMonitor>,
    /// Latest validated, path-free controller-profile JSON for the dedicated
    /// authenticated import/export endpoint. Keeping it outside AppSnapshot
    /// avoids rebroadcasting up to 256 KiB on every render frame.
    controller_profile_export: std::sync::RwLock<Vec<u8>>,
    /// At most one remote controller may own one scalar destination. The
    /// server retains whether an authored value crossed the Begin barrier so
    /// a hostile dirty Cancel cannot strand Main's matching transaction.
    browser_history_gesture: Mutex<Option<BrowserHistoryGesture>>,
    /// Thumbnail cache: filename → JPEG bytes (generated on library scan)
    pub thumbnails: std::sync::RwLock<HashMap<String, Vec<u8>>>,
    /// Preview frames: filename → vec of JPEG frames (for hover animation)
    pub preview_frames: std::sync::RwLock<HashMap<String, Vec<Vec<u8>>>>,
    /// Aggregate bytes retained by both convenience caches. The mutex is also
    /// their cross-cache publication/clear barrier, so concurrent helper
    /// completions cannot independently over-admit the same remaining budget.
    library_media_cache_bytes: std::sync::Mutex<u64>,
    /// One metadata/decode pipeline at a time across startup, folder changes,
    /// and repeated rescans. Per-pipeline helper fan-out is therefore also the
    /// process-wide concurrency ceiling rather than a per-request ceiling.
    library_media_helper_gate: std::sync::Mutex<()>,
    /// Library folder for clip uploads (set by the app; None until known).
    pub library_folder: std::sync::RwLock<Option<std::path::PathBuf>>,
    /// One process-wide admission ledger shared by every listener role.
    upload_admission: UploadAdmission,
    /// Clone of Main's Arc-backed host-local media policy. Mode changes made
    /// by Main are therefore visible to upload preflight without a new wire
    /// or patch-owned policy surface.
    upload_media_safety_policy: std::sync::RwLock<crate::media_safety::MediaSafetyPolicy>,
    /// Linearizes the final upload name operation against library changes and
    /// control-server retirement. The upload has already synced and probed
    /// its file before taking this short gate.
    upload_publication_gate: std::sync::Mutex<()>,
    /// Truthful independent listener lifecycles plus the two deliberately
    /// local bearer-token URL surfaces.
    control_server: std::sync::RwLock<ControlServerInfo>,
    /// Rejects late status writes from a retired/crashed prior server after a
    /// restart has installed a newer generation.
    control_server_generation: AtomicU64,
    /// The current generation is no longer allowed to publish uploads once
    /// retirement begins, even while in-flight socket futures are draining.
    control_server_stopping_generation: AtomicU64,
    /// Rejects thumbnail/preview writes from a folder that is no longer the
    /// active library, even if its background ffmpeg worker finishes later.
    library_generation: AtomicU64,
    /// Immutable, bounded index currently visible to Main and authenticated
    /// page/search requests. Publication is generation checked so a slow old
    /// directory scan cannot replace a newer folder's result.
    library_index: std::sync::RwLock<Arc<crate::library_index::LibraryIndex>>,
    library_index_revision: AtomicU64,
    /// Server-owned phone-stream membership and monotonic sample freshness.
    gyro_streams: std::sync::Mutex<GyroStreamRegistry>,
    /// B11: which clients currently declare themselves watching the
    /// monitoring bay, by freshest declaration instant. Watchers re-assert on
    /// a heartbeat, so a vanished tab expires by [`MONITOR_WATCH_TIMEOUT`]
    /// instead of pinning the readback armed.
    monitor_watchers: std::sync::Mutex<HashMap<u64, Instant>>,
    next_client_id: AtomicU64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BrowserHistoryGesture {
    client_id: u64,
    gesture_id: u64,
    coalesce_key: Option<String>,
    dirty: bool,
}

/// A phone normally publishes at roughly 30 Hz. This allows substantial
/// mobile scheduling jitter without permitting a vanished pose to remain
/// applied indefinitely.
pub const GYRO_SAMPLE_TIMEOUT: Duration = Duration::from_millis(1_500);

/// B11: a watching panel re-asserts its declaration every few seconds, so a
/// crashed or silently discarded tab stops arming the monitor readback
/// within this window even if its socket lingers.
pub const MONITOR_WATCH_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Default)]
struct GyroStreamRegistry {
    /// Clients which have used the explicit start/stop protocol.
    declared_clients: HashSet<u64>,
    /// Clients currently claiming ownership of a live sensor stream.
    streamers: HashSet<u64>,
    last_sample: Option<Instant>,
    ever_enabled: bool,
}

impl GyroStreamRegistry {
    fn set_stream(&mut self, client_id: u64, enabled: bool) -> bool {
        let mut changed = self.declared_clients.insert(client_id);
        if enabled {
            changed |= !self.ever_enabled;
            self.ever_enabled = true;
            changed |= self.streamers.insert(client_id);
        } else {
            changed |= self.streamers.remove(&client_id);
        }
        changed
    }

    fn note_sample_at(&mut self, client_id: u64, now: Instant) {
        // Older panels have no gyro_stream action. Their first valid sample
        // implicitly starts a stream, while an explicitly stopped new panel
        // cannot be reactivated by a late in-flight sample.
        if !self.declared_clients.contains(&client_id) {
            self.ever_enabled = true;
            self.streamers.insert(client_id);
        }
        if self.streamers.contains(&client_id) {
            self.last_sample = Some(now);
        }
    }

    fn disconnect(&mut self, client_id: u64) {
        self.streamers.remove(&client_id);
        self.declared_clients.remove(&client_id);
    }

    fn status_at(&self, now: Instant) -> GyroStatusSnapshot {
        let sample_age_ms = self.last_sample.map(|sample| {
            now.saturating_duration_since(sample)
                .as_millis()
                .min(u128::from(u64::MAX)) as u64
        });
        let active = !self.streamers.is_empty()
            && self
                .last_sample
                .is_some_and(|sample| now.saturating_duration_since(sample) <= GYRO_SAMPLE_TIMEOUT);
        GyroStatusSnapshot {
            active,
            stale: self.ever_enabled && !active,
            streamers: self.streamers.len(),
            sample_age_ms,
        }
    }
}

/// Keep browser input bounded even if a client produces controls faster than
/// the render loop can consume them. A small reserve guarantees that safety
/// commands are still admitted while ordinary fader traffic is saturated.
pub const MAX_PENDING_ACTIONS: usize = 512;
const PRIORITY_ACTION_RESERVE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueOutcome {
    Added,
    Coalesced,
    Dropped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionIngressDisposition {
    Queued,
    QueuedAfterCoalescing,
    Coalesced,
    Refused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ActionIngressAck {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub sequence: u64,
    pub disposition: ActionIngressDisposition,
}

/// Owns one minted ingress identity until it is synchronously transferred to
/// the bounded action queue or given a terminal disposition. Because the
/// guard itself is captured by every async admission future, cancellation at
/// any lock await records exactly one Refused receipt instead of orphaning the
/// sequence.
pub(crate) struct ActionIngressTerminalGuard {
    state: Arc<WebState>,
    identity: Option<ActionIdentity>,
}

impl ActionIngressTerminalGuard {
    fn new(state: Arc<WebState>, identity: ActionIdentity) -> Self {
        Self {
            state,
            identity: Some(identity),
        }
    }

    pub(crate) fn identity(&self) -> ActionIdentity {
        self.identity
            .expect("an armed ingress guard has an identity")
    }

    pub(crate) fn terminalize(mut self, disposition: ActionDisposition) -> ActionIngressAck {
        let identity = self
            .identity
            .take()
            .expect("an ingress identity is terminalized exactly once");
        self.state
            .terminal_action_identity_with_ack(identity, disposition)
    }

    fn disarm(&mut self) {
        self.identity = None;
    }
}

impl Drop for ActionIngressTerminalGuard {
    fn drop(&mut self) {
        if let Some(identity) = self.identity.take() {
            self.state
                .record_terminal_action_identity(identity, ActionDisposition::Refused);
        }
    }
}

fn default_new_layer_fit() -> FitMode {
    FitMode::Fit
}

/// Decimal-ID scope used by rack snapshots and mutations. Numeric runtime IDs
/// never cross the JSON boundary, avoiding JavaScript precision loss.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum CreativeScopeSnapshot {
    Master,
    Layer { layer_id: String },
    Group { group_id: String },
}

/// Typed preset destination. The first three variants intentionally retain
/// the established `{ "scope": ... }` wire shapes; controller and StageMap
/// documents use distinct non-creative scopes and can never be mistaken for a
/// layer/group values transfer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum PresetTargetSnapshot {
    Master,
    Layer { layer_id: String },
    Group { group_id: String },
    ControllerProfile,
    StageMap,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum CreativeImageSourceSnapshot {
    SelectedLayer {
        layer_id: String,
        #[serde(default)]
        stage: LayerImageStage,
    },
    /// Output-only diagnostic retained when a saved donor cannot be resolved.
    /// Ingress never accepts this as a newly authored route.
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
        #[serde(default)]
        stage: LayerImageStage,
    },
    OneBelow,
    AllBelow,
    GroupOutput {
        group_id: String,
    },
    /// Output-only diagnostic retained after deletion of a referenced group.
    MissingGroupOutput {
        group_id: String,
    },
    CleanProgram,
    /// The etched gesture field. A master-scope singleton: it carries no ID and
    /// no saved position, and it is authorable from the browser at either
    /// timing.
    GestureCanvas,
    /// The finished programme at N-1 (the pre-blackout opaque audience image).
    /// The same master-scope singleton shape: no ID, no saved position,
    /// authorable at either timing because the tap is N-1 by construction.
    ProgramTap,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreativeImageTapSnapshot {
    pub input: CreativeImageSourceSnapshot,
    pub timing: EdgeTiming,
}

/// One addressed Symmetry Field route. The closed `slot` tag names the route
/// class and `index` names the fixed slot inside that class, because slot index
/// is route identity: clearing slot 0 must never slide slot 1's donor down.
///
/// An image slot carries a full image tap. A motion slot carries a stable layer
/// ID or `None` to clear it; a motion route never enters the image dependency
/// graph, so it has no timing, stage, channel, or inversion of its own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "slot", rename_all = "snake_case")]
pub enum SymmetryRouteSnapshot {
    Image {
        index: u8,
        route: CreativeImageTapSnapshot,
    },
    Motion {
        index: u8,
        #[serde(default)]
        layer_id: Option<String>,
    },
}

impl SymmetryRouteSnapshot {
    /// The addressed slot index, whichever class it belongs to.
    pub const fn index(&self) -> u8 {
        match self {
            Self::Image { index, .. } | Self::Motion { index, .. } => *index,
        }
    }
}

/// Which of a Residual node's two authored routes an ordered reroute names.
/// The vocabulary is closed and tagged so an out-of-range slot is a
/// deserialization error rather than a positional fallback onto the other
/// route — the same law that keeps a stale node ID from rebinding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualRouteSlotSnapshot {
    Structure,
    Detail,
}

impl ResidualRouteSlotSnapshot {
    /// The authored slot index this wire token names.
    pub const fn slot(self) -> u8 {
        match self {
            Self::Structure => crate::visual_rack::RESIDUAL_STRUCTURE_SLOT,
            Self::Detail => crate::visual_rack::RESIDUAL_DETAIL_SLOT,
        }
    }

    /// Operator-facing name of the slot, so a commit status names the input
    /// that actually moved rather than a bare index.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Structure => "structure",
            Self::Detail => "detail",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreativeMatteSnapshot {
    pub route: CreativeImageTapSnapshot,
    pub channel: String,
    pub invert: bool,
    pub amount: f32,
    pub threshold: f32,
    pub softness: f32,
    /// Non-empty for a retained missing donor or another inert route.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub diagnostic: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VisualNodeSnapshot {
    pub node_id: String,
    pub enabled: bool,
    pub wet: f32,
    pub blend: String,
    pub kind: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VisualRackSnapshot {
    #[serde(default)]
    pub nodes: Vec<VisualNodeSnapshot>,
    pub next_node_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompositionRootSnapshot {
    Layer { layer_id: String, bus: String },
    Group { group_id: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CompositionGroupSnapshot {
    pub group_id: String,
    pub name: String,
    #[serde(default)]
    pub member_layer_ids: Vec<String>,
    pub opacity: f32,
    pub transform: SpatialTransform,
    pub rack: VisualRackSnapshot,
    /// A group matte is independent from Mask nodes in the group's rack.
    #[serde(default)]
    pub matte: Option<CreativeMatteSnapshot>,
    #[serde(default)]
    pub solo: bool,
    #[serde(default)]
    pub bypass: bool,
    pub bus: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CreativeCompositionSnapshot {
    /// Latest transactional authoring/preflight result. Empty means there is
    /// no outstanding diagnostic; Main overwrites this live-session field
    /// immediately before publication.
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub master_rack: VisualRackSnapshot,
    #[serde(default)]
    pub layer_racks: Vec<(String, VisualRackSnapshot)>,
    #[serde(default)]
    pub groups: Vec<CompositionGroupSnapshot>,
    #[serde(default)]
    pub root: Vec<CompositionRootSnapshot>,
    #[serde(default = "default_matte_threshold")]
    pub bus_crossfade: f32,
    /// The B8 bus-mixer state. Additive: an absent block decodes to the
    /// exact pre-B8 dissolve bus.
    #[serde(default)]
    pub mixer: crate::mixing_boundary::BusMixerState,
    pub next_group_id: String,
}

fn creative_channel_key(channel: crate::visual_rack::MatteChannel) -> &'static str {
    match channel {
        crate::visual_rack::MatteChannel::Alpha => "alpha",
        crate::visual_rack::MatteChannel::Luma => "luma",
        crate::visual_rack::MatteChannel::Red => "red",
        crate::visual_rack::MatteChannel::Green => "green",
        crate::visual_rack::MatteChannel::Blue => "blue",
    }
}

pub fn parse_creative_channel(value: &str) -> Option<crate::visual_rack::MatteChannel> {
    Some(match value {
        "alpha" => crate::visual_rack::MatteChannel::Alpha,
        "luma" => crate::visual_rack::MatteChannel::Luma,
        "red" => crate::visual_rack::MatteChannel::Red,
        "green" => crate::visual_rack::MatteChannel::Green,
        "blue" => crate::visual_rack::MatteChannel::Blue,
        _ => return None,
    })
}

impl CreativeImageTapSnapshot {
    pub fn from_runtime(tap: crate::visual_rack::ResolvedImageTap) -> Self {
        use crate::visual_rack::ResolvedImageSource;
        let input = match tap.source {
            ResolvedImageSource::SelectedLayer {
                layer_id, stage, ..
            } => CreativeImageSourceSnapshot::SelectedLayer {
                layer_id: layer_id.get().to_string(),
                stage,
            },
            ResolvedImageSource::MissingSelectedLayer {
                saved_position,
                stage,
            } => CreativeImageSourceSnapshot::MissingSelectedLayer {
                saved_position,
                stage,
            },
            ResolvedImageSource::OneBelow => CreativeImageSourceSnapshot::OneBelow,
            ResolvedImageSource::AllBelow => CreativeImageSourceSnapshot::AllBelow,
            ResolvedImageSource::GroupOutput(group_id) => {
                CreativeImageSourceSnapshot::GroupOutput {
                    group_id: group_id.get().to_string(),
                }
            }
            ResolvedImageSource::MissingGroupOutput(group_id) => {
                CreativeImageSourceSnapshot::MissingGroupOutput {
                    group_id: group_id.get().to_string(),
                }
            }
            ResolvedImageSource::CleanProgram => CreativeImageSourceSnapshot::CleanProgram,
            ResolvedImageSource::GestureCanvas => CreativeImageSourceSnapshot::GestureCanvas,
            ResolvedImageSource::ProgramTap => CreativeImageSourceSnapshot::ProgramTap,
        };
        Self {
            input,
            timing: tap.timing,
        }
    }

    /// Resolve an ingress route without permitting diagnostic missing forms or
    /// inventing saved-position provenance for a live layer.
    pub fn to_runtime(
        &self,
        mut saved_position_of: impl FnMut(
            crate::image_routing::StableLayerId,
        ) -> Option<SavedLayerPosition>,
        group_exists: impl Fn(crate::visual_rack::GroupId) -> bool,
    ) -> Result<crate::visual_rack::ResolvedImageTap, String> {
        use crate::visual_rack::{GroupId, ResolvedImageSource};
        let decimal_id = |value: &str, kind: &str| {
            if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(format!("{kind} must be a non-zero decimal ID"));
            }
            value
                .parse::<u64>()
                .ok()
                .filter(|value| *value != 0)
                .ok_or_else(|| format!("{kind} must be a non-zero decimal ID"))
        };
        let source = match &self.input {
            CreativeImageSourceSnapshot::SelectedLayer { layer_id, stage } => {
                let layer_id =
                    crate::image_routing::StableLayerId::new(decimal_id(layer_id, "layer ID")?)
                        .ok_or_else(|| "layer ID must be non-zero".to_string())?;
                let saved_position = saved_position_of(layer_id).ok_or_else(|| {
                    format!(
                        "live layer {} has no saved-position provenance",
                        layer_id.get()
                    )
                })?;
                ResolvedImageSource::SelectedLayer {
                    layer_id,
                    saved_position,
                    stage: *stage,
                }
            }
            CreativeImageSourceSnapshot::OneBelow => ResolvedImageSource::OneBelow,
            CreativeImageSourceSnapshot::AllBelow => ResolvedImageSource::AllBelow,
            CreativeImageSourceSnapshot::GroupOutput { group_id } => {
                let group_id = GroupId::new(decimal_id(group_id, "group ID")?)
                    .ok_or_else(|| "group ID must be non-zero".to_string())?;
                if !group_exists(group_id) {
                    return Err(format!("group {} does not exist", group_id.get()));
                }
                ResolvedImageSource::GroupOutput(group_id)
            }
            CreativeImageSourceSnapshot::CleanProgram => {
                if self.timing != EdgeTiming::PreviousFrame {
                    return Err("CleanProgram is only valid at previous_frame timing".into());
                }
                ResolvedImageSource::CleanProgram
            }
            // No ID to parse and no position to invent: the singletons resolve
            // to themselves, and their availability is a planner fact rather
            // than an ingress one.
            CreativeImageSourceSnapshot::GestureCanvas => ResolvedImageSource::GestureCanvas,
            CreativeImageSourceSnapshot::ProgramTap => ResolvedImageSource::ProgramTap,
            CreativeImageSourceSnapshot::MissingSelectedLayer { .. }
            | CreativeImageSourceSnapshot::MissingGroupOutput { .. } => {
                return Err("missing creative image sources are output-only diagnostics".into());
            }
        };
        Ok(crate::visual_rack::ResolvedImageTap {
            source,
            timing: self.timing,
        })
    }
}

impl CreativeMatteSnapshot {
    pub fn from_runtime(matte: crate::visual_rack::RuntimeImageMatte) -> Self {
        let diagnostic = creative_route_diagnostic(matte.tap.source);
        Self {
            route: CreativeImageTapSnapshot::from_runtime(matte.tap),
            channel: creative_channel_key(matte.channel).into(),
            invert: matte.invert,
            amount: matte.amount,
            threshold: matte.threshold,
            softness: matte.softness,
            diagnostic,
        }
    }
}

fn creative_kind_key(kind: crate::visual_rack::NodeKindTag) -> &'static str {
    crate::visual_rack::NODE_KIND_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.tag == kind)
        .map_or("unknown", |descriptor| descriptor.key)
}

fn creative_node_params(kind: crate::visual_rack::RuntimeVisualNodeKind) -> serde_json::Value {
    use crate::visual_rack::{RuntimeMaskParams, RuntimeVisualNodeKind};
    match kind {
        RuntimeVisualNodeKind::LegacyCanonical | RuntimeVisualNodeKind::LegacyTemporal => {
            serde_json::json!({})
        }
        RuntimeVisualNodeKind::Transform(value) => serde_json::json!({
            "position": value.position,
            "scale": value.scale,
            "anchor": value.anchor,
            "rotation_deg": value.rotation_deg,
            "skew_deg": value.skew_deg,
            "skew_axis_deg": value.skew_axis_deg,
            "fit_mode": value.fit,
            "crop_left": value.crop[0],
            "crop_top": value.crop[1],
            "crop_right": value.crop[2],
            "crop_bottom": value.crop[3],
            "edge_mode": value.edge,
            "sampling": value.sampling,
        }),
        RuntimeVisualNodeKind::DigitalColor(value) => {
            serde_json::to_value(value).unwrap_or_default()
        }
        RuntimeVisualNodeKind::Key(value) => serde_json::to_value(value).unwrap_or_default(),
        RuntimeVisualNodeKind::Cellular(value) => serde_json::to_value(value).unwrap_or_default(),
        RuntimeVisualNodeKind::Shift(value) => serde_json::to_value(value).unwrap_or_default(),
        RuntimeVisualNodeKind::Grain(value) => serde_json::to_value(value).unwrap_or_default(),
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Rectangle(value)) => serde_json::json!({
            "variant": "rectangle",
            "rectangle_center": value.center,
            "rectangle_size": value.size,
            "rectangle_rotation_deg": value.rotation_deg,
            "rectangle_feather": value.feather,
            "rectangle_invert": value.invert,
        }),
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Ellipse(value)) => serde_json::json!({
            "variant": "ellipse",
            "ellipse_center": value.center,
            "ellipse_radii": value.radii,
            "ellipse_rotation_deg": value.rotation_deg,
            "ellipse_feather": value.feather,
            "ellipse_invert": value.invert,
        }),
        RuntimeVisualNodeKind::Mask(RuntimeMaskParams::Image(value)) => serde_json::json!({
            "variant": "image",
            "image_tap": CreativeImageTapSnapshot::from_runtime(value.tap),
            "image_channel": creative_channel_key(value.channel),
            "image_invert": value.invert,
            "image_amount": value.amount,
            "image_threshold": value.threshold,
            "image_softness": value.softness,
        }),
        RuntimeVisualNodeKind::Displace(value) => serde_json::json!({
            "donor_tap": CreativeImageTapSnapshot::from_runtime(value.tap),
            "amount_x": value.amount_x,
            "amount_y": value.amount_y,
            "boundary": value.boundary,
            "diagnostic": creative_route_diagnostic(value.tap.source),
        }),
        // Both image slots and both motion slots are published by slot index,
        // each with its own diagnostic, so a controller can tell which of the
        // four routes lost its donor without inferring it from an ordering.
        RuntimeVisualNodeKind::Symmetry(value) => serde_json::json!({
            "symmetry_mode": value.mode,
            "symmetry_boundary": value.boundary,
            "symmetry_seed": value.seed,
            "symmetry_base_folds": value.base_folds,
            "symmetry_fold_offset": value.fold_offset,
            "symmetry_effective_folds": value.values().effective_folds(),
            "symmetry_radial_phase_deg": value.radial_phase_deg,
            "symmetry_orbit_phase": value.orbit_phase,
            "symmetry_planar_axis_deg": value.planar_axis_deg,
            "symmetry_planar_phase": value.planar_phase,
            "symmetry_cell_skew": value.cell_skew,
            "symmetry_spiral_scale": value.spiral_scale,
            "symmetry_orbit_radius": value.orbit_radius,
            "symmetry_orbit_spin_deg": value.orbit_spin_deg,
            "symmetry_motion_gain": value.motion_gain,
            "symmetry_hue_span": value.hue_span,
            "symmetry_center": value.center,
            "symmetry_source_carrier": value.source_mask.carrier,
            "symmetry_source_donor0": value.source_mask.donor0,
            "symmetry_source_donor1": value.source_mask.donor1,
            "symmetry_source_history": value.source_mask.clean_history,
            "symmetry_motion_slot0": value.motion_mask.slot0,
            "symmetry_motion_slot1": value.motion_mask.slot1,
            "symmetry_donor0_tap": CreativeImageTapSnapshot::from_runtime(value.donors[0]),
            "symmetry_donor1_tap": CreativeImageTapSnapshot::from_runtime(value.donors[1]),
            "symmetry_motion0_donor": creative_motion_donor(value.motion[0]),
            "symmetry_motion1_donor": creative_motion_donor(value.motion[1]),
            "symmetry_exact_bypass": value.is_exact_bypass(),
            "donor0_diagnostic": creative_route_diagnostic(value.donors[0].source),
            "donor1_diagnostic": creative_route_diagnostic(value.donors[1].source),
            "motion0_diagnostic": creative_motion_diagnostic(value.motion[0]),
            "motion1_diagnostic": creative_motion_diagnostic(value.motion[1]),
        }),
        RuntimeVisualNodeKind::Residual(value) => serde_json::json!({
            "structure_tap": CreativeImageTapSnapshot::from_runtime(value.structure),
            "detail_tap": CreativeImageTapSnapshot::from_runtime(value.detail),
            "block": value.block,
            "quantization": value.quantization,
            "mix": value.mix,
            "detail_gain": value.detail_gain,
            "seed": value.seed,
            "diagnostic": creative_two_slot_route_diagnostic(value.structure.source, value.detail.source),
        }),
        // The digest is the node's whole authored surface; the document body
        // never rides the snapshot (it is bounded at a megabyte and the
        // snapshot broadcasts every frame). The panel reads the summary the
        // host publishes beside the racks.
        RuntimeVisualNodeKind::Study(value) => serde_json::json!({
            "document_digest": value.digest_hex(),
            "study_exact_bypass": value.is_exact_bypass(),
        }),
        RuntimeVisualNodeKind::ScanProcessor(value) => serde_json::json!({
            "scan_lines": value.lines,
            "scan_samples": value.samples_per_line,
            "scan_amount": value.amount,
            "scan_ribbon_width": value.ribbon_width,
            "scan_velocity_mix": value.velocity_mix,
            "scan_tilt_x": value.tilt_x,
            "scan_tilt_y": value.tilt_y,
            "scan_perspective": value.perspective,
            "scan_s_curve": value.s_curve,
            "scan_skew": value.skew,
            "scan_collapse": value.collapse,
            "scan_reverse_h": value.reverse_h,
            "scan_reverse_v": value.reverse_v,
            "scan_osc_amount": value.osc_amount,
            "scan_osc_freq": value.osc_freq,
            "scan_osc_lock": value.osc_lock,
            "scan_lissajous": value.lissajous,
            "scan_mono": value.mono,
            "scan_hue": value.hue,
            // Derived, read-only: the deflection wake law and the instanced
            // draw the authored geometry requests.
            "scan_exact_bypass": value.is_exact_bypass(),
            "scan_vertex_count": value.vertex_count(),
        }),
        RuntimeVisualNodeKind::BlockDct(value) => serde_json::json!({
            "dct_amount": value.amount,
            "dct_quantize": value.quantize,
            "dct_hf_penalty": value.hf_penalty,
            "dct_chroma_crush": value.chroma_crush,
            "dct_block": value.block,
            // Derived, read-only: the mapped block edge in texels.
            "dct_block_edge": value.block_edge(),
        }),
        RuntimeVisualNodeKind::PixelSort(value) => serde_json::json!({
            "sort_amount": value.amount,
            "sort_threshold": value.threshold,
        }),
        RuntimeVisualNodeKind::Avalanche(value) => serde_json::json!({
            "avalanche_amount": value.amount,
            "avalanche_run": value.run,
            "avalanche_axis": value.axis,
        }),
    }
}

/// Publish one motion route by slot in the established [`MotionDonorSnapshot`]
/// shape: a stable decimal layer ID for a live donor, a saved position only for
/// a retained tombstone. The stable ID is the panel's addressing key — the same
/// one `CreativeImageSourceSnapshot::SelectedLayer` already publishes for an
/// image slot — and a tombstone never carries one, so it can never rebind.
fn creative_motion_donor(donor: crate::motion::MotionDonor) -> serde_json::Value {
    let snapshot = match donor {
        crate::motion::MotionDonor::None => MotionDonorSnapshot::None,
        crate::motion::MotionDonor::Selected {
            layer_id,
            saved_position,
        } => MotionDonorSnapshot::Selected {
            layer_id: layer_id.get().to_string(),
            saved_position: saved_position.get(),
        },
        crate::motion::MotionDonor::Missing { saved_position } => MotionDonorSnapshot::Missing {
            saved_position: saved_position.get(),
        },
    };
    serde_json::to_value(snapshot).unwrap_or(serde_json::Value::Null)
}

/// Operator-facing text for a motion route that resolves to nothing.
fn creative_motion_diagnostic(donor: crate::motion::MotionDonor) -> String {
    match donor {
        crate::motion::MotionDonor::Missing { saved_position } => {
            format!("missing saved layer {}", saved_position.get())
        }
        crate::motion::MotionDonor::None | crate::motion::MotionDonor::Selected { .. } => {
            String::new()
        }
    }
}

/// Operator-facing text for a two-slot route pair. Each dead slot names itself
/// so a tombstone can never be read as belonging to the other route, and a
/// fully live pair stays empty exactly as the single-route diagnostic does.
fn creative_two_slot_route_diagnostic(
    structure: crate::visual_rack::ResolvedImageSource,
    detail: crate::visual_rack::ResolvedImageSource,
) -> String {
    let mut parts = Vec::new();
    for (label, source) in [("structure", structure), ("detail", detail)] {
        let diagnostic = creative_route_diagnostic(source);
        if !diagnostic.is_empty() {
            parts.push(format!("{label}: {diagnostic}"));
        }
    }
    parts.join("; ")
}

/// Operator-facing text for a route that resolves to nothing. Empty means the
/// route is live; a retained tombstone always names its saved provenance.
fn creative_route_diagnostic(source: crate::visual_rack::ResolvedImageSource) -> String {
    match source {
        crate::visual_rack::ResolvedImageSource::MissingSelectedLayer {
            saved_position, ..
        } => {
            format!("missing saved layer {}", saved_position.get())
        }
        crate::visual_rack::ResolvedImageSource::MissingGroupOutput(group_id) => {
            format!("missing group output {}", group_id.get())
        }
        _ => String::new(),
    }
}

impl VisualRackSnapshot {
    pub fn from_runtime(rack: &crate::visual_rack::RuntimeVisualRack) -> Self {
        let nodes = rack
            .iter()
            .map(|node| VisualNodeSnapshot {
                node_id: node.stable_id.get().to_string(),
                enabled: node.enabled,
                wet: node.wet,
                blend: serde_json::to_value(node.blend)
                    .ok()
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .unwrap_or_else(|| "normal".into()),
                kind: creative_kind_key(node.kind.tag()).into(),
                params: creative_node_params(node.kind),
            })
            .collect();
        Self {
            nodes,
            next_node_id: rack.next_node_id_raw().to_string(),
        }
    }
}

impl CreativeCompositionSnapshot {
    pub fn from_runtime(
        master_rack: &crate::visual_rack::RuntimeVisualRack,
        layer_racks: &[(
            crate::image_routing::StableLayerId,
            crate::visual_rack::RuntimeVisualRack,
        )],
        composition: &crate::composition::RuntimeComposition,
    ) -> Self {
        use crate::composition::{BusAssignment, RuntimeRootItem};
        let bus_key = |bus| match bus {
            BusAssignment::Program => "program",
            BusAssignment::A => "a",
            BusAssignment::B => "b",
        };
        let groups = composition
            .groups()
            .map(|group| CompositionGroupSnapshot {
                group_id: group.id.get().to_string(),
                name: group.name.as_str().to_owned(),
                member_layer_ids: group
                    .members
                    .iter()
                    .map(|layer_id| layer_id.get().to_string())
                    .collect(),
                opacity: group.opacity,
                transform: group.transform,
                rack: VisualRackSnapshot::from_runtime(&group.rack),
                matte: group.matte.map(CreativeMatteSnapshot::from_runtime),
                solo: group.solo,
                bypass: group.bypass,
                bus: bus_key(group.bus).into(),
            })
            .collect();
        let root = composition
            .root()
            .iter()
            .map(|item| match *item {
                RuntimeRootItem::Layer { layer_id, bus } => CompositionRootSnapshot::Layer {
                    layer_id: layer_id.get().to_string(),
                    bus: bus_key(bus).into(),
                },
                RuntimeRootItem::Group { group_id } => CompositionRootSnapshot::Group {
                    group_id: group_id.get().to_string(),
                },
            })
            .collect();
        Self {
            status: String::new(),
            master_rack: VisualRackSnapshot::from_runtime(master_rack),
            layer_racks: layer_racks
                .iter()
                .map(|(layer_id, rack)| {
                    (
                        layer_id.get().to_string(),
                        VisualRackSnapshot::from_runtime(rack),
                    )
                })
                .collect(),
            groups,
            root,
            bus_crossfade: composition.bus_crossfade(),
            mixer: composition.mixer(),
            next_group_id: composition.next_group_id_raw().to_string(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistorySnapshot {
    #[serde(default)]
    pub can_undo: bool,
    #[serde(default)]
    pub can_redo: bool,
    #[serde(default)]
    pub undo_depth: usize,
    #[serde(default)]
    pub redo_depth: usize,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub max_entries: usize,
    #[serde(default)]
    pub max_bytes: u64,
    #[serde(default)]
    pub generation: u64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub undo_label: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub redo_label: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
}

impl HistorySnapshot {
    pub fn from_history<T>(
        history: &crate::history::ManualHistory<T>,
        status: impl Into<String>,
    ) -> Self {
        let metrics = history.metrics();
        Self {
            can_undo: metrics.can_undo,
            can_redo: metrics.can_redo,
            undo_depth: metrics.undo_depth,
            redo_depth: metrics.redo_depth,
            bytes: metrics.bytes,
            max_entries: metrics.max_entries,
            max_bytes: metrics.max_bytes,
            generation: metrics.generation,
            undo_label: history.undo_label().unwrap_or_default().to_string(),
            redo_label: history.redo_label().unwrap_or_default().to_string(),
            status: status.into(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetSummarySnapshot {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub created_ordinal: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetLibrarySnapshot {
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub presets: Vec<PresetSummarySnapshot>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
}

impl PresetLibrarySnapshot {
    pub fn from_library(
        revision: u64,
        library: &crate::preset::PresetLibrary,
        status: impl Into<String>,
    ) -> Self {
        Self {
            revision,
            presets: library
                .iter()
                .map(|preset| PresetSummarySnapshot {
                    id: preset.id.get().to_string(),
                    name: preset.name.clone(),
                    kind: preset.payload.kind().key().to_string(),
                    created_ordinal: preset.created_ordinal,
                })
                .collect(),
            status: status.into(),
        }
    }
}

/// Convert a read-only journal scan into the two additive AppSnapshot fields.
/// A bad tail remains visible even when a valid prefix offers recovery.
#[allow(
    dead_code,
    reason = "retained as a pure legacy-scan adapter while App publishes asynchronous writer status"
)]
pub fn recovery_snapshot_fields(scan: &crate::recovery_journal::RecoveryScan) -> (bool, String) {
    let status = scan.warning.clone().unwrap_or_else(|| {
        scan.latest.as_ref().map_or_else(String::new, |checkpoint| {
            format!(
                "Recovery checkpoint {} is available ({} valid entr{}).",
                checkpoint.sequence,
                scan.valid_entries,
                if scan.valid_entries == 1 { "y" } else { "ies" }
            )
        })
    });
    (scan.recovery_available(), status)
}

pub const EXPORT_MOTION_SNAPSHOT_SCHEMA_VERSION: u16 = 3;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExportMotionScopeSnapshot {
    #[serde(default)]
    pub scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved_position: Option<u32>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub stable_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_tap_id: String,
    #[serde(default)]
    pub algorithm_version: u16,
    #[serde(default)]
    pub requested_source: String,
    #[serde(default)]
    pub lattice_quality: String,
    #[serde(default)]
    pub source_origin: String,
    #[serde(default)]
    pub rendered_source_origin: String,
    #[serde(default)]
    pub field_planned: bool,
    #[serde(default)]
    pub field_attached: bool,
    #[serde(default)]
    pub source_diagnostic: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub codec_provenance: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_ordinal: Option<u64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub codec_product_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec_transition_count: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec_elapsed_seconds: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub donor_saved_position: Option<u32>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub donor_stable_id: String,
    #[serde(default)]
    pub carrier: String,
    #[serde(default)]
    pub transplant_admitted: bool,
    #[serde(default)]
    pub shutter_active: bool,
    #[serde(default)]
    pub shutter_angle_degrees: f32,
    #[serde(default)]
    pub shutter_quality: String,
    #[serde(default)]
    pub shutter_sample_count: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportMotionSnapshot {
    pub schema_version: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_frame: Option<u64>,
    #[serde(default)]
    pub algorithm_version: u16,
    #[serde(default)]
    pub scopes: Vec<ExportMotionScopeSnapshot>,
    #[serde(default)]
    pub scopes_truncated: bool,
    /// Motion vectors and codec provenance are deterministic inputs, but the
    /// project deliberately makes no cross-adapter floating-point pixel claim.
    #[serde(default)]
    pub cross_gpu_pixel_identity_guaranteed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl Default for ExportMotionSnapshot {
    fn default() -> Self {
        Self {
            schema_version: EXPORT_MOTION_SNAPSHOT_SCHEMA_VERSION,
            accepted_frame: None,
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            scopes: Vec::new(),
            scopes_truncated: false,
            cross_gpu_pixel_identity_guaranteed: false,
            warnings: Vec::new(),
        }
    }
}

impl ExportMotionSnapshot {
    pub fn from_export(
        metadata: &crate::render_export::ExportMotionMetadata,
        warnings: &[String],
    ) -> Self {
        const MAX_SNAPSHOT_WARNINGS: usize = 128;
        let scopes = metadata
            .scopes
            .iter()
            .map(|scope| {
                let (scope_key, saved_position, stable_id, source_tap_id) = match scope.scope {
                    crate::render_export::ExportMotionScopeIdentity::Master => {
                        ("master", None, String::new(), String::new())
                    }
                    crate::render_export::ExportMotionScopeIdentity::Layer {
                        saved_position,
                        stable_id,
                        source_tap_id,
                    } => (
                        "layer",
                        Some(saved_position),
                        stable_id.to_string(),
                        source_tap_id.to_string(),
                    ),
                };
                ExportMotionScopeSnapshot {
                    scope: scope_key.into(),
                    saved_position,
                    stable_id,
                    source_tap_id,
                    algorithm_version: scope.algorithm_version,
                    requested_source: motion_source_key(scope.requested_source).into(),
                    lattice_quality: lattice_quality_key(scope.lattice_quality).into(),
                    source_origin: motion_origin_key(scope.source_origin).into(),
                    rendered_source_origin: motion_origin_key(scope.rendered_source_origin).into(),
                    field_planned: scope.field_planned,
                    field_attached: scope.field_attached,
                    source_diagnostic: motion_diagnostic_key(scope.source_diagnostic).into(),
                    codec_provenance: scope
                        .codec_provenance
                        .map(codec_provenance_key)
                        .unwrap_or_default()
                        .into(),
                    source_generation: scope.source_generation,
                    frame_ordinal: scope.frame_ordinal,
                    codec_product_sha256: scope
                        .codec_product_sha256
                        .map(crate::render_export::sha256_bytes_hex)
                        .unwrap_or_default(),
                    codec_transition_count: scope.codec_transition_count,
                    codec_elapsed_seconds: scope.codec_elapsed_seconds,
                    donor_saved_position: scope.donor_saved_position,
                    donor_stable_id: scope
                        .donor_stable_id
                        .map_or_else(String::new, |id| id.to_string()),
                    carrier: motion_carrier_key(scope.carrier).into(),
                    transplant_admitted: scope.transplant_admitted,
                    shutter_active: scope.shutter_active,
                    shutter_angle_degrees: scope.shutter_angle_degrees,
                    shutter_quality: shutter_quality_key(scope.shutter_quality).into(),
                    shutter_sample_count: scope.shutter_sample_count,
                }
            })
            .collect();
        Self {
            schema_version: EXPORT_MOTION_SNAPSHOT_SCHEMA_VERSION,
            accepted_frame: metadata.accepted_frame,
            algorithm_version: metadata.algorithm_version,
            scopes,
            scopes_truncated: metadata.scopes_truncated,
            cross_gpu_pixel_identity_guaranteed: false,
            warnings: warnings
                .iter()
                .take(MAX_SNAPSHOT_WARNINGS)
                .cloned()
                .collect(),
        }
    }
}

const fn motion_source_key(value: crate::motion::MotionFieldSource) -> &'static str {
    match value {
        crate::motion::MotionFieldSource::Auto => "auto",
        crate::motion::MotionFieldSource::CodecVectors => "codec_vectors",
        crate::motion::MotionFieldSource::Lattice => "lattice",
        crate::motion::MotionFieldSource::Procedural(kind) => kind.source_key(),
    }
}

const fn lattice_quality_key(value: crate::motion::MotionLatticeQuality) -> &'static str {
    match value {
        crate::motion::MotionLatticeQuality::Draft => "draft",
        crate::motion::MotionLatticeQuality::Live => "live",
        crate::motion::MotionLatticeQuality::High => "high",
    }
}

const fn motion_origin_key(value: crate::motion::MotionFieldOrigin) -> &'static str {
    match value {
        crate::motion::MotionFieldOrigin::None => "none",
        crate::motion::MotionFieldOrigin::CodecVectors => "codec_vectors",
        crate::motion::MotionFieldOrigin::Lattice => "lattice",
        crate::motion::MotionFieldOrigin::LatticeFallback => "lattice_fallback",
        crate::motion::MotionFieldOrigin::Procedural(kind) => kind.source_key(),
    }
}

const fn motion_diagnostic_key(value: crate::motion::MotionSourceDiagnostic) -> &'static str {
    match value {
        crate::motion::MotionSourceDiagnostic::None => "none",
        crate::motion::MotionSourceDiagnostic::CodecUnavailable => "codec_unavailable",
        crate::motion::MotionSourceDiagnostic::CodecUnavailableFallback => {
            "codec_unavailable_fallback"
        }
    }
}

const fn codec_provenance_key(value: crate::video::CodecMotionProvenance) -> &'static str {
    match value {
        crate::video::CodecMotionProvenance::FfmpegExportMvs => "ffmpeg_export_mvs",
    }
}

const fn motion_carrier_key(value: crate::motion::MotionCarrier) -> &'static str {
    match value {
        crate::motion::MotionCarrier::Transparent => "transparent",
        crate::motion::MotionCarrier::Black => "black",
        crate::motion::MotionCarrier::FirstSourceFrame => "first_source_frame",
    }
}

const fn shutter_quality_key(value: crate::motion::CurvedShutterQuality) -> &'static str {
    match value {
        crate::motion::CurvedShutterQuality::Sharp => "sharp",
        crate::motion::CurvedShutterQuality::Draft => "draft",
        crate::motion::CurvedShutterQuality::Live => "live",
        crate::motion::CurvedShutterQuality::High => "high",
    }
}

/// Full app state snapshot sent to the browser each frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSnapshot {
    #[serde(rename = "type")]
    pub msg_type: String,
    /// Additive browser publication protocol. Version 1 is the legacy full
    /// state shape; version 2 adds revision-checked live-domain messages.
    #[serde(default)]
    pub wire_version: u16,
    #[serde(default)]
    pub authored_revision: u64,
    #[serde(default)]
    pub operational_revision: u64,
    #[serde(default)]
    pub telemetry_revision: u64,
    #[serde(default)]
    pub live_interval_ms: u16,
    pub effects: EffectsSnapshot,
    /// Authored transform applied after the layer stack. Missing data from an
    /// older server/client is the exact legacy full-frame identity.
    #[serde(default)]
    pub master_transform: SpatialTransform,
    pub ntsc: NtscSnapshot,
    /// B5 codec mosh live diagnostics: additive, defaulting to the inactive
    /// stage for older snapshots.
    #[serde(default)]
    pub codec_mosh: CodecMoshLiveSnapshot,
    /// Host-local admission limits for future media source allocations. This is
    /// deliberately independent from patches and output/export dimensions.
    #[serde(default)]
    pub media_safety: MediaSafetySnapshot,
    /// Host-session preference for framing future interactive layers. It is
    /// intentionally absent from artistic PatchState and never rewrites an
    /// existing layer.
    #[serde(default = "default_new_layer_fit")]
    pub new_layer_fit: FitMode,
    /// Host-session authored proxy settings: the exact tuple future proxy
    /// encodes and cache consultations use. Each tuple is its own
    /// content-addressed cache key by design, patches never own this value,
    /// and changing it rewrites no live layer and invalidates no artifact.
    #[serde(default)]
    pub proxy_settings: ProxySettingsSnapshot,
    pub layers: Vec<LayerSnapshot>,
    /// Prepared cross-layer scene bank and its current staging state. This is
    /// additive so an older engine/panel pair continues to deserialize an
    /// otherwise complete snapshot as an empty performance set.
    #[serde(default)]
    pub performance: PerformanceSnapshot,
    /// Monotonic topology generation used to reject stale multi-controller
    /// reorder requests. Zero means an older server/client with no revision.
    #[serde(default)]
    pub layer_stack_revision: u64,
    /// Independent optimistic-concurrency token for rack/group topology.
    /// It is live session state and is deliberately absent from PatchState.
    #[serde(default)]
    pub composition_revision: u64,
    #[serde(default)]
    pub creative: CreativeCompositionSnapshot,
    /// Stable planner refusals. Legacy prose remains in `creative.status`,
    /// while new controllers branch only on these closed codes and scopes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraint_diagnostics: Vec<crate::diagnostics::ConstraintDiagnostic>,
    /// Bounded first visual page retained for older bundled panels.
    pub library: Vec<String>,
    /// Revision/count/status plus the bounded current page. The complete
    /// index is available only through the authenticated paging endpoint.
    #[serde(default)]
    pub library_index: crate::library_index::LibraryIndexSnapshot,
    /// Legacy mirror of `program_frozen` for older bundled clients.
    pub paused: bool,
    #[serde(default)]
    pub program_frozen: bool,
    #[serde(default)]
    pub media_frozen: bool,
    /// Modulation matrix state (BPM, LFOs, routings)
    #[serde(default)]
    pub modulation: ModSnapshot,
    /// Audio input analysis state
    #[serde(default)]
    pub audio: AudioSnapshot,
    /// MIDI input state
    #[serde(default)]
    pub midi: MidiSnapshot,
    /// Typed controller-profile identity plus live MIDI supervisor truth.
    #[serde(default)]
    pub controller_runtime: ControllerRuntimeSnapshot,
    /// Read-only OSC listener, exposure, and bounded counter truth.
    #[serde(default)]
    pub osc_runtime: OscRuntimeSnapshot,
    /// Temporal (feedback/slit-scan) effect state
    #[serde(default)]
    pub temporal: TemporalSnapshot,
    /// Gesture-field recording truth. Additive: an older panel deserializes an
    /// otherwise complete snapshot as a disarmed, empty recording.
    #[serde(default)]
    pub gesture: GestureStatusSnapshot,
    /// B9 performance-recorder truth. Additive: an older panel deserializes
    /// an otherwise complete snapshot as a disarmed recorder with no take.
    /// (`performance` is the clip/scene block, so this one carries its full
    /// name.)
    #[serde(default)]
    pub performance_recorder: PerformanceStatusSnapshot,
    /// Program-wide Curved Shutter authoring and read-only execution truth.
    /// Faraday controls remain disabled for this master scope.
    #[serde(default)]
    pub master_motion: MotionSnapshot,
    /// Spout output state
    #[serde(default)]
    pub spout: SpoutSnapshot,
    /// Bounded live recorder lifecycle and drop counters. Host paths never
    /// cross this boundary; only the committed artifact's file name is shown.
    #[serde(default)]
    pub recorder: ProgramRecorderSnapshot,
    /// Preview/editor telemetry plus endpoint-scoped calibration controls.
    /// The snapshot is informational and has no audience/composite pixels.
    #[serde(default)]
    pub stage_health: crate::stage_health::StageHealthSnapshot,
    /// B11 monitoring bay: instrument bitmaps and the live source strip,
    /// additive and defaulting to the inactive (empty-payload) bay. The
    /// payloads are present only while an observer keeps the bay armed.
    #[serde(default)]
    pub monitor_bay: crate::monitor_bay::MonitorBaySnapshot,
    /// Remote control URL (LAN address with access token) for the QR code
    #[serde(default)]
    #[serde(skip_serializing_if = "ControlAccessUrl::is_empty")]
    pub remote_url: ControlAccessUrl,
    /// Independent LAN HTTPS listener truth. Empty/default older snapshots are
    /// treated as unavailable by the panel rather than preserving a stale QR.
    #[serde(default)]
    pub remote_status: String,
    /// Whether the fullscreen output window is open
    #[serde(default)]
    pub output_window: bool,
    /// Exact state of the legacy fullscreen Output controlled by
    /// `set_output_window`. The older `output_window` aggregate is retained for
    /// compatibility with clients that treated StageMap surfaces as Output.
    #[serde(default)]
    pub legacy_output_window: bool,
    /// Session-local display selected for the legacy fullscreen Output. An
    /// empty identifier means Automatic; unlike StageMap, this is deliberately
    /// absent from artistic patch state.
    #[serde(default)]
    pub output_display: String,
    /// Current host display inventory. Opaque identifiers are valid only for
    /// this running host session and are revalidated by Main before use.
    #[serde(default)]
    pub output_displays: Vec<OutputDisplaySnapshot>,
    /// Generation of the retained host monitor inventory used to construct
    /// `output_displays`. Current clients echo this with display-targeting
    /// commands; legacy clients omit it and still receive ID revalidation.
    #[serde(default)]
    pub output_display_generation: u64,
    /// Non-empty when creating or maintaining an output surface failed.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub output_error: String,
    /// Patch morph crossfader state
    #[serde(default)]
    pub morph: MorphSnapshot,
    /// Output is currently cut to black
    #[serde(default)]
    pub blackout: bool,
    /// Export progress: 0.0 = idle, 0.0..1.0 = rendering, 1.0 = done
    #[serde(default)]
    pub export_progress: f32,
    /// Non-empty when export encountered an error
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub export_error: String,
    /// Recoverable source substitutions made by the current or most recent
    /// export. Kept separate from `export_error` so a compatible legacy
    /// render can succeed while still reporting exactly what rendered black.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub export_warnings: Vec<String>,
    /// Stable lifecycle state:
    /// idle | running | cancelling | succeeded | failed | cancelled.
    #[serde(default)]
    pub export_status: String,
    /// Bounded accepted-frame Motion provenance for the current/last export.
    #[serde(default)]
    pub export_motion: ExportMotionSnapshot,
    /// Transactional manual undo/redo truth. Automation never contributes.
    #[serde(default)]
    pub history: HistorySnapshot,
    /// Values-only, identity-safe creative preset library.
    #[serde(default)]
    pub presets: PresetLibrarySnapshot,
    /// A valid durable checkpoint is offered, never auto-applied.
    #[serde(default)]
    pub recovery_available: bool,
    /// Recovery scan/restore/discard outcome, including corrupt-tail warnings.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub recovery_status: String,
    /// Result of the most recent frictionless patch capture. The engine owns
    /// this text so clients never manufacture a successful-save indication.
    #[serde(default)]
    pub patch_save_status: String,
    /// Result of the most recent exact-load or Apply Look request.
    #[serde(default)]
    pub patch_load_status: String,
    /// Number of coalesced actions waiting for the next downbeat.
    #[serde(default)]
    pub quantized_pending: usize,
}

impl Default for AppSnapshot {
    fn default() -> Self {
        Self {
            msg_type: "state".to_string(),
            wire_version: 2,
            authored_revision: 0,
            operational_revision: 0,
            telemetry_revision: 0,
            live_interval_ms: 84,
            effects: EffectsSnapshot::default(),
            master_transform: SpatialTransform::default(),
            ntsc: NtscSnapshot::default(),
            codec_mosh: CodecMoshLiveSnapshot::default(),
            media_safety: MediaSafetySnapshot::default(),
            new_layer_fit: default_new_layer_fit(),
            proxy_settings: ProxySettingsSnapshot::default(),
            layers: Vec::new(),
            performance: PerformanceSnapshot::default(),
            layer_stack_revision: 0,
            composition_revision: 0,
            creative: CreativeCompositionSnapshot::default(),
            constraint_diagnostics: Vec::new(),
            library: Vec::new(),
            library_index: crate::library_index::LibraryIndexSnapshot::default(),
            paused: false,
            program_frozen: false,
            media_frozen: false,
            modulation: ModSnapshot::default(),
            audio: AudioSnapshot::default(),
            midi: MidiSnapshot::default(),
            controller_runtime: ControllerRuntimeSnapshot::default(),
            osc_runtime: OscRuntimeSnapshot::default(),
            temporal: TemporalSnapshot::default(),
            gesture: GestureStatusSnapshot::default(),
            performance_recorder: PerformanceStatusSnapshot::default(),
            master_motion: MotionSnapshot::default(),
            spout: SpoutSnapshot::default(),
            recorder: ProgramRecorderSnapshot::default(),
            stage_health: crate::stage_health::StageHealthSnapshot::default(),
            monitor_bay: crate::monitor_bay::MonitorBaySnapshot::default(),
            remote_url: ControlAccessUrl::default(),
            remote_status: String::new(),
            output_window: false,
            legacy_output_window: false,
            output_display: String::new(),
            output_displays: Vec::new(),
            output_display_generation: 0,
            output_error: String::new(),
            morph: MorphSnapshot::default(),
            blackout: false,
            export_progress: 0.0,
            export_error: String::new(),
            export_warnings: Vec::new(),
            export_status: "idle".to_string(),
            export_motion: ExportMotionSnapshot::default(),
            history: HistorySnapshot::default(),
            presets: PresetLibrarySnapshot::default(),
            recovery_available: false,
            recovery_status: String::new(),
            patch_save_status: String::new(),
            patch_load_status: String::new(),
            quantized_pending: 0,
        }
    }
}

/// Medium-frequency host/runtime truth. It is deliberately separate from the
/// authored graph and from fast meters, even though both domains share one
/// newest-only transport publication for bounded fan-out.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebOperationalSnapshot {
    pub program_frozen: bool,
    pub media_frozen: bool,
    pub export_progress: f32,
    pub export_error: String,
    pub export_status: String,
    pub export_warnings: Vec<String>,
    pub export_motion: ExportMotionSnapshot,
    pub controller_runtime: ControllerRuntimeSnapshot,
    pub osc_runtime: OscRuntimeSnapshot,
    pub spout: SpoutSnapshot,
    pub recorder: ProgramRecorderSnapshot,
    pub remote_url: ControlAccessUrl,
    pub remote_status: String,
    pub output_window: bool,
    pub output_error: String,
    pub output_display: String,
    pub output_displays: Vec<OutputDisplaySnapshot>,
    pub output_display_generation: u64,
    pub blackout: bool,
    pub recovery_available: bool,
    pub recovery_status: String,
    pub patch_save_status: String,
    pub patch_load_status: String,
    pub quantized_pending: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraint_diagnostics: Vec<crate::diagnostics::ConstraintDiagnostic>,
}

/// The bounded 10–15 Hz meter domain. No layer graph, scene bank, library
/// listing, rack, or other authored collection is carried here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebFastTelemetrySnapshot {
    pub modulation: ModSnapshot,
    pub audio: AudioSnapshot,
    pub midi: MidiSnapshot,
    pub temporal: TemporalSnapshot,
    pub gesture: GestureStatusSnapshot,
    pub performance_recorder: PerformanceStatusSnapshot,
    pub master_motion: MotionSnapshot,
    pub codec_mosh: CodecMoshLiveSnapshot,
    pub morph: MorphSnapshot,
    pub stage_health: crate::stage_health::StageHealthSnapshot,
    pub monitor_bay: crate::monitor_bay::MonitorBaySnapshot,
}

/// Revision-checked live publication. A client may apply it only to the exact
/// authored base named here; a mismatch requires a fresh full state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebLiveSnapshot {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub wire_version: u16,
    pub authored_revision: u64,
    pub operational_revision: u64,
    pub telemetry_revision: u64,
    pub live_interval_ms: u16,
    pub operational: WebOperationalSnapshot,
    pub telemetry: WebFastTelemetrySnapshot,
}

/// A JSON-friendly snapshot of the current effect parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsSnapshot {
    /// Public deterministic pattern seed. Zero selects the legacy sequence.
    #[serde(default)]
    pub random_seed: u32,
    pub pixelate: f32,
    pub downsample: f32,
    #[serde(default)]
    pub shift_amount: f32,
    #[serde(default = "default_shift_block_size")]
    pub shift_block_size: f32,
    #[serde(default = "default_shift_density")]
    pub shift_density: f32,
    #[serde(default = "default_shift_speed")]
    pub shift_speed: f32,
    pub rgb_split: f32,
    pub hue_shift: f32,
    pub saturation: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub posterize: f32,
    pub invert: bool,
    pub grain_intensity: f32,
    pub grain_size: f32,
    pub grain_algo: u32,
    pub color_grain: bool,
    pub vignette: f32,
    pub color_drift: f32,
    pub breathe_scale: f32,
    pub breathe_rotation: f32,
    pub breathe_position: f32,
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default = "default_key_color")]
    pub key_color: [f32; 3],
    #[serde(default = "default_key_threshold")]
    pub key_threshold: f32,
    #[serde(default = "default_key_softness")]
    pub key_softness: f32,
    #[serde(default = "default_key_tolerance")]
    pub key_tolerance: f32,
    #[serde(default)]
    pub cellular_amount: f32,
    #[serde(default = "default_cellular_scale")]
    pub cellular_scale: f32,
    #[serde(default = "default_cellular_warp")]
    pub cellular_warp: f32,
    #[serde(default = "default_cellular_speed")]
    pub cellular_speed: f32,
    #[serde(default)]
    pub cellular_gap_amount: f32,
    #[serde(default = "default_cellular_gap_threshold")]
    pub cellular_gap_threshold: f32,
    #[serde(default = "default_cellular_gap_softness")]
    pub cellular_gap_softness: f32,
    // B13 small effects: additive, defaulting to the exact prior path.
    #[serde(default)]
    pub contour: f32,
    #[serde(default = "default_contour_bands")]
    pub contour_bands: f32,
    #[serde(default = "default_contour_width")]
    pub contour_width: f32,
    #[serde(default)]
    pub contour_hue: f32,
    #[serde(default = "default_contour_fill")]
    pub contour_fill: f32,
    #[serde(default)]
    pub flatten: f32,
    #[serde(default = "default_flatten_levels")]
    pub flatten_levels: f32,
    #[serde(default)]
    pub contour_dither: f32,
    #[serde(default)]
    pub solarize: f32,
    #[serde(default)]
    pub negative: f32,
    #[serde(default)]
    pub negative_mode: u32,
    #[serde(default)]
    pub colourpass: f32,
    #[serde(default)]
    pub colourpass_hue: f32,
    #[serde(default = "default_colourpass_width")]
    pub colourpass_width: f32,
    #[serde(default)]
    pub edge_amount: f32,
    #[serde(default)]
    pub edge_hue: f32,
    #[serde(default)]
    pub emboss: f32,
    #[serde(default = "default_emboss_angle")]
    pub emboss_angle: f32,
    #[serde(default)]
    pub halftone: f32,
    #[serde(default = "default_halftone_pitch")]
    pub halftone_pitch: f32,
    #[serde(default)]
    pub halftone_angle: f32,
    #[serde(default)]
    pub moire: f32,
    #[serde(default = "default_moire_freq")]
    pub moire_freq: f32,
    #[serde(default)]
    pub row_smear: f32,
    #[serde(default)]
    pub bitcrush: f32,
    #[serde(default = "default_bitcrush_levels")]
    pub bitcrush_levels: f32,
    #[serde(default = "default_bitcrush_dither")]
    pub bitcrush_dither: f32,
    #[serde(default = "default_multi_grid")]
    pub multi_grid_x: f32,
    #[serde(default = "default_multi_grid")]
    pub multi_grid_y: f32,
    /// Master-only optics; layer snapshots carry their defaults.
    #[serde(default)]
    pub barrel: f32,
    #[serde(default)]
    pub chroma_aberration: f32,
    #[serde(default)]
    pub anamorphic_streak: f32,
    /// B8 key dressing: border/shadow join the key signal at both scopes.
    #[serde(default)]
    pub key_border: f32,
    #[serde(default)]
    pub key_border_color: u32,
    #[serde(default)]
    pub key_shadow: f32,
}

fn default_cellular_scale() -> f32 {
    10.0
}

fn default_contour_bands() -> f32 {
    10.0
}

fn default_contour_width() -> f32 {
    1.2
}

fn default_contour_fill() -> f32 {
    0.25
}

fn default_flatten_levels() -> f32 {
    5.0
}

fn default_colourpass_width() -> f32 {
    0.25
}

fn default_emboss_angle() -> f32 {
    45.0
}

fn default_halftone_pitch() -> f32 {
    0.4
}

fn default_moire_freq() -> f32 {
    0.4
}

fn default_bitcrush_levels() -> f32 {
    2.0
}

fn default_bitcrush_dither() -> f32 {
    1.0
}

fn default_multi_grid() -> f32 {
    1.0
}

fn default_key_color() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

fn default_key_threshold() -> f32 {
    0.5
}

fn default_key_softness() -> f32 {
    0.1
}

fn default_key_tolerance() -> f32 {
    0.15
}

fn default_cellular_warp() -> f32 {
    0.35
}

fn default_cellular_speed() -> f32 {
    0.25
}

fn default_cellular_gap_threshold() -> f32 {
    0.65
}

fn default_cellular_gap_softness() -> f32 {
    0.08
}

fn default_shift_block_size() -> f32 {
    8.0
}

fn default_shift_density() -> f32 {
    0.5
}

fn default_shift_speed() -> f32 {
    3.0
}

fn finite_effect_value(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

impl Default for EffectsSnapshot {
    fn default() -> Self {
        Self {
            random_seed: 0,
            pixelate: 1.0,
            downsample: 1.0,
            shift_amount: 0.0,
            shift_block_size: default_shift_block_size(),
            shift_density: default_shift_density(),
            shift_speed: default_shift_speed(),
            rgb_split: 0.0,
            hue_shift: 0.0,
            saturation: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            posterize: 0.0,
            invert: false,
            grain_intensity: 0.0,
            grain_size: 1.0,
            grain_algo: 0,
            color_grain: false,
            vignette: 0.0,
            color_drift: 0.0,
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
            key_mode: 0,
            key_color: default_key_color(),
            key_threshold: default_key_threshold(),
            key_softness: default_key_softness(),
            key_tolerance: default_key_tolerance(),
            cellular_amount: 0.0,
            cellular_scale: default_cellular_scale(),
            cellular_warp: default_cellular_warp(),
            cellular_speed: default_cellular_speed(),
            cellular_gap_amount: 0.0,
            cellular_gap_threshold: default_cellular_gap_threshold(),
            cellular_gap_softness: default_cellular_gap_softness(),
            contour: 0.0,
            contour_bands: default_contour_bands(),
            contour_width: default_contour_width(),
            contour_hue: 0.0,
            contour_fill: default_contour_fill(),
            flatten: 0.0,
            flatten_levels: default_flatten_levels(),
            contour_dither: 0.0,
            solarize: 0.0,
            negative: 0.0,
            negative_mode: 0,
            colourpass: 0.0,
            colourpass_hue: 0.0,
            colourpass_width: default_colourpass_width(),
            edge_amount: 0.0,
            edge_hue: 0.0,
            emboss: 0.0,
            emboss_angle: default_emboss_angle(),
            halftone: 0.0,
            halftone_pitch: default_halftone_pitch(),
            halftone_angle: 0.0,
            moire: 0.0,
            moire_freq: default_moire_freq(),
            row_smear: 0.0,
            bitcrush: 0.0,
            bitcrush_levels: default_bitcrush_levels(),
            bitcrush_dither: default_bitcrush_dither(),
            multi_grid_x: 1.0,
            multi_grid_y: 1.0,
            barrel: 0.0,
            chroma_aberration: 0.0,
            anamorphic_streak: 0.0,
            key_border: 0.0,
            key_border_color: 0,
            key_shadow: 0.0,
        }
    }
}

/// NTSC/VHS effect parameters sent to the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NtscSnapshot {
    pub enabled: bool,
    pub tape_speed: u32,
    pub chroma_loss: f32,
    pub edge_wave_enabled: bool,
    pub edge_wave_intensity: f32,
    pub edge_wave_speed: f32,
    pub head_switching_enabled: bool,
    pub head_switching_height: i32,
    pub head_switching_shift: f32,
    pub tracking_noise_enabled: bool,
    pub tracking_noise_height: i32,
    pub tracking_noise_wave: f32,
    pub tracking_noise_snow: f32,
    pub snow_intensity: f32,
    pub composite_noise_intensity: f32,
    pub luma_noise_intensity: f32,
    pub chroma_noise_intensity: f32,
    pub luma_smear: f32,
    pub composite_sharpening: f32,
    /// Last asynchronous worker error, if any.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
    /// Process-lifetime, path-specific admission diagnostics for the bounded
    /// live VHS pipelines. Additive defaults preserve older snapshots.
    #[serde(default)]
    pub live_metrics: NtscLiveMetricsSnapshot,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NtscLiveMetricsSnapshot {
    /// off | global. `selective` remains readable for older snapshots.
    #[serde(default)]
    pub active_path: String,
    #[serde(default)]
    pub global: crate::ntsc::NtscPathMetrics,
    #[serde(default)]
    pub selective: crate::ntsc::NtscPathMetrics,
    /// True while the active final-program path owns bounded CPU work.
    #[serde(default)]
    pub busy: bool,
}

/// B5 codec-mosh live diagnostics: the same counter law as the NTSC paths
/// (`attempted`/`accepted`/`skipped`/`unavailable`/`stale`, with `skipped`
/// reserved for healthy bounded backpressure) plus the stage's own error
/// channel — failure surfaces through this status, never a silent bypass.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CodecMoshLiveSnapshot {
    /// Whether the authored stage is armed (the amount deadband law).
    pub active: bool,
    /// True while the bounded worker owns an in-flight round trip.
    pub busy: bool,
    /// Last worker error, if any. A failed worker is terminal and this is
    /// its named record.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub error: String,
    pub metrics: crate::ntsc::NtscPathMetrics,
}

/// Browser-facing view of the host's bounded source-admission policy. Keeping
/// the status text alongside exact numeric limits lets the UI explain a reject
/// without implying that `wgpu` exposes portable free-VRAM information.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaSafetySnapshot {
    #[serde(default)]
    pub mode: crate::media_safety::MediaSafetyMode,
    #[serde(default = "default_safe_media_max_pixels")]
    pub safe_max_pixels: u64,
    #[serde(default = "default_safe_media_max_rgba_bytes")]
    pub safe_max_rgba_bytes: u64,
    #[serde(default = "default_expert_media_max_pixels")]
    pub expert_max_pixels: u64,
    #[serde(default = "default_expert_media_max_rgba_bytes")]
    pub expert_max_rgba_bytes: u64,
    #[serde(default = "default_absolute_media_max_edge")]
    pub absolute_max_edge: u32,
    #[serde(default)]
    pub physical_memory_bytes: Option<u64>,
    #[serde(default)]
    pub planning_budget_bytes: u64,
    #[serde(default)]
    pub reserved_bytes: u64,
    #[serde(default)]
    pub available_planning_bytes: u64,
    #[serde(default)]
    pub device_max_texture_dimension_2d: Option<u32>,
    #[serde(default)]
    pub device_max_buffer_size: Option<u64>,
    #[serde(default)]
    pub portable_vram_budget_available: bool,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub status: String,
}

const fn default_safe_media_max_pixels() -> u64 {
    crate::media_safety::SAFE_MEDIA_MAX_PIXELS
}

const fn default_safe_media_max_rgba_bytes() -> u64 {
    crate::media_safety::SAFE_MEDIA_MAX_RGBA_BYTES
}

const fn default_expert_media_max_pixels() -> u64 {
    crate::media_safety::EXPERT_MEDIA_MAX_PIXELS
}

const fn default_expert_media_max_rgba_bytes() -> u64 {
    crate::media_safety::EXPERT_MEDIA_MAX_RGBA_BYTES
}

const fn default_absolute_media_max_edge() -> u32 {
    crate::media_safety::ABSOLUTE_MEDIA_MAX_EDGE
}

impl Default for MediaSafetySnapshot {
    fn default() -> Self {
        Self {
            mode: crate::media_safety::MediaSafetyMode::Safe,
            safe_max_pixels: default_safe_media_max_pixels(),
            safe_max_rgba_bytes: default_safe_media_max_rgba_bytes(),
            expert_max_pixels: default_expert_media_max_pixels(),
            expert_max_rgba_bytes: default_expert_media_max_rgba_bytes(),
            absolute_max_edge: default_absolute_media_max_edge(),
            physical_memory_bytes: None,
            planning_budget_bytes: 0,
            reserved_bytes: 0,
            available_planning_bytes: 0,
            device_max_texture_dimension_2d: None,
            device_max_buffer_size: None,
            portable_vram_budget_available: false,
            rationale: String::new(),
            status: String::new(),
        }
    }
}

impl MediaSafetySnapshot {
    pub fn from_policy(
        policy: crate::media_safety::MediaSafetySnapshotData,
        status: String,
    ) -> Self {
        let rationale = policy.rationale();
        Self {
            mode: policy.mode,
            safe_max_pixels: policy.safe_max_pixels,
            safe_max_rgba_bytes: policy.safe_max_rgba_bytes,
            expert_max_pixels: policy.expert_max_pixels,
            expert_max_rgba_bytes: policy.expert_max_rgba_bytes,
            absolute_max_edge: policy.absolute_max_edge,
            physical_memory_bytes: policy.physical_memory_bytes,
            planning_budget_bytes: policy.planning_budget_bytes,
            reserved_bytes: policy.reserved_bytes,
            available_planning_bytes: policy.available_planning_bytes,
            device_max_texture_dimension_2d: policy.device_max_texture_dimension_2d,
            device_max_buffer_size: policy.device_max_buffer_size,
            portable_vram_budget_available: policy.portable_vram_budget_available,
            rationale,
            status,
        }
    }
}

/// Browser-facing view of the host-session authored proxy settings. Only the
/// three operator choices cross the wire; the schema and algorithm versions
/// are always the engine's own constants and are deliberately not published,
/// so a client can never echo a foreign version back into a cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProxySettingsSnapshot {
    #[serde(default = "default_proxy_scale")]
    pub scale: crate::proxy::ProxyScale,
    #[serde(default = "default_proxy_frame_rate")]
    pub frame_rate: crate::proxy::ProxyFrameRate,
    #[serde(default = "default_proxy_include_audio")]
    pub include_audio: bool,
}

const fn default_proxy_scale() -> crate::proxy::ProxyScale {
    crate::proxy::ProxyScale::Half
}

const fn default_proxy_frame_rate() -> crate::proxy::ProxyFrameRate {
    crate::proxy::ProxyFrameRate::Source
}

const fn default_proxy_include_audio() -> bool {
    true
}

impl Default for ProxySettingsSnapshot {
    fn default() -> Self {
        Self::from_settings(crate::proxy::ProxySettings::default())
    }
}

impl ProxySettingsSnapshot {
    pub fn from_settings(settings: crate::proxy::ProxySettings) -> Self {
        Self {
            scale: settings.scale,
            frame_rate: settings.frame_rate,
            include_audio: settings.include_audio,
        }
    }
}

impl Default for NtscSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            tape_speed: 0,
            chroma_loss: 0.0,
            edge_wave_enabled: false,
            edge_wave_intensity: 0.0,
            edge_wave_speed: 0.5,
            head_switching_enabled: false,
            head_switching_height: 8,
            head_switching_shift: 0.0,
            tracking_noise_enabled: false,
            tracking_noise_height: 24,
            tracking_noise_wave: 0.0,
            tracking_noise_snow: 0.0,
            snow_intensity: 0.0,
            composite_noise_intensity: 0.0,
            luma_noise_intensity: 0.0,
            chroma_noise_intensity: 0.0,
            luma_smear: 0.0,
            composite_sharpening: 0.0,
            error: String::new(),
            live_metrics: NtscLiveMetricsSnapshot::default(),
        }
    }
}

impl NtscSnapshot {
    pub fn from_params(p: &crate::ntsc::NtscParams) -> Self {
        Self {
            enabled: p.enabled,
            tape_speed: p.tape_speed,
            chroma_loss: p.chroma_loss,
            edge_wave_enabled: p.edge_wave_enabled,
            edge_wave_intensity: p.edge_wave_intensity,
            edge_wave_speed: p.edge_wave_speed,
            head_switching_enabled: p.head_switching_enabled,
            head_switching_height: p.head_switching_height,
            head_switching_shift: p.head_switching_shift,
            tracking_noise_enabled: p.tracking_noise_enabled,
            tracking_noise_height: p.tracking_noise_height,
            tracking_noise_wave: p.tracking_noise_wave,
            tracking_noise_snow: p.tracking_noise_snow,
            snow_intensity: p.snow_intensity,
            composite_noise_intensity: p.composite_noise_intensity,
            luma_noise_intensity: p.luma_noise_intensity,
            chroma_noise_intensity: p.chroma_noise_intensity,
            luma_smear: p.luma_smear,
            composite_sharpening: p.composite_sharpening,
            error: String::new(),
            live_metrics: NtscLiveMetricsSnapshot::default(),
        }
    }
}

/// Modulation matrix state sent to the browser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModSnapshot {
    pub bpm: f32,
    /// Beat position (quarter notes); the panel's beat light pulses on it.
    #[serde(default)]
    pub beat: f64,
    pub lfos: Vec<LfoSnapshot>,
    pub routings: Vec<RoutingSnapshot>,
    /// Phone orientation [yaw, pitch, roll], 0..1 (0.5 = level).
    #[serde(default)]
    pub gyro: [f32; 3],
    /// XY performance pad position, each 0..1.
    #[serde(default)]
    pub pad: [f32; 2],
    /// Engine-owned response settings for each orientation axis.
    #[serde(default)]
    pub gyro_config: GyroConfigSnapshot,
    /// Authoritative server view of phone stream ownership and freshness.
    #[serde(default)]
    pub gyro_status: GyroStatusSnapshot,
    /// Engine-owned response, quantize, and spring settings for the XY pad.
    #[serde(default)]
    pub pad_config: PadConfigSnapshot,
    /// B10 envelope configurations and live levels. Additive: an older panel
    /// deserializes an otherwise complete snapshot with the block absent.
    #[serde(default)]
    pub envelopes: Vec<EnvelopeSnapshot>,
    /// B10 macro knob values, each 0..1.
    #[serde(default)]
    pub macros: Vec<f32>,
    /// B10 bend pad held states (runtime truth, never persisted).
    #[serde(default)]
    pub bends: Vec<bool>,
    /// B10 deterministic-generator seed.
    #[serde(default)]
    pub generator_seed: u32,
}

/// One B10 envelope's authored configuration plus its live level meter.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EnvelopeSnapshot {
    pub attack: f32,
    pub decay: f32,
    pub trigger: String,
    pub mode: String,
    /// Live output level 0..1, for the panel meter.
    #[serde(default)]
    pub level: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GyroStatusSnapshot {
    /// At least one declared/legacy streamer has supplied a recent sample.
    #[serde(default)]
    pub active: bool,
    /// A stream existed or was requested, but no sample is currently fresh.
    #[serde(default)]
    pub stale: bool,
    /// Number of connected clients currently claiming an enabled stream.
    #[serde(default)]
    pub streamers: usize,
    /// Monotonic age of the most recently accepted sample.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_age_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LfoSnapshot {
    pub shape: String,
    pub beats: f32,
    pub phase: f32,
    #[serde(default)]
    pub seed: u32,
    /// Live output value in [-1, 1], for UI meters.
    pub value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingSnapshot {
    /// Stable runtime identity. It is deliberately not persisted in patches.
    #[serde(default)]
    pub route_id: String,
    pub source: String,
    pub target: String,
    pub depth: f32,
    #[serde(default = "default_curve")]
    pub curve: String,
    #[serde(default)]
    pub curve_amount: f32,
    /// Rise time in seconds (zero is immediate).
    #[serde(default)]
    pub attack: f32,
    /// Fall time in seconds (zero is immediate).
    #[serde(default)]
    pub release: f32,
    /// Cached shaped/slewed source value before route depth, in -1..1.
    #[serde(default)]
    pub value: f32,
}

fn default_curve() -> String {
    "linear".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfigSnapshot {
    /// Degrees from calibrated center to full-scale output.
    pub range: f32,
    /// Signed response exponent amount in -2..2.
    pub expo: f32,
    pub invert: bool,
}

impl Default for AxisConfigSnapshot {
    fn default() -> Self {
        Self {
            range: 90.0,
            expo: 0.0,
            invert: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GyroConfigSnapshot {
    pub yaw: AxisConfigSnapshot,
    pub pitch: AxisConfigSnapshot,
    pub roll: AxisConfigSnapshot,
}

impl Default for GyroConfigSnapshot {
    fn default() -> Self {
        Self {
            yaw: AxisConfigSnapshot {
                range: 180.0,
                ..Default::default()
            },
            pitch: AxisConfigSnapshot::default(),
            roll: AxisConfigSnapshot::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadAxisConfigSnapshot {
    #[serde(default = "default_curve")]
    pub curve: String,
    #[serde(default)]
    pub curve_amount: f32,
    /// Number of discrete positions; 0/1 disables quantization.
    #[serde(default)]
    pub quantize: u32,
}

impl Default for PadAxisConfigSnapshot {
    fn default() -> Self {
        Self {
            curve: default_curve(),
            curve_amount: 0.0,
            quantize: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadConfigSnapshot {
    #[serde(default)]
    pub x: PadAxisConfigSnapshot,
    #[serde(default)]
    pub y: PadAxisConfigSnapshot,
    #[serde(default)]
    pub spring_enabled: bool,
    #[serde(default = "default_spring_rate")]
    pub spring_rate: f32,
}

fn default_spring_rate() -> f32 {
    4.0
}

impl Default for PadConfigSnapshot {
    fn default() -> Self {
        Self {
            x: PadAxisConfigSnapshot::default(),
            y: PadAxisConfigSnapshot::default(),
            spring_enabled: false,
            spring_rate: default_spring_rate(),
        }
    }
}

/// Audio analysis state sent to the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSnapshot {
    pub enabled: bool,
    /// `live` for device/Windows playback capture, `file` for deterministic
    /// circular analysis of an imported clip.
    #[serde(default = "default_audio_source_kind")]
    pub source_kind: String,
    pub gain: f32,
    pub level: f32,
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    pub onset: f32,
    #[serde(default)]
    pub bright: f32,
    #[serde(default)]
    pub noise: f32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub device: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
    /// Available input device names, for the device select.
    #[serde(default)]
    pub devices: Vec<String>,
    /// Windows output endpoints available through WASAPI loopback capture.
    #[serde(default)]
    pub system_playback_devices: Vec<String>,
    /// Preferred device name ("" = system default).
    #[serde(default)]
    pub selected: String,
    /// Device that is actually producing samples after fallback resolution.
    #[serde(default)]
    pub active_device: String,
    /// True when `active_device` differs from the requested `selected` device.
    #[serde(default)]
    pub using_fallback: bool,
    /// Audio-only files in the active library.
    #[serde(default)]
    pub clip_files: Vec<String>,
    /// Persisted/selected clip source identity.
    #[serde(default)]
    pub clip_path: String,
    #[serde(default)]
    pub clip_loading: bool,
    #[serde(default)]
    pub clip_duration_secs: f64,
    #[serde(default = "default_audio_band_count")]
    pub band_count: usize,
    /// The active count - 1 ordered crossovers.
    #[serde(default)]
    pub band_edges: Vec<f32>,
    #[serde(default = "default_audio_band_ceiling")]
    pub band_ceiling_hz: f32,
    /// Normalized configurable band meters, one per active band.
    #[serde(default)]
    pub bands: Vec<f32>,
    #[serde(default)]
    pub spectrum: Vec<f32>,
}

fn default_audio_band_count() -> usize {
    3
}

fn default_audio_source_kind() -> String {
    crate::modulation::AUDIO_SOURCE_LIVE.to_string()
}

fn default_audio_band_ceiling() -> f32 {
    8000.0
}

impl Default for AudioSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            source_kind: default_audio_source_kind(),
            gain: 0.0,
            level: 0.0,
            bass: 0.0,
            mid: 0.0,
            high: 0.0,
            onset: 0.0,
            bright: 0.0,
            noise: 0.0,
            device: String::new(),
            error: String::new(),
            devices: Vec::new(),
            system_playback_devices: Vec::new(),
            selected: String::new(),
            active_device: String::new(),
            using_fallback: false,
            clip_files: Vec::new(),
            clip_path: String::new(),
            clip_loading: false,
            clip_duration_secs: 0.0,
            band_count: default_audio_band_count(),
            band_edges: vec![250.0, 2000.0],
            band_ceiling_hz: default_audio_band_ceiling(),
            bands: vec![0.0; default_audio_band_count()],
            spectrum: Vec::new(),
        }
    }
}

impl ModSnapshot {
    pub fn from_matrix(m: &crate::modulation::ModMatrix) -> Self {
        Self {
            bpm: m.clock.bpm,
            beat: m.current_beat,
            lfos: m
                .lfos
                .iter()
                .zip(m.lfo_values.iter())
                .map(|(l, &v)| LfoSnapshot {
                    shape: l.shape.as_str().to_string(),
                    beats: l.beats,
                    phase: l.phase,
                    seed: l.seed,
                    value: v,
                })
                .collect(),
            routings: m
                .routings
                .iter()
                .map(|r| RoutingSnapshot {
                    route_id: r.route_id().to_string(),
                    source: r.source.as_str().to_string(),
                    target: r.target().to_owned(),
                    depth: r.depth,
                    curve: r.curve.as_str().to_string(),
                    curve_amount: r.curve_amount,
                    attack: r.attack,
                    release: r.release,
                    value: r.cached_value(),
                })
                .collect(),
            gyro: m.gyro,
            pad: m.pad,
            gyro_config: GyroConfigSnapshot {
                yaw: AxisConfigSnapshot {
                    range: m.gyro_config[0].range_degrees,
                    expo: m.gyro_config[0].expo,
                    invert: m.gyro_config[0].invert,
                },
                pitch: AxisConfigSnapshot {
                    range: m.gyro_config[1].range_degrees,
                    expo: m.gyro_config[1].expo,
                    invert: m.gyro_config[1].invert,
                },
                roll: AxisConfigSnapshot {
                    range: m.gyro_config[2].range_degrees,
                    expo: m.gyro_config[2].expo,
                    invert: m.gyro_config[2].invert,
                },
            },
            gyro_status: GyroStatusSnapshot::default(),
            pad_config: PadConfigSnapshot {
                x: PadAxisConfigSnapshot {
                    curve: m.pad_config.axes[0].curve.as_str().to_string(),
                    curve_amount: m.pad_config.axes[0].curve_amount,
                    quantize: m.pad_config.axes[0].quantize,
                },
                y: PadAxisConfigSnapshot {
                    curve: m.pad_config.axes[1].curve.as_str().to_string(),
                    curve_amount: m.pad_config.axes[1].curve_amount,
                    quantize: m.pad_config.axes[1].quantize,
                },
                spring_enabled: m.pad_config.spring_enabled,
                spring_rate: m.pad_config.spring_rate,
            },
            envelopes: m
                .envelopes
                .iter()
                .map(|envelope| EnvelopeSnapshot {
                    attack: envelope.attack,
                    decay: envelope.decay,
                    trigger: envelope.trigger.as_str().to_string(),
                    mode: envelope.mode.as_str().to_string(),
                    level: envelope.level(),
                })
                .collect(),
            macros: m.macros.to_vec(),
            bends: m.bend_held.to_vec(),
            generator_seed: m.generator_seed,
        }
    }
}

/// Temporal (frame-history) effect parameters sent to the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSnapshot {
    pub feedback: f32,
    pub fb_zoom: f32,
    pub fb_rotate: f32,
    pub slitscan: f32,
    #[serde(default)]
    pub slit_angle: f32,
    pub slit_axis: f32,
    /// Additive B12 time-displace map token; absent means the exact `ramp`.
    #[serde(default = "default_time_displace_map")]
    pub slit_map: String,
    /// Additive B12 interpolation toggle; absent means the banded floor law.
    #[serde(default)]
    pub slit_interp: bool,
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default = "default_temporal_key_threshold")]
    pub key_threshold: f32,
    #[serde(default = "default_temporal_key_softness")]
    pub key_softness: f32,
    #[serde(default = "default_temporal_key_history")]
    pub key_history: f32,
    /// Additive M3 authoring state. Every default is an exact no-op.
    #[serde(default)]
    pub originals: TemporalOriginalsSnapshot,
    /// Additive B3 feedback rig. The default is the exact historical path.
    #[serde(default)]
    pub rig: TemporalRigSnapshot,
    /// Additive B4 display physics. The params struct is its own sanitizing
    /// serde block with snake_case discrete tokens; absent means exact-off.
    #[serde(default)]
    pub display: crate::display_physics::DisplayPhysicsParams,
    /// Additive B8 melting edge. The params struct is its own sanitizing
    /// serde block; absent means exact-off.
    #[serde(default)]
    pub melt: crate::mixing_boundary::MeltParams,
    /// Additive B5 codec mosh. The params struct is its own sanitizing
    /// serde block; absent means exact bypass.
    #[serde(default)]
    pub mosh: crate::codec_mosh::CodecMoshParams,
    /// Additive B14 sync latch. The params struct is its own sanitizing serde
    /// block; absent means exact-off.
    #[serde(default)]
    pub sync: crate::sync_latch::SyncLatchParams,
    /// Read-only renderer truth: whether the sync latch is currently holding
    /// accumulated shear. The authored switch says what the operator asked
    /// for; this says whether the program is actually still broken, which is
    /// the fact a failure switch exists to make visible.
    #[serde(default)]
    pub sync_damaged: bool,
    /// Read-only renderer truth. Main fills this DTO when the active executor
    /// exposes metrics; older/exact paths safely report the zero placeholder.
    #[serde(default)]
    pub telemetry: TemporalTelemetrySnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalLoomSnapshot {
    pub amount: f32,
    pub topology: String,
    pub interpolation: String,
    pub depth: f32,
    pub phase: f32,
    pub scale: f32,
    pub angle: f32,
    pub folds: u8,
    pub quantization: u8,
}

impl Default for TemporalLoomSnapshot {
    fn default() -> Self {
        Self {
            amount: 0.0,
            topology: "linear".into(),
            interpolation: "floor".into(),
            depth: 1.0,
            phase: 0.0,
            scale: 1.0,
            angle: 0.0,
            folds: 1,
            quantization: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CollisionAtlasSnapshot {
    pub amount: f32,
    pub seed: u32,
    pub territories: u8,
    pub collision: f32,
}

impl Default for CollisionAtlasSnapshot {
    fn default() -> Self {
        Self {
            amount: 0.0,
            seed: 0,
            territories: 8,
            collision: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RefreshGardenSnapshot {
    pub amount: f32,
    pub gate: String,
    pub threshold: f32,
    pub softness: f32,
    pub decay: f32,
    pub max_hold_ticks: u32,
    #[serde(default)]
    pub matte_route: RefreshGardenMatteRouteSnapshot,
    #[serde(default)]
    pub motion_route: RefreshGardenMotionRouteSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LongExposureSnapshot {
    pub amount: f32,
    pub shutter_frames: u8,
}

impl Default for LongExposureSnapshot {
    fn default() -> Self {
        Self {
            amount: 0.0,
            shutter_frames: 12,
        }
    }
}

impl Default for RefreshGardenSnapshot {
    fn default() -> Self {
        Self {
            amount: 0.0,
            gate: "temporal_delta".into(),
            threshold: 0.1,
            softness: 0.03,
            decay: 1.0,
            max_hold_ticks: 0,
            matte_route: RefreshGardenMatteRouteSnapshot::None,
            motion_route: RefreshGardenMotionRouteSnapshot::None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RefreshGardenMatteRouteSnapshot {
    #[default]
    None,
    SelectedLayer {
        layer_id: String,
        saved_position: u32,
        stage: LayerImageStage,
    },
    MissingSelectedLayer {
        saved_position: u32,
        stage: LayerImageStage,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RefreshGardenMotionRouteSnapshot {
    #[default]
    None,
    SelectedLayer {
        layer_id: String,
        saved_position: u32,
    },
    MissingSelectedLayer {
        saved_position: u32,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CollisionScoreLoopDriverSnapshot {
    #[default]
    None,
    SelectedLayer {
        layer_id: String,
        saved_position: u32,
    },
    MissingSelectedLayer {
        saved_position: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CollisionScoreSnapshot {
    pub enabled: bool,
    pub seed: u32,
    pub state_count: u8,
    pub trigger: String,
    pub loop_driver: CollisionScoreLoopDriverSnapshot,
}

impl Default for CollisionScoreSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            seed: 0,
            state_count: 4,
            trigger: "boundary".into(),
            loop_driver: CollisionScoreLoopDriverSnapshot::None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalResetPolicySnapshot {
    pub loop_boundary: String,
    pub downbeat: String,
}

impl Default for TemporalResetPolicySnapshot {
    fn default() -> Self {
        Self {
            loop_boundary: "none".into(),
            downbeat: "none".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TemporalOriginalsSnapshot {
    pub loom: TemporalLoomSnapshot,
    pub atlas: CollisionAtlasSnapshot,
    pub garden: RefreshGardenSnapshot,
    pub long_exposure: LongExposureSnapshot,
    pub score: CollisionScoreSnapshot,
    pub reset: TemporalResetPolicySnapshot,
}

/// Operator-facing gesture recording truth.
///
/// The honesty law lives in these fields. `recording` is the armed flag;
/// `recorded_events` counts only what actually entered the replayable track;
/// `live_only_events` counts normalized samples that affected this session and
/// were never recorded. A snapshot must never let the second be mistaken for
/// the first, so they are separate counters and neither is derived from the
/// other.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GestureStatusSnapshot {
    /// True while recording is armed. While false, a live gesture is
    /// session-local and is not replayable.
    pub recording: bool,
    /// Events in the replayable track.
    pub recorded_events: u32,
    /// The track reached its bounded cap and newer events stayed live-only.
    pub truncated: bool,
    /// Strokes still open in the recorded track. A track with open strokes is
    /// valid but explicitly incomplete and is never auto-closed.
    pub open_strokes: u32,
    /// Normalized samples that reached this session without being recorded.
    pub live_only_events: u64,
    /// Canonical checksum of the recorded track, empty while it is empty.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub checksum: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
    /// Authored canvas controls. These are values, not a recording, and are
    /// published beside the recording truth rather than inside it.
    #[serde(default)]
    pub canvas: GestureCanvasStatusSnapshot,
}

/// B9 performance-recorder truth. Additive: an older panel deserializes an
/// otherwise complete snapshot as a disarmed recorder with no take.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PerformanceStatusSnapshot {
    /// `off`, `recording`, or `playing`. The two transports are mutually
    /// exclusive by refusal, so one mode word tells the whole truth.
    #[serde(default)]
    pub mode: String,
    /// Events and distinct control addresses in the loaded take.
    pub recorded_events: u32,
    pub recorded_controls: u32,
    /// Declared take length in 30 Hz reference ticks, stamped at disarm so
    /// trailing silence stays part of the loop period.
    pub length_ticks: u32,
    /// Replay playhead address while playback is armed, zero otherwise.
    pub playhead_tick: u32,
    pub loop_playback: bool,
    /// The take reached its bounded event cap and newer edits stayed
    /// live-only.
    pub truncated: bool,
    /// Recordable-family edits whose param has no declared value law, and
    /// edits whose value the law could not represent. Separate counters,
    /// never merged into the recorded count.
    pub unsupported_edits: u64,
    pub rejected_edits: u64,
    /// Canonical checksum of the loaded take, empty while it is empty.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub checksum: String,
    /// Named per-address diagnostics for controls the current program could
    /// not resolve at playback arm — degraded, never retargeted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub degraded: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
}

/// Authored gesture-canvas controls and the session-local field they drive.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GestureCanvasStatusSnapshot {
    pub radius: f32,
    pub strength: f32,
    pub retention: f32,
    pub grid_width: u32,
    pub grid_height: u32,
    /// Committed canvas frames since the last reset. This is session state and
    /// says nothing about whether any of it was recorded.
    pub generation: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalTelemetrySnapshot {
    pub history_valid: u32,
    pub history_capacity: u32,
    pub carrier_valid: bool,
    pub freeze_hold_valid: bool,
    pub total_reference_ticks: u64,
    pub score_state: u8,
    pub score_event_ordinal: u64,
    #[serde(default)]
    pub recorded_event_points: u32,
    #[serde(default)]
    pub event_track_truncated: bool,
    pub frame_staged: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub last_reset: String,
}

impl Default for TemporalTelemetrySnapshot {
    fn default() -> Self {
        Self {
            history_valid: 0,
            history_capacity: 24,
            carrier_valid: false,
            freeze_hold_valid: false,
            total_reference_ticks: 0,
            score_state: 0,
            score_event_ordinal: 0,
            recorded_event_points: 0,
            event_track_truncated: false,
            frame_staged: false,
            last_reset: String::new(),
        }
    }
}

fn default_temporal_key_threshold() -> f32 {
    0.1
}

fn default_time_displace_map() -> String {
    "ramp".into()
}

fn default_temporal_key_softness() -> f32 {
    0.03
}

fn default_temporal_key_history() -> f32 {
    1.0
}

impl Default for TemporalSnapshot {
    fn default() -> Self {
        Self {
            feedback: 0.0,
            fb_zoom: 1.0,
            fb_rotate: 0.0,
            slitscan: 0.0,
            slit_angle: 0.0,
            slit_axis: 0.0,
            slit_map: default_time_displace_map(),
            slit_interp: false,
            key_mode: 0,
            key_threshold: default_temporal_key_threshold(),
            key_softness: default_temporal_key_softness(),
            key_history: default_temporal_key_history(),
            originals: TemporalOriginalsSnapshot::default(),
            rig: TemporalRigSnapshot::default(),
            display: crate::display_physics::DisplayPhysicsParams::default(),
            melt: crate::mixing_boundary::MeltParams::default(),
            mosh: crate::codec_mosh::CodecMoshParams::default(),
            sync: crate::sync_latch::SyncLatchParams::default(),
            sync_damaged: false,
            telemetry: TemporalTelemetrySnapshot::default(),
        }
    }
}

/// The B3 feedback rig as the browser sees it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalRigSnapshot {
    pub offset_x: f32,
    pub offset_y: f32,
    pub reflect_x: bool,
    pub reflect_y: bool,
    pub hue_rotate: f32,
    pub saturation: f32,
    pub gain_r: f32,
    pub gain_g: f32,
    pub gain_b: f32,
    pub chroma_displace: f32,
    pub blur: f32,
    pub sharpen: f32,
    pub shape: String,
    pub drive: f32,
    pub pivot: f32,
    pub threshold: f32,
    pub noise: f32,
    pub edge: String,
    pub servo: bool,
    pub servo_defeated: bool,
}

impl Default for TemporalRigSnapshot {
    fn default() -> Self {
        Self::from_params(crate::effects::params::FeedbackRigParams::default())
    }
}

impl TemporalRigSnapshot {
    fn from_params(rig: crate::effects::params::FeedbackRigParams) -> Self {
        use crate::effects::params::FeedbackShape;
        use crate::motion::MotionBoundaryMode;
        let rig = rig.sanitized();
        Self {
            offset_x: rig.offset_x,
            offset_y: rig.offset_y,
            reflect_x: rig.reflect_x,
            reflect_y: rig.reflect_y,
            hue_rotate: rig.hue_rotate,
            saturation: rig.saturation,
            gain_r: rig.gain_r,
            gain_g: rig.gain_g,
            gain_b: rig.gain_b,
            chroma_displace: rig.chroma_displace,
            blur: rig.blur,
            sharpen: rig.sharpen,
            shape: match rig.shape {
                FeedbackShape::Clamp => "clamp",
                FeedbackShape::Soft => "soft",
                FeedbackShape::Wrap => "wrap",
                FeedbackShape::Fold => "fold",
            }
            .into(),
            drive: rig.drive,
            pivot: rig.pivot,
            threshold: rig.threshold,
            noise: rig.noise,
            edge: match rig.edge {
                MotionBoundaryMode::Transparent => "transparent",
                MotionBoundaryMode::Mirror => "mirror",
                MotionBoundaryMode::Wrap => "wrap",
                MotionBoundaryMode::Hold => "hold",
            }
            .into(),
            servo: rig.servo,
            servo_defeated: rig.servo_defeated,
        }
    }
}

impl TemporalSnapshot {
    pub fn from_params(p: &crate::effects::params::TemporalParams) -> Self {
        use crate::effects::params::{
            CollisionScoreTrigger, RefreshGardenGate, TemporalInterpolation, TemporalTopology,
            TimeDisplaceMap,
        };
        use crate::temporal::{
            CollisionScoreLoopDriver, RefreshGardenMatteRoute, RefreshGardenMotionRoute,
            TemporalEventResetMode,
        };

        let topology = match p.originals.loom.topology {
            TemporalTopology::Linear => "linear",
            TemporalTopology::Radial => "radial",
            TemporalTopology::Spiral => "spiral",
            TemporalTopology::Contour => "contour",
            TemporalTopology::Folded => "folded",
            TemporalTopology::Kaleidoscopic => "kaleidoscopic",
        };
        let interpolation = match p.originals.loom.interpolation {
            TemporalInterpolation::Floor => "floor",
            TemporalInterpolation::Linear => "linear",
        };
        let slit_map = match p.slit_map {
            TimeDisplaceMap::Ramp => "ramp",
            TimeDisplaceMap::Brightness => "brightness",
            TimeDisplaceMap::Radial => "radial",
            TimeDisplaceMap::TbcRamp => "tbc_ramp",
            TimeDisplaceMap::Sweep => "sweep",
        };
        let gate = match p.originals.garden.gate {
            RefreshGardenGate::TemporalDelta => "temporal_delta",
            RefreshGardenGate::Luma => "luma",
            RefreshGardenGate::Chroma => "chroma",
            RefreshGardenGate::CellularRidge => "cellular_ridge",
            RefreshGardenGate::AudioEnergy => "audio_energy",
            RefreshGardenGate::AudioOnset => "audio_onset",
            RefreshGardenGate::Matte => "matte",
            RefreshGardenGate::Motion => "motion",
        };
        let trigger = match p.originals.score.trigger {
            CollisionScoreTrigger::Boundary => "boundary",
            CollisionScoreTrigger::Downbeat => "downbeat",
            CollisionScoreTrigger::AudioOnset => "audio_onset",
            CollisionScoreTrigger::Manual => "manual",
        };
        let reset_key = |mode| match mode {
            TemporalEventResetMode::None => "none",
            TemporalEventResetMode::Score => "score",
            TemporalEventResetMode::Memory => "memory",
            TemporalEventResetMode::All => "all",
        };
        let loop_driver = match p.originals.score.loop_driver {
            CollisionScoreLoopDriver::None => CollisionScoreLoopDriverSnapshot::None,
            CollisionScoreLoopDriver::SelectedLayer {
                layer_id,
                saved_position,
            } => CollisionScoreLoopDriverSnapshot::SelectedLayer {
                layer_id: layer_id.get().to_string(),
                saved_position: saved_position.get(),
            },
            CollisionScoreLoopDriver::MissingSelectedLayer { saved_position } => {
                CollisionScoreLoopDriverSnapshot::MissingSelectedLayer {
                    saved_position: saved_position.get(),
                }
            }
        };
        let matte_route = match p.originals.garden.matte_route {
            RefreshGardenMatteRoute::None => RefreshGardenMatteRouteSnapshot::None,
            RefreshGardenMatteRoute::SelectedLayer {
                layer_id,
                saved_position,
                stage,
            } => RefreshGardenMatteRouteSnapshot::SelectedLayer {
                layer_id: layer_id.get().to_string(),
                saved_position: saved_position.get(),
                stage,
            },
            RefreshGardenMatteRoute::MissingSelectedLayer {
                saved_position,
                stage,
            } => RefreshGardenMatteRouteSnapshot::MissingSelectedLayer {
                saved_position: saved_position.get(),
                stage,
            },
        };
        let motion_route = match p.originals.garden.motion_route {
            RefreshGardenMotionRoute::None => RefreshGardenMotionRouteSnapshot::None,
            RefreshGardenMotionRoute::SelectedLayer {
                layer_id,
                saved_position,
            } => RefreshGardenMotionRouteSnapshot::SelectedLayer {
                layer_id: layer_id.get().to_string(),
                saved_position: saved_position.get(),
            },
            RefreshGardenMotionRoute::MissingSelectedLayer { saved_position } => {
                RefreshGardenMotionRouteSnapshot::MissingSelectedLayer {
                    saved_position: saved_position.get(),
                }
            }
        };
        Self {
            feedback: p.feedback,
            fb_zoom: p.fb_zoom,
            fb_rotate: p.fb_rotate,
            slitscan: p.slitscan,
            slit_angle: p.slit_angle,
            slit_axis: p.slit_axis,
            slit_map: slit_map.into(),
            slit_interp: p.slit_interp,
            key_mode: p.key_mode.round().clamp(0.0, 4.0) as u32,
            key_threshold: p.key_threshold,
            key_softness: p.key_softness,
            key_history: p.key_history,
            rig: TemporalRigSnapshot::from_params(p.rig),
            display: p.display.sanitized(),
            melt: p.melt.sanitized(),
            mosh: p.mosh.sanitized(),
            sync: p.sync.sanitized(),
            // Renderer truth, filled by main beside the temporal telemetry.
            sync_damaged: false,
            originals: TemporalOriginalsSnapshot {
                loom: TemporalLoomSnapshot {
                    amount: p.originals.loom.amount,
                    topology: topology.into(),
                    interpolation: interpolation.into(),
                    depth: p.originals.loom.depth,
                    phase: p.originals.loom.phase,
                    scale: p.originals.loom.scale,
                    angle: p.originals.loom.angle,
                    folds: p.originals.loom.folds,
                    quantization: p.originals.loom.quantization,
                },
                atlas: CollisionAtlasSnapshot {
                    amount: p.originals.atlas.amount,
                    seed: p.originals.atlas.seed,
                    territories: p.originals.atlas.territories,
                    collision: p.originals.atlas.collision,
                },
                garden: RefreshGardenSnapshot {
                    amount: p.originals.garden.amount,
                    gate: gate.into(),
                    threshold: p.originals.garden.threshold,
                    softness: p.originals.garden.softness,
                    decay: p.originals.garden.decay,
                    max_hold_ticks: p.originals.garden.max_hold_ticks,
                    matte_route,
                    motion_route,
                },
                long_exposure: LongExposureSnapshot {
                    amount: p.originals.long_exposure.amount,
                    shutter_frames: p.originals.long_exposure.shutter_frames,
                },
                score: CollisionScoreSnapshot {
                    enabled: p.originals.score.enabled,
                    seed: p.originals.score.seed,
                    state_count: p.originals.score.state_count,
                    trigger: trigger.into(),
                    loop_driver,
                },
                reset: TemporalResetPolicySnapshot {
                    loop_boundary: reset_key(p.originals.reset.loop_boundary).into(),
                    downbeat: reset_key(p.originals.reset.downbeat).into(),
                },
            },
            telemetry: TemporalTelemetrySnapshot::default(),
        }
    }
}

/// Stable browser address for Motion authoring. Numeric runtime IDs are sent
/// as decimal strings so JavaScript never rounds them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum MotionScopeSnapshot {
    Master,
    Layer { layer_id: String },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MotionDonorSnapshot {
    #[default]
    None,
    Selected {
        layer_id: String,
        saved_position: u32,
    },
    /// Output-only tombstone. Ingress can select a current stable layer or
    /// clear the donor; it can never manufacture or retarget this variant.
    Missing { saved_position: u32 },
}

/// The addressed Field Collider input, as a closed named token.
///
/// A token rather than an index, following the Residual slot precedent: an
/// unknown slot is a deserialization rejection instead of a positional fallback
/// onto the partner input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldColliderInputSnapshot {
    #[default]
    A,
    B,
}

impl FieldColliderInputSnapshot {
    pub const fn to_runtime(self) -> crate::motion::FieldColliderInput {
        match self {
            Self::A => crate::motion::FieldColliderInput::A,
            Self::B => crate::motion::FieldColliderInput::B,
        }
    }
}

/// Browser view of the Field Collider block.
///
/// Both inputs are published through the established [`MotionDonorSnapshot`]
/// vocabulary, preserving their Selected/Missing tombstone semantics — there is
/// deliberately no parallel donor encoding for the collider. `diagnostic` is
/// the engine's typed admission answer rendered for the operator; `admitted`
/// says whether the derived field actually owns the carrier this frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldColliderSnapshot {
    pub algorithm_version: u16,
    pub enabled: bool,
    pub mode: String,
    pub boundary: String,
    pub input_a: MotionDonorSnapshot,
    pub input_b: MotionDonorSnapshot,
    #[serde(default)]
    pub admitted: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub diagnostic: String,
}

impl Default for FieldColliderSnapshot {
    fn default() -> Self {
        Self::from_params(crate::motion::FieldColliderParams::default())
    }
}

impl FieldColliderSnapshot {
    pub fn from_params(params: crate::motion::FieldColliderParams) -> Self {
        use crate::motion::{FieldColliderMode, MotionBoundaryMode};
        let params = params.sanitized();
        let mode = match params.mode {
            FieldColliderMode::Sum => "sum",
            FieldColliderMode::Difference => "difference",
            FieldColliderMode::Curl => "curl",
            FieldColliderMode::Projection => "projection",
            FieldColliderMode::CollisionBoundary => "collision_boundary",
        };
        let boundary = match params.boundary {
            MotionBoundaryMode::Transparent => "transparent",
            MotionBoundaryMode::Mirror => "mirror",
            MotionBoundaryMode::Wrap => "wrap",
            MotionBoundaryMode::Hold => "hold",
        };
        Self {
            algorithm_version: params.algorithm_version,
            enabled: params.enabled,
            mode: mode.into(),
            boundary: boundary.into(),
            input_a: motion_donor_snapshot(params.input_a),
            input_b: motion_donor_snapshot(params.input_b),
            admitted: false,
            diagnostic: String::new(),
        }
    }
}

/// The one runtime-to-wire donor mapping. Every collider slot and the Faraday
/// transplant share it, so a tombstone is published identically everywhere.
fn motion_donor_snapshot(donor: crate::motion::MotionDonor) -> MotionDonorSnapshot {
    use crate::motion::MotionDonor;
    match donor {
        MotionDonor::None => MotionDonorSnapshot::None,
        MotionDonor::Selected {
            layer_id,
            saved_position,
        } => MotionDonorSnapshot::Selected {
            layer_id: layer_id.get().to_string(),
            saved_position: saved_position.get(),
        },
        MotionDonor::Missing { saved_position } => MotionDonorSnapshot::Missing {
            saved_position: saved_position.get(),
        },
    }
}

/// The B2 procedural field scalars as the browser sees them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProceduralFieldSnapshot {
    pub scale: f32,
    pub rate: f32,
}

/// The B2 flow-shaping controls as the browser sees them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowShapingSnapshot {
    pub stretch: f32,
    pub edge_repel: f32,
    pub vector_trash: f32,
    pub trash_block_size: f32,
}

impl Default for FlowShapingSnapshot {
    fn default() -> Self {
        let params = crate::motion::FlowShapingParams::default();
        Self {
            stretch: params.stretch,
            edge_repel: params.edge_repel,
            vector_trash: params.vector_trash,
            trash_block_size: params.trash_block_size,
        }
    }
}

impl Default for ProceduralFieldSnapshot {
    fn default() -> Self {
        let params = crate::motion::ProceduralFieldParams::default();
        Self {
            scale: params.scale,
            rate: params.rate,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaradayMotionSnapshot {
    pub amount: f32,
    pub donor: MotionDonorSnapshot,
    pub carrier: String,
    pub confidence_threshold: f32,
    pub confidence_softness: f32,
    pub refresh: f32,
    pub decay: f32,
    pub occlusion: f32,
}

impl Default for FaradayMotionSnapshot {
    fn default() -> Self {
        Self {
            amount: 0.0,
            donor: MotionDonorSnapshot::None,
            carrier: "transparent".into(),
            confidence_threshold: 0.1,
            confidence_softness: 0.05,
            refresh: 1.0,
            decay: 1.0,
            occlusion: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurvedShutterSnapshot {
    pub angle_degrees: f32,
    pub phase: f32,
    pub curvature: f32,
    pub chromatic_lag: f32,
    pub quality: String,
    pub sample_count: u8,
}

impl Default for CurvedShutterSnapshot {
    fn default() -> Self {
        Self {
            angle_degrees: 0.0,
            phase: 0.0,
            curvature: 0.0,
            chromatic_lag: 0.0,
            quality: "sharp".into(),
            sample_count: 1,
        }
    }
}

/// Read-only planner plus committed-renderer truth. Planned fields describe
/// immutable admission; rendered fields remain `none`/unattached until the
/// matching GPU parity slot has actually committed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MotionTelemetrySnapshot {
    #[serde(default)]
    pub effective_source: String,
    #[serde(default)]
    pub rendered_source: String,
    #[serde(default)]
    pub codec_vectors_available: bool,
    #[serde(default)]
    pub fallback_active: bool,
    #[serde(default)]
    pub field_dimensions: [u32; 2],
    #[serde(default)]
    pub vector_count: u64,
    #[serde(default)]
    pub field_planned: bool,
    #[serde(default)]
    pub field_attached: bool,
    #[serde(default)]
    pub required_as_donor: bool,
    #[serde(default)]
    pub transplant_admitted: bool,
    #[serde(default)]
    pub donor_missing: bool,
    #[serde(default)]
    pub carrier_valid: bool,
    #[serde(default)]
    pub memory_generation: u64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub diagnostic: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionSnapshot {
    pub algorithm_version: u16,
    pub field_source: String,
    pub lattice_quality: String,
    /// Additive: the B2 procedural field scalars. An older client that never
    /// reads this key sees exactly the pre-B2 snapshot, and a snapshot written
    /// without it still deserializes to the neutral default.
    #[serde(default)]
    pub procedural: ProceduralFieldSnapshot,
    /// Additive like `procedural`: absent means the neutral default.
    #[serde(default)]
    pub shaping: FlowShapingSnapshot,
    pub transplant: FaradayMotionSnapshot,
    pub shutter: CurvedShutterSnapshot,
    /// Additive: an older client that never reads this key sees exactly the
    /// pre-collider snapshot, and a snapshot written without it still
    /// deserializes to the disabled default.
    #[serde(default)]
    pub collider: FieldColliderSnapshot,
    #[serde(default)]
    pub telemetry: MotionTelemetrySnapshot,
}

impl Default for MotionSnapshot {
    fn default() -> Self {
        Self::from_params(crate::motion::MotionParams::default())
    }
}

impl MotionSnapshot {
    pub fn from_params(params: crate::motion::MotionParams) -> Self {
        use crate::motion::{
            CurvedShutterQuality, MotionCarrier, MotionFieldSource, MotionLatticeQuality,
        };
        let params = params.sanitized();
        let field_source = match params.field_source {
            MotionFieldSource::Auto => "auto",
            MotionFieldSource::CodecVectors => "codec_vectors",
            MotionFieldSource::Lattice => "lattice",
            MotionFieldSource::Procedural(kind) => kind.source_key(),
        };
        let lattice_quality = match params.lattice_quality {
            MotionLatticeQuality::Draft => "draft",
            MotionLatticeQuality::Live => "live",
            MotionLatticeQuality::High => "high",
        };
        let donor = motion_donor_snapshot(params.transplant.donor);
        let carrier = match params.transplant.carrier {
            MotionCarrier::Transparent => "transparent",
            MotionCarrier::Black => "black",
            MotionCarrier::FirstSourceFrame => "first_source_frame",
        };
        let quality = match params.shutter.quality {
            CurvedShutterQuality::Sharp => "sharp",
            CurvedShutterQuality::Draft => "draft",
            CurvedShutterQuality::Live => "live",
            CurvedShutterQuality::High => "high",
        };
        Self {
            algorithm_version: params.algorithm_version,
            field_source: field_source.into(),
            lattice_quality: lattice_quality.into(),
            procedural: ProceduralFieldSnapshot {
                scale: params.procedural.scale,
                rate: params.procedural.rate,
            },
            shaping: FlowShapingSnapshot {
                stretch: params.shaping.stretch,
                edge_repel: params.shaping.edge_repel,
                vector_trash: params.shaping.vector_trash,
                trash_block_size: params.shaping.trash_block_size,
            },
            transplant: FaradayMotionSnapshot {
                amount: params.transplant.amount,
                donor,
                carrier: carrier.into(),
                confidence_threshold: params.transplant.confidence_threshold,
                confidence_softness: params.transplant.confidence_softness,
                refresh: params.transplant.refresh,
                decay: params.transplant.decay,
                occlusion: params.transplant.occlusion,
            },
            shutter: CurvedShutterSnapshot {
                angle_degrees: params.shutter.angle_degrees,
                phase: params.shutter.phase,
                curvature: params.shutter.curvature,
                chromatic_lag: params.shutter.chromatic_lag,
                quality: quality.into(),
                sample_count: params.shutter.quality.sample_count(),
            },
            collider: FieldColliderSnapshot::from_params(params.collider),
            telemetry: MotionTelemetrySnapshot::default(),
        }
    }
}

/// Spout output state sent to the browser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpoutSnapshot {
    pub enabled: bool,
    pub active: bool,
    /// Stable session preference: `native` or `1080p`.
    #[serde(default)]
    pub resolution: String,
    /// Dimensions most recently delivered by the active sender. Zero means no
    /// frame has yet been accepted for this worker generation.
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
}

/// One physical display offered to the legacy fullscreen Output selector.
/// The browser treats `id` as opaque and sends it back verbatim.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputDisplaySnapshot {
    pub id: String,
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub x: i32,
    pub y: i32,
    #[serde(default)]
    pub editor: bool,
}

/// Browser-safe program recorder status. Counter fields are monotonic for one
/// worker run and reset only when Main accepts a new recorder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProgramRecorderSnapshot {
    /// idle | starting | recording | finishing | succeeded | failed | cancelled
    #[serde(default)]
    pub status: String,
    pub attempted: u64,
    pub accepted: u64,
    pub encoded: u64,
    pub duplicated: u64,
    pub dropped_not_ready: u64,
    pub dropped_source_unavailable: u64,
    pub dropped_pool_empty: u64,
    pub dropped_queue_full: u64,
    pub rejected_metadata: u64,
    pub encoder_failures: u64,
    /// This foundation publishes exact audio-clock correlation but truthfully
    /// does not mux audio until Main has a bounded program PCM source.
    #[serde(default = "bool_true")]
    pub audio_not_muxed: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub artifact_name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
}

impl Default for ProgramRecorderSnapshot {
    fn default() -> Self {
        Self {
            status: "idle".into(),
            attempted: 0,
            accepted: 0,
            encoded: 0,
            duplicated: 0,
            dropped_not_ready: 0,
            dropped_source_unavailable: 0,
            dropped_pool_empty: 0,
            dropped_queue_full: 0,
            rejected_metadata: 0,
            encoder_failures: 0,
            audio_not_muxed: true,
            artifact_name: String::new(),
            error: String::new(),
        }
    }
}

/// Stable capture source selected by the operator. Display positions are not
/// accepted, so reorder/delete races become explicit Main-side failures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "target", rename_all = "snake_case")]
pub enum CaptureTargetSnapshot {
    Program,
    Layer { layer_id: String },
    Group { group_id: String },
}

/// Patch morph crossfader state sent to the browser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MorphSnapshot {
    pub has_a: bool,
    pub has_b: bool,
    /// True while both slots are set (crossfader engaged).
    pub active: bool,
    pub t: f32,
    #[serde(default)]
    pub blend_law: String,
    #[serde(default)]
    pub gliding: bool,
    #[serde(default)]
    pub glide_target: f32,
    /// Remaining beat duration at the published authoritative clock.
    #[serde(default)]
    pub glide_duration_beats: f64,
    /// B15 snapshot bank: which of the eight slots hold a rig, so the panel
    /// can light its lamps. Additive — an older client sees an empty vector
    /// and shows no bank at all.
    #[serde(default)]
    pub bank_filled: Vec<bool>,
    /// How long a bank recall takes to travel, in beats.
    #[serde(default)]
    pub bank_glide_beats: f64,
}

/// MIDI input state sent to the browser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MidiSnapshot {
    pub enabled: bool,
    pub slots: Vec<MidiSlotSnapshot>,
    /// Slot index currently armed for MIDI learn, if any.
    pub learning: Option<usize>,
    /// Follow external MIDI timing clock.
    #[serde(default)]
    pub clock_sync: bool,
    /// True while external clock pulses are actively driving the beat.
    #[serde(default)]
    pub clock_active: bool,
    /// Current tempo estimate while the external clock is active.
    #[serde(default)]
    pub clock_bpm: f32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub port: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiSlotSnapshot {
    pub cc: u8,
    /// Live value 0..1, for UI meters.
    pub value: f32,
}

/// Browser-safe truth from the typed controller profile and MIDI supervisor.
/// This complements the legacy four-slot `MidiSnapshot`; it does not add a
/// generic remote-dispatch surface or expose host paths.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControllerRuntimeSnapshot {
    #[serde(default)]
    pub profile_revision: u64,
    #[serde(default)]
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
    #[serde(default)]
    pub midi: MidiRuntimeSnapshot,
}

impl ControllerRuntimeSnapshot {
    pub fn from_runtime(
        profile_revision: u64,
        profile: &crate::controller_profile::ControllerProfileDocument,
        status: &str,
        runtime: &crate::midi::MidiRuntimeSnapshot,
    ) -> Self {
        Self {
            profile_revision,
            name: bounded_runtime_text(&profile.name),
            status: bounded_runtime_text(status),
            midi: MidiRuntimeSnapshot::from_runtime(runtime),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiRuntimeSnapshot {
    #[serde(default)]
    pub phase: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub input_port: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub output_port: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_inputs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_outputs: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
    #[serde(default)]
    pub counters: MidiCountersSnapshot,
}

impl MidiRuntimeSnapshot {
    pub fn from_runtime(runtime: &crate::midi::MidiRuntimeSnapshot) -> Self {
        Self {
            phase: midi_runtime_phase_key(runtime.phase).into(),
            input_port: runtime
                .input_port
                .as_deref()
                .map_or_else(String::new, bounded_runtime_text),
            output_port: runtime
                .output_port
                .as_deref()
                .map_or_else(String::new, bounded_runtime_text),
            available_inputs: runtime
                .available_inputs
                .iter()
                .take(128)
                .map(|name| bounded_runtime_text(name))
                .collect(),
            available_outputs: runtime
                .available_outputs
                .iter()
                .take(128)
                .map(|name| bounded_runtime_text(name))
                .collect(),
            error: runtime
                .error
                .as_deref()
                .map_or_else(String::new, bounded_runtime_text),
            counters: runtime.counters.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiCountersSnapshot {
    pub raw_received: u64,
    pub malformed: u64,
    pub input_queue_dropped: u64,
    pub decoded_events: u64,
    pub event_queue_dropped: u64,
    pub channel_or_unmapped: u64,
    pub loop_suppressed: u64,
    pub feedback_queued: u64,
    pub feedback_coalesced: u64,
    pub feedback_dropped: u64,
    pub feedback_sent: u64,
    pub feedback_rate_limited: u64,
    pub scans: u64,
    pub reconnects: u64,
    pub disconnects: u64,
}

impl From<crate::midi::MidiCounters> for MidiCountersSnapshot {
    fn from(counters: crate::midi::MidiCounters) -> Self {
        Self {
            raw_received: counters.raw_received,
            malformed: counters.malformed,
            input_queue_dropped: counters.input_queue_dropped,
            decoded_events: counters.decoded_events,
            event_queue_dropped: counters.event_queue_dropped,
            channel_or_unmapped: counters.channel_or_unmapped,
            loop_suppressed: counters.loop_suppressed,
            feedback_queued: counters.feedback_queued,
            feedback_coalesced: counters.feedback_coalesced,
            feedback_dropped: counters.feedback_dropped,
            feedback_sent: counters.feedback_sent,
            feedback_rate_limited: counters.feedback_rate_limited,
            scans: counters.scans,
            reconnects: counters.reconnects,
            disconnects: counters.disconnects,
        }
    }
}

/// Read-only OSC listener/runtime truth. `lan_warning` is deliberately always
/// serialized: operators must never infer exposure from an absent field.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OscRuntimeSnapshot {
    #[serde(default)]
    pub phase: String,
    #[serde(default)]
    pub bind_address: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub bound_address: String,
    #[serde(default)]
    pub running: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
    #[serde(default)]
    pub counters: OscCountersSnapshot,
    #[serde(default)]
    pub lan_warning: bool,
}

impl OscRuntimeSnapshot {
    pub fn from_runtime(
        config: &crate::osc::OscConfigDocument,
        status: &str,
        runtime: &crate::osc::OscRuntimeSnapshot,
    ) -> Self {
        let runtime_error = runtime.error.as_deref().unwrap_or_default();
        let error = match (status.is_empty(), runtime_error.is_empty()) {
            (true, true) => String::new(),
            (false, true) => bounded_runtime_text(status),
            (true, false) => bounded_runtime_text(runtime_error),
            (false, false) => bounded_runtime_text(&format!("{status}; {runtime_error}")),
        };
        Self {
            phase: osc_runtime_phase_key(runtime.phase).into(),
            bind_address: config.bind.address().to_string(),
            bound_address: runtime
                .bound_address
                .map_or_else(String::new, |address| address.to_string()),
            running: matches!(runtime.phase, crate::osc::OscRuntimePhase::Listening),
            error,
            counters: runtime.counters.into(),
            lan_warning: config.bind.lan_warning() || runtime.lan_warning,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OscCountersSnapshot {
    pub datagrams_received: u64,
    pub messages_received: u64,
    pub malformed: u64,
    pub rate_dropped: u64,
    pub queue_dropped: u64,
    pub loop_suppressed: u64,
    pub feedback_queued: u64,
    pub feedback_coalesced: u64,
    pub feedback_dropped: u64,
    pub feedback_rate_limited: u64,
    pub feedback_sent: u64,
    pub feedback_send_errors: u64,
}

impl From<crate::osc::OscCounters> for OscCountersSnapshot {
    fn from(counters: crate::osc::OscCounters) -> Self {
        Self {
            datagrams_received: counters.datagrams_received,
            messages_received: counters.messages_received,
            malformed: counters.malformed,
            rate_dropped: counters.rate_dropped,
            queue_dropped: counters.queue_dropped,
            loop_suppressed: counters.loop_suppressed,
            feedback_queued: counters.feedback_queued,
            feedback_coalesced: counters.feedback_coalesced,
            feedback_dropped: counters.feedback_dropped,
            feedback_rate_limited: counters.feedback_rate_limited,
            feedback_sent: counters.feedback_sent,
            feedback_send_errors: counters.feedback_send_errors,
        }
    }
}

fn bounded_runtime_text(value: &str) -> String {
    crate::controller_profile::bounded_status(value.to_string())
}

const fn midi_runtime_phase_key(phase: crate::midi::MidiConnectionPhase) -> &'static str {
    match phase {
        crate::midi::MidiConnectionPhase::Disabled => "disabled",
        crate::midi::MidiConnectionPhase::Scanning => "scanning",
        crate::midi::MidiConnectionPhase::WaitingForDevice => "waiting_for_device",
        crate::midi::MidiConnectionPhase::Connected => "connected",
        crate::midi::MidiConnectionPhase::Error => "error",
    }
}

const fn osc_runtime_phase_key(phase: crate::osc::OscRuntimePhase) -> &'static str {
    match phase {
        crate::osc::OscRuntimePhase::Disabled => "disabled",
        crate::osc::OscRuntimePhase::Binding => "binding",
        crate::osc::OscRuntimePhase::Listening => "listening",
        crate::osc::OscRuntimePhase::Error => "error",
    }
}

/// Per-layer info sent to the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSnapshot {
    /// Immutable engine identity. Unlike the display index, this survives a
    /// reorder and lets concurrent controllers address the intended layer.
    #[serde(default)]
    pub layer_id: String,
    pub filename: String,
    pub visible: bool,
    /// True when the layer skips inherited Digital/Analog/Cellular/Motion
    /// processing. VHS remains a final-program finish. Any contributing bypass
    /// links the complete shared Temporal family dry while history stays warm.
    #[serde(default)]
    pub bypass_master_fx: bool,
    /// Independently authored request to bypass inherited Temporal FX for
    /// this layer. Missing values from older engines default to inheritance.
    #[serde(default)]
    pub bypass_temporal_fx: bool,
    /// Video-only deterministic pattern reroll at each decoded loop boundary.
    #[serde(default)]
    pub reroll_on_loop: bool,
    pub paused: bool,
    pub opacity: f32,
    /// Continuous spatial contribution to the one shared Codec Mosh stage.
    #[serde(default = "default_layer_mosh_send")]
    pub mosh_send: f32,
    pub speed: f32,
    #[serde(default = "default_layer_fps")]
    pub fps: f32,
    pub blend_mode: String,
    /// P4c authored delivery policy token (`legacy_rgba` default, or
    /// `metadata_managed`). Additive: legacy clients ignore it.
    #[serde(default = "default_layer_delivery")]
    pub delivery: String,
    /// Whether the pixels currently on the GPU arrived through the planar
    /// path — the truthful delivery fact, distinct from the authored ask.
    #[serde(default)]
    pub delivery_active_planar: bool,
    pub progress: f32,
    #[serde(default)]
    pub key_mode: u32,
    #[serde(default)]
    pub key_threshold: f32,
    #[serde(default)]
    pub key_softness: f32,
    #[serde(default = "default_key_color")]
    pub key_color: [f32; 3],
    #[serde(default = "default_key_tolerance")]
    pub key_tolerance: f32,
    #[serde(default)]
    pub effects: EffectsSnapshot,
    /// Authored field/transplant/shutter law plus renderer telemetry. Defaults
    /// are an exact no-op for legacy clients.
    #[serde(default)]
    pub motion: MotionSnapshot,
    /// Resolution-independent authored transform. The serde default preserves
    /// the exact inactive pre-transform sampling path for old snapshots while
    /// exposed canvas becomes transparent after an authored transform.
    #[serde(default)]
    pub transform: SpatialTransform,
    /// `video` for ordinary decoded clips or `spout` for a live receiver.
    #[serde(default)]
    pub source_kind: String,
    /// Requested/connected Spout sender name. Empty for video layers.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_name: String,
    #[serde(default)]
    pub source_active: bool,
    #[serde(default)]
    pub source_width: u32,
    #[serde(default)]
    pub source_height: u32,
    #[serde(default)]
    pub source_sequence: u64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_error: String,
    /// Explicitly tells the performer how non-file sources behave in export.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub offline_export_policy: String,
    /// First eight hex characters of the proxy cache key while a validated
    /// artifact backs this layer's decoder — the same prefix the native HUD
    /// shows. Empty when the layer is un-proxied. Never a path.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub proxy_backing_prefix: String,
    /// The session's latest proxy lifecycle/refusal note for this layer's
    /// identity (requested, running, ready/adopting, adopted, refused …) —
    /// the same note the native HUD appends. Keys and byte counts only.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub proxy_note: String,
    /// Prepared sources, true per-clip transport, and matte routing owned by
    /// this persistent visual layer identity.
    #[serde(default)]
    pub performance: LayerPerformanceSnapshot,
    /// B7 pattern-synth authored state, present only on a pattern layer.
    /// Additive: legacy clients that ignore it keep their exact meaning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<crate::patch::PatternSynthConfig>,
    /// B7 text-page authored state, present only on a text layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_page: Option<crate::patch::TextPageConfig>,
}

fn default_layer_delivery() -> String {
    crate::video::PlanarDeliveryPolicy::LegacyRgba
        .key()
        .to_string()
}

fn default_layer_fps() -> f32 {
    30.0
}

fn default_layer_mosh_send() -> f32 {
    1.0
}

/// Browser-safe view of one prepared source. Host paths never cross the web
/// boundary: `filename` is a logical library identity, while the engine owns
/// canonical/content-addressed resolution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipSlotSnapshot {
    pub id: ClipSlotId,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    pub filename: String,
    #[serde(default)]
    pub transport: ClipTransportConfig,
    #[serde(default)]
    pub playhead: NormalizedTime,
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub prepared: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
}

/// Live image-input identity. Selected donors are always addressed by the
/// same non-zero decimal stable layer ID used by every current layer action;
/// display positions are deliberately absent from mutable routing messages.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum ImageInputSnapshot {
    SelectedLayer {
        layer_id: String,
        #[serde(default)]
        stage: LayerImageStage,
    },
    /// Exact patch restore may expose an authored donor that no longer maps.
    /// Browsers can display this diagnostic but server validation rejects it
    /// as a newly authored route.
    MissingSelectedLayer {
        saved_position: SavedLayerPosition,
        #[serde(default)]
        stage: LayerImageStage,
    },
    #[default]
    OneBelow,
    AllBelow,
    CleanProgram,
    ProgramHistory,
    /// Non-zero decimal group identity; strings avoid JavaScript precision
    /// loss for the shared u64 creative-ID domain.
    GroupOutput {
        group_id: String,
    },
    /// Retained output-only tombstone for a deleted group. It is exposed so
    /// clients can diagnose the inert authored route, but ingress must never
    /// accept it as a newly selected input.
    MissingGroupOutput {
        group_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatteSnapshot {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub input: ImageInputSnapshot,
    #[serde(default)]
    pub channel: MatteChannel,
    #[serde(default)]
    pub invert: bool,
    #[serde(default = "default_matte_amount")]
    pub amount: f32,
    #[serde(default = "default_matte_threshold")]
    pub threshold: f32,
    #[serde(default = "default_matte_softness")]
    pub softness: f32,
    /// disabled | ready | missing | cycle
    #[serde(default)]
    pub diagnostic: String,
}

const fn default_matte_amount() -> f32 {
    1.0
}

const fn default_matte_threshold() -> f32 {
    0.5
}

const fn default_matte_softness() -> f32 {
    0.1
}

impl Default for MatteSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            input: ImageInputSnapshot::default(),
            channel: MatteChannel::Alpha,
            invert: false,
            amount: default_matte_amount(),
            threshold: default_matte_threshold(),
            softness: default_matte_softness(),
            diagnostic: String::new(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LayerPerformanceSnapshot {
    #[serde(default)]
    pub active_slot_id: Option<ClipSlotId>,
    #[serde(default)]
    pub slots: Vec<ClipSlotSnapshot>,
    #[serde(default)]
    pub matte: MatteSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneBindingSnapshot {
    /// Live stable identity, not a saved position or current display index.
    pub layer_id: String,
    pub slot_id: ClipSlotId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cue_id: Option<CueId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneSnapshot {
    pub id: SceneId,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default)]
    pub trigger_mode: TriggerMode,
    #[serde(default)]
    pub bindings: Vec<SceneBindingSnapshot>,
    #[serde(default)]
    pub prepared: bool,
    #[serde(default)]
    pub pending: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
}

/// Serializable browser vocabulary for the pure Autopilot scheduler state.
/// Keeping the wire enum here lets the domain scheduler remain free of web
/// serialization concerns while Main performs one exhaustive conversion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutopilotPhaseSnapshot {
    #[default]
    Stopped,
    Starting,
    Running,
    Paused,
    Stalled,
    Faulted,
    Complete,
}

impl From<AutopilotState> for AutopilotPhaseSnapshot {
    fn from(state: AutopilotState) -> Self {
        match state {
            AutopilotState::Stopped => Self::Stopped,
            AutopilotState::Starting => Self::Starting,
            AutopilotState::Running => Self::Running,
            AutopilotState::Paused => Self::Paused,
            AutopilotState::Stalled => Self::Stalled,
            AutopilotState::Faulted => Self::Faulted,
            AutopilotState::Complete => Self::Complete,
        }
    }
}

/// Authored sequence plus session-only transport truth shown by the browser.
/// Step positions are zero-based indices into `plan.steps`; the stable Scene
/// IDs remain in the plan and are never replaced by transient live identities.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutopilotSnapshot {
    #[serde(default)]
    pub plan: AutopilotPlan,
    #[serde(default)]
    pub phase: AutopilotPhaseSnapshot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_step: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_step: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beats_remaining: Option<u64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    #[serde(default)]
    pub scenes: Vec<SceneSnapshot>,
    #[serde(default)]
    pub prepared_scene_id: Option<SceneId>,
    #[serde(default)]
    pub pending_scene_id: Option<SceneId>,
    /// Scene-only beat Autopilot authoring and session transport state.
    #[serde(default)]
    pub autopilot: AutopilotSnapshot,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
    /// Atomic Scene prepare/quantize/commit/rejection status.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scene_staging_status: String,
    /// Prepared-source open/decode/upload/cache admission status.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_staging_status: String,
    /// Frame-plan route diagnostics, including missing donors and rejected
    /// same-frame cycles. Kept separate from source/scene transaction status.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub image_routing_status: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerollScope {
    Master,
    Layer,
    Group,
    All,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerollMode {
    #[default]
    Pattern,
    Variation,
}

fn default_reroll_amount() -> f32 {
    0.7
}

/// Actions the browser can request (processed by the render loop).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum WebAction {
    /// Defer an otherwise ordinary action until the next four-beat downbeat.
    #[serde(rename = "quantized")]
    Quantized { inner: Box<WebAction> },
    /// Set a master effect parameter
    #[serde(rename = "set_param")]
    SetParam {
        param: String,
        value: serde_json::Value,
    },
    /// Set one absolute authored master-transform field.
    #[serde(rename = "set_master_transform")]
    SetMasterTransform {
        param: String,
        value: serde_json::Value,
    },
    /// Coalescible value edit. `node_kind` selects the authoritative parameter
    /// descriptor; the engine verifies it again against the addressed node.
    #[serde(rename = "set_visual_node_param")]
    SetVisualNodeParam {
        scope: CreativeScopeSnapshot,
        node_id: String,
        node_kind: String,
        param: String,
        value: serde_json::Value,
        composition_revision: u64,
    },
    /// Topology mutations are ordered barriers and require an exact current
    /// composition revision. Node/group IDs are always engine allocated.
    #[serde(rename = "insert_visual_node")]
    InsertVisualNode {
        scope: CreativeScopeSnapshot,
        index: usize,
        node_kind: String,
        composition_revision: u64,
    },
    #[serde(rename = "remove_visual_node")]
    RemoveVisualNode {
        scope: CreativeScopeSnapshot,
        node_id: String,
        composition_revision: u64,
    },
    #[serde(rename = "move_visual_node")]
    MoveVisualNode {
        scope: CreativeScopeSnapshot,
        node_id: String,
        to: usize,
        composition_revision: u64,
    },
    /// Replace only a Mask node's structural variant. Selecting image installs
    /// the deterministic OneBelow/current-frame/alpha/non-inverted default;
    /// the route action may refine it in a later ordered transaction.
    #[serde(rename = "set_visual_node_mask_variant")]
    SetVisualNodeMaskVariant {
        scope: CreativeScopeSnapshot,
        node_id: String,
        variant: String,
        composition_revision: u64,
    },
    #[serde(rename = "set_visual_node_route")]
    SetVisualNodeRoute {
        scope: CreativeScopeSnapshot,
        node_id: String,
        route: CreativeImageTapSnapshot,
        channel: String,
        invert: bool,
        composition_revision: u64,
    },
    /// Reroute a Displace node's donor. The route is the only Displace field
    /// that rewrites the image dependency graph, so it is an ordered,
    /// revision-protected, uncoalesced barrier of its own. The two gains and
    /// the boundary law travel on the ordinary coalescible parameter action.
    #[serde(rename = "set_visual_node_displace_route")]
    SetVisualNodeDisplaceRoute {
        scope: CreativeScopeSnapshot,
        node_id: String,
        route: CreativeImageTapSnapshot,
        composition_revision: u64,
    },
    /// Reroute one of a Symmetry Field's four fixed slots. Slot index is route
    /// identity, so the addressed slot travels inside the payload and an edit
    /// to slot 1 can never be mistaken for an edit to slot 0. Rewiring either
    /// class rewrites the image dependency graph or the motion field request,
    /// so this is an ordered, revision-protected, uncoalesced barrier. Every
    /// other Symmetry field — geometry, masks, seed, mode, boundary — travels
    /// on the ordinary coalescible parameter action.
    #[serde(rename = "set_visual_node_symmetry_route")]
    SetVisualNodeSymmetryRoute {
        scope: CreativeScopeSnapshot,
        node_id: String,
        route: SymmetryRouteSnapshot,
        composition_revision: u64,
    },
    /// Reroute one slot of a Residual node. Both routes rewrite the image
    /// dependency graph, so the action is an ordered, revision-protected,
    /// uncoalesced barrier and carries a closed slot token rather than an
    /// index: a slot the engine does not know names no route at all instead of
    /// silently landing on its partner. The two gains, both discrete laws and
    /// the seed travel on the ordinary coalescible parameter action.
    #[serde(rename = "set_visual_node_residual_route")]
    SetVisualNodeResidualRoute {
        scope: CreativeScopeSnapshot,
        node_id: String,
        slot: ResidualRouteSlotSnapshot,
        route: CreativeImageTapSnapshot,
        composition_revision: u64,
    },
    /// Assign, replace, or clear (`document: null`) a Study node's document.
    /// The engine validates and compiles the document into the bounded host
    /// library and the node keeps only the canonical digest — one action, so
    /// a document can never land in the library without its node or vice
    /// versa. Coalesced per node (an absolute value: the newest paste wins),
    /// never quantized, and no route is involved: a Study reads only its
    /// carrier and the master ring, so this is not a dependency-graph edit.
    #[serde(rename = "set_visual_node_study_document")]
    SetVisualNodeStudyDocument {
        scope: CreativeScopeSnapshot,
        node_id: String,
        #[serde(default)]
        document: serde_json::Value,
    },
    /// Enable/disable or reroute the separate group matte. `None` disables it;
    /// numeric matte values remain untouched and use the coalescible action.
    #[serde(rename = "set_composition_group_matte_route")]
    SetCompositionGroupMatteRoute {
        group_id: String,
        route: Option<CreativeImageTapSnapshot>,
        channel: String,
        invert: bool,
        composition_revision: u64,
    },
    #[serde(rename = "set_composition_group_matte_param")]
    SetCompositionGroupMatteParam {
        group_id: String,
        param: String,
        value: serde_json::Value,
        composition_revision: u64,
    },
    #[serde(rename = "set_composition_group_param")]
    SetCompositionGroupParam {
        group_id: String,
        param: String,
        value: serde_json::Value,
        composition_revision: u64,
    },
    #[serde(rename = "create_composition_group")]
    CreateCompositionGroup {
        name: String,
        #[serde(default)]
        member_layer_ids: Vec<String>,
        root_index: usize,
        composition_revision: u64,
    },
    #[serde(rename = "remove_composition_group")]
    RemoveCompositionGroup {
        group_id: String,
        composition_revision: u64,
    },
    #[serde(rename = "set_composition_group_members")]
    SetCompositionGroupMembers {
        group_id: String,
        member_layer_ids: Vec<String>,
        composition_revision: u64,
    },
    #[serde(rename = "move_composition_root_item")]
    MoveCompositionRootItem {
        item: CompositionRootSnapshot,
        to: usize,
        composition_revision: u64,
    },
    /// Continuous A/B performance control; it does not mutate topology or
    /// advance the optimistic composition revision.
    #[serde(rename = "set_composition_bus_crossfade")]
    SetCompositionBusCrossfade { value: f32 },
    /// One B8 bus-mixer value (wipe, blend, dirt, or melt). Like the
    /// crossfade it is an ordinary coalescible value edit: no topology, no
    /// revision. The closed param vocabulary lives in
    /// `mixing_boundary::BusMixerEdit`.
    #[serde(rename = "set_composition_bus_mix")]
    SetCompositionBusMixParam {
        param: String,
        value: serde_json::Value,
    },
    /// Stable direct-layer bus assignment is an ordered topology edit.
    #[serde(rename = "set_composition_layer_bus")]
    SetCompositionLayerBus {
        layer_id: String,
        bus: String,
        composition_revision: u64,
    },
    /// Restore the exact legacy full-frame master-transform identity.
    #[serde(rename = "reset_master_transform")]
    ResetMasterTransform,
    /// Atomically replace the complete master transform (paste/preset).
    #[serde(rename = "apply_master_transform")]
    ApplyMasterTransform { transform: SpatialTransform },
    /// Add a layer from the library by filename
    #[serde(rename = "add_layer")]
    AddLayer { filename: String },
    /// Add a B7 pattern-synth layer with default authored values. Topology
    /// edit: immediate, never coalesced, never quantized.
    #[serde(rename = "add_pattern_layer")]
    AddPatternLayer,
    /// Add a B7 text-page layer with the default page. Same topology law.
    #[serde(rename = "add_text_layer")]
    AddTextLayer,
    /// Stage a library source into a persistent layer. `None` appends a new
    /// engine-assigned slot; `Some` replaces that exact stable slot. Source
    /// open and first-frame decode complete before any optional activation.
    #[serde(rename = "load_clip_into_slot")]
    LoadClipIntoSlot {
        layer_id: String,
        #[serde(default)]
        slot_id: Option<ClipSlotId>,
        filename: String,
        #[serde(default)]
        activate: bool,
        #[serde(default)]
        trigger_mode: TriggerMode,
    },
    /// Remove one exact prepared source. The active/last slot protections are
    /// enforced again by the engine against its authoritative live set.
    #[serde(rename = "remove_clip_slot")]
    RemoveClipSlot {
        layer_id: String,
        slot_id: ClipSlotId,
    },
    /// Atomically swap the prepared source while retaining the visual layer's
    /// stable identity, effects, transform, modulation, and image routes.
    #[serde(rename = "activate_clip_slot")]
    ActivateClipSlot {
        layer_id: String,
        slot_id: ClipSlotId,
        #[serde(default)]
        trigger_mode: TriggerMode,
    },
    /// Edit one scalar field of a slot's authored source-time law. Ingress
    /// accepts only a closed field/value vocabulary and coalesces by target.
    #[serde(rename = "set_clip_transport")]
    SetClipTransport {
        layer_id: String,
        slot_id: ClipSlotId,
        param: String,
        value: serde_json::Value,
    },
    /// Add or move one bounded cue. Cue position is a scalar edit and may
    /// coalesce; cue removal/triggering remain ordered barriers.
    #[serde(rename = "set_clip_cue")]
    SetClipCue {
        layer_id: String,
        slot_id: ClipSlotId,
        cue_id: CueId,
        at: NormalizedTime,
    },
    #[serde(rename = "remove_clip_cue")]
    RemoveClipCue {
        layer_id: String,
        slot_id: ClipSlotId,
        cue_id: CueId,
    },
    #[serde(rename = "trigger_clip_cue")]
    TriggerClipCue {
        layer_id: String,
        slot_id: ClipSlotId,
        cue_id: CueId,
    },
    /// Direct discontinuous playhead movement. Consecutive requests for the
    /// same stable layer/slot coalesce newest-only; no intervening command is
    /// crossed, because every accepted seek advances decoder generation.
    #[serde(rename = "seek_clip_slot")]
    SeekClipSlot {
        layer_id: String,
        slot_id: ClipSlotId,
        position: NormalizedTime,
    },
    /// SMPTE-style discontinuous seek. Main resolves this typed address only
    /// against the exact active source duration, then sends the same
    /// generation-tagged normalized request used by scratching.
    #[serde(rename = "seek_clip_slot_timecode")]
    SeekClipSlotTimecode {
        layer_id: String,
        slot_id: ClipSlotId,
        timecode: SourceTimecode,
    },
    /// Decode/upload every source required by a scene without changing output.
    #[serde(rename = "prepare_scene")]
    PrepareScene { scene_id: SceneId },
    /// Capture the active slot on every current live layer as one bounded,
    /// positional patch Scene. `None` requests a new engine-assigned stable
    /// Scene ID; `Some` recaptures that exact existing Scene. Live layer IDs
    /// deliberately never cross this authoring boundary into persistence.
    #[serde(rename = "capture_scene")]
    CaptureScene {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scene_id: Option<SceneId>,
        #[serde(default)]
        name: String,
        #[serde(default)]
        trigger_mode: TriggerMode,
    },
    /// Remove one exact authored Scene. This is an ordered topology barrier so
    /// a pending prepare/trigger can never be silently retargeted by ID reuse.
    #[serde(rename = "remove_scene")]
    RemoveScene { scene_id: SceneId },
    /// Commit all prepared scene bindings at one program-clock boundary or do
    /// nothing. `None` uses the scene's authored trigger mode.
    #[serde(rename = "trigger_scene")]
    TriggerScene {
        scene_id: SceneId,
        #[serde(default)]
        trigger_mode: Option<TriggerMode>,
    },
    /// Replace the complete bounded authored Scene sequence as one ordered,
    /// history-bearing transaction. Partial row edits never cross the wire.
    #[serde(rename = "replace_autopilot_plan")]
    ReplaceAutopilotPlan { plan: AutopilotPlan },
    /// Start from the beginning, or resume the explicitly paused sequence.
    /// Main owns validation, preparation, and the future-beat release law.
    #[serde(rename = "autopilot_play")]
    AutopilotPlay,
    /// Hold cursor/countdown while retaining the one prepared lookahead.
    #[serde(rename = "autopilot_pause")]
    AutopilotPause,
    /// Stop and return the cursor to step zero without changing visible media.
    #[serde(rename = "autopilot_reset")]
    AutopilotReset,
    /// Add a live Spout receiver layer by sender name.
    #[serde(rename = "add_spout_layer")]
    AddSpoutLayer { sender: String },
    /// Remove a layer by index
    #[serde(rename = "remove_layer")]
    RemoveLayer {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
    },
    /// Move one layer to another stack position.
    #[serde(rename = "move_layer")]
    MoveLayer {
        from: usize,
        to: usize,
        #[serde(default)]
        layer_id: Option<String>,
        #[serde(default)]
        stack_revision: Option<u64>,
    },
    /// Toggle layer visibility
    #[serde(rename = "toggle_visibility")]
    ToggleVisibility { index: usize },
    /// Toggle layer play/pause
    #[serde(rename = "toggle_layer_pause")]
    ToggleLayerPause { index: usize },
    /// Toggle master play/pause
    #[serde(rename = "toggle_master_pause")]
    ToggleMasterPause,
    /// Legacy compatibility command: reset only the direct master effect
    /// uniforms. NTSC, temporal state, modulation, morph, and queued
    /// automation remain untouched, matching the original protocol contract.
    #[serde(rename = "reset_fx")]
    ResetFx,
    /// Revert the complete master visual program. Layers, transport, BPM, and
    /// input/device selections are preserved.
    #[serde(rename = "reset_visual_program")]
    ResetVisualProgram,
    /// Reset a specific effect group (digital, analog, motion)
    #[serde(rename = "reset_group")]
    ResetGroup { group: String },
    /// Set a per-layer parameter (opacity, speed, blend_mode)
    #[serde(rename = "set_layer_param")]
    SetLayerParam {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        param: String,
        value: serde_json::Value,
    },
    /// Explicitly confirm one immutable planner-authored remediation preview.
    /// The engine resolves the ID only against its currently published
    /// diagnostic, verifies the exact composition revision, and then routes
    /// the declared operation through the ordinary action/history path.
    #[serde(rename = "apply_constraint_remediation")]
    ApplyConstraintRemediation {
        candidate_id: crate::diagnostics::RemediationCandidateId,
        composition_revision: u64,
    },
    /// Set one direct per-layer effect parameter.
    #[serde(rename = "set_layer_effect")]
    SetLayerEffect {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        param: String,
        value: serde_json::Value,
    },
    /// Set one absolute authored B7 pattern-synth value on a pattern layer.
    /// Coalescible per param; the three discrete vocabularies ride as closed
    /// snake_case tokens and coalesce like every absolute choice.
    #[serde(rename = "set_layer_pattern")]
    SetLayerPattern {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        param: String,
        value: serde_json::Value,
    },
    /// Set one absolute authored B7 text-page value on a text layer. The
    /// page re-rasters only when the sanitized state actually changes.
    #[serde(rename = "set_layer_text")]
    SetLayerText {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        param: String,
        value: serde_json::Value,
    },
    /// Set one absolute authored transform field on a layer. The wire shape
    /// keeps the optional ID for serde compatibility, but server ingress
    /// requires a present nonzero stable ID; index is diagnostic context only.
    #[serde(rename = "set_layer_transform")]
    SetLayerTransform {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        param: String,
        value: serde_json::Value,
    },
    /// Restore one layer's exact legacy full-frame transform identity.
    #[serde(rename = "reset_layer_transform")]
    ResetLayerTransform {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
    },
    /// Atomically replace one complete layer transform (paste/preset).
    #[serde(rename = "apply_layer_transform")]
    ApplyLayerTransform {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        transform: SpatialTransform,
    },
    /// Reset direct effects, Motion, and Codec-Mosh Send on one layer.
    #[serde(rename = "reset_layer_fx")]
    ResetLayerFx {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
    },
    /// Idempotent layer safety/transport setters. Legacy toggles remain below
    /// for old clients, but the bundled panel never depends on toggle parity.
    #[serde(rename = "set_layer_visibility")]
    SetLayerVisibility {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        visible: bool,
    },
    #[serde(rename = "set_layer_paused")]
    SetLayerPaused {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        paused: bool,
    },
    /// Edit one matte field. Amount/threshold/softness are coalesced scalar
    /// controls; enabled/channel/invert remain ordered atomic mode changes.
    /// Image donor changes use the separate barrier action below.
    #[serde(rename = "set_layer_matte_param")]
    SetLayerMatteParam {
        layer_id: String,
        param: String,
        value: serde_json::Value,
        /// Required by current clients when `enabled` can admit/remove a DAG
        /// edge; optional for backward-compatible scalar/channel edits.
        #[serde(default)]
        composition_revision: Option<u64>,
    },
    /// Replace the complete typed image input. Selected donors require a
    /// non-zero decimal stable ID and never accept a display-position fallback.
    #[serde(rename = "set_layer_matte_input")]
    SetLayerMatteInput {
        layer_id: String,
        input: ImageInputSnapshot,
        /// Current clients bind route replacement to creative topology. Old
        /// clients may omit this but still pass transactional live preflight.
        #[serde(default)]
        composition_revision: Option<u64>,
    },
    #[serde(rename = "set_master_paused")]
    SetMasterPaused { paused: bool },
    /// Canonical spelling for the complete program freeze. The legacy master
    /// pause actions remain accepted indefinitely.
    #[serde(rename = "set_program_frozen")]
    SetProgramFrozen { frozen: bool },
    /// Hold only source textures; clocks, effects, modulation, and histories run.
    #[serde(rename = "set_media_frozen")]
    SetMediaFrozen { frozen: bool },
    /// Change the bounded host-session policy for future source allocations.
    /// This setting is never persisted in patches or applied retroactively.
    #[serde(rename = "set_media_safety_mode")]
    SetMediaSafetyMode {
        mode: crate::media_safety::MediaSafetyMode,
    },
    /// Change only the host-session framing preference for future interactive
    /// file, still, and Spout layers. Patches never own this value.
    #[serde(rename = "set_new_layer_fit")]
    SetNewLayerFit { fit: FitMode },
    /// Change the host-session proxy settings for future encodes and cache
    /// consultations. The action always carries the complete tuple; the
    /// engine validates it and installs it into the single authored owner.
    /// Each tuple is its own content-addressed cache key by design, patches
    /// never own this value, and no live layer or published artifact is
    /// rewritten by a change.
    #[serde(rename = "set_proxy_settings")]
    SetProxySettings {
        scale: crate::proxy::ProxyScale,
        frame_rate: crate::proxy::ProxyFrameRate,
        include_audio: bool,
    },
    /// Deterministically choose a new stochastic pattern and optionally make
    /// bounded visual-control variations. Every press is an ordered barrier.
    #[serde(rename = "reroll")]
    Reroll {
        scope: RerollScope,
        #[serde(default)]
        index: Option<usize>,
        #[serde(default)]
        layer_id: Option<String>,
        #[serde(default)]
        group_id: Option<String>,
        #[serde(default)]
        stack_revision: Option<u64>,
        #[serde(default)]
        seed: Option<u32>,
        #[serde(default)]
        mode: RerollMode,
        #[serde(default = "default_reroll_amount")]
        amount: f32,
        #[serde(default)]
        include_grain_controls: bool,
        /// Explicit performer opt-in for bounded spatial variation. Pattern
        /// rerolls and automatic per-loop rerolls leave transforms untouched.
        #[serde(default)]
        include_transform: bool,
        /// Opt in to numeric/wet mutation for addressed M2 rack nodes. Routes,
        /// IDs, order, enabled state, and blends are never randomized.
        #[serde(default)]
        include_rack_controls: bool,
        /// Opt in to group opacity/transform values. Membership, solo/bypass,
        /// bus assignment, routes, and root order remain authored topology.
        #[serde(default)]
        include_group_controls: bool,
        /// B15 keep-masks. Each protects one domain from an otherwise ordinary
        /// throw, and each defaults to false — the established behaviour — so
        /// an unflagged action is byte-identical to every Dice before them.
        ///
        /// They are safe to compose because every Dice draw already runs in
        /// its own stable, domain-separated stream: the master uses stream 0
        /// keyed by the master seed, each layer uses stream `index + 1` keyed
        /// by that layer's own seed. Skipping a domain therefore cannot shift
        /// what another domain draws.
        ///
        /// Keep the layers exactly as they are.
        #[serde(default)]
        keep_source: bool,
        /// Keep the modulation matrix's diced state (the LFO seeds).
        #[serde(default)]
        keep_modulation: bool,
        /// Keep the master chain: its effects, transform, rack, composition
        /// values, motion, and temporal originals. Note that the master seed
        /// is part of that chain, so with this set the program's dice cursor
        /// does not advance and a throw that keeps everything else is a pure
        /// function of the current seed.
        #[serde(default)]
        keep_output_chain: bool,
    },
    #[serde(rename = "set_layer_reroll_on_loop")]
    SetLayerRerollOnLoop {
        index: usize,
        #[serde(default)]
        layer_id: Option<String>,
        enabled: bool,
    },
    #[serde(rename = "set_blackout")]
    SetBlackout { enabled: bool },
    /// Set an NTSC/VHS effect parameter
    #[serde(rename = "set_ntsc_param")]
    SetNtscParam {
        param: String,
        value: serde_json::Value,
    },
    /// Tap tempo: each tap refines BPM and re-anchors the downbeat
    #[serde(rename = "tap_tempo")]
    TapTempo,
    /// Set BPM directly
    #[serde(rename = "set_bpm")]
    SetBpm { value: f32 },
    /// Set an LFO parameter ("shape" | "beats" | "phase")
    #[serde(rename = "set_lfo")]
    SetLfo {
        index: usize,
        param: String,
        value: serde_json::Value,
    },
    /// B10: configure one envelope slot (`attack`/`decay` seconds, `trigger`
    /// and `mode` as closed tokens). Ordinary coalescible value edit.
    #[serde(rename = "set_envelope")]
    SetEnvelope {
        index: usize,
        param: String,
        value: serde_json::Value,
    },
    /// B10: set one macro knob (0..1). Ordinary coalescible value edit.
    #[serde(rename = "set_macro")]
    SetMacro { index: usize, value: f32 },
    /// B10: reseed the deterministic chaos/spike generators. Applying a new
    /// seed restarts their trajectories from zero, deterministically.
    #[serde(rename = "set_mod_seed")]
    SetModSeed { seed: u32 },
    /// B10: one bend pad's press or release edge.
    ///
    /// A stream event on the gesture-sample law: it deliberately has no
    /// coalesce key (a pending press must not be replaced by its release, or
    /// an envelope trigger would never see the edge), and both edges are
    /// priority — a dropped release would latch the pad on, and a dropped
    /// press would fire nothing the operator played.
    #[serde(rename = "bend_pad")]
    BendPad { index: usize, held: bool },
    /// Append a new modulation routing (defaults)
    #[serde(rename = "add_routing")]
    AddRouting,
    /// Remove a modulation routing by index
    #[serde(rename = "remove_routing")]
    RemoveRouting {
        index: usize,
        #[serde(default)]
        route_id: Option<String>,
    },
    /// Set a routing parameter ("source" | "target" | "depth")
    #[serde(rename = "set_routing")]
    SetRouting {
        index: usize,
        #[serde(default)]
        route_id: Option<String>,
        /// Stable identity of a positional `layerN_*` target. Bundled clients
        /// include this when changing a route target so an intervening stack
        /// edit cannot silently retarget the route to a different layer.
        #[serde(default)]
        target_layer_id: Option<String>,
        /// Stack generation observed when `target_layer_id` was captured.
        /// The stable ID remains authoritative across harmless index drift;
        /// this revision is diagnostic/precondition metadata for receivers.
        #[serde(default)]
        layer_stack_revision: Option<u64>,
        param: String,
        value: serde_json::Value,
    },
    /// Set an audio input parameter (`enabled`, `gain`, `device`,
    /// `band_count`, or the atomic `band_edges` layout object).
    #[serde(rename = "set_audio")]
    SetAudio {
        param: String,
        value: serde_json::Value,
    },
    /// Set a MIDI parameter ("enabled" | "learn" | "cc0".."cc3")
    #[serde(rename = "set_midi")]
    SetMidi {
        param: String,
        value: serde_json::Value,
    },
    /// Bounded, path-free typed controller-profile document transfer. Large
    /// documents use the dedicated authenticated HTTP endpoint; the same shape
    /// remains valid over WebSocket when it fits the general message cap.
    #[serde(rename = "controller_profile")]
    ControllerProfile {
        request: crate::controller_profile::ControllerProfileAction,
    },
    /// Phone orientation sample (degrees, DeviceOrientation convention)
    #[serde(rename = "gyro")]
    Gyro { alpha: f32, beta: f32, gamma: f32 },
    /// Declare this WebSocket as an enabled/disabled phone sensor streamer.
    /// Older clients remain compatible: their first gyro sample implicitly
    /// enables their connection until it disconnects.
    #[serde(rename = "gyro_stream")]
    GyroStream { enabled: bool },
    /// Store the latest raw orientation as the centered (0.5) pose.
    #[serde(rename = "gyro_calibrate")]
    GyroCalibrate,
    /// Set one gyro axis response parameter (range, expo, invert).
    #[serde(rename = "set_gyro_config")]
    SetGyroConfig {
        axis: String,
        param: String,
        value: serde_json::Value,
    },
    /// XY performance pad position (0..1 each)
    #[serde(rename = "pad")]
    Pad {
        x: f32,
        y: f32,
        /// True while a pointer owns the pad; false starts spring return.
        #[serde(default = "bool_true")]
        active: bool,
    },
    /// Set one pad response or spring parameter.
    #[serde(rename = "set_pad_config")]
    SetPadConfig {
        axis: String,
        param: String,
        value: serde_json::Value,
    },
    /// One phone/browser gesture-field sample.
    ///
    /// This is a stream event, not an absolute value, so it deliberately has no
    /// `coalesce_key`: replacing an older pending sample would silently delete
    /// path points and the recorded track would then differ from what the
    /// operator actually drew. `phase` and `mode` cross the wire as the engine's
    /// own closed vocabularies, so an unknown token is a deserialization error
    /// rather than a silently defaulted value.
    #[serde(rename = "gesture_sample")]
    GestureSample {
        stroke: u8,
        phase: crate::gesture::GesturePhase,
        mode: crate::gesture::GestureMode,
        x: f32,
        y: f32,
        /// Absent means full contact, matching a surface with no pressure axis.
        #[serde(default = "unit_pressure")]
        pressure: f32,
        /// Absent means an unset direction, which is inert rather than an
        /// invented axis.
        #[serde(default)]
        direction_x: f32,
        #[serde(default)]
        direction_y: f32,
    },
    /// Arm or disarm gesture recording.
    ///
    /// The honesty law made explicit: while this is off a live gesture affects
    /// the current session only and nothing is added to the replayable track.
    /// It is an ordered barrier so a sample can never cross an arm/disarm edge
    /// and land in the wrong recording, and it is protected by the current
    /// layer-stack revision so an arm decision taken against one program can
    /// never arrive after a patch load has replaced it.
    #[serde(rename = "set_gesture_recording")]
    SetGestureRecording {
        enabled: bool,
        layer_stack_revision: u64,
    },
    /// Coalescible bounded gesture-canvas value edit.
    ///
    /// Unlike the recording barrier this is an ordinary absolute scalar: the
    /// newest value for one control is the only one worth applying. It reaches
    /// authored canvas controls only and has no path to the recorded track.
    #[serde(rename = "set_gesture_canvas")]
    SetGestureCanvas {
        param: String,
        value: serde_json::Value,
    },
    /// Arm or disarm B9 performance recording.
    ///
    /// An ordered barrier on the `set_gesture_recording` law: it is never
    /// coalesced, never latched, and carries the layer-stack revision so an
    /// arm decision taken against one program can never arrive after a patch
    /// load has replaced it. Arming starts a fresh take at tick zero.
    #[serde(rename = "set_performance_recording")]
    SetPerformanceRecording {
        enabled: bool,
        layer_stack_revision: u64,
    },
    /// Arm, retune, or disarm B9 performance playback.
    ///
    /// The same barrier law as the recording control. Compilation resolves
    /// every recorded address against the current program exactly once at
    /// arm; a repeated arm while playing only updates the loop flag.
    #[serde(rename = "set_performance_playback")]
    SetPerformancePlayback {
        enabled: bool,
        #[serde(default)]
        loop_playback: bool,
        layer_stack_revision: u64,
    },
    /// Discard the loaded take and stop both performance transports.
    #[serde(rename = "clear_performance_take")]
    ClearPerformanceTake,
    /// Set fullscreen audience output explicitly. Current clients use this
    /// idempotent command so a delayed/retried packet cannot invert the
    /// performer's requested state.
    #[serde(rename = "set_output_window")]
    SetOutputWindow { enabled: bool },
    /// Select the physical display used by legacy fullscreen Output. Empty is
    /// Automatic; all non-empty IDs come from `output_displays` and are
    /// revalidated against the retained winit inventory before any window
    /// moves. Current clients echo the inventory generation; older clients
    /// omit it and retain the ID-membership barrier.
    #[serde(rename = "set_output_display")]
    SetOutputDisplay {
        display_id: String,
        #[serde(default)]
        inventory_generation: Option<u64>,
    },
    /// Explicitly refresh the monitor inventory shared by legacy Output and
    /// StageMap. This is a host operation and never enters creative history.
    #[serde(rename = "rescan_output_displays")]
    RescanOutputDisplays,
    /// Legacy open/close command retained for older control panels.
    #[serde(rename = "toggle_output_window")]
    ToggleOutputWindow,
    /// Capture current parameters into morph slot "a" or "b"
    #[serde(rename = "morph_capture")]
    MorphCapture {
        slot: String,
        /// Optional stack generation supplied by current clients. A capture
        /// queued against an older topology is rejected at execution time.
        #[serde(default)]
        stack_revision: Option<u64>,
        /// Independent creative-topology generation. Legacy clients omit it;
        /// current clients supply both barriers so capture cannot combine a
        /// new stack with stale rack/group ownership.
        #[serde(default)]
        composition_revision: Option<u64>,
    },
    /// B15 snapshot bank: store the current rig in one of eight slots.
    /// Carries the same two revision barriers a Morph capture does, because it
    /// captures the same thing and would otherwise attach positional slot data
    /// to a different stack.
    #[serde(rename = "snapshot_bank_save")]
    SnapshotBankSave {
        slot: usize,
        #[serde(default)]
        stack_revision: Option<u64>,
        #[serde(default)]
        composition_revision: Option<u64>,
    },
    /// B15 snapshot bank: recall a slot by loading it into the Morph A/B pair
    /// and gliding. Revision-barriered for the same reason as a save — the
    /// recall captures the current rig into A before it travels.
    #[serde(rename = "snapshot_bank_recall")]
    SnapshotBankRecall {
        slot: usize,
        #[serde(default)]
        stack_revision: Option<u64>,
        #[serde(default)]
        composition_revision: Option<u64>,
    },
    /// B15 snapshot bank: empty one slot.
    #[serde(rename = "snapshot_bank_clear")]
    SnapshotBankClear { slot: usize },
    /// B15 snapshot bank: how long a recall takes to travel, in beats.
    #[serde(rename = "set_snapshot_bank_glide")]
    SetSnapshotBankGlide { beats: f64 },
    /// Clear both morph slots (crossfader disengages)
    #[serde(rename = "morph_clear")]
    MorphClear,
    /// Set the morph crossfader position (0 = A, 1 = B)
    #[serde(rename = "set_morph")]
    SetMorph { value: f32 },
    /// Select linear or equal-power interpolation.
    #[serde(rename = "set_morph_law")]
    SetMorphLaw { law: String },
    /// Begin an explicit beat-duration glide to A or B.
    #[serde(rename = "morph_glide")]
    MorphGlide { target: f32, duration_beats: f64 },
    /// Rescan the library folder (pushed internally after an upload)
    #[serde(rename = "rescan_library")]
    RescanLibrary,
    /// Cut the output to black (toggle)
    #[serde(rename = "toggle_blackout")]
    ToggleBlackout,
    /// Set a temporal effect parameter
    #[serde(rename = "set_temporal")]
    SetTemporal {
        param: String,
        value: serde_json::Value,
    },
    /// Clear clean-history/Garden carrier/Score memory at one ordered engine
    /// boundary. Authored temporal parameters and a paused audience hold stay.
    #[serde(rename = "clear_temporal_memory")]
    ClearTemporalMemory,
    /// Emit one explicit manual Collision Score event. It is an ordered event,
    /// never a coalesced scalar edit.
    #[serde(rename = "trigger_collision_score")]
    TriggerCollisionScore,
    /// Emit one explicit all-pixel Refresh Garden admission. Like Score
    /// events, this is counted, ordered, and never coalesced into a scalar.
    #[serde(rename = "trigger_refresh_garden")]
    TriggerRefreshGarden,
    /// Select or clear the stable current-frame layer route used by the
    /// Refresh Garden Matte gate. This is an ordered topology barrier.
    #[serde(rename = "set_refresh_garden_matte_route")]
    SetRefreshGardenMatteRoute {
        #[serde(default)]
        layer_id: Option<String>,
        #[serde(default)]
        stage: LayerImageStage,
        layer_stack_revision: u64,
    },
    /// Select or clear the stable admitted-motion route used by the Refresh
    /// Garden Motion gate. This is an ordered topology barrier.
    #[serde(rename = "set_refresh_garden_motion_route")]
    SetRefreshGardenMotionRoute {
        #[serde(default)]
        layer_id: Option<String>,
        layer_stack_revision: u64,
    },
    /// Clear only the portable explicit-event recording. Authored controls and
    /// live Temporal/Garden/Score memory are unchanged.
    #[serde(rename = "clear_temporal_event_track")]
    ClearTemporalEventTrack,
    /// Coalescible bounded Motion value/tier edit. Donor identity uses the
    /// separate ordered action below and algorithm provenance is read-only.
    #[serde(rename = "set_motion")]
    SetMotion {
        scope: MotionScopeSnapshot,
        param: String,
        value: serde_json::Value,
    },
    /// Select or clear one layer recipient's stable donor. This is a topology
    /// barrier protected by the current layer-stack revision.
    #[serde(rename = "set_motion_donor")]
    SetMotionDonor {
        layer_id: String,
        #[serde(default)]
        donor_layer_id: Option<String>,
        layer_stack_revision: u64,
    },
    /// Select or clear one of a Field Collider's two fixed inputs.
    ///
    /// Rewiring an input rewrites the collider's motion-field request, so this
    /// is an ordered, revision-protected, uncoalesced barrier exactly like
    /// [`WebAction::SetMotionDonor`], and it is refused inside a `Quantized`
    /// batch. The addressed slot travels as a closed named token rather than an
    /// index, so an unknown slot is a deserialization rejection instead of a
    /// positional fallback onto the partner input. `enabled`, `mode`, and
    /// `boundary` are values and travel on the ordinary coalescible
    /// [`WebAction::SetMotion`].
    #[serde(rename = "set_motion_collider_input")]
    SetMotionColliderInput {
        layer_id: String,
        input: FieldColliderInputSnapshot,
        #[serde(default)]
        donor_layer_id: Option<String>,
        layer_stack_revision: u64,
    },
    /// Clear all lattice/transplant carrier history at one ordered engine
    /// boundary while leaving authored Motion values untouched.
    #[serde(rename = "clear_motion_memory")]
    ClearMotionMemory,
    /// Ask the engine to encode (and, on completion, hot-adopt) a proxy for
    /// one layer's verified content identity — the browser twin of the
    /// native Y key. The stable ID is mandatory and authoritative: there is
    /// no positional fallback, and a vanished ID is a safe no-op. Every
    /// refusal the Y key enforces (no verified identity, not a video,
    /// already backed, busy worker, unavailable cache) is answered by the
    /// same engine function and surfaces in the layer's `proxy_note`, so
    /// the browser cannot bypass the content-identity ladder. Priority,
    /// never coalesced, never quantized — a request is an event, not an
    /// absolute value.
    #[serde(rename = "request_layer_proxy")]
    RequestLayerProxy { layer_id: String },
    /// Enable/disable the Spout output sender
    #[serde(rename = "set_spout")]
    SetSpout { enabled: bool },
    /// Select sender dimensions without resizing the live renderer. Main
    /// accepts only the closed `native` / `1080p` vocabulary.
    #[serde(rename = "set_spout_resolution")]
    SetSpoutResolution { resolution: String },
    /// Begin a real-time program recording. Main owns the native destination
    /// picker; browser payloads can never nominate a host filesystem path.
    #[serde(rename = "start_program_recording")]
    StartProgramRecording {
        #[serde(default)]
        auto_import: bool,
    },
    /// Drain the bounded recorder queue and close at an explicit CFR clock.
    #[serde(rename = "finish_program_recording")]
    FinishProgramRecording,
    /// Abort capture and remove every temporary artifact.
    #[serde(rename = "cancel_program_recording")]
    CancelProgramRecording,
    /// Publish one complete PNG through the same no-replace artifact law.
    #[serde(rename = "capture_still")]
    CaptureStill {
        #[serde(flatten)]
        target: CaptureTargetSnapshot,
        #[serde(default)]
        auto_import: bool,
    },
    /// Begin recording one stable scope. A successful commit emits an intent
    /// for a newly allocated ClipSlot; it never mutates the live source itself.
    #[serde(rename = "start_resample")]
    StartResample {
        #[serde(flatten)]
        target: CaptureTargetSnapshot,
        destination_layer_id: String,
        #[serde(default)]
        activate: bool,
    },
    /// Show/hide operator health telemetry in the editor preview only.
    #[serde(rename = "set_stage_health_hud")]
    SetStageHealthHud { enabled: bool },
    /// B11: show/hide the monitoring-bay overlay in the editor preview only.
    #[serde(rename = "set_monitor_bay")]
    SetMonitorBay { enabled: bool },
    /// B11: select the monitoring bay's PROBE signal. The token rides the
    /// closed `MonitorProbe` vocabulary; an unknown token is refused at the
    /// gate and again at the applier through the same parse table.
    #[serde(rename = "set_monitor_probe")]
    SetMonitorProbe { probe: String },
    /// B11: per-client declaration that this panel is currently watching the
    /// bay. Handled at the socket layer exactly like `gyro_stream` — never
    /// queued, cleaned up on disconnect, and expired by the watch timeout so
    /// a vanished tab cannot keep the readback armed.
    #[serde(rename = "monitor_watch")]
    MonitorWatch { enabled: bool },
    /// Substitute a test card only on one exact physical output endpoint.
    #[serde(rename = "set_stage_test_card")]
    SetStageTestCard {
        mode: crate::stage_map::TestCardMode,
        #[serde(default)]
        output_endpoint_id: Option<String>,
    },
    /// Overlay endpoint identity only on one exact physical output endpoint.
    #[serde(rename = "set_output_identification")]
    SetOutputIdentification {
        enabled: bool,
        #[serde(default)]
        output_endpoint_id: Option<String>,
    },
    /// Open one coalesced manual scalar gesture. IDs are non-zero JS-safe
    /// integers and boundaries remain ordered queue barriers.
    #[serde(rename = "begin_history_gesture")]
    BeginHistoryGesture { gesture_id: u64 },
    /// Commit the one before-checkpoint after the final scalar edit.
    #[serde(rename = "end_history_gesture")]
    EndHistoryGesture { gesture_id: u64 },
    /// Discard an unchanged/interrupted gesture boundary. A client that has
    /// already emitted a value must end rather than cancel it.
    #[serde(rename = "cancel_history_gesture")]
    CancelHistoryGesture { gesture_id: u64 },
    #[serde(rename = "undo_manual")]
    UndoManual,
    #[serde(rename = "redo_manual")]
    RedoManual,
    /// Capture values from one exact stable scope into the preset library.
    #[serde(rename = "capture_scoped_preset")]
    CaptureScopedPreset {
        name: String,
        kind: crate::preset::PresetKind,
        target: PresetTargetSnapshot,
        preset_revision: u64,
        layer_stack_revision: u64,
        composition_revision: u64,
    },
    /// Apply a values-only preset. Main resolves the ID, checks the target
    /// kind/topology, performs unified preflight, then records one history
    /// transaction; no route or stable identity may be replaced.
    #[serde(rename = "apply_scoped_preset")]
    ApplyScopedPreset {
        preset_id: String,
        target: PresetTargetSnapshot,
        preset_revision: u64,
        layer_stack_revision: u64,
        composition_revision: u64,
    },
    #[serde(rename = "delete_scoped_preset")]
    DeleteScopedPreset {
        preset_id: String,
        preset_revision: u64,
    },
    /// Recovery is always an explicit operator choice; startup never recalls
    /// a journal automatically.
    #[serde(rename = "restore_recovery_journal")]
    RestoreRecoveryJournal,
    #[serde(rename = "discard_recovery_journal")]
    DiscardRecoveryJournal,
    /// Open a host-native picker and replace the complete performance state.
    #[serde(rename = "open_patch_snapshot")]
    OpenPatchSnapshot,
    /// Open a host-native picker and transfer only visual state onto the
    /// current positional stack. The revision protects that mapping.
    #[serde(rename = "open_patch_look")]
    OpenPatchLook { stack_revision: u64 },
    /// Capture the complete current performance state into the local patch
    /// corpus without opening a native file dialog.
    #[serde(rename = "quick_save_patch")]
    QuickSavePatch,
    /// Start an offline render export
    #[serde(rename = "start_export")]
    StartExport {
        width: u32,
        height: u32,
        fps: u32,
        duration_secs: f32,
        /// Defaults to the established half-resolution live look for legacy
        /// clients. Native processes VHS at the selected export dimensions.
        #[serde(default)]
        ntsc_quality: crate::ntsc::NtscExportQuality,
        /// Explicit offline Curved Shutter sample request. Omitted legacy
        /// clients preserve each scope's authored live tier.
        #[serde(default)]
        shutter_samples: crate::render_export::ExportShutterSamples,
        #[serde(default)]
        audio_layer: Option<usize>,
        #[serde(default)]
        audio_layer_id: Option<String>,
    },
    /// Cancel a running export
    #[serde(rename = "cancel_export")]
    CancelExport,
}

fn bool_true() -> bool {
    true
}

/// A surface with no pressure axis reports full contact.
fn unit_pressure() -> f32 {
    1.0
}

impl WebAction {
    fn creative_scope_key(scope: &CreativeScopeSnapshot) -> String {
        match scope {
            CreativeScopeSnapshot::Master => "master".to_string(),
            CreativeScopeSnapshot::Layer { layer_id } => format!("layer:{layer_id}"),
            CreativeScopeSnapshot::Group { group_id } => format!("group:{group_id}"),
        }
    }

    fn layer_key(index: usize, layer_id: &Option<String>) -> String {
        layer_id
            .as_deref()
            .filter(|id| !id.is_empty())
            .map_or_else(|| format!("index:{index}"), |id| format!("id:{id}"))
    }

    fn routing_key(index: usize, route_id: &Option<String>) -> String {
        route_id
            .as_deref()
            .filter(|id| !id.is_empty())
            .map_or_else(|| format!("index:{index}"), |id| format!("id:{id}"))
    }

    /// Absolute controls may replace an older pending value with the same
    /// semantic destination. The replacement is moved to the queue tail so a
    /// later value never jumps ahead of an intervening reset or topology edit.
    fn coalesce_key(&self) -> Option<String> {
        match self {
            Self::Quantized { inner } => inner.coalesce_key().map(|key| format!("quantized:{key}")),
            Self::SetParam { param, .. } => Some(format!("master:{param}")),
            Self::SetMasterTransform { param, .. } => Some(format!("master:transform:{param}")),
            Self::SetVisualNodeParam {
                scope,
                node_id,
                param,
                ..
            } => Some(format!(
                "creative:{}:node:{node_id}:{param}",
                Self::creative_scope_key(scope)
            )),
            Self::SetVisualNodeStudyDocument { scope, node_id, .. } => Some(format!(
                "creative:{}:node:{node_id}:study_document",
                Self::creative_scope_key(scope)
            )),
            Self::SetCompositionGroupParam {
                group_id, param, ..
            } => Some(format!("creative:group:{group_id}:{param}")),
            Self::SetCompositionGroupMatteParam {
                group_id, param, ..
            } => Some(format!("creative:group:{group_id}:matte:{param}")),
            Self::SetCompositionBusCrossfade { .. } => {
                Some("creative:composition:bus-crossfade".into())
            }
            Self::SetCompositionBusMixParam { param, .. } => {
                Some(format!("creative:composition:bus-mix:{param}"))
            }
            Self::SetLayerParam {
                index,
                layer_id,
                param,
                ..
            } => Some(format!(
                "layer:{}:param:{param}",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetLayerEffect {
                index,
                layer_id,
                param,
                ..
            } => Some(format!(
                "layer:{}:effect:{param}",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetLayerTransform {
                index,
                layer_id,
                param,
                ..
            } => Some(format!(
                "layer:{}:transform:{param}",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetLayerVisibility {
                index, layer_id, ..
            } => Some(format!(
                "layer:{}:visible",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetLayerPaused {
                index, layer_id, ..
            } => Some(format!(
                "layer:{}:paused",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetClipTransport {
                layer_id,
                slot_id,
                param,
                ..
            } => Some(format!(
                "layer:id:{layer_id}:slot:{}:transport:{param}",
                slot_id.get()
            )),
            Self::SetClipCue {
                layer_id,
                slot_id,
                cue_id,
                ..
            } => Some(format!(
                "layer:id:{layer_id}:slot:{}:cue:{}",
                slot_id.get(),
                cue_id.get()
            )),
            Self::SetLayerMatteParam {
                layer_id, param, ..
            } if matches!(param.as_str(), "amount" | "threshold" | "softness") => {
                Some(format!("layer:id:{layer_id}:matte:{param}"))
            }
            Self::SetLayerPattern {
                index,
                layer_id,
                param,
                ..
            } => Some(format!(
                "layer:{}:pattern:{param}",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetLayerText {
                index,
                layer_id,
                param,
                ..
            } => Some(format!(
                "layer:{}:text:{param}",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetMasterPaused { .. } => Some("master:paused".into()),
            Self::SetProgramFrozen { .. } => Some("master:paused".into()),
            Self::SetMediaFrozen { .. } => Some("media:frozen".into()),
            Self::SetMediaSafetyMode { .. } => Some("media:safety-mode".into()),
            Self::SetNewLayerFit { .. } => Some("host:new-layer-fit".into()),
            Self::SetProxySettings { .. } => Some("host:proxy-settings".into()),
            Self::SetLayerRerollOnLoop {
                index, layer_id, ..
            } => Some(format!(
                "layer:{}:reroll-on-loop",
                Self::layer_key(*index, layer_id)
            )),
            Self::SetBlackout { .. } => Some("master:blackout".into()),
            Self::SetOutputWindow { .. } => Some("output:enabled".into()),
            Self::SetOutputDisplay { .. } => Some("output:display".into()),
            Self::SetNtscParam { param, .. } => Some(format!("ntsc:{param}")),
            Self::SetBpm { .. } => Some("mod:bpm".into()),
            Self::SetLfo { index, param, .. } => Some(format!("lfo:{index}:{param}")),
            Self::SetEnvelope { index, param, .. } => Some(format!("env:{index}:{param}")),
            Self::SetMacro { index, .. } => Some(format!("macro:{index}")),
            Self::SetModSeed { .. } => Some("mod:seed".into()),
            Self::SetRouting {
                index,
                route_id,
                param,
                ..
            } => Some(format!(
                "routing:{}:{param}",
                Self::routing_key(*index, route_id)
            )),
            Self::SetAudio { param, .. } => Some(format!("audio:{param}")),
            Self::SetMidi { param, .. } => Some(format!("midi:{param}")),
            Self::Gyro { .. } => Some("gyro:sample".into()),
            Self::SetGyroConfig { axis, param, .. } => Some(format!("gyro:{axis}:{param}")),
            Self::Pad { .. } => Some("pad:position".into()),
            Self::SetPadConfig { axis, param, .. } => Some(format!("pad:{axis}:{param}")),
            Self::SetSnapshotBankGlide { .. } => Some("snapshot-bank:glide".into()),
            Self::SetMorph { .. } => Some("morph:position".into()),
            Self::SetMorphLaw { .. } => Some("morph:law".into()),
            Self::SetTemporal { param, .. } => Some(format!("temporal:{param}")),
            // An authored canvas scalar coalesces per control. The recording
            // barrier deliberately gets no arm here: `_ => None` is what stops
            // a later absolute value from jumping an arm/disarm edge.
            Self::SetGestureCanvas { param, .. } => Some(format!("gesture:canvas:{param}")),
            Self::SetMotion { scope, param, .. } => Some(match scope {
                MotionScopeSnapshot::Master => format!("motion:master:{param}"),
                MotionScopeSnapshot::Layer { layer_id } => {
                    format!("motion:layer:{layer_id}:{param}")
                }
            }),
            Self::SetSpout { .. } => Some("spout:enabled".into()),
            Self::SetSpoutResolution { .. } => Some("spout:resolution".into()),
            Self::SetStageHealthHud { .. } => Some("stage:health-hud".into()),
            Self::SetMonitorBay { .. } => Some("stage:monitor-bay".into()),
            Self::SetMonitorProbe { .. } => Some("stage:monitor-probe".into()),
            Self::SetStageTestCard { .. } => Some("stage:test-card".into()),
            Self::SetOutputIdentification { .. } => Some("stage:output-identification".into()),
            _ => None,
        }
    }

    /// These live/performance commands deliberately bypass manual history in
    /// Main. The WebSocket owner gate uses the same frozen classification so
    /// telemetry, triggers, and output operations cannot be mistaken for a
    /// second controller's authored scalar edit.
    fn is_performance_only_for_history(&self) -> bool {
        matches!(
            self,
            Self::Quantized { .. }
                | Self::ActivateClipSlot { .. }
                | Self::TriggerClipCue { .. }
                | Self::SeekClipSlot { .. }
                | Self::SeekClipSlotTimecode { .. }
                | Self::PrepareScene { .. }
                | Self::TriggerScene { .. }
                | Self::AutopilotPlay
                | Self::AutopilotPause
                | Self::AutopilotReset
                | Self::TapTempo
                | Self::Gyro { .. }
                | Self::GyroStream { .. }
                | Self::GyroCalibrate
                | Self::Pad { .. }
                | Self::BendPad { .. }
                | Self::GestureSample { .. }
                | Self::SetGestureRecording { .. }
                | Self::SetPerformanceRecording { .. }
                | Self::SetPerformancePlayback { .. }
                | Self::ClearPerformanceTake
                | Self::TriggerCollisionScore
                | Self::TriggerRefreshGarden
                | Self::ClearTemporalEventTrack
                | Self::ClearTemporalMemory
                | Self::ClearMotionMemory
                | Self::SetOutputWindow { .. }
                | Self::SetOutputDisplay { .. }
                | Self::RescanOutputDisplays
                | Self::ToggleOutputWindow
                | Self::SetSpout { .. }
                | Self::SetSpoutResolution { .. }
                | Self::StartProgramRecording { .. }
                | Self::FinishProgramRecording
                | Self::CancelProgramRecording
                | Self::CaptureStill { .. }
                | Self::StartResample { .. }
                | Self::SetStageHealthHud { .. }
                | Self::SetMonitorBay { .. }
                | Self::SetMonitorProbe { .. }
                | Self::MonitorWatch { .. }
                | Self::SetStageTestCard { .. }
                | Self::SetOutputIdentification { .. }
                | Self::RescanLibrary
                | Self::QuickSavePatch
                | Self::StartExport { .. }
                | Self::CancelExport
                | Self::ControllerProfile {
                    request: crate::controller_profile::ControllerProfileAction::Export {},
                }
        )
    }

    fn is_priority(&self) -> bool {
        match self {
            Self::CancelExport
            | Self::StartProgramRecording { .. }
            | Self::FinishProgramRecording
            | Self::CancelProgramRecording
            | Self::CaptureStill { .. }
            | Self::StartResample { .. }
            | Self::SetStageTestCard { .. }
            | Self::SetOutputIdentification { .. }
            | Self::BeginHistoryGesture { .. }
            | Self::EndHistoryGesture { .. }
            | Self::CancelHistoryGesture { .. }
            | Self::UndoManual
            | Self::RedoManual
            | Self::CaptureScopedPreset { .. }
            | Self::ApplyScopedPreset { .. }
            | Self::DeleteScopedPreset { .. }
            | Self::RestoreRecoveryJournal
            | Self::DiscardRecoveryJournal
            | Self::ControllerProfile { .. }
            | Self::ToggleBlackout
            | Self::SetBlackout { .. }
            | Self::SetOutputWindow { .. }
            | Self::SetOutputDisplay { .. }
            | Self::RescanOutputDisplays
            | Self::ToggleOutputWindow
            | Self::SetMasterPaused { .. }
            | Self::SetProgramFrozen { .. }
            | Self::SetMediaFrozen { .. }
            | Self::Reroll { .. }
            | Self::SetLayerRerollOnLoop { .. }
            | Self::ResetFx
            | Self::ResetVisualProgram
            | Self::ClearTemporalMemory
            | Self::TriggerCollisionScore
            | Self::TriggerRefreshGarden
            | Self::SetRefreshGardenMatteRoute { .. }
            | Self::SetRefreshGardenMotionRoute { .. }
            | Self::ClearTemporalEventTrack
            | Self::SetGestureRecording { .. }
            | Self::SetPerformanceRecording { .. }
            | Self::SetPerformancePlayback { .. }
            | Self::ClearPerformanceTake
            | Self::SetMotionDonor { .. }
            | Self::SetMotionColliderInput { .. }
            | Self::RequestLayerProxy { .. }
            | Self::ClearMotionMemory
            | Self::SetLayerPaused { .. }
            | Self::SetLayerVisibility { .. }
            | Self::LoadClipIntoSlot { .. }
            | Self::RemoveClipSlot { .. }
            | Self::ActivateClipSlot { .. }
            | Self::RemoveClipCue { .. }
            | Self::TriggerClipCue { .. }
            | Self::SeekClipSlot { .. }
            | Self::SeekClipSlotTimecode { .. }
            | Self::PrepareScene { .. }
            | Self::CaptureScene { .. }
            | Self::RemoveScene { .. }
            | Self::TriggerScene { .. }
            | Self::ReplaceAutopilotPlan { .. }
            | Self::AutopilotPlay
            | Self::AutopilotPause
            | Self::AutopilotReset
            | Self::SetLayerMatteInput { .. }
            | Self::InsertVisualNode { .. }
            | Self::RemoveVisualNode { .. }
            | Self::MoveVisualNode { .. }
            | Self::SetVisualNodeMaskVariant { .. }
            | Self::SetVisualNodeRoute { .. }
            | Self::SetVisualNodeDisplaceRoute { .. }
            | Self::SetVisualNodeSymmetryRoute { .. }
            | Self::SetVisualNodeResidualRoute { .. }
            | Self::SetCompositionGroupMatteRoute { .. }
            | Self::CreateCompositionGroup { .. }
            | Self::RemoveCompositionGroup { .. }
            | Self::SetCompositionGroupMembers { .. }
            | Self::MoveCompositionRootItem { .. }
            | Self::SetCompositionLayerBus { .. }
            | Self::OpenPatchSnapshot
            | Self::OpenPatchLook { .. }
            | Self::QuickSavePatch
            | Self::RescanLibrary => true,
            Self::SetLayerMatteParam { param, .. }
                if !matches!(param.as_str(), "amount" | "threshold" | "softness") =>
            {
                true
            }
            Self::SetMediaSafetyMode {
                mode: crate::media_safety::MediaSafetyMode::Safe,
            } => true,
            Self::Pad { active, .. } => !active,
            // A bend edge pair carries no interpolated middle: a dropped
            // release latches the pad on, and a dropped press fires nothing
            // the operator played, so both edges hold an admission reservation.
            Self::BendPad { .. } => true,
            // A dropped `Begin` orphans every later point of that stroke and a
            // dropped `End` leaves it permanently open, so both edges hold an
            // admission reservation. An intermediate `Move` is ordinary: under
            // saturation the queue sheds a path point, and the track then
            // honestly records what the host accepted.
            Self::GestureSample { phase, .. } => {
                !matches!(phase, crate::gesture::GesturePhase::Move)
            }
            Self::Quantized { inner } => inner.is_priority(),
            _ => false,
        }
    }
}

trait QueuedWebAction {
    fn web_action(&self) -> &WebAction;
}

impl QueuedWebAction for WebAction {
    fn web_action(&self) -> &WebAction {
        self
    }
}

impl QueuedWebAction for ActionEnvelope<WebAction> {
    fn web_action(&self) -> &WebAction {
        self.payload()
    }
}

struct EnqueueReport<T> {
    outcome: EnqueueOutcome,
    terminal: Option<(T, ActionDisposition)>,
}

fn enqueue_bounded_detailed<T: QueuedWebAction>(queue: &mut Vec<T>, action: T) -> EnqueueReport<T> {
    // A rescan has no payload. Keep the earliest pending barrier in place so
    // later uploads cannot move it behind a clip-selection action.
    if matches!(
        action.web_action(),
        WebAction::RescanLibrary | WebAction::RescanOutputDisplays
    ) && queue.iter().any(|candidate| {
        std::mem::discriminant(candidate.web_action())
            == std::mem::discriminant(action.web_action())
    }) {
        return EnqueueReport {
            outcome: EnqueueOutcome::Coalesced,
            terminal: Some((action, ActionDisposition::Coalesced)),
        };
    }
    let seek_target = match action.web_action() {
        WebAction::SeekClipSlot {
            layer_id, slot_id, ..
        }
        | WebAction::SeekClipSlotTimecode {
            layer_id, slot_id, ..
        } => Some((layer_id, slot_id)),
        _ => None,
    };
    if let Some((layer_id, slot_id)) = seek_target {
        let replaces_last = match queue.last() {
            Some(candidate) => match candidate.web_action() {
                WebAction::SeekClipSlot {
                    layer_id: pending_layer,
                    slot_id: pending_slot,
                    ..
                }
                | WebAction::SeekClipSlotTimecode {
                    layer_id: pending_layer,
                    slot_id: pending_slot,
                    ..
                } => pending_layer == layer_id && pending_slot == slot_id,
                _ => false,
            },
            None => false,
        };
        if replaces_last {
            let prior = std::mem::replace(
                queue.last_mut().expect("the matching final seek exists"),
                action,
            );
            return EnqueueReport {
                outcome: EnqueueOutcome::Coalesced,
                terminal: Some((prior, ActionDisposition::Coalesced)),
            };
        }
    }
    if let Some(key) = action.web_action().coalesce_key() {
        // Captures, resets, saves, topology changes, and other uncoalesced
        // commands observe ordered state. Never move a later absolute value
        // across one of those semantic barriers.
        let barrier = queue
            .iter()
            .rposition(|candidate| candidate.web_action().coalesce_key().is_none())
            .map_or(0, |position| position + 1);
        if let Some(position) = queue[barrier..]
            .iter()
            .rposition(|candidate| {
                candidate.web_action().coalesce_key().as_deref() == Some(key.as_str())
            })
            .map(|position| barrier + position)
        {
            let prior = queue.remove(position);
            queue.push(action);
            return EnqueueReport {
                outcome: EnqueueOutcome::Coalesced,
                terminal: Some((prior, ActionDisposition::Coalesced)),
            };
        }
    }

    if action.web_action().is_priority() {
        if queue.len() >= MAX_PENDING_ACTIONS {
            let position = queue
                .iter()
                .position(|candidate| !candidate.web_action().is_priority())
                .unwrap_or(0);
            let prior = queue.remove(position);
            queue.push(action);
            return EnqueueReport {
                outcome: EnqueueOutcome::Added,
                terminal: Some((prior, ActionDisposition::Superseded)),
            };
        }
    } else if queue.len() >= MAX_PENDING_ACTIONS - PRIORITY_ACTION_RESERVE {
        return EnqueueReport {
            outcome: EnqueueOutcome::Dropped,
            terminal: Some((action, ActionDisposition::Refused)),
        };
    }

    queue.push(action);
    EnqueueReport {
        outcome: EnqueueOutcome::Added,
        terminal: None,
    }
}

#[cfg(test)]
fn enqueue_bounded(queue: &mut Vec<WebAction>, action: WebAction) -> EnqueueOutcome {
    enqueue_bounded_detailed(queue, action).outcome
}

fn web_action_source(action: &WebAction) -> ActionSourceClass {
    match action {
        WebAction::Quantized { inner } => web_action_source(inner),
        WebAction::Gyro { .. }
        | WebAction::GyroStream { .. }
        | WebAction::Pad { .. }
        | WebAction::GestureSample { .. }
        | WebAction::BendPad { .. } => ActionSourceClass::Phone,
        _ => ActionSourceClass::Browser,
    }
}

impl EffectsSnapshot {
    pub fn from_uniforms(u: &EffectUniforms) -> Self {
        Self {
            random_seed: u.random_seed,
            pixelate: u.pixelate_size,
            downsample: u.downsample,
            shift_amount: u.shift_amount,
            shift_block_size: u.shift_block_size,
            shift_density: u.shift_density,
            shift_speed: u.shift_speed,
            rgb_split: u.rgb_split,
            hue_shift: u.hue_shift,
            saturation: u.saturation,
            brightness: u.brightness,
            contrast: u.contrast,
            posterize: u.posterize,
            invert: u.invert > 0.5,
            grain_intensity: u.grain_intensity,
            grain_size: u.grain_size,
            grain_algo: u.grain_algo as u32,
            color_grain: u.color_grain > 0.5,
            vignette: u.vignette,
            color_drift: u.color_drift,
            breathe_scale: u.breathe_scale,
            breathe_rotation: u.breathe_rotation,
            breathe_position: u.breathe_position,
            key_mode: u.key_mode.round().clamp(0.0, 4.0) as u32,
            key_color: u.key_color,
            key_threshold: u.key_threshold,
            key_softness: u.key_softness,
            key_tolerance: u.key_tolerance,
            cellular_amount: u.cellular_amount,
            cellular_scale: u.cellular_scale,
            cellular_warp: u.cellular_warp,
            cellular_speed: u.cellular_speed,
            cellular_gap_amount: u.cellular_gap_amount,
            cellular_gap_threshold: u.cellular_gap_threshold,
            cellular_gap_softness: u.cellular_gap_softness,
            contour: u.contour,
            contour_bands: u.contour_bands,
            contour_width: u.contour_width,
            contour_hue: u.contour_hue,
            contour_fill: u.contour_fill,
            flatten: u.flatten,
            flatten_levels: u.flatten_levels,
            contour_dither: u.contour_dither,
            solarize: u.solarize,
            negative: u.negative,
            negative_mode: u.negative_mode.round().clamp(0.0, 2.0) as u32,
            colourpass: u.colourpass,
            colourpass_hue: u.colourpass_hue,
            colourpass_width: u.colourpass_width,
            edge_amount: u.edge_amount,
            edge_hue: u.edge_hue,
            emboss: u.emboss,
            emboss_angle: u.emboss_angle,
            halftone: u.halftone,
            halftone_pitch: u.halftone_pitch,
            halftone_angle: u.halftone_angle,
            moire: u.moire,
            moire_freq: u.moire_freq,
            row_smear: u.row_smear,
            bitcrush: u.bitcrush,
            bitcrush_levels: u.bitcrush_levels,
            bitcrush_dither: u.bitcrush_dither,
            multi_grid_x: u.multi_grid_x,
            multi_grid_y: u.multi_grid_y,
            barrel: u.barrel,
            chroma_aberration: u.chroma_aberration,
            anamorphic_streak: u.anamorphic_streak,
            key_border: u.key_border,
            key_border_color: (u.key_border_color.max(0.0) as u32).min(7),
            key_shadow: u.key_shadow,
        }
    }

    pub fn apply_to_uniforms(&self, u: &mut EffectUniforms) {
        u.random_seed = self.random_seed;
        u.pixelate_size = self.pixelate.clamp(1.0, 32.0);
        u.downsample = self.downsample.clamp(0.05, 1.0);
        u.shift_amount = finite_effect_value(self.shift_amount, 0.0).clamp(0.0, 1.0);
        u.shift_block_size = finite_effect_value(self.shift_block_size, 8.0).clamp(2.0, 256.0);
        u.shift_density = finite_effect_value(self.shift_density, 0.5).clamp(0.0, 1.0);
        u.shift_speed = finite_effect_value(self.shift_speed, 3.0).clamp(0.0, 20.0);
        u.rgb_split = self.rgb_split.clamp(0.0, 30.0);
        u.hue_shift = self.hue_shift.clamp(-180.0, 180.0);
        u.saturation = self.saturation.clamp(-1.0, 1.0);
        u.brightness = self.brightness.clamp(-1.0, 1.0);
        u.contrast = self.contrast.clamp(-1.0, 1.0);
        u.posterize = self.posterize.clamp(0.0, 16.0);
        u.invert = if self.invert { 1.0 } else { 0.0 };
        u.grain_intensity = self.grain_intensity.clamp(0.0, 0.3);
        u.grain_size = self.grain_size.clamp(1.0, 4.0);
        u.grain_algo = (self.grain_algo.min(3)) as f32;
        u.color_grain = if self.color_grain { 1.0 } else { 0.0 };
        u.vignette = self.vignette.clamp(0.0, 1.5);
        u.color_drift = self.color_drift.clamp(0.0, 0.02);
        u.breathe_scale = self.breathe_scale.clamp(0.0, 0.05);
        u.breathe_rotation = self.breathe_rotation.clamp(0.0, 2.0);
        u.breathe_position = self.breathe_position.clamp(0.0, 0.02);
        u.key_mode = self.key_mode.min(4) as f32;
        u.key_color = self.key_color.map(|channel| channel.clamp(0.0, 1.0));
        u.key_threshold = self.key_threshold.clamp(0.0, 1.0);
        u.key_softness = self.key_softness.clamp(0.0, 0.5);
        u.key_tolerance = self.key_tolerance.clamp(0.0, 1.0);
        u.cellular_amount = self.cellular_amount.clamp(0.0, 1.0);
        u.cellular_scale = self.cellular_scale.clamp(2.0, 32.0);
        u.cellular_warp = self.cellular_warp.clamp(0.0, 1.0);
        u.cellular_speed = self.cellular_speed.clamp(0.0, 2.0);
        u.cellular_gap_amount = self.cellular_gap_amount.clamp(0.0, 1.0);
        u.cellular_gap_threshold = self.cellular_gap_threshold.clamp(0.0, 1.0);
        u.cellular_gap_softness = self.cellular_gap_softness.clamp(0.0, 0.5);
        u.contour = finite_effect_value(self.contour, 0.0).clamp(0.0, 1.0);
        u.contour_bands = finite_effect_value(self.contour_bands, 10.0).clamp(2.0, 40.0);
        u.contour_width = finite_effect_value(self.contour_width, 1.2).clamp(0.2, 6.0);
        u.contour_hue = finite_effect_value(self.contour_hue, 0.0).clamp(0.0, 1.0);
        u.contour_fill = finite_effect_value(self.contour_fill, 0.25).clamp(0.0, 1.0);
        u.flatten = finite_effect_value(self.flatten, 0.0).clamp(0.0, 1.0);
        u.flatten_levels = finite_effect_value(self.flatten_levels, 5.0).clamp(2.0, 16.0);
        u.contour_dither = finite_effect_value(self.contour_dither, 0.0).clamp(0.0, 1.0);
        u.solarize = finite_effect_value(self.solarize, 0.0).clamp(0.0, 1.0);
        u.negative = finite_effect_value(self.negative, 0.0).clamp(0.0, 1.0);
        u.negative_mode = self.negative_mode.min(2) as f32;
        u.colourpass = finite_effect_value(self.colourpass, 0.0).clamp(0.0, 1.0);
        u.colourpass_hue = finite_effect_value(self.colourpass_hue, 0.0).clamp(-180.0, 180.0);
        u.colourpass_width = finite_effect_value(self.colourpass_width, 0.25).clamp(0.0, 1.0);
        u.edge_amount = finite_effect_value(self.edge_amount, 0.0).clamp(0.0, 1.0);
        u.edge_hue = finite_effect_value(self.edge_hue, 0.0).clamp(-180.0, 180.0);
        u.emboss = finite_effect_value(self.emboss, 0.0).clamp(0.0, 1.0);
        u.emboss_angle = finite_effect_value(self.emboss_angle, 45.0).clamp(-180.0, 180.0);
        u.halftone = finite_effect_value(self.halftone, 0.0).clamp(0.0, 1.0);
        u.halftone_pitch = finite_effect_value(self.halftone_pitch, 0.4).clamp(0.0, 1.0);
        u.halftone_angle = finite_effect_value(self.halftone_angle, 0.0).clamp(-180.0, 180.0);
        u.moire = finite_effect_value(self.moire, 0.0).clamp(0.0, 1.0);
        u.moire_freq = finite_effect_value(self.moire_freq, 0.4).clamp(0.0, 1.0);
        u.row_smear = finite_effect_value(self.row_smear, 0.0).clamp(0.0, 1.0);
        u.bitcrush = finite_effect_value(self.bitcrush, 0.0).clamp(0.0, 1.0);
        u.bitcrush_levels = finite_effect_value(self.bitcrush_levels, 2.0).clamp(2.0, 16.0);
        u.bitcrush_dither = finite_effect_value(self.bitcrush_dither, 1.0).clamp(0.0, 1.0);
        u.multi_grid_x = finite_effect_value(self.multi_grid_x, 1.0).clamp(1.0, 8.0);
        u.multi_grid_y = finite_effect_value(self.multi_grid_y, 1.0).clamp(1.0, 8.0);
        u.barrel = finite_effect_value(self.barrel, 0.0).clamp(-1.0, 1.0);
        u.chroma_aberration = finite_effect_value(self.chroma_aberration, 0.0).clamp(0.0, 1.0);
        u.anamorphic_streak = finite_effect_value(self.anamorphic_streak, 0.0).clamp(0.0, 1.0);
        u.key_border = finite_effect_value(self.key_border, 0.0).clamp(0.0, 1.0);
        u.key_border_color = self.key_border_color.min(7) as f32;
        u.key_shadow = finite_effect_value(self.key_shadow, 0.0).clamp(0.0, 1.0);
    }

    pub fn apply_param(&mut self, param: &str, value: &serde_json::Value) {
        let v = value;
        match param {
            "random_seed" => {
                if let Some(n) = v.as_u64().and_then(|n| u32::try_from(n).ok()) {
                    self.random_seed = n;
                }
            }
            "pixelate" => {
                if let Some(n) = v.as_f64() {
                    self.pixelate = n as f32;
                }
            }
            "downsample" => {
                if let Some(n) = v.as_f64() {
                    self.downsample = n as f32;
                }
            }
            "shift_amount" => {
                if let Some(n) = v.as_f64() {
                    self.shift_amount = n as f32;
                }
            }
            "shift_block_size" => {
                if let Some(n) = v.as_f64() {
                    self.shift_block_size = n as f32;
                }
            }
            "shift_density" => {
                if let Some(n) = v.as_f64() {
                    self.shift_density = n as f32;
                }
            }
            "shift_speed" => {
                if let Some(n) = v.as_f64() {
                    self.shift_speed = n as f32;
                }
            }
            "rgb_split" => {
                if let Some(n) = v.as_f64() {
                    self.rgb_split = n as f32;
                }
            }
            "hue_shift" => {
                if let Some(n) = v.as_f64() {
                    self.hue_shift = n as f32;
                }
            }
            "saturation" => {
                if let Some(n) = v.as_f64() {
                    self.saturation = n as f32;
                }
            }
            "brightness" => {
                if let Some(n) = v.as_f64() {
                    self.brightness = n as f32;
                }
            }
            "contrast" => {
                if let Some(n) = v.as_f64() {
                    self.contrast = n as f32;
                }
            }
            "posterize" => {
                if let Some(n) = v.as_f64() {
                    self.posterize = n as f32;
                }
            }
            "invert" => {
                if let Some(b) = v.as_bool() {
                    self.invert = b;
                }
            }
            "grain_intensity" => {
                if let Some(n) = v.as_f64() {
                    self.grain_intensity = n as f32;
                }
            }
            "grain_size" => {
                if let Some(n) = v.as_f64() {
                    self.grain_size = n as f32;
                }
            }
            "grain_algo" => {
                if let Some(n) = v.as_u64() {
                    self.grain_algo = n as u32;
                }
            }
            "color_grain" => {
                if let Some(b) = v.as_bool() {
                    self.color_grain = b;
                }
            }
            "vignette" => {
                if let Some(n) = v.as_f64() {
                    self.vignette = n as f32;
                }
            }
            "color_drift" => {
                if let Some(n) = v.as_f64() {
                    self.color_drift = n as f32;
                }
            }
            "breathe_scale" => {
                if let Some(n) = v.as_f64() {
                    self.breathe_scale = n as f32;
                }
            }
            "breathe_rotation" => {
                if let Some(n) = v.as_f64() {
                    self.breathe_rotation = n as f32;
                }
            }
            "breathe_position" => {
                if let Some(n) = v.as_f64() {
                    self.breathe_position = n as f32;
                }
            }
            "key_mode" => {
                if let Some(n) = v.as_u64() {
                    self.key_mode = (n as u32).min(4);
                }
            }
            "key_color_r" | "key_color_g" | "key_color_b" => {
                if let Some(n) = v.as_f64() {
                    let index = match param {
                        "key_color_r" => 0,
                        "key_color_g" => 1,
                        _ => 2,
                    };
                    self.key_color[index] = n as f32;
                }
            }
            "key_threshold" => {
                if let Some(n) = v.as_f64() {
                    self.key_threshold = n as f32;
                }
            }
            "key_softness" => {
                if let Some(n) = v.as_f64() {
                    self.key_softness = n as f32;
                }
            }
            "key_tolerance" => {
                if let Some(n) = v.as_f64() {
                    self.key_tolerance = n as f32;
                }
            }
            "cellular_amount" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_amount = n as f32;
                }
            }
            "cellular_scale" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_scale = n as f32;
                }
            }
            "cellular_warp" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_warp = n as f32;
                }
            }
            "cellular_speed" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_speed = n as f32;
                }
            }
            "cellular_gap_amount" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_gap_amount = n as f32;
                }
            }
            "cellular_gap_threshold" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_gap_threshold = n as f32;
                }
            }
            "cellular_gap_softness" => {
                if let Some(n) = v.as_f64() {
                    self.cellular_gap_softness = n as f32;
                }
            }
            "negative_mode" => {
                if let Some(n) = v.as_u64() {
                    self.negative_mode = (n as u32).min(2);
                }
            }
            "key_border_color" => {
                if let Some(n) = v.as_u64() {
                    self.key_border_color = (n as u32).min(7);
                }
            }
            "key_border" => {
                if let Some(n) = v.as_f64() {
                    self.key_border = n as f32;
                }
            }
            "key_shadow" => {
                if let Some(n) = v.as_f64() {
                    self.key_shadow = n as f32;
                }
            }
            // B13 small effects: every remaining control is a plain float.
            "contour" | "contour_bands" | "contour_width" | "contour_hue" | "contour_fill"
            | "flatten" | "flatten_levels" | "contour_dither" | "solarize" | "negative"
            | "colourpass" | "colourpass_hue" | "colourpass_width" | "edge_amount" | "edge_hue"
            | "emboss" | "emboss_angle" | "halftone" | "halftone_pitch" | "halftone_angle"
            | "moire" | "moire_freq" | "row_smear" | "bitcrush" | "bitcrush_levels"
            | "bitcrush_dither" | "multi_grid_x" | "multi_grid_y" | "barrel"
            | "chroma_aberration" | "anamorphic_streak" => {
                if let Some(n) = v.as_f64() {
                    let slot = match param {
                        "contour" => &mut self.contour,
                        "contour_bands" => &mut self.contour_bands,
                        "contour_width" => &mut self.contour_width,
                        "contour_hue" => &mut self.contour_hue,
                        "contour_fill" => &mut self.contour_fill,
                        "flatten" => &mut self.flatten,
                        "flatten_levels" => &mut self.flatten_levels,
                        "contour_dither" => &mut self.contour_dither,
                        "solarize" => &mut self.solarize,
                        "negative" => &mut self.negative,
                        "colourpass" => &mut self.colourpass,
                        "colourpass_hue" => &mut self.colourpass_hue,
                        "colourpass_width" => &mut self.colourpass_width,
                        "edge_amount" => &mut self.edge_amount,
                        "edge_hue" => &mut self.edge_hue,
                        "emboss" => &mut self.emboss,
                        "emboss_angle" => &mut self.emboss_angle,
                        "halftone" => &mut self.halftone,
                        "halftone_pitch" => &mut self.halftone_pitch,
                        "halftone_angle" => &mut self.halftone_angle,
                        "moire" => &mut self.moire,
                        "moire_freq" => &mut self.moire_freq,
                        "row_smear" => &mut self.row_smear,
                        "bitcrush" => &mut self.bitcrush,
                        "bitcrush_levels" => &mut self.bitcrush_levels,
                        "bitcrush_dither" => &mut self.bitcrush_dither,
                        "multi_grid_x" => &mut self.multi_grid_x,
                        "multi_grid_y" => &mut self.multi_grid_y,
                        "barrel" => &mut self.barrel,
                        "chroma_aberration" => &mut self.chroma_aberration,
                        _ => &mut self.anamorphic_streak,
                    };
                    *slot = n as f32;
                }
            }
            _ => {}
        }
    }
}

impl WebState {
    pub fn new() -> Result<Arc<Self>, String> {
        let (tx, _) = watch::channel(Arc::new(String::new()));
        Ok(Arc::new(Self {
            app: RwLock::new(AppSnapshot::default()),
            tx,
            snapshot_requested: AtomicBool::new(false),
            full_snapshot_generation: AtomicU64::new(0),
            last_full_snapshot: std::sync::RwLock::new(Arc::new(String::new())),
            serialized_publications: AtomicU64::new(0),
            actions: Mutex::new(Vec::with_capacity(MAX_PENDING_ACTIONS)),
            action_sequencer: Arc::new(ActionSequencer::default()),
            action_correlation: std::sync::Mutex::new(ActionCorrelationMonitor::default()),
            controller_profile_export: std::sync::RwLock::new(
                crate::controller_profile::export_controller_profile_json(
                    &crate::controller_profile::ControllerProfileDocument::default(),
                )
                .map_err(|error| format!("default controller profile is invalid: {error}"))?,
            ),
            browser_history_gesture: Mutex::new(None),
            thumbnails: std::sync::RwLock::new(HashMap::new()),
            preview_frames: std::sync::RwLock::new(HashMap::new()),
            library_media_cache_bytes: std::sync::Mutex::new(0),
            library_media_helper_gate: std::sync::Mutex::new(()),
            library_folder: std::sync::RwLock::new(None),
            upload_admission: UploadAdmission::default(),
            upload_media_safety_policy: std::sync::RwLock::new(
                crate::media_safety::MediaSafetyPolicy::default(),
            ),
            upload_publication_gate: std::sync::Mutex::new(()),
            control_server: std::sync::RwLock::new(ControlServerInfo::default()),
            control_server_generation: AtomicU64::new(0),
            control_server_stopping_generation: AtomicU64::new(0),
            library_generation: AtomicU64::new(0),
            library_index: std::sync::RwLock::new(Arc::new(
                crate::library_index::LibraryIndex::empty(),
            )),
            library_index_revision: AtomicU64::new(0),
            gyro_streams: std::sync::Mutex::new(GyroStreamRegistry::default()),
            monitor_watchers: std::sync::Mutex::new(HashMap::new()),
            next_client_id: AtomicU64::new(1),
        }))
    }

    pub async fn enqueue_action(self: &Arc<Self>, action: WebAction) -> EnqueueOutcome {
        self.enqueue_action_with_ack(action).await.0
    }

    pub(crate) async fn enqueue_action_with_ack(
        self: &Arc<Self>,
        action: WebAction,
    ) -> (EnqueueOutcome, ActionIngressAck) {
        let action = self.envelope_web_action(action);
        self.enqueue_enveloped_action_with_ack(action).await
    }

    pub(crate) fn envelope_web_action(&self, action: WebAction) -> ActionEnvelope<WebAction> {
        let source = web_action_source(&action);
        self.action_sequencer.envelope(source, action)
    }

    pub(crate) fn action_ingress_terminal_guard(
        self: &Arc<Self>,
        identity: ActionIdentity,
    ) -> ActionIngressTerminalGuard {
        ActionIngressTerminalGuard::new(self.clone(), identity)
    }

    pub(crate) fn enqueue_enveloped_action_with_ack(
        self: &Arc<Self>,
        action: ActionEnvelope<WebAction>,
    ) -> impl std::future::Future<Output = (EnqueueOutcome, ActionIngressAck)> + Send + 'static
    {
        let guard = self.action_ingress_terminal_guard(action.identity());
        self.enqueue_guarded_action_with_ack(action, guard)
    }

    pub(crate) fn enqueue_guarded_action_with_ack(
        self: &Arc<Self>,
        action: ActionEnvelope<WebAction>,
        mut guard: ActionIngressTerminalGuard,
    ) -> impl std::future::Future<Output = (EnqueueOutcome, ActionIngressAck)> + Send + 'static
    {
        debug_assert_eq!(guard.identity(), action.identity());
        let state = self.clone();
        async move {
            let sequence = action.sequence().get();
            let mut queue = state.actions.lock().await;
            // From this synchronous point onward either the queue owns the exact
            // envelope or enqueue_bounded_detailed returns it as a terminal.
            guard.disarm();
            let report = enqueue_bounded_detailed(&mut queue, action);
            drop(queue);
            if let Some((terminal, disposition)) = report.terminal {
                state.record_terminal_action(&terminal, disposition);
            }
            let disposition = match report.outcome {
                EnqueueOutcome::Added => ActionIngressDisposition::Queued,
                EnqueueOutcome::Coalesced => ActionIngressDisposition::QueuedAfterCoalescing,
                EnqueueOutcome::Dropped => ActionIngressDisposition::Refused,
            };
            (
                report.outcome,
                ActionIngressAck {
                    kind: "action_ack",
                    sequence,
                    disposition,
                },
            )
        }
    }

    pub(crate) fn terminal_action_identity_with_ack(
        &self,
        identity: ActionIdentity,
        disposition: ActionDisposition,
    ) -> ActionIngressAck {
        self.record_terminal_action_identity(identity, disposition);
        ActionIngressAck {
            kind: "action_ack",
            sequence: identity.sequence().get(),
            disposition: if disposition == ActionDisposition::Coalesced {
                ActionIngressDisposition::Coalesced
            } else {
                ActionIngressDisposition::Refused
            },
        }
    }

    /// A connecting browser requests one fresh full snapshot. Main consumes
    /// the flag at its next accepted event-loop boundary; no serializer or
    /// application mutation runs on the socket task.
    pub(crate) fn request_snapshot(&self) -> u64 {
        self.snapshot_requested.store(true, Ordering::Release);
        self.full_snapshot_generation.load(Ordering::Acquire)
    }

    pub(crate) fn take_snapshot_request(&self) -> bool {
        self.snapshot_requested.swap(false, Ordering::AcqRel)
    }

    pub(crate) fn publish_serialized_snapshot(&self, message: String) -> u64 {
        let message = Arc::new(message);
        *self
            .last_full_snapshot
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = message.clone();
        self.tx.send_replace(message);
        self.serialized_publications.fetch_add(1, Ordering::Relaxed);
        self.full_snapshot_generation
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1)
    }

    pub(crate) fn publish_serialized_telemetry(&self, message: String) -> u64 {
        self.tx.send_replace(Arc::new(message));
        self.serialized_publications
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1)
    }

    pub(crate) fn full_snapshot_generation(&self) -> u64 {
        self.full_snapshot_generation.load(Ordering::Acquire)
    }

    pub(crate) fn last_full_snapshot(&self) -> Arc<String> {
        self.last_full_snapshot
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn serialized_publications_for_test(&self) -> u64 {
        self.serialized_publications.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn record_action_apply<T>(
        &self,
        action: &ActionEnvelope<T>,
        applied: Instant,
    ) -> bool {
        self.action_correlation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record_apply(action, applied)
    }

    pub(crate) fn record_action_identity_apply(
        &self,
        identity: ActionIdentity,
        applied: Instant,
    ) -> bool {
        self.action_correlation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record_apply_identity(identity, applied)
    }

    pub(crate) fn record_terminal_action<T>(
        &self,
        action: &ActionEnvelope<T>,
        disposition: ActionDisposition,
    ) {
        self.action_correlation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record_terminal(action.sequence(), action.source(), disposition);
    }

    pub(crate) fn record_terminal_action_identity(
        &self,
        identity: ActionIdentity,
        disposition: ActionDisposition,
    ) {
        self.action_correlation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record_terminal_identity(identity, disposition);
    }

    /// Clone the one process-wide sequencer into another transport adapter.
    /// The sequencer is deliberately internal and has no serialization path.
    pub(crate) fn action_sequencer(&self) -> Arc<ActionSequencer> {
        self.action_sequencer.clone()
    }

    pub(crate) fn record_action_submission(
        &self,
        highest_applied_sequence: u64,
        submission_generation: u64,
        submitted: Instant,
    ) {
        let Some(highest_applied) =
            crate::action_correlation::ActionSequence::from_nonzero(highest_applied_sequence)
        else {
            return;
        };
        self.action_correlation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record_submission(highest_applied, submission_generation, submitted);
    }

    pub(crate) fn action_timing_snapshot(&self) -> ActionTimingSnapshot {
        self.action_correlation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .snapshot()
    }

    /// Visit newly completed, payload-free action receipts without allocating
    /// or exposing the underlying fixed ring. The cursor follows completion
    /// order, not action sequence: independent adapters may complete older
    /// ingress identities after a newer action has already terminated.
    pub(crate) fn for_each_action_receipt_after(
        &self,
        after_completion: u64,
        mut visit: impl FnMut(crate::action_correlation::ActionCorrelationReceipt),
    ) -> u64 {
        let monitor = self
            .action_correlation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut cursor = after_completion;
        for (completion, receipt) in monitor.receipts_after(after_completion) {
            cursor = cursor.max(completion);
            visit(*receipt);
        }
        cursor
    }

    pub(crate) fn supersede_all_pending_actions(&self) {
        let mut actions = self.actions.blocking_lock();
        for action in actions.drain(..) {
            self.record_terminal_action(&action, ActionDisposition::Superseded);
        }
    }

    pub(crate) fn terminalize_all_applied_actions_not_yet_presented(&self) -> usize {
        self.action_correlation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .terminalize_all_pending(ActionDisposition::NotYetPresented)
    }

    pub(crate) fn retain_pending_actions(&self, mut keep: impl FnMut(&WebAction) -> bool) {
        let mut actions = self.actions.blocking_lock();
        let mut retained = Vec::with_capacity(actions.capacity());
        for action in actions.drain(..) {
            if keep(action.payload()) {
                retained.push(action);
            } else {
                self.record_terminal_action(&action, ActionDisposition::Superseded);
            }
        }
        *actions = retained;
    }

    #[cfg(test)]
    pub(crate) fn envelope_for_test(&self, action: WebAction) -> ActionEnvelope<WebAction> {
        self.action_sequencer
            .envelope(web_action_source(&action), action)
    }

    #[cfg(test)]
    pub(crate) fn action_receipts_for_test(
        &self,
    ) -> Vec<crate::action_correlation::ActionCorrelationReceipt> {
        self.action_correlation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .receipts()
            .copied()
            .collect()
    }

    pub fn publish_controller_profile_export(
        &self,
        document: &crate::controller_profile::ControllerProfileDocument,
    ) -> Result<(), crate::controller_profile::ControllerProfileError> {
        let bytes = crate::controller_profile::export_controller_profile_json(document)?;
        match self.controller_profile_export.write() {
            Ok(mut slot) => *slot = bytes,
            Err(poisoned) => *poisoned.into_inner() = bytes,
        }
        Ok(())
    }

    pub fn controller_profile_export(&self) -> Vec<u8> {
        match self.controller_profile_export.read() {
            Ok(bytes) => bytes.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Atomically establish socket-local gesture ownership only after the
    /// matching Begin envelope has crossed the bounded queue barrier. The
    /// terminal guard remains armed while either lock is awaited, so shutdown
    /// cancellation cannot leave an owner without a queued boundary.
    pub(crate) async fn enqueue_browser_history_begin_with_ack(
        self: &Arc<Self>,
        client_id: u64,
        gesture_id: u64,
        action: ActionEnvelope<WebAction>,
        guard: ActionIngressTerminalGuard,
    ) -> Result<(EnqueueOutcome, ActionIngressAck), ActionIngressAck> {
        let mut active = self.browser_history_gesture.lock().await;
        if active.is_some() {
            return Err(guard.terminalize(ActionDisposition::Refused));
        }
        let admitted = self.enqueue_guarded_action_with_ack(action, guard).await;
        if admitted.0 != EnqueueOutcome::Dropped {
            *active = Some(BrowserHistoryGesture {
                client_id,
                gesture_id,
                coalesce_key: None,
                dirty: false,
            });
        }
        Ok(admitted)
    }

    /// Validate and enqueue End/Cancel under the same ownership lock, then
    /// clear the owner synchronously only when queue admission succeeded.
    pub(crate) async fn enqueue_browser_history_finish_with_ack(
        self: &Arc<Self>,
        client_id: u64,
        gesture_id: u64,
        cancel: bool,
        action: ActionEnvelope<WebAction>,
        guard: ActionIngressTerminalGuard,
    ) -> Result<(EnqueueOutcome, ActionIngressAck), ActionIngressAck> {
        let mut active = self.browser_history_gesture.lock().await;
        let valid = active.as_ref().is_some_and(|active| {
            active.client_id == client_id
                && active.gesture_id == gesture_id
                && (!cancel || !active.dirty)
        });
        if !valid {
            return Err(guard.terminalize(ActionDisposition::Refused));
        }
        let admitted = self.enqueue_guarded_action_with_ack(action, guard).await;
        if admitted.0 != EnqueueOutcome::Dropped {
            *active = None;
        }
        Ok(admitted)
    }

    /// Perform gesture ownership admission and queue admission as one
    /// cancellation-safe transaction. Dirty/coalesce ownership is published
    /// only after the exact action envelope has entered the bounded queue.
    pub(crate) async fn enqueue_browser_action_during_gesture_with_ack(
        self: &Arc<Self>,
        client_id: u64,
        action: ActionEnvelope<WebAction>,
        guard: ActionIngressTerminalGuard,
    ) -> Result<(EnqueueOutcome, ActionIngressAck), ActionIngressAck> {
        let performance_only = action.payload().is_performance_only_for_history();
        let candidate_key = (!performance_only)
            .then(|| action.payload().coalesce_key())
            .flatten();
        let mut active = self.browser_history_gesture.lock().await;
        let mut publish_key = None;
        if !performance_only {
            if let Some(owner) = active.as_ref() {
                let valid = owner.client_id == client_id
                    && candidate_key.as_ref().is_some_and(|key| {
                        owner.coalesce_key.as_ref().is_none_or(|owned| owned == key)
                    });
                if !valid {
                    return Err(guard.terminalize(ActionDisposition::Refused));
                }
                publish_key = candidate_key;
            }
        }
        let admitted = self.enqueue_guarded_action_with_ack(action, guard).await;
        if admitted.0 != EnqueueOutcome::Dropped {
            if let Some(owner) = active.as_mut() {
                if let Some(key) = publish_key {
                    owner.coalesce_key.get_or_insert(key);
                    owner.dirty = true;
                }
            }
        }
        Ok(admitted)
    }

    pub async fn begin_browser_history_gesture(&self, client_id: u64, gesture_id: u64) -> bool {
        let mut active = self.browser_history_gesture.lock().await;
        if active.is_some() {
            return false;
        }
        *active = Some(BrowserHistoryGesture {
            client_id,
            gesture_id,
            coalesce_key: None,
            dirty: false,
        });
        true
    }

    /// Admit an action while preserving one-controller/one-destination
    /// gesture ownership. Performance-only events remain live but never mark
    /// the transaction dirty. An authored action from another client, a
    /// topology command, or a second scalar destination is rejected before it
    /// reaches Main and therefore cannot be folded into the owner's entry.
    pub async fn admit_browser_action_during_gesture(
        &self,
        client_id: u64,
        action: &WebAction,
    ) -> bool {
        if action.is_performance_only_for_history() {
            return true;
        }
        let mut active = self.browser_history_gesture.lock().await;
        let Some(active) = active.as_mut() else {
            return true;
        };
        if active.client_id != client_id {
            return false;
        }
        let Some(key) = action.coalesce_key() else {
            return false;
        };
        if active
            .coalesce_key
            .as_ref()
            .is_some_and(|owned| owned != &key)
        {
            return false;
        }
        active.coalesce_key.get_or_insert(key);
        active.dirty = true;
        true
    }

    /// A Cancel is valid only before any authored scalar crossed the Begin
    /// barrier. Dirty clients must End, matching Main's fingerprint guard.
    pub async fn may_finish_browser_history_gesture(
        &self,
        client_id: u64,
        gesture_id: u64,
        cancel: bool,
    ) -> bool {
        self.browser_history_gesture
            .lock()
            .await
            .as_ref()
            .is_some_and(|active| {
                active.client_id == client_id
                    && active.gesture_id == gesture_id
                    && (!cancel || !active.dirty)
            })
    }

    pub async fn finish_browser_history_gesture(&self, client_id: u64, gesture_id: u64) {
        let mut active = self.browser_history_gesture.lock().await;
        if active
            .as_ref()
            .is_some_and(|active| active.client_id == client_id && active.gesture_id == gesture_id)
        {
            *active = None;
        }
    }

    /// Close a vanished client's gesture without exposing a new owner until
    /// the matching End barrier is already ordered in Main's action queue.
    /// Clearing ownership first would let another client enqueue Begin(new)
    /// ahead of End(old), leaving the server and Main with different owners.
    pub async fn disconnect_browser_history_gesture(&self, client_id: u64) -> Option<u64> {
        let mut active = self.browser_history_gesture.lock().await;
        let gesture_id = match active.as_ref() {
            Some(active) if active.client_id == client_id => Some(active.gesture_id),
            _ => None,
        };
        if let Some(gesture_id) = gesture_id {
            // Keep the ownership lock through queue admission. Every other
            // Begin must therefore observe the old owner until End(old) is
            // present. End is a priority action and is never dropped by the
            // bounded queue; the outcome is retained as a debug assertion so
            // a future queue-policy change cannot silently reopen this race.
            let outcome = {
                let mut queue = self.actions.lock().await;
                let action = self.action_sequencer.envelope(
                    ActionSourceClass::Browser,
                    WebAction::EndHistoryGesture { gesture_id },
                );
                let report = enqueue_bounded_detailed(&mut queue, action);
                if let Some((terminal, disposition)) = report.terminal {
                    self.record_terminal_action(&terminal, disposition);
                }
                report.outcome
            };
            debug_assert_ne!(outcome, EnqueueOutcome::Dropped);
            *active = None;
        }
        gesture_id
    }

    pub fn allocate_client_id(&self) -> u64 {
        self.next_client_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn control_server_info(&self) -> ControlServerInfo {
        self.control_server
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub(crate) fn upload_admission(&self) -> UploadAdmission {
        self.upload_admission.clone()
    }

    pub(crate) fn set_upload_media_safety_policy(
        &self,
        policy: crate::media_safety::MediaSafetyPolicy,
    ) {
        *self
            .upload_media_safety_policy
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = policy;
    }

    pub(crate) fn upload_media_safety_policy(&self) -> crate::media_safety::MediaSafetyPolicy {
        self.upload_media_safety_policy
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub(crate) fn with_upload_publication_gate<T>(&self, operation: impl FnOnce() -> T) -> T {
        let _gate = self
            .upload_publication_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        operation()
    }

    pub(crate) fn control_server_generation(&self) -> u64 {
        self.control_server_generation.load(Ordering::Acquire)
    }

    pub(crate) fn control_server_generation_accepts_upload(&self, generation: u64) -> bool {
        generation != 0
            && self.control_server_generation() == generation
            && self
                .control_server_stopping_generation
                .load(Ordering::Acquire)
                != generation
    }

    pub(crate) fn mark_control_server_stopping(&self, generation: u64) {
        self.with_upload_publication_gate(|| {
            if self.control_server_generation() == generation {
                self.control_server_stopping_generation
                    .store(generation, Ordering::Release);
            }
        });
    }

    pub(crate) fn begin_control_server_generation(&self) -> u64 {
        let _gate = self
            .upload_publication_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let generation = self
            .control_server_generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |generation| {
                Some(generation.wrapping_add(1).max(1))
            })
            .unwrap_or_else(|generation| generation)
            .wrapping_add(1)
            .max(1);
        self.control_server_stopping_generation
            .store(0, Ordering::Release);
        generation
    }

    pub(crate) fn publish_control_server_info(&self, info: ControlServerInfo) {
        if info.generation != self.control_server_generation.load(Ordering::Acquire) {
            return;
        }
        let mut published = self
            .control_server
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *published = info;
    }

    pub(crate) fn update_control_listener(
        &self,
        generation: u64,
        slot: ControlListenerSlot,
        status: ControlListenerStatus,
    ) {
        if generation != self.control_server_generation.load(Ordering::Acquire) {
            return;
        }
        let mut info = self
            .control_server
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if info.generation != generation {
            return;
        }
        match slot {
            ControlListenerSlot::LoopbackIpv4 => {
                if !matches!(status, ControlListenerStatus::Listening { .. }) {
                    info.loopback_ipv4_url = None;
                }
                info.loopback_ipv4 = status;
            }
            ControlListenerSlot::LoopbackIpv6 => {
                if !matches!(status, ControlListenerStatus::Listening { .. }) {
                    info.loopback_ipv6_url = None;
                }
                info.loopback_ipv6 = status;
            }
            ControlListenerSlot::LanTls => {
                if !matches!(status, ControlListenerStatus::Listening { .. }) {
                    info.lan_url = None;
                }
                info.lan_tls = status;
            }
        }
    }

    pub(crate) fn mark_control_server_stopped(&self, generation: u64) {
        if generation != self.control_server_generation.load(Ordering::Acquire) {
            return;
        }
        self.control_server_stopping_generation
            .store(generation, Ordering::Release);
        let mut info = self
            .control_server
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if info.generation == generation {
            info.loopback_ipv4 = ControlListenerStatus::Stopped;
            info.loopback_ipv6 = ControlListenerStatus::Stopped;
            info.lan_tls = ControlListenerStatus::Stopped;
            info.loopback_ipv4_url = None;
            info.loopback_ipv6_url = None;
            info.lan_url = None;
        }
    }

    /// Start a new folder identity. Workers must compare their captured value
    /// at the cache-write boundary before publishing decoded previews.
    pub fn begin_library_generation(&self) -> u64 {
        let _gate = self
            .upload_publication_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.library_generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |generation| {
                Some(generation.wrapping_add(1).max(1))
            })
            .unwrap_or_else(|generation| generation)
            .wrapping_add(1)
            .max(1)
    }

    /// Publish a new library generation only after the caller's bounded worker
    /// admission succeeds. The same gate used by upload publication excludes a
    /// concurrent generation change between preparing and committing the scan.
    pub fn admit_library_generation<E>(
        &self,
        admit: impl FnOnce(u64) -> Result<(), E>,
    ) -> Result<u64, E> {
        let _gate = self
            .upload_publication_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current = self.library_generation.load(Ordering::Acquire);
        let generation = current.wrapping_add(1).max(1);
        admit(generation)?;
        self.library_generation.store(generation, Ordering::Release);
        Ok(generation)
    }

    pub fn library_generation(&self) -> u64 {
        self.library_generation.load(Ordering::Acquire)
    }

    pub fn library_generation_is_current(&self, generation: u64) -> bool {
        generation != 0 && self.library_generation() == generation
    }

    /// Publishes one immutable bounded result only while its folder identity
    /// is still current. The second check occurs while holding the publication
    /// lock, closing the ordinary stale-worker check/write race.
    pub fn publish_library_index(
        &self,
        generation: u64,
        mut index: crate::library_index::LibraryIndex,
    ) -> bool {
        if index.generation() != generation || !self.library_generation_is_current(generation) {
            return false;
        }
        let mut published = self
            .library_index
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !self.library_generation_is_current(generation) {
            return false;
        }
        let revision = self
            .library_index_revision
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |revision| {
                Some(revision.wrapping_add(1).max(1))
            })
            .unwrap_or_else(|revision| revision)
            .wrapping_add(1)
            .max(1);
        index.set_revision(revision);
        *published = Arc::new(index);
        true
    }

    pub fn library_index(&self) -> Arc<crate::library_index::LibraryIndex> {
        self.library_index
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn library_index_snapshot(&self) -> crate::library_index::LibraryIndexSnapshot {
        self.library_index().snapshot()
    }

    pub(crate) fn publish_thumbnail_with_budget(
        &self,
        generation: u64,
        filename: String,
        bytes: Vec<u8>,
        byte_limit: u64,
    ) -> bool {
        let Ok(new_bytes) = u64::try_from(bytes.len()) else {
            return false;
        };
        let mut retained = self
            .library_media_cache_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut cache = self
            .thumbnails
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !self.library_generation_is_current(generation) {
            return false;
        }
        let old_bytes = cache
            .get(&filename)
            .and_then(|old| u64::try_from(old.len()).ok())
            .unwrap_or(0);
        let Some(next_retained) = retained
            .checked_sub(old_bytes)
            .and_then(|bytes| bytes.checked_add(new_bytes))
            .filter(|bytes| *bytes <= byte_limit)
        else {
            return false;
        };
        cache.insert(filename, bytes);
        *retained = next_retained;
        true
    }

    pub(crate) fn publish_preview_with_budget(
        &self,
        generation: u64,
        filename: String,
        frames: Vec<Vec<u8>>,
        byte_limit: u64,
    ) -> bool {
        let Some(new_bytes) = frames.iter().try_fold(0_u64, |total, frame| {
            total.checked_add(u64::try_from(frame.len()).ok()?)
        }) else {
            return false;
        };
        let mut retained = self
            .library_media_cache_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut cache = self
            .preview_frames
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !self.library_generation_is_current(generation) {
            return false;
        }
        let old_bytes = cache
            .get(&filename)
            .and_then(|old| {
                old.iter().try_fold(0_u64, |total, frame| {
                    total.checked_add(u64::try_from(frame.len()).ok()?)
                })
            })
            .unwrap_or(0);
        let Some(next_retained) = retained
            .checked_sub(old_bytes)
            .and_then(|bytes| bytes.checked_add(new_bytes))
            .filter(|bytes| *bytes <= byte_limit)
        else {
            return false;
        };
        cache.insert(filename, frames);
        *retained = next_retained;
        true
    }

    pub(crate) fn clear_library_media_caches(&self) {
        let mut retained = self
            .library_media_cache_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.thumbnails
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.preview_frames
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        *retained = 0;
    }

    pub(crate) fn remove_library_media_cache_entry(&self, filename: &str) {
        let mut retained = self
            .library_media_cache_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let thumbnail_bytes = self
            .thumbnails
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(filename)
            .and_then(|bytes| u64::try_from(bytes.len()).ok())
            .unwrap_or(0);
        let preview_bytes = self
            .preview_frames
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(filename)
            .and_then(|frames| {
                frames.iter().try_fold(0_u64, |total, frame| {
                    total.checked_add(u64::try_from(frame.len()).ok()?)
                })
            })
            .unwrap_or(0);
        *retained = retained.saturating_sub(thumbnail_bytes.saturating_add(preview_bytes));
    }

    #[cfg(test)]
    pub(crate) fn library_media_cache_bytes(&self) -> u64 {
        *self
            .library_media_cache_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn lock_library_media_helpers(&self) -> std::sync::MutexGuard<'_, ()> {
        self.library_media_helper_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// B11: record a client's monitor-watch declaration. `enabled` refreshes
    /// the watcher's instant; `false` removes it immediately.
    pub fn set_monitor_watch(&self, client_id: u64, enabled: bool) -> bool {
        let mut watchers = self
            .monitor_watchers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if enabled {
            watchers.insert(client_id, Instant::now());
            true
        } else {
            watchers.remove(&client_id).is_some()
        }
    }

    /// B11: drop a disconnected client's watch, exactly as the gyro registry
    /// forgets its streamers.
    pub fn disconnect_monitor_client(&self, client_id: u64) {
        self.monitor_watchers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&client_id);
    }

    /// B11: whether any browser watcher is fresh. Expired entries are pruned
    /// here so the map stays bounded by live-ish clients.
    pub fn monitor_watch_active(&self) -> bool {
        let now = Instant::now();
        let mut watchers = self
            .monitor_watchers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        watchers
            .retain(|_, instant| now.saturating_duration_since(*instant) <= MONITOR_WATCH_TIMEOUT);
        !watchers.is_empty()
    }

    pub fn set_gyro_stream(&self, client_id: u64, enabled: bool) -> bool {
        self.gyro_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .set_stream(client_id, enabled)
    }

    pub fn note_gyro_sample(&self, client_id: u64) {
        self.gyro_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .note_sample_at(client_id, Instant::now());
    }

    pub fn disconnect_gyro_client(&self, client_id: u64) {
        self.gyro_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .disconnect(client_id);
    }

    pub fn gyro_status(&self) -> GyroStatusSnapshot {
        self.gyro_streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .status_at(Instant::now())
    }
}

#[cfg(test)]
mod protocol_tests {
    use super::*;

    fn assert_json_number_near(value: &serde_json::Value, expected: f64) {
        let actual = value.as_f64().expect("JSON number");
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn native_control_server_info_keeps_every_listener_lifecycle_independent() {
        assert_eq!(
            ControlListenerStatus::default(),
            ControlListenerStatus::Stopped
        );
        let state = WebState::new().expect("test state");
        assert_eq!(state.control_server_info(), ControlServerInfo::default());

        let generation = state.begin_control_server_generation();
        let local_url =
            ControlAccessUrl::new("http://127.0.0.1:3030/?key=seeded-secret".to_string());
        state.publish_control_server_info(ControlServerInfo {
            generation,
            loopback_ipv4: ControlListenerStatus::Starting,
            loopback_ipv6: ControlListenerStatus::Starting,
            lan_tls: ControlListenerStatus::Starting,
            loopback_ipv4_url: Some(local_url.clone()),
            loopback_ipv6_url: None,
            lan_url: None,
            session_fingerprint: "0123456789ab".to_string(),
        });
        state.update_control_listener(
            generation,
            ControlListenerSlot::LoopbackIpv4,
            ControlListenerStatus::Listening {
                address: "127.0.0.1:3030".parse().unwrap(),
            },
        );
        state.update_control_listener(
            generation,
            ControlListenerSlot::LanTls,
            ControlListenerStatus::Unavailable {
                reason: "address already in use".to_string(),
            },
        );
        let info = state.control_server_info();
        assert!(matches!(
            info.loopback_ipv4,
            ControlListenerStatus::Listening { .. }
        ));
        assert_eq!(info.loopback_ipv6, ControlListenerStatus::Starting);
        assert_eq!(
            info.lan_tls,
            ControlListenerStatus::Unavailable {
                reason: "address already in use".to_string()
            }
        );
        assert!(info.lan_url.is_none());
        assert_eq!(
            info.local_url().unwrap().expose_to_local_ui(),
            local_url.expose_to_local_ui()
        );
        assert!(!format!("{info:?}").contains("seeded-secret"));
        let snapshot = AppSnapshot {
            remote_url: local_url.clone(),
            ..AppSnapshot::default()
        };
        assert!(!format!("{snapshot:?}").contains("seeded-secret"));
        assert_eq!(
            serde_json::to_value(&snapshot).unwrap()["remote_url"],
            local_url.expose_to_local_ui(),
            "the authenticated browser wire deliberately receives the URL"
        );

        let retired_generation = generation;
        let current_generation = state.begin_control_server_generation();
        state.publish_control_server_info(ControlServerInfo {
            generation: current_generation,
            ..ControlServerInfo::default()
        });
        state.update_control_listener(
            retired_generation,
            ControlListenerSlot::LoopbackIpv4,
            ControlListenerStatus::Unavailable {
                reason: "late task crash".to_string(),
            },
        );
        assert_eq!(state.control_server_info().generation, current_generation);
    }

    #[test]
    fn library_generations_are_nonzero_monotonic_stale_write_barriers() {
        let state = WebState::new().expect("test token");
        assert_eq!(state.library_generation(), 0);

        let first = state.begin_library_generation();
        let second = state.begin_library_generation();

        assert_ne!(first, 0);
        assert!(second > first);
        assert!(!state.library_generation_is_current(first));
        assert!(state.library_generation_is_current(second));
        assert!(!state.library_generation_is_current(0));
    }

    #[test]
    fn bounded_library_index_publication_rejects_stale_generations_and_revises() {
        let state = WebState::new().expect("test state");
        let first = state.begin_library_generation();
        let second = state.begin_library_generation();
        assert!(!state
            .publish_library_index(first, crate::library_index::LibraryIndex::scanning(first),));
        assert!(state
            .publish_library_index(second, crate::library_index::LibraryIndex::scanning(second),));
        let scanning = state.library_index_snapshot();
        assert_eq!(scanning.generation, second);
        assert_eq!(scanning.revision, 1);
        assert_eq!(
            scanning.status,
            crate::library_index::LibraryScanStatus::Scanning
        );

        assert!(state.publish_library_index(
            second,
            crate::library_index::LibraryIndex::error(second, "injected failure"),
        ));
        let failed = state.library_index_snapshot();
        assert_eq!(failed.revision, 2);
        assert_eq!(failed.error, "injected failure");
    }

    fn set_bpm(value: f32) -> WebAction {
        WebAction::SetBpm { value }
    }

    #[test]
    fn web_action_queue_is_bounded() {
        let mut queue = Vec::new();
        for value in 0..1000 {
            assert_ne!(
                enqueue_bounded(&mut queue, set_bpm(value as f32)),
                EnqueueOutcome::Dropped
            );
        }
        assert_eq!(queue.len(), 1, "absolute fader traffic coalesces");

        for _ in 0..MAX_PENDING_ACTIONS * 2 {
            let _ = enqueue_bounded(&mut queue, WebAction::AddRouting);
        }
        assert!(queue.len() <= MAX_PENDING_ACTIONS - PRIORITY_ACTION_RESERVE);
        assert_ne!(
            enqueue_bounded(&mut queue, WebAction::CancelExport),
            EnqueueOutcome::Dropped
        );
        assert!(queue.len() <= MAX_PENDING_ACTIONS);
        assert!(matches!(queue.last(), Some(WebAction::CancelExport)));
    }

    #[test]
    fn output_window_absolute_state_is_priority_and_coalesces() {
        let enabled: WebAction =
            serde_json::from_str(r#"{"action":"set_output_window","enabled":true}"#).unwrap();
        assert!(matches!(
            enabled,
            WebAction::SetOutputWindow { enabled: true }
        ));
        assert_eq!(enabled.coalesce_key().as_deref(), Some("output:enabled"));
        assert!(enabled.is_priority());

        let mut queue = vec![WebAction::AddRouting; MAX_PENDING_ACTIONS];
        assert_eq!(enqueue_bounded(&mut queue, enabled), EnqueueOutcome::Added);
        assert!(matches!(
            queue.last(),
            Some(WebAction::SetOutputWindow { enabled: true })
        ));
        assert_eq!(
            enqueue_bounded(&mut queue, WebAction::SetOutputWindow { enabled: false },),
            EnqueueOutcome::Coalesced
        );
        assert!(matches!(
            queue.last(),
            Some(WebAction::SetOutputWindow { enabled: false })
        ));

        let legacy: WebAction =
            serde_json::from_str(r#"{"action":"toggle_output_window"}"#).unwrap();
        assert!(matches!(legacy, WebAction::ToggleOutputWindow));
        assert!(legacy.is_priority());
        assert!(legacy.coalesce_key().is_none());

        let display: WebAction = serde_json::from_str(
            r#"{"action":"set_output_display","display_id":"display-0123456789abcdef-2"}"#,
        )
        .unwrap();
        assert!(matches!(
            &display,
            WebAction::SetOutputDisplay {
                display_id,
                inventory_generation: None,
            } if display_id == "display-0123456789abcdef-2"
        ));
        assert_eq!(display.coalesce_key().as_deref(), Some("output:display"));
        assert!(display.is_priority());
        assert!(display.is_performance_only_for_history());

        let automatic: WebAction =
            serde_json::from_str(r#"{"action":"set_output_display","display_id":""}"#).unwrap();
        assert!(matches!(
            automatic,
            WebAction::SetOutputDisplay { display_id, .. } if display_id.is_empty()
        ));
        let generation_checked: WebAction = serde_json::from_str(
            r#"{"action":"set_output_display","display_id":"display-0123456789abcdef-2","inventory_generation":7}"#,
        )
        .unwrap();
        assert!(matches!(
            generation_checked,
            WebAction::SetOutputDisplay {
                inventory_generation: Some(7),
                ..
            }
        ));
        let rescan: WebAction =
            serde_json::from_str(r#"{"action":"rescan_output_displays"}"#).unwrap();
        assert!(matches!(rescan, WebAction::RescanOutputDisplays));
        assert!(rescan.is_priority());
        assert!(rescan.is_performance_only_for_history());
    }

    #[test]
    fn morph_capture_is_an_ordering_barrier_for_fader_coalescing() {
        let mut queue = Vec::new();
        assert_eq!(
            enqueue_bounded(&mut queue, WebAction::SetMorph { value: 0.2 }),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(
                &mut queue,
                WebAction::MorphCapture {
                    slot: "a".into(),
                    stack_revision: Some(7),
                    composition_revision: Some(11),
                },
            ),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(&mut queue, WebAction::SetMorph { value: 0.8 }),
            EnqueueOutcome::Added
        );
        assert_eq!(queue.len(), 3);
        assert!(matches!(queue[0], WebAction::SetMorph { value } if value == 0.2));
        assert!(matches!(
            queue[1],
            WebAction::MorphCapture {
                stack_revision: Some(7),
                composition_revision: Some(11),
                ..
            }
        ));
        assert!(matches!(queue[2], WebAction::SetMorph { value } if value == 0.8));

        let legacy: WebAction =
            serde_json::from_str(r#"{"action":"morph_capture","slot":"b"}"#).unwrap();
        assert!(matches!(
            legacy,
            WebAction::MorphCapture {
                stack_revision: None,
                composition_revision: None,
                ..
            }
        ));
    }

    #[test]
    fn temporal_originals_snapshot_is_additive_exact_and_reports_runtime_identity() {
        let mut legacy_garden = serde_json::to_value(RefreshGardenSnapshot::default()).unwrap();
        legacy_garden.as_object_mut().unwrap().remove("matte_route");
        legacy_garden
            .as_object_mut()
            .unwrap()
            .remove("motion_route");
        assert_eq!(
            serde_json::from_value::<RefreshGardenSnapshot>(legacy_garden).unwrap(),
            RefreshGardenSnapshot::default()
        );

        let mut legacy = serde_json::to_value(TemporalSnapshot::default()).unwrap();
        let object = legacy.as_object_mut().unwrap();
        object.remove("originals");
        object.remove("telemetry");
        let restored: TemporalSnapshot = serde_json::from_value(legacy).unwrap();
        assert_eq!(restored.originals, TemporalOriginalsSnapshot::default());
        assert_eq!(restored.telemetry, TemporalTelemetrySnapshot::default());

        let mut params = crate::effects::params::TemporalParams::default();
        params.originals.loom.amount = 0.7;
        params.originals.loom.topology = crate::effects::params::TemporalTopology::Kaleidoscopic;
        params.originals.loom.interpolation = crate::effects::params::TemporalInterpolation::Linear;
        params.originals.atlas.seed = 0xdead_beef;
        params.originals.garden.gate = crate::effects::params::RefreshGardenGate::AudioOnset;
        params.originals.long_exposure.amount = 0.6;
        params.originals.long_exposure.shutter_frames = 19;
        params.originals.score.enabled = true;
        params.originals.score.trigger = crate::effects::params::CollisionScoreTrigger::Manual;
        let layer_id = crate::image_routing::StableLayerId::new(91).unwrap();
        let saved_position = SavedLayerPosition::new(3).unwrap();
        params.originals.garden.matte_route =
            crate::temporal::RefreshGardenMatteRoute::SelectedLayer {
                layer_id,
                saved_position,
                stage: LayerImageStage::PreLocalEffects,
            };
        params.originals.garden.motion_route =
            crate::temporal::RefreshGardenMotionRoute::MissingSelectedLayer { saved_position };
        params.originals.score.loop_driver =
            crate::temporal::CollisionScoreLoopDriver::SelectedLayer {
                layer_id,
                saved_position,
            };
        params.originals.reset.loop_boundary = crate::temporal::TemporalEventResetMode::Memory;

        let snapshot = TemporalSnapshot::from_params(&params);
        assert_eq!(snapshot.originals.loom.amount, 0.7);
        assert_eq!(snapshot.originals.loom.topology, "kaleidoscopic");
        assert_eq!(snapshot.originals.loom.interpolation, "linear");
        assert_eq!(snapshot.originals.atlas.seed, 0xdead_beef);
        assert_eq!(snapshot.originals.garden.gate, "audio_onset");
        assert_eq!(snapshot.originals.long_exposure.amount, 0.6);
        assert_eq!(snapshot.originals.long_exposure.shutter_frames, 19);
        assert_eq!(
            snapshot.originals.garden.matte_route,
            RefreshGardenMatteRouteSnapshot::SelectedLayer {
                layer_id: layer_id.get().to_string(),
                saved_position: saved_position.get(),
                stage: LayerImageStage::PreLocalEffects,
            }
        );
        assert_eq!(
            snapshot.originals.garden.motion_route,
            RefreshGardenMotionRouteSnapshot::MissingSelectedLayer {
                saved_position: saved_position.get(),
            }
        );
        assert_eq!(snapshot.originals.score.trigger, "manual");
        assert_eq!(snapshot.originals.reset.loop_boundary, "memory");
        assert_eq!(
            snapshot.originals.score.loop_driver,
            CollisionScoreLoopDriverSnapshot::SelectedLayer {
                layer_id: layer_id.get().to_string(),
                saved_position: saved_position.get(),
            }
        );
        assert_eq!(snapshot.telemetry, TemporalTelemetrySnapshot::default());
        let value = serde_json::to_value(snapshot).unwrap();
        assert_eq!(
            value["originals"]["score"]["loop_driver"]["kind"],
            "selected_layer"
        );
        assert_eq!(value["originals"]["score"]["loop_driver"]["layer_id"], "91");
        assert_eq!(
            value["originals"]["garden"]["matte_route"]["kind"],
            "selected_layer"
        );
        assert_eq!(
            value["originals"]["garden"]["matte_route"]["stage"],
            "pre_local_effects"
        );
        assert_eq!(
            value["originals"]["garden"]["motion_route"]["kind"],
            "missing_selected_layer"
        );
    }

    #[test]
    fn motion_snapshot_is_additive_exact_and_reports_runtime_identity() {
        use crate::motion::{
            CurvedShutterParams, CurvedShutterQuality, FaradayParams, MotionCarrier, MotionDonor,
            MotionFieldSource, MotionLatticeQuality, MotionParams,
        };

        let defaults = MotionSnapshot::default();
        assert_eq!(defaults.algorithm_version, 1);
        assert_eq!(defaults.field_source, "auto");
        assert_eq!(defaults.lattice_quality, "live");
        assert_eq!(defaults.transplant, FaradayMotionSnapshot::default());
        assert_eq!(defaults.shutter, CurvedShutterSnapshot::default());
        assert_eq!(defaults.telemetry, MotionTelemetrySnapshot::default());
        let layer_id = crate::image_routing::StableLayerId::new(91).unwrap();
        let saved_position = SavedLayerPosition::new(3).unwrap();
        let snapshot = MotionSnapshot::from_params(MotionParams {
            field_source: MotionFieldSource::CodecVectors,
            lattice_quality: MotionLatticeQuality::High,
            transplant: FaradayParams {
                amount: 0.75,
                donor: MotionDonor::Selected {
                    layer_id,
                    saved_position,
                },
                carrier: MotionCarrier::FirstSourceFrame,
                confidence_threshold: 0.4,
                confidence_softness: 0.2,
                refresh: 0.8,
                decay: 0.6,
                occlusion: 0.3,
            },
            shutter: CurvedShutterParams {
                angle_degrees: 270.0,
                phase: -0.25,
                curvature: 1.5,
                chromatic_lag: 0.4,
                quality: CurvedShutterQuality::High,
            },
            ..MotionParams::default()
        });
        assert_eq!(snapshot.field_source, "codec_vectors");
        assert_eq!(snapshot.lattice_quality, "high");
        assert_eq!(snapshot.transplant.amount, 0.75);
        assert_eq!(snapshot.transplant.carrier, "first_source_frame");
        assert_eq!(
            snapshot.transplant.donor,
            MotionDonorSnapshot::Selected {
                layer_id: "91".into(),
                saved_position: 3,
            }
        );
        assert_eq!(snapshot.shutter.quality, "high");
        assert_eq!(snapshot.shutter.sample_count, 16);
        let value = serde_json::to_value(snapshot).unwrap();
        assert_eq!(value["transplant"]["donor"]["kind"], "selected");
        assert_eq!(value["transplant"]["donor"]["layer_id"], "91");

        let mut legacy = serde_json::to_value(AppSnapshot::default()).unwrap();
        legacy.as_object_mut().unwrap().remove("master_motion");
        let legacy: AppSnapshot = serde_json::from_value(legacy).unwrap();
        assert_eq!(legacy.master_motion, MotionSnapshot::default());
    }

    #[test]
    fn temporal_memory_commands_are_priority_ordering_barriers() {
        let clear: WebAction =
            serde_json::from_str(r#"{"action":"clear_temporal_memory"}"#).unwrap();
        let trigger: WebAction =
            serde_json::from_str(r#"{"action":"trigger_collision_score"}"#).unwrap();
        let refresh: WebAction =
            serde_json::from_str(r#"{"action":"trigger_refresh_garden"}"#).unwrap();
        let clear_track: WebAction =
            serde_json::from_str(r#"{"action":"clear_temporal_event_track"}"#).unwrap();
        assert!(matches!(clear, WebAction::ClearTemporalMemory));
        assert!(matches!(trigger, WebAction::TriggerCollisionScore));
        assert!(matches!(refresh, WebAction::TriggerRefreshGarden));
        assert!(matches!(clear_track, WebAction::ClearTemporalEventTrack));
        assert!(clear.is_priority());
        assert!(trigger.is_priority());
        assert!(refresh.is_priority());
        assert!(clear_track.is_priority());
        assert!(clear.coalesce_key().is_none());
        assert!(trigger.coalesce_key().is_none());
        assert!(refresh.coalesce_key().is_none());
        assert!(clear_track.coalesce_key().is_none());

        let edit = |value| WebAction::SetTemporal {
            param: "loom_amount".into(),
            value: serde_json::json!(value),
        };
        let mut queue = Vec::new();
        assert_eq!(
            enqueue_bounded(&mut queue, edit(0.2)),
            EnqueueOutcome::Added
        );
        assert_eq!(enqueue_bounded(&mut queue, clear), EnqueueOutcome::Added);
        assert_eq!(
            enqueue_bounded(&mut queue, edit(0.8)),
            EnqueueOutcome::Added
        );
        assert_eq!(enqueue_bounded(&mut queue, trigger), EnqueueOutcome::Added);
        assert_eq!(enqueue_bounded(&mut queue, refresh), EnqueueOutcome::Added);
        assert_eq!(
            enqueue_bounded(&mut queue, clear_track),
            EnqueueOutcome::Added
        );
        assert_eq!(queue.len(), 6);
        assert!(
            matches!(&queue[0], WebAction::SetTemporal { value, .. } if value == &serde_json::json!(0.2))
        );
        assert!(matches!(queue[1], WebAction::ClearTemporalMemory));
        assert!(
            matches!(&queue[2], WebAction::SetTemporal { value, .. } if value == &serde_json::json!(0.8))
        );
        assert!(matches!(queue[3], WebAction::TriggerCollisionScore));
        assert!(matches!(queue[4], WebAction::TriggerRefreshGarden));
        assert!(matches!(queue[5], WebAction::ClearTemporalEventTrack));
    }

    #[test]
    fn the_collider_input_action_is_an_uncoalesced_ordered_slot_named_barrier() {
        let action = WebAction::SetMotionColliderInput {
            layer_id: "9".into(),
            input: FieldColliderInputSnapshot::B,
            donor_layer_id: Some("4".into()),
            layer_stack_revision: 12,
        };
        // Ordered barrier: it reserves priority admission and has NO coalesce
        // key at all, which is what makes the bounded queue treat it as a
        // barrier rather than replacing an older pending edit.
        assert!(action.is_priority());
        assert!(action.coalesce_key().is_none());

        let value = serde_json::to_value(&action).unwrap();
        assert_eq!(value["action"], "set_motion_collider_input");
        assert_eq!(value["layer_stack_revision"], 12);
        // The addressed slot is a closed NAMED token, never an index.
        assert_eq!(value["input"], "b");
        let WebAction::SetMotionColliderInput {
            layer_id,
            input,
            donor_layer_id,
            layer_stack_revision,
        } = serde_json::from_value::<WebAction>(value).unwrap()
        else {
            panic!("the collider input action must decode to its own variant");
        };
        assert_eq!(layer_id, "9");
        assert_eq!(input, FieldColliderInputSnapshot::B);
        assert_eq!(input.to_runtime(), crate::motion::FieldColliderInput::B);
        assert_eq!(donor_layer_id.as_deref(), Some("4"));
        assert_eq!(layer_stack_revision, 12);

        // Clearing a slot omits the donor entirely.
        let cleared = serde_json::from_str::<WebAction>(
            r#"{"action":"set_motion_collider_input","layer_id":"9","input":"a","layer_stack_revision":3}"#,
        )
        .unwrap();
        let WebAction::SetMotionColliderInput {
            input,
            donor_layer_id,
            ..
        } = cleared
        else {
            panic!("a cleared slot is still the collider action");
        };
        assert_eq!(input, FieldColliderInputSnapshot::A);
        assert_eq!(donor_layer_id, None);

        // An unknown slot token is a deserialization rejection, never a
        // positional fallback onto the partner input.
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_motion_collider_input","layer_id":"9","input":"c","layer_stack_revision":3}"#,
        )
        .is_err());
        // The revision is mandatory, so a reroute can never arrive unbarriered.
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_motion_collider_input","layer_id":"9","input":"a"}"#,
        )
        .is_err());
    }

    #[test]
    fn the_collider_snapshot_is_additive_and_publishes_both_slots_as_motion_donors() {
        // Absent is exactly the pre-collider snapshot.
        let defaults = FieldColliderSnapshot::default();
        assert!(!defaults.enabled);
        assert_eq!(defaults.mode, "sum");
        assert_eq!(defaults.boundary, "transparent");
        assert_eq!(defaults.input_a, MotionDonorSnapshot::None);
        assert_eq!(defaults.input_b, MotionDonorSnapshot::None);
        assert!(!defaults.admitted);
        assert!(defaults.diagnostic.is_empty());
        assert_eq!(MotionSnapshot::default().collider, defaults);

        let json = serde_json::to_string(&MotionSnapshot::default()).unwrap();
        let mut without: serde_json::Value = serde_json::from_str(&json).unwrap();
        without.as_object_mut().unwrap().remove("collider");
        let restored: MotionSnapshot = serde_json::from_value(without).unwrap();
        assert_eq!(restored.collider, defaults);

        let first = crate::image_routing::StableLayerId::new(31).unwrap();
        let snapshot = MotionSnapshot::from_params(crate::motion::MotionParams {
            collider: crate::motion::FieldColliderParams {
                enabled: true,
                mode: crate::motion::FieldColliderMode::CollisionBoundary,
                boundary: crate::motion::MotionBoundaryMode::Hold,
                input_a: crate::motion::MotionDonor::Selected {
                    layer_id: first,
                    saved_position: SavedLayerPosition::new(2).unwrap(),
                },
                input_b: crate::motion::MotionDonor::Missing {
                    saved_position: SavedLayerPosition::new(5).unwrap(),
                },
                ..crate::motion::FieldColliderParams::default()
            },
            ..crate::motion::MotionParams::default()
        });
        assert!(snapshot.collider.enabled);
        assert_eq!(snapshot.collider.mode, "collision_boundary");
        assert_eq!(snapshot.collider.boundary, "hold");
        // Both slots ride the established donor vocabulary, preserving their
        // Selected/Missing tombstone semantics. There is no parallel encoding.
        assert_eq!(
            snapshot.collider.input_a,
            MotionDonorSnapshot::Selected {
                layer_id: "31".into(),
                saved_position: 2,
            }
        );
        assert_eq!(
            snapshot.collider.input_b,
            MotionDonorSnapshot::Missing { saved_position: 5 }
        );
        // A tombstone is published as a tombstone and never as a selection.
        let wire = serde_json::to_value(&snapshot.collider).unwrap();
        assert_eq!(wire["input_b"]["kind"], "missing");
        assert!(wire["input_b"].get("layer_id").is_none());
    }

    #[test]
    fn the_collider_barrier_is_never_quantizable_in_the_panel_or_the_engine() {
        // The panel's quantizable set is a hand-maintained literal; a topology
        // barrier must never appear in it.
        let js = include_str!("../../static/app.js");
        let start = js.find("const QUANTIZABLE_ACTIONS").unwrap();
        let end = js[start..].find("]);").unwrap() + start;
        let quantizable = &js[start..end];
        assert!(!quantizable.contains("set_motion_collider_input"));
        assert!(!quantizable.contains("set_motion_donor"));
        assert!(!quantizable.contains("set_motion"));

        // The panel does send it, with its slot token and the current revision.
        assert!(js.contains("action: 'set_motion_collider_input'"));
        assert!(js.contains("motion-collider-select"));
        assert!(js.contains("collider_enabled"));
        assert!(js.contains("collider_mode"));
        assert!(js.contains("collider_boundary"));
        // Accessible names survive for both slots.
        assert!(js.contains("Field Collider input A"));
        assert!(js.contains("Field Collider input B"));
    }

    #[test]
    fn motion_scalars_coalesce_but_donor_and_memory_are_priority_barriers() {
        let edit = |value| WebAction::SetMotion {
            scope: MotionScopeSnapshot::Layer {
                layer_id: "91".into(),
            },
            param: "transplant_amount".into(),
            value: serde_json::json!(value),
        };
        let donor = WebAction::SetMotionDonor {
            layer_id: "91".into(),
            donor_layer_id: Some("77".into()),
            layer_stack_revision: 4,
        };
        let clear = WebAction::ClearMotionMemory;
        assert_eq!(
            edit(0.2).coalesce_key().as_deref(),
            Some("motion:layer:91:transplant_amount")
        );
        assert_eq!(donor.coalesce_key(), None);
        assert_eq!(clear.coalesce_key(), None);
        assert!(donor.is_priority());
        assert!(clear.is_priority());

        let mut queue = Vec::new();
        assert_eq!(
            enqueue_bounded(&mut queue, edit(0.2)),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(&mut queue, edit(0.4)),
            EnqueueOutcome::Coalesced
        );
        assert_eq!(enqueue_bounded(&mut queue, donor), EnqueueOutcome::Added);
        assert_eq!(
            enqueue_bounded(&mut queue, edit(0.8)),
            EnqueueOutcome::Added
        );
        assert_eq!(enqueue_bounded(&mut queue, clear), EnqueueOutcome::Added);
        assert_eq!(queue.len(), 4);
        assert!(
            matches!(&queue[0], WebAction::SetMotion { value, .. } if value == &serde_json::json!(0.4))
        );
        assert!(matches!(queue[1], WebAction::SetMotionDonor { .. }));
        assert!(
            matches!(&queue[2], WebAction::SetMotion { value, .. } if value == &serde_json::json!(0.8))
        );
        assert!(matches!(queue[3], WebAction::ClearMotionMemory));
    }

    #[test]
    fn layer_identity_and_direct_effect_protocol_round_trip() {
        let effect: WebAction = serde_json::from_str(
            r#"{"action":"set_layer_effect","index":2,"layer_id":"layer-abc","param":"downsample","value":0.5}"#,
        )
        .unwrap();
        assert!(
            matches!(effect, WebAction::SetLayerEffect { index: 2, layer_id: Some(id), param, .. } if id == "layer-abc" && param == "downsample")
        );

        let bypass: WebAction = serde_json::from_str(
            r#"{"action":"set_layer_param","index":2,"layer_id":"layer-abc","param":"bypass_master_fx","value":true}"#,
        )
        .unwrap();
        assert!(
            matches!(bypass, WebAction::SetLayerParam { index: 2, layer_id: Some(id), param, value }
                if id == "layer-abc" && param == "bypass_master_fx" && value == serde_json::json!(true))
        );

        let temporal_bypass: WebAction = serde_json::from_str(
            r#"{"action":"set_layer_param","index":2,"layer_id":"layer-abc","param":"bypass_temporal_fx","value":true}"#,
        )
        .unwrap();
        assert!(
            matches!(temporal_bypass, WebAction::SetLayerParam { index: 2, layer_id: Some(id), param, value }
                if id == "layer-abc" && param == "bypass_temporal_fx" && value == serde_json::json!(true))
        );

        let reorder: WebAction = serde_json::from_str(
            r#"{"action":"move_layer","from":2,"to":0,"layer_id":"layer-abc","stack_revision":7}"#,
        )
        .unwrap();
        assert!(matches!(
            reorder,
            WebAction::MoveLayer {
                stack_revision: Some(7),
                ..
            }
        ));

        let export: WebAction = serde_json::from_str(
            r#"{"action":"start_export","width":1280,"height":720,"fps":30,"duration_secs":1,"audio_layer":2,"audio_layer_id":"layer-abc"}"#,
        )
        .unwrap();
        assert!(matches!(
            export,
            WebAction::StartExport {
                audio_layer_id: Some(id),
                ntsc_quality: crate::ntsc::NtscExportQuality::LiveParity,
                shutter_samples: crate::render_export::ExportShutterSamples::Authored,
                ..
            } if id == "layer-abc"
        ));

        let native: WebAction = serde_json::from_str(
            r#"{"action":"start_export","width":1280,"height":720,"fps":30,"duration_secs":1,"ntsc_quality":"native"}"#,
        )
        .unwrap();
        let encoded = serde_json::to_value(&native).unwrap();
        assert_eq!(encoded["ntsc_quality"], "native");
        assert!(matches!(
            native,
            WebAction::StartExport {
                ntsc_quality: crate::ntsc::NtscExportQuality::Native,
                ..
            }
        ));
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"start_export","width":1280,"height":720,"fps":30,"duration_secs":1,"ntsc_quality":"unknown"}"#
        )
        .is_err());
        let exact_shutter: WebAction = serde_json::from_str(
            r#"{"action":"start_export","width":1280,"height":720,"fps":30,"duration_secs":1,"shutter_samples":"samples_16"}"#,
        )
        .unwrap();
        assert!(matches!(
            exact_shutter,
            WebAction::StartExport {
                shutter_samples: crate::render_export::ExportShutterSamples::Samples16,
                ..
            }
        ));
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"start_export","width":1280,"height":720,"fps":30,"duration_secs":1,"shutter_samples":"samples_2"}"#
        )
        .is_err());
    }

    #[test]
    fn curated_blend_actions_round_trip_every_exact_key_without_aliasing() {
        for blend_mode in crate::layers::BlendMode::ALL {
            let wire = serde_json::json!({
                "action": "set_layer_param",
                "index": 2,
                "layer_id": "17",
                "param": "blend_mode",
                "value": blend_mode.key(),
            });
            let action: WebAction = serde_json::from_value(wire).unwrap();
            assert!(matches!(
                &action,
                WebAction::SetLayerParam { index: 2, layer_id: Some(id), param, value }
                    if id == "17" && param == "blend_mode" && value.as_str() == Some(blend_mode.key())
            ));
            assert_eq!(
                serde_json::to_value(&action).unwrap()["value"],
                blend_mode.key(),
                "browser protocol collapsed {:?}",
                blend_mode
            );
            assert_eq!(
                action.coalesce_key().as_deref(),
                Some("layer:id:17:param:blend_mode")
            );
        }
    }

    #[test]
    fn spatial_transform_protocol_coalesces_absolute_fields_and_preserves_barriers() {
        let master: WebAction = serde_json::from_str(
            r#"{"action":"set_master_transform","param":"rotation_deg","value":12.5}"#,
        )
        .unwrap();
        assert!(matches!(
            &master,
            WebAction::SetMasterTransform { param, value }
                if param == "rotation_deg" && value.as_f64() == Some(12.5)
        ));
        assert_eq!(
            master.coalesce_key().as_deref(),
            Some("master:transform:rotation_deg")
        );

        let first: WebAction = serde_json::from_str(
            r#"{"action":"set_layer_transform","index":2,"layer_id":"17","param":"position_x","value":0.25}"#,
        )
        .unwrap();
        let latest: WebAction = serde_json::from_str(
            r#"{"action":"set_layer_transform","index":0,"layer_id":"17","param":"position_x","value":0.75}"#,
        )
        .unwrap();
        assert_eq!(first.coalesce_key(), latest.coalesce_key());

        let reset = WebAction::ResetLayerTransform {
            index: 0,
            layer_id: Some("17".into()),
        };
        let apply = WebAction::ApplyLayerTransform {
            index: 0,
            layer_id: Some("17".into()),
            transform: SpatialTransform::default(),
        };
        assert_eq!(reset.coalesce_key(), None);
        assert_eq!(apply.coalesce_key(), None);

        let mut coalesced = Vec::new();
        assert_eq!(
            enqueue_bounded(&mut coalesced, first.clone()),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(&mut coalesced, latest.clone()),
            EnqueueOutcome::Coalesced
        );
        assert!(matches!(
            coalesced.as_slice(),
            [WebAction::SetLayerTransform { value, .. }] if value.as_f64() == Some(0.75)
        ));

        let mut ordered = Vec::new();
        enqueue_bounded(&mut ordered, first);
        enqueue_bounded(&mut ordered, reset);
        enqueue_bounded(&mut ordered, latest);
        enqueue_bounded(&mut ordered, apply);
        assert!(matches!(
            ordered.as_slice(),
            [
                WebAction::SetLayerTransform { .. },
                WebAction::ResetLayerTransform { .. },
                WebAction::SetLayerTransform { .. },
                WebAction::ApplyLayerTransform { .. },
            ]
        ));
    }

    #[test]
    fn spatial_every_field_sentinel_round_trips_browser_edits_and_atomic_actions() {
        use crate::spatial::{EdgeMode, SamplingMode};

        // Keep every scalar distinct. A missing/wrong field mapping can then
        // neither hide behind a default nor accidentally compare equal to a
        // neighbouring component.
        let sentinel = SpatialTransform {
            position: [-0.75, 0.625],
            scale: [-1.25, 1.75],
            anchor: [0.2, 0.8],
            rotation_deg: 37.0,
            skew_deg: -11.0,
            skew_axis_deg: 73.0,
            fit: FitMode::Fill,
            crop: [0.05, 0.1, 0.15, 0.2],
            edge: EdgeMode::Mirror,
            sampling: SamplingMode::Nearest,
        };
        let edits = [
            ("position_x", serde_json::json!(-0.75)),
            ("position_y", serde_json::json!(0.625)),
            ("scale_x", serde_json::json!(-1.25)),
            ("scale_y", serde_json::json!(1.75)),
            ("anchor_x", serde_json::json!(0.2)),
            ("anchor_y", serde_json::json!(0.8)),
            ("rotation_deg", serde_json::json!(37.0)),
            ("skew_deg", serde_json::json!(-11.0)),
            ("skew_axis_deg", serde_json::json!(73.0)),
            ("fit", serde_json::json!("fill")),
            ("crop_left", serde_json::json!(0.05)),
            ("crop_top", serde_json::json!(0.1)),
            ("crop_right", serde_json::json!(0.15)),
            ("crop_bottom", serde_json::json!(0.2)),
            ("edge", serde_json::json!("mirror")),
            ("sampling", serde_json::json!("nearest")),
        ];

        let mut edited = SpatialTransform::default();
        for (param, value) in &edits {
            assert!(
                crate::App::apply_spatial_transform_edit(&mut edited, param, value),
                "browser edit mapping rejected canonical field {param}"
            );
        }
        assert_eq!(edited, sentinel);

        let actions = [
            WebAction::ApplyMasterTransform {
                transform: sentinel,
            },
            WebAction::ApplyLayerTransform {
                // Deliberately stale positional context: stable identity is
                // the only authoritative selector at engine ingress.
                index: usize::MAX,
                layer_id: Some("71".into()),
                transform: sentinel,
            },
        ];
        for action in actions {
            let wire = serde_json::to_string(&action).unwrap();
            let decoded: WebAction = serde_json::from_str(&wire).unwrap();
            let restored = match decoded {
                WebAction::ApplyMasterTransform { transform } => transform,
                WebAction::ApplyLayerTransform {
                    index,
                    layer_id,
                    transform,
                } => {
                    assert_eq!(index, usize::MAX);
                    assert_eq!(layer_id.as_deref(), Some("71"));
                    transform
                }
                _ => panic!("atomic spatial action changed variant"),
            };
            assert_eq!(restored, sentinel);
        }

        // Static coverage couples the table above to both the numerical
        // master controls and the data-driven layer-control generator.
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js").replace("\r\n", "\n");
        let range_specs = js
            .split_once("const TRANSFORM_RANGE_SPECS = [")
            .and_then(|(_, rest)| rest.split_once("];"))
            .map(|(table, _)| table)
            .expect("transform range table");
        for param in [
            "position_x",
            "position_y",
            "scale_x",
            "scale_y",
            "anchor_x",
            "anchor_y",
            "rotation_deg",
            "skew_deg",
            "skew_axis_deg",
            "crop_left",
            "crop_top",
            "crop_right",
            "crop_bottom",
        ] {
            assert!(
                range_specs.contains(&format!("['{param}',")),
                "layer control generator omitted {param}"
            );
            assert!(
                html.contains(&format!("data-master-transform=\"{param}\"")),
                "master numerical controls omitted {param}"
            );
            assert!(
                js.contains(&format!("{param}:")),
                "browser transform projection omitted {param}"
            );
        }
        for param in ["fit", "edge", "sampling"] {
            assert!(html.contains(&format!("data-master-transform=\"{param}\"")));
            assert!(js.contains(&format!("data-layer-transform=\"{param}\"")));
            assert!(js.contains(&format!("{param}: t.{param}")));
        }
    }

    /// The B14 latch publishes `sync_damaged` as renderer truth, and the
    /// panel must actually surface it. A fact wired to the DOM and then left
    /// invisible is the same as not publishing it, so this pins the styling
    /// rule *and* the text swap — the state must be readable without relying
    /// on colour perception.
    /// The panel carries a control's default in two places: the value span
    /// authored in `index.html`, which is what the operator first sees, and
    /// the JavaScript defaults table, which is what a double-click reset
    /// actually sends. An unlisted key falls back to the slider's *minimum*,
    /// so whenever a default is not the minimum the two silently disagree and
    /// a reset authors the wrong value — which is exactly what B14's
    /// `sync_bias` did, resetting to -1 instead of 0.
    ///
    /// This compares the two sources for every temporal row rather than
    /// pinning a hand list, so the whole class is caught for future tranches
    /// too.
    /// Read one attribute value out of a fragment of the served markup.
    fn attribute(fragment: &str, name: &str) -> Option<String> {
        let needle = format!("{name}=\"");
        let start = fragment.find(&needle)? + needle.len();
        let rest = &fragment[start..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    }

    #[test]
    fn every_panel_default_agrees_between_the_markup_and_the_reset_table() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");

        let table_for = |name: &str| -> &str {
            js.split(&format!("const {name} = Object.freeze({{"))
                .nth(1)
                .and_then(|chunk| chunk.split("});").next())
                .unwrap_or_else(|| panic!("the {name} table must exist"))
        };
        // The tables pack several entries per line and carry comments, so parse
        // comma-separated pairs rather than lines. The `:` guard stops a short
        // key matching the prefix of a longer one.
        let listed_in = |table: &str, param: &str| -> Option<f64> {
            table
                .lines()
                .map(|line| line.split("//").next().unwrap_or(""))
                .flat_map(|line| line.split(','))
                .find_map(|chunk| {
                    let rest = chunk.trim().strip_prefix(param)?.strip_prefix(':')?;
                    rest.trim().parse::<f64>().ok()
                })
        };

        let families = [
            ("data-param", "MASTER_PARAM_DEFAULTS"),
            ("data-temporal", "TEMPORAL_PARAM_DEFAULTS"),
            ("data-ntsc", "NTSC_PARAM_DEFAULTS"),
        ];

        let mut compared = 0usize;
        let mut disagreements: Vec<String> = Vec::new();
        for row in html.split("<div class=\"param-row\"").skip(1) {
            let row = match row.find("</div>") {
                Some(end) => &row[..end],
                None => continue,
            };
            let Some((attr, table_name)) = families
                .iter()
                .find(|(attr, _)| attribute(row, attr).is_some())
            else {
                continue;
            };
            let table = table_for(table_name);
            let Some(param) = attribute(row, attr) else {
                continue;
            };
            // Only range rows have a reset fallback at all.
            if !row.contains("<input type=\"range\">") {
                continue;
            }
            let Some(min) = attribute(row, "data-min").and_then(|v| v.parse::<f64>().ok()) else {
                continue;
            };
            let Some(shown) = row
                .split("<span class=\"value\">")
                .nth(1)
                .and_then(|rest| rest.split('<').next())
                .map(|text| {
                    text.trim()
                        .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
                        .to_string()
                })
                .and_then(|text| text.parse::<f64>().ok())
            else {
                continue;
            };

            let effective = listed_in(table, &param).unwrap_or(min);
            if (effective - shown).abs() >= 1.0e-6 {
                disagreements.push(format!(
                    "{attr} {param}: panel shows {shown}, reset would send {effective} (min {min})"
                ));
            }
            compared += 1;
        }
        assert!(
            compared > 100,
            "only {compared} rows were compared; the parser stopped matching the markup"
        );
        assert!(
            disagreements.is_empty(),
            "these controls reset to the wrong value; add them to their \
             defaults table:\n  {}",
            disagreements.join("\n  ")
        );
    }

    /// B15's search and filters are a *view* over data the panel already
    /// holds. The load-bearing claim is that they cost the engine nothing:
    /// no wire action, no round trip, and no work at all on a state packet
    /// while no filter is engaged.
    #[test]
    fn control_search_is_client_side_accessible_and_sends_nothing() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        let css = include_str!("../../static/style.css");

        for contract in [
            "id=\"control-search\"",
            "id=\"filter-moving\"",
            "id=\"filter-changed\"",
            "id=\"control-search-count\"",
            "role=\"search\"",
            "aria-pressed=\"false\"",
            "aria-label=\"Search master controls by name, section, or help text\"",
        ] {
            assert!(
                html.contains(contract),
                "missing search contract: {contract}"
            );
        }
        assert!(
            css.contains(".control-hidden"),
            "the filter needs a hiding rule"
        );
        assert!(
            css.contains(".control-filter[aria-pressed=\"true\"]"),
            "an engaged filter must look engaged"
        );

        // The whole feature lives in one block, and that block must never
        // dispatch. If search ever needs the engine, this test is the place
        // that decision gets made deliberately.
        // Bound the block at the next section banner: everything after it
        // belongs to another feature, and the no-dispatch claim is about this
        // one.
        let block = js
            .split("// ===== B15: control search, filters, and help")
            .nth(1)
            .and_then(|rest| {
                rest.split(
                    "
// =====",
                )
                .next()
            })
            .expect("the search engine block must exist");
        assert!(
            !block.contains("sendAction("),
            "control search must not dispatch a wire action"
        );
        assert!(
            !block.contains("ws.send"),
            "control search must not touch the socket"
        );

        // The three transcribed target rules, and nothing more: an unmappable
        // target must light nothing rather than light the wrong row.
        assert!(block.contains("if (target === param) return true;"));
        assert!(block.contains("target === `temporal_${param}`"));
        assert!(block.contains("`display_${param.slice(5)}`"));

        // Idle cost: a state packet does nothing unless a filter is engaged.
        assert!(
            block.contains("if (CONTROL_SEARCH.changed || (changed && CONTROL_SEARCH.moving)) applyControlFilter();"),
            "an idle filter must not walk the DOM on every 30 Hz packet"
        );
        // Hidden, never disabled.
        assert!(block.contains("classList.toggle('control-hidden'"));
        // The slash shortcut yields to any field the operator is typing in.
        assert!(block.contains("if (event.key !== '/'"));
        assert!(block.contains("active.isContentEditable"));

        // Help reaches the rows as native tooltips, from the generated table.
        assert!(block.contains("window.CONTROL_HELP"));
        assert!(block.contains("row.title = help"));

        // The filter is wired to the packet the panel already receives.
        assert!(
            js.contains("syncControlFilters(msg.modulation);"),
            "MOVING is derived from the compiled route table in the snapshot"
        );
    }

    /// The panel's chrome: the program icon at every size a browser and a
    /// desktop ask for, the search sharing the transport ribbon rather than a
    /// line of its own, Scenes in the left column, and the static prose folded
    /// into a `?` beside each section.
    #[test]
    fn panel_chrome_carries_the_icon_the_ribbon_search_and_the_folded_notes() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        let css = include_str!("../../static/style.css");
        let files = include_str!("static_files.rs");

        // The icon is served at every provided size and linked from the page.
        for size in ["16", "32", "48", "256"] {
            assert!(
                files.contains(&format!("\"icon-{size}.png\"")),
                "icon-{size}.png must be served"
            );
            assert!(
                html.contains(&format!("href=\"icon-{size}.png\"")),
                "icon-{size}.png must be linked from the panel"
            );
        }
        assert!(
            files.contains("image/png"),
            "icons must be served with an image content type"
        );
        assert!(
            !html.contains("data:image/svg+xml"),
            "the placeholder inline favicon should be gone"
        );

        // The search shares the transport ribbon. Its markup must sit inside
        // that row, and it must keep its own landmark while doing so.
        let ribbon = html
            .split("<div class=\"transport-row\">")
            .nth(1)
            .and_then(|rest| rest.split("</div>\n        <h3").next())
            .expect("the transport row must exist");
        assert!(
            ribbon.contains("id=\"control-search\""),
            "the search belongs in the transport ribbon, not on a line of its own"
        );
        assert!(
            ribbon.contains("role=\"search\""),
            "moving the search must not cost it its landmark"
        );
        assert!(
            css.contains(".transport-row .control-search {"),
            "the ribbon search needs its own layout rule"
        );
        assert!(
            css.contains("display: contents;"),
            "the wrapper must not become a nested box inside the ribbon"
        );

        // Scenes lives in the left column, above the library grid.
        let left = html
            .split("<aside class=\"col-left\">")
            .nth(1)
            .and_then(|rest| rest.split("</aside>").next())
            .expect("the left column must exist");
        assert!(
            left.contains("class=\"scenes-panel\""),
            "Scenes belongs in the left column"
        );
        let scenes_at = left.find("class=\"scenes-panel\"").expect("scenes");
        let grid_at = left.find("id=\"library-grid\"").expect("library grid");
        assert!(
            scenes_at < grid_at,
            "Scenes should sit above the library grid, which is the long scrolling surface"
        );
        // The Library heading and its + upload control label the grid, so they
        // belong immediately above it rather than at the top of a column whose
        // other sections have nothing to do with the library.
        let title_at = left
            .find("<h3 class=\"panel-title\">Library")
            .expect("the library title must exist");
        let upload_at = left
            .find("class=\"lib-upload-btn\"")
            .expect("the library upload control must exist");
        assert!(
            scenes_at < title_at && title_at < grid_at,
            "the Library heading belongs directly above the grid it names"
        );
        assert!(
            title_at < upload_at && upload_at < grid_at,
            "the + upload control stays inside the Library heading, above the grid"
        );

        // Static prose folds into a ? beside its section; live notes do not.
        assert!(
            js.contains(".audio-status:not([id])"),
            "only notes without an id are static description"
        );
        assert!(js.contains("mark.className = 'group-help'"));
        assert!(
            js.contains("event.stopPropagation();"),
            "the question mark must not collapse the section it explains"
        );
        assert!(css.contains(".group-help {"), "the ? needs a styling rule");
        assert!(
            css.contains(".audio-status:empty"),
            "an empty live note must not reserve a blank line"
        );
        // Every note the engine writes into keeps its id, and therefore its place.
        for live in [
            "id=\"ntsc-metrics\"",
            "id=\"morph-status\"",
            "id=\"reroll-status\"",
            "id=\"sync-latch-status\"",
        ] {
            assert!(html.contains(live), "live status note missing: {live}");
        }
    }

    /// The generated help asset must actually be reachable, or the panel's
    /// tooltips and its search corpus are both empty.
    #[test]
    fn the_generated_help_asset_is_served() {
        let files = include_str!("static_files.rs");
        assert!(
            files.contains("\"help.js\" =>"),
            "help.js must be served alongside the other panel assets"
        );
        assert!(
            files.contains("crate::control_help::panel_javascript"),
            "the served help must be generated from the Rust table, never a second copy"
        );
    }

    #[test]
    fn the_sync_latch_damage_state_is_visible_and_not_colour_only() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        let css = include_str!("../../static/style.css");

        assert!(
            html.contains("id=\"sync-latch-status\""),
            "the sync latch needs a status region to report damage into"
        );
        assert!(
            css.contains(".audio-status.sync-damaged"),
            "the damage class must carry a styling rule, or the published fact is invisible"
        );
        // Colour is the secondary cue; the text is the primary one.
        assert!(
            js.contains("DAMAGE HELD"),
            "the damage state must be readable as text, not colour alone"
        );
        assert!(
            js.contains("syncStatus.dataset.baseText"),
            "the help text must be restored when the damage clears"
        );
        for contract in [
            "sync_amount: syncLatch.amount",
            "sync_latched: syncLatch.latched",
            "const syncLatch = t.sync || {}",
        ] {
            assert!(
                js.contains(contract),
                "missing sync latch binding: {contract}"
            );
        }
        for contract in [
            "data-temporal=\"sync_amount\"",
            "data-temporal=\"sync_rate\"",
            "data-temporal=\"sync_spread\"",
            "data-temporal=\"sync_bias\"",
            "data-temporal=\"sync_latched\"",
        ] {
            assert!(
                html.contains(contract),
                "missing sync latch row: {contract}"
            );
        }
        // The switch is a checkbox and carries its own accessible label.
        assert!(
            html.contains(
                "aria-label=\"Latch horizontal sync faults instead of letting them heal\""
            ),
            "the latch switch needs an accessible label"
        );
    }

    #[test]
    fn spatial_transform_browser_contract_is_complete_accessible_and_stable_id_based() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        let css = include_str!("../../static/style.css");
        for contract in [
            "id=\"master-transform-group\"",
            "data-master-transform=\"position_x\"",
            "data-master-transform=\"skew_axis_deg\"",
            "data-master-transform=\"crop_bottom\"",
            "class=\"transform-scale-link\"",
            "class=\"transform-preset\"",
            "role=\"status\" aria-live=\"polite\"",
            "value=\"transparent\" selected",
            "id=\"new-layer-fit\"",
            "Host-session preference for future file, still, and Spout layers",
        ] {
            assert!(
                html.contains(contract),
                "missing master transform contract: {contract}"
            );
        }
        for contract in [
            "const LEGACY_SPATIAL_TRANSFORM",
            "function normalizeSpatialTransform(raw)",
            "function layerTransformControlsHtml(index)",
            "action: 'set_master_transform'",
            "action: 'reset_master_transform'",
            "action: 'apply_master_transform'",
            "action: 'set_layer_transform'",
            "action: 'reset_layer_transform'",
            "action: 'apply_layer_transform'",
            "...currentLayerSelector(card, layer, index)",
            "syncTransformPanel(card.querySelector('.layer-transform-body'), layer.transform)",
            "if (!control || !canSync(control)) return",
            "'set_layer_transform'",
            "action: 'set_new_layer_fit'",
            "syncNewLayerFit(msg.new_layer_fit)",
        ] {
            assert!(
                js.contains(contract),
                "missing transform JS contract: {contract}"
            );
        }
        let master_targets = js
            .split_once("const MOD_TARGETS = [")
            .and_then(|(_, rest)| rest.split_once("];"))
            .map(|(targets, _)| targets)
            .expect("master modulation target table");
        let layer_targets = js
            .split_once("const LAYER_FX_TARGETS = [")
            .and_then(|(_, rest)| rest.split_once("];"))
            .map(|(targets, _)| targets)
            .expect("layer modulation target table");
        for target in [
            "position_x",
            "position_y",
            "scale_x",
            "scale_y",
            "anchor_x",
            "anchor_y",
            "rotation_deg",
            "skew_deg",
            "skew_axis_deg",
            "crop_left",
            "crop_top",
            "crop_right",
            "crop_bottom",
        ] {
            let needle = format!("['{target}',");
            assert!(
                master_targets.contains(&needle),
                "master target menu omitted {target}"
            );
            assert!(
                layer_targets.contains(&needle),
                "layer target menu omitted {target}"
            );
        }
        for contract in [
            ".layer-transform-body[hidden]",
            ".transform-toolbar",
            ".transform-status",
        ] {
            assert!(
                css.contains(contract),
                "missing transform CSS contract: {contract}"
            );
        }
    }

    #[test]
    fn new_layer_fit_is_a_defaulted_host_preference_and_coalesced_action() {
        let legacy: AppSnapshot = serde_json::from_value(serde_json::json!({
            "type": "state",
            "effects": EffectsSnapshot::default(),
            "ntsc": NtscSnapshot::default(),
            "layers": [],
            "library": [],
            "paused": false
        }))
        .expect("legacy snapshot");
        assert_eq!(legacy.new_layer_fit, FitMode::Fit);

        let action: WebAction =
            serde_json::from_str(r#"{"action":"set_new_layer_fit","fit":"fill"}"#)
                .expect("new-layer preference action");
        assert!(matches!(
            action,
            WebAction::SetNewLayerFit { fit: FitMode::Fill }
        ));
        assert_eq!(action.coalesce_key().as_deref(), Some("host:new-layer-fit"));
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_new_layer_fit","fit":"diagonal"}"#
        )
        .is_err());
    }

    #[test]
    fn export_panel_exposes_safe_ntsc_and_exact_shutter_choices() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        assert!(html.contains("id=\"export-ntsc-quality\""));
        assert!(html.contains("value=\"live_parity\" selected"));
        assert!(html.contains("value=\"native\""));
        assert!(html.contains("id=\"export-shutter-samples\""));
        for value in [
            "authored",
            "samples_1",
            "samples_4",
            "samples_8",
            "samples_16",
        ] {
            assert!(html.contains(&format!("value=\"{value}\"")));
        }
        assert!(html.contains("id=\"export-warnings\""));
        assert!(html.contains("id=\"export-warnings-list\""));
        assert!(js.contains("['live_parity', 'native'].includes(requestedNtscQuality)"));
        assert!(js.contains("ntsc_quality: ntscQuality"));
        assert!(js.contains("requestedShutterSamples"));
        assert!(js.contains(
            "['authored', 'samples_1', 'samples_4', 'samples_8', 'samples_16'].includes(requestedShutterSamples)"
        ));
        assert!(js.contains("shutter_samples: shutterSamples"));
        assert!(js.contains("msg.export_warnings"));
        assert!(js.contains("function syncExportWarnings(warnings = [])"));
        assert!(js.contains("item.textContent = message"));
        assert!(js.contains("if (warningsKey === exportWarningsKey) return"));
        assert!(js.contains(
            "scope.field_attached && scope.rendered_source_origin === 'lattice_fallback'"
        ));
        assert!(js.contains("scope.field_planned && !scope.field_attached"));
        assert!(js.contains("rendered fallback"));
        assert!(js.contains("unattached"));
    }

    #[test]
    fn dynamic_layer_controls_use_current_identity_and_explain_morph_ownership() {
        let js = include_str!("../../static/app.js");
        for contract in [
            "function currentLayerContext(card, fallbackLayer, fallbackIndex)",
            "function currentLayerSelector(card, fallbackLayer, fallbackIndex)",
            "card?._layerState || fallbackLayer",
            "card.dataset.index = String(index)",
            "...currentLayerSelector(card, layer, index)",
            "editing a captured control disengages A/B",
            "const moshSendId = `layer-mosh-send-${stableLayerToken}`",
            "data-param=\"mosh_send\"",
        ] {
            assert!(
                js.contains(contract),
                "missing dynamic layer contract: {contract}"
            );
        }
    }

    #[test]
    fn legacy_layer_snapshot_defaults_independent_bypasses_to_off() {
        let current = LayerSnapshot {
            layer_id: "17".into(),
            filename: "plate.png".into(),
            delivery: "legacy_rgba".into(),
            delivery_active_planar: false,
            visible: true,
            bypass_master_fx: true,
            bypass_temporal_fx: true,
            reroll_on_loop: false,
            paused: false,
            opacity: 1.0,
            mosh_send: 0.25,
            speed: 1.0,
            fps: 30.0,
            blend_mode: "normal".into(),
            progress: 0.0,
            key_mode: 0,
            key_threshold: 0.5,
            key_softness: 0.1,
            key_color: default_key_color(),
            key_tolerance: default_key_tolerance(),
            effects: EffectsSnapshot::default(),
            motion: MotionSnapshot::default(),
            transform: SpatialTransform::default(),
            source_kind: "image".into(),
            source_name: String::new(),
            source_active: true,
            source_width: 1920,
            source_height: 1080,
            source_sequence: 0,
            source_error: String::new(),
            offline_export_policy: String::new(),
            proxy_backing_prefix: String::new(),
            proxy_note: String::new(),
            performance: LayerPerformanceSnapshot::default(),
            pattern: None,
            text_page: None,
        };
        let mut value = serde_json::to_value(current).unwrap();
        // Empty proxy state stays off the wire entirely rather than shipping
        // two empty strings on every un-proxied layer.
        assert!(value.get("proxy_backing_prefix").is_none());
        assert!(value.get("proxy_note").is_none());
        value.as_object_mut().unwrap().remove("bypass_master_fx");
        value.as_object_mut().unwrap().remove("bypass_temporal_fx");
        value.as_object_mut().unwrap().remove("mosh_send");
        value.as_object_mut().unwrap().remove("reroll_on_loop");
        value.as_object_mut().unwrap().remove("transform");
        value.as_object_mut().unwrap().remove("motion");
        value.as_object_mut().unwrap().remove("performance");
        let legacy: LayerSnapshot = serde_json::from_value(value).unwrap();
        assert!(!legacy.bypass_master_fx);
        assert!(!legacy.bypass_temporal_fx);
        assert_eq!(legacy.mosh_send, 1.0);
        assert_eq!(legacy.motion, MotionSnapshot::default());
        assert!(!legacy.reroll_on_loop);
        assert_eq!(legacy.transform, SpatialTransform::default());
        assert_eq!(legacy.performance, LayerPerformanceSnapshot::default());
        assert!(legacy.proxy_backing_prefix.is_empty());
        assert!(legacy.proxy_note.is_empty());
    }

    #[test]
    fn study_document_action_is_engine_validated_coalescible_and_panel_wired() {
        // Wire shape: document is an arbitrary JSON value (null clears); the
        // engine, not the panel, owns validation and compilation.
        let action: WebAction = serde_json::from_str(
            r#"{"action":"set_visual_node_study_document","scope":{"scope":"master"},"node_id":"4","document":null}"#,
        )
        .unwrap();
        let WebAction::SetVisualNodeStudyDocument {
            node_id, document, ..
        } = &action
        else {
            panic!("expected the study document action");
        };
        assert_eq!(node_id, "4");
        assert!(document.is_null());
        // An absolute value per node: coalescible, so the newest paste wins,
        // and never quantized.
        assert_eq!(
            action.coalesce_key().as_deref(),
            Some("creative:master:node:4:study_document")
        );

        let js = include_str!("../../static/app.js");
        let start = js.find("const QUANTIZABLE_ACTIONS").unwrap();
        let end = js[start..].find("]);").unwrap() + start;
        assert!(!js[start..end].contains("set_visual_node_study_document"));
        // The panel pastes a document into the node card and can clear it;
        // parse failures stay client-side in a polite status region.
        assert!(js.contains("action: 'set_visual_node_study_document'"));
        assert!(js.contains("creative-study-document"));
        assert!(js.contains("Assign study document"));
        assert!(js.contains("creative-study-error"));
        assert!(js.contains("document: null"));
    }

    #[test]
    fn proxy_browser_surface_is_id_addressed_priority_and_never_quantizable() {
        // Wire shape: the stable ID is mandatory — there is no positional
        // field for this action at all, so a fallback cannot exist.
        let action: WebAction =
            serde_json::from_str(r#"{"action":"request_layer_proxy","layer_id":"42"}"#).unwrap();
        assert!(matches!(&action, WebAction::RequestLayerProxy { layer_id } if layer_id == "42"));
        assert!(serde_json::from_str::<WebAction>(r#"{"action":"request_layer_proxy"}"#).is_err());
        // A request is an event, not an absolute value: priority admission,
        // never coalesced with a later request.
        assert!(action.is_priority());
        assert_eq!(action.coalesce_key(), None);

        // The panel's hand-maintained quantizable set must not contain it.
        let js = include_str!("../../static/app.js");
        let start = js.find("const QUANTIZABLE_ACTIONS").unwrap();
        let end = js[start..].find("]);").unwrap() + start;
        assert!(!js[start..end].contains("request_layer_proxy"));
        // The panel sends it stable-ID-only with an accessible affordance,
        // and the status region mirrors the engine-owned note.
        assert!(js.contains("action: 'request_layer_proxy'"));
        assert!(js.contains("layer-proxy-btn"));
        assert!(js.contains("Encode proxy for layer"));
        assert!(js.contains("currentStableLayerId(card, layer, index)"));
        assert!(js.contains("layer-proxy-status"));
        assert!(js.contains("proxy_backing_prefix"));
    }

    #[test]
    fn downsample_snapshot_and_browser_control_are_complete() {
        let mut uniforms = EffectUniforms {
            downsample: 0.42,
            ..EffectUniforms::default()
        };
        let snapshot = EffectsSnapshot::from_uniforms(&uniforms);
        assert!((snapshot.downsample - 0.42).abs() < f32::EPSILON);
        let mut changed = snapshot.clone();
        changed.apply_param("downsample", &serde_json::json!(0.25));
        changed.apply_to_uniforms(&mut uniforms);
        assert!((uniforms.downsample - 0.25).abs() < f32::EPSILON);
        assert!(include_str!("../../static/index.html").contains("data-param=\"downsample\""));
        assert!(include_str!("../../static/app.js").contains("'set_layer_effect'"));
    }

    #[test]
    fn cellular_snapshot_controls_and_modulation_targets_are_complete() {
        let mut uniforms = EffectUniforms {
            cellular_amount: 0.7,
            cellular_scale: 18.0,
            cellular_warp: 0.6,
            cellular_speed: 1.25,
            cellular_gap_amount: 0.85,
            cellular_gap_threshold: 0.45,
            cellular_gap_softness: 0.14,
            ..EffectUniforms::default()
        };
        let snapshot = EffectsSnapshot::from_uniforms(&uniforms);
        assert!((snapshot.cellular_amount - 0.7).abs() < f32::EPSILON);
        assert!((snapshot.cellular_scale - 18.0).abs() < f32::EPSILON);
        assert!((snapshot.cellular_warp - 0.6).abs() < f32::EPSILON);
        assert!((snapshot.cellular_speed - 1.25).abs() < f32::EPSILON);
        assert!((snapshot.cellular_gap_amount - 0.85).abs() < f32::EPSILON);
        assert!((snapshot.cellular_gap_threshold - 0.45).abs() < f32::EPSILON);
        assert!((snapshot.cellular_gap_softness - 0.14).abs() < f32::EPSILON);

        let mut changed = EffectsSnapshot::default();
        for (param, value) in [
            ("cellular_amount", 0.5),
            ("cellular_scale", 24.0),
            ("cellular_warp", 0.8),
            ("cellular_speed", 1.5),
            ("cellular_gap_amount", 0.75),
            ("cellular_gap_threshold", 0.6),
            ("cellular_gap_softness", 0.11),
        ] {
            changed.apply_param(param, &serde_json::json!(value));
        }
        changed.apply_to_uniforms(&mut uniforms);
        assert!((uniforms.cellular_amount - 0.5).abs() < f32::EPSILON);
        assert!((uniforms.cellular_scale - 24.0).abs() < f32::EPSILON);
        assert!((uniforms.cellular_warp - 0.8).abs() < f32::EPSILON);
        assert!((uniforms.cellular_speed - 1.5).abs() < f32::EPSILON);
        assert!((uniforms.cellular_gap_amount - 0.75).abs() < f32::EPSILON);
        assert!((uniforms.cellular_gap_threshold - 0.6).abs() < f32::EPSILON);
        assert!((uniforms.cellular_gap_softness - 0.11).abs() < f32::EPSILON);

        let mut legacy_json = serde_json::to_value(EffectsSnapshot::default()).unwrap();
        let legacy_object = legacy_json.as_object_mut().unwrap();
        legacy_object.remove("cellular_gap_amount");
        legacy_object.remove("cellular_gap_threshold");
        legacy_object.remove("cellular_gap_softness");
        let legacy: EffectsSnapshot = serde_json::from_value(legacy_json).unwrap();
        assert_eq!(legacy.cellular_gap_amount, 0.0);
        assert_eq!(legacy.cellular_gap_threshold, 0.65);
        assert_eq!(legacy.cellular_gap_softness, 0.08);

        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        for field in [
            "cellular_amount",
            "cellular_scale",
            "cellular_warp",
            "cellular_speed",
        ] {
            assert!(html.contains(&format!("data-param=\"{field}\"")));
            assert!(js.contains(&format!("'{field}'")));
        }
        for field in [
            "cellular_gap_amount",
            "cellular_gap_threshold",
            "cellular_gap_softness",
        ] {
            assert!(html.contains(&format!("data-param=\"{field}\"")));
            assert!(js.contains(&format!("['{field}',")));
        }
        assert!(html.contains("keyed ridges can reveal lower content"));
        assert!(js.contains("'Master Cell Gap Key'"));
        assert!(html.contains("data-group=\"cellular\""));
        assert!(js.contains("class=\"layer-cellular-toggle\""));
        assert!(js.contains("aria-controls=\"layer-cellular-body-${index}\""));
        assert!(js.contains("class=\"layer-cellular-body\""));
        for contract in [
            "const MASTER_MOD_TARGETS = MOD_TARGETS.slice()",
            "const liveLayerCount = latestLayerIdentities.length",
            "targets.push([`layer${layer}_${suffix}`, `L${layer} ${label}`])",
            "groups.push([`Layer ${layer}`, targets])",
            "const layerMatch = /^layer([1-9]\\d*)_/.exec(target)",
            "routes: routings.map((routing, index)",
        ] {
            assert!(
                js.contains(contract),
                "missing dynamic target menu contract: {contract}"
            );
        }
        assert!(!js.contains("MAX_MOD_LAYERS"));
        assert!(!js.contains("Math.min(MAX_MOD_LAYERS"));
        assert!(!js.contains("/^layer([1-9]|1[0-6])_/"));
    }

    #[test]
    fn small_effects_snapshot_round_trips_and_sanitizes() {
        use crate::effects::EffectUniforms;

        let uniforms = EffectUniforms {
            contour: 0.8,
            flatten_levels: 7.0,
            negative_mode: 2.0,
            colourpass_hue: -120.0,
            bitcrush_levels: 6.0,
            multi_grid_x: 4.0,
            barrel: -0.5,
            anamorphic_streak: 0.3,
            ..EffectUniforms::default()
        };
        let snapshot = EffectsSnapshot::from_uniforms(&uniforms);
        assert_eq!(snapshot.contour, 0.8);
        assert_eq!(snapshot.flatten_levels, 7.0);
        assert_eq!(snapshot.negative_mode, 2);
        assert_eq!(snapshot.colourpass_hue, -120.0);
        assert_eq!(snapshot.bitcrush_levels, 6.0);
        assert_eq!(snapshot.multi_grid_x, 4.0);
        assert_eq!(snapshot.barrel, -0.5);
        assert_eq!(snapshot.anamorphic_streak, 0.3);

        let mut restored = EffectUniforms::default();
        snapshot.apply_to_uniforms(&mut restored);
        assert_eq!(
            bytemuck::bytes_of(&restored),
            bytemuck::bytes_of(&uniforms),
            "the snapshot round trip must be lossless"
        );

        // Wire edits land through the shared apply_param path.
        let mut edited = EffectsSnapshot::default();
        edited.apply_param("halftone", &serde_json::json!(0.7));
        edited.apply_param("negative_mode", &serde_json::json!(1));
        edited.apply_param("multi_grid_y", &serde_json::json!(3.0));
        assert_eq!(edited.halftone, 0.7);
        assert_eq!(edited.negative_mode, 1);
        assert_eq!(edited.multi_grid_y, 3.0);

        // Hostile values sanitize to the neutral value or clamp.
        let hostile = EffectsSnapshot {
            contour_bands: f32::NAN,
            bitcrush_levels: 99.0,
            barrel: f32::INFINITY,
            ..EffectsSnapshot::default()
        };
        let mut sanitized = EffectUniforms::default();
        hostile.apply_to_uniforms(&mut sanitized);
        assert_eq!(sanitized.contour_bands, 10.0);
        assert_eq!(sanitized.bitcrush_levels, 16.0);
        assert_eq!(sanitized.barrel, 0.0);

        // An older snapshot without the B13 fields decodes to the exact
        // prior path.
        let legacy: EffectsSnapshot = serde_json::from_str(
            r#"{"pixelate":1.0,"downsample":1.0,"rgb_split":0.0,"hue_shift":0.0,
                "saturation":0.0,"brightness":0.0,"contrast":0.0,"posterize":0.0,
                "invert":false,"grain_intensity":0.0,"grain_size":1.0,
                "grain_algo":0,"color_grain":false,"vignette":0.0,
                "color_drift":0.0,"breathe_scale":0.0,"breathe_rotation":0.0,
                "breathe_position":0.0}"#,
        )
        .unwrap();
        assert_eq!(legacy.contour, 0.0);
        assert_eq!(legacy.contour_bands, 10.0);
        assert_eq!(legacy.bitcrush_dither, 1.0);
        assert_eq!(legacy.multi_grid_x, 1.0);
        assert_eq!(legacy.negative_mode, 0);
        assert_eq!(legacy.barrel, 0.0);
    }

    #[test]
    fn shift_snapshot_controls_and_legacy_defaults_are_complete() {
        let mut uniforms = EffectUniforms {
            shift_amount: 0.7,
            shift_block_size: 40.0,
            shift_density: 0.6,
            shift_speed: 8.25,
            ..EffectUniforms::default()
        };
        let snapshot = EffectsSnapshot::from_uniforms(&uniforms);
        assert_eq!(snapshot.shift_amount, 0.7);
        assert_eq!(snapshot.shift_block_size, 40.0);
        assert_eq!(snapshot.shift_density, 0.6);
        assert_eq!(snapshot.shift_speed, 8.25);

        let mut changed = EffectsSnapshot::default();
        for (param, value) in [
            ("shift_amount", 0.5),
            ("shift_block_size", 32.0),
            ("shift_density", 0.8),
            ("shift_speed", 7.5),
        ] {
            changed.apply_param(param, &serde_json::json!(value));
        }
        changed.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.shift_amount, 0.5);
        assert_eq!(uniforms.shift_block_size, 32.0);
        assert_eq!(uniforms.shift_density, 0.8);
        assert_eq!(uniforms.shift_speed, 7.5);

        let mut legacy_json = serde_json::to_value(EffectsSnapshot::default()).unwrap();
        let legacy_object = legacy_json.as_object_mut().unwrap();
        for field in [
            "shift_amount",
            "shift_block_size",
            "shift_density",
            "shift_speed",
        ] {
            legacy_object.remove(field);
        }
        let legacy: EffectsSnapshot = serde_json::from_value(legacy_json).unwrap();
        assert_eq!(legacy.shift_amount, 0.0);
        assert_eq!(legacy.shift_block_size, 8.0);
        assert_eq!(legacy.shift_density, 0.5);
        assert_eq!(legacy.shift_speed, 3.0);

        let invalid = EffectsSnapshot {
            shift_amount: f32::NAN,
            shift_block_size: -4.0,
            shift_density: 9.0,
            shift_speed: f32::INFINITY,
            ..EffectsSnapshot::default()
        };
        invalid.apply_to_uniforms(&mut uniforms);
        assert_eq!(uniforms.shift_amount, 0.0);
        assert_eq!(uniforms.shift_block_size, 2.0);
        assert_eq!(uniforms.shift_density, 1.0);
        assert_eq!(uniforms.shift_speed, 3.0);

        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        for field in [
            "shift_amount",
            "shift_block_size",
            "shift_density",
            "shift_speed",
        ] {
            assert!(html.contains(&format!("data-param=\"{field}\"")));
            assert!(js.contains(&format!("'{field}'")));
        }
        assert!(html.contains("data-group=\"shift\""));
        assert!(js.contains("'set_layer_effect'"));
    }

    #[test]
    fn master_control_groups_stay_in_their_intended_columns() {
        let html = include_str!("../../static/index.html");
        let video = html.find("data-fx-column=\"video\"").unwrap();
        let mod_morph = html.find("data-fx-column=\"mod-morph\"").unwrap();
        let sources = html.find("data-fx-column=\"sources\"").unwrap();
        let io = html.find("data-fx-column=\"io\"").unwrap();
        assert!(video < mod_morph && mod_morph < sources && sources < io);

        let video_column = &html[video..mod_morph];
        assert!(video_column.contains("data-group=\"digital\""));
        assert!(video_column.contains("data-group=\"analog\""));
        assert!(video_column.contains("id=\"vhs-group\""));
        assert!(!video_column.contains("data-group=\"cellular\""));

        let second_column = &html[mod_morph..sources];
        let ordered = [
            "id=\"morph-group\"",
            "class=\"fx-group\" data-group=\"cellular\"",
            "class=\"fx-group\" data-group=\"motion\"",
            "id=\"motion-fields-group\"",
            "id=\"temporal-group\"",
            "id=\"gesture-group\"",
            "id=\"performance-group\"",
            "id=\"audio-group\"",
        ];
        let mut previous = 0;
        for marker in ordered {
            let position = second_column.find(marker).unwrap();
            assert!(position >= previous, "{marker} is out of column order");
            previous = position;
        }
        // 7 -> 8 when B9 added the TAKE RECORDER group beside the gesture
        // field it composes with.
        assert_eq!(second_column.matches("<div class=\"fx-group\"").count(), 8);
        assert!(!second_column.contains("id=\"mod-group\""));
        assert!(!second_column.contains("id=\"pad-group\""));
        assert!(!second_column.contains("id=\"midi-group\""));

        let third_column = &html[sources..io];
        let ordered = [
            "id=\"mod-group\"",
            "id=\"pad-group\"",
            "id=\"perform-group\"",
            "id=\"midi-group\"",
        ];
        let mut previous = 0;
        for marker in ordered {
            let position = third_column.find(marker).unwrap();
            assert!(position >= previous, "{marker} is out of column order");
            previous = position;
        }
        // 3 -> 4 when B10 added the PERFORM SOURCES group (bend pads,
        // envelopes, macros, generator seed) beside the matrix they feed.
        assert_eq!(third_column.matches("<div class=\"fx-group\"").count(), 4);
        assert!(!third_column.contains("id=\"audio-group\""));
        assert!(!third_column.contains("id=\"gyro-group\""));

        assert!(html.contains(
            "data-fx-column=\"mod-morph\"><!-- morph, time-domain effects, and audio -->"
        ));
        assert!(html
            .contains("data-fx-column=\"sources\"><!-- live modulation controls and sources -->"));
        for group in ["mod", "pad", "perform", "midi", "audio"] {
            assert_eq!(html.matches(&format!("id=\"{group}-group\"")).count(), 1);
        }

        let io_column = &html[io..];
        let remote = io_column.find("id=\"remote-group\"").unwrap();
        let gyro = io_column.find("id=\"gyro-group\"").unwrap();
        assert!(remote < gyro);
        assert_eq!(html.matches("id=\"gyro-group\"").count(), 1);
        assert_eq!(html.matches("id=\"temporal-group\"").count(), 1);
    }

    #[test]
    fn quick_patch_capture_uses_an_authoritative_snapshot_status() {
        let action: WebAction = serde_json::from_str(r#"{"action":"quick_save_patch"}"#).unwrap();
        assert!(matches!(action, WebAction::QuickSavePatch));
        let value = serde_json::to_value(WebAction::QuickSavePatch).unwrap();
        assert_eq!(value["action"], "quick_save_patch");

        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        assert!(html.contains("id=\"patch-capture\""));
        assert!(html.contains("id=\"patch-save-status\" role=\"status\" aria-live=\"polite\""));
        assert!(js.contains("sendAction({ action: 'quick_save_patch' })"));
        assert!(js.contains("syncPatchSave(msg.patch_save_status || '')"));
        assert!(js.contains("text.startsWith('Saving')"));
        assert!(!js.contains("setTimeout(() => patchCaptureButton"));

        let snapshot_action: WebAction =
            serde_json::from_str(r#"{"action":"open_patch_snapshot"}"#).unwrap();
        assert!(matches!(snapshot_action, WebAction::OpenPatchSnapshot));
        let look_action: WebAction =
            serde_json::from_str(r#"{"action":"open_patch_look","stack_revision":17}"#).unwrap();
        assert!(matches!(
            look_action,
            WebAction::OpenPatchLook { stack_revision: 17 }
        ));
        assert!(html.contains("id=\"patch-load-snapshot\""));
        assert!(html.contains("id=\"patch-apply-look\""));
        assert!(html.contains("id=\"patch-load-status\" role=\"status\" aria-live=\"polite\""));
        assert!(js.contains("{ action: 'open_patch_snapshot' }"));
        assert!(js.contains("{ action: 'open_patch_look', stack_revision: layerStackRevision }"));
        assert!(js.contains("syncPatchLoad(msg.patch_load_status || '')"));
    }

    #[test]
    fn browser_scrubs_bootstrap_key_and_reports_every_layer_source_error() {
        let js = include_str!("../../static/app.js");
        let html = include_str!("../../static/index.html");
        assert!(js.contains("url.searchParams.delete('key')"));
        assert!(js.contains("window.history.replaceState"));
        assert!(js.contains("<div class=\"layer-source-status\" role=\"status\""));
        assert!(js.contains("video decoder: ${layer.source_error}"));
        assert!(js.contains("chevron?.setAttribute('aria-expanded'"));
        assert!(js.contains("aria-keyshortcuts=\"ArrowUp ArrowDown Home End\""));
        assert!(js.contains("item.setAttribute('role', 'group')"));
        assert!(js.contains("className = 'library-actions'"));
        assert!(js.contains("window.confirm"));
        assert!(html.contains("id=\"output-status\" role=\"status\" aria-live=\"polite\""));
        assert!(js.contains("syncOutputWindow(msg.legacy_output_window ?? msg.output_window, msg.output_error, msg.output_display, msg.output_displays, msg.output_display_generation)"));
        assert!(js.contains("sendAction({ action: 'set_output_window', enabled })"));
        assert!(js.contains("sendAction({ action: 'set_output_display', display_id: displayId, inventory_generation: outputAuthoritativeGeneration })"));
        assert!(js.contains("sendAction({ action: 'rescan_output_displays' })"));
        assert!(html.contains("id=\"output-display\""));
        assert!(js.contains("outputPendingOpen"));
        assert!(!js.contains("sendAction({ action: 'toggle_output_window' })"));
    }

    #[test]
    fn browser_live_domains_are_versioned_and_refuse_the_wrong_authored_base() {
        let js = include_str!("../../static/app.js");
        assert!(js.contains("msg.type === 'live'"));
        assert!(js.contains("Number(msg.wire_version) !== 2"));
        assert!(js.contains("authored !== webAuthoredRevision"));
        assert!(js.contains("ws.close(4001, 'state revision mismatch')"));
        assert!(js.contains("const op = msg.operational || {}"));
        assert!(js.contains("const telemetry = msg.telemetry || {}"));
        assert!(js.contains("syncStageHealth(telemetry.stage_health)"));

        let full = AppSnapshot {
            authored_revision: 7,
            operational_revision: 11,
            telemetry_revision: 13,
            ..AppSnapshot::default()
        };
        let encoded = serde_json::to_value(full).unwrap();
        assert_eq!(encoded["wire_version"], 2);
        assert_eq!(encoded["authored_revision"], 7);
    }

    #[test]
    fn refresh_garden_browser_controls_publish_stable_ordered_routes_and_diagnostics() {
        let js = include_str!("../../static/app.js");
        let html = include_str!("../../static/index.html");
        for contract in [
            "value=\"motion\">Motion</option>",
            "id=\"temporal-garden-matte-route\"",
            "id=\"temporal-garden-matte-stage\"",
            "id=\"temporal-garden-motion-route\"",
            "id=\"temporal-garden-route-status\" role=\"status\" aria-live=\"polite\"",
        ] {
            assert!(
                html.contains(contract),
                "missing Garden HTML contract: {contract}"
            );
        }
        for contract in [
            "function syncRefreshGardenRoutes(garden)",
            "garden?.matte_route || { kind: 'none' }",
            "garden?.motion_route || { kind: 'none' }",
            "action: 'set_refresh_garden_matte_route'",
            "action: 'set_refresh_garden_motion_route'",
            "layer_stack_revision: layerStackRevision",
            "Missing saved layer ${position}",
            "gate zero",
        ] {
            assert!(
                js.contains(contract),
                "missing Garden JS contract: {contract}"
            );
        }
        let quantizable = js
            .split_once("const QUANTIZABLE_ACTIONS")
            .unwrap()
            .1
            .split_once("]);")
            .unwrap()
            .0;
        assert!(!quantizable.contains("set_refresh_garden_matte_route"));
        assert!(!quantizable.contains("set_refresh_garden_motion_route"));
    }

    #[test]
    fn every_range_control_has_a_complete_editable_numeric_contract() {
        fn assert_range_tags_are_bounded(source: &str, allow_row_metadata: bool) -> usize {
            let marker = "<input type=\"range\"";
            let mut cursor = 0;
            let mut count = 0;
            while let Some(relative) = source[cursor..].find(marker) {
                let start = cursor + relative;
                let end = start
                    + source[start..]
                        .find('>')
                        .expect("range input must have a closing bracket");
                let tag = &source[start..=end];
                let context_start = if allow_row_metadata {
                    source[..start].rfind("<div").unwrap_or(start)
                } else {
                    start
                };
                let context = &source[context_start..=end];
                for attribute in ["min", "max", "step"] {
                    assert!(
                        tag.contains(&format!("{attribute}=\""))
                            || context.contains(&format!("data-{attribute}=\"")),
                        "range #{count} is missing {attribute}: {tag}"
                    );
                }
                count += 1;
                cursor = end + 1;
            }
            count
        }

        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        let css = include_str!("../../static/style.css");

        // Exact declaration counts keep every currently shipped static and
        // generated range under this universal contract. B13 added 31 master
        // sliders (28 small effects plus the 3 master-only optics). B1's Scan
        // Processor rows render through the existing generated-card template,
        // so the literal tag counts do not move. B4 added the 17 display-
        // physics sliders to the temporal group. B5 now owns 11 codec-mosh
        // sliders there too (the recycle law is a toggle, not a range).
        // B8 added the seventeen bus-mixer sliders (wipe 5, dirt 6, melt 6),
        // the six master melting-edge sliders, and the key-dressing pair at
        // both scopes (two static master rows, two layer template rows).
        // B7 added the two generated generator-card templates (one pattern
        // row template, one text row template); their rows render from
        // tables, so only the two literal template tags move the JS count.
        // B10 added three JS template tags: the envelope row's attack and
        // decay sliders and the macro row's knob (rendered four times each;
        // the pin counts literal tags). Its HTML additions are buttons and a
        // number input, so the HTML count stays.
        // B14 added the four sync-latch sliders (shear, slip rate, band
        // height, drift bias) to the temporal section; the latch switch
        // itself is a checkbox, so only these four move the HTML count. The
        // JS count is untouched — the group is static markup.
        // B15's snapshot bank added one more: the recall glide.
        // Long Exposure Ghosting adds its amount and bounded shutter-length
        // sliders to the static Temporal Originals section.
        assert_eq!(assert_range_tags_are_bounded(html, true), 208);
        assert_eq!(assert_range_tags_are_bounded(js, false), 24);

        for contract in [
            "function normalizeRangeValue(slider, rawValue)",
            "Math.min(max, Math.max(min, value))",
            "Math.round((value - base) / step) * step",
            "binding.slider.dispatchEvent(new Event('input', { bubbles: true }))",
            "event.key === 'Enter'",
            "event.key === 'Escape'",
            "editor.addEventListener('blur'",
            "input,select,[data-range-editor]",
            "if (document.activeElement === el) return true;",
            "if (binding.editor.textContent !== textValue)",
            "binding.editor.getAttribute('aria-valuenow') !== ariaValue",
            "if (binding.disabled === disabled) return;",
            "const activeRangeBindings = new Set();",
            "if (!slider.isConnected)",
            "const peer = rangeControlPeers.get(el)",
            "syncRangeEditors(document)",
            "bindRangeEditors(card)",
            "bindRangeEditors(row)",
            "new MutationObserver",
            "editor.setAttribute('contenteditable'",
            "editor.setAttribute('role', 'spinbutton')",
            "editor.setAttribute('inputmode', min < 0 ? 'text' : 'decimal')",
            "event.key === 'ArrowUp'",
            "event.key === 'PageUp'",
            "editor.addEventListener('paste'",
            "event.clipboardData?.getData('text/plain')",
            "editor.setAttribute('aria-valuemin'",
            "editor.setAttribute('aria-valuemax'",
            "editor.setAttribute('aria-valuenow'",
            "editor.setAttribute('aria-invalid'",
        ] {
            assert!(js.contains(contract), "missing range contract: {contract}");
        }

        for coverage in [
            "data-param=\"pixelate\"",
            "data-param=\"cellular_speed\"",
            "data-ntsc=\"snow_intensity\"",
            "data-temporal=\"feedback\"",
            "data-temporal=\"loom_amount\"",
            "data-temporal=\"atlas_collision\"",
            "data-temporal=\"garden_decay\"",
            "data-temporal=\"long_exposure_amount\"",
            "data-temporal=\"long_exposure_frames\"",
            "data-temporal=\"score_state_count\"",
            "data-motion-param=\"shutter_angle\"",
            "id=\"audio-gain\"",
            "id=\"morph-t\"",
            "id=\"pad-x-curve-amount\"",
            "id=\"gyro-yaw-expo\"",
        ] {
            assert!(html.contains(coverage), "missing static range: {coverage}");
        }
        for coverage in [
            "const LAYER_EFFECT_CONTROLS",
            "data-layer-effect=\"${param}\"",
            "data-param=\"opacity\"",
            "motionRangeHtml('transplant_amount'",
            "class=\"routing-depth\"",
            "class=\"routing-curve-amount\"",
        ] {
            assert!(js.contains(coverage), "missing generated range: {coverage}");
        }
        for contract in [
            ".range-value[contenteditable=\"true\"]",
            ".range-value[contenteditable=\"true\"]:focus",
            ".range-value.range-value-invalid",
            ".range-editor-wrap",
            "touch-action: manipulation",
            "min-height: 24px",
        ] {
            assert!(css.contains(contract), "missing range styling: {contract}");
        }

        assert!(html.contains("id=\"btn-revert-master\""));
        assert!(html.contains("Revert master visual state"));
        assert!(js.contains(
            "title=\"Reset direct effects, Motion, and Mosh Send (rack, transform, opacity, and transport unchanged)\" aria-label=\"Reset direct effects, Motion, and Mosh Send; rack, transform, opacity, and transport stay unchanged\""
        ));
    }

    #[test]
    fn temporal_originals_browser_surface_is_accessible_zeroed_and_protocol_complete() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        let css = include_str!("../../static/style.css");

        for contract in [
            "class=\"temporal-originals\" role=\"region\" aria-label=\"Temporal originals\"",
            "<summary>TOPOLOGY LOOM</summary>",
            "<summary>COLLISION ATLAS</summary>",
            "<summary>REFRESH GARDEN</summary>",
            "<summary>COLLISION SCORE</summary>",
            "<summary>MEMORY LAW</summary>",
            "data-temporal=\"loom_amount\" data-min=\"0\" data-max=\"1\"",
            "data-temporal=\"atlas_seed\" data-default=\"0\"",
            "data-temporal=\"garden_amount\" data-min=\"0\" data-max=\"1\"",
            "data-temporal=\"score_enabled\"",
            "id=\"temporal-score-loop-driver\" aria-label=\"Collision Score loop driver\"",
            "id=\"temporal-clear-memory\"",
            "id=\"temporal-score-trigger\"",
            "id=\"temporal-garden-trigger\"",
            "id=\"temporal-clear-event-track\"",
            "id=\"temporal-telemetry\" role=\"status\" aria-live=\"polite\"",
        ] {
            assert!(html.contains(contract), "missing temporal HTML: {contract}");
        }
        for contract in [
            "sendAction({ action: 'set_temporal', param, value",
            "sendAction({ action: 'clear_temporal_memory' })",
            "sendAction({ action: 'trigger_collision_score' })",
            "sendAction({ action: 'trigger_refresh_garden' })",
            "sendAction({ action: 'clear_temporal_event_track' })",
            "syncTemporalLoopDriver(score.loop_driver)",
            "const telemetry = t.telemetry || {}",
            "telemetry.freeze_hold_valid",
            "telemetry.total_reference_ticks",
            "missing_selected_layer",
        ] {
            assert!(js.contains(contract), "missing temporal JS: {contract}");
        }
        assert!(css.contains(".temporal-study > summary"));
        assert!(css.contains(".temporal-telemetry"));
    }

    #[test]
    fn motion_browser_surface_is_accessible_zeroed_and_protocol_complete() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        let css = include_str!("../../static/style.css");
        for contract in [
            "id=\"motion-fields-group\"",
            "id=\"master-motion-panel\" role=\"region\" aria-label=\"Master motion field and curved shutter\"",
            "data-motion-param=\"field_source\"",
            "data-motion-param=\"shutter_angle\" data-min=\"0\" data-max=\"360\"",
            "id=\"motion-clear-memory\"",
            "id=\"master-motion-telemetry\" role=\"status\" aria-live=\"polite\"",
            "v1 · planned idle · 0 vectors · no field planned · carrier empty",
        ] {
            assert!(html.contains(contract), "missing Motion HTML: {contract}");
        }
        for contract in [
            "function layerMotionControlsHtml(layer, index)",
            "class=\"layer-motion-body motion-authoring\"",
            "aria-label=\"Layer ${index + 1} motion field, Faraday transplant, and curved shutter\"",
            "action: 'set_motion', scope, param, value",
            "action: 'set_motion_donor'",
            "action: 'clear_motion_memory'",
            "function syncMasterMotion(motion)",
            "function syncLayerMotion(card, layer)",
            "donor.kind === 'missing'",
            "motion.telemetry?.diagnostic",
            "telemetry.field_attached ? ` · rendered ${rendered}`",
            "' · field priming/unavailable'",
            "' · no field planned'",
            "motion_shutter_angle",
            "motion_transplant_amount",
        ] {
            assert!(js.contains(contract), "missing Motion JS: {contract}");
        }
        for contract in [
            ".layer-motion-heading",
            ".layer-motion-toggle",
            ".layer-motion-body[hidden]",
            ".motion-telemetry",
            ".motion-telemetry.error",
        ] {
            assert!(css.contains(contract), "missing Motion CSS: {contract}");
        }
    }

    #[test]
    fn windows_run_helper_restarts_only_the_exact_executable_with_bounded_waits() {
        let script = include_str!("../../scripts/build-windows.ps1");
        for contract in [
            "function Test-ExactExecutableProcess",
            "[System.StringComparer]::OrdinalIgnoreCase.Equals",
            "$process.CloseMainWindow()",
            "$process.Kill()",
            "[DateTime]::UtcNow.AddSeconds(5)",
            "[DateTime]::UtcNow.AddSeconds(10)",
            "$remainingExactCopies",
            "Close it manually and retry",
        ] {
            assert!(
                script.contains(contract),
                "missing restart contract: {contract}"
            );
        }
        assert!(!script.contains("Wait-Process"));
        assert!(!script.contains("Stop-Process -Id"));
        assert!(!script.contains("Get-Process -Id"));
    }

    #[test]
    fn windows_build_helper_preserves_native_version_exit_codes() {
        let script = include_str!("../../scripts/build-windows.ps1");
        for contract in [
            "$ffmpegVersionOutput = @(& $ffmpegExecutable -hide_banner -version 2>&1)",
            "$ffmpegExitCode = $LASTEXITCODE",
            "$ffmpegVersionLine = ($ffmpegVersionOutput | Select-Object -First 1)",
            "$ffprobeVersionOutput = @(& $ffprobeExecutable -hide_banner -version 2>&1)",
            "$ffprobeExitCode = $LASTEXITCODE",
            "$ffprobeVersionLine = ($ffprobeVersionOutput | Select-Object -First 1)",
        ] {
            assert!(
                script.contains(contract),
                "missing native version-probe contract: {contract}"
            );
        }
        assert!(!script
            .contains("(& $ffmpegExecutable -hide_banner -version 2>&1 | Select-Object -First 1)"));
        assert!(!script.contains(
            "(& $ffprobeExecutable -hide_banner -version 2>&1 | Select-Object -First 1)"
        ));
    }

    #[test]
    fn persistent_layer_cards_toggle_from_the_latest_snapshot() {
        let js = include_str!("../../static/app.js");
        assert!(js.contains("card._layerState = layer"));
        assert!(js.contains("const current = currentLayerContext(card, layer, index);"));
        assert!(js.contains("paused: !current.layer.paused"));
        assert!(js.contains("visible: !current.layer.visible"));
    }

    #[test]
    fn curated_blend_browser_contract_is_complete_accessible_and_order_explicit() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        let block_start = js.find("const LAYER_BLEND_MODES").unwrap();
        let block_tail = &js[block_start..];
        let block_end = block_tail.find("]);\n").unwrap() + 3;
        let block = &block_tail[..block_end];
        assert_eq!(block.matches("{ key: '").count(), 25);

        for (key, label) in [
            ("normal", "Normal"),
            ("screen", "Screen"),
            ("multiply", "Multiply"),
            ("difference", "Difference"),
            ("add", "Add"),
            ("subtract", "Subtract"),
            ("darken", "Darken"),
            ("lighten", "Lighten"),
            ("overlay", "Overlay"),
            ("soft_light", "Soft Light"),
            ("hard_light", "Hard Light"),
            ("exclusion", "Exclusion"),
            ("dodge", "Dodge"),
            ("burn", "Burn"),
            ("alpha_cut", "Alpha Cut"),
            ("vivid_light", "Vivid Light"),
            ("pin_light", "Pin Light"),
            ("divide", "Divide"),
            ("wrap_add", "Wrap Add"),
            ("xor", "Xor Bits"),
            ("and", "And Bits"),
            ("hue", "Hue"),
            ("saturation", "Saturation"),
            ("color", "Color"),
            ("luminosity", "Luminosity"),
        ] {
            let contract = format!("key: '{key}', label: '{label}'");
            assert!(block.contains(&contract), "missing blend option {contract}");
        }

        for contract in [
            "<label for=\"${blendSelectId}\">Blend</label>",
            "aria-describedby=\"layer-blend-policy ${blendDescriptionId}\"",
            "class=\"visually-hidden blend-mode-description\"",
            "title=\"${escapeHtml(layerBlendTitle(blendMode.key))}\"",
            "function layerBlendOptionsHtml(selected)",
            "if (param === 'blend_mode') syncLayerBlendDescription(row, v)",
            "select.value = layerBlendModeInfo(layer.blend_mode).key",
        ] {
            assert!(
                js.contains(contract),
                "missing blend UI contract: {contract}"
            );
        }

        for contract in [
            "Reordering changes the content below; the saved blend choice remains unchanged.",
            "Subtract computes below minus layer.",
            "Alpha Cut erases accumulated content; it is a no-op without content below.",
        ] {
            assert!(html.contains(contract), "missing blend policy: {contract}");
            assert!(js.contains(contract), "missing blend tooltip: {contract}");
        }
    }

    #[test]
    fn layer_master_fx_bypass_ui_is_stable_id_accessible_and_explicitly_scoped() {
        let js = include_str!("../../static/app.js");
        for contract in [
            "<label>Bypass Master FX</label>",
            "aria-label=\"Bypass Master FX for layer ${index + 1}\"",
            "aria-describedby=\"layer-master-bypass-help-${index}\"",
            "param: 'bypass_master_fx'",
            "...currentLayerSelector(card, layer, index)",
            "bypassMasterFx.checked = !!layer.bypass_master_fx",
            "Skips inherited Digital/Analog/Cellular/Motion processing; own Layer FX/opacity/key/blend remain. VHS still finishes the complete program once; any contributing bypass links the shared Temporal family dry while history stays warm.",
        ] {
            assert!(
                js.contains(contract),
                "missing bypass UI contract: {contract}"
            );
        }
    }

    #[test]
    fn layer_temporal_fx_bypass_ui_is_stable_id_accessible_and_independent() {
        let js = include_str!("../../static/app.js");
        for contract in [
            "<label>Bypass Temporal FX</label>",
            "aria-label=\"Bypass Temporal FX for layer ${index + 1}\"",
            "aria-describedby=\"layer-temporal-bypass-help-${index}\"",
            "param: 'bypass_temporal_fx'",
            "...currentLayerSelector(card, layer, index)",
            "bypassTemporalFx.checked = !!layer.bypass_temporal_fx",
            "Authored independently from Bypass Master FX.",
        ] {
            assert!(
                js.contains(contract),
                "missing Temporal bypass UI contract: {contract}"
            );
        }
    }

    #[test]
    fn master_transport_uses_absolute_pending_state_and_priority_revert() {
        let js = include_str!("../../static/app.js");
        for contract in [
            "let transportAuthoritativePaused = false",
            "let transportPendingPaused = null",
            "let transportRequestSequence = 0",
            "if (transportPendingPaused !== null) return",
            "sendAction({ action: 'set_program_frozen', frozen: target })",
            "transportPendingPaused === transportAuthoritativePaused",
            "renderMasterTransport(target, true)",
            "btn.toggleAttribute('aria-busy', pending)",
            "window.setTimeout(() =>",
            "transportRequestSequence === requestSequence",
            "sendAction({ action: 'reset_visual_program' })",
            "let mediaAuthoritativeFrozen = false",
            "let mediaPendingFrozen = null",
            "sendAction({ action: 'set_media_frozen', frozen: target })",
            "syncTransport(msg.program_frozen ?? msg.paused, msg.media_frozen)",
            "btn.setAttribute('aria-pressed', String(!!frozen))",
        ] {
            assert!(
                js.contains(contract),
                "missing transport contract: {contract}"
            );
        }

        for action in [WebAction::ResetFx, WebAction::ResetVisualProgram] {
            let mut queue = vec![WebAction::AddRouting; MAX_PENDING_ACTIONS];
            assert_ne!(
                enqueue_bounded(&mut queue, action.clone()),
                EnqueueOutcome::Dropped
            );
            assert!(
                matches!(queue.last(), Some(queued) if std::mem::discriminant(queued) == std::mem::discriminant(&action))
            );
        }
    }

    #[test]
    fn reset_protocol_versions_legacy_fx_and_broad_visual_program_commands() {
        let legacy: WebAction = serde_json::from_str(r#"{"action":"reset_fx"}"#).unwrap();
        let broad: WebAction =
            serde_json::from_str(r#"{"action":"reset_visual_program"}"#).unwrap();

        assert!(matches!(&legacy, WebAction::ResetFx));
        assert!(matches!(&broad, WebAction::ResetVisualProgram));
        assert!(legacy.is_priority());
        assert!(broad.is_priority());
    }

    #[test]
    fn reroll_protocol_defaults_and_ordered_barriers_are_explicit() {
        let reroll: WebAction =
            serde_json::from_str(r#"{"action":"reroll","scope":"master"}"#).unwrap();
        assert!(matches!(
            &reroll,
            WebAction::Reroll {
                scope: RerollScope::Master,
                index: None,
                layer_id: None,
                group_id: None,
                stack_revision: None,
                seed: None,
                mode: RerollMode::Pattern,
                amount,
                include_grain_controls: false,
                include_transform: false,
                include_rack_controls: false,
                include_group_controls: false,
                keep_source: false,
                keep_modulation: false,
                keep_output_chain: false,
            } if (*amount - 0.7).abs() < f32::EPSILON
        ));
        assert_eq!(reroll.coalesce_key(), None);
        assert!(reroll.is_priority());

        let explicit = WebAction::Reroll {
            scope: RerollScope::All,
            index: None,
            layer_id: None,
            group_id: None,
            stack_revision: Some(19),
            seed: Some(u32::MAX),
            mode: RerollMode::Variation,
            amount: 1.25,
            include_grain_controls: true,
            include_transform: true,
            include_rack_controls: true,
            include_group_controls: true,
            keep_source: false,
            keep_modulation: false,
            keep_output_chain: false,
        };
        let encoded = serde_json::to_value(&explicit).unwrap();
        assert_eq!(encoded["action"], "reroll");
        assert_eq!(encoded["scope"], "all");
        assert_eq!(encoded["stack_revision"], 19);
        assert_eq!(encoded["seed"], u32::MAX);
        assert_eq!(encoded["mode"], "variation");
        assert_json_number_near(&encoded["amount"], 1.25);
        assert_eq!(encoded["include_grain_controls"], true);
        assert_eq!(encoded["include_transform"], true);
        assert_eq!(encoded["include_rack_controls"], true);
        assert_eq!(encoded["include_group_controls"], true);

        let mut queue = Vec::new();
        assert_eq!(
            enqueue_bounded(
                &mut queue,
                WebAction::SetParam {
                    param: "brightness".into(),
                    value: serde_json::json!(0.1),
                },
            ),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(&mut queue, reroll.clone()),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(
                &mut queue,
                WebAction::SetParam {
                    param: "brightness".into(),
                    value: serde_json::json!(0.9),
                },
            ),
            EnqueueOutcome::Added,
            "an absolute edit after a reroll must not replace the edit observed before it"
        );
        assert_eq!(
            enqueue_bounded(&mut queue, explicit),
            EnqueueOutcome::Added,
            "each dice press is an independently ordered command"
        );
        assert!(matches!(
            queue.as_slice(),
            [
                WebAction::SetParam { value: before, .. },
                WebAction::Reroll {
                    scope: RerollScope::Master,
                    ..
                },
                WebAction::SetParam { value: after, .. },
                WebAction::Reroll {
                    scope: RerollScope::All,
                    ..
                },
            ] if before == &serde_json::json!(0.1) && after == &serde_json::json!(0.9)
        ));

        assert!(
            serde_json::from_str::<WebAction>(r#"{"action":"reroll","scope":"unknown"}"#).is_err()
        );
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"reroll","scope":"master","mode":"unknown"}"#
        )
        .is_err());
    }

    #[test]
    fn creative_routes_keep_decimal_missing_diagnostics_and_typed_timing() {
        let missing = CreativeImageTapSnapshot {
            input: CreativeImageSourceSnapshot::MissingGroupOutput {
                group_id: u64::MAX.to_string(),
            },
            timing: EdgeTiming::PreviousFrame,
        };
        let value = serde_json::to_value(&missing).unwrap();
        assert_eq!(value["input"]["source"], "missing_group_output");
        assert_eq!(value["input"]["group_id"], u64::MAX.to_string());
        assert_eq!(value["timing"], "previous_frame");
        assert_eq!(
            serde_json::from_value::<CreativeImageTapSnapshot>(value).unwrap(),
            missing
        );
        assert!(serde_json::from_str::<CreativeImageTapSnapshot>(
            r#"{"input":{"source":"one_below"},"timing":"later"}"#
        )
        .is_err());
    }

    #[test]
    fn creative_coalescing_and_topology_barriers_are_stable_id_scoped() {
        let matte_value = WebAction::SetCompositionGroupMatteParam {
            group_id: "17".into(),
            param: "softness".into(),
            value: serde_json::json!(0.2),
            composition_revision: 9,
        };
        assert_eq!(
            matte_value.coalesce_key().as_deref(),
            Some("creative:group:17:matte:softness")
        );
        assert!(!matte_value.is_priority());

        let crossfade = WebAction::SetCompositionBusCrossfade { value: 0.25 };
        assert_eq!(
            crossfade.coalesce_key().as_deref(),
            Some("creative:composition:bus-crossfade")
        );
        assert!(!crossfade.is_priority());

        let matte_route = WebAction::SetCompositionGroupMatteRoute {
            group_id: "17".into(),
            route: None,
            channel: "alpha".into(),
            invert: false,
            composition_revision: 9,
        };
        let layer_bus = WebAction::SetCompositionLayerBus {
            layer_id: "99".into(),
            bus: "a".into(),
            composition_revision: 9,
        };
        assert_eq!(matte_route.coalesce_key(), None);
        assert!(matte_route.is_priority());
        assert_eq!(layer_bus.coalesce_key(), None);
        assert!(layer_bus.is_priority());

        // Every Symmetry Field slot is its own ordered barrier, while its
        // geometry, masks, and seed ride the one coalescible value action.
        for route in [
            SymmetryRouteSnapshot::Image {
                index: 0,
                route: CreativeImageTapSnapshot {
                    input: CreativeImageSourceSnapshot::OneBelow,
                    timing: EdgeTiming::CurrentFrame,
                },
            },
            SymmetryRouteSnapshot::Motion {
                index: 1,
                layer_id: Some("31".into()),
            },
        ] {
            let action = WebAction::SetVisualNodeSymmetryRoute {
                scope: CreativeScopeSnapshot::Master,
                node_id: "7".into(),
                route,
                composition_revision: 9,
            };
            assert_eq!(action.coalesce_key(), None);
            assert!(action.is_priority());
        }
        let seed = WebAction::SetVisualNodeParam {
            scope: CreativeScopeSnapshot::Master,
            node_id: "7".into(),
            node_kind: "symmetry".into(),
            param: "symmetry_seed".into(),
            value: serde_json::json!(9_001),
            composition_revision: 9,
        };
        assert_eq!(
            seed.coalesce_key().as_deref(),
            Some("creative:master:node:7:symmetry_seed")
        );
        assert!(!seed.is_priority());

        // A Residual value rides the one coalescible node action and keys on
        // its own parameter, while both of its routes stay ordered barriers.
        let residual_value = WebAction::SetVisualNodeParam {
            scope: CreativeScopeSnapshot::Master,
            node_id: "3".into(),
            node_kind: "residual".into(),
            param: "detail_gain".into(),
            value: serde_json::json!(2.0),
            composition_revision: 9,
        };
        assert_eq!(
            residual_value.coalesce_key().as_deref(),
            Some("creative:master:node:3:detail_gain")
        );
        assert!(!residual_value.is_priority());
        for slot in [
            ResidualRouteSlotSnapshot::Structure,
            ResidualRouteSlotSnapshot::Detail,
        ] {
            let residual_route = WebAction::SetVisualNodeResidualRoute {
                scope: CreativeScopeSnapshot::Master,
                node_id: "3".into(),
                slot,
                route: CreativeImageTapSnapshot {
                    input: CreativeImageSourceSnapshot::OneBelow,
                    timing: EdgeTiming::CurrentFrame,
                },
                composition_revision: 9,
            };
            assert_eq!(residual_route.coalesce_key(), None);
            assert!(residual_route.is_priority());
        }
    }

    #[test]
    fn random_dice_and_layer_loop_browser_contract_is_complete() {
        let mut legacy_effects = serde_json::to_value(EffectsSnapshot::default()).unwrap();
        legacy_effects
            .as_object_mut()
            .unwrap()
            .remove("random_seed");
        let restored: EffectsSnapshot = serde_json::from_value(legacy_effects).unwrap();
        assert_eq!(restored.random_seed, 0);

        let loop_action: WebAction = serde_json::from_str(
            r#"{"action":"set_layer_reroll_on_loop","index":3,"layer_id":"22","enabled":true}"#,
        )
        .unwrap();
        assert!(matches!(
            &loop_action,
            WebAction::SetLayerRerollOnLoop {
                index: 3,
                layer_id: Some(layer_id),
                enabled: true,
            } if layer_id == "22"
        ));
        assert_eq!(
            loop_action.coalesce_key().as_deref(),
            Some("layer:id:22:reroll-on-loop")
        );
        assert!(loop_action.is_priority());

        let html = include_str!("../../static/index.html");
        for contract in [
            "id=\"random-group\"",
            "RANDOM / DICE",
            "id=\"reroll-scope\"",
            "<option value=\"master\">Master</option>",
            "<option value=\"all\">Everything</option>",
            "<option value=\"group\">Group</option>",
            "id=\"reroll-group\"",
            "id=\"reroll-mode\"",
            "<option value=\"pattern\">Pattern only</option>",
            "<option value=\"variation\">Bounded variation</option>",
            "id=\"reroll-seed\"",
            "max=\"4294967295\"",
            "id=\"reroll-amount\"",
            "max=\"2\"",
            "id=\"reroll-grain-controls\"",
            "id=\"reroll-transform-controls\"",
            "id=\"reroll-rack-controls\"",
            "id=\"reroll-group-controls\"",
            "id=\"reroll-button\"",
            "id=\"reroll-status\"",
            "bounded Motion numeric values for Master / Layer / Everything",
            "never changes Motion algorithm/version, field or quality tiers, donor, carrier",
        ] {
            assert!(
                html.contains(contract),
                "missing random HTML contract: {contract}"
            );
        }

        let js = include_str!("../../static/app.js");
        for contract in [
            "const scope = ['all', 'group'].includes(rerollScope.value) ? rerollScope.value : 'master'",
            "action: 'reroll'",
            "mode: rerollMode.value === 'variation' ? 'variation' : 'pattern'",
            "include_grain_controls: !!rerollGrainControls.checked",
            "include_transform: !!rerollTransformControls.checked",
            "include_rack_controls: !!rerollRackControls?.checked",
            "include_group_controls: scope === 'group' && !!rerollGroupControls?.checked",
            "if (scope === 'all') action.stack_revision = layerStackRevision",
            "if (scope === 'group')",
            "action.group_id = rerollGroup.value",
            "Number(effects.random_seed) >>> 0",
            "class=\"layer-random-seed seed-input\"",
            "class=\"layer-reroll\"",
            "scope: 'layer'",
            "...currentLayerSelector(card, layer, index)",
            "action: 'set_layer_reroll_on_loop'",
            "layer.reroll_on_loop ? 'checked' : ''",
            "Number(layer.effects?.random_seed || 0) >>> 0",
        ] {
            assert!(
                js.contains(contract),
                "missing random JS contract: {contract}"
            );
        }
    }

    #[test]
    fn split_freeze_protocol_snapshot_and_controls_remain_distinct() {
        let program: WebAction =
            serde_json::from_str(r#"{"action":"set_program_frozen","frozen":true}"#).unwrap();
        let media: WebAction =
            serde_json::from_str(r#"{"action":"set_media_frozen","frozen":true}"#).unwrap();
        assert!(matches!(
            program,
            WebAction::SetProgramFrozen { frozen: true }
        ));
        assert!(matches!(media, WebAction::SetMediaFrozen { frozen: true }));
        assert_eq!(program.coalesce_key().as_deref(), Some("master:paused"));
        assert_eq!(media.coalesce_key().as_deref(), Some("media:frozen"));
        assert_eq!(
            program.coalesce_key(),
            WebAction::SetMasterPaused { paused: false }.coalesce_key(),
            "the canonical program action must replace a pending legacy master action"
        );
        assert_ne!(program.coalesce_key(), media.coalesce_key());
        assert!(program.is_priority());
        assert!(media.is_priority());

        let mut legacy = serde_json::to_value(AppSnapshot::default()).unwrap();
        legacy.as_object_mut().unwrap().remove("program_frozen");
        legacy.as_object_mut().unwrap().remove("media_frozen");
        let restored: AppSnapshot = serde_json::from_value(legacy).unwrap();
        assert!(!restored.program_frozen);
        assert!(!restored.media_frozen);

        let html = include_str!("../../static/index.html");
        for contract in [
            "id=\"btn-play-all\"",
            "aria-label=\"Freeze program\"",
            "Freeze the complete visual program, including clocks and histories",
            "id=\"btn-freeze-media\"",
            "aria-label=\"Freeze media\"",
            "Hold video and Spout frames while effects and modulation continue",
        ] {
            assert!(
                html.contains(contract),
                "missing freeze HTML contract: {contract}"
            );
        }

        let js = include_str!("../../static/app.js");
        for contract in [
            "sendAction({ action: 'set_program_frozen', frozen: target })",
            "sendAction({ action: 'set_media_frozen', frozen: target })",
            "syncTransport(msg.program_frozen ?? msg.paused, msg.media_frozen)",
            "renderMasterTransport(target, true)",
            "renderMediaFreeze(target, true)",
        ] {
            assert!(
                js.contains(contract),
                "missing freeze JS contract: {contract}"
            );
        }
    }

    #[test]
    fn lfo_seed_snapshot_action_and_browser_contract_are_backward_compatible() {
        let legacy: LfoSnapshot = serde_json::from_str(
            r#"{"shape":"sample_hold","beats":4.0,"phase":0.25,"value":-0.5}"#,
        )
        .unwrap();
        assert_eq!(legacy.seed, 0);

        let current = LfoSnapshot {
            shape: "sample_hold".into(),
            beats: 8.0,
            phase: 0.5,
            seed: u32::MAX,
            value: 0.75,
        };
        let value = serde_json::to_value(current).unwrap();
        assert_eq!(value["seed"], u32::MAX);

        let mut matrix = crate::modulation::ModMatrix::new();
        matrix.lfos[7].shape = crate::modulation::LfoShape::Square;
        matrix.lfos[7].seed = 0x7654_3210;
        matrix.update_at_beat(0.0, 1.0 / 30.0);
        let snapshot = ModSnapshot::from_matrix(&matrix);
        assert_eq!(snapshot.lfos.len(), crate::modulation::NUM_LFOS);
        assert_eq!(snapshot.lfos[7].shape, "square");
        assert_eq!(snapshot.lfos[7].seed, 0x7654_3210);
        assert_eq!(snapshot.lfos[7].value, 1.0);

        let action: WebAction = serde_json::from_str(
            r#"{"action":"set_lfo","index":2,"param":"seed","value":4294967295}"#,
        )
        .unwrap();
        assert!(matches!(
            &action,
            WebAction::SetLfo {
                index: 2,
                param,
                value,
            } if param == "seed" && value == &serde_json::json!(u32::MAX)
        ));
        assert_eq!(action.coalesce_key().as_deref(), Some("lfo:2:seed"));

        let js = include_str!("../../static/app.js");
        for contract in [
            "class=\"lfo-seed seed-input\" type=\"number\" min=\"0\" max=\"4294967295\"",
            "row.querySelector('.lfo-seed').hidden = e.target.value !== 'sample_hold'",
            "Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff",
            "sendAction({ action: 'set_lfo', index: i, param: 'seed', value: seed })",
            "seedInput.value = String(Number(lfo.seed || 0) >>> 0)",
            "seedInput.hidden = lfo.shape !== 'sample_hold'",
            "const NUM_LFOS = 8;",
            "for (let i = 0; i < NUM_LFOS; i++)",
            "['lfo4', 'L5']",
            "['lfo5', 'L6']",
            "['lfo6', 'L7']",
            "['lfo7', 'L8']",
        ] {
            assert!(
                js.contains(contract),
                "missing LFO seed contract: {contract}"
            );
        }
    }

    #[test]
    fn new_control_actions_deserialize_with_explicit_protocol_names() {
        let action: WebAction = serde_json::from_str(r#"{"action":"gyro_calibrate"}"#).unwrap();
        assert!(matches!(action, WebAction::GyroCalibrate));

        let action: WebAction = serde_json::from_str(
            r#"{"action":"set_gyro_config","axis":"roll","param":"invert","value":true}"#,
        )
        .unwrap();
        assert!(
            matches!(action, WebAction::SetGyroConfig { axis, param, value }
            if axis == "roll" && param == "invert" && value == serde_json::json!(true))
        );

        let action: WebAction = serde_json::from_str(
            r#"{"action":"set_pad_config","axis":"x","param":"quantize","value":8}"#,
        )
        .unwrap();
        assert!(
            matches!(action, WebAction::SetPadConfig { axis, param, value }
            if axis == "x" && param == "quantize" && value == serde_json::json!(8))
        );

        let action: WebAction =
            serde_json::from_str(r#"{"action":"pad","x":0.2,"y":0.8,"active":false}"#).unwrap();
        assert!(matches!(action, WebAction::Pad { x, y, active }
            if (x - 0.2).abs() < f32::EPSILON && (y - 0.8).abs() < f32::EPSILON && !active));

        let action: WebAction =
            serde_json::from_str(r#"{"action":"move_layer","from":3,"to":1}"#).unwrap();
        assert!(matches!(
            action,
            WebAction::MoveLayer {
                from: 3,
                to: 1,
                layer_id: None,
                stack_revision: None
            }
        ));

        let action: WebAction =
            serde_json::from_str(r#"{"action":"add_spout_layer","sender":"Resolume Composition"}"#)
                .unwrap();
        assert!(matches!(action, WebAction::AddSpoutLayer { sender }
            if sender == "Resolume Composition"));

        let action: WebAction =
            serde_json::from_str(r#"{"action":"set_spout_resolution","resolution":"1080p"}"#)
                .unwrap();
        assert!(matches!(
            &action,
            WebAction::SetSpoutResolution { resolution } if resolution == "1080p"
        ));
        assert_eq!(action.coalesce_key().as_deref(), Some("spout:resolution"));
        assert!(action.is_performance_only_for_history());

        let value = serde_json::to_value(WebAction::GyroCalibrate).unwrap();
        assert_eq!(value["action"], "gyro_calibrate");
        let value = serde_json::to_value(WebAction::SetPadConfig {
            axis: "both".into(),
            param: "spring_rate".into(),
            value: serde_json::json!(4.0),
        })
        .unwrap();
        assert_eq!(value["action"], "set_pad_config");
        assert_eq!(value["axis"], "both");
        assert_eq!(value["param"], "spring_rate");
    }

    #[test]
    fn legacy_pad_action_defaults_to_active() {
        let action: WebAction =
            serde_json::from_str(r#"{"action":"pad","x":0.5,"y":0.5}"#).unwrap();
        assert!(matches!(action, WebAction::Pad { active: true, .. }));
    }

    #[test]
    fn quantized_action_wraps_existing_protocol_without_changing_it() {
        let action: WebAction = serde_json::from_str(
            r#"{"action":"quantized","inner":{"action":"set_morph","value":0.75}}"#,
        )
        .unwrap();
        assert!(matches!(
            action,
            WebAction::Quantized { inner }
                if matches!(*inner, WebAction::SetMorph { value } if (value - 0.75).abs() < f32::EPSILON)
        ));
    }

    #[test]
    fn routing_snapshot_accepts_legacy_and_round_trips_response_fields() {
        let legacy: RoutingSnapshot =
            serde_json::from_str(r#"{"source":"lfo0","target":"rgb_split","depth":0.5}"#).unwrap();
        assert_eq!(legacy.curve, "linear");
        assert_eq!(legacy.curve_amount, 0.0);
        assert_eq!(legacy.attack, 0.0);
        assert_eq!(legacy.release, 0.0);
        assert!(legacy.route_id.is_empty());
        assert_eq!(legacy.value, 0.0);

        let routing = RoutingSnapshot {
            route_id: "42".into(),
            source: "pad_x".into(),
            target: "morph".into(),
            depth: -0.75,
            curve: "s_curve".into(),
            curve_amount: 0.4,
            attack: 0.12,
            release: 1.5,
            value: -0.25,
        };
        let value = serde_json::to_value(&routing).unwrap();
        assert_eq!(value["curve"], "s_curve");
        assert_json_number_near(&value["curve_amount"], 0.4);
        assert_json_number_near(&value["attack"], 0.12);
        assert_json_number_near(&value["release"], 1.5);
        assert_eq!(value["route_id"], "42");
        assert_json_number_near(&value["value"], -0.25);
    }

    #[test]
    fn routing_actions_accept_stable_ids_and_rescan_keeps_its_barrier_position() {
        let action: WebAction = serde_json::from_str(
            r#"{"action":"set_routing","index":7,"route_id":"42","param":"depth","value":0.5}"#,
        )
        .unwrap();
        assert!(matches!(
            action,
            WebAction::SetRouting {
                index: 7,
                route_id: Some(id),
                target_layer_id: None,
                layer_stack_revision: None,
                ..
            } if id == "42"
        ));

        let mut queue = Vec::new();
        assert_eq!(
            enqueue_bounded(&mut queue, WebAction::RescanLibrary),
            EnqueueOutcome::Added
        );
        let clip = WebAction::SetAudio {
            param: "clip".into(),
            value: serde_json::json!("second.wav"),
        };
        assert_eq!(enqueue_bounded(&mut queue, clip), EnqueueOutcome::Added);
        assert_eq!(
            enqueue_bounded(&mut queue, WebAction::RescanLibrary),
            EnqueueOutcome::Coalesced
        );
        assert!(matches!(queue.first(), Some(WebAction::RescanLibrary)));
        assert!(matches!(
            queue.get(1),
            Some(WebAction::SetAudio { param, .. }) if param == "clip"
        ));
    }

    #[test]
    fn routing_target_identity_is_backward_compatible_and_coalesces_latest_metadata() {
        let js = include_str!("../../static/app.js");
        for contract in [
            "latestLayerIdentities = layers.map((layer) => String(layer.layer_id || ''))",
            "const targetLayerId = latestLayerIdentities[Number(layerMatch[1]) - 1]",
            "targetIdentity = { target_layer_id: targetLayerId }",
            "targetIdentity.layer_stack_revision = layerStackRevision",
            "...selector(), ...targetIdentity, param: 'target', value: target",
        ] {
            assert!(
                js.contains(contract),
                "missing target identity contract: {contract}"
            );
        }

        let legacy: WebAction = serde_json::from_str(
            r#"{"action":"set_routing","index":2,"param":"target","value":"layer2_opacity"}"#,
        )
        .unwrap();
        assert!(matches!(
            legacy,
            WebAction::SetRouting {
                target_layer_id: None,
                layer_stack_revision: None,
                ..
            }
        ));

        let first: WebAction = serde_json::from_str(
            r#"{"action":"set_routing","index":2,"route_id":"42","target_layer_id":"20","layer_stack_revision":7,"param":"target","value":"layer2_opacity"}"#,
        )
        .unwrap();
        let latest: WebAction = serde_json::from_str(
            r#"{"action":"set_routing","index":0,"route_id":"42","target_layer_id":"20","layer_stack_revision":8,"param":"target","value":"layer1_opacity"}"#,
        )
        .unwrap();
        let mut queue = Vec::new();
        assert_eq!(enqueue_bounded(&mut queue, first), EnqueueOutcome::Added);
        assert_eq!(
            enqueue_bounded(&mut queue, latest),
            EnqueueOutcome::Coalesced
        );
        assert!(matches!(
            queue.as_slice(),
            [WebAction::SetRouting {
                route_id: Some(route_id),
                target_layer_id: Some(layer_id),
                layer_stack_revision: Some(8),
                value,
                ..
            }] if route_id == "42" && layer_id == "20" && value == "layer1_opacity"
        ));
    }

    #[test]
    fn legacy_mod_snapshot_defaults_new_configs() {
        let legacy = r#"{
            "bpm":120.0,
            "beat":0.0,
            "lfos":[],
            "routings":[],
            "gyro":[0.5,0.5,0.5],
            "pad":[0.5,0.5]
        }"#;
        let snapshot: ModSnapshot = serde_json::from_str(legacy).unwrap();
        assert_eq!(snapshot.gyro_config.yaw.range, 180.0);
        assert_eq!(snapshot.gyro_config.pitch.range, 90.0);
        assert_eq!(snapshot.pad_config.x.curve, "linear");
        assert!(!snapshot.pad_config.spring_enabled);
        assert_eq!(snapshot.pad_config.spring_rate, 4.0);
    }

    #[test]
    fn audio_protocol_round_trips_and_defaults_legacy_snapshots() {
        let legacy: AudioSnapshot = serde_json::from_str(
            r#"{"enabled":false,"gain":1.0,"level":0.0,"bass":0.1,"mid":0.2,"high":0.3,"onset":0.0}"#,
        )
        .unwrap();
        assert_eq!(legacy.band_count, 3);
        assert_eq!(legacy.band_ceiling_hz, 8000.0);
        assert!(legacy.bands.is_empty());
        assert_eq!(legacy.source_kind, "live");
        assert!(legacy.system_playback_devices.is_empty());
        assert!(legacy.clip_files.is_empty());
        assert!(legacy.clip_path.is_empty());
        assert!(!legacy.clip_loading);
        assert_eq!(legacy.clip_duration_secs, 0.0);

        let current = AudioSnapshot {
            band_count: 8,
            band_edges: vec![100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0],
            band_ceiling_hz: 12_800.0,
            bands: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            source_kind: "file".into(),
            system_playback_devices: vec!["Speakers".into()],
            clip_files: vec!["pulse-loop.wav".into()],
            clip_path: "pulse-loop.wav".into(),
            clip_loading: true,
            clip_duration_secs: 2.5,
            ..AudioSnapshot::default()
        };
        let value = serde_json::to_value(current).unwrap();
        assert_eq!(value["band_count"], 8);
        assert_eq!(value["band_edges"].as_array().unwrap().len(), 7);
        assert_eq!(value["bands"].as_array().unwrap().len(), 8);
        assert_json_number_near(&value["band_ceiling_hz"], 12_800.0);
        assert_eq!(value["source_kind"], "file");
        assert_eq!(value["system_playback_devices"][0], "Speakers");
        assert_eq!(value["clip_files"][0], "pulse-loop.wav");
        assert_eq!(value["clip_path"], "pulse-loop.wav");
        assert_eq!(value["clip_loading"], true);
        assert_json_number_near(&value["clip_duration_secs"], 2.5);

        for raw in [
            r#"{"action":"set_audio","param":"band_count","value":6}"#,
            r#"{"action":"set_audio","param":"band_edges","value":{"count":6,"edges":[120,480,1500,4000,9000],"ceiling":16000}}"#,
            r#"{"action":"set_audio","param":"source_kind","value":"file"}"#,
            r#"{"action":"set_audio","param":"clip","value":"pulse-loop.wav"}"#,
        ] {
            let action: WebAction = serde_json::from_str(raw).unwrap();
            assert!(matches!(action, WebAction::SetAudio { .. }));
        }
    }

    #[test]
    fn audio_band_browser_contract_exposes_count_edges_meters_and_sources() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        assert!(html.contains("id=\"audio-band-count\""));
        assert!(html.contains("id=\"audio-band-edges\""));
        assert!(html.contains("id=\"audio-high-edge\""));
        assert!(html.contains("id=\"audio-extra-band-meters\""));
        assert!(html.contains("id=\"audio-source-kind\""));
        assert!(html.contains("id=\"audio-clip\""));
        assert!(html.contains("id=\"audio-import\""));
        assert!(html.contains(".wav,.mp3,.flac,.ogg,.opus,.m4a,.aac"));
        assert!(html.contains("Windows system playback"));
        assert!(js.contains("param: 'band_count'"));
        assert!(js.contains("param: 'source_kind'"));
        assert!(js.contains("param: 'clip'"));
        assert!(js.contains("MAX_AUDIO_IMPORT_BYTES = 512 * 1024 * 1024"));
        assert!(
            js.contains("slider.value = Number.isFinite(declaredDefault) ? declaredDefault : min")
        );
        assert!(js.contains("const fileMode = audioSourceKind.value === 'file'"));
        assert!(js.contains("audioClipRow.hidden = !fileMode"));
        assert!(js.contains("audioDevice.closest('.param-row').hidden = fileMode"));
        assert!(include_str!("../../static/style.css").contains(".param-row[hidden]"));
        assert!(js.contains("class=\"layer-fx-body\""));
        assert!(js.contains("class=\"layer-cellular-body\""));
        assert!(js.contains("role=\"region\" aria-label=\"Layer ${index + 1} effects\" hidden"));
        assert!(js.contains("system-playback:default"));
        assert!(js.contains("deterministic program-time analysis"));
        assert!(js.contains(".filter(({ layer }) => layer.source_kind === 'video')"));
        assert!(js.contains("const hasAnimatedPreview = /\\.(mp4|webm|mov|avi|mkv)$/i"));
        assert!(js.contains("value: { count, edges, ceiling }"));
        for index in 1..=crate::audio::MAX_AUDIO_BANDS {
            assert!(
                js.contains(&format!("['audio_band{index}', 'Band {index}']")),
                "browser route menu must expose audio band {index}"
            );
        }
    }

    #[test]
    fn config_snapshots_serialize_every_browser_control_field() {
        let gyro = GyroConfigSnapshot {
            yaw: AxisConfigSnapshot {
                range: 30.0,
                expo: 0.5,
                invert: true,
            },
            pitch: AxisConfigSnapshot {
                range: 45.0,
                expo: -0.25,
                invert: false,
            },
            roll: AxisConfigSnapshot {
                range: 60.0,
                expo: 1.0,
                invert: true,
            },
        };
        let gyro_value = serde_json::to_value(gyro).unwrap();
        assert_json_number_near(&gyro_value["yaw"]["range"], 30.0);
        assert_json_number_near(&gyro_value["yaw"]["expo"], 0.5);
        assert_eq!(gyro_value["yaw"]["invert"], true);
        assert_json_number_near(&gyro_value["pitch"]["range"], 45.0);
        assert_json_number_near(&gyro_value["roll"]["range"], 60.0);

        let pad = PadConfigSnapshot {
            x: PadAxisConfigSnapshot {
                curve: "exp".into(),
                curve_amount: 0.8,
                quantize: 8,
            },
            y: PadAxisConfigSnapshot {
                curve: "steps".into(),
                curve_amount: -0.4,
                quantize: 16,
            },
            spring_enabled: true,
            spring_rate: 6.5,
        };
        let pad_value = serde_json::to_value(pad).unwrap();
        assert_eq!(pad_value["x"]["curve"], "exp");
        assert_json_number_near(&pad_value["x"]["curve_amount"], 0.8);
        assert_eq!(pad_value["x"]["quantize"], 8);
        assert_eq!(pad_value["y"]["curve"], "steps");
        assert_json_number_near(&pad_value["y"]["curve_amount"], -0.4);
        assert_eq!(pad_value["y"]["quantize"], 16);
        assert_eq!(pad_value["spring_enabled"], true);
        assert_json_number_near(&pad_value["spring_rate"], 6.5);
    }

    #[test]
    fn legacy_app_snapshot_defaults_export_status() {
        let mut value = serde_json::to_value(AppSnapshot::default()).unwrap();
        value.as_object_mut().unwrap().remove("export_status");
        value.as_object_mut().unwrap().remove("export_warnings");
        value.as_object_mut().unwrap().remove("patch_save_status");
        value.as_object_mut().unwrap().remove("master_transform");
        let restored: AppSnapshot = serde_json::from_value(value).unwrap();
        assert_eq!(restored.export_status, "");
        assert!(restored.export_warnings.is_empty());
        assert_eq!(restored.patch_save_status, "");
        assert_eq!(restored.master_transform, SpatialTransform::default());

        let current = AppSnapshot {
            export_warnings: vec!["Layer 1 substituted deterministic black.".into()],
            ..Default::default()
        };
        let serialized = serde_json::to_value(current).unwrap();
        assert_eq!(serialized["export_warnings"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn legacy_snapshot_defaults_recorder_and_preview_health_safely() {
        let mut partial = serde_json::to_value(AppSnapshot::default()).unwrap();
        partial["recorder"]
            .as_object_mut()
            .unwrap()
            .remove("dropped_source_unavailable");
        let restored: AppSnapshot = serde_json::from_value(partial).unwrap();
        assert_eq!(restored.recorder.dropped_source_unavailable, 0);

        let mut value = serde_json::to_value(AppSnapshot::default()).unwrap();
        value.as_object_mut().unwrap().remove("recorder");
        value.as_object_mut().unwrap().remove("stage_health");
        let restored: AppSnapshot = serde_json::from_value(value).unwrap();
        assert_eq!(restored.recorder.status, "idle");
        assert!(restored.recorder.audio_not_muxed);
        assert_eq!(restored.stage_health.frame_samples, 0);
        assert!(!restored.stage_health.tools.health_hud_enabled);
    }

    #[test]
    fn recorder_commands_are_ordered_and_stage_controls_coalesce_by_destination() {
        let barriers = [
            WebAction::StartProgramRecording { auto_import: false },
            WebAction::FinishProgramRecording,
            WebAction::CancelProgramRecording,
            WebAction::CaptureStill {
                target: CaptureTargetSnapshot::Program,
                auto_import: false,
            },
            WebAction::StartResample {
                target: CaptureTargetSnapshot::Layer {
                    layer_id: "1".into(),
                },
                destination_layer_id: "2".into(),
                activate: false,
            },
        ];
        for action in barriers {
            assert_eq!(action.coalesce_key(), None);
            assert!(action.is_priority());
        }
        let hud = WebAction::SetStageHealthHud { enabled: true };
        let card = WebAction::SetStageTestCard {
            mode: crate::stage_map::TestCardMode::Grid,
            output_endpoint_id: Some("legacy-output-1".into()),
        };
        let identify = WebAction::SetOutputIdentification {
            enabled: true,
            output_endpoint_id: Some("legacy-output-1".into()),
        };
        assert_eq!(hud.coalesce_key().as_deref(), Some("stage:health-hud"));
        assert_eq!(card.coalesce_key().as_deref(), Some("stage:test-card"));
        assert_eq!(
            identify.coalesce_key().as_deref(),
            Some("stage:output-identification")
        );
        assert!(card.is_priority());
        assert!(identify.is_priority());
    }

    #[test]
    fn gyro_registry_tracks_explicit_legacy_timeout_and_disconnect_states() {
        let start = Instant::now();
        let mut registry = GyroStreamRegistry::default();
        assert_eq!(registry.status_at(start), GyroStatusSnapshot::default());

        registry.set_stream(7, true);
        let waiting = registry.status_at(start);
        assert!(!waiting.active);
        assert!(waiting.stale);
        assert_eq!(waiting.streamers, 1);
        assert_eq!(waiting.sample_age_ms, None);

        registry.note_sample_at(7, start);
        let live = registry.status_at(start + Duration::from_millis(250));
        assert!(live.active);
        assert!(!live.stale);
        assert_eq!(live.sample_age_ms, Some(250));

        let timed_out = registry.status_at(start + GYRO_SAMPLE_TIMEOUT + Duration::from_millis(1));
        assert!(!timed_out.active);
        assert!(timed_out.stale);

        registry.disconnect(7);
        let stopped = registry.status_at(start + Duration::from_secs(2));
        assert!(!stopped.active);
        assert!(stopped.stale);
        assert_eq!(stopped.streamers, 0);

        let mut legacy = GyroStreamRegistry::default();
        legacy.note_sample_at(42, start);
        assert!(legacy.status_at(start).active);
        legacy.disconnect(42);
        assert!(legacy.status_at(start).stale);

        let mut explicitly_stopped = GyroStreamRegistry::default();
        explicitly_stopped.set_stream(9, true);
        explicitly_stopped.set_stream(9, false);
        explicitly_stopped.note_sample_at(9, start);
        assert_eq!(explicitly_stopped.status_at(start).streamers, 0);
    }

    #[test]
    fn gyro_stream_protocol_and_telemetry_are_backward_compatible() {
        let action: WebAction =
            serde_json::from_str(r#"{"action":"gyro_stream","enabled":true}"#).unwrap();
        assert!(matches!(action, WebAction::GyroStream { enabled: true }));

        let mut legacy = serde_json::to_value(ModSnapshot::default()).unwrap();
        legacy.as_object_mut().unwrap().remove("gyro_status");
        let restored: ModSnapshot = serde_json::from_value(legacy).unwrap();
        assert_eq!(restored.gyro_status, GyroStatusSnapshot::default());

        let js = include_str!("../../static/app.js");
        assert!(js.contains("sendAction({ action: 'gyro_stream', enabled: true })"));
        assert!(js.contains("sendAction({ action: 'gyro_stream', enabled: false })"));
        assert!(js.contains("syncGyroStatus(m.gyro_status)"));
        assert!(js.contains("sensor data stale"));
        assert!(js.contains("output centered"));
    }

    #[test]
    fn media_safety_protocol_is_strict_coalesced_and_safe_disable_is_priority() {
        let expert: WebAction =
            serde_json::from_str(r#"{"action":"set_media_safety_mode","mode":"expert"}"#).unwrap();
        assert!(matches!(
            expert,
            WebAction::SetMediaSafetyMode {
                mode: crate::media_safety::MediaSafetyMode::Expert
            }
        ));
        assert_eq!(expert.coalesce_key().as_deref(), Some("media:safety-mode"));
        assert!(!expert.is_priority());

        let safe: WebAction =
            serde_json::from_str(r#"{"action":"set_media_safety_mode","mode":"safe"}"#).unwrap();
        assert!(safe.is_priority());
        assert_eq!(safe.coalesce_key().as_deref(), Some("media:safety-mode"));
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_media_safety_mode","mode":"unbounded"}"#
        )
        .is_err());

        let mut queue = vec![WebAction::AddRouting; MAX_PENDING_ACTIONS];
        assert_ne!(enqueue_bounded(&mut queue, safe), EnqueueOutcome::Dropped);
        assert!(matches!(
            queue.last(),
            Some(WebAction::SetMediaSafetyMode {
                mode: crate::media_safety::MediaSafetyMode::Safe
            })
        ));
    }

    #[test]
    fn proxy_settings_action_is_a_strict_coalesced_ordinary_host_action() {
        // The action always carries the complete tuple; both frame-rate forms
        // parse through the engine's own closed vocabularies.
        let source: WebAction = serde_json::from_str(
            r#"{"action":"set_proxy_settings","scale":"quarter","frame_rate":"source","include_audio":false}"#,
        )
        .unwrap();
        assert!(matches!(
            source,
            WebAction::SetProxySettings {
                scale: crate::proxy::ProxyScale::Quarter,
                frame_rate: crate::proxy::ProxyFrameRate::Source,
                include_audio: false,
            }
        ));
        let fixed: WebAction = serde_json::from_str(
            r#"{"action":"set_proxy_settings","scale":"original","frame_rate":{"fixed":{"numerator":30,"denominator":1}},"include_audio":true}"#,
        )
        .unwrap();
        assert!(matches!(
            fixed,
            WebAction::SetProxySettings {
                scale: crate::proxy::ProxyScale::Original,
                frame_rate: crate::proxy::ProxyFrameRate::Fixed {
                    numerator: 30,
                    denominator: 1,
                },
                include_audio: true,
            }
        ));

        // An unknown scale token is a deserialization rejection, never a
        // fallback onto a neighbouring choice; a partial tuple is refused.
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_proxy_settings","scale":"eighth","frame_rate":"source","include_audio":true}"#
        )
        .is_err());
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_proxy_settings","scale":"half"}"#
        )
        .is_err());

        // Ordinary host preference: coalesced under one key (newest tuple
        // wins), no admission priority.
        assert_eq!(
            source.coalesce_key().as_deref(),
            Some("host:proxy-settings")
        );
        assert!(!source.is_priority());
    }

    #[test]
    fn proxy_settings_snapshot_is_additive_and_mirrors_the_engine_default() {
        // An older snapshot without the field restores the exact default
        // tuple, so legacy clients and snapshots stay compatible.
        let mut app_value = serde_json::to_value(AppSnapshot::default()).unwrap();
        app_value.as_object_mut().unwrap().remove("proxy_settings");
        let restored: AppSnapshot = serde_json::from_value(app_value).unwrap();
        assert_eq!(restored.proxy_settings, ProxySettingsSnapshot::default());
        assert_eq!(
            restored.proxy_settings,
            ProxySettingsSnapshot::from_settings(crate::proxy::ProxySettings::default())
        );

        // Only the three operator choices cross the wire — the schema and
        // algorithm versions stay engine-internal.
        let wire = serde_json::to_value(ProxySettingsSnapshot::default()).unwrap();
        let object = wire.as_object().unwrap();
        assert_eq!(object.len(), 3);
        assert_eq!(object["scale"], "half");
        assert_eq!(object["frame_rate"], "source");
        assert_eq!(object["include_audio"], true);
    }

    #[test]
    fn media_safety_and_ntsc_metrics_default_old_snapshots_without_raising_limits() {
        let mut app_value = serde_json::to_value(AppSnapshot::default()).unwrap();
        app_value.as_object_mut().unwrap().remove("media_safety");
        let restored: AppSnapshot = serde_json::from_value(app_value).unwrap();
        assert_eq!(
            restored.media_safety.mode,
            crate::media_safety::MediaSafetyMode::Safe
        );
        assert_eq!(
            restored.media_safety.safe_max_pixels,
            crate::media_safety::SAFE_MEDIA_MAX_PIXELS
        );
        assert_eq!(restored.media_safety.planning_budget_bytes, 0);

        let mut ntsc_value = serde_json::to_value(NtscSnapshot::default()).unwrap();
        ntsc_value.as_object_mut().unwrap().remove("live_metrics");
        let restored: NtscSnapshot = serde_json::from_value(ntsc_value).unwrap();
        assert_eq!(restored.live_metrics, NtscLiveMetricsSnapshot::default());

        let metrics = NtscLiveMetricsSnapshot {
            active_path: "selective".to_string(),
            global: crate::ntsc::NtscPathMetrics {
                attempted: 7,
                accepted: 5,
                skipped: 2,
                unavailable: 0,
                stale: 1,
            },
            selective: crate::ntsc::NtscPathMetrics {
                attempted: 11,
                accepted: 9,
                skipped: 2,
                unavailable: 1,
                stale: 3,
            },
            busy: true,
        };
        let round_trip: NtscLiveMetricsSnapshot =
            serde_json::from_value(serde_json::to_value(&metrics).unwrap()).unwrap();
        assert_eq!(round_trip, metrics);
    }

    #[test]
    fn media_safety_and_vhs_metrics_static_contract_is_accessible_and_authoritative() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");

        assert!(html.contains("class=\"media-safety\""));
        assert!(html.contains("aria-labelledby=\"media-safety-title\""));
        assert!(html.contains("id=\"expert-media-toggle\""));
        assert!(html.contains("aria-label=\"Enable bounded Expert media mode\""));
        assert!(html.contains("aria-describedby=\"media-safety-summary media-safety-rationale\""));
        assert!(html.contains("id=\"media-safety-rationale\""));
        assert!(html.contains("id=\"media-safety-status\" role=\"status\" aria-live=\"polite\""));

        let metrics_tag = html
            .lines()
            .find(|line| line.contains("id=\"ntsc-metrics\""))
            .expect("dedicated VHS metrics element");
        assert!(metrics_tag.contains("aria-label=\"Live VHS worker metrics\""));
        assert!(!metrics_tag.contains("aria-live"));

        assert!(js.contains("syncMediaSafety(msg.media_safety)"));
        assert!(js.contains("action: 'set_media_safety_mode'"));
        assert!(js.contains("window.confirm("));
        assert!(js.contains("authoritativeMediaSafetyMode === 'expert'"));
        assert!(js.contains("toggleAttribute('aria-busy', false)"));
        assert!(js.contains("Final-program live"));
        assert!(!js.contains("Selective live"));
        assert!(js.contains("skipped"));
        assert!(js.contains("unavailable"));
        assert!(js.contains("stale"));
    }

    #[test]
    fn proxy_settings_static_contract_is_accessible_and_authoritative() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");

        assert!(html.contains("aria-labelledby=\"proxy-settings-title\""));
        assert!(html.contains("id=\"proxy-settings-title\""));
        assert!(html.contains("for=\"proxy-scale\""));
        assert!(html.contains("id=\"proxy-scale\""));
        assert!(html.contains("for=\"proxy-frame-rate\""));
        assert!(html.contains("id=\"proxy-frame-rate\""));
        assert!(html.contains("for=\"proxy-include-audio\""));
        assert!(html.contains("id=\"proxy-include-audio\""));
        assert!(html.contains("id=\"proxy-settings-help\""));

        assert!(js.contains("syncProxySettings(msg.proxy_settings)"));
        assert!(js.contains("action: 'set_proxy_settings'"));
        // A fixed rate another controller authored outside the presets is
        // represented honestly rather than snapped to the nearest preset.
        assert!(js.contains("data-authored-elsewhere"));

        // A host policy is never client-quantizable.
        let quantizable = js
            .split("const QUANTIZABLE_ACTIONS")
            .nth(1)
            .expect("panel declares its quantizable set")
            .split("]);")
            .next()
            .expect("quantizable set is closed");
        assert!(!quantizable.contains("set_proxy_settings"));
    }

    #[test]
    fn prepared_performance_snapshot_is_additive_and_uses_stable_image_identity() {
        let mut legacy_app = serde_json::to_value(AppSnapshot::default()).unwrap();
        legacy_app.as_object_mut().unwrap().remove("performance");
        let restored: AppSnapshot = serde_json::from_value(legacy_app).unwrap();
        assert_eq!(restored.performance, PerformanceSnapshot::default());

        let selected = ImageInputSnapshot::SelectedLayer {
            layer_id: "42".into(),
            stage: LayerImageStage::PreLocalEffects,
        };
        let wire = serde_json::to_value(&selected).unwrap();
        assert_eq!(wire["source"], "selected_layer");
        assert_eq!(wire["layer_id"], "42");
        assert_eq!(wire["stage"], "pre_local_effects");
        assert_eq!(
            serde_json::from_value::<ImageInputSnapshot>(wire).unwrap(),
            selected
        );
    }

    #[test]
    fn prepared_performance_actions_have_typed_ids_and_only_scalar_edits_coalesce() {
        let transport: WebAction = serde_json::from_str(
            r#"{"action":"set_clip_transport","layer_id":"7","slot_id":3,"param":"in_point","value":0.2}"#,
        )
        .unwrap();
        assert_eq!(
            transport.coalesce_key().as_deref(),
            Some("layer:id:7:slot:3:transport:in_point")
        );
        let cue: WebAction = serde_json::from_str(
            r#"{"action":"set_clip_cue","layer_id":"7","slot_id":3,"cue_id":12,"at":0.4}"#,
        )
        .unwrap();
        assert_eq!(
            cue.coalesce_key().as_deref(),
            Some("layer:id:7:slot:3:cue:12")
        );
        let matte: WebAction = serde_json::from_str(
            r#"{"action":"set_layer_matte_param","layer_id":"7","param":"amount","value":0.7,"composition_revision":19}"#,
        )
        .unwrap();
        assert_eq!(
            matte.coalesce_key().as_deref(),
            Some("layer:id:7:matte:amount")
        );
        assert!(matches!(
            matte,
            WebAction::SetLayerMatteParam {
                composition_revision: Some(19),
                ..
            }
        ));
        let matte_channel: WebAction = serde_json::from_str(
            r#"{"action":"set_layer_matte_param","layer_id":"7","param":"channel","value":"luma"}"#,
        )
        .unwrap();
        assert_eq!(matte_channel.coalesce_key(), None);
        assert!(matte_channel.is_priority());

        for barrier in [
            r#"{"action":"activate_clip_slot","layer_id":"7","slot_id":3}"#,
            r#"{"action":"seek_clip_slot","layer_id":"7","slot_id":3,"position":0.5}"#,
            r#"{"action":"seek_clip_slot_timecode","layer_id":"7","slot_id":3,"timecode":{"hours":0,"minutes":1,"seconds":0,"frames":2,"rate":"ntsc30_drop"}}"#,
            r#"{"action":"trigger_clip_cue","layer_id":"7","slot_id":3,"cue_id":12}"#,
            r#"{"action":"prepare_scene","scene_id":4}"#,
            r#"{"action":"capture_scene","name":"Opening study","trigger_mode":"next_bar"}"#,
            r#"{"action":"capture_scene","scene_id":4,"name":"Reframed study","trigger_mode":"next_beat"}"#,
            r#"{"action":"remove_scene","scene_id":4}"#,
            r#"{"action":"trigger_scene","scene_id":4}"#,
            r#"{"action":"set_layer_matte_input","layer_id":"7","input":{"source":"program_history"},"composition_revision":19}"#,
        ] {
            let action: WebAction = serde_json::from_str(barrier).unwrap();
            assert_eq!(action.coalesce_key(), None, "barrier coalesced: {barrier}");
            assert!(action.is_priority(), "barrier lacked reserve: {barrier}");
        }

        let mut ordered = Vec::new();
        enqueue_bounded(&mut ordered, transport.clone());
        enqueue_bounded(
            &mut ordered,
            serde_json::from_str(r#"{"action":"activate_clip_slot","layer_id":"7","slot_id":3}"#)
                .unwrap(),
        );
        enqueue_bounded(
            &mut ordered,
            serde_json::from_str(
                r#"{"action":"set_clip_transport","layer_id":"7","slot_id":3,"param":"in_point","value":0.8}"#,
            )
            .unwrap(),
        );
        assert_eq!(ordered.len(), 3, "scalar edit crossed a source barrier");

        let seek = |layer_id: &str, slot_id: u16, position: f64| WebAction::SeekClipSlot {
            layer_id: layer_id.into(),
            slot_id: ClipSlotId::new(slot_id).unwrap(),
            position: NormalizedTime::clamped(position),
        };
        let mut scratches = Vec::new();
        assert_eq!(
            enqueue_bounded(&mut scratches, seek("7", 3, 0.1)),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(&mut scratches, seek("7", 3, 0.7)),
            EnqueueOutcome::Coalesced
        );
        assert!(matches!(
            scratches.as_slice(),
            [WebAction::SeekClipSlot { layer_id, slot_id, position }]
                if layer_id == "7" && slot_id.get() == 3 && position.get() == 0.7
        ));
        let timecode_seek: WebAction = serde_json::from_str(
            r#"{"action":"seek_clip_slot_timecode","layer_id":"7","slot_id":3,"timecode":{"hours":0,"minutes":0,"seconds":2,"frames":12,"rate":"fps24"}}"#,
        )
        .unwrap();
        assert_eq!(
            enqueue_bounded(&mut scratches, timecode_seek),
            EnqueueOutcome::Coalesced,
            "adjacent normalized and timecode scratches share one newest-only slot"
        );
        assert!(matches!(
            scratches.as_slice(),
            [WebAction::SeekClipSlotTimecode { layer_id, slot_id, .. }]
                if layer_id == "7" && slot_id.get() == 3
        ));
        enqueue_bounded(
            &mut scratches,
            WebAction::ActivateClipSlot {
                layer_id: "7".into(),
                slot_id: ClipSlotId::new(3).unwrap(),
                trigger_mode: TriggerMode::Immediate,
            },
        );
        assert_eq!(
            enqueue_bounded(&mut scratches, seek("7", 3, 0.9)),
            EnqueueOutcome::Added,
            "a newer scratch must never replace a seek across a command barrier"
        );
        assert_eq!(scratches.len(), 3);
        assert_eq!(
            enqueue_bounded(&mut scratches, seek("8", 3, 0.4)),
            EnqueueOutcome::Added,
            "different stable targets must retain independent seeks"
        );

        for command in [
            WebAction::OpenPatchSnapshot,
            WebAction::OpenPatchLook { stack_revision: 9 },
            WebAction::QuickSavePatch,
            WebAction::CancelExport,
        ] {
            assert_eq!(command.coalesce_key(), None, "command was not a barrier");
        }
    }

    #[test]
    fn autopilot_protocol_replaces_one_bounded_plan_and_keeps_transport_history_free() {
        let replace: WebAction = serde_json::from_str(
            r#"{"action":"replace_autopilot_plan","plan":{"repeat":"once","steps":[{"scene_id":4,"hold_beats":2},{"scene_id":9}]}}"#,
        )
        .unwrap();
        let WebAction::ReplaceAutopilotPlan { plan } = &replace else {
            panic!("whole-plan action");
        };
        assert_eq!(plan.repeat, crate::performance::AutopilotRepeat::Once);
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps.get(0).unwrap().scene_id.get(), 4);
        assert_eq!(plan.steps.get(0).unwrap().hold_beats.get(), 2);
        assert_eq!(
            plan.steps.get(1).unwrap().hold_beats.get(),
            crate::performance::DEFAULT_AUTOPILOT_HOLD_BEATS
        );
        assert!(replace.is_priority());
        assert_eq!(replace.coalesce_key(), None);
        assert!(
            !replace.is_performance_only_for_history(),
            "authored plan replacement must reach manual history"
        );

        for wire in [
            r#"{"action":"autopilot_play"}"#,
            r#"{"action":"autopilot_pause"}"#,
            r#"{"action":"autopilot_reset"}"#,
        ] {
            let action: WebAction = serde_json::from_str(wire).unwrap();
            assert!(action.is_priority(), "transport lacked reserve: {wire}");
            assert!(
                action.is_performance_only_for_history(),
                "transport entered authored history: {wire}"
            );
            assert_eq!(action.coalesce_key(), None);
        }

        let mut saturated = vec![WebAction::AddRouting; MAX_PENDING_ACTIONS];
        assert_eq!(
            enqueue_bounded(&mut saturated, replace),
            EnqueueOutcome::Added
        );
        assert!(matches!(
            saturated.last(),
            Some(WebAction::ReplaceAutopilotPlan { .. })
        ));
    }

    #[test]
    fn autopilot_wire_rejects_invalid_beats_ids_and_oversized_sequences() {
        for invalid in [
            r#"{"action":"replace_autopilot_plan","plan":{"steps":[{"scene_id":0,"hold_beats":4}]}}"#,
            r#"{"action":"replace_autopilot_plan","plan":{"steps":[{"scene_id":1,"hold_beats":0}]}}"#,
            r#"{"action":"replace_autopilot_plan","plan":{"steps":[{"scene_id":1,"hold_beats":257}]}}"#,
        ] {
            assert!(
                serde_json::from_str::<WebAction>(invalid).is_err(),
                "invalid Autopilot wire was accepted: {invalid}"
            );
        }

        let steps = (0..=crate::performance::MAX_AUTOPILOT_STEPS)
            .map(|_| serde_json::json!({"scene_id": 1, "hold_beats": 4}))
            .collect::<Vec<_>>();
        assert!(
            serde_json::from_value::<WebAction>(serde_json::json!({
                "action": "replace_autopilot_plan",
                "plan": {"repeat": "loop", "steps": steps}
            }))
            .is_err(),
            "the web protocol must retain the domain allocation bound"
        );
    }

    #[test]
    fn autopilot_snapshot_defaults_legacy_clients_and_exhaustively_maps_runtime_phase() {
        let legacy: PerformanceSnapshot = serde_json::from_str("{}").unwrap();
        assert!(legacy.autopilot.plan.is_empty());
        assert_eq!(legacy.autopilot.phase, AutopilotPhaseSnapshot::Stopped);
        assert_eq!(legacy.autopilot.current_step, None);

        for (runtime, snapshot, wire) in [
            (
                AutopilotState::Stopped,
                AutopilotPhaseSnapshot::Stopped,
                "stopped",
            ),
            (
                AutopilotState::Starting,
                AutopilotPhaseSnapshot::Starting,
                "starting",
            ),
            (
                AutopilotState::Running,
                AutopilotPhaseSnapshot::Running,
                "running",
            ),
            (
                AutopilotState::Paused,
                AutopilotPhaseSnapshot::Paused,
                "paused",
            ),
            (
                AutopilotState::Stalled,
                AutopilotPhaseSnapshot::Stalled,
                "stalled",
            ),
            (
                AutopilotState::Faulted,
                AutopilotPhaseSnapshot::Faulted,
                "faulted",
            ),
            (
                AutopilotState::Complete,
                AutopilotPhaseSnapshot::Complete,
                "complete",
            ),
        ] {
            let mapped = AutopilotPhaseSnapshot::from(runtime);
            assert_eq!(mapped, snapshot);
            assert_eq!(serde_json::to_value(mapped).unwrap(), wire);
        }
    }

    #[test]
    fn prepared_performance_ids_reject_zero_or_out_of_range_at_deserialization() {
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"activate_clip_slot","layer_id":"7","slot_id":0}"#
        )
        .is_err());
        assert!(
            serde_json::from_str::<WebAction>(r#"{"action":"prepare_scene","scene_id":0}"#)
                .is_err()
        );
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"trigger_clip_cue","layer_id":"7","slot_id":3,"cue_id":4096}"#
        )
        .is_err());
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"capture_scene","scene_id":0,"name":"Invalid"}"#
        )
        .is_err());
        assert!(
            serde_json::from_str::<WebAction>(r#"{"action":"remove_scene","scene_id":65536}"#)
                .is_err()
        );
    }

    #[test]
    fn scene_capture_protocol_distinguishes_create_recapture_and_remove() {
        let create: WebAction = serde_json::from_str(
            r#"{"action":"capture_scene","name":"Runway cut","trigger_mode":"next_bar"}"#,
        )
        .unwrap();
        assert!(matches!(
            &create,
            WebAction::CaptureScene {
                scene_id: None,
                name,
                trigger_mode: TriggerMode::NextBar,
            } if name == "Runway cut"
        ));
        let create_wire = serde_json::to_value(&create).unwrap();
        assert_eq!(create_wire["action"], "capture_scene");
        assert!(create_wire.get("scene_id").is_none());

        let recapture: WebAction = serde_json::from_str(
            r#"{"action":"capture_scene","scene_id":23,"name":"Runway cut II","trigger_mode":"next_beat"}"#,
        )
        .unwrap();
        assert!(matches!(
            &recapture,
            WebAction::CaptureScene {
                scene_id: Some(id),
                name,
                trigger_mode: TriggerMode::NextBeat,
            } if id.get() == 23 && name == "Runway cut II"
        ));
        assert_eq!(recapture.coalesce_key(), None);
        assert!(recapture.is_priority());

        let remove: WebAction =
            serde_json::from_str(r#"{"action":"remove_scene","scene_id":23}"#).unwrap();
        assert!(matches!(remove, WebAction::RemoveScene { scene_id } if scene_id.get() == 23));
        assert_eq!(remove.coalesce_key(), None);
        assert!(remove.is_priority());

        let mut ordered = Vec::new();
        assert_eq!(
            enqueue_bounded(&mut ordered, WebAction::SetBpm { value: 90.0 }),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(&mut ordered, create.clone()),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(&mut ordered, WebAction::SetBpm { value: 130.0 }),
            EnqueueOutcome::Added,
            "capture must prevent scalar traffic from coalescing across it"
        );
        assert!(matches!(&ordered[0], WebAction::SetBpm { value } if *value == 90.0));
        assert!(matches!(&ordered[1], WebAction::CaptureScene { .. }));
        assert!(matches!(&ordered[2], WebAction::SetBpm { value } if *value == 130.0));

        let mut saturated = vec![WebAction::AddRouting; MAX_PENDING_ACTIONS];
        assert_eq!(
            enqueue_bounded(&mut saturated, create),
            EnqueueOutcome::Added
        );
        assert_eq!(saturated.len(), MAX_PENDING_ACTIONS);
        assert!(matches!(
            saturated.last(),
            Some(WebAction::CaptureScene { .. })
        ));
    }

    #[test]
    fn prepared_performance_browser_contract_is_explicit_stable_and_accessible() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        let css = include_str!("../../static/style.css");
        for contract in [
            "id=\"library-slot-target\"",
            "id=\"library-slot-trigger\"",
            "id=\"slot-load-status\" class=\"lib-upload-status\" role=\"status\" aria-live=\"polite\"",
            "id=\"scene-list\" role=\"list\"",
            "id=\"scene-status\" role=\"status\" aria-live=\"polite\"",
            "id=\"scene-capture-name\"",
            "id=\"scene-capture-mode\"",
            "id=\"scene-capture-submit\"",
            "id=\"autopilot-plan-form\"",
            "id=\"autopilot-repeat\"",
            "id=\"autopilot-step-list\" class=\"autopilot-step-list\" role=\"list\"",
            "id=\"autopilot-add-step\"",
            "id=\"autopilot-apply-plan\"",
            "id=\"autopilot-play\"",
            "id=\"autopilot-pause\"",
            "id=\"autopilot-reset\"",
            "id=\"autopilot-status\" class=\"scene-status autopilot-status\" role=\"status\" aria-live=\"polite\"",
        ] {
            assert!(
                html.contains(contract),
                "missing accessible HTML: {contract}"
            );
        }
        assert!(js.contains("clipSeek.addEventListener('input', sendSeek)"));
        assert!(js.contains("clipSeek.addEventListener('change', sendSeek)"));
        assert!(js.contains("class=\"clip-timecode-hours\""));
        assert!(js.contains("class=\"clip-timecode-rate\""));
        for action in [
            "action: 'load_clip_into_slot'",
            "action: 'activate_clip_slot'",
            "action: 'set_clip_transport'",
            "action: 'set_clip_cue'",
            "action: 'trigger_clip_cue'",
            "action: 'seek_clip_slot'",
            "action: 'seek_clip_slot_timecode'",
            "action: 'prepare_scene'",
            "action: 'capture_scene'",
            "action: 'remove_scene'",
            "action: 'trigger_scene'",
            "action: 'replace_autopilot_plan'",
            "action: 'set_layer_matte_param'",
            "action: 'set_layer_matte_input'",
        ] {
            assert!(js.contains(action), "missing browser action: {action}");
        }
        assert!(js.contains("function stableLayerId(layer)"));
        assert!(js.contains("/^(?:[1-9][0-9]*)$/"));
        assert!(js.contains("source: 'selected_layer', layer_id: donorId, stage"));
        assert!(js.contains("Staging ${filename}; the current source stays live until ready."));
        assert!(js.contains("source_staging_status"));
        assert!(js.contains("scene_staging_status"));
        assert!(js.contains("image_routing_status"));
        assert!(js.contains("function validSceneName(name)"));
        assert!(js.contains("new TextEncoder().encode(name).length <= 128"));
        assert!(js.contains("class=\"scene-recapture\""));
        assert!(js.contains("class=\"scene-remove\""));
        assert!(js.contains("sendAutopilotTransport('autopilot_play'"));
        assert!(js.contains("sendAutopilotTransport('autopilot_pause'"));
        assert!(js.contains("sendAutopilotTransport('autopilot_reset'"));
        assert!(js.contains("Missing Scene ${sceneId} (kept)"));
        assert!(js.contains("tombstone.dataset.tombstone = ''"));
        assert!(css.contains(".scene-row.pending"));
        assert!(css.contains(".scene-capture-form"));
        assert!(css.contains(".autopilot-step-row.current"));
        assert!(css.contains(".autopilot-step-row.next"));
        assert!(css.contains(".autopilot-step-row.missing"));
        assert!(css.contains(".library-actions"));
        assert!(css.contains(".layer-performance-body[hidden]"));
        assert!(css.contains(".layer-matte-body[hidden]"));
    }

    #[test]
    fn creative_browser_contract_covers_authoring_stable_targets_and_missing_tombstones() {
        let js = include_str!("../../static/app.js");
        assert!(js.contains("syncCreative(msg.creative)"));
        assert!(js.contains("function syncCreative(creative)"));
        for action in [
            "action: 'set_visual_node_param'",
            "action: 'insert_visual_node'",
            "action: 'remove_visual_node'",
            "action: 'move_visual_node'",
            "action: 'set_visual_node_mask_variant'",
            "action: 'set_visual_node_route'",
            "action: 'set_visual_node_displace_route'",
            "action: 'set_visual_node_symmetry_route'",
            "action: 'set_visual_node_displace_route'",
            "action: 'set_visual_node_residual_route'",
            "action: 'set_composition_group_param'",
            "action: 'set_composition_group_matte_route'",
            "action: 'set_composition_group_matte_param'",
            "action: 'create_composition_group'",
            "action: 'remove_composition_group'",
            "action: 'set_composition_group_members'",
            "action: 'move_composition_root_item'",
            "action: 'set_composition_bus_crossfade'",
            "action: 'set_composition_layer_bus'",
        ] {
            assert!(
                js.contains(action),
                "missing creative browser action: {action}"
            );
        }
        for stable_prefix in [
            "node/${scopeKey}/${node.node_id}",
            "nodeTargets('master'",
            "nodeTargets(`layer/${layerId}`",
            "nodeTargets(`group/${group.group_id}`",
            "`group/${group.group_id}/opacity`",
            "`group/${group.group_id}/matte.amount`",
            "`group/${group.group_id}/matte.threshold`",
            "`group/${group.group_id}/matte.softness`",
            "['composition/bus_crossfade'",
        ] {
            assert!(
                js.contains(stable_prefix),
                "missing stable modulation prefix: {stable_prefix}"
            );
        }

        // Missing routes are rendered as disabled diagnostics only. Every
        // ingress construction path accepts current stable IDs and never
        // constructs either output-only tombstone variant.
        assert!(!js.contains("input = { source: 'missing_selected_layer'"));
        assert!(!js.contains("input = { source: 'missing_group_output'"));
        assert!(js.contains("selected disabled>Missing group"));
        assert!(js.contains("input = { source: 'group_output', group_id: groupId }"));
        assert!(js.contains("const groupOutputs = (latestCreative?.groups || []).map"));
        assert!(js.contains("composition_revision: compositionRevision"));
        assert!(js.contains("|| scope === 'all' && !!rerollGroupControls?.checked"));
        let html = include_str!("../../static/index.html");
        assert!(html.contains("id=\"reroll-group-controls\""));
        assert!(html.contains("matte, and A/B value variation are explicit opt-ins"));
        assert!(html.contains("bounded Loom, Atlas, and Garden numeric values"));
        assert!(html.contains("bounded Motion numeric values for Master / Layer / Everything"));
        assert!(html.contains(
            "never changes Motion algorithm/version, field or quality tiers, donor, carrier"
        ));
        assert!(
            html.contains("temporal topology, seeds, Score configuration, reset law, loop driver")
        );
    }

    #[test]
    fn m5_gesture_boundaries_coalesce_one_scalar_without_crossing_ordered_barriers() {
        let begin = WebAction::BeginHistoryGesture { gesture_id: 41 };
        let end = WebAction::EndHistoryGesture { gesture_id: 41 };
        assert!(begin.is_priority());
        assert!(end.is_priority());
        assert_eq!(begin.coalesce_key(), None);
        let mut queue = Vec::new();
        enqueue_bounded(&mut queue, begin);
        enqueue_bounded(
            &mut queue,
            WebAction::SetParam {
                param: "brightness".into(),
                value: serde_json::json!(0.2),
            },
        );
        enqueue_bounded(
            &mut queue,
            WebAction::SetParam {
                param: "brightness".into(),
                value: serde_json::json!(0.8),
            },
        );
        enqueue_bounded(&mut queue, end);
        assert_eq!(queue.len(), 3);
        assert!(matches!(queue[0], WebAction::BeginHistoryGesture { .. }));
        assert!(matches!(
            &queue[1],
            WebAction::SetParam { value, .. } if value == &serde_json::json!(0.8)
        ));
        assert!(matches!(queue[2], WebAction::EndHistoryGesture { .. }));
    }

    #[tokio::test]
    async fn m5_remote_gesture_owner_rejects_dirty_cancel_cross_client_and_second_destination() {
        let state = WebState::new().expect("test web state");
        assert!(state.begin_browser_history_gesture(10, 41).await);
        assert!(!state.begin_browser_history_gesture(11, 42).await);

        let brightness = WebAction::SetParam {
            param: "brightness".into(),
            value: serde_json::json!(0.2),
        };
        let another_brightness = WebAction::SetParam {
            param: "brightness".into(),
            value: serde_json::json!(0.8),
        };
        let contrast = WebAction::SetParam {
            param: "contrast".into(),
            value: serde_json::json!(0.5),
        };
        assert!(
            state
                .admit_browser_action_during_gesture(10, &brightness)
                .await
        );
        assert!(
            state
                .admit_browser_action_during_gesture(10, &another_brightness)
                .await
        );
        assert!(
            !state
                .admit_browser_action_during_gesture(10, &contrast)
                .await
        );
        assert!(
            !state
                .admit_browser_action_during_gesture(11, &brightness)
                .await
        );
        assert!(
            state
                .admit_browser_action_during_gesture(11, &WebAction::TriggerCollisionScore)
                .await
        );
        assert!(!state.may_finish_browser_history_gesture(10, 41, true).await);
        assert!(
            state
                .may_finish_browser_history_gesture(10, 41, false)
                .await
        );
        state.finish_browser_history_gesture(10, 41).await;
        assert!(
            !state
                .may_finish_browser_history_gesture(10, 41, false)
                .await
        );

        assert!(state.begin_browser_history_gesture(10, 43).await);
        assert!(state.may_finish_browser_history_gesture(10, 43, true).await);
    }

    #[tokio::test]
    async fn m5_disconnect_queues_old_end_before_a_new_client_can_begin() {
        let state = WebState::new().expect("test web state");
        assert!(state.begin_browser_history_gesture(10, 41).await);
        assert_eq!(
            state
                .enqueue_action(WebAction::BeginHistoryGesture { gesture_id: 41 })
                .await,
            EnqueueOutcome::Added
        );

        assert_eq!(state.disconnect_browser_history_gesture(10).await, Some(41));
        assert!(state.begin_browser_history_gesture(11, 42).await);
        assert_eq!(
            state
                .enqueue_action(WebAction::BeginHistoryGesture { gesture_id: 42 })
                .await,
            EnqueueOutcome::Added
        );

        let queue = state.actions.lock().await;
        assert_eq!(queue.len(), 3);
        assert!(matches!(
            queue[0].payload(),
            WebAction::BeginHistoryGesture { gesture_id: 41 }
        ));
        assert!(matches!(
            queue[1].payload(),
            WebAction::EndHistoryGesture { gesture_id: 41 }
        ));
        assert!(matches!(
            queue[2].payload(),
            WebAction::BeginHistoryGesture { gesture_id: 42 }
        ));
    }

    #[test]
    fn m5_history_preset_and_recovery_snapshot_fields_are_additive_defaults() {
        let mut legacy = serde_json::to_value(AppSnapshot::default()).unwrap();
        let object = legacy.as_object_mut().unwrap();
        for field in [
            "history",
            "presets",
            "recovery_available",
            "recovery_status",
            "export_motion",
            "controller_runtime",
            "osc_runtime",
        ] {
            object.remove(field);
        }
        let restored: AppSnapshot = serde_json::from_value(legacy).unwrap();
        assert_eq!(restored.history, HistorySnapshot::default());
        assert_eq!(restored.presets, PresetLibrarySnapshot::default());
        assert!(!restored.recovery_available);
        assert_eq!(restored.export_motion, ExportMotionSnapshot::default());
        assert_eq!(
            restored.controller_runtime,
            ControllerRuntimeSnapshot::default()
        );
        assert_eq!(restored.osc_runtime, OscRuntimeSnapshot::default());
    }

    #[test]
    fn controller_and_osc_runtime_dtos_map_all_bounded_facts_and_exposure_truth() {
        let profile = crate::controller_profile::ControllerProfileDocument::default();
        let midi = crate::midi::MidiRuntimeSnapshot {
            phase: crate::midi::MidiConnectionPhase::Connected,
            input_port: Some("Input A".into()),
            output_port: Some("Output B".into()),
            available_inputs: vec!["Input A".into()],
            available_outputs: vec!["Output B".into()],
            error: None,
            counters: crate::midi::MidiCounters {
                raw_received: 12,
                decoded_events: 7,
                feedback_sent: 3,
                reconnects: 2,
                ..Default::default()
            },
        };
        let controller = ControllerRuntimeSnapshot::from_runtime(
            9,
            &profile,
            "profile ready\ncontrol removed",
            &midi,
        );
        assert_eq!(controller.profile_revision, 9);
        assert_eq!(controller.name, "Legacy four CC");
        assert_eq!(controller.status, "profile readycontrol removed");
        assert_eq!(controller.midi.phase, "connected");
        assert_eq!(controller.midi.input_port, "Input A");
        assert_eq!(controller.midi.counters.raw_received, 12);
        assert_eq!(controller.midi.counters.decoded_events, 7);
        assert_eq!(controller.midi.counters.feedback_sent, 3);
        assert_eq!(controller.midi.counters.reconnects, 2);

        let config = crate::osc::OscConfigDocument {
            bind: crate::osc::OscBindMode::Lan {
                port: 9_111,
                enabled: true,
            },
            ..Default::default()
        };
        let runtime = crate::osc::OscRuntimeSnapshot {
            phase: crate::osc::OscRuntimePhase::Listening,
            bound_address: Some("0.0.0.0:9111".parse().unwrap()),
            lan_warning: true,
            error: None,
            counters: crate::osc::OscCounters {
                datagrams_received: 22,
                messages_received: 19,
                queue_dropped: 4,
                feedback_send_errors: 1,
                ..Default::default()
            },
        };
        let osc = OscRuntimeSnapshot::from_runtime(&config, "", &runtime);
        assert_eq!(osc.phase, "listening");
        assert_eq!(osc.bind_address, "0.0.0.0:9111");
        assert_eq!(osc.bound_address, "0.0.0.0:9111");
        assert!(osc.running);
        assert!(osc.lan_warning);
        assert_eq!(osc.counters.datagrams_received, 22);
        assert_eq!(osc.counters.messages_received, 19);
        assert_eq!(osc.counters.queue_dropped, 4);
        assert_eq!(osc.counters.feedback_send_errors, 1);
        let json = serde_json::to_value(OscRuntimeSnapshot::default()).unwrap();
        assert_eq!(json.get("lan_warning"), Some(&serde_json::json!(false)));

        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        assert!(html.contains("id=\"controller-profile-import\""));
        assert!(html.contains("id=\"controller-profile-export\""));
        assert!(html.contains("MIDI input is never echoed back to the same origin"));
        assert!(js.contains("fetch('/controller-profile'"));
        assert!(js.contains("{ action: 'import', document: documentValue }"));
        assert!(js.contains("{ action: 'export' }"));
        assert!(!js.contains("controller_profile_path"));
    }

    #[test]
    fn export_motion_snapshot_exposes_bounded_provenance_without_field_pixels() {
        let metadata = crate::render_export::ExportMotionMetadata {
            accepted_frame: Some(12),
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            scopes: vec![crate::render_export::ExportMotionScopeMetadata {
                scope: crate::render_export::ExportMotionScopeIdentity::Layer {
                    saved_position: 2,
                    stable_id: 9,
                    source_tap_id: 17,
                },
                algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
                requested_source: crate::motion::MotionFieldSource::Auto,
                lattice_quality: crate::motion::MotionLatticeQuality::Live,
                source_origin: crate::motion::MotionFieldOrigin::LatticeFallback,
                rendered_source_origin: crate::motion::MotionFieldOrigin::LatticeFallback,
                field_planned: true,
                field_attached: true,
                source_diagnostic: crate::motion::MotionSourceDiagnostic::CodecUnavailableFallback,
                codec_provenance: None,
                source_generation: None,
                frame_ordinal: None,
                codec_product_sha256: None,
                codec_transition_count: None,
                codec_elapsed_seconds: None,
                donor_saved_position: Some(1),
                donor_stable_id: Some(7),
                carrier: crate::motion::MotionCarrier::Transparent,
                transplant_admitted: true,
                shutter_active: true,
                shutter_angle_degrees: 90.0,
                shutter_quality: crate::motion::CurvedShutterQuality::Live,
                shutter_sample_count: 8,
            }],
            scopes_truncated: true,
            field_collider: None,
        };
        let snapshot = ExportMotionSnapshot::from_export(&metadata, &["fallback".into()]);
        assert_eq!(snapshot.accepted_frame, Some(12));
        assert_eq!(snapshot.scopes[0].source_origin, "lattice_fallback");
        assert_eq!(
            snapshot.scopes[0].rendered_source_origin,
            "lattice_fallback"
        );
        assert!(snapshot.scopes[0].field_attached);
        assert_eq!(snapshot.scopes[0].stable_id, "9");
        assert_eq!(snapshot.scopes[0].codec_transition_count, None);
        assert_eq!(snapshot.scopes[0].codec_elapsed_seconds, None);
        assert!(snapshot.scopes_truncated);
        assert!(!snapshot.cross_gpu_pixel_identity_guaranteed);
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(!json.contains("vectors"));
        assert!(!json.contains("pixels"));
    }

    #[test]
    fn export_motion_snapshot_omits_codec_provenance_when_explicit_codec_is_unattached() {
        let metadata = crate::render_export::ExportMotionMetadata {
            accepted_frame: Some(31),
            algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
            scopes: vec![crate::render_export::ExportMotionScopeMetadata {
                scope: crate::render_export::ExportMotionScopeIdentity::Layer {
                    saved_position: 0,
                    stable_id: 41,
                    source_tap_id: 73,
                },
                algorithm_version: crate::motion::MOTION_ALGORITHM_VERSION,
                requested_source: crate::motion::MotionFieldSource::CodecVectors,
                lattice_quality: crate::motion::MotionLatticeQuality::Live,
                source_origin: crate::motion::MotionFieldOrigin::None,
                rendered_source_origin: crate::motion::MotionFieldOrigin::None,
                field_planned: true,
                field_attached: false,
                source_diagnostic: crate::motion::MotionSourceDiagnostic::CodecUnavailable,
                codec_provenance: None,
                source_generation: None,
                frame_ordinal: None,
                codec_product_sha256: None,
                codec_transition_count: None,
                codec_elapsed_seconds: None,
                donor_saved_position: None,
                donor_stable_id: None,
                carrier: crate::motion::MotionCarrier::Transparent,
                transplant_admitted: false,
                shutter_active: true,
                shutter_angle_degrees: 180.0,
                shutter_quality: crate::motion::CurvedShutterQuality::Live,
                shutter_sample_count: crate::motion::CurvedShutterQuality::Live.sample_count(),
            }],
            scopes_truncated: false,
            field_collider: None,
        };
        let snapshot = ExportMotionSnapshot::from_export(&metadata, &[]);
        let scope = &snapshot.scopes[0];
        assert_eq!(scope.requested_source, "codec_vectors");
        assert_eq!(scope.source_origin, "none");
        assert_eq!(scope.rendered_source_origin, "none");
        assert!(scope.field_planned);
        assert!(!scope.field_attached);
        assert!(scope.codec_provenance.is_empty());
        assert!(scope.codec_product_sha256.is_empty());
        assert_eq!(scope.source_generation, None);
        assert_eq!(scope.frame_ordinal, None);

        let json = serde_json::to_value(&snapshot).unwrap();
        let scope_json = &json["scopes"][0];
        for absent in [
            "codec_provenance",
            "source_generation",
            "frame_ordinal",
            "codec_product_sha256",
            "codec_transition_count",
            "codec_elapsed_seconds",
        ] {
            assert!(
                scope_json.get(absent).is_none(),
                "unattached explicit-codec snapshot serialized {absent}: {scope_json}"
            );
        }
    }

    #[test]
    fn m5_browser_contract_is_accessible_and_never_auto_restores() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        for needle in [
            "id=\"history-undo\"",
            "id=\"history-redo\"",
            "id=\"preset-list\"",
            "id=\"recovery-restore\"",
            "id=\"recovery-discard\"",
            "id=\"export-motion-status\"",
            "id=\"controller-runtime-panel\" aria-labelledby=\"controller-runtime-heading\"",
            "id=\"midi-runtime-counters\" aria-label=\"MIDI runtime counters\"",
            "id=\"osc-runtime-panel\" aria-labelledby=\"osc-runtime-heading\"",
            "id=\"osc-runtime-lan-warning\" role=\"status\" aria-live=\"polite\"",
            "id=\"osc-runtime-counters\" aria-label=\"OSC runtime counters\"",
        ] {
            assert!(html.contains(needle), "missing browser control {needle}");
        }
        for action in [
            "action: 'begin_history_gesture'",
            "action: 'end_history_gesture'",
            "action: 'undo_manual'",
            "action: 'redo_manual'",
            "action: 'capture_scoped_preset'",
            "action: 'apply_scoped_preset'",
            "action: 'delete_scoped_preset'",
            "action: 'restore_recovery_journal'",
            "action: 'discard_recovery_journal'",
        ] {
            assert!(js.contains(action), "missing browser action {action}");
        }
        assert!(!js.contains("restore_recovery_journal' });\nconnect"));
        assert!(html.contains("will never be applied automatically"));
        assert!(html.contains("value=\"controller_profile\">Controller Profile"));
        assert!(html.contains("value=\"stage_map\">Stage Map"));
        assert!(html.contains("complete, separately bounded typed documents"));
        assert!(js.contains("return { scope: 'controller_profile' }"));
        assert!(js.contains("return { scope: 'stage_map' }"));
        assert!(js.contains("syncControllerRuntime(msg.controller_runtime)"));
        assert!(js.contains("syncOscRuntime(msg.osc_runtime)"));
        assert!(js.contains("function syncControllerRuntime(snapshot = {})"));
        assert!(js.contains("function syncOscRuntime(runtime = {})"));
        assert!(js.contains("if (activeHistoryGestures.size) return false;"));
        assert!(html.contains("Browser messages cannot invent OSC addresses"));
        assert!(!js.contains("action: 'dispatch_osc_json'"));
    }

    #[test]
    fn displace_route_action_is_an_uncoalesced_ordered_barrier() {
        let route = CreativeImageTapSnapshot {
            input: CreativeImageSourceSnapshot::OneBelow,
            timing: EdgeTiming::CurrentFrame,
        };
        let action = WebAction::SetVisualNodeDisplaceRoute {
            scope: CreativeScopeSnapshot::Master,
            node_id: "7".into(),
            route: route.clone(),
            composition_revision: 12,
        };

        // Topology edits are ordered, revision-protected barriers and must
        // never coalesce behind a later absolute value.
        assert!(action.is_priority());
        assert!(action.coalesce_key().is_none());

        // The wire name is explicit and the payload round trips exactly.
        let value = serde_json::to_value(&action).unwrap();
        assert_eq!(value["action"], "set_visual_node_displace_route");
        assert_eq!(value["composition_revision"], 12);
        assert_eq!(value["input"], serde_json::Value::Null);
        let WebAction::SetVisualNodeDisplaceRoute {
            scope: decoded_scope,
            node_id: decoded_node,
            route: decoded_route,
            composition_revision: decoded_revision,
        } = serde_json::from_value::<WebAction>(value).unwrap()
        else {
            panic!("the displace route action must decode to its own variant");
        };
        assert_eq!(decoded_scope, CreativeScopeSnapshot::Master);
        assert_eq!(decoded_node, "7");
        assert_eq!(decoded_route, route);
        assert_eq!(decoded_revision, 12);

        // The well-formed message is accepted...
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_visual_node_displace_route","scope":{"scope":"master"},"node_id":"7","route":{"input":{"source":"one_below"},"timing":"current_frame"},"composition_revision":1}"#
        )
        .is_ok());
        // ...the revision is mandatory, so a route edit can never arrive
        // unbarriered...
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_visual_node_displace_route","scope":{"scope":"master"},"node_id":"7","route":{"input":{"source":"one_below"},"timing":"current_frame"}}"#
        )
        .is_err());
        // ...and an unknown donor token is a closed-vocabulary rejection
        // rather than a defaulted route.
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_visual_node_displace_route","scope":{"scope":"master"},"node_id":"7","route":{"input":{"source":"teleport"},"timing":"current_frame"},"composition_revision":1}"#
        )
        .is_err());
    }

    #[test]
    fn displace_snapshot_publishes_its_route_gains_boundary_and_diagnostic() {
        use crate::visual_rack::{
            DisplaceBoundary, ResolvedImageSource, ResolvedImageTap, RuntimeDisplaceParams,
            RuntimeVisualNodeKind,
        };

        let live = creative_node_params(RuntimeVisualNodeKind::Displace(RuntimeDisplaceParams {
            tap: ResolvedImageTap {
                source: ResolvedImageSource::OneBelow,
                timing: EdgeTiming::CurrentFrame,
            },
            amount_x: 0.25,
            amount_y: -0.75,
            boundary: DisplaceBoundary::Mirror,
        }));
        assert_eq!(live["amount_x"], 0.25);
        assert_eq!(live["amount_y"], -0.75);
        assert_eq!(live["boundary"], "mirror");
        assert_eq!(live["donor_tap"]["input"]["source"], "one_below");
        assert_eq!(live["donor_tap"]["timing"], "current_frame");
        assert_eq!(
            live["diagnostic"], "",
            "a live route reports no diagnostic text"
        );

        // A retained tombstone names its saved provenance instead of rebinding.
        let missing =
            creative_node_params(RuntimeVisualNodeKind::Displace(RuntimeDisplaceParams {
                tap: ResolvedImageTap {
                    source: ResolvedImageSource::MissingSelectedLayer {
                        saved_position: crate::performance::SavedLayerPosition::new(4).unwrap(),
                        stage: crate::image_routing::LayerImageStage::PostLocalEffects,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                ..RuntimeDisplaceParams::default()
            }));
        assert_eq!(missing["diagnostic"], "missing saved layer 4");
        assert_eq!(
            missing["donor_tap"]["input"]["source"],
            "missing_selected_layer"
        );
    }

    /// The snapshot publishes both image slots and both motion slots by slot
    /// index, each with its own diagnostic, so a controller can tell which of
    /// the four routes lost its donor without inferring it from an ordering.
    /// A retained tombstone names its saved provenance and never rebinds, and
    /// no process identity is ever published.
    #[test]
    fn symmetry_snapshot_publishes_both_image_slots_both_motion_slots_and_its_masks() {
        use crate::motion::MotionDonor;
        use crate::symmetry::{
            RuntimeSymmetryParams, SymmetryBoundary, SymmetryMode, SymmetryMotionMask,
            SymmetrySourceMask,
        };
        use crate::visual_rack::{ResolvedImageSource, ResolvedImageTap, RuntimeVisualNodeKind};

        let position = |value: u32| crate::performance::SavedLayerPosition::new(value).unwrap();
        let live = creative_node_params(RuntimeVisualNodeKind::Symmetry(RuntimeSymmetryParams {
            mode: SymmetryMode::PlanarPm,
            boundary: SymmetryBoundary::CellularReentry,
            base_folds: 6.0,
            fold_offset: 0.4,
            hue_span: 0.5,
            motion_gain: -0.25,
            seed: 77,
            source_mask: SymmetrySourceMask {
                carrier: true,
                donor0: false,
                donor1: true,
                clean_history: true,
            },
            motion_mask: SymmetryMotionMask {
                slot0: true,
                slot1: false,
            },
            donors: [
                ResolvedImageTap {
                    source: ResolvedImageSource::OneBelow,
                    timing: EdgeTiming::CurrentFrame,
                },
                ResolvedImageTap {
                    source: ResolvedImageSource::MissingSelectedLayer {
                        saved_position: position(4),
                        stage: crate::image_routing::LayerImageStage::PostLocalEffects,
                    },
                    timing: EdgeTiming::PreviousFrame,
                },
            ],
            motion: [
                MotionDonor::Selected {
                    layer_id: crate::image_routing::StableLayerId::new(910).unwrap(),
                    saved_position: position(2),
                },
                MotionDonor::Missing {
                    saved_position: position(6),
                },
            ],
            ..RuntimeSymmetryParams::default()
        }));

        assert_eq!(live["symmetry_mode"], "planar_pm");
        assert_eq!(live["symmetry_boundary"], "cellular_reentry");
        assert_eq!(live["symmetry_seed"], 77);
        assert_eq!(live["symmetry_base_folds"], 6.0);
        assert_eq!(live["symmetry_fold_offset"], f64::from(0.4_f32));
        assert_eq!(
            live["symmetry_effective_folds"], 6,
            "the published fold count is rounded exactly once"
        );
        assert_eq!(live["symmetry_hue_span"], 0.5);
        assert_eq!(live["symmetry_motion_gain"], -0.25);
        assert_eq!(live["symmetry_exact_bypass"], false);

        assert_eq!(live["symmetry_source_carrier"], true);
        assert_eq!(live["symmetry_source_donor0"], false);
        assert_eq!(live["symmetry_source_donor1"], true);
        assert_eq!(live["symmetry_source_history"], true);
        assert_eq!(live["symmetry_motion_slot0"], true);
        assert_eq!(live["symmetry_motion_slot1"], false);

        assert_eq!(live["symmetry_donor0_tap"]["input"]["source"], "one_below");
        assert_eq!(live["symmetry_donor0_tap"]["timing"], "current_frame");
        assert_eq!(live["donor0_diagnostic"], "");
        assert_eq!(
            live["symmetry_donor1_tap"]["input"]["source"],
            "missing_selected_layer"
        );
        assert_eq!(live["symmetry_donor1_tap"]["timing"], "previous_frame");
        assert_eq!(live["donor1_diagnostic"], "missing saved layer 4");

        // A live motion slot publishes the same stable decimal layer ID the
        // image slots already publish, because that ID is how the panel
        // addresses a layer. A tombstone publishes only its saved provenance
        // and never a layer ID, so it can never rebind.
        assert_eq!(live["symmetry_motion0_donor"]["kind"], "selected");
        assert_eq!(live["symmetry_motion0_donor"]["layer_id"], "910");
        assert_eq!(live["symmetry_motion0_donor"]["saved_position"], 2);
        assert_eq!(live["motion0_diagnostic"], "");
        assert_eq!(live["symmetry_motion1_donor"]["kind"], "missing");
        assert_eq!(live["symmetry_motion1_donor"]["saved_position"], 6);
        assert!(
            live["symmetry_motion1_donor"]["layer_id"].is_null(),
            "a retained motion tombstone never publishes a layer identity"
        );
        assert_eq!(live["motion1_diagnostic"], "missing saved layer 6");

        let neutral = creative_node_params(RuntimeVisualNodeKind::Symmetry(
            RuntimeSymmetryParams::default(),
        ));
        assert_eq!(neutral["symmetry_exact_bypass"], true);
        assert_eq!(neutral["symmetry_motion0_donor"]["kind"], "none");
        assert_eq!(neutral["symmetry_effective_folds"], 1);
    }

    /// The Scan Processor snapshot publishes all nineteen authored values
    /// plus the derived read-only wake law and the vertex request; the
    /// default reads as an exact bypass.
    #[test]
    fn scan_processor_snapshot_publishes_every_value_and_the_derived_wake_law() {
        use crate::scan_processor::ScanProcessorParams;
        use crate::visual_rack::RuntimeVisualNodeKind;

        let live =
            creative_node_params(RuntimeVisualNodeKind::ScanProcessor(ScanProcessorParams {
                amount: 0.5,
                lines: 240,
                samples_per_line: 96,
                reverse_h: true,
                osc_freq: 0.75,
                mono: 0.25,
                ..ScanProcessorParams::default()
            }));
        assert_eq!(live["scan_lines"], 240);
        assert_eq!(live["scan_samples"], 96);
        assert_eq!(live["scan_amount"], 0.5);
        assert_eq!(live["scan_reverse_h"], true);
        assert_eq!(live["scan_reverse_v"], false);
        assert_eq!(live["scan_osc_freq"], 0.75);
        assert_eq!(live["scan_mono"], 0.25);
        assert_eq!(live["scan_ribbon_width"], f64::from(0.12_f32));
        assert_eq!(live["scan_velocity_mix"], f64::from(0.8_f32));
        assert_eq!(live["scan_exact_bypass"], false);
        assert_eq!(live["scan_vertex_count"], 240 * 96 * 2);

        let neutral = creative_node_params(RuntimeVisualNodeKind::ScanProcessor(
            ScanProcessorParams::default(),
        ));
        assert_eq!(neutral["scan_exact_bypass"], true);
        assert_eq!(neutral["scan_vertex_count"], 320 * 256 * 2);
    }

    /// Closes `is_priority` and `coalesce_key` for the Symmetry route action.
    /// A missing `is_priority` arm silently loses the admission reservation and
    /// the barrier semantics under queue pressure; an accidental `coalesce_key`
    /// arm would let a later absolute value jump ahead of a reroute.
    #[test]
    fn symmetry_route_action_is_an_uncoalesced_ordered_slot_addressed_barrier() {
        let image = SymmetryRouteSnapshot::Image {
            index: 1,
            route: CreativeImageTapSnapshot {
                input: CreativeImageSourceSnapshot::CleanProgram,
                timing: EdgeTiming::PreviousFrame,
            },
        };
        let action = WebAction::SetVisualNodeSymmetryRoute {
            scope: CreativeScopeSnapshot::Master,
            node_id: "7".into(),
            route: image.clone(),
            composition_revision: 12,
        };
        assert!(action.is_priority());
        assert!(action.coalesce_key().is_none());

        // The wire name is explicit, the slot travels inside the payload, and
        // the whole message round trips exactly.
        let value = serde_json::to_value(&action).unwrap();
        assert_eq!(value["action"], "set_visual_node_symmetry_route");
        assert_eq!(value["composition_revision"], 12);
        assert_eq!(value["route"]["slot"], "image");
        assert_eq!(value["route"]["index"], 1);
        let WebAction::SetVisualNodeSymmetryRoute {
            scope: decoded_scope,
            node_id: decoded_node,
            route: decoded_route,
            composition_revision: decoded_revision,
        } = serde_json::from_value::<WebAction>(value).unwrap()
        else {
            panic!("the symmetry route action must decode to its own variant");
        };
        assert_eq!(decoded_scope, CreativeScopeSnapshot::Master);
        assert_eq!(decoded_node, "7");
        assert_eq!(decoded_route, image);
        assert_eq!(decoded_revision, 12);

        // A motion slot clears with an absent layer and names one otherwise.
        let cleared = serde_json::from_str::<WebAction>(
            r#"{"action":"set_visual_node_symmetry_route","scope":{"scope":"master"},"node_id":"7","route":{"slot":"motion","index":0},"composition_revision":3}"#,
        )
        .unwrap();
        let WebAction::SetVisualNodeSymmetryRoute { route, .. } = cleared else {
            panic!("motion slot")
        };
        assert_eq!(
            route,
            SymmetryRouteSnapshot::Motion {
                index: 0,
                layer_id: None
            }
        );
        assert_eq!(route.index(), 0);

        // The revision is mandatory, so a reroute can never arrive unbarriered.
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_visual_node_symmetry_route","scope":{"scope":"master"},"node_id":"7","route":{"slot":"motion","index":0}}"#
        )
        .is_err());
        // The slot vocabulary is closed: an unknown class is a deserialization
        // error, not a defaulted image slot.
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_visual_node_symmetry_route","scope":{"scope":"master"},"node_id":"7","route":{"slot":"aux","index":0},"composition_revision":3}"#
        )
        .is_err());
        // So is an unknown donor token inside an image slot.
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_visual_node_symmetry_route","scope":{"scope":"master"},"node_id":"7","route":{"slot":"image","index":0,"route":{"input":{"source":"teleport"},"timing":"current_frame"}},"composition_revision":3}"#
        )
        .is_err());
    }

    /// The four hand-maintained browser registries have no cross-check of their
    /// own, so this asserts the Symmetry Field reaches all of them: the kind
    /// picker option, `CREATIVE_NODE_INFO`, `CREATIVE_NODE_PARAMS`, and the
    /// slot-aware route editors. It also asserts the ordered actions stay out
    /// of the quantized allowlist.
    #[test]
    fn symmetry_browser_surface_is_registered_slot_aware_and_never_quantized() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");

        assert!(
            html.contains(r#"<option value="symmetry">Symmetry Field</option>"#),
            "the kind picker is a hand-maintained registry and must list the node"
        );
        assert!(js.contains("symmetry: { label: 'Symmetry Field' }"));

        // Every browser-editable descriptor appears in CREATIVE_NODE_PARAMS,
        // and no route key does — routes own the ordered action instead.
        let params = js
            .split_once("  symmetry: [")
            .expect("CREATIVE_NODE_PARAMS declares a symmetry block")
            .1
            .split_once("\n  ],")
            .expect("the symmetry block terminates")
            .0;
        for descriptor in crate::visual_rack::NODE_PARAM_DESCRIPTORS
            .iter()
            .filter(|descriptor| descriptor.kind == crate::visual_rack::NodeKindTag::Symmetry)
        {
            let is_route = matches!(
                descriptor.value_type,
                crate::visual_rack::NodeParamType::ImageTap
                    | crate::visual_rack::NodeParamType::MotionDonor
            );
            assert_eq!(
                params.contains(&format!("'{}'", descriptor.key)),
                !is_route,
                "browser registry disagrees with the descriptor registry for {}",
                descriptor.key
            );
        }

        // Both discrete vocabularies are published in full.
        for token in [
            "'cyclic'",
            "'dihedral'",
            "'planar_p1'",
            "'planar_pm'",
            "'planar_p2'",
            "'planar_pmm'",
            "'log_spiral'",
            "'orbit'",
            "'cellular_reentry'",
        ] {
            assert!(
                params.contains(token),
                "missing symmetry enum token {token}"
            );
        }

        // The route editors are slot addressed, and all four slots are folded
        // into the structural fingerprint so a reroute rebuilds the card.
        for marker in [
            "'image:0'",
            "'image:1'",
            "'motion:0'",
            "'motion:1'",
            "data-route-slot",
            "card.querySelectorAll('.creative-route-editor')",
            "slot: 'image', index: slotIndex(editor)",
            "slot: 'motion', index: slotIndex(editor)",
            "symmetryRoutes:",
        ] {
            assert!(
                js.contains(marker),
                "missing slot-aware route marker {marker}"
            );
        }

        // Ordered topology actions are never beat latched.
        let quantizable = js
            .split_once("const QUANTIZABLE_ACTIONS")
            .unwrap()
            .1
            .split_once("]);")
            .unwrap()
            .0;
        assert!(!quantizable.contains("set_visual_node_symmetry_route"));
        assert!(!quantizable.contains("set_visual_node_displace_route"));
    }

    #[test]
    fn residual_route_action_is_an_uncoalesced_ordered_barrier_that_names_its_slot() {
        let route = CreativeImageTapSnapshot {
            input: CreativeImageSourceSnapshot::OneBelow,
            timing: EdgeTiming::CurrentFrame,
        };

        // The slot token is the whole reason this action exists once instead of
        // twice, so it must resolve to the authored index and nothing else.
        assert_eq!(
            ResidualRouteSlotSnapshot::Structure.slot(),
            crate::visual_rack::RESIDUAL_STRUCTURE_SLOT
        );
        assert_eq!(
            ResidualRouteSlotSnapshot::Detail.slot(),
            crate::visual_rack::RESIDUAL_DETAIL_SLOT
        );
        assert_ne!(
            ResidualRouteSlotSnapshot::Structure.slot(),
            ResidualRouteSlotSnapshot::Detail.slot()
        );

        for (slot, wire) in [
            (ResidualRouteSlotSnapshot::Structure, "structure"),
            (ResidualRouteSlotSnapshot::Detail, "detail"),
        ] {
            let action = WebAction::SetVisualNodeResidualRoute {
                scope: CreativeScopeSnapshot::Master,
                node_id: "7".into(),
                slot,
                route: route.clone(),
                composition_revision: 12,
            };

            // Both slots are ordered, revision-protected barriers and neither
            // may coalesce behind a later absolute value.
            assert!(action.is_priority());
            assert!(action.coalesce_key().is_none());

            let value = serde_json::to_value(&action).unwrap();
            assert_eq!(value["action"], "set_visual_node_residual_route");
            assert_eq!(value["slot"], wire);
            assert_eq!(value["composition_revision"], 12);
            assert_eq!(value["input"], serde_json::Value::Null);
            let WebAction::SetVisualNodeResidualRoute {
                scope: decoded_scope,
                node_id: decoded_node,
                slot: decoded_slot,
                route: decoded_route,
                composition_revision: decoded_revision,
            } = serde_json::from_value::<WebAction>(value).unwrap()
            else {
                panic!("the residual route action must decode to its own variant");
            };
            assert_eq!(decoded_scope, CreativeScopeSnapshot::Master);
            assert_eq!(decoded_node, "7");
            assert_eq!(decoded_slot, slot);
            assert_eq!(decoded_route, route);
            assert_eq!(decoded_revision, 12);
        }

        // The well-formed message is accepted for either slot...
        for slot in ["structure", "detail"] {
            assert!(
                serde_json::from_str::<WebAction>(&format!(
                    r#"{{"action":"set_visual_node_residual_route","scope":{{"scope":"master"}},"node_id":"7","slot":"{slot}","route":{{"input":{{"source":"one_below"}},"timing":"current_frame"}},"composition_revision":1}}"#
                ))
                .is_ok(),
                "slot {slot} must be accepted"
            );
        }
        // ...the revision is mandatory, so a route edit can never arrive
        // unbarriered...
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_visual_node_residual_route","scope":{"scope":"master"},"node_id":"7","slot":"structure","route":{"input":{"source":"one_below"},"timing":"current_frame"}}"#
        )
        .is_err());
        // ...the slot is mandatory, so a reroute can never default onto one of
        // the two inputs...
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_visual_node_residual_route","scope":{"scope":"master"},"node_id":"7","route":{"input":{"source":"one_below"},"timing":"current_frame"},"composition_revision":1}"#
        )
        .is_err());
        // ...an unknown slot token is a closed-vocabulary rejection rather than
        // a positional fallback onto the partner route...
        for hostile in [r#""dc""#, r#""2""#, "1", "null"] {
            assert!(
                serde_json::from_str::<WebAction>(&format!(
                    r#"{{"action":"set_visual_node_residual_route","scope":{{"scope":"master"}},"node_id":"7","slot":{hostile},"route":{{"input":{{"source":"one_below"}},"timing":"current_frame"}},"composition_revision":1}}"#
                ))
                .is_err(),
                "hostile slot {hostile} must be refused"
            );
        }
        // ...and an unknown donor token is rejected, not defaulted.
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_visual_node_residual_route","scope":{"scope":"master"},"node_id":"7","slot":"detail","route":{"input":{"source":"teleport"},"timing":"current_frame"},"composition_revision":1}"#
        )
        .is_err());
    }

    #[test]
    fn residual_snapshot_publishes_both_routes_values_and_per_slot_diagnostics() {
        use crate::visual_rack::{
            ResidualBlock, ResidualQuantization, ResolvedImageSource, ResolvedImageTap,
            RuntimeResidualParams, RuntimeVisualNodeKind,
        };

        let live = creative_node_params(RuntimeVisualNodeKind::Residual(RuntimeResidualParams {
            structure: ResolvedImageTap {
                source: ResolvedImageSource::AllBelow,
                timing: EdgeTiming::CurrentFrame,
            },
            detail: ResolvedImageTap {
                source: ResolvedImageSource::CleanProgram,
                timing: EdgeTiming::PreviousFrame,
            },
            block: ResidualBlock::ThirtyTwo,
            quantization: ResidualQuantization::Medium,
            mix: 0.75,
            detail_gain: 2.5,
            seed: 0x00c0_ffee,
            ..RuntimeResidualParams::default()
        }));
        assert_eq!(live["structure_tap"]["input"]["source"], "all_below");
        assert_eq!(live["structure_tap"]["timing"], "current_frame");
        assert_eq!(live["detail_tap"]["input"]["source"], "clean_program");
        assert_eq!(live["detail_tap"]["timing"], "previous_frame");
        assert_eq!(live["block"], "thirty_two");
        assert_eq!(live["quantization"], "medium");
        assert_eq!(live["mix"], 0.75);
        assert_eq!(live["detail_gain"], 2.5);
        assert_eq!(live["seed"], 0x00c0_ffee_u32);
        assert_eq!(
            live["diagnostic"], "",
            "a fully live route pair reports no diagnostic text"
        );
        // The published key set is exactly the frozen snapshot contract.
        let mut keys: Vec<_> = live.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "block",
                "detail_gain",
                "detail_tap",
                "diagnostic",
                "mix",
                "quantization",
                "seed",
                "structure_tap",
            ]
        );

        // Each dead slot names itself, so a tombstone can never be read as
        // belonging to the other route, and neither slot rebinds.
        let structure_gone =
            creative_node_params(RuntimeVisualNodeKind::Residual(RuntimeResidualParams {
                structure: ResolvedImageTap {
                    source: ResolvedImageSource::MissingSelectedLayer {
                        saved_position: crate::performance::SavedLayerPosition::new(4).unwrap(),
                        stage: crate::image_routing::LayerImageStage::PostLocalEffects,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                ..RuntimeResidualParams::default()
            }));
        assert_eq!(
            structure_gone["diagnostic"],
            "structure: missing saved layer 4"
        );
        assert_eq!(
            structure_gone["structure_tap"]["input"]["source"],
            "missing_selected_layer"
        );
        assert_eq!(structure_gone["detail_tap"]["input"]["source"], "one_below");

        let detail_gone =
            creative_node_params(RuntimeVisualNodeKind::Residual(RuntimeResidualParams {
                detail: ResolvedImageTap {
                    source: ResolvedImageSource::MissingGroupOutput(
                        crate::visual_rack::GroupId::new(9).unwrap(),
                    ),
                    timing: EdgeTiming::PreviousFrame,
                },
                ..RuntimeResidualParams::default()
            }));
        assert_eq!(detail_gone["diagnostic"], "detail: missing group output 9");
        assert_eq!(
            detail_gone["structure_tap"]["input"]["source"], "one_below",
            "a dead detail slot must never tombstone its partner"
        );

        let both_gone =
            creative_node_params(RuntimeVisualNodeKind::Residual(RuntimeResidualParams {
                structure: ResolvedImageTap {
                    source: ResolvedImageSource::MissingSelectedLayer {
                        saved_position: crate::performance::SavedLayerPosition::new(4).unwrap(),
                        stage: crate::image_routing::LayerImageStage::PostLocalEffects,
                    },
                    timing: EdgeTiming::CurrentFrame,
                },
                detail: ResolvedImageTap {
                    source: ResolvedImageSource::MissingGroupOutput(
                        crate::visual_rack::GroupId::new(9).unwrap(),
                    ),
                    timing: EdgeTiming::PreviousFrame,
                },
                ..RuntimeResidualParams::default()
            }));
        assert_eq!(
            both_gone["diagnostic"],
            "structure: missing saved layer 4; detail: missing group output 9"
        );
    }

    #[test]
    fn every_browser_node_registry_matches_the_rust_descriptor_tables() {
        let js = include_str!("../../static/app.js");
        let html = include_str!("../../static/index.html");
        let node_info = js
            .split_once("const CREATIVE_NODE_INFO = Object.freeze({")
            .expect("app.js declares CREATIVE_NODE_INFO")
            .1
            .split_once("});")
            .expect("CREATIVE_NODE_INFO is closed")
            .0;
        let node_params = js
            .split_once("const CREATIVE_NODE_PARAMS = Object.freeze({")
            .expect("app.js declares CREATIVE_NODE_PARAMS")
            .1
            .split_once("\n});")
            .expect("CREATIVE_NODE_PARAMS is closed")
            .0;

        // The panel's kind picker, its label registry and its parameter
        // registry are hand-maintained beside the Rust table. Cross-check all
        // three so a kind can never be fully functional yet uninsertable.
        for descriptor in crate::visual_rack::NODE_KIND_DESCRIPTORS {
            let key = descriptor.key;
            let marker = matches!(
                descriptor.tag,
                crate::visual_rack::NodeKindTag::LegacyCanonical
                    | crate::visual_rack::NodeKindTag::LegacyTemporal
            );
            assert!(
                node_info.contains(&format!("\n  {key}: {{")),
                "CREATIVE_NODE_INFO is missing node kind {key}"
            );
            assert_eq!(
                html.contains(&format!("<option value=\"{key}\">")),
                !marker,
                "index.html kind picker disagrees about node kind {key}"
            );
            assert_eq!(
                node_params.contains(&format!("\n  {key}: [")),
                !marker,
                "CREATIVE_NODE_PARAMS disagrees about node kind {key}"
            );
        }

        // Every browser-editable Residual field is declared, and neither route
        // is: both belong to the ordered slot-naming action.
        let residual = node_params
            .split_once("\n  residual: [")
            .expect("CREATIVE_NODE_PARAMS declares residual")
            .1
            .split_once("\n  ]")
            .expect("the residual block is closed")
            .0;
        for descriptor in crate::visual_rack::NODE_PARAM_DESCRIPTORS
            .iter()
            .filter(|descriptor| descriptor.kind == crate::visual_rack::NodeKindTag::Residual)
        {
            let declared = residual.contains(&format!("'{}'", descriptor.key));
            let routed = descriptor.value_type == crate::visual_rack::NodeParamType::ImageTap;
            assert_eq!(
                declared, !routed,
                "residual param {} is on the wrong browser path",
                descriptor.key
            );
        }
        assert!(residual.contains("floatDef('mix', 'Mix', 0, 1,"));
        assert!(residual.contains("floatDef('detail_gain', 'Detail gain', 0, 4,"));
        assert!(residual.contains("uintDef('seed', 'Seed')"));

        // Both discrete vocabularies are published exactly as the engine spells
        // them, or the server's Enum allowlist drops the select at ingress.
        for token in ["four", "eight", "sixteen", "thirty_two", "sixty_four"] {
            assert!(
                residual.contains(&format!("['{token}',")),
                "residual block vocabulary is missing {token}"
            );
        }
        for token in ["off", "coarse", "medium", "fine"] {
            assert!(
                residual.contains(&format!("['{token}',")),
                "residual quantization vocabulary is missing {token}"
            );
        }

        // Two routes need two independently wired, slot-tagged editors and two
        // structural fingerprint slots.
        assert!(js.contains("data-route-slot=\"${escapeHtml(slot)}\""));
        assert!(js.contains("card.querySelectorAll('.creative-route-editor')"));
        assert!(js.contains("routeEditor.dataset.routeSlot"));
        assert!(js.contains("'structure', 'Structure donor'"));
        assert!(js.contains("'detail', 'Detail donor'"));
        assert!(js.contains("detailRoute: node.kind === 'residual'"));

        // A topology barrier is never beat-latched by the panel.
        let quantizable = js
            .split_once("const QUANTIZABLE_ACTIONS")
            .unwrap()
            .1
            .split_once("]);")
            .unwrap()
            .0;
        assert!(!quantizable.contains("set_visual_node_residual_route"));
        assert!(!quantizable.contains("set_visual_node_displace_route"));
        assert!(!quantizable.contains("set_visual_node_route"));
    }
    #[test]
    fn gesture_samples_are_uncoalesced_stream_events_whose_stroke_edges_hold_priority() {
        use crate::gesture::{GestureMode, GesturePhase};

        let begin = WebAction::GestureSample {
            stroke: 0,
            phase: GesturePhase::Begin,
            mode: GestureMode::Push,
            x: 0.25,
            y: 0.5,
            pressure: 1.0,
            direction_x: 0.0,
            direction_y: 0.0,
        };
        let motion = WebAction::GestureSample {
            stroke: 0,
            phase: GesturePhase::Move,
            mode: GestureMode::Push,
            x: 0.5,
            y: 0.5,
            pressure: 1.0,
            direction_x: 1.0,
            direction_y: 0.0,
        };
        let end = WebAction::GestureSample {
            stroke: 0,
            phase: GesturePhase::End,
            mode: GestureMode::Push,
            x: 0.75,
            y: 0.5,
            pressure: 1.0,
            direction_x: 1.0,
            direction_y: 0.0,
        };

        // A sample is a stream event, never an absolute value: coalescing one
        // would silently delete a path point the operator actually drew.
        for action in [&begin, &motion, &end] {
            assert!(
                action.coalesce_key().is_none(),
                "a gesture sample must never coalesce"
            );
            assert!(action.is_performance_only_for_history());
        }
        // A dropped Begin orphans the stroke and a dropped End leaves it open,
        // so both edges hold an admission reservation while ordinary motion
        // may shed under saturation.
        assert!(begin.is_priority());
        assert!(end.is_priority());
        assert!(!motion.is_priority());

        // Because a sample carries no coalesce key it is also an ordering
        // barrier, so a later absolute value can never jump ahead of a stroke.
        let mut queue = Vec::new();
        assert_eq!(
            enqueue_bounded(
                &mut queue,
                WebAction::SetParam {
                    param: "brightness".into(),
                    value: serde_json::json!(0.1),
                }
            ),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(&mut queue, motion.clone()),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(
                &mut queue,
                WebAction::SetParam {
                    param: "brightness".into(),
                    value: serde_json::json!(0.9),
                }
            ),
            EnqueueOutcome::Added
        );
        assert_eq!(queue.len(), 3, "the sample separates the two scalars");

        // Wire shape: a well-formed sample decodes, an absent pressure means
        // full contact, an absent direction is the inert zero, and both
        // vocabularies are closed rather than defaulted.
        let value = serde_json::to_value(&begin).unwrap();
        assert_eq!(value["action"], "gesture_sample");
        assert_eq!(value["phase"], "begin");
        assert_eq!(value["mode"], "push");

        let defaulted = serde_json::from_str::<WebAction>(
            r#"{"action":"gesture_sample","stroke":0,"phase":"move","mode":"curl","x":0.5,"y":0.5}"#,
        )
        .unwrap();
        let WebAction::GestureSample {
            mode,
            pressure,
            direction_x,
            direction_y,
            ..
        } = defaulted
        else {
            panic!("a gesture sample must decode to its own variant");
        };
        assert_eq!(mode, GestureMode::Curl);
        assert_eq!(pressure, 1.0);
        assert_eq!(direction_x, 0.0);
        assert_eq!(direction_y, 0.0);

        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"gesture_sample","stroke":0,"phase":"scrub","mode":"push","x":0.5,"y":0.5}"#
        )
        .is_err());
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"gesture_sample","stroke":0,"phase":"move","mode":"smear","x":0.5,"y":0.5}"#
        )
        .is_err());
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"gesture_sample","stroke":0,"phase":"move","mode":"push","y":0.5}"#
        )
        .is_err());
    }

    #[test]
    fn gesture_recording_control_is_an_uncoalesced_priority_barrier() {
        let arm = WebAction::SetGestureRecording {
            enabled: true,
            layer_stack_revision: 9,
        };
        assert!(arm.is_priority());
        assert!(
            arm.coalesce_key().is_none(),
            "a sample must never cross an arm/disarm edge into the wrong recording"
        );
        assert!(arm.is_performance_only_for_history());
        assert_eq!(
            serde_json::to_value(&arm).unwrap(),
            serde_json::json!({
                "action": "set_gesture_recording",
                "enabled": true,
                "layer_stack_revision": 9
            })
        );
        let decoded = serde_json::from_str::<WebAction>(
            r#"{"action":"set_gesture_recording","enabled":false,"layer_stack_revision":9}"#,
        )
        .unwrap();
        assert!(matches!(
            decoded,
            WebAction::SetGestureRecording {
                enabled: false,
                layer_stack_revision: 9,
            }
        ));
        // Both fields are mandatory: the revision is the barrier that keeps an
        // arm decision attached to the program it was taken against.
        for hostile in [
            r#"{"action":"set_gesture_recording"}"#,
            r#"{"action":"set_gesture_recording","enabled":true}"#,
            r#"{"action":"set_gesture_recording","layer_stack_revision":9}"#,
        ] {
            assert!(
                serde_json::from_str::<WebAction>(hostile).is_err(),
                "accepted an incomplete recording barrier: {hostile}"
            );
        }
    }

    #[test]
    fn performance_transport_controls_are_uncoalesced_priority_barriers() {
        // The B9 transports ride the set_gesture_recording law exactly: never
        // coalesced (an absolute value must not jump an arm/disarm edge),
        // priority admission, and no manual-history transaction.
        let record = WebAction::SetPerformanceRecording {
            enabled: true,
            layer_stack_revision: 9,
        };
        let playback = WebAction::SetPerformancePlayback {
            enabled: true,
            loop_playback: true,
            layer_stack_revision: 9,
        };
        let clear = WebAction::ClearPerformanceTake;
        for action in [&record, &playback, &clear] {
            assert!(action.is_priority());
            assert!(action.coalesce_key().is_none());
            assert!(action.is_performance_only_for_history());
        }
        assert_eq!(
            serde_json::to_value(&record).unwrap(),
            serde_json::json!({
                "action": "set_performance_recording",
                "enabled": true,
                "layer_stack_revision": 9
            })
        );
        let decoded = serde_json::from_str::<WebAction>(
            r#"{"action":"set_performance_playback","enabled":true,"layer_stack_revision":4}"#,
        )
        .unwrap();
        assert!(
            matches!(
                decoded,
                WebAction::SetPerformancePlayback {
                    enabled: true,
                    loop_playback: false,
                    layer_stack_revision: 4,
                }
            ),
            "the loop flag defaults off for an older client"
        );
        for hostile in [
            r#"{"action":"set_performance_recording"}"#,
            r#"{"action":"set_performance_recording","enabled":true}"#,
            r#"{"action":"set_performance_playback","loop_playback":true}"#,
        ] {
            assert!(
                serde_json::from_str::<WebAction>(hostile).is_err(),
                "accepted an incomplete transport barrier: {hostile}"
            );
        }
    }

    #[test]
    fn b10_source_actions_classify_on_their_own_laws() {
        // A bend edge is a stream event: never coalesced (a pending press
        // must not be replaced by its release), both edges priority, and
        // performance-only for history.
        let press = WebAction::BendPad {
            index: 2,
            held: true,
        };
        let release = WebAction::BendPad {
            index: 2,
            held: false,
        };
        for edge in [&press, &release] {
            assert!(edge.is_priority());
            assert!(edge.coalesce_key().is_none());
            assert!(edge.is_performance_only_for_history());
        }
        assert_eq!(
            serde_json::to_value(&press).unwrap(),
            serde_json::json!({ "action": "bend_pad", "index": 2, "held": true })
        );

        // Envelope, macro, and seed edits are ordinary coalescible values.
        let envelope = WebAction::SetEnvelope {
            index: 1,
            param: "attack".to_string(),
            value: serde_json::json!(0.1),
        };
        assert_eq!(envelope.coalesce_key().as_deref(), Some("env:1:attack"));
        assert!(!envelope.is_performance_only_for_history());
        let knob = WebAction::SetMacro {
            index: 3,
            value: 0.5,
        };
        assert_eq!(knob.coalesce_key().as_deref(), Some("macro:3"));
        let seed = WebAction::SetModSeed { seed: 9 };
        assert_eq!(seed.coalesce_key().as_deref(), Some("mod:seed"));
    }

    #[test]
    fn the_perform_panel_wires_b10_sources_and_never_latches_a_bend() {
        let html = include_str!("../../static/index.html");
        for contract in [
            "id=\"perform-group\"",
            "id=\"bend-pads\"",
            "data-bend=\"5\"",
            "id=\"envelope-list\"",
            "id=\"macro-list\"",
            "id=\"mod-seed\"",
        ] {
            assert!(html.contains(contract), "missing perform HTML: {contract}");
        }
        let js = include_str!("../../static/app.js");
        for contract in [
            "action: 'set_envelope', index: i, param: 'trigger'",
            "action: 'set_macro', index: i, value",
            "action: 'set_mod_seed', seed",
            "action: 'bend_pad', index, held",
            "releaseAllBends",
            "['bend1', 'Bend 1']",
            "['env1', 'Env 1']",
            "['macro1', 'Macro 1']",
            "['chaos', 'Chaos']",
            "['video_cut', 'Vid Cut']",
            "chaos|drift",
        ] {
            assert!(js.contains(contract), "missing perform JS: {contract}");
        }
        let start = js.find("const QUANTIZABLE_ACTIONS").expect("allowlist");
        let tail = &js[start..];
        let end = tail.find("]);").expect("allowlist end") + 3;
        let allowlist = &tail[..end];
        assert!(
            !allowlist.contains("bend_pad"),
            "a bend edge must never be latchable"
        );
    }

    #[test]
    fn the_monitor_bay_protocol_is_additive_gated_and_watch_reaches_the_registry() {
        // An older snapshot without the block decodes as the inactive bay.
        let legacy = serde_json::to_value(AppSnapshot::default())
            .map(|mut value| {
                value.as_object_mut().unwrap().remove("monitor_bay");
                value
            })
            .unwrap();
        let decoded: AppSnapshot = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            decoded.monitor_bay,
            crate::monitor_bay::MonitorBaySnapshot::default()
        );
        assert!(!decoded.monitor_bay.active);

        // The three actions parse under their wire names; the two authored
        // stage tools coalesce per control and every one is a host operation
        // outside manual history.
        let bay: WebAction =
            serde_json::from_str(r#"{"action":"set_monitor_bay","enabled":true}"#).unwrap();
        assert_eq!(bay.coalesce_key().as_deref(), Some("stage:monitor-bay"));
        assert!(bay.is_performance_only_for_history());
        let probe: WebAction =
            serde_json::from_str(r#"{"action":"set_monitor_probe","probe":"program_tap"}"#)
                .unwrap();
        assert_eq!(probe.coalesce_key().as_deref(), Some("stage:monitor-probe"));
        assert!(probe.is_performance_only_for_history());
        let watch: WebAction =
            serde_json::from_str(r#"{"action":"monitor_watch","enabled":true}"#).unwrap();
        assert!(watch.coalesce_key().is_none());
        assert!(watch.is_performance_only_for_history());

        // The watch registry arms on a fresh declaration, disarms on an
        // explicit stop, and forgets a disconnected client immediately —
        // the gyro registry's lifecycle, without its implicit enrollment.
        let state = WebState::new().expect("test web state");
        assert!(!state.monitor_watch_active());
        state.set_monitor_watch(7, true);
        assert!(state.monitor_watch_active());
        state.set_monitor_watch(7, false);
        assert!(!state.monitor_watch_active());
        state.set_monitor_watch(9, true);
        state.disconnect_monitor_client(9);
        assert!(!state.monitor_watch_active());

        // The bundled panel actually sends the vocabulary and consumes the
        // block, so the engine-side gates guard live strings, not intent.
        let js = include_str!("../../static/app.js");
        assert!(js.contains("sendAction({ action: 'monitor_watch', enabled: watching })"));
        assert!(js.contains(
            "sendAction({ action: 'set_monitor_bay', enabled: monitorBayNative.checked })"
        ));
        assert!(js.contains("sendAction({ action: 'set_monitor_probe', probe })"));
        assert!(js.contains("syncMonitorBay(msg.monitor_bay)"));
        let html = include_str!("../../static/index.html");
        assert!(html.contains("id=\"monitor-bay-group\""));
        assert!(html.contains("id=\"monitor-bay-waveform\""));
        assert!(html.contains("id=\"monitor-bay-scope\""));
    }

    #[test]
    fn the_performance_snapshot_is_additive_and_publishes_honest_counters() {
        // An older snapshot without the block decodes as a disarmed recorder.
        let legacy = serde_json::to_value(AppSnapshot::default())
            .map(|mut value| {
                value
                    .as_object_mut()
                    .unwrap()
                    .remove("performance_recorder");
                value
            })
            .unwrap();
        let decoded: AppSnapshot = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            decoded.performance_recorder,
            PerformanceStatusSnapshot::default()
        );

        // Empty checksum and degraded lists are omitted from the wire, so an
        // empty take can never read as a verified one.
        let idle = serde_json::to_value(PerformanceStatusSnapshot::default()).unwrap();
        assert!(idle.get("checksum").is_none());
        assert!(idle.get("degraded").is_none());
        let busy = serde_json::to_value(PerformanceStatusSnapshot {
            mode: "playing".to_string(),
            checksum: "abc".to_string(),
            degraded: vec!["layer_effect:3:pixelate".to_string()],
            unsupported_edits: 2,
            rejected_edits: 1,
            ..PerformanceStatusSnapshot::default()
        })
        .unwrap();
        assert_eq!(busy["checksum"], "abc");
        assert_eq!(busy["degraded"][0], "layer_effect:3:pixelate");
        assert_eq!(busy["unsupported_edits"], 2);
    }

    #[test]
    fn the_panel_wires_the_performance_recorder_and_never_latches_its_transports() {
        let html = include_str!("../../static/index.html");
        for contract in [
            "id=\"performance-group\"",
            "id=\"performance-record-toggle\"",
            "id=\"performance-play-toggle\"",
            "id=\"performance-loop-toggle\"",
            "id=\"performance-clear\"",
            "id=\"performance-state\"",
            "id=\"performance-telemetry\"",
            "id=\"performance-checksum\"",
            "id=\"performance-degraded\"",
        ] {
            assert!(html.contains(contract), "missing recorder HTML: {contract}");
        }
        // The fast counters must not sit inside a live announcement region.
        let telemetry = html
            .find("id=\"performance-telemetry\"")
            .expect("performance telemetry element");
        let telemetry_tag = &html[telemetry..telemetry + html[telemetry..].find('>').unwrap()];
        assert!(!telemetry_tag.contains("role=\"status\""));
        assert!(!telemetry_tag.contains("aria-live=\"polite\""));

        let js = include_str!("../../static/app.js");
        for contract in [
            "syncPerformanceRecorder(msg.performance_recorder)",
            "action: 'set_performance_recording',",
            "action: 'set_performance_playback',",
            "action: 'clear_performance_take'",
            "loop_playback: performanceLoop,",
            "'No recorded take'",
            "recorded edit(s)",
        ] {
            assert!(js.contains(contract), "missing recorder JS: {contract}");
        }

        // No transport may ever be wrapped in a quantized batch.
        let start = js.find("const QUANTIZABLE_ACTIONS").expect("allowlist");
        let tail = &js[start..];
        let end = tail.find("]);").expect("allowlist end") + 3;
        let allowlist = &tail[..end];
        for forbidden in [
            "set_performance_recording",
            "set_performance_playback",
            "clear_performance_take",
        ] {
            assert!(
                !allowlist.contains(forbidden),
                "{forbidden} must never be latchable"
            );
        }
    }

    #[test]
    fn gesture_snapshot_reports_live_only_samples_separately_from_the_recorded_track() {
        // The honesty law on the wire: an unrecorded gesture is counted in its
        // own field and never merged into the recorded total, and an empty
        // track publishes no checksum because it has nothing to verify.
        let disarmed = GestureStatusSnapshot {
            recording: false,
            recorded_events: 0,
            truncated: false,
            open_strokes: 0,
            live_only_events: 512,
            checksum: String::new(),
            status: String::new(),
            canvas: GestureCanvasStatusSnapshot::default(),
        };
        let value = serde_json::to_value(&disarmed).unwrap();
        assert_eq!(value["recording"], false);
        assert_eq!(value["recorded_events"], 0);
        assert_eq!(value["live_only_events"], 512);
        assert!(
            value.get("checksum").is_none(),
            "an empty track must not offer a digest that implies it is replayable"
        );

        let recorded = GestureStatusSnapshot {
            recording: true,
            recorded_events: 3,
            truncated: true,
            open_strokes: 1,
            live_only_events: 4,
            checksum: "abc".into(),
            status: "held".into(),
            canvas: GestureCanvasStatusSnapshot::default(),
        };
        let value = serde_json::to_value(&recorded).unwrap();
        assert_eq!(value["truncated"], true);
        assert_eq!(value["open_strokes"], 1);
        assert_eq!(value["checksum"], "abc");
        assert_eq!(value["status"], "held");

        // The section is additive: an older snapshot with no gesture block
        // deserializes as a disarmed, empty, unrecorded one.
        let legacy = serde_json::to_value(AppSnapshot::default()).unwrap();
        let mut trimmed = legacy.clone();
        trimmed.as_object_mut().unwrap().remove("gesture");
        let restored = serde_json::from_value::<AppSnapshot>(trimmed).unwrap();
        assert_eq!(restored.gesture, GestureStatusSnapshot::default());
        assert!(!restored.gesture.recording);
        assert_eq!(restored.gesture.recorded_events, 0);
        assert_eq!(restored.gesture.live_only_events, 0);
    }

    /// The canvas scalar is an ordinary absolute value that coalesces per
    /// control - and it can never jump the recording barrier, because the
    /// barrier has no coalesce key at all.
    #[test]
    fn gesture_canvas_scalars_coalesce_per_control_and_never_cross_the_recording_barrier() {
        let radius = WebAction::SetGestureCanvas {
            param: "radius".into(),
            value: serde_json::json!(0.2),
        };
        let strength = WebAction::SetGestureCanvas {
            param: "strength".into(),
            value: serde_json::json!(0.4),
        };
        assert_eq!(
            radius.coalesce_key().as_deref(),
            Some("gesture:canvas:radius")
        );
        assert_eq!(
            strength.coalesce_key().as_deref(),
            Some("gesture:canvas:strength")
        );
        assert!(!radius.is_priority());
        assert!(
            !radius.is_performance_only_for_history(),
            "a canvas value is an authored edit and belongs in manual history"
        );
        assert_eq!(
            serde_json::to_value(&radius).unwrap(),
            serde_json::json!({"action": "set_gesture_canvas", "param": "radius", "value": 0.2})
        );
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_gesture_canvas","param":"retention","value":0.5}"#
        )
        .is_ok());
        assert!(serde_json::from_str::<WebAction>(
            r#"{"action":"set_gesture_canvas","param":"retention"}"#
        )
        .is_err());

        // Two edits to one control coalesce; the arm/disarm barrier between
        // two edits stops the later one from replacing the earlier.
        let mut queue = Vec::new();
        assert_eq!(
            enqueue_bounded(&mut queue, radius.clone()),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(
                &mut queue,
                WebAction::SetGestureCanvas {
                    param: "radius".into(),
                    value: serde_json::json!(0.3),
                }
            ),
            EnqueueOutcome::Coalesced
        );
        assert_eq!(queue.len(), 1);
        assert_eq!(
            enqueue_bounded(
                &mut queue,
                WebAction::SetGestureRecording {
                    enabled: true,
                    layer_stack_revision: 4,
                }
            ),
            EnqueueOutcome::Added
        );
        assert_eq!(
            enqueue_bounded(
                &mut queue,
                WebAction::SetGestureCanvas {
                    param: "radius".into(),
                    value: serde_json::json!(0.9),
                }
            ),
            EnqueueOutcome::Added,
            "an absolute value must never jump an arm/disarm edge"
        );
        assert_eq!(queue.len(), 3);
        assert!(matches!(queue[1], WebAction::SetGestureRecording { .. }));
    }

    /// The panel must state which of the two counts is replayable. The armed
    /// state is the only thing announced; the per-frame counters live outside
    /// the live region so a stream of samples cannot flood a screen reader.
    #[test]
    fn gesture_browser_surface_is_accessible_truthful_and_stays_out_of_a_quantized_batch() {
        let html = include_str!("../../static/index.html");
        let js = include_str!("../../static/app.js");
        for contract in [
            "id=\"gesture-group\"",
            "role=\"region\" aria-label=\"Gesture field etching\"",
            "data-gesture-canvas=\"radius\" data-min=\"0\" data-max=\"1\" data-step=\"0.001\"",
            "data-gesture-canvas=\"strength\"",
            "data-gesture-canvas=\"retention\"",
            "id=\"gesture-record-toggle\" aria-pressed=\"false\"",
            "id=\"gesture-recording-state\" role=\"status\" aria-live=\"polite\"",
            "id=\"gesture-telemetry\" aria-live=\"off\"",
            "id=\"gesture-checksum\" aria-live=\"off\"",
            "etches this session only and is never added to the replayable track",
        ] {
            assert!(html.contains(contract), "missing gesture HTML: {contract}");
        }
        // The fast counters must not sit inside a live announcement region.
        let telemetry = html
            .find("id=\"gesture-telemetry\"")
            .expect("gesture telemetry element");
        let telemetry_tag = &html[telemetry..telemetry + html[telemetry..].find('>').unwrap()];
        assert!(!telemetry_tag.contains("role=\"status\""));
        assert!(!telemetry_tag.contains("aria-live=\"polite\""));

        for contract in [
            "syncGesture(msg.gesture)",
            "action: 'set_gesture_canvas', param, value",
            "action: 'set_gesture_recording',",
            "layer_stack_revision: layerStackRevision,",
            "recorded event(s)",
            "live-only sample(s)",
            "'No recorded track'",
            "['gesture_radius', 'Gesture Radius']",
            "['gesture_strength', 'Gesture Strength']",
            "['gesture_retention', 'Gesture Retention']",
        ] {
            assert!(js.contains(contract), "missing gesture JS: {contract}");
        }

        // Neither gesture action may ever be wrapped in a quantized batch.
        let start = js.find("const QUANTIZABLE_ACTIONS").expect("allowlist");
        let tail = &js[start..];
        let end = tail.find("]);").expect("allowlist end") + 3;
        let allowlist = &tail[..end];
        for forbidden in [
            "set_gesture_recording",
            "set_gesture_canvas",
            "gesture_sample",
        ] {
            assert!(
                !allowlist.contains(forbidden),
                "{forbidden} must never be latchable"
            );
        }
    }

    /// The published canvas block is session state and sits beside the
    /// recording truth rather than inside it.
    #[test]
    fn the_gesture_snapshot_separates_authored_canvas_state_from_recording_truth() {
        let mut snapshot = GestureStatusSnapshot {
            recording: false,
            recorded_events: 0,
            truncated: false,
            open_strokes: 2,
            live_only_events: 90,
            checksum: String::new(),
            status: String::new(),
            canvas: GestureCanvasStatusSnapshot {
                radius: 0.2,
                strength: 0.4,
                retention: 0.6,
                grid_width: 320,
                grid_height: 180,
                generation: 77,
            },
        };
        let value = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(value["canvas"]["generation"], 77);
        assert_eq!(value["canvas"]["grid_width"], 320);
        assert_eq!(value["recorded_events"], 0);
        assert_eq!(value["live_only_events"], 90);
        assert!(
            value.get("checksum").is_none(),
            "a busy canvas with nothing recorded must not publish a digest"
        );

        // An older panel that never learned about the canvas keeps working.
        let mut trimmed = value.clone();
        trimmed.as_object_mut().unwrap().remove("canvas");
        let restored = serde_json::from_value::<GestureStatusSnapshot>(trimmed).unwrap();
        assert_eq!(restored.canvas, GestureCanvasStatusSnapshot::default());
        assert_eq!(restored.live_only_events, 90);

        snapshot.recording = true;
        snapshot.recorded_events = 5;
        snapshot.checksum = "d".repeat(64);
        let value = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(value["recording"], true);
        assert_eq!(value["recorded_events"], 5);
        assert_eq!(value["live_only_events"], 90);
        assert_eq!(value["checksum"], snapshot.checksum);
    }

    #[tokio::test]
    async fn enveloped_coalescing_keeps_the_new_ingress_and_classifies_the_old_one() {
        let state = WebState::new().unwrap();
        let set = |value| WebAction::SetParam {
            param: "brightness".to_string(),
            value: serde_json::json!(value),
        };
        assert_eq!(state.enqueue_action(set(0.25)).await, EnqueueOutcome::Added);
        assert_eq!(
            state.enqueue_action(set(0.75)).await,
            EnqueueOutcome::Coalesced
        );

        let queue = state.actions.lock().await;
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].sequence().get(), 2);
        assert!(matches!(
            queue[0].payload(),
            WebAction::SetParam { value, .. } if value.as_f64() == Some(0.75)
        ));
        drop(queue);

        let receipts = state.action_receipts_for_test();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].sequence.get(), 1);
        assert_eq!(receipts[0].disposition, ActionDisposition::Coalesced);
    }

    #[tokio::test]
    async fn every_socket_enqueue_gets_one_monotonic_payload_free_disposition() {
        let state = WebState::new().unwrap();
        let action = |value| WebAction::SetParam {
            param: "brightness".to_string(),
            value: serde_json::json!(value),
        };
        let (first_outcome, first_ack) = state.enqueue_action_with_ack(action(0.25)).await;
        let (second_outcome, second_ack) = state.enqueue_action_with_ack(action(0.75)).await;
        assert_eq!(first_outcome, EnqueueOutcome::Added);
        assert_eq!(first_ack.sequence, 1);
        assert_eq!(first_ack.disposition, ActionIngressDisposition::Queued);
        assert_eq!(second_outcome, EnqueueOutcome::Coalesced);
        assert_eq!(second_ack.sequence, 2);
        assert_eq!(
            second_ack.disposition,
            ActionIngressDisposition::QueuedAfterCoalescing
        );
        let encoded = serde_json::to_string(&second_ack).unwrap();
        assert!(encoded.contains("\"type\":\"action_ack\""));
        assert!(!encoded.contains("brightness"));
        assert!(!encoded.contains("0.75"));
    }

    #[tokio::test]
    async fn cancelling_while_waiting_for_the_action_queue_refuses_exactly_once() {
        let state = WebState::new().unwrap();
        let held_queue = state.actions.lock().await;
        let envelope = state.envelope_web_action(WebAction::SetBlackout { enabled: true });
        let pending = state.enqueue_enveloped_action_with_ack(envelope);
        let task = tokio::spawn(pending);
        tokio::task::yield_now().await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        drop(held_queue);

        let receipts = state.action_receipts_for_test();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].sequence.get(), 1);
        assert_eq!(receipts[0].disposition, ActionDisposition::Refused);
        assert!(state.actions.lock().await.is_empty());
    }

    #[tokio::test]
    async fn cancelling_at_the_gesture_lock_refuses_without_publishing_an_owner() {
        let state = WebState::new().unwrap();
        let held_gesture = state.browser_history_gesture.lock().await;
        let envelope = state.envelope_web_action(WebAction::BeginHistoryGesture { gesture_id: 41 });
        let terminal = state.action_ingress_terminal_guard(envelope.identity());
        let task_state = state.clone();
        let task = tokio::spawn(async move {
            task_state
                .enqueue_browser_history_begin_with_ack(7, 41, envelope, terminal)
                .await
        });
        tokio::task::yield_now().await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        drop(held_gesture);

        assert!(state.browser_history_gesture.lock().await.is_none());
        assert!(state.actions.lock().await.is_empty());
        let receipts = state.action_receipts_for_test();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].sequence.get(), 1);
        assert_eq!(receipts[0].disposition, ActionDisposition::Refused);
    }

    #[tokio::test]
    async fn browser_phone_and_native_adapters_share_one_engine_sequencer() {
        let state = WebState::new().unwrap();
        let shared = state.action_sequencer();
        assert_eq!(
            state
                .enqueue_action(WebAction::SetBlackout { enabled: true })
                .await,
            EnqueueOutcome::Added
        );
        let native = shared.envelope(ActionSourceClass::Native, ());
        assert_eq!(
            state
                .enqueue_action(WebAction::Gyro {
                    alpha: 1.0,
                    beta: 2.0,
                    gamma: 3.0,
                })
                .await,
            EnqueueOutcome::Added
        );

        let queue = state.actions.lock().await;
        assert_eq!(queue[0].sequence().get(), 1);
        assert_eq!(queue[0].source(), ActionSourceClass::Browser);
        assert_eq!(native.sequence().get(), 2);
        assert_eq!(native.source(), ActionSourceClass::Native);
        assert_eq!(queue[1].sequence().get(), 3);
        assert_eq!(queue[1].source(), ActionSourceClass::Phone);
    }

    #[tokio::test]
    async fn dequeue_apply_and_audience_submit_share_one_engine_sequence() {
        let state = WebState::new().unwrap();
        assert_eq!(
            state
                .enqueue_action(WebAction::Gyro {
                    alpha: 1.0,
                    beta: 2.0,
                    gamma: 3.0,
                })
                .await,
            EnqueueOutcome::Added
        );
        let action = state.actions.lock().await.pop().unwrap();
        assert_eq!(action.source(), ActionSourceClass::Phone);
        assert!(state.record_action_apply(&action, Instant::now()));
        state.record_action_submission(action.sequence().get(), 19, Instant::now());

        let snapshot = state.action_timing_snapshot();
        assert_eq!(snapshot.last_presented_sequence, action.sequence().get());
        assert_eq!(snapshot.last_submission_generation, 19);
        assert_eq!(snapshot.pending, 0);
        let receipts = state.action_receipts_for_test();
        assert_eq!(
            receipts.last().unwrap().disposition,
            ActionDisposition::Presented
        );
    }
}
