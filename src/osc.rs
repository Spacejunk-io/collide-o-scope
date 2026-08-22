//! Safe, typed, bounded bidirectional OSC over UDP.
//!
//! Network packets can only carry a closed control address and one scalar;
//! they cannot select files, peers, bind interfaces, or other authority.

#![allow(
    dead_code,
    reason = "M5 OSC runtime seams are consumed by Main and Web after API freeze"
)]

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::controller_profile::{
    bounded_status, default_control_state_dir, read_bounded_document, write_atomic_document,
    AutomationOrigin, AutomationValue, BoundedDocumentReadError, ControlParameter,
    PersistedDocumentLoadStatus, RuntimeControlAddress, RuntimeNodeScope,
};
use crate::image_routing::StableLayerId;
use crate::performance::SceneId;
use crate::visual_rack::{GroupId, NodeId};

pub const OSC_CONFIG_VERSION: u16 = 1;
pub const OSC_CONFIG_FILE_NAME: &str = "osc_config.json";
pub const OSC_CONFIG_MAX_BYTES: usize = 32 * 1024;
pub const OSC_MAX_FEEDBACK_PEERS: usize = 8;
pub const OSC_MAX_DATAGRAM_BYTES: usize = 16 * 1024;
const OSC_UDP_RECEIVE_BYTES: usize = 65_535;
/// A continuously flooded socket must yield to feedback, stop, and the worker
/// sleep instead of draining until the OS happens to report WouldBlock.
const OSC_MAX_RECEIVE_ATTEMPTS_PER_TICK: usize = 64;
pub const OSC_MAX_STRING_BYTES: usize = 256;
pub const OSC_MAX_BUNDLE_DEPTH: usize = 4;
pub const OSC_MAX_MESSAGES_PER_PACKET: usize = 128;
pub const OSC_EVENT_QUEUE_CAPACITY: usize = 1_024;
pub const OSC_FEEDBACK_KEYS_CAPACITY: usize = 512;
pub const OSC_MAX_PACKETS_PER_SECOND: u64 = 2_000;
pub const OSC_FEEDBACK_RATE_PER_SECOND: u32 = 120;
const OSC_WORKER_TICK: Duration = Duration::from_millis(2);
const OSC_LOOP_SUPPRESSION_WINDOW: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OscConfigLoad {
    pub path: PathBuf,
    pub document: OscConfigDocument,
    pub status: PersistedDocumentLoadStatus,
}

pub fn default_osc_config_path() -> PathBuf {
    default_control_state_dir().join(OSC_CONFIG_FILE_NAME)
}

pub fn load_default_osc_config() -> OscConfigLoad {
    load_osc_config_or_default(&default_osc_config_path())
}

pub fn load_osc_config_or_default(path: &Path) -> OscConfigLoad {
    let (document, status) = match read_bounded_document(path, OSC_CONFIG_MAX_BYTES) {
        Ok(Some(bytes)) => match OscConfigDocument::from_json_bytes(&bytes) {
            Ok(document) => (document, PersistedDocumentLoadStatus::Loaded),
            Err(error) => (
                OscConfigDocument::default(),
                PersistedDocumentLoadStatus::DefaultInvalid(bounded_status(error.to_string())),
            ),
        },
        Ok(None) => (
            OscConfigDocument::default(),
            PersistedDocumentLoadStatus::DefaultMissing,
        ),
        Err(BoundedDocumentReadError::TooLarge(bytes)) => (
            OscConfigDocument::default(),
            PersistedDocumentLoadStatus::DefaultInvalid(bounded_status(format!(
                "document is {bytes} bytes; limit is {OSC_CONFIG_MAX_BYTES}"
            ))),
        ),
        Err(BoundedDocumentReadError::Io(error)) => (
            OscConfigDocument::default(),
            PersistedDocumentLoadStatus::DefaultIo(bounded_status(error)),
        ),
    };
    OscConfigLoad {
        path: path.to_path_buf(),
        document,
        status,
    }
}

pub fn save_osc_config_atomic(document: &OscConfigDocument, path: &Path) -> Result<(), OscError> {
    let bytes = document.to_json_bytes()?;
    write_atomic_document(path, &bytes)
        .map_err(|error| OscError::Io(bounded_status(error.to_string())))
}

pub fn save_default_osc_config_atomic(document: &OscConfigDocument) -> Result<PathBuf, OscError> {
    let path = default_osc_config_path();
    save_osc_config_atomic(document, &path)?;
    Ok(path)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum OscBindMode {
    Loopback { port: u16 },
    Lan { port: u16, enabled: bool },
}

impl Default for OscBindMode {
    fn default() -> Self {
        Self::Loopback { port: 9_000 }
    }
}

impl OscBindMode {
    pub const fn address(self) -> SocketAddr {
        match self {
            Self::Loopback { port } => SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port),
            Self::Lan { port, .. } => SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port),
        }
    }

    pub const fn lan_warning(self) -> bool {
        matches!(self, Self::Lan { enabled: true, .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OscConfigDocument {
    pub version: u16,
    pub bind: OscBindMode,
    pub feedback_peers: Vec<SocketAddr>,
}

impl Default for OscConfigDocument {
    fn default() -> Self {
        Self {
            version: OSC_CONFIG_VERSION,
            bind: OscBindMode::default(),
            feedback_peers: Vec::new(),
        }
    }
}

impl OscConfigDocument {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, OscError> {
        if bytes.len() > OSC_CONFIG_MAX_BYTES {
            return Err(OscError::ConfigBytes(bytes.len()));
        }
        let config: Self = serde_json::from_slice(bytes)
            .map_err(|error| OscError::ConfigJson(error.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, OscError> {
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| OscError::ConfigJson(error.to_string()))?;
        if bytes.len() > OSC_CONFIG_MAX_BYTES {
            return Err(OscError::ConfigBytes(bytes.len()));
        }
        Ok(bytes)
    }

    pub fn validate(&self) -> Result<(), OscError> {
        if self.version != OSC_CONFIG_VERSION {
            return Err(OscError::ConfigVersion(self.version));
        }
        if matches!(self.bind, OscBindMode::Lan { enabled: false, .. }) {
            return Err(OscError::LanNotExplicitlyEnabled);
        }
        if self.feedback_peers.len() > OSC_MAX_FEEDBACK_PEERS {
            return Err(OscError::FeedbackPeerCount(self.feedback_peers.len()));
        }
        for peer in &self.feedback_peers {
            if peer.port() == 0
                || peer.ip().is_unspecified()
                || peer.ip().is_multicast()
                || peer.ip() == IpAddr::V4(Ipv4Addr::BROADCAST)
                || (matches!(self.bind, OscBindMode::Loopback { .. }) && !peer.ip().is_loopback())
            {
                return Err(OscError::UnsafeFeedbackPeer(*peer));
            }
        }
        Ok(())
    }
}

pub fn format_control_address(address: RuntimeControlAddress) -> Result<String, OscError> {
    let path = match address {
        RuntimeControlAddress::LegacyMidiSlot(_) => return Err(OscError::UnsupportedAddress),
        RuntimeControlAddress::Master(parameter) => {
            format!("/collide/v1/master/{}", parameter.key())
        }
        RuntimeControlAddress::Layer {
            layer_id,
            parameter,
        } => format!("/collide/v1/layer/{}/{}", layer_id.get(), parameter.key()),
        RuntimeControlAddress::Group {
            group_id,
            parameter,
        } => format!("/collide/v1/group/{}/{}", group_id.get(), parameter.key()),
        RuntimeControlAddress::Node {
            scope,
            node_id,
            parameter,
        } => match scope {
            RuntimeNodeScope::Master => format!(
                "/collide/v1/node/master/{}/{}",
                node_id.get(),
                parameter.key()
            ),
            RuntimeNodeScope::Layer(layer_id) => format!(
                "/collide/v1/node/layer/{}/{}/{}",
                layer_id.get(),
                node_id.get(),
                parameter.key()
            ),
            RuntimeNodeScope::Group(group_id) => format!(
                "/collide/v1/node/group/{}/{}/{}",
                group_id.get(),
                node_id.get(),
                parameter.key()
            ),
        },
        RuntimeControlAddress::Transport(parameter) => {
            format!("/collide/v1/transport/{}", parameter.key())
        }
        RuntimeControlAddress::ScenePrepare { scene_id } => {
            format!("/collide/v1/scene/{}/prepare", scene_id.get())
        }
        RuntimeControlAddress::SceneTrigger { scene_id } => {
            format!("/collide/v1/scene/{}/trigger", scene_id.get())
        }
    };
    if path.len() > OSC_MAX_STRING_BYTES {
        Err(OscError::StringBytes(path.len()))
    } else {
        Ok(path)
    }
}

pub fn parse_control_address(path: &str) -> Result<RuntimeControlAddress, OscError> {
    if path.len() > OSC_MAX_STRING_BYTES || path.chars().any(char::is_control) {
        return Err(OscError::StringBytes(path.len()));
    }
    let segments = path.split('/').collect::<Vec<_>>();
    if segments.get(0..3) != Some(&["", "collide", "v1"]) {
        return Err(OscError::Address);
    }
    let parameter = |value: Option<&&str>| {
        value
            .and_then(|value| ControlParameter::parse(value))
            .ok_or(OscError::Parameter)
    };
    match segments.as_slice() {
        ["", "collide", "v1", "master", key] => {
            Ok(RuntimeControlAddress::Master(parameter(Some(key))?))
        }
        ["", "collide", "v1", "layer", layer, key] => Ok(RuntimeControlAddress::Layer {
            layer_id: StableLayerId::new(parse_u64(layer)?).ok_or(OscError::StableId)?,
            parameter: parameter(Some(key))?,
        }),
        ["", "collide", "v1", "group", group, key] => Ok(RuntimeControlAddress::Group {
            group_id: GroupId::new(parse_u64(group)?).ok_or(OscError::StableId)?,
            parameter: parameter(Some(key))?,
        }),
        ["", "collide", "v1", "transport", key] => {
            Ok(RuntimeControlAddress::Transport(parameter(Some(key))?))
        }
        ["", "collide", "v1", "scene", scene, "prepare"] => {
            Ok(RuntimeControlAddress::ScenePrepare {
                scene_id: parse_scene_id(scene)?,
            })
        }
        ["", "collide", "v1", "scene", scene, "trigger"] => {
            Ok(RuntimeControlAddress::SceneTrigger {
                scene_id: parse_scene_id(scene)?,
            })
        }
        ["", "collide", "v1", "node", "master", node, key] => Ok(RuntimeControlAddress::Node {
            scope: RuntimeNodeScope::Master,
            node_id: NodeId::new(parse_u64(node)?).ok_or(OscError::StableId)?,
            parameter: parameter(Some(key))?,
        }),
        ["", "collide", "v1", "node", "layer", layer, node, key] => {
            Ok(RuntimeControlAddress::Node {
                scope: RuntimeNodeScope::Layer(
                    StableLayerId::new(parse_u64(layer)?).ok_or(OscError::StableId)?,
                ),
                node_id: NodeId::new(parse_u64(node)?).ok_or(OscError::StableId)?,
                parameter: parameter(Some(key))?,
            })
        }
        ["", "collide", "v1", "node", "group", group, node, key] => {
            Ok(RuntimeControlAddress::Node {
                scope: RuntimeNodeScope::Group(
                    GroupId::new(parse_u64(group)?).ok_or(OscError::StableId)?,
                ),
                node_id: NodeId::new(parse_u64(node)?).ok_or(OscError::StableId)?,
                parameter: parameter(Some(key))?,
            })
        }
        _ => Err(OscError::Address),
    }
}

fn parse_u64(value: &str) -> Result<u64, OscError> {
    if value.is_empty() || value.len() > 20 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(OscError::StableId);
    }
    value.parse().map_err(|_| OscError::StableId)
}

fn parse_scene_id(value: &str) -> Result<SceneId, OscError> {
    let value = parse_u64(value)?;
    let value = u16::try_from(value).map_err(|_| OscError::StableId)?;
    SceneId::new(value).ok_or(OscError::StableId)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OscEvent {
    pub address: RuntimeControlAddress,
    pub value: AutomationValue,
    pub origin: AutomationOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DecodedOscMessage {
    address: RuntimeControlAddress,
    value: f32,
}

pub fn decode_packet(bytes: &[u8], peer: SocketAddr) -> Result<Vec<OscEvent>, OscError> {
    if bytes.len() > OSC_MAX_DATAGRAM_BYTES {
        return Err(OscError::DatagramBytes(bytes.len()));
    }
    let mut messages = Vec::new();
    decode_packet_inner(bytes, 0, &mut messages)?;
    Ok(messages
        .into_iter()
        .map(|message| OscEvent {
            address: message.address,
            // OSC has no held-button decoder. An asserted Scene action is one
            // message pulse; a false/zero packet is an inert release. MIDI
            // reaches the same adapter as a true rising-edge Trigger.
            value: if message.address.is_scene_action() {
                if message.value >= 0.5 {
                    AutomationValue::Trigger
                } else {
                    AutomationValue::Gate(false)
                }
            } else {
                AutomationValue::Absolute(message.value)
            },
            origin: AutomationOrigin::Osc(peer),
        })
        .collect())
}

fn decode_packet_inner(
    bytes: &[u8],
    depth: usize,
    messages: &mut Vec<DecodedOscMessage>,
) -> Result<(), OscError> {
    if depth > OSC_MAX_BUNDLE_DEPTH {
        return Err(OscError::BundleDepth(depth));
    }
    if messages.len() >= OSC_MAX_MESSAGES_PER_PACKET {
        return Err(OscError::MessageCount(messages.len() + 1));
    }
    if bytes.starts_with(b"#bundle\0") {
        if bytes.len() < 16 {
            return Err(OscError::Malformed);
        }
        let mut offset = 16;
        while offset < bytes.len() {
            let length = read_i32(bytes, offset)?;
            offset += 4;
            let length = usize::try_from(length).map_err(|_| OscError::Malformed)?;
            let end = offset.checked_add(length).ok_or(OscError::Malformed)?;
            let element = bytes.get(offset..end).ok_or(OscError::Malformed)?;
            decode_packet_inner(element, depth + 1, messages)?;
            offset = end;
        }
        return Ok(());
    }
    let (address, offset) = read_osc_string(bytes, 0)?;
    let (types, offset) = read_osc_string(bytes, offset)?;
    let address = parse_control_address(address)?;
    let (value, expected_length) = match types {
        ",f" => {
            let bits = u32::from_be_bytes(
                bytes
                    .get(offset..offset + 4)
                    .ok_or(OscError::Malformed)?
                    .try_into()
                    .map_err(|_| OscError::Malformed)?,
            );
            let value = f32::from_bits(bits);
            if !value.is_finite() {
                return Err(OscError::NonFinite);
            }
            (value.clamp(0.0, 1.0), offset.saturating_add(4))
        }
        ",i" => (
            read_i32(bytes, offset)?.clamp(0, 1) as f32,
            offset.saturating_add(4),
        ),
        ",T" => (1.0, offset),
        ",F" => (0.0, offset),
        _ => return Err(OscError::TypeTag),
    };
    if expected_length != bytes.len() {
        return Err(OscError::Malformed);
    }
    messages.push(DecodedOscMessage { address, value });
    Ok(())
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, OscError> {
    Ok(i32::from_be_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or(OscError::Malformed)?
            .try_into()
            .map_err(|_| OscError::Malformed)?,
    ))
}

fn read_osc_string(bytes: &[u8], offset: usize) -> Result<(&str, usize), OscError> {
    let tail = bytes.get(offset..).ok_or(OscError::Malformed)?;
    let length = tail
        .iter()
        .position(|byte| *byte == 0)
        .ok_or(OscError::Malformed)?;
    if length > OSC_MAX_STRING_BYTES {
        return Err(OscError::StringBytes(length));
    }
    let value = std::str::from_utf8(&tail[..length]).map_err(|_| OscError::Utf8)?;
    let padded = length
        .checked_add(1)
        .and_then(|value| value.checked_add(3))
        .map(|value| value & !3)
        .ok_or(OscError::Malformed)?;
    let next = offset.checked_add(padded).ok_or(OscError::Malformed)?;
    if next > bytes.len() || tail[length + 1..padded].iter().any(|byte| *byte != 0) {
        return Err(OscError::Malformed);
    }
    Ok((value, next))
}

pub fn encode_feedback(address: RuntimeControlAddress, value: f32) -> Result<Vec<u8>, OscError> {
    if !value.is_finite() {
        return Err(OscError::NonFinite);
    }
    let address = format_control_address(address)?;
    let mut bytes = Vec::with_capacity(address.len() + 16);
    push_osc_string(&mut bytes, &address)?;
    push_osc_string(&mut bytes, ",f")?;
    bytes.extend_from_slice(&value.clamp(0.0, 1.0).to_bits().to_be_bytes());
    Ok(bytes)
}

fn push_osc_string(output: &mut Vec<u8>, value: &str) -> Result<(), OscError> {
    if value.len() > OSC_MAX_STRING_BYTES || value.as_bytes().contains(&0) {
        return Err(OscError::StringBytes(value.len()));
    }
    output.extend_from_slice(value.as_bytes());
    output.push(0);
    while !output.len().is_multiple_of(4) {
        output.push(0);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OscCounters {
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

#[derive(Default)]
struct OscAtomicCounters {
    datagrams_received: AtomicU64,
    messages_received: AtomicU64,
    malformed: AtomicU64,
    rate_dropped: AtomicU64,
    queue_dropped: AtomicU64,
    loop_suppressed: AtomicU64,
    feedback_rate_limited: AtomicU64,
    feedback_sent: AtomicU64,
    feedback_send_errors: AtomicU64,
}

impl OscAtomicCounters {
    fn snapshot(&self, local: OscCounters) -> OscCounters {
        OscCounters {
            datagrams_received: self
                .datagrams_received
                .load(Ordering::Relaxed)
                .saturating_add(local.datagrams_received),
            messages_received: self
                .messages_received
                .load(Ordering::Relaxed)
                .saturating_add(local.messages_received),
            malformed: self
                .malformed
                .load(Ordering::Relaxed)
                .saturating_add(local.malformed),
            rate_dropped: self
                .rate_dropped
                .load(Ordering::Relaxed)
                .saturating_add(local.rate_dropped),
            queue_dropped: self
                .queue_dropped
                .load(Ordering::Relaxed)
                .saturating_add(local.queue_dropped),
            loop_suppressed: self
                .loop_suppressed
                .load(Ordering::Relaxed)
                .saturating_add(local.loop_suppressed),
            feedback_queued: local.feedback_queued,
            feedback_coalesced: local.feedback_coalesced,
            feedback_dropped: local.feedback_dropped,
            feedback_rate_limited: self
                .feedback_rate_limited
                .load(Ordering::Relaxed)
                .saturating_add(local.feedback_rate_limited),
            feedback_sent: self
                .feedback_sent
                .load(Ordering::Relaxed)
                .saturating_add(local.feedback_sent),
            feedback_send_errors: self
                .feedback_send_errors
                .load(Ordering::Relaxed)
                .saturating_add(local.feedback_send_errors),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OscRuntimePhase {
    Disabled,
    Binding,
    Listening,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OscRuntimeSnapshot {
    pub phase: OscRuntimePhase,
    pub bound_address: Option<SocketAddr>,
    pub lan_warning: bool,
    pub error: Option<String>,
    pub counters: OscCounters,
}

impl Default for OscRuntimeSnapshot {
    fn default() -> Self {
        Self {
            phase: OscRuntimePhase::Disabled,
            bound_address: None,
            lan_warning: false,
            error: None,
            counters: OscCounters::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FeedbackEntry {
    address: RuntimeControlAddress,
    value: f32,
}

#[derive(Debug, Clone, Copy)]
struct RecentFeedback {
    peer: SocketAddr,
    address: RuntimeControlAddress,
    value_bits: u32,
    sent_at: Instant,
}

struct OscShared {
    events: Mutex<VecDeque<OscEvent>>,
    feedback: Mutex<BTreeMap<(SocketAddr, RuntimeControlAddress), FeedbackEntry>>,
    recent: Mutex<VecDeque<RecentFeedback>>,
    counters: OscAtomicCounters,
}

impl Default for OscShared {
    fn default() -> Self {
        Self {
            events: Mutex::new(VecDeque::with_capacity(OSC_EVENT_QUEUE_CAPACITY)),
            feedback: Mutex::new(BTreeMap::new()),
            recent: Mutex::new(VecDeque::with_capacity(OSC_FEEDBACK_KEYS_CAPACITY)),
            counters: OscAtomicCounters::default(),
        }
    }
}

impl OscShared {
    fn push_event(&self, event: OscEvent) {
        let Ok(mut queue) = self.events.try_lock() else {
            self.counters.queue_dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if queue.len() == OSC_EVENT_QUEUE_CAPACITY {
            self.counters.queue_dropped.fetch_add(1, Ordering::Relaxed);
        } else {
            queue.push_back(event);
        }
    }

    fn clear_protocol_queues(&self) {
        if let Ok(mut events) = self.events.lock() {
            events.clear();
        }
        if let Ok(mut feedback) = self.feedback.lock() {
            feedback.clear();
        }
        if let Ok(mut recent) = self.recent.lock() {
            recent.clear();
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PacketRateWindow {
    started: Instant,
    accepted: u64,
}

impl PacketRateWindow {
    fn new(now: Instant) -> Self {
        Self {
            started: now,
            accepted: 0,
        }
    }

    fn admit(&mut self, now: Instant) -> bool {
        if now.duration_since(self.started) >= Duration::from_secs(1) {
            self.started = now;
            self.accepted = 0;
        }
        if self.accepted >= OSC_MAX_PACKETS_PER_SECOND {
            return false;
        }
        self.accepted = self.accepted.saturating_add(1);
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct FeedbackRateGate {
    last: Instant,
    interval: Duration,
}

impl FeedbackRateGate {
    fn ready(now: Instant) -> Self {
        let interval = Duration::from_secs_f64(1.0 / OSC_FEEDBACK_RATE_PER_SECOND as f64);
        Self {
            last: now.checked_sub(interval).unwrap_or(now),
            interval,
        }
    }

    fn admit(&mut self, now: Instant) -> bool {
        if now.duration_since(self.last) < self.interval {
            return false;
        }
        self.last = now;
        true
    }
}

struct OscWorker {
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<OscRuntimeSnapshot>>,
    join: Option<JoinHandle<()>>,
}

pub struct OscEngine {
    config: OscConfigDocument,
    shared: Arc<OscShared>,
    worker: Option<OscWorker>,
    local_counters: OscCounters,
}

impl OscEngine {
    pub fn new(config: OscConfigDocument) -> Result<Self, OscError> {
        config.validate()?;
        Ok(Self {
            config,
            shared: Arc::new(OscShared::default()),
            worker: None,
            local_counters: OscCounters::default(),
        })
    }

    pub fn start(&mut self) -> Result<(), OscError> {
        if self.worker.is_some() {
            return Ok(());
        }
        self.config.validate()?;
        let stop = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(OscRuntimeSnapshot {
            phase: OscRuntimePhase::Binding,
            lan_warning: self.config.bind.lan_warning(),
            ..OscRuntimeSnapshot::default()
        }));
        let worker_stop = stop.clone();
        let worker_status = status.clone();
        let shared = self.shared.clone();
        let config = self.config.clone();
        let join = std::thread::Builder::new()
            .name("collide-osc-udp".into())
            .spawn(move || osc_worker(config, shared, worker_stop, worker_status))
            .map_err(|error| OscError::Io(error.to_string()))?;
        self.worker = Some(OscWorker {
            stop,
            status,
            join: Some(join),
        });
        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(mut worker) = self.worker.take() {
            worker.stop.store(true, Ordering::Release);
            if let Some(join) = worker.join.take() {
                let _ = join.join();
            }
        }
    }

    pub fn apply_config(&mut self, config: OscConfigDocument) -> Result<(), OscError> {
        config.validate()?;
        let restart = self.worker.is_some();
        if restart {
            self.stop();
        }
        self.shared.clear_protocol_queues();
        self.config = config;
        if restart {
            self.start()?;
        }
        Ok(())
    }

    pub fn drain_events(&mut self, output: &mut Vec<OscEvent>) {
        if let Ok(mut events) = self.shared.events.try_lock() {
            output.extend(events.drain(..));
        }
    }

    pub fn queue_feedback(
        &mut self,
        address: RuntimeControlAddress,
        value: f32,
        origin: AutomationOrigin,
    ) {
        if !value.is_finite() {
            self.local_counters.feedback_dropped =
                self.local_counters.feedback_dropped.saturating_add(1);
            return;
        }
        for peer in &self.config.feedback_peers {
            if matches!(origin, AutomationOrigin::Osc(source) if source == *peer) {
                self.local_counters.loop_suppressed =
                    self.local_counters.loop_suppressed.saturating_add(1);
                continue;
            }
            let Ok(mut feedback) = self.shared.feedback.try_lock() else {
                self.local_counters.feedback_dropped =
                    self.local_counters.feedback_dropped.saturating_add(1);
                continue;
            };
            let key = (*peer, address);
            if feedback.contains_key(&key) {
                self.local_counters.feedback_coalesced =
                    self.local_counters.feedback_coalesced.saturating_add(1);
            } else if feedback.len() >= OSC_FEEDBACK_KEYS_CAPACITY {
                self.local_counters.feedback_dropped =
                    self.local_counters.feedback_dropped.saturating_add(1);
                continue;
            }
            feedback.insert(
                key,
                FeedbackEntry {
                    address,
                    value: value.clamp(0.0, 1.0),
                },
            );
            self.local_counters.feedback_queued =
                self.local_counters.feedback_queued.saturating_add(1);
        }
    }

    pub fn runtime_snapshot(&self) -> OscRuntimeSnapshot {
        let mut snapshot = self
            .worker
            .as_ref()
            .and_then(|worker| worker.status.lock().ok().map(|value| value.clone()))
            .unwrap_or_default();
        snapshot.counters = self.shared.counters.snapshot(self.local_counters);
        snapshot
    }
}

impl Drop for OscEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

fn osc_worker(
    config: OscConfigDocument,
    shared: Arc<OscShared>,
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<OscRuntimeSnapshot>>,
) {
    let socket = match UdpSocket::bind(config.bind.address()) {
        Ok(socket) => socket,
        Err(error) => {
            if let Ok(mut current) = status.lock() {
                current.phase = OscRuntimePhase::Error;
                current.error = Some(error.to_string());
            }
            return;
        }
    };
    if let Err(error) = socket.set_nonblocking(true) {
        if let Ok(mut current) = status.lock() {
            current.phase = OscRuntimePhase::Error;
            current.error = Some(error.to_string());
        }
        return;
    }
    if let Ok(mut current) = status.lock() {
        current.phase = OscRuntimePhase::Listening;
        current.bound_address = socket.local_addr().ok();
    }
    let mut buffer = [0_u8; OSC_UDP_RECEIVE_BYTES];
    let now = Instant::now();
    let mut packet_rate = PacketRateWindow::new(now);
    let mut feedback_rate = FeedbackRateGate::ready(now);
    while !stop.load(Ordering::Acquire) {
        if let Err(error) =
            receive_osc_batch(&socket, &mut buffer, &shared, &stop, &mut packet_rate)
        {
            if let Ok(mut current) = status.lock() {
                current.phase = OscRuntimePhase::Error;
                current.error = Some(error.to_string());
            }
        }

        let feedback_pending = shared
            .feedback
            .try_lock()
            .is_ok_and(|feedback| !feedback.is_empty());
        if feedback_pending && feedback_rate.admit(Instant::now()) {
            let next = shared
                .feedback
                .try_lock()
                .ok()
                .and_then(|mut feedback| feedback.pop_first());
            if let Some(((peer, _), entry)) = next {
                match encode_feedback(entry.address, entry.value).and_then(|bytes| {
                    socket
                        .send_to(&bytes, peer)
                        .map_err(|e| OscError::Io(e.to_string()))
                }) {
                    Ok(_) => {
                        shared
                            .counters
                            .feedback_sent
                            .fetch_add(1, Ordering::Relaxed);
                        if let Ok(mut recent) = shared.recent.try_lock() {
                            if recent.len() == OSC_FEEDBACK_KEYS_CAPACITY {
                                recent.pop_front();
                            }
                            recent.push_back(RecentFeedback {
                                peer,
                                address: entry.address,
                                value_bits: entry.value.to_bits(),
                                sent_at: Instant::now(),
                            });
                        }
                    }
                    Err(_) => {
                        shared
                            .counters
                            .feedback_send_errors
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        } else if feedback_pending {
            shared
                .counters
                .feedback_rate_limited
                .fetch_add(1, Ordering::Relaxed);
        }
        std::thread::sleep(OSC_WORKER_TICK);
    }
    if let Ok(mut current) = status.lock() {
        current.phase = OscRuntimePhase::Disabled;
    }
}

/// Process one bounded receive batch. The stop flag is checked between every
/// datagram so `OscEngine::stop` cannot block its caller behind a sustained
/// UDP flood, and the fixed attempt ceiling ensures each tick reaches feedback
/// and its scheduler yield even when the socket never becomes empty.
fn receive_osc_batch(
    socket: &UdpSocket,
    buffer: &mut [u8; OSC_UDP_RECEIVE_BYTES],
    shared: &OscShared,
    stop: &AtomicBool,
    packet_rate: &mut PacketRateWindow,
) -> std::io::Result<usize> {
    let mut attempts = 0_usize;
    while attempts < OSC_MAX_RECEIVE_ATTEMPTS_PER_TICK && !stop.load(Ordering::Acquire) {
        match socket.recv_from(buffer) {
            Ok((length, peer)) => {
                attempts += 1;
                shared
                    .counters
                    .datagrams_received
                    .fetch_add(1, Ordering::Relaxed);
                if length > OSC_MAX_DATAGRAM_BYTES {
                    shared.counters.malformed.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                if !packet_rate.admit(Instant::now()) {
                    shared.counters.rate_dropped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                match decode_packet(&buffer[..length], peer) {
                    Ok(events) => {
                        shared
                            .counters
                            .messages_received
                            .fetch_add(events.len() as u64, Ordering::Relaxed);
                        for event in events {
                            if is_osc_feedback_echo(shared, event) {
                                shared
                                    .counters
                                    .loop_suppressed
                                    .fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            shared.push_event(event);
                        }
                    }
                    Err(_) => {
                        shared.counters.malformed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(error) => return Err(error),
        }
    }
    Ok(attempts)
}

fn is_osc_feedback_echo(shared: &OscShared, event: OscEvent) -> bool {
    let AutomationOrigin::Osc(peer) = event.origin else {
        return false;
    };
    let value = match event.value {
        AutomationValue::Absolute(value) => value,
        // Scene packets decode asserted feedback as a pulse and zero as the
        // inert release half. Compare those typed values to the exact float
        // put on the wire, or a looped-back ready LED packet can become an
        // unsolicited Scene launch.
        AutomationValue::Trigger if event.address.is_scene_action() => 1.0,
        AutomationValue::Gate(value) if event.address.is_scene_action() => {
            if value {
                1.0
            } else {
                0.0
            }
        }
        AutomationValue::Delta(_) | AutomationValue::Trigger | AutomationValue::Gate(_) => {
            return false
        }
    };
    if !value.is_finite() {
        return false;
    }
    let Ok(mut recent) = shared.recent.try_lock() else {
        return false;
    };
    let now = Instant::now();
    while recent
        .front()
        .is_some_and(|entry| now.duration_since(entry.sent_at) > OSC_LOOP_SUPPRESSION_WINDOW)
    {
        recent.pop_front();
    }
    recent.iter().any(|entry| {
        entry.peer == peer && entry.address == event.address && entry.value_bits == value.to_bits()
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OscError {
    ConfigBytes(usize),
    ConfigJson(String),
    ConfigVersion(u16),
    LanNotExplicitlyEnabled,
    FeedbackPeerCount(usize),
    UnsafeFeedbackPeer(SocketAddr),
    DatagramBytes(usize),
    StringBytes(usize),
    BundleDepth(usize),
    MessageCount(usize),
    Address,
    Parameter,
    StableId,
    UnsupportedAddress,
    TypeTag,
    NonFinite,
    Utf8,
    Malformed,
    Io(String),
}

impl fmt::Display for OscError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "OSC error: {self:?}")
    }
}

impl std::error::Error for OscError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::AtomicU64;
    use std::thread;

    static TEST_PATH_SEQUENCE: AtomicU64 = AtomicU64::new(1);

    fn test_config_path(label: &str) -> PathBuf {
        let ordinal = TEST_PATH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "collide-o-scope-osc-config-{label}-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        directory.join(OSC_CONFIG_FILE_NAME)
    }

    fn remove_test_config(path: &Path) {
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

    fn peer() -> SocketAddr {
        "127.0.0.1:32123".parse().unwrap()
    }

    fn layer(value: u64) -> StableLayerId {
        StableLayerId::new(value).unwrap()
    }

    fn scene(value: u16) -> SceneId {
        SceneId::new(value).unwrap()
    }

    #[test]
    fn typed_addresses_round_trip_stable_scopes_and_closed_parameters() {
        let addresses = [
            RuntimeControlAddress::Master(ControlParameter::Brightness),
            RuntimeControlAddress::Layer {
                layer_id: layer(42),
                parameter: ControlParameter::Opacity,
            },
            RuntimeControlAddress::Group {
                group_id: GroupId::new(7).unwrap(),
                parameter: ControlParameter::Solo,
            },
            RuntimeControlAddress::Node {
                scope: RuntimeNodeScope::Layer(layer(42)),
                node_id: NodeId::new(9).unwrap(),
                parameter: ControlParameter::Wet,
            },
            RuntimeControlAddress::Transport(ControlParameter::ProgramFreeze),
            RuntimeControlAddress::ScenePrepare {
                scene_id: scene(12),
            },
            RuntimeControlAddress::SceneTrigger {
                scene_id: scene(12),
            },
        ];
        for address in addresses {
            let encoded = format_control_address(address).unwrap();
            assert_eq!(parse_control_address(&encoded), Ok(address));
        }
        assert_eq!(
            format_control_address(RuntimeControlAddress::ScenePrepare {
                scene_id: scene(12),
            })
            .unwrap(),
            "/collide/v1/scene/12/prepare"
        );
        assert_eq!(
            format_control_address(RuntimeControlAddress::SceneTrigger {
                scene_id: scene(12),
            })
            .unwrap(),
            "/collide/v1/scene/12/trigger"
        );
        assert_eq!(
            parse_control_address("/collide/v1/layer/42/../../file"),
            Err(OscError::Address)
        );
        assert_eq!(
            parse_control_address("/collide/v1/master/not_a_parameter"),
            Err(OscError::Parameter)
        );
        assert_eq!(
            parse_control_address("/collide/v1/scene/0/trigger"),
            Err(OscError::StableId)
        );
        assert_eq!(
            parse_control_address("/collide/v1/scene/65536/prepare"),
            Err(OscError::StableId)
        );
        assert_eq!(
            parse_control_address("/collide/v1/scene/12/fire"),
            Err(OscError::Address)
        );
    }

    #[test]
    fn scene_action_packets_are_asserted_pulses_with_inert_releases() {
        let trigger = RuntimeControlAddress::SceneTrigger { scene_id: scene(4) };
        let asserted = decode_packet(&encode_feedback(trigger, 1.0).unwrap(), peer()).unwrap();
        assert_eq!(asserted.len(), 1);
        assert_eq!(asserted[0].address, trigger);
        assert_eq!(asserted[0].value, AutomationValue::Trigger);

        let released = decode_packet(&encode_feedback(trigger, 0.0).unwrap(), peer()).unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].address, trigger);
        assert_eq!(released[0].value, AutomationValue::Gate(false));
    }

    #[test]
    fn looped_scene_feedback_pulses_and_releases_are_suppressed() {
        let shared = OscShared::default();
        let address = RuntimeControlAddress::SceneTrigger { scene_id: scene(4) };
        {
            let mut recent = shared.recent.lock().unwrap();
            recent.push_back(RecentFeedback {
                peer: peer(),
                address,
                value_bits: 1.0_f32.to_bits(),
                sent_at: Instant::now(),
            });
            recent.push_back(RecentFeedback {
                peer: peer(),
                address,
                value_bits: 0.0_f32.to_bits(),
                sent_at: Instant::now(),
            });
        }

        let asserted = decode_packet(&encode_feedback(address, 1.0).unwrap(), peer()).unwrap();
        let released = decode_packet(&encode_feedback(address, 0.0).unwrap(), peer()).unwrap();
        assert!(is_osc_feedback_echo(&shared, asserted[0]));
        assert!(is_osc_feedback_echo(&shared, released[0]));

        let other_peer: SocketAddr = "127.0.0.1:9010".parse().unwrap();
        let remote = decode_packet(&encode_feedback(address, 1.0).unwrap(), other_peer).unwrap();
        assert!(!is_osc_feedback_echo(&shared, remote[0]));
    }

    #[test]
    fn scalar_packets_round_trip_and_nan_is_rejected() {
        let address = RuntimeControlAddress::Layer {
            layer_id: layer(8),
            parameter: ControlParameter::Opacity,
        };
        let bytes = encode_feedback(address, 0.625).unwrap();
        let decoded = decode_packet(&bytes, peer()).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].address, address);
        assert_eq!(decoded[0].value, AutomationValue::Absolute(0.625));

        let mut nan = bytes;
        let end = nan.len();
        nan[end - 4..].copy_from_slice(&f32::NAN.to_bits().to_be_bytes());
        assert_eq!(decode_packet(&nan, peer()), Err(OscError::NonFinite));
    }

    fn bundle(element: &[u8]) -> Vec<u8> {
        let mut bytes = b"#bundle\0".to_vec();
        bytes.extend_from_slice(&0_u64.to_be_bytes());
        bytes.extend_from_slice(&(element.len() as i32).to_be_bytes());
        bytes.extend_from_slice(element);
        bytes
    }

    #[test]
    fn hostile_bundle_depth_count_size_and_padding_are_rejected() {
        let message =
            encode_feedback(RuntimeControlAddress::Master(ControlParameter::Amount), 0.5).unwrap();
        let mut nested = message.clone();
        for _ in 0..=OSC_MAX_BUNDLE_DEPTH {
            nested = bundle(&nested);
        }
        assert!(matches!(
            decode_packet(&nested, peer()),
            Err(OscError::BundleDepth(_))
        ));
        assert!(matches!(
            decode_packet(&vec![0; OSC_MAX_DATAGRAM_BYTES + 1], peer()),
            Err(OscError::DatagramBytes(_))
        ));
        let mut bad_padding = message.clone();
        bad_padding[31] = 1;
        assert_eq!(
            decode_packet(&bad_padding, peer()),
            Err(OscError::Malformed)
        );

        let mut trailing =
            encode_feedback(RuntimeControlAddress::Master(ControlParameter::Amount), 0.5).unwrap();
        trailing.push(0);
        assert_eq!(decode_packet(&trailing, peer()), Err(OscError::Malformed));

        let mut too_many = b"#bundle\0".to_vec();
        too_many.extend_from_slice(&0_u64.to_be_bytes());
        for _ in 0..=OSC_MAX_MESSAGES_PER_PACKET {
            too_many.extend_from_slice(&(message.len() as i32).to_be_bytes());
            too_many.extend_from_slice(&message);
        }
        assert!(matches!(
            decode_packet(&too_many, peer()),
            Err(OscError::MessageCount(_))
        ));
    }

    #[test]
    fn loopback_is_default_and_lan_requires_visible_explicit_opt_in() {
        let default = OscConfigDocument::default();
        assert!(default.validate().is_ok());
        assert!(default.bind.address().ip().is_loopback());
        let hidden_lan = OscConfigDocument {
            bind: OscBindMode::Lan {
                port: 9_000,
                enabled: false,
            },
            ..OscConfigDocument::default()
        };
        assert_eq!(
            hidden_lan.validate(),
            Err(OscError::LanNotExplicitlyEnabled)
        );
        let visible_lan = OscConfigDocument {
            bind: OscBindMode::Lan {
                port: 9_000,
                enabled: true,
            },
            ..OscConfigDocument::default()
        };
        assert!(visible_lan.validate().is_ok());
        assert!(visible_lan.bind.lan_warning());

        let bytes = visible_lan.to_json_bytes().unwrap();
        assert_eq!(OscConfigDocument::from_json_bytes(&bytes), Ok(visible_lan));
        assert!(matches!(
            OscConfigDocument::from_json_bytes(&vec![b' '; OSC_CONFIG_MAX_BYTES + 1]),
            Err(OscError::ConfigBytes(_))
        ));

        let mut engine = OscEngine::new(default.clone()).unwrap();
        let queued_address = RuntimeControlAddress::Master(ControlParameter::Amount);
        engine.shared.push_event(OscEvent {
            address: queued_address,
            value: AutomationValue::Absolute(0.5),
            origin: AutomationOrigin::Osc(peer()),
        });
        engine.shared.feedback.lock().unwrap().insert(
            (peer(), queued_address),
            FeedbackEntry {
                address: queued_address,
                value: 0.5,
            },
        );
        assert_eq!(
            engine.apply_config(OscConfigDocument {
                bind: OscBindMode::Lan {
                    port: 9_000,
                    enabled: false,
                },
                ..OscConfigDocument::default()
            }),
            Err(OscError::LanNotExplicitlyEnabled)
        );
        assert_eq!(engine.config, default);
        assert_eq!(engine.shared.events.lock().unwrap().len(), 1);
        assert_eq!(engine.shared.feedback.lock().unwrap().len(), 1);

        engine
            .apply_config(OscConfigDocument {
                bind: OscBindMode::Loopback { port: 9_001 },
                feedback_peers: vec![peer()],
                ..OscConfigDocument::default()
            })
            .unwrap();
        assert!(engine.shared.events.lock().unwrap().is_empty());
        assert!(engine.shared.feedback.lock().unwrap().is_empty());
    }

    #[test]
    fn osc_config_persistence_defaults_safely_and_replaces_atomically() {
        let path = test_config_path("atomic");
        fs::remove_dir(path.parent().unwrap()).unwrap();
        let missing = load_osc_config_or_default(&path);
        assert_eq!(missing.document, OscConfigDocument::default());
        assert_eq!(missing.status, PersistedDocumentLoadStatus::DefaultMissing);

        let first = OscConfigDocument {
            bind: OscBindMode::Loopback { port: 9_001 },
            feedback_peers: vec![peer()],
            ..OscConfigDocument::default()
        };
        save_osc_config_atomic(&first, &path).unwrap();
        assert_eq!(load_osc_config_or_default(&path).document, first);

        let replacement = OscConfigDocument {
            bind: OscBindMode::Lan {
                port: 9_002,
                enabled: true,
            },
            feedback_peers: vec!["192.0.2.4:9002".parse().unwrap()],
            ..OscConfigDocument::default()
        };
        save_osc_config_atomic(&replacement, &path).unwrap();
        let loaded = load_osc_config_or_default(&path);
        assert_eq!(loaded.document, replacement);
        assert_eq!(loaded.status, PersistedDocumentLoadStatus::Loaded);
        let published = fs::read(&path).unwrap();
        let invalid = OscConfigDocument {
            bind: OscBindMode::Lan {
                port: 9_003,
                enabled: false,
            },
            ..OscConfigDocument::default()
        };
        assert_eq!(
            save_osc_config_atomic(&invalid, &path),
            Err(OscError::LanNotExplicitlyEnabled)
        );
        assert_eq!(fs::read(&path).unwrap(), published);
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
        remove_test_config(&path);
    }

    #[test]
    fn osc_config_persistence_rejects_hostile_and_open_serde_documents() {
        let path = test_config_path("hostile");
        fs::write(&path, vec![b'x'; OSC_CONFIG_MAX_BYTES + 1]).unwrap();
        let oversized = load_osc_config_or_default(&path);
        assert_eq!(oversized.document, OscConfigDocument::default());
        assert!(matches!(
            oversized.status,
            PersistedDocumentLoadStatus::DefaultInvalid(_)
        ));

        let config = OscConfigDocument::default();
        let mut top_level = serde_json::to_value(&config).unwrap();
        top_level["unexpected"] = serde_json::json!(true);
        assert!(matches!(
            OscConfigDocument::from_json_bytes(&serde_json::to_vec(&top_level).unwrap()),
            Err(OscError::ConfigJson(_))
        ));
        let mut nested = serde_json::to_value(&config).unwrap();
        nested["bind"]["unexpected"] = serde_json::json!(true);
        assert!(matches!(
            OscConfigDocument::from_json_bytes(&serde_json::to_vec(&nested).unwrap()),
            Err(OscError::ConfigJson(_))
        ));

        let invalid_lan = serde_json::to_vec(&OscConfigDocument {
            bind: OscBindMode::Lan {
                port: 9_000,
                enabled: false,
            },
            ..OscConfigDocument::default()
        })
        .unwrap();
        fs::write(&path, invalid_lan).unwrap();
        let defaulted = load_osc_config_or_default(&path);
        assert_eq!(defaulted.document, OscConfigDocument::default());
        assert!(matches!(
            defaulted.status,
            PersistedDocumentLoadStatus::DefaultInvalid(_)
        ));
        assert_eq!(
            default_osc_config_path().file_name(),
            Some(std::ffi::OsStr::new(OSC_CONFIG_FILE_NAME))
        );
        remove_test_config(&path);
    }

    #[test]
    fn feedback_is_coalesced_bounded_and_suppresses_its_osc_origin() {
        let mut engine = OscEngine::new(OscConfigDocument {
            feedback_peers: vec![peer()],
            ..OscConfigDocument::default()
        })
        .unwrap();
        let address = RuntimeControlAddress::Master(ControlParameter::Amount);
        engine.queue_feedback(address, 0.1, AutomationOrigin::HostAutomation);
        engine.queue_feedback(address, 0.9, AutomationOrigin::HostAutomation);
        engine.queue_feedback(address, 0.5, AutomationOrigin::Osc(peer()));
        let pending = engine.shared.feedback.lock().unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending.values().next().unwrap().value, 0.9);
        drop(pending);
        let snapshot = engine.runtime_snapshot();
        assert_eq!(snapshot.counters.feedback_queued, 2);
        assert_eq!(snapshot.counters.feedback_coalesced, 1);
        assert_eq!(snapshot.counters.loop_suppressed, 1);
    }

    #[test]
    fn packet_event_and_feedback_rates_are_hard_bounded() {
        let now = Instant::now();
        let mut packet_rate = PacketRateWindow::new(now);
        for _ in 0..OSC_MAX_PACKETS_PER_SECOND {
            assert!(packet_rate.admit(now));
        }
        assert!(!packet_rate.admit(now));
        assert!(packet_rate.admit(now + Duration::from_secs(1)));

        let shared = OscShared::default();
        let event = OscEvent {
            address: RuntimeControlAddress::Master(ControlParameter::Amount),
            value: AutomationValue::Absolute(0.5),
            origin: AutomationOrigin::Osc(peer()),
        };
        for _ in 0..OSC_EVENT_QUEUE_CAPACITY + 17 {
            shared.push_event(event);
        }
        assert_eq!(
            shared.events.lock().unwrap().len(),
            OSC_EVENT_QUEUE_CAPACITY
        );
        assert_eq!(shared.counters.queue_dropped.load(Ordering::Relaxed), 17);

        let mut feedback_rate = FeedbackRateGate::ready(now);
        let interval = Duration::from_secs_f64(1.0 / OSC_FEEDBACK_RATE_PER_SECOND as f64);
        for index in 0..OSC_FEEDBACK_RATE_PER_SECOND {
            assert!(feedback_rate.admit(now + interval * index));
            assert!(!feedback_rate.admit(now + interval * index + interval / 2));
        }

        let mut engine = OscEngine::new(OscConfigDocument {
            feedback_peers: vec![peer()],
            ..OscConfigDocument::default()
        })
        .unwrap();
        for stable_id in 1..=OSC_FEEDBACK_KEYS_CAPACITY as u64 + 9 {
            engine.queue_feedback(
                RuntimeControlAddress::Layer {
                    layer_id: layer(stable_id),
                    parameter: ControlParameter::Amount,
                },
                0.5,
                AutomationOrigin::HostAutomation,
            );
        }
        assert_eq!(
            engine.shared.feedback.lock().unwrap().len(),
            OSC_FEEDBACK_KEYS_CAPACITY
        );
        assert_eq!(engine.local_counters.feedback_dropped, 9);
    }

    #[test]
    fn receive_batch_is_bounded_and_honors_stop_with_a_queued_flood() {
        let receiver = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        receiver.set_nonblocking(true).unwrap();
        let destination = receiver.local_addr().unwrap();
        let sender = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let packet = encode_feedback(
            RuntimeControlAddress::Transport(ControlParameter::Play),
            1.0,
        )
        .unwrap();
        for _ in 0..OSC_MAX_RECEIVE_ATTEMPTS_PER_TICK * 2 {
            sender.send_to(&packet, destination).unwrap();
        }

        let shared = OscShared::default();
        let stop = AtomicBool::new(false);
        let mut packet_rate = PacketRateWindow::new(Instant::now());
        let mut buffer = [0_u8; OSC_UDP_RECEIVE_BYTES];
        let attempts =
            receive_osc_batch(&receiver, &mut buffer, &shared, &stop, &mut packet_rate).unwrap();
        assert_eq!(attempts, OSC_MAX_RECEIVE_ATTEMPTS_PER_TICK);

        stop.store(true, Ordering::Release);
        assert_eq!(
            receive_osc_batch(&receiver, &mut buffer, &shared, &stop, &mut packet_rate,).unwrap(),
            0,
            "a pending socket backlog must not outrank worker stop"
        );
    }

    #[test]
    fn udp_worker_stop_remains_bounded_under_sustained_flood() {
        let mut engine = OscEngine::new(OscConfigDocument {
            bind: OscBindMode::Loopback { port: 0 },
            ..OscConfigDocument::default()
        })
        .unwrap();
        engine.start().unwrap();
        // Binding is setup, not the claim under test, so it waits on the same
        // deadline as the loopback fixture rather than a 500 ms budget the host
        // scheduler owns. The stop bound asserted below is deliberately left
        // exact: that latency IS this test's subject.
        let bound = poll_until(|| engine.runtime_snapshot().bound_address)
            .expect("OSC worker should bind its flood-test socket");

        let packet = encode_feedback(
            RuntimeControlAddress::Transport(ControlParameter::Play),
            1.0,
        )
        .unwrap();
        let flood_stop = Arc::new(AtomicBool::new(false));
        let flood_flag = flood_stop.clone();
        let flood = thread::spawn(move || {
            let sender = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
            sender.set_nonblocking(true).unwrap();
            while !flood_flag.load(Ordering::Acquire) {
                let _ = sender.send_to(&packet, bound);
            }
        });
        thread::sleep(Duration::from_millis(25));

        let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
        let stopper = thread::spawn(move || {
            let started = Instant::now();
            engine.stop();
            let _ = done_tx.send(started.elapsed());
        });
        let stopped = done_rx.recv_timeout(Duration::from_millis(500));
        flood_stop.store(true, Ordering::Release);
        flood.join().unwrap();
        stopper.join().unwrap();
        let elapsed = stopped.expect("OSC stop was starved by sustained UDP ingress");
        assert!(
            elapsed < Duration::from_millis(500),
            "OSC stop took {elapsed:?} under flood"
        );
    }

    /// Poll `probe` until it yields a value or a generous deadline passes.
    ///
    /// The OSC worker is a real thread on a real UDP socket, so delivery
    /// latency belongs to the host scheduler and the loopback stack rather
    /// than to this crate. A fixed iteration budget therefore encodes a
    /// latency bound the test never meant to assert: the previous 100 polls at
    /// 5 ms gave the worker 500 ms to be scheduled, which a loaded hosted macOS
    /// runner exceeded, failing a test whose actual claim is that the event
    /// arrives at all and that the host is never blocked while waiting. The
    /// deadline below is far longer than any plausible scheduling delay, so a
    /// failure here means the datagram genuinely never arrived rather than that
    /// the runner was busy. Success still returns as soon as the value appears,
    /// so the ordinary run is no slower.
    fn poll_until<T>(mut probe: impl FnMut() -> Option<T>) -> Option<T> {
        const BUDGET: Duration = Duration::from_secs(30);
        let deadline = Instant::now() + BUDGET;
        loop {
            if let Some(value) = probe() {
                return Some(value);
            }
            if Instant::now() >= deadline {
                return None;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn udp_worker_receives_typed_loopback_packets_without_blocking_the_host() {
        let mut engine = OscEngine::new(OscConfigDocument {
            bind: OscBindMode::Loopback { port: 0 },
            ..OscConfigDocument::default()
        })
        .unwrap();
        engine.start().unwrap();
        let bound = poll_until(|| engine.runtime_snapshot().bound_address)
            .expect("OSC worker should bind its loopback socket");
        assert!(bound.ip().is_loopback());

        let sender = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let sender_address = sender.local_addr().unwrap();
        let address = RuntimeControlAddress::Transport(ControlParameter::Play);
        let packet = encode_feedback(address, 1.0).unwrap();
        sender.send_to(&packet, bound).unwrap();

        let received = poll_until(|| {
            let mut events = Vec::new();
            engine.drain_events(&mut events);
            events.into_iter().next()
        });
        let event = received.expect("OSC worker should publish the loopback event");
        assert_eq!(event.address, address);
        assert_eq!(event.origin, AutomationOrigin::Osc(sender_address));
        assert_eq!(event.value, AutomationValue::Absolute(1.0));

        // The decoder unit law above proves the exact
        // `OSC_MAX_DATAGRAM_BYTES + 1` rejection. Do not require the host OS to
        // transmit that synthetic datagram here: some UDP stacks reject it
        // with `EMSGSIZE` before the worker can observe it. A small malformed
        // packet exercises the same worker counter/rejection path portably.
        sender.send_to(&[0], bound).unwrap();
        let rejected = poll_until(|| {
            let malformed = engine.runtime_snapshot().counters.malformed;
            (malformed > 0).then_some(malformed)
        })
        .is_some_and(|malformed| malformed == 1);
        engine.stop();
        assert!(
            rejected,
            "malformed UDP datagram should be counted and rejected"
        );
    }
    #[test]
    fn gesture_scalars_are_addressable_over_osc_without_a_new_wire_surface() {
        // The gesture surface adds four entries to the one closed
        // cross-protocol vocabulary, so both the OSC path grammar and the MIDI
        // profile document address them with no protocol-specific code.
        let parameters = [
            (ControlParameter::GestureX, "gesture_x"),
            (ControlParameter::GestureY, "gesture_y"),
            (ControlParameter::GesturePressure, "gesture_pressure"),
            (ControlParameter::GestureContact, "gesture_contact"),
        ];
        for (parameter, key) in parameters {
            assert_eq!(parameter.key(), key);
            assert_eq!(ControlParameter::parse(key), Some(parameter));
            // The saved profile document uses the same token, so a MIDI
            // binding reaches the identical adapter.
            assert_eq!(
                serde_json::to_value(parameter).unwrap(),
                serde_json::Value::String(key.to_string())
            );

            let address = RuntimeControlAddress::Master(parameter);
            let encoded = format_control_address(address).unwrap();
            assert_eq!(encoded, format!("/collide/v1/master/{key}"));
            assert_eq!(parse_control_address(&encoded), Ok(address));

            let bytes = encode_feedback(address, 0.5).unwrap();
            let decoded = decode_packet(&bytes, peer()).unwrap();
            assert_eq!(decoded.len(), 1);
            assert_eq!(decoded[0].address, address);
            assert_eq!(decoded[0].value, AutomationValue::Absolute(0.5));
        }

        // A neighbouring token stays outside the vocabulary rather than
        // resolving to one of the four.
        assert_eq!(ControlParameter::parse("gesture"), None);
        assert_eq!(ControlParameter::parse("gesture_z"), None);
        assert_eq!(
            parse_control_address("/collide/v1/master/gesture_stroke"),
            Err(OscError::Parameter)
        );
    }
}
