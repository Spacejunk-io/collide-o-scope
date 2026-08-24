//! Bounded bidirectional MIDI runtime.
//!
//! A midir input connection parses Control Change messages on any channel
//! into a table of 128 normalized CC values (lock-free atomics — the MIDI
//! callback runs on the driver's thread). The app reads the four bound
//! slot values each frame and pushes them into the `ModMatrix`, exactly
//! like the audio analyzer does.
//!
//! MIDI learn: the callback also records the last CC number seen; while a
//! slot is armed, the render loop takes that number and binds the slot to
//! it — twist the knob you want, and it's yours.

#![allow(
    dead_code,
    reason = "M5 profile/runtime seams are consumed by Main and Web after API freeze"
)]

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use midir::{Ignore, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};

use crate::action_correlation::{
    ActionEnvelope, ActionIdentity, ActionSequencer, ActionSourceClass,
};
use crate::controller_profile::{
    is_well_formed_controller_midi, AutomationOrigin, ControllerDecoder, ControllerEvent,
    ControllerProfileError, MidiDeviceSelector, ResolvedControllerProfile, RuntimeControlAddress,
};

pub const MIDI_RAW_QUEUE_CAPACITY: usize = 1_024;
pub const MIDI_EVENT_DRAIN_CAPACITY: usize = 4_096;
pub const MIDI_FEEDBACK_KEYS_CAPACITY: usize = 256;
pub const MIDI_FEEDBACK_RATE_PER_SECOND: u32 = 120;
pub const MIDI_LOOP_SUPPRESSION_MILLIS: u64 = 80;
const MIDI_RESCAN_INTERVAL: Duration = Duration::from_millis(500);
const MIDI_WORKER_TICK: Duration = Duration::from_millis(5);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MidiCounters {
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

#[derive(Default)]
struct MidiAtomicCounters {
    raw_received: AtomicU64,
    malformed: AtomicU64,
    input_queue_dropped: AtomicU64,
    loop_suppressed: AtomicU64,
    feedback_dropped: AtomicU64,
    feedback_sent: AtomicU64,
    feedback_rate_limited: AtomicU64,
    scans: AtomicU64,
    reconnects: AtomicU64,
    disconnects: AtomicU64,
}

impl MidiAtomicCounters {
    fn snapshot(&self, local: MidiCounters) -> MidiCounters {
        MidiCounters {
            raw_received: self
                .raw_received
                .load(Ordering::Relaxed)
                .saturating_add(local.raw_received),
            malformed: self
                .malformed
                .load(Ordering::Relaxed)
                .saturating_add(local.malformed),
            input_queue_dropped: self
                .input_queue_dropped
                .load(Ordering::Relaxed)
                .saturating_add(local.input_queue_dropped),
            decoded_events: local.decoded_events,
            event_queue_dropped: local.event_queue_dropped,
            channel_or_unmapped: local.channel_or_unmapped,
            loop_suppressed: self
                .loop_suppressed
                .load(Ordering::Relaxed)
                .saturating_add(local.loop_suppressed),
            feedback_queued: local.feedback_queued,
            feedback_coalesced: local.feedback_coalesced,
            feedback_dropped: self
                .feedback_dropped
                .load(Ordering::Relaxed)
                .saturating_add(local.feedback_dropped),
            feedback_sent: self
                .feedback_sent
                .load(Ordering::Relaxed)
                .saturating_add(local.feedback_sent),
            feedback_rate_limited: self
                .feedback_rate_limited
                .load(Ordering::Relaxed)
                .saturating_add(local.feedback_rate_limited),
            scans: self
                .scans
                .load(Ordering::Relaxed)
                .saturating_add(local.scans),
            reconnects: self
                .reconnects
                .load(Ordering::Relaxed)
                .saturating_add(local.reconnects),
            disconnects: self
                .disconnects
                .load(Ordering::Relaxed)
                .saturating_add(local.disconnects),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidiConnectionPhase {
    Disabled,
    Scanning,
    WaitingForDevice,
    Connected,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MidiRuntimeSnapshot {
    pub phase: MidiConnectionPhase,
    pub input_port: Option<String>,
    pub output_port: Option<String>,
    pub available_inputs: Vec<String>,
    pub available_outputs: Vec<String>,
    pub error: Option<String>,
    pub counters: MidiCounters,
}

impl Default for MidiRuntimeSnapshot {
    fn default() -> Self {
        Self {
            phase: MidiConnectionPhase::Disabled,
            input_port: None,
            output_port: None,
            available_inputs: Vec::new(),
            available_outputs: Vec::new(),
            error: None,
            counters: MidiCounters::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RawMidiMessage {
    timestamp_us: u64,
    bytes: [u8; 3],
    len: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MidiClockUpdate {
    Pulse { timestamp_us: u64 },
    Start,
}

/// One driver-minted raw action after bounded decoding. Compatibility CC
/// state remains inert inside this value until Main applies it at the same
/// correlated lifecycle seam as the typed controller events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MidiActionBatch {
    events: std::ops::Range<usize>,
    compatibility_cc: Option<(u8, u8)>,
    clock_update: Option<MidiClockUpdate>,
    decoder_state_changed: bool,
}

impl MidiActionBatch {
    pub fn event_range(&self) -> std::ops::Range<usize> {
        self.events.clone()
    }

    pub fn decoder_state_changed(&self) -> bool {
        self.decoder_state_changed
    }
}

#[derive(Debug, Clone, Copy)]
struct LoopFingerprint {
    bytes: [u8; 3],
    sent_at: Instant,
}

#[derive(Debug, Clone, Copy)]
struct MidiFeedbackRateGate {
    last: Instant,
    interval: Duration,
}

impl MidiFeedbackRateGate {
    fn ready(now: Instant) -> Self {
        let interval = Duration::from_secs_f64(1.0 / MIDI_FEEDBACK_RATE_PER_SECOND as f64);
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

struct MidiActionQueues {
    admitted: VecDeque<ActionEnvelope<RawMidiMessage>>,
    displaced: VecDeque<ActionIdentity>,
}

impl MidiActionQueues {
    fn new() -> Self {
        Self {
            admitted: VecDeque::with_capacity(MIDI_RAW_QUEUE_CAPACITY),
            displaced: VecDeque::with_capacity(MIDI_RAW_QUEUE_CAPACITY),
        }
    }

    fn unresolved_len(&self) -> usize {
        self.admitted.len().saturating_add(self.displaced.len())
    }

    fn displace_admitted(&mut self) {
        debug_assert!(self.unresolved_len() <= MIDI_RAW_QUEUE_CAPACITY);
        while let Some(action) = self.admitted.pop_front() {
            self.displaced.push_back(action.identity());
        }
        debug_assert!(self.unresolved_len() <= MIDI_RAW_QUEUE_CAPACITY);
    }

    fn drain_displaced(&mut self, output: &mut [Option<ActionIdentity>]) -> usize {
        let count = output.len().min(self.displaced.len());
        for slot in output.iter_mut().take(count) {
            *slot = self.displaced.pop_front();
        }
        count
    }
}

struct MidiShared {
    /// CC value table: f32 bits, 0..1, indexed by CC number.
    cc_values: [AtomicU32; 128],
    /// Last CC number seen + 1 (0 = none). Consumed by MIDI learn.
    last_cc: AtomicU32,
    /// Timing clock (0xF8) pulses since the last Start (24 per quarter note).
    clock_pulses: std::sync::atomic::AtomicU64,
    /// EMA of the pulse interval in microseconds (f64 bits; 0 = no estimate).
    clock_interval_us: std::sync::atomic::AtomicU64,
    /// Driver timestamp of the previous pulse, microseconds.
    clock_prev_ts: std::sync::atomic::AtomicU64,
    action_queues: Mutex<MidiActionQueues>,
    recent_feedback: Mutex<VecDeque<LoopFingerprint>>,
    feedback: Mutex<BTreeMap<(u8, u8), [u8; 3]>>,
    counters: MidiAtomicCounters,
    action_sequencer: Arc<ActionSequencer>,
}

impl MidiShared {
    fn new(action_sequencer: Arc<ActionSequencer>) -> Self {
        Self {
            cc_values: std::array::from_fn(|_| AtomicU32::new(0)),
            last_cc: AtomicU32::new(0),
            clock_pulses: std::sync::atomic::AtomicU64::new(0),
            clock_interval_us: std::sync::atomic::AtomicU64::new(0),
            clock_prev_ts: std::sync::atomic::AtomicU64::new(0),
            action_queues: Mutex::new(MidiActionQueues::new()),
            recent_feedback: Mutex::new(VecDeque::with_capacity(MIDI_FEEDBACK_KEYS_CAPACITY)),
            feedback: Mutex::new(BTreeMap::new()),
            counters: MidiAtomicCounters::default(),
            action_sequencer,
        }
    }

    /// Parse one complete MIDI message from the input callback.
    ///
    /// MIDI Learn deliberately observes Control Change messages only. Notes,
    /// pitch bend, aftertouch, and transport messages must never overwrite an
    /// armed CC binding.
    fn handle_message(&self, timestamp: u64, message: &[u8]) {
        self.handle_message_at(Instant::now(), timestamp, message);
    }

    fn handle_message_at(&self, ingress: Instant, timestamp: u64, message: &[u8]) {
        self.counters.raw_received.fetch_add(1, Ordering::Relaxed);
        if !is_well_formed_controller_midi(message) {
            self.counters.malformed.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let expected = message.len();
        let mut bytes = [0_u8; 3];
        bytes[..expected].copy_from_slice(message);
        if self.is_feedback_echo(bytes) {
            self.counters
                .loop_suppressed
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        let raw = RawMidiMessage {
            timestamp_us: timestamp,
            bytes,
            len: expected as u8,
        };
        let Ok(mut queues) = self.action_queues.try_lock() else {
            self.counters
                .input_queue_dropped
                .fetch_add(1, Ordering::Relaxed);
            return;
        };
        // Admitted messages and identities displaced by a lifecycle barrier
        // share the original fixed 1,024-action ceiling. Until Main consumes
        // displaced identities, new callback work is refused before minting;
        // no reconfiguration can overwrite correlation evidence or grow a
        // second unbounded queue.
        if queues.unresolved_len() >= MIDI_RAW_QUEUE_CAPACITY {
            self.counters
                .input_queue_dropped
                .fetch_add(1, Ordering::Relaxed);
        } else {
            queues.admitted.push_back(self.action_sequencer.envelope_at(
                ActionSourceClass::Midi,
                ingress,
                raw,
            ));
        }
    }

    fn is_feedback_echo(&self, bytes: [u8; 3]) -> bool {
        let Ok(mut recent) = self.recent_feedback.try_lock() else {
            return false;
        };
        let now = Instant::now();
        let window = Duration::from_millis(MIDI_LOOP_SUPPRESSION_MILLIS);
        while recent
            .front()
            .is_some_and(|entry| now.duration_since(entry.sent_at) > window)
        {
            recent.pop_front();
        }
        recent.iter().any(|entry| entry.bytes == bytes)
    }

    fn clear_protocol_queues(&self) {
        // Lifecycle barriers must retain every already-minted identity for
        // Main to terminalize as Superseded. This transfer is FIFO and cannot
        // exceed MIDI_RAW_QUEUE_CAPACITY because callback admission accounts
        // for both halves of the fixed action queue.
        self.action_queues
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .displace_admitted();
        if let Ok(mut feedback) = self.feedback.lock() {
            feedback.clear();
        }
        if let Ok(mut recent) = self.recent_feedback.lock() {
            recent.clear();
        }
    }

    fn drain_displaced_action_identities(&self, output: &mut [Option<ActionIdentity>]) -> usize {
        self.action_queues
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain_displaced(output)
    }

    fn displaced_action_identity_count(&self) -> usize {
        self.action_queues
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .displaced
            .len()
    }
}

struct MidiWorker {
    stop: Arc<AtomicBool>,
    wake_rescan: Arc<AtomicBool>,
    status: Arc<Mutex<MidiRuntimeSnapshot>>,
    join: Option<JoinHandle<()>>,
}

impl MidiWorker {
    fn stop(mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub struct MidiEngine {
    shared: Arc<MidiShared>,
    worker: Option<MidiWorker>,
    decoder: ControllerDecoder,
    local_counters: MidiCounters,
    pub port_name: String,
    pub error: String,
    /// Pulse count at the last `clock_state` call, to detect activity.
    seen_pulses: u64,
    /// When the pulse count last advanced.
    last_advance: std::time::Instant,
}

impl MidiEngine {
    pub fn new() -> Self {
        Self::with_action_sequencer(Arc::new(ActionSequencer::default()))
    }

    pub fn with_action_sequencer(action_sequencer: Arc<ActionSequencer>) -> Self {
        Self {
            shared: Arc::new(MidiShared::new(action_sequencer)),
            worker: None,
            decoder: ControllerDecoder::new(ResolvedControllerProfile::legacy_four_cc()),
            local_counters: MidiCounters::default(),
            port_name: String::new(),
            error: String::new(),
            seen_pulses: 0,
            last_advance: std::time::Instant::now(),
        }
    }

    pub fn is_running(&self) -> bool {
        self.worker.is_some()
    }

    /// Connect to the first available MIDI input port. Failure is soft:
    /// the error is recorded for the UI and slot values read 0.
    pub fn start(&mut self) {
        if self.worker.is_some() {
            return;
        }
        self.error.clear();
        let stop = Arc::new(AtomicBool::new(false));
        let wake_rescan = Arc::new(AtomicBool::new(true));
        let status = Arc::new(Mutex::new(MidiRuntimeSnapshot {
            phase: MidiConnectionPhase::Scanning,
            ..MidiRuntimeSnapshot::default()
        }));
        let shared = self.shared.clone();
        let worker_stop = stop.clone();
        let worker_rescan = wake_rescan.clone();
        let worker_status = status.clone();
        let input_selector = self.decoder.profile().input.clone();
        let output_selector = self.decoder.profile().output.clone();
        let join = std::thread::Builder::new()
            .name("collide-midi-supervisor".into())
            .spawn(move || {
                midi_worker(
                    shared,
                    worker_stop,
                    worker_rescan,
                    worker_status,
                    input_selector,
                    output_selector,
                );
            });
        match join {
            Ok(join) => {
                self.worker = Some(MidiWorker {
                    stop,
                    wake_rescan,
                    status,
                    join: Some(join),
                });
            }
            Err(error) => {
                self.error = format!("midi worker: {error}");
            }
        }
    }

    pub fn stop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.stop();
        }
        for v in self.shared.cc_values.iter() {
            v.store(0, Ordering::Relaxed);
        }
        self.shared.last_cc.store(0, Ordering::Relaxed);
        self.shared.clock_pulses.store(0, Ordering::Relaxed);
        self.shared.clock_interval_us.store(0, Ordering::Relaxed);
        self.shared.clock_prev_ts.store(0, Ordering::Relaxed);
        self.seen_pulses = 0;
        self.shared.clear_protocol_queues();
    }

    /// External clock state, if a MIDI clock is actively pulsing:
    /// (bpm, beat position in quarter notes). Call once per frame — activity
    /// is judged by whether the pulse count advanced within the last second.
    pub fn clock_state(&mut self) -> Option<(f32, f64)> {
        self.clock_state_at(std::time::Instant::now())
    }

    fn clock_state_at(&mut self, now: std::time::Instant) -> Option<(f32, f64)> {
        let pulses = self.shared.clock_pulses.load(Ordering::Relaxed);
        if pulses != self.seen_pulses {
            self.seen_pulses = pulses;
            self.last_advance = now;
        }
        if pulses == 0 || now.duration_since(self.last_advance).as_secs_f32() > 1.0 {
            return None;
        }
        let interval_us = f64::from_bits(self.shared.clock_interval_us.load(Ordering::Relaxed));
        if interval_us <= 0.0 {
            return None;
        }
        let bpm = (60_000_000.0 / (interval_us * 24.0)) as f32;
        let beat = pulses as f64 / 24.0;
        Some((bpm.clamp(30.0, 300.0), beat))
    }

    /// Current normalized value of a CC number.
    pub fn cc_value(&self, cc: u8) -> f32 {
        f32::from_bits(self.shared.cc_values[(cc & 0x7F) as usize].load(Ordering::Relaxed))
    }

    /// Take the last CC number seen (for MIDI learn). Consuming read.
    pub fn take_last_cc(&self) -> Option<u8> {
        let v = self.shared.last_cc.swap(0, Ordering::Relaxed);
        if v == 0 {
            None
        } else {
            Some((v - 1) as u8)
        }
    }

    #[cfg(test)]
    pub fn inject_message_for_test(&self, ingress: Instant, timestamp: u64, message: &[u8]) {
        self.shared.handle_message_at(ingress, timestamp, message);
    }

    /// Apply legacy-slot/MIDI-learn compatibility state exactly once, after
    /// dequeue and under the batch's original ActionIdentity. Returning false
    /// means the raw CC was an exact compatibility no-op; typed decoded events
    /// may still independently apply under the same identity.
    pub fn apply_correlated_compatibility(&self, batch: &MidiActionBatch) -> bool {
        let Some((cc, raw_value)) = batch.compatibility_cc else {
            return false;
        };
        let value_bits = (f32::from(raw_value) / 127.0).to_bits();
        let prior_value =
            self.shared.cc_values[usize::from(cc)].swap(value_bits, Ordering::Relaxed);
        let prior_last = self
            .shared
            .last_cc
            .swap(u32::from(cc) + 1, Ordering::Relaxed);
        prior_value != value_bits || prior_last != u32::from(cc) + 1
    }

    /// Apply transport-clock state only after Main dequeues the action under
    /// its original identity. The driver callback merely admits a fixed raw
    /// message; no rendered clock fact can change before this seam.
    pub fn apply_correlated_clock(&self, batch: &MidiActionBatch) -> bool {
        match batch.clock_update {
            Some(MidiClockUpdate::Pulse { timestamp_us }) => {
                let prev = self
                    .shared
                    .clock_prev_ts
                    .swap(timestamp_us, Ordering::Relaxed);
                if prev > 0 && timestamp_us > prev {
                    let dt = (timestamp_us - prev) as f64;
                    if (2_000.0..=100_000.0).contains(&dt) {
                        let old =
                            f64::from_bits(self.shared.clock_interval_us.load(Ordering::Relaxed));
                        let ema = if old > 0.0 { old * 0.9 + dt * 0.1 } else { dt };
                        self.shared
                            .clock_interval_us
                            .store(ema.to_bits(), Ordering::Relaxed);
                    }
                }
                self.shared.clock_pulses.fetch_add(1, Ordering::Relaxed);
                true
            }
            Some(MidiClockUpdate::Start) => {
                let pulses = self.shared.clock_pulses.swap(0, Ordering::Relaxed);
                let previous = self.shared.clock_prev_ts.swap(0, Ordering::Relaxed);
                pulses != 0 || previous != 0
            }
            None => false,
        }
    }

    pub fn apply_profile(
        &mut self,
        profile: ResolvedControllerProfile,
    ) -> Result<(), ControllerProfileError> {
        profile.validate()?;
        let restart = self.worker.is_some();
        if restart {
            self.stop();
        }
        self.shared.clear_protocol_queues();
        self.decoder.replace_profile(profile);
        if restart {
            self.start();
        }
        Ok(())
    }

    pub fn resolved_profile(&self) -> &ResolvedControllerProfile {
        self.decoder.profile()
    }

    /// Drain identities displaced by `stop` or `apply_profile` into caller-owned
    /// fixed storage, in original ingress order.
    ///
    /// Only the returned prefix is written. If `output` is smaller than the
    /// pending set, the remainder stays queued and is returned by a later call.
    /// Main must terminalize every returned identity as `Superseded`. The
    /// admitted and displaced halves share `MIDI_RAW_QUEUE_CAPACITY`, so this
    /// evidence path cannot grow beyond the pre-existing input ceiling.
    pub fn drain_displaced_action_identities(
        &self,
        output: &mut [Option<ActionIdentity>],
    ) -> usize {
        self.shared.drain_displaced_action_identities(output)
    }

    pub fn displaced_action_identity_count(&self) -> usize {
        self.shared.displaced_action_identity_count()
    }

    pub fn request_rescan(&self) {
        if let Some(worker) = &self.worker {
            worker.wake_rescan.store(true, Ordering::Release);
        }
    }

    /// Decode bounded raw input while retaining the one envelope minted by
    /// the driver callback for each admitted MIDI message. A range may be
    /// empty when a well-formed but unmapped message is explicitly refused by
    /// the application; fan-out remains one correlated transport action.
    pub fn drain_correlated_events(
        &mut self,
        output: &mut Vec<ControllerEvent>,
        batches: &mut Vec<ActionEnvelope<MidiActionBatch>>,
    ) {
        let before = output.len();
        let output_limit = before.saturating_add(MIDI_EVENT_DRAIN_CAPACITY);
        let mut raw = VecDeque::new();
        if let Ok(mut queues) = self.shared.action_queues.try_lock() {
            std::mem::swap(&mut queues.admitted, &mut raw);
        }
        for message in raw {
            let event_start = output.len();
            let report = self.decoder.decode_bounded(
                message.payload().timestamp_us,
                &message.payload().bytes[..usize::from(message.payload().len)],
                output,
                output_limit,
            );
            self.local_counters.event_queue_dropped = self
                .local_counters
                .event_queue_dropped
                .saturating_add(report.dropped_events as u64);
            if report.matched_bindings == 0 && message.payload().bytes[0] < 0xf0 {
                self.local_counters.channel_or_unmapped =
                    self.local_counters.channel_or_unmapped.saturating_add(1);
            }
            let event_end = output.len();
            let compatibility_cc = (message.payload().bytes[0] & 0xf0 == 0xb0)
                .then(|| (message.payload().bytes[1], message.payload().bytes[2]));
            let clock_update = match message.payload().bytes[0] {
                0xf8 => Some(MidiClockUpdate::Pulse {
                    timestamp_us: message.payload().timestamp_us,
                }),
                0xfa => Some(MidiClockUpdate::Start),
                _ => None,
            };
            batches.push(message.map(|_| MidiActionBatch {
                events: event_start..event_end,
                compatibility_cc,
                clock_update,
                decoder_state_changed: report.state_changed,
            }));
        }
        self.local_counters.decoded_events = self
            .local_counters
            .decoded_events
            .saturating_add((output.len() - before) as u64);
    }

    #[cfg(test)]
    pub fn drain_events(&mut self, output: &mut Vec<ControllerEvent>) {
        let mut batches = Vec::new();
        self.drain_correlated_events(output, &mut batches);
    }

    pub fn queue_feedback(
        &mut self,
        address: RuntimeControlAddress,
        value: f32,
        origin: AutomationOrigin,
    ) {
        if matches!(origin, AutomationOrigin::Midi) || !value.is_finite() {
            self.local_counters.loop_suppressed =
                self.local_counters.loop_suppressed.saturating_add(1);
            return;
        }
        for (bytes, _) in self.decoder.profile().feedback_bytes(address, value) {
            let Ok(mut feedback) = self.shared.feedback.try_lock() else {
                self.local_counters.feedback_dropped =
                    self.local_counters.feedback_dropped.saturating_add(1);
                continue;
            };
            let key = (bytes[0], bytes[1]);
            if feedback.contains_key(&key) {
                self.local_counters.feedback_coalesced =
                    self.local_counters.feedback_coalesced.saturating_add(1);
            } else if feedback.len() >= MIDI_FEEDBACK_KEYS_CAPACITY {
                self.local_counters.feedback_dropped =
                    self.local_counters.feedback_dropped.saturating_add(1);
                continue;
            }
            feedback.insert(key, bytes);
            self.local_counters.feedback_queued =
                self.local_counters.feedback_queued.saturating_add(1);
        }
    }

    pub fn runtime_snapshot(&self) -> MidiRuntimeSnapshot {
        let mut snapshot = self
            .worker
            .as_ref()
            .and_then(|worker| worker.status.lock().ok().map(|value| value.clone()))
            .unwrap_or_default();
        snapshot.counters = self.shared.counters.snapshot(self.local_counters);
        snapshot
    }
}

impl Drop for MidiEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

fn midi_worker(
    shared: Arc<MidiShared>,
    stop: Arc<AtomicBool>,
    wake_rescan: Arc<AtomicBool>,
    status: Arc<Mutex<MidiRuntimeSnapshot>>,
    input_selector: MidiDeviceSelector,
    output_selector: MidiDeviceSelector,
) {
    let mut input: Option<(String, MidiInputConnection<()>)> = None;
    let mut output: Option<(String, MidiOutputConnection)> = None;
    let mut ever_connected = false;
    let mut last_scan = Instant::now() - MIDI_RESCAN_INTERVAL;
    let mut feedback_rate = MidiFeedbackRateGate::ready(Instant::now());
    while !stop.load(Ordering::Acquire) {
        if wake_rescan.swap(false, Ordering::AcqRel) || last_scan.elapsed() >= MIDI_RESCAN_INTERVAL
        {
            last_scan = Instant::now();
            shared.counters.scans.fetch_add(1, Ordering::Relaxed);
            let inputs = midi_input_names();
            let outputs = midi_output_names();
            let selected_input = select_name(&inputs, &input_selector);
            let selected_output = select_name(&outputs, &output_selector);

            if input
                .as_ref()
                .is_some_and(|(name, _)| Some(name.as_str()) != selected_input.as_deref())
            {
                input = None;
                shared.counters.disconnects.fetch_add(1, Ordering::Relaxed);
            }
            if output
                .as_ref()
                .is_some_and(|(name, _)| Some(name.as_str()) != selected_output.as_deref())
            {
                output = None;
                shared.counters.disconnects.fetch_add(1, Ordering::Relaxed);
            }
            let mut error = None;
            if input.is_none() {
                if let Some(name) = selected_input.as_deref() {
                    match connect_input(name, shared.clone()) {
                        Ok(connection) => {
                            if ever_connected {
                                shared.counters.reconnects.fetch_add(1, Ordering::Relaxed);
                            }
                            ever_connected = true;
                            input = Some((name.to_string(), connection));
                        }
                        Err(message) => error = Some(message),
                    }
                }
            }
            if output.is_none() {
                if let Some(name) = selected_output.as_deref() {
                    match connect_output(name) {
                        Ok(connection) => output = Some((name.to_string(), connection)),
                        Err(message) => error = Some(message),
                    }
                }
            }
            if let Ok(mut current) = status.lock() {
                current.phase = if error.is_some() {
                    MidiConnectionPhase::Error
                } else if input.is_some() {
                    MidiConnectionPhase::Connected
                } else {
                    MidiConnectionPhase::WaitingForDevice
                };
                current.input_port = input.as_ref().map(|(name, _)| name.clone());
                current.output_port = output.as_ref().map(|(name, _)| name.clone());
                current.available_inputs = inputs;
                current.available_outputs = outputs;
                current.error = error;
            }
        }

        let feedback_pending = shared
            .feedback
            .try_lock()
            .is_ok_and(|feedback| !feedback.is_empty());
        if feedback_pending && output.is_some() && feedback_rate.admit(Instant::now()) {
            let next = shared
                .feedback
                .try_lock()
                .ok()
                .and_then(|mut pending| pending.pop_first());
            if let Some((key, bytes)) = next {
                if let Some((_, connection)) = &mut output {
                    match connection.send(&bytes) {
                        Ok(()) => {
                            shared
                                .counters
                                .feedback_sent
                                .fetch_add(1, Ordering::Relaxed);
                            if let Ok(mut recent) = shared.recent_feedback.try_lock() {
                                if recent.len() == MIDI_FEEDBACK_KEYS_CAPACITY {
                                    recent.pop_front();
                                }
                                recent.push_back(LoopFingerprint {
                                    bytes,
                                    sent_at: Instant::now(),
                                });
                            }
                        }
                        Err(error) => {
                            let restored = shared.feedback.try_lock().is_ok_and(|mut pending| {
                                if pending.len() >= MIDI_FEEDBACK_KEYS_CAPACITY {
                                    false
                                } else {
                                    pending.insert(key, bytes);
                                    true
                                }
                            });
                            if !restored {
                                shared
                                    .counters
                                    .feedback_dropped
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                            output = None;
                            shared.counters.disconnects.fetch_add(1, Ordering::Relaxed);
                            if let Ok(mut current) = status.lock() {
                                current.phase = MidiConnectionPhase::Error;
                                current.error = Some(format!("MIDI feedback send: {error}"));
                            }
                        }
                    }
                }
            }
        } else if feedback_pending && output.is_some() {
            shared
                .counters
                .feedback_rate_limited
                .fetch_add(1, Ordering::Relaxed);
        }
        std::thread::sleep(MIDI_WORKER_TICK);
    }
    if let Ok(mut current) = status.lock() {
        current.phase = MidiConnectionPhase::Disabled;
        current.input_port = None;
        current.output_port = None;
    }
}

fn bounded_device_name(value: String) -> String {
    let mut bounded = String::new();
    for character in value.chars().filter(|character| !character.is_control()) {
        if bounded.len().saturating_add(character.len_utf8())
            > crate::controller_profile::CONTROLLER_DEVICE_NAME_MAX_BYTES
        {
            break;
        }
        bounded.push(character);
    }
    bounded
}

fn midi_input_names() -> Vec<String> {
    let Ok(input) = MidiInput::new("collide-o-scope-scan-input") else {
        return Vec::new();
    };
    input
        .ports()
        .into_iter()
        .take(128)
        .filter_map(|port| input.port_name(&port).ok())
        .map(bounded_device_name)
        .collect()
}

fn midi_output_names() -> Vec<String> {
    let Ok(output) = MidiOutput::new("collide-o-scope-scan-output") else {
        return Vec::new();
    };
    output
        .ports()
        .into_iter()
        .take(128)
        .filter_map(|port| output.port_name(&port).ok())
        .map(bounded_device_name)
        .collect()
}

fn select_name(names: &[String], selector: &MidiDeviceSelector) -> Option<String> {
    names.iter().find(|name| selector.matches(name)).cloned()
}

fn connect_input(
    selected_name: &str,
    shared: Arc<MidiShared>,
) -> Result<MidiInputConnection<()>, String> {
    let mut input = MidiInput::new("collide-o-scope").map_err(|error| error.to_string())?;
    input.ignore(Ignore::None);
    let port = input
        .ports()
        .into_iter()
        .find(|port| input.port_name(port).ok().as_deref() == Some(selected_name))
        .ok_or_else(|| "selected MIDI input disappeared".to_string())?;
    input
        .connect(
            &port,
            "collide-in",
            move |timestamp, message, ()| shared.handle_message(timestamp, message),
            (),
        )
        .map_err(|error| error.to_string())
}

fn connect_output(selected_name: &str) -> Result<MidiOutputConnection, String> {
    let output = MidiOutput::new("collide-o-scope-output").map_err(|error| error.to_string())?;
    let port = output
        .ports()
        .into_iter()
        .find(|port| output.port_name(port).ok().as_deref() == Some(selected_name))
        .ok_or_else(|| "selected MIDI output disappeared".to_string())?;
    output
        .connect(&port, "collide-out")
        .map_err(|error| error.to_string())
}

/// Pure hotplug transition law documenting the worker's reconnect behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MidiHotplugState {
    pub connected: bool,
    pub reconnects: u64,
    pub disconnects: u64,
}

impl MidiHotplugState {
    pub fn observe(&mut self, selected_device_present: bool) -> MidiConnectionPhase {
        match (self.connected, selected_device_present) {
            (false, true) => {
                if self.disconnects > 0 {
                    self.reconnects = self.reconnects.saturating_add(1);
                }
                self.connected = true;
                MidiConnectionPhase::Connected
            }
            (true, false) => {
                self.connected = false;
                self.disconnects = self.disconnects.saturating_add(1);
                MidiConnectionPhase::WaitingForDevice
            }
            (true, true) => MidiConnectionPhase::Connected,
            (false, false) => MidiConnectionPhase::WaitingForDevice,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller_profile::{
        ControlParameter, MidiChannelFilter, MidiInputSource, MidiOutputMessage, MidiValueEncoding,
        ResolvedControllerBinding,
    };
    use std::time::{Duration, Instant};

    const PULSE_US_120_BPM: u64 = 20_833;

    fn feed_120_bpm_quarter(engine: &mut MidiEngine) {
        let first_timestamp = 1_000_000;
        for pulse in 0..24 {
            engine
                .shared
                .handle_message(first_timestamp + pulse * PULSE_US_120_BPM, &[0xF8]);
        }
        assert_eq!(engine.shared.clock_pulses.load(Ordering::Relaxed), 0);
        assert_eq!(engine.shared.clock_prev_ts.load(Ordering::Relaxed), 0);
        let mut events = Vec::new();
        let mut batches = Vec::new();
        engine.drain_correlated_events(&mut events, &mut batches);
        assert_eq!(batches.len(), 24);
        assert!(batches
            .iter()
            .all(|batch| engine.apply_correlated_clock(batch.payload())));
    }

    #[test]
    fn learn_accepts_control_change_on_any_channel() {
        let mut engine = MidiEngine::new();

        engine.shared.handle_message(0, &[0xB7, 74, 100]);
        assert_eq!(engine.take_last_cc(), None);
        assert_eq!(engine.cc_value(74), 0.0);
        let mut events = Vec::new();
        let mut batches = Vec::new();
        engine.drain_correlated_events(&mut events, &mut batches);
        assert_eq!(batches.len(), 1);
        assert!(engine.apply_correlated_compatibility(batches[0].payload()));
        assert_eq!(engine.take_last_cc(), Some(74));
        assert!((engine.cc_value(74) - 100.0 / 127.0).abs() < f32::EPSILON);
    }

    #[test]
    fn malformed_messages_are_counted_once_without_protocol_mutation() {
        let engine = MidiEngine::new();
        let malformed: &[&[u8]] = &[
            &[],
            &[0xb0, 74],
            &[0xb0, 74, 100, 0],
            &[0xb0, 0xca, 100],
            &[0xb0, 74, 0xe4],
            &[0x90, 0xbc, 100],
            &[0x80, 60, 0x80],
            &[0xf8, 0],
            &[0xfa, 0],
            &[0xfb, 0],
            &[0xfc, 0],
            &[0xf0],
            &[74, 100],
        ];
        for (timestamp, message) in malformed.iter().enumerate() {
            engine.shared.handle_message(timestamp as u64, message);
        }

        assert_eq!(
            engine.shared.counters.raw_received.load(Ordering::Relaxed),
            malformed.len() as u64
        );
        assert_eq!(
            engine.shared.counters.malformed.load(Ordering::Relaxed),
            malformed.len() as u64
        );
        assert_eq!(engine.shared.last_cc.load(Ordering::Relaxed), 0);
        assert_eq!(engine.shared.clock_pulses.load(Ordering::Relaxed), 0);
        assert_eq!(engine.shared.clock_prev_ts.load(Ordering::Relaxed), 0);
        assert_eq!(engine.cc_value(74), 0.0);
        assert!(engine
            .shared
            .action_queues
            .lock()
            .unwrap()
            .admitted
            .is_empty());
    }

    #[test]
    fn learn_ignores_notes_and_keys() {
        let engine = MidiEngine::new();

        engine.shared.handle_message(0, &[0x90, 60, 127]);
        engine.shared.handle_message(1, &[0x80, 60, 0]);
        engine.shared.handle_message(2, &[0xA4, 60, 64]);

        assert_eq!(engine.take_last_cc(), None);
    }

    #[test]
    fn timing_clock_uses_24_ppqn_for_bpm_and_beat() {
        let mut engine = MidiEngine::new();
        let sampled_at = Instant::now();
        engine.last_advance = sampled_at;
        feed_120_bpm_quarter(&mut engine);

        let (bpm, beat) = engine
            .clock_state_at(sampled_at)
            .expect("24 timing pulses should establish an external clock");

        assert!((bpm - 120.0).abs() < 0.01, "estimated BPM was {bpm}");
        assert!((beat - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn midi_start_resets_clock_position() {
        let mut engine = MidiEngine::new();
        let sampled_at = Instant::now();
        engine.last_advance = sampled_at;
        feed_120_bpm_quarter(&mut engine);
        assert!(engine.clock_state_at(sampled_at).is_some());

        engine.shared.handle_message(2_000_000, &[0xFA]);
        assert_ne!(engine.shared.clock_pulses.load(Ordering::Relaxed), 0);
        let mut events = Vec::new();
        let mut batches = Vec::new();
        engine.drain_correlated_events(&mut events, &mut batches);
        assert_eq!(batches.len(), 1);
        assert!(engine.apply_correlated_clock(batches[0].payload()));
        assert_eq!(engine.shared.clock_pulses.load(Ordering::Relaxed), 0);
        assert_eq!(engine.shared.clock_prev_ts.load(Ordering::Relaxed), 0);
        assert_eq!(engine.clock_state_at(sampled_at), None);
    }

    #[test]
    fn external_clock_remains_active_for_one_second_then_falls_back_continuously() {
        let mut clock = crate::modulation::Clock::new();
        let mut engine = MidiEngine::new();
        let sampled_at = Instant::now();
        engine.last_advance = sampled_at;
        feed_120_bpm_quarter(&mut engine);
        let active = engine
            .clock_state_at(sampled_at)
            .expect("clock should become active");

        assert_eq!(
            engine.clock_state_at(sampled_at + Duration::from_secs(1)),
            Some(active)
        );
        assert_eq!(
            engine.clock_state_at(sampled_at + Duration::from_millis(1_001)),
            None
        );

        let fallback_at = sampled_at + Duration::from_millis(1_001);
        clock.set_bpm_at(active.0, sampled_at);
        clock.set_external_beat(Some(active.1), sampled_at);
        let before_fallback = clock.beat(fallback_at);
        clock.set_external_beat(None, fallback_at);
        assert!((clock.beat(fallback_at) - before_fallback).abs() < 1.0e-6);
        assert!(clock.beat(fallback_at + Duration::from_millis(10)) > before_fallback);
    }

    #[test]
    fn exact_channel_filter_rejects_other_channels_while_omni_legacy_still_works() {
        let binding = ResolvedControllerBinding {
            id: 9,
            source: MidiInputSource::ControlChange { controller: 74 },
            channel: MidiChannelFilter::Exact { channel: 2 },
            encoding: MidiValueEncoding::Absolute,
            button_mode: None,
            press_threshold: 64,
            relative_step: 1.0 / 127.0,
            target: RuntimeControlAddress::Master(ControlParameter::Amount),
            feedback: None,
        };
        let mut decoder = ControllerDecoder::new(ResolvedControllerProfile {
            name: "channel".into(),
            input: MidiDeviceSelector::FirstAvailable,
            output: MidiDeviceSelector::FirstAvailable,
            bindings: vec![binding].into_boxed_slice(),
        });
        let mut events = Vec::new();
        decoder.decode(0, &[0xb0, 74, 127], &mut events);
        decoder.decode(1, &[0xb1, 74, 127], &mut events);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].timestamp_us, 1);
    }

    #[test]
    fn callback_queue_is_bounded_and_reports_saturation_without_blocking() {
        let engine = MidiEngine::new();
        for timestamp in 0..MIDI_RAW_QUEUE_CAPACITY as u64 + 17 {
            engine.shared.handle_message(timestamp, &[0xb0, 1, 64]);
        }
        assert_eq!(
            engine.shared.action_queues.lock().unwrap().admitted.len(),
            MIDI_RAW_QUEUE_CAPACITY
        );
        assert_eq!(
            engine
                .shared
                .counters
                .input_queue_dropped
                .load(Ordering::Relaxed),
            17
        );
    }

    #[test]
    fn callback_mints_once_and_decode_preserves_the_exact_shared_identity() {
        let sequencer = Arc::new(ActionSequencer::default());
        let mut engine = MidiEngine::with_action_sequencer(sequencer.clone());
        let ingress = Instant::now()
            .checked_sub(Duration::from_millis(7))
            .unwrap();

        engine
            .shared
            .handle_message_at(ingress, 123, &[0xb0, 1, 64]);
        let queued_identity = engine
            .shared
            .action_queues
            .lock()
            .unwrap()
            .admitted
            .front()
            .unwrap()
            .identity();
        assert_eq!(queued_identity.sequence().get(), 1);
        assert_eq!(queued_identity.source(), ActionSourceClass::Midi);
        assert_eq!(queued_identity.ingress(), ingress);

        let next_transport = sequencer.envelope(ActionSourceClass::Browser, ());
        assert_eq!(next_transport.sequence().get(), 2);

        let mut events = Vec::new();
        let mut batches = Vec::new();
        engine.drain_correlated_events(&mut events, &mut batches);
        assert_eq!(events.len(), 1);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].identity(), queued_identity);
        assert_eq!(batches[0].payload().event_range(), 0..1);
        assert_eq!(engine.take_last_cc(), None);
        assert_eq!(engine.cc_value(1), 0.0);
        assert!(engine.apply_correlated_compatibility(batches[0].payload()));
        assert_eq!(engine.take_last_cc(), Some(1));
        assert!((engine.cc_value(1) - 64.0 / 127.0).abs() < f32::EPSILON);
    }

    #[test]
    fn stop_and_profile_replacement_return_every_displaced_identity_once_in_fifo_order() {
        let sequencer = Arc::new(ActionSequencer::default());
        let mut engine = MidiEngine::with_action_sequencer(sequencer);
        let origin = Instant::now();
        for offset in 0..3 {
            engine.shared.handle_message_at(
                origin + Duration::from_micros(offset),
                offset,
                &[0xb0, 1, 64],
            );
        }
        let mut expected = engine
            .shared
            .action_queues
            .lock()
            .unwrap()
            .admitted
            .iter()
            .map(ActionEnvelope::identity)
            .collect::<Vec<_>>();

        engine.stop();
        assert_eq!(engine.displaced_action_identity_count(), 3);
        assert!(engine
            .shared
            .action_queues
            .lock()
            .unwrap()
            .admitted
            .is_empty());

        engine
            .shared
            .handle_message_at(origin + Duration::from_micros(3), 3, &[0xb0, 1, 65]);
        expected.push(
            engine
                .shared
                .action_queues
                .lock()
                .unwrap()
                .admitted
                .front()
                .unwrap()
                .identity(),
        );
        engine
            .apply_profile(engine.resolved_profile().clone())
            .unwrap();
        assert_eq!(engine.displaced_action_identity_count(), expected.len());

        let mut returned = Vec::new();
        let mut scratch = [None; 2];
        while engine.displaced_action_identity_count() != 0 {
            let count = engine.drain_displaced_action_identities(&mut scratch);
            assert!(count > 0);
            returned.extend(scratch[..count].iter().copied().flatten());
        }
        assert_eq!(returned, expected);
        assert_eq!(engine.drain_displaced_action_identities(&mut scratch), 0);
        assert_eq!(engine.displaced_action_identity_count(), 0);
    }

    #[test]
    fn hotplug_disconnect_and_reconnect_are_explicit_and_monotonic() {
        let mut state = MidiHotplugState::default();
        assert_eq!(state.observe(false), MidiConnectionPhase::WaitingForDevice);
        assert_eq!(state.observe(true), MidiConnectionPhase::Connected);
        assert_eq!(state.observe(false), MidiConnectionPhase::WaitingForDevice);
        assert_eq!(state.disconnects, 1);
        assert_eq!(state.observe(true), MidiConnectionPhase::Connected);
        assert_eq!(state.reconnects, 1);
        assert_eq!(state.observe(true), MidiConnectionPhase::Connected);
        assert_eq!((state.reconnects, state.disconnects), (1, 1));
    }

    #[test]
    fn feedback_coalesces_by_wire_control_and_suppresses_midi_origin() {
        let address = RuntimeControlAddress::Master(ControlParameter::Amount);
        let profile = ResolvedControllerProfile {
            name: "feedback".into(),
            input: MidiDeviceSelector::FirstAvailable,
            output: MidiDeviceSelector::FirstAvailable,
            bindings: vec![ResolvedControllerBinding {
                id: 1,
                source: MidiInputSource::ControlChange { controller: 1 },
                channel: MidiChannelFilter::Omni,
                encoding: MidiValueEncoding::Absolute,
                button_mode: None,
                press_threshold: 64,
                relative_step: 1.0 / 127.0,
                target: address,
                feedback: Some(MidiOutputMessage::ControlChange {
                    channel: 1,
                    controller: 17,
                }),
            }]
            .into_boxed_slice(),
        };
        let mut engine = MidiEngine::new();
        engine.apply_profile(profile).unwrap();
        engine.queue_feedback(address, 0.2, AutomationOrigin::HostAutomation);
        engine.queue_feedback(address, 0.8, AutomationOrigin::HostAutomation);
        engine.queue_feedback(address, 1.0, AutomationOrigin::Midi);
        let feedback = engine.shared.feedback.lock().unwrap();
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[&(0xb0, 17)], [0xb0, 17, 102]);
        drop(feedback);
        assert_eq!(engine.local_counters.feedback_queued, 2);
        assert_eq!(engine.local_counters.feedback_coalesced, 1);
        assert_eq!(engine.local_counters.loop_suppressed, 1);
    }

    #[test]
    fn resolved_profile_application_is_atomic_and_decoded_fanout_is_bounded() {
        let mut engine = MidiEngine::new();
        let invalid = ResolvedControllerProfile {
            name: "hostile".into(),
            input: MidiDeviceSelector::FirstAvailable,
            output: MidiDeviceSelector::FirstAvailable,
            bindings: vec![ResolvedControllerBinding {
                id: 1,
                source: MidiInputSource::ControlChange { controller: 1 },
                channel: MidiChannelFilter::Omni,
                encoding: MidiValueEncoding::Absolute,
                button_mode: None,
                press_threshold: u8::MAX,
                relative_step: 1.0 / 127.0,
                target: RuntimeControlAddress::Master(ControlParameter::Amount),
                feedback: None,
            }]
            .into_boxed_slice(),
        };
        assert!(matches!(
            engine.apply_profile(invalid),
            Err(ControllerProfileError::PressThreshold(1))
        ));
        assert_eq!(engine.decoder.profile().name, "Legacy four CC");

        engine
            .shared
            .feedback
            .lock()
            .unwrap()
            .insert((0xb0, 1), [0xb0, 1, 64]);
        engine.shared.handle_message(1, &[0xb0, 1, 64]);

        let bindings = (1..=crate::controller_profile::CONTROLLER_PROFILE_MAX_BINDINGS)
            .map(|index| ResolvedControllerBinding {
                id: index as u16,
                source: MidiInputSource::ControlChange { controller: 7 },
                channel: MidiChannelFilter::Omni,
                encoding: MidiValueEncoding::Absolute,
                button_mode: None,
                press_threshold: 64,
                relative_step: 1.0 / 127.0,
                target: RuntimeControlAddress::Master(ControlParameter::Amount),
                feedback: None,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        engine
            .apply_profile(ResolvedControllerProfile {
                name: "bounded fanout".into(),
                input: MidiDeviceSelector::FirstAvailable,
                output: MidiDeviceSelector::FirstAvailable,
                bindings,
            })
            .unwrap();
        assert!(engine.shared.feedback.lock().unwrap().is_empty());
        assert!(engine
            .shared
            .action_queues
            .lock()
            .unwrap()
            .admitted
            .is_empty());
        let displaced = engine.displaced_action_identity_count();
        assert_eq!(displaced, 1);
        let admitted_capacity = MIDI_RAW_QUEUE_CAPACITY - displaced;
        for timestamp in 0..MIDI_RAW_QUEUE_CAPACITY as u64 {
            engine.shared.handle_message(timestamp, &[0xb0, 7, 127]);
        }
        assert_eq!(
            engine.shared.action_queues.lock().unwrap().admitted.len(),
            admitted_capacity
        );
        assert_eq!(
            engine
                .shared
                .counters
                .input_queue_dropped
                .load(Ordering::Relaxed),
            displaced as u64
        );
        let mut events = Vec::new();
        engine.drain_events(&mut events);
        assert_eq!(events.len(), MIDI_EVENT_DRAIN_CAPACITY);
        assert_eq!(
            engine.local_counters.event_queue_dropped,
            (admitted_capacity * crate::controller_profile::CONTROLLER_PROFILE_MAX_BINDINGS
                - MIDI_EVENT_DRAIN_CAPACITY) as u64
        );
    }

    #[test]
    fn device_selection_names_and_feedback_cadence_are_bounded() {
        let names = vec!["A".to_string(), "B".to_string()];
        assert_eq!(
            select_name(&names, &MidiDeviceSelector::FirstAvailable),
            Some("A".to_string())
        );
        assert_eq!(
            select_name(&names, &MidiDeviceSelector::Exact { name: "B".into() }),
            Some("B".to_string())
        );
        assert_eq!(
            select_name(&names, &MidiDeviceSelector::Exact { name: "C".into() }),
            None
        );
        let bounded = bounded_device_name(format!("{}\nignored", "é".repeat(200)));
        assert!(bounded.len() <= crate::controller_profile::CONTROLLER_DEVICE_NAME_MAX_BYTES);
        assert!(!bounded.chars().any(char::is_control));

        let now = Instant::now();
        let mut rate = MidiFeedbackRateGate::ready(now);
        let interval = Duration::from_secs_f64(1.0 / MIDI_FEEDBACK_RATE_PER_SECOND as f64);
        for index in 0..MIDI_FEEDBACK_RATE_PER_SECOND {
            assert!(rate.admit(now + interval * index));
            assert!(!rate.admit(now + interval * index + interval / 2));
        }

        let target = RuntimeControlAddress::Master(ControlParameter::Amount);
        let bindings = (0..MIDI_FEEDBACK_KEYS_CAPACITY)
            .map(|index| ResolvedControllerBinding {
                id: (index + 1) as u16,
                source: MidiInputSource::ControlChange { controller: 1 },
                channel: MidiChannelFilter::Omni,
                encoding: MidiValueEncoding::Absolute,
                button_mode: None,
                press_threshold: 64,
                relative_step: 1.0 / 127.0,
                target,
                feedback: Some(MidiOutputMessage::ControlChange {
                    channel: (index / 128 + 1) as u8,
                    controller: (index % 128) as u8,
                }),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut engine = MidiEngine::new();
        engine
            .apply_profile(ResolvedControllerProfile {
                name: "feedback capacity".into(),
                input: MidiDeviceSelector::FirstAvailable,
                output: MidiDeviceSelector::FirstAvailable,
                bindings,
            })
            .unwrap();
        engine.queue_feedback(target, 0.5, AutomationOrigin::HostAutomation);
        assert_eq!(
            engine.shared.feedback.lock().unwrap().len(),
            MIDI_FEEDBACK_KEYS_CAPACITY
        );
        engine.queue_feedback(target, 0.75, AutomationOrigin::HostAutomation);
        assert_eq!(
            engine.shared.feedback.lock().unwrap().len(),
            MIDI_FEEDBACK_KEYS_CAPACITY
        );
        assert_eq!(
            engine.local_counters.feedback_coalesced,
            MIDI_FEEDBACK_KEYS_CAPACITY as u64
        );
    }
}
