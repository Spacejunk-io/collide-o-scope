//! Bounded, payload-free action correlation for the live control path.
//!
//! Transport adapters mint an [`ActionEnvelope`] at engine ingress.  The
//! envelope deliberately contains an [`Instant`], not a client timestamp, and
//! is never serialized.  Once the action is applied, the live frame carries
//! the highest accepted sequence through to the audience submission seam.
//! Only timings, source classes, dispositions, and generation numbers enter
//! receipts; authored values and transport payloads never do.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const ACTION_TIMING_WINDOW: usize = 512;
pub const MAX_CORRELATED_ACTIONS_IN_FLIGHT: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActionSequence(u64);

impl ActionSequence {
    pub const fn get(self) -> u64 {
        self.0
    }

    pub const fn from_nonzero(value: u64) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    #[cfg(test)]
    const fn new_for_test(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionSourceClass {
    Browser,
    Phone,
    Native,
    Midi,
    Osc,
    Automation,
    Replay,
}

/// One engine-owned ingress record.  `payload` may be inspected or moved only
/// inside the process; this type intentionally has no serde implementation.
#[derive(Debug, Clone, Copy)]
pub struct ActionEnvelope<T> {
    sequence: ActionSequence,
    source: ActionSourceClass,
    ingress: Instant,
    payload: T,
}

impl<T> ActionEnvelope<T> {
    pub fn sequence(&self) -> ActionSequence {
        self.sequence
    }

    pub fn source(&self) -> ActionSourceClass {
        self.source
    }

    pub fn ingress(&self) -> Instant {
        self.ingress
    }

    /// Payload-independent identity retained when an adapter hands ownership
    /// of the payload to the application.  This remains process-local because
    /// its monotonic [`Instant`] has no valid wire representation.
    pub fn identity(&self) -> ActionIdentity {
        ActionIdentity {
            sequence: self.sequence,
            source: self.source,
            ingress: self.ingress,
        }
    }

    pub fn payload(&self) -> &T {
        &self.payload
    }

    pub fn into_parts(self) -> (ActionSequence, ActionSourceClass, Instant, T) {
        (self.sequence, self.source, self.ingress, self.payload)
    }

    pub fn into_payload(self) -> T {
        self.payload
    }

    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> ActionEnvelope<U> {
        ActionEnvelope {
            sequence: self.sequence,
            source: self.source,
            ingress: self.ingress,
            payload: map(self.payload),
        }
    }
}

/// The correlation fields that must survive payload conversion and
/// application. Like [`ActionEnvelope`], this deliberately has no serde
/// implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionIdentity {
    sequence: ActionSequence,
    source: ActionSourceClass,
    ingress: Instant,
}

impl ActionIdentity {
    pub fn sequence(self) -> ActionSequence {
        self.sequence
    }

    pub fn source(self) -> ActionSourceClass {
        self.source
    }

    pub fn ingress(self) -> Instant {
        self.ingress
    }
}

/// The single monotonic vocabulary shared by native, MIDI, OSC, phone, and
/// browser adapters.  Zero remains the explicit "no correlated action" value
/// on frame plans and wire snapshots.
#[derive(Debug)]
pub struct ActionSequencer {
    next: AtomicU64,
}

impl Default for ActionSequencer {
    fn default() -> Self {
        Self {
            next: AtomicU64::new(1),
        }
    }
}

impl ActionSequencer {
    pub fn envelope<T>(&self, source: ActionSourceClass, payload: T) -> ActionEnvelope<T> {
        self.envelope_at(source, Instant::now(), payload)
    }

    pub fn envelope_at<T>(
        &self,
        source: ActionSourceClass,
        ingress: Instant,
        payload: T,
    ) -> ActionEnvelope<T> {
        // A live process cannot approach u64 exhaustion, but preserving the
        // nonzero invariant keeps hostile/unit-test construction explicit.
        let sequence = self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(if current == u64::MAX { 1 } else { current + 1 })
            })
            .unwrap_or(1)
            .max(1);
        ActionEnvelope {
            sequence: ActionSequence(sequence),
            source,
            ingress,
            payload,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionDisposition {
    Presented,
    Coalesced,
    Refused,
    Superseded,
    Quantized,
    NotYetPresented,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionCorrelationReceipt {
    pub sequence: ActionSequence,
    pub source: ActionSourceClass,
    pub disposition: ActionDisposition,
    pub ingress_to_apply_us: Option<u32>,
    pub apply_to_submit_us: Option<u32>,
    pub submission_generation: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LatencyPercentiles {
    pub p50_us: u32,
    pub p95_us: u32,
    pub p99_us: u32,
    pub samples: u16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ActionTimingSnapshot {
    pub ingress_to_apply: LatencyPercentiles,
    pub apply_to_submit: LatencyPercentiles,
    pub last_presented_sequence: u64,
    pub last_submission_generation: u64,
    pub pending: u16,
    pub uncorrelated_over_capacity: u64,
}

#[derive(Debug, Clone, Copy)]
struct PendingAction {
    sequence: ActionSequence,
    source: ActionSourceClass,
    ingress: Instant,
    applied: Instant,
}

#[derive(Debug, Clone, Copy)]
struct StoredActionReceipt {
    completion_ordinal: u64,
    receipt: ActionCorrelationReceipt,
}

#[derive(Debug, Clone)]
struct DurationWindow {
    values: [u32; ACTION_TIMING_WINDOW],
    next: usize,
    count: usize,
}

impl Default for DurationWindow {
    fn default() -> Self {
        Self {
            values: [0; ACTION_TIMING_WINDOW],
            next: 0,
            count: 0,
        }
    }
}

impl DurationWindow {
    fn record(&mut self, duration: Duration) {
        self.values[self.next] = duration_micros_u32(duration);
        self.next = (self.next + 1) % ACTION_TIMING_WINDOW;
        self.count = (self.count + 1).min(ACTION_TIMING_WINDOW);
    }

    fn percentiles(&self) -> LatencyPercentiles {
        let mut sorted = self.values;
        sorted[..self.count].sort_unstable();
        LatencyPercentiles {
            p50_us: percentile(&sorted[..self.count], 50),
            p95_us: percentile(&sorted[..self.count], 95),
            p99_us: percentile(&sorted[..self.count], 99),
            samples: u16::try_from(self.count).unwrap_or(u16::MAX),
        }
    }
}

/// Fixed-capacity state owned by the live application.  All arrays are
/// allocated with the application, so a warmed action path performs no engine
/// allocation.  A one-slot-over action is classified instead of growing the
/// tracker or blocking the render thread.
#[derive(Debug)]
pub struct ActionCorrelationMonitor {
    pending: [Option<PendingAction>; MAX_CORRELATED_ACTIONS_IN_FLIGHT],
    pending_count: usize,
    receipts: [Option<StoredActionReceipt>; ACTION_TIMING_WINDOW],
    next_receipt: usize,
    receipt_count: usize,
    next_completion_ordinal: u64,
    ingress_to_apply: DurationWindow,
    apply_to_submit: DurationWindow,
    last_presented_sequence: u64,
    last_submission_generation: u64,
    uncorrelated_over_capacity: u64,
}

impl Default for ActionCorrelationMonitor {
    fn default() -> Self {
        Self {
            pending: [None; MAX_CORRELATED_ACTIONS_IN_FLIGHT],
            pending_count: 0,
            receipts: [None; ACTION_TIMING_WINDOW],
            next_receipt: 0,
            receipt_count: 0,
            next_completion_ordinal: 1,
            ingress_to_apply: DurationWindow::default(),
            apply_to_submit: DurationWindow::default(),
            last_presented_sequence: 0,
            last_submission_generation: 0,
            uncorrelated_over_capacity: 0,
        }
    }
}

impl ActionCorrelationMonitor {
    #[cfg(test)]
    pub fn record_apply<T>(&mut self, envelope: &ActionEnvelope<T>, applied: Instant) -> bool {
        self.record_apply_identity(envelope.identity(), applied)
    }

    pub fn record_apply_identity(&mut self, identity: ActionIdentity, applied: Instant) -> bool {
        if self.pending_count >= MAX_CORRELATED_ACTIONS_IN_FLIGHT {
            self.uncorrelated_over_capacity = self.uncorrelated_over_capacity.saturating_add(1);
            self.record_terminal(
                identity.sequence,
                identity.source,
                ActionDisposition::NotYetPresented,
            );
            return false;
        }
        let Some(slot) = self.pending.iter_mut().find(|slot| slot.is_none()) else {
            self.uncorrelated_over_capacity = self.uncorrelated_over_capacity.saturating_add(1);
            self.record_terminal(
                identity.sequence,
                identity.source,
                ActionDisposition::NotYetPresented,
            );
            return false;
        };
        let applied = applied.max(identity.ingress);
        self.ingress_to_apply
            .record(applied.saturating_duration_since(identity.ingress));
        *slot = Some(PendingAction {
            sequence: identity.sequence,
            source: identity.source,
            ingress: identity.ingress,
            applied,
        });
        self.pending_count += 1;
        true
    }

    pub fn record_terminal(
        &mut self,
        sequence: ActionSequence,
        source: ActionSourceClass,
        disposition: ActionDisposition,
    ) {
        debug_assert_ne!(disposition, ActionDisposition::Presented);
        self.push_receipt(ActionCorrelationReceipt {
            sequence,
            source,
            disposition,
            ingress_to_apply_us: None,
            apply_to_submit_us: None,
            submission_generation: None,
        });
    }

    pub fn record_terminal_identity(
        &mut self,
        identity: ActionIdentity,
        disposition: ActionDisposition,
    ) {
        self.record_terminal(identity.sequence, identity.source, disposition);
    }

    /// Close every action that applied but never crossed a real audience
    /// presentation seam. Shutdown calls this after ingress is stopped; the
    /// fixed scratch array preserves bounded storage and makes repeat calls
    /// exact no-ops.
    pub fn terminalize_all_pending(&mut self, disposition: ActionDisposition) -> usize {
        debug_assert_ne!(disposition, ActionDisposition::Presented);
        let mut completed = [None; MAX_CORRELATED_ACTIONS_IN_FLIGHT];
        let mut completed_count = 0;
        for slot in &mut self.pending {
            if let Some(pending) = slot.take() {
                completed[completed_count] = Some(pending);
                completed_count += 1;
            }
        }
        self.pending_count = self.pending_count.saturating_sub(completed_count);
        for pending in completed.into_iter().flatten().take(completed_count) {
            self.record_terminal(pending.sequence, pending.source, disposition);
        }
        completed_count
    }

    /// Correlate all applied sequences admitted into this immutable frame.
    /// `highest_applied` is copied into the frame plan; the generation is
    /// minted at the actual audience submission seam.
    pub fn record_submission(
        &mut self,
        highest_applied: ActionSequence,
        submission_generation: u64,
        submitted: Instant,
    ) {
        let mut completed = [None; MAX_CORRELATED_ACTIONS_IN_FLIGHT];
        let mut completed_count = 0;
        for slot in &mut self.pending {
            let Some(pending) = *slot else {
                continue;
            };
            if pending.sequence <= highest_applied {
                completed[completed_count] = Some(pending);
                completed_count += 1;
                *slot = None;
            }
        }
        self.pending_count = self.pending_count.saturating_sub(completed_count);
        for pending in completed.into_iter().flatten().take(completed_count) {
            let submitted = submitted.max(pending.applied);
            let apply_to_submit = submitted.saturating_duration_since(pending.applied);
            self.apply_to_submit.record(apply_to_submit);
            self.push_receipt(ActionCorrelationReceipt {
                sequence: pending.sequence,
                source: pending.source,
                disposition: ActionDisposition::Presented,
                ingress_to_apply_us: Some(duration_micros_u32(
                    pending.applied.saturating_duration_since(pending.ingress),
                )),
                apply_to_submit_us: Some(duration_micros_u32(apply_to_submit)),
                submission_generation: Some(submission_generation),
            });
            self.last_presented_sequence = self.last_presented_sequence.max(pending.sequence.get());
        }
        self.last_submission_generation =
            self.last_submission_generation.max(submission_generation);
    }

    pub fn snapshot(&self) -> ActionTimingSnapshot {
        ActionTimingSnapshot {
            ingress_to_apply: self.ingress_to_apply.percentiles(),
            apply_to_submit: self.apply_to_submit.percentiles(),
            last_presented_sequence: self.last_presented_sequence,
            last_submission_generation: self.last_submission_generation,
            pending: u16::try_from(self.pending_count).unwrap_or(u16::MAX),
            uncorrelated_over_capacity: self.uncorrelated_over_capacity,
        }
    }

    #[cfg(test)]
    pub fn receipts(&self) -> impl Iterator<Item = &ActionCorrelationReceipt> {
        self.stored_receipts().map(|stored| &stored.receipt)
    }

    /// Iterate by completion order rather than action sequence. Independent
    /// adapters can finish out of ingress order, so using the action sequence
    /// itself as a polling cursor could permanently skip a late receipt.
    pub fn receipts_after(
        &self,
        completion_ordinal: u64,
    ) -> impl Iterator<Item = (u64, &ActionCorrelationReceipt)> {
        self.stored_receipts()
            .filter(move |stored| stored.completion_ordinal > completion_ordinal)
            .map(|stored| (stored.completion_ordinal, &stored.receipt))
    }

    fn stored_receipts(&self) -> impl Iterator<Item = &StoredActionReceipt> {
        let count = self.receipt_count;
        let start = if count == ACTION_TIMING_WINDOW {
            self.next_receipt
        } else {
            0
        };
        (0..count).filter_map(move |offset| {
            self.receipts[(start + offset) % ACTION_TIMING_WINDOW].as_ref()
        })
    }

    fn push_receipt(&mut self, receipt: ActionCorrelationReceipt) {
        let completion_ordinal = self.next_completion_ordinal;
        self.next_completion_ordinal = self.next_completion_ordinal.saturating_add(1);
        self.receipts[self.next_receipt] = Some(StoredActionReceipt {
            completion_ordinal,
            receipt,
        });
        self.next_receipt = (self.next_receipt + 1) % ACTION_TIMING_WINDOW;
        self.receipt_count = (self.receipt_count + 1).min(ACTION_TIMING_WINDOW);
    }
}

fn duration_micros_u32(duration: Duration) -> u32 {
    u32::try_from(duration.as_micros()).unwrap_or(u32::MAX)
}

fn percentile(sorted: &[u32], percentile: usize) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let index = sorted
        .len()
        .saturating_mul(percentile)
        .saturating_add(99)
        .saturating_div(100)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_sequences_are_nonzero_monotonic_and_payload_preserving() {
        let sequencer = ActionSequencer::default();
        let first = sequencer.envelope(ActionSourceClass::Browser, "first");
        let second = sequencer.envelope(ActionSourceClass::Midi, "second");
        assert_eq!(first.sequence().get(), 1);
        assert_eq!(second.sequence().get(), 2);
        assert_eq!(first.payload(), &"first");
        assert_eq!(second.source(), ActionSourceClass::Midi);
    }

    #[test]
    fn payload_mapping_and_consumption_preserve_the_original_ingress_identity() {
        let ingress = Instant::now();
        let sequencer = ActionSequencer::default();
        let envelope = sequencer.envelope_at(ActionSourceClass::Osc, ingress, 41_u32);
        let expected = envelope.identity();
        let mapped = envelope.map(|value| value + 1);

        assert_eq!(mapped.identity(), expected);
        assert_eq!(mapped.into_payload(), 42);
        assert_eq!(expected.sequence().get(), 1);
        assert_eq!(expected.source(), ActionSourceClass::Osc);
        assert_eq!(expected.ingress(), ingress);

        let mut monitor = ActionCorrelationMonitor::default();
        assert!(monitor.record_apply_identity(expected, ingress));
        monitor.record_submission(expected.sequence(), 9, ingress);
        let receipt = monitor.receipts().next().unwrap();
        assert_eq!(receipt.sequence, expected.sequence());
        assert_eq!(receipt.source, expected.source());
        assert_eq!(receipt.disposition, ActionDisposition::Presented);
    }

    #[test]
    fn apply_and_submit_percentiles_share_one_engine_clock() {
        let origin = Instant::now();
        let sequencer = ActionSequencer::default();
        let envelope = sequencer.envelope_at(ActionSourceClass::Phone, origin, ());
        let mut monitor = ActionCorrelationMonitor::default();
        assert!(monitor.record_apply(&envelope, origin + Duration::from_millis(3)));
        monitor.record_submission(envelope.sequence(), 17, origin + Duration::from_millis(11));

        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.ingress_to_apply.p50_us, 3_000);
        assert_eq!(snapshot.apply_to_submit.p50_us, 8_000);
        assert_eq!(snapshot.last_presented_sequence, 1);
        assert_eq!(snapshot.last_submission_generation, 17);
        assert_eq!(snapshot.pending, 0);
        let receipt = monitor.receipts().next().unwrap();
        assert_eq!(receipt.disposition, ActionDisposition::Presented);
        assert_eq!(receipt.submission_generation, Some(17));
    }

    #[test]
    fn every_nonpresented_disposition_is_explicit_and_payload_free() {
        let mut monitor = ActionCorrelationMonitor::default();
        for (index, disposition) in [
            ActionDisposition::Coalesced,
            ActionDisposition::Refused,
            ActionDisposition::Superseded,
            ActionDisposition::Quantized,
            ActionDisposition::NotYetPresented,
        ]
        .into_iter()
        .enumerate()
        {
            monitor.record_terminal(
                ActionSequence::new_for_test(index as u64 + 1),
                ActionSourceClass::Native,
                disposition,
            );
        }
        let dispositions = monitor
            .receipts()
            .map(|receipt| receipt.disposition)
            .collect::<Vec<_>>();
        assert_eq!(
            dispositions,
            vec![
                ActionDisposition::Coalesced,
                ActionDisposition::Refused,
                ActionDisposition::Superseded,
                ActionDisposition::Quantized,
                ActionDisposition::NotYetPresented,
            ]
        );
        assert!(monitor.receipts().all(|receipt| {
            receipt.ingress_to_apply_us.is_none()
                && receipt.apply_to_submit_us.is_none()
                && receipt.submission_generation.is_none()
        }));
    }

    #[test]
    fn completion_cursor_cannot_skip_a_late_lower_action_sequence() {
        let mut monitor = ActionCorrelationMonitor::default();
        monitor.record_terminal(
            ActionSequence::new_for_test(2),
            ActionSourceClass::Osc,
            ActionDisposition::Refused,
        );
        let (cursor, first) = monitor.receipts_after(0).next().unwrap();
        assert_eq!(first.sequence.get(), 2);

        monitor.record_terminal(
            ActionSequence::new_for_test(1),
            ActionSourceClass::Midi,
            ActionDisposition::Coalesced,
        );
        let late = monitor.receipts_after(cursor).collect::<Vec<_>>();
        assert_eq!(late.len(), 1);
        assert_eq!(late[0].1.sequence.get(), 1);
        assert!(late[0].0 > cursor);
    }

    #[test]
    fn one_slot_over_is_explicitly_unpresented_without_growing_the_tracker() {
        let origin = Instant::now();
        let sequencer = ActionSequencer::default();
        let mut monitor = ActionCorrelationMonitor::default();
        for _ in 0..MAX_CORRELATED_ACTIONS_IN_FLIGHT {
            let envelope = sequencer.envelope_at(ActionSourceClass::Automation, origin, ());
            assert!(monitor.record_apply(&envelope, origin));
        }
        let over = sequencer.envelope_at(ActionSourceClass::Automation, origin, ());
        assert!(!monitor.record_apply(&over, origin));
        let snapshot = monitor.snapshot();
        assert_eq!(
            usize::from(snapshot.pending),
            MAX_CORRELATED_ACTIONS_IN_FLIGHT
        );
        assert_eq!(snapshot.uncorrelated_over_capacity, 1);
        assert_eq!(
            monitor.receipts().last().unwrap().disposition,
            ActionDisposition::NotYetPresented
        );
    }

    #[test]
    fn shutdown_terminalizes_applied_unpresented_actions_exactly_once() {
        let origin = Instant::now();
        let sequencer = ActionSequencer::default();
        let action = sequencer.envelope_at(ActionSourceClass::Native, origin, ());
        let mut monitor = ActionCorrelationMonitor::default();
        assert!(monitor.record_apply(&action, origin + Duration::from_millis(1)));
        assert_eq!(monitor.snapshot().pending, 1);

        assert_eq!(
            monitor.terminalize_all_pending(ActionDisposition::NotYetPresented),
            1
        );
        assert_eq!(monitor.snapshot().pending, 0);
        assert_eq!(
            monitor.terminalize_all_pending(ActionDisposition::NotYetPresented),
            0
        );
        let receipts = monitor.receipts().collect::<Vec<_>>();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].sequence, action.sequence());
        assert_eq!(receipts[0].disposition, ActionDisposition::NotYetPresented);
    }

    #[test]
    fn receipt_and_percentile_windows_are_newest_only_and_bounded() {
        let origin = Instant::now();
        let sequencer = ActionSequencer::default();
        let mut monitor = ActionCorrelationMonitor::default();
        for index in 0..ACTION_TIMING_WINDOW * 2 {
            let envelope = sequencer.envelope_at(ActionSourceClass::Osc, origin, ());
            assert!(monitor.record_apply(&envelope, origin + Duration::from_micros(index as u64)));
            monitor.record_submission(
                envelope.sequence(),
                index as u64 + 1,
                origin + Duration::from_micros(index as u64 + 1),
            );
        }
        assert_eq!(monitor.receipts().count(), ACTION_TIMING_WINDOW);
        assert_eq!(
            usize::from(monitor.snapshot().ingress_to_apply.samples),
            ACTION_TIMING_WINDOW
        );
    }
}
