//! Bounded ownership and retirement for threaded video decoders.
//!
//! Decoder handles live here from the instant a worker is spawned.  A live
//! [`ThreadedDecoder`](super::threaded::ThreadedDecoder) keeps only a token;
//! dropping or replacing it marks that exact worker (source fingerprint plus
//! generation) retired.  The supervisor is the only code that joins decoder
//! threads, and it joins only handles which already report `is_finished()`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use super::threaded::DecodeCommandMailbox;

/// The authored graph admits at most 256 live layers.  Sixty-four additional
/// slots leave room for prepared sources and ordinary source replacement
/// without allowing a stuck codec/input to grow process handles forever.
pub const DECODER_WORKER_HARD_CAP: usize = 320;
/// Once this many old workers are still retiring, new decoder opens are
/// refused.  Existing live workers keep their reserved retirement slots, so a
/// mass remove/shutdown can still transfer every handle without overflow.
pub const DECODER_RETIREMENT_CHURN_CAP: usize = 64;
/// The event-loop shutdown path waits on supervisor progress, never on a
/// decoder `JoinHandle`, for at most this duration.
pub const DECODER_RETIREMENT_SHUTDOWN_DEADLINE: Duration = Duration::from_secs(2);
const REAPER_POLL_INTERVAL: Duration = Duration::from_millis(2);

/// A path-free stable identifier for one decoder source spelling.
///
/// The full host path never enters retirement telemetry or status text.  The
/// 128-bit SHA-256 prefix is ample for process-lifetime correlation while
/// keeping the status payload fixed-size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DecoderSourceFingerprint(pub [u8; 16]);

impl DecoderSourceFingerprint {
    pub fn from_source(source: &str) -> Self {
        let digest = Sha256::digest(source.as_bytes());
        let mut fingerprint = [0_u8; 16];
        fingerprint.copy_from_slice(&digest[..16]);
        Self(fingerprint)
    }

    pub fn short_hex(self) -> String {
        let mut result = String::with_capacity(16);
        for byte in &self.0[..8] {
            use std::fmt::Write as _;
            let _ = write!(result, "{byte:02x}");
        }
        result
    }
}

/// Stable correlation carried with a retiring decoder handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderRetirementIdentity {
    pub worker_id: u64,
    pub source: DecoderSourceFingerprint,
    pub source_generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderRetirementHealth {
    Healthy,
    Saturated,
    Stuck,
}

/// Fixed-size process-wide decoder ownership status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderRetirementSnapshot {
    pub health: DecoderRetirementHealth,
    pub active_workers: usize,
    pub retiring_workers: usize,
    pub stuck_workers: usize,
    pub owned_workers: usize,
    pub peak_owned_workers: usize,
    pub peak_retiring_workers: usize,
    pub oldest_retirement_age: Option<Duration>,
    pub oldest_retiree: Option<DecoderRetirementIdentity>,
    pub completed_workers: u64,
    pub panicked_workers: u64,
    pub admission_refusals: u64,
    pub hard_cap: usize,
    pub churn_cap: usize,
    pub accepting_new_workers: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderRetirementDrainReceipt {
    pub completed: bool,
    pub elapsed: Duration,
    pub snapshot: DecoderRetirementSnapshot,
}

#[derive(Debug, Clone, Copy)]
struct SupervisorConfig {
    hard_cap: usize,
    churn_cap: usize,
    poll_interval: Duration,
}

impl SupervisorConfig {
    const PRODUCTION: Self = Self {
        hard_cap: DECODER_WORKER_HARD_CAP,
        churn_cap: DECODER_RETIREMENT_CHURN_CAP,
        poll_interval: REAPER_POLL_INTERVAL,
    };
}

#[derive(Debug, Clone, Copy)]
enum WorkerLifecycle {
    Starting,
    Active,
    Retiring { since: Instant },
    Stuck { since: Instant },
}

struct WorkerRecord {
    identity: DecoderRetirementIdentity,
    lifecycle: WorkerLifecycle,
    cancel: Arc<AtomicBool>,
    mailbox: Arc<DecodeCommandMailbox>,
    handle: Option<JoinHandle<()>>,
}

struct SupervisorState {
    records: BTreeMap<u64, WorkerRecord>,
    next_worker_id: u64,
    completed_workers: u64,
    panicked_workers: u64,
    admission_refusals: u64,
    peak_owned_workers: usize,
    peak_retiring_workers: usize,
    accepting_new_workers: bool,
    stop_reaper: bool,
    #[cfg(test)]
    last_decoder_join_thread: Option<std::thread::ThreadId>,
}

impl Default for SupervisorState {
    fn default() -> Self {
        Self {
            records: BTreeMap::new(),
            next_worker_id: 0,
            completed_workers: 0,
            panicked_workers: 0,
            admission_refusals: 0,
            peak_owned_workers: 0,
            peak_retiring_workers: 0,
            accepting_new_workers: true,
            stop_reaper: false,
            #[cfg(test)]
            last_decoder_join_thread: None,
        }
    }
}

struct SupervisorShared {
    config: SupervisorConfig,
    state: Mutex<SupervisorState>,
    wake: Condvar,
}

struct DecoderRetirementSupervisor {
    shared: Arc<SupervisorShared>,
    /// Kept for process lifetime in production.  Test supervisors explicitly
    /// stop and join this supervisor thread only after all decoder handles
    /// have been reaped.
    #[allow(
        dead_code,
        reason = "retaining the supervisor JoinHandle is the ownership contract"
    )]
    reaper: Mutex<Option<JoinHandle<()>>>,
}

impl DecoderRetirementSupervisor {
    fn new(config: SupervisorConfig) -> Self {
        assert!(config.hard_cap > 0);
        assert!(config.churn_cap > 0);
        assert!(config.churn_cap <= config.hard_cap);
        let shared = Arc::new(SupervisorShared {
            config,
            state: Mutex::new(SupervisorState::default()),
            wake: Condvar::new(),
        });
        let reaper_shared = shared.clone();
        let reaper = std::thread::Builder::new()
            .name("decoder-retirement-reaper".to_owned())
            .spawn(move || reaper_loop(reaper_shared))
            .expect("decoder retirement supervisor must be spawnable");
        Self {
            shared,
            reaper: Mutex::new(Some(reaper)),
        }
    }

    fn admit(
        &self,
        source: DecoderSourceFingerprint,
        cancel: Arc<AtomicBool>,
        mailbox: Arc<DecodeCommandMailbox>,
    ) -> Result<DecoderWorkerToken, String> {
        let mut state = lock_recover(&self.shared.state);
        let retiring = retiring_count(&state);
        let refusal = if !state.accepting_new_workers {
            Some("decoder supervisor is draining for shutdown".to_owned())
        } else if state.records.len() >= self.shared.config.hard_cap {
            Some(format!(
                "decoder worker ownership reached its {}-handle hard cap",
                self.shared.config.hard_cap
            ))
        } else if retiring >= self.shared.config.churn_cap {
            Some(format!(
                "decoder retirement backlog reached its {}-worker churn cap",
                self.shared.config.churn_cap
            ))
        } else {
            None
        };
        if let Some(refusal) = refusal {
            state.admission_refusals = state.admission_refusals.saturating_add(1);
            return Err(refusal);
        }

        state.next_worker_id = state
            .next_worker_id
            .checked_add(1)
            .ok_or_else(|| "decoder worker identity exhausted".to_owned())?;
        let worker_id = state.next_worker_id;
        state.records.insert(
            worker_id,
            WorkerRecord {
                identity: DecoderRetirementIdentity {
                    worker_id,
                    source,
                    source_generation: 0,
                },
                lifecycle: WorkerLifecycle::Starting,
                cancel,
                mailbox,
                handle: None,
            },
        );
        state.peak_owned_workers = state.peak_owned_workers.max(state.records.len());
        drop(state);
        self.shared.wake.notify_all();
        Ok(DecoderWorkerToken {
            shared: self.shared.clone(),
            worker_id: Some(worker_id),
        })
    }

    fn snapshot(&self) -> DecoderRetirementSnapshot {
        snapshot_locked(&lock_recover(&self.shared.state), self.shared.config)
    }

    fn drain_with_deadline(&self, deadline: Duration) -> DecoderRetirementDrainReceipt {
        let started = Instant::now();
        let mut state = lock_recover(&self.shared.state);
        state.accepting_new_workers = false;
        let now = Instant::now();
        for record in state.records.values_mut() {
            record.cancel.store(true, Ordering::Release);
            record.mailbox.stop();
            record.identity.source_generation = record
                .identity
                .source_generation
                .max(record.mailbox.generation());
            if matches!(
                record.lifecycle,
                WorkerLifecycle::Starting | WorkerLifecycle::Active
            ) {
                record.lifecycle = WorkerLifecycle::Retiring { since: now };
            }
        }
        update_retirement_peak(&mut state);
        self.shared.wake.notify_all();

        while !state.records.is_empty() {
            let remaining = deadline.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                break;
            }
            let wait = remaining.min(
                self.shared
                    .config
                    .poll_interval
                    .max(Duration::from_millis(1)),
            );
            let (next, _) = self
                .shared
                .wake
                .wait_timeout(state, wait)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
        }

        let completed = state.records.is_empty();
        if !completed {
            let stuck_at = Instant::now();
            for record in state.records.values_mut() {
                let since = match record.lifecycle {
                    WorkerLifecycle::Retiring { since } | WorkerLifecycle::Stuck { since } => since,
                    WorkerLifecycle::Starting | WorkerLifecycle::Active => stuck_at,
                };
                record.lifecycle = WorkerLifecycle::Stuck { since };
            }
            update_retirement_peak(&mut state);
        }
        DecoderRetirementDrainReceipt {
            completed,
            elapsed: started.elapsed(),
            snapshot: snapshot_locked(&state, self.shared.config),
        }
    }

    #[cfg(test)]
    fn stop_reaper_after_tests(&self) {
        {
            let mut state = lock_recover(&self.shared.state);
            assert!(state.records.is_empty(), "test left decoder handles owned");
            state.stop_reaper = true;
        }
        self.shared.wake.notify_all();
        if let Some(reaper) = lock_recover(&self.reaper).take() {
            reaper.join().expect("test retirement reaper panicked");
        }
    }
}

/// Admission token held by exactly one live `ThreadedDecoder`.
pub(super) struct DecoderWorkerToken {
    shared: Arc<SupervisorShared>,
    worker_id: Option<u64>,
}

impl DecoderWorkerToken {
    pub(super) fn attach(&self, handle: JoinHandle<()>) {
        let worker_id = self
            .worker_id
            .expect("an abandoned decoder token cannot attach a handle");
        let mut state = lock_recover(&self.shared.state);
        let record = state
            .records
            .get_mut(&worker_id)
            .expect("an admitted decoder token retains its supervisor record");
        assert!(record.handle.is_none(), "decoder handle attached twice");
        record.handle = Some(handle);
        if matches!(record.lifecycle, WorkerLifecycle::Starting) {
            record.lifecycle = WorkerLifecycle::Active;
        }
        drop(state);
        self.shared.wake.notify_all();
    }

    pub(super) fn abandon(mut self) {
        let Some(worker_id) = self.worker_id.take() else {
            return;
        };
        let mut state = lock_recover(&self.shared.state);
        let record = state
            .records
            .remove(&worker_id)
            .expect("an unspawned decoder reservation remains registered");
        assert!(
            record.handle.is_none(),
            "spawned decoder cannot be abandoned"
        );
        drop(state);
        self.shared.wake.notify_all();
    }

    pub(super) fn retire(mut self, source_generation: u64) {
        let Some(worker_id) = self.worker_id.take() else {
            return;
        };
        retire_worker(&self.shared, worker_id, source_generation);
    }
}

impl Drop for DecoderWorkerToken {
    fn drop(&mut self) {
        if let Some(worker_id) = self.worker_id.take() {
            retire_worker(&self.shared, worker_id, 0);
        }
    }
}

fn retire_worker(shared: &SupervisorShared, worker_id: u64, source_generation: u64) {
    let mut state = lock_recover(&shared.state);
    let Some(record) = state.records.get_mut(&worker_id) else {
        // Shutdown may already have canceled, reaped, and removed this live
        // token's worker before the owner itself is dropped.
        return;
    };
    record.cancel.store(true, Ordering::Release);
    record.mailbox.stop();
    record.identity.source_generation = source_generation.max(record.mailbox.generation());
    if matches!(
        record.lifecycle,
        WorkerLifecycle::Starting | WorkerLifecycle::Active
    ) {
        record.lifecycle = WorkerLifecycle::Retiring {
            since: Instant::now(),
        };
    }
    update_retirement_peak(&mut state);
    drop(state);
    shared.wake.notify_all();
}

fn reaper_loop(shared: Arc<SupervisorShared>) {
    loop {
        let ready = {
            let mut state = lock_recover(&shared.state);
            loop {
                let ready_ids: Vec<u64> = state
                    .records
                    .iter()
                    .filter_map(|(worker_id, record)| {
                        (matches!(
                            record.lifecycle,
                            WorkerLifecycle::Retiring { .. } | WorkerLifecycle::Stuck { .. }
                        ) && record.handle.as_ref().is_some_and(JoinHandle::is_finished))
                        .then_some(*worker_id)
                    })
                    .collect();
                if !ready_ids.is_empty() {
                    let mut ready = Vec::with_capacity(ready_ids.len());
                    for worker_id in ready_ids {
                        if let Some(handle) = state
                            .records
                            .get_mut(&worker_id)
                            .and_then(|record| record.handle.take())
                        {
                            ready.push((worker_id, handle));
                        }
                    }
                    break ready;
                }
                if state.stop_reaper && state.records.is_empty() {
                    return;
                }
                if retiring_count(&state) == 0 {
                    state = shared
                        .wake
                        .wait(state)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                } else {
                    let (next, _) = shared
                        .wake
                        .wait_timeout(state, shared.config.poll_interval)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    state = next;
                }
            }
        };

        for (worker_id, handle) in ready {
            let panicked = handle.join().is_err();
            let mut state = lock_recover(&shared.state);
            #[cfg(test)]
            {
                state.last_decoder_join_thread = Some(std::thread::current().id());
            }
            if state.records.remove(&worker_id).is_some() {
                state.completed_workers = state.completed_workers.saturating_add(1);
                state.panicked_workers = state.panicked_workers.saturating_add(u64::from(panicked));
            }
            drop(state);
            shared.wake.notify_all();
        }
    }
}

fn retiring_count(state: &SupervisorState) -> usize {
    state
        .records
        .values()
        .filter(|record| {
            matches!(
                record.lifecycle,
                WorkerLifecycle::Retiring { .. } | WorkerLifecycle::Stuck { .. }
            )
        })
        .count()
}

fn update_retirement_peak(state: &mut SupervisorState) {
    state.peak_retiring_workers = state.peak_retiring_workers.max(retiring_count(state));
}

fn snapshot_locked(state: &SupervisorState, config: SupervisorConfig) -> DecoderRetirementSnapshot {
    let now = Instant::now();
    let mut active_workers = 0_usize;
    let mut retiring_workers = 0_usize;
    let mut stuck_workers = 0_usize;
    let mut oldest: Option<(Duration, DecoderRetirementIdentity)> = None;
    for record in state.records.values() {
        match record.lifecycle {
            WorkerLifecycle::Starting | WorkerLifecycle::Active => active_workers += 1,
            WorkerLifecycle::Retiring { since } => {
                retiring_workers += 1;
                let age = now.saturating_duration_since(since);
                if oldest.is_none_or(|(oldest_age, _)| age > oldest_age) {
                    oldest = Some((age, record.identity));
                }
            }
            WorkerLifecycle::Stuck { since } => {
                stuck_workers += 1;
                let age = now.saturating_duration_since(since);
                if oldest.is_none_or(|(oldest_age, _)| age > oldest_age) {
                    oldest = Some((age, record.identity));
                }
            }
        }
    }
    let retirement_backlog = retiring_workers.saturating_add(stuck_workers);
    let health = if stuck_workers > 0 {
        DecoderRetirementHealth::Stuck
    } else if !state.accepting_new_workers
        || state.records.len() >= config.hard_cap
        || retirement_backlog >= config.churn_cap
    {
        DecoderRetirementHealth::Saturated
    } else {
        DecoderRetirementHealth::Healthy
    };
    DecoderRetirementSnapshot {
        health,
        active_workers,
        retiring_workers,
        stuck_workers,
        owned_workers: state.records.len(),
        peak_owned_workers: state.peak_owned_workers,
        peak_retiring_workers: state.peak_retiring_workers,
        oldest_retirement_age: oldest.map(|(age, _)| age),
        oldest_retiree: oldest.map(|(_, identity)| identity),
        completed_workers: state.completed_workers,
        panicked_workers: state.panicked_workers,
        admission_refusals: state.admission_refusals,
        hard_cap: config.hard_cap,
        churn_cap: config.churn_cap,
        accepting_new_workers: state.accepting_new_workers,
    }
}

fn global_supervisor() -> &'static DecoderRetirementSupervisor {
    static SUPERVISOR: OnceLock<DecoderRetirementSupervisor> = OnceLock::new();
    SUPERVISOR.get_or_init(|| DecoderRetirementSupervisor::new(SupervisorConfig::PRODUCTION))
}

pub(super) fn admit_decoder_worker(
    source: DecoderSourceFingerprint,
    cancel: Arc<AtomicBool>,
    mailbox: Arc<DecodeCommandMailbox>,
) -> Result<DecoderWorkerToken, String> {
    global_supervisor().admit(source, cancel, mailbox)
}

#[allow(
    dead_code,
    reason = "live Stage Health consumers may sample this fixed-size publication independently of shutdown"
)]
pub fn decoder_retirement_snapshot() -> DecoderRetirementSnapshot {
    global_supervisor().snapshot()
}

pub fn drain_decoder_retirements_with_deadline(
    deadline: Duration,
) -> DecoderRetirementDrainReceipt {
    global_supervisor().drain_with_deadline(deadline)
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_supervisor(hard_cap: usize, churn_cap: usize) -> DecoderRetirementSupervisor {
        DecoderRetirementSupervisor::new(SupervisorConfig {
            hard_cap,
            churn_cap,
            poll_interval: Duration::from_millis(1),
        })
    }

    fn attach_cancel_aware_worker(
        supervisor: &DecoderRetirementSupervisor,
        source: &str,
    ) -> DecoderWorkerToken {
        let mailbox = DecodeCommandMailbox::new();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let token = supervisor
            .admit(
                DecoderSourceFingerprint::from_source(source),
                cancel,
                mailbox,
            )
            .expect("test worker admitted");
        let handle = std::thread::spawn(move || {
            while !worker_cancel.load(Ordering::Acquire) {
                std::thread::park_timeout(Duration::from_millis(1));
            }
        });
        token.attach(handle);
        token
    }

    fn wait_until_empty(supervisor: &DecoderRetirementSupervisor) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while supervisor.snapshot().owned_workers != 0 {
            assert!(Instant::now() < deadline, "decoder reaper did not catch up");
            std::thread::yield_now();
        }
    }

    #[test]
    fn source_identity_is_stable_and_path_free() {
        let first = DecoderSourceFingerprint::from_source(r"C:\private\show\clip.mov");
        let second = DecoderSourceFingerprint::from_source(r"C:\private\show\clip.mov");
        let other = DecoderSourceFingerprint::from_source(r"C:\private\show\other.mov");
        assert_eq!(first, second);
        assert_ne!(first, other);
        assert_eq!(first.short_hex().len(), 16);
        assert!(!format!("{first:?}").contains("private"));
    }

    #[test]
    fn rapid_replace_remove_and_proxy_adopt_keep_handles_bounded() {
        let supervisor = test_supervisor(8, 4);
        let render_thread = std::thread::current().id();
        let mut live: Option<DecoderWorkerToken> = None;
        let mut created = 0_u64;

        for generation in 1..=96_u64 {
            // Replacement and proxy adoption both prepare the successor first,
            // then retire the exact displaced source.  Every ninth iteration
            // models removal before the next adoption.
            let successor = attach_cancel_aware_worker(
                &supervisor,
                if generation % 3 == 0 {
                    "proxy-cache-artifact"
                } else {
                    "original-source"
                },
            );
            created += 1;
            if let Some(displaced) = live.replace(successor) {
                displaced.retire(generation - 1);
            }
            if generation % 9 == 0 {
                live.take().unwrap().retire(generation);
            }
            while supervisor.snapshot().retiring_workers >= 3 {
                std::thread::yield_now();
            }
        }
        if let Some(last) = live.take() {
            last.retire(96);
        }
        wait_until_empty(&supervisor);

        let snapshot = supervisor.snapshot();
        assert_eq!(snapshot.completed_workers, created);
        assert!(snapshot.peak_owned_workers <= snapshot.hard_cap);
        assert!(snapshot.peak_retiring_workers <= snapshot.hard_cap);
        let join_thread = lock_recover(&supervisor.shared.state)
            .last_decoder_join_thread
            .expect("at least one decoder was joined");
        assert_ne!(
            join_thread, render_thread,
            "render/main thread joined a decoder"
        );
        supervisor.stop_reaper_after_tests();
    }

    #[test]
    fn hard_cap_reserves_retirement_for_every_live_worker() {
        let supervisor = test_supervisor(3, 2);
        let first = attach_cancel_aware_worker(&supervisor, "one");
        let second = attach_cancel_aware_worker(&supervisor, "two");
        let third = attach_cancel_aware_worker(&supervisor, "three");
        let refusal = supervisor
            .admit(
                DecoderSourceFingerprint::from_source("four"),
                Arc::new(AtomicBool::new(false)),
                DecodeCommandMailbox::new(),
            )
            .err()
            .expect("one over the hard cap is refused");
        assert!(refusal.contains("3-handle hard cap"));

        first.retire(1);
        second.retire(2);
        third.retire(3);
        assert!(supervisor.snapshot().owned_workers <= 3);
        wait_until_empty(&supervisor);
        assert_eq!(supervisor.snapshot().completed_workers, 3);
        supervisor.stop_reaper_after_tests();
    }

    #[test]
    fn stuck_worker_times_out_visibly_and_blocks_successor_churn() {
        let supervisor = test_supervisor(3, 1);
        let mailbox = DecodeCommandMailbox::new();
        let cancel = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let worker_release = release.clone();
        let source = DecoderSourceFingerprint::from_source("injected-stuck-source");
        let token = supervisor
            .admit(source, cancel, mailbox)
            .expect("stuck fixture admitted");
        token.attach(std::thread::spawn(move || {
            while !worker_release.load(Ordering::Acquire) {
                std::thread::park_timeout(Duration::from_millis(1));
            }
        }));
        token.retire(73);

        let refusal = supervisor
            .admit(
                DecoderSourceFingerprint::from_source("unbounded-successor"),
                Arc::new(AtomicBool::new(false)),
                DecodeCommandMailbox::new(),
            )
            .err()
            .expect("churn is refused while the stuck predecessor owns a handle");
        assert!(refusal.contains("1-worker churn cap"));

        let receipt = supervisor.drain_with_deadline(Duration::from_millis(20));
        assert!(!receipt.completed);
        assert_eq!(receipt.snapshot.health, DecoderRetirementHealth::Stuck);
        assert_eq!(receipt.snapshot.stuck_workers, 1);
        assert_eq!(receipt.snapshot.owned_workers, 1);
        assert_eq!(
            receipt.snapshot.oldest_retiree,
            Some(DecoderRetirementIdentity {
                worker_id: 1,
                source,
                source_generation: 73,
            })
        );

        release.store(true, Ordering::Release);
        wait_until_empty(&supervisor);
        assert_eq!(supervisor.snapshot().completed_workers, 1);
        supervisor.stop_reaper_after_tests();
    }
}
