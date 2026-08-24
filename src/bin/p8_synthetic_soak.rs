//! Deterministic, allocation-stable CPU soak for P8 verification.
//!
//! This is not a codec/GPU substitute. It continuously exercises the pure
//! 1/3/8-layer scheduling, temporal state, controller, output, recording,
//! proxy, coalescing, cancellation, and publication laws without copyrighted
//! fixtures. The default CLI budget is one hour; `--iterations` is the exact
//! replay mode used by tests and failure reports.

use std::array;
use std::time::{Duration, Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};

#[path = "../publication_gate.rs"]
mod publication_gate;

use publication_gate::{LatestOnlyPublicationGate, PublicationToken};

const SOAK_SEED: u64 = 0xC011_1DE0_50A8_0001;
const LAYER_COUNTS: [usize; 3] = [1, 3, 8];
const TEMPORAL_SLOTS: usize = 24;
const SCENARIO_QUANTUM: u64 = 4_096;
const DEFAULT_SECONDS: u64 = 3_600;
const DEFAULT_REPORT_SECONDS: u64 = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SyntheticMedia {
    #[serde(rename = "h264_color_bars")]
    H264,
    #[serde(rename = "vp9_color_bars")]
    Vp9,
    #[serde(rename = "vfr_color_bars")]
    Vfr,
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }
}

#[derive(Debug, Clone)]
struct LayerState {
    media: SyntheticMedia,
    frame_ordinal: u64,
    temporal: [u64; TEMPORAL_SLOTS],
    temporal_cursor: usize,
    mosh_state: u64,
    vhs_state: u64,
    proxy_gate: LatestOnlyPublicationGate,
    pending_proxy: Option<PublicationToken>,
    published_proxy_generation: Option<u64>,
}

impl LayerState {
    fn new(index: usize) -> Self {
        let media = match index % 3 {
            0 => SyntheticMedia::H264,
            1 => SyntheticMedia::Vp9,
            _ => SyntheticMedia::Vfr,
        };
        Self {
            media,
            frame_ordinal: 0,
            temporal: [0; TEMPORAL_SLOTS],
            temporal_cursor: 0,
            mosh_state: 0,
            vhs_state: 0,
            proxy_gate: LatestOnlyPublicationGate::default(),
            pending_proxy: None,
            published_proxy_generation: None,
        }
    }

    fn publish_pending(&mut self, telemetry: &mut ScenarioTelemetry) {
        let Some(token) = self.pending_proxy.take() else {
            return;
        };
        assert!(self.proxy_gate.try_publish(token));
        self.published_proxy_generation = Some(token.generation());
        telemetry.proxy_publications = telemetry.proxy_publications.saturating_add(1);
    }

    fn step(
        &mut self,
        layer_index: usize,
        tick: u64,
        random: u64,
        telemetry: &mut ScenarioTelemetry,
    ) {
        self.publish_pending(telemetry);
        self.frame_ordinal = self.frame_ordinal.wrapping_add(1);
        self.temporal[self.temporal_cursor] = self.frame_ordinal ^ random;
        self.temporal_cursor = (self.temporal_cursor + 1) % TEMPORAL_SLOTS;

        // Pure fixed-storage stand-ins for Temporal/Mosh/VHS transitions. The
        // arithmetic is intentionally stateful and source-dependent so a
        // broken loop or stale frame changes the final reproducible digest.
        self.mosh_state = self
            .mosh_state
            .rotate_left(7)
            .wrapping_add(random ^ self.frame_ordinal);
        self.vhs_state = self
            .vhs_state
            .rotate_right(3)
            .wrapping_add(self.temporal[self.temporal_cursor]);

        if tick.wrapping_add(layer_index as u64).is_multiple_of(257) {
            telemetry.proxy_requests = telemetry.proxy_requests.saturating_add(1);
            self.pending_proxy = Some(self.proxy_gate.request());
        }
        if tick.wrapping_add(layer_index as u64).is_multiple_of(4_099) {
            // Seed one exact stale-generation race. The first completion must
            // be refused and the newest token remains the sole pending owner.
            let stale = self.proxy_gate.request();
            let newest = self.proxy_gate.request();
            telemetry.proxy_requests = telemetry.proxy_requests.saturating_add(2);
            assert!(!self.proxy_gate.try_publish(stale));
            telemetry.stale_publications_refused =
                telemetry.stale_publications_refused.saturating_add(1);
            self.pending_proxy = Some(newest);
        }
    }

    fn hash_into(&self, hasher: &mut Sha256) {
        hasher.update([match self.media {
            SyntheticMedia::H264 => 1,
            SyntheticMedia::Vp9 => 2,
            SyntheticMedia::Vfr => 3,
        }]);
        hasher.update(self.frame_ordinal.to_le_bytes());
        for value in self.temporal {
            hasher.update(value.to_le_bytes());
        }
        hasher.update(self.mosh_state.to_le_bytes());
        hasher.update(self.vhs_state.to_le_bytes());
        hasher.update(
            self.published_proxy_generation
                .unwrap_or_default()
                .to_le_bytes(),
        );
    }
}

#[derive(Debug, Clone)]
struct ScenarioState {
    active_layers: usize,
    layers: [LayerState; 8],
    output_enabled: bool,
    recording_gate: LatestOnlyPublicationGate,
    pending_recording: Option<PublicationToken>,
    telemetry: ScenarioTelemetry,
}

impl ScenarioState {
    fn new(active_layers: usize) -> Self {
        Self {
            active_layers,
            layers: array::from_fn(LayerState::new),
            output_enabled: false,
            recording_gate: LatestOnlyPublicationGate::default(),
            pending_recording: None,
            telemetry: ScenarioTelemetry {
                active_layers,
                ..ScenarioTelemetry::default()
            },
        }
    }

    fn step(&mut self, tick: u64, rng: &mut DeterministicRng) {
        if let Some(token) = self.pending_recording.take() {
            assert!(self.recording_gate.try_publish(token));
            self.telemetry.recording_publications =
                self.telemetry.recording_publications.saturating_add(1);
        }

        for layer_index in 0..self.active_layers {
            self.layers[layer_index].step(layer_index, tick, rng.next(), &mut self.telemetry);
        }

        // Fixed-seed controller traffic remains finite and bounded.
        let controller = (rng.next() >> 40) as u32;
        let normalized = controller as f32 / 16_777_215.0;
        assert!(normalized.is_finite() && (0.0..=1.0).contains(&normalized));
        self.telemetry.controller_events = self.telemetry.controller_events.saturating_add(1);

        if tick.is_multiple_of(521) {
            self.output_enabled = !self.output_enabled;
            self.telemetry.output_toggles = self.telemetry.output_toggles.saturating_add(1);
        }
        if tick.is_multiple_of(997) {
            self.pending_recording = Some(self.recording_gate.request());
            self.telemetry.recording_requests = self.telemetry.recording_requests.saturating_add(1);
        }

        self.telemetry.iterations = self.telemetry.iterations.saturating_add(1);
        let pending = self
            .layers
            .iter()
            .take(self.active_layers)
            .filter(|layer| layer.pending_proxy.is_some())
            .count()
            + usize::from(self.pending_recording.is_some());
        self.telemetry.max_pending_publications =
            self.telemetry.max_pending_publications.max(pending);
        assert!(pending <= self.active_layers + 1);
    }

    fn drain(&mut self) {
        for layer in self.layers.iter_mut().take(self.active_layers) {
            layer.publish_pending(&mut self.telemetry);
        }
        if let Some(token) = self.pending_recording.take() {
            assert!(self.recording_gate.try_publish(token));
            self.telemetry.recording_publications =
                self.telemetry.recording_publications.saturating_add(1);
        }
    }

    fn pending_publications(&self) -> usize {
        self.layers
            .iter()
            .take(self.active_layers)
            .filter(|layer| layer.pending_proxy.is_some())
            .count()
            + usize::from(self.pending_recording.is_some())
    }

    fn hash_into(&self, hasher: &mut Sha256) {
        hasher.update((self.active_layers as u64).to_le_bytes());
        hasher.update([u8::from(self.output_enabled)]);
        for layer in self.layers.iter().take(self.active_layers) {
            layer.hash_into(hasher);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
struct ScenarioTelemetry {
    active_layers: usize,
    iterations: u64,
    controller_events: u64,
    output_toggles: u64,
    proxy_requests: u64,
    proxy_publications: u64,
    recording_requests: u64,
    recording_publications: u64,
    stale_publications_refused: u64,
    max_pending_publications: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct SoakReceipt {
    schema_version: u16,
    seed_hex: String,
    steps: u64,
    fixed_temporal_slots: usize,
    fixed_layer_objects: usize,
    max_pending_publication_bound: usize,
    pending_publications_after_drain: usize,
    state_sha256: String,
    scenarios: [ScenarioTelemetry; 3],
}

struct SyntheticSoak {
    rng: DeterministicRng,
    scenarios: [ScenarioState; 3],
    steps: u64,
}

impl SyntheticSoak {
    fn new(seed: u64) -> Self {
        Self {
            rng: DeterministicRng::new(seed),
            scenarios: LAYER_COUNTS.map(ScenarioState::new),
            steps: 0,
        }
    }

    fn step(&mut self) {
        let scenario_index = ((self.steps / SCENARIO_QUANTUM) % 3) as usize;
        self.scenarios[scenario_index].step(self.steps, &mut self.rng);
        self.steps = self.steps.saturating_add(1);
    }

    fn receipt(mut self) -> SoakReceipt {
        for scenario in &mut self.scenarios {
            scenario.drain();
        }
        let pending_publications_after_drain = self
            .scenarios
            .iter()
            .map(ScenarioState::pending_publications)
            .sum();
        assert_eq!(pending_publications_after_drain, 0);

        let mut hasher = Sha256::new();
        hasher.update(SOAK_SEED.to_le_bytes());
        hasher.update(self.steps.to_le_bytes());
        for scenario in &self.scenarios {
            scenario.hash_into(&mut hasher);
        }
        let state_sha256 = format!("{:x}", hasher.finalize());
        let scenarios = self.scenarios.map(|scenario| scenario.telemetry);
        let max_pending_publication_bound = scenarios
            .iter()
            .map(|scenario| scenario.max_pending_publications)
            .max()
            .unwrap_or_default();
        SoakReceipt {
            schema_version: 1,
            seed_hex: format!("{SOAK_SEED:016X}"),
            steps: self.steps,
            fixed_temporal_slots: 3 * 8 * TEMPORAL_SLOTS,
            fixed_layer_objects: 3 * 8,
            max_pending_publication_bound,
            pending_publications_after_drain,
            state_sha256,
            scenarios,
        }
    }
}

fn run_iterations(iterations: u64) -> SoakReceipt {
    let mut soak = SyntheticSoak::new(SOAK_SEED);
    for _ in 0..iterations {
        soak.step();
    }
    soak.receipt()
}

fn print_interval(soak: &SyntheticSoak, elapsed: Duration) {
    let snapshot = serde_json::json!({
        "type": "p8_soak_interval",
        "seed_hex": format!("{SOAK_SEED:016X}"),
        "elapsed_seconds": elapsed.as_secs(),
        "steps": soak.steps,
        "scenario_iterations": soak.scenarios.each_ref().map(|scenario| scenario.telemetry.iterations),
        "max_pending_publications": soak.scenarios.each_ref().map(|scenario| scenario.telemetry.max_pending_publications),
        "fixed_temporal_slots": 3 * 8 * TEMPORAL_SLOTS,
        "fixed_layer_objects": 3 * 8,
    });
    println!("{snapshot}");
}

fn run_duration(duration: Duration, report_every: Duration) -> SoakReceipt {
    let mut soak = SyntheticSoak::new(SOAK_SEED);
    let started = Instant::now();
    let deadline = started + duration;
    let mut next_report = started + report_every;
    while Instant::now() < deadline {
        soak.step();
        let now = Instant::now();
        if now >= next_report {
            print_interval(&soak, now.duration_since(started));
            next_report = now + report_every;
        }
    }
    soak.receipt()
}

fn parse_positive(value: Option<String>, flag: &str) -> Result<u64, String> {
    let value = value.ok_or_else(|| format!("{flag} requires a value"))?;
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("{flag} requires a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{flag} requires a positive integer"));
    }
    Ok(parsed)
}

fn main() -> Result<(), String> {
    let mut duration_seconds = None;
    let mut iterations = None;
    let mut report_seconds = DEFAULT_REPORT_SECONDS;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--duration-seconds" => {
                duration_seconds = Some(parse_positive(args.next(), "--duration-seconds")?);
            }
            "--iterations" => {
                iterations = Some(parse_positive(args.next(), "--iterations")?);
            }
            "--report-every-seconds" => {
                report_seconds = parse_positive(args.next(), "--report-every-seconds")?;
            }
            _ => return Err(format!("unknown argument '{argument}'")),
        }
    }
    if duration_seconds.is_some() && iterations.is_some() {
        return Err("choose --duration-seconds or --iterations, not both".to_owned());
    }

    let receipt = if let Some(iterations) = iterations {
        run_iterations(iterations)
    } else {
        run_duration(
            Duration::from_secs(duration_seconds.unwrap_or(DEFAULT_SECONDS)),
            Duration::from_secs(report_seconds),
        )
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("serialize soak receipt: {error}"))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_seed_replays_exactly_and_visits_1_3_8_layer_scenarios() {
        let first = run_iterations(SCENARIO_QUANTUM * 4);
        let second = run_iterations(SCENARIO_QUANTUM * 4);
        assert_eq!(first, second);
        assert!(first
            .scenarios
            .iter()
            .all(|scenario| scenario.iterations > 0));
        assert_eq!(
            first.scenarios.map(|scenario| scenario.active_layers),
            LAYER_COUNTS
        );
    }

    #[test]
    fn hundred_thousand_steps_keep_fixed_storage_and_drain_every_publication() {
        let receipt = run_iterations(100_000);
        assert_eq!(receipt.fixed_temporal_slots, 576);
        assert_eq!(receipt.fixed_layer_objects, 24);
        assert!(receipt.max_pending_publication_bound <= 9);
        assert_eq!(receipt.pending_publications_after_drain, 0);
        assert!(receipt
            .scenarios
            .iter()
            .all(|scenario| scenario.stale_publications_refused > 0));
    }
}
