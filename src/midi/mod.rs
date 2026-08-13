//! MIDI input: hardware knobs and faders as modulation sources.
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

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use midir::{Ignore, MidiInput, MidiInputConnection};

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
}

impl MidiShared {
    fn new() -> Self {
        Self {
            cc_values: std::array::from_fn(|_| AtomicU32::new(0)),
            last_cc: AtomicU32::new(0),
            clock_pulses: std::sync::atomic::AtomicU64::new(0),
            clock_interval_us: std::sync::atomic::AtomicU64::new(0),
            clock_prev_ts: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Parse one complete MIDI message from the input callback.
    ///
    /// MIDI Learn deliberately observes Control Change messages only. Notes,
    /// pitch bend, aftertouch, and transport messages must never overwrite an
    /// armed CC binding.
    fn handle_message(&self, timestamp: u64, message: &[u8]) {
        let Some(&status) = message.first() else {
            return;
        };
        match status {
            // Control Change on any channel: [0xBn, cc, value]
            s if s & 0xF0 == 0xB0 && message.len() >= 3 => {
                let cc = (message[1] & 0x7F) as usize;
                let value = (message[2] & 0x7F) as f32 / 127.0;
                self.cc_values[cc].store(value.to_bits(), Ordering::Relaxed);
                self.last_cc.store(cc as u32 + 1, Ordering::Relaxed);
            }
            // Timing Clock: 24 pulses per quarter note.
            0xF8 => {
                let prev = self.clock_prev_ts.swap(timestamp, Ordering::Relaxed);
                if prev > 0 && timestamp > prev {
                    let dt = (timestamp - prev) as f64;
                    if (2_000.0..=100_000.0).contains(&dt) {
                        let old = f64::from_bits(self.clock_interval_us.load(Ordering::Relaxed));
                        let ema = if old > 0.0 { old * 0.9 + dt * 0.1 } else { dt };
                        self.clock_interval_us
                            .store(ema.to_bits(), Ordering::Relaxed);
                    }
                }
                self.clock_pulses.fetch_add(1, Ordering::Relaxed);
            }
            // Start: rewind to beat zero. (Continue 0xFB resumes without reset.)
            0xFA => {
                self.clock_pulses.store(0, Ordering::Relaxed);
                self.clock_prev_ts.store(0, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

pub struct MidiEngine {
    conn: Option<MidiInputConnection<()>>,
    shared: Arc<MidiShared>,
    pub port_name: String,
    pub error: String,
    /// Pulse count at the last `clock_state` call, to detect activity.
    seen_pulses: u64,
    /// When the pulse count last advanced.
    last_advance: std::time::Instant,
}

impl MidiEngine {
    pub fn new() -> Self {
        Self {
            conn: None,
            shared: Arc::new(MidiShared::new()),
            port_name: String::new(),
            error: String::new(),
            seen_pulses: 0,
            last_advance: std::time::Instant::now(),
        }
    }

    pub fn is_running(&self) -> bool {
        self.conn.is_some()
    }

    /// Connect to the first available MIDI input port. Failure is soft:
    /// the error is recorded for the UI and slot values read 0.
    pub fn start(&mut self) {
        if self.conn.is_some() {
            return;
        }
        self.error.clear();

        let mut midi_in = match MidiInput::new("collide-o-scope") {
            Ok(m) => m,
            Err(e) => {
                self.error = format!("midi init: {e}");
                return;
            }
        };
        midi_in.ignore(Ignore::None);

        let ports = midi_in.ports();
        let Some(port) = ports.first() else {
            self.error = "no MIDI input ports".to_string();
            return;
        };
        self.port_name = midi_in
            .port_name(port)
            .unwrap_or_else(|_| "unknown".to_string());

        let shared = self.shared.clone();
        match midi_in.connect(
            port,
            "collide-in",
            move |timestamp, message, _| {
                shared.handle_message(timestamp, message);
            },
            (),
        ) {
            Ok(conn) => {
                log::info!("MIDI input: {}", self.port_name);
                self.conn = Some(conn);
            }
            Err(e) => {
                self.error = format!("midi connect: {e}");
            }
        }
    }

    pub fn stop(&mut self) {
        self.conn = None;
        for v in self.shared.cc_values.iter() {
            v.store(0, Ordering::Relaxed);
        }
        self.shared.last_cc.store(0, Ordering::Relaxed);
        self.shared.clock_pulses.store(0, Ordering::Relaxed);
        self.shared.clock_interval_us.store(0, Ordering::Relaxed);
        self.shared.clock_prev_ts.store(0, Ordering::Relaxed);
        self.seen_pulses = 0;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const PULSE_US_120_BPM: u64 = 20_833;

    fn feed_120_bpm_quarter(engine: &MidiEngine) {
        let first_timestamp = 1_000_000;
        for pulse in 0..24 {
            engine
                .shared
                .handle_message(first_timestamp + pulse * PULSE_US_120_BPM, &[0xF8]);
        }
    }

    #[test]
    fn learn_accepts_control_change_on_any_channel() {
        let engine = MidiEngine::new();

        engine.shared.handle_message(0, &[0xB7, 74, 100]);

        assert_eq!(engine.take_last_cc(), Some(74));
        assert!((engine.cc_value(74) - 100.0 / 127.0).abs() < f32::EPSILON);
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
        feed_120_bpm_quarter(&engine);

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
        feed_120_bpm_quarter(&engine);
        assert!(engine.clock_state_at(sampled_at).is_some());

        engine.shared.handle_message(2_000_000, &[0xFA]);

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
        feed_120_bpm_quarter(&engine);
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
}
