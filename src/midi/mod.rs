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
            shared: Arc::new(MidiShared {
                cc_values: std::array::from_fn(|_| AtomicU32::new(0)),
                last_cc: AtomicU32::new(0),
                clock_pulses: std::sync::atomic::AtomicU64::new(0),
                clock_interval_us: std::sync::atomic::AtomicU64::new(0),
                clock_prev_ts: std::sync::atomic::AtomicU64::new(0),
            }),
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
                let Some(&status) = message.first() else {
                    return;
                };
                match status {
                    // Control Change on any channel: [0xBn, cc, value]
                    s if s & 0xF0 == 0xB0 && message.len() >= 3 => {
                        let cc = (message[1] & 0x7F) as usize;
                        let value = (message[2] & 0x7F) as f32 / 127.0;
                        shared.cc_values[cc].store(value.to_bits(), Ordering::Relaxed);
                        shared.last_cc.store(cc as u32 + 1, Ordering::Relaxed);
                    }
                    // Timing Clock: 24 pulses per quarter note
                    0xF8 => {
                        let prev = shared.clock_prev_ts.swap(timestamp, Ordering::Relaxed);
                        if prev > 0 && timestamp > prev {
                            let dt = (timestamp - prev) as f64;
                            // Sane pulse interval: 8.3ms (300bpm) .. 83ms (30bpm)
                            if (2_000.0..=100_000.0).contains(&dt) {
                                let old = f64::from_bits(
                                    shared.clock_interval_us.load(Ordering::Relaxed),
                                );
                                let ema = if old > 0.0 { old * 0.9 + dt * 0.1 } else { dt };
                                shared
                                    .clock_interval_us
                                    .store(ema.to_bits(), Ordering::Relaxed);
                            }
                        }
                        shared.clock_pulses.fetch_add(1, Ordering::Relaxed);
                    }
                    // Start: rewind to beat zero. (Continue 0xFB resumes without reset.)
                    0xFA => {
                        shared.clock_pulses.store(0, Ordering::Relaxed);
                        shared.clock_prev_ts.store(0, Ordering::Relaxed);
                    }
                    _ => {}
                }
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
        let pulses = self.shared.clock_pulses.load(Ordering::Relaxed);
        let now = std::time::Instant::now();
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
