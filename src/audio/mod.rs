//! Audio input analysis: a modulation source, not an effect.
//!
//! A cpal input stream mixes incoming frames to mono into a shared buffer.
//! Each render frame, `analyze` runs a windowed FFT over the newest samples
//! and produces normalized levels: overall RMS, three frequency bands, and
//! a transient "onset" envelope (spectral flux with instant attack and
//! exponential decay — the shape of a kick drum, not its volume).
//!
//! All outputs are 0..1, adaptively normalized against a slowly-decaying
//! running peak, so the matrix behaves the same for a whisper-quiet line-in
//! and a hot club feed. These values feed `ModMatrix` exactly like LFOs do.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

const FFT_SIZE: usize = 1024;
/// Keep a little more than one FFT window buffered.
const MAX_BUFFERED: usize = FFT_SIZE * 4;

/// Band edges in Hz.
const BASS_HI: f32 = 250.0;
const MID_HI: f32 = 2000.0;
const HIGH_HI: f32 = 8000.0;

/// Normalized audio levels, all 0..1.
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioLevels {
    pub level: f32,
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    pub onset: f32,
    /// Spectral centroid: where the energy lives. A pure sine bass reads
    /// low; a bright saw lead or hi-hats read high. Waveform *character*,
    /// independent of loudness.
    pub bright: f32,
    /// Spectral flatness: tonal (0) vs noisy (1). A sung note or synth pad
    /// reads low; snares, cymbals, and static read high.
    pub noise: f32,
}

struct SharedAudio {
    samples: Mutex<VecDeque<f32>>,
    sample_rate: AtomicU32,
}

/// Adaptive normalizer: tracks a slowly-decaying running peak.
struct AutoNorm {
    peak: f32,
}

impl AutoNorm {
    fn new() -> Self {
        Self { peak: 1e-6 }
    }

    fn norm(&mut self, value: f32) -> f32 {
        self.peak = (self.peak * 0.998).max(value).max(1e-6);
        (value / self.peak).clamp(0.0, 1.0)
    }
}

/// Pure spectral-character math: given accumulated magnitude sums over the
/// analysis band, return (centroid in Hz, flatness 0..1), or None for
/// silence. Kept free of state so it can be tested with synthetic spectra.
fn spectral_character(
    mag_sum: f32,
    weighted_hz: f32,
    log_sum: f32,
    bins: u32,
) -> Option<(f32, f32)> {
    if mag_sum <= 1e-6 || bins == 0 {
        return None;
    }
    let centroid = weighted_hz / mag_sum;
    let geo_mean = (log_sum / bins as f32).exp();
    let arith_mean = mag_sum / bins as f32;
    let flatness = (geo_mean / arith_mean.max(1e-9)).clamp(0.0, 1.0);
    Some((centroid, flatness))
}

pub struct AudioAnalyzer {
    shared: Arc<SharedAudio>,
    stream: Option<cpal::Stream>,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    prev_mags: Vec<f32>,
    onset_env: f32,
    norm_level: AutoNorm,
    norm_bass: AutoNorm,
    norm_mid: AutoNorm,
    norm_high: AutoNorm,
    norm_flux: AutoNorm,
    norm_bright: AutoNorm,
    smoothed: AudioLevels,
    pub device_name: String,
    pub error: String,
}

impl AudioAnalyzer {
    pub fn new() -> Self {
        let fft = FftPlanner::new().plan_fft_forward(FFT_SIZE);
        let window = (0..FFT_SIZE)
            .map(|i| {
                let x = i as f32 / (FFT_SIZE - 1) as f32;
                0.5 - 0.5 * (std::f32::consts::TAU * x).cos()
            })
            .collect();
        Self {
            shared: Arc::new(SharedAudio {
                samples: Mutex::new(VecDeque::with_capacity(MAX_BUFFERED)),
                sample_rate: AtomicU32::new(48000),
            }),
            stream: None,
            fft,
            window,
            prev_mags: vec![0.0; FFT_SIZE / 2],
            onset_env: 0.0,
            norm_level: AutoNorm::new(),
            norm_bass: AutoNorm::new(),
            norm_mid: AutoNorm::new(),
            norm_high: AutoNorm::new(),
            norm_flux: AutoNorm::new(),
            norm_bright: AutoNorm::new(),
            smoothed: AudioLevels::default(),
            device_name: String::new(),
            error: String::new(),
        }
    }

    pub fn is_running(&self) -> bool {
        self.stream.is_some()
    }

    /// Open the default input device and start capturing. Failure is soft:
    /// the error is recorded for the UI and every level reads 0.
    pub fn start(&mut self) {
        if self.stream.is_some() {
            return;
        }
        self.error.clear();

        let host = cpal::default_host();
        let Some(device) = host.default_input_device() else {
            self.error = "no audio input device".to_string();
            return;
        };
        self.device_name = device.name().unwrap_or_else(|_| "unknown".to_string());

        let config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                self.error = format!("input config: {e}");
                return;
            }
        };
        let sample_rate = config.sample_rate();
        let channels = config.channels() as usize;
        self.shared.sample_rate.store(sample_rate.0, Ordering::Relaxed);

        let shared = self.shared.clone();
        let push = move |mono: &mut dyn Iterator<Item = f32>| {
            if let Ok(mut buf) = shared.samples.lock() {
                for s in mono {
                    buf.push_back(s);
                }
                while buf.len() > MAX_BUFFERED {
                    buf.pop_front();
                }
            }
        };

        let err_fn = |e| log::warn!("audio stream error: {e}");
        let stream_result = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    let mut mono = data.chunks(channels).map(|f| {
                        f.iter().sum::<f32>() / channels as f32
                    });
                    push(&mut mono);
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    let mut mono = data.chunks(channels).map(|f| {
                        f.iter().map(|&s| s as f32 / i16::MAX as f32).sum::<f32>()
                            / channels as f32
                    });
                    push(&mut mono);
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                &config.into(),
                move |data: &[u16], _| {
                    let mut mono = data.chunks(channels).map(|f| {
                        f.iter()
                            .map(|&s| s as f32 / u16::MAX as f32 * 2.0 - 1.0)
                            .sum::<f32>()
                            / channels as f32
                    });
                    push(&mut mono);
                },
                err_fn,
                None,
            ),
            other => {
                self.error = format!("unsupported sample format: {other:?}");
                return;
            }
        };

        match stream_result {
            Ok(stream) => {
                if let Err(e) = stream.play() {
                    self.error = format!("stream play: {e}");
                    return;
                }
                log::info!("Audio input: {} @ {} Hz", self.device_name, sample_rate.0);
                self.stream = Some(stream);
            }
            Err(e) => {
                self.error = format!("stream build: {e}");
            }
        }
    }

    pub fn stop(&mut self) {
        self.stream = None;
        self.smoothed = AudioLevels::default();
        self.onset_env = 0.0;
    }

    /// Analyze the newest buffered samples. Call once per render frame.
    pub fn analyze(&mut self, gain: f32) -> AudioLevels {
        if self.stream.is_none() {
            return AudioLevels::default();
        }

        // Copy the newest FFT_SIZE samples out of the shared buffer.
        let mut frame = [0.0f32; FFT_SIZE];
        let got = {
            let Ok(buf) = self.shared.samples.lock() else {
                return self.smoothed;
            };
            if buf.len() < FFT_SIZE {
                false
            } else {
                let start = buf.len() - FFT_SIZE;
                for (i, s) in buf.iter().skip(start).enumerate() {
                    frame[i] = *s;
                }
                true
            }
        };
        if !got {
            // Not enough audio yet; decay toward silence.
            self.smoothed.level *= 0.9;
            self.smoothed.bass *= 0.9;
            self.smoothed.mid *= 0.9;
            self.smoothed.high *= 0.9;
            self.smoothed.bright *= 0.9;
            self.smoothed.noise *= 0.9;
            self.onset_env *= 0.85;
            self.smoothed.onset = self.onset_env;
            return self.smoothed;
        }

        let gain = gain.clamp(0.0, 8.0);

        // RMS level.
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / FFT_SIZE as f32).sqrt();

        // Windowed FFT.
        let mut spectrum: Vec<Complex<f32>> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(s, w)| Complex::new(s * w, 0.0))
            .collect();
        self.fft.process(&mut spectrum);

        let sample_rate = self.shared.sample_rate.load(Ordering::Relaxed) as f32;
        let hz_per_bin = sample_rate / FFT_SIZE as f32;
        let half = FFT_SIZE / 2;

        let mut bass = 0.0f32;
        let mut mid = 0.0f32;
        let mut high = 0.0f32;
        let mut flux = 0.0f32;
        let (mut n_bass, mut n_mid, mut n_high) = (0u32, 0u32, 0u32);
        // Spectral character accumulators (centroid + flatness).
        let mut mag_sum = 0.0f32;
        let mut weighted_hz = 0.0f32;
        let mut log_sum = 0.0f32;
        let mut char_bins = 0u32;

        for i in 1..half {
            let mag = spectrum[i].norm() / FFT_SIZE as f32;
            let hz = i as f32 * hz_per_bin;
            if hz < BASS_HI {
                bass += mag;
                n_bass += 1;
            } else if hz < MID_HI {
                mid += mag;
                n_mid += 1;
            } else if hz < HIGH_HI {
                high += mag;
                n_high += 1;
            }
            if hz < HIGH_HI {
                mag_sum += mag;
                weighted_hz += mag * hz;
                log_sum += (mag + 1e-9).ln();
                char_bins += 1;
            }
            flux += (mag - self.prev_mags[i]).max(0.0);
            self.prev_mags[i] = mag;
        }
        if n_bass > 0 {
            bass /= n_bass as f32;
        }
        if n_mid > 0 {
            mid /= n_mid as f32;
        }
        if n_high > 0 {
            high /= n_high as f32;
        }

        // Spectral character: centroid (brightness) and flatness (noisiness).
        // These describe *what kind* of sound is playing, not how loud —
        // gain is deliberately not applied, and silence decays them to 0.
        let (bright_raw, noise_raw) =
            match spectral_character(mag_sum, weighted_hz, log_sum, char_bins) {
                Some((centroid_hz, flatness)) => (self.norm_bright.norm(centroid_hz), flatness),
                None => (0.0, 0.0),
            };

        // Normalize adaptively, apply gain, smooth (fast attack, soft release).
        let raw = AudioLevels {
            level: (self.norm_level.norm(rms) * gain).clamp(0.0, 1.0),
            bass: (self.norm_bass.norm(bass) * gain).clamp(0.0, 1.0),
            mid: (self.norm_mid.norm(mid) * gain).clamp(0.0, 1.0),
            high: (self.norm_high.norm(high) * gain).clamp(0.0, 1.0),
            onset: 0.0,
            bright: bright_raw,
            noise: noise_raw,
        };

        let smooth = |old: f32, new: f32| {
            if new > old {
                new
            } else {
                old * 0.7 + new * 0.3
            }
        };
        self.smoothed.level = smooth(self.smoothed.level, raw.level);
        self.smoothed.bass = smooth(self.smoothed.bass, raw.bass);
        self.smoothed.mid = smooth(self.smoothed.mid, raw.mid);
        self.smoothed.high = smooth(self.smoothed.high, raw.high);
        // Character features smooth both directions — they describe timbre,
        // not hits, so they should glide rather than snap.
        self.smoothed.bright = self.smoothed.bright * 0.8 + raw.bright * 0.2;
        self.smoothed.noise = self.smoothed.noise * 0.8 + raw.noise * 0.2;

        // Onset: instant attack, exponential decay.
        let flux_n = (self.norm_flux.norm(flux) * gain).clamp(0.0, 1.0);
        self.onset_env = (self.onset_env * 0.85).max(flux_n);
        self.smoothed.onset = self.onset_env;

        self.smoothed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accumulate(mags: &[(f32, f32)]) -> (f32, f32, f32, u32) {
        // mags: (hz, magnitude) pairs standing in for FFT bins.
        let mut mag_sum = 0.0;
        let mut weighted = 0.0;
        let mut log_sum = 0.0;
        for &(hz, mag) in mags {
            mag_sum += mag;
            weighted += mag * hz;
            log_sum += (mag + 1e-9_f32).ln();
        }
        (mag_sum, weighted, log_sum, mags.len() as u32)
    }

    /// A pure tone reads tonal (low flatness) with the centroid at its
    /// frequency; white noise reads flat (≈1); silence reads None.
    #[test]
    fn spectral_character_discriminates_waveforms() {
        // 440 Hz sine: one hot bin among quiet ones.
        let mut sine = vec![(110.0, 1e-6), (220.0, 1e-6), (880.0, 1e-6), (1760.0, 1e-6)];
        sine.push((440.0, 0.2));
        let (m, w, l, n) = accumulate(&sine);
        let (centroid, flatness) = spectral_character(m, w, l, n).unwrap();
        assert!((centroid - 440.0).abs() < 1.0, "centroid {centroid} ≈ 440");
        assert!(flatness < 0.1, "sine flatness {flatness} should be near 0");

        // White noise: equal energy everywhere.
        let noise: Vec<(f32, f32)> = (1..100).map(|i| (i as f32 * 80.0, 0.01)).collect();
        let (m, w, l, n) = accumulate(&noise);
        let (_, flatness) = spectral_character(m, w, l, n).unwrap();
        assert!(flatness > 0.95, "noise flatness {flatness} should be near 1");

        // Bright saw vs dark sine: centroid orders them correctly.
        let dark = vec![(100.0, 0.2), (200.0, 0.1), (300.0, 0.05)];
        let bright = vec![(100.0, 0.05), (2000.0, 0.1), (6000.0, 0.2)];
        let (m, w, l, n) = accumulate(&dark);
        let (dark_c, _) = spectral_character(m, w, l, n).unwrap();
        let (m, w, l, n) = accumulate(&bright);
        let (bright_c, _) = spectral_character(m, w, l, n).unwrap();
        assert!(bright_c > dark_c * 5.0, "bright {bright_c} >> dark {dark_c}");

        // Silence gates to None.
        assert!(spectral_character(0.0, 0.0, 0.0, 100).is_none());
    }
}
