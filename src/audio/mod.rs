//! Audio input analysis: a modulation source, not an effect.
//!
//! A cpal input stream mixes incoming frames to mono into a shared buffer.
//! Each render frame, `analyze` runs a windowed FFT over the newest samples
//! and produces normalized levels: overall RMS, three to eight frequency bands, and
//! a transient "onset" envelope (spectral flux with instant attack and
//! exponential decay — the shape of a kick drum, not its volume).
//!
//! All outputs are 0..1, adaptively normalized against a slowly-decaying
//! running peak, so the matrix behaves the same for a whisper-quiet line-in
//! and a hot club feed. These values feed `ModMatrix` exactly like LFOs do.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

const FFT_SIZE: usize = 1024;
/// Keep a little more than one FFT window buffered.
const MAX_BUFFERED: usize = FFT_SIZE * 4;
/// A live CPAL stream should deliver callbacks continuously, including for
/// digital silence. Treat a longer gap as a disconnected/stalled device.
const STALE_STREAM_TIMEOUT: Duration = Duration::from_secs(2);

pub const AUDIO_SPECTRUM_BINS: usize = 32;
pub const MIN_AUDIO_BANDS: usize = 3;
pub const MAX_AUDIO_BANDS: usize = 8;
pub const MAX_AUDIO_CROSSOVERS: usize = MAX_AUDIO_BANDS - 1;
pub const MIN_BAND_EDGE_HZ: f32 = 20.0;
pub const MAX_BAND_EDGE_HZ: f32 = 20_000.0;
const MIN_BAND_GAP_HZ: f32 = 10.0;

/// Canonical audio-band layout.
///
/// `count` bands require `count - 1` crossovers. `ceiling_hz` is deliberately
/// separate: bins at or above it are excluded from band energy and spectral
/// character analysis. This distinction preserves the historical default,
/// where 250 Hz and 2 kHz were crossovers and 8 kHz was the analysis ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioBandConfig {
    count: usize,
    crossovers: [f32; MAX_AUDIO_CROSSOVERS],
    ceiling_hz: f32,
}

impl Default for AudioBandConfig {
    fn default() -> Self {
        Self::new(3, &[250.0, 2000.0], 8000.0)
    }
}

impl AudioBandConfig {
    pub fn new(count: usize, crossovers: &[f32], ceiling_hz: f32) -> Self {
        let count = count.clamp(MIN_AUDIO_BANDS, MAX_AUDIO_BANDS);
        let ceiling_hz = if ceiling_hz.is_finite() {
            ceiling_hz
        } else {
            8000.0
        }
        .clamp(
            MIN_BAND_EDGE_HZ + MIN_BAND_GAP_HZ * (count - 1) as f32,
            MAX_BAND_EDGE_HZ,
        );

        let suggested = Self::suggested_crossovers(count, ceiling_hz);
        let active = count - 1;
        let mut values = [0.0; MAX_AUDIO_CROSSOVERS];
        for i in 0..active {
            values[i] = crossovers
                .get(i)
                .copied()
                .filter(|value| value.is_finite())
                .unwrap_or(suggested[i])
                .clamp(MIN_BAND_EDGE_HZ, ceiling_hz);
        }
        values[..active].sort_by(f32::total_cmp);

        // Enforce an ordered, non-collapsed layout while retaining as much of
        // the performer's requested shape as possible.
        for i in 0..active {
            let lower = if i == 0 {
                MIN_BAND_EDGE_HZ
            } else {
                values[i - 1] + MIN_BAND_GAP_HZ
            };
            let remaining = active - i;
            let upper = ceiling_hz - MIN_BAND_GAP_HZ * remaining as f32;
            values[i] = values[i].clamp(lower, upper.max(lower));
        }

        Self {
            count,
            crossovers: values,
            ceiling_hz,
        }
    }

    fn suggested_crossovers(count: usize, ceiling_hz: f32) -> [f32; MAX_AUDIO_CROSSOVERS] {
        let mut values = [0.0; MAX_AUDIO_CROSSOVERS];
        values[0] = 250.0_f32.min(ceiling_hz - MIN_BAND_GAP_HZ * 2.0);
        values[1] = 2000.0_f32.min(ceiling_hz - MIN_BAND_GAP_HZ);

        // Additional default bands subdivide the former high band on a log
        // scale. The first three bands therefore remain exactly compatible.
        let additional = count.saturating_sub(3);
        if additional > 0 {
            let start = values[1].max(MIN_BAND_EDGE_HZ);
            let ratio = (ceiling_hz / start).max(1.0);
            for i in 0..additional {
                let t = (i + 1) as f32 / (additional + 1) as f32;
                values[i + 2] = start * ratio.powf(t);
            }
        }
        values
    }

    pub fn count(self) -> usize {
        self.count
    }

    pub fn crossovers(&self) -> &[f32] {
        &self.crossovers[..self.count - 1]
    }

    pub fn ceiling_hz(self) -> f32 {
        self.ceiling_hz
    }

    pub fn with_count(self, count: usize) -> Self {
        let count = count.clamp(MIN_AUDIO_BANDS, MAX_AUDIO_BANDS);
        if count <= self.count {
            return Self::new(count, &self.crossovers()[..count - 1], self.ceiling_hz);
        }

        let mut crossovers = self.crossovers().to_vec();
        let additions = count - self.count;
        let start = crossovers
            .last()
            .copied()
            .unwrap_or(MIN_BAND_EDGE_HZ)
            .max(MIN_BAND_EDGE_HZ);
        let ratio = (self.ceiling_hz / start).max(1.0);
        for index in 1..=additions {
            let t = index as f32 / (additions + 1) as f32;
            crossovers.push(start * ratio.powf(t));
        }
        Self::new(count, &crossovers, self.ceiling_hz)
    }

    fn band_index(self, hz: f32) -> Option<usize> {
        if !hz.is_finite() || hz < 0.0 || hz >= self.ceiling_hz {
            return None;
        }
        Some(
            self.crossovers()
                .iter()
                .position(|edge| hz < *edge)
                .unwrap_or(self.count - 1),
        )
    }
}

fn accumulate_band_magnitude(
    config: AudioBandConfig,
    hz: f32,
    magnitude: f32,
    sums: &mut [f32; MAX_AUDIO_BANDS],
    counts: &mut [u32; MAX_AUDIO_BANDS],
) {
    if !magnitude.is_finite() || magnitude < 0.0 {
        return;
    }
    if let Some(band) = config.band_index(hz) {
        sums[band] += magnitude;
        counts[band] = counts[band].saturating_add(1);
    }
}

#[inline]
fn sanitize_input_sample(sample: f32) -> f32 {
    if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Runtime-configurable audio band boundaries, in Hz.
///
/// Construction and analyzer setters sanitize non-finite, out-of-order, and
/// out-of-range values. Keeping that policy here gives patches, the native
/// UI, and the browser panel one canonical source of truth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioBandEdges {
    pub bass_hz: f32,
    pub mid_hz: f32,
    pub high_hz: f32,
}

impl Default for AudioBandEdges {
    fn default() -> Self {
        Self {
            bass_hz: 250.0,
            mid_hz: 2000.0,
            high_hz: 8000.0,
        }
    }
}

impl AudioBandEdges {
    pub fn new(bass_hz: f32, mid_hz: f32, high_hz: f32) -> Self {
        let defaults = Self::default();
        let finite_or = |value: f32, fallback: f32| {
            if value.is_finite() {
                value
            } else {
                fallback
            }
        };
        let mut edges = [
            finite_or(bass_hz, defaults.bass_hz).clamp(MIN_BAND_EDGE_HZ, MAX_BAND_EDGE_HZ),
            finite_or(mid_hz, defaults.mid_hz).clamp(MIN_BAND_EDGE_HZ, MAX_BAND_EDGE_HZ),
            finite_or(high_hz, defaults.high_hz).clamp(MIN_BAND_EDGE_HZ, MAX_BAND_EDGE_HZ),
        ];
        // Treat the three inputs as boundaries even if a malformed patch or a
        // racing UI update supplied them out of order, then enforce a small
        // gap so every band remains meaningful.
        edges.sort_by(f32::total_cmp);
        let bass_hz = edges[0].clamp(MIN_BAND_EDGE_HZ, MAX_BAND_EDGE_HZ - MIN_BAND_GAP_HZ * 2.0);
        let mid_hz = edges[1].clamp(
            bass_hz + MIN_BAND_GAP_HZ,
            MAX_BAND_EDGE_HZ - MIN_BAND_GAP_HZ,
        );
        let high_hz = edges[2].clamp(mid_hz + MIN_BAND_GAP_HZ, MAX_BAND_EDGE_HZ);
        Self {
            bass_hz,
            mid_hz,
            high_hz,
        }
    }
}

/// Normalized audio levels, all 0..1.
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioLevels {
    pub level: f32,
    /// Configurable band outputs. Entries at or above the active band count
    /// are always zero.
    pub bands: [f32; MAX_AUDIO_BANDS],
    /// Legacy aliases for bands 1, 2, and 3. These stay in the protocol and
    /// route grammar so old patches continue to behave identically.
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
    /// Monotonically increasing count used to distinguish a fresh FFT window
    /// from a dead stream whose final samples are still in the ring.
    sample_count: AtomicU64,
    /// Runtime failures arrive on cpal's callback thread and are consumed by
    /// the render thread on its next analysis pass.
    stream_error: Mutex<Option<String>>,
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

/// Map a physical frequency to one of the fixed display bins. Log spacing
/// preserves useful bass detail without giving the browser hundreds of FFT
/// values to paint on every state update.
fn display_bin_for_hz(hz: f32, ceiling_hz: f32) -> Option<usize> {
    if !hz.is_finite()
        || !ceiling_hz.is_finite()
        || hz < MIN_BAND_EDGE_HZ
        || hz > ceiling_hz
        || ceiling_hz <= MIN_BAND_EDGE_HZ
    {
        return None;
    }
    let span = (ceiling_hz / MIN_BAND_EDGE_HZ).ln();
    let position = (hz / MIN_BAND_EDGE_HZ).ln() / span;
    Some(((position * AUDIO_SPECTRUM_BINS as f32) as usize).min(AUDIO_SPECTRUM_BINS - 1))
}

fn update_display_spectrum(
    output: &mut [f32; AUDIO_SPECTRUM_BINS],
    running_peak: &mut f32,
    sums: &[f32; AUDIO_SPECTRUM_BINS],
    counts: &[u16; AUDIO_SPECTRUM_BINS],
) {
    let mut averages = [0.0f32; AUDIO_SPECTRUM_BINS];
    let mut frame_peak = 0.0f32;
    for i in 0..AUDIO_SPECTRUM_BINS {
        let average = if counts[i] > 0 {
            sums[i] / counts[i] as f32
        } else {
            0.0
        };
        averages[i] = if average.is_finite() {
            average.max(0.0)
        } else {
            0.0
        };
        frame_peak = frame_peak.max(averages[i]);
    }

    *running_peak = (*running_peak * 0.96).max(frame_peak).max(1e-9);
    for i in 0..AUDIO_SPECTRUM_BINS {
        let normalized = (averages[i] / *running_peak).clamp(0.0, 1.0);
        output[i] = if normalized > output[i] {
            normalized
        } else {
            output[i] * 0.72 + normalized * 0.28
        };
    }
}

pub struct AudioAnalyzer {
    shared: Arc<SharedAudio>,
    stream: Option<cpal::Stream>,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    prev_mags: Vec<f32>,
    onset_env: f32,
    norm_level: AutoNorm,
    norm_bands: [AutoNorm; MAX_AUDIO_BANDS],
    norm_flux: AutoNorm,
    smoothed: AudioLevels,
    band_config: AudioBandConfig,
    display_spectrum: [f32; AUDIO_SPECTRUM_BINS],
    spectrum_peak: f32,
    last_sample_count: u64,
    last_sample_at: Instant,
    /// Preference used to open the current/most recently attempted stream.
    /// Empty means the system default, matching `ModMatrix::audio_device`.
    requested_device: String,
    using_device_fallback: bool,
    /// Name reported by CPAL for the device backing the live stream. This is
    /// intentionally distinct from `requested_device` when fallback occurs.
    pub device_name: String,
    pub error: String,
    /// Input device names, refreshed on every start (for the UI select).
    pub devices: Vec<String>,
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
        let mut analyzer = Self {
            shared: Arc::new(SharedAudio {
                samples: Mutex::new(VecDeque::with_capacity(MAX_BUFFERED)),
                sample_rate: AtomicU32::new(48000),
                sample_count: AtomicU64::new(0),
                stream_error: Mutex::new(None),
            }),
            stream: None,
            fft,
            window,
            prev_mags: vec![0.0; FFT_SIZE / 2],
            onset_env: 0.0,
            norm_level: AutoNorm::new(),
            norm_bands: std::array::from_fn(|_| AutoNorm::new()),
            norm_flux: AutoNorm::new(),
            smoothed: AudioLevels::default(),
            band_config: AudioBandConfig::default(),
            display_spectrum: [0.0; AUDIO_SPECTRUM_BINS],
            spectrum_peak: 1e-9,
            last_sample_count: 0,
            last_sample_at: Instant::now(),
            requested_device: String::new(),
            using_device_fallback: false,
            device_name: String::new(),
            error: String::new(),
            devices: Vec::new(),
        };
        analyzer.refresh_devices();
        analyzer
    }

    /// Enumerate available input devices (names) into the cache.
    pub fn refresh_devices(&mut self) {
        let host = cpal::default_host();
        self.devices = host
            .input_devices()
            .map(|iter| iter.filter_map(|d| d.name().ok()).collect())
            .unwrap_or_default();
    }

    pub fn is_running(&self) -> bool {
        self.stream.is_some()
    }

    /// Whether the live stream was opened for this exact preference.
    ///
    /// This compares the requested preference rather than the active device
    /// name. If a named device is unavailable and CPAL falls back to the
    /// system default, the fallback stream still counts as running for that
    /// request and will not be reopened on every frame.
    pub fn is_running_for(&self, preferred: &str) -> bool {
        self.is_running() && self.requested_device == preferred
    }

    /// CPAL name of the device backing the live stream, or empty while stopped.
    pub fn active_device(&self) -> &str {
        if self.is_running() {
            &self.device_name
        } else {
            ""
        }
    }

    /// True only while a live stream is using the system default because its
    /// non-empty requested device name could not be found.
    pub fn is_using_device_fallback(&self) -> bool {
        self.is_running() && self.using_device_fallback
    }

    /// Current validated frequency boundaries for legacy callers.
    ///
    /// When more than three bands are active, `high_hz` is the third band's
    /// upper crossover. The configurable API is [`Self::band_config`].
    pub fn band_edges(&self) -> AudioBandEdges {
        let crossovers = self.band_config.crossovers();
        AudioBandEdges {
            bass_hz: crossovers[0],
            mid_hz: crossovers[1],
            high_hz: crossovers
                .get(2)
                .copied()
                .unwrap_or(self.band_config.ceiling_hz()),
        }
    }

    pub fn band_config(&self) -> AudioBandConfig {
        self.band_config
    }

    /// Legacy three-band setter. The third value remains the analysis ceiling.
    pub fn set_band_edges(&mut self, bass_hz: f32, mid_hz: f32, high_hz: f32) -> AudioBandEdges {
        let legacy = AudioBandEdges::new(bass_hz, mid_hz, high_hz);
        self.set_band_config(AudioBandConfig::new(
            3,
            &[legacy.bass_hz, legacy.mid_hz],
            legacy.high_hz,
        ));
        legacy
    }

    /// Replace the entire band layout atomically, resetting normalization
    /// state only when the validated layout actually changes.
    pub fn set_band_config(&mut self, config: AudioBandConfig) -> AudioBandConfig {
        if config != self.band_config {
            self.band_config = config;
            self.norm_bands = std::array::from_fn(|_| AutoNorm::new());
            self.smoothed.bands.fill(0.0);
            self.sync_legacy_bands();
            self.display_spectrum.fill(0.0);
            self.spectrum_peak = 1e-9;
        }
        self.band_config
    }

    /// A fixed-size, logarithmically spaced display spectrum. Values are
    /// normalized to 0..1 and the returned slice remains owned by the
    /// analyzer, so a render-frame read performs no allocation.
    pub fn spectrum(&self) -> &[f32; AUDIO_SPECTRUM_BINS] {
        &self.display_spectrum
    }

    /// Open an input device and start capturing. `preferred` selects a
    /// device by name; empty means the system default. Failure is soft:
    /// the error is recorded for the UI and every level reads 0.
    pub fn start(&mut self, preferred: &str) {
        if self.stream.is_some() {
            return;
        }
        self.requested_device.clear();
        self.requested_device.push_str(preferred);
        self.device_name.clear();
        self.using_device_fallback = false;
        self.error.clear();
        self.reset_analysis();
        if let Ok(mut error) = self.shared.stream_error.lock() {
            *error = None;
        }
        self.refresh_devices();

        let host = cpal::default_host();
        let (device, using_fallback) = if preferred.is_empty() {
            (host.default_input_device(), false)
        } else {
            let selected = host.input_devices().ok().and_then(|mut iter| {
                iter.find(|d| d.name().map(|n| n == preferred).unwrap_or(false))
            });
            match selected {
                Some(device) => (Some(device), false),
                None => {
                    log::warn!("Audio device '{preferred}' not found; using default");
                    (host.default_input_device(), true)
                }
            }
        };
        let Some(device) = device else {
            self.error = "no audio input device".to_string();
            return;
        };
        let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());

        let config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                self.error = format!("input config: {e}");
                return;
            }
        };
        let sample_rate = config.sample_rate();
        let channels = config.channels() as usize;
        self.shared
            .sample_rate
            .store(sample_rate.0, Ordering::Relaxed);

        let shared = self.shared.clone();
        let push = move |mono: &mut dyn Iterator<Item = f32>| {
            if let Ok(mut buf) = shared.samples.lock() {
                let mut added = 0u64;
                for s in mono {
                    buf.push_back(sanitize_input_sample(s));
                    added += 1;
                }
                while buf.len() > MAX_BUFFERED {
                    buf.pop_front();
                }
                shared.sample_count.fetch_add(added, Ordering::Relaxed);
            }
        };

        let error_shared = self.shared.clone();
        let err_fn = move |e| {
            let message = format!("audio stream error: {e}");
            log::warn!("{message}");
            if let Ok(mut slot) = error_shared.stream_error.lock() {
                *slot = Some(message);
            }
        };
        let stream_result = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    let mut mono = data
                        .chunks(channels)
                        .map(|f| f.iter().sum::<f32>() / channels as f32);
                    push(&mut mono);
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    let mut mono = data.chunks(channels).map(|f| {
                        f.iter().map(|&s| s as f32 / i16::MAX as f32).sum::<f32>() / channels as f32
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
                log::info!("Audio input: {device_name} @ {} Hz", sample_rate.0);
                self.device_name = device_name;
                self.using_device_fallback = using_fallback;
                self.stream = Some(stream);
            }
            Err(e) => {
                self.error = format!("stream build: {e}");
            }
        }
    }

    pub fn stop(&mut self) {
        self.stream = None;
        self.device_name.clear();
        self.using_device_fallback = false;
        self.reset_analysis();
    }

    /// Analyze the newest buffered samples. Call once per render frame.
    pub fn analyze(&mut self, gain: f32) -> AudioLevels {
        if self.stream.is_none() {
            return AudioLevels::default();
        }

        let runtime_error = self
            .shared
            .stream_error
            .lock()
            .ok()
            .and_then(|mut error| error.take());
        if let Some(error) = runtime_error {
            self.error = error;
            // Drop the failed stream and zero its sources immediately. The app
            // can now turn the requested enable state off instead of leaving a
            // latched FFT window driving the matrix forever.
            self.stream = None;
            self.device_name.clear();
            self.using_device_fallback = false;
            self.reset_analysis();
            return AudioLevels::default();
        }

        let sample_count = self.shared.sample_count.load(Ordering::Relaxed);
        if sample_count == self.last_sample_count {
            if self.last_sample_at.elapsed() >= STALE_STREAM_TIMEOUT {
                self.error = "audio input stopped delivering samples".to_string();
                self.stream = None;
                self.device_name.clear();
                self.using_device_fallback = false;
                self.reset_analysis();
                return AudioLevels::default();
            }
            return self.decay_toward_silence();
        }
        self.last_sample_count = sample_count;
        self.last_sample_at = Instant::now();

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
            return self.decay_toward_silence();
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

        let mut band_sums = [0.0f32; MAX_AUDIO_BANDS];
        let mut band_counts = [0u32; MAX_AUDIO_BANDS];
        let mut flux = 0.0f32;
        // Spectral character accumulators (centroid + flatness).
        let mut mag_sum = 0.0f32;
        let mut weighted_hz = 0.0f32;
        let mut log_sum = 0.0f32;
        let mut char_bins = 0u32;
        let mut display_sums = [0.0f32; AUDIO_SPECTRUM_BINS];
        let mut display_counts = [0u16; AUDIO_SPECTRUM_BINS];
        let display_ceiling = self.band_config.ceiling_hz().min(sample_rate * 0.5);

        for (i, bin) in spectrum.iter().enumerate().take(half).skip(1) {
            let mag = bin.norm() / FFT_SIZE as f32;
            if !mag.is_finite() {
                self.prev_mags[i] = 0.0;
                continue;
            }
            let hz = i as f32 * hz_per_bin;
            accumulate_band_magnitude(self.band_config, hz, mag, &mut band_sums, &mut band_counts);
            if hz < self.band_config.ceiling_hz() {
                mag_sum += mag;
                weighted_hz += mag * hz;
                log_sum += (mag + 1e-9).ln();
                char_bins += 1;
            }
            if let Some(bin) = display_bin_for_hz(hz, display_ceiling) {
                display_sums[bin] += mag;
                display_counts[bin] = display_counts[bin].saturating_add(1);
            }
            flux += (mag - self.prev_mags[i]).max(0.0);
            self.prev_mags[i] = mag;
        }
        update_display_spectrum(
            &mut self.display_spectrum,
            &mut self.spectrum_peak,
            &display_sums,
            &display_counts,
        );
        for i in 0..self.band_config.count() {
            if band_counts[i] > 0 {
                band_sums[i] /= band_counts[i] as f32;
            }
        }

        // Spectral character: centroid (brightness) and flatness (noisiness).
        // These describe *what kind* of sound is playing, not how loud —
        // gain is deliberately not applied, and silence decays them to 0.
        let (bright_raw, noise_raw) =
            match spectral_character(mag_sum, weighted_hz, log_sum, char_bins) {
                Some((centroid_hz, flatness)) => (
                    brightness_from_centroid(centroid_hz, self.band_config.ceiling_hz()),
                    flatness,
                ),
                None => (0.0, 0.0),
            };

        // Normalize adaptively, apply gain, smooth (fast attack, soft release).
        let mut normalized_bands = [0.0; MAX_AUDIO_BANDS];
        for i in 0..self.band_config.count() {
            normalized_bands[i] = (self.norm_bands[i].norm(band_sums[i]) * gain).clamp(0.0, 1.0);
        }
        let raw = AudioLevels {
            level: (self.norm_level.norm(rms) * gain).clamp(0.0, 1.0),
            bands: normalized_bands,
            bass: normalized_bands[0],
            mid: normalized_bands[1],
            high: normalized_bands[2],
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
        for i in 0..MAX_AUDIO_BANDS {
            self.smoothed.bands[i] = if i < self.band_config.count() {
                smooth(self.smoothed.bands[i], raw.bands[i])
            } else {
                0.0
            };
        }
        self.sync_legacy_bands();
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

    fn decay_toward_silence(&mut self) -> AudioLevels {
        self.smoothed.level *= 0.9;
        for band in &mut self.smoothed.bands {
            *band *= 0.9;
        }
        self.sync_legacy_bands();
        self.smoothed.bright *= 0.9;
        self.smoothed.noise *= 0.9;
        for bin in &mut self.display_spectrum {
            *bin *= 0.85;
        }
        self.onset_env *= 0.85;
        self.smoothed.onset = self.onset_env;
        self.smoothed
    }

    fn reset_analysis(&mut self) {
        if let Ok(mut samples) = self.shared.samples.lock() {
            samples.clear();
        }
        self.prev_mags.fill(0.0);
        self.onset_env = 0.0;
        self.norm_level = AutoNorm::new();
        self.norm_bands = std::array::from_fn(|_| AutoNorm::new());
        self.norm_flux = AutoNorm::new();
        self.smoothed = AudioLevels::default();
        self.display_spectrum.fill(0.0);
        self.spectrum_peak = 1e-9;
        self.last_sample_count = self.shared.sample_count.load(Ordering::Relaxed);
        self.last_sample_at = Instant::now();
    }

    fn sync_legacy_bands(&mut self) {
        self.smoothed.bass = self.smoothed.bands[0];
        self.smoothed.mid = self.smoothed.bands[1];
        self.smoothed.high = self.smoothed.bands[2];
    }
}

/// Spectral centroid already has an absolute physical scale. Normalizing it
/// against its own running maximum makes the first bass note read as fully
/// bright; map it to the configured analysis ceiling instead.
fn brightness_from_centroid(centroid_hz: f32, ceiling_hz: f32) -> f32 {
    if !centroid_hz.is_finite() || !ceiling_hz.is_finite() || ceiling_hz <= 0.0 {
        0.0
    } else {
        (centroid_hz / ceiling_hz).clamp(0.0, 1.0)
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
        assert!(
            flatness > 0.95,
            "noise flatness {flatness} should be near 1"
        );

        // Bright saw vs dark sine: centroid orders them correctly.
        let dark = vec![(100.0, 0.2), (200.0, 0.1), (300.0, 0.05)];
        let bright = vec![(100.0, 0.05), (2000.0, 0.1), (6000.0, 0.2)];
        let (m, w, l, n) = accumulate(&dark);
        let (dark_c, _) = spectral_character(m, w, l, n).unwrap();
        let (m, w, l, n) = accumulate(&bright);
        let (bright_c, _) = spectral_character(m, w, l, n).unwrap();
        assert!(
            bright_c > dark_c * 5.0,
            "bright {bright_c} >> dark {dark_c}"
        );

        // Silence gates to None.
        assert!(spectral_character(0.0, 0.0, 0.0, 100).is_none());
    }

    #[test]
    fn brightness_uses_an_absolute_frequency_scale() {
        assert!((brightness_from_centroid(100.0, 8000.0) - 0.0125).abs() < 1e-6);
        assert_eq!(brightness_from_centroid(4000.0, 8000.0), 0.5);
        assert_eq!(brightness_from_centroid(8000.0, 8000.0), 1.0);
        assert_eq!(brightness_from_centroid(16_000.0, 8000.0), 1.0);
        assert_eq!(brightness_from_centroid(4000.0, 16_000.0), 0.25);
    }

    #[test]
    fn malformed_float_samples_are_finite_and_bounded_before_feedback() {
        assert_eq!(sanitize_input_sample(f32::NAN), 0.0);
        assert_eq!(sanitize_input_sample(f32::INFINITY), 0.0);
        assert_eq!(sanitize_input_sample(f32::NEG_INFINITY), 0.0);
        assert_eq!(sanitize_input_sample(2.0), 1.0);
        assert_eq!(sanitize_input_sample(-2.0), -1.0);
        assert_eq!(sanitize_input_sample(0.25), 0.25);
    }

    #[test]
    fn band_edges_are_finite_ordered_and_bounded() {
        assert_eq!(
            AudioBandEdges::default(),
            AudioBandEdges {
                bass_hz: 250.0,
                mid_hz: 2000.0,
                high_hz: 8000.0,
            }
        );

        // Out-of-order values are treated as the three intended boundaries.
        assert_eq!(
            AudioBandEdges::new(8000.0, 250.0, 2000.0),
            AudioBandEdges::default()
        );

        let malformed = AudioBandEdges::new(f32::NAN, f32::NEG_INFINITY, f32::INFINITY);
        assert!(malformed.bass_hz.is_finite());
        assert!(malformed.mid_hz.is_finite());
        assert!(malformed.high_hz.is_finite());
        assert!(malformed.bass_hz >= MIN_BAND_EDGE_HZ);
        assert!(malformed.bass_hz < malformed.mid_hz);
        assert!(malformed.mid_hz < malformed.high_hz);
        assert!(malformed.high_hz <= MAX_BAND_EDGE_HZ);

        let collapsed = AudioBandEdges::new(99_000.0, 99_000.0, 99_000.0);
        assert_eq!(collapsed.bass_hz, MAX_BAND_EDGE_HZ - 20.0);
        assert_eq!(collapsed.mid_hz, MAX_BAND_EDGE_HZ - 10.0);
        assert_eq!(collapsed.high_hz, MAX_BAND_EDGE_HZ);
    }

    #[test]
    fn band_classification_honors_configured_boundaries() {
        let config = AudioBandConfig::new(3, &[100.0, 1000.0], 5000.0);
        assert_eq!(config.band_index(99.9), Some(0));
        assert_eq!(config.band_index(100.0), Some(1));
        assert_eq!(config.band_index(999.9), Some(1));
        assert_eq!(config.band_index(1000.0), Some(2));
        assert_eq!(config.band_index(4999.9), Some(2));
        assert_eq!(config.band_index(5000.0), None);
        assert_eq!(config.band_index(f32::NAN), None);
    }

    #[test]
    fn configurable_bands_discriminate_all_eight_regions() {
        let config = AudioBandConfig::new(
            8,
            &[100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0],
            12_800.0,
        );
        assert_eq!(config.count(), 8);
        for (index, hz) in [50.0, 150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0, 9600.0]
            .into_iter()
            .enumerate()
        {
            assert_eq!(config.band_index(hz), Some(index), "frequency {hz}");
        }
        assert_eq!(config.band_index(12_800.0), None);

        let sanitized = AudioBandConfig::new(
            99,
            &[f32::NAN, 8000.0, 100.0, 100.0, 50_000.0],
            f32::INFINITY,
        );
        assert_eq!(sanitized.count(), MAX_AUDIO_BANDS);
        assert!(sanitized
            .crossovers()
            .windows(2)
            .all(|pair| pair[0].is_finite() && pair[1] - pair[0] >= MIN_BAND_GAP_HZ));
        assert!(sanitized.crossovers().last().unwrap() < &sanitized.ceiling_hz());

        // Synthetic narrow-band peaks exercise the same accumulator used by
        // the FFT loop: each frequency energizes only its intended source.
        let mut sums = [0.0; MAX_AUDIO_BANDS];
        let mut counts = [0; MAX_AUDIO_BANDS];
        for (index, hz) in [50.0, 150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0, 9600.0]
            .into_iter()
            .enumerate()
        {
            accumulate_band_magnitude(config, hz, (index + 1) as f32, &mut sums, &mut counts);
        }
        assert_eq!(counts, [1; MAX_AUDIO_BANDS]);
        assert_eq!(sums, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn default_three_band_layout_preserves_legacy_semantics() {
        let config = AudioBandConfig::default();
        assert_eq!(config.count(), 3);
        assert_eq!(config.crossovers(), &[250.0, 2000.0]);
        assert_eq!(config.ceiling_hz(), 8000.0);

        let mut analyzer = AudioAnalyzer::new();
        let legacy = analyzer.set_band_edges(250.0, 2000.0, 8000.0);
        assert_eq!(legacy, AudioBandEdges::default());
        assert_eq!(analyzer.band_config(), config);

        let customized = AudioBandConfig::new(3, &[100.0, 5000.0], 20_000.0).with_count(5);
        assert_eq!(&customized.crossovers()[..2], &[100.0, 5000.0]);
        assert!(customized.crossovers()[2] > 5000.0);
        assert!(customized.crossovers()[3] < 20_000.0);
    }

    #[test]
    fn display_spectrum_is_log_spaced_finite_and_bounded() {
        assert_eq!(display_bin_for_hz(MIN_BAND_EDGE_HZ, 8000.0), Some(0));
        assert_eq!(display_bin_for_hz(8000.0, 8000.0), Some(31));
        assert!(
            display_bin_for_hz(200.0, 8000.0).unwrap()
                < display_bin_for_hz(2000.0, 8000.0).unwrap()
        );
        assert_eq!(display_bin_for_hz(10.0, 8000.0), None);
        assert_eq!(display_bin_for_hz(9000.0, 8000.0), None);

        let mut output = [0.0; AUDIO_SPECTRUM_BINS];
        let mut peak = 1e-9;
        let mut sums = [0.0; AUDIO_SPECTRUM_BINS];
        let mut counts = [0u16; AUDIO_SPECTRUM_BINS];
        sums[0] = 0.25;
        counts[0] = 1;
        sums[10] = 1.0;
        counts[10] = 1;
        sums[20] = f32::NAN;
        counts[20] = 1;
        sums[30] = -1.0;
        counts[30] = 1;
        update_display_spectrum(&mut output, &mut peak, &sums, &counts);

        assert_eq!(output[10], 1.0);
        assert!(output[0] > 0.0 && output[0] < output[10]);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn reset_clears_the_display_spectrum() {
        let mut analyzer = AudioAnalyzer::new();
        analyzer.display_spectrum.fill(0.75);
        analyzer.spectrum_peak = 2.0;
        analyzer.reset_analysis();
        assert!(analyzer.spectrum().iter().all(|value| *value == 0.0));
        assert_eq!(analyzer.spectrum_peak, 1e-9);
    }

    #[test]
    fn stopped_analyzer_reports_device_identity_honestly() {
        let mut analyzer = AudioAnalyzer::new();
        analyzer.requested_device = "requested input".to_string();
        analyzer.using_device_fallback = true;

        assert_eq!(analyzer.requested_device, "requested input");
        assert_eq!(analyzer.active_device(), "");
        assert!(!analyzer.is_running_for("requested input"));
        assert!(!analyzer.is_using_device_fallback());

        analyzer.stop();
        assert_eq!(analyzer.requested_device, "requested input");
        assert_eq!(analyzer.active_device(), "");
        assert!(!analyzer.is_using_device_fallback());
    }
}
