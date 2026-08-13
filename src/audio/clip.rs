//! Deterministic, frame-addressed analysis of imported audio media.
//!
//! FFmpeg decodes the first audio stream once into bounded mono `f32` PCM.
//! Analysis is intentionally pure: a timestamp selects one circular FFT
//! window and all normalization is local to that window. Consequently the
//! same clip, band layout, gain, and time always yield the same result,
//! independent of live render cadence or previous calls.

use ffmpeg_next as ffmpeg;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{
    accumulate_band_magnitude, brightness_from_centroid, display_bin_for_hz, spectral_character,
    AudioBandConfig, AudioLevels, AUDIO_SPECTRUM_BINS, FFT_SIZE, MAX_AUDIO_BANDS,
};

/// Decode at one canonical rate so analysis is identical across source codecs.
pub const CLIP_SAMPLE_RATE: u32 = 48_000;
/// Prevent hostile or accidental multi-hour files from allocating unbounded PCM.
pub const MAX_AUDIO_CLIP_SECONDS: u64 = 10 * 60;
pub const MAX_AUDIO_CLIP_SAMPLES: usize =
    CLIP_SAMPLE_RATE as usize * MAX_AUDIO_CLIP_SECONDS as usize;
const AUDIO_CLIP_DECODE_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_RESAMPLER_FLUSH_FRAMES: usize = 256;

/// Formats presented by the picker. FFmpeg probing remains authoritative, so
/// another local format with a decodable audio stream can still be accepted.
pub const AUDIO_FILE_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac"];

pub fn is_supported_audio_extension(extension: &str) -> bool {
    AUDIO_FILE_EXTENSIONS
        .iter()
        .any(|supported| extension.eq_ignore_ascii_case(supported))
}

pub fn is_supported_audio_file(path: impl AsRef<Path>) -> bool {
    path.as_ref()
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(is_supported_audio_extension)
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioClipInfo {
    pub path: String,
    pub duration_secs: f64,
    pub sample_rate: u32,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioClipAnalysis {
    pub levels: AudioLevels,
    pub spectrum: [f32; AUDIO_SPECTRUM_BINS],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioClipLoadState {
    Idle,
    Loading,
}

pub struct AudioClipLoadResult {
    pub generation: u64,
    pub path: String,
    pub result: Result<AudioClip, String>,
}

/// One-worker, latest-generation mailbox for render-thread-safe clip loads.
///
/// `request` never blocks on a previous decoder. A bounded one-slot completion
/// channel ensures an abandoned request cannot accumulate results; `poll`
/// discards completions whose generation is no longer current.
pub struct AudioClipLoader {
    generation: u64,
    requested_path: String,
    active_receiver: Option<Receiver<AudioClipLoadResult>>,
    active_generation: u64,
    queued: Option<(u64, String)>,
    spawn_error: Option<AudioClipLoadResult>,
}

impl Default for AudioClipLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioClipLoader {
    pub fn new() -> Self {
        Self {
            generation: 0,
            requested_path: String::new(),
            active_receiver: None,
            active_generation: 0,
            queued: None,
            spawn_error: None,
        }
    }

    pub fn request(&mut self, path: impl Into<String>) -> u64 {
        self.generation = self.generation.wrapping_add(1).max(1);
        let generation = self.generation;
        let path = path.into();
        self.requested_path.clone_from(&path);
        self.spawn_error = None;
        if self.active_receiver.is_some() {
            // One active decoder, one replaceable latest request. Intermediate
            // rapid selections are overwritten without spawning more work.
            self.queued = Some((generation, path));
        } else {
            self.start_request(generation, path);
        }
        generation
    }

    fn start_request(&mut self, generation: u64, path: String) {
        let (sender, receiver) = mpsc::sync_channel(1);
        match std::thread::Builder::new()
            .name("audio-clip-loader".into())
            .spawn(move || {
                let result = AudioClip::open(&path);
                let _ = sender.send(AudioClipLoadResult {
                    generation,
                    path,
                    result,
                });
            }) {
            Ok(_) => {
                self.active_receiver = Some(receiver);
                self.active_generation = generation;
            }
            Err(error) => {
                self.spawn_error = Some(AudioClipLoadResult {
                    generation,
                    path: self.requested_path.clone(),
                    result: Err(format!("cannot start audio clip loader: {error}")),
                });
            }
        }
    }

    /// Invalidate any result and queued request without waiting for FFmpeg.
    /// The one active worker remains bounded and its eventual completion is
    /// discarded; a later `request` replaces the single queued slot.
    pub fn cancel(&mut self) {
        self.generation = self.generation.wrapping_add(1).max(1);
        self.requested_path.clear();
        self.queued = None;
        self.spawn_error = None;
    }

    pub fn state(&self) -> AudioClipLoadState {
        if self.spawn_error.is_some()
            || (self.active_receiver.is_some() && self.active_generation == self.generation)
            || self.queued.is_some()
        {
            AudioClipLoadState::Loading
        } else {
            AudioClipLoadState::Idle
        }
    }

    #[cfg(test)]
    pub fn requested_path(&self) -> &str {
        &self.requested_path
    }

    pub fn poll(&mut self) -> Option<AudioClipLoadResult> {
        if let Some(error) = self.spawn_error.take() {
            return Some(error);
        }
        let outcome = self.active_receiver.as_ref()?.try_recv();
        match outcome {
            Ok(completion) => {
                self.active_receiver = None;
                self.active_generation = 0;
                if let Some((generation, path)) = self.queued.take() {
                    self.start_request(generation, path);
                }
                (completion.generation == self.generation
                    && completion.path == self.requested_path
                    && self.active_receiver.is_none())
                .then_some(completion)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                let failed_generation = self.active_generation;
                self.active_receiver = None;
                self.active_generation = 0;
                if let Some((generation, path)) = self.queued.take() {
                    self.start_request(generation, path);
                    None
                } else {
                    Some(AudioClipLoadResult {
                        generation: failed_generation,
                        path: self.requested_path.clone(),
                        result: Err("audio clip loader stopped unexpectedly".into()),
                    })
                }
            }
        }
    }
}

pub struct AudioClip {
    info: AudioClipInfo,
    samples: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl AudioClip {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let display_path = path.to_string_lossy().into_owned();
        ffmpeg::init().map_err(|error| format!("failed to initialize audio decoding: {error}"))?;

        let deadline = Instant::now() + AUDIO_CLIP_DECODE_TIMEOUT;
        let mut input =
            ffmpeg::format::input_with_interrupt(path, move || Instant::now() >= deadline)
                .map_err(|error| format!("cannot open audio clip {display_path}: {error}"))?;
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .ok_or_else(|| format!("no audio stream found in {display_path}"))?;
        let stream_index = stream.index();
        let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|error| {
                format!("invalid audio codec parameters in {display_path}: {error}")
            })?;
        let mut decoder = context
            .decoder()
            .audio()
            .map_err(|error| format!("cannot create audio decoder for {display_path}: {error}"))?;

        let input_layout = if decoder.channel_layout().is_empty() {
            ffmpeg::ChannelLayout::default(i32::from(decoder.channels()))
        } else {
            decoder.channel_layout()
        };
        if input_layout.is_empty() || decoder.rate() == 0 {
            return Err(format!(
                "audio clip {display_path} has no valid channel layout or sample rate"
            ));
        }
        decoder.set_channel_layout(input_layout);
        let mut resampler = ffmpeg::software::resampling::Context::get(
            decoder.format(),
            input_layout,
            decoder.rate(),
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            ffmpeg::ChannelLayout::MONO,
            CLIP_SAMPLE_RATE,
        )
        .map_err(|error| {
            format!("cannot configure audio conversion for {display_path}: {error}")
        })?;

        let mut samples = Vec::new();
        loop {
            if Instant::now() >= deadline {
                return Err(format!(
                    "audio clip {display_path} did not decode within {} seconds",
                    AUDIO_CLIP_DECODE_TIMEOUT.as_secs()
                ));
            }
            let mut packet = ffmpeg::Packet::empty();
            match packet.read(&mut input) {
                Ok(()) => {}
                Err(ffmpeg::Error::Eof) => break,
                Err(error) => {
                    return Err(format!(
                        "cannot read audio packets from {display_path}: {error}"
                    ))
                }
            }
            if packet.stream() != stream_index {
                continue;
            }
            decoder.send_packet(&packet).map_err(|error| {
                format!("cannot submit audio packet from {display_path}: {error}")
            })?;
            receive_decoded(&mut decoder, &mut resampler, &mut samples, &display_path)?;
        }
        decoder
            .send_eof()
            .map_err(|error| format!("cannot flush audio decoder for {display_path}: {error}"))?;
        receive_decoded(&mut decoder, &mut resampler, &mut samples, &display_path)?;
        flush_resampler(&mut resampler, &mut samples, &display_path)?;

        if samples.is_empty() {
            return Err(format!(
                "audio stream in {display_path} yielded no decoded samples"
            ));
        }
        let duration_secs = samples.len() as f64 / CLIP_SAMPLE_RATE as f64;
        let info = AudioClipInfo {
            path: display_path,
            duration_secs,
            sample_rate: CLIP_SAMPLE_RATE,
            sample_count: samples.len(),
        };
        let fft = FftPlanner::new().plan_fft_forward(FFT_SIZE);
        Ok(Self { info, samples, fft })
    }

    #[cfg(test)]
    pub(super) fn from_samples(samples: Vec<f32>, sample_rate: u32) -> Self {
        assert!(!samples.is_empty());
        assert!(sample_rate > 0);
        let info = AudioClipInfo {
            path: "test.wav".into(),
            duration_secs: samples.len() as f64 / sample_rate as f64,
            sample_rate,
            sample_count: samples.len(),
        };
        let fft = FftPlanner::new().plan_fft_forward(FFT_SIZE);
        Self { info, samples, fft }
    }

    pub fn info(&self) -> &AudioClipInfo {
        &self.info
    }

    pub fn analyze_at_time(
        &self,
        time_secs: f64,
        gain: f32,
        config: AudioBandConfig,
    ) -> AudioClipAnalysis {
        let mut frame = [0.0; FFT_SIZE];
        let mut previous = [0.0; FFT_SIZE];
        let finite_time = if time_secs.is_finite() {
            time_secs.max(0.0)
        } else {
            0.0
        };
        let loop_time = finite_time % self.info.duration_secs;
        let position = ((loop_time * self.info.sample_rate as f64).floor() as usize)
            .min(self.samples.len() - 1);
        // Fixed physical comparison interval, independent of render FPS.
        let onset_hop = (self.info.sample_rate / 60).max(1) as usize;
        // Windows end at the selected time. Circular addressing makes the
        // clip an exact, gap-free loop without codec seek or cadence state.
        for (index, output) in frame.iter_mut().enumerate() {
            let source = (position + self.samples.len() + index - FFT_SIZE) % self.samples.len();
            *output = self.samples[source];
            let previous_source =
                (position + self.samples.len() * 2 + index - FFT_SIZE - onset_hop)
                    % self.samples.len();
            previous[index] = self.samples[previous_source];
        }
        analyze_window(
            &frame,
            &previous,
            self.info.sample_rate,
            gain,
            config,
            &self.fft,
        )
    }
}

fn push_converted(
    frame: &ffmpeg::frame::Audio,
    samples: &mut Vec<f32>,
    path: &str,
) -> Result<(), String> {
    let plane = frame.plane::<f32>(0);
    let remaining = MAX_AUDIO_CLIP_SAMPLES.saturating_sub(samples.len());
    if plane.len() > remaining {
        return Err(format!(
            "audio clip {path} exceeds the {} minute decode limit",
            MAX_AUDIO_CLIP_SECONDS / 60
        ));
    }
    samples.extend(plane.iter().copied().map(|sample| {
        if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }));
    Ok(())
}

fn receive_decoded(
    decoder: &mut ffmpeg::decoder::Audio,
    resampler: &mut ffmpeg::software::resampling::Context,
    samples: &mut Vec<f32>,
    path: &str,
) -> Result<(), String> {
    let mut decoded = ffmpeg::frame::Audio::empty();
    loop {
        match decoder.receive_frame(&mut decoded) {
            Ok(()) => {
                let delayed = resampler
                    .delay()
                    .map(|delay| delay.output.max(0) as usize)
                    .unwrap_or(0);
                let output_samples = ((decoded.samples() as u64 * CLIP_SAMPLE_RATE as u64)
                    .div_ceil(decoded.rate() as u64)
                    + delayed as u64
                    + 64) as usize;
                let mut converted = ffmpeg::frame::Audio::new(
                    resampler.output().format,
                    output_samples,
                    resampler.output().channel_layout,
                );
                converted.set_rate(resampler.output().rate);
                resampler.run(&decoded, &mut converted).map_err(|error| {
                    format!("cannot convert audio samples from {path}: {error}")
                })?;
                if converted.samples() > 0 {
                    push_converted(&converted, samples, path)?;
                }
            }
            Err(ffmpeg::Error::Other {
                errno: ffmpeg::error::EAGAIN,
            })
            | Err(ffmpeg::Error::Eof) => return Ok(()),
            Err(error) => {
                return Err(format!(
                    "audio decoder failed while reading {path}: {error}"
                ))
            }
        }
    }
}

fn flush_resampler(
    resampler: &mut ffmpeg::software::resampling::Context,
    samples: &mut Vec<f32>,
    path: &str,
) -> Result<(), String> {
    for _ in 0..MAX_RESAMPLER_FLUSH_FRAMES {
        let output_samples = resampler
            .delay()
            .map(|delay| delay.output.max(0) as usize + 64)
            .unwrap_or(64);
        let mut converted = ffmpeg::frame::Audio::new(
            resampler.output().format,
            output_samples,
            resampler.output().channel_layout,
        );
        converted.set_rate(resampler.output().rate);
        let delay = match resampler.flush(&mut converted) {
            Ok(delay) => delay,
            // FFmpeg may report this after successfully emitting every
            // sample when the caller supplied an initially empty output
            // frame. There is no pending input left to recover here.
            Err(ffmpeg::Error::OutputChanged) => return Ok(()),
            Err(error) => return Err(format!("cannot flush audio conversion for {path}: {error}")),
        };
        if converted.samples() > 0 {
            push_converted(&converted, samples, path)?;
        }
        if delay.is_none() {
            return Ok(());
        }
    }
    Err(format!(
        "audio conversion for {path} did not flush within {MAX_RESAMPLER_FLUSH_FRAMES} frames"
    ))
}

fn analyze_window(
    frame: &[f32; FFT_SIZE],
    previous_frame: &[f32; FFT_SIZE],
    sample_rate: u32,
    gain: f32,
    config: AudioBandConfig,
    fft: &Arc<dyn Fft<f32>>,
) -> AudioClipAnalysis {
    let gain = if gain.is_finite() {
        gain.clamp(0.0, 8.0)
    } else {
        1.0
    };
    let rms = (frame.iter().map(|sample| sample * sample).sum::<f32>() / FFT_SIZE as f32).sqrt();
    let windowed = |samples: &[f32; FFT_SIZE]| {
        samples
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let x = index as f32 / (FFT_SIZE - 1) as f32;
                let window = 0.5 - 0.5 * (std::f32::consts::TAU * x).cos();
                Complex::new(sample * window, 0.0)
            })
            .collect::<Vec<_>>()
    };
    let mut spectrum = windowed(frame);
    let mut previous_spectrum = windowed(previous_frame);
    fft.process(&mut spectrum);
    fft.process(&mut previous_spectrum);

    let hz_per_bin = sample_rate as f32 / FFT_SIZE as f32;
    let mut band_sums = [0.0; MAX_AUDIO_BANDS];
    let mut band_counts = [0; MAX_AUDIO_BANDS];
    let mut mag_sum = 0.0;
    let mut weighted_hz = 0.0;
    let mut log_sum = 0.0;
    let mut char_bins = 0;
    let mut display_sums = [0.0; AUDIO_SPECTRUM_BINS];
    let mut display_counts = [0u16; AUDIO_SPECTRUM_BINS];

    for (index, bin) in spectrum.iter().enumerate().take(FFT_SIZE / 2).skip(1) {
        let magnitude = (bin.norm() / FFT_SIZE as f32).max(0.0);
        let hz = index as f32 * hz_per_bin;
        accumulate_band_magnitude(config, hz, magnitude, &mut band_sums, &mut band_counts);
        if hz < config.ceiling_hz() {
            mag_sum += magnitude;
            weighted_hz += magnitude * hz;
            log_sum += (magnitude + 1e-9).ln();
            char_bins += 1;
        }
        if let Some(display) =
            display_bin_for_hz(hz, config.ceiling_hz().min(sample_rate as f32 * 0.5))
        {
            display_sums[display] += magnitude;
            display_counts[display] = display_counts[display].saturating_add(1);
        }
    }

    for index in 0..config.count() {
        if band_counts[index] > 0 {
            band_sums[index] /= band_counts[index] as f32;
        }
    }
    let band_total = band_sums[..config.count()].iter().sum::<f32>().max(1e-9);
    let mut bands = [0.0; MAX_AUDIO_BANDS];
    for index in 0..config.count() {
        bands[index] = (band_sums[index] / band_total * gain).clamp(0.0, 1.0);
    }
    let flux_numerator = spectrum
        .iter()
        .zip(previous_spectrum.iter())
        .take(FFT_SIZE / 2)
        .skip(1)
        .map(|(current, previous)| (current.norm() - previous.norm()).max(0.0))
        .sum::<f32>();
    let flux_denominator = previous_spectrum
        .iter()
        .take(FFT_SIZE / 2)
        .skip(1)
        .map(|bin| bin.norm())
        .sum::<f32>()
        .max(1e-6);
    let flux = flux_numerator / flux_denominator;
    let (bright, noise) = spectral_character(mag_sum, weighted_hz, log_sum, char_bins)
        .map(|(centroid, flatness)| {
            (
                brightness_from_centroid(centroid, config.ceiling_hz()),
                flatness,
            )
        })
        .unwrap_or((0.0, 0.0));
    let levels = AudioLevels {
        level: (rms * std::f32::consts::SQRT_2 * gain).clamp(0.0, 1.0),
        bands,
        bass: bands[0],
        mid: bands[1],
        high: bands[2],
        onset: (flux * 4.0 * gain).clamp(0.0, 1.0),
        bright,
        noise,
    };
    let mut display = [0.0; AUDIO_SPECTRUM_BINS];
    let peak = display_sums
        .iter()
        .enumerate()
        .map(|(index, sum)| {
            if display_counts[index] > 0 {
                *sum / display_counts[index] as f32
            } else {
                0.0
            }
        })
        .fold(1e-9, f32::max);
    for index in 0..AUDIO_SPECTRUM_BINS {
        if display_counts[index] > 0 {
            display[index] =
                (display_sums[index] / display_counts[index] as f32 / peak).clamp(0.0, 1.0);
        }
    }
    AudioClipAnalysis {
        levels,
        spectrum: display,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::Duration;

    fn tone(seconds: f32, hz: f32) -> AudioClip {
        let samples = (0..(CLIP_SAMPLE_RATE as f32 * seconds) as usize)
            .map(|index| {
                (std::f32::consts::TAU * hz * index as f32 / CLIP_SAMPLE_RATE as f32).sin() * 0.5
            })
            .collect();
        AudioClip::from_samples(samples, CLIP_SAMPLE_RATE)
    }

    #[test]
    fn extension_filter_is_case_insensitive_and_canonical() {
        for extension in AUDIO_FILE_EXTENSIONS {
            assert!(is_supported_audio_file(format!("music.{extension}")));
            assert!(is_supported_audio_extension(extension));
        }
        assert!(is_supported_audio_extension("MP3"));
        assert!(is_supported_audio_file("MUSIC.MP3"));
        assert!(!is_supported_audio_file("music.txt"));
    }

    #[test]
    fn linked_ffmpeg_has_decoders_for_advertised_audio_families() {
        ffmpeg::init().unwrap();
        for codec in [
            ffmpeg::codec::Id::PCM_S16LE,
            ffmpeg::codec::Id::MP3,
            ffmpeg::codec::Id::FLAC,
            ffmpeg::codec::Id::VORBIS,
            ffmpeg::codec::Id::OPUS,
            ffmpeg::codec::Id::AAC,
        ] {
            assert!(
                ffmpeg::codec::decoder::find(codec).is_some(),
                "linked FFmpeg lacks decoder {codec:?}"
            );
        }
    }

    #[test]
    fn ffmpeg_open_decodes_and_resamples_a_pcm_wav() {
        let unique = format!(
            "collide-o-scope-audio-{}-{}.wav",
            std::process::id(),
            Instant::now().elapsed().as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let source_rate = 8_000u32;
        let source_samples = source_rate / 10;
        let data_bytes = source_samples * 2;
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"RIFF").unwrap();
        file.write_all(&(36 + data_bytes).to_le_bytes()).unwrap();
        file.write_all(b"WAVEfmt ").unwrap();
        file.write_all(&16u32.to_le_bytes()).unwrap();
        file.write_all(&1u16.to_le_bytes()).unwrap();
        file.write_all(&1u16.to_le_bytes()).unwrap();
        file.write_all(&source_rate.to_le_bytes()).unwrap();
        file.write_all(&(source_rate * 2).to_le_bytes()).unwrap();
        file.write_all(&2u16.to_le_bytes()).unwrap();
        file.write_all(&16u16.to_le_bytes()).unwrap();
        file.write_all(b"data").unwrap();
        file.write_all(&data_bytes.to_le_bytes()).unwrap();
        for index in 0..source_samples {
            let sample = (std::f32::consts::TAU * 440.0 * index as f32 / source_rate as f32).sin()
                * 16_000.0;
            file.write_all(&(sample as i16).to_le_bytes()).unwrap();
        }
        drop(file);

        let clip = AudioClip::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(clip.info().sample_rate, CLIP_SAMPLE_RATE);
        assert!(
            (clip.info().duration_secs - 0.1).abs() < 0.01,
            "decoded duration was {}",
            clip.info().duration_secs
        );
        assert!(
            clip.analyze_at_time(0.05, 1.0, AudioBandConfig::default())
                .levels
                .level
                > 0.1
        );
    }

    #[test]
    fn analysis_is_pure_and_cadence_independent() {
        let clip = tone(2.0, 440.0);
        let first = clip.analyze_at_time(0.5, 1.0, AudioBandConfig::default());
        let _ = clip.analyze_at_time(0.25, 5.0, AudioBandConfig::new(8, &[50.0], 20_000.0));
        let second = clip.analyze_at_time(0.5, 1.0, AudioBandConfig::default());
        assert_eq!(first, second);

        // A frame reached at 30 fps and the corresponding even frame at
        // 60 fps address exactly the same program timestamp.
        for frame_30 in 0..30 {
            let at_30 =
                clip.analyze_at_time(frame_30 as f64 / 30.0, 1.0, AudioBandConfig::default());
            let at_60 = clip.analyze_at_time(
                (frame_30 * 2) as f64 / 60.0,
                1.0,
                AudioBandConfig::default(),
            );
            assert_eq!(at_30, at_60);
        }
    }

    #[test]
    fn loop_position_is_exact_and_negative_or_nonfinite_time_is_safe() {
        let clip = tone(1.0, 440.0);
        let at_zero = clip.analyze_at_time(0.0, 1.0, AudioBandConfig::default());
        let after_loop = clip.analyze_at_time(1.0, 1.0, AudioBandConfig::default());
        let negative = clip.analyze_at_time(-3.0, 1.0, AudioBandConfig::default());
        let nan = clip.analyze_at_time(f64::NAN, 1.0, AudioBandConfig::default());
        assert_eq!(at_zero, after_loop);
        assert_eq!(at_zero, negative);
        assert_eq!(at_zero, nan);
    }

    #[test]
    fn temporal_flux_detects_a_transient_not_a_steady_tone() {
        let steady = tone(1.0, 440.0);
        let mut impulse = vec![0.0; CLIP_SAMPLE_RATE as usize];
        // At t=.5 this lies near the Hann peak in the current window and
        // outside the canonical preceding window.
        impulse[CLIP_SAMPLE_RATE as usize / 2 - FFT_SIZE / 2] = 1.0;
        let impulse = AudioClip::from_samples(impulse, CLIP_SAMPLE_RATE);
        let steady_level = steady
            .analyze_at_time(0.5, 1.0, AudioBandConfig::default())
            .levels
            .onset;
        let impulse_level = impulse
            .analyze_at_time(0.5, 1.0, AudioBandConfig::default())
            .levels
            .onset;
        assert!(
            impulse_level > steady_level + 0.4,
            "{impulse_level} <= {steady_level}"
        );
    }

    #[test]
    fn loader_poll_is_nonblocking_and_cancel_invalidates_pending_generation() {
        let mut loader = AudioClipLoader::new();
        assert_eq!(loader.state(), AudioClipLoadState::Idle);
        let generation = loader.request("this-file-does-not-exist.wav");
        assert!(generation > 0);
        assert_eq!(loader.state(), AudioClipLoadState::Loading);
        let _ = loader.poll();
        loader.cancel();
        assert_eq!(loader.state(), AudioClipLoadState::Idle);
        assert_eq!(loader.requested_path(), "");
        std::thread::sleep(Duration::from_millis(10));
        assert!(loader.poll().is_none());
    }

    #[test]
    fn rapid_requests_keep_one_active_worker_and_only_latest_queued() {
        let mut loader = AudioClipLoader::new();
        let first = loader.request("missing-one.wav");
        let second = loader.request("missing-two.wav");
        let third = loader.request("missing-three.wav");
        assert!(first < second && second < third);
        assert_eq!(loader.active_generation, first);
        assert_eq!(loader.queued.as_ref().map(|entry| entry.0), Some(third));
        assert_eq!(loader.requested_path(), "missing-three.wav");
        assert_eq!(loader.state(), AudioClipLoadState::Loading);

        let deadline = Instant::now() + Duration::from_secs(3);
        let mut completion = None;
        while Instant::now() < deadline {
            if let Some(result) = loader.poll() {
                completion = Some(result);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let completion = completion.expect("latest queued request should complete");
        assert_eq!(completion.generation, third);
        assert_eq!(completion.path, "missing-three.wav");
        assert!(completion.result.is_err());
        assert_eq!(loader.state(), AudioClipLoadState::Idle);
    }
}
