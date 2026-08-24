//! Offline optical action-to-photon fixture analysis.
//!
//! Engine ingress/apply/submit timestamps are intentionally absent from this
//! schema.  A trial begins only when the camera/photodiode observes the input
//! LED edge and ends when that same sensor timeline observes the display's
//! known full-frame transition.  The result is therefore labelled physical
//! action-to-photon evidence, never inferred presentation evidence.

use std::fmt;

use serde::{Deserialize, Serialize};

pub const ACTION_PHOTON_SCHEMA_VERSION: u16 = 1;
pub const MAX_PHYSICAL_TRIALS: usize = 4_096;
pub const MAX_SAMPLES_PER_TRIAL: usize = 16_384;
pub const DEFAULT_EDGE_THRESHOLD_Q16: u16 = 32_768;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalPresentMode {
    Fifo,
    FifoRelaxed,
    Immediate,
    Mailbox,
    AutoVsync,
    AutoNoVsync,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalDisplayMode {
    pub raster_width: u32,
    pub raster_height: u32,
    pub refresh_millihertz: u32,
    pub present_mode: PhysicalPresentMode,
    pub fullscreen: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpticalSample {
    pub trial: u32,
    pub elapsed_nanoseconds: u64,
    /// Normalized optical intensity, 0..=65535, for the simultaneously visible
    /// input LED or equivalent electrically coupled marker.
    pub input_led_q16: u16,
    /// Normalized optical intensity for the known full-frame output patch.
    pub display_q16: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionPhotonFixtureInput {
    pub schema_version: u16,
    pub fixture_digest: [u8; 32],
    pub display: PhysicalDisplayMode,
    /// Camera/ADC sample interval. It is carried into the receipt as the
    /// measurement quantization bound rather than hidden in a rounded result.
    pub sample_interval_nanoseconds: u64,
    #[serde(default = "default_edge_threshold")]
    pub edge_threshold_q16: u16,
    pub samples: Vec<OpticalSample>,
}

const fn default_edge_threshold() -> u16 {
    DEFAULT_EDGE_THRESHOLD_Q16
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalLatencyPercentiles {
    pub p50_nanoseconds: u64,
    pub p95_nanoseconds: u64,
    pub p99_nanoseconds: u64,
    pub minimum_nanoseconds: u64,
    pub maximum_nanoseconds: u64,
    pub trials: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionPhotonReceipt {
    pub schema_version: u16,
    pub measurement_domain: PhysicalMeasurementDomain,
    pub fixture_digest: [u8; 32],
    pub display: PhysicalDisplayMode,
    pub sample_interval_nanoseconds: u64,
    pub edge_threshold_q16: u16,
    pub latency: PhysicalLatencyPercentiles,
    /// Kept explicit so a downstream report cannot silently relabel an engine
    /// queue submission as emitted light.
    pub engine_submission_is_not_photon_time: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalMeasurementDomain {
    PhysicalActionToPhoton,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionPhotonError {
    WrongSchema(u16),
    InvalidDisplayMode,
    InvalidSampleInterval,
    TooManySamples,
    TooManyTrials,
    TrialOrder(u32),
    MissingInputEdge(u32),
    MissingDisplayEdge(u32),
    DisplayPrecedesInput(u32),
}

impl fmt::Display for ActionPhotonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongSchema(version) => write!(formatter, "unsupported fixture schema {version}"),
            Self::InvalidDisplayMode => formatter.write_str("fixture display mode is invalid"),
            Self::InvalidSampleInterval => {
                formatter.write_str("fixture sample interval must be nonzero")
            }
            Self::TooManySamples => formatter.write_str("fixture exceeds its bounded sample cap"),
            Self::TooManyTrials => formatter.write_str("fixture exceeds its bounded trial cap"),
            Self::TrialOrder(trial) => {
                write!(formatter, "trial {trial} timestamps are not ordered")
            }
            Self::MissingInputEdge(trial) => {
                write!(formatter, "trial {trial} has no input LED edge")
            }
            Self::MissingDisplayEdge(trial) => {
                write!(formatter, "trial {trial} has no display edge")
            }
            Self::DisplayPrecedesInput(trial) => {
                write!(
                    formatter,
                    "trial {trial} display edge precedes its input edge"
                )
            }
        }
    }
}

impl std::error::Error for ActionPhotonError {}

pub fn analyze_action_photon_fixture(
    input: &ActionPhotonFixtureInput,
) -> Result<ActionPhotonReceipt, ActionPhotonError> {
    if input.schema_version != ACTION_PHOTON_SCHEMA_VERSION {
        return Err(ActionPhotonError::WrongSchema(input.schema_version));
    }
    if input.display.raster_width == 0
        || input.display.raster_height == 0
        || input.display.refresh_millihertz == 0
    {
        return Err(ActionPhotonError::InvalidDisplayMode);
    }
    if input.sample_interval_nanoseconds == 0 {
        return Err(ActionPhotonError::InvalidSampleInterval);
    }
    let total_cap = MAX_PHYSICAL_TRIALS.saturating_mul(MAX_SAMPLES_PER_TRIAL);
    if input.samples.len() > total_cap {
        return Err(ActionPhotonError::TooManySamples);
    }

    let mut latencies = Vec::new();
    let mut start = 0;
    while start < input.samples.len() {
        if latencies.len() >= MAX_PHYSICAL_TRIALS {
            return Err(ActionPhotonError::TooManyTrials);
        }
        let trial = input.samples[start].trial;
        let mut end = start + 1;
        while end < input.samples.len() && input.samples[end].trial == trial {
            end += 1;
        }
        let samples = &input.samples[start..end];
        if samples.len() > MAX_SAMPLES_PER_TRIAL {
            return Err(ActionPhotonError::TooManySamples);
        }
        if samples
            .windows(2)
            .any(|pair| pair[1].elapsed_nanoseconds <= pair[0].elapsed_nanoseconds)
        {
            return Err(ActionPhotonError::TrialOrder(trial));
        }
        let input_edge = rising_edge(samples, input.edge_threshold_q16, |sample| {
            sample.input_led_q16
        })
        .ok_or(ActionPhotonError::MissingInputEdge(trial))?;
        let display_edge = rising_edge(samples, input.edge_threshold_q16, |sample| {
            sample.display_q16
        })
        .ok_or(ActionPhotonError::MissingDisplayEdge(trial))?;
        if display_edge < input_edge {
            return Err(ActionPhotonError::DisplayPrecedesInput(trial));
        }
        latencies.push(display_edge - input_edge);
        start = end;
    }
    if latencies.is_empty() {
        return Err(ActionPhotonError::MissingInputEdge(0));
    }
    latencies.sort_unstable();
    let latency = PhysicalLatencyPercentiles {
        p50_nanoseconds: percentile(&latencies, 50),
        p95_nanoseconds: percentile(&latencies, 95),
        p99_nanoseconds: percentile(&latencies, 99),
        minimum_nanoseconds: latencies[0],
        maximum_nanoseconds: *latencies.last().expect("nonempty latency set"),
        trials: u32::try_from(latencies.len()).unwrap_or(u32::MAX),
    };
    Ok(ActionPhotonReceipt {
        schema_version: ACTION_PHOTON_SCHEMA_VERSION,
        measurement_domain: PhysicalMeasurementDomain::PhysicalActionToPhoton,
        fixture_digest: input.fixture_digest,
        display: input.display,
        sample_interval_nanoseconds: input.sample_interval_nanoseconds,
        edge_threshold_q16: input.edge_threshold_q16,
        latency,
        engine_submission_is_not_photon_time: true,
    })
}

fn rising_edge(
    samples: &[OpticalSample],
    threshold: u16,
    value: impl Fn(&OpticalSample) -> u16,
) -> Option<u64> {
    samples.windows(2).find_map(|pair| {
        (value(&pair[0]) < threshold && value(&pair[1]) >= threshold)
            .then_some(pair[1].elapsed_nanoseconds)
    })
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    let rank = sorted.len().saturating_mul(percentile).saturating_add(99) / 100;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(trial: u32, input_at: u64, display_at: u64) -> Vec<OpticalSample> {
        (0..=10)
            .map(|index| {
                let elapsed_nanoseconds = index * 1_000_000;
                OpticalSample {
                    trial,
                    elapsed_nanoseconds,
                    input_led_q16: if elapsed_nanoseconds >= input_at {
                        u16::MAX
                    } else {
                        0
                    },
                    display_q16: if elapsed_nanoseconds >= display_at {
                        u16::MAX
                    } else {
                        0
                    },
                }
            })
            .collect()
    }

    fn input(samples: Vec<OpticalSample>) -> ActionPhotonFixtureInput {
        ActionPhotonFixtureInput {
            schema_version: ACTION_PHOTON_SCHEMA_VERSION,
            fixture_digest: [7; 32],
            display: PhysicalDisplayMode {
                raster_width: 1920,
                raster_height: 1080,
                refresh_millihertz: 60_000,
                present_mode: PhysicalPresentMode::Fifo,
                fullscreen: true,
            },
            sample_interval_nanoseconds: 1_000_000,
            edge_threshold_q16: DEFAULT_EDGE_THRESHOLD_Q16,
            samples,
        }
    }

    #[test]
    fn physical_percentiles_share_only_the_optical_timeline() {
        let samples = [
            trial(1, 1_000_000, 3_000_000),
            trial(2, 1_000_000, 5_000_000),
            trial(3, 1_000_000, 8_000_000),
        ]
        .concat();
        let receipt = analyze_action_photon_fixture(&input(samples)).unwrap();
        assert_eq!(receipt.latency.p50_nanoseconds, 4_000_000);
        assert_eq!(receipt.latency.p95_nanoseconds, 7_000_000);
        assert_eq!(receipt.latency.p99_nanoseconds, 7_000_000);
        assert_eq!(receipt.latency.trials, 3);
        assert!(receipt.engine_submission_is_not_photon_time);
        assert_eq!(
            receipt.measurement_domain,
            PhysicalMeasurementDomain::PhysicalActionToPhoton
        );
    }

    #[test]
    fn missing_or_reversed_edges_are_refused_not_repaired() {
        let missing = trial(1, 1_000_000, u64::MAX);
        assert_eq!(
            analyze_action_photon_fixture(&input(missing)),
            Err(ActionPhotonError::MissingDisplayEdge(1))
        );

        let reversed = trial(2, 4_000_000, 2_000_000);
        assert_eq!(
            analyze_action_photon_fixture(&input(reversed)),
            Err(ActionPhotonError::DisplayPrecedesInput(2))
        );
    }

    #[test]
    fn display_mode_and_sample_quantization_are_mandatory_receipt_facts() {
        let mut fixture = input(trial(1, 1_000_000, 2_000_000));
        fixture.display.refresh_millihertz = 0;
        assert_eq!(
            analyze_action_photon_fixture(&fixture),
            Err(ActionPhotonError::InvalidDisplayMode)
        );
        fixture.display.refresh_millihertz = 60_000;
        fixture.sample_interval_nanoseconds = 0;
        assert_eq!(
            analyze_action_photon_fixture(&fixture),
            Err(ActionPhotonError::InvalidSampleInterval)
        );
    }
}
