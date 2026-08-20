//! The B6 Pixel Sort law — bright runs stretch into streaks.
//!
//! Not a true sort, and honestly so: a true sort is unbounded work per
//! pixel. The law is derived from BENDR (MIT, © 2026 Steve Blythe) and
//! transcribed faithfully with attribution: a pixel whose luma exceeds the
//! threshold searches upward through at most 32 taps, each stepping two
//! rows, for the end of its bright run, then takes the colour at the run's
//! far end mixed by the amount — so every pixel in a bright run inherits the
//! run's end colour, which is the streak. Luma is Rec.601 on the encoded
//! values (the instrument observes the stored picture; the corruption trio's
//! artefacts are storage artefacts, the B8 code-byte precedent). The search
//! clamps at the frame edge rather than wrapping — BENDR's own comment
//! records that wrapping stretched a false streak from the seam.
//!
//! This module is the independent CPU reference the rack shader is checked
//! against, in the `gesture.rs` tradition: no wgpu, clock, filesystem, or UI
//! dependency.

use serde::{Deserialize, Serialize};

/// The bounded run search: at most 32 taps, each two rows up, so the maximum
/// reach is 64 rows. Charged honestly in the node's descriptor.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const PIXEL_SORT_MAX_TAPS: u32 = 32;
/// Rows per search tap.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub const PIXEL_SORT_STEP_ROWS: u32 = 2;

/// Authored Pixel Sort state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PixelSortParams {
    /// Streak mix. The wake law: zero is an exact bypass.
    pub amount: f32,
    /// Luma threshold a pixel must exceed to join a run, and below which the
    /// run search stops.
    pub threshold: f32,
}

impl Default for PixelSortParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            threshold: 0.45,
        }
    }
}

impl PixelSortParams {
    /// Clamp every authored value into its declared range. Hostile
    /// non-finite input takes the neutral default rather than a clamped
    /// extreme.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        Self {
            amount: finite_clamp(self.amount, defaults.amount, 0.0, 1.0),
            threshold: finite_clamp(self.threshold, defaults.threshold, 0.0, 1.0),
        }
    }

    /// True when no streak is authored: the planner collects nothing extra
    /// and the pass is value-bypassed. Threshold alone wakes nothing — a
    /// threshold shapes runs that exist only once the amount does.
    pub fn is_exact_bypass(self) -> bool {
        self.sanitized().amount == 0.0
    }
}

/// Rec.601 luma on encoded straight values.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn sort_luma(rgb: [f32; 3]) -> f32 {
    0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
}

/// The whole law over one straight encoded-RGB image, row-major with y
/// increasing downward: BENDR's `+y` (toward the top of the picture in GL
/// coordinates) is `-y` here. Coordinates clamp at the frame edge.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the independent CPU reference the GPU fixtures compare against"
    )
)]
pub fn pixel_sort_reference(
    image: &[[f32; 3]],
    width: usize,
    height: usize,
    params: PixelSortParams,
) -> Vec<[f32; 3]> {
    let clean = params.sanitized();
    let mut out = image.to_vec();
    if clean.amount == 0.0 || width == 0 || height == 0 || image.len() < width * height {
        return out;
    }
    let sample = |x: usize, y: isize| -> [f32; 3] {
        let y = y.clamp(0, height as isize - 1) as usize;
        image[y * width + x]
    };
    for y in 0..height {
        for x in 0..width {
            let here = image[y * width + x];
            if sort_luma(here) <= clean.threshold {
                continue;
            }
            let mut reach: isize = 0;
            for k in 1..=PIXEL_SORT_MAX_TAPS as isize {
                let probe = y as isize - k * PIXEL_SORT_STEP_ROWS as isize;
                if sort_luma(sample(x, probe)) <= clean.threshold {
                    break;
                }
                reach = k * PIXEL_SORT_STEP_ROWS as isize;
            }
            let end = sample(x, y as isize - reach);
            let px = &mut out[y * width + x];
            for channel in 0..3 {
                px[channel] = here[channel] + (end[channel] - here[channel]) * clean.amount;
            }
        }
    }
    out
}

fn finite_clamp(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(width: usize, height: usize, value: f32) -> Vec<[f32; 3]> {
        vec![[value; 3]; width * height]
    }

    fn assert_close(a: [f32; 3], b: [f32; 3], what: &str) {
        for channel in 0..3 {
            assert!(
                (a[channel] - b[channel]).abs() < 1e-5,
                "{what}: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn sanitize_is_neutral_and_bypass_is_amount_alone() {
        let hostile = PixelSortParams {
            amount: f32::NAN,
            threshold: 5.0,
        };
        let clean = hostile.sanitized();
        assert_eq!(clean.amount, 0.0);
        assert_eq!(clean.threshold, 1.0);
        assert!(hostile.is_exact_bypass());
        assert!(PixelSortParams::default().is_exact_bypass());
        assert!(!PixelSortParams {
            amount: 0.5,
            ..Default::default()
        }
        .is_exact_bypass());
        // A threshold alone wakes nothing.
        assert!(PixelSortParams {
            threshold: 0.1,
            ..Default::default()
        }
        .is_exact_bypass());
    }

    #[test]
    fn dark_pixels_and_zero_amount_are_untouched() {
        let image = flat(4, 8, 0.2);
        let sorted = pixel_sort_reference(
            &image,
            4,
            8,
            PixelSortParams {
                amount: 1.0,
                threshold: 0.45,
            },
        );
        assert_eq!(sorted, image, "below-threshold pixels never move");
        let bright = flat(4, 8, 0.9);
        let untouched = pixel_sort_reference(&bright, 4, 8, PixelSortParams::default());
        assert_eq!(untouched, bright, "amount zero is the identity");
    }

    #[test]
    fn a_bright_run_takes_its_far_end_colour_at_full_amount() {
        // One column, 32 rows: dark above, a bright ramp below, with a
        // distinct colour at the run's top end.
        let width = 1usize;
        let height = 32usize;
        let mut image = flat(width, height, 0.1);
        // Rows 10..=25 are a bright run; row 10 is the distinctive end.
        for px in image.iter_mut().take(26).skip(10) {
            *px = [0.8, 0.6, 0.7];
        }
        image[10] = [0.9, 0.5, 0.2];
        let sorted = pixel_sort_reference(
            &image,
            width,
            height,
            PixelSortParams {
                amount: 1.0,
                threshold: 0.45,
            },
        );
        // A pixel deep in the run whose search (2-row steps) lands on or
        // above the run end takes colour from the end region; the reach is
        // capped by the first below-threshold tap.
        // Row 24: taps at 22,20,18,16,14,12,10 are bright, 8 is dark -> reach 14 -> row 10.
        assert_close(
            sorted[24],
            [0.9, 0.5, 0.2],
            "the streak carries the end colour",
        );
        // A dark pixel adjacent to the run is untouched.
        assert_eq!(sorted[9], image[9]);
        // The run end itself searches upward into darkness and holds.
        assert_eq!(sorted[10], image[10]);
    }

    #[test]
    fn the_search_is_bounded_at_64_rows_and_clamps_at_the_edge() {
        // A fully bright tall column: the search saturates all 32 taps.
        let width = 1usize;
        let height = 200usize;
        let mut image = flat(width, height, 0.9);
        image[0] = [0.2, 0.9, 0.3]; // bright but distinct top row
        image[86] = [0.9, 0.4, 0.6]; // bright, distinct: the exact 64-row mark
        let sorted = pixel_sort_reference(
            &image,
            width,
            height,
            PixelSortParams {
                amount: 1.0,
                threshold: 0.45,
            },
        );
        // Row 150 reaches at most 64 rows: row 86, not row 0.
        assert_close(
            sorted[150],
            image[86],
            "the reach is exactly 32 taps x 2 rows",
        );
        // Row 30's saturated search would pass row 0; taps clamp at the
        // edge (which is bright), so reach saturates and the fetch clamps.
        assert_close(sorted[30], image[0], "edge taps clamp, never wrap");
        // Hostile short input is returned unchanged.
        let short = pixel_sort_reference(
            &image[..10],
            4,
            8,
            PixelSortParams {
                amount: 1.0,
                threshold: 0.4,
            },
        );
        assert_eq!(short, image[..10].to_vec());
    }
}
