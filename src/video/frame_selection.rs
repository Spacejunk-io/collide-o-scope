//! Dependency-light frame-selection policy shared by the decoder and P8.

/// Whether an already uploaded frame remains the exact selection for a newer
/// desire. Generation mismatch, hostile timestamps/FPS, reverse travel beyond
/// the half-frame window, and forward travel beyond the same window all fail
/// closed.
pub fn accepted_frame_remains_selected(
    requested_generation: u64,
    target_seconds: f64,
    accepted_generation: Option<u64>,
    accepted_source_seconds: Option<f64>,
    source_fps: f32,
) -> bool {
    let Some(accepted_source_seconds) = accepted_source_seconds else {
        return false;
    };
    if accepted_generation != Some(requested_generation)
        || !target_seconds.is_finite()
        || !accepted_source_seconds.is_finite()
        || !source_fps.is_finite()
        || source_fps <= 0.0
        || target_seconds + 0.5 / f64::from(source_fps) < accepted_source_seconds
    {
        return false;
    }
    target_seconds <= accepted_source_seconds + 0.5 / f64::from(source_fps)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn generation_finite_and_half_frame_boundaries_fail_closed() {
        assert!(accepted_frame_remains_selected(
            7,
            10.0,
            Some(7),
            Some(10.0),
            30.0
        ));
        assert!(!accepted_frame_remains_selected(
            7,
            10.0,
            Some(6),
            Some(10.0),
            30.0
        ));
        assert!(!accepted_frame_remains_selected(
            7,
            f64::NAN,
            Some(7),
            Some(10.0),
            30.0
        ));
        assert!(!accepted_frame_remains_selected(
            7,
            10.02,
            Some(7),
            Some(10.0),
            30.0
        ));
    }
}
