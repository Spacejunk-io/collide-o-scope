//! Small, dependency-light public contracts shared by the application and
//! maintenance binaries.

pub mod action_photon;
pub mod alpha_export;
pub mod capability;
pub mod diagnostics;
#[allow(
    dead_code,
    reason = "the library reuses only the durable publication subset needed by alpha export"
)]
mod durable_file;
pub mod gpu_recovery;
pub mod mosh_domains;
pub mod photosensitivity_advisor;
pub mod photosensitivity_gpu;
pub mod presentation_profile;
pub mod source_transition;
