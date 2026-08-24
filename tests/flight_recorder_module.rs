// Compile the recorder through an independent integration-test seam and prove
// its public default location agrees with the host-path policy. Production
// producer wiring is exercised by the main binary tests.
#![allow(dead_code)]

#[path = "../src/build_identity.rs"]
mod build_identity;
#[path = "../src/flight_recorder.rs"]
mod flight_recorder;
#[path = "../src/host_paths.rs"]
mod host_paths;

#[test]
fn isolated_module_exposes_the_per_user_default_location() {
    assert_eq!(
        flight_recorder::recorder_directory(),
        host_paths::state_root().join("flight-recorder-v1")
    );
}
