#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../src/motion_sidecar_wire.rs"]
mod motion_sidecar_wire;

const FUZZ_MAX_BYTES: usize = 1024 * 1024;
const CURRENT_SCHEMA: u16 = motion_sidecar_wire::MOTION_SIDECAR_SCHEMA_VERSION;

fuzz_target!(|data: &[u8]| {
    if data.len() > FUZZ_MAX_BYTES {
        return;
    }
    let Ok(value) = motion_sidecar_wire::parse_motion_sidecar(data, CURRENT_SCHEMA) else {
        return;
    };
    let canonical = serde_json::to_vec(&value).expect("admitted motion sidecar serializes");
    let reparsed = motion_sidecar_wire::parse_motion_sidecar(&canonical, CURRENT_SCHEMA)
        .expect("canonical motion sidecar reparses");
    assert_eq!(reparsed, value);
});
