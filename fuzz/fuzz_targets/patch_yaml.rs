#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../src/patch/yaml_boundary.rs"]
mod yaml_boundary;

const TARGET_MAX_BYTES: usize = 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    // CI intentionally explores a smaller envelope than the production
    // 32-MiB cap. The production boundary remains authoritative and bounded.
    if data.is_empty() || data.len() > TARGET_MAX_BYTES {
        return;
    }
    let Ok(value) = yaml_boundary::parse_patch_yaml_value(data) else {
        return;
    };
    // `serde_yaml::Value` accepts tagged mapping keys that its fallible emitter
    // cannot represent. Production converts this bounded tree directly into a
    // typed `PatchState`; only exercise round-trip laws for emitter-supported
    // values instead of turning an ordinary dependency error into a fuzz crash.
    let Ok(canonical) = serde_yaml::to_string(&value) else {
        return;
    };
    assert!(canonical.len() <= yaml_boundary::MAX_PATCH_FILE_BYTES);
    let reparsed = yaml_boundary::parse_patch_yaml_value(canonical.as_bytes())
        .expect("canonical bounded YAML reparses");
    assert_eq!(reparsed, value);
});
