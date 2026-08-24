#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../src/web/action_wire.rs"]
mod action_wire;

fuzz_target!(|data: &[u8]| {
    if data.len() > action_wire::MAX_WEB_ACTION_BYTES {
        return;
    }
    let Ok(value) = action_wire::parse_web_action_value(data) else {
        return;
    };
    let action = value
        .get("action")
        .and_then(serde_json::Value::as_str)
        .expect("admitted action has a string tag");
    assert!(!action.is_empty() && action.len() <= 128);

    let canonical = serde_json::to_vec(&value).expect("admitted action value serializes");
    let reparsed = action_wire::parse_web_action::<serde_json::Value>(&canonical)
        .expect("canonical action reparses through production boundary");
    assert_eq!(reparsed, value);
});
