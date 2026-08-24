#![no_main]

use libfuzzer_sys::fuzz_target;

// Compile the production OSC adapter against the production, process-local
// ingress vocabulary. Keeping the real module at the fuzz crate root mirrors
// `crate::action_correlation` in the application without a fuzz-only envelope
// or lifecycle shim that could drift from the transport boundary under test.
#[allow(
    dead_code,
    reason = "the OSC fuzz target exercises only the adapter-facing subset"
)]
#[path = "../../src/action_correlation.rs"]
mod action_correlation;

#[path = "support/control_types.rs"]
mod control_types;

mod host_paths {
    pub use crate::control_types::state_root_from;
}
mod image_routing {
    pub use crate::control_types::StableLayerId;
}
mod performance {
    pub use crate::control_types::{SavedLayerPosition, SceneId};
}
mod visual_rack {
    pub use crate::control_types::{GroupId, NodeId};
}

#[path = "../../src/controller_profile.rs"]
mod controller_profile;
#[path = "../../src/osc.rs"]
mod osc;

use controller_profile::AutomationValue;

fn exercise_packet(bytes: &[u8]) {
    let peer = "127.0.0.1:32123".parse().expect("fixed peer");
    let Ok(events) = osc::decode_packet(bytes, peer) else {
        return;
    };
    assert!(events.len() <= osc::OSC_MAX_MESSAGES_PER_PACKET);
    for event in events {
        match event.value {
            AutomationValue::Absolute(value) => {
                assert!(value.is_finite() && (0.0..=1.0).contains(&value));
            }
            AutomationValue::Trigger | AutomationValue::Gate(false) => {}
            AutomationValue::Delta(_) | AutomationValue::Gate(true) => {
                panic!("OSC decoder emitted an unsupported value shape")
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > osc::OSC_MAX_DATAGRAM_BYTES {
        return;
    }
    exercise_packet(data);

    if let Ok(config) = osc::OscConfigDocument::from_json_bytes(data) {
        let canonical = config
            .to_json_bytes()
            .expect("admitted OSC config serializes");
        let reparsed = osc::OscConfigDocument::from_json_bytes(&canonical)
            .expect("canonical OSC config reparses");
        assert_eq!(reparsed, config);
    }

    // A text seed/mutation can reach a structurally valid OSC packet without
    // requiring a binary corpus containing NUL padding.
    if let Some(address) = data.strip_prefix(b"address:") {
        if let Ok(address) = std::str::from_utf8(address) {
            if let Ok(address) = osc::parse_control_address(address) {
                let packet = osc::encode_feedback(address, 0.5)
                    .expect("admitted address encodes as bounded feedback");
                exercise_packet(&packet);
            }
        }
    }
});
