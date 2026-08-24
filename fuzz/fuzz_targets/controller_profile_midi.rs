#![no_main]

use libfuzzer_sys::fuzz_target;

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

use controller_profile::{
    AutomationValue, ControllerDecoder, ControllerEventKind, ControllerProfileAction,
    ControllerProfileDocument, ResolvedControllerProfile,
};

fuzz_target!(|data: &[u8]| {
    if data.len() > controller_profile::CONTROLLER_PROFILE_ACTION_MAX_BYTES {
        return;
    }

    if let Ok(document) = ControllerProfileDocument::from_json_bytes(data) {
        document.validate().expect("admitted profile remains valid");
        let canonical = document
            .to_json_bytes()
            .expect("admitted controller profile serializes");
        let reparsed = ControllerProfileDocument::from_json_bytes(&canonical)
            .expect("canonical controller profile reparses");
        assert_eq!(reparsed, document);
    }
    if let Ok(action) = ControllerProfileAction::from_json_bytes(data) {
        let canonical = action
            .to_json_bytes()
            .expect("admitted controller action serializes");
        let reparsed = ControllerProfileAction::from_json_bytes(&canonical)
            .expect("canonical controller action reparses");
        assert_eq!(reparsed, action);
    }

    // The same target covers the complete-message MIDI ingress. Decoder
    // output is caller-bounded even if a future profile fans one message out.
    let mut decoder = ControllerDecoder::new(ResolvedControllerProfile::legacy_four_cc());
    let mut events = Vec::with_capacity(8);
    let report = decoder.decode_bounded(7, data, &mut events, 8);
    assert!(events.len() <= 8);
    assert_eq!(events.len(), report.emitted_events);
    assert!(report.emitted_events <= report.matched_bindings);
    for event in events {
        if let ControllerEventKind::Control { value, .. } = event.kind {
            match value {
                AutomationValue::Absolute(value) => {
                    assert!(value.is_finite() && (0.0..=1.0).contains(&value));
                }
                AutomationValue::Delta(value) => {
                    assert!(value.is_finite() && (-1.0..=1.0).contains(&value));
                }
                AutomationValue::Trigger | AutomationValue::Gate(_) => {}
            }
        }
    }
});
