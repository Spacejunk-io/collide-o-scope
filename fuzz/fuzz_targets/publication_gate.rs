#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../src/publication_gate.rs"]
mod publication_gate;

use publication_gate::{LatestOnlyPublicationGate, PublicationToken};

fuzz_target!(|operations: &[u8]| {
    let mut gate = LatestOnlyPublicationGate::default();
    let mut tokens = Vec::<PublicationToken>::new();
    for operation in operations.iter().copied().take(4_096) {
        match operation % 4 {
            0 => tokens.push(gate.request()),
            1 => gate.cancel_all(),
            2 => {
                if let Some(token) = gate.claim_latest() {
                    let previous = gate.published_generation();
                    assert!(gate.try_publish(token));
                    assert!(gate.published_generation() > previous);
                }
            }
            _ => {
                if let Some(token) = tokens.get(usize::from(operation) % tokens.len().max(1)) {
                    let previous = gate.published_generation();
                    let accepted = gate.try_publish(*token);
                    if accepted {
                        assert!(gate.published_generation() > previous);
                    } else {
                        assert_eq!(gate.published_generation(), previous);
                    }
                }
            }
        }
    }
});
