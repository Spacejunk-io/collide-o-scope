#![no_main]

use libfuzzer_sys::fuzz_target;

mod temporal {
    pub const TEMPORAL_HISTORY_LEN: usize = 24;
}

#[path = "../../src/study.rs"]
mod study;

fuzz_target!(|data: &[u8]| {
    // Keep both local and parser-owned bounds so a mistaken cargo-fuzz option
    // can never turn this target into an unbounded allocator.
    if data.is_empty() || data.len() > study::STUDY_MAX_DOCUMENT_BYTES {
        return;
    }
    let Ok(document) = study::StudyDocument::from_json_bytes(data) else {
        return;
    };
    if document.validate().is_err() {
        return;
    }
    let canonical = document
        .to_json_bytes()
        .expect("validated Study serializes");
    assert!(canonical.len() <= study::STUDY_MAX_DOCUMENT_BYTES);
    let reparsed =
        study::StudyDocument::from_json_bytes(&canonical).expect("canonical Study reparses");
    assert_eq!(reparsed, document);
    let authority = reparsed.authority();
    assert!(!authority.native_code);
    assert!(!authority.shader_source);
    assert!(!authority.filesystem);
    assert!(!authority.network);
    assert!(!authority.process);
    assert!(!authority.device);
    assert!(!authority.host_mutation);
});
