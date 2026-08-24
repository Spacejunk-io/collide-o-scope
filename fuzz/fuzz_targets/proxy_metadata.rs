#![no_main]
#![allow(dead_code)]

use libfuzzer_sys::fuzz_target;

// `proxy.rs` is deliberately a pure planning/parser module. Its only host
// dependency is this two-field content identity; the production parser and
// every proxy validation rule below are included verbatim.
mod media_source {
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ContentIdentity {
        pub sha256: String,
        pub byte_len: u64,
    }
}

mod media_safety {
    pub const ABSOLUTE_MEDIA_MAX_EDGE: u32 = 16_384;
}

#[path = "../../src/proxy.rs"]
mod proxy;

use proxy::{ProxyCacheKey, ProxyPlaybackObservation, ProxySettings};

const MAX_PROXY_METADATA_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_PROXY_METADATA_BYTES {
        return;
    }

    if let Ok(settings) = serde_json::from_slice::<ProxySettings>(data) {
        settings.validate().expect("admitted settings remain valid");
        let canonical = serde_json::to_vec(&settings).expect("settings serialize");
        let reparsed: ProxySettings =
            serde_json::from_slice(&canonical).expect("canonical settings reparse");
        assert_eq!(reparsed, settings);
    }

    if let Ok(observation) = serde_json::from_slice::<ProxyPlaybackObservation>(data) {
        observation
            .validate()
            .expect("admitted playback observation remains valid");
        let canonical = serde_json::to_vec(&observation).expect("observation serializes");
        let reparsed: ProxyPlaybackObservation =
            serde_json::from_slice(&canonical).expect("canonical observation reparses");
        assert_eq!(reparsed, observation);
    }

    if let Ok(key) = serde_json::from_slice::<ProxyCacheKey>(data) {
        let canonical = serde_json::to_vec(&key).expect("cache key serializes");
        let reparsed: ProxyCacheKey =
            serde_json::from_slice(&canonical).expect("canonical cache key reparses");
        assert_eq!(reparsed, key);
    }

    if let Ok(text) = std::str::from_utf8(data) {
        if let Ok(key) = ProxyCacheKey::from_hex(text) {
            assert_eq!(ProxyCacheKey::from_hex(&key.to_hex()).unwrap(), key);
        }
    }
});
