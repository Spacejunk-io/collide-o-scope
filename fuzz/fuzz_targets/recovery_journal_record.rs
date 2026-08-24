#![no_main]

use libfuzzer_sys::fuzz_target;

mod patch {
    // The target owns journal framing/checksum/ordering. Patch YAML's exact
    // hostile tree boundary is exercised independently by `patch_yaml`.
    pub type PatchState = serde_yaml::Value;
}

#[allow(dead_code)]
#[path = "../../src/recovery_journal.rs"]
mod recovery_journal;

const TARGET_MAX_BYTES: usize = 1024 * 1024;

fn verify(bytes: &[u8]) {
    let scan =
        recovery_journal::scan_bytes(bytes, recovery_journal::RecoveryLimits::default(), false)
            .expect("in-memory recovery scan has no I/O failure");
    assert!(scan.valid_entries <= recovery_journal::RECOVERY_MAX_ENTRIES);
    assert!(scan.valid_bytes <= bytes.len() as u64);
    assert_eq!(scan.latest.is_some(), scan.valid_entries != 0);
    if let Some(checkpoint) = scan.latest {
        assert!(checkpoint.sequence != 0);
    }
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > TARGET_MAX_BYTES {
        return;
    }
    verify(data);

    // Half the input space is wrapped with the production checksum/record
    // encoder. That keeps the payload parser and valid-record ordering path
    // reachable instead of asking mutation to guess 256 checksum bits.
    if data[0] & 1 != 0 {
        let payload = &data[1..];
        let record = recovery_journal::encode_record(1, payload)
            .expect("one-megabyte structured payload fits the hard record cap");
        verify(&record);
    }
});
