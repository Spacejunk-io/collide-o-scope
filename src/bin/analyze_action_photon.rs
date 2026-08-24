use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use collide_o_scope::action_photon::{analyze_action_photon_fixture, ActionPhotonFixtureInput};

const MAX_FIXTURE_BYTES: u64 = 64 * 1024 * 1024;

fn main() {
    if let Err(error) = run() {
        eprintln!("action-to-photon fixture rejected: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let input_path = args.next().map(PathBuf::from).ok_or_else(|| {
        "usage: analyze_action_photon <optical-input.json> <receipt.json>".to_string()
    })?;
    let output_path = args.next().map(PathBuf::from).ok_or_else(|| {
        "usage: analyze_action_photon <optical-input.json> <receipt.json>".to_string()
    })?;
    if args.next().is_some() {
        return Err("usage: analyze_action_photon <optical-input.json> <receipt.json>".to_string());
    }
    let metadata = fs::metadata(&input_path).map_err(|error| error.to_string())?;
    if !metadata.is_file() || metadata.len() > MAX_FIXTURE_BYTES {
        return Err(format!(
            "input must be one regular file no larger than {MAX_FIXTURE_BYTES} bytes"
        ));
    }
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    fs::File::open(&input_path)
        .and_then(|file| file.take(MAX_FIXTURE_BYTES + 1).read_to_end(&mut bytes))
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > MAX_FIXTURE_BYTES {
        return Err("input grew beyond its byte cap while reading".to_string());
    }
    let input: ActionPhotonFixtureInput =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    let receipt = analyze_action_photon_fixture(&input).map_err(|error| error.to_string())?;
    let mut encoded = serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?;
    encoded.push(b'\n');
    publish_create_new(&output_path, &encoded).map_err(|error| error.to_string())?;
    println!(
        "physical action-to-photon receipt: {} trials, p95 {:.3} ms",
        receipt.latency.trials,
        receipt.latency.p95_nanoseconds as f64 / 1_000_000.0
    );
    Ok(())
}

fn publish_create_new(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "receipt already exists",
        ));
    }
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("action-photon-receipt.json");
    let staging = parent.join(format!(".{filename}.{}.staging", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staging)?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        // A same-filesystem hard link is a true create-new publication on
        // Windows and Unix; an external winner can never be overwritten.
        fs::hard_link(&staging, path)?;
        fs::remove_file(&staging)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staging);
    }
    result
}
