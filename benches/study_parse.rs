//! Pure parser benchmark; it deliberately has no renderer or FFmpeg dependency.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

mod temporal {
    pub const TEMPORAL_HISTORY_LEN: usize = 24;
}

#[path = "../src/video/frame_selection.rs"]
mod frame_selection;
#[allow(dead_code)]
#[path = "../src/photosensitivity_advisor.rs"]
mod photosensitivity_advisor;
#[path = "../src/publication_gate.rs"]
mod publication_gate;
#[path = "../src/study.rs"]
mod study;
#[path = "../src/patch/yaml_boundary.rs"]
mod yaml_boundary;

mod media_source {
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ContentIdentity {
        pub sha256: String,
        pub byte_len: u64,
    }

    impl ContentIdentity {
        pub fn new(sha256: impl Into<String>, byte_len: u64) -> Result<Self, &'static str> {
            let sha256 = sha256.into().to_ascii_lowercase();
            if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err("invalid SHA-256");
            }
            Ok(Self { sha256, byte_len })
        }
    }
}

mod media_safety {
    pub const ABSOLUTE_MEDIA_MAX_EDGE: u32 = 16_384;
}

#[allow(dead_code)]
#[path = "../src/proxy.rs"]
mod proxy;

const VALID_STUDY: &[u8] = br#"{
  "schema_version": 1,
  "abi": { "major": 1, "minor": 0 },
  "metadata": {
    "name": "Benchmark tint",
    "author": "collide-o-scope",
    "description": "Bounded parser benchmark fixture",
    "license": {
      "identifier": "CC0-1.0",
      "notice": "This notice covers the Study data only.",
      "publication_boundary": "study_data_only_does_not_license_host"
    }
  },
  "capabilities": ["current_color"],
  "instructions": [
    { "op": "load_current_color", "dst": 0 },
    { "op": "output_color", "color": 0 }
  ]
}"#;

const VALID_PATCH_YAML: &[u8] = br#"master: {}
layers:
  - filename: color-bars.mp4
    source_path: cos-sha256://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/1024
"#;

const VALID_PROXY_OBSERVATION: &[u8] = br#"{
  "sampled_frames": 600,
  "visible_layers": 8,
  "frame_budget_micros": 16667,
  "decode_p95_micros": 12000,
  "upload_p95_micros": 2000,
  "frame_age_p95_micros": 8000,
  "delivery_hold_p95_micros": 0,
  "delivery_hold_peak_micros": 0,
  "dropped_frames": 0,
  "pending_frames_peak": 1,
  "hardware_decode_active": false,
  "zero_copy_active": false
}"#;

fn bench_study_parse(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("study_document");
    group.throughput(Throughput::Bytes(VALID_STUDY.len() as u64));
    group.bench_function("parse_valid_minimal_json", |bencher| {
        bencher.iter(|| {
            let parsed = study::StudyDocument::from_json_bytes(black_box(VALID_STUDY));
            black_box(parsed).expect("benchmark fixture remains valid")
        });
    });
    group.finish();

    let mut group = criterion.benchmark_group("patch_yaml_boundary");
    group.throughput(Throughput::Bytes(VALID_PATCH_YAML.len() as u64));
    group.bench_function("parse_hostile_boundary_valid_minimal", |bencher| {
        bencher.iter(|| {
            let parsed = yaml_boundary::parse_patch_yaml_value(black_box(VALID_PATCH_YAML));
            black_box(parsed).expect("benchmark patch fixture remains bounded and valid")
        });
    });
    group.finish();

    let mut group = criterion.benchmark_group("proxy_planner");
    group.throughput(Throughput::Bytes(VALID_PROXY_OBSERVATION.len() as u64));
    group.bench_function("parse_validate_and_assess_observation", |bencher| {
        bencher.iter(|| {
            let observation: proxy::ProxyPlaybackObservation =
                serde_json::from_slice(black_box(VALID_PROXY_OBSERVATION))
                    .expect("benchmark observation remains valid");
            black_box(proxy::assess_proxy(observation)).expect("proxy assessment remains valid")
        });
    });
    group.finish();

    let mut group = criterion.benchmark_group("latest_only_batching");
    group.throughput(Throughput::Elements(64));
    group.bench_function("coalesce_64_requests_and_publish_one", |bencher| {
        bencher.iter(|| {
            let mut gate = publication_gate::LatestOnlyPublicationGate::default();
            for _ in 0..64 {
                black_box(gate.request());
            }
            let newest = gate.claim_latest().expect("batch has a newest request");
            assert!(gate.try_publish(newest));
            black_box(gate.published_generation())
        });
    });
    group.finish();

    let mut group = criterion.benchmark_group("frame_selection");
    group.bench_function("accepted_frame_half_window", |bencher| {
        bencher.iter(|| {
            black_box(frame_selection::accepted_frame_remains_selected(
                black_box(42),
                black_box(10.0),
                black_box(Some(42)),
                black_box(Some(10.0)),
                black_box(59.94),
            ))
        });
    });
    group.finish();

    let policy = photosensitivity_advisor::AdvisorPolicy {
        transition_threshold_q: 4_000,
        red_saturation_q: 40_000,
        red_dominance_q: 12_000,
        min_affected_cells: 384,
        min_reversal_cells: 384,
        min_red_cells: 384,
        window_ticks: 120,
        attention_transition_events: 2,
        elevated_transition_events: 4,
        elevated_reversal_events: 2,
        elevated_red_events: 2,
        elevated_sustained_ticks: 4,
    }
    .validate()
    .expect("benchmark policy remains valid");
    let raster = [17_u8, 220, 41, 255].repeat(256 * 144);
    let mut reference = photosensitivity_advisor::PhotosensitivityCpuReference::default();
    let mut group = criterion.benchmark_group("color_conversion_reference");
    group.throughput(Throughput::Bytes(raster.len() as u64));
    group.bench_function("pinned_srgb_to_linear_lattice", |bencher| {
        bencher.iter(|| {
            black_box(
                reference
                    .analyze_rgba8_srgb(black_box(&raster), 256, 144, policy)
                    .expect("reference raster remains valid"),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, bench_study_parse);
criterion_main!(benches);
