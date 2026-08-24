# P8 adversarial verification closure receipt

Date: 2026-08-24<br>
Disposition: **retained partial closure; the full P8 promotion gate remains open**

This receipt separates executed evidence, configured campaigns, synthetic
stand-ins, and unexecuted physical gates. No configured workflow is reported as
an observed run.

## Bounded fuzz surface

There are nine cargo-fuzz binaries. Every target accepts only an in-memory byte
slice with an explicit cap; none accepts a host path or opens a user file.

| Target | Shared production boundary | Seed corpus | CI `max_len` |
| --- | --- | --- | ---: |
| `study_document` | Study JSON parser | minimal versioned Study | 1 MiB |
| `publication_gate` | latest-only state machine | short operation stream | 4 KiB |
| `patch_yaml` | hostile patch YAML boundary | minimal patch YAML | 1 MiB |
| `controller_profile_midi` | profile/action JSON plus complete-message MIDI decode | minimal profile | 257 KiB |
| `osc_packet` | OSC packet decode plus OSC configuration | address/configuration seeds | 16 KiB |
| `recovery_journal_record` | journal record encoder and header/checksum/order scanner | structured YAML payload | 1 MiB |
| `proxy_metadata` | proxy settings, playback observation, and cache-key wire laws | settings/observation | 256 KiB |
| `web_action_json` | bounded, duplicate-key-rejecting WebAction JSON preflight now used by the WebSocket server | direct and quantized actions | 16 KiB |
| `motion_sidecar_json` | bounded schema/list validator now run on the exporter's serialized sidecar before publication | schema-9 minimal sidecar | 1 MiB |

The patch target and loader share `src/patch/yaml_boundary.rs`: 32 MiB file,
depth 64, 250,000 nodes and collection entries, 4 MiB scalar, and 500,000
lexical-token limits. The fuzz campaign adds the tighter 1 MiB ingress cap.
The recovery target uses the exact binary scanner with `serde_yaml::Value` only
as a stand-in for renderer-heavy `PatchState`; hostile patch semantics are
owned by the separate exact YAML target.

`src/bounded_json.rs` is the shared duplicate-key-rejecting parser underneath
the WebAction and motion-sidecar boundaries. Production WebAction parsing uses
that boundary and then deserializes the real `WebAction` enum. The fuzz crate
exercises the exact preflight but cannot instantiate that application-coupled
enum. Motion-sidecar schema version and list/byte caps have one production
authority; a source test parses the checked-in corpus at that same constant, so
a future schema bump cannot silently leave the fuzz seed inert.

`.github/workflows/adversarial.yml` pins nightly `nightly-2026-08-20` and
`cargo-fuzz 0.13.2`. Pushes, pull requests, release tags, and manual dispatches
run 10,000 cases per target with `-timeout=5`, `-rss_limit_mb=2048`, and the
target cap. The weekly schedule gives each of nine targets an independent
one-hour matrix job with a 75-minute job deadline and `fail-fast: false`.

## Deterministic state and concurrency

`src/history.rs` executes exactly 10,000 proptest sequences with fixed seed
`0xC011_1DE0_5C0F_E001`. Its reference model covers add/remove/reorder/group/
reroute/Morph/scene/quantized/automation/undo/redo and asserts stable identity,
bounded and finite state, serialization, live-state atomicity, and manual
history equivalence.

`tests/loom_publication_gate.rs` models cancellation, stale generation, burst
coalescing, and exactly-one publication ownership. A local deliberately broken
variant checks without consuming a token; Loom observes two owners and raises
`broken gate permits duplicate publication ownership`. The expected-panic test
proves the model catches that seeded mutation.

Important limitation: `LatestOnlyPublicationGate` is used by the fuzz, Loom,
benchmark, and synthetic soak targets, but live decoder/proxy/recorder/device
workers have not migrated to it. Loom therefore proves the shared proposed
kernel, not every current production mailbox implementation. Production
migration or separate exact models remain a keep gate.

## Fault-seam inventory

Existing deterministic tests cover allocation/ledger refusal and upload
validation/OOM (`video/payload`), worker stall and fixed shutdown deadline
(`recovery_journal`), disk-full/prepublication/rename/collision/cancel/crash
(`durable_file`, `show_bundle`), proxy-cache corruption (`proxy_worker`), stale
library generation, device-loss phase transitions (`gpu_recovery`), and export/
record cancellation (`render_export`). The 14-test recovery group was rerun for
this receipt. Other groups remain inventory backed by their tranche receipts,
not a claim that each was rerun here.

An explicit WebSocket reconnect-storm injector, synthetic decoder EOF/error
injector independent of live media, and exact resize/surface-loss injector were
not added in this closure. Device-loss and real EOF fixtures are adjacent
coverage, not substitutes for those three named fault campaigns.

## Synthetic soak

`src/bin/p8_synthetic_soak.rs` is a deterministic CPU/planner harness. Its
default CLI duration is 3,600 seconds; `--iterations` is the exact replay mode.
It preallocates fixed storage for 1/3/8-layer H.264-color-bar, VP9-color-bar,
and VFR-color-bar stand-ins, advances fixed Temporal/Mosh/VHS state, injects
finite controller traffic and output toggles, exercises proxy/recording
publication and stale generations, reports interval telemetry, bounds pending
publication slots, drains to zero, and emits a final SHA-256.

The executed 100,000-step smoke used seed `C0111DE050A80001`, visited all three
layer counts, retained 576 fixed temporal slots and 24 layer objects, peaked at
2 pending publications, drained to 0, and produced state SHA-256
`9bf55c70198ceef6875775dca995e6eafd87d375b5933aa638587eef6aaa53d1`.

This is deliberately not FFmpeg/wgpu/RSS evidence. It uses fixed-storage
stand-ins rather than decoding H.264/VP9/VFR, running the real effects, or
measuring process RSS/GPU-ledger/resource objects. The audit's physical
one-hour mixed-media gate remains unexecuted.

## Benchmarks

One Criterion harness now has six production-seam groups: Study parsing, patch
YAML parsing, proxy planning, 64-to-1 latest-only batching, accepted-frame
selection, and the pinned sRGB-to-linear photosensitivity CPU reference.

Executed on Windows 11 Pro 10.0.26200 build 26200; Intel Core i9-13900K, 32
logical processors, 34,088,050,688 bytes RAM; rustc 1.98.0 `88d9e12ae`
(`x86_64-pc-windows-msvc`, LLVM 22.1.8); cargo 1.98.0 `797e8a9bc`.

With `--sample-size 10 --warm-up-time 1 --measurement-time 1`, Criterion
reported estimate intervals:

- Study parse: `[1.1208, 1.1536, 1.2163] us`;
- patch YAML boundary: `[4.0955, 4.1456, 4.2159] us`;
- proxy parse/validate/assess: `[345.46, 350.97, 356.87] ns`;
- coalesce 64/publish 1: `[18.934, 19.070, 19.191] ns`;
- accepted-frame selection: `[4.0029, 4.0438, 4.0803] ns`;
- color reference: `[44.103, 44.679, 45.486] us`.

These short Criterion intervals are smoke measurements, not p95/p99 latency
distributions and not an approved comparative baseline. Actual GPU command
encoding is adapter-specific and has no dependency-light benchmark here; its
signed local hardware receipt and the statistically material p95/p99 rejection
gate remain open.

## Executed commands

Native commands used the prescribed FFmpeg, LLVM, and MSVC environment.

| Command | Result |
| --- | --- |
| `cargo check --locked --manifest-path fuzz/Cargo.toml --bins` | pass; all 9 fuzz binaries compile |
| `cargo check --locked --bin collide-o-scope` | pass |
| `cargo test --locked --bin collide-o-scope patch::editor::tests:: -- --nocapture` | 7 passed |
| fixed-seed history state-machine filter | 1 passed; 10,000 cases in 5.34 s |
| `cargo test --locked --test loom_publication_gate -- --nocapture` | 5 passed, including caught mutant |
| recovery-journal test filter | 14 passed |
| WebAction boundary test filter | 2 passed |
| motion-sidecar boundary test filter | 1 passed |
| frame-selection test filter | 1 passed |
| `cargo test --locked --bin p8_synthetic_soak -- --nocapture` | 2 passed |
| `cargo run --quiet --locked --bin p8_synthetic_soak -- --iterations 100000` | pass; digest and bounds above |
| `cargo bench --locked --bench study_parse --no-run` | pass |
| final six-group Criterion smoke command above | pass |
| Python `yaml.safe_load` of the adversarial workflow | pass |

## Open gates and stop reasons

Pinned nightly and cargo-fuzz are installed locally, but the Windows fuzz
executables cannot start: the pinned nightly is LLVM 23.1 while the installed
ASan runtime is LLVM 22. Without that DLL Windows returns
`STATUS_DLL_NOT_FOUND`; with the LLVM-22 DLL it returns
`STATUS_ENTRYPOINT_NOT_FOUND` for newer sanitizer cleanup exports. No local
fuzz execution is claimed. The exact-commit Linux workflow is the next
authoritative 10,000-run and scheduled one-hour evidence.

FFmpeg container-probe metadata remains uncovered. The pure proxy JSON target
is not a substitute for fuzzing libavformat container metadata. Exposing that
boundary requires a bounded sandbox/in-memory fixture contract; P8 does not
fuzz arbitrary user files or paths.

The full-enum WebAction semantic layer is covered by ordinary production tests
after the fuzzed preflight, but not instantiated inside the fuzz crate. Live
worker Loom migrations, the three named fault injectors above, actual command
encoding, an approved repeated p95/p99 baseline, and the physical one-hour
mixed-media/RSS/GPU-ledger/deadlock/artifact gate remain open. Physical
MIDI/OSC devices and heterogeneous GPU/output fixtures were not exercised.

No fuzz-discovered crash exists to minimize and seed; the new corpora are
static reachability/schema-drift seeds, not disguised regression discoveries.
These stop conditions are not waivers. The independently safe deterministic
coverage is retained, but P8 must not be promoted as complete until the open
campaigns have observed receipts.
