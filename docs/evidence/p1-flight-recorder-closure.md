# P1 bounded flight-recorder closure receipt

Status: retained source-level support-bundle recorder; physical optical and
instrumentation-overhead gates remain open.

This receipt closes the local flight-recorder portion of P1 without turning
engine submission into photon time or treating a modelled crash as a real
process failure. The recorder is production-wired. Its `try_record` producer
boundary is fixed-shape, performs no serialization or filesystem I/O, and uses
a nonblocking bounded send. At an accepted authored-transaction seam, Main
reuses the canonical SHA-256 fingerprint already produced by history/recovery
in the legacy-named `patch_plan_digest` field. That fingerprint covers the
broader authored world (including stable IDs/selection, base directory,
StageMap, presets, and controller profile), so it conservatively splits some
identical patch plans; it is not claimed as patch-only identity. A retained
signature of stable layer ID, source kind,
and verified content SHA-256/byte length causes the order-independent source-set
digest to rebuild only when the logical source set really changes. An
unverified legacy source contributes a session-local opaque change witness,
advanced at logical-source commits but not at backing-only proxy/resize epochs;
the raw reference is never hashed. The rebuild uses retained facts and never
reads a path. This is accepted-transaction work, not a warmed-frame or
every-frame claim; warm telemetry copies only the cached fixed-size fact.

## Bounded production contract

- `FlightRecorder` owns a fixed 512-entry synchronous channel. Live producers
  use `try_send`; full and disconnected states are counted dispositions rather
  than waits. JSON serialization, rotation, sync, and rename run on the named
  recorder worker.
- Production configuration rotates every 45 seconds, within the enforced
  30–60 second law, and admits at most 3 MiB across two completed rotations
  plus the active rotation. Each file has its own byte ceiling. Completed
  rotations are marker-validated; active or torn files are never reported as
  durable evidence.
- The versioned private recorder directory and create-new private files are
  owned independently of user media. Retention removes only names matching the
  recorder's exact active/completed grammar; an unrelated-file fixture proves
  that other files are never pruned.
- Every writable event is a closed, fixed-shape fact: Host, Stage, Worker,
  Action, Error, ContentIdentity, ResourceLedger, or AdapterCalibration. There
  is no arbitrary string, path, byte-vector, JSON, or message field.
- The rotation header carries the build-identity snapshot (version, commit,
  Rust/Cargo target and versions, FFmpeg/ffprobe identity), plus the most recent
  path-free host, content-digest, resource-ledger, and adapter-calibration
  facts. Later rotations repeat those cached facts so a quiet failure retains
  the last complete context.
- Source digest domain `collide-o-scope/flight-source-set/v2` refreshes at every
  logical add/remove, prepared-source, whole-stack, or verified-identity commit.
  It preserves the patch digest during a source-only refresh. Reorder and
  backing-only proxy/resize changes neither rebuild the source digest nor
  advance the source-only publication generation; a broader authored-world
  fingerprint change may still publish the enclosing content fact.
- Production emits bounded CPU/GPU stage facts, correlated action facts,
  typed errors, content/resource facts, and a decoder-worker aggregate. The
  worker fact derives queue depth/capacity and completion/drop totals from
  bounded live mailboxes; no stable restart counter exists yet, so
  `restart_count` remains the explicit unreported value zero. No worker fact
  can contain a source name or path.

## Privacy and failure proof

`SensitiveDiagnostic` is a one-way destruction boundary for filesystem paths,
access tokens, cookie headers, media bytes, controller secrets, and authored
text. `ErrorFact::redact` retains only a closed domain/code, retryable bit, and
occurrence count. A hostile fixture supplies all six forbidden classes and
asserts that none enters any recorder artifact while aggregate content and
resource facts remain readable.

Two distinct crash gates are retained and named accurately:

1. A torn-file model proves that a malformed active tail is excluded, restart
   cleanup removes only that tail, and the previous completed rotation stays
   byte-identical.
2. A real subprocess starts the recorder, forces a byte-boundary publication,
   opens the next active rotation, signals its parent, and is killed without
   recorder shutdown. The parent verifies the prior completed rotation and
   marker byte-for-byte before and after restart cleanup. The ignored helper is
   not itself a skipped acceptance test; it is invoked by the passing parent.

The real kill gate ran on the audit Windows host. It proves the implemented
Windows process/filesystem path, not every filesystem or abrupt-power-loss
mode on macOS and Linux.

## Focused verification

Executed on the shared Windows workspace after production Worker wiring and
the real killed-subprocess fixture:

- `cargo test --locked --bin collide-o-scope flight_recorder::tests -- --nocapture`:
  9 passed, 0 failed, 1 subprocess helper ignored; the parent process-kill
  test passed.
- `cargo test --locked --bin collide-o-scope app_state_tests::video_decode_worker_flight_fact_is_bounded_and_reports_real_mailbox_state -- --exact --nocapture`:
  1 passed.
- Final lifecycle/adapter/upload/socket pass: 20 exact test executions passed,
  including cancellation at both async WebState locks, complete shutdown
  terminalization, worker-mailbox failure preservation, durable upload/delete
  correlation, directory-drop admission, and source-cache warm/change cases.
  A subsequent seven-test source-identity pass passed: warm-cache and verified
  replacement/reconstruction, verified async mint, unverified add/remove and
  A-to-B changes, identical-source whole-stack/history and prepared-source
  stability, and backing-only digest/generation stability. Core action
  correlation separately passed 7/7.
- `cargo check --locked --bin collide-o-scope`, strict binary and
  all-target/all-feature Clippy, Rustfmt, and `git diff --check`: pass.

## Gates deliberately not claimed

The electrically coupled LED/sensor/display fixture was unavailable, so
physical action-to-photon p50/p95/p99 remain null. The fixed three-layer
instrumentation overhead comparison, ten-minute mixed-media runs, and one-hour
fault/leak soak were not executed. Those stops remain binding in
`p1-action-to-photon-fixture-unexecuted.json` and cannot be inferred from this
recorder receipt.
