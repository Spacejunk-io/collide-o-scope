# P5 — copy-free ordinary LegacyExact accumulation

Date: 2026-08-24

## Landed scope

The ordinary `LegacyPostComposite` renderer now treats composite slots 0 and
2 as two retained accumulator identities while slot 1 remains the local layer
output. Every layer composite reads the logical current accumulator and slot 1,
writes the alternate accumulator, and swaps the logical identity. The global
Master pass performs the same swap. No full-frame texture copy is encoded by
either operation.

The immutable frame decision counts the ordinary wet composites plus the
optional global Master pass before encoding. It selects the initial identity
from that count's parity, so the completed plan always lands on the established
slot-0 engine publication seam. This gives Temporal, selective bypass, Mosh,
VHS, display stages, recording, StageMap/output, and offline-export callers the
same stable downstream identity without a normalization copy.

The historical and new topologies perform the same number and order of shader
passes and `Rgba8UnormSrgb` attachment stores. Only the identity of the retained
texture changes. The removed `copy_texture_to_texture` operations never added a
quantization boundary, so deleting them preserves the historical sRGB rounding
law.

## Numeric receipt

`FullFrameWorkCounters` records completed ordinary plans independently from GPU
object-construction counters:

- `planned`: immutable ping-pong render/copy pass counts and copy bytes;
- `executed`: the work actually encoded after all fallible preparation succeeds;
- `legacy_baseline`: the same render passes plus the predecessor's one
  full-frame copy after each layer composite and Master transform.

At 1920×1080, one RGBA8 full-frame payload is 8,294,400 bytes. With the ordinary
global Master pass present, the following per-frame predecessor copies are
eliminated:

| Visible wet layers | Old copy passes | Old copy bytes | New copy passes/bytes |
|---:|---:|---:|---:|
| 0 | 1 | 8,294,400 | 0 / 0 |
| 1 | 2 | 16,588,800 | 0 / 0 |
| 3 | 4 | 33,177,600 | 0 / 0 |
| 8 | 9 | 74,649,600 | 0 / 0 |

Counters are cumulative and include only completed ordinary accumulator plans.
They deliberately exclude copies whose purpose is a different ownership or
time domain.

## Copies retained by semantic purpose

- Temporal clean-history and feedback copies preserve observations across
  frames; pause/freeze restore is also a time-domain transfer.
- `encode_program_history_copy` publishes clean Program N into routed history
  N−1.
- transactional Program-tap, blackout-hold, selective-bypass, and post-Mosh
  candidate publication copies keep rejected work from replacing an accepted
  audience frame.
- recorder, selective NTSC, Codec-Mosh, Spout, and CPU readback copies cross the
  GPU/CPU or distinct-consumer ownership boundary.
- routed matte and conditional per-layer Master paths retain slot-0
  materialization. Their three-surface topology simultaneously needs an
  accumulated base, a local or Master-processed overlay, and a destination.
  Reusing either source as the attachment would be invalid; allocating a fourth
  mandatory full-frame surface was rejected by the renderer memory floor.

## Acceptance evidence

- Pure parity/topology tests cover 0, 1, 3, and 8 wet layers with and without
  the global Master pass, prove final slot 0, prove distinct sampled/attachment
  identities, and reconcile planned/executed/predecessor pass and byte counts.
- Source-contract tests prove the ordinary composite and accumulator Master
  bodies contain no full-frame copy, while the routed semantic materialization
  remains named and present.
- The existing live/export real-GPU fixed golden exercises local effects,
  multi-layer blending, global Master, Temporal compatibility, opaque resolve,
  readback, and offline parity at 24/30/60 fps. Exact command results are listed
  in the final validation section once run on the audit host.

## Fusion stop gate — fired

No pass fusion is shipped.

The historical local-FX pass stores into an `Rgba8UnormSrgb` attachment and the
composite pass samples that stored value through an sRGB view. Naively fusing
the two shaders keeps the intermediate in floating point and removes that
hardware encode/decode quantization boundary. Reproducing the backend's exact
store rounding in WGSL has not been proven across admitted adapters, so such a
plan cannot be labelled Exact.

There is also no paired affected-frame p95/p99 benchmark receipt establishing
the audit's required ≥10% p95 improvement without p99 regression. The existing
eight-layer debug smoke ceiling is a safety test, not that statistical proof.
Both required gates therefore fail independently. The immutable-plan fusion
prototype stops at the decision record; no fused shader, pipeline, feature
flag, or non-Exact mode is retained.

## Final validation

The decoder fixture migration completed and the shared test target compiled.
Exact audit-host commands and results:

- `cargo check --tests`: exit 0.
- `cargo test --bin collide-o-scope p5_`: 6 passed, 0 failed. This includes
  0/1/3/8-layer parity, with/without-Master publication, 720p/1080p/4K copy-byte
  ledgers, sampled/attachment identity, history resolver, source-copy
  rejection, and independent planned/executed/predecessor counters.
- `cargo test --bin collide-o-scope
  render_export::tests::gpu_two_layer_live_and_export_full_stack_matches_fixed_golden_at_24_30_and_60_fps
  -- --ignored --exact --nocapture`: 1 passed, 0 failed. Live and offline pixels
  matched the fixed golden and one another at 24/30/60 fps.
- `cargo test --bin collide-o-scope
  renderer::state::temporal_state_tests::gpu_temporal_originals_topology_interpolation_atlas_and_startup_goldens
  -- --ignored --exact --nocapture`: 1 passed, 0 failed.
- `cargo test --bin collide-o-scope
  renderer::state::temporal_state_tests::gpu_refresh_garden_gates_recurrence_max_hold_freeze_blackout_reset_and_rate_goldens
  -- --ignored --exact --nocapture`: 1 passed, 0 failed.
- `cargo test --bin collide-o-scope
  renderer::composition::tests::recorder_scope_capture_is_post_effects_fifo_warm_and_never_falls_back
  -- --ignored --exact --nocapture`: 1 passed, 0 failed.
- `cargo test --bin collide-o-scope
  render_export::tests::gpu_1080p60_eight_transformed_layers_complete_within_debug_smoke_ceiling
  -- --ignored --exact --nocapture`: 1 passed, 0 failed; measured 11.50 ms
  against the 16.67 ms realtime target and 250 ms debug ceiling. This is a
  smoke result, not the statistical fusion gate.
- Eight focused host-law tests also passed: accepted Program-tap publication
  held under blackout; paused selective-blackout restoration; delayed-Mosh
  invalidation at temporal-dry re-entry; selective-VHS generation reset;
  frozen scope-capture backend selection; single-surface output policy;
  program/media freeze plus hidden blackout evolution; and Advanced temporal
  bypass across both Mosh paths.
- Two additional physical-GPU downstream tests passed: exact/warm StageMap
  identity slicing on the audit adapter and Display Physics parity with
  blackout clearing its wake.

The audit-media Mosh/VHS/performance-recorder render pipelines could not run:
the explicitly required local `videos/audit.mp4` fixture is absent. Their
downstream texture identity remains the established slot-0 publication seam,
proven by the immutable parity plan and source-contract test; no broader claim
is made for those unavailable media renders.
