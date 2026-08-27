# Monitor bay deferred probes — implementation and proof note

Date: 2026-08-27  
Topic: `feat/monitor-bay-deferred-probes`  
Pinned integration base: `4c0618e2ea65d4db8146fa985a6d93eafa508257`

This tranche closes perfection handover §3.5 by appending its three named
diagnostic sources to the B11 monitoring bay. It does not widen the bay's
audience, persistence, modulation, performance-recorder, or export boundary.

## Permanent vocabulary and producer identity

`MonitorProbe` is closed and append-only. Its order and numeric codes are:

| Code | Token | Exact producer meaning |
| ---: | --- | --- |
| 0 | `program` | Finished pre-blackout programme picture |
| 1 | `program_tap` | Honest N-1 programme re-entry tap |
| 2 | `gesture_canvas` | Presented gesture-canvas donor |
| 3 | `ntsc_line_state` | The sync latch's live applied per-line offset table |
| 4 | `melt_band_mask` | The retained bus-melt band mask for that rendered frame |
| 5 | `motion_field` | Master's exact admitted primitive or derived motion field |

Codes 0–2 retain their established values. Future probes append after code 5;
none may rename, renumber, or reuse an existing token. The shared
`MonitorProbe::try_from_str` table remains the wire gate and applier parser;
unknown and near-miss tokens are refusals, never aliases.

The `motion_field` token has one meaning only: the exact parity selected by
Master's `MotionScopeBindings` after the motion planner's admission decision.
Primitive motion uses that field's staged vector/gate parity and grid. A Field
Collider recipient uses the collider's staged derived parity and output grid.
An absent or unmaterialized Master field reports `unavailable`; the probe never
rebinds to a raw, layer, donor, nearest, or otherwise convenient field.

## Independent diagnostic oracle

All diagnostic pixels are exact RGBA8 with alpha 255. Non-finite scalar input
is sanitized to the neutral zero state before encoding, unit results clamp to
`[0,1]`, and bytes use nearest rounding.

- Sync-line state is a fixed 128×72 grid. Output row `y` reads source row
  `floor(y × source_rows / 72)` and repeats that line's colour across all 128
  columns. A table shorter than 72 rows is unmaterialized and yields opaque
  black. Each offset is divided by `SYNC_LATCH_MAX_OFFSET`: positive magnitude
  is red, negative magnitude is blue, and zero is black.
- Melt band mask is the engine shader's one shared `melt_band_sample` result,
  including the existing creep law, copied equally to red, green, and blue.
- Motion field uses the engine-owned `MOTION_MAX_UV_PER_SECOND`: red is
  `(vx / MAX + 1) / 2`, green is `(1 - vy / MAX) / 2`, and blue is
  `confidence × visibility`.

Pure functions in `src/monitor_bay.rs` are the byte oracle for renderer proof;
they own no GPU state and cannot make a source available.

## Arming, cadence, and frame ordering

- The same observer gate remains authoritative: visible enabled native bay OR
  a fresh browser watcher. A never-armed process creates no monitor stage.
- Disarm clears instruments and cadence debt, revokes a pending melt-mask
  request, invalidates the current mask, and advances the readback generation.
  A mapped stale result is recycled immediately; an already in-flight map may
  finish only to be recycled. Re-arm cannot publish it.
- The armed fact is refreshed after browser, native, and controller action
  drains and before diagnostic-only work. A disable accepted in that frame
  therefore requests, encodes, and maps nothing further.
- Sampling remains 10 Hz on accepted programme seconds. Pause and rejected
  frames add no debt. A busy two-slot pool drops cleanly and retains cadence
  debt for the next accepted frame.
- A due melt mask is requested before the creative encoder reaches B8. The
  accepted frame encoder is submitted before monitor scheduling, so the
  reduction observes that frame's mask. A rejected frame cannot publish it.
- Motion schedules after the accepted advanced frame has encoded but before
  `commit_frame_history`, so it borrows the exact staged Master parity rather
  than a later committed or fallback field.

## Exact retained allocation ledger

The original lazy bay machine remains one 36,864-byte reduction target plus
two 36,864-byte staging buffers: 110,592 nominal bytes. Its image path retains
one target/view, two buffers, one shader, one bind layout, one pipeline layout,
one pipeline, and one sampler. Per-sample source bind groups are transient.

- `ntsc_line_state` writes the already-reduced CPU RGBA bytes directly into an
  idle staging buffer: zero additional texture, shader, pipeline, or buffer.
- The deferred texture paths share one lazy 36,864-byte linear RGBA8 exact
  target. `melt_band_mask` lazily adds its exact-view shader/layout/pipeline.
  `motion_field` lazily adds its motion shader/layout/pipeline and one 16-byte
  uniform containing the engine-owned velocity bound.
- The melt producer separately adds one lazy 128×72 RGBA8 mask (36,864 bytes),
  one view, bind group, shader, pipeline layout, and pipeline. Its full-frame
  history remains independently armed by creative melt; observing inactive
  melt creates only the small diagnostic surface and returns opaque black.

All diagnostic resources sit outside the unchanged 30-full-frame-surface
floor. Nothing is preallocated for a deferred path before that path wins an
armed due sample.

## UI, persistence, and negative boundaries

- The panel adds exactly three `<option>` elements and extends the event and
  snapshot allowlists to the same six tokens. No range control was added:
  the exact pins remain HTML 208 and JavaScript template 24.
- The bay remains host-session state. No patch field, capability row,
  modulation address, Morph/Look/Dice surface, performance-take address,
  audience paint, or export arm was added.
- `program` stays the new-process default. Existing programme pixels and the
  original three probe meanings did not move.
- The protected root artifacts remain operator-owned. `videos/audit.mp4` was
  absent throughout and was neither copied nor minted.

## Observed proof

Host/toolchain: Windows x86_64 MSVC; `rustc 1.98.0`
(`88d9e12ae`, 2026-08-18); Cargo `1.98.0`; Node `v26.5.1`; shared FFmpeg
`9.0.1`; AMD Radeon RX 6950 XT driver `32.0.21045.5002`; Vulkan backend for
the release export receipt.

Focused hosted contracts:

- `cargo test --locked --all-features monitor_bay -- --nocapture` — 16 passed,
  0 failed, 2 explicitly ignored GPU cases.
- Vocabulary/order/codes, hostile near misses, sync row mapping, finite/clamp
  laws, allocation snapshots, shader-owner pins, protocol/static topology, and
  stale-generation behavior all passed.
- `node --check static/app.js` and `node --check docs/ui-ux/wireframe.js`
  passed.

Explicit physical-GPU receipts:

- `gpu_deferred_probes_preserve_exact_bytes_and_motion_color_law` — passed on
  production `RG16Float` vector and `RG8Unorm` gate formats. It proves direct
  CPU bytes, exact retained-RGBA bytes, motion colour bytes, the shared busy
  pool, and an end-to-end pre-invalidation completion rejection.
- `gpu_melt_diagnostic_mask_matches_the_cpu_band_and_creep_oracles` — passed.
  It proves inactive opaque black, active GPU/CPU parity to one RGBA8 LSB,
  current-frame validity, disarm revocation, and the exact lazy charge. The
  fixture initially exposed an invalid `rows_per_image` constant in its
  variable-size readback helper; the helper now uses its actual height and the
  rerun passed.
- `gpu_sync_latch_shears_each_line_by_its_own_offset_and_default_encodes_nothing`
  — passed, binding the borrowed applied-offset table to the existing physical
  sync-latch producer.

Default programme-pixel A/B:

- The pre-existing self-provisioned
  `render_export::effects_audit::render_pattern_synth_pipeline` ran in release
  mode with Vulkan at pinned base `4c0618e...` and on this implementation.
  It uses only the built-in Pattern Synth, performs no media lookup, and emits
  three 320×180, 24 fps, one-second outputs through the real finished-Program
  export path.
- Normalized FFmpeg `framemd5` sequences (only `#software:` omitted) were
  byte-identical across all 24 frames for each of
  `audit_pattern_synth.mp4`, `_default.mp4`, and `_repeat.mp4`. The fixture's
  own authored-vs-default difference and authored-vs-repeat equality also
  passed.
- Supplemental self-provisioned performance-recorder-v2 A/B likewise matched
  all 24 decoded frames for each taken, untaken, and repeat output.
- This A/B proves pre-existing finished Programme pixels did not move. It does
  not pretend export arms the monitor bay; the explicit GPU fixtures above are
  the monitor readback proof.

The complete CI-form topic gate is recorded in the closing fields after its
final run; focused and opt-in receipts do not replace it.

## Closing fields

- Topic implementation commit: **`6aa9d94`**
- Topic receipt commit: **PENDING**
- Integration commit on `feat/web-control-panel`: **PENDING**
- Exact-commit CI: **PENDING**
- Hosted full gate: **OBSERVED PASS** — the exact six-command CI-form gate
  passed: formatting and both JavaScript parsers; all-target/all-feature
  compile; 2,132 tests passed with zero failures and 163 explicitly ignored
  external/GPU seats; all six bench harnesses reported success; clippy passed
  with `-D warnings`
- Protected-root and `videos/audit.mp4` recheck: **OBSERVED PASS** — the three
  named root artifacts remain the only untracked root files, with SHA-256
  `494b63ad...ab1eea4`, `ee1cfc47...13d034a0`, and
  `2b51dda2...722630a4`; `videos/audit.mp4` remains absent

## Deliberate non-claims

This tranche does not claim an export monitor arm, physical audience output,
new patch/control surface, or a live UI screenshot. It does not make a missing
Master motion field available, allocate a source while never armed, or turn a
compiled path into a physical receipt beyond the three explicitly run GPU
fixtures above.
