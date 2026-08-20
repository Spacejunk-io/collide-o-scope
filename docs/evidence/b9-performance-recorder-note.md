# B9 — the performance recorder

**Claim.** The instrument records gestures rather than pixels: while recording
is armed, every accepted authored value edit is written down as a `(tick,
param_address, value)` event on the 30 Hz authoring reference, and a finished
take can be played back in real time — or replayed offline, frame-indexed —
against completely different footage. The law is derived from BENDR (MIT,
© 2026 Steve Blythe, `p42_capture.js`); the house adaptation records accepted
edits at the coalesced drain rather than change-sampling a flat control
surface, addresses time in reference ticks rather than a 24 Hz wall-clock
accumulator, and holds that **the patch is the opening state**: a take rides
whole inside the patch that carries it, so no synthetic keyframe exists and
"same take, same patch" is a complete replay contract.

## The portable contract (`src/performance_track.rs`)

The gesture-track substrate, arm for arm: `PERFORMANCE_REFERENCE_FPS` *is*
`TEMPORAL_REFERENCE_FPS` (a source-text test refuses a second literal), events
are quantized codes only (8 bytes: `tick:u32le | address:u16le | value:u16le`),
the checksum is SHA-256 over a domain-separated explicit little-endian field
stream (`collide-o-scope/performance-take/v1\0`) with the `truncated` and
`incomplete` honesty flags *inside* the digest, serde is bounded on both sides
with hand-validated flat address encoding (serde ignores `deny_unknown_fields`
on tagged enums, so the address codec is written by hand and refuses hostile
extra fields), and the recorder clock derives its tick before adding the
frame's delta so the first accepted frame lands at tick 0.

What is new in kind is the **address table**: a take interns up to 256 distinct
control addresses, each a closed typed `PerformanceControl` (append-only codes
0–14) plus the `PerformanceValueLaw` its value lane is quantized against —
`Unit {min,max}` (Q16 lattice over the declared range), `Discrete {vocab}`
(token index), `Toggle`, and `Stepped {min,max}` (exact integers, because
integer-valued appliers reject a non-integer wire number). The law is captured
at first sight and hashed with the events: a take's lattice can never shift
under its own stream. Layers are addressed by **saved stack position** — the
patch-persistent identity morph slots already use — never by process-lifetime
live IDs, which a patch load deliberately re-mints.

## The value-law oracle

`App::performance_value_law_for` declares each recordable control's lattice by
consulting the engine's own tables: `modulation::target_range` for every
continuous family (master effects by wire name, layer effects and pattern
scalars through the `layer1_…` suffix table, temporal through the
`fb_*→temporal_fb_*` / `disp_*→display_*` name maps), the owning enums' `ALL`
tables for every discrete vocabulary (blend modes, wipe patterns, back
colours, pattern shapes — a widened vocabulary widens the law without a second
hand-list), and the validators' own integer clamps for the stepped laws.
`None` is a *deliberate* refusal, counted and published, never guessed at:
seeds and algorithm identities, `score_loop_driver` (names a live layer),
motion sources/qualities/carriers (they rewrite what the block observes), and
the imperative reset tokens. Safety controls — blackout, freeze, pause —
topology, and routes are outside the recordable vocabulary entirely, by law.

## Recording: the drain tap and the acceptance gate

The record tap sits on `handle_web_action_inner_with_feedback`, the one seam
every final application funnels through — the drained browser batch, the
downbeat release, native RECOVERY, and the transform gizmo — so a take stores
what the program actually did: a coalesced-away value never dispatches and so
never records, and a batch remainder dropped by a snapshot/look barrier
records nothing. Staged edits commit at the same accepted, program-advancing
frame gate the temporal and gesture recorders share (as a free function over
disjoint fields, because the renderer borrow is live there), so a rejected
frame or a frozen program neither consumes a tick nor records an edit. Arming
starts a fresh take at tick zero (BENDR's own law); disarming stamps the
declared length so trailing silence stays part of the loop period; a patch
captured mid-recording carries the take marked explicitly incomplete inside
the hashed flags.

## Replay: delegation, not reimplementation

Playback compiles the take once at arm — each event becomes a real `WebAction`
with the layer position bound to the live stable ID occupying that position —
and dispatches due events per frame through
`handle_web_action_inner_with_feedback`, the transform-gizmo delegation seam,
so Morph ownership transfer, beat-latch materialization, engine sanitize, and
every refusal apply exactly as they would to a hand-made edit, while
`MutationOrigin` law keeps every replayed edit out of manual history (proven:
`undo_label()` stays `None`). A guard flag keeps the record tap from observing
the replayer's own dispatches. An address the program cannot resolve at arm
degrades to a **named no-op** — `layer_effect:3:pixelate` in the snapshot's
`degraded` list — and is never retargeted (the stale-ID law). Replay dispatch
runs after the downbeat release, only while the program advances: a take is an
automatic clock, and Pause freezes automatic clocks. The playhead advances at
the acceptance gate; the loop flag rewinds cursor and clock to tick zero.

## Closure

- **Patch**: `performance_take` carried whole beside `gesture_track`,
  skip-serialized when absent (pre-B9 patches keep their bytes and canonical
  hashes), gated by the document's own checksum-verifying deserializer, and
  restored strictly after the generation barrier. The three authored
  generation barriers (patch load, Apply Look, broad revert) clear the
  recorder; a source cut deliberately does not — the gesture asymmetry.
- **Wire**: `set_performance_recording` / `set_performance_playback` (both
  revision-guarded at ingress and at dispatch) and `clear_performance_take` —
  uncoalesced priority barriers, refused inside `Quantized` batches at the
  server gate and the engine latch, excluded from manual history in both
  classifiers.
- **Snapshot**: additive `performance_recorder` block (the `performance` name
  belongs to the clip/scene subsystem) with mode, counts, length, playhead,
  loop, truncation, the unsupported/rejected honesty counters, the cached
  checksum (empty take publishes none), and the named degraded list.
- **Panel**: the TAKE RECORDER group beside GESTURE FIELD (second-column group
  pin 7 → 8); buttons only, so the 198/21 range pins do not move; the fast
  counters sit outside the live announcement region; none of the three
  transports is in `QUANTIZABLE_ACTIONS`.
- **Morph/Dice/generator/modulation**: a take is authored topology — no
  modulation address of any kind reaches it, Morph slots do not capture it,
  and Dice and the generator preserve it exactly. `GENERATOR_VERSION` stays
  "12".
- **Export**: the job carries the take document, verifies its checksum before
  the first frame, replays due events at
  `export_temporal_reference_tick(frame, fps)` into the authored bases
  through the *same appliers the live arms use* — `EffectsSnapshot`,
  `apply_spatial_transform_edit`, `apply_motion_param`, the newly extracted
  `apply_temporal_wire_edit` (the `set_temporal` match body moved whole out of
  the action arm, so live and export mutate `TemporalParams` through identical
  code), `BusMixerEdit`, `PatternSynthEdit`, and the shared layer scalar
  clamps — then publishes `<output>.performance.json` through the staged
  no-replace commit, cleanup-coupled to the video and retired at the output
  claim. Export replays once, straight through: the loop flag is live
  transport, not part of the take. The motion sidecar schema stays 6 (a
  separate sidecar file is the `.gesture.json` precedent).

## Deliberate boundaries (v1)

- Values only: node/group rack params, text-page values, pad/gyro/audio/MIDI/
  LFO/routing configuration, and morph law/glide/capture are not yet
  recordable; the address vocabulary is append-only, so each can join without
  breaking a stored take.
- MIDI/OSC direct-control edits (`apply_automation_control`) bypass the wire
  action seam and are not recorded; the drain tap records what the *program's
  action stream* did.
- An engaged Morph pair offline keeps its established owned-base law: live
  recording already transferred ownership at edit time, so a patch captured
  after a recording session typically carries no engaged pair over the
  touched families; either way replay is deterministic.
- A refused live edit (e.g. a stale composition revision from another
  controller) may still enter the take at its dispatched value; replay
  revalidates through the same doors, so nothing can replay that the engine
  refuses.

## Proof

Hosted: 20 law-module tests (lattices, hostile serde, checksum coverage of
the address table, flags inside the digest, cursor monotonicity, clock law,
frame-grouping invariance); the oracle answering from the engines' own tables
(including alpha_cut's exclusion and the master-only optics refusing layer
scope); record-tap capture/commit/skip-and-count; revision guards and mutual
transport exclusion; replay through the inner seam with undo exclusion; loop
rewind; per-address degradation by name; barrier/source-cut asymmetry with
restore-after-barrier source pins; patch carriage with incomplete-capture
marking and checksum-gated section rejection; never-latchable refusals at
both gates; the shared-applier source audits; the sidecar no-replace/
retirement/cleanup laws; offline applier behavior for every family with
stale-position and absent-morph no-ops; panel wiring and additive snapshot
contracts. Opt-in (GPU + ffmpeg + `videos/audit.mp4`):
`render_performance_recorder_pipeline` — the `_untaken` twin must decode
differently (the take reaches the pixels) and the `_repeat` render must
decode identically (record/replay determinism), with the published sidecar
byte-verified against the carried document.
