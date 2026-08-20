# B11 — the monitoring bay

The difference between "the picture is doing something odd" and knowing which
part of the model is doing it. Preview-only by our own law: waveform,
vectorscope, and PROBE instruments over a low-resolution readback of an
internal signal, rendered to the editor preview and the browser panel and to
no audience surface, at a cost of exactly zero while nobody is watching.

The law is derived from BENDR (MIT, © 2026 Steve Blythe), whose scope dock is
this instrument shipped whole: a 128×72 GPU downscale of the finished
programme read back once per sample, gated on dock-tab visibility, cadenced at
10 Hz, plotted as a Rec.601 luma waveform and a U/V vectorscope with the six
75% colour-bar targets. `src/monitor_bay.rs` is the independent CPU reference
in the `gesture.rs` tradition.

## What shipped

- **The readback stage** (`src/renderer/monitor_bay.rs`,
  `src/shaders/monitor_bay.wgsl`): the B10 `video_analysis` pool at BENDR's
  shipped 128×72 (inside the tranche's ≤160×90 bound — the B10 precedent of
  following the shipped law). Lazily constructed on the first armed sample;
  two-slot FIFO readback pool, busy-drop, strict oldest-first harvest; its
  own tiny buffers (one 36,864-byte RGBA8 target, two 36,864-byte staging
  buffers), deliberately outside the three full-frame audience slots and the
  renderer texture floor (still 30). The reduction is the B10 expression at
  the bay's dimensions; the CPU reference is the generalized
  `modulation::reduce_analysis_grid` (B10's `reduce_video_analysis_grid` now
  delegates to it — one law, two sizes).
- **The cadence** is the B10 law verbatim: three reference ticks (10 Hz) on
  an accumulator fed by accepted program seconds at the program-tap
  acceptance seam, so Pause holds the instruments and a busy pool retries
  without debt. Under blackout the probed slot-2 image is the held
  pre-blackout picture (program memory, the tap's own law): the operator's
  scope keeps showing what the program holds while the audience is dark.
- **The arming predicate** (`App::monitor_bay_armed`): the native preview
  overlay (bay toggle ∧ `native_controls_visible`) OR a fresh browser
  watcher. Unarmed: no pass encodes, no buffer maps, the snapshot block is
  the empty default, and the disarm edge clears the instruments once so a
  re-arm never resurrects a stale picture (the B4 wake-clearing instinct).
- **Browser watchers** (`monitor_watch { enabled }`): the `gyro_stream`
  socket-layer shape — per-client registry on `WebState`, set at the socket
  without a queue round-trip, cleaned on disconnect — hardened with a watch
  timeout (`MONITOR_WATCH_TIMEOUT`, 10 s): the panel re-asserts its
  declaration on a 4 s heartbeat exactly while its MONITORING BAY section is
  expanded and the tab is visible, so a silently discarded tab expires
  server-side instead of pinning the readback armed. The panel declares on
  section toggle, `visibilitychange`, socket reconnect, and the heartbeat;
  repeated `false` is elided so a closed section costs no traffic.
- **The instruments** (`monitor_bay::reduce_instruments`): computed on the
  CPU from the harvested grid — one law, two dumb presenters. Waveform:
  Rec.601 luma on the encoded bytes (the `VideoAnalysisState` constants, the
  same encoded space BENDR plots), one column per grid column, additive
  saturating accumulation. Vectorscope: BENDR's projection
  `u = (b−y)·0.565, v = (r−y)·0.713` at gain 1.4; the six 75%-bar targets
  are derived from the same projection (`scope_targets()`), never restated
  as constants, so the graticule cannot drift from the cloud.
- **PROBE v1**: a closed append-only vocabulary of retained renderer-owned
  images — `program` (pre-blackout slot 2, default), `program_tap` (the
  honest N-1 image), `gesture_canvas` (the presented etch donor) — plus the
  modulation matrix's live source values
  (`modulation::MONITOR_SOURCE_LIST`, 45 sources, pure reads) as the CPU
  half of the strip. An unavailable probe producer is a named
  `probe_status: "unavailable"` with the instruments holding and the
  readback idle — never a silent rebind onto another producer. The named
  deferred probes (NTSC per-line state, melt band mask, a motion-field
  visualizer) join by appending codes.
- **The sealed permit** (`MonitorBayPermit`): the transform-gizmo seal,
  deliberately not stage_health's weaker file-scope shape — declaration and
  single mint inside a private submodule, so the file's own tests cannot
  forge a token, pinned by the source-audit test. It folds all three
  conditions (bay toggle, `native_controls_visible`, `EditorPreview`), so a
  painter cannot be reached for an audience surface even by a caller that
  got one condition wrong. `show_monitor_bay` joins the one-predicate block
  in `main.rs`.
- **Native paint**: `paint_monitor_bay` consumes the permit by reference and
  draws two egui textures re-uploaded only when a fresh sample lands
  (dirty-flag pre-resolve before the closure, the stage-health law), the
  graticule, the probe line, and the source strip.
- **Wire**: `set_monitor_bay { enabled }` (coalesce `stage:monitor-bay`) and
  `set_monitor_probe { probe }` (coalesce `stage:monitor-probe`, validated
  at the server gate and the applier through the one
  `MonitorProbe::try_from_str` table) — host operations outside manual
  history, like the stage-health HUD toggle; `monitor_watch` never queues.
- **Snapshot**: the additive `monitor_bay` block. Probe token and
  native-overlay toggle always truthful; the base64 instrument payloads
  (~16 KB) exist only while armed; `sample` is a freshness counter so the
  panel redraws at the 10 Hz arrival rate, not the snapshot rate. A pre-B11
  snapshot decodes to the inactive bay.
- **Panel**: the MONITORING BAY group — native-overlay toggle, probe select,
  and the panel's first two `<canvas>` elements (128×64 waveform, 64×64
  scope, pixelated upscale). No new range sliders: the pinned range counts
  (html 198, app.js template tags 24) did not move.

## Closure

Nothing persists in patches: the bay toggle and probe are host-session state
like the media-safety mode and the stage tools, absent from `PatchState`, and
a new process starts disabled on `program`. No modulation address, no Morph
surface, no Dice/generator interaction (`GENERATOR_VERSION` stays "12"), no
export arm — export has no observer, so the bay never runs offline and there
is no labeled export case of its own. The sidecar schema stays 6.

## Proof

- Law tests (hosted): the Rec.601/projection fixtures, targets derived from
  the projection with complementary bars mirrored through the centre, the
  flat-grey field reducing to one saturated waveform row and one centred
  scope point, the two-tone split, hostile short grids drawing nothing, the
  RFC 4648 base64 vectors, the closed probe vocabulary with near-miss
  rejection, snapshot default inactivity and fresh-sample publication, the
  permit refused for every audience surface / a single-monitor audience
  output / a disabled bay, and the source audit pinning exactly one permit
  declaration and one mint inside the seal.
- Protocol tests (hosted): the additive snapshot back-compat strip, wire
  names and coalesce keys, performance-only history classification, the
  watch registry's arm/disarm/disconnect lifecycle, and the live panel
  strings asserted in `app.js`/`index.html` (the gyro precedent).
- `the_monitor_source_list_is_complete_stable_and_pure_to_read` (hosted):
  45 distinct sources, every name round-tripping, two reads identical.
- `gpu_monitor_bay_reduction_matches_the_cpu_reference` (opt-in, RX 6950 XT
  / Vulkan): the GPU reduction against `reduce_analysis_grid` at 128×72
  under the B7 statistical contract (≥95% of bytes within 4 code values,
  mean error < 2), plus the two-slot pool's clean saturation drop.
- Zero-readback-when-hidden: the arming predicate is the single gate on the
  schedule seam; unarmed frames never construct the stage (lazy), never
  encode, and publish the empty block — pinned by the predicate's placement
  at the one seam and the snapshot tests.
- Exactness A/B: `render_feedback_rig_pipeline` rendered on a pinned
  `0de2063` worktree and on this branch, `framemd5` identical over all 33
  frames (`#software` excluded) — the default path moved no pixels. The
  full gate ran green at the final tree state (fmt, both JS checks,
  `cargo check --locked --all-targets`, 1,543 tests / 0 failed under
  `--test-threads=1`, clippy `-D warnings`, rustc 1.98.0), and the opt-in
  GPU fixture was re-run post-fmt.

## Traps hit

(recorded during the tranche)
- `egui::Stroke::new(1.0, …)` trips the new `float_literal_f32_fallback`
  future-incompat warning under rustc 1.98 — pin `1.0_f32`.
- A `&mut self.<field>` pre-resolve taken before a later `&self` method call
  in the same setup block is a borrow error; take the mutable field borrows
  beside the closure's other `&mut` captures, after every method call.
