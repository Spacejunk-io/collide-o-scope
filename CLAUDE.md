# Collide-o-Scope developer notes

Native Rust VDJ instrument. It decodes video or receives live Spout frames,
composites a dynamic layer stack with GPU effects, exposes browser controls, and can
render a saved patch offline.

## Stack

- **winit** — native and fullscreen-output windows, input events
- **wgpu 29** — GPU effects, compositing, temporal passes, readback
- **ffmpeg-next 8** — video decode and media-stream inspection
- **ffmpeg CLI** — thumbnails and final H.264/AAC muxing
- **ntsc-rs** — CPU VHS emulation on bounded workers
- **cpal** — live audio capture
- **midir** — supervised MIDI input/output, clock, typed profiles, and feedback
- **spout2-rs** — Windows Spout sender and receiver
- **axum + tokio** — HTTP/HTTPS panel and WebSocket state/action protocol
- **egui** — native preview/recovery shell and patch parameter editor

The `ffmpeg-next = "8"` crate must match the installed FFmpeg major.

## Module map

```text
src/
├── main.rs              winit loop, app transactions, history, capture, web/native actions
├── pattern_synth.rs     B7 pattern-synth source law: authored params, closed vocabularies, CPU reference, wire-edit table
├── text_page.rs         B7 text-page source law: authored page, bundled two-face raster, wire-edit table
├── performance_track.rs B9 portable take contract: typed control addresses, value laws, checksum, clock
├── composition.rs       stable-ID groups, buses, mattes, and authored composition topology
├── evaluated_composition.rs unified LegacyExact/Advanced planner and resource admission
├── visual_rack.rs       ordered scope racks, typed nodes, taps, validation, persistence
├── image_routing.rs     stable layer/group image routes and missing-target tombstones
├── media_safety.rs      Safe/Expert source planning, device bounds, reservations
├── media_source.rs      shared resolution, bounded SHA-256 fingerprinting, content references
├── spatial.rs           canonical authored transforms and packed GPU pass uniforms
├── transform_gizmo.rs   preview-only direct manipulation of that same transform
├── motion.rs            canonical codec/lattice/procedural fields, Motion authoring, Field Collider, resource preflight
├── symmetry.rs          closed symmetry groups, 32-sector table, 1,024-byte uniform
├── temporal.rs          Loom/Atlas/Garden/Score state, feedback rig, events, resets, commit/discard
├── gesture.rs           portable quantized gesture events, checksum, one normalized adapter
├── gesture_canvas.rs    bounded vector canvas CPU reference, Push/Curl laws, transactions
├── display_physics.rs   B4 display-physics law: fields, phosphor, display model CPU reference
├── mixing_boundary.rs   B8 mixing-boundary law: wipes, blend meet, dirty mixer, melt CPU reference
├── sync_latch.rs        B14 sync-latch law: deterministic shear draws, the bounded per-line table
├── block_dct.rs         B6 Block DCT law: DCT-II CPU reference, quantiser, chroma crush
├── filter_avalanche.rs  B6 Filter Avalanche law: predictors, deterministic lanes, cascade reference
├── pixel_sort.rs        B6 Pixel Sort law: bounded bright-run search CPU reference
├── scan_processor.rs    B1 Scan Processor law: authored params, beam CPU reference, vertex budget
├── renderer/state.rs    LegacyExact passes, audience history, readbacks, output blits
├── renderer/composition.rs shared Advanced GPU executor and transactional histories
├── renderer/display_physics.rs single-seam slot-0 display stage, lazy field/phosphor surfaces
├── renderer/melting_edge.rs B8 slot-0 master melt over the program's own coverage, lazy history
├── renderer/sync_latch.rs B14 slot-0 shear stage, no texture: one bounded table and one uniform
├── renderer/monitor_bay.rs B11 armed-on-demand 128×72 probe reduction and bounded readback pool
├── renderer/pattern_synth.rs lazy per-layer pattern pass executor shared by live and export
├── renderer/corruption.rs B6 corruption-trio executor: four pipelines, shared DCT intermediates
├── renderer/scan_processor.rs dedicated instanced-ribbon executor and shared accumulator
├── renderer/symmetry_field.rs dedicated eight-texture sampler-free Symmetry pass
├── renderer/gesture_canvas.rs ping-pong etch canvas and the presented donor image
├── renderer/stage_map.rs fixed-resource multi-endpoint venue presenter
├── renderer/video_analysis.rs B10 armed-on-demand 32×18 program-image reduction and bounded readback pool
├── renderer/study.rs    fixed-pipeline Study interpreter executor, two textures, no sampler
├── video/decoder.rs     synchronous ffmpeg decode core and RGBA row repacking
├── video/hw_decode.rs   evaluation-only D3D11VA session and the interop probe seam
├── video/threaded.rs    request decoder, codec motion, telemetry, latest-only mailbox
├── layers/mod.rs        video/Spout layer sources, texture upload, frame pacer
├── effects/params.rs    effect and temporal parameters/normalization
├── modulation/mod.rs    stable typed targets, clock/LFO/audio/MIDI routes, curves, slew
├── monitor_bay.rs       B11 preview-only waveform/vectorscope/PROBE law, sealed permit, instrument bitmaps
├── audio/mod.rs         cpal capture, FFT sources, configurable edges/spectrum
├── controller_profile.rs bounded saved-position profiles and atomic persistence
├── midi/mod.rs          supervised typed MIDI events, hotplug, clock, feedback
├── osc.rs               bounded typed OSC ingress/feedback and LAN-safe configuration
├── history.rs           bounded manual gestures and two-phase undo/redo
├── preset.rs            identity-safe scoped presets and atomic library persistence
├── recovery_journal.rs  checksummed append-only PatchState recovery journal
├── morph.rs             A/B snapshots, blend laws, beat glides, persistence
├── ntsc/mod.rs          ntsc-rs parameters/state and worker
├── spout_in.rs          newest-frame-wins live receiver worker
├── spout_out.rs         bounded/drop-new output worker
├── program_recorder.rs  nonblocking CFR video/still publication and sidecar reports
├── stage_map.rs         venue endpoints/slices, monitor bindings, calibration tools
├── stage_health.rs      preview-only timing/resource/source health HUD
├── proxy.rs             measured proxy recommendation, bounded decode/audio input contract, content-addressed cache plan
├── proxy_worker.rs      bounded FFV1/Matroska encode worker, sealed atomic cache, consumption
├── precision.rs         objective color metrics and precision/capability accounting
├── study.rs             closed data-only SSA Study schema and authority validation
├── study_eval.rs        pure CPU reference evaluator: R1 history guard, R2 randomness, rack hue law
├── patch/               YAML model, capture/apply, editor and file dialogs
├── procedural.rs        deterministic v8 typed patch walk, manifests/preflight, capture worker
├── render_export.rs     deterministic shared executor, motion report, optional audio mux
├── web/                 panel server, protocol snapshots/actions, embedded assets
├── input/keyboard.rs    key-to-action mapping
└── shaders/
    ├── fullscreen.wgsl  fullscreen triangle vertex shader
    ├── effects.wgsl     LegacyExact and Advanced layer/master effects
    ├── rack_node.wgsl   Collision Rack nodes and image-tap effects
    ├── study_interpreter.wgsl fixed Study interpreter over a bounded instruction buffer
    ├── display_physics.wgsl field domain, N-1 phosphor store, beam/mask display pass
    ├── pattern_synth.wgsl B7 pattern source: shape/oscillator/wavefolder/comparator/colouriser, no texture
    ├── melting_edge.wgsl  B8 coverage-boundary probe, band drag, self-feeding hold
    ├── sync_latch.wgsl    B14 per-line horizontal shear with the tape wrap, table-driven
    ├── monitor_bay.wgsl   B11 128×72 monitor reduction, 16 bilinear taps per cell
    ├── corruption.wgsl    B6 trio: DCT coefficient/reconstruction stages, pixel sort, avalanche
    ├── scan_processor.wgsl instanced ribbon geometry, vertex-stage fetch, additive accumulate + resolve
    ├── symmetry_field.wgsl dedicated eight-texture group fold, no sampler
    ├── video_analysis.wgsl B10 32×18 content-analysis reduction, 16 bilinear taps per cell
    ├── composition_host.wgsl straight storage; premultiplied A/B/group math; the B8 bus mixer
    ├── motion_*.wgsl    field acquisition, transform shutter, Faraday memory
    ├── motion_collide.wgsl two-pass Field Collider map and recombination
    ├── gesture_etch.wgsl one ordered etch sample per pass plus the donor present
    └── temporal*.wgsl   legacy Temporal plus Loom/Atlas/Garden/Score
```

`src/bin/spout_probe.rs` is the external-process probe for the Spout output.

## Build and run

### Windows

One-time setup:

```powershell
winget install -e --id Gyan.FFmpeg.Shared --version 8.1.2
winget install -e --id LLVM.LLVM
# plus Visual Studio 2022 "Desktop development with C++"
```

Then use the helper, which discovers FFmpeg, LLVM, and vcvars:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1 -Release
```

The MSVC environment is required when `ffmpeg-sys-next` regenerates bindings.
Runtime needs the matching FFmpeg DLL directory on `PATH`.

### macOS/Linux

```sh
brew install ffmpeg
# or install ffmpeg 8 development libraries, clang, and pkg-config
cargo build
cargo run -- videos/some-file.mp4
```

Spout reports unavailable on non-Windows platforms.

## One architectural law

Every expressive input is a `ModSource`. A route transforms the source, scales
it into a target range, and contributes an offset to a **copy** of the base
state for the current frame. Never mutate a base slider value as a side effect
of modulation, and never wire a source directly to one effect.

The live frame flow is conceptually:

```text
nonblocking source receive/decode
→ web-action drain and downbeat-latch release
→ audio/MIDI/input pump and clock
→ matrix update (curve/slew/pad spring) and one immutable frame sample
→ Morph materializes at the sampled beat, including that sample's Morph offset
→ the same sample creates master, transport, and layer-local copies
→ local layer effects
  ├─ VHS off: stack → master → Temporal → opaque resolve
  ├─ VHS on, no contributing bypass: stack → master → Temporal
  │  → opaque resolve → async global VHS → audience replacement
  └─ VHS on, contributing bypass: conditional direct master per slice
     → async VHS on inherited slices only → exact stack recomposite
     → Temporal → opaque resolve
→ blackout as the absolute final audience operation
→ generation-matched Spout/readback consumers and window blits
```

The exporter follows the same effect, morph, modulation, NTSC, and temporal
helpers where possible. Time-dependent offline behavior must derive from
`frame_index`, export FPS, and patch state—not wall time or a live input.

### Spatial transform contract

`SpatialTransform` is the sole authored geometry for master and layer scopes.
Position is normalized composition space; anchor is original-source UV. The
forward order is crop/framing, independent scale, axis-directed shear,
rotation, then position about the anchor. Rotation/shear is conjugated through
output-aspect space so physical angles stay correct on non-square outputs.
Changing anchor alone is visually inert exactly while the authored linear part
is the identity — every Fit mode at scale `[1, 1]` with no rotation and no
skew, because the fit factor cancels. The precise condition is
`forward == diag(fit_size)`, and an anchor step `d` moves the sampled
coordinate by exactly `(Identity - inverse * diag(fit_size)) * d`. Once a
scale, rotation, or shear is authored the anchor is a genuine pivot, and moving
a pivot moves the image; that is what a pivot is, not a defect. This was
previously stated here without its condition, and the repository's only guard
exercised the default transform, where it happens to hold unconditionally.
Sanitize all finite inputs, wrap angles, prevent crop collapse, and fail
singular transforms to transparent.

`SpatialTransform::default()` is Stretch + Transparent + Linear identity. Its
identity is special: `spatial_modes.w == 0` selects the exact historical shader
sample, so old patches stay pixel-compatible. Once a position, scale, crop,
angle, fit, edge, or sampling choice activates the path, Transparent owns newly
exposed canvas unless Clamp was explicitly authored. The host-session new-layer
preference defaults to Fit and applies only to future interactive file/still/
Spout layers, always with Transparent edges; PatchState and exact recall do not
own or consume it. Do not reintroduce an implicit border clamp in Shift, Cellular, or
another UV effect after the spatial path becomes active.

Every effects pass uploads the 224-byte `EffectPassUniforms`, never the legacy
160-byte block by itself. Build it only through `EffectPassUniforms::for_target`:
layers use actual source dimensions and masters use output/output. Both live and
export bind Linear and Nearest samplers and apply the same transform in ordinary,
conditional-master, final-master, and selective-VHS passes. Spatial state is
frame-local evaluated data, not topology; a Morph or modulation change must not
reset/starve an asynchronous selective-VHS plan.

Transforms persist in patches and Apply Look, are optional Morph ownership,
support continuous modulation, and enter Dice only when `include_transform` is
explicitly true. Keep Pattern-only and automatic loop rerolls transform-free.
Discrete Fit/Edge/Sampling choices switch at the Morph midpoint and procedural
generation preserves them while mutating continuous geometry through a separate
deterministic RNG domain. Stable layer IDs are mandatory for remote transform
edits; do not fall back to a stale position.

### The preview transform gizmo

`transform_gizmo.rs` is a **preview-only editing surface over the canonical
`SpatialTransform`**. It introduces no editor-only transform, no second
geometry, no parallel authored field, and no persisted state of any kind. A
gizmo that could author something the browser numeric editor cannot author
would be a defect, not a feature, so the vocabulary is closed and shared: every
drag resolves to absolute values in the exact `param` strings
`set_master_transform` / `set_layer_transform` already carry, and the host
dispatches them as those very actions.

Resource delta: **all zeros on every audience-facing surface.**

| Item | Exact charge |
|---|---:|
| Render passes | 0 |
| Sampled textures | 0 |
| GPU buffers, bind groups, pipelines | 0 |
| Persistent surfaces | 0 |
| `PatchState` fields | 0 |
| Snapshot fields | 0 |
| New wire actions | 0 |
| New modulation addresses | 0 |

The gizmo paints into the editor window's own egui layer. It is not a
composition step, so it charges nothing against the creative ledger and cannot
appear in a ledger reconciliation.

**Delegation, not reimplementation.** `apply_transform_gizmo_edits` sends real
`SetMasterTransform` / `SetLayerTransform` actions through
`handle_web_action_inner_with_feedback`. That single choice buys three separate
laws at once, and each would be a defect if hand-rolled instead:

- it reaches `apply_spatial_transform_edit`, the one authoring function the
  browser numeric editor uses, so a gizmo edit and the identical numeric edit
  are the same call with the same arguments — byte identity is structural;
- it passes through `release_active_morph_for_manual_edit`, so a drag onto
  Morph-owned state transfers ownership exactly as a numeric edit does instead
  of authoring under an engaged A/B pair and snapping back;
- it deliberately does **not** call `handle_web_action`, whose open-gesture
  guard rejects edits while a `NativeManual` gesture is active. That guard keys
  on the *gesture's* origin rather than the action's, so a dispatch through the
  outer entry point would refuse the gizmo's own edits. This is the identical
  seam the browser-gesture arm already takes.

**One predicate for the whole leakage boundary.**
`stage_map::native_controls_visible(output_on_main)` is the single source, and
`show_editor_panel`, `show_native_recovery_strip`, `show_stage_editor_health`,
`show_native_gesture_surface`, and `show_transform_gizmo` all answer from it —
five copies of `!output_on_main` were four too many. `PreviewGizmoPermit`
answers from the same predicate *and* requires
`StageSurface::EditorPreview`, so folding both conditions into the token
matters: a permit that checked only the surface would mint happily on a
single-monitor audience output, because the surface argument at a preview call
site is a constant.

The permit is sealed inside a private submodule rather than merely private to
the file. `stage_health::EditorPreviewPermit` proved the shape, but its barrier
is module-scoped — that file's own `mod tests` could construct one. Nesting
makes the gizmo's tests a *sibling* of the private field, so not even they can
forge a token, and a source audit pins the single declaration and single
construction. The existing permit is deliberately **not** reused: it is
conjoined with `health_hud_enabled()`, and a gizmo must work with the HUD off.

**Coordinate law.** `SpatialGpuUniforms::map_output_to_local` was promoted out
of `#[cfg(test)]` and is now the crate's only inverse; `hit_test_local` is its
fail-closed wrapper. `GizmoFrame` derives its forward map as the algebraic
inverse of that matrix rather than as a second authored transform — two
inverses that agree today would disagree the first time the forward order
changes. Uniforms come from the production `SpatialTransform::gpu_uniforms`
with the same dimension convention `EffectPassUniforms::for_target` uses: a
layer's actual source size, or the output size twice for master.

`PreviewPaneRect` composes the preview's own letterboxing explicitly, and its
mapping is deliberately **unclamped** — a scale or rotate drag that leaves the
image must keep tracking rather than freezing at the border. A singular or
non-finite transform fails hit testing **closed**: `GizmoFrame::new` returns
`None` and there are no handles at all. There is no identity fallback, because
a grabbable handle over a transform that renders nothing is a control that lies
about what it will do.

Hit testing only ever *reads*. Nothing outside `GizmoDrag::update` and
`nudge_edits` mutates a transform, so opening, hovering, and hit-testing leave
`spatial_modes.w == 0` untouched and a patch nobody moved still renders through
the exact historical sample.

**The gizmo owns only its handles.** Translation is a point handle below the
footprint, not the footprint body, and that is a boundary rather than a style
choice: the preview surface is *already* the gesture-etch canvas, and an
untransformed source covers the whole composition, so a body-sized translate
target would have claimed every drag over the image and silently taken the etch
surface away. A pointer that is not within `HANDLE_PICK_RADIUS_UV` of a handle
reports no hit, and the host routes that drag to the etch stroke it always
belonged to. Both share one `egui::Response`: two overlapping interactions on
one rect would let egui's own resolution decide which control an operator
grabbed, so the routing is written down here instead. An open drag keeps the
pointer for its whole life; a new drag claims it only by starting on a handle.

**Drag law.** Everything is captured at `Begin` — scope, handle, the authored
transform, and the derived frame — and every subsequent delta reads that one
immutable snapshot. Answering "where should this be, given where it started"
rather than "how much has it moved since last frame" is what keeps a crop edge
from chasing its own motion and a scale handle from compounding. Scale is the
ratio of lever arms about the anchor and refuses an axis whose arm is zero
rather than dividing; rotation is measured in physical space, so a quarter turn
on a 16:9 output authors 90 degrees. A non-finite computation lands on the
field's documented neutral value, never on a clamped extreme.

**Modifiers and nudges.** Shift is the constraint law — axis lock for translate
and anchor, uniform for scale, `ROTATE_SNAP_DEGREES` (15°) snapping for
rotation, symmetric opposite edge for crop. Alt is the fine law for *every*
handle uniformly, damping the gesture by `FINE_DRAG_FACTOR`; an operator
holding Alt wants a finer version of the gesture they are making, not a
different gesture. Arrow keys nudge position by `NUDGE_STEP` (1/128 output UV),
Shift takes `NUDGE_COARSE_STEP` (1/16), Alt takes `NUDGE_FINE_STEP` (1/1024),
and Alt wins when both are held. A nudge is a complete authored gesture and
takes the ordinary manual-history boundary; it is refused outright while a drag
owns the scope.

**History and cancel.** One pointer drag is exactly one `NativeManual` entry,
routed through the same `GestureHistoryRouter` the etch surface uses rather
than a second open/close law written beside it — every intermediate `Move` is
invisible, so a five-hundred-sample drag costs the bounded stack one
checkpoint. Escape is consumed **only** while a drag is open, so it keeps its
existing meaning everywhere else including quitting. Before the first committed
value it cancels: the captured transform is restored and the gesture is
abandoned. After a value has committed a cancel would be a lie — the program
already moved — so the gesture closes normally and an ordinary undo runs.

**Selection cannot retarget.** The scope is derived from the host's existing
`selected_layer` and converted to a `StableLayerId` at `Begin` and nowhere
else. `bump_layer_stack_revision` is the one barrier every topology edit
crosses — add, remove, reorder, and patch apply all bump it — so it is the
single place an open drag is abandoned. That is what stops a drag begun on one
stack from delivering its remaining delta to whatever now occupies the vacated
position.

**Scope vocabulary.** Master and layer only. A group carries a
`SpatialTransform` in the composition model but has **no interactive authoring
action at all** — no web action, no native editor — so a group handle would
author something no other controller can, which is precisely the defect this
module exists to avoid. Adding one is a wire and protocol change, not a gizmo
change.

Because the authored result is an ordinary `SpatialTransform` edit, patch,
Look, Morph, modulation, Dice, procedural generation, preset, recovery, and
export close over it already. That is the entire point of forbidding an
editor-only transform: S6 proves that closure rather than adding to it.

## Threading and backpressure

- The **render/event thread** owns wgpu rendering and drains input/action
  queues without waiting for decoder, NTSC, or Spout workers.
- Each video layer owns a **request-driven threaded decoder**. The frame pacer
  accumulates fractional media-frame debt, submits only bounded advances, and
  retires only work the decoder accepted. The worker performs every accepted
  advance in order but publishes through a one-frame overwrite mailbox, so the
  renderer takes the newest completion instead of building an image backlog.
  Opening the decoder also publishes a first-frame seed; harvest it before the
  paused-layer early return so a clip opened while paused never displays
  uninitialized texture data. Pause/resume resets transport debt.
- Each Spout layer owns a **receiver worker**. It publishes only the newest
  complete RGBA frame; the render thread resizes the layer texture when sender
  dimensions change.
- The legacy **NTSC worker** accepts at most one in-flight composite. Newer
  work is dropped instead of queued while it is busy.
- The selective **NTSC worker** likewise accepts one coherent contributing-
  layer batch with no pending backlog. Hidden and finite non-positive-opacity
  layers are omitted. Generation, topology, dimensions, sampled transforms,
  and sampled NTSC parameters travel with the pixels; obsolete work is rejected.
  Preflight the complete incremental selective working set with
  `validate_selective_ntsc_live_memory`; its 320 MiB cap includes aligned GPU
  staging, two GPU scratch frames, the tight host batch, and the worker output.
  The renderer's single baseline audience-hold texture belongs to the global
  Pause/blackout contract and is intentionally outside that incremental cap.
  An over-budget or failed selective generation must surface through the VHS
  status and hold the prior exact audience image—never fall back to global VHS.
  While paused, no new batch is submitted and the last materialized audience
  image is held. A dedicated copy preserves it across blackout and path changes;
  blackout stays absolute, then a paused release restores the copy. Spout first
  advances through a non-black epoch barrier and then receives a tagged readback
  of that exact held image, retrying a failed map. A playing selective-path
  transition clears incompatible retained output until the first valid
  replacement arrives.
- The **Spout output worker** owns its thread-affine DX11 sender and uses a
  bounded/drop policy.
- Up to three **GPU staging readbacks** are active. Completed readbacks are
  harvested in submission order so an older composite cannot overwrite a
  newer result.
- The **web server** runs a tokio runtime. Its bounded `WebAction` queue
  coalesces older absolute values for the same semantic control, reserves
  admission for safety/release actions, and broadcasts a complete
  `AppSnapshot`. Do not replace it with an unbounded collection.
- The **proxy encode worker** owns one thread and a one-slot job queue that
  refuses new work while busy instead of queueing a backlog. A request
  resolves its identity mode first: `Verified` carries the layer's retained
  reference, while `Mint` fingerprints the source through the same bounded
  `FingerprintSession` machinery and reports the minted identity with its
  claim (stable layer ID plus source-resource epoch) before any encode; the
  drain re-validates the claim and lands the identity into the layer's
  persistence, and an unlanded identity simply leaves no layer for the
  completion to adopt into. Either way the job then
  re-fingerprints its source, probes and plans through `plan_proxy_input`,
  holds a `MediaSafetyPolicy` reservation for the encode's life, and babysits
  one ffmpeg child with an absolute plan-derived deadline, a staging-size
  kill at the per-artifact cap, bounded captured output, and a caller-owned
  cancel flag (deliberately not the library generation — a proxy is
  content-keyed and survives library changes). Validation decodes the staged
  artifact's identity before publication; publication renames the artifact
  and then its SHA-256 seal, so recovery can remove unsealed residue without
  serving it. Events return to the render thread through a nonblocking
  channel drained once per frame.
- The **proxy adoption worker** owns one thread and a one-slot job queue
  that refuses new work while busy. On an encode completion the render
  thread captures, per matching live layer, a claim — stable layer ID,
  source-resource epoch, and the live playhead — and the worker prepares off
  the render thread: one shared `consult_proxy_cache` (seal re-hash, source
  re-probe/re-plan, decoded-identity validation — exactly the patch-load
  adoption law, never a laxer private path), then one decoder open and one
  playhead-seeded frame per candidate through the same
  `select_seed_frame_at` dance the performance preparer uses. The render
  thread's per-frame drain re-validates every claim against the live layer —
  a stale epoch, vanished ID, changed identity, or already-backed layer is
  discarded with a named reason, never applied to whatever now occupies the
  position — then installs through the infallible `commit_adopted_proxy`
  field swap behind fallible GPU staging. That swap is deliberately not a
  clip switch: slots, transport position/direction/generation state, speed,
  target FPS, pending seeks, pause, the authored filename, and the persisted
  content reference are all untouched, so the audience keeps the exact
  playhead — and a completed OneShot stays transparent — while the decoder
  underneath moves to the artifact.
- **Thumbnail/preview helpers** invoke FFmpeg outside the render path only after
  a metadata probe and `MediaSafetyPolicy` plan. Keep candidates, elapsed time,
  captured stdout/stderr, and concurrency bounded; Safe admits at most four
  thumbnail and two preview helpers, while Expert serializes each class. Every
  helper retains its source reservation and is killed/reaped when its library
  generation becomes stale, without suspending its absolute deadline. Keep
  output within 180×180 and 512 KiB per JPEG, and charge both caches against the
  shared 64 MiB retained-byte budget on insert, replacement, deletion, and
  folder clear. The entire scan pipeline is guarded process-wide; every rescan
  advances the library generation so repeated requests replace rather than
  multiply the documented helper fan-out.
- The native **RECOVERY** strip is a browser-independent control path. Collect
  its absolute actions during egui construction, then dispatch them through the
  same App handler after queued browser actions and before sampling ProgramClock
  for the frame. This gives a deterministic local-last result without inserting
  native actions into browser ingress. Listener lifecycle and browser receiver
  count are separate facts; never infer bind health from receiver count. Keep
  output and media-source status visible in the second recovery row. The strip
  is preview-only and must remain absent from every audience surface.
- `rfd` library-folder selection is modal on the render/event thread. Pause the
  program clock for the dialog, then restore its prior pause state and rebase
  frame, modulation, decoder, and Spout timing so dialog wall time cannot become
  catch-up debt. Cancellation is a no-op. A committed folder switch advances
  `library_generation`, clears basename-keyed thumbnail/preview caches, updates
  the shared upload target, and starts scans under the new generation. Workers
  must recheck that generation while holding the cache-write lock.

## Layer sources and resource bounds

The stack has no fixed layer-count policy. Layer, patch, morph, compositor, and
modulation storage are dynamic, and every current index must remain available
to `target_range` and the panel target list. Preserve the independent source,
decoded-image, GPU-adapter, output-size, route-count, and selective-VHS memory
bounds; they report concrete resource failures instead of silently imposing a
topology ceiling.

Source admission is governed by the process-local `MediaSafetyPolicy`:

- `Safe` is the launch default and preserves the exact legacy per-source limit
  of 8,294,400 pixels / 33,177,600 RGBA bytes (3840×2160 area), plus the 16,384
  px absolute edge and any known device texture-edge/per-buffer limits.
- `Expert` affects future video, still, and Spout allocations only. Its absolute
  source ceiling is DCI-8K area: 35,389,440 pixels / 141,557,760 RGBA bytes,
  still intersected with every edge and device limit above.
- An above-Safe plan reserves a conservative combined CPU/GPU working-set
  weight: 4× RGBA for video or Spout and 6× for a still. Aggregate reservations
  cannot exceed `min(detected_physical_memory / 8, 2 GiB)`. Safe-sized sources
  retain legacy behavior and reserve zero Expert bytes. The reservation must
  live as long as the accepted above-Safe source and release on Drop.
- `wgpu` exposes capability limits but no portable live/free-VRAM budget. Never
  relabel the host-memory reservation as VRAM detection or allocation proof;
  keep actual GPU texture/buffer creation and queue uploads inside recoverable
  Validation/Internal/OutOfMemory scopes. Only mark a source frame initialized
  after those scopes return cleanly, and propagate failures to layer/export
  status.
- Changing back to Safe governs future plans and must not invalidate accepted
  sources. The mode is intentionally absent from patches and starts Safe again
  in a new process.

This override applies to source admission, not program surfaces. Renderer,
fullscreen-output, and export-output dimensions retain their established UHD-
area validation. The 320 MiB incremental selective-VHS budget is an independent
pipeline bound and must not be raised by Expert mode.

Before the initial media file influences preview dimensions, probe only its
metadata under Safe policy. A rejected probe uses 1280×720. If renderer creation
at an otherwise admitted source size fails, drop that window and make one
1280×720 recovery attempt; publish the recovery through `output_error` and the
native strip. Never loop retries or turn Expert source admission into a larger
program-surface policy.

`LayerSource` distinguishes request-driven video, immutable still image, and a
live `SpoutIn`. Every live layer keeps a stable `source_path`:

- video/still: canonical file path when available, otherwise the supplied path;
- Spout: `spout://<sanitized-sender-name>`.

A persisted procedural `LayerConfig` may instead carry path-independent
`cos-sha256://<sha256>/<byte-length>`. The constructed live file layer keeps
that identity separately from the resolved canonical runtime path: capture and
save emit the retained identity, while export uses the runtime path only as a
candidate that must still satisfy the recorded length and digest.

PNG/JPEG/BMP/WebP stills decode once to bounded straight-alpha RGBA, publish
exactly one upload, and have no transport clock. Live and offline rendering
therefore hold the same source pixels while time-authored effects continue.
Keep extension classification, decoded-content format validation, adapter edge
limits, aggregate pixel/RGBA limits, and changed-between-probe-and-decode checks
aligned; an immutable source does not justify unbounded image allocation.

Every live layer also receives a nonzero, immutable process-lifetime
`layer_id`. It survives reorder but is deliberately not serialized: a patch
load constructs a new topology with new live identities. Bundled-client layer
actions must include this ID. When an optional ID is supplied it is
authoritative: reject an unknown/stale value without falling back to the
positional index. Bundled reorder requests must also include and validate the
snapshot's `layer_stack_revision`; reject a stale revision instead of applying
an old index to a different layer. Index-only handling exists for legacy
protocol clients, not for new UI code.

The panel's layer card is a direct editor as well as a routing surface. Keep
target decode FPS and every per-layer effect exposed there, including
`downsample` and the cellular amount/scale/warp/speed/gap controls. Keep
protocol ranges aligned with patch normalization and modulation target ranges.

Patch load prefers this identity and falls back to the legacy library
filename. Missing video and Spout sender errors are surfaced; a live Spout
texture begins as transparent black and is never uninitialized.

Offline export cannot replay an external live Spout sender. It retains that
layer's saved stack index and renders an explicit black placeholder so
layer-numbered modulation routes do not slide onto another source.

## Modulation

The static `TARGETS` table covers continuous master, NTSC, temporal, and morph
values. `target_range` additionally recognizes every positive, parseable
`layerN_…` index for:

- opacity, speed, and target FPS;
- static key threshold/softness, RGB target, and chroma tolerance;
- pixelate, RGB split, hue, saturation, brightness, contrast, posterize;
- grain intensity/size, vignette, color drift;
- breathing scale/rotation/position;
- Shift amount/block size/density/speed;
- spatial position/scale/anchor/rotation/skew/skew-axis/crop;
- downsample;
- cellular amount/scale/warp/speed and gap amount/threshold/softness.

All route consumers share the same shaping/slew state. Curves are Linear,
Exp, Log, SCurve, and Steps; signed shaping preserves bipolar source sign.
Attack and release are independent seconds-based time constants, updated once
per frame. `MAX_ROUTINGS` is 64.

Compile each route destination when the routing changes; never parse
`layerN_*`, format target names, or search the target table at frame rate.
`ModMatrix::frame(layer_count)` performs one O(routes) accumulation into
caller-sized indexed storage. Out-of-stack targets remain dormant and must not
cause allocation proportional to their authored index. Its immutable
`ModulationFrame` is the sole per-frame source for the
Morph offset, master/NTSC/Temporal copies, layer transport, and the batched
layer-effect copies. Morph materializes first at the sampled beat; the remaining
offsets then operate around those materialized bases. Pause freezes clock,
spring, slew, and Morph progress rather than advancing transients. Export must
use this same frame-indexed order and one-sample reuse.

Sources include four clocked LFOs, seven audio characteristics, four MIDI CC
slots, three gyro axes, two pad axes, and the B10 performance sources below.
LFOs, chaos, and drift are bipolar; every other source is normalized to 0…1.

### B10 performance sources

Pure `ModSource` extension through the one matrix law — no new modulation
targets, no morph surface, `LAYER_TARGET_SUFFIXES` untouched, and the laws
derived from BENDR (MIT, © 2026 Steve Blythe) with one house hardening BENDR
never claimed: **deterministic replay**. Every generator is a pure function
of accumulated program seconds, the persisted `generator_seed`, and the
frame inputs, so the same patch replays the same trajectory live and
offline.

- **Bend pads** `bend1..bend6`: momentary sources on the asymmetric ramp
  (24/s toward held, 7/s released). Native digit row 1–6 through the
  dedicated `map_bend_key` mapper (`map_key`'s release-is-inert law is
  pinned and unmoved; releases are honored even when egui consumes the
  event, and focus loss releases every pad); panel pads copy the XY pad's
  pointer-capture/blur/visibility/reconnect machinery; controller profiles
  bind notes or buttons onto the appended `ControlParameter::Bend1..6`
  engine surfaces (the GestureContact precedent). Held state never
  persists; `bend_pad` has no coalesce key, both edges are priority, and it
  is refused in `Quantized` batches at both gates.
- **Envelopes** `env1..env4`: linear attack resuming from the current level
  on retrigger, exponential decay, modes `once | gate | loop`, triggers
  `bend1..6 | audio_onset | scene_cut | beat | beat2 | bar`. Beat triggers
  fire on whole-multiple crossings and anchor without firing on first
  observation. Attack 0.005..10 s, decay 0.02..30 s, neutral non-finite
  fallbacks.
- **Macros** `macro1..macro4`: an authored knob that is a source.
- **Generators**: chaos (hashed hold-interval walk, eased 7/s), drift (the
  fixed three-sine sum), spike (per-reference-tick firing at 1.6 events/s,
  hashed amplitude, 9/s decay — tick-addressed and therefore frame-rate
  invariant). One persisted seed, two hash domains; reseeding restarts the
  trajectories deterministically.
- **Video-reactive** `video_motion` / `video_brightness` / `video_cut`:
  BENDR's content analysis whole — one 32×18 encoded-luma grid; brightness
  is the mean, cut an onset against `max(0.06, 3.5·EMA)` with `exp(-5t)`
  release, motion peak-normalized frame difference; the first grid after a
  gap zeroes motion/cut. Two deliberate deviations from the tranche sketch,
  both toward BENDR's shipped law: motion is frame difference (not a
  Motion-lattice readback) and the grid is 32×18 (not 16×16). Armed on
  demand (`video_analysis_armed`: a consuming route or a scene-cut
  trigger); live, a lazy `VideoAnalysisGpu` reduces the pre-blackout slot-2
  image at the program-tap acceptance seam, 10 Hz on the reference grid,
  through its own two-slot FIFO readback pool (busy pool drops cleanly;
  ledger: one 2,304-byte target + two 4,608-byte buffers, outside the
  full-frame texture floor on the pattern-synth precedent; the three
  full-frame audience readback slots are untouched). Offline, the export
  loop runs the same CPU law (`reduce_video_analysis_grid`, same UVs, same
  linear-light filtering) on the frame bytes it already reads for ffmpeg,
  landing each sample at N-1 — video reactivity is deterministic offline.

Patch closure: `ModConfig.envelopes` / `macros` / `generator_seed`,
skip-serialized at their defaults so pre-B10 patches keep their bytes and
canonical hashes; runtime state resets at apply. Wire: `set_envelope` /
`set_macro` / `set_mod_seed` coalescible and gated by the engine's own
vocabularies; a new seed restarts the generators. Snapshot: additive
`envelopes`/`macros`/`bends`/`generator_seed` on `ModSnapshot`. Panel: the
PERFORM SOURCES group (third-column pin 3 → 4; JS range-template pin
21 → 24, HTML 198 unmoved). Dice and the generator preserve the sections
exactly (`GENERATOR_VERSION` stays "12").
`render_mod_sources_pipeline` is the labeled export case with its
`_unrouted` difference twin and `_repeat` determinism assertion.

### Clock and beat latch

Tap tempo drives the internal beat. MIDI 0xF8 pulses provide 24-PPQN external
clock; 0xFA resets the beat; after a pulse timeout, the internal clock resumes
from the same position.

The web protocol can wrap eligible actions as `{"action":"quantized",…}`.
The app coalesces pending actions by control identity and applies them when the
four-beat bar number advances. Emergency actions and operations that are not
declared quantizable remain immediate.

### Audio

`AudioAnalyzer` captures the selected/default cpal input, builds a 1024-point
FFT, and exposes normalized level, 3–8 configurable bands, onset,
centroid-derived brightness, and spectral flatness/noisiness. N bands use N−1
finite ordered crossover fields plus an analysis ceiling, all patch-persistent.
`audio_band1` through `audio_band8` enter the same matrix as every other source;
legacy `audio_bass`/`audio_mid`/`audio_high` names remain aliases for bands 1–3.
A 32-bin log spectrum is display telemetry, not 32 new route sources.

On Windows, enumerate output endpoints as explicitly prefixed WASAPI system-
playback sources; never infer loopback from an ordinary device name. Looping-
file mode accepts WAV/MP3/FLAC/Ogg/Opus/M4A/AAC, enforces the 512 MiB upload,
10-minute decode, and 60-second decode-time limits, decodes once, and samples a
circular analysis window from program time. The selected clip need not be
audible. The same clip/configuration/timestamp must yield the same matrix values
live and offline, and Pause must hold that timestamp.

Keep requested and runtime audio state distinct. `audio_device` is the saved
preference (empty means system default); `AudioAnalyzer::active_device()` is
the CPAL device backing the current stream; and
`is_using_device_fallback()` identifies the case where a missing named device
opened the system default instead. That fallback still satisfies the original
request and must not trigger a reopen loop. Build/play/runtime/stale-stream
failure is soft: stop the stream, reset all sources to zero, expose the error,
and return the requested enable toggle to off.

### Gyroscope

The browser sends raw DeviceOrientation degrees. The engine stores raw values,
per-axis calibration centers, full-swing ranges, expo, and invert settings.
Yaw wrapping is calibration-relative. The normalized center is 0.5. These
settings and raw positions persist in patches.

iOS sensor permission requires the HTTPS panel. A desktop simulation is not
proof of physical orientation axes or browser permission behavior.

### XY pad

Pointer capture makes each gesture ownership explicit. Curves and step counts
are per axis. When spring is enabled, pointer release begins an engine-side,
dt-based return to center; otherwise the pad holds. Current position and
configuration persist, while patch load marks the pointer released so spring
motion can resume deterministically.

Quantization is a count of positions, not intervals: for N in 2..=64, map to
exactly N evenly spaced values including 0 and 1 (divide by N - 1). Values 0
and 1 disable quantization. Preserve this invariant for every response curve.

## Morph

Morph slots capture continuous master, NTSC, temporal, and per-layer
performance state. Discrete choices switch at the midpoint. Blend law is
linear or equal-power; the latter uses complementary normalized contributions
so equal endpoint values remain equal. Glides use beat durations, can be
retargeted continuously, and serialize as remaining movement rather than an
absolute live-clock origin.

Shift's four continuous values follow the same slot and ownership law. Runtime
pattern seeds remain intentionally outside Morph slots, so a pattern-only
reroll can rearrange Shift without rewriting or interpolating its captured
controls; Bounded variation follows the ordinary owned-base transfer law.

The morph target is itself routable. Before capture, materialize the effective
Morph result at the authoritative beat, including the current Morph-route
offset; all other route offsets remain transient. A capture action is an
ordering/coalescing barrier and carries `layer_stack_revision`. Reject a stale
revision instead of attaching positional slot data to a different stack.

Manual position and blend-law edits materialize immediately even while program
time is paused; glides and automatic clocks do not advance while paused.
Completed glides settle and clear even if Morph has only one slot. Persist the
true remaining glide duration—a remainder below the UI's 0.25-beat minimum must
not be stretched on recall. Interpolate hue and slit orientation along their
shortest wrapped arcs with deterministic 180° ties and exact endpoints.

Adding a layer at the end preserves both slots and leaves the new layer outside
the existing Morph. Removing or moving a layer remaps both slots so surviving
positional identities remain aligned; topology changes also purge queued stale
captures. Live and export sample the same persisted state and apply modulation
around the same materialized bases.

An engaged A/B pair owns the master, NTSC, temporal, and intersecting captured
layer positions. Before a direct manual edit to owned state, materialize the
current beat plus Morph-route offset once, clear both slots, and then apply the
edit. This transfers control without a visual jump or later snap-back. Do this
when a beat-latched action executes, not when it is enqueued. A single slot and
an appended layer absent from the slot intersection remain directly editable
without being cleared.

## Temporal state

Temporal effects use two memories:

1. A 24-layer clean-composite history ring for slit-scan.
2. A separate previous post-temporal frame for recursive feedback.

The ring advances at `TEMPORAL_REFERENCE_FPS` (30 Hz), independent of live or
export FPS. At 30 Hz it spans about 0.8 seconds. Valid-history and
valid-feedback counters prevent sampling unwritten texture content. History is
kept warm while effects are off. Slit direction is derived from a normalized
angle and aspect-aware direction vector; legacy row/column patches map to
0°/90°.

History Key compares the current clean composite with a selected 1–23-sample-
old history entry. Its discrete modes keep motion, stillness, brightening, or
darkening; threshold, softness, and history depth remain continuous modulation
targets. Apply its mask after feedback/slit processing. Off must preserve the
established Temporal path.

Live rendering passes elapsed dt; export passes frame-index-derived dt. Keep
the shader uniform layout tests current when changing temporal uniforms.

### The B3 feedback rig

`FeedbackRigParams` on `TemporalParams` is everything the loop does to the
fed-back sample beyond the frozen zoom/rotate/retention trio: per-tick offset,
the two discrete reflections (the regime no rotation can reach), in-loop hue
rotation / saturation / per-channel gain, chromatic displacement of the
lookup, the blur+sharpen activator–inhibitor pair over fixed two-texel cross
taps, the `FeedbackShape` waveshaper (`Clamp | Soft | Wrap | Fold`, permanent
codes 0–3, `Clamp` at drive 1 the identity) with drive/pivot, a threshold that
decays sub-threshold light out of the loop, deterministic loop noise on the
shared `cellular_avalanche` hash keyed by pixel and 30 Hz reference tick, an
`fb_edge` boundary law on the frozen program-wide
`Transparent = 0, Mirror = 1, Wrap = 2, Hold = 3` numbering (`Transparent` is
the exact historical inside test), and the servo.

**Identity is the exact prior path.** The rig rides its own 96-byte
`TemporalRigGpuUniforms` third fixed binding — the legacy 64-byte uniform and
its byte golden are untouched — and the shader's activity flag answers from
the *authored* identity, so a default patch executes the historical feedback
expression byte for byte in both variants of both temporal shaders. That
exactness is proven three ways: the startup pixel goldens, the M6 receipt's
six output SHAs (unchanged under the re-pinned shader-bundle digest), and a
decoded-frame-identical cross-build A/B of an unrigged labeled export case.

**Rate law.** Rate-like controls (offset, hue, chroma displace, blur, sharpen,
noise) scale linearly per 1/30-second reference tick; multiplicative controls
(saturation, gains) exponentiate, the `feedback`/`fb_zoom` law; the nonlinear
stage and the servo mix toward identity by the clamped tick fraction
(`rig_tick_mix`), exact at the 30 Hz reference. The loop's gains may exceed
unity — that is what the servo is for. The servo is deliberately a
**deterministic per-pixel compressive auto-level**, not a measured-mean loop:
a readback-driven servo would give live and export different dynamics, which
the export contract forbids. `servo_defeated` wins over engage — defeated, the
loop may run to white or black and stay there (B14's philosophy, landed
here). The rig hangs off an active `feedback`; with feedback zero there is no
loop to shape.

**Refresh Garden keeps its frozen carrier law.** In the originals shader the
shared carrier read serves Garden and the rig-inactive feedback; an active rig
takes its own transformed read so Garden's bounded identity/warp law never
changes. The shared-read predicate narrows to `garden || (feedback && !rig)`,
which simplifies to the original at rig identity. The rig's extra reads (one
base + two chromatic + four cross taps, each gated by its own authored
amount) are counted in the shader-contract test, and the frozen
`temporal.wgsl` SHA was re-pinned for the change.

**CPU reference and regime proof.** `temporal::feedback_rig_resolve` and
`feedback_rig_grade` are the law the shaders follow expression for expression.
The regime fixtures run a low-resolution CPU loop: a quarter-turn-per-frame
rotation locks an impulse into four arms carrying the analytic retention
powers, detune shears them off, reflection produces the two-cycle alternating
regime no rotation reaches, and the servo bounds a gain-2 loop that runs away
monotonically when defeated.

**Closure.** Patch: `TemporalRigConfig` is skip-serialized at identity, so
pre-B3 patches keep their bytes and canonical hashes; hostile scalars sanitize
to neutral values and unknown fields are rejected. Wire: eighteen `fb_*`
params on the ordinary coalescible `set_temporal` (both validators plus the
applier agree on ranges and tokens). Snapshot: an additive `rig` block. Morph:
continuous values blend, the in-loop hue on its wrapped arc, and reflections,
shape, edge, and both servo switches recall an endpoint at the midpoint.
Modulation adds fourteen `temporal_fb_*` continuous addresses (inserted
before `morph`, which stays last); the six discrete laws have none. The
generator mutates the fourteen continuous values in fresh per-field domains
(`mutate_temporal_rig`), never the discrete laws; live Dice continues not to
touch legacy temporal at all. Export rides `TemporalConfig::to_params` and
the shared plan; `render_feedback_rig_pipeline` is the labeled export case —
it renders an `_unrigged` twin (decoded frames must differ) and a repeat
(decoded frames must match), so reach and determinism are both measured.

### B12 time-displace maps

Slit-scan's map is the instrument. `TimeDisplaceMap` on `TemporalParams` is a
closed vocabulary with permanent codes 0–4 — `Ramp` (the exact existing angle
path, default), `Brightness` (the current sample's alpha-covered Rec.709 luma:
bright things lag dark ones), `Radial` (aspect-correct distance from centre,
reach 1.6), `TbcRamp` (a sawtooth over each 8-scanline group,
`fract(uv.y · height / 8)` — per-line by design, so different output heights
legitimately band differently), and `Sweep` (a wrapped horizontal ramp
travelling one full crossing per 600 reference ticks — 20 s at 30 Hz, fixed
law, phase from the same accumulated `total_reference_ticks` the rig's noise
epoch uses, so Freeze holds it and export replays it structurally).
`slit_interp: bool` selects linear interpolation between the two adjacent ring
layers; off is the exact banded floor law. The vocabulary is derived from
BENDR (MIT, © 2026 Steve Blythe); every law is a rewrite. The CPU reference is
`temporal::time_displace_coord` plus `time_displace_sweep_phase`, followed by
the shader expression for expression.

**Identity is the frozen legacy shader.** `TemporalParams::time_displace_active`
(slit-scan on ∧ (non-Ramp map ∨ interp)) joins `originals.is_zero()` in the
plan's `originals_shader_active` predicate, so a non-default B12 state runs in
`temporal_originals.wgsl` — the same seam Loom/Atlas/Garden use — and
`temporal.wgsl` was not edited: its pinned SHA did not move and the authored
default keeps executing the frozen legacy shader byte for byte. Within the
originals shader, Ramp+floor routes through `history_age_sample` with layer
arithmetic identical to the removed inline read, so an already-active Loom
patch with default slit state is also pixel-exact. The Advanced host's
pre-Garden originals predicate answers to the slit lanes only while slit-scan
is active, mirroring the plan predicate.

**Ledger: zero new surfaces, zero new uniform bytes, ≤ 1 extra history load
per pixel.** The map code and interp flag ride the two reserved
`loom_geometry` lanes; the sweep phase rides a reserved `atlas_values` lane
and is populated only when Sweep is authored, so a default patch's uniform
bytes never vary with the tick counter. Depth clamps against the
valid-history counter exactly as History Key does — both slit blocks read
history only through the age helpers whose age 0 is the virtual current
image. Ring depth stays 24.

**Closure.** Patch: `slit_map`/`slit_interp` on `TemporalConfig`,
skip-serialized at default so pre-B12 patches keep their bytes and canonical
hashes; an unknown map token is a deserialization rejection. Wire: both fields
ride the ordinary coalescible `set_temporal` (`slit_map` as the closed token
vocabulary `ramp | brightness | radial | tbc_ramp | sweep`, `slit_interp` as a
boolean), validated in both validators plus the applier. Snapshot: additive
`slit_map`/`slit_interp` fields defaulting to the exact prior path. Panel: a
select and a toggle beside the slit controls. Both fields are discrete laws:
Morph recalls an endpoint at the midpoint, no modulatable address exists, and
Dice/the generator continue not to touch legacy temporal. Export rides the
shared plan; `render_time_displace_pipeline` is the labeled export case — it
renders a `_ramp` twin (decoded frames must differ) and a repeat (decoded
frames must match), so reach and the sweep clock's determinism are both
measured.

### B13 small effects

One tranche, one law, fifteen looks. The shared effect uniform grew from ten
to eighteen vec4s (160 → 288 bytes; `EffectPassUniforms` 224 → 352, spatial
slots now at byte 288), and every new control's default takes an exact no-op
shader branch, so a default patch's pixels are byte-identical — proven by the
re-pinned M6 shader-bundle digest with all six output SHAs unchanged. The
laws are derived from BENDR (MIT, © 2026 Steve Blythe); every one is a
rewrite in linear light with Rec.709 luma.

**Both scopes** (layer and master, 28 continuous controls + 1 discrete):
`contour`/`contour_bands`/`contour_width`/`contour_hue`/`contour_fill`
(isolines between smoothed luma bands, screen-space derivative distance,
white lines near hue phase zero), `flatten`/`flatten_levels` +
`contour_dither` (luma quantized to solid fields with an ordered 4×4 Bayer
dither), `solarize` (fold-back exposure), `negative` + `negative_mode`
(permanent codes: 0 rgb `1-c`, 1 luma-only `c + (1-2·luma)`, 2 hue-flip
`2·luma - c`), `colourpass`/`colourpass_hue`/`colourpass_width` (one YIQ hue
window survives, the rest goes mono; hue in degrees on the wrap allowlist),
`edge_amount`/`edge_hue` (Sobel outline over source luma, HSV-coloured),
`emboss`/`emboss_angle` (directional difference lit from one side),
`halftone`/`halftone_pitch`/`halftone_angle` (brightness-sized dots on a
rotatable screen, before the colour adjustments so the dots receive them),
`moire`/`moire_freq` (interference against a virtual grid on the established
effect time), `row_smear` (wrong-predictor row shear, a UV effect on the
Shift precedent), `bitcrush`/`bitcrush_levels`/`bitcrush_dither` (mono
ordered-dither quantize; two levels is the classic 1-bit crush), and
`multi_grid_x`/`multi_grid_y` (the dumb 1–8 × 1–8 tile with odd cells
mirrored so tiles meet; the Symmetry Field's p1 lattice stays the smart one).

**Master-only optics** (3 continuous controls): `barrel` (radial distortion),
`chroma_aberration` (per-primary radial scale), `anamorphic_streak`
(thresholded horizontal flare, blue because the coatings that cause it are;
ten 1/i-weighted taps each way). Master-only is enforced at every layer
authoring seam through `EffectUniforms::clear_master_only_effects` — the
layer wire applier, layer patch/Look application, layer Dice, and the offline
layer builders — so a hostile patch or legacy client cannot install an optic
on a layer copy, and no `layerN_` modulation address exists for the three.

**UV and sampling contract.** Multi grid, barrel, and row smear are UV
effects in the established order (multi grid first, then breathing, cellular,
shift, barrel, row smear, downsample, pixelate); their legacy branches keep
the historical clamp/wrap while an active spatial transform owns every
exposed coordinate. Every neighbour read (contour smoothing 4, find-edge 4,
emboss 2, halftone 1, chroma aberration 2, anamorphic 20) goes through the
one canonical `sample_source` chain — there is no second sampling path — and
each is gated by its own authored amount, so an inactive effect costs zero
extra lookups.

**Closure.** Patch: every field on `EffectsConfig`, skip-serialized at its
default so pre-B13 patches keep their bytes and canonical hashes; hostile
scalars sanitize to neutral values. Wire: all controls ride the ordinary
coalescible `set_param`/`set_layer_effect` (`negative_mode` as an integer,
the grain_algo precedent); the layer applier drops master-only names.
Snapshot: additive fields with prior-path defaults. Panel: two master
fx-groups (SMALL FX, OPTICS) with group resets, and a layer-card SMALL FX
disclosure beside CELLULAR. Modulation: 31 master targets inserted before
`morph` (which stays last) and 28 layer suffixes appended at indices 64–91;
`negative_mode` is a discrete law with no address; the four angle/hue
controls are degree-wrapped in Morph. Morph blends every continuous value and
recalls `negative_mode` at the midpoint. Dice mutates all 31 in a fresh
domain-separated stream (pre-B13 streams proven byte-stable against a pinned
pre-B13 golden) and the generator mutates them in a fresh per-scope domain —
`GENERATOR_VERSION` is now "9" — with layer optics never mutated. Export
rides the same shader; `render_small_effects_pipeline` is the labeled export
case with its `_plain` difference and `_repeat` determinism assertions.

### B4 display physics

Everything the program renders is *watched through something*.
`DisplayPhysicsParams` (`src/display_physics.rs`, the independent CPU
reference in the `gesture.rs` tradition) rides `TemporalParams.display` and
drives one new master stage on the **slot-0 seam between the temporal pass
and the opaque resolve** — the one adjacency live LegacyExact, live
Advanced, the selective-VHS path, and export all share, so a single
implementation (`renderer/display_physics.rs` + `display_physics.wgsl`)
serves all four; this is the `encode_opaque_output` single-seam precedent,
deliberately not the dual-implementation temporal one. The laws are derived
from BENDR (MIT, © 2026 Steve Blythe), rewritten in linear light with
Rec.709 luma. Three sub-blocks, each defaulting to exact-off; a dormant
stage encodes nothing, touches no surface, and slot 0 reaches the resolve
untouched.

- **Fields** (`il_amount` wakes it; mode `Weave | Bob | Blend`, the
  `il_order` dominance fault, twitter, 3:2 judder as dressing): real
  interlace against one retained previous-field surface in the slot format
  (RGBA8, one texture copy per reference tick). Field parity and the 3:2
  film clock (`film_frame = ticks*4/5`, two of five held) advance on the
  stage's own 30 Hz reference accumulator — the exact
  `history_ticks_for_delta` law, owned by the stage because the Exact
  temporal state does not advance on Advanced frames. Weave interleaves,
  Bob fills from the current image's neighbours, Blend ghosts; twitter
  flips high vertical detail per field.
- **Phosphor** (`phosphor` wakes it; `phos_r/g/b` default 0.86/1.0/0.66 —
  the P22 signature: green outlasts red outlasts blue): one accumulator in
  the established feedback shape as an `Rgba16Float` ping/pong parity pair
  (charged honestly at 16 B/px total; explicitly **not** a second history
  ring). The store law is `max(current, trail * k)` over the pre-field
  signal, with `k = clamp(phos_rgb * phosphor, 0, 0.995)` exponentiated by
  the frame's fractional reference ticks (the multiplicative rate law), and
  the display reads the **N-1** trail — decay lives in the store, exactly
  BENDR's accumulator.
- **Display model** (`model != Flat`, `bloom`, `defocus`, or `sag` wakes
  it): the closed vocabulary `Flat | ApertureGrille | SlotMask |
  ShadowMask | LcdStripe | Mono | GreenScreen` (codes 0–6, append-only),
  Lottes-style beam-profile scanlines that widen with brightness, the mask
  families at framebuffer coordinates, the fixed 12-tap gather ring for
  defocus/bloom with halation's faceplate tint, and HV sag measured at the
  picture centre. Scanlines/beam/mask act only under a non-Flat model
  (BENDR's own gate); perspective-free dressing wakes nothing alone.

**Laws.** Frame-local evaluated state like a spatial transform — never
topology. An **active** stage flattens coverage (a screen has no
transparency): it observes covered light and outputs alpha one, so the
downstream opaque resolve becomes the identity on it and the flatten still
happens exactly once. Pause holds (the stage is clocked by the
program-advancing delta only). **Blackout clears the phosphor accumulator
and the held field** inside `clear_composite` — a blacked-out audience must
not retain a glowing wake. Disarming the whole stage invalidates both
memories, so a re-arm never resurrects a stale trail. Surfaces are lazy
(BENDR's own rule: the persistence pair only exists once phosphor is turned
up): a default session charges nothing, the first armed frame allocates
once, and a warmed armed frame allocates nothing. The frozen `temporal.wgsl`
SHA and both temporal uniform goldens are untouched: the stage owns its own
shader, its own 128-byte uniform (compile-time asserted), and its own three
pipelines (field, model, store), all sampler-free `textureLoad` with one
explicit-load covered bilinear for the sag-warped read.

**Ledger.** Field pass ≤ 5 loads/px (armed only); display pass ≤ 15/px
(1 base bilinear + 1 trail + 12-tap ring when armed + sag centre); store
pass 2/px (armed only); ≤ 2 sampled textures per pass, 0 samplers; retained
bytes when armed: 4 B/px field + 16 B/px phosphor pair; 0 when never armed.

**Closure.** Patch: `TemporalConfig.display`, skip-serialized at the
exact-off default so pre-B4 patches keep their bytes and canonical hashes;
hostile scalars sanitize to neutral and unknown fields are rejected. Wire:
twenty `disp_*` params on the ordinary coalescible `set_temporal` (both
validators plus the applier; `disp_il_mode`/`disp_model` as closed tokens,
`disp_il_order` boolean). Snapshot: an additive `display` block on the
temporal snapshot. Panel: a DISPLAY PHYSICS group in the temporal section
(17 sliders — the static range count is 165 — two selects, one toggle).
Modulation: seventeen `display_*` continuous master addresses (the three
discrete laws have none; `morph` stays last). Morph: continuous values
blend; mode, order, and model recall an endpoint at the midpoint. Live Dice
continues not to touch temporal-adjacent state; the generator mutates the
seventeen values in fresh field-isolated domains (`mutate_display_physics`,
`GENERATOR_VERSION` is now "10"). Export rides the same encode on the same
seam with frame-index-derived dt; `render_display_physics_pipeline` is the
labeled export case.

### B8 the mixing boundary

Four pillars, one law module (`src/mixing_boundary.rs`, the independent CPU
reference in the `gesture.rs` tradition), all derived from BENDR (MIT,
© 2026 Steve Blythe) and rewritten for this tree.

**The blend audit.** `BlendMode` is 25 append-only codes: the frozen 0–14
plus `VividLight` 15, `PinLight` 16, `Divide` 17, `WrapAdd` 18, `Xor` 19,
`And` 20, `Hue` 21, `Saturation` 22, `Color` 23, `Luminosity` 24, all in the
one `blend.wgsl` kernel with its CPU twin, serde tokens asserted equal to
`key()`. The HSV component-swap quartet is non-separable; the bitwise pair
operates on the **stored sRGB code bytes** (encode → round → bitwise →
decode) — a truncating linear quantizer flips a bit whenever the CPU and GPU
transfer decodes disagree by one ulp, so the code-byte law is both the
robust one and the faithful one (BENDR XORs what the framebuffer holds).
Existing laws keep their indices, proven by the frozen vector rows 0..=14
staying byte-identical and the re-pinned FNV signature. Rack `NodeBlend` is
a separate vocabulary and did not change. Widening the choice set moved
`GENERATOR_VERSION` to "11" (the selection draw is modulo the count; the
draw count itself is proven unchanged).

**The bus mixer.** `CompositionTree`/`RuntimeComposition` carry one
`BusMixerState` bundle (mix, dirt, melt) beside `bus_crossfade` — values,
never topology; skip-serialized at the exact-legacy default so pre-B8
patches keep their bytes. The bus pass (`fs_bus`) owns the whole law behind
textually explicit default branches, so a default bus is byte-identical
(the M6 receipt's six output SHAs did not move):

- **Wipes**: `WipePattern` codes 0–12 (`Dissolve` is the exact historical
  constant crossfade; then WipeH/WipeV/Diagonal/Box/Circle/SplitH/SplitV/
  BlindsV/BlindsH/Clock/DiagBars/Blocks), only Circle aspect-corrected,
  MULTI tiling (rep 1..=4, origin inside each tile), softness with the
  fader remap that keeps exact endpoints, invert on the field, a border
  rule from the closed eight-colour bench table (`BackColor`, codes 0–7),
  and Blocks hashed on the integer avalanche — never a float hash.
- **The blend meet**: the `BlendMode` family at the A/B crossfade.
  `Normal` is the exact legacy premultiplied lerp as an explicit branch; a
  non-Normal meet blends where both lanes carry coverage, and `AlphaCut`
  is not authorable at the bus (a crossfade has no destination to cut) —
  it sanitizes to Normal.
- **The dirty mixer**: an event clock (`0.5 + rate·15` ticks/s) whose tick
  index is the only state, and four fault laws — knock (timebase shove,
  sheared down the frame, wrapped vertical hop, upstream of everything),
  cut (the crossbar thrown to one input), dropout (line bands through to
  the other side of the crossbar or to dead grey hash), and noise
  (band-limited monochrome spray with colour dropping out toward Rec.709
  luma). Bit-clean between firings; every draw is the Shift
  band/epoch/seed law via `cellular_avalanche` in a fresh per-lane domain,
  keyed by the master `random_seed`, clocked by frame-plan time only —
  Pause holds every fault and export replays them structurally. Faults are
  coverage-honest (they never mint coverage), which is exactly what keeps
  an all-Program LegacyExact composition inert under authored dirt without
  touching the eligibility gate. While dirt is authored, neither lane can
  be culled on the fader position (`layer_effectively_contributes`).
- **The bus melt**: the analytic mix matte probed at four points
  (X aspect-corrected, the normal deliberately not — the shipped
  anisotropy), band = disagreement × 1.25, swirl rotates the normal by
  ±90°, creep selects the outgoing side, the incoming lane drags by
  `en·band·melt·0.055`, and the hold dissolves the stage's own previous
  output back in under the cap law
  (`min(0.94 + max(hold−1,0)·0.11, 0.995)`) with the chroma-runs-further
  second tap mixed through the coherent 601 YIQ round trip (the B3 rig
  matrices — the law reconstructs RGB, so a mixed-standard inverse would
  be wrong). A plain dissolve has no boundary, so nothing happens.

**The melt histories.** Both seats keep **one** retained surface each on
the temporal-feedback single-surface precedent, lazily allocated on the
first armed frame (`melt > ε ∧ hold > ε`), retained after, invalidated —
never freed — on disarm, and advanced by `copy_texture` at most once per
30 Hz reference tick on the stage's own rational accumulator, so live and
export creep at the same rate and Pause holds the trail still. The bus
history is working-format (8 B/px), allocated inside the executor before
the warm-allocation snapshot and staged/committed through the
frame-history transaction; the master history is slot-format RGBA8
(4 B/px, the B4 held-field charge). Melt histories are **program memory,
not display memory**: blackout does not clear them (the temporal-ring
precedent — the audience goes dark, the program keeps its state), unlike
B4's phosphor which models the screen itself. Exactly two melt scopes
exist (the bus meet and the master seam), so the ≤2-armed ledger cap is
structural.

**The master melting edge** (`renderer/melting_edge.rs`,
`melting_edge.wgsl`) is a Recipe-B stage on the slot-0 seam immediately
before the B4 display stage at all three live call sites and both export
sites. Its matte is the composite's own alpha, so static key alpha,
cellular gap, and group mattes melt through one mechanism. It reads
through a filtering sampler (the opaque-resolve precedent at this seam;
every tap is level-0, so nothing needs implicit derivatives), preserves
coverage — the trail legitimately carries alpha and the downstream opaque
resolve still flattens exactly once — and its 48-byte uniform is
compile-time asserted. Params ride `TemporalParams.melt` (`MeltParams`:
melt 0..2, width 0..2, hold 0..1.5, swirl −1..1, chroma, creep).

**Key dressing.** `key_border`, `key_border_color` (the closed
`BackColor` table as an integer, the `grain_algo` precedent), and
`key_shadow` on both static-key scopes, in the new vec4 #19 of the shared
effect block (`EffectUniforms` 288 → 304, `EffectPassUniforms` 352 → 368,
spatial slots at byte 304). The house adaptation: a layer has no composite
underneath it, so the dressing **joins the key signal** (fill + matte)
exactly as a broadcast border generator adds fill to a key — border via
the six-tap asymmetric dilation (four axis + two diagonals, BENDR's own
kernel, radius `0.002 + border·0.02`, X aspect-corrected) and shadow as
one offset matte tap darkened to black at `0.8 × amount`. Neighbour mattes
evaluate the spatially mapped source through the one canonical
`sample_source` chain — a bounded approximation the dressing's own amounts
gate, so an undressed key costs nothing and is byte-identical.

**Closure.** Patch: the `mixer` block on the composition tree,
`TemporalConfig.melt`, and the three `EffectsConfig` dressing fields, all
skip-serialized at their defaults with neutral hostile sanitize and
unknown-field rejection; the mixer transfers with the crossfade under the
same Apply-Look identity gate. Wire: `set_composition_bus_mix
{ param, value }` — coalescible per param, revision-free, quantizable —
parsed by the single shared `BusMixerEdit::parse` table both the server
gate and the applier call, so the accepted and applied vocabularies are
structurally one; six `melt_*` params on `set_temporal` in both validators
plus the applier; the dressing rides `set_param` / `set_layer_param`
beside the existing key vocabulary. Snapshot: additive `mixer`, `melt`,
and dressing fields. Panel: the BUS MIXER group beside the A/B fader, the
MELTING EDGE temporal group, and dressing rows in both KEYING surfaces
(static range pin 190; app.js template pin 19). Morph: continuous values
blend, discrete laws (pattern, invert, rep, border colours, bus blend)
recall an endpoint at the midpoint. Modulation: seventeen
`composition/bus_*` addresses beside `bus_crossfade`, six `melt_*` master
addresses, `key_border`/`key_shadow` at master and layer suffixes 92/93;
every discrete law has no address. Dice: the seventeen bus values in a
fresh domain-separated stream; dressing draws appended to the end of the
B13 small-fx stream so every earlier draw is byte-stable. The generator
preserves the bus mixer exactly (the crossfade precedent), mutates the
master melt in fresh field-isolated domains (`mutate_master_melt`), and
appends the dressing draws per scope. Export rides the same shaders on
every path; `render_bus_mixing_boundary_pipeline` and
`render_melting_edge_and_key_dressing_pipeline` are the labeled export
cases.

### B16 program re-entry

The missing producer was the programme itself. `SavedImageSource::ProgramTap`
(serde tag `program_tap`, plan hash code 8, append-only) joins the closed
image-route vocabulary exactly as `GestureCanvas` did: a master-scope
singleton with no scope, no ID, and no saved position, selectable by any
existing image tap — Displace donor, matte, group matte — with no new node
kind, no new wire action, and no new modulation address. The law is derived
from BENDR (MIT, © 2026 Steve Blythe): any channel may source the finished
programme, and whatever it reads is one frame old, which is what makes the
loop stable rather than an infinite regress.

**What the tap holds.** One retained full-frame `Rgba8UnormSrgb` copy of the
**pre-blackout opaque audience image**: final composite slot 2 after the
opaque resolve — so it includes display physics, the melting edge, temporal,
and synchronous selective VHS — before the blackout clear and before the
asynchronous global-VHS replacement, which lands in slot 2 only after the
copy on both paths, so live and export publish the same image by
construction. The copy is published in its own encoder at the
**frame-acceptance decision** (after the frame encoder is submitted and
committed, the only point at which acceptance is known — the gesture-donor /
ProgramHistory N-1 law), so a routed tap reads the finished programme as of
the previous accepted frame and no same-frame cycle is expressible. Both
timings read the same committed copy: the tap *is* the N-1 image and has no
parity pair, so an N-1 route to it stages nothing.

**The blackout decision, made explicitly.** Blackout suspends publication
instead of clearing: `publish_program_tap` is gated on
`temporal_frame_accepted && !blackout`, so the tap **holds** the last
pre-blackout accepted image while the emergency cut is engaged — program
memory on the temporal-ring/melt precedent, not an audience wake like B4's
phosphor. No frame rendered under the cut can enter a re-entry loop, a
release resumes the loop from the picture the cut interrupted, and blackout
stays absolute because the tap re-enters only through the composite, which
the downstream clear blacks on every cut frame. Export has no blackout, so
the gate's branch is never taken offline rather than differently taken.

**Availability.** Before the first committed frame — process start, and
again after a patch load (`invalidate_program_tap` beside the renderer's
PatchGeneration reset, because a new program must not composite its first
frame through the previous program's image) — a routed tap plans
`Transparent` with the named `ProgramTapUnavailable` diagnostic and never
rebinds to another producer. Live admission is `renderer.program_tap_valid()`
consulted at every in-loop plan construction; export admits unconditionally
(`with_program_tap(true)`, the offline-canvas precedent) because its
job-lifetime surface reads defined transparent at frame zero — pixel
identical to the diagnostic path, proven by arithmetic in the GPU fixture. A
rebuilt renderer is a new tap texture: the executor's `ProgramTapBinding`
(the canvas binding's exact shape) carries a host epoch, so prepared tap
bind groups rebuild instead of keeping a destroyed surface's view. Apply
Look, broad revert, and source cuts deliberately do not invalidate — the
program is continuous and the tap stays honest N-1.

**Ledger.** One persistent full-frame surface, charged by raising the
renderer-owned full-frame texture floor from 29 to 30 (`state.rs`, exact
byte literals re-pinned); zero passes beyond the one `copy_texture_to_texture`
per accepted frame; zero retained tap surfaces on either side of the
composition ledger (`TapBacking::ProgramTap` counts zero and
`validate_actual_surface_ledger` reconciles it, exactly as the canvas);
zero new wire actions, zero snapshot cost beyond the route token, zero
modulation addresses. The copy is unconditional on accepted non-blackout
frames — content must not depend on whether anything currently routes it.

**Closure.** Patch: routes ride the ordinary `SavedImageTap` serde, so the
walker claims no edge for them dormant or woken; no new patch section
exists. Wire: the token rides every existing route action through
`CreativeImageSourceSnapshot::ProgramTap` at both timings. Morph, Dice,
generator, Look, preset: route equality covers the new variant with no new
arm anywhere. Panel: the fixed `program_tap` option beside `one_below` /
`all_below` / `clean_program` / `gesture_canvas` (no new sliders — the
range pins do not move). Export consumes the same plan and publishes at the
same acceptance seam; the `.motion.json` sidecar records the route as
`program_tap`. `render_program_reentry_pipeline` is the labeled export case
(the `_untapped` twin holds the identical node at exact bypass inside the
same Advanced plan family and must decode differently; `_repeat` must decode
identically, proving the whole two-frame feedback chain deterministic).

### B5 codec mosh

Nothing here is a shader imitating a codec: `src/codec_mosh.rs` wires a real
mpeg4 encoder and decoder back to back in-process (`ffmpeg-next` library,
never the CLI) and breaks the bitstream between them, so the artefacts are
the decoder's own. The laws are derived from BENDR (MIT, © 2026 Steve
Blythe), whose codec stage settles every control's semantics; the one
deliberate deviation is that BENDR's `Math.random()` fault clock becomes the
shared deterministic avalanche hash (domain "CMSH", independent lanes per
decision) keyed by the master `random_seed`, the stage's 30 Hz reference
ordinal, and the packet index — because our export contract demands a
replayable fault stream and BENDR disables its stage offline outright.

**Authored state.** `CodecMoshParams` on `TemporalParams.mosh` (the B3-rig
closure pattern): eight continuous controls — `amount` (dry/wet in the
stored sRGB bytes, BENDR's own framebuffer blend), `key_removal`
(per-key-chunk dice, deliberately NOT scaled by `rate`; the first key after
any reset always passes because the decoder needs one whole picture to
damage, and a forced resync key still faces the dice, so `key_removal = 1`
never recovers), `hold` (1–6 extra re-applications of the same delta under
fresh monotonic timestamps), `drop` (starve the decoder; a dropped chunk
skips its own hold/shuffle dice but still enters the ring), `shuffle`
(re-inject a ring chunk at least six chunks stale), `rate` (the event-rate
multiplier for hold/drop/shuffle only), `bitrate_starve`
(`4 Mbps × 0.02^q`, ±25% reconfigure hysteresis, every reconfigure forces a
full re-acquire), `resync` (period `max(2, round((1−r)·300)+2)` encoder-fed
frames; zero never recovers) — plus one discrete law, `recycle` (CLEAN
feeds the encoder the clean image; RECYCLED feeds the stage's own previous
blended output, so every pass builds on the last one's wreckage). The wake
law is `amount` alone at BENDR's own 0.003 deadband, and it is a **true
bypass**: no encoder alive, no readback armed, byte-identical prior path —
never "run the codec and hope it is identity", because the round trip is
lossy even at zero fault pressure.

**The engine.** Encode at BENDR's resolution cap (≤ 640 wide, aspect
preserved, even, ≥ 64 — "the artefact is the codec, not the detail"),
`threads = 1` set before open (the per-host determinism lever), GOP at the
mpeg4 encoder's own 600 ceiling with keyframes forced explicitly per frame
(a volunteered key is just another key chunk facing the removal dice), no
B-frames, decoder opened with `OUTPUT_CORRUPT`/`SHOW_ALL` so concealed
pictures are handed over rather than hidden. Every receive loop carries a
finite budget; the decoder-resurrection policy is transcribed (a fault
rebuilds the decoder and forces a bootstrap key — the picture snaps back
and starts falling apart again; more than six cycles gives the stage up
with a named error; a thirty-good-frame streak forgives to one). The chunk
ring holds ≤ 90 delta chunks AND ≤ 8 MiB, FIFO eviction on either bound;
shuffle fires only past ten entries and never picks the newest six. The
last decoded picture is held across starvation, so a dropped chunk smears
rather than flashing dry.

**Live seat.** The global-VHS worker shape verbatim: one `MoshWorker`
(sync_channel(1) both ways, one in flight, drop-new-while-busy counted as
healthy `skipped`, terminal on failure with the error named in the additive
`AppSnapshot::codec_mosh` block), lazily constructed on the first armed
frame. An armed mosh extends `raw_audience_readback_required` on **every**
path — slot 2 already holds the selective recomposite, and bypass is a VHS
bypass, not a general one, so the mosh treats the finished programme
uniformly like the display stage does. `MoshFrameMetadata` (sampled params,
reference ordinal, seed) travels with the pixels on the readback tag, the
NTSC metadata law. On the global-VHS path the metadata carries the sampled
NTSC params and the worker runs the VHS kernel first **in the same hop** —
one admission, one frame of latency, the exact offline order, and the NTSC
worker is deliberately unfed while the mosh is armed. Results are validated
by generation AND by the stage still being armed; the newest moshed frame
is retained and re-written into slot 2 each frame (`write_composite`),
downstream of the VHS replacement and upstream of blackout, which stays the
absolute final audience operation. The B16 programme tap keeps copying slot
2 before every asynchronous replacement, so a routed tap reads the pre-mosh
image — the same law it already had for VHS. Pause holds the fault stream
still (the ordinal rides the program clock); the documented cost of the
stage is one to two frames of audience latency and the occasional
deliberate re-acquisition.

**Export honesty.** Offline runs the identical engine synchronously per
frame, after global NTSC (codec-after-analog; selective frames arrive
already VHS-treated, so the mosh applies uniformly), with the ordinal from
the same paused-aware reference frame the NTSC phase uses. The engine opens
lazily on the first active frame (modulation can wake a dormant stage
mid-job) and a missing mpeg4 pair or a given-up decoder is an actionable
export error, never a silent bypass. **Repeatability is claimed per host**
(two renders, equal decoded framemd5 — the `_repeat` assertion);
cross-machine bit-identity is explicitly not claimed, and the `.motion.json`
sidecar — schema bumped 5 → **6**, its first bump since the Field Collider —
records the additive `codec_mosh` section (authored recipe, encode
dimensions, `mpeg4/avcodec-<version>` encoder identity) only when an
accepted frame actually ran the round trip. The mutated bitstream bytes
never enter the sidecar.

**Closure.** Patch: `TemporalConfig.mosh`, skip-serialized at the
exact-bypass default so pre-B5 patches keep their bytes and canonical
hashes; hostile scalars sanitize to neutral and unknown fields are
rejected. Wire: nine `mosh_*` params on the ordinary coalescible
`set_temporal` (both validators plus the applier; `mosh_recycle` boolean).
Snapshot: an additive `mosh` block on the temporal snapshot plus the
`codec_mosh` diagnostics block. Panel: a CODEC MOSH group in the temporal
section (8 sliders — the static range count is 198 — and one toggle).
Modulation: eight `mosh_*` continuous master addresses (`morph` stays
last); the recycle law has none. Morph: continuous values blend; recycle
recalls an endpoint at the midpoint. Live Dice continues not to touch
temporal-adjacent state; the generator mutates the eight values in fresh
field-isolated domains (`mutate_codec_mosh`, `GENERATOR_VERSION` is now
"12"). `render_codec_mosh_pipeline` is the labeled export case with its
`_clean` difference and `_repeat` per-host determinism assertions.

### B7 generator sources

The first sources with **perfect offline reconstruction**: no file identity,
no content reference, no black placeholder — the patch carries everything.
Two new `LayerSource` arms on the `spout://` sentinel stability rules
(`synth://pattern`, `text://page`, fixed singletons: the kind is the
identity, everything else is values). Laws derived from BENDR (MIT, © 2026
Steve Blythe); `src/pattern_synth.rs` and `src/text_page.rs` are the
independent CPU references in the `gesture.rs` tradition.

**Pattern synth.** One GPU pass per frame (`pattern_synth.wgsl` — no
texture, no sampler, one 128-byte compile-time-asserted uniform) computes a
fixed 1920×1080 page: framing (centre/zoom/rotate/skew/domain-warp) → shape
(closed vocabulary, codes 0–11: Scan, Radial, Spiral, Plasma, Lissajous,
Rings, Starburst, Grid, Tunnel, Cells, Interference, Polygon) → oscillator
(codes 0–5: Sine, Triangle, Saw, Square, Pulse, S&H) → cross-modulation →
wavefolder → comparator → colouriser (codes 0–4: Mono, RgbPhase, HsvSweep,
Duotone, Bands; the default is BENDR's own Scan/Sine/RgbPhase). The whole
path is stateless — a pure function of the 22 authored values and frame-plan
time (`t = time × rate`, BENDR's GPU-synth law; SPD is inert exactly as on a
still) — so Pause holds the picture and export replays it structurally. The
page is fixed-size deliberately: an output-sized page would make an export
at another resolution a different picture. The computed colour is
display-domain; the shader decodes it through the exact piecewise sRGB
transfer so the stored bytes are the picture. The pass encodes into the
frame encoder immediately after its creation — before Advanced prepare and
LegacyExact rendering — on both live and export, from the plan's modulated
copies (`EvaluatedLayer.pattern`, resolved by `modulate_layer_pattern` from
the same frame-local offsets accumulator every per-layer consumer reads), so
live/export parity is structural. The executor
(`renderer/pattern_synth.rs`) is lazy: a session with no pattern layer
charges nothing. The layer texture takes the ordinary still-image
media-safety plan (Safe-sized, zero Expert bytes) plus `RENDER_ATTACHMENT`;
the renderer-owned full-frame texture floor (30) is untouched.

**Text page.** A static typeset page rastered on the CPU (1920×1080 opaque
RGBA) from its own authored state: body (≤ 4,096 bytes, truncated on a char
boundary), one of **two bundled licensed faces** (Hack MIT / Ubuntu-Light
UFL via `epaint_default_fonts` — already in the dependency tree, zero new
embedded bytes, so the same page rasters byte-identically on every host),
size/track/x/y/rotate/repeat/outline, ink/bg colours, and the shape fan
(closed vocabulary codes 0–9) with BENDR's `1 − f·0.55` taper. Re-rastered
**only on authored change** — between edits it costs what a still costs, and
the upload rides the still-image publish law verbatim (pending frame →
ready-frame pump → checked upload, restore-on-failure). The deliberate
deviation: BENDR's clocked terms (scroll, spin, pulse) are absent — the
page's law is re-render-on-change, and movement is authored downstream
through the spatial transform, effects, and Motion. Outline is a bounded
morphological band (radius ≤ 10 px); rotation is a per-row rotate-blit
about each row's own anchor.

**Closure.** Patch: `LayerConfig.pattern` / `.text_page`, skip-serialized at
`None` (pre-B7 bytes and canonical hashes keep); hostile scalars sanitize to
neutral; unknown tokens/fields are rejections. Resolution:
`resolve_visual_source` short-circuits both sentinels before any filesystem
work, like Spout; patch apply and export reconstruction build the layer from
the config alone. Wire: `add_pattern_layer` / `add_text_layer` (topology,
immediate) and the coalescible `set_layer_pattern` / `set_layer_text`, both
validated at the gate and applied by the engine through the **single shared
parse tables** (`PatternSynthEdit::parse` / `TextPageEdit::parse`, the B8
`BusMixerEdit` law). Modulation: 22 `pattern_*` layer suffixes at compiled
indices 94–115 (`LAYER_TARGET_SUFFIXES` is 116); the three vocabularies and
every text-page field have no address — modulating a page would force a
re-raster per frame, which the still-cost law forbids. Morph: pattern values
blend only when both slots captured a pattern source at that position (two
kinds are two pieces, not two ends of a blend), hue on its shortest wrapped
unit arc, discrete laws at the midpoint, kind-gated application, ownership
released on manual edit; the text page is deliberately outside Morph (a body
is content identity, not performance state). Look: pattern values transfer
kind-gated (the matte match-then-move precedent); the text page does not.
Dice and the generator preserve generator source state exactly (the
source-identity law) — **no `GENERATOR_VERSION` bump**; generated manifests
record `kind: pattern_synth|text_page`, `offline_policy: reconstructed`,
verified with no bytes, never tripping `--allow-black-sources`. Prepared
sources: staging a sentinel into a clip slot is a typed refusal. Snapshot:
additive `pattern` / `text_page` layer blocks; `offline_export_policy` is
empty for generators. Panel: generated card sections (range pins:
`index.html` stays 198, `app.js` template tags 19 → 21). Export provenance:
`ExportMotionSourceKind` values `pattern_synth` / `text_page` — additive
values in an existing sidecar field, no schema bump (stays 6).
`render_pattern_synth_pipeline` and `render_text_page_pipeline` are the
labeled export cases; each patch's only layer is a generator, which is the
self-containment proof.

## Effects and compositing

- One combined uniform-driven effect shader avoids pipeline switches.
- The disabled cellular branch must skip its fixed 3×3 Worley search. Feature
  points remain bounded within cells, use integer hashing and smooth target
  interpolation, and keep live/export effect time anchored to patch generation.
- Shift partitions output-space Y into 2–256 px bands, gates them by a hash of
  band, discrete program-time epoch, and `random_seed`, and wraps horizontal
  displacement at no more than 25% of width. Preserve the explicit amount-zero
  branch as the exact established sampling path. Its four controls are finite
  and clamped, patch/Morph/modulation/Dice aware at master and layer scope, and
  must use the same shader and frame-indexed time in export. Domain-separate
  Shift's bounded-variation RNG so adding it does not perturb older Dice streams.
- Spatial sampling must remain one pass with direct effects: output UV effects
  feed the canonical inverse transform, then the selected edge law and crop map
  to source UV. The inactive legacy branch keeps its historical clamp/wrap
  behavior; an active transform delegates exposed coordinates exclusively to
  Transparent/Clamp/Repeat/Mirror. Keep CPU/WGSL layout assertions and Naga
  validation current when changing the four appended spatial vec4 slots.
- A fullscreen triangle computes UVs from `vertex_index`.
- Static keying at layer and program scope supports Off, Keep Bright, Keep
  Dark, Remove Chroma, and Keep Chroma. Luminance modes use threshold/softness;
  chroma modes use RGB target/tolerance/softness. The shader produces straight
  RGB plus modified alpha; the compositor handles opacity/premultiplication.
  Do not multiply keyed RGB by alpha a second time. Layer keys reveal the stack
  beneath; the final program key resolves removed pixels over black.
- Cellular gap key uses the same straight-alpha contract. Preserve coverage
  through layer, master, and temporal passes, then flatten once in linear light
  for the opaque audience image consumed by preview, output, Spout, and export.
- The master Cellular panel owns the same gap amount, threshold, and softness
  values as each layer. A conventional post-stack master gap resolves over
  black. In the selective path, direct master Cellular runs only on inherited
  slices, so its straight-alpha gaps can reveal lower or bypassed content before
  the exact stack recomposite.
- Selective VHS activates only when VHS is enabled and at least one visible,
  finite positive-opacity bypass layer contributes. Plan bottom-to-top slices
  after local FX and conditional direct master FX. Apply VHS only to inherited
  slices and leave bypass slices byte-exact, then mirror `composite.wgsl` in straight-alpha,
  sRGB-to-linear blend math with the established per-pass quantization. Upload
  that pre-Temporal composite; Temporal and opaque resolve remain global.
  Otherwise preserve the established global post-composite VHS path exactly.
- egui-wgpu expects a gamma/raw sampled texture. Register the non-sRGB twin
  view of the opaque audience texture; registering its sRGB view causes a
  second transfer decode and a visibly dark preview.
- Hidden layers clear their intermediate contribution so old texture contents
  cannot leak into the stack.
- The chain operates with sRGB decode, linear-light math, and sRGB encode.
- Blackout occurs before audience-facing readback/blit consumers.

### Advanced execution order

An Advanced composition is scheduled by two topological sorts in
`evaluated_composition.rs`. `execution_order` orders scopes; then
`atomic_group_execution_order` collapses each group and its members into one
task and re-sorts, and **its** output is the plan's `execution_order`. Both
sorts break ties between equally-ready scopes by the **composite rank** —
`BelowTopology::composite_rank`, the `(root_index, member_index)` pair
`below_topology` already records — never by `VisualScopeId` ordering. The scope
id remains the final tiebreak, so the sort stays total and deterministic.

The rank is not cosmetic. `build_block_schedules` in `renderer/composition.rs`
drains the composition's back-to-front stack as each scope renders, and only the
first drain after a task may read that task's own output through
`ScheduledSource::Ping`; every later one needs a `RetainedTap`. A composition
whose layers own **no image tap** has no edges between siblings at all, so the
tie-break alone decides the order — and before the rank landed, an ascending-id
fallback disagreed with the composite order whenever ids ascended front-to-back.
That is exactly what an export job produces, numbering layers `position + 1`, so
a tapless Advanced composition could not be prepared offline at all: a plain
two-layer Faraday transplant failed with *"executed before structural admission
without a current retained tap."*

Both sorts must keep the same rank. A group is ranked at its own root slot, so a
collapsed group task sorts into exactly the slot its members occupy and the two
sorts cannot disagree about where the block belongs. Ranking only one of them
leaves the defect intact, because the second sort's output is the one that
reaches the renderer.

Changing the tie-break changes only which of several valid topological orders is
chosen among genuinely independent scopes, and therefore only whether a drain
reads `Ping` or an equivalent `RetainedTap`. It must not move a pixel: the six
labeled export cases are decoded-frame identical across the change, verified by
paired `framemd5`.

### Named two-input Displace

`Displace` is a Collision Rack node that warps its carrier by a vector field
read from a second stable image tap. `NodeKindTag::Displace` holds append-only
signature code 10; kind codes are never renumbered or reused.

Authored state is `DisplaceParams { tap, amount_x, amount_y, boundary }`.
Amounts are independent finite UV gains clamped to `[-1, 1]`; non-finite input
takes the neutral `0.0` fallback rather than a clamped extreme.
`DisplaceBoundary` is `Transparent | Mirror | Wrap | Hold` with shader codes
0…3 and `Transparent` as the default — the only law that removes coverage.

The donor decode is alpha-covered:

```text
vector = (premultiplied_rg - 0.5 * alpha) * 2
```

Neutral donor encoding is `RG = 0.5` at full coverage. Because the decode reads
*premultiplied* RG against the filtered alpha, a transparent donor, a partially
covered neutral donor, and a missing binding all yield exact zero, so hostile
hidden RGB at alpha zero can never reach the vector field.

`DisplaceParams::is_exact_bypass()` is true when both gains sanitize to zero.
That is a real delegation, not a cosmetic one: the planner collects no tap, the
executor encodes no pass, and the saved-patch dependency walk claims no edge.
Admission is therefore identical in `collect_rack_taps`, `flush_segment`, and
`collect_rack_dependencies` — enabled ∧ wet > 0 ∧ at least one nonzero amount.

Resource delta per active node, charged through the existing descriptor ledger:

| Item | Exact charge |
|---|---:|
| Render passes | 1 |
| Logical lookups/pixel | 3 |
| Explicit texture operations/pixel | 12 |
| Simultaneously sampled textures | 2 |
| Cross-scope image taps | 1 |
| New persistent surfaces | 0 |

The three logical lookups are the dry carrier, the displaced carrier, and the
donor, each a four-load premultiplied bilinear. Displace reuses the established
carrier/donor/sampler bind layout and the rack-owned 1×1 zero texture; it adds
no pipeline, no bind-group layout, and no full-frame surface.

Route and boundary are stable authored topology. Morph interpolates the two
gains only when both slots name the exact same tap and switches the boundary
discretely at the midpoint; Dice and procedural generation mutate only the
gains; modulation exposes only `amount_x` and `amount_y`. The browser edits the
gains and the boundary through the ordinary coalescible parameter action, while
the donor — the only field that rewrites the image dependency graph — uses the
ordered, revision-protected `SetVisualNodeDisplaceRoute { scope, node_id,
route, composition_revision }`. Snapshot params are
`{ donor_tap, amount_x, amount_y, boundary, diagnostic }`. Export consumes the
same evaluated plan and the same rack shader; there is no export-only
displacement path.

### The dedicated Symmetry Field

`Symmetry` folds its carrier through a finite symmetry group and recolours the
result from a frozen 32-record sector table. It is the first node that is not
encoded inside an ordinary rack segment: `NodeKindTag::occupies_dedicated_pass`
is true only for it, and `flush_segment` closes the accumulated segment, emits
one `EvaluatedScopeStep::SymmetryField` at the exact authored position, then
resumes segmentation behind it. `NodeKindTag::Symmetry` holds append-only
signature code 11; kind codes are never renumbered or reused.

**Two closed vocabularies.** `SymmetryMode` has permanent codes 0…7: cyclic
`Cn` 0, dihedral `Dn` 1, planar `p1` 2, `pm` 3, `p2` 4, `pmm` 5, bounded log
spiral 6, orbit 7. Every mode is a genuine finite group — the composition table
closes, the generator returns to the identity after the full point-group order,
the action agrees with the table, and classification recovers each sample
through its own element. The log spiral closes only on its log-polar **quotient
torus**: `folds` steps climb exactly one log-radius period, the period is hard
clamped to 6.0 with the per-step climb derived back out of the clamped value so
closure stays exact, and below a 0.25 period the mode degenerates to pure cyclic
geometry instead of collapsing every radius onto one circle. Orbit is a
presentation of `Cn`, not a distinct group: its satellite radius and spin are a
uniform post-fold frame applied deliberately outside the group action.
`SymmetryBoundary` is `Transparent | Mirror | Wrap | Hold | CellularReentry`
with shader codes 0…4. Codes 0…3 stay byte-compatible with the Displace
boundary vocabulary; `Transparent` is the default and the only law that removes
coverage. CellularReentry is **one deterministic D4 cell transform, never
recursive sampling**: floor to a cell index, select one of eight D4 elements
from two cell parity bits plus one supercell parity bit, apply it about the cell
centre. Non-recursion is proven structurally — no self-call, no loop, no
iterator in any of its three functions — plus by totality and cell-index
periodicity, not by comment.

**The four phase semantics.** Radial phase rotates the sector **origin** and
carries the folded coordinate with it. Orbit phase rotates sector
**classification** only and never moves the folded coordinate; it is named for
the orbit of the group action and applies in every mode, not only
`SymmetryMode::Orbit`. Planar axis rotates the **lattice basis**. Planar phase
translates the **primary lattice coordinate** by whole cell periods. Reuse
`spatial.rs`'s `output_aspect_basis` / `conjugate_through_output_aspect` and its
wrap/mirror helpers rather than re-deriving them: physical angles must stay
correct on non-square outputs. `mirrored_walls()` is a per-axis `[bool; 2]`
declaration — `pm` is `[true, false]`, continuous across its mirror and seamed
across the other — and the identity-seam continuity law is measured against it
rather than against a single per-mode boolean.

`effective_folds()` is the **sole rounding point**. It sums the already-modulated
`base_folds + fold_offset`, guards the sum for overflow, rounds exactly once,
then clamps `1..=32`. Nothing else rounds a fold count, and a source-text audit
asserts `.round()` appears exactly once in `symmetry.rs`.

**The 32-sector table.** `SYMMETRY_SECTOR_RECORDS = 32`, and history ages run
`0..=SYMMETRY_MAX_HISTORY_AGE`, derived from the committed ring as
`TEMPORAL_HISTORY_LEN - 1 = 23` rather than restated as a literal. Each record
chooses a source (`Carrier | Donor0 | Donor1 | CleanHistory`, codes 0…3), an
optional motion slot (`0 | 1 | none`, stored as slot+1 so neutral is also the
zero word), one history age, and one hue offset. Four independently keyed lanes
are drawn by a pure SplitMix64 counter hash over `(stable node domain, authored
seed, sector index, lane-domain constant)` — no sequential state, so any record
recomputes alone and one lane can never perturb another. **Runtime donor
availability must never enter that hash.** Losing a selected donor binds the
neutral view for that sector and rerolls nothing; the fold clamp guarantees a
sector index is always a legal record index at any fold count. An exact-default
node short circuits to the frozen neutral table and therefore does not depend on
its seed at all.

The domain comes from the single derivation `SymmetryNodeDomain::for_scope`, and
**only authored identity may enter it**. The node's persisted `stable_id` and
its authored `seed` do; a `GroupId` does, because it is serialized in the
composition. A live `StableLayerId` does **not**: it is process lifetime and
deliberately never serialized, an export job numbers layers `position + 1`, and
a fresh process or a replaced clip mints a different value for the same authored
layer, so consuming it would reroll all 32 records on every reload and make an
exported file disagree with the program it was rendered from. The layer arm of
`symmetry_scope_owner` therefore contributes only its scope kind. The stated
consequence is that two Symmetry Fields carrying the same node id in two
different layer racks share a table until they are given different seeds — a
bounded correlation between two authored nodes, which is not comparable to a
table that changes under the operator's feet. Reordering the stack never touches
the table, and moving a node to master or into a group is a different authored
identity that legitimately owns a different one.

**Slot index is route identity.** Exactly two fixed image slots and two fixed
motion slots, never a variable count — a variable count would make an authored
route depend on how many other routes happen to exist. Selected donors capture
saved positions and resolve once; `Missing` donors retain their saved positions
and never rebind. Admission is answered **per slot**: a donor whose source-mask
bit is clear can never be chosen by any sector record, so it claims no
dependency edge, no tombstone diagnostic and no binding, and clearing slot 0
never slides slot 1's route down. `ImageTapConsumer::RackNode` therefore carries
`slot: u8`, which enters both the consumer key and the topology signature;
without it two taps from one node compare equal and the first-match binding
lookup aliases them. Motion slots request their primitive vector/gate fields
through the established `required_as_donor` flag, so an armed donor yields a
field even when its own Motion is exactly zero; there is no second path.

**Exact default and bypass.** Cyclic, fold 1, carrier-only, no motion, no
history, no hue, neutral phase/axis/centre, OneBelow/current-frame routes.
`is_exact_bypass()` ANDs geometric identity with table neutrality (carrier-only
mask, empty motion mask, zero hue span): an identity fold whose table can still
read a donor or the ring is emphatically not a bypass. Only cyclic geometry can
claim a bypass at all — over-claiming is a pixel bug, under-claiming only costs
a pass. The delegation is real but sits one level up: because a dedicated step
writes a surface that is *not* its carrier, omitting the pass would leave that
target holding stale content, so `encode_at` always encodes exactly one pass and
`SymmetryFieldExecutor::is_inert` makes the composition skip the step and the
copy entirely. The shader's identity and wet-zero branches `textureLoad` the
carrier texel directly, textually before any filter, so a default readback is
byte-for-byte its carrier.

**Eight sampled textures in one pass — and why the ordinary ceiling did not
move.** `MAX_SAMPLED_TEXTURES_PER_PASS = 3` governs ordinary rack segments and
is unchanged; the two hardcoded LegacyExact matte `3`s (`renderer/compositor.rs`,
`evaluated_frame.rs`) are independent of both ceilings and did not move either.
A **separate** `MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS = 8` sits beside it,
`RackResourceBudget` carries a second accumulator so a dedicated kind never
inflates the fixed rack layout's ceiling, and the dedicated step is checked
against `limits.max_sampled_textures_per_shader_stage` **raw** — the ordinary
path's `.min(MAX_SAMPLED_TEXTURES_PER_PASS)` clamp must never be applied there
and that check must never relax the clamp. Eight bindings: carrier, donor 0,
donor 1, the clean-history **D2 array**, and a vector/gate pair per motion slot.
There is **no texture sampler**: every lookup is an explicit `textureLoad` and
the covered premultiplied bilinear is transplanted from
`rack_node.wgsl`'s `source_premultiplied_linear`. Every planner unit test builds
`CreativeResourceLimits::default()`, whose `max_sampled_textures_per_shader_stage`
*is* the ordinary constant 3, so a Symmetry fixture must raise
`input.resource_limits` explicitly; the admission fixtures refuse at 3 and admit
at the enforced device floor of 16, and that discrimination is exactly what
fails if the clamp ever creeps back in.

Uniforms are the exact 1,024-byte `SymmetryGpuUniforms` with a compile-time
`const _: () = assert!(size_of::<SymmetryGpuUniforms>() == 1_024)`: `meta` 64 B
at offset 0, `params` 64 B at 64, `motion_rows` 128 B at 128, the 32 sector
records at 256 (512 B), the renderer-owned `frame` and `frame_modes` lanes at
768 and 784, then a 224-byte reserved tail. The tail exists so renderer-owned
fields can be added without moving the stride or any existing offset. The arena
strides by `align_up(1_024, min_uniform_buffer_offset_alignment)` with
`has_dynamic_offset: true` and `min_binding_size` 1,024; 1,024 is never written
as a stride literal.

**Three bind groups, not two — a deliberate, documented deviation.** The frozen
contract said "Bind groups: 2". Honouring it would have left the motion pair
permanently neutral, because a `MotionGpuField` owns a **third** committed
ping/pong parity (`MotionMemoryStage::render_field_index`, chosen per field per
frame) above the carrier parity and the composition's N-1 tap parity. All three
in one input group multiply: 4 × 4 = 16 prebuilt groups per node. Splitting the
vector/gate pair into its own group makes them add instead:

- **group 0, image** — carrier, donor 0, donor 1, the clean-history D2 array;
  prebuilt per carrier parity (`std::array::from_fn(|parity| …)`), and the
  composition prepares both committed N-1 read parities above that, so **4 image
  groups per node** in a live frame, exactly as a rack segment does;
- **group 1, uniform** — the 1,024-byte dynamic-offset record;
- **group 2, motion** — the two slots' vector/gate pairs, prebuilt for every
  `[slot 0 parity][slot 1 parity]` combination, so **4 motion groups per node**.
  The two slots are two independent fields and are deliberately **not** required
  to share a parity. The motion group holds no image view, so it is prepared
  **once per node**, not once per N-1 read parity — that is what keeps the count
  at four rather than eight.

**8 prebuilt groups per node, 3 bound per pass.** Do not undercount either. The
sampled-texture count, the pass count and the per-pixel operation ledger are
unchanged: a fragment stage's sampled-texture budget is counted across every
bound group, so the split moved bindings between groups without touching the
eight-texture claim.

`MotionGpuResources::field_primitive_views` hands over *both* parities of a
field's vector/gate pair at prepare time, and `field_read_parity` returns the
committed index — the same `render_field_index` motion rendering wrote through —
at encode. An admitted slot whose field the resources do not own is a
renderer/planner disagreement and is refused by name; an *unadmitted* slot, and a
slot whose committed parity is not materialized yet, bind the defined-zero
neutral pair and close the record's validity lane, which decodes to exactly zero
displacement. Live and offline share one `CompositionGpuExecutor`, so there is no
export-only motion path.

**Reuse the committed Compat8 ring.** The clean-history binding is
`CompositionHost::temporal_history_view()`, the existing D2Array **read** view of
the committed 24-layer ring, with the cursor taken through
`temporal_history_read_cursor()` → `temporal::temporal_read_snapshot`, never raw
`TemporalState` fields, so an age names the same layer here and in the temporal
pass. The 24 single-layer views stay unexposed: a second writer would corrupt
the ring the temporal pass is mid-frame reading. Guard every age against the
valid count exactly as `temporal_originals.wgsl` does. **A new RGBA16F
full-frame history ring is prohibited** (398.1 MB — 379.7 MiB — at 1080p,
`1920 × 1080 × 8 × 24` bytes) absent an explicit
product decision. An unbound source — a lost donor, an unmaterialized age, a
missing binding — falls back to the **carrier**, not to transparent, so a
missing donor changes nothing and the cost stays donor-state independent.

Resource delta per active node, charged through the existing descriptor ledger:

| Item | Exact charge |
|---|---:|
| Render passes | 1 |
| Logical lookups/pixel | 4 |
| Explicit texture operations/pixel | 10 |
| Simultaneously sampled textures | 8 |
| Bind groups bound per pass | 3 (image, uniform, motion) |
| Prebuilt bind groups per node | 8 (4 image + 4 motion) |
| Cross-scope image taps | 2 |
| Uniform bytes | 1,024 |
| Neutral textures / views | 3 / 4 |
| New full-frame persistent surfaces | 0 |

Three bound and eight prebuilt is the whole reason for the deviation: giving the
motion pair its own group makes the three parity dimensions — carrier, N-1 tap,
and each `MotionGpuField`'s own committed ping/pong index — **add** as
4 image + 4 motion instead of **multiplying** as 4 × 4 = 16 groups per node, and
a fragment stage's sampled-texture budget is counted across every bound group,
so nothing else in this table moved.

The dedicated step re-derives `SymmetryFieldResourcePlan` from the **emitted**
steps, modelled on `RefreshGardenResourcePlan` rather than extending
`NodeResourceBudget`, so the segmenter and the ledger cannot disagree about how
many passes a frame encodes. Only its simultaneous-binding ceiling is gated;
adding its per-pixel terms in `resource_preflight` would double count the pass
the rack ledger already charges.

Motion displacement is scaled by exactly **one reference tick**
(`1.0 / TEMPORAL_REFERENCE_FPS`), never by program or wall time, which bounds
the ±64 UV/second vector to ±2.13 UV at full gain. Both gate lanes survive:
`clamp(gate.x) * clamp(gate.y)`, confidence times validity/occlusion, so a field
is never applied at full confidence through a closed gate. The Symmetry hue
rotation is byte-identical to `rack_node.wgsl`'s, asserted by source text so an
edit to one without the other fails.

Modulation exposes exactly thirteen continuous controls — `base_folds`,
`fold_offset`, `radial_phase_deg`, `orbit_phase`, `planar_axis_deg`,
`planar_phase`, `cell_skew`, `spiral_scale`, `orbit_radius`, `orbit_spin_deg`,
`motion_gain`, `hue_span`, and `center` — with the three angular keys on the
degree-wrap allowlist. Mode, boundary, seed, the four routes and the six mask
bits are enumerable authored topology and get no modulatable address. Dice and
procedural generation mutate the same thirteen values and nothing else, so
neither can reroll the sector table. Morph interpolates the values only on an
exact four-slot route match that also compares both masks — a differently armed
field describes a different dependency graph, not two ends of a blend — switches
mode and boundary at the midpoint, and recalls the seed as an endpoint rather
than interpolating an RNG. The browser edits the thirteen values, both discrete
vocabularies, the six mask bits and the seed through the ordinary coalescible
parameter action; the seed selects a table but routes nothing, so it rides
`SetVisualNodeParam` like the Cellular, Shift and Grain pattern seeds. Only the
four fields that rewrite the dependency graph use the ordered,
revision-protected `SetVisualNodeSymmetryRoute { scope, node_id, route,
composition_revision }`, whose `route` is the closed slot-tagged vocabulary
`Image { index, route } | Motion { index, layer_id }`. It is never coalesced and
never quantized, and an out-of-range slot is a typed refusal, never a fallback
onto slot 0. `NodeParamType::MotionDonor` exists so a motion route is not
mislabelled as an image tap.

Export consumes the same evaluated plan, the same `SymmetryFieldExecutor` and
the same `symmetry_field.wgsl`; there is no export-only symmetry path, and the
pass's only time input is the shared frame-plan context, which offline derives
from `frame_num` and the export FPS. The `.motion.json` sidecar records, at
schema version 4, the resolved-or-missing identity of **every** image and motion
slot **by slot index** for every authored Symmetry Field — armed or not, and
never compacted.

What is proven and what is not: group closure, identity seams, the analytic
sector and boundary fixtures, CellularReentry's structural non-recursion, the
sector-table hash law under donor loss, the single rounding point, the
1,024-byte assertion, the 8-admits/9-refuses ledger with ordinary segments still
capped at 3, and the full patch/Morph/modulation/Dice/generator/preset/browser/
native/export closure are covered by ordinary CPU tests. Default-readback
carrier bit-identity, CPU-reference agreement per mode, missing-donor and
incomplete-motion neutrality, live/export byte equality across the two layer
identity schemes, warm-allocation invariance, and the labeled export render are
covered only by opt-in `#[ignore]` fixtures on one adapter
(AMD Radeon RX 6950 XT / Vulkan 26.7.1). The eight-texture portability claim
rests on the S2 receipt's enforced-cap argument — every device in this tranche
is requested with `Features::empty()` + `Limits::default()`, byte for byte the
production request — and not on backend coverage. The motion hand-off is
likewise proven only on that adapter: independent committed-parity selection per
slot, the two-current-PreLocal-donor refusal, and — the payoff — a known uniform
codec field driven through a donor whose own Motion is exactly zero, moving the
frame against both a stationary field and an unarmed slot while the live and
export layer-identity schemes stay byte-identical and warm frames still allocate
nothing.

### The Field Collider

`Field Collider` recombines **two** primitive motion fields into one derived
field and hands that field to the existing Faraday transplant to advect its
carrier. It is a **Motion-subsystem block, not a Collision Rack node**: it takes
no `NodeKindTag` code, occupies no rack segment, appears in no image dependency
graph, and claims no image tap. `FIELD_COLLIDER_ALGORITHM_VERSION` is 1 and
append-only.

**Two closed vocabularies.** `FieldColliderMode` has permanent codes 0…4: `Sum`
0, `Difference` 1, `Curl` 2, `Projection` 3, `CollisionBoundary` 4.
`MotionBoundaryMode` is `Transparent | Mirror | Wrap | Hold` with codes 0…3 —
**the same frozen numbering `DisplaceBoundary` and `SymmetryBoundary` already
carry**. §5 lists those four names in the order "transparent, hold, mirror,
wrap"; that listing is prose enumerating the vocabulary, not a code assignment.
Motion deliberately does **not** differ: minting a fourth incompatible boundary
table so that `1` meant Mirror for an image and Hold for a field would be a
persistence and shader hazard for no authored benefit. One boundary numbering
serves the whole program. `Transparent` is the default and the only law that
removes a lookup; a non-finite coordinate is removed by **every** law, including
the three that otherwise always produce a sample, because `clamp`, `fract`, and
the triangular map are all meaningless on NaN.

**The recombination law.** For validated recipient-local `a` and `b`, with
`d = a - b`, `m = (a + b) / 2`, and `eps = 1e-12`: Sum is `a + b`; Difference is
`d`; Curl is `(-d.y, d.x)`; Projection is zero when `dot(b,b) <= eps` and
`b * dot(a,b) / dot(b,b)` otherwise; CollisionBoundary is `m` when
`dot(d,d) <= eps` and `m - d * dot(m,d) / dot(d,d)` otherwise. After the
per-mode formula and **before** gating, every component clamps to the canonical
Motion velocity range — exactly the interval `pack_velocity` encodes and
`unpack_velocity` recovers, so no mode can emit a velocity the frozen M4 field
contract cannot represent. `clamp_motion_velocity` clamps without quantizing:
the derived surface is `Rg16Float`, so applying the 16-bit lattice on the CPU
side would make the reference disagree with the shader by construction.

Confidence and visibility are **componentwise minima**. The Faraday gate then
applies threshold/softness/occlusion exactly once, downstream, in
`motion_apply.wgsl` and `motion_refresh.wgsl`; the collider never pre-applies
it. Any missing, aliased, out-of-range, non-finite, or singular-transform input
yields the **exact invalid/zero sample** — it never reuses the surviving input
and never reuses a prior derived field, because either would present an
observation the collider did not make.

**Admission is one predicate.** `MotionParams::collider_admission(is_master)` is
the whole law, and the planner-collect, executor-encode, and dependency-walk
sites all call *that function* rather than three hand-copied predicates that can
drift apart — the S1–S4 three-site discipline satisfied by construction.
Authored inertness is reported before any environmental fault, so a disabled
block never accuses its scope of a problem it does not have. `enabled = false`
is exact M4 and delegates before any admission or allocation. Enabling **parks**
the single-donor transplant recipe rather than ambiguously running both: the
authored donor, amount, carrier, confidence, refresh, decay, and occlusion are
all retained verbatim, and disabling resumes them exactly because nothing was
ever erased. `FieldColliderDiagnostic` is typed and telemetry-safe —
`InputMissing`/`InputUnselected` name their slot, plus `AliasedInputs`,
`MasterRecipient`, `NoActiveTransplant`, `InputFieldUnavailable`, and
`SingularTransform` — and carries authored identity only, never a host path.

**Both inputs demand honest primitive fields.** Each names its donor through the
established `required_as_donor` flag, so a donor whose own Motion is exactly
zero still yields a field, and each resolves through
`EvaluatedMotionScopePlan::admitted_field_slot` so the collider observes the
same field motion rendering wrote. Input A may equal the recipient and input B
may equal the recipient — a layer colliding its own field against another's is
authored topology, not a cycle — but **A and B may never alias each other**.
Slot identity is route identity: A and B are named fields, never a list, so
clearing A can never slide B's donor into its place.

**Two low-resolution passes inside the unchanged three-texture ceiling.** Pass 1
binds A's and B's vector parities (two sampled textures) and writes
`[a.xy, b.xy]` into one transient `Rgba16Float` pair surface, mapping each
vector by `linear(inverse(R) · D)` — translation excluded. Pass 2 binds that
pair plus both gates (three sampled textures) and writes the transactional
`Rg16Float` derived vector and `Rg8Unorm` derived gate. Coordinates map by
`uD = inverse(D) · R · uR` and the boundary applies **independently** to each
input's vector and gate lookup, so one input leaving its extent never silences
the other. Because the split does both maps, the derived field is already
recipient-local and indexed in composition output UV, so the Faraday advection
pass consumes it under **identity** transforms — a collider recipient has no
`donor_scope` to read, and reading one would be a category error rather than a
missing scope.

The uniform is exactly 144 bytes — two 64-byte `MotionTransformGpu` records plus
one 16-byte mode/status lane — with a compile-time
`const _: () = assert!(size_of::<ColliderGpuUniforms>() == 144)`. The two
admitted status bits are independent, so one input's singular transform closes
only its own lane.

**Eight prebuilt parity bind groups, exactly one bound per pass.** The two
inputs are independent fields whose committed ping/pong parities are selected
separately, so all four `[A parity][B parity]` combinations are prebuilt for
each pass — four plus four, never one shared table — mirroring S4's motion-group
split. A warm frame binds one per pass and creates nothing.

Resource delta per admitted collider, charged through
`FieldColliderResourcePlan`:

| Item | Bytes/cell |
|---|---:|
| Derived vector parity (two `Rg16Float`) | 8 |
| Derived gate parity (two `Rg8Unorm`) | 4 |
| Transient mapped pair (one `Rgba16Float`) | 8 |
| **Collider-specific total** | **20** |

| Item | Exact charge |
|---|---:|
| Low-resolution passes | 2 |
| Nearest lookups | 5 |
| Simultaneously sampled textures | 3 |
| Prebuilt bind groups per collider | 8 (4 map + 4 collide) |
| Bind groups bound per pass | 1 |
| New full-frame persistent surfaces | 0 |
| Uniform bytes | 144 |

`MOTION_MAX_ACTIVE_COLLIDERS` is 1, matching the single admitted transplant.
Both primitive input fields and the sole carrier stay separately and honestly
accounted through the M4 ledger; only these three surfaces are new, and
resource admission precedes every allocation. Derived attachments are internal
executor values: `CodecMotionProduct`, the live codec field cache, and export
codec acquisition describe **primitive** acquisition only and are never extended
to carry them.

**Transactional in the established shape.** Staging order is primitive →
derived → carrier, and all three commit or discard together with the
prior/current spatial state. Program Freeze stages nothing and derives nothing —
the last committed derived field is what the carrier keeps reading. Media Freeze
does advance the program, so it re-derives transactionally from committed
primitive observations. A frame with an unmaterialized input still *derives*,
writing the exact zero sample, so a derived parity can never hold a stale field
from an earlier topology. Reset invalidates every derived parity and the pending
recipe **without reallocating**: the surfaces and the eight prebuilt groups
survive, only the published validity does not.

**Closure.** Persistence stores strict version, mode, boundary, and two saved
donor identities, serialized through the existing `MotionDonorConfig` and
published through the existing `MotionDonorSnapshot` — there is no parallel
donor encoding. Both slots recompute their saved positions independently at
capture, both survive `MotionConfig::sanitized` with their Selected-versus-
Missing intent intact, and a `Missing` tombstone never rebinds after reorder,
removal, replacement, Morph, or patch load. The two collider donors are remapped
on the Motion-subsystem path the transplant donor already travels —
`remap_motion_collider_inputs_after_move`/`_after_remove` beside
`remap_motion_donor_after_move` in `morph.rs`, and `motion_donors_mut` in
`main.rs` for the runtime `preserve_motion_donor_after_remove` /
`refresh_motion_donor_saved_position` pair — never the `saved_node_*` /
`remap_saved_*` visual-rack walkers, which match on `VisualNodeKind` and would
never see a Motion-block donor.

Version 1 adds no collider-only continuous control, so Dice, the procedural
generator, and modulation preserve the block **exactly** and it has no
modulatable address of any kind. Morph chooses the entire discrete block as one
endpoint recall at the midpoint rather than picking field by field: a per-field
pick would be identical only by accident and would start synthesizing third
configurations the instant v2 adds a field whose meaning depends on the mode or
on which pair of donors is armed. Look carries the recipe — enabled, mode,
boundary — while both inputs stay live topology. An omitted patch section is
exactly the pre-collider path, and a declared version other than 1 is rejected
at deserialize time rather than migrated.

Browser topology uses the ordered, revision-protected, uncoalesced barrier
`SetMotionColliderInput { layer_id, input, donor_layer_id,
layer_stack_revision }`, forbidden inside `Quantized` batches. `input` is a
closed named token (`"a"` | `"b"`), following the Residual slot precedent, so an
unknown slot is a deserialization rejection rather than a positional fallback
onto the partner input. The aliasing law is answered by the engine, which is the
only side that knows both current values; the edit is refused rather than
silently clearing the partner. `enabled`, `mode`, and `boundary` are values and
travel on the ordinary coalescible `set_motion`.

Export consumes the same evaluated plan, the same two passes, and the same
`motion_collide.wgsl`; there is no export-only collider path. The
`.motion.json` sidecar is at schema version 5, whose one additive
`field_collider` section records — only after an accepted frame — the authored
identity of **both** slots by name, the admitted output slot, the typed
diagnostic, the byte-exact budget, and the discrete law. Both slots are emitted
always and never compacted, so a retained tombstone is recorded as a tombstone
rather than re-resolved against whatever now occupies the vacated position.
Vectors, pair texels, gate parities, raw codec records, host paths, and
filesystem metadata never enter it.

What is proven and what is not: the two closed vocabularies and their frozen
codes, every mode against its analytic definition, the clamp into the canonical
velocity range, componentwise gate minima, the exact zero sample under every
hostile input, all four boundary laws including NaN removal, the complete
admission law with alias/missing/unselected/master/no-transplant refusals, the
20-byte ledger with one-byte-over rejection on every independent bound and the
one-collider cap, the 144-byte compile-time assertion, the coordinate-versus-
vector transform law, the transactional commit/discard/freeze/reset lifecycle, a
checksummed recovery-journal round trip, and the full
patch/Morph/modulation/Dice/generator/preset/browser/native/export closure are
covered by ordinary CPU tests. The pixel claims —
`production_field_collider_derived_field_reaches_the_pixels` in
`renderer::composition::tests`, which proves the derived field advects the
carrier and reaches the audience image, that both inputs demonstrably
contribute, that a missing input is byte-identical to the neutral pair, that
live and export layer identities render the same authored patch identically, and
that a disabled collider is byte-identical to exact M4 — plus
`render_field_collider_pipeline` as the labeled export case, are opt-in
`#[ignore]` fixtures measured on one adapter (AMD Radeon RX 6950 XT / Vulkan
26.7.1). Cross-platform portability rests on hosted three-platform CI, not on
that adapter.

### B14 the sync latch

"A model that always recovers is a model that cannot actually break."
`servo_defeated` landed inside B3; `sync_latched` is B14's other half and
closes it. The seat is the tape/NTSC-adjacent horizontal shear: each
reference tick some bands of scanlines lose sync and slip sideways.
Unlatched a slip lives exactly as long as its own tick and the picture heals
— the B8 knock's law, bit-clean between firings. **Latched, every slip is
written into a bounded per-line offset table and stays there, accumulating,
until the switch is released and the whole displacement unwinds in one
step.** `src/sync_latch.rs` is the independent CPU reference in the
`gesture.rs` tradition and `sync_latch.wgsl` consumes the table it produces.

**Bounded state may latch but never grow.** The table is the entire latched
state: one `f32` per output line, capped at `SYNC_LATCH_MAX_LINES` (2,160)
and `SYNC_LATCH_MAX_OFFSET` (0.25 output UV) per line — 8,640 bytes at the
absolute cap. The stage owns **no texture at all**, allocates no surface, and
appears in no ledger; its only GPU resource is one 8,656-byte uniform (a
16-byte header plus the capped table, compile-time asserted). Accumulation is
bounded three ways, each with its own fixture: per slip
(`SYNC_LATCH_SLIP_UV`, 0.02 UV at full amount), per line (clamped on every
fold), and per frame (`SYNC_LATCH_MAX_TICK_BURST` = 24, the
`history_ticks_for_delta` burst clamp, so a long stall cannot bill the table
for every skipped tick at once).

**The seat** is a Recipe-B stage on the shared slot-0 seam, between the B8
melting edge and the B4 display stage — `temporal → melt → sync latch →
display → opaque resolve`. That order is the law: a sync fault happens in the
signal, and the screen model downstream then shows it. Live LegacyExact, live
Advanced, the selective-VHS path, and export all converge on this adjacency,
so one implementation and one shader serve all four (three live call sites,
two offline). The three existing shear producers were surveyed and rejected
as seats: the B8 knock is bus-scope and absent from LegacyExact, the ntsc-rs
worker is third-party CPU work with no latch vocabulary, and Shift is a
layer/master effect rather than tape-adjacent. The stage therefore takes the
B8 dirt law as its *model*, not its code.

**Laws.** Every draw is the Shift band/epoch/seed law on the shared integer
avalanche (`mixing_boundary::lane_unit`) in the fresh `LANE_SYNC_FIRE` /
`LANE_SYNC_SLIP` domains ("SYN" 1 and 2), keyed by the master `random_seed`,
the stage's own 30 Hz ordinal, and the band index — no sequential RNG state,
so a tick recomputes alone. `band_height(spread)` spans 1 line to 64, and
every line in a band carries the identical offset, which is what makes a tear
a tear rather than static. `band_fires` draws against `rate * 0.5`. `bias`
folds the symmetric draw toward one side while keeping its magnitude, so at
±1 accumulation is monotonic to the cap. A sheared line **wraps**: `fract`
puts the coordinate back inside the frame and the sampler repeats on U, so
the bilinear tap straddling the seam filters across it rather than clamping.
The wake law is `amount > ε ∧ rate > ε` — neither control alone wakes the
stage, and the switch deliberately does not appear in it, because latching an
inert stage accumulates nothing. Beyond that the executor **skips encoding
entirely whenever every offset is zero**, so the exact prior path never
resamples at all. Release is expressed as "unlatched implies an empty table"
rather than as a falling-edge handler, so the two cannot drift apart. Pulling
`amount` to zero while latched stops new damage and holds what is already
done: the switch is what repairs.

**The table is program memory.** Blackout does **not** clear it — the
temporal-ring and bus-melt precedent, deliberately not B4's phosphor, which
models the screen itself. The audience goes dark; the program keeps its
damage, and a release resumes from the picture the cut interrupted.
`reset_for` clears on exactly the causes that begin a new program
(`PatchGeneration`, `ApplyLook`, `BroadRevert`, `Resize`, `ManualClear`) and
holds through `SourceCut`, `Seek`, and `BlackoutTransition`; one hook,
`Renderer::reset_visual_generation_for`, already carries all three required
causes. **The table is deliberately absent from patches**: the switch and its
four controls persist, while the accumulation is runtime state like a
temporal ring, regrowing deterministically from the seed and the clock — so
pre-B14 bytes and canonical hashes keep and live and offline agree from any
common start. Program Freeze needs no special handling: the stage is fed
`program_advancing_delta()` like its two neighbours, so Pause holds the fault
clock still, and blackout stays the absolute final audience operation.

**Closure.** Patch: `TemporalConfig.sync`, skip-serialized at the exact-off
default; hostile scalars sanitize to neutral and unknown fields are rejected.
Wire: five `sync_*` params on the ordinary coalescible `set_temporal` in both
validators plus the shared `apply_temporal_wire_edit`. Snapshot: an additive
`sync` block plus the read-only `sync_damaged` fact — the authored switch
says what was asked for, that says whether the program is *actually* still
broken, which is the fact a failure switch exists to make visible. Panel: a
SYNC LATCH group in the temporal section (four sliders — the static range
count is **202** — plus the switch; the `app.js` template pin stays 24).
Modulation: four continuous master addresses (`sync_amount`, `sync_rate`,
`sync_spread`, `sync_bias`) inserted before `morph`, which stays last; the
switch is a discrete law with no address, because a failure switch is thrown,
never swept. Morph blends the four values and recalls the switch at the
midpoint. Dice and the generator preserve the whole block exactly
(`GENERATOR_VERSION` stays "12"). The sidecar schema stays 6, the renderer
texture floor stays 30, and `temporal.wgsl`'s pinned SHA is untouched — the
stage owns its own shader. `render_sync_latch_pipeline` is the labeled export
case: its `_healed` twin carries the identical four controls with the switch
off, so both renders draw the identical fault stream and differ only in
whether the faults heal.

**Two named boundaries.** The switch has no native surface, matching every
other temporal discrete law (`fb_servo_defeated`, `mosh_recycle`,
`disp_model`): the native patch editor captures temporal state rather than
editing it, and giving temporal discrete laws a native surface is a separate
change across roughly forty controls. And `active` is a WGSL reserved
keyword, so the uniform's arming lane is named `armed` on both sides.

### B2 procedural motion fields

`MotionFieldSource` gained one arm: `Procedural(ProceduralFieldKind)`, a
deterministic synthetic field computed by `motion_procedural.wgsl` and defined
by the CPU reference `motion::procedural_field_sample` — the law the shader
follows expression for expression. `ProceduralFieldKind` is a closed vocabulary
with permanent append-only codes: `Curl` 0, `Radial` 1, `Spiral` 2, `Contour`
3, `Chroma` 4, `Weave` 5. Wire/patch/sidecar tokens come from the one
`ProceduralFieldKind::source_key` table (`procedural_curl` … `procedural_weave`)
so no stringify site can disagree.

**The field laws.** All six share `freq = 1 + scale * 15` cycles across the
frame, `phase = program_time * rate` in turns, and the
`PROCEDURAL_FIELD_MAX_SPEED = 8` UV/s amplitude — one eighth of the canonical
±64 ceiling, and every component still passes `clamp_motion_velocity`. Curl is
the analytic curl of a three-octave sinusoidal stream function (the frozen
`CURL_OCTAVES` constants), divergence-free by construction. Radial pulses
outward on `cos(TAU * (freq * r - phase))`; Spiral pitches the same ring 45°;
Weave is orthogonal sinusoidal shear. Contour and Chroma observe the
recipient's image as **covered premultiplied linear RGBA** — the exact
`covered_source_linear` quantity, so hostile RGB behind zero coverage steers
nothing by arithmetic — flowing along luma isolines (perpendicular to a
central-difference gradient one field cell out) and along the phase-rotated YIQ
chroma pair respectively. The four pure kinds bind a 1×1 defined-zero neutral
their shader never reads and report fully open gates; Contour and Chroma report
an honest gradient/saturation confidence so flat content contributes nothing.

**A procedural field is a primitive field.** It flows through
`admitted_field_slot`, is eligible as a Collider input and a donor via
`required_as_donor`, and writes the existing `Rg16Float`/`Rg8Unorm` parity
pair. The ledger delta is one low-resolution pass and **zero bytes**: no luma
ping-pong is charged (the `luma_bytes` predicate never matches a procedural
origin) and no new surface exists. The origin carries its kind because Contour
and Chroma bind the scope's image while the pure kinds bind nothing —
`MotionFieldOrigin::signature_code` (0–3 frozen, kinds at 4–9) feeds the
topology signature so a kind change re-prepares rather than reusing stale bind
groups. Publication is the codec-upload law: a freshly synthesized parity every
program-advancing frame, valid from the first. Synthesis is derived from
program time, not acquired from media, so Media Freeze and a paused layer hold
decoders while the field keeps advancing; Program Freeze holds it exactly as it
holds everything. The pass's only time input is the shared frame-plan context
(`FramePlanContext::time_seconds`), never wall time, so Pause holds the field
still and export parity is structural.

**Authored scalars.** `ProceduralFieldParams { scale, rate }` are scope state
like the shutter's: they persist and modulate whether or not the current source
is procedural, so switching kinds never erases them. `scale` is unit-clamped
(neutral 0.5); `rate` clamps to ±2 turns/second (neutral 0.25); non-finite
input takes the neutral, never a clamped extreme.

**Closure.** Patch: the `procedural` block is skip-serialized at default, so
every pre-B2 patch keeps its bytes and canonical hash; hostile scalars sanitize
on load and unknown fields are rejected. Wire: the kind rides the existing
`field_source` value (MemoryTopology impact); `field_scale`/`field_rate` are
ordinary coalescible values at both scopes (ValuesOnly). Modulation adds
exactly two continuous addresses per scope — `motion_field_scale` `[0,1]` and
`motion_field_rate` `[-2,2]`, master and `layerN_` — appended to the end of
`LAYER_TARGET_SUFFIXES` (58, 59) so every compiled suffix index is stable; the
kind itself is discrete authored state with no address, exactly as the field
source always was. Morph interpolates the two scalars and switches the kind at
the midpoint through `field_source`'s existing pick. Dice and generator v8
mutate only the two scalars, each in a fresh RNG domain, so every pre-B2 stream
is byte-stable; the `GENERATOR_VERSION` bump to "8" names the fact that a
generated piece now carries the two new values. The snapshot's additive
`procedural` block and the sidecar's existing `requested_source` string carry
the state; no sidecar schema bump, because no field was added. Export consumes
the same plan and shader, codec acquisition skips procedural origins, and
`render_procedural_motion_field_pipeline` is the labeled export case — a
deliberately tapless single-layer stack, legal since the composite rank landed.

**Flow shaping.** `FlowShapingParams { stretch, edge_repel, vector_trash,
trash_block_size }` shapes the field the advection pass *applies* — after
sampling and gating, before the trajectory offset — so it acts on every field
kind: codec, lattice, procedural, or the derived collided field. The law is
the CPU reference `motion::shape_flow_velocity`, mirrored in
`motion_apply.wgsl` and ordered stretch → repel → trash → canonical clamp.
Stretch grows the flow radially by local field magnitude; edge repel pushes
down the carrier's covered-luma gradient with the push saturating at one full
luma step per texel; vector trash shoves whole cells by hashed garbage vectors
gated per cell per tick on the fixed `FLOW_TRASH_EVENT_HZ` (8 Hz) event clock,
with `vector_trash` the firing probability, under the shared
`cellular_avalanche` integer hash in the fixed "MTRS" domain — no authored
seed, so live and offline replay identically from frame-plan time. Shaping
runs only under a valid applied field and only when an amount is authored:
all-zero shaping is byte-exact with the unshaped path (no clamp, no texture
operation), proven by a decoded-frame-identical A/B across the change. Edge
repel charges exactly four covered-luma taps per fragment in
`motion_pass_budget`, only while nonzero; stretch and trash are arithmetic.
The four controls are ordinary continuous values everywhere: ValuesOnly
ingress, coalescible wire params at both scopes, `motion_stretch` /
`motion_edge_repel` / `motion_vector_trash` / `motion_trash_block_size`
modulation addresses (layer suffixes 60–63), Morph blend, Dice/generator v8
mutation in fresh domains, a skip-at-default `shaping` patch block, and an
additive snapshot block. `render_motion_flow_shaping_pipeline` is the labeled
export case and renders an `_unshaped` twin, asserting the decoded frames
differ — shaping demonstrably reaches the pixels and an authored zero
demonstrably remains a different program.

What is proven and what is not: kind codes/keys, resolution at every scope, the
analytic per-kind fixtures, numeric divergence-freedom for Curl, alpha-covered
Chroma neutrality, canonical-range clamping, zero luma bytes, the
procedural-versus-codec Collider planner fixture, the shaping law's analytic
stretch/repel fixtures, deterministic trash firing with its probability gate,
shaped-velocity clamping under hostile inputs, and the full
patch/Morph/modulation/Dice/generator/browser/export closure are hosted CPU
tests. `gpu_procedural_field_matches_the_cpu_reference_for_every_kind`
(worst |GPU − CPU| ≤ 0.008 UV/s across all six kinds) and the two labeled
export cases are opt-in `#[ignore]` fixtures measured on one adapter (AMD
Radeon RX 6950 XT / Vulkan 26.7.1).

### Named two-input Residual Counterpoint

`Residual` is a Collision Rack node that recombines one route's large-scale
structure with the carrier's detail measured against a second route.
`NodeKindTag::Residual` holds append-only signature code 11; kind codes are
never renumbered or reused.

The recombination law, in linear premultiplied space:

```text
dc  = quantize(mean(structure))
ac  = quantize(carrier_premultiplied - mean(detail))
out = dc + detail_gain * ac
```

Authored state is `ResidualParams { algorithm_version, structure, detail,
block, quantization, mix, detail_gain, seed }`. `mix` is
`finite_clamp(v, 0.0, 0.0, 1.0)` and `detail_gain` is
`finite_clamp(v, 1.0, 0.0, 4.0)`, so a non-finite input takes the neutral value
rather than a clamped extreme. `ResidualBlock` is
`Four | Eight | Sixteen | ThirtyTwo | SixtyFour` with codes 0…4, edges
4/8/16/32/64 and `Eight` as the default. `ResidualQuantization` is
`Off | Coarse | Medium | Fine` with codes 0…3 and levels 0/8/32/128; `Off`
means exact identity, never a one-level collapse. A fixed `seed` shifts the
quantization lattice per cell and is recalled, never re-drawn or interpolated.

The node carries **two** authored route slots — slot 0 `structure`, slot 1
`detail` — and both are read only through their reduced block means, never at
full resolution. That is what keeps the recombination pass inside three
simultaneously sampled textures: the carrier, `mean[0]` and `mean[1]`. A
route's history age is exactly N or N-1 per slot, because `EdgeTiming` is
`CurrentFrame | PreviousFrame` and nothing deeper is representable; the 24-deep
clean-composite ring is a separate Temporal budget and is not reachable here.

`ResidualParams::is_exact_bypass()` is `sanitized().mix == 0.0`, and the
default is therefore an exact bypass. It is a real delegation: the planner
collects no tap, the executor encodes no pass and allocates no mean surface,
and the saved-patch dependency walk claims no edge. The admission predicate is
character-identical in `collect_rack_taps`, `flush_segment`, and
`collect_rack_dependencies`: enabled ∧ wet > 0 ∧ `!is_exact_bypass()`.

Each mean cell is the premultiplied average of **four quadrant-centre loads**
of its block — a bounded 4-tap estimator, not an exact box integral. Resource
delta per active node, charged through the existing descriptor ledger:

| Item | Exact charge |
|---|---:|
| Full-frame render passes | 1 |
| Reduced-resolution passes | 2 |
| Logical lookups/pixel | 3 |
| Explicit texture operations/pixel | 12 |
| Simultaneously sampled textures | 3 |
| Cross-scope image taps | 2 |
| New full-frame persistent surfaces | 0 |
| Reduced-resolution surfaces | 2 |

Block-mean bytes are charged through the byte-exact `ResidualResourcePlan`,
never through `additional_rgba16_layers`, and meet the creative number only at
the shared `MAX_CREATIVE_GPU_BYTES` cap where motion bytes join it. Every
independent bound is its own typed rejection before any allocation: grid edge
≤ 2,048, ≤ 2,100,000 cells per node, exactly 8 bytes per cell, exactly 2
surfaces per node, ≤ 32 MiB per node, ≤ 64 MiB aggregate, ≤ 3 sampled
textures, a 256-byte uniform stride, and ≤ 16 nominal active nodes. An
over-budget grid is rejected with an actionable error, never silently clamped
to a coarser one. `MAX_SAMPLED_TEXTURES_PER_PASS` stays 3.

Routes, block, quantization and seed are stable authored topology. Morph
interpolates `mix` and `detail_gain` only when both slots name the exact same
pair of routes, switches `block` and `quantization` discretely at the midpoint,
and carries slot A's `seed` verbatim. Dice and procedural generation mutate
only `mix` and `detail_gain`, each node from its own stable RNG domain, and a
generated mix that wakes a dormant edge records a transactional fallback.
Modulation exposes only `mix` and `detail_gain`, under globally unique
modulatable descriptor keys.

The browser edits `mix`, `detail_gain`, `block`, `quantization` and `seed`
through the ordinary coalescible parameter action. The two donors — the only
fields that rewrite the image dependency graph — use the ordered,
revision-protected `SetVisualNodeResidualRoute { scope, node_id, slot, route,
composition_revision }`, which is priority, never coalesced and never latched
into a Quantized batch. `slot` is a closed tagged vocabulary
(`structure | detail`), not an index, so an unknown slot is a deserialization
rejection rather than a positional fallback onto the partner input. Snapshot
params are `{ structure_tap, detail_tap, block, quantization, mix,
detail_gain, seed, diagnostic }`, and the diagnostic names the dead slot so a
tombstone can never be read as belonging to the other route.

Export consumes the same evaluated plan, the same two reduced block-mean
passes and the same rack shader; there is no export-only recombination path.
The `.motion.json` sidecar (schema 4) records, per admitted node, the scope,
the discrete recombination law, the seed, and each slot's resolved or
tombstoned route identity — stable identities only, never a host path or
filesystem metadata.

### The Study rack node

`Study` is the data-only Study ABI's authored audience surface: a Collision
Rack node holding append-only signature code 13, lifted — like the Symmetry
Field — into its own dedicated pass because it binds the committed
clean-history D2 array and owns its own uniform layout
(`occupies_dedicated_pass` is true for exactly these two kinds).

**Content-addressed by construction.** `VisualNodeKind` is `Copy`, so
`StudyRackParams` carries only `document_digest: Option<[u8; 32]>` — the
same canonical digest `CompiledStudy` derives. Documents live in the bounded
host `StudyProgramLibrary` (16 documents, hard cap, typed refusal, never an
eviction) and travel with patches in the `studies` section, carried whole
like the gesture track, so a patch stays self-contained and the distribution
question never opens. An unresolved digest — no document in the library —
plans an inert pass with `program: None`: byte-identical to no node at all,
never a fallback onto another document.

**Resolution is plan-visible identity.** The planner resolves the digest
against `CompositionPlanInput::with_studies` at plan time and the encoded
program rides the evaluated plan, so live and export execute the identical
instruction stream. Node id, digest, resolvedness, and instruction count
hash into the advanced topology signature; assigning a document — or a
library insert that resolves a previously missing digest — re-prepares the
renderer and re-uploads the program arena.

**Frame inputs.** `FramePlanContext` carries `study_audio_bands` and
`study_beat_phase`, sampled from the same immutable frame facts the
modulation matrix consumed (live) or from the export's own audio evaluation
and beat clock (offline). Ring validity and the write cursor come from
`temporal_history_read_cursor` at encode. The node's wet/blend apply through
the engine-wide node law, composed from the one canonical `blend.wgsl`
kernel.

**The admission budget.** The descriptor declares eight logical lookups per
pixel: the carrier load plus up to seven `LoadHistoryColor` loads
(`LoadCurrentColor` reads the already-loaded carrier register and costs
nothing). A valid document exceeding that stays valid ABI but is refused at
plan time by name (`StudyLoadBudget`) — the over-budget Residual-grid law,
never a silent clamp.

Resource delta per active node, charged through the dedicated-pass ledger
(`StudyFieldResourcePlan`, re-derived from emitted steps):

| Item | Exact charge |
|---|---:|
| Render passes | 1 |
| Logical lookups/pixel | ≤ 8 (declared admission budget) |
| Simultaneously sampled textures | 2 |
| Samplers | 0 |
| Uniform bytes | 8,256 (64 frame + 8,192 program) |
| Cross-scope image taps | 0 |
| New full-frame persistent surfaces | 0 |

**Closure.** Version 1 has no continuous authored value and no routes, the
Field Collider precedent: no modulatable address, Dice and the generator
preserve the node exactly (its common wet dices like every node's), Morph
recalls the pair as one discrete endpoint at the midpoint, and Look/preset
value application leaves the digest untouched. The browser assigns,
replaces, or clears the document through the coalescible
`set_visual_node_study_document { scope, node_id, document }` — the engine
validates and compiles into the library and the node keeps only the digest,
in one action, so neither can exist without the other. A malformed document
is a typed refusal; panel JSON parse errors stay client-side in a polite
status region. Export resolves the same digests from the patch's own
`studies` section through `ExportCreativeGraph`; there is no export-only
Study path, and `render_study_field_pipeline` is the labeled export case.

### The B1 Scan Processor

`ScanProcessor` is a Rutt/Etra-style drawn raster and the tree's **first
non-fullscreen-triangle pass**: one instanced triangle-strip ribbon per
scanline, no vertex buffers — position from `vertex_index`/`instance_index`
(the fullscreen-triangle tradition, extended), the carrier fetched in the
**vertex stage** through the explicit-load premultiplied bilinear,
sampler-free like every dedicated pass. Ribbons accumulate additively into a
shared transient `Rgba16Float` accumulator cleared to alpha one
(contributions carry alpha zero, so coverage cannot stack past unity where
lines bunch), and a fullscreen resolve applies the engine-wide node wet/blend
law through the one canonical `blend.wgsl` kernel. What the mechanism buys is
line *density*: bright caustic ridges where scanlines bunch and dark gaps
where they splay, which no fragment-shader displacement can produce, because
a fragment shader has no notion of line density. `NodeKindTag::ScanProcessor`
holds append-only signature code 14 and `occupies_dedicated_pass = true`
(the Symmetry/Study lift, for a stronger reason: no ordinary segment could
ever encode geometry).

**The algorithm, kept whole.** The beam law is derived from BENDR (MIT,
© 2026 Steve Blythe); the `beam_position` composition order and the
beam-energy law are transcribed faithfully with attribution, and
`scan_processor.rs` is the independent CPU reference the WGSL follows
expression for expression (the `gesture.rs` tradition: no wgpu, clock,
filesystem, or UI dependency). Composition order: sweep/field reversal
(applied to the *read*, so reversing mirrors the picture, never the raster),
S-curve, skew, deflection oscillator, raster collapse, luminance into
vertical deflection about the 0.35 luma pivot, then tilt/perspective as a
photographed 2D deflection — never a scene. The oscillator locked
(`osc_lock = 1`) quantizes to a whole multiple of the field rate and stands
still; detuned it crawls, and the crawl is the instrument's gesture. The
central-difference tangent gives the ribbon normal and the beam speed at
once, and `gain = clamp(2 / speed, 0.05, 8)` mixed by `velocity_mix` is the
beam-energy law — a slower beam deposits more energy per unit length, and
that one term is the difference between this and a displacement map. Luma is
alpha-covered Rec.709 in linear light (the house rewrite of BENDR's 601
gamma-space luma), so hostile RGB behind zero coverage steers and draws
nothing by arithmetic. A degenerate tangent takes the vertical normal rather
than dividing by zero — the one transcription deviation, and it is the house
non-finite law, changing no finite path.

**Authored state.** Nineteen params, prefixed `scan_*` on the wire.
Fifteen continuous and modulatable: `amount`, `ribbon_width`,
`velocity_mix`, `tilt_x`, `tilt_y`, `perspective`, `s_curve`, `skew`,
`collapse`, `osc_amount`, `osc_freq`, `osc_lock`, `lissajous`, `mono`,
`hue`. Two plan-time geometry integers with no modulatable address —
`lines` (16–1,080, default 320) and `samples_per_line` (64–512, default
256), the Residual block-grid law, because they size the instanced draw and
the vertex ledger. Two discrete laws: `reverse_h`, `reverse_v`. Non-finite
input takes the neutral default, never a clamped extreme.

**The wake law.** `is_exact_bypass()` is true when no *deflection* is
authored: amount, collapse, oscillator amount, S-curve, skew, both tilts all
zero and both reversals off. Dressing controls (ribbon width, velocity mix,
perspective, oscillator frequency/lock, Lissajous, mono, hue) shape a raster
that exists only once a deflection is authored — perspective without a
depth term is arithmetic identity — so they do not wake the node. BENDR's
own stage gate is the precedent; ours widens it to include skew and the
tilts, which genuinely author deflection alone. A default node is an exact
bypass: the executor encodes nothing and the carrier passes through
untouched, byte-identical to no node at all.

**Ledger.** Charged through the node descriptor plus the dedicated
`ScanProcessorResourcePlan` re-derived from emitted steps:

| Item | Exact charge |
|---|---:|
| Render passes | 2 (instanced geometry + fullscreen resolve) |
| Vertices | `lines × samples × 2`, the tree's one named vertex budget, cap `MAX_SCAN_PROCESSOR_VERTICES` = 1,105,920 |
| Vertex-stage carrier fetches | 3 covered bilinears per vertex (here/ahead/back), 12 loads |
| Resolve lookups/pixel | 2 (dry carrier + accumulator, one `textureLoad` each) |
| Simultaneously sampled textures | 2 (geometry binds 1; resolve binds 2) |
| Samplers | 0 |
| Uniform bytes | 128 (compile-time asserted) |
| Shared transient accumulator | 1 full-frame `Rgba16Float`, 8 B/px, charged once while any step exists |
| Cross-scope image taps | 0 |
| New full-frame persistent surfaces | 0 |

The vertex cap is structural (the authored maxima admit exactly it) and the
lift still refuses one vertex over with the typed
`ScanProcessorVertexBudget` — the Residual grid-edge law, defense in depth,
never a silent clamp. The topology signature hashes pass count and layout
only, deliberately never the vertex total: lines/samples are draw-call
arguments, so a geometry edit re-encodes the next frame without re-preparing
pipelines, arenas, or the accumulator.

**Closure.** Patch: the params ride the node's ordinary serde
(`kind: scan_processor`), absent from every pre-B1 patch so old bytes and
canonical hashes keep; unknown fields are rejected and hostile scalars
sanitize to neutral. Wire: all nineteen params ride the ordinary coalescible
`set_visual_node_param` — no routes exist, so the node has no topology
action at all. Snapshot: the nineteen values plus derived read-only
`scan_exact_bypass` and `scan_vertex_count`. Panel: a generated node card
with fifteen sliders, two number inputs, and two toggles. Modulation:
fifteen `scan_*` stable addresses (none angular — the tilts are authored in
signed radian units, not degrees); geometry and reversals have no address.
Morph: the fifteen values blend, and lines/samples/reversals recall an
endpoint at the midpoint; no route gate, so any scan pair interpolates.
Look/preset: the whole params bundle transfers as values (no route to
preserve). Dice and generator v9 mutate the fifteen continuous values in
each node's own stable domain and never touch geometry or reversals; no
`GENERATOR_VERSION` bump, because no pre-existing anchor contains a scan
node and no existing seed's output changes. Export consumes the same
evaluated plan and the same `scan_processor.wgsl`, with time from the shared
frame-plan context only — Pause holds the detuned oscillator still and
export replays it structurally; `render_scan_processor_pipeline` is the
labeled export case.

### The B6 corruption trio

Three block-domain corruption mechanisms as three Collision Rack node kinds
— **Block DCT** (append-only kind code 15), **Pixel Sort** (16), **Filter
Avalanche** (17) — lifted into one dedicated executor
(`renderer/corruption.rs`, `corruption.wgsl` composed with the canonical
`blend.wgsl`; four pipelines, one 80-byte compile-time-asserted
dynamic-offset uniform record, sampler-free, two bound textures per pass).
The laws are derived from BENDR (MIT, © 2026 Steve Blythe) and transcribed
from its shipped chain, then hardened; every law runs in the **encoded sRGB
domain on straight-alpha values** (the B8 code-byte / B5 real-codec
precedent — storage artefacts are quantised where they are stored).
`src/block_dct.rs`, `src/pixel_sort.rs`, and `src/filter_avalanche.rs` are
the independent CPU references in the `gesture.rs` tradition.

**Why dedicated, three different reasons.** The DCT is four full-frame
passes through two shared float coefficient intermediates (BENDR's fused
O(N²) fragment reassociated into O(N) coefficient/reconstruction stages per
axis — the same sums, proven against the transcription); the sort's
faithful 32-tap run search is 34 honest lookups per pixel, more than the
frozen 32-lookup ordinary-rack budget admits for a whole rack; the
avalanche reads its own previous output, a retained per-node history the
ordinary rack cannot express. This forced a principled ledger split:
dedicated passes' per-pixel terms leave the frozen per-rack ceilings (which
keep their exact meaning for the ordinary pass chain) and are governed by
`MAX_LOGICAL_TEXTURE_LOOKUPS_PER_DEDICATED_PASS` (= 65, exactly the widest
pass in the tree — the avalanche's carrier plus 32 gradient taps at two
loads each — so a wider future kind is a deliberate raise, never silent
headroom), while still summing into the frame-level ceilings. The lift is
kind-only (the Scan shape): a default node still owns its step and uniform
slots, and `is_active` gates every encode.

**The laws.** Block DCT (`dct_amount/quantize/hf_penalty/chroma_crush/
block`, all five continuous and modulatable; block edge 4–16 via
`floor(4 + block·12)`): DCT-II with orthonormal scaling both directions,
quantiser step `(0.004 + q·0.5)·(1 + u·tilt·2)` rounded to nearest, chroma
crushed in the coefficient domain against Rec.601 luma before the round;
alpha rides untransformed; wake is amount alone (the 0.004 step floor means
quantise-zero is never claimed as identity). Pixel Sort
(`sort_amount/threshold`): a pixel above the encoded-luma threshold
searches upward through ≤32 taps stepping two rows for its run's end and
takes the end colour mixed by amount; taps clamp, never wrap. Filter
Avalanche (`avalanche_amount/run` continuous, `avalanche_axis ∈
sub|up|average` codes 0–2 discrete with no modulatable address): per-lane
gate at `amount·0.5`, bounded gradient sum (`span = 2 + run·40`, ≤32 taps,
out-of-frame taps masked but non-terminating), per-lane epoch-invariant DC
seed, `fract` wrap. Two hardenings BENDR never claimed: the lane epoch is
`floor(frame-plan seconds × 3)` through the shared integer-avalanche hash
keyed by the node's **stable authored id** (persisted topology — never a
process-lifetime layer id), so Pause holds the fault stream and export
replays it; and the accumulation reads the node's own previous output —
one retained working-format surface per node on the bus-melt transaction
(lazy before the warm-allocation snapshot, invalidated never freed on
disarm, staged/committed with the frame history, blackout-immune program
memory), advanced at most once per 30 Hz reference tick so live and export
cascade at the same speed. A cold history reads the carrier, which is
exactly BENDR's shipped single-frame law. `MAX_AVALANCHE_NODES` is 4
(8 B/px each), a typed `AvalancheHistoryBudget` refusal at five.

**Ledger** (per step, re-derived from emitted steps as
`CorruptionResourcePlan`): DCT 4 passes / 17 lookups per widest pass / 2
textures / two shared full-frame `Rgba16Float` intermediates charged once
per frame at 16 B/px (sequential reuse, the Scan-accumulator law); sort 1
pass / 34 lookups / 2 textures; avalanche 1 pass / 65 lookups / 2 textures
/ one 8 B/px retained history per node; 80 uniform bytes per pass; zero
samplers, zero image taps, zero new wire actions.

**Closure.** Patch: ordinary tagged node serde (`kind: block_dct |
pixel_sort | avalanche`), absent from pre-B6 patches so old bytes and
canonical hashes keep; unknown fields/tokens rejected; hostile scalars
neutral. Wire: values on the ordinary coalescible `set_visual_node_param`
with prefixed keys (`dct_*`, `sort_*`, `avalanche_*` — bare names
cross-resolve under `same_wire_parameter`); `avalanche_axis` on the server
gate's closed enum allowlist; no routes, so no topology action. Modulation:
the nine continuous values only. Morph: any same-kind pair interpolates
(route-free); the axis recalls an endpoint at the midpoint. Look/preset:
whole-bundle value transfer in all three appliers. Dice and the generator
mutate the nine values in per-node stable domains, never the axis; no
`GENERATOR_VERSION` bump (new kind arms, the B1/B7 precedent). Panel:
generated node cards (the pinned range counts do not move). Export rides
the same plan and shader; `render_block_dct_pipeline`,
`render_pixel_sort_pipeline`, and `render_filter_avalanche_pipeline` are
the labeled export cases, each with a `_clean` exact-bypass difference twin
and a `_repeat` determinism assertion.

## The B9 performance recorder

It records gestures rather than pixels: while recording is armed, every
accepted authored value edit is written down as a `(tick, param_address,
value)` event on the 30 Hz authoring reference, and a finished take plays
back in real time — or offline, frame-indexed — against completely different
footage. The law is derived from BENDR (MIT, © 2026 Steve Blythe); the house
adaptation records accepted edits at the coalesced drain rather than
change-sampling a control surface, and **the patch is the opening state**: a
take is carried whole inside the patch, so no synthetic keyframe exists.

**The portable contract** (`src/performance_track.rs`, the gesture-track
substrate arm for arm): `PERFORMANCE_REFERENCE_FPS` *is*
`TEMPORAL_REFERENCE_FPS`; events are quantized codes only (8 bytes: `tick:
u32le | address:u16le | value:u16le`, cap 16,384, `truncated` honesty);
SHA-256 over a domain-separated little-endian field stream
(`collide-o-scope/performance-take/v1\0`) with the `truncated`/`incomplete`
flags *inside* the digest; bounded serde both ways with a hand-validated flat
address codec (serde ignores `deny_unknown_fields` on tagged enums, so the
codec refuses hostile extra fields by hand); the clock derives its tick
before adding the frame's delta. The address table (cap 256, hashed with the
events) interns each control once: a closed typed `PerformanceControl`
(append-only codes 0–14) plus the `PerformanceValueLaw` its value lane rides —
`Unit {min,max}` (Q16 over the declared range), `Discrete {vocab}` (token
index), `Toggle`, `Stepped {min,max}` (exact integers, because integer
appliers reject a non-integer wire number). The law is captured at first
sight, so a take's lattice can never shift under its own events. Layers are
addressed by saved stack position — the morph-slot identity — never by
process-lifetime live IDs.

**The value-law oracle** (`performance_value_law_for`) answers from the
engine's own tables: `modulation::target_range` for continuous families
(master by wire name; layer effects and pattern scalars through `layer1_…`
suffixes; temporal through `fb_*→temporal_fb_*` / `disp_*→display_*` maps),
the owning enums' `ALL` tables for every discrete vocabulary, and the
validators' integer clamps for stepped laws. `None` is a counted refusal,
never a guess: seeds, `score_loop_driver`, motion sources/qualities/carriers,
and the reset tokens. Safety controls (blackout, freeze, pause), topology,
and routes are outside the vocabulary by law, not omission.

**Recording.** The tap sits on `handle_web_action_inner_with_feedback` — the
seam every final application funnels through (browser drain, downbeat
release, native RECOVERY, transform gizmo) — so a take stores what the
program actually did: coalesced-away values never dispatch and never record.
Staged edits commit at the same accepted, program-advancing gate the temporal
and gesture recorders share (a free function over disjoint fields, because
the renderer borrow is live there); a rejected or frozen frame neither
consumes a tick nor records. Arming starts a fresh take at tick zero;
disarming stamps the declared length; a capture mid-recording carries the
take marked incomplete inside the hashed flags.

**Replay is delegation, not reimplementation.** Playback compiles the take
once at arm — each event becomes a real `WebAction`, layer positions bound to
the live stable IDs occupying them — and dispatches due events per frame
through the transform-gizmo seam, so Morph ownership transfer, sanitize, and
every engine refusal apply exactly as to a hand-made edit while manual
history records nothing. A guard flag hides the replayer's dispatches from
the record tap. An unresolvable address degrades to a named no-op in the
snapshot's `degraded` list and is never retargeted. Dispatch runs after the
downbeat release, only while the program advances — a take is an automatic
clock and Pause freezes it; the playhead advances at the acceptance gate;
loop rewinds cursor and clock. Recording and playback are mutually exclusive
by refusal.

**Closure.** Patch: `performance_take` carried whole beside `gesture_track`,
skip-serialized when absent, gated by its checksum-verifying deserializer,
restored strictly after the generation barrier; the three authored barriers
clear the recorder and a source cut deliberately does not. Wire:
`set_performance_recording` / `set_performance_playback` (revision-guarded at
ingress and dispatch) and `clear_performance_take` — uncoalesced priority
barriers, refused in `Quantized` batches at both gates, excluded from manual
history. Snapshot: the additive `performance_recorder` block (the
`performance` name belongs to the clip/scene subsystem). Panel: the TAKE
RECORDER group beside GESTURE FIELD (second-column group pin 7 → 8; buttons
only, range pins 198/21 unmoved). A take has no modulation address, no Morph
slot, and Dice/the generator preserve it exactly (`GENERATOR_VERSION` stays
"12"). Export verifies the checksum before the first frame, replays due
events at `export_temporal_reference_tick` into the authored bases through
the same appliers the live arms use — `EffectsSnapshot`,
`apply_spatial_transform_edit`, `apply_motion_param`, the extracted
`apply_temporal_wire_edit` (the `set_temporal` match body moved whole so live
and export mutate `TemporalParams` through identical code), `BusMixerEdit`,
`PatternSynthEdit`, the shared layer scalar clamps — replays once straight
through (loop is live transport), and publishes `<output>.performance.json`
via the staged no-replace commit, cleanup-coupled and claim-retired like the
gesture sidecar. The motion sidecar schema stays 6.

**v1 boundaries.** Values only: node/group rack params, text-page values,
pad/gyro/audio/MIDI/LFO/routing configuration, and morph law/glide/capture
are not yet recordable — the address vocabulary is append-only, so each can
join without breaking a stored take. MIDI/OSC direct-control edits bypass the
wire-action seam and are not recorded. `render_performance_recorder_pipeline`
is the labeled export case: the `_untaken` twin must decode differently and
the `_repeat` render identically — the record/replay determinism claim.

## The B11 monitoring bay

Preview-only instruments over a low-resolution readback of an internal
signal: the difference between "the picture is doing something odd" and
knowing which part of the model is doing it. The law is derived from BENDR
(MIT, © 2026 Steve Blythe), whose scope dock is this instrument whole;
`src/monitor_bay.rs` is the independent CPU reference in the `gesture.rs`
tradition, and the two presenters — native egui and the panel's first
`<canvas>` pair — draw the identical CPU-derived bitmaps.

**The readback.** One 128×72 reduction (BENDR's shipped scope size, inside
the tranche's ≤160×90 bound) of the selected probe image, on the B10
`video_analysis` machine verbatim: the same 16-tap linear-light reduction
expression (CPU reference `modulation::reduce_analysis_grid`, the B10 law
generalized over grid size — `reduce_video_analysis_grid` now delegates to
it), a lazy stage with its own two-slot FIFO busy-drop readback pool
(one 36,864-byte target + two 36,864-byte buffers, outside the full-frame
floor, which stays 30; the three full-frame audience slots untouched), and
the same 10 Hz cadence: three reference ticks on an accepted-program-seconds
accumulator at the program-tap acceptance seam. Pause holds the instruments;
under blackout the probed slot-2 image is the held pre-blackout picture
(program memory, the tap's own law). The bay's bind group is rebuilt per
10 Hz sample against the probe's view — cheaper than every epoch hazard a
cached group would carry.

**Zero cost hidden.** `App::monitor_bay_armed` is the one gate: the native
overlay (bay toggle ∧ `native_controls_visible`) OR a fresh browser watcher.
Unarmed, the stage is never constructed, no pass encodes, no buffer maps,
the snapshot block is the empty default, and the disarm edge clears the
instruments once so a re-arm never resurrects a stale picture. Browser
watching is the `gyro_stream` socket-layer shape — `monitor_watch
{ enabled }` per client, set without a queue round-trip, dropped on
disconnect — hardened with `MONITOR_WATCH_TIMEOUT` (10 s): the panel
re-asserts on a 4 s heartbeat exactly while its MONITORING BAY section is
expanded and the tab visible, so a silently discarded tab expires instead of
pinning the readback armed.

**The instruments.** Computed on the CPU from the harvested grid: the
waveform plots Rec.601 luma of the encoded bytes (the `VideoAnalysisState`
constants — the instrument observes the stored picture, not a linear
reconstruction), one column per grid column, additive saturating
accumulation; the vectorscope plots BENDR's projection
`u = (b−y)·0.565, v = (r−y)·0.713` at gain 1.4, and the six 75% colour-bar
targets are derived from that same projection (`scope_targets()`), never
restated as constants, so the graticule cannot drift from the cloud.

**PROBE.** A closed append-only vocabulary of retained renderer-owned
images: `program` (pre-blackout slot 2, default), `program_tap` (the honest
N-1 image), `gesture_canvas` (the presented etch donor). An unavailable
producer is the named `probe_status: "unavailable"` — instruments hold, the
readback stays idle, and the probe never silently rebinds. The strip beside
the scopes carries the modulation matrix's live source values
(`modulation::MONITOR_SOURCE_LIST`, 45 sources, append-only order, pure
reads). The named deferred probes — NTSC per-line state, the melt band mask,
a motion-field visualizer — join by appending codes.

**The sealed permit.** `MonitorBayPermit` is the transform-gizmo seal,
deliberately not stage_health's weaker file-scope shape: declaration and the
single mint live inside a private submodule, pinned by a source-audit test,
and the token folds all three conditions — bay toggle,
`native_controls_visible`, `StageSurface::EditorPreview` — so the painter
cannot be reached for an audience surface even by a caller that got one
condition wrong. `show_monitor_bay` joins the one-predicate block in
`main.rs`. Native paint re-uploads its two egui textures only when a fresh
sample lands (dirty-flag pre-resolve before the closure, the stage-health
law).

**Closure.** Nothing persists in patches: the bay toggle and probe are
host-session state like the media-safety mode, absent from `PatchState`; a
new process starts disabled on `program`. Wire: `set_monitor_bay` /
`set_monitor_probe` (ordinary coalescible host operations outside manual
history; the probe token validated at the gate and the applier through the
one `MonitorProbe::try_from_str` table) and the never-queued
`monitor_watch`. Snapshot: the additive `monitor_bay` block — probe and
native-overlay always truthful, base64 instrument payloads only while
armed, a `sample` counter so the panel redraws at the 10 Hz arrival rate.
No modulation address, no Morph surface, no Dice/generator interaction
(`GENERATOR_VERSION` stays "12"), no export arm — export has no observer,
so there is no labeled export case of its own; the range pins (html 198,
app.js template tags 24) did not move.

## Gesture-field etching

A gesture is an ordered stream of quantized events addressed on the 30 Hz
authoring reference, not a stroke of pixels. `src/gesture.rs` owns the portable
contract; `src/gesture_canvas.rs` owns the bounded vector canvas it etches.
Neither has a `wgpu`, clock, filesystem, or UI dependency, so the CPU field is
the independent reference the GPU is checked against rather than a description
of the shader.

### The portable event contract

`GESTURE_ALGORITHM_VERSION` is 1 and append-only. `GESTURE_REFERENCE_FPS` *is*
`TEMPORAL_REFERENCE_FPS` — the same 30 Hz constant reused, never a second
literal — so a gesture recorded at 24, 30, or 60 fps lands on the same tick.
`MAX_GESTURE_EVENTS` is 4,096, `MAX_ACTIVE_STROKES` is 16 and equals the width
of the open-stroke `u16` mask by compile-time assertion,
`MAX_GESTURE_SERIALIZED_BYTES` is 256 KiB, and `MAX_GESTURE_DECAY_TICKS` is
4,096.

One event is 18 fixed-width bytes: `tick:u32le`, `stroke:u8`, `phase:u8`,
`mode:u8`, `reserved:u8`, `x:u16le`, `y:u16le`, `pressure:u16le`, `dx:i16le`,
`dy:i16le`. Position and pressure are Q16 (`value/65_535` → `[0,1]`); direction
components are Q15 (`value/32_767` → `[-1,1]`), renormalized on decode.
Quantization is the *only* representation — there is no float path, so live and
replay see identical bits. `GesturePhase` is `Begin|Move|End` and `GestureMode`
is `Push|Curl`, both with permanent append-only codes.

Well-formedness is validated identically on ingest and on decode: ticks
non-decreasing, `stroke` in range, a `Move`/`End` with no open `Begin` and a
second `Begin` for an open stroke rejected rather than repaired, and a track
whose strokes are not all closed valid but explicitly *incomplete* and never
auto-closed. Over-cap ingest sets `truncated` and returns `false` — the
`TemporalEventTrack::record_accepted` law — and never panics or drops silently.

The canonical checksum is SHA-256 over a domain-separated explicit
little-endian field stream, imitating `recovery_journal::record_checksum`
rather than hashing JSON:

```text
b"collide-o-scope/gesture-track/v1\0"
|| version:u16le || flags:u16le || origin_tick:u64le || event_count:u32le
|| each event's 18 bytes
```

`GESTURE_FLAG_TRUNCATED` and `GESTURE_FLAG_INCOMPLETE` are inside the hashed
stream, so a truncated or open-stroke recording can never present itself under
a complete recording's digest. The checksum covers the portable stream only and
is therefore invariant to how events were grouped into frames. Serde is bounded
on both sides: count-capped `visit_seq` visitors, `deny_unknown_fields`, and a
byte cap checked on encode *and* decode. Hostile input is refused by the cap,
never by allocating first and measuring after.

### Reference ticks and the pause law

Live recording uses `GestureEventRecorder`, `TemporalEventRecorder`'s exact
shape: it accumulates accepted, program-advancing seconds and derives the tick
*before* adding this frame's delta, so the first accepted frame records at tick
0. Offline uses `export_temporal_reference_tick(frame, fps)` — the rounded
rational map, no accumulator. Program Freeze does not call the recorder and a
rejected frame does not advance it, so neither accumulates catch-up debt for
time the audience never saw; a frozen canvas frame likewise neither decays nor
etches and consumes no reference address. Wall time never enters the track, the
checksum, the canvas, or anything derived from them. Restoring a track from a
patch resumes the clock at that track's last recorded address instead of
rebasing, so a recovered recording's digest is byte-identical to the saved one.

### One normalized adapter

`normalize_gesture_input(origin, raw, tick)` is the only path from an input
surface to an event. `GestureOrigin` is `NativePointer | Phone | Midi | Osc`
and is provenance only: it reaches `GestureIngestError` and the host's status
text, never the event bits, so the same logical gesture drawn on a tablet, sent
by the phone panel, played from a MIDI controller, or received over OSC records
byte-identical tracks and one identical checksum. An out-of-range stroke or a
non-finite position/pressure is refused, not clamped, because clamping would
merge two strokes or invent a mark the operator never made; only the direction
vector is sanitized, to unit length with an inert zero fallback.

**The honesty law.** An unrecorded gesture is never implied replayable. While
recording is disarmed a normalized sample still etches the current session's
canvas and nothing enters the track. `GestureStatusSnapshot` keeps
`recorded_events` and `live_only_events` as separate counters and derives
neither from the other, publishes `truncated` and `open_strokes` as their own
facts, and an empty track publishes no checksum at all rather than a digest of
nothing. Export decodes the recorded document through the same validator and
re-derives the canonical checksum before the first frame renders; a mismatch is
an actionable export error, never a silent re-render.

### The canvas

| Item | Bound |
|---|---:|
| Grid edge | ≤ 2,048 |
| Grid cells | ≤ 2,100,000 |
| Working bytes per cell | 12 |
| Presented donor bytes per cell | 8 |
| Bytes per canvas | ≤ 32 MiB |
| Active canvases | ≤ 2 |
| Aggregate bytes | ≤ 64 MiB |
| Uniform stride | 256 |
| Decay ticks per update | ≤ 4,096 |
| Ordered samples per update | ≤ 256 |

Each is an independently checked limit with its own typed `GestureCanvasError`,
evaluated before any allocation and reconciled afterwards against the resources
actually created. Twelve bytes a cell is an `Rg16Float` signed-vector ping-pong
pair plus an `Rg8Unorm` coverage/hold ping-pong pair, breaking format uniformity
for small surfaces exactly as `renderer/motion.rs` already does. The presented
donor is one `Rgba16Float` image charged as its own named class against the same
narrowable ceilings, deliberately never folded into the frozen twelve, so the
working-set reconcile keeps its exact prior meaning.

`GESTURE_CANVAS_HOST_MAX_CELLS` (262,144) is a *host* narrowing, not a second
table: the portable field is the CPU reference and runs on the render thread,
and a `const _: () = assert!(…)` makes the constant structurally unable to
exceed the frozen bound. `gesture_canvas_host_grid` halves both edges together
until the edge cap, the cell cap, and the host budget all hold.

`Push` displaces along the stroke direction; `Curl` along its perpendicular.
Both are analytic — a closed-form `(1 - d/r)²` falloff, zero with zero slope at
the rim, never an iterative solve — so live and offline agree exactly.
Overlapping strokes compose in *recorded order*: each sample blends the field
toward its own etched vector, which does not commute, so reordering two
overlapping strokes is a visible difference and order is part of the contract.
Decay is closed form, so a long gap is one operation rather than a loop; the gap
clamps to the tick budget and reports `clamped` instead of billing every tick,
mirroring `history_ticks_for_delta`'s 24-tick burst clamp. `hold` decays at the
authored rate too, so retention stays finite and nothing etches permanently.

### Transactions and resets

`GestureCanvasState` is transactional in the `temporal.rs` shape: `stage_frame`
snapshots before it changes anything, `commit_staged` drops the snapshot, and
`discard_staged` restores it, so a discarded frame leaves no visible change on
either side. The staged plan and the staged *evaluated* parameters travel with
the snapshot, so a renderer encodes the frame the CPU reference actually applied
— including this frame's modulated radius, strength, and retention — rather than
the authored values a later read would return.

`GestureCanvasResetCause` is a typed vocabulary, never a boolean, resolving to
two independent domains: the etched field and the decay clock. `PatchGeneration`,
`ApplyLook`, `Resize`, `BroadRevert`, `ManualClear`, and `ExportCancelled` are
hard resets; `SourceCut` and `SourceReplacement` rebase the clock and keep the
etch, because a seek must stop the canvas billing skipped program time as decay
without erasing what the operator drew. A reset abandons an open transaction
rather than restoring it — a reset is not an undo — and every cause also raises
the device flag that clears both working parities and the presented donor before
the next etch.

### The routable field

`SavedImageSource::GestureCanvas` / `ResolvedImageSource::GestureCanvas` /
`PlannedImageSource::GestureCanvas` join the closed image-route vocabulary
(serde tag `gesture_canvas`, plan hash code 7), selectable by any existing image
tap — Displace donor, Mask image matte, group matte — with no new node kind, no
new bind slot, and no new surface. It is a master-scope singleton with no ID and
no saved position: every positional accessor answers `None` and no missing-layer
or missing-group tombstone can touch it. It is a producer with **no scope**, so
it claims no dependency and no ordering edge, a same-frame route cannot close a
cycle even from the master scope that owns it, and it charges zero retained tap
surfaces on both sides of the fail-closed composition ledger — its bytes are
charged once, by `GestureCanvasPlan`. With no canvas admitted the route resolves
to `Transparent` with the named `GestureCanvasUnavailable` diagnostic and never
rebinds to a layer or a group.

The presented donor is the exact inverse of the frozen `displace_node` decode:
straight `RG = clamp(vector, -1, 1) * 0.5 + 0.5`, premultiplied by the gate's
coverage as alpha, blue an explicit zero. Because the decode subtracts the same
alpha the presentation multiplied in, an un-etched cell, a zero-gate cell, and a
missing binding all decode to *exactly* zero — the hostile-hidden-RGB law holds
by arithmetic rather than by a second rule — while coverage scales the decoded
displacement, which is the gate doing its job.

The device half publishes that donor once per committed frame, from the
committed parity, at the acceptance decision, so a routed donor reads the field
as of the previous accepted frame. Acceptance is only known after the frame
encoder has been submitted, and encoding into that encoder would advance the
device field on rejected frames; this is the same N-1 law ProgramHistory already
obeys, and offline export applies byte-for-byte the same ordering.

### Closure

Modulation exposes exactly three continuous destinations — `gesture_radius`,
`gesture_strength`, `gesture_retention`, each `0…1` — and no address of any kind
reaches the recorded track. `PatchState` gains two optional sections:
`gesture_track` carries the whole checksum-verified document, because a track is
topology rather than a value, and `gesture_canvas` carries the three authored
scalars. An absent section is exactly the pre-gesture path, a hostile section
sanitizes on load, and unknown fields are rejected. A Morph slot holds the
recording's *identity* — its canonical checksum — and blends the three canvas
values only when both slots name the exact same recording, so no morph position
can synthesize a third recording neither slot captured. Dice and the generator
move authored values only and never invent or mutate a recording.

A completed authored gesture is **exactly one** manual-history entry, opened at
its first `Begin` and closed at the matching final `End`. Every `Move` between
them is deliberately invisible, so a long stroke cannot flood the bounded stack,
and an automation origin (`Midi`, `Osc`) records no entry at all, matching
`MutationOrigin::records_manual_history`.

The browser sends `gesture_sample`, which deliberately has **no** coalesce key:
replacing an older pending sample would delete path points the operator drew.
`Begin` and `End` hold an admission reservation, because a dropped edge orphans
a stroke or leaves it permanently open; an intermediate `Move` is ordinary and
may be shed under saturation, and the track then honestly records what the host
accepted. `set_gesture_recording { enabled, layer_stack_revision }` is an
ordered priority barrier, never coalesced and never latched into a `Quantized`
batch, so a sample cannot cross an arm/disarm edge into the wrong recording and
a stale arm decision cannot arrive after a patch load replaced the program.
`set_gesture_canvas` is an ordinary coalescible absolute scalar with no path to
the track. `phase` and `mode` cross the wire as the engine's own closed
vocabularies, so an unknown token is a deserialization error rather than a
silently defaulted value.

Export writes `<output>.gesture.json` beside the render —
`{ version, origin_tick, event_count, truncated, checksum, events }` — through
the staged atomic no-replace commit idiom, refusing to overwrite an existing
sidecar and cleaned up with the video on cancellation. Operational paths and
filesystem metadata never enter it, and a job with no recorded track writes no
sidecar at all.

## Patches and native parameter editor

`PatchState::capture` includes master and layer state, stable source identity,
master pause, NTSC, temporal, the complete modulation/input configuration, and
a normalized morph snapshot. `Ctrl+S`/`Ctrl+O` use file dialogs;
`Ctrl+E` exposes the patch parameter editor in the native panel. `Ctrl+S`
serializes the complete `PatchState` as YAML; the native editor intentionally
edits the live master/layer parameter subset rather than presenting itself as
a full YAML text editor.

The native preview also owns the browser-independent **RECOVERY** strip: truthful
panel-listener state and browser count, the exact authenticated loopback link,
absolute Freeze Program and Blackout setters, broad `ResetVisualProgram`, and
active-library Choose/Rescan. Its second row carries non-empty `output_error`
and media-source status. Hide it whenever single-monitor Output reuses the main
surface. A dedicated output surface is already clean and may leave the strip on
the preview. Folder selection is host-local and is not an arbitrary remote
filesystem action; choosing or rescanning a folder does not add a layer.

Before `patch.apply`, rebuild and validate the complete layer stack and stage
saved imported analysis audio through full decode. Missing/corrupt visual or
audio input rejects the snapshot without changing live master/audio/topology or
generations; a legacy patch with no modulation section preserves current audio
state. After commit, clear the immediate web queue, downbeat-latched queue, and
already-drained action-batch remainder; advance `layer_stack_revision` and the
application visual epoch, clear renderer temporal history and retained NTSC
output, and invalidate pending readbacks so downstream Spout/NTSC consumers
reject work from the previous patch.

A successful Apply Look has a narrower barrier. Filter conflicting master,
mapped-layer, reroll, all topology, and present NTSC/Temporal actions from the
drained remainder plus immediate/latched queues, including work admitted while
the native dialog was open. Preserve unrelated transport/safety, unmapped-layer,
and omitted-section actions in order. Cancel/error/stale look selection is not a
barrier.

Backward compatibility rules matter:

- no `modulation` section leaves the existing matrix untouched;
- old layers without `source_path` resolve by library filename;
- old temporal `slit_axis` values map to 0° or 90°;
- legacy `reset_fx` resets only direct master effect uniforms; the bundled
  panel uses `reset_visual_program` for the broader master-program revert;
- absent Shift fields mean amount 0, block size 8 px, density 0.5, and speed 3,
  preserving the exact pre-Shift visual path;
- absent spatial state means the inactive historical full-frame sample with a
  Transparent authored edge for any later movement; explicit Clamp persists;
- no media-safety field means Safe; media mode is process-local rather than
  patch-persistent, so an untrusted or legacy patch cannot enable Expert;
- new finite/clamp defaults reject NaN/overflow without panicking.

## Procedural generation and source identity

`media_source` is the single resolver for exact visual patch load, offline
visual reconstruction, and imported analysis-audio reconstruction. Preserve
ordinary patch compatibility: try an explicit path, patch-relative and active-
library logical names first. A `cos-sha256://<sha256>/<bytes>` reference is a
different contract: accept only a file with the recorded length and SHA-256,
including a bounded non-recursive search of patch/library roots. Never let a
same-named mismatch satisfy a content reference.
Do not collapse a resolved content reference back into a host path. Visual
layers and imported analysis audio retain the persisted reference separately;
capture/save emits it, and UI export passes any live resolved path only as a
transient hint alongside the expected identity. Re-fingerprint that hint so a
post-load mutation cannot bypass provenance, including for patch-adjacent files
outside the active library.

Fingerprinting uses a fixed 1 MiB streaming buffer, per-invocation canonical-
path cache, cancellation checks, before/after metadata consistency, at most
4,096 searched entries, and a default 64 GiB total-read budget. Generation's
`--max-fingerprint-bytes` may lower or raise that invocation budget explicitly;
zero is invalid. Operational paths and filesystem metadata must not enter
shareable manifests or receipts.

Generator v8 normalizes the anchor, replaces verified file sources with content
references, reduces filenames to logical names, and hashes version-prefixed
canonical JSON with SHA-256. `anchor_sha256`, `piece_sha256`, and lineage must be
path-independent and source-byte-sensitive. Schema-v2 manifests retain defaulted
v1 fields for deserialization compatibility. Generator v8 also applies bounded,
reflected spatial mutations through independent RNG domains without changing
saved Fit/Edge/Sampling choices. Each generated piece stages
`patch.yaml`, `manifest.json`, and deterministic `preflight.json`, then commits
the directory no-replace; preflight all known names and serializations before
the first commit. The receipt claims canonical configuration and source bytes,
sets `pixel_identity_claimed` false, and records exact limits and warnings.

The CLI accepts optional `--library`, `--max-fingerprint-bytes`, and
`--allow-unverified-sources`. The last is an explicit incomplete-identity mode:
preserve only a logical name, set `identity_complete` false, and emit a privacy-
safe warning. It is independent from `--allow-black-sources`, which acknowledges
Spout's deterministic-black offline policy.

Generation remains patch-only. Do not claim that it batch-renders MP4s or that
its preflight receipt is an artifact/pixel proof. A cancellable sequential GPU
session with time/disk budgets and artifact receipts is still deferred; bounded
clip statistics/cache work follows profiling; visual-parameter audio DSP remains
research-gated behind a defined signal, smoothing, loudness, and test contract.

## Offline export

The offline renderer uses synchronous decoders intentionally. It evaluates
beat, route slew, pad spring, morph glide, layer transport, temporal history,
and frame counts from the selected FPS. Live audio/MIDI values are zero.
Unavailable/missing/live sources become deterministic black placeholders at
their original source indices.

Source reconstruction may use the current host's media policy and must retain
any above-Safe reservation for the export lifetime. This does not authorize a
larger export target: `validate_export_dimensions` keeps the established UHD-
area output boundary. Selective export keeps its separate staging/working-set
checks. Do not describe Expert as an 8K-output or native-VHS-budget override.

Selective VHS export uses the shared planner and persistent NTSC processor
synchronously. Render contributing slices after local effects and conditional
direct master effects, process only inherited slices through VHS, composite in
the shared bottom-to-top law, upload the returned straight-alpha image, and run
Temporal once afterward. Honor master Pause by holding materialized Morph/mod
bases and time. Reuse one staging allocation sequentially; validate dimensions,
slice count/length, and cancellation. If planning, readback, processing, or
upload fails, report an actionable export error and abort—never fall back to a
path that violates the layer's bypass. The no-bypass/VHS-off legacy order stays
unchanged.

Optional audio uses the selected video's first audio stream under a deliberate
independent policy: start at source time zero, run once at 1×, ignore visual
pause/speed/modulation/looping, pad silence when short, and trim when long.
FFmpeg maps raw video and audio explicitly, strips metadata, and fixes the
program duration. Do not change this policy silently; it is part of the output
contract.

The `.motion.json` sidecar is written once, atomically, after FFmpeg succeeds
and is removed with the video by `remove_started_output`. Schema version 4
appended a `symmetry_fields` section carrying, per authored Symmetry Field, the
owning scope and the resolved-or-missing identity of every image and motion slot
**by slot index** — armed or not, never compacted, and a retained tombstone is
recorded as a tombstone rather than re-resolved against whatever now occupies
the vacated position. Routes are resolved exactly once per job by
`resolve_export_creative_graph` and Morph carries values only, so the section is
recorded beside the authored motion scopes rather than per accepted frame and
cannot inflate the distinct-state list. Operational paths and filesystem
metadata stay out of it.

Cancellation closes FFmpeg and removes the partial output. Export validation
must include paired `framemd5` runs, not just successful process exit.

## Web and remote control

The server binds HTTP `0.0.0.0:3030` and self-signed HTTPS
`0.0.0.0:3031`. Every client, including loopback, needs the rotating
cryptographic session token. The app opens a tokenized desktop URL and
`/qr.svg` carries the tokenized LAN URL; successful navigation exchanges it
for an HttpOnly `SameSite=Strict` cookie, after which the page removes the key
from the visible URL. WebSocket upgrades and mutating POSTs additionally need
an exact same-origin `Origin` matching the serving `Host`. Never restore a
loopback authentication bypass. The certificate is persisted beneath
`%LOCALAPPDATA%\collide-o-scope\tls` and regenerated when the LAN identity
changes.

Layer messages use immutable live IDs, while reorder messages also carry the
snapshot revision. The engine resolves/rejects them; the UI must not assume an
array index remains authoritative after another controller changes topology.
Morph capture messages also carry that revision and are ordering barriers in
both immediate and beat-latched queues; they must never be coalesced as ordinary
absolute-value edits.
The bounded/coalescing ingress queue is a load-shedding boundary, not an excuse
to omit pointer throttling or final pointer-release messages.

`AppSnapshot::output_error` is the operator-facing failure channel for the
fullscreen output window. Keep the checkbox synchronized to the renderer's
actual surface state and show creation/surface errors instead of implying that
an unavailable output is open.

`AppSnapshot::media_safety` is additive and defaults to Safe when absent. The
bundled panel sends the idempotent, immediate
`set_media_safety_mode { mode: "safe" | "expert" }` action; never make this a
beat-latched creative control. Publish the effective area/byte/device/host-plan
limits and reservation totals so the visible rationale remains authoritative.
Do not publish or invent a portable free-VRAM figure. Switching to Safe changes
future allocations only.

Per-layer proxy state is additive in the layer snapshot:
`proxy_backing_prefix` carries the HUD's eight-character cache-key prefix
while a validated artifact backs the decoder (never a path), and `proxy_note`
carries the engine's session lifecycle/refusal note. Both stay off the wire
when empty. The layer card's Encode proxy control sends
`request_layer_proxy { layer_id }` — priority, never coalesced, never
quantized, stable-ID-only with no positional field at all; a vanished ID is a
safe no-op. Every refusal the native Y key enforces is answered by the same
engine ladder (`request_proxy_for_layer`) and surfaces in `proxy_note`, so
the browser cannot bypass the content-identity contract, and a request is an
operational event: it records no manual history and survives an Apply Look
unfiltered.

Host-session proxy settings are authored, not fixed. `set_proxy_settings
{ scale, frame_rate, include_audio }` is an ordinary immediate coalescible
host action (`host:proxy-settings`, never quantized, never priority — the
`set_media_safety_mode` shape), validated at the server gate and at the
handler by `ProxySettings::authored`, the one authoring door, which always
stamps this build's schema and algorithm versions so a wire tuple can never
smuggle a foreign version into a cache key. The engine holds exactly one
authored owner (`App::proxy_settings`); the HUD assessment, the patch-load
consultation, the encode request, and the hot-adoption job all answer from
it — pinned by a source audit on `ProxySettings::default()` — so the
program can never encode under one settings tuple and consult under
another. `AppSnapshot::proxy_settings` publishes only the three operator
choices. Each tuple is its own content-addressed cache key by design: a
change governs future encodes and consultations, touches no live
proxy-backed layer and no published artifact, and a load consults the
cache under the current session tuple only. Like the media-safety mode,
the value is process-local, absent from patches, and a new process starts
at the default; an operator-facing summary reports every installed tuple
and every typed refusal, which leaves the authored owner untouched.

Live NTSC diagnostics remain path-specific. For both the global and selective
paths, `attempted` counts admission decisions, `accepted` counts admitted work,
`skipped` counts only healthy Busy backpressure, and `unavailable` separately
counts a disconnected/failed worker. `stale` counts an accepted asynchronous
result rejected later by the current visual-generation, topology, or path gate.
Use saturating counters. The active-path and busy fields are presentation
context; absent metric data defaults to zero and off. These metrics exclude
synchronous export and do not claim that every accepted result reached an
audience surface.

The panel uses fixed functional columns on desktop and one column under 900
px. Layer reordering sends one atomic `move_layer` on release. Range controls
support double-click/double-tap reset, and group reset buttons send explicit
reset actions. Every range also has an editable spinbutton peer; commits must
dispatch the range's existing `input` path, validate against its min/max/step,
and remain protected from snapshot writes while focused or touched.

After rebuilding embedded assets, hard-refresh any open panel. Only the process
that owns ports 3030/3031 serves the panel; a second instance can otherwise
mislead browser tests.

## Keyboard

- `P` / `Shift+P` — pixelate up/down
- `G` / `Shift+G` — RGB split up/down
- `0` — reset effects
- `Space` — selected-layer pause/resume, or Program Freeze with no selection
- `M` — Media Freeze
- `F` — main-window fullscreen
- `O` — fullscreen output window
- `B` — blackout
- `Y` — encode a proxy for the selected video layer; a layer without a
  verified content identity first mints one through the bounded fingerprint
  machinery (entering persistence when its claim re-validates). Every
  refusal and completion reports through that layer's HUD status line, and a
  completion hot-adopts into every matching live layer at its current
  playhead (falling back to reapply-the-patch wording only when adoption is
  refused or no live layer matches)
- `Ctrl+E` — patch parameter editor
- `Ctrl+S` / `Ctrl+O` — save/load patch
- `Ctrl+Shift+I` / `Ctrl+Shift+X` — import/export the bounded controller profile
- Arrow keys — nudge the selected scope's transform position by `NUDGE_STEP`
  (1/128 output UV); Shift takes the coarse 1/16 step, Alt the fine 1/1024 one,
  and Alt wins when both are held. Refused while a gizmo drag owns the scope.
- `Escape` — cancels an open transform-gizmo drag that has not committed a
  value, undoes one that has, and otherwise keeps its existing close/quit
  behavior

## Verification

- The release gate is `cargo fmt --all -- --check`, both JavaScript syntax
  checks, `cargo check --locked --all-targets`,
  `cargo test --locked --all-targets -- --test-threads=1`, and
  `cargo clippy --locked --all-targets --all-features -- -D warnings` — the
  exact commands, so prose and gate cannot drift. (`Cargo.toml` declares no
  `[features]`, so `--all-features` is a no-op wherever it appears; the test
  step carries no such flag and never has.) A publication claim additionally
  requires Linux, macOS, and Windows CI success for the exact published
  commit SHA; an older green run is not transferable evidence, and CI
  installs `stable` fresh — verify the local toolchain matches it before
  claiming a gate. Verify CI with `python scripts/check-ci-status.py <sha>`:
  a SHA can carry multiple check suites (push and PR runs), so counting
  success conclusions across the flat check-runs list can declare green
  while one suite's job is mid-failure — the script answers per suite, per
  named job.
- Physical-GPU proofs are opt-in and therefore separate from ordinary CI.
  StageMap uses the five `renderer::stage_map::tests::physical_gpu_` fixtures.
  M6 precision uses
  `gpu_precision_receipt_measures_real_still_and_temporal_workloads` plus its
  premultiplied-edge, temporal-feedback, LegacyExact-spatial, and 24/30/60
  temporal parity companions. Keep the adapter/backend, exact command, source
  manifest, and receipt hash with any claim. The Gate 6 Full-16 history
  candidate uses
  `gpu_full16_history_candidate_measures_temporal_gain_and_writes_the_receipt`,
  which regenerates the tracked
  `docs/evidence/full16-history-candidate-receipt.json` in place — the
  S2-receipt law applies: a changed receipt after an opt-in run is a new
  measurement, commit it. The hosted
  `full16_history_plan_charges_eight_bytes_per_temporal_pixel_and_discriminates`
  pins the candidate plan's 8-byte temporal class against the settled 4-byte
  one in both directions with one-byte-under rejection. The Gate 4 hardware
  decode backend uses
  `hw_decode_interop_probe_measures_agreement_and_writes_the_receipt`
  (Windows, a real D3D11VA device, `videos/audit.mp4`), which regenerates
  the tracked `docs/evidence/hw-decode-interop-receipt.json` in place under
  the same S2-receipt law; the hosted
  `the_production_probe_defers_every_capability_with_its_actionable_reason`
  pins the per-platform progression, including hardware decode stopping at
  exactly `EvaluationRequired(InteroperabilityProof)` on Windows.
- `s2-eight-texture-floor-receipt.json` is a tracked artifact that the probe in
  `tests/eight_texture_floor_probe.rs` regenerates in place. It is tracked
  because `MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS` and the Symmetry Field's
  single-pass shape cite it by name as their justification; deleting it would
  leave two source files pointing at evidence the repository does not carry.
  A changed receipt after an opt-in run is therefore **a new measurement on new
  hardware, not drift — commit it**. `claim_first_proven` is the frozen original
  proof and never moves; `measured_at` is resolved from git at run time, so a
  receipt always names the commit and branch that actually produced it, and
  honestly reports `unknown` in a tree with no git metadata. The one consequence
  to expect: running that opt-in probe dirties the working tree, so a gate whose
  evidence claims a clean tree must be run before it or the receipt committed
  after it. The probe commit `5a10b79` is reachable only through
  `probe/s2-eight-texture-floor`; keep that branch, or the receipt's own
  provenance and the `renderer/symmetry_field.rs` citation both dangle.
- `cargo test` covers pure protocol, persistence, modulation, morph, temporal,
  transport, audio, Spout lifecycle helpers, and export-argument behavior.
- `cargo test effects_audit -- --ignored` renders the labeled effect suite
  through the real export path. It requires a working GPU, FFmpeg on `PATH`,
  and `videos/audit.mp4`.
- Export repeatability requires two independent renders and equal decoded
  `framemd5` sequences.
- Advanced execution-order tests must cover a tapless two-layer stack whose ids
  ascend front-to-back scheduling with an empty retained map, a tapless stack
  containing a collapsed group whose members stay contiguous and back-to-front,
  and — because the tie-break is shared scheduling for every Advanced
  composition — paired `framemd5` equality across the change for every labeled
  export case. `render_tapless_advanced_motion_pipeline` is the labeled export
  case that could not be prepared before the composite rank landed.
- Browser QA must include desktop and narrow viewports, touch/pointer release,
  multi-controller stale-topology rejection, token and foreign-Origin denial,
  direct layer FPS/effects, reorder, beat latch, input configurations,
  media-safety rationale/confirmation/reconciliation, NTSC path metrics,
  patch/parameter-editor state, and export.
- Media-policy tests must cover Safe legacy acceptance/rejection, every Expert
  hard/device/aggregate bound, reservation release, missing host-memory data,
  non-persistence, and output/export-output non-expansion. Static panel tests
  must preserve accessible labels/descriptions and keep fast NTSC counters out
  of a live announcement region.
- Shift tests must preserve the amount-zero shader branch and uniform layout,
  old-seed Dice streams, bounded deterministic variation, patch defaults,
  Morph/modulation wiring, resets, static controls, and a labeled export case.
- Displace tests must cover the append-only kind code and frozen legacy rack
  signatures, sanitize/exact-bypass laws including hostile non-finite gains,
  the 1/3/12/2/1 descriptor ledger, the CPU reference with analytic ±X/±Y
  fixtures for all four boundaries, a transparent hostile-hidden-RGB donor
  decoding to exact zero, planner admission and current-frame self-cycle
  rejection with an admitted N-1 edge, saved-patch dormant-versus-woken edges,
  tombstones that never rebind after replacement, Morph route-match
  interpolation with endpoint-exact boundary, values-only Look/preset apply,
  Dice/generator gain-only mutation, `amount_x`/`amount_y` modulation
  addresses, the uncoalesced revision-barriered browser route action, and a
  labeled export case. The two `renderer::rack::tests::gpu_displace_` fixtures
  carry the physical-GPU claim.
- Symmetry Field tests must cover group closure and the identity seam for every
  one of the eight modes, analytic sector fixtures for the radial and planar
  families, all five boundary laws with CellularReentry's non-recursion proven
  structurally, `effective_folds` as the only rounding point under fractional
  modulation, the 32-record table's stability by owner/node/seed with every
  record bit-identical after a donor is lost, a table domain that carries no
  process-lifetime layer ID so live and offline agree for the same authored
  patch, history ages accepted at 23 and
  rejected at 24, the exact-default bypass, the 1,024-byte compile-time size
  assertion, the dedicated ledger admitted at eight and refused at nine while
  ordinary rack segments stay capped at three, per-slot planner admission and
  tombstones that never rebind, a motion donor yielding a field at exactly zero
  Motion, an armed motion slot resolving to that donor's `admitted_field_slot`,
  two image slots planning two distinct current-frame PreLocal donors, the full
  patch/Morph/modulation/Dice/generator/preset/browser/native closure, per-slot
  export provenance, and a labeled export case. Fixtures must
  raise `input.resource_limits` explicitly, because
  `CreativeResourceLimits::default()` reports the ordinary three-texture
  constant. The six `renderer::symmetry_field::tests::gpu_symmetry_field_`
  fixtures — including the one proving each motion slot's committed parity is
  selected independently and that an unmaterialized parity displaces by exactly
  nothing — plus the four `renderer::composition::tests::`
  `production_symmetry_field_` fixtures — in-place execution with warm-frame
  allocation invariance, the authored motion donor reaching a prepared motion
  bind group, the typed refusal of two current PreLocal donors on one node, and
  `production_symmetry_field_authored_motion_route_reaches_the_pixels`, which
  drives a known uniform codec field through a donor whose own Motion is exactly
  zero and proves in one fixture that the frame moves against both a stationary
  field and an unarmed slot, that live and export layer identities render the
  frame byte-identically, that an unarmed or unmaterialized slot is
  byte-identical to the neutral pair, and that the warm-allocation snapshot
  survives eight prebuilt groups — carry the physical-GPU claim.
- Residual Counterpoint tests must cover the append-only kind code 11 and the
  two closed vocabularies, sanitize/exact-bypass laws including hostile
  non-finite `mix`, the 1/2/3/12/3/2/2 descriptor ledger, the independent CPU
  reference for the 4-tap block mean and the seeded lattice, constant-colour
  (pure DC) and zero-mean (pure AC) fixtures, grid-border and non-divisible
  dimensions, a transparent hostile-hidden-RGB donor decoding to exact zero,
  `Off` as exact identity, fixed-seed stability with no leak into the route
  table or the block grid, every independent resource limit rejected one unit
  over with the byte cap proven to bind before the cell cap, per-slot planner
  admission, self-cycle rejection with an admitted N-1 edge, per-slot
  tombstones that never rebind after replacement, saved-patch
  dormant-versus-woken edges, Morph both-slots route-match interpolation,
  values-only Look/preset apply, Dice/generator value-only mutation,
  `mix`/`detail_gain` modulation addresses, the uncoalesced
  revision-barriered slot-naming browser route action, `ResetVisualProgram`
  releasing every mean surface, bounded path-free export provenance per slot,
  live/export payload parity at 24/30/60 fps, and a labeled export case. The
  three `renderer::rack::tests::gpu_residual_` fixtures carry the
  physical-GPU claim.
- Gesture-field tests must cover the canonical checksum's domain-separated
  field stream and both recording flags, hostile serde bounds (over-cap counts
  and bytes on encode and decode, unknown fields, non-monotonic ticks, orphan
  `Move`/`End`, a second `Begin`), Q16/Q15 lattice round trips, identical
  grouped-versus-ungrouped reference-tick replay producing one canvas and one
  checksum, analytic Push/Curl and falloff fixtures, overlapping strokes whose
  reordering visibly changes the field, canvas-edge and corner samples, decay
  and hold with the tick-budget clamp, commit/discard/freeze/over-cap and every
  typed reset cause with its exact domains, the portable sidecar round trip
  with no path or filesystem metadata and a no-replace re-export, a checksum
  mismatch refused before any frame renders, live/export field equality at
  24/30/60 fps, one completed authored gesture as exactly one undo entry with
  automation origins excluded, every independent resource limit rejected one
  unit over, and the honesty law that live-only samples are counted and
  reported separately from the recorded track. The same logical gesture driven
  through all four `GestureOrigin` values must record byte-identical tracks.
  The five `renderer::gesture_canvas::tests::gpu_` fixtures and
  `renderer::composition::tests::gpu_a_recorded_gesture_reaches_the_image_through_a_routed_displace_donor`
  carry the physical-GPU claim; `render_gesture_field_etching_pipeline` and
  `render_gesture_canvas_displace_donor_pipeline` are the labeled export cases.
- Field Collider tests must cover the two closed vocabularies and the boundary
  codes shared with `DisplaceBoundary`/`SymmetryBoundary`, every mode against
  its analytic definition including both degenerate guards, the clamp into the
  canonical `pack_velocity`/`unpack_velocity` range, componentwise confidence
  and visibility minima with no pre-gating, the exact zero sample for a
  missing/NaN/out-of-range/aliased input that never reuses its partner, all four
  boundary laws with NaN removed by every one of them, the complete admission
  law refusing alias/missing/unselected/master/no-transplant and permitting an
  input that names its own recipient, both inputs admitted at exactly zero
  Motion through `required_as_donor` and resolved through
  `admitted_field_slot`, the exact 20-byte/cell ledger with one-unit-over
  rejection on every independent bound and the one-collider cap, the 144-byte
  compile-time uniform assertion, the coordinate-versus-vector transform law
  with translation excluded from vectors and a singular pair yielding no map,
  the transactional commit/discard/Program-Freeze/reset lifecycle with no stale
  derived field, a checksummed recovery-journal PatchState round trip carrying
  no pixels or paths, per-slot export provenance, and the full
  patch/Look/Morph/modulation/Dice/generator/browser closure. The
  `renderer::composition::tests::production_field_collider_derived_field_reaches_the_pixels`
  fixture carries the physical-GPU claim — the derived field advecting the
  carrier into the audience image, both inputs contributing, live/export
  identity parity, warm-allocation invariance, and byte-identical exact M4 when
  disabled — and `render_field_collider_pipeline` is the labeled export case.
- B2 procedural-field tests must cover the closed kind vocabulary with its
  permanent codes and the single `source_key` token table, resolution at every
  scope with no codec dependency and no fallback, neutral (never
  clamped-extreme) sanitize for both scalars, the four pure kinds proven never
  to consult the image (a panicking closure) with fully open gates, the
  analytic Radial/Spiral/Weave fixtures, numeric divergence-freedom for Curl at
  base frequency, Contour's perpendicular-to-gradient law with gradient
  confidence and flat content contributing nothing, Chroma's alpha-covered
  neutrality (zero coverage steers nothing), clamping into the canonical
  velocity range with non-finite time taking neutral zero, zero luma bytes in
  preflight beside byte-identical vector/gate charges, the append-only origin
  signature codes 4–9, a Collider planner fixture with one procedural and one
  codec input, the patch round trip with absent-section byte identity, Morph
  scalar interpolation with the kind switching at the midpoint, the two
  modulation addresses at master and layer scope, ingress classification
  (kind = MemoryTopology, scalars = ValuesOnly) with a bare `procedural` token
  refused, and the wire vocabulary in `valid_motion_edit`. Flow-shaping tests
  must additionally cover neutral sanitize with a block size alone shaping
  nothing, the analytic stretch and saturated/linear repel fixtures,
  deterministic trash firing against the exact hash law with its probability
  gate honest over many cells, canonical-range clamping under hostile inputs,
  the four-taps-only-when-active pass budget, and the shaping closure ladder.
  `gpu_procedural_field_matches_the_cpu_reference_for_every_kind` carries the
  physical-GPU claim; `render_procedural_motion_field_pipeline` and
  `render_motion_flow_shaping_pipeline` (whose `_unshaped` twin must decode
  differently) are the labeled export cases.
- Feedback-rig tests must cover the identity/exactness laws (authored identity
  keeps the activity flag closed, the legacy 64-byte golden untouched, the
  96-byte rig uniform assertion), neutral sanitize with a reflection alone
  refusing identity, the complete rate law (linear halves, multiplicative
  square roots, authored nonlinear values beside the clamped tick fraction),
  the shape codes 0–3, the edge laws on the frozen boundary numbering, the
  regime fixtures (four-arm lock with analytic retention powers, detune shear,
  the reflection two-cycle, servo bound versus monotonic defeated runaway),
  the uniform lanes (epoch low bits, defeat-wins servo strength, reflect/
  shape/edge codes), the patch round trip with absent-section byte identity,
  Morph blend/wrapped-hue/discrete-recall, the fourteen modulation addresses
  with the six discrete laws refused, the eighteen-param wire vocabulary in
  both validators, generator mutation in fresh domains preserving discrete
  laws, and the shader-contract counts (one shared legacy carrier sample plus
  the rig's seven gated taps; rig binding 2 in both shaders). The re-pinned
  `temporal.wgsl` SHA and M6 shader-bundle digest are deliberate B3
  re-measurements. `render_feedback_rig_pipeline` is the labeled export case
  with its `_unrigged` difference and `_repeat` determinism assertions.
- Time-displace tests must cover the closed map codes 0–4 with `Ramp` the
  default, the analytic per-map fixtures (Ramp equal to the legacy dot law,
  Brightness as clamped covered-luma passthrough, aspect-correct Radial with
  its 1.6 reach and corner clamp, TbcRamp's 8-scanline sawtooth constant in
  x, Sweep's wrap and phase travel) with hostile inputs staying inside the
  unit coordinate, the deterministic 600-tick sweep phase, the
  unwritten-history depth-clamp sweep over every validity count for both the
  floor and interpolated laws, the plan fixture proving the originals shader
  is selected only off the exact Ramp/floor path with the reserved-lane
  assignments (and the sweep lane zero for every other map at nonzero
  ticks), the patch round trip with absent-section byte identity and unknown
  tokens rejected, Morph endpoint recall at the midpoint, the wire vocabulary
  in both validators, and the shader-contract counts (legacy-prefix inline
  samples at 11, the interpolation toggle and valid-history clamp present in
  both variants). `temporal.wgsl` is deliberately untouched — its pinned SHA
  is a B12 non-measurement — and `render_time_displace_pipeline` is the
  labeled export case with its `_ramp` difference and `_repeat` determinism
  assertions.
- Small-effects tests must cover the eighteen-vec4 layout assertions (with
  the spatial slots moved to byte 288 and the pass at 352), the per-effect
  amount-gate source audit with the multi-grid `>= 1.5` law and the WGSL
  field order between `shift_speed` and the spatial rows, exact-off defaults
  with reset coverage, the master-only law at every seam
  (`clear_master_only_effects` unit coverage, Look application clearing the
  optics, `master_only_effect_param` beside `valid_effect_edit`'s closed
  vocabulary with `negative_mode` capped at 2, layer optics absent from
  modulation and never generator-mutated), the patch round trip with
  skip-at-default omission and neutral hostile sanitize, the snapshot round
  trip with a pre-B13 JSON decoding to the exact prior path, Morph blending
  values with wrapped hue arcs and midpoint recall of the negative mode, the
  modulation targets resolving at both scopes with `morph` still last and
  bases immutable, the pinned pre-B13 Dice golden (established streams
  byte-stable beside the fresh domain-separated small-effects stream), the
  generator's fresh per-scope domain with temperature-zero no-op and rounded
  integer counts, and the bumped 148-slider range contract. The re-pinned M6
  shader-bundle digest is a deliberate B13 re-measurement whose six output
  SHAs must not move. `render_small_effects_pipeline` is the labeled export
  case with its `_plain` difference and `_repeat` determinism assertions.
- Preview transform-gizmo tests must cover the pane/output/local round trip at
  multiple aspect ratios and DPI scales including a non-square output, an
  active crop, a nonzero shear and a letterboxed pane; the forward map agreeing
  with the canonical inverse; anchor-only inertness proven exactly rather than
  by example, with the non-inert cases matching the closed form; a singular and
  a non-finite transform failing hit-testing closed with no identity fallback;
  hit testing leaving `spatial_modes.w` at zero; handle pick priority; each
  drag law with its Shift and Alt variants and the zero-lever-arm refusal;
  every authored value landing inside the spatial contract with a non-finite
  computation taking the neutral value; the bounded allocation-free edit set;
  one drag as exactly one undo entry and a no-op drag as none; Escape
  cancelling before the first commit and undoing after it, and consuming
  nothing outside a drag; a topology bump aborting an open drag without
  retargeting and a stale Move afterwards authoring nothing; the nudge step law
  and its refusal under an open drag; the permit refused for every
  audience-facing surface and for a single-monitor Output, agreeing with
  `show_transform_gizmo` and with every other native-control predicate; and the
  source audit pinning the permit's single sealed declaration and construction.
  `render_native_gizmo_transform_pipeline` is the labeled export case: it
  renders a gizmo-authored transform, its numerically-authored twin, and the
  untouched identity, and the claim is that the first two are decoded-frame
  identical while both differ from the third. The gizmo introduces no export
  path, so the six pre-existing labeled cases must additionally be proven
  `framemd5`-identical across the tranche by a same-branch A/B.
- Scan Processor tests must cover the append-only kind code 14 with the
  dedicated list `[Symmetry, Study, ScanProcessor]`, the 19-row descriptor
  contract (modulatable ⇔ Float ⇔ Dice-eligible; geometry Unsigned;
  reversals Bool), the analytic beam-law fixtures (flat-raster identity,
  luma pivot/span, read-side reversal, collapse, locked-oscillator standing
  pattern and whole-multiple quantization, detuned crawl, the `2/speed`
  clamp with its mix, central-difference speed with its floor, fail-safe
  ribbon normal, pixel width law, colorize with black-stays-black,
  tilt-authors-while-perspective-alone-does-not), hostile neutral sanitize,
  node serde with unknown-field rejection, the default-bypass wake law, the
  planner lift at the authored position with segmentation resuming behind
  it, the derived dedicated ledger (2 passes / 2 textures / 128 uniform
  bytes / summed vertices / 8 B-px transient) with the structural vertex
  cap, a topology signature that discriminates presence but is invariant to
  geometry edits, the fifteen stable modulation addresses with
  geometry/reversals unreachable, the Morph blend with midpoint recall of
  the discrete class, Dice value-only mutation with neighbour invariance,
  and the snapshot's derived wake law. The production GPU fixture
  `production_scan_processor_density_exceeds_any_displacement_and_default_is_bypass`
  carries the physical-GPU claim — with the beam-energy law disengaged, a
  collapsed raster's additive line density exceeds twice the flat source's
  maximum (impossible for any single-sample displacement), the authored
  default is byte-identical to no node, warm frames allocate nothing —
  and `render_scan_processor_pipeline` is the labeled export case with its
  `_bypass` difference and `_repeat` determinism assertions.
- Display-physics tests must cover the exact-off wake law per sub-block
  (dressing controls wake nothing alone), hostile neutral sanitize, the
  per-tick field parity with the order fault, the 3:2 film clock, the
  weave/bob/blend laws with amount-zero passthrough, twitter's per-field
  sign, the phosphor max law against closed-form trails with the P22
  ordering and the fractional-tick rate law, the mask families, the
  Lottes beam profile, sag/bloom/halation extraction, the mono/green
  tints, the 128-byte uniform assertion, the patch round trip with
  skip-at-default omission and unknown-token rejection, the Morph blend
  with midpoint recall of the three discrete laws, the seventeen
  modulation addresses with the discrete laws refused, the twenty-param
  wire vocabulary in both validators, and generator mutation in fresh
  domains preserving discrete laws. The frozen `temporal.wgsl` SHA and
  temporal goldens must not move — the stage owns its own shader.
  `gpu_display_physics_follows_the_cpu_laws_and_blackout_clears_the_wake`
  carries the physical-GPU claim (dormant delegation, the real two-moment
  comb, closed-form decay, blackout clearing the wake), and
  `render_display_physics_pipeline` is the labeled export case with its
  `_flat` difference and `_repeat` determinism assertions.
- Sync-latch tests must cover the exact-off default and the wake law
  (neither control alone wakes the stage, and the switch alone never wakes an
  inert one), hostile neutral sanitize with out-of-range clamping, the band
  height span with a band's lines carrying one identical offset, the firing
  law at zero and full rate, bias forcing the slip sign while keeping its
  magnitude, every slip inside the declared bound, unlatched slips healing
  with their own tick, a frame inside a tick holding the shear still,
  **latching accumulating monotonically and stopping at the cap**, **release
  unwinding the whole table in one step** with damage surviving a magnitude
  pulled to zero, the 24-tick burst clamp, determinism per seed and
  distinctness across seeds, the line cap and hard clear, the wrap law
  keeping every sample inside the frame, the frozen 16/8,656-byte uniform
  sizes, distinct hash lanes, the patch round trip with the accumulated table
  proven absent from the YAML, the Morph blend with midpoint recall of the
  switch, the four modulation addresses with the switch refused, generator
  preservation with no mutator in the source, both wire validators, and the
  B9 value-law oracle. The two `renderer::sync_latch::tests::gpu_sync_latch_`
  fixtures carry the physical-GPU claim — each line sheared to where the CPU
  reference says (compared against a model reproducing the filtering
  sampler's wrapped bilinear tap), the authored default encoding no pass at
  all, and reset clearing only the causes that begin a new program while a
  source cut, a seek, and a blackout transition all keep the damage.
  `render_sync_latch_pipeline` is the labeled export case with its `_healed`
  difference twin, its `_off` dormant twin, and its `_repeat` determinism
  assertion.
- Mixing-boundary tests must cover the 25 append-only blend codes with rows
  0..=14 of the frozen vector tables byte-identical and the code-byte law
  for the bitwise pair, the closed wipe/back-colour vocabularies, the
  analytic wipe landmarks with exact fader endpoints and MULTI tiling, the
  border band profile with its end gates, hostile neutral sanitize for all
  three param blocks, the wake laws (dressing controls wake nothing alone;
  melt-armed needs both melt and hold), the event clock's bit-clean quiet
  ticks with the analytic envelope and honest dropout probability, per-seed
  determinism with independent hash lanes, the melt band/normal/swirl/
  creep/cap laws with the vertical-edge fixture and
  no-boundary-nothing-happens, the coherent 601 YIQ round trip, the shared
  `BusMixerEdit` parse/apply table with typed rejections (`alpha_cut`
  refused at the bus), composition serde with the default tree omitting the
  mixer block and resolve/capture carrying it, morph blending with midpoint
  recall of every discrete law, the modulation address tables with discrete
  refusals at both scopes, and Dice bounds in the fresh domains. The
  `gpu_all_blend_modes_…` compositor parity fixture,
  `gpu_melting_edge_drags_the_band_holds_history_and_needs_a_boundary`, and
  the re-pinned M6 receipt (six output SHAs unmoved across the bus rewrite
  and the effect-block growth) carry the physical-GPU claim;
  `render_bus_mixing_boundary_pipeline` and
  `render_melting_edge_and_key_dressing_pipeline` are the labeled export
  cases, each with a difference twin and a `_repeat` determinism assertion.
- Program re-entry tests must cover the closed vocabulary (serde tag
  `program_tap` with a near-miss rejected, plan hash code 8 append-only, the
  fixed-point saved/runtime round trip with no positional accessor and no
  tombstone), the planner law (a routed tap at both timings claims no
  dependency, no ordering edge, no staging, and no ledger surface, with the
  bare topological order unchanged and both scopes' resource numbers
  identical; unavailable resolves `Transparent` with the named
  `ProgramTapUnavailable` diagnostic and never rebinds), the fail-closed
  ledger reconcile at zero with one-over refusal, the saved-graph walker
  claiming no edge dormant or woken through a YAML round trip, ingress
  acceptance at both timings with the panel strings asserted in `app.js`,
  the live source-order law (bind before prepare; publish only after
  `commit_temporal_frame` under the `temporal_frame_accepted &&
  !self.blackout` gate; the blackout clear downstream of the copy; all three
  in-loop plan constructions consulting `program_tap_valid`; the
  patch-generation barrier invalidating the tap), the offline source-order
  mirror (job-lifetime surface, bind before prepare, publish after the
  ffmpeg write, the copy reading slot 2, unconditional admission), and the
  renderer texture floor at exactly 30 with its re-pinned byte literals.
  `gpu_a_program_tap_donor_feeds_the_previous_frame_back_through_a_routed_displace`
  carries the physical-GPU claim (never-published equals unbound by
  arithmetic, a published copy demonstrably reaches the pixels, rebinding
  under a new epoch re-prepares), and `render_program_reentry_pipeline` is
  the labeled export case with its `_untapped` difference twin inside the
  same Advanced plan family and its `_repeat` determinism assertion.
- Codec-mosh tests split along the codec boundary. Hosted (no codec
  touched): hostile neutral sanitize, the amount-alone wake law at the
  0.003 deadband, the transcribed bitrate map with its ±25% hysteresis, the
  resync period table with zero-never-recovers, the encode-dimension law,
  per-lane deterministic fault dice, the key bootstrap always passing with
  later keys facing the dice (and forced resync keys too), delta decisions
  honoring the probability gates with drop's early return and `rate`
  exempting key removal, the chunk ring bounded in entries AND bytes with
  FIFO eviction and the newest-six shield, the worker's
  one-in-flight/drop-new/slot-release-on-error ladder (tolerant of a host
  without the codec pair), the patch round trip with skip-at-default
  omission and unknown-field rejection, the Morph blend with midpoint
  recall of the recycle law, the eight modulation addresses with the
  recycle law refused and the offset landing in the frame copy only, the
  nine-param wire vocabulary in both validators, generator mutation in
  fresh domains preserving the recycle law, the extended
  `raw_audience_readback_required` law (armed mosh requires the readback on
  every path, never through a hold or blackout), and the sidecar schema-6
  pins with the section absent from a moshless job. Opt-in (`--ignored`,
  the host FFmpeg's mpeg4 pair required):
  `mosh_round_trip_is_deterministic_per_host_and_reaches_the_pixels` (two
  runs byte-identical, the mosh reaches the pixels, amount zero is a
  no-touch bypass even on a warm engine), and `render_codec_mosh_pipeline`
  is the labeled export case with its `_clean` difference, `_repeat`
  per-host determinism, and sidecar encoder-identity assertions.
  Cross-machine bit-identity is deliberately not asserted anywhere.
- Performance-recorder tests must cover the portable contract (the Q16/
  discrete/toggle/stepped lattices with hostile refusals, first-law interning,
  the checksum covering the address table with both honesty flags inside the
  digest, bounded serde both ways including hostile extra fields inside one
  address, the monotonic cursor, the derive-before-add clock, event-cap
  truncation, and frame-grouping digest invariance), the value-law oracle
  answering from the engines' own tables (blend and bus vocabularies from
  `ALL` with `alpha_cut` excluded, master-only optics refusing layer scope,
  seeds and the loop driver refused by name), the drain tap's
  capture/commit/skip-and-count behavior, revision guards with mutual
  transport exclusion, replay through the inner seam with undo exclusion and
  loop rewind, per-address degradation by name that never retargets, the
  generation-barrier/source-cut asymmetry with restore-after-barrier source
  pins, checksum-gated patch carriage with incomplete-capture marking,
  never-latchable refusals at the server gate and the engine latch, the
  shared-applier source audits (one `apply_temporal_wire_edit`, consumed by
  both the live arm and export replay), the sidecar's no-replace/
  claim-retirement/cleanup laws, and the offline applier for every family
  with stale-position and absent-morph no-ops.
  `render_performance_recorder_pipeline` is the labeled export case with its
  `_untaken` difference twin and `_repeat` determinism assertion.
- Mod-source (B10) tests must cover the closed source and trigger
  vocabularies, every envelope law against its closed form (attack timing,
  exponential decay, gate hold, loop re-fire, retrigger resuming from the
  current level, the beat-crossing anchor law), chaos determinism per seed
  with bounds, spike tick-addressing with frame-rate invariance, the
  analytic drift sum, bend ramp asymmetry with out-of-range refusal and
  reset release, macro clamping, the video-analysis law (first-frame
  honesty, cut onset/decay, no-source decay, hostile input), the reduction
  reference's flat-field exactness and gradient monotonicity, the
  arm-on-demand predicate, the ModConfig round trip with skip-at-default
  omission and hostile sanitize, both gates' vocabularies with the
  never-latchable bend edge, the panel contracts, `map_bend_key` with
  `map_key`'s release-is-inert law intact, and reseed determinism through
  the real action door.
  `gpu_video_analysis_reduction_matches_the_cpu_reference` carries the
  physical-GPU claim (the B7 statistical contract: ≥95% of grid bytes
  within 4 code values, plus the two-slot saturation drop), and
  `render_mod_sources_pipeline` is the labeled export case with its
  `_unrouted` difference twin and `_repeat` determinism assertion.
- Corruption-trio (B6) tests must cover the append-only kind codes 15/16/17
  with the six-kind dedicated list, the per-kind param surface (modulatable
  ⇔ Float ⇔ Dice-eligible; the avalanche axis refused everywhere), the
  tagged-serde round trips with unknown-field/token rejection and neutral
  hostile sanitize, the DCT's forward∘inverse identity with the quantiser
  disengaged plus the floored/tilted/monotonic step law, the
  coefficient-domain chroma crush with grey invariance, flat-block DC
  stability and coarse-quantiser ringing, the BENDR block-edge map, the
  sort's identity/inheritance/64-row-reach/edge-clamp laws, the avalanche's
  frozen axis codes, span and `floor(t·3)` epoch laws, deterministic
  per-key gates with the honest amount/2 firing fraction, flat-history
  DC-only movement, warm-versus-cold history discrimination, the kind-only
  planner lift with per-node steps at authored positions and inactive
  bypassed steps, the re-derived combined ledger (shared DCT transient
  charged once; per-node avalanche histories; the honest 17+34+65 tap sum),
  and the four-admitted/five-refused avalanche cap.
  `gpu_corruption_trio_matches_the_cpu_references` carries the physical-GPU
  claim (all three laws under the B7 statistical contract; the avalanche
  proven cold and warm with the history binding demonstrably
  participating), and `render_block_dct_pipeline`,
  `render_pixel_sort_pipeline`, and `render_filter_avalanche_pipeline` are
  the labeled export cases with `_clean` difference twins and `_repeat`
  determinism assertions.
- Monitoring-bay (B11) tests must cover the Rec.601/vectorscope projection
  fixtures with the six targets derived from the projection (complementary
  bars mirrored through the centre), the flat-grey field reducing to one
  saturated waveform row and one centred scope point, the two-tone split,
  hostile short grids drawing nothing, the RFC 4648 base64 vectors, the
  closed probe vocabulary with near-miss rejection, snapshot default
  inactivity with fresh-sample publication and the additive back-compat
  strip, the wire names/coalesce keys/history classification, the watch
  registry's arm/disarm/disconnect lifecycle, the live panel strings
  asserted in `app.js`/`index.html`, the 45-entry
  `MONITOR_SOURCE_LIST` round trip with pure double reads, the permit
  refused for every audience surface / a single-monitor audience output / a
  disabled bay, and the source audit pinning exactly one sealed permit
  constructor. `gpu_monitor_bay_reduction_matches_the_cpu_reference`
  carries the physical-GPU claim (the B7 statistical contract at 128×72
  plus the two-slot saturation drop). The bay has no export arm, so its
  exactness claim is the pinned-worktree `framemd5` A/B of a pre-existing
  labeled case, not a labeled case of its own.
- Proxy-worker tests split along the CLI boundary. Hosted (all three CI
  platforms, no ffmpeg CLI): the crash test written reproduction-first — a
  staging leftover removed and never published or counted, an unsealed
  artifact and an orphan seal both removed as interrupted publications; the
  atomic publish law with the prior artifact readable until replacement and
  the seal following the artifact; mid-file corruption refused by the seal
  and discarded; eviction following the pure plan with a path-free receipt;
  the advisory cross-session recency record — the mid-write reproduction
  recovered beside a healthy sealed cache with the prior record applied in
  full, cross-session eviction order surviving a reopen and reaching the
  pure preflight, every hostile record shape (torn, wrong version, unknown
  field, malformed or duplicate key, oversized) discarded whole without
  refusing the cache, ghost rows resurrecting nothing and never advancing
  the counter, and removals rewriting the record;
  foreign files counted but never touched; the contract-derived argv; garbage
  bytes failing decoded-identity validation; mutated/unreadable sources
  refused before any encode; the Y-key mapping; hot adoption's
  CLI-free half — an empty cache and a refused consultation each producing
  one named, job-level `ProxyAdoptionEvent::Refused` through exactly the
  patch-load consultation law, with no per-layer preparation fabricated;
  and identity minting's CLI-free half — the mint matching the fingerprint
  law byte-for-byte with unreadable sources typed-refused, and the worker
  reporting `IdentityMinted` with its claim passed through verbatim before
  any encode outcome, an unreadable source yielding one layer-keyed
  `MintFailed` with nothing started under a fabricated identity.
  Opt-in (`--ignored`, ffmpeg CLI required, like `effects_audit`):
  `proxy_worker_end_to_end_encode_publish_rename_and_corruption_survival` —
  encode, validate, publish, cache hit, identical bytes at a renamed path
  hitting the same key and adopting, corruption refused at consultation and
  at the job's own cache-hit path, crash recovery beside a live cache, and
  both audio laws — plus
  `proxy_encode_kill_bounds_are_typed_and_publish_nothing` for the deadline
  and size-cap kills,
  `proxy_hot_adoption_prepares_playhead_seeded_decoders_end_to_end` — two
  candidates at different playheads each receiving their own half-scale
  decoder whose seed frames demonstrably differ, claims passed through
  verbatim — and (GPU adapter additionally required)
  `gpu_proxy_hot_adoption_swaps_a_live_layer_and_keeps_identity_and_playhead`
  — the infallible `commit_adopted_proxy` swap keeps identity, filename, and
  playhead while moving decoder, texture, dimensions, and runtime path, and
  advances the source-resource epoch — and
  `proxy_identity_mint_end_to_end_encodes_and_hits_the_cache_under_the_minted_key`
  — a real source with no retained identity walks mint → encode →
  publication under the minted identity, and the same bytes again are a
  cache hit under the same key. Windows fsync law: `FlushFileBuffers`
  demands writable handles for both the staging file and the parent
  directory; do not "fix" a publish failure by dropping either sync.
- Study evaluator tests are all hosted CPU: every arithmetic opcode against
  analytic expectations, the rack hue law with its exact HSL fixtures, the
  R1 validity guard mirroring `temporal_originals.wgsl` (nothing committed →
  the virtual current image, deep requests clamped to the oldest valid
  layer), the R2 randomness as a document constant with independent domains
  and frame-context invariance, the bound law discriminated from an
  unbounded evaluator, frame-input sanitization to documented neutrals,
  Vector2 evaluated honestly as the recorded dead end, compile-time
  required-age listing, compile-refuses-invalid, and the R3 backward-minor
  window (newer minor and other majors rejected). The S10b interpreter adds:
  the frozen GPU encoding golden (append-only opcodes 0…15, aux in the high
  half-word, resolved randomness as an immediate so the GPU never hashes,
  zeroed tail slots so one 8,192-byte buffer serves every study), the
  semantics-version-2 hue unorm-input clamp, and the source-text law that
  `study_interpreter.wgsl` shares `rack_node.wgsl`'s three hue functions
  character for character. The two
  `renderer::study::tests::gpu_study_` fixtures carry the physical-GPU
  claim: the interpreter matches the CPU reference across every opcode at
  2e-5 with the R1 guard observably discriminating a young ring from a
  committed one, and a study swap is two writes into fixed buffers with a
  deterministic re-render. The authored surface (S11) adds: the hex-digest
  serde and exact-default-bypass laws, the planner's flush-lift-resume with
  a kind-only dormant position, the re-derived dedicated ledger, digest
  resolution and the identity hash as plan-visible topology, the
  `StudyLoadBudget` refusal at nine loads with eight admitted exactly, the
  patch `studies` round trip and digest walk, the Morph midpoint endpoint
  recall, and the coalescible document action with its panel wiring. The
  pixels claim is
  `production_study_field_reaches_the_pixels_and_unresolved_digests_are_inert`
  (resolved reaches the audience image; an unresolved digest is
  byte-identical to no node; warm frames allocate nothing), and
  `render_study_field_pipeline` is the labeled export case.
- Generator-source tests split along the GPU boundary. Hosted: the pattern
  transcription fixtures (oscillator waveforms including the S&H cell law,
  radial symmetry without cross-mod, Scan separability, the hard comparator
  on constructed pre-comparator signals, wavefolder range and reach,
  colouriser laws, centre invariance under zoom/rotate, hostile neutral
  sanitize, the frozen shape/wave/colour code tables, the 128-byte uniform
  assertion with hostile-time neutralization); the text raster fixtures
  (opaque page, byte-identical re-raster, glyphs landing at the anchor, the
  two faces differing, shape-fan fill/stroke laws with rings always
  stroked, repeat and outline reach, body-cap truncation on a character
  boundary, frozen code tables); the patch round trip with absent-section
  byte identity, neutral hostile sanitize, and unknown-token/field
  rejection; the wire accept/reject battery through the shared parse
  tables plus the `add_pattern_layer`/`add_text_layer` spellings; the
  modulation battery (all 22 ranges, discrete refusals, suffix indices
  pinned at 94..=115 with the table at 116, offsets landing in the frame
  copy and clamping, dormant identity); and the Morph battery (blend,
  wrapped hue crossing the unit seam, midpoint recall of all three
  vocabularies, exact endpoints, the both-slots gate and appended-layer
  editability). Opt-in:
  `gpu_pattern_synth_matches_the_cpu_reference_for_every_shape` (all 12
  shapes x 5 colourisers on a 144-point grid, >=95% of channel samples
  within four sRGB code values — statistical because BENDR's screen hash
  amplifies single ulps at isolated pixels — opaque pages, deterministic
  double render) carries the physical-GPU claim, and
  `render_pattern_synth_pipeline` / `render_text_page_pipeline` are the
  labeled export cases: each patch's only layer is a generator, so the
  render succeeding with no file anywhere is the self-containment proof,
  with difference twins and `_repeat` determinism assertions.
- Spatial tests must cover the exact inactive identity, Transparent exposure,
  explicit Clamp, 4:3 Fit/Fill/Native landmarks, source-space anchor behavior,
  aspect-correct rotation/skew, crop/hostile inputs, every edge/sampling mode,
  patch/Look/Morph/modulation/Dice/procedural round trips, stable-ID ingress,
  selective-VHS compatibility, live/export frame-rate parity, and at least one
  real GPU reference/golden path.
- Source-identity tests must cover cross-root canonical equality, changed-byte
  inequality, digest-enforcing load/export resolution, fingerprint/search
  budgets and cancellation, v1 manifest compatibility, private-path absence,
  and three-file atomic no-overwrite generation.
- Windows Spout requires real sender and receiver applications; use
  `spout_probe` for the output side.

Keep hardware deferrals explicit. Do not convert an unrun checklist item into
a passing claim.

## Known constraints

- FFmpeg library/runtime major must remain 8 unless the Rust dependency changes
  with it.
- Bindgen on Windows needs LLVM and the Visual C++ include environment.
- The `block` crate may emit an upstream future-incompatibility warning.
- Spout is Windows-only and live Spout input becomes deterministic black
  offline.
- Audio exposes 3–8 routable bands; the compact 32-bin spectrum remains
  display-only telemetry.
- Portable wgpu exposes no live/free-VRAM budget. Expert media status reports a
  conservative host plan and capability limits, not detected VRAM headroom.
- Procedural generation emits patches/manifests/preflight receipts only; MP4
  batch rendering, clip-statistics curation, and visual-driven audio DSP remain
  explicit deferred/research boundaries.
- The proxy loop is closed for content-referenced video, with two honest
  edges. `proxy_worker.rs` executes the `plan_proxy_input` contract: the Y
  key requests a bounded FFV1/Matroska encode (single helper, absolute
  deadline, staging-size kill, `MediaSafetyPolicy` reservation held for the
  encode, source re-fingerprinted first), publication follows the atomic
  commit law with a SHA-256 seal published after the artifact, recovery
  removes staging and unsealed residue without ever serving it, and both
  patch load and publication-time hot adoption consult the cache — a
  validated artifact backs the decoder while the layer keeps the original
  identity, so a proxy can never enter a patch, an export, or Dice. Hot
  adoption closed the former reapply-the-patch edge: an encode completion
  captures per-layer claims, the adoption worker prepares a playhead-seeded
  decoder off the render thread, and the drain installs it only after every
  claim re-validates (see the threading section). The browser surface closed
  the former native-only edge: the layer card's Encode proxy control drives
  the same engine ladder the Y key uses (see the web section). Identity
  minting closed the last edge: a request on a path-based video layer
  fingerprints the source through the same bounded machinery (mint mode),
  lands the identity behind claim guards, and — the operator's S9 ruling —
  enters it into persistence, so the next patch capture emits the content
  reference exactly as generation would have. Settings beyond the default
  are now authored host-session state through the one-owner law in the web
  section: each tuple keys its own cache entry, a load consults under the
  current session tuple only, and the default tuple remains the process
  start. Eviction recency survives sessions through the cache directory's
  single advisory `recency.json`, written by the artifact publication's own
  staged atomic replace: it orders eviction and nothing else, a missing or
  hostile record degrades whole to session-local order, a row naming an
  absent key resurrects nothing, and no record can refuse the cache or
  bypass a seal — consumption re-hashes regardless. Spout layers still cannot be
  proxied (no file bytes exist), which is a category fact rather than an
  edge. A host killed
  mid-encode may orphan one ffmpeg process bounded by its own completion;
  the staged file it writes is recovery residue, never an artifact. The
  Unix CI FFmpeg build carries `--disable-programs`, so end-to-end encode
  fixtures are opt-in like `effects_audit` and hosted CI proves the
  CLI-free cache half only.
- Hardware decode now has a backend in the tree, and it is
  **evaluation-only**: `video::hw_decode` is a Windows/D3D11VA session that
  decodes through FFmpeg's library hwaccel path and downloads every hardware
  surface for comparison. It is constructed by the opt-in interop probe
  alone — no production decode path, no wire action, no toggle. Landing it
  moved the capability to exactly
  `EvaluationRequired(InteroperabilityProof)` on Windows through the
  module's own `backend_integrated` seam, and deliberately no further: the
  tracked interop receipt is evidence for the operator's next decision, not
  a runtime fact, so `Available`, live usage, and zero-copy are separate
  operator-decided tranches. `hardware_decode_active` stays false because
  `EvaluationRequired` is not `Available`. Export determinism is a standing
  boundary for any future live tranche: the offline renderer keeps its
  synchronous software decoders unless per-adapter bit-exactness is proven —
  "the same patch exports differently on different GPUs" is never an
  acceptable trade.
- `ExperimentalFull16History` now has an implemented render path, but it is
  **measurement-only**: `CompositionHost::new_with_history_storage` widens
  exactly the 25-layer temporal class (ring plus feedback) to RGBA16Float, is
  constructed by the Gate 6 receipt fixture alone, and has no wire action, no
  patch field, no env toggle, and no production call site. The settled
  `AdvancedWorking16HistoryCompat8` default has not moved — the production
  `CompositionHost::new` delegates with `Compat8` and the M6 receipt's pinned
  output SHAs prove byte identity. Because the ring and feedback are written
  only by render passes and read only by texture loads, both storages present
  identical linear values to every consumer; the candidate changes
  quantization, never value domain, and no consumer shader changed. The
  Symmetry-Field section's prohibition on a *new* RGBA16F full-frame history
  ring is untouched: the candidate widens the existing ring under the
  documented budget, it does not add a second one.
- The Symmetry Field's eight-texture single pass is a *floor* claim resting on
  the S2 receipt's enforced-cap argument, measured on one adapter and one
  backend. It is a capability claim only, not performance, bandwidth, or cache
  behaviour.
- The Symmetry Field pipeline layout is **three** bind groups, not the frozen
  table's two. See the dedicated section: the motion pair owns its own group so
  a `MotionGpuField`'s committed parity adds to the carrier and N-1 parities
  instead of multiplying with them.
- A master-scope Symmetry Field counts as a global step for the canonical
  reordering law, so it disables selective-VHS bypass authoring
  (`AmbiguousMasterBypass`) exactly as any other non-marker master node does.
- The Field Collider is a Motion-subsystem block and deliberately takes no
  `NodeKindTag` code. Its `MotionBoundaryMode` reuses the frozen
  `Transparent = 0, Mirror = 1, Wrap = 2, Hold = 3` numbering rather than
  minting a motion-specific table; §5 of the enrichment plan lists those names
  in a different textual order, which is prose, not a code assignment.
- Version 1 of the Field Collider adds no collider-only continuous control, so
  it has no modulatable address and Dice, the generator, and modulation preserve
  the block exactly. A control added in a later version would need its own
  address, its own Morph law, and a revisit of the endpoint-exact block pick.
- Only one Field Collider is admitted per composition, matching the single
  admitted Faraday transplant it advects. A second would need its own derived
  slot range and a second entry in the byte ledger.
- `resolved_export_motion` now delegates to `MotionConfig::resolve_runtime`
  rather than binding one donor by hand, so an offline render resolves exactly
  the Motion-subsystem donors a live one does. Any donor added to the block in
  future is bound in both paths by that single resolver.
- An Advanced composition whose layers own no image tap at all schedules
  normally. This bullet previously recorded the opposite as a standing
  constraint; the tranche it deferred has landed, and the composite-rank
  tie-break in "Advanced execution order" above is the binding statement.
  `render_tapless_advanced_motion_pipeline` is the labeled export case that
  could not be prepared before it, so the older claim that every labeled
  Advanced case carries a rack node is also no longer true — do not add a node
  to a fixture on that reasoning.
- Physical MIDI, phone, audio-interface, Spout-host, and multi-monitor proof is
  separate from software tests. Gesture ingress is proven through the one
  normalized adapter and its four origins in software; a real tablet, phone
  touch surface, MIDI controller, or OSC peer authoring a stroke is hardware
  proof and is not transferable from those tests.
- The gesture canvas is one master-scope singleton. The frozen table admits two
  active canvases, but the second is the offline job's own; a genuinely second
  *routable* canvas would need an index in the route vocabulary, which is a
  wire and persistence change rather than a renderer one.
- `GESTURE_CANVAS_HOST_MAX_CELLS` is a render-thread cost budget, not the
  frozen resource table. The host runs the portable CPU reference every frame,
  so raising it toward the 2,100,000-cell ceiling belongs with a presenter that
  moves the per-cell work off that reference entirely.
- The preview transform gizmo addresses master and layer scopes only. Groups
  carry a `SpatialTransform` but have no interactive authoring action of any
  kind, so a group handle would author what no other controller can; adding one
  is a wire and protocol change.
- The gizmo edits and displays the **authored base** transform, exactly as the
  browser numeric fields do. A modulation route offsets a per-frame copy, so
  while one is driving a transform the rendered image sits away from the
  handles by exactly that offset. This is the same relationship the numeric
  editor already has, not a gizmo defect.
- Gizmo hit testing and painting are proven in software. A physical operator
  dragging a handle on a real pointer or tablet is hardware proof and is not
  transferable from those tests.
- Upstream original code has no blanket MIT grant; `LICENSE` only covers the
  additions described there. Publication/distribution of the combined fork is
  conditional on the publisher having authorization for the original portions
  or a later upstream license that permits it. Record this boundary without
  presenting project documentation as legal advice, and do not broaden the MIT
  claim.
