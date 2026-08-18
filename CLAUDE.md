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
├── composition.rs       stable-ID groups, buses, mattes, and authored composition topology
├── evaluated_composition.rs unified LegacyExact/Advanced planner and resource admission
├── visual_rack.rs       ordered scope racks, typed nodes, taps, validation, persistence
├── image_routing.rs     stable layer/group image routes and missing-target tombstones
├── media_safety.rs      Safe/Expert source planning, device bounds, reservations
├── media_source.rs      shared resolution, bounded SHA-256 fingerprinting, content references
├── spatial.rs           canonical authored transforms and packed GPU pass uniforms
├── transform_gizmo.rs   preview-only direct manipulation of that same transform
├── motion.rs            canonical codec/lattice fields, Motion authoring, Field Collider, resource preflight
├── symmetry.rs          closed symmetry groups, 32-sector table, 1,024-byte uniform
├── temporal.rs          Loom/Atlas/Garden/Score state, events, resets, commit/discard
├── gesture.rs           portable quantized gesture events, checksum, one normalized adapter
├── gesture_canvas.rs    bounded vector canvas CPU reference, Push/Curl laws, transactions
├── renderer/state.rs    LegacyExact passes, audience history, readbacks, output blits
├── renderer/composition.rs shared Advanced GPU executor and transactional histories
├── renderer/symmetry_field.rs dedicated eight-texture sampler-free Symmetry pass
├── renderer/gesture_canvas.rs ping-pong etch canvas and the presented donor image
├── renderer/stage_map.rs fixed-resource multi-endpoint venue presenter
├── video/decoder.rs     synchronous ffmpeg decode core and RGBA row repacking
├── video/threaded.rs    request decoder, codec motion, telemetry, latest-only mailbox
├── layers/mod.rs        video/Spout layer sources, texture upload, frame pacer
├── effects/params.rs    effect and temporal parameters/normalization
├── modulation/mod.rs    stable typed targets, clock/LFO/audio/MIDI routes, curves, slew
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
├── patch/               YAML model, capture/apply, editor and file dialogs
├── procedural.rs        deterministic v7 typed patch walk, manifests/preflight, capture worker
├── render_export.rs     deterministic shared executor, motion report, optional audio mux
├── web/                 panel server, protocol snapshots/actions, embedded assets
├── input/keyboard.rs    key-to-action mapping
└── shaders/
    ├── fullscreen.wgsl  fullscreen triangle vertex shader
    ├── effects.wgsl     LegacyExact and Advanced layer/master effects
    ├── rack_node.wgsl   Collision Rack nodes and image-tap effects
    ├── symmetry_field.wgsl dedicated eight-texture group fold, no sampler
    ├── composition_host.wgsl straight storage; premultiplied A/B/group math
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
  refuses new work while busy instead of queueing a backlog. A job
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
slots, three gyro axes, and two pad axes. LFOs are bipolar; external sources
are normalized to 0…1.

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
full-frame history ring is prohibited** (~398 MiB at 1080p) absent an explicit
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

Generator v7 normalizes the anchor, replaces verified file sources with content
references, reduces filenames to logical names, and hashes version-prefixed
canonical JSON with SHA-256. `anchor_sha256`, `piece_sha256`, and lineage must be
path-independent and source-byte-sensitive. Schema-v2 manifests retain defaulted
v1 fields for deserialization compatibility. Generator v7 also applies bounded,
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
- `Y` — encode a proxy for the selected layer's verified content identity;
  every refusal and completion reports through that layer's HUD status line
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
  checks, `cargo check --locked --all-targets`, strict all-feature Clippy, and
  the single-threaded locked all-target/all-feature test matrix. A publication
  claim additionally requires Linux, macOS, and Windows CI success for the
  exact published commit SHA; an older green run is not transferable evidence.
- Physical-GPU proofs are opt-in and therefore separate from ordinary CI.
  StageMap uses the five `renderer::stage_map::tests::physical_gpu_` fixtures.
  M6 precision uses
  `gpu_precision_receipt_measures_real_still_and_temporal_workloads` plus its
  premultiplied-edge, temporal-feedback, LegacyExact-spatial, and 24/30/60
  temporal parity companions. Keep the adapter/backend, exact command, source
  manifest, and receipt hash with any claim.
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
- Proxy-worker tests split along the CLI boundary. Hosted (all three CI
  platforms, no ffmpeg CLI): the crash test written reproduction-first — a
  staging leftover removed and never published or counted, an unsealed
  artifact and an orphan seal both removed as interrupted publications; the
  atomic publish law with the prior artifact readable until replacement and
  the seal following the artifact; mid-file corruption refused by the seal
  and discarded; eviction following the pure plan with a path-free receipt;
  foreign files counted but never touched; the contract-derived argv; garbage
  bytes failing decoded-identity validation; mutated/unreadable sources
  refused before any encode; and the Y-key mapping. Opt-in (`--ignored`,
  ffmpeg CLI required, like `effects_audit`):
  `proxy_worker_end_to_end_encode_publish_rename_and_corruption_survival` —
  encode, validate, publish, cache hit, identical bytes at a renamed path
  hitting the same key and adopting, corruption refused at consultation and
  at the job's own cache-hit path, crash recovery beside a live cache, and
  both audio laws — plus
  `proxy_encode_kill_bounds_are_typed_and_publish_nothing` for the deadline
  and size-cap kills. Windows fsync law: `FlushFileBuffers` demands writable
  handles for both the staging file and the parent directory; do not "fix" a
  publish failure by dropping either sync.
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
- The proxy loop is closed for content-referenced video, with three honest
  edges. `proxy_worker.rs` executes the `plan_proxy_input` contract: the Y
  key requests a bounded FFV1/Matroska encode (single helper, absolute
  deadline, staging-size kill, `MediaSafetyPolicy` reservation held for the
  encode, source re-fingerprinted first), publication follows the atomic
  commit law with a SHA-256 seal published after the artifact, recovery
  removes staging and unsealed residue without ever serving it, and patch
  load consults the cache — a validated artifact backs the decoder while the
  layer keeps the original identity, so a proxy can never enter a patch, an
  export, or Dice. The edges: adoption happens at patch (re)apply, not by
  hot-swapping a live decoder; only sources with a verified `cos-sha256`
  identity can be proxied, because the key is content-addressed; and the
  browser panel has no proxy surface — request and status are native
  (Y key + stage-health HUD). A host killed mid-encode may orphan one ffmpeg
  process bounded by its own completion; the staged file it writes is
  recovery residue, never an artifact. The Unix CI FFmpeg build carries
  `--disable-programs`, so end-to-end encode fixtures are opt-in like
  `effects_audit` and hosted CI proves the CLI-free cache half only.
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
