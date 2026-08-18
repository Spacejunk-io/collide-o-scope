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
├── motion.rs            canonical codec/lattice fields, Motion authoring, resource preflight
├── temporal.rs          Loom/Atlas/Garden/Score state, events, resets, commit/discard
├── gesture.rs           portable quantized gesture events, checksum, one normalized adapter
├── gesture_canvas.rs    bounded vector canvas CPU reference, Push/Curl laws, transactions
├── renderer/state.rs    LegacyExact passes, audience history, readbacks, output blits
├── renderer/composition.rs shared Advanced GPU executor and transactional histories
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
├── proxy.rs             measured proxy recommendation and content-addressed cache plan
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
    ├── composition_host.wgsl straight storage; premultiplied A/B/group math
    ├── motion_*.wgsl    field acquisition, transform shutter, Faraday memory
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
Changing anchor alone must remain visually inert. Sanitize all finite inputs,
wrap angles, prevent crop collapse, and fail singular transforms to transparent.

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
- `Ctrl+E` — patch parameter editor
- `Ctrl+S` / `Ctrl+O` — save/load patch
- `Ctrl+Shift+I` / `Ctrl+Shift+X` — import/export the bounded controller profile
- `Escape` — close/quit as appropriate

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
- `cargo test` covers pure protocol, persistence, modulation, morph, temporal,
  transport, audio, Spout lifecycle helpers, and export-argument behavior.
- `cargo test effects_audit -- --ignored` renders the labeled effect suite
  through the real export path. It requires a working GPU, FFmpeg on `PATH`,
  and `videos/audit.mp4`.
- Export repeatability requires two independent renders and equal decoded
  `framemd5` sequences.
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
- Upstream original code has no blanket MIT grant; `LICENSE` only covers the
  additions described there. Publication/distribution of the combined fork is
  conditional on the publisher having authorization for the original portions
  or a later upstream license that permits it. Record this boundary without
  presenting project documentation as legal advice, and do not broaden the MIT
  claim.
