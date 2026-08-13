# Collide-o-Scope developer notes

Native Rust VDJ instrument. It decodes video or receives live Spout frames,
composites up to 16 layers with GPU effects, exposes browser controls, and can
render a saved patch offline.

## Stack

- **winit** — native and fullscreen-output windows, input events
- **wgpu 29** — GPU effects, compositing, temporal passes, readback
- **ffmpeg-next 8** — video decode and media-stream inspection
- **ffmpeg CLI** — thumbnails and final H.264/AAC muxing
- **ntsc-rs** — CPU VHS emulation on a worker
- **cpal** — live audio capture
- **midir** — MIDI CC and clock input
- **spout2-rs** — Windows Spout sender and receiver
- **axum + tokio** — HTTP/HTTPS panel and WebSocket state/action protocol
- **egui** — native preview shell and patch parameter editor

The `ffmpeg-next = "8"` crate must match the installed FFmpeg major.

## Module map

```text
src/
├── main.rs              winit loop, app state, web actions, patch reconstruction
├── renderer/state.rs    wgpu passes, temporal state, async readbacks/output blits
├── video/decoder.rs     synchronous ffmpeg decode core and RGBA row repacking
├── video/threaded.rs    request-driven decoder, first-frame seed, latest-only mailbox
├── layers/mod.rs        video/Spout layer sources, texture upload, frame pacer
├── effects/params.rs    effect and temporal parameters/normalization
├── modulation/mod.rs    clock, LFOs, expressive inputs, routes, curves, slew
├── audio/mod.rs         cpal capture, FFT sources, configurable edges/spectrum
├── midi/mod.rs          MIDI CC learn table and clock
├── morph.rs             A/B snapshots, blend laws, beat glides, persistence
├── ntsc/mod.rs          ntsc-rs parameters/state and worker
├── spout_in.rs          newest-frame-wins live receiver worker
├── spout_out.rs         bounded/drop-new output worker
├── patch/               YAML model, capture/apply, editor and file dialogs
├── render_export.rs     deterministic offline renderer and optional audio mux
├── web/                 panel server, protocol snapshots/actions, embedded assets
├── input/keyboard.rs    key-to-action mapping
└── shaders/
    ├── fullscreen.wgsl  fullscreen triangle vertex shader
    ├── effects.wgsl     layer/master effect shader
    ├── composite.wgsl   layer blending
    └── temporal.wgsl    feedback and arbitrary-angle slit-scan
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
→ matrix update (curve/slew/pad spring)
→ morph sample writes frame-local bases
→ route offsets create frame-local effect/layer copies
→ layers → master → temporal → blackout
→ async readback consumers (NTSC/Spout) and window blits
```

The exporter follows the same effect, morph, modulation, NTSC, and temporal
helpers where possible. Time-dependent offline behavior must derive from
`frame_index`, export FPS, and patch state—not wall time or a live input.

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
- The **NTSC worker** accepts at most one in-flight composite. Newer work is
  dropped instead of queued while it is busy.
- The **Spout output worker** owns its thread-affine DX11 sender and uses a
  bounded/drop policy.
- Up to three **GPU staging readbacks** are active. Completed readbacks are
  harvested in submission order so an older composite cannot overwrite a
  newer result.
- The **web server** runs a tokio runtime. Its bounded `WebAction` queue
  coalesces older absolute values for the same semantic control, reserves
  admission for safety/release actions, and broadcasts a complete
  `AppSnapshot`. Do not replace it with an unbounded collection.
- The **thumbnail worker** invokes FFmpeg outside the render path.

## Layer sources and limits

`MAX_MOD_LAYERS` is 16. Do not raise the app's layer limit without also making
every layer index available to `target_range` and the panel target list.

`LayerSource` distinguishes decoded video from a live `SpoutIn`. Every layer
keeps a stable `source_path`:

- video: canonical file path when available, otherwise the supplied path;
- Spout: `spout://<sanitized-sender-name>`.

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
`downsample`, and keep their protocol ranges aligned with patch normalization
and modulation target ranges.

Patch load prefers this identity and falls back to the legacy library
filename. Missing video and Spout sender errors are surfaced; a live Spout
texture begins as transparent black and is never uninitialized.

Offline export cannot replay an external live Spout sender. It retains that
layer's saved stack index and renders an explicit black placeholder so
layer-numbered modulation routes do not slide onto another source.

## Modulation

The static `TARGETS` table covers continuous master, NTSC, temporal, and morph
values. `target_range` additionally recognizes `layer1_…` through
`layer16_…` for:

- opacity, speed, key threshold;
- pixelate, RGB split, hue, saturation, brightness, contrast, posterize;
- grain intensity/size, vignette, color drift;
- breathing scale/rotation/position;
- key softness and downsample.

All route consumers share the same shaping/slew state. Curves are Linear,
Exp, Log, SCurve, and Steps; signed shaping preserves bipolar source sign.
Attack and release are independent seconds-based time constants, updated once
per frame. `MAX_ROUTINGS` is 64.

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

The morph target is itself routable. Live and export sample the same persisted
state; modulation offsets are applied around the sampled frame bases.

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

Live rendering passes elapsed dt; export passes frame-index-derived dt. Keep
the shader uniform layout tests current when changing temporal uniforms.

## Effects and compositing

- One combined uniform-driven effect shader avoids pipeline switches.
- A fullscreen triangle computes UVs from `vertex_index`.
- Luma key produces straight RGB plus modified alpha; the compositor handles
  opacity/premultiplication. Do not multiply keyed RGB by alpha a second time.
- Hidden layers clear their intermediate contribution so old texture contents
  cannot leak into the stack.
- The chain operates with sRGB decode, linear-light math, and sRGB encode.
- Blackout occurs before audience-facing readback/blit consumers.

## Patches and native parameter editor

`PatchState::capture` includes master and layer state, stable source identity,
master pause, NTSC, temporal, the complete modulation/input configuration, and
a normalized morph snapshot. `Ctrl+S`/`Ctrl+O` use file dialogs;
`Ctrl+E` exposes the patch parameter editor in the native panel. `Ctrl+S`
serializes the complete `PatchState` as YAML; the native editor intentionally
edits the live master/layer parameter subset rather than presenting itself as
a full YAML text editor.

After a patch has rebuilt and validated its complete layer stack, commit new
topology and visual generations. Clear both the immediate web queue and the
downbeat-latched queue, advance `layer_stack_revision` and the application
visual epoch, clear renderer temporal history and retained NTSC output, and
invalidate pending readbacks so downstream Spout/NTSC consumers reject work
from the previous patch. Do not reset the current world before reconstruction
succeeds.

Backward compatibility rules matter:

- no `modulation` section leaves the existing matrix untouched;
- old layers without `source_path` resolve by library filename;
- old temporal `slit_axis` values map to 0° or 90°;
- new finite/clamp defaults reject NaN/overflow without panicking.

## Offline export

The offline renderer uses synchronous decoders intentionally. It evaluates
beat, route slew, pad spring, morph glide, layer transport, temporal history,
and frame counts from the selected FPS. Live audio/MIDI values are zero.
Unavailable/missing/live sources become deterministic black placeholders at
their original source indices.

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
The bounded/coalescing ingress queue is a load-shedding boundary, not an excuse
to omit pointer throttling or final pointer-release messages.

`AppSnapshot::output_error` is the operator-facing failure channel for the
fullscreen output window. Keep the checkbox synchronized to the renderer's
actual surface state and show creation/surface errors instead of implying that
an unavailable output is open.

The panel uses fixed functional columns on desktop and one column under 900
px. Layer reordering sends one atomic `move_layer` on release. Range controls
support double-click/double-tap reset, and group reset buttons send explicit
reset actions.

After rebuilding embedded assets, hard-refresh any open panel. Only the process
that owns ports 3030/3031 serves the panel; a second instance can otherwise
mislead browser tests.

## Keyboard

- `P` / `Shift+P` — pixelate up/down
- `G` / `Shift+G` — RGB split up/down
- `0` — reset effects
- `Space` — selected-layer pause/resume
- `F` — main-window fullscreen
- `O` — fullscreen output window
- `B` — blackout
- `Ctrl+E` — patch parameter editor
- `Ctrl+S` / `Ctrl+O` — save/load patch
- `Escape` — close/quit as appropriate

## Verification

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
  patch/parameter-editor state, and export.
- Windows Spout requires real sender and receiver applications; use
  `spout_probe` for the output side.

Keep dated verification evidence and hardware deferrals in local, untracked
records. Do not convert an unrun checklist item into a passing claim.

## Known constraints

- FFmpeg library/runtime major must remain 8 unless the Rust dependency changes
  with it.
- Bindgen on Windows needs LLVM and the Visual C++ include environment.
- The `block` crate may emit an upstream future-incompatibility warning.
- Spout is Windows-only and live Spout input becomes deterministic black
  offline.
- Audio exposes 3–8 routable bands; the compact 32-bin spectrum remains
  display-only telemetry.
- Physical MIDI, phone, audio-interface, Spout-host, and multi-monitor proof is
  separate from software tests.
- Upstream original code has no blanket MIT grant; `LICENSE` only covers the
  additions described there. Publication/distribution of the combined fork is
  conditional on the publisher having authorization for the original portions
  or a later upstream license that permits it. Record this boundary without
  presenting project documentation as legal advice, and do not broaden the MIT
  claim.
