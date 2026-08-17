# collide-o-scope

A native VDJ (video DJ) instrument for live visual performance. It composites
video, still-image, and live Spout sources with GPU effects, then lets LFOs, audio analysis,
MIDI, and a phone's touch and tilt modulate the performance from a browser
control panel.

> This is a fork of [collide-o-scope by Luis Queral](https://github.com/luismqueral/collide-o-scope)
> ([queral.studio](https://queral.studio)). The original engine, compositing
> architecture, and effect suite are his work. This fork adds the modulation
> matrix, remote control, audio/MIDI reactivity, temporal effects, Spout I/O,
> offline export, and Windows support. See [LICENSE](LICENSE) for the precise
> licensing and attribution boundary.

<img width="1919" height="1040" alt="collide-o-scope" src="https://github.com/user-attachments/assets/3690c19f-8ab4-4a9d-9672-45476da4dede" />

## What it does

- Composites a dynamically sized stack of video, PNG/JPEG/BMP/WebP still, or
  live Spout layers with normal, screen, multiply, and difference blending.
  There is no fixed layer-count policy; usable depth is governed by source,
  GPU, output-resolution, and selective-VHS memory resources.
- Applies per-layer and master pixelation, RGB split, hue, saturation,
  brightness, contrast, posterize, invert, film grain, vignette, color drift,
  breathing motion, seeded horizontal block Shift, luminance/chroma keying,
  and bounded animated cellular/Worley warping with a separately feathered
  cell-gap key.
- Runs [ntsc-rs](https://github.com/ntsc-rs/ntsc-rs) VHS simulation without
  blocking the render thread. Existing all-inherited stacks keep the global
  post-composite path; a contributing **Bypass Master FX** layer selects an
  exact per-layer path so VHS touches inherited layers only.
- Provides feedback trails and arbitrary-angle slit-scan from a 24-sample
  temporal history. History advances at a fixed 30 Hz, so its approximately
  0.8-second span does not change with display or export frame rate.
- Saves complete YAML performance patches. **Load snapshot** reconstructs the
  saved performance, while **Apply look** transfers its visual treatment onto
  the sources already on stage.
- Keeps live decoding request-driven and bounded: each layer's worker performs
  only transport advances requested by the engine and publishes the newest
  completed image through a one-frame overwrite slot. A first-frame seed is
  decoded at open time, so a clip added or restored while paused still has a
  defined image.
- Keeps a browser-independent **RECOVERY** strip on the native preview with
  truthful panel-listener and browser-connection status, an authenticated
  **Open Panel** link, absolute Freeze Program and Blackout controls, broad
  visual Revert, active-library selection/rescan, and surfaced output/media
  status. It is a preview-only operator surface and never contaminates an
  audience output.
- Defaults source admission to **Safe** and offers an explicit, host-session
  **Expert** mode for future large video, still, and Spout opens. Expert remains
  bounded by fixed area/edge ceilings, device limits, and a conservative host-
  memory plan; it is not an unbounded allocation switch.
- Renders MP4 files through a deterministic frame-indexed offline path; real
  MIDI, phone, Spout, audio-device, and multi-GPU output still require the
  corresponding hardware for end-to-end validation.

## The modulation matrix

Every expressive input follows one rule: sources enter the matrix, routings
shape and scale them, and the resulting offsets are applied to per-frame
copies. Base values—the values shown by sliders and stored in patches—remain
unchanged.

| Sources | Detail |
|---|---|
| 4 LFOs | Sine, triangle, saw, square, or deterministic sample-and-hold; musical divisions; tap-tempo or MIDI clock |
| Audio | Live input, Windows system-playback loopback, or deterministic looping WAV/MP3/FLAC/Ogg/Opus/M4A/AAC analysis; level; 3–8 configurable bands; onset, brightness, noisiness; and a 32-bin display spectrum |
| MIDI | Four MIDI-learn CC slots plus 24-PPQN clock/start handling |
| Phone | Calibrated yaw/pitch/roll and a multitouch XY pad |

Each routing supports signed depth, Linear/Exp/Log/SCurve/Steps response, curve
amount, and separate attack/release slew. A route meter—centered for bipolar
sources—shows the shaped and slewed value before depth is applied. Each row
also carries a stable runtime ID, so a delayed browser edit or remove remains
attached to the intended route even when another controller changes the matrix
first. Runtime IDs are intentionally recreated when a patch is loaded; the
route settings, not process-local identifiers, are what patches preserve.

The engine resolves route destinations ahead of the render hot path and builds
one immutable modulation result per rendered frame. Morph position, master
targets, transport, and every layer target reuse that same sample; source,
curve, and slew work is not repeated for each consumer. Offline rendering uses
the same frame-indexed ordering.

The matrix exposes every continuous master and NTSC value, Shift amount/block
size/density/speed, temporal feedback, slit-scan and history-key values, morph
position, and each layer's opacity, speed, target FPS, key controls, and
continuous effects. This includes RGB
chroma targets/tolerance, key thresholds and softness, temporal key history,
and VHS edge-wave speed, tracking wave, composite/chroma noise, luma smear,
and sharpening. Selector choices such as static/temporal key mode, blend mode,
and grain algorithm remain deliberate discrete controls and are not modulation
targets. The legacy patch target `layerN_key` is read as the canonical
`layerN_key_threshold`; if both spellings occur, the canonical route wins.
Every current layer is routable, including stacks beyond the original panel's
former 16-layer ceiling; the independent limit remains 64 simultaneous routes.

Phone input is configurable at the engine rather than being a one-off browser
effect:

- Gyro axes have **Zero here** calibration, range, exponential response, and
  invert controls.
- XY axes have independent curves and step quantization. Optional spring
  return moves the released pad back to center at a configurable rate.

Gyro and XY-pad routing is bipolar: the calibrated or physical center produces
zero, with travel on either side producing negative or positive modulation.
When layers move, their positional modulation targets are remapped through the
same permutation so routes follow the same logical sources. Removing a layer
drops routes aimed at that layer and shifts targets above it to match the new
stack.

For pad quantization, a value of N from 2 through 64 means exactly N evenly
spaced positions, including both endpoints; 0 or 1 disables quantization.

Live audio keeps three states distinct: the saved/requested device preference,
the device actually backing the stream, and whether that stream is the system-
default fallback because a named device disappeared. A failed or stalled
stream is stopped, its modulation sources return to zero, and the enable state
returns to off instead of retrying every frame.

Imported audio is a different source mode. In **Looping file**, **Choose
imported audio…** opens the native multi-file chooser filtered to WAV, MP3,
FLAC, Ogg, Opus, M4A, and AAC. Each audio upload is limited to 512 MiB; FFmpeg
decodes at most 10 minutes and abandons a decode that has not completed within
60 seconds. A successful clip is decoded once, then program time selects a
circular analysis window. It does not need to play through the speakers. The
same clip, gain, band layout, and timestamp produce the same routing values
live and offline; Freeze Program holds that timestamp exactly, while Freeze
Media leaves it running.

The optional beat latch coalesces eligible control changes and releases them
on the next four-beat downbeat. The morph section supports linear or
equal-power interpolation plus beat-duration glides to A or B. A capture is an
ordering barrier: it first materializes the current Morph result, including a
Morph-target routing offset, then records the bases at the current layer-stack
revision. Stale captures are rejected. Manual fader and law edits still apply
to the materialized bases while Freeze Program holds automatic glide and clock
motion.
While both slots are engaged, A/B owns the controls it captured. Moving or
resetting one of those controls commits the currently displayed interpolation,
clears A/B, and then applies the manual edit so it cannot snap back. A newly
appended layer remains outside older slots and can be edited without disengaging
them. With beat latch enabled, that ownership transfer occurs on the downbeat.
Freeze Program also snapshots the exact audience across a blackout: the cut
remains absolutely black while active, then releasing it while still frozen
restores the pre-cut image. A frozen selective-VHS transition remains held
until the program resumes and can produce a complete replacement. Hue and slit
angle take the shortest wrapped arc,
discrete choices switch at the midpoint, and stored layer slots follow
reorder/removal while a newly appended layer remains untouched.
Slots, law, position, and the exact remaining glide—even below the UI's
quarter-beat minimum—are patch-persistent. Other modulation offsets remain
frame-local and do not rewrite the captured bases.

## Freeze and Random / Dice

**Freeze Program** holds the complete visual program: file and Spout images,
shader/VHS time, temporal history, LFO and morph phase, routing slew, and
imported-audio analysis all resume without catch-up. **Freeze Media** (`M` or
the **MED** button) holds only file and Spout images; program time, animated
effects, modulation, temporal/VHS processing, and imported-audio analysis keep
running. The two freeze states are independent and patch-persistent. `Space`
pauses the selected layer; only when no layer is selected does it toggle
Freeze Program.

**RANDOM / DICE** is deterministic. **Pattern only** changes stochastic shader
seeds without moving effect controls. **Bounded variation** first chooses the
seeds, then makes range-safe changes to continuous Digital, Analog, Motion,
Shift, and core Cellular amount/scale/warp/speed controls; **Amount** sets their scale
and **Grain mode** additionally allows the grain algorithm and color-grain
switch to change. Sources, stack, opacity, visibility, blend, keying, Bypass
Master FX, transport, routings, VHS, and Temporal are not randomized.

**Master** targets the master pattern plus all four LFO seeds; **Everything**
also targets every current layer. A layer card has its own seed and
pattern-only reroll. Leave **Exact seed** blank to advance deterministically
from stored master/layer seeds, or enter an unsigned 32-bit base value to
replay it. Master keeps that base exactly; Everything derives reproducible
per-position layer streams. Master and Everything derive all four LFO seeds
from the resulting master seed. `0` explicitly restores the legacy pattern
family everywhere. Each LFO exposes its own 32-bit seed, and sample-and-hold
holds one deterministic value for each complete LFO cycle. A decoded video can
also enable **each loop**, which advances that layer's seed at every loop
boundary live and offline; stills and Spout inputs do not reroll. All seeds and
per-video loop choices persist in snapshots.

**Shift** divides the image into output-pixel horizontal bands and displaces a
seeded subset sideways with wrapping edges rather than exposing blank pixels.
**Amount** bounds displacement to at most one quarter of the image width;
**Block px**, **Density**, and **Speed** control band height, the participating
fraction, and deterministic time epochs. Amount zero takes the exact established
shader path. Freeze Program holds the epoch; Freeze Media lets it advance while
source images remain held. Pattern-only Dice or a layer **Reroll** changes the
arrangement through the stored seed, while Bounded variation can also move the
four Shift controls. Master and layer Shift values are patch-persistent,
Morph-interpolated, modulation targets, and evaluated by the same shader during
offline export.

## Browser control

The app opens a per-session desktop URL of the form
`http://127.0.0.1:3030/?key=<session-token>` and serves the tokenized HTTPS
phone panel on `:3031`. The token is required on loopback as well as the LAN,
is exchanged for a strict HttpOnly session cookie, and is then removed from
the visible address bar. WebSocket and mutating HTTP requests must also be
same-origin. The REMOTE section shows the current tokenized QR code. HTTPS is
required for iOS motion permission; a bare, stale, unauthenticated, or
cross-origin control request receives 403.

The native preview's **RECOVERY** strip remains useful when no browser is
connected or the panel listener cannot bind. It reports the listener lifecycle
separately from browser count, exposes the current tokenized **Open Panel** URL,
and dispatches Freeze Program, Blackout, and **Revert Visuals** directly through
the engine. It also shows the active library and provides **Choose Library** and
**Rescan**, and it surfaces recoverable output and media-source status. The strip
belongs only to the operator preview: it is hidden when single-monitor Output
owns the main surface, and a dedicated output display is always clean.

The panel is mobile-first below 900 px. Touch controls include pointer-safe XY
input, layer drag-to-reorder, group resets, and double-tap/double-click reset
for individual sliders. Layer cards expose direct transport, target decode
FPS, keying, and the complete per-layer effect set, including downsample; those
effect values are also modulation targets. The **Layer effects** disclosure and
its nested **CELLULAR** disclosure start closed for each new layer card and
remember their state by layer identity while that layer remains present. All
connected panels receive the same engine state.

The Library column makes the source-allocation policy visible. **Safe** is the
default and preserves the established per-source UHD-area ceiling of 8,294,400
pixels / 33,177,600 RGBA bytes. **Expert** applies only to future source opens
and permits at most DCI-8K area (35,389,440 pixels / 141,557,760 RGBA bytes),
subject to the 16,384 px absolute edge, the device's 2D-texture edge and per-
buffer limits, and an aggregate host-memory planning budget no larger than one
eighth of detected physical RAM or 2 GiB, whichever is smaller. Portable wgpu
does not report free VRAM headroom, so the eventual GPU allocation can still
reject a source recoverably. Texture creation and every source upload are
error-scoped; a failed upload remains uninitialized/inactive and surfaces on
the layer instead of being reported healthy. The mode is not stored in patches;
returning to Safe affects future allocations and does not destroy accepted
sources.

Library thumbnail and hover-preview work is subject to that same admission
planner before FFmpeg decodes a candidate. Helpers have a fixed timeout,
bounded captured output, and library-generation cancellation that does not
suppress the deadline. Both encoded dimensions are capped at 180 px, each
accepted JPEG is at most 512 KiB, and thumbnails plus preview strips share a
64 MiB retained-cache ceiling. Safe mode permits at most four thumbnail and two
preview helpers process-wide, while Expert mode serializes helpers one at a
time. Startup, folder changes, and repeated rescans share a single-flight gate;
a newer rescan cancels and then replaces the older generation. This background
convenience path cannot evade the DCI-8K-area, device-edge,
max-buffer, host-planning, or cache boundaries.

Each layer can independently enable **Bypass Master FX**. This skips inherited
Digital, Analog, Cellular, Motion, and VHS processing for that layer while its
own Layer FX, opacity, key, and blend remain active. Temporal remains a
program-wide history stage. With VHS enabled and a visible, positive-opacity
bypass layer contributing, the engine renders coherent per-layer slices,
applies direct master effects and VHS only to inherited slices, recomposites in
stack order, and then runs Temporal. Hidden or non-positive-opacity layers do
not allocate selective work. Live selective processing is latest-only and has a
320 MiB incremental safety budget. If a resolution/layer combination exceeds
that budget or processing fails, the engine holds the prior exact audience
frame and reports the VHS error; it never falls back to applying global VHS to
a bypassed layer. The bypass is non-destructive: neither **Reset FX** on the
layer nor master **Revert** changes it. Expert media mode does not enlarge this
selective-VHS budget. The VHS panel labels the active global or selective live
path and reports admitted work and healthy busy/backpressure skips; it separates
unavailable worker failures, stale completions, and the current busy state.

Every slider's displayed value is also an editable numeric field. Select it,
type any in-range value, and press Enter or leave the field to commit through
the slider's normal action path; Escape restores the engine value. Inputs are
clamped and quantized to the control's advertised range and step. Each layer
card's Cellular controls additionally expose Gap Key, Gap Threshold, and Gap
Softness so its Voronoi boundaries can reveal lower layers without dark
fringes. The master Cellular panel exposes the same controls. In an ordinary
post-stack master pass its keyed gaps resolve over black; when per-layer master
bypass makes master processing conditional, inherited-layer gaps can reveal
content beneath them.

Static keying has five modes at both layer and program scope: Off, Keep Bright,
Keep Dark, Remove Chroma, and Keep Chroma. Luminance modes use threshold and
softness; chroma modes use an RGB target, tolerance, and softness. A layer key
changes that layer's alpha and reveals the stack below. A program key runs on
the flattened image, so removed pixels become black rather than exposing a
nonexistent lower program layer.

Temporal history keying compares the current clean composite with a selected
prior sample in the fixed 30 Hz history. It can retain motion, stillness,
brightening, or darkening, with threshold, softness, and a selectable history
depth of 1–23 samples. Its mask gates the temporal output after feedback and
slit-scan processing; Off preserves the established temporal path.

Browser input is bounded and coalesced before the render thread consumes it:
new absolute values replace older pending values for the same control, while
safety/release actions retain admission under heavy fader traffic. Each live
layer also has an immutable session ID, and reorder commands carry the current
stack revision. If another controller has already changed the layer stack, a
stale ID or reorder revision is rejected rather than being applied to the clip
that now occupies the old index. See
[docs/remote-control.md](docs/remote-control.md) for setup and troubleshooting.

## Sources and output

- **Video and still layers:** add files from the active library or drag/drop.
  The default active folder is `videos/`; the native **Choose Library** control
  can switch the visual/audio scan and browser-upload destination. **Rescan**
  refreshes that folder. Neither operation adds a layer automatically. Patches
  retain a stable path and fall back to the active-library filename for older
  patches.
- **Spout input (Windows):** enter an exact sender name and choose **Add
  live**. The receiver is an ordinary composited layer with transport/effect
  controls and a visible connection status. A missing or warming sender stays
  black. Live Spout cannot be sampled reproducibly offline, so export keeps
  its stack position as deterministic black.
- **Spout output (Windows):** enable the `collide-o-scope` sender to feed OBS,
  Resolume, MadMapper, or another Spout host. `cargo run --bin spout_probe`
  exercises the output receiver path when a real sender is running.
- **Fullscreen output:** press `O` or use the OUTPUT control. A second monitor
  is preferred when available; on a single-monitor system, Output promotes the
  existing main preview to a clean fullscreen audience surface. Window/surface
  creation failures are returned to the panel instead of leaving its switch in
  a false-open state.
- **Startup recovery:** the initial visual is metadata-probed under Safe policy
  before it can choose preview dimensions. A rejected probe uses 1280×720, and
  a source-sized renderer initialization failure receives one 1280×720 retry;
  the RECOVERY strip and browser snapshot surface the recoverable status.
- **Blackout:** press `B` or use the panel button. Preview, output window,
  Spout, and NTSC/readback consumers receive the same black frame.

## Offline render and audio

The exporter derives clock-driven modulation, slew, pad spring, temporal
history, and morph glide from frame number and the selected FPS. Live audio and
MIDI input sources read zero offline; a selected imported analysis clip is
sampled at that exact frame time. Live Spout layers render as black placeholders.
For selective VHS, export uses the same contributing-layer plan, conditional
master processing, inherited-only VHS, straight-alpha stack composition, and
post-composite Temporal order as live rendering, but evaluates it synchronously
for each output frame. The legacy no-bypass and VHS-off paths remain unchanged.

Expert media mode may admit a larger saved source during offline source
reconstruction, under the same host-local reservation policy. It does not raise
the renderer, fullscreen-output, or export-output UHD-area limits, and it does
not relax the separate 320 MiB selective-VHS working-set budget.

When VHS is enabled, **VHS quality** selects its spatial processing resolution
for both global and selective export. **Live parity (half)** is the default and
matches the real-time path by processing at half width and height, then
upscaling. **Native (full resolution)** processes at the selected export size;
it avoids that downscale/upscale step but is slower and more memory-intensive.
This choice does not change Bypass Master FX routing or pipeline order.

An optional video layer can supply its first audio stream. That audio starts at
source time zero, plays once at 1×, and is independent of visual pause, speed,
modulation, and looping. It is padded with silence or trimmed to the requested
program duration, then muxed as AAC. This explicit policy avoids implying that
arbitrarily modulated visual transport can be represented by one audio tempo.

## Patches

`Ctrl+S` saves a YAML patch through a native dialog. The browser's **Capture
snapshot** writes a uniquely named YAML file under `patches/` without blocking
the render loop or overwriting an earlier capture. Recall has two deliberately
different paths:

- **Load snapshot…** (`Ctrl+O`) reconstructs the saved sources and layer order,
  then restores visual state, layer and program transport, both freeze states,
  modulation and input configuration, LFO state, and morph automation. The
  replacement is atomic across the visual stack and saved imported analysis
  audio: every file is resolved and the audio is fully decoded before commit,
  so a missing, invalid, or corrupt source leaves the current performance in
  place. A legacy patch with no modulation section preserves current audio state.
- **Apply look…** (`Ctrl+Shift+O`) keeps the current sources, layer identities,
  order, layer count, speed/FPS/pause, per-video loop-reroll choices, both
  freeze states, BPM, modulation, and input state. It applies master effects
  and maps each saved layer by position onto the corresponding current layer:
  direct effects/keying/pattern seed, opacity, blend, visibility, and Bypass
  Master FX. Saved NTSC and Temporal sections apply when present; a legacy
  patch that omits either leaves that current section unchanged. Extra current
  layers remain visually unchanged and extra saved layers are reported unused.
  A stack change while the picker is open rejects the transfer. An engaged
  current A/B morph is materialized and cleared before the look is applied;
  the patch's saved morph is not imported.

After a successful Apply Look, actions that could overwrite its applied master,
mapped-layer, reroll, topology, and present NTSC/Temporal scope are discarded,
including conflicting input queued while the native picker was open. Unrelated
transport/safety actions, unmapped-layer edits, and edits to an omitted section
keep their order. Cancelling, failing, or rejecting a stale picker is not a
barrier.

`Ctrl+E` opens the native patch parameter editor; the file itself remains
ordinary YAML and can also be edited in a text editor. Current snapshots
include:

- master, per-layer, NTSC, and temporal values, including visual pattern seeds;
- layer order, visibility, pause, speed, blend, keying, master-FX bypass,
  per-video loop reroll, and stable source paths;
- Freeze Program, Freeze Media, and complete modulation state, including LFO
  sample-and-hold seeds;
- routing curves/slew, audio band count/crossovers/ceiling, gyro calibration/configuration, and XY
  configuration/current position;
- morph A/B slots, crossfader law/position, and remaining beat glide.

Old patches remain accepted through serde defaults and legacy filename/slit
axis fallbacks. New seeds default to `0` (the legacy pattern family), while
per-video loop reroll and Freeze Media default to off.

Exact patch load and offline export share one visual/imported-analysis-audio
resolver. Ordinary snapshots keep compatible path-first and nearby/library
filename fallbacks. Procedurally generated `cos-sha256://` references instead
require the candidate byte length and SHA-256 to match; a same-named different
file is not silently substituted.
The running layer keeps that persisted identity separately from its resolved
host path. Capture/save writes the identity, while UI export treats the host
path only as a candidate and re-fingerprints it against the recorded digest;
moving the file remains operationally harmless and mutating it after load is a
hard preflight error.

A successful patch load starts new topology and visual generations. Immediate
browser work, downbeat-latched actions, and the already-drained remainder of the
current action batch are cleared; temporal history, retained NTSC output, and
pending asynchronous readbacks are invalidated so neither an old command nor an
old frame can bleed into the restored world.

## Procedural patch generation

The patch-only generator creates a deterministic, reviewable sequence without
starting GPU exports:

```powershell
target\release\collide-o-scope.exe generate `
  --anchor patches\anchor.yaml `
  --output generated `
  --library path\to\media `
  --count 10 `
  --temperature 0.5 `
  --seed 424242 `
  --max-fingerprint-bytes 68719476736
```

Generator v2 resolves and SHA-256 fingerprints visual and imported
analysis-audio files before it commits output. It replaces private local paths
with `cos-sha256://<digest>/<bytes>` references, so canonical anchor/piece hashes
and lineage are independent of the host root and change when source bytes do.
`--library` is optional; `--max-fingerprint-bytes` defaults to 64 GiB and bounds
one invocation. `--allow-unverified-sources` explicitly permits a logical-name
fallback with an incomplete-identity warning.

Each new piece directory contains `patch.yaml`, schema-v2 `manifest.json`, and
deterministic `preflight.json`. The receipt records source digests, byte/search
limits, warnings, and a narrow configuration/source-byte claim; it explicitly
does not claim rendered-pixel identity. Generation uses typed, reflected,
mean-reverting mutations, including Shift; preserves source/layer/routing
topology; rejects active two-slot morphs and in-flight glides; and requires
explicit `--allow-black-sources` before accepting live Spout layers. The three
files are committed atomically per piece and never overwritten.

Generation still does not render MP4 batches. Clip-statistics work remains
deferred pending a bounded analysis/cache design, and visual-parameter-driven
audio DSP remains research-gated. See
[procedural video generation](docs/blogs/procedural-video-generation.md) for
the mutation design, shared source resolver, reproducibility boundary, and
remaining research trajectory.

## Build

### Windows

```powershell
winget install -e --id Gyan.FFmpeg.Shared --version 8.1.2
winget install -e --id LLVM.LLVM
# plus Visual Studio 2022 "Desktop development with C++"
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1
```

### macOS / Linux

```sh
brew install ffmpeg   # or apt: libav*-dev clang pkg-config (ffmpeg 8.x)
cargo build
```

Spout input/output is Windows-only. The common APIs report an unavailable
status instead of pretending to provide Spout on other platforms.

## Run

```sh
# Open with a video file (its parent becomes the library)
cargo run -- path/to/clip.mp4

# Open with a folder
cargo run -- path/to/clips/

# No arguments: use/create ./videos as the initial library
cargo run
```

The control panel opens in the default browser. If automatic opening fails, use
the current authenticated **Open Panel** link in the native **RECOVERY** strip.
Use **Choose Library** there to select another folder without restarting; a
cancelled chooser leaves the current library and program state unchanged.

## Keyboard

| Key | Action |
|---|---|
| Space | Pause/resume selected layer; with no selected layer, toggle Freeze Program |
| M | Toggle Freeze Media |
| F | Toggle main-window fullscreen |
| O | Toggle fullscreen output window |
| B | Blackout/unblackout |
| P / Shift+P | Increase/decrease pixelate |
| G / Shift+G | Increase/decrease RGB split |
| 0 | Reset effects |
| Ctrl+E | Toggle patch parameter editor |
| Ctrl+S / Ctrl+O / Ctrl+Shift+O | Save / Load snapshot / Apply look |
| Esc | Quit or close output window as appropriate |

## Validation boundary

Software checks do not replace physical validation. MIDI controllers and MIDI
clock, real phone sensors, venue audio hardware, external Spout applications,
and multi-monitor stage output require tests on the corresponding equipment.
Do not treat a successful build as hardware proof.

## Publication and license boundary

The MIT grant in [LICENSE](LICENSE) applies only to the modifications and
additions identified there. The original upstream code did not carry a blanket
license when this fork was made. Publication or distribution of the combined
fork is therefore conditional on the publisher having the needed authorization
for the original portions (or on a later upstream license that permits it).
This project notice records the boundary; it is not legal advice.

## Credits

- [Luis Queral](https://github.com/luismqueral) — original
  collide-o-scope engine, effects, and vision.
- [ntsc-rs](https://github.com/ntsc-rs/ntsc-rs) — VHS signal simulation.
- Fork development with AI-assisted review and implementation.
