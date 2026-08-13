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

## What it does

- Composites up to 16 video, PNG/JPEG/BMP/WebP still, or live Spout layers with normal, screen,
  multiply, and difference blending.
- Applies per-layer and master pixelation, RGB split, hue, saturation,
  brightness, contrast, posterize, invert, film grain, vignette, color drift,
  breathing motion, luminance/chroma keying, and bounded animated
  cellular/Worley warping with a separately feathered cell-gap key.
- Runs [ntsc-rs](https://github.com/ntsc-rs/ntsc-rs) VHS simulation without
  blocking the render thread. Existing all-inherited stacks keep the global
  post-composite path; a contributing **Bypass Master FX** layer selects an
  exact per-layer path so VHS touches inherited layers only.
- Provides feedback trails and arbitrary-angle slit-scan from a 24-sample
  temporal history. History advances at a fixed 30 Hz, so its approximately
  0.8-second span does not change with display or export frame rate.
- Saves and restores complete YAML performance patches, including stable
  source identities, layer pause/speed, master pause, modulation/input
  configuration, temporal and NTSC state, and morph slots/glides.
- Keeps live decoding request-driven and bounded: each layer's worker performs
  only transport advances requested by the engine and publishes the newest
  completed image through a one-frame overwrite slot. A first-frame seed is
  decoded at open time, so a clip added or restored while paused still has a
  defined image.
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

The matrix exposes every continuous master and NTSC value, temporal feedback,
slit-scan and history-key values, morph position, and each layer's opacity,
speed, target FPS, key controls, and continuous effects. This includes RGB
chroma targets/tolerance, key thresholds and softness, temporal key history,
and VHS edge-wave speed, tracking wave, composite/chroma noise, luma smear,
and sharpening. Selector choices such as static/temporal key mode, blend mode,
and grain algorithm remain deliberate discrete controls and are not modulation
targets. The legacy patch target `layerN_key` is read as the canonical
`layerN_key_threshold`; if both spellings occur, the canonical route wins.

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
live and offline; Pause holds that timestamp exactly.

The optional beat latch coalesces eligible control changes and releases them
on the next four-beat downbeat. The morph section supports linear or
equal-power interpolation plus beat-duration glides to A or B. A capture is an
ordering barrier: it first materializes the current Morph result, including a
Morph-target routing offset, then records the bases at the current layer-stack
revision. Stale captures are rejected. Manual fader and law edits still apply
to the materialized bases while Pause holds automatic glide and clock motion.
Pause also snapshots the exact audience across a blackout: the cut remains
absolutely black while active, then releasing it while still paused restores
the pre-cut image. A paused selective-VHS transition remains held until Resume
can produce a complete replacement. Hue and slit angle take the shortest wrapped arc,
discrete choices switch at the midpoint, and stored layer slots follow
reorder/removal while a newly appended layer remains untouched.
Slots, law, position, and the exact remaining glide—even below the UI's
quarter-beat minimum—are patch-persistent. Other modulation offsets remain
frame-local and do not rewrite the captured bases.

## Browser control

The app opens a per-session desktop URL of the form
`http://127.0.0.1:3030/?key=<session-token>` and serves the tokenized HTTPS
phone panel on `:3031`. The token is required on loopback as well as the LAN,
is exchanged for a strict HttpOnly session cookie, and is then removed from
the visible address bar. WebSocket and mutating HTTP requests must also be
same-origin. The REMOTE section shows the current tokenized QR code. HTTPS is
required for iOS motion permission; a bare, stale, unauthenticated, or
cross-origin control request receives 403.

The panel is mobile-first below 900 px. Touch controls include pointer-safe XY
input, layer drag-to-reorder, group resets, and double-tap/double-click reset
for individual sliders. Layer cards expose direct transport, target decode
FPS, keying, and the complete per-layer effect set, including downsample; those
effect values are also modulation targets. The **Layer effects** disclosure and
its nested **CELLULAR** disclosure start closed for each new layer card and
remember their state by layer identity while that layer remains present. All
connected panels receive the same engine state.

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
layer nor master **Revert** changes it.

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

- **Video layers:** add files from the library or drag/drop. Patches retain a
  stable path and fall back to the library filename for older patches.
- **Spout input (Windows):** enter an exact sender name and choose **Add
  live**. The receiver is an ordinary composited layer with transport/effect
  controls and a visible connection status. A missing or warming sender stays
  black. Live Spout cannot be sampled reproducibly offline, so export keeps
  its stack position as deterministic black.
- **Spout output (Windows):** enable the `collide-o-scope` sender to feed OBS,
  Resolume, MadMapper, or another Spout host. `cargo run --bin spout_probe`
  exercises the output receiver path when a real sender is running.
- **Fullscreen output:** press `O` or use the OUTPUT control. A second monitor
  is preferred when available. Window/surface creation failures are returned
  to the panel instead of leaving its switch in a false-open state.
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

An optional video layer can supply its first audio stream. That audio starts at
source time zero, plays once at 1×, and is independent of visual pause, speed,
modulation, and looping. It is padded with silence or trimmed to the requested
program duration, then muxed as AAC. This explicit policy avoids implying that
arbitrarily modulated visual transport can be represented by one audio tempo.

## Patches

`Ctrl+S` saves and `Ctrl+O` loads a YAML patch. `Ctrl+E` opens the native patch
parameter editor; the saved file itself is ordinary YAML and may also be
edited in a text editor. Current patches include:

- master, per-layer, NTSC, and temporal values;
- layer order, visibility, pause, speed, blend, keying, master-FX bypass, and stable source path;
- master pause and complete modulation state;
- routing curves/slew, audio band count/crossovers/ceiling, gyro calibration/configuration, and XY
  configuration/current position;
- morph A/B slots, crossfader law/position, and remaining beat glide.

Old patches remain accepted through serde defaults and legacy filename/slit
axis fallbacks.

The browser's **Capture patch** control writes a unique YAML snapshot under
`patches/` through a bounded background writer. Existing captures are never
overwritten and the render loop does not wait for disk I/O.

A successful patch load starts new topology and visual generations. Immediate
browser work and downbeat-latched actions from the prior patch are cleared;
temporal history, retained NTSC output, and pending asynchronous readbacks are
invalidated so neither an old command nor an old frame can bleed into the
restored world.

## Procedural patch generation

The patch-only generator creates a deterministic, reviewable sequence without
starting GPU exports:

```powershell
target\release\collide-o-scope.exe generate `
  --anchor patches\anchor.yaml `
  --output generated `
  --count 10 `
  --temperature 0.5 `
  --seed 424242
```

Each new piece directory contains `patch.yaml` and `manifest.json`. Generation
uses typed, reflected, mean-reverting mutations; preserves source/layer/routing
topology; rejects active two-slot morphs and in-flight glides; and requires explicit
`--allow-black-sources` before accepting live Spout layers. Output directories
are committed atomically and never overwritten. See
[procedural video generation](docs/blogs/procedural-video-generation.md) for
the mathematical design, cellular effect, reproducibility boundary, and
deliberately deferred clip-analysis/audio-DSP work.

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

# No arguments: drag and drop files/folders after launch
cargo run
```

The control panel opens in the default browser.

## Keyboard

| Key | Action |
|---|---|
| Space | Pause/resume selected layer |
| F | Toggle main-window fullscreen |
| O | Toggle fullscreen output window |
| B | Blackout/unblackout |
| P / Shift+P | Increase/decrease pixelate |
| G / Shift+G | Increase/decrease RGB split |
| 0 | Reset effects |
| Ctrl+E | Toggle patch parameter editor |
| Ctrl+S / Ctrl+O | Save/load patch |
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
