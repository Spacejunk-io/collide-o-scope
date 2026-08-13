# collide-o-scope

A native VDJ (video DJ) instrument for live visual performance. It composites
video and live Spout sources with GPU effects, then lets LFOs, audio analysis,
MIDI, and a phone's touch and tilt modulate the performance from a browser
control panel.

> This is a fork of [collide-o-scope by Luis Queral](https://github.com/luismqueral/collide-o-scope)
> ([queral.studio](https://queral.studio)). The original engine, compositing
> architecture, and effect suite are his work. This fork adds the modulation
> matrix, remote control, audio/MIDI reactivity, temporal effects, Spout I/O,
> offline export, and Windows support. See [LICENSE](LICENSE) for the precise
> licensing and attribution boundary.

## What it does

- Composites up to 16 video or live Spout layers with normal, screen,
  multiply, and difference blending.
- Applies per-layer and master pixelation, RGB split, hue, saturation,
  brightness, contrast, posterize, invert, film grain, vignette, color drift,
  breathing motion, and luma keying.
- Runs [ntsc-rs](https://github.com/ntsc-rs/ntsc-rs) VHS simulation on a
  nonblocking worker.
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
- Renders MP4 files through a frame-indexed offline path with repeatability,
  cancellation, and stream-contract regression coverage.

## The modulation matrix

Every expressive input follows one rule: sources enter the matrix, routings
shape and scale them, and the resulting offsets are applied to per-frame
copies. Base values—the values shown by sliders and stored in patches—remain
unchanged.

| Sources | Detail |
|---|---|
| 4 LFOs | Sine, triangle, saw, square, or deterministic sample-and-hold; musical divisions; tap-tempo or MIDI clock |
| Audio | Level; 3–8 configurable bands; legacy bass/mid/high aliases; onset, spectral brightness, noisiness; and a 32-bin display spectrum |
| MIDI | Four MIDI-learn CC slots plus 24-PPQN clock/start handling |
| Phone | Calibrated yaw/pitch/roll and a multitouch XY pad |

Each routing supports signed depth, Linear/Exp/Log/SCurve/Steps response, curve
amount, and separate attack/release slew. The matrix exposes every continuous
master, NTSC, and temporal parameter; morph position; and opacity, speed, key,
and all continuous effects for each of 16 layers.

Phone input is configurable at the engine rather than being a one-off browser
effect:

- Gyro axes have **Zero here** calibration, range, exponential response, and
  invert controls.
- XY axes have independent curves and step quantization. Optional spring
  return moves the released pad back to center at a configurable rate.

For pad quantization, a value of N from 2 through 64 means exactly N evenly
spaced positions, including both endpoints; 0 or 1 disables quantization.

Live audio keeps three states distinct: the saved/requested device preference,
the device actually backing the stream, and whether that stream is the system-
default fallback because a named device disappeared. A failed or stalled
stream is stopped, its modulation sources return to zero, and the enable state
returns to off instead of retrying every frame.

The optional beat latch coalesces eligible control changes and releases them
on the next four-beat downbeat. The morph section supports linear or
equal-power interpolation plus beat-duration glides to A or B; slots, blend
law, position, and any remaining glide are patch-persistent.

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
effect values are also modulation targets. All connected panels receive the
same engine state.

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
history, and morph glide from frame number and the selected FPS. Audio and MIDI
input sources read zero offline; live Spout layers render as black placeholders.

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
- layer order, visibility, pause, speed, blend, keying, and stable source path;
- master pause and complete modulation state;
- routing curves/slew, audio band count/crossovers/ceiling, gyro calibration/configuration, and XY
  configuration/current position;
- morph A/B slots, crossfader law/position, and remaining beat glide.

Old patches remain accepted through serde defaults and legacy filename/slit
axis fallbacks.

A successful patch load starts new topology and visual generations. Immediate
browser work and downbeat-latched actions from the prior patch are cleared;
temporal history, retained NTSC output, and pending asynchronous readbacks are
invalidated so neither an old command nor an old frame can bleed into the
restored world.

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

Software tests cover the browser, GPU, export, media, and protocol paths.
Physical MIDI controllers and MIDI clock, real phone sensors, venue audio
hardware, external Spout applications, and multi-monitor stage output still
require tests on the corresponding equipment; do not treat a successful build
as hardware proof.

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
