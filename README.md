# collide-o-scope

A native VDJ (video DJ) instrument for live visual performance. Plays and
layers video clips with real-time GPU effects, driven by a modulation matrix
that routes LFOs, audio analysis, MIDI, and a phone's touch and tilt to any
parameter — controlled from a browser panel that works on the desktop and,
via QR code, on a phone over the LAN.

> This is a fork of [collide-o-scope by Luis Queral](https://github.com/luismqueral/collide-o-scope)
> ([queral.studio](https://queral.studio)) — the original engine, compositing
> architecture, and effect suite are his work. This fork adds the modulation
> matrix, remote control, audio/MIDI reactivity, temporal effects, and
> Windows support. See [LICENSE](LICENSE) for how the two are credited.

## What it does

- Multi-layer video compositing with blend modes (normal, screen, multiply, difference)
- Per-layer and master effects: pixelate, RGB split, hue/saturation/brightness/contrast,
  posterize, invert, film grain (4 noise algorithms), vignette, color drift, breathing
- **Per-layer luma keying** — carve shapes out of layers by brightness, with
  modulatable threshold so keyed shapes breathe in and out of frame
- **VHS simulation** via [ntsc-rs](https://github.com/ntsc-rs/ntsc-rs) (head switching,
  tracking noise, snow, chroma loss…), running on a worker thread
- **Temporal effects** from a 24-frame history ring: feedback trails
  (zoom/rotate tunnel) and slit-scan time-warping
- **Offline export** to MP4 — deterministic: the same patch renders the same
  file every time, modulation included

## The modulation matrix

Every expressive source routes to any of 30+ targets with signed depth.
Base values (what the sliders say) are never mutated — modulation breathes
around them.

| Sources | |
|---|---|
| 4 LFOs | sine / triangle / saw / square / sample&hold, rates in musical divisions, tap-tempo BPM clock, MIDI clock sync (24 ppqn) |
| Audio | level, bass/mid/high bands, transient onset, spectral brightness & noisiness (FFT, adaptively normalized) |
| MIDI | 4 CC slots with MIDI-learn (twist a knob to bind) |
| Phone | gyroscope yaw/pitch/roll, multitouch XY pad |

Targets include every master effect, VHS parameters, temporal feedback and
slit-scan, and per-layer opacity / speed / key threshold — so a bass line can
crossfade layers while a hardware knob rides the time-warp.

## Remote control

The app serves a control panel at `http://127.0.0.1:3030` (and HTTPS on
`:3031` with a self-signed certificate). A QR code in the panel's REMOTE
section carries a per-session access token — scan it with a phone on the
same network, accept the certificate warning once, and the panel opens
mobile-first with touch-sized controls, the XY pad, and gyroscope streaming
(HTTPS is what unlocks iOS motion sensors). Unknown LAN clients get 403.

## Output

- **Spout** sender (`collide-o-scope`) on Windows — pipe the composite
  straight into OBS, Resolume, or MadMapper. Verify with
  `cargo run --bin spout_probe`.
- Patches (full performance state: effects, layers, matrix, temporal) save
  and load as YAML with Ctrl+S / Ctrl+O.

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

## Run

```sh
# Open with a video file (parent folder becomes library)
cargo run -- path/to/clip.mp4

# Open with a folder (browse clips in library panel)
cargo run -- path/to/clips/

# No args — drag and drop files/folders onto the window
cargo run
```

The control panel opens in your browser automatically.

## Keyboard

| Key | Action |
|-----|--------|
| Space | Pause/resume selected layer |
| F | Toggle fullscreen |
| P / Shift+P | Increase/decrease pixelate |
| G / Shift+G | Increase/decrease RGB split |
| 0 | Reset effects |
| Ctrl+S / Ctrl+O | Save / load patch |
| Esc | Quit |

## Architecture notes

Render thread never blocks: decoding runs on a thread per layer (bounded
channels), ntsc-rs and Spout on workers fed by async GPU readbacks, and the
NTSC output trails live by ~2 frames — invisible for a VHS look. See
[CLAUDE.md](CLAUDE.md) for the full threading and module map.

## Credits

- **[Luis Queral](https://github.com/luismqueral)** — the original
  collide-o-scope: engine, compositing, effects, and the vision.
- **[ntsc-rs](https://github.com/ntsc-rs/ntsc-rs)** — VHS signal simulation.
- Fork development with [Claude Code](https://claude.com/claude-code).
