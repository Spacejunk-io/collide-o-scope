# Collide-o-Scope

Native Rust VDJ (video DJ) performance tool. Plays video clips with real-time GPU effects for live visual performance, controlled from a browser-based panel.

## Stack

- **winit** — window management (fullscreen, input events)
- **wgpu** (v29) — GPU rendering via Vulkan / DX12 / Metal
- **ffmpeg-next** (v8, must match system ffmpeg 8.x) — video decoding
- **ntsc-rs** — analog VHS emulation (CPU, runs on a worker thread)
- **axum + tokio** — web control panel server (HTTP + WebSocket, port 3030)
- **egui** — displays the video output in the native window (panel UI moved to web)
- **bytemuck** — zero-cost casting for GPU uniform buffers

Future: midir (MIDI controller), cpal (audio reactivity)

## Module layout

```
src/
├── main.rs             — winit event loop, frame timing, app state, web action handling
├── renderer/state.rs   — wgpu setup, pipelines, composite textures, async readback slots
├── video/decoder.rs    — ffmpeg frame extraction, YUV→RGBA, looping (synchronous core)
├── video/threaded.rs   — per-layer decode thread + bounded channel (used by live path)
├── layers/mod.rs       — Layer struct (decoder + texture + blend/opacity/speed)
├── effects/params.rs   — EffectUniforms struct, parameter adjustments
├── modulation/mod.rs   — mod matrix: BPM clock (tap tempo), 4 LFOs, routings
├── audio/mod.rs        — cpal input capture + FFT band/onset analysis (mod source)
├── midi/mod.rs         — midir CC input, 4 learnable slots (mod source)
├── ntsc/mod.rs         — ntsc-rs wrapper (NtscParams, NtscState) + NtscWorker thread
├── patch/              — YAML patch save/load + editor state
├── render_export.rs    — offline high-quality MP4 export (own decoders, own NtscState)
├── web/                — axum server, WebSocket state sync, embedded static files
├── input/keyboard.rs   — key→action mapping
└── shaders/
    ├── fullscreen.wgsl — vertex shader (fullscreen triangle, no VBO)
    ├── effects.wgsl    — combined per-layer/master effects fragment shader
    └── composite.wgsl  — layer blend (normal/screen/multiply/difference)
```

## Build & run

### Windows

One-time setup:

```powershell
winget install -e --id Gyan.FFmpeg.Shared --version 8.1.2
winget install -e --id LLVM.LLVM
# plus Visual Studio 2022 "Desktop development with C++" workload
```

Then build with the helper (locates ffmpeg/LLVM/vcvars automatically):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1        # debug
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1 -Release
```

The MSVC env (vcvars64) is only required when `ffmpeg-sys-next` regenerates
bindings (first build / after `cargo clean`); afterwards plain `cargo build`
works. Runtime needs the ffmpeg DLLs on PATH — the winget install adds them.

### macOS / Linux

```sh
brew install ffmpeg          # macOS (ffmpeg 8.x)
# apt: libavcodec-dev libavformat-dev libavutil-dev libswscale-dev clang pkg-config
cargo build
cargo run -- videos/some-file.mp4
```

### Fonts

Optional: drop `IBMPlexSans-Regular.otf` / `IBMPlexMono-Regular.otf` into
`assets/fonts/` to use them in the native window. Missing fonts fall back to
egui's built-ins — never a build error.

## Threading architecture

- **Render thread** (winit event loop): GPU passes, egui, web action handling.
  Never blocks on decode or NTSC — only non-blocking `try_recv`s.
- **Decode threads** (one per layer, `video/threaded.rs`): ffmpeg decode +
  YUV→RGBA into a bounded channel (depth 2). Backpressure paces them; loop-point
  reopen hitches stay off the render thread. Layer removal drops the receiver,
  which exits the thread.
- **NTSC worker** (`ntsc/mod.rs::NtscWorker`): receives composite readbacks,
  applies ntsc-rs, sends processed frames back. At most one job in flight;
  excess frames are dropped, never queued.
- **Async readback** (`renderer/state.rs`): up to 3 staging-buffer slots.
  Per frame when NTSC enabled: encode copy → submit → `map_async` (no wait);
  completed maps are harvested on later frames. NTSC output trails live by
  ~2 frames — imperceptible for a VHS look.
- **Web server thread**: tokio runtime with axum; actions queue into
  `WebState.actions`, drained once per frame tick by the render thread.
  Binds 0.0.0.0:3030 (HTTP) and 0.0.0.0:3031 (HTTPS, self-signed via
  rcgen) for phone remote control. The QR points at HTTPS — iOS only
  exposes motion sensors to secure contexts; the phone accepts the
  certificate warning once. The cert (SANs: localhost + LAN IP) persists
  under %LOCALAPPDATA%\collide-o-scope\tls so that trust survives
  restarts; it regenerates only when the LAN IP changes. app.js picks
  ws:/wss: from the page protocol. Access: loopback is free; LAN clients
  need the per-session token (8 hex chars, OS entropy) via `?key=` —
  carried by the QR at `/qr.svg` — after which an HttpOnly cookie keeps
  them in. Unknown LAN clients get 403. The panel is responsive: under
  900px it stacks single-column with touch-sized controls.
- **Thumbnail thread**: shells out to ffmpeg CLI for library thumbs/previews.

## Keyboard controls

- P / Shift+P — increase / decrease pixelate
- G / Shift+G — increase / decrease RGB split
- 0 — reset all effects
- Space — pause/resume
- F — toggle fullscreen
- O — toggle the fullscreen output window (second display)
- B — blackout (cut output to black; also a panel button)
- Ctrl+E — toggle YAML editor · Ctrl+S — save patch · Ctrl+O — load patch
- Escape — quit

## Architecture notes

- Single combined effect shader (uniform-driven, no pipeline switching)
- Fullscreen triangle drawn with 3 vertices — UVs computed from vertex_index
- 3-texture composite chain: effects → [1], blend [0]+[1] → [2], copy → [0]
- Frame timing: render loop gated at 30fps; per-layer decode paced by fps × speed
- Web panel state is broadcast as a full JSON snapshot each frame tick
- Modulation matrix: sources (4 LFOs — sine/tri/saw/square/S&H, rates in
  beats, tap-tempo BPM clock — plus audio level/bass/mid/high/onset and
  MIDI slots A–D) route to master-effect and NTSC params via
  `modulation::TARGETS`. Modulation is applied to *copies* each frame — UI
  sliders edit base values, which are never mutated. New sources (gyro,
  MIDI clock) should be added as `ModSource` variants, not as bespoke
  per-effect wiring.
- MIDI: midir connects to the first input port; CC messages on any channel
  fill a lock-free 128-entry value table. Four slots (A–D) each bind a CC
  number — editable directly or via MIDI learn (arm a slot, twist a knob).
  No port is a soft failure, same pattern as audio.
- MIDI clock sync: 0xF8 timing pulses (24 ppqn) drive BPM (EMA of pulse
  interval) and beat position (pulses/24); 0xFA Start rewinds to beat 0.
  While pulses arrive the external clock owns the beat; when they stop
  (>1s), the internal clock resumes from the same position.
- Layer modulation: targets layerN_opacity / layerN_speed / layerN_key
  (N=1..4) modulate per-layer values — audio-driven crossfades, beat-synced
  speed, keyed shapes breathing in and out. Applied via per-frame modulated
  copies passed to render_layers; bases untouched. Same math in export.
- Luma key (per layer): key_mode (off/bright/dark) + threshold + softness
  in EffectUniforms (fills former padding, still 96 bytes). The shader
  carves alpha by smoothstepped luminance and premultiplies rgb, so keyed
  shapes composite correctly on any blend mode and fade to black on the
  bottom layer.
- Spectral character sources: audio_bright (centroid — sine bass low, saw
  lead high) and audio_noise (flatness — tonal 0, percussive/noisy 1)
  describe *what kind* of sound is playing, independent of loudness; gain
  is not applied and silence gates them to 0.
- Gyroscope sources: the web panel's GYRO group streams DeviceOrientation
  from whichever device enables it (~30Hz over the existing WS) as
  gyro_yaw/pitch/roll, unipolar 0..1 with 0.5 = level. iOS needs HTTPS for
  sensors; the UI reports this. Last value holds if the stream stops.
- XY pad sources: the panel's XY PAD group (pointer events, multitouch-safe
  via pointer capture) streams pad_x/pad_y at ~30Hz; position holds on
  release like a hardware pad, and syncs across clients when untouched.
- Licensing: upstream (luismqueral/collide-o-scope) has NO license — the
  original code is © Luis Queral, all rights reserved. LICENSE here grants
  MIT over this fork's additions only and says so explicitly. Do not claim
  MIT over the original portions unless Luis licenses upstream.
- Temporal effects (renderer + shaders/temporal.wgsl): TWO memories, not
  one. The 24-frame ring records CLEAN pre-temporal composites — slit-scan
  reads real past frames there (recording post-effect output made it eat
  its own black). A separate feedback texture holds last frame's
  post-temporal output so trails compound. Both record every frame even
  when off, so the effects are warm instantly. Params are mod targets
  (temporal_*), persist in patches, and render identically in export via
  shared helpers (renderer::state::{build_temporal_pipeline,
  encode_temporal, build_history_texture, build_feedback_texture}).
- Effects audit harness: `cargo test effects_audit -- --ignored` renders
  every effect (17 labeled clips) through the real export path into
  renders/audit_*.mp4 for objective ffprobe verification (needs GPU,
  ffmpeg on PATH, and videos/audit.mp4 — any short colorful clip).
  This audit caught the slit-scan self-cannibalization and proved the
  bottom-layer opacity fix (0.3 opacity → exact 0.3 in linear light).
- Output window (keyboard O, panel OUTPUT group, or web action): a second
  winit window blits the final composite fullscreen — on the second
  monitor when one exists, letterboxed, Escape/O closes. Surface
  capabilities MUST be queried against the stored adapter (Renderer keeps
  instance + adapter); a freshly requested adapter handle invalidates the
  device.
- Patch morph (panel MORPH group): slots A/B capture the continuous
  performance state (master fx, NTSC, temporal, per-layer
  opacity/speed/key); while both are set the crossfader writes the BASE
  params as their interpolation each frame — sliders visibly follow, and
  the mod matrix breathes on top. Discrete values switch at t=0.5. The
  "morph" mod target lets an LFO/knob sweep between worlds
  (ModMatrix::target_offset).
- Panel layout: fixed cluster columns (.fx-columns/.fx-col — video
  effects | matrix+morph | sources | I/O), NOT CSS masonry: expanding a
  group must never reflow other columns mid-performance.
- Spout output (spout_out.rs, spout2-rs crate): a worker thread owns a DX11
  sender named "collide-o-scope"; frames come from the same async readback
  the NTSC path uses (raw composite, or NTSC-processed when VHS is on — 
  always what the audience sees). Bounded queue, drop-don't-block.
- build.rs embeds a ComCtl32 v6 manifest — the Spout SDK imports
  TaskDialogIndirect (ordinal 345, v6-only); without the manifest the exe
  dies at load with STATUS_ORDINAL_NOT_FOUND before main().
- `cargo run --bin spout_probe` verifies Spout end-to-end (connects to the
  running sender, checks frames are non-black).
- Audio: cpal default input → mono ring buffer → per-frame 1024-pt FFT.
  Bands are adaptively normalized (slow-decay running peak) so levels are
  0..1 regardless of input loudness; onset = spectral flux with instant
  attack and exponential decay. LFOs are bipolar [-1,1]; audio is [0,1].
  No input device is a soft failure: error surfaces in the panel, levels
  read 0, and the enable toggle flips back off.
- Patches (YAML, Ctrl+S/Ctrl+O) persist the mod matrix (`modulation:`
  section — BPM, LFOs, routings, audio enable/gain, MIDI enable/CC slots).
  Old patches without the section load fine and leave the matrix untouched.
- Offline export renders modulation deterministically: beat is derived from
  the frame index (`time × bpm/60`), so the same patch produces an
  identical file every run (verified via framemd5). Audio and MIDI sources
  read 0 offline — only clock-driven sources (LFOs) animate in exports.

## Known gotchas

- `ffmpeg-next` major version must match the installed ffmpeg major (currently 8)
- On Windows, bindgen needs LIBCLANG_PATH and the MSVC includes (vcvars) — use
  `scripts/build-windows.ps1`
- The `block` crate emits future-incompat warnings (upstream issue, harmless)
- `render_export.rs` deliberately uses the synchronous `VideoDecoder` and its
  own `NtscState` — offline export wants determinism, not throughput
