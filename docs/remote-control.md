# Connecting to the control panel

Collide-o-scope serves plain HTTP for the computer running the engine and HTTPS
for phones. iOS exposes motion sensors only to a secure page.

| Use | URL | Access |
|---|---|---|
| Desktop | `http://127.0.0.1:3030/?key=…` | Fresh per-session token opened by the app |
| Phone/other LAN device | `https://<lan-ip>:3031/?key=…` | Per-session token from the QR code |

## Desktop

1. Launch collide-o-scope. The panel should open in the default browser.
2. If it does not, choose **Open Panel** in the native **RECOVERY** strip. That
   link carries the current process's exact tokenized loopback URL. A bare
   `http://127.0.0.1:3030` URL is intentionally denied.

Loopback is not an authentication bypass. The first successful navigation
exchanges the key for an HttpOnly, `SameSite=Strict` session cookie and removes
the key from the visible address bar. WebSocket upgrades, uploads, deletes,
and other mutating requests must also originate from the exact page origin;
scripts hosted by another site receive 403 even if they target localhost.

Only one running instance can own ports 3030 and 3031. If a newly built app
appears to have an old panel, stop the earlier instance and hard-refresh the
tab; the HTML, CSS, and JavaScript are embedded in the executable.

The native preview remains a recovery surface when the browser is closed or the
listener fails. Its panel status truthfully distinguishes **not started**,
**starting**, **ready**, and **unavailable** from the separate browser count;
zero connected browsers is not a server failure. Freeze Program, Blackout, and
**Revert Visuals** dispatch directly through the engine and do not depend on a
browser queue. A second status row surfaces recoverable output and media-source
messages. This is an operator-preview surface only: the strip is hidden while
single-monitor Output owns the main surface, so audience pixels remain clean;
leaving that output mode restores it, and a dedicated audience window never
contains the strip.

## Phone

On first launch, Windows can show firewall prompts for HTTP 3030 and HTTPS
3031. Allow both on **private networks** if phones must connect.

For each app session:

1. Put the phone and PC on the same trusted Wi-Fi network.
2. Open the desktop panel's collapsed **REMOTE** section.
3. Scan its QR code. The URL includes a fresh per-session access token.
4. Accept the self-signed certificate warning once:
   - iOS Safari: **Advanced** → **Visit Website**
   - Android Chrome: **Advanced** → **Proceed**
5. The responsive, touch-sized panel loads. Under 900 px it uses one column.

The certificate persists across restarts and is regenerated when the LAN
identity changes. Authentication is session-specific: the strict cookie keeps
the phone connected during the current run, but the token rotates at the next
launch, so old bookmarks answer 403.

## Library and layers

- Use the Library **+** to upload video, PNG/JPEG/BMP/WebP stills, or supported
  audio from the current device. Uploads land in the active library folder and
  name collisions get numbered suffixes. A no-argument launch uses or creates
  `videos/` as that initial folder.
- The native **RECOVERY** strip shows the active folder and visual/audio counts.
  **Choose Library** changes the scan and browser-upload destination; cancelling
  the chooser is a no-op. **Rescan** refreshes the current folder. Neither action
  adds a layer automatically. Dialog time is removed from beat, frame, and video
  pacing without changing the visible Freeze state. A folder change clears
  basename-keyed previews and advances a generation so workers from the old
  folder cannot repopulate the new cache.
- Thumbnail and hover-preview helpers first metadata-probe each candidate under
  the current Safe/Expert source policy. Their captured output and run time are
  bounded, a folder-generation change cancels stale helpers, and Expert-mode
  helpers are serialized rather than fanning out large decodes. Generated JPEGs
  fit within 180×180 and 512 KiB; both caches share a 64 MiB retained-byte cap,
  and the helper deadline still applies after cancellation begins. The helper
  pool is process-wide and single-flight: a repeated Rescan supersedes the prior
  generation instead of multiplying its concurrency.
- Double-click/double-tap a library tile to add a video or still-image layer.
- Drag the handle on a layer card to reorder it. The browser keeps one pointer
  owner for the gesture and sends one atomic move on release.
- Expand a layer's controls to edit its target decode FPS, transport, blend,
  keying, and complete direct effect set, including downsample. The same
  continuous effect values remain available as modulation targets. Within the
  card, **Layer effects** and its nested **CELLULAR** subsection start closed;
  their disclosure state follows that layer's immutable session identity.
- Use the layer **×** to remove it. Library deletion is separate: it moves the
  source file to the OS Recycle Bin. If the OS reports the file as in use, the
  operation is left recoverably unchanged and the panel reports the conflict.
- Double-click/double-tap an individual slider to reset it. Section-level
  **reset** buttons restore the corresponding group.
- The value beside every slider is an editable numeric field. Enter commits
  (blur does too), Escape cancels, and arrow keys move by the slider step.
  Values are constrained to the same range and step as the slider.
- **Freeze Program** (the top pause button) holds the whole visual program:
  decoder and Spout images, shader/VHS time, temporal accumulation, LFO and
  morph phase, routing slew, and imported-audio analysis all resume without
  catch-up. Live input telemetry may continue updating its meters while its
  rendered modulation value remains frozen. On the native window, `Space`
  pauses the selected layer; it toggles Freeze Program only when no layer is
  selected.
- **Freeze Media** (the **MED** button or `M`) holds only decoded-video and
  Spout images. Program time, animated effects, modulation, temporal/VHS
  processing, routing slew, morphs, and imported-audio analysis continue.
  Media resumes from a fresh pacing boundary rather than catching up. The two
  freeze states are independent and both persist in a snapshot.

| Freeze Program | Freeze Media | Program clocks/effects | Video/Spout images |
|---|---|---|---|
| Off | Off | Run | Advance |
| Off | On | Run | Hold |
| On | Off or On | Hold | Hold |

If both are on, releasing Freeze Program leaves the media held until Freeze
Media is also released.

- **Revert master visual state** restores master, VHS, and temporal effects,
  resets all four LFO configurations/seeds, and clears modulation routings and
  Morph automation so those defaults stay in force. It preserves layers and
  their effects, visibility and transport,
  BPM, and audio/MIDI/device choices. A layer card's **Reset FX** remains local
  to that layer and deliberately leaves opacity and transport unchanged. The
  bundled panel sends `reset_visual_program` for this broad operation. The
  legacy `reset_fx` protocol action remains accepted but resets only direct
  master effect uniforms, matching the original remote-control contract. It
  deliberately leaves VHS, Temporal, routings, Morph, and queued automation
  live, so those systems may continue changing the rendered master afterward.
- **Bypass Master FX** on a layer skips inherited Digital, Analog, Cellular,
  Motion, and VHS processing for that layer. Its own Layer FX, opacity, key,
  and blend remain active; Temporal still affects the final program. When VHS
  is enabled and a visible, positive-opacity bypass layer contributes, direct
  master effects and VHS run only on inherited layer slices; the engine then
  recomposites the stack before Temporal. Offline render follows the same
  selective order. Hidden or non-positive-opacity layers create no selective
  work. Live selective processing has a 320 MiB safety budget and no silent
  global-VHS fallback: if the current output size and contributing stack exceed
  it, the prior exact audience frame is held and the VHS panel reports how to
  reduce the load. Layer **Reset FX** and master **Revert** preserve this
  non-destructive switch.
- A layer's Cellular **Gap Key** controls how completely Voronoi ridges are removed;
  **Gap Threshold** selects which ridge strengths open, and **Gap Softness**
  feathers the edge. Those gaps reveal layers beneath it. The master Cellular
  panel exposes the same controls: its ordinary post-stack cut resolves over
  black, while a conditional per-layer master stack can reveal content below
  inherited layers. The matrix exposes all three master and per-layer values.
- Static layer and program keying each provide Off, Keep Bright, Keep Dark,
  Remove Chroma, and Keep Chroma. Luminance modes use threshold and softness;
  chroma modes add an RGB target and tolerance. A layer key changes alpha to
  reveal lower layers. Program keying acts on the flattened image, so its cut
  pixels become black.
- Temporal **History Key** compares the current clean composite with a selected
  sample from the fixed 30 Hz history. Keep Motion, Keep Stillness, Keep
  Brightening, and Keep Darkening share threshold and softness controls; the
  History control selects 1–23 samples back. The resulting mask gates the
  temporal output after feedback and slit-scan processing.

On Windows, the top-bar **Spout input** field accepts the exact name of an
external sender. **Add live** creates a normal layer card with receiver status,
dimensions, and frame activity. A missing/warming sender remains black. A
phone can issue the command, but the actual Spout transport still runs on the
Windows engine.

Every live layer has an immutable session identity in addition to its visible
stack number. Bundled-panel layer commands use that identity, and a reorder
carries the stack revision from the panel's latest snapshot. If another
controller has added, removed, or moved a layer first, the engine rejects the
stale ID or topology revision; it does not fall back to applying the old index
to a different clip. The next engine snapshot resynchronizes the panel.
Per-layer modulation targets follow the same move permutation. Removing a
layer discards its routes and shifts higher-numbered layer targets to keep them
attached to their original logical sources.

### Media safety

The Library column exposes the current source-admission mode and its rationale.
**Safe** is the launch default and preserves the established per-source ceiling
of 8,294,400 pixels / 33,177,600 RGBA bytes—the area of 3840×2160. **Expert** is
an explicit host-session override for future video, still, and Spout source
opens. It permits at most DCI-8K area (35,389,440 pixels / 141,557,760 RGBA
bytes), intersected with the 16,384 px absolute edge and the current device's
2D-texture edge and per-buffer limits.

Above-Safe sources must also reserve a conservative aggregate planning weight:
four RGBA frames for video or Spout and six for a still. The total budget is no
greater than one eighth of detected physical RAM or 2 GiB, whichever is smaller.
This is a host-memory plan, not free-VRAM detection. Portable wgpu does not
report available VRAM headroom, so actual texture creation remains recoverable
and can still reject a source. Texture creation and source uploads are both
error-scoped; upload failure keeps the layer inactive and visible in its source
status. Safe-sized sources retain their existing path and consume no Expert
reservation.

Expert affects future allocations only. Returning to Safe does not destroy an
already accepted source, and patches cannot enable Expert on another host. It
does not raise the live renderer, fullscreen-output, or export-output UHD-area
caps, and it does not raise the separate 320 MiB selective-VHS budget. The
bundled panel sends the idempotent action
`{"action":"set_media_safety_mode","mode":"safe"}` (or `"expert"`). The additive
`media_safety` snapshot defaults to Safe when absent, so older snapshots and
clients remain compatible; the action is immediate rather than beat-latched.

## Performing controls

All connected panels receive the same engine snapshot. Immediate browser input
enters a bounded queue: repeated absolute values for the same control are
coalesced so the latest wins, while emergency and pointer-release actions keep
reserved capacity. This prevents a fast slider, sensor, or disconnected client
from building an unbounded render-thread backlog. Beat latch adds the separate
downbeat behavior below.

### Live VHS admission metrics

The VHS panel labels the current live path as global or selective and exposes
separate saturating counters for each path. `attempted` counts bounded admission
attempts and `accepted` counts work admitted by that path. `skipped` is reserved
for healthy bounded backpressure when a worker or selective staging slot is
busy; `unavailable` separately counts a disconnected or failed worker so a
fault cannot look like ordinary load shedding. `stale` is orthogonal: it counts
work that was accepted but whose asynchronous result was later rejected at a
visual-generation, topology, or path-compatibility boundary. A disabled path
generates no attempts. `busy` is current presentation context for the selected
path, not a cumulative counter; the panel reports it beside that path's own
bucket.

These process-session diagnostics describe live bounded-work decisions. They
are not an export metric, a count of every presentation drop, or proof that an
accepted frame reached an audience surface. Worker errors remain in the VHS
error status. The additive metric snapshot defaults to zero with the active path
off when absent, preserving older-client compatibility.

### Random / Dice

Randomization uses only stored deterministic seeds—never the wall clock or OS
entropy—so patches and exact-seed performances are replayable.

- **Master** changes the master shader-pattern seed and all four LFO seeds. In
  Bounded variation mode it also varies the master effect controls listed
  below.
- **Everything** does the same and includes every current layer. With an exact
  base seed, the master keeps that value while each layer position and LFO gets
  a reproducible derived stream. The panel supplies the current stack revision,
  so a concurrent add/remove/reorder rejects the whole request instead of
  targeting the wrong sources.
- Each layer card exposes its exact pattern seed and a pattern-only **Reroll**.

**Pattern only** changes seeds without moving effect controls and does not
disengage an active A/B morph. **Bounded variation** first chooses those seeds,
then makes reflected, range-safe changes to pixelate/downsample, RGB split,
hue, saturation, brightness, contrast, posterize, grain amount/size, vignette,
color drift, breathing motion, Shift, and core Cellular
amount/scale/warp/speed.
**Amount** ranges from 0 to 2. **Grain mode** additionally permits the discrete
grain algorithm and color-grain switch to change. Source identity, topology,
opacity, visibility, blend, keying, Bypass Master FX, transport, routings, VHS,
and Temporal stay unchanged. Because variation edits visual bases, a valid
variation materializes and clears an engaged A/B morph before applying.

Leave **Exact seed** blank to advance deterministically from the stored master
and targeted-layer seeds. Master and Everything always derive all four LFO
seeds from the resulting master seed; Everything also derives a positional
stream after advancing each layer's stored seed. Enter a whole number from `0`
through `4294967295` to replay a base seed; `0` deliberately restores the
legacy shader and sample-and-hold sequences. Repeating a Bounded variation also
requires the same starting effect values.

Each LFO shows its own seed field when **Sample & Hold** is selected. It holds
one deterministic value for a complete LFO cycle; changing rate changes that
hold duration. Master and Everything rerolls update all four LFO seeds, even
if another waveform is currently selected, and snapshots preserve them.

Decoded-video layers additionally offer **each loop**. At every authoritative
loop boundary the engine advances that layer's pattern seed once, including
every boundary represented by a newest-only decoder result. Stills and Spout
inputs ignore this option. The setting and seed persist in snapshots and the
offline renderer follows the same loop-boundary rule.

### Shift

Master and every layer have the same four Shift controls:

- **Amount** sets horizontal displacement, from an exact no-op at zero to a
  maximum of one quarter of the image width.
- **Block px** divides output space into horizontal bands 2–256 pixels high.
- **Density** is the seeded fraction of bands displaced in an epoch.
- **Speed** advances deterministic epochs from program time; zero holds one
  arrangement.

Displaced bands wrap horizontally, so Shift never creates an uncovered edge.
The arrangement is a function of the stored pattern seed, output-space band,
and epoch. Pattern-only Dice and per-layer **Reroll** therefore rearrange Shift
without moving its four controls; Bounded variation can change both seed and
controls. Freeze Program holds the epoch, whereas Freeze Media holds source
images but lets the Shift clock continue. The four values persist in snapshots,
interpolate through Morph, appear as master and dynamic `layerN_shift_*`
modulation targets, and use the same effect shader in offline export.

### Beat latch

Enable **Next downbeat** in the timing/modulation area to defer eligible
parameter, layer, NTSC, temporal, morph, capture, clear, and glide actions.
Pending actions coalesce by control and are released together when the next
four-beat bar begins. The status shows the number waiting. Emergency actions
such as blackout remain immediate.

### XY pad

The pad streams pointer position at approximately 30 Hz. Route `Pad X` or
`Pad Y` to any target in the matrix. Routing values are bipolar: the center is
zero, the left/bottom side is negative, and the right/top side is positive.

- Each axis can use Linear, Exp, Log, SCurve, or Steps response.
- Each axis has its own curve amount and position count. A setting of N from 2
  through 64 produces exactly N evenly spaced positions including 0 and 1;
  0 or 1 disables quantization.
- With **Spring** off, the last position holds after release.
- With **Spring** on, the engine returns the released pad toward center at the
  configured rate. This continues independently of the browser pointer.

Pointer capture, cancellation, page blur, and visibility loss all send a
release, preventing a disconnected client from leaving the pad falsely held.

### Gyroscope

On the phone that should steer, enable **Stream** and grant motion permission.
The three matrix sources are `Gyro Yaw`, `Gyro Pitch`, and `Gyro Roll`.

- **Zero here** records the current orientation as the normalized center.
- **Range** controls the degrees required for full swing.
- **Expo** changes center sensitivity.
- **Invert** reverses an axis.

Calibration and per-axis configuration live in the engine and persist in a
patch. Each calibrated center is zero in the bipolar modulation path; travel
to either side produces negative or positive values. Yaw is meaningful
relative to calibration because compass fusion can drift. Use the QR's HTTPS
URL on iOS; plain HTTP never offers sensor permission.

### Routing response

Every route row supports signed depth, response curve and amount, plus
independent attack/release slew. These controls apply equally to LFO, audio,
MIDI, gyro, and pad sources. The target menu covers master/NTSC/temporal/morph
values and all continuous controls for every current layer, including layer
target FPS and Shift amount/block size/density/speed. The stack has no fixed
count ceiling; the panel grows this menu
from the authoritative live stack. Newly complete targets include static key threshold, softness, RGB
color and chroma tolerance; temporal key threshold, softness and history; and
VHS edge-wave speed, tracking wave, composite/chroma noise, luma smear, and
sharpening. Static and temporal key-mode selectors remain discrete and cannot
be routed.

The meter in each row is centered for bipolar sources and shows the cached
value after source shaping and attack/release slew, before route depth. A
stable runtime route ID keeps edits and removal bound to the intended row when
multiple panels change the matrix concurrently. These IDs are process-local
and are freshly assigned when saved route settings are loaded from a patch.
The engine computes one immutable routing result for each rendered frame and
reuses it for Morph, master, transport, and every layer, so one source/route is
not sampled repeatedly by different consumers.

### Morph

Capture A and B, choose **Linear** or **Equal power**, and move the morph
fader. **Glide A/B** travels for the chosen number of beats. Hue and slit angle
travel over their shortest circular path; discrete choices switch at the
midpoint. A capture first commits the current Morph result, including a routed
Morph-position offset, then records it against the panel's current layer-stack
revision. It is never coalesced past earlier queued edits, and a stale capture
is rejected and corrected by the next snapshot.

Manual fader and blend-law edits are accepted and materialize their bases while
Freeze Program holds automatic glide/clock motion. If Blackout is engaged while
the program is frozen, the cut remains absolute and releasing it restores the
exact pre-cut audience. A frozen selective-VHS audience frame stays held until
the program resumes and can process the complete replacement. Adding a layer
leaves it outside existing A/B slots; reorder and removal remap the slots to
their surviving layers. The slots, law, position, and exact remaining glide are
restored with the patch, including a remainder shorter than the new-glide
control's quarter-beat minimum. Morph commands can also use beat latch. Other
matrix offsets stay frame-local rather than being written into a capture.

While A and B are both engaged, they own the master, temporal, VHS, and captured
layer controls. Moving or resetting one of those controls commits the visible
interpolation, clears A/B, and then applies the manual value, so the control and
audience output stay together. A layer appended after capture remains outside
the old slots and does not disengage them when edited. With beat latch enabled,
the transfer to manual control occurs on the downbeat rather than at enqueue.

### Audio

Enable analysis in **AUDIO**, choose **Live input** or **Looping file**, and set
gain. Live input lists microphones/line inputs plus Windows **System playback**
output endpoints (WASAPI “what you hear”). Looping file accepts WAV, MP3, FLAC,
Ogg, Opus, M4A, and AAC imported into the library. Select **Choose imported
audio…** or the adjacent **Choose…** button to open the operating system's
filtered, multi-file picker. Cancel leaves the current clip selected. Each
audio upload may be at most 512 MiB; decoded audio is limited to 10 minutes,
and a decode that has not completed within 60 seconds is abandoned with a
visible error. Successful files are uploaded in order, and the last successful
one is selected for analysis.

The selected file is decoded once, loops without a seam in analysis time, and
deterministically drives the same audio matrix sources live and during offline
export; it need not be audible.
Choose 3–8 bands, then edit the N−1 ordered crossover fields and the separate
analysis ceiling. Every active band is routable as **Band 1** through **Band
8**; **Bass/Mid/High** remain compatible aliases for bands 1–3. The compact
32-bin spectrum is display feedback, while spectral brightness and noisiness
remain separate matrix sources. Source mode, selected file/device, gain, band
count, crossovers, and ceiling persist in the patch.

The requested preference and active capture device are reported separately.
If a saved named device is unavailable, the engine can keep that request while
using the system-default input and marks the stream as a fallback; it does not
reopen the same fallback every frame. If no stream can be opened, or a running
stream fails or stops producing samples, capture turns off, every audio source
returns to zero, and the error remains visible.


## Output and render controls

- **Output window** opens a fullscreen audience window, preferring a second
  monitor when one exists. On a single-monitor system it instead promotes the
  existing main preview to a clean fullscreen audience surface. If window or
  GPU-surface creation fails, the switch returns to the actual closed state and
  the panel shows the error.
- At startup, the requested initial visual is metadata-probed under Safe policy
  before its dimensions can size the preview. A rejected probe uses 1280×720;
  if source-sized renderer initialization fails, the app makes one 1280×720
  recovery attempt and surfaces the result in RECOVERY and the browser state.
- **Blackout** cuts the shared final output, including preview/output-window
  and Spout/readback consumers.
- **Spout output** (Windows) publishes the named `collide-o-scope` sender.
- **Render** exports at the chosen resolution, FPS, and duration.

Selective VHS export is synchronous but follows the live layer law: bypassed
layers retain only their direct effects, inherited layers receive master effects
and VHS, the stack is recomposited, and Temporal remains program-wide. A
selective-processing failure stops the export with an error instead of silently
applying VHS to a bypassed layer.

**VHS quality** affects both the ordinary global path and those inherited
selective slices:

- **Live parity (half)** is the default, including for older clients. It
  downsamples to half width and height, runs VHS, and upscales, matching the
  real-time renderer's spatial path.
- **Native (full resolution)** runs VHS at the selected export dimensions. It
  avoids the half-resolution downscale/upscale but is slower and more
  memory-intensive; live rendering remains half-resolution.

Both modes preserve keyed alpha and the same global/selective routing and
composition order; quality changes only VHS spatial resolution. When VHS is
off, the choice has no effect.

Expert media mode may admit a larger source while reconstructing an offline
patch, but it does not enlarge the selected export output. Export-output
dimensions retain the established UHD-area validation, and selective export
retains its independent bounded working-set validation.

The optional **Audio (1× independent)** selector muxes the chosen video
layer's first audio stream. Audio starts at source time zero, ignores visual
pause/speed/modulation/looping, and is padded or trimmed to the requested
duration. Live Spout layers cannot be selected for audio and render as black
offline because an external live sender is not reproducible.

## Patch controls

- **Capture snapshot** writes a unique YAML file under `patches/` without
  opening a picker, blocking the render loop, or overwriting an earlier
  capture.
- `Ctrl+S` opens a native Save dialog for the complete YAML performance state.
- **Load snapshot…** or `Ctrl+O` selects exact snapshot reconstruction.
- **Apply look…** or `Ctrl+Shift+O` selects visual transfer onto the live stack.
- `Ctrl+E` opens the native patch parameter editor. It edits the live
  master/layer parameter subset; it is not a full YAML text editor. The saved
  file remains ordinary YAML and can be edited separately in a text editor.

| State | Load snapshot | Apply look |
|---|---|---|
| Sources and topology | Rebuilds the saved sources, order, and layer count | Keeps current sources, identities, order, and count |
| Master/layer visuals | Restores all saved values | Applies master values; maps saved layer opacity, blend, visibility, Bypass Master FX, direct effects/keying, and pattern seed by stack position |
| Layer transport | Restores pause, speed, target FPS, and loop-reroll policy | Preserves all current values |
| Program transport | Restores Freeze Program and Freeze Media | Preserves both current freeze states |
| Modulation/input | Restores BPM, LFOs/seeds, routes, and input configuration | Preserves the current matrix and input state |
| Morph | Restores saved slots, law, position, and remaining glide | Does not import saved morph; an engaged current A/B pair is materialized and cleared first |
| NTSC/Temporal | Applies each saved section; a legacy omitted section remains current | Applies each saved section; a legacy omitted section remains current |

Apply Look maps only `min(saved layers, current layers)` positions. Extra
current layers stay visually unchanged, extra saved layers are reported
unused, and master values still apply. The browser includes the observed stack
revision and the engine checks it again after the picker closes, so a
concurrent topology change rejects the request rather than shifting the look
onto different sources.

On a successful Apply Look, conflicting work queued before or during the picker
is filtered from the already-drained batch, immediate queue, and beat-latched
queue. This includes topology/reroll actions and edits to the applied master,
mapped layers, or present NTSC/Temporal sections. Transport/safety work,
unmapped-layer edits, and edits to omitted sections retain their order. A
cancelled, failed, or stale chooser changes neither state nor queue ordering.

Saved state includes stable source identities (legacy paths or retained
`cos-sha256://` length/digest references); layer order, visibility, pause,
speed, effects and pattern seeds; per-video loop reroll; both freeze states;
NTSC and Temporal; modulation routes, LFO sample-and-hold seeds and input
configuration; tempo/morph state; gyro calibration; pad state; and audio band
edges. MIDI and audio hardware connections are reopened as available rather
than serialized as live device handles.

For a content-addressed visual or imported analysis clip, the resolved host
path is runtime-only. Capture/save keeps the portable identity, and browser UI
export re-fingerprints that runtime path as a candidate against the saved
digest. A post-load mutation therefore fails preflight instead of being
silently rendered, including when the file lives beside the patch rather than
inside the active library.

Older route targets named `layerN_key` are loaded as
`layerN_key_threshold`. If a transitional patch contains both names for the
same layer, the canonical `layerN_key_threshold` route wins instead of applying
the destination twice.

Snapshot reconstruction is atomic across the visual stack and saved imported
analysis audio. The engine resolves every file and fully decodes that audio
before committing; a missing, invalid, or corrupt source leaves the current
master, audio, topology, generations, and performance in place. A legacy patch
with no modulation section preserves current audio state. On success, the engine
starts new topology and visual generations, clears immediate and beat-latched
browser work plus the already-drained action-batch remainder, and invalidates
temporal history, retained NTSC output, and pending asynchronous readbacks.

## Troubleshooting

| Symptom | Cause and response |
|---|---|
| **403 Access denied** | The token belongs to an earlier app session, the request came from another web origin, or a bare untokenized URL was opened. Use the desktop URL opened by the current app or scan its current QR. |
| **Phone cannot reach the panel** | Confirm one network, private-network firewall permission for both ports, and no guest-Wi-Fi client isolation. A PC hotspot is often the clean venue fallback. |
| **Certificate warning returned** | The LAN identity changed and the certificate regenerated. Accept it again. |
| **GYRO says HTTPS is required** | Open the QR's `https://…:3031` URL, not HTTP. |
| **GYRO reports no sensor data** | Enable it on the physical phone, check browser permission, and confirm the device has orientation sensors. |
| **XY pad stays away from center** | Spring is disabled, or another connected pointer still owns the pad. Release the gesture or enable Spring. |
| **Queued value has not moved** | Beat latch is waiting for the next four-beat downbeat; check its pending count. |
| **Spout layer stays black** | Confirm Windows, exact sender name, sender activity, matching GPU/Spout environment, and the layer's status text. |
| **Spout output is not visible** | Enable it, keep the app rendering, and verify with a real receiver or `cargo run --bin spout_probe`. |
| **Output window will not open** | Read the error shown under OUTPUT; confirm a usable monitor/GPU surface and try again. The switch reflects the actual closed state. |
| **Panel server is unavailable** | Read the native RECOVERY status and concrete bind error. Stop any earlier process holding ports 3030/3031, then restart. Browser count alone is not listener health. |
| **Library is empty or points at the wrong folder** | Check the active path in RECOVERY, choose the intended folder, and use Rescan. Choosing or rescanning does not add a layer automatically. |
| **A large source is rejected in Safe or Expert mode** | Safe intentionally stops above UHD area. Expert still enforces DCI-8K area, absolute/device edge, per-buffer, and aggregate host-planning limits; it is not a guarantee of available VRAM. Reduce the source or close other above-Safe sources. |
| **Selective VHS reports a memory budget error** | The requested output size and number of visible, positive-opacity layers exceed the bounded live working set. Hide unneeded layers, set their opacity to zero, or lower the output resolution. The engine holds the prior exact audience frame and does not violate Bypass Master FX. |
| **Audio meters remain zero** | Select/enable a real input, grant OS access, and inspect the panel error. Software-only tests do not prove venue hardware. |
| **QR contains 127.0.0.1** | No LAN address was found at launch. Connect networking and restart. |

## Hardware validation boundary

Browser simulation can validate protocol and layout but not sensor axes,
microphone drivers, MIDI timing, Spout interoperability, or monitor selection.
Before a performance, validate the actual phone, audio interface, MIDI
controller/clock source, Spout sender/receiver applications, and display setup
on the exact stage hardware.

## Publication and license boundary

The MIT terms in [../LICENSE](../LICENSE) apply to the fork additions described
there, not automatically to the original upstream portions. Publication or
distribution of the combined fork is conditional on the publisher having the
needed upstream authorization or a later upstream license that permits it.
This is a project boundary notice, not legal advice.
