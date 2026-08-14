# Connecting to the control panel

Collide-o-scope serves plain HTTP for the computer running the engine and HTTPS
for phones. iOS exposes motion sensors only to a secure page.

| Use | URL | Access |
|---|---|---|
| Desktop | `http://127.0.0.1:3030/?key=…` | Fresh per-session token opened by the app |
| Phone/other LAN device | `https://<lan-ip>:3031/?key=…` | Per-session token from the QR code |

## Desktop

1. Launch collide-o-scope. The panel should open in the default browser.
2. If it does not, use the exact tokenized URL reported by the current app or
   relaunch it. A bare `http://127.0.0.1:3030` URL is intentionally denied.

Loopback is not an authentication bypass. The first successful navigation
exchanges the key for an HttpOnly, `SameSite=Strict` session cookie and removes
the key from the visible address bar. WebSocket upgrades, uploads, deletes,
and other mutating requests must also originate from the exact page origin;
scripts hosted by another site receive 403 even if they target localhost.

Only one running instance can own ports 3030 and 3031. If a newly built app
appears to have an old panel, stop the earlier instance and hard-refresh the
tab; the HTML, CSS, and JavaScript are embedded in the executable.

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
  audio from the current device. Uploads land
  in the app's `videos/` folder and name collisions get numbered suffixes.
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
- The top play/pause control freezes the whole visual program: decoder and
  Spout frames, shader/NTSC time, temporal accumulation, LFO and morph phase,
  routing slew, and imported-audio analysis all hold and resume without
  catch-up. Live input telemetry may continue updating its meters while its
  rendered modulation value remains frozen.
- **Revert master visual state** restores master, VHS, and temporal effects,
  and clears modulation routings and morph automation so those defaults stay
  in force. It preserves layers and their effects, visibility and transport,
  BPM, and audio/MIDI/device choices. A layer card's **Reset FX** remains local
  to that layer and deliberately leaves opacity and transport unchanged.
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

## Performing controls

All connected panels receive the same engine snapshot. Immediate browser input
enters a bounded queue: repeated absolute values for the same control are
coalesced so the latest wins, while emergency and pointer-release actions keep
reserved capacity. This prevents a fast slider, sensor, or disconnected client
from building an unbounded render-thread backlog. Beat latch adds the separate
downbeat behavior below.

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
values and all continuous controls for each of 16 layers, including layer
target FPS. Newly complete targets include static key threshold, softness, RGB
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
Pause holds automatic glide/clock motion. If Blackout is engaged while paused,
the cut remains absolute and releasing it restores the exact pre-cut audience.
A paused selective-VHS audience frame stays held until Resume can process the
complete replacement. Adding a layer
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
- **Blackout** cuts the shared final output, including preview/output-window
  and Spout/readback consumers.
- **Spout output** (Windows) publishes the named `collide-o-scope` sender.
- **Render** exports at the chosen resolution, FPS, and duration.

Selective VHS export is synchronous but follows the live layer law: bypassed
layers retain only their direct effects, inherited layers receive master effects
and VHS, the stack is recomposited, and Temporal remains program-wide. A
selective-processing failure stops the export with an error instead of silently
applying VHS to a bypassed layer.

The optional **Audio (1× independent)** selector muxes the chosen video
layer's first audio stream. Audio starts at source time zero, ignores visual
pause/speed/modulation/looping, and is padded or trimmed to the requested
duration. Live Spout layers cannot be selected for audio and render as black
offline because an external live sender is not reproducible.

## Patch controls

- `Ctrl+S` saves the complete YAML performance state.
- `Ctrl+O` loads it and reconstructs layers by stable source identity.
- `Ctrl+E` opens the native patch parameter editor. It edits the live
  master/layer parameter subset; it is not a full YAML text editor. The saved
  file remains ordinary YAML and can be edited separately in a text editor.

Saved state includes layer order/source/pause/speed/effects, master pause, NTSC
and temporal settings, modulation routes and input configuration, tempo/morph
state, gyro calibration, pad state, and audio edges. MIDI and audio hardware
connections are reopened as available rather than serialized as live device
handles.

Older route targets named `layerN_key` are loaded as
`layerN_key_threshold`. If a transitional patch contains both names for the
same layer, the canonical `layerN_key_threshold` route wins instead of applying
the destination twice.

Patch reconstruction is atomic. Once the new stack succeeds, the engine starts
new topology and visual generations, clears immediate and beat-latched browser
work from the previous patch, and invalidates temporal history, retained NTSC
output, and pending asynchronous readbacks. If reconstruction fails, the
current performance remains in place.

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
