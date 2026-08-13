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

- Use the Library **+** to upload video from the current device. Uploads land
  in the app's `videos/` folder and name collisions get numbered suffixes.
- Double-click/double-tap a library tile to add a video layer.
- Drag the handle on a layer card to reorder it. The browser keeps one pointer
  owner for the gesture and sends one atomic move on release.
- Expand a layer's controls to edit its target decode FPS, transport, blend,
  keying, and complete direct effect set, including downsample. The same
  continuous effect values remain available as modulation targets.
- Use the layer **×** to remove it. Library deletion is separate: it moves the
  source file to the OS Recycle Bin and refuses to remove a clip that is still
  loaded.
- Double-click/double-tap an individual slider to reset it. Section-level
  **reset** buttons restore the corresponding group.

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
`Pad Y` to any target in the matrix.

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
patch. Yaw is meaningful relative to calibration because compass fusion can
drift. Use the QR's HTTPS URL on iOS; plain HTTP never offers sensor permission.

### Routing response

Every route row supports signed depth, response curve and amount, plus
independent attack/release slew. These controls apply equally to LFO, audio,
MIDI, gyro, and pad sources. The target menu covers master/NTSC/temporal/morph
values and all continuous controls for each of 16 layers.

### Morph

Capture A and B, choose **Linear** or **Equal power**, and move the morph
fader. **Glide A/B** travels for the chosen number of beats. The slots, law,
position, and remaining glide are restored with the patch. Morph commands can
also use beat latch.

### Audio

Enable the input in **AUDIO**, choose a device when available, and set gain.
Choose 3–8 bands, then edit the N−1 ordered crossover fields and the separate
analysis ceiling. Every active band is routable as **Band 1** through **Band
8**; **Bass/Mid/High** remain compatible aliases for bands 1–3. The compact
32-bin spectrum is display feedback, while spectral brightness and noisiness
remain separate matrix sources. Device selection, gain, band count,
crossovers, and ceiling persist in the patch.

The requested preference and active capture device are reported separately.
If a saved named device is unavailable, the engine can keep that request while
using the system-default input and marks the stream as a fallback; it does not
reopen the same fallback every frame. If no stream can be opened, or a running
stream fails or stops producing samples, capture turns off, every audio source
returns to zero, and the error remains visible.


## Output and render controls

- **Output window** opens a fullscreen audience window, preferring a second
  monitor when one exists. If window or GPU-surface creation fails, the switch
  returns to the actual closed state and the panel shows the error.
- **Blackout** cuts the shared final output, including preview/output-window
  and Spout/readback consumers.
- **Spout output** (Windows) publishes the named `collide-o-scope` sender.
- **Render** exports at the chosen resolution, FPS, and duration.

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
| **Audio meters remain zero** | Select/enable a real input, grant OS access, and inspect the panel error. Software-only tests do not prove venue hardware. |
| **QR contains 127.0.0.1** | No LAN address was found at launch. Connect networking and restart. |

## Hardware validation boundary

Browser simulation can validate protocol and layout but not sensor axes,
microphone drivers, MIDI timing, Spout interoperability, or monitor selection.
Before a performance, validate the actual phone, audio interface, MIDI
controller/clock source, Spout sender/receiver applications, and display setup
on the machine and network that will be used at the venue.

## Publication and license boundary

The MIT terms in [../LICENSE](../LICENSE) apply to the fork additions described
there, not automatically to the original upstream portions. Publication or
distribution of the combined fork is conditional on the publisher having the
needed upstream authorization or a later upstream license that permits it.
This is a project boundary notice, not legal advice.
