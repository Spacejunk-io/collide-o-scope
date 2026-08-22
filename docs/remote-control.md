# Connecting to the control panel

Collide-o-scope serves plain HTTP for the computer running the engine and HTTPS
for phones. iOS exposes motion sensors only to a secure page.

This guide covers the live browser surface. For exact persistence, history,
recording, controller, health, and venue-failure contracts, see
[Professional console and stage](professional-console-and-stage.md). For the
measured RGBA16F/proxy/Study evaluation and the capabilities that remain
explicitly deferred, see [Precision and scale](precision-and-scale.md).

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

- **Revert master visual state** restores master effects/transform/Motion,
  replaces the master Collision Rack with its exact legacy markers, and resets
  VHS and all Temporal/Originals state,
  resets all eight LFO configurations/seeds, and clears modulation routings and
  Morph automation so those defaults stay in force. It preserves layers and
  their effects, visibility and transport,
  BPM, and audio/MIDI/device choices. A layer card's **Reset FX** remains local:
  it resets that layer's direct effects and Motion, but deliberately leaves its
  rack, transform, opacity, and transport unchanged; Transform and Rack have
  their own controls. The
  bundled panel sends `reset_visual_program` for this broad operation. The
  legacy `reset_fx` protocol action remains accepted but resets only direct
  master effect uniforms, matching the original remote-control contract. It
  deliberately leaves the Collision Rack, Motion, VHS, Temporal, routings,
  Morph, and queued automation
  live, so those systems may continue changing the rendered master afterward.
- **Bypass Master FX** on a layer skips inherited Digital, Analog, Cellular,
  Motion, and VHS processing for that layer. Its own Layer FX, opacity, key,
  and blend remain active. This switch retains the v1.2 LinkedDry contract: a
  visible, positive-opacity bypass layer links the complete shared Temporal
  family dry for the whole program when no explicit **Bypass Temporal FX**
  route is active. Feedback, Slit-Scan, Temporal Originals, Melt, Sync Latch,
  Display Physics, and Codec Mosh receive neutral frame parameters. The clean
  program still warms Temporal history, and the authored/modulated controls
  resume unchanged when no contributing bypass remains. With VHS enabled,
  direct master effects and VHS run only on inherited layer slices; the engine
  then recomposites the stack before that linked Temporal boundary. Offline
  render follows the same law. Hidden or non-positive-opacity layers neither
  link Temporal dry nor create selective work. Live selective processing has
  a 320 MiB safety budget and no silent
  global-VHS fallback: if the current output size and contributing stack exceed
  it, the prior exact audience frame is held and the VHS panel reports how to
  reduce the load. Layer **Reset FX** and master **Revert** preserve this
  non-destructive switch.
- **Bypass Temporal FX** is the independent, exact Temporal route. It defaults
  off and is controlled remotely with `set_layer_param` parameter
  `bypass_temporal_fx`. Every enabled layer must form one contiguous prefix at
  the top/front of a flat LegacyExact Program stack: Layer 1 is topmost, so
  Layer 1 alone or Layers 1–N may be dry, with no inheriting gap between them.
  The lower inherited stack continues through Feedback, Slit-Scan, Temporal
  Originals, Melt, Sync Latch, Display Physics, and Codec Mosh; afterward the
  engine recomposites the dry prefix bottom-to-top with each layer's original
  Layer FX, key, opacity, and blend. Changing prefix membership resets the
  Temporal-family history before the new partition renders, without changing
  authored controls. Interleaving, groups, A/B or non-Program buses, active
  mattes, advanced layer/master racks, advanced Motion, any authored
  Master/layer Motion modulation route (even at zero source or depth), routed
  Refresh Garden, and currently VHS are rejected transactionally: the switch and audience
  remain at their prior accepted state. When admitted, this explicit route
  takes precedence over Master-only LinkedDry: lower inherited layers remain
  temporally wet even if a contributing layer also bypasses Master FX, while
  each switch continues to govern its named boundary. Patches that omit the
  additive field load `bypass_temporal_fx: false`.
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

### Collision Rack, groups, A/B, and image routes

Open **COLLISION RACK + GROUPS** above the layer stack. Choose Master, one live
layer, or one group as the rack scope. A rack holds at most eight stable-ID
nodes; add Transform, Digital/Color, Key, Cellular, Shift, Grain, or Mask,
then edit enabled, wet mix, blend, and bounded node values. Reordering changes
evaluation order but not `NodeId`-addressed routes. The panel reports a
preflight rejection instead of installing an over-budget rack, a current-frame
cycle, or an invalid dependency. An old patch is represented by exact Legacy
Canonical/Temporal markers rather than a visually similar rewrite.

**GROUPS + ROOT ORDER** creates up to 16 one-level groups from a contiguous run
of root layers. A group can remain addressable while empty and owns its own
opacity, transform, rack, matte, solo, bypass, and Program/A/B bus. Root order
is back-to-front. Assign independent layers or whole groups to A or B and use
the top **A / B** fader to linearly mix those premultiplied-linear bus results;
Program-bus content is then composited directly over the mix. Stable group and
node IDs prevent reorder from retargeting routes. Removing a referenced donor
leaves an explicit missing item for repair rather than choosing a new donor.

Layer cards expose **Matte / Image Input**. Image-mask rack nodes and group
mattes expose the same typed input law. Inputs can name a selected stable layer
(pre- or post-local effects), One Below, All Below, Clean Program, Program
History N−1, or a stable Group output. Choose Alpha/Luma/R/G/B, optional invert,
amount, threshold, and softness. Same-frame routes are accepted only when the
complete graph is acyclic; **Previous frame N−1** is a deliberate retained edge,
not a timing-dependent approximation.

### Performance Set, transport, and Scenes

To prepare another source without cutting the audience, select a stable layer
under Library **LOAD INTO SLOT**, choose Immediate/Next beat/Next bar, then use
that tile's **Load slot** command. Probe, decode, and GPU upload finish before
the new `ClipSlotId` can activate; failure leaves the current source live. Each
layer owns at most 32 prepared slots.

Open the layer's **Slots / Transport** disclosure to select/activate/remove a
prepared source. Per-slot transport includes normalized seek and in/out, Forward
or Reverse direction, Loop/Ping pong/One shot/Hold ending, 0–16× rate, optional
sample FPS, clip BPM/length/program sync, a bounded beat loop, and up to 64
numeric cues. A missing cue or stale slot ID is non-destructive. Freeze Media
stops source and Scene boundary progress; Freeze Program also stops the program
clock and therefore all quantized activation.

The **Scenes** panel captures the active slot for every live layer into a stable
named Scene. The typed Scene format can also carry an optional cue per binding.
Up to 128 Scenes and 256 bindings per Scene are accepted. **Prepare** makes all
referenced sources ready;
**Trigger** commits the whole binding set immediately, on the next beat, or on
the next bar. One missing layer/slot/cue, stale topology, preparation error, or
cancel prevents every part of the Scene from changing. Recapture repairs the
same stable Scene; Remove deletes exactly that ID.

The **Autopilot** editor below Scenes is an authored, Scene-only beat sequence:
up to 128 ordered stable Scene IDs, 1–256 accepted media beats per step, and a
Loop or Once ending. **Play** prepares ahead and releases the first Scene only
on a future beat. A late Scene stalls; readiness then waits for another future
beat, so Autopilot never skips or catches up with multiple cuts in one frame.
Pause, Freeze Media, and Freeze Program preserve the remaining dwell. A failed
prepare or commit holds the last visible Scene and enters a repairable fault;
manual clip/Scene activation disarms the run. Removing a referenced Scene keeps
its ID as a visible tombstone, and automatic Scene capture never reuses that
reserved ID. The plan is patch state; Play/Pause/Reset are live performance
state. Offline export refuses a moving Starting/Running/Stalled sequence while
allowing a Paused or otherwise static plan.

### Spatial transforms

The master panel and every layer card expose one complete transform:

- **Position X/Y** uses normalized composition coordinates, centered at zero.
- **Scale X/Y** is independent; the link control edits both axes together and
  negative scale mirrors the source.
- **Anchor X/Y** is measured in original source UV space. Anchor alone does not
  move the picture; it chooses the pivot for scale, skew, and rotation.
- **Rotation** is clockwise in screen space. **Skew** acts along the separately
  authored **Skew axis**.
- **Fit** shows the complete cropped source, **Fill** covers the composition,
  **Stretch** reproduces the historical full-frame mapping, and **Native** maps
  source pixels one-for-one to output pixels.
- **Crop** records left/top/right/bottom source fractions. **Transparent**,
  **Clamp**, **Repeat**, and **Mirror** select edge behavior; **Linear** and
  **Nearest** select filtering.

The forward authored order is crop/framing, scale, axis-directed skew,
rotation, and position about the source-space anchor. All finite values are
bounded by the engine, and a collapsed transform becomes transparent. Reset
restores the inactive historical full-frame sample with a Transparent authored
edge, so moving or shrinking it cannot smear a border; Clamp remains an
explicit choice. **New layer framing** in the Library is an engine-authoritative
host-session preference (Fit by default) for future interactive file, still,
and Spout layers; it never changes existing layers and is not PatchState. Older
patches with no transform retain the byte-compatible inactive shader bypass.

Transform presets and copy/paste replace the complete transform atomically.
Scalar drags coalesce per field, while reset, preset, and paste are ordering
barriers. Layer actions name the immutable runtime layer ID, so a delayed edit
cannot land on a different source after reorder. Master and layer transforms
are saved, transferred by Apply Look, optionally owned/interpolated by Morph,
and evaluated identically in live and offline rendering.

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

- **Master** changes the master shader-pattern seed and all eight LFO seeds. In
  Bounded variation mode it also varies the master effect controls listed
  below.
- **Everything** does the same and includes every current layer. With an exact
  base seed, the master keeps that value while each layer position and LFO gets
  a reproducible derived stream. The panel supplies the current stack revision,
  so a concurrent add/remove/reorder rejects the whole request instead of
  targeting the wrong sources.
- **Group** changes only explicitly opted compatible Rack/Composition values for
  the selected stable group; it does not reroll source or layer pattern seeds.
- Each layer card exposes its exact pattern seed and a pattern-only **Reroll**.

**Pattern only** changes seeds without moving effect controls and does not
disengage an active A/B morph. **Bounded variation** first chooses those seeds,
then makes reflected, range-safe changes to pixelate/downsample, RGB split,
hue, saturation, brightness, contrast, posterize, grain amount/size, vignette,
color drift, breathing motion, Shift, and core Cellular
amount/scale/warp/speed. At applicable scopes it also changes bounded numeric
values for Temporal Loom/Atlas/Garden and M4 Motion.
**Amount** ranges from 0 to 2. **Transform** separately opts in bounded
position/scale/anchor/rotation/skew/axis/crop. **Rack values** opts in compatible
node wet/numeric values. **Composition values** opts in group opacity,
transform, and matte values and, for Everything, the A/B crossfade. All three
are off by default. Pattern-only and automatic each-loop rerolls never move
those controls. **Grain mode** additionally permits the discrete grain
algorithm and color-grain switch to change.

Dice never changes source identity; rack/group/layer topology; image routes;
node IDs/order/enabled/blend; group membership/solo/bypass/bus; layer opacity,
visibility, blend, keying, Bypass Master FX, or Bypass Temporal FX; transport;
modulation routes;
VHS; Temporal topology/seeds/Score/reset/loop-driver law; or Motion algorithm,
field-source/quality, donor, or carrier law. Because variation edits visual
bases, a valid variation materializes and clears an engaged A/B Morph before
applying.

Leave **Exact seed** blank to advance deterministically from the stored master
and targeted-layer seeds. Master and Everything always derive all eight LFO
seeds from the resulting master seed; Everything also derives a positional
stream after advancing each layer's stored seed. Enter a whole number from `0`
through `4294967295` to replay a base seed; `0` deliberately restores the
legacy shader and sample-and-hold sequences. Repeating a Bounded variation also
requires the same starting effect values.

Each LFO shows its own seed field when **Sample & Hold** is selected. It holds
one deterministic value for a complete LFO cycle; changing rate changes that
hold duration. Master and Everything rerolls update all eight LFO seeds, even
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

### Temporal Originals

The **TEMPORAL** panel still begins with Feedback, Slit-Scan, and History Key,
all sampled from the fixed 30 Hz, 24-frame history. Five additional disclosures
reuse that bounded memory:

- **Topology Loom** maps pixel position to history age using Linear, Radial,
  Spiral, Contour, Folded, or Kaleidoscopic topology. Sampling can floor to one
  age or interpolate between two; Depth, Phase, Scale, Angle, Folds, and
  Quantize shape the map.
- **Collision Atlas** assigns seeded territories and mixes their temporal
  collisions. The seed and territory count are authored, deterministic state.
- **Long Exposure Ghosting** averages the clean current frame across a 2–24
  frame shutter. It is exact through eight frames and uniformly stratifies
  longer shutters into eight total samples, so the full trail span survives
  with at most seven unfiltered same-pixel history reads and no 24-tap
  full-resolution bottleneck, extra pass, or allocation.
- **Refresh Garden** admits/holds memory through Temporal Δ, Luma, Chroma,
  Cellular ridge, Audio energy, Audio onset, Matte, or Motion gates, with
  bounded threshold, softness, decay, and maximum hold. **Matte layer** selects
  one stable layer plus its pre- or post-local current-frame stage. **Motion
  layer** selects that layer's actually admitted motion field—not a luma proxy—
  so codec/lattice source resolution and an admitted Faraday donor are the same
  live signal Garden observes. The route status reports None and saved-layer
  tombstones as closed zero gates.
- **Collision Score** walks a seeded 2–16-state score on loop boundary,
  downbeat, audio onset, or the explicit **Trigger score** event. A loop driver
  names one stable layer; a removed layer becomes visibly missing and does not
  retarget another source.

**MEMORY LAW** separately chooses None/Score/Memory/All for loop-boundary and
downbeat resets. **Clear temporal memory** is an ordered barrier: it clears
history/Garden/Score runtime memory without changing authored controls or a
paused audience hold. Freeze Program stops temporal ticks and events; Freeze
Media keeps them advancing while source images remain held. Zero amounts retain
the legacy temporal path, and the same frame-indexed law is used by offline
rendering.

Garden route changes are ordered, revision-protected topology actions. They are
manual-history edits rather than coalesced or beat-quantized values. Reorder
keeps the selected stable identity and refreshes only saved-position provenance;
removal creates a missing tombstone that cannot bind a replacement at the old
position. Patch, Morph, reset, and offline export preserve that law. Export maps
saved selections to deterministic job-local identities and publishes a warning
when a Matte or Motion route resolves to zero. The routed signal is applied by
the same dedicated post-temporal pass used live, capped at three sampled
textures per pass.

**Refresh now** emits one ordered counted Garden event. Accepted Score and
Garden events are stored in a bounded reference-tick track and replayed by
offline export at the same accepted-frame boundaries. **Clear event recording**
clears that track only; it does not clear authored controls, Garden/Score
memory, Program history, or the audience hold.

### Motion fields

The master **MOTION FIELDS** panel and each layer's **Motion / Faraday**
disclosure expose versioned motion authoring. **Auto** uses valid codec vectors
for a layer and reports a deterministic Motion Lattice fallback; Master Auto is
always lattice. **Codec vectors** is strict—missing/invalid side data yields a
zero field and diagnostic—and **Motion lattice** always uses deterministic block
matching. Draft/Live/High fix block/search/update quality and never silently
change under load.

Only layers can enable **Faraday Motion Transplant**. Choose one stable donor,
Transparent/Black/First source frame carrier initialization, and bounded
confidence, refresh, decay, and occlusion values. One transplant is admitted
composition-wide; a removed donor remains missing. **Curved Shutter** is
available at master and layer scope: 0° is an exact bypass, 360° spans one frame
period, and Sharp/Draft/Live/High use fixed 1/4/8/16 samples with authored phase,
curvature, and chromatic lag. **Clear motion memory** is an ordered carrier
reset. Telemetry reports requested/resolved field origin, diagnostics, vector
count, resource admission, donor, carrier, and shutter truth.

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

The matrix exposes eight LFO lanes (`L1`–`L8`). Lanes 5–8 begin with the
standard LFO defaults and affect nothing until routed; loading a legacy patch
preserves lanes 1–4 exactly and leaves the appended lanes neutral.

Every route row supports signed depth, response curve and amount, plus
independent attack/release slew. These controls apply equally to LFO, audio,
MIDI, gyro, and pad sources. The target menu covers master/NTSC/temporal/morph
values and all continuous controls for every current layer, including target
FPS, Shift, spatial transform, bounded Loom/Atlas/Garden, master/layer Curved
Shutter, and layer Faraday values. Stable-ID targets cover compatible rack-node
wet/numeric values, group opacity/transform/matte values, and the composition
A/B crossfade. The composition ceiling is 256 layers; the panel grows its menu
from the authoritative live stack/graph. Other complete targets include static
key threshold/softness/RGB/chroma tolerance, temporal key threshold/softness/
history, and VHS edge-wave speed/tracking wave/composite/chroma noise/luma
smear/sharpening. Image routes, key modes, rack order/blend, Temporal topology/
gate/Score/reset law, Motion field/quality/donor/carrier, and other discrete
selectors cannot be routed.

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
midpoint. Spatial position, anchor and crop interpolate continuously, scale
uses geometric magnitude when both endpoints are nonzero, rotation and skew
axis take their shortest arc, and Fit/Edge/Sampling switch at the midpoint. A
capture first commits the current Morph result, including a routed
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

### Live recorder, stills, and resampling

**RECORDER** is separate from offline **Render**. **Record program…** opens a
native final-file picker and starts a video-only capture of the exact final
Program image after NTSC and absolute blackout. The render thread only submits
to fixed readback/pool/queue capacity; it never waits for FFmpeg. Readback,
pool, queue, source, and worker drops are counted, and a cadence gap repeats the
last admitted frame in the video instead of falsifying its duration. **Finish**
stops admission, drains in-flight work, then publishes. **Cancel** removes the
temporary artifact.

**STILL SNAPSHOT** can choose Program or one current stable layer/group scope
and publishes a PNG. **RESAMPLE TO CLIP** records Program or one layer/group,
then allocates a new prepared slot in the chosen destination layer; **Activate
after prepare** cuts only after its normal media preparation succeeds. A
deleted/missing target is a visible drop and never falls back to Program.
**Auto-import** likewise occurs only after the destination file has been synced
and committed without replacement.

Recorder output has a bounded `.recording.json` truth sidecar. It includes
capture/program cadence, freeze/blackout observations, drop counters, and an
audio-clock correlation stamp when available. The current live recorder does
not mux audio. Offline Render's independent 1× audio policy below is not a live
recording-audio claim.

### Health HUD and StageMap

**STAGE HEALTH → Preview HUD** paints FPS, p50/p95/p99 frame time, missed
deadlines, per-layer decode age/queue/drop state, active output identity/mode,
and known media/performance/NTSC/motion budgets on the editor preview only. The
HUD cannot enter audience, Composite, Spout, recording, or export pixels.

`StageMap` is a separately persisted venue YAML document, never PatchState.
It can define up to 16 endpoints, 64 slices per endpoint, and 256 slices total.
Each slice uses a source rectangle, four-corner perspective or a bounded convex
polygon mask, edge feather, and linear calibration. The browser selects an
existing endpoint to enable a test card or identification overlay; it is an
operator tool, not a full StageMap authoring editor.

Only an endpoint explicitly bound to a monitor can acquire/present a physical
surface. Each endpoint has an independent window/surface/acquire/present result;
a closed, lost, invalid, or over-budget endpoint is reported without authorizing
another endpoint to fail or substitute for it. Unassigned endpoints remain
offscreen. Test cards, identification, and calibration affect that endpoint
only, after the completed creative Program, and never alter the artwork saved
in a patch.

Selective VHS export is synchronous but follows the live **Bypass Master FX**
law: bypassed layers retain only their direct effects, inherited layers receive
master effects and VHS, and the stack is recomposited. If any Bypass Master FX
layer contributes, the complete shared Temporal family then runs neutral while
accepting the clean program into warm history. A selective-processing failure
stops the export with an error instead of silently applying VHS to a bypassed
layer.

An admitted **Bypass Temporal FX** top prefix also follows its live law during
offline render: the inherited background receives the complete Temporal family,
including Codec Mosh, before the dry prefix is recomposited in authored order.

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

**Shutter samples** is an exact, closed offline policy. **Authored per scope**
keeps each master/layer Sharp, Draft, Live, or High tier. The other accepted
wire values are `samples_1`, `samples_4`, `samples_8`, and `samples_16`; they
replace only the quality tier after that frame's Morph/modulation values are
resolved. Candidate and final immutable plans consume the same resulting count
before resource preflight. An explicit count does not activate a 0° shutter,
so the exact zero/legacy path remains delegated. Omitted fields default to
`authored`; for example, `"shutter_samples":"samples_16"` requests exactly 16.
Arbitrary counts and unknown strings are rejected.

Expert media mode may admit a larger source while reconstructing an offline
patch, but it does not enlarge the selected export output. Export-output
dimensions retain the established UHD-area validation, and selective export
retains its independent bounded working-set validation.

The optional **Audio (1× independent)** selector muxes the chosen video
layer's first audio stream. Audio starts at source time zero, ignores visual
pause/speed/modulation/looping, and is padded or trimmed to the requested
duration. Live Spout layers cannot be selected for audio and render as black
offline because an external live sender is not reproducible.

After the MP4 succeeds, offline Render atomically publishes
`<video>.motion.json`. The bounded schema-v3 provenance report records source
fingerprints when available; requested shutter policy and exact count;
separate authored and effective motion scope/algorithm/quality/donor/carrier
values; the final planner source, actual rendered source, and field-attachment
truth (including planned-but-unprimed fields); codec transition count, elapsed
source time, and exact proof/vector digest only when a proven codec field was
attached; diagnostics and dynamic-state changes; the last accepted
frame; and warnings. It explicitly does not guarantee cross-GPU pixel identity.
Cancel/failure removes partial video/sidecar work, and unavailable codec vectors
or fingerprints remain warnings rather than being rewritten as proof.

Live Motion cards use the same distinction: `planned` is the immutable source
decision, while `rendered` is published only after a matching field parity slot
commits. Lattice frame one can therefore say priming/unavailable; Media Freeze
retains the prior committed field identity; and Faraday recipients report the
admitted donor field and grid.

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
| Sources and Performance Set | Rebuilds saved sources/order, prepared clip slots, slot transport/cues, active slots, and Scenes | Keeps current sources, identities/order/count, prepared slots, transport, and Scenes |
| Collision Rack/composition | Restores exact rack nodes/IDs/routes, groups/root/membership/buses/A-B, and mattes/image routes | Copies compatible rack/group/matte/A-B values only; preserves topology, IDs, routes, donors, membership, order, solo/bypass, and bus assignment |
| Master/layer visuals | Restores all saved values, including spatial transforms and Motion | Applies master values and maps saved layer transform, opacity, blend, visibility, Bypass Master FX, Bypass Temporal FX, direct effects/keying, pattern seed, and Motion values by stack position; preserves Motion donor routes |
| Layer transport | Restores pause plus legacy speed/FPS/loop-reroll and the complete per-slot transport | Preserves all current values |
| Program transport | Restores Freeze Program and Freeze Media | Preserves both current freeze states |
| Modulation/input | Restores BPM, LFOs/seeds, routes, and input configuration | Preserves the current matrix and input state |
| Morph | Restores saved slots, law, position, and remaining glide | Does not import saved Morph; an engaged current pair is materialized and cleared first |
| NTSC/Temporal Originals | Applies each saved section; a legacy omitted section remains current | Applies compatible authored values while preserving the live Score loop-driver route; a legacy omitted section remains current |

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
`cos-sha256://` length/digest references); prepared slots, their complete
transport/cues, and Scenes; Collision Racks, composition/groups/A-B, image
routes/mattes; master/layer effects, transforms, Temporal Originals, and Motion;
layer order/visibility/pause/blend/Bypass Master FX/Bypass Temporal FX/reroll;
both freeze states; modulation routes, LFO seeds/input configuration,
tempo/Morph, gyro/pad, and audio bands.
MIDI/audio devices are reopened as available rather than serialized as live
handles.

`PatchState` is additive typed YAML. `visual_schema_version: 1` declares an
explicit M2 rack/composition topology; `0` or omission means the exact legacy
layout synthesized with frozen markers. Unknown future visual topology versions
are rejected rather than guessed. Omitted prepared-performance, Temporal
Originals, Motion, and Scene fields take their exact inactive/legacy defaults.
The generator's schema-v2 `manifest.json` is a different document; current
procedural output is generator v7. StageMap, controller/OSC configuration,
preset library, recovery journal, recorder state, runtime pixels, and GPU
resources are intentionally not PatchState.

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

## Manual history, presets, and recovery

**Manual history** under RENDER records browser-manual and native-manual edits.
One pointer drag is one entry; a legacy client that sends one absolute edit gets
one entry. Use **Undo/Redo** or Ctrl/Cmd+Z and Ctrl/Cmd+Shift+Z (outside a text
field). MIDI, OSC, LFO, audio, clip triggers, Collision Score, and other host
automation are live inputs but are not manual-history entries.

Each checkpoint is a complete authored world: PatchState, runtime-stable layer
identities/revisions/selection, patch base directory, StageMap, preset library,
and controller profile. It contains no GPU textures, decoded frames, temporal
carriers, or recorder buffers. Undo/redo preflights and durably publishes the
candidate before advancing its cursor; rejection leaves the live world and both
depths unchanged. Bounds are 128 entries, 64 MiB total, and 8 MiB per entry.

Open **SCOPED PRESETS** to capture/apply/delete one of these kinds:

- Transform copies the complete bounded transform;
- Rack requires compatible node topology and copies values only;
- Matte copies bounded values, channel, and invert while preserving its image
  route;
- Group copies opacity, transform, rack, matte, solo, bypass, and bus values
  while preserving group identity, membership, name, root position, and donors;
  and
- Controller Profile and Stage Map copy their complete validated typed
  documents, still outside PatchState.

The library is capped at 128 presets / 8 MiB. Capture/delete/apply use the
snapshot's current library, layer-stack, and composition revisions, so a stale
panel cannot overwrite newer structure. A candidate is synced and atomically
published before the live revision changes.

The recovery journal is different from manual undo: it is append-only,
checksum-framed, bounded, and PatchState-only. A successful manual checkpoint
or native patch save appends it. On startup, a valid newer checkpoint appears as
**Recovery checkpoint** with explicit **Restore checkpoint** and **Discard**;
it is never auto-applied and never overwrites the user's patch file. A corrupt
or truncated tail is ignored after the last valid prefix and reported.

## MIDI profiles and OSC

The four learnable CC rows remain the compatibility/default profile. The
**Controller runtime** block below them reports the loaded typed profile,
requested/active MIDI input/output, reconnect state, queue/drop counters, and
feedback truth. `controller_profile.json` is a separately bounded per-user
document, not patch state. It supports device selectors, omni/single-channel
filters, note and CC sources, absolute plus three relative CC encodings,
momentary/toggle/gate buttons, Start/Continue/Stop/24-PPQN Clock, and bounded
feedback. Saved layer positions resolve once to live stable IDs and never
retarget after reorder; group and rack-node mappings retain stable IDs. Scene
launch pads use additive v1 targets
`{"scope":"scene_prepare","scene_id":12}` and
`{"scope":"scene_trigger","scene_id":12}`. They resolve against the live
authored Scene set by `SceneId`, never by card position. A missing ID rejects a
profile install; if its Scene is later removed, the installed address becomes
an inert, visibly rejected tombstone rather than selecting a neighbouring
Scene. Scene targets require `button_mode: "momentary"`: Note/CC input emits
one action on the physical rising edge and none while held or on release.

The callback rejects malformed wire input before all state: supported channel
messages are exactly three bytes with both data bytes below `0x80`, and the
four supported transport/realtime messages are exactly one byte. Wrong or
extra lengths, running-status fragments, high-bit channel data, and unsupported
statuses increment the malformed counter once without changing Learn, CC,
clock, queue, button, or event state.
That counter names rejection by Collide-O-Scope's closed controller vocabulary;
it does not classify every other valid MIDI-spec family for other software.

Portable profile import/export uses bounded JSON. A native picker owned by the
desktop host chooses a source/destination; the parser receives bytes only. A
browser request posts a closed tagged `import` document or empty `export`
action to the authenticated `/controller-profile` endpoint, never a path or
URL, and unknown fields are rejected. Import first validates
and resolves one private document/runtime pair. The host can then persist and
install that exact pair as a transaction without re-resolving against a later
layer order. Use **Ctrl+Shift+I** in the native window to import and
**Ctrl+Shift+X** to export. Import records one native-manual history entry;
export changes no authored state.

`osc_config.json` selects the typed OSC listener/feedback peers. The default is
`127.0.0.1:9000`. A LAN bind requires explicit `enabled: true` configuration and
the **OSC RUNTIME** block always shows a LAN warning. Addresses use one closed
namespace, for example `/collide/v1/master/<parameter>`,
`/collide/v1/layer/<stable-id>/<parameter>`,
`/collide/v1/group/<group-id>/<parameter>`, and stable node/transport forms.
Authored Scenes add the canonical action paths
`/collide/v1/scene/<scene-id>/prepare` and
`/collide/v1/scene/<scene-id>/trigger`. An asserted float/integer/true message
is one OSC pulse; zero/false is an inert release. Trigger follows the Scene's
authored Immediate/next-beat/next-bar mode, so MIDI, OSC, native, and browser
launches share one timing law. Optional MIDI/OSC feedback reports `1` while
that exact Scene transaction is GPU-ready and `0` when it is absent, staging,
scheduled, or consumed.
Unknown parameters, arbitrary JSON actions, file selection, peer/bind changes,
and unbounded bundles are rejected. Datagram/string/nesting/fanout, event queue,
packet rate, and feedback rate are all capped. Feedback suppresses only its
source protocol or OSC peer, so an accepted change can still update the other
surfaces.

The default per-user documents are `controller_profile.json`, `osc_config.json`,
`preset_library.yaml`, `stage_map.yaml`, and
`recovery/recovery-v1.journal`. Invalid, unknown-field, or oversized input is a
read-only startup failure: safe defaults remain live and the hostile document
is not rewritten.

## Precision and external-capability boundary

The compatibility renderer remains byte-exact `LegacyCompat8`. The minimum
Advanced executor uses eight straight-linear RGBA16Float working surfaces and
25 Compat8 temporal surfaces: 24 clean-history frames plus one feedback image.
Spatial filtering and accumulation use premultiplied covered color at Advanced
boundaries. Only final audience presentation receives deterministic 8×8
ordered dithering; the separately prepared Compat8 history/feedback conversion
does not feed dither back into temporal memory. Full-16 temporal storage has an
exact resource ledger but is not an authored or implemented mode.

The local 192×108 Windows/Vulkan physical-GPU
[receipt](evidence/m6-precision-gpu-receipt.json) measures production still and
active-feedback paths. Advanced improves RGBA16F working RMSE and final 8×8
spatial-mean error, while final temporal pointwise RMSE/gradient direction is
worse under the intentional dither distribution; the receipt preserves both
results. Recorded one-shot wall times are smoke observations, not renderer
throughput evidence. The bounded still/temporal working and spatial-presentation
gains satisfy this evaluation's measured artist-relevant evidence gate without
a blanket subjective claim. A local receipt alone does not close the
cross-platform boundary: the exact published SHA must pass hosted Linux,
macOS, and Windows jobs with durable URLs.

The current content-addressed proxy work can key, assess, and preflight a future
bounded cache transaction without paths. It does not encode a proxy, mutate a
cache, or switch playback. Study schema 1 / ABI 1.0 validates at-most-1-MiB
typed SSA data with fixed read-only creative inputs and permanently denies
native code, shader injection, filesystem, network, process, device, and host
mutation. It is not a general plugin host, and a Study's data license cannot
license Collide-O-Scope.

Hardware/zero-copy decode, Syphon, NDI, external capture-input backends,
full-16 history, and
bounded mesh warp remain explicitly deferred until the required platform,
backend, policy/license, need, and interoperability evidence exists. See
[Precision and scale](precision-and-scale.md) for the exact ledger and decision
table. A schema field, platform assumption, build, or CI job is not runtime
proof of any deferred backend.

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
| **A slot or Scene will not activate** | Read its preparation/status text. One missing/stale layer, slot, or cue rejects the complete transaction; reload/prepare the source or recapture the Scene instead of expecting a partial cut. |
| **Recorder reports drops or cannot publish** | Check the fixed-pool/readback/worker counters, FFmpeg availability, destination permissions, and whether the chosen stable scope still exists. Finish drains admitted work; Cancel intentionally removes the temporary artifact. |
| **OSC is not listening or shows a LAN warning** | Inspect requested/bound address and counters under OSC RUNTIME. Loopback is the safe default; LAN requires explicit enabled configuration and remains visibly warned. Unknown addresses are rejected. |
| **One StageMap endpoint is unavailable** | Inspect that endpoint's surface/acquire/budget status and monitor binding. Failure is isolated; it does not fall back to Program output or another endpoint. Validate the actual display setup on the show computer. |
| **Spout layer stays black** | Confirm Windows, exact sender name, sender activity, matching GPU/Spout environment, and the layer's status text. |
| **Spout output is not visible** | Enable it, keep the app rendering, and verify with a real receiver or `cargo run --bin spout_probe`. |
| **Output window will not open** | Read the error shown under OUTPUT; confirm a usable monitor/GPU surface and try again. The switch reflects the actual closed state. |
| **Panel server is unavailable** | Read the native RECOVERY status and concrete bind error. Stop any earlier process holding ports 3030/3031, then restart. Browser count alone is not listener health. |
| **Library is empty or points at the wrong folder** | Check the active path in RECOVERY, choose the intended folder, and use Rescan. Choosing or rescanning does not add a layer automatically. |
| **A large source is rejected in Safe or Expert mode** | Safe intentionally stops above UHD area. Expert still enforces DCI-8K area, absolute/device edge, per-buffer, and aggregate host-planning limits; it is not a guarantee of available VRAM. Reduce the source or close other above-Safe sources. |
| **Selective VHS reports a memory budget error** | The requested output size and number of visible, positive-opacity layers exceed the bounded live working set. Hide unneeded layers, set their opacity to zero, or lower the output resolution. The engine holds the prior exact audience frame and does not violate Bypass Master FX. |
| **Bypass Temporal FX is rejected** | Put every dry layer in one contiguous prefix beginning at Layer 1/top and use a flat LegacyExact Program stack. Remove groups, A/B or non-Program bus routing, mattes, advanced racks or advanced Motion, authored Master/layer Motion modulation routes (including zero-depth routes), and routed Refresh Garden; disable VHS. The rejected transaction leaves the previous switch state and audience frame intact. |
| **Audio meters remain zero** | Select/enable a real input, grant OS access, and inspect the panel error. Software-only tests do not prove venue hardware. |
| **QR contains 127.0.0.1** | No LAN address was found at launch. Connect networking and restart. |

## Hardware validation boundary

Browser simulation, CPU fixtures, physical-GPU fixtures, and configured CI can
validate bounded contracts, protocol/layout, selected GPU math, and requested
build targets. They do not prove sensor axes, microphone drivers, MIDI/OSC
timing and feedback, FFmpeg installation, Spout interoperability, monitor
selection/fullscreen behavior, or an external signal chain. A CI definition is
also not evidence that every matrix job passed. Before a performance, validate
the actual phone, audio interface, MIDI controller/clock source, OSC peers,
encoder, Spout sender/receiver applications, StageMap, and display setup on the
exact show computer and stage hardware.

## Publication and license boundary

This fork is distributed under the GNU General Public License, version 3 or
later — see [../LICENSE](../LICENSE). Upstream's MIT grant is carried into the
combined work with its notice retained, and is not revoked by that choice; the
full boundary record is in [../COPYRIGHT.md](../COPYRIGHT.md). This is a
project boundary notice, not legal advice.
