# collide-o-scope

A native VDJ (video DJ) instrument for live visual performance. It composites
video, still-image, and live Spout sources with GPU effects, then lets LFOs, audio analysis,
MIDI, and a phone's touch and tilt modulate the performance from a browser
control panel.

> This is a fork of [collide-o-scope by Luis Queral](https://github.com/luismqueral/collide-o-scope)
> ([queral.studio](https://queral.studio)). The original engine, compositing
> architecture, and effect suite are his work. This fork adds the modulation
> matrix, remote control, prepared performance, Collision Rack, temporal and
> motion studies, professional control/stage tools, Spout I/O, offline export,
> and Windows support. See [LICENSE](LICENSE) for the precise licensing and
> attribution boundary.

<img width="1919" height="1040" alt="collide-o-scope" src="https://github.com/user-attachments/assets/3690c19f-8ab4-4a9d-9672-45476da4dede" />

## What it does

- Composites a dynamically sized stack of video, PNG/JPEG/BMP/WebP still, or
  live Spout layers with a curated 15-mode blend set, typed image routing,
  one-level groups, and A/B buses.
  The composition contract caps a stack at 256 layers; practical usable depth
  can be lower under source, GPU, output-resolution, motion, and selective-VHS
  resource plans.
- Applies per-layer and master pixelation, RGB split, hue, saturation,
  brightness, contrast, posterize, invert, film grain, vignette, color drift,
  breathing motion, seeded horizontal block Shift, luminance/chroma keying,
  and bounded animated cellular/Worley warping with a separately feathered
  cell-gap key.
- Applies one resolution-independent spatial transform at every layer and at
  master scope: Fit/Fill/Stretch/Native framing, crop, position, independent
  X/Y scale, source-space anchor, rotation, axis-directed skew, explicit edge
  behavior, and linear or nearest sampling. Live preview, selective VHS, and
  offline export use the same packed transform/effect pass.
- Runs [ntsc-rs](https://github.com/ntsc-rs/ntsc-rs) VHS simulation without
  blocking the render thread. Existing all-inherited stacks keep the global
  post-composite path; a contributing **Bypass Master FX** layer selects an
  exact per-layer path so VHS touches inherited layers only.
- Provides feedback trails and arbitrary-angle slit-scan from a 24-sample
  temporal history. History advances at a fixed 30 Hz, so its approximately
  0.8-second span does not change with display or export frame rate.
- Adds prepared clip slots, bounded cue/loop transport, and atomic Scenes;
  bounded Collision Racks at master, layer, and group scope; Temporal Topology
  Loom, Collision Atlas, Refresh Garden, and Collision Score; and deterministic
  motion fields, Faraday Motion Transplant, and Curved Shutter Sculpture.
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
- Renders MP4 files through a deterministic frame-indexed offline path and can
  publish a bounded motion-provenance sidecar. A separate nonblocking live
  recorder captures Program or an exact layer/group scope, takes PNG stills,
  and resamples durable results into a prepared clip slot.
- Provides bounded manual undo/redo, scoped presets, a recovery journal with
  explicit restore/discard, controller profiles, typed OSC, a preview-only
  health HUD, and a separately
  persisted StageMap. Real MIDI, phone, Spout, audio-device, encoder, and venue
  display behavior still require end-to-end validation on the intended hardware.

## Creative composition and prepared performance

The **Collision Rack + Groups** panel replaces a fixed effect order with a
small, preflighted graph while preserving the exact historical path for old
patches. A master, layer, or group rack contains at most eight stable-ID nodes:
Transform, Digital/Color, Key, Cellular, Shift, Grain, or geometric/image Mask.
Every node has enable, wet mix, and one of 15 curated blends. Legacy Canonical
and Legacy Temporal markers retain the established order during migration.
Unsafe sample counts, image dependencies, current-frame cycles, or GPU resource
plans are rejected before the live graph changes.

Entirely new studies and operators are intentionally never half-shipped. The
named Displace, Residual Counterpoint, Gesture-Field Etching, Symmetry Field,
Field Collider, native transform gizmo, and evidence-gated scale tranches that
were once planned here have all landed; the resource ledgers, acceptance
gates, and measurements that governed each landing are preserved per tranche
in the evidence notes and receipts under `docs/evidence/`.

The composition root can contain layers and up to 16 one-level groups. Groups
own contiguous members, opacity, transform, rack, matte, solo/bypass, and a
Program/A/B bus assignment. Root order and stable `GroupId`/`NodeId` references
survive edits; deletion leaves a visible missing-reference tombstone instead of
silently retargeting another image. The A/B fader linearly crossfades the two
premultiplied-linear bus accumulators while Program-bus items remain direct.

Layer mattes and image-mask nodes can read a selected layer before or after its
local effects, one layer below, all content below, the clean program, the
previous program-history frame, or a stable group output. Channel, invert,
amount, threshold, and softness are authored values. Same-frame dependencies
must be acyclic; explicit Previous Frame is exactly N−1.

Each layer's **Performance Set** owns up to 32 stable clip slots. Loading a file
into a slot probes, decodes, and uploads a replacement before it can become
active. Transport supports forward/reverse playback; Loop, Ping-Pong, One Shot,
or Hold endings; in/out points; rate and fixed sample-FPS policies; beat grids
and beat loops; and up to 64 numeric cues. Activation can be immediate, next
beat, or next bar. Up to 128 named Scenes bind prepared slots (and optional
cues) across as many as 256 layer positions; prepare/trigger is one atomic
transaction, so one bad or stale binding prevents a partial scene change.

## Temporal originals and motion

The fixed 30 Hz, 24-sample program history also powers four original studies:

- **Temporal Topology Loom** draws age as linear, radial, spiral, contour,
  folded, or kaleidoscopic surfaces with floor or linear history sampling.
- **Collision Atlas** assigns seeded temporal territories and controls their
  collisions.
- **Refresh Garden** admits and holds memory through temporal-delta, luma,
  chroma, cellular-ridge, audio-energy, audio-onset, matte, or motion gates.
  Matte selects a stable layer's pre- or post-local current-frame image;
  Motion reads the selected layer's actually admitted motion field, including
  its resolved lattice/codec source and any admitted donor transplant. Missing
  or unselected routes are explicit zero signals and never positional fallbacks.
- **Collision Score** advances a deterministic seeded state on a selected loop
  boundary, downbeat, audio onset, or explicit manual event. Separate reset law
  chooses whether loop/downbeat events clear Score, memory, both, or neither.

Temporal zero modes preserve the earlier renderer. Freeze, reset, event, patch,
Morph, modulation, Dice, and offline-frame laws are explicit and covered by
deterministic fixtures; this is software evidence, not a claim about every GPU.
Routed Garden admission runs in one dedicated post-temporal pass with at most
three sampled textures. Route identity and tombstones survive patch, Morph,
undo/redo, and layer reorder; removal cannot retarget a replacement layer, and
offline export resolves the same saved route against deterministic job-local
IDs while warning when a route remains closed.

**Refresh now** is an ordered counted Garden event, not an untracked UI pulse.
Accepted temporal events enter a bounded reference-tick track that offline
export replays at the same accepted-frame boundaries. **Clear event recording**
clears only that replay track; it does not alter authored temporal controls,
Garden/Score memory, or the audience image.

At master and layer scope, **Motion Fields** selects valid codec vectors,
deterministic Motion Lattice block matching, or Auto (visible lattice fallback
on layers; lattice on master). Draft/Live/High field tiers are fixed rather
than silently degraded under load. A layer can donate stable-ID motion to the
single admitted **Faraday Motion Transplant** carrier, and **Curved Shutter**
uses fixed Sharp/Draft/Live/High sample counts. Field, carrier, and transplant
resources are hard-bounded and a zero amount/angle is an exact bypass.
The live Motion status distinguishes the planned source from the field
actually committed by the renderer. First-frame lattice priming and an
unavailable source that has never committed remain visibly unattached; Media
Freeze retains and identifies the prior committed field. A Faraday recipient
reports the admitted donor field and its grid rather than inventing
recipient-local truth.

Offline export writes `<video>.motion.json` only after the video succeeds. The
bounded schema-v3 report records source fingerprints when available; the
requested export shutter policy and literal count; distinct authored and
effective scope/algorithm/quality/carrier choices; the final planner source,
the source actually rendered, and whether its field was attached; codec
transition count and elapsed source time only for an attached proven codec
field; an exact codec proof/vector digest; planned-but-unattached field truth;
dynamic-state changes; the last accepted frame; diagnostics; and
warnings. It explicitly sets cross-GPU pixel identity to false; it is
provenance, not a promise that different drivers render identical pixels.

## Spatial transforms

The master and every layer expose the same authored transform. Position is in
normalized composition coordinates; `[0, 0]` is centered. Scale is independent
on X and Y, negative values mirror, and the anchor is expressed in original
source UV coordinates. Changing only the anchor is visually inert: it chooses
the pivot used by scale, skew, and rotation rather than translating the image.
Positive rotation is clockwise in screen space. Skew has its own axis angle.

The forward order is crop and framing, authored scale, skew about its axis,
rotation, then position about the selected anchor. Fit shows the complete
cropped source, Fill covers the composition, Stretch reproduces the historical
full-frame mapping, and Native maps source pixels one-for-one to output pixels.
Transparent, Clamp, Repeat, and Mirror define out-of-bounds sampling; Linear
and Nearest select filtering. Collapsed or invalid transforms produce
transparent output rather than invalid GPU coordinates.

Old patches have no transform section and therefore retain the exact historical
inactive full-frame sample, including the original shader expression. Their
authored edge default is Transparent, so moving or shrinking that identity does
not smear a border into newly exposed canvas; Clamp remains an explicit choice.
The host-session **New layer framing** preference defaults to Fit; every future
interactive file, still, or Spout layer combines that choice with Transparent
edges, while existing and patch-recalled layers remain untouched. It is not
artistic patch state. The browser supplies
linked-scale editing, reset, copy/paste, and framing presets. Absolute
per-layer actions use immutable layer IDs, so reorder cannot redirect a late
transform edit.

Transforms are complete patch and Apply-look state, optional Morph A/B state,
continuous modulation destinations, and an opt-in part of Bounded-variation
Dice. Procedural generator v7 mutates continuous spatial values through
independent deterministic streams while preserving the saved discrete fit,
edge, and sampling choices. The same evaluated state reaches live and offline
rendering.

## The modulation matrix

Every expressive input follows one rule: sources enter the matrix, routings
shape and scale them, and the resulting offsets are applied to per-frame
copies. Base values—the values shown by sliders and stored in patches—remain
unchanged.

| Sources | Detail |
|---|---|
| 4 LFOs | Sine, triangle, saw, square, or deterministic sample-and-hold; musical divisions; tap-tempo or MIDI clock |
| Audio | Live input, Windows system-playback loopback, or deterministic looping WAV/MP3/FLAC/Ogg/Opus/M4A/AAC analysis; level; 3–8 configurable bands; onset, brightness, noisiness; and a 32-bin display spectrum |
| MIDI | Legacy four-CC learn surface plus a separately persisted typed profile: selected input/output, channel filters, note and absolute/relative CC sources, button modes, Start/Continue/Stop/24-PPQN Clock, and bounded feedback |
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
size/density/speed, spatial position/scale/anchor/rotation/skew/skew-axis/crop,
temporal feedback/slit-scan/history-key plus bounded Loom/Atlas/Garden values,
master/layer Curved Shutter and layer Faraday values, Morph position, and each
layer's opacity, speed, target FPS, key controls, spatial values, and continuous
effects. Stable-ID targets also cover compatible rack-node wet/numeric values,
group opacity/transform/matte values, and the composition A/B crossfade. This includes RGB
chroma targets/tolerance, key thresholds and softness, temporal key history,
and VHS edge-wave speed, tracking wave, composite/chroma noise, luma smear,
and sharpening. Selector/topology choices—image routes, static/temporal key
mode, rack order/blend, Temporal topology/gate/Score/reset law, Motion field/
quality/donor/carrier, and grain algorithm—remain deliberate discrete controls,
not modulation targets. The legacy patch target `layerN_key` is read as the canonical
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
seeds, then makes range-safe changes to continuous Digital/Color, analog-motion,
Shift, core Cellular, Temporal Loom/Atlas/Garden, and M4 Motion values for the
applicable scope. **Amount** sets their scale and **Grain mode** additionally
allows the grain algorithm and color-grain switch to change. **Transform** opts
bounded position, scale, anchor, rotation, skew/axis, and crop in; **Rack
values** opts compatible node wet/numeric values in; and **Composition values**
opts group opacity/transform/matte values (plus A/B crossfade for Everything)
in. These three are off by default. Pattern-only and automatic per-loop rerolls
never move those controls. Sources, topology, image routes, node IDs/order/
enabled/blend, group membership/solo/bypass/bus, layer opacity/visibility/blend/key,
master-FX bypass, transport, modulation routes, VHS, temporal topology/seeds/
Score/reset/loop-driver law, and Motion algorithm/source/quality/donor/carrier
law are never randomized.

**Master** targets the master pattern plus all four LFO seeds; **Everything**
also targets every current layer and can include compatible composition values.
**Group** changes only explicitly opted compatible rack/composition values for
the selected stable group. A layer card has its own seed and pattern-only
reroll. Leave **Exact seed** blank to advance deterministically
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

## Professional console and stage

Manual browser and native-editor work uses one bounded, gesture-coalesced
undo/redo history. A checkpoint includes the complete authored world—patch,
stable layer identities and selection, patch base directory, StageMap, preset
library, and controller profile—but never GPU/decoder/history pixels. MIDI,
OSC, LFO, audio, clip triggers, and Collision Score do not flood that history.
Undo/redo validates and durably publishes a candidate before changing the live
world or history cursor.

The separately persisted preset library stores values-only Transform, Rack,
Matte, and Group presets that preserve identity/topology, plus exact bounded
Controller Profile and StageMap documents. The append-only recovery journal is
patch-only, checksum-framed, capped, and tolerant of a truncated/corrupt tail.
Startup can offer its newest valid checkpoint, but never applies it without an
explicit **Restore checkpoint** action.

A typed controller profile is separate from the artistic patch. It can select
MIDI input and output devices, filter channels, map notes or absolute/relative
CC encodings and button modes to the closed control vocabulary, follow
Start/Continue/Stop/Clock, and send bounded feedback. Saved layer positions are
resolved once to stable live identities and never retarget after reorder. Typed
OSC uses the same address vocabulary and bounded queues/rates. It defaults to
loopback port 9000; LAN binding requires explicit configuration and always
shows a warning. OSC packets cannot choose files, change peers/bind authority,
invent controls, or dispatch arbitrary browser JSON.

MIDI ingress admits only exact supported wire shapes and rejects high-bit data
or wrong/extra lengths before Learn, clock, queues, decoder state, or events can
change. Controller-profile JSON import/export is bounded and portable: native
paths belong only to desktop pickers, while the browser contract can carry a
typed document or request an export but has no path authority. A candidate is
validated and resolved into one matching persisted/runtime pair before swap.
See [Controller profile JSON](docs/controller-profile.md) for the exact schema,
bounds, native shortcuts, and authenticated pathless browser endpoint.

The live recorder uses fixed readback/pool capacity and never waits for FFmpeg
on the render thread. Program recording taps the final audience after NTSC and
absolute blackout; still/resample targeting can instead name one exact stable
post-effects layer or group. Missing targets drop visibly rather than falling
back to Program. Native pickers choose destinations, publication is
create-new/sync/atomic-no-replace, and auto-import or a new prepared
`ClipSlotId` occurs only after durable success. Recording is currently
video-only; timing/drop truth and audio-clock correlation go in a bounded
`.recording.json` sidecar, but audio is not claimed to be muxed.

The stage-health HUD reports bounded frame percentiles/deadlines, per-layer
decode age/queue/drop health, output identity/mode, and known resource budgets.
Video decoder telemetry measures publish/consume age, the zero-or-one command
and completed-frame depths, published/consumed totals, overwrite/drop counts,
fixed-64 rolling decode/upload/publish-to-consume p95, and successful CPU upload wall timing without
allocating in the decode hot path; absent decoder telemetry remains absent.
It has an editor-preview-only permit and cannot enter audience, Spout,
recording, or export pixels. `StageMap` is a different persisted venue document,
not part of `PatchState`: up to 16 endpoints and 256 total slices can apply
source rectangles, four-corner perspective or bounded convex polygon masks,
edge feather, linear calibration, test cards, and endpoint identification.
Monitor-bound endpoints use independent windows/surfaces and failure states;
an unassigned endpoint remains offscreen and cannot be presented.

The exact persistence, failure, recorder, health, and StageMap contracts are in
[Professional console and stage](docs/professional-console-and-stage.md).

## Precision and scale boundary

Milestone 6 froze measurement and trust contracts rather than pretending that
every evaluated integration shipped. `LegacyCompat8` remains the byte-exact
compatibility path. The minimum Advanced executor uses eight straight-linear
RGBA16Float working surfaces plus 25 Compat8 temporal surfaces: the 24-frame
clean ring and one recursive-feedback image. Advanced filters and accumulates
covered color premultiplied at spatial/temporal boundaries, while final
presentation alone receives deterministic 8×8 ordered dithering; intermediate
Compat8 history/feedback conversion is not dithered. Full-16 temporal storage is
only a resource-accounted candidate.

The local 192×108 Windows/Vulkan physical-GPU receipt measures production still
and active-feedback shaders against an independent reference. Advanced improves
RGBA16F working RMSE for both workloads and improves final 8×8 spatial-mean
error, while final temporal pointwise RMSE/gradient direction regresses under
the intentional dither pattern; both facts are reported. Its one-shot wall
times are smoke observations, not comparative throughput evidence. See the
[receipt](docs/evidence/m6-precision-gpu-receipt.json) and
[precision boundary](docs/precision-and-scale.md). The bounded still/temporal
working and spatial-presentation gains satisfy this evaluation's measured
artist-relevant evidence gate without making a blanket subjective claim. A
local receipt alone does not close the cross-platform boundary: the exact
published SHA must pass hosted Linux, macOS, and Windows jobs with durable URLs.

Content-addressed proxy keys/plans and bounded playback observations are pure
assessment today—there is no integrated proxy encoder or cache worker. Study
schema 1 / ABI 1.0 is a closed, at-most-1-MiB data-only SSA format with fixed
read-only creative inputs and no native code, shader injection, filesystem,
network, process, device, or host-mutation authority. It is not a general plugin
ABI and its data license cannot license the host application.

Live proxy observations assess every measured video source, including ordinary
paths, and can recommend a proxy from decode, upload, age, drop, and queue
pressure. Only a content-addressed cache key/preflight requires an
already-verified retained `cos-sha256://` identity; ordinary paths do not cause
background or warm-path fingerprinting merely to obtain one.

Hardware decode, zero-copy decode, Syphon, NDI, external capture-input backends, full-16
history, and bounded mesh warping remain explicitly evidence-gated/deferred;
no menu, schema, platform assumption, or CI configuration is reported as a
working backend. See [Precision and scale](docs/precision-and-scale.md) for the
exact ledger, proxy law, Study validation, and capability evidence table.

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
- **StageMap endpoints:** separately persisted monitor bindings can open
  independent calibrated endpoint surfaces. They do not replace the legacy
  fullscreen Output control, and one endpoint failure does not authorize a
  fallback onto another monitor.
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

**Shutter samples** is a closed export request: **Authored per scope**, or an
exact **1 / 4 / 8 / 16 samples** for every active Curved Shutter scope. An
explicit count replaces only the authored quality tier after the frame's
Morph/modulation values are resolved; the same resulting count is used by the
candidate and final immutable plans before resource preflight. It cannot turn
on a 0° shutter, so zero angle retains exact delegation. Older clients omit the
field and therefore preserve authored per-scope tiers; arbitrary counts and
silent quality substitution are rejected.

An optional video layer can supply its first audio stream. That audio starts at
source time zero, plays once at 1×, and is independent of visual pause, speed,
modulation, and looping. It is padded with silence or trimmed to the requested
program duration, then muxed as AAC. This explicit policy avoids implying that
arbitrarily modulated visual transport can be represented by one audio tempo.

After a successful export, the exporter atomically publishes the bounded
`<video>.motion.json` provenance report described above. Cancel/failure removes
partial video and sidecar work. Missing source fingerprints and unavailable
codec-vector provenance become explicit warnings; the report never upgrades
those omissions into a reproducibility claim.

## Patches

`Ctrl+S` saves a YAML patch through a native dialog. The browser's **Capture
snapshot** writes a uniquely named YAML file under `patches/` without blocking
the render loop or overwriting an earlier capture. Recall has two deliberately
different paths:

- **Load snapshot…** (`Ctrl+O`) reconstructs the saved sources, layer order,
  prepared clip slots, and Scenes, then restores Collision Racks/composition,
  mattes and image routes, temporal/motion authoring, layer and program
  transport, both freeze states, modulation/input/LFO state, and Morph
  automation. The
  replacement is atomic across the visual stack and saved imported analysis
  audio: every file is resolved and the audio is fully decoded before commit,
  so a missing, invalid, or corrupt source leaves the current performance in
  place. A legacy patch with no modulation section preserves current audio state.
- **Apply look…** (`Ctrl+Shift+O`) keeps the current sources, layer identities,
  order, layer count, prepared slots/Scenes, speed/FPS/pause, per-video
  loop-reroll choices, both freeze states, BPM, modulation, and input state. It
  applies master values and maps saved layer values positionally: direct
  effects/keying/pattern seed, transform, motion values, matte values, opacity,
  blend, visibility, and Bypass Master FX. Racks require identical node
  topology; groups require the same stable ID, membership, rack signature,
  matte presence/route, and root identity. Apply Look copies compatible values
  while preserving live node/group/layer IDs, image routes, motion donors,
  Score loop driver, membership, ordering, solo/bypass, and bus assignment.
  Saved NTSC and Temporal sections apply when present; a legacy omission leaves
  that current section unchanged. Extra current layers remain visually
  unchanged and extra saved layers are reported unused. A stack change while
  the picker is open rejects the transfer. An engaged current A/B Morph is
  materialized and cleared first; the patch's Morph is not imported.

After a successful Apply Look, actions that could overwrite its applied master,
mapped-layer, reroll, topology, and present NTSC/Temporal scope are discarded,
including conflicting input queued while the native picker was open. Unrelated
transport/safety actions, unmapped-layer edits, and edits to an omitted section
keep their order. Cancelling, failing, or rejecting a stale picker is not a
barrier.

`Ctrl+E` opens the native patch parameter editor; the file itself remains
ordinary YAML and can also be edited in a text editor. Current snapshots
include:

- master, per-layer, NTSC, Temporal Originals, and Motion values, including
  spatial transforms, pattern seeds, reset/quality laws, and stable donors;
- typed Collision Racks, image taps/mattes, the one-level group/root graph,
  Program/A/B assignments, and crossfade;
- layer order, visibility, pause, blend/keying/master-FX bypass, and the complete
  prepared Performance Set: stable clip slots, source identities, transport,
  cues, active slots, and atomic Scenes;
- Freeze Program, Freeze Media, and complete modulation state, including LFO
  sample-and-hold seeds;
- routing curves/slew, audio band count/crossovers/ceiling, gyro calibration/configuration, and XY
  configuration/current position;
- Morph A/B slots, crossfader law/position, and remaining beat glide.

`PatchState` is an additive typed YAML document, not the generator manifest.
Omitted fields keep explicit compatibility defaults and legacy filename/slit
axis fallbacks. `visual_schema_version: 1` declares explicit M2 rack/composition
topology; `0` or omission is the exact pre-rack layout synthesized with legacy
markers, and unknown future visual-topology versions are rejected. New seeds
default to `0` (the legacy pattern family), while prepared performance,
Temporal Originals, Motion, per-video loop reroll, Scenes, and Freeze Media
default to their exact inactive/legacy laws. StageMap, controller/OSC documents,
preset library, recovery journal, recorder state, runtime pixels, and GPU
resources are intentionally outside `PatchState`.

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

Generator v7 resolves and SHA-256 fingerprints visual and imported
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
mean-reverting mutations in isolated deterministic domains, including Shift,
bounded spatial values, compatible rack/group/matte/A-B values, Temporal
Originals numeric values, and Motion numeric values. It preserves
source/layer/routing/rack/composition/image-route/motion-donor topology and
discrete temporal/motion laws; rejects active two-slot Morphs and in-flight glides; and requires
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
| Ctrl+Shift+I / Ctrl+Shift+X | Import / export the portable controller profile JSON |
| Esc | Quit or close output window as appropriate |

## Validation boundary

Software, browser, and physical-GPU fixtures plus configured CI establish only
their tested contracts and requested build targets. They do not prove a
particular machine's MIDI/OSC devices and timing, phone sensors, venue audio,
FFmpeg installation, external Spout applications, monitor/fullscreen behavior,
or signal chain. A CI configuration is not itself a passed matrix. Validate the
complete performance on the exact show computer and stage hardware; do not
treat a successful build or shader fixture as hardware proof.

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
