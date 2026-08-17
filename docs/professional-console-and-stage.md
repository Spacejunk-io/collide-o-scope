# Professional console and stage

Milestone 5 turns the browser and native editor into transactional operator
surfaces, while keeping artwork, venue calibration, controller mappings, and
runtime memory in separate ownership domains. This document describes those
boundaries and the failure behavior they guarantee.

## Manual history

Manual browser and native-editor changes share one bounded undo/redo history.
A pointer or keyboard gesture has an explicit begin/end boundary, so all
coalesced slider updates inside it become one entry. An old client that sends a
single absolute edit without gesture boundaries still creates one entry.

Only browser-manual and native-manual origins are recorded. MIDI, OSC, LFO,
audio, clip triggers, Collision Score, and host automation update the live
instrument without flooding manual history. Browser and native gestures cannot
overlap; disconnecting a browser mid-gesture closes its dirty gesture in order.

Each checkpoint is an exact authored world: the patch, runtime-stable layer
identities and revisions, selection, patch base directory, StageMap, preset
library, and controller profile. It never contains GPU textures, decoded
frames, temporal carriers, recorder buffers, or other runtime pixels.

Undo and redo are two-phase transactions. The candidate is validated and
preflighted as a complete world, its separate documents are durably published,
and only then is it installed and the history token committed. A rejection
leaves both live state and history depths unchanged. Limits are 128 entries,
64 MiB total, and 8 MiB per checkpoint; oldest checkpoints are evicted first.

## Scoped presets

The separately persisted preset library contains at most 128 entries and 8 MiB.
Its creative presets are values-only and identity-safe:

- **Transform** copies the complete bounded spatial value.
- **Rack** requires compatible node topology and copies values while preserving
  node IDs, routes, donor identity, and allocator cursors.
- **Matte** copies bounded values/channel/invert while preserving the target's
  image route.
- **Group** copies values for opacity, transform, rack, matte, solo, bypass, and
  bus while preserving GroupId, membership, name, root position, and donors.

Controller Profile and StageMap presets contain exact validated typed
documents. Applying either is atomic and does not smuggle those documents into
artistic `PatchState`. Capture and delete publish a candidate library through a
same-directory temporary file, file sync, atomic replacement, and parent sync
before changing the live revision.

## Recovery journal

The recovery journal is append-only, bounded, and patch-only. Each binary v1
record carries a monotonic sequence, payload length, SHA-256 checksum, and one
canonical serialized `PatchState`. It never stores or overwrites the user's
patch path.

- payload limit: 8 MiB;
- journal limit: 256 records / 64 MiB;
- truncated or corrupt tails are ignored after the last valid prefix and are
  reported visibly;
- startup offers recovery but never applies it automatically; and
- compaction uses a synced same-directory temporary file and atomic replace.

A successful manual checkpoint or native patch save appends recovery. Restore
still passes through the unified patch preflight; discard is explicit.

## Controller profiles and OSC

Controller profiles and OSC configuration are bounded documents outside the
patch. Persisted layer references use saved positions, resolve once to live
`StableLayerId`s, and never retarget after reorder. Group and rack-node targets
retain `GroupId` and `NodeId` identity.

The MIDI supervisor supports independently selected input/output devices,
channel filters, note and CC sources, absolute and three relative CC encodings,
momentary/toggle/gate buttons, Start/Continue/Stop/Clock, bounded feedback,
hotplug rescans, and visible queue/reconnect/drop counters. The legacy four-CC
surface remains the default profile. Its complete-message boundary accepts
only exact three-byte Note Off, Note On, and Control Change packets, or exact
one-byte Start, Continue, Stop, and Timing Clock packets. A missing/extra byte,
a channel-data high bit, running-status fragment, or unsupported status counts
once as malformed and cannot touch learn state, CC values, clock state, raw
queues, decoder button state, or emitted events.
Here, “malformed” is relative to this deliberately closed Collide controller
protocol, not a claim that every other MIDI-spec message family is malformed on
the wider MIDI wire.

Controller-profile JSON import/export has one portable 256-KiB document cap.
Native destinations and sources come only from Main-owned pickers; the shared
byte API receives no ambient path authority. The browser-safe tagged action is
data-only (`import` with a typed document, or an empty `export` request), has a
bounded envelope, denies unknown fields, and contains no path/URL/action-string
escape hatch. Import validates and resolves the document into one inseparable
document/runtime pair before any live revision, MIDI worker, or persisted file
can change. Export serializes only that validated portable document.
The native affordances are **Ctrl+Shift+I** for import and **Ctrl+Shift+X** for
export. Import is a `NativeManual` history transaction; export is read-only and
does not create a history entry.

OSC uses the same closed typed control-address vocabulary. It accepts bounded
messages/bundles only; packets cannot select files, change peers, change bind
authority, or invent parameter names. The default is loopback port 9000. LAN
binding requires explicit document opt-in and always publishes a warning.
Datagram size, string size, nesting depth, message fanout, event queues, and
packet/feedback rates are bounded. Feedback suppresses only its source protocol
or OSC peer, so accepted host changes can still update other surfaces.

Default document locations are under the platform's per-user state directory:

- `controller_profile.json`;
- `osc_config.json`;
- `preset_library.yaml`;
- `stage_map.yaml`; and
- `recovery/recovery-v1.journal`.

Invalid or oversized files are read-only failures: the app starts from safe
defaults and reports the reason rather than rewriting hostile input.

## Nonblocking recording, stills, and resampling

Recording is a bounded worker transaction, not the offline exporter. The render
thread never waits for FFmpeg and never allocates recorder frame buffers on the
warm path. Four fixed RGBA buffers feed a queue of two; pool/queue/source misses
increment explicit drop counters. Capture metadata is frozen at the GPU copy,
including capture/program clocks, visual epoch, freeze/blackout facts, and an
optional audio-clock correlation stamp. The current recorder is video-only and
does not claim to mux audio without a bounded program-PCM source.

Program capture taps the final audience image after NTSC and absolute blackout.
Layer/group capture taps the requested stable post-effects scope. A missing or
deleted scope drops visibly and never falls back to Program. Program Freeze,
Media Freeze, held frames, and blackout capture the exact pixels that were
materialized.

The browser never supplies filesystem paths. Native pickers choose a final
destination, and the worker writes a random create-new file in that directory.
Success requires encoder completion, file sync, and paired publication of the
media and its bounded `.recording.json` truth sidecar via atomic no-replace
commits.
Cancel and final publication share one nonblocking atomic gate: whichever
claims the transaction first wins. A winning Cancel removes both temporary
artifacts; once publication has claimed the gate, a later Cancel cannot relabel
or interrupt the commit, and the worker reports its actual terminal result.
Auto-import and resampling are emitted only after durable success; resampling
allocates a new `ClipSlotId`, prepares it through the normal media transaction,
and activates it only after preparation. Stills obey the same paired
publication law; their sidecar preserves the exact frozen capture metadata and
declares a single-frame/no-drop policy.

## Health HUD and StageMap

The health HUD exposes bounded FPS and p50/p95/p99 frame time, missed deadlines,
per-layer decoded age/queue/drop health, active output identity/resolution/rate,
and known media/performance/NTSC/motion budgets. Its painter requires an
`EditorPreview` permit; audience, Composite, Spout, recording, and export paths
cannot call it.

Decoder health is measured at the newest-only mailboxes rather than inferred
from display FPS. Each video source publishes allocation-free last-publish and
last-consume ages, pending command depth (zero or one), command and completed-
frame depth (zero or one), monotonic published/consumed totals, overwrite/drop
counters, and decode/upload sample, last, lifetime-peak, and rolling-p95 wall
time. Rolling p95 uses fixed 64-sample rings; successful harvests also record a
fixed-ring p95 from exact publish to consume, and pending-frame peak remains
bounded to one. The upload interval covers the validated queue-write/error-
scope seam; it
is explicitly CPU wall time, not a claimed GPU timestamp. Still and live inputs
report telemetry as unavailable instead of manufacturing zero-valued decoder
facts.

Live proxy assessment uses those measurements for every video source. A source
without a retained identity can still report `OriginalSufficient`,
`MeasurementRequired`, or the objective reasons a proxy is recommended. Only
the content-addressed cache key/preflight binds to an already-retained, valid
`cos-sha256://<digest>/<bytes>` identity. The layer accessor parses that
reference in memory; an ordinary or malformed host path yields no cache identity
and never triggers warm-path file fingerprinting.

`StageMap` is a separate venue document, never part of an artistic patch. It
supports up to 16 named endpoints, 64 slices per endpoint, and 256 slices total;
slices use source rectangles, four-corner perspective or bounded convex polygon
masks, edge feather, and linear calibration. Test cards and output-identification
are endpoint-specific venue tools, not creative pixels.

The GPU presenter cold-prepares fixed RGBA8-sRGB endpoint textures, uniforms,
pipelines, and bindings. Warm encode creates no GPU objects. Every monitor-bound
endpoint owns an independent window, compatible surface, acquire result, and
present call. A missing, closed, lost, invalid, or over-budget endpoint is
reported without preventing another endpoint from rendering. Unassigned
endpoints remain offscreen and cannot be surface-presented.

## Validation boundary

CPU fixtures run in the ordinary suite. The opt-in StageMap physical-GPU suite
was run on 2026-08-17 with:

```text
cargo test --locked renderer::stage_map::tests::physical_gpu_ -- --ignored --nocapture --test-threads=1
```

It passed 5/5 on an AMD Radeon RX 6950 XT through wgpu's Vulkan backend, AMD
proprietary driver 26.7.1. The fixtures exercise warm allocation invariance,
projective/polygon/feather/calibration math, test-card and identity pixels,
independent endpoint textures, RGBA/BGRA surface-format conversion, and failure
isolation. These opt-in tests are skipped by ordinary CI, and this one-adapter
receipt does not prove a particular venue's monitor selectors, OS fullscreen
behavior, MIDI/OSC hardware, encoder installation, other adapters/backends, or
external signal chain. Validate those on the actual show computer and displays
before a performance.
