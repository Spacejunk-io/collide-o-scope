# §3.4 — performance-recorder v2 addresses: evidence note

Date: 2026-08-27. Status: hosted implementation and exact local gate observed.
The labeled GPU/export receipts below are also observed; topic commit,
integration commit, and exact-commit CI remain pending until the branch
closes.

## Claim and compatibility boundary

This tranche extends B9's closed performance-address vocabulary to the value
families its v1 note deliberately deferred: visual-rack node values,
composition-group and group-matte values, text-page values, Morph law/glide/
capture, and existing-routing configuration. “v2 addresses” names this
append-only capability tranche; it does **not** introduce a v2 take document.
`PERFORMANCE_ALGORITHM_VERSION` remains 1,
`PERFORMANCE_CHECKSUM_DOMAIN` remains
`collide-o-scope/performance-take/v1\0`, and an event remains exactly
`tick:u32le | address:u16le | value:u16le`. The existing address/event caps,
value law captured at first sight, serialized-byte cap, honesty flags, and
checksum rules are unchanged. One bounded admission limit intentionally grows:
`MAX_PERFORMANCE_VOCAB_TOKENS` is 64 rather than 32 because the complete
engine-owned routing-source vocabulary contains more than 32 canonical and
compatibility tokens. The document remains bounded and hostile vocabularies
above 64 are refused.

Codes 0–14 are frozen exactly as shipped by B9. The new canonical codes and
wire tokens append after them:

| Code | Typed control | Wire token | Stable identity |
| ---: | --- | --- | --- |
| 15 | `RackNodeMaster` | `rack_node_master` | node ID + node kind + parameter |
| 16 | `RackNodeLayer` | `rack_node_layer` | saved layer position + node ID + node kind + parameter |
| 17 | `RackNodeGroup` | `rack_node_group` | group ID + node ID + node kind + parameter |
| 18 | `GroupParam` | `group_param` | group ID + parameter |
| 19 | `GroupMatteParam` | `group_matte_param` | group ID + parameter |
| 20 | `LayerText` | `layer_text` | saved layer position + parameter |
| 21 | `MorphLaw` | `morph_law` | closed law token in the event lane |
| 22 | `MorphGlide` | `morph_glide` | Q16 target in the address; duration in the event lane |
| 23 | `MorphCapture` | `morph_capture` | closed A/B slot token in the event lane |
| 24 | `Routing` | `routing` | saved routing position + parameter |

Node and group IDs serialize as canonical non-zero decimal strings so JSON
does not lose `u64` precision; leading zeroes, zero, signs, overflow, missing
identity fields, and surplus identity fields are refusals. The canonical
checksum tail encodes those identities as fixed little-endian binary, includes
the node kind where applicable, and remains unambiguous. Layers and routings
bind by saved position when playback is armed. Rack playback also requires the
saved node ID and node kind to match, so an edited topology degrades by name
instead of silently retargeting another node.

## Engine-owned value laws

The recorder does not maintain a second parameter registry. Each new address
asks the module that owns the live edit for an `AuthoringValueLaw`, converted
without widening into the existing `PerformanceValueLaw`:

- `visual_rack::node_param_value_law` reads the rack kind/control/parameter
  descriptors and the owning enum tables. Bounded floats, booleans, bounded
  integers, and closed enum tokens are recordable. Vectors, colours, image
  taps, motion donors, routes, Study documents, legacy marker bodies, and
  unbounded seeds have no one-event scalar law and are counted refusals.
- `composition::group_value_law` delegates transform fields to
  `spatial::spatial_transform_value_law` and owns opacity, solo, bypass, and
  the `BusAssignment::ALL` vocabulary. Group name is excluded.
  `composition::group_matte_value_law` admits only amount, threshold, and
  softness; matte source/channel/invert remain one topology-bearing edit and
  are excluded.
- `text_page::text_page_value_law` reads the text-page scalar ranges,
  `TextPageFont::ALL`, `TextPageShape::ALL`, and the repeat/shape-count integer
  bounds. The UTF-8 body is deliberately refused: one recorder event carries
  one scalar or one closed token, not a bounded document.
- Morph law and capture use `MorphBlendLaw::ALL` and
  `MorphCaptureSlot::ALL`; glide target is Q16 on `[0,1]`, and duration uses
  the owner's `[0,64]`-beat law, preserving zero as snap and the owner's
  positive-duration normalization.
- `modulation::routing_value_law` admits source, depth, curve, curve amount,
  attack, and release. Source vocabulary comes from the monitor-ordered
  `MONITOR_SOURCE_LIST` plus accepted legacy aliases, curve comes from
  `Curve::ALL`, and numeric ranges come from the routing owner constants.
  Target is refused because its vocabulary depends on the current stable
  address world; add/remove/count operations remain topology and have no
  performance address.

Every `None` is observable through the existing unsupported/rejected counters.
No fallback range, unknown-token coercion, node target/body/composite payload,
routing target, topology operation, safety transport, or unbounded identity
value is recorded.

## Capture, live replay, and export replay

Capture remains at D4's accepted-creative-mutation boundary after the ordinary
wire action has passed authentication, revision/stable-ID checks, owner parsing,
Morph ownership transfer, planner/preflight, and actual state change. It does
not add a second drain tap: coalesced-away, stale, refused, replay-origin, and
dropped batch work still records nothing, and same-frame canonical duplicates
still collapse independent of source.

Live playback compiles the new controls to the same real `WebAction` families
the operator uses. It resolves saved layer/routing positions once at arm,
checks group/node identities and node kind before compilation, and then
dispatches through the existing source-aware application seam. Text edits use
the shared text-page edit/raster path; Morph law/glide/capture use the Morph
owner's strict token and beat-space laws; routing configuration mutates only an
already-bound routing. Replay remains excluded from rerecording and manual
history.

For codes 15–24, both live arm and export admission independently require the
serialized value law to equal the current engine-owner oracle before decoding
or dispatch. The hashed take law determines quantization but is not authority
to widen a lattice or make an excluded seed, group name, text body, or routing
target recordable. Codes 0–14 retain their stored-v1 admission law unchanged.

Offline replay applies the same typed value laws and shared owner appliers to
the export creative graph. Export text pages retain their authored parameters
and reraster only after an accepted text value edit. Morph law and glide are
materialized against the export beat, and Morph capture builds the full slot
from a CPU-only view of the current export layers/composition/motion state,
including the recorded gesture-track checksum; it does not require live
GPU-backed `Layer` objects and does not fabricate a partial slot. Routing
replay mutates the saved routing position only and never changes its target or
topology. A bad or absent identity is a named no-op/refusal, never a retarget.

## Frozen-v1 proof anchor

`tests/fixtures/performance-take-v1-brightness.json` is a literal stored take,
not a take regenerated by current code. It contains one master-brightness event
at tick 7, declared length 12, and checksum:

`be4bb410f3984214fc13667f4135208d089d14aefbbff2fb2f6e19ff5a0758d6`

The compatibility test must decode it, validate it, replay the expected value,
and re-encode the document byte-for-byte while retaining that digest. The
append-only code/token test pins every code 0–24; hostile-serde tests pin the
new identity fields and decimal-ID law; canonical-tail tests pin binary,
collision-free identities. These tests are the hard guard against shifting an
old lattice or assigning a prior code a new meaning.

## Provenance and repository boundaries

The recorder architecture retains the B9 provenance: BENDR (MIT, © 2026 Steve
Blythe, `p42_capture.js`) supplied the gesture-not-pixels design law; the house
adaptation records accepted edits at a 30 Hz reference tick and carries a take
inside its opening patch. This §3.4 work is a house-authored extension over
existing Collide-O-Scope engine tables and appliers. No new third-party source,
asset, model output, or license obligation is introduced by the address
extension.

The following untracked root artifacts are protected operator material and are
outside every test fixture, provenance input, hash, archive, cleanup, and
commit operation for this tranche:

- `.da-vinci-canon-pre-refinement-backup-20260822.zip`
- `4K_Nature_Cinematography_recorded_with_Nikon_D5300.webm.1080p.vp9.webm`
- `Black_swan_(Cygnus_atratus).webm.1080p.vp9.webm`

`videos/audit.mp4` is absent. It must not be invented, copied from another
asset, or silently replaced. Any older opt-in fixture that requires it remains
honestly unrunnable until the operator re-provisions that exact fixture; the
§3.4 labeled proof should be self-provisioning.

## Verification and receipt

Hosted proof must cover, at minimum:

- frozen codes/tokens 0–24, hostile address shapes, canonical identity tails,
  value-code validation, and literal stored-v1 byte/checksum compatibility;
- owner-law coverage and explicit refusals for each rack, group/matte, text,
  Morph, and routing family;
- accepted record → commit → live replay round trips for representative values
  in every new family, including stable-position binding, stale/mismatched
  identity degradation, replay/undo exclusion, and unchanged duplicate/tick
  laws;
- export application through the shared rack/group/text/Morph/routing appliers,
  including text reraster, routing-target refusal, and full portable Morph
  capture rather than a refusal or partial slot;
- the pre-existing B9/D4 recorder, sidecar, patch, transport, generation-
  barrier, and source-origin suites with no regression.

The exact repository gate is:

```text
cargo fmt --all -- --check
node --check static/app.js
node --check docs/ui-ux/wireframe.js
cargo check --locked --all-targets --all-features
cargo test --locked --all-targets --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
```

Focused hosted proof observed before the exact gate:

- `cargo test --locked --all-features "v2_" -- --nocapture`: 12 passed,
  0 failed, 2 explicitly ignored GPU cases;
- `v2_layer_addresses_bind_at_arm_and_follow_the_stable_layer_through_reorder`
  run explicitly with `--ignored`: 1 passed, exercising an actual text layer,
  layer rack, post-arm reorder, stable-ID resolution, and replay exclusion;
- literal stored-v1 fixture: 1 passed with exact bytes, decoded brightness, and
  checksum `be4bb410f3984214fc13667f4135208d089d14aefbbff2fb2f6e19ff5a0758d6`;
- production export staging helper: 1 passed for master/layer/group racks,
  group/matte values, missing IDs, and mismatched node kinds;
- `export_v2_morph_capture_preserves_the_complete_portable_world` run
  explicitly with `--ignored`: 1 passed, directly asserting the captured
  layer values, master/layer/group racks, composition group and matte, master
  values, and gesture-track identity.

The labeled visual/GPU/export receipt is the self-provisioning
`render_performance_recorder_v2_pipeline`. It ran explicitly with the shared
FFmpeg 9.0.1 build on `PATH` and passed: the text-page patch contains
master/layer/group racks, group/matte state, Morph capture, and routing; the
take changes decoded frames relative to `_untaken`; `_repeat` decodes
identically; both taken sidecars byte-verify; and the untaken render publishes
no performance sidecar. The fixture neither requires nor creates
`videos/audit.mp4`.

Closing fields:

- Topic commit: **PENDING**
- Integration commit on `feat/web-control-panel`: **PENDING**
- Exact-commit CI: **PENDING**
- Hosted gate result/toolchain: **OBSERVED PASS** — the exact six-command gate
  passed with 2,123 tests passing, zero failures, and only the explicitly
  external/GPU seats ignored; `rustc 1.98.0 (88d9e12ae 2026-08-18)`,
  `cargo 1.98.0 (797e8a9bc 2026-08-05)`, Node.js `v26.5.1`, Visual Studio
  Developer Command Prompt `17.14.38`, and shared FFmpeg
  `9.0.1-full_build-www.gyan.dev`
- Labeled GPU/export receipt: **OBSERVED PASS** — self-provisioned v2 taken /
  untaken / repeat matrix under FFmpeg 9.0.1
- Protected-root status after integration: **PENDING RE-CHECK**

## Deliberate non-claims

This tranche does not change the take schema/version/domain, event width,
address/event/serialized-byte capacities, recorder UI/transport vocabulary,
modulation of a take,
topology recording, safety-control recording, route-target recording, raw
browser/MIDI/OSC packet recording, or automatic fixture minting. It does not
claim that arbitrary text bodies fit in a scalar event, that a missing saved
identity can be rebound by proximity, or that the vocabulary-token bound stayed
at 32: it is deliberately 64 for the complete routing-source owner table.
