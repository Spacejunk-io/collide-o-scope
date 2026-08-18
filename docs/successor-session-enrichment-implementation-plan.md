# Successor-session enrichment implementation plan

Status: implementation specification; none of the elements below are claimed
as shipped by the repair release that produced this document.

This plan covers the genuinely new creative elements and external-runtime
tranches that remain after repairing the already-delivered M3–M6 systems. It
is deliberately separate from defect repair: a successor session must land
each element as a complete authored-state, planner, renderer, live/export, UI,
and evidence transaction. A pure type, an isolated shader, or a hidden test
harness is not delivery.

## Non-negotiable laws

Every tranche must preserve all of these boundaries:

- `LegacyExact` stays byte-for-byte compatible, including active legacy
  spatial sampling. An exact default or zero-wet configuration must delegate
  before allocating or encoding a new pass.
- Runtime identities are stable nonzero IDs. Persisted relationships use saved
  positions and preserve `Selected` versus `Missing`; missing routes never
  silently rebind after reorder, removal, replacement, Morph, or patch load.
- Live and offline rendering consume the same immutable evaluated plan. Export
  may choose an explicitly requested quality, but it may not invent a hidden
  source, seed, gesture, carrier, route, or sample count.
- Working storage remains straight-linear RGBA16F in Advanced; filtering,
  interpolation, and accumulation occur in premultiplied space. Hidden RGB at
  alpha zero must not leak.
- Resource admission happens before GPU allocation. Warm frame execution must
  create no textures, buffers, bind groups, pipelines, vectors, or strings.
- No new per-layer full-resolution history ring is allowed. Every texture and
  buffer must appear in the canonical byte ledger and actual-allocation
  reconciliation.
- A pass may sample at most three textures unless a new portable limit is
  justified, preflighted, and proven on all supported backends.
- Patch, Look, Morph, modulation, Dice, procedural generation, manual history,
  recovery, presets, browser ingress, native editing, telemetry, and export
  provenance must be closed over the new authored state.
- Browser values use closed tagged vocabularies and strict finite bounds.
  Topology edits are ordered, revision-protected, uncoalesced barriers and are
  forbidden inside Quantized batches.
- Determinism is reference-tick/event based, never wall-clock based. GPU pixel
  identity is claimed only where a named physical fixture proves it.

## Dependency order

1. Named two-input **Displace** node.
2. **Residual Counterpoint** and **Gesture-Field Etching** foundations, in
   parallel once Displace route semantics are frozen.
3. **Symmetry Field**, including its complete stable sector-route table.
4. **Field Collider**, after Symmetry and the primitive motion-field contract
   are frozen.
5. Native preview transform manipulation.
6. Evidence-gated precision/scale runtimes (proxy worker, external I/O,
   bounded mesh warp, and Study execution) as independent tranches.

Do not merge adjacent tranches merely because they share an enum or shader.
Each must have a separately reviewable resource delta and exact-zero proof.

## 1. Named two-input Displace

### Authored and runtime model

Add `DisplaceBoundary::{Transparent, Mirror, Wrap, Hold}` and saved/runtime
payloads `DisplaceParams` / `RuntimeDisplaceParams`:

- stable image `tap` route;
- independent finite `amount_x` and `amount_y`, each clamped to `[-1, 1]` UV;
- boundary default `Transparent`;
- neutral donor encoding `RG = 0.5`;
- exact bypass when wet is zero or both amounts are zero.

Add `VisualNodeKind::Displace`, `RuntimeVisualNodeKind::Displace`, and
`NodeKindTag::Displace` with a permanent append-only kind code. Capture and
resolve the route using the generic stable image-tap laws. Missing, removed,
or self-dependent routes remain visible diagnostics and never fall back to
the current layer or a positional slot.

### Planner and renderer

Use the existing carrier/donor/sampler ABI and zero texture. The donor vector
is alpha-covered:

```text
vector = (premultiplied_rg - 0.5 * alpha) * 2
```

Thus transparent hidden RGB and a missing donor both yield exact zero. Filter
the carrier manually in premultiplied space and apply all four boundary laws
without allocating a new full-frame persistent surface.

Descriptor budget per active node:

| Item | Exact charge |
|---|---:|
| Render passes | 1 |
| Logical lookups/pixel | 3 |
| Explicit texture operations/pixel | 12 |
| Simultaneously sampled textures | 2 |
| Cross-scope image taps | 1 |

The planner collects the tap only when the node is enabled, wet is positive,
and at least one amount is nonzero. Existing current-frame cycle rejection is
unchanged.

### State and controls

- Morph interpolates amounts only when routes match exactly and selects the
  boundary discretely at endpoints.
- Dice/procedural generation may mutate only the two amounts; route and
  boundary are stable authored topology.
- Modulation exposes `amount_x` and `amount_y` only.
- Browser uses a dedicated ordered action:
  `SetVisualNodeDisplaceRoute { scope, node_id, route, composition_revision }`.
- Snapshot params are `{ donor_tap, amount_x, amount_y, boundary, diagnostic }`.
- Export records resolved/missing route identity and boundary; it uses the
  shared node implementation rather than an export-only effect.

### Required acceptance

- exact default/wet-zero/amount-zero delegation and frozen Legacy pixels;
- analytic ±X/±Y fixtures for every boundary;
- transparent donor with hostile hidden RGB produces zero displacement;
- reorder, removal, replacement, self-cycle, and stale-revision rejection;
- patch/Look/Morph/modulation/Dice/generator/history/recovery round trips;
- live/export GPU equality and warm-allocation invariance;
- logical and explicit texture-operation ledger tests.

## 2. Residual Counterpoint

### Creative law

Given two stable inputs, compute a low-resolution block mean for each, split
each image into DC and AC components, apply seeded fixed quantization, then
recombine one input's large-scale structure with the other's detail in a
single linear-premultiplied pass. Exact default delegates without allocation.

The initial bounded vocabulary should retain:

- algorithm version 1;
- fixed block-size enum and fixed quantization enum;
- two stable image/history-age routes;
- finite mix/quantization controls and explicit seed;
- missing-route diagnostics with no positional fallback.

### Frozen resource proposal

| Item | Bound |
|---|---:|
| Active nodes | 16 (subject to the aggregate byte cap) |
| Block-grid edge | 2,048 per dimension |
| Block cells | 2,100,000 per node, independently enforced |
| Bytes per block-mean cell | 8 |
| Block-mean surfaces/node | 2 |
| Bytes/node | 32 MiB |
| Aggregate bytes | 64 MiB (therefore at most two full-cap nodes) |
| Mean sample operations/full-cap node | 16,800,000 |
| Aggregate mean sample operations | 33,600,000 |
| Recombination sampled textures | 3 |
| Uniform stride | 256 bytes |

No output-sized history is permitted. Node count, each dimension, cell count,
per-node bytes, and aggregate bytes are independent checked limits: admitting
16 nodes is possible only at smaller grids, while no more than two nodes can
reach the full cell cap. Admission must include both mean surfaces, uniform
arena, staging/transient bytes, and the final pass's logical and explicit
texture operations.

### Required acceptance

- independent CPU reference for DC/AC decomposition and recombination;
- constant-color/DC-only, zero-mean/AC-only, edge, transparent-hidden-RGB,
  quantization, and fixed-seed fixtures;
- exact default and zero-mix delegation;
- route tombstone/reorder/cycle tests;
- live/export GPU equality, reset/freeze laws, and warm allocation;
- complete state/control integration and bounded sidecar provenance.

## 3. Gesture-Field Etching

### Portable event contract

Use a low-resolution signed vector canvas driven by a bounded reference-tick
track, not pointer wall time. Version 1 should retain:

- 30 Hz reference clock;
- at most 4,096 events and 256 KiB serialized track;
- at most 16 active strokes;
- quantized Q16 positions/pressure and Q15 direction;
- explicit Begin/Move/End or equivalent phase;
- `Push` and `Curl` modes;
- deterministic boundary and finite-retention/hold law;
- canonical SHA-256 over the portable event stream.

The track must be recordable from native pointer/tablet, phone, MIDI, and OSC
only through one normalized event adapter. Export replays the exact track and
checksum. An unrecorded live gesture must never be implied replayable.

### Resource proposal

| Item | Bound |
|---|---:|
| Grid edge | 2,048 |
| Grid cells | 2,100,000 |
| Ping-pong vector + gate | 12 bytes/cell |
| Bytes/canvas | 32 MiB |
| Active canvases | 2 |
| Aggregate bytes | 64 MiB |
| Uniform stride | 256 bytes |
| Maximum decay ticks processed | 4,096 |

Both the two-canvas count and 64 MiB aggregate are hard preflight limits;
dimension and cell caps are checked independently. Canvas updates are
transactional: submitted/discarded frames, freezes, cuts,
manual clear, patch load, source replacement, resize, and export cancellation
must each have explicit state laws.

### Required acceptance

- canonical checksum and hostile serde bounds;
- identical grouped/ungrouped reference-tick replay;
- analytic push/curl strokes, overlapping stroke order, boundaries, decay,
  and hold;
- transaction commit/discard/freeze/reset tests;
- portable sidecar round trip and live/export field readback equality;
- no manual-history flood: a completed authored gesture is one entry, while
  automation-origin gestures are excluded.

## 4. Symmetry Field

### Authored law

Implement one dedicated node with exact default bypass and these closed modes:

- cyclic `Cn` and dihedral `Dn`;
- planar `p1`, `pm`, `p2`, and `pmm`;
- bounded log spiral and orbit.

Sector count is bounded at 32. A fixed sector table contains deterministic
source route, history age, hue offset, and motion-donor selection for every
sector. The table is stable under reorder and donor loss; missing entries
remain tombstones. Controls include fold offset, phase/axis, bounded cell
skew, spiral scale, orbit radius/spin, motion gain, hue span, and explicit
source/motion masks. Audio or gyro modulation affects only declared continuous
controls and never rewrites the route table.

All five boundaries—transparent, mirror, wrap, hold, and a single bounded
`CellularReentry` transform—must be fixed in the domain enum and CPU reference
before WGSL is written. Cellular re-entry is one deterministic D4-style cell
transform, never recursive sampling.

Freeze the domain at 32 sector records and history ages 0…23. Each record
chooses `Carrier`, `Donor0`, `Donor1`, or `CleanHistory`, an optional motion
donor 0/1, one history age, and one hue offset. Generate the complete table
with a stable counter/hash keyed by the stable node domain, authored seed,
sector index, and lane-domain constant. Runtime donor availability must never
enter that hash: losing a selected donor binds neutral/transparent without
rerolling any sector.

The exact default is cyclic fold 1, carrier-only, no motion/history/hue, and
neutral phase/axis/center. `effective_folds()` is the only rounding point:
round the already modulated `base_folds + fold_offset`, then clamp 1…32.
Freeze phase semantics before CPU or WGSL implementation:

- radial phase rotates the sector origin;
- orbit phase rotates sector classification;
- planar axis rotates the lattice basis;
- planar phase translates the primary lattice coordinate by one cell period.

Saved/runtime routing uses exactly two fixed image slots and two fixed motion
slots. Selected donors capture saved positions and resolve once to runtime
stable IDs; Missing donors retain their saved positions and never rebind.
Fixed array index is route identity, so planner tap consumers must include the
slot number rather than keying both donors only by node ID.

### Renderer/resource proposal

The dedicated renderer may own only neutral tiny resources plus a uniform
arena; no full-frame persistent surface:

| Item | Exact initial contract |
|---|---:|
| Sampled textures/pass | 8 |
| Bind groups | 2 — **shipped as 3**, see below |
| Render passes/node | 1 |
| Worst-case texture operations/pixel | 10 |
| Uniform arena/node | 1,024 bytes |
| Neutral textures | 3 tiny textures / 4 views |
| Full-frame persistent surfaces | 0 |

**Deviation as shipped: three bind groups, not two.** A `MotionGpuField` owns a
committed ping/pong parity of its own (`MotionMemoryStage::render_field_index`),
a third parity dimension above the carrier parity and the composition's N-1 tap
parity. Held in one input group the three multiply to 16 prebuilt groups per
node; splitting the motion vector/gate pair into its own group makes them add —
4 image groups (carrier × N-1) plus 4 motion groups (the two slots' committed
parities) = 8 per node, with three groups bound per pass. Honouring "2" would
have left an authored motion route binding the neutral pair and decoding to
exactly zero. Every other row of this table is unchanged, including the
eight-texture count: a fragment stage's sampled-texture budget is counted across
every bound group. See the Symmetry Field section of `CLAUDE.md`.

The frozen successor ABI should use two named image-donor slots, two named
motion-donor slots, and the already committed 24-layer clean-history array.
Each of 32 fixed sector records independently chooses carrier, either image
donor, clean-history age 0…23, hue offset, and either motion donor. Uniforms
are one exactly 1,024-byte dynamic-offset record per node. The identity and
wet-zero shader branches directly `textureLoad` the carrier; they must not
pass through bilinear conversion.

Use this exact 1,024-byte uniform layout (with a compile-time size assertion):

```rust
#[repr(C)]
pub struct SymmetryGpuUniforms {
    pub meta: [[u32; 4]; 4],
    pub params: [[f32; 4]; 4],
    pub motion_rows: [[f32; 4]; 8],
    pub sectors: [[u32; 4]; 32],
    pub padding: [[u32; 4]; 16],
}
```

Bind carrier, donor 0, donor 1, the clean-history D2 array, and vector/gate
pairs for both motion donors—eight sampled textures and no texture sampler.
Use custom premultiplied bilinear loads: dry carrier 4 + processed source 4 +
vector/gate 2 = at most ten texture operations. Reuse the existing committed
Compat8 clean-history ring; a new RGBA16F history ring is prohibited unless a
later product/resource decision explicitly admits its 398.1 MB (379.7 MiB)
1080p cost.

The evaluator must flush an ordinary rack segment, emit one dedicated
`SymmetryField` step at the authored position, and then resume segmentation.
Motion donors request their primitive vector/gate fields even when the donor's
own visible Motion effect is zero. Prepare both carrier-parity input bind
groups per node so warm execution allocates nothing; do not undercount this as
one bind group. Missing image or incomplete motion pairs bind the admitted
neutral views and expose stable diagnostics.

The successor session must first confirm that eight simultaneous sampled
textures is portable under the existing device floor. If not, split the node
into bounded passes and update the ledger before shipping.

### Required acceptance

- group-closure and identity-seam proofs for every mode;
- default GPU readback bit-identical to its carrier;
- analytic sector mapping and all boundary fixtures;
- stable sector-table seed, reorder, missing donor, and history-age bounds;
- patch/Morph/modulation/Dice/generator/preset/browser/native/export closure;
- live/export physical GPU equality and warm-allocation fingerprint.

## 5. Field Collider

### Authored law

Add a versioned collider block to Motion with two stable primitive field
inputs, transforms, confidence, and a closed mode enum:

- sum;
- difference;
- curl;
- projection;
- collision boundary.

Boundaries are transparent, hold, mirror, and wrap. `enabled = false` is exact
M4 behavior. Enabling the collider parks the existing single-donor transplant
recipe rather than ambiguously running both. Existing transplant amount,
carrier, confidence, refresh, decay, and occlusion remain the shared
carrier/advection controls. Aliased, missing, or invalid inputs are inert with
typed diagnostics.

Freeze version 1 as:

```rust
pub const FIELD_COLLIDER_ALGORITHM_VERSION: u16 = 1;

pub enum FieldColliderMode {
    Sum,
    Difference,
    Curl,
    Projection,
    CollisionBoundary,
}

pub enum MotionBoundaryMode { Transparent, Hold, Mirror, Wrap }

pub struct FieldColliderParams {
    pub algorithm_version: u16,
    pub enabled: bool,
    pub mode: FieldColliderMode,
    pub boundary: MotionBoundaryMode,
    pub input_a: MotionDonor,
    pub input_b: MotionDonor,
}
```

Version 1 adds no collider-only continuous controls. Existing Faraday
`amount`, carrier, confidence threshold/softness, refresh, decay, and
occlusion remain the one shared carrier/advection law. Consequently Dice and
modulation preserve the collider block exactly. Enabling Collider parks but
does not erase the existing single-donor transplant route; disabling resumes
the frozen M4 recipe.

For validated recipient-local vectors `a`, `b`, let `d = a - b`,
`m = (a + b) / 2`, and `eps = 1e-12`. The CPU and WGSL formulas are:

- Sum: `a + b`.
- Difference: `a - b`.
- Curl: `(-d.y, d.x)`.
- Projection: zero when `dot(b,b) <= eps`; otherwise
  `b * dot(a,b) / dot(b,b)`.
- Collision boundary: `m` when `dot(d,d) <= eps`; otherwise remove the mean
  flow normal to disagreement:
  `m - d * dot(m,d) / dot(d,d)`.

Clamp final components to the canonical Motion velocity range. Both inputs
must be selected, admitted, finite, in range, and transformed through finite
nonsingular matrices. Any failure returns the exact invalid/zero sample; it
may never reuse one input or a prior derived field. Derived confidence and
visibility are componentwise minima. The existing Faraday gate then applies
threshold/softness/occlusion exactly once.

For donor-local-to-composition affine `D` and
recipient-local-to-composition affine `R`, map coordinates with
`uD = inverse(D) * R * uR` and vectors with
`vR = linear(inverse(R) * D) * vD`; translation never affects a vector.
Transparent accepts inclusive `[0,1]`, Hold clamps, Wrap uses
`x - floor(x)`, and Mirror uses the period-two triangular map. Apply the same
boundary independently to each input's vector and gate lookup.

### Two-pass GPU law

Preserve the current maximum of three sampled textures per pass:

1. map donor A and B vectors through their field-space transforms into one
   transient RGBA16F pair texture (two sampled textures);
2. sample the pair plus A and B gates (three textures) and write one derived
   transactional RG16F vector plus RG8 gate field.

Collider-specific resources per grid cell:

| Resource | Bytes/cell |
|---|---:|
| Derived vector parity | 8 |
| Derived gate parity | 4 |
| Transient mapped pair | 8 |
| **Collider-specific total** | **20** |

Both primitive fields and the existing sole carrier remain separately and
honestly accounted. Initially admit one collider and one carrier only. Derived
slots append after primitive fields, own no decoder attachment or luma source,
and are never confused with codec/lattice acquisition slots.

The frozen two-pass uniform is exactly 144 bytes: two 64-byte
`MotionTransformGpu` records plus one 16-byte mode/status lane. Pass 1 binds A
and B vectors and writes `[a.xy, b.xy]` into the RGBA16F pair surface. Pass 2
binds the pair and both gates, validates sentinels/ranges, and writes the
transactional derived vector/gate parity. Prebuild the four A/B parity
combinations for each pass (eight bind groups total, one bound per pass). Add
two low-resolution passes and five nearest lookups per collider to the frame
budget while retaining the three-sampled-texture ceiling.

Planner state must keep primitive and derived provenance distinct:

```rust
pub struct EvaluatedFieldColliderPlan {
    pub output_slot: u8,
    pub recipient_scope: VisualScopeId,
    pub input_a_scope: VisualScopeId,
    pub input_a_slot: u8,
    pub input_b_scope: VisualScopeId,
    pub input_b_slot: u8,
    pub output_grid: MotionGrid,
    pub algorithm_version: u16,
    pub mode: FieldColliderMode,
    pub boundary: MotionBoundaryMode,
}
```

Both inputs request honest primitive fields even if their own Motion effect is
zero. Input A may equal the recipient and B may equal the recipient, but A and
B may not alias each other. Derived attachments are internal executor values:
never extend `CodecMotionProduct`, the live codec field cache, or export codec
acquisition to carry them.

Stage primitive fields first, derive Collider output second, and advect the
carrier third. Primitive, derived, carrier, and prior/current spatial state
commit or discard together. Program Freeze stages nothing; Media Freeze may
reuse committed primitive observations but must derive the new output
transactionally. Reset invalidates every derived parity and pending recipe
without reallocating.

Persistence stores strict version/mode/boundary and two saved donor identities
only. Selected routes resolve once to stable IDs; Missing tombstones never
rebind. Look applies the recipe while preserving live A/B topology; Morph
chooses the entire discrete block endpoint-exact. Browser topology uses an
ordered revision barrier such as
`SetMotionColliderInput { layer_id, input, donor_layer_id,
layer_stack_revision }`. Export metadata records both authored identities,
admission/output slot, diagnostics, budgets, and topology facts only after an
accepted frame—never vectors, pair textures, or raw codec records.

### Required acceptance

- analytic CPU fixtures for every mode/boundary/confidence law;
- donor-local → composition → recipient-local transform proofs;
- alias, missing, removed, replaced, malformed, NaN, and singular-transform
  diagnostics;
- exact 20-byte/cell ledger, one-byte-under rejection, one-collider cap, and
  three-texture-pass validation;
- transactional GPU commit/discard/freeze/reset and no stale derived field;
- patch/Morph/modulation/Dice/generator/history/browser/export closure;
- live/export physical readback and exact disabled M4 compatibility.

## 6. Native preview transform manipulation

Add a preview-only transform gizmo for translate, scale, rotate, anchor, and
crop. It operates on the same canonical `SpatialTransform` as browser/native
numeric editing; it must not introduce an editor-only transform.

- Hit testing uses the final preview mapping and is disabled when the main
  window is serving a clean audience output.
- One pointer drag emits one `NativeManual` history gesture with Begin/End;
  Escape cancels only before a value commit.
- Shift/Alt modifiers and keyboard nudges have fixed documented laws.
- Stable layer/group/master selection is captured at gesture Begin and cannot
  retarget after reorder.
- The gizmo paints through an editor-preview permit and never reaches
  Composite, Audience, Spout, Record, Export, or physical StageMap surfaces.

Acceptance requires coordinate round trips at multiple aspect ratios/DPI,
gesture-coalesced undo/redo, reorder during drag, single-monitor no-leakage,
and live/export pixel invariance for the resulting authored transform.

## 7. Evidence-gated precision and scale runtimes

These are separate capabilities, not implied by the existing evaluation
types:

### Content-addressed proxy worker

Implement an FFV1/Matroska worker only after defining bounded decode/audio
inputs. Use the existing path-independent cache key and preflight. The worker
must stage and fsync a same-directory create-new temporary artifact, validate
its decoded identity/settings, atomically replace only the matching cache key,
and recover from crashes without treating partial output as valid. Add LRU
eviction receipts and real decoder A/B telemetry; never call assessment alone
a proxy implementation.

### Hardware/zero-copy and external I/O

Hardware decode, zero-copy decode, Syphon input/output, NDI input/output, and
capture input each require typed capability evidence, platform-specific
resource accounting, lifecycle/reset tests, and the relevant SDK/license or
network authorization. Absence remains `Deferred`, not a silent software
fallback branded as the requested capability.

### Bounded mesh warp

Freeze vertex/index/mesh/endpoint caps, stable control-point identities,
degenerate-triangle rejection, GPU byte accounting, exact identity bypass,
preview editing, persistence, Morph/modulation laws, and live/export fixtures
before adding a mesh renderer.

### Experimental full-16 history

Keep this evaluated-only until the exact additional history surfaces fit the
accepted device budget and representative temporal workloads demonstrate a
documented gain. It must remain an explicit precision path and may not change
the settled Advanced RGBA16F-working/Compat8-history default.

### Study execution and distribution

The current data-only Study ABI grants no native, shader, filesystem, network,
process, device, or host-mutation authority. A future evaluator must execute
only the validated SSA instruction vocabulary against declared read-only
capabilities, with fixed instruction/register budgets and deterministic CPU
reference fixtures. Marketplace, binary plugin, signing, sandbox, update, and
license-distribution systems are separate governance/security projects and
must not be inferred from the data ABI.

## Cross-cutting completion matrix

For each creative element, the implementing session must check every row and
record the exact test name or an explicit not-applicable reason:

| Surface | Required proof |
|---|---|
| Domain | strict version, sanitization, exact default, hostile bounds |
| Persistence | patch round trip, selected/missing tombstone, no runtime pixels |
| Look/Morph | values-only identity preservation and endpoint-exact discrete laws |
| Modulation | continuous bounded fields only; stable addresses |
| Dice/generator | deterministic domain-separated streams; topology preserved |
| Planner | immutable stable routes, cycle rejection, exact resource preflight |
| GPU | premult filtering/math, transaction laws, warm allocation |
| Reset/freeze | patch, Look, cut, seek, resize, manual clear, both freezes, blackout |
| History/recovery | one manual transaction; automation excluded |
| Browser/native | strict actions, revision barriers, accessible diagnostics |
| Export | same evaluated plan, explicit initial state, bounded provenance |
| Compatibility | exact zero/default Legacy delegation and frozen pixels |

## Successor-session evidence and publication gate

1. Begin from the exact published repair commit, not this working directory.
2. Land one tranche at a time with its resource-delta table in the change.
3. Run focused CPU/serde/protocol tests, configured all-target check, full test
   suite, `cargo fmt --all -- --check`, strict all-feature Clippy, and both
   JavaScript syntax checks.
4. Run named physical-GPU fixtures on a recorded adapter/backend. Persist
   source and shader manifests only after all receipt-bound files freeze.
5. Publish the clean allowlisted tree, then require Linux, macOS, and Windows
   hosted CI at the exact commit SHA. A local run or a workflow definition is
   not cross-platform evidence.
6. Keep upstream-license/publication authority and external SDK/license facts
   explicit; successful code tests do not answer those governance questions.
