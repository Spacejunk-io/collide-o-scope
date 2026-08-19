# B2 — Synthetic and shaped motion fields

Tranche B2 of the enrichment plan, in two slices: the six procedural
field kinds (slice 1) and the flow-shaping controls (slice 2 — `stretch`,
`edge_repel`, `vector_trash` + `trash_block_size`, the plan's "Also here"
paragraph). With slice 2 landed, B2 is complete.

## What landed

- `MotionFieldSource::Procedural(ProceduralFieldKind)` — `Curl | Radial |
  Spiral | Contour | Chroma | Weave`, permanent codes 0–5, wire tokens from the
  single `source_key` table.
- The CPU reference `motion::procedural_field_sample` and its GPU twin
  `src/shaders/motion_procedural.wgsl` (one dual-target pass writing the
  existing vector/gate parity surfaces; the two image-reading kinds bind the
  scope's image alpha-covered, the four pure kinds bind a 1×1 defined-zero
  neutral).
- `ProceduralFieldParams { scale, rate }` with modulation addresses
  `motion_field_scale` / `motion_field_rate` at master and layer scope
  (`LAYER_TARGET_SUFFIXES` 58, 59).
- Full closure: skip-at-default patch block (pre-B2 bytes and hashes
  unchanged), Morph scalar blend + midpoint kind switch, Dice/generator
  mutation in fresh domains (`GENERATOR_VERSION` "7" → "8"), snapshot
  `procedural` block, panel selects/ranges, sidecar provenance through
  `requested_source` (no schema bump), export through the shared plan with
  codec acquisition skipping procedural origins.
- Topology-signature law: `MotionFieldOrigin::signature_code` keeps 0–3 and
  assigns kinds 4–9, so a kind change re-prepares bind groups.
- Publication law: the codec-upload parity flip on every program-advancing
  frame, valid from the first; synthesis advances under Media Freeze and layer
  pause (it is derived from program time, not acquired from media) and holds
  under Program Freeze. Time is `FramePlanContext::time_seconds` only.

## Measurements on this host

Adapter: AMD Radeon RX 6950 XT / Vulkan (driver 26.7.1), Windows 11.

- `gpu_procedural_field_matches_the_cpu_reference_for_every_kind`
  (opt-in `--ignored`): worst |GPU − CPU| velocity disagreement per kind, on a
  64×48 source at High quality (16×12 grid), scale 0.25, rate 0.8, t = 1.7 s:

  | kind | worst |GPU − CPU| (UV/s) |
  |---|---:|
  | curl | 0.0077 |
  | radial | 0.0039 |
  | spiral | 0.0039 |
  | contour | 0.00032 |
  | chroma | 0.0036 |
  | weave | 0.0037 |

  All within the 0.08 assertion bound; the dominant term is Rg16Float storage
  quantization (ulp ≈ 0.0156 at magnitude 16).

- `render_procedural_motion_field_pipeline` (opt-in, GPU + ffmpeg +
  `videos/audit.mp4`): rendered `renders/audit_procedural_motion_field.mp4`
  through the real export path. The `.motion.json` sidecar records
  `requested_source: procedural_curl`, `source_origin: procedural_curl`,
  `rendered_source_origin: procedural_curl`, `field_planned: true`,
  `field_attached: true`, `transplant_admitted: true` — the synthesized field
  was planned, rendered, attached, and advected the carrier.

## Hosted proof

`cargo test --locked --all-targets -- --test-threads=1`: 1,326 tests passing
after the tranche, including the new analytic per-kind fixtures, the
pure-kinds-never-observe-the-image law, alpha-covered Chroma neutrality,
zero-luma-bytes preflight, the procedural-versus-codec Collider planner
fixture, patch/Morph/modulation/Dice/wire closure tests, and the ingress
impact classification.

## Slice 2 — flow shaping

- `FlowShapingParams` on `MotionParams`: `stretch`, `edge_repel`,
  `vector_trash` (unit amounts), `trash_block_size` (2–256 px). The law is
  `motion::shape_flow_velocity`, mirrored in `motion_apply.wgsl`, ordered
  stretch → repel → trash → canonical clamp, operating on the gated sampled
  velocity so shaping never manufactures motion without a valid applied field.
- Trash fires per cell per 8 Hz event tick with probability `vector_trash`,
  under the shared `cellular_avalanche` hash in the fixed "MTRS" domain — no
  authored seed, so replay is structural.
- Apply uniform grew 1,664 → 1,680 bytes (`shaping_values` lane) with the
  compile-time assertion updated; `motion_pass_budget` charges exactly four
  covered-luma taps per fragment while `edge_repel` is nonzero and nothing
  otherwise.
- Full closure: skip-at-default `shaping` patch block, Morph blend of all four,
  Dice/generator v8 fresh domains, modulation addresses at both scopes
  (layer suffixes 60–63), ValuesOnly ingress, wire vocabulary, additive
  snapshot block, panel ranges.

Measurements on this host (same adapter):

- `render_motion_flow_shaping_pipeline` renders a shaped file and an
  `_unshaped` twin and asserts their decoded framemd5 sequences differ —
  shaping reaches the pixels through the real export path. Passed.
- Cross-build exactness: `renders/audit_procedural_motion_field.mp4` rendered
  before and after the shaping change is **decoded-frame identical**
  (framemd5), so the all-zero shaping path did not move a pixel.
- The env-gated `gpu_motion_formats_pipelines_and_codec_upload_are_valid_when_opted_in`
  validates the modified `motion_apply.wgsl` through real pipeline creation.
- Hosted suite after slice 2: 1,332 tests passing.

## Explicitly not claimed

- No cross-adapter portability claim beyond hosted three-platform CI.
- No new modulation source, no Symmetry-side change, no change to the M4
  velocity contract, byte ledger, or selective-VHS budgets.
- The trash event clock (8 Hz) and shove amplitude (16 UV/s) are fixed law,
  not authored controls; a future authored rate would need its own address,
  Morph law, and Dice domain.
