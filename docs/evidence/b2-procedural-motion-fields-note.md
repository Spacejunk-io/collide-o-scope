# B2 — Synthetic and shaped motion fields (procedural field sources)

Tranche B2 of the Bendr derivation plan, first slice: the six procedural field
kinds. The flow-shaping controls (`stretch`, `edge_repel`, `vector_trash`)
named in the plan's "Also here" paragraph are deliberately **not** in this
slice and remain open B2 work.

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

## Explicitly not claimed

- No cross-adapter portability claim beyond hosted three-platform CI.
- The flow-shaping controls (`stretch`, `edge_repel`, `vector_trash`) are not
  implemented.
- No new modulation source, no Symmetry-side change, no change to the M4
  velocity contract, byte ledger, or selective-VHS budgets.
