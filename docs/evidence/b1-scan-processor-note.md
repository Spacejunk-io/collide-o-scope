# B1 — The Scan Processor: evidence note

The enrichment plan's flagship tranche: a Rutt/Etra-style drawn raster as a
Collision Rack node, and the tree's **first non-fullscreen-triangle pass**.
The beam law is derived from BENDR (MIT, © 2026 Steve Blythe); the
`beam_position` composition order and the beam-energy law (`gain = 2/speed`)
are transcribed faithfully with attribution, and everything around them is a
rewrite (Rust / wgpu 29 / WGSL, linear light, alpha-covered Rec.709 luma).

## What landed

- `src/scan_processor.rs` — the portable law module and independent CPU
  reference (`gesture.rs` tradition: no wgpu/clock/filesystem/UI
  dependency): `ScanProcessorParams` (19 authored fields), `beam_source_uv`,
  `beam_position`, `beam_speed`, `beam_gain`, `ribbon_normal`,
  `ribbon_half_width_clip`, `scan_colorize`, the wake law
  (`is_exact_bypass`), and the named vertex budget
  (`MAX_SCAN_PROCESSOR_VERTICES` = 1,105,920, compile-time asserted equal to
  the authored maxima).
- `src/shaders/scan_processor.wgsl` — the instanced ribbon geometry pass
  (carrier fetched in the **vertex stage** through the explicit-load covered
  bilinear, sampler-free; two vertices per beam sample, one instance per
  scanline, additive ONE/ONE into a transient cleared to alpha one) plus the
  fullscreen resolve applying the engine-wide node law through the one
  canonical `blend.wgsl` kernel. The CPU reference is followed expression
  for expression.
- `src/renderer/scan_processor.rs` — the dedicated executor: two pipelines
  from one module, a 128-byte dynamic-offset uniform arena (compile-time
  asserted), and the one shared full-frame `Rgba16Float` accumulator
  allocated at prepare and reused by every scan pass in the frame.
- `NodeKindTag::ScanProcessor`, append-only signature code 14,
  `occupies_dedicated_pass = true`. The planner lifts it in `flush_segment`
  exactly as Study/Symmetry (kind-only), derives
  `ScanProcessorResourcePlan` from the emitted steps (2 passes, 2
  simultaneous textures, 128 uniform bytes, summed vertices, 8 B/px
  transient), refuses one vertex over the budget with the typed
  `ScanProcessorVertexBudget`, and hashes pass layout — deliberately never
  the vertex total — into the topology signature, so a lines/samples edit
  re-encodes without re-preparing.

## Closure

- **Patch**: params ride the node's ordinary tagged serde
  (`kind: scan_processor`); absent from every pre-B1 patch, unknown fields
  rejected, hostile scalars sanitize to the neutral default.
- **Wire**: all 19 params on the ordinary coalescible
  `set_visual_node_param` (15 floats, 2 unsigned geometry counts, 2 bools);
  ingress validation is fully descriptor-driven; no topology action exists
  because no route exists.
- **Snapshot**: the 19 values plus derived read-only `scan_exact_bypass`
  and `scan_vertex_count`.
- **Panel**: generated node card (15 sliders, 2 number inputs, 2 toggles);
  the literal range-tag counts (148 static / 17 JS) do not move because the
  rows render through the existing template.
- **Modulation**: 15 stable `scan_*` addresses (none angular); geometry and
  reversals unreachable by any route.
- **Morph**: the 15 values blend; lines/samples/reversals recall an
  endpoint at the midpoint; no route gate, so any scan pair interpolates.
- **Look/preset**: the whole params bundle transfers as values through the
  morph appliers.
- **Dice/generator**: the 15 continuous values mutate in each node's own
  stable domain; geometry and reversals preserved exactly; no
  `GENERATOR_VERSION` bump (no pre-existing anchor carries a scan node, so
  no existing seed's output changes — the version stays "9").
- **Export**: the shared plan and the same shader; time only from the
  frame-plan context, so Pause holds the detuned oscillator and export
  replays it structurally. `render_scan_processor_pipeline` is the labeled
  export case.

## The wake law

`is_exact_bypass()` is true when no *deflection* is authored (amount,
collapse, oscillator amount, S-curve, skew, both tilts zero; both reversals
off). Dressing controls (width, velocity mix, perspective, oscillator
freq/lock, Lissajous, mono, hue) shape a raster that exists only once a
deflection is authored. BENDR's own stage gate is the precedent; ours widens
it to include skew and the tilts, which genuinely author deflection alone. A
default node encodes nothing — proven byte-identical through both the
production GPU fixture and the labeled export case's `_bypass`/`_plain`
comparison.

## Proof inventory

Hosted CPU (ordinary `cargo test`):
- `scan_processor::tests` — the analytic law fixtures: flat-raster identity,
  luma pivot and span, read-side reversal, collapse, locked-oscillator
  standing pattern and quantization, detuned crawl, the 2/speed gain clamp,
  central-difference speed with its floor, fail-safe ribbon normal, the
  pixel width law, colorize (mono fold, black-stays-black, identity at
  zero), tilt-authors/perspective-does-not, hostile sanitize, serde bounds.
- `visual_rack` — registry now 14 kinds, code 14 append-only, dedicated
  list `[Symmetry, Study, ScanProcessor]`, the 19-row descriptor contract
  (modulatable ⇔ Float ⇔ Dice-eligible), default-bypass/wake/vertex-ledger,
  node serde round trip with unknown-field rejection.
- `modulation` — 15 stable addresses in declared order, offsets clamp (no
  wrap), geometry/reversals unreachable.
- `morph` — value blend with midpoint recall of the discrete class.
- `randomization` — Dice moves the 15 values only, repeats exactly,
  amount-zero no-op, neighbour byte-identical with and without the scan arm.
- `evaluated_composition` — the lift at the authored position with
  segmentation resuming behind it, the derived dedicated ledger (2 passes /
  2 textures / 128 B / vertices / 8 B-px transient), a plain rack charging
  the default, signature discriminating presence but invariant to geometry
  edits, and the kind-only bypass position.
- `web::state` — snapshot publishes every value plus the derived wake law.

Opt-in (`--ignored`; measured on AMD Radeon RX 6950 XT / Vulkan):
- `renderer::composition::tests::production_scan_processor_density_exceeds_any_displacement_and_default_is_bypass`
  — the pixels claim and the fixture that distinguishes the mechanism from
  the imitation: with `velocity_mix = 0` (beam-energy law disengaged), a
  collapsed raster's additive line overlap drives pixels above **twice** the
  flat source's maximum linear value, which no single-sample displacement of
  the same image can do; the authored default is byte-identical to no node;
  warm frames allocate nothing; the pass is deterministic.
- `render_scan_processor_pipeline` — the labeled export case: the authored
  deflection decodes differently from the `_bypass` twin (the node at its
  exact default on identical topology), and the `_repeat` render decodes
  identically. The bypass-equals-no-node byte identity is deliberately
  claimed inside one plan variant by the production GPU fixture, not by the
  export case: adding any node flips the plan from the frozen LegacyExact
  path to Advanced, and the two paths are equivalent, not byte-equal.

## Exactness A/B — measured

`render_study_field_pipeline` (a pre-existing labeled case) was rendered
from a pinned `git worktree` at the base commit `4a8747f` and again on this
branch, and the two outputs are **decoded-frame identical** by
`ffmpeg -f framemd5` (all 24 frame hashes equal; only the mux software tag
line excluded). The default path did not move across the tranche.

Measured on this host: AMD Radeon RX 6950 XT / Vulkan, ffmpeg 8.1.2,
rustc 1.97.1. The six-step gate passed in full at the final tree state
(fmt, both node checks, `cargo check --locked --all-targets`,
`cargo test --locked --all-targets -- --test-threads=1` — 1,383 passed —
and clippy `-D warnings`).

## Deliberate boundaries

- The vertex-budget refusal is defense in depth: sanitized ranges cannot
  exceed the cap today, so the typed error is structural insurance (the
  Residual grid-edge law), exercised through the ledger arithmetic rather
  than a reachable hostile path.
- The accumulator is one shared surface, not one per node; its 8 B/px charge
  does not scale with step count and is freed only with the prepared
  executor.
- Colorize is applied per beam sample in the vertex stage (BENDR colorizes
  the interpolated color per fragment); the CPU reference is per-sample, so
  vertex-stage application is what keeps reference and shader in exact
  agreement. The visual difference is sub-quantization at authored ribbon
  widths.
- BENDR's `scanGain` level control is deliberately not carried: the node's
  wet/blend law is the output-level authority.
