# B6 — the block-domain corruption trio

Three distinct digital-corruption mechanisms, all previously absent, as three
Collision Rack node kinds lifted into one dedicated executor: **Block DCT**
(kind code 15), **Pixel Sort** (16), and **Filter Avalanche** (17). The laws
are derived from BENDR (MIT, © 2026 Steve Blythe) — all three exist in its
shipped chain (`dct`, the `glitch` stage's pixel sort, the `lab` stage's
reconstruction-filter avalanche) and were transcribed verbatim, then
hardened. Every law runs in the **encoded sRGB domain on straight-alpha
values** — the B8 code-byte / B5 real-codec precedent: these are storage
artefacts, so quantising or wrapping linear light would manufacture
different ones. `src/block_dct.rs`, `src/pixel_sort.rs`, and
`src/filter_avalanche.rs` are the independent CPU references in the
`gesture.rs` tradition.

## Why all three are dedicated passes

- **Block DCT** is a four-full-frame-pass pipeline through two float
  coefficient intermediates. BENDR evaluates all N coefficients per fragment
  (O(N²) taps); at full output resolution that is hostile, so each axis
  splits into a coefficient pass (each texel holds its own block-relative
  coefficient, N taps) and a reconstruction pass (N coefficient taps) —
  O(N) per pixel with byte-identical mathematics, the same sums
  reassociated. No ordinary segment can express a multi-pass pipeline.
- **Pixel Sort**'s faithful 32-tap run search plus the carrier and the
  run-end fetch is 34 honest lookups per pixel — more than the frozen
  32-lookup ordinary-rack budget admits *for a whole rack*. Dedication
  charges the taps honestly instead of coarsening the transcribed law.
- **Filter Avalanche** reads its own previous output — a retained per-node
  history the ordinary rack has no vocabulary for at all.

The ledger split this forced is principled and additive: dedicated passes
never share a pass with rack nodes, so their per-pixel terms now leave the
frozen per-rack ceilings (which keep their exact meaning for the ordinary
pass chain) and are governed by the new
`MAX_LOGICAL_TEXTURE_LOOKUPS_PER_DEDICATED_PASS` (= 65, exactly the widest
pass in the tree — the avalanche's carrier + 32 gradient taps × 2 — so any
wider future kind is a deliberate raise). Dedicated lookups still sum into
the frame-level ceilings, which stay honest. This mirrors the established
texture-ceiling split (`MAX_SAMPLED_TEXTURES_PER_DEDICATED_PASS`).

## The laws

- **Block DCT** (`dct_amount/quantize/hf_penalty/chroma_crush/block`, all
  five continuous and modulatable — BENDR's own surface, block size 4–16 via
  `floor(4 + block·12)`): DCT-II with orthonormal scaling on analysis and
  synthesis, quantiser step `(0.004 + q·0.5)·(1 + u·tilt·2)` with
  round-to-nearest, chroma crushed in the **coefficient domain** against
  Rec.601 luma before the round (`keep = 1 − chroma·0.85`). Alpha rides the
  chain untransformed. Wake law: amount alone; the quantiser's 0.004 step
  floor means "quantise zero" is never claimed as identity — bypass is the
  amount gate.
- **Pixel Sort** (`sort_amount/threshold`): a pixel whose Rec.601 encoded
  luma exceeds the threshold searches upward through ≤32 taps stepping two
  rows (64-row reach) for its run's end and takes the run-end colour mixed
  by the amount — every pixel in a bright run inherits the end colour,
  which is the streak. Taps clamp at the frame edge, never wrap (BENDR's
  own comment: wrapping stretched a false streak from the seam).
- **Filter Avalanche** (`avalanche_amount/run` continuous;
  `avalanche_axis ∈ sub|up|average`, codes 0–2, discrete, no modulatable
  address): per-lane corruption gate firing at `amount·0.5`, bounded
  gradient accumulation (`span = 2 + run·40`, ≤32 taps, out-of-frame taps
  masked but non-terminating), per-lane epoch-invariant DC seed, and the
  `fract` wrap that makes the hard hue flips. Two house hardenings BENDR
  never claimed:
  - **Determinism.** BENDR re-rolls corrupt lanes on wall-clock time at
    3 Hz. Here the lane epoch is `floor(frame-plan seconds × 3)` through
    the shared integer-avalanche hash (`mixing_boundary::lane_unit`, fresh
    "AVL" domains), keyed by the node's **stable authored id** (persisted
    topology, identical live and offline — never a process-lifetime layer
    id), so Pause holds the fault stream and export replays it.
  - **The cascade.** BENDR accumulates gradients of the current frame, so
    its corruption never travels. Here the accumulation reads the node's
    own previous output — one retained working-format surface per node,
    advanced at most once per 30 Hz reference tick on the frame-plan clock
    (the melt rate law: live and export cascade at the same speed) — so an
    error written last tick is re-inherited this tick and the avalanche
    becomes visible motion. Before the first committed history the
    accumulation reads the carrier itself: a cold node degrades to exactly
    BENDR's shipped single-frame law, never to nothing.

## Machinery

One executor (`renderer/corruption.rs`, `corruption.wgsl` composed with the
canonical `blend.wgsl`): four pipelines, one 80-byte dynamic-offset uniform
arena (compile-time asserted), sampler-free (every lookup a `textureLoad`),
two bound textures per pass. The DCT's two full-frame `Rgba16Float`
intermediates are shared by every DCT step in the frame (sequential reuse,
the Scan-accumulator law, charged once at 16 B/px). The avalanche history is
the bus-melt shape verbatim: lazily allocated per node on the first armed
frame before the warm-allocation snapshot, retained thereafter, invalidated
(never freed) on disarm, staged/committed through the frame-history
transaction so a discarded frame leaves no stale validity; blackout never
clears it (program memory, the melt precedent); a re-prepare (patch apply,
topology change) rebuilds the executor and the cascade starts cold.
`MAX_AVALANCHE_NODES = 4` per composition (each history is 8 B/px), a typed
`AvalancheHistoryBudget` refusal at five. The combined
`CorruptionResourcePlan` is re-derived from the emitted steps (the
Study/Scan asymmetry) and exposed as `corruption_resources()`.

The planner lift is kind-only, the Scan shape: flush before, one
`EvaluatedScopeStep::CorruptionField` per node at its authored position,
segmentation resuming behind it; a default node still owns its step and its
uniform slots, so numbering never depends on frame-local values, and
`is_active` (enabled ∧ wet > 0 ∧ !exact bypass) gates every encode.

## Closure

Patch: the three kinds ride `VisualNodeKind`'s ordinary tagged serde
(`kind: block_dct | pixel_sort | avalanche`), absent from every pre-B6 patch
so old bytes and canonical hashes keep; unknown fields and tokens are
rejections; hostile scalars sanitize to neutral. Wire: all values ride the
ordinary coalescible `set_visual_node_param` (keys prefixed — `dct_*`,
`sort_*`, `avalanche_*` — because bare names cross-resolve under
`same_wire_parameter`); `avalanche_axis` is a closed token on the server
gate's enum allowlist; no route exists, so no node has a topology action.
Modulation: the nine continuous values through `apply_stable_node_offset`;
the axis has no address. Morph: any same-kind pair interpolates (route-free);
the axis recalls an endpoint at the midpoint. Look/preset: whole-bundle
values transfer in all three appliers. Dice and generator v12 mutate the
nine continuous values in each node's own stable domain and never the axis —
no `GENERATOR_VERSION` bump (new kind arms, the B1/B7 precedent; every
pre-existing stream proven byte-stable by the full suite's pinned goldens).
Panel: three generated node cards plus add-node options (generated rows —
the pinned range counts 198/24 did not move). Export consumes the same plan
and the same shader on the shared executor; there is no export-only path.

## Proof

- Law tests (hosted): DCT forward∘inverse identity at N ∈ {4, 8, 16} with
  the quantiser disengaged, the floored/tilted/monotonic step law, the
  15%-retained chroma crush with grey invariance, flat-block DC stability
  and coarse-quantiser ringing, the BENDR block-edge map; the sort's
  below-threshold/zero-amount identity, run-end colour inheritance, the
  exact 64-row reach with edge clamping, hostile short input; the
  avalanche's frozen axis codes, span/epoch laws (`floor(t·3)`, hostile
  time to epoch zero), deterministic per-key gates with the honest
  `amount/2` firing fraction, flat-history DC-only movement, gradient
  accumulation across an edge, and the warm-history cascade differing from
  the cold carrier read with hostile short history degrading to cold.
- Closure tests (hosted): the 17-kind registry with append-only codes
  15/16/17, the six-kind dedicated list, the per-kind param surface
  (modulatable ⇔ Float ⇔ Dice-eligible; the axis refused), tagged-serde
  round trips with unknown-field/token rejection and neutral sanitize, the
  planner lift with per-node steps at authored positions, the re-derived
  combined ledger (6 passes, 480 uniform bytes, 16 B/px DCT transient
  charged once, 8 B/px avalanche history, the honest 17+34+65 tap sum), a
  bypassed node still owning its inactive step, and the four-admitted /
  five-refused avalanche cap.
- `gpu_corruption_trio_matches_the_cpu_references` (opt-in, RX 6950 XT /
  Vulkan): all three GPU laws against their CPU references under the B7
  statistical contract (≥95% of channel samples within 4 code values — the
  DCT's f16 intermediates and per-adapter transcendentals move isolated
  samples; a wrong law moves most and fails flat), the avalanche proven
  both cold (history = carrier, BENDR's law) and warm (a distinct history
  demonstrably participating in the cascade). It also caught the tranche's
  WGSL reserved word (`meta`) exactly as trap 3 predicts — the hosted
  suite never parses new WGSL.
- Labeled export cases (opt-in): `render_block_dct_pipeline`,
  `render_pixel_sort_pipeline`, `render_filter_avalanche_pipeline`, each
  with a `_clean` exact-bypass twin that must decode differently and a
  `_repeat` that must decode identically — the avalanche repeat covers the
  whole deterministic chain (lane epochs, node-id-seeded hashes, the
  tick-clocked history).
- Exactness A/B: `render_feedback_rig_pipeline` on a pinned `f69e9e8`
  worktree versus this branch, `framemd5` identical over all 33 frames
  (`#software` excluded) — the default path moved no pixels. The full gate
  ran green at the final tree state (fmt, both JS checks,
  `cargo check --locked --all-targets`, 1,560 tests / 0 failed under
  `--test-threads=1`, clippy `-D warnings`, rustc 1.98.0), with the opt-in
  GPU fixture and all three labeled export cases re-run on the RX 6950 XT.

## Deviations, named

- BENDR's DCT is one fused O(N²) pass per axis at reduced processing
  resolution; ours is the O(N) coefficient/reconstruction split at full
  resolution — the same sums reassociated, proven against the CPU
  transcription.
- BENDR's avalanche has no feedback and a wall-clock lane roll; the spec's
  own design (retained history, determinism) overrides, with the history
  advance moved from "per frame" to per reference tick because live/export
  parity outranks the sketch's loose wording.
- BENDR's pixel sort searches +y in GL coordinates (toward the top of the
  picture); ours searches −y in image coordinates, which is the same
  direction on screen.
