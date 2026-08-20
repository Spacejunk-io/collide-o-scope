# B4 — Display physics: evidence note

Everything the program renders is watched through something. B4 lands the
three mechanisms that were absent: real interlace fields, per-primary
phosphor persistence, and the mask/beam display models — a master stage on
the slot-0 seam between the temporal pass and the opaque resolve. Laws
derived from BENDR (MIT, © 2026 Steve Blythe), rewritten in linear light
with Rec.709 luma.

## What landed

- `src/display_physics.rs` — the portable law module and independent CPU
  reference (`gesture.rs` tradition): `DisplayPhysicsParams` (17 continuous
  + 3 discrete laws, each sub-block exact-off by default), the two closed
  vocabularies (`InterlaceMode` codes 0–2, `DisplayModel` codes 0–6,
  append-only), `field_parity`, the 3:2 film clock (`judder_held`, four
  film frames per five reference ticks), `field_resolve` (weave/bob/blend,
  twitter, judder), `phosphor_combine` (`max(cur, trail·k)`) with the
  multiplicative per-tick rate law, the mask families, the Lottes beam
  profile, HV sag, the fixed 12-tap gather ring, bloom/halation extraction,
  mono/green tints, and the 128-byte `DisplayPhysicsGpuUniforms`
  (compile-time asserted).
- `src/shaders/display_physics.wgsl` — three fragment stages over one
  uniform record, sampler-free (`textureLoad` plus one explicit-load
  covered bilinear for the sag-warped base read), following the CPU
  reference expression for expression.
- `src/renderer/display_physics.rs` — the single-seam executor. Live
  LegacyExact, live Advanced, selective-VHS, and export all converge on
  composite slot 0 immediately before `render_opaque_output`, so ONE
  implementation serves all four (the `encode_opaque_output` precedent,
  deliberately not the dual-implementation temporal one). The stage owns
  its own 30 Hz rational-accumulator reference clock (the
  `history_ticks_for_delta` law) because the Exact temporal state does not
  advance on Advanced frames; it is fed the program-advancing delta, so
  Pause holds and export replays structurally.

## The laws

- **Seat and coverage.** After Temporal, before opaque resolve, on every
  path (three live call sites in `main.rs`, two in `render_export.rs`). An
  ACTIVE stage flattens coverage — a screen has no transparency — so it
  observes covered light and outputs alpha one; the downstream resolve
  becomes the identity on it and the flatten still happens exactly once. A
  dormant stage encodes nothing and slot 0 reaches the resolve untouched.
- **N-1 phosphor.** The store decays and accumulates
  (`max(current, trail · k^Δticks)` over the pre-field signal, k clamped at
  0.995 per primary); the display reads the accumulator as of the previous
  frame — BENDR's exact persistence structure, with the decay
  exponentiated by fractional reference ticks so live at any fps and
  export at any fps share one trail length.
- **Fields.** Parity alternates per reference tick; the held field updates
  by one texture copy per tick, in the slot format (RGBA8 — the signal
  domain interlace lives in). The `il_order` fault swaps dominance.
- **Blackout clears the wake** inside `clear_composite` (now `&mut`):
  the phosphor parities clear and both validity gates drop. Disarming the
  stage also invalidates both memories, so a re-arm never resurrects a
  stale trail.
- **Lazy surfaces.** BENDR's own rule: the persistence pair exists only
  once the stage is armed. Default sessions charge nothing; the first
  armed frame allocates once; warmed armed frames allocate nothing.
  Retained bytes when armed: 4 B/px field + 16 B/px phosphor parity pair.
- **Frozen contracts untouched.** `temporal.wgsl`'s pinned SHA, both
  temporal uniform goldens, and the temporal pixel-sequence goldens did not
  move: the stage owns its own shader, pipelines, and uniform.

## Closure

Patch: `TemporalConfig.display`, skip-serialized at the exact-off default
(pre-B4 bytes and canonical hashes keep); hostile scalars sanitize to
neutral; unknown fields and unknown discrete tokens are deserialization
rejections. Wire: twenty `disp_*` params on the ordinary coalescible
`set_temporal`, mirrored in both validators plus the applier. Snapshot: an
additive `display` block on the temporal snapshot. Panel: the DISPLAY
PHYSICS group in the temporal section (17 sliders — static range count now
165 — two selects, one toggle, double-click resets at authored defaults).
Modulation: seventeen `display_*` continuous master addresses; the three
discrete laws have none; `morph` stays last. Morph: values blend, the
three discrete laws recall an endpoint at the midpoint. Live Dice
continues not to touch temporal-adjacent state; the generator mutates the
seventeen values in fresh field-isolated domains (`mutate_display_physics`)
— `GENERATOR_VERSION` bumped to "10" because a given seed's output now
carries the new values. Export rides the same encode on the same seam.

## Proof inventory

Hosted CPU (1,400 tests green at the final tree state):
- `display_physics::tests` — exact-off defaults with the wake law per
  sub-block (dressing wakes nothing), hostile neutral sanitize, parity and
  order-fault law, the 3:2 film clock, weave/bob/blend with
  amount-zero passthrough, twitter sign per field, judder hold, the
  phosphor max law with closed-form trails and the P22 ordering
  (green > red > blue), the fractional-tick rate law
  (k^0.5 squared = k), mask families per model with mask-dark zero
  transmitting everything, beam widening with brightness, sag/bloom
  extraction, mono/green tints, serde bounds.
- `patch` — the display block skip-serialized at default (pre-B4 bytes
  keep), whole round trip including all three vocabularies, hostile
  neutral sanitize, unknown fields and unknown tokens rejected.
- `morph` — continuous blend with midpoint recall of mode/order/model and
  exact endpoint recall.
- `modulation` — all seventeen addresses with their declared ranges, the
  three discrete laws with none, offsets applied to the temporal copy and
  clamped, discrete state untouched.
- `procedural` — deterministic replay, bounded mutation, discrete laws
  preserved, zero temperature byte-exact.

Opt-in (measured on AMD Radeon RX 6950 XT / Vulkan):
- `renderer::display_physics::tests::gpu_display_physics_follows_the_cpu_laws_and_blackout_clears_the_wake`
  — a dormant stage encodes nothing, allocates nothing, and leaves slot 0
  byte-identical; the weave comb interleaves two genuine moments (4 red +
  4 green rows from two ticks); the phosphor trail matches the closed form
  `k^(m-1)` per primary within sRGB quantization with the P22 ordering;
  and blackout clears the glowing wake.
- `render_display_physics_pipeline` — the labeled export case: the
  authored stage decodes differently from the `_flat` twin, and the
  `_repeat` render decodes identically (the stage clock and phosphor store
  are frame-indexed deterministic offline).

## Exactness A/B — measured

`render_study_field_pipeline` rendered from a pinned worktree at the base
merge `2ef10df` and again on this branch: decoded-frame **identical** by
`ffmpeg -f framemd5`. The default path did not move across the tranche.

## Deliberate boundaries

- The B4 spec's narrowed display-model surface is respected: BENDR's
  curvature/corner/vignette/output-transform/overlay/lens family is out of
  scope (B13 already owns barrel and vignette at the effect layer).
- The phosphor accumulator is a ping/pong parity pair charged at
  16 B/px total — the transactional-parity cost over the spec's
  single-surface estimate, accounted exactly.
- The stage clock is its own accumulator rather than the temporal state's
  counter: the Exact `TemporalState` does not advance on Advanced frames,
  and the stage runs on every path. Same law, same addresses.
- A held-audience restore bypasses the stage (it operates on the resolved
  slot 2); the trail resumes from its held state on release, which is the
  physically sensible reading of a hold.
