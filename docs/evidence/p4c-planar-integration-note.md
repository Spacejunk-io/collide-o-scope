# P4c Phase B — planar delivery integrated: evidence note

The candidate receipt
([`p4c-planar-gpu-candidate-receipt.json`](p4c-planar-gpu-candidate-receipt.json),
documented in [`p4c-planar-gpu-candidate-note.md`](p4c-planar-gpu-candidate-note.md))
measured the seams and cleared the reopen gate: 62.5% staging reduction,
delivery p95 down ~70%, upload p95 down 14–28%, CPU/GPU equality within one
code. This tranche is the integration that receipt authorized — the audit's
P4c items 10–14, landed as authored, opt-in, per-layer state whose default is
the exact prior byte path.

Branch: `feat/p4c-planar-integration` from the P4c candidate merge.

## The design, compressed

**One materialization seam, one added branch.** The decoder's single image
seam (`materialize_source_image`, formerly `scale_frame`) freezes the source
descriptor exactly as before on both branches; when the authored policy is
`metadata_managed` and the complete admission law passes — the shared
`prototype_delivery_decision` ladder over the frozen declared truth, plus the
frame actually being progressive 8-bit yuv420p at the decoder's geometry —
the frame's three planes are row-copied into one pooled allocation and
swscale never runs. Every other frame takes the exact legacy packed path.
A planar materialization fault falls back to packed for that frame with a
logged warning: planar delivery is an optimization, never a new way for a
decode to fail.

**The P4a machinery absorbed the format instead of growing a twin.**
`DecodedPixelFormat` gained `PlanarYuv420p8`; the planar payload is an
ordinary `DecodedImagePayload` from a second lazily created
`DecodedRasterPool` — same recycling, same `DecodedImageLedger` lease
(charged at the honest 1.5 bytes per pixel), same owner counters. The
reverse cache, the threaded mailbox, seek/loop/cue identity, and codec-motion
pairing therefore needed **zero structural change**: their laws never touch
pixel bytes, and cache byte accounting automatically charges actual planar
bytes (audit item 11). The one new invariant is fail-closed byte access:
`expect_packed_rgba8` turns a planar payload reaching a packed-only consumer
(seeds, still restores, compat accessors) into a typed refusal instead of
silently misread pixels.

**The recipe travels with the frame.** A planar payload carries its
`PlanarConversionRecipe`, derived once from the frozen descriptor through
the same `CpuConversionContract` the CPU oracle consumes. The upload seam
converts under exactly the truth the decoder admitted — nothing downstream
re-derives color from telemetry, so generation boundaries cannot race the
conversion parameters.

**One conversion application, shared verbatim.**
`layers::convert_planar_into_layer_texture` is called by the live upload
seam (inside the same three error scopes the packed `write_texture` upload
uses) and by the export upload wrapper, so live and offline cannot drift.
Video-layer textures gained `RENDER_ATTACHMENT`, `COPY_SRC`, and the
non-sRGB `Rgba8Unorm` twin view — free at rest — and the pass writes
source-encoded bytes through that twin, so a converted store is
byte-equivalent in kind to what `write_texture` stores. Per-layer conversion
state (pipeline, plane textures, bind group, target view) is lazy, built on
the first admitted planar frame, and dropped wherever the texture identity
changes (slot switches, proxy adoption, resize) with the authored policy
re-applied to the incoming decoder.

**Policy is a source contract.** Per-layer `delivery` rides `LayerConfig`
(skip-serialized at the legacy default, unknown tokens rejected), the
coalescible `set_layer_param { param: "delivery" }` with a closed token
vocabulary validated at the gate and the applier, an additive snapshot pair
(`delivery` authored ask + `delivery_active_planar` truthful fact), and a
video-only panel select beside Blend. It is deliberately outside Morph
ownership, Apply Look, Dice, the generator, modulation, and the B9 recorder
— the same exclusion class as source identity. Live edits reach the worker
through a shared atomic cell read per materialization: policy is a law, not
a selection, so no command ordering question exists. Prep/seed decoders
(performance preparer, proxy adoption) keep the legacy default structurally,
so seeds stay packed.

**Export obeys the same authored law.** The offline decoder receives the
config's policy before its seed decode, and both export upload sites route
through the shared conversion. `.motion.json` provenance already carries the
source color descriptor and conversion policy per layer.

## What is proven

Gate on the final tree: fmt, both node checks, and the exact CI-form
check / test / clippy — main binary **1,985 passed / 0 failed / 151
ignored**, every pre-existing golden, hash pin, and panel contract intact
with the integration in place. The default is byte-exact by construction
(the packed branch is the verbatim prior body) and by suite.

Hosted additions: the pooled planar materializer (strided planes packed
tight, recipe attached, honest ledger bytes, recycling, typed refusals in
both pool directions), the patch round trip (absent at default, persists
managed, unknown token rejected), the ingress gate battery for the closed
token vocabulary, and recipe/descriptor uniform-derivation agreement.

Opt-in, run on the receipt adapter (AMD RX 6950 XT / Vulkan):

- `gpu_planar_conversion_matches_the_cpu_reference_battery` — unchanged and
  green over the promoted executor.
- `gpu_planar_integration_layer_upload_matches_the_cpu_oracle_and_legacy_stays_exact`
  — the live-seam proof: a real `Layer` over a real threaded decoder; a
  packed upload stores exactly its payload bytes; flipping to
  `metadata_managed` live delivers a planar frame whose upload converts in
  place within one code of the CPU oracle and publishes the truthful active
  fact; flipping back restores the exact packed store.
- `render_planar_delivery_pipeline` — the labeled export case: the managed
  render must decode differently from its `_legacy` twin (the difference
  *is* the admission proof — a silent fallback would collapse them), and
  the `_repeat` render decodes identically, so the conversion is
  deterministic through the complete offline pipeline.

## Boundaries, stated plainly

- Production admission is deliberately `Yuv420p8` only — the audit's common
  8-bit 4:2:0 path. NV12/P010 remain converter-supported but unadmitted;
  10-bit sources keep the exact swscale path with no fidelity change, and
  the HDR/tone-map question stays its own evidence-gated tranche (item 13).
- Managed and legacy produce *different* pixels by design: swscale's
  converter law versus the declared-truth oracle law. That difference is
  authored, persisted with the patch, and identical live and offline.
- The legacy default remains the program's law. The audit's integrated
  total-frame p99 matrix remains the standing gate for any future
  default-flip or auto-selection; nothing here claims it. What is claimed
  is measured at the seams (the Phase A receipt) and proven at the seams
  (the fixtures above): an operator who authors `metadata_managed` buys
  ~70% cheaper delivery and ~62.5% smaller uploads for admitted sources.
- The Phase A candidate receipt is retained unmodified as the measurement
  that authorized this integration; its `measured_at` names the candidate
  tree, which is exactly what it measured.
