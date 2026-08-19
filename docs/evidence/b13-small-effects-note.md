# B13 — The small-effects tranche

Tranche B13 of the enrichment plan: fifteen looks land as one tranche under
one law. Twelve effect families at layer and master scope — contour isolines
(+ flatten + ordered dither), solarize, negative with three inversion modes,
find-edge, emboss, colourpass, halftone, moiré, bitcrush, row smear, and the
dumb multi grid — plus three master-only optics: barrel, chromatic
aberration, and the anamorphic streak. Every law is derived from BENDR (MIT,
© 2026 Steve Blythe) and rewritten in linear light with Rec.709 luma.

## Design decisions worth recording

- **One shared uniform extension.** `EffectUniforms` grew from ten to
  eighteen vec4s (160 → 288 bytes); `EffectPassUniforms` is now 352 bytes
  with the four spatial slots at byte 288. No renderer buffer size is
  hand-written, so every consumer followed through `size_of`.
- **Every default is an exact no-op branch.** Each effect gates on its own
  authored amount (`> 0.0001`, multi grid on `>= 1.5`), pinned by a source
  audit, so a default patch never enters a new branch. Byte exactness is
  proven by the re-pinned M6 shader-bundle digest (`3e280fd7…`) with all six
  output SHAs unchanged, and by a cross-build framemd5 A/B of
  `audit_feedback_rig.mp4` against a pinned pre-B13 worktree build.
- **One sampling path.** Every neighbour read (contour smoothing 4,
  find-edge 4, emboss 2, halftone 1, chroma aberration 2, anamorphic 20)
  goes through the canonical `sample_source` chain, so an active spatial
  transform keeps owning every exposed coordinate; multi grid, barrel, and
  row smear keep the historical clamp/wrap only on the inactive legacy
  branch, the Shift/Cellular precedent.
- **The optics are master-only at every seam, not by convention.**
  `EffectUniforms::clear_master_only_effects` runs after the layer wire
  applier, layer patch application, Look application, layer Dice, and the
  three offline layer builders; no `layerN_` modulation address exists for
  the three; the generator's layer form never mutates them. A hostile patch
  or a legacy protocol client cannot install an optic on a layer copy.
- **Dice byte-stability is measured, not argued.** The small effects mutate
  in a domain-separated stream (`0x534d_4c46_5800_0001`), and a golden
  measured on the pre-B13 build (a7c700c) pins eight established Dice values
  for seed 1234 / stream 9 — the hosted suite now proves those bytes did not
  move. `GENERATOR_VERSION` bumped to "9": a generated piece now carries the
  new values, mutated per scope in the fresh `PROCEDURAL_SMALL_FX_DOMAIN`.
- **`negative_mode` is the tranche's one discrete law** (permanent codes:
  0 rgb, 1 luma-only, 2 hue-flip): no modulation address, midpoint recall in
  Morph, never reroll-mutated — the grain_algo/B3 precedent. The four
  angle/hue controls blend on wrapped degree arcs in Morph.

## Measurements on this host

Adapter: AMD Radeon RX 6950 XT / Vulkan (driver 26.7.1), Windows 11.

- The M6 precision receipt re-ran under the re-pinned shader-bundle digest
  and **all six output SHAs are unchanged** — the default-off branches are
  byte-exact through the real still and temporal workloads. The still
  fixture's input-identity lane legitimately moved (it hashes the raw
  `EffectPassUniforms` bytes, which grew 224 → 352) and was re-pinned; the
  tracked receipt diff is otherwise confined to the bundle/source digests,
  timestamps, and wall-clock lanes.
- Cross-build exactness A/B: `renders/audit_feedback_rig.mp4` and its
  `_unrigged` twin rendered on the pre-B13 build (a7c700c worktree) and on
  this build are decoded-frame identical (framemd5) through the real export
  path.
- `render_small_effects_pipeline` (opt-in, GPU + ffmpeg + `videos/audit.mp4`):
  a master program authoring eleven of the twelve shared families plus all
  three optics over a layer authoring find-edge and emboss renders; its
  `_plain` twin decodes differently (the tranche reaches the pixels); its
  `_repeat` twin decodes identically (the moiré clock and every dither
  replay deterministically).

## Hosted proof

`cargo test --locked --all-targets -- --test-threads=1` green (1,358 tests)
under the full six-step gate, including the layout/order/gate audits, the
master-only law at every seam, the patch/snapshot/Morph/modulation/Dice/
generator closure fixtures, the pinned pre-B13 Dice golden, and the bumped
148-slider range contract.

## Explicitly not claimed

- No cross-adapter portability claim beyond hosted three-platform CI.
- No per-effect GPU golden exists; the pixel claims are the unchanged M6
  output SHAs, the cross-build A/B, and the labeled export case.
- The halftone dot colour reads the pre-adjustment source (the derived
  stage order); BENDR's separate-pass ordering differs in that its colour
  stage runs after its glitch pass — ours runs halftone before the colour
  adjustments so the dots receive them, which is the same intent in one
  pass.
- The anamorphic streak's spread and coating tint are fixed law (B2
  remainder-decision precedent), not authored controls.
