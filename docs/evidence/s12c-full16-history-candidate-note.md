# S12c — the Full-16 history candidate: evidence note

Gate 6, opened by the operator's commission. S8 stated the gate precisely:
the byte-exact budget was done, and what was missing was "representative
temporal workloads demonstrate a documented gain" — a measurement that
required building at least an experimental render path, which was itself a
product decision. The commission made that decision; this tranche builds
the path, runs the measurement, and documents the result. The candidate
remains **evaluation-only** — whether the measured trade is worth paying
live is a further product decision this measurement now informs.

Branch point: `124f7cf` on `feat/proxy-authored-settings` (stacked on the
two Gate 1 tranches; mainline `69204b7` suite-green). Baseline:
**1308 passed / 0 failed / 96 ignored**; with this tranche
**1309 / 0 / 97** — one hosted plan-discrimination test and one opt-in
GPU measurement fixture.

## The design, compressed

**One constructor parameter, one pipeline target, one byte width.**
`CompositionHost::new_with_history_storage` selects the storage of exactly
the 25-layer temporal class — the 24-layer clean-history ring and the
recursive feedback image. Three things change under `Rgba16Float`: the two
texture formats, the no-dither conversion pipeline's color target, and the
plan validator's byte width for the class. Nothing else: no consumer
shader, no bind-group layout, no working or present surface, no dither.

**Value domain is preserved by construction.** The class is written only
by render passes and read only by texture loads, so both storages present
identical *linear* values to every consumer — sRGB8 encodes on write and
decodes on read, f16 carries linear directly. The candidate changes
quantization, never meaning, which is what lets the temporal shaders, the
History Key, the Symmetry Field's `CleanHistory` lane, and the Study
interpreter's `LoadHistoryColor` all read either ring unmodified.

**The settled default did not move.** The production `CompositionHost::new`
delegates with `Compat8`; the candidate is constructed by the receipt
fixture alone — no wire action, no patch field, no env toggle, no
production call site. The M6 receipt fixture pins exact output SHA-256s
for the settled still and temporal workloads, so settled byte identity
across this tranche is asserted, not assumed; the 31-case decoded-
`framemd5` A/B (below) proves it end to end through the real export path.

**The ledger charges the truth.** The candidate's plan declares the
temporal class at 8 bytes per pixel and `validate_host_resource_plan`
computes the class width from the requested storage, discriminating in
both directions with one-byte-under rejection. The exact candidate delta
is the documented figure: 25 surfaces × 4 additional bytes per pixel —
+207,360,000 bytes (197.753906 MiB) at 1080p.

## The measurement

Tracked receipt: `docs/evidence/full16-history-candidate-receipt.json`,
regenerated in place by the opt-in fixture (S2-receipt law: a changed
receipt after an opt-in run is a new measurement — commit it). Adapter:
AMD Radeon RX 6950 XT / Vulkan, the production device request
(`Features::empty()`, `Limits::default()`), 192×108, analytic f32
references no candidate output ever touches.

| Lane | Settled RMSE | Candidate RMSE | Notes |
|---|---:|---:|---|
| Clean-history storage fidelity | 0.0005051289 | 0.0000147806 | ~34× lower error; settled loses 1,194 / 11,506 reference gradients to 8-bit quantization, candidate retains all |
| Feedback recursion (12 frames) | 0.0000903414 | 0.0000216753 | ~4× lower accumulated error; max error 0.0030 → 0.0004 |

`ArtisticGainAssessment` verdict on both lanes: **"resource or metric
tradeoff"** — the vocabulary's exact phrase for a measured objective gain
at a nonzero resource cost, never a subjective declaration. The gate's
"documented gain" evidence now exists; the trade's price is documented
beside it.

| Surface | Required proof | Status |
|---|---|---|
| Plan discrimination | 8-byte class accepted/refused both directions, one byte under refused, exact delta | **Covered, hosted.** `full16_history_plan_charges_eight_bytes_per_temporal_pixel_and_discriminates`. |
| Measurement | representative temporal workloads, analytic references, documented verdict | **Covered, opt-in GPU, run on this host.** `gpu_full16_history_candidate_measures_temporal_gain_and_writes_the_receipt` — the fixture also asserts ring-lane improvement and feedback-lane non-regression, so a future adapter where the claim fails is a loud failure, not a silently regenerated receipt. |
| Settled byte identity | default path unmoved | **Covered twice.** The M6 receipt fixture's pinned output SHAs pass unmodified (opt-in, this host), and the 31-case decoded-`framemd5` A/B is byte-identical across the tranche — baseline captured from a pristine pinned worktree at `124f7cf`, fix side from the final tree, both directions of the claim. |
| Consumer closure | ring readers unchanged | **By construction, argued.** No consumer shader or bind-group layout changed; the value-domain argument above is stated in the constructor's contract and the design record. |
| Authored closure | no wire/patch/Morph/Dice surface | **By absence.** The candidate has no authored state anywhere; there is nothing for patches, Morph, modulation, Dice, the generator, or the browser to close over. |

Gate on `RUSTUP_TOOLCHAIN=1.97.1`: fmt, both node checks, check, tests,
clippy `-D warnings` — run on the final tree before commit.

Opt-in verification on the receipt adapter: the full ignored set minus the
audit module (run separately as the A/B) and the floor probe (never part of
ordinary verification) — 84 fixtures green in one sweep, including the M6
receipt goldens, warmed-encode invariance, and every production
symmetry/study/collider/gesture/stage-map fixture. Two sweep artifacts,
neither a defect of this tranche:
`gpu_two_layer_live_and_export_full_stack_matches_fixed_golden_at_24_30_and_60_fps`
fails inside a whole-binary sweep because winit permits one `EventLoop` per
process and an earlier windowed fixture claims it — it passes green in
isolation, which is how these fixtures are documented to run; and
`live_demo_sender_delivers_a_coloured_frame` requires a real external Spout
sender, the documented hardware-proof boundary.
