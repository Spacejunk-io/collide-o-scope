# Enrichment handover — for the successor session

The enrichment plan is a sixteen-tranche program (`B1`–`B16`, three waves)
enlarging the instrument's expressive surface. The full tranche specifications
live in the operator's plan document (a claude.ai artifact the operator will
cite); this handover carries everything a successor session needs to continue
without it, and the operator can supply the artifact when a tranche's fine
detail is wanted.

Execution order: **Wave 1** B2, B3, B12, B13 — feed machinery we already own.
**Wave 2** B1, B4, B8, B16, B5, B7 — new machinery. **Wave 3** B9, B10, B11,
B6, B14, B15 — performance, monitoring, ergonomics. B13 and B15 may interleave
anywhere as relief work. Nothing in the plan raises the selective-VHS budget,
moves the media-safety bounds, touches the Study authority model, or adds a
second full-frame history ring.

## Status

| Tranche | State | Where |
|---|---|---|
| B2 — synthetic + shaped motion fields | **Complete** | PR #30 (merged), PR #31 (merged); `docs/evidence/b2-procedural-motion-fields-note.md` |
| B3 — the feedback rig | **Complete** | PR #32 (merged); `docs/evidence/b3-feedback-rig-note.md` |
| B12 — time-displace maps | **Complete** | PR #33/#34 (merged); `docs/evidence/b12-time-displace-note.md`; CLAUDE.md "B12 time-displace maps" |
| B13 — the small-effects tranche | **Complete** | `docs/evidence/b13-small-effects-note.md`; CLAUDE.md "B13 small effects" |
| B1 — the Scan Processor | **Complete** | `docs/evidence/b1-scan-processor-note.md`; CLAUDE.md "The B1 Scan Processor" |
| B4 — display physics | **Complete** | `docs/evidence/b4-display-physics-note.md`; CLAUDE.md "B4 display physics" |
| B14 — failure switches | **Partially landed early** | `servo_defeated` shipped inside B3; the remaining piece is `sync_latched` on the tape/NTSC-adjacent shear model |
| B8, B16, B5, B7, B9, B10, B11, B6, B15 | Open | — |

Each landed tranche documents itself in `CLAUDE.md` (B2 under "B2 procedural
motion fields" and the Motion sections; B3 under "The B3 feedback rig") and in
its evidence note. Read those before extending either subsystem.

## Next up

Wave 2 continues with **B8, the mixing boundary** — the melting edge over
every coverage matte (evaluate the matte four points out; disagreement is
the edge, its direction the normal; drag the incoming picture along it and
dissolve the stage's own previous frame back in, so the smear creeps), the
dirty-mixer fault stage on buses (event clock, four fault laws, hashed like
Shift in a fresh RNG domain), and the blend/wipe audit (the classic blend
family as a closed append-only enum with existing laws keeping their
indices, plus the wipe-pattern vocabulary and key dressing). Melt at zero
never allocates — the delegation law. Fetch the plan document before
starting it.

## The working method every tranche follows

1. **Branch** from `feat/web-control-panel` (or stack on the open PR chain);
   the operator (George) merges PRs via the GitHub web UI — never self-merge.
2. **Map the seams first.** Every authored value closes over: patch DTO
   (skip-serialized at default so old bytes and canonical hashes keep),
   sanitize-on-load with *neutral* non-finite fallbacks, wire action in BOTH
   validators (`web/server.rs` + `main.rs`) plus the applier, snapshot
   (additive `#[serde(default)]` block), panel rows (mind the range-count
   assertion in `web/state.rs` protocol tests), modulation targets (append to
   the END of `LAYER_TARGET_SUFFIXES`; master targets insert BEFORE `morph`,
   which must stay last), Morph law (values blend, angles wrap, discrete laws
   recall an endpoint at the midpoint), Dice/generator RNG in fresh per-field
   domains (old streams byte-stable; bump `GENERATOR_VERSION` only when a
   given seed's output changes), export through the shared plan with
   frame-indexed time, and a labeled export case.
3. **CPU reference first.** The law is a pure function the WGSL follows
   expression for expression; analytic fixtures test the reference, and a GPU
   fixture or labeled A/B ties the shader to it.
4. **Exactness A/B.** Keep a pre-change render of an existing labeled case on
   disk, re-render on the new build, compare `ffmpeg -f framemd5` — the
   default path must be decoded-frame identical across the change.
5. **The six-step gate**, all of it, before any claim (see
   `CLAUDE.md` "Verification"; the vcvars preamble and the CI-toolchain check
   live in the session memory notes). Opt-in GPU fixtures run on this host
   (RX 6950 XT / Vulkan): `cargo test --locked <name> -- --ignored`.
6. **Evidence note** in `docs/evidence/` + a `CLAUDE.md` section + a
   verification bullet, then commit, push, and open the PR via the GitHub API
   (token available through `git credential fill`; the `gh` CLI is absent).
   CI is verified per suite with `python scripts/check-ci-status.py <sha>`.

## Receipts and frozen contracts a successor will trip over

- `temporal.rs` pins the SHA-256 of `temporal.wgsl` in
  `originals_shader_contract_is_additive_bounded_and_reuses_three_texture_resources`,
  plus per-region `textureSample`/`textureLoad` counts. Editing either
  temporal shader means re-pinning deliberately, in the same commit, with a
  comment naming the tranche.
- The M6 precision receipt fixture
  (`gpu_precision_receipt_measures_real_still_and_temporal_workloads`,
  opt-in) pins the shader-bundle digest inline and prints the receipt JSON as
  `M6_GPU_RECEIPT=…`. After any change to its source manifest: update the
  inline digest, re-run, and overlay the printed values onto the tracked
  `docs/evidence/m6-precision-gpu-receipt.json` — preserving that file's field
  order and its extra metadata (`measured_at`, `toolchain`), which the printed
  JSON does not carry. The six output SHAs must not move unless pixels
  legitimately changed.
- `web/state.rs` pins the exact count of `<input type="range"` tags in
  `index.html` (117 after B3) — bump it with every added slider.
- The apply-pass uniform (`MotionApplyGpuUniforms`, 1,680 bytes after B2) and
  the rig uniform (96 bytes) carry compile-time size assertions; WGSL structs
  change in the same commit.
- Generated-piece hashes: any new DTO field must be skip-serialized at its
  default or every pre-existing canonical hash breaks.

## Standing decisions made during B2/B3

- One boundary numbering serves the whole program
  (`Transparent = 0, Mirror = 1, Wrap = 2, Hold = 3`); reuse
  `MotionBoundaryMode`/its config for any new edge law.
- Deterministic per-cell randomness uses the shared `cellular_avalanche`
  integer hash with a fixed domain constant and 24-bit unit floats; no wall
  time, no authored seed unless Dice needs to reroll it.
- Anything rate-like normalizes per 1/30-second reference tick (linear for
  additive terms, exponentiation for multiplicative ones, clamped
  tick-fraction mix for nonlinear stages).
- Servos and other self-regulating loops must be deterministic per-pixel laws,
  never readback-driven — live/export parity forbids measured-mean feedback.
- The B2 remainder decision stands: the trash event clock (8 Hz) and shove
  amplitude are fixed law, not authored controls.
