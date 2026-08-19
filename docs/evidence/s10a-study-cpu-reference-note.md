# S10a — the Study CPU reference evaluator: evidence note

Gate 2's first tranche, unblocked by the operator's R1/R2/R3 rulings
(recorded on the opcodes, PR #23). This module gives the Study ABI's
opcodes their meaning: a pure CPU reference with no `wgpu`, clock,
filesystem, or UI dependency — the `gesture.rs` shape — that the S10b WGSL
interpreter will be checked against. Shader-source generation remains
permanently refused by `StudyAuthority`.

Branch point: `e6220c7` (`feat/web-control-panel`, the S9 + rulings
merges), verified green. Baseline: **1281 passed / 0 failed / 92 ignored**
(after both merges); with this tranche **1289 / 0 / 92** — seven evaluator
law tests plus the R3 window test.

## Semantics anchored, not invented

- **R1 guard**: `LoadHistoryColor` mirrors `temporal_originals.wgsl` — depth
  clamps to `valid_history - 1`, depth zero is the virtual current image, so
  a young program never reads unwritten content. Proven with
  nothing-committed, one-sample, and deep-request-clamps-to-oldest fixtures.
- **R2 randomness**: `deterministic_random` is the frozen layout — first
  eight canonical-digest bytes little-endian, XOR the ABI lane, XOR the
  mixed domain lane, XOR the tag, through the exact SplitMix64 finalizer
  `symmetry.rs` uses — mapped to `[0, 1)` by 24 mantissa-exact bits. The
  canonical digest is SHA-256 over `StudyDocument::to_json_bytes`'s exact
  output. Proven: document constant across recompiles and arbitrary frame
  contexts; independent domains; a renamed document rerolls.
- **R3 window**: `validate()` now accepts `major == current && minor <=
  current` — landed in this commit as the ruling required, behaviorally
  identical to exact equality until the first minor bump. Newer minors and
  other majors reject; pinned by test.
- **Hue law**: `HueRotate` mirrors `rack_node.wgsl`'s HSL round trip line
  for line with the shader's `fract` wrap; red ±⅓ turn lands on green/blue
  at f32 epsilon, a full turn is the identity, alpha never rotates.
- **The bound law**: every computed component clamps to
  `±STUDY_MAX_FINITE_VALUE`; non-finite lands on the documented neutral 0.
  Discriminated by a fixture whose output differs measurably from an
  unbounded evaluator's.
- **Compile/evaluate split**: `CompiledStudy::compile` validates, resolves
  every R2 value to a constant, and lists required history ages once;
  `evaluate_pixel` is infallible and allocation-free per pixel — the
  compile-on-change, O(n)-per-frame law.

Also honored: Vector2 evaluates honestly and stays the recorded dead end
(no conversion opcode was added); frame inputs sanitize to documented
neutrals (NaN band → 0, out-of-range phase clamps); compile refuses
whatever `validate()` refuses, so a compiled study can never consume an
undeclared capability.

## What this tranche deliberately does not do

No renderer wiring, no GPU pass, no snapshot or wire surface — the module's
only callers are its law tests until S10b, and it carries a scoped
`#![allow(dead_code)]` naming that consumer honestly rather than faking a
premature integration. No new authored state anywhere, so
Look/Morph/modulation/Dice rows are structurally not applicable. The
decoded-`framemd5` A/B is likewise not applicable: no render, export, or
decode path file changed (the one `study.rs` edit is the validator gate).

Gate on `RUSTUP_TOOLCHAIN=1.97.1`: fmt, both node checks, check, **1289 /
0 / 92**, clippy `-D warnings` — green.
