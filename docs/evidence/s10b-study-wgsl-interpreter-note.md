# S10b — the fixed Study WGSL interpreter: evidence note

Gate 2's second tranche: the CPU reference gains its GPU twin. One shader
(`study_interpreter.wgsl`), compiled once and never generated — a compiled
Study arrives as a bounded uniform instruction buffer and the fragment
stage walks it with a typed register file, the Symmetry sector-table
precedent. Shader-source generation stays permanently refused by
`StudyAuthority`.

Branch point: `1ffb52d` (`feat/web-control-panel`, the S10a merge),
verified green. Baseline: **1289 passed / 0 failed / 92 ignored**; with
this tranche **1292 / 0 / 94** — three hosted law tests and the two GPU
agreement fixtures.

## The design, compressed

**The frozen encoding.** GPU opcodes 0…15 are append-only exactly like
`NodeKindTag` codes; `words[0]` carries the opcode low and the auxiliary
operand (mix amount, hue turns register, history age, audio band) high;
the immediate carries constants — and every `LoadDeterministicRandom` is
resolved to an immediate at compile, so the GPU never hashes and the R2
randomness cannot drift between the halves. The full array is always 256
slots (8,192 bytes, uniform-safe); unused slots are zero, so **a study
swap is exactly two `write_buffer` calls** into fixed buffers — no
reallocation, no pipeline change, proven deterministic by fixture.

**Two textures, no sampler, one pass.** Carrier plus the committed
clean-history D2 array, every lookup a `textureLoad`, inside the ordinary
three-texture ceiling. The ring layer index is the temporal shader's
cursor arithmetic in integer form, and the R1 guard is the same law the
CPU reference implements.

**One hue law, three copies, one test.** The interpreter carries
`rack_node.wgsl`'s `rgb_to_hsl` / `hue_to_rgb` / `hsl_to_rgb` character
for character, asserted by source text exactly as the Symmetry Field
asserts its copy. The S10b domain clamp (semantics version 2) clamps the
hue operand to unorm on **both** evaluators before the round trip — the
range the HSL math is defined on — because outside it the divisions can
go non-finite and WGSL leaves non-finite handling implementation-defined;
with the clamp, no instruction chain can produce a non-finite intermediate
from valid inputs on either half, and the bound law's non-finite branch
becomes purely defensive. Version bumped 1 → 2 honestly, days after v1
and before any consumer existed.

**One sanitation law, applied once per half.** `StudyGpuFrameUniforms::
from_context` sanitizes bands and phase with the identical rule the CPU
reference applies at load, so the two halves observe the same numbers.
`LoadMotionVector` binds nothing: ABI 1.0's Vector2 is provably unable to
reach the output, so the dead lane loads zero with zero image consequence
— stated in the shader rather than hidden.

| Surface | Required proof | Status |
|---|---|---|
| Encoding | frozen layout, append-only codes | **Covered, hosted.** `the_gpu_encoding_is_the_frozen_layout` pins every opcode's words and immediates, the resolved-random immediate against the R2 hash, the instruction count, and the zeroed tail. |
| Hue law | one law, shared by source text | **Covered, hosted.** `the_interpreter_shader_shares_the_rack_hue_law_character_for_character`, plus `hue_rotation_clamps_its_operand_to_the_unorm_domain` for the v2 clamp on the CPU half. |
| CPU/GPU agreement | every opcode, the guard, randomness | **Covered, opt-in GPU** (AMD Radeon RX 6950 XT / Vulkan, this host): `gpu_study_interpreter_matches_the_cpu_reference_across_every_opcode` — the shared every-opcode document at 2e-5 per channel over a 32×32 gradient carrier and a 24-layer ring, with the young-ring (valid 5, age 7 clamped) and committed-ring frames both agreeing **and demonstrably differing from each other**, so the guard's work is observable, not vacuous. |
| Swap/warm | fixed buffers, deterministic | **Covered, opt-in GPU.** `gpu_study_swap_is_two_writes_into_fixed_buffers_and_stays_deterministic` — a second document through the same executor and bind group matches its own CPU reference, and re-rendering is byte-identical. |
| Resource shape | inside the ordinary ceilings | **By construction.** 1 render pass, 2 sampled textures (≤ 3), 0 samplers, 64 + 8,192 uniform bytes, 0 persistent surfaces — the executor owns two fixed buffers and nothing else. |
| Authored surface | patch/Morph/Dice/browser closure | **Deliberately absent.** Where a Study plugs into the composition (rack node kind, master slot) is an unopened product decision; the whole chain stays behind a scoped dead-code allow that names that tranche. No wire action, no snapshot field, no PatchState field. |
| Render/export A/B | decoded-`framemd5` parity | **Not applicable, argued.** Nothing reachable from the frame loop changed; the one edit outside the new modules is a test-visibility change in `study_eval`'s test module. |

Gate on `RUSTUP_TOOLCHAIN=1.97.1`: fmt, both node checks, check, **1292 /
0 / 94**, clippy `-D warnings` — green. GPU fixtures green on the receipt
adapter.
