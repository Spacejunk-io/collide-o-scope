# RFC D1 — Study motion ABI 1.1

Status: **retained; ABI 1.1 implemented additively and ABI 1.0 remains frozen**.

ABI 1.1 appends, and never renumbers, six typed operations after the sixteen
ABI 1.0 opcodes: vector X, vector Y, magnitude, normalized direction, dot with
a finite constant vector, and bounded scalar-to-color construction. The last
operation clamps its scalar input to `[0,1]` and mixes two finite authored
colors; it does not reinterpret register storage. A document selects the
opcode table and cost model by exact `(major, minor)` validation. ABI 1.0 must
reject every appended opcode and preserve its canonical bytes, error classes,
GPU words, and corpus hashes.

Every appended operation costs one instruction, uses the existing 64-register
and 256-instruction ceilings, applies the existing finite `±65504` bound after
each result, and has identical CPU and fixed-WGSL semantics. Magnitude uses
`sqrt(x*x+y*y)` after finite bounding; normalized zero returns `[0,0]`; hostile
NaN/Inf host inputs first follow the current bound law. No shader source,
dynamic loop, filesystem/network authority, implicit coercion, or extra load
capability is introduced.

The composition planner treats a resolved, observable ABI 1.1 motion load as
an explicit primitive-field consumer. It admits the scope field even when
ordinary Motion is exactly zero, carries `required_as_study_input` separately
from donor and Refresh Garden routing, and includes that fact in resource
preflight and topology identity. Enable/wet changes do not move its slot.
Unresolved documents, group scopes (which have no Motion v1 primitive), and
ABI 1.0's deliberately dead vector load receive the defined neutral input and
do not allocate speculative fields.

## Promotion evidence

Retained on the available physical adapter after all named semantic gates:

- the CPU/reference suite covers zero, axes, diagonal, maximum, NaN/Inf,
  absent motion, the 256-instruction edge, exact-version rejection, and a
  motion-to-color program;
- the fixed GPU interpreter agrees with the CPU reference across every ABI
  1.0 opcode and across the ABI 1.1 hostile/edge corpus;
- a pure production-planner test proves one exact-zero field is admitted only
  for a resolved ABI 1.1 Study, remains topology-stable while disabled/dry,
  and is absent for unresolved and ABI 1.0 programs;
- a physical-GPU composition test drives a `[3,4]` codec vector through the
  production motion encoder, parity-selected Study binding, and fixed
  interpreter to audience pixels. Moving differs from stationary; stationary
  equals absent; repeated warm renders are deterministic and allocation-free.

Focused commands (all exit 0):

```text
cargo test --bin collide-o-scope study -- --test-threads=1
cargo test --bin collide-o-scope renderer::study::tests::gpu_study_abi_1_1_motion_reaches_output_and_absence_is_neutral -- --ignored --test-threads=1
cargo test --bin collide-o-scope renderer::study::tests::gpu_study_interpreter_matches_the_cpu_reference_across_every_opcode -- --ignored --test-threads=1
cargo test --bin collide-o-scope observable_study_motion_admits_an_exact_zero_scope_field -- --test-threads=1
cargo test --bin collide-o-scope production_study_abi_1_1_motion_field_reaches_the_pixels -- --ignored --test-threads=1
```

No implicit coercion, old-bytecode reinterpretation, general shader source,
or new host authority was admitted.
