# R1 bounded prepared-source transition kernel

`src/source_transition.rs` freezes the additive `Cut`/`Dissolve` vocabulary,
reference-tick or rational-beat durations, explicit refuse-on-interruption law,
and an all-or-none Scene admission plan capped at two simultaneous dissolves.
Absent, explicit Cut, and zero-duration inputs produce the exact zero-resource
legacy plan. Both sources must be ready, generation-current, independently
mapped, and reserved before admission; one late or stale member refuses the
whole Scene. Scratch accounting is checked and refuses one byte over cap.

The deterministic clock advances only from accepted 30 Hz reference ticks or
accepted rational beat positions. Program Pause and Media Freeze hold phase;
redraws and blackout do not invent time. A CPU premultiplied reference pins
bit-exact endpoints and deterministic intermediate arithmetic. Each source has
its own fixed-point output-to-source UV mapping, and the named blend seam is
pre-layer-effects so an eventual GPU implementation applies the layer rack
once.

This is the bounded planner/reference prototype requested by R1, not a shipped
renderer capability. It does not yet retain outgoing `LayerSourceActivation`
objects in the live engine, add the blend shader, expose authoring controls, or
retire completed sources through the reaper. The required live/export pixel
fixture and three-layer 720p p95/p99 campaign have therefore not run. Until
those gates exist and pass, capability truth must keep prepared activation at
Cut and describe Dissolve as research-only.
