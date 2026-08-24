# RFC D4 — one accepted creative mutation seam

Status: **implemented; the complete origin inventory and frozen-v1 compatibility
fixtures pass**.

`AcceptedCreativeMutation/v1` is produced only after origin parsing,
stable-identity and revision validation, safety refusal, and the shared finite
value law. It contains an accepted 30 Hz reference tick, canonical stable
address, typed value, monotonic admission identity, and a bounded origin tag
used only for provenance. Live application and take recording observe that one
immutable value; replay semantics never depend on the origin.

The v1 vocabulary covers only currently recordable scalar/color/enum creative
parameters. Blackout, freeze, pause, topology, route edits, recorder control,
replay, refusals, and dropped/coalesced batches remain excluded. Prepared clip
or Scene activation requires a future append-only vocabulary entry only after
its exact reference-tick/beat law is proven. Raw MIDI/OSC/touch events and wall
clock timestamps are never recorded.

The exhaustive origin matrix covers native UI, browser, phone gesture/pad,
MIDI, OSC, host automation, and replay. For each supported value, every live
origin produces the same address/value/tick once; a shared admission identity
deduplicates simultaneous duplicate delivery. Stale, unsafe, replay-origin,
and refused work produces no mutation. Exact v1 take hashes and machine-checked
coverage prove that no enumerated origin bypasses or double-applies the seam.

The implementation and complete gate receipt are recorded in
`docs/evidence/d4-accepted-creative-mutation.md`.
