# P2a stable modulation topology safe-subset receipt

Status: retained address-book/offset safe subset; broader structural-plan and
planning-p99 candidate stopped.

The audit explicitly permits P2a to stop after `StableModAddressBook` and
offset caching when complete invalidation for a broader compiled plan has not
been proved. This receipt claims only that safe subtraction. It does not infer
a frame-latency win from reuse counters.

## Retained scope

- `StableModTopologyCache` owns one exact `StableModAddressBook`, its retained
  per-frame offset vector, and retained topology/signature scratch. Warmed
  value-only queries reuse the same address-book allocation and offset
  capacity; they update bounded values in place.
- The cache key is an exact stable-ID topology signature, never a positional
  layer index. It contains layer/group order, rack scope, node stable ID and
  kind, modulation shape, and group-matte presence.
- A topology candidate is compiled completely before publication. Failed
  compilation leaves the previous signature/address book intact.
- Invalidation counters name cold start, layer order/membership, group
  order/membership, node order/membership, node kind/modulation shape,
  group-matte topology, and unclassified signature changes. The structural
  revision advances only with a published rebuild.
- The optional `COLLIDE_O_SCOPE_VERIFY_STABLE_MOD_CACHE` diagnostic independently recompiles
  the address book. A mismatch increments a dedicated counter, repairs the
  cached book, records the forced-recompile invalidator, and never silently
  reuses the stale value.
- Live Main owns one cache across accepted frames. Offline export seeds its own
  cache from the immutable address book compiled at job admission; the focused
  equality fixture proves both paths produce the same address map and stable
  frame offsets.

## Explicit stop boundary

The retained signature deliberately stops before Study program topology,
source texture/raster epochs, output size/precision, renderer generation,
flattened composition/dependency execution shape, and GPU resource/uniform
slots. Those domains are not used to authorize reuse here. A focused test
changes Study program content and proves that this address-only cache remains a
hit while the separately compiled Study/render topology remains authoritative.

Consequently this receipt does **not** close the audit's broader P2a item 3 or
the 8/24-layer planning-p99 gate. No 25% planning improvement, no complete
engine-allocation count, and no one/two-layer non-regression result is claimed.
P2b has its own independent GPU-object receipt.

## Focused verification

Executed on the shared Windows workspace after the final safe-subset code:

- `cargo test --locked --bin collide-o-scope stable_topology_cache -- --nocapture`:
  2 passed, 0 failed. The tests cover warmed book/offset reuse and every
  admitted address-topology invalidator.
- `cargo test --locked --bin collide-o-scope modulation::tests::precompiled_export_cache_and_live_cache_share_exact_plan_semantics -- --exact --nocapture`:
  1 passed.
- `cargo test --locked --bin collide-o-scope modulation::tests::forced_recompile_mode_repairs_a_stale_compiled_signature -- --exact --nocapture`:
  1 passed.
- `cargo test --locked --bin collide-o-scope modulation::tests::address_cache_stops_before_study_program_topology -- --exact --nocapture`:
  1 passed.

These five focused tests establish the retained safe subset and its stop. The
final all-target suite and strict Clippy remain release-level gates recorded in
the aggregate release evidence, not substitutes for the unexecuted p99 matrix.
