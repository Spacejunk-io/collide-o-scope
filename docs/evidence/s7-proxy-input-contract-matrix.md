# S7 — proxy bounded decode/audio input contract: cross-cutting completion matrix

Every row of the plan's matrix, with the exact test name that proves it or an
explicit reason it does not apply. An unstated not-applicable is
indistinguishable from an unrun check, so each is argued rather than asserted.

This tranche is the *definition* half of §7's content-addressed proxy: the
plan's ordering clause — "implement an FFV1/Matroska worker only after
defining bounded decode/audio inputs" — was unmet, and no worker may exist
ahead of the definition because the moment a worker gives `include_audio:
true` a meaning, that meaning is what algorithm version 1 means permanently
for every artifact anyone has keyed. The loop is therefore still open, and
this document says so in plain words: no encoder is integrated, no artifact
is produced, no decoder consults a cache, and the operator-facing "proxy
recommended" status remains unactionable until a worker tranche lands behind
this contract.

Branch point: `730a5f2` (`feat/web-control-panel`, the S6 merge). Its tree is
byte-identical to `4efa9da` (`git rev-parse <sha>^{tree}` agrees:
`d826ddf1…`), where hosted Linux, macOS, and Windows CI are all green and the
recorded gate was **1256 passed / 0 failed / 87 ignored**, plus `spout_probe`
0/0/0 and `eight_texture_floor_probe` 0/0/2. The baseline transfers by tree
identity rather than by assumption.

Encoder availability, confirmed rather than assumed: this host's FFmpeg
8.1.2 (`Gyan.FFmpeg.Shared`) lists the `ffv1` encoder and the `matroska`
muxer. `ProxyFormat::Ffv1Matroska` remains a cache-key vocabulary that does
not claim an encoder is installed; the confirmation is recorded here for the
worker tranche, not promoted into code.

| Surface | Required proof | Status |
|---|---|---|
| Domain | strict version, sanitization, exact default, hostile bounds | **Covered.** `proxy::tests::the_input_contract_gives_include_audio_its_meaning` (both absence causes kept distinct, first-ordered-stream copy, frame-timing mapping), `the_scale_law_is_even_floored_with_a_floor_of_two_and_original_is_exact`, `the_deadline_law_is_duration_derived_with_no_second_literal` (one-microsecond-over refusal at the source cap; maximum derived from base, factor, and cap rather than restated), `the_input_contract_refuses_hostile_probes_with_typed_errors` (stream-count cap at 64/65, zero-video refusal, inconsistent counts, checked-arithmetic overflow, zero and 16,385 px dimensions with 16,384 accepted, unknown duration, and settings validation preceding every probe check). Non-finite inputs are structurally unrepresentable: every probe field is an integer, which the type declares deliberately. |
| Version binding | the contract is owned by the algorithm version, tested against the key | **Covered.** `proxy::tests::the_audio_and_frame_rate_policies_and_the_versions_are_hashed_into_the_key` proves `include_audio` and the frame-rate policy each change the derived key, and — through the same private `update_cache_key` the derivation uses — that bumping either version changes the hashed stream, so a future semantic change provably changes every key. The pre-existing golden key in `cache_key_is_path_independent_settings_sensitive_and_has_a_golden` is byte-unchanged, proving this tranche attached meaning to version 1 without moving a single key bit. |
| Persistence | patch round trip, selected/missing tombstone, no runtime pixels | **Not applicable — nothing persists.** No `PatchState` field, no snapshot field, no wire action, and no sidecar were added. `ProxySettings` serde strictness is unchanged and still covered by `settings_and_observations_reject_hostile_serde`. There is no route, so there is no tombstone to preserve. |
| Look/Morph | values-only identity preservation and endpoint-exact discrete laws | **Not applicable — no authored creative state.** The contract is compile-time law plus pure functions; nothing here enters a Look, a Morph slot, or a blend. |
| Modulation | continuous bounded fields only; stable addresses | **Not applicable — no new address.** The contract exposes nothing modulatable and touches no `TARGETS` entry. |
| Dice/generator | deterministic domain-separated streams; topology preserved | **Not applicable — no RNG.** No seed, no stream, no generated value. |
| Planner | immutable stable routes, cycle rejection, exact resource preflight | **Not applicable — not a composition node.** The contract claims no scope, no dependency edge, and no image tap. The cache preflight's accounting is untouched and its tests are unchanged. |
| GPU | premult filtering/math, transaction laws, warm allocation | **Not applicable — zero GPU resources.** No pass, no texture, no buffer, no bind group, no pipeline, no persistent surface. The resource-delta table in the commit records all of it as zero. |
| Reset/freeze | patch, Look, cut, seek, resize, manual clear, both freezes, blackout | **Not applicable — no runtime state.** The contract is pure `const`s, law enums, and pure functions; there is nothing for a reset to clear or a freeze to hold. |
| History/recovery | one manual transaction; automation excluded | **Not applicable — no gesture.** Nothing here is operator-drivable yet; that is precisely the open loop. |
| Browser/native | strict actions, revision barriers, accessible diagnostics | **Not applicable — no action added.** `main.rs` is byte-unchanged; the operator-facing proxy status strings are exactly the pre-tranche ones, still covered by their existing tests, and still honest about sample counts including the sub-60-frame `MeasurementRequired` case. |
| Export | same evaluated plan, explicit initial state, bounded provenance | **Not applicable — no export path touched, argued from the diff.** The only non-test, non-documentation changes in this tranche are additive declarations in `src/proxy.rs` that no live code calls (every one carries `#[allow(dead_code)]` naming the deferred worker), plus `Display` arms reachable only through error variants that only the new dead-allowed function constructs. No renderer, exporter, decoder, or `main.rs` line changed, so the labeled export cases render through bit-identical logic. This is a weaker claim than a `framemd5` A/B and is stated as such; the worker tranche, which will touch the filesystem and a subprocess, must run the full same-branch A/B that S5a and S6 established — this tranche's argument does not transfer to it. |
| Compatibility | exact zero/default Legacy delegation and frozen pixels | **Covered.** The golden cache key is byte-unchanged (`cache_key_is_path_independent_settings_sensitive_and_has_a_golden`, untouched and passing), `ProxySettings::default()` is unchanged, and every pre-existing proxy test passes unmodified. Defining what version 1 means changed no bit of what version 1 derives. |

## What the worker tranche inherits

The twelve `#[allow(dead_code)]` attributes in `src/proxy.rs` before this
tranche named the worker's API; this tranche adds the input-contract items to
that same checklist under the same idiom. The worker must:

- answer every consumption question from `plan_proxy_input` — one predicate,
  many callers; restating a law beside it is the drift defect;
- hold a `MediaSafetyPolicy` plan for the encode's lifetime, exactly as every
  other media helper does — source admission is deliberately not re-derived
  in the contract;
- execute `ATOMIC_PROXY_CACHE_COMMIT_LAW` end to end with the crash test
  written first, validate decoded identity rather than exit codes, and run
  the full `framemd5` A/B;
- and either wire the decoder to consult the cache or state, as this
  document does, that the operator's recommendation is still unactionable.

## Gate

Six steps in CI order, on rustc 1.97.1 — the exact version today's stable
channel resolves to, matching what CI installs fresh (this host's default
`stable` lags at 1.96.1 and was not used):

1. `cargo fmt --all -- --check` — 0
2. `node --check static/app.js` — 0
3. `node --check docs/ui-ux/wireframe.js` — 0
4. `cargo check --locked --all-targets` — 0
5. `cargo test --locked --all-targets -- --test-threads=1` — **1261 passed /
   0 failed / 87 ignored**
6. `cargo clippy --locked --all-targets --all-features -- -D warnings` — 0

Delta against the branch-point baseline: **+5 passing** — exactly the five
new contract tests, with zero ignored added because the contract is pure CPU
law and needs no opt-in fixture. `spout_probe` 0/0/0 and
`eight_texture_floor_probe` 0/0/2, both unchanged, and the floor probe was
not run — it rewrites its tracked receipt in place, and this tree's
cleanliness is claimed by the gate above.

Three-platform CI at the branch point `730a5f2` was verified green on all
three jobs via the GitHub API before branching (this host has no `gh`; the
public check-runs endpoint answers with plain `curl`). Any cross-platform
claim for *this* tranche requires hosted CI at its own published SHA; a
local Windows run is not that evidence.
