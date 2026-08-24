# P10 — capability registry and audit-campaign truth closure

Date: 2026-08-24

Status: **implemented; generated artifacts and contradiction gates pass**.

## Final D1–D5 dispositions

`docs/campaigns/audit-campaign-status.json` now uses one closed five-value
campaign vocabulary: `retained`, `implemented`, `evaluation`, `deferred`, and
`rejected`. The final audit records are:

| Campaign | Status | Remaining gate |
|---|---|---|
| D1 Study motion ABI 1.1 | `implemented` | `complete` |
| D2 photosensitivity advisor | `evaluation` | accessibility/legal review and P1 GPU timing |
| D3 portable show bundle | `retained` | machine-A → clean-machine-B live-export reproduction |
| D4 accepted creative mutation | `implemented` | `complete` |
| D5 straight-alpha/key-fill | `retained` | application action, live acquisition, and P1 readback evidence |

D3's deterministic build/inspect/import core and D5's exact artifact
publishers remain valuable retained work, but neither has an operator action.
D2 remains evaluation-only and is not constructed by the application. Their
open gates are not erased by calling their cores implemented. The D4 RFC was
updated from its stale pre-implementation status to its completed origin/hash
receipt. The earlier D5 planning RFC is now an explicit superseded redirect,
not a conflicting availability statement.

## Executable registry decision

Six additive registry keys are justified because their production paths are
operator-visible:

- `study_motion_abi_1_1`: implemented for live Program and offline export,
  with ABI-pinning and physical-adapter coverage limitations;
- `accepted_creative_mutation_v1`: implemented for browser/native control and
  live performance-take recording, limited to the frozen v1 record vocabulary.
- `transactional_control_listeners`: independently bound and reported IPv4/
  IPv6 loopback HTTP plus fail-closed LAN HTTPS, with one versioned, synced,
  atomically replaced TLS identity. Browser/native/backend surfaces are
  implemented; second-host reachability and packet-capture proof remains an
  evaluation surface.
- `correlated_engine_gpu_timing`: bounded ingress-to-apply/apply-to-submit
  correlation and asynchronous six-stage GPU timing are published in native
  and browser Stage Health. Timestamp support remains adapter-conditional, the
  fixed-fixture instrumentation-overhead gate is unexecuted, and engine submit
  time is explicitly never called photon time. The separate physical optical
  fixture remains unexecuted.
- `source_descriptor_color_truth`: frozen, provenance-carrying color/display
  descriptors and actual conversion policy reach the decoder, native Stage
  Health, proxy receipts, and export provenance. Live/export pixel surfaces
  remain `EvaluationRequired`: clean aperture, SAR, rotation, and mirror are
  not applied by the renderer, exactly as the P4b stop receipt states.
- `supervised_gpu_recovery_phase_a`: the device-loss latch, distinct exit 75,
  bounded shutdown hooks, one-relaunch supervisor, and operator-owned recovery
  surface are implemented. Physical packaged relaunch timing remains an
  evaluation surface. No live-Program surface is advertised; in-process Phase
  B and transparent audience continuity remain unavailable.

The registry deliberately has no D2 advisor, D3 show-bundle, or D5 alpha-export
key. Adding any would advertise a surface that the current application cannot
reach. Their campaign records and RFC receipts remain the truthful discovery
surface until integration gates pass.

## Evidence-boundary completeness

Every generated capability status record now carries at least one nonempty
receipt ID. The nine formerly empty external/deferred keys share the dedicated
`p10-external-deferred-capability-evidence-boundary` receipt: bounded mesh warp,
capture input, NDI input/output, Spout input/output, Syphon input/output, and
zero-copy decode. A source-level test pins that exact nine-key set so the
boundary cannot silently spread to an unrelated capability.

The receipt does not promote a stopped capability. In particular, Windows
Spout input/output remain internally implemented while their real external
sender/receiver interoperability proof remains unexecuted and their physical
surfaces remain evaluation-only. The other seven keys retain their existing
deferred or platform-unavailable decisions and typed reasons.

The capability generator now validates, before either writing or checking:

- campaign schema, unique IDs, and the closed five-value status vocabulary;
- exact D1–D5 status/gate tuples;
- D1/D4 implemented registry records on every generated platform;
- implemented runtime-fact records, nonempty surfaces/evidence/limitations,
  and exact conservative surface boundaries for P0, P1, P4b, and P7 Phase A;
- P1's typed non-photon and unexecuted physical/performance limitations,
  P4b's evaluation-only live/export integration, and P7's absence from the
  live-Program surface plus its typed Phase-B-unavailable limitation;
- absence of D2/D3/D5 operator capability claims;
- the authoritative status phrases in all five RFCs, including the superseded
  D5 redirect;
- existing proxy and evaluation-only hardware-decode contradiction gates;
- nonempty evidence for every generated status record, with seeded missing-ID
  and empty-ID variants proving that generation fails closed.

Generated JSON key strings are machine-checked against every
`CapabilityKey::as_str()` value, closing the initially detected
`study_motion_abi11`/`study_motion_abi_1_1` drift. The generator refreshed
`docs/capability-registry.json`, `docs/capability-registry.md`, and the bounded
README summary from executable facts.

## Mechanical lint closure

`ControlListenerStatus` now derives `Default` with `Stopped` explicitly marked
as the default variant. Its existing listener-lifecycle test directly asserts
that law. No wire representation or runtime state changed.

## Typed remediation transaction closure

The early authored Master-bypass validator now preserves the planner's typed
`AmbiguousMasterBypass` refusal instead of flattening that condition into a
generic diagnostic. It still examines every authored bypass bit, including a
currently hidden or non-contributing layer, so the existing future-frame safety
law is unchanged. Ordinary preflight now publishes the same stable constraint
code, invariant, affected layer identity, revision-bound candidate, and
planner-evaluated immutable preview as the composition planner.

The physical-GPU transaction fixture constructs an actual custom Master rack
whose authored step precedes the canonical marker, while leaving that refused
rack detached from the live program. Ordinary preflight derives the one-bit
`DisableConflictingBypass` candidate and an Advanced-plan consequence; cancel
and preview selection are exact no-ops, a stale revision is refused, and normal
action dispatch performs confirmation through one browser-manual history and
recovery transaction. The exact refused staged rack then preflights successfully
without being published. Ordinary undo and redo restore the complete canonical
world, issue one retained-pixel barrier apiece, and leave the final redone patch
durable in an isolated recovery journal. No diagnostic, candidate, operation, or
preview is injected by the fixture.

## Gates

Executed with the repository's Visual Studio x64 and FFmpeg environment:

```text
cargo run --locked --bin generate_capabilities -- --check
cargo test --locked --lib capability::tests -- --nocapture
cargo test --locked --bin generate_capabilities -- --nocapture
cargo test --locked --bin collide-o-scope tls_identity::tests -- --nocapture
cargo test --locked --bin collide-o-scope action_correlation::tests -- --nocapture
cargo test --locked --bin collide-o-scope gpu_timing::tests -- --nocapture
cargo test --locked --bin collide-o-scope source_descriptor::tests -- --nocapture
cargo test --locked --bin collide-o-scope gpu_recovery::tests -- --nocapture
cargo test --locked --bin collide-o-scope app_state_tests::authored_bypass_capability_is_independent_of_final_program_vhs -- --exact --nocapture
cargo test --locked --bin collide-o-scope contributing_bypass -- --nocapture
cargo test --locked --bin collide-o-scope remediation -- --nocapture
cargo test --locked --bin collide-o-scope app_state_tests::p10_confirmed_remediation_uses_revision_history_preflight_and_undo_transaction -- --ignored --exact --nocapture
cargo clippy --locked --lib --bin generate_capabilities -- -D warnings
cargo clippy --locked --bin collide-o-scope -- -D warnings
rustfmt --edition 2021 --check src/capability.rs src/bin/generate_capabilities.rs src/motion_sidecar_wire.rs src/web/action_wire.rs
rustfmt --edition 2021 --config skip_children=true --check src/main.rs
git diff --check -- src/capability.rs src/bin/generate_capabilities.rs README.md docs/capability-registry.json docs/capability-registry.md docs/evidence/p10-capability-campaign-truth-closure.md docs/evidence/p10-external-deferred-capability-evidence-boundary.md docs/evidence/d4-accepted-creative-mutation.md
git diff --check -- src/main.rs docs/evidence/p10-capability-campaign-truth-closure.md docs/evidence/v1.7.0-improvement-audit-release-receipt.md
```

Final results:

- generator check: exit 0;
- capability truth tests: 10 passed, 0 failed;
- generator/contradiction tests: 3 passed, 0 failed;
- TLS identity: 6 passed, 0 failed;
- action correlation: 6 passed, 0 failed;
- GPU timing: 4 passed, 0 failed;
- source descriptor/color truth: 7 passed, 0 failed, 1 ignored external-FFmpeg fixture;
- supervised GPU recovery model: 4 passed, 0 failed;
- authored-bypass validator: 1 passed, 0 failed;
- contributing-bypass planner focus: 3 passed, 0 failed;
- non-GPU remediation focus: 2 passed, 0 failed, 1 physical fixture ignored by the default focused command;
- exact module-qualified physical remediation transaction: 1 passed, 0 failed;
- strict library + generator Clippy: exit 0;
- strict production-binary Clippy: exit 0;
- focused Rustfmt checks: exit 0;
- focused diff checks: exit 0.

The remediation follow-up changes `src/main.rs` only at the early typed
Master-bypass seam and its physical transaction proof. It does not change
`src/web/state.rs`, `static/app.js`, `src/diagnostics.rs`, the capability
registry facts, or any generated capability artifact.
