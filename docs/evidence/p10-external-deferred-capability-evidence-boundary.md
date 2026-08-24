# P10 external/deferred capability evidence boundary

Date: 2026-08-24

Status: **truth boundary recorded; no deferred or unavailable capability is promoted**.

Receipt ID: `p10-external-deferred-capability-evidence-boundary`

## Scope

This receipt supplies the nonempty evidence boundary for exactly nine registry
keys that previously had no receipt identifier. A receipt identifier means the
status decision is auditable; it does not mean an external backend, SDK,
device, venue, or interoperability run exists.

| Registry key | Current-tree fact | Evidence still required |
|---|---|---|
| `bounded_mesh_warp` | No venue requirement or backend is recorded; the evaluator remains deferred on the venue-requirement gate. | Demonstrated venue need, bounded backend integration, and physical proof. |
| `capture_input` | No external capture backend is integrated. | Supported target hardware, backend integration, and interoperability proof. |
| `ndi_input` | No SDK/license or network-policy authorization is recorded and no backend is integrated. | Explicit SDK/license and network authorization before endpoint integration/proof. |
| `ndi_output` | No SDK/license or network-policy authorization is recorded and no backend is integrated. | Explicit SDK/license and network authorization before endpoint integration/proof. |
| `spout_input` | The Windows receiver path is internally integrated in `src/spout_in.rs`, layer/runtime preparation, and the live upload path. macOS/Linux remain unavailable. | A real external sender on the target Windows adapter; the ignored live-sender fixture has not been executed for this candidate. Offline export remains deterministic black. |
| `spout_output` | The Windows sender path is internally integrated in `src/spout_out.rs`, live final-program submission, and the Windows-only `spout_probe` consumer. macOS/Linux remain unavailable. | A real external receiver on the target Windows adapter; the sender/receiver pair has not been executed for this candidate. |
| `syphon_input` | Syphon is macOS-only by definition and no backend is integrated; other platforms are unavailable. | macOS backend integration and an external sender proof. |
| `syphon_output` | Syphon is macOS-only by definition and no backend is integrated; other platforms are unavailable. | macOS backend integration and an external receiver proof. |
| `zero_copy_decode` | Decode still crosses a system-memory boundary; no zero-copy backend is integrated. | A supported decode/upload interop path, exact ownership/accounting proof, and target-hardware validation. |

## Spout boundary

The Windows Spout rows are `Implemented` for internal control/live-program
surfaces because production input and output modules exist and are wired. Their
`PhysicalVenue` surfaces remain `EvaluationRequired`, and this receipt does not
stand in for the missing two-application, real-adapter interoperability run.
Internal implementation evidence and physical interoperability evidence are
therefore deliberately different facts.

## Executable closure

`src/capability.rs` attaches this receipt ID to exactly the nine keys above on
every generated platform. Every capability record now has at least one
nonempty receipt ID. The production generator rejects a generated document if
any status record has no receipt or contains an empty receipt ID; its seeded
broken-variant test clears each form and proves the check fails closed.

Generated status, typed reasons, surfaces, and limitations remain derived from
`CapabilityRuntimeFacts` and `evaluate_scale_capability`; this document is not
a runtime predicate.

## Gates executed

With the repository Visual Studio x64, FFmpeg, and libclang environment:

```text
cargo run --locked --bin generate_capabilities
cargo run --locked --bin generate_capabilities -- --check
cargo test --locked --lib capability::tests -- --nocapture
cargo test --locked --bin generate_capabilities -- --nocapture
cargo clippy --locked --lib --bin generate_capabilities -- -D warnings
rustfmt --edition 2021 --check src/capability.rs src/bin/generate_capabilities.rs
```

Results: production generation and check exited 0; capability tests passed
10/10; generator tests passed 3/3, including the missing/empty receipt seeded
variants; strict Clippy and focused Rustfmt exited 0. The generated JSON has
zero missing/empty evidence arrays and exactly nine boundary-receipt records on
each of Windows, macOS, and Linux.
