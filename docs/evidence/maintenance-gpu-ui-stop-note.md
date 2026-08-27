# wgpu 30 and egui 0.36 maintenance — evidence-backed STOP note

Date prepared: 2026-08-27
Topic: `docs/maintenance-gpu-ui-stop`
Read-only audit snapshot:
`48d85ea943069dbcec7c718d42daeac025da4547`
Pinned integration base:
`a5b0de1584ca0cad95895577c384a06915ae2047`
Status: **integrated evidence-backed STOP; exact-commit CI green — retain
wgpu 29.0.4 and the egui 0.34.3 family until the physical GPU/UI matrix is
available**

This closes perfection handover §3.8(d) as a bounded negative result. No
GPU/UI dependency, source, vendor tree, resource ceiling, action, schema, or
capability claim changes in this tranche.

## Upstream ruling

The reviewed candidate is wgpu 30.0.1 with the egui 0.36.1 family:

| Evidence | Exact observation |
| --- | --- |
| wgpu candidate | 30.0.1 |
| wgpu-hal package archive SHA-256 | `b6b7fb58561a792bc237628ba0792e332de418fefe145f13b5ed8201e6d52f58` |
| Candidate upstream `wgpu-hal/src/vulkan/swapchain/native.rs` SHA-256 | `9a0be6f9bda9e160d9fea944d661467bb5017018b7a0612d20d47fd3eb2f26f9` |
| egui candidate family | 0.36.1 |

Primary records:

- [wgpu 30.0.1 release](https://github.com/gfx-rs/wgpu/releases/tag/v30.0.1)
- [wgpu issue 9029](https://github.com/gfx-rs/wgpu/issues/9029)
- [wgpu 30 `Queue::present`](https://docs.rs/wgpu/30.0.1/wgpu/struct.Queue.html#method.present)
- [egui 0.36.0 release](https://github.com/emilk/egui/releases/tag/0.36.0)
- [egui 0.36.1 release](https://github.com/emilk/egui/releases/tag/0.36.1)
- [egui-wgpu 0.36.1 changelog](https://github.com/emilk/egui/blob/0.36.1/crates/egui-wgpu/CHANGELOG.md)

At the audit date, issue 9029 remained open with no linked branch or pull
request. Wgpu-hal 30.0.1 still maps the post-acquire Vulkan fence timeout
through the unexpected device-error path. The project's authenticated
wgpu-hal 29.0.4 patch remains necessary; moving versions requires porting and
re-proving that exact disposition before application migration.

Egui-wgpu 0.36 uses wgpu 30, so renderer and UI families must move together.
Egui 0.35/0.36 also changes font shaping, drag/input, focus, and layout
behavior; 0.36.1 corrects a `Sense::drag` regression. Compilation cannot
prove the native editor's pointer, focus, IME, layout, or DPI behavior.

## Exact source and policy map

| Seat | Current owner | Required campaign movement |
| --- | --- | --- |
| Dependency selection | `Cargo.toml`, `Cargo.lock` | Move the coherent wgpu family to 30.0.1 and egui/epaint integrations to 0.36.1; reject split families |
| Timeout patch | `third_party/wgpu-hal-29.0.4/`, `third_party/wgpu-hal-29.0.4.vendor.json` | Rebase the sole timeout arm onto authenticated wgpu-hal 30.0.1 bytes before application migration |
| Vendor proof | `scripts/verify-vendored-wgpu-hal.py` | Update archive, tree, file, and patch identities while preserving seeded-drift rejection |
| Release/SBOM policy | release scripts and `.github/workflows/{ci,release-trust}.yml` | Update exact vendor identities, structural assertions, receipts, and remeasured SBOM facts together |
| Renderer migration | `src/main.rs`, `src/renderer/`, export and GPU support modules | Apply wgpu 30 API changes without weakening error handling |
| Native UI | `src/main.rs`, `src/patch/editor.rs`, `src/transform_gizmo.rs` | Port the egui family and physically re-prove interaction behavior |

The exact audit snapshot found three `SurfaceConfiguration` constructors that
need explicit color-space handling, seven presentation calls moving to
`Queue::present`, and 34 mapped-range reads across 22 files becoming fallible.
`src/renderer/stage_map.rs` also needs the optional vertex-buffer-layout form.
Those counts are a planning pin; a future topic must rerun the census.

## Deterministic and operator boundary

Deterministic evidence can prove coherent dependency resolution, compilation,
shader and source tests, mapped-readback error propagation, vendor-tree
authenticity, dependency policy, SBOM truth, the six-command gate, and
exact-head hosted CI.

It cannot prove:

- Windows Vulkan timeout recovery on AMD and Intel Arc hardware;
- DX12, Linux Vulkan, and macOS Metal acquisition/presentation behavior;
- resize, minimize/restore, borderless-fullscreen, multi-monitor, and
  device-loss behavior; or
- native pointer capture, transform-gizmo drags, focus transfer, keyboard
  navigation, IME composition, font/layout behavior, and fractional DPI.

Those are physical operator seats and remain mandatory before integration.
The hosted GPU-ignored tests cannot replace them.

## Precise reopening gate

Reopen only when one campaign can do all of the following:

1. Reconfirm exact current wgpu/egui releases, source identities, licenses,
   issue-9029 state, and lock coherence.
2. Authenticate wgpu-hal source and port exactly the timeout disposition
   before changing application code.
3. Rerun the source census and handle every fallible mapped-range result
   without `unwrap`, stale-data reuse, or weakened failure reporting.
4. Pass vendor verification and its hostile self-test, dependency
   audit/deny/SBOM checks, the exact six-command gate, ignored GPU goldens on
   a deterministic software adapter where applicable, and exact-head CI.
5. Produce physical receipts for Windows Vulkan on AMD and Intel Arc, Windows
   DX12, Linux Vulkan, and macOS Metal, including surface lifecycle and the
   issue-9029 timeout path.
6. Produce native UI receipts at representative DPI scales for pointer/drag,
   focus, keyboard, IME, popup/layout, editor visibility, and gizmo ownership.
7. Integrate only if every physical seat passes. Any timeout-induced device
   loss, backend/input regression, or unbounded workaround is a STOP.

## Repository and protected-artifact boundary

This closeout changes tracked evidence only. It does not alter manifests,
locks, source, workflows, vendor bytes, generated capability records, or
release artifacts.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `.da-vinci-canon-pre-refinement-backup-20260822.zip` | 66,225 | `494b63ad0bd96cfb1c7f20a37ad574075a26dd289c14f9f24e71ffe48ab1eea4` |
| `4K_Nature_Cinematography_recorded_with_Nikon_D5300.webm.1080p.vp9.webm` | 56,984,527 | `ee1cfc47671617f8bdf8031dd19cb00f9359e4bf47bdddc7f1fca9df13d034a0` |
| `Black_swan_(Cygnus_atratus).webm.1080p.vp9.webm` | 60,528,641 | `2b51dda28643af61a163d8b3457fd5885c596c51632576a53a6c4c06722630a4` |

They were not modified, copied, renamed, or staged. `videos/audit.mp4` remains
absent and must not be minted as substitute evidence.

## Closing fields

- Disposition: **EVIDENCE-BACKED STOP**
- Topic evidence commit: **`52f2745`**
- Topic receipt commit: **`ff978ed4d46058226114cf8180252716a90ca4f7`**
- Integration commit on `feat/web-control-panel`:
  **`05c8d6cd399843236ea393e15f41a74d4b793913`**
- Exact-commit CI: **OBSERVED PASS** —
  [run 33061058445](https://github.com/Spacejunk-io/collide-o-scope/actions/runs/33061058445)
  passed at exact head `05c8d6cd399843236ea393e15f41a74d4b793913`:
  dependency policy in 31 seconds, Linux 24.04 in 522 seconds, macOS 15 in
  587 seconds, and Windows VS 2022 in 1,023 seconds
- CI-form six-command gate: **OBSERVED PASS** — 2,148 tests passed,
  163 ignored external/physical seats, six benches green, clippy clean
- Dependency/source mutation: **NONE**
- Physical GPU/UI matrix: **NOT RUN — REQUIRED TO REOPEN**
- Protected-root and `videos/audit.mp4` recheck: **OBSERVED PASS**

## Deliberate non-claims

This is not a wgpu 30 compatibility receipt, an egui 0.36 interaction
receipt, a claim that either upstream is generally defective, or a permanent
rejection. It does not claim hosted CI represents a physical adapter. It does
not authorize dropping the timeout patch, increasing GPU/RAM ceilings,
weakening readback failures, or changing the supported backend claims.
