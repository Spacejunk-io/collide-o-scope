# P4c — the planar-delivery GPU candidate: evidence note

The stop receipt ([`p4c-planar-delivery-stop-receipt.md`](p4c-planar-delivery-stop-receipt.md))
named its own reopen condition: "a dedicated renderer branch and the audit's
720p/1080p two-source CPU/GPU equality plus p95/p99 fixture." This note records
that historical reopening, in the `hw_decode`/full-16 shape: a
**measurement-only candidate**. At the measured candidate tree no decoder
selected planar delivery, no upload path changed, and no patch surface
existed. Phase B subsequently integrated the authored opt-in path; its current
truth lives in [`p4c-planar-integration-note.md`](p4c-planar-integration-note.md).

Branch: `feat/p4c-planar-delivery` from `a5f9043` (v1.8.1 mainline,
suite-green on the newest run of every CI suite).

## The design, compressed

**One conversion contract, consumed twice.** The stopped prototype's
`CpuConversionContract` — admission, Kr/Kb, range law, chroma siting — was
made crate-visible, and `planar_convert_uniforms` derives the GPU uniform
record (48 bytes, compile-time asserted) from exactly that derivation. Every
refusal the CPU oracle makes (unspecified metadata, PQ/HLG, unsupported
matrix, descriptor mismatch) is the GPU path's refusal too, structurally: one
table, two consumers — the shared-parse-table law applied to color.

**Integer planes, explicit filtering.** Planes upload into uint textures
(`R8Uint`/`Rg8Uint` for 8-bit, `R16Uint`/`Rg16Uint` for P010), and every read
is a `textureLoad`, so the shader sees the exact stored codes and no hardware
sampler law enters. The WGSL follows `to_rgba8_cpu_reference` expression for
expression: the same explicit bilinear over the chroma plane, the same
limited/full normalization, the same reconstruction. Declared tolerance: one
8-bit code value per channel (`f64` round-half-away versus `f32`
round-nearest at the quantize boundary), alpha exact.

**Shader identity followed promotion.** At the candidate tree the
evaluation-only WGSL was a module constant rather than a production bundle
member. Phase B made the converter a production upload path, and the follow-on
therefore moved those exact bytes to `src/shaders/planar_convert.wgsl` with an
`include_str!` consumer. `build.rs` now includes it in the production
shader-bundle identity; pipeline creation still validates it where it runs.

**No production consumer at the candidate tree, audited.** The candidate
carried the S10a measurement-only discipline and a source audit that rejected
any production reference. Phase B deliberately removed that stop only after
this receipt cleared the reopen gate.

## The measurement

Tracked receipt: [`p4c-planar-gpu-candidate-receipt.json`](p4c-planar-gpu-candidate-receipt.json).
It is immutable evidence for its named candidate commit, branch, dirty-tree
state, and host. The current follow-up fixture writes a distinct untracked
`target/p4c-planar-gpu-followup-receipt.json`; it cannot overwrite this Phase-A
receipt. Adapter: AMD Radeon RX 6950 XT / Vulkan (driver 26.8.1), release
profile — the fixture refuses to run under debug, because debug timings of
Rust loops against optimized C are not evidence. Sources: two generated
H.264 yuv420p clips (720p and 1080p testsrc2 at 30 fps) carrying complete
declared tv/bt709/bt709/left metadata; the **real decoder's frozen
descriptor** drove admission through the then-public
`prototype_delivery_decision` compatibility name and both
conversions, so the fixture cannot invent color truth. 240 measured frames
per source; upload timings take 60 warm-up plus 240 fenced iterations.

| Claim | 720p | 1080p | Audit floor |
|---|---:|---:|---|
| Staging bytes/frame, packed → planar | 3,686,400 → 1,382,400 | 8,294,400 → 3,110,400 | ≥ 50% reduction — **62.5% both** |
| CPU/GPU equality on real decoded frames | max Δ 1 code | max Δ 1 code | declared tolerance 1 |
| Delivery p50 / p95 / p99, packed → planar | 648.0 / 726.2 / 1,015.6 µs → 197.2 / 245.8 / 318.8 µs | 1,469.1 / 1,644.1 / 1,831.0 µs → 447.7 / 515.6 / 781.7 µs | improvement required; p95 **−66.2% / −68.6%** |
| Upload p50 / p95 / p99, packed → planar | 505.6 / 603.4 / 771.9 µs → 358.9 / 488.0 / 757.0 µs | 969.3 / 1,058.9 / 1,148.7 µs → 653.8 / 763.5 / 918.3 µs | improvement required; p95 **−19.1% / −27.9%** |

Both upload p99 values improved in the immutable receipt. That is still not
the audit's promotion clause: "without worsening total frame p99" is a
property of the **integrated** two-source matrix, which this candidate did not
claim. The receipt records delivery and upload seams, not total-frame latency.

The synthetic equality battery
(`gpu_planar_conversion_matches_the_cpu_reference_battery`) covers what real
media cannot: all three formats including NV12 and P010, the 601/709/2020
matrices, limited and full range, all six declared chroma sitings over a
hard chroma edge (with Left and Center proven distinguishable), odd
dimensions, and deterministic noise — every case within one code of the CPU
oracle on the same adapter.

| Surface | Required proof | Status |
|---|---|---|
| Contract unity | GPU uniforms derive from the CPU contract; refusal parity | **Covered, hosted.** `uniforms_are_48_bytes_and_answer_from_the_one_shared_contract`. |
| Staging arithmetic | ≥ 50% for common 8-bit 4:2:0 | **Covered, hosted.** `planar_staging_bytes_meet_the_audit_reduction_floor` (62.5%; P010's 25% recorded as the fidelity case). |
| Measurement-only discipline | no production consumer at the candidate tree | **Covered historically, hosted, as a source audit.** |
| CPU/GPU equality | synthetic battery + real decoded frames | **Covered, opt-in GPU, run on this host.** Battery plus the candidate fixture's three spot frames per source. |
| Percentiles | 720p/1080p two-source delivery and upload p50/p95/p99 | **Covered, opt-in GPU + ffmpeg CLI, release profile, run on this host.** Recorded, never asserted — a truthful negative would still be a valid receipt, exactly as it was for D3D11VA. |
| Legacy byte identity | the packed path unmoved | **By absence.** No production module changed behavior; the only non-test edit outside the new module is contract visibility in `planar.rs`. |

## What the receipt authorizes — and does not

The measured result cleared the reopen gate decisively at the delivery seam
and meaningfully at the upload seam. Phase B then integrated authored
`metadata_managed` delivery, pooled plane staging, ledger accounting,
reverse-cache planar bytes, and live/export parity for progressive YUV420P8.
It did not prove the integrated total-frame p99 clause, flip the legacy
default, admit NV12/P010, or solve the 10-bit/HDR output surface. Those remain
separate gates, and a failed total-frame receipt must preserve the legacy
default.
