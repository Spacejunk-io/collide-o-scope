# P4c — the planar-delivery GPU candidate: evidence note

The stop receipt ([`p4c-planar-delivery-stop-receipt.md`](p4c-planar-delivery-stop-receipt.md))
named its own reopen condition: "a dedicated renderer branch and the audit's
720p/1080p two-source CPU/GPU equality plus p95/p99 fixture." This tranche is
that reopening, in the `hw_decode`/full-16 shape: **measurement-only**. It
builds the GPU conversion twin, runs the prescribed equality and percentile
fixtures on the receipt host, and documents the result. The candidate remains
**evaluation-only** — no decoder selects planar delivery, no upload path
changed, no patch surface exists, and whether to pay the integration is a
further product decision this measurement now informs.

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

**The shader is a module constant, not a bundle member.** `build.rs` hashes
`src/shaders/` into the production shader-bundle identity; an evaluation-only
shader does not belong in that identity, so it lives in
`src/video/planar_gpu.rs` and is validated where it runs, at pipeline
creation inside the opt-in fixtures.

**No production consumer, audited.** The module carries the S10a
measurement-only discipline, and
`no_production_module_consumes_the_planar_gpu_prototype` pins it as a source
audit: a reference from any other module is a test failure, so promotion
cannot happen by accident.

## The measurement

Tracked receipt: [`p4c-planar-gpu-candidate-receipt.json`](p4c-planar-gpu-candidate-receipt.json),
regenerated in place by the opt-in fixture (the S2-receipt law: a changed
receipt after an opt-in run is a new measurement on new hardware — commit
it). Adapter: AMD Radeon RX 6950 XT / Vulkan (driver 26.8.1), release
profile — the fixture refuses to run under debug, because debug timings of
Rust loops against optimized C are not evidence. Sources: two generated
H.264 yuv420p clips (720p and 1080p testsrc2 at 30 fps) carrying complete
declared tv/bt709/bt709/left metadata; the **real decoder's frozen
descriptor** drives admission through `prototype_delivery_decision` and both
conversions, so the fixture cannot invent color truth. 240 measured frames
per source; upload timings take 60 warm-up plus 240 fenced iterations.

| Claim | 720p | 1080p | Audit floor |
|---|---:|---:|---|
| Staging bytes/frame, packed → planar | 3,686,400 → 1,382,400 | 8,294,400 → 3,110,400 | ≥ 50% reduction — **62.5% both** |
| CPU/GPU equality on real decoded frames | max Δ 1 code | max Δ 1 code | declared tolerance 1 |
| Delivery p95 (swscale+repack → plane copy) | 718.7 µs → 219.0 µs (**−69.5%**) | 1,671.1 µs → 503.0 µs (**−69.9%**) | improvement required |
| Upload p95 (packed write → plane writes + conversion pass) | 621.7 µs → 536.9 µs (**−13.6%**) | 1,111.7 µs → 814.5 µs (**−26.7%**) | improvement required |

One honest tail: the 720p planar upload p99 (1,330.5 µs) exceeded the packed
p99 (833.9 µs) in this run while its p50/p95 improved; the 1080p tail
improved at every percentile. The audit's promotion clause — "without
worsening total frame p99" — is a property of the **integrated** two-source
matrix, which this candidate deliberately does not claim; the receipt records
the seam percentiles and nothing more.

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
| Measurement-only discipline | no production consumer | **Covered, hosted, as a source audit.** |
| CPU/GPU equality | synthetic battery + real decoded frames | **Covered, opt-in GPU, run on this host.** Battery plus the candidate fixture's three spot frames per source. |
| Percentiles | 720p/1080p two-source delivery and upload p50/p95/p99 | **Covered, opt-in GPU + ffmpeg CLI, release profile, run on this host.** Recorded, never asserted — a truthful negative would still be a valid receipt, exactly as it was for D3D11VA. |
| Legacy byte identity | the packed path unmoved | **By absence.** No production module changed behavior; the only non-test edit outside the new module is contract visibility in `planar.rs`. |

## What the receipt authorizes — and does not

The measured result clears the reopen gate decisively at the delivery seam
(~70% p95 reduction is the swscale conversion cost itself) and meaningfully
at the upload seam. Integration — a decoder that selects planar under the
additive `metadata_managed` policy, pooled plane staging, ledger accounting,
reverse-cache planar bytes, live/export parity, and the integrated
total-frame p99 non-regression proof — is the audit's P4c items 10–14 and
remains its own tranche, taken only if the operator elects it after reading
this receipt. If a future integrated measurement fails the total-frame
clause, the stop receipt's disposal instruction stands: retain P4a/P4b and
leave the prototype unused.
