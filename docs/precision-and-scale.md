# Precision and scale

Milestone 6 is grounded in measurement and bounded contracts, not feature claims. The current implementation can identify when a proxy may help, execute and account for the Advanced precision path, validate a data-only Study, and describe evidence required by external capabilities. It does not add a proxy encoder, hardware decoder, zero-copy path, Syphon, NDI, capture backend, or mesh-warp renderer.

## Settled precision law

The compatibility path remains `LegacyCompat8`. The settled Advanced path is `AdvancedWorking16HistoryCompat8`:

- Advanced working and accumulation surfaces are RGBA16Float at exactly 8 bytes per pixel.
- Advanced temporal history and feedback remain Compat8 at exactly 4 bytes per pixel.
- `ExperimentalFull16History` is an evaluation-only resource candidate. It is not an authored mode or an implemented renderer path.
- Passing resource preflight is not a free-VRAM measurement. Portable `wgpu` does not expose one truthful cross-backend free-VRAM budget, so the plan reports that fact explicitly.

### Exact 1080p resource fixture

The production minimum proof uses 1,920 × 1,080 = 2,073,600 pixels, eight RGBA16Float working surfaces, and twenty-five Compat8 temporal surfaces. The temporal count is the complete physical allocation: 24 clean-history layers plus one recursive-feedback image. The evaluation-only Full-16 candidate below upgrades all 25 temporal surfaces, including feedback. The frozen transfer fixture adds 4,096 staging bytes and 8,192 readback bytes.

| Ledger item | Settled Advanced | Full-16-temporal candidate |
| --- | ---: | ---: |
| 8 working surfaces | 132,710,400 bytes (126.562500 MiB) | 132,710,400 bytes (126.562500 MiB) |
| 25 temporal surfaces | 207,360,000 bytes (197.753906 MiB) | 414,720,000 bytes (395.507812 MiB) |
| GPU surfaces | 340,070,400 bytes | 547,430,400 bytes |
| Frozen staging + readback fixture | 12,288 bytes | 12,288 bytes |
| Exact fixture total | 340,082,688 bytes (324.328125 MiB) | 547,442,688 bytes (522.082031 MiB) |
| Exact increase | — | 207,360,000 bytes (197.753906 MiB) |

Eight is the minimum executor topology, not a universal constant: accepted N-1 Program history, rack/image taps, and other planned surfaces increase the creative allocation explicitly. `CompositionAllocationSnapshot` reports accepted creative and motion bytes; selective NTSC reports two RGBA8 scratch textures plus its current staging capacity only after allocation; composition staging and readback report their actual capacities. `RuntimeResourceLedger::reconcile` uses checked arithmetic across creative, motion, NTSC, staging, and readback, requires planned creative bytes to equal the physical allocation snapshot, and has an exact proof that a cap one byte below the computed total is rejected.

Advanced premultiplied bilinear lookups are four explicit shader texture loads. Descriptor admission charges all four operations while retaining separate frozen limits of 32 logical lookups per rack and 1,024 per frame; explicit shader-operation limits are 128 and 4,096. An eight-node worst-case Advanced rack is accepted at exactly 32 logical lookups and 128 shader operations. A frame at exactly 1,024 logical lookups is accepted and 1,025 is rejected; 86 LegacyCanonical-only racks are also rejected at 1,032. The accounting neither disguises four loads as one sample nor grants Legacy/mixed plans four times the former logical budget.

### Objective precision fixture

`measure_precision` compares bounded finite linear-RGBA sample sequences against the same reference. It reports channel RMSE, maximum absolute error, boundary-clamp events, reference gradient events, and gradients retained with the correct direction. `ArtisticGainAssessment` compares those facts and the exact resource delta; its verdict is deliberately “measured gain,” “tradeoff,” “no measured gain,” or “regression,” never a subjective declaration of artistic success.

The small CPU unit proof uses reference RGB levels `0.1`, `0.10002`, and `0.4`. Its baseline uses `0`, `0`, and `0.5`; the candidate reproduces the reference exactly.

| Objective fact | Baseline | Exact candidate |
| --- | ---: | ---: |
| RGBA channel RMSE | approximately 0.0866083143 | 0 |
| Maximum absolute error | 0.10002 | 0 |
| Clamped RGB-channel events | 6 | 0 |
| Reference gradient events | 2 | 2 |
| Retained gradient events | 1 | 2 |

That synthetic fixture remains a planner/unit-law proof. The representative local GPU receipt runs the production effects and temporal shaders at 192×108 on an AMD Radeon RX 6950 XT through Vulkan. It compares an independent f32 reference with real Compat8 output, Advanced RGBA16F working output, and Advanced deterministic Compat8 presentation; no candidate output is copied into the reference.

| Workload/output | RMSE | Maximum error | Clamp events | Gradients retained |
| --- | ---: | ---: | ---: | ---: |
| Still Compat8 | 0.0009368063 | 0.0048468411 | 8 | 11,942 / 16,210 |
| Still Advanced working RGBA16F | 0.0000469949 | 0.0003821552 | 6 | 16,208 / 16,210 |
| Still Advanced presented Compat8 | 0.0006848643 | 0.0042063594 | 40 | 12,523 / 16,210 |
| Temporal Compat8 | 0.0005016257 | 0.0037881434 | 0 | 10,931 / 11,949 |
| Temporal Advanced working RGBA16F | 0.0000903414 | 0.0030384660 | 0 | 11,949 / 11,949 |
| Temporal Advanced presented Compat8 | 0.0006938302 | 0.0065565109 | 0 | 8,606 / 11,949 |

The working-path result is a measured precision gain: RMSE falls for both still and active-feedback workloads, and the temporal working path retains every measured gradient. Final audience presentation has a mixed but measurable result. Pointwise ordered-dither RMSE and gradient direction are worse than Compat8 for the temporal fixture, because the metric treats the intentional spatial code distribution as noise. Over complete 8×8 dither cells, Advanced reduces spatial-mean RMSE from 0.0002552071 to 0.0000636980 for still and from 0.0000704233 to 0.0000596264 for temporal; all 260 still and 311 temporal block gradients retain the correct direction. Compat8 clean history and feedback use a separate no-dither conversion pipeline, so only the final audience conversion receives the pattern and temporal memory does not accumulate it.

The physical-GPU dither gate proves that an encoded code value of 100.25 produces exactly 48 code-100 and 16 code-101 cells over one 8×8 tile, with mean 100.25 and identical repeated frames. It also proves that filtering an opaque-red texel beside transparent hidden green creates partial coverage without a green fringe. These objective results support a bounded spatial-precision gain; they are not a subjective declaration that every image is artistically better.

The receipt records fixture/output SHA-256 values and its Cargo.lock-plus-source manifest in [`m6-precision-gpu-receipt.json`](evidence/m6-precision-gpu-receipt.json). It also records one-shot, per-frame-wait wall observations. These are deterministic-fixture smoke observations only: Advanced includes its required presentation pass, the scopes are asymmetric, and there are no warmup distributions or repeated statistics from which to infer renderer throughput. A local Windows/Vulkan receipt alone does not close the cross-platform boundary: the exact published SHA must pass hosted Linux, macOS, and Windows jobs with durable URLs.

## Content-addressed proxy assessment

A proxy cache key is derived only from:

- the already verified source SHA-256 digest decoded to its 32 canonical bytes;
- the authoritative source byte length; and
- fixed, versioned proxy settings.

Absolute and relative paths are not accepted by the key or assessment records. Renaming or relocating identical bytes therefore preserves the key, while changing source bytes, byte length, scale, frame-rate policy, audio policy, schema version, or algorithm version changes it. The current declarative format vocabulary contains lossless FFV1 in Matroska; that is an open cache-plan choice, not a claim that an encoder has been integrated or is installed.

Playback assessment consumes bounded observations such as sampled frames, visible-layer count, frame budget, decode/upload/frame-age p95 values, drops, and queue pressure. Fewer than 60 frames yields `MeasurementRequired`. A recommendation identifies its objective reasons; it does not silently replace authoritative original media.

The pure cache preflight:

- enforces hard entry, per-artifact, and total-byte caps with checked arithmetic;
- deterministically evicts by `(last-used ordinal, cache key)`;
- counts the retained artifact and the complete staged replacement simultaneously;
- keeps an existing artifact readable until publication; and
- requires create-new temporary output, temporary-file sync, atomic replacement, and parent-directory sync.

No filesystem mutation occurs in this module. A future cache worker must execute that commit law and verify its published artifact.

## Data-only Study ABI

Study schema 1 / ABI 1.0 is closed, typed data with these hard boundaries:

- at most 1 MiB per JSON or YAML document;
- at most 256 SSA instructions, 64 registers, and 16 declared capabilities;
- bounded metadata and license strings;
- fixed instructions for current color, bounded history, motion, audio bands, beat phase, deterministic random input, finite constants, arithmetic, mix, clamp, hue rotation, and one final color output;
- definition-before-use, single assignment, exact value types, one final output, and a unique sorted capability list exactly matching instruction use; and
- unknown fields, versions, operations, oversized lists, non-finite values, and inconsistent capabilities rejected atomically.

The authority record is permanently false for native code, shader-source injection, filesystem access, network access, process launch, device access, and host mutation. A Study cannot enlarge its authority by declaring a capability; capabilities only name fixed read-only creative inputs.

Every Study carries the explicit `StudyDataOnlyDoesNotLicenseHost` publication boundary. Its license notice covers the Study data only. It does not grant, replace, characterize, or imply a license for upstream portions of Collide-O-Scope. This is a technical trust and publication boundary, not a marketplace promise or legal conclusion.

## Evidence-gated capability decisions

The scale capability evaluator starts from evidence, not platform assumptions. With no evidence, every requested capability is deferred.

| Capability | Evidence required before availability |
| --- | --- |
| Hardware decode | Supported platform, integrated backend, and interoperability proof |
| Zero-copy decode | Supported platform, integrated backend, and interoperability proof |
| Syphon input/output | Supported platform, integrated backend, and interoperability proof |
| NDI input/output | Explicit SDK/license authorization, explicit network-policy authorization, supported platform, integrated backend, and interoperability proof |
| Capture input | Supported platform, integrated backend, and interoperability proof |
| Bounded mesh warp | A demonstrated venue requirement, supported platform, integrated backend, and interoperability proof |

An integrated backend without proof yields `EvaluationRequired`; missing platform/backend/policy evidence yields a typed `Deferred` reason. This prevents a schema, menu item, or compile flag from being reported as a working external-video or venue feature.
