# Precision and scale

Milestone 6 is grounded in measurement and bounded contracts, not feature claims. The current implementation can identify when a proxy may help, encode and serve a content-addressed FFV1/Matroska proxy behind that measurement, execute and account for the Advanced precision path, validate a data-only Study, and describe evidence required by external capabilities. It does not add a hardware decoder, zero-copy path, Syphon, NDI, capture backend, or mesh-warp renderer.

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

### Bounded decode and audio inputs

The plan's ordering clause — an FFV1/Matroska worker only after defining bounded decode/audio inputs — is satisfied by `plan_proxy_input` in `src/proxy.rs`, the one function that decides what a proxy encode may consume. The worker, its artifact validator, and any receipt writer must all answer from it; a second statement of these laws beside it would be a predicate waiting to drift. Every law below is owned by `PROXY_ALGORITHM_VERSION`; changing any of them requires bumping that version, which provably changes every cache key because the version is hashed into the key. No artifact has ever been produced, so version 1 could still be given its meaning without invalidating anything — that window closes the day the worker lands.

**Video.** The proxy carries the `streams().best(Type::Video)` stream — byte-for-byte the selection every live decode path performs (open, reopen, dimension probe, keyframe index) — because a proxy is a decode substitute and must carry exactly the stream live decode would read. A source with no video stream is refused. `Original` scale preserves exact source dimensions; `Half` and `Quarter` floor-divide and round down to even with a floor of 2, so every scaled artifact is legal in chroma-subsampled pixel formats, and the source's decoded pixel format is preserved rather than converted, keeping the even law uniform. `Source` frame rate preserves every frame and its timestamps — the request decoder is timestamp-driven, so variable frame rate passes through faithfully; a fixed rate resamples to exactly the authored constant rate by duplicate/drop.

**Audio.** `include_audio` is the proxy's own policy, deliberately a third policy beside the two the program already carries: export selects the first ordered audio stream, starts it at zero at 1×, and pads/trims to program duration; analysis audio decodes once under bounded limits and samples a circular window. The proxy does neither. With `include_audio: true` the artifact carries the source's first ordered audio stream — exactly the `a:0` stream export's `-map 1:a:0` would select from the original — as a bit-exact stream copy: no re-encode, no resample, no gain, no timing edit, carried whole. A stream longer or shorter than the video is not padded or trimmed, because those are consumption-time policies and baking them into the artifact would change what downstream consumers decode. Streams beyond the first are not carried. A source with no audio stream yields an artifact with no audio track as the defined result, not an error, and the plan records which of the two causes produced an audio-less artifact so a receipt can never conflate them. The probe type deliberately has no audio-duration field: carried-whole means no law consumes one.

**Bounds.** Refusals are typed and ordered: probe consistency, the 64-stream container cap, video-stream presence, the absolute 16,384 px edge, unknown duration, then the one-hour source cap. The encode deadline is `120 + 2 × ceil(duration)` seconds, computed once at admission and never suspended — the thumbnail-helper law — with its maximum derived from the base, factor, and source cap rather than restated. Concurrency is one encode process-wide. Source admission itself — Safe/Expert pixel, byte, and device bounds plus any Expert reservation — is deliberately not re-derived: it is answered by `MediaSafetyPolicy::plan`, the single existing predicate, and the worker must hold that plan for the encode's lifetime exactly as every other media helper does. Staging disk is charged through the existing cache preflight's simultaneous retained-plus-staged accounting.

### The cache worker

`src/proxy_worker.rs` executes the contract. The Y key requests an encode for the selected layer's verified content identity; the single worker thread re-fingerprints the source (a post-verification byte change is refused, never encoded under a stale digest), probes and plans through `plan_proxy_input`, holds a `MediaSafetyPolicy` reservation for the encode's lifetime, and babysits one ffmpeg child under the plan's absolute deadline, a staging-size kill at the per-artifact cap, bounded captured output, and a caller-owned cancel flag. The staged artifact's decoded identity is validated before publication — container, codec, geometry, stream layout, and a decoded first frame — and its exact bytes are then sealed by a SHA-256 sidecar published *after* the artifact, so a crash between the two renames leaves an unsealed artifact that recovery removes rather than serves. Consumption re-hashes the artifact against its seal before every adoption, so corruption anywhere in the file is refused and discarded, not just corruption a first-frame decode would notice; the job's own cache-hit path performs the same check, so a corrupt artifact can never be reported as already cached. Eviction follows the pure preflight's `(last-used ordinal, key)` order and returns a path-free receipt. The directory is the index — a scan rebuilds it, so there is no metadata file to corrupt — and last-used ordinals are session-local with the key order breaking ties deterministically.

Patch load consults the cache for content-referenced video sources: a validated artifact backs the decoder while the layer keeps the original's identity, so a proxy can never enter a patch, an export, Dice, or a Morph — export's digest-gated hint rejects the artifact path and re-resolves the original by content. The HUD layer status reports the whole lifecycle (requested, running, ready, refused, active), and once a proxy is active it reports the measured decode p95 beside the p95 recorded when the encode was requested — the decoder A/B, honest about being a session-local before/after rather than a controlled experiment.

Three edges stay open, stated rather than implied: adoption happens at patch (re)apply, not by hot-swapping a live decoder; only sources with a verified `cos-sha256` identity can be proxied, because the key is content-addressed; and the browser panel has no proxy surface — request and status are native. The hosted CI FFmpeg build on Unix carries `--disable-programs`, so the end-to-end encode fixtures are opt-in like the effects audit, and hosted three-platform CI proves the CLI-free cache half: the commit law, crash recovery, seals, eviction, and refusals.

**Operator observation, 2026-08-18.** The complete loop was run by hand on the development host (AMD Radeon RX 6950 XT, Windows 11, FFmpeg 8.1.2): a content-referenced piece generated from `audit.mp4`, a Y-key encode under default settings (Half scale, source timing, audio carried), and a patch reload to adopt. The HUD reported `proxy active (bfc1add0…): decode p95 7489 us vs 62692 us before` — an 8.4× decode improvement, from roughly four times over the 16.7 ms frame budget to comfortably inside it — with the key prefix matching the published artifact (`bfc1add0….mkv`, 965,036 bytes, plus its 64-byte seal) and nothing else in the cache. This is one clip on one host, and the A/B is the session-local before/after the HUD is documented to be, not a controlled benchmark; it is recorded because it is the first time the recommendation, the encode, the sealed publication, the adoption, and the measured gain were observed end to end by an operator rather than a test.

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

`probe_capability_evidence` is the one production source of evidence, and `scale_capability_decision` the one predicate built on it; before it landed, the only constructors of `CapabilityEvidence` were test fixtures typing literals. The probe answers `platform_supported` from the compile target (the platform API a backend would integrate against exists; Syphon is macOS-only by definition) and reports every other field honestly false: no backend is integrated anywhere in this tree, no authorization store exists, no interoperability receipt exists, and no venue requirement has been recorded. The probe therefore moves nothing to `Available` — a test pins each capability's exact deferred reason per platform, so an accidental flip cannot ship silently, and only NDI's reasons name a purchase or an authorization; hardware decode, zero-copy, and capture stop at `BackendNotIntegrated`, which is engineering, not an external gate.

The chain has a live consumer: the decoder telemetry's `hardware_decode_active` / `zero_copy_active` claims — published into every proxy playback observation and the HUD status line — are now derived through the evaluator instead of typed as literals. A capability that is not `Available` cannot be active, so the claims are theorems today and start telling the truth the moment a backend lands, with no edit at that seam.
