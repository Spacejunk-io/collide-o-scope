# S12d — the hardware decode backend: evidence note

Gate 4's first tranche, opened by the operator's commission and walked in
the order the S12 prompt set: hardware decode first, zero-copy and capture
later, each its own tranche. The shape is Gate 6's, deliberately:
**evaluation-only**. The backend exists, is measured, and the capability
evaluator now answers `EvaluationRequired(InteroperabilityProof)` on
Windows — the exact progression the capability law has demanded since S7 —
and deliberately nothing more. `Available`, live decode, and zero-copy are
separate tranches the operator decides after reading the receipt.

Branch point: `1d25517` on `feat/proxy-authored-settings` (stacked on the
three prior S12 tranches). Baseline: **1309 passed / 0 failed / 97
ignored**; with this tranche **1310 / 0 / 98** — one hosted refusal test,
the reworked per-platform progression pin, and one opt-in interop probe.

## The design, compressed

**The library path, never the CLI.** `video::hw_decode` opens the best
video stream through the same bounded, cancellation-aware `open_input` the
software decoder uses, creates a D3D11VA device with
`av_hwdevice_ctx_create`, and routes decoding through it with a
`get_format` that picks `AV_PIX_FMT_D3D11` — the same narrow-ffi
discipline as the decoder core's EXPORT_MVS flag, written before decoder
open. Every hardware surface is downloaded with
`av_hwframe_transfer_data`; a frame the codec declined to route through
hardware is counted honestly as a software fallback, never folded into the
hardware count. Non-Windows is one typed refusal
(`PlatformUnsupported`) — the backend is D3D11VA and does not pretend
otherwise.

**The evidence seam finally does what it promised.**
`probe_capability_evidence` now answers `backend_integrated` for hardware
decode from the backend module's own
`hardware_decode_backend_exists_on_this_platform` — the tree change that
integrated the backend is the same change that flipped it, exactly as the
S7 field documentation said it would be. Zero-copy deliberately does not
ride the flag: downloading frames is not a zero-copy path, so it stays
`BackendNotIntegrated`. `decode_activity_claims` needed no edit and
`hardware_decode_active` stays false everywhere, because
`EvaluationRequired` is not `Available` — the claims-are-theorems seam
paying out a second time.

**The wrong comparison was caught by its own fixture.** The first probe
compared RGBA conversions of both paths and failed loudly at a
117-code-value disagreement — which turned out to be swscale's different
`nv12`→RGBA and `yuv420p`→RGBA conversion paths, not the decoders. The
redesigned measurement compares the raw decoded 4:2:0 planes with no
conversion in between, which is the actual decoder-agreement claim, and
asserts byte-exact equality because H.264 decoding is spec-exact — a
deviating driver is a finding the receipt must surface, never a tolerance
to hide behind. The S11 law again: the fixture that breaks a first design
has done more work than ten that pass it.

## The measurement

Tracked receipt: `docs/evidence/hw-decode-interop-receipt.json`,
regenerated in place by the opt-in probe (S2-receipt law). This host
(Windows, AMD Radeon RX 6950 XT, FFmpeg 8.1.2 shared libraries,
`videos/audit.mp4` — H.264 High, 640×360, 72 frames):

| Fact | Measured |
|---|---:|
| Frames decoded through D3D11VA | 72 / 72 |
| Software fallbacks in the hardware session | 0 |
| Max luma delta vs software reference | **0** |
| Max chroma delta vs software reference | **0** |
| Differing samples | 0 / 24,883,200 |

The hardware decoder on this host agrees with the spec-exact software
reference byte for byte across every decoded sample. Wall timings are
recorded as fixture-local smoke observations only — the probe downloads
every frame to system memory, which a live integration would avoid, and at
640×360 that download dominates.

| Surface | Required proof | Status |
|---|---|---|
| Capability progression | `EvaluationRequired` on Windows, `Deferred(BackendNotIntegrated)` elsewhere, nothing `Available` | **Covered, hosted.** `the_production_probe_defers_every_capability_with_its_actionable_reason`, reworked to pin full decisions per platform, plus an explicit not-`Available` assertion for the claims seam. |
| Typed refusals | missing file, absent platform | **Covered, hosted.** `a_missing_file_is_a_typed_open_refusal_or_a_platform_refusal` — both arms, whichever platform runs it. |
| Interoperability | real device, real stream, honest agreement metric | **Covered, opt-in, run on this host.** The probe above; byte-exact across 72 frames. |
| Claims stay honest | `hardware_decode_active` false throughout | **By construction.** Derived through the evaluator; `EvaluationRequired ≠ Available`; no edit at the seam. |
| Production decode | unchanged | **By absence.** No production path constructs a session; the one production-alive item is the `backend_integrated` fact. The only shared-code edit is `open_input`'s visibility (`pub(super)`). |
| Render/export A/B | decoded-`framemd5` parity | **Not applicable, argued.** No render, export, or production decode path changed; the module is probe-only and `open_input` gained visibility, not behavior. |
| CLI boundary | hosted half CLI-free | **By construction.** The probe uses FFmpeg libraries only; hosted tests need no ffmpeg binary, no GPU, no clip. |

What is deliberately not claimed: no live-path integration, no zero-copy,
no cross-adapter portability (one adapter, one driver, one clip — the
receipt names them), no performance claim of any kind, and no export-path
implication — the standing boundary that offline export keeps its
synchronous software decoders unless per-adapter bit-exactness is proven
is written into the constraints, and this receipt is one adapter's first
data point toward that question, not its answer.

Gate on `RUSTUP_TOOLCHAIN=1.97.1`: fmt, both node checks, check, tests,
clippy `-D warnings` — run on the final tree before commit.
