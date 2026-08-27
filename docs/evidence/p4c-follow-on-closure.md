# P4c follow-on truth and gate closure

Date prepared: 2026-08-27
Topic: `feat/p4c-follow-on-closure`
Integration base: `e81bddfd6ee823b6247b08f24f5c60cb5de2a011`
Status: **IMPLEMENTED PATH RETAINED; DEFAULT FLIP AND HDR REMAIN GATED**

This follow-on executes the deterministic build directives left by the P4c
handover without converting an unrun physical or performance gate into a
feature claim. The production capability is the authored, persisted
`metadata_managed` path for progressive YUV420P8. `legacy_rgba` remains the
default and every unadmitted frame retains the packed fallback.

## Deterministic closure

- The executable capability registry now has the stable
  `metadata_managed_planar_delivery` row on Windows, macOS, and Linux.
  Browser authoring, live program, offline export, and backend surfaces are
  implemented. Its limitations explicitly preserve the legacy default,
  progressive-YUV420P8-only admission, and the unimplemented HDR output
  surface.
- Production decoder code now calls `planar_delivery_decision`. The historical
  public `prototype_delivery_decision` name remains as a compatibility wrapper,
  and the historical `PrototypePlanar` enum variant remains public
  source-compatibility vocabulary;
  comments no longer falsely call the integrated path evaluation-only.
- The promoted converter WGSL now lives under `src/shaders/` and is consumed
  with `include_str!`, so the existing production shader-bundle identity covers
  the exact program the P4c upload path executes.
- The Phase-A JSON receipt is immutable. Its evidence note now transcribes its
  exact stored p50/p95/p99 values, including the fact that both recorded upload
  p99 values improved. A current opt-in fixture writes only
  `target/p4c-planar-gpu-followup-receipt.json` and cannot overwrite historical
  evidence.
- The integration note now records the exact candidate and Phase-B merge/CI
  receipts and distinguishes delivered items 10/11/12/14 from gated item 13.
- The integrated total-frame fixture times the production decoder harvest,
  layer upload/conversion, full offscreen composite, queue submission, and
  completion fence for paired legacy/managed 720p and 1080p sources. It writes
  raw samples and environment identity to an untracked receipt. The default
  can flip only if that release-only fixture records managed p99 no worse than
  paired legacy at both resolutions and the operator then decides to flip it.
- The 10-bit/HDR output-surface ruling is explicit: a format-only
  `Rgb10a2Unorm` change is not HDR. P010/HDR promotion waits for an API that can
  select and report color space, retained-precision/tone-map/export work, and
  named physical backend/display receipts.

## Immutable Phase-A measurement

The tracked `p4c-planar-gpu-candidate-receipt.json` remains byte-for-byte
historical evidence for candidate commit
`2b7dfb55f3cea0f3df61c583293a35816379bacf`. Its AMD Radeon RX 6950 XT /
Vulkan 26.8.1 result measured 240 frames per source:

| Seam | 720p packed → planar p50 / p95 / p99 (µs) | 1080p packed → planar p50 / p95 / p99 (µs) |
| --- | --- | --- |
| Delivery | 648.0 / 726.2 / 1,015.6 → 197.2 / 245.8 / 318.8 | 1,469.1 / 1,644.1 / 1,831.0 → 447.7 / 515.6 / 781.7 |
| Upload | 505.6 / 603.4 / 771.9 → 358.9 / 488.0 / 757.0 | 969.3 / 1,058.9 / 1,148.7 → 653.8 / 763.5 / 918.3 |

Those seam results authorized integration. They are not an integrated
total-frame receipt and do not authorize the default flip by themselves.

## Integrated total-frame performance decision

The release-only production fixture ran at exact implementation commit
`bcdb99cf93d76cad1536273994f4714e3f84e316` on the named Phase-A adapter
(AMD Radeon RX 6950 XT, Vulkan, driver 26.8.1) with Rust 1.98.0 and FFmpeg
9.0.1. It warmed 300 accepted frames per policy/source, then collected five
paired AB/BA runs over 600 aggregate seconds while timing decoder harvest,
production layer upload/conversion, full Exact composite, temporal and opaque
output, queue submission, and the completion fence. The ignored raw receipt is
33,587,500 bytes with SHA-256
`4fffec11688836c59b5e9d7b05b34bdd85854af1b5041c7f74be73e16aab8c59`.

| Source | Legacy total-frame p50 / p95 / p99 (ms) | Managed total-frame p50 / p95 / p99 (ms) | Managed p99 delta | Paired-run p99 decisions |
| --- | --- | --- | --- | --- |
| 720p | 2.4838 / 3.3530 / 6.6311 | 2.3640 / 3.0451 / 6.2583 | -0.3728 ms | fail / fail / fail / pass / pass |
| 1080p | 4.9042 / 6.7275 / 15.0076 | 3.9890 / 5.3364 / 13.0950 | -1.9126 ms | pass / fail / pass / pass / pass |

The aggregate sample counts were 55,190 legacy / 60,673 managed at 720p and
28,095 legacy / 34,258 managed at 1080p. In chronological pair order, the
managed-minus-legacy p99 deltas were +0.0637, +0.0269, +2.1857, -3.3786, and
-0.1951 ms at 720p, then -2.2629, +3.3081, -0.2336, -7.6710, and -3.2537 ms
at 1080p.

The aggregate managed p99 improved at both resolutions, but three of five
720p pairs and one of five 1080p pairs regressed. The fixture therefore passed
as an execution and evidence capture, while its strict default-flip decision
was negative (`default_gate.passed=false`; `all_sources_passed=false`). The
standing directive forbids weakening a keep gate after seeing its result.
`legacy_rgba` consequently remains the default and no automatic selection was
added. Reopening requires a fresh prespecified campaign that explains and
controls the paired variance; this receipt remains evidence of the present
negative decision, not a favorable-average promotion claim.

## Closing fields

- Deterministic implementation commit: **PASS** —
  `bcdb99cf93d76cad1536273994f4714e3f84e316`
- CI-form six-command gate: **PASS** at exact topic
  `f1f3eea5dd05377940dc7dbc16284a255c235d57` — 2,149 tests passed,
  164 explicit external/physical fixtures ignored, six benches green, and
  warnings-as-errors clippy clean
- Integrated total-frame release fixture: **PASS (EXECUTION); NEGATIVE
  DEFAULT-FLIP DECISION** — 600 aggregate seconds; ignored receipt 33,587,500
  bytes; SHA-256
  `4fffec11688836c59b5e9d7b05b34bdd85854af1b5041c7f74be73e16aab8c59`
- Topic integration commit on `feat/web-control-panel`: **PASS** —
  `2c4dd7b0767d16e31f5b36237985abfd85c3b906`
- Exact-commit CI: **PASS**, run `33068097719` — dependency 34 s,
  Linux 8m51s, macOS 10m42s, Windows 15m29s
- Capability registry regeneration/check: **PASS**
- Phase-A JSON identity recheck: **PASS** — 4,505 bytes, SHA-256
  `fc682e51a549f33b1b70ed684b575eb8da65d989e69cc648be569c9e9e3e082e`;
  Git blob identity matches `HEAD` at
  `5395bd448d1d7931ad4224da3357595fedc48238`
- Default flip / auto-selection: **NOT AUTHORIZED**
- Production P010/NV12 admission: **NOT IMPLEMENTED**
- HDR/tone-map/10-bit output: **DEFERRED — API + PHYSICAL GATE**

## Protected boundary

The three protected root artifacts are not inputs to this work and must remain
unmodified. `videos/audit.mp4` is absent and must not be minted. The current
measurement writers emit only under ignored `target/`; no opt-in run may
rewrite the immutable Phase-A receipt.

## Non-claims

This closure does not call 8-bit sRGB HDR, claim NV12/P010 production
admission, claim total-frame or photon-time performance before its exact
fixture runs, claim one adapter proves every platform, or authorize a default
change without the prescribed receipt and explicit operator decision.
