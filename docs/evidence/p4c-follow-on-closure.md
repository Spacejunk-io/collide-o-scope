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

## Closing fields

- Deterministic implementation commit: **PENDING**
- CI-form six-command gate: **PENDING**
- Integrated total-frame release fixture: **PENDING EXECUTION**
- Topic integration commit on `feat/web-control-panel`: **PENDING**
- Exact-commit CI: **PENDING**
- Capability registry regeneration/check: **PENDING**
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
