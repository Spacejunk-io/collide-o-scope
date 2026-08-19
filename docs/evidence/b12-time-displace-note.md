# B12 — Time-displace maps

Tranche B12 of the enrichment plan: slit-scan's single angled ramp becomes a
closed instrument vocabulary. `TimeDisplaceMap` is
`Ramp | Brightness | Radial | TbcRamp | Sweep` with permanent codes 0–4, and
`slit_interp` toggles linear interpolation between adjacent ring layers —
off is the exact banded prior path. The map vocabulary is derived from BENDR
(MIT, © 2026 Steve Blythe); every law is a rewrite.

## The five laws

- **Ramp** (default, exact): the existing
  `clamp(dot(uv - 0.5, slit_direction) + 0.5)` angle path, byte for byte.
- **Brightness**: the picture times itself — the coordinate is the current
  sample's alpha-covered Rec.709 luma, so bright things lag dark ones.
- **Radial**: aspect-correct distance from the centre with reach 1.6 — time
  pushed out from the centre.
- **TbcRamp**: a sawtooth over each 8-scanline group
  (`fract(uv.y · height / 8)`) — exactly what a time-base corrector does when
  it fails. The 8-line period is fixed law, not an authored control.
- **Sweep**: a wrapped horizontal ramp travelling on the 30 Hz reference
  clock, one full crossing per 600 reference ticks (20 seconds — BENDR's
  map-drift rate of 0.05 cycles/second expressed on our clock). The phase
  derives from the same accumulated `total_reference_ticks` the rig's noise
  epoch uses, so Program Freeze holds it and export replays it structurally.

Depth continues to clamp against the valid-history counter exactly as
History Key does; ring depth stays 24.

## Design decisions worth recording

- **`temporal.wgsl` is untouched.** A non-default map or the interpolation
  toggle selects the bounded additive originals pipeline
  (`originals_shader_active`), the same seam Loom/Atlas/Garden use, so the
  frozen legacy shader SHA (`07e5cde0…`) did not move and needed no re-pin.
  The authored default (Ramp + floor) keeps running the frozen legacy shader
  byte for byte.
- **Zero new uniform bytes.** The map code and interpolation flag ride the
  two reserved `loom_geometry` lanes and the sweep phase rides a reserved
  `atlas_values` lane of the existing 128-byte originals block. The sweep
  lane is populated only when Sweep is authored, so a default patch's
  uniform bytes never vary with the tick counter.
- **The interpolation path reuses `history_age_sample`.** Both slit blocks
  now route their history reads through the existing age helpers (whose age
  0 is the virtual current image), so the unwritten-history guard is shared
  rather than duplicated, and the ledger delta is at most one extra history
  load per pixel. The legacy prefix's inline `textureSample` count in the
  shader-contract test moved 12 → 11 — one inline read replaced by the
  helper, none added.
- **The routed-Garden pre-pass predicate widened.** The Advanced host's
  pre-Garden originals predicate (`loom || atlas`) now also answers to the
  slit lanes while slit-scan is active, mirroring the plan's own activity
  predicate.
- **Both discrete laws close without modulation addresses.** Map and toggle
  follow the B3 discrete-law precedent: patch (skip-serialized at default),
  wire (`slit_map` token / `slit_interp` boolean on the ordinary coalescible
  `set_temporal`, validated in both validators), snapshot (additive fields),
  Morph (endpoint recall at the midpoint), Dice/generator (legacy temporal
  untouched), no modulatable address.

## Measurements on this host

Adapter: AMD Radeon RX 6950 XT / Vulkan (driver 26.7.1), Windows 11.

- `render_time_displace_pipeline` (opt-in, GPU + ffmpeg + `videos/audit.mp4`):
  the swept, interpolated displacement renders; its `_ramp` twin decodes
  differently (the map reaches the pixels); its `_repeat` twin decodes
  identically (the reference-tick sweep clock replays deterministically).
- Cross-build exactness A/B: `renders/audit_feedback_rig.mp4` rendered on the
  pre-B12 build and on this build is decoded-frame identical (framemd5)
  through the real export path — the default temporal program did not move.
- The M6 precision receipt re-ran; the pinned shader-bundle digest is
  unchanged (B12 does not touch its four shaders) and all six output SHAs
  are unchanged. The tracked receipt diff is confined to the source-manifest
  digest, timestamps, toolchain, and wall-clock lanes.

## Hosted proof

`cargo test --locked --all-targets -- --test-threads=1` green, including the
closed code table, the analytic per-map fixtures, the deterministic sweep
phase, the unwritten-history depth-clamp sweep over every validity count,
the plan-level shader-selection fixture with the reserved-lane assignments,
and the patch/Morph/wire closure fixtures.

## Explicitly not claimed

- No cross-adapter portability claim beyond hosted three-platform CI.
- The analytic map fixtures run on the CPU reference; the GPU claims are the
  labeled export case and the unchanged M6 output SHAs.
- TbcRamp is resolution-dependent by design (it is per physical scanline);
  a 720p live preview and a 1080p export legitimately band differently.
- Live Dice continues not to touch legacy temporal; the generator mutates no
  B12 state (both fields are discrete laws).
