# B3 — The feedback rig, completed

Tranche B3 of the enrichment plan: video feedback grows from three controls
(`feedback`, `fb_zoom`, `fb_rotate`) to the full rig — per-tick offset, the
two discrete reflections, in-loop hue/saturation/per-channel gain, chromatic
displacement, the blur+sharpen activator–inhibitor pair, the
`Clamp | Soft | Wrap | Fold` waveshaper with drive/pivot, threshold decay,
deterministic loop noise, the `fb_edge` boundary law on the frozen program-wide
numbering, and the auto-level servo with its defeat switch.

## Design decisions worth recording

- **The legacy 64-byte uniform stays frozen.** The rig rides a new 96-byte
  `TemporalRigGpuUniforms` third fixed binding (`@group(1) @binding(2)`) in
  both temporal shaders and both renderer paths, exactly as the originals took
  a second binding. The byte golden of the legacy block is untouched.
- **The servo is deterministic by construction.** A measured-mean servo (the
  obvious readback design) would give live and export different loop dynamics,
  because live readbacks land N frames late while offline ones are
  synchronous — the export contract forbids that divergence. The landed servo
  is a per-pixel compressive auto-level (Reinhard-style about unity, mixed by
  the tick fraction). Defeat wins over engage; defeated, a hot loop runs to
  white or black and stays, which is the B14 philosophy landed early.
- **Garden's carrier law is untouched.** An active rig takes its own
  transformed read; the shared carrier read narrows to
  `garden || (feedback && !rig_active)`, which is the original predicate at
  rig identity.
- **Rate law.** Linear scaling for rate-like terms, exponentiation for
  multiplicative terms (the established `feedback`/`fb_zoom` law), and a
  clamped tick-fraction mix toward identity for the nonlinear stage — exact at
  the 30 Hz reference, saturating below it.

## Measurements on this host

Adapter: AMD Radeon RX 6950 XT / Vulkan (driver 26.7.1), Windows 11.

- **Byte exactness of the identity rig, three ways:**
  1. `gpu_temporal_originals_topology_interpolation_atlas_and_startup_goldens`
     passes unchanged — the pinned startup pixel goldens did not move.
  2. The M6 precision receipt re-ran under the re-pinned shader-bundle digest
     (`d3db27d4…`) and **all six output SHAs are unchanged**; the tracked
     receipt diff is confined to the bundle/source digests, timestamps,
     toolchain, and wall-clock lanes.
  3. `renders/audit_motion_flow_shaping.mp4` rendered before and after B3 is
     decoded-frame identical (framemd5) through the real export path.
- `render_feedback_rig_pipeline` (opt-in, GPU + ffmpeg + `videos/audit.mp4`):
  the fully rigged loop renders; its `_unrigged` twin decodes differently
  (the rig reaches the pixels); its `_repeat` twin decodes identically
  (the loop replays deterministically, noise epochs included).
- `advanced_temporal_feedback_filters_hidden_rgb_in_premultiplied_space`
  passes unchanged.

## Hosted proof

`cargo test --locked --all-targets -- --test-threads=1`: 1,342 tests passing,
including the regime fixtures (four-arm lock carrying the analytic retention
powers, detune shear, the reflection two-cycle, servo bound versus monotonic
defeated runaway), the CPU reference edge laws on the frozen boundary
numbering, the rate-law fixtures, uniform-lane laws, and the full
patch/Morph/modulation/wire/generator closure.

## Explicitly not claimed

- No cross-adapter portability claim beyond hosted three-platform CI.
- The regime fixtures run on the CPU reference loop, not on GPU readback; the
  GPU claims are the goldens, the receipt SHAs, and the labeled export cases.
- Live Dice continues not to touch legacy temporal; only the offline generator
  mutates the rig's continuous values.
- The Turing-pattern regime (blur+sharpen under the max-combine loop) is
  reachable but not separately characterized; a dedicated fixture would be a
  follow-up, not a blocker.
