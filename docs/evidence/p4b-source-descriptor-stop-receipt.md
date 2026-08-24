# P4b source color/display descriptor and UV stop receipt

Date: 2026-08-24

## Landed contract

- `SourceColorDescriptor` and `SourceDisplayDescriptor` are fixed-size serde
  values. Every fact carries `DescriptorProvenance`; direct container, codec,
  frame, still-header, and still-EXIF declarations are distinct from pixel-
  format derivation, declared-value derivation, and inferred fallback.
- Color truth covers pixel family, component bit depth, range, matrix,
  primaries, transfer, chroma location, and chroma subsampling.
- Display truth covers coded and display raster dimensions, normalized bounded
  rational SAR, validated clean-aperture crop, rotation, mirror, all eight
  normalized EXIF orientations, the exact nine-element FFmpeg display matrix,
  and field order.
- FFmpeg codec declarations are read from `AVCodecParameters`, not from values
  the opened decoder may have inferred. The first decoded frame may fill facts
  that the codec parameters omitted. The result freezes before that frame's
  first software conversion, giving live and offline decoders the same law.
- Still EXIF is parsed under the same decoder limits as the image decode and
  normalized without calling `apply_orientation`; coded dimensions and legacy
  RGBA bytes therefore remain unchanged.
- The existing libswscale context is explicitly configured only when both a
  supported matrix and range are directly declared. Unspecified/derived input
  takes the exact historical `ScalerContext::get(..., BILINEAR)` path without a
  color-details call. Unsupported or rejected explicit requests remain named
  in `SourceConversionPolicy`; a rejected setter rebuilds the historical
  context before any bytes are converted.
- Frozen descriptor and actual policy travel through `VideoDecoder`,
  `ThreadedDecoder`, `DecoderTelemetry`, Layer, bounded Stage Health rows and
  its native HUD, proxy completion receipts, and export sidecar source
  provenance. Export sidecar schema is 9. Stage Health additions use serde
  defaults for old snapshots.

## Hard bounds

- Rational numerator and denominator: each at most 1,000,000 after reduction;
  zero, negative, or over-cap values are Unspecified.
- Display matrix: exactly 9 native-endian `i32` values / 36 bytes; malformed,
  all-zero, or zero-homogeneous matrices are refused.
- Frame-cropping side data: exactly 16 bytes; every sum is checked and the
  aperture must retain at least one coded pixel on each axis.
- Pixel-format component depth: 1 through 32 bits; chroma shifts are clamped to
  0 through 4. No pixel-format name or arbitrary metadata string is retained.
- Stage Health remains capped by its existing layer cap; every new per-layer
  value is fixed-size. Proxy receipts and export sources retain one descriptor
  pair per already-bounded source record.

## Reference and acceptance fixtures

The deterministic tests cover:

- BT.601 limited, BT.709 limited/full, and BT.2020 limited color bars;
- opposite chroma edges and monotonic 8-/10-bit ramps;
- explicit libswscale configuration for 601/709/2020 and both 709 ranges;
- byte-exact historical scaler output for Unspecified policy;
- rational SAR plus clean aperture, with axis-correct SAR inversion on 90/270;
- 0/90/180/270, horizontal/vertical mirrors, transpose/transverse, malformed
  display matrices, and all eight EXIF orientations;
- an actual JPEG EXIF-6 decode proving coded pixels stay 2x1 while display
  truth becomes 1x2;
- serde defaults that never promote inferred fallback to declared;
- an ignored, external-FFmpeg 10-bit BT.709-limited/SAR fixture proving live
  and offline decoders freeze identical descriptors, policy, and pixels.

## Orientation integration stop gate

The existing spatial uniform is authored solely from coded source dimensions
and has a byte/pixel-compatibility identity bypass. Source orientation is not
present in the evaluated frame or rack plan, and composing it inside the
renderer would touch actively changing core GPU/upload ownership while making
the old identity path active. P4b therefore stops before renderer integration.

`SourceUvReference` is the allocation-free, no-CPU-rotation reference law for
clean aperture plus all eight normalized orientations, with exhaustive corner
fixtures. It is intentionally not wired to the renderer until a later audited
change can carry immutable source-display metadata through the evaluated plan,
measure the GPU path, and preserve the exact legacy identity bypass. No claim
is made that rotation/mirror/SAR is currently applied to displayed pixels.
