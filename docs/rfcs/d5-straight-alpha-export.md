# RFC D5 — explicitly ordered straight-alpha plate and fill/key export

Status: **exact artifact contract and transactional publishers implemented;
application action/live acquisition deferred at the integration stop gate**.

## Frozen seam

The stable seam name is `pre_opaque_straight_alpha_v1`. It is the slot-0
`RGBA8_UNORM_SRGB` program image after layer/local/master composition,
Temporal, the selective dry overlay when applicable, and display effects, but
before `opaque_output.wgsl` multiplies RGB by coverage, composites over black,
and forces alpha to one. Both the live renderer and offline renderer resolve
the same slot through an explicitly named helper.

The seam carries straight, not premultiplied, RGB. RGB beneath zero coverage is
meaningful and must be preserved in a straight-alpha plate. A format or effect
change cannot redefine this name; it requires a new seam/version.

## Opt-in artifacts

The public `alpha_export` contract admits three transactional PNG generation
forms and one lossless video form:

- straight RGBA PNG sequence;
- paired opaque fill/key PNG sequences;
- straight RGBA plus paired fill/key PNG sequences;
- FFV1 v3 in a Matroska container, stored as planar `gbrap` and required to
  decode byte-exactly back to the packed RGBA8 input.

PNG and FFV1 publishers accept frames only in strict zero-based order, enforce
exact raster byte length and declared frame count, and bound dimensions, frame
bytes, frame count, and frame rate. Frames are written into a cryptographically
named sibling directory. A receipt is written and synced with the artifacts,
then the complete directory generation is published atomically without
replacement. Cancellation, sequence failure, encoding failure, and Drop remove
the unpublished generation; no fill can become visible without its key.

The paired fill is the straight RGB decoded through the pinned sRGB-to-linear
Q0.16 table, multiplied by alpha with integer rounding, then encoded through
pinned integer thresholds over opaque black. The key is `RGB = source alpha`,
`alpha = 255`. The straight plate is the only artifact that retains hidden RGB;
fill/key is an audience interchange pair and does not claim hidden-RGB
round-trip reconstruction.

Every generation carries `alpha-export-receipt.json`, schema 1, with the seam,
dimensions, rational frame rate, artifact/storage type, source-byte SHA-256,
file patterns, premultiplication law, color law, key law, and effect law.

## Effect and ordinary-output law

Codec-Mosh repurposes the existing opaque readback alpha as an influence
channel, and final-program VHS is a CPU replacement after the opaque boundary.
Neither has a proven straight-alpha propagation law. An alpha plan naming
either effect is therefore refused before any output directory is staged. The
implementation never synthesizes alpha one and calls it a key, and it never
silently labels a pre-effect matte as a post-effect plate.

The ordinary MP4 action/configuration, FFmpeg arguments, `libx264`, `yuv420p`,
frame order, and opaque shader pass are unchanged. The sole offline production
edit names the same existing slot-0 view when constructing the existing opaque
bind group; the live edit does the same. Alpha publishers are opt-in library
objects and cannot be reached by the current MP4 action.

## Promotion stop

The existing bounded offline staging-buffer readback is retained behind a
named, unused acquisition wrapper. Wiring it into a new action would require a
new output-format schema, progress/cancellation ownership, a dedicated bounded
in-flight readback plan, and performance evidence. `main.rs` and the action
schema were concurrently owned, and no exact live-recorder encoder/action seam
or P1 readback performance campaign was available. Those steps are therefore
deferred rather than widening or changing ordinary MP4.

The exact implementation and test receipt is
`docs/evidence/d5-straight-alpha-export.md`.
