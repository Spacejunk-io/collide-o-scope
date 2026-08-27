# RFC D5 — explicitly ordered straight-alpha plate and fill/key export

Status: **implemented as an offline-export application action; live-recorder
acquisition remains deferred**. The exact artifact contract and transactional
publishers landed first at the integration stop gate; the application action,
the invoked offline readback at the named seam, and the action receipt landed
in the follow-on tranche
(`docs/evidence/d5-alpha-export-action-note.md`).

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
frame order, and opaque shader pass are unchanged. Alpha publishers are
reached by the ordinary MP4 action's explicit opt-in `alpha` field
(`start_export`, closed token vocabulary; omitted legacy clients keep the
exact prior path). When the field is set, the offline frame loop reads the
named seam through the existing bounded staging buffer — one sequential
in-flight readback, after the audience readback, with the established
cancellation law — and stages one atomic `<output>.mp4.alpha` generation that
publishes only after the MP4 and every sidecar. The authored effect refusal
runs before any directory exists, and the frame loop re-checks both effect
laws so a Morph or modulation wake aborts the job instead of publishing a
mislabeled plate.

## Remaining boundary

Live-recorder alpha acquisition remains deferred: the recorder owns no
alpha-capable encoder, and the capability registry carries that surface as
deferred. The keep/stop receipt for the retained core is
`docs/evidence/d5-straight-alpha-export.md`; the application-action receipt is
`docs/evidence/d5-alpha-export-action-note.md`.
