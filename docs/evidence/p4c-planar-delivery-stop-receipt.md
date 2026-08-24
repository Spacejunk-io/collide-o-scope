# P4c planar delivery prototype — evidence and stop receipt

Date: 2026-08-24<br>
Baseline named by the audit: `v1.6.0` / `000411d`

## Outcome

P4c stops at a bounded software contract. `src/video/planar.rs` proves that an
immutable packed/planar image enum can represent YUV420P, NV12, and P010 while
keeping the existing frame metadata and codec-motion product on the same owned
frame. It does **not** replace `DecodedVideoFrame`, select planar delivery in a
decoder, add textures/pipelines, alter an upload, or change an authored patch.
The current production byte path therefore remains the exact P4a packed-RGBA
path.

The prototype is retained as a reference seam; planar delivery is not promoted.
P4a immutable packed ownership and P4b source descriptor truth stand
independently.

## Frozen contract and limits

- Formats: exactly `Yuv420p8`, `Nv12`, and little-endian high-bit `P010Le`.
- Plane count: fixed at two or three; no caller-controlled plane vector.
- Dimension edge: the existing absolute media edge, 16,384 pixels.
- Physical bytes: 128 MiB maximum for one planar frame.
- Aggregate prototype budget: configurable from 1 byte through 256 MiB.
- CPU RGBA reference output: separately capped at 256 MiB.
- Storage: one tightly packed `Vec<u8>` containing every plane, frozen behind
  one `Arc` inner and one aggregate byte lease. Clones retain the same identity
  and charge no physical bytes. Padding is validated but not retained.
- Validation order: format/shape, every borrowed row span, frame cap, aggregate
  reservation, then a fallible exact reservation. A failed construction leaves
  the aggregate budget unchanged.
- `DecodedDeliveryFrame` moves the existing `FrameMetadata` and
  `CodecMotionProduct` beside either image variant. Packed wrap/unwrap preserves
  payload identity and bytes without copying or retagging. A planar frame cannot
  be laundered back to the legacy type without an explicit conversion.
- `PlanarDeliveryPolicy::default()` and missing serde fields both select
  `legacy_rgba`. `metadata_managed` is opt-in contract vocabulary only; it is
  not added to patch persistence in this stopped prototype.

## Color reference

The CPU oracle implements declared full/limited code-range normalization,
BT.601 (`SMPTE 170M`/`BT.470BG`), BT.709, and non-constant-luminance BT.2020
matrix coefficients, six declared 4:2:0 chroma locations with bilinear edge
clamping, 8-bit samples, and P010 ten-bit samples. Alpha is always opaque.

Its result is explicitly labeled
`source_encoded_sdr_rgba8_no_gamut_mapping`: it performs matrix/range/chroma
reconstruction only. It does not claim transfer linearization, gamut mapping,
tone mapping, or display-ready sRGB. Unspecified metadata is rejected. PQ and
HLG return `HdrToneMapRequired`; known interlaced sources fall back to packed
RGBA. No 8-bit surface is labeled HDR.

## Deterministic proof cases

The six `video::planar::tests` cases cover:

1. exact 4:2:0 plane geometry, a frame-cap rejection, the absolute edge, and a
   12-byte frame refused by an 11-byte aggregate budget;
2. padded NV12 input packed into one 12-byte immutable allocation, one charge
   across an `Arc` clone, cap refusal while retained, and release to zero;
3. BT.601 and BT.709 limited saturated bars, BT.709 full black/white/mid ramp,
   left-versus-center chroma edges, BT.2020 P010 limited 10-bit ramp and color;
4. exact legacy packed bytes, payload identity, generation/PTS identity, and
   codec-motion identity across wrap/unwrap plus backward-compatible policy;
5. the same generation/PTS/codec-motion identity retained on a planar frame,
   with legacy conversion refused rather than silently copying or retagging;
6. opt-in admission, ordinary P010 SDR contract admission, and explicit HDR,
   interlace, and incomplete-metadata stop/fallback laws.

## Transport arithmetic (not a performance claim)

Tightly packed 8-bit 4:2:0 contains 1.5 bytes per even-dimension pixel versus
4 bytes for RGBA8, a 62.5% byte reduction. At 1920×1080 this contract describes
3,110,400 plane bytes versus 8,294,400 packed bytes. P010 contains 3 bytes per
pixel and therefore saves 25% versus RGBA8 before row alignment. These are
layout arithmetic only. No decoder-delivery, upload, or total-frame percentile
was measured.

## Stop gate

The audit requires a measured GPU path, at least 50% common-8-bit staging-byte
reduction, CPU/GPU equality, and an improvement to decode-delivery or upload p95
without worsening total-frame p99. None can be established without adding and
measuring planar textures, bind groups, first-pass conversion, pooled staging,
patch migration, and an HDR/tone-map policy. Adding those now would enter the
actively changing renderer and could silently alter legacy pixels.

Accordingly:

- no GPU path was integrated;
- no hardware-decode dependency was introduced;
- no HDR output or planar default was promoted;
- no planar reverse-cache accounting claim is made;
- no performance keep claim is made.

Reopen P4c only with a dedicated renderer branch and the audit's 720p/1080p
two-source CPU/GPU equality plus p95/p99 fixture. If that receipt fails, delete
or leave unused the prototype and retain P4a/P4b only.
