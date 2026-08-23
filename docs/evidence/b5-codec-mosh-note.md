# B5 / v1.5 — codec mosh: evidence note

Tranche B5 of the enrichment plan. Nothing here is a shader imitating a
codec: a real mpeg4 encoder and decoder are wired back to back in-process
(`ffmpeg-next` library, never the CLI) with the bitstream broken between
them, so the artefacts are the decoder's own. The original fault laws are
derived from BENDR (MIT, © 2026 Steve Blythe), whose codec stage
(`p42_capture.js`) settled their semantics. v1.5 preserves that codec and adds
a bounded motion wake around it.

## What landed

- **Law module `src/codec_mosh.rs`.** The pure half (parameter sanitize,
  the bitrate/resync laws, the per-chunk fault decisions, the bounded chunk
  ring) has no FFmpeg dependency and is the reference the round trip
  follows. The engine half owns two codec contexts (`threads = 1` set
  before open — the per-host determinism lever), the two software scalers
  between RGBA and YUV420P, and the wire-breaking loop.
- **Eight original BENDR continuous controls plus one discrete law**, transcribed from
  BENDR with their exact semantics: `amount` (dry/wet in the stored sRGB
  bytes; the wake law at BENDR's own 0.003 deadband), `key_removal`
  (per-key dice, NOT rate-scaled; the first key after any reset always
  passes; forced resync keys face the dice, so `key_removal = 1` never
  recovers), `hold` (1–5 extra re-applications under fresh monotonic
  timestamps), `drop` (starves the decoder; the chunk still enters the ring
  and its own hold/shuffle dice are suppressed), `shuffle` (re-inject at
  least six chunks stale, ring gate > 10), `rate` (multiplies hold/drop/
  shuffle only), `bitrate_starve` (`4 Mbps × 0.02^q`, ±25% hysteresis,
  reconfigure forces re-acquire), `resync`
  (`max(2, round((1−r)·300)+2)` frames; zero never recovers), and
  `recycle` (CLEAN vs RECYCLED encoder feed — worker-local in our
  replacement model, so it lands as a cheap discrete law).
- **Three additive v1.5 motion-wake controls** are continuous unit values.
  **Motion Wipe** is patch field `wipe` and control/modulation target
  `mosh_wipe`: zero keeps the historical uniform blend, while one makes moving
  macroblocks and their retained wake the complete reveal. **Vector Smear** is
  `smear` / `mosh_smear`: it pulls the damaged sample backward along the
  decoder's forward displacement, using one clamped nearest read fused into the
  existing blend. **Motion Trail** is `trail` / `mosh_trail`: it retains the
  observed wake on the 30 Hz reference clock, from current-observation-only at
  zero to held-until-reset at one.
- **Exact default compatibility.** All three new controls default to zero. At
  that combined default the pre-v1.5 path is exact: no motion-side-data request,
  luma analysis, wake allocation, displaced read, or changed pixel arithmetic.
  Trail cannot allocate invisible history by itself; analysis wakes only when
  Motion Wipe or Vector Smear can expose it.
- **A bounded motion surface.** Analysis stays on the already capped codec
  image, whose longest edge is at most 640 pixels. Native 16×16 MPEG-4
  macroblocks cap even a square maximum image at 40×40 / 1,600 cells. The
  full-resolution blend loop already exists; v1.5 folds its matte lookup and at
  most one smear read into that loop, with no extra full-frame pass, GPU
  surface, readback, or worker.
- **Per-layer Mosh Send without per-layer codecs.** Every layer adds the
  finite unit field `mosh_send` (patch / `set_layer_param`) and appended
  modulation suffix `layerN_mosh_send`, default one. The GPU composes each
  wet layer's evaluated base-effect coverage into one programme-space R8
  control field, then carries it in the alpha byte of the existing opaque
  RGBA readback. The CPU blend consumes alpha only when the tagged frame says
  the field is present and restores output alpha to 255. RGB, codec count,
  readback count, queue depth, and worker latency remain unchanged.
- **The live seat** keeps the final-program-VHS worker shape — one
  `MoshWorker`, `sync_channel(1)` both ways, one in flight,
  drop-new-while-busy counted as healthy `skipped`, terminal on failure
  with the error named in the additive `AppSnapshot::codec_mosh` block,
  lazily constructed on the first armed frame. `MoshFrameMetadata` travels
  with the pixels on the readback tag. When VHS is also armed, that same worker
  hop runs **Mosh then VHS** — one admission and one asynchronous frame of live
  latency. Results are generation-checked before replacement; blackout remains
  the final absolute operation.
- **One audience order everywhere:** creative composition / Temporal → Codec
  Mosh → final-program VHS → blackout. Master bypass is resolved during
  creative composition and does not create a selective or per-layer VHS path.
- **Export honesty**: the identical engine runs synchronously per frame
  before final-program VHS, with the fault ordinal on the same paused-aware
  reference frame used live. A missing mpeg4 pair or a given-up decoder is an
  actionable error, never a silent bypass. Motion-sidecar schema 6 introduced
  `codec_mosh`; schema **7** appends `wipe`, `smear`, and `trail`, accepted-frame
  min/max values for every continuous recipe control, both observed Recycle
  states, and bounded per-layer authored/observed Mosh Send plus wet-partition
  admission. Encode dimensions and the `mpeg4/avcodec-<version>` identity stay
  explicit. The section appears only when an accepted frame ran the round trip.
- **Persistence and automation.** The three controls persist in the typed patch
  block, interpolate continuously through Morph, have bounded modulation
  targets, and mutate in independent generator-v14 domains. Layer Mosh Send
  separately persists, morphs, accepts the bounded appended layer modulation
  address, and is exposed in native/browser layer state.

## Deliberate deviations from BENDR, named

1. **Deterministic fault dice.** BENDR uses `Math.random()` and disclaims
   reproducibility (it hard-disables the stage offline). Our export
   contract demands a replayable fault stream, so every decision draws the
   shared avalanche hash in the fixed "CMSH" domain with independent lanes,
   keyed by the master `random_seed`, the 30 Hz reference ordinal, and the
   packet index. Pause holds the fault stream still on both paths.
2. **GOP 600, not unbounded.** The mpeg4 encoder clamps any larger GOP to
   600, so we author the ceiling explicitly. A volunteered key is just
   another key chunk facing the removal dice, so the "never recovers" law
   survives at the chunk level, where BENDR's law lives too.
3. **A byte cap on the chunk ring** (≤ 90 chunks AND ≤ 8 MiB, FIFO
   eviction on either bound): a bounded surface must be bounded in bytes.
4. **BENDR's `moshRate` id collision** (its flow TRASH RATE overwrites the
   codec EVENT RATE) is deliberately not reproduced. Codec Event Rate remains
   `mosh_rate`; v1.5 uses the distinct stable targets `mosh_wipe`,
   `mosh_smear`, and `mosh_trail`.
5. **Bounded motion rather than audience-resolution optical flow.** The wake is
   measured at native codec-macroblock scale and reused by the existing blend.
   This creates object-led wipe, smear, and trail behavior without multiplying
   full-frame analysis across pixels or layers.

## The honesty boundary

**Repeatability is claimed per host**: two renders on one machine with
`threads = 1` decode to equal framemd5 sequences (the `_repeat`
assertion). **Cross-machine bit-identity is explicitly not claimed** — a
different libavcodec build may encode differently — and the sidecar's
encoder identity is the record of why. The stage's documented live costs
are one to two frames of audience latency and the occasional deliberate
re-acquisition (bitrate sweeps and decoder deaths snap the picture back);
both are behavior, not defects.

The motion wake and codec history remain program-level. Layer Mosh Send is a
spatial influence field over that one result, not an assertion of separate
codec state or pixel provenance. Its coverage follows evaluated base layer
transform/effects before support-changing Advanced racks, groups, master
effects, and Temporal; those later operations may carry pixels outside the
field. Exact **Bypass Temporal FX** layers are excluded because they never
enter Codec Mosh.

## Ledger

| Item | Bound |
|---|---:|
| Chunk ring | ≤ 90 delta chunks AND ≤ 8 MiB, FIFO eviction |
| Codec contexts | 2 (one encoder, one decoder), plus 2 scalers |
| Encode resolution | Longest edge ≤ 640, aspect preserved, even, each edge ≥ 64 |
| Motion analysis grid | 16×16 codec pixels per cell; ≤ 40×40 / 1,600 cells |
| Motion-analysis admission | Only while `wipe` or `smear` is visibly armed |
| Output shaping | Existing blend loop plus at most one clamped smear read |
| In-flight jobs | 1 (drop-new-while-busy) |
| Decoder feeds per frame | ≤ 16, decode drains ≤ 8 per chunk |
| Staging | one full-frame RGBA job vec, shared with final-program VHS |
| Layer-send GPU resources | Lazy only below full send: 2×R8 + 1×RGBA8 = 6 nominal texel bytes/output pixel (backend alignment/metadata excluded) |
| Layer-send resource churn | 0 per-layer buffers/bind groups after source-cache warmup; dynamic uniform arenas grow only with admitted layer topology |
| Layer-send GPU work | Default/unkeyed/unwarped layer: one R8 composite; support-changing layer: one base-FX coverage + one R8 composite; then one byte copy + alpha-only pack |
| New readbacks or worker hops | 0 |
| New wire actions | 0 (twelve `mosh_*` params ride `set_temporal`; `mosh_send` rides `set_layer_param`) |

## Proof

The hosted proof surface covers scalar sanitation, the amount-only wake law,
the exact all-zero compatibility path, bitrate hysteresis, resync, landscape
and portrait dimension caps, deterministic independent fault lanes, key
bootstrap, delta gates, ring bounds, worker admission, patch round-trip,
continuous Morph, the eleven continuous modulation addresses with no address
for `recycle`, the twelve-parameter validator vocabulary, deterministic
generator-v14 mutation, bounded 16×16 wake analysis, and schema-7 authored plus
accepted-frame sidecar fields.

The opt-in codec/export proofs require the linked FFmpeg mpeg4 pair, a GPU, and
the audit source. They compare dry, moshed, and same-host repeat renders; verify
that real decoded pixels change; retain amount-zero as a no-touch bypass on a
warm engine; and assert that the moshed job records the schema-7 recipe,
observed bounds, and layer sends while the exact-bypass job omits `codec_mosh`.

## What is deliberately not here

- No cross-machine determinism claim, no sidecar record of bitstream bytes.
- No codec choice control: mpeg4 is the law (the right artefacts, native
  to every FFmpeg build); a vocabulary would be a new discrete law with
  its own sidecar and determinism story.
- No per-layer Codec Mosh context, wake, trail buffer, codec round trip, or
  readback. Mosh Send is one bounded programme-space control matte over the
  shared result.
- No chained Mosh/VHS workers. The combined hop runs Mosh→VHS once and its
  counters describe that one bounded admission.
- Live/export pixel parity is not claimed for the stage (live is
  best-effort drop-new, export is the authoritative synchronous rendition —
  the final-program VHS precedent exactly).
- Blackout is not hidden inside either effect; it remains last and absolute.
