# B5 — codec mosh: evidence note

Tranche B5 of the enrichment plan. Nothing here is a shader imitating a
codec: a real mpeg4 encoder and decoder are wired back to back in-process
(`ffmpeg-next` library, never the CLI) with the bitstream broken between
them, so the artefacts are the decoder's own. The laws are derived from
BENDR (MIT, © 2026 Steve Blythe), whose codec stage (`p42_capture.js`)
settled every control's semantics. This note records what landed, the
deliberate deviations, and the evidence run on this host.

## What landed

- **Law module `src/codec_mosh.rs`.** The pure half (parameter sanitize,
  the bitrate/resync laws, the per-chunk fault decisions, the bounded chunk
  ring) has no FFmpeg dependency and is the reference the round trip
  follows. The engine half owns two codec contexts (`threads = 1` set
  before open — the per-host determinism lever), the two software scalers
  between RGBA and YUV420P, and the wire-breaking loop.
- **Eight continuous controls plus one discrete law**, transcribed from
  BENDR with their exact semantics: `amount` (dry/wet in the stored sRGB
  bytes; the wake law at BENDR's own 0.003 deadband), `key_removal`
  (per-key dice, NOT rate-scaled; the first key after any reset always
  passes; forced resync keys face the dice, so `key_removal = 1` never
  recovers), `hold` (1–6 extra re-applications under fresh monotonic
  timestamps), `drop` (starves the decoder; the chunk still enters the ring
  and its own hold/shuffle dice are suppressed), `shuffle` (re-inject at
  least six chunks stale, ring gate > 10), `rate` (multiplies hold/drop/
  shuffle only), `bitrate_starve` (`4 Mbps × 0.02^q`, ±25% hysteresis,
  reconfigure forces re-acquire), `resync`
  (`max(2, round((1−r)·300)+2)` frames; zero never recovers), and
  `recycle` (CLEAN vs RECYCLED encoder feed — worker-local in our
  replacement model, so it lands as a cheap discrete law).
- **The live seat**: the global-VHS worker shape verbatim — one
  `MoshWorker`, `sync_channel(1)` both ways, one in flight,
  drop-new-while-busy counted as healthy `skipped`, terminal on failure
  with the error named in the additive `AppSnapshot::codec_mosh` block,
  lazily constructed on the first armed frame. `MoshFrameMetadata` travels
  with the pixels on the readback tag (the NTSC metadata law). On the
  global-VHS path the worker runs the VHS kernel and the round trip **in
  one hop** — one admission, one frame of latency, the exact offline
  ordering — and the NTSC worker is deliberately unfed while the mosh is
  armed. Results are validated by generation AND by the stage still being
  armed; the retained newest frame is re-written into slot 2 downstream of
  the VHS replacement and upstream of blackout, which stays absolute.
- **Every path**: an armed mosh extends `raw_audience_readback_required`
  on the disabled, global, and selective paths alike — slot 2 already
  holds the selective recomposite, and bypass is a *VHS* bypass, not a
  general one, so the mosh treats the finished programme uniformly exactly
  as the display stage does.
- **Export honesty**: the identical engine runs synchronously per frame
  after global NTSC (codec-after-analog; selective frames arrive already
  VHS-treated), with the fault ordinal on the same paused-aware reference
  frame the NTSC phase uses. A missing mpeg4 pair or a given-up decoder is
  an actionable export error, never a silent bypass. The `.motion.json`
  sidecar bumped 5 → **6** (first bump since the Field Collider) with the
  additive `codec_mosh` section: authored recipe, encode dimensions, and
  the `mpeg4/avcodec-<version>` encoder identity — present only when an
  accepted frame actually ran the round trip.

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
   codec EVENT RATE) is deliberately not reproduced: our params are
   unambiguous `mosh_*` names.

## The honesty boundary

**Repeatability is claimed per host**: two renders on one machine with
`threads = 1` decode to equal framemd5 sequences (the `_repeat`
assertion). **Cross-machine bit-identity is explicitly not claimed** — a
different libavcodec build may encode differently — and the sidecar's
encoder identity is the record of why. The stage's documented live costs
are one to two frames of audience latency and the occasional deliberate
re-acquisition (bitrate sweeps and decoder deaths snap the picture back);
both are behavior, not defects.

## Ledger

| Item | Bound |
|---|---:|
| Chunk ring | ≤ 90 delta chunks AND ≤ 8 MiB, FIFO eviction |
| Codec contexts | 2 (one encoder, one decoder), plus 2 scalers |
| Encode resolution | ≤ 640 wide, aspect preserved, even, ≥ 64 |
| In-flight jobs | 1 (drop-new-while-busy) |
| Decoder feeds per frame | ≤ 16, decode drains ≤ 8 per chunk |
| Staging | one full-frame RGBA job vec, the NTSC job shape |
| New GPU surfaces / passes / shaders | 0 (the stage is CPU-side) |
| New wire actions | 0 (nine `mosh_*` params ride `set_temporal`) |

## Proof

Hosted (all pass; the suite is 1448 tests): the eleven `codec_mosh` law
tests (sanitize, wake deadband, bitrate map + hysteresis, resync table,
dimension law, per-lane dice, key bootstrap, delta gates with drop's early
return and the rate exemption, ring bounds + newest-six shield, encoder
identity, worker admission ladder tolerant of a codec-less host), the
patch round trip with skip-at-default omission, the Morph blend with
midpoint recycle recall, the modulation addresses with the recycle refusal
and frame-copy-only offsets, the nine-param wire vocabulary in both
validators, generator mutation in fresh domains (`GENERATOR_VERSION`
"12"), the extended `raw_audience_readback_required` law, and the
schema-6 sidecar pins.

Opt-in, run on this host (AMD Radeon RX 6950 XT; FFmpeg 8.1.2 shared):

- `mosh_round_trip_is_deterministic_per_host_and_reaches_the_pixels` —
  two runs byte-identical across 12 frames; the moshed output demonstrably
  differs from the dry input; amount zero is a no-touch bypass even on a
  warm engine. **Passed.**
- `render_codec_mosh_pipeline` — the labeled export case through the real
  export path: the `_clean` twin decodes differently, the `_repeat` render
  decodes identically, the moshed job's sidecar carries schema 6 with the
  `mpeg4/avcodec-…` identity, and the clean job omits the section.
  **Passed.**

Exactness A/B: `render_gesture_canvas_displace_donor_pipeline` rendered on
a pinned worktree at the base merge (99d6003) and on this branch —
framemd5 sequences identical excluding `#software` (33 lines). The default
path did not move a pixel.

## What is deliberately not here

- No cross-machine determinism claim, no sidecar record of bitstream bytes.
- No codec choice control: mpeg4 is the law (the right artefacts, native
  to every FFmpeg build); a vocabulary would be a new discrete law with
  its own sidecar and determinism story.
- No chaining latency hiding: when global VHS and the mosh are both armed,
  the combined hop replaces the NTSC worker's feed rather than cascading
  two workers — the NTSC global counters legitimately go quiet while the
  mosh is armed, and the mosh counters carry the admissions.
- Live/export pixel parity is not claimed for the stage (live is
  best-effort drop-new, export is the authoritative synchronous rendition —
  the global-VHS precedent exactly).
