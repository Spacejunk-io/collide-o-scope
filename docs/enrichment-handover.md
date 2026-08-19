# Enrichment handover — for the successor session

The enrichment plan is a sixteen-tranche program (`B1`–`B16`, three waves)
enlarging the instrument's expressive surface, derived by studying BENDR
(MIT, © 2026 Steve Blythe) — a single-file browser circuit-bent video
processor. Every implementation is a rewrite by construction (JS/WebGL2 →
Rust/wgpu 29/WGSL); where an algorithm is transcribed closely enough to be a
"substantial portion" in spirit (the scan processor's beam-energy law, the
melt band), carry an attribution comment naming BENDR and its author. This
does not alter the fork's own `LICENSE` boundary.

This document is now **self-sufficient**: it carries the status board, the
complete specifications for every remaining tranche, the per-tranche working
method, and the frozen contracts a successor will trip over. The operator's
original plan artifact remains the source of record, and the BENDR source
itself (`~/Downloads/bendr-main`, or github.com/clickysteve/bendr) settles
any law the spec leaves ambiguous — consulting it resolved B12's Sweep and
TbcRamp laws and B13's stage ordering in minutes.

Execution order: **Wave 1** B2, B3, B12, B13 — feed machinery we already own
(**complete**). **Wave 2** B1, B4, B8, B16, B5, B7 — new machinery.
**Wave 3** B9, B10, B11, B6, B14, B15 — performance, monitoring, ergonomics.
B15 may interleave anywhere as relief work. Nothing in the plan raises the
selective-VHS budget, moves the media-safety bounds, touches the Study
authority model, or adds a second full-frame history ring; every new surface
is individually charged and one-over-refused per standing law.

## Status

| Tranche | State | Where |
|---|---|---|
| B2 — synthetic + shaped motion fields | **Complete** | PR #30 (merged), PR #31 (merged); `docs/evidence/b2-procedural-motion-fields-note.md` |
| B3 — the feedback rig | **Complete** | PR #32 (merged); `docs/evidence/b3-feedback-rig-note.md` |
| B12 — time-displace maps | **Complete** | PR #33/#34 (merged); `docs/evidence/b12-time-displace-note.md`; CLAUDE.md "B12 time-displace maps" |
| B13 — the small-effects tranche | **Complete** | PR #36; `docs/evidence/b13-small-effects-note.md`; CLAUDE.md "B13 small effects" |
| B14 — failure switches | **Partially landed early** | `servo_defeated` shipped inside B3; the remaining piece is `sync_latched` (spec below) |
| B1, B4, B8, B16, B5, B7, B9, B10, B11, B6, B15 | Open | Specs below |

Each landed tranche documents itself in `CLAUDE.md` and in its evidence note.
Read those before extending any subsystem a tranche touched.

## Wave 2 — new machinery

### B1 — The Scan Processor (dedicated geometry pass) — NEXT

**Why.** The crown jewel, and the one stage that cannot be faked: a Rutt/Etra
scan processor's output is *drawn*, not sampled. Line density — bright caustic
ridges where scanlines bunch, dark gaps where they splay — does not exist in a
fragment shader's world model, which is why every displacement-map imitation
reads as digital. BENDR proves the whole mechanism in ~100 lines of vertex
shader; it is the single largest aesthetic capability we lack.

**Seat.** A Collision Rack node, `NodeKindTag::ScanProcessor`, next free
append-only signature code, `occupies_dedicated_pass = true` (the
Symmetry/Study precedent), executed by a new `renderer/scan_processor.rs`.
This is the tree's **first non-fullscreen-triangle pass**: one instanced
triangle-strip ribbon per scanline, no vertex buffers — position from
`@builtin(vertex_index)` / `@builtin(instance_index)` (the fullscreen-triangle
tradition, extended), carrier fetched in the **vertex stage** via the
established explicit-load bilinear helper (sampler-free, like every dedicated
pass). Accumulation is additive into an `Rgba16Float` transient with alpha
contribution zero (coverage must not stack past unity where lines bunch);
a resolve blend then applies the node's wet/blend law.

**The algorithm, kept whole** (this is the part worth transcribing faithfully,
with attribution): `beamAt(sx, line)` composes, in order — sweep/field
reversal, S-curve, skew, deflection oscillator (locked to a multiple of the
field rate it stands still; detuned it crawls — the crawl is the instrument's
gesture), raster collapse, then *luminance into vertical deflection*, then
tilt/perspective as a photographed 2D deflection, never a scene. The ribbon
normal comes from the central-difference tangent, and the same tangent gives
beam speed: `gain = 2/speed`, because a slower beam deposits more energy per
unit length. That one term is the difference between this and a displacement
map. The pass redraws the whole raster, so it legitimately clears the coverage
matte — the hardware does too.

**Authored state.** `lines` (16–1,080), `samples_per_line` (64–512), amount,
ribbon width, velocity-brightness mix, tilt X/Y, perspective, S-curve, skew,
collapse, reverse H/V (discrete), oscillator amount/freq/lock/lissajous, mono,
hue. Continuous set modulatable; reverses discrete.

**Ledger.** 1 render pass; ≤ `lines × samples × 2` vertices (cap ≈ 1.1 M,
charged as its own named vertex budget — a new ledger class, since no prior
pass owns geometry); 3 vertex-stage carrier fetches per vertex (here/ahead/
back); 1 transient full-frame `Rgba16Float` (8 B/px, charged, freed after
resolve); 1 sampled texture; 0 persistent surfaces. Dedicated-pass plan
re-derived from emitted steps (`SymmetryFieldResourcePlan` shape).

**Proof.** CPU reference for `beamAt` composition order; the density claim
proven on GPU: a hard luma edge must produce a brighter ridge than any
displacement of the same image (this is the fixture that distinguishes the
mechanism from the imitation); default-bypass byte identity; warm-allocation
invariance; labeled export case `render_scan_processor_pipeline`. [closure]

### B4 — Display physics (fields, phosphor, CRT master stage)

**Why.** Everything BENDR renders is *watched through something*. Three
mechanisms matter and are absent: real interlace (two moments in one frame —
the serrated pan edge no drawn comb reproduces, and the swapped-field-order
stutter that is almost impossible to fake), per-primary phosphor persistence
(green wake, blue leading edge — one buffer, three decay constants), and the
mask/beam display models.

**Seat.** A master-scope display stage after Temporal, before opaque resolve —
audience-affecting, therefore export-identical by the shared-executor law.
Three sub-blocks, each defaulting to exact-off:

- **Fields**: hold the previous field (one full-res RGBA8 surface, ~8.3 MB at
  1080p, charged); `Weave | Bob | Blend` closed vocab; field-order swap as the
  documented fault; twitter on high vertical detail; 3:2 judder. Parity
  advances on the 30 Hz reference clock, frame-indexed offline.
- **Phosphor**: one full-res `Rgba16Float` accumulator (16.6 MB at 1080p,
  charged), `max(current, prev * k_rgb)` with three authored decay constants.
  This is a single accumulator in the established feedback shape — explicitly
  not a new history *ring*, so the standing RGBA16F-ring prohibition is not
  touched.
- **Display model**: `Flat | ApertureGrille | SlotMask | ShadowMask |
  LcdStripe | Mono | GreenScreen` closed vocab; beam-profile scanlines that
  widen with brightness; bloom/halation/defocus/sag as continuous params.

**Law.** All three are frame-local evaluated state like spatial transforms —
never topology; blackout remains the absolute final audience operation and the
phosphor accumulator is cleared by it (a blacked-out audience must not retain
a glowing wake). Pause holds; the selective-VHS audience-hold contract is
unaffected because this stage sits downstream of the recomposite.

**Proof.** Field-parity determinism at 24/30/60; phosphor decay constants
against closed form; blackout-clears-accumulator; labeled export case.
[closure]

### B8 — The mixing boundary (melt, faults, key dressing, blend/wipe audit)

**Why.** The melting edge is BENDR's most original algorithm and it maps onto
machinery we already have: every coverage boundary in our program (static key
alpha, cellular gap, group matte) is currently a clean seam. The mechanism:
evaluate the coverage matte at four points a chosen distance out; where they
disagree, the pixel stands on the edge and the disagreement direction is the
edge normal. That yields a band of controlled width with a normal through it —
then drag the incoming picture along the normal, dissolve the stage's own
previous frame back in within the band, and because the band feeds itself the
smear creeps outward instead of washing out.

**Seat.** A per-scope melt block available where a matte exists (group buses
and the master key resolve), plus the dirty-mixer fault stage on buses.
Controls per BENDR: `melt`, `width`, `hold`, `swirl` (drag along the edge
instead of across), `chroma` (colour runs further than luma), `creep` (which
side of the seam melts — a keyed shape can bleed into the background while the
background never eats the shape). At zero the history surface is never touched
and never allocated — the delegation law, matching BENDR's own "costs nothing
off" rule.

**Dirty mixer**: an event clock (`dirt` = firing probability, `rate` = clock),
four fault laws — dropout (line bands cross the crossbar or go to nothing),
cut (whole mix thrown to one input), knock (timebase shoved sideways,
crawling back), noise (switching transient with colour dropping out).
Completely clean between firings; hashed like Shift (band, epoch,
`random_seed`) in a fresh RNG domain so it is export-deterministic.

**Blend/wipe audit**: enumerate the current bus A/B law; adopt the classic
blend family (`Add | Screen | Multiply | Difference | Darken | Exclusion |
Subtract | Overlay | HardLight | SoftLight | VividLight | PinLight | Dodge |
Burn | Divide | WrapAdd | Xor | And | Hue | Saturation | Color | Luminosity`)
as a closed append-only enum with existing laws keeping their indices, and a
wipe-pattern vocabulary (H/V/diag/box/circle/splits/blinds/clock/bars/blocks)
with softness, origin, border colour, and MULTI tiling as bus mix laws.
Key dressing: `key_border` (matte grow + fill from a closed colour table) and
`key_shadow` (offset darkened copy) on static keys.

**Ledger.** Melt: one history surface per armed scope, ≤ 2 armed (the
gesture-canvas cap shape), 8 B/px `Rgba16Float`, charged with one-over
refusal; 4 matte taps + 2 colour taps per pixel in-band. Faults/blends/wipes:
zero surfaces.

**Proof.** Analytic edge-normal fixture (a vertical matte edge yields a
horizontal normal band of exactly `width`); melt-at-zero never-allocates;
dissolve-has-no-boundary-so-nothing-happens (BENDR's own correctness case);
fault clock byte-stable per seed; labeled export case. [closure]

### B16 — Program re-entry

**Why.** BENDR's re-entry (any channel sources the finished programme, one
frame old) turns the whole rig into the feedback loop. Our image routes cover
scope-to-scope; the missing producer is the programme itself. The stability
rule ("whatever it reads is one frame old") is already our law — the
gesture-canvas donor publishes N-1 at the acceptance decision, and
ProgramHistory obeys the same ordering.

**Seat.** `SavedImageSource::ProgramTap` joining the closed route vocabulary
exactly as `GestureCanvas` did: a master-scope singleton producer with no
scope, no ID, no saved position; claims no dependency edge (it is N-1 by
construction, so no same-frame cycle is expressible); resolves Transparent
with a named diagnostic before the first committed frame. One retained
full-frame copy of the pre-blackout opaque audience image, published at the
acceptance decision (blackout stays absolute: the tap must hold the
pre-blackout image or a blackout would leak one stale frame — decide and
test this explicitly). Offline export applies byte-identical ordering.

**Ledger.** One persistent full-frame surface (charged); 0 passes beyond the
copy.

**Proof.** N-1 law fixture; patch-load invalidation; a two-frame
feedback-through-the-full-chain fixture; labeled export case. [closure]

### B5 — Codec Mosh (the real round trip)

**Why.** "Nothing here is a shader imitating a codec, so the artefacts are the
decoder's own." BENDR does this with WebCodecs; we ship FFmpeg's libraries and
already babysit its CLI under deadlines. A real encoder and decoder wired back
to back with the bitstream broken between them — keyframes starved,
differences held/dropped/re-injected out of order, quantizer starved, resync
letting a keyframe through so the picture snaps back and starts falling apart
again.

**Seat.** A master-scope stage backed by a bounded worker in the established
shape (one in-flight composite, drop-new-while-busy, generation/topology/
dimension tags travel with the pixels, stale results rejected by name — the
selective-NTSC law verbatim). Codec via ffmpeg-next library (not CLI):
prefer mpeg4/mpeg2video over x264 — simpler, faster, deterministic with
`threads=1`, and their macroblock artefacts are the right artefacts. The
stage costs one to two frames of audience latency and occasional deliberate
re-acquisition; both are documented behavior, not defects. While paused, no
new batch is submitted and the last materialized image holds (the audience-
hold contract). Failure surfaces through its own status and holds the prior
image — never a silent bypass.

**Controls.** `amount`, `key_starve`, `hold`, `drop`, `shuffle` (re-inject
from a bounded chunk ring seconds back), `bitrate_starve`, `resync`.

**Export honesty.** Offline runs the round trip synchronously per frame with
`threads=1` and pinned codec parameters. The repeatability claim (two renders,
equal framemd5) holds per host; cross-machine bit-identity is explicitly
not claimed — the encoder's identity and version are recorded in the
`.motion.json` sidecar (additive schema bump), the same honesty shape as the
hw-decode receipt.

**Ledger.** Bounded chunk ring (≤ N MiB, charged), two codec contexts, one
staging frame.

**Proof.** Worker lifecycle/backpressure fixtures hosted; end-to-end round
trip opt-in like `effects_audit`; labeled export case. [closure]

### B7 — Generator sources (pattern synth, text page)

**Why.** A generator is the first source with perfect offline
reconstruction — no file identity, no content reference, no Spout black
placeholder: the patch carries everything. BENDR's synth architecture is a
video synthesizer's: coordinates → shape (`Scan | Radial | Spiral | Plasma |
Lissajous | Rings | Starburst | Grid | Tunnel | Cells | Interference |
Polygon`) → oscillator (waveform vocab) → cross-modulation between axes →
wavefolder → comparator → colouriser (`Mono | RgbPhase | HsvSweep | Duotone |
Bands`). Every continuous control is a modulation destination.

**Seat.** Two new `LayerSource` arms. Pattern renders one GPU pass per frame
into the layer texture, clocked by program time (frame-indexed offline — the
law already covers it); it has a clock but no transport, so the panel follows
the BENDR rule: SPD drives the clock, file controls absent. TextPage rasters
on CPU (glyph rendering wants a font stack; bundle 2–3 licensed faces, not 33)
into a bounded RGBA upload re-rendered only when authored state changes —
between edits it costs what a still costs. Both serialize wholly in
`LayerConfig`; `source_path` becomes `synth://` / `text://` sentinels with the
same stability rules as `spout://`.

**Ledger.** Pattern: 1 pass, 0 sampled textures, uniform-driven. Text: bounded
raster (≤ source-admission bytes, same policy gate as stills).

**Proof.** Pattern CPU reference per shape at sample points; live/export
identity; text re-raster-only-on-change; patch round trip with zero file
identity; labeled export cases. [closure]

## Wave 3 — performance, monitoring, ergonomics

### B9 — The performance recorder

**Why.** "It records gestures rather than pixels, so a take built slowly over
an hour can be played back in real time, against completely different
footage." This is the single feature that most changes what the instrument
is — from a surface you play live to one you can compose on. And we have
already built its hard parts once: the gesture track's contract (30 Hz
reference ticks, quantized events, domain-separated SHA-256 checksum, bounded
serde, truncation honesty, patch carriage, export sidecar) is the proven
template.

**Seat.** `performance_track.rs` in the `gesture.rs` shape. An event is
`(tick, param_address, value)` where `param_address` is the existing
compiled control identity (the coalesce key / modulation target id — never a
string parsed at frame rate) and `value` is Q16 normalized against the
control's declared `target_range`. Discrete controls record their closed-vocab
code. Recording captures accepted authored edits at the drain — after
coalescing, so a take stores what the program actually did.

**Playback law — delegation, not reimplementation** (the transform-gizmo
precedent): replay dispatches real actions through
`handle_web_action_inner_with_feedback` with an automation origin, so Morph
ownership transfer, beat latching, and revision guards all apply for free, and
`MutationOrigin::records_manual_history` excludes replay from the undo stack.
Topology actions are deliberately not recordable in v1 — a take records
values, and a take whose stack no longer matches degrades per-address with
named diagnostics rather than retargeting (the stale-ID law). Takes are
patch-carried whole, checksummed; export replays by tick, frame-indexed.

**Proof.** Record/replay determinism (same take, same patch → identical
framemd5); replay-against-different-footage smoke; truncation and
open-recording honesty flags; undo exclusion. [closure]

### B10 — Modulation source expansion

**Why.** The matrix law makes each of these pure extension: a new `ModSource`
enters the same shaping/slew, and every consumer benefits.

**Contents.**

- Envelopes (4): trigger vocab `Pad | AudioOnset | SceneCut | Beat`;
  attack/decay seconds; retrigger law stated. Pads are 6 momentary bend
  sources (keyboard row + MIDI notes + panel buttons) that are themselves
  sources and envelope triggers.
- Chaos / drift / spike generators (deterministic: seeded, frame-indexed
  offline — "random" sources must replay identically in export, which BENDR
  does not guarantee and we must).
- Macros (4): an authored knob that is a source; fan-out is just routes.
- Video-reactive: `video_motion` (aggregate magnitude from the Motion
  lattice at Draft quality, armed on demand), `video_brightness` (mean luma),
  `video_cut` (frame-difference onset). Brightness/cut need one 16×16
  reduction + readback at ~10 Hz through the existing bounded readback
  machinery, slewed like every external source. Live-only zeros in export
  except where derivable frame-indexed (brightness/cut can be evaluated
  offline from the decoded frames — state which, and test parity).

**Proof.** Envelope closed-form decay; deterministic chaos replay; readback
budget unchanged (still ≤ 3 in flight); [closure] for the config sections.

### B11 — The monitoring bay

**Why.** "The difference between 'the picture is doing something odd' and
knowing which part of the model is doing it." Preview-only by our own law.

**Seat.** stage_health's exact shape: a sealed preview permit
(`native_controls_visible` + surface check), low-res readback (≤ 160×90) at
10 Hz only while the panel tab or native pane is visible, zero cost
otherwise. Three instruments: waveform (luma vs. x, graticule at black/white),
vectorscope (U/V cloud with the six colour-bar targets), and PROBE — internal
signals rendered to the preview surface only, never audience: the NTSC/tape
model's per-line state, the modulation matrix's live source values, the melt
band mask, a motion-field visualizer. Web panel gets the same data over the
snapshot at the same gated rate.

**Proof.** Permit refusal on every audience surface; zero-readback-when-
hidden; [closure] n/a beyond snapshot additivity.

### B6 — Block-domain corruption (DCT, pixel sort, avalanche)

**Why.** Three distinct digital-corruption mechanisms, all absent.

- **Block DCT** (rack node): real separable 8-point DCT + inverse, two passes
  one per axis, quantizer with HF penalty, chroma quantized harder — the
  artefacts are a codec's because the math is a codec's. Continuous:
  `quantize`, `hf_penalty`, `chroma_crush`.
- **Pixel sort** (rack node): threshold-gated bright-run stretch with a
  bounded run search (≤ 64 taps, charged honestly — a true sort is
  unbounded; the bounded version is the honest version).
- **Avalanche** (rack node): PNG-style row-filter corruption. Full sequential
  propagation is hostile to GPU; the honest bounded design propagates the
  prediction error one row-band per frame through the node's own previous
  output (one retained surface, charged) — the cascade becomes visible motion,
  which is more tape-like than an instant tear, and it is deterministic.

**Proof.** DCT against a CPU reference (forward∘inverse = identity at
quantize 0); sort boundedness; avalanche determinism; three labeled export
cases. [closure]

### B14 — Failure switches (letting it fail)

**Why.** "A model that always recovers is a model that cannot actually
break." Two switches, and a philosophy worth writing into law: bounded state
may latch (stay broken until released) but never grow — latching is a state
flag, not an accumulating buffer, so no resource law is threatened.

**Seat.** `sync_latched` on the tape/NTSC-adjacent shear model (every shear
stays where it happened; releasing unwinds the accumulated displacement at
once — one stored per-line offset table, bounded); `servo_defeated` landed in
B3. Both are ordinary discrete authored state, patch-persistent, momentary or
latched from panel and native. Blackout and Program Freeze remain senior to
both.

**Proof.** Latch accumulation bounded; release-unwinds-at-once. [closure]

### B15 — Panel ergonomics and the snapshot bank

**Why.** We have more parameters than BENDR's 404 and no way to find one.

**Contents.**

- `/` search over control name, section, and help text; MOVING filter
  (any control a route currently drives — derivable from the compiled route
  table in the snapshot) and CHANGED filter (off-default, derivable
  client-side from defaults shipped with the panel). Client-side feature over
  existing snapshot data; zero new wire actions.
- Per-control help: a static text table embedded with the panel assets,
  also surfaced as native tooltips. Written in the house voice: what it does
  and why it behaves that way.
- Snapshot bank: 8 whole-rig slots with a glide time. Design decision to
  take at implementation: either widen Morph or — cleaner — a bank whose
  recall loads a slot into the existing Morph A/B and glides, reusing the
  ownership/materialization law wholesale rather than minting a second one.
  Slots persist in patches; recall is a revision-carrying barrier like Morph
  capture.
- Dice keep-masks: `keep_source | keep_modulation | keep_output_chain`
  flags on the existing Dice action, each defaulting to the current behavior.

**Proof.** Static-panel a11y tests for search/filters; snapshot recall
ownership fixture; [closure] for bank persistence.

## Remaining sequence

| Order | Tranche | Depends on | Character |
|---|---|---|---|
| 5 | B1 scan processor | dedicated-pass precedent | new machinery, flagship |
| 6 | B4 display physics | — | new surfaces, audience stage |
| 7 | B8 mixing boundary | blend audit first | novel algorithm |
| 8 | B16 program re-entry | — | small, high leverage |
| 9 | B5 codec mosh | worker precedents | worker + honesty boundary |
| 10 | B7 generators | — | new source class |
| 11 | B9 performance recorder | gesture track | instrument-defining |
| 12 | B10 mod sources | B2 (lattice arm-on-demand) | pure extension |
| 13 | B11 monitoring bay | B14 probe data | preview-only |
| 14 | B6 corruption trio | — | rack nodes |
| 15 | B14 failure switches | B3 servo (landed) | philosophy + small state |
| 16 | B15 ergonomics | — | panel + native |

## The working method every tranche follows

1. **Branch** from `feat/web-control-panel` (or stack on the open PR chain —
   verify with the GitHub API which branch carries the fullest landed state
   before branching); the operator (George) merges PRs via the GitHub web
   UI — never self-merge.
2. **Map the seams first.** Every authored value closes over: patch DTO
   (skip-serialized at default so old bytes and canonical hashes keep),
   sanitize-on-load with *neutral* non-finite fallbacks, wire action in BOTH
   validators (`web/server.rs` + `main.rs`) plus the applier, snapshot
   (additive `#[serde(default)]` block), panel rows (mind the range-count
   assertion in `web/state.rs` protocol tests), modulation targets (append to
   the END of `LAYER_TARGET_SUFFIXES`; master targets insert BEFORE `morph`,
   which must stay last), Morph law (values blend, angles wrap, discrete laws
   recall an endpoint at the midpoint), Dice/generator RNG in fresh
   domain-separated streams (old streams byte-stable — pin a golden from a
   worktree at the base commit, the B13 method; bump `GENERATOR_VERSION` only
   when a given seed's output changes), export through the shared plan with
   frame-indexed time, and a labeled export case.
3. **Consult the BENDR source** (`~/Downloads/bendr-main/src/`,
   `p20_shaders.js` + `p21_params.js`) whenever a spec's law is ambiguous —
   it is the ground truth for every derived mechanism.
4. **CPU reference first.** The law is a pure function the WGSL follows
   expression for expression; analytic fixtures test the reference, and a GPU
   fixture or labeled A/B ties the shader to it. **The hosted suite never
   parses most WGSL** — reserved keywords (`target` bit B13) surface only on
   the opt-in GPU fixtures, so run one early after any shader edit.
5. **Exactness A/B.** Render an existing labeled case from a pinned `git
   worktree` at the base commit, re-render on the new build, compare
   `ffmpeg -f framemd5` — the default path must be decoded-frame identical
   across the change.
6. **The six-step gate**, all of it, before any claim (see `CLAUDE.md`
   "Verification"; the vcvars preamble lives in the session memory notes).
   The host toolchain now tracks `stable` (matching CI) — still compare
   `rustc --version` against the CI log per tranche. Opt-in GPU fixtures run
   on this host (RX 6950 XT / Vulkan): `cargo test --locked <name> --
   --ignored`.
7. **Evidence note** in `docs/evidence/` + a `CLAUDE.md` section + a
   verification bullet + this status board, then commit, push, and open the
   PR via the GitHub API (token available through `git credential fill`; the
   `gh` CLI is absent). CI is verified per suite with
   `python scripts/check-ci-status.py <sha>`.

## Receipts and frozen contracts a successor will trip over

- `temporal.rs` pins the SHA-256 of `temporal.wgsl` (`07e5cde0…`, unmoved
  since B3 — B12 deliberately routed around it) in
  `originals_shader_contract_is_additive_bounded_and_reuses_three_texture_resources`,
  plus per-region `textureSample`/`textureLoad` counts for both temporal
  shaders. Editing either means re-pinning deliberately, in the same commit,
  with a comment naming the tranche.
- The M6 precision receipt fixture
  (`gpu_precision_receipt_measures_real_still_and_temporal_workloads`,
  opt-in) pins the shader-bundle digest inline (`3e280fd7…` after B13 —
  fullscreen + effects + temporal + composition_host WGSL, length-prefixed)
  and prints the receipt JSON as `M6_GPU_RECEIPT=…`. After any change to its
  source manifest: update the inline digest, re-run, and overlay the printed
  values onto the tracked `docs/evidence/m6-precision-gpu-receipt.json` —
  preserving that file's field order and its extra `measured_at` metadata.
  The six output SHAs must not move unless pixels legitimately changed. The
  still-fixture lane hashes the raw `EffectPassUniforms` bytes, so growing
  the effect uniform legitimately moves it (input identity, not pixels —
  B13 re-pinned it with a comment).
- `web/state.rs` pins the exact count of `<input type="range"` tags in
  `index.html` (**148 after B13**) — bump it with every added slider.
- Compile-time uniform size assertions: `MotionApplyGpuUniforms` 1,680 B
  (B2), `TemporalRigGpuUniforms` 96 B (B3), `EffectUniforms` 288 B /
  `EffectPassUniforms` 352 B with spatial slots at byte 288 (B13). WGSL
  structs change in the same commit.
- `LAYER_TARGET_SUFFIXES` compiled indices end at 91 after B13; new layer
  suffixes append from 92.
- `GENERATOR_VERSION` is "9" after B13.
- Generated-piece hashes: any new DTO field must be skip-serialized at its
  default or every pre-existing canonical hash breaks.
- WGSL reserved keywords (`target` among them) parse fine in the hosted
  suite and explode only when a GPU fixture builds the shader module —
  validate early.

## Standing decisions made during Wave 1

- One boundary numbering serves the whole program
  (`Transparent = 0, Mirror = 1, Wrap = 2, Hold = 3`); reuse
  `MotionBoundaryMode`/its config for any new edge law.
- Deterministic per-cell randomness uses the shared `cellular_avalanche`
  integer hash with a fixed domain constant and 24-bit unit floats; no wall
  time, no authored seed unless Dice needs to reroll it.
- Anything rate-like normalizes per 1/30-second reference tick (linear for
  additive terms, exponentiation for multiplicative ones, clamped
  tick-fraction mix for nonlinear stages).
- Servos and other self-regulating loops must be deterministic per-pixel laws,
  never readback-driven — live/export parity forbids measured-mean feedback.
- Fixed-law event clocks stay fixed law: the B2 trash clock (8 Hz), the B12
  sweep period (600 reference ticks), and the B13 anamorphic spread/tint are
  mechanisms, not authored controls.
- Non-default authored state may select the *originals* pipeline instead of
  editing a frozen shader (the B12 seam): if the frozen SHA does not need to
  move, do not move it.
- Master-only controls are enforced at every layer authoring seam with a
  clearing helper (the B13 `clear_master_only_effects` shape), never by
  panel convention alone.
- Discrete enums ride the wire as integers or closed tokens with permanent
  append-only codes; they get no modulation address, midpoint recall in
  Morph, and no Dice reroll unless the plan says otherwise.
