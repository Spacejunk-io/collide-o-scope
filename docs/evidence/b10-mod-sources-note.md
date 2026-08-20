# B10 — modulation source expansion

**Claim.** Every new expressive input is a `ModSource` and nothing else: six
momentary bend pads, four triggered envelopes, four macro knobs, three
deterministic generators (chaos, drift, spike), and three video-reactive
observations (motion, brightness, cut) all enter the one matrix law — the
same shaping, slew, routing, and depth machinery every existing source uses —
with no new modulation targets, no morph surface, and no change to
`LAYER_TARGET_SUFFIXES`. The laws are derived from BENDR (MIT, © 2026 Steve
Blythe, `p3_mod_ui.js` and `p43_render.js`); the house hardening is the
deterministic-replay claim BENDR never made: every generator and the video
analysis are pure functions of accumulated program seconds, the persisted
seed, and the frame inputs, so the same patch replays the same trajectory
live and offline.

## The sources

- **Bend pads** (`bend1..bend6`, unipolar): momentary engine surfaces on
  BENDR's asymmetric ramp (24/s toward held, 7/s toward released — a tap
  reads as an attack with a natural tail). Driven by the native digit row
  1–6 (a dedicated `map_bend_key` mapper, so `map_key`'s pinned
  release-is-inert law never moves), six panel pads copying the XY pad's
  pointer-capture/blur/visibility/reconnect machinery literally, and six
  appended `ControlParameter::Bend*` variants so a controller profile binds a
  note or button with `Gate` mode straight onto the engine surface (the
  GestureContact precedent — never a queued `WebAction`). Held state is
  runtime-only: a patch can never restore a held pad, focus loss releases
  every pad, and a bend edge held for a downbeat would move a hand-timed
  trigger, so `bend_pad` is refused in `Quantized` batches at both gates,
  carries no coalesce key (a pending press must not be replaced by its
  release), and both edges are priority.
- **Envelopes** (`env1..env4`, unipolar): BENDR's exact laws — linear attack
  that *resumes from the current level* on retrigger, exponential decay,
  modes `once`/`gate`/`loop` — with the closed trigger vocabulary
  `bend1..6 | audio_onset | scene_cut | beat | beat2 | bar`. Beat triggers
  fire on whole-multiple crossings of the global beat and anchor without
  firing on their first observation, so loading a patch mid-bar fires
  nothing. Attack clamps 0.005..10 s, decay 0.02..30 s, non-finite takes the
  neutral default.
- **Macros** (`macro1..macro4`, unipolar): an authored knob that is a source;
  fan-out is just routes.
- **Chaos** (bipolar): BENDR's hold-interval random walk made deterministic —
  interval durations (0.12..0.62 s) and targets both come from the
  sample-and-hold hash law (the LFO seed precedent) over the interval
  ordinal, eased at 7/s. **Drift** (bipolar): BENDR's fixed three-sine sum,
  already deterministic, on accumulated program seconds. **Spike**
  (unipolar): firing decisions evaluated per 30 Hz reference tick
  (`TEMPORAL_REFERENCE_FPS`, not a second literal) at 1.6 expected events per
  second with hashed amplitudes 0.7..1.0 and 9/s decay — tick-addressed, so
  the trajectory is frame-rate invariant. One persisted `generator_seed`
  (zero default) serves both hashed generators through separate domain
  words; reseeding restarts the trajectories deterministically.
- **Video-reactive** (`video_motion`, `video_brightness`, `video_cut`, all
  unipolar): BENDR's content-analysis law transcribed whole — one 32×18 luma
  grid (Rec.601 luma on encoded values) serving all three: brightness is the
  mean, cut is an onset against `max(0.06, 3.5 × EMA)` with `exp(-5t)`
  release, and motion is peak-normalized frame difference. The
  first-frame-after-a-gap law zeroes motion and cut so a re-arm never reads
  as one enormous cut. **Two deliberate deviations from the tranche sketch,
  both toward BENDR's shipped law:** motion is frame difference, not a
  Motion-lattice readback (the constants are tuned to it, it shares the one
  reduction all three sources use, and the lattice would add a readback the
  budget does not want), and the grid is BENDR's 32×18, not the sketch's
  16×16.

## Arming, cadence, and the two halves of the analysis

`ModMatrix::video_analysis_armed` is BENDR's arm-on-demand law: the analysis
runs only while a route sources a video value or an envelope fires on the
scene cut; everything else costs nothing. Live, a lazily constructed
`VideoAnalysisGpu` reduces the pre-blackout opaque audience image (composite
slot 2, the program-tap seam, in its own encoder at the acceptance decision)
into a 32×18 target — 16 bilinear taps per cell, filtered in linear light —
and reads it back through its own two-slot bounded pool (FIFO by sequence; a
busy pool drops the sample cleanly, freshness never backlog). Cadence is
10 Hz on the reference grid (three ticks), accumulated from accepted
program-advancing deltas, so Pause holds the analysis still. Offline, the
export loop feeds the *same CPU law* (`reduce_video_analysis_grid`, bilinear
at the same UV addresses in the same linear space) from the frame bytes it
already reads back for ffmpeg, at the same cadence, landing each sample in
the matrix for the *next* frame — the N-1 law the live readback obeys — so
video reactivity is deterministic offline. GPU/CPU agreement is the B7
statistical contract (≥95 % of the 2,304 grid bytes within 4 code values),
proven by the opt-in `gpu_video_analysis_reduction_matches_the_cpu_reference`
fixture.

**Ledger** (deliberately outside the full-frame texture floor, the
pattern-synth precedent): one 32×18 RGBA8 target (2,304 B) plus two 4,608-B
staging buffers, lazily allocated on the first armed sample; one render pass
(576 fragments × 16 taps) and one 2,304-B copy per armed 10 Hz sample. The
three full-frame audience readback slots are untouched.

## Closure

- **Patch**: `ModConfig` gains `envelopes` (emitted whole once any slot moves
  off default), `macros` (emitted once any is nonzero), and `generator_seed`
  (zero omitted) — all skip-serialized so pre-B10 patches keep their bytes
  and canonical hashes; hostile scalars sanitize to the documented ranges and
  unknown trigger/mode tokens keep the slot defaults. Runtime state (bend
  holds, envelope levels, generator clocks, video values) resets at apply so
  a restored program replays from its own zero. The procedural normalizer
  round-trips through `apply_to_matrix`/`from_matrix` and therefore
  sanitizes the new sections with no new code.
- **Wire**: `set_envelope` / `set_macro` / `set_mod_seed` are ordinary
  coalescible value edits (`env:{i}:{param}`, `macro:{i}`, `mod:seed`),
  gated at the server by the engine's own closed vocabularies; `bend_pad` is
  the stream event described above. Applying a new seed restarts the
  generators — the reseed itself is replayable.
- **Snapshot**: additive `envelopes` (config + live level meters), `macros`,
  `bends`, and `generator_seed` blocks on `ModSnapshot`.
- **Panel**: the PERFORM SOURCES group in the third column (pin 3 → 4): six
  press-and-hold pads, four JS-rendered envelope rows, four macro knobs, and
  the seed field. Twenty new entries in `MOD_SOURCES`; chaos and drift join
  the bipolar meter classification. Three new JS range-template tags move
  the app.js pin 21 → 24; the HTML pin stays 198.
- **Modulation targets**: none added — these are sources. `morph` stays
  last; the suffix table is untouched. Dice and the generator do not touch
  the new sections (`GENERATOR_VERSION` stays "12"; the modulation config
  rides patches exactly as authored).
- **Export**: envelopes (Beat and AudioOnset triggers), chaos, drift, spike,
  and macros are pure functions of `(beat, dt, seed, config)` and replay
  offline through the existing `update_at_beat` seam; bend pads are
  live-only zeros (a held pad is a hand); video sources are derived offline
  from the job's own frames as above. `render_mod_sources_pipeline` is the
  labeled export case: an envelope into brightness, chaos into hue, drift
  into position, and video brightness into contrast — the `_unrouted` twin
  must decode differently and the `_repeat` render identically, which is the
  deterministic-replay claim measured end to end.

## Proof

Hosted: the closed source and trigger vocabularies; every envelope law
against its closed form (attack timing, exponential decay, gate hold, loop
re-fire, retrigger-resumes-from-level, beat-crossing anchor law); chaos
determinism per seed with bounds; spike tick-addressing and frame-rate
invariance; the analytic drift sum; bend ramp asymmetry, out-of-range
refusal, and reset release; macro clamping; the video analysis law
(first-frame honesty, cut onset and decay, no-source decay, hostile input),
the reduction reference's flat-field exactness and gradient monotonicity,
and the arm-on-demand predicate; the ModConfig round trip with
skip-at-default omission and hostile sanitize; wire classifications and both
gates' vocabularies; the panel contracts; keyboard mapping with the
release-is-inert law intact; applier sanitize and the never-latchable bend;
reseed determinism through the real action door. Opt-in (RX 6950 XT /
Vulkan): the GPU reduction parity fixture and `render_mod_sources_pipeline`
with its difference twin and `_repeat` determinism assertions.
