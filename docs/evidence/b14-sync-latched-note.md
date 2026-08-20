# B14 — `sync_latched`, the second failure switch

"A model that always recovers is a model that cannot actually break."
`servo_defeated` landed inside B3 and proved the shape at the feedback loop;
`sync_latched` is the other half, and it closes B14. The seat is the
tape/NTSC-adjacent horizontal shear: each reference tick some bands of
scanlines lose sync and slip sideways. Unlatched, a slip lives exactly as
long as its own tick and the picture heals — the B8 knock's law, bit-clean
between firings. **Latched, every slip is written into a bounded per-line
offset table and stays there, accumulating, until the operator releases the
switch and the whole accumulated displacement unwinds in one step.**

`src/sync_latch.rs` is the independent CPU reference in the `gesture.rs`
tradition — no `wgpu`, clock, filesystem, or UI dependency — and
`src/shaders/sync_latch.wgsl` consumes the table it produces.

## Bounded state may latch, but it may never grow

The table is the entire latched state: one `f32` per output line, hard-capped
at `SYNC_LATCH_MAX_LINES` (2,160 — a 4K line count) and
`SYNC_LATCH_MAX_OFFSET` (0.25 output UV) of displacement per line. That is
8,640 bytes at the absolute cap. Latching therefore costs one flag and one
fixed table rather than an accumulating buffer, and **no resource law is
threatened**: the stage owns no texture at all, allocates no surface, and does
not appear in any ledger. Its only GPU resource is one 8,656-byte uniform
buffer (a 16-byte header plus the capped table), compile-time asserted.

Accumulation is bounded three separate ways, each with its own fixture: per
slip (`SYNC_LATCH_SLIP_UV`, 0.02 UV at full amount), per line (the 0.25 cap,
clamped on every fold), and per frame (`SYNC_LATCH_MAX_TICK_BURST` = 24, the
same burst clamp `history_ticks_for_delta` already applies, so a long stall
cannot bill the table for every skipped tick at once).

## The seat

A Recipe-B stage on the shared slot-0 seam, between the B8 melting edge and
the B4 display stage:

```text
temporal → melting edge → sync latch → display physics → opaque resolve
```

That ordering is the law, not a preference: **a sync fault happens in the
signal, and the screen model downstream then shows it.** Live LegacyExact,
live Advanced, the selective-VHS path, and export all converge on this
adjacency, so one implementation and one shader serve all four — the
`encode_opaque_output` precedent, exactly the B4 and B8 shape. Five
production call sites: three live (`main.rs`) and two offline
(`render_export.rs`).

The three shear producers were surveyed before choosing. The B8 dirty
mixer's knock is bus-scope and does not exist on LegacyExact; the ntsc-rs
worker's line effects are third-party CPU work with no latch vocabulary;
Shift's band displacement is a layer/master effect, not tape-adjacent. None
could carry full live/export parity, so the stage is new — which the
directive named as the practical seat — and takes the B8 dirt law as its
*model* rather than its code.

## The laws

- **Draws.** Every draw is the Shift band/epoch/seed law on the shared
  integer avalanche (`mixing_boundary::lane_unit`) in two fresh domains,
  `LANE_SYNC_FIRE`/`LANE_SYNC_SLIP` ("SYN" 1 and 2), keyed by the master
  `random_seed`, the stage's own 30 Hz reference ordinal, and the band index.
  Nothing consumes sequential RNG state, so a tick recomputes alone and one
  lane can never perturb another.
- **Bands.** `band_height(spread)` spans 1 line (maximum shred) to 64 (a tape
  tear). Every line inside a band carries the identical offset, which is what
  makes a tear a tear rather than static.
- **Firing.** `band_fires` draws against `rate * 0.5`, so at full rate half
  the bands slip every tick and at zero nothing ever fires.
- **Bias.** `band_slip` folds the symmetric draw toward one side while keeping
  its magnitude, so at ±1 every slip carries the same sign and a latched table
  accumulates monotonically to the cap. That is the analytic handle the
  boundedness fixture uses.
- **Wrap.** A sheared line wraps around the frame, as a tape losing horizontal
  sync does. `fract` puts the coordinate back inside the frame and the sampler
  repeats on U, so the bilinear tap that straddles the seam filters across it
  rather than clamping.
- **Wake.** `is_active()` is `amount > ε ∧ rate > ε` — neither control alone
  wakes the stage, and the switch deliberately does not appear in it, because
  latching an inert stage accumulates nothing. Beyond that the executor skips
  encoding entirely whenever every offset is zero, so **the exact prior path
  never resamples at all**; a default patch encodes no pass.
- **Release.** Expressed as "unlatched implies an empty table" rather than as
  a falling-edge handler, so the two can never drift apart. One frame with the
  switch off and the whole accumulation is gone — not decayed toward home over
  several frames, home.
- **Silence is not repair.** Pulling `amount` to zero while latched stops new
  damage and holds what is already done. The switch is what repairs.

## The two rulings the directive asked a successor to write down

1. **The table is program memory.** Blackout does **not** clear it — the
   temporal-ring and bus-melt precedent, deliberately not B4's phosphor
   (which models the screen itself). The audience goes dark; the program keeps
   its damage, and a release resumes from the picture the cut interrupted.
   `reset_for` clears on exactly the causes that begin a new program
   (`PatchGeneration`, `ApplyLook`, `BroadRevert`, `Resize`, `ManualClear`)
   and holds through `SourceCut`, `Seek`, and `BlackoutTransition`, which are
   moves inside the same program. One hook —
   `Renderer::reset_visual_generation_for` — already carries all three
   required causes, and `revert_master_visual_state` (the `ResetVisualProgram`
   handler) reaches it as `BroadRevert`.
2. **The table does not persist in patches.** The switch and its four controls
   do; the accumulation is runtime state like a temporal ring, regrowing
   deterministically from the seed and the clock. So a patch never carries a
   frame's worth of shear history, pre-B14 bytes and canonical hashes keep,
   and live and offline agree from any common start — which is exactly what
   the labeled export case's `_repeat` measures.

Program Freeze needs no special handling: the stage is fed
`program_advancing_delta()` like its two neighbours, so Pause holds the fault
clock still. Blackout stays the absolute final audience operation, downstream
regardless.

## Closure

| Seam | B14 |
|---|---|
| Patch | `TemporalConfig.sync`, skip-serialized at the exact-off default |
| Wire | five `sync_*` params on the coalescible `set_temporal`, in **both** validators plus the shared `apply_temporal_wire_edit` |
| Snapshot | additive `sync` block plus the read-only `sync_damaged` fact |
| Panel | a SYNC LATCH group in the temporal section: four sliders and the switch |
| Modulation | four continuous master addresses; the switch has none |
| Morph | the four values blend; the switch recalls an endpoint at the midpoint |
| Dice / generator | preserve the block exactly; `GENERATOR_VERSION` stays "12" |
| Export | the same stage on the same seam, frame-indexed |

`AppSnapshot`'s `sync_damaged` is the honest half of the pair: the authored
switch says what the operator asked for, and this says whether the program is
**actually** still carrying accumulated shear — the fact a failure switch
exists to make visible.

Range pins: `index.html` 198 → **202** (the four sliders; the switch is a
checkbox). `app.js` literal template tags stay **24** — the group is static
markup. `GENERATOR_VERSION` stays "12", the sidecar schema stays 6, the
renderer texture floor stays 30 (the stage owns no texture), and
`temporal.wgsl`'s pinned SHA is untouched — the stage owns its own shader.

## Proof

Hosted (all three CI platforms), 22 law fixtures in `sync_latch.rs`: the
exact-off default; hostile values taking the neutral default rather than a
clamped extreme; out-of-range clamping; neither control alone waking the
stage; the switch alone never waking an inert one; the band-height span; a
zero rate never firing and a full rate firing about half the bands; bias
forcing the slip sign while keeping its magnitude; every slip inside the
declared bound; unlatched slips healing with their own tick; a frame inside a
tick holding the shear still; **latching accumulating monotonically and
stopping at the cap**; **release unwinding the whole table in one step**;
damage surviving a magnitude pulled to zero while latched; a stalled frame
folding at most the burst clamp; determinism per seed and distinctness across
seeds; the line cap; a hard clear; spread shearing contiguous blocks; the wrap
law keeping every sample inside the frame; the frozen uniform sizes; and the
hash lanes being distinct domains. Closure fixtures cover the patch round trip
(with the table proven absent from the YAML), the morph law, the modulation
addresses, generator preservation, both wire validators, and the B9
performance-recorder value-law oracle.

Opt-in on this host (AMD Radeon RX 6950 XT / Vulkan 26.7.1):

- `gpu_sync_latch_shears_each_line_by_its_own_offset_and_default_encodes_nothing`
  — the authored default encodes no pass and moves no pixel; an armed latched
  stage shears every line to where the CPU reference says, compared against a
  model that reproduces the filtering sampler's wrapped bilinear tap exactly;
  and releasing leaves at most one transient slip per line.
- `gpu_sync_latch_reset_clears_only_the_causes_that_begin_a_new_program` — a
  source cut, a seek, and a blackout transition all keep the damage; a broad
  revert clears it.
- `render_sync_latch_pipeline` — the labeled export case. The `_healed` twin
  carries the identical four controls with the switch **off**, so both renders
  draw the identical fault stream from the identical hash lanes and differ
  only in whether the faults heal; the decoded frames must differ, which is
  the tranche's whole claim made visible. `_repeat` must decode identically,
  proving a table deliberately absent from the patch still regrows
  deterministically offline. A third `_off` render pins the dormant default as
  a different program again.

Exactness A/B: `render_feedback_rig_pipeline` rendered on this branch and on a
pinned `a01dbba` worktree, decoded `framemd5` compared with `#software`
excluded — **33 rows, byte-identical**. B14 does not move a pixel on the prior
path.

Full six-step gate green at the final tree state: `cargo fmt --all -- --check`;
`node --check static/app.js`; `node --check docs/ui-ux/wireframe.js`;
`cargo check --locked --all-targets`;
`cargo test --locked --all-targets -- --test-threads=1` (**1,586 passed, 0
failed**); `cargo clippy --locked --all-targets --all-features -- -D warnings`.

## Deviations, named

- **Four sliders, where the directive's checklist estimated none.** The
  directive allowed either a new Recipe-B stage or "a widening of an existing
  one", and its closure checklist assumed the latter would need only a
  checkbox. No existing producer could carry the law with live/export parity,
  so the stage is new — and a slot-0 stage with no authored magnitude or rate
  is not an instrument, it is a control that can never be woken. The four
  continuous controls therefore exist, take four modulation addresses by the
  ordinary house law, and move the HTML range pin 198 → 202. The switch itself
  is exactly the checkbox the directive described, with no modulation address.
- **No native surface for the switch.** The spec's Laws say "momentary or
  latched from panel and native". The native patch editor *captures* temporal
  state rather than editing it, so **no** temporal discrete law has a native
  surface — not `fb_servo_defeated` (B14's own other half), not
  `mosh_recycle`, not `disp_model`. B14 matches the subsystem it lives in.
  Giving temporal discrete laws a native surface is a separate change across
  roughly forty existing controls, not a B14 change.
- **`armed`, not `active`, on the uniform.** `active` is a WGSL reserved
  keyword. The lane is named `armed` on both sides of the boundary. (The
  hosted suite never parses a shader; the GPU fixture caught it, as it has
  every tranche since B13.)
