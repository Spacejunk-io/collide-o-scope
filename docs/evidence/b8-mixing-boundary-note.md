# B8 — the mixing boundary: evidence note

Branch `feat/mixing-boundary` off `942c415` (the B4 merge). Laws derived from
BENDR (MIT, © 2026 Steve Blythe), `src/p20_shaders.js` FS_MIX (:214-542),
`p21_params.js`, and `p43_render.js` (:704-744), rewritten for this tree; the
stale monoliths `p2_engine.js`/`p4_io_main.js` were identified via `build.py`
and ignored. `src/mixing_boundary.rs` is the independent CPU reference in the
`gesture.rs` tradition.

## What landed

Four pillars, one law module:

1. **The blend audit.** `BlendMode` grew append-only from 15 to 25 codes
   (VividLight 15 … Luminosity 24); every existing law kept its index, proven
   by the frozen vector tests whose rows 0..=14 kept their pre-B8 literals
   byte for byte and by the FNV signature over the first fifteen modes
   reproducing the old pin before the matrix was widened. The four HSV
   component-swap modes are non-separable and take a whole-triple path; the
   bitwise pair operates on the **stored sRGB code bytes**
   (encode → round → XOR/AND → decode), deliberately not on truncated linear
   values — a truncating linear quantizer flipped a bit on the live adapter
   whenever the CPU and GPU sRGB decodes disagreed by one ulp, which the
   `gpu_all_blend_modes_…` parity fixture caught and the code-byte law fixed.
   BENDR's own XOR runs on stored framebuffer bytes, so this is also the more
   faithful transcription.

2. **The bus mix laws.** The bus pass (`fs_bus`) now owns the wipe vocabulary
   (13 closed codes; `Dissolve` is the exact historical constant crossfade),
   MULTI tiling, softness with the fader remap, invert, movable origin, the
   border rule from the closed eight-colour bench table, and the blend family
   at the A/B meet. `Normal` keeps the historical premultiplied lerp as a
   **textually explicit branch**, so the default bus is byte-identical — the
   M6 receipt's six output SHAs did not move across the rewrite.

3. **The dirty mixer.** An event clock (`0.5 + rate·15` Hz) whose tick index
   is the only state; four fault laws (knock, cut, dropout, noise), each
   bit-clean between firings. All randomness is the Shift band/epoch/seed
   law on the integer avalanche in fresh per-lane domains — never BENDR's
   float hash — so live and export replay identical faults from frame-plan
   time. Faults are coverage-honest: they tint or replace covered light and
   never mint coverage, which is what keeps an all-Program composition
   (LegacyExact) exactly inert under authored dirt.

4. **The melting edge, two seats.**
   - *Bus melt* inside `fs_bus`: the analytic mix matte probed at four
     points, band/normal/swirl/creep, the incoming-lane drag, and the hold
     that dissolves the stage's own previous output back into the band. The
     history is **one** retained working-format surface on the
     temporal-feedback single-surface precedent, lazily allocated inside the
     executor *before* the warm-allocation snapshot (first armed frame
     allocates once, a warmed armed frame allocates nothing — the existing
     production fixtures enforce this at runtime), stored by `copy_texture`
     at most once per 30 Hz reference tick on the stage's own accumulator,
     and staged/committed through the executor's frame-history transaction
     so a discarded frame publishes no validity bit.
   - *Master melt* (`renderer/melting_edge.rs` + `melting_edge.wgsl`): a
     Recipe-B stage on the slot-0 seam immediately before the B4 display
     stage. Its matte is the composite's own alpha, so static key alpha,
     cellular gap, and group mattes all melt through one mechanism. History
     is one slot-format RGBA8 surface (the B4 held-field charge, 4 B/px),
     copied per reference tick. Params ride `TemporalParams.melt` with the
     full B4-shaped closure.

5. **Key dressing.** `key_border`, `key_border_color` (closed eight-colour
   table, discrete), and `key_shadow` on both static-key scopes. The house
   adaptation, documented in the shader: a layer has no composite underneath
   it, so the dressing **joins the key signal** (fill + matte) exactly as a
   broadcast border generator adds fill to a key — border as a six-tap
   asymmetric dilation (four axis + two diagonals, BENDR's own kernel) and
   shadow as a single offset matte tap darkened to black. Neighbour mattes
   evaluate the spatially mapped source through the one canonical
   `sample_source` chain (a bounded approximation the dressing's own
   amounts gate).

## Interpretations and deviations, recorded

- **"Group buses and the master key resolve"** (the melt seat) is read as
  the two mixer surfaces this program actually has: the one bus A/B meet
  and the program's own coverage at the slot-0 seam. That is exactly two
  armed melt scopes — the ledger's cap is structural rather than checked.
- **Melt histories are program memory, not display memory**: blackout does
  not clear them, on the temporal-ring/temporal-feedback precedent (the
  audience goes dark; the program underneath keeps its state). B4's
  phosphor clears at blackout because it models the *screen*; the melt
  models the *image process*. Disarm invalidates; re-arm never resurrects a
  stale trail.
- **The melt store advances at most one step per rendered frame** at the
  30 Hz reference; below 30 fps the creep runs at frame rate. Export at the
  same fps replays identically (the accumulator is fed by the
  program-advancing delta on both sides).
- **The dirty mixer seats strictly at the A/B meet**, before Program-over.
  Program-lane content bypasses the crossbar, which is what a program bus
  is; the fault laws' zero-preservation keeps the LegacyExact eligibility
  gate untouched.
- **Slide/stretch transitions and BENDR's PIP keyer are not in the B8
  vocabulary** (the spec's wipe list is the twelve analytic fields plus
  dissolve); BENDR's help-text claim that `detail` sizes Box/Circle is
  false of its own code and deliberately not reproduced.
- **`wipe_rep` x9 exists** (rep 3); BENDR's UI offered x4/x16 but its law
  is `fract(uv·rep)` for any rep, which the closed 1..=4 range exposes.
- **Sampler use in the melting-edge stage** (level-0 sampled reads through
  a filtering sampler) follows the opaque-resolve precedent at the same
  seam rather than B4's sampler-free loads; every tap is `SampleLevel`, so
  no read requires implicit derivatives.
- **Bus uniform is 128 bytes** (compile-time asserted), replacing the
  16-byte crossfade block; the bus pipeline gained a third bind group for
  the melt history so the shared universal copy/bus texture layout kept its
  exact prior shape.
- The **contribution predicate** widened: while dirt is authored, neither
  A nor B can be culled on the fader position, because a firing can throw
  the crossbar or drop line bands through to either input.

## Closure

- **Patch**: `CompositionTree` carries `mixer` (skip-serialized at the
  exact-legacy default — a default tree's YAML has no `mixer` key, so every
  pre-B8 patch keeps its bytes and canonical hash); `TemporalConfig.melt`
  and the three `EffectsConfig` dressing fields are likewise skip-at-default.
  Hostile scalars sanitize to neutral values; unknown fields are rejected.
- **Wire**: `set_composition_bus_mix { param, value }` — ordinary,
  coalescible per param, revision-free, quantizable — with the closed
  vocabulary parsed by the single shared `BusMixerEdit::parse` table used
  by the server gate and the applier alike (they structurally cannot
  drift). The master melt rides six `melt_*` params on `set_temporal` in
  both validators plus the applier; key dressing rides `set_param` /
  `set_layer_param` beside the existing key vocabulary.
- **Snapshot**: additive `mixer` block on the creative composition
  snapshot, additive `melt` block on the temporal snapshot, three additive
  effects-snapshot fields.
- **Panel**: the BUS MIXER group beside the A/B fader (17 sliders + wipe/
  blend/border-colour/MULTI selects + invert toggle), the MELTING EDGE
  temporal group (6 sliders), and the key-dressing rows in both KEYING
  surfaces. The static range pin moved 165 → 190; the app.js template pin
  17 → 19.
- **Morph**: the mixer bundle blends its continuous values and recalls
  pattern/invert/rep/border-colour/blend at the midpoint
  (`interpolate_bus_mixer`); the master melt blends all six
  (`interpolate_master_melt`); key dressing blends border/shadow and
  recalls the colour at the midpoint; the mixer transfers with the
  crossfade under the same Look identity gate.
- **Modulation**: seventeen `composition/bus_*` addresses beside
  `bus_crossfade` (`CompositionModParameter::ALL` = 18); six `melt_*`
  master addresses; `key_border`/`key_shadow` at master and as layer
  suffixes 92/93. Every discrete law has no address.
- **Dice**: the seventeen bus values mutate in a fresh domain-separated
  stream (`DICE_BUS_MIXER_DOMAIN`, field-separated by owner index); key
  dressing appends to the end of the B13 small-fx stream, so every earlier
  draw is byte-stable. Discrete laws never reroll.
- **Generator**: `mutate_master_melt` in fresh field-isolated domains; the
  bus mixer is preserved exactly (the crossfade precedent); key dressing
  appends to the end of the per-scope effects stream.
  `GENERATOR_VERSION` is now **"11"**: widening `BlendMode::ALL` changes
  which mode a firing blend-mutation draw lands on (the draw count itself
  is proven unchanged).
- **Export**: the shared executor and the same shaders on every path; the
  melting edge encodes at all three live seats and both export seats,
  always immediately before the display stage.

## Measurements (AMD Radeon RX 6950 XT / Vulkan, this machine)

- **M6 receipt**: re-pinned twice in this tranche — the shader-bundle
  digest (host bus rewrite, then the effects key dressing) and the
  still-fixture input-identity lane (`EffectPassUniforms` 352 → 368) — and
  after each re-pin the **six output SHAs did not move**, which is the
  byte-exactness proof for every default branch B8 added.
- `gpu_all_blend_modes_match_linear_cpu_opaque_transparent_and_half_alpha_vectors`
  passes for all 25 modes on the legacy and matte compositors (renamed from
  `…fifteen…`).
- `gpu_melting_edge_drags_the_band_holds_history_and_needs_a_boundary`:
  dormant identity, band-only drag with a byte-exact far field, the hold
  dissolving red history into a green present, and the
  no-boundary-nothing-happens law — all on the real adapter.
- The 18 `renderer::composition::tests` production fixtures (Symmetry,
  Field Collider, scan processor, warm-allocation invariance among them)
  pass with the bus rewrite and the melt store in the frame.
- `render_bus_mixing_boundary_pipeline` and
  `render_melting_edge_and_key_dressing_pipeline` are the labeled export
  cases, each with a difference twin and a `_repeat` determinism assertion.

## What is proven and what is not

Vocabulary codes/keys/tables, sanitize laws, wake laws, the wipe fields'
analytic landmarks, MULTI tiling, endpoint exactness under the fader remap,
the border band profile and gates, the event clock's bit-clean quiet ticks
and analytic envelope, per-seed determinism and seed-lane independence, the
honest dropout probability gate, the melt band/normal/swirl/creep/cap laws,
the 601 YIQ chroma walk, the shared wire-edit table, serde round trips with
unknown-field rejection, morph blending with midpoint recall, modulation
ranges with discrete refusals, and Dice bounds are hosted CPU tests. The
pixel claims above are opt-in `#[ignore]` fixtures measured on this one
adapter; portability rests on hosted three-platform CI, not on it.
