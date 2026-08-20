# B7 — generator sources (pattern synth, text page): evidence note

Branch `feat/generator-sources`, from `770545b` (the B5 merge). Laws derived
from BENDR (MIT, © 2026 Steve Blythe); the FS_GEN signal path and the
text-page layout laws are transcribed with attribution, and the surrounding
machinery is a rewrite.

## What landed

Two new `LayerSource` arms — the first sources with perfect offline
reconstruction. No file identity, no content reference, no black placeholder:
the patch carries everything.

- **Pattern synth** (`synth://pattern`): the picture is computed by one GPU
  pass per frame (`pattern_synth.wgsl`, no texture, no sampler) on a fixed
  1920×1080 page, from 22 continuous authored values plus three closed
  vocabularies (12 shapes × 6 waveforms × 5 colourisers, BENDR's own default
  Scan/Sine/RgbPhase). The whole signal path — framing, shape, oscillator,
  cross-modulation, wavefolder, comparator, colouriser — is stateless: a pure
  function of authored values and frame-plan time (`t = time × rate`, BENDR's
  own law, so Pause holds the picture and export replays it structurally).
  `src/pattern_synth.rs` is the independent CPU reference the WGSL follows
  expression for expression, keeping BENDR's literals (3.14159, 6.2831853,
  the 0.003 gates). The computed value is display-domain; the shader decodes
  through the exact piecewise sRGB transfer so the stored bytes are the
  picture BENDR computes.
- **Text page** (`text://page`): a static typeset page rastered on the CPU
  (1920×1080 RGBA, opaque) from its own authored state — body, one of two
  bundled licensed faces (Hack MIT / Ubuntu-Light UFL via
  `epaint_default_fonts`, already in the dependency tree: zero new embedded
  bytes), size/track/rotate/repeat/outline, and the shape fan with BENDR's
  `1 − f·0.55` taper. Re-rastered **only on authored change** — between edits
  it costs exactly what a still costs. The deliberate deviation from BENDR:
  the clocked terms (text scroll, shape spin, shape pulse) are absent because
  the page's law is re-render-on-change; movement is authored downstream
  through the spatial transform, effects, and Motion. Stroke text is a
  bounded morphological band (dilate−erode, radius ≤ 10 px); rotation is a
  per-row rotate-blit about each row's own anchor.

## Seats

- **Live**: the pattern pass encodes into the frame encoder immediately after
  its creation, before the Advanced prepare and the LegacyExact
  `render_evaluated_frame`, so both paths sample a picture the frame plan
  alone determined. The executor (`renderer/pattern_synth.rs`) is
  renderer-owned and lazy — a session with no pattern layer charges nothing.
  Uniforms come from the plan: `EvaluatedFramePlan` carries each pattern
  layer's modulated copy (`modulate_layer_pattern`, the same frame-local
  offsets accumulator every per-layer consumer reads), so live/export parity
  is structural. Text pages ride the still-image publish law verbatim
  (pending frame → ready-frame pump → checked upload, restore on failure).
- **Export**: the identical executor type, export-owned, encoding into the
  export frame encoder at the same seat; the text page rasters once through
  the same law-module function. `resolve_visual_source` short-circuits both
  sentinels before any filesystem work, exactly like `spout://`.

## Ledger

Pattern: 1 render pass per pattern layer per frame, 0 sampled textures,
0 samplers, one 128-byte uniform (compile-time asserted), one layer-owned
1920×1080 RGBA8 texture (RENDER_ATTACHMENT added; charged through the
ordinary still-image media-safety plan, Safe-sized, zero Expert bytes). The
renderer-owned full-frame texture floor (30) is untouched — the page is a
layer texture. Text: one bounded raster ≤ 8.3 MiB on authored change, one
upload, then still-cost.

## Closure

- **Patch**: `LayerConfig.pattern` / `LayerConfig.text_page`, skip-serialized
  at `None` so every pre-B7 patch keeps its bytes and canonical hashes.
  Hostile scalars sanitize to neutral values; unknown tokens and unknown
  fields are deserialization rejections (closed serde enums).
- **Wire**: `add_pattern_layer` / `add_text_layer` (topology, immediate);
  `set_layer_pattern` / `set_layer_text` (coalescible per param,
  `layer:{key}:pattern:{param}` / `:text:{param}`), validated at the server
  gate and applied by the engine through the **single shared parse tables**
  (`PatternSynthEdit::parse`, `TextPageEdit::parse` — the B8 `BusMixerEdit`
  law), so the accepted and applied vocabularies are structurally one.
- **Modulation**: 22 `pattern_*` layer suffixes appended at compiled indices
  94–115 (`LAYER_TARGET_SUFFIXES` now 116); the three vocabularies have no
  address. Routes on non-pattern layers are dormant (no base, no cost).
- **Morph**: `LayerMorphSnapshot.pattern` captures only pattern layers;
  interpolation requires both slots to carry it (two source kinds are two
  pieces, not two ends of a blend), hue blends on its shortest wrapped unit
  arc, the three vocabularies recall an endpoint at the midpoint, and
  application is kind-gated. Ownership (`LayerMorphControl::Pattern`)
  releases on a manual pattern edit exactly as effects do. The text page is
  deliberately outside Morph: its body is content identity, not performance
  state.
- **Look**: pattern values transfer kind-gated (the matte match-then-move
  precedent); the text page is deliberately not transferred.
- **Dice / generator**: both preserve generator source state exactly (the
  source-identity law: Dice has never touched source state, and the
  generator's mutation walk does not know these fields). **No
  `GENERATOR_VERSION` bump** — no existing seed's output changes. Generated
  manifests record generators as `kind: pattern_synth|text_page`,
  `offline_policy: reconstructed`, verified with no bytes — they never trip
  `--allow-black-sources`.
- **Snapshot/panel**: additive `pattern`/`text_page` layer blocks; the layer
  card renders a PATTERN SYNTH section (22 sliders + 3 selects) or a TEXT
  PAGE section (textarea, 12 sliders, 2 selects, 3 colour wells); generator
  add buttons beside the Spout form; `SYN`/`TXT` thumbnails;
  `offline_export_policy` is empty for generators (they reconstruct). Range
  pins: `index.html` stays **198**; `app.js` template tags 19 → **21** (one
  row template per generator section — rows render from tables).
- **Prepared sources**: staging a sentinel into a clip slot is a typed
  refusal ("authored on its layer"), never a positional fallback.
- **Export provenance**: `ExportMotionSourceKind::{PatternSynth, TextPage}`
  with sidecar keys `pattern_synth` / `text_page` — additive values in an
  existing field, **no sidecar schema bump** (the B2 precedent; schema stays
  6).

## Proof

Hosted (all CPU, in the ordinary suite — 1474 tests green through the gate):
the transcription fixtures (oscillator waveforms, radial symmetry, Scan
separability, hard comparator, wavefolder range+reach, colouriser laws,
frame invariances, 128-byte uniform), the text raster fixtures (opacity,
determinism, glyphs land, faces differ, shape fan fill/stroke laws, repeat
and outline reach, body-cap truncation on a char boundary, frozen code
tables), patch round trips with absent-section byte identity and rejection
laws, the wire accept/reject battery through the shared tables, the
modulation address battery (ranges, discrete refusals, stable indices
94–115, copy-never-base), and the Morph blend/wrap/midpoint/gate battery.

Opt-in, this host (AMD Radeon RX 6950 XT / Vulkan):

- `gpu_pattern_synth_matches_the_cpu_reference_for_every_shape` — all 12
  shapes × 5 colourisers at a 144-point grid, ≥95% of channel samples within
  four sRGB code values of the CPU reference (the bar is statistical because
  BENDR's own screen hash ends in `fract` of ~4000-scale products, so one
  GPU/CPU ulp is legitimately amplified at isolated pixels; a wrong law
  moves every sample by dozens), pages opaque, double render byte-identical.
- `render_pattern_synth_pipeline` — the patch's only layer is a pattern
  synth (no file anywhere); `_default` twin decodes differently, `_repeat`
  decodes identically.
- `render_text_page_pipeline` — same shape for the page (`_alt` twin with a
  different body/fan differs; `_repeat` identical).
- Exactness A/B: `render_program_reentry_pipeline` rendered from a pinned
  `770545b` worktree and from this branch — decoded `framemd5` identical, so
  the default path did not move a pixel.

## Deviations and decisions (for the successor)

- The synth clock is `t = frame-plan time × rate` — BENDR's own GPU-synth
  law (`renderGen` takes the global clock and ignores `patClock`); the layer
  SPD control is inert for a pattern layer exactly as it is for a still.
- Fixed 1920×1080 page for both sources: output-sized pages would make an
  export at a different resolution a different picture, breaking the
  reconstruction claim.
- Layer-scope only; the modulatable surface is the 22 continuous values.
- Generator layers cannot be staged into clip slots in v1 (typed refusal);
  making them slot-loadable is a wire and staging change, not a source one.
