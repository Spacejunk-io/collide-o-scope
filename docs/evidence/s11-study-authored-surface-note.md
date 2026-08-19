# S11 — the Study authored surface: evidence note

The authored-surface decision, opened by the operator and designed here: a
Study becomes a Collision Rack node — append-only kind code 13, lifted into
its own dedicated pass exactly like the Symmetry Field — whose authored
state is one content-addressed document digest. Documents live in the
bounded host library and travel whole inside patches (`studies` section,
the gesture-track precedent), so a patch stays self-contained and the
distribution question never opens.

Branch point: `301e775` on the mainline (S10b merge `7badc47` verified
green). Baseline: **1292 passed / 0 failed / 94 ignored**; with this
tranche **1297 / 0 / 96** — five hosted law tests plus the production
pixels fixture and the labeled export case.

## Design decisions, compressed

- **Copy forces content addressing.** `VisualNodeKind` is `Copy`; a
  document is heap. The node carries `Option<[u8; 32]>` — the canonical
  digest `CompiledStudy` already derives — and the `StudyProgramLibrary`
  (16 documents, hard cap, typed refusal, no eviction) owns the content.
- **Resolution is plan-visible identity.** The planner resolves digests
  through `CompositionPlanInput::with_studies`; the encoded program rides
  the plan, so live and export execute identical instruction streams; node
  id + digest + resolvedness + instruction count hash into the topology
  signature so a document assignment — or a library insert resolving a
  missing digest — re-prepares the renderer and re-uploads the arena.
  Without that hash the executor would serve a stale program; the test
  pins it.
- **Unresolved is inert, never a fallback** — proven to the pixel: the
  production fixture renders resolved ≠ control and unresolved ==
  control byte-identically.
- **The admission budget replaced the worst case.** The first descriptor
  declared the ABI worst case (65 loads/pixel) and instantly broke the
  32-lookup rack budget — a real design lesson caught by the new planner
  test failing. The resolution: declare eight (carrier + up to seven
  history loads; `LoadCurrentColor` reads the loaded register free), and
  refuse an over-budget *valid* document at plan time by name
  (`StudyLoadBudget`) — the over-budget Residual-grid law. Pinned: nine
  loads refused, eight admitted exactly.
- **V1 closure by absence** (the Field Collider precedent): no continuous
  authored value, no routes, no donors — so no modulatable address, no
  Dice/generator mutation (common wet dices like every node), Morph
  recalls the pair as one discrete endpoint at the midpoint (pinned), and
  Look/preset value application leaves the digest untouched (`_ => false`
  arm, correct by construction).
- **One wire action** — `set_visual_node_study_document`, coalescible per
  node (newest paste wins), never quantized; the engine validates and
  compiles into the library and sets the digest in one action, so neither
  can exist without the other. Panel: paste surface on the node card with
  client-side JSON errors in a polite status region.
- **Frame inputs**: `FramePlanContext` gains `study_audio_bands` /
  `study_beat_phase` from the same immutable frame facts the modulation
  matrix consumes; export supplies its own audio evaluation and beat
  clock, so a file-driven analysis clip drives a Study identically live
  and offline.
- **Patch apply is all-or-nothing**: every carried document must compile
  and the library must have room before anything live changes; capture
  carries exactly the referenced documents, deduplicated, in digest order.

| Surface | Proof |
|---|---|
| Kind/codes | code 13 append-only + registry/browser tables extended (existing freeze tests, updated deliberately). |
| Params serde | hex digest round trip, hostile digests rejected, unknown fields rejected, default = exact bypass (hosted). |
| Planner | flush-lift-resume at the authored position, kind-only (dormant keeps its slot), ledger re-derived from emitted steps (1 pass / 2 textures / 8,256 uniform bytes), resolution + signature + load-budget laws (hosted). |
| Patch | `studies` section round trip, digest walk across master/layer/group racks, empty section byte-absent (hosted). |
| Morph | discrete endpoint recall at midpoint, equal documents carried (hosted). |
| Wire | action parse, coalesce key, panel wiring asserted from source (hosted). |
| Pixels | `production_study_field_reaches_the_pixels_and_unresolved_digests_are_inert` — resolved reaches the audience image, unresolved byte-equals no node, warm frames allocate nothing, deterministic (opt-in GPU, RX 6950 XT / Vulkan). |
| Export | `render_study_field_pipeline` labeled case renders through the real export path; digests resolve from the patch's own section via `ExportCreativeGraph`; no export-only Study path. |
| A/B | **verified**: all 30 pre-existing labeled outputs decoded-`framemd5` byte-identical across the tranche (renders cleared before each launch, base `301e775`); the fix side adds the 31st, `audit_study_field.mp4`. |

Gate on `RUSTUP_TOOLCHAIN=1.97.1`: fmt, both node checks, check, tests,
clippy `-D warnings` — run on the final tree before commit.
