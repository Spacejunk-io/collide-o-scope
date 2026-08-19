# S8 — proxy hot adoption: cross-cutting completion matrix

Every row of the plan's matrix, with the exact test name that proves it or an
explicit reason it does not apply. An unstated not-applicable is
indistinguishable from an unrun check, so each is argued rather than asserted.

This is Gate 1's first edge, opened by the operator: a published proxy now
adopts into every matching live layer at its current playhead, instead of
waiting for the next patch (re)apply. The Y-key lifecycle completes in one
motion — request, encode, sealed publication, live adoption — and the A/B
telemetry improves structurally, because both halves of the decode
comparison now come from one continuous session on one layer.

Branch point: `ae842b6` (`feat/web-control-panel`, the S7 capability-evidence
merge), verified green by the suite-aware `scripts/check-ci-status.py`
(1 complete suite, 0 failed, 0 pending). Baseline there: **1275 passed /
0 failed / 89 ignored**. With this tranche: **1277 passed / 0 failed /
91 ignored** — the two new hosted adoption tests and the two new opt-in
fixtures, nothing else moved.

## The design, compressed

**One consultation law, three callers.** Hot adoption trusts an artifact
only through `consult_proxy_cache` — the same seal re-hash, source
re-probe/re-plan, and decoded-identity validation the patch-load path uses.
There is no laxer adoption-time private path, and the hosted tests prove the
refusals arrive through exactly that shared law.

**One seed dance, two callers.** The playhead-seeded decoder open was
already solved for clip-slot resume in `performance_runtime::prepare_file`.
That dance was extracted to `ThreadedDecoder::select_seed_frame_at` — with
its exact messages preserved caller-side — and hot adoption calls the same
method, so the two cannot drift. The extraction is mechanical; the
performance preparer's mapping keeps every error string byte-identical.

**Claims, not indices.** The render thread captures a per-layer claim at the
encode completion — stable layer ID, source-resource epoch, live playhead —
and the drain re-validates every part of it against the live layer before
installing. A stale epoch (clip-slot switch), a vanished ID (patch apply
mints new IDs), a changed identity, or an already-backed layer is discarded
with a named reason, never applied to whatever now occupies the position.

**Not a clip switch.** `commit_adopted_proxy` is the infallible install
behind fallible GPU staging, and it deliberately touches only decoder-facing
fields: runtime path, source, texture, dimensions, preload weight, codec
motion (reset, as a transport discontinuity resets it). Slots, transport
position/direction/generation state, speed, target FPS, pending seeks,
pause, the authored filename, and the persisted content reference are all
untouched — so the audience keeps the exact playhead, and a completed
OneShot stays transparent instead of flashing back.

## The CLI boundary, stated once

Unchanged from S7: hosted CI's Unix FFmpeg has no CLI. The adoption tranche
splits the same way — consultation-refusal behavior and event mapping run
hosted on all three platforms with hand-built sealed stores; the
decoder-opening halves are opt-in (`--ignored`) and were run on this host
against FFmpeg 8.1.2, the GPU half additionally on the receipt adapter
(AMD Radeon RX 6950 XT / Vulkan).

## What failing tests taught this time

The GPU fixture's first run failed at cleanup, not at the claim: on
Windows, `remove_dir_all` raced the adopted decoder's file handle, because
`ThreadedDecoder::drop` signals its worker without joining. That is now a
load-bearing comment in the fixture — the platform law is real (an adopted
artifact stays open as long as its decoder lives), and the eviction path
already tolerates it because a refused artifact is discarded by the store,
which owns no decoder.

| Surface | Required proof | Status |
|---|---|---|
| Domain | typed refusals, hostile bounds, exact claims | **Covered.** `hot_adoption_prepares_nothing_without_an_artifact_and_names_the_refusal` (empty candidates → no events, no consultation; empty cache → one named job-level refusal, never a fabricated per-layer preparation) and `hot_adoption_refuses_through_the_same_consultation_law_as_patch_load` (unprobeable original refused; post-seal corruption refused *and discarded*, `error.contains("seal")`). Every event is typed `ProxyAdoptionEvent`; every drain discard is a named reason string. |
| Threading | bounded worker, refuse-busy, nonblocking drain | **Covered structurally, the encode worker's exact shape.** One thread, one-slot `sync_channel` refusing while busy ("a proxy adoption is already preparing"), events drained once per frame beside the encode worker's drain, Quit cancels both workers. Preparation — seal re-hash of the whole artifact, source re-probe, decoder open, playhead seek — never runs on the render thread; only the infallible field swap and one GPU staging upload do, which is precisely what clip-slot commits already do there. |
| Seed correctness | the swap presents the playhead, not frame 0 | **Covered, opt-in.** `proxy_hot_adoption_prepares_playhead_seeded_decoders_end_to_end`: two candidates at 0.0 and 0.5 each receive their own half-scale (32×18 from 64×36) decoder, claims passed through verbatim, and the two seed frames demonstrably differ — a mid-clip playhead cannot be seeding the start. |
| The install | identity, filename, playhead kept; decoder, texture, dims moved | **Covered, opt-in GPU.** `gpu_proxy_hot_adoption_swaps_a_live_layer_and_keeps_identity_and_playhead`: after the swap, `proxy_backing` is the key, dimensions are the artifact's, the runtime path is the artifact path ("the only proxy fact"), while filename, `source_reference_for_persistence`, and `clip_transport.position` are byte-identical and the source-resource epoch advanced. |
| Staleness | a stale claim never lands on another layer | **Covered by construction, argued.** The claim carries the stable layer ID (never positional index) and the source-resource epoch; `commit_prepared_source` bumps the epoch on every slot switch and patch apply constructs entirely new IDs, so both invalidation routes are the same mechanisms the rest of the program already relies on. The drain's four guards each produce a distinct named discard. App-level tests require a GPU adapter in this harness; this is the same stated seam-level gap as S7's consumption wiring row. |
| Persistence | no proxy in patch, export, or Dice | **Covered by exclusion, unchanged from S7.** `commit_adopted_proxy` never touches `persisted_source_reference` (the activation's identity fields are explicitly ignored — "identity is not the activation's to change"), so patch capture and export's digest gate behave exactly as the S7 tests already pin. No new `PatchState` field, no snapshot field, no wire action. |
| Look/Morph/Modulation/Dice | values-only laws | **Not applicable — no authored creative state.** Adoption changes no authored value; it is a runtime source-resource exchange, the same category as a decoder reopen. |
| Planner/GPU | resource preflight, warm allocation | **Covered by equivalence.** The one GPU operation is `LayerSourceActivation::stage` — the identical texture-create-and-upload the performance runtime already performs per prepared slot, inside the same checked scopes. The old texture is dropped with the displaced source; net persistent surfaces unchanged (one per layer). The half-scale artifact strictly shrinks the layer texture and decode working set. |
| Reset/freeze | pause and freezes hold their meaning | **Covered by construction.** A paused or frozen layer's transport is untouched, so the seed at its held position is exact and nothing advances. `SourceCut`/`SourceReplacement`-style clock rebasing is not needed: transport is program-clock driven and the position is preserved, not rebased. |
| History/recovery | manual transactions | **Not applicable — no undoable gesture.** Adoption authors nothing a patch captures; there is nothing for undo to restore, exactly as the S7 row argued for the encode. |
| Browser/native | strict actions, accessible diagnostics | **Covered, native-only, deliberately.** The browser protocol is byte-identical; request and status remain the Y key and the stage-health HUD. The completion note now reports the adoption disposition ("adopting into N live layer(s)", or the reapply fallback with a named reason), and a successful install reports "proxy adopted live (key…)" before the HUD's standing "proxy active … vs … before" line takes over. |
| Export | same evaluated plan, no proxy in export | **Covered.** Export uses synchronous decoders and its own digest-gated resolution; nothing in this tranche is reachable from the export path. The same-branch `framemd5` A/B below is the binding proof that no pixel moved. |
| Compatibility | frozen pixels, exact prior behavior | **Covered.** The only shared-path change is the mechanical seed-dance extraction, whose error messages and control flow the performance preparer preserves byte-for-byte; live slot-switch behavior is pinned by the existing performance-runtime suite (1277/0 hosted). |

## A pre-existing edge, observed and left honest

A slot dance A→B→A around an adopted (or patch-load-adopted) proxy can
reactivate the displaced artifact-backed decoder while `proxy_backing`
reports `None`, because `commit_prepared_source` deliberately clears the
backing claim and the displaced pool retains the decoder. The HUD then
shows the measured assessment instead of "proxy active", and a Y-key
request would walk the already-cached path back to a correct re-adoption.
This staleness is S7's patch-load design, not introduced here; fixing it
belongs to a slot-aware backing claim, which is its own small tranche.

## Pixel evidence

Adapter: **AMD Radeon RX 6950 XT / Vulkan**, this host. FFmpeg 8.1.2,
`videos/audit.mp4` present. Toolchain pinned `RUSTUP_TOOLCHAIN=1.97.1`.

Same-branch A/B, minutes apart on one host: render every labeled export
case with this tranche applied, stash the tranche back to the branch point
`ae842b6`, render again, and diff decoded `framemd5`. Hot adoption must not
move a pixel — the frame must **not** move.

Result: all ten labeled export tests, **30 rendered outputs per side, every
decoded `framemd5` byte-identical** between the tranche and `ae842b6`.

One process note, recorded because it nearly weakened the evidence: the
first after-side capture was contaminated by a mid-run directory swap
(clearing stale renders while the suite was already writing), leaving 8 of
30 outputs unhashed. The comparison above is from a clean re-render of the
after side — never from the partial set.
