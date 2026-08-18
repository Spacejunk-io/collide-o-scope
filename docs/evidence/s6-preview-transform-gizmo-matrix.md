# S6 — preview transform gizmo: cross-cutting completion matrix

Every row of the plan's matrix, with the exact test name that proves it or an
explicit reason it does not apply. An unstated not-applicable is
indistinguishable from an unrun check, so each is argued rather than asserted.

Branch point: `c99e043` (`feat/web-control-panel`, with S5 `74cbd81` and S5a
`f6ceec5` both already merged). Baseline measured on that exact commit:
**1217 passed / 0 failed / 86 ignored**, plus `spout_probe` 0/0/0 and
`eight_texture_floor_probe` 0/0/2.

| Surface | Required proof | Status |
|---|---|---|
| Domain | strict version, sanitization, exact default, hostile bounds | **Covered.** `transform_gizmo::tests::a_non_finite_computation_takes_the_neutral_value_not_a_clamped_extreme`, `every_authored_value_lands_inside_the_spatial_contract`, `the_edit_set_is_bounded_and_allocation_free`, `degenerate_and_non_finite_panes_refuse`, `non_finite_and_zero_dimension_inputs_fail_closed`. **Strict version: not applicable** — the gizmo has no serialized form to version. Its authored output is `SpatialTransform`, whose domain laws are already frozen and tested in `spatial.rs`. |
| Persistence | patch round trip, selected/missing tombstone, no runtime pixels | **Not applicable — nothing persists.** No `PatchState` field, no snapshot field, and no wire action for the gizmo itself were added; the diff is the proof, and the resource table in `CLAUDE.md` records all four as zero. The *result* of a drag is an ordinary `SpatialTransform`, already covered by the existing patch round-trip tests. There is no route, so there is no tombstone to preserve. |
| Look/Morph | values-only identity preservation and endpoint-exact discrete laws | **Covered.** `app_state_tests::a_gizmo_drag_releases_morph_ownership_before_authoring` proves the load-bearing half: a drag onto Morph-owned state transfers ownership through `release_active_morph_for_manual_edit` instead of authoring under an engaged A/B pair. Interpolation and endpoint-exact discrete laws are `SpatialTransform::interpolate`'s, unchanged by this tranche. |
| Modulation | continuous bounded fields only; stable addresses | **Not applicable — no new address.** The gizmo exposes no modulatable field; it writes the same authored bases the existing `layerN_*` and master transform addresses already target. Consequence documented in `CLAUDE.md`: while a route drives a transform the rendered image sits away from the handles by exactly that offset, the same relationship the numeric editor already has. |
| Dice/generator | deterministic domain-separated streams; topology preserved | **Not applicable — no RNG.** The gizmo is pointer-driven and carries no seed, no stream, and no generated value. Dice and procedural generation continue to mutate `SpatialTransform` exactly as before; nothing here enters or perturbs their domains. |
| Planner | immutable stable routes, cycle rejection, exact resource preflight | **Not applicable — the gizmo is not a composition node.** It claims no scope, no dependency edge, and no image tap, so there is no ordering to plan and no cycle to reject. Its *selection* identity law — the analogue of a stable route — is proven by `transform_gizmo::tests::a_drag_captures_its_scope_and_cannot_retarget` and `app_state_tests::a_topology_change_during_a_drag_aborts_it_without_retargeting`. |
| Input routing | the gizmo and the S3b etch surface share one preview | **Covered.** `transform_gizmo::tests::the_gizmo_claims_only_its_handles_and_leaves_the_rest_of_the_preview_alone`. Translation is a point handle rather than the footprint body, because an untransformed source covers the whole composition and a body-sized target would have claimed every drag over the image — silently removing the gesture-etch surface. This row is not in the plan's matrix; it is here because the defect was real and was caught during implementation. |
| GPU | premult filtering/math, transaction laws, warm allocation | **Not applicable — zero GPU resources.** No pass, no sampled texture, no buffer, no bind group, no pipeline, no persistent surface. The gizmo paints into the editor window's own egui layer and never into the composition. Proven negatively by the same-branch `framemd5` A/B below. |
| Reset/freeze | patch, Look, cut, seek, resize, manual clear, both freezes, blackout | **Covered for the only state that exists.** The gizmo's sole transient state is an open drag, and `app_state_tests::a_topology_change_during_a_drag_aborts_it_without_retargeting` proves it is abandoned at `bump_layer_stack_revision` — the one barrier every topology edit crosses, including patch apply. Freeze and blackout are **not applicable**: the gizmo holds no clock, no decay, and no accumulated field, so there is nothing for a freeze to hold or a blackout to suppress. |
| History/recovery | one manual transaction; automation excluded | **Covered.** `app_state_tests::one_gizmo_drag_is_exactly_one_undo_entry` (64 moves, one entry), `a_gizmo_drag_that_authors_nothing_records_no_history_entry`, `escape_cancels_before_a_commit_and_undoes_after_one`. Automation exclusion is inherited rather than re-implemented: the drag routes through `GestureHistoryRouter`, whose `observe` refuses every automation origin before allocating an identity. |
| Browser/native | strict actions, revision barriers, accessible diagnostics | **Covered.** The gizmo adds no browser action; it emits the existing `SetMasterTransform` / `SetLayerTransform` / `ApplyLayerTransform`, whose strictness and stale-ID rejection are already covered by `stable_transform_actions_follow_reorder_and_noop_after_target_deletion`. The revision barrier is `app_state_tests::a_topology_change_during_a_drag_aborts_it_without_retargeting`. Native diagnostics land in `transform_gizmo_status` and are asserted there. |
| Export | same evaluated plan, explicit initial state, bounded provenance | **Covered.** `render_export::effects_audit::render_native_gizmo_transform_pipeline` renders a gizmo-authored transform, its numerically-authored twin, and the untouched identity. There is no export-only gizmo path to exercise, because the gizmo exists only in the editor preview and what crosses into a patch is an ordinary `SpatialTransform`. |
| Compatibility | exact zero/default Legacy delegation and frozen pixels | **Covered.** `transform_gizmo::tests::hit_testing_never_authors_anything` and `app_state_tests::hovering_a_gizmo_leaves_a_legacy_patch_on_the_historical_sample` prove that opening, hovering, and hit-testing leave `spatial_modes.w == 0`, so a patch nobody moved still renders through the exact historical sample. Frozen pixels are proven by the A/B below. |

## Pixel evidence

Adapter: **AMD Radeon RX 6950 XT / Vulkan**, this host. FFmpeg 8.1.2,
`videos/audit.mp4` present.

### The delivery claim, both halves

`cargo test --locked render_native_gizmo_transform_pipeline -- --ignored`
renders three files; decoded `framemd5` comparison:

| Comparison | Result | Meaning |
|---|---|---|
| gizmo-authored vs numerically-authored | **identical** | the two authoring surfaces are indistinguishable in pixels |
| gizmo-authored vs untouched identity | **differs** | the drag genuinely reached the audience image |

The second comparison is what makes the first discriminating: without it, a
gizmo that authored nothing at all would still produce two identical files and
read as a pass.

### The inverted claim: the frame must *not* move

Same-branch A/B, minutes apart on one host and one adapter — render every
pre-existing labeled export case with this tranche applied, check out the
parent `c99e043`, render again, diff decoded `framemd5`. A commit cannot cite
its own hash, so the tranche side is named by its parent rather than by a SHA
that changes when this file is folded in:

| Labeled case | Result |
|---|---|
| `audit_displace_two_input` | identical |
| `audit_field_collider` | identical |
| `audit_gesture_canvas_displace_donor` | identical |
| `audit_gesture_field_etching` | identical |
| `audit_residual_counterpoint` | identical |
| `audit_selective_vhs_bypass` | identical |
| `audit_tapless_advanced_motion` | identical |

All seven are decoded-frame identical across the tranche. This is the check a
permit cannot satisfy by merely happening to be unused today.

## Gate

Six steps in CI order, all green with this tranche applied:

1. `cargo fmt --all -- --check` — 0
2. `node --check static/app.js` — 0
3. `node --check docs/ui-ux/wireframe.js` — 0
4. `cargo check --locked --all-targets` — 0
5. `cargo test --locked --all-targets -- --test-threads=1` — **1256 passed / 0 failed / 87 ignored**
6. `cargo clippy --locked --all-targets --all-features -- -D warnings` — 0

Delta against the branch-point baseline: **+39 passing, +1 ignored** (the new
labeled export case). `spout_probe` 0/0/0 and `eight_texture_floor_probe` 0/0/2
are unchanged; the floor probe was deliberately **not** run, because it rewrites
its tracked receipt and would dirty a tree whose cleanliness this gate claims.

## What is not proven here

- **Cross-platform CI.** Only Windows was run, locally. Linux and macOS hosted
  CI at this exact SHA remain required before any publication claim; `gh` is not
  installed on this host, so CI could not be launched or inspected from here.
- **A physical operator.** Hit testing, the drag laws, and the permit are proven
  in software. A person dragging a handle on a real pointer or tablet is
  hardware proof and is not transferable from these tests.
- **Layer-scope host wiring at runtime.** The layer arm of
  `transform_gizmo_frame` resolves through `resolve_stable_layer_id`, the same
  resolver the browser transform actions use and which is already covered, but
  an `App` fixture cannot construct a `Layer` without a real decoded media
  source, so the host-level drag tests exercise the master scope. The
  scope-capture and no-retarget laws are proven directly in
  `transform_gizmo::tests`.
