# Group transform authoring — action, panel, and gizmo proof note

Date: 2026-08-27  
Topic: `feat/group-transform-authoring`  
Pinned integration base: `0b5584eb0ef3a1bd139a5560804bdf5272dbcda4`

This tranche closes perfection handover §3.6 in its required logical order:
dedicated action family, canonical browser panel, then explicit stable-ID gizmo
scope. It adds no authored field: `RuntimeGroup::transform` remains the one
patch, Morph, Look, Dice, modulation, controller, recorder, renderer, and
offline-export owner.

## Dedicated action and revision law

The wire family is:

- `set_group_transform { group_id, param, value, composition_revision }`;
- `reset_group_transform { group_id, composition_revision }`;
- `apply_group_transform { group_id, transform, composition_revision }`.

The two session-only gizmo-selection actions are
`target_group_transform_gizmo { group_id, composition_revision }` and
`clear_group_transform_gizmo_target { composition_revision }`.

Every action requires a nonzero decimal stable group ID and the exact current
composition revision. Main checks the revision before group lookup, staging, or
Morph release. A current-revision missing ID is an exact no-op; a stale revision
is a typed refusal. Value edits stage the whole creative graph, run the full
planner, and commit with `topology_changed = false`, so an accepted transform
does not advance its own revision.

`SetGroupTransform` coalesces by group ID plus exact transform parameter.
Reset and Apply are ordered barriers. The beat latch uses the same key. The
pre-dedicated generic `SetCompositionGroupParam` transform spelling remains a
revision-guarded compatibility alias with that same key; it is not a stale
topology backdoor. Non-transform group scalar behavior is unchanged.

Every ingress and action-classification site was revisited. The three authored
actions remain ordinary, manual-history edits. Apply Look drops all three when
the Look owns that stable group. Only Set is quantizable. Reset, Apply, Target,
and Clear are all rejected inside a Quantized wrapper rather than falling
through to immediate execution.

## Permanent performance and controller identity

No recorder vocabulary was appended. `PerformanceControl::GroupParam` retains
permanent numeric code 18 and token `group_param`; its engine-owned value law
already delegates transform parameters to the spatial table. A live
`SetGroupTransform` records into code 18, and live playback of a transform
parameter emits the dedicated action. Stored-v1 takes therefore decode and
replay unchanged. Offline export keeps using the same group owner applier.

Controller profiles retain `RuntimeControlAddress::Group`; their five spatial
parameters now emit `SetGroupTransform`, and reverse feedback recognizes the
same stable address. No schema, positional fallback, modulation address, RNG
stream, or capability row was added.

## Canonical group panel

The former duplicate group editor was removed. Group cards now reuse the
master/layer transform machinery and exact engine-aligned ranges for all 16
fields, linked scale, reset, copy/paste, presets, range-history behavior,
status live region, and accessibility labels. The shared crop maximum is the
exact engine ceiling `1 - 1/4096` (`0.999755859375`), with exact `1/4096`
steps and post-rounding clamping so numeric entry can never turn it into `1`.

Every callback re-resolves the current group DTO from `latestCreative` using
the card's stable `data-group-id`; it never closes over an old transform
snapshot. Ordinary value packets call `syncTransformPanel` on the existing
card, preserving focus and disclosure state. Deleted group UI state is pruned
by stable group ID. The existing literal range law remains HTML 208 / JS 24.

The group toolbar adds `Target gizmo`. It sends the session-only
`target_group_transform_gizmo` action with the current stable group ID and
composition revision. A persistent control outside generated group cards sends
`clear_group_transform_gizmo_target`, so the ordinary master/layer gizmo can be
recovered even after the targeted group was deleted. Both actions are strictly
non-authored, non-recorded, non-history, unquantized, and rejected inside a
Quantized wrapper. Duplicate group names remain accessible because transform
region labels include the stable `(#id)` discriminator.

## Explicit group gizmo scope

`GizmoScope::Group(GroupId)` is an immutable stable identity. The session target
stores `(GroupId, composition_revision)`, and the host admits that scope only
when both the witness revision and group remain current. It never infers a
group from a selected member layer. A missing, deleted, or old-generation
target resolves to no scope, never to master, a layer, a reused numeric ID, or
a member position. Only the explicit Clear action restores master/layer scope.

Group geometry uses `GizmoFrame::new(group.transform, output, output)`, exactly
matching the renderer's group `MaterializeSpatial` step. Moves and nudges emit
the dedicated Set action. A group drag captures the exact composition revision
at Begin and reuses that stamp for every Move. Pre-commit Escape dispatches no
authoring action at all, so an untouched drag cannot materialize or clear an
active Morph. Retarget, Clear, and topology/revision barriers abandon an
uncommitted drag but first close a committed drag normally. One live mutation
therefore remains inside exactly one undo entry; an unchanged drag remains no
entry.

## Selective-VHS and planner boundary

A nonidentity group transform remains its own scope-local `MaterializeSpatial`
step. It does not alter advanced topology signature, canonical dry/wet layer
membership, Temporal path, or master-bypass ordering. The identical grouped
fixture still produces the same typed `AmbiguousMasterBypass` refusal before
and after the transform. No selective worker, VHS routing law, or historical
sample changed.

## Observed proof

Focused hosted contracts observed before the exact gate:

- `cargo check --locked --all-targets --all-features` — passed;
- `node --check static/app.js` — passed;
- protocol coalescing/barrier, every-field sentinel, and strict ingress suites
  — passed;
- canonical spatial-transform browser/accessibility/stable-ID contract —
  passed, including clear-after-delete recovery, duplicate-name ID labels, and
  exact crop normalization; literal ranges remained HTML 208 / JS 24;
- `cargo test --locked group_transform_ -- --nocapture` — 6 passed, covering
  exact-revision action behavior, Morph release, controller/feedback identity,
  Look/quantized identity, selective-VHS ordering, and the unchanged typed
  ambiguity refusal;
- the native `gizmo` slice — 39 passed and only its pre-existing
  GPU/`videos/audit.mp4` seat was ignored; this includes numeric/gizmo byte
  identity, one-entry history, committed-barrier settlement, Morph-inert
  precommit Escape, revision-witness refusal across same-ID generations,
  deleted-target fail-closed behavior, explicit recovery, and output/output
  geometry;
- `v2_hosted_arm_record_commit_compile_and_replay_roundtrip` — passed with the
  dedicated group transform recorded and replayed through permanent code 18.

The protected root artifacts were not read into, copied into, renamed by, or
added to this tranche. `videos/audit.mp4` remained absent and was not minted;
the pure planner regressions are the reproducible selective-VHS proof.

## Closing fields

- Topic implementation commit: **PENDING**
- Topic receipt commit: **PENDING**
- Integration commit on `feat/web-control-panel`: **PENDING**
- Exact-commit CI: **PENDING**
- Hosted full gate: **OBSERVED PASS** — the exact six-command CI-form gate
  passed: formatting and both JavaScript parsers; all-target/all-feature
  compile; 2,143 tests passed with zero failures and 163 explicitly ignored
  external/GPU seats; all six bench harnesses reported success; clippy passed
  with `-D warnings`
- Protected-root and `videos/audit.mp4` recheck: **OBSERVED PASS** — the three
  named root artifacts remain the only untracked root files, with their exact
  lengths and SHA-256 values unchanged at 66,225 / `494b63ad...ab1eea4`,
  56,984,527 / `ee1cfc47...13d034a0`, and 60,528,641 /
  `2b51dda2...722630a4`; `videos/audit.mp4` remains absent

## Deliberate non-claims

This tranche does not claim a physical pointer/tablet receipt, a new audience
overlay, a second transform owner, topology recording, a new recorder code,
or a persisted gizmo selection. It does not convert the generic group scalar
family into topology edits, make transform values bump composition revision,
or weaken the operator-gated status of any earlier hardware receipt.
