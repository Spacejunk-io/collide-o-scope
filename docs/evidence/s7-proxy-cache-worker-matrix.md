# S7 — proxy cache worker: cross-cutting completion matrix

Every row of the plan's matrix, with the exact test name that proves it or an
explicit reason it does not apply. An unstated not-applicable is
indistinguishable from an unrun check, so each is argued rather than asserted.

This is the second half of §7's content-addressed proxy: the worker that
executes the bounded decode/audio input contract landed in the first half.
The sentence the program could not finish — telling an operator a proxy would
help while offering no way to make one — now completes natively: the Y key
requests an encode for the selected layer's verified content identity, the
HUD layer status reports the whole lifecycle, and patch load consults the
cache so a validated artifact backs the decoder while the layer keeps the
original's identity.

Branch point: `9e2fc07` (`feat/web-control-panel`, the S7 input-contract
merge), tree-identical to `016bbfa` where hosted Linux, macOS, and Windows CI
were verified green via the GitHub API; the merge-commit run completed green
as well. Baseline there: **1261 passed / 0 failed / 87 ignored**.

## The CLI boundary, stated once

Hosted CI's Unix FFmpeg is built with `--disable-programs`, so no ffmpeg CLI
exists on the Linux/macOS runners. The tranche is therefore split so that
everything filesystem- and law-shaped is CLI-free and runs hosted on all
three platforms — which is where the prompt said the risk lives: "path
semantics, fsync behaviour, and atomic-replace guarantees differ across the
three platforms". The encode integration itself is opt-in (`--ignored`),
exactly like `effects_audit`, and was run on this host against FFmpeg 8.1.2.

Two Windows-specific fsync laws were found by the hosted tests themselves
and are now load-bearing comments in `proxy_worker.rs`: `FlushFileBuffers`
refuses a read-only handle for both the staging file and the parent
directory, so both syncs open writable (the directory via
`FILE_FLAG_BACKUP_SEMANTICS`).

## The reproduction-first law, honored literally

The crash test was written against a recovery scan that deliberately did not
remove staging leftovers, watched to fail (`staging_removed` 0 vs 1), and
then made to pass. The same discipline caught a real design defect: the
first corruption fixture flipped bytes mid-file and the artifact still
validated, because decoded-identity validation reads the first frame. The
fix is the seal law — every published artifact carries a SHA-256 sidecar
published *after* it, consumption and the job's cache-hit path re-hash
against it, and recovery removes unsealed artifacts and orphan seals as
interrupted publications. The failing fixture became
`corruption_anywhere_in_a_published_artifact_is_refused_by_its_seal`, which
runs hosted with no CLI at all.

| Surface | Required proof | Status |
|---|---|---|
| Domain | strict version, sanitization, exact default, hostile bounds | **Covered.** All consumption questions delegate to `plan_proxy_input`, whose ladder landed with the contract. Worker-local bounds: `a_job_refuses_a_mutated_or_unreadable_source_before_encoding` (a post-verification byte change is refused before a single byte is consumed), `garbage_bytes_fail_decoded_identity_validation`, `proxy_encode_kill_bounds_are_typed_and_publish_nothing` (deadline and per-artifact size cap both kill and publish nothing, opt-in). Every refusal is a typed `ProxyWorkerError`. |
| Commit law | create-new staging, staging fsync, atomic replace, parent sync, prior readable | **Covered, hosted, all platforms.** `the_atomic_publish_law_keeps_the_prior_artifact_readable_until_replacement` — the prior artifact reads back its exact bytes while the complete replacement is staged beside it, and the seal follows the artifact. The retained-plus-staged double-count is the pure preflight's, already proven; eviction executes its plan verbatim in `eviction_follows_the_pure_plan_and_returns_a_path_free_receipt`. |
| Crash recovery | partial output never valid, written as a failing reproduction first | **Covered, hosted.** `a_crash_leftover_staging_file_is_removed_and_never_published_or_counted` — staging leftovers, unsealed artifacts, and orphan seals all removed and never counted; watched to fail before it passed (see above). The opt-in end-to-end fixture repeats the scan beside a live cache. |
| Corruption | a corrupted artifact refused rather than served | **Covered at both boundaries.** Hosted: `corruption_anywhere_in_a_published_artifact_is_refused_by_its_seal` (consultation). Opt-in end-to-end: refusal at consultation *and* at the job's own cache-hit path, with a fresh re-encode following each — a corrupt artifact can never be reported as already cached. |
| Key | content, never a path | **Covered.** The key derivation and its golden are the contract's, unchanged. The end-to-end fixture proves the operational half: identical bytes copied to a renamed, relocated path fingerprint to the same identity, hit the same key, and adopt the same artifact. No path, mtime, or filesystem metadata enters any key, receipt, or feedback note — receipts carry keys and byte counts only. |
| Eviction | deterministic by (last-used ordinal, key), receipted | **Covered, hosted.** `eviction_follows_the_pure_plan_and_returns_a_path_free_receipt` — a touch changes the outcome, the tiebreak is key order, both artifact and seal are deleted, and the receipt's totals are the plan's. Ordinals are deliberately session-local; the directory is the index, so there is no metadata file to corrupt, and a fresh process starts every entry at zero with deterministic key-order ties. |
| Helper bounds | deadline, concurrency, captured output, reservation release | **Covered.** The absolute deadline is the plan's, computed once and checked first each poll; the size poll kills at the per-artifact cap (both proven in the opt-in kill fixture). Concurrency is structural: one worker thread, a one-slot `sync_channel` that refuses while busy. Captured stdout/stderr are bounded by constants. The `MediaSafetyPolicy` reservation is RAII on the job's stack frame, exactly the thumbnail-helper shape. Cancellation is a caller-owned flag, deliberately not the library generation — a proxy is content-keyed and survives library changes. |
| Persistence | patch round trip, no runtime pixels | **Covered by exclusion, argued.** A proxy adds no `PatchState` field: the layer keeps its retained `cos-sha256` reference, so patch capture emits the original identity (`resolved_runtime_path_does_not_replace_retained_content_identity`, pre-existing). Export's digest-gated hint rejects the artifact path and re-resolves the original by content — the pre-existing export resolution tests cover that gate. `proxy_backing` is process-lifetime, reset on clip-slot activation swaps, and never serialized. |
| Look/Morph/Modulation/Dice | values-only laws, stable addresses, deterministic streams | **Not applicable — no authored creative state.** The cache is operational, not creative: no modulatable address, no Morph slot content, no RNG stream, no Dice surface. The one authored-adjacent value, `ProxySettings::default()`, is fixed and versioned under the contract. |
| Planner/GPU | resource preflight, warm allocation | **Not applicable — zero GPU resources.** No pass, texture, buffer, bind group, pipeline, or persistent surface; the resource-delta table in the commit records every audience-facing charge as zero. The cache's own budgets are the pure `ProxyCacheLimits` preflight, already proven one-over in the contract tranche. |
| Reset/freeze | patch, freezes, blackout | **Covered where state exists.** The worker's only cross-frame state is the cache itself (crash-recovered, seal-verified) and bounded session feedback maps. Patch apply *is* the adoption boundary. Freezes and blackout are not applicable: no clock, no decay, no audience surface. Quit cancels a running encode; a host killed outright orphans at most one ffmpeg child whose staged file is recovery residue — stated in CLAUDE.md rather than hidden. |
| History/recovery | manual transactions | **Not applicable — no undoable gesture.** An encode is not a creative edit; it authors nothing a patch captures. The Y key changes no authored state, so there is nothing for undo to restore. |
| Browser/native | strict actions, accessible diagnostics | **Covered, native-only, deliberately.** The proxy recommendation has only ever surfaced in the native stage-health HUD, so the request lives beside it: `y_requests_a_selected_layer_proxy_and_release_is_inert`, with every refusal ("no verified identity", "only video layers", "already proxy-backed", busy worker, unavailable cache) landing in that layer's HUD status line. No wire action, no snapshot field, no panel change — the browser protocol is byte-identical. A future browser surface is a wire and protocol decision, recorded in CLAUDE.md as an open edge. |
| Consumption wiring | the decoder consults the cache | **Covered at the seam; the call site is argued.** `consult_proxy_cache` is fully tested (adoption, no-artifact inertness, seal refusal with discard, probe-failure refusal, corrupt-original refusal) and the end-to-end fixture drives it against real artifacts. The patch-apply call site is thirty lines that select the open path and mark the backing; it is exercised only by inspection plus the seam tests, because App-level tests require a GPU adapter in this harness. That is a stated gap, not a claimed cover; the A/B telemetry line ("proxy active … vs … before") is likewise session-local and labeled as such. |
| Export | same evaluated plan, no proxy in export | **Covered.** Export resolves sources itself through the digest gate: an artifact path offered as a hint fails the recorded digest and the original is re-resolved by content — the strict content-addressed export tests already pin that behavior, and the layer's persisted reference is the original's. The same-branch `framemd5` A/B below proves the render path itself is untouched. |
| Compatibility | frozen pixels, exact defaults | **Covered.** No renderer, shader, or export line changed. The A/B below is the binding proof. |

## Pixel evidence

Adapter: **AMD Radeon RX 6950 XT / Vulkan**, this host. FFmpeg 8.1.2,
`videos/audit.mp4` present.

Same-branch A/B, minutes apart on one host: render every pre-existing
labeled export case with this tranche applied, stash the tranche back to the
branch point `9e2fc07`, render again, and diff decoded `framemd5`. A cache
worker must not move a pixel — here, as in the S5a half of that law, the
frame must **not** move.

`cargo test --locked effects_audit -- --ignored --test-threads=1` on each
side rendered all ten labeled pipelines plus the full effects matrix — 30
output files per side. Decoded `framemd5` comparison: **all 30 identical**,
including `audit_tapless_advanced_motion`, `audit_field_collider`,
`audit_symmetry_field`, both gesture cases, `audit_residual_counterpoint`,
`audit_selective_vhs_bypass`, all three gizmo files, and the complete
per-effect matrix. The cache worker does not touch the render path, and now
that is measured rather than argued.

The opt-in worker fixtures were run on this host against FFmpeg 8.1.2
(`ffv1` encoder and `matroska` muxer confirmed present):
`proxy_worker_end_to_end_encode_publish_rename_and_corruption_survival` and
`proxy_encode_kill_bounds_are_typed_and_publish_nothing`, both green. The
end-to-end fixture is the delivery proof: encode → decoded-identity
validation → sealed atomic publication → cache hit → identical bytes at a
renamed, relocated path hitting the same key and adopting → corruption
refused and re-encoded at both consultation and the job's cache-hit path →
crash recovery beside a live cache → both audio laws (first-ordered-stream
copy, and no-audio-track as the defined result for a silent source).

## Gate

Six steps in CI order, on rustc 1.97.1 (the version CI's fresh `stable`
resolves to today; the host default 1.96.1 was not used):

1. `cargo fmt --all -- --check` — 0
2. `node --check static/app.js` — 0
3. `node --check docs/ui-ux/wireframe.js` — 0
4. `cargo check --locked --all-targets` — 0
5. `cargo test --locked --all-targets -- --test-threads=1` — **1272 passed /
   0 failed / 89 ignored**
6. `cargo clippy --locked --all-targets --all-features -- -D warnings` — 0

Delta against the branch-point baseline of 1261/87: **+11 passing** (ten
hosted worker tests plus the Y-key mapping) and **+2 ignored** (the two
opt-in encode fixtures, both run locally as recorded above). Clippy on
1.97.1 caught one `explicit_counter_loop` in validation, fixed as the lint
suggests — the same lesson as S6's gate: a subset of the six steps proves
nothing. `spout_probe` 0/0/0 and `eight_texture_floor_probe` 0/0/2, both
unchanged and the floor probe not run.

A cross-platform claim for this tranche requires hosted CI at its published
SHA. The hosted suite exercises the complete CLI-free cache half — commit
law, both fsync laws, crash recovery, seals, eviction, refusals — on real
Linux, macOS, and Windows filesystems, which is precisely where this
tranche's platform risk lives.
