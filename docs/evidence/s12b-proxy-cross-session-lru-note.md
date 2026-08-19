# S12b — cross-session proxy LRU: evidence note

Gate 1's second and last remainder (S12 prompt), opened by the operator's
commission. S8 recorded the deliberate trade — "a mutable index file buys
better eviction at the cost of a new crash surface. Do not build it
speculatively" — and the commission is the decision that closes that
clause. The build keeps the law that made the cache crash-safe: **the
directory is the index**, and the new file is advisory eviction input
only.

Branch point: `b903eff` on `feat/proxy-authored-settings` (stacked on the
S12 settings tranche; mainline `69204b7` suite-green). Baseline:
**1304 passed / 0 failed / 96 ignored**; with this tranche
**1308 / 0 / 96** — four hosted CLI-free law tests, no new ignored
fixtures.

## The design, compressed

**One advisory file, the publication's own commit law.** `recency.json`
in the cache root holds `{version, touches: [[64-hex key, ordinal], …]}`
and is rewritten through the exact staged-atomic-replace idiom the
artifact publication uses (create-new staging, fsync, rename, parent
sync), so its crash residue is a `.staging` file the ordinary recovery
scan already removes, and the prior record stays readable throughout a
rewrite. Every recency-changing mutation persists it: touch, publication,
eviction, and invalid-artifact discard — which means the existing
consultation and cache-hit paths (both already call `touch`) feed it with
no new call sites.

**Advisory means advisory, enforced in both directions.** On open, a
valid record seeds `last_used_ordinal` for directory-backed keys only and
the session counter resumes above the recorded maximum; a row naming a
key the directory does not back is ignored — applied to nothing,
resurrecting nothing, not even advancing the counter. A missing, torn,
oversized (256 KiB cap, checked before parsing), wrong-version,
unknown-field, malformed-key, or duplicate-key record is discarded
*whole* — advisory data is fully believed or not believed at all — and
eviction order degrades to exactly the old session-local behavior with
`(ordinal, key)` breaking ties. The record can never refuse the cache,
and it can never bypass a seal: consumption re-hashes every artifact
regardless of how the entry was ordered. A recency *write* failure is
soft (logged, swallowed): it degrades a future session's eviction order
and nothing else, never failing a touch, publication, or eviction that
already succeeded.

**What changed for the operator.** Before: a fresh process zeroed every
ordinal, so the first eviction of a session was decided by key bytes
alone. Now the least-recently-used artifact across sessions is the one
evicted, proven end to end into the pure preflight's plan.

| Surface | Required proof | Status |
|---|---|---|
| Crash reproduction | mid-write residue removed, prior record applies, nothing lost | **Covered, hosted, written first.** `a_recency_record_interrupted_mid_write_is_removed_while_the_prior_record_applies` — staged half removed by the ordinary staging law, both ordinals applied, session counter resumes above the recorded maximum. |
| Cross-session LRU | order survives reopen and reaches eviction | **Covered, hosted.** `recency_orders_cross_session_eviction_and_survives_reopen` — publish two, touch one, reopen; the pure preflight evicts the cross-session least-recently-used, which key order alone would not have chosen. |
| Hostile records | every invalid shape degrades whole, cache never refused | **Covered, hosted.** `hostile_recency_records_degrade_to_session_local_order_without_refusing_the_cache` — torn, wrong version, unknown field, malformed key, duplicate key, oversized: each discarded, artifacts served, ordinals zero, file removed rather than retried. |
| Directory is the index | ghost rows resurrect nothing; removals rewrite | **Covered, hosted.** `recency_rows_never_resurrect_artifacts_and_removals_rewrite_the_record` — an absent key's row is ignored and does not advance the counter; a discarded artifact's row cannot re-seed a future session. |
| Seal law | recency cannot bypass consumption's re-hash | **Covered by construction, argued.** The record carries ordinals only — no paths, no digests, no admission input; `consult_proxy_cache` and the cache-hit path re-hash against the seal unchanged (existing corruption fixtures untouched and passing). |
| Foreign files | untouched | **Covered.** The scan recognizes `recency.json` as the store's own file (not foreign, not an artifact); everything else keeps the existing foreign-count law, fixtures unchanged. |
| CLI boundary | law half hosted on all three platforms | **By construction.** No ffmpeg involvement anywhere in the recency path; all four fixtures are filesystem-only, hosted like the rest of the cache half. The opt-in end-to-end encode fixtures are unmodified. |
| Render/export A/B | decoded-`framemd5` parity | **Not applicable, argued.** No render, export, or decode path file changed; the diff is `proxy_worker.rs` store metadata plus docs. Which artifact a consultation returns is unchanged — recency orders eviction only. |

Gate on `RUSTUP_TOOLCHAIN=1.97.1`: fmt, both node checks, check, tests,
clippy `-D warnings` — run on the final tree before commit.
