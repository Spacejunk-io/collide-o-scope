# S9 — identity minting: evidence note

The proxy loop's last honest edge, closed under the operator's two rulings:
a proxy request on a path-based video layer now mints its verified
`cos-sha256` identity instead of refusing, and — the persistence ruling —
the minted identity enters `persisted_source_reference`, so the next patch
capture emits the content reference exactly as generation would have
recorded it.

Branch point: `67ee58e` (`feat/web-control-panel`, the S8c browser-surface
merge), verified green by the suite-aware check. Baseline: **1278 passed /
0 failed / 91 ignored**; with this tranche **1280 / 0 / 92** (two hosted
mint-law tests, one opt-in end-to-end fixture).

## The design, compressed

**One fingerprint law, one more caller.** `mint_source_identity` is the
same bounded `FingerprintSession` machinery every resolver uses, with the
worker's cancel flag. The worker resolves a request's identity mode first —
`Verified` carries the retained reference; `Mint` fingerprints and reports
`IdentityMinted` with its claim — and then both modes run the unchanged
`run_proxy_encode_job`, whose own re-fingerprint means the mint-to-encode
window carries the same byte-change refusal the verified path has always
had. One encode, one refusal ladder, one publication law.

**Claims, again.** The mint claim is the adoption claim's shape — stable
layer ID plus source-resource epoch — and the drain re-validates every part
of it before the identity lands: a stale epoch, vanished ID,
already-identified, or already-backed layer is discarded with a named
reason. The encode continues regardless; an unlanded identity simply leaves
no layer for the completion to adopt into.

**Both surfaces improve without a wire change.** The Y key and the panel's
Encode proxy button call the same `request_proxy_for_layer` ladder, whose
identity-less arm now submits a mint instead of refusing. The browser
gained the capability without gaining an action.

| Surface | Required proof | Status |
|---|---|---|
| Mint law | fingerprint-identical identity, typed refusals | **Covered, hosted.** `mint_source_identity_matches_the_fingerprint_law_and_refuses_unreadable_sources` — minted identity equals `FingerprintSession`'s byte-for-byte, unreadable source is a typed `SourceUnreadable`. |
| Event order | minted-with-claim before any outcome; no fabricated identity | **Covered, hosted.** `a_mint_request_reports_its_minted_identity_with_the_claim_before_any_encode_outcome` — non-media bytes mint then refuse at the probe (`IdentityMinted` → `Started` → `Failed`, claim verbatim); an unreadable source is one layer-keyed `MintFailed` with nothing started. |
| End to end | mint → encode → publish under the minted key; re-request hits cache | **Covered, opt-in, run on this host.** `proxy_identity_mint_end_to_end_encodes_and_hits_the_cache_under_the_minted_key`. |
| Persistence ruling | minted identity enters patch capture | **Covered live.** See below: a layer added as a plain path captured to a patch whose `source_path` is the minted `cos-sha256://…` reference. |
| Staleness | a stale mint claim never lands on another layer | **Covered by construction, argued.** Identical mechanism to hot adoption's claims (S8 matrix row), same epoch/ID invalidation routes; the drain guards each produce a distinct named discard. App-level tests need a GPU adapter in this harness — the same stated seam gap as S7/S8. |
| A/B baseline | "before" half captured once an identity exists | **Covered by placement.** Verified requests capture at submit as before; mint requests capture at the drain, from the same live telemetry, keyed by the minted sha. |
| Render/export | decoded-`framemd5` parity | **Not applicable, argued.** The diff touches `proxy_worker.rs` and `main.rs` request/drain plumbing only; no render, export, or decode path file changed, and export's digest-gated resolution is unchanged — a minted reference resolves exactly as a generated one always has. |

## Live QA on this host

The marquee case is the one the program used to refuse. `audit.mp4` added
from the library panel as a plain path-based layer (no identity, no
positional patch), one click on Encode proxy: the engine minted
`cos-sha256://e8b8d433…/283134`, found no artifact for that identity,
encoded fresh, published `2502a420….mkv` (670,454 bytes plus its seal), and
hot-adopted — the card settled at `proxy active (2502a420…)` with the
control disabled and `proxy_note` reading "proxy adopted live". A
`quick_save_patch` then captured the stack, and the saved YAML carries
`source_path: cos-sha256://e8b8d433…/283134` for that layer — the
persistence ruling, on disk. (The key differing from S7's `bfc1add0…` is
itself evidence: the S7 artifact belonged to a different `audit.mp4` copy's
bytes, and the mint pinned the file actually playing.)

Session note, recorded honestly: the app absorbed one forced termination at
QA teardown after the second Escape failed to land — minutes after the
encode completed, with nothing staged; the next launch's recovery scan is
the designed answer to exactly this, and the cache held both artifacts
sealed.

Gate on `RUSTUP_TOOLCHAIN=1.97.1`: fmt, both node checks, check, **1280 /
0 / 92**, clippy `-D warnings` — green.
