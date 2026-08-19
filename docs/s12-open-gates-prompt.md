# S12 — the opened gates

Enriched successor-session prompt, written after S11 completed. Everything
above the horizontal rule is context a successor should read once; the
prompt proper begins below it. Gate numbering throughout is S8's
(`docs/s8-open-gates-prompt.md`), kept deliberately so the two maps can be
read side by side.

**This time gates have been opened, and that is the single most important
sentence here.** S8 ended with "S8 begins from a decision the operator
makes" — and the decisions have now been made. The operator's commission,
recorded verbatim as the authorization this project's own rules require:

> The nimble and busy Da Vinci encodes in a most adroit manner that of
> gates 4, 6, as well as gate 1 remainders.

Three gates are therefore open: **Gate 4** (decode backends), **Gate 6**
(the Full-16 history measurement, which is first a product decision to
build the experimental path), and **Gate 1's remainders** (proxy settings
beyond the default, and cross-session LRU — the two edges S8 named that
S8/S8b/S8c/S9 did not consume). **Gates 3 and 5 remain closed** and no
sentence here reopens them: the mesh warp still waits on a written venue
requirement, and NDI still waits on a purchase and a network policy —
`SdkOrLicenseRequired` first, by design. A session that walks through
either uninvited is the failure mode the S8 page exists to prevent.

Gate 2 is consumed and gone: the operator's R1/R2/R3 rulings (PR #23),
the CPU reference (S10a), the fixed WGSL interpreter (S10b), and the
authored Study rack node (S11) closed the whole arc. Do not reopen it to
"improve" the evaluator as a side effect of other work; the ABI window law
(`major == current && minor <= current`) and the append-only GPU opcodes
are frozen surfaces now.

---

## Carried forward from S1–S11 — the laws, compressed

The S7 and S8 prompts carry the full statements of the older laws; they
all still bind. One predicate, many callers. Delegate to the seam behind
the guard, not the entry point in front of it. A shared surface has an
owner, written down. Audit every resolver before adding a field that must
be resolved. Write the reproduction first and watch it fail. Prove
invariance by same-branch A/B, and prove the inverted claim too.
Non-finite inputs land on documented defaults, never clamped extremes.
"Deferred" is a deliverable. Verify CI per suite, per named job
(`scripts/check-ci-status.py`), never by counting successes. Split every
ffmpeg-CLI feature at the CLI boundary (hosted Unix CI has no `ffmpeg`
binary). Windows fsync demands writable handles, for directories too.
When a validation test is easy to pass, ask what it cannot observe. Give
evidence one producer and every claim a live consumer. Run the gate on
the CI toolchain (`RUSTUP_TOOLCHAIN` pinned; this host's default lags).
Run `cargo fmt --all` before the gate. Re-read the Known-constraints list
at the start of a tranche, not the end.

Four more are new since S8, each paid for:

**A claim is captured once and re-validated at the drain, always.** Hot
adoption (S8) and identity minting (S9) both live or die on the same
shape: stable layer ID plus source-resource epoch captured at submit,
every part re-checked before anything lands, each failure a distinct
named discard. Any new worker that completes asynchronously into live
state — and Gate 4's decode backends are exactly that — uses this shape,
not a fresh one.

**Both surfaces improve without a wire change when the capability lives
in the shared ladder.** S9 put minting inside `request_proxy_for_layer`,
and the browser gained it without gaining an action. Before adding a wire
action for a new capability, ask whether the existing ladder is the right
owner.

**Anchor semantics to an existing law; never invent one beside it.**
S10a's R1 guard mirrors `temporal_originals.wgsl`; its R2 hash reuses
`symmetry.rs`'s SplitMix64 finalizer; its hue law is `rack_node.wgsl`'s,
asserted by source text. Where two implementations must agree, one is
designated the reference and the other is *checked against it* — and a
shared-by-source-text assertion is cheaper than a drifted copy.

**Bump a version the day you learn the law, before consumers exist.**
S10b found the hue-operand domain clamp days after semantics v1 and
bumped to v2 immediately, while the window was free. The Gate 1
settings tranche inherits the mirror of this: `PROXY_ALGORITHM_VERSION`'s
free window closed the day the first artifact was published — behavior
changes there now invalidate real caches and must be versioned honestly.

And one design lesson from S11 worth its own line: **a failing planner
test is design input.** The Study descriptor first declared the ABI worst
case (65 loads/pixel), instantly broke the 32-lookup rack budget, and the
*failure* produced the better design — an eight-load admission budget
with a named plan-time refusal (`StudyLoadBudget`). The fixture that
breaks a first design has done more work than ten that pass it.

## The verified current state

Read this as *a* reading to re-verify, not gospel — re-derive the
topology with `git log` and the suite-aware check rather than trusting
this page. At the time of writing: S11 (the Study rack node) sits on
`feat/study-authored-surface` at `909304e`, gate-green locally at
**1297 passed / 0 failed / 96 ignored** (plus `spout_probe` 0/0/0 and the
floor probe 0/0/2, never run in ordinary verification), awaiting its
merge into `feat/web-control-panel`. The proxy loop is closed on all
four S8 edges that have been consumed — hot adoption (S8), slot-backed
claims (S8b), the browser surface (S8c), identity minting (S9) — and is
operator-proven twice over: 8.4× decode p95 on the first hand-run loop,
and a 4,698,340 µs → 54,726 µs settle on the deliberately pathological
live hot-adoption observation (`docs/precision-and-scale.md`, both dated
observations). The capability evaluator's production probe still defers
everything, with the per-platform reason table pinned in
`src/precision.rs`; `decode_activity_claims()` in `main.rs` derives the
HUD's hardware-decode/zero-copy claims through it, so those lines are
theorems today and start telling the truth the moment a backend lands.

---

## The prompt

S12 and its successors — the three opened gates, one tranche per session,
one commit per tranche carrying its resource-delta table. Branch from
`feat/web-control-panel`'s tip after `scripts/check-ci-status.py` says
green — which means after the S11 merge lands there. The suggested order
is smallest honest step first: the Gate 1 remainders (self-contained
product tranches), then Gate 6 (an experimental path plus a measurement),
then Gate 4 (the milestone). The order is advisory; the boundaries are
not. Do not braid two gates into one commit.

### Gate 1 remainders — proxy settings and cross-session LRU (opened)

**Settings beyond the default.** `ProxySettings::default()` is consumed
at exactly four production sites in `main.rs` today — the patch-load
consultation, the hot-adoption consultation, and both request-submit
paths (verified and mint) — plus the worker's own plan derivation. An
authored settings value must reach *all of them from one owner*, or the
program will encode under one settings tuple and consult under another
and the cache will honestly report a miss for an artifact it holds. That
is the trap S8 named as "a UI story for which artifact a load consults,"
made concrete: each settings tuple is its own cache key **by design**,
so the choice of what to consult is a real product decision, not a
lookup bug. Things already true that the tranche must not re-derive:
`ProxySettings::validate` already bounds fixed frame rates (240 fps cap,
nonzero terms); the settings hash into the key through
`update_cache_key`; scale halving already floors to even for
chroma-subsampled formats. Things to decide deliberately: where the
authored value lives (a host-session preference like the new-layer fit
default, or per-request — a proxy is content-keyed and survives library
changes, which argues host-session); that it must never enter a patch (a
proxy can never enter a patch, and its settings follow); and what the
wire carries (the S8c action `request_layer_proxy` is stable-ID-only —
extending it versus a separate settings action is the same
priority/uncoalesced discipline either way). The golden cache-key test
(`cache_key_is_path_independent_settings_sensitive_and_has_a_golden`)
must stay byte-unchanged: exposing settings changes no key derivation,
only which tuple the operator selects.

**Cross-session LRU.** S8 said "do not build it speculatively"; the
operator's commission is the recorded decision that closes that clause.
Build it as the deliberate trade S8 described, keeping the law that made
the cache crash-safe: **the directory is the index.** A scan must remain
the authority that rebuilds the cache; any persisted recency data is
advisory input to eviction order only. A corrupt, missing, truncated, or
foreign recency record degrades to exactly today's behavior
(session-local ordinals, key order breaking ties) — never a refused
cache, never a served artifact the seal check did not pass. The seal
law is untouchable: consumption re-hashes before every adoption, and
recency metadata must not become a second file whose absence or
corruption can make a sealed artifact unservable. Write the crash
reproduction first, in the S7 shape: a recency record interrupted
mid-write beside a healthy sealed cache, recovered without loss.

### Gate 6 — Full-16 history (opened; the product decision is made)

The operator's opening *is* the explicit product decision that
`ExperimentalFull16History` has been waiting for — and the explicit
product decision CLAUDE.md's Symmetry-Field section demands before any
RGBA16F full-frame history allocation. Scope it exactly: the candidate
upgrades the storage of the **existing** 25 committed temporal surfaces
(24 clean-history layers plus feedback) from Compat8 to RGBA16Float. It
is not a second ring; the byte-exact budget is already done
(`docs/precision-and-scale.md`: +207,360,000 bytes / +197.753906 MiB at
1080p, fixture total 522.082031 MiB). Two constraints the S8 page set
still bind absolutely: **the settled `AdvancedWorking16HistoryCompat8`
default must not move**, and the experimental path is an opt-in
evaluation mode, not an authored patch value — media-safety-mode
precedent: process-local, absent from patches, Safe/settled again in a
new process.

The traps, from reading the code as it now stands:

- **The ring has four reader families, not one.** Since S8 was written,
  the committed clean-history ring gained consumers: the temporal
  shaders and History Key, the Symmetry Field's `CleanHistory` sector
  lane, and the Study interpreter's `LoadHistoryColor` — the last two
  documented as binding "the committed Compat8 ring" specifically. An
  experimental Full-16 ring changes what every one of them reads. The
  value domain is the sharp edge: Compat8 history stores sRGB-encoded
  bytes decoded at load, RGBA16Float stores linear — every reader must
  observe identical linear-light values on both paths, and the no-dither
  history conversion law (dither is presentation-only; temporal memory
  never accumulates it) must survive, or the A/B measures the bug
  instead of the precision.
- **The measurement is the deliverable, and its vocabulary exists.**
  "Representative temporal workloads demonstrate a documented gain" runs
  through `measure_precision` and `ArtisticGainAssessment` — verdicts
  are "measured gain," "tradeoff," "no measured gain," or "regression,"
  never a subjective declaration. The M6 GPU receipt with its
  temporal-feedback and 24/30/60 parity companions is the fixture shape;
  extend that receipt family rather than minting a new evidence format.
  An honest "no measured gain" closes this gate as truly as a gain does
  — the settled default is already good, and proving *that* is a
  deliverable too.
- **The ledger, not beside it.** `RuntimeResourceLedger::reconcile`
  requires planned creative bytes to equal the physical allocation
  snapshot with checked arithmetic; the experimental surfaces enter that
  reconciliation and the resource preflight, and the independent 320 MiB
  selective-VHS budget does not move. The commit carries the exact
  resource-delta table, both modes.

### Gate 4 — decode backends (opened; the largest, walked last)

Hardware decode, zero-copy, and capture all stop at
`BackendNotIntegrated` today, and the seams built for this day are
load-bearing: `probe_capability_evidence`'s `backend_integrated` is a
deliberate source-tree constant so **the tree change that integrates a
backend is the same change that flips it**; the evaluator then yields
`EvaluationRequired(InteroperabilityProof)`, and only a real receipt in
the S2 shape — a tracked artifact regenerated by an opt-in probe, with
adapter, backend, exact command, and hash — moves it to `Available`.
Never flip evidence fields straight to proven; the per-platform reason
table in `src/precision.rs` is pinned by test precisely so an accidental
flip cannot ship silently. When the flip is earned,
`decode_activity_claims()` and the HUD begin telling the truth without
an edit — that is the payoff of the S7 evidence law, collect it.

Sequencing inside the gate: hardware decode first (Windows/D3D11VA is
this host's provable half; the probe already answers `platform_supported`
per target), zero-copy second (it composes with decode and inherits the
Spout worker's thread-affine DX11 discipline), capture third (its own
tranche; same shape, different device story). Traps known before any
code:

- **Export stays software until proven otherwise.** Hardware decoders
  are not obliged to be bit-exact with software decode across vendors,
  and export's contract is two independent renders with equal decoded
  `framemd5`. Hardware decode is therefore a *live-path* capability
  first; the offline renderer keeps its synchronous software decoders
  unless a per-adapter receipt proves bit-exactness — and "the same
  patch exports differently on different GPUs" is never an acceptable
  trade for offline speed. Say this in the commit; it is a boundary, not
  a deferral.
- **The threaded-decoder contract is the seam, not a victim.** The
  request-driven decoder owns the pacer, bounded advances, the
  latest-only mailbox, and the first-frame seed law. A hardware backend
  slots in behind `video/decoder.rs`'s decode core — same request
  protocol, same RGBA publication contract — or, for zero-copy, defines
  a new *explicit* upload path with its own recoverable error scopes.
  Fallback on backend failure is soft and per-layer, the audio-device
  shape: refuse or fall back to software with a truthful status, never a
  reopen loop.
- **The proxy loop changed this gate's economics; measure against it.**
  Software already bought 8.4× and ~85× p95 improvements on this host.
  A backend's receipt should therefore include the same decode-p95
  telemetry the proxy A/B uses, on a long-GOP source, against both the
  original and its FFV1 proxy — hardware decode and the proxy compose
  (the proxy is intra-only and software-friendly; hardware favors the
  long-GOP original), and the honest claim is which tool wins where,
  not that a backend landed.
- **Completion into live state is a claim.** A backend that opens
  asynchronously adopts into a playing layer through captured,
  drain-revalidated claims — the S8/S9 shape, already twice proven.

### Verification expectations, whatever the gate

The six-step gate on the CI toolchain, exactly as pinned in CLAUDE.md,
run on the final tree. `scripts/check-ci-status.py` for the branch point
and the published SHA. A same-branch decoded-`framemd5` A/B whenever
anything in the render, export, or decode path is touched — for Gate 4
and Gate 6 that A/B is the heart of the tranche, both directions: the
default path byte-identical across the change, and the experimental or
hardware path's difference measured and stated. An evidence matrix in
the established format (`docs/evidence/`), with every not-applicable
argued. And the delivery gate in its S7 form: the thing must reach an
operator, or the commit must say in plain words that it does not yet.

---

## Appendix — process notes since S8, kept honest

S9's live QA absorbed one forced termination at teardown (recorded in
its note); the next launch's recovery scan held both sealed artifacts —
the crash-shaped design doing its job, worth keeping in mind when Gate 1
adds a recency record to that directory. S11's worst-case-versus-budget
descriptor failure is written above as a law because it was cheap this
time; Gate 4 is where it would be expensive. And the S8 appendix's
verification lesson has not aged: a method that can be right by accident
will eventually be wrong by accident — which is why every gate above
ends in a receipt, a measurement, or a named refusal rather than a
landing announcement.
