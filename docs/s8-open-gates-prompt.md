# S8 — the open gates

Enriched successor-session prompt, written after S7 completed. Everything
above the horizontal rule is context a successor should read once; the
prompt proper begins below it, and the appendix records where this
project's own process failed during S7 and what that cost.

**The enrichment plan's tranche list is exhausted, and that is the single
most important sentence here.** S1–S7 landed every tranche
`docs/successor-session-enrichment-implementation-plan.md` scoped: the
composition elements, the Field Collider, the gesture field, the preview
gizmo, and finally §7's content-addressed proxy — contract first, then the
worker — plus the Study authority hardening and the capability-evidence
producer. There is no pre-scoped S8 tranche. **A session that invents one
to feel productive is the failure mode this page exists to prevent.** S8
begins from a decision the operator makes, and this page is the map of the
gates, what opens each, what becomes landable behind it, and the traps
already known from reading the code.

Do not confuse the two kinds of gate below. Some are *product choices* —
nothing external blocks them, and a session may land them the day the
operator asks. Others are *evidence gates* — they need a purchase, a venue
fact, a measurement, or an ABI ruling, and a session that walks through one
uninvited will produce exactly the overclaim §7 was written to prevent.

---

## Carried forward from S1–S7 — the laws, compressed

The S7 prompt carries the full statements of the older laws; they all still
bind. One predicate, many callers. Delegate to the seam behind the guard,
not the entry point in front of it. A shared surface has an owner, written
down. Audit every resolver before adding a field that must be resolved.
Write the reproduction first and watch it fail. Prove invariance by
same-branch A/B, and prove the inverted claim too. Non-finite inputs land
on documented defaults, never clamped extremes. "Deferred" is a
deliverable. Run the gate on the toolchain CI uses (`rustup` `stable` is
fresh in CI; this host's default lags — pin with `RUSTUP_TOOLCHAIN`). Run
`cargo fmt --all` before the gate. Re-read the Known-constraints list at
the start of a tranche, not the end.

Four laws are new with S7, each paid for:

**Verify CI per suite, per named job — never by counting successes.** A SHA
can carry multiple check suites (a branch push and a PR open different
concurrency groups; both run). Counting `success` conclusions across the
flat check-runs list declared a false green at `6c06237` while the push
suite's Linux job was ten minutes into a mirror stall it would die of.
`scripts/check-ci-status.py <sha>` is the committed verdict tool: exit 0
only when a complete suite's three named jobs all succeeded and nothing is
pending. Use it for the branch point before branching and for the published
SHA before claiming.

**Split every ffmpeg-CLI feature at the CLI boundary.** Hosted CI's Unix
FFmpeg is built with `--disable-programs` — no `ffmpeg` binary exists on
the Linux or macOS runners; only Windows CI puts one on PATH. Design so
the law-shaped half (filesystem, commit ordering, refusals, caps) is
CLI-free and runs hosted on all three platforms — which is where the
platform risk lives — and the encode-integration half is opt-in
`#[ignore]` like `effects_audit`, run locally as the receipt.

**Windows fsync demands writable handles — for directories too.**
`File::sync_all()` on a read-only handle fails with Access denied, and the
parent-directory flush needs `.read(true).write(true)` plus
`FILE_FLAG_BACKUP_SEMANTICS` (0x0200_0000). Both are load-bearing comments
in `src/proxy_worker.rs`; reuse those helpers rather than re-deriving.
Both were found by hosted tests, which is the argument for the split above.

**When a validation test is easy to pass, ask what it cannot observe.**
S7's first corruption fixture flipped bytes mid-file and the artifact still
validated, because decoded-identity validation reads the first frame. The
answer was not a bigger decode — it was the seal design: publication
renames the artifact, then a sidecar carrying the SHA-256 of its exact
bytes; recovery removes unsealed residue as interrupted publication;
consumption and the cache-hit path re-hash before serving. The general
law: a validator's blind spots are design input, and the fixture that
exposes one has done more work than ten that pass.

And one law about delivery, sharpened twice in one day: **give evidence one
producer and every claim a live consumer.** `CapabilityEvidence` sat for a
milestone as a pure type whose only constructors were test fixtures — and
separately, `hardware_decode_active: false` sat in production as a bare
literal that would have stayed false forever after a backend landed. The
fix was one chain: a probe, an evaluator, and a derivation feeding the
telemetry the HUD already publishes. When you find a literal in production
that encodes a claim, ask what should be deriving it.

## The verified current state

Read this as *a* reading to re-verify, not gospel — the S5a law. At the
time of writing: `feat/web-control-panel` tip is `ae842b6`, three-platform
green by suite-aware check. The full-gate baseline there is **1275 passed /
0 failed / 89 ignored** (plus `spout_probe` 0/0/0 and the floor probe
0/0/2, never run in ordinary verification). The exact gate commands are
pinned verbatim in CLAUDE.md's Verification section, so prose and gate can
no longer drift. The proxy loop is closed and **operator-proven with a
measured result**: decode p95 fell from 62,692 µs to 7,489 µs (8.4×) on
the development host — recorded with provenance and caveats in
`docs/precision-and-scale.md`. The Study ABI's three open decisions are
documented *on the opcodes* in `src/study.rs`. The capability evaluator's
production probe defers everything, with a pinned per-platform
reason table in `src/precision.rs`.

---

## The prompt

S8 — whichever gate the operator has opened. Branch from
`feat/web-control-panel`'s tip after `scripts/check-ci-status.py` says
green; re-derive the topology with `git log` rather than trusting this
page. Land one tranche in one commit carrying its resource-delta table.
If no gate has been opened, say so and stop — do not select one
unilaterally.

### Gate 1 — proxy follow-ups (product choices; no external evidence needed)

The loop works and is measured. Three named edges remain, each a
self-contained tranche the operator may simply request:

- **Hot adoption.** Today a published proxy is adopted at patch (re)apply.
  Swapping a live layer's decoder on publication is the missing comfort.
  The trap is transport state: the threaded decoder owns a pacer,
  accumulated debt, and a mailbox; the clip-slot activation swap
  (`layers/mod.rs`, the `LayerSourceActivation` exchange) is the existing
  seam that already swaps a source under a live layer and resets
  `proxy_backing` — study it before inventing a second swap. The A/B
  telemetry gets better under hot adoption: both measurements come from
  one continuous session on one layer.
- **A browser proxy surface.** Request and status are native-only today,
  deliberately. A panel surface is a wire and protocol decision: a new
  action (priority, uncoalesced), snapshot fields for per-layer proxy
  state, and panel UI. Do not let the browser bypass the content-identity
  refusals the Y key enforces.
- **Settings beyond the default.** `ProxySettings::default()` is the only
  value ever used live. Exposing scale/frame-rate/audio choices multiplies
  cache entries per source (each settings tuple is its own key — that is
  the design, not a bug) and needs a UI story for which artifact a load
  consults. The contract already handles all of it; this is purely an
  operator-surface decision.
- **Cross-session LRU.** Ordinals are session-local by design (the
  directory is the index; there is no metadata file to corrupt). If real
  cache pressure ever appears, persisting recency is a deliberate
  trade — a mutable index file buys better eviction at the cost of a new
  crash surface. Do not build it speculatively.

### Gate 2 — the Study evaluator (three ABI rulings, then codeable)

The decisions are documented on the opcodes in `src/study.rs` and must be
made by the operator, not an evaluator session picking silently:

1. **The history-age convention** (doc on `STUDY_MAX_HISTORY_AGE`): either
   Study age maps to ring age `age - 1`, or the cap is one too large. The
   two fixes differ; the exact-equality ABI gate makes the wrong pick
   permanent.
2. **Seedless determinism** (doc on `LoadDeterministicRandom`): what a
   `domain` hashes against is undefined. Design it before any evaluator
   gives the opcode a value.
3. **The versioning law**: the ABI gate is exact equality with no window
   (doc on `StudyAbiVersion`). Any instruction-set growth needs the
   versioning story first.

Once ruled: the evaluator tranche is a pure CPU reference plus a **fixed,
pre-compiled WGSL interpreter** reading a bounded instruction buffer — the
Symmetry sector-table precedent — never shader-source generation, which
`StudyAuthority` marks permanently false. The capability table is now
compiler-enforced and value-pinned (S7), so a new opcode cannot ride in
without declaring its authority. `StudyValueType::Vector2` remains a
dead-end type; an evaluator must not "fix" that as a side effect. A young
program must guard history reads against the valid-sample count exactly as
`temporal_originals.wgsl` does. And the *distribution* half of the plan's
heading remains governance, not code: nothing about an evaluator implies a
marketplace, signing, or a license for the host.

### Gate 3 — the mesh warp freeze-half (one operator fact opens it)

The evidence required is "a demonstrated venue requirement" — an operator
writing down a real surface, its geometry, and what the current flat
output costs. That document is the key. Behind it, the *freeze* half is
landable before any renderer: caps, stable control-point identities, the
GPU byte ledger, the exact-identity bypass, and the Morph/modulation laws —
the same freeze-before-render order that made S6's gizmo possible with a
zero delta. The scoping trap is recorded in the S7 prompt's appendix and
still bites: grep the mechanism, not the noun. StageMap already owns
`StageMeshVertex`, per-slice vertex/index buffers, convexity and winding
validation, `solve_homography`, a byte ledger, and a real `draw_indexed` —
and only `PerspectiveQuad` carries a projective map; `Polygon` slices
interpolate per triangle, which is load-bearing for any warp design.

### Gate 4 — decode backends (engineering milestone, urgency now measured)

Hardware decode, zero-copy, and capture stop at `BackendNotIntegrated` —
no purchase, no external permission, just a large milestone. Two facts
temper it: the proxy loop just bought an 8.4× measured decode improvement
through pure software, and the evaluator's pinned test is the standing
reminder that a backend's capability must move through
`EvaluationRequired` with a real interoperability receipt (the S2-receipt
shape: a tracked artifact regenerated by an opt-in probe), never straight
to `Available`. The moment a backend lands, `decode_activity_claims()` in
`main.rs` begins telling the truth without an edit — that seam was built
for it.

### Gate 5 — NDI (a purchase and a policy, then Gate 4's work)

`SdkOrLicenseRequired` is the first refusal by design. An SDK license is a
purchase; network egress at a venue is an operator policy. Both must exist
as recorded authorizations before any code, and then the work is Gate 4's
shape plus a network boundary. Nothing here is a session's to grant.

### Gate 6 — Full-16 history (a measurement that needs a path built first)

The byte-exact budget has been done for a milestone
(`docs/precision-and-scale.md`: +197.75 MiB at 1080p across all 25
temporal surfaces). What is missing is "representative temporal workloads
demonstrate a documented gain" — and honestly, measuring that requires
building at least an experimental render path, which is itself a product
decision because `ExperimentalFull16History` is deliberately not an
implemented mode. Whatever happens, the settled
`AdvancedWorking16HistoryCompat8` default must not move, and the mislabeled
satellite figure that once confused this area is fixed everywhere
(398.1 MB / 379.7 MiB — the tracked docs agree now; only the untracked
`MASTER_PLAN.md` may still carry the old unit).

### Verification expectations, whatever the gate

The six-step gate on the CI toolchain, exactly as pinned in CLAUDE.md.
`scripts/check-ci-status.py` for the branch point and the published SHA.
A same-branch decoded-`framemd5` A/B whenever anything in the render,
export, or decode path is touched — both directions where the tranche
claims motion. An evidence matrix in the S6/S7 format
(`docs/evidence/`), with every not-applicable argued. And the delivery
gate in its S7 form: the thing must reach an operator, or the commit must
say in plain words that it does not yet.

---

## Appendix — where S7's own process failed, and what it cost

**A false green CI claim, caught by the operator's log, not by the
session.** The naive success-count poll declared `6c06237` green while the
push suite's Linux job was mid-timeout. The claim happened to be
*materially* right — a complete parallel suite had passed — but the method
was luck. The fix is the suite-aware script, now committed, and the law
above. The general lesson: a verification method that can be right by
accident will eventually be wrong by accident.

**The mirror stall itself.** The Linux job died in `apt-get install`
because a 28.8 MB download dribbled for ten minutes without tripping
`Acquire::http::Timeout` (it catches dead connections, not slow ones) — on
a step whose own comment records the identical failure in `apt-get update`
and bounds *that* with a `timeout`+retry loop nobody extended to the
install. When a step earns a protection, its siblings with the same
failure mode earn it the same day. A plausible-sounding external diagnosis
(swap `libclang-dev` for `libclang-18-dev`) did not survive contact with
the log: apt already resolves the meta-package to the versioned one. Read
the log before adopting a fix, however confident its author.

**The first corruption fixture passed when it should have failed** — and
that failing test produced the seal design, the best piece of the worker.
Reproduction-first is not ceremony; it is where designs come from.

The measured operator proof closed S7: recommendation → encode → sealed
publication → adoption → 8.4× decode improvement, one identity visible in
the HUD line and on disk. The plan that began with "one architectural law"
ends its tranche list with an operator pressing one key and reading a
number that used to be four times over budget. What happens next is,
properly, not this document's decision.
