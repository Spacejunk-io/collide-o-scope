Enriched successor-session prompt for §7 of
`docs/successor-session-enrichment-implementation-plan.md`, carrying forward the
laws S1–S6 established and the traps they cost time on. Everything above the
horizontal rule is context a successor should read once; the prompt proper
begins below it, and the appendix records the places this project's own
documentation has been wrong — including three found while writing this page,
one of which was actively contradicting shipped code.

Written after S6 landed. **§7 is not shaped like S1–S6, and that is the single
most important sentence here.** The previous six tranches each asked for one
creative element to be built, and the only real question was how well. §7 asks
for five *capabilities*, and four of them are gates a coding session must not
walk through. The section exists precisely to stop a schema, a menu item, a
compile flag, or an evaluator from being reported as a working feature. A
session that reads §7 as "implement five things" will produce exactly the
overclaim it was written to prevent, and will do so while feeling productive.

Read the landability table before you read anything else.

---

## Carried forward from S1–S6 — read once, then apply throughout

These are not restatements of §7. They are the disciplines the previous
tranches paid for, phrased so this session inherits them instead of
rediscovering them. Four are new since the S6 prompt and are marked.

**One predicate, many callers — not many identical predicates.** S5 wrote a
shared decision once as a method on the authored params and had all three
call sites call *that*, so drift became impossible by construction rather than
by review. S6 found the same shape in the leakage boundary: four separate
`const fn`s each computing `!output_on_main`, about to become five. They now all
answer from `stage_map::native_controls_visible`, and the gizmo's paint permit
answers from it too. For §7 the analogue is sharp and immediate: **there must be
exactly one function that decides whether a capability is available**, and every
surface that reports, gates, or enables must call it. Two places that both
decide "can we do hardware decode" will disagree, and the disagreement will be
resolved in favour of whichever one is more optimistic.

**Delegate to the seam behind the guard, not the entry point in front of it.**
*(New with S6, and the sharpest thing it found.)* The gizmo needed to author
transform values that the browser already authors. Dispatching through the
obvious public entry point, `App::handle_web_action`, compiled, ran, and
silently did nothing: that function's open-gesture guard rejects edits while a
`NativeManual` gesture is active, and it keys on the *gesture's* origin rather
than the *action's*, so it cannot tell a native surface's own action from a
remote one. The correct seam was `handle_web_action_inner_with_feedback` — the
one the browser-gesture arm itself takes, behind the guard. Taking that seam
also picked up `release_active_morph_for_manual_edit` for free, which is the
difference between transferring Morph ownership and authoring underneath an
engaged A/B pair. The generalisation: when a new surface must do what an
existing surface already does, find the point the existing path reaches *after*
its guards and call that. Read the guard before you route around it, and read
what else lives between the guard and the work.

**A shared surface has an owner, and it must be written down.** *(New with S6.)*
The native preview was already the S3b gesture-etch canvas. The first cut of the
gizmo made translation a footprint-body drag — and an untransformed source
covers the entire composition, so the gizmo would have claimed every drag over
the image and silently removed etching altogether. Nothing would have failed; a
feature would just have quietly stopped existing. Translation became a point
handle, the two now share one `egui::Response` with the routing written down
explicitly rather than left to egui's own resolution, and a named test asserts
that a pointer not on a handle reports no hit. Whenever a tranche adds a second
consumer to an existing input, an existing worker, an existing queue, or an
existing budget, **name the owner in code and pin it with a test that fails if
the new consumer takes everything.**

**Audit every resolver, not just the obvious one.** S5 added two donors to a
block that already had one. The live path resolved all three; a hand-rolled
`resolved_export_motion` bound only the original by hand, so no offline render
ever resolved a new one — silently, with the render succeeding and simply doing
the wrong thing. Before adding any field that must be *resolved*, grep for every
site that resolves its siblings and make them all delegate to one resolver.

**A law stated without its condition is a defect in the law, not in the code.**
*(New with S6.)* CLAUDE.md and `spatial.rs` both asserted "changing anchor alone
must remain visually inert," unconditionally. It is true exactly while
`forward == diag(fit_size)` — every Fit mode at scale `[1,1]` with no rotation
and no skew — and false the moment a genuine linear part is authored, because
then the anchor is a pivot and moving a pivot moves the image. The repository's
only guard exercised the default transform, the one case where it holds. The
code was right and the law was wrong. When a documented invariant and the code
disagree, **derive the condition before assuming either side is the defect**,
and prove both halves: the case where it holds and the case where it must not.

**Seal a capability token one level deeper than the file.** *(New with S6.)*
`stage_health::EditorPreviewPermit` is a tuple struct with a private unit field,
and that is a genuinely unforgeable token — except from its own module, whose
`mod tests` could construct one. S6's `PreviewGizmoPermit` lives in a private
submodule, so the file's own tests are a *sibling* of the private field rather
than a descendant, and a source-text audit pins one declaration and one
construction. §7 will want tokens like this — for "this proxy artifact was
validated", for "this capability was proven" — and a token minted by a boolean
somebody remembered to check is not a token.

**Anything that changes what a prebuilt thing *means* belongs in the topology
signature.** S5's collider left every scope and field identical while swapping
which input each prebuilt bind group addressed; the signature did not notice, so
stale bindings would have survived a reroute. If two configurations produce
different bindings, different admission, or a different permit, they must hash
differently.

**Write the reproduction first, and never trust a fix you have not watched
fail.** S5a's fix looked complete and the reproduction still failed, because the
defect lived in *two* topological sorts and the plan had named only one. Had the
test been written after the fix, a no-op would have shipped under a confident
narrative. Grep for the **pattern**, never for the function name your notes
happen to mention, and confirm the count of sites before editing any of them.

**Prove invariance by A/B on the same branch — and prove the inverted claim
too.** S5a changed scheduling shared by every Advanced composition and proved it
moved no pixel by rendering the labeled export cases, `git stash`-ing the
change, rendering again, and diffing decoded `framemd5` — same branch, same
host, same adapter, minutes apart. S6 added the other half: its delivery fixture
renders the gizmo-authored transform, its numerically-authored twin, **and the
untouched identity**, because without the third render a gizmo that authored
nothing at all would still produce two identical files and read as a pass. State
both claims and test both: the thing must move, and everything else must not.

**A non-finite input lands on the documented default, never a clamped extreme.**
The `finite_or(value, DEFAULT).clamp(lo, hi)` idiom. An infinity is a broken
reading, not a very large one; clamping it to the maximum invents the strongest
possible value out of a fault. §7 will receive non-finite timings, byte counts,
and measured percentiles — apply the same law, and test it.

**"Deferred" is a deliverable, and "Available" is a claim you must be able to
pay for.** *(New, and specific to §7.)* Four of the five sub-tranches below
cannot be built by any amount of good engineering on this host. For those, the
work is not code: it is writing down, in typed code and in prose, exactly what
evidence would open the gate, who can produce it, and what the capability would
cost if it opened. `src/precision.rs` already models this properly —
`CapabilityDecision::Deferred(CapabilityDeferredReason::SdkOrLicenseRequired)`
is a *result*, not a failure. Extending and sharpening that record is real work
and should be committed as such. Turning it into `Available` because a stub
compiles is the one thing §7 forbids.

**Run the gate correctly or it proves nothing.** `scripts\build-windows.ps1`
only ever runs `cargo build`; it **cannot** run the gate. On this Windows host
`cargo check`/`test`/`clippy` all fail from a bare shell — without the env,
`ffmpeg-sys-next` panics on missing pkg-config; with `FFMPEG_DIR` +
`LIBCLANG_PATH` but no vcvars, bindgen panics on a missing `stdint.h`. Wrap
every cargo step:

```
cmd /c "<vcvars64.bat> >nul && set FFMPEG_DIR=<ffmpeg-8.1.2-full_build-shared>&& set LIBCLANG_PATH=C:\Program Files\LLVM\bin&& <cargo command>"
```

vcvars prints a harmless `'vswhere.exe' is not recognized` line to stderr;
ignore it. Never `2>&1` a native command in PowerShell 5.1 — every stderr line
becomes a false ErrorRecord and `$?` goes false on a successful build.

**Run `cargo fmt --all` before the gate, not after.** There is no
`rustfmt.toml`, so stock defaults are the contract and step 1 fails loudly on
whitespace you never considered. Note also that rustfmt will reflow string
literals inside test bodies — an audit test that splits source text on a marker
string should build the marker with `format!` or split on a short fragment,
because rustfmt happily wraps a long literal across lines and breaks the match.

**The measured baseline to diff against.** On `feat/native-preview-transform` at
`69ce2f1` (S6, pushed, **not yet merged**): **1256 passed, 0 failed, 87
ignored**, plus `src/bin/spout_probe.rs` 0/0/0 and
`tests/eight_texture_floor_probe.rs` 0/0/2. That branch is a *linear descendant*
of `origin/feat/web-control-panel` at `c99e043` — its parent is exactly
`c99e043` — so it fast-forwards with no merge at all. Re-measure on your actual
branch point rather than trusting this number. Never run the eight-texture floor
probe as part of ordinary verification: it rewrites the tracked
`s2-eight-texture-floor-receipt.json` in place and dirties a tree whose
cleanliness you are about to claim.

**Cross-platform CI could not be verified from this host.** `gh` is not
installed, so S6's three-platform status was never inspected and is not claimed.
If your tranche's publication story depends on S6 being green, check that first
rather than inheriting an assumption.

**Fixture-topology traps that still bite.** When building a plan fixture by
hand: non-node layers need
`RuntimeVisualRack::synthetic_legacy(LegacyRackScope::Layer)`, not
`RuntimeVisualRack::empty()`; a layer that must accept a codec attachment needs
both `field_source: MotionFieldSource::CodecVectors` and `codec.available =
true`; and every planner unit test builds `CreativeResourceLimits::default()`,
whose `max_sampled_textures_per_shader_stage` is the ordinary constant 3, so any
fixture needing more must raise `input.resource_limits` explicitly. Stable-id
ordering is **not** a trap any more — S5a made both execution sorts break ties by
composite rank, so ids may be assigned freely.

---

## The prompt

S7 — Evidence-gated precision and scale runtimes, per §7 of
`docs/successor-session-enrichment-implementation-plan.md`, with
`docs/precision-and-scale.md` as the binding design record.

Branch from `feat/web-control-panel`'s newest three-platform-green tip. S6 is
pushed as `feat/native-preview-transform` at `69ce2f1` and is not yet merged;
because it is a linear descendant of `c99e043`, branch from it directly if it
has not landed — there is nothing to merge. Land **exactly one** tranche in one
commit carrying its resource-delta table.

### What §7 actually is

Five independent capabilities. This is the landability table, established by
reading the code rather than the plan, and it is the first thing to act on:

| Sub-tranche | State in code today | Landable now? | Gate, and who opens it |
|---|---|---|---|
| **Content-addressed proxy** | `src/proxy.rs` is a complete, *pure* planner: cache key, preflight, assessment, eviction ordering. It performs **zero** filesystem mutation and spawns **zero** processes. `main.rs` already consumes it and publishes an operator-facing recommendation the operator cannot act on. | **Yes — this is the tranche**, but its first commit is the *definition*, not the worker. | No external gate: local filesystem plus the ffmpeg CLI, both of which this repo already does elsewhere. §7's own ordering clause is the gate, and it is unmet — see below. |
| **Study execution** | `src/study.rs` is a complete validated data ABI — 16 opcodes, 64 registers, 256 instructions, 6 capabilities — with `validate()`, serde, and **no evaluator whatsoever**. Nothing in the crate consumes `study::` at all. | **Codeable, but fails the delivery gate alone.** See below. | A product decision about what a Study is *for*. |
| **Hardware / zero-copy / Syphon / NDI / capture** | `src/precision.rs` has a complete typed capability evaluator with five `Deferred` reasons and **nothing behind it**. No feature flag, no trait object, no stub. | **No.** | `SdkOrLicenseRequired` needs a purchase and a legal decision; `NetworkPolicyRequired` needs an authorization; `PlatformUnsupported` needs hardware. None is an engineering task. |
| **Bounded mesh warp** | **A working mesh pipeline already exists and is easy to miss.** `StageMeshVertex { source_uv, output_uv }` (`stage_map.rs:838`), `StageSlicePlan.vertices/indices` (`:846`), convexity and winding validation, `solve_homography` (`:948`), plus a real GPU path: `mesh_buffer_bytes` accounting (`renderer/stage_map.rs:404`), `STAGE_VERTEX_ATTRIBUTES` (`:473`), and `draw_indexed` with `Uint16` indices. Caps are `MAX_POLYGON_VERTICES = 8`, `MAX_SLICES_PER_ENDPOINT = 64`, `MAX_STAGE_SLICES = 256`. | **No — and emphatically not greenfield.** | The evidence table in `docs/precision-and-scale.md` requires "a demonstrated venue requirement". That is an operator fact, not a coding one. |
| **Experimental full-16 history** | The committed ring is `TEMPORAL_HISTORY_LEN = 24` on the Compat8 path. | **No.** | A measurement plus an explicit product decision, because the surfaces do not fit the accepted budget. See the corrected arithmetic in the appendix. |

**Do not attempt more than one of these.** The plan's own dependency section is
explicit that adjacent tranches must not be merged merely because they share an
enum, and §7's five share almost nothing.

### The tranche to land: the content-addressed proxy — but read the ordering first

Choose this one, and the reason is the delivery gate rather than convenience.
`proxy_assessment_status_from_observation` in `src/main.rs` already publishes
strings like *"proxy recommended from measured decode p95, frame drops"* to a
live operator who then has **no way whatsoever to make a proxy**. The program
currently gives advice it cannot act on. Completing that sentence is genuine,
operator-visible delivery; every other §7 sub-tranche either cannot be built or
lands with no consumer.

**But §7's first clause is an ordering constraint, and it is currently unmet.**
It reads: "Implement an FFV1/Matroska worker *only after* defining bounded
decode/audio inputs." Those inputs are not defined anywhere. `ProxySettings`
(`src/proxy.rs:91`) already declares `format: Ffv1Matroska`, `scale: Half`,
`frame_rate: Source`, and `include_audio: true` — and every one of those is
**hashed into the cache key**. They are cache-plan *choices* with no semantics
behind them yet.

That makes the decision unusually expensive to get wrong. The moment a worker
gives `include_audio: true` a meaning, that meaning is what algorithm version 1
means, permanently, for every artifact anyone has keyed. And there is no obvious
default to inherit, because this repository already has **two** explicitly
non-interchangeable audio policies: the export policy (first audio stream,
source time zero, 1×, ignore visual pause/speed/modulation/looping, pad short,
trim long — which CLAUDE.md calls part of the output contract and forbids
changing silently) and the analysis-audio policy (bounded upload, bounded decode
time, a circular analysis window). A proxy is a *decode substitute* and is
therefore a third thing, needing a third policy stated in its own right.

So the honest first commit is **defining the bounded decode and audio inputs**:
what a proxy is allowed to consume, what `include_audio` means, what happens to
a source with no audio stream, with several streams, or with a stream longer
than the video — written down, versioned, and tested against the key. That may
be the whole tranche. If the worker also lands, it lands behind that definition,
never in front of it. A worker written first would violate the plan's own
ordering and would freeze an unexamined choice into a content hash.

The worker's exact surface is already marked for you: twelve
`#[allow(dead_code, reason = "…deferred cache worker")]` attributes in
`src/proxy.rs` name precisely the API a worker must wake, and
`ProxyCacheEntry`/`ProxyCachePlan::preflight` currently have **no non-test
producer at all**. `ATOMIC_PROXY_CACHE_COMMIT_LAW` is four `bool`s set true —
declarative data with no executor. Treat that list as the checklist; anything
left dead after the tranche is either unnecessary or unimplemented, and the
commit should say which.

The commit law is already written and must be executed exactly as
`docs/precision-and-scale.md` states it: create-new temporary output in the
**same directory**, temporary-file sync, atomic replacement, parent-directory
sync, an existing artifact readable until publication, and the retained artifact
plus the complete staged replacement counted **simultaneously** against the
caps. Do not invent this idiom — the repository already implements it more than
once. Read `procedural.rs`'s three-file staged no-replace directory commit, the
`.motion.json` and `.gesture.json` export sidecars, and `recovery_journal.rs`,
and reuse the strongest of them. Two atomic-commit idioms in one codebase is the
"one predicate, many callers" law failing in slow motion.

Bound the worker the way this repository bounds every other helper that leaves
the render thread. `CLAUDE.md`'s thumbnail/preview law is the model and it is
strict: metadata probe first, a `MediaSafetyPolicy` plan, bounded candidates,
bounded elapsed time, captured stdout/stderr, bounded concurrency, a reservation
retained for the helper's life, and a kill-and-reap when its generation goes
stale — *without* suspending the absolute deadline. A proxy encode is far longer
than a thumbnail, so its deadline, its concurrency, and its disk budget are new
numbers that must be frozen in the commit, not discovered at runtime.

Four things are easy to get wrong here and each is a real defect:

- **A partial artifact is not a cache hit.** The whole point of create-new plus
  fsync plus atomic replace is crash recovery. Write the crash test: leave a
  temporary file behind, restart, and prove the worker neither publishes it nor
  treats it as valid. This is the reproduction-first law — write it before the
  worker exists and watch it fail.
- **Validate the decoded identity, not the encode's exit code.** The doc says
  "validate its decoded identity/settings". An `ffmpeg` process returning zero
  proves a process ran. Decode the artifact back and check it against the
  recorded settings and the source's frame geometry before publication, and make
  a deliberately corrupted artifact fail that check in a test.
- **The key is content, never a path.** The existing key derives from the
  verified SHA-256, the authoritative byte length, and versioned settings.
  Renaming or relocating identical bytes must preserve it. Do not let a host
  path, a mtime, or a filesystem stat enter the key, the eviction record, or the
  receipt — the same privacy law that keeps operational paths out of every other
  shareable artifact in this repo.
- **Assessment is not implementation, and neither is a worker with no reader.**
  §7 says "never call assessment alone a proxy implementation". The mirror-image
  trap is a worker that publishes artifacts nothing ever opens. State plainly in
  the commit whether the decoder consults the cache, and if this tranche stops
  short of that, say so as a boundary rather than implying the loop is closed.

LRU eviction receipts and decoder A/B telemetry are named in §7 and belong here.
Both are measurements, so both must be honest about sample counts the way the
existing assessment already is — fewer than 60 frames yields
`MeasurementRequired`, and an A/B claim with less evidence than that is worth
less than no claim.

### The one that is codeable but is not delivery on its own

A Study evaluator is pure CPU, fully deterministic, has no external dependency,
and is exhaustively testable over sixteen opcodes. It is the most *pleasant*
thing in §7 and it is a trap, because **nothing in the crate consumes `study::`
at all**. Landing an interpreter with no caller is the "a pure type is not
delivery" failure that bit S3b and S4, in its purest available form.

If a later session does take it, three things must be settled first and none is
a coding decision:

- **What executes it live.** A Study is a per-pixel colour program. 256
  instructions across two million pixels is not a CPU frame budget, so the only
  viable live consumer is a **fixed, pre-compiled WGSL interpreter** reading a
  bounded instruction buffer — exactly how the Symmetry Field already consumes
  its 32-record sector table. That is emphatically **not** shader-source
  injection, which `StudyAuthority` marks permanently false, and the distinction
  must be argued explicitly in the commit rather than assumed. Anything that
  generates or concatenates shader text is forbidden outright.
- **The history-age convention, which currently disagrees with itself.** Study
  validates `LoadHistoryColor { age }` as `1..=STUDY_MAX_HISTORY_AGE` where that
  constant is **24**, rejecting age 0. The committed ring uses
  `SYMMETRY_MAX_HISTORY_AGE = TEMPORAL_HISTORY_LEN - 1` = **23**, valid `0..=23`,
  where 0 means the current frame. Both are internally consistent and they are
  off by one relative to each other: Study's age is 1-based, the ring's is
  0-based, and they span the same 24 layers. An evaluator must map Study age
  `a` to ring age `a - 1`. A direct index is wrong at every age and reads out of
  bounds at 24. Reconcile this in the commit — and note it is exactly the
  "audit every resolver" law: two subsystems already name the same ring with
  different conventions.
- **What a reference evaluator is allowed to claim.** The plan asks for
  "deterministic CPU reference fixtures". A reference evaluator with no live
  consumer is a conformance artifact and must be labelled one. It is not
  "Studies now run".

**The one structural defect worth fixing even if nothing else here is
touched.** `StudyInstruction::capability()` (`study.rs:258`) ends in a `_ =>
None` wildcard, while `destination()` and the validator's big match are both
exhaustive. Add any new `Load*` opcode and the compiler *forces* you to write a
destination arm and a validation arm, but **silently derives no capability** —
so the document sails through the canonical-capability law while reading a host
input nothing declared. That is the precise mechanism by which Study authority
enlarges without a single line of the authority record changing, and the fix is
to replace the wildcard with explicit `Self::Add { .. } | Self::Subtract { .. }
| … => None` arms so the compiler refuses the omission. This is the
"one predicate, many callers" law and the sealed-token law arriving at the same
place: an authority boundary that depends on someone remembering is not a
boundary.

Two smaller facts, free to document and expensive to "fix" carelessly.
`StudyValueType::Vector2` is a **dead-end type** — no opcode converts it to
Scalar or Color, and `OutputColor` requires Color — so any Study declaring
`MotionFieldRead` performs provably dead computation. And
`LoadDeterministicRandom` carries only `domain: u32`; there is **no seed field
anywhere in the document**, so the determinism law the plan requires is
currently undefined rather than merely unimplemented. Both are ABI decisions.

And one trap that will bite before any of that, if the vocabulary is touched at
all: **the ABI gate is exact equality, not a compatibility window.** Validation
rejects any document whose `abi` is not exactly
`{ major: STUDY_ABI_MAJOR, minor: STUDY_ABI_MINOR }`. There is no forward or
backward tolerance. Adding one purely additive opcode and honestly bumping the
minor version to 1.1 instantly invalidates **every** previously published 1.0
Study; leaving the version alone silently redefines what 1.0 means. The current
code offers no third option, so if the instruction set must grow, the versioning
law has to be designed before the opcode is written — and that is a
compatibility decision, not a coding one.

Note also that the validator bounds the *authored* age only. It cannot know how
much of the ring has actually been written, so an evaluator must additionally
guard every age against the valid-sample count exactly as `temporal_originals.wgsl`
already does, or a young program will read unwritten texture content.

### The three you must not walk through

For each of these the deliverable is a sharpened, typed, tested **boundary** —
never a stub that reports success.

**External I/O** — hardware decode, zero-copy, Syphon, NDI, capture. One fact
settles the shape of this sub-tranche: **`CapabilityDecision::Available` appears
exactly once in the entire repository — `precision.rs:623`, the return statement
itself. That arm has never executed.** `CapabilityEvidence` is six naked bools
with no `impl` block, no probe, and no producer; its only callers are test
fixtures typing literals. The evaluator in `src/precision.rs` is already
correct; leave its decisions alone.
If anything is done here at all, it is making the `Deferred` reasons more
actionable: what exact evidence, from whom, at what cost. NDI additionally
requires *both* an SDK/licence authorization and a network-policy
authorization, and neither is a thing a coding session may grant itself. A
silent software fallback branded as the requested capability is the specific
failure §7 names.

**Bounded mesh warp** — gated on a demonstrated venue requirement, and the
first thing to correct is the scoping. A name-based audit finds only
`BoundedMeshWarp` and concludes the area is open; that audit is wrong. StageMap
already owns a complete mesh pipeline, CPU and GPU: `StageMeshVertex` with
`source_uv`/`output_uv` (`stage_map.rs:838`), per-slice `vertices`/`indices`
(`:846`), convexity and winding validation, `solve_homography` (`:948`), a
`mesh_buffer_bytes` ledger (`renderer/stage_map.rs:404`), vertex attributes
(`:473`), and a real `draw_indexed` with `Uint16` indices. **Plan against that
code, not against the name.** Grep for the mechanism, never for the noun — the
S5a lesson, in its documentation form.

What is landable within the gate, if anything, is the *freeze* half of §7's
list: the caps, degenerate-triangle rejection, and checked index conversions are
CPU-side and testable. The renderer is not, and the scope question is a product
decision. Freeze the caps, the stable control-point identities, the GPU byte
ledger, the exact-identity bypass, and the Morph/modulation laws **before** any
renderer grows — the same order S6's spatial contract was frozen in, and the
reason its gizmo could be preview-only with a zero resource delta.

**Experimental full-16 history** — gated on a measurement and a product
decision, and the arithmetic is in the appendix below because the number in
CLAUDE.md is mislabelled. Whatever else happens, it may not change the settled
Advanced RGBA16F-working / Compat8-history default, and it must remain an
explicit precision path rather than a new default.

### Verification ladder

For the proxy worker tranche:

- the commit law proven end to end: create-new temp in the same directory,
  temp fsync, atomic replace, parent-directory sync, and the prior artifact
  readable throughout;
- crash recovery: an abandoned temporary neither published nor counted as
  valid, written as a failing reproduction first;
- decoded-identity validation, with a deliberately corrupted artifact refused
  before publication;
- the cache key proven path-independent — identical bytes at two paths give one
  key, and changed bytes, length, scale, frame-rate policy, audio policy, schema
  version, or algorithm version each give a different one;
- every cap rejected one unit over, with checked arithmetic, and the retained
  plus staged double-count proven at the boundary;
- deterministic eviction by `(last-used ordinal, cache key)` with its receipt;
- every bound on the helper: deadline, concurrency, captured output, generation
  staleness, reservation release on drop;
- no host path, mtime, or filesystem metadata in any key, receipt, or
  shareable record;
- non-finite and hostile measurements landing on documented defaults;
- the operator-facing status telling the truth about sample counts, including
  the sub-60-frame `MeasurementRequired` case;
- and the S5a/S6 pair: a same-branch `framemd5` A/B proving every labeled export
  case is unchanged, because a proxy worker must not touch the render path at
  all. Here, as in S6, the frame must **not** move.

Fill in every row of the plan's cross-cutting completion matrix with a test name
or an explicit not-applicable reason, following the format S6 established in
`docs/evidence/s6-preview-transform-gizmo-matrix.md`. Several rows are
legitimately not-applicable for a cache worker — no modulation address, no Morph
law, no GPU resource — and each must be argued rather than asserted, because an
unstated not-applicable is indistinguishable from an unrun check.

### Gate

Six steps in CI order, under the vcvars preamble above: `cargo fmt --all --
--check` → `node --check static/app.js` → `node --check
docs/ui-ux/wireframe.js` → `cargo check --locked --all-targets` → `cargo test
--locked --all-targets -- --test-threads=1` → `cargo clippy --locked
--all-targets --all-features -- -D warnings`.

One wording discrepancy is worth resolving before you quote either side.
CLAUDE.md's Verification section describes step 5 as "the single-threaded locked
all-target/**all-feature** test matrix", while the command actually recorded and
run is `cargo test --locked --all-targets -- --test-threads=1`, with no
`--all-features`. These are equivalent in effect — `Cargo.toml` declares no
`[features]` at all, so `--all-features` is a no-op there exactly as it is in
the clippy step — but the prose is looser than the command. The 1256/0/87 figure
was produced by the command as written above.

Report the test delta against the baseline you measured on your own branch
point, not against a number quoted here. Any GPU or cross-platform claim
requires the named opt-in `#[ignore]` fixture on the recorded adapter (AMD
Radeon RX 6950 XT / Vulkan 26.7.1) **and** hosted Linux + macOS + Windows CI at
the exact published SHA. A local run is not cross-platform evidence, and a
workflow definition is not a run. A proxy worker touches the filesystem and an
external process, so it is *more* platform-sensitive than the creative tranches,
not less: path semantics, fsync behaviour, and atomic-replace guarantees differ
across the three platforms, and a Windows-only green run proves the least here
of any tranche so far.

### The delivery gate — the lesson that bit S3b, S4, and nearly S5 and S6

A pure type, an isolated widget, or a hidden test is not delivery. For S5 the
test was "the frame must move". For S6 it was "a human dragging a handle must
change the audience image, and that same drag must be provably absent from every
non-preview surface". For S7's proxy worker it is: **a source the program has
already told the operator would benefit from a proxy must actually get one, and
the artifact must survive a crash, a rename, and a relocation without ever being
served in a partial state.**

Prove it in one fixture shaped like
`renderer::composition::tests::production_field_collider_derived_field_reaches_the_pixels`
and S6's `render_native_gizmo_transform_pipeline` — several discriminating
observations from one warm harness, each isolating one claim: the artifact is
published and readable; an interrupted publication leaves the previous artifact
intact and serves nothing partial; identical bytes at a different path hit the
same key; and a corrupted artifact is refused rather than served. And, as in S6,
include the negative render: the labeled export cases must be byte-identical
across the tranche, because a cache worker that changed a pixel would be a far
worse bug than one that never ran.

---

## Appendix — S6 landed, and the corrections it forced

Landed as `69ce2f1` on `feat/native-preview-transform`, a single linear commit
on top of `c99e043`, pushed and not yet merged. Local six-step gate green:
**1256 passed / 0 failed / 87 ignored**. Three-platform CI was **not** verified
— `gh` is not installed on that host — so no cross-platform claim was made.

S6 delivered a preview-only transform gizmo with an all-zero resource delta,
dispatching through the same authoring function the browser numeric editor uses,
so gizmo-authored and numerically-authored transforms are decoded-frame
identical while both differ from the untouched identity, and all seven
pre-existing labeled export cases are `framemd5`-identical across the tranche.
`docs/evidence/s6-preview-transform-gizmo-matrix.md` carries the full matrix.

**The correction S6 forced on the spatial contract.** CLAUDE.md and
`src/spatial.rs` both stated "changing anchor alone must remain visually inert"
with no condition, and the repository's only guard exercised the default
transform — the one case where it happens to hold unconditionally. It holds
exactly while `forward == diag(fit_size)`; an anchor step `d` otherwise moves
the sampled coordinate by `(Identity - inverse * diag(fit_size)) * d`. Both
statements are now corrected and both halves are proven. The lesson is in the
carried-forward list above: a documented invariant that disagrees with the code
is not automatically a code defect.

**A correction this page forces on CLAUDE.md, relevant to §7.** The full-16
history section of CLAUDE.md states that a new RGBA16F full-frame history ring
would be "~398 MiB at 1080p". The arithmetic is
`1920 × 1080 × 8 bytes × 24 layers = 398,131,200 bytes`, which is **398.1 MB
decimal, or 379.7 MiB binary**. The numeral 398 is correct and the unit is not.
It is a five-percent error and it sits directly under the one sub-tranche whose
entire gate is whether those surfaces fit an accepted device budget — so if the
full-16 decision is ever taken to a real budget comparison, fix the unit first.

It is also repeated in four places — `CLAUDE.md:986`,
`docs/successor-session-enrichment-implementation-plan.md:329`,
`src/renderer/symmetry_field.rs:52`, and the untracked root `MASTER_PLAN.md:256`
— so a correction has to be a sweep, not an edit. It is additionally a *gross*
cost quoted where a *delta* is implied.

This one was flagged rather than fixed, while the stale constraint below was
fixed rather than flagged, and the distinction is deliberate: the unit is
imprecise but not misleading — the numeral is right and the prohibition it
supports is unaffected — whereas the constraint bullet asserted the opposite of
shipped behaviour and would have misdirected this very session. Correct a
document when it is wrong about what the code does; leave a precision nit to the
tranche that has a reason to care.

**A stale constraint this page found and corrected.** CLAUDE.md's
Known-constraints list still carried the S5-era bullet asserting that an
Advanced composition whose layers own no image tap "cannot be scheduled", that
the tie-break is by ascending stable id, and that "repairing the tie-break is
its own tranche". That tranche landed as S5a and is documented two sections
earlier in the same file under "Advanced execution order", which is the binding
statement. The bullet was therefore flatly false and self-contradicting, and a
successor reading constraints first — which is exactly what a §7 session should
do — would have believed tapless Advanced compositions were unschedulable and
that every labeled Advanced export case must carry a rack node. It has been
rewritten to record the repair instead. The general lesson is worth more than
the fix: **a Known-constraints list is where corrected claims go to survive**,
because every other section gets rewritten by the tranche that changes it while
the constraint bullet is nobody's job. Re-read that list against the code at the
start of a tranche, not at the end.

**A wording looseness in the gate, resolved rather than inherited.** CLAUDE.md
describes step 5 as the "all-target/all-feature test matrix" while the recorded
command carries no `--all-features`. They are equivalent because `Cargo.toml`
declares no `[features]`, so the flag is a no-op wherever it appears — but the
two should be made to say the same thing by whichever tranche next touches the
Verification section, so that nobody re-derives this.

**Where this document was wrong in its own first draft, and why that is worth
keeping.** The landability table originally said of the mesh warp: "Nothing
mesh-specific exists." That was false. A complete mesh pipeline already ships —
`StageMeshVertex`, per-slice vertex and index buffers, convexity and winding
validation, a homography solver, a byte ledger, vertex attributes, and a real
`draw_indexed`. The error came from searching for the *noun*: a grep for "mesh
warp" and `BoundedMeshWarp` finds an evaluator variant and little else, and it
is easy to stop there. The mechanism was under `StageSlicePlan` all along. This
is the S5a lesson wearing different clothes — grep for the pattern, never for
the name your notes happen to use — and it is recorded here rather than quietly
fixed because the next person to scope that sub-tranche will make the same
search.

**What this document got right by checking rather than assuming.** Every claim
in the landability table was established by reading source, not by reading the
plan: that `proxy.rs` performs no filesystem mutation and spawns no process;
that `main.rs` already consumes its assessment and publishes an operator-facing
recommendation; that `study.rs` has a complete instruction vocabulary and no
evaluator, and that nothing in the crate references `study::` at all; that
`precision.rs`'s capability evaluator has no backend behind it; and that
StageMap's existing slice geometry makes a mesh warp an extension rather than a
new system. Treat every one of those as *a* reading rather than *the* reading
until you have confirmed the count yourself — which is the S5a lesson, and the
reason this appendix exists at all.
