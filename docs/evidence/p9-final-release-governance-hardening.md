# P9 final-release governance hardening

Status: **implemented and locally policy-validated; live `v1.7.0` publication not executed**
Scope: release workflow and bounded release-evidence helpers only

## Retained result

The release workflow now admits only an annotated release tag object. At the
initial candidate resolution, before draft creation, at the first draft
redownload, immediately before the public transition, and again after the final
published 12-file redownload, the same production verifier requires both exact remote rows:
`refs/tags/v1.7.0` and `refs/tags/v1.7.0^{}`. The local tag object SHA must equal
the remote unpeeled row, the peeled row must equal the one verified commit, and
the initially resolved tag-object SHA must remain unchanged. There is no
lightweight-tag fallback.

The newest exact-commit CI and adversarial runs remain the publication gate.
The bounded gate receipt now carries each selected run ID, URL, number,
attempt, conclusion, and head SHA. From the selected CI run's jobs API it
requires the exact Linux, macOS, Windows, and dependency-policy job/step pairs.
The checked-in policy also binds those names to the exact reviewed matrix
runner mapping and whole step bodies. Multi-command portable steps use explicit
`bash` plus `set -euo pipefail`; the Windows-specific check/test/Clippy step
retains `pwsh` and an immediate `$LASTEXITCODE` guard after each command. Check,
test, and strict Clippy all cover every target and feature, while the all-target
test command forwards no libtest-only argument to Criterion benches.
It downloads each selected platform job's own bounded log by stable job ID and
requires at least one successful cargo-test summary in each; summaries from
one job cannot satisfy another. The receipt records exact final-candidate
format/check/test/strict-Clippy results, per-job and aggregate test counts,
ignored names and external-fixture occurrences, the P10 contradiction result,
dependency-exception status, and vendored-source result. Missing, renamed,
truncated, or contradictory jobs, logs, summaries, or steps stop publication.

The ten immutable initial assets are assembled, checksummed, Sigstore signed,
and GitHub-attested. An authenticated, bounded, paginated
`GET /releases?per_page=100&page=N` inventory rejects any existing draft or
published release with the exact tag;
then `gh release create --draft` provides the race-safe create boundary and one
no-clobber `gh release upload` attaches the exact ten files to the authenticated
draft. A rerun or race that finds a release stops before upload or body
mutation. Redownload
verification authenticates `SHA256SUMS` before executing the package verifier,
then verifies a GitHub attestation for every one of those ten exact files. Each
verification explicitly requires the release repository, exact
`release-trust.yml@refs/tags/v1.7.0` certificate identity, GitHub Actions OIDC
issuer, SLSA provenance-v1 predicate, exact tag source ref, and exact peeled
commit source digest. It does not treat the Windows ZIP as a substitute for the
other nine results or accept an attestation from another same-repository
workflow/ref.

After that redownload succeeds, a separate bounded machine-readable external
final-release receipt is built. It contains the annotated tag and peel, the
captured draft database ID and prepublication state,
selected workflow identities, exact final-candidate results, two-build hashes,
BuildIdentity/Cargo/shader/FFmpeg/SBOM/inventory/review/vendor hashes, source
receipt reconciliation hashes, the explicit unavailable/unsigned Authenticode
disposition, every immutable initial asset hash and attestation result,
checksum Sigstore identity and bundle hash, the deterministic expected public
tag URL (never the draft's `untagged-*` HTML URL), and the downloaded
version/package/FFmpeg/shader/dependency verification report. Closed-schema
validation cross-links those values instead of accepting status-only claims.

The final receipt freezes before its own signature, which avoids a circular
self-claim. The workflow then keylessly signs and verifies it, creates and
verifies its GitHub build-provenance attestation, uploads only the receipt and
its Sigstore bundle without overwrite, and replaces the release body from the
same bounded summary list. It re-reads the exact draft ID, tag, state, body,
and 12-asset inventory; freshly re-downloads all 12 files; compares every exact
name and SHA-256 with the already verified local bytes; authenticates the
fresh `SHA256SUMS`; reruns the package verifier and all ten initial GitHub
attestations; then revalidates the final receipt and repeats its Sigstore and
GitHub-attestation verification. It re-verifies the exact annotated tag object
and peel immediately before the sole final `gh release edit --draft=false`
publishes it. A final read-only view must
prove the same database ID/tag, `isDraft:false`, and the exact public URL,
body, and 12 asset names. One more fresh empty-directory download then proves
all 12 published file names and SHA-256 values still equal the verified draft
bytes. The same annotated-tag verifier runs once more after that completed
download, so a moved tag cannot survive either side of the public boundary; no
mutation follows it.

The final receipt records that closed GitHub-attestation policy for every
initial asset and for the post-freeze receipt attestation. A status-only
`verified` claim cannot substitute another repository, workflow identity,
issuer, predicate, source ref, or source digest.

## Fail-closed and mutation coverage

- Annotated-tag row parsing rejects missing tag rows, missing peeled rows,
  duplicates, unrelated refs, a non-`tag` local object, a changed tag object,
  and a changed peeled commit.
- Required-run selection rejects stale head SHAs, pending/failed conclusions,
  malformed URLs, missing jobs, incomplete platform steps, unbounded logs, ZIP
  expansion, and absent or failing cargo-test summaries.
- The log parser fixture includes GitHub's ISO timestamp prefix and proves that
  an ignored external FFmpeg fixture is still counted. A real timestamped
  `test result: FAILED.` fixture is rejected, at least one successful summary
  is always required, and final-candidate collection requires at least one
  successful summary for each of the three selected platform test steps.
  Separate mutations prove that exact steps under an arbitrary job and three
  successful summaries confined to one platform job both fail closed. The
  final receipt independently enforces the same exact job names, job IDs, log
  URLs, per-job counts, and recomputed aggregates.
- Static CI mutations replace each of the seven required source step bodies
  with an echoed fake successful test summary and separately inject
  `continue-on-error`; all fail. Matrix runner remapping, job-level shell
  defaults, changed `if`/shell/run fields, omitted all-feature coverage, or a
  libtest-only argument on the Criterion-bearing all-target command also fail.
- Every one of the four GitHub attestation call sites has per-flag hostile
  substitutions for repository, certificate identity, issuer, predicate,
  source ref, and source digest. Input-receipt and final-receipt mutations also
  reject another workflow, branch ref, or commit digest.
- Final-receipt validation rejects mutated workflows, final validation,
  reproducibility, exact asset inventory, checksum/Sigstore cross-links,
  package and BuildIdentity associations, evidence/vendor/source hashes,
  downloaded verification, Authenticode disposition, self-signing boundary,
  and unbounded summary text. Checked-in one-field mutations specifically
  cover the SBOM/source/inventory/review/buildconf/provenance asset rows, an
  altered or omitted source-evidence receipt, downloaded version versus tag,
  FFmpeg source/archive/buildconf versus policy and package, Cargo.lock,
  shaders, vendor archive, package notices, source ZIP, checksum inventory,
  provenance artifacts, and dependency evidence.
- The static workflow policy fixes the ordering from authenticated checksum to
  executable verification, all-asset attestation, receipt build/validate,
  keyless sign/verify, attestation, no-clobber upload, body edit, and remote
  persistence re-verification.
- The create-only publication self-test covers an existing same-tag draft,
  terminal empty inventory, duplicate IDs/tags, malformed pages, canonical
  two-page traversal, and a hostile next-page link. Only a bounded terminal
  authenticated inventory with no exact draft or published tag admits create;
  every HTTP, shape, pagination, duplicate, or identity error stops without a
  mutation. The static policy rejects missing/reordered preflight,
  update-before-preflight, overwrite flags, and create/update actions.
- Draft-policy mutation tests require draft creation before either upload,
  prove the draft remains private through the exact-12 persistence checks, and
  reject missing/early publication, an upload or delete after the sole final
  `--draft=false` transition, and deletion or substitution of either final
  annotated-tag check. The created database ID, exact tag, and draft
  state are captured as job outputs, carried across jobs, embedded in the
  receipt, and checked before and after draft operations.
- Draft body and final public URL/body comparisons are ordinal/case-sensitive;
  both published byte comparisons already use case-sensitive SHA equality.
  Only terminal whitespace in the generated notes body is intentionally
  normalized with `TrimEnd()` on both sides. Case-insensitive
  comparator mutations for body, URL, and either SHA comparison fail policy.

## Local evidence on the implementation tree

Passed:

- `python -B scripts/wait-required-workflows.py --self-test`
- `python -B scripts/verify-release.py self-test`
- `python -B scripts/finalize-release-receipt.py self-test`
- `python -B scripts/check-release-workflow.py`
- Python AST validation
- PyYAML parse of all three workflows
- PowerShell AST parse of all 18 `shell: pwsh` workflow blocks after replacing
  GitHub expressions with inert parser tokens
- pinned local `actionlint` 1.7.12 over `ci.yml`, `adversarial.yml`, and
  `release-trust.yml`
- Local `gh 2.98.0` help inspection confirmed support for the explicitly used
  `--cert-identity`, `--cert-oidc-issuer`, `--predicate-type`, `--source-ref`,
  and `--source-digest` flags; an absent/unsupported flag makes the workflow
  command fail before publication. The upstream contract recommends exact
  signer-workflow or certificate-identity validation:
  <https://cli.github.com/manual/gh_attestation_verify>.

These local parsers and the checked-in static policy are not claimed as a
substitute for the exact-commit GitHub Actions execution.

## External stop conditions

No annotated `v1.7.0` tag, final candidate SHA, exact-SHA CI/adversarial run,
two-build Windows result, Sigstore identity, GitHub attestation, release URL,
or redownloaded final receipt exists on this uncommitted implementation tree.
Those facts are intentionally generated only by the tagged workflow and must
remain absent from source evidence. Publication stops on any missing/moved tag
row, stale workflow result, unparseable test count, non-identical build,
inventory/hash/identity mismatch, missing attestation, failed persistence
re-read, or required Authenticode claim. Authenticode itself remains
unavailable and the PE must remain unsigned; Sigstore does not replace a PE
trust chain. A partial first upload can leave only an incomplete authenticated
draft, never a public partial release. All automatic reruns refuse that draft;
an operator must inspect and deliberately remove it before a clean retry
rather than silently overwriting any asset or body.
