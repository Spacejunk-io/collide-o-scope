# Maintenance actions and Cosign v3 trust migration — proof note

Date prepared: 2026-08-27
Topic: `feat/maintenance-actions-cosign3`
Pinned integration base:
`7e79ed773e1278f65da2ce15e32927b1d5847fa0`
Status: **integrated with exact-commit CI observed; live v0.3 signing remains
operator-gated**

This is the §3.8(a) maintenance tranche. It updates the immutable GitHub Action
pins, advances the release signer and verifier to the patched Cosign 3 line,
and binds the resulting trust decisions in the release-workflow policy. It
does not create, move, sign, publish, or replace a release tag.

## Immutable action inventory

The reviewed `uses:` inventory across `adversarial.yml`, `ci.yml`, and
`release-trust.yml` is closed over these exact tag commits and occurrence
counts:

| Action | Exact tag commit and comment | Occurrences |
| --- | --- | ---: |
| `actions/checkout` | `3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1` | 9: adversarial 2, CI 2, release trust 5 |
| `actions/cache` | `55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0` | 7: adversarial 2, CI 4, release trust 1 |
| `sigstore/cosign-installer` | `6f9f17788090df1f26f669e9d70d6ae9567deba6 # v4.1.2` | 2: release trust 2 |
| `actions/attest-build-provenance` | `977bb373ede98d70efdf65b84cb5f73e068dcc2a # v3.0.0` | 2: release trust 2 |
| `actions/upload-artifact` | `ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2` | 1: release trust 1 |

The policy checker binds the exact three-workflow inventory, full lowercase
commit SHAs, version comments, and per-file counts. Its hostile fixtures cover
a removed use, a duplicate, a one-nibble SHA substitution, a changed tag
comment, an unreviewed action, and movement of an otherwise allowed pin between
workflows. The two security-critical signing steps are additionally bound by
reviewed whole-block SHA-256 values, so an inert wrapper, reordered statement,
swallowed failure, or hidden assignment changes the admitted body.

The checker also binds each complete reviewed workflow document by SHA-256.
This closes YAML-equivalent encodings that a line-oriented inventory could
otherwise miss, including escaped quoted keys and flow-style step mappings.
Any future workflow edit must therefore update its reviewed digest together
with the semantic pin/count policy and hostile fixtures.

Checkout v7 and cache v6 use the Node 24 action runtime. Upstream documents
Actions Runner **2.327.1 or later** as the minimum Node 24 runner. These
workflows use GitHub-hosted runners and contain no `container:` job. This
tranche therefore makes no self-hosted-runner or authenticated-Git-from-a-
Docker-container compatibility claim.

Primary pin and runtime references:

- <https://github.com/actions/checkout/releases/tag/v7.0.1>
- <https://github.com/actions/checkout#checkout-v5>
- <https://github.com/actions/cache/releases/tag/v6.1.0>
- <https://github.com/actions/cache#v5>
- <https://github.com/sigstore/cosign-installer/releases/tag/v4.1.2>

## Cosign v3 binary identity

Both installer seats select `cosign-release: v3.1.3`. The upstream annotated
`v3.1.3` tag object is
`2f3a85b04907df5b770eb049d7e4d08d4b018d86`; it peels to release commit
`11926fa5bbbbde47e88fc006b625a17769b743b2`. The two pinned installer steps are
each followed immediately by the unconditional PowerShell step
`Verify pinned cosign identity`. That step resolves the installed executable,
requires this exact binary SHA-256, propagates a native version-command
failure, and checks the four release identity fields:

| Property | Required value |
| --- | --- |
| Windows amd64 executable length | 198,819,314 bytes |
| Windows amd64 executable SHA-256 | `9fe59be0eca1271873ce019061335eb1ac419b7059202e797828467ddabe33be` |
| `gitVersion` | `v3.1.3` |
| `gitCommit` | `11926fa5bbbbde47e88fc006b625a17769b743b2` |
| `gitTreeState` | `clean` |
| `platform` | `windows/amd64` |

The byte length is evidence about the downloaded upstream Windows artifact;
the workflow's executable admission decision is the exact SHA-256 plus the
four version fields. Upstream release:
<https://github.com/sigstore/cosign/releases/tag/v3.1.3>.

## Exact release-workflow trust changes

- Every `cosign verify-blob` retains an explicit `--bundle` and the exact OIDC
  issuer `https://token.actions.githubusercontent.com`.
- Every certificate identity is the exact current workflow ref,
  `https://github.com/${{ github.repository }}/.github/workflows/release-trust.yml@${{ github.ref }}`.
  The former broad `refs/.*` certificate-identity regular expression is gone.
- Immediately after each of the two `cosign sign-blob` calls, PowerShell parses
  the emitted bundle and requires the exact top-level media type
  `application/vnd.dev.sigstore.bundle.v0.3+json` before the workflow may
  verify, attest, upload, or publish it.
- The checksum bundle and frozen final-receipt bundle retain their existing
  exact issuer, identity, authenticated checksum, attestation, immutable
  inventory, and redownload/persistence gates.

Cosign v3 is required here because
[GHSA-fx35-mq7g-6g98](https://github.com/sigstore/cosign/security/advisories/GHSA-fx35-mq7g-6g98),
published 2026-08-06 with High severity and CVSS 7.4, reports that a malformed
legacy JSON bundle could make `verify-blob` or `verify-blob-attestation` fall
back to an embedded raw public key and bypass X.509 chain, certificate
identity, and issuer enforcement. Cosign v2 versions through 2.6.4 and v3
versions through 3.1.2 are affected; v2.6.5 and v3.1.3 are patched. The
standardized bundle verifier used by Cosign v3's default format is not affected.
The historical v1.8.1 receipt correctly records v2.6.5 and remains unchanged.

## Online trust and TUF boundary

The pinned installer downloads the selected release artifact during the job,
and keyless signing and verification use the public Sigstore services. The
workflow does not vendor the Cosign executable, a TUF repository, or a trusted
root, and it does not pass an offline trusted-root configuration. Upstream's
release-verification procedure initializes a TUF client against
`https://tuf-repo-cdn.sigstore.dev` to retrieve the artifact verification key:
<https://github.com/sigstore/docs/blob/main/content/en/cosign/system_config/installation.md#verifying-cosign-releases>.

Accordingly, this is an online hosted-runner trust chain. Neither the workflow
nor the local compatibility replay below is evidence of air-gapped operation,
offline TUF freshness, or a reproducible offline bootstrap.

## Observed v1.8.1 backward replay

An independent local replay downloaded the published v3.1.3 Windows amd64
executable and the published v1.8.1 `SHA256SUMS` plus
`SHA256SUMS.sigstore.json` assets. The executable was exactly 198,819,314 bytes
with SHA-256
`9fe59be0eca1271873ce019061335eb1ac419b7059202e797828467ddabe33be`.
`cosign version --json` reported exactly `v3.1.3`, commit
`11926fa5bbbbde47e88fc006b625a17769b743b2`, tree state `clean`, and platform
`windows/amd64`.

The downloaded historical bytes were:

| Asset | Length | SHA-256 |
| --- | ---: | --- |
| `SHA256SUMS` | 767 | `f970b73bf5860fbbe6b3ea561aa811b8733597957859ac4a091f8cde113ee649` |
| `SHA256SUMS.sigstore.json` | 8,993 | `d414c503ffe40d045f45b259feea9bd29c74c667e16cf0ec92fcb8a4df44bb7b` |

The historical bundle has no top-level `mediaType`, confirming that this seat
tests legacy-bundle compatibility rather than the new v0.3 emission gate.
Cosign v3.1.3 verified those immutable published bytes in normal online mode
with:

- certificate identity
  `https://github.com/Spacejunk-io/collide-o-scope/.github/workflows/release-trust.yml@refs/tags/v1.8.1`
- OIDC issuer `https://token.actions.githubusercontent.com`
- the published `SHA256SUMS.sigstore.json` supplied through `--bundle`

The command exited zero and printed exactly `Verified OK`. This proves that
the selected v3 binary can replay the existing v1.8.1 legacy checksum bundle
unchanged under its exact historical workflow identity and issuer. It does not
re-sign, rewrite, republish, or otherwise alter the v1.8.1 release.

## Live signing and persistence gate

A source inspection and backward replay cannot prove that a newly minted
Cosign v3 bundle survives the repository's complete sign, verify, upload,
redownload, and post-publication path. That proof requires either the next
operator-approved immutable release tag or a separately authorized isolated
canary with equivalent OIDC identity and persistence checks.

The existing `v1.8.1` annotated tag object is
`0f7e832a6546b947b28711d2a7004654ee1049c1` and peels to commit
`a5f9043348b047729ac73a3f7f0252e532737b4f`; neither identity may be moved,
deleted, or reused for this test. The primary workflow's `workflow_dispatch`
pre-tag path intentionally refuses an already existing target tag, so it is
not a back door for replaying v1.8.1. Live v0.3 signing and persistence remain
operator-gated.

## Repository and protected-artifact boundary

This tranche is confined to the three tracked workflows, their structural
checker, and evidence. The three protected binary root artifacts remain the
only non-ignored untracked root artifacts at 66,225, 56,984,527, and
60,528,641 bytes. They were not opened, copied, renamed, or staged.
`videos/audit.mp4` remains absent and was not minted.

## Closing fields

- Topic implementation commits: **`a8764df`**, **`22588b1`**
- Topic receipt commit: **`bfda487`**
- Integration commit on `feat/web-control-panel`: **`96bcb73`**
- Exact-commit CI: **OBSERVED PASS** —
  [run 33054896096](https://github.com/Spacejunk-io/collide-o-scope/actions/runs/33054896096)
  passed at exact head `96bcb73` across Linux 24.04, macOS 15, Windows
  VS 2022, and dependency policy
- Hosted full gate: **OBSERVED PASS** — the exact six-command CI-form gate
  passed on the topic tree: formatting and both JavaScript parsers;
  all-target/all-feature compile; 2,143 tests passed with zero failures and
  163 explicitly ignored tests; all six benchmark probes succeeded; and
  Clippy passed with warnings denied
- Focused workflow gates: **OBSERVED PASS** — policy checker and hostile
  self-tests, Python AST, PyYAML parse, actionlint 1.7.12, 41 PowerShell run
  blocks parsed after GitHub-expression substitution, and `git diff --check`
- Live v0.3 sign/verify/persist/redownload proof: **PENDING — OPERATOR-GATED**
- Focused immutable-pin and trust-policy source inspection: **OBSERVED PASS**
- Local Cosign v3.1.3 binary identity and v1.8.1 legacy-bundle replay:
  **OBSERVED PASS** — exact result `Verified OK`
- Protected-root and `videos/audit.mp4` recheck: **OBSERVED PASS**

## Deliberate non-claims

This note is not a new release receipt, v0.3 persistence receipt, or
offline-verification receipt.
It does not claim that Node 24 actions run on an older or unreviewed self-hosted
runner, that containerized authenticated checkout was exercised, or that a
local parser substitutes for GitHub Actions. Historical release tags, assets,
and evidence remain immutable.
