# serde_yaml replacement — evidence-backed STOP note

Date prepared: 2026-08-27
Topic: `docs/maintenance-yaml-stop`
Read-only audit snapshot:
`48d85ea943069dbcec7c718d42daeac025da4547`
Pinned integration base:
`05c8d6cd399843236ea393e15f41a74d4b793913`
Status: **STOP — retain the bounded legacy parser until corpus,
compatibility, and fuzz gates exist**

This closes perfection handover §3.8(e) without pretending that replacing one
crate closes the security boundary. Production currently uses both
`serde_yaml 0.9.34+deprecated` and direct `unsafe-libyaml 0.2.11`; final
replacement requires both to disappear from the root and fuzz graphs.

The upstream `serde_yaml` repository is archived and identifies the project as
no longer maintained. Its manifest confirms the `unsafe-libyaml` dependency.
Primary records:

- [serde_yaml archived repository](https://github.com/dtolnay/serde-yaml)
- [serde_yaml 0.9.34 manifest](https://github.com/dtolnay/serde-yaml/blob/master/Cargo.toml)
- [unsafe-libyaml repository](https://github.com/dtolnay/unsafe-libyaml)
- [RUSTSEC-2025-0068 for serde_yml](https://rustsec.org/advisories/RUSTSEC-2025-0068.html)

## Existing boundary and compatibility law

`src/patch/yaml_boundary.rs` currently uses libyaml twice conceptually:

1. incremental token scanning rejects anchors and aliases and enforces the
   lexical/resource boundary; then
2. `serde_yaml::Value` is constructed and traversed for value depth, node,
   collection, and scalar bounds before typed `PatchState` deserialization.

The exact retained limits are:

| Limit | Exact value |
| --- | ---: |
| Document bytes | 32 MiB |
| Syntax/value depth | 64 |
| Value nodes | 250,000 |
| Collection entries | 250,000 |
| Decoded scalar bytes | 4 MiB |
| Structural tokens | 500,000 |

`src/show_bundle.rs` makes emitted bytes part of the format-v1 contract. It
requires a hostile parse/re-serialize cycle to produce byte-identical
`patch.yaml`, and import rejects a valid document that is not canonical. A
replacement emitter must reproduce historical bytes exactly or wait for an
explicitly approved bundle-version and backward-reader design. Removing or
weakening that comparison is not a maintenance solution.

## Exact source, lock, and file map

| Seat | Files |
| --- | --- |
| Root dependency and lock | `Cargo.toml`, `Cargo.lock` |
| Fuzz dependency and lock | `fuzz/Cargo.toml`, `fuzz/Cargo.lock` |
| Hostile parser boundary | `src/patch/yaml_boundary.rs`, `src/patch/editor.rs` |
| Byte-canonical portable bundles | `src/show_bundle.rs` |
| Other production YAML entry points | `src/preset.rs`, `src/study.rs`, `src/stage_map.rs`, `src/recovery_journal.rs`, `src/procedural.rs` |
| Unbounded bypass to remove | procedural CLI anchor loading in `src/main.rs` |
| Current differential target | `fuzz/fuzz_targets/patch_yaml.rs` |
| Current seed corpus | three small files under `fuzz/corpus/patch_yaml/` |
| Graph/policy aftermath | dependency policy, audit/deny, release SBOM, and their exact self-tests |

Every production read must eventually pass one bounded application wrapper.
Replacing only patch-editor parsing would leave the remaining surfaces and the
CLI bypass outside the security claim.

## Candidate ruling

| Candidate | Exact reviewed identity | Ruling |
| --- | --- | --- |
| `noyalib` | 0.0.28; archive SHA-256 `9f075ef19fa3bcf8697c0ef96c37d5c435d339a40ab8081cae3aac3a4e7fee9a` | Closest feasibility spike: pure safe Rust, generic value conversion, and configurable document/depth/event/node/anchor/scalar policies. Pre-1.0 churn, YAML-1.2 defaults, and emitter compatibility remain unproved. |
| `serde-saphyr` | 1.1.0; archive SHA-256 `a1ec1f5cac0eb96063c64b28705255a7ed6e7d77f95c1d25e9f8b8c928006ce1` | Credible pure-Rust alternative with parser budgets, but its data model and aggregate scalar budget do not directly reproduce the current generic-value and per-scalar law. |
| `serde_yaml_ng`, `serde_norway` | Not selected | Retain unsafe-libyaml variants; they may bridge compatibility but do not close the stated security objective. |
| `serde_yml` | Prohibited | RUSTSEC-2025-0068 reports it unsound and unmaintained. |

No production candidate is selected by this note.

## Deterministic and operator boundary

The promotion decision is predominantly deterministic. Physical hardware
proof is not required.

The operator/provenance boundary is acquisition of authentic historical inputs
if they are not already available in tracked fixtures: every released `.cos`
bundle, released loose patch, preset, study, stage map, and genuine
recovery-journal payload. Generated approximations cannot replace missing
released bytes.

The deterministic oracle must preserve:

- acceptance/rejection of every historical and hostile document;
- typed state, tags, arbitrary mapping keys, mapping order, integer ranges,
  and exact floating-point bits;
- canonical emitted bytes, especially format-v1 bundle `patch.yaml`;
- duplicate-key, directive, Unicode, quoting, alias/anchor,
  nonfinite-number, and limit±1 behavior; and
- all six exact resource limits above.

`serde_json::Value` is not an adequate differential oracle. The current 1 MiB
fuzz-input ceiling also cannot prove the 4 MiB scalar or 32 MiB document
boundaries; dedicated deterministic boundary tests are required.

## Precise reopening gate

1. Freeze exact `patch.yaml` bytes from every released `.cos`, all released
   loose YAML documents, genuine recovery records, generated default/maximal
   `PatchState`, and every legacy/hostile inline fixture.
2. Add one exactly pinned candidate in side-by-side dev/fuzz mode only.
3. Run at least 10,000 deterministic differential cases and an uninterrupted
   one-hour differential fuzz campaign.
4. STOP on any legacy rejection, typed-state divergence, tag/key/order/numeric
   mismatch, resource-limit gap, panic, unbounded allocation, or bundle-byte
   difference.
5. If bundle bytes differ, obtain approval for a new bundle format and backward
   reader before any production flip; do not weaken format-v1 canonicality.
6. Route patch editor, presets, studies, stage maps, recovery, show bundles,
   procedural generation, and the CLI anchor path through one bounded wrapper.
7. Remove both `serde_yaml` and `unsafe-libyaml` from root and fuzz graphs;
   prove their absence with locked inverse-tree checks.
8. Pass the exact six-command gate, fuzz and hostile self-tests, dependency
   audit/deny/SBOM checks, and exact-head hosted CI before integration.

## Repository and protected-artifact boundary

This closeout changes tracked evidence only. It does not alter the parser,
serializer, manifests, locks, bundle version, recovery format, fuzz corpus, or
release artifacts.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `.da-vinci-canon-pre-refinement-backup-20260822.zip` | 66,225 | `494b63ad0bd96cfb1c7f20a37ad574075a26dd289c14f9f24e71ffe48ab1eea4` |
| `4K_Nature_Cinematography_recorded_with_Nikon_D5300.webm.1080p.vp9.webm` | 56,984,527 | `ee1cfc47671617f8bdf8031dd19cb00f9359e4bf47bdddc7f1fca9df13d034a0` |
| `Black_swan_(Cygnus_atratus).webm.1080p.vp9.webm` | 60,528,641 | `2b51dda28643af61a163d8b3457fd5885c596c51632576a53a6c4c06722630a4` |

They were not modified, copied, renamed, or staged. `videos/audit.mp4` remains
absent and was not minted.

## Closing fields

- Disposition: **EVIDENCE-BACKED STOP**
- Topic evidence commit: **`fe06fdd`**
- Topic receipt commit: **`7e47d238fbec62184e1f9e4924105862af996101`**
- Integration commit on `feat/web-control-panel`:
  **`e81bddfd6ee823b6247b08f24f5c60cb5de2a011`**
- Exact-commit CI: **PASS**, run `33063795621` — dependency 30 s,
  Linux 9m00s, macOS 8m21s, Windows 13m56s
- CI-form six-command gate: **OBSERVED PASS** — fresh rerun completed with
  2,148 tests passed, 163 ignored external/physical seats, six benches green,
  and clippy clean
- Gate anomaly: **BOUNDED AND REPROVED** — the first whole-suite pass exited
  with Windows `STATUS_HEAP_CORRUPTION` after
  `render_export::tests::drop_deadline_includes_the_cancel_request`; the exact
  focused test then passed in 1.00 s and a fresh full six-command gate passed
- Production replacement: **NOT ATTEMPTED**
- Frozen historical corpus: **NOT YET COMPLETE**
- Differential campaigns: **NOT RUN**
- Protected-root and `videos/audit.mp4` recheck: **OBSERVED PASS**

## Deliberate non-claims

This note does not endorse a replacement, claim the current bounded parser is
vulnerability-free, or claim that pure Rust alone proves safe resource
behavior. It does not claim semantic round-trip is enough where format-v1
requires byte identity. It does not authorize a bundle-version change, removal
of hostile limits, or synthetic documents as substitutes for authentic
released bytes.
