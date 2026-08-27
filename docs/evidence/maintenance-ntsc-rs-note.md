# ntsc-rs untagged-commit review — compatibility STOP note

Date prepared: 2026-08-27
Topic: `feat/maintenance-ntsc-rs`
Pinned integration base:
`48d85ea943069dbcec7c718d42daeac025da4547`
Status: **integrated compatibility STOP; exact-commit CI green on rerun**

This is the §3.8(c) maintenance tranche. It reviews every commit between the
shipping ntsc-rs pin and upstream `main`, freezes the application-visible
preset, patch, and pixel contracts before changing the dependency, and stops
at the first observed visual change. The shipping pin remains
`4b79500dfac64efcfb393eebc89f5c75565ee5ae`; no candidate output is promoted
by this note.

## Exact upstream ledger and resource ruling

The published `v0.9.4` tag predates the shipping git pin and is not an upgrade
candidate. The eight untagged commits reviewed in order were:

| Commit | Relevant disposition |
| --- | --- |
| `b3af8c63710a80471016c3834d526ebd833e4977` | Adds `yiq_fielding` tests; no production change |
| `4284bf1465f6607b8e277f203a59588546218d3c` | Moves to fearless_simd 0.5 and rewrites simplex mask/sign operations |
| `7a5fd8bf410d0774c681811a43e8f2a10a7f153b` | GUI dependency and `SettingID` debug maintenance |
| `f76c218c51e6fa7218dcb72f6f19a72d81bcd778` | xtask/cargo_toml maintenance only |
| `57369f280e0721d0cbdc1c88083c787909bc78a9` | Moves the core to fearless_simd 0.6; last bounded candidate |
| `6654769d4b2a4b77ba0fe28acf5c2aefc759bac7` | Doubles every ntsc Rayon worker stack from 2 MiB to 4 MiB |
| `7eeab92a92cf18d88e7f34c29fab01c42cbfc023` | GUI Clippy-only change |
| `af9833b4bb81f195f7fe4a3667211f2a94139a42` | Workflow/Windows bundle maintenance only |

The project owns five persistent ntsc worker pools and can transiently own a
sixth during replacement. The current 2 MiB stack law reserves 240 MiB of
virtual stack at 24 workers per pool, or 288 MiB transiently. Commit `6654769`
would double those figures to 480 MiB and 576 MiB without adding a bounding
seam. It and every descendant are therefore outside the resource ledger.
`57369f2` was the only meaningful bounded candidate.

## Oracle frozen before dependency movement

The baseline was measured in an isolated detached worktree at exact base
`48d85ea943069dbcec7c718d42daeac025da4547`, with the shipping ntsc-rs pin and
the CI Rust/FFmpeg/MSVC environment. The temporary worktree was removed after
the comparison.

The external-file-free pixel fixture uses odd 33×25 RGBA geometry, varied
color and alpha bytes, reference frames 0, 1, 17, and 61, and every exposed
application control at a non-default value: EP tape speed; chroma loss; edge
wave; head switching; tracking wave and snow; snow; composite, luma, and
chroma noise; luma smear; and composite sharpening. Each route runs twice
from fresh state, requires exact repeatability, requires RGB to change, and
requires alpha to remain byte-identical.

Shipping-pin hashes:

| Frame | LiveParity SHA-256 | Native SHA-256 |
| ---: | --- | --- |
| 0 | `b8063b318c4aa605806c4371fd20c0bfbf330811344a73cc57794eb21bfa2191` | `a468f190a91ffa79bf5598d902addfa8a35c76e348009bf55c8d7ee77575aefb` |
| 1 | `066cefbb71b94ade2cda3039dcdd59c75dbce85569f7ad4500c4b19b5afa5437` | `d78fc5708a2db73c408a3f42fd0471b16bf2d72a2b74c3f5d4766ab3565c3171` |
| 17 | `563a12f52753af31ae078ff0b523625f087e5bb50a7868f395098bcf8bb41108` | `14e1534486060becc2c59f27a860c84ba582b8387e80386b174689682d62254e` |
| 61 | `a83a6589843f193db781390e39d45aecbdca308286fba396f1cf953ca6e1b184` | `0aee7c1732578632653c625f35ab88c67e7ac09be9b222258424d2364bbf5c7d` |

The preset oracle uses `SettingsList::<NtscEffect>::new()` and pins all 62
flattened `(id, name)` descriptors in exact depth-first order, the complete
default canonical JSON, the complete configured canonical JSON, parse
equality, and byte-identical reserialization. The patch oracle independently
pins all 19 `NtscConfig::from_params` fields, exact JSON bytes, exact YAML
bytes and field order, typed round trips, and byte-identical reserialization.

## Candidate comparison and STOP

`b3af8c6` reproduced all eight shipping hashes plus both preset documents and
the 62-entry descriptor table. That establishes the tests-only commit as
semantically inert but gives the application no maintenance benefit.

The first meaningful commit, `4284bf1`, changed every pixel hash:

| Frame | LiveParity SHA-256 | Native SHA-256 |
| ---: | --- | --- |
| 0 | `ae3b4e85baadc0e862812901faaaf76e397b46a7df1149ffc7afcd7905a67024` | `81542346244419cf5bd119e0b02933433f072c4ebb79ca6f927f844f9261eaff` |
| 1 | `f7c8d3c658cdab9217e7d8160219bfdd43382cf2abf0c4542015dfffa84cdea2` | `59a14948ea6f6405436cded4ae78bf473bb76e1bd79c5435162935a7a1471566` |
| 17 | `1eb1d505107bba957d42e0e47cda6245da48bc0e76ff1e94e1b9540417d2d3b2` | `80e8818787e1d58cb982b0db6413d13de1db75d85b2801af1a8133100baee028` |
| 61 | `c617e28f8c12cf09b36bbe082d10aea134df09a882970eaec12de865fa4d9e67` | `380fc04e51c87415986501f0f28694ee54041cdbfb024f3c0c7723e464a1656c` |

The bounded candidate `57369f2` reproduced those same eight changed hashes,
not the shipping hashes. The difference begins with the fearless_simd/noise
rewrite rather than the later settings, xtask, or stack-size commits. Preset
and patch structure are not the blocker; pixel semantics are.

The predeclared rule was exact compatibility across both processing routes,
with no tolerance, platform-specific expected values, scalar-only fallback,
or post-bump redefinition of the oracle. The candidate therefore stopped.
`Cargo.toml`, `Cargo.lock`, checkout-path policy, release workflow, and SBOM
references were restored to the shipping pin before commit. The retained
source change is only the regression oracle in `src/ntsc/mod.rs` and
`src/patch/mod.rs`.

## Precise reopening gate

Reopen only as a new visual-semantics campaign, not as an automatic dependency
bump. It must:

1. Re-audit the exact upstream range and resource stacks; an unbounded worker
   stack increase remains forbidden.
2. Explain and review the noise-sign change as an intentional authored visual
   change, or select an upstream commit/patch that preserves the frozen bytes.
3. If changing the look intentionally, obtain an operator-approved aesthetic
   receipt using an authentic provisioned source; do not mint a replacement
   `videos/audit.mp4` or silently replace the old hashes.
4. Freeze a new versioned visual oracle while retaining the shipping oracle as
   historical compatibility evidence.
5. Pass both preset/patch byte contracts, exact new pixel goldens on Linux,
   macOS, and Windows, dependency policy, remeasured SBOM/release policy, the
   exact six-command gate, and exact-head CI.

Until those conditions exist, `4b79500dfac64efcfb393eebc89f5c75565ee5ae`
is the settled pin.

## Repository and protected-artifact boundary

The three protected binary root artifacts remain the only non-ignored
untracked root artifacts:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `.da-vinci-canon-pre-refinement-backup-20260822.zip` | 66,225 | `494b63ad0bd96cfb1c7f20a37ad574075a26dd289c14f9f24e71ffe48ab1eea4` |
| `4K_Nature_Cinematography_recorded_with_Nikon_D5300.webm.1080p.vp9.webm` | 56,984,527 | `ee1cfc47671617f8bdf8031dd19cb00f9359e4bf47bdddc7f1fca9df13d034a0` |
| `Black_swan_(Cygnus_atratus).webm.1080p.vp9.webm` | 60,528,641 | `2b51dda28643af61a163d8b3457fd5885c596c51632576a53a6c4c06722630a4` |

They were not modified, copied, renamed, or staged. `videos/audit.mp4` remains
absent and was not minted.

## Closing fields

- Disposition: **EVIDENCE-BACKED COMPATIBILITY STOP**
- Topic oracle commit: **`6723f97`**
- Topic evidence commit: **`2b4d8ae`**
- Topic receipt commit: **`c51510a31b2c54d90f6825087026b50a65209def`**
- Integration commit on `feat/web-control-panel`:
  **`a5b0de1584ca0cad95895577c384a06915ae2047`**
- Exact-commit CI: **OBSERVED PASS** —
  [run 33060694392, attempt 2](https://github.com/Spacejunk-io/collide-o-scope/actions/runs/33060694392/attempts/2)
  passed at exact head `a5b0de1584ca0cad95895577c384a06915ae2047`:
  dependency policy in 35 seconds, Linux 24.04 in 540 seconds, macOS 15 in
  626 seconds, and Windows VS 2022 in 991 seconds. Attempt 1 was cancelled by
  the later GPU/UI integration push; it was not counted as green and this
  explicit rerun supplies the exact-head receipt.
- CI-form six-command gate: **OBSERVED PASS** — formatting and both JavaScript
  parsers; all-target/all-feature compile; 2,148 tests passed with zero
  failures and 163 explicitly ignored tests; all six benchmark probes
  succeeded; and Clippy passed with warnings denied
- Shipping-pin focused pixel/preset oracle: **OBSERVED PASS**
- Shipping-pin focused patch JSON/YAML oracle: **OBSERVED PASS**
- `b3af8c6` comparison: **OBSERVED PASS — byte-identical**
- `4284bf1` comparison: **OBSERVED STOP — all eight pixel hashes changed**
- `57369f2` comparison: **OBSERVED STOP — same eight changed hashes**
- Dependency/policy mutation retained: **NONE**
- Physical/aesthetic receipt: **NOT RUN — REQUIRED TO AUTHOR A NEW LOOK**
- Protected-root and `videos/audit.mp4` recheck: **OBSERVED PASS**

## Deliberate non-claims

This note does not claim the newer upstream pixels are incorrect, unsafe, or
unfit for all users. It does not claim the tests-only commit is a useful
upgrade. It is not an aesthetic or physical-output receipt, and it does not
authorize redefining old bytes, increasing worker stacks, minting
`videos/audit.mp4`, or treating hosted compilation as visual approval.
