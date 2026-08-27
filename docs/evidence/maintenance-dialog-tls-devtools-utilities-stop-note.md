# Dialog, TLS, developer-tool, and utility maintenance — evidence-backed STOP

Date prepared: 2026-08-27  
Topic: `docs/maintenance-dialog-tls-devtools-utilities-stop`  
Pinned audit base: `05c8d6cd399843236ea393e15f41a74d4b793913`  
Current integration base: `7fd0221a66d7ed8d87994eb91bf833e40e4fad1c`

Status: **STOP — retain rfd 0.15.4, rcgen 0.13.2, criterion 0.7.0,
direct getrandom 0.3.4, pollster 0.4.0, and sha2 0.10.9. Getrandom
0.4.3 is a bounded update candidate, not a proved promotion.**

This closes the remaining dialog, TLS, developer-tool, and utility dependency
review as a bounded negative result. No candidate manifest, lock, source,
benchmark, build-script, fuzz, workflow, vendor, or release change was
attempted. A STOP is completion here: compilation or an attractive version
number cannot pay the physical, persistence, methodology, liveness, and byte
identity gates these seats own.

## Authenticated candidate ruling

| Seat | Current locked state | Reviewed candidate | Candidate `.crate` SHA-256 | Disposition |
| --- | --- | --- | --- | --- |
| Native dialogs | rfd 0.15.4 | rfd 0.17.2 | `20dafead71c16a34e1ff357ddefc8afc11e7d51d6d2b9fbd07eaa48e3e540220` | **PHYSICAL-UI HOLD** |
| TLS identity | rcgen 0.13.2 | rcgen 0.14.9 | `091e7a8e7d86e6feb87a27ce8e2cba29d49eff9507afeebefab7eeb2ca667fb4` | **API/PERSISTENCE HOLD** |
| Benchmark harness | criterion 0.7.0 | criterion 0.8.2 | `950046b2aa2492f9a536f5f4f9a3de7b9e2476e575e05bd6c333371add4d98f3` | **BENCHMARK-METHODOLOGY HOLD** |
| Direct OS entropy | getrandom 0.3.4 | getrandom 0.4.3 | `300e883d756b2e4ec94e02791f39b04b522276138852cfc41d9fb7e904106099` | **BOUNDED UPDATE CANDIDATE — NOT YET PROVEN** |
| Blocking executor | pollster 0.4.0 | pollster 1.0.1 | `bc6355899e1c9462875b6757c79f3caa011a1fdae12bbb1a2e72dd1f234f8336` | **EXECUTOR/GPU HOLD** |
| SHA-256 implementation | sha2 0.10.9 | sha2 0.11.0 | `446ba717509524cb3f22f17ecc096f10f4822d76ab5c0b9822c5f9c284e825f4` | **BYTE-ORACLE HOLD** |

The archive hashes identify the reviewed bytes; they do not prove application
behavior. These are evidence holds, not permanent rejections.

## Exact audited dependency boundary

The manifest, lock, fuzz, and build-identity inputs audited at `05c8d6c`
remain byte-identical at `7fd0221`:

| Artifact | Bytes / packages | SHA-256 |
| --- | ---: | --- |
| `Cargo.toml` | 3,141 bytes | `58a33db92ab8a27d24fa50658fba4535fa00548a2b813ebcfef04adfa21126dd` |
| `Cargo.lock` | 151,355 bytes / 594 packages | `dfba17a889e054cdae6885cf7c07cca8636e95d56539308a04f0944c02031489` |
| `fuzz/Cargo.toml` | 1,532 bytes | `2b227cd80fd9c5f05c3849e6bf7cd455b3429792b9fa0ee9a905bcf9dffddb17` |
| `fuzz/Cargo.lock` | 8,447 bytes / 38 packages | `8d26b0aefa0ab763d36be43ea6cbb178565f948d250bb1dc7092dc4eaa3b9ee5` |
| `build.rs` | 19,885 bytes | `feb75ceb9dd27ce2956f3936696f47d1cdcf6f24ded50fbdea54a3305ac8c048` |

The current source census found:

- ten `rfd::FileDialog` constructions in two files, covering open, save,
  folder, cancellation, filters, default directory, and default-name paths;
- four qualified rcgen references plus two `CertifiedKey::key_pair` field
  uses around certificate generation and persisted-certificate validation;
- one Criterion benchmark source, two macro invocations, and six named groups;
- ten `getrandom::fill` calls across seven production files;
- 158 `pollster::block_on` calls across 27 files; and
- 91 `Sha256`-bearing lines across 26 Rust files, including the two P4c
  Windows-test helper lines added after the audit base. The sha2 0.11 migration
  hazard remains the 25 digest-output `LowerHex` formatting sites across 13
  files; the new P4c helper already byte-encodes and adds no such site.

The root graph contains getrandom 0.2.17 through ring, 0.3.4 through the
application/ahash/rand_core, and 0.4.3 through jobserver/tempfile. Repointing
the direct edge does not consolidate the graph. Rfd requires pollster 0.4, so
moving only the application's direct edge to pollster 1.0 creates a split
executor graph. Sha2 is declared by the root normal and build graphs and by
the independent fuzz graph; a migration is necessarily one atomic campaign.

The resolved root feature boundary is also part of the ruling: rfd carries
`default`, `xdg-portal`, `async-std`, `ashpd`, `pollster`, and `urlencoding`;
rcgen carries `default`, `crypto`, `pem`, `ring`, and `x509-parser`; Criterion
carries `default`, `rayon`, `plotters`, and `cargo_bench_support`; direct
getrandom resolves `std`; pollster has no feature enabled; and sha2 resolves
`default,std`. The controlled Rust 1.98.0 toolchain satisfies the published
candidate floors, but an MSRV comparison is not behavioral proof.

The independent fuzz and experiment locks are not opportunistic cleanup
targets. The fuzz graph owns sha2 0.10.9 directly and getrandom 0.4.3 through
`libfuzzer-sys -> cc -> jobserver`; experiment locks separately carry their
own getrandom versions. Candidate licenses fit already-admitted policy
families, but any newly resolved target package or native runtime must still
pass the ordinary license, source, and SBOM court.

## Why the candidates stop

### rfd 0.17.2 — physical UI boundary

The used `FileDialog` methods remain source-shaped, but the Linux portal path
was rewritten, old async-runtime features disappeared, and the default runtime
boundary now includes portal/libdbus behavior with a Zenity fallback. A build
cannot prove file/folder/save semantics, URL-decoded paths, filters,
cancellation, overwrite handling, focus return, or the modal-clock
pause/resume law. Windows and macOS behavior also require real UI sessions.

Reopening requires the complete dialog matrix on Windows, supported macOS,
and Linux GNOME/KDE under supported Wayland/X11 seats, including missing
portal/libdbus, fallback or explicit refusal, Unicode, permission failure,
cancel/window-close, focus restoration, and no destination write on failure.

### rcgen 0.14.9 — identity persistence boundary

`CertifiedKey::key_pair` becomes `signing_key`. More importantly,
`CertificateParams::from_ca_cert_der` no longer provides the project's
certificate-only SAN inspection boundary. The candidate `Issuer` route
requires a signing key and does not expose the SAN collection used by the
persisted version-1 identity envelope validator.

Reopening requires one app-owned, directly pinned SAN parser; frozen 0.13.2
identity fixtures loading without rotation or digest change; generation,
required-SAN expansion, key matching, atomic publication, corruption,
permission, truncation, and injected-fault tests; and live Windows/macOS/Linux
rustls loopback and LAN handshakes whose advertised address is in the actual
certificate.

### criterion 0.8.2 — measurement-method boundary

The synchronous API is close, but 0.8 introduces alloca-based memory-layout
randomization and changes the measurement instrument. A benchmark executable
exiting zero does not make its distributions comparable with the six retained
Criterion 0.7 probes or their history.

Reopening requires a frozen 0.7 host/invocation/input/power policy and raw
results, at least ten alternating retained/candidate sessions on the same
quiet pinned host, preserved samples and tail statistics, and an explicit new
methodology/baseline ruling before candidate values are compared to history.

### getrandom 0.4.3 — bounded but unproved

The ten direct `fill` calls are source-compatible, and the candidate contains
corrected Windows `ProcessPrng` error handling. It is nevertheless already in
the graph transitively, while 0.3.4 and 0.2.17 remain owned elsewhere. The
direct calls guard authentication material and transactional staging names.

Reopening requires a direct-edge-only topic with frozen inverse trees and
SBOM facts; deterministic entropy-failure injection at all ten calls; forced
collision proof preserving each existing retry ceiling and never overwriting;
and successful affected operations on Windows, macOS, and Linux with fixed
token encoding, no secret logging, atomic cleanup, and unchanged resources.

### pollster 1.0.1 — executor/GPU boundary

The called `block_on` signature remains familiar, but the implementation
moves from mutex/condition-variable signaling to thread park/unpark with a
reusable thread-local waker. The 158-call surface is predominantly wgpu work,
and rfd still owns a pollster 0.4 edge.

Reopening requires deterministic pending/wake, cross-thread wake, completion,
panic, drop, and cancellation tests; software-adapter coverage of every
adapter/device, error-scope, pipeline, readback, and teardown path; and
repeated physical receipts on Windows Vulkan AMD/Intel Arc and DX12, Linux
Vulkan, and macOS Metal without hangs, lost errors, or resource growth.

### sha2 0.11.0 — canonical-byte boundary

Sha2 0.11 moves to digest 0.11 and newtype output. Those outputs no longer
implement the `LowerHex` formatting used at 25 application sites. The hashes
are persistent identities and byte oracles, not cosmetic strings.

Reopening requires frozen fixed-input/lowercase-encoding oracles, a single
explicit byte-to-lowercase-hex encoder, and an atomic root-normal/root-build/
fuzz manifest plus root/fuzz lock migration. Every fixed payload must remain
exactly 32 bytes rendered as 64 lowercase hexadecimal characters. The lock's
own changed hash and derived build identity must be re-ledgered honestly.

## Coupling and campaign order

- Rfd and pollster stay separately attributable; neither may hide the other's
  split graph or physical/liveness proof.
- Rcgen persistence is proved against retained sha2 bytes before any sha2
  migration, and rcgen and direct getrandom land serially around TLS identity.
- Criterion 0.8 cannot be the measuring instrument for another dependency's
  campaign until its own methodology has been ruled.
- Sha2 moves across root normal, root build, root lock, fuzz manifest, and fuzz
  lock together. Partial or split movement is a STOP.
- Any admitted candidate re-runs dependency policy, license/SBOM,
  reproducibility, the exact six-command gate, and exact-head hosted CI.
- `build.rs` hashes the complete root lock into build identity. Even a dev-only
  dependency change therefore invalidates current lock/SBOM/release facts;
  historical release evidence must never be rewritten.

## Repository and protected-artifact boundary

The authorized tracked write is this evidence note only. It does not alter
either manifest or lock, `build.rs`, source, benches, fuzz targets, workflows,
vendor bytes, capability records, or release artifacts.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `.da-vinci-canon-pre-refinement-backup-20260822.zip` | 66,225 | `494b63ad0bd96cfb1c7f20a37ad574075a26dd289c14f9f24e71ffe48ab1eea4` |
| `4K_Nature_Cinematography_recorded_with_Nikon_D5300.webm.1080p.vp9.webm` | 56,984,527 | `ee1cfc47671617f8bdf8031dd19cb00f9359e4bf47bdddc7f1fca9df13d034a0` |
| `Black_swan_(Cygnus_atratus).webm.1080p.vp9.webm` | 60,528,641 | `2b51dda28643af61a163d8b3457fd5885c596c51632576a53a6c4c06722630a4` |

They remain unmodified and unstaged. `videos/audit.mp4` remains absent and
must not be minted as substitute dialog, TLS, benchmark, entropy, executor,
or hash evidence.

## Closing fields

- Disposition: **EVIDENCE-BACKED STOP**
- Pinned versions retained: **rfd 0.15.4 / rcgen 0.13.2 / criterion 0.7.0 /
  direct getrandom 0.3.4 / pollster 0.4.0 / sha2 0.10.9**
- Getrandom 0.4.3 promotion: **NOT YET PROVEN**
- Candidate manifest/lock mutation: **NOT ATTEMPTED**
- Topic evidence commit: **PENDING**
- Integration commit on `feat/web-control-panel`: **PENDING**
- CI-form six-command gate: **PENDING**
- Exact-commit CI: **PENDING**
- rfd physical dialog matrix: **NOT RUN**
- rcgen persistence/live-handshake campaign: **NOT RUN**
- Criterion side-by-side methodology campaign: **NOT RUN**
- getrandom failure/collision/platform campaign: **NOT RUN**
- pollster executor/GPU campaign: **NOT RUN**
- sha2 frozen-byte migration: **NOT RUN**
- Protected-root and `videos/audit.mp4` recheck: **OBSERVED PASS**
- Recovery: **evidence-only; revert the eventual note commit if rejected — no
  dependency state requires restoration**

## Deliberate non-claims

This note does not claim any candidate is defective, insecure, or permanently
rejected. It is not a dialog compatibility receipt, TLS persistence receipt,
benchmark comparison, entropy-quality certification, executor/GPU receipt, or
sha2 0.11 byte-compatibility proof. It authorizes no new native runtime,
silent TLS identity rotation, benchmark-baseline redefinition, weakened
collision behavior, split executor graph, digest-input change, resource-cap
increase, or substitute `videos/audit.mp4`.
