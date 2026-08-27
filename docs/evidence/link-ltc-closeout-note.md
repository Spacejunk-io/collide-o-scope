# Ableton Link and LTC closeout — defer and doctrine-stop note

Date: 2026-08-27

Topic: `docs/link-ltc-closeout`

Pinned integration base: `d43cffaa596952741f669e018d3ea63c93aa6c01`

Status: **Ableton Link remains deferred; LTC/MTC remains a doctrine change.**

This tranche closes perfection handover §3.7 without adding a dependency,
worker, socket, audio consumer, schema field, action, UI control, capability
claim, or patch byte. It records why neither external clock source has earned
an implementation seat under the current evidence law.

## Existing deterministic seam

The existing external-clock precedent is MIDI Timing Clock (`0xF8`). Accepted
pulses update the beat latch; loss re-anchors the internal clock rather than
making wall time or a remote peer authoritative. A future Link source may feed
that same optional input seam only after its gates below pass. It must not
change patch determinism, silently persist remote session state, or make live
network timing authoritative for replay or offline export.

LTC is not merely another pulse producer. Chasing timecode raises unresolved
transport semantics: discontinuity, dropout, seek, frame-rate conversion,
freewheel, pause/freeze, recording, replay, live/export, and loss. Standing
directive N01 therefore remains controlling: until a separate doctrine RFC
settles those cases, no LTC/MTC schema, backend, audio callback consumer, or UI
may land. The existing `ProgramAudioTap` tee is only a bounded callback design
precedent; it is not authorization to attach a timecode decoder.

## Official Link baseline checked

The candidate review was pinned to the official Ableton material available on
2026-08-27:

- Link 4.0 is the current stable release (`e9a2e41` as shown by the official
  release), adding Link Audio;
- the official implementation is header-only C++17 and its documented Windows
  floor is Visual Studio 2022;
- the grant is GPL-2.0-or-later or a separate proprietary license. The
  “or-later” grant is compatible with this repository's
  `GPL-3.0-or-later`; the repository trap remains GPL-2.0-only, not Link's
  actual GPL-2.0-or-later option;
- Ableton's Test Plan, not compilation alone, is the interoperability oracle.

Primary records:

- <https://github.com/Ableton/link/releases/tag/Link-4.0>
- <https://github.com/Ableton/link/blob/master/README.md>
- <https://github.com/Ableton/link/blob/master/LICENSE.md>
- <https://github.com/Ableton/link/blob/master/TEST-PLAN.md>

## Published Rust candidate matrix

Checksums below are the exact crates.io package checksums observed during this
review. An acceptable choice must be a published, lockable artifact; an
unpublished Git tag is not a substitute.

| Candidate | Exact published identity | License / implementation | Blocking evidence gap | Ruling |
| --- | --- | --- | --- | --- |
| `rusty_link` | `0.4.9`; `4169045a50ee3c874ee11128b8f06a46947776b23e3ee5f4bca293b6f3bb6f07` | GPL-2.0-or-later; CMake/bindgen wrapper embedding Ableton source | Its changelog identifies Link `4.0.0b3`, not stable 4.0; native toolchain/offline packaging and the full physical Test Plan remain unproved | **DEFER** |
| `ableton-link-rs` | `0.1.2`; `567dd73ccb0cc603f3eb83d81cea167292ae832d8f5b3879ecd293bba0b726e3` | GPL-3.0; native Rust | Repository tags `v0.2.0` and `v0.3.0` exist, but crates.io still exposes `0.1.2`; the newer work is therefore not a published registry-lock candidate and has no project physical receipt | **DEFER** |
| `ableton-link` | `0.1.0`; `3822f1325ab253e39437a2c8f704d90b554870bba4ed895b9a2d178c944adced` | Old native wrapper; license file describes GPL-2.0-or-later absent a proprietary grant | Published in 2019, behind the official stable baseline, with no current packaging or interoperability proof | **DEFER** |
| `ablink` | `0.1.0`; `f02b37398043bf65e355cb6943dd0c227d2235517ba1f7830cf78495ca708062` | MIT; pure Rust | crates.io names `https://github.com/akx/ablink`, but that repository was not reachable in this review; source provenance, notices, maintenance identity, and physical behavior therefore cannot be independently closed | **DEFER** |
| `link-bpm-rs` | `0.1.0`; `f99ab0709270dce39ecf3b35deae2a13ddc0446ce294c05af9ca02ee2735626a` | MIT OR Apache-2.0 | Reads advertised tempo/BPM only; it is not a complete phase/start-stop session source and has no full Test Plan receipt | **DEFER** |

No row clears artifact identity, license/notices, current upstream semantics,
build/offline policy, bounded lifecycle, and physical interoperability at the
same time. Choosing the least incomplete row would turn an evaluation gap into
a product claim, so no Link dependency is added.

## Link reopening gate

A future Link campaign must supply all of the following in one candidate:

1. an exact published registry artifact and checksum, with reachable source,
   notices, and an allowed license expression;
2. an identified official stable Link source/version, with no beta drift;
3. reproducible Windows/macOS/Linux build and offline/vendor behavior under
   the repository's locked dependency and supply-chain policies;
4. a bounded, explicitly stopped network/thread lifecycle that never mutates a
   patch beyond a separately versioned input-config section;
5. an optional source feeding the existing beat-latch seam, with deterministic
   internal-clock fallback on loss and explicit latency-compensation law;
6. Ableton Test Plan evidence across at least two physical peers/apps,
   including tempo, phase, start/stop, join/leave, dropout/timeout, recovery,
   and the claimed latency behavior.

Only after those gates pass may a separate implementation topic choose a
binding. This note does not reserve a crate, architecture, action name, or
capability status.

## LTC/MTC reopening gate

Standing directive N01 remains an exact **DOCTRINE_CHANGE** stop. Reopening
requires a dedicated RFC that decides whether external time merely generates
deterministically accepted events or becomes authoritative transport, and
defines every discontinuity/dropout/seek/rate-conversion/freewheel/pause/
record/replay/live/export/loss case. Until that RFC is accepted, a decoder
experiment may not cross into Cargo dependencies, cpal capture, PatchState,
wire actions, the browser/native panels, performance takes, or export.

## Repository and protected-artifact boundary

This closeout changes only tracked evidence. It deliberately leaves
`Cargo.toml`, `Cargo.lock`, source, workflows, generated capability records,
and shipped documentation untouched. The three protected untracked root
artifacts were not opened, copied, renamed, or staged. `videos/audit.mp4`
remained absent and was not minted.

## Closing fields

- Topic evidence commit: **`a25a55c`**
- Topic receipt commit: **PENDING**
- Integration commit on `feat/web-control-panel`: **PENDING**
- Exact-commit CI: **PENDING**
- Hosted full gate: **OBSERVED PASS** — the exact six-command CI-form gate
  passed: formatting and both JavaScript parsers; all-target/all-feature
  compile; 2,143 tests passed with zero failures and 163 explicitly ignored
  external/GPU seats; all six bench harnesses reported success; clippy passed
  with `-D warnings`
- Protected-root and `videos/audit.mp4` recheck: **OBSERVED PASS** — the three
  protected root files remain the only untracked root artifacts at 66,225 /
  `494b63ad...ab1eea4`, 56,984,527 / `ee1cfc47...13d034a0`, and 60,528,641 /
  `2b51dda2...722630a4`; `videos/audit.mp4` remains absent

## Deliberate non-claims

This is not a Link interoperability receipt, LTC decoder prototype, latency
measurement, compatibility endorsement, or dependency selection. It makes no
claim that the named crates are unsafe or permanently unsuitable; it records
only that none presently clears every repository gate. It does not alter the
operator-gated status of any prior physical or cross-machine receipt.
