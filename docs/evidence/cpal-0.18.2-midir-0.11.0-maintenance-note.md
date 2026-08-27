# CPAL 0.18.2 / midir 0.11.0 — evidence-backed STOP note

Date prepared: 2026-08-27
Topic: `fix/maintenance-audio-midi-stop`
Pinned integration base:
`2c4dd7b0767d16e31f5b36237985abfd85c3b906`
Status: **STOP — retain CPAL 0.16.0 and midir 0.10.4 until the
cross-platform physical audio/MIDI gate exists**

This closes perfection handover §3.8(f) without treating compilation or a
version number as physical audio/MIDI proof. The current direct dependencies
remain `cpal = "0.16"` and `midir = "0.10"`; the locked nodes remain CPAL
0.16.0 and midir 0.10.4. No manifest or lockfile was repinned.

The same tranche closes two deterministic truth gaps discovered while auditing
the maintenance boundary:

- callback failures and non-loopback sample stalls now enter one terminal
  teardown path, which drops the stream, clears active-device facts, zeroes
  analysis state, and marks/removes an armed `ProgramAudioTap`; and
- the macOS microphone disclosure now truthfully says that live audio can be
  included in Program recordings only after the operator starts recording. A
  source-contract test pins the key and exact disclosure once and forbids the
  former false sentence, `Nothing is recorded`.

Those changes harden the pinned stack. They are not candidate-upgrade proof.

## Exact candidate identities

The authenticated local crates.io archives reviewed for this decision were:

| Candidate | Bytes | SHA-256 | Present ruling |
| --- | ---: | --- | --- |
| `cpal 0.18.2` | 233,947 | `6f02e8d0327b42d3e2e4ab2119af397344eb9fc54a34bf0ddeaa1277af8681f1` | **API / OS-FLOOR / PHYSICAL-AUDIO HOLD** |
| `midir 0.11.0` | 49,766 | `77c12e74a8604bc07f9a3fcf3d5889a81bc96ca6e07d6a114d63fc0371f3e5a4` | **PHYSICAL-MIDI / GRAPH HOLD** |

The controlled Rust 1.98.0 toolchain exceeds CPAL 0.18.2's declared Rust 1.85
floor. Toolchain compatibility is therefore not the deciding blocker.

## Current production boundary

`src/audio/mod.rs` owns all CPAL use. It enumerates input and WASAPI loopback
sources, selects a requested device without silently retargeting a named one,
builds the capture stream, drives the bounded FFT/modulation ring, and tees
raw interleaved Program PCM into the recorder-owned bounded tap. Current input
callbacks explicitly implement only F32, I16, and U16 conversion.

`src/midi/mod.rs` owns all midir use. It enumerates stable input/output names,
receives notes/CC/clock, emits controller feedback, preserves connection and
disconnect truth, and feeds the existing action/beat-latch laws. The Windows
dependency graphs are deliberately distinct today: CPAL uses `windows 0.54`
while midir uses `windows 0.56`.

The recorder makes an audio update broader than an analyzer-only change. A
candidate must preserve the exact Program PCM clock, channel layout, bounded
ring, explicit overflow silence, device-loss stop, and final mux receipt in
addition to preserving visual modulation.

## Why CPAL 0.18.2 stops

The candidate is a substantial API and backend migration, not a compatible
patch bump:

1. `DeviceTrait::name()` is replaced by structured `description()` plus
   `Display`/stable ID vocabulary. The application must decide which identity
   is authored, which label is shown, and how disconnect/re-enumeration avoids
   retargeting.
2. Stream construction and configuration APIs changed, including ownership
   of the stream configuration and the unified typed error model. The current
   callback stores every runtime error as terminal text; candidate
   `DeviceChanged`, `RealtimeDenied`, and `Xrun` have different continuation
   semantics. Treating all three identically would either kill a stream that
   remains active or conceal a discontinuity.
3. Candidate device selection can prefer F64, I32, or I24 configurations.
   Those are real WASAPI/CoreAudio/ALSA formats, while the current application
   has conversion callbacks only for F32, I16, and U16. An exhaustive compile
   match is not a physical sample-value oracle.
4. CPAL 0.18.2's own platform table names macOS 14.2, with loopback requiring
   macOS 14.6 or later. The bundle intentionally declares macOS 11.0. The
   candidate cannot silently raise that product floor, and a runtime-gated
   compatibility design has not been implemented or proven.
5. WASAPI, CoreAudio, ALSA, PipeWire/Pulse/JACK, and optional ASIO behavior and
   graphs changed. The present product depends specifically on Windows
   loopback enumeration/capture as well as ordinary microphones; hosted CI
   cannot demonstrate either device class.

The deterministic terminalization test added in this topic gives a future
migration one exact failure-state oracle. It does not replace the physical
matrix.

## Why midir 0.11.0 stops

The public calls used by the application are close enough for a bounded port,
but the release provides no evidence for this application's real device law.
The candidate changes target graphs (including Windows bindings), and optional
WinRT/CoreMIDI-timestamp/JACK choices remain product decisions rather than
automatic upgrades. None of those features independently proves:

- stable enumeration and explicit no-retarget behavior across hotplug;
- note, CC, and 0xF8 clock acceptance with the exact timeout fallback;
- output feedback ordering and loop suppression;
- long SysEx/hostile packet bounds;
- disconnect/reconnect terminal state; or
- simultaneous MIDI traffic, audio analysis, Program recording, browser,
  OSC, and render load without action duplication or clock drift.

Because CPAL and midir meet in the operator's timing workflow, they may share
a physical session but must remain separately attributable. A CPAL failure
cannot be masked by a MIDI success, or vice versa.

## Exact reopening campaign

1. Create an evaluation topic with exact candidate versions and archive
   identities. Freeze the old graph and candidate graph before changing code;
   review every added target package, feature, license, OS floor, and native
   runtime requirement.
2. Port CPAL behind one application-owned device identity, sample conversion,
   and typed runtime-error policy. Cover every selected sample format with
   limit±1, zero, full-scale, nonfinite, channel-interleave, and repeat
   oracles. Preserve the named-device no-retarget law.
3. Prove `DeviceChanged`, `RealtimeDenied`, `Xrun`, device removal, permission
   denial, stream stall, restart, and ordinary stop separately. Every terminal
   path must lose the Program tap exactly once; a documented nonterminal path
   must keep its clock truthful.
4. On named Windows hardware, exercise microphone and system-playback
   loopback, silence, hotplug/default-device changes, mono/stereo/multichannel,
   each admitted sample format, repeated start/stop, and a real Program MP4
   whose decoded audio/video duration and sample-count receipt pass.
5. On named macOS 11-compatible and current macOS hosts, prove ordinary input,
   TCC denial/grant, bundle disclosure, supported system-audio behavior, and
   whether the candidate preserves or truthfully changes the product floor.
6. On named Linux ALSA and the supported user audio service, prove enumerate,
   capture, contention, xrun recovery, daemon loss, hotplug, and recorder mux.
7. Port midir separately, then exercise named physical input and output devices
   on Windows, macOS, and Linux: note/CC/clock, feedback, hotplug, reconnect,
   timeout fallback, sustained traffic, and no loop/duplicate/retarget.
8. Run combined audio + MIDI + real recorder sessions under the standing
   1/3/8-layer render and controller load. Record hardware, OS, backend,
   driver, device IDs, sample format/rate/channels, clocks, counters, hashes,
   failures, and exact commits in machine-readable receipts.
9. STOP on any product-floor regression, unsupported format, silent
   retargeting, unclassified error, lost/duplicated action, audio-clock shift,
   recorder discontinuity, resource-cap increase, or platform without named
   physical proof.
10. Only after all deterministic and physical seats pass may the manifest and
    lock advance, followed by dependency policy/SBOM/reproducibility checks,
    the exact six-command gate, and exact-head hosted CI.

## Protected boundary

The three protected root artifacts are excluded from this work and remain
unmodified. `videos/audit.mp4` remains absent and must not be minted. A future
physical recorder campaign must write a new, explicitly scoped ignored
receipt; it must not invent the absent audit asset or overwrite historical
evidence.

## Closing fields

- Disposition: **EVIDENCE-BACKED STOP**
- Pinned versions retained: **CPAL 0.16.0 / midir 0.10.4**
- Deterministic terminalization/disclosure implementation commit: **PENDING**
- CI-form six-command gate: **PENDING**
- Topic integration commit on `feat/web-control-panel`: **PENDING**
- Exact-commit CI: **PENDING**
- Candidate manifest/lock change: **NOT ATTEMPTED**
- Windows physical audio/MIDI/recorder matrix: **NOT RUN**
- macOS physical audio/MIDI/recorder matrix: **NOT RUN**
- Linux physical audio/MIDI/recorder matrix: **NOT RUN**

## Deliberate non-claims

This note does not claim CPAL 0.18.2 or midir 0.11.0 is defective, insecure,
or permanently rejected. It does not claim source compatibility, successful
compilation, a hosted test, device enumeration, or a synthetic PCM/MIDI packet
is physical proof. It does not claim macOS 11 can run the candidate, that
system playback is a microphone, or that analyzer levels prove the recorder's
Program PCM clock and mux. It authorizes no dependency repin, OS-floor change,
new audio backend, device retargeting, or relaxation of resource limits.
