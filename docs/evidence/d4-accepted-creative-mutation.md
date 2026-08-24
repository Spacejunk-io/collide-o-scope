# D4 accepted creative mutation receipt

## Outcome

D4 replaces the pre-application performance recorder tap with one typed,
post-validation admission boundary.  A transport action may prepare a
`CreativeMutationCandidate`, but only an applier success path can promote it to
`AcceptedCreativeMutation`.  The accepted item carries the existing v1
`PerformanceControl`, `PerformanceValueLaw`, raw carrier, canonical `u16` value,
and a bounded process-only origin.  The accepted-frame gate supplies the tick
and calls the unchanged `PerformanceTake::record_accepted` encoder.

No Scene field, Scene binding, performance address, replay action, take document
field, algorithm version, checksum domain, or prepared-Scene vocabulary changed.

## Complete origin inventory

`ActionSourceClass` is the engine ingress vocabulary.  D4 exhaustively matches
all seven variants; adding a variant without a policy is a compile error.

| Engine origin | Existing ingress/application path | D4 policy |
| --- | --- | --- |
| Browser | authenticated `ActionEnvelope<WebAction>` drain | recordable after live acceptance |
| Phone | phone gyro/pad/bend/gesture envelope classes | explicitly covered; current phone vocabulary has no v1 take address, and D4 does not invent one |
| Native | recovery controls and transform-gizmo/nudge typed actions | recordable after live acceptance; safety/recovery values remain outside the take vocabulary |
| MIDI | decoded `RuntimeControlAddress` through `apply_automation_control` | recordable after typed normalization and live acceptance |
| OSC | authenticated/bounded OSC event through the same typed adapter | recordable after typed normalization and live acceptance |
| Automation | host automation through the same typed adapter | recordable after typed normalization and live acceptance |
| Replay | compiled v1 event dispatch | always excluded, independent of the recorder guard flag |

Browser and phone retain the source from their action envelope before consuming
its payload.  MIDI, OSC, and host automation map their closed
`AutomationOrigin` before applying the generated `WebAction`.  Native and replay
call the same source-aware application seam explicitly.  Raw MIDI bytes, OSC
packets, browser JSON, and native pointer events are never take events.

## Admission and exclusion law

1. Existing transport parsing/authentication and bounded queues run first.
2. Existing stable-layer identity resolution and the closed v1 record view
   produce a typed address and raw carrier.
3. The existing address law normalizes the carrier to its canonical `u16` code.
4. The normal live applier performs value validation, revision checks, Morph
   safety release, planner/preflight, and live commit.
5. Only an explicit success marker promotes and stages the typed mutation.
6. The next accepted program frame supplies the 30 Hz reference tick and calls
   the unchanged v1 encoder.

The following record nothing: malformed/unrepresentable values, stale or absent
stable IDs, stale revision/planner refusal, failed Morph safety release,
blackout/freeze/pause, topology and routing edits, recorder controls, replay,
quantized work not yet released, and the unprocessed remainder of a dropped
browser batch.  Unsupported keys in an otherwise recordable action family keep
the existing visible counter; rejected values and pending-cap overflow keep the
existing rejected counter.

## Duplicate and bound law

Simultaneous duplicate delivery is equality of canonical
`(PerformanceControl, PerformanceValueLaw, value_code)` within one pending
accepted-frame interval.  Origin is intentionally absent from equality and from
serialization.  The first delivery wins process-only provenance; later equal
deliveries are no-ops.  A different canonical value retains ordering, and the
same value on a later accepted tick remains a new performance event.

Bounds remain explicit:

- pending accepted mutations: 512 (`MAX_PENDING_ACTIONS`), one-over is counted
  and refused;
- take events: 16,384;
- distinct take addresses: 256;
- serialized take document: 512 KiB;
- origin vocabulary: six live variants plus one replay-excluded variant;
- canonical value lane: one `u16` under the existing v1 address law.

No wall-clock value enters candidate identity, duplicate identity, the take, or
its hash.

## v1 compatibility proof

The D4 item is destructured back into the exact three arguments the old frame
gate passed to `PerformanceTake::record_accepted`.  The origin and precomputed
canonical code never enter the document.  Machine tests build a take through D4
and by a direct frozen-v1 encoder call, then require identical canonical bytes
and SHA-256.  The fixed brightness/tick fixture remains
`be4bb410f3984214fc13667f4135208d089d14aefbbff2fb2f6e19ff5a0758d6`.
A second test requires browser, phone, native, MIDI, OSC, and host
automation provenance to produce the same address table, event tick/value,
canonical bytes, and checksum.

`PERFORMANCE_ALGORITHM_VERSION` remains 1 and
`PERFORMANCE_CHECKSUM_DOMAIN` remains
`collide-o-scope/performance-take/v1\0`.

## Machine gates

The focused gate covers:

- exhaustive origin policy and replay exclusion;
- all-live-origin canonical address/value/tick/hash equality;
- exact D4-versus-direct-v1 byte/hash equality;
- cross-origin same-frame duplicate collapse;
- pending cap plus one refusal;
- browser/native/MIDI/OSC/automation live application parity;
- refused value, stale stable ID, safety action, replay, and dropped work record
  nothing;
- the established B9 performance recorder/replayer tests and v1 document tests.

Executed on Windows/MSVC with the repository's FFmpeg 8.1.2 shared development
prefix and locked dependencies:

- `cargo test --locked --bin collide-o-scope creative_mutation::tests` — 7
  passed, 0 failed;
- `cargo test --locked --bin collide-o-scope d4_` — the four D4 application
  fixtures passed (plus one unrelated symmetry D4-group fixture selected by the
  substring), 0 failed;
- `cargo test --locked --bin collide-o-scope performance_track::tests` — 20
  passed, 0 failed;
- `cargo test --locked --bin collide-o-scope record_tap` — 2 passed, 0 failed;
- `cargo test --locked --bin collide-o-scope a_take_replays` — 1 passed, 0
  failed;
- `cargo check --locked --bin collide-o-scope` — passed with no warnings.

## Stop conditions

D4 stops at the accepted mutation/take boundary.  It does not add recordable
topology, routes, Scenes, transport/safety controls, raw device events, wall
clocks, origin fields to take documents, a second live applier, or a v2 take
format.  Any future origin or performance control must extend the exhaustive
origin policy or append the separately versioned performance vocabulary and its
compatibility fixtures; it may not silently pass through a stringly fallback.
