# Maintenance web-stack migration — proof note

Date prepared: 2026-08-27
Topic: `feat/maintenance-web-stack`
Pinned integration base:
`96bcb734b4670ecd9941a8725c7941e5b66f1284`
Status: **topic implementation and full gate complete; integration and
exact-commit CI pending**

This is the §3.8(b) maintenance tranche. It advances the production control
server from axum 0.7.9 and axum-server 0.7.3 to axum 0.8.9 and axum-server
0.8.0, preserves one shared WebSocket protocol implementation, removes an
unused production dependency, and remeasures the locked release SBOM. It does
not change the action vocabulary, authentication model, listener topology,
release workflow, tag, or release artifacts.

## Selected graph and deliberate version ruling

The root graph now contains:

| Seat | Selected version | Ruling |
| --- | ---: | --- |
| `axum` | 0.8.9 | current reviewed 0.8 server/API line |
| `axum-server` | 0.8.0 | current reviewed TLS listener line |
| `tokio-tungstenite` | 0.29.0 | deliberately matches axum 0.8.9's exact WebSocket engine line |
| production `tower-http` | absent | removed because no production source used it |

The maintenance ladder's provisional tokio-tungstenite 0.30 address was not
followed blindly: axum 0.8.9 depends on tokio-tungstenite 0.29. Selecting 0.30
as the direct test client would retain production 0.29 and add a second 0.30
protocol implementation, so the tests would no longer exercise the same
Tungstenite parser and message types shipped by axum. `cargo tree --locked -i
tokio-tungstenite@0.29.0` instead shows one 0.29.0 package reached by axum and
the root dev dependency. No 0.30 package exists.

The root `tower-http` dependency had no `src/` use. Production static assets
are served by the bounded embedded implementation in `src/web/static_files.rs`.
Removing the dead direct edge also pruned its production-only support rows.
The independent `experiments/web-ui` prototype does use `ServeDir`; its own
manifest and lock therefore advance to axum 0.8.9 and tower-http 0.7.0 and pass
a separate locked check. This note does not claim that tower-http 0.7 ships in
the application.

## Source migration and bounded WebSocket posture

- Axum 0.8 brace captures replace the removed colon syntax:
  `/thumb/{filename}` and `/preview/{filename}/{index}`.
- Axum-server 0.8's address-generic shutdown handle is explicit as
  `Handle<SocketAddr>`. Its now-fallible `from_tcp` and `from_tcp_rustls`
  construction propagates errors into the existing listener-failure path.
- Axum and Tungstenite text messages now carry `Utf8Bytes`; every production
  and test constructor converts owned strings explicitly while receive-side
  byte parsing remains unchanged.
- Axum 0.8 raised the default WebSocket read buffer to 128 KiB. The server now
  pins it to 8 KiB while retaining the existing action-wire maximum for both
  message and frame size. This keeps per-connection allocation bounded without
  weakening the accepted action envelope.
- Authentication, exact same-origin checks, cookie separation, TLS identity,
  no-store/browser-hardening headers, action admission, and disconnect cleanup
  remain on their existing production seams.

## New transport regressions

`thumbnail_and_preview_routes_preserve_decoding_arity_and_cache_misses` starts
the real production router, seeds both caches under `set final.mp4`, and proves:

- percent-decoded authenticated thumbnail and indexed-preview paths return
  HTTP 200, `image/jpeg`, and exact cached bytes;
- a missing thumbnail and out-of-range preview index return 404; and
- missing or extra preview path segments return 404.

`bounded_reconnect_storm_retires_socket_state_and_keeps_fresh_snapshots`
performs 24 authenticated same-origin upgrades. Each client requests and
receives its own newly published full snapshot revision. Selected clients arm
the gyro stream and monitoring watch or begin and dirty a history gesture,
then every socket is dropped without a close handshake. The bounded proof
requires the watch receiver count to return to baseline, gyro streamers to
return to zero, the monitor watch to disarm, the dirty gesture's ordered End
barrier to enter the queue, and a new owner to acquire the released gesture
seat. The twenty-fourth reconnect still receives fresh authoritative state.

The existing real-socket round trip, exact-origin/cookie tests, independent
listener-crash test, fixed-port retire/rebind/secret-rotation test, and TLS
private-key mismatch test remain the surrounding compatibility matrix.

## Dependency policy and release SBOM

Axum-server 0.8 removed the sole `rustls-pemfile` edge. The corresponding
RUSTSEC-2025-0134 ignored-advisory exception is therefore removed from both
`deny.toml` and `policy/dependency-exceptions.toml`; keeping it would fail the
repository's `unused-ignored-advisory = "deny"` law. The other active exception
is unchanged.

The locked production graph shrank from 604 to 594 packages. A clean detached
worktree at implementation commit `8565c85812e65a90bde3419e9b13ad315411a4fb`
was generated with pinned `cargo-cyclonedx-cyclonedx 0.5.9` and the exact
commit epoch. The reviewed profile is now:

| Measure | Exact value |
| --- | ---: |
| Top dependency components | 359 |
| Target components | 6 |
| Registry components | 357 |
| Git components | 1 |
| Dependency rows | 360 |
| Dependency edges | 856 |
| Root edges | 35 |
| Local declarations | 8 |
| Rewritten source references | 13 |

The normalized semantic-profile SHA-256 is
`d8333dbde0a319d518a7461ee68d8e3617221ddbb0b47b99480c1b43ecd1942b`.
The normalized document contained 366 declared components and validated at
the exact candidate source URI. The policy self-test fixture now derives its
root-edge and three-edge-row topology from the reviewed constants, so a future
honest graph change does not require a magic fixture threshold while hostile
mutations remain fail-closed. The policy file's whole-file SHA-256 pin and the
checker’s duplicated edge/root/profile literals were updated together.

The temporary raw-SBOM worktree and normalized document were removed after
validation. No `collide-o-scope.cdx.json` was left in the repository. Existing
v1.8.1 lock/SBOM receipt hashes are historical evidence and remain unchanged.

## Repository and protected-artifact boundary

The three protected binary root artifacts remain the only non-ignored
untracked root artifacts. A final read-only size/hash recheck observed:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `.da-vinci-canon-pre-refinement-backup-20260822.zip` | 66,225 | `494b63ad0bd96cfb1c7f20a37ad574075a26dd289c14f9f24e71ffe48ab1eea4` |
| `4K_Nature_Cinematography_recorded_with_Nikon_D5300.webm.1080p.vp9.webm` | 56,984,527 | `ee1cfc47671617f8bdf8031dd19cb00f9359e4bf47bdddc7f1fca9df13d034a0` |
| `Black_swan_(Cygnus_atratus).webm.1080p.vp9.webm` | 60,528,641 | `2b51dda28643af61a163d8b3457fd5885c596c51632576a53a6c4c06722630a4` |

They were not modified, copied, renamed, or staged. `videos/audit.mp4` remains
absent and was not minted.

## Closing fields

- Topic implementation commits: **`8565c85`**, **`462787f`**, **`476511f`**
- Topic receipt commit: **PENDING**
- Integration commit on `feat/web-control-panel`: **PENDING**
- Exact-commit CI: **PENDING**
- CI-form six-command gate: **OBSERVED PASS** — formatting and both JavaScript
  parsers; all-target/all-feature compile; 2,146 tests passed with zero
  failures and 163 explicitly ignored tests; all six benchmark probes
  succeeded; and Clippy passed with warnings denied
- Focused migrated route and 24-client reconnect regressions: **OBSERVED PASS**
- Production graph unity/absence checks: **OBSERVED PASS** — one
  tokio-tungstenite 0.29.0; no production tower-http or rustls-pemfile
- Dependency policy/advisory/license/audit gates: **OBSERVED PASS**
- Pinned SBOM generation, normalization, validation, and hostile self-test:
  **OBSERVED PASS**
- Release-workflow policy and release-verifier self-tests: **OBSERVED PASS**
- Independent prototype locked check: **OBSERVED PASS**
- Protected-root and `videos/audit.mp4` recheck: **OBSERVED PASS**

## Deliberate non-claims

This note is not a release receipt and does not claim a changed release
workflow or newly published artifact. It does not claim that the independent
prototype's file server ships in production. It does not treat a different
tokio-tungstenite major/minor line as equivalent test coverage. Historical
release tags, assets, and evidence remain immutable.
