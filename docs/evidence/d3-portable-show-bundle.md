# D3 portable verified show-bundle receipt

Date: 2026-08-24<br>
Target: Collide-o-Scope v1.7.0 release-candidate working tree<br>
Disposition: deterministic core accepted; cross-machine/live-export promotion
gate remains open

## Implemented contract

- `src/show_bundle.rs` implements build, side-effect-free inspect/preview, and
  transactional import for binary `COSBNDL\0` format version 1.
- The manifest is canonical JSON in strict lexical entry-path order. Payloads
  are uncompressed and contiguous, making stored length equal expanded length
  and fixing the only admitted expansion ratio at 1:1.
- A canonical hostile-round-trip patch is rewritten to stable
  `cos-sha256://<digest>/<bytes>` original-media identities. Every referenced
  identity must have an original entry. Proxies are explicitly derived, link
  to an original included in the same bundle, never rewrite patch identity,
  and are reported non-authoritative.
- Canonical Study, gesture, and performance-take sidecars are regenerated from
  the patch, parsed again, and required to match the patch's exact identity set
  during inspection. Optional controller/venue profiles and receipts are added
  only from explicit bounded inputs. No arbitrary directory collector exists.
- The preview reports format/bundle/patch identities, total/expanded bytes, and
  each logical path, role, digest, size, license, and authoritative status. It
  performs no library writes.

## Admission and publication law

- Default caps: 64 GiB bundle, 4 MiB manifest, 4,096 entries, 64 GiB per media
  entry, 32 MiB per authored document, 64 GiB aggregate expanded bytes, depth
  4, and 240 ASCII bytes per portable component. Counts and metadata lengths
  are checked before media hashing or manifest allocation; bytes and totals are
  checked again while streaming.
- Absolute, drive/UNC, traversal, empty, control, Windows device, separator,
  duplicate, and ASCII case-fold-colliding names are rejected. Source media,
  bundle input, import root, existing generation, and cleanup targets receive
  no-follow type checks. The format has no link/device entry type and rejects
  unknown types and compressed/ZIP input.
- Build uses the P3 same-directory `StagedPublication`: unpredictable
  create-new staging, complete re-read/hash, file flush/sync, atomic replace or
  true no-replace, then parent sync.
- Import verifies the entire bundle before mutation, writes and hashes a
  complete create-new generation, syncs files and directories, and performs
  one true no-replace directory publication. `ReuseVerified` rehashes every
  existing file; an external final-name winner is never overwritten.
- Cancellation and injected I/O failures drop unpublished staging. Startup
  cleanup is bounded and removes only direct children matching the fixed
  `.cosbundle-import-stage-*.part` namespace.
- Bounded authored text and metadata are scanned for conservative
  secret-bearing markers. This can reject an artistic string that resembles a
  credential; rejection is the intended safe stop, and media bytes are never
  interpreted as configuration.

## Deterministic/adversarial evidence

Command (with the prescribed Visual Studio, FFmpeg, and libclang environment):

```text
cargo test --locked --bin collide-o-scope show_bundle::tests:: -- --nocapture
```

Result: **12 passed, 0 failed, 0 ignored**.

The suite proves deterministic byte identity over property-generated media;
canonical build/preview/import and verified reuse; original/proxy identity and
authority; optional venue/receipt documents; atomic replace; one-byte-over
entry/document/aggregate/bundle/count caps; traversal/absolute/device/
case-fold/duplicate rejection; unknown link type and ZIP envelope rejection;
tamper, short-read, and missing-original refusal before import mutation;
no-follow source refusal; injected build/import disk-full, cancellation, and
prepublication crash behavior; bounded orphan cleanup; and build/import
final-name races that preserve the external winner.

The targeted test build emitted no D3 warning. The only warning was the shared,
pre-existing unused `FlightRecorder::start` constructor.

## Remaining promotion gate

No two-machine fixture or prescribed live-export A/B harness is present in the
workspace. The tests use a freshly created empty import library and prove exact
stable-media/patch resolution there, but they do not claim machine-A export to
clean-machine-B live-export reproduction. Operator UI wiring is also outside
this core tranche. Those two items remain explicit promotion stop conditions;
the independently safe deterministic core does not weaken them.
