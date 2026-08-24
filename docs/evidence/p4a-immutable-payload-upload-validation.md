# P4a immutable decoded payload and upload validation receipt

Date: 2026-08-24<br>
Baseline named by the audit: `v1.6.0` / `000411d`

## Landed contract

- `DecodedImagePayload` freezes packed RGBA bytes behind one `Arc` inner. Its
  stable payload identity follows the physical frame through forward delivery,
  reverse-cache retention, GPU validation, and read-only consumers.
- A pooled physical allocation owns exactly one `DecodedImageLease`. Logical
  forward, reverse-cache, upload, and read-only handles increment fixed owner
  counters but never reserve the same bytes again.
- Every decoder owns a fixed-format/fixed-stride `DecodedRasterPool`, capped at
  two idle slots and at the decoder's reverse-cache allowance plus two packed
  frames. A buffer returns to the pool only from the payload inner's final
  `Drop`; no live `Arc`, cache entry, or upload epoch can observe later writes.
- FFmpeg's scaler destination is retained, and warmed packed materialization
  reuses an exclusive pooled `Vec`. Allocation instrumentation distinguishes
  physical allocations, pool reuses, unavoidable stride/format copies, and
  compatibility/reference copies.
- Reverse-cache insertion and hits share the same payload allocation and copy
  zero pixel bytes. A hit may retag selection metadata only under the existing
  generation law. Codec motion survives only when its exact source generation,
  destination PTS/ordinal, and payload pairing remain valid; hostile or
  retagged identities cannot be laundered.
- Prepared-source activation and proxy hot adoption carry the same immutable
  payload identity and exact source/render generations into upload validation.
- Live selected-frame and Spout writes enqueue three popped wgpu error scopes
  and return without `block_on`, `Maintain::Wait`, or a device wait. Main owns
  one nonblocking `PollType::Poll` progress turn, after which each layer drains
  ready futures once.
- Validation retains at most two uploads per layer, four bounded fault records,
  and four event-loop turns of pending age. Every fault reports upload, layer,
  payload, source generation, render generation, and scope kind. Renderer
  replacement invalidates stale epochs. Terminal OOM is never hidden by fault
  saturation and enters the existing supervised GPU-health close law.
- Aggregate decoded-image instrumentation reports live/peak/max physical bytes,
  physical allocations, reuse, materialization/reference copied bytes,
  invalidations, and logical owner counts. The shared policy ledger has a
  256-MiB minimum ceiling so its cap is not weaker than already-admitted safe
  media planning; per-decoder pools remain independently bounded.

## Deterministic proof cases

The retained tests prove:

1. one physical charge across forward/cache/upload/read-only handles and six
   cache owners, with zero reference-copy bytes;
2. warmed buffer reuse with no new engine allocation and no overwrite while an
   older payload `Arc` remains live;
3. refusal at exactly one byte over the aggregate cap and complete release to
   baseline after cache eviction/source retirement;
4. tight and padded RGBA materialization preserve exact legacy bytes;
5. reverse insert/hit/eviction, forward mailbox delivery, seek generation,
   reverse selection, loop generation, proxy adoption, cancellation, and codec
   motion retain or invalidate the exact paired identity as required;
6. upload queue saturation, pending age, fixed deadline, stale renderer
   invalidation, exact Validation/Internal/OOM attribution, terminal OOM
   retention under fault saturation, and the health-latch bridge;
7. source scans of the selected upload body contain no blocking wait or device
   poll. The render loop contains the sole bounded nonblocking progress poll.

## Verification gates

Executed in the shared Windows workspace with the prescribed FFmpeg, LLVM, and
Visual C++ environment:

- `cargo fmt --all -- --check`: pass.
- `git diff --check`: pass; Git emitted only existing LF-to-CRLF notices for
  `.gitignore` and `Cargo.lock`.
- `cargo check --locked --bin collide-o-scope`: pass. The only warnings in that
  run were unused items in the separately owned, intentionally stopped P4c
  prototype.
- `cargo test --locked --bin collide-o-scope --no-run`: pass; three existing
  flight-recorder dead-code warnings.
- `cargo test --locked --bin collide-o-scope video::`: 117 passed, 0 failed,
  10 ignored.
- `cargo test --locked --bin collide-o-scope layers::tests`: 18 passed,
  0 failed, 1 ignored (physical GPU adapter fixture).
- `performance_runtime::tests`: 9 passed, 0 failed, 3 ignored (GPU fixtures).
- `proxy_worker::tests`: 18 passed, 0 failed, 5 ignored (external/GPU fixtures).
- `media_safety::tests`: 10 passed, 0 failed.
- saved-playhead generation fixture: 1 passed, 0 failed.
- Full non-ignored binary suite: 1,887 passed, 0 failed, 145 ignored.

## Performance and stop boundary

The repository contains `tests/fixtures/loop-72f.mp4`, but no genuinely
available prescribed two-source decoded-materialization/upload p99 comparison
harness or pre-change measurement for P4a. The audit's `>=10%` p99 improvement
gate was therefore **not executed**, and this receipt makes no latency claim.

The implementation retains only independently safe changes: immutable
ownership, exact accounting, zero-copy cache/reference transfer, warmed
allocation reuse, bounded asynchronous error draining, and removal of blocking
waits from live per-frame uploads. Synchronous validation remains only at
startup/offline test compatibility seams and GPU resource constructors, not on
the selected-frame or Spout render paths. External long-video, two-source,
FFmpeg-generated, and physical-GPU fixtures remain ignored unless their
explicit environment/hardware prerequisites are supplied.
