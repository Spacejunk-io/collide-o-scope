# D2 — photosensitivity-risk advisor evaluation prototype

Date: 2026-08-24

Status: **prototype retained; live capability deferred**.

This is an operator-advisory research instrument, not a medical or regulatory
safety system. It cannot change, attenuate, replace, veto, record, or publish
program pixels. The application and renderer do not construct it. Its only
availability value is `deferred_p1_and_review` until the separately required
accessibility/legal review and P1 p95/p99 campaign both exist.

## Deterministic measurement law

`PhotosensitivityCpuReference` is the canonical integer reference. For every
accepted raster it visits a fixed 64×36 lattice. Each of the 2,304 cells takes
a fixed 4×4 set of integer-coordinate samples, for exactly 36,864 source loads
regardless of output width or height. RGB code values are converted from sRGB
to linear-light Q0.16 through a pinned complete 256-entry table (no runtime
`powf`), averaged with integer rounding, and reduced to Rec.709 luma using
weights `13933 + 46871 + 4732 = 65536`.

Per-cell history retains only Q0.16 RGB/luma, the prior qualifying luma
direction, and initialization state. A frame produces these eight bounded
`u32` aggregates and nothing else:

- sampled and initialized cell counts;
- affected, reversing, and red-transition cell counts;
- luma- and maximum-channel-delta sums;
- a reserved zero word that makes schema drift fail closed.

The classifier owns a 120-observation maximum `VecDeque`. It measures event
count, reversing-event count, red-event count, and the longest uninterrupted
run in accepted 30 Hz reference ticks; a missing tick breaks the sustained run.
Sequences and ticks must be strictly increasing. Malformed subset relations,
stale observations, invalid policy bounds, incomplete rasters, and nonzero
reserved data are rejected without changing the last accepted observation.

There is deliberately no default venue policy. Construction requires explicit
numeric thresholds, validates them, and returns the exact policy and algorithm
version in telemetry. The synthetic thresholds in tests are fixture values,
not standards or venue advice.

The raster input is a borrowed immutable slice. Neither the CPU reference nor
the GPU contract owns an API capable of writing a source texture. Telemetry is
a closed aggregate schema containing numeric thresholds/counters and typed
enums; it has no field for pixels, authored text, source names, paths, media,
tokens, or arbitrary strings.

## Isolated GPU contract

`PhotosensitivityAdvisorGpu::new_evaluation_only` accepts a fixed read-only
sRGB RGBA8/BGRA8 texture and creates its own default view after querying the
actual backing format. There is intentionally no rebind method: backing or view
generation changes require stage recreation and therefore fresh history.

One compute submission dispatches 36 workgroups of 64 invocations. Each cell
executes the same 16 `textureLoad` operations as the CPU reference. The source
has no storage binding and the WGSL contains no `textureStore` or sampler. The
only GPU-to-CPU copy is the 32-byte aggregate counter struct.

The retained allocation is bounded and raster-independent:

| Resource | Count | Bytes |
|---|---:|---:|
| Cell-history storage | 1 | 73,728 |
| Atomic aggregate storage | 1 | 32 |
| Immutable reduction policy | 1 | 16 |
| Async map/readback slots | 3 | 96 |
| **Total buffers** | **6** | **73,872** |

The three readback slots are strict FIFO by submission sequence. Saturation
drops a new sample instead of allocating or queueing. History reset is refused
until every submitted slot has been mapped and harvested; queue ordering then
places the clear before the next reduction. Numeric runtime counters reconcile
scheduled/dropped/completed samples, workgroups, planned texture loads,
submitted/mapped bytes, map failures, malformed aggregates, and reset outcomes.

## Hostile fixtures and exact results

Commands used on the audit host:

```text
cargo check --tests
cargo test --lib d2_
cargo test --lib photosensitivity_gpu::tests::d2_gpu_flat_hostile_fixtures_match_cpu_and_pool_is_bounded -- --ignored --exact --nocapture
cargo clippy --lib -- -D warnings
```

Results:

- `cargo check --tests`: exit 0. Only unrelated warning-level findings were
  emitted at that point in the shared worktree.
- `cargo test --lib d2_`: 9 passed, 0 failed, 1 adapter test ignored by default.
- Physical GPU opt-in test: 1 passed, 0 failed. WGSL creation and dispatch
  succeeded; black/white reversal and saturated-red aggregate counters matched
  the CPU reference exactly. A mid-code RGB fixture retained exact categorical
  counts while magnitude sums stayed inside the declared two-Q0.16-units per
  initialized-cell hardware-sRGB decode bound. FIFO order, three-slot
  saturation/drop, busy reset refusal, successful idle reset, and byte/work
  ledgers reconciled.
- Strict library Clippy: exit 0 with `-D warnings`.

The broader `cargo clippy --all-targets --all-features -- -D warnings` command
was also executed. It reported no D2 finding, but remained red on pre-existing
or concurrently owned findings in `analyze_action_photon`, `flight_recorder`,
`layers`, `performance_runtime`, `renderer/gpu_timing`, `video/planar`,
`video/source_descriptor`, and `web/state`; those regions were outside this
bounded tranche and were not modified here.

The default suite covers deterministic repeat analysis without input mutation;
static and long frozen frames; slow fade; small-area flash; full-field
alternation; saturated-red transitions; blackout then held black; irregular and
stale cadence; malformed/hostile aggregate relations; policy/raster refusal;
ring capacity; sustained-run breaking; compact privacy-safe serialization; and
the shader's read-only-source/aggregate-only contract.

## Mandatory stop gate

The physical test proves shader validity, CPU/GPU agreement for exact hostile
flat fixtures, bounded mapping, and constant declared work. It is not a P1
performance receipt: it does not compare advisor-off versus advisor-on GPU and
readback p50/p95/p99 on the fixed production fixture, and it does not prove the
audit's instrumentation budget on an admitted adapter/backend/driver/raster.

Therefore the P1 GPU/performance gate is unexecuted. The prototype remains
library-only, unavailable to the live UI, absent from production telemetry and
the flight recorder, and incapable of affecting Program, blackout, recording,
StageMap/output, or export. Promotion must not occur until the missing paired
P1 receipt and independent review both pass.
