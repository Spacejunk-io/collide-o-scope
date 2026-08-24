# P2b core GPU-object retention receipt

Scope: live Compat8/Legacy Exact renderer, built after the audited `v1.6.0`
baseline. This receipt distinguishes proved retention from deliberate stop
boundaries; it does not claim an allocation-count win as a latency result.

## Retained paths

- `EffectPassUniforms`, `CompositeUniforms`, and routed
  `MatteCompositeUniforms` use renderer-owned dynamic-offset arenas. Their
  strides use `min_uniform_buffer_offset_alignment`; total bytes and last
  offsets are checked against `max_buffer_size` and `u32` before any GPU
  object is constructed.
- Arena capacities come only from admitted bounds: 256 composition layers and
  64 materialized image taps. The worst same-submission Exact combination is
  covered explicitly by a unit test. One-slot-over and one-byte-under limit
  fixtures refuse before construction.
- Slots are unique within the frame encoder. The live owner submits that
  encoder before the next `render_evaluated_frame` call. wgpu queue writes and
  submissions share one ordered queue timeline, so reuse on the next frame is
  ordered after every earlier reader; no blocking device poll was added.
- Live resources clone the layer-owned texture-view handle instead of creating
  a view per frame. Source effects bind groups are cached by stable layer ID,
  source backing/view epoch, raster, surface generation, and renderer
  generation. The cache is capped at 256 entries and is pruned when a stable
  source leaves the captured frame.
- The standard Exact stack, conditional-master stack, source image-tap
  materialization, routed matte stack, selective per-layer VHS batch, and
  recorder-tap warmed path therefore construct no buffers, bind groups,
  pipelines, textures, or samplers after their admitted warm-up.
- Composite view pairs and the three internal effects inputs are built once.
  Routed matte groups are bounded by overlay slot and the admitted
  ProgramHistory/accumulated/tap donor identities; tap backing reallocation
  clears the complete routed-matte cache.
- Advanced composition retains its pre-existing `HostUniformArena` and
  `HostAllocationSnapshot` law. Codec-Mosh Send retains its pre-existing
  aligned arenas and source/backing-aware layer bindings. P2b adds those lazy
  Mosh constructions to the same five-domain snapshot without changing its
  pixel or ordering law.
- The Advanced executor now lends each retained Temporal-bypass surface's
  stable layer ID, actual `wgpu::Texture` handle, and view to the late Exact
  overlay. Engine and audience bindings are cached independently by stable ID.
  A hit additionally requires identical source and base backing handles,
  Master-bypass topology, internal Master output slot, surface generation, and
  renderer generation. A topology/raster reprepare necessarily replaces the
  retained Advanced texture and therefore invalidates even when the stable ID
  is unchanged. Removed IDs are pruned. The cache has a hard two-times-256
  bound: at most one entry per admitted layer for each engine/audience base.
- Optional resources stay lazy. Selective-VHS scratch/readback, temporal-dry
  audience candidate, image-routing textures, Mosh Send textures, and the two
  recorder staging buffers are counted at their admission boundary. The first
  Advanced dry-overlay use similarly constructs its retained texture bindings;
  the next unchanged frame has a zero construction delta.

The exact physical arena payload is available through
`Renderer::core_uniform_arena_bytes` and is saturating-added to both Stage
Health's accepted GPU-byte total and the private flight-recorder resource
ledger (including Legacy Exact's otherwise-zero creative-plan payload).
Cumulative buffer/bind-group/pipeline/texture/sampler construction is available through
`Renderer::core_gpu_object_construction_snapshot`; source cache invalidation
reasons are available through `Renderer::core_texture_cache_invalidations`.
These are numeric, path-free resource facts suitable for the private flight
recorder and runtime resource ledger.

## Remaining keep gate

There is no generation-incomplete cache seam in the retained implementation:
the Advanced executor exports its real backing handle, so pointer, positional,
or slice-index identities are not used. This closes the earlier safe-subset
stop for precomposed Temporal bypass.

The performance keep gate is still intentionally open. A representative
warmed command-encoding p95/p99 receipt must show at least the audit's 10%
improvement (or below 0.5 ms), and the full fixed hash/order matrix must remain
green. Zero construction deltas and passing compatibility tests are necessary
evidence, not a latency claim.

## Verification gates

- Pure admission tests: alignment, overflow, one-slot-over, one-byte-under.
- Physical GPU tests: 10,000 warmed arena writes with a zero five-domain
  construction delta; stable source cache reuse and each invalidator.
- Source-contract tests: warmed fixture bodies contain no construction calls,
  frame capture contains no texture-view construction, the Advanced resolver
  carries the actual backing handle, and the bounded overlay cache checks every
  source/base/topology/surface/renderer invalidator before reuse.
- Existing compatibility/GPU fixtures remain authoritative for pixel SHA-256,
  hidden RGB, alpha, image taps, Mosh/VHS ordering, and live/export equality.
- Performance keep gates still require representative warmed p95/p99 receipts;
  zero construction deltas alone are intentionally insufficient.

Executed on the shared Windows workspace after the retained-cache and ledger
integration:

- `cargo fmt --check` and `git diff --check`: pass.
- `cargo check --locked --tests --benches`: pass (pre-existing/unrelated
  warnings only).
- Full non-ignored binary suite: 1,853 passed, 0 failed, 142 ignored.
- P2b admission/cache contract: 2 passed; core arena/cache physical suite: 7
  passed, including 10,000 warmed writes and backing invalidators; widened
  warmed-body construction scan: 1 passed.
- Temporal-bypass ordering/admission suite: 16 passed.
- Compositor reference suite: 5 passed, 3 ignored; the same 3 physical GPU
  fixtures passed explicitly with `--ignored`.
- Codec-Mosh influence physical fixtures: 2 passed explicitly with
  `--ignored`; Advanced hidden-RGB physical fixture: 1 passed explicitly with
  `--ignored`.
- `cargo clippy --locked --bin collide-o-scope`: exits 0; unrelated tranche
  dead-code/style warnings remain visible and were not rewritten by P2b.
