<div class="mw6 center tl mb4">

### procedural video generation

the offline render pipeline is deterministic from frame-indexed inputs, and the
live/offline paths share the same core shaders and composition helpers. live
output is still influenced by redraw cadence, asynchronous workers, and hardware
controllers, so equality is claimed only for controlled offline replay. that
makes procedural work primarily a question of composition, not a second renderer.

the original proposal was directionally right, but several details needed to
become more precise before they were safe to build. “randomly change every
number,” “a markov chain,” and “process audio sample by sample” are sketches,
not specifications. this revision records what is implemented, what changed,
and what still needs evidence.

---

### decisions

| idea | decision | reason |
|---|---|---|
| one-click patch collection | implemented | a bounded background writer keeps disk work out of the render loop |
| brownian mutation of patches | implemented with correction | a typed, reflected, mean-reverting walk stays within declared generator bounds and avoids boundary pile-up |
| voronoi motion | implemented as a bounded cellular effect | one shared shader path gives live/export parity without creating a new source lifecycle |
| clip luminance/color/motion statistics | deferred | motion statistics require bounded multi-frame decoding and a persistent cache; they are not “almost free” |
| second-order markov names from two-word phrases | replaced | two-word examples cannot train a genuine second-order model; a weighted finite grammar is honest and deterministic |
| visual-parameter-driven audio DSP | research gate | the current exporter promises explicit 1x media-audio muxing; pitch shift, convolution, dynamics, smoothing, and loudness need a real DSP contract |
| automatic batch video rendering | deferred one stage | version one generates reviewable patches and manifests without monopolizing the live GPU or hiding expensive failures |

---

### patch collection

the web panel now has **Capture patch**. it snapshots the complete current
`PatchState` and writes a new YAML file below `patches/`.

capture uses one worker and a capacity-one queue. submission is non-blocking;
the render thread never waits for serialization or storage. `Queued` means the
worker accepted the request; only `Saved …` means the file has been flushed and
committed. shutdown gives accepted work a bounded grace interval, then refuses
to let a stalled filesystem hang the process indefinitely. the writer creates a
unique temporary file, flushes it, and performs an operating-system no-replace
rename. parent-directory synchronization is attempted where the platform permits
it; this is not a universal power-loss guarantee. existing files are never
overwritten. the panel reports authoritative
`Saving…`, `Saved …`, or `Error: …` state from the engine rather than inventing
success in JavaScript.

this produces the anchor corpus: configurations selected by a performer because
they already have a useful visual logic.

---

### the stochastic walk through patch space

a patch is not a vector of interchangeable floats. it contains continuous,
circular, logarithmic, quantized, categorical, boolean, and structural values.
source identity, layer order, visibility, pause state, and routing topology are
composition—not noise—and are preserved by version one.

for a bounded continuous parameter, the generator uses an OU-inspired reflected
AR(1) step around the anchor:

\[
x_{k+1} = R_{[l,h]}\left(a + \rho(x_k-a) + T\sigma\xi_k\right),
\quad \rho = 0.85,\quad \xi_k \sim U[-1,1]
\]

where `a` is the anchor value, `T` is temperature, and `R` reflects at the legal
walls. reflection matters: clamping a random walk creates an artificial heap of
samples at zero and one. hue, slit angle, and LFO phase wrap on a circle instead.
speed, downsample fraction, pixel size, and cellular scale move in log space.
pixel/posterize/FPS values are quantized after movement. blend algorithms and
booleans change through rare transitions, not float arithmetic.

the walk is sequential and mean-reverting: siblings have local continuity but
do not drift without bound. temperature is finite and constrained to 0–2. at
temperature zero, output is a canonical anchor: defaults are resolved and
runtime-supported numeric, enum, temporal, modulation, VHS, and morph values are
sanitized while source identity and topology remain unchanged. live Spout inputs are rejected
unless the caller explicitly accepts their documented deterministic-black offline
policy. active two-slot A/B morphs and in-flight glides are rejected because there
is no unique anchor until the performer settles or clears them. a single inert
stored slot is permitted.

the pseudo-random generator is a locally implemented SplitMix64 with published
golden vectors. visual domains, layers, temporal state, modulation, VHS state,
and naming receive separate streams. changing title vocabulary therefore cannot
change a patch.

---

### command-line generation

version one is deliberately patch-only:

```powershell
target\release\collide-o-scope.exe generate `
  --anchor patches\anchor.yaml `
  --output generated `
  --count 10 `
  --temperature 0.5 `
  --seed 424242
```

`--count` is bounded to 1–256. every piece is committed to its own new directory:

```text
generated/
  0001-grid-lattice-00067932/
    patch.yaml
    manifest.json
```

output is no-overwrite. each directory is assembled under a sibling temporary
name, its files are flushed, and the directory is committed with an
operating-system no-replace rename. all names and serializations are preflighted
before the first commit. the transaction boundary is one piece—not the whole
invocation—so an exceptional later storage failure does not erase already
committed work; the returned error names those paths explicitly. rerunning the
same command into a clean directory yields byte-identical YAML and manifest files.

the manifest records a schema version, generator algorithm version, seed, index,
temperature, deterministic title/slug, canonical-anchor FNV-1a digest, lineage,
logical source names, and warnings. it deliberately excludes local absolute media
paths.

this manifest proves patch-generation replay, not universal pixel identity. a
future artifact manifest must additionally record media content digests, export
settings, application/build identity, FFmpeg identity, and relevant GPU/backend
facts. “same library” is too vague when two different files can share a name.

---

### cellular motion: the useful intersection of voronoi and stochastic movement

the most useful visual result from the Brownian/Voronoi idea is a new **CELLULAR**
effect with seven controls:

- **Amount** — overall domain-warp/ridge mix, 0–1; zero bypasses all cellular work.
- **Scale** — cells across frame height, 2–32.
- **Warp** — bounded displacement within each cell, 0–1.
- **Drift** — feature-target transitions per second, 0–2.
- **Gap Key** — layer/master removal strength for the cellular boundary, 0–1.
- **Gap Threshold** — layer/master ridge strength at which the gap opens, 0–1.
- **Gap Softness** — layer/master feather width around that threshold, 0–0.5.

the shader evaluates a fixed 3×3 neighborhood of a jittered grid. feature points
remain in the central part of their cells, so that neighborhood is sufficient.
an integer avalanche hash produces each epoch’s targets. cubic smoothstep
interpolates between consecutive targets; there is no independent frame jitter.
coordinates are aspect-correct, UV displacement is capped, and `F2-F1` supplies
a restrained ridge at cellular boundaries. the disabled branch skips every hash
and distance operation. the gap controls convert that ridge into straight-alpha
coverage. layer gaps expose the existing stack below. the master exposes the
same three controls: an ordinary post-stack master gap resolves over black; in
the selective master-bypass path, direct master cellular and VHS run only on
inherited slices, so inherited gaps can reveal lower or bypassed content before
the stack is recomposited. temporal processing remains program-wide, and the
final program is composited over black once for opaque preview, Spout, and MP4.

this is bounded interpolation between independently hashed stochastic feature
targets, not literal Brownian motion or a random walk. that deviation is intentional:

- unbounded Brownian displacement eventually escapes its useful cell and makes a
  fixed neighborhood incorrect;
- independent random displacement per frame flickers and changes with frame rate;
- a multi-octave fBm plus a larger neighbor search would multiply UHD cost across
  every layer;
- smooth bounded targets give temporal continuity, deterministic replay, and a
  fixed computational budget.

the effect is present on master and every layer, persists in YAML, interpolates
through morphs, and is a modulation target. live effect time is reset when a patch
generation changes, aligning the live clock origin near offline export's `t=0`;
the first presented live frame necessarily has a small positive elapsed time.

at render time, morph materializes first at the sampled beat and one immutable
modulation result then supplies the master and every layer's offsets. morph
captures are layer-topology-revision guarded; removal/reorder remaps stored
slots, a newly appended layer stays outside existing slots, and hue/slit angles
use their shortest wrapped arcs. offline replay uses the same frame-indexed
ordering. selective VHS is asynchronous live and synchronous in export, so this
is a semantic/order guarantee rather than a claim of bit-identical live pixels.

the graphics basis is inspired by Steven Worley’s cellular texture function,
which defines fields from distances to scattered feature points
([SIGGRAPH 1996](https://doi.org/10.1145/237170.237267)). the mathematical theorem
about random walks on increasingly dense Poisson–Voronoi cells converging, up to
a time change, to Brownian motion applies to a particular random surface and scaling limit
([Gwynne, Miller, and Sheffield](https://arxiv.org/abs/1809.02091)); it is not a
description of this shader. likewise, self-propelled Voronoi tissue models include
motility and multibody mechanics
([Bi et al., Physical Review X](https://doi.org/10.1103/PhysRevX.6.021011)). moving
rendering seeds alone does not simulate tissue fluidity, energy landscapes, or
phase transitions. those are scientific inspirations, not product claims.

---

### titles

the title generator uses a small weighted, tagged grammar. analog-heavy patches
draw preferentially from tracking, phosphor, raster, chroma, drift, ghost, decay,
snow, and smear. digital/cellular patches draw from pixel, grid, threshold,
cellular, field, lattice, phase, and sweep.

this produces two-token names such as `grid lattice` and `tracking decay` from a
dedicated random stream. it is finite, inspectable, testable, and better matched
to the small corpus than pretending that a second-order chain was trained on
two-word examples. a real second-order chain would require trigram observations
and enough source text to estimate them.

---

### why clip statistics wait

average luminance and coarse color can be computed cheaply from frames that are
already decoded. motion energy cannot: it requires at least two time-separated
samples per clip, a sampling policy, decode limits, error handling, and a cache
keyed by media content—not filename. doing that ad hoc during every library scan
would contend with thumbnail generation and make large folders unpredictable.

the next acceptable design is a bounded analysis worker with:

1. a fixed number of uniformly spaced samples;
2. resolution-reduced luma/color/motion measures;
3. media-content or size/mtime cache invalidation;
4. cancellation and decode timeouts;
5. explicit unknown/error states;
6. measured scan-time and memory budgets.

until then, version one preserves the anchor's source assignments rather than
pretending it can make informed random pairings.

---

### why sonification waits

the visual/audio table remains an excellent research prompt, but each row needs
an actual signal-processing definition. hue-to-pitch is not “math on floats” once
duration and timbre must be preserved. luma-smear-to-convolution needs an impulse
response and latency policy. contrast-to-compression needs threshold, ratio,
attack, release, makeup gain, and channel-link behavior. frame-rate parameters
must become sample-rate envelopes without zipper noise.

an acceptable DSP experiment must be opt-in and separate from the current audio
mux contract. it needs:

- block/sample-rate parameter smoothing;
- stable resampling and pitch processing;
- deterministic noise seeds;
- channel and sample-rate conversion rules;
- true-peak and integrated-loudness limits;
- golden impulse/sine/noise tests;
- listening tests for clicks, pumping, aliasing, and phase errors;
- an explicit metadata record of the DSP graph and version.

until those exist, selected media audio continues to start at time zero at 1×,
pad when short, trim when long, and remain independent of visual speed/pause.

---

### next build order

1. **implemented:** bounded atomic patch capture.
2. **implemented:** deterministic typed generator, weighted titles, and manifests.
3. **implemented:** bounded cellular/Worley effect across live and offline paths.
4. **next:** shared, content-aware source resolution and stronger artifact hashes.
5. **then:** cancellable sequential batch rendering with explicit GPU/time/disk budgets.
6. **then:** bounded clip statistics and a curation view, after profiling.
7. **research branch:** opt-in audio DSP, only after the signal contract and tests.

the closed-system ambition survives, but in layers. a generated patch can already
be replayed, inspected, edited, and rendered. titles and lineage are deterministic.
the shader adds a new organic cellular vocabulary without compromising fixed-cost
rendering. audio becomes part of the same generative system only when its behavior
is as explicit and verifiable as the image path.

</div>
