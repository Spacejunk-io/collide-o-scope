# P4c 10-bit/HDR output surface — evidence-backed ruling

Date prepared: 2026-08-27

Read-only audit basis:
`e81bddfd6ee823b6247b08f24f5c60cb5de2a011`

Status: **DEFERRED — API + physical-output gate**

This is the explicit output-surface ruling required by the perfection
handover. It is distinct from the measured rejection of Full-16 temporal
history. The current product remains SDR8. A 10-bit texture format by itself
does not constitute HDR, and no current output is to be labeled HDR.

The exact present decision is:

- **Reject promotion of a format-only 10-bit SDR swapchain now.** The current
  Program image has already passed through an 8-bit sRGB endpoint, so changing
  only the swapchain to `Rgb10a2Unorm` would add codes without preserving
  source precision or establishing an artistic gain.
- **Defer genuine HDR output, rather than rejecting it permanently.** Reopen
  only after an API can select and report the exact format/color-space pair,
  an admitted high-precision path applies an explicit transfer/gamut/tone-map
  policy, and named physical displays close the backend gate.
- **Keep `Rgba8UnormSrgb`/SDR as the compatibility default.** Capability loss,
  absent old fields, unsupported displays, and ambiguous metadata must retain
  or return to that truthful state without rewriting an authored patch.

## Authority and P4c boundary

`COLLIDE_O_SCOPE_IMPROVEMENT_AUDIT.md:276-299` is the item-13 authority. It
requires 10-bit/HDR precision to survive through an admitted working format,
requires an explicit output tone-map/gamut policy, forbids calling an 8-bit
sRGB surface HDR, and keeps HDR output as a separate evidence-gated tranche.

`COLLIDE_O_SCOPE_PERFECTION_HANDOVER.md:316-329` keeps P010 production
admission behind that fidelity argument. `:331-341` names the explicit 10-bit
output-surface ruling as the remaining unconsidered decision and distinguishes
it from Full-16 history. The mandatory comparable workload, measurement, and
run law is `COLLIDE_O_SCOPE_STANDING_DIRECTIVES_LEDGER.md:379-406`.

This ruling is **not a blocker** for the integrated, authored-opt-in
`Yuv420p8` metadata-managed P4c path or for its separate default-flip
performance gate. It **is a blocker** for production P010/HDR admission and
for every 10-bit-fidelity or HDR-output claim. Ordinary SDR P010 can be
evaluated separately, but it cannot claim end-to-end 10-bit fidelity while
the Program, readback, and export endpoints remain Compat8.

## Current precision facts

### Live renderer and window surfaces

- `src/renderer/state.rs:1276-1282` fixes the internal composite format at
  `Rgba8UnormSrgb` and exposes only its raw 8-bit twin for egui.
- `src/renderer/state.rs:5676-5693`, `:5765-5782`, and `:6365-6392` select the
  first advertised sRGB surface format for the main and audience windows.
- `src/renderer/state.rs:5933-5971` creates all three Program composite
  textures in the fixed 8-bit format.
- `src/renderer/state.rs:6403-6486` and `src/shaders/blit.wgsl:1-10` perform a
  plain sample into the audience surface. They contain no transfer, gamut,
  tone-map, peak-luminance, or HDR-signal policy.
- `src/shaders/opaque_output.wgsl:1-15` defines the final opaque image as an
  sRGB source/target consumed by preview, projector, Spout, NTSC, and MP4.

### StageMap physical output

- `src/renderer/stage_map.rs:25-30` fixes every StageMap endpoint texture at
  `Rgba8UnormSrgb`, four bytes per pixel.
- `src/shaders/stage_map.wgsl:141-149` clamps calibrated color to `[0, 1]`;
  `:213-215` then plain-samples that already-8-bit endpoint to the physical
  surface.
- `src/main.rs:381-387` and `:9822-9851` select and configure StageMap monitor
  surfaces sRGB-first, without a color-space or HDR-metadata choice.

### Advanced composition is not HDR presentation

- `src/renderer/composition_host.rs:29-30` uses `Rgba16Float` for Advanced
  working surfaces but fixes presentation at `Rgba8UnormSrgb`.
- `src/renderer/composition_host.rs:1216-1225` builds that fixed-format present
  pipeline, and `:2378-2392` rejects every other target format. The wrapper
  repeats the refusal at `src/renderer/composition.rs:1471-1492`.
- `src/shaders/composition_host.wgsl:113-134` states that its present entry is
  valid only for `Rgba8UnormSrgb`, clamps linear RGB to `[0, 1]`, applies sRGB,
  and dithers between adjacent 8-bit codes.
- `docs/precision-and-scale.md:5-12` names the settled mode
  `AdvancedWorking16HistoryCompat8`; `:88-103` proves a working-precision gain
  and deterministic Compat8 presentation, not HDR output.

### P4c source delivery

- `src/video/planar.rs:46-69` can represent `Yuv420p8`, `Nv12`, and `P010Le`,
  and the GPU converter has bounded P010 fixtures.
- Production vocabulary remains only packed RGBA8 and planar YUV420P8 at
  `src/video/payload.rs:20-30`; production admission is YUV420P8-only at
  `src/video/decoder.rs:1381-1437`.
- `src/video/planar.rs:467-474` says the current CPU oracle performs no
  transfer linearization, gamut conversion, or tone map. PQ and HLG return
  typed `HdrToneMapRequired` at `:691-699`, `:743-754`, and `:927-938`.
- `docs/evidence/p4c-planar-integration-note.md:109-123` therefore keeps
  NV12/P010 unadmitted and the HDR/tone-map question separate.

### Recording and export

- `src/render_export.rs:3040-3066` and `:6629-6689` create export source and
  Program composite textures as `Rgba8UnormSrgb`.
- `src/render_export.rs:6774-6781` and `src/renderer/readback.rs:13-19` use
  four-byte RGBA8 readback.
- `src/render_export.rs:3696-3769` sends raw `rgba` to libx264 `yuv420p`;
  `src/program_recorder.rs:1091-1130` does the same for live recording.

Changing only FFmpeg's output `-pix_fmt` would therefore pad or transform
8-bit input; it would not preserve ten source bits or prove an HDR artifact.

## Candidate APIs and formats

The repository pins `wgpu = "29"` at `Cargo.toml:12` and `wgpu-types 29.0.4`
at `Cargo.lock:5358-5361`. That version contains renderable/filterable
`Rgb10a2Unorm` and `Rgba16Float`, but its `SurfaceCapabilities` reports only
formats, present modes, alpha modes, and usages; `SurfaceConfiguration` has no
color-space field. It guarantees only BGRA8 and BGRA8-sRGB surface formats.
See the pinned crate source at
`wgpu-types-29.0.4/src/surface.rs:134-177,231-240` and
`wgpu-types-29.0.4/src/texture/format.rs:199-230,971-987,1086-1100`.

The already-planned wgpu-30/egui-0.36 maintenance campaign is the first
portable reopening candidate. wgpu 30 adds `SurfaceConfiguration.color_space`,
per-format color-space capabilities, and explicit `SurfaceColorSpace` values
including `ExtendedSrgbLinear`, `Bt2100Pq`, and `Bt2100Hlg`. The application,
not the format name, remains responsible for writing values in the selected
encoding. Primary API references:

- [wgpu 30 `SurfaceConfiguration`](https://docs.rs/wgpu/30.0.1/wgpu/type.SurfaceConfiguration.html)
- [wgpu 30 `SurfaceColorSpace`](https://docs.rs/wgpu/30.0.1/wgpu/enum.SurfaceColorSpace.html)

Candidate output pairs are evidence subjects, not selected production modes:

| Candidate | Intended signal | Present ruling |
| --- | --- | --- |
| `Rgb10a2Unorm` + `Bt2100Pq` | Encoded BT.2100/HDR10 PQ | Deferred; query the exact pair after wgpu 30 and prove a PQ output transform plus physical sink. |
| `Rgb10a2Unorm` + `Bt2100Hlg` | Encoded BT.2100 HLG | Deferred; same pair, transform, and sink gate. |
| `Rgba16Float` + `ExtendedSrgbLinear` | Linear extended-range scRGB/EDR | Deferred; prove luminance mapping, headroom adaptation, and backend/display behavior. |
| `Rgb10a2Unorm` + SDR/sRGB semantics | Ten-bit SDR | Evaluation only; retain only if end-to-end retained precision demonstrates a named gain. Never call it HDR. |

No native-backend escape hatch is authorized before the portable campaign is
assessed. A backend-specific path would need its own bounded lifecycle,
surface-loss, device-loss, hotplug, and interoperability evidence.

## Platform and physical blockers

Format availability is not display capability or signal truth.

- On Windows, Microsoft's Advanced Color guidance says an HDR10
  `R10G10B10A2_UNORM` swapchain is sRGB by default and must explicitly select
  `DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020`; display color volume and
  luminance can change and must be re-queried/tone-mapped. See
  [Use DirectX with Advanced Color](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/high-dynamic-range)
  and [`IDXGIOutput6::GetDesc1`](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_6/nf-dxgi1_6-idxgioutput6-getdesc1).
- On Metal, EDR requires explicit extended-range opt-in, a compatible extended
  color space, an accommodating format such as `RGBA16Float`, and current
  display headroom/metadata behavior. See Apple's
  [system tone-mapping guidance](https://developer.apple.com/documentation/metal/using-system-tone-mapping-on-video-content)
  and [`CAMetalLayer.edrMetadata`](https://developer.apple.com/documentation/quartzcore/cametallayer/edrmetadata).
- On Vulkan, the implementation must advertise the exact format/color-space
  pair. PQ and HLG have distinct `VkColorSpaceKHR` values, and HDR metadata is
  a separate extension rather than a consequence of a 10-bit image. See
  [`VkColorSpaceKHR`](https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VkColorSpaceKHR.html)
  and [`vkSetHdrMetadataEXT`](https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/vkSetHdrMetadataEXT.html).

Hosted CI cannot close these seats. Promotion requires a named HDR-capable
monitor/sink, OS HDR/EDR state, adapter, backend, driver, format/color-space
pair, present mode, raster, and an independent signal/metadata observation.
A screenshot of a window or a successful surface configuration is not proof
of ten-bit or HDR output.

## Deterministic implementation required before reopening

1. Add an append-only output-signal contract with an SDR8 compatibility
   default and explicit evaluation/gated states for SDR10, HDR10 PQ, and HLG.
   Absent old fields select SDR8. Do not infer HDR from source resolution,
   source bit depth, texture format, monitor name, or OS mode.
2. Add one pure selector over requested policy and an exact advertised
   `(TextureFormat, SurfaceColorSpace)` pair. Its result includes a typed
   refusal/fallback reason. Main, audience, and StageMap surfaces consume the
   same oracle; no path silently chooses a different signal.
3. Publish requested and active policy, working format, surface format, color
   space/EOTF, source descriptor, tone-map/gamut law, fallback reason,
   adapter/backend/driver, display identity, and present mode in snapshots and
   receipts.
4. For HDR source admission, convert P010 integer planes into an admitted
   linear high-precision working format without an RGBA8 intermediate. Define
   independently testable transfer, gamut, reference-white, peak-luminance,
   and tone-map behavior. SDR presentation maps explicitly to BT.709/sRGB;
   HDR presentation encodes exactly the selected PQ, HLG, or extended-linear
   signal. Output remains opaque.
5. Build evaluation-only offscreen pipelines for `Rgb10a2Unorm` and
   `Rgba16Float`. Prove that adjacent admitted ten-bit source codes remain
   distinguishable until the declared final transform. Do not change a
   production surface selector or default during this stage.
6. Reconcile resource accounting. `Rgb10a2Unorm` remains four bytes per pixel;
   `Rgba16Float` is eight. The candidate must fit current CPU/GPU/media caps;
   the tranche does not authorize a cap increase.
7. Treat ten-bit/HDR export as a separate matching-policy tranche. It needs a
   wider or direct-planar readback/input contract, a proven available and
   license-compatible ten-bit encoder, explicit range/matrix/primaries/
   transfer metadata, decode-back code/hash evidence, and live/export parity.
   A `-pix_fmt`-only edit is rejected.

## Deterministic tests

Extend the existing surface-selection tests at
`src/renderer/state.rs:2445-2475` and add at least these contracts:

- `output_signal_default_is_sdr8_and_absent_state_is_compatible`
- `output_signal_selector_never_infers_hdr_from_texture_format`
- `output_signal_selector_requires_exact_format_color_space_pair`
- `hdr_capability_loss_returns_typed_sdr_fallback_without_patch_mutation`
- `all_physical_output_seats_resolve_one_signal_contract`
- `pq_hlg_and_srgb_cpu_oracles_match_independent_known_value_vectors`
- `p010_adjacent_codes_survive_the_high_precision_working_path`
- `rgb10a2_and_rgba16float_gpu_outputs_match_the_cpu_oracle`
- `legacy_sdr_live_and_export_hashes_remain_exact`
- `ten_bit_export_round_trips_codes_and_declared_color_metadata`

The CPU corpus must cover finite/NaN/Inf handling, black/reference white/peak,
limit±1 code values, monotonic 10-bit ramps, saturated BT.709 and BT.2020
colors, out-of-gamut values, tone-map knee and clipping, alpha/opaque law, and
repeat determinism. GPU comparisons declare code- and float-space tolerances
before running.

## Physical reopening gate

1. Complete the wgpu-30/egui-0.36 campaign and re-prove the vendored
   acquire-timeout patch before coupling HDR work to it.
2. On every proposed admitted backend, capture the exact advertised
   format/color-space pairs and prove truthful refusal on unsupported pairs.
3. Exercise main, audience, and StageMap surfaces through create, resize,
   fullscreen, loss/recreate, suspend/resume, monitor move, hotplug, and
   OS-HDR-mode changes. A mode loss returns safely to explicitly labeled SDR.
4. Use an actual HDR sink and independent signal/metadata inspection to prove
   active bit depth, primaries, transfer, range, and luminance behavior for PQ
   or HLG. Repeat on SDR-only and mixed-display arrangements.
5. Run the standing G01 matrix: 1/3/8 layers at 720p/1080p, the 10-bit
   BT.2020/PQ ramp, Advanced/temporal/effects, recorder/export, Output and
   StageMap, blackout, and controller traffic. Preserve pixel/hash truth and
   live/export parity.
6. Warm at least 300 accepted ticks, run performance candidates for ten
   minutes, compare at least five runs, record p50/p95/p99 plus resource
   ledgers, and change only the output-signal variable. Any failed keep gate
   closes the candidate.

## Closing fields

- Disposition: **DEFERRED — API + PHYSICAL-OUTPUT GATE**
- Format-only 10-bit production promotion: **REJECTED AT CURRENT TREE**
- Current SDR8 default: **RETAINED**
- P4c YUV420P8 integration: **NOT BLOCKED**
- P010/HDR production admission: **BLOCKED BY ITEM 13**
- Topic evidence commit: **PASS** —
  `bcdb99cf93d76cad1536273994f4714e3f84e316`
- Integration commit on `feat/web-control-panel`: **PENDING**
- Exact-commit CI: **PENDING**
- Deterministic implementation/tests: **NOT YET RUN**
- Physical HDR/backend/display matrix: **NOT RUN**
- Production renderer/export change: **NOT ATTEMPTED**

## Deliberate non-claims

This note does not claim that `Rgb10a2Unorm` is available on any current
surface, that ten stored bits imply ten effective bits, or that a successful
swapchain configuration emits HDR. It does not claim Advanced working RGBA16F
is an HDR Program endpoint, that ordered dithering restores source bit depth,
or that the existing P010 oracle performs transfer/gamut/tone mapping. It does
not claim hosted CI can verify a physical signal, authorize a native API
bypass, select an encoder, change a patch default, raise a resource cap, admit
P010/PQ/HLG in production, or label any current output HDR.
