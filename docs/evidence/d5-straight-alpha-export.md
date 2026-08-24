# D5 — straight-alpha export keep/stop receipt

Date: 2026-08-24

Status: **named seam, exact PNG/fill-key publishers, and FFV1/RGBA round-trip
retained; application action/live acquisition stopped**.

## Retained implementation

`src/alpha_export.rs` provides the stable
`pre_opaque_straight_alpha_v1` contract, closed plan/receipt schemas,
transactional PNG generation, transactional FFV1 v3/`gbrap` generation, exact
fill/key derivation, hostile fixtures, and effect refusal. `src/lib.rs` exports
the contract and privately reuses the staged publication implementation.

`src/renderer/state.rs` and `src/render_export.rs` name the same slot-0
straight-alpha view at the existing live and offline opaque bind-group
construction points. The offline module also retains an unused
`readback_pre_opaque_straight_alpha_v1` wrapper around its existing bounded
staging buffer and cancellation-aware readback. It adds no readback, pass,
copy, allocation, branch, or synchronization to ordinary MP4.

The output generation remains invisible until all children and the receipt are
synced and a single no-replace directory rename succeeds. The per-file writer
uses the existing `StagedPublication` transaction; the generation uses the
existing `publish_directory_noreplace` and directory barrier. Drop kills and
waits for a live FFmpeg child before deleting its private staging directory.

## Pixel/order receipt

- Source: straight `RGBA8_UNORM_SRGB`; hidden RGB is preserved byte-for-byte.
- Order: creative composition → Temporal/selective overlay → display effects →
  `pre_opaque_straight_alpha_v1` → opaque audience resolve.
- Straight PNG: RGBA bytes unchanged after lossless decode.
- FFV1: packed RGBA8 input → FFV1 v3 planar `gbrap` storage → packed RGBA8
  decode, required byte-exact.
- Fill: pinned sRGB Q0.16 decode, integer `linear * alpha / 255` with rounding,
  pinned threshold encode, opaque alpha.
- Key: source alpha copied to RGB, opaque alpha.
- Codec-Mosh/final-program VHS: refused before staging; no opaque fake key.
- Ordinary MP4: same slot-0 source, same opaque pass, same H.264/yuv420p path.

The eight-pixel hostile fixture covers transparent hidden RGB, a soft key,
cellular zero-coverage gap, partial group matte, low/high transform-edge
coverage, black fill, and opaque identity. Its straight source SHA-256 is
`39757AEAF173C4675FD703D990348E55EE5D8B6C5DDFF3497998C78545564C96`.
All 256 opaque sRGB code values round-trip through the pinned linear table and
inverse thresholds without a code change.

## Exact gates

Commands are run with the repository's Visual Studio x64 environment and
`FFMPEG_DIR` pointing at the admitted FFmpeg 8.1.2 shared distribution.

```text
cargo test --lib d5_ -- --nocapture
cargo test --lib d5_ffv1_gbrap_round_trips_exact_rgba -- --ignored --nocapture
cargo test --bin collide-o-scope opaque_output_flattens_once_and_blit_does_not_repeat_it
cargo test --bin collide-o-scope gpu_flattens_straight_alpha_in_linear_light_and_supports_raw_egui_view -- --ignored --nocapture
cargo test --bin collide-o-scope gpu_two_layer_live_and_export_full_stack_matches_fixed_golden_at_24_30_and_60_fps -- --ignored --nocapture
cargo clippy --lib -- -D warnings
cargo check --bin collide-o-scope
cargo check --tests
rustfmt --edition 2021 --check src/alpha_export.rs src/lib.rs src/photosensitivity_advisor.rs src/render_export.rs src/renderer/state.rs
```

Results at the retained-code gate:

- Default D5 library suite: 6 passed, 0 failed, 1 physical FFmpeg test ignored.
- Physical FFV1 encode/decode: 1 passed, 0 failed. Both hostile frames decoded
  to the exact concatenated RGBA bytes submitted to the encoder.
- Opaque shader/source boundary: 1 passed, 0 failed.
- Physical straight-alpha-to-opaque GPU resolve: 1 passed, 0 failed.
- Physical 24/30/60 Hz live/offline full-stack fixed golden: 1 passed, 0
  failed; fixed SHA-256 remained exact at all three rates.
- Strict library Clippy: exit 0 with `-D warnings`.
- Production binary check: exit 0. It emitted one unrelated warning for the
  concurrently retained `handle_web_action_inner_with_feedback` helper.
- All test targets compile: exit 0. It emitted one unrelated warning for the
  pre-existing `FlightRecorder::start` helper.
- Focused Rustfmt check: exit 0.

The default suite proves early Codec-Mosh/VHS refusal; exact hidden-RGB PNG
round-trip; soft/cellular/partial/transform/black fixtures; pinned sRGB opaque
identity; fill/key atomic visibility; receipt round-trip and fixed source hash;
strict frame order; and cancellation cleanup.

## Mandatory stop fired

No application action/configuration selects these artifacts, no live recorder
owns an alpha-capable encoder, and no new offline frame loop invokes the named
readback wrapper. Shipping those integrations without the action-schema owner,
a bounded multi-flight readback/performance receipt, and end-to-end
cancellation/progress tests would overstate the evidence and risk changing the
ordinary path.

Accordingly the independently safe seam, publishers, receipts, effect refusal,
and exact fixtures are retained. UI/action/live acquisition remains deferred.
No alpha capability should be advertised as available until that missing
end-to-end integration and P1 performance gate pass.
