# P1 physical action-to-photon fixture protocol

Status: harness implemented; no physical measurement is claimed by the automated build host.

This fixture is deliberately independent of engine submit timestamps. It measures one optical timeline containing both the input marker and the display. The analyzer will only emit a receipt labelled `physical_action_to_photon`; it has no field in which an engine submission time can be supplied.

## Apparatus and stimulus

1. Prepare a deterministic fixture patch whose unblackout image is uniform full-frame white and whose initial audience state is absolute black. Disable adaptive display processing, record the actual raster, refresh, fullscreen state, and present mode, and warm for at least 300 accepted 30 Hz reference ticks.
2. Use a momentary switch whose electrical edge is split to (a) the controller input under test and (b) an LED in the camera/photodiode field of view. The LED is the authoritative input edge. A software-painted input marker is not acceptable.
3. Aim a high-speed camera or dual-channel photodiode/ADC at both the LED and the active display area. Use at least 10 optical samples per display refresh interval where practical. Record the exact sample interval in nanoseconds.
4. For each trial, begin with LED and display below the fixed optical threshold. Press once to request the known black-to-white full-frame transition. Let both regions settle before the next numbered trial. Capture at least 30 trials per display/profile combination; five independently restarted runs are preferred.

The ordinary control path is under test: browser, phone, native, MIDI, or OSC ingress still receives its engine-minted sequence and follows normal coalescing/queue rules. The fixture does not use a client wall clock and does not bypass the renderer.

## Extraction format

Create one bounded JSON document matching `ActionPhotonFixtureInput` in `src/action_photon.rs`. Samples are ordered by trial and then strictly increasing `elapsed_nanoseconds`. `input_led_q16` and `display_q16` are normalized optical intensities from 0 through 65535. `fixture_digest` is SHA-256 over the immutable extraction configuration and source capture identity; it is not a path.

Run:

```text
cargo run --locked --bin analyze_action_photon -- optical-input.json physical-receipt.json
```

The output is create-new and contains physical p50/p95/p99, minimum/maximum, trial count, sample quantization, and the exact display mode. Missing edges, reversed edges, invalid display facts, reordered timestamps, oversized captures, and an existing output name are refused rather than repaired or overwritten.

## Acceptance and interpretation

- Engine ingress-to-apply and apply-to-submit remain separate P1 domains.
- Queue submission, swapchain presentation, and optical emission are never synonyms.
- A Low Latency profile may be retained only if physical p95 improves by at least 25% on the named target display while temporal 30 Hz reference behavior, hashes, blackout, and stage outputs remain correct.
- Camera exposure/rolling-shutter uncertainty is bounded by the reported sample interval and must be discussed with the receipt. No result is a claim about another monitor, refresh mode, backend, or driver.

The repository receipt `p1-action-to-photon-fixture-unexecuted.json` is the truthful current result for this host: the software harness and refusal rules exist, but the required physical sensor and simultaneously visible input LED were not available.
