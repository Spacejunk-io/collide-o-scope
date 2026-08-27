# D5 — the straight-alpha application action: evidence note

Date: 2026-08-26. The follow-on tranche the keep/stop receipt
(`docs/evidence/d5-straight-alpha-export.md`) named: the application action,
the invoked offline readback at the frozen `pre_opaque_straight_alpha_v1`
seam, and this receipt. The retained core — contract, transactional
publishers, derive laws, hostile fixtures, effect refusal — is unchanged;
this tranche wires it to the operator.

## The action

`start_export` gains one additive, defaulted `alpha` field carrying the
closed `AlphaArtifactKind` vocabulary (`straight_png_sequence`,
`fill_key_png_sequence`, `straight_png_and_fill_key`, `ffv1_rgba`). An
omitted or null field is the exact prior path — no readback, pass, copy,
branch, or allocation is added to ordinary MP4 — and an unknown token
rejects the whole action at deserialization. The panel's export group gains
an "Alpha plates" select that sends the token only when chosen.
`ExportConfig.alpha` carries it into the job.

## The offline loop

- **Refusal before anything exists.** The complete `AlphaExportPlan` —
  bounds plus the authored effect state read from the saved patch
  (`ntsc.enabled`, `temporal.mosh.sanitized().is_active()`) — validates
  before any device, texture, or directory is created.
- **One sequential in-flight readback.** When armed, each accepted frame
  reads the named seam through `readback_pre_opaque_straight_alpha_v1` —
  the retained wrapper, now invoked — reusing the existing bounded staging
  buffer sequentially after the audience readback, under the established
  cancellation law. No new staging allocation exists.
- **The per-frame effect guard.** Morph endpoints and modulation can wake
  Codec-Mosh or final-program VHS after the authored refusal read the saved
  state, so the frame loop re-checks both laws (`mosh_active`, the global
  VHS arm) and aborts the whole job by name rather than publishing a plate
  that mislabels a moshed or VHS-replaced programme. Selective VHS is
  upstream of the seam and legitimately included.
- **Publish last.** The generation stages before the encoder starts (its
  staging directory is private and removed on drop by every failure and
  cancellation path) and `finish` runs strictly after the MP4, the motion
  sidecar, and the gesture/performance sidecars, so no exit can leave a
  visible half-artifact. The destination is the derived sibling
  `<output>.mp4.alpha`, no-replace.

## The measurement

Hosted (all platforms, CLI-free): wire serde (omitted/null = exact prior
path, all four tokens, unknown token rejected), the authored effect-state
truth over ntsc/mosh/absent sections with plan refusal both ways, the
derived destination, and the lib source-audit pinning that the frame loop
invokes the named wrapper, that both effect guards exist, that staging
begins before the encoder, and that alpha publication follows the sidecar.

Opt-in, run on this host (AMD RX 6950 XT / Vulkan, FFmpeg 9.0.1):
`render_straight_alpha_action_pipeline` — self-provisioning on the P4c
precedent (it generates its own declared source clip;
`videos/audit.mp4` is no longer assumed):

- an authored-VHS request refuses with the named error and neither the MP4
  nor any `.alpha` name exists afterward;
- a real 24-frame straight+fill/key job publishes the MP4 and one atomic
  generation whose receipt parses with the frozen seam name; the fill and
  key laws are **re-derived from the published straight PNGs and compared
  byte-exactly for all 24 frames**, and the receipt's
  `raw_straight_rgba_sha256` is recomputed from the decoded PNGs and
  matches;
- FFV1 through the same action publishes `plate.mkv` with the
  `FFV1_v3_GBRAP` receipt via the resolved host ffmpeg;
- `d5_ffv1_gbrap_round_trips_exact_rgba` re-run under FFmpeg 9.0.1: the
  decoded plate is byte-exact against the submitted RGBA — the 8.1.2-era
  claim holds on the current toolchain.

Performance (release, this host): 24 frames at 320×180 — plain MP4
301 ms, MP4 + straight+fill/key generation 603 ms, ≈12.5 ms/frame for the
seam readback plus three PNG encodes. The overhead is per-frame CPU PNG
work plus one extra sequential readback; ordinary MP4 without the field is
structurally unchanged (no readback exists to measure).

## Registry and campaign truth

Capability `straight_alpha_key_fill` enters the registry as Implemented
with surfaces BrowserControl/OfflineExport implemented and LiveRecording
**deferred** — the live recorder owns no alpha-capable encoder, exactly as
the keep/stop receipt said. The former forbidden-string guard (registry
must not mention the capability) is replaced by positive record assertions.
Campaign `d5_straight_alpha_key_fill` moves `retained` →
`implemented/offline_action_complete_live_acquisition_deferred`; the RFC's
"cannot be reached by the current MP4 action" claim is updated with its
document-truth pin.

What is deliberately not claimed: no live-recorder alpha acquisition, no
alpha propagation law for Codec-Mosh or final-program VHS (still refused,
now at two layers), no change to ordinary MP4 bytes, and no multi-flight
readback plan — the bounded plan is one sequential readback per frame,
measured above.
