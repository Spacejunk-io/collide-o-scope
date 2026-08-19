# S8b — the slot-dance backing claim: evidence note

A small correctness tranche closing the edge the S8 hot-adoption matrix
recorded under "A pre-existing edge, observed and left honest": a clip-slot
dance A→B→A around an adopted (or patch-load-adopted) proxy reactivated the
displaced artifact-backed decoder from the prepared pool while
`Layer::proxy_backing` reported `None` — the HUD then showed a measured
assessment for a layer actually decoding the artifact, and the Y key would
walk a redundant already-cached cycle.

Branch point: `97b01e0` (`feat/web-control-panel`, the S8 hot-adoption
merge, PR #18). Baseline there: **1277 passed / 0 failed / 91 ignored**;
this tranche changes no test counts — it strengthens one existing opt-in
GPU fixture rather than adding one.

## The law

The backing claim now travels with the source resources it describes:
`LayerSourceActivation` carries `proxy_backing`. A freshly staged source is
`None` (stated at the single staging constructor); both commit paths move
the layer's current claim into the displaced record and install the
incoming activation's claim. A reactivated displaced artifact therefore
restores a truthful `proxy_backing`, and every fresh open still clears it —
the old comment's rule ("a slot activation swaps in a freshly opened
source") is now enforced by where the field lives instead of asserted by a
hard `None`.

`commit_adopted_proxy` ignores the activation's claim explicitly — the
adoption key is the caller's, exactly as the identity fields are the
layer's — so no path can smuggle a claim through a staged bundle.

## Reproduction-first, honored literally

The strengthened fixture
(`gpu_proxy_hot_adoption_swaps_a_live_layer_and_keeps_identity_and_playhead`)
was run against the unfixed code by stashing only `src/layers/mod.rs`:

```
assertion `left == right` failed: a reactivated displaced artifact
restores its truthful claim
```

— then the fix was restored and the fixture passes. The same fixture also
asserts the inverse law: a freshly opened clip switch is *not* proxy-backed.

## Verification

- Full six-step gate on `RUSTUP_TOOLCHAIN=1.97.1`: fmt, both node checks,
  check, **1277 / 0 / 91**, clippy `-D warnings` — all green on the final
  tree.
- Opt-in fixtures re-run on the final tree (FFmpeg 8.1.2, AMD Radeon
  RX 6950 XT / Vulkan): both pass.
- Same-branch decoded-`framemd5` A/B against `97b01e0`, renders directory
  cleared before each launch (the S8 process lesson): all 30 labeled export
  outputs — result recorded below.

Result: all 30 decoded `framemd5` outputs byte-identical between the fix
and `97b01e0`. The commit paths this tranche touches are live-only; the
A/B is the binding proof that export is untouched.
