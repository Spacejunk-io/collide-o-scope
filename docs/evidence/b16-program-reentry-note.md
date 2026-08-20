# B16 — program re-entry: evidence note

Tranche B16 of the enrichment plan. The law is derived from BENDR (MIT,
© 2026 Steve Blythe): any channel may source the finished programme, and
whatever it reads is one frame old — that single frame of delay is what
makes the loop stable rather than an infinite regress. This note records
what landed, the decisions taken where the spec asked for one, and the
evidence run on this host.

## What landed

`SavedImageSource::ProgramTap` joins the closed image-route vocabulary on
the `GestureCanvas` recipe verbatim:

- **Vocabulary.** Serde tag `program_tap`; plan hash code **8**
  (append-only, after `gesture_canvas` at 7); `ResolvedImageSource` /
  `PlannedImageSource` / `TapRouteSource` / `TapBacking` /
  `CreativeImageSourceSnapshot` sibling arms; the
  `ProgramTapUnavailable { consumer }` plan diagnostic. A master-scope
  singleton: no scope, no ID, no saved position, no tombstone — a route to
  it survives every reorder, deletion, and insertion unchanged, and the
  saved-graph dependency walker claims no edge for it dormant or woken.
- **The surface.** One retained full-frame `Rgba8UnormSrgb` texture on the
  renderer (`"Program re-entry tap"`, `TEXTURE_BINDING | COPY_DST`),
  raising the renderer-owned full-frame texture floor from 29 to **30**
  with its byte literals re-pinned (1280×720 → 110,592,000; 3840×2160 →
  995,328,000). Export owns a job-lifetime twin built beside the composite
  textures.
- **What the tap holds.** The **pre-blackout opaque audience image**: final
  composite slot 2 after the opaque resolve — including display physics,
  the melting edge, temporal, and synchronous selective VHS — before the
  blackout clear and before asynchronous global-VHS replacement, which on
  both paths lands only after the copy, so live and export publish the same
  image by construction.
- **The N-1 law.** `Renderer::publish_program_tap` copies slot 2 in its own
  encoder at the frame-acceptance decision, after
  `commit_temporal_frame` — the only point at which acceptance is known,
  exactly the gesture-donor/ProgramHistory ordering. Offline publishes at
  the same seam: after the frame reaches ffmpeg, before the CPU commits
  close the transaction. The tap *is* the N-1 image: both timings read the
  same committed copy, no parity pair exists, and an N-1 route to it stages
  nothing.
- **Executor binding.** `ProgramTapBinding` (a type alias of
  `GestureCanvasBinding` — identical shape, separate bind seam) with a host
  epoch in the prepare-reuse identity, bound before `prepare` on both
  paths. An unbound tap binds the rack-owned zero texture and reports
  `donor_valid = false`.

## The two decisions the spec left to the implementation

**Blackout holds rather than clears or leaks.** Frames keep rendering under
blackout (the clear is a later, separate pass in the post-process encoder),
so an ungated copy at the acceptance decision would keep publishing live
images through an engaged emergency cut. Publication is therefore gated on
`temporal_frame_accepted && !self.blackout`: the tap **holds** the last
pre-blackout accepted image — program memory on the temporal-ring/melt
precedent (B8's ruling), not an audience wake like B4's phosphor. No frame
rendered under the cut can enter a re-entry loop, a release resumes the
loop from the picture the cut interrupted, and blackout stays absolute
because the tap re-enters only through the composite, which the downstream
clear blacks on every cut frame. Export has no blackout, so the branch is
never taken offline rather than differently taken — the source-order tests
pin both sides.

**Availability is a planner fact with asymmetric admission.** Live
admission is `renderer.program_tap_valid()` — false at process start,
false again after a patch load (`invalidate_program_tap` sits beside the
renderer's PatchGeneration reset), false in every freshly built renderer by
construction — consulted at every in-loop plan construction, so the first
accepted frame re-plans a routed tap from the named transparent diagnostic
to the tap itself. Export admits unconditionally (`with_program_tap(true)`,
the offline-canvas precedent): its surface exists for the whole job and
frame zero reads wgpu-guaranteed zero-initialized transparent — pixel
identical to live's diagnostic path, proven by arithmetic in the GPU
fixture (the frozen donor decode yields exactly zero for a fully
transparent donor). Apply Look, broad revert, and source cuts deliberately
do not invalidate: the program is continuous and the tap stays honest N-1.

## Ledger

| Item | Exact charge |
|---|---:|
| Persistent full-frame surfaces | 1 (renderer floor 29 → 30) |
| Passes beyond the copy | 0 |
| Copies per accepted non-blackout frame | 1 (`copy_texture_to_texture`) |
| Retained tap surfaces in the composition ledger | 0 (either side, fail-closed reconcile) |
| N-1 staging surfaces | 0 (the tap *is* the N-1 image) |
| New wire actions | 0 (the token rides every existing route action) |
| New modulation addresses | 0 |
| New panel sliders | 0 (range pins 190 / 19 unmoved) |
| New patch sections | 0 (routes ride ordinary `SavedImageTap` serde) |

The copy is unconditional on accepted non-blackout frames: making it
route-dependent would let an armed route read a stale or never-written tap.

## Proof

Hosted (all pass, `cargo test --locked program_tap -- --test-threads=1`):

- `visual_rack::…::the_program_tap_route_is_a_positionless_singleton_in_the_closed_vocabulary`
- `evaluated_composition::…::a_program_tap_donor_plans_outside_scope_ordering_and_charges_no_tap_surface`
- `evaluated_composition::…::an_unavailable_program_tap_route_is_transparent_and_named_rather_than_rebound`
- `patch::…::a_saved_program_tap_route_claims_no_edge_dormant_or_woken`
- `web::server::…::the_program_tap_route_is_accepted_at_ingress_at_both_timings`
- `renderer::composition::…::a_program_tap_charges_no_retained_surface_on_either_side_of_the_ledger`
- `app_state_tests::the_live_loop_publishes_the_program_tap_at_the_acceptance_decision_and_holds_it_under_blackout`
- `render_export::…::the_offline_job_publishes_the_program_tap_at_the_acceptance_decision`
- `renderer::state` floor test re-pinned at 30 with exact byte literals.

Opt-in, run on this host (AMD Radeon RX 6950 XT / Vulkan):

- `gpu_a_program_tap_donor_feeds_the_previous_frame_back_through_a_routed_displace`
  — never-published equals unbound byte-for-byte (arithmetic, not
  tolerance); a published programme copy demonstrably reaches the pixels
  through the routed Displace donor; rebinding a different tap under a new
  epoch on an unchanged topology re-prepares rather than keeping the stale
  view. **Passed.**
- `render_program_reentry_pipeline` — the labeled export case: every frame
  warped by the previous frame's audience image through the real export
  path. The `_untapped` twin carries the identical node at exact bypass
  (zero gains) inside the same Advanced plan family and decodes
  differently; the `_repeat` render decodes identically, proving the whole
  two-frame feedback chain (decode → composite → opaque resolve → tap
  publish → next frame's donor) deterministic frame-indexed offline.
  **Passed.**

Exactness A/B: `render_gesture_canvas_displace_donor_pipeline` rendered on
a pinned worktree at the base merge (8f59aac) and on this branch —
`ffmpeg -f framemd5` sequences identical excluding the `#software` line
(33 lines compared). The default path did not move a pixel.

## What is deliberately not here

- No group-scope or per-layer tap variants: the programme is one singleton.
  A second tap (e.g. per-bus) would be a new route token, not a parameter.
- No NTSC-inclusive tap: live global VHS is an asynchronous CPU replacement
  whose latency the export contract cannot reproduce, so the tap reads the
  deterministic pre-NTSC seam on both paths. Synchronous selective VHS is
  upstream of the resolve and therefore included.
- No modulation, Morph, Dice, or generator surface: a route is topology,
  and equality-based route matching covers the new variant with no new arm
  anywhere.
