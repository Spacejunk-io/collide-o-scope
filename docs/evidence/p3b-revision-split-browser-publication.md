# P3b revision-split browser publication

This receipt records the bounded browser-publication design implemented after
the v1.6.0 audit. It does not treat a submitted GPU frame as a photon event and
does not use a browser or remote clock for engine timing.

## Wire and ownership law

- Wire version 2 has one complete `type: "state"` base and a separate
  `type: "live"` message containing explicit operational and fast-telemetry
  domains.
- Every live message names the exact `authored_revision` it requires. The
  bundled panel refuses a mismatched base and reconnects; it never renders the
  live payload against a different graph.
- A connecting socket waits for a separately cached *full* generation. Live
  publications may replace the newest-only watch value but cannot replace that
  connection base.
- The full authored object is rebuilt only for a first-client request or a new
  authored revision. Ordinary live publications contain no top-level layer,
  rack, scene, library, or effects collection.
- Operational and telemetry revisions are sampled at 12 Hz with a declared
  maximum interval of 84 ms. A newly required full state bypasses that cadence.
- With zero receivers Main returns before recovery-status capture, snapshot
  construction, JSON serialization, or publication.
- State fan-out is one `watch<Arc<String>>` value: serialize once, retain one
  newest generation, and share the same owned string with every socket.
- Action ACK/refusal messages use a separate bounded capacity-64 MPSC stream.
  They are not coalesced with state.

## Monitor inventory law

- One retained vector owns the current `MonitorHandle` inventory and opaque
  browser entries.
- Refresh boundaries are startup, winit move/scale topology signals, output
  migration, explicit Output/StageMap rescan, and a 10-second slow fallback.
- A no-change fallback scan replaces handles but does not advance the external
  generation. A changed topology/editor placement advances it exactly once.
- Current clients echo the inventory generation with display-targeting
  actions. A mismatched generation or absent ID is refused before a window is
  moved. Legacy clients retain ID-membership validation.
- The accepted-frame seam contains neither `available_monitors()` nor a monitor
  `Vec` collection.

## Automated evidence

- `web_publication_is_absent_without_receivers_and_newest_only_with_one`
  proves zero-receiver silence, one serialization for four receivers,
  revision-triggered full state, 12 Hz live state, a separate retained full
  connection base, and an immediate explicit full request.
- `browser_live_domains_are_versioned_and_refuse_the_wrong_authored_base`
  proves additive revision fields and the bundled panel's mismatch refusal.
- `authenticated_websocket_round_trip_dispatches_and_returns_authoritative_state`
  proves the authenticated socket still dispatches actions and starts from an
  authoritative full state.
- `every_socket_enqueue_gets_one_monotonic_payload_free_disposition` proves the
  separate ACK vocabulary is monotonic and payload-free.
- `newest_only_state_receiver_observes_one_latest_generation` proves backlog
  depth one semantics.
- `monitor_inventory_policy_is_event_driven_with_a_bounded_fallback` and
  `accepted_frame_path_never_enumerates_os_monitors` pin cache cadence and the
  hot-path subtraction.

## Physical/manual evidence not inferred

The Windows build host available for this implementation did not provide an
automated physical hot-plug/unplug rig or a ten-minute multi-monitor churn
fixture. Those venue checks remain a release-candidate manual matrix item; the
unit/source gates above prove the policy and hot-path structure without
claiming a physical connector event that was not run.
