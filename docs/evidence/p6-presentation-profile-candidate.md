# P6 presentation-profile admission candidate

`src/presentation_profile.rs` makes Stable executable as the exact 30 Hz,
FIFO, maximum-frame-latency-2 baseline and defines a host-local Low Latency
candidate capped at 60 Hz with hint 1. Unsupported FIFO/hints/refresh are
unavailable rather than emulated. The candidate can be admitted only by an
exact adapter/driver/surface/build/patch calibration containing at least 30
optical trials, at least 25% action-to-photon p95 improvement, a ten-minute
60 Hz present run, no dropped/late/cadence regression or tearing, CPU/GPU p99
budget compliance, and identical accepted-reference-state and export hashes.

The pure scheduler separates 30 Hz reference-state advancement from bounded
redraws, coalesces a controller storm to one newest non-temporal revision, and
never lets a host profile alter offline export cadence. Tests pin every gate.

No qualifying optical/PresentMon campaign exists on this host, so the live app
continues to select Stable and no Low Latency control is exposed or advertised.
This is the audit-mandated stop condition, not a silent promotion based on a
frame-latency hint. The candidate may be reopened only with the P1 physical
fixture and ten-minute stage receipt named above.
