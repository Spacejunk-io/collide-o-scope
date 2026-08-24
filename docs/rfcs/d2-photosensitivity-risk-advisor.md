# RFC D2 — opt-in photosensitivity risk advisor

Status: **evaluation-only reference prototype implemented; production remains
deferred pending accessibility/legal review and P1 GPU timing evidence**.

Version 1 is local and advisory-only. When explicitly enabled with an operator
venue policy, it samples a fixed 64×36 lattice from final Program after creative
effects and before audience blackout publication. A fixed four-second 30 Hz
ring retains only aggregate luma/color transition magnitude, affected-cell
count, reversal count, and sustained-window count. Asynchronous readback carries
compact counters only; no pixel, source name, authored text, path, or frame image
may enter diagnostics or the flight recorder.

The advisor emits a typed level (`clear`, `attention`, `elevated`) plus algorithm
version and the exact operator-supplied thresholds. It never attenuates pixels,
disables an effect, changes recording/export, blocks a control, or claims medical
or regulatory certification. No standards-derived default ships without a
separately cited review; absent policy means unavailable, not a guessed policy.

Promotion requires deterministic static, slow-fade, small-area flash,
full-field alternating, red-saturated, irregular-cadence, blackout, and frozen
fixtures, constant 2,304-sample work independent of output raster, bounded
readback delay, and P1 proof that GPU/readback p95/p99 remains inside budget.

The isolated prototype and its stop-gate receipt are recorded in
`docs/evidence/d2-photosensitivity-advisor-prototype.md`. It is not constructed
by the application or renderer, so this RFC does not declare an available live
capability.
