# S8c — the browser proxy surface: evidence note

Gate 1's second edge, selected under the operator's delegated gate choice:
the proxy lifecycle leaves native-only. The panel's layer card gains an
Encode proxy control and a status region; the engine gains one wire action
and two additive per-layer snapshot fields; and the refusal ladder gains a
second caller without gaining a second implementation.

Branch point: `3cd8b45` (`feat/web-control-panel`, the observation-docs
merge). Its content commit `3b61022` is three-platform green; the merge
commit's own Linux job died pre-gate in "Install Linux native
prerequisites" — the apt mirror-stall infrastructure flake the S7 appendix
documented, with every gate step skipped and macOS/Windows green — and
needs an operator re-run for the record. Baseline: **1277 passed / 0
failed / 91 ignored**; with this tranche **1278 / 0 / 91** (one new
protocol-surface test).

## The design, compressed

**One ladder, two callers.** The Y-key body was extracted verbatim into
`request_proxy_for_layer(index)`; the key wraps it with selection, the new
`request_layer_proxy { layer_id }` action wraps it with stable-ID
resolution. The browser cannot bypass a refusal because there is no second
predicate to drift — the "no verified content identity" wording the panel
displays is the engine's own string.

**Stable-ID-only by construction.** The action has no positional field at
all, so a fallback cannot exist; a vanished ID is a safe no-op, the
transform-action precedent. Priority admission, no coalesce key (a request
is an event, not an absolute value), absent from the panel's hand-maintained
`QUANTIZABLE_ACTIONS`, listed in `history_action_is_performance_only` (it
authors nothing a patch captures), and non-conflicting under an Apply Look.

**Additive snapshot state.** `proxy_backing_prefix` (the HUD's own
eight-character key prefix — never a path) and `proxy_note` (the session
lifecycle/refusal note, keys and byte counts only) ride the existing layer
snapshot, `skip_serializing_if` empty, so an un-proxied layer ships zero new
bytes and a legacy snapshot deserializes to empty defaults — proven in the
extended legacy-snapshot test.

| Surface | Required proof | Status |
|---|---|---|
| Protocol | strict parse, priority, uncoalesced, unquantizable | **Covered, hosted.** `proxy_browser_surface_is_id_addressed_priority_and_never_quantizable`: mandatory `layer_id` (payload without it is a deserialization error), `is_priority()`, `coalesce_key() == None`, absent from `QUANTIZABLE_ACTIONS`, and the panel sends stable-ID-only via `currentStableLayerId`. |
| Snapshot | additive, empty-off-the-wire, legacy-safe | **Covered, hosted.** The extended `legacy_layer_snapshot_defaults_master_fx_bypass_to_off` asserts both fields absent when empty and defaulting to empty on legacy payloads. |
| No-bypass | every Y-key refusal answered identically | **Covered by construction and live.** One shared function; live QA below shows the engine's exact content-identity refusal rendered in the panel status region. |
| History/Look | no manual-history entry, Look-safe | **Covered by classification.** Listed in `history_action_is_performance_only`; the Look conflict filter's default arm preserves it, correct for an operational event. |
| Accessibility | labeled control, polite status region | **Covered.** `aria-label="Encode proxy for layer N"`, status span `role="status" aria-live="polite"` — the source-status precedent; notes change only on lifecycle events, so the region is not a fast counter. |
| Render/export/A-B | decoded-`framemd5` parity | **Not applicable, argued.** The diff touches `web/state.rs` (types/tests), static panel assets, and `main.rs` action dispatch, snapshot composition, and the Y-key refactor. No file on the render, export, or decode path changed, and export builds no `AppSnapshot`; there is no line this diff adds that a labeled export case can traverse. (The two prior tranches ran the A/B because they touched `layers/` and `video/`; this one does not.) |

## Live QA on this host (the panel itself, not a simulation)

Release build with embedded panel assets, live app, panel driven in the
Claude browser pane against the tokenized session:

1. **Request path.** A fresh content-referenced 1080p60 piece loaded; the
   card showed the enabled control; one click walked the status region
   through `proxy ready (25088665 bytes, 0s) — adopting into 1 live
   layer(s)` (12:44:27.3) to `proxy active (a147e146…)` one second later,
   with the button flipping to disabled — encode to live adoption through
   the browser in about one second, no patch reload, key matching the
   sealed artifact on disk.
2. **Refusal path.** A plain library layer (no verified identity) clicked:
   the panel rendered the engine's exact refusal — "proxy encode refused:
   this layer has no verified content identity; load it through a
   content-referenced (cos-sha256) patch first" — even though that
   identity's bytes had a cached artifact from S7, proving the ladder, not
   the cache, answers the browser.
3. **Backed rendering.** A previously adopted layer rendered
   `proxy active (be8cfa10…)` with the control disabled directly from the
   snapshot fields.

Gate on `RUSTUP_TOOLCHAIN=1.97.1`: fmt, both node checks, check, **1278 /
0 / 91**, clippy `-D warnings` — green.
