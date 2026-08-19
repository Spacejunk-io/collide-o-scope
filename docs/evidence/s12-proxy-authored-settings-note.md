# S12 — authored proxy settings: evidence note

Gate 1's first remainder (S12 prompt), opened by the operator's commission:
`ProxySettings::default()` stops being the only value ever used live.
The tuple becomes authored host-session state with exactly one owner, and
the trap the prompt named — encode under one settings tuple, consult under
another — is made structurally impossible rather than procedurally avoided.

Branch point: `69204b7` (`feat/web-control-panel`, the S11 + S12-prompt
merges), verified green by the suite-aware check (`green=1 failed=0
pending=0`; the suite was still mid-run when the branch was cut and was
re-checked to completion before this claim).
Baseline: **1297 passed / 0 failed / 96 ignored**; with this tranche
**1304 / 0 / 96** — seven hosted law tests, no new ignored fixtures.

## The design, compressed

**One owner, four consumers, one audit.** `App::proxy_settings` is the
single authored owner. The HUD assessment
(`proxy_assessment_status`), the patch-load cache consultation, the
encode request, and the hot-adoption job all answer from it — the four
production `ProxySettings::default()` sites the prompt counted are now
one, the field initializer, and a source audit pins that count so a second
production default cannot return silently.

**One authoring door.** `ProxySettings::authored(scale, frame_rate,
include_audio)` is the only constructor of a non-default tuple. It stamps
this build's schema and algorithm versions — the wire carries only the
three operator choices, so a client can never smuggle a foreign version
into a cache key — and validation is built in: a zero term or over-cap
fixed rate is a typed refusal that leaves the authored owner untouched,
never a clamp onto a nearby legal tuple. The server gate applies the same
predicate, so the queue never carries a tuple the engine would refuse.

**The `set_media_safety_mode` shape, deliberately.** `set_proxy_settings`
is an ordinary immediate coalescible host action (`host:proxy-settings`,
newest complete tuple wins), never quantized, never priority, preserved
by an Apply Look barrier (operational, not creative — the
`request_layer_proxy` precedent), recording no manual history. Every edit
carries the complete absolute tuple, so one control's change can never
silently reset another under coalescing.

**Which artifact a load consults — answered, not dodged.** Each settings
tuple is its own content-addressed cache key by design. The rule is: a
load consults under the current session tuple only. Changing settings
governs future encodes and consultations; it touches no live proxy-backed
layer, invalidates no published artifact, and — like the media-safety
mode — is process-local, absent from patches, and reset to the default in
a new process. The snapshot publishes the effective tuple so every
controller sees what the next encode and the next load will use.

**The panel is honest about foreign tuples.** The three controls live
beside the media-safety section; a fixed rate authored by another
controller outside the preset list renders as its own labeled option
rather than snapping to the nearest preset. The golden cache-key test is
byte-untouched: exposing settings changed no key derivation, only which
tuple the operator selects.

| Surface | Required proof | Status |
|---|---|---|
| One owner | single production default, all consumers rewired | **Covered, hosted.** `proxy_settings_default_has_exactly_one_production_call_site` (source audit); the four consumer sites now read `self.proxy_settings`. |
| Authoring door | versions pinned, typed refusals, default identity | **Covered, hosted.** `authored_settings_carry_this_builds_versions_and_refuse_invalid_rates`. |
| Wire | strict parse, complete tuple, coalesce, no priority, no quantize | **Covered, hosted.** `proxy_settings_action_is_a_strict_coalesced_ordinary_host_action` plus the server-gate additions to the quantized-refusal fixture (invalid fixed rates refused at the gate with the engine's own predicate). |
| Engine law | install + status, refusal leaves owner untouched | **Covered, hosted.** `proxy_settings_have_one_authored_owner_and_never_enter_patches`. |
| Snapshot | additive default, three fields only | **Covered, hosted.** `proxy_settings_snapshot_is_additive_and_mirrors_the_engine_default` — an older snapshot restores the default tuple; versions never cross the wire. |
| Non-persistence | patches never own the tuple | **Covered, hosted.** Patch capture YAML carries no settings field — same fixture, the media-safety precedent. |
| Apply Look | operational action survives the barrier | **Covered, hosted.** `proxy_settings_action_survives_an_applied_look_unfiltered`. |
| Compatibility | golden cache key byte-unchanged | **Covered.** `cache_key_is_path_independent_settings_sensitive_and_has_a_golden` untouched and passing; `ProxySettings::default()` unchanged; every pre-existing proxy test passes unmodified. |
| Panel | accessible labels, sync, honest foreign tuples, not quantizable | **Covered, hosted.** `proxy_settings_static_contract_is_accessible_and_authoritative`. |
| Render/export A/B | decoded-`framemd5` parity | **Not applicable, argued.** No shader, exporter, or decoder file changed. The patch-load consultation's behavior under the default tuple — the only tuple that can exist before the new action is sent — is byte-identical to the prior hardcoded default, and export never consults the proxy cache (its digest-gated resolution rejects artifact paths by design, unchanged). |

Gate on `RUSTUP_TOOLCHAIN=1.97.1`: fmt, both node checks, check, tests,
clippy `-D warnings` — run on the final tree before commit.
