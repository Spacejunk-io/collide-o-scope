# Adversarial parser targets

The corpus is bounded, deterministic seed material; crashes and minimized
regressions belong under `fuzz/artifacts/<target>/` and must be promoted to a
normal regression test before a release.

Pinned runner commands:

```text
cargo +nightly-2026-08-20 fuzz run study_document fuzz/corpus/study_document -- -dict=fuzz/study.dict -max_len=1048576 -rss_limit_mb=2048 -timeout=5 -runs=10000
cargo +nightly-2026-08-20 fuzz run publication_gate fuzz/corpus/publication_gate -- -max_len=4096 -rss_limit_mb=2048 -timeout=5 -runs=10000
cargo +nightly-2026-08-20 fuzz run patch_yaml fuzz/corpus/patch_yaml -- -max_len=1048576 -rss_limit_mb=2048 -timeout=5 -runs=10000
cargo +nightly-2026-08-20 fuzz run controller_profile_midi fuzz/corpus/controller_profile_midi -- -max_len=263168 -rss_limit_mb=2048 -timeout=5 -runs=10000
cargo +nightly-2026-08-20 fuzz run osc_packet fuzz/corpus/osc_packet -- -max_len=16384 -rss_limit_mb=2048 -timeout=5 -runs=10000
cargo +nightly-2026-08-20 fuzz run recovery_journal_record fuzz/corpus/recovery_journal_record -- -max_len=1048576 -rss_limit_mb=2048 -timeout=5 -runs=10000
cargo +nightly-2026-08-20 fuzz run proxy_metadata fuzz/corpus/proxy_metadata -- -max_len=262144 -rss_limit_mb=2048 -timeout=5 -runs=10000
cargo +nightly-2026-08-20 fuzz run web_action_json fuzz/corpus/web_action_json -- -max_len=16384 -rss_limit_mb=2048 -timeout=5 -runs=10000
cargo +nightly-2026-08-20 fuzz run motion_sidecar_json fuzz/corpus/motion_sidecar_json -- -max_len=1048576 -rss_limit_mb=2048 -timeout=5 -runs=10000
```

Scheduled automation runs each target in its own bounded one-hour matrix job by
replacing `-runs=10000` with `-max_total_time=3600`; targets therefore do not
silently receive less time as the matrix grows.

`study_document`, `patch_yaml`, `controller_profile_midi`, `osc_packet`,
`recovery_journal_record`, `proxy_metadata`, `web_action_json`, and
`motion_sidecar_json` include the production parser/decoder source directly.
The patch target stops at the exact production hostile YAML tree boundary before
the renderer-heavy `PatchState` conversion. The recovery target uses the exact
record header/checksum/order scanner with a YAML-value stand-in because the
separate patch target owns hostile patch semantics. `publication_gate` covers
the pure latest-only state machine.

The proxy target exercises the exact pure `ProxySettings`, playback-observation,
and cache-key metadata wire laws; it does not open a media path or invoke a
codec. The WebAction target owns the exact bounded/duplicate-key-rejecting JSON
boundary now used before production deserializes the full `WebAction` enum; the
fuzz crate cannot instantiate the application-coupled enum itself. The motion
sidecar target uses the exact bounded schema/list/duplicate-key validator that
the offline exporter now runs against its serialized bytes before publication.
FFmpeg container-probe metadata remains a named expansion target: it is not
represented by a surrogate schema because its parser is libavformat itself and
needs a separately sandboxed, bounded in-memory/container fixture boundary.
