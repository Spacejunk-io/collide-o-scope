# D3 — the show-bundle operator UI: evidence note

Date: 2026-08-26. The operator surface the keep receipt
(`docs/evidence/d3-portable-show-bundle.md`) named as outside its core
tranche. The deterministic build/inspect/import core is unchanged; this
tranche gives the operator a way to reach it. The cross-machine promotion
gate — machine-A export to clean-machine-B live/export reproduction — is
**not** claimed here: it needs a second machine, the campaign row stays
`retained` with that exact gate, and the capability stays deliberately
unadvertised in the registry until that receipt exists.

## The surface

Four wire actions beside the patch controls, each uncoalesced priority and
outside manual history (a bundle operation is an operational event):

- `export_show_bundle` — host-native save picker, then one transactional
  build of the complete current performance state.
- `preview_show_bundle` — host-native picker, then the side-effect-free
  inspection; the verified preview is held as pending state and published.
- `confirm_show_bundle_import { load }` — imports the pending bundle into
  the active library as one atomic no-replace generation; `load` applies
  the imported patch as a complete snapshot through the exact
  `apply_loaded_patch` path the patch-load dialog uses.
- `cancel_show_bundle_import` — discards the pending preview.

The panel's Show bundle section carries the buttons, the preview region
(path, entry count, expanded bytes, patch digest, per-entry
kind/name/size/authority/license, bounded at 64 rows with an honest
"more" line), and a live status row. The RFC's stated requirement — the
side-effect-free preview presented **before** commit — is structural: the
confirm actions exist only inside the preview region, and the engine
refuses a confirm with no standing preview.

## The laws

- **The media collector mirrors the rewriter.**
  `collect_bundle_media_inputs` (in `show_bundle.rs`, beside the walker it
  must match) enumerates exactly the references
  `rewrite_patch_to_content_identities` visits — layer source, every clip
  slot, the file analysis-audio clip — resolves each through the one
  shared `media_source` resolver (content references included, with the
  bounded fingerprint budget), skips self-contained sources exactly as the
  rewriter skips them, groups by canonicalized resolved path so one file
  referenced three ways becomes one original entry, and refuses an
  unresolvable source by name rather than building a bundle with a
  silently absent original.
- **Preview is display, never import authority.** Confirmation re-inspects
  the bundle file and refuses a digest that moved since the preview, so
  the operator imports exactly the artifact they inspected; the core's own
  full verification then runs again inside `import_show_bundle`.
- **Modal honesty.** Every flow — picker, capture, fingerprinting, build,
  inspect, import, optional load — runs under the native-modal clock pause
  the patch dialogs use, so wall time inside a long hash never becomes
  program catch-up debt. A long export stalls the held frame for its
  duration; a worker seat is a possible follow-on, deliberately not
  smuggled in here.
- **Failure surfaces.** Every refusal lands in the published status; a
  successful import that then fails to load reports both facts and points
  at the imported patch path.

## The measurement

Hosted (all platforms, CLI-free): the collector fixture proves the
three-spelling grouping (content reference + plain path + library-filename
resolve to one original with both rewrite spellings and the recorded
identity), that the collected inputs build a bundle whose preview carries
every original, self-contained skipping, and the named missing-source
refusal; the protocol test pins the four-action vocabulary (confirm `load`
defaulting false), priority/history/coalesce classification, the snapshot
round trip with pre-D3 decode compatibility, and every panel id, action
string, and both sync sites. The existing 12-test deterministic core suite
is untouched and green beside them.

What is deliberately not claimed: no cross-machine reproduction (the
standing promotion gate), no registry capability row, no background
build/import worker, no bundled controller/venue documents or proxies from
this surface yet (the core supports them; the surface exports patch +
originals in v1), and no browser-remote file transfer — the pickers are
host-native by design, on the `open_patch_snapshot` precedent.
