# RFC D3 — portable verified `.cosbundle`

Status: **version-1 deterministic core and the operator UI implemented; the
machine-A / clean-machine-B promotion fixture remains gated**. The operator
surface landed in the follow-on tranche
(`docs/evidence/d3-show-bundle-ui-note.md`): panel Export/Preview/Import
controls over a wire action family, with the side-effect-free preview
presented and explicitly confirmed before any import commits — exactly the
requirement stated below. The capability stays deliberately unadvertised in
the registry until the cross-machine reproduction receipt exists.

The `COSBNDL\0` container version 1 contains a canonical JSON manifest and an
uncompressed, contiguous payload. Entries include canonical hostile-round-trip
patch bytes; original media keyed by SHA-256, byte length, and a bounded logical
name; complete canonical Study documents; gesture/take sidecars; and optional,
individually selected controller/venue profiles and evidence receipts. A proxy
is permitted only as a derived alternative naming an original digest present in
the same bundle. It can never become patch identity or satisfy a missing
original.

The container uses deterministic lexical entry order, normalized `/` separators,
portable ASCII logical components, and no timestamps. The manifest declares
every entry's stored/uncompressed length, digest, role, optional original link,
and license-preview text. Version-1 defaults cap the manifest at 4 MiB, entries
at 4,096, path depth at 4, each component at 240 bytes, authored documents at
32 MiB, and both individual media and aggregate payload at 64 GiB. Compression
is deliberately absent, so the only admitted expansion ratio is exactly 1:1.
Limits are checked before large reads or extraction and again while streaming.

Inspection is side-effect free and verifies the complete bundle, every entry,
canonical manifest/patch/sidecar bytes, and all original/proxy links before
import creates staging. Import uses a create-new generation directory beneath a
caller-selected no-follow library root, refuses absolute/drive/UNC/`..`/empty/
device/reserved/case-fold-colliding names, links, duplicate names, short reads,
and undeclared data. Each create-new staged file is hashed and synced before the
generation and parent directories are synced and published with true no-replace
semantics. Collision policy is fail or reuse only after byte-for-byte
reverification. Cancel, disk-full, crash, or verification failure leaves the
existing library unchanged; startup cleanup removes only a bounded set of
fixed-prefix staging residue.

Promotion still requires machine-A export / clean-machine-B import plus the
live-export reproduction fixture. The implemented core already covers missing,
tamper, compressed/zip-bomb, duplicate, case-fold, traversal, symlink,
short-read, one-byte-over, disk-full, cancel, crash, and final-name-race
fixtures. The operator UI presents the side-effect-free preview (total size,
logical paths, licenses, roles) before commit, and confirmation re-inspects
the file and refuses a digest that moved since the preview.
