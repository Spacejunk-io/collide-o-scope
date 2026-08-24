#!/usr/bin/env python3
"""Prove the vendored wgpu-hal tree is upstream plus one declared patch."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "third_party" / "wgpu-hal-29.0.3.vendor.json"
VENDOR_ROOT = ROOT / "third_party" / "wgpu-hal-29.0.3"
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024


class VerificationError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_manifest() -> dict:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"read {MANIFEST_PATH}: {error}") from error
    if manifest.get("schema_version") != 1:
        raise VerificationError("unsupported vendor manifest schema")
    return manifest


def cached_archive(name: str, version: str) -> Path | None:
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    candidates = sorted((cargo_home / "registry" / "cache").glob(f"*/{name}-{version}.crate"))
    return candidates[0] if candidates else None


def read_archive(manifest: dict, explicit: Path | None, offline: bool) -> bytes:
    crate = manifest["crate"]
    source = explicit or cached_archive(crate["name"], crate["version"])
    if source is not None:
        try:
            data = source.read_bytes()
        except OSError as error:
            raise VerificationError(f"read crate archive {source}: {error}") from error
    elif offline:
        raise VerificationError("crate archive is not cached and --offline forbids download")
    else:
        request = urllib.request.Request(
            crate["archive_url"],
            headers={"User-Agent": "collide-o-scope-vendor-verifier/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(MAX_ARCHIVE_BYTES + 1)
        except OSError as error:
            raise VerificationError(f"download crate archive: {error}") from error
    if len(data) > MAX_ARCHIVE_BYTES:
        raise VerificationError(f"crate archive exceeds {MAX_ARCHIVE_BYTES} bytes")
    actual = sha256(data)
    expected = crate["archive_sha256"].lower()
    if actual != expected:
        raise VerificationError(f"crate archive SHA-256 {actual} != pinned {expected}")
    return data


def archive_files(data: bytes, name: str, version: str) -> dict[str, bytes]:
    prefix = f"{name}-{version}"
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    raise VerificationError(f"unsafe archive member {member.name!r}")
                if path.parts[0] != prefix:
                    raise VerificationError(f"archive member outside {prefix}/: {member.name!r}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise VerificationError(f"non-regular archive member {member.name!r}")
                relative = PurePosixPath(*path.parts[1:]).as_posix()
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise VerificationError(f"cannot read archive member {member.name!r}")
                files[relative] = extracted.read()
    except (tarfile.TarError, OSError) as error:
        raise VerificationError(f"read crate tarball: {error}") from error
    return files


def disk_files(root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise VerificationError(f"vendored tree contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files[relative] = path.read_bytes()
    return files


def apply_single_file_patch(original: bytes, patch_bytes: bytes, expected_path: str) -> bytes:
    try:
        old = original.decode("utf-8").splitlines(keepends=True)
        patch = patch_bytes.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise VerificationError(f"patch or patched source is not UTF-8: {error}") from error
    old_header = f"--- a/{expected_path}"
    new_header = f"+++ b/{expected_path}"
    if not patch or patch[0].rstrip("\r\n") != old_header or len(patch) < 3:
        raise VerificationError(f"patch does not start with {old_header!r}")
    if patch[1].rstrip("\r\n") != new_header:
        raise VerificationError(f"patch does not target only {new_header!r}")

    import re

    hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    result: list[str] = []
    source_index = 0
    index = 2
    hunks = 0
    while index < len(patch):
        header = patch[index].rstrip("\r\n")
        match = hunk_pattern.match(header)
        if match is None:
            raise VerificationError(f"unexpected patch line outside a hunk: {header!r}")
        old_start = int(match.group(1)) - 1
        if old_start < source_index or old_start > len(old):
            raise VerificationError("patch hunk has an invalid or overlapping source range")
        result.extend(old[source_index:old_start])
        source_index = old_start
        index += 1
        consumed_old = 0
        emitted_new = 0
        expected_old = int(match.group(2) or "1")
        expected_new = int(match.group(4) or "1")
        while index < len(patch) and not patch[index].startswith("@@ "):
            line = patch[index]
            if line.startswith("\\ No newline at end of file"):
                raise VerificationError("newline marker is unsupported; preserve exact LF source")
            if not line or line[0] not in " +-":
                raise VerificationError(f"invalid unified patch line: {line!r}")
            payload = line[1:]
            if line[0] in " -":
                if source_index >= len(old) or old[source_index] != payload:
                    raise VerificationError("patch context does not match pinned upstream source")
                source_index += 1
                consumed_old += 1
            if line[0] in " +":
                result.append(payload)
                emitted_new += 1
            index += 1
        if consumed_old != expected_old or emitted_new != expected_new:
            raise VerificationError("patch hunk line counts do not match its header")
        hunks += 1
    if hunks != 1:
        raise VerificationError(f"expected exactly one patch hunk, found {hunks}")
    result.extend(old[source_index:])
    return "".join(result).encode("utf-8")


def verify_tree(manifest: dict, upstream: dict[str, bytes], vendor_root: Path) -> dict:
    vendor = disk_files(vendor_root)
    normalization = manifest["normalization"]
    omitted = set(normalization["upstream_files_intentionally_omitted"])
    generated = normalization["vendor_generated_files"]
    delta = manifest["intended_delta"]
    delta_path = delta["path"]

    upstream_paths = set(upstream)
    vendor_paths = set(vendor)
    missing = sorted((upstream_paths - omitted) - vendor_paths)
    unexpected = sorted(vendor_paths - (upstream_paths - omitted) - set(generated))
    wrongly_present = sorted(omitted & vendor_paths)
    absent_omissions = sorted(omitted - upstream_paths)
    if missing or unexpected or wrongly_present or absent_omissions:
        raise VerificationError(
            "vendor path-set mismatch: "
            f"missing={missing}, unexpected={unexpected}, "
            f"wrongly_present={wrongly_present}, absent_omissions={absent_omissions}"
        )

    for path, expected_hash in generated.items():
        actual = vendor.get(path)
        if actual is None or sha256(actual) != expected_hash.lower():
            raise VerificationError(f"generated vendor file {path} does not match its pinned hash")

    changed = []
    for path in sorted((upstream_paths - omitted) & vendor_paths):
        if upstream[path] != vendor[path]:
            changed.append(path)
    if changed != [delta_path]:
        raise VerificationError(f"vendored content deltas {changed!r} != declared {[delta_path]!r}")

    upstream_source = upstream[delta_path]
    vendored_source = vendor[delta_path]
    if sha256(upstream_source) != delta["upstream_sha256"].lower():
        raise VerificationError("declared upstream source hash does not match the crate")
    if sha256(vendored_source) != delta["vendored_sha256"].lower():
        raise VerificationError("declared vendored source hash does not match the tree")
    patch_path = ROOT / delta["patch"]
    patch_bytes = patch_path.read_bytes()
    if sha256(patch_bytes) != delta["patch_sha256"].lower():
        raise VerificationError("checked-in patch hash does not match the vendor manifest")
    if apply_single_file_patch(upstream_source, patch_bytes, delta_path) != vendored_source:
        raise VerificationError("declared patch does not exactly reproduce the vendored source")

    return {
        "schema_version": 1,
        "verified": True,
        "crate": f"{manifest['crate']['name']}@{manifest['crate']['version']}",
        "archive_sha256": manifest["crate"]["archive_sha256"],
        "normalized_file_count": len(upstream_paths - omitted),
        "intended_delta": delta_path,
        "vendored_sha256": delta["vendored_sha256"],
    }


def self_test(manifest: dict, upstream: dict[str, bytes]) -> None:
    verify_tree(manifest, upstream, VENDOR_ROOT)
    with tempfile.TemporaryDirectory(prefix="cos-vendor-verifier-") as temp:
        mutated = Path(temp) / "wgpu-hal"
        shutil.copytree(VENDOR_ROOT, mutated)
        target = mutated / "README.md"
        target.write_bytes(target.read_bytes() + b"\nmutation the verifier must reject\n")
        try:
            verify_tree(manifest, upstream, mutated)
        except VerificationError:
            return
        raise VerificationError("self-test mutation was incorrectly accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="explicit .crate archive")
    parser.add_argument("--offline", action="store_true", help="never download a missing archive")
    parser.add_argument("--self-test", action="store_true", help="also prove undeclared drift is rejected")
    args = parser.parse_args()
    try:
        manifest = read_manifest()
        data = read_archive(manifest, args.archive, args.offline)
        crate = manifest["crate"]
        upstream = archive_files(data, crate["name"], crate["version"])
        if args.self_test:
            self_test(manifest, upstream)
        receipt = verify_tree(manifest, upstream, VENDOR_ROOT)
    except VerificationError as error:
        print(f"vendor verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
