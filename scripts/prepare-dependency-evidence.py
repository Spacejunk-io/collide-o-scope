#!/usr/bin/env python3
"""Create a deterministic, path-free cargo-deny license inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import urllib.parse


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CARGO_DENY_VERSION = "cargo-deny 0.20.2"
PATH_SOURCE = re.compile(r"^(?P<package>[^\r\n]+) path\+file://(?P<path>[^\r\n]+)$")
CANONICAL_VENDOR_SOURCE = "path+third_party/wgpu-hal-29.0.4"
CANONICAL_ROOT_SOURCE = "path+."
MAX_INVENTORY_BYTES = 4 * 1024 * 1024


class EvidenceError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_package_identity() -> str:
    with (ROOT / "Cargo.toml").open("rb") as source:
        package = tomllib.load(source)["package"]
    return f"{package['name']} {package['version']}"


def cargo_deny_version() -> str:
    try:
        completed = subprocess.run(
            ["cargo", "deny", "--version"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EvidenceError(f"read cargo-deny version: {error}") from error
    version = completed.stdout.strip()
    if version != EXPECTED_CARGO_DENY_VERSION:
        raise EvidenceError(
            f"expected {EXPECTED_CARGO_DENY_VERSION!r}, observed {version!r}"
        )
    return version


def cargo_deny_inventory() -> dict:
    try:
        completed = subprocess.run(
            ["cargo", "deny", "list", "--format", "json"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EvidenceError(f"generate cargo-deny inventory: {error}") from error
    if len(completed.stdout) > MAX_INVENTORY_BYTES:
        raise EvidenceError(
            f"cargo-deny inventory exceeds {MAX_INVENTORY_BYTES} bytes"
        )
    try:
        inventory = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cargo-deny emitted invalid JSON: {error}") from error
    if (
        not isinstance(inventory, dict)
        or not isinstance(inventory.get("licenses"), list)
        or not isinstance(inventory.get("unlicensed"), list)
    ):
        raise EvidenceError("cargo-deny inventory has an unsupported shape")
    return inventory


def normalize_string(value: str, normalized_sources: list[str]) -> str:
    match = PATH_SOURCE.fullmatch(value)
    if match is None:
        if "path+file://" in value:
            raise EvidenceError(f"unrecognized absolute path source in {value!r}")
        return value
    package = match.group("package")
    encoded_path = match.group("path").replace("%5C", "/").replace("%5c", "/")
    root_identity = root_package_identity()
    decoded_path = urllib.parse.unquote(encoded_path).replace("\\", "/").rstrip("/")
    expected_root = ROOT.as_posix().rstrip("/")
    if re.fullmatch(r"/[A-Za-z]:/.*", decoded_path):
        decoded_path = decoded_path[1:]
    paths_equal = (
        decoded_path.casefold() == expected_root.casefold()
        if re.match(r"^[A-Za-z]:/", expected_root)
        else decoded_path == expected_root
    )
    if package == root_identity and paths_equal:
        normalized_sources.append(value)
        return f"{package} {CANONICAL_ROOT_SOURCE}"
    expected_vendor = (ROOT / "third_party" / "wgpu-hal-29.0.4").as_posix().rstrip("/")
    vendor_paths_equal = (
        decoded_path.casefold() == expected_vendor.casefold()
        if re.match(r"^[A-Za-z]:/", expected_vendor)
        else decoded_path == expected_vendor
    )
    if package != "wgpu-hal 29.0.4" or not vendor_paths_equal:
        raise EvidenceError(f"undeclared path dependency in inventory: {value!r}")
    normalized_sources.append(value)
    return f"{package} {CANONICAL_VENDOR_SOURCE}"


def normalize(value: object, normalized_sources: list[str]) -> object:
    if isinstance(value, str):
        return normalize_string(value, normalized_sources)
    if isinstance(value, list):
        return [normalize(item, normalized_sources) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item, normalized_sources) for key, item in value.items()}
    return value


def prepare(output: Path) -> dict:
    version = cargo_deny_version()
    source_inventory = cargo_deny_inventory()
    normalized_sources: list[str] = []
    inventory = normalize(source_inventory, normalized_sources)
    root_occurrences = sum(
        value.startswith(f"{root_package_identity()} path+file://")
        for value in normalized_sources
    )
    vendor_occurrences = sum(
        value.startswith("wgpu-hal 29.0.4 path+file://")
        for value in normalized_sources
    )
    if root_occurrences != 1 or vendor_occurrences < 1:
        raise EvidenceError(
            "expected exactly the root and declared wgpu-hal path sources, "
            f"observed root={root_occurrences}, vendor={vendor_occurrences}"
        )
    if inventory.get("unlicensed"):
        raise EvidenceError("cargo-deny inventory contains unlicensed dependencies")
    evidence = {
        "schema_version": 1,
        "generator": version,
        "cargo_lock_sha256": sha256(ROOT / "Cargo.lock"),
        "normalization": {
            "absolute_path_sources_removed": len(normalized_sources),
            "declared_root_source": CANONICAL_ROOT_SOURCE,
            "declared_vendor_source": CANONICAL_VENDOR_SOURCE,
            "root_source_occurrences": root_occurrences,
            "vendor_source_occurrences": vendor_occurrences,
        },
        "inventory": inventory,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return evidence


def self_test() -> None:
    with (ROOT / "Cargo.toml").open("rb") as source:
        root_package = tomllib.load(source)["package"]
    sample = {
        "licenses": [["MIT", [
            "example 1.0.0 registry+https://github.com/rust-lang/crates.io-index",
            f"{root_package['name']} {root_package['version']} path+{ROOT.as_uri()}",
            f"wgpu-hal 29.0.4 path+{(ROOT / 'third_party' / 'wgpu-hal-29.0.4').as_uri()}",
        ]]],
        "unlicensed": [],
    }
    observed: list[str] = []
    normalized = normalize(sample, observed)
    assert len(observed) == 2
    assert CANONICAL_ROOT_SOURCE in normalized["licenses"][0][1][1]
    assert CANONICAL_VENDOR_SOURCE in normalized["licenses"][0][1][2]
    assert str(ROOT) not in json.dumps(normalized)
    try:
        normalize_string("unknown 1.0.0 path+file:///tmp/unknown", [])
    except EvidenceError:
        pass
    else:
        raise AssertionError("undeclared path dependency was not rejected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        if args.output is not None:
            evidence = prepare(args.output.resolve())
            print(json.dumps({
                "dependency_inventory": str(args.output),
                "cargo_lock_sha256": evidence["cargo_lock_sha256"],
            }, sort_keys=True))
        elif not args.self_test:
            parser.error("one of --output or --self-test is required")
    except (EvidenceError, OSError) as error:
        print(f"dependency evidence failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
