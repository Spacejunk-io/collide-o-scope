#!/usr/bin/env python3
"""Normalize and validate the reviewed cargo-cyclonedx release profile.

This is intentionally stricter than a general CycloneDX parser.  The release
accepts only the exact shape emitted by pinned cargo-cyclonedx 0.5.9 for this
workspace, replaces checkout-local component identities through an injective
reference map, and rejects every unreviewed reference-bearing family.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import tomllib
from typing import Any, Callable
import urllib.parse


EXPECTED_BOM_FORMAT = "CycloneDX"
EXPECTED_SPEC_VERSION = "1.5"
EXPECTED_DOCUMENT_VERSION = 1
EXPECTED_PACKAGE_NAME = "collide-o-scope"
EXPECTED_REPOSITORY_URL = "https://github.com/Spacejunk-io/collide-o-scope"
EXPECTED_TOOL = [{"vendor": "CycloneDX", "name": "cargo-cyclonedx", "version": "0.5.9"}]
EXPECTED_TARGET_PROPERTY = [
    {"name": "cdx:rustc:sbom:target:triple", "value": "x86_64-pc-windows-msvc"}
]
EXPECTED_TOP_COMPONENTS = 364
EXPECTED_TARGET_COMPONENTS = 6
EXPECTED_REGISTRY_COMPONENTS = 362
EXPECTED_GIT_COMPONENTS = 1
EXPECTED_DEPENDENCY_ROWS = 365
EXPECTED_DEPENDENCY_EDGES = 873
EXPECTED_ROOT_EDGES = 36
EXPECTED_LOCAL_DECLARATIONS = 8
EXPECTED_REWRITTEN_REFERENCES = 13
EXPECTED_SEMANTIC_PROFILE_SHA256 = (
    "c1d433f8cf2d592f686d042e96703ef238b137a45ffa895bbe1d88f44c8d1331"
)
SEMANTIC_SOURCE_PLACEHOLDER = "<exact-collide-o-scope-source-commit>"
SEMANTIC_TIMESTAMP_PLACEHOLDER = "<exact-source-date-epoch>"
EXPECTED_VENDOR_PURL = (
    "pkg:cargo/wgpu-hal@29.0.3?download_url=file://third_party\\wgpu-hal-29.0.3"
)
MAX_SBOM_BYTES = 8 * 1024 * 1024
ALLOWED_TOP_LEVEL_KEYS = {
    "bomFormat",
    "specVersion",
    "version",
    "metadata",
    "components",
    "dependencies",
}
EXPECTED_METADATA_KEYS = {"timestamp", "tools", "component", "properties"}
EXPECTED_ROOT_COMPONENT_KEYS = {
    "type",
    "bom-ref",
    "name",
    "version",
    "description",
    "scope",
    "licenses",
    "purl",
    "externalReferences",
    "components",
}
EXPECTED_TARGET_COMPONENT_KEYS = {"type", "bom-ref", "name", "version", "purl"}
ALLOWED_DEPENDENCY_COMPONENT_KEYS = {
    "type",
    "bom-ref",
    "author",
    "name",
    "version",
    "description",
    "scope",
    "hashes",
    "licenses",
    "purl",
    "externalReferences",
}
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
VERSION = re.compile(r"^\d+\.\d+\.\d+$")
REGISTRY_REFERENCE = re.compile(
    r"^registry\+https://github\.com/rust-lang/crates\.io-index#[A-Za-z0-9_.-]+@[^\s#]+$"
)
GIT_REFERENCE = re.compile(
    r"^git\+https://github\.com/ntsc-rs/ntsc-rs\?rev="
    r"4b79500dfac64efcfb393eebc89f5c75565ee5ae#0\.1\.2$"
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:^|[^A-Za-z0-9])(?:[A-Z]:[\\/])")
WINDOWS_FILE_URI = re.compile(r"(?i)file:(?:[/\\]{0,4})[A-Z]:[\\/]")
UNC_PATH = re.compile(r"(?i)(?:^|[\s=\"'(])(?:\\\\|//)[A-Za-z0-9.$_-]+[\\/]")
UNC_FILE_URI = re.compile(r"(?i)file://(?![./])[^/?#\\\s]+[\\/]")
POSIX_BUILDER_PATH = re.compile(
    r"(?:^|[\s=\"'(])/(?:home|Users|private|tmp|var(?:/folders)?|workspace|"
    r"workspaces|runner|__w|mnt|opt|srv)(?:/|\\)",
    re.IGNORECASE,
)
POSIX_ABSOLUTE_PATH = re.compile(
    r"(?:^|[\s=\"'(])/(?!/)[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+",
    re.IGNORECASE,
)
POSIX_FILE_URI = re.compile(
    r"(?i)file:/+(?:[A-Za-z0-9._-]+[/\\])+(?:[A-Za-z0-9._-]+)?"
)
PathKey = tuple[str | int, ...]


class SbomPolicyError(ValueError):
    """Raised when an SBOM violates the reviewed release profile."""


def fail(message: str) -> None:
    raise SbomPolicyError(message)


def canonical_source_uri(commit: str) -> str:
    if not isinstance(commit, str) or GIT_SHA.fullmatch(commit) is None:
        fail("SBOM commit must be one exact lowercase Git SHA")
    return f"{EXPECTED_REPOSITORY_URL}/tree/{commit}"


def expected_timestamp(source_date_epoch: int) -> str:
    if type(source_date_epoch) is not int or source_date_epoch < 0:
        fail("SOURCE_DATE_EPOCH must be a non-negative integer")
    try:
        instant = datetime.fromtimestamp(source_date_epoch, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        fail(f"SOURCE_DATE_EPOCH is outside the supported range: {error}")
    return instant.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"SBOM JSON contains duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    fail(f"SBOM JSON contains non-finite constant {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("SBOM JSON contains a non-finite number")
    return parsed


def parse_json_bytes(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_SBOM_BYTES:
        fail("SBOM JSON exceeds the bounded input size")
    if raw.startswith(b"\xef\xbb\xbf"):
        fail("SBOM JSON must be UTF-8 without a byte-order mark")
    try:
        text = raw.decode("utf-8")
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        fail(f"SBOM input is not strict UTF-8 JSON: {error}")
    if not isinstance(document, dict):
        fail("SBOM root must be an object")
    return document


def _has_reparse_attribute(status: os.stat_result) -> bool:
    return bool(getattr(status, "st_file_attributes", 0) & 0x400)


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        fail(f"cannot inspect SBOM path {path}: {error}")


def require_ordinary_file(path: Path, label: str) -> os.stat_result:
    status = _lstat(path)
    if stat.S_ISLNK(status.st_mode) or _has_reparse_attribute(status):
        fail(f"{label} must not be a link or reparse point")
    if not stat.S_ISREG(status.st_mode):
        fail(f"{label} must be an ordinary file")
    return status


def require_ordinary_directory(path: Path, label: str) -> os.stat_result:
    status = _lstat(path)
    if stat.S_ISLNK(status.st_mode) or _has_reparse_attribute(status):
        fail(f"{label} must not be a link or reparse point")
    if not stat.S_ISDIR(status.st_mode):
        fail(f"{label} must be an ordinary directory")
    return status


def require_unlinked_ancestry(path: Path, label: str) -> None:
    lexical = Path(os.path.abspath(path))
    candidates = [lexical, *lexical.parents]
    for candidate in candidates:
        if not candidate.exists() and not candidate.is_symlink():
            continue
        status = _lstat(candidate)
        if stat.S_ISLNK(status.st_mode) or _has_reparse_attribute(status):
            fail(f"{label} ancestry contains a link or reparse point: {candidate}")


def read_json(path: Path) -> dict[str, Any]:
    lexical = Path(os.path.abspath(path))
    before = require_ordinary_file(lexical, "SBOM input")
    try:
        with lexical.open("rb") as source:
            opened = os.fstat(source.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                fail("SBOM input changed while it was opened")
            raw = source.read(MAX_SBOM_BYTES + 1)
    except OSError as error:
        fail(f"cannot read SBOM input: {error}")
    after = require_ordinary_file(lexical, "SBOM input")
    if (after.st_dev, after.st_ino, after.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ):
        fail("SBOM input changed while it was read")
    return parse_json_bytes(raw)


def manifest_identity(source_root: Path) -> tuple[str, str]:
    manifest_path = source_root / "Cargo.toml"
    lock_path = source_root / "Cargo.lock"
    try:
        with manifest_path.open("rb") as source:
            manifest = tomllib.load(source)
        with lock_path.open("rb") as source:
            lock = tomllib.load(source)
        package = manifest["package"]
        name = package["name"]
        version = package["version"]
        repository = package["repository"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        fail(f"cannot derive package identity from reviewed Cargo inputs: {error}")
    if (
        name != EXPECTED_PACKAGE_NAME
        or not isinstance(version, str)
        or VERSION.fullmatch(version) is None
        or repository != EXPECTED_REPOSITORY_URL
    ):
        fail("Cargo package or repository identity is outside the release profile")
    root_rows = [
        package_row
        for package_row in lock.get("package", [])
        if isinstance(package_row, dict)
        and package_row.get("name") == name
        and package_row.get("version") == version
        and "source" not in package_row
    ]
    if len(root_rows) != 1:
        fail("Cargo.lock does not contain one exact root package identity")
    return name, version


def source_uri(source_root: Path) -> str:
    try:
        return "path+" + source_root.as_uri()
    except ValueError as error:
        fail(f"source root cannot be represented as a file URI: {error}")


def _json_path(path: PathKey) -> str:
    result = "$"
    for value in path:
        result += f"[{value}]" if isinstance(value, int) else f".{value}"
    return result


def iter_string_paths(value: Any, path: PathKey = ()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from iter_string_paths(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_string_paths(item, (*path, index))
    elif isinstance(value, str):
        yield path, value


def iter_dict_paths(value: Any, path: PathKey = ()):
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from iter_dict_paths(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_dict_paths(item, (*path, index))


def decoded_forms(value: str):
    observed: set[str] = set()
    current = value
    for _ in range(17):
        if current in observed:
            return
        observed.add(current)
        yield current
        decoded = urllib.parse.unquote(current)
        if decoded == current:
            return
        current = decoded
    fail("SBOM string uses excessive nested percent-encoding")


def looks_like_absolute_builder_path(value: str) -> bool:
    for form in decoded_forms(value):
        if (
            WINDOWS_ABSOLUTE_PATH.search(form)
            or WINDOWS_FILE_URI.search(form)
            or UNC_PATH.search(form)
            or UNC_FILE_URI.search(form)
            or POSIX_BUILDER_PATH.search(form)
            or POSIX_ABSOLUTE_PATH.search(form)
            or POSIX_FILE_URI.search(form)
        ):
            return True
    return False


def component_layout(document: dict[str, Any]) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[int, PathKey]
]:
    metadata = document.get("metadata")
    root = metadata.get("component") if isinstance(metadata, dict) else None
    top = document.get("components")
    if not isinstance(root, dict) or not isinstance(top, list):
        fail("SBOM component inventory is absent")
    targets = root.get("components")
    if not isinstance(targets, list):
        fail("SBOM root target inventory is absent")
    if len(targets) != EXPECTED_TARGET_COMPONENTS:
        fail("SBOM target component count differs from the reviewed profile")
    if len(top) != EXPECTED_TOP_COMPONENTS:
        fail("SBOM dependency component count differs from the reviewed profile")
    if any(not isinstance(item, dict) for item in targets + top):
        fail("SBOM component inventory contains a non-object")
    for item in targets + top:
        children = item.get("components", [])
        if not isinstance(children, list) or children:
            fail("SBOM release profile does not allow deeper component nesting")
    for index, item in enumerate(targets):
        if set(item) != EXPECTED_TARGET_COMPONENT_KEYS:
            fail(f"SBOM binary target shape is unexpected: {index}")
    for index, item in enumerate(top):
        if (
            not {"type", "bom-ref", "name", "version", "purl"}.issubset(item)
            or not set(item).issubset(ALLOWED_DEPENDENCY_COMPONENT_KEYS)
            or item.get("type") != "library"
            or not all(isinstance(item.get(field), str) and item[field] for field in ("name", "version", "purl"))
        ):
            fail(f"SBOM dependency component shape is unexpected: {index}")
    locations = {id(root): ("metadata", "component")}
    locations.update(
        {id(item): ("metadata", "component", "components", index) for index, item in enumerate(targets)}
    )
    locations.update({id(item): ("components", index) for index, item in enumerate(top)})
    return root, targets, top, locations


def reference_slots(
    document: dict[str, Any],
) -> list[tuple[Any, str | int, PathKey]]:
    root, targets, top, locations = component_layout(document)
    slots: list[tuple[Any, str | int, PathKey]] = []
    for component in [root, *targets, *top]:
        path = (*locations[id(component)], "bom-ref")
        reference = component.get("bom-ref")
        if not isinstance(reference, str) or not reference:
            fail(f"SBOM component lacks a nonempty bom-ref: {_json_path(path)}")
        slots.append((component, "bom-ref", path))
    dependencies = document.get("dependencies")
    if not isinstance(dependencies, list):
        fail("SBOM dependency graph is absent")
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, dict):
            fail(f"SBOM dependency row is not an object: dependencies[{index}]")
        reference = dependency.get("ref")
        if not isinstance(reference, str) or not reference:
            fail(f"SBOM dependency row lacks a ref: dependencies[{index}]")
        slots.append((dependency, "ref", ("dependencies", index, "ref")))
        depends_on = dependency.get("dependsOn", [])
        if not isinstance(depends_on, list):
            fail(f"SBOM dependency edges are not an array: dependencies[{index}]")
        for edge_index, edge in enumerate(depends_on):
            if not isinstance(edge, str) or not edge:
                fail(f"SBOM dependency edge is malformed: dependencies[{index}]")
            slots.append(
                (
                    depends_on,
                    edge_index,
                    ("dependencies", index, "dependsOn", edge_index),
                )
            )
    return slots


def validate_header(
    document: dict[str, Any], package_version: str, source_date_epoch: int
) -> dict[str, Any]:
    if set(document) != ALLOWED_TOP_LEVEL_KEYS:
        fail("SBOM top-level shape differs from pinned cargo-cyclonedx output")
    if (
        document.get("bomFormat") != EXPECTED_BOM_FORMAT
        or document.get("specVersion") != EXPECTED_SPEC_VERSION
        or document.get("version") != EXPECTED_DOCUMENT_VERSION
    ):
        fail("SBOM format, specification, or document version is unexpected")
    if "serialNumber" in document:
        fail("reproducible SBOM must omit a random serial number")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != EXPECTED_METADATA_KEYS:
        fail("SBOM metadata differs from pinned cargo-cyclonedx output")
    if metadata.get("timestamp") != expected_timestamp(source_date_epoch):
        fail("SBOM timestamp does not match the exact commit epoch")
    if metadata.get("tools") != EXPECTED_TOOL:
        fail("SBOM generator identity is not pinned cargo-cyclonedx 0.5.9")
    if metadata.get("properties") != EXPECTED_TARGET_PROPERTY:
        fail("SBOM target identity is not x86_64-pc-windows-msvc")
    root = metadata.get("component")
    if not isinstance(root, dict) or set(root) != EXPECTED_ROOT_COMPONENT_KEYS:
        fail("SBOM root component shape differs from the reviewed profile")
    if (
        root.get("type") != "application"
        or root.get("name") != EXPECTED_PACKAGE_NAME
        or root.get("version") != package_version
        or root.get("description") != "A native VDJ instrument for live visual performance."
        or root.get("scope") != "required"
        or root.get("licenses") != [{"expression": "GPL-3.0-or-later"}]
        or root.get("purl")
        != f"pkg:cargo/{EXPECTED_PACKAGE_NAME}@{package_version}?download_url=file://."
        or root.get("externalReferences")
        != [{"type": "vcs", "url": EXPECTED_REPOSITORY_URL}]
    ):
        fail("SBOM root package or repository identity is unexpected")
    return root


def validate_component_references(
    document: dict[str, Any], package_version: str, source_base: str
) -> tuple[set[str], str, str]:
    root, targets, top, locations = component_layout(document)
    root_reference = f"{source_base}#{EXPECTED_PACKAGE_NAME}@{package_version}"
    if root.get("bom-ref") != root_reference:
        fail("SBOM root bom-ref is not bound to the exact source identity")
    target_references: list[str] = []
    for index, target in enumerate(targets):
        expected = f"{root_reference} bin-target-{index}"
        if (
            target.get("bom-ref") != expected
            or target.get("version") != package_version
            or not isinstance(target.get("name"), str)
        ):
            fail(f"SBOM binary target identity is unexpected: {index}")
        target_references.append(expected)

    vendor_reference = (
        f"{source_base}/third_party/wgpu-hal-29.0.3#wgpu-hal@29.0.3"
    )
    vendor_count = 0
    registry_count = 0
    git_count = 0
    for component in top:
        reference = component.get("bom-ref")
        if not isinstance(reference, str) or not reference:
            fail(f"SBOM component lacks bom-ref: {_json_path(locations[id(component)])}")
        if reference == vendor_reference:
            vendor_count += 1
            if component.get("name") != "wgpu-hal" or component.get("version") != "29.0.3":
                fail("SBOM vendored wgpu-hal identity is unexpected")
        elif REGISTRY_REFERENCE.fullmatch(reference):
            registry_count += 1
        elif GIT_REFERENCE.fullmatch(reference):
            git_count += 1
        else:
            fail(f"SBOM contains an unreviewed component reference: {reference!r}")
    if (
        vendor_count != 1
        or registry_count != EXPECTED_REGISTRY_COMPONENTS
        or git_count != EXPECTED_GIT_COMPONENTS
    ):
        fail("SBOM component source inventory differs from the reviewed profile")

    all_components = [root, *targets, *top]
    references = [component["bom-ref"] for component in all_components]
    if len(references) != 1 + EXPECTED_TARGET_COMPONENTS + EXPECTED_TOP_COMPONENTS:
        fail("SBOM component declaration count is unexpected")
    if len(references) != len(set(references)):
        fail("SBOM component bom-ref values are not globally unique")
    return set(references), root_reference, vendor_reference


def validate_reference_graph(
    document: dict[str, Any], known: set[str], root_reference: str
) -> dict[str, tuple[str, ...]]:
    _, _, top, _ = component_layout(document)
    dependencies = document.get("dependencies")
    if not isinstance(dependencies, list) or len(dependencies) != EXPECTED_DEPENDENCY_ROWS:
        fail("SBOM dependency row count differs from the reviewed profile")
    expected_rows = {root_reference, *(component["bom-ref"] for component in top)}
    observed: dict[str, tuple[str, ...]] = {}
    edge_count = 0
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, dict) or set(dependency) - {"ref", "dependsOn"}:
            fail(f"SBOM dependency row shape is unexpected: dependencies[{index}]")
        reference = dependency.get("ref")
        depends_on = dependency.get("dependsOn", [])
        if not isinstance(reference, str) or reference not in known:
            fail(f"SBOM dependency ref has no component: dependencies[{index}]")
        if reference in observed:
            fail("SBOM dependency ref rows are duplicated")
        if not isinstance(depends_on, list) or any(
            not isinstance(value, str) or value not in known for value in depends_on
        ):
            fail(f"SBOM dependency edge has no component: dependencies[{index}]")
        if len(depends_on) != len(set(depends_on)):
            fail(f"SBOM dependency edges are duplicated: dependencies[{index}]")
        if reference in depends_on:
            fail(f"SBOM dependency row contains a self-edge: dependencies[{index}]")
        observed[reference] = tuple(depends_on)
        edge_count += len(depends_on)
    if set(observed) != expected_rows:
        fail("SBOM dependency rows do not exactly cover root and top-level components")
    if edge_count != EXPECTED_DEPENDENCY_EDGES:
        fail("SBOM dependency edge count differs from the reviewed profile")
    if len(observed[root_reference]) != EXPECTED_ROOT_EDGES:
        fail("SBOM root dependency edge count differs from the reviewed profile")
    return observed


def validate_path_policy(
    document: dict[str, Any],
    reference_paths: set[PathKey],
    source_spellings: tuple[str, ...],
    source_base: str,
    normalized: bool,
) -> None:
    folded_spellings = tuple(
        spelling.casefold() for spelling in source_spellings if spelling
    )
    canonical = source_base.casefold()
    _, _, top_components, _ = component_layout(document)
    reviewed_relative_purl_paths = {
        ("components", index, "purl")
        for index, component in enumerate(top_components)
        if component.get("bom-ref")
        == f"{source_base}/third_party/wgpu-hal-29.0.3#wgpu-hal@29.0.3"
        and component.get("purl") == EXPECTED_VENDOR_PURL
    }
    for path, value in iter_string_paths(document):
        forms = tuple(decoded_forms(value))
        folded_forms = tuple(form.casefold() for form in forms)
        if path in reference_paths:
            raw_local_reference = not normalized and value.startswith(source_base)
            if normalized and any("path+file:" in form for form in folded_forms):
                fail(f"normalized SBOM retains path+file at {_json_path(path)}")
            if not raw_local_reference and looks_like_absolute_builder_path(value):
                fail(f"SBOM reference embeds an absolute builder path: {_json_path(path)}")
            continue
        if any(canonical in form for form in folded_forms):
            fail(f"SBOM source identity occurs outside a reference: {_json_path(path)}")
        if any(
            spelling in form
            for spelling in folded_spellings
            for form in folded_forms
        ):
            fail(f"SBOM embeds a builder-specific source path: {_json_path(path)}")
        if any("path+file:" in form for form in folded_forms):
            fail(f"SBOM embeds a local path URI outside a reference: {_json_path(path)}")
        if path not in reviewed_relative_purl_paths and looks_like_absolute_builder_path(value):
            fail(f"SBOM embeds an absolute builder path: {_json_path(path)}")


def semantic_profile_digest(document: dict[str, Any], source_base: str) -> str:
    profile = copy.deepcopy(document)
    for container, key, _ in reference_slots(profile):
        value = container[key]
        if isinstance(value, str) and value.startswith(source_base):
            container[key] = SEMANTIC_SOURCE_PLACEHOLDER + value[len(source_base) :]
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("timestamp"), str):
        fail("SBOM semantic profile has no timestamp")
    metadata["timestamp"] = SEMANTIC_TIMESTAMP_PLACEHOLDER
    try:
        payload = json.dumps(
            profile,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        fail(f"cannot encode SBOM semantic profile: {error}")
    return hashlib.sha256(payload).hexdigest()


def validate_profile(
    document: dict[str, Any],
    package_version: str,
    source_base: str,
    source_date_epoch: int,
    *,
    normalized: bool,
    source_spellings: tuple[str, ...] = (),
    expected_semantic_sha256: str = EXPECTED_SEMANTIC_PROFILE_SHA256,
) -> dict[str, Any]:
    if not isinstance(package_version, str) or VERSION.fullmatch(package_version) is None:
        fail("SBOM package version is malformed")
    validate_header(document, package_version, source_date_epoch)
    known, root_reference, vendor_reference = validate_component_references(
        document, package_version, source_base
    )
    slots = reference_slots(document)
    allowed_reference_paths = {path for _, _, path in slots}
    _, _, _, component_locations = component_layout(document)
    allowed_bom_ref_paths = {
        (*location, "bom-ref") for location in component_locations.values()
    }
    for path, value in iter_dict_paths(document):
        if "bom-ref" in value and (*path, "bom-ref") not in allowed_bom_ref_paths:
            fail(f"SBOM contains an unsupported bom-ref: {_json_path((*path, 'bom-ref'))}")
    graph = validate_reference_graph(document, known, root_reference)
    validate_path_policy(
        document,
        allowed_reference_paths,
        source_spellings,
        source_base,
        normalized,
    )
    local_occurrences = sum(
        1 for container, key, _ in slots if str(container[key]).startswith(source_base)
    )
    if local_occurrences != EXPECTED_REWRITTEN_REFERENCES:
        fail("SBOM source-bound reference occurrence count is unexpected")
    observed_semantic_sha256 = semantic_profile_digest(document, source_base)
    if observed_semantic_sha256 != expected_semantic_sha256:
        fail("SBOM semantic profile differs from the reviewed component and graph topology")
    return {
        "component_count": len(known),
        "dependency_rows": len(graph),
        "dependency_edges": sum(len(edges) for edges in graph.values()),
        "root_reference": root_reference,
        "vendor_reference": vendor_reference,
        "source_bound_references": local_occurrences,
        "semantic_profile_sha256": observed_semantic_sha256,
    }


def _validate_normalized_sbom(
    document: dict[str, Any],
    *,
    package_name: str,
    package_version: str,
    commit: str,
    source_date_epoch: int,
    expected_semantic_sha256: str,
) -> dict[str, Any]:
    if package_name != EXPECTED_PACKAGE_NAME:
        fail("SBOM package name is outside the reviewed release profile")
    canonical = canonical_source_uri(commit)
    summary = validate_profile(
        document,
        package_version,
        canonical,
        source_date_epoch,
        normalized=True,
        expected_semantic_sha256=expected_semantic_sha256,
    )
    return summary


def validate_normalized_sbom(
    document: dict[str, Any],
    *,
    package_name: str,
    package_version: str,
    commit: str,
    source_date_epoch: int,
) -> dict[str, Any]:
    return _validate_normalized_sbom(
        document,
        package_name=package_name,
        package_version=package_version,
        commit=commit,
        source_date_epoch=source_date_epoch,
        expected_semantic_sha256=EXPECTED_SEMANTIC_PROFILE_SHA256,
    )


def graph_rows(document: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    dependencies = document["dependencies"]
    return {
        dependency["ref"]: tuple(dependency.get("dependsOn", []))
        for dependency in dependencies
    }


def diff_paths(left: Any, right: Any, path: PathKey = ()) -> set[PathKey]:
    if type(left) is not type(right):
        return {path}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {path}
        result: set[PathKey] = set()
        for key in left:
            result.update(diff_paths(left[key], right[key], (*path, key)))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return {path}
        result = set()
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result.update(diff_paths(left_item, right_item, (*path, index)))
        return result
    return set() if left == right else {path}


def _normalize_document(
    document: dict[str, Any],
    source_root: Path,
    source_date_epoch: int,
    commit: str,
    expected_semantic_sha256: str,
) -> tuple[dict[str, Any], int]:
    package_name, package_version = manifest_identity(source_root)
    raw_source = source_uri(source_root)
    source_spellings = (
        str(source_root),
        source_root.as_posix(),
        source_root.as_uri(),
        raw_source,
    )
    validate_profile(
        document,
        package_version,
        raw_source,
        source_date_epoch,
        normalized=False,
        source_spellings=source_spellings,
        expected_semantic_sha256=expected_semantic_sha256,
    )
    canonical = canonical_source_uri(commit)
    root, targets, top, _ = component_layout(document)
    declarations = [root, *targets, *top]
    mapping = {
        component["bom-ref"]: canonical + component["bom-ref"][len(raw_source) :]
        for component in declarations
        if component["bom-ref"].startswith(raw_source)
    }
    if len(mapping) != EXPECTED_LOCAL_DECLARATIONS:
        fail("SBOM does not contain the exact local declaration inventory")
    if len(mapping) != len(set(mapping.values())):
        fail("SBOM local reference normalization is not injective")
    nonlocal_references = {
        component["bom-ref"]
        for component in declarations
        if component["bom-ref"] not in mapping
    }
    if set(mapping.values()) & nonlocal_references:
        fail("SBOM canonical references collide with existing component identities")

    normalized = copy.deepcopy(document)
    changed_paths: set[PathKey] = set()
    rewritten = 0
    for container, key, path in reference_slots(normalized):
        value = container[key]
        if value in mapping:
            container[key] = mapping[value]
            changed_paths.add(path)
            rewritten += 1
        elif isinstance(value, str) and value.casefold().startswith(raw_source.casefold()):
            fail(f"SBOM source reference is not an exact declared mapping: {_json_path(path)}")
    if rewritten != EXPECTED_REWRITTEN_REFERENCES:
        fail("SBOM did not rewrite the exact reviewed reference occurrences")
    observed_changes = diff_paths(document, normalized)
    if observed_changes != changed_paths:
        fail("SBOM normalization changed data outside mapped reference slots")

    before_graph = graph_rows(document)
    after_graph = graph_rows(normalized)
    expected_graph = {
        mapping.get(reference, reference): tuple(mapping.get(edge, edge) for edge in edges)
        for reference, edges in before_graph.items()
    }
    if after_graph != expected_graph:
        fail("SBOM dependency graph changed outside the injective reference map")
    _validate_normalized_sbom(
        normalized,
        package_name=package_name,
        package_version=package_version,
        commit=commit,
        source_date_epoch=source_date_epoch,
        expected_semantic_sha256=expected_semantic_sha256,
    )
    for path, value in iter_string_paths(normalized):
        folded = value.casefold()
        if any(spelling.casefold() in folded for spelling in source_spellings):
            fail(f"normalized SBOM retains its builder source root: {_json_path(path)}")
    return normalized, rewritten


def normalize_document(
    document: dict[str, Any], source_root: Path, source_date_epoch: int, commit: str
) -> tuple[dict[str, Any], int]:
    return _normalize_document(
        document,
        source_root,
        source_date_epoch,
        commit,
        expected_semantic_sha256=EXPECTED_SEMANTIC_PROFILE_SHA256,
    )


def encoded_document(document: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        fail(f"cannot serialize canonical SBOM: {error}")


def write_new_file(path: Path, payload: bytes) -> None:
    require_unlinked_ancestry(path.parent, "SBOM output")
    require_ordinary_directory(path.parent, "SBOM output parent")
    if path.exists() or path.is_symlink():
        fail("normalized SBOM output must be absent")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        fail(f"normalized SBOM output must be create-only: {error}")
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    require_ordinary_file(path, "normalized SBOM output")


def normalize_file(
    input_path: Path,
    source_root: Path,
    output_path: Path,
    source_date_epoch: int,
    commit: str,
) -> dict[str, Any]:
    lexical_source = Path(os.path.abspath(source_root))
    lexical_input = Path(os.path.abspath(input_path))
    lexical_output = Path(os.path.abspath(output_path))
    require_unlinked_ancestry(lexical_source, "SBOM source root")
    require_unlinked_ancestry(lexical_input, "raw SBOM input")
    require_unlinked_ancestry(lexical_output.parent, "normalized SBOM output")
    require_ordinary_directory(lexical_source, "SBOM source root")
    require_ordinary_file(lexical_input, "raw SBOM input")
    require_ordinary_directory(lexical_output.parent, "normalized SBOM output parent")
    try:
        resolved_source = lexical_source.resolve(strict=True)
        resolved_input = lexical_input.resolve(strict=True)
        resolved_output_parent = lexical_output.parent.resolve(strict=True)
    except OSError as error:
        fail(f"SBOM path resolution failed: {error}")
    resolved_output = resolved_output_parent / lexical_output.name
    if (
        resolved_input.parent != resolved_source
        or resolved_input.name != "collide-o-scope.cdx.json"
    ):
        fail("raw SBOM input must be the exact source-root output file")
    if resolved_output.is_relative_to(resolved_source):
        fail("normalized SBOM output must be outside the source checkout")
    if resolved_output.exists() or resolved_output.is_symlink():
        fail("normalized SBOM output must be absent")

    document = read_json(resolved_input)
    normalized, rewritten = normalize_document(
        document, resolved_source, source_date_epoch, commit
    )
    payload = encoded_document(normalized)
    write_new_file(resolved_output, payload)
    observed = resolved_output.read_bytes()
    if observed != payload:
        fail("normalized SBOM output changed after create-only publication")
    return {
        "schema_version": 1,
        "input_name": resolved_input.name,
        "output_name": resolved_output.name,
        "commit": commit,
        "rewritten_local_references": rewritten,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_fixture_source(source_root: Path, version: str) -> None:
    source_root.mkdir(parents=True)
    (source_root / "Cargo.toml").write_text(
        "[package]\n"
        f'name = "{EXPECTED_PACKAGE_NAME}"\n'
        f'version = "{version}"\n'
        f'repository = "{EXPECTED_REPOSITORY_URL}"\n',
        encoding="utf-8",
        newline="\n",
    )
    (source_root / "Cargo.lock").write_text(
        "version = 4\n\n"
        "[[package]]\n"
        f'name = "{EXPECTED_PACKAGE_NAME}"\n'
        f'version = "{version}"\n',
        encoding="utf-8",
        newline="\n",
    )


def fixture_document(
    source_root: Path, source_date_epoch: int, commit: str, *, normalized: bool = False
) -> dict[str, Any]:
    version = "1.7.2"
    source_base = canonical_source_uri(commit) if normalized else source_uri(source_root)
    root_ref = f"{source_base}#{EXPECTED_PACKAGE_NAME}@{version}"
    target_refs = [f"{root_ref} bin-target-{index}" for index in range(6)]
    vendor_ref = f"{source_base}/third_party/wgpu-hal-29.0.3#wgpu-hal@29.0.3"
    git_ref = (
        "git+https://github.com/ntsc-rs/ntsc-rs?rev="
        "4b79500dfac64efcfb393eebc89f5c75565ee5ae#0.1.2"
    )
    registry_refs = [
        f"registry+https://github.com/rust-lang/crates.io-index#fixture{index:03d}@1.0.0"
        for index in range(EXPECTED_REGISTRY_COMPONENTS)
    ]
    targets = [
        {
            "type": "application" if index else "library",
            "bom-ref": reference,
            "name": f"fixture_target_{index}",
            "version": version,
            "purl": (
                f"pkg:cargo/{EXPECTED_PACKAGE_NAME}@{version}"
                f"?download_url=file://.#src/bin/fixture_{index}.rs"
            ),
        }
        for index, reference in enumerate(target_refs)
    ]
    top = [
        {
            "type": "library",
            "bom-ref": vendor_ref,
            "name": "wgpu-hal",
            "version": "29.0.3",
            "purl": EXPECTED_VENDOR_PURL,
        },
        {
            "type": "library",
            "bom-ref": git_ref,
            "name": "ntsc-rs",
            "version": "0.1.2",
            "purl": "pkg:cargo/ntsc-rs@0.1.2",
        },
        *[
            {
                "type": "library",
                "bom-ref": reference,
                "name": f"fixture{index:03d}",
                "version": "1.0.0",
                "purl": f"pkg:cargo/fixture{index:03d}@1.0.0",
            }
            for index, reference in enumerate(registry_refs)
        ],
    ]
    dependencies: list[dict[str, Any]] = [
        {
            "ref": root_ref,
            "dependsOn": [vendor_ref, target_refs[0], *registry_refs[:34]],
        },
        {
            "ref": vendor_ref,
            "dependsOn": [target_refs[1], registry_refs[0], registry_refs[1]],
        },
    ]
    remaining_rows = [git_ref, *registry_refs]
    nonlocal_pool = [git_ref, *registry_refs]
    for row_index, reference in enumerate(remaining_rows):
        wanted = 3 if row_index < 108 else 2
        edges: list[str] = []
        cursor = row_index + 1
        while len(edges) < wanted:
            candidate = nonlocal_pool[cursor % len(nonlocal_pool)]
            cursor += 1
            if candidate != reference and candidate not in edges:
                edges.append(candidate)
        dependencies.append({"ref": reference, "dependsOn": edges})
    return {
        "bomFormat": EXPECTED_BOM_FORMAT,
        "specVersion": EXPECTED_SPEC_VERSION,
        "version": EXPECTED_DOCUMENT_VERSION,
        "metadata": {
            "timestamp": expected_timestamp(source_date_epoch),
            "tools": copy.deepcopy(EXPECTED_TOOL),
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": EXPECTED_PACKAGE_NAME,
                "version": version,
                "description": "A native VDJ instrument for live visual performance.",
                "scope": "required",
                "licenses": [{"expression": "GPL-3.0-or-later"}],
                "purl": f"pkg:cargo/{EXPECTED_PACKAGE_NAME}@{version}?download_url=file://.",
                "externalReferences": [
                    {"type": "vcs", "url": EXPECTED_REPOSITORY_URL}
                ],
                "components": targets,
            },
            "properties": copy.deepcopy(EXPECTED_TARGET_PROPERTY),
        },
        "components": top,
        "dependencies": dependencies,
    }


def expect_rejection(action: Callable[[], Any], expected: str = "") -> None:
    try:
        action()
    except (OSError, SbomPolicyError) as error:
        if expected and expected not in str(error):
            fail(f"self-test expected {expected!r}, received {str(error)!r}")
        return
    fail(f"SBOM self-test accepted hostile input requiring {expected or 'rejection'}")


def self_test() -> None:
    epoch = 1_700_000_000
    commit = "1" * 40
    with tempfile.TemporaryDirectory(prefix="collide-sbom-policy-") as temporary:
        root = Path(temporary)
        source_a = root / "a"
        source_b = root / "source-b-with-deliberately-different-path-length"
        write_fixture_source(source_a, "1.7.2")
        write_fixture_source(source_b, "1.7.2")
        raw_a = fixture_document(source_a.resolve(), epoch, commit)
        raw_b = fixture_document(source_b.resolve(), epoch, commit)
        fixture_semantic_sha256 = semantic_profile_digest(
            raw_a, source_uri(source_a.resolve())
        )
        if fixture_semantic_sha256 != semantic_profile_digest(
            raw_b, source_uri(source_b.resolve())
        ):
            fail("fixture semantic profile depends on its source root")

        def normalize_fixture(document: dict[str, Any]):
            return _normalize_document(
                document,
                source_a.resolve(),
                epoch,
                commit,
                expected_semantic_sha256=fixture_semantic_sha256,
            )

        def validate_fixture(document: dict[str, Any], exact_commit: str = commit):
            return _validate_normalized_sbom(
                document,
                package_name=EXPECTED_PACKAGE_NAME,
                package_version="1.7.2",
                commit=exact_commit,
                source_date_epoch=epoch,
                expected_semantic_sha256=fixture_semantic_sha256,
            )

        normalized_a, rewritten_a = _normalize_document(
            raw_a,
            source_a.resolve(),
            epoch,
            commit,
            expected_semantic_sha256=fixture_semantic_sha256,
        )
        normalized_b, rewritten_b = _normalize_document(
            raw_b,
            source_b.resolve(),
            epoch,
            commit,
            expected_semantic_sha256=fixture_semantic_sha256,
        )
        payload_a = encoded_document(normalized_a)
        payload_b = encoded_document(normalized_b)
        if (
            payload_a != payload_b
            or rewritten_a != EXPECTED_REWRITTEN_REFERENCES
            or rewritten_b != EXPECTED_REWRITTEN_REFERENCES
        ):
            fail("real-profile unequal roots did not normalize to identical bytes")
        validate_fixture(normalized_a)

        hostile = copy.deepcopy(raw_a)
        hostile["components"][3]["description"] = source_uri(source_a.resolve())
        expect_rejection(
            lambda: normalize_fixture(hostile),
            "outside a reference",
        )
        hostile = copy.deepcopy(raw_a)
        hostile["components"][3]["externalReferences"] = [
            {"type": "website", "url": source_uri(source_a.resolve())}
        ]
        expect_rejection(
            lambda: normalize_fixture(hostile)
        )
        hostile = copy.deepcopy(raw_a)
        hostile["x-hostile"] = source_uri(source_a.resolve())
        expect_rejection(
            lambda: normalize_fixture(hostile),
            "top-level",
        )
        hostile = copy.deepcopy(raw_a)
        hostile["dependencies"].pop()
        expect_rejection(
            lambda: normalize_fixture(hostile),
            "row count",
        )
        hostile = copy.deepcopy(raw_a)
        hostile["components"][2]["bom-ref"] = hostile["components"][1]["bom-ref"]
        expect_rejection(
            lambda: normalize_fixture(hostile)
        )
        hostile = copy.deepcopy(raw_a)
        hostile["dependencies"][0]["dependsOn"].append("registry+https://example.invalid#missing@1")
        expect_rejection(
            lambda: normalize_fixture(hostile),
            "no component",
        )
        hostile = copy.deepcopy(raw_a)
        hostile["dependencies"][0]["dependsOn"].append(
            hostile["dependencies"][0]["dependsOn"][0]
        )
        expect_rejection(
            lambda: normalize_fixture(hostile),
            "duplicated",
        )
        hostile = copy.deepcopy(raw_a)
        hostile["services"] = [{"bom-ref": raw_a["metadata"]["component"]["bom-ref"]}]
        expect_rejection(
            lambda: normalize_fixture(hostile),
            "top-level",
        )
        hostile = copy.deepcopy(raw_a)
        hostile["serialNumber"] = "urn:uuid:00000000-0000-0000-0000-000000000000"
        expect_rejection(
            lambda: normalize_fixture(hostile),
            "top-level",
        )
        hostile = copy.deepcopy(raw_a)
        hostile["metadata"]["timestamp"] = expected_timestamp(epoch + 1)
        expect_rejection(
            lambda: normalize_fixture(hostile),
            "timestamp",
        )
        for leaked in (
            r"D:\private\builder",
            "file:///D:/private/builder",
            r"\\server\share\builder",
            "pkg:cargo/hostile@1?download_url=file%3A%2F%2F%2FD%3A%2Fprivate",
            "/home/runner/work/private",
            "/root/private/build",
            "/build/agent/work",
            "/github/workspace/private",
            "file:///home/runner/private",
            "file:///tmp/private",
            "pkg:cargo/hostile@1?download_url=file%252525253A%252525252F%252525252F%252525252FD%252525253A%252525252Fprivate",
        ):
            hostile = copy.deepcopy(raw_a)
            hostile["components"][3]["purl"] = leaked
            expect_rejection(
                lambda hostile=hostile: normalize_fixture(hostile),
                "builder path",
            )

        hostile = copy.deepcopy(raw_a)
        old_reference = hostile["components"][3]["bom-ref"]
        hostile_reference = (
            "registry+https://github.com/rust-lang/crates.io-index#"
            "invented@C:\\private\\builder"
        )
        for container, key, _ in reference_slots(hostile):
            if container[key] == old_reference:
                container[key] = hostile_reference
        expect_rejection(lambda: normalize_fixture(hostile), "reference embeds")

        canonical_hostile = copy.deepcopy(normalized_a)
        canonical_hostile["components"][3]["description"] = canonical_source_uri(commit)
        expect_rejection(
            lambda: validate_fixture(canonical_hostile),
            "outside a reference",
        )
        expect_rejection(
            lambda: validate_fixture(normalized_a, "2" * 40),
            "root bom-ref",
        )
        canonical_hostile = copy.deepcopy(normalized_a)
        canonical_hostile["components"][2]["bom-ref"] = "urn:cdx:hostile"
        expect_rejection(
            lambda: validate_fixture(canonical_hostile),
            "unreviewed",
        )
        canonical_hostile = copy.deepcopy(normalized_a)
        graph_row = canonical_hostile["dependencies"][2]
        replacement_edge = canonical_hostile["components"][20]["bom-ref"]
        if replacement_edge in graph_row.get("dependsOn", []) or replacement_edge == graph_row["ref"]:
            replacement_edge = canonical_hostile["components"][21]["bom-ref"]
        graph_row["dependsOn"][0] = replacement_edge
        expect_rejection(
            lambda: validate_fixture(canonical_hostile),
            "semantic profile",
        )
        canonical_hostile = copy.deepcopy(normalized_a)
        canonical_hostile["components"][20]["name"] = "substituted-component"
        expect_rejection(
            lambda: validate_fixture(canonical_hostile),
            "semantic profile",
        )
        canonical_hostile = copy.deepcopy(normalized_a)
        canonical_hostile["metadata"]["component"]["components"][0]["purl"] = (
            "pkg:cargo/collide-o-scope@1.7.2?download_url=file://../../private"
        )
        expect_rejection(
            lambda: validate_fixture(canonical_hostile),
            "builder path",
        )
        expect_rejection(
            lambda: normalize_fixture(normalized_a),
            "root bom-ref",
        )

        for encoded, expected in (
            (b'{"a":1,"a":2}', "duplicate"),
            (b'{"value":NaN}', "non-finite"),
            (b'{"value":1e999}', "non-finite"),
            (b"[]", "root"),
            (b"\xef\xbb\xbf{}", "byte-order"),
            (b" " * (MAX_SBOM_BYTES + 1), "bounded"),
        ):
            expect_rejection(lambda encoded=encoded: parse_json_bytes(encoded), expected)

        output = root / "normalized.json"
        write_new_file(output, payload_a)
        expect_rejection(lambda: write_new_file(output, payload_a), "absent")
        link = root / "linked-input.json"
        try:
            link.symlink_to(output)
        except OSError:
            pass
        else:
            expect_rejection(lambda: read_json(link), "link")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    normalize = commands.add_parser("normalize")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--source-root", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--source-date-epoch", type=int, required=True)
    normalize.add_argument("--commit", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--package-version", required=True)
    validate.add_argument("--source-date-epoch", type=int, required=True)
    validate.add_argument("--commit", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "self-test":
            self_test()
            print("CycloneDX release-profile self-test passed")
            return 0
        if args.command == "normalize":
            receipt = normalize_file(
                args.input,
                args.source_root,
                args.output,
                args.source_date_epoch,
                args.commit,
            )
            print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
            return 0
        document = read_json(args.input)
        summary = validate_normalized_sbom(
            document,
            package_name=EXPECTED_PACKAGE_NAME,
            package_version=args.package_version,
            commit=args.commit,
            source_date_epoch=args.source_date_epoch,
        )
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, SbomPolicyError) as error:
        print(f"CycloneDX SBOM policy failed: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
