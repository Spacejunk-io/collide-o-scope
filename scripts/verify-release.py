#!/usr/bin/env python3
"""Assemble or verify deterministic, attributable release evidence."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from cyclonedx_sbom import (
    EXPECTED_PACKAGE_NAME as SBOM_PACKAGE_NAME,
    SbomPolicyError,
    read_json as read_cyclonedx_json,
    self_test as self_test_cyclonedx_policy,
    validate_normalized_sbom,
)


ROOT = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_VERIFICATION_REPORT_BYTES = 256 * 1024
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
TAG = re.compile(r"^v(\d+\.\d+\.\d+)$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
EXPECTED_CARGO_DENY_VERSION = "cargo-deny 0.20.2"
CANONICAL_VENDOR_SOURCE = "path+third_party/wgpu-hal-29.0.4"
CANONICAL_ROOT_SOURCE = "path+."
RELEASE_REVIEW_PATH = ROOT / "policy" / "windows-release-license-review.toml"
EXPECTED_FFMPEG_WINDOWS_DLL_SHA256 = {
    "avcodec-63.dll": "f958e8ae31ce50b58e228c354411e406cd46c0021a6d250e90cf007fe65740d3",
    "avdevice-63.dll": "cc2de2187efd18aed52d3021d90934337bfe0e8ec60d988797b87ae7664f5ee0",
    "avfilter-12.dll": "8f8e2d63f6658450169d7fe2f5696fa9b01df3c1d3820cf706e142ba80758924",
    "avformat-63.dll": "8c0615789d41737051cf082351d4b9c869dd2f0abac4b792ff838041638752e5",
    "avutil-61.dll": "e289456490e190e0d74aa34980aeaa68903a6656248e2e7ef830e17acd80eb49",
    "swresample-7.dll": "d240955beb927ff2fb46cc4f80f83db20a10a9b9032f4092a10f492836fb0213",
    "swscale-10.dll": "89df1925fc718639cb13e849bc940dca114a10e97d93f8fdfd6c14369941a964",
}
EXPECTED_FFMPEG_WINDOWS_DLLS = tuple(EXPECTED_FFMPEG_WINDOWS_DLL_SHA256)
EXPECTED_FFMPEG_EXTERNAL_LIBRARY_VERSIONS_SHA256 = (
    "a99c7c74f9dc649795b436603585e461315febbe6760d1187653750420d4843c"
)
FFMPEG_REVIEW_ONLY_FIELDS = (
    "archive_size",
    "source_archive_sha256",
    "source_archive_size",
    "source_signature_sha256",
    "source_signature_size",
    "signing_key_sha256",
    "signing_key_size",
    "signing_key_fingerprint",
    "source_tag",
)
WINDOWS_RESERVED_BASENAMES = {
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class ReleaseError(RuntimeError):
    pass


def require_release_absent(
    repository: str,
    tag: str,
    token: str,
    opener=urllib.request.urlopen,
) -> None:
    if (
        REPOSITORY.fullmatch(repository) is None
        or TAG.fullmatch(tag) is None
        or not isinstance(token, str)
        or not 1 <= len(token) <= 4096
    ):
        raise ReleaseError("create-only release preflight identity is malformed")
    owner, name = repository.split("/", 1)
    base_url = (
        "https://api.github.com/repos/"
        f"{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(name, safe='')}/releases"
    )
    observed_ids: set[int] = set()
    observed_tags: set[str] = set()
    for page in range(1, 11):
        query = urllib.parse.urlencode({"per_page": "100", "page": str(page)})
        url = f"{base_url}?{query}"
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "collide-o-scope-create-only-release/1",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with opener(request, timeout=30) as response:
                payload = response.read(1024 * 1024 + 1)
                if len(payload) > 1024 * 1024:
                    raise ReleaseError("release-list preflight response is unbounded")
                link = str(response.headers.get("Link", ""))
        except urllib.error.HTTPError as error:
            raise ReleaseError(
                f"create-only release preflight returned HTTP {error.code}"
            ) from error
        except (OSError, urllib.error.URLError) as error:
            raise ReleaseError(f"create-only release preflight failed: {error}") from error
        try:
            releases = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ReleaseError(f"decode bounded release-list preflight: {error}") from error
        if not isinstance(releases, list) or len(releases) > 100:
            raise ReleaseError("release-list preflight has an unsupported page shape")
        for release in releases:
            release_id = release.get("id") if isinstance(release, dict) else None
            release_tag = release.get("tag_name") if isinstance(release, dict) else None
            release_draft = release.get("draft") if isinstance(release, dict) else None
            if (
                type(release_id) is not int
                or release_id <= 0
                or not isinstance(release_tag, str)
                or not 1 <= len(release_tag) <= 160
                or not isinstance(release_draft, bool)
                or release_id in observed_ids
                or release_tag in observed_tags
            ):
                raise ReleaseError("release-list preflight is malformed or duplicated")
            observed_ids.add(release_id)
            observed_tags.add(release_tag)
            if release_tag == tag:
                state = "draft" if release_draft else "published"
                raise ReleaseError(
                    f"{state} release already exists; refusing every publication mutation"
                )
        relations: dict[str, str] = {}
        if link:
            for component in link.split(","):
                match = re.fullmatch(
                    r'\s*<([^>]+)>;\s*rel="(next|prev|first|last)"\s*',
                    component,
                )
                if match is None or match.group(2) in relations:
                    raise ReleaseError("release-list preflight has malformed pagination")
                relations[match.group(2)] = match.group(1)
        if "next" not in relations:
            return
        if page == 10:
            raise ReleaseError("release-list preflight exceeds ten bounded pages")
        parsed_next = urllib.parse.urlparse(relations["next"])
        if (
            parsed_next.scheme != "https"
            or parsed_next.netloc != "api.github.com"
            or parsed_next.path != urllib.parse.urlparse(base_url).path
            or urllib.parse.parse_qs(parsed_next.query, strict_parsing=True)
            != {"per_page": ["100"], "page": [str(page + 1)]}
            or parsed_next.fragment
        ):
            raise ReleaseError("release-list preflight has a noncanonical next page")
    raise ReleaseError("release-list preflight did not reach a bounded terminal page")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_ffmpeg_windows_dll_names(names: list[str] | tuple[str, ...]) -> None:
    observed = list(names)
    if any(not isinstance(name, str) or not name for name in observed):
        raise ReleaseError("FFmpeg runtime DLL inventory contains a malformed name")
    folded = [name.casefold() for name in observed]
    if len(folded) != len(set(folded)):
        raise ReleaseError("FFmpeg runtime DLL inventory contains duplicate names")
    expected = list(EXPECTED_FFMPEG_WINDOWS_DLLS)
    if sorted(observed) != expected:
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise ReleaseError(
            "FFmpeg runtime DLL inventory differs from the reviewed FFmpeg 9 set: "
            f"missing={missing}, unexpected={unexpected}"
        )


def ffmpeg_windows_dll_evidence(ffmpeg_bin: Path) -> dict[str, str]:
    try:
        observed = sorted(
            path.name
            for path in ffmpeg_bin.iterdir()
            if path.suffix.casefold() == ".dll"
        )
    except OSError as error:
        raise ReleaseError(f"inspect FFmpeg runtime DLL directory: {error}") from error
    validate_ffmpeg_windows_dll_names(observed)
    evidence: dict[str, str] = {}
    for name in EXPECTED_FFMPEG_WINDOWS_DLLS:
        path = ffmpeg_bin / name
        if not path.is_file():
            raise ReleaseError(f"FFmpeg runtime DLL is not a regular file: {name}")
        actual = digest(path)
        expected = EXPECTED_FFMPEG_WINDOWS_DLL_SHA256[name]
        if actual != expected:
            raise ReleaseError(
                f"FFmpeg runtime DLL hash differs from the reviewed distribution: {name}"
            )
        evidence[name] = actual
    return evidence


def shader_bundle_digest(root: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"collide-o-scope shader bundle v1\0")
    for path in sorted((root / "src" / "shaders").glob("*.wgsl")):
        name = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        hasher.update(len(name).to_bytes(8, "little"))
        hasher.update(name)
        hasher.update(len(data).to_bytes(8, "little"))
        hasher.update(data)
    return hasher.hexdigest()


def create_source_archive(destination: Path, tag: str, commit: str) -> None:
    prefix = f"collide-o-scope-{tag}-source/"
    archive_environment = os.environ.copy()
    archive_environment["TZ"] = "UTC"
    try:
        subprocess.run(
            [
                "git", "-C", str(ROOT), "archive", "--format=zip",
                f"--prefix={prefix}", f"--output={destination}", commit,
            ],
            check=True,
            capture_output=True,
            env=archive_environment,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleaseError(f"create deterministic source archive: {error}") from error


def git_text(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleaseError(f"inspect release Git source: {error}") from error
    return completed.stdout.strip()


def local_ref_sha(reference: str) -> str | None:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "--verify",
                "--quiet",
                reference,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleaseError(f"inspect release Git reference: {error}") from error
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 1 and not stdout and not stderr:
        return None
    if completed.returncode != 0 or stderr or GIT_SHA.fullmatch(stdout) is None:
        raise ReleaseError("release Git reference query returned an unsupported result")
    return stdout.lower()


def parse_annotated_tag_rows(rows: list[str], tag: str) -> tuple[str, str]:
    tag_ref = f"refs/tags/{tag}"
    peeled_ref = f"{tag_ref}^{{}}"
    observed: dict[str, str] = {}
    for row in rows:
        match = re.fullmatch(r"([0-9a-f]{40})\t([^\t\r\n]+)", row)
        if match is None or match.group(2) not in {tag_ref, peeled_ref}:
            raise ReleaseError("remote tag query returned an unsupported row")
        if match.group(2) in observed:
            raise ReleaseError("remote tag query returned a duplicate row")
        observed[match.group(2)] = match.group(1)
    if set(observed) != {tag_ref, peeled_ref}:
        raise ReleaseError("annotated release tag requires exact remote tag and peeled rows")
    return observed[tag_ref], observed[peeled_ref]


def annotated_tag(args: argparse.Namespace) -> dict:
    if TAG.fullmatch(args.tag) is None or GIT_SHA.fullmatch(args.commit) is None:
        raise ReleaseError("annotated tag verification requires a release tag and exact commit")
    tag_ref = f"refs/tags/{args.tag}"
    if git_text("cat-file", "-t", tag_ref) != "tag":
        raise ReleaseError("release ref is not an annotated tag object")
    local_tag_object = git_text("rev-parse", tag_ref).lower()
    local_commit = git_text("rev-parse", f"{tag_ref}^{{commit}}").lower()
    if GIT_SHA.fullmatch(local_tag_object) is None or local_commit != args.commit:
        raise ReleaseError("local annotated tag does not peel to the exact release commit")
    try:
        completed = subprocess.run(
            [
                "git", "-C", str(ROOT), "ls-remote", "--tags", args.remote,
                tag_ref, f"{tag_ref}^{{}}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleaseError(f"query remote annotated tag: {error}") from error
    remote_tag_object, remote_commit = parse_annotated_tag_rows(
        completed.stdout.splitlines(), args.tag
    )
    if remote_tag_object != local_tag_object or remote_commit != args.commit:
        raise ReleaseError("remote annotated tag object or peeled commit changed")
    if args.tag_object is not None and args.tag_object != local_tag_object:
        raise ReleaseError("annotated tag object changed after initial resolution")
    document = {
        "schema_version": 1,
        "tag": args.tag,
        "tag_object_sha": local_tag_object,
        "peeled_commit_sha": local_commit,
        "remote_tag_row_present": True,
        "remote_peeled_row_present": True,
        "annotated": True,
    }
    if args.output is not None:
        write_new_json(args.output.resolve(), document, 16 * 1024)
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8", newline="\n") as output:
            output.write(f"tag_object={local_tag_object}\n")
            output.write(f"peeled_commit={local_commit}\n")
    print(json.dumps(document, sort_keys=True))
    return document


def read_json(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ReleaseError(f"JSON exceeds {MAX_JSON_BYTES} bytes: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError(f"{path.name} must contain one JSON object")
    return value


def validate_dependency_inventory(document: dict, cargo_lock_sha256: str) -> None:
    if document.get("schema_version") != 1:
        raise ReleaseError("unsupported dependency inventory schema")
    if document.get("generator") != EXPECTED_CARGO_DENY_VERSION:
        raise ReleaseError("dependency inventory was not generated by pinned cargo-deny")
    if document.get("cargo_lock_sha256") != cargo_lock_sha256:
        raise ReleaseError("dependency inventory does not match BuildIdentity Cargo.lock")
    normalization = document.get("normalization", {})
    root_occurrences = normalization.get("root_source_occurrences")
    vendor_occurrences = normalization.get("vendor_source_occurrences")
    if (
        root_occurrences != 1
        or not isinstance(vendor_occurrences, int)
        or vendor_occurrences < 1
        or normalization.get("absolute_path_sources_removed")
        != root_occurrences + vendor_occurrences
        or normalization.get("declared_root_source") != CANONICAL_ROOT_SOURCE
        or normalization.get("declared_vendor_source") != CANONICAL_VENDOR_SOURCE
    ):
        raise ReleaseError("dependency inventory lacks the declared path normalization")
    inventory = document.get("inventory")
    if (
        not isinstance(inventory, dict)
        or not isinstance(inventory.get("licenses"), list)
        or not inventory["licenses"]
        or inventory.get("unlicensed") != []
    ):
        raise ReleaseError("dependency inventory is empty, unlicensed, or malformed")
    serialized = json.dumps(inventory, sort_keys=True)
    if (
        "path+file://" in serialized
        or serialized.count(CANONICAL_ROOT_SOURCE) != root_occurrences
        or serialized.count(CANONICAL_VENDOR_SOURCE) != vendor_occurrences
    ):
        raise ReleaseError("dependency inventory leaks a path or misstates the vendor source")


def command_text(program: Path, arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            [str(program), *arguments],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleaseError(f"inspect {program.name}: {error}") from error
    combined = completed.stdout + completed.stderr
    if len(combined) > 256 * 1024:
        raise ReleaseError(f"{program.name} identity output is unexpectedly large")
    try:
        text = combined.decode("utf-8").replace("\r\n", "\n").strip() + "\n"
    except UnicodeError as error:
        raise ReleaseError(f"{program.name} identity is not UTF-8: {error}") from error
    if any(character not in "\n\t" and not " " <= character <= "~" for character in text):
        raise ReleaseError(f"{program.name} identity contains unsafe control text")
    return text


def ffmpeg_distribution_evidence(
    ffmpeg_bin: Path,
    version: str,
    source_commit: str,
    *,
    license_path: Path | None = None,
    readme_path: Path | None = None,
) -> tuple[str, dict]:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ReleaseError("FFmpeg version must be semantic numeric text")
    executable = ffmpeg_bin / "ffmpeg.exe"
    buildconf = command_text(executable, ["-hide_banner", "-buildconf"])
    required_flags = ("--enable-gpl", "--enable-version3", "--enable-shared")
    if any(flag not in buildconf.split() for flag in required_flags):
        raise ReleaseError("FFmpeg build configuration lacks a required GPL/shared flag")
    if "--enable-nonfree" in buildconf.split():
        raise ReleaseError("the non-redistributable FFmpeg configuration is forbidden")
    license_text = command_text(executable, ["-hide_banner", "-L"])
    if (
        "GNU General Public License" not in license_text
        or "either version 3" not in license_text
        or "any later version" not in license_text
    ):
        raise ReleaseError("FFmpeg runtime does not report GPL version 3-or-later")
    root = ffmpeg_bin.parent
    if license_path is None:
        license_path = root / "LICENSE"
    if readme_path is None:
        readme_path = root / "README.txt"
    if not license_path.is_file() or not readme_path.is_file():
        raise ReleaseError("FFmpeg distribution license or README is missing")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ReleaseError("FFmpeg source commit must be a full lowercase Git SHA")
    readme = readme_path.read_text(encoding="utf-8")
    source_match = re.search(
        r"Source Code: https://github\.com/FFmpeg/FFmpeg/commit/([0-9a-f]{7,40})",
        readme,
    )
    if (
        f"Version: {version}-full_build-www.gyan.dev" not in readme
        or "License: GPL v3" not in readme
        or source_match is None
        or not source_commit.startswith(source_match.group(1))
    ):
        raise ReleaseError("FFmpeg README lacks the pinned build, license, or source identity")
    marker = "release-full external libraries' versions:"
    readme_lines = readme.splitlines()
    try:
        marker_index = next(
            index for index, line in enumerate(readme_lines) if line.strip() == marker
        )
    except StopIteration as error:
        raise ReleaseError("FFmpeg README lacks its external-library version inventory") from error
    external_library_versions: list[str] = []
    for line in readme_lines[marker_index + 1 :]:
        value = line.strip()
        if not value:
            if external_library_versions:
                break
            continue
        external_library_versions.append(value)
    external_inventory_text = "\n".join(external_library_versions) + "\n"
    external_inventory_sha256 = hashlib.sha256(
        external_inventory_text.encode("utf-8")
    ).hexdigest()
    if external_inventory_sha256 != EXPECTED_FFMPEG_EXTERNAL_LIBRARY_VERSIONS_SHA256:
        raise ReleaseError("FFmpeg external-library version inventory is not the reviewed set")
    windows_runtime_dll_sha256 = ffmpeg_windows_dll_evidence(ffmpeg_bin)
    return buildconf, {
        "version": version,
        "distribution": "Gyan full shared Windows build",
        "runtime_license": "GPL-3.0-or-later",
        "buildconf_sha256": hashlib.sha256(buildconf.encode("utf-8")).hexdigest(),
        "runtime_license_text_sha256": hashlib.sha256(license_text.encode("utf-8")).hexdigest(),
        "distribution_license_sha256": digest(license_path),
        "distribution_readme_sha256": digest(readme_path),
        "external_library_versions": external_library_versions,
        "external_library_versions_sha256": external_inventory_sha256,
        "windows_runtime_dlls": list(EXPECTED_FFMPEG_WINDOWS_DLLS),
        "windows_runtime_dll_sha256": windows_runtime_dll_sha256,
        "source_commit": source_commit,
        "source_url": f"https://github.com/FFmpeg/FFmpeg/commit/{source_commit}",
        "source_identity_recorded_in": "FFMPEG-README.txt",
    }


def pinned_ffmpeg_distribution() -> tuple[str, str, str]:
    workflow = (ROOT / ".github" / "workflows" / "release-trust.yml").read_text(
        encoding="utf-8"
    )
    version = re.search(r"^\s*FFMPEG_VERSION:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*$", workflow, re.MULTILINE)
    archive = re.search(r"^\s*FFMPEG_WINDOWS_SHA256:\s*([0-9a-f]{64})\s*$", workflow, re.MULTILINE)
    source = re.search(r"^\s*FFMPEG_SOURCE_COMMIT:\s*([0-9a-f]{40})\s*$", workflow, re.MULTILINE)
    if version is None or archive is None or source is None:
        raise ReleaseError("release workflow lacks a typed FFmpeg distribution pin")
    return version.group(1), archive.group(1), source.group(1)


def checked_release_review() -> dict:
    try:
        with RELEASE_REVIEW_PATH.open("rb") as source:
            review = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError(f"read checked-in release review: {error}") from error
    if review.get("schema_version") != 1:
        raise ReleaseError("unsupported checked-in release review schema")
    if (
        not isinstance(review.get("review_id"), str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", review["review_id"])
        or not isinstance(review.get("review_owner"), str)
        or not review["review_owner"].strip()
        or not isinstance(review.get("reviewed"), date)
        or review.get("legal_conclusion") is not False
        or review.get("decision") != "admit_only_the_exact_reviewed_distribution"
        or "Mechanical review" not in str(review.get("scope"))
    ):
        raise ReleaseError("checked-in release review lacks its typed decision boundary")
    ffmpeg = review.get("ffmpeg")
    version, archive_sha256, source_commit = pinned_ffmpeg_distribution()
    required_ffmpeg = {
        "version": version,
        "distribution": "Gyan full shared Windows build",
        "archive_sha256": archive_sha256,
        "source_tag": f"n{version}",
        "source_commit": source_commit,
        "source_url": f"https://github.com/FFmpeg/FFmpeg/commit/{source_commit}",
        "runtime_license": "GPL-3.0-or-later",
    }
    if not isinstance(ffmpeg, dict) or any(
        ffmpeg.get(key) != value for key, value in required_ffmpeg.items()
    ):
        raise ReleaseError("checked-in release review disagrees with FFmpeg workflow pins")
    for field in (
        "source_archive_sha256",
        "source_signature_sha256",
        "signing_key_sha256",
        "buildconf_sha256",
        "runtime_license_text_sha256",
        "distribution_license_sha256",
        "distribution_readme_sha256",
        "external_library_versions_sha256",
    ):
        if not isinstance(ffmpeg.get(field), str) or SHA256.fullmatch(ffmpeg[field]) is None:
            raise ReleaseError(f"checked-in release review FFmpeg field {field} is invalid")
    for field, expected in (
        ("archive_size", 97_261_187),
        ("source_archive_size", 12_036_420),
        ("source_signature_size", 520),
        ("signing_key_size", 1_709),
    ):
        if ffmpeg.get(field) != expected:
            raise ReleaseError(f"checked-in release review FFmpeg field {field} is invalid")
    if ffmpeg.get("signing_key_fingerprint") != "FCF986EA15E6E293A5644F10B4322F04D67658D8":
        raise ReleaseError("checked-in release review has the wrong FFmpeg signing key")
    if ffmpeg.get("windows_runtime_dlls") != list(EXPECTED_FFMPEG_WINDOWS_DLLS):
        raise ReleaseError("checked-in release review has the wrong FFmpeg DLL inventory")
    if ffmpeg.get("windows_runtime_dll_sha256") != EXPECTED_FFMPEG_WINDOWS_DLL_SHA256:
        raise ReleaseError("checked-in release review has the wrong FFmpeg DLL hashes")
    external_versions = ffmpeg.get("external_library_versions")
    if (
        not isinstance(external_versions, list)
        or any(not isinstance(value, str) or not value for value in external_versions)
        or len(external_versions) != 82
        or hashlib.sha256(("\n".join(external_versions) + "\n").encode("utf-8")).hexdigest()
        != EXPECTED_FFMPEG_EXTERNAL_LIBRARY_VERSIONS_SHA256
        or ffmpeg.get("external_library_versions_sha256")
        != EXPECTED_FFMPEG_EXTERNAL_LIBRARY_VERSIONS_SHA256
        or "AMF v1.5.2-2-gc35f613" not in external_versions
        or "ffnvcodec n13.1.15.0-1-geddcea9" not in external_versions
    ):
        raise ReleaseError("checked-in release review has the wrong external-library inventory")
    if (
        ffmpeg.get("required_build_flags")
        != ["--enable-gpl", "--enable-version3", "--enable-shared"]
        or ffmpeg.get("forbidden_build_flags") != ["--enable-nonfree"]
    ):
        raise ReleaseError("checked-in release review has the wrong FFmpeg flag policy")
    notices = review.get("packaging", {}).get("required_package_notices")
    if notices != [
        "LICENSE",
        "COPYRIGHT.md",
        "FFMPEG-BUILDCONF.txt",
        "FFMPEG-README.txt",
        "LICENSES/FFmpeg-GPL-3.0-or-later.txt",
    ]:
        raise ReleaseError("checked-in release review has an incomplete notice policy")
    authenticode = review.get("authenticode")
    if (
        not isinstance(authenticode, dict)
        or authenticode.get("status") != "unavailable"
        or "No managed Authenticode" not in str(authenticode.get("reason"))
        or "unsigned" not in str(authenticode.get("artifact_claim"))
        or "Sigstore" not in str(authenticode.get("artifact_claim"))
    ):
        raise ReleaseError("checked-in release review misstates Authenticode availability")
    return review


def pe_has_authenticode(path: Path) -> bool:
    with path.open("rb") as source:
        dos = source.read(64)
        if len(dos) != 64 or dos[:2] != b"MZ":
            raise ReleaseError(f"release executable is not a PE file: {path}")
        pe_offset = struct.unpack_from("<I", dos, 0x3C)[0]
        if pe_offset > 16 * 1024 * 1024:
            raise ReleaseError("PE header offset is outside the bounded prefix")
        source.seek(pe_offset)
        header = source.read(24 + 240)
    if len(header) < 24 + 112 or header[:4] != b"PE\0\0":
        raise ReleaseError("release executable has an invalid PE header")
    optional_size = struct.unpack_from("<H", header, 20)[0]
    optional = header[24:24 + optional_size]
    if len(optional) != optional_size:
        raise ReleaseError("release executable has a truncated PE optional header")
    magic = struct.unpack_from("<H", optional, 0)[0]
    directory_offset = {0x10B: 96, 0x20B: 112}.get(magic)
    if directory_offset is None or len(optional) < directory_offset + 5 * 8:
        raise ReleaseError("release executable has an unsupported PE optional header")
    certificate_offset, certificate_size = struct.unpack_from(
        "<II", optional, directory_offset + 4 * 8
    )
    if (certificate_offset == 0) != (certificate_size == 0):
        raise ReleaseError("PE Authenticode directory is internally inconsistent")
    return certificate_offset != 0


def validate_dependency_review(
    review: dict,
    inventory_path: Path,
    buildconf_path: Path,
    identity: dict,
    tag: str,
    commit: str,
) -> None:
    if review.get("schema_version") != 1:
        raise ReleaseError("unsupported dependency review schema")
    if review.get("tag") != tag or review.get("commit") != commit.lower():
        raise ReleaseError("dependency review does not match the release source")
    checked = checked_release_review()
    if (
        review.get("checked_in_review_id") != checked["review_id"]
        or review.get("checked_in_review_owner") != checked["review_owner"]
        or review.get("checked_in_reviewed_on") != checked["reviewed"].isoformat()
        or review.get("checked_in_review_sha256") != digest(RELEASE_REVIEW_PATH)
        or review.get("build_identity_sha256") != identity["identity_sha256"]
        or review.get("cargo_lock_sha256") != identity["cargo_lock_sha256"]
        or review.get("dependency_inventory_sha256") != digest(inventory_path)
        or review.get("dependency_exception_policy_sha256")
        != digest(ROOT / "policy" / "dependency-exceptions.toml")
    ):
        raise ReleaseError("dependency review is not bound to its release inputs")
    expected_gates = {
        "dependency_policy": "passed",
        "cargo_audit": "passed_with_only_declared_unexpired_exceptions",
        "cargo_deny_advisories_bans_sources": "passed",
        "cargo_deny_licenses": "passed",
        "vendored_wgpu_hal_upstream_plus_declared_patch": "passed_with_seeded_drift_rejection",
    }
    if review.get("technical_gates") != expected_gates:
        raise ReleaseError("dependency review does not record every release gate")
    if review.get("legal_conclusion") is not False or "not legal advice" not in str(review.get("scope")):
        raise ReleaseError("dependency review confuses mechanical status with legal advice")
    expected_notices = checked["packaging"]["required_package_notices"]
    notices = review.get("packaged_notices")
    if notices != expected_notices:
        raise ReleaseError("dependency review has an incomplete notice inventory")
    ffmpeg = review.get("ffmpeg_distribution")
    version, archive_sha256, source_commit = pinned_ffmpeg_distribution()
    if (
        not isinstance(ffmpeg, dict)
        or ffmpeg.get("version") != version
        or ffmpeg.get("archive_sha256") != archive_sha256
        or ffmpeg.get("source_commit") != source_commit
        or ffmpeg.get("source_url")
        != f"https://github.com/FFmpeg/FFmpeg/commit/{source_commit}"
        or ffmpeg.get("runtime_license") != "GPL-3.0-or-later"
        or ffmpeg.get("buildconf_sha256") != digest(buildconf_path)
        or ffmpeg.get("source_identity_recorded_in") != "FFMPEG-README.txt"
    ):
        raise ReleaseError("dependency review has the wrong FFmpeg distribution identity")
    for field in (
        "buildconf_sha256",
        "runtime_license_text_sha256",
        "distribution_license_sha256",
        "distribution_readme_sha256",
    ):
        if not isinstance(ffmpeg.get(field), str) or SHA256.fullmatch(ffmpeg[field]) is None:
            raise ReleaseError(f"dependency review FFmpeg field {field} is not SHA-256")
        if ffmpeg[field] != checked["ffmpeg"][field]:
            raise ReleaseError(f"dependency review FFmpeg field {field} was not reviewed")
    for field in FFMPEG_REVIEW_ONLY_FIELDS + (
        "external_library_versions",
        "external_library_versions_sha256",
        "windows_runtime_dlls",
        "windows_runtime_dll_sha256",
    ):
        if ffmpeg.get(field) != checked["ffmpeg"].get(field):
            raise ReleaseError(f"dependency review FFmpeg field {field} was not reviewed")
    if review.get("authenticode") != checked["authenticode"]:
        raise ReleaseError("dependency review misstates the Authenticode stop disposition")


def identity_payload(identity: dict) -> bytes:
    keys = [
        "package_name", "version", "git_sha", "git_dirty", "profile", "target",
        "enabled_features", "rustc_vv", "cargo_version", "linker_identity",
        "sdk_identity", "ffmpeg_libraries", "ffmpeg_binary_version",
        "ffmpeg_binary_sha256", "ffprobe_binary_version", "ffprobe_binary_sha256",
        "shader_bundle_sha256", "cargo_lock_sha256", "published_artifact",
    ]
    missing = [key for key in keys if key not in identity]
    if missing:
        raise ReleaseError(f"BuildIdentity is missing fields: {missing}")
    lines = ["domain=collide-o-scope build identity v1"]
    for key in keys:
        value = identity[key]
        if isinstance(value, bool):
            value = "true" if value else "false"
        lines.append(f"{key}={value}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def validate_resolved_tag_state(
    tag_state: str,
    local_tag_sha: str | None,
    local_tag_type: str | None,
    peeled_commit: str | None,
    commit: str,
) -> None:
    if tag_state not in {"absent", "annotated"}:
        raise ReleaseError("release tag state is not absent or annotated")
    if tag_state == "absent":
        if any(value is not None for value in (local_tag_sha, local_tag_type, peeled_commit)):
            raise ReleaseError("pre-tag qualification requires the local release tag to be absent")
        return
    if (
        local_tag_sha is None
        or GIT_SHA.fullmatch(local_tag_sha) is None
        or local_tag_type != "tag"
        or peeled_commit != commit.lower()
    ):
        raise ReleaseError("tagged release verification requires the exact annotated tag")


def validate_tag_binding(tag: str, commit: str, tag_state: str) -> None:
    if tag_state not in {"absent", "annotated"}:
        validate_resolved_tag_state(tag_state, None, None, None, commit)
    tag_ref = f"refs/tags/{tag}"
    local_tag_sha = local_ref_sha(tag_ref)
    if tag_state == "absent":
        validate_resolved_tag_state(tag_state, local_tag_sha, None, None, commit)
        return
    if local_tag_sha is None:
        validate_resolved_tag_state(tag_state, None, None, None, commit)
    validate_resolved_tag_state(
        tag_state,
        local_tag_sha,
        git_text("cat-file", "-t", tag_ref),
        git_text("rev-parse", f"{tag_ref}^{{commit}}").lower(),
        commit,
    )


def validate_identity(identity: dict, tag: str, commit: str, tag_state: str) -> None:
    match = TAG.fullmatch(tag)
    if match is None:
        raise ReleaseError(f"release tag is not vMAJOR.MINOR.PATCH: {tag!r}")
    if identity.get("schema_version") != 1:
        raise ReleaseError("unsupported BuildIdentity schema")
    if identity.get("version") != match.group(1):
        raise ReleaseError("Cargo version, BuildIdentity, and release tag disagree")
    if identity.get("git_sha") != commit.lower() or not re.fullmatch(r"[0-9a-f]{40}", commit.lower()):
        raise ReleaseError("BuildIdentity Git SHA does not match the release commit")
    validate_tag_binding(tag, commit, tag_state)
    if identity.get("git_dirty") is not False or identity.get("published_artifact") is not True:
        raise ReleaseError("a dirty/local BuildIdentity cannot carry a published release badge")
    if identity.get("profile") != "release" or identity.get("target") != "x86_64-pc-windows-msvc":
        raise ReleaseError("release BuildIdentity has the wrong profile or target")
    with (ROOT / "rust-toolchain.toml").open("rb") as source:
        toolchain = tomllib.load(source).get("toolchain", {}).get("channel")
    rustc_lines = identity.get("rustc_vv", "").splitlines()
    if (
        not isinstance(toolchain, str)
        or len(rustc_lines) != 7
        or rustc_lines[0] != f"rustc {toolchain}"
        or identity.get("cargo_version") != f"cargo {toolchain}"
    ):
        raise ReleaseError("BuildIdentity does not match the pinned Rust toolchain")
    if (
        not str(identity.get("linker_identity", "")).lower().startswith("link.exe;microsoft ")
        or "windows-sdk:" not in str(identity.get("sdk_identity", ""))
        or "msvc-tools:" not in str(identity.get("sdk_identity", ""))
    ):
        raise ReleaseError("BuildIdentity lacks the Windows linker or SDK identity")
    ffmpeg_version, _, _ = pinned_ffmpeg_distribution()
    ffmpeg_libraries = str(identity.get("ffmpeg_libraries", ""))
    expected_ffmpeg_libraries = ",".join(
        sorted((*EXPECTED_FFMPEG_WINDOWS_DLLS, f"ffmpeg={ffmpeg_version}"))
    )
    if ffmpeg_libraries != expected_ffmpeg_libraries:
        raise ReleaseError("BuildIdentity lacks the exact reviewed FFmpeg DLL identity")
    if identity.get("ffmpeg_binary_version") != f"ffmpeg version {ffmpeg_version}":
        raise ReleaseError("BuildIdentity has the wrong FFmpeg binary version")
    if identity.get("ffprobe_binary_version") != f"ffprobe version {ffmpeg_version}":
        raise ReleaseError("BuildIdentity has the wrong ffprobe binary version")
    expected = hashlib.sha256(identity_payload(identity)).hexdigest()
    if identity.get("identity_sha256") != expected:
        raise ReleaseError("BuildIdentity digest is invalid")
    for field in (
        "identity_sha256", "shader_bundle_sha256", "cargo_lock_sha256",
        "ffmpeg_binary_sha256", "ffprobe_binary_sha256",
    ):
        value = identity.get(field)
        if field.startswith("ff") and value == "unreported":
            raise ReleaseError(f"published BuildIdentity has no {field}")
        if not isinstance(value, str) or SHA256.fullmatch(value) is None:
            raise ReleaseError(f"BuildIdentity field {field} is not a SHA-256")
    if identity["cargo_lock_sha256"] != digest(ROOT / "Cargo.lock"):
        raise ReleaseError("BuildIdentity Cargo.lock digest does not match release source")
    if identity["shader_bundle_sha256"] != shader_bundle_digest(ROOT):
        raise ReleaseError("BuildIdentity shader digest does not match release source")


def validate_signature_tag_state(require_signature: bool, tag_state: str) -> None:
    if require_signature and tag_state != "annotated":
        raise ReleaseError("signed release verification requires the annotated tag state")


def executable_identity(executable: Path, ffmpeg_bin: Path) -> dict:
    environment = os.environ.copy()
    environment["PATH"] = str(ffmpeg_bin) + os.pathsep + environment.get("PATH", "")
    try:
        completed = subprocess.run(
            [str(executable), "--version-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
        )
        identity = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ReleaseError(f"read BuildIdentity from {executable}: {error}") from error
    if not isinstance(identity, dict):
        raise ReleaseError("--version-json did not return an object")
    return identity


def zip_entry(archive: zipfile.ZipFile, name: str, data: bytes, executable: bool = False) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def assemble_package(
    destination: Path,
    executable: Path,
    ffmpeg_bin: Path,
    ffmpeg_buildconf: Path,
) -> None:
    payload: dict[str, tuple[Path, bool]] = {
        "collide-o-scope.exe": (executable, True),
    }
    tools = [ffmpeg_bin / "ffmpeg.exe", ffmpeg_bin / "ffprobe.exe"]
    ffmpeg_windows_dll_evidence(ffmpeg_bin)
    libraries = [ffmpeg_bin / name for name in EXPECTED_FFMPEG_WINDOWS_DLLS]
    for path in tools + libraries:
        if not path.is_file():
            raise ReleaseError(f"release runtime file is missing: {path}")
        payload[path.name] = (path, path.suffix.lower() == ".exe")
    payload["LICENSE"] = (ROOT / "LICENSE", False)
    payload["COPYRIGHT.md"] = (ROOT / "COPYRIGHT.md", False)
    payload["FFMPEG-BUILDCONF.txt"] = (ffmpeg_buildconf, False)
    payload["FFMPEG-README.txt"] = (ffmpeg_bin.parent / "README.txt", False)
    payload["LICENSES/FFmpeg-GPL-3.0-or-later.txt"] = (
        ffmpeg_bin.parent / "LICENSE",
        False,
    )
    licenses = ROOT / "LICENSES"
    for path in sorted(licenses.rglob("*")):
        if path.is_file():
            payload[f"LICENSES/{path.relative_to(licenses).as_posix()}"] = (path, False)

    with zipfile.ZipFile(destination, "w") as archive:
        for name in sorted(payload):
            path, is_executable = payload[name]
            zip_entry(archive, name, path.read_bytes(), is_executable)


def validate_sbom(
    sbom: dict, version: str, commit: str, source_date_epoch: int
) -> dict:
    try:
        return validate_normalized_sbom(
            sbom,
            package_name=SBOM_PACKAGE_NAME,
            package_version=version,
            commit=commit.lower(),
            source_date_epoch=source_date_epoch,
        )
    except SbomPolicyError as error:
        raise ReleaseError(f"SBOM release-profile validation failed: {error}") from error


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ReleaseError(f"release output directory must be empty: {output}")
    first_hash = digest(args.executable)
    second_hash = digest(args.second_executable)
    if first_hash != second_hash:
        raise ReleaseError(f"independent executable hashes differ: {first_hash} != {second_hash}")
    identity = executable_identity(args.executable, args.ffmpeg_bin)
    second_identity = executable_identity(args.second_executable, args.ffmpeg_bin)
    if identity != second_identity:
        raise ReleaseError("independent builds expose different BuildIdentity documents")
    validate_identity(identity, args.tag, args.commit, args.tag_state)
    commit_epoch = int(git_text("show", "-s", "--format=%ct", args.commit))
    if args.source_date_epoch != commit_epoch:
        raise ReleaseError("SOURCE_DATE_EPOCH does not match the tagged commit timestamp")
    if SHA256.fullmatch(args.ffmpeg_archive_sha256) is None:
        raise ReleaseError("FFmpeg archive digest is not SHA-256")
    if (
        args.ffmpeg_version,
        args.ffmpeg_archive_sha256,
        args.ffmpeg_source_commit,
    ) != pinned_ffmpeg_distribution():
        raise ReleaseError("FFmpeg arguments do not match the release workflow pins")
    try:
        sbom = read_cyclonedx_json(args.sbom)
    except SbomPolicyError as error:
        raise ReleaseError(f"read strict SBOM JSON: {error}") from error
    validate_sbom(
        sbom,
        identity["version"],
        args.commit,
        args.source_date_epoch,
    )
    dependency_inventory = read_json(args.dependency_inventory)
    validate_dependency_inventory(dependency_inventory, identity["cargo_lock_sha256"])
    checked_review = checked_release_review()
    if pe_has_authenticode(args.executable) or pe_has_authenticode(args.second_executable):
        raise ReleaseError(
            "reproducibility inputs must be unsigned while Authenticode is unavailable"
        )

    ffmpeg_buildconf, ffmpeg_distribution = ffmpeg_distribution_evidence(
        args.ffmpeg_bin, args.ffmpeg_version, args.ffmpeg_source_commit
    )
    ffmpeg_distribution["archive_sha256"] = args.ffmpeg_archive_sha256
    for field in FFMPEG_REVIEW_ONLY_FIELDS:
        ffmpeg_distribution[field] = checked_review["ffmpeg"][field]
    for field, expected in checked_review["ffmpeg"].items():
        if field in {"required_build_flags", "forbidden_build_flags"}:
            continue
        if ffmpeg_distribution.get(field) != expected:
            raise ReleaseError(f"FFmpeg distribution differs from reviewed field {field}")
    ffmpeg_buildconf_path = output / "ffmpeg-buildconf.txt"
    ffmpeg_buildconf_path.write_text(
        ffmpeg_buildconf,
        encoding="utf-8",
        newline="\n",
    )

    package_name = f"collide-o-scope-{args.tag}-windows-x86_64.zip"
    package = output / package_name
    assemble_package(package, args.executable, args.ffmpeg_bin, ffmpeg_buildconf_path)
    source_name = f"collide-o-scope-{args.tag}-source.zip"
    source_archive = output / source_name
    create_source_archive(source_archive, args.tag, args.commit.lower())
    sbom_path = output / "collide-o-scope.cdx.json"
    shutil.copyfile(args.sbom, sbom_path)
    dependency_inventory_path = output / "dependency-license-inventory.json"
    dependency_inventory_path.write_text(
        json.dumps(dependency_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checked_review_path = output / "windows-release-license-review.toml"
    shutil.copyfile(RELEASE_REVIEW_PATH, checked_review_path)
    dependency_review = {
        "schema_version": 1,
        "tag": args.tag,
        "commit": args.commit.lower(),
        "checked_in_review_id": checked_review["review_id"],
        "checked_in_review_owner": checked_review["review_owner"],
        "checked_in_reviewed_on": checked_review["reviewed"].isoformat(),
        "checked_in_review_sha256": digest(checked_review_path),
        "build_identity_sha256": identity["identity_sha256"],
        "cargo_lock_sha256": identity["cargo_lock_sha256"],
        "dependency_inventory_sha256": digest(dependency_inventory_path),
        "dependency_exception_policy_sha256": digest(
            ROOT / "policy" / "dependency-exceptions.toml"
        ),
        "technical_gates": {
            "dependency_policy": "passed",
            "cargo_audit": "passed_with_only_declared_unexpired_exceptions",
            "cargo_deny_advisories_bans_sources": "passed",
            "cargo_deny_licenses": "passed",
            "vendored_wgpu_hal_upstream_plus_declared_patch": "passed_with_seeded_drift_rejection",
        },
        "ffmpeg_distribution": ffmpeg_distribution,
        "packaged_notices": checked_review["packaging"]["required_package_notices"],
        "authenticode": checked_review["authenticode"],
        "legal_conclusion": False,
        "scope": (
            "mechanical dependency/license inventory and attributed release review; "
            "this record is not legal advice"
        ),
    }
    dependency_review_path = output / "dependency-license-review.json"
    dependency_review_path.write_text(
        json.dumps(dependency_review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    provenance = {
        "schema_version": 1,
        "tag": args.tag,
        "commit": args.commit.lower(),
        "source_date_epoch": int(args.source_date_epoch),
        "build_identity": identity,
        "reproducibility": {
            "independent_clean_builds": 2,
            "build_a_executable_sha256": first_hash,
            "build_b_executable_sha256": second_hash,
            "byte_identical": True,
            "authenticode": "unsigned_unavailable",
        },
        "artifacts": {
            package.name: digest(package),
            source_archive.name: digest(source_archive),
            sbom_path.name: digest(sbom_path),
            dependency_inventory_path.name: digest(dependency_inventory_path),
            dependency_review_path.name: digest(dependency_review_path),
            ffmpeg_buildconf_path.name: digest(ffmpeg_buildconf_path),
            checked_review_path.name: digest(checked_review_path),
        },
        "authenticode": checked_review["authenticode"],
        "signing_order": (
            "unsigned builds compared; Sigstore signs checksum/provenance material; "
            "no Authenticode claim"
        ),
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checksums = {
        package.name: digest(package),
        source_archive.name: digest(source_archive),
        sbom_path.name: digest(sbom_path),
        dependency_inventory_path.name: digest(dependency_inventory_path),
        dependency_review_path.name: digest(dependency_review_path),
        ffmpeg_buildconf_path.name: digest(ffmpeg_buildconf_path),
        checked_review_path.name: digest(checked_review_path),
        provenance_path.name: digest(provenance_path),
    }
    (output / "SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in sorted(checksums.items())),
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps({"verified_reproducible": True, "package": package.name, "sha256": digest(package)}))


def parse_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None or match.group(2) in checksums:
            raise ReleaseError(f"invalid SHA256SUMS line: {line!r}")
        checksums[match.group(2)] = match.group(1)
    if not checksums:
        raise ReleaseError("SHA256SUMS is empty")
    return checksums


def validate_provenance_header(provenance: dict, tag: str, commit: str) -> None:
    if provenance.get("schema_version") != 1:
        raise ReleaseError("unsupported release provenance schema")
    if provenance.get("tag") != tag:
        raise ReleaseError("provenance tag does not match the requested release")
    if provenance.get("commit") != commit.lower():
        raise ReleaseError("provenance commit does not match the requested release")


def validate_provenance_artifacts(
    provenance: dict,
    checksums: dict[str, str],
    expected_names: set[str],
) -> None:
    artifacts = provenance.get("artifacts")
    expected_artifact_names = expected_names - {"provenance.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifact_names:
        raise ReleaseError("provenance has a missing or unexpected artifact")
    for name in expected_artifact_names:
        if checksums.get(name) != artifacts.get(name):
            raise ReleaseError(f"{name} hash disagrees between checksums and provenance")


def validate_release_directory_inventory(
    directory: Path,
    checksummed_names: set[str],
    require_signature: bool,
) -> None:
    expected = set(checksummed_names)
    expected.add("SHA256SUMS")
    if require_signature:
        expected.add("SHA256SUMS.sigstore.json")
    if not directory.is_dir():
        raise ReleaseError("release directory is absent")
    entries = list(directory.iterdir())
    unsafe = sorted(
        entry.name for entry in entries if entry.is_symlink() or not entry.is_file()
    )
    if unsafe:
        raise ReleaseError(f"release directory contains non-regular entries: {unsafe}")
    actual = {entry.name for entry in entries}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ReleaseError(
            f"release directory inventory mismatch; missing={missing}, unexpected={unexpected}"
        )


def safe_package_member_parts(name: str) -> tuple[str, ...]:
    if (
        not name
        or not name.isascii()
        or "\\" in name
        or any(ord(character) < 32 for character in name)
    ):
        raise ReleaseError(f"unsafe package entry {name!r}")
    raw_parts = name.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ReleaseError(f"unsafe package entry {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or tuple(path.parts) != tuple(raw_parts):
        raise ReleaseError(f"unsafe package entry {name!r}")
    for part in raw_parts:
        basename = part.split(".", 1)[0].upper()
        if (
            ":" in part
            or any(character in '<>"|?*' for character in part)
            or part.endswith((" ", "."))
            or basename in WINDOWS_RESERVED_BASENAMES
        ):
            raise ReleaseError(f"unsafe Windows package entry {name!r}")
    return tuple(raw_parts)


def write_new_json(path: Path, document: dict, maximum_bytes: int) -> None:
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > maximum_bytes:
        raise ReleaseError(f"JSON output exceeds {maximum_bytes} bytes: {path.name}")
    if path.exists():
        raise ReleaseError(f"refusing to overwrite JSON output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def verify(args: argparse.Namespace) -> dict:
    validate_signature_tag_state(args.require_signature, args.tag_state)
    directory = args.directory.resolve()
    checksums = parse_checksums(directory / "SHA256SUMS")
    package_name = f"collide-o-scope-{args.tag}-windows-x86_64.zip"
    source_name = f"collide-o-scope-{args.tag}-source.zip"
    expected_names = {
        package_name,
        source_name,
        "collide-o-scope.cdx.json",
        "dependency-license-inventory.json",
        "dependency-license-review.json",
        "ffmpeg-buildconf.txt",
        "windows-release-license-review.toml",
        "provenance.json",
    }
    if set(checksums) != expected_names:
        raise ReleaseError("checksum manifest has a missing or unexpected artifact")
    validate_release_directory_inventory(directory, expected_names, args.require_signature)
    for name, expected in checksums.items():
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            raise ReleaseError(f"checksum mismatch for {name}")
    provenance = read_json(directory / "provenance.json")
    validate_provenance_header(provenance, args.tag, args.commit)
    identity = provenance.get("build_identity")
    if not isinstance(identity, dict):
        raise ReleaseError("provenance has no BuildIdentity")
    validate_identity(identity, args.tag, args.commit, args.tag_state)
    commit_epoch = int(git_text("show", "-s", "--format=%ct", args.commit))
    if provenance.get("source_date_epoch") != commit_epoch:
        raise ReleaseError("provenance timestamp does not match the tagged commit")
    reproducibility = provenance.get("reproducibility", {})
    if reproducibility.get("independent_clean_builds") != 2 or reproducibility.get("byte_identical") is not True:
        raise ReleaseError("provenance lacks a successful two-build comparison")
    if reproducibility.get("build_a_executable_sha256") != reproducibility.get("build_b_executable_sha256"):
        raise ReleaseError("provenance records non-identical executable builds")
    checked_review = checked_release_review()
    if (
        reproducibility.get("authenticode") != "unsigned_unavailable"
        or provenance.get("authenticode") != checked_review["authenticode"]
        or provenance.get("signing_order")
        != "unsigned builds compared; Sigstore signs checksum/provenance material; no Authenticode claim"
    ):
        raise ReleaseError("provenance misstates the Authenticode/Sigstore boundary")
    package = directory / package_name
    validate_provenance_artifacts(provenance, checksums, expected_names)
    with tempfile.TemporaryDirectory(prefix="cos-source-verify-") as temp:
        reproduced_source = Path(temp) / source_name
        create_source_archive(reproduced_source, args.tag, args.commit.lower())
        if digest(reproduced_source) != checksums[source_name]:
            raise ReleaseError("published source archive does not reproduce from the tagged commit")
    sbom_path = directory / "collide-o-scope.cdx.json"
    try:
        sbom = read_cyclonedx_json(sbom_path)
    except SbomPolicyError as error:
        raise ReleaseError(f"read strict SBOM JSON: {error}") from error
    validate_sbom(sbom, identity["version"], args.commit, commit_epoch)
    dependency_inventory_path = directory / "dependency-license-inventory.json"
    validate_dependency_inventory(
        read_json(dependency_inventory_path), identity["cargo_lock_sha256"]
    )
    dependency_review = read_json(directory / "dependency-license-review.json")
    checked_review_path = directory / "windows-release-license-review.toml"
    if digest(checked_review_path) != digest(RELEASE_REVIEW_PATH):
        raise ReleaseError("published checked-in release review differs from tagged source")
    ffmpeg_buildconf_path = directory / "ffmpeg-buildconf.txt"
    validate_dependency_review(
        dependency_review,
        dependency_inventory_path,
        ffmpeg_buildconf_path,
        identity,
        args.tag,
        args.commit,
    )

    package_entry_count = 0
    required_notice_hashes: dict[str, str] = {}
    observed_ffmpeg: dict = {}
    with zipfile.ZipFile(package) as archive:
        infos = archive.infolist()
        package_entry_count = len(infos)
        names = [info.filename for info in infos]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ReleaseError("package entries are not uniquely sorted")
        validated_infos: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
        portable_names: set[str] = set()
        for info in infos:
            parts = safe_package_member_parts(info.filename)
            portable_name = "/".join(part.casefold() for part in parts)
            if portable_name in portable_names or info.is_dir():
                raise ReleaseError(f"aliased or non-file package entry {info.filename!r}")
            portable_names.add(portable_name)
            if info.date_time != FIXED_ZIP_TIME:
                raise ReleaseError(f"unsafe or nondeterministic package entry {info.filename!r}")
            validated_infos.append((info, parts))
        validate_ffmpeg_windows_dll_names(
            [
                info.filename
                for info, _parts in validated_infos
                if PurePosixPath(info.filename).suffix.casefold() == ".dll"
            ]
        )
        with tempfile.TemporaryDirectory(prefix="cos-release-verify-") as temp:
            extracted = Path(temp)
            extraction_root = extracted.resolve()
            for info, parts in validated_infos:
                destination = extracted.joinpath(*parts)
                resolved_destination = destination.resolve()
                try:
                    resolved_destination.relative_to(extraction_root)
                except ValueError as error:
                    raise ReleaseError(
                        f"package entry escapes extraction root: {info.filename!r}"
                    ) from error
                if resolved_destination == extraction_root:
                    raise ReleaseError(f"package entry names extraction root: {info.filename!r}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info))
            observed = executable_identity(extracted / "collide-o-scope.exe", extracted)
            if observed != identity:
                raise ReleaseError("downloaded executable BuildIdentity differs from provenance")
            if digest(extracted / "collide-o-scope.exe") != reproducibility["build_a_executable_sha256"]:
                raise ReleaseError("packaged executable differs from the reproduced binary")
            if pe_has_authenticode(extracted / "collide-o-scope.exe"):
                raise ReleaseError("package claims Authenticode unavailable but carries a PE signature")
            if digest(extracted / "ffmpeg.exe") != identity["ffmpeg_binary_sha256"]:
                raise ReleaseError("packaged FFmpeg binary differs from BuildIdentity")
            if digest(extracted / "ffprobe.exe") != identity["ffprobe_binary_sha256"]:
                raise ReleaseError("packaged ffprobe binary differs from BuildIdentity")
            required_notice_hashes = {
                "LICENSE": digest(ROOT / "LICENSE"),
                "COPYRIGHT.md": digest(ROOT / "COPYRIGHT.md"),
                "FFMPEG-BUILDCONF.txt": digest(ffmpeg_buildconf_path),
                "FFMPEG-README.txt": dependency_review["ffmpeg_distribution"]["distribution_readme_sha256"],
                "LICENSES/FFmpeg-GPL-3.0-or-later.txt": dependency_review["ffmpeg_distribution"]["distribution_license_sha256"],
            }
            for name, expected in required_notice_hashes.items():
                path = extracted.joinpath(*PurePosixPath(name).parts)
                if not path.is_file() or digest(path) != expected:
                    raise ReleaseError(f"packaged notice differs or is missing: {name}")
            version, archive_sha256, source_commit = pinned_ffmpeg_distribution()
            observed_buildconf, observed_ffmpeg = ffmpeg_distribution_evidence(
                extracted,
                version,
                source_commit,
                license_path=extracted / "LICENSES/FFmpeg-GPL-3.0-or-later.txt",
                readme_path=extracted / "FFMPEG-README.txt",
            )
            observed_ffmpeg["archive_sha256"] = archive_sha256
            for field in FFMPEG_REVIEW_ONLY_FIELDS:
                observed_ffmpeg[field] = checked_review["ffmpeg"][field]
            if observed_buildconf != ffmpeg_buildconf_path.read_text(encoding="utf-8"):
                raise ReleaseError("packaged FFmpeg build configuration differs from evidence")
            if observed_ffmpeg != dependency_review["ffmpeg_distribution"]:
                raise ReleaseError("packaged FFmpeg distribution differs from review evidence")
            library_facts = [
                fact for fact in identity["ffmpeg_libraries"].split(",")
                if not fact.startswith("ffmpeg=")
            ]
            for library in library_facts:
                if not (extracted / library).is_file():
                    raise ReleaseError(f"BuildIdentity FFmpeg library is absent from package: {library}")
    report = {
        "schema_version": 1,
        "release_verified": True,
        "tag": args.tag,
        "commit": args.commit.lower(),
        "version_json": {
            "status": "passed",
            "identity_sha256": identity["identity_sha256"],
            "version": identity["version"],
            "git_sha": identity["git_sha"],
            "published_artifact": identity["published_artifact"],
        },
        "package": {
            "status": "passed",
            "name": package_name,
            "sha256": checksums[package_name],
            "source_archive_name": source_name,
            "source_archive_sha256": checksums[source_name],
            "entry_count": package_entry_count,
            "executable_sha256": reproducibility["build_a_executable_sha256"],
            "source_archive_reproduced": True,
            "required_notice_sha256": required_notice_hashes,
        },
        "ffmpeg": {
            "status": "passed",
            "version": observed_ffmpeg["version"],
            "binary_sha256": identity["ffmpeg_binary_sha256"],
            "ffprobe_sha256": identity["ffprobe_binary_sha256"],
            "archive_sha256": observed_ffmpeg["archive_sha256"],
            "source_commit": observed_ffmpeg["source_commit"],
            "buildconf_sha256": observed_ffmpeg["buildconf_sha256"],
        },
        "shader": {
            "status": "passed",
            "bundle_sha256": identity["shader_bundle_sha256"],
        },
        "sbom": {
            "status": "passed",
            "sha256": checksums["collide-o-scope.cdx.json"],
        },
        "dependency_evidence": {
            "status": "passed",
            "inventory_sha256": checksums["dependency-license-inventory.json"],
            "review_sha256": checksums["dependency-license-review.json"],
            "checked_review_sha256": checksums["windows-release-license-review.toml"],
        },
        "authenticode": "unavailable_and_unsigned_verified",
    }
    if args.report_output is not None:
        write_new_json(
            args.report_output.resolve(), report, MAX_VERIFICATION_REPORT_BYTES
        )
    print(json.dumps(report, sort_keys=True))
    return report


def expect_release_error(action, expected_fragment: str) -> None:
    try:
        action()
    except ReleaseError as error:
        if expected_fragment not in str(error):
            raise ReleaseError(
                f"self-test expected {expected_fragment!r}, received {str(error)!r}"
            ) from error
    else:
        raise ReleaseError(f"self-test did not reject {expected_fragment!r}")


def self_test() -> None:
    tag = "v9.8.7"
    commit = "1" * 40
    canonical_dlls = list(EXPECTED_FFMPEG_WINDOWS_DLLS)
    validate_ffmpeg_windows_dll_names(canonical_dlls)
    hostile_dll_inventories = (
        (canonical_dlls[:-1], "differs"),
        (canonical_dlls + ["unreviewed.dll"], "differs"),
        (canonical_dlls + [canonical_dlls[0]], "duplicate"),
        (["avcodec-62.dll", *canonical_dlls[1:]], "differs"),
        ([canonical_dlls[0].upper(), *canonical_dlls[1:]], "differs"),
        ([f"nested/{canonical_dlls[0]}", *canonical_dlls[1:]], "differs"),
    )
    for inventory, expected in hostile_dll_inventories:
        expect_release_error(
            lambda inventory=inventory: validate_ffmpeg_windows_dll_names(inventory),
            expected,
        )
    with tempfile.TemporaryDirectory(prefix="cos-ffmpeg-dll-self-test-") as temp:
        dll_root = Path(temp)
        for name in canonical_dlls:
            (dll_root / name).write_bytes(b"seeded altered DLL")
        expect_release_error(
            lambda: ffmpeg_windows_dll_evidence(dll_root),
            "hash differs",
        )
    with tempfile.TemporaryDirectory(prefix="cos-ffmpeg-dll-dir-self-test-") as temp:
        dll_root = Path(temp)
        for name in canonical_dlls:
            (dll_root / name).mkdir()
        expect_release_error(
            lambda: ffmpeg_windows_dll_evidence(dll_root),
            "not a regular file",
        )
    self_test_cyclonedx_policy()
    expect_release_error(
        lambda: validate_sbom({}, "9.8.7", commit, 1_700_000_000),
        "SBOM release-profile",
    )
    validate_resolved_tag_state("absent", None, None, None, commit)
    validate_resolved_tag_state("annotated", "2" * 40, "tag", commit, commit)
    validate_signature_tag_state(False, "absent")
    validate_signature_tag_state(True, "annotated")
    expect_release_error(
        lambda: validate_signature_tag_state(True, "absent"),
        "requires the annotated tag state",
    )
    for hostile_tag_state in (
        ("unexpected", None, None, None, "not absent or annotated"),
        ("absent", "2" * 40, None, None, "local release tag"),
        ("annotated", None, None, None, "exact annotated tag"),
        ("annotated", "2" * 40, "commit", commit, "exact annotated tag"),
        ("annotated", "2" * 40, "tag", "3" * 40, "exact annotated tag"),
    ):
        state, tag_sha, tag_type, peeled, expected = hostile_tag_state
        expect_release_error(
            lambda state=state, tag_sha=tag_sha, tag_type=tag_type, peeled=peeled: validate_resolved_tag_state(
                state, tag_sha, tag_type, peeled, commit
            ),
            expected,
        )
    tag_ref = f"refs/tags/{tag}"
    assert parse_annotated_tag_rows(
        [f"{'2' * 40}\t{tag_ref}", f"{commit}\t{tag_ref}^{{}}"], tag
    ) == ("2" * 40, commit)
    for hostile_rows in (
        [f"{commit}\t{tag_ref}"],
        [f"{commit}\t{tag_ref}^{{}}"],
        [f"{commit}\t{tag_ref}", f"{commit}\t{tag_ref}"],
        [f"{commit}\trefs/heads/main", f"{commit}\t{tag_ref}^{{}}"],
    ):
        expect_release_error(
            lambda hostile_rows=hostile_rows: parse_annotated_tag_rows(
                hostile_rows, tag
            ),
            "remote tag",
        )
    observed_methods: list[str] = []

    class ReleaseListResponse:
        def __init__(self, document, link: str = "") -> None:
            self.payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
            self.headers = {"Link": link}

        def __enter__(self):
            return self

        def __exit__(self, _kind, _value, _traceback) -> None:
            return None

        def read(self, limit: int) -> bytes:
            return self.payload[:limit]

    def existing_release_opener(request, timeout: int):
        assert timeout == 30
        observed_methods.append(request.get_method())
        return ReleaseListResponse([{"id": 1, "tag_name": tag, "draft": True}])

    expect_release_error(
        lambda: require_release_absent(
            "acme/project", tag, "token", existing_release_opener
        ),
        "draft release already exists",
    )
    assert observed_methods == ["GET"]
    observed_methods.clear()

    def existing_published_opener(request, timeout: int):
        assert timeout == 30
        observed_methods.append(request.get_method())
        return ReleaseListResponse([{"id": 2, "tag_name": tag, "draft": False}])

    expect_release_error(
        lambda: require_release_absent(
            "acme/project", tag, "token", existing_published_opener
        ),
        "published release already exists",
    )
    assert observed_methods == ["GET"]
    observed_methods.clear()

    def absent_release_opener(request, timeout: int):
        assert timeout == 30
        observed_methods.append(request.get_method())
        return ReleaseListResponse([])

    require_release_absent("acme/project", tag, "token", absent_release_opener)
    assert observed_methods == ["GET"]
    observed_methods.clear()

    duplicate_rows = [
        {"id": 7, "tag_name": "v1.0.0", "draft": False},
        {"id": 7, "tag_name": "v1.0.1", "draft": True},
    ]
    expect_release_error(
        lambda: require_release_absent(
            "acme/project",
            tag,
            "token",
            lambda _request, timeout: ReleaseListResponse(duplicate_rows)
            if timeout == 30
            else None,
        ),
        "duplicated",
    )
    duplicate_tag_rows = [
        {"id": 8, "tag_name": "v1.0.0", "draft": False},
        {"id": 9, "tag_name": "v1.0.0", "draft": True},
    ]
    expect_release_error(
        lambda: require_release_absent(
            "acme/project",
            tag,
            "token",
            lambda _request, timeout: ReleaseListResponse(duplicate_tag_rows)
            if timeout == 30
            else None,
        ),
        "duplicated",
    )
    expect_release_error(
        lambda: require_release_absent(
            "acme/project",
            tag,
            "token",
            lambda _request, timeout: ReleaseListResponse({"not": "a list"})
            if timeout == 30
            else None,
        ),
        "page shape",
    )
    expect_release_error(
        lambda: require_release_absent(
            "acme/project",
            tag,
            "token",
            lambda _request, timeout: ReleaseListResponse(
                [{"id": True, "tag_name": "v1.0.2", "draft": False}]
            )
            if timeout == 30
            else None,
        ),
        "malformed",
    )
    first_page = [
        {"id": index, "tag_name": f"v0.0.{index}", "draft": False}
        for index in range(1, 101)
    ]
    page_requests: list[int] = []

    def paginated_opener(request, timeout: int):
        assert timeout == 30
        page = int(urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)["page"][0])
        page_requests.append(page)
        if page == 1:
            return ReleaseListResponse(
                first_page,
                '<https://api.github.com/repos/acme/project/releases?per_page=100&page=2>; rel="next"',
            )
        return ReleaseListResponse([])

    require_release_absent("acme/project", tag, "token", paginated_opener)
    assert page_requests == [1, 2]
    expect_release_error(
        lambda: require_release_absent(
            "acme/project",
            tag,
            "token",
            lambda _request, timeout: ReleaseListResponse(
                [],
                '<https://evil.example/releases?page=2&per_page=100>; rel="next"',
            )
            if timeout == 30
            else None,
        ),
        "noncanonical next page",
    )
    expect_release_error(
        lambda: require_release_absent(
            "acme/project",
            tag,
            "token",
            lambda _request, timeout: ReleaseListResponse([], "malformed-link")
            if timeout == 30
            else None,
        ),
        "malformed pagination",
    )

    def failed_release_opener(request, timeout: int):
        assert timeout == 30
        observed_methods.append(request.get_method())
        raise urllib.error.HTTPError(
            request.full_url, 503, "Service Unavailable", None, None
        )

    expect_release_error(
        lambda: require_release_absent(
            "acme/project", tag, "token", failed_release_opener
        ),
        "HTTP 503",
    )
    assert observed_methods == ["GET"]
    validate_provenance_header(
        {"schema_version": 1, "tag": tag, "commit": commit}, tag, commit
    )
    for field, value, expected in (
        ("schema_version", 2, "schema"),
        ("tag", "v9.8.6", "tag"),
        ("commit", "2" * 40, "commit"),
    ):
        document = {"schema_version": 1, "tag": tag, "commit": commit}
        document[field] = value
        expect_release_error(
            lambda document=document: validate_provenance_header(
                document, tag, commit
            ),
            expected,
        )

    artifact_checksums = {"artifact.bin": "a" * 64, "provenance.json": "b" * 64}
    artifact_provenance = {"artifacts": {"artifact.bin": "a" * 64}}
    validate_provenance_artifacts(
        artifact_provenance, artifact_checksums, set(artifact_checksums)
    )
    artifact_provenance["artifacts"]["stale.bin"] = "c" * 64
    expect_release_error(
        lambda: validate_provenance_artifacts(
            artifact_provenance, artifact_checksums, set(artifact_checksums)
        ),
        "unexpected artifact",
    )
    expect_release_error(
        lambda: validate_provenance_artifacts(
            {"artifacts": []}, artifact_checksums, set(artifact_checksums)
        ),
        "unexpected artifact",
    )
    expect_release_error(
        lambda: validate_provenance_artifacts(
            {"artifacts": {}}, artifact_checksums, set(artifact_checksums)
        ),
        "missing or unexpected artifact",
    )
    assert safe_package_member_parts("LICENSES/FFmpeg-GPL-3.0-or-later.txt") == (
        "LICENSES",
        "FFmpeg-GPL-3.0-or-later.txt",
    )
    for hostile_name in (
        "../escape.bin",
        r"..\escape.bin",
        "C:/escape.bin",
        r"\\server\share\escape.bin",
        "NUL.txt",
        "folder/./alias.bin",
        "folder/trailing. ",
    ):
        expect_release_error(
            lambda hostile_name=hostile_name: safe_package_member_parts(hostile_name),
            "unsafe",
        )

    with tempfile.TemporaryDirectory(prefix="cos-release-self-test-") as temp:
        directory = Path(temp)
        (directory / "artifact.bin").write_bytes(b"artifact")
        (directory / "SHA256SUMS").write_text(
            f"{digest(directory / 'artifact.bin')}  artifact.bin\n",
            encoding="ascii",
            newline="\n",
        )
        validate_release_directory_inventory(directory, {"artifact.bin"}, False)
        (directory / "stale.bin").write_bytes(b"stale")
        expect_release_error(
            lambda: validate_release_directory_inventory(
                directory, {"artifact.bin"}, False
            ),
            "unexpected",
        )
        (directory / "stale.bin").unlink()
        expect_release_error(
            lambda: validate_release_directory_inventory(
                directory, {"artifact.bin"}, True
            ),
            "missing",
        )
        (directory / "SHA256SUMS.sigstore.json").write_text(
            "{}\n", encoding="utf-8", newline="\n"
        )
        validate_release_directory_inventory(directory, {"artifact.bin"}, True)
    with tempfile.TemporaryDirectory(prefix="cos-source-timezone-self-test-") as temp:
        directory = Path(temp)
        current_commit = git_text("rev-parse", "HEAD")
        original_timezone = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "America/New_York"
            first_archive = directory / "first.zip"
            create_source_archive(first_archive, tag, current_commit)
            if os.environ.get("TZ") != "America/New_York":
                raise ReleaseError("source archive creation mutated the caller timezone")
            os.environ["TZ"] = "Pacific/Honolulu"
            second_archive = directory / "second.zip"
            create_source_archive(second_archive, tag, current_commit)
            if os.environ.get("TZ") != "Pacific/Honolulu":
                raise ReleaseError("source archive creation mutated the caller timezone")
        finally:
            if original_timezone is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = original_timezone
        if digest(first_archive) != digest(second_archive):
            raise ReleaseError("source archive bytes depend on the caller timezone")
    print(
        "release verifier self-test valid: create-only preflight, provenance, "
        "timezone-independent source archive, and exact asset inventory fail closed"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    tag_command = commands.add_parser("annotated-tag")
    tag_command.add_argument("--tag", required=True)
    tag_command.add_argument("--commit", required=True)
    tag_command.add_argument("--tag-object")
    tag_command.add_argument("--remote", default="origin")
    tag_command.add_argument("--output", type=Path)
    tag_command.add_argument("--github-output", type=Path)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--executable", type=Path, required=True)
    prepare_command.add_argument("--second-executable", type=Path, required=True)
    prepare_command.add_argument("--ffmpeg-bin", type=Path, required=True)
    prepare_command.add_argument("--ffmpeg-version", required=True)
    prepare_command.add_argument("--ffmpeg-archive-sha256", required=True)
    prepare_command.add_argument("--ffmpeg-source-commit", required=True)
    prepare_command.add_argument("--sbom", type=Path, required=True)
    prepare_command.add_argument("--dependency-inventory", type=Path, required=True)
    prepare_command.add_argument("--tag", required=True)
    prepare_command.add_argument("--tag-state", choices=("absent", "annotated"), required=True)
    prepare_command.add_argument("--commit", required=True)
    prepare_command.add_argument("--source-date-epoch", required=True, type=int)
    prepare_command.add_argument("--output", required=True, type=Path)
    verify_command = commands.add_parser("verify")
    verify_command.add_argument("--directory", type=Path, required=True)
    verify_command.add_argument("--tag", required=True)
    verify_command.add_argument("--tag-state", choices=("absent", "annotated"), required=True)
    verify_command.add_argument("--commit", required=True)
    verify_command.add_argument("--require-signature", action="store_true")
    verify_command.add_argument("--report-output", type=Path)
    absent_command = commands.add_parser("release-absent")
    absent_command.add_argument("--repository", required=True)
    absent_command.add_argument("--tag", required=True)
    absent_command.add_argument("--token-env", default="GH_TOKEN")
    commands.add_parser("self-test")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "verify":
            verify(args)
        elif args.command == "annotated-tag":
            annotated_tag(args)
        elif args.command == "release-absent":
            token = os.environ.get(args.token_env, "")
            require_release_absent(args.repository, args.tag, token)
        else:
            self_test()
    except (OSError, ReleaseError, zipfile.BadZipFile) as error:
        print(f"release verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
