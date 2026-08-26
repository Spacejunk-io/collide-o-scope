#!/usr/bin/env python3
"""Build and validate the bounded post-redownload release receipt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_RECEIPT_BYTES = 512 * 1024
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TAG = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
REQUIRED_WORKFLOWS = {"ci.yml", "adversarial.yml"}
GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
SLSA_PROVENANCE_V1 = "https://slsa.dev/provenance/v1"
PLATFORM_TEST_STEPS = {
    "Linux (Ubuntu 24.04)": "Check, test, and lint on Unix",
    "macOS 15": "Check, test, and lint on Unix",
    "Windows (VS 2022)": "Check, test, and lint on Windows",
}
DEPENDENCY_JOB = "Dependency policy and supply-chain provenance"
FORMAT_STEP = "Check Rust formatting and JavaScript syntax"
CAPABILITY_STEP = "Check generated capability registry"
PLATFORM_VENDOR_STEP = "Verify the vendored wgpu-hal archive and sole patch"
DEPENDENCY_EXCEPTION_STEP = "Reject stale or unowned advisory exceptions"
DEPENDENCY_VENDOR_STEP = "Fetch locked dependencies and verify vendored source"
REQUIRED_STEP_GROUPS = {
    "format": {(job, FORMAT_STEP) for job in PLATFORM_TEST_STEPS},
    "capability": {(job, CAPABILITY_STEP) for job in PLATFORM_TEST_STEPS},
    "check_test_clippy": set(PLATFORM_TEST_STEPS.items()),
    "dependency_exception": {(DEPENDENCY_JOB, DEPENDENCY_EXCEPTION_STEP)},
    "vendor": {
        *((job, PLATFORM_VENDOR_STEP) for job in PLATFORM_TEST_STEPS),
        (DEPENDENCY_JOB, DEPENDENCY_VENDOR_STEP),
    },
}
IDENTITY_PAYLOAD_KEYS = [
    "package_name", "version", "git_sha", "git_dirty", "profile", "target",
    "enabled_features", "rustc_vv", "cargo_version", "linker_identity",
    "sdk_identity", "ffmpeg_libraries", "ffmpeg_binary_version",
    "ffmpeg_binary_sha256", "ffprobe_binary_version", "ffprobe_binary_sha256",
    "shader_bundle_sha256", "cargo_lock_sha256", "published_artifact",
]


class ReceiptError(RuntimeError):
    pass


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def json_document_digest(document: dict) -> str:
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checksum_manifest_digest(checksums: dict[str, str]) -> str:
    encoded = "".join(
        f"{value}  {name}\n" for name, value in sorted(checksums.items())
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def shader_bundle_digest() -> str:
    hasher = hashlib.sha256()
    hasher.update(b"collide-o-scope shader bundle v1\0")
    for path in sorted((ROOT / "src" / "shaders").glob("*.wgsl")):
        name = path.relative_to(ROOT).as_posix().encode("utf-8")
        data = path.read_bytes()
        hasher.update(len(name).to_bytes(8, "little"))
        hasher.update(name)
        hasher.update(len(data).to_bytes(8, "little"))
        hasher.update(data)
    return hasher.hexdigest()


def identity_payload(identity: dict) -> bytes:
    lines = ["domain=collide-o-scope build identity v1"]
    for key in IDENTITY_PAYLOAD_KEYS:
        if key not in identity:
            raise ReceiptError(f"BuildIdentity is missing {key}")
        value = identity[key]
        if isinstance(value, bool):
            value = "true" if value else "false"
        lines.append(f"{key}={value}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def checked_release_policy() -> dict:
    path = ROOT / "policy" / "windows-release-license-review.toml"
    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReceiptError(f"read checked release policy: {error}") from error
    ffmpeg = document.get("ffmpeg")
    authenticode = document.get("authenticode")
    if (
        document.get("schema_version") != 1
        or not isinstance(ffmpeg, dict)
        or not isinstance(authenticode, dict)
    ):
        raise ReceiptError("checked release policy has an unsupported schema")
    native = {
        "version": ffmpeg.get("version"),
        "archive_sha256": ffmpeg.get("archive_sha256"),
        "source_commit": ffmpeg.get("source_commit"),
        "buildconf_sha256": ffmpeg.get("buildconf_sha256"),
        "runtime_license_text_sha256": ffmpeg.get("runtime_license_text_sha256"),
        "distribution_license_sha256": ffmpeg.get("distribution_license_sha256"),
        "distribution_readme_sha256": ffmpeg.get("distribution_readme_sha256"),
    }
    hash_fields = set(native) - {"version", "source_commit"}
    if (
        not isinstance(native["version"], str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", native["version"]) is None
        or not isinstance(native["source_commit"], str)
        or SHA40.fullmatch(native["source_commit"]) is None
        or any(
            not isinstance(native[field], str)
            or SHA256.fullmatch(native[field]) is None
            for field in hash_fields
        )
        or authenticode.get("status") != "unavailable"
    ):
        raise ReceiptError("checked release policy contains an invalid native identity")
    return {
        "sha256": digest(path),
        "native_distribution": native,
        "authenticode": authenticode,
    }


def validate_build_identity(
    identity: object,
    tag: str,
    commit: str,
    evidence: dict,
    native_distribution: dict,
) -> None:
    if not isinstance(identity, dict) or set(identity) != {
        "schema_version", *IDENTITY_PAYLOAD_KEYS, "identity_sha256"
    }:
        raise ReceiptError("final receipt BuildIdentity shape is incomplete")
    expected_digest = hashlib.sha256(identity_payload(identity)).hexdigest()
    version = tag.removeprefix("v")
    if (
        identity.get("schema_version") != 1
        or identity.get("package_name") != "collide-o-scope"
        or identity.get("version") != version
        or identity.get("git_sha") != commit
        or identity.get("git_dirty") is not False
        or identity.get("profile") != "release"
        or identity.get("target") != "x86_64-pc-windows-msvc"
        or identity.get("published_artifact") is not True
        or identity.get("identity_sha256") != expected_digest
        or evidence.get("build_identity_sha256") != expected_digest
        or identity.get("cargo_lock_sha256") != evidence.get("cargo_lock_sha256")
        or identity.get("shader_bundle_sha256")
        != evidence.get("shader_bundle_sha256")
        or identity.get("ffmpeg_binary_sha256")
        != evidence.get("ffmpeg_binary_sha256")
        or identity.get("ffprobe_binary_sha256")
        != evidence.get("ffprobe_binary_sha256")
        or identity.get("ffmpeg_binary_version")
        != f"ffmpeg version {native_distribution.get('version')}"
        or identity.get("ffprobe_binary_version")
        != f"ffprobe version {native_distribution.get('version')}"
        or evidence.get("cargo_lock_sha256") != digest(ROOT / "Cargo.lock")
        or evidence.get("shader_bundle_sha256") != shader_bundle_digest()
    ):
        raise ReceiptError("final receipt BuildIdentity associations are contradictory")


def validate_native_distribution(
    native: object,
    report_ffmpeg: dict,
    package_notices: dict,
    checksum_inventory: dict[str, str],
    policy: dict,
) -> None:
    if not isinstance(native, dict) or native != policy["native_distribution"]:
        raise ReceiptError("final receipt native distribution differs from checked policy")
    if (
        report_ffmpeg.get("version") != native["version"]
        or report_ffmpeg.get("archive_sha256") != native["archive_sha256"]
        or report_ffmpeg.get("source_commit") != native["source_commit"]
        or report_ffmpeg.get("buildconf_sha256") != native["buildconf_sha256"]
        or checksum_inventory.get("ffmpeg-buildconf.txt")
        != native["buildconf_sha256"]
        or package_notices.get("FFMPEG-BUILDCONF.txt")
        != native["buildconf_sha256"]
        or package_notices.get("FFMPEG-README.txt")
        != native["distribution_readme_sha256"]
        or package_notices.get("LICENSES/FFmpeg-GPL-3.0-or-later.txt")
        != native["distribution_license_sha256"]
    ):
        raise ReceiptError("final receipt FFmpeg review/package associations are contradictory")


def read_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ReceiptError(f"JSON input is not one regular file: {path}")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ReceiptError(f"JSON input exceeds {MAX_INPUT_BYTES} bytes: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ReceiptError(f"{path.name} must contain one JSON object")
    return value


def parse_checksums(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024:
        raise ReceiptError("SHA256SUMS is missing, unsafe, or unbounded")
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]{1,160})", line)
        if match is None or match.group(2) in checksums:
            raise ReceiptError("SHA256SUMS has an invalid or duplicate row")
        checksums[match.group(2)] = match.group(1)
    if len(checksums) != 8:
        raise ReceiptError("initial checksum inventory must contain exactly eight assets")
    return checksums


def expected_initial_names(tag: str) -> set[str]:
    return {
        f"collide-o-scope-{tag}-windows-x86_64.zip",
        f"collide-o-scope-{tag}-source.zip",
        "collide-o-scope.cdx.json",
        "dependency-license-inventory.json",
        "dependency-license-review.json",
        "ffmpeg-buildconf.txt",
        "windows-release-license-review.toml",
        "provenance.json",
        "SHA256SUMS",
        "SHA256SUMS.sigstore.json",
    }


def validate_step_receipts(
    value: object,
    expected_pairs: set[tuple[str, str]],
) -> list[dict]:
    if not isinstance(value, list) or len(value) != len(expected_pairs):
        raise ReceiptError("final-candidate step receipt count is incomplete")
    observed = []
    identities: set[tuple[str, str]] = set()
    for row in value:
        if (
            not isinstance(row, dict)
            or set(row) != {"job", "step", "conclusion"}
            or not isinstance(row.get("job"), str)
            or not 1 <= len(row["job"]) <= 256
            or not isinstance(row.get("step"), str)
            or (row["job"], row["step"]) not in expected_pairs
            or row.get("conclusion") != "success"
        ):
            raise ReceiptError("final-candidate step receipt is malformed or unsuccessful")
        identity = (row["job"], row["step"])
        if identity in identities:
            raise ReceiptError("final-candidate step receipt is duplicated")
        identities.add(identity)
        observed.append(dict(row))
    if identities != expected_pairs:
        raise ReceiptError("final-candidate step receipts use the wrong CI jobs")
    return observed


def validate_final_candidate_validation(
    document: object, ci_run_id: int, ci_run_attempt: int, repository: str
) -> dict:
    if not isinstance(document, dict) or set(document) != {
        "ci_run_id", "ci_run_attempt", "format", "check_test_clippy",
        "capability_registry_contradiction_gate", "dependency_exception_policy",
        "vendor_verifier",
    } or document.get("ci_run_id") != ci_run_id or document.get("ci_run_attempt") != ci_run_attempt:
        raise ReceiptError("final-candidate validation is absent or bound to another CI run")
    formatting = document.get("format")
    check_test = document.get("check_test_clippy")
    capability = document.get("capability_registry_contradiction_gate")
    exceptions = document.get("dependency_exception_policy")
    vendor = document.get("vendor_verifier")
    if (
        not isinstance(formatting, dict)
        or set(formatting) != {"commands", "steps", "conclusion"}
        or formatting.get("commands")
        != [
            "cargo fmt --all -- --check",
            "node --check static/app.js",
            "node --check docs/ui-ux/wireframe.js",
        ]
        or formatting.get("conclusion") != "success"
    ):
        raise ReceiptError("final-candidate format result is incomplete")
    validate_step_receipts(
        formatting.get("steps"), REQUIRED_STEP_GROUPS["format"]
    )
    if (
        not isinstance(check_test, dict)
        or set(check_test) != {"commands", "steps", "conclusion", "test_results"}
        or check_test.get("commands")
        != [
            "cargo check --locked --all-targets --all-features",
            "cargo test --locked --all-targets --all-features",
            "cargo clippy --locked --all-targets --all-features -- -D warnings",
        ]
        or check_test.get("conclusion") != "success"
    ):
        raise ReceiptError("final-candidate check/test/Clippy result is incomplete")
    validate_step_receipts(
        check_test.get("steps"),
        REQUIRED_STEP_GROUPS["check_test_clippy"],
    )
    test_results = check_test.get("test_results")
    expected_result_fields = {
        "summary_records", "passed", "failed", "ignored", "measured",
        "filtered_out", "ignored_test_names", "external_fixture_ignored_count",
        "external_fixture_ignored_names", "platform_jobs", "source",
    }
    if (
        not isinstance(test_results, dict)
        or set(test_results) != expected_result_fields
        or not isinstance(test_results.get("summary_records"), int)
        or test_results["summary_records"] < 3
        or not isinstance(test_results.get("passed"), int)
        or test_results["passed"] <= 0
        or test_results.get("failed") != 0
        or any(
            not isinstance(test_results.get(field), int) or test_results[field] < 0
            for field in ("ignored", "measured", "filtered_out")
        )
        or test_results.get("source") != "selected exact-SHA CI platform job logs"
    ):
        raise ReceiptError("final-candidate test counts are absent or unsuccessful")
    ignored_names = test_results.get("ignored_test_names")
    external_names = test_results.get("external_fixture_ignored_names")
    if (
        not isinstance(ignored_names, list)
        or not isinstance(external_names, list)
        or len(ignored_names) > 256
        or any(not isinstance(name, str) or not 1 <= len(name) <= 512 for name in ignored_names)
        or not set(external_names).issubset(set(ignored_names))
        or not isinstance(test_results.get("external_fixture_ignored_count"), int)
        or not len(external_names)
        <= test_results["external_fixture_ignored_count"]
        <= test_results["ignored"]
    ):
        raise ReceiptError("final-candidate ignored/external fixture counts are malformed")
    platform_rows = test_results.get("platform_jobs")
    observed_jobs: set[str] = set()
    observed_job_ids: set[int] = set()
    if not isinstance(platform_rows, list) or len(platform_rows) != 3:
        raise ReceiptError("final-candidate per-platform test evidence is incomplete")
    for row in platform_rows:
        if not isinstance(row, dict) or set(row) != {
            "job", "job_id", "logs_url", "summary_records", "passed", "failed",
            "ignored", "measured", "filtered_out", "ignored_test_names",
            "external_fixture_ignored_count", "external_fixture_ignored_names",
        }:
            raise ReceiptError("final-candidate per-platform test row is malformed")
        job = row.get("job")
        job_id = row.get("job_id")
        row_ignored = row.get("ignored_test_names")
        row_external = row.get("external_fixture_ignored_names")
        if (
            not isinstance(job, str)
            or job not in PLATFORM_TEST_STEPS
            or job in observed_jobs
            or not isinstance(job_id, int)
            or job_id <= 0
            or job_id in observed_job_ids
            or row.get("logs_url")
            != f"https://api.github.com/repos/{repository}/actions/jobs/{job_id}/logs"
            or not isinstance(row.get("summary_records"), int)
            or row["summary_records"] < 1
            or not isinstance(row.get("passed"), int)
            or row["passed"] <= 0
            or row.get("failed") != 0
            or any(
                not isinstance(row.get(field), int) or row[field] < 0
                for field in ("ignored", "measured", "filtered_out")
            )
            or not isinstance(row_ignored, list)
            or not isinstance(row_external, list)
            or len(row_ignored) > 256
            or any(
                not isinstance(name, str) or not 1 <= len(name) <= 512
                for name in row_ignored
            )
            or not set(row_external).issubset(set(row_ignored))
            or not isinstance(row.get("external_fixture_ignored_count"), int)
            or not len(row_external)
            <= row["external_fixture_ignored_count"]
            <= row["ignored"]
        ):
            raise ReceiptError("final-candidate per-platform test row is contradictory")
        observed_jobs.add(job)
        observed_job_ids.add(job_id)
    if observed_jobs != set(PLATFORM_TEST_STEPS):
        raise ReceiptError("final-candidate per-platform job inventory is not exact")
    aggregate_fields = (
        "summary_records", "passed", "ignored", "measured", "filtered_out",
        "external_fixture_ignored_count",
    )
    if (
        any(
            test_results[field] != sum(row[field] for row in platform_rows)
            for field in aggregate_fields
        )
        or ignored_names
        != sorted({name for row in platform_rows for name in row["ignored_test_names"]})
        or external_names
        != sorted(
            {
                name
                for row in platform_rows
                for name in row["external_fixture_ignored_names"]
            }
        )
    ):
        raise ReceiptError("final-candidate platform test aggregates are contradictory")
    if (
        not isinstance(capability, dict)
        or set(capability) != {"command", "steps", "conclusion"}
        or capability.get("command")
        != "cargo run --locked --bin generate_capabilities -- --check"
        or capability.get("conclusion") != "success"
    ):
        raise ReceiptError("final P10 contradiction-gate result is incomplete")
    validate_step_receipts(
        capability.get("steps"), REQUIRED_STEP_GROUPS["capability"]
    )
    if (
        not isinstance(exceptions, dict)
        or set(exceptions) != {"command", "steps", "conclusion"}
        or exceptions.get("command") != "python scripts/check-dependency-policy.py"
        or exceptions.get("conclusion") != "success"
    ):
        raise ReceiptError("dependency exception status is incomplete")
    validate_step_receipts(
        exceptions.get("steps"), REQUIRED_STEP_GROUPS["dependency_exception"]
    )
    if (
        not isinstance(vendor, dict)
        or set(vendor) != {"command", "steps", "conclusion"}
        or vendor.get("command") != "python scripts/verify-vendored-wgpu-hal.py --self-test"
        or vendor.get("conclusion") != "success"
    ):
        raise ReceiptError("vendor-verifier result is incomplete")
    validate_step_receipts(
        vendor.get("steps"),
        REQUIRED_STEP_GROUPS["vendor"],
    )
    return copy.deepcopy(document)


def validate_required_runs(
    document: dict, repository: str, commit: str
) -> tuple[list[dict], dict]:
    if (
        document.get("schema_version") != 1
        or document.get("repository") != repository
        or document.get("commit") != commit
    ):
        raise ReceiptError("required-workflow receipt is not bound to this release")
    runs = document.get("selected_newest_exact_commit_runs")
    if not isinstance(runs, list) or len(runs) != len(REQUIRED_WORKFLOWS):
        raise ReceiptError("required-workflow receipt has the wrong run count")
    observed: set[str] = set()
    normalized: list[dict] = []
    for run in runs:
        if not isinstance(run, dict) or set(run) != {
            "workflow", "run_id", "run_number", "run_attempt", "url", "conclusion"
        }:
            raise ReceiptError("required-workflow run has an unsupported shape")
        workflow = run.get("workflow")
        run_id = run.get("run_id")
        url = run.get("url")
        if (
            not isinstance(workflow, str)
            or workflow not in REQUIRED_WORKFLOWS
            or workflow in observed
            or not isinstance(run_id, int)
            or run_id <= 0
            or not isinstance(run.get("run_number"), int)
            or run["run_number"] <= 0
            or not isinstance(run.get("run_attempt"), int)
            or run["run_attempt"] <= 0
            or run.get("conclusion") != "success"
            or url != f"https://github.com/{repository}/actions/runs/{run_id}"
        ):
            raise ReceiptError("required-workflow selection is stale or malformed")
        observed.add(workflow)
        normalized.append({**run, "head_sha": commit})
    if observed != REQUIRED_WORKFLOWS:
        raise ReceiptError("required-workflow receipt omits a release gate")
    ci_run_id = next(
        row["run_id"] for row in normalized if row["workflow"] == "ci.yml"
    )
    ci_run_attempt = next(
        row["run_attempt"] for row in normalized if row["workflow"] == "ci.yml"
    )
    validation = validate_final_candidate_validation(
        document.get("final_candidate_validation"),
        ci_run_id,
        ci_run_attempt,
        repository,
    )
    return sorted(normalized, key=lambda item: item["workflow"]), validation


def github_attestation_policy(repository: str, tag: str, commit: str) -> dict:
    return {
        "repository": repository,
        "certificate_identity": (
            f"https://github.com/{repository}/.github/workflows/"
            f"release-trust.yml@refs/tags/{tag}"
        ),
        "certificate_oidc_issuer": GITHUB_OIDC_ISSUER,
        "predicate_type": SLSA_PROVENANCE_V1,
        "source_ref": f"refs/tags/{tag}",
        "source_digest": commit,
    }


def validate_attestations(
    document: dict,
    repository: str,
    tag: str,
    commit: str,
    assets: dict[str, str],
) -> list[dict]:
    expected_policy = github_attestation_policy(repository, tag, commit)
    if (
        not isinstance(document, dict)
        or set(document) != {
            "schema_version", "repository", "commit", "policy", "assets"
        }
        or document.get("schema_version") != 2
        or document.get("repository") != repository
        or document.get("commit") != commit
        or document.get("policy") != expected_policy
    ):
        raise ReceiptError(
            "asset attestation results are not bound to the exact release signer policy"
        )
    rows = document.get("assets")
    if not isinstance(rows, list) or len(rows) != len(assets):
        raise ReceiptError("asset attestation result count is incomplete")
    observed: set[str] = set()
    normalized: list[dict] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "sha256", "verified"}:
            raise ReceiptError("asset attestation result has an unsupported shape")
        name = row.get("name")
        if (
            not isinstance(name, str)
            or SAFE_NAME.fullmatch(name) is None
            or name in observed
            or assets.get(name) != row.get("sha256")
            or row.get("verified") is not True
        ):
            raise ReceiptError("asset attestation result is missing, stale, or false")
        observed.add(name)
        normalized.append(
            {"name": name, "sha256": row["sha256"], "github_attestation": "verified"}
        )
    if observed != set(assets):
        raise ReceiptError("not every exact initial asset has an attestation result")
    return sorted(normalized, key=lambda item: item["name"])


def vendor_hashes() -> dict:
    path = ROOT / "third_party" / "wgpu-hal-29.0.4.vendor.json"
    document = read_json(path)
    crate = document.get("crate")
    delta = document.get("intended_delta")
    if document.get("schema_version") != 1 or not isinstance(crate, dict) or not isinstance(delta, dict):
        raise ReceiptError("vendor evidence has an unsupported schema")
    values = {
        "manifest_sha256": digest(path),
        "archive_sha256": crate.get("archive_sha256"),
        "patch_sha256": delta.get("patch_sha256"),
        "upstream_sha256": delta.get("upstream_sha256"),
        "vendored_sha256": delta.get("vendored_sha256"),
    }
    if any(not isinstance(value, str) or SHA256.fullmatch(value) is None for value in values.values()):
        raise ReceiptError("vendor evidence contains a non-SHA-256 identity")
    return values


def source_evidence_receipts() -> list[dict]:
    evidence_root = ROOT / "docs" / "evidence"
    names = {
        "v1.7.0-improvement-audit-release-receipt.md",
        "v1.7.1-release-recovery-receipt.md",
        "v1.7.2-release-recovery-receipt.md",
        "v1.7.3-release-recovery-receipt.md",
        "v1.7.4-release-recovery-receipt.md",
        "v1.8.0-ffmpeg-9-software-baseline-receipt.md",
        "v1.8.1-patch-refresh-receipt.md",
    }
    for prefix in ("p3", "p9", "p10"):
        names.update(path.name for path in evidence_root.glob(f"{prefix}*"))
    if not 2 <= len(names) <= 32:
        raise ReceiptError("source evidence receipt inventory is absent or unbounded")
    rows = []
    for name in sorted(names):
        if SAFE_NAME.fullmatch(name) is None:
            raise ReceiptError("source evidence receipt has an unsafe name")
        path = evidence_root / name
        if not path.is_file() or path.is_symlink():
            raise ReceiptError(f"source evidence receipt is not regular: {name}")
        rows.append({"path": f"docs/evidence/{name}", "sha256": digest(path)})
    return rows


def build_receipt(args: argparse.Namespace) -> tuple[dict, list[str]]:
    tag = args.tag
    commit = args.commit.lower()
    tag_object = args.tag_object.lower()
    repository = args.repository
    if (
        TAG.fullmatch(tag) is None
        or SHA40.fullmatch(commit) is None
        or SHA40.fullmatch(tag_object) is None
        or REPOSITORY.fullmatch(repository) is None
        or type(args.release_database_id) is not int
        or args.release_database_id <= 0
    ):
        raise ReceiptError("release identity arguments are malformed")
    expected_release_url = f"https://github.com/{repository}/releases/tag/{tag}"
    expected_identity = (
        f"https://github.com/{repository}/.github/workflows/release-trust.yml@refs/tags/{tag}"
    )
    if args.release_url != expected_release_url or args.workflow_identity != expected_identity:
        raise ReceiptError("release URL or Sigstore workflow identity is not exact")

    directory = args.directory.resolve()
    if not directory.is_dir():
        raise ReceiptError("redownload directory is absent")
    entries = list(directory.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ReceiptError("redownload directory contains a non-regular entry")
    actual_names = {entry.name for entry in entries}
    expected_names = expected_initial_names(tag)
    if actual_names != expected_names:
        raise ReceiptError("redownload inventory is not the exact initial asset set")
    checksums = parse_checksums(directory / "SHA256SUMS")
    if set(checksums) != expected_names - {"SHA256SUMS", "SHA256SUMS.sigstore.json"}:
        raise ReceiptError("checksum inventory is not the exact initial payload set")
    assets = {name: digest(directory / name) for name in sorted(expected_names)}
    for name, expected in checksums.items():
        if assets[name] != expected:
            raise ReceiptError(f"redownloaded asset hash changed: {name}")

    provenance = read_json(directory / "provenance.json")
    identity = provenance.get("build_identity")
    reproducibility = provenance.get("reproducibility")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("tag") != tag
        or provenance.get("commit") != commit
        or not isinstance(identity, dict)
        or not isinstance(reproducibility, dict)
        or reproducibility.get("independent_clean_builds") != 2
        or reproducibility.get("byte_identical") is not True
        or reproducibility.get("build_a_executable_sha256")
        != reproducibility.get("build_b_executable_sha256")
    ):
        raise ReceiptError("release provenance lacks the exact two-build proof")
    for field in (
        "identity_sha256", "cargo_lock_sha256", "shader_bundle_sha256",
        "ffmpeg_binary_sha256", "ffprobe_binary_sha256",
    ):
        if not isinstance(identity.get(field), str) or SHA256.fullmatch(identity[field]) is None:
            raise ReceiptError(f"BuildIdentity field {field} is invalid")
    policy = checked_release_policy()
    dependency_review = read_json(directory / "dependency-license-review.json")
    review_distribution = dependency_review.get("ffmpeg_distribution")
    if not isinstance(review_distribution, dict):
        raise ReceiptError("dependency review lacks the native distribution")
    native_distribution = {
        field: review_distribution.get(field)
        for field in policy["native_distribution"]
    }
    if native_distribution != policy["native_distribution"]:
        raise ReceiptError("dependency review differs from checked native policy")
    if checksums["windows-release-license-review.toml"] != policy["sha256"]:
        raise ReceiptError("checked release review asset differs from tagged source")

    report = read_json(args.verification_report)
    package_name = f"collide-o-scope-{tag}-windows-x86_64.zip"
    if (
        report.get("schema_version") != 1
        or report.get("release_verified") is not True
        or report.get("tag") != tag
        or report.get("commit") != commit
        or report.get("authenticode") != "unavailable_and_unsigned_verified"
        or report.get("version_json", {}).get("status") != "passed"
        or report.get("version_json", {}).get("identity_sha256") != identity["identity_sha256"]
        or report.get("version_json", {}).get("version") != tag.removeprefix("v")
        or report.get("version_json", {}).get("git_sha") != commit
        or report.get("version_json", {}).get("published_artifact") is not True
        or report.get("package", {}).get("status") != "passed"
        or report.get("package", {}).get("sha256") != assets[package_name]
        or report.get("package", {}).get("executable_sha256")
        != reproducibility["build_a_executable_sha256"]
        or report.get("package", {}).get("source_archive_reproduced") is not True
        or not isinstance(report.get("package", {}).get("entry_count"), int)
        or report.get("package", {}).get("entry_count") <= 0
        or report.get("ffmpeg", {}).get("status") != "passed"
        or report.get("ffmpeg", {}).get("binary_sha256")
        != identity["ffmpeg_binary_sha256"]
        or report.get("ffmpeg", {}).get("ffprobe_sha256")
        != identity["ffprobe_binary_sha256"]
        or report.get("shader", {}).get("status") != "passed"
        or report.get("shader", {}).get("bundle_sha256") != identity["shader_bundle_sha256"]
        or report.get("sbom", {}).get("status") != "passed"
        or report.get("sbom", {}).get("sha256") != checksums["collide-o-scope.cdx.json"]
        or report.get("dependency_evidence", {}).get("status") != "passed"
        or report.get("dependency_evidence", {}).get("inventory_sha256")
        != checksums["dependency-license-inventory.json"]
        or report.get("dependency_evidence", {}).get("review_sha256")
        != checksums["dependency-license-review.json"]
        or report.get("dependency_evidence", {}).get("checked_review_sha256")
        != checksums["windows-release-license-review.toml"]
    ):
        raise ReceiptError("downloaded verification report is stale or incomplete")

    runs, final_validation = validate_required_runs(
        read_json(args.required_runs), repository, commit
    )
    attested_assets = validate_attestations(
        read_json(args.attestations), repository, tag, commit, assets
    )
    attestation_policy = github_attestation_policy(repository, tag, commit)
    test_results = final_validation["check_test_clippy"]["test_results"]
    summary = [
        f"Annotated tag {tag} object {tag_object} peeled exactly to {commit} with both remote rows present.",
        f"Authenticated draft database ID {args.release_database_id} remained bound to tag {tag} before publication.",
        "Newest exact-commit CI and adversarial workflow runs completed successfully.",
        (
            "Exact-SHA format/check/Clippy and test gates passed: "
            f"{test_results['passed']} passed, {test_results['failed']} failed, "
            f"{test_results['ignored']} ignored, and "
            f"{test_results['external_fixture_ignored_count']} external-fixture ignored occurrences."
        ),
        "Dependency exception policy, vendored-source verifier, and final capability contradiction gate passed.",
        "Two independent clean Windows builds were byte-identical before signing.",
        (
            f"All {len(attested_assets)} immutable initial assets were hash-verified "
            "and GitHub-attested by the exact release workflow, tag ref, and source digest."
        ),
        "Downloaded BuildIdentity, FFmpeg, shader bundle, SBOM, dependency evidence, notices, and package checks passed.",
        "Authenticode remains unavailable; the executable was verified unsigned and no Authenticode claim is made.",
        "This final receipt is frozen before its own keyless Sigstore sidecar and GitHub attestation are created, avoiding circular self-claims.",
    ]
    receipt = {
        "schema_version": 1,
        "receipt_kind": "collide_o_scope_external_final_release",
        "release": {
            "repository": repository,
            "tag": tag,
            "annotated_tag_object_sha": tag_object,
            "peeled_commit_sha": commit,
            "remote_tag_row_present": True,
            "remote_peeled_row_present": True,
            "release_database_id": args.release_database_id,
            "prepublication_state": "authenticated_draft",
            "url": args.release_url,
        },
        "required_workflows": runs,
        "final_candidate_validation": final_validation,
        "reproducibility": {
            "independent_clean_builds": 2,
            "build_a_executable_sha256": reproducibility["build_a_executable_sha256"],
            "build_b_executable_sha256": reproducibility["build_b_executable_sha256"],
            "byte_identical": True,
            "authenticode": "unsigned_unavailable",
        },
        "evidence_hashes": {
            "build_identity": identity,
            "build_identity_sha256": identity["identity_sha256"],
            "cargo_lock_sha256": identity["cargo_lock_sha256"],
            "shader_bundle_sha256": identity["shader_bundle_sha256"],
            "ffmpeg_binary_sha256": identity["ffmpeg_binary_sha256"],
            "ffprobe_binary_sha256": identity["ffprobe_binary_sha256"],
            "sbom_sha256": checksums["collide-o-scope.cdx.json"],
            "dependency_inventory_sha256": checksums["dependency-license-inventory.json"],
            "dependency_review_sha256": checksums["dependency-license-review.json"],
            "checked_release_review_sha256": checksums["windows-release-license-review.toml"],
            "provenance_sha256": checksums["provenance.json"],
            "native_distribution": native_distribution,
            "vendor": vendor_hashes(),
            "source_evidence_receipts": source_evidence_receipts(),
        },
        "initial_publication": {
            "inventory_immutable": True,
            "checksum_manifest_sha256": assets["SHA256SUMS"],
            "checksum_inventory": checksums,
            "assets": attested_assets,
            "github_attestation_policy": attestation_policy,
            "provenance": provenance,
            "sigstore": {
                "subject": "SHA256SUMS",
                "subject_sha256": assets["SHA256SUMS"],
                "bundle": "SHA256SUMS.sigstore.json",
                "bundle_sha256": assets["SHA256SUMS.sigstore.json"],
                "certificate_oidc_issuer": "https://token.actions.githubusercontent.com",
                "certificate_identity": args.workflow_identity,
                "verification": "passed_before_any_downloaded_executable_ran",
            },
        },
        "downloaded_verification": report,
        "authenticode": {
            "status": "unavailable",
            "pe_signature_observed": False,
            "claim": "unsigned; Sigstore authenticates release evidence, not a PE trust chain",
        },
        "final_receipt_boundary": {
            "keyless_sigstore_sidecar": "created_and_verified_after_receipt_freeze",
            "github_attestation": {
                "lifecycle": "created_and_verified_after_receipt_freeze",
                **attestation_policy,
            },
            "uploaded_to_existing_release": "after_both_verifications",
        },
        "summary": summary,
    }
    validate_final_receipt(receipt)
    return receipt, summary


def validate_final_receipt(receipt: dict) -> None:
    if set(receipt) != {
        "schema_version", "receipt_kind", "release", "required_workflows",
        "final_candidate_validation", "reproducibility", "evidence_hashes", "initial_publication",
        "downloaded_verification", "authenticode", "final_receipt_boundary", "summary",
    }:
        raise ReceiptError("final receipt has a missing or unexpected top-level field")
    release = receipt.get("release", {})
    if not isinstance(release, dict):
        raise ReceiptError("final receipt does not prove one annotated tag boundary")
    repository = str(release.get("repository", ""))
    tag = str(release.get("tag", ""))
    commit = str(release.get("peeled_commit_sha", ""))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_kind") != "collide_o_scope_external_final_release"
        or set(release) != {
            "repository", "tag", "annotated_tag_object_sha", "peeled_commit_sha",
            "remote_tag_row_present", "remote_peeled_row_present",
            "release_database_id", "prepublication_state", "url",
        }
        or REPOSITORY.fullmatch(repository) is None
        or TAG.fullmatch(tag) is None
        or SHA40.fullmatch(str(release.get("annotated_tag_object_sha", ""))) is None
        or SHA40.fullmatch(commit) is None
        or release.get("remote_tag_row_present") is not True
        or release.get("remote_peeled_row_present") is not True
        or type(release.get("release_database_id")) is not int
        or release["release_database_id"] <= 0
        or release.get("prepublication_state") != "authenticated_draft"
        or release.get("url") != f"https://github.com/{repository}/releases/tag/{tag}"
    ):
        raise ReceiptError("final receipt does not prove one annotated tag boundary")
    runs = receipt.get("required_workflows")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ReceiptError("final receipt does not contain both required workflow runs")
    run_workflows: set[str] = set()
    for run in runs:
        if not isinstance(run, dict) or set(run) != {
            "workflow", "run_id", "run_number", "run_attempt", "url",
            "conclusion", "head_sha",
        }:
            raise ReceiptError("final receipt required workflow row is malformed")
        run_id = run.get("run_id")
        workflow = run.get("workflow")
        if (
            not isinstance(workflow, str)
            or workflow not in REQUIRED_WORKFLOWS
            or workflow in run_workflows
            or not isinstance(run_id, int)
            or run_id <= 0
            or not isinstance(run.get("run_number"), int)
            or run["run_number"] <= 0
            or not isinstance(run.get("run_attempt"), int)
            or run["run_attempt"] <= 0
            or run.get("url") != f"https://github.com/{repository}/actions/runs/{run_id}"
            or run.get("conclusion") != "success"
            or run.get("head_sha") != commit
        ):
            raise ReceiptError("final receipt required workflow row is stale")
        run_workflows.add(workflow)
    if run_workflows != REQUIRED_WORKFLOWS:
        raise ReceiptError("final receipt omits a required workflow")
    ci_run_id = next(
        run["run_id"] for run in runs if run["workflow"] == "ci.yml"
    )
    ci_run_attempt = next(
        run["run_attempt"] for run in runs if run["workflow"] == "ci.yml"
    )
    validate_final_candidate_validation(
        receipt.get("final_candidate_validation"),
        ci_run_id,
        ci_run_attempt,
        repository,
    )
    reproducibility = receipt.get("reproducibility", {})
    if (
        not isinstance(reproducibility, dict)
        or set(reproducibility) != {
            "independent_clean_builds", "build_a_executable_sha256",
            "build_b_executable_sha256", "byte_identical", "authenticode",
        }
        or
        reproducibility.get("independent_clean_builds") != 2
        or reproducibility.get("byte_identical") is not True
        or reproducibility.get("build_a_executable_sha256")
        != reproducibility.get("build_b_executable_sha256")
        or SHA256.fullmatch(
            str(reproducibility.get("build_a_executable_sha256", ""))
        )
        is None
        or reproducibility.get("authenticode") != "unsigned_unavailable"
    ):
        raise ReceiptError("final receipt does not prove byte-identical builds")
    initial = receipt.get("initial_publication", {})
    if not isinstance(initial, dict):
        raise ReceiptError("final receipt has incomplete initial asset attestations")
    assets = initial.get("assets")
    checksum_inventory = initial.get("checksum_inventory")
    provenance = initial.get("provenance")
    if (
        set(initial) != {
            "inventory_immutable", "checksum_manifest_sha256", "checksum_inventory",
            "assets", "github_attestation_policy", "provenance", "sigstore"
        }
        or
        initial.get("inventory_immutable") is not True
        or not isinstance(assets, list)
        or len(assets) != 10
    ):
        raise ReceiptError("final receipt has incomplete initial asset attestations")
    observed_assets: set[str] = set()
    for row in assets:
        if not isinstance(row, dict) or set(row) != {
            "name", "sha256", "github_attestation"
        }:
            raise ReceiptError("final receipt initial asset row is malformed")
        name = row.get("name")
        if (
            not isinstance(name, str)
            or name in observed_assets
            or SHA256.fullmatch(str(row.get("sha256", ""))) is None
            or row.get("github_attestation") != "verified"
        ):
            raise ReceiptError("final receipt has incomplete initial asset attestations")
        observed_assets.add(name)
    if observed_assets != expected_initial_names(tag):
        raise ReceiptError("final receipt initial asset inventory is not exact")
    expected_attestation_policy = github_attestation_policy(repository, tag, commit)
    if initial.get("github_attestation_policy") != expected_attestation_policy:
        raise ReceiptError("final receipt GitHub attestation policy is not exact")
    asset_hashes = {row["name"]: row["sha256"] for row in assets}
    checksummed_names = expected_initial_names(tag) - {
        "SHA256SUMS", "SHA256SUMS.sigstore.json"
    }
    if (
        not isinstance(checksum_inventory, dict)
        or set(checksum_inventory) != checksummed_names
        or any(
            not isinstance(value, str) or SHA256.fullmatch(value) is None
            for value in checksum_inventory.values()
        )
        or any(
            checksum_inventory[name] != asset_hashes[name]
            for name in checksummed_names
        )
        or checksum_manifest_digest(checksum_inventory) != asset_hashes["SHA256SUMS"]
    ):
        raise ReceiptError("final receipt checksum inventory is not exact")
    sigstore = initial.get("sigstore")
    if (
        not isinstance(sigstore, dict)
        or set(sigstore) != {
            "subject", "subject_sha256", "bundle", "bundle_sha256",
            "certificate_oidc_issuer", "certificate_identity", "verification",
        }
        or sigstore.get("subject") != "SHA256SUMS"
        or SHA256.fullmatch(str(sigstore.get("subject_sha256", ""))) is None
        or sigstore.get("bundle") != "SHA256SUMS.sigstore.json"
        or SHA256.fullmatch(str(sigstore.get("bundle_sha256", ""))) is None
        or sigstore.get("certificate_oidc_issuer")
        != "https://token.actions.githubusercontent.com"
        or sigstore.get("certificate_identity")
        != f"https://github.com/{repository}/.github/workflows/release-trust.yml@refs/tags/{tag}"
        or sigstore.get("verification")
        != "passed_before_any_downloaded_executable_ran"
        or sigstore.get("subject_sha256") != asset_hashes["SHA256SUMS"]
        or sigstore.get("bundle_sha256")
        != asset_hashes["SHA256SUMS.sigstore.json"]
        or initial.get("checksum_manifest_sha256") != asset_hashes["SHA256SUMS"]
    ):
        raise ReceiptError("final receipt Sigstore evidence is incomplete")
    evidence = receipt.get("evidence_hashes")
    required_hashes = {
        "build_identity_sha256", "cargo_lock_sha256", "shader_bundle_sha256",
        "ffmpeg_binary_sha256", "ffprobe_binary_sha256", "sbom_sha256",
        "dependency_inventory_sha256", "dependency_review_sha256",
        "checked_release_review_sha256", "provenance_sha256",
    }
    if not isinstance(evidence, dict) or set(evidence) != required_hashes | {
        "build_identity", "native_distribution", "vendor", "source_evidence_receipts"
    }:
        raise ReceiptError("final receipt evidence hash inventory is incomplete")
    if any(SHA256.fullmatch(str(evidence.get(field, ""))) is None for field in required_hashes):
        raise ReceiptError("final receipt evidence hash is malformed")
    evidence_assets = {
        "sbom_sha256": "collide-o-scope.cdx.json",
        "dependency_inventory_sha256": "dependency-license-inventory.json",
        "dependency_review_sha256": "dependency-license-review.json",
        "checked_release_review_sha256": "windows-release-license-review.toml",
        "provenance_sha256": "provenance.json",
    }
    if any(
        evidence[field] != checksum_inventory[name]
        or evidence[field] != asset_hashes[name]
        for field, name in evidence_assets.items()
    ):
        raise ReceiptError("final receipt evidence hashes are not bound to initial assets")
    policy = checked_release_policy()
    if evidence["checked_release_review_sha256"] != policy["sha256"]:
        raise ReceiptError("final receipt checked review differs from tagged source")
    vendor = evidence.get("vendor")
    if vendor != vendor_hashes():
        raise ReceiptError("final receipt vendor hashes are incomplete")
    validate_build_identity(
        evidence.get("build_identity"),
        tag,
        commit,
        evidence,
        evidence.get("native_distribution", {}),
    )
    if evidence.get("native_distribution") != policy["native_distribution"]:
        raise ReceiptError("final receipt native distribution differs from tagged policy")
    if (
        not isinstance(provenance, dict)
        or set(provenance) != {
            "schema_version", "tag", "commit", "source_date_epoch", "build_identity",
            "reproducibility", "artifacts", "authenticode", "signing_order",
        }
        or provenance.get("schema_version") != 1
        or provenance.get("tag") != tag
        or provenance.get("commit") != commit
        or not isinstance(provenance.get("source_date_epoch"), int)
        or provenance["source_date_epoch"] <= 0
        or provenance.get("build_identity") != evidence.get("build_identity")
        or provenance.get("authenticode") != policy["authenticode"]
        or provenance.get("signing_order")
        != "unsigned builds compared; Sigstore signs checksum/provenance material; no Authenticode claim"
        or json_document_digest(provenance) != evidence["provenance_sha256"]
    ):
        raise ReceiptError("final receipt provenance associations are contradictory")
    provenance_reproducibility = provenance.get("reproducibility")
    if (
        not isinstance(provenance_reproducibility, dict)
        or set(provenance_reproducibility) != {
            "independent_clean_builds", "build_a_executable_sha256",
            "build_b_executable_sha256", "byte_identical", "authenticode",
        }
        or provenance_reproducibility != reproducibility
    ):
        raise ReceiptError("final receipt provenance reproducibility is contradictory")
    provenance_artifacts = provenance.get("artifacts")
    expected_provenance_artifacts = checksummed_names - {"provenance.json"}
    if (
        not isinstance(provenance_artifacts, dict)
        or set(provenance_artifacts) != expected_provenance_artifacts
        or any(
            provenance_artifacts[name] != checksum_inventory[name]
            for name in expected_provenance_artifacts
        )
    ):
        raise ReceiptError("final receipt provenance artifact inventory is contradictory")
    source_receipts = evidence.get("source_evidence_receipts")
    source_paths = {
        row.get("path")
        for row in source_receipts
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    } if isinstance(source_receipts, list) else set()
    if (
        not isinstance(source_receipts, list)
        or not 2 <= len(source_receipts) <= 32
        or len(source_paths) != len(source_receipts)
        or "docs/evidence/v1.7.0-improvement-audit-release-receipt.md"
        not in source_paths
        or "docs/evidence/v1.7.1-release-recovery-receipt.md"
        not in source_paths
        or "docs/evidence/v1.7.2-release-recovery-receipt.md"
        not in source_paths
        or "docs/evidence/v1.7.3-release-recovery-receipt.md"
        not in source_paths
        or "docs/evidence/v1.7.4-release-recovery-receipt.md"
        not in source_paths
        or "docs/evidence/v1.8.0-ffmpeg-9-software-baseline-receipt.md"
        not in source_paths
        or "docs/evidence/v1.8.1-patch-refresh-receipt.md" not in source_paths
        or not any(
            isinstance(path, str) and path.startswith("docs/evidence/p10")
            for path in source_paths
        )
        or any(
            not isinstance(row, dict)
            or set(row) != {"path", "sha256"}
            or not isinstance(row.get("path"), str)
            or not row["path"].startswith("docs/evidence/")
            or SAFE_NAME.fullmatch(row["path"].removeprefix("docs/evidence/")) is None
            or SHA256.fullmatch(str(row.get("sha256", ""))) is None
            or not (ROOT / row["path"]).is_file()
            or (ROOT / row["path"]).is_symlink()
            or digest(ROOT / row["path"]) != row["sha256"]
            for row in source_receipts
        )
        or source_receipts != source_evidence_receipts()
    ):
        raise ReceiptError("final receipt source-evidence reconciliation is incomplete")
    report = receipt.get("downloaded_verification")
    package_name = f"collide-o-scope-{tag}-windows-x86_64.zip"
    version_value = report.get("version_json") if isinstance(report, dict) else None
    package_value = report.get("package") if isinstance(report, dict) else None
    ffmpeg_value = report.get("ffmpeg") if isinstance(report, dict) else None
    shader_value = report.get("shader") if isinstance(report, dict) else None
    sbom_value = report.get("sbom") if isinstance(report, dict) else None
    dependency_value = report.get("dependency_evidence") if isinstance(report, dict) else None
    version_json = version_value if isinstance(version_value, dict) else {}
    package = package_value if isinstance(package_value, dict) else {}
    ffmpeg = ffmpeg_value if isinstance(ffmpeg_value, dict) else {}
    shader = shader_value if isinstance(shader_value, dict) else {}
    sbom = sbom_value if isinstance(sbom_value, dict) else {}
    dependency = dependency_value if isinstance(dependency_value, dict) else {}
    if (
        not isinstance(report, dict)
        or set(report) != {
            "schema_version", "release_verified", "tag", "commit", "version_json",
            "package", "ffmpeg", "shader", "sbom", "dependency_evidence",
            "authenticode",
        }
        or report.get("schema_version") != 1
        or report.get("release_verified") is not True
        or report.get("tag") != tag
        or report.get("commit") != commit
        or version_json.get("status") != "passed"
        or set(version_json) != {
            "status", "identity_sha256", "version", "git_sha", "published_artifact"
        }
        or version_json.get("identity_sha256") != evidence["build_identity_sha256"]
        or version_json.get("version") != tag.removeprefix("v")
        or version_json.get("git_sha") != commit
        or version_json.get("published_artifact") is not True
        or package.get("status") != "passed"
        or set(package) != {
            "status", "name", "sha256", "source_archive_name",
            "source_archive_sha256", "entry_count", "executable_sha256",
            "source_archive_reproduced", "required_notice_sha256",
        }
        or package.get("name") != package_name
        or package.get("sha256") != asset_hashes[package_name]
        or package.get("source_archive_name")
        != f"collide-o-scope-{tag}-source.zip"
        or package.get("source_archive_sha256")
        != asset_hashes[f"collide-o-scope-{tag}-source.zip"]
        or package.get("executable_sha256")
        != reproducibility["build_a_executable_sha256"]
        or package.get("source_archive_reproduced") is not True
        or not isinstance(package.get("entry_count"), int)
        or package["entry_count"] <= 0
        or not isinstance(package.get("required_notice_sha256"), dict)
        or set(package["required_notice_sha256"])
        != {
            "LICENSE", "COPYRIGHT.md", "FFMPEG-BUILDCONF.txt",
            "FFMPEG-README.txt", "LICENSES/FFmpeg-GPL-3.0-or-later.txt",
        }
        or any(
            SHA256.fullmatch(str(value)) is None
            for value in package["required_notice_sha256"].values()
        )
        or package["required_notice_sha256"].get("LICENSE")
        != digest(ROOT / "LICENSE")
        or package["required_notice_sha256"].get("COPYRIGHT.md")
        != digest(ROOT / "COPYRIGHT.md")
        or ffmpeg.get("status") != "passed"
        or set(ffmpeg) != {
            "status", "version", "binary_sha256", "ffprobe_sha256",
            "archive_sha256", "source_commit", "buildconf_sha256",
        }
        or ffmpeg.get("binary_sha256") != evidence["ffmpeg_binary_sha256"]
        or ffmpeg.get("ffprobe_sha256") != evidence["ffprobe_binary_sha256"]
        or SHA256.fullmatch(str(ffmpeg.get("archive_sha256", ""))) is None
        or SHA256.fullmatch(str(ffmpeg.get("buildconf_sha256", ""))) is None
        or SHA40.fullmatch(str(ffmpeg.get("source_commit", ""))) is None
        or shader.get("status") != "passed"
        or set(shader) != {"status", "bundle_sha256"}
        or shader.get("bundle_sha256") != evidence["shader_bundle_sha256"]
        or sbom.get("status") != "passed"
        or set(sbom) != {"status", "sha256"}
        or sbom.get("sha256") != evidence["sbom_sha256"]
        or dependency.get("status") != "passed"
        or set(dependency) != {
            "status", "inventory_sha256", "review_sha256", "checked_review_sha256"
        }
        or dependency.get("inventory_sha256")
        != evidence["dependency_inventory_sha256"]
        or dependency.get("review_sha256") != evidence["dependency_review_sha256"]
        or dependency.get("checked_review_sha256")
        != evidence["checked_release_review_sha256"]
        or report.get("authenticode") != "unavailable_and_unsigned_verified"
    ):
        raise ReceiptError("final receipt downloaded verification is incomplete")
    validate_native_distribution(
        evidence.get("native_distribution"),
        ffmpeg,
        package["required_notice_sha256"],
        checksum_inventory,
        policy,
    )
    authenticode = receipt.get("authenticode")
    if authenticode != {
        "status": "unavailable",
        "pe_signature_observed": False,
        "claim": "unsigned; Sigstore authenticates release evidence, not a PE trust chain",
    }:
        raise ReceiptError("final receipt overclaims Authenticode")
    if receipt.get("final_receipt_boundary") != {
        "keyless_sigstore_sidecar": "created_and_verified_after_receipt_freeze",
        "github_attestation": {
            "lifecycle": "created_and_verified_after_receipt_freeze",
            **expected_attestation_policy,
        },
        "uploaded_to_existing_release": "after_both_verifications",
    }:
        raise ReceiptError("final receipt self-signing boundary is misstated")
    summary = receipt.get("summary")
    if (
        not isinstance(summary, list)
        or not 1 <= len(summary) <= 16
        or any(not isinstance(line, str) or not 1 <= len(line) <= 512 for line in summary)
    ):
        raise ReceiptError("final release summary is absent or unbounded")


def write_outputs(output: Path, summary_output: Path, receipt: dict, summary: list[str]) -> None:
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ReceiptError("final release receipt exceeds its bounded size")
    for path in (output, summary_output):
        if path.exists():
            raise ReceiptError(f"refusing to overwrite final release output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    summary_output.write_text(
        "# Collide-o-scope final release trust summary\n\n"
        + "".join(f"- {line}\n" for line in summary),
        encoding="utf-8",
        newline="\n",
    )


def expect_error(action, fragment: str) -> None:
    try:
        action()
    except ReceiptError as error:
        if fragment not in str(error):
            raise ReceiptError(
                f"self-test expected {fragment!r}, received {str(error)!r}"
            ) from error
    else:
        raise ReceiptError(f"self-test accepted mutation: {fragment}")


def self_test() -> None:
    value_hash = lambda label: hashlib.sha256(label.encode("utf-8")).hexdigest()
    sha = value_hash("generic")
    tag = "v1.7.0"
    commit = "2" * 40
    policy = checked_release_policy()
    native = policy["native_distribution"]
    executable_sha = value_hash("executable")
    ffmpeg_binary_sha = value_hash("ffmpeg")
    ffprobe_binary_sha = value_hash("ffprobe")
    cargo_lock_sha = digest(ROOT / "Cargo.lock")
    shader_sha = shader_bundle_digest()
    identity = {
        "schema_version": 1,
        "package_name": "collide-o-scope",
        "version": "1.7.0",
        "git_sha": commit,
        "git_dirty": False,
        "profile": "release",
        "target": "x86_64-pc-windows-msvc",
        "enabled_features": "default",
        "rustc_vv": "rustc fixture",
        "cargo_version": "cargo fixture",
        "linker_identity": "link.exe;fixture",
        "sdk_identity": "windows-sdk:fixture;msvc-tools:fixture",
        "ffmpeg_libraries": f"ffmpeg={native['version']}",
        "ffmpeg_binary_version": f"ffmpeg version {native['version']}",
        "ffmpeg_binary_sha256": ffmpeg_binary_sha,
        "ffprobe_binary_version": f"ffprobe version {native['version']}",
        "ffprobe_binary_sha256": ffprobe_binary_sha,
        "shader_bundle_sha256": shader_sha,
        "cargo_lock_sha256": cargo_lock_sha,
        "published_artifact": True,
    }
    identity["identity_sha256"] = hashlib.sha256(identity_payload(identity)).hexdigest()
    package_name = f"collide-o-scope-{tag}-windows-x86_64.zip"
    source_name = f"collide-o-scope-{tag}-source.zip"
    checksum_inventory = {
        package_name: value_hash("package"),
        source_name: value_hash("source"),
        "collide-o-scope.cdx.json": value_hash("sbom"),
        "dependency-license-inventory.json": value_hash("inventory"),
        "dependency-license-review.json": value_hash("review"),
        "ffmpeg-buildconf.txt": native["buildconf_sha256"],
        "windows-release-license-review.toml": policy["sha256"],
    }
    reproducibility = {
        "independent_clean_builds": 2,
        "build_a_executable_sha256": executable_sha,
        "build_b_executable_sha256": executable_sha,
        "byte_identical": True,
        "authenticode": "unsigned_unavailable",
    }
    provenance = {
        "schema_version": 1,
        "tag": tag,
        "commit": commit,
        "source_date_epoch": 1,
        "build_identity": identity,
        "reproducibility": reproducibility,
        "artifacts": dict(checksum_inventory),
        "authenticode": policy["authenticode"],
        "signing_order": "unsigned builds compared; Sigstore signs checksum/provenance material; no Authenticode claim",
    }
    provenance_sha = json_document_digest(provenance)
    checksum_inventory["provenance.json"] = provenance_sha
    manifest_sha = checksum_manifest_digest(checksum_inventory)
    bundle_sha = value_hash("checksum-sigstore-bundle")
    asset_hashes = {
        **checksum_inventory,
        "SHA256SUMS": manifest_sha,
        "SHA256SUMS.sigstore.json": bundle_sha,
    }
    attestation_policy = github_attestation_policy("acme/project", tag, commit)
    attestation_document = {
        "schema_version": 2,
        "repository": "acme/project",
        "commit": commit,
        "policy": attestation_policy,
        "assets": [
            {"name": name, "sha256": asset_hashes[name], "verified": True}
            for name in sorted(asset_hashes)
        ],
    }
    validate_attestations(
        attestation_document, "acme/project", tag, commit, asset_hashes
    )
    for field, value in (
        ("repository", "acme/other"),
        ("certificate_identity", "https://github.com/acme/project/.github/workflows/other.yml@refs/tags/v1.7.0"),
        ("certificate_oidc_issuer", "https://issuer.invalid"),
        ("predicate_type", "https://example.invalid/predicate"),
        ("source_ref", "refs/heads/main"),
        ("source_digest", "3" * 40),
    ):
        mutated_attestations = copy.deepcopy(attestation_document)
        mutated_attestations["policy"][field] = value
        expect_error(
            lambda mutated_attestations=mutated_attestations: validate_attestations(
                mutated_attestations, "acme/project", tag, commit, asset_hashes
            ),
            "signer policy",
        )
    missing_attestation_policy = copy.deepcopy(attestation_document)
    missing_attestation_policy["policy"].pop("source_digest")
    expect_error(
        lambda: validate_attestations(
            missing_attestation_policy, "acme/project", tag, commit, asset_hashes
        ),
        "signer policy",
    )
    platform_jobs = ["Linux (Ubuntu 24.04)", "macOS 15", "Windows (VS 2022)"]
    final_validation = {
        "ci_run_id": 2,
        "ci_run_attempt": 1,
        "format": {
            "commands": [
                "cargo fmt --all -- --check",
                "node --check static/app.js",
                "node --check docs/ui-ux/wireframe.js",
            ],
            "steps": [
                {
                    "job": job,
                    "step": "Check Rust formatting and JavaScript syntax",
                    "conclusion": "success",
                }
                for job in platform_jobs
            ],
            "conclusion": "success",
        },
        "check_test_clippy": {
            "commands": [
                "cargo check --locked --all-targets --all-features",
                "cargo test --locked --all-targets --all-features",
                "cargo clippy --locked --all-targets --all-features -- -D warnings",
            ],
            "steps": [
                {
                    "job": job,
                    "step": (
                        "Check, test, and lint on Windows"
                        if job.startswith("Windows")
                        else "Check, test, and lint on Unix"
                    ),
                    "conclusion": "success",
                }
                for job in platform_jobs
            ],
            "conclusion": "success",
            "test_results": {
                "summary_records": 3,
                "passed": 300,
                "failed": 0,
                "ignored": 3,
                "measured": 0,
                "filtered_out": 0,
                "ignored_test_names": ["video::external_ffmpeg_fixture"],
                "external_fixture_ignored_count": 1,
                "external_fixture_ignored_names": ["video::external_ffmpeg_fixture"],
                "platform_jobs": [
                    {
                        "job": job,
                        "job_id": 100 + index,
                        "logs_url": (
                            "https://api.github.com/repos/acme/project/actions/jobs/"
                            f"{100 + index}/logs"
                        ),
                        "summary_records": 1,
                        "passed": 100,
                        "failed": 0,
                        "ignored": 3 if job == "Linux (Ubuntu 24.04)" else 0,
                        "measured": 0,
                        "filtered_out": 0,
                        "ignored_test_names": (
                            ["video::external_ffmpeg_fixture"]
                            if job == "Linux (Ubuntu 24.04)"
                            else []
                        ),
                        "external_fixture_ignored_count": (
                            1 if job == "Linux (Ubuntu 24.04)" else 0
                        ),
                        "external_fixture_ignored_names": (
                            ["video::external_ffmpeg_fixture"]
                            if job == "Linux (Ubuntu 24.04)"
                            else []
                        ),
                    }
                    for index, job in enumerate(platform_jobs)
                ],
                "source": "selected exact-SHA CI platform job logs",
            },
        },
        "capability_registry_contradiction_gate": {
            "command": "cargo run --locked --bin generate_capabilities -- --check",
            "steps": [
                {
                    "job": job,
                    "step": "Check generated capability registry",
                    "conclusion": "success",
                }
                for job in platform_jobs
            ],
            "conclusion": "success",
        },
        "dependency_exception_policy": {
            "command": "python scripts/check-dependency-policy.py",
            "steps": [
                {
                    "job": "Dependency policy and supply-chain provenance",
                    "step": "Reject stale or unowned advisory exceptions",
                    "conclusion": "success",
                }
            ],
            "conclusion": "success",
        },
        "vendor_verifier": {
            "command": "python scripts/verify-vendored-wgpu-hal.py --self-test",
            "steps": [
                {
                    "job": job,
                    "step": "Verify the vendored wgpu-hal archive and sole patch",
                    "conclusion": "success",
                }
                for job in platform_jobs
            ]
            + [
                {
                    "job": "Dependency policy and supply-chain provenance",
                    "step": "Fetch locked dependencies and verify vendored source",
                    "conclusion": "success",
                }
            ],
            "conclusion": "success",
        },
    }
    receipt = {
        "schema_version": 1,
        "receipt_kind": "collide_o_scope_external_final_release",
        "release": {
            "repository": "acme/project",
            "tag": tag,
            "annotated_tag_object_sha": "1" * 40,
            "peeled_commit_sha": commit,
            "remote_tag_row_present": True,
            "remote_peeled_row_present": True,
            "release_database_id": 314159,
            "prepublication_state": "authenticated_draft",
            "url": "https://github.com/acme/project/releases/tag/v1.7.0",
        },
        "required_workflows": [
            {
                "workflow": workflow,
                "run_id": index,
                "run_number": index,
                "run_attempt": 1,
                "url": f"https://github.com/acme/project/actions/runs/{index}",
                "conclusion": "success",
                "head_sha": commit,
            }
            for index, workflow in enumerate(sorted(REQUIRED_WORKFLOWS), start=1)
        ],
        "final_candidate_validation": final_validation,
        "reproducibility": reproducibility,
        "evidence_hashes": {
            "build_identity": identity,
            "build_identity_sha256": identity["identity_sha256"],
            "cargo_lock_sha256": cargo_lock_sha,
            "shader_bundle_sha256": shader_sha,
            "ffmpeg_binary_sha256": ffmpeg_binary_sha,
            "ffprobe_binary_sha256": ffprobe_binary_sha,
            "sbom_sha256": checksum_inventory["collide-o-scope.cdx.json"],
            "dependency_inventory_sha256": checksum_inventory["dependency-license-inventory.json"],
            "dependency_review_sha256": checksum_inventory["dependency-license-review.json"],
            "checked_release_review_sha256": policy["sha256"],
            "provenance_sha256": provenance_sha,
            "native_distribution": native,
            "vendor": vendor_hashes(),
            "source_evidence_receipts": source_evidence_receipts(),
        },
        "initial_publication": {
            "inventory_immutable": True,
            "checksum_manifest_sha256": manifest_sha,
            "checksum_inventory": checksum_inventory,
            "assets": [
                {
                    "name": name,
                    "sha256": asset_hashes[name],
                    "github_attestation": "verified",
                }
                for name in sorted(expected_initial_names(tag))
            ],
            "github_attestation_policy": attestation_policy,
            "provenance": provenance,
            "sigstore": {
                "subject": "SHA256SUMS",
                "subject_sha256": manifest_sha,
                "bundle": "SHA256SUMS.sigstore.json",
                "bundle_sha256": bundle_sha,
                "certificate_oidc_issuer": "https://token.actions.githubusercontent.com",
                "certificate_identity": "https://github.com/acme/project/.github/workflows/release-trust.yml@refs/tags/v1.7.0",
                "verification": "passed_before_any_downloaded_executable_ran",
            },
        },
        "downloaded_verification": {
            "schema_version": 1,
            "release_verified": True,
            "tag": tag,
            "commit": commit,
            "version_json": {
                "status": "passed",
                "identity_sha256": identity["identity_sha256"],
                "version": "1.7.0",
                "git_sha": commit,
                "published_artifact": True,
            },
            "package": {
                "status": "passed",
                "name": package_name,
                "sha256": checksum_inventory[package_name],
                "source_archive_name": source_name,
                "source_archive_sha256": checksum_inventory[source_name],
                "executable_sha256": executable_sha,
                "source_archive_reproduced": True,
                "entry_count": 12,
                "required_notice_sha256": {
                    "LICENSE": digest(ROOT / "LICENSE"),
                    "COPYRIGHT.md": digest(ROOT / "COPYRIGHT.md"),
                    "FFMPEG-BUILDCONF.txt": native["buildconf_sha256"],
                    "FFMPEG-README.txt": native["distribution_readme_sha256"],
                    "LICENSES/FFmpeg-GPL-3.0-or-later.txt": native["distribution_license_sha256"],
                },
            },
            "ffmpeg": {
                "status": "passed",
                "version": native["version"],
                "binary_sha256": ffmpeg_binary_sha,
                "ffprobe_sha256": ffprobe_binary_sha,
                "archive_sha256": native["archive_sha256"],
                "source_commit": native["source_commit"],
                "buildconf_sha256": native["buildconf_sha256"],
            },
            "shader": {"status": "passed", "bundle_sha256": shader_sha},
            "sbom": {
                "status": "passed",
                "sha256": checksum_inventory["collide-o-scope.cdx.json"],
            },
            "dependency_evidence": {
                "status": "passed",
                "inventory_sha256": checksum_inventory["dependency-license-inventory.json"],
                "review_sha256": checksum_inventory["dependency-license-review.json"],
                "checked_review_sha256": policy["sha256"],
            },
            "authenticode": "unavailable_and_unsigned_verified",
        },
        "authenticode": {
            "status": "unavailable",
            "pe_signature_observed": False,
            "claim": "unsigned; Sigstore authenticates release evidence, not a PE trust chain",
        },
        "final_receipt_boundary": {
            "keyless_sigstore_sidecar": "created_and_verified_after_receipt_freeze",
            "github_attestation": {
                "lifecycle": "created_and_verified_after_receipt_freeze",
                **attestation_policy,
            },
            "uploaded_to_existing_release": "after_both_verifications",
        },
        "summary": ["bounded summary"],
    }
    validate_final_receipt(receipt)
    mutations = (
        ("release", "remote_peeled_row_present", False, "annotated tag"),
        ("release", "annotated_tag_object_sha", "", "annotated tag"),
        ("release", "release_database_id", 0, "annotated tag"),
        ("release", "prepublication_state", "published", "annotated tag"),
        ("reproducibility", "byte_identical", False, "byte-identical"),
        ("initial_publication", "inventory_immutable", False, "initial asset"),
        ("authenticode", "status", "available", "Authenticode"),
    )
    for section, field, value, fragment in mutations:
        mutated = copy.deepcopy(receipt)
        mutated[section][field] = value
        expect_error(lambda mutated=mutated: validate_final_receipt(mutated), fragment)
    mutated = copy.deepcopy(receipt)
    mutated["initial_publication"]["assets"].pop()
    expect_error(lambda: validate_final_receipt(mutated), "initial asset")
    for field, value in (
        ("certificate_identity", "https://github.com/acme/project/.github/workflows/other.yml@refs/tags/v1.7.0"),
        ("source_ref", "refs/heads/main"),
        ("source_digest", "3" * 40),
    ):
        mutated = copy.deepcopy(receipt)
        mutated["initial_publication"]["github_attestation_policy"][field] = value
        expect_error(lambda mutated=mutated: validate_final_receipt(mutated), "attestation policy")
        mutated = copy.deepcopy(receipt)
        mutated["final_receipt_boundary"]["github_attestation"][field] = value
        expect_error(lambda mutated=mutated: validate_final_receipt(mutated), "self-signing")
    mutated = copy.deepcopy(receipt)
    mutated["summary"] = ["x" * 513]
    expect_error(lambda: validate_final_receipt(mutated), "unbounded")
    for field, fragment in (
        ("required_workflows", "workflow"),
        ("final_candidate_validation", "validation"),
        ("evidence_hashes", "evidence hash"),
        ("downloaded_verification", "downloaded verification"),
        ("final_receipt_boundary", "self-signing"),
    ):
        mutated = copy.deepcopy(receipt)
        mutated[field] = []
        expect_error(lambda mutated=mutated: validate_final_receipt(mutated), fragment)
    mutated = copy.deepcopy(receipt)
    mutated["initial_publication"]["sigstore"] = {}
    expect_error(lambda: validate_final_receipt(mutated), "Sigstore")
    mutated = copy.deepcopy(receipt)
    mutated["initial_publication"]["checksum_manifest_sha256"] = "b" * 64
    expect_error(lambda: validate_final_receipt(mutated), "Sigstore")
    mutated = copy.deepcopy(receipt)
    mutated["downloaded_verification"]["package"]["sha256"] = "b" * 64
    expect_error(lambda: validate_final_receipt(mutated), "downloaded verification")
    mutated = copy.deepcopy(receipt)
    mutated["downloaded_verification"]["version_json"]["identity_sha256"] = "b" * 64
    expect_error(lambda: validate_final_receipt(mutated), "downloaded verification")
    mutated = copy.deepcopy(receipt)
    mutated["authenticode"]["pe_signature_observed"] = True
    expect_error(lambda: validate_final_receipt(mutated), "Authenticode")
    mutated = copy.deepcopy(receipt)
    mutated["reproducibility"]["build_a_executable_sha256"] = "not-a-digest"
    mutated["reproducibility"]["build_b_executable_sha256"] = "not-a-digest"
    expect_error(lambda: validate_final_receipt(mutated), "byte-identical")
    mutated = copy.deepcopy(receipt)
    mutated["downloaded_verification"]["package"]["unexpected"] = True
    expect_error(lambda: validate_final_receipt(mutated), "downloaded verification")
    mutated = copy.deepcopy(receipt)
    mutated["initial_publication"]["assets"][0] = "not-an-asset-row"
    expect_error(lambda: validate_final_receipt(mutated), "asset row")

    bad_sha = value_hash("one-field mutation")
    asset_indexes = {
        row["name"]: index
        for index, row in enumerate(receipt["initial_publication"]["assets"])
    }

    def set_nested(document: dict, path: tuple[object, ...], value: object) -> None:
        target: object = document
        for component in path[:-1]:
            if isinstance(target, dict):
                target = target[component]
            elif isinstance(target, list) and isinstance(component, int):
                target = target[component]
            else:
                raise AssertionError(f"invalid mutation path: {path}")
        leaf = path[-1]
        if isinstance(target, dict):
            target[leaf] = value
        elif isinstance(target, list) and isinstance(leaf, int):
            target[leaf] = value
        else:
            raise AssertionError(f"invalid mutation leaf: {path}")

    one_field_mutations = (
        (
            ("initial_publication", "assets", asset_indexes["collide-o-scope.cdx.json"], "sha256"),
            bad_sha,
            "checksum inventory",
            "SBOM initial asset SHA",
        ),
        (
            ("initial_publication", "assets", asset_indexes[source_name], "sha256"),
            bad_sha,
            "checksum inventory",
            "source archive initial asset SHA",
        ),
        (
            (
                "initial_publication", "assets",
                asset_indexes["dependency-license-inventory.json"], "sha256",
            ),
            bad_sha,
            "checksum inventory",
            "dependency inventory initial asset SHA",
        ),
        (
            (
                "initial_publication", "assets",
                asset_indexes["dependency-license-review.json"], "sha256",
            ),
            bad_sha,
            "checksum inventory",
            "dependency review initial asset SHA",
        ),
        (
            ("initial_publication", "assets", asset_indexes["ffmpeg-buildconf.txt"], "sha256"),
            bad_sha,
            "checksum inventory",
            "FFmpeg buildconf initial asset SHA",
        ),
        (
            ("initial_publication", "assets", asset_indexes["provenance.json"], "sha256"),
            bad_sha,
            "checksum inventory",
            "provenance initial asset SHA",
        ),
        (
            ("evidence_hashes", "source_evidence_receipts", 0, "sha256"),
            bad_sha,
            "source-evidence",
            "source evidence receipt SHA",
        ),
        (
            ("downloaded_verification", "version_json", "version"),
            "9.9.9",
            "downloaded verification",
            "downloaded version/tag association",
        ),
        (
            ("downloaded_verification", "ffmpeg", "source_commit"),
            "3" * 40,
            "FFmpeg",
            "FFmpeg source commit",
        ),
        (
            ("downloaded_verification", "ffmpeg", "archive_sha256"),
            bad_sha,
            "FFmpeg",
            "FFmpeg archive SHA",
        ),
        (
            ("downloaded_verification", "ffmpeg", "buildconf_sha256"),
            bad_sha,
            "FFmpeg",
            "FFmpeg buildconf SHA",
        ),
        (
            ("evidence_hashes", "cargo_lock_sha256"),
            bad_sha,
            "BuildIdentity",
            "Cargo.lock evidence SHA",
        ),
        (
            ("evidence_hashes", "shader_bundle_sha256"),
            bad_sha,
            "BuildIdentity",
            "shader evidence SHA",
        ),
        (
            ("evidence_hashes", "vendor", "archive_sha256"),
            bad_sha,
            "vendor",
            "vendor archive SHA",
        ),
        (
            ("downloaded_verification", "package", "required_notice_sha256", "LICENSE"),
            bad_sha,
            "downloaded verification",
            "package LICENSE notice SHA",
        ),
        (
            (
                "downloaded_verification", "package", "required_notice_sha256",
                "FFMPEG-README.txt",
            ),
            bad_sha,
            "FFmpeg",
            "package FFmpeg notice SHA",
        ),
        (
            ("evidence_hashes", "build_identity", "version"),
            "9.9.9",
            "BuildIdentity",
            "BuildIdentity version",
        ),
        (
            ("initial_publication", "checksum_inventory", package_name),
            bad_sha,
            "checksum inventory",
            "checksum inventory package SHA",
        ),
        (
            ("initial_publication", "provenance", "artifacts", source_name),
            bad_sha,
            "provenance",
            "provenance source archive SHA",
        ),
        (
            ("downloaded_verification", "package", "source_archive_sha256"),
            bad_sha,
            "downloaded verification",
            "downloaded source archive SHA",
        ),
        (
            ("downloaded_verification", "dependency_evidence", "review_sha256"),
            bad_sha,
            "downloaded verification",
            "downloaded dependency review SHA",
        ),
    )
    for path, value, fragment, label in one_field_mutations:
        mutated = copy.deepcopy(receipt)
        set_nested(mutated, path, value)
        try:
            validate_final_receipt(mutated)
        except ReceiptError as error:
            if fragment not in str(error):
                raise ReceiptError(
                    f"self-test mutation {label!r} expected {fragment!r}, "
                    f"received {str(error)!r}"
                ) from error
        else:
            raise ReceiptError(f"self-test accepted one-field mutation: {label}")

    mutated = copy.deepcopy(receipt)
    mutated["evidence_hashes"]["source_evidence_receipts"].pop()
    expect_error(lambda: validate_final_receipt(mutated), "source-evidence")
    mutated = copy.deepcopy(receipt)
    mutated["final_candidate_validation"]["format"]["steps"][0]["job"] = (
        "Arbitrary successful runner"
    )
    expect_error(lambda: validate_final_receipt(mutated), "step receipt")
    mutated = copy.deepcopy(receipt)
    mutated["final_candidate_validation"]["check_test_clippy"]["test_results"][
        "platform_jobs"
    ][0]["job"] = "Arbitrary successful runner"
    expect_error(lambda: validate_final_receipt(mutated), "per-platform")
    mutated = copy.deepcopy(receipt)
    rows = mutated["final_candidate_validation"]["check_test_clippy"]["test_results"][
        "platform_jobs"
    ]
    rows[0]["summary_records"] = 3
    rows[1]["summary_records"] = 0
    rows[2]["summary_records"] = 0
    expect_error(lambda: validate_final_receipt(mutated), "per-platform")
    print("final release receipt self-test passed: critical mutations fail closed")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--directory", type=Path, required=True)
    build.add_argument("--verification-report", type=Path, required=True)
    build.add_argument("--attestations", type=Path, required=True)
    build.add_argument("--required-runs", type=Path, required=True)
    build.add_argument("--tag", required=True)
    build.add_argument("--commit", required=True)
    build.add_argument("--tag-object", required=True)
    build.add_argument("--repository", required=True)
    build.add_argument("--release-url", required=True)
    build.add_argument("--release-database-id", required=True, type=int)
    build.add_argument("--workflow-identity", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--summary-output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    commands.add_parser("self-test")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "build":
            receipt, summary = build_receipt(args)
            write_outputs(args.output.resolve(), args.summary_output.resolve(), receipt, summary)
            print(json.dumps({"final_release_receipt": str(args.output), "valid": True}))
        elif args.command == "validate":
            if args.receipt.stat().st_size > MAX_RECEIPT_BYTES:
                raise ReceiptError("final release receipt exceeds its bounded size")
            validate_final_receipt(read_json(args.receipt))
            print(json.dumps({"final_release_receipt": str(args.receipt), "valid": True}))
        else:
            self_test()
    except (OSError, UnicodeError, ReceiptError) as error:
        print(f"final release receipt failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
