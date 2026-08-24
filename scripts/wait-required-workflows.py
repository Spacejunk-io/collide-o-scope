#!/usr/bin/env python3
"""Fail closed until the newest exact-commit verification runs are green."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


SHA = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
WORKFLOW = re.compile(r"^[A-Za-z0-9_.-]+\.(?:yml|yaml)$")
RUN_URL = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/actions/runs/[0-9]+$")
REQUIRED_WORKFLOWS = {"ci.yml", "adversarial.yml"}
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
REQUIRED_JOB_NAMES = set(PLATFORM_TEST_STEPS) | {DEPENDENCY_JOB}
MAX_RECEIPT_BYTES = 64 * 1024
MAX_LOG_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_LOG_UNCOMPRESSED_BYTES = 256 * 1024 * 1024


class GateError(RuntimeError):
    pass


def newest_exact_run(payload: object, commit: str) -> dict | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
        raise GateError("GitHub returned an unsupported workflow-runs document")
    exact = [
        run
        for run in payload["workflow_runs"]
        if isinstance(run, dict) and run.get("head_sha") == commit
    ]
    if not exact:
        return None
    return max(
        exact,
        key=lambda run: (
            int(run.get("run_number") or 0),
            int(run.get("run_attempt") or 0),
            int(run.get("id") or 0),
        ),
    )


def run_state(run: dict | None) -> str:
    if run is None:
        return "missing"
    status = run.get("status")
    if status != "completed":
        return "pending"
    return "success" if run.get("conclusion") == "success" else "failed"


def github_json(url: str, token: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "collide-o-scope-release-gate/1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(4 * 1024 * 1024 + 1)
        if len(payload) > 4 * 1024 * 1024:
            raise GateError("GitHub workflow response is unexpectedly large")
        return json.loads(payload)
    except (OSError, UnicodeError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        raise GateError(f"query GitHub workflow runs: {error}") from error


def github_bytes(url: str, token: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "collide-o-scope-release-gate/1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(MAX_LOG_ARCHIVE_BYTES + 1)
    except (OSError, urllib.error.HTTPError) as error:
        raise GateError(f"download GitHub workflow logs: {error}") from error
    if len(payload) > MAX_LOG_ARCHIVE_BYTES:
        raise GateError("GitHub workflow log archive exceeds its bounded size")
    return payload


def query_workflow(repository: str, workflow: str, commit: str, token: str) -> dict | None:
    encoded_workflow = urllib.parse.quote(workflow, safe="")
    query = urllib.parse.urlencode({"head_sha": commit, "per_page": "100"})
    url = (
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        f"{encoded_workflow}/runs?{query}"
    )
    return newest_exact_run(github_json(url, token), commit)


def query_run_jobs(
    repository: str, run_id: int, run_attempt: int, token: str
) -> list[dict]:
    url = (
        f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/"
        f"attempts/{run_attempt}/jobs?per_page=100"
    )
    payload = github_json(url, token)
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise GateError("GitHub returned an unsupported workflow-jobs document")
    jobs = payload["jobs"]
    if payload.get("total_count") != len(jobs) or not 1 <= len(jobs) <= 100:
        raise GateError("workflow job inventory is empty, truncated, or unbounded")
    return jobs


def exact_required_step_receipts(
    jobs: list[dict],
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    pair_groups = {
        pair: group
        for group, pairs in REQUIRED_STEP_GROUPS.items()
        for pair in pairs
    }
    expected_pairs = set(pair_groups)
    observed_pairs: set[tuple[str, str]] = set()
    grouped = {group: [] for group in REQUIRED_STEP_GROUPS}
    required_jobs: dict[str, dict] = {}
    for job in jobs:
        job_name = job.get("name")
        steps = job.get("steps")
        if not isinstance(job_name, str) or not isinstance(steps, list):
            raise GateError("workflow job has an unsupported shape")
        if job_name not in REQUIRED_JOB_NAMES:
            continue
        job_id = job.get("id")
        if (
            job_name in required_jobs
            or not isinstance(job_id, int)
            or job_id <= 0
            or job.get("status") != "completed"
            or job.get("conclusion") != "success"
        ):
            raise GateError(f"required CI job is duplicated or unsuccessful: {job_name}")
        required_jobs[job_name] = job
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("name"), str):
                raise GateError(f"required CI job has a malformed step: {job_name}")
            pair = (job_name, step["name"])
            group = pair_groups.get(pair)
            if group is None:
                continue
            if pair in observed_pairs or step.get("conclusion") != "success":
                raise GateError(f"required CI step is duplicated or unsuccessful: {pair}")
            observed_pairs.add(pair)
            grouped[group].append(
                {"job": job_name, "step": step["name"], "conclusion": "success"}
            )
    if set(required_jobs) != REQUIRED_JOB_NAMES or observed_pairs != expected_pairs:
        raise GateError("exact required CI job/step mapping is incomplete")
    return (
        {
            group: sorted(rows, key=lambda item: (item["job"], item["step"]))
            for group, rows in grouped.items()
        },
        required_jobs,
    )


def job_log_text(log_payload: bytes) -> str:
    if len(log_payload) > 32 * 1024 * 1024:
        raise GateError("selected CI job log exceeds its bounded size")
    if not log_payload.startswith(b"PK"):
        try:
            return log_payload.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise GateError(f"decode selected CI job log: {error}") from error
    texts: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(log_payload)) as archive:
            infos = archive.infolist()
            if not 1 <= len(infos) <= 128:
                raise GateError("selected CI job log archive has an unbounded file count")
            total_size = sum(info.file_size for info in infos)
            if total_size > 32 * 1024 * 1024:
                raise GateError("selected CI job log archive has an unsafe expansion size")
            for info in infos:
                if info.is_dir():
                    continue
                texts.append(archive.read(info).decode("utf-8", errors="strict"))
    except (UnicodeError, RuntimeError, zipfile.BadZipFile) as error:
        raise GateError(f"read selected CI job log archive: {error}") from error
    return "\n".join(texts)


def parse_job_test_result(log_payload: bytes, job_name: str) -> dict:
    if job_name not in PLATFORM_TEST_STEPS:
        raise GateError("cargo test evidence is bound to an unexpected CI job")
    summary_pattern = re.compile(
        r"test result: ok\. ([0-9]+) passed; ([0-9]+) failed; "
        r"([0-9]+) ignored; ([0-9]+) measured; ([0-9]+) filtered out;"
    )
    failed_summary_pattern = re.compile(
        r"test result: FAILED\. ([0-9]+) passed; ([0-9]+) failed; "
        r"([0-9]+) ignored; ([0-9]+) measured; ([0-9]+) filtered out;"
    )
    ignored_pattern = re.compile(
        r"(?m)^(?:[0-9]{4}-[0-9T:.+-]+Z? )?test "
        r"([^\r\n]+?) \.\.\. ignored(?:,.*)?$"
    )
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    text = ansi_pattern.sub("", job_log_text(log_payload)).replace("\r\n", "\n")
    summaries = [
        tuple(int(value) for value in match.groups())
        for match in summary_pattern.finditer(text)
    ]
    failed_summaries = [
        tuple(int(value) for value in match.groups())
        for match in failed_summary_pattern.finditer(text)
    ]
    ignored_observations = ignored_pattern.findall(text)
    if failed_summaries:
        raise GateError(f"selected CI job log contains a failed cargo test summary: {job_name}")
    if not summaries or any(summary[1] != 0 for summary in summaries):
        raise GateError(f"selected CI job log lacks a successful cargo test summary: {job_name}")
    ignored_names = set(ignored_observations)
    if len(ignored_names) > 256 or any(len(name) > 512 for name in ignored_names):
        raise GateError("ignored test-name evidence is unbounded")
    external_observations = [
        name
        for name in ignored_observations
        if re.search(r"(?i)(external|fixture|live|physical|ffmpeg)", name)
    ]
    external_names = sorted(set(external_observations))
    return {
        "job": job_name,
        "summary_records": len(summaries),
        "passed": sum(summary[0] for summary in summaries),
        "failed": 0,
        "ignored": sum(summary[2] for summary in summaries),
        "measured": sum(summary[3] for summary in summaries),
        "filtered_out": sum(summary[4] for summary in summaries),
        "ignored_test_names": sorted(ignored_names),
        "external_fixture_ignored_count": len(external_observations),
        "external_fixture_ignored_names": external_names,
    }


def collect_platform_test_results(
    repository: str,
    required_jobs: dict[str, dict],
    token: str,
    downloader=github_bytes,
) -> dict:
    rows = []
    for job_name in sorted(PLATFORM_TEST_STEPS):
        job = required_jobs.get(job_name)
        job_id = job.get("id") if isinstance(job, dict) else None
        if not isinstance(job_id, int) or job_id <= 0:
            raise GateError(f"selected platform CI job has no stable ID: {job_name}")
        logs_url = f"https://api.github.com/repos/{repository}/actions/jobs/{job_id}/logs"
        row = parse_job_test_result(downloader(logs_url, token), job_name)
        row["job_id"] = job_id
        row["logs_url"] = logs_url
        rows.append(row)
    if {row["job"] for row in rows} != set(PLATFORM_TEST_STEPS):
        raise GateError("selected CI job-log evidence is incomplete")
    ignored_names = sorted(
        {name for row in rows for name in row["ignored_test_names"]}
    )
    external_names = sorted(
        {name for row in rows for name in row["external_fixture_ignored_names"]}
    )
    return {
        "summary_records": sum(row["summary_records"] for row in rows),
        "passed": sum(row["passed"] for row in rows),
        "failed": 0,
        "ignored": sum(row["ignored"] for row in rows),
        "measured": sum(row["measured"] for row in rows),
        "filtered_out": sum(row["filtered_out"] for row in rows),
        "ignored_test_names": ignored_names,
        "external_fixture_ignored_count": sum(
            row["external_fixture_ignored_count"] for row in rows
        ),
        "external_fixture_ignored_names": external_names,
        "platform_jobs": rows,
        "source": "selected exact-SHA CI platform job logs",
    }


def final_candidate_validation(repository: str, ci_run: dict, token: str) -> dict:
    run_id = ci_run.get("id")
    run_attempt = ci_run.get("run_attempt")
    if (
        not isinstance(run_id, int)
        or run_id <= 0
        or not isinstance(run_attempt, int)
        or run_attempt <= 0
    ):
        raise GateError("CI run has no stable ID or attempt")
    jobs = query_run_jobs(repository, run_id, run_attempt, token)
    step_receipts, required_jobs = exact_required_step_receipts(jobs)
    format_steps = step_receipts["format"]
    capability_steps = step_receipts["capability"]
    check_test_clippy_steps = step_receipts["check_test_clippy"]
    exception_steps = step_receipts["dependency_exception"]
    vendor_steps = step_receipts["vendor"]
    test_results = collect_platform_test_results(
        repository, required_jobs, token
    )
    return {
        "ci_run_id": run_id,
        "ci_run_attempt": run_attempt,
        "format": {
            "commands": [
                "cargo fmt --all -- --check",
                "node --check static/app.js",
                "node --check docs/ui-ux/wireframe.js",
            ],
            "steps": format_steps,
            "conclusion": "success",
        },
        "check_test_clippy": {
            "commands": [
                "cargo check --locked --all-targets --all-features",
                "cargo test --locked --all-targets --all-features",
                "cargo clippy --locked --all-targets --all-features -- -D warnings",
            ],
            "steps": check_test_clippy_steps,
            "conclusion": "success",
            "test_results": test_results,
        },
        "capability_registry_contradiction_gate": {
            "command": "cargo run --locked --bin generate_capabilities -- --check",
            "steps": capability_steps,
            "conclusion": "success",
        },
        "dependency_exception_policy": {
            "command": "python scripts/check-dependency-policy.py",
            "steps": exception_steps,
            "conclusion": "success",
        },
        "vendor_verifier": {
            "command": "python scripts/verify-vendored-wgpu-hal.py --self-test",
            "steps": vendor_steps,
            "conclusion": "success",
        },
    }


def wait_for_workflows(
    repository: str,
    commit: str,
    workflows: list[str],
    token: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict[str, dict]:
    deadline = time.monotonic() + timeout_seconds
    last_summary = ""
    while True:
        observed = {
            workflow: query_workflow(repository, workflow, commit, token)
            for workflow in workflows
        }
        summary = ", ".join(
            f"{workflow}={run_state(run)}"
            + (
                f"(run={run.get('id')},attempt={run.get('run_attempt')},"
                f"conclusion={run.get('conclusion')})"
                if run is not None
                else ""
            )
            for workflow, run in observed.items()
        )
        if summary != last_summary:
            print(f"required workflow state for {commit}: {summary}", flush=True)
            last_summary = summary
        failures = [
            workflow
            for workflow, run in observed.items()
            if run_state(run) == "failed"
        ]
        if failures:
            raise GateError(
                "newest exact-commit workflow run failed: " + ", ".join(failures)
            )
        if all(run_state(run) == "success" for run in observed.values()):
            return {workflow: run for workflow, run in observed.items() if run is not None}
        if time.monotonic() >= deadline:
            raise GateError(
                f"timed out after {timeout_seconds}s waiting for: {summary}"
            )
        time.sleep(poll_seconds)


def required_runs_receipt(
    repository: str,
    commit: str,
    workflows: list[str],
    runs: dict[str, dict],
    validation: dict | None = None,
) -> dict:
    if set(runs) != set(workflows):
        raise GateError("required-run receipt is missing a workflow")
    selected = []
    for workflow in workflows:
        run = runs[workflow]
        run_id = run.get("id")
        run_number = run.get("run_number")
        attempt = run.get("run_attempt")
        url = run.get("html_url")
        if (
            not isinstance(run_id, int)
            or run_id <= 0
            or not isinstance(run_number, int)
            or run_number <= 0
            or not isinstance(attempt, int)
            or attempt <= 0
            or not isinstance(url, str)
            or RUN_URL.fullmatch(url) is None
            or run.get("head_sha") != commit
            or run.get("status") != "completed"
            or run.get("conclusion") != "success"
        ):
            raise GateError(f"selected {workflow} run cannot form a release receipt")
        selected.append(
            {
                "workflow": workflow,
                "run_id": run_id,
                "run_number": run_number,
                "run_attempt": attempt,
                "url": url,
                "conclusion": "success",
            }
        )
    receipt = {
        "schema_version": 1,
        "repository": repository,
        "commit": commit,
        "selected_newest_exact_commit_runs": selected,
    }
    if validation is not None:
        receipt["final_candidate_validation"] = validation
    return receipt


def canonical_json(document: dict) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise GateError("required-run receipt exceeds its bounded size")
    return encoded


def write_receipt(path: Path, document: dict) -> str:
    encoded = canonical_json(document)
    if path.exists():
        raise GateError(f"refusing to overwrite required-run receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + "\n", encoding="utf-8", newline="\n")
    return encoded


def collect_final_candidate_evidence(
    repository: str,
    workflows: list[str],
    runs: dict[str, dict],
    token: str,
    collector=final_candidate_validation,
) -> dict:
    if (
        len(workflows) != len(REQUIRED_WORKFLOWS)
        or set(workflows) != REQUIRED_WORKFLOWS
        or set(runs) != REQUIRED_WORKFLOWS
    ):
        raise GateError(
            "final-candidate evidence requires exactly ci.yml and adversarial.yml"
        )
    ci_run = runs.get("ci.yml")
    if not isinstance(ci_run, dict):
        raise GateError("final-candidate evidence has no selected CI run")
    return collector(repository, ci_run, token)


def self_test() -> None:
    commit = "a" * 40
    payload = {
        "workflow_runs": [
            {
                "id": 1,
                "head_sha": commit,
                "run_number": 8,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://github.com/acme/project/actions/runs/1",
            },
            {
                "id": 2,
                "head_sha": "b" * 40,
                "run_number": 99,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://github.com/acme/project/actions/runs/2",
            },
            {
                "id": 3,
                "head_sha": commit,
                "run_number": 9,
                "run_attempt": 2,
                "status": "in_progress",
                "conclusion": None,
                "html_url": "https://github.com/acme/project/actions/runs/3",
            },
        ]
    }
    newest = newest_exact_run(payload, commit)
    assert newest is not None and newest["id"] == 3
    assert run_state(newest) == "pending"
    newest["status"] = "completed"
    newest["conclusion"] = "failure"
    assert run_state(newest) == "failed"
    newest["conclusion"] = "success"
    assert run_state(newest) == "success"
    assert newest_exact_run({"workflow_runs": []}, commit) is None
    receipt = required_runs_receipt(
        "acme/project", commit, ["ci.yml"], {"ci.yml": newest}
    )
    assert receipt["selected_newest_exact_commit_runs"][0]["run_attempt"] == 2
    mutated = dict(newest)
    mutated["head_sha"] = "b" * 40
    try:
        required_runs_receipt(
            "acme/project", commit, ["ci.yml"], {"ci.yml": mutated}
        )
    except GateError:
        pass
    else:
        raise AssertionError("receipt accepted a stale workflow run")
    required_jobs_fixture = []
    for index, (job_name, test_step) in enumerate(
        PLATFORM_TEST_STEPS.items(), start=100
    ):
        required_jobs_fixture.append(
            {
                "id": index,
                "name": job_name,
                "status": "completed",
                "conclusion": "success",
                "steps": [
                    {"name": FORMAT_STEP, "conclusion": "success"},
                    {"name": CAPABILITY_STEP, "conclusion": "success"},
                    {"name": test_step, "conclusion": "success"},
                    {"name": PLATFORM_VENDOR_STEP, "conclusion": "success"},
                ],
            }
        )
    required_jobs_fixture.append(
        {
            "id": 200,
            "name": DEPENDENCY_JOB,
            "status": "completed",
            "conclusion": "success",
            "steps": [
                {"name": DEPENDENCY_EXCEPTION_STEP, "conclusion": "success"},
                {"name": DEPENDENCY_VENDOR_STEP, "conclusion": "success"},
            ],
        }
    )
    groups, selected_jobs = exact_required_step_receipts(required_jobs_fixture)
    assert len(groups["format"]) == 3
    assert len(groups["check_test_clippy"]) == 3
    assert len(groups["vendor"]) == 4
    wrong_job_fixture = [dict(job) for job in required_jobs_fixture]
    wrong_job_fixture[0] = dict(wrong_job_fixture[0])
    wrong_job_fixture[0]["name"] = "Arbitrary successful runner"
    try:
        exact_required_step_receipts(wrong_job_fixture)
    except GateError:
        pass
    else:
        raise AssertionError("exact CI evidence accepted required steps under a wrong job")

    linux_log = (
        "2026-08-24T11:22:33.1234567Z test video::external_ffmpeg_fixture ... ignored\n"
        "2026-08-24T11:22:33.1234567Z test result: ok. 17 passed; 0 failed; "
        "1 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"
    ).encode("utf-8")
    success_log = (
        "test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; "
        "0 filtered out; finished in 0.01s\n"
    ).encode("utf-8")
    payloads = {
        selected_jobs[job_name]["id"]: (
            linux_log if job_name == "Linux (Ubuntu 24.04)" else success_log
        )
        for job_name in PLATFORM_TEST_STEPS
    }

    def fixture_downloader(url: str, _token: str) -> bytes:
        job_id = int(url.rsplit("/", 2)[-2])
        return payloads[job_id]

    test_results = collect_platform_test_results(
        "acme/project", selected_jobs, "token", fixture_downloader
    )
    assert test_results["passed"] == 39
    assert test_results["ignored"] == 1
    assert test_results["external_fixture_ignored_count"] == 1
    assert {row["job"] for row in test_results["platform_jobs"]} == set(
        PLATFORM_TEST_STEPS
    )
    failed_log = (
        "2026-08-24T11:22:33.1234567Z test result: FAILED. "
        "16 passed; 1 failed; 0 ignored; 0 measured; "
        "0 filtered out; finished in 0.01s\n"
    ).encode("utf-8")
    try:
        parse_job_test_result(failed_log, "Windows (VS 2022)")
    except GateError:
        pass
    else:
        raise AssertionError("test-count receipt accepted a failing cargo summary")
    three_summaries_one_job = success_log * 3
    unbound_payloads = {
        selected_jobs[job_name]["id"]: (
            three_summaries_one_job
            if job_name == "Linux (Ubuntu 24.04)"
            else b"setup completed without cargo test evidence\n"
        )
        for job_name in PLATFORM_TEST_STEPS
    }

    def unbound_downloader(url: str, _token: str) -> bytes:
        job_id = int(url.rsplit("/", 2)[-2])
        return unbound_payloads[job_id]

    try:
        collect_platform_test_results(
            "acme/project", selected_jobs, "token", unbound_downloader
        )
    except GateError:
        pass
    else:
        raise AssertionError("three summaries in one CI job satisfied three platform jobs")
    sentinel = {"validation": "selected exact-SHA CI evidence"}
    calls: list[tuple[str, int, str]] = []

    def fake_collector(repository: str, ci_run: dict, token: str) -> dict:
        calls.append((repository, ci_run["id"], token))
        return sentinel

    selected_runs = {
        "ci.yml": newest,
        "adversarial.yml": dict(newest),
    }
    assert (
        collect_final_candidate_evidence(
            "acme/project",
            ["ci.yml", "adversarial.yml"],
            selected_runs,
            "token",
            fake_collector,
        )
        == sentinel
    )
    assert calls == [("acme/project", newest["id"], "token")]
    try:
        collect_final_candidate_evidence(
            "acme/project",
            ["ci.yml"],
            {"ci.yml": newest},
            "token",
            fake_collector,
        )
    except GateError:
        pass
    else:
        raise AssertionError("final-candidate branch accepted an incomplete workflow set")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository")
    parser.add_argument("--commit")
    parser.add_argument("--workflow", action="append", default=[])
    parser.add_argument("--token-env", default="GH_TOKEN")
    parser.add_argument("--timeout-seconds", type=int, default=7_200)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--final-candidate-evidence", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("required-workflow gate self-test passed")
            return 0
        repository = args.repository or ""
        commit = (args.commit or "").lower()
        if REPOSITORY.fullmatch(repository) is None:
            raise GateError("repository must be owner/name")
        if SHA.fullmatch(commit) is None:
            raise GateError("commit must be a 40-character lowercase Git SHA")
        if not args.workflow or any(WORKFLOW.fullmatch(item) is None for item in args.workflow):
            raise GateError("at least one bounded .yml/.yaml workflow name is required")
        if not 60 <= args.timeout_seconds <= 14_400:
            raise GateError("timeout-seconds must be in 60..=14400")
        if not 5 <= args.poll_seconds <= 300:
            raise GateError("poll-seconds must be in 5..=300")
        token = os.environ.get(args.token_env, "")
        if not token:
            raise GateError(f"{args.token_env} is unavailable")
        workflows = list(dict.fromkeys(args.workflow))
        runs = wait_for_workflows(
            repository,
            commit,
            workflows,
            token,
            args.timeout_seconds,
            args.poll_seconds,
        )
        validation = None
        if args.final_candidate_evidence:
            validation = collect_final_candidate_evidence(
                repository, workflows, runs, token
            )
        receipt = required_runs_receipt(
            repository, commit, workflows, runs, validation
        )
        encoded = canonical_json(receipt)
        if args.receipt_output is not None:
            encoded = write_receipt(args.receipt_output, receipt)
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8", newline="\n") as output:
                output.write(f"required_runs={encoded}\n")
    except (GateError, OSError, UnicodeError) as error:
        print(f"required-workflow gate failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
