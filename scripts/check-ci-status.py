#!/usr/bin/env python3
"""Suite-aware three-platform CI verdict for one commit.

Usage: python scripts/check-ci-status.py <sha>

A SHA can carry MULTIPLE check suites (a branch push and a PR run have
different concurrency groups, so both execute). Counting `success`
conclusions across the flat check-runs list is how a false green happens:
three successes can accumulate across suites while one suite's Linux job is
still running — and then fail. That exact false green occurred at 6c06237
on 2026-08-18, hiding a real Linux timeout.

This script groups runs by check-suite id and answers per suite: exit 0
when at least one complete suite has all three named jobs concluded
success and no complete suite is still pending; exit 1 when a complete
suite failed and none is pending (a definitive red); exit 2 while anything
is still running. No authentication is needed for a public repository.
"""

import json
import sys
import urllib.request

REPOSITORY = "Spacejunk-io/collide-o-scope"
REQUIRED_JOBS = {"Linux (Ubuntu 24.04)", "macOS 15", "Windows (VS 2022)"}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    sha = sys.argv[1]
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}/commits/{sha}"
        "/check-runs?per_page=100",
        headers={"Accept": "application/vnd.github+json"},
    )
    data = json.load(urllib.request.urlopen(request))
    suites: dict[int, dict[str, tuple[str, str]]] = {}
    for run in data.get("check_runs", []):
        suite = run["check_suite"]["id"]
        suites.setdefault(suite, {})[run["name"]] = (
            run["status"],
            run["conclusion"],
        )
    green = failed = pending = 0
    for runs in suites.values():
        if not REQUIRED_JOBS.issubset(runs.keys()):
            continue
        conclusions = [runs[job][1] for job in REQUIRED_JOBS]
        statuses = [runs[job][0] for job in REQUIRED_JOBS]
        if all(conclusion == "success" for conclusion in conclusions):
            green += 1
        elif any(
            conclusion in ("failure", "cancelled", "timed_out")
            for conclusion in conclusions
        ):
            failed += 1
        elif any(status != "completed" for status in statuses):
            pending += 1
    print(f"complete-suites green={green} failed={failed} pending={pending}")
    if green >= 1 and pending == 0:
        return 0
    if failed and not pending:
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
