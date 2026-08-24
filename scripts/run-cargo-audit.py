#!/usr/bin/env python3
"""Run cargo-audit using only current, reviewed exception metadata."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import runpy


def main() -> int:
    policy_module = runpy.run_path(
        str(Path(__file__).with_name("check-dependency-policy.py"))
    )
    try:
        entries = policy_module["validated_exceptions"](
            __import__("datetime").date.today()
        )
    except ValueError as error:
        print(f"dependency policy failed: {error}", file=sys.stderr)
        return 1
    # Vulnerabilities are fatal. Informational advisories use cargo-deny's
    # configured workspace/transitive policy so the two tools do not encode
    # contradictory severity rules.
    command = ["cargo", "audit"]
    for entry in entries:
        command.extend(("--ignore", entry["id"]))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
