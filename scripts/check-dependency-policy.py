#!/usr/bin/env python3
"""Fail closed on missing, stale, or unsynchronised advisory exceptions."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
RUSTSEC_ID = re.compile(r"^RUSTSEC-\d{4}-\d{4}$")


def load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"read {path}: {error}") from error


def validated_exceptions(as_of: date) -> list[dict]:
    policy = load_toml(ROOT / "policy" / "dependency-exceptions.toml")
    deny = load_toml(ROOT / "deny.toml")
    if policy.get("schema_version") != 1:
        raise ValueError("unsupported dependency exception policy schema")
    entries = policy.get("exceptions")
    if not isinstance(entries, list):
        raise ValueError("dependency exceptions must be an array")

    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each dependency exception must be a table")
        advisory_id = entry.get("id")
        if not isinstance(advisory_id, str) or not RUSTSEC_ID.fullmatch(advisory_id):
            raise ValueError(f"invalid RustSec advisory id {advisory_id!r}")
        if advisory_id in seen:
            raise ValueError(f"duplicate dependency exception {advisory_id}")
        seen.add(advisory_id)
        for field in ("owner", "reason"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise ValueError(f"{advisory_id} needs a bounded, non-empty {field}")
        reviewed = entry.get("reviewed")
        if not isinstance(reviewed, date):
            raise ValueError(f"{advisory_id} reviewed must be an ISO date")
        if reviewed > as_of:
            raise ValueError(
                f"{advisory_id} has a future review date {reviewed.isoformat()}"
            )
        expires = entry.get("expires")
        if not isinstance(expires, date):
            raise ValueError(f"{advisory_id} expires must be an ISO date")
        if expires < reviewed:
            raise ValueError(
                f"{advisory_id} expires before its review date "
                f"({expires.isoformat()} < {reviewed.isoformat()})"
            )
        if expires < as_of:
            raise ValueError(f"{advisory_id} expired on {expires.isoformat()}")

    ignored = deny.get("advisories", {}).get("ignore", [])
    if not isinstance(ignored, list) or not all(isinstance(item, str) for item in ignored):
        raise ValueError("deny.toml advisories.ignore must be an array of strings")
    if set(ignored) != seen or len(ignored) != len(seen):
        raise ValueError(
            "deny.toml advisories.ignore and policy/dependency-exceptions.toml "
            f"must match exactly (deny={sorted(ignored)}, policy={sorted(seen)})"
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    try:
        entries = validated_exceptions(args.as_of)
    except ValueError as error:
        print(f"dependency policy failed: {error}", file=sys.stderr)
        return 1
    print(f"dependency policy valid: {len(entries)} active exception(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
