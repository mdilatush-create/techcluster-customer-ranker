"""
Weekly Customer Engagement Ranker

This repo currently contains the implementation spec and a GitHub Actions schedule.
The full ranker implementation (Yobi metric extraction, identity resolution, scoring,
HubSpot writes, Slack summary) should be added here next.

For now, this script is a safe placeholder so the scheduled workflow can run without
doing any writes.
"""

from __future__ import annotations

import os
import sys


REQUIRED_ENVS = [
    "HUBSPOT_ACCESS_TOKEN",
    "YOBI_EMAIL",
    "YOBI_PASSWORD",
]


def main() -> int:
    missing = [k for k in REQUIRED_ENVS if not os.environ.get(k)]
    if missing:
        print("Customer Engagement Ranker is not configured yet.")
        print("Missing required secrets:", ", ".join(missing))
        print("Set these as GitHub Actions repo secrets before enabling real writes.")
        # Exit 0 to avoid a failing scheduled workflow in a public repo.
        return 0

    print("Ranker not implemented yet. See `Customer Ranker Update/spec.md`.")
    # When implemented, return non-zero on failure, 0 on success.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

