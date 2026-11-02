#!/usr/bin/env python3
"""Future PM-1 one-shot execution. No authorization accompanies preparation."""
import argparse
from pathlib import Path

from trottertracks.resource_applicability.pm1_discard_execution import run_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    result = run_once(args.project_root, args.plan, args.authorization)
    print(result["status"])
    return 0 if result["failure_reason"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
