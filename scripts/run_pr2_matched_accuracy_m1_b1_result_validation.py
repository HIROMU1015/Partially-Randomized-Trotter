#!/usr/bin/env python3
"""Validate and analyze the completed PR-2 M1-B1 compile map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trotterlib.pr2_matched_accuracy_m1_b1_result_validation import (
    validate_and_analyze,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    payload = validate_and_analyze(root)
    write_json(payload, output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["external_research_review"]["decision"],
                "validation_fingerprint": payload["validation_fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
