#!/usr/bin/env python3
"""Run the authorized compile-free PR-2 matched-accuracy M1-A stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from trotterlib.pr2_matched_accuracy_m1_execution import (
    run_m1_a,
    write_m1_a_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    output = arguments.output
    if not output.is_absolute():
        output = root / output
    result = run_m1_a(root, authorization_path=arguments.authorization)
    write_m1_a_result(output, result)
    print(f"status={result['status']}")
    print(f"candidate_count={len(result['candidate_ledger'])}")
    print(
        "selected_random_cell_count="
        f"{result['compile_selection']['selected_count']}"
    )
    print(f"result_fingerprint={result['result_fingerprint']}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
