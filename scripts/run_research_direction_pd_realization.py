#!/usr/bin/env python3
"""Run the preregistered P-D signed-time/internal-H_D Go/No-Go gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np
import scipy

from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_energy_tail_pareto import (
    read_json_object,
    write_json_nonoverwriting,
)
from trotterlib.research_direction_pd_realization import (
    evaluate_pd_realization,
    expected_task_manifest_body,
    finalize_expected_task_manifest,
    finalize_result,
    validate_expected_task_manifest,
)


DEFAULT_EXPECTED = Path(
    "artifacts/research_direction_pd_realization/2026-09-25/"
    "pd_realization_expected_tasks_v2.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_pd_realization/2026-09-25/"
    "pd_realization_go_no_go_v2.json"
)
DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
SOURCE_PATHS = (
    Path("src/trotterlib/research_direction_pd_realization.py"),
    Path("scripts/run_research_direction_pd_realization.py"),
    Path("src/trotterlib/research_direction_energy_tail_pareto.py"),
    Path("src/trotterlib/product_formula.py"),
    Path("src/trotterlib/pf_decomposition.py"),
    Path("src/trotterlib/rte.py"),
    Path("docs/research/pd_realization_go_no_go_preregistration.md"),
    DEFAULT_SNAPSHOT,
)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _source_hashes(root: Path) -> dict[str, str]:
    return {str(path): file_sha256(root / path) for path in SOURCE_PATHS}


def _provenance(
    root: Path,
    *,
    command: str,
    mode: str,
    elapsed_seconds: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_dirty_worktree_not_externally_reproduced",
        "command": command,
        "mode": mode,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "source_sha256": _source_hashes(root),
        "cpu_count_visible": os.cpu_count(),
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = float(elapsed_seconds)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    expected_path = _resolve(root, args.expected)
    snapshot_path = _resolve(root, args.snapshot)
    output_path = _resolve(root, args.output)
    command = shlex.join(
        [
            ".venv311/bin/python",
            "scripts/run_research_direction_pd_realization.py",
            *sys.argv[1:],
        ]
    )

    if args.dry_run:
        expected = finalize_expected_task_manifest(
            expected_task_manifest_body(),
            provenance=_provenance(
                root,
                command=command,
                mode="freeze_d1_d2_and_fresh_ld5_before_compute",
            ),
        )
        write_json_nonoverwriting(expected, expected_path)
        print(expected_path)
        print(expected["content_fingerprint"])
        print(expected["task_count"])
        return 0

    if not expected_path.is_file():
        raise FileNotFoundError(
            "Run --dry-run before D1/D2 and the fresh L_D=5 holdout."
        )
    expected = read_json_object(expected_path)
    validate_expected_task_manifest(expected)
    if expected["provenance"]["source_sha256"] != _source_hashes(root):
        raise ValueError("P-D realization sources changed after expected-task freeze.")
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Missing fixed H4 snapshot: {snapshot_path}")

    started = time.perf_counter()
    print("P-D D1: signed-time finite-RTE and controlled-phase oracle", flush=True)
    print("P-D D2: fragment-level internal H_D at L_D=3,4", flush=True)
    print("P-D D3: fresh L_D=5 selection holdout", flush=True)
    body = evaluate_pd_realization(snapshot_path, expected)
    elapsed = time.perf_counter() - started
    artifact = finalize_result(
        body,
        provenance=_provenance(
            root,
            command=command,
            mode="frozen_d1_d2_d3_full_computation",
            elapsed_seconds=elapsed,
        ),
        source_evidence=tuple(
            {
                "path": str(path),
                "sha256": file_sha256(root / path),
                "role": "frozen_source_or_input",
            }
            for path in SOURCE_PATHS
        )
        + (
            {
                "path": str(args.expected),
                "sha256": file_sha256(expected_path),
                "role": "compile_before_expected_tasks",
            },
        ),
    )
    write_json_nonoverwriting(artifact, output_path)
    print(output_path)
    print(artifact["content_fingerprint"])
    print(artifact["decision"]["status"])
    print(f"elapsed_seconds={elapsed:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

