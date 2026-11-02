#!/usr/bin/env python3
"""Run the preregistered P-A nondegenerate mechanism validation."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import qiskit

from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_joint_synthesis_mechanism_validation import (
    build_expected_task_manifest,
    evaluate_mechanism_validation,
    finalize_mechanism_validation_artifact,
    validate_expected_task_manifest,
    write_expected_task_manifest,
    write_mechanism_validation_artifact,
)
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_ROOT = Path(
    "artifacts/research_direction_joint_synthesis_mechanism_validation/2026-09-25"
)
DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_FORMALIZATION = Path(
    "artifacts/research_direction_joint_synthesis_formalization/2026-09-25/"
    "pa_v1_formalization_and_mechanism_audit_v1.json"
)
EXPECTED_NAME = "pa_forced_support_order2_expected_tasks_v1.json"
OUTPUT_NAME = "pa_forced_support_order2_mechanism_validation_v1.json"


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _prepare(snapshot: Path):
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(snapshot)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 3),
        identity_policy="extract_identity_phase",
    )
    return hamiltonian, preparation


def _compiler() -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--formalization",
        type=Path,
        default=DEFAULT_FORMALIZATION,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Freeze task inputs and event digests without compiling.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected_path = root / EXPECTED_NAME
    output_path = root / OUTPUT_NAME
    hamiltonian, preparation = _prepare(args.snapshot)

    if args.dry_run:
        formalization = _read_json(args.formalization)
        source_evidence = {
            "snapshot": {
                "path": str(args.snapshot),
                "sha256": file_sha256(args.snapshot),
            },
            "formalization": {
                "path": str(args.formalization),
                "sha256": file_sha256(args.formalization),
                "content_fingerprint": formalization.get("content_fingerprint"),
            },
        }
        expected = build_expected_task_manifest(
            preparation,
            source_evidence=source_evidence,
        )
        write_expected_task_manifest(expected, expected_path)
        print(expected_path)
        print(expected["content_fingerprint"])
        print(expected["task_count"])
        return 0

    if not expected_path.exists():
        raise ValueError("Run --dry-run and record the fingerprint before compilation.")
    expected = _read_json(expected_path)
    validate_expected_task_manifest(expected)
    if output_path.exists():
        raise ValueError(f"Refusing to replace existing artifact: {output_path}")

    body = evaluate_mechanism_validation(
        hamiltonian,
        preparation,
        _compiler(),
        expected,
    )
    source_paths = (
        Path(
            "src/trotterlib/"
            "research_direction_joint_synthesis_mechanism_validation.py"
        ),
        Path(
            "scripts/"
            "run_research_direction_joint_synthesis_mechanism_validation.py"
        ),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_dirty_worktree_not_externally_reproduced",
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qiskit_version": qiskit.__version__,
        "expected_task_file": {
            "path": str(expected_path),
            "sha256": file_sha256(expected_path),
            "content_fingerprint": expected["content_fingerprint"],
        },
        "source_sha256": {
            str(path): file_sha256(path) for path in source_paths
        },
    }
    artifact = finalize_mechanism_validation_artifact(
        body,
        provenance=provenance,
    )
    write_mechanism_validation_artifact(artifact, output_path)
    print(output_path)
    print(artifact["content_fingerprint"])
    print(artifact["decision"]["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
