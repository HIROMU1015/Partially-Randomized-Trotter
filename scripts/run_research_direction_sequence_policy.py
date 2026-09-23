#!/usr/bin/env python3
"""Run focused WP06-b sequence policy selection and proxy recalibration."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit

from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.research_direction_prevalidation import file_sha256
from trotterlib.research_direction_sequence_policy import (
    evaluate_wp06b_sequence_policy,
    finalize_wp06b_artifact,
    write_wp06b_artifact,
)
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_WP06A = Path(
    "artifacts/research_direction_structure_pilot/2026-09-22/"
    "wp06a_circuit_structure_pilot_v1.json"
)
DEFAULT_WP04 = Path(
    "artifacts/research_direction_ablation/2026-09-21/"
    "wp04_finite_rte_statistical_ablation_v1.json"
)
DEFAULT_PROXY_DIRECTORY = Path(
    "artifacts/research_direction_prevalidation/2026-09-21/wp01s_calibrations"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_sequence_policy/2026-09-22/"
    "wp06b_sequence_policy_proxy_bridge_v1.json"
)
SCHEDULE_LENGTHS = (1, 2, 4, 8, 16, 32)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _provenance() -> dict[str, object]:
    sources = (
        Path("src/trotterlib/df_rte_circuit.py"),
        Path("src/trotterlib/df_rte_qiskit.py"),
        Path("src/trotterlib/research_direction_sequence_policy.py"),
        Path("scripts/run_research_direction_sequence_policy.py"),
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_sequence_policy.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qiskit_version": qiskit.__version__,
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }


def _proxy_path(directory: Path, rte_steps: int) -> Path:
    return directory / (
        f"ld3_dt0p02_r{rte_steps}_k2_q1_q2_mc8.proxy.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--wp06a", type=Path, default=DEFAULT_WP06A)
    parser.add_argument("--wp04", type=Path, default=DEFAULT_WP04)
    parser.add_argument(
        "--proxy-directory", type=Path, default=DEFAULT_PROXY_DIRECTORY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-samples", type=int, default=8)
    parser.add_argument("--holdout-samples", type=int, default=12)
    parser.add_argument("--transfer-samples", type=int, default=8)
    args = parser.parse_args()

    wp06a = json.loads(args.wp06a.read_text(encoding="utf-8"))
    if wp06a["decision"]["next_action"] != (
        "WP06b_sequence_aware_full_vs_support_basis_policy_and_recalibration"
    ):
        raise ValueError("WP06-a does not route to focused WP06-b.")
    wp04 = json.loads(args.wp04.read_text(encoding="utf-8"))
    proxy_paths = {
        rte_steps: _proxy_path(args.proxy_directory, rte_steps)
        for rte_steps in SCHEDULE_LENGTHS
    }
    proxies = {
        rte_steps: json.loads(path.read_text(encoding="utf-8"))
        for rte_steps, path in proxy_paths.items()
    }
    for rte_steps, proxy in proxies.items():
        if (
            int(proxy["ld"]) != 3
            or float(proxy["delta_time"]) != 0.02
            or int(proxy["finite_taylor_order"]) != 2
            or int(proxy["rte_steps_per_occurrence"]) != rte_steps
        ):
            raise ValueError("A legacy proxy does not match the WP06-b scope.")

    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 3),
        identity_policy="extract_identity_phase",
    )
    compiler = CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )
    body = evaluate_wp06b_sequence_policy(
        hamiltonian,
        preparation,
        compiler,
        wp04,
        proxies,
        training_sample_count=args.training_samples,
        holdout_sample_count=args.holdout_samples,
        transfer_sample_count=args.transfer_samples,
    )
    body["source_evidence"] = {
        "snapshot": {
            "path": str(args.snapshot),
            "sha256": file_sha256(args.snapshot),
        },
        "wp06a": {
            "path": str(args.wp06a),
            "sha256": file_sha256(args.wp06a),
            "content_fingerprint": wp06a["content_fingerprint"],
        },
        "wp04": {
            "path": str(args.wp04),
            "sha256": file_sha256(args.wp04),
            "content_fingerprint": wp04["content_fingerprint"],
        },
        "legacy_proxies": [
            {
                "rte_steps": rte_steps,
                "path": str(proxy_paths[rte_steps]),
                "sha256": file_sha256(proxy_paths[rte_steps]),
                "proxy_fingerprint": proxies[rte_steps]["proxy_fingerprint"],
            }
            for rte_steps in SCHEDULE_LENGTHS
        ],
    }
    artifact = finalize_wp06b_artifact(body, provenance=_provenance())
    write_wp06b_artifact(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": artifact["content_fingerprint"],
                "overall_pass": artifact["overall_pass"],
                "summary": artifact["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
