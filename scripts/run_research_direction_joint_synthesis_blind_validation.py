#!/usr/bin/env python3
"""Run the preregistered H5/H4 P-A v1 blind transfer validation."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import qiskit

from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.parallel_validation_executor import atomic_write_json, file_sha256
from trotterlib.research_direction_joint_synthesis_blind_validation import (
    FROZEN_CORE_HASHES,
    FROZEN_PILOT_CONTENT_FINGERPRINT,
    FROZEN_PILOT_FILE_SHA256,
    H4_COMPILER_TRANSFER,
    H5_PHYSICAL_TRANSFER,
    STRATA,
    build_expected_task_manifest,
    compile_stratum_rows,
    finalize_blind_validation_artifact,
    finalize_checkpoint,
    fingerprint,
    run_operator_probes,
    summarize_stratum,
    validate_blind_validation_artifact,
    validate_checkpoint,
    validate_expected_task_manifest,
)
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_ROOT = Path(
    "artifacts/research_direction_joint_synthesis_blind_validation/2026-09-25"
)
DEFAULT_H5_SNAPSHOT = Path(
    "artifacts/rte_cost_system_size_h5/2026-08-25/snapshots/"
    "h5_sto3g_d100_rank9_df_snapshot_v1.npz"
)
DEFAULT_H4_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_PILOT_ARTIFACT = Path(
    "artifacts/research_direction_joint_synthesis_pilot/2026-09-25/"
    "pa_h4_interval_union_joint_synthesis_v1.json"
)
OUTPUT_NAME = "pa_v1_h5_physical_h4_opt2_blind_v1.json"
EXPECTED_NAME = "pa_v1_blind_expected_tasks_v1.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
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


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path)
        if existing != dict(payload):
            raise ValueError(f"Refusing to replace different fixed file: {path}")
        return
    atomic_write_json(path, payload)


def _compiler(level: int) -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=level,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )


def _checkpoint_path(root: Path, key: str) -> Path:
    return root / "checkpoints" / f"{key}.json"


def _load_checkpoints(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: dict[str, Any] = {}
    probes: dict[str, Any] = {}
    for path in sorted((root / "checkpoints").glob("*.json")):
        payload = _read_json(path)
        validate_checkpoint(payload)
        row = dict(payload["row"])
        if payload["kind"] == "holdout":
            key = str(row["task_key"])
            target = rows
        else:
            key = str(row["probe_key"])
            target = probes
        if key in target and target[key] != row:
            raise ValueError(f"Duplicate mismatched checkpoint: {key}")
        target[key] = row
    return rows, probes


def _write_checkpoint(root: Path, kind: str, row: Mapping[str, Any]) -> None:
    key = str(row["task_key"] if kind == "holdout" else row["probe_key"])
    payload = finalize_checkpoint(kind, row)
    _write_once(_checkpoint_path(root, key), payload)


def _prepare(snapshot: Path, ld: int):
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(snapshot)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, ld),
        identity_policy="extract_identity_phase",
    )
    return hamiltonian, preparation


def _frozen_source_evidence(
    pilot_artifact_path: Path,
    h5_snapshot: Path,
    h4_snapshot: Path,
) -> dict[str, Any]:
    actual_core_hashes = {
        path: file_sha256(path) for path in FROZEN_CORE_HASHES
    }
    pilot = _read_json(pilot_artifact_path)
    pilot_file_hash = file_sha256(pilot_artifact_path)
    return {
        "frozen_core_hashes": {
            "expected": FROZEN_CORE_HASHES,
            "actual": actual_core_hashes,
        },
        "frozen_core_hashes_match": actual_core_hashes == FROZEN_CORE_HASHES,
        "frozen_pilot_artifact": {
            "path": str(pilot_artifact_path),
            "expected_file_sha256": FROZEN_PILOT_FILE_SHA256,
            "actual_file_sha256": pilot_file_hash,
            "expected_content_fingerprint": FROZEN_PILOT_CONTENT_FINGERPRINT,
            "actual_content_fingerprint": pilot.get("content_fingerprint"),
        },
        "frozen_pilot_artifact_matches": (
            pilot_file_hash == FROZEN_PILOT_FILE_SHA256
            and pilot.get("content_fingerprint")
            == FROZEN_PILOT_CONTENT_FINGERPRINT
        ),
        "snapshots": {
            H5_PHYSICAL_TRANSFER.stratum_id: {
                "path": str(h5_snapshot),
                "sha256": file_sha256(h5_snapshot),
            },
            H4_COMPILER_TRANSFER.stratum_id: {
                "path": str(h4_snapshot),
                "sha256": file_sha256(h4_snapshot),
            },
        },
    }


def _build_expected_bundle(
    h5_preparation,
    h4_preparation,
) -> dict[str, Any]:
    strata = {
        H5_PHYSICAL_TRANSFER.stratum_id: build_expected_task_manifest(
            h5_preparation, H5_PHYSICAL_TRANSFER
        ),
        H4_COMPILER_TRANSFER.stratum_id: build_expected_task_manifest(
            h4_preparation, H4_COMPILER_TRANSFER
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": "pa_v1_blind_expected_bundle_v1",
        "created_before_compilation": True,
        "strata": strata,
    }
    payload["content_fingerprint"] = fingerprint(payload)
    return payload


def _validate_expected_bundle(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "pa_v1_blind_expected_bundle_v1":
        raise ValueError("Unsupported P-A blind expected bundle.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-A blind expected bundle fingerprint mismatch.")
    if set(payload.get("strata", {})) != {
        spec.stratum_id for spec in STRATA
    }:
        raise ValueError("P-A blind expected bundle strata are incomplete.")
    for manifest in payload["strata"].values():
        validate_expected_task_manifest(manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--h5-snapshot", type=Path, default=DEFAULT_H5_SNAPSHOT)
    parser.add_argument("--h4-snapshot", type=Path, default=DEFAULT_H4_SNAPSHOT)
    parser.add_argument(
        "--pilot-artifact", type=Path, default=DEFAULT_PILOT_ARTIFACT
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Freeze and validate inputs and expected event digests without compiling.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the current status file without starting work.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    output = root / OUTPUT_NAME
    expected_path = root / EXPECTED_NAME
    status_path = root / "status.json"
    plan_path = root / "plan.json"
    if args.status:
        if not status_path.exists():
            print("not_started")
            return 0
        print(status_path.read_text(encoding="utf-8"), end="")
        return 0

    if qiskit.__version__ != "1.3.0":
        raise RuntimeError(f"Expected Qiskit 1.3.0, found {qiskit.__version__}.")
    for path in (args.h5_snapshot, args.h4_snapshot, args.pilot_artifact):
        if not path.exists():
            raise FileNotFoundError(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)

    if output.exists():
        payload = _read_json(output)
        validate_blind_validation_artifact(payload)
        print(output)
        print(payload["content_fingerprint"])
        print(payload["decision"]["status"])
        return 0

    started = time.perf_counter()
    source_evidence = _frozen_source_evidence(
        args.pilot_artifact, args.h5_snapshot, args.h4_snapshot
    )
    if not source_evidence["frozen_core_hashes_match"]:
        raise RuntimeError("Frozen P-A v1 core hashes changed.")
    if not source_evidence["frozen_pilot_artifact_matches"]:
        raise RuntimeError("Frozen P-A pilot artifact changed.")

    h5_hamiltonian, h5_preparation = _prepare(
        args.h5_snapshot, H5_PHYSICAL_TRANSFER.ld
    )
    h4_hamiltonian, h4_preparation = _prepare(
        args.h4_snapshot, H4_COMPILER_TRANSFER.ld
    )
    generated_expected = _build_expected_bundle(
        h5_preparation, h4_preparation
    )
    if expected_path.exists():
        expected_bundle = _read_json(expected_path)
        _validate_expected_bundle(expected_bundle)
        if expected_bundle != generated_expected:
            raise ValueError("Saved expected tasks differ from regenerated tasks.")
    else:
        expected_bundle = generated_expected
        atomic_write_json(expected_path, expected_bundle)
    _validate_expected_bundle(expected_bundle)
    source_evidence["expected_task_bundle"] = {
        "path": str(expected_path),
        "sha256": file_sha256(expected_path),
        "content_fingerprint": expected_bundle["content_fingerprint"],
    }
    command = shlex.join([sys.executable, *sys.argv])
    plan = {
        "schema_version": "pa_v1_blind_run_plan_v1",
        "output": str(output),
        "expected_task_bundle": str(expected_path),
        "expected_task_bundle_content_fingerprint": expected_bundle[
            "content_fingerprint"
        ],
        "stratum_order": [spec.stratum_id for spec in STRATA],
        "expected_holdout_tasks": 48,
        "expected_operator_probes": 6,
        "resume_policy": "validate_and_reuse_fingerprinted_per-task_checkpoints",
    }
    if plan_path.exists():
        existing_plan = _read_json(plan_path)
        # The first dry-run plan recorded the invocation including --dry-run.
        # Invocation mode is provenance, not a scientific plan field.
        comparable_plan = dict(existing_plan)
        comparable_plan.pop("command", None)
        if comparable_plan != plan:
            raise ValueError(f"Saved run plan differs from fixed plan: {plan_path}")
        if existing_plan != plan:
            atomic_write_json(plan_path, plan)
    else:
        atomic_write_json(plan_path, plan)
    if args.dry_run:
        print(expected_path)
        print(expected_bundle["content_fingerprint"])
        print("dry_run_complete_no_compilation")
        return 0

    status: dict[str, Any] = {
        "schema_version": "pa_v1_blind_run_status_v1",
        "status": "running",
        "stage": H5_PHYSICAL_TRANSFER.stratum_id,
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "expected_holdout_tasks": 48,
        "expected_operator_probes": 6,
        "completed_holdout_tasks": 0,
        "completed_operator_probes": 0,
        "output": str(output),
    }
    atomic_write_json(status_path, status)
    try:
        existing_rows, existing_probes = _load_checkpoints(root)
        status["completed_holdout_tasks"] = len(existing_rows)
        status["completed_operator_probes"] = len(existing_probes)
        atomic_write_json(status_path, status)

        stratum_results: dict[str, Any] = {}
        inputs = (
            (
                H5_PHYSICAL_TRANSFER,
                h5_hamiltonian,
                h5_preparation,
            ),
            (
                H4_COMPILER_TRANSFER,
                h4_hamiltonian,
                h4_preparation,
            ),
        )
        for spec, hamiltonian, preparation in inputs:
            status["stage"] = spec.stratum_id
            status["updated_at_utc"] = _utc_now()
            atomic_write_json(status_path, status)
            manifest = expected_bundle["strata"][spec.stratum_id]
            compiler = _compiler(spec.optimization_level)
            stratum_existing_rows = {
                key: value
                for key, value in existing_rows.items()
                if key.startswith(spec.stratum_id + "__")
            }

            def on_row(row: Mapping[str, Any]) -> None:
                _write_checkpoint(root, "holdout", row)
                status["completed_holdout_tasks"] += 1
                status["updated_at_utc"] = _utc_now()
                status["last_completed"] = row["task_key"]
                atomic_write_json(status_path, status)

            rows = compile_stratum_rows(
                hamiltonian,
                preparation,
                compiler,
                spec,
                manifest,
                existing_rows=stratum_existing_rows,
                on_row=on_row,
            )
            stratum_existing_probes = {
                key: value
                for key, value in existing_probes.items()
                if key.startswith(spec.stratum_id + "__")
            }

            def on_probe(row: Mapping[str, Any]) -> None:
                _write_checkpoint(root, "operator_probe", row)
                status["completed_operator_probes"] += 1
                status["updated_at_utc"] = _utc_now()
                status["last_completed"] = row["probe_key"]
                atomic_write_json(status_path, status)

            probes = run_operator_probes(
                hamiltonian,
                preparation,
                spec,
                manifest,
                existing_probes=stratum_existing_probes,
                on_probe=on_probe,
            )
            stratum_results[spec.stratum_id] = summarize_stratum(
                hamiltonian,
                preparation,
                compiler,
                spec,
                manifest,
                rows,
                probes,
            )

        source_paths = (
            Path(
                "src/trotterlib/"
                "research_direction_joint_synthesis_blind_validation.py"
            ),
            Path(
                "scripts/"
                "run_research_direction_joint_synthesis_blind_validation.py"
            ),
        )
        provenance = {
            "generated_at_utc": _utc_now(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_worktree_status_before_generation": _git(
                ["status", "--short"]
            ),
            "evidence_status": "local_dirty_worktree_not_externally_reproduced",
            "command": command,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "qiskit_version": qiskit.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "source_sha256": {
                str(path): file_sha256(path) for path in source_paths
            },
        }
        artifact = finalize_blind_validation_artifact(
            stratum_results,
            source_evidence=source_evidence,
            provenance=provenance,
        )
        if output.exists():
            raise ValueError(f"Refusing to replace existing artifact: {output}")
        atomic_write_json(output, artifact)
        status.update(
            {
                "status": "completed",
                "stage": "completed",
                "updated_at_utc": _utc_now(),
                "finished_at_utc": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - started),
                "content_fingerprint": artifact["content_fingerprint"],
                "decision": artifact["decision"],
            }
        )
        atomic_write_json(status_path, status)
        print(output)
        print(artifact["content_fingerprint"])
        print(artifact["decision"]["status"])
        return 0
    except BaseException as exc:
        status.update(
            {
                "status": "failed",
                "stage": "failed",
                "updated_at_utc": _utc_now(),
                "finished_at_utc": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - started),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
