"""Result-prior, bounded M2 transfer implementation; import never opens data.

Only a separately committed execution authorization can cross the held-out
access boundary. Planning and tests use saved development JSON or synthetics.
Every scientific terminal status stops for external research-direction review.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import pr2_matched_accuracy_m2_transfer_contract as contract

CONTRACT_PLAN_PATH = (
    "artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/"
    "pr2_matched_accuracy_m2_transfer_zero_compute_plan_v2.json"
)
CONTRACT_PLAN_SHA256 = "d2bb5c5e57002fac5e8045f89a048913f4dadd5177d6f6d1465cc40a8755af7c"
CONTRACT_PLAN_FINGERPRINT = "7880c8fed57a02f30a07ff7463eb65e423098ad9420e38410f365fad2e24cc4f"
RESULT_SCHEMA_PATH = (
    "artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/"
    "pr2_matched_accuracy_m2_transfer_result_schema_v2.json"
)
PLAN_SCHEMA = "pr2_matched_accuracy_m2_execution_plan_v1"
AUTH_SCHEMA = "pr2_matched_accuracy_m2_execution_authorization_v1"
RESULT_SCHEMA = "pr2_matched_accuracy_m2_transfer_result_v2"
FAILURE = "IMPLEMENTATION_GATE_FAILED"
TOTAL_TIME = 0.8
HELD_OUT_IDENTITY = {
    "path_literal": contract.HELD_OUT_RELATIVE_LITERAL,
    # Public S0 snapshot metadata only; no held-out file was accessed to obtain these.
    "file_sha256": "ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37",
    "hamiltonian_hash": "a70d9619e794a0238aae57096a33759bf72d7b294a251b235508cd2ccc6b16c0",
    "state_hash": "cc882797fd8107f1da639d71c091da3f409a583595c208c60284ac9602a83d8c",
    "state_vector_hash": "b5a73ea469a7d171d0ceebd3eec4f3bc1490604ccd2b47bf0a1b8e208dbb992d",
    "identity_origin": "committed_S0_snapshot_metadata_not_new_held_out_access",
}
PERMISSIONS = {
    "held_out_access_authorized": True,
    "signal_evaluation_authorized": True,
    "trajectory_sampling_authorized": True,
    "circuit_build_authorized": True,
    "compile_authorized": True,
    "transfer_execution_authorized": True,
    "candidate_search_authorized": False,
    "additional_trajectories_authorized": False,
    "next_stage_authorized": False,
    "automatic_research_decision_authorized": False,
}
ENVIRONMENT = {
    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
}
PRIMARY_SOURCE_PATHS = (
    "src/trotterlib/pr2_matched_accuracy_m2_transfer_execution.py",
    "src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py",
    "scripts/run_pr2_matched_accuracy_m2_transfer.py",
    "tests/test_pr2_matched_accuracy_m2_transfer_execution.py",
    RESULT_SCHEMA_PATH,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True,
    ).stdout


def source_paths(root: Path, source_commit: str) -> tuple[str, ...]:
    """Freeze all local library Python, including transitively imported helpers."""
    paths = git(root, "ls-tree", "-r", "--name-only", source_commit, "--",
                "src/trotterlib").decode().splitlines()
    return tuple(sorted(set(PRIMARY_SOURCE_PATHS) | {
        path for path in paths if path.endswith(".py")
    }))


def committed_source_hashes(root: Path, source_commit: str) -> dict[str, str]:
    require(len(source_commit) == 40 and all(c in "0123456789abcdef" for c in source_commit),
            "source_commit must be a full Git commit")
    require(git(root, "rev-parse", "HEAD").decode().strip() == source_commit,
            "plan freeze requires HEAD equal source_commit")
    hashes = {}
    for path in source_paths(root, source_commit):
        digest = hashlib.sha256(git(root, "show", f"{source_commit}:{path}")).hexdigest()
        require(contract.file_sha256(root / path) == digest, f"uncommitted source: {path}")
        hashes[path] = digest
    return hashes


def environment_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "qiskit")},
        "compiler_identity": dict(contract.COMPILER_IDENTITY),
        "environment": dict(ENVIRONMENT),
    }


def wrapper_key(cell: Mapping[str, Any], source_commit: str, axis: str, index: int,
                seed: int | None) -> str:
    return contract.fingerprint({
        "policy": "m2_full_wrapper_source_candidate_axis_seed_v1",
        "source_commit": source_commit,
        "candidate_fingerprint": cell["execution_candidate_fingerprint"],
        "compiler": contract.COMPILER_IDENTITY,
        "wrapper_semantics": cell["transfer_configuration"]["wrapper_semantics"],
        "construction_policy": "boundary_optimized",
        "axis": axis, "trajectory_index": index, "trajectory_seed": seed,
    })


def build_execution_plan(contract_plan: Mapping[str, Any], *, source_commit: str,
                         source_hashes: Mapping[str, str],
                         environment: Mapping[str, Any]) -> dict[str, Any]:
    contract.validate_plan(contract_plan)
    require(contract_plan["status"] == contract.STATUS
            and contract_plan["plan_fingerprint"] == CONTRACT_PLAN_FINGERPRINT,
            "requires the frozen commit-bound contract v2")
    require(len(source_commit) == 40 and all(c in "0123456789abcdef" for c in source_commit),
            "invalid execution source commit")
    require(set(PRIMARY_SOURCE_PATHS) <= set(source_hashes), "missing execution source paths")
    require(all(len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                for v in source_hashes.values()), "invalid source SHA-256")
    require(environment["qiskit"] == contract.COMPILER_IDENTITY["qiskit_version"],
            "Qiskit identity mismatch")
    require(environment["compiler_identity"] == contract.COMPILER_IDENTITY
            and environment["environment"] == ENVIRONMENT, "environment policy mismatch")
    cells = json.loads(contract.canonical_json(contract_plan["frozen_candidates"]))
    for cell in cells:
        candidate = {
            "transfer_configuration_fingerprint": cell["transfer_configuration_fingerprint"],
            "held_out_identity": HELD_OUT_IDENTITY,
        }
        cell["execution_candidate_fingerprint"] = contract.fingerprint(candidate)
        seeds = cell["future_trajectory_seeds"] or [None]
        cell["wrapper_cache_keys"] = {
            axis: [wrapper_key(cell, source_commit, axis, index, seed)
                   for index, seed in enumerate(seeds)] for axis in contract.AXES
        }
        cell["task_fingerprint"] = contract.fingerprint(cell)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "M2_EXECUTION_PLAN_FROZEN_EXECUTION_NOT_AUTHORIZED",
        "source_commit": source_commit, "source_hashes": dict(sorted(source_hashes.items())),
        "contract_plan_sha256": CONTRACT_PLAN_SHA256,
        "contract_plan_fingerprint": CONTRACT_PLAN_FINGERPRINT,
        "held_out_identity": dict(HELD_OUT_IDENTITY),
        "environment_identity": dict(environment),
        "frozen_candidates": cells,
        "resource_caps": dict(contract_plan["resource_caps"]),
        "terminal_decision_rule": dict(contract.TERMINAL_DECISION_RULE),
        "execution_authorized": False, "next_stage_authorized": False,
        "zero_science_counters": dict(contract_plan["zero_science_counters"]),
    }
    plan["plan_fingerprint"] = contract.fingerprint(plan)
    return plan


def validate_execution_plan(plan: Mapping[str, Any], contract_plan: Mapping[str, Any]) -> None:
    expected = build_execution_plan(
        contract_plan, source_commit=plan["source_commit"],
        source_hashes=plan["source_hashes"], environment=plan["environment_identity"],
    )
    require(plan == expected, "execution plan differs from frozen contract/source/seed schedule")
    keys = [key for cell in plan["frozen_candidates"]
            for values in cell["wrapper_cache_keys"].values() for key in values]
    require(len(keys) == len(set(keys)) == 196, "missing or duplicated wrapper identities")


def validate_authorization(root: Path, authorization: Mapping[str, Any],
                           plan: Mapping[str, Any], *, plan_sha256: str,
                           authorization_path: str, authorization_sha256: str,
                           output_relative: str, workers: int) -> None:
    # This function deliberately does not construct a held-out Path.
    require(type(workers) is int and 1 <= workers <= contract.MAXIMUM_WORKERS,
            "workers outside 1..5")
    require(authorization.get("schema_version") == AUTH_SCHEMA
            and authorization.get("status") == "M2_EXECUTION_AUTHORIZED_ONCE",
            "M2 execution is not authorized")
    expected = {
        "source_commit": plan["source_commit"], "source_hashes": plan["source_hashes"],
        "execution_plan_sha256": plan_sha256,
        "execution_plan_fingerprint": plan["plan_fingerprint"],
        "contract_plan_sha256": CONTRACT_PLAN_SHA256,
        "held_out_identity": HELD_OUT_IDENTITY, "permissions": PERMISSIONS,
        "resource_caps": plan["resource_caps"], "execution_run_limit": 1,
        "fixed_output_relative": output_relative,
        "result_terminal_statuses": list(contract.TRANSFER_STATUSES),
    }
    require(all(authorization.get(k) == v for k, v in expected.items()),
            "authorization identity/permissions/budget mismatch")
    require(output_relative.startswith("artifacts/") and ".." not in Path(output_relative).parts
            and not Path(output_relative).is_absolute()
            and "held_out" not in output_relative
            and output_relative != CONTRACT_PLAN_PATH,
            "output must be an explicit safe artifact directory")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    require(git(root, "merge-base", plan["source_commit"], head).decode().strip()
            == plan["source_commit"], "execution source is not an ancestor of HEAD")
    require(git(root, "merge-base", contract.M1_B1_EVIDENCE_COMMIT,
                plan["source_commit"]).decode().strip() == contract.M1_B1_EVIDENCE_COMMIT,
            "source ancestry mismatch")
    require(set(plan["source_hashes"]) == set(source_paths(root, plan["source_commit"])),
            "transitive source manifest is incomplete")
    for path, digest in plan["source_hashes"].items():
        require(contract.file_sha256(root / path) == digest
                == hashlib.sha256(git(root, "show", f"{plan['source_commit']}:{path}")).hexdigest(),
                f"source identity mismatch: {path}")
    require(hashlib.sha256(git(root, "show", f"HEAD:{authorization_path}")).hexdigest()
            == authorization_sha256, "authorization must be committed before execution")
    # Exact package identity is frozen by the future execution plan, not inferred.
    require(environment_identity() == plan["environment_identity"], "environment identity mismatch")
    require(all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), "BLAS/process environment mismatch")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(contract.canonical_json(payload) + b"\n")
    temporary.replace(path)


def load_held_out(root: Path, counters: dict[str, int]) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Private authorized boundary, invoked only after all result-prior gates."""
    from .pr2_new_series_validation import _load_snapshot_once
    from .pr2_s0_s1_validation import _to_qiskit_state

    path = root / contract.HELD_OUT_RELATIVE_LITERAL
    counters["held_out_hash_reads"] += 1
    require(contract.file_sha256(path) == HELD_OUT_IDENTITY["file_sha256"],
            "held-out byte identity mismatch")
    counters["held_out_npz_loads"] += 1
    hamiltonian, _sector, state, _sector_state, metadata, _layout = _load_snapshot_once(path)
    for key in ("hamiltonian_hash", "state_hash", "state_vector_hash"):
        require(metadata[key] == HELD_OUT_IDENTITY[key], f"held-out {key} mismatch")
    require(hamiltonian.n_qubits == 8 and hamiltonian.n_blocks == 12, "held-out model dimension mismatch")
    require(metadata["hamiltonian_metadata"]["distance"] == 1.30
            and metadata["hamiltonian_metadata"]["basis"].lower() == "sto-3g",
            "held-out geometry/basis mismatch")
    return hamiltonian, _to_qiskit_state(state, hamiltonian.n_qubits), metadata


def evaluate_signals(hamiltonian: Any, state: np.ndarray, cells: Sequence[Mapping[str, Any]]
                     ) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any]]:
    """The five frozen M1 formulae, not a selector or parameter search."""
    from .pr2_matched_accuracy_m1_execution import (
        _prepare, _prepare_discard, _target_signal, _dense_block_operators,
        _preparation_eigensystems, _deterministic_signal_record, _random_signal_record,
        _eigendecomposition, RECONSTRUCTION_TOLERANCE,
    )
    from .pr2_s0_s1_validation import _dense_df_operator_qiskit
    from .df_rte_tail import extraction_to_normalized_rte_tail

    require([item["candidate_id"] for item in cells]
            == [item["candidate_id"] for item in contract.FROZEN_CANDIDATES], "signal candidate set mismatch")
    require(abs(float(np.vdot(state, state).real) - 1.0) <= 1e-12
            and np.isfinite(state).all(), "state normalization gate failed")
    energy, target, residual = _target_signal(hamiltonian, state)
    one_body, fragments, reconstruction = _dense_block_operators(hamiltonian)
    full = _dense_df_operator_qiskit(hamiltonian)
    block_cache: dict[str, Any] = {}
    preparations, records, tail_errors = [], [], {}
    for cell in cells:
        method, rank = cell["method"], int(cell["rank"])
        prep = _prepare_discard(hamiltonian, rank) if method == "B0" else _prepare(hamiltonian, rank)
        preparations.append(prep)
        deterministic = _preparation_eigensystems(prep, one_body, fragments, block_cache)
        candidate = dict(cell["transfer_configuration"])
        candidate.update(candidate_id=cell["candidate_id"],
                         candidate_fingerprint=cell["execution_candidate_fingerprint"])
        if method in {"B0", "B1"}:
            signal = _deterministic_signal_record(candidate, prep, deterministic, state, target)
        else:
            normalized = extraction_to_normalized_rte_tail(
                prep.tail_extraction, max_dense_qubits=8,
            ).normalized_hamiltonian
            prefix = _dense_df_operator_qiskit(hamiltonian.select_blocks(prep.deterministic_fragment_indices))
            error = float(np.linalg.norm(full - prefix - prep.extracted_identity_coefficient
                                         * np.eye(len(state)) - prep.exact_rte_lambda_r * normalized, ord=2))
            require(error / max(1.0, float(np.linalg.norm(full, ord=2))) <= RECONSTRUCTION_TOLERANCE,
                    "held-out tail reconstruction gate failed")
            tail_errors[cell["candidate_id"]] = error
            signal = _random_signal_record(candidate, prep, deterministic,
                                           _eigendecomposition(normalized), state, target)
        require(math.isfinite(signal["normalization_multiplier"])
                and math.isfinite(signal["corrected_bias_abs"]), "nonfinite signal gate")
        signal["accuracy_feasibility_misclassification"] = not signal["accuracy_eligible"]
        signal["signal_record_fingerprint"] = contract.fingerprint(signal)
        records.append(signal)
    return records, preparations, {
        "reference_energy": energy, "reference_state_rayleigh_residual": residual,
        "block_reconstruction": reconstruction, "tail_reconstruction_absolute": tail_errors,
        "ground_state_solves": 0, "method_parameter_searches": 0,
    }


def _trajectory_request(preparation: Any, cell: Mapping[str, Any], seed: int | None) -> Any:
    """Use the plan's seed directly; keep canonical independent step/occurrence sampling."""
    from .df_partial_s2_repeated_cost import _request_from_step_event_sequences
    from .pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    from .rte import make_rte_config

    cfg = cell["transfer_configuration"]
    q, delta = int(cfg["q"]), float(cfg["delta"])
    require(delta == TOTAL_TIME / q, "dynamic delta mismatch")
    if seed is None:
        require(cell["method"] in {"B0", "B1"}, "missing random trajectory seed")
        config, distribution, step_seeds = None, None, (None,) * q
        sequences = ((),) * q
    else:
        require(cell["method"] in {"B2", "B3"}, "random seed in deterministic candidate")
        r, cutoff = int(cfg["r"]), int(cfg["K"])
        tau = preparation.exact_rte_lambda_r * delta / r
        config, distribution = make_rte_config(
            preparation.rte_preparation.symbolic_tail, evolution_time=delta,
            rte_steps=r, finite_taylor_order=cutoff,
            truncation_tolerance=_explicit_cutoff_tolerance(tau, cutoff), seed=seed,
        )
        if q == 1:
            step_seeds = (seed,)
        else:
            rng = random.Random(seed)
            step_seeds = tuple(rng.randrange(2**63) for _ in range(q))
        sequences = tuple(tuple(preparation.rte_preparation.iter_sample_events(
            distribution, sample_count=r, seed=step_seed,
        )) for step_seed in step_seeds)
    return _request_from_step_event_sequences(
        preparation, step_time=delta, repetition_count=q, config=config, distribution=distribution,
        step_event_sequences=sequences, step_seeds=step_seeds, master_seed=contract.MASTER_SEED,
        trajectory_seed=seed, sampling_policy="m2_frozen_trajectory_step_occurrence_v1",
        controlled=True, ancilla_qubit=preparation.num_system_qubits,
        cancel_adjacent_equal_bases=True, construction_policy="boundary_optimized",
    )


@dataclass(frozen=True)
class CompileJob:
    cell: Mapping[str, Any]
    preparation: Any
    checkpoint_directory: str


def _read_wrapper_checkpoint(path: Path, expected_key: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = contract.load_json(path)
    require(payload.get("wrapper_key") == expected_key, "checkpoint wrapper identity mismatch")
    require(payload.get("state") == "complete", "unresolved compile attempt: automatic retry prohibited")
    body = {k: v for k, v in payload.items() if k != "checkpoint_fingerprint"}
    require(payload.get("checkpoint_fingerprint") == contract.fingerprint(body), "checkpoint checksum mismatch")
    require(set(payload["metrics"]) == set(contract.METRICS)
            and all(type(v) is int and v >= 0 for v in payload["metrics"].values()),
            "checkpoint metric gate failed")
    require(isinstance(payload.get("actual_circuit_fingerprint"), str)
            and len(payload["actual_circuit_fingerprint"]) == 64, "checkpoint missing circuit identity")
    return payload


def compile_cell(job: CompileJob) -> dict[str, Any]:
    from .df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder
    from .rpe_hadamard_compiled_cost_benchmark import QiskitRPEHadamardBenchmarkCircuitBuilder
    from .rpe_hadamard_interrogation import RPEHadamardInterrogationRequest
    from .rte import CompilerSettings
    from .rte_compiled_cost import transpile_and_measure_cost

    cell = job.cell
    directory = Path(job.checkpoint_directory)
    seeds = cell["future_trajectory_seeds"] or [None]
    require(len(seeds) == (32 if cell["method"] in {"B2", "B3"} else 1), "trajectory cap mismatch")
    rows, computed, reused, sampled, occurrences = [], 0, 0, 0, 0
    started = time.perf_counter()
    cpu_started = time.process_time()
    compiler = CompilerSettings(**{**contract.COMPILER_IDENTITY,
                                  "basis_gates": tuple(contract.COMPILER_IDENTITY["basis_gates"])})
    for index, seed in enumerate(seeds):
        saved = {axis: _read_wrapper_checkpoint(directory / f"{index:02d}_{axis}.json",
                                                cell["wrapper_cache_keys"][axis][index])
                 for axis in contract.AXES}
        row = {"trajectory_index": index, "trajectory_seed": seed, "axes": {}}
        # One request and evolution are shared by the two measured axes.
        evolution = None
        if any(record is None for record in saved.values()):
            request = _trajectory_request(job.preparation, cell, seed)
            if seed is not None:
                sampled += 1
                occurrences += int(cell["q"]) * int(cell["r"])
            evolution = QiskitDFPartialS2RepeatedCircuitBuilder().build(
                request, construction_policy="boundary_optimized",
            )
        for axis in contract.AXES:
            path = directory / f"{index:02d}_{axis}.json"
            key = cell["wrapper_cache_keys"][axis][index]
            record = saved[axis]
            if record is not None:
                reused += 1
            else:
                wrapper = QiskitRPEHadamardBenchmarkCircuitBuilder(maximum_repetition_count=8).build(
                    RPEHadamardInterrogationRequest(evolution=evolution, axis=axis, include_measurement=True),
                )
                require(wrapper.include_measurement and wrapper.circuit.num_clbits == 1
                        and not wrapper.state_preparation_included and not wrapper.additional_control_applied,
                        "full measured wrapper correctness gate failed")
                require(wrapper.circuit.size() <= 10_000_000, "wrapper instruction safety cap")
                # Persist BEFORE invoking the compiler. An interrupted unresolved attempt
                # cannot be silently recompiled and exceed the one-shot budget.
                _atomic_json(path, {"wrapper_key": key, "state": "compile_reserved"})
                cost = transpile_and_measure_cost(wrapper.circuit, compiler)
                require(cost.actual_circuit_fingerprint is not None, "missing actual circuit fingerprint")
                record = {
                    "wrapper_key": key, "state": "complete",
                    "metrics": {m: int(getattr(cost, m)) for m in contract.METRICS},
                    "actual_circuit_fingerprint": cost.actual_circuit_fingerprint,
                    "wrapper_semantics_fingerprint": wrapper.wrapper_circuit_semantics_fingerprint,
                    "shared_evolution_fingerprint": wrapper.wrapped_trajectory_fingerprint,
                    "step_seeds": list(request.step_seeds),
                }
                record["checkpoint_fingerprint"] = contract.fingerprint(record)
                _atomic_json(path, record)
                computed += 1
            row["axes"][axis] = record
        require(row["axes"]["cosine"]["shared_evolution_fingerprint"]
                == row["axes"]["sine"]["shared_evolution_fingerprint"], "paired axis trajectory mismatch")
        rows.append(row)
    return {
        "candidate_id": cell["candidate_id"], "task_fingerprint": cell["task_fingerprint"],
        "paired_trajectory_rows": rows, "full_wrappers_computed": computed,
        "full_wrappers_reused": reused, "wall_time_s": time.perf_counter() - started,
        "trajectory_samples": sampled, "occurrence_samples": occurrences,
        "cpu_time_s": time.process_time() - cpu_started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def candidate_summary(cell: Mapping[str, Any], signal: Mapping[str, Any],
                      compiled: Mapping[str, Any]) -> dict[str, Any]:
    require(signal["candidate_id"] == cell["candidate_id"]
            and signal["candidate_fingerprint"] == cell["execution_candidate_fingerprint"],
            "signal/cost candidate fingerprint mismatch")
    require(compiled["candidate_id"] == cell["candidate_id"]
            and compiled["task_fingerprint"] == cell["task_fingerprint"], "compile task mismatch")
    rows = compiled["paired_trajectory_rows"]
    expected_seeds = cell["future_trajectory_seeds"] or [None]
    require([r["trajectory_seed"] for r in rows] == expected_seeds, "compiled seed schedule mismatch")
    require([r["trajectory_index"] for r in rows] == list(range(len(expected_seeds))), "row index mismatch")
    for row in rows:
        for axis in contract.AXES:
            item = row["axes"][axis]
            require(item["wrapper_key"] == cell["wrapper_cache_keys"][axis][row["trajectory_index"]],
                    "compiled wrapper key mismatch")
            require(set(item["metrics"]) == set(contract.METRICS)
                    and all(type(v) is int and v >= 0 for v in item["metrics"].values()),
                    "compiled metric gate failed")
    means = {axis: {m: float(np.mean([row["axes"][axis]["metrics"][m] for row in rows]))
                    for m in contract.METRICS} for axis in contract.AXES}
    eligible = signal["accuracy_eligible"] is True
    work, se, prediction, underestimate = {}, {}, {}, {}
    if eligible:
        shots = signal["axis_shots"]
        require(all(type(shots[a]) is int and shots[a] > 0 for a in ("real", "imag")),
                "eligible shot count is invalid")
        for metric in contract.METRICS:
            # Covariance is retained: combine the paired axes BEFORE taking SE.
            paired = np.asarray([shots["real"] * row["axes"]["cosine"]["metrics"][metric]
                                 + shots["imag"] * row["axes"]["sine"]["metrics"][metric]
                                 for row in rows], dtype=float)
            work[metric] = float(np.mean(paired))
            se[metric] = float(np.std(paired, ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else 0.0
            prediction[metric] = contract.predicted_work(
                cell["development_prediction"]["axis_one_shot_compiled_cost"], shots, metric,
            )
            underestimate[metric] = contract.underestimate_fraction(
                predicted=prediction[metric], actual=work[metric],
            )
            require(math.isfinite(work[metric]) and math.isfinite(se[metric]), "nonfinite work")
    major = eligible and work["rz_count"] > contract.MATERIALITY_RATIO * prediction["rz_count"]
    return {
        "candidate_id": cell["candidate_id"], "method": cell["method"],
        "rank": cell["rank"], "q": cell["q"], "r": cell["r"], "K": cell["K"],
        "development_candidate_fingerprint": cell["candidate_fingerprint"],
        "execution_candidate_fingerprint": cell["execution_candidate_fingerprint"],
        "accuracy_eligible": eligible,
        "accuracy_feasibility_misclassification": not eligible,
        "major_cost_underestimate": bool(major),
        "transfer_support_usable": bool(cell["method"] == "B2" and eligible and not major),
        "axis_shots": dict(signal["axis_shots"]), "signal": dict(signal),
        "axis_one_shot_compiled_means": means,
        "work_by_metric": work, "work_standard_errors": se,
        "predicted_work_by_metric": prediction, "underestimate_fraction_by_metric": underestimate,
        "primary_work": work.get("rz_count"), "primary_standard_error": se.get("rz_count"),
        "point_six_metric_pareto": False, "compiled": dict(compiled),
    }


def secondary_envelope(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in records if r["accuracy_eligible"]]
    if not eligible:
        return {"crossings": [], "intervals": [], "secondary_only": True}
    lines = [(r["candidate_id"], r["primary_work"], sum(r["axis_shots"].values())) for r in eligible]
    crossings = []
    points = {0.0}
    for index, left in enumerate(lines):
        for right in lines[index + 1:]:
            if left[2] == right[2]:
                continue
            p = (right[1] - left[1]) / (left[2] - right[2])
            if p >= 0 and math.isfinite(p):
                crossings.append({"left": left[0], "right": right[0], "P": p})
                points.add(p)
    points = sorted(points)
    intervals = []
    for index, start in enumerate(points):
        end = points[index + 1] if index + 1 < len(points) else None
        probe = (start + end) / 2 if end is not None else start + max(1.0, start)
        winner = min(lines, key=lambda line: line[1] + line[2] * probe)
        intervals.append({"P_start_inclusive": start, "P_end": end,
                          "candidate_id": winner[0], "rz_intercept": winner[1],
                          "shot_slope": winner[2]})
    return {"crossings": crossings, "intervals": intervals, "secondary_only": True}


def assemble_result(plan: Mapping[str, Any], signals: Sequence[Mapping[str, Any]],
                    compiled: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells = plan["frozen_candidates"]
    require(len(signals) == len(compiled) == len(cells) == 5, "result candidate count mismatch")
    records = [candidate_summary(cell, signal, cost)
               for cell, signal, cost in zip(cells, signals, compiled, strict=True)]
    eligible = [r for r in records if r["accuracy_eligible"]]
    for record in eligible:
        target = record["work_by_metric"]
        record["point_six_metric_pareto"] = not any(
            all(other["work_by_metric"][m] <= target[m] for m in contract.METRICS)
            and any(other["work_by_metric"][m] < target[m] for m in contract.METRICS)
            for other in eligible if other is not record
        )
    decision = contract.classify_transfer(records)
    result = {
        "schema_version": RESULT_SCHEMA, "series_id": contract.SERIES_ID,
        "status": decision["status"], "transfer_decision": decision,
        "transfer_plan_fingerprint": CONTRACT_PLAN_FINGERPRINT,
        "execution_plan_fingerprint": plan["plan_fingerprint"], "source_commit": plan["source_commit"],
        "candidate_results": records, "terminal_decision_rule": dict(contract.TERMINAL_DECISION_RULE),
        "primary_metric": "rz_count", "point_pareto_metrics": list(contract.METRICS),
        "state_preparation_lower_envelope": secondary_envelope(records),
        "held_out_reoptimized": False, "held_out_candidate_searches": 0,
        "additional_trajectories_executed": 0, "next_stage_authorized": False,
        "automatic_next_stage": None, "mandatory_stop_reached": True,
        "formal_confidence_interval_claimed": False,
    }
    return result


def validate_result(result: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    from jsonschema import Draft202012Validator
    Draft202012Validator(schema).validate(result)
    expected = [c["candidate_id"] for c in contract.FROZEN_CANDIDATES]
    require([r["candidate_id"] for r in result["candidate_results"]] == expected,
            "terminal result candidate identity mismatch")
    require(result["mandatory_stop_reached"] is True and result["next_stage_authorized"] is False,
            "terminal mandatory STOP mismatch")
    require(result["transfer_decision"] == contract.classify_transfer(result["candidate_results"]),
            "terminal decision mismatch")
    require(result["status"] == result["transfer_decision"]["status"], "result/decision status mismatch")


def partial_wrapper_audit(runtime: Path, cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete, unresolved, unreadable = 0, 0, 0
    for cell in cells:
        for axis in contract.AXES:
            for index, key in enumerate(cell["wrapper_cache_keys"][axis]):
                path = runtime / "checkpoints" / cell["task_fingerprint"] / f"{index:02d}_{axis}.json"
                if not path.exists():
                    continue
                try:
                    payload = contract.load_json(path)
                    if payload.get("state") == "compile_reserved" and payload.get("wrapper_key") == key:
                        unresolved += 1
                    else:
                        _read_wrapper_checkpoint(path, key)
                        complete += 1
                except Exception:
                    unreadable += 1
    return {"complete_records": complete, "unresolved_reservations": unresolved,
            "unreadable_records": unreadable,
            "successful_compiles_lower_bound": complete,
            "not_a_complete_resource_count": True}


def run_transfer(root: Path, *, plan_path: Path, authorization_path: Path,
                 output_relative: str, workers: int = 5) -> dict[str, Any]:
    """Not called by implementation tests; actual execution needs later review."""
    for public_path in (plan_path, authorization_path):
        require(public_path.suffix == ".json" and "held_out" not in str(public_path),
                "plan/authorization must be public non-held-out JSON")
    plan = contract.load_json(plan_path)
    contract_plan = contract.load_json(root / CONTRACT_PLAN_PATH)
    require(contract.file_sha256(root / CONTRACT_PLAN_PATH) == CONTRACT_PLAN_SHA256, "contract plan hash mismatch")
    validate_execution_plan(plan, contract_plan)
    authorization = contract.load_json(authorization_path)
    auth_sha = contract.file_sha256(authorization_path)
    plan_sha = contract.file_sha256(plan_path)
    auth_relative = str(authorization_path.relative_to(root))
    validate_authorization(root, authorization, plan, plan_sha256=plan_sha,
                           authorization_path=auth_relative, authorization_sha256=auth_sha,
                           output_relative=output_relative, workers=workers)
    schema = contract.load_json(root / RESULT_SCHEMA_PATH)
    output = root / output_relative
    identity = {"source_commit": plan["source_commit"], "plan_sha256": plan_sha,
                "authorization_sha256": auth_sha, "workers": workers, "output_relative": output_relative}
    runtime = output / ".runtime"
    # Single committed authorization + fixed output + exclusive registry reservation.
    registry = root / "artifacts/pr2_m2_execution_registry" / (auth_sha + ".json")
    require(not output.exists() and not registry.exists(), "authorization/output already used")
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("xb") as handle:
        handle.write(contract.canonical_json(identity) + b"\n")
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(runtime / "run_identity.json", identity)
    counters = {name: 0 for name in contract.ZERO_SCIENCE_COUNTER_NAMES}
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        # ONLY here, after source, authorization, compiler, output, and budget gates.
        hamiltonian, state, _metadata = load_held_out(root, counters)
        signals, preparations, numerical_audit = evaluate_signals(hamiltonian, state, plan["frozen_candidates"])
        counters["signal_evaluations"] = len(signals)
        jobs = [CompileJob(cell, preparation, str(runtime / "checkpoints" / cell["task_fingerprint"]))
                for cell, preparation in zip(plan["frozen_candidates"], preparations, strict=True)]
        costs = {}
        if workers == 1:
            for job in jobs:
                costs[job.cell["candidate_id"]] = compile_cell(job)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
                futures = {executor.submit(compile_cell, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    costs[job.cell["candidate_id"]] = future.result()
        ordered = [costs[cell["candidate_id"]] for cell in plan["frozen_candidates"]]
        result = assemble_result(plan, signals, ordered)
        computed = sum(r["full_wrappers_computed"] for r in ordered)
        reused = sum(r["full_wrappers_reused"] for r in ordered)
        require(computed + reused == 196, "full wrapper total exceeds/is below frozen cap")
        counters.update(full_wrappers_compiled=computed, compiler_invocations=computed,
                        trajectory_samples=sum(c["trajectory_samples"] for c in ordered),
                        occurrence_samples=sum(c["occurrence_samples"] for c in ordered),
                        circuits_built=computed)
        result.update(
            authorization_sha256=auth_sha, execution_plan_sha256=plan_sha,
            input_snapshot_identity=dict(HELD_OUT_IDENTITY), numerical_audit=numerical_audit,
            execution={
                "started_utc": started_at, "ended_utc": datetime.now(timezone.utc).isoformat(),
                "wall_time_s": time.perf_counter() - started,
                "parent_cpu_time_s": time.process_time() - cpu_started,
                "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "max_worker_peak_rss_kib": max(r["peak_rss_kib"] for r in ordered),
                "process_workers": workers, "process_start_method": "spawn",
                "blas_threads_per_worker": 1, "environment_identity": environment_identity(),
                "full_wrappers_computed_this_invocation": computed,
                "full_wrappers_reused_this_invocation": reused,
                "unique_wrapper_evaluations": computed + reused,
                "counters": counters, "gpu_query_allocation_kernel": [0, 0, 0],
            },
        )
        result["result_fingerprint"] = contract.fingerprint(result)
        validate_result(result, schema)
        _atomic_json(output / "pr2_matched_accuracy_m2_transfer_result_v2.json", result)
        manifest = {"files": [{"path": "pr2_matched_accuracy_m2_transfer_result_v2.json",
                              "sha256": contract.file_sha256(output / "pr2_matched_accuracy_m2_transfer_result_v2.json"),
                              "bytes": (output / "pr2_matched_accuracy_m2_transfer_result_v2.json").stat().st_size}]}
        _atomic_json(output / "manifest.json", manifest)
        _atomic_json(output / "M2_COMPLETE.json", {
            "status": result["status"], "result_fingerprint": result["result_fingerprint"],
            "next_stage_authorized": False, "mandatory_stop_reached": True,
        })
        return result
    except Exception as exc:
        partial = partial_wrapper_audit(runtime, plan["frozen_candidates"])
        for counter in ("trajectory_samples", "occurrence_samples", "circuits_built",
                        "compiler_invocations", "full_wrappers_compiled"):
            counters[counter] = None  # Unknown/partial is not zero.
        _atomic_json(output / "M2_FAILURE.json", {
            "status": FAILURE, "exception_type": type(exc).__name__, "exception_message": str(exc),
            "stage": "authorized_execution", "counts_may_be_partial": True,
            "resource_audit_policy": "per_wrapper_checkpoint_ledgers_are_authoritative",
            "counters": counters, "wrapper_checkpoint_audit": partial,
            "next_stage_authorized": False, "mandatory_stop_reached": True,
        })
        raise
