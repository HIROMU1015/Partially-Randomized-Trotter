"""Result-prior execution path for the bounded PR-2 M1-B1 compile map.

The module consumes the frozen M1-A signal result.  It does not recompute a
signal, inspect a held-out path, select candidates, extend sampling beyond 32
trajectories, or turn the compiled map into a research decision.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Mapping, Sequence

from .df_partial_s2 import DFPartialS2Preparation
from .pr2_matched_accuracy_m1_b1_contract import (
    AXES,
    BASELINE_CELL_COUNT,
    COMPILER_IDENTITY,
    MAXIMUM_WORKERS,
    M1_A_RESULT_FINGERPRINT,
    M1_A_RESULT_RELATIVE,
    M1_A_RESULT_SHA256,
    RANDOM_CELL_COUNT,
    RANDOM_TRAJECTORY_COUNT,
    RANDOM_WRAPPER_COUNT,
    SERIES_ID,
    TOTAL_WRAPPER_COUNT,
    TRAJECTORIES_PER_RANDOM_CELL,
    canonical_json,
    file_sha256,
    fingerprint,
    frozen_cells,
    load_json,
    validate_m1_a_result,
)
from .pr2_matched_accuracy_m1_execution import (
    DEVELOPMENT_RELATIVE_PATH,
    _explicit_cutoff_tolerance,
    _load_development_only,
    _prepare,
    _prepare_discard,
)
from .rpe_hadamard_compiled_cost_benchmark import (
    RPEHadamardCompiledCostBenchmarkRequest,
    generate_rpe_hadamard_compiled_cost_benchmark_dataset,
)
from .rte import CompilerSettings, make_rte_config
from .rte_compiled_cost import TranspiledCircuitCostCache


PLAN_SCHEMA_VERSION = "pr2_matched_accuracy_m1_b1_execution_plan_v2"
AUTHORIZATION_SCHEMA_VERSION = "pr2_matched_accuracy_m1_b1_execution_authorization_v1"
RESULT_SCHEMA_VERSION = "pr2_matched_accuracy_m1_b1_result_v2"
PLAN_STATUS = "M1_B1_EXECUTION_PLAN_FROZEN_EXECUTION_NOT_AUTHORIZED"
COMPLETE_STATUS = "M1_B1_COMPILE_MAP_COMPLETE_AWAITING_REVIEW"
FAILURE_STATUS = "IMPLEMENTATION_GATE_FAILED"
PROCESS_START_METHOD = "spawn"
TOTAL_TIME = 0.8
BASELINE_WRAPPER_COUNT = 32
RESEARCH_DECISIONS = (
    "CONTINUE_RESOURCE_STUDY",
    "NARROW_TO_TECHNICAL_NOTE",
    "STOP_DUPLICATIVE",
    "COMPILE_RESULT_INCONCLUSIVE",
)


def _full_commit(value: str) -> str:
    text = str(value)
    if len(text) != 40 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError("source_commit must be a full lowercase Git object name")
    return text


def _request_master_seed(source_commit: str, candidate_fingerprint: str) -> int:
    payload = {
        "policy": "pr2_m1_b1_request_master_seed_sha256_v2",
        "source_commit": _full_commit(source_commit),
        "candidate_fingerprint": candidate_fingerprint,
    }
    return int.from_bytes(hashlib.sha256(canonical_json(payload)).digest()[:8], "big") % (
        2**63
    )


def _point_seed(request_master_seed: int, q: int) -> int:
    digest = hashlib.sha256(
        (
            "rpe_hadamard_benchmark_partition_seed_v1|"
            f"{request_master_seed}|calibration|{q}"
        ).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def expected_trajectory_seeds(request_master_seed: int, q: int) -> list[int]:
    rng = random.Random(_point_seed(request_master_seed, q))
    return [rng.randrange(0, 2**63) for _ in range(TRAJECTORIES_PER_RANDOM_CELL)]


def _wrapper_key(
    *,
    source_commit: str,
    candidate_fingerprint: str,
    compiler_fingerprint: str,
    axis: str,
    trajectory_index: int | None,
    trajectory_seed: int | None,
) -> str:
    return fingerprint(
        {
            "schema_version": "pr2_matched_accuracy_m1_b1_wrapper_identity_v2",
            "source_commit": _full_commit(source_commit),
            "m1_a_result_sha256": M1_A_RESULT_SHA256,
            "compiler_fingerprint": compiler_fingerprint,
            "candidate_fingerprint": candidate_fingerprint,
            "axis": axis,
            "trajectory_index": trajectory_index,
            "trajectory_seed": trajectory_seed,
            "wrapper_semantics": (
                "full_measured_hadamard_wrapper_without_state_preparation"
            ),
        }
    )


def build_execution_plan(
    *,
    m1_a_result: Mapping[str, Any],
    m1_a_result_sha256: str,
    source_commit: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build a zero-compute plan tied to the actual execution source commit."""
    source_commit = _full_commit(source_commit)
    validate_m1_a_result(m1_a_result, sha256=m1_a_result_sha256)
    random_cells, baseline_cells = frozen_cells(m1_a_result)
    compiler_fingerprint = fingerprint(COMPILER_IDENTITY)

    planned_random: list[dict[str, Any]] = []
    planned_baseline: list[dict[str, Any]] = []
    all_keys: list[str] = []
    for cell in random_cells:
        master_seed = _request_master_seed(source_commit, cell["candidate_fingerprint"])
        seeds = expected_trajectory_seeds(master_seed, int(cell["q"]))
        keys = {
            axis: [
                _wrapper_key(
                    source_commit=source_commit,
                    candidate_fingerprint=cell["candidate_fingerprint"],
                    compiler_fingerprint=compiler_fingerprint,
                    axis=axis,
                    trajectory_index=index,
                    trajectory_seed=seed,
                )
                for index, seed in enumerate(seeds)
            ]
            for axis in AXES
        }
        all_keys.extend(key for axis in AXES for key in keys[axis])
        planned_random.append(
            {
                **cell,
                "request_master_seed": master_seed,
                "point_master_seed": _point_seed(master_seed, int(cell["q"])),
                "sampled_trajectory_seeds": seeds,
                "wrapper_cache_keys": keys,
                "task_fingerprint": fingerprint(
                    {
                        "source_commit": source_commit,
                        "compiler_fingerprint": compiler_fingerprint,
                        "candidate": cell,
                        "request_master_seed": master_seed,
                        "sampled_trajectory_seeds": seeds,
                    }
                ),
            }
        )
    for cell in baseline_cells:
        keys = {
            axis: [
                _wrapper_key(
                    source_commit=source_commit,
                    candidate_fingerprint=cell["candidate_fingerprint"],
                    compiler_fingerprint=compiler_fingerprint,
                    axis=axis,
                    trajectory_index=None,
                    trajectory_seed=None,
                )
            ]
            for axis in AXES
        }
        all_keys.extend(key for axis in AXES for key in keys[axis])
        planned_baseline.append(
            {
                **cell,
                "request_master_seed": None,
                "point_master_seed": None,
                "sampled_trajectory_seeds": None,
                "wrapper_cache_keys": keys,
                "task_fingerprint": fingerprint(
                    {
                        "source_commit": source_commit,
                        "compiler_fingerprint": compiler_fingerprint,
                        "candidate": cell,
                        "request_master_seed": None,
                    }
                ),
            }
        )

    body: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "series_id": SERIES_ID,
        "status": PLAN_STATUS,
        "source_commit": source_commit,
        "source_hashes": dict(sorted(source_hashes.items())),
        "m1_a_result_sha256": m1_a_result_sha256,
        "m1_a_result_fingerprint": M1_A_RESULT_FINGERPRINT,
        "compiler_identity": COMPILER_IDENTITY,
        "compiler_fingerprint": compiler_fingerprint,
        "random_cells": planned_random,
        "baseline_cells": planned_baseline,
        "ordered_wrapper_cache_keys_sha256": fingerprint(all_keys),
        "resource_caps": {
            "random_cells": RANDOM_CELL_COUNT,
            "trajectories_per_random_cell": TRAJECTORIES_PER_RANDOM_CELL,
            "random_trajectories": RANDOM_TRAJECTORY_COUNT,
            "random_full_wrappers": RANDOM_WRAPPER_COUNT,
            "baseline_cells": BASELINE_CELL_COUNT,
            "baseline_full_wrappers": BASELINE_WRAPPER_COUNT,
            "total_full_wrappers": TOTAL_WRAPPER_COUNT,
            "maximum_process_workers": MAXIMUM_WORKERS,
            "blas_threads_per_worker": 1,
            "extension_trajectories": 0,
        },
        "execution_authorized": False,
        "held_out_access_authorized": False,
        "signal_reevaluation_authorized": False,
        "research_decision_automation_authorized": False,
    }
    body["plan_fingerprint"] = fingerprint(body)
    validate_execution_plan(body)
    return body


def validate_execution_plan(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unexpected M1-B1 execution plan schema")
    if payload.get("status") != PLAN_STATUS or payload.get("execution_authorized") is not False:
        raise ValueError("execution plan improperly authorizes science")
    body = {key: value for key, value in payload.items() if key != "plan_fingerprint"}
    if fingerprint(body) != payload.get("plan_fingerprint"):
        raise ValueError("execution plan fingerprint mismatch")
    random_cells = list(payload.get("random_cells", []))
    baseline_cells = list(payload.get("baseline_cells", []))
    if len(random_cells) != RANDOM_CELL_COUNT or len(baseline_cells) != BASELINE_CELL_COUNT:
        raise ValueError("execution plan cell count mismatch")
    keys = [
        key
        for cell in random_cells + baseline_cells
        for axis in AXES
        for key in cell["wrapper_cache_keys"][axis]
    ]
    if len(keys) != TOTAL_WRAPPER_COUNT or len(keys) != len(set(keys)):
        raise ValueError("execution plan wrapper identities are missing or duplicated")
    if fingerprint(keys) != payload.get("ordered_wrapper_cache_keys_sha256"):
        raise ValueError("execution plan wrapper identity digest mismatch")
    for cell in random_cells:
        expected = expected_trajectory_seeds(cell["request_master_seed"], int(cell["q"]))
        if cell["sampled_trajectory_seeds"] != expected:
            raise ValueError("planned random trajectory seeds do not match benchmark policy")
        if len(cell["wrapper_cache_keys"]["cosine"]) != TRAJECTORIES_PER_RANDOM_CELL:
            raise ValueError("random cell wrapper count mismatch")
    if any(cell["sampled_trajectory_seeds"] is not None for cell in baseline_cells):
        raise ValueError("baseline plan unexpectedly contains random trajectories")


def _compiler() -> CompilerSettings:
    return CompilerSettings(
        basis_gates=tuple(COMPILER_IDENTITY["basis_gates"]),
        backend_name=COMPILER_IDENTITY["backend_name"],
        coupling_map=COMPILER_IDENTITY["coupling_map"],
        optimization_level=COMPILER_IDENTITY["optimization_level"],
        layout_method=COMPILER_IDENTITY["layout_method"],
        routing_method=COMPILER_IDENTITY["routing_method"],
        transpiler_seed=COMPILER_IDENTITY["transpiler_seed"],
        qiskit_version=COMPILER_IDENTITY["qiskit_version"],
    )


@dataclass(frozen=True)
class M1B1CompileJob:
    ordinal: int
    plan_cell: Mapping[str, Any]
    preparation: DFPartialS2Preparation
    cache_path: str


def compile_cell(job: M1B1CompileJob) -> dict[str, Any]:
    """Compile one frozen cell; this is the only science worker entry point."""
    cell = dict(job.plan_cell)
    q = int(cell["q"])
    delta = TOTAL_TIME / q
    random_cell = str(cell["method"]) in {"B2", "B3"}
    if random_cell:
        r = int(cell["r"])
        cutoff = int(cell["K"])
        tau = float(job.preparation.exact_rte_lambda_r) * delta / r
        config, distribution = make_rte_config(
            job.preparation.rte_preparation.symbolic_tail,
            evolution_time=delta,
            rte_steps=r,
            truncation_tolerance=_explicit_cutoff_tolerance(tau, cutoff),
            finite_taylor_order=cutoff,
            seed=int(cell["request_master_seed"]),
        )
        evaluation_method = "monte_carlo"
        sample_count: int | None = TRAJECTORIES_PER_RANDOM_CELL
        request_seed: int | None = int(cell["request_master_seed"])
    else:
        r = 0
        cutoff = 0
        config = None
        distribution = None
        evaluation_method = "exact"
        sample_count = None
        request_seed = None

    cache = TranspiledCircuitCostCache(
        maximum_entries=256,
        persistent_path=job.cache_path,
    )
    request = RPEHadamardCompiledCostBenchmarkRequest(
        preparation=job.preparation,
        delta_time=delta,
        calibration_repetition_counts=(q,),
        holdout_repetition_counts=(),
        rte_steps_per_occurrence=r,
        finite_taylor_order=cutoff,
        rte_config=config,
        rte_distribution=distribution,
        compiler=_compiler(),
        evaluation_method=evaluation_method,
        sample_count=sample_count,
        seed=request_seed,
        generation_id=f"pr2-m1-b1-{cell['candidate_fingerprint']}",
        maximum_repetition_count=8,
        maximum_trajectories=1_000_000,
        maximum_samples=TRAJECTORIES_PER_RANDOM_CELL,
        maximum_untranspiled_circuit_size=10_000_000,
        maximum_retained_trajectory_records=TRAJECTORIES_PER_RANDOM_CELL,
        maximum_build_requests=1_000_000,
        maximum_transpile_requests=1_000_000,
        maximum_planned_instruction_applications=2_000_000_000,
        construction_policy="boundary_optimized",
        cache=cache,
    )
    result = generate_rpe_hadamard_compiled_cost_benchmark_dataset(request)
    if not result.dataset.complete or len(result.dataset.records) != 2:
        raise RuntimeError("M1-B1 cell did not produce two complete axis records")
    axes = {point.axis: point.to_dict() for point in result.dataset.records}
    if set(axes) != set(AXES) or any(point["status"] != "complete" for point in axes.values()):
        raise RuntimeError("M1-B1 cell axis result is incomplete")
    if random_cell:
        cosine_seeds = axes["cosine"]["sampled_trajectory_seeds"]
        sine_seeds = axes["sine"]["sampled_trajectory_seeds"]
        expected = list(cell["sampled_trajectory_seeds"])
        if cosine_seeds != expected or sine_seeds != expected:
            raise RuntimeError("compiled axes do not share the frozen trajectories")
    return {
        "ordinal": job.ordinal,
        "candidate_fingerprint": cell["candidate_fingerprint"],
        "task_fingerprint": cell["task_fingerprint"],
        "method": cell["method"],
        "rank": cell["rank"],
        "q": q,
        "delta": delta,
        "r": int(cell["r"]),
        "K": int(cell["K"]),
        "accuracy_eligible": bool(cell["accuracy_eligible"]),
        "axes": axes,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(payload) + b"\n")
    os.replace(temporary, path)


def _checkpoint(path: Path, job: M1B1CompileJob, result: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": "pr2_matched_accuracy_m1_b1_cell_checkpoint_v1",
        "task_fingerprint": job.plan_cell["task_fingerprint"],
        "candidate_fingerprint": job.plan_cell["candidate_fingerprint"],
        "result": result,
    }
    payload["checkpoint_fingerprint"] = fingerprint(payload)
    _atomic_json(path, payload)


def _read_checkpoint(path: Path, plan_cell: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    body = {key: value for key, value in payload.items() if key != "checkpoint_fingerprint"}
    if fingerprint(body) != payload.get("checkpoint_fingerprint"):
        raise ValueError(f"checkpoint fingerprint mismatch: {path}")
    if payload.get("task_fingerprint") != plan_cell["task_fingerprint"]:
        raise ValueError(f"checkpoint task identity mismatch: {path}")
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("candidate_fingerprint") != plan_cell["candidate_fingerprint"]:
        raise ValueError(f"checkpoint result identity mismatch: {path}")
    return result


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def validate_execution_inputs(
    root: Path,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    plan: Mapping[str, Any],
    plan_sha256: str,
) -> None:
    validate_execution_plan(plan)
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unexpected M1-B1 authorization schema")
    if authorization.get("status") != "M1_B1_EXECUTION_AUTHORIZED_ONCE":
        raise ValueError("M1-B1 science execution is not authorized")
    source_commit = _full_commit(str(authorization.get("source_commit")))
    head = _git_head(root)
    if not _is_ancestor(root, source_commit, head):
        raise ValueError("authorized execution source commit is not an ancestor of HEAD")
    if plan.get("source_commit") != source_commit:
        raise ValueError("execution plan does not use the authorized source commit")
    if authorization.get("execution_plan_sha256") != plan_sha256:
        raise ValueError("execution plan SHA-256 differs from authorization")
    if authorization.get("execution_plan_fingerprint") != plan.get("plan_fingerprint"):
        raise ValueError("execution plan fingerprint differs from authorization")
    if authorization.get("authorization_sha256") not in (None, authorization_sha256):
        raise ValueError("authorization self-audit SHA-256 is inconsistent")
    for relative, expected in dict(authorization.get("source_hashes", {})).items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"authorized source hash mismatch: {relative}")
    if authorization.get("result_terminal_statuses") != [COMPLETE_STATUS, FAILURE_STATUS]:
        raise ValueError("authorization permits an unfrozen terminal status")
    if any(decision in authorization.get("result_terminal_statuses", []) for decision in RESEARCH_DECISIONS):
        raise ValueError("authorization delegates a research decision to the runner")


def _preparations(hamiltonian: Any, cells: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], DFPartialS2Preparation]:
    required = {(str(cell["method"]), int(cell["rank"])) for cell in cells}
    prepared: dict[tuple[str, int], DFPartialS2Preparation] = {}
    for method, rank in sorted(required):
        prepared[(method, rank)] = _prepare_discard(hamiltonian, rank) if method == "B0" else _prepare(hamiltonian, rank)
    return prepared


def _metric_mean(record: Mapping[str, Any], metric: str) -> float:
    return float(record["metric_statistics"][metric]["mean"])


def assemble_compile_map(
    *,
    m1_a: Mapping[str, Any],
    compile_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    signal_by_fp = {
        record["candidate"]["candidate_fingerprint"]: record
        for record in m1_a["signal_records"]
    }
    output: list[dict[str, Any]] = []
    for compiled in compile_records:
        signal = signal_by_fp[compiled["candidate_fingerprint"]]
        axis_shots = signal["axis_shots"]
        eligible = bool(signal["accuracy_eligible"])
        if eligible:
            real_shots = int(axis_shots["real"])
            imag_shots = int(axis_shots["imag"])
            metric_totals: dict[str, float] | None = {
                metric: (
                    real_shots * _metric_mean(compiled["axes"]["cosine"], metric)
                    + imag_shots * _metric_mean(compiled["axes"]["sine"], metric)
                )
                for metric in ("rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size")
            }
            state_preparation_affine: dict[str, Any] | None = {
                "intercept_by_metric": metric_totals,
                "shot_slope": real_shots + imag_shots,
                "formula": "G_metric(P)=intercept_metric+(N_real+N_imag)*P",
            }
        else:
            metric_totals = None
            state_preparation_affine = None
        output.append(
            {
                "candidate": signal["candidate"],
                "signal_record_fingerprint": fingerprint(signal),
                "accuracy_eligible": eligible,
                "fixed_q8": int(compiled["q"]) == 8,
                "axis_shots": axis_shots,
                "compiled_axes": compiled["axes"],
                "matched_accuracy_compiled_work_no_state_preparation": metric_totals,
                "state_preparation_affine": state_preparation_affine,
                "matched_accuracy_frontier_eligible": eligible,
            }
        )
    return output


def validate_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("unexpected M1-B1 result schema")
    if payload.get("status") != COMPLETE_STATUS:
        raise ValueError("completed M1-B1 artifact must await external review")
    if payload.get("research_decision") is not None:
        raise ValueError("science runner must not make the post-B1 research decision")
    counts = payload.get("resource_counts", {})
    expected = {
        "random_cells": RANDOM_CELL_COUNT,
        "trajectories_per_random_cell": TRAJECTORIES_PER_RANDOM_CELL,
        "random_trajectories": RANDOM_TRAJECTORY_COUNT,
        "random_full_wrappers": RANDOM_WRAPPER_COUNT,
        "baseline_cells": BASELINE_CELL_COUNT,
        "baseline_full_wrappers": BASELINE_WRAPPER_COUNT,
        "total_full_wrappers": TOTAL_WRAPPER_COUNT,
        "extension_trajectories": 0,
    }
    if any(counts.get(key) != value for key, value in expected.items()):
        raise ValueError("M1-B1 resource count mismatch")
    if len(payload.get("compile_map", [])) != RANDOM_CELL_COUNT + BASELINE_CELL_COUNT:
        raise ValueError("M1-B1 compile map cell count mismatch")
    if payload.get("held_out_accessed") is not False or payload.get("signal_reevaluations") != 0:
        raise ValueError("M1-B1 crossed a forbidden execution boundary")
    body = {key: value for key, value in payload.items() if key != "result_fingerprint"}
    if fingerprint(body) != payload.get("result_fingerprint"):
        raise ValueError("M1-B1 result fingerprint mismatch")


def run_m1_b1(
    root: Path,
    *,
    authorization_path: Path,
    plan_path: Path,
    output_dir: Path,
    workers: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the authorized compile map once and stop for external review."""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAXIMUM_WORKERS:
        raise ValueError(f"workers must be an integer between 1 and {MAXIMUM_WORKERS}")
    authorization_sha = file_sha256(authorization_path)
    plan_sha = file_sha256(plan_path)
    authorization = load_json(authorization_path)
    plan = load_json(plan_path)
    validate_execution_inputs(root, authorization, authorization_sha, plan, plan_sha)
    m1_path = root / M1_A_RESULT_RELATIVE
    m1_sha = file_sha256(m1_path)
    m1_a = load_json(m1_path)
    validate_m1_a_result(m1_a, sha256=m1_sha)

    run_identity = {
        "schema_version": "pr2_matched_accuracy_m1_b1_run_identity_v1",
        "source_commit": plan["source_commit"],
        "authorization_sha256": authorization_sha,
        "execution_plan_sha256": plan_sha,
        "execution_plan_fingerprint": plan["plan_fingerprint"],
        "workers": workers,
    }
    identity_path = output_dir / ".runtime" / "run_identity.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        if not identity_path.is_file() or load_json(identity_path) != run_identity:
            raise ValueError("resume output does not have the exact frozen run identity")
        if (output_dir / "M1_B1_COMPLETE.json").exists():
            raise ValueError("completed M1-B1 output cannot be resumed")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(identity_path, run_identity)
    checkpoint_dir = output_dir / ".runtime" / "checkpoints"
    cache_dir = output_dir / ".runtime" / "cache" / str(plan["source_commit"])
    checkpoint_dir.mkdir(parents=True, exist_ok=resume)
    cache_dir.mkdir(parents=True, exist_ok=resume)
    counters = {
        "development_raw_hash_checks": 0,
        "development_npz_loads": 0,
        "held_out_path_stats": 0,
        "held_out_hash_reads": 0,
        "held_out_npz_loads": 0,
    }
    started = time.perf_counter()
    hamiltonian, _state, metadata = _load_development_only(root, counters)
    cells = list(plan["random_cells"]) + list(plan["baseline_cells"])
    preparations = _preparations(hamiltonian, cells)
    jobs: list[M1B1CompileJob] = []
    completed: dict[int, dict[str, Any]] = {}
    for ordinal, cell in enumerate(cells):
        checkpoint_path = checkpoint_dir / f"{ordinal:03d}_{cell['candidate_fingerprint']}.json"
        cached = _read_checkpoint(checkpoint_path, cell)
        if cached is not None:
            completed[ordinal] = cached
            continue
        jobs.append(
            M1B1CompileJob(
                ordinal=ordinal,
                plan_cell=cell,
                preparation=preparations[(str(cell["method"]), int(cell["rank"]))],
                cache_path=str(cache_dir / f"{cell['candidate_fingerprint']}.sqlite3"),
            )
        )
    if workers == 1:
        for job in jobs:
            result = compile_cell(job)
            completed[job.ordinal] = result
            _checkpoint(checkpoint_dir / f"{job.ordinal:03d}_{job.plan_cell['candidate_fingerprint']}.json", job, result)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context(PROCESS_START_METHOD)) as executor:
            futures = {executor.submit(compile_cell, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                completed[job.ordinal] = result
                _checkpoint(checkpoint_dir / f"{job.ordinal:03d}_{job.plan_cell['candidate_fingerprint']}.json", job, result)
    ordered = [completed[index] for index in range(len(cells))]
    compile_map = assemble_compile_map(m1_a=m1_a, compile_records=ordered)
    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "series_id": SERIES_ID,
        "status": COMPLETE_STATUS,
        "source_commit": plan["source_commit"],
        "authorization_sha256": authorization_sha,
        "execution_plan_sha256": plan_sha,
        "execution_plan_fingerprint": plan["plan_fingerprint"],
        "m1_a_result_sha256": M1_A_RESULT_SHA256,
        "m1_a_result_fingerprint": M1_A_RESULT_FINGERPRINT,
        "development_snapshot": {
            "path": DEVELOPMENT_RELATIVE_PATH,
            "hamiltonian_hash": metadata["hamiltonian_hash"],
        },
        "compile_map": compile_map,
        "resource_counts": {
            "random_cells": RANDOM_CELL_COUNT,
            "trajectories_per_random_cell": TRAJECTORIES_PER_RANDOM_CELL,
            "random_trajectories": RANDOM_TRAJECTORY_COUNT,
            "random_full_wrappers": RANDOM_WRAPPER_COUNT,
            "baseline_cells": BASELINE_CELL_COUNT,
            "baseline_full_wrappers": BASELINE_WRAPPER_COUNT,
            "total_full_wrappers": TOTAL_WRAPPER_COUNT,
            "extension_trajectories": 0,
            "process_workers": workers,
            "blas_threads_per_worker": 1,
        },
        "signal_reevaluations": 0,
        "held_out_accessed": False,
        "additional_96_trajectories_executed": False,
        "transfer_executed": False,
        "s3_authorized": False,
        "research_decision": None,
        "allowed_later_external_review_decisions": list(RESEARCH_DECISIONS),
        "automatic_next_stage": None,
        "execution": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_s": time.perf_counter() - started,
            "worker_count": workers,
            "process_start_method": PROCESS_START_METHOD,
            "gpu_queries": 0,
            "gpu_allocations": 0,
            "gpu_kernels": 0,
        },
    }
    payload["result_fingerprint"] = fingerprint(payload)
    validate_result(payload)
    _atomic_json(output_dir / "pr2_matched_accuracy_m1_b1_compile_map_result_v2.json", payload)
    _atomic_json(
        output_dir / "M1_B1_COMPLETE.json",
        {"status": COMPLETE_STATUS, "result_fingerprint": payload["result_fingerprint"]},
    )
    return payload
