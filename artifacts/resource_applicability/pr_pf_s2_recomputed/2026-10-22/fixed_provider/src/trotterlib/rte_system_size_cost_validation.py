"""System-size transfer validation for paired connected compiled-cost models."""

from __future__ import annotations

import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian
from .df_partial_randomized_pf import df_hamiltonian_hash, split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
from .rte import CompilerSettings, require_integer_count
from .rte_order_stratified_cost_validation import (
    _METRICS,
    _aggregate_paired_condition,
    _canonical_json,
    _compile_paired_length_strata,
    _compiler_payload,
    _fingerprint,
    _parse_pattern,
    _pattern_key,
)


SYSTEM_SIZE_PAIRED_SCHEMA_VERSION = "rte_system_size_paired_cluster_v1"
SYSTEM_SIZE_PAIRED_METHOD = "paired_k1_k3_k1_k4_l4_l6_l8_v1"


def _checkpoint_identity(
    *,
    preparation_hash: str,
    partition_hash: str,
    compiler: CompilerSettings,
    ld: int,
    short_step_time: float,
    sequence_length: int,
    maximum_cluster_length: int,
    order_pattern: Sequence[int],
    sample_count: int,
    seed: int,
    maximum_circuit_size: int,
) -> dict[str, Any]:
    return {
        "method": SYSTEM_SIZE_PAIRED_METHOD,
        "preparation_hash": preparation_hash,
        "partition_hash": partition_hash,
        "compiler": _compiler_payload(compiler),
        "ld": ld,
        "short_step_time": short_step_time,
        "sequence_length": sequence_length,
        "maximum_cluster_length": maximum_cluster_length,
        "order_pattern": list(order_pattern),
        "sample_count": sample_count,
        "seed": seed,
        "maximum_circuit_size": maximum_circuit_size,
    }


def _checkpoint_path(
    directory: Path,
    *,
    maximum_cluster_length: int,
    sequence_length: int,
    pattern: Sequence[int],
) -> Path:
    key = _pattern_key(pattern).replace(",", "-")
    return directory / f"k{maximum_cluster_length}_L{sequence_length}_p{key}.json"


def _load_checkpoint(
    path: Path, *, expected_fingerprint: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if envelope.get("schema_version") != SYSTEM_SIZE_PAIRED_SCHEMA_VERSION:
        raise ValueError(f"Unsupported system-size checkpoint: {path}")
    if envelope.get("task_fingerprint") != expected_fingerprint:
        raise ValueError(f"System-size checkpoint fingerprint mismatch: {path}")
    result = envelope.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(f"System-size checkpoint result is missing: {path}")
    return dict(result)


def _write_checkpoint(
    path: Path, *, task_fingerprint: str, result: Mapping[str, Any]
) -> None:
    envelope = {
        "schema_version": SYSTEM_SIZE_PAIRED_SCHEMA_VERSION,
        "task_fingerprint": task_fingerprint,
        "result": result,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            envelope,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_stage(
    tasks: Sequence[Mapping[str, Any]],
    *,
    checkpoint_directory: Path,
    preparation_hash: str,
    partition_hash: str,
    maximum_workers: int,
) -> tuple[list[dict[str, Any]], int]:
    outputs: list[dict[str, Any]] = []
    pending: list[tuple[Mapping[str, Any], Path, str]] = []
    for task in tasks:
        pattern = tuple(task["patterns"][0])
        sample_count = (
            int(task["common_sample_count"])
            if 2 not in pattern
            else int(task["single_rare_sample_count"])
        )
        identity = _checkpoint_identity(
            preparation_hash=preparation_hash,
            partition_hash=partition_hash,
            compiler=task["compiler"],
            ld=int(task["ld"]),
            short_step_time=float(task["short_step_time"]),
            sequence_length=int(task["length"]),
            maximum_cluster_length=int(task["maximum_cluster_length"]),
            order_pattern=pattern,
            sample_count=sample_count,
            seed=int(task["master_seed"]),
            maximum_circuit_size=int(task["maximum_circuit_size"]),
        )
        fingerprint = _fingerprint(identity)
        path = _checkpoint_path(
            checkpoint_directory,
            maximum_cluster_length=int(task["maximum_cluster_length"]),
            sequence_length=int(task["length"]),
            pattern=pattern,
        )
        cached = _load_checkpoint(path, expected_fingerprint=fingerprint)
        if cached is None:
            pending.append((task, path, fingerprint))
        else:
            outputs.append(cached)

    if maximum_workers == 1:
        for task, path, fingerprint in pending:
            result = _compile_paired_length_strata(task)
            _write_checkpoint(path, task_fingerprint=fingerprint, result=result)
            outputs.append(result)
    elif pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(maximum_workers, len(pending)), mp_context=context
        ) as executor:
            futures = {
                executor.submit(_compile_paired_length_strata, task): (
                    path,
                    fingerprint,
                )
                for task, path, fingerprint in pending
            }
            for future in as_completed(futures):
                path, fingerprint = futures[future]
                result = future.result()
                _write_checkpoint(path, task_fingerprint=fingerprint, result=result)
                outputs.append(result)
    return outputs, len(tasks) - len(pending)


def _maximum(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else float(max(present))


def validate_system_size_paired_cluster_models(
    hamiltonian: DFHamiltonian,
    *,
    ld: int,
    reference_delta_time: float,
    reference_rte_steps: int,
    compiler: CompilerSettings,
    common_sample_count: int,
    single_rare_sample_count: int,
    seed: int,
    sequence_lengths: Sequence[int] = (4, 6, 8),
    cluster_lengths: Sequence[int] = (3, 4),
    maximum_workers: int = 3,
    cache_maximum_entries: int = 32_768,
    persistent_cache_path: str | Path | None = None,
    checkpoint_directory: str | Path | None = None,
    maximum_circuit_size: int = 300_000,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare paired K1--K3 and K1--K4 predictions on unused full circuits."""
    started = time.perf_counter()
    ld = require_integer_count(ld, name="ld")
    reference_rte_steps = require_integer_count(
        reference_rte_steps, name="reference_rte_steps", minimum=1
    )
    common_sample_count = require_integer_count(
        common_sample_count, name="common_sample_count", minimum=2
    )
    single_rare_sample_count = require_integer_count(
        single_rare_sample_count, name="single_rare_sample_count", minimum=2
    )
    maximum_workers = require_integer_count(
        maximum_workers, name="maximum_workers", minimum=1
    )
    lengths = tuple(
        require_integer_count(value, name="sequence_length", minimum=4)
        for value in sequence_lengths
    )
    clusters = tuple(
        require_integer_count(value, name="cluster_length", minimum=1)
        for value in cluster_lengths
    )
    if not lengths or len(lengths) != len(set(lengths)):
        raise ValueError("sequence_lengths must be non-empty and unique.")
    if not clusters or len(clusters) != len(set(clusters)):
        raise ValueError("cluster_lengths must be non-empty and unique.")
    if max(clusters) > min(lengths):
        raise ValueError("A cluster length exceeds the shortest sequence.")
    if not math.isfinite(reference_delta_time) or reference_delta_time <= 0.0:
        raise ValueError("reference_delta_time must be finite and positive.")

    partition = split_df_hamiltonian_by_ld(hamiltonian, ld)
    preparation = prepare_df_partial_s2(
        hamiltonian, partition, identity_policy="extract_identity_phase"
    )
    if preparation.is_deterministic_only:
        raise ValueError("System-size validation requires a non-empty RTE tail.")
    short_step_time = float(reference_delta_time) / reference_rte_steps
    checkpoint_root = Path(
        checkpoint_directory or "artifacts/rte_system_size_cost_validation/checkpoints"
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    raw_by_model: dict[str, Any] = {}
    results_by_model: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    performance: dict[str, Any] = {}
    reused_total = 0
    for maximum_cluster_length in clusters:
        tasks = []
        for length in lengths:
            patterns = [(0,) * length]
            patterns.extend(
                tuple(2 if position == rare else 0 for position in range(length))
                for rare in range(length)
            )
            for pattern in patterns:
                tasks.append(
                    {
                        "hamiltonian": hamiltonian,
                        "ld": ld,
                        "short_step_time": short_step_time,
                        "compiler": compiler,
                        "length": length,
                        "master_seed": seed,
                        "common_sample_count": common_sample_count,
                        "single_rare_sample_count": single_rare_sample_count,
                        "multi_rare_sample_count": None,
                        "maximum_cluster_length": maximum_cluster_length,
                        "cache_maximum_entries": cache_maximum_entries,
                        "persistent_cache_path": (
                            None
                            if persistent_cache_path is None
                            else str(persistent_cache_path)
                        ),
                        "maximum_circuit_size": maximum_circuit_size,
                        "patterns": (pattern,),
                    }
                )
        outputs, reused = _run_stage(
            tasks,
            checkpoint_directory=checkpoint_root,
            preparation_hash=preparation.preparation_hash,
            partition_hash=preparation.partition_hash,
            maximum_workers=maximum_workers,
        )
        reused_total += reused
        strata_by_length: dict[str, dict[str, Any]] = {
            str(length): {} for length in lengths
        }
        for output in outputs:
            if output["preparation_hash"] != preparation.preparation_hash:
                raise RuntimeError("System-size worker preparation differs.")
            if output["partition_hash"] != preparation.partition_hash:
                raise RuntimeError("System-size worker partition differs.")
            raw_length = str(output["sequence_length"])
            overlap = set(strata_by_length[raw_length]).intersection(output["strata"])
            if overlap:
                raise RuntimeError("System-size worker strata overlap.")
            strata_by_length[raw_length].update(output["strata"])
        model_results: dict[str, Any] = {}
        primary = []
        all_metrics = []
        for length in lengths:
            raw_length = str(length)
            strata = strata_by_length[raw_length]
            if len(strata) != length + 1:
                raise RuntimeError("System-size worker strata are incomplete.")
            zero = tuple(
                key for key, value in strata.items() if value["rare_order_count"] == 0
            )
            one = tuple(
                key for key, value in strata.items() if value["rare_order_count"] == 1
            )
            model_results[raw_length] = {
                "zero_order2_condition": _aggregate_paired_condition(strata, zero),
                "exactly_one_order2_condition": _aggregate_paired_condition(strata, one),
            }
            for scope in model_results[raw_length].values():
                primary.append(scope["metrics"]["rz_count"])
                all_metrics.extend(scope["metrics"].values())
        model_key = f"k1_k{maximum_cluster_length}"
        raw_by_model[model_key] = strata_by_length
        results_by_model[model_key] = model_results
        summaries[model_key] = {
            "primary_maximum_absolute_relative_error": _maximum(
                [value["absolute_relative_error"] for value in primary]
            ),
            "primary_maximum_absolute_z_score": _maximum(
                [value["absolute_z_score"] for value in primary]
            ),
            "primary_maximum_pointwise_normal_95_upper_diagnostic": _maximum(
                [
                    value["pointwise_normal_relative_95_upper_diagnostic"]
                    for value in primary
                ]
            ),
            "all_metrics_maximum_absolute_relative_error": _maximum(
                [value["absolute_relative_error"] for value in all_metrics]
            ),
            "primary_point_tolerance_passed": all(
                value["absolute_relative_error"] is not None
                and value["absolute_relative_error"] <= 0.05
                for value in primary
            ),
        }
        performance[model_key] = {
            "task_count": len(tasks),
            "reused_checkpoint_count": reused,
            "worker_seconds_sum": float(
                math.fsum(output["performance"]["total_seconds"] for output in outputs)
            ),
        }

    selected = None
    for maximum_cluster_length in sorted(clusters):
        key = f"k1_k{maximum_cluster_length}"
        if summaries[key]["primary_point_tolerance_passed"]:
            selected = maximum_cluster_length
            break
    payload: dict[str, Any] = {
        "schema_version": SYSTEM_SIZE_PAIRED_SCHEMA_VERSION,
        "validation_method": SYSTEM_SIZE_PAIRED_METHOD,
        "final_cost_evaluation_performed": False,
        "acceptance_policy": {
            "primary_metric": "rz_count",
            "relative_point_tolerance": 0.05,
            "normal_approximation_is_diagnostic_only": True,
            "selection_rule": "choose_smallest_tested_cluster_length_passing_all_scopes",
        },
        "hamiltonian": {
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": hamiltonian.n_blocks,
            "hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
            "metadata": dict(hamiltonian.metadata),
            "preparation_hash": preparation.preparation_hash,
            "partition_hash": preparation.partition_hash,
        },
        "configuration": {
            "ld": ld,
            "reference_delta_time": float(reference_delta_time),
            "reference_rte_steps": reference_rte_steps,
            "short_step_time": short_step_time,
            "finite_taylor_order": 2,
            "sequence_lengths": list(lengths),
            "cluster_lengths": list(clusters),
            "common_sample_count_per_length": common_sample_count,
            "single_rare_sample_count_per_position": single_rare_sample_count,
            "seed": seed,
            "compiler": _compiler_payload(compiler),
            "checkpoint_directory": str(checkpoint_root),
            "persistent_cache_path": (
                None if persistent_cache_path is None else str(persistent_cache_path)
            ),
        },
        "paired_strata_by_model": raw_by_model,
        "paired_residual_results_by_model": results_by_model,
        "model_summaries": summaries,
        "decision": {
            "selected_maximum_cluster_length": selected,
            "point_tolerance_passed": selected is not None,
        },
        "performance": {
            "total_seconds": float(time.perf_counter() - started),
            "maximum_workers": maximum_workers,
            "total_reused_checkpoint_count": reused_total,
            "by_model": performance,
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    validate_system_size_paired_payload(payload)
    return payload


def validate_system_size_paired_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SYSTEM_SIZE_PAIRED_SCHEMA_VERSION:
        raise ValueError("Unsupported system-size paired schema.")
    if payload.get("validation_method") != SYSTEM_SIZE_PAIRED_METHOD:
        raise ValueError("Unsupported system-size paired method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("This schema cannot contain a final cost evaluation.")
    configuration = payload.get("configuration", {})
    lengths = tuple(int(value) for value in configuration.get("sequence_lengths", ()))
    clusters = tuple(int(value) for value in configuration.get("cluster_lengths", ()))
    models = payload.get("paired_strata_by_model", {})
    results = payload.get("paired_residual_results_by_model", {})
    summaries = payload.get("model_summaries", {})
    expected_models = {f"k1_k{value}" for value in clusters}
    if set(models) != expected_models or set(results) != expected_models:
        raise ValueError("System-size paired models are incomplete.")
    if set(summaries) != expected_models:
        raise ValueError("System-size paired summaries are incomplete.")
    for model in expected_models:
        if set(models[model]) != {str(value) for value in lengths}:
            raise ValueError("System-size paired lengths are incomplete.")
        for length in lengths:
            raw_length = str(length)
            strata = models[model][raw_length]
            if len(strata) != length + 1:
                raise ValueError("System-size paired strata are incomplete.")
            if set(results[model][raw_length]) != {
                "zero_order2_condition",
                "exactly_one_order2_condition",
            }:
                raise ValueError("System-size paired scopes are incomplete.")
    if {3, 4}.issubset(set(clusters)):
        for length in lengths:
            raw_length = str(length)
            k3 = models["k1_k3"][raw_length]
            k4 = models["k1_k4"][raw_length]
            if set(k3) != set(k4):
                raise ValueError("K3 and K4 patterns differ.")
            for key in k3:
                if _parse_pattern(key) != tuple(k3[key]["order_pattern"]):
                    raise ValueError("System-size pattern key differs.")
                if (
                    k3[key]["event_stream_rolling_digest"]
                    != k4[key]["event_stream_rolling_digest"]
                ):
                    raise ValueError("K3 and K4 paired trajectories differ.")
                if k3[key]["actual_statistics"] != k4[key]["actual_statistics"]:
                    raise ValueError("K3 and K4 full-circuit statistics differ.")
    fingerprint = payload.get("validation_fingerprint")
    without = dict(payload)
    without.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without):
        raise ValueError("System-size paired fingerprint mismatch.")


def write_system_size_paired_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_system_size_paired_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        _canonical_json(payload) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
