"""Parallel execution layer for the frozen PR-2 S2 development comparison.

This module changes execution scheduling only.  Every scientific cell is still
evaluated by :mod:`pr2_v4_s2_development_validation`, with the same seeds,
trajectory counts, compiler settings, stage barriers, pooling, and decision
rules.  The held-out snapshot remains unopened.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import pr2_v4_s2_development_validation as validation
from .df_partial_s2 import DFPartialS2Preparation
from .rte_compiled_cost import TranspiledCircuitCostCache


DEFAULT_PARALLEL_WORKERS = 4
MAXIMUM_PARALLEL_WORKERS = 8
PROCESS_START_METHOD = "spawn"


@dataclass(frozen=True)
class S2CompileJob:
    """One immutable PR-2 compiled-cost cell."""

    ordinal: int
    preparation: DFPartialS2Preparation
    stage: str
    stream: str
    method: str
    rank: int
    q: int
    rte_steps: int
    cutoff: int
    sample_count: int | None
    master_seed: int

    def keyword_arguments(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "stream": self.stream,
            "method": self.method,
            "rank": self.rank,
            "q": self.q,
            "rte_steps": self.rte_steps,
            "cutoff": self.cutoff,
            "sample_count": self.sample_count,
            "master_seed": self.master_seed,
        }


_WORKER_CACHE: TranspiledCircuitCostCache | None = None


def _validate_worker_count(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer.")
    if workers < 1 or workers > MAXIMUM_PARALLEL_WORKERS:
        raise ValueError(
            f"workers must be between 1 and {MAXIMUM_PARALLEL_WORKERS}."
        )
    return workers


def _initialize_compile_worker(
    maximum_cache_entries: int,
    persistent_cache_path: str | None,
) -> None:
    global _WORKER_CACHE
    _WORKER_CACHE = TranspiledCircuitCostCache(
        maximum_entries=maximum_cache_entries,
        persistent_path=persistent_cache_path,
    )


def _execute_compile_job(job: S2CompileJob) -> tuple[int, dict[str, Any]]:
    if _WORKER_CACHE is None:
        raise RuntimeError("Parallel compile worker was not initialized.")
    return (
        job.ordinal,
        validation._compile_cost_batch(
            job.preparation,
            cache=_WORKER_CACHE,
            **job.keyword_arguments(),
        ),
    )


class S2CompileDispatcher:
    """Execute compile cells serially or in a bounded spawned process pool."""

    def __init__(
        self,
        *,
        workers: int,
        persistent_cache_path: str | Path | None,
        maximum_cache_entries: int = 8192,
    ) -> None:
        self.workers = _validate_worker_count(workers)
        if maximum_cache_entries < 1:
            raise ValueError("maximum_cache_entries must be positive.")
        self.maximum_cache_entries = int(maximum_cache_entries)
        self.persistent_cache_path = (
            None
            if persistent_cache_path is None
            else Path(persistent_cache_path).resolve()
        )
        self._serial_cache: TranspiledCircuitCostCache | None = None
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> "S2CompileDispatcher":
        cache_path = (
            None
            if self.persistent_cache_path is None
            else str(self.persistent_cache_path)
        )
        if self.workers == 1:
            self._serial_cache = TranspiledCircuitCostCache(
                maximum_entries=self.maximum_cache_entries,
                persistent_path=cache_path,
            )
        else:
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=get_context(PROCESS_START_METHOD),
                initializer=_initialize_compile_worker,
                initargs=(self.maximum_cache_entries, cache_path),
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=exc is not None)
        self._executor = None
        self._serial_cache = None

    def compile(self, jobs: Sequence[S2CompileJob]) -> list[dict[str, Any]]:
        """Compile jobs and return results in the caller's canonical order."""
        normalized = list(jobs)
        ordinals = [job.ordinal for job in normalized]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("Compile-job ordinals must be unique within a phase.")
        if not normalized:
            return []
        if self.workers == 1:
            if self._serial_cache is None:
                raise RuntimeError("Compile dispatcher is not active.")
            observed = [
                (
                    job.ordinal,
                    validation._compile_cost_batch(
                        job.preparation,
                        cache=self._serial_cache,
                        **job.keyword_arguments(),
                    ),
                )
                for job in normalized
            ]
        else:
            if self._executor is None:
                raise RuntimeError("Compile dispatcher is not active.")
            observed = list(
                self._executor.map(_execute_compile_job, normalized, chunksize=1)
            )
        if [ordinal for ordinal, _result in observed] != ordinals:
            raise RuntimeError("Parallel compile results changed canonical order.")
        return [result for _ordinal, result in observed]


def _job(
    ordinal: int,
    preparation: DFPartialS2Preparation,
    *,
    stage: str,
    stream: str,
    method: str,
    rank: int,
    rte_steps: int,
    cutoff: int,
    sample_count: int | None,
    master_seed: int,
) -> S2CompileJob:
    return S2CompileJob(
        ordinal=ordinal,
        preparation=preparation,
        stage=stage,
        stream=stream,
        method=method,
        rank=rank,
        q=validation.S2_Q,
        rte_steps=rte_steps,
        cutoff=cutoff,
        sample_count=sample_count,
        master_seed=master_seed,
    )


def run_s2_development_parallel(
    root: Path,
    v4_payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    workers: int = DEFAULT_PARALLEL_WORKERS,
    persistent_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run frozen S2 with parallel compile cells and unchanged stage barriers."""
    worker_count = _validate_worker_count(workers)
    validation.validate_v4_payload(v4_payload)
    if (
        v4_payload["status"] != validation.V4_PASS_STATUS
        or v4_payload["deviations"]
    ):
        raise RuntimeError("S2 requires a deviation-free V4 PASS artifact.")

    counters = validation._new_counters()
    hamiltonian, state, metadata = validation._load_inputs(root, counters)
    energy, exact_target = validation._target_signal(
        hamiltonian,
        state,
        validation.S2_Q * validation.DELTA_TIME,
    )

    deterministic: list[dict[str, Any]] = []
    random_candidates: dict[str, list[dict[str, Any]]] = {"B2": [], "B3": []}
    controls: list[dict[str, Any]] = []
    expanded: list[str] = []

    effective_provenance = dict(provenance)
    effective_provenance["parallel_execution"] = {
        "workers": worker_count,
        "process_start_method": PROCESS_START_METHOD,
        "phase_barriers": [
            "deterministic",
            "initial32",
            "extension96",
            "rank3_rank9_controls",
        ],
        "canonical_result_order": True,
        "persistent_compile_cache_enabled": persistent_cache_path is not None,
        "scientific_parameters_changed": False,
    }

    with S2CompileDispatcher(
        workers=worker_count,
        persistent_cache_path=persistent_cache_path,
    ) as dispatcher:
        deterministic_rows: list[
            tuple[str, int, DFPartialS2Preparation, dict[str, Any]]
        ] = []
        deterministic_jobs: list[S2CompileJob] = []
        for ordinal, (method, rank) in enumerate(
            (("B0", validation.PRIMARY_RANK), ("B1", 12))
        ):
            preparation = validation._prepare_deterministic(hamiltonian, rank)
            signal = validation._deterministic_signal_point(
                preparation,
                state,
                exact_target,
                method=method,
                rank=rank,
                q=validation.S2_Q,
            )
            deterministic_rows.append((method, rank, preparation, signal))
            deterministic_jobs.append(
                _job(
                    ordinal,
                    preparation,
                    stage="S2",
                    stream="deterministic",
                    method=method,
                    rank=rank,
                    rte_steps=0,
                    cutoff=0,
                    sample_count=None,
                    master_seed=0,
                )
            )
        deterministic_batches = dispatcher.compile(deterministic_jobs)
        for (method, rank, _preparation, signal), batch in zip(
            deterministic_rows,
            deterministic_batches,
            strict=True,
        ):
            candidate = {
                "method": method,
                "rank": rank,
                "r": 0,
                "K": 0,
                "signal": signal,
                "cost_batches": [batch],
            }
            validation._refresh_candidate(candidate)
            deterministic.append(candidate)

        random_preparations = {
            "B2": validation._prepare_random(
                hamiltonian,
                "B2",
                validation.PRIMARY_RANK,
            ),
            "B3": validation._prepare_random(hamiltonian, "B3", 0),
        }
        initial_candidates: list[dict[str, Any]] = []
        initial_jobs: list[S2CompileJob] = []
        ordinal = 0
        for method, preparation in random_preparations.items():
            rank = validation.PRIMARY_RANK if method == "B2" else 0
            for rte_steps in validation.S1_R_VALUES:
                for cutoff in validation.S1_K_VALUES:
                    signal = validation._random_signal_point(
                        preparation,
                        state,
                        exact_target,
                        stage="S2",
                        method=method,
                        rank=rank,
                        q=validation.S2_Q,
                        rte_steps=rte_steps,
                        cutoff=cutoff,
                    )
                    candidate = {
                        "method": method,
                        "rank": rank,
                        "r": rte_steps,
                        "K": cutoff,
                        "signal": signal,
                        "cost_batches": [],
                    }
                    initial_candidates.append(candidate)
                    initial_jobs.append(
                        _job(
                            ordinal,
                            preparation,
                            stage="S2",
                            stream="initial32",
                            method=method,
                            rank=rank,
                            rte_steps=rte_steps,
                            cutoff=cutoff,
                            sample_count=validation.S2_INITIAL_SAMPLE_COUNT,
                            master_seed=validation.S2_MASTER_SEED,
                        )
                    )
                    ordinal += 1
        initial_batches = dispatcher.compile(initial_jobs)
        for candidate, batch in zip(
            initial_candidates,
            initial_batches,
            strict=True,
        ):
            candidate["cost_batches"].append(batch)
            validation._refresh_candidate(candidate)
            random_candidates[str(candidate["method"])].append(candidate)

        expansion_keys, initial_diagnostics = validation._initial_expansion_keys(
            random_candidates["B2"],
            random_candidates["B3"],
            deterministic,
        )
        expansion_candidates: list[dict[str, Any]] = []
        expansion_jobs: list[S2CompileJob] = []
        ordinal = 0
        for method in ("B2", "B3"):
            preparation = random_preparations[method]
            for candidate in random_candidates[method]:
                key = validation._candidate_key(candidate)
                if key not in expansion_keys:
                    continue
                expansion_candidates.append(candidate)
                expansion_jobs.append(
                    _job(
                        ordinal,
                        preparation,
                        stage="S2",
                        stream="extension96",
                        method=method,
                        rank=int(candidate["rank"]),
                        rte_steps=int(candidate["r"]),
                        cutoff=int(candidate["K"]),
                        sample_count=validation.S2_EXTENSION_SAMPLE_COUNT,
                        master_seed=validation.S2_EXTENSION_MASTER_SEED,
                    )
                )
                ordinal += 1
        extension_batches = dispatcher.compile(expansion_jobs)
        for candidate, extension in zip(
            expansion_candidates,
            extension_batches,
            strict=True,
        ):
            candidate["cost_batches"].append(extension)
            validation._refresh_candidate(candidate)
            if (
                candidate["pooled_cost"]["sample_count"]
                != validation.S2_TOTAL_SAMPLE_COUNT
            ):
                raise RuntimeError("Expanded cost cell did not pool to 128 samples.")
            expanded.append(validation._candidate_key(candidate))

        decision = validation._decision(
            random_candidates["B2"],
            random_candidates["B3"],
            deterministic,
        )
        selected_b2 = validation._selected_candidate(random_candidates["B2"])
        if selected_b2 is not None:
            selected_r = int(selected_b2["r"])
            selected_k = int(selected_b2["K"])
            control_candidates: list[dict[str, Any]] = []
            control_jobs: list[S2CompileJob] = []
            for ordinal, rank in enumerate(validation.CONTROL_RANKS):
                preparation = validation._prepare_random(hamiltonian, "B2", rank)
                signal = validation._random_signal_point(
                    preparation,
                    state,
                    exact_target,
                    stage="S2-control",
                    method="B2",
                    rank=rank,
                    q=validation.S2_Q,
                    rte_steps=selected_r,
                    cutoff=selected_k,
                )
                control_candidates.append(
                    {
                        "method": "B2",
                        "rank": rank,
                        "r": selected_r,
                        "K": selected_k,
                        "signal": signal,
                        "cost_batches": [],
                    }
                )
                control_jobs.append(
                    _job(
                        ordinal,
                        preparation,
                        stage="S2-control",
                        stream="fixed32",
                        method="B2",
                        rank=rank,
                        rte_steps=selected_r,
                        cutoff=selected_k,
                        sample_count=validation.S2_INITIAL_SAMPLE_COUNT,
                        master_seed=validation.S2_MASTER_SEED,
                    )
                )
            control_batches = dispatcher.compile(control_jobs)
            for candidate, batch in zip(
                control_candidates,
                control_batches,
                strict=True,
            ):
                candidate["cost_batches"].append(batch)
                validation._refresh_candidate(candidate)
                controls.append(candidate)

    all_random = [
        *random_candidates["B2"],
        *random_candidates["B3"],
        *controls,
    ]
    counters["signal_evaluations"] = len(deterministic) + len(all_random)
    counters["random_trajectories_compiled"] = sum(
        int(batch["sample_count"])
        for item in all_random
        for batch in item["cost_batches"]
    )
    counters["full_wrappers_compiled"] = (
        2 * counters["random_trajectories_compiled"] + 2 * len(deterministic)
    )
    payload: dict[str, Any] = {
        "schema_version": validation.S2_SCHEMA_VERSION,
        "series_id": validation.SERIES_ID,
        "status": decision["status"],
        "authorization_commit": validation.AUTHORIZATION_COMMIT,
        "authorization_sha256": validation.AUTHORIZATION_SHA256,
        "specification_sha256": validation.SPECIFICATION_SHA256,
        "V4_result_fingerprint": v4_payload["result_fingerprint"],
        "development_snapshot": {
            "path": validation.DEVELOPMENT_RELATIVE_PATH,
            "raw_file_sha256": validation.EXPECTED_DEVELOPMENT_FILE_SHA256,
            "hamiltonian_hash": metadata["hamiltonian_hash"],
        },
        "held_out": {
            "path": validation.HELD_OUT_RELATIVE_PATH,
            "raw_file_sha256": validation.EXPECTED_HELD_OUT_FILE_SHA256,
            "npz_loaded": False,
            "signal_cost_ranking_evaluated": False,
        },
        "task": {
            "T": validation.S2_Q * validation.DELTA_TIME,
            "delta": validation.DELTA_TIME,
            "q": validation.S2_Q,
            "energy_hartree": energy,
            "exact_target": validation._complex_record(exact_target),
            "complex_signal_error": 0.05,
            "axis_error": validation.AXIS_ERROR,
            "axis_alpha": validation.AXIS_ALPHA,
        },
        "deterministic_candidates": deterministic,
        "B2_candidates": random_candidates["B2"],
        "B3_candidates": random_candidates["B3"],
        "rank3_rank9_controls": controls,
        "sampling": {
            "initial_sample_count": validation.S2_INITIAL_SAMPLE_COUNT,
            "extension_sample_count": validation.S2_EXTENSION_SAMPLE_COUNT,
            "maximum_total_sample_count": validation.S2_TOTAL_SAMPLE_COUNT,
            "initial_master_seed": validation.S2_MASTER_SEED,
            "extension_master_seed": validation.S2_EXTENSION_MASTER_SEED,
            "rz_relative_standard_error_trigger": validation.RZ_RSE_TRIGGER,
            "expansion_keys": sorted(expansion_keys),
            "expanded_keys": sorted(expanded),
            "initial_diagnostics": initial_diagnostics,
            "additional_sampling_authorized": False,
        },
        "decision": decision,
        "counters": counters,
        "provenance": effective_provenance,
        "deviations": [],
        "state_preparation_included_in_primary": False,
        "quantum_shots_executed": 0,
        "held_out_npz_loaded": False,
        "held_out_signal_cost_ranking_evaluated": False,
        "S3_authorized": False,
        "automatic_next_stage": None,
        "mandatory_stop_reached": True,
    }
    payload["result_fingerprint"] = validation._fingerprint(payload)
    validation.validate_s2_payload(payload)
    return payload
