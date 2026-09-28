from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import trotterlib.pr2_v4_s2_development_validation as validation
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.pr2_v4_s2_parallel_execution import (
    MAXIMUM_PARALLEL_WORKERS,
    S2CompileDispatcher,
    S2CompileJob,
    run_s2_development_parallel,
)


def _toy_hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.13,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.4, -0.3]),
        g_matrices=(
            np.asarray([[1.0]], dtype=np.complex128),
            np.asarray([[0.7]], dtype=np.complex128),
        ),
        metadata={"name": "pr2-v4-s2-parallel-test"},
    )


def _jobs() -> list[S2CompileJob]:
    preparation = validation._prepare_random(_toy_hamiltonian(), "B2", 1)
    return [
        S2CompileJob(
            ordinal=ordinal,
            preparation=preparation,
            stage="S2-parallel-test",
            stream="initial2",
            method="B2",
            rank=1,
            q=1,
            rte_steps=rte_steps,
            cutoff=2,
            sample_count=2,
            master_seed=20260928,
        )
        for ordinal, rte_steps in enumerate((1, 2))
    ]


def test_parallel_compile_cells_match_serial_and_preserve_order(
    tmp_path: Path,
) -> None:
    jobs = _jobs()
    with S2CompileDispatcher(
        workers=1,
        persistent_cache_path=None,
        maximum_cache_entries=64,
    ) as serial_dispatcher:
        serial = serial_dispatcher.compile(jobs)
    with S2CompileDispatcher(
        workers=2,
        persistent_cache_path=tmp_path / "parallel-cache.sqlite",
        maximum_cache_entries=64,
    ) as parallel_dispatcher:
        parallel = parallel_dispatcher.compile(jobs)

    assert parallel == serial
    assert [(item["r"], item["K"]) for item in parallel] == [(1, 2), (2, 2)]
    assert (tmp_path / "parallel-cache.sqlite").is_file()


def test_persistent_parallel_cache_reproduces_identical_cells(tmp_path: Path) -> None:
    jobs = _jobs()
    cache_path = tmp_path / "resume-cache.sqlite"
    with S2CompileDispatcher(
        workers=2,
        persistent_cache_path=cache_path,
        maximum_cache_entries=64,
    ) as first_dispatcher:
        first = first_dispatcher.compile(jobs)
    with S2CompileDispatcher(
        workers=2,
        persistent_cache_path=cache_path,
        maximum_cache_entries=64,
    ) as resumed_dispatcher:
        resumed = resumed_dispatcher.compile(jobs)
    assert resumed == first


def test_parallel_dispatcher_rejects_unsafe_worker_counts_and_ordinals() -> None:
    with pytest.raises(ValueError, match="between 1"):
        S2CompileDispatcher(workers=0, persistent_cache_path=None)
    with pytest.raises(ValueError, match="between 1"):
        S2CompileDispatcher(
            workers=MAXIMUM_PARALLEL_WORKERS + 1,
            persistent_cache_path=None,
        )

    jobs = _jobs()
    duplicate = [jobs[0], jobs[1].__class__(**{**jobs[1].__dict__, "ordinal": 0})]
    with S2CompileDispatcher(
        workers=1,
        persistent_cache_path=None,
        maximum_cache_entries=64,
    ) as dispatcher:
        with pytest.raises(ValueError, match="ordinals"):
            dispatcher.compile(duplicate)


def test_parallel_toy_s2_matches_serial_across_all_stage_barriers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    block_count = 12
    hamiltonian = DFHamiltonian(
        constant=0.07,
        one_body=np.asarray([[0.19]], dtype=np.complex128),
        lambdas=np.linspace(0.05, 0.16, block_count),
        g_matrices=tuple(
            np.asarray([[0.25 + 0.01 * index]], dtype=np.complex128)
            for index in range(block_count)
        ),
        metadata={"name": "pr2-v4-s2-parallel-integration-test"},
    )
    state = np.zeros(2**hamiltonian.n_qubits, dtype=np.complex128)
    state[0] = 1.0

    def fake_load_inputs(_root: Path, counters: dict[str, int]):
        counters["development_raw_hash_checks"] += 1
        counters["held_out_raw_hash_checks"] += 1
        counters["development_npz_loads"] += 1
        return hamiltonian, state, {"hamiltonian_hash": "toy-hash"}

    monkeypatch.setattr(validation, "validate_v4_payload", lambda _payload: None)
    monkeypatch.setattr(validation, "_load_inputs", fake_load_inputs)
    monkeypatch.setattr(validation, "S1_R_VALUES", (1,))
    monkeypatch.setattr(validation, "S1_K_VALUES", (2,))
    monkeypatch.setattr(validation, "S2_Q", 1)
    monkeypatch.setattr(validation, "S2_INITIAL_SAMPLE_COUNT", 2)
    monkeypatch.setattr(validation, "S2_EXTENSION_SAMPLE_COUNT", 2)
    monkeypatch.setattr(validation, "S2_TOTAL_SAMPLE_COUNT", 4)

    v4_payload = {
        "status": validation.V4_PASS_STATUS,
        "deviations": [],
        "result_fingerprint": "toy-v4-fingerprint",
    }
    serial = validation.run_s2_development(
        tmp_path,
        v4_payload,
        provenance={"execution": "comparison"},
    )
    parallel = run_s2_development_parallel(
        tmp_path,
        v4_payload,
        provenance={"execution": "comparison"},
        workers=2,
        persistent_cache_path=tmp_path / "integration-cache.sqlite",
    )

    def scientific_payload(payload: dict) -> dict:
        normalized = copy.deepcopy(payload)
        normalized.pop("result_fingerprint")
        normalized.pop("provenance")
        return normalized

    assert scientific_payload(parallel) == scientific_payload(serial)
