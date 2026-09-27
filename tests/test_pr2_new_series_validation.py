from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import trotterlib.pr2_new_series_validation as validation
from trotterlib.df_hamiltonian import DFHamiltonian


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / validation.DEVELOPMENT_RELATIVE_PATH
HELD_OUT = ROOT / validation.HELD_OUT_RELATIVE_PATH
AUTHORIZATION = (
    ROOT
    / "artifacts"
    / "pr2_new_series_validation"
    / "2026-09-28"
    / "pr2_v1_v3_authorization_v1.json"
)
AMENDMENT = ROOT / "docs" / "research" / "pr2_new_series_amendment_v4.md"


def _toy_rank12() -> DFHamiltonian:
    lambdas = np.asarray(
        [0.31, -0.27, 0.23, -0.19, 0.17, -0.13, 0.11, -0.09, 0.07, -0.05, 0.03, -0.02],
        dtype=np.float64,
    )
    matrices = []
    for index in range(12):
        matrices.append(
            np.diag(
                np.asarray(
                    [0.4 + 0.07 * index, (-0.3 + 0.05 * index)],
                    dtype=np.float64,
                )
            ).astype(np.complex128)
        )
    return DFHamiltonian(
        constant=0.17,
        one_body=np.asarray([[0.2, 0.01], [0.01, -0.1]], dtype=np.complex128),
        lambdas=lambdas,
        g_matrices=tuple(matrices),
        metadata={"name": "pr2-new-series-rank12-toy"},
    )


def _minimal_stop_payload() -> dict:
    payload = {
        "schema_version": validation.SCHEMA_VERSION,
        "series_id": validation.SERIES_ID,
        "status": "STOP_V1_SNAPSHOT_INTEGRITY",
        "specification_commit": validation.SPECIFICATION_COMMIT,
        "amendment_sha256": validation.AMENDMENT_SHA256,
        "authorization_manifest_sha256": validation.AUTHORIZATION_MANIFEST_SHA256,
        "provenance": {"test": True},
        "old_series": {
            "status": "STOP_INPUT_REPRODUCTION_MISMATCH",
            "s1_authorized": False,
            "superseded": False,
        },
        "stages": {"V3": {"status": "V3_PASS"}},
        "failure": {"type": "test", "message": "test"},
        "counters": validation._new_counters(),
        "deviations": [],
        "V4_authorized": False,
        "S1_prime_authorized": False,
        "automatic_next_stage": None,
        "held_out_signal_cost_ranking_evaluated": False,
        "S2_executed": False,
        "S3_executed": False,
        "resource_winner_determined": False,
    }
    payload["result_fingerprint"] = validation._fingerprint(payload)
    return payload


def test_frozen_authorization_and_inputs_have_exact_raw_hashes() -> None:
    assert validation.file_sha256(AMENDMENT) == validation.AMENDMENT_SHA256
    assert (
        validation.file_sha256(AUTHORIZATION)
        == validation.AUTHORIZATION_MANIFEST_SHA256
    )
    assert (
        validation.file_sha256(DEVELOPMENT)
        == validation.EXPECTED_DEVELOPMENT_FILE_SHA256
    )
    assert validation.file_sha256(HELD_OUT) == validation.EXPECTED_HELD_OUT_FILE_SHA256


def test_v1_loads_development_twice_and_never_loads_held_out_npz() -> None:
    counters = validation._new_counters()

    record, hamiltonian = validation.run_v1(DEVELOPMENT, HELD_OUT, counters)

    assert record["status"] == "V1_PASS"
    assert hamiltonian.n_blocks == 12
    assert record["development"]["two_load_digest_match"]
    assert record["development"]["singlet_proven"] is False
    assert record["held_out"]["npz_loaded"] is False
    assert counters["input_files_read"] == 4
    assert counters["development_snapshot_loads"] == 2
    assert counters["held_out_raw_hash_checks"] == 1
    assert counters["held_out_npz_loads"] == 0
    assert counters["molecular_calculations"] == 0


def test_v1_rejects_raw_file_mutation_before_snapshot_load(tmp_path: Path) -> None:
    mutated = tmp_path / "mutated.npz"
    mutated.write_bytes(DEVELOPMENT.read_bytes() + b"mutation")
    counters = validation._new_counters()

    with pytest.raises(
        validation.SnapshotIntegrityError,
        match="Development raw file SHA-256 differs",
    ):
        validation.run_v1(mutated, HELD_OUT, counters)

    assert counters["development_snapshot_loads"] == 0


def test_snapshot_loader_detects_internal_array_tampering(tmp_path: Path) -> None:
    with np.load(DEVELOPMENT, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["one_body"] = arrays["one_body"].copy()
    arrays["one_body"][0, 0] += 1e-7
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)

    with pytest.raises(
        validation.SnapshotIntegrityError,
        match="one-body digest mismatch",
    ):
        validation._load_snapshot_once(tampered)


def test_v2_checks_all_frozen_ranks_without_sampling_or_cost() -> None:
    counters = validation._new_counters()

    record = validation.run_v2(_toy_rank12(), counters)

    assert record["status"] == "V2_PASS"
    assert [row["rank"] for row in record["ranks"]] == [3, 6, 9]
    assert counters["operator_reconstructions"] == 13
    assert counters["signal_evaluations"] == 0
    assert counters["wrapper_probe_trajectories"] == 0
    assert counters["candidate_trajectories"] == 0
    assert counters["circuits_compiled"] == 0
    assert counters["quantum_shots"] == 0
    for row in record["ranks"]:
        for method in ("B2-G", "B2-W"):
            method_record = row["methods"][method]
            assert method_record["exact_cover"]
            assert method_record["reconstruction_pass"]
            assert method_record["all_sampling_coefficient_signs_match"]
            assert method_record["repeated_preparation_identical"]
            assert method_record["probability_sum_error"] <= 1e-12


def test_result_fingerprint_guards_non_overwrite_and_forbidden_flags(
    tmp_path: Path,
) -> None:
    payload = _minimal_stop_payload()
    validation.validate_result_payload(payload)

    tampered = deepcopy(payload)
    tampered["V4_authorized"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validation.validate_result_payload(tampered)

    target = tmp_path / "result.json"
    validation.write_json_artifact(payload, target)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        validation.write_json_artifact(payload, target)


def test_v4_is_unconditionally_guarded() -> None:
    with pytest.raises(RuntimeError, match="not authorized"):
        validation.run_v4()

