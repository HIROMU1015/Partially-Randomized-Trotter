from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.research_direction_ablation import (
    AffineAxisCostModel,
    PairCompiledCostModel,
)
from trotterlib.research_direction_pf_sensitivity import (
    evaluate_wp03_sensitivity,
    extract_pf_coefficient_audit,
    finalize_wp03_artifact,
    validate_wp03_artifact,
)


PF_DIRECTORY = Path(
    "artifacts/research_direction_prevalidation/2026-09-21/"
    "pf_delta_same_snapshot"
)


def _hamiltonian() -> DFHamiltonian:
    matrix = np.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=np.complex128)
    return DFHamiltonian(
        constant=0.1,
        one_body=np.asarray(
            [[0.2, 0.01], [0.01, -0.1]], dtype=np.complex128
        ),
        lambdas=np.full(12, 1e-4),
        g_matrices=tuple(matrix.copy() for _ in range(12)),
        metadata={"name": "research-direction-pf-sensitivity-toy"},
    )


def _axis(q1: float, q2: float) -> AffineAxisCostModel:
    return AffineAxisCostModel(q1, q2, 0.0, 0.0)


def _model(r_m: int, k_m: int, q1: float, q2: float) -> PairCompiledCostModel:
    return PairCompiledCostModel(
        rte_steps=r_m,
        finite_taylor_order=k_m,
        cosine=_axis(q1, q2),
        sine=_axis(q1, q2),
        dataset_fingerprint="0" * 64,
        proxy_fingerprint="1" * 64,
    )


def test_wp03_artifact_fingerprint_is_tamper_evident() -> None:
    payload = finalize_wp03_artifact(
        {
            "scope": {
                "final_total_cost_evaluation_performed": False,
                "decision_grade": False,
            },
            "checks": {"test": True},
            "overall_pass": True,
        },
        provenance={"test": True},
    )
    validate_wp03_artifact(payload)
    tampered = deepcopy(payload)
    tampered["overall_pass"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_wp03_artifact(tampered)


def test_coefficient_audit_detects_signed_convention_mismatch() -> None:
    payloads = {
        ld: json.loads(
            (PF_DIRECTORY / f"h4_sto3g_d100_rank12_ld{ld}_v5.json").read_text(
                encoding="utf-8"
            )
        )
        for ld in (0, 3, 12)
    }
    common_fits = {
        ld: {
            "delta_values": [0.05, 0.1, 0.2, 0.4],
            "fixed_second_order_coefficient": (
                0.0 if ld == 0 else 0.01 + ld * 1e-4
            ),
            "signed_energy_biases": [0.0, 1e-4, 4e-4, 1.6e-3],
        }
        for ld in (0, 3, 12)
    }
    audit = extract_pf_coefficient_audit(
        payloads,
        common_window_hd_fits=common_fits,
    )

    assert not audit["all_signed_conventions_directly_aligned"]
    assert audit["common_delta_window"] == [0.05, 0.1, 0.2, 0.4]
    assert audit["rows"]["0"]["coefficients"]["hd_surrogate"] == 0.0
    assert audit["rows"]["3"]["paper_d6_conditioned_deltas"] == [
        0.05,
        0.1,
        0.2,
        0.4,
    ]
    assert not audit["rows"]["3"]["commutator_term_decomposition_available"]


def test_wp03_sensitivity_changes_only_coefficient_and_builds_regret() -> None:
    hamiltonian = _hamiltonian()
    preparations = {
        ld: prepare_df_partial_s2(
            hamiltonian,
            split_df_hamiltonian_by_ld(hamiltonian, ld),
            identity_policy="extract_identity_phase",
        )
        for ld in (3, 12)
    }
    coefficient_audit = {
        "rows": {
            str(ld): {
                "coefficients": {
                    "hd_surrogate": 1e-5,
                    "paper_d6": 1.1e-5,
                    "dominant_eigenphase": 1.05e-5,
                },
                "paper_d6_relative_difference_vs_dominant_eigenphase": (
                    1.1e-5 / 1.05e-5 - 1.0
                ),
            }
            for ld in (3, 12)
        },
        "all_paper_d6_estimators_pass": True,
        "all_single_phase_cost_models_pass": True,
    }
    body = evaluate_wp03_sensitivity(
        preparations,
        cost_models={
            3: (_model(1, 0, 20.0, 40.0),),
            12: (_model(0, 0, 10.0, 20.0),),
        },
        coefficient_audit=coefficient_audit,
        delta_values=(0.01,),
        target_energy_precision=1.0,
    )

    assert body["overall_pass"]
    assert len(body["scenarios"]) == 6
    assert body["selection_sensitivity"]["ld_and_delta_selection_invariant"]
    assert all(
        row["relative_regret"] >= 0.0
        for row in body["cross_policy_regret"]
    )


def test_wp03_rejects_incomplete_costed_ld_set() -> None:
    with pytest.raises(ValueError, match="requires L_D=3 and L_D=12"):
        evaluate_wp03_sensitivity(
            {},
            cost_models={},
            coefficient_audit={"rows": {}},
        )
