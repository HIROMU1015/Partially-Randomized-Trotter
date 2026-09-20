from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import qiskit

from trotterlib.rpe_four_round_accounting_validation import (
    validate_rpe_four_round_accounting,
    validate_rpe_four_round_accounting_payload,
)
from trotterlib.rpe_hadamard_compiled_cost_proxy import (
    RPEHadamardCompiledCostProxyValidationResult,
)
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / (
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
ALLOCATION = ROOT / (
    "artifacts/rpe_allocation_sensitivity_validation/2026-09-01/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_"
    "beta_alpha_sensitivity_v1.json"
)
DIRECT = ROOT / (
    "artifacts/rpe_round_cost_connection_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_mc8_v1.json"
)
PROXY = ROOT / (
    "artifacts/rpe_hadamard_proxy_resource_validation/2026-09-01/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_cal_q1_q2_q4_"
    "hold_q8_mc8_v1.proxy_validation.json"
)
FAILURE = ROOT / (
    "artifacts/rpe_hadamard_failure_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_"
    "marginal100000_fresh_v1.json"
)
OUTPUT = ROOT / (
    "artifacts/rpe_four_round_accounting_validation/2026-09-18/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_limited_v1.json"
)


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


def test_saved_four_round_result_recomputes_and_is_tamper_evident() -> None:
    saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
    validate_rpe_four_round_accounting_payload(saved)
    recomputed = validate_rpe_four_round_accounting(
        load_connected_cluster_hamiltonian_snapshot(SNAPSHOT),
        _compiler(),
        json.loads(ALLOCATION.read_text(encoding="utf-8")),
        json.loads(DIRECT.read_text(encoding="utf-8")),
        RPEHadamardCompiledCostProxyValidationResult.read_json(PROXY),
        json.loads(FAILURE.read_text(encoding="utf-8")),
    )
    assert recomputed["summary"]["overall_pass"]
    assert recomputed["limited_aggregation"]["total_rz_cost"] == pytest.approx(
        saved["limited_aggregation"]["total_rz_cost"]
    )
    assert recomputed["limited_aggregation"]["guarantee_status"] == (
        "empirical_screening"
    )
    assert recomputed["short_round_exact_binomial"]["q_values"] == [1, 2, 4]
    assert recomputed["scope"]["q8_physical_signal_or_phase_evaluated"] is False

    tampered = deepcopy(saved)
    tampered["limited_aggregation"]["total_rz_cost"] += 1.0
    with pytest.raises(ValueError, match="content_fingerprint"):
        validate_rpe_four_round_accounting_payload(tampered)


def test_four_round_result_rejects_mismatched_physical_signal_source() -> None:
    allocation = json.loads(ALLOCATION.read_text(encoding="utf-8"))
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    failure["hamiltonian"]["hamiltonian_hash"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_rpe_four_round_accounting(
            load_connected_cluster_hamiltonian_snapshot(SNAPSHOT),
            _compiler(),
            allocation,
            json.loads(DIRECT.read_text(encoding="utf-8")),
            RPEHadamardCompiledCostProxyValidationResult.read_json(PROXY),
            failure,
        )
