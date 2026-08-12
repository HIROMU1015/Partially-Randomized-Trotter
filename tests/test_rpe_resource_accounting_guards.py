from __future__ import annotations

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.rpe_resource_accounting import (
    RPEErrorAllocation,
    RPEPFErrorModel,
    RPERoundCompiledCost,
    RPERoundSpecification,
    build_rpe_resource_summary,
    evaluate_rpe_round_candidate,
)
from trotterlib.rte import CircuitCost, CompilerSettings


def _compiler(seed: int = 17) -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=seed,
        qiskit_version=qiskit.__version__,
    )


def _cost(compiler: CompilerSettings) -> CircuitCost:
    return CircuitCost(
        rz_count=2.0,
        rz_depth=2.0,
        cx_count=3.0,
        cx_depth=3.0,
        total_depth=5.0,
        circuit_size=7.0,
        compiler=compiler,
        fidelity_level=5,
        estimate_kind="exact_compiled_repeated_partial_s2_expectation",
    )


class _CostProvider:
    def __init__(
        self,
        *,
        compiler: CompilerSettings | None = None,
        scope: str = "compiled_time_evolution_subcircuit",
        cost_model_fingerprint: str | None = None,
    ) -> None:
        self.compiler = compiler or _compiler()
        self.scope = scope
        self.cost_model_fingerprint = cost_model_fingerprint

    def evaluate(self, request) -> RPERoundCompiledCost:
        cost = _cost(self.compiler)
        return RPERoundCompiledCost(
            cosine_expected_cost=cost,
            sine_expected_cost=cost,
            cosine_standard_error=None,
            sine_standard_error=None,
            evaluation_method="guard_test_provider",
            classical_sample_count=None,
            circuit_cost_scope=self.scope,
            cost_model_fingerprint=self.cost_model_fingerprint,
        )


def _preparation(
    *,
    primary_lambda: float = 2.4,
    deterministic_only: bool = False,
    with_threshold_error: bool = False,
):
    lambdas = [primary_lambda]
    if with_threshold_error:
        lambdas.append(0.002)
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray(lambdas),
        g_matrices=tuple(
            np.asarray([[1.0]], dtype=np.complex128) for _ in lambdas
        ),
        metadata={"name": f"rpe-guard-{primary_lambda}-{len(lambdas)}"},
    )
    ld = hamiltonian.n_blocks if deterministic_only else 0
    return prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, ld),
        identity_policy="extract_identity_phase",
        coefficient_atol=0.0011 if with_threshold_error else 0.0,
    )


def _allocation() -> RPEErrorAllocation:
    return RPEErrorAllocation(
        beta_pf_budget=0.001,
        beta_rte_budget=0.05,
        beta_stat_budget=0.30,
        alpha_cosine=0.01,
        alpha_sine=0.01,
    )


def _candidate(
    round_index: int,
    *,
    preparation=None,
    delta_time: float = 0.1,
    beta_rpe: float = 0.4,
    rte_seed: int = 0,
    pf_error_model: RPEPFErrorModel | None = None,
    provider: _CostProvider | None = None,
):
    return evaluate_rpe_round_candidate(
        preparation or _preparation(),
        RPERoundSpecification(round_index, delta_time),
        _allocation(),
        pf_error_model
        or RPEPFErrorModel(0.02, "external_certified_bound", True),
        beta_rpe=beta_rpe,
        rte_steps_per_occurrence=2,
        finite_taylor_order=2,
        cost_metric="rz_count",
        cost_provider=provider or _CostProvider(),
        rte_seed=rte_seed,
    )


def test_rigorous_but_infeasible_candidate_is_not_certified() -> None:
    candidate = evaluate_rpe_round_candidate(
        _preparation(),
        RPERoundSpecification(2, 1.0),
        RPEErrorAllocation(0.01, 1.0, 0.01, 0.01, 0.01),
        RPEPFErrorModel(0.02, "external_certified_bound", True),
        beta_rpe=1.1,
        rte_steps_per_occurrence=1,
        finite_taylor_order=0,
        cost_metric="rz_count",
        cost_provider=_CostProvider(),
    )

    assert not candidate.feasible
    assert candidate.guarantee_status == "not_certified"


@pytest.mark.parametrize(
    "source",
    ("state_specific_phase_bias_surrogate", "df_phase_bias_surrogate_v3"),
)
def test_known_empirical_pf_sources_cannot_be_marked_rigorous(source: str) -> None:
    with pytest.raises(ValueError, match="empirical screening surrogate"):
        RPEPFErrorModel(0.02, source, True)


def test_rigorous_union_bound_failure_is_not_certified() -> None:
    provider = _CostProvider(cost_model_fingerprint="verified-model")
    first = _candidate(0, provider=provider)
    second = _candidate(1, provider=provider)

    summary = build_rpe_resource_summary(
        (first, second), total_alpha_budget=0.03, cost_metric="rz_count"
    )

    assert not summary.union_bound_satisfied
    assert summary.guarantee_status == "not_certified"


@pytest.mark.parametrize(
    "mismatch",
    (
        "preparation",
        "delta_time",
        "beta_rpe",
        "rte_seed",
        "pf_error_model",
        "compiler",
        "cost_model",
    ),
)
def test_summary_rejects_mixed_round_provenance(mismatch: str) -> None:
    verified_provider = _CostProvider(cost_model_fingerprint="verified-model")
    first = _candidate(0, provider=verified_provider)
    kwargs = {"provider": verified_provider}
    if mismatch == "preparation":
        kwargs["preparation"] = _preparation(primary_lambda=2.6)
    elif mismatch == "delta_time":
        kwargs["delta_time"] = 0.11
    elif mismatch == "beta_rpe":
        kwargs["beta_rpe"] = 0.41
    elif mismatch == "rte_seed":
        kwargs["rte_seed"] = 1
    elif mismatch == "pf_error_model":
        kwargs["pf_error_model"] = RPEPFErrorModel(
            0.03, "another_external_certified_bound", True
        )
    elif mismatch == "compiler":
        kwargs["provider"] = _CostProvider(
            compiler=_compiler(18),
            cost_model_fingerprint="verified-model",
        )
    elif mismatch == "cost_model":
        first = _candidate(
            0,
            provider=_CostProvider(cost_model_fingerprint="model-a"),
        )
        kwargs["provider"] = _CostProvider(cost_model_fingerprint="model-b")
    second = _candidate(1, **kwargs)

    with pytest.raises(ValueError):
        build_rpe_resource_summary(
            (first, second), total_alpha_budget=0.05, cost_metric="rz_count"
        )


def test_multi_round_summary_requires_verified_cost_model() -> None:
    with pytest.raises(ValueError, match="verified cost_model_fingerprint"):
        build_rpe_resource_summary(
            (_candidate(0), _candidate(1)),
            total_alpha_budget=0.05,
            cost_metric="rz_count",
        )


def test_nonzero_threshold_error_cannot_be_certified() -> None:
    preparation = _preparation(with_threshold_error=True)
    assert preparation.threshold_operator_error_bound > 0.0

    candidate = _candidate(0, preparation=preparation)

    assert candidate.feasible
    assert candidate.guarantee_status == "not_certified"


def test_deterministic_only_candidate_requires_canonical_r_and_k() -> None:
    preparation = _preparation(deterministic_only=True)
    common = dict(
        preparation=preparation,
        specification=RPERoundSpecification(0, 0.1),
        allocation=_allocation(),
        pf_error_model=RPEPFErrorModel(0.02, "external_certified_bound", True),
        beta_rpe=0.4,
        cost_metric="rz_count",
        cost_provider=_CostProvider(),
    )

    with pytest.raises(ValueError):
        evaluate_rpe_round_candidate(
            **common, rte_steps_per_occurrence=1, finite_taylor_order=0
        )
    with pytest.raises(ValueError):
        evaluate_rpe_round_candidate(
            **common, rte_steps_per_occurrence=0, finite_taylor_order=2
        )

    canonical = evaluate_rpe_round_candidate(
        **common, rte_steps_per_occurrence=0, finite_taylor_order=0
    )
    assert canonical.feasible


def test_summary_rejects_missing_intermediate_round() -> None:
    with pytest.raises(ValueError):
        build_rpe_resource_summary(
            (_candidate(0), _candidate(2)),
            total_alpha_budget=0.05,
            cost_metric="rz_count",
        )


def test_exclusion_assumption_depends_on_provider_scope() -> None:
    compiled = _candidate(0)
    full = _candidate(0, provider=_CostProvider(scope="full_rpe_circuit"))
    exclusion = "state_preparation_hadamard_measurement_noise_and_backend_runs_excluded"

    assert exclusion in compiled.assumptions
    assert exclusion not in full.assumptions
    assert "ordinary_controlled_diag_I_U_for_level5r_provider" in compiled.assumptions
    assert "ordinary_controlled_diag_I_U_for_level5r_provider" not in full.assumptions
