from __future__ import annotations

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.df_rpe_resource import DFLevel5RCompiledCostProvider
from trotterlib.rpe_resource_accounting import (
    RPEErrorAllocation,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPERoundCostRequest,
    RPERoundSpecification,
    evaluate_rpe_round_candidate,
)
from trotterlib.rte import CompilerSettings



FRESH_IID_POLICY = RPEHadamardSamplingPolicy(
    rte_trajectory_mode="fresh_iid_per_hadamard_shot",
    independent_bounded_outcomes_within_each_round_axis=True,
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


def _preparation(*, deterministic: bool = False):
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([2.4]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "df-rpe-resource-provider"},
    )
    return prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1 if deterministic else 0),
        identity_policy="extract_identity_phase",
    )


def _allocation(*, rte_budget: float = 0.01) -> RPEErrorAllocation:
    return RPEErrorAllocation(
        beta_pf_budget=1e-5,
        beta_rte_budget=rte_budget,
        beta_stat_budget=0.30,
        alpha_cosine=0.02,
        alpha_sine=0.03,
    )


def test_exact_provider_connects_to_ordinary_controlled_level5r() -> None:
    preparation = _preparation()
    provider = DFLevel5RCompiledCostProvider(
        compiler=_compiler(),
        evaluation_method="exact",
        maximum_trajectories=10,
    )
    candidate = evaluate_rpe_round_candidate(
        preparation,
        RPERoundSpecification(0, 0.05),
        _allocation(),
        RPEPFErrorModel(0.02, "state_specific_phase_bias_surrogate", False),
        beta_rpe=0.4,
        rte_steps_per_occurrence=1,
        finite_taylor_order=0,
        cost_metric="rz_count",
        cost_provider=provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )

    assert candidate.feasible
    assert candidate.cost_evaluation_method == "exact"
    assert candidate.classical_cost_sample_count is None
    assert candidate.circuit_cost_scope == "compiled_time_evolution_subcircuit"
    assert candidate.cosine_expected_cost is candidate.sine_expected_cost
    assert candidate.cosine_expected_cost.fidelity_level == 5
    assert candidate.cosine_expected_cost.estimate_kind == (
        "exact_compiled_repeated_partial_s2_expectation"
    )
    assert candidate.round_total_cost == pytest.approx(
        (candidate.cosine_shots + candidate.sine_shots)
        * candidate.cosine_expected_cost.rz_count
    )
    metadata = dict(candidate.cost_metadata)
    assert metadata["ordinary_control_semantics"] == "diag(I,U)"
    assert metadata["hadamard_test_included"] is False
    assert metadata["controlled"] is True
    assert metadata["ancilla_qubit"] == preparation.num_system_qubits
    assert metadata["backend_context_canonical"] is True
    assert metadata["backend_fingerprint"] == "no_backend"
    assert metadata["rte_prng_type"] == "numpy.random.PCG64"
    assert metadata["rte_sampling_convention_version"]
    assert metadata["numpy_version"]
    assert candidate.cost_model_fingerprint is not None
    assert len(candidate.cost_model_fingerprint) == 64


def test_monte_carlo_sample_count_is_metadata_not_round_cost_multiplier() -> None:
    preparation = _preparation()
    provider = DFLevel5RCompiledCostProvider(
        compiler=_compiler(),
        evaluation_method="monte_carlo",
        sample_count=3,
        seed=123,
        maximum_samples=3,
    )
    candidate = evaluate_rpe_round_candidate(
        preparation,
        RPERoundSpecification(0, 0.05),
        _allocation(),
        RPEPFErrorModel(0.02, "external_certified_bound", True),
        beta_rpe=0.4,
        rte_steps_per_occurrence=1,
        finite_taylor_order=2,
        cost_metric="circuit_size",
        cost_provider=provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )

    expected = (
        candidate.cosine_shots * candidate.cosine_expected_cost.circuit_size
        + candidate.sine_shots * candidate.sine_expected_cost.circuit_size
    )
    assert candidate.classical_cost_sample_count == 3
    assert candidate.round_total_cost == pytest.approx(expected)
    assert candidate.round_total_cost != pytest.approx(3 * expected)


def test_empty_tail_short_circuits_requested_monte_carlo_to_exact() -> None:
    preparation = _preparation(deterministic=True)
    provider = DFLevel5RCompiledCostProvider(
        compiler=_compiler(),
        evaluation_method="monte_carlo",
        sample_count=5,
        seed=123,
        maximum_samples=5,
    )
    candidate = evaluate_rpe_round_candidate(
        preparation,
        RPERoundSpecification(1, 0.05),
        _allocation(rte_budget=0.0),
        RPEPFErrorModel(0.02, "state_specific_phase_bias_surrogate", False),
        beta_rpe=0.4,
        rte_steps_per_occurrence=0,
        finite_taylor_order=0,
        cost_metric="total_depth",
        cost_provider=provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )

    assert candidate.feasible
    assert candidate.epsilon_z == 0.0
    assert candidate.attenuation == 1.0
    assert candidate.cost_evaluation_method == "exact"
    assert candidate.classical_cost_sample_count is None
    metadata = dict(candidate.cost_metadata)
    assert metadata["requested_evaluation_method"] == "monte_carlo"
    assert metadata["deterministic_exact_short_circuit"] is True
    assert metadata["processed_trajectory_count"] == 1


def test_provider_rejects_incomplete_monte_carlo_configuration() -> None:
    with pytest.raises(ValueError, match="requires sample_count and seed"):
        DFLevel5RCompiledCostProvider(
            compiler=_compiler(), evaluation_method="monte_carlo"
        )


@pytest.mark.parametrize(
    ("rte_steps_per_occurrence", "finite_taylor_order"),
    ((1, 0), (0, 2)),
)
def test_empty_tail_requires_canonical_rte_candidate(
    rte_steps_per_occurrence: int,
    finite_taylor_order: int,
) -> None:
    preparation = _preparation(deterministic=True)
    request = RPERoundCostRequest(
        preparation=preparation,
        specification=RPERoundSpecification(0, 0.05),
        allocation=_allocation(rte_budget=0.0),
        rte_steps_per_occurrence=rte_steps_per_occurrence,
        finite_taylor_order=finite_taylor_order,
        rte_config=None,
        rte_distribution=None,
    )

    with pytest.raises(ValueError, match="canonical"):
        DFLevel5RCompiledCostProvider(compiler=_compiler()).evaluate(request)


def test_provider_default_short_round_guard_rejects_q_above_four() -> None:
    preparation = _preparation(deterministic=True)
    request = RPERoundCostRequest(
        preparation=preparation,
        specification=RPERoundSpecification(3, 0.05),
        allocation=_allocation(rte_budget=0.0),
        rte_steps_per_occurrence=0,
        finite_taylor_order=0,
        rte_config=None,
        rte_distribution=None,
    )

    with pytest.raises(ValueError, match=r"q_m=8.*maximum_repetition_count=4"):
        DFLevel5RCompiledCostProvider(compiler=_compiler()).evaluate(request)
