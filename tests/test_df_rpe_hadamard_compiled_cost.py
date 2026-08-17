from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import qiskit
from qiskit.quantum_info import Operator

import trotterlib.df_rpe_hadamard_compiled_cost as hadamard_cost_module
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.df_partial_s2_repeated import (
    QiskitDFPartialS2RepeatedCircuitBuilder,
)
from trotterlib.df_partial_s2_repeated_cost import (
    make_exact_df_partial_s2_repeated_trajectory_stream,
)
from trotterlib.df_rpe_hadamard_compiled_cost import (
    DFRPEHadamardCompiledCostProvider,
    estimate_exact_compiled_rpe_hadamard_cost,
    estimate_monte_carlo_compiled_rpe_hadamard_cost,
    plan_compiled_rpe_hadamard_workload,
)
from trotterlib.df_rpe_resource import DFLevel5RCompiledCostProvider
from trotterlib.rpe_hadamard_interrogation import (
    QiskitRPEHadamardInterrogationBuilder,
    RPEHadamardInterrogationRequest,
)
from trotterlib.rpe_resource_accounting import (
    RPEErrorAllocation,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPERoundCostRequest,
    RPERoundSpecification,
    evaluate_rpe_round_candidate,
)
from trotterlib.rte import CompilerSettings, make_rte_config
from trotterlib.rte_compiled_cost import (
    TranspiledCircuitCostCache,
    transpile_and_measure_cost,
)


METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)
FRESH_IID_POLICY = RPEHadamardSamplingPolicy(
    rte_trajectory_mode="fresh_iid_per_hadamard_shot",
    independent_bounded_outcomes_within_each_round_axis=True,
)


def _compiler(*, seed: int = 17, optimization_level: int = 1) -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=optimization_level,
        layout_method=None,
        routing_method=None,
        transpiler_seed=seed,
        qiskit_version=qiskit.__version__,
    )


def _case(*, deterministic: bool = False, rte_steps: int = 1):
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "rpe-hadamard-compiled-cost"},
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1 if deterministic else 0),
        identity_policy="extract_identity_phase",
    )
    if deterministic:
        return preparation, None, None, 0.2
    step_time = 1.2 * rte_steps / preparation.exact_rte_lambda_r
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=step_time,
        rte_steps=rte_steps,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    return preparation, config, distribution, step_time


def _allocation(*, rte_budget: float) -> RPEErrorAllocation:
    return RPEErrorAllocation(
        beta_pf_budget=1e-5,
        beta_rte_budget=rte_budget,
        beta_stat_budget=0.3,
        alpha_cosine=0.02,
        alpha_sine=0.03,
    )


def _round_request(
    preparation,
    config,
    distribution,
    *,
    round_index: int,
    step_time: float,
) -> RPERoundCostRequest:
    return RPERoundCostRequest(
        preparation=preparation,
        specification=RPERoundSpecification(round_index, step_time),
        allocation=_allocation(rte_budget=0.0 if config is None else 0.01),
        rte_steps_per_occurrence=0 if config is None else config.rte_steps,
        finite_taylor_order=(
            0 if config is None else config.finite_taylor_order
        ),
        rte_config=config,
        rte_distribution=distribution,
    )


def _assert_metric_values_equal(left, right) -> None:
    for metric in METRICS:
        assert getattr(left, metric) == pytest.approx(getattr(right, metric))


class _CountingCache(TranspiledCircuitCostCache):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.requested_fingerprints: list[str] = []

    def get_or_transpile(self, circuit, compiler, *, circuit_fingerprint, backend=None):
        self.call_count += 1
        self.requested_fingerprints.append(circuit_fingerprint)
        return super().get_or_transpile(
            circuit,
            compiler,
            circuit_fingerprint=circuit_fingerprint,
            backend=backend,
        )


class _FailingCache(TranspiledCircuitCostCache):
    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at
        self.call_count = 0

    def get_or_transpile(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count == self.fail_at:
            raise RuntimeError("injected transpile failure")
        return super().get_or_transpile(*args, **kwargs)


@pytest.mark.parametrize(
    ("repetition_count", "round_index"),
    ((1, 0), (2, 1), (4, 2)),
)
def test_deterministic_wrapper_cost_supports_all_short_rounds(
    repetition_count: int,
    round_index: int,
) -> None:
    preparation, config, distribution, step_time = _case(deterministic=True)
    result = estimate_exact_compiled_rpe_hadamard_cost(
        preparation,
        step_time,
        repetition_count,
        config,
        distribution,
        _compiler(),
    )

    assert result.round_index == round_index
    assert result.q_m == repetition_count
    assert result.t_m == pytest.approx(repetition_count * step_time)
    assert result.processed_trajectory_count == 1
    assert result.complete
    assert result.measurement_included
    assert not result.state_preparation_included
    assert result.circuit_scope == (
        "single_hadamard_interrogation_without_state_preparation"
    )


def test_randomized_exact_cost_uses_one_shared_trajectory_for_both_axes() -> None:
    preparation, config, distribution, step_time = _case()
    cache = _CountingCache()
    result = estimate_exact_compiled_rpe_hadamard_cost(
        preparation,
        step_time,
        1,
        config,
        distribution,
        _compiler(),
        maximum_trajectories=10,
        cache=cache,
    )

    assert result.processed_trajectory_count == result.trajectory_space_size == 2
    assert cache.call_count == 2 * result.processed_trajectory_count
    assert result.shared_axis_trajectory_set
    assert result.cosine.axis == "cosine"
    assert result.sine.axis == "sine"
    assert result.cosine.compiled_cost_evaluation_fingerprint != (
        result.sine.compiled_cost_evaluation_fingerprint
    )
    for record in result.retained_trajectory_records:
        assert record.evolution_provenance_fingerprint
        assert record.cosine_wrapper_circuit_semantics_fingerprint != (
            record.sine_wrapper_circuit_semantics_fingerprint
        )
    assert cache.requested_fingerprints[0::2] == [
        item.cosine_wrapper_circuit_semantics_fingerprint
        for item in result.retained_trajectory_records
    ]
    assert cache.requested_fingerprints[1::2] == [
        item.sine_wrapper_circuit_semantics_fingerprint
        for item in result.retained_trajectory_records
    ]


def test_provider_matches_direct_full_wrapper_transpilation() -> None:
    preparation, config, distribution, step_time = _case(deterministic=True)
    compiler = _compiler()
    stream = make_exact_df_partial_s2_repeated_trajectory_stream(
        preparation,
        step_time,
        2,
        config,
        distribution,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
    )
    request, probability = next(iter(stream.records))
    assert probability == 1.0
    evolution = QiskitDFPartialS2RepeatedCircuitBuilder().build(request)
    builder = QiskitRPEHadamardInterrogationBuilder()
    direct_costs = {}
    for axis in ("cosine", "sine"):
        wrapper = builder.build(
            RPEHadamardInterrogationRequest(evolution, axis, True)
        )
        assert wrapper.circuit.data[-1].operation.name == "measure"
        direct_costs[axis] = transpile_and_measure_cost(
            wrapper.circuit,
            compiler,
            circuit_fingerprint=wrapper.compiler_independent_fingerprint,
        )

    estimated = estimate_exact_compiled_rpe_hadamard_cost(
        preparation,
        step_time,
        2,
        config,
        distribution,
        compiler,
    )

    _assert_metric_values_equal(estimated.cosine.expected_cost, direct_costs["cosine"])
    _assert_metric_values_equal(estimated.sine.expected_cost, direct_costs["sine"])
    record = estimated.retained_trajectory_records[0]
    assert record.constant_phase == evolution.constant_phase
    assert record.extracted_identity_phase == evolution.extracted_identity_phase
    assert record.rte_relative_phase == evolution.rte_relative_phase


def test_axes_are_compiled_independently_instead_of_reusing_one_cost() -> None:
    preparation, config, distribution, step_time = _case(deterministic=True)
    compiler = CompilerSettings(
        basis_gates=("h", "sdg", "rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=0,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )
    result = estimate_exact_compiled_rpe_hadamard_cost(
        preparation,
        step_time,
        1,
        config,
        distribution,
        compiler,
    )

    assert result.sine.expected_cost.total_depth == (
        result.cosine.expected_cost.total_depth + 1
    )
    assert result.sine.expected_cost.circuit_size == (
        result.cosine.expected_cost.circuit_size + 1
    )


def test_exact_expectation_is_probability_weighted_not_a_simple_mean() -> None:
    preparation, config, distribution, step_time = _case()
    result = estimate_exact_compiled_rpe_hadamard_cost(
        preparation,
        step_time,
        1,
        config,
        distribution,
        _compiler(),
        maximum_trajectories=10,
    )
    records = result.retained_trajectory_records
    assert len(records) == 2
    assert records[0].probability != pytest.approx(records[1].probability)

    for axis in ("cosine", "sine"):
        values = [getattr(record, f"{axis}_cost") for record in records]
        weighted = math.fsum(
            record.probability * value.rz_count
            for record, value in zip(records, values, strict=True)
            if record.probability is not None
        )
        simple = math.fsum(value.rz_count for value in values) / len(values)
        expected = getattr(result, axis).expected_cost.rz_count
        assert expected == pytest.approx(weighted)
        assert expected != pytest.approx(simple)
    assert result.trajectory_probability_sum == pytest.approx(1.0)


def test_exact_result_and_fingerprint_are_enumeration_order_independent() -> None:
    preparation, config, distribution, step_time = _case()
    compiler = _compiler()
    stream = make_exact_df_partial_s2_repeated_trajectory_stream(
        preparation,
        step_time,
        1,
        config,
        distribution,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        maximum_trajectories=10,
    )
    records = tuple(stream.records)
    plan = plan_compiled_rpe_hadamard_workload(
        preparation,
        1,
        config,
        distribution,
        trajectory_count=len(records),
    )
    forward = hadamard_cost_module._compile_hadamard_trajectory_stream_with_plan(
        replace(stream, records=iter(records)),
        step_time,
        compiler,
        construction_policy="boundary_optimized",
        workload_plan=plan,
        maximum_untranspiled_circuit_size=100_000,
        maximum_retained_trajectory_records=10,
        maximum_build_requests=100,
        maximum_transpile_requests=100,
        maximum_planned_instruction_applications=1_000_000,
        cache=None,
        backend=None,
    )
    reverse = hadamard_cost_module._compile_hadamard_trajectory_stream_with_plan(
        replace(stream, records=iter(reversed(records))),
        step_time,
        compiler,
        construction_policy="boundary_optimized",
        workload_plan=plan,
        maximum_untranspiled_circuit_size=100_000,
        maximum_retained_trajectory_records=10,
        maximum_build_requests=100,
        maximum_transpile_requests=100,
        maximum_planned_instruction_applications=1_000_000,
        cache=None,
        backend=None,
    )

    _assert_metric_values_equal(
        forward.cosine.expected_cost,
        reverse.cosine.expected_cost,
    )
    _assert_metric_values_equal(forward.sine.expected_cost, reverse.sine.expected_cost)
    assert forward.compiled_cost_evaluation_fingerprint == (
        reverse.compiled_cost_evaluation_fingerprint
    )
    assert forward.trajectory_provenance_digest == (
        reverse.trajectory_provenance_digest
    )


def test_monte_carlo_is_reproducible_and_records_statistics() -> None:
    preparation, config, distribution, step_time = _case()
    arguments = dict(
        preparation=preparation,
        step_time=step_time,
        repetition_count=2,
        rte_config=config,
        rte_distribution=distribution,
        compiler=_compiler(),
        sample_count=8,
        seed=123,
        maximum_samples=8,
    )
    first = estimate_monte_carlo_compiled_rpe_hadamard_cost(**arguments)
    second = estimate_monte_carlo_compiled_rpe_hadamard_cost(**arguments)
    changed = estimate_monte_carlo_compiled_rpe_hadamard_cost(
        **{**arguments, "seed": 124}
    )

    assert first.sampled_trajectory_seeds == second.sampled_trajectory_seeds
    assert first.compiled_cost_evaluation_fingerprint == (
        second.compiled_cost_evaluation_fingerprint
    )
    assert first.trajectory_provenance_digest == second.trajectory_provenance_digest
    assert first.trajectory_provenance_digest != changed.trajectory_provenance_digest
    assert first.sampled_trajectory_seed_digest != (
        changed.sampled_trajectory_seed_digest
    )
    assert first.compiled_cost_evaluation_fingerprint != (
        changed.compiled_cost_evaluation_fingerprint
    )

    for axis in ("cosine", "sine"):
        axis_result = getattr(first, axis)
        samples = np.asarray(
            [
                getattr(getattr(record, f"{axis}_cost"), "circuit_size")
                for record in first.retained_trajectory_records
            ],
            dtype=np.float64,
        )
        stats = dict(axis_result.metric_statistics)["circuit_size"]
        assert stats.mean == pytest.approx(float(np.mean(samples)))
        assert stats.unbiased_sample_variance == pytest.approx(
            float(np.var(samples, ddof=1))
        )
        assert stats.standard_error == pytest.approx(
            float(np.std(samples, ddof=1) / math.sqrt(samples.size))
        )
        assert axis_result.standard_error is not None
        assert axis_result.standard_error.circuit_size == pytest.approx(
            stats.standard_error
        )


def test_deterministic_unused_seed_does_not_change_wrapper_semantics() -> None:
    preparation, config, distribution, step_time = _case(deterministic=True)
    first = estimate_monte_carlo_compiled_rpe_hadamard_cost(
        preparation,
        step_time,
        1,
        config,
        distribution,
        _compiler(),
        sample_count=1,
        seed=12,
        maximum_samples=1,
    )
    second = estimate_monte_carlo_compiled_rpe_hadamard_cost(
        preparation,
        step_time,
        1,
        config,
        distribution,
        _compiler(),
        sample_count=1,
        seed=13,
        maximum_samples=1,
    )
    first_record = first.retained_trajectory_records[0]
    second_record = second.retained_trajectory_records[0]

    assert first_record.evolution_provenance_fingerprint != (
        second_record.evolution_provenance_fingerprint
    )
    assert first_record.evolution_circuit_semantics_fingerprint == (
        second_record.evolution_circuit_semantics_fingerprint
    )
    assert first_record.cosine_wrapper_circuit_semantics_fingerprint == (
        second_record.cosine_wrapper_circuit_semantics_fingerprint
    )
    assert first_record.cosine_wrapper_fingerprint != (
        second_record.cosine_wrapper_fingerprint
    )


def test_compile_failure_and_incomplete_exact_search_do_not_return_estimates() -> None:
    preparation, config, distribution, step_time = _case()
    failing = _FailingCache(fail_at=2)
    with pytest.raises(RuntimeError, match="injected transpile failure"):
        estimate_exact_compiled_rpe_hadamard_cost(
            preparation,
            step_time,
            1,
            config,
            distribution,
            _compiler(),
            maximum_trajectories=10,
            cache=failing,
        )
    assert failing.call_count == 2

    untouched = _CountingCache()
    with pytest.raises(ValueError, match="above maximum_trajectories"):
        estimate_exact_compiled_rpe_hadamard_cost(
            preparation,
            step_time,
            1,
            config,
            distribution,
            _compiler(),
            maximum_trajectories=1,
            cache=untouched,
        )
    assert untouched.call_count == 0


@pytest.mark.parametrize("axis", ("cosine", "sine"))
def test_measurement_free_wrapper_preserves_operator_semantics_after_transpile(
    axis: str,
) -> None:
    preparation, config, distribution, step_time = _case(deterministic=True)
    stream = make_exact_df_partial_s2_repeated_trajectory_stream(
        preparation,
        step_time,
        1,
        config,
        distribution,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
    )
    request, _weight = next(iter(stream.records))
    evolution = QiskitDFPartialS2RepeatedCircuitBuilder().build(request)
    phased_circuit = evolution.circuit.copy()
    phased_circuit.global_phase += 0.23
    phased_evolution = replace(
        evolution,
        circuit=phased_circuit,
        circuit_semantics_fingerprint="operator-semantics-global-phase",
        compiler_independent_fingerprint="operator-semantics-global-phase",
    )
    wrapper = QiskitRPEHadamardInterrogationBuilder().build(
        RPEHadamardInterrogationRequest(phased_evolution, axis, False)
    )
    compiler = _compiler()
    transpiled = qiskit.transpile(
        wrapper.circuit,
        basis_gates=list(compiler.basis_gates),
        optimization_level=compiler.optimization_level,
        seed_transpiler=compiler.transpiler_seed,
    )

    assert Operator(wrapper.circuit).equiv(Operator(transpiled))
    assert float(wrapper.circuit.global_phase) == pytest.approx(
        float(phased_circuit.global_phase)
    )
    assert wrapper.constant_phase == evolution.constant_phase
    assert wrapper.extracted_identity_phase == evolution.extracted_identity_phase
    assert wrapper.additional_control_applied is False


def test_resource_accounting_uses_axis_costs_without_mc_sample_multiplier() -> None:
    preparation, _config, _distribution, _step_time = _case()
    provider = DFRPEHadamardCompiledCostProvider(
        compiler=_compiler(),
        evaluation_method="monte_carlo",
        sample_count=3,
        seed=123,
        maximum_samples=3,
    )
    candidate = evaluate_rpe_round_candidate(
        preparation,
        RPERoundSpecification(0, 0.05),
        _allocation(rte_budget=0.01),
        RPEPFErrorModel(0.02, "external_certified_bound", True),
        beta_rpe=0.4,
        rte_steps_per_occurrence=1,
        finite_taylor_order=2,
        cost_metric="circuit_size",
        cost_provider=provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )

    assert candidate.feasible
    assert candidate.classical_cost_sample_count == 3
    expected = (
        candidate.cosine_shots * candidate.cosine_expected_cost.circuit_size
        + candidate.sine_shots * candidate.sine_expected_cost.circuit_size
    )
    assert candidate.round_total_cost == pytest.approx(expected)
    assert candidate.round_total_cost != pytest.approx(3 * expected)
    assert candidate.circuit_cost_scope == (
        "single_hadamard_interrogation_without_state_preparation"
    )
    metadata = dict(candidate.cost_metadata)
    assert metadata["shared_axis_trajectory_set"] is True
    assert metadata["measurements_included"] is True
    assert metadata["classical_samples_are_quantum_shots"] is False
    assert metadata["fresh_iid_trajectory_per_hadamard_shot_verified"] is False
    assert (
        "single_hadamard_interrogation_includes_ancilla_gates_"
        "controlled_evolution_and_measurement"
        in candidate.assumptions
    )
    assert (
        "fresh_iid_rte_trajectory_per_hadamard_shot_assumed"
        in candidate.assumptions
    )


def test_existing_time_evolution_only_provider_scope_is_unchanged() -> None:
    preparation, config, distribution, step_time = _case(deterministic=True)
    request = _round_request(
        preparation,
        config,
        distribution,
        round_index=0,
        step_time=step_time,
    )
    existing = DFLevel5RCompiledCostProvider(_compiler()).evaluate(request)
    wrapper = DFRPEHadamardCompiledCostProvider(_compiler()).evaluate(request)

    assert existing.circuit_cost_scope == "compiled_time_evolution_subcircuit"
    assert dict(existing.metadata)["hadamard_test_included"] is False
    assert dict(existing.metadata)["measurements_included"] is False
    assert wrapper.circuit_cost_scope == (
        "single_hadamard_interrogation_without_state_preparation"
    )
    assert dict(wrapper.metadata)["hadamard_test_included"] is True
    assert existing.cosine_expected_cost.estimate_kind == (
        "exact_compiled_repeated_partial_s2_expectation"
    )
