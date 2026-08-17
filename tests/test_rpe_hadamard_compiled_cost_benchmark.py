from __future__ import annotations

import math
import statistics
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import qiskit

import trotterlib.df_rpe_hadamard_compiled_cost as cost_module
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
)
from trotterlib.rpe_hadamard_compiled_cost_benchmark import (
    QiskitRPEHadamardBenchmarkCircuitBuilder,
    RPEHadamardCompiledCostBenchmarkDataset,
    RPEHadamardCompiledCostBenchmarkRequest,
    generate_rpe_hadamard_compiled_cost_benchmark_dataset,
    round_index_for_benchmark_repetition_count,
)
from trotterlib.rpe_hadamard_interrogation import (
    QiskitRPEHadamardInterrogationBuilder,
    RPEHadamardInterrogationRequest,
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


def _compiler(*, seed: int = 17) -> CompilerSettings:
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


def _case(*, deterministic: bool):
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "medium-q-rpe-benchmark"},
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1 if deterministic else 0),
        identity_policy="extract_identity_phase",
    )
    if deterministic:
        return preparation, None, None, 0.2
    step_time = 1.2 / preparation.exact_rte_lambda_r
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=step_time,
        rte_steps=1,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    return preparation, config, distribution, step_time


def _request(
    *,
    deterministic: bool,
    calibration: tuple[int, ...] = (8,),
    holdout: tuple[int, ...] = (),
    method: str = "exact",
    seed: int | None = None,
    sample_count: int | None = None,
    cache=None,
    **limits,
):
    preparation, config, distribution, step_time = _case(
        deterministic=deterministic
    )
    return RPEHadamardCompiledCostBenchmarkRequest(
        preparation=preparation,
        delta_time=step_time,
        calibration_repetition_counts=calibration,
        holdout_repetition_counts=holdout,
        rte_steps_per_occurrence=0 if deterministic else config.rte_steps,
        finite_taylor_order=0 if deterministic else config.finite_taylor_order,
        rte_config=config,
        rte_distribution=distribution,
        compiler=_compiler(),
        evaluation_method=method,
        sample_count=sample_count,
        seed=seed,
        generation_id="test-run",
        maximum_repetition_count=32,
        cache=cache,
        **limits,
    )


def _one_evolution(*, deterministic: bool, repetition_count: int):
    preparation, config, distribution, step_time = _case(
        deterministic=deterministic
    )
    stream = make_exact_df_partial_s2_repeated_trajectory_stream(
        preparation,
        step_time,
        repetition_count,
        config,
        distribution,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        maximum_trajectories=1 if deterministic else 1_000,
    )
    request, _weight = next(iter(stream.records))
    return (
        QiskitDFPartialS2RepeatedCircuitBuilder().build(request),
        step_time,
    )


def _single_event_randomized_exact_request():
    preparation, _config, _distribution, step_time = _case(deterministic=False)
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=step_time,
        rte_steps=1,
        truncation_tolerance=10.0,
        finite_taylor_order=0,
    )
    return RPEHadamardCompiledCostBenchmarkRequest(
        preparation=preparation,
        delta_time=step_time,
        calibration_repetition_counts=(8,),
        holdout_repetition_counts=(),
        rte_steps_per_occurrence=1,
        finite_taylor_order=0,
        rte_config=config,
        rte_distribution=distribution,
        compiler=_compiler(),
        evaluation_method="exact",
        generation_id="single-event-exact",
        maximum_repetition_count=8,
        maximum_trajectories=1,
    )


@pytest.mark.parametrize(("q_m", "m"), ((1, 0), (2, 1), (4, 2), (8, 3), (32, 5)))
def test_benchmark_round_index_accepts_bounded_powers_of_two(q_m, m) -> None:
    assert (
        round_index_for_benchmark_repetition_count(
            q_m,
            maximum_repetition_count=32,
        )
        == m
    )


@pytest.mark.parametrize("q_m", (0, 3, 6, 12))
def test_benchmark_round_index_rejects_non_positive_or_non_power_of_two(q_m) -> None:
    with pytest.raises(ValueError):
        round_index_for_benchmark_repetition_count(
            q_m,
            maximum_repetition_count=32,
        )


def test_benchmark_builder_supports_q8_while_short_builder_rejects_it() -> None:
    evolution, step_time = _one_evolution(
        deterministic=True,
        repetition_count=8,
    )
    request = RPEHadamardInterrogationRequest(
        evolution=evolution,
        axis="sine",
        include_measurement=True,
    )
    with pytest.raises(ValueError, match="1, 2, 4"):
        QiskitRPEHadamardInterrogationBuilder().build(request)

    result = QiskitRPEHadamardBenchmarkCircuitBuilder(
        maximum_repetition_count=8
    ).build(request)
    assert result.round_index == 3
    assert result.t_m == pytest.approx(8 * step_time)
    assert result.include_measurement
    assert result.circuit.num_clbits == 1
    assert not result.state_preparation_included
    assert result.wrapped_evolution_already_controlled
    assert not result.additional_control_applied


def test_benchmark_builder_checks_cap_before_shared_build(monkeypatch) -> None:
    evolution, _step_time = _one_evolution(
        deterministic=True,
        repetition_count=8,
    )
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("shared builder must not run")

    monkeypatch.setattr(
        QiskitRPEHadamardInterrogationBuilder,
        "_build_validated",
        forbidden,
    )
    with pytest.raises(ValueError, match="maximum_repetition_count"):
        QiskitRPEHadamardBenchmarkCircuitBuilder(
            maximum_repetition_count=4
        ).build(
            RPEHadamardInterrogationRequest(
                evolution=evolution,
                axis="cosine",
            )
        )
    assert not called


def test_deterministic_medium_q_dataset_records_scope_and_phases() -> None:
    result = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(deterministic=True, calibration=(8,), holdout=(16,))
    )
    dataset = result.dataset
    assert dataset.complete
    assert dataset.requested_point_count == dataset.completed_point_count == 4
    assert dataset.failed_point_count == 0
    assert not dataset.proxy_fit_performed
    assert not dataset.holdout_used_for_proxy_fit
    for point in dataset.records:
        assert point.round_index == int(math.log2(point.q_m))
        assert point.t_m == pytest.approx(point.q_m * point.delta_time)
        assert point.tail_kind == "deterministic"
        assert point.evaluation_method == "exact"
        assert point.measurement_included
        assert not point.state_preparation_included
        assert not point.backend_execution_included
        assert point.quantum_shots_executed == 0
        assert point.benchmark_validation_path
        assert point.circuit_scope == (
            "single_hadamard_interrogation_without_state_preparation"
        )
        trajectory = point.retained_trajectory_records[0]
        assert trajectory.constant_phase != 0.0
        assert trajectory.extracted_identity_phase == pytest.approx(-0.0)
        assert trajectory.rte_relative_phase == pytest.approx(0.0)


def test_randomized_exact_is_probability_weighted_without_mc_fallback() -> None:
    result = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(deterministic=False, calibration=(1,))
    )
    assert result.dataset.complete
    for point in result.dataset.records:
        assert point.tail_kind == "randomized"
        assert point.evaluation_method == "exact"
        assert point.trajectory_probability_sum == pytest.approx(1.0)
        for metric, summary in point.metric_statistics:
            weighted_mean = sum(
                record.probability * getattr(record.cost, metric)
                for record in point.retained_trajectory_records
            )
            assert summary.mean == pytest.approx(weighted_mean)

    capped = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(
            deterministic=False,
            calibration=(8,),
            maximum_trajectories=8,
        )
    ).dataset
    assert not capped.complete
    assert capped.failed_point_count == 2
    assert all(point.evaluation_method == "exact" for point in capped.records)
    assert all(
        "maximum_trajectories" in point.failure_reason
        for point in capped.records
    )


def test_randomized_exact_medium_q_runs_when_trajectory_space_is_within_cap() -> None:
    result = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _single_event_randomized_exact_request()
    )
    assert result.dataset.complete
    assert result.estimates[0].q_m == 8
    assert result.estimates[0].trajectory_space_size == 1
    assert all(point.tail_kind == "randomized" for point in result.dataset.records)
    assert all(point.evaluation_method == "exact" for point in result.dataset.records)


class _CountingCache(TranspiledCircuitCostCache):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def get_or_transpile(self, *args, **kwargs):
        self.call_count += 1
        return super().get_or_transpile(*args, **kwargs)


def test_mc_statistics_and_shared_axis_trajectory_set_are_recomputable() -> None:
    cache = _CountingCache()
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(
            deterministic=False,
            calibration=(8,),
            method="monte_carlo",
            sample_count=4,
            seed=123,
            cache=cache,
        )
    ).dataset
    assert dataset.complete
    assert cache.call_count == 8
    cosine = next(point for point in dataset.records if point.axis == "cosine")
    sine = next(point for point in dataset.records if point.axis == "sine")
    assert cosine.sampled_trajectory_seeds == sine.sampled_trajectory_seeds
    assert cosine.step_seed_hierarchy == sine.step_seed_hierarchy
    assert [
        record.evolution_provenance_fingerprint
        for record in cosine.retained_trajectory_records
    ] == [
        record.evolution_provenance_fingerprint
        for record in sine.retained_trajectory_records
    ]
    for point in (cosine, sine):
        for metric, summary in point.metric_statistics:
            values = [
                getattr(record.cost, metric)
                for record in point.retained_trajectory_records
            ]
            assert summary.mean == pytest.approx(statistics.fmean(values))
            assert summary.unbiased_sample_variance == pytest.approx(
                statistics.variance(values)
            )
            assert summary.standard_error == pytest.approx(
                math.sqrt(statistics.variance(values) / len(values))
            )


def test_mc_seed_reproducibility_and_partition_seed_separation() -> None:
    keywords = dict(
        deterministic=False,
        calibration=(8,),
        holdout=(16,),
        method="monte_carlo",
        sample_count=3,
    )
    first = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(**keywords, seed=101)
    ).dataset
    second = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(**keywords, seed=101)
    ).dataset
    changed = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(**keywords, seed=102)
    ).dataset
    assert first == second
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert first.dataset_fingerprint != changed.dataset_fingerprint
    calibration_seeds = {seed for _q, seed in first.calibration_mc_seeds}
    holdout_seeds = {seed for _q, seed in first.holdout_mc_seeds}
    assert calibration_seeds.isdisjoint(holdout_seeds)
    assert {
        point.sampling_provenance_fingerprint for point in first.records
    } != {
        point.sampling_provenance_fingerprint for point in changed.records
    }


def test_retention_limit_keeps_full_stream_digests() -> None:
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(
            deterministic=False,
            calibration=(8,),
            method="monte_carlo",
            sample_count=3,
            seed=101,
            maximum_retained_trajectory_records=0,
        )
    ).dataset
    for point in dataset.records:
        assert point.trajectory_records_truncated
        assert not point.retained_trajectory_records
        assert point.trajectory_provenance_digest
        assert point.evolution_circuit_semantics_digest
        assert point.wrapper_circuit_semantics_digest
        assert point.actual_circuit_fingerprint_digest


def test_partition_overlap_is_rejected_and_partition_changes_fingerprint() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        _request(deterministic=True, calibration=(8,), holdout=(8,))

    calibration = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(deterministic=True, calibration=(8,), holdout=())
    ).dataset
    holdout = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(deterministic=True, calibration=(), holdout=(8,))
    ).dataset
    assert calibration.dataset_fingerprint != holdout.dataset_fingerprint
    assert {point.partition for point in calibration.records} == {"calibration"}
    assert {point.partition for point in holdout.records} == {"holdout"}


def test_axis_and_compiler_context_change_fingerprints() -> None:
    request = _request(deterministic=True, calibration=(8,))
    first = generate_rpe_hadamard_compiled_cost_benchmark_dataset(request).dataset
    changed = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        replace(request, compiler=_compiler(seed=18))
    ).dataset
    assert len({point.point_fingerprint for point in first.records}) == 2
    assert first.dataset_fingerprint != changed.dataset_fingerprint
    assert first.compiler_settings_fingerprint != (
        changed.compiler_settings_fingerprint
    )


def test_record_order_is_canonical_and_json_round_trip_is_exact(tmp_path) -> None:
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(deterministic=True, calibration=(8,), holdout=(16,))
    ).dataset
    reordered = replace(
        dataset,
        records=tuple(reversed(dataset.records)),
        dataset_fingerprint="",
    )
    assert reordered == dataset
    assert reordered.dataset_fingerprint == dataset.dataset_fingerprint

    output = tmp_path / "benchmark.json"
    dataset.write_json(output)
    loaded = RPEHadamardCompiledCostBenchmarkDataset.read_json(output)
    assert loaded == dataset
    assert loaded.dataset_fingerprint == dataset.dataset_fingerprint
    assert loaded.to_dict() == dataset.to_dict()


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    (
        ("m", 99, "m must equal round_index"),
        ("status", "unknown", "point status"),
        ("tail_kind", "unknown", "tail kind"),
        ("benchmark_validation_path", False, "must be true"),
    ),
)
def test_tampered_point_json_is_rejected(field, invalid_value, message) -> None:
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(deterministic=True, calibration=(8,))
    ).dataset
    payload = deepcopy(dataset.to_dict())
    payload["records"][0][field] = invalid_value
    with pytest.raises((TypeError, ValueError), match=message):
        RPEHadamardCompiledCostBenchmarkDataset.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    (
        (
            "schema_version",
            "rpe_hadamard_compiled_cost_benchmark_dataset_v1",
            "schema",
        ),
        ("product_formula_order", 4, "second-order"),
        ("cost_metrics", ["bogus"], "cost_metrics"),
        ("measurement_policy", "bogus", "measurement policy"),
    ),
)
def test_tampered_fixed_dataset_convention_is_rejected(
    field,
    invalid_value,
    message,
) -> None:
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(deterministic=True, calibration=(8,))
    ).dataset
    payload = deepcopy(dataset.to_dict())
    payload[field] = invalid_value
    with pytest.raises((TypeError, ValueError), match=message):
        RPEHadamardCompiledCostBenchmarkDataset.from_dict(payload)


def test_tampered_incomplete_reasons_are_rejected() -> None:
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(
            deterministic=True,
            calibration=(8,),
            maximum_build_requests=1,
        )
    ).dataset
    payload = deepcopy(dataset.to_dict())
    payload["incomplete_reasons"] = ["forged reason"]
    with pytest.raises(ValueError, match="incomplete_reasons"):
        RPEHadamardCompiledCostBenchmarkDataset.from_dict(payload)


def test_programmatic_invalid_audit_values_are_rejected() -> None:
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(deterministic=True, calibration=(8,))
    ).dataset
    point = dataset.records[0]
    with pytest.raises(ValueError, match="point status"):
        replace(point, status="unknown", point_fingerprint="")
    with pytest.raises(ValueError, match="tail kind"):
        replace(point, tail_kind="unknown", point_fingerprint="")
    with pytest.raises(ValueError, match="must be true"):
        replace(point, benchmark_validation_path=False, point_fingerprint="")
    with pytest.raises(ValueError, match="second-order"):
        replace(dataset, product_formula_order=4, dataset_fingerprint="")


def test_full_wrapper_direct_transpile_matches_medium_q_record() -> None:
    request = _request(deterministic=True, calibration=(8,))
    generated = generate_rpe_hadamard_compiled_cost_benchmark_dataset(request)
    cosine_point = next(
        point for point in generated.dataset.records if point.axis == "cosine"
    )
    evolution, _step_time = _one_evolution(
        deterministic=True,
        repetition_count=8,
    )
    wrapper = QiskitRPEHadamardBenchmarkCircuitBuilder(
        maximum_repetition_count=8
    ).build(
        RPEHadamardInterrogationRequest(
            evolution=evolution,
            axis="cosine",
            include_measurement=True,
        )
    )
    direct = transpile_and_measure_cost(
        wrapper.circuit,
        request.compiler,
        circuit_fingerprint=wrapper.compiler_independent_fingerprint,
    )
    retained = cosine_point.retained_trajectory_records[0]
    for metric in METRICS:
        assert getattr(retained.cost, metric) == getattr(direct, metric)
    assert retained.actual_circuit_fingerprint == direct.actual_circuit_fingerprint


def test_workload_failure_happens_before_circuit_build(monkeypatch) -> None:
    build_count = 0
    original = cost_module.QiskitDFPartialS2RepeatedCircuitBuilder.build

    def counted(self, *args, **kwargs):
        nonlocal build_count
        build_count += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        cost_module.QiskitDFPartialS2RepeatedCircuitBuilder,
        "build",
        counted,
    )
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(
            deterministic=True,
            calibration=(8,),
            maximum_build_requests=1,
        )
    ).dataset
    assert build_count == 0
    assert not dataset.complete
    assert dataset.failed_point_count == 2
    assert all("build requests" in point.failure_reason for point in dataset.records)
    for point in dataset.records:
        assert point.failure_stage == "preflight"
        assert point.requested_measurement_included
        assert not point.requested_state_preparation_included
        assert point.requested_control_convention == (
            "ordinary_controlled_diag_I_U_m"
        )
        assert point.measurement_included is None
        assert point.state_preparation_included is None
        assert point.wrapped_evolution_already_controlled is None
        assert point.additional_control_applied is None
        assert point.circuit_build_completed is False
        assert point.transpile_completed is False
        assert point.actual_build_requests == 0
        assert point.actual_transpile_requests == 0
        assert point.actual_built_instruction_total == 0


class _FailOnceCache(TranspiledCircuitCostCache):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def get_or_transpile(self, *args, **kwargs):
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected compile failure")
        return super().get_or_transpile(*args, **kwargs)


def test_compile_failure_is_recorded_in_partial_dataset(tmp_path) -> None:
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(
            deterministic=True,
            calibration=(8,),
            holdout=(16,),
            cache=_FailOnceCache(),
        )
    ).dataset
    assert not dataset.complete
    assert dataset.completed_point_count == 2
    assert dataset.failed_point_count == 2
    failures = [point for point in dataset.records if point.status == "failed"]
    assert len(failures) == 2
    assert all("injected compile failure" in point.failure_reason for point in failures)
    for point in failures:
        assert point.failure_stage == "circuit_build_or_transpile"
        assert point.measurement_included is None
        assert point.wrapped_evolution_already_controlled is None
        assert point.circuit_build_completed is None
        assert point.transpile_completed is None
        assert point.actual_build_requests is None
        assert point.actual_transpile_requests is None
    output = tmp_path / "partial.json"
    dataset.write_json(output)
    assert RPEHadamardCompiledCostBenchmarkDataset.read_json(output) == dataset


def test_short_provider_remains_capped_and_no_proxy_is_connected() -> None:
    with pytest.raises(ValueError, match="must not exceed 4"):
        DFRPEHadamardCompiledCostProvider(
            compiler=_compiler(),
            maximum_repetition_count=8,
        )
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _request(deterministic=True, calibration=(8,))
    ).dataset
    assert dataset.control_convention == "ordinary_controlled_diag_I_U_m"
    assert not dataset.proxy_fit_performed
    assert not dataset.holdout_used_for_proxy_fit
