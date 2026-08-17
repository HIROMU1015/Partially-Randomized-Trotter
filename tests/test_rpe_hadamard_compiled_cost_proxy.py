from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.df_rpe_hadamard_compiled_cost import (
    DFRPEHadamardCompiledCostProvider,
)
from trotterlib.rpe_hadamard_compiled_cost_benchmark import (
    RPEHadamardCompiledCostBenchmarkDataset,
    RPEHadamardCompiledCostBenchmarkRequest,
    generate_rpe_hadamard_compiled_cost_benchmark_dataset,
)
from trotterlib.rpe_hadamard_compiled_cost_proxy import (
    RPEHadamardCompiledCostProxy,
    RPEHadamardCompiledCostProxyFitRequest,
    RPEHadamardCompiledCostProxyValidationRequest,
    RPEHadamardCompiledCostProxyValidationResult,
    RPEHadamardProxyMetricTolerance,
    fit_rpe_hadamard_compiled_cost_proxy,
    validate_rpe_hadamard_compiled_cost_proxy,
)
from trotterlib.rpe_resource_accounting import RPE_COST_METRICS
from trotterlib.rte import CompilerSettings, make_rte_config
from trotterlib.rte_compiled_cost import CompiledMetricStatistics


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


def _benchmark_request(
    *,
    calibration: tuple[int, ...] = (1, 2, 4),
    holdout: tuple[int, ...] = (8, 16),
    compiler: CompilerSettings | None = None,
    maximum_build_requests: int = 1_000_000,
) -> RPEHadamardCompiledCostBenchmarkRequest:
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "large-q-proxy-tests"},
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
        identity_policy="extract_identity_phase",
    )
    return RPEHadamardCompiledCostBenchmarkRequest(
        preparation=preparation,
        delta_time=0.2,
        calibration_repetition_counts=calibration,
        holdout_repetition_counts=holdout,
        rte_steps_per_occurrence=0,
        finite_taylor_order=0,
        rte_config=None,
        rte_distribution=None,
        compiler=_compiler() if compiler is None else compiler,
        evaluation_method="exact",
        generation_id="proxy-test",
        maximum_repetition_count=32,
        maximum_build_requests=maximum_build_requests,
    )


@pytest.fixture(scope="module")
def base_dataset() -> RPEHadamardCompiledCostBenchmarkDataset:
    return generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _benchmark_request()
    ).dataset


@pytest.fixture(scope="module")
def monte_carlo_dataset() -> RPEHadamardCompiledCostBenchmarkDataset:
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "large-q-proxy-monte-carlo-tests"},
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 0),
        identity_policy="extract_identity_phase",
    )
    delta_time = 1.2 / preparation.exact_rte_lambda_r
    rte_config, rte_distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=delta_time,
        rte_steps=1,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    request = RPEHadamardCompiledCostBenchmarkRequest(
        preparation=preparation,
        delta_time=delta_time,
        calibration_repetition_counts=(1, 2, 4),
        holdout_repetition_counts=(8,),
        rte_steps_per_occurrence=rte_config.rte_steps,
        finite_taylor_order=rte_config.finite_taylor_order,
        rte_config=rte_config,
        rte_distribution=rte_distribution,
        compiler=_compiler(),
        evaluation_method="monte_carlo",
        sample_count=3,
        seed=123,
        generation_id="proxy-monte-carlo-test",
        maximum_repetition_count=8,
    )
    return generate_rpe_hadamard_compiled_cost_benchmark_dataset(request).dataset


def _affine_coefficients(axis: str, metric: str) -> tuple[float, float]:
    metric_index = RPE_COST_METRICS.index(metric)
    axis_offset = 10 if axis == "sine" else 0
    return float(axis_offset + metric_index + 1), float(20 + 3 * metric_index)


def _affine_mean(partition: str, axis: str, metric: str, q_m: int) -> float:
    del partition
    slope, intercept = _affine_coefficients(axis, metric)
    return slope * q_m + intercept


def _with_statistics(
    dataset: RPEHadamardCompiledCostBenchmarkDataset,
    *,
    mean_fn,
    standard_error_fn=lambda _partition, _axis, _metric, _q_m: None,
) -> RPEHadamardCompiledCostBenchmarkDataset:
    records = []
    for point in dataset.records:
        statistics = []
        for metric in RPE_COST_METRICS:
            mean = float(mean_fn(point.partition, point.axis, metric, point.q_m))
            standard_error = standard_error_fn(
                point.partition,
                point.axis,
                metric,
                point.q_m,
            )
            statistics.append(
                (
                    metric,
                    CompiledMetricStatistics(
                        mean=mean,
                        unbiased_sample_variance=(
                            None
                            if standard_error is None
                            else float(standard_error) ** 2
                        ),
                        standard_error=standard_error,
                        minimum=mean,
                        maximum=mean,
                    ),
                )
            )
        records.append(
            replace(
                point,
                metric_statistics=tuple(statistics),
                point_fingerprint="",
            )
        )
    return replace(dataset, records=tuple(records), dataset_fingerprint="")


def _fit(
    dataset: RPEHadamardCompiledCostBenchmarkDataset,
    *,
    weighting: str = "uniform",
) -> RPEHadamardCompiledCostProxy:
    return fit_rpe_hadamard_compiled_cost_proxy(
        RPEHadamardCompiledCostProxyFitRequest(
            dataset=dataset,
            weighting=weighting,
        )
    )


def _tolerances(
    absolute: float,
    relative: float = 0.0,
    standard_error_multiplier: float = 0.0,
) -> tuple[RPEHadamardProxyMetricTolerance, ...]:
    return tuple(
        RPEHadamardProxyMetricTolerance(
            metric=metric,
            absolute_tolerance=absolute,
            relative_tolerance=relative,
            standard_error_multiplier=standard_error_multiplier,
        )
        for metric in RPE_COST_METRICS
    )


def _validate(
    proxy: RPEHadamardCompiledCostProxy,
    dataset: RPEHadamardCompiledCostBenchmarkDataset,
    tolerances: tuple[RPEHadamardProxyMetricTolerance, ...],
) -> RPEHadamardCompiledCostProxyValidationResult:
    return validate_rpe_hadamard_compiled_cost_proxy(
        RPEHadamardCompiledCostProxyValidationRequest(
            proxy=proxy,
            dataset=dataset,
            metric_tolerances=tolerances,
        )
    )


def test_known_affine_data_recovers_axis_metric_coefficients(base_dataset) -> None:
    dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)
    proxy = _fit(dataset)
    assert len(proxy.models) == 2 * len(RPE_COST_METRICS)
    for axis in ("cosine", "sine"):
        for metric in RPE_COST_METRICS:
            slope, intercept = _affine_coefficients(axis, metric)
            model = proxy.model(axis, metric)
            assert model.slope == pytest.approx(slope)
            assert model.intercept == pytest.approx(intercept)
            assert model.root_mean_square_residual == pytest.approx(0.0)
            assert model.maximum_absolute_residual == pytest.approx(0.0)


def test_uniform_fit_matches_manual_regression(base_dataset) -> None:
    observed = {1: 2.0, 2: 2.0, 4: 8.0}
    dataset = _with_statistics(
        base_dataset,
        mean_fn=lambda partition, axis, metric, q_m: (
            observed[q_m] if partition == "calibration" else 17.0
        ),
    )
    model = _fit(dataset, weighting="uniform").model("cosine", "rz_count")
    assert model.slope == pytest.approx(15.0 / 7.0)
    assert model.intercept == pytest.approx(-1.0)


def test_inverse_variance_fit_matches_manual_regression(monte_carlo_dataset) -> None:
    observed = {1: 2.0, 2: 2.0, 4: 8.0}
    standard_errors = {1: 1.0, 2: 0.5, 4: 1.0}
    dataset = _with_statistics(
        monte_carlo_dataset,
        mean_fn=lambda partition, axis, metric, q_m: (
            observed[q_m] if partition == "calibration" else 17.0
        ),
        standard_error_fn=lambda partition, axis, metric, q_m: (
            standard_errors[q_m] if partition == "calibration" else 1.0
        ),
    )
    proxy = _fit(dataset, weighting="inverse_variance")
    model = proxy.model("cosine", "rz_count")
    assert model.slope == pytest.approx(66.0 / 29.0)
    assert model.intercept == pytest.approx(-56.0 / 29.0)
    assert proxy.fit_specification_fingerprint != _fit(
        dataset,
        weighting="uniform",
    ).fit_specification_fingerprint


@pytest.mark.parametrize("standard_error", (None, 0.0))
def test_inverse_variance_rejects_missing_or_nonpositive_se(
    monte_carlo_dataset,
    standard_error,
) -> None:
    dataset = _with_statistics(
        monte_carlo_dataset,
        mean_fn=_affine_mean,
        standard_error_fn=lambda partition, axis, metric, q_m: standard_error,
    )
    with pytest.raises(ValueError, match="finite positive standard errors"):
        _fit(dataset, weighting="inverse_variance")


def test_fit_uses_only_calibration_and_holdout_changes_do_not_change_fit(
    base_dataset,
) -> None:
    first_dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)

    def changed_holdout(partition, axis, metric, q_m):
        value = _affine_mean(partition, axis, metric, q_m)
        return value if partition == "calibration" else value + 10_000.0

    second_dataset = _with_statistics(base_dataset, mean_fn=changed_holdout)
    first = _fit(first_dataset)
    second = _fit(second_dataset)
    assert first.models == second.models
    assert first.calibration_subset_fingerprint == (
        second.calibration_subset_fingerprint
    )
    assert first.fit_fingerprint == second.fit_fingerprint
    assert first.source_dataset_fingerprint != second.source_dataset_fingerprint
    assert first.proxy_fingerprint != second.proxy_fingerprint
    assert not first.holdout_used_for_fit


def test_calibration_change_changes_fit_fingerprint(base_dataset) -> None:
    first_dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)

    def changed_calibration(partition, axis, metric, q_m):
        value = _affine_mean(partition, axis, metric, q_m)
        if partition == "calibration" and axis == "cosine" and q_m == 2:
            return value + 1.0
        return value

    second_dataset = _with_statistics(base_dataset, mean_fn=changed_calibration)
    assert _fit(first_dataset).fit_fingerprint != _fit(
        second_dataset
    ).fit_fingerprint


def test_fit_rejects_one_calibration_q_m() -> None:
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _benchmark_request(calibration=(2,), holdout=(4,))
    ).dataset
    with pytest.raises(ValueError, match="at least two calibration"):
        _fit(dataset)


def test_fit_rejects_incomplete_dataset() -> None:
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _benchmark_request(maximum_build_requests=1)
    ).dataset
    assert not dataset.complete
    with pytest.raises(ValueError, match="complete benchmark dataset"):
        _fit(dataset)


@pytest.mark.parametrize("q_m", (0, -2, 3, 6))
def test_prediction_requires_positive_power_of_two(base_dataset, q_m) -> None:
    proxy = _fit(_with_statistics(base_dataset, mean_fn=_affine_mean))
    with pytest.raises(ValueError, match="positive power of two"):
        proxy.predict(q_m, axis="cosine", metric="rz_count")


def test_validation_requires_holdout() -> None:
    dataset = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _benchmark_request(holdout=())
    ).dataset
    proxy = _fit(dataset)
    with pytest.raises(ValueError, match="at least one holdout"):
        _validate(proxy, dataset, _tolerances(1.0))


def test_validation_does_not_refit_or_mutate_inputs(base_dataset) -> None:
    dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)
    dataset_before = deepcopy(dataset.to_dict())
    proxy = _fit(dataset)
    proxy_before = deepcopy(proxy.to_dict())
    result = _validate(proxy, dataset, _tolerances(1.0e-9))
    assert result.overall_pass
    assert not result.validation_refit_performed
    assert proxy.to_dict() == proxy_before
    assert dataset.to_dict() == dataset_before
    assert not dataset.proxy_fit_performed
    assert not dataset.holdout_used_for_proxy_fit


def test_validation_errors_and_holdout_fingerprints_are_recorded(
    base_dataset,
) -> None:
    def shifted_holdout(partition, axis, metric, q_m):
        value = _affine_mean(partition, axis, metric, q_m)
        return value if partition == "calibration" else value - 2.0

    dataset = _with_statistics(base_dataset, mean_fn=shifted_holdout)
    proxy = _fit(dataset)
    result = _validate(proxy, dataset, _tolerances(2.0))
    entry = next(
        item
        for item in result.entries
        if item.q_m == 8 and item.axis == "cosine" and item.metric == "rz_count"
    )
    assert entry.predicted_cost == pytest.approx(entry.observed_mean + 2.0)
    assert entry.signed_error == pytest.approx(2.0)
    assert entry.absolute_error == pytest.approx(2.0)
    assert entry.relative_error == pytest.approx(2.0 / entry.observed_mean)
    assert entry.point_fingerprint in {
        point.point_fingerprint for point in result.holdout_points
    }
    assert entry.outside_calibration_range
    assert entry.passed


def test_zero_observed_mean_has_no_relative_error(base_dataset) -> None:
    def zero_holdout(partition, axis, metric, q_m):
        if partition == "holdout" and axis == "cosine" and metric == "rz_count":
            return 0.0
        return _affine_mean(partition, axis, metric, q_m)

    dataset = _with_statistics(base_dataset, mean_fn=zero_holdout)
    result = _validate(_fit(dataset), dataset, _tolerances(1_000.0))
    entry = next(
        item
        for item in result.entries
        if item.axis == "cosine" and item.metric == "rz_count"
    )
    assert entry.observed_mean == 0.0
    assert entry.relative_error is None


def test_acceptance_formula_uses_absolute_relative_and_se_terms(
    monte_carlo_dataset,
) -> None:
    def shifted_holdout(partition, axis, metric, q_m):
        value = _affine_mean(partition, axis, metric, q_m)
        return value if partition == "calibration" else value - 5.0

    dataset = _with_statistics(
        monte_carlo_dataset,
        mean_fn=shifted_holdout,
        standard_error_fn=lambda partition, axis, metric, q_m: (
            None if partition == "calibration" else 0.5
        ),
    )
    result = _validate(
        _fit(dataset),
        dataset,
        _tolerances(absolute=1.0, relative=0.1, standard_error_multiplier=2.0),
    )
    entry = result.entries[0]
    expected_limit = 1.0 + 0.1 * abs(entry.observed_mean) + 2.0 * 0.5
    assert entry.acceptance_limit == pytest.approx(expected_limit)
    assert entry.passed == (entry.absolute_error <= expected_limit)


def test_exact_evaluation_without_se_uses_zero_se_term(base_dataset) -> None:
    dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)
    result = _validate(
        _fit(dataset),
        dataset,
        _tolerances(absolute=1.25, standard_error_multiplier=99.0),
    )
    assert all(entry.observed_standard_error is None for entry in result.entries)
    assert all(entry.acceptance_limit == 1.25 for entry in result.entries)


def test_negative_prediction_fails_even_with_large_tolerance(base_dataset) -> None:
    def negative_extrapolation(partition, axis, metric, q_m):
        if axis == "cosine" and metric == "rz_count":
            return max(0.0, 6.0 - q_m)
        return _affine_mean(partition, axis, metric, q_m)

    dataset = _with_statistics(base_dataset, mean_fn=negative_extrapolation)
    result = _validate(_fit(dataset), dataset, _tolerances(1_000_000.0))
    entries = [
        entry
        for entry in result.entries
        if entry.axis == "cosine" and entry.metric == "rz_count"
    ]
    assert any(not entry.prediction_is_nonnegative for entry in entries)
    assert all(not entry.passed for entry in entries)
    assert not result.overall_pass


def test_nonfinite_prediction_is_recorded_and_fails(
    base_dataset,
    monkeypatch,
) -> None:
    dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)
    proxy = _fit(dataset)

    def nonfinite_predict(self, q_m, *, axis, metric):
        del self, q_m, axis, metric
        return math.inf

    monkeypatch.setattr(RPEHadamardCompiledCostProxy, "predict", nonfinite_predict)
    result = _validate(proxy, dataset, _tolerances(1_000_000.0))
    assert not result.overall_pass
    assert all(not entry.prediction_is_finite for entry in result.entries)
    assert all(entry.predicted_cost is None for entry in result.entries)
    assert all(
        entry.prediction_nonfinite_kind == "positive_infinity"
        for entry in result.entries
    )


def test_overall_pass_requires_every_holdout_entry(base_dataset) -> None:
    dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)
    passing = _validate(_fit(dataset), dataset, _tolerances(1.0e-9))
    assert passing.overall_pass
    assert passing.failed_entry_count == 0

    def one_failure(partition, axis, metric, q_m):
        value = _affine_mean(partition, axis, metric, q_m)
        if partition == "holdout" and axis == "sine" and metric == "cx_depth":
            return value + 1.0
        return value

    failing_dataset = _with_statistics(base_dataset, mean_fn=one_failure)
    failing = _validate(_fit(failing_dataset), failing_dataset, _tolerances(0.0))
    assert not failing.overall_pass
    assert failing.failed_entry_count >= 1


def test_validation_records_only_tested_range_without_broader_guarantee(
    base_dataset,
) -> None:
    dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)
    result = _validate(_fit(dataset), dataset, _tolerances(0.0))
    assert result.validated_q_m_values == (8, 16)
    assert result.validated_q_m_min == 8
    assert result.validated_q_m_max == 16
    assert not result.accuracy_guaranteed_beyond_validated_range
    assert result.accuracy_claim_scope == "validated_holdout_q_m_range_only"


def test_proxy_and_validation_json_round_trip(base_dataset, tmp_path) -> None:
    dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)
    proxy = _fit(dataset)
    result = _validate(proxy, dataset, _tolerances(0.0))
    proxy_path = tmp_path / "proxy.json"
    validation_path = tmp_path / "validation.json"
    proxy.write_json(proxy_path)
    result.write_json(validation_path)
    assert RPEHadamardCompiledCostProxy.read_json(proxy_path) == proxy
    assert RPEHadamardCompiledCostProxyValidationResult.read_json(
        validation_path
    ) == result


@pytest.mark.parametrize("target", ("fit_fingerprint", "proxy_fingerprint"))
@pytest.mark.parametrize("mutation", ("empty", "missing", "modified"))
def test_proxy_json_rejects_invalid_fingerprints(
    base_dataset,
    target,
    mutation,
) -> None:
    proxy = _fit(_with_statistics(base_dataset, mean_fn=_affine_mean))
    payload = deepcopy(proxy.to_dict())
    if mutation == "empty":
        payload[target] = ""
    elif mutation == "missing":
        del payload[target]
    else:
        payload[target] = "0" * 64
    with pytest.raises((TypeError, ValueError), match="fingerprint"):
        RPEHadamardCompiledCostProxy.from_dict(payload)


@pytest.mark.parametrize(
    "target",
    (
        "holdout_subset_fingerprint",
        "acceptance_policy_fingerprint",
        "validation_fingerprint",
    ),
)
@pytest.mark.parametrize("mutation", ("empty", "missing", "modified"))
def test_validation_json_rejects_invalid_fingerprints(
    base_dataset,
    target,
    mutation,
) -> None:
    dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)
    result = _validate(_fit(dataset), dataset, _tolerances(0.0))
    payload = deepcopy(result.to_dict())
    if mutation == "empty":
        payload[target] = ""
    elif mutation == "missing":
        del payload[target]
    else:
        payload[target] = "0" * 64
    with pytest.raises((TypeError, ValueError), match="fingerprint"):
        RPEHadamardCompiledCostProxyValidationResult.from_dict(payload)


def test_json_rejects_redundant_aggregate_tampering(base_dataset) -> None:
    dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)
    proxy = _fit(dataset)
    proxy_payload = deepcopy(proxy.to_dict())
    proxy_payload["model_count"] = 999
    with pytest.raises(ValueError, match="model_count"):
        RPEHadamardCompiledCostProxy.from_dict(proxy_payload)

    result = _validate(proxy, dataset, _tolerances(0.0))
    validation_payload = deepcopy(result.to_dict())
    validation_payload["overall_pass"] = not result.overall_pass
    with pytest.raises(ValueError, match="overall_pass"):
        RPEHadamardCompiledCostProxyValidationResult.from_dict(validation_payload)


@pytest.mark.parametrize("invalid", (-1.0, math.inf, math.nan))
def test_validation_tolerances_must_be_finite_and_nonnegative(invalid) -> None:
    with pytest.raises(ValueError, match="finite|non-negative"):
        RPEHadamardProxyMetricTolerance(
            metric="rz_count",
            absolute_tolerance=invalid,
            relative_tolerance=0.0,
            standard_error_multiplier=0.0,
        )


def test_validation_rejects_incompatible_compiler_context(base_dataset) -> None:
    dataset = _with_statistics(base_dataset, mean_fn=_affine_mean)
    proxy = _fit(dataset)
    changed = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        _benchmark_request(compiler=_compiler(seed=18))
    ).dataset
    changed = _with_statistics(changed, mean_fn=_affine_mean)
    with pytest.raises(ValueError, match="compiler_settings_fingerprint"):
        _validate(proxy, changed, _tolerances(0.0))


def test_proxy_remains_disconnected_from_short_provider_and_resources(
    base_dataset,
) -> None:
    proxy = _fit(_with_statistics(base_dataset, mean_fn=_affine_mean))
    assert not proxy.resource_accounting_connected
    assert not proxy.state_preparation_included
    assert not proxy.backend_execution_included
    assert proxy.quantum_shots_executed == 0
    with pytest.raises(ValueError, match="must not exceed 4"):
        DFRPEHadamardCompiledCostProvider(
            compiler=_compiler(),
            maximum_repetition_count=8,
        )
