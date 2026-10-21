"""Calibration-only affine proxies for RPE Hadamard compiled costs.

The proxy covers one measured Hadamard interrogation without state preparation.
It is deliberately disconnected from short-round providers and resource accounting.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias

from .df_partial_s2_repeated import RepeatedCircuitConstructionPolicy
from .rpe_hadamard_compiled_cost_benchmark import (
    RPE_HADAMARD_COMPILED_COST_BENCHMARK_SCHEMA_VERSION,
    RPE_HADAMARD_CONTROL_CONVENTION,
    RPEHadamardCompiledCostBenchmarkDataset,
)
from .rpe_hadamard_interrogation import (
    RPE_HADAMARD_INTERROGATION_SCOPE,
    RPEHadamardAxis,
)
from .rpe_resource_accounting import RPE_COST_METRICS, RPECostMetric
from .rte import require_integer_count


RPEHadamardProxyFitWeighting: TypeAlias = Literal[
    "uniform",
    "inverse_variance",
]
PredictionNonfiniteKind: TypeAlias = Literal[
    "nan",
    "positive_infinity",
    "negative_infinity",
]

RPE_HADAMARD_COMPILED_COST_PROXY_SCHEMA_VERSION = (
    "rpe_hadamard_compiled_cost_affine_proxy_v1"
)
RPE_HADAMARD_COMPILED_COST_PROXY_VALIDATION_SCHEMA_VERSION = (
    "rpe_hadamard_compiled_cost_proxy_validation_v1"
)
RPE_HADAMARD_COMPILED_COST_PROXY_MODEL_FAMILY = "axis_metric_affine_v1"
RPE_HADAMARD_COMPILED_COST_PROXY_MODEL_FORMULA = "slope*q_m+intercept"
RPE_HADAMARD_COMPILED_COST_PROXY_ACCURACY_SCOPE = (
    "validated_holdout_q_m_range_only"
)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Proxy fingerprints require finite floats.")
        return {"float_hex": value.hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    return value


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"{name} must be a 64-character lowercase hexadecimal SHA-256."
        )
    return value


def _require_backend_fingerprint(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if value == "no_backend":
        return value
    return _require_sha256(value, name=name)


def _require_boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean.")
    return value


def _require_finite_float(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and normalized < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return normalized


def _require_optional_finite_float(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, name=name, nonnegative=nonnegative)


def _require_axis(value: Any) -> RPEHadamardAxis:
    if value not in ("cosine", "sine"):
        raise ValueError("axis must be 'cosine' or 'sine'.")
    return value


def _require_metric(value: Any) -> RPECostMetric:
    if value not in RPE_COST_METRICS:
        raise ValueError("metric is not a supported RPE cost metric.")
    return value


def _require_positive_power_of_two(value: Any, *, name: str) -> int:
    try:
        q_m = require_integer_count(value, name=name, minimum=1)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive power of two.") from exc
    if q_m & (q_m - 1):
        raise ValueError(f"{name} must be a positive power of two.")
    return q_m


def _set_or_validate_fingerprint(
    supplied: str,
    expected: str,
    *,
    name: str,
) -> str:
    if supplied:
        _require_sha256(supplied, name=name)
        if supplied != expected:
            raise ValueError(f"{name} does not match its content.")
    return expected


@dataclass(frozen=True)
class RPEHadamardProxyReferencePoint:
    """Minimal immutable calibration or holdout snapshot."""

    partition: Literal["calibration", "holdout"]
    round_index: int
    q_m: int
    axis: RPEHadamardAxis
    metric_means: tuple[tuple[str, float], ...]
    metric_standard_errors: tuple[tuple[str, float | None], ...]
    point_fingerprint: str

    def __post_init__(self) -> None:
        if self.partition not in ("calibration", "holdout"):
            raise ValueError("Unsupported proxy reference partition.")
        q_m = _require_positive_power_of_two(self.q_m, name="q_m")
        round_index = require_integer_count(self.round_index, name="round_index")
        if round_index != q_m.bit_length() - 1:
            raise ValueError("round_index must equal log2(q_m).")
        object.__setattr__(self, "q_m", q_m)
        object.__setattr__(self, "round_index", round_index)
        object.__setattr__(self, "axis", _require_axis(self.axis))
        _require_sha256(self.point_fingerprint, name="point_fingerprint")

        means = dict(self.metric_means)
        standard_errors = dict(self.metric_standard_errors)
        if len(means) != len(self.metric_means) or set(means) != set(
            RPE_COST_METRICS
        ):
            raise ValueError("metric_means must contain each fixed cost metric once.")
        if len(standard_errors) != len(
            self.metric_standard_errors
        ) or set(standard_errors) != set(RPE_COST_METRICS):
            raise ValueError(
                "metric_standard_errors must contain each fixed cost metric once."
            )
        normalized_means = tuple(
            (
                metric,
                _require_finite_float(
                    means[metric],
                    name=f"{metric} mean",
                    nonnegative=True,
                ),
            )
            for metric in RPE_COST_METRICS
        )
        normalized_standard_errors = tuple(
            (
                metric,
                _require_optional_finite_float(
                    standard_errors[metric],
                    name=f"{metric} standard_error",
                    nonnegative=True,
                ),
            )
            for metric in RPE_COST_METRICS
        )
        object.__setattr__(self, "metric_means", normalized_means)
        object.__setattr__(
            self,
            "metric_standard_errors",
            normalized_standard_errors,
        )

    def mean(self, metric: RPECostMetric) -> float:
        checked = _require_metric(metric)
        return dict(self.metric_means)[checked]

    def standard_error(self, metric: RPECostMetric) -> float | None:
        checked = _require_metric(metric)
        return dict(self.metric_standard_errors)[checked]

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "round_index": self.round_index,
            "m": self.round_index,
            "q_m": self.q_m,
            "axis": self.axis,
            "metric_means": dict(self.metric_means),
            "metric_standard_errors": dict(self.metric_standard_errors),
            "point_fingerprint": self.point_fingerprint,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RPEHadamardProxyReferencePoint":
        round_index = require_integer_count(
            payload["round_index"],
            name="round_index",
        )
        serialized_m = require_integer_count(payload["m"], name="m")
        if serialized_m != round_index:
            raise ValueError("Serialized m must equal round_index.")
        means = payload["metric_means"]
        standard_errors = payload["metric_standard_errors"]
        return cls(
            partition=payload["partition"],
            round_index=round_index,
            q_m=_require_positive_power_of_two(payload["q_m"], name="q_m"),
            axis=payload["axis"],
            metric_means=tuple((metric, means[metric]) for metric in means),
            metric_standard_errors=tuple(
                (metric, standard_errors[metric]) for metric in standard_errors
            ),
            point_fingerprint=_require_sha256(
                payload.get("point_fingerprint"),
                name="point_fingerprint",
            ),
        )


@dataclass(frozen=True)
class RPEHadamardCalibrationResidual:
    q_m: int
    observed_mean: float
    observed_standard_error: float | None
    predicted_cost: float
    signed_residual: float
    point_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "q_m",
            _require_positive_power_of_two(self.q_m, name="q_m"),
        )
        observed = _require_finite_float(
            self.observed_mean,
            name="observed_mean",
            nonnegative=True,
        )
        standard_error = _require_optional_finite_float(
            self.observed_standard_error,
            name="observed_standard_error",
            nonnegative=True,
        )
        predicted = _require_finite_float(
            self.predicted_cost,
            name="predicted_cost",
        )
        residual = _require_finite_float(
            self.signed_residual,
            name="signed_residual",
        )
        if residual != predicted - observed:
            raise ValueError("signed_residual is inconsistent.")
        _require_sha256(self.point_fingerprint, name="point_fingerprint")
        object.__setattr__(self, "observed_mean", observed)
        object.__setattr__(self, "observed_standard_error", standard_error)
        object.__setattr__(self, "predicted_cost", predicted)
        object.__setattr__(self, "signed_residual", residual)

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_m": self.q_m,
            "observed_mean": self.observed_mean,
            "observed_standard_error": self.observed_standard_error,
            "predicted_cost": self.predicted_cost,
            "signed_residual": self.signed_residual,
            "absolute_residual": abs(self.signed_residual),
            "point_fingerprint": self.point_fingerprint,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RPEHadamardCalibrationResidual":
        residual = cls(
            q_m=payload["q_m"],
            observed_mean=payload["observed_mean"],
            observed_standard_error=payload["observed_standard_error"],
            predicted_cost=payload["predicted_cost"],
            signed_residual=payload["signed_residual"],
            point_fingerprint=_require_sha256(
                payload.get("point_fingerprint"),
                name="point_fingerprint",
            ),
        )
        serialized_absolute = _require_finite_float(
            payload["absolute_residual"],
            name="absolute_residual",
            nonnegative=True,
        )
        if serialized_absolute != abs(residual.signed_residual):
            raise ValueError("absolute_residual is inconsistent.")
        return residual


@dataclass(frozen=True)
class RPEHadamardAffineMetricModel:
    axis: RPEHadamardAxis
    metric: RPECostMetric
    slope: float
    intercept: float
    fit_weighting: RPEHadamardProxyFitWeighting
    calibration_residuals: tuple[RPEHadamardCalibrationResidual, ...]
    root_mean_square_residual: float
    maximum_absolute_residual: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", _require_axis(self.axis))
        object.__setattr__(self, "metric", _require_metric(self.metric))
        if self.fit_weighting not in ("uniform", "inverse_variance"):
            raise ValueError("Unsupported proxy fit weighting.")
        slope = _require_finite_float(self.slope, name="slope")
        intercept = _require_finite_float(self.intercept, name="intercept")
        residuals = tuple(sorted(self.calibration_residuals, key=lambda item: item.q_m))
        if len(residuals) < 2 or len({item.q_m for item in residuals}) != len(
            residuals
        ):
            raise ValueError(
                "An affine metric model requires at least two distinct q_m points."
            )
        for residual in residuals:
            if not isinstance(residual, RPEHadamardCalibrationResidual):
                raise TypeError("calibration_residuals contain an invalid value.")
            expected_prediction = slope * residual.q_m + intercept
            if residual.predicted_cost != expected_prediction:
                raise ValueError("Calibration prediction is inconsistent with model.")
        expected_rmse = math.sqrt(
            math.fsum(item.signed_residual**2 for item in residuals)
            / len(residuals)
        )
        expected_maximum = max(abs(item.signed_residual) for item in residuals)
        rmse = _require_finite_float(
            self.root_mean_square_residual,
            name="root_mean_square_residual",
            nonnegative=True,
        )
        maximum = _require_finite_float(
            self.maximum_absolute_residual,
            name="maximum_absolute_residual",
            nonnegative=True,
        )
        if rmse != expected_rmse or maximum != expected_maximum:
            raise ValueError(
                "Serialized calibration residual aggregates are inconsistent."
            )
        object.__setattr__(self, "slope", slope)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "calibration_residuals", residuals)
        object.__setattr__(self, "root_mean_square_residual", rmse)
        object.__setattr__(self, "maximum_absolute_residual", maximum)

    @property
    def calibration_point_count(self) -> int:
        return len(self.calibration_residuals)

    @property
    def calibration_q_m_min(self) -> int:
        return self.calibration_residuals[0].q_m

    @property
    def calibration_q_m_max(self) -> int:
        return self.calibration_residuals[-1].q_m

    def predict(self, q_m: int) -> float:
        checked_q_m = _require_positive_power_of_two(q_m, name="q_m")
        try:
            return self.slope * checked_q_m + self.intercept
        except OverflowError:
            if self.slope == 0.0:
                return self.intercept
            return math.copysign(math.inf, self.slope)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "metric": self.metric,
            "slope": self.slope,
            "intercept": self.intercept,
            "fit_weighting": self.fit_weighting,
            "calibration_point_count": self.calibration_point_count,
            "calibration_q_m_min": self.calibration_q_m_min,
            "calibration_q_m_max": self.calibration_q_m_max,
            "calibration_residuals": [
                residual.to_dict() for residual in self.calibration_residuals
            ],
            "root_mean_square_residual": self.root_mean_square_residual,
            "maximum_absolute_residual": self.maximum_absolute_residual,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RPEHadamardAffineMetricModel":
        model = cls(
            axis=payload["axis"],
            metric=payload["metric"],
            slope=payload["slope"],
            intercept=payload["intercept"],
            fit_weighting=payload["fit_weighting"],
            calibration_residuals=tuple(
                RPEHadamardCalibrationResidual.from_dict(item)
                for item in payload["calibration_residuals"]
            ),
            root_mean_square_residual=payload["root_mean_square_residual"],
            maximum_absolute_residual=payload["maximum_absolute_residual"],
        )
        redundant = (
            (
                "calibration_point_count",
                require_integer_count(
                    payload["calibration_point_count"],
                    name="calibration_point_count",
                ),
                model.calibration_point_count,
            ),
            (
                "calibration_q_m_min",
                require_integer_count(
                    payload["calibration_q_m_min"],
                    name="calibration_q_m_min",
                ),
                model.calibration_q_m_min,
            ),
            (
                "calibration_q_m_max",
                require_integer_count(
                    payload["calibration_q_m_max"],
                    name="calibration_q_m_max",
                ),
                model.calibration_q_m_max,
            ),
        )
        for name, serialized, expected in redundant:
            if serialized != expected:
                raise ValueError(f"Serialized {name} is inconsistent.")
        return model


@dataclass(frozen=True)
class RPEHadamardCompiledCostProxy:
    source_dataset_fingerprint: str
    calibration_points: tuple[RPEHadamardProxyReferencePoint, ...]
    models: tuple[RPEHadamardAffineMetricModel, ...]
    fit_weighting: RPEHadamardProxyFitWeighting
    preparation_fingerprint: str
    hamiltonian_fingerprint: str
    partition_fingerprint: str
    ld: int
    num_system_qubits: int
    delta_time: float
    rte_steps_per_occurrence: int
    finite_taylor_order: int
    product_formula_order: int
    control_convention: str
    construction_policy: RepeatedCircuitConstructionPolicy
    compiler_settings_fingerprint: str
    backend_fingerprint: str | None
    compiler_context_fingerprint: str
    source_evaluation_method: Literal["exact", "monte_carlo"]
    calibration_subset_fingerprint: str = ""
    fit_specification_fingerprint: str = ""
    fit_fingerprint: str = ""
    proxy_fingerprint: str = ""
    holdout_used_for_fit: bool = False
    state_preparation_included: bool = False
    backend_execution_included: bool = False
    quantum_shots_executed: int = 0
    resource_accounting_connected: bool = False
    circuit_scope: str = RPE_HADAMARD_INTERROGATION_SCOPE
    model_family: str = RPE_HADAMARD_COMPILED_COST_PROXY_MODEL_FAMILY
    model_formula: str = RPE_HADAMARD_COMPILED_COST_PROXY_MODEL_FORMULA
    schema_version: str = RPE_HADAMARD_COMPILED_COST_PROXY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RPE_HADAMARD_COMPILED_COST_PROXY_SCHEMA_VERSION:
            raise ValueError("Unsupported RPE Hadamard compiled-cost proxy schema.")
        if self.model_family != RPE_HADAMARD_COMPILED_COST_PROXY_MODEL_FAMILY:
            raise ValueError("Unsupported proxy model family.")
        if self.model_formula != RPE_HADAMARD_COMPILED_COST_PROXY_MODEL_FORMULA:
            raise ValueError("Unsupported proxy model formula.")
        if self.fit_weighting not in ("uniform", "inverse_variance"):
            raise ValueError("Unsupported proxy fit weighting.")
        if self.source_evaluation_method not in ("exact", "monte_carlo"):
            raise ValueError("Unsupported source evaluation method.")
        if self.circuit_scope != RPE_HADAMARD_INTERROGATION_SCOPE:
            raise ValueError("Unsupported proxy circuit scope.")
        if self.product_formula_order != 2:
            raise ValueError("The proxy requires second-order partial-S2 data.")
        if self.control_convention != RPE_HADAMARD_CONTROL_CONVENTION:
            raise ValueError("Unsupported controlled-evolution convention.")
        if self.construction_policy not in (
            "raw_concatenation",
            "boundary_optimized",
        ):
            raise ValueError("Unsupported repeated-circuit construction policy.")
        if self.holdout_used_for_fit is not False:
            raise ValueError("Holdout data must not be used for proxy fitting.")
        if self.state_preparation_included is not False:
            raise ValueError("The proxy must exclude state preparation.")
        if self.backend_execution_included is not False:
            raise ValueError("The proxy must exclude backend execution.")
        if self.quantum_shots_executed != 0:
            raise ValueError("The proxy must not execute quantum shots.")
        if self.resource_accounting_connected is not False:
            raise ValueError("The proxy must not be connected to resource accounting.")

        for name in (
            "source_dataset_fingerprint",
            "preparation_fingerprint",
            "hamiltonian_fingerprint",
            "partition_fingerprint",
            "compiler_settings_fingerprint",
            "compiler_context_fingerprint",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_backend_fingerprint(
            self.backend_fingerprint,
            name="backend_fingerprint",
        )
        expected_compiler_context_fingerprint = _compiler_context_fingerprint(
            self.compiler_settings_fingerprint,
            self.backend_fingerprint,
        )
        if (
            self.compiler_context_fingerprint
            != expected_compiler_context_fingerprint
        ):
            raise ValueError(
                "compiler_context_fingerprint is inconsistent with the compiler "
                "settings and backend fingerprints."
            )
        object.__setattr__(
            self,
            "ld",
            require_integer_count(self.ld, name="ld"),
        )
        object.__setattr__(
            self,
            "num_system_qubits",
            require_integer_count(
                self.num_system_qubits,
                name="num_system_qubits",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "delta_time",
            _require_finite_float(
                self.delta_time,
                name="delta_time",
            ),
        )
        if self.delta_time <= 0.0:
            raise ValueError("delta_time must be positive.")
        object.__setattr__(
            self,
            "rte_steps_per_occurrence",
            require_integer_count(
                self.rte_steps_per_occurrence,
                name="rte_steps_per_occurrence",
            ),
        )
        finite_order = require_integer_count(
            self.finite_taylor_order,
            name="finite_taylor_order",
        )
        if finite_order % 2:
            raise ValueError("finite_taylor_order must be non-negative and even.")
        object.__setattr__(self, "finite_taylor_order", finite_order)

        calibration_points = tuple(
            sorted(self.calibration_points, key=lambda item: (item.axis, item.q_m))
        )
        if any(
            not isinstance(point, RPEHadamardProxyReferencePoint)
            for point in calibration_points
        ):
            raise TypeError("calibration_points contain an invalid value.")
        if any(point.partition != "calibration" for point in calibration_points):
            raise ValueError("Proxy calibration_points must be calibration-only.")
        q_values = tuple(sorted({point.q_m for point in calibration_points}))
        if len(q_values) < 2:
            raise ValueError("Affine fitting requires at least two calibration q_m.")
        expected_point_keys = {
            (axis, q_m) for axis in ("cosine", "sine") for q_m in q_values
        }
        actual_point_keys = {
            (point.axis, point.q_m) for point in calibration_points
        }
        if (
            len(actual_point_keys) != len(calibration_points)
            or actual_point_keys != expected_point_keys
        ):
            raise ValueError(
                "Calibration points must cover each axis/q_m exactly once."
            )
        object.__setattr__(self, "calibration_points", calibration_points)

        models = tuple(
            sorted(
                self.models,
                key=lambda item: (
                    ("cosine", "sine").index(item.axis),
                    RPE_COST_METRICS.index(item.metric),
                ),
            )
        )
        if any(not isinstance(model, RPEHadamardAffineMetricModel) for model in models):
            raise TypeError("models contain an invalid value.")
        expected_model_keys = {
            (axis, metric)
            for axis in ("cosine", "sine")
            for metric in RPE_COST_METRICS
        }
        actual_model_keys = {(model.axis, model.metric) for model in models}
        if len(actual_model_keys) != len(models) or actual_model_keys != (
            expected_model_keys
        ):
            raise ValueError("Proxy must contain one model per axis and cost metric.")
        expected_models = _fit_all_models(calibration_points, self.fit_weighting)
        if models != expected_models:
            raise ValueError("Serialized affine models do not match calibration data.")
        object.__setattr__(self, "models", models)

        expected_calibration_fingerprint = _reference_subset_fingerprint(
            calibration_points
        )
        expected_fit_specification_fingerprint = _sha256_json(
            self._fit_specification_payload()
        )
        object.__setattr__(
            self,
            "calibration_subset_fingerprint",
            _set_or_validate_fingerprint(
                self.calibration_subset_fingerprint,
                expected_calibration_fingerprint,
                name="calibration_subset_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "fit_specification_fingerprint",
            _set_or_validate_fingerprint(
                self.fit_specification_fingerprint,
                expected_fit_specification_fingerprint,
                name="fit_specification_fingerprint",
            ),
        )
        expected_fit_fingerprint = _sha256_json(self._fit_fingerprint_payload())
        object.__setattr__(
            self,
            "fit_fingerprint",
            _set_or_validate_fingerprint(
                self.fit_fingerprint,
                expected_fit_fingerprint,
                name="fit_fingerprint",
            ),
        )
        expected_proxy_fingerprint = _sha256_json(
            self._payload(include_proxy_fingerprint=False)
        )
        object.__setattr__(
            self,
            "proxy_fingerprint",
            _set_or_validate_fingerprint(
                self.proxy_fingerprint,
                expected_proxy_fingerprint,
                name="proxy_fingerprint",
            ),
        )

    @property
    def calibration_q_m_values(self) -> tuple[int, ...]:
        return tuple(sorted({point.q_m for point in self.calibration_points}))

    @property
    def calibration_q_m_min(self) -> int:
        return self.calibration_q_m_values[0]

    @property
    def calibration_q_m_max(self) -> int:
        return self.calibration_q_m_values[-1]

    @property
    def calibration_point_count(self) -> int:
        return len(self.calibration_points)

    def model(
        self,
        axis: RPEHadamardAxis,
        metric: RPECostMetric,
    ) -> RPEHadamardAffineMetricModel:
        checked_axis = _require_axis(axis)
        checked_metric = _require_metric(metric)
        return next(
            model
            for model in self.models
            if model.axis == checked_axis and model.metric == checked_metric
        )

    def predict(
        self,
        q_m: int,
        *,
        axis: RPEHadamardAxis,
        metric: RPECostMetric,
    ) -> float:
        checked_q_m = _require_positive_power_of_two(q_m, name="q_m")
        return self.model(axis, metric).predict(checked_q_m)

    def _fit_specification_payload(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_formula": self.model_formula,
            "fit_weighting": self.fit_weighting,
            "axes": ["cosine", "sine"],
            "cost_metrics": list(RPE_COST_METRICS),
            "canonical_sort": "q_m_ascending",
            "coefficient_clamping": False,
            "prediction_rounding": False,
        }

    def _configuration_payload(self) -> dict[str, Any]:
        return {
            "preparation_fingerprint": self.preparation_fingerprint,
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "partition_fingerprint": self.partition_fingerprint,
            "ld": self.ld,
            "num_system_qubits": self.num_system_qubits,
            "delta_time": self.delta_time,
            "rte_steps_per_occurrence": self.rte_steps_per_occurrence,
            "finite_taylor_order": self.finite_taylor_order,
            "product_formula_order": self.product_formula_order,
            "control_convention": self.control_convention,
            "construction_policy": self.construction_policy,
            "compiler_settings_fingerprint": self.compiler_settings_fingerprint,
            "backend_fingerprint": self.backend_fingerprint,
            "compiler_context_fingerprint": self.compiler_context_fingerprint,
            "circuit_scope": self.circuit_scope,
        }

    def _fit_fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_subset_fingerprint": (
                self.calibration_subset_fingerprint
            ),
            "fit_specification_fingerprint": (
                self.fit_specification_fingerprint
            ),
            "configuration": self._configuration_payload(),
            "models": [model.to_dict() for model in self.models],
            "holdout_used_for_fit": self.holdout_used_for_fit,
        }

    def _payload(self, *, include_proxy_fingerprint: bool) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "model_formula": self.model_formula,
            "fit_weighting": self.fit_weighting,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "source_evaluation_method": self.source_evaluation_method,
            "calibration_subset_fingerprint": (
                self.calibration_subset_fingerprint
            ),
            "fit_specification_fingerprint": (
                self.fit_specification_fingerprint
            ),
            "fit_fingerprint": self.fit_fingerprint,
            "calibration_points": [
                point.to_dict() for point in self.calibration_points
            ],
            "calibration_point_count": self.calibration_point_count,
            "calibration_q_m_values": list(self.calibration_q_m_values),
            "calibration_q_m_min": self.calibration_q_m_min,
            "calibration_q_m_max": self.calibration_q_m_max,
            "models": [model.to_dict() for model in self.models],
            "model_count": len(self.models),
            **self._configuration_payload(),
            "holdout_used_for_fit": self.holdout_used_for_fit,
            "state_preparation_included": self.state_preparation_included,
            "backend_execution_included": self.backend_execution_included,
            "quantum_shots_executed": self.quantum_shots_executed,
            "resource_accounting_connected": self.resource_accounting_connected,
        }
        if include_proxy_fingerprint:
            payload["proxy_fingerprint"] = self.proxy_fingerprint
        return payload

    def to_dict(self) -> dict[str, Any]:
        return self._payload(include_proxy_fingerprint=True)

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "RPEHadamardCompiledCostProxy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RPEHadamardCompiledCostProxy":
        if payload.get("schema_version") != (
            RPE_HADAMARD_COMPILED_COST_PROXY_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported RPE Hadamard compiled-cost proxy schema.")
        proxy = cls(
            source_dataset_fingerprint=_require_sha256(
                payload.get("source_dataset_fingerprint"),
                name="source_dataset_fingerprint",
            ),
            calibration_points=tuple(
                RPEHadamardProxyReferencePoint.from_dict(item)
                for item in payload["calibration_points"]
            ),
            models=tuple(
                RPEHadamardAffineMetricModel.from_dict(item)
                for item in payload["models"]
            ),
            fit_weighting=payload["fit_weighting"],
            preparation_fingerprint=payload["preparation_fingerprint"],
            hamiltonian_fingerprint=payload["hamiltonian_fingerprint"],
            partition_fingerprint=payload["partition_fingerprint"],
            ld=payload["ld"],
            num_system_qubits=payload["num_system_qubits"],
            delta_time=payload["delta_time"],
            rte_steps_per_occurrence=payload["rte_steps_per_occurrence"],
            finite_taylor_order=payload["finite_taylor_order"],
            product_formula_order=payload["product_formula_order"],
            control_convention=payload["control_convention"],
            construction_policy=payload["construction_policy"],
            compiler_settings_fingerprint=payload["compiler_settings_fingerprint"],
            backend_fingerprint=payload["backend_fingerprint"],
            compiler_context_fingerprint=payload["compiler_context_fingerprint"],
            source_evaluation_method=payload["source_evaluation_method"],
            calibration_subset_fingerprint=_require_sha256(
                payload.get("calibration_subset_fingerprint"),
                name="calibration_subset_fingerprint",
            ),
            fit_specification_fingerprint=_require_sha256(
                payload.get("fit_specification_fingerprint"),
                name="fit_specification_fingerprint",
            ),
            fit_fingerprint=_require_sha256(
                payload.get("fit_fingerprint"),
                name="fit_fingerprint",
            ),
            proxy_fingerprint=_require_sha256(
                payload.get("proxy_fingerprint"),
                name="proxy_fingerprint",
            ),
            holdout_used_for_fit=_require_boolean(
                payload["holdout_used_for_fit"],
                name="holdout_used_for_fit",
            ),
            state_preparation_included=_require_boolean(
                payload["state_preparation_included"],
                name="state_preparation_included",
            ),
            backend_execution_included=_require_boolean(
                payload["backend_execution_included"],
                name="backend_execution_included",
            ),
            quantum_shots_executed=payload["quantum_shots_executed"],
            resource_accounting_connected=_require_boolean(
                payload["resource_accounting_connected"],
                name="resource_accounting_connected",
            ),
            circuit_scope=payload["circuit_scope"],
            model_family=payload["model_family"],
            model_formula=payload["model_formula"],
            schema_version=payload["schema_version"],
        )
        redundant = (
            ("calibration_point_count", proxy.calibration_point_count),
            ("calibration_q_m_values", list(proxy.calibration_q_m_values)),
            ("calibration_q_m_min", proxy.calibration_q_m_min),
            ("calibration_q_m_max", proxy.calibration_q_m_max),
            ("model_count", len(proxy.models)),
        )
        for name, expected in redundant:
            if payload[name] != expected:
                raise ValueError(f"Serialized {name} is inconsistent.")
        return proxy


@dataclass(frozen=True)
class RPEHadamardCompiledCostProxyFitRequest:
    dataset: RPEHadamardCompiledCostBenchmarkDataset
    weighting: RPEHadamardProxyFitWeighting = "uniform"

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, RPEHadamardCompiledCostBenchmarkDataset):
            raise TypeError(
                "dataset must be an RPEHadamardCompiledCostBenchmarkDataset."
            )
        if self.weighting not in ("uniform", "inverse_variance"):
            raise ValueError("Unsupported proxy fit weighting.")


@dataclass(frozen=True)
class RPEHadamardProxyMetricTolerance:
    metric: RPECostMetric
    absolute_tolerance: float
    relative_tolerance: float
    standard_error_multiplier: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _require_metric(self.metric))
        for name in (
            "absolute_tolerance",
            "relative_tolerance",
            "standard_error_multiplier",
        ):
            object.__setattr__(
                self,
                name,
                _require_finite_float(
                    getattr(self, name),
                    name=name,
                    nonnegative=True,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "standard_error_multiplier": self.standard_error_multiplier,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RPEHadamardProxyMetricTolerance":
        return cls(
            metric=payload["metric"],
            absolute_tolerance=payload["absolute_tolerance"],
            relative_tolerance=payload["relative_tolerance"],
            standard_error_multiplier=payload["standard_error_multiplier"],
        )


@dataclass(frozen=True)
class RPEHadamardProxyValidationEntry:
    partition: Literal["holdout"]
    round_index: int
    q_m: int
    axis: RPEHadamardAxis
    metric: RPECostMetric
    point_fingerprint: str
    observed_mean: float
    observed_standard_error: float | None
    predicted_cost: float | None
    prediction_nonfinite_kind: PredictionNonfiniteKind | None
    signed_error: float | None
    absolute_error: float | None
    relative_error: float | None
    outside_calibration_range: bool
    prediction_is_finite: bool
    prediction_is_nonnegative: bool
    acceptance_limit: float
    passed: bool

    def __post_init__(self) -> None:
        if self.partition != "holdout":
            raise ValueError("Proxy validation entries must be holdout-only.")
        q_m = _require_positive_power_of_two(self.q_m, name="q_m")
        round_index = require_integer_count(self.round_index, name="round_index")
        if round_index != q_m.bit_length() - 1:
            raise ValueError("round_index must equal log2(q_m).")
        object.__setattr__(self, "q_m", q_m)
        object.__setattr__(self, "round_index", round_index)
        object.__setattr__(self, "axis", _require_axis(self.axis))
        object.__setattr__(self, "metric", _require_metric(self.metric))
        _require_sha256(self.point_fingerprint, name="point_fingerprint")
        observed = _require_finite_float(
            self.observed_mean,
            name="observed_mean",
            nonnegative=True,
        )
        standard_error = _require_optional_finite_float(
            self.observed_standard_error,
            name="observed_standard_error",
            nonnegative=True,
        )
        acceptance_limit = _require_finite_float(
            self.acceptance_limit,
            name="acceptance_limit",
            nonnegative=True,
        )
        for name in (
            "outside_calibration_range",
            "prediction_is_finite",
            "prediction_is_nonnegative",
            "passed",
        ):
            _require_boolean(getattr(self, name), name=name)

        if self.prediction_is_finite:
            predicted = _require_finite_float(
                self.predicted_cost,
                name="predicted_cost",
            )
            if self.prediction_nonfinite_kind is not None:
                raise ValueError("A finite prediction cannot have a nonfinite kind.")
            signed_error = _require_finite_float(
                self.signed_error,
                name="signed_error",
            )
            absolute_error = _require_finite_float(
                self.absolute_error,
                name="absolute_error",
                nonnegative=True,
            )
            if signed_error != predicted - observed:
                raise ValueError("signed_error is inconsistent.")
            if absolute_error != abs(signed_error):
                raise ValueError("absolute_error is inconsistent.")
            expected_relative = (
                None if observed == 0.0 else absolute_error / abs(observed)
            )
            relative_error = _require_optional_finite_float(
                self.relative_error,
                name="relative_error",
                nonnegative=True,
            )
            if relative_error != expected_relative:
                raise ValueError("relative_error is inconsistent.")
            expected_nonnegative = predicted >= 0.0
            expected_passed = (
                expected_nonnegative and absolute_error <= acceptance_limit
            )
            if self.prediction_is_nonnegative != expected_nonnegative:
                raise ValueError("prediction_is_nonnegative is inconsistent.")
            if self.passed != expected_passed:
                raise ValueError("Validation entry pass/fail is inconsistent.")
            object.__setattr__(self, "predicted_cost", predicted)
            object.__setattr__(self, "signed_error", signed_error)
            object.__setattr__(self, "absolute_error", absolute_error)
            object.__setattr__(self, "relative_error", relative_error)
        else:
            if self.prediction_nonfinite_kind not in (
                "nan",
                "positive_infinity",
                "negative_infinity",
            ):
                raise ValueError("A nonfinite prediction requires its kind.")
            if any(
                value is not None
                for value in (
                    self.predicted_cost,
                    self.signed_error,
                    self.absolute_error,
                    self.relative_error,
                )
            ):
                raise ValueError("Nonfinite prediction numeric fields must be None.")
            if self.prediction_is_nonnegative or self.passed:
                raise ValueError("A nonfinite prediction must fail validation.")
        object.__setattr__(self, "observed_mean", observed)
        object.__setattr__(self, "observed_standard_error", standard_error)
        object.__setattr__(self, "acceptance_limit", acceptance_limit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "round_index": self.round_index,
            "m": self.round_index,
            "q_m": self.q_m,
            "axis": self.axis,
            "metric": self.metric,
            "point_fingerprint": self.point_fingerprint,
            "observed_mean": self.observed_mean,
            "observed_standard_error": self.observed_standard_error,
            "predicted_cost": self.predicted_cost,
            "prediction_nonfinite_kind": self.prediction_nonfinite_kind,
            "signed_error": self.signed_error,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
            "outside_calibration_range": self.outside_calibration_range,
            "prediction_is_finite": self.prediction_is_finite,
            "prediction_is_nonnegative": self.prediction_is_nonnegative,
            "acceptance_limit": self.acceptance_limit,
            "pass": self.passed,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RPEHadamardProxyValidationEntry":
        round_index = require_integer_count(
            payload["round_index"],
            name="round_index",
        )
        if require_integer_count(payload["m"], name="m") != round_index:
            raise ValueError("Serialized m must equal round_index.")
        return cls(
            partition=payload["partition"],
            round_index=round_index,
            q_m=payload["q_m"],
            axis=payload["axis"],
            metric=payload["metric"],
            point_fingerprint=_require_sha256(
                payload.get("point_fingerprint"),
                name="point_fingerprint",
            ),
            observed_mean=payload["observed_mean"],
            observed_standard_error=payload["observed_standard_error"],
            predicted_cost=payload["predicted_cost"],
            prediction_nonfinite_kind=payload["prediction_nonfinite_kind"],
            signed_error=payload["signed_error"],
            absolute_error=payload["absolute_error"],
            relative_error=payload["relative_error"],
            outside_calibration_range=_require_boolean(
                payload["outside_calibration_range"],
                name="outside_calibration_range",
            ),
            prediction_is_finite=_require_boolean(
                payload["prediction_is_finite"],
                name="prediction_is_finite",
            ),
            prediction_is_nonnegative=_require_boolean(
                payload["prediction_is_nonnegative"],
                name="prediction_is_nonnegative",
            ),
            acceptance_limit=payload["acceptance_limit"],
            passed=_require_boolean(payload["pass"], name="pass"),
        )


@dataclass(frozen=True)
class RPEHadamardCompiledCostProxyValidationRequest:
    proxy: RPEHadamardCompiledCostProxy
    dataset: RPEHadamardCompiledCostBenchmarkDataset
    metric_tolerances: tuple[RPEHadamardProxyMetricTolerance, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.proxy, RPEHadamardCompiledCostProxy):
            raise TypeError("proxy must be an RPEHadamardCompiledCostProxy.")
        if not isinstance(self.dataset, RPEHadamardCompiledCostBenchmarkDataset):
            raise TypeError(
                "dataset must be an RPEHadamardCompiledCostBenchmarkDataset."
            )
        tolerances = _canonical_tolerances(self.metric_tolerances)
        object.__setattr__(self, "metric_tolerances", tolerances)


@dataclass(frozen=True)
class RPEHadamardCompiledCostProxyValidationResult:
    proxy: RPEHadamardCompiledCostProxy
    source_holdout_dataset_fingerprint: str
    holdout_points: tuple[RPEHadamardProxyReferencePoint, ...]
    metric_tolerances: tuple[RPEHadamardProxyMetricTolerance, ...]
    entries: tuple[RPEHadamardProxyValidationEntry, ...]
    holdout_subset_fingerprint: str = ""
    acceptance_policy_fingerprint: str = ""
    validation_fingerprint: str = ""
    validation_refit_performed: bool = False
    accuracy_guaranteed_beyond_validated_range: bool = False
    accuracy_claim_scope: str = RPE_HADAMARD_COMPILED_COST_PROXY_ACCURACY_SCOPE
    schema_version: str = (
        RPE_HADAMARD_COMPILED_COST_PROXY_VALIDATION_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if self.schema_version != (
            RPE_HADAMARD_COMPILED_COST_PROXY_VALIDATION_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported proxy validation schema.")
        if not isinstance(self.proxy, RPEHadamardCompiledCostProxy):
            raise TypeError("proxy must be an RPEHadamardCompiledCostProxy.")
        _require_sha256(
            self.source_holdout_dataset_fingerprint,
            name="source_holdout_dataset_fingerprint",
        )
        if self.validation_refit_performed is not False:
            raise ValueError("Holdout validation must not refit the proxy.")
        if self.accuracy_guaranteed_beyond_validated_range is not False:
            raise ValueError("Validation cannot guarantee accuracy beyond holdout q_m.")
        if self.accuracy_claim_scope != (
            RPE_HADAMARD_COMPILED_COST_PROXY_ACCURACY_SCOPE
        ):
            raise ValueError("Unsupported proxy accuracy claim scope.")

        holdout_points = tuple(
            sorted(self.holdout_points, key=lambda item: (item.axis, item.q_m))
        )
        if any(
            not isinstance(point, RPEHadamardProxyReferencePoint)
            for point in holdout_points
        ):
            raise TypeError("holdout_points contain an invalid value.")
        if not holdout_points or any(
            point.partition != "holdout" for point in holdout_points
        ):
            raise ValueError("Validation requires at least one holdout q_m.")
        q_values = tuple(sorted({point.q_m for point in holdout_points}))
        expected_point_keys = {
            (axis, q_m) for axis in ("cosine", "sine") for q_m in q_values
        }
        actual_point_keys = {(point.axis, point.q_m) for point in holdout_points}
        if (
            len(actual_point_keys) != len(holdout_points)
            or actual_point_keys != expected_point_keys
        ):
            raise ValueError("Holdout points must cover each axis/q_m exactly once.")
        overlapping_q_m = set(q_values).intersection(
            self.proxy.calibration_q_m_values
        )
        if overlapping_q_m:
            raise ValueError(
                "Holdout q_m must be disjoint from proxy calibration q_m."
            )
        if (
            self.source_holdout_dataset_fingerprint
            != self.proxy.source_dataset_fingerprint
        ):
            raise ValueError(
                "Holdout validation must use the proxy source benchmark dataset."
            )
        object.__setattr__(self, "holdout_points", holdout_points)
        tolerances = _canonical_tolerances(self.metric_tolerances)
        object.__setattr__(self, "metric_tolerances", tolerances)

        entries = tuple(
            sorted(
                self.entries,
                key=lambda item: (
                    item.q_m,
                    ("cosine", "sine").index(item.axis),
                    RPE_COST_METRICS.index(item.metric),
                ),
            )
        )
        if any(
            not isinstance(entry, RPEHadamardProxyValidationEntry)
            for entry in entries
        ):
            raise TypeError("entries contain an invalid value.")
        expected_entries = _build_validation_entries(
            self.proxy,
            holdout_points,
            tolerances,
        )
        if entries != expected_entries:
            raise ValueError("Serialized validation entries are inconsistent.")
        object.__setattr__(self, "entries", entries)

        expected_holdout_fingerprint = _reference_subset_fingerprint(holdout_points)
        expected_policy_fingerprint = _sha256_json(
            [tolerance.to_dict() for tolerance in tolerances]
        )
        object.__setattr__(
            self,
            "holdout_subset_fingerprint",
            _set_or_validate_fingerprint(
                self.holdout_subset_fingerprint,
                expected_holdout_fingerprint,
                name="holdout_subset_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "acceptance_policy_fingerprint",
            _set_or_validate_fingerprint(
                self.acceptance_policy_fingerprint,
                expected_policy_fingerprint,
                name="acceptance_policy_fingerprint",
            ),
        )
        expected_validation_fingerprint = _sha256_json(
            self._payload(include_validation_fingerprint=False)
        )
        object.__setattr__(
            self,
            "validation_fingerprint",
            _set_or_validate_fingerprint(
                self.validation_fingerprint,
                expected_validation_fingerprint,
                name="validation_fingerprint",
            ),
        )

    @property
    def validated_q_m_values(self) -> tuple[int, ...]:
        return tuple(sorted({point.q_m for point in self.holdout_points}))

    @property
    def validated_q_m_min(self) -> int:
        return self.validated_q_m_values[0]

    @property
    def validated_q_m_max(self) -> int:
        return self.validated_q_m_values[-1]

    @property
    def overall_pass(self) -> bool:
        return bool(self.entries) and all(entry.passed for entry in self.entries)

    @property
    def passed_entry_count(self) -> int:
        return sum(entry.passed for entry in self.entries)

    @property
    def failed_entry_count(self) -> int:
        return sum(not entry.passed for entry in self.entries)

    def _payload(self, *, include_validation_fingerprint: bool) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "proxy": self.proxy.to_dict(),
            "fit_fingerprint": self.proxy.fit_fingerprint,
            "source_holdout_dataset_fingerprint": (
                self.source_holdout_dataset_fingerprint
            ),
            "holdout_points": [point.to_dict() for point in self.holdout_points],
            "holdout_subset_fingerprint": self.holdout_subset_fingerprint,
            "metric_tolerances": [
                tolerance.to_dict() for tolerance in self.metric_tolerances
            ],
            "acceptance_policy_fingerprint": (
                self.acceptance_policy_fingerprint
            ),
            "entries": [entry.to_dict() for entry in self.entries],
            "entry_count": len(self.entries),
            "passed_entry_count": self.passed_entry_count,
            "failed_entry_count": self.failed_entry_count,
            "overall_pass": self.overall_pass,
            "validated_q_m_values": list(self.validated_q_m_values),
            "validated_q_m_min": self.validated_q_m_min,
            "validated_q_m_max": self.validated_q_m_max,
            "validation_refit_performed": self.validation_refit_performed,
            "accuracy_guaranteed_beyond_validated_range": (
                self.accuracy_guaranteed_beyond_validated_range
            ),
            "accuracy_claim_scope": self.accuracy_claim_scope,
        }
        if include_validation_fingerprint:
            payload["validation_fingerprint"] = self.validation_fingerprint
        return payload

    def to_dict(self) -> dict[str, Any]:
        return self._payload(include_validation_fingerprint=True)

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(
        cls,
        path: str | Path,
    ) -> "RPEHadamardCompiledCostProxyValidationResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RPEHadamardCompiledCostProxyValidationResult":
        if payload.get("schema_version") != (
            RPE_HADAMARD_COMPILED_COST_PROXY_VALIDATION_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported proxy validation schema.")
        proxy = RPEHadamardCompiledCostProxy.from_dict(payload["proxy"])
        serialized_fit_fingerprint = _require_sha256(
            payload.get("fit_fingerprint"),
            name="fit_fingerprint",
        )
        if serialized_fit_fingerprint != proxy.fit_fingerprint:
            raise ValueError("Serialized fit_fingerprint is inconsistent.")
        result = cls(
            proxy=proxy,
            source_holdout_dataset_fingerprint=_require_sha256(
                payload.get("source_holdout_dataset_fingerprint"),
                name="source_holdout_dataset_fingerprint",
            ),
            holdout_points=tuple(
                RPEHadamardProxyReferencePoint.from_dict(item)
                for item in payload["holdout_points"]
            ),
            metric_tolerances=tuple(
                RPEHadamardProxyMetricTolerance.from_dict(item)
                for item in payload["metric_tolerances"]
            ),
            entries=tuple(
                RPEHadamardProxyValidationEntry.from_dict(item)
                for item in payload["entries"]
            ),
            holdout_subset_fingerprint=_require_sha256(
                payload.get("holdout_subset_fingerprint"),
                name="holdout_subset_fingerprint",
            ),
            acceptance_policy_fingerprint=_require_sha256(
                payload.get("acceptance_policy_fingerprint"),
                name="acceptance_policy_fingerprint",
            ),
            validation_fingerprint=_require_sha256(
                payload.get("validation_fingerprint"),
                name="validation_fingerprint",
            ),
            validation_refit_performed=_require_boolean(
                payload["validation_refit_performed"],
                name="validation_refit_performed",
            ),
            accuracy_guaranteed_beyond_validated_range=_require_boolean(
                payload["accuracy_guaranteed_beyond_validated_range"],
                name="accuracy_guaranteed_beyond_validated_range",
            ),
            accuracy_claim_scope=payload["accuracy_claim_scope"],
            schema_version=payload["schema_version"],
        )
        redundant = (
            ("entry_count", len(result.entries)),
            ("passed_entry_count", result.passed_entry_count),
            ("failed_entry_count", result.failed_entry_count),
            ("overall_pass", result.overall_pass),
            ("validated_q_m_values", list(result.validated_q_m_values)),
            ("validated_q_m_min", result.validated_q_m_min),
            ("validated_q_m_max", result.validated_q_m_max),
        )
        for name, expected in redundant:
            if payload[name] != expected:
                raise ValueError(f"Serialized {name} is inconsistent.")
        return result


def _reference_subset_fingerprint(
    points: tuple[RPEHadamardProxyReferencePoint, ...],
) -> str:
    return _sha256_json([point.to_dict() for point in points])


def _canonical_tolerances(
    tolerances: tuple[RPEHadamardProxyMetricTolerance, ...],
) -> tuple[RPEHadamardProxyMetricTolerance, ...]:
    if not isinstance(tolerances, tuple):
        raise TypeError("metric_tolerances must be a tuple.")
    if any(
        not isinstance(tolerance, RPEHadamardProxyMetricTolerance)
        for tolerance in tolerances
    ):
        raise TypeError("metric_tolerances contain an invalid value.")
    by_metric = {tolerance.metric: tolerance for tolerance in tolerances}
    if len(by_metric) != len(tolerances) or set(by_metric) != set(RPE_COST_METRICS):
        raise ValueError("metric_tolerances must contain each cost metric once.")
    return tuple(by_metric[metric] for metric in RPE_COST_METRICS)


def _require_complete_dataset(
    dataset: RPEHadamardCompiledCostBenchmarkDataset,
) -> None:
    if not isinstance(dataset, RPEHadamardCompiledCostBenchmarkDataset):
        raise TypeError("dataset must be an RPEHadamardCompiledCostBenchmarkDataset.")
    if dataset.schema_version != RPE_HADAMARD_COMPILED_COST_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("Proxy inputs require a schema-v2 benchmark dataset.")
    if not dataset.complete or dataset.failed_point_count:
        raise ValueError("Proxy inputs require a complete benchmark dataset.")
    if set(dataset.calibration_repetition_counts).intersection(
        dataset.holdout_repetition_counts
    ):
        raise ValueError("Calibration and holdout partitions must be disjoint.")
    if dataset.product_formula_order != 2:
        raise ValueError("Proxy inputs require second-order partial-S2 data.")
    if dataset.cost_metrics != RPE_COST_METRICS:
        raise ValueError("Dataset cost metrics do not match the fixed proxy metrics.")


def _reference_points_from_dataset(
    dataset: RPEHadamardCompiledCostBenchmarkDataset,
    *,
    partition: Literal["calibration", "holdout"],
) -> tuple[RPEHadamardProxyReferencePoint, ...]:
    points = []
    for point in dataset.records:
        if point.partition != partition:
            continue
        if point.status != "complete":
            raise ValueError("Proxy reference points must be complete.")
        statistics = dict(point.metric_statistics)
        if set(statistics) != set(RPE_COST_METRICS):
            raise ValueError("Point cost metrics do not match the fixed proxy metrics.")
        if dataset.requested_evaluation_method == "exact" and any(
            statistics[metric].standard_error is not None
            for metric in RPE_COST_METRICS
        ):
            raise ValueError(
                "Exact benchmark points must not contain standard errors."
            )
        points.append(
            RPEHadamardProxyReferencePoint(
                partition=partition,
                round_index=point.round_index,
                q_m=point.q_m,
                axis=point.axis,
                metric_means=tuple(
                    (metric, statistics[metric].mean) for metric in RPE_COST_METRICS
                ),
                metric_standard_errors=tuple(
                    (metric, statistics[metric].standard_error)
                    for metric in RPE_COST_METRICS
                ),
                point_fingerprint=point.point_fingerprint,
            )
        )
    return tuple(sorted(points, key=lambda item: (item.axis, item.q_m)))


def _fit_affine_metric_model(
    points: tuple[RPEHadamardProxyReferencePoint, ...],
    *,
    axis: RPEHadamardAxis,
    metric: RPECostMetric,
    weighting: RPEHadamardProxyFitWeighting,
) -> RPEHadamardAffineMetricModel:
    axis_points = tuple(
        sorted((point for point in points if point.axis == axis), key=lambda p: p.q_m)
    )
    if len(axis_points) < 2 or len({point.q_m for point in axis_points}) != len(
        axis_points
    ):
        raise ValueError("Affine fitting requires at least two distinct q_m points.")
    xs = tuple(float(point.q_m) for point in axis_points)
    ys = tuple(point.mean(metric) for point in axis_points)
    if weighting == "uniform":
        weights = tuple(1.0 for _point in axis_points)
    else:
        weights_list = []
        for point in axis_points:
            standard_error = point.standard_error(metric)
            if (
                standard_error is None
                or not math.isfinite(standard_error)
                or standard_error <= 0.0
            ):
                raise ValueError(
                    "inverse_variance fitting requires finite positive standard errors."
                )
            squared = standard_error * standard_error
            weight = math.inf if squared == 0.0 else 1.0 / squared
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(
                    "inverse_variance fitting produced a nonfinite weight."
                )
            weights_list.append(weight)
        weights = tuple(weights_list)

    weight_sum = math.fsum(weights)
    x_mean = math.fsum(weight * x for weight, x in zip(weights, xs)) / weight_sum
    y_mean = math.fsum(weight * y for weight, y in zip(weights, ys)) / weight_sum
    denominator = math.fsum(
        weight * (x - x_mean) ** 2 for weight, x in zip(weights, xs)
    )
    numerator = math.fsum(
        weight * (x - x_mean) * (y - y_mean)
        for weight, x, y in zip(weights, xs, ys)
    )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Affine calibration q_m values are numerically degenerate.")
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("Affine fitting produced nonfinite coefficients.")

    residuals = tuple(
        RPEHadamardCalibrationResidual(
            q_m=point.q_m,
            observed_mean=point.mean(metric),
            observed_standard_error=point.standard_error(metric),
            predicted_cost=slope * point.q_m + intercept,
            signed_residual=(slope * point.q_m + intercept) - point.mean(metric),
            point_fingerprint=point.point_fingerprint,
        )
        for point in axis_points
    )
    rmse = math.sqrt(
        math.fsum(residual.signed_residual**2 for residual in residuals)
        / len(residuals)
    )
    maximum = max(abs(residual.signed_residual) for residual in residuals)
    return RPEHadamardAffineMetricModel(
        axis=axis,
        metric=metric,
        slope=slope,
        intercept=intercept,
        fit_weighting=weighting,
        calibration_residuals=residuals,
        root_mean_square_residual=rmse,
        maximum_absolute_residual=maximum,
    )


def _fit_all_models(
    calibration_points: tuple[RPEHadamardProxyReferencePoint, ...],
    weighting: RPEHadamardProxyFitWeighting,
) -> tuple[RPEHadamardAffineMetricModel, ...]:
    return tuple(
        _fit_affine_metric_model(
            calibration_points,
            axis=axis,
            metric=metric,
            weighting=weighting,
        )
        for axis in ("cosine", "sine")
        for metric in RPE_COST_METRICS
    )


def _compiler_context_fingerprint(
    compiler_settings_fingerprint: str,
    backend_fingerprint: str | None,
) -> str:
    return _sha256_json(
        {
            "compiler_settings_fingerprint": compiler_settings_fingerprint,
            "backend_fingerprint": backend_fingerprint,
        }
    )


def fit_rpe_hadamard_compiled_cost_proxy(
    request: RPEHadamardCompiledCostProxyFitRequest,
) -> RPEHadamardCompiledCostProxy:
    """Fit 12 independent affine models from calibration points only."""
    if not isinstance(request, RPEHadamardCompiledCostProxyFitRequest):
        raise TypeError("request must be an RPEHadamardCompiledCostProxyFitRequest.")
    dataset = request.dataset
    _require_complete_dataset(dataset)
    if len(dataset.calibration_repetition_counts) < 2:
        raise ValueError("Affine fitting requires at least two calibration q_m.")

    # This is the only data passed to the regression core.  No holdout cost,
    # uncertainty, or point fingerprint enters calibration fitting.
    calibration_points = _reference_points_from_dataset(
        dataset,
        partition="calibration",
    )
    models = _fit_all_models(calibration_points, request.weighting)
    compiler_context = _compiler_context_fingerprint(
        dataset.compiler_settings_fingerprint,
        dataset.backend_fingerprint,
    )
    return RPEHadamardCompiledCostProxy(
        source_dataset_fingerprint=dataset.dataset_fingerprint,
        calibration_points=calibration_points,
        models=models,
        fit_weighting=request.weighting,
        preparation_fingerprint=dataset.preparation_fingerprint,
        hamiltonian_fingerprint=dataset.hamiltonian_fingerprint,
        partition_fingerprint=dataset.partition_fingerprint,
        ld=dataset.ld,
        num_system_qubits=dataset.num_system_qubits,
        delta_time=dataset.delta_time,
        rte_steps_per_occurrence=dataset.rte_steps_per_occurrence,
        finite_taylor_order=dataset.finite_taylor_order,
        product_formula_order=dataset.product_formula_order,
        control_convention=dataset.control_convention,
        construction_policy=dataset.construction_policy,
        compiler_settings_fingerprint=dataset.compiler_settings_fingerprint,
        backend_fingerprint=dataset.backend_fingerprint,
        compiler_context_fingerprint=compiler_context,
        source_evaluation_method=dataset.requested_evaluation_method,
    )


def _configuration_mismatches(
    proxy: RPEHadamardCompiledCostProxy,
    dataset: RPEHadamardCompiledCostBenchmarkDataset,
) -> tuple[str, ...]:
    expected_compiler_context = _compiler_context_fingerprint(
        dataset.compiler_settings_fingerprint,
        dataset.backend_fingerprint,
    )
    comparisons = (
        (
            "preparation_fingerprint",
            proxy.preparation_fingerprint,
            dataset.preparation_fingerprint,
        ),
        (
            "hamiltonian_fingerprint",
            proxy.hamiltonian_fingerprint,
            dataset.hamiltonian_fingerprint,
        ),
        (
            "partition_fingerprint",
            proxy.partition_fingerprint,
            dataset.partition_fingerprint,
        ),
        ("ld", proxy.ld, dataset.ld),
        ("num_system_qubits", proxy.num_system_qubits, dataset.num_system_qubits),
        ("delta_time", proxy.delta_time, dataset.delta_time),
        (
            "rte_steps_per_occurrence",
            proxy.rte_steps_per_occurrence,
            dataset.rte_steps_per_occurrence,
        ),
        (
            "finite_taylor_order",
            proxy.finite_taylor_order,
            dataset.finite_taylor_order,
        ),
        (
            "product_formula_order",
            proxy.product_formula_order,
            dataset.product_formula_order,
        ),
        ("control_convention", proxy.control_convention, dataset.control_convention),
        ("construction_policy", proxy.construction_policy, dataset.construction_policy),
        (
            "compiler_settings_fingerprint",
            proxy.compiler_settings_fingerprint,
            dataset.compiler_settings_fingerprint,
        ),
        ("backend_fingerprint", proxy.backend_fingerprint, dataset.backend_fingerprint),
        (
            "compiler_context_fingerprint",
            proxy.compiler_context_fingerprint,
            expected_compiler_context,
        ),
        ("circuit_scope", proxy.circuit_scope, dataset.circuit_scope),
    )
    return tuple(name for name, actual, expected in comparisons if actual != expected)


def _prediction_nonfinite_kind(value: float) -> PredictionNonfiniteKind:
    if math.isnan(value):
        return "nan"
    return "positive_infinity" if value > 0.0 else "negative_infinity"


def _build_validation_entries(
    proxy: RPEHadamardCompiledCostProxy,
    holdout_points: tuple[RPEHadamardProxyReferencePoint, ...],
    tolerances: tuple[RPEHadamardProxyMetricTolerance, ...],
) -> tuple[RPEHadamardProxyValidationEntry, ...]:
    tolerance_by_metric = {item.metric: item for item in tolerances}
    entries = []
    for point in sorted(holdout_points, key=lambda item: (item.q_m, item.axis)):
        for metric in RPE_COST_METRICS:
            tolerance = tolerance_by_metric[metric]
            observed = point.mean(metric)
            standard_error = point.standard_error(metric)
            standard_error_term = (
                0.0
                if standard_error is None
                else tolerance.standard_error_multiplier * standard_error
            )
            acceptance_limit = (
                tolerance.absolute_tolerance
                + tolerance.relative_tolerance * abs(observed)
                + standard_error_term
            )
            if not math.isfinite(acceptance_limit):
                raise ValueError("Validation acceptance limit must be finite.")
            predicted = proxy.predict(point.q_m, axis=point.axis, metric=metric)
            is_finite = math.isfinite(predicted)
            if is_finite:
                signed_error = predicted - observed
                absolute_error = abs(signed_error)
                relative_error = (
                    None if observed == 0.0 else absolute_error / abs(observed)
                )
                nonnegative = predicted >= 0.0
                nonfinite_kind = None
                serialized_prediction = predicted
            else:
                signed_error = None
                absolute_error = None
                relative_error = None
                nonnegative = False
                nonfinite_kind = _prediction_nonfinite_kind(predicted)
                serialized_prediction = None
            entries.append(
                RPEHadamardProxyValidationEntry(
                    partition="holdout",
                    round_index=point.round_index,
                    q_m=point.q_m,
                    axis=point.axis,
                    metric=metric,
                    point_fingerprint=point.point_fingerprint,
                    observed_mean=observed,
                    observed_standard_error=standard_error,
                    predicted_cost=serialized_prediction,
                    prediction_nonfinite_kind=nonfinite_kind,
                    signed_error=signed_error,
                    absolute_error=absolute_error,
                    relative_error=relative_error,
                    outside_calibration_range=(
                        point.q_m < proxy.calibration_q_m_min
                        or point.q_m > proxy.calibration_q_m_max
                    ),
                    prediction_is_finite=is_finite,
                    prediction_is_nonnegative=nonnegative,
                    acceptance_limit=acceptance_limit,
                    passed=(
                        is_finite
                        and nonnegative
                        and absolute_error is not None
                        and absolute_error <= acceptance_limit
                    ),
                )
            )
    return tuple(
        sorted(
            entries,
            key=lambda item: (
                item.q_m,
                ("cosine", "sine").index(item.axis),
                RPE_COST_METRICS.index(item.metric),
            ),
        )
    )


def validate_rpe_hadamard_compiled_cost_proxy(
    request: RPEHadamardCompiledCostProxyValidationRequest,
) -> RPEHadamardCompiledCostProxyValidationResult:
    """Validate fixed proxy coefficients against holdout points without refitting."""
    if not isinstance(request, RPEHadamardCompiledCostProxyValidationRequest):
        raise TypeError(
            "request must be an RPEHadamardCompiledCostProxyValidationRequest."
        )
    proxy = request.proxy
    dataset = request.dataset
    fit_fingerprint_before = proxy.fit_fingerprint
    models_before = proxy.models
    _require_complete_dataset(dataset)
    if not dataset.holdout_repetition_counts:
        raise ValueError("Proxy validation requires at least one holdout q_m.")
    mismatches = _configuration_mismatches(proxy, dataset)
    if mismatches:
        raise ValueError(
            "Proxy and holdout dataset configurations differ: "
            + ", ".join(mismatches)
        )
    if set(proxy.calibration_q_m_values).intersection(
        dataset.holdout_repetition_counts
    ):
        raise ValueError(
            "Holdout q_m must be disjoint from proxy calibration q_m."
        )
    if dataset.dataset_fingerprint != proxy.source_dataset_fingerprint:
        raise ValueError(
            "Proxy validation must use the proxy source benchmark dataset."
        )
    holdout_points = _reference_points_from_dataset(
        dataset,
        partition="holdout",
    )
    entries = _build_validation_entries(
        proxy,
        holdout_points,
        request.metric_tolerances,
    )
    if proxy.fit_fingerprint != fit_fingerprint_before or proxy.models != models_before:
        raise RuntimeError("Holdout validation modified fitted proxy coefficients.")
    return RPEHadamardCompiledCostProxyValidationResult(
        proxy=proxy,
        source_holdout_dataset_fingerprint=dataset.dataset_fingerprint,
        holdout_points=holdout_points,
        metric_tolerances=request.metric_tolerances,
        entries=entries,
    )
