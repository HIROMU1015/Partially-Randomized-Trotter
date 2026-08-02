"""Transpiled DF RTE event costs and finite-distribution expectations."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Sequence

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile

from .df_rte_circuit import DFRTEEventPreparation
from .df_rte_qiskit import (
    QiskitDFRTEEventCircuitBuilder,
    estimate_df_rte_untranspiled_size_upper_bound,
)
from .rte import (
    CircuitCost,
    CompilerSettings,
    FidelityLevel,
    PROBABILITY_ATOL,
    RTEConfig,
    RTEFiniteDistribution,
    enumerate_rte_events,
    require_integer_count,
)


CostEstimateKind = Literal[
    "exact_compiled_expectation",
    "monte_carlo_compiled_expectation",
]
_COST_METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)


def compiler_settings_hash(settings: CompilerSettings) -> str:
    """Return a canonical hash separating incompatible compiler conditions."""
    payload = {
        **asdict(settings),
        "compiler_hash_policy": "qiskit_compiler_settings_v1",
        "connectivity": (
            "explicit_coupling_map"
            if settings.coupling_map is not None
            else "all_to_all_logical"
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _numeric_parameter_payload(value: Any, *, allow_symbolic: bool = False) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Circuit parameters must be finite.")
        return {"float_hex": value.hex()}
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("Circuit parameters must be finite.")
        return {"complex_hex": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object-array circuit parameters are not cacheable.")
        return {
            "array_dtype": str(array.dtype),
            "array_shape": list(array.shape),
            "array_sha256": hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest(),
        }
    if isinstance(value, (tuple, list)):
        return [
            _numeric_parameter_payload(item, allow_symbolic=allow_symbolic)
            for item in value
        ]
    parameters = getattr(value, "parameters", None)
    if parameters:
        if not allow_symbolic:
            raise ValueError(
                "Symbolic circuits are not cacheable; bind all parameters first."
            )
        return {
            "symbolic_expression": str(value),
            "symbolic_parameters": sorted(str(parameter.name) for parameter in parameters),
        }
    try:
        numeric = complex(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Unsupported circuit parameter type: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        ) from exc
    if numeric.imag == 0.0:
        return _numeric_parameter_payload(float(numeric.real))
    return _numeric_parameter_payload(numeric)


def _condition_payload(condition: Any, circuit: QuantumCircuit) -> Any:
    if condition is None:
        return None
    if not isinstance(condition, tuple) or len(condition) != 2:
        raise TypeError("Unsupported classical condition is not cacheable.")
    target, value = condition
    if hasattr(target, "bits"):
        bits = [circuit.find_bit(bit).index for bit in target.bits]
    else:
        bits = [circuit.find_bit(target).index]
    return {"clbits": bits, "value": int(value)}


def _operation_payload(
    operation: Any,
    circuit: QuantumCircuit,
    *,
    active_definitions: set[int],
    allow_symbolic: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": f"{type(operation).__module__}.{type(operation).__qualname__}",
        "name": str(operation.name),
        "num_qubits": int(operation.num_qubits),
        "num_clbits": int(operation.num_clbits),
        "params": [
            _numeric_parameter_payload(parameter, allow_symbolic=allow_symbolic)
            for parameter in operation.params
        ],
        "condition": _condition_payload(getattr(operation, "condition", None), circuit),
    }
    for name in ("num_ctrl_qubits", "ctrl_state"):
        if hasattr(operation, name):
            payload[name] = int(getattr(operation, name))
    base_gate = getattr(operation, "base_gate", None)
    if base_gate is not None:
        payload["base_gate"] = _operation_payload(
            base_gate,
            circuit,
            active_definitions=active_definitions,
            allow_symbolic=allow_symbolic,
        )
    blocks = getattr(operation, "blocks", ())
    if blocks:
        payload["control_flow_blocks"] = [
            _circuit_payload(
                block,
                active_definitions=active_definitions,
                allow_symbolic=allow_symbolic,
            )
            for block in blocks
        ]
    definition = getattr(operation, "definition", None)
    if definition is not None:
        identifier = id(definition)
        if identifier in active_definitions:
            raise ValueError("Recursive custom-gate definitions are not cacheable.")
        active_definitions.add(identifier)
        try:
            payload["definition"] = _circuit_payload(
                definition,
                active_definitions=active_definitions,
                allow_symbolic=allow_symbolic,
            )
        finally:
            active_definitions.remove(identifier)
    return payload


def _circuit_payload(
    circuit: QuantumCircuit,
    *,
    active_definitions: set[int] | None = None,
    allow_symbolic: bool = False,
) -> dict[str, Any]:
    if active_definitions is None:
        active_definitions = set()
    if circuit.parameters and not allow_symbolic:
        raise ValueError("Symbolic circuits are not cacheable; bind all parameters first.")
    if getattr(circuit, "layout", None) is not None:
        raise ValueError("Circuits carrying transpiler layout metadata are not cacheable.")
    calibrations = getattr(circuit, "calibrations", {})
    if calibrations:
        raise ValueError("Circuits with pulse calibrations are not cacheable.")
    instructions = []
    for item in circuit.data:
        instructions.append(
            {
                "operation": _operation_payload(
                    item.operation,
                    circuit,
                    active_definitions=active_definitions,
                    allow_symbolic=allow_symbolic,
                ),
                "qubits": [circuit.find_bit(qubit).index for qubit in item.qubits],
                "clbits": [circuit.find_bit(clbit).index for clbit in item.clbits],
            }
        )
    return {
        "num_qubits": int(circuit.num_qubits),
        "num_clbits": int(circuit.num_clbits),
        "global_phase": _numeric_parameter_payload(
            circuit.global_phase,
            allow_symbolic=allow_symbolic,
        ),
        "instructions": instructions,
        "fingerprint_policy": "qiskit_recursive_numeric_circuit_v2",
    }


def canonical_qiskit_circuit_fingerprint(circuit: QuantumCircuit) -> str:
    """Fingerprint the complete semantics of a fully numeric Qiskit circuit."""
    payload = _circuit_payload(circuit)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TranspiledCircuitCost:
    """Integer gate metrics measured from one actually transpiled circuit."""

    pretranspile_gate_counts: tuple[tuple[str, int], ...]
    posttranspile_gate_counts: tuple[tuple[str, int], ...]
    rz_count: int
    rz_depth: int
    cx_count: int
    cx_depth: int
    total_depth: int
    circuit_size: int
    qubit_count: int
    global_phase: float
    compiler: CompilerSettings
    compiler_settings_hash: str
    circuit_fingerprint: str
    actual_circuit_fingerprint: str | None
    backend_fingerprint: str | None
    transpiled_circuit: QuantumCircuit | None = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "rz_count",
            "rz_depth",
            "cx_count",
            "cx_depth",
            "total_depth",
            "circuit_size",
            "qubit_count",
        ):
            object.__setattr__(
                self,
                name,
                require_integer_count(getattr(self, name), name=name),
            )


@dataclass(frozen=True)
class CompiledMetricStatistics:
    """Expectation or Monte Carlo statistics for one compiled metric."""

    mean: float
    unbiased_sample_variance: float | None
    standard_error: float | None
    minimum: float
    maximum: float


@dataclass(frozen=True)
class CompiledEventCostEstimate:
    """Exact-enumeration or Monte Carlo compiled single-event expectation."""

    estimate_kind: CostEstimateKind
    expected_cost: CircuitCost
    standard_error: CircuitCost | None
    metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...]
    sample_count: int | None
    enumerated_event_count: int | None
    unique_compiled_circuit_count: int
    transpile_cache_hit_count: int
    transpile_cache_miss_count: int
    transpile_cache_bypass_count: int
    transpile_cache_eviction_count: int
    transpile_cache_maximum_entries: int
    compiler: CompilerSettings
    controlled: bool
    ancilla_qubit: int | None
    event_stream_rolling_digest: str
    basis_reuse_policy: Literal["disabled", "raw_adjacent_equal_basis"]
    seed: int | None
    event_probability_sum: float | None
    maximum_work_items: int
    workload_limit_kind: Literal["max_events", "maximum_samples"]
    statistics_policy: Literal["online_welford_v1"] = "online_welford_v1"


class TranspiledCircuitCostCache:
    """Bounded LRU cache keyed by actual circuit, compiler, and backend semantics."""

    def __init__(
        self,
        *,
        maximum_entries: int = 256,
        retain_transpiled_circuits: bool = False,
    ) -> None:
        self.maximum_entries = require_integer_count(
            maximum_entries,
            name="maximum_entries",
            minimum=1,
        )
        self._costs: OrderedDict[str, TranspiledCircuitCost] = OrderedDict()
        self.retain_transpiled_circuits = bool(retain_transpiled_circuits)
        self.hit_count = 0
        self.miss_count = 0
        self.bypass_count = 0
        self.eviction_count = 0

    def get_or_transpile(
        self,
        circuit: QuantumCircuit,
        compiler: CompilerSettings,
        *,
        circuit_fingerprint: str,
        backend: Any | None = None,
    ) -> tuple[TranspiledCircuitCost, str, bool]:
        _validate_transpile_context(compiler, backend)
        settings_hash = compiler_settings_hash(compiler)
        try:
            actual_fingerprint = canonical_qiskit_circuit_fingerprint(circuit)
        except (TypeError, ValueError):
            actual_fingerprint = None
        backend_fingerprint = _canonical_backend_fingerprint_or_none(backend)
        cacheable = actual_fingerprint is not None and (
            backend is None or backend_fingerprint is not None
        )
        key = hashlib.sha256(
            json.dumps(
                {
                    "actual_circuit_fingerprint": actual_fingerprint,
                    "compiler_settings_hash": settings_hash,
                    "backend_fingerprint": backend_fingerprint,
                    "cacheable": cacheable,
                    "bypass_object_identity": None if cacheable else id(circuit),
                    "cache_key_policy": "actual_circuit_compiler_backend_v2",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        cached = self._costs.get(key) if cacheable else None
        if cached is not None:
            self._costs.move_to_end(key)
            self.hit_count += 1
            return replace(cached, circuit_fingerprint=circuit_fingerprint), key, True
        cost = transpile_and_measure_cost(
            circuit,
            compiler,
            circuit_fingerprint=circuit_fingerprint,
            backend=backend,
            actual_circuit_fingerprint=actual_fingerprint,
            backend_fingerprint=backend_fingerprint,
        )
        self.miss_count += 1
        if cacheable:
            # Compiled circuits can dominate memory for trajectory studies.
            # Cache only metrics by default; the miss result still returns the
            # actual circuit so callers may retain a few selected diagnostics.
            self._costs[key] = (
                cost
                if self.retain_transpiled_circuits
                else replace(cost, transpiled_circuit=None)
            )
            self._costs.move_to_end(key)
            if len(self._costs) > self.maximum_entries:
                self._costs.popitem(last=False)
                self.eviction_count += 1
        else:
            self.bypass_count += 1
        return cost, key, False

    def __len__(self) -> int:
        return len(self._costs)


def _backend_name(backend: Any) -> str:
    candidate = getattr(backend, "name", None)
    return str(candidate() if callable(candidate) else candidate)


def _validate_transpile_context(
    compiler: CompilerSettings,
    backend: Any | None,
) -> None:
    if compiler.qiskit_version != qiskit.__version__:
        raise ValueError(
            "CompilerSettings.qiskit_version does not match the active Qiskit."
        )
    if compiler.backend_name is not None:
        if backend is None:
            raise ValueError("A named backend requires the corresponding backend object.")
        if _backend_name(backend) != compiler.backend_name:
            raise ValueError("The supplied backend does not match backend_name.")
    elif backend is not None:
        raise ValueError("An explicit backend requires backend_name in compiler settings.")


def _instruction_properties_payload(properties: Any) -> dict[str, Any] | None:
    if properties is None:
        return None
    calibration = getattr(properties, "calibration", None)
    if calibration is not None:
        raise ValueError("Backends with pulse calibrations are not cacheable.")
    payload: dict[str, Any] = {}
    for name in ("duration", "error"):
        value = getattr(properties, name, None)
        payload[name] = None if value is None else _numeric_parameter_payload(float(value))
    return payload


def _canonical_backend_fingerprint(backend: Any) -> str:
    target = getattr(backend, "target", None)
    if target is None:
        raise ValueError("A backend without a canonical Target is not cacheable.")
    operation_records = []
    for operation_name in sorted(str(name) for name in target.operation_names):
        operation = target.operation_from_name(operation_name)
        qarg_properties = target[operation_name]
        qarg_records = []
        for qargs, properties in sorted(
            qarg_properties.items(),
            key=lambda item: () if item[0] is None else tuple(item[0]),
        ):
            qarg_records.append(
                {
                    "qargs": None if qargs is None else [int(index) for index in qargs],
                    "properties": _instruction_properties_payload(properties),
                }
            )
        operation_records.append(
            {
                "name": operation_name,
                "type": f"{type(operation).__module__}.{type(operation).__qualname__}",
                "num_qubits": int(operation.num_qubits),
                "num_clbits": int(operation.num_clbits),
                "params": [
                    _numeric_parameter_payload(parameter, allow_symbolic=True)
                    for parameter in operation.params
                ],
                "qargs": qarg_records,
            }
        )
    qubit_records = []
    for properties in getattr(target, "qubit_properties", None) or ():
        qubit_records.append(
            {
                name: (
                    None
                    if getattr(properties, name, None) is None
                    else _numeric_parameter_payload(float(getattr(properties, name)))
                )
                for name in ("t1", "t2", "frequency")
            }
        )
    target_fields = {}
    for name in (
        "num_qubits",
        "dt",
        "granularity",
        "min_length",
        "pulse_alignment",
        "acquire_alignment",
    ):
        value = getattr(target, name, None)
        target_fields[name] = (
            None if value is None else _numeric_parameter_payload(value)
        )
    payload = {
        "backend_name": _backend_name(backend),
        "backend_version": str(getattr(backend, "backend_version", "")),
        "target_fields": target_fields,
        "qubit_properties": qubit_records,
        "operations": operation_records,
        "backend_fingerprint_policy": "qiskit_target_v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_backend_fingerprint_or_none(backend: Any | None) -> str | None:
    if backend is None:
        return "no_backend"
    try:
        return _canonical_backend_fingerprint(backend)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _gate_counts(circuit: QuantumCircuit) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted((str(name).lower(), int(count)) for name, count in circuit.count_ops().items())
    )


def _filtered_depth(circuit: QuantumCircuit, gate_name: str) -> int:
    depth = circuit.depth(
        filter_function=lambda item: item.operation.name.lower() == gate_name
    )
    return int(depth or 0)


def transpile_and_measure_cost(
    circuit: QuantumCircuit,
    compiler: CompilerSettings,
    *,
    circuit_fingerprint: str | None = None,
    backend: Any | None = None,
    actual_circuit_fingerprint: str | None = None,
    backend_fingerprint: str | None = None,
) -> TranspiledCircuitCost:
    """Transpile one circuit under fully recorded settings and measure counts."""
    _validate_transpile_context(compiler, backend)
    if actual_circuit_fingerprint is None:
        try:
            actual_circuit_fingerprint = canonical_qiskit_circuit_fingerprint(circuit)
        except (TypeError, ValueError):
            actual_circuit_fingerprint = None
    if backend_fingerprint is None:
        backend_fingerprint = _canonical_backend_fingerprint_or_none(backend)
    kwargs: dict[str, Any] = {
        "basis_gates": list(compiler.basis_gates),
        "optimization_level": compiler.optimization_level,
        "seed_transpiler": compiler.transpiler_seed,
    }
    if compiler.coupling_map is not None:
        kwargs["coupling_map"] = list(compiler.coupling_map)
    if compiler.layout_method is not None:
        kwargs["layout_method"] = compiler.layout_method
    if compiler.routing_method is not None:
        kwargs["routing_method"] = compiler.routing_method
    if backend is not None:
        kwargs["backend"] = backend
    transpiled_circuit = transpile(circuit, **kwargs)
    post_counts = dict(_gate_counts(transpiled_circuit))
    fingerprint = circuit_fingerprint or actual_circuit_fingerprint or "uncacheable-circuit"
    return TranspiledCircuitCost(
        pretranspile_gate_counts=_gate_counts(circuit),
        posttranspile_gate_counts=tuple(sorted(post_counts.items())),
        rz_count=post_counts.get("rz", 0),
        rz_depth=_filtered_depth(transpiled_circuit, "rz"),
        cx_count=post_counts.get("cx", 0),
        cx_depth=_filtered_depth(transpiled_circuit, "cx"),
        total_depth=int(transpiled_circuit.depth() or 0),
        circuit_size=int(transpiled_circuit.size()),
        qubit_count=transpiled_circuit.num_qubits,
        global_phase=float(transpiled_circuit.global_phase),
        compiler=compiler,
        compiler_settings_hash=compiler_settings_hash(compiler),
        circuit_fingerprint=fingerprint,
        actual_circuit_fingerprint=actual_circuit_fingerprint,
        backend_fingerprint=backend_fingerprint,
        transpiled_circuit=transpiled_circuit,
    )


def _metric_values(cost: Any) -> tuple[float, ...]:
    return tuple(float(getattr(cost, name)) for name in _COST_METRICS)


class CompiledMetricAccumulator:
    """Online compiled-metric statistics for bounded-memory studies."""

    def __init__(self, *, weighted: bool) -> None:
        self.weighted = bool(weighted)
        self.count = 0
        self.total_weight = 0.0
        self._means = [0.0] * len(_COST_METRICS)
        self._m2 = [0.0] * len(_COST_METRICS)
        self._minimums = [math.inf] * len(_COST_METRICS)
        self._maximums = [-math.inf] * len(_COST_METRICS)

    def update(self, cost: Any, *, weight: float | None = None) -> None:
        if self.weighted:
            if weight is None:
                raise ValueError("A weighted metric accumulator requires a weight.")
            normalized_weight = float(weight)
            if not math.isfinite(normalized_weight) or normalized_weight < 0.0:
                raise ValueError("Metric weights must be finite and non-negative.")
        elif weight is not None:
            raise ValueError("An unweighted metric accumulator does not accept weights.")
        else:
            normalized_weight = 1.0

        values = _metric_values(cost)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Compiled metrics must be finite.")
        self.count += 1
        previous_weight = self.total_weight
        self.total_weight += normalized_weight
        for index, value in enumerate(values):
            self._minimums[index] = min(self._minimums[index], value)
            self._maximums[index] = max(self._maximums[index], value)
            if normalized_weight == 0.0:
                continue
            delta = value - self._means[index]
            self._means[index] += (
                normalized_weight / self.total_weight
            ) * delta
            if not self.weighted:
                self._m2[index] += delta * (value - self._means[index])
        if self.total_weight < previous_weight:
            raise RuntimeError("Metric weight accumulation overflowed.")

    def finalize(self) -> tuple[tuple[str, CompiledMetricStatistics], ...]:
        if self.count == 0 or self.total_weight <= 0.0:
            raise ValueError("At least one positive-weight compiled metric is required.")
        statistics: list[tuple[str, CompiledMetricStatistics]] = []
        for index, name in enumerate(_COST_METRICS):
            if self.weighted or self.count == 1:
                variance = None
                standard_error = None
            else:
                variance = self._m2[index] / (self.count - 1)
                standard_error = math.sqrt(variance / self.count)
            statistics.append(
                (
                    name,
                    CompiledMetricStatistics(
                        mean=float(self._means[index]),
                        unbiased_sample_variance=(
                            None if variance is None else float(variance)
                        ),
                        standard_error=(
                            None
                            if standard_error is None
                            else float(standard_error)
                        ),
                        minimum=float(self._minimums[index]),
                        maximum=float(self._maximums[index]),
                    ),
                )
            )
        return tuple(statistics)


def _metric_statistics(
    costs: Sequence[Any],
    *,
    weights: Sequence[float] | None,
) -> tuple[tuple[str, CompiledMetricStatistics], ...]:
    if not costs:
        raise ValueError("At least one transpiled circuit cost is required.")
    if weights is not None and len(weights) != len(costs):
        raise ValueError("Metric weights length must match compiled costs.")
    accumulator = CompiledMetricAccumulator(weighted=weights is not None)
    if weights is None:
        for cost in costs:
            accumulator.update(cost)
    else:
        for cost, weight in zip(costs, weights, strict=True):
            accumulator.update(cost, weight=weight)
    return accumulator.finalize()


def compiled_metric_statistics(
    costs: Sequence[Any],
    *,
    weights: Sequence[float] | None,
) -> tuple[tuple[str, CompiledMetricStatistics], ...]:
    """Public shared statistics implementation for higher fidelity levels."""
    return _metric_statistics(costs, weights=weights)


def _circuit_cost_from_statistics(
    statistics: tuple[tuple[str, CompiledMetricStatistics], ...],
    *,
    compiler: CompilerSettings,
    estimate_kind: str,
    use_standard_error: bool = False,
    fidelity_level: FidelityLevel = 3,
) -> CircuitCost | None:
    mapping = dict(statistics)
    if use_standard_error:
        if any(mapping[name].standard_error is None for name in _COST_METRICS):
            return None
        values = {
            name: float(mapping[name].standard_error) for name in _COST_METRICS
        }
        kind = "compiled_cost_standard_error"
    else:
        values = {name: float(mapping[name].mean) for name in _COST_METRICS}
        kind = estimate_kind
    return CircuitCost(
        rz_count=values["rz_count"],
        rz_depth=values["rz_depth"],
        cx_count=values["cx_count"],
        cx_depth=values["cx_depth"],
        total_depth=values["total_depth"],
        circuit_size=values["circuit_size"],
        compiler=compiler,
        fidelity_level=fidelity_level,
        estimate_kind=kind,
    )


def circuit_cost_from_metric_statistics(
    statistics: tuple[tuple[str, CompiledMetricStatistics], ...],
    *,
    compiler: CompilerSettings,
    estimate_kind: str,
    use_standard_error: bool = False,
    fidelity_level: FidelityLevel = 3,
) -> CircuitCost | None:
    """Create a floating expectation/standard-error record from statistics."""
    return _circuit_cost_from_statistics(
        statistics,
        compiler=compiler,
        estimate_kind=estimate_kind,
        use_standard_error=use_standard_error,
        fidelity_level=fidelity_level,
    )


def _estimate_event_costs(
    preparation: DFRTEEventPreparation,
    compiler: CompilerSettings,
    *,
    events: Sequence[Any],
    estimate_kind: CostEstimateKind,
    weights: Sequence[float] | None,
    event_probability_sum: float | None,
    controlled: bool,
    ancilla_qubit: int | None,
    cancel_adjacent_equal_bases: bool,
    seed: int | None,
    maximum_work_items: int,
    workload_limit_kind: Literal["max_events", "maximum_samples"],
    cache: TranspiledCircuitCostCache | None,
    backend: Any | None,
) -> CompiledEventCostEstimate:
    working_cache = cache if cache is not None else TranspiledCircuitCostCache()
    initial_misses = working_cache.miss_count
    initial_bypasses = working_cache.bypass_count
    initial_evictions = working_cache.eviction_count
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    if weights is not None and len(weights) != len(events):
        raise ValueError("Event weights length must match events.")
    accumulator = CompiledMetricAccumulator(weighted=weights is not None)
    event_digest = hashlib.sha256()
    keys: set[str] = set()
    cache_hits = 0
    for index, event in enumerate(events):
        encoded_event = json.dumps(
            event.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        event_digest.update(len(encoded_event).to_bytes(8, "big"))
        event_digest.update(encoded_event)
        request = preparation.request_for_event(
            event,
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
        )
        built = builder.build_event(request)
        cost, key, was_cached = working_cache.get_or_transpile(
            built.circuit,
            compiler,
            circuit_fingerprint=built.circuit_fingerprint,
            backend=backend,
        )
        if weights is None:
            accumulator.update(cost)
        else:
            accumulator.update(cost, weight=weights[index])
        keys.add(key)
        cache_hits += int(was_cached)
    statistics = accumulator.finalize()
    expected = _circuit_cost_from_statistics(
        statistics,
        compiler=compiler,
        estimate_kind=estimate_kind,
    )
    if expected is None:
        raise RuntimeError("Compiled expected cost could not be constructed.")
    standard_error = _circuit_cost_from_statistics(
        statistics,
        compiler=compiler,
        estimate_kind=estimate_kind,
        use_standard_error=True,
    )
    is_exact = estimate_kind == "exact_compiled_expectation"
    return CompiledEventCostEstimate(
        estimate_kind=estimate_kind,
        expected_cost=expected,
        standard_error=standard_error,
        metric_statistics=statistics,
        sample_count=None if is_exact else len(events),
        enumerated_event_count=len(events) if is_exact else None,
        unique_compiled_circuit_count=len(keys),
        transpile_cache_hit_count=cache_hits,
        transpile_cache_miss_count=working_cache.miss_count - initial_misses,
        transpile_cache_bypass_count=(
            working_cache.bypass_count - initial_bypasses
        ),
        transpile_cache_eviction_count=(
            working_cache.eviction_count - initial_evictions
        ),
        transpile_cache_maximum_entries=working_cache.maximum_entries,
        compiler=compiler,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        event_stream_rolling_digest=event_digest.hexdigest(),
        basis_reuse_policy=(
            "raw_adjacent_equal_basis"
            if cancel_adjacent_equal_bases
            else "disabled"
        ),
        seed=None if is_exact else seed,
        event_probability_sum=event_probability_sum,
        maximum_work_items=maximum_work_items,
        workload_limit_kind=workload_limit_kind,
    )


def estimate_exact_compiled_event_cost(
    preparation: DFRTEEventPreparation,
    distribution: RTEFiniteDistribution,
    compiler: CompilerSettings,
    *,
    controlled: bool = False,
    ancilla_qubit: int | None = None,
    cancel_adjacent_equal_bases: bool = True,
    max_events: int = 10_000,
    cache: TranspiledCircuitCostCache | None = None,
    backend: Any | None = None,
) -> CompiledEventCostEstimate:
    """Enumerate and probability-weight every finite RTE event exactly."""
    events = enumerate_rte_events(
        preparation.symbolic_tail.components,
        distribution,
        max_events=max_events,
    )
    weights = tuple(event.event_probability for event in events)
    probability_sum = math.fsum(weights)
    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_ATOL,
    ):
        raise ValueError("Enumerated RTE event probabilities must sum to one.")
    normalized_weights = tuple(weight / probability_sum for weight in weights)
    return _estimate_event_costs(
        preparation,
        compiler,
        events=events,
        estimate_kind="exact_compiled_expectation",
        weights=normalized_weights,
        event_probability_sum=probability_sum,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
        seed=None,
        maximum_work_items=max_events,
        workload_limit_kind="max_events",
        cache=cache,
        backend=backend,
    )


def estimate_monte_carlo_compiled_event_cost(
    preparation: DFRTEEventPreparation,
    distribution: RTEFiniteDistribution,
    compiler: CompilerSettings,
    *,
    sample_count: int,
    seed: int,
    maximum_samples: int = 10_000,
    controlled: bool = False,
    ancilla_qubit: int | None = None,
    cancel_adjacent_equal_bases: bool = True,
    cache: TranspiledCircuitCostCache | None = None,
    backend: Any | None = None,
) -> CompiledEventCostEstimate:
    """Estimate compiled event cost by an unweighted classical sample mean."""
    sample_count = require_integer_count(
        sample_count,
        name="sample_count",
        minimum=1,
    )
    maximum_samples = require_integer_count(
        maximum_samples,
        name="maximum_samples",
        minimum=1,
    )
    if sample_count > maximum_samples:
        raise ValueError(
            f"sample_count={sample_count} exceeds maximum_samples={maximum_samples}."
        )
    events = preparation.sample_events(
        distribution,
        sample_count=sample_count,
        seed=seed,
    )
    return _estimate_event_costs(
        preparation,
        compiler,
        events=events,
        estimate_kind="monte_carlo_compiled_expectation",
        weights=None,
        event_probability_sum=None,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
        seed=seed,
        maximum_work_items=maximum_samples,
        workload_limit_kind="maximum_samples",
        cache=cache,
        backend=backend,
    )


@dataclass(frozen=True)
class _MetricVector:
    rz_count: float
    rz_depth: float
    cx_count: float
    cx_depth: float
    total_depth: float
    circuit_size: float


@dataclass(frozen=True)
class CompiledSequenceCostEstimate:
    """Monte Carlo compiled cost of complete short RTE occurrences."""

    sequence_expected_cost: CircuitCost
    additive_expected_cost: CircuitCost
    nonadditive_difference: CircuitCost
    sequence_standard_error: CircuitCost | None
    additive_standard_error: CircuitCost | None
    difference_standard_error: CircuitCost | None
    sequence_metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...]
    additive_metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...]
    difference_metric_statistics: tuple[tuple[str, CompiledMetricStatistics], ...]
    sample_count: int
    event_count_per_sample: int
    unique_compiled_circuit_count: int
    unique_sequence_circuit_count: int
    transpile_cache_hit_count: int
    transpile_cache_miss_count: int
    transpile_cache_bypass_count: int
    transpile_cache_eviction_count: int
    transpile_cache_maximum_entries: int
    compiler: CompilerSettings
    controlled: bool
    ancilla_qubit: int | None
    event_stream_rolling_digest: str
    basis_reuse_policy: Literal["disabled", "raw_adjacent_equal_basis"]
    seed: int
    maximum_rte_steps: int
    maximum_untranspiled_circuit_size: int
    maximum_samples: int
    statistics_policy: Literal["online_welford_v1"] = "online_welford_v1"


def _sum_costs(costs: Sequence[TranspiledCircuitCost]) -> _MetricVector:
    return _MetricVector(
        **{
            name: math.fsum(float(getattr(cost, name)) for cost in costs)
            for name in _COST_METRICS
        }
    )


def _subtract_costs(left: Any, right: Any) -> _MetricVector:
    return _MetricVector(
        **{
            name: float(getattr(left, name)) - float(getattr(right, name))
            for name in _COST_METRICS
        }
    )


def sum_compiled_costs(costs: Sequence[TranspiledCircuitCost]) -> _MetricVector:
    """Sum actual compiled metrics without duplicating the metric schema."""
    return _sum_costs(costs)


def subtract_compiled_costs(left: Any, right: Any) -> _MetricVector:
    """Subtract two actual or aggregate compiled metric records."""
    return _subtract_costs(left, right)


def estimate_compiled_occurrence_cost(
    preparation: DFRTEEventPreparation,
    config: RTEConfig,
    distribution: RTEFiniteDistribution,
    compiler: CompilerSettings,
    *,
    sequence_sample_count: int,
    seed: int,
    maximum_samples: int = 10_000,
    controlled: bool = False,
    ancilla_qubit: int | None = None,
    cancel_adjacent_equal_bases: bool = True,
    maximum_rte_steps: int = 16,
    maximum_untranspiled_circuit_size: int = 100_000,
    cache: TranspiledCircuitCostCache | None = None,
    backend: Any | None = None,
) -> CompiledSequenceCostEstimate:
    """Compile short sampled occurrences and compare additive event costs."""
    sequence_sample_count = require_integer_count(
        sequence_sample_count,
        name="sequence_sample_count",
        minimum=1,
    )
    maximum_samples = require_integer_count(
        maximum_samples,
        name="maximum_samples",
        minimum=1,
    )
    if sequence_sample_count > maximum_samples:
        raise ValueError(
            "sequence_sample_count exceeds the configured maximum_samples."
        )
    seed = require_integer_count(seed, name="seed")
    maximum_rte_steps = require_integer_count(
        maximum_rte_steps,
        name="maximum_rte_steps",
        minimum=1,
    )
    maximum_untranspiled_circuit_size = require_integer_count(
        maximum_untranspiled_circuit_size,
        name="maximum_untranspiled_circuit_size",
        minimum=1,
    )
    if config.rte_steps > maximum_rte_steps:
        raise ValueError(
            "Configured rte_steps exceeds maximum_rte_steps for sequence expansion."
        )
    working_cache = cache if cache is not None else TranspiledCircuitCostCache()
    initial_misses = working_cache.miss_count
    initial_bypasses = working_cache.bypass_count
    initial_evictions = working_cache.eviction_count
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    rng = random.Random(seed)
    sequence_accumulator = CompiledMetricAccumulator(weighted=False)
    additive_accumulator = CompiledMetricAccumulator(weighted=False)
    difference_accumulator = CompiledMetricAccumulator(weighted=False)
    all_keys: set[str] = set()
    sequence_keys: set[str] = set()
    cache_hits = 0
    event_digest = hashlib.sha256()

    for _ in range(sequence_sample_count):
        occurrence_seed = rng.randrange(0, 2**63)
        request = preparation.sample_occurrence_request(
            config,
            distribution,
            seed=occurrence_seed,
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
        )
        for event in request.events:
            encoded_event = json.dumps(
                event.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            event_digest.update(len(encoded_event).to_bytes(8, "big"))
            event_digest.update(encoded_event)
        planned_sequence_size = estimate_df_rte_untranspiled_size_upper_bound(
            request
        )
        if planned_sequence_size > maximum_untranspiled_circuit_size:
            raise ValueError(
                "Planned RTE occurrence circuit upper bound exceeds the configured "
                "size limit before circuit construction."
            )
        built_sequence = builder.build_sequence(request)
        if built_sequence.circuit.size() > maximum_untranspiled_circuit_size:
            raise ValueError(
                "Untranspiled occurrence circuit exceeds the configured size limit."
            )
        sequence_cost, sequence_key, was_cached = working_cache.get_or_transpile(
            built_sequence.circuit,
            compiler,
            circuit_fingerprint=built_sequence.circuit_fingerprint,
            backend=backend,
        )
        sequence_accumulator.update(sequence_cost)
        all_keys.add(sequence_key)
        sequence_keys.add(sequence_key)
        cache_hits += int(was_cached)

        event_costs: list[TranspiledCircuitCost] = []
        for event in request.events:
            event_request = preparation.request_for_event(
                event,
                controlled=controlled,
                ancilla_qubit=ancilla_qubit,
                cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
            )
            planned_event_size = estimate_df_rte_untranspiled_size_upper_bound(
                event_request
            )
            if planned_event_size > maximum_untranspiled_circuit_size:
                raise ValueError(
                    "Planned RTE event circuit upper bound exceeds the configured "
                    "size limit before circuit construction."
                )
            built_event = builder.build_event(event_request)
            if built_event.circuit.size() > maximum_untranspiled_circuit_size:
                raise ValueError(
                    "Untranspiled RTE event circuit exceeds the configured size limit."
                )
            event_cost, event_key, event_cached = working_cache.get_or_transpile(
                built_event.circuit,
                compiler,
                circuit_fingerprint=built_event.circuit_fingerprint,
                backend=backend,
            )
            event_costs.append(event_cost)
            all_keys.add(event_key)
            cache_hits += int(event_cached)
        additive = _sum_costs(event_costs)
        additive_accumulator.update(additive)
        difference_accumulator.update(_subtract_costs(sequence_cost, additive))

    sequence_statistics = sequence_accumulator.finalize()
    additive_statistics = additive_accumulator.finalize()
    difference_statistics = difference_accumulator.finalize()

    def make_cost(
        statistics: tuple[tuple[str, CompiledMetricStatistics], ...],
        *,
        estimate_kind: str,
        standard_error: bool = False,
    ) -> CircuitCost | None:
        return _circuit_cost_from_statistics(
            statistics,
            compiler=compiler,
            estimate_kind=estimate_kind,
            use_standard_error=standard_error,
            fidelity_level=4,
        )

    sequence_expected = make_cost(
        sequence_statistics,
        estimate_kind="monte_carlo_compiled_sequence_expectation",
    )
    additive_expected = make_cost(
        additive_statistics,
        estimate_kind="monte_carlo_compiled_sequence_expectation",
    )
    difference_expected = make_cost(
        difference_statistics,
        estimate_kind="compiled_sequence_nonadditive_difference",
    )
    if (
        sequence_expected is None
        or additive_expected is None
        or difference_expected is None
    ):
        raise RuntimeError("Compiled sequence expectation could not be constructed.")
    return CompiledSequenceCostEstimate(
        sequence_expected_cost=sequence_expected,
        additive_expected_cost=additive_expected,
        nonadditive_difference=difference_expected,
        sequence_standard_error=make_cost(
            sequence_statistics,
            estimate_kind="monte_carlo_compiled_sequence_expectation",
            standard_error=True,
        ),
        additive_standard_error=make_cost(
            additive_statistics,
            estimate_kind="monte_carlo_compiled_sequence_expectation",
            standard_error=True,
        ),
        difference_standard_error=make_cost(
            difference_statistics,
            estimate_kind="compiled_sequence_nonadditive_difference",
            standard_error=True,
        ),
        sequence_metric_statistics=sequence_statistics,
        additive_metric_statistics=additive_statistics,
        difference_metric_statistics=difference_statistics,
        sample_count=sequence_sample_count,
        event_count_per_sample=config.rte_steps,
        unique_compiled_circuit_count=len(all_keys),
        unique_sequence_circuit_count=len(sequence_keys),
        transpile_cache_hit_count=cache_hits,
        transpile_cache_miss_count=working_cache.miss_count - initial_misses,
        transpile_cache_bypass_count=(
            working_cache.bypass_count - initial_bypasses
        ),
        transpile_cache_eviction_count=(
            working_cache.eviction_count - initial_evictions
        ),
        transpile_cache_maximum_entries=working_cache.maximum_entries,
        compiler=compiler,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        event_stream_rolling_digest=event_digest.hexdigest(),
        basis_reuse_policy=(
            "raw_adjacent_equal_basis"
            if cancel_adjacent_equal_bases
            else "disabled"
        ),
        seed=seed,
        maximum_rte_steps=maximum_rte_steps,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
        maximum_samples=maximum_samples,
    )
