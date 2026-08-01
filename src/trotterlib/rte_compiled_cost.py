"""Transpiled DF RTE event costs and finite-distribution expectations."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import qiskit
from qiskit import QuantumCircuit, transpile

from .df_rte_circuit import DFRTEEventPreparation
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .rte import (
    CircuitCost,
    CompilerSettings,
    FidelityLevel,
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


def canonical_qiskit_circuit_fingerprint(circuit: QuantumCircuit) -> str:
    """Fingerprint a numeric circuit without materializing its unitary."""
    instructions = []
    for item in circuit.data:
        instructions.append(
            {
                "name": item.operation.name,
                "params": [str(parameter) for parameter in item.operation.params],
                "qubits": [circuit.find_bit(qubit).index for qubit in item.qubits],
                "clbits": [circuit.find_bit(clbit).index for clbit in item.clbits],
            }
        )
    payload = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "global_phase": str(circuit.global_phase),
        "instructions": instructions,
        "fingerprint_policy": "qiskit_circuit_structure_v1",
    }
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
    transpiled_circuit: QuantumCircuit = field(repr=False, compare=False)

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
    compiler: CompilerSettings
    controlled: bool
    basis_reuse_policy: Literal["disabled", "raw_adjacent_equal_basis"]
    seed: int | None
    event_probability_sum: float | None


class TranspiledCircuitCostCache:
    """In-memory cache keyed by circuit semantics and compiler settings."""

    def __init__(self) -> None:
        self._costs: dict[str, TranspiledCircuitCost] = {}
        self.hit_count = 0
        self.miss_count = 0

    def get_or_transpile(
        self,
        circuit: QuantumCircuit,
        compiler: CompilerSettings,
        *,
        circuit_fingerprint: str,
        backend: Any | None = None,
    ) -> tuple[TranspiledCircuitCost, str, bool]:
        settings_hash = compiler_settings_hash(compiler)
        key = hashlib.sha256(
            f"{circuit_fingerprint}:{settings_hash}".encode()
        ).hexdigest()
        cached = self._costs.get(key)
        if cached is not None:
            self.hit_count += 1
            return cached, key, True
        cost = transpile_and_measure_cost(
            circuit,
            compiler,
            circuit_fingerprint=circuit_fingerprint,
            backend=backend,
        )
        self._costs[key] = cost
        self.miss_count += 1
        return cost, key, False

    def __len__(self) -> int:
        return len(self._costs)


def _backend_name(backend: Any) -> str:
    candidate = getattr(backend, "name", None)
    return str(candidate() if callable(candidate) else candidate)


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
) -> TranspiledCircuitCost:
    """Transpile one circuit under fully recorded settings and measure counts."""
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
    fingerprint = circuit_fingerprint or canonical_qiskit_circuit_fingerprint(circuit)
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
        transpiled_circuit=transpiled_circuit,
    )


def _metric_values(cost: Any) -> tuple[float, ...]:
    return tuple(float(getattr(cost, name)) for name in _COST_METRICS)


def _metric_statistics(
    costs: Sequence[Any],
    *,
    weights: Sequence[float] | None,
) -> tuple[tuple[str, CompiledMetricStatistics], ...]:
    if not costs:
        raise ValueError("At least one transpiled circuit cost is required.")
    columns = tuple(zip(*(_metric_values(cost) for cost in costs), strict=True))
    statistics: list[tuple[str, CompiledMetricStatistics]] = []
    for name, values in zip(_COST_METRICS, columns, strict=True):
        if weights is None:
            mean = math.fsum(values) / len(values)
            if len(values) > 1:
                variance = math.fsum((value - mean) ** 2 for value in values) / (
                    len(values) - 1
                )
                standard_error = math.sqrt(variance / len(values))
            else:
                variance = None
                standard_error = None
        else:
            mean = math.fsum(
                weight * value for weight, value in zip(weights, values, strict=True)
            )
            variance = None
            standard_error = None
        statistics.append(
            (
                name,
                CompiledMetricStatistics(
                    mean=float(mean),
                    unbiased_sample_variance=(
                        None if variance is None else float(variance)
                    ),
                    standard_error=(
                        None if standard_error is None else float(standard_error)
                    ),
                    minimum=float(min(values)),
                    maximum=float(max(values)),
                ),
            )
        )
    return tuple(statistics)


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


def _estimate_event_costs(
    preparation: DFRTEEventPreparation,
    compiler: CompilerSettings,
    *,
    events: Sequence[Any],
    estimate_kind: CostEstimateKind,
    weights: Sequence[float] | None,
    controlled: bool,
    ancilla_qubit: int | None,
    cancel_adjacent_equal_bases: bool,
    seed: int | None,
    cache: TranspiledCircuitCostCache | None,
    backend: Any | None,
) -> CompiledEventCostEstimate:
    working_cache = cache if cache is not None else TranspiledCircuitCostCache()
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    costs: list[TranspiledCircuitCost] = []
    keys: set[str] = set()
    cache_hits = 0
    for event in events:
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
        costs.append(cost)
        keys.add(key)
        cache_hits += int(was_cached)
    statistics = _metric_statistics(costs, weights=weights)
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
        compiler=compiler,
        controlled=controlled,
        basis_reuse_policy=(
            "raw_adjacent_equal_basis"
            if cancel_adjacent_equal_bases
            else "disabled"
        ),
        seed=None if is_exact else seed,
        event_probability_sum=(math.fsum(weights) if weights is not None else None),
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
    if not math.isclose(probability_sum, 1.0, abs_tol=1e-12):
        raise ValueError("Enumerated RTE event probabilities must sum to one.")
    return _estimate_event_costs(
        preparation,
        compiler,
        events=events,
        estimate_kind="exact_compiled_expectation",
        weights=weights,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
        seed=None,
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
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
        seed=seed,
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
    compiler: CompilerSettings
    controlled: bool
    basis_reuse_policy: Literal["disabled", "raw_adjacent_equal_basis"]
    seed: int
    maximum_rte_steps: int
    maximum_untranspiled_circuit_size: int


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


def estimate_compiled_occurrence_cost(
    preparation: DFRTEEventPreparation,
    config: RTEConfig,
    distribution: RTEFiniteDistribution,
    compiler: CompilerSettings,
    *,
    sequence_sample_count: int,
    seed: int,
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
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    rng = random.Random(seed)
    sequence_costs: list[TranspiledCircuitCost] = []
    additive_costs: list[_MetricVector] = []
    difference_costs: list[_MetricVector] = []
    all_keys: set[str] = set()
    sequence_keys: set[str] = set()
    cache_hits = 0

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
        sequence_costs.append(sequence_cost)
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
            built_event = builder.build_event(event_request)
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
        additive_costs.append(additive)
        difference_costs.append(_subtract_costs(sequence_cost, additive))

    sequence_statistics = _metric_statistics(sequence_costs, weights=None)
    additive_statistics = _metric_statistics(additive_costs, weights=None)
    difference_statistics = _metric_statistics(difference_costs, weights=None)

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
        compiler=compiler,
        controlled=controlled,
        basis_reuse_policy=(
            "raw_adjacent_equal_basis"
            if cancel_adjacent_equal_bases
            else "disabled"
        ),
        seed=seed,
        maximum_rte_steps=maximum_rte_steps,
        maximum_untranspiled_circuit_size=maximum_untranspiled_circuit_size,
    )
