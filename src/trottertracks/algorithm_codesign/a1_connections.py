"""A1-0 thin connections, with no CLI, science-file loading, or RNG.

Real sampler inputs and real selected_full_prices calls require a separately
authorized science scope. Tests supply invented inputs and mock price objects.
The original sampler, CV adapter, quantum builders and compiler are unchanged.
"""
from dataclasses import dataclass
from math import fsum, isclose
from types import SimpleNamespace

from .a1_control_variates import (
    ControlBank, LocalControl, OccurrenceLaw, law_from_choice_cdf,
)


@dataclass(frozen=True)
class SamplerConnection:
    law: OccurrenceLaw
    order_choice_input: tuple[float, ...]
    label_choice_input: tuple[float, ...]
    order_cdf: tuple[float, ...]
    label_cdf: tuple[float, ...]
    component_probability_sum: float


def _choice_cdf(probabilities):
    """NumPy 1.26.4 choice(replace=True,p=...) arithmetic, without sampling."""
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 1 or not p.size or not np.isfinite(p).all()
            or (p < 0).any() or not isclose(fsum(p), 1., abs_tol=1e-12, rel_tol=0.)):
        raise ValueError("normalized finite categorical input required")
    cdf = p.cumsum()
    cdf /= cdf[-1]
    return tuple(map(float, p)), tuple(map(float, cdf))


def connect_sampler_inputs(components, distribution, *, source, validate_components):
    """Connect already prepared sampler inputs; never prepare a Hamiltonian.

    Preserve component order and the exact _validate_components / division /
    np.asarray / cumsum / final-normalization order used by canonical choice.
    Returned grid masses may be zero for a positive binary64 input label.
    Caller supplies the existing rte._validate_components function, avoiding
    trotterlib package initialization here. Source must externally bind the
    validator, preparation, distribution and NumPy version.
    """
    import numpy as np
    if np.__version__ != "1.26.4":
        raise ValueError("the inspected choice profile is NumPy 1.26.4")
    if not source or tuple(distribution.orders) != (0, 2):
        raise ValueError("source binding and K=2 order profile required")
    components = tuple(components)
    probability_sum = validate_components(components)
    label_input, label_cdf = _choice_cdf(
        [c.probability / probability_sum for c in components])
    order_input, order_cdf = _choice_cdf(distribution.order_probabilities)
    law = OccurrenceLaw(
        law_from_choice_cdf(distribution.orders, order_cdf, source=source + ":order"),
        law_from_choice_cdf(tuple(c.component_id for c in components), label_cdf,
                            source=source + ":label"))
    return SamplerConnection(law, order_input, label_input, order_cdf,
                             label_cdf, probability_sum)


def proposed_control_banks(connection, *, law_id, a, b):
    """Construct caller-fixed banks; missing/zero grid support raises, no fallback.

    CV has exactly three controls. The extra order indicator is used only by
    the exhaustive order/label stratified comparator, through the existing API.
    No means, joint moments, stratum masses or feature search are evaluated here.
    """
    controls = (LocalControl(0, 0, ((0, a),)),
                LocalControl(0, 2, ((0, a),)),
                LocalControl(0, 2, ((0, a), (1, b))))
    occurrences = (connection.law,)
    cv = ControlBank(law_id, occurrences, controls)
    order = ControlBank(law_id, occurrences, (LocalControl(0, 0, ()),))
    strata = ControlBank(law_id, occurrences,
                         (LocalControl(0, 0, ()),) + controls)
    return cv, order, strata


def _price_dependencies():
    # Lazy imports: artificial tests never import a quantum builder or compiler.
    from trotterlib.df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder
    from trotterlib.research_direction_full_scope import apply_run_threshold_policy
    from trotterlib.rpe_hadamard_compiled_cost_benchmark import (
        QiskitRPEHadamardBenchmarkCircuitBuilder,
    )
    from trotterlib.rpe_hadamard_interrogation import RPEHadamardInterrogationRequest
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    return SimpleNamespace(
        apply_policy=apply_run_threshold_policy,
        repeated_builder=QiskitDFPartialS2RepeatedCircuitBuilder(),
        wrapper_builder=QiskitRPEHadamardBenchmarkCircuitBuilder(maximum_repetition_count=1),
        wrapper_request=RPEHadamardInterrogationRequest,
        compile=transpile_and_measure_cost,
    )


def selected_full_prices(request, support_definitions, compiler, *,
                         training_fingerprint, axes=("cosine", "sine"),
                         cache=None, _dependencies=None):
    """One selected evolution, requested full wrappers; no full-basis branch.

    This DOES build/compile with real dependencies; A1-0 tests inject mocks only.
    Caller binds H/snapshot/target and the unchanged WP06b training fingerprint.
    A cache hit avoids compilation, never a charged quantum interrogation.
    Existing get_or_transpile can compile on a miss and write its own cache.
    For an independently sampled one-axis trial, pass axes=(axis,) so the other
    readout is not built or compiled merely to price that trial.
    """
    expected = dict(basis_gates=("rz", "sx", "x", "cx"), backend_name=None,
                    coupling_map=None, optimization_level=2, layout_method=None,
                    routing_method=None, transpiler_seed=17, qiskit_version="1.3.0")
    if any(getattr(compiler, k) != v for k, v in expected.items()):
        raise ValueError("original opt2 compiler profile required")
    axes = tuple(axes)
    if not axes or len(set(axes)) != len(axes) or any(a not in ("cosine", "sine") for a in axes):
        raise ValueError("distinct requested X/Y readouts required")
    prep, config = request.preparation, request.rte_config
    if (request.repetition_count != 1 or request.step_time != .02
            or not request.controlled or request.ancilla_qubit != 8
            or prep.ld != 3 or prep.num_system_qubits != 8
            or prep.coefficient_atol != 0. or config is None or config.rte_steps != 1
            or config.finite_taylor_order != 2 or not training_fingerprint):
        raise ValueError("m0 opt2 threshold-zero controlled scope required")
    dep = _dependencies if _dependencies is not None else _price_dependencies()
    selected = dep.apply_policy(request, support_definitions,
                                maximum_support_run_length=1,
                                training_fingerprint=training_fingerprint)
    evolution = dep.repeated_builder.build(selected, construction_policy="boundary_optimized")
    result = []
    for axis in axes:
        wrapper = dep.wrapper_builder.build(dep.wrapper_request(
            evolution=evolution, axis=axis, include_measurement=True))
        fingerprint = wrapper.compiler_independent_fingerprint
        if cache is None:
            cost = dep.compile(wrapper.circuit, compiler, circuit_fingerprint=fingerprint,
                               actual_circuit_fingerprint=fingerprint)
            key, hit = None, False
        else:
            cost, key, hit = cache.get_or_transpile(
                wrapper.circuit, compiler, circuit_fingerprint=fingerprint)
        result.append((axis, wrapper, cost, key, hit))
    return tuple(result)
