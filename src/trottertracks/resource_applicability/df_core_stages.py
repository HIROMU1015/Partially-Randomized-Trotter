"""Explicit bindings and thin stage calls to existing core APIs.

The public composer refuses scientific providers in P1. These connections are
available for interface tests and a later, separately authorized scientific path.
No fake fallback or automatic optional-dependency installation is provided.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from importlib import import_module
from collections.abc import Mapping
import math
from typing import Any, Callable

from .pipeline import PipelineError, ScientificExecutionDisabled


@dataclass(frozen=True)
class CoreBindings:
    hamiltonian: Callable
    prepare_s2: Callable
    configure_rte: Callable
    event_sequence: Callable
    step_request: Callable
    s2_builder: Callable
    compile_cost: Callable
    compiler_settings: Callable
    rpe_accounting: Callable | None
    signal_shots: Callable | None
    signal_wrapper: Callable | None
    estimand: str = "fixed_complex_signal"


def bind_existing_core(*, signal_shots=None, signal_wrapper=None,
                       estimand="fixed_complex_signal") -> CoreBindings:
    """Resolve same-checkout APIs with explicit study callbacks; run no stage.

    The unchanged legacy package initializer still controls transitive imports.
    This binding requests a phase-RPE callback only for that explicit estimand.
    """
    if estimand not in ("fixed_complex_signal", "phase_rpe"):
        raise PipelineError("an explicit supported estimand is required")
    if not callable(signal_wrapper):
        raise PipelineError("an explicit signal_wrapper callback is required")
    if estimand == "fixed_complex_signal" and not callable(signal_shots):
        raise PipelineError("an explicit signal_shots callback is required")
    if signal_shots is not None and not callable(signal_shots):
        raise PipelineError("signal_shots must be a callback or None")
    try:
        h = import_module("trotterlib.df_hamiltonian")
        s = import_module("trotterlib.df_partial_s2")
        r = import_module("trotterlib.rte")
        e = import_module("trotterlib.df_rte_circuit")
        cost = import_module("trotterlib.rte_compiled_cost")
        accounting = (import_module("trotterlib.rpe_resource_accounting")
                      if estimand == "phase_rpe" else None)
    except ModuleNotFoundError as exc:
        raise PipelineError(f"optional core dependency unavailable: {exc.name}; no fallback") from exc
    return CoreBindings(h.DFHamiltonian, s.prepare_df_partial_s2, r.make_rte_config,
                        e.DFRTEEventSequenceCircuitRequest, s.DFPartialS2StepRequest,
                        s.QiskitDFPartialS2CircuitBuilder, cost.transpile_and_measure_cost,
                        r.CompilerSettings,
                        accounting.evaluate_rpe_round_candidate if accounting is not None else None,
                        signal_shots, signal_wrapper, estimand=estimand)


def prepare(core: CoreBindings, hamiltonian, partition, *, policy=None):
    """Pass an explicit study policy to the existing API; supply no defaults."""
    keys = {"identity_policy", "coefficient_atol", "diagonal_sort",
            "basis_construction_policy", "partition_policy"}
    if not isinstance(policy, Mapping) or set(policy) != keys:
        raise PipelineError("explicit complete preparation policy is required")
    for key in keys - {"coefficient_atol"}:
        if not isinstance(policy[key], str) or not policy[key].strip():
            raise PipelineError(f"explicit preparation policy {key} is required")
    tolerance = policy["coefficient_atol"]
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or tolerance < 0):
        raise PipelineError("invalid preparation policy coefficient_atol")
    return core.prepare_s2(hamiltonian, partition, **dict(policy))


def finite_rte(core: CoreBindings, context, preparation, *, truncation_tolerance):
    """Configure the fixed law; deliberately do not sample an event."""
    return core.configure_rte(preparation.rte_preparation.symbolic_tail,
                              evolution_time=context.evolution_time,
                              rte_steps=context.rte_steps,
                              truncation_tolerance=truncation_tolerance,
                              finite_taylor_order=context.finite_taylor_order, seed=context.seed)


def event_request(core: CoreBindings, context, preparation, config, distribution, *, events):
    """Only explicitly supplied events; no make_* helper with hidden sampling."""
    if events is None or len(events) != context.rte_steps:
        raise PipelineError("explicit fixed events required; sampling is not a fallback")
    sequence = core.event_sequence(events=tuple(events),
                                   component_specs=preparation.rte_preparation.component_specs,
                                   controlled=True, ancilla_qubit=preparation.num_system_qubits,
                                   preserve_event_order=True,
                                   cancel_adjacent_equal_bases=True,
                                   tail_id=config.tail_id, tail_hash=config.tail_hash,
                                   occurrence_rte_steps=context.rte_steps)
    return core.step_request(preparation=preparation, step_time=context.evolution_time,
                             rte_config=config, rte_distribution=distribution,
                             rte_occurrence=sequence, controlled=True,
                             ancilla_qubit=preparation.num_system_qubits, seed=context.seed,
                             pf_label=context.formula)


def account_fixed_signal(core: CoreBindings, context, *, normalization, bias_by_axis, epsilon):
    if context.estimand != "fixed_complex_signal" or core.estimand != context.estimand:
        raise PipelineError("fixed-signal and phase-RPE budgets must not be mixed")
    if not callable(core.signal_shots):
        raise PipelineError("fixed-signal signal_shots callback is not bound")
    if normalization is None or epsilon is None or any(bias_by_axis.get(a) is None for a in ("cosine", "sine")):
        return {"N_cosine": None, "N_sine": None, "G": None, "status": "MISSING"}
    counts = {"N_" + a: core.signal_shots(normalization, bias_by_axis[a], epsilon)
              for a in ("cosine", "sine")}
    return {**counts, "G": None, "status": "UNQUALIFIED" if None in counts.values() else "ACCOUNTING_ONLY"}


def account_phase_rpe(core: CoreBindings, context, *args, **kwargs):
    """Existing API available separately; not H4 fixed-signal accounting."""
    if context.estimand != "phase_rpe" or core.estimand != context.estimand:
        raise PipelineError("phase-RPE evaluator cannot accept H4 fixed-signal scope")
    if not callable(core.rpe_accounting):
        raise PipelineError("phase-RPE callback was not explicitly bound")
    return core.rpe_accounting(*args, **kwargs)


def compile_axis(core: CoreBindings, context, request, *, axis):
    if axis not in ("cosine", "sine"):
        raise PipelineError("explicit signal axis required")
    if core.estimand != context.estimand:
        raise PipelineError("compiler callbacks and context estimand disagree")
    if not callable(core.signal_wrapper):
        raise PipelineError("signal_wrapper callback is not bound")
    evolution = core.s2_builder().build_step(request).circuit
    circuit = core.signal_wrapper(evolution, axis)
    compiler = core.compiler_settings(**asdict(context.compiler))
    return core.compile_cost(circuit, compiler)


def scientific_providers(*args, **kwargs):
    raise ScientificExecutionDisabled(
        "P1 has no scientific launch provider: new SOURCE/input/contract and numerical validation required")
