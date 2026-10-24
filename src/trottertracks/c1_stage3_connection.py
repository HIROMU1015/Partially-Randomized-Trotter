"""Minimal general-K partial-S2 / existing RTE / serial native-cost bridges.

No molecular arrays, circuits, sampling or shots are produced on import.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math

from . import c1_stage2_native as native
from .c1_stage3_accuracy import METRICS


@dataclass(frozen=True)
class PartialPlan:
    model: native.NativeModel
    ld: int
    atoms: tuple
    frames: dict
    extracted_identity: float


def split_native_model(model, ld):
    """Use both CDF and XDF NativeModel records without square impersonation.

    Frame primitives at unit time determine I/Z/ZZ coefficients. All tail I
    terms become one deterministic phase, never part of lambda_R. Tiny nonzero
    coefficients survive. Original outer order is preserved.
    """
    from .pr_dictionary_validation.prototype import MixedAtom
    rank = len(model.blocks)-1
    if type(ld) is not int or not 0 <= ld <= rank:
        raise ValueError('L_D must be an original-frame prefix length')
    atoms, frames, identities = [], {}, []
    for i, block in enumerate(model.blocks[ld+1:], start=ld):
        frame_id = model.label+':'+block.block_id
        frames[frame_id] = block.rotation
        # Exact equality of declared basis data; no approximate basis grouping.
        key = hashlib.sha256(repr(block.rotation).encode()).hexdigest()
        identities.append(-block.diagonal.global_phase)
        terms = [( (q,), a/2) for q, a in block.diagonal.rz]
        terms += [((p,q), a/2) for p,q,a in block.diagonal.rzz]
        for support, value in terms:
            if value == 0:
                continue
            if not math.isfinite(value):
                raise ValueError('Nonfinite tail coefficient')
            word = tuple('Z' if p in support else 'I' for p in range(model.modes))
            atoms.append(MixedAtom(frame_id+':'+','.join(map(str,support)), 'FRAME_Z',
                Fraction.from_float(float(value)), (frame_id,), word, key, frame_id))
    return PartialPlan(model, ld, tuple(atoms), frames, math.fsum(identities))


def configure_rte(plan, step_time, rte_steps, cutoff, *, seed=0):
    """Reuse canonical tail/configuration and finite law, no new fitter/law."""
    from .pr_dictionary_validation.prototype import make_tail
    from trotterlib.rte import make_rte_config, step_taylor_truncation_residual_bound
    tail = make_tail(plan.atoms, plan.model.label+':C1Stage3')
    if tail.deterministic_only_certified:
        return tail, None, None
    if type(rte_steps) is not int or rte_steps < 1:
        raise ValueError('Positive RTE microstep count required')
    residual = step_taylor_truncation_residual_bound(
        tail.lambda_r*float(Fraction(step_time))/rte_steps, cutoff)
    # Fixed K; this API field records its achieved residual, not a phase/CA
    # acceptance threshold. assess_round() checks the actual error budget.
    return (tail, *make_rte_config(tail, evolution_time=float(Fraction(step_time)),
        rte_steps=rte_steps, truncation_tolerance=max(residual,math.ulp(0.)),
        finite_taylor_order=cutoff, seed=seed))


def sample_occurrence(tail, distribution, count, seed):
    from trotterlib.rte import iter_sample_rte_events
    return tuple(iter_sample_rte_events(tail.components, distribution,
                                      sample_count=count, seed=seed))


def trajectory_seed(master, arm, candidate, round_index, axis, shot, occurrence):
    """Separate cost-calibration/quantum-shot domains at the caller's label."""
    text = repr((master, arm, candidate, round_index, axis, shot, occurrence)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little')


class FullFrameProvider:
    """Use existing full-frame Givens only; assume no new singleton reduction."""
    def __new__(cls, plan):
        from .pr_dictionary_validation.circuits import BasisProvider
        class Provider(BasisProvider):
            def basis(self, atom, singleton):
                return super().basis(atom, singleton=False)
        return Provider(plan.model.modes, plan.frames)


def partial_kernel(plan, step_time, events, *, rte_config=None, builder=None, provider=None):
    """One D-forward half / existing finite RTE sequence / D-reverse half.

    Native kernel is subsequently repeated coherently with FRESH independent
    event sequences, not the same random trajectory reused q times.
    """
    from qiskit import QuantumCircuit
    from .pr_dictionary_validation.circuits import build_event_sequence
    if not plan.atoms:
        # LD=rank keeps the complete deterministic S2 baseline, no empty RTE.
        if plan.ld != len(plan.model.blocks)-1:
            raise ValueError('Identity-only random tail needs explicit deterministic treatment')
        return (builder or native.NativeBuilder()).ordered(plan.model, step_time, optimized=True)
    events = tuple(events)
    if rte_config is None or len(events) != rte_config.rte_steps or not math.isclose(
            float(Fraction(step_time)),rte_config.evolution_time,rel_tol=1e-14,abs_tol=0):
        raise ValueError('One correctly timed finite-RTE occurrence required')
    signs={a.atom_id:1 if a.coefficient>0 else -1 for a in plan.atoms}
    for event in events:
        expected=signs[event.rotation_component_id]*math.atan(
            rte_config.dimensionless_step_time/(event.taylor_order+1))
        if not math.isclose(event.rotation_angle,expected,rel_tol=1e-14,abs_tol=1e-15):
            raise ValueError('Event angle/configuration mismatch')
    builder = builder or native.NativeBuilder()
    provider = provider or FullFrameProvider(plan)
    half = tuple((i, Fraction(step_time)/2) for i in range(plan.ld+1))
    qc = QuantumCircuit(plan.model.modes+1)
    qc.compose(builder.sequence(plan.model, half, optimized=True, scalar_time=0), inplace=True)
    qc.compose(build_event_sequence(tuple(events), plan.atoms, provider, optimized=True), inplace=True)
    qc.compose(builder.sequence(plan.model, tuple(reversed(half)), optimized=True, scalar_time=0), inplace=True)
    qc.p(-(plan.model.constant+plan.extracted_identity)*float(Fraction(step_time)), plan.model.modes)
    return qc


def compose_serial_cost(kernel, shell, *, m, substeps, target_id):
    """An implemented native-kernel repetition policy, not a fitted long-q proxy.

    Counts add exactly for native concatenation. Depths here are the explicit
    serial staged depth sums (barrier-separated execution), not globally
    recompiled circuit.depth(). Shared interval errors scale linearly with the
    reuse multiplicity; no false independent-calibration variance reduction.
    """
    if type(m) is not int or m < 0 or type(substeps) is not int or substeps < 1:
        raise ValueError('Valid round/substep counts required')
    if kernel['target_id'] != target_id:
        raise ValueError('Kernel target identity differs')
    if kernel['policy'] != 'native_kernel_repeat_with_fences' or shell['policy'] != 'separate_native_readout':
        raise ValueError('Whole-circuit optimized Stage2 cost is not an arbitrary long-q proxy')
    n = (1<<m)*substeps
    mean = {k:n*kernel['mean'][k]+shell['mean'][k] for k in METRICS}
    qualified = kernel.get('simultaneous_interval_qualified',False) and shell.get('simultaneous_interval_qualified',False)
    lower = upper = None
    if qualified:
        lower={k:n*kernel['lower'][k]+shell['lower'][k] for k in METRICS}
        upper={k:n*kernel['upper'][k]+shell['upper'][k] for k in METRICS}
    return dict(m=m, axis=shell['axis'], target_id=target_id, mean=mean, lower=lower, upper=upper,
        scope='serial_native_kernel_composition_without_state_preparation',
        long_round_coverage=True, simultaneous_interval_qualified=qualified,
        kernel_occurrences=n, depth_semantics='scheduled_serial_sum',
        classical_cost_samples=kernel.get('classical_cost_samples'), state_preparation_cost=None)


def serial_native_sequence(kernel_factory, *, occurrences, modes):
    """Materialize the declared policy when a small/approved build is needed.

    The factory is invoked for EVERY occurrence so PR trajectories can be
    refreshed. Compile kernels first; do not globally recompile this sequence.
    All-wire barriers implement the scheduled serial depth convention.
    """
    from qiskit import QuantumCircuit
    if type(occurrences) is not int or occurrences < 1 or type(modes) is not int or modes < 1:
        raise ValueError('Positive occurrence/mode counts required')
    qc=QuantumCircuit(modes+1)
    for j in range(occurrences):
        kernel=kernel_factory(j)
        if kernel.num_qubits != modes+1 or kernel.num_clbits:
            raise ValueError('Coherent ancilla-last native kernel required')
        if any(x.operation.name in ('measure','reset') for x in kernel.data):
            raise ValueError('Nonunitary kernel')
        qc.barrier(*range(modes+1))
        qc.compose(kernel,inplace=True)
    qc.barrier(*range(modes+1))
    return qc


def readout_segments(modes, axis):
    """Existing Hadamard convention, split around the controlled sequence.

    Separate segments prevent an empty-shell H/H cancellation during compile.
    Neither segment prepares the system eigenstate. State preparation remains
    an explicitly excluded/unknown cost, never an HF eigenstate claim.
    """
    from qiskit import QuantumCircuit
    if axis not in ('X','Y') or type(modes) is not int or modes < 1:
        raise ValueError('Valid ancilla-last X/Y readout required')
    head=QuantumCircuit(modes+1);tail=QuantumCircuit(modes+1,1)
    head.h(modes)
    if axis=='Y':tail.sdg(modes)
    tail.h(modes);tail.measure(modes,0)
    return head,tail
