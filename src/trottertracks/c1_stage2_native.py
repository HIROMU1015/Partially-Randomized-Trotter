"""Dormant native connection for saved CDF/XDF; import never builds circuits.

Model preparation/eigendecomposition and every build are scientific operations.
The preparation CLI never invokes them. Tests inject artificial data and mocks.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
from types import SimpleNamespace

from . import c1_general_k_adapter as general


@dataclass(frozen=True)
class NativeBlock:
    block_id: str
    rotation: tuple
    diagonal: object  # existing DiagonalEvolutionPrimitives at unit time


@dataclass(frozen=True)
class NativeModel:
    label: str
    blocks: tuple  # one-body first, then original outer frame order
    constant: float
    source_identity: str
    input_identity: str

    def __post_init__(self):
        if not self.blocks or not all(isinstance(b, NativeBlock) for b in self.blocks):
            raise ValueError('One-body and ordered frame blocks required')
        n = len(self.blocks[0].rotation)
        if n < 1 or any(len(b.rotation) != n or any(len(r) != n for r in b.rotation)
                        for b in self.blocks):
            raise ValueError('Square compatible Gaussian frames required')
        if len({b.block_id for b in self.blocks}) != len(self.blocks):
            raise ValueError('Duplicate block identity')
        if not math.isfinite(self.constant) or not self.source_identity or not self.input_identity:
            raise ValueError('Finite scalar and separate source/input identity required')

    @property
    def modes(self):
        return len(self.blocks[0].rotation)


def _rotation(matrix):
    return tuple(tuple(complex(x) for x in row) for row in matrix)


def prepare_cdf(model):
    """Existing corrected kappa + general K; no fit or recorrection."""
    from trotterlib.df_trotter.decompose import diag_hermitian
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives
    if not isinstance(model, general.GeneralKModel):
        raise TypeError('Explicit validated number GeneralKModel required')
    U, eps = diag_hermitian(model.one_body_tensor, assume_hermitian=True)
    blocks = [NativeBlock('one_body', _rotation(U), one_body_diagonal_primitives(eps, 1.))]
    blocks.extend(NativeBlock(f.frame_id, f.orbital_rotation, general.diagonal_primitives(f, 1.))
                  for f in model.frames)
    return NativeModel('CDF', tuple(blocks), model.constant, model.source_identity, model.input_identity)


def prepare_xdf(one, lambdas, matrices, constant, *, source_identity, input_identity):
    """Reuse saved square XDF directly, keeping its signed lambdas and order."""
    from trotterlib.df_trotter.decompose import diag_hermitian
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives, df_squared_diagonal_primitives
    lambdas, matrices = tuple(lambdas), tuple(matrices)
    if len(lambdas) != len(matrices):
        raise ValueError('XDF rank mismatch')
    U, eps = diag_hermitian(one, assume_hermitian=True)
    blocks = [NativeBlock('one_body', _rotation(U), one_body_diagonal_primitives(eps, 1.))]
    for i, (lam, matrix) in enumerate(zip(lambdas, matrices)):
        U, eta = diag_hermitian(matrix, assume_hermitian=True)
        blocks.append(NativeBlock(f'F{i}', _rotation(U), df_squared_diagonal_primitives(eta, lam, 1.)))
    return NativeModel('XDF', tuple(blocks), float(constant), source_identity, input_identity)


def ordered_schedule(block_count, total_time):
    """One symmetric second-order step; tuples are chronological application order.

    exp(-it H0/2), ..., exp(-it Hlast), ..., exp(-it H0/2).
    This is an ordered finite circuit target, never an exp(-it sum H) claim.
    """
    if type(block_count) is not int or block_count < 1:
        raise ValueError('Positive block count required')
    t = Fraction(total_time)
    if not t:
        raise ValueError('Nonzero declared time required')
    half = [(i, t/2) for i in range(block_count-1)]
    return tuple(half+[(block_count-1, t)]+list(reversed(half)))


def runs(model, schedule, *, optimized):
    """Only adjacent EXACT identical rotations may share a Gaussian boundary."""
    if type(optimized) is not bool:
        raise ValueError('Explicit route required')
    result = []
    for index, time in schedule:
        if type(index) is not int or not 0 <= index < len(model.blocks):
            raise ValueError('Invalid block index')
        time = Fraction(time)
        if optimized and result and model.blocks[result[-1][-1][0]].rotation == model.blocks[index].rotation:
            result[-1].append((index, time))
        else:
            result.append([(index, time)])
    return tuple(tuple(run) for run in result)


def combined_diagonal(model, run):
    """Within one common basis, sum phases/RZ/RZZ by fsum; exact-zero only."""
    from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives
    phases, rz, rzz = [], {}, {}
    for index, time in run:
        d = model.blocks[index].diagonal
        phases.append(float(time)*d.global_phase)
        for q, angle in d.rz:
            rz.setdefault(q, []).append(float(time)*angle)
        for a, b, angle in d.rzz:
            rzz.setdefault((a, b), []).append(float(time)*angle)
    return DiagonalEvolutionPrimitives(math.fsum(phases),
        tuple((q, math.fsum(v)) for q, v in sorted(rz.items()) if math.fsum(v) != 0.),
        tuple((a, b, math.fsum(v)) for (a, b), v in sorted(rzz.items()) if math.fsum(v) != 0.))


class NativeBuilder:
    """Native Givens only; basis-only unitaries remain uncontrolled and cancel.

    If B=Gamma(V), chronological B^-1, controlled D, B produces B D B^-1
    on control1 and identity on control0, including every diagonal scalar.
    """
    def __init__(self, *, call=None):
        self.cache = {}
        self.call = call or (lambda *args: None)

    def _basis(self, model, block):
        from .pr_dictionary_validation.circuits import native_gaussian
        key = (model.source_identity, model.input_identity, block.rotation)
        if key not in self.cache:
            self.call('gaussian_build', model.label+':'+block.block_id)
            self.cache[key] = native_gaussian(block.rotation)
        return self.cache[key]

    def sequence(self, model, schedule, *, optimized, scalar_time):
        from qiskit import QuantumCircuit
        from trotterlib.df_trotter.ops import append_diagonal_primitives
        qc = QuantumCircuit(model.modes+1)
        for run in runs(model, schedule, optimized=optimized):
            b = self._basis(model, model.blocks[run[0][0]])
            qc.compose(b.inverse(), qubits=range(model.modes), inplace=True)
            append_diagonal_primitives(qc, combined_diagonal(model, run),
                                      controlled=True, ancilla_qubit=model.modes)
            qc.compose(b, qubits=range(model.modes), inplace=True)
        # Nuclear/model scalar occurs once. Frame I terms remain in diagonals.
        qc.p(-model.constant*float(Fraction(scalar_time)), model.modes)
        return qc

    def local(self, model, index, time):
        return self.sequence(model, ((index, Fraction(time)),), optimized=False,
                             scalar_time=time if index == 0 else 0)

    def ordered(self, model, time, *, optimized):
        return self.sequence(model, ordered_schedule(len(model.blocks), time),
                             optimized=optimized, scalar_time=time)


@dataclass(frozen=True)
class WrapperContext:
    total_time: str
    q: int = 1
    r: int = 1  # metadata only, no RTE operations

    @property
    def delta(self):
        return Fraction(self.total_time)

    def validate(self):
        if not Fraction(self.total_time) or self.q != 1 or self.r != 1:
            raise ValueError('Single ordered step metadata required')


def evolution_result(circuit, model, time, *, optimized):
    from .pr_dictionary_validation.circuits import MixedEvolutionResult
    from .pr_dictionary_validation.fixture import digest
    schedule = ordered_schedule(len(model.blocks), time)
    records = tuple(dict(block=model.blocks[i].block_id, time=str(t)) for i, t in schedule)
    fp = digest(dict(model=model.label, source=model.source_identity, input=model.input_identity,
                     schedule=records, optimized=optimized, scalar=model.constant))
    return MixedEvolutionResult(circuit, model.modes, model.modes, WrapperContext(str(time)),
        (records,), model.input_identity, model.source_identity, fp, 'C1_stage2_ordered_S2')


def actual_adjoint(circuit):
    from .pr_dictionary_validation.circuits import actual_adjoint as existing
    return existing(SimpleNamespace(circuit=circuit))


def wrapper(circuit, model, time, axis, *, optimized, occupied):
    from .pr_dictionary_validation.circuits import hadamard_wrapper
    occupied = tuple(occupied)
    if len(set(occupied)) != len(occupied) or any(type(p) is not int or not 0 <= p < model.modes for p in occupied):
        raise ValueError('Invalid occupied modes')
    return hadamard_wrapper(evolution_result(circuit, model, time, optimized=optimized),
                            SimpleNamespace(modes=model.modes, occupied=occupied), axis, measure=True)


def xdf_car_generator(one, lam, bilinears):
    """Independent original G/one-body CAR path, not candidate eigenbasis/K."""
    import numpy as np
    one = np.asarray(one)
    density = one.reshape(-1) @ bilinears.reshape(one.size, -1)
    density = density.reshape(bilinears.shape[-2:])
    return density if lam is None else float(lam)*(density @ density)
