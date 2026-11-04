"""Artificial parameters/operators only. No molecular file or saved state loads."""
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import math
import sys

import numpy as np
import pytest
from flint import acb, ctx
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RZZGate, XXPlusYYGate
from qiskit.quantum_info import Operator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'rpe_controlled_hadamard_preparation_20261104'))
sys.path.insert(0, str(HERE.parents[2] / 'src'))
import connection_adapter as c
import test_circuit_semantics as old
from trotterlib.df_trotter.ops import one_body_diagonal_primitives, df_squared_diagonal_primitives
from trotterlib.df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder


@pytest.mark.parametrize('q', [1, 2, 131072, 2**80])
@pytest.mark.parametrize('constants', [(.1, .2), (1.e16, -1.e16+2), (.17, -.31)])
def test_scalar_bound_exact_rational_and_transcendental(q, constants):
    # Verify exact parameter capture and exp Lipschitz bound independently in Acb.
    dt, (a, b) = .037, constants
    raw = (-dt*a, -dt*b)
    optimized = -dt*q*(a+b)
    gap = c.scalar_gap(q, raw, optimized)
    assert gap == abs(Q.from_float(optimized)-q*sum(map(Q.from_float, raw)))
    ball, _ = c.modules()
    with ctx.workprec(224):
        diff = acb(0, ball.real(optimized)).exp()-acb(0, ball.real(q*sum(map(Q.from_float, raw)))).exp()
        assert abs(diff).lower() <= ball.real(min(Q(2), gap)).upper()


def test_zero_scalar_gap_and_missing_evidence():
    assert c.scalar_gap(2, [.125, -.25], -.25) == 0
    assert c.total_transfer(131072, 32, boundary=Q(0), component=None, lowering=0) is None
    assert c.total_transfer(2, 3, boundary=Q(1,100), component=Q(1,1000), lowering=0) == Q(28,1000)
    with pytest.raises(ValueError): c.scalar_gap(0, [0.], 0.)
    with pytest.raises(ValueError): c.scalar_gap(2, [math.inf], 0.)
    with pytest.raises(ValueError): c.boundary_bound(2, -.1, 0)
    with pytest.raises(ValueError): c.total_transfer(1, 1, boundary=-1, component=0, lowering=0)


def test_underflow_block_and_lowering_are_not_silently_exact():
    tiny = math.ulp(0.)
    half = one_body_diagonal_primitives(np.array([tiny]), .5)
    full = one_body_diagonal_primitives(np.array([tiny]), 1.)
    assert c.block_gap(half, full) > 0
    assert not c.half_angle_exact(tiny)
    assert c.half_angle_exact(.375)


@pytest.mark.parametrize('kind', ['one', 'squared'])
def test_block_gap_bounds_deliberate_phase_and_rotation_mutation(kind):
    from trotterlib.df_trotter.ops import append_diagonal_primitives
    make = (lambda t: one_body_diagonal_primitives(np.array([.4, -.7]), t)) if kind == 'one' else (
        lambda t: df_squared_diagonal_primitives(np.array([.3, -.6]), -.27, t))
    half, full = make(.05), make(.1)
    bad = replace(full, global_phase=full.global_phase+.07,
                  rz=tuple((i, a+(.09 if i == 0 else 0)) for i,a in full.rz))
    gap = c.block_gap(half, bad)
    assert gap > Q(1,10)
    def matrix(p):
        circuit = QuantumCircuit(2); append_diagonal_primitives(circuit, p)
        return Operator(circuit).data
    observed = np.linalg.norm(matrix(bad)-matrix(half)@matrix(half), 2)
    assert .01 < observed <= float(gap)+1e-14


@pytest.mark.parametrize('q,r,k', [(3,1,2), (5,3,4)])
def test_production_boundary_bound_on_noncommuting_artificial_trajectory(q,r,k):
    _, prep = old.artificial_preparation.__wrapped__()
    request, _, _ = old.forced_request(prep, r, k, q=q)
    dt = request.step_time
    raw_angles = [-dt*prep.constant_coefficient, -dt*prep.extracted_identity_coefficient]
    optimized = -dt*q*(prep.constant_coefficient+prep.extracted_identity_coefficient)
    block = prep.deterministic_blocks[0]
    half = one_body_diagonal_primitives(np.array(block.diagonal_eigenvalues), dt/2)
    full = one_body_diagonal_primitives(np.array(block.diagonal_eigenvalues), dt)
    bound = c.boundary_bound(q, c.scalar_gap(q, raw_angles, optimized), c.block_gap(half, full))
    builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    raw = Operator(builder.build(request, construction_policy='raw_concatenation').circuit).data
    opt = Operator(builder.build(request, construction_policy='boundary_optimized').circuit).data
    assert np.linalg.norm(opt-raw, 2) <= float(bound)+3e-13  # Float test tolerance, not proof evidence.


@pytest.mark.parametrize('gate', [RZGate(.375), RZZGate(-.625)])
def test_qiskit_control_shape_phase_and_sign(gate):
    controlled = gate.control(1)
    names = [x.operation.name for x in controlled.definition.data]
    if gate.name == 'rz':
        assert names == ['rz', 'cx', 'rz', 'cx']
        assert [float(x.operation.params[0]) for x in controlled.definition.data if x.operation.name == 'rz'] == [.1875, -.1875]
    else:
        assert names == ['ccx', 'crz', 'ccx']
        assert controlled.definition.data[1].operation.params == [-.625]
    assert float(controlled.definition.global_phase) == 0
    # Local control is least-significant: extract even/odd rows and columns.
    actual = Operator(controlled).data
    np.testing.assert_allclose(actual[::2,::2], np.eye(2**gate.num_qubits), atol=1e-14)
    np.testing.assert_allclose(actual[1::2,1::2], Operator(gate).data, atol=1e-14)
    assert np.max(abs(actual[::2,1::2])) < 1e-14
    assert np.max(abs(Operator(gate).data-Operator(gate.inverse()).data)) > .1


def descriptor(name, wires, params):
    return dict(name=name, qubits=list(wires), params=[float(x).hex() for x in params])


def fock(ops, n):
    circuit = QuantumCircuit(n)
    for op in ops:
        gate = {'rz': RZGate, 'xx_plus_yy': XXPlusYYGate}[op['name']](
            *(float.fromhex(p) for p in op['params']))
        circuit.append(gate, op['qubits'])
    return Operator(circuit).data


def annihilation(u):
    # Explicit Jordan-Wigner signs; independent of the one-particle adapter.
    n = len(u); result = np.zeros((2**n, 2**n), complex)
    for state in range(2**n):
        for j in range(n):
            if (state >> j) & 1:
                parity = (state & ((1<<j)-1)).bit_count()
                result[state ^ (1<<j), state] += u[j].conjugate()*(-1)**parity
    return result


@pytest.mark.parametrize('n', [3,4])
@pytest.mark.parametrize('support', [(0,), (0,1)])
def test_gaussian_car_column_certificate(n, support):
    ops = [descriptor('rz',(0,),(.17,)), descriptor('xx_plus_yy',(1,0),(.3,.21)),
           descriptor('xx_plus_yy',(1,2),(-.27,.4)), descriptor('rz',(2,),(-.12,))]
    other = ops+[descriptor('xx_plus_yy',(0,1),(.002,-.13))]
    b, _ = c.modules()
    with ctx.workprec(96):
        full, alt = c.one_particle(ops,n), c.one_particle(other,n)
        bound = c.component_gap(full,alt,support)
    u, v = fock(ops,n), fock(other,n)
    g = u[np.ix_([1<<i for i in range(n)], [1<<i for i in range(n)])]/u[0,0]
    for i in range(n):
        np.testing.assert_allclose([complex(full.center[j,i]) for j in range(n)], g[:,i], atol=1e-14)
    z = np.diag([(-1)**sum((s>>i)&1 for i in support) for s in range(2**n)])
    expected = np.eye(2**n, dtype=complex)
    for i in support:
        a = annihilation(g[:,i]); expected = expected @ (np.eye(2**n)-2*a.conj().T@a)
    np.testing.assert_allclose(u@z@u.conj().T, expected, atol=1e-14)
    observed = np.linalg.norm(u@z@u.conj().T-v@z@v.conj().T,2)
    assert 1e-4 < observed <= float(bound)
    # Retaining radii gives a nonzero certified upper even for identical circuits.
    assert c.component_gap(full,full,support) > 0


def test_unselected_columns_need_not_match_and_nonadjacent_gate_is_rejected():
    ops = [descriptor('rz',(0,),(.2,))]
    other = ops+[descriptor('xx_plus_yy',(1,2),(.7,.2))]
    with ctx.workprec(96):
        a, b = c.one_particle(ops,3), c.one_particle(other,3)
        assert c.component_gap(a,b,(0,)) < Q(1,10**20)
        assert c.component_gap(a,b,(1,)) > Q(1,10)
    with pytest.raises(ValueError): c.one_particle([descriptor('xx_plus_yy',(0,2),(.2,.3))],3)
    with pytest.raises(ValueError): c.one_particle([descriptor('rx',(0,),(.2,))],3)


def test_iid_factorization_with_noncommuting_halves_and_reuse_counterexample():
    x = np.array([[0,1],[1,0]],complex); z=np.diag([1.,-1.]); eye=np.eye(2)
    events = [math.cos(.31)*eye-1j*math.sin(.31)*x,
              math.cos(.23)*eye-1j*math.sin(.23)*z]
    weights = [Q(1,4),Q(3,4)]
    f = math.cos(.17)*eye-1j*math.sin(.17)*x
    rev = math.cos(.19)*eye-1j*math.sin(.19)*z
    mean = sum(float(p)*e for p,e in zip(weights,events))
    expectation = np.zeros((2,2),complex)
    for ids in product(range(2),repeat=6):  # r=2, q=3; fresh draw at each position
        value=eye.copy(); probability=Q(1)
        for s in range(3):
            value=f@value
            for i in ids[2*s:2*s+2]: value=events[i]@value; probability*=weights[i]
            value=rev@value
        expectation+=float(probability)*value
    target=np.linalg.matrix_power(rev@mean@mean@f,3)
    np.testing.assert_allclose(expectation,target,atol=2e-14)
    reused=sum(float(p)*np.linalg.matrix_power(rev@e@e@f,3) for p,e in zip(weights,events))
    assert np.linalg.norm(reused-target,2) > .01
