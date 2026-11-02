"""Only explicit two-qubit artificial systems; never load H6 inputs."""
from fractions import Fraction
from types import SimpleNamespace
import sys
import types
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm
package = types.ModuleType("p1_study")
package.__path__ = [str(Path(__file__).resolve().parents[1]/"src/trotterlib")]
sys.modules[package.__name__] = package
from p1_study.pauli_signal_adapter import (
    PauliHamiltonian, paired_polynomial_action, sparse_mean_action)
from p1_study.pauli_mechanism_bias import (
    split_operators, propagate, enclosure, difference)

STATE = np.array([1+.2j, .3-.4j, -.7j, .8], dtype=complex)*1.7
NORM = str(Fraction(float(np.vdot(STATE, STATE).real)))


def hamiltonian(commuting=True, scalar=.31):
    return PauliHamiltonian.from_terms(
        [("II", scalar), ("ZI", .8),
         ("IZ" if commuting else "XI", -.63), ("XX", .17)],
        num_qubits=2)


def oracle(ops, mode, T, q, r, K):
    dt = T/q
    hd, tail = ops.hd.toarray(), ops.tail.toarray()
    blocks = [x.toarray() for x in ops.blocks]
    if mode == "B":
        forward = reverse = expm(-.5j*dt*hd)
    else:
        forward = np.eye(4, dtype=complex)
        for op in blocks:
            forward = expm(-.5j*dt*op)@forward
        reverse = np.eye(4, dtype=complex)
        for op in reversed(blocks):
            reverse = expm(-.5j*dt*op)@reverse
    if mode == "Dpoly":
        x = -1j*dt/r*tail
        polynomial = np.eye(4, dtype=complex)
        power = np.eye(4, dtype=complex)
        for k in range(1, K+2):
            power = power@x/k
            polynomial += power
        middle = np.linalg.matrix_power(polynomial, r)
    else:
        middle = expm(-1j*dt*tail)
    unit = np.exp(-1j*ops.scalar*dt)*reverse@middle@forward
    return np.vdot(STATE, np.linalg.matrix_power(unit, q)@STATE)/float(Fraction(NORM))


@pytest.mark.parametrize("T", [.73, -.73])
@pytest.mark.parametrize("commuting", [True, False])
def test_exact_tail_half_order_scalar_and_raw_norm(T, commuting):
    ops = split_operators(hamiltonian(commuting), 2)
    z = {}
    for mode in ("B", "C"):
        row = propagate(ops, STATE, NORM, mode=mode, T=T, q=2, r=2, K=2)
        z[mode] = row["real"]+1j*row["imag"]
        assert abs(z[mode]-oracle(ops, mode, T, 2, 2, 2))<3e-14
        assert row["norm_divisions"]==1 and row["B_multiplications"]==0
    if commuting:
        assert abs(z["B"]-z["C"])<3e-14
    else:
        assert abs(z["B"]-z["C"])>1e-6


@pytest.mark.parametrize("T", [.73, -.73])
@pytest.mark.parametrize("K", [0, 2])
def test_degree_r_q_and_no_B_multiplier(T, K):
    ops = split_operators(hamiltonian(False), 2)
    row = propagate(ops, STATE, NORM, mode="Dpoly", T=T, q=2, r=3, K=K)
    assert abs(row["real"]+1j*row["imag"]-oracle(ops, "Dpoly", T, 2, 3, K))<3e-14
    assert row["polynomial_matvecs"]==2*3*(K+1)


def test_legacy_raw_mean_preserved_by_shared_recurrence():
    ops = split_operators(hamiltonian(), 2)
    a = SimpleNamespace(operators=lambda:(None, ops.blocks, ops.tail, ops.scalar),
                        step_time=.2, randomized=True, r=3, K=2,
                        distribution=SimpleNamespace(exact_finite_distribution=1.17))
    raw = sparse_mean_action(a, STATE, q=2)
    corrected = propagate(ops, STATE, NORM, mode="Dpoly", T=.4, q=2, r=3, K=2)
    z = np.vdot(STATE, raw)/float(Fraction(NORM))*1.17**6
    assert abs(z-(corrected["real"]+1j*corrected["imag"]))<3e-14


def test_L0_exact_equals_full_H_only_on_artificial_system():
    ham = hamiltonian(False)
    ops = split_operators(ham, 0)
    a = np.vdot(STATE, expm(-.8j*ham.sparse_operator().toarray())@STATE)/float(Fraction(NORM))
    for mode in ("B", "C"):
        row = propagate(ops, STATE, NORM, mode=mode, T=.8, q=8, r=2, K=2)
        assert abs(row["real"]+1j*row["imag"]-a)<3e-14


def test_semigroup_check_and_polynomial_target_cannot_be_changed():
    ops = split_operators(hamiltonian(), 2)
    for mode in ("B", "C"):
        rows = [propagate(ops, STATE, NORM, mode=mode, T=-.4, q=2, r=2, K=2,
                          tail_subdivisions=s) for s in (1, 2)]
        assert abs(complex(rows[0]["real"], rows[0]["imag"])-
                   complex(rows[1]["real"], rows[1]["imag"]))<3e-14
    with pytest.raises(ValueError):
        propagate(ops, STATE, NORM, mode="Dpoly", T=.4, q=2, r=2, K=2,
                  tail_subdivisions=2)


def test_saved_physical_enclosure_and_unresolved_not_zero():
    rect={a:dict(lower="1/3",upper="2/3") for a in ("real","imag")}
    row=enclosure(rect)
    assert row["real"]==.5 and row["axis_resolution"]["real"]==float(Fraction(1,6))
    d=difference(row,dict(real=.50000000000001,imag=.5,axis_resolution=1e-10))
    assert d["status"]=="UNRESOLVED" and d["absolute"]>0
