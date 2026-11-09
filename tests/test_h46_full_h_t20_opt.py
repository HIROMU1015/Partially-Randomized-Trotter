"""Focused validation of new contraction and action-verifier acceleration."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_full_h_t20_opt as opt
opt.full.setup()
from h46_new2_core import spectrum


def test_s_equivalence_without_deterministic_blocks():
    ctx=SimpleNamespace(state=np.array([1.,0.],complex),ham=SimpleNamespace(constant=.2),specs=(),
        tails={0:dict(spec=spectrum(np.array([[.3,.11j],[-.11j,-.2]])),scalar=.1,lam=.6)},refs={2.0:0j})
    for k in (0,2,4,6,8):
        a=opt.full.signal.evolve_batch(ctx,[opt.row('H4',4,8,k,'SYNTHETIC')])[0]['signal']
        b=opt.full.signal.evolve_batch(ctx,[opt.row('H4',1,32,k,'SYNTHETIC')])[0]['signal']
        assert abs(a['log_B']-b['log_B'])<2e-14
        for key in ('corrected','raw'):
            assert max(abs(a[key][axis]-b[key][axis]) for axis in ('real','imag'))<2e-13


def test_aer_matches_original_statevector_with_global_phase_and_gaussian():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    from qiskit.quantum_info import Statevector
    qc=QuantumCircuit(3);qc.global_phase=.17
    qc.append(XXPlusYYGate(.31,.22),(0,1));qc.p(-.14,1);qc.crz(.47,2,1);qc.cx(0,2);qc.rz(.13,2)
    initial=np.arange(1,9,dtype=complex)+1j*np.arange(8);initial/=np.linalg.norm(initial)
    expected=np.asarray(Statevector(initial).evolve(qc).data)
    actual=opt.aer_statevector(qc,initial)
    assert np.linalg.norm(actual-expected)<1e-13
