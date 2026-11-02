"""Artificial phase bookkeeping/cost invariance; no saved H6 decode."""
from pathlib import Path
import sys
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator,Statevector
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.c1_h6_compiler_precision import compiled_wrapper
from trottertracks.c1_h6_native_precision import gaussian
from trottertracks.pr_dictionary_validation.gates import compiled_wrapper as old
from trottertracks.pr_dictionary_validation.fixture import CompilerProfile


def body(q):
    return [(x.operation.name,list(x.operation.params),[q.find_bit(b).index for b in x.qubits]) for x in q.data]


@pytest.mark.parametrize('phase',[0.,.37,-2.1])
def test_declared_phase_retained_and_gate_body_identical(phase):
    q=QuantumCircuit(3);q.global_phase=phase;q.h(0);q.crz(.217,2,1);q.rzz(-.31,0,2)
    q.append(gaussian([[.8,-.6],[.6,.8]])[0].to_gate(),[0,1])
    new=compiled_wrapper(q,CompilerProfile());baseline=old(q,CompilerProfile())
    np.testing.assert_allclose(Operator(new).data,Operator(q).data,atol=4e-14,rtol=0)
    assert body(new)==body(baseline) and not new.parameters and q.global_phase==phase%(2*np.pi)


def test_long_artificial_control0_identity_phase_and_unchanged_cost(tmp_path):
    import json
    rng=np.random.default_rng(83);q=QuantumCircuit(5)
    for _ in range(300):
        u=np.linalg.qr(rng.normal(size=(4,4)))[0];b,_=gaussian(u)
        q.compose(b.inverse(),range(4),inplace=True);q.crz(.17,4,1);q.compose(b,range(4),inplace=True)
    psi=np.r_[rng.normal(size=16)+1j*rng.normal(size=16),np.zeros(16)];psi/=np.linalg.norm(psi)
    baseline=old(q,CompilerProfile());new=compiled_wrapper(q,CompilerProfile())
    old_error=float(np.max(abs(Statevector(psi.copy()).evolve(baseline).data-psi)))
    new_error=float(np.max(abs(Statevector(psi.copy()).evolve(new).data-psi)))
    assert body(new)==body(baseline) and new.count_ops()==baseline.count_ops()
    assert old_error>20*new_error and new_error<1e-10
    (tmp_path/'phase.json').write_text(json.dumps(dict(old_error=old_error,new_error=new_error,gate_body_identical=True)))


def test_unbound_input_fails_without_fallback():
    q=QuantumCircuit(1);q.rz(Parameter('unbound'),0)
    with pytest.raises(ValueError,match='BOUND_INPUT'):compiled_wrapper(q,CompilerProfile())
