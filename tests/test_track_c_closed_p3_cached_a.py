"""Exact-context memoization is compared to A, including global cancellation."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import XXPlusYYGate
from trottertracks.track_c_compiler_a import compile_reference
from trottertracks.track_c_closed_p3_cached_a import compile_cached
from trottertracks.track_c_d_resources import metrics


def test_full_context_translation_and_inverse_passes():
    for mask in range(16):
        q=QuantumCircuit(3);q.global_phase=.13
        q.append(XXPlusYYGate(.123,-.27),[0,1]);q.rz(1e-15,2)
        if mask&1:q.crz(.0013,2,1)
        if mask&2:q.cz(2,0);q.p(-.27,2)
        if mask&4:q.h(2);q.h(2);q.sdg(1)
        if mask&8:q.ccx(0,1,2)
        q.cx(1,2);q.cx(1,2);q.x(0);q.x(0)
        q.append(XXPlusYYGate(-.123,-.27),[0,1])
        a=compile_reference(q);b=compile_cached(q);assert metrics(a)==metrics(b)
        x,y=Operator(a).data,Operator(b).data;phase=np.vdot(x,y);phase/=abs(phase)
        np.testing.assert_allclose(x,y/phase,atol=1e-12)
        # Same context, a different exact angle and repeated cache hits.
        c=q.copy();c.rz(.000011,0)
        assert metrics(compile_reference(c))==metrics(compile_cached(c))
