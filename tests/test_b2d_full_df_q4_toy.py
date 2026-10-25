"""q4 independent toy product, scalar, control and prepared X/Y checks."""
import numpy as np
import pytest
from qiskit.quantum_info import Operator
from trottertracks.b2d_evaluator import full_df_q4 as f
from test_b2d_full_df_toy import toy, independent_s2, ordinary


@pytest.mark.parametrize('index',range(2))
def test_q4_existing_builders_and_reference_match_independent_four_macro_product(toy,index):
    native = object.__new__(f.Native)
    native.__dict__.update(toy.__dict__)
    row = f.cases()[index]
    ref = native.full_reference(row)
    w1 = 1/(2-2**(1/3));w0 = -2**(1/3)/(2-2**(1/3))
    expected = np.eye(4,dtype=complex)
    for _ in range(4):
        for w in (w1,w0,w1):
            expected = independent_s2(native.full_blocks,float(row['total_time'])*w/4)@expected
    expected *= np.exp(-1j*float(row['total_time'])*.127)
    assert np.linalg.norm(ref-expected,ord=2) < 1e-12
    system = native.system(row);control = native.controlled_system(row)
    assert np.linalg.norm(Operator(system).data-ref,ord=2) < 1e-12
    assert np.linalg.norm(Operator(control).data-ordinary(ref),ord=2) < 1e-12
    assert np.linalg.norm(Operator(control).data[:4,:4]-np.eye(4),ord=2) < 1e-12
    z = np.vdot(native.state,ref@native.state)
    for axis,part in (('cosine',z.real),('sine',z.imag)):
        qc = native.wrapper(control,axis).remove_final_measurements(inplace=False)
        actual = Operator(qc).data
        assert np.linalg.norm(actual-native.wrapper_reference(ref,axis),ord=2) < 1e-12
        state = actual[:,0]
        assert abs(np.sum(abs(state[:4])**2)-np.sum(abs(state[4:])**2)-part) < 1e-12
    result = native.signals(row,ref)
    assert result['u_num'] is None and result['gamma'] is None
    assert result['observed_bias_budget']==5e-10
