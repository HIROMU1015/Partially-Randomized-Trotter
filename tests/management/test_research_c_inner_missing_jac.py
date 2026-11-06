"""Synthetic missing-jac qualification; never loads molecular arrays."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_inner_point_solver as m


def artificial():
    spec=m.stable.pr.Objective(2,1,1,1.,1.,0.)
    return m.stable.Coefficients(np.array([[[[.9]]]]),np.ones((2,1,1)),spec),np.array([.1,.2])[:,None,None]


def solver_case(mode):
    def solve(fun,zero,**kwargs):
        value,gradient=fun(zero)
        endpoint=zero.copy()
        if mode=='moved_endpoint':endpoint+=.001
        if mode=='different_callback_point':fun(zero+.001)
        if mode=='accepted_step':kwargs['callback'](OptimizeResult(x=zero.copy(),fun=value))
        return OptimizeResult(x=endpoint,fun=value,jac=gradient+1 if mode=='forged_jac' else None,
            status=0 if mode=='success_without_jac' else 3,nit=1 if mode=='later_CG' else 0,
            success=mode=='success_without_jac',message='ARTIFICIAL_INITIAL_CG_FAILURE',nfev=1,njev=1,nhev=1)
    return solve


def test_missing_initial_jac_uses_captured_input_and_preserves_raw_failure():
    p,j=artificial();records=[]
    out,info=m.solve_j(p,j,solver=solver_case('initial_CG'),record=lambda a,b:records.append((a,b)))
    assert not info['success'] and info['status']==3 and info['nit']==0
    assert not info['raw_jac_available']
    assert info['gradient_qualification_route']=='CAPTURED_SOLVER_INPUT_INITIAL_CG_FAILURE'
    assert not info['consistency']['raw_jac_at_endpoint']['PASS']
    assert info['consistency']['captured_solver_gradient_at_endpoint']['PASS']
    assert 'solver_jac' not in records[0][1]
    np.testing.assert_array_equal(records[0][1]['captured_solver_point'],np.zeros(2))
    assert info['value_gradient_calls_in_block']==4  # Same existing post-solver calls.
    np.testing.assert_array_equal(out,j)


@pytest.mark.parametrize('mode',['moved_endpoint','different_callback_point','accepted_step','forged_jac','success_without_jac','later_CG'])
def test_unavailable_or_wrong_gradients_do_not_get_a_general_exception(mode):
    p,j=artificial()
    with pytest.raises(ValueError,match='PR_INNER_POINT_CONSISTENCY_MISMATCH'):
        m.solve_j(p,j,solver=solver_case(mode))


def test_no_actual_solver_VG_observation_cannot_qualify_missing_jac():
    p,j=artificial()
    # A numeric value alone is deliberately insufficient: this fake solver
    # never asks the registered VG callback for its actual input gradient.
    def no_callback(fun,zero,**kw):
        return OptimizeResult(x=zero.copy(),fun=0.,jac=None,status=3,nit=0,
            success=False,message='ARTIFICIAL',nfev=0,njev=0,nhev=0)
    with pytest.raises(ValueError,match='PR_INNER_POINT_CONSISTENCY_MISMATCH'):
        m.solve_j(p,j,solver=no_callback)
