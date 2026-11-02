import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_j_u_consistency_repair as m

def test_Newton_CG_returns_pre_step_jac_not_endpoint():
    f=lambda x:(float(.5*np.dot(x,x)),x.copy())
    r=minimize(f,np.array([3.,-2.]),jac=True,hess=lambda x:np.eye(2),method='Newton-CG',options=dict(maxiter=1,xtol=1e-12))
    assert r.status==1 and not np.allclose(r.jac,r.x)
    np.testing.assert_array_equal(r.jac,[3.,-2.])
    assert np.linalg.norm(r.x)<1e-12

def test_comparison_keeps_fixed_allowances_and_rejects_bad_values():
    assert m.compare([1.],[1.],atol=1e-8,rtol=1e-7)['PASS']
    a=m.compare([1.+2e-6],[1.],atol=1e-8,rtol=1e-7)
    assert not a['PASS'] and a['maximum_allowance_ratio']>1
    assert not m.compare([np.nan],[1.],atol=1e-8,rtol=1e-7)['PASS']
    assert not m.compare([1.,2.],[1.],atol=1e-8,rtol=1e-7)['PASS']
