"""Time/scalar/finite-mean tests on a synthetic noncommuting two-level model."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_long_time_signal as m
m.install()
from h46_new2_core import spectrum,compare_prefix


@pytest.mark.parametrize('T,q,formula',[(1.2,6,'2nd'),(2.,4,'4th'),(.8,2,'4th(m5_best)')])
def test_variable_time_matches_existing_finite_mean(T,q,formula,monkeypatch):
    import h46_new2_core
    from h46_m5_contract import core
    monkeypatch.setattr(h46_new2_core,'weights',core.weights)
    a=np.array([[.2,.1],[.1,-.2]],complex)
    b=np.diag([.07,-.07]).astype(complex)
    tail=np.array([[.03,.02j],[-.02j,-.03]],complex)
    state=np.array([1.,0.],complex);scalar=.17;identity=.04
    ctx=SimpleNamespace(state=state,ham=SimpleNamespace(constant=scalar),
        specs=(spectrum(a),spectrum(b)),tails={1:dict(spec=spectrum(tail),scalar=identity,lam=.15)},refs={T:.3+.2j})
    rows=[dict(system='H4',method='PR',prefix=1,T=T,q=q,formula=formula,R_outer=32,K=k) for k in (2,4,6)]
    actual=m.evolve_batch(ctx,rows)
    expected=compare_prefix(state,ctx.specs,ctx.tails[1]['spec'],scalar+identity,.15,
                            formula,q,T=T,total_steps=32,cutoffs=(2,4,6))
    for a,b in zip(actual,expected['finite_rows']):
        z=complex(a['signal']['corrected']['real'],a['signal']['corrected']['imag'])
        assert abs(z-b['corrected_signal'])<2e-13
        assert abs(a['signal']['log_B']-b['log_B'])<2e-14
        assert a['signal']['allowance_evidence']['raw_corrected_vector_closure']<1e-13


def test_b0_identity_is_once_at_total_time():
    from scipy.linalg import expm
    h=np.array([[.2,.1],[.1,-.2]],complex);state=np.array([1.,0.],complex)
    ctx=SimpleNamespace(state=state,ham=SimpleNamespace(constant=.17),specs=(spectrum(h),),
                         tails={0:dict(scalar=.04)},refs={2.:0j})
    rows=[dict(method=method,prefix=0,T=2.,q=10,formula='4th') for method in ('B0','B0I')]
    out=m.evolve_batch(ctx,rows)
    exact=np.vdot(state,expm(-2j*h)@state)
    for rec,scalar in zip(out,(.17,.21)):
        signal=rec['signal']['corrected']
        assert abs(complex(signal['real'],signal['imag'])-exact*np.exp(-2j*scalar))<2e-13
