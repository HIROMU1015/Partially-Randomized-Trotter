"""Synthetic L_D=0 checks of the reused time-variable mean and seed split."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_b3 as b
b.setup()
from h46_new2_core import spectrum, compare_prefix


@pytest.mark.parametrize('f,r', [('2nd',4),('4th',8),('4th(m5_best)',16)])
def test_zero_prefix_retains_one_body_signed_scalar_and_normalization(f,r,monkeypatch):
    from h46_m5_contract import core
    import h46_new2_core
    monkeypatch.setattr(h46_new2_core,'weights',core.weights)
    one=np.array([[.2,.07],[.07,-.1]],complex)
    tail=np.array([[.03,.02j],[-.02j,-.04]],complex)
    ctx=SimpleNamespace(state=np.array([1.,0.],complex),ham=SimpleNamespace(constant=.17),
        specs=(spectrum(one),),tails={0:dict(spec=spectrum(tail),scalar=.11,lam=.15)},refs={1.2:.3+.2j})
    rows=[b.row('H4',1.2,6,f,r,k,'SYNTHETIC') for k in (2,4,6)]
    actual=b.signal.evolve_batch(ctx,rows)
    expected=compare_prefix(ctx.state,ctx.specs,ctx.tails[0]['spec'],.28,.15,f,6,T=1.2,total_steps=r,cutoffs=(2,4,6))
    for a,e in zip(actual,expected['finite_rows']):
        z=a['signal']['corrected']
        assert abs(complex(z['real'],z['imag'])-e['corrected_signal'])<3e-13
        assert abs(a['signal']['log_B']-e['log_B'])<3e-14
        assert a['signal']['allowance_evidence']['raw_corrected_vector_closure']<1e-12


def test_new_seed_namespace_and_confirmation_separation():
    import h46_long_time_native as native
    r=b.row('H6',2.,16,'4th(m5_best)',512,6,'SYNTHETIC')
    a={b.trial_seed(r,'B3_EXPLORATION',i,0,p) for i in range(9) for p in range(11)}
    c={b.trial_seed(r,'B3_CONFIRMATION',i,0,p) for i in range(33) for p in range(11)}
    assert len(a)==99 and len(c)==363 and not a&c
    assert native.PRIVATE==b.PRIVATE and native.PUBLIC==b.PUBLIC
