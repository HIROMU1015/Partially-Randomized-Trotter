"""Exact epigraph identities on invented arrays only."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.diagnostic-packages')]
from trottertracks import c_h6_conic_diagnostic as conic
from test_research_c_h6_factored_newton import chart


@pytest.mark.parametrize('arm',['PR','ENTRY'])
def test_epigraph_equals_original_value_at_actual_decoded_origin(arm):
    c=chart(arm);P,q,A,b,(k,nr,nc)=conic.problem(c)
    zero=np.zeros(k);_,_,r,coeff,h,m,value=c._primal(zero)
    x=np.r_[zero,np.asarray(r.ravel()/np.sqrt(c.p.spec.loss_scale),float),
        np.asarray(h/c.p.spec.mass_scale,float),float(m/c.p.spec.mass_scale)]
    slack=b-A@x
    np.testing.assert_allclose(slack[:nr+1],0.,atol=1e-14)
    soc=slack[nr+1:].reshape(nc,3)
    np.testing.assert_allclose(soc[:,0],np.linalg.norm(soc[:,1:],axis=1),atol=1e-14)
    assert float(.5*x@(P@x))==pytest.approx(float(value),abs=1e-12)


@pytest.mark.parametrize('arm',['PR','ENTRY'])
def test_small_conic_solution_is_checked_with_original_physical_gate(arm):
    from trottertracks import c_h6_fit_conditioning as fit
    c=chart(arm);J,meta=conic.solve_and_polish(c)
    local=fit.Chart(c.p,J,c.policy)
    value,gx,_=local.independent(np.zeros(local.dimension))
    assert meta['conic']['status'] in ('Solved','AlmostSolved')
    assert value==pytest.approx(meta['conic']['objective'],abs=1e-9)
    # A solver status alone is never used to qualify a scientific endpoint.
    assert np.max(abs(gx))<1e-5


@pytest.mark.parametrize('arm',['PR','ENTRY'])
def test_rank14_near_degenerate_head_and_nonzero_tail(arm):
    from test_research_c_h6_conditioning import invented
    from trottertracks import c_h6_fit_conditioning as fit
    from trottertracks.c_near_transfer import objective_specs,System
    from trottertracks.c1_h6_initialization import reconstruct
    _,J,U,_=invented(angle=1e-8)
    rng=np.random.default_rng(310)
    perturb=rng.normal(size=J.shape);perturb=(perturb+perturb.transpose(0,2,1))/2
    g=reconstruct(J+1e-5*perturb,U)
    s=objective_specs(g,J,U,System(6,14,4,1.,'a'*64),strength=.1,epsilon=1e-8)[arm]
    p=fit.backend(s).Coefficients(g,U,s)
    c=fit.Chart(p,J,fit.inner.DEFAULT)
    endpoint,meta=conic.solve_and_polish(c)
    local=fit.Chart(p,endpoint,c.policy)
    _,gx,_=local.independent(np.zeros(local.dimension))
    assert np.max(abs(gx))<1e-5,dict(meta=meta,J_norm=float(np.linalg.norm(endpoint)))
    assert meta['total_iterations']<=1000
