"""Artificial arrays only. Decimal is an independent small reference, not a fitter."""
from decimal import Decimal,localcontext
from dataclasses import replace
from collections import Counter
from pathlib import Path
import sys,time
import numpy as np
import jax
import pytest
jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_j_u_precision as precision
from trottertracks import c_j_u_stable as m
from test_research_c_separated_J_U import invented


def decimal_physical_U(g,J,U,spec,digits=80):
    """Independent product-rule reference at EXACT binary64 array inputs.

    Only the artificial n=2 cases use this O(n^6) scalar-loop calculation.
    No exp/log, finite difference of noisy residual, molecular array or fit.
    """
    with localcontext() as ctx:
        ctx.prec=digits
        d=lambda x:Decimal.from_float(float(x))
        n,rank=spec.norb,spec.rank
        u=[[[d(x) for x in row] for row in frame] for frame in U]
        j=[[[d(x) for x in row] for row in frame] for frame in J]
        scale=d(spec.loss_scale);factor=d(float(1/np.sqrt(2.)))
        residual={}
        for p,q,r,s in np.ndindex((n,)*4):
            value=sum(j[t][i][k]*u[t][p][i]*u[t][q][i]*u[t][r][k]*u[t][s][k]
                      for t in range(rank) for i in range(n) for k in range(n))
            residual[p,q,r,s]=value-d(g[p,q,r,s])
        result=[]
        for t in range(rank):
            for a,b in zip(*np.triu_indices(n,1)):
                du=[[Decimal(0) for _ in range(n)] for _ in range(n)]
                for p in range(n):du[p][a]=-u[t][p][b]*factor;du[p][b]=u[t][p][a]*factor
                grad=Decimal(0)
                for (p,q,r,s),res in residual.items():
                    delta=sum(j[t][i][k]*(
                        du[p][i]*u[t][q][i]*u[t][r][k]*u[t][s][k]+
                        u[t][p][i]*du[q][i]*u[t][r][k]*u[t][s][k]+
                        u[t][p][i]*u[t][q][i]*du[r][k]*u[t][s][k]+
                        u[t][p][i]*u[t][q][i]*u[t][r][k]*du[s][k])
                        for i in range(n) for k in range(n))
                    grad+=res*delta/scale
                result.append(float(grad))
    return np.array(result)


@pytest.mark.parametrize('scale',[1.,1e-9,1e-12])
def test_same_array_primal_precision_against_independent_Decimal(scale):
    g,J,U,s=invented(role='rc2',scale=scale)
    # Rounded exact reconstruction is a severe residual-cancellation example.
    g=m.linear.reconstruct(J,U)
    ref=decimal_physical_U(g,J,U,s)
    _,_,old=m.previous.state(g,J,U,s)
    parts,jg,gu,chart,meta=precision.evaluate(g,J,U,s)
    error=np.linalg.norm(gu-ref);old_error=np.linalg.norm(old-ref)
    assert error<max(1e-13,old_error/30)
    assert meta['rigorous_gradient_error_bound'] is None
    np.testing.assert_array_equal(gu,chart)
    # The NumPy reconstruction residual is exactly zero, but different JAX
    # accumulation can produce a nonzero gradient. Repeating it agrees exactly
    # without certifying either evaluation's true error.
    np.testing.assert_array_equal(m.linear.reconstruct(J,U)-g,0.)
    np.testing.assert_array_equal(old,m.previous.state(g,J,U,s)[2])
    if scale==1e-12:
        assert old_error>1e-6 and error<1e-6


@pytest.mark.parametrize('role',['rc2','pr'])
def test_objective_J_gradient_and_chart_pullback_same_point(role):
    g,J,U,s=invented(role=role);eta=np.array([.12,-.08,.19])
    V=precision.geometry(U,eta)
    row,gj,gu,chart,_=m.state(g,J,V,s,base_U=U,eta=eta)
    assert row['value_same_array']['PASS'] and row['forward_pullback_same_array']['PASS']
    p=m.Coefficients(g,V,s);x=p.pack(J);d=np.linspace(-.3,.4,len(x));h=1e-6
    f=lambda x:precision.parts_gradient_J(g,p.unpack(x),V,s)[0]['objective']
    assert (f(x+h*d)-f(x-h*d))/(2*h)==pytest.approx(gj@d,abs=3e-9)
    direction=np.array([.3,-.2,.4])
    def fU(eta):
        return precision.parts_gradient_J(g,J,precision.geometry(U,eta),s)[0]['objective']
    assert (fU(eta+h*direction)-fU(eta-h*direction))/(2*h)==pytest.approx(chart@direction,abs=3e-9)
    with pytest.raises(ValueError,match='UNEQUAL_GEOMETRY_ARRAY_POINT'):
        precision.evaluate(g,J,V,s,base_U=U,eta=eta+.01)


def test_RC2_residual_and_mixed_sensitivity_bound_not_a_floating_enclosure():
    g,J,U,s=invented(role='rc2');p=m.Coefficients(g,U,s)
    a,info=m.blocks.solve_j(p,J);a,_=m.previous.refine_j(p,a)
    _,_,reference,_,_=precision.evaluate(g,a,U,s)
    _,_,gradient,_,_=precision.evaluate(g,J,U,s)
    bound=precision.rc2_residual_bound(g,J,U,s)
    assert bound['mu']==pytest.approx(2*s.strength/s.l2_scale)
    assert np.linalg.norm(J-a)<=bound['solution_error_plugin']+1e-12
    assert np.linalg.norm(gradient-reference)<=bound['outer_gradient_error_plugin']
    assert bound['qualification']=='UNRESOLVED_FLOATING_RESIDUAL_AND_ARITHMETIC'
    assert not bound['rigorous_numerical_enclosure']


@pytest.mark.parametrize('role',['rc2','pr'])
def test_reduced_FD_resolves_each_perturbed_U_explicit_calls(role):
    g,J,U,s=invented(role=role);calls=Counter()
    def call(k,l,f):calls[k]+=1;return f()
    oracle=m.StableReducedOracle(g,J,U,s,call=call)
    z=np.zeros(3);d=np.array([.3,-.7,.2]);h=2e-5
    point=oracle.evaluate(z);plus=oracle.evaluate(z+h*d);minus=oracle.evaluate(z-h*d)
    assert (plus['value']-minus['value'])/(2*h)==pytest.approx(point['gradient']@d,abs=3e-7)
    assert calls==dict(inner_J_solve=9,precision_state=6,reduced_admission=3)
    assert oracle.evaluate(z) is point and len(oracle.points)==3


def test_PR_universal_gauge_and_additional_null_preserved():
    g,J,U,s=invented(rank=3,ld=2)
    N=np.zeros_like(J);N[0]=1.;N[1]=-1.
    a=precision.evaluate(g,J,U,s);b=precision.evaluate(g,J+N,U,s)
    assert a[0]['objective']==pytest.approx(b[0]['objective'],abs=1e-13)
    np.testing.assert_allclose(a[2],b[2],atol=1e-13)
    U[1]=U[0];_,_,result=m.optimize(g,J,U,s)
    assert result['status']=='REDUCED_GRADIENT_UNQUALIFIED'
    assert not result['qualification']['applicability']['admitted']


@pytest.mark.parametrize('role',['rc2','pr'])
def test_progress_physical_first_order_and_no_orthogonal_projection(role):
    g,J,U,s=invented(role=role)
    a,V,r=m.optimize(g,J,U,s)
    assert r['first_order_qualified'] and r['status']=='BOTH_FIRST_ORDER_MET'
    assert r['inner_solves']==3*r['points']
    assert r['terminal']['J_stationarity']['status']==r['terminal']['U_stationarity']['status']=='MET'
    assert r['terminal']['orthogonality_maxabs']<1e-12
    assert r['terminal']['components']['objective']<m.previous.state(g,J,U,s)[0]['components']['objective']


@pytest.mark.parametrize('mode',['precision','J','points','outer'])
def test_nonconvergence_stability_unqualified_and_NO_fallback(mode):
    g,J,U,s=invented(role='pr' if mode=='J' else 'rc2')
    p=replace(m.DEFAULT,refinement_U_l2=1e-30) if mode=='precision' else replace(m.DEFAULT,j_maxiter=1) if mode=='J' else replace(m.DEFAULT,max_points=1) if mode=='points' else replace(m.DEFAULT,outer_maxiter=1)
    _,_,r=m.optimize(g,J,U,s,p)
    assert not r['first_order_qualified']
    assert r['status'] in ('J_UNQUALIFIED','REDUCED_GRADIENT_UNQUALIFIED','OUTER_LIMIT_UNQUALIFIED','U_UNQUALIFIED')
    assert r['retry']==r['resume']==0


def test_more_precision_does_not_cure_unqualified_ill_conditioned_inner():
    g,J,U,s=invented(role='rc2',scale=1e-12)
    _,_,r=m.optimize(g,J,U,s)
    assert r['status']=='J_UNQUALIFIED' and not r['first_order_qualified']


def test_extended_backend_absent_and_nonfinite_are_explicit_STOP(monkeypatch):
    g,J,U,s=invented()
    original=np.finfo
    monkeypatch.setattr(np,'finfo',lambda dtype:original(np.float64) if dtype is np.longdouble else original(dtype))
    with pytest.raises(ValueError,match='EXTENDED_ACCUMULATION_UNAVAILABLE'):precision.evaluate(g,J,U,s)
    monkeypatch.undo();J=J.copy();J[0,0,0]=np.nan
    with pytest.raises(ValueError,match='FINITE_BINARY64_ARRAY_POINT'):precision.evaluate(g,J,U,s)


def test_joint_alternative_same_objective_and_gradient_no_envelope():
    for role in ('rc2','pr'):
        g,J,U,s=invented(role=role);p=m.Coefficients(g,U,s);z=np.r_[np.zeros(3),p.pack(J)]
        value,grad=m.coupled_value_gradient(z,g,U,s)
        parts,gj,gu,_,_=precision.evaluate(g,J,U,s)
        assert value==pytest.approx(parts['objective'],abs=1e-14)
        np.testing.assert_allclose(grad,np.r_[gu,gj],atol=1e-13)


@pytest.mark.parametrize('seed,admit',[(41,True),(42,False)])
def test_n4_RC2_old_stability_failure_can_improve_or_remain_unqualified(seed,admit):
    g,J,U,s=invented(seed=seed,n=4,rank=8,ld=2,role='rc2',scale=2.1839632053132894e-9)
    old=m.previous.ReducedOracle(g,J,U,s)
    with pytest.raises(m.Unqualified,match='REDUCED_GRADIENT_UNQUALIFIED'):old.evaluate(np.zeros(48))
    new=m.StableReducedOracle(g,J,U,s)
    if admit:
        point=new.evaluate(np.zeros(48));assert point['qualification']['latest_refinement_U_shift_l2']<1e-6
    else:
        with pytest.raises(m.Unqualified,match='REDUCED_GRADIENT_UNQUALIFIED'):new.evaluate(np.zeros(48))


def test_PR_latest_pair_after_real_inner_improvement_not_old_result_reclassification():
    g,J,U,s=invented(role='pr',scale=1e-10)
    old=m.previous.ReducedOracle(g,J,U,s)
    with pytest.raises(m.Unqualified):old.evaluate(np.zeros(3))
    assert old.points[0]['qualification']['refinement_U_shift_l2']>1e-6
    point=m.StableReducedOracle(g,J,U,s).evaluate(np.zeros(3))
    assert point['qualification']['latest_refinement_U_shift_l2']<1e-6
    assert point['qualification']['legacy_result_not_reclassified']


def test_direct_array_reference_does_not_pack_unpack_or_mutate_J(monkeypatch):
    g,J,U,s=invented()
    J[0,0,1]=J[0,1,0]=.19
    p=m.Coefficients(g,U,s)
    assert not np.array_equal(p.unpack(p.pack(J)),J)  # expose the lossy boundary
    J0,U0=J.copy(),U.copy()
    monkeypatch.setattr(m.Coefficients,'pack',lambda *a:pytest.fail('reference packed J'))
    monkeypatch.setattr(m.Coefficients,'unpack',lambda *a:pytest.fail('reference unpacked J'))
    row,*_=m.state(g,J,U,s)
    assert row['value_same_array']['PASS']
    np.testing.assert_array_equal(J,J0);np.testing.assert_array_equal(U,U0)


def test_det_sign_and_pi_geometry_preserve_physical_tangent():
    g,J,U,s=invented();U[0]=np.diag([-1.,1.]);U[1]=-np.eye(2)
    eta=np.array([.3,.2,-.1]);V=precision.geometry(U,eta)
    row,gj,gu,chart,*_=m.state(g,J,V,s,base_U=U,eta=eta)
    assert row['orthogonality_maxabs']<1e-12
    np.testing.assert_allclose(np.linalg.det(V),np.linalg.det(U),atol=1e-12)
    np.testing.assert_allclose(gu,decimal_physical_U(g,J,V,s),atol=1e-13)


def test_terminal_reference_uses_decoded_arrays_without_weighted_roundtrip(monkeypatch):
    from trottertracks import c_j_u_stable_pilot as pilot
    g,J,U,s=invented();J[0,0,1]=J[0,1,0]=.19
    terminal=m.state(g,J,U,s)[0]
    J0,U0=J.copy(),U.copy()
    monkeypatch.setattr(m.Coefficients,'vg',lambda *a,**k:pytest.fail('packed-coordinate terminal reference'))
    x,gradient,r=pilot.native_diagnostic(g,J,U,m.pr,s,terminal,m.DEFAULT)
    assert r['status']=='PASS' and r['checks']['J_decode_identical']
    assert 'NO isometric pack/unpack' in r['reference_point']
    np.testing.assert_array_equal(J,J0);np.testing.assert_array_equal(U,U0)


@pytest.mark.parametrize('role',['rc2','pr'])
def test_small_scale_reduced_directional_FD_with_re_solved_inner(role):
    g,J,U,s=invented(role=role,scale=1e-9);o=m.StableReducedOracle(g,J,U,s)
    z=np.zeros(3);d=np.array([.3,-.7,.2]);p=o.evaluate(z);h=1e-5
    plus=o.evaluate(z+h*d);minus=o.evaluate(z-h*d)
    # FD includes inner approximation and truncation, not a strict enclosure.
    fd=(plus['value']-minus['value'])/(2*h)
    assert fd==pytest.approx(p['gradient']@d,abs=5e-7,rel=2e-7)


def joint_control(g,J,U,s,maxiter=256):
    """Stronger artificial control, existing full-direction SVD coordinates."""
    from scipy.optimize import minimize
    p=m.Coefficients(g,U,s);nrot=s.rank*s.norb*(s.norb-1)//2
    _,D=m.blocks.residual_jacobian(np.zeros(nrot),g,J,U,s)
    C=(np.sqrt(2*s.gamma)*p.T/s.mass_scale if p.is_pr else
       np.sqrt(2*s.strength/s.l2_scale)*np.eye(len(p.pack(J))))
    B=np.vstack([np.column_stack([D,p.A/np.sqrt(s.loss_scale)]),
                 np.column_stack([np.zeros((len(C),nrot)),C])])
    W,sv=m.blocks.metric(B);z0=np.r_[np.zeros(nrot),p.pack(J)]
    def fun(t):
        v,d=m.coupled_value_gradient(z0+W@t,g,U,s);return v,W.T@d
    r=minimize(fun,np.zeros(len(z0)),jac=True,method='BFGS',options=dict(maxiter=maxiter,gtol=1e-7))
    z=z0+W@r.x;V=precision.geometry(U,z[:nrot]);j=p.unpack(z[nrot:]);q=m.Coefficients(g,V,s)
    j,first=m.blocks.solve_j(q,j);j,second=m.previous.refine_j(q,j)
    return dict(raw=m.old._raw(r),terminal=m.state(g,j,V,s)[0],post_J_raw=[first,second],
        metric='Existing full-direction invertible SVD coordinates,no added penalty',
        maxiter=maxiter,metric_singular_values=sv.tolist(),budget_not_matched_to_reduced=True)


def conditioning_study():
    rows=[]
    for role in ('rc2','pr'):
        for scale in (1.,1e-9):
            g,J,U,s=invented(role=role,scale=scale);p=m.Coefficients(g,U,s);x=p.pack(J)
            H=p.hess(x);eigen=np.linalg.eigvalsh(H)
            h=1e-5;directions=np.eye(len(x))
            mixed=np.column_stack([(precision.evaluate(g,p.unpack(x+h*d),U,s)[2]-
                precision.evaluate(g,p.unpack(x-h*d),U,s)[2])/(2*h) for d in directions])
            row=dict(role=role,loss_scale=scale,smallest_floating_HJJ_eigen=float(eigen[0]),
                largest_floating_HJJ_eigen=float(eigen[-1]),mixed_block_frobenius=float(np.linalg.norm(mixed)),
                interpretation='Invented arrays only;floating normal-product eigenvalues do NOT certify strict convexity or PR curvature lower bounds')
            if role=='rc2':
                B=np.vstack([p.A/np.sqrt(scale),np.sqrt(2*s.strength/s.l2_scale)*np.eye(len(x))]);sv=np.linalg.svd(B,compute_uv=False)
                row['condition_augmented']=float(sv[0]/sv[-1]);row['theoretical_mu']=2*s.strength/s.l2_scale
                j,_=m.blocks.solve_j(p,J);j,_=m.previous.refine_j(p,j)
                row['floating_error_bound_after_refinement']=precision.rc2_residual_bound(g,j,U,s)
            rows.append(row)
    return rows


def artificial_study():
    from scipy.optimize import minimize
    rows=[]
    for role in ('rc2','pr'):
        for scale in (1.,1e-9,1e-12):
            g,J,U,s=invented(role=role,scale=scale)
            row=dict(role=role,norb=2,rank=3,ld=1,seed=41,loss_scale=scale)
            for name,algorithm in [('legacy_reduced',m.previous),('stable_reduced',m)]:
                count=Counter();inners=[]
                def call(k,l,f):count[k]+=1;return f()
                def record(k,l,d,a):
                    if k=='inner_summary':inners.extend(d['solves'])
                tick=time.monotonic();_,_,r=algorithm.optimize(g,J,U,s,call=call,record=record)
                r['elapsed_seconds_in_study']=time.monotonic()-tick
                r['named_calls']=dict(count)
                r['solver_evaluation_totals']={k:sum(i.get(k,0) for i in inners) for k in ('nfev','njev','nhev','linear_solves')}
                row[name]=r
            p=m.Coefficients(g,U,s);z=np.r_[np.zeros(3),p.pack(J)]
            joint=minimize(lambda z:m.coupled_value_gradient(z,g,U,s),z,jac=True,method='BFGS',options=dict(maxiter=64,gtol=1e-7))
            V=precision.geometry(U,joint.x[:3]);j=p.unpack(joint.x[3:]);pj=m.Coefficients(g,V,s)
            j,raw=m.blocks.solve_j(pj,j);j,ref=m.previous.refine_j(pj,j)
            terminal=m.state(g,j,V,s)[0]
            row['joint_alternative']=dict(raw=m.old._raw(joint),terminal=terminal,post_J_raw=[raw,ref],
                rigorous_convergence=False,budget_not_matched_to_reduced=True)
            row['joint_full_direction_control']=joint_control(g,J,U,s)
            rows.append(row)
    accuracy=[]
    for scale in (1.,1e-9,1e-12):
        g,J,U,s=invented(role='rc2',scale=scale);g=m.linear.reconstruct(J,U)
        tick=time.monotonic();ref=decimal_physical_U(g,J,U,s);reference_seconds=time.monotonic()-tick
        tick=time.monotonic();legacy=m.previous.state(g,J,U,s)[2];legacy_seconds=time.monotonic()-tick
        tick=time.monotonic();new=precision.evaluate(g,J,U,s)[2];stable_seconds=time.monotonic()-tick
        accuracy.append(dict(loss_scale=scale,reference_gradient=ref.tolist(),legacy_gradient=legacy.tolist(),
            stable_gradient=new.tolist(),legacy_error_l2=float(np.linalg.norm(legacy-ref)),
            stable_error_l2=float(np.linalg.norm(new-ref)),seconds=dict(decimal80=reference_seconds,legacy=legacy_seconds,stable=stable_seconds),
            interpretation='Array-defined polynomial,not a measured H4 arithmetic error;timings are not uniform-budget speedups'))
    admission=[]
    for role,n,rank,ld,scale,seeds in [('rc2',4,8,2,2.1839632053132894e-9,(41,42)),
                                    ('pr',4,8,2,2.1839632053132894e-9,(41,42)),
                                    ('pr',2,3,1,1e-10,(41,42))]:
        for seed in seeds:
            g,J,U,s=invented(seed=seed,n=n,rank=rank,ld=ld,role=role,scale=scale)
            result=dict(role=role,seed=seed,norb=n,rank=rank,ld=ld,loss_scale=scale)
            for name,cls in [('legacy',m.previous.ReducedOracle),('stable',m.StableReducedOracle)]:
                oracle=cls(g,J,U,s)
                try:point=oracle.evaluate(np.zeros(rank*n*(n-1)//2));status='ADMITTED'
                except m.Unqualified as e:status=e.status;point=oracle.points[0]
                except ValueError as e:result[name]=dict(status='TECHNICAL_STOP',reason=str(e));continue
                result[name]=dict(status=status,qualification=point['qualification'],state=point['state'])
            admission.append(result)
    return dict(schema='C_J_U_STABILITY_ARTIFICIAL_V1',rows=rows,decimal_accuracy=accuracy,
        conditioning=conditioning_study(),admission_examples=admission,
        real_molecular_decodes=0,scientific_calls=0,old_stops_reclassified=False)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    with a.report.open('x') as f:json.dump(artificial_study(),f,ensure_ascii=False,indent=2,allow_nan=False)
