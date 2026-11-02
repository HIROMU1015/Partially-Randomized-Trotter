"""Synthetic tensors/operators/LPs only. Stored H4 J/U are never decoded."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_fixed_u_tail_lp as lp
from trottertracks import c_fixed_u_tail_lp_runner as m


def write(path,value):Path(path).write_text(json.dumps(value,allow_nan=False)+'\n')


def test_passive_import():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_fixed_u_tail_lp_runner as m;assert not any(k in sys.modules for k in ('numpy','scipy','ffsim','qiskit','jax'));print(m.MAX_CALLS)"
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,check=True)
    assert int(r.stdout)==sum(m.LIMITS.values()) and m.LIMITS['LP_solve']==1


def test_apply_adjoint_explicit_map_offdiagonal_and_exact_coefficients():
    rng=np.random.default_rng(71);U=np.array([np.linalg.qr(rng.normal(size=(3,3)))[0] for _ in range(3)])
    U[0,:,0]*=-1
    assert np.linalg.det(U[0])<0 and np.linalg.det(U[1])>0
    x=rng.normal(size=(3,3,3));J=x+x.transpose(0,2,1);j=lp.pack(J);M=lp.FixedU(U)
    np.testing.assert_array_equal(lp.unpack(j,3,3),J)
    np.testing.assert_allclose(M.matrix()@j,M.apply(j).ravel(),atol=3e-15,rtol=0)
    tensor=rng.normal(size=(3,)*4)
    np.testing.assert_allclose(M.adjoint(tensor),M.matrix().T@tensor.ravel(),atol=3e-15,rtol=0)
    assert np.vdot(M.apply(j),tensor)==pytest.approx(np.vdot(j,M.adjoint(tensor)),abs=3e-14)
    # A single off-diagonal packed value creates BOTH symmetric tensor terms.
    basis=np.zeros((1,2,2));basis[0,0,1]=basis[0,1,0]=2.
    t=lp.FixedU(np.eye(2)[None]).apply(lp.pack(basis))
    assert t[0,0,1,1]==t[1,1,0,0]==2 and t.sum()==4
    from trottertracks.c1_general_k_adapter import general_k_coefficients
    K=np.repeat(np.repeat(J/2,2,axis=1),2,axis=2)
    independent=np.array([list(v for s,v in general_k_coefficients(k) if s) for k in K]).ravel()
    np.testing.assert_allclose(M.coefficient_matrix()@j,independent,atol=3e-15,rtol=0)
    assert np.sum(abs(independent))!=pytest.approx(np.sum(abs(J)))


@pytest.mark.parametrize('values,expected,improved',[
    ([1.,-1.],0.,True),([1.,2.],3.75,False),([-2.,1.],1.25,True),([-1.,-2.],3.75,False),([0.,0.],0.,False)])
def test_known_optima_and_scipy_dual_sign(values,expected,improved):
    J=np.array(values)[:,None,None];p=lp.problem(J,np.ones_like(J));r=lp.solve_once(p);d=lp.diagnose(p,r)
    assert d['numerical_optimality'] and d['strict_improvement']==improved
    assert d['new_lambda']==pytest.approx(expected,abs=1e-12)
    assert d['dual_objective_pilot']==pytest.approx(expected,abs=1e-12)
    assert abs(d['primal_dual_gap'])<1e-12 and d['rigorous_certificate'] is None
    if sum(values)!=0:assert np.sign(d['y'][0])==np.sign(sum(values))
    assert max(d['residuals'].values())<1e-12


def test_uniform_shift_inclusion_and_independent_mass_formula():
    rng=np.random.default_rng(7);x=rng.normal(size=(3,2,2));J=x+x.transpose(0,2,1)
    U=np.array([np.eye(2),[[0.,1.],[-1.,0.]],np.eye(2)])
    new=lp.uniform_shift(J,np.array([.2,-.1,-.1]));p=lp.problem(J,U)
    np.testing.assert_allclose(p.mapping.apply(lp.pack(new)),p.mapping.apply(p.j0),atol=1e-14,rtol=0)
    formula=lambda a:sum(np.sum(abs(j.sum(axis=1)))+.25*np.sum(abs(np.diag(j)))+abs(j[0,1]) for j in a)
    assert formula(new)==pytest.approx(np.sum(abs(p.mapping.coefficients(lp.pack(new)))))
    result=lp.diagnose(p,lp.solve_once(p));assert result['numerical_optimality']
    assert result['new_lambda']<=formula(new)+1e-9
    with pytest.raises(ValueError,match='ZERO_SUM'):lp.uniform_shift(J,[.2,0,0])


def test_conditioning_bad_dual_near_null_and_tiny_improvement_not_exact():
    U=np.array([np.eye(2),np.eye(2)]);J=np.array([np.eye(2),-np.eye(2)])
    p=lp.problem(J,U);r=lp.solve_once(p)
    assert p.conditioning['near_null_columns']>0 and not p.conditioning['equality_rows_dropped']
    bad=deepcopy(r);bad.eqlin.marginals[0]+=.1
    assert not lp.diagnose(p,bad)['numerical_optimality']
    p.conditioning['retained_condition']=1e14
    assert lp.diagnose(p,r)['status']=='UNRESOLVED_LP_DIAGNOSTICS'
    tiny=lp.problem(np.array([[[1e-12]],[[-1e-12]]]),np.ones((2,1,1)))
    d=lp.diagnose(tiny,lp.solve_once(tiny))
    assert d['original_lambda']>0 and d['strict_improvement'] is False
    U[0,0,0]+=1e-13;M=lp.FixedU(U);np.testing.assert_array_equal(M.U,U)
    assert M.orthogonality>0
    U[0,0,0]+=.001
    with pytest.raises(ValueError,match='NONORTHOGONAL'):lp.FixedU(U)
    with pytest.raises(ValueError):lp.unpack(np.zeros(6,dtype=complex),2,2)
    M=lp.FixedU(np.ones((1,1,1)))
    with pytest.raises(ValueError,match='UNDERFLOW'):M.coefficients(np.array([np.nextafter(0.,1.)]))


def test_large_cancelling_dual_objective_is_unresolved_despite_small_gap():
    J=np.array([[[0.,.25],[.25,0.]],[[0.,.75],[.75,0.]]])
    p=lp.problem(J,np.array([np.eye(2)]*2));r=lp.solve_once(p)
    # Tensor symmetry makes rows (00,11) and (11,00) identical.
    r.eqlin.marginals[3]+=2.**40;r.eqlin.marginals[12]-=2.**40
    d=lp.diagnose(p,r)
    assert d['objective_arithmetic_guard']>1e-6
    assert not d['numerical_optimality'] and d['strict_improvement'] is None


def test_partial_PF_K0_preserved_but_full_ordered_S2_need_not_be():
    b=m.b;ctx=m.old.old.molecule.SectorCAR(2);V=np.array([[.8,-.6],[.6,.8]])
    U=np.array([np.eye(2),V,np.eye(2),V,np.eye(2)])
    J=np.zeros((5,2,2));J[0]=np.diag([.2,.1]);J[1]=np.diag([-.1,.15])
    J[2]=np.diag([.7,0]);J[3]=np.diag([.2,-.5])
    new=J.copy();new[4]=new[2];new[2]=0
    np.testing.assert_allclose(lp.reconstruct(J[2:],U[2:]),lp.reconstruct(new[2:],U[2:]),atol=1e-14)
    packet=dict(h=np.array([[-.2,.1],[.1,.3]]),g=np.zeros((2,)*4),constant=np.array(.31))
    kernels=[];full=[]
    for a,identity in [(J,'before'),(new,'after')]:
        arm=m.prepare(packet,a,U,identity,physical_identity='invented')
        G=ctx.generators(**packet,J=a,U=U);tail,c,_=b.configure(arm,Fraction(1,10),128,0)
        assert b.tail_gate(ctx,arm,G,tail)['status']=='PASS'
        kernels.append(b.kernels(arm,G,Fraction(1,10),c))
        full.append(b.reference.reference_kernels(G,Fraction(1,10),ld=5)['pf'])
    for key in ('pf','corrected'):np.testing.assert_allclose(kernels[0][key],kernels[1][key],atol=1e-13,rtol=0)
    assert np.max(abs(full[0]-full[1]))>1e-8


def test_zero_lambda_no_division_and_ideal_attenuation_shots_monotonic():
    from trottertracks import c_rc2_pr_conditional_shots as shots
    b=m.b;ctx=m.old.old.molecule.SectorCAR(2);U=np.array([np.eye(2)]*4)
    J=np.zeros((4,2,2));J[2]=np.diag([.4,-.2]);J[3]=-J[2]
    new=J.copy();new[2:]=0
    packet=dict(h=np.array([[-.2,.1],[.1,.3]]),g=np.zeros((2,)*4),constant=np.array(.31))
    kernels=[];configs=[]
    for a,identity in [(J,'before'),(new,'after')]:
        arm=m.prepare(packet,a,U,identity,physical_identity='invented')
        G=ctx.generators(**packet,J=a,U=U);tail,c,_=b.configure(arm,Fraction(1,10),128,0)
        kernels.append(b.kernels(arm,G,Fraction(1,10),c));configs.append(c)
    assert configs[1] is None
    for key in ('pf','corrected'):np.testing.assert_allclose(kernels[0][key],kernels[1][key],atol=1e-13,rtol=0)
    profile=m.attenuation(*configs);assert all(r['new']==1 and r['old']<=1 for r in profile['rows'])
    target=shots.accuracy.AccuracyTarget(**shots.TARGET)
    for row in profile['rows']:
        plans=[]
        for A in (row['old'],row['new']):
            p=dict(arm='invented',identity={},target_id='invented',df_bound_Ha=1e-8,optimizer_qualification='UNQUALIFIED',error_evidence=None)
            ref=dict(m=row['m'],reference=dict(attenuation=A,pf_operator_error=.001,rte_corrected_operator_error=.002))
            p['rounds']=[dict(ref,m=i) for i in range(13)]
            q=shots.qualification(p,'ideal_exact_native');plans.append(shots.assess_round(target,p,q,ref))
        assert plans[1]['shots']<=plans[0]['shots']
    assert profile['shots'] is profile['actual_CA'] is profile['total_resources'] is None


def invented(root):
    rng=np.random.default_rng(99);x=rng.normal(size=(3,4,4))*.005;tail=x+x.transpose(0,2,1)
    J=np.zeros((8,4,4));J[:2]=np.diag([.02,-.01,.01,.03]);J[2:]=np.array([tail[0],-tail[0],tail[1],-tail[1],tail[2],-tail[2]])
    U=np.array([np.eye(4)]*8);packet=dict(h=np.diag([-.2,-.1,.1,.2]),g=np.zeros((4,)*4),constant=np.array(.31))
    np.savez(root/'input.npz',**packet);np.savez(root/'export.npz',J=J,U=U,x=np.zeros(128),jac=np.zeros(128))
    identity=dict(role='pr',export='export.npz',sha256=m.io.sha(root/'export.npz'),fit_source=m.old.FIT_SOURCE,physical_input=m.io.sha(root/'input.npz'))
    arm=m.prepare(packet,J,U,'invented',physical_identity=identity['physical_input']);ctx=m.old.old.molecule.SectorCAR(4)
    G=ctx.generators(**packet,J=J,U=U);tail,c,_=m.b.configure(arm,Fraction(1,10),128,0);kernel=m.b.kernels(arm,G,Fraction(1,10),c)
    np.savez(root/'reference.npz',generators=np.array(G),H=sum(G),pf=kernel['pf'],corrected=kernel['corrected'])
    records={k:k+'.json' for k in ('B_result','B_STOP','B_SOURCE','B_contract','B_manifest','pr_tail')}
    values=dict(B_result=dict(status='SAVED_ENDPOINT_STAGE_B_COMPLETE',arms=dict(pr=dict(identity=identity,DF_budget='PASS',optimizer_qualification='UNQUALIFIED',saved_DF_bound_Ha=1e-8))),
        B_STOP=dict(failed=False,scientific_units=143,call_counts=m.old.LIMITS,retry=0,resume=0,mandatory_stop=True),
        B_SOURCE=dict(execution_commit='d6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6',files={}),
        B_contract=dict(policy=m.old.POLICY,fit_source=m.old.FIT_SOURCE,identities=dict(pr=identity)),B_manifest=dict(files={}),
        pr_tail=dict(status='PASS',lambda_R=tail.lambda_r))
    for k,p in records.items():write(root/p,values[k])
    write(root/'SOURCE.json',dict(files={}))
    names={'input.npz','export.npz','reference.npz',*records.values()}
    contract=dict(scope=m.SCOPE,implementation=lp.VERSION,base_commit=m.BASE,policy=m.POLICY,solver_options=lp.OPTIONS,
        allowances=lp.ALLOWANCES,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,fresh_output=m.FRESH,retry=0,resume=0,
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),records=records,
        input='input.npz',export='export.npz',reference='reference.npz',identity=identity,
        saved_inputs={p:m.io.sha(root/p) for p in names},packages={})
    write(root/'contract.json',contract);return contract


def test_passive_preflight_and_input_source_mismatch(tmp_path,monkeypatch):
    with threadpool_limits(limits=1):c=invented(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('passive decode'))
    monkeypatch.setattr(lp,'solve_once',lambda *a,**kw:pytest.fail('passive LP'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    c['saved_inputs']['export.npz']='0'*64;write(tmp_path/'contract.json',c)
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')


@pytest.mark.parametrize('fault',[None,'solver','tensor'])
def test_synthetic_end_to_end_one_LP_no_fits_STOP_and_preservation(tmp_path,monkeypatch,fault):
    from scipy.optimize import OptimizeResult
    with threadpool_limits(limits=1):c=invented(tmp_path)
    protected=dict(c['saved_inputs'])
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'SYNTHETIC_LP_SOURCE\n')
    def forbidden(*a,**kw):pytest.fail('forbidden fit/native/sampling/physical norm')
    monkeypatch.setattr(m.b,'build_kernel',forbidden);monkeypatch.setattr(m.b,'independent_occurrence',forbidden)
    monkeypatch.setattr(lp.objective,'fit',forbidden);monkeypatch.setattr(m.old.old.molecule.SectorCAR,'physical',forbidden)
    count=[];original=lp.solve_once
    def solve(p):
        count.append(1)
        if fault=='solver':return OptimizeResult(success=False,status=1,message='invented stop',nit=0,fun=None,x=None)
        return original(p)
    monkeypatch.setattr(lp,'solve_once',solve)
    if fault=='tensor':
        original_replace=lp.replace_tail
        def changed(*a):
            j,u=original_replace(*a);u[0,0,0]+=.001;return j,u
        monkeypatch.setattr(lp,'replace_tail',changed)
    with threadpool_limits(limits=1):
        if fault=='tensor':
            with pytest.raises(ValueError,match='U_OR_HEAD_CHANGED'):m.run(tmp_path,tmp_path/'contract.json',tmp_path/m.FRESH)
        else:result=m.run(tmp_path,tmp_path/'contract.json',tmp_path/m.FRESH)
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json');assert len(count)==1 and stop['retry']==stop['resume']==0
    assert stop['mandatory_stop'] and stop['failed']==(fault=='tensor')
    if fault is None:
        assert stop['scientific_units']==m.MAX_CALLS and stop['call_counts']==m.LIMITS
        assert result['status']=='NUMERICAL_LP_IMPROVEMENT' and result['preservation']['status']=='NUMERICAL_PRESERVATION_PASS'
        assert result['shots'] is result['native_cost'] is result['total_resources'] is None
    elif fault=='solver':
        assert result['status']=='UNRESOLVED_LP' and result['candidate'] is None
        assert not (tmp_path/m.FRESH/'candidate.npz').exists()
    else:assert (tmp_path/m.FRESH/'LP_raw.npz').exists() and not (tmp_path/m.FRESH/'PF_RTE.json').exists()
    assert all(m.io.sha(tmp_path/p)==h for p,h in protected.items())
    with pytest.raises(ValueError,match='FRESH_LP'):m.run(tmp_path,tmp_path/'contract.json',tmp_path/m.FRESH)


def test_locked_ffsim_symmetric_pack_compatibility_no_rotation_codec_or_optimizer():
    python=Path('/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python')
    if not python.exists():pytest.skip('separate historical ffsim test environment unavailable')
    code="""import sys;sys.path.insert(0,'src')
import numpy as np,importlib.metadata
from ffsim.linalg.util import real_symmetrics_to_parameters,real_symmetrics_from_parameters
from trottertracks import c_fixed_u_tail_lp as m
assert importlib.metadata.version('ffsim')=='0.0.83'
r=np.random.default_rng(7);a=r.normal(size=(6,4,4));J=a+a.transpose(0,2,1)
np.testing.assert_array_equal(m.pack(J),real_symmetrics_to_parameters(J))
np.testing.assert_array_equal(m.unpack(m.pack(J),6,4),real_symmetrics_from_parameters(m.pack(J),4,6))
print('ffsim0.0.83 J pack/unpack PASS; no U codec/fit/optimizer called')
"""
    subprocess.run([str(python),'-B','-c',code],cwd=ROOT,check=True,capture_output=True,text=True)
