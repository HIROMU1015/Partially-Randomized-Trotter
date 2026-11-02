"""Synthetic implementation checks; do not execute the frozen scientific inputs."""
from fractions import Fraction as F
import itertools
import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from trottertracks.representation_exploration import n13_redesign as n
from trottertracks.representation_exploration.algorithm_design import rotation


@pytest.mark.parametrize('nu',[0,1,2])
def test_car_direction(nu):
    v=np.array([.2,-.3,.1]);h=np.zeros((4,4));h[:3,3]=v;h[3,:3]=v
    full=n.second_quantize_one_body(h);sec=n.sector_indices(4,nu);a=[s for s in sec if not(s>>3&1)];b=[s for s in sec if s>>3&1]
    B=full[np.ix_(a,b)];rank=n.second_quantize_one_body(np.outer(v,v))[np.ix_(a,a)]
    assert np.allclose(B@B.T.conj(),rank,atol=2e-16)


def test_gaussian_interval_and_orientation():
    u=rotation(3,0,2,.23)@rotation(3,1,2,-.31)
    qc=n.gaussian_circuit(u.astype(complex));q=n.gaussian_onebody(qc);numeric=Operator(qc).data;one=numeric[np.ix_([1,2,4],[1,2,4])]/numeric[0,0]
    assert np.linalg.norm(n.ivmid(q)-one)<2e-15
    assert np.linalg.norm(n.ivmid(q)-u)<2e-15
    # Interval endpoints are compared to high-precision evaluation of exact gate angles.
    import mpmath as mp
    with mp.workprec(256):
        a=F(.23);y=mp.sin(mp.mpf(a.numerator)/a.denominator)
        lo,hi=n.endpoints(mp.iv.sin(n.exact_iv(a)))
        assert mp.mpf(lo.numerator)/lo.denominator<=y<=mp.mpf(hi.numerator)/hi.denominator


def test_pauli_interval_vs_explicit_jw():
    g=np.array([[.2,.13+.07j],[.13-.07j,-.3]])
    terms=n.squared_interval(n.ivmatrix(g),-.4);target=-.4*n.second_quantize_one_body(g)@n.second_quantize_one_body(g)
    from qiskit.quantum_info import Pauli
    built=sum(float(sum(n.endpoints(v.real))/2)*Pauli(k).to_matrix() for k,v in terms.items())
    assert np.linalg.norm(built-target)<2e-16


def test_exact_combined_commutators():
    certs=[{'X':[n.D,n.D]},{'Z':[n.D,n.D]}]
    assert n.fs(n.pf_certificate(certs)['alpha2'])==F(1,2)
    assert n.fs(n.pf_certificate([{'Z':[n.D,n.D]},{'Z':[n.D,n.D]}])['alpha2'])==0
    noisy=[{'X':[n.D-1,n.D+1]},{'Z':[n.D-1,n.D+1]}]
    assert n.fs(n.pf_certificate(noisy)['alpha2'])>=F(1,2)


def synthetic_h(external=10,coupled=True):
    h=[[F(0) for _ in range(4)] for _ in range(4)]
    for i,e in enumerate([0,1,2,external]):h[i][i]=F(e)
    if coupled:h[1][3]=h[3][1]=F(1,10)
    h[0][1]=h[1][0]=F(1,20)
    return h


def test_ldl_spd_and_exact_inverse():
    a=[[F(2),F(1)],[F(1),F(3)]];ldl=n.rational_ldl(a)
    assert n.quadratic_inverse(ldl,[F(1),F(2)])==F(7,5)
    assert n.rational_ldl([[F(0)]]) is None
    assert n.rational_ldl([[F(-1)]]) is None


def test_direction_non_diagonal_and_soundness():
    h=synthetic_h();cert=n.direction_certificate(h,{},[0,1,2],2,F(1,100),'direction')
    assert cert['accepted'];H=n.second_quantize_one_body(np.array(h,float));sec=n.sector_indices(4,2);active=[s for s in sec if not(s>>3&1)]
    assert np.linalg.eigvalsh(H[np.ix_(active,active)])[0]-np.linalg.eigvalsh(H[np.ix_(sec,sec)])[0]<=.01


def test_zero_coupling_requires_external_lower_bound():
    assert n.direction_certificate(synthetic_h(coupled=False),{},[0,1,2],2,n.GOAL,'direction')['accepted']
    assert not n.direction_certificate(synthetic_h(-10,False),{},[0,1,2],2,n.GOAL,'direction')['accepted']


def test_attractive_density_and_two_external_refused():
    with pytest.raises(ValueError,match='attractive'):n.direction_certificate(synthetic_h(),{(0,1):F(-1)},[0,1,2],2,n.GOAL,'direction')
    with pytest.raises(ValueError,match='one external'):n.direction_certificate(synthetic_h(),{},[0,1],2,n.GOAL,'direction')


def test_goal_boundary_exact_rational():
    # Norm-only acceptance includes equality without tolerance.
    h=[[F(0),F(0),F(0)],[F(0),F(1),F(1)],[F(0),F(1),F(2)]]
    cert=n.direction_certificate(h,{},[0,1],1,F(1),'norm_only')
    assert cert['accepted']


def test_legacy_blocks_and_oracle_isolation(monkeypatch):
    h=synthetic_h(20);h[2][2]=F(10);active=[0,1]
    cert=n.legacy_certificate(h,{},active,2,n.GOAL)
    assert cert['accepted']
    H=n.second_quantize_one_body(np.asarray(h,float));sec=n.sector_indices(4,2);a=[s for s in sec if not(s&12)]
    assert np.linalg.eigvalsh(H[np.ix_(a,a)])[0]-np.linalg.eigvalsh(H[np.ix_(sec,sec)])[0]<=float(n.GOAL)
    def forbidden(*a,**k):raise AssertionError('spectral oracle in coefficient selector')
    monkeypatch.setattr(np.linalg,'eigvalsh',forbidden);monkeypatch.setattr(np.linalg,'eigh',forbidden)
    assert n.legacy_certificate(h,{},active,2,n.GOAL)['accepted']
    n.direction_certificate(synthetic_h(),{},[0,1,2],2,n.GOAL,'direction')


def test_scalar_control_and_vectorized_action(tmp_path):
    blocks=[(QuantumCircuit(4),{'ZIII':F(1,8),'IIII':F(1,7)})]
    qc,rounding=n.build_product(blocks,.11,2,'Y');assert rounding>0
    target=n.wrapper_reference(n.ideal_operator(blocks,.11,2),'Y')
    compiler=n.Compiler(tmp_path);cost=compiler.compile(qc,'synthetic',target)
    import gzip
    ir=json.loads(gzip.decompress((tmp_path/cost['ir_file']).read_bytes()))
    assert np.linalg.norm(n.native_action(ir,np.eye(32,dtype=complex))-target)<3e-9
    assert compiler.compile(qc,'same',target)==cost
    assert compiler.calls==1 and compiler.hits==1


def test_oracle_input_refused(monkeypatch,tmp_path):
    # Exact whitelist, independently of the bound input fingerprint.
    c={k:None for k in n.CONTEXT_KEYS};c['generator_candidates']=[];c['true_pf_error']=0
    data=n.canonical({'contexts':[c]});p=tmp_path/'input.json';p.write_bytes(data)
    monkeypatch.setattr(n,'INPUT_SHA',__import__('hashlib').sha256(data).hexdigest())
    with pytest.raises(ValueError,match='oracle'):n.load_contexts(p)


def test_weighted_l1_lp_formulation():
    # Independent synthetic J: fixture in run_n2 is not executed by this test.
    S=np.array([[1,0],[0,1],[1,1]],int);pairs=list(itertools.combinations_with_replacement(range(3),2))
    w=np.array([1 if i==j else 2 for i,j in pairs]);K=np.array([[.1,.03],[.03,.2]]);J=S@K@S.T;J[0,2]+=.017;J[2,0]+=.017
    assert sum(w[i]*abs(J[a,b]-(S@K@S.T)[a,b]) for i,(a,b) in enumerate(pairs))==pytest.approx(np.sum(abs(J-S@K@S.T)))


def test_safe_shots_rounding_and_ineligible():
    assert n.safe_shots(F(0))>=math_shots(0)
    assert n.safe_shots(F(1)) is None


def math_shots(b):
    import math
    return math.ceil(2*math.log(80)/(.04/math.sqrt(2)-b)**2)


def test_synthetic_n1_end_to_end(tmp_path,monkeypatch):
    monkeypatch.setattr(n.resource,'setrlimit',lambda *a:None)
    identity=n.mj(np.eye(4));factors=[];inputs=[]
    for j,e in enumerate([[.1,.15,-.03,.01],[.07,-.04,.02,.03]]):
        g=n.mj(np.diag(e));inputs.append(g)
        factors.append([{'id':f'synthetic/factor{j}/exact','family':'exact','eta':sorted(e),'approx_eta':e,'frame':identity,'ungauged_frame':identity,'ungauged_eta':e,'g':g,'groups':[[i] for i in range(4)]}])
    c={'context':'synthetic_reference','modes':4,'particles':2,'df_rank':2,'weights':[.3,-.1],
       'one_body':n.mj(np.diag([.01,-.02,.03,.04])),'scalar':.07,'input_factors':inputs,'time':.6,'model_budget':.02,'q_grid':[1,2,4,8],'generator_candidates':factors}
    result=n.run_n1((c,str(tmp_path/'synthetic')))
    saved=json.loads((tmp_path/'synthetic/result.json').read_bytes())
    assert result['status']=='LIMITED_N1_COMPLETE'
    assert len(saved['rows'])==8 and len(saved['whole_pauli'])==12
    assert all(x['pf_error']<2e-15 for x in saved['diagnostics']['rows'])


def test_independent_saved_audit_primitives(tmp_path):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('saved_n13_auditor',Path(__file__).resolve().parents[1]/'scripts/verify_n13_redesign.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    certs=[{'X':[n.D,n.D]},{'Z':[n.D,n.D]}]
    assert audit.independently_pf(certs)==F(1,2)
    qc=QuantumCircuit(2);qc.h(0);qc.cx(0,1);qc.p(.17,1)
    compiler=n.Compiler(tmp_path);cost=compiler.compile(qc,'reference')
    import gzip
    ir=json.loads(gzip.decompress((tmp_path/cost['ir_file']).read_bytes()))
    assert np.linalg.norm(audit.native(ir)-Operator(qc).data)<3e-9
