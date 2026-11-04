"""Invented sector4 data only. Real H4 files are byte-preflighted, never decoded."""
from copy import deepcopy
from fractions import Fraction as F
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from trottertracks import c_reference_bound_contract as m
from trottertracks.c1_stage3_integrated import Calls

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def toy():
    from trottertracks.c1_molecule_extension import SectorCAR
    ctx = SectorCAR(2); eye = np.eye(4,dtype=complex)
    p = eye-2*ctx.bilinears[0,0].toarray()
    tau = .125
    saved = dict(J=np.zeros((3,2,2)),U=np.array([np.eye(2)]*3),
                 generators=np.array([0*eye]*3+[p]),corrected=np.linalg.matrix_power(eye-1j*tau*p,2))
    atom = dict(atom_id='CDF_proposed:F2:0',frame_id='CDF_proposed:F2',support=[0],coefficient='1')
    d = dict(atoms=dict(atoms=[atom]),table=dict(ideal_probabilities=['1']),
             mean=dict(report=dict(angle=math.atan(tau),scalar_phase_excluded=True)),
             normalization=dict(r=2,tau=tau,lambda_R=1.,rows=[dict(m=i,attenuation=(1+tau*tau)**(-(1<<i))) for i in range(13)]),
             law=dict(lambda_R=1.,identity=0.,atoms=1))
    c = m.fixed(); c['dimensions'] = dict(spatial=2,modes=4,sector=4,rank=3,atoms=1,ld=2,r=2)
    return c,d,saved,p


def execute_toy(tmp_path,monkeypatch,toy):
    c,d,saved,p = toy
    monkeypatch.setattr(m,'metadata',lambda *a:deepcopy(d))
    monkeypatch.setattr(m,'arrays',lambda *a:deepcopy(saved))
    out = tmp_path/'artificial'; out.mkdir()
    call = Calls(out,m.LIMITS,m.MAX_CALLS)
    result = m.execute(ROOT,c,out,call,parallel=False)
    return result,call


def test_actual_dispatch_has_27_calls_and_both_reference_partitions(tmp_path,monkeypatch,toy):
    result,call = execute_toy(tmp_path,monkeypatch,toy)
    assert m.MAX_CALLS == sum(m.LIMITS.values()) == 27
    assert dict(call.counts) == m.LIMITS and not call.stop.is_set()
    assert len(result['rounds']) == 13 and result['Stage_C_ready'] is False
    assert not result['saved_ground_precision_qualified'] and not result['PCG64_joint_stream_qualified']
    assert result['actual_CA_achieved'] is None
    bridge,unc = (result['records'][k] for k in ('reference_bridge','reference_uncertainty'))
    assert set(bridge['included_error_terms']).isdisjoint(unc['included_error_terms'])
    assert all(x['reference_uncertainty_included'] is False for x in bridge['rows'])
    assert all(x['kind']=='rigorous_bound' and x['rounding_enclosed'] for r in (bridge,unc) for x in r['rows'])
    assert not result['categorical_unitary_trajectory_premise_qualified']


def test_bounds_contain_independent_high_precision_diagonal_targets(tmp_path,monkeypatch,toy):
    from trottertracks.pr_dictionary_validation.reference import precision_context,exact
    result,_ = execute_toy(tmp_path,monkeypatch,toy)
    c,d,saved,p = toy; mp = precision_context(128)
    theta = exact(mp,str(F(d['mean']['report']['angle']))); tau = exact(mp,'1/8')
    for row in result['rounds']:
        m0 = row['m']; q = 1 << m0
        atom = mp.exp(-mp.j*theta*2*q)
        car = (1-mp.j*tau)**(2*q)/(1+tau*tau)**q
        assert abs(atom-car) <= exact(mp,row['bridge']['upper_exact'])
        # Independently reproduce the actual binary64 scalar/power path as bits.
        num = d['normalization']['rows'][m0]['attenuation']*np.linalg.matrix_power(saved['corrected'],q)
        j = next(i for i in range(4) if p[i,i] == 1)
        value = mp.mpc(exact(mp,str(F(num[j,j].real))),exact(mp,str(F(num[j,j].imag))))
        assert abs(car-value) <= exact(mp,row['reference_uncertainty']['upper_exact'])
        assert abs(atom-value) <= exact(mp,row['total_upper_exact'])


def test_normalized_state_conversion_uses_no_ground_vector(tmp_path,monkeypatch,toy):
    result,_ = execute_toy(tmp_path,monkeypatch,toy)
    for row in result['rounds']:
        assert not row['saved_state_used']
        assert 'every normalized psi' in row['norm_conversion']
    # This is an operator result, not a claim that an arbitrary saved state is ground.
    for r in result['records'].values():
        assert r['error_model']['saved_state_used'] is False
        assert r['error_model']['ground_oracle_precision_claimed'] is False


@pytest.mark.parametrize('fault',['p','zero','support','frame','lambda','rounds','angle'])
def test_wrong_metadata_fails_before_reference_calculation(toy,fault):
    c,d,_,_ = toy
    if fault=='p': d['table']['ideal_probabilities']=['1/2']
    if fault=='zero': d['atoms']['atoms'][0]['coefficient']='0'
    if fault=='support': d['atoms']['atoms'][0]['support']=[0,0]
    if fault=='frame': d['atoms']['atoms'][0]['frame_id']='CDF_proposed:F9'
    if fault=='lambda': d['law']['lambda_R']=2
    if fault=='rounds': d['normalization']['rows'].pop()
    if fault=='angle': d['mean']['report']['angle']=0
    with pytest.raises(ValueError): m.bind(d,c)


def test_candidate_binding_and_array_schema_on_artificial_npz(tmp_path,toy):
    c,_,s,_ = toy
    np.savez(tmp_path/'candidate.npz',J=s['J'],U=s['U'])
    np.savez(tmp_path/'factors.npz',**s)
    c['arrays']=dict(candidate='candidate.npz',factors='factors.npz',CAR='factors.npz')
    assert np.array_equal(m.arrays(tmp_path,c)['corrected'],s['corrected'])
    np.savez(tmp_path/'candidate.npz',J=s['J']+1,U=s['U'])
    with pytest.raises(ValueError,match='FIXED_REFERENCE_ARRAYS'): m.arrays(tmp_path,c)


def test_parallel_exponentials_match_serial_with_parent_call_registration(tmp_path):
    tasks=[np.zeros((2,2),complex)]*2
    call=Calls(tmp_path,m.LIMITS,m.MAX_CALLS)
    parallel=m.parallel_calls(call,'head_exponential',tasks,m._head_task,parallel=True)
    assert call.counts['head_exponential']==2
    assert parallel==[m._head_task(x) for x in tasks]


def test_head_cache_preserves_noncommuting_forward_and_reverse_order():
    from flint import ctx
    from scipy.linalg import expm
    from trottertracks import c_pr_aware_df_zsxx_enclosure as z
    g=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1,-1]).astype(complex)]
    e=[expm(-.05j*x) for x in g]
    with z._BALL_LOCK,ctx.workprec(m.PRECISION):
        left,right=m.head_products([m._head_task(x) for x in g],2)
        assert F(z._norm(left-z._matrix(e[2]@e[1]@e[0]))['upper_exact'])<F(1,10**14)
        assert F(z._norm(right-z._matrix(e[0]@e[1]@e[2]))['upper_exact'])<F(1,10**14)
        assert F(z._norm(left-right)['lower_exact'])>F(1,1000)


def test_nonunitary_growth_is_not_clamped_and_negative_bounds_refused():
    from flint import acb_mat,ctx
    from trottertracks.c_pr_aware_df_fixed_car import pack
    with ctx.workprec(m.PRECISION):
        task=dict(m=0,atom=pack(acb_mat([[4]])),CAR=pack(acb_mat([[1]])),numeric=np.array([[1.]],complex),tau=0.,r=32,size=1)
        r=m._round_task(task)
    assert F(r['bridge']['upper_exact'])>=3>2
    rows=[dict(r,m=i) for i in range(13)]
    rows[0]['bridge']['upper_exact']='-1'
    with pytest.raises(ValueError,match='NONNEGATIVE_BOUND'):m.records(rows)


def test_unenclosed_state_specific_records_never_acquire_qualification():
    rows=[dict(m=i,bridge=dict(upper_exact='0'),reference_uncertainty=dict(upper_exact='0'),rounding_enclosed=False,saved_state_used=False) for i in range(13)]
    with pytest.raises(ValueError,match='ROUNDING_COVERAGE'):m.records(rows)
    rows[0].update(rounding_enclosed=True,saved_state_used=True)
    with pytest.raises(ValueError,match='ROUNDING_COVERAGE'):m.records(rows)


def test_integration_stays_blocked_without_B1_even_with_artificial_B2_bounds(tmp_path,monkeypatch,toy):
    from test_research_c_b1_b2_scope_integration import fixture
    result,_=execute_toy(tmp_path,monkeypatch,toy)
    packet,records,pubs,cov=fixture();records.update(result['records']);records['synthesis']=None
    _,q=m.old.integration.integrate(packet,records,pubs,cov)
    assert not q['eligible'] and q['actual_CA_achieved'] is None


def test_permission_gate_and_artificial_failure_preserve_partial_STOP(tmp_path,monkeypatch,toy):
    c,d,saved,_=toy;c['fresh_output']='run01';c['source_manifest']='source.json'
    (tmp_path/'source.json').write_text('{}')
    monkeypatch.setattr(m,'preflight',lambda *a:(c,{}))
    monkeypatch.setattr(m,'metadata',lambda *a:deepcopy(d))
    monkeypatch.setattr(m,'arrays',lambda *a:deepcopy(saved))
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'artificial-only')
    # Keep the run-path test focused on fresh/partial STOP; full graph above uses real toy numerics.
    def fail(root,c,out,call):
        call('metadata_load','artificial',lambda:None)
        return call('car_tail','artificial failure',lambda:(_ for _ in ()).throw(ValueError('invented failure')))
    monkeypatch.setattr(m,'execute',fail)
    with pytest.raises(PermissionError):m.run(tmp_path,'contract')
    assert not (tmp_path/'run01').exists()
    with pytest.raises(ValueError,match='invented failure'):m.run(tmp_path,'contract',execute_science=True)
    p=tmp_path/'run01/STOP.json';before=p.read_bytes();stop=json.loads(before)
    assert stop['failed'] and stop['scientific_units']==2 and stop['first_failure']['kind']=='car_tail'
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    with pytest.raises(FileExistsError):m.run(tmp_path,'contract',execute_science=True)
    assert p.read_bytes()==before


@pytest.mark.parametrize('field,value',[('candidate_sha256','bad'),('model',{}),('precision_bits',192),
 ('retry',1),('resume',1),('maximum_logical_units',38),('Stage_C','READY'),('metadata',{}),('arrays',{})])
def test_fixed_contract_mutation_refused_before_numerical_import(tmp_path,field,value):
    c=m.fixed();c[field]=value;p=tmp_path/'contract.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='FIXED_REFERENCE_BOUND_CONTRACT'):m.preflight(tmp_path,p)


def test_real_preflight_is_passive_and_execution_not_implicitly_authorized():
    code='''
import sys
from pathlib import Path
class Block:
 def find_spec(self,fullname,*a):
  if fullname.split('.')[0] in ('numpy','scipy','qiskit','flint','mpmath','pyscf','openfermion','jax'):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from trottertracks import c_reference_bound_contract as m
c,r=m.preflight(Path.cwd(),Path(m.CONTRACT))
assert c['execution_authorized'] is False and r['scientific_calls']==0
assert r['maximum_logical_units']==27 and not r['Stage_C_ready'] and not r['H4_bound_obtained']
try:m.run(Path.cwd(),Path(m.CONTRACT))
except PermissionError:pass
else:raise AssertionError('No implicit science')
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
