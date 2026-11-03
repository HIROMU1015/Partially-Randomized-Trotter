"""Invented distributions/scalars/circuits only; no H4 arrays or fresh sample."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from unittest.mock import patch
import json
import math
import numpy as np
import pytest
from trottertracks import c_iid_grid_expected_bias as m
from trottertracks.c_iid_grid_bridge import MODEL


def policy(n):
    return dict(model=deepcopy(MODEL), PCG64_joint_stream_qualified=False,
        origin='fresh_IID_grid_calibration', pool_old_samples=False, adaptive=False,
        fixed_samples=n, reuse='calibration_only_not_measurement_shots', iid_assumed=True,
        sample_ids=['invented:'+str(i) for i in range(n)])


def support(M='2'):
    return dict(upper_exact=M, qualification='deterministic_population_bound', provenance='invented exact unitary theorem')


def enclosure(lo='0', hi='1/100'):
    return dict(kind='rigorous_enclosure', lower_exact=lo, upper_exact=hi, provenance='invented true interval')


def test_scalar_arithmetic_outward_and_one_sided_nonzero_uncertainty():
    for x in ('0', '2', '1/3', '1000000000000000000000000001/19'):
        assert m.sqrt_upper(x)**2 >= F(x)
    assert m.log_inverse_upper('1/20') >= F.from_float(math.log(20)) - F(1,10**15)
    r = m.mean_upper([enclosure()]*20, support=support(), delta='1/20', policy=policy(20))
    assert F(r['bound']) >= F(1,100)+m.hoeffding_radius(2,20,'1/20')
    assert r['all_word_uniform'] is False and r['actual_CA_achieved'] is None
    assert m.planned_samples(2,'1/100','1/20') >= 59914


def test_hoeffding_exact_enumeration_bernoulli_coverage():
    n=8; delta=F(1,10); violations=0
    for bits in product((0,1), repeat=n):
        samples=[enclosure(str(x),str(x)) for x in bits]
        r=m.mean_upper(samples,support=support('1'),delta=delta,policy=policy(n))
        violations += F(r['bound']) < F(1,2)
    assert F(violations,2**n) <= delta


@pytest.mark.parametrize('fault',['negative','outside','reversed','sample_max','fixed_word','old_native32',
    'adaptive','duplicate_id','PCG64','not_IID','wrong_count'])
def test_invalid_range_or_dependencies_refused(fault):
    rows=[enclosure()]; s=support(); p=policy(1)
    if fault=='negative': rows[0]['lower_exact']='-1'
    if fault=='outside': rows[0]['upper_exact']='3'
    if fault=='reversed': rows[0].update(lower_exact='1',upper_exact='0')
    if fault=='sample_max': s['qualification']='observed_maximum'
    if fault in ('fixed_word','old_native32'): p['origin']=fault
    if fault=='adaptive': p['adaptive']=True
    if fault=='duplicate_id': p['sample_ids']=['same','same']
    if fault=='PCG64': p['model']=dict(MODEL,id='PCG64_PROVED')
    if fault=='not_IID': p['iid_assumed']=False
    if fault=='wrong_count': p['fixed_samples']=2
    with pytest.raises(ValueError): m.mean_upper(rows,support=s,delta='1/20',policy=p)


@pytest.mark.parametrize('missing',['support','enclosure','observation'])
def test_unknowns_remain_null(missing):
    s=support(); rows=[enclosure()]
    if missing=='support':s=None
    if missing=='enclosure':rows[0]['upper_exact']=None
    if missing=='observation':rows[0]['kind']='observed_residual'
    assert m.mean_upper(rows,support=s,delta='1/20',policy=policy(1))['bound'] is None


def test_signed_complex_pair_joint_13_round_event_allocation():
    samples=[dict(kind='rigorous_enclosure',X=['1/100','11/1000'],Y=['-1/50','-19/1000'],provenance='invented')]*100
    s=dict(lower_exact='-2',upper_exact='2',qualification='deterministic_population_bound',provenance='exact support theorem')
    records=[m.paired_round_upper(samples,axis_support=s,delta_round='1/260',policy=policy(100)) for i in range(13)]
    assert sum(F(r['delta_cal']) for r in records)==F(1,20)
    assert all(F(r['bound']) > F(1,50) for r in records)
    samples[0]=dict(samples[0],Y=None)
    assert m.paired_round_upper(samples,axis_support=s,delta_round='1/260',policy=policy(100))['bound'] is None


def test_grid_boundaries_and_zero_bins_match_numpy_searchsorted_without_sampling():
    cdf=[0.,.25,.25,.75,1.]
    for k in (0,1,1<<51,(1<<51)+1,3<<51,(1<<53)-1):
        assert m.grid_index(k,cdf)==np.searchsorted(cdf,k/(1<<53),side='right')
    with pytest.raises(ValueError):m.grid_index(1<<53,cdf)
    with pytest.raises(ValueError):m.grid_index(1,[.8,.5,1.])


def test_mean_unitary_channel_IID_shared_word_and_correlated_inverse():
    Z=np.diag([1.,-1.]); I=np.eye(2); H=np.array([[1.,1.],[1.,-1.]])/math.sqrt(2)
    U=[I,Z]; avg=sum(U)/2
    iid=sum((b@a for a,b in product(U,repeat=2)))/4
    shared=sum((a@a for a in U))/2
    assert np.allclose(iid,avg@avg) and not np.allclose(iid,shared)
    rho=H@np.diag([1.,0.])@H
    channel=sum((a@rho@a.conj().T for a in U))/2
    assert not np.allclose(channel,avg@rho@avg.conj().T)
    correlated=sum((a.conj().T@a for a in U))/2
    redraw=sum((a.conj().T@b for a,b in product(U,repeat=2)))/4
    assert np.allclose(correlated,I) and not np.allclose(correlated,redraw)


def test_actual_inverse_and_complex_XY_with_phase_sensitive_synthetic_circuits():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from trottertracks.c1_stage2_native import actual_adjoint
    from trottertracks.c1_stage3_connection import readout_segments
    theta=.3;qc=QuantumCircuit(2);qc.cp(theta,1,0)
    # System |1>, ancilla |0>; measured Y convention is +Im.
    initial=np.array([0.,1.,0.,0.],complex);observed=[]
    for axis in ('X','Y'):
        head,tail=readout_segments(1,axis);tail.remove_final_measurements()
        v=Operator(tail).data@Operator(qc).data@Operator(head).data@initial
        observed.append(float(np.sum(abs(v[:2])**2)-np.sum(abs(v[2:])**2)))
    assert complex(*observed)==pytest.approx(np.exp(1j*theta))
    assert np.allclose(Operator(actual_adjoint(qc)).data@Operator(qc).data,np.eye(4))
    changed=qc.copy();changed.global_phase=.2
    assert not np.allclose(Operator(changed).data,Operator(qc).data)


@pytest.mark.parametrize('a',['1','1001/1000'])
def test_expectation_propagation_reuses_existing_both_norm_formula(a):
    from trottertracks.c1_h4_full_synthesis import propagate
    shells={k:'1/100000' for k in ('head','X','Y')}
    for i in (0,2,5):
        ours=m.propagate_expected('1/1000000',norm_upper=a,shells=shells,m=i)
        old=propagate(dict(operator_bound_pilot=1e-6,norm_upper_pilot=float(F(a))),
            {k:dict(operator_bound_pilot=float(F(v)),norm_upper_pilot=float(F(a))) for k,v in shells.items()},
            s=1,m=i,saved_pf_error=0.)  # Pure transport leg, no H4 PF error replaced.
        assert float(F(ours['bound']))==pytest.approx(old['complex_signal_bound_pilot'],rel=2e-14)
    assert m.propagate_expected(None,norm_upper=a,shells=shells,m=12)['bound'] is None
    with pytest.raises(ValueError):m.propagate_expected('1/10',norm_upper='99/100',shells=shells,m=1)


def test_state_specific_kernel_signal_does_not_bound_later_state_actions():
    U=np.eye(2);V=np.diag([1.,-1.]);psi=np.array([1.,0.]);H=np.array([[1.,1.],[1.,-1.]])/math.sqrt(2)
    assert np.vdot(psi,(U-V)@psi)==0
    assert abs(np.vdot(H@psi,(U-V)@H@psi)) > .9


def test_sample_enclosure_scope_actual_word_unknown_and_b2_refusal():
    a=dict(scope='per_word_full512',word_id='invented',leg='compiled_to_raw',kind='rigorous_enclosure',
        provenance='invented gate certificate',rounding_enclosed=True,phase_fit=False,upper_exact='1/1000')
    b=dict(a,leg='raw_to_literal_atom',upper_exact='1/10000')
    assert F(m.sample_enclosure(a,b,word_id='invented')['upper_exact'])==F(11,10000)
    assert m.sample_enclosure(a,None,word_id='invented')['upper_exact'] is None
    with pytest.raises(ValueError):m.sample_enclosure(a,dict(b,scope='sector36_reference'),word_id='invented')
    with pytest.raises(ValueError):m.sample_enclosure(a,b,word_id='other')


def record():
    from trottertracks.c_b1_b2_scope_integration import identity
    return dict(identity=identity(),scope='probabilistic_expectation',quantity=m.TOTAL_ESTIMAND,norm='raw_complex_signal',
        all_word_uniform=False,included_error_terms=list(m.TERMS),qualification='bound_from_qualified_enclosures',
        state_coverage='all_normalized_sector_inputs_embedded_full512',provenance='invented direct total bound',
        phase_sensitive=True,shells_included=True,fresh_IID_calibration=True,fixed_N_no_optional_stopping=True,
        coherent_model='independent_IID_kernels',calibration_events=[dict(id='shared_kernel_mean',delta='1/20')],
        rows=[dict(m=i,bound='1/1000000',calibration_event_ids=['shared_kernel_mean']) for i in range(13)])


def test_probability_qualification_separate_and_existing_accuracy_reused_only_if_adopted():
    from test_research_c_b1_b2_scope_integration import fixture
    from trottertracks.c1_stage3_accuracy import AccuracyTarget
    p,_,_,_=fixture();q=m.qualify(record());p['expectation_calibration_identity']=q['identity']
    assert q['eligible'] and q['all_word_uniform'] is False and q['actual_error_evidence'] is None
    assert q['confidence']['unconditional_failure_upper']=='3/50'
    assert m.assess_if_adopted(None,p,q)['rounds']==[]
    target=AccuracyTarget(.00159,1/8,.1,math.pi/6,math.pi/12,.01/26,.01)
    value=m.assess_if_adopted(target,p,q,policy_adopted=True)
    assert len(value['rounds'])==13 and value['actual_CA_achieved'] is None
    assert value['confidence']['conditional_RPE_failure_upper']=='1/100'


@pytest.mark.parametrize('fault',['candidate','SOURCE','model','round','unknown','double_count','order','null_error',
    'finite','uniform','event_missing','event_conflict','unqualified','no_state','no_shell'])
def test_accuracy_rejects_missing_or_bad_qualification(fault):
    r=record();extra=()
    if fault=='candidate':r['identity']['candidate']='wrong'
    if fault=='SOURCE':r['identity']['PR_scientific_SOURCE']='wrong'
    if fault=='model':r['identity']['model']['id']='PCG64'
    if fault=='round':r['rows'].pop()
    if fault=='unknown':r['rows'][1]['bound']=None
    if fault=='double_count':extra=(deepcopy(r),)
    if fault=='order':r['included_error_terms'].reverse()
    if fault=='null_error':r.update(quantity=m.ESTIMAND,included_error_terms=['native_to_q_atom'])
    if fault=='finite':r['scope']='fixed_finite_family'
    if fault=='uniform':r['all_word_uniform']=True
    if fault=='event_missing':r['rows'][0]['calibration_event_ids']=[]
    if fault=='event_conflict':r['calibration_events'].append(dict(id='shared_kernel_mean',delta='1/10'))
    if fault=='unqualified':r['qualification']='observed_residual'
    if fault=='no_state':r['state_coverage']='saved_vector_uncertified'
    if fault=='no_shell':r['shells_included']=False
    if fault in ('unknown','null_error','unqualified','no_state','no_shell'):
        q=m.qualify(r,additional=extra);assert not q['eligible']
        assert m.assess_if_adopted(None,{},q,policy_adopted=True)['rounds']==[]
    else:
        with pytest.raises(ValueError):m.qualify(r,additional=extra)


def test_confidence_event_reuse_is_once_and_no_free_joint_alpha():
    q=m.qualify(record())
    assert q['confidence']['calibration_success_lower']=='19/20'
    assert q['confidence']['unconditional_failure_upper']!='1/100'
    assert m.failure_accounting(None)['unconditional_failure_upper'] is None


def test_no_scientific_execution_even_after_generic_permission():
    with pytest.raises(PermissionError):m.run_science()
    with pytest.raises(NotImplementedError,match='NOT_READY'):m.run_science(execute_science=True)


def test_passive_preflight_source_fresh_retry_and_stop(tmp_path):
    f=tmp_path/'tiny.txt';f.write_text('invented')
    import hashlib
    c=dict(scope=m.SCOPE,base_commit=m.BASE,executable=False,maximum_logical_units=None,retry=0,resume=0,
        mandatory_stop=True,source_files={'tiny.txt':hashlib.sha256(f.read_bytes()).hexdigest()},saved_inputs={},fresh_output='run01')
    manifest=tmp_path/'SOURCE.json';manifest.write_text(json.dumps(dict(files=c['source_files'])))
    c.update(source_manifest='SOURCE.json',source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    p=tmp_path/'contract.json';p.write_text(json.dumps(c))
    assert m.preflight(tmp_path,p)['scientific_calls']==0
    (tmp_path/'run01').mkdir()
    with pytest.raises(ValueError,match='FRESH'):m.preflight(tmp_path,p)
    (tmp_path/'run01').rmdir();c['retry']=1;p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='FIXED'):m.preflight(tmp_path,p)



def test_known_bias_has_full_operator_bound_and_complex_propagation():
    # A known phase error on the controlled-1 branch; all normalized states.
    U=np.eye(4,dtype=complex);C=np.diag([1.,1.,1.,np.exp(.001j)])
    mu=np.linalg.norm(C-U,2)
    bound=m.propagate_expected(str(mu),norm_upper='1',shells={k:'0' for k in ('head','X','Y')},m=4)
    exact=abs(np.exp(16*.001j)-1)
    assert exact <= float(F(bound['bound']))


def test_primary_bias_and_saved_reference_total_are_distinct():
    r=record();r.update(quantity=m.ESTIMAND,included_error_terms=[m.TERMS[0]])
    q=m.qualify(r)
    assert not q['eligible'] and 'complete_transport_to_binary64_CAR_reference' in q['missing']
    with pytest.raises(ValueError,match='ESTIMAND'):
        m.qualify(dict(r,included_error_terms=list(m.TERMS)))


def test_passive_real_design_without_scientific_imports():
    import subprocess,sys
    code="""
import sys
from pathlib import Path
class Block:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('numpy','scipy','qiskit','pyscf','ffsim','openfermion','flint'):
   raise AssertionError('NO_SCIENTIFIC_IMPORT:'+fullname)
sys.meta_path.insert(0,Block())
from trottertracks import c_iid_grid_expected_bias as m
r=m.preflight(Path.cwd(),m.CONTRACT)
assert r['scientific_calls']==r['arrays_decoded']==r['samples_generated']==0
assert r['execution_ready'] is False
"""
    subprocess.run([sys.executable,'-c',code],check=True)



def test_accuracy_packet_identity_not_transferred_from_unrelated_saved_target():
    from test_research_c_b1_b2_scope_integration import fixture
    p,_,_,_=fixture();q=m.qualify(record())
    with pytest.raises(ValueError,match='PACKET_CANDIDATE'):
        m.assess_if_adopted(None,p,q,policy_adopted=True)
