"""Artificial evidence and metadata only; absent public B2 is never invented."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from trottertracks import c_b1_b2_scope_integration as m

ROOT=Path(__file__).resolve().parents[2]


def records():
    spaces=dict(synthesis='full512_native',categorical='ideal_trajectory_signal',
        reference_bridge='sector36_reference',reference_uncertainty='sector36_reference')
    return {name:dict(identity=m.identity(),quantity=name,norm='raw_complex_signal',
        included_error_terms=[name],space=spaces[name],phase_sensitive=True,
        actual_inverse_and_shells=True,coherent_model='independent_IID_kernels',
        qualification='bound',scope='all_words' if name=='synthesis' else 'model_reference',
        rows=[dict(m=i,quantity=name,norm='raw_complex_signal',bound='1/100000000',
            kind='numerical_bound',provenance='invented qualified bound,not H4',
            **({'reference_uncertainty_included':False} if name=='reference_bridge' else {})) for i in range(13)]) for name in m.iid.PARTS}


def fixture():
    from test_research_c_all_law_signal_ca import packet,premises
    return packet(),records(),dict(B1_result=m.BASE,B2_result='b'*40,B2_SOURCE='c'*40),premises()


def test_common_identity_and_distinct_role_sources_reuse_existing_qualification():
    p,r,pubs,cov=fixture();attached,q=m.integrate(p,r,pubs,cov)
    assert q['eligible'] and attached['error_evidence'] is None
    assert q['iid_qualification']['kind']=='IID_GRID_CONDITIONAL_NUMERICAL_BOUND'
    assert not q['PCG64_joint_stream_qualified'] and q['PCG64_joint_stream_error'] is None
    assert q['rigorous_CA'] is q['actual_CA_achieved'] is None


def test_missing_public_B2_blocks_even_numerically_populated_artificial_records():
    p,r,pubs,cov=fixture();pubs['B2_result']=None
    with patch.object(m.iid,'qualify',side_effect=AssertionError('NO_QUALIFICATION_WITHOUT_B2')):
        attached,q=m.integrate(p,r,pubs,cov)
    assert not q['eligible'] and q['iid_qualification'] is None
    with patch.object(m.iid,'assess',side_effect=AssertionError('NO_STAGE_C')):
        result=m.assess_if_ready(None,attached,q)
    assert result['rounds']==[] and result['energy'] is None


@pytest.mark.parametrize('field',['candidate','PR_scientific_SOURCE','stage_A_scientific_SOURCE','model','compiler','conditions','accuracy_target','phase_policy','native_policy'])
def test_scientific_identity_or_model_mismatch_refused(field):
    p,r,pubs,cov=fixture();r['reference_bridge']['identity'][field]='wrong'
    with pytest.raises(ValueError,match='CANDIDATE_SCIENTIFIC'):m.integrate(p,r,pubs,cov)


@pytest.mark.parametrize('fault',['units','duplicate','PF','DF','phase','sector_native','inverse','coherent','rounds'])
def test_partition_scope_and_double_count_errors(fault):
    p,r,pubs,cov=fixture()
    if fault=='units':r['reference_uncertainty']['norm']='Ha'
    if fault=='duplicate':r['reference_uncertainty']['included_error_terms']=['reference_bridge']
    if fault in ('PF','DF'):r['reference_bridge']['included_error_terms']=[fault]
    if fault=='phase':r['synthesis']['phase_sensitive']=False
    if fault=='sector_native':r['synthesis']['space']='sector36_reference'
    if fault=='inverse':r['synthesis']['actual_inverse_and_shells']=False
    if fault=='coherent':r['synthesis']['coherent_model']='reuse_same_realized_kernel'
    if fault=='rounds':r['categorical']['rows'].pop()
    with pytest.raises(ValueError):m.integrate(p,r,pubs,cov)


@pytest.mark.parametrize('name',m.iid.PARTS)
def test_unknown_component_is_not_zero(name):
    p,r,pubs,cov=fixture();r[name]=None
    attached,q=m.integrate(p,r,pubs,cov)
    assert not q['eligible'] and q['classification'][name]=='UNAVAILABLE'
    assert attached['error_evidence'] is None


@pytest.mark.parametrize('scope',['fixed_finite_family','probabilistic_expectation'])
def test_finite_or_probabilistic_scope_needs_a_separate_qualification(scope):
    p,r,pubs,cov=fixture();r['synthesis']['scope']=scope
    _,q=m.integrate(p,r,pubs,cov)
    assert not q['eligible'] and 'NOT_OLD_ALL_WORD' in q['classification']['synthesis']


@pytest.mark.parametrize('kind',['precision_sensitivity','numerical_estimate','numerical_observation'])
def test_sensitivity_and_estimate_never_become_bound(kind):
    p,r,pubs,cov=fixture();r['reference_uncertainty']['qualification']=kind
    _,q=m.integrate(p,r,pubs,cov);assert not q['eligible']
    r['reference_uncertainty']['rows'][0].update(kind='rigorous_bound',rounding_enclosed=True)
    with pytest.raises(ValueError,match='NOT_RIGOROUS'):m.integrate(p,r,pubs,cov)


def test_confidence_failure_claims_are_separate_and_policy_is_not_adopted():
    args=dict(measurement_alpha='1/20',calibration_delta='1/100',reported_alpha='1/20')
    assert not m.confidence_schema(**args,mode='unconditional_union')['compatible']
    r=m.confidence_schema(**args,mode='conditional_on_calibration_event')
    assert r['compatible'] and not r['adopted'] and not r['measurement_policy_changed']
    args['calibration_delta']=None
    assert not m.confidence_schema(**args,mode='unconditional_union')['compatible']


def test_synthetic_partial_round_STOP_uses_unchanged_engine():
    from trottertracks.c1_stage3_accuracy import AccuracyTarget
    from trottertracks.c_pr_aware_df_stage_c import TARGET
    p,r,pubs,cov=fixture();attached,q=m.integrate(p,r,pubs,cov)
    p=attached;p['rounds'][2]['reference']['pf_operator_error']=2.
    calls=[]
    def call(k,l,f):calls.append((k,l));return f()
    target=AccuracyTarget(**TARGET)
    result=m.assess_if_ready(target,p,q,call)
    assert len(result['rounds'])==3 and result['energy'] is None
    assert not any(k=='energy_assessment' for k,l in calls)
    assert result['actual_CA_achieved'] is None


def test_no_scientific_entry_and_real_passive_metadata():
    with pytest.raises(PermissionError):m.run_science()
    with pytest.raises(NotImplementedError,match='B2_PENDING'):m.run_science(execute_science=True)
    code='''
import sys
from pathlib import Path
class Block:
 def find_spec(self,fullname,*a):
  if fullname.split('.')[0] in ('numpy','scipy','qiskit','pyscf','openfermion','ffsim','jax','flint'):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from trottertracks import c_b1_b2_scope_integration as m
r=m.preflight(Path.cwd(),Path(m.CONTRACT))
assert not r['execution_ready'] and r['scientific_calls']==0
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
