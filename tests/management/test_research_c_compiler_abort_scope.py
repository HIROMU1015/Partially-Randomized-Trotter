"""Invented saved outcomes and scalar policies; no real sampling/compilation."""
from copy import deepcopy
from fractions import Fraction as F
import pytest

from trottertracks import c_compiler_abort_scope as m
from trottertracks import c_fresh_iid_calibration as fresh
from trottertracks import c_iid_calibration_integration as integration
from trottertracks import c_pr_aware_df_fixed_certificate as fixed
from trottertracks.c_b1_b2_scope_integration import identity
from trottertracks.c_pr_aware_df_stage_c2 import choice_grid


def fixtures(statuses):
    c=fresh.make_context(source_id='3'*64,identity=identity(),atom_ids=['INVENTED:atom'],
        table=choice_grid([1.]),event_angle_bits={'INVENTED:atom':0.0.hex()})
    rows=[]
    for i,status in enumerate(statuses):
        s=fresh.sample_from_integers(c,dataset_id='INVENTED-abort-scope',sample_index=i,
            integers=[[0]*64],unit='kernel',axis='kernel')
        raw=integration.bind_operator(fixed.identity(3),s,c,0,role='raw')
        compiled=integration.bind_operator(fixed.identity(3),s,c,0,role='compiled',raw_id=raw['record_id'])
        part=dict(occurrence=0,raw=raw,compiled=compiled,raw_status='SUCCESS',compile_status='SUCCESS')
        if status=='RAW_BUILD_FAILED':part.update(raw_status='FAILED',compile_status='UNKNOWN')
        if status=='P0_COMPILE_FAILED':part.update(compile_status='FAILED')
        if status=='INCOMPLETE':part.update(compile_status='NONTERMINATION_UNRESOLVED')
        if status=='NATIVE_INELIGIBLE':part.update(compile_status='NATIVE_INELIGIBLE')
        rows.append(fresh.outcome(s,status=status,stage=fresh.FAILURE_STAGES.get(status),
            error=RuntimeError('INVENTED_FAILURE') if status in fresh.FAILURE_STAGES else None,
            evidence=dict(occurrences=[part])))
    return c,rows,[x['sample']['sample_id'] for x in rows]


def test_unadopted_defaults_no_invented_N_or_delta():
    c,rows,ids=fixtures(['SUCCESS'])
    r=m.audit(rows,context=c,planned_ids=ids)
    assert all(x['failure_upper'] is None for x in r['rates'].values())
    p=m.requirements()
    for k in ('N','delta_compile','delta_cal','operational_abort_allowance','cache_policy','nontermination_policy'):
        assert p[k] is None
    assert p['executable'] is False and m.operational_requests()['requests'] is None
    with pytest.raises(PermissionError):m.run_science()
    with pytest.raises(NotImplementedError):m.run_science(execute_science=True)


def test_denominators_and_certificate_failures_not_compiler_failures():
    c,rows,ids=fixtures(['RAW_BUILD_FAILED','P0_COMPILE_FAILED','PER_WORD_CERTIFICATE_FAILED','SUCCESS'])
    r=m.audit(rows,context=c,planned_ids=ids,delta_compile='1/20',failure_policy='record_all_fixed_N',iid_assumed=True,correctness='PASS')
    a,b,d=(r['rates'][k] for k in ('pipeline','raw_build','compile_given_raw_success'))
    assert (a['denominator'],a['failures'])==(4,2)
    assert (b['denominator'],b['failures'])==(4,1)
    assert (d['denominator'],d['failures'])==(3,1)
    assert r['native_success_with_unqualified_certificate']==1
    assert r['fresh_fixed_N_runner_implemented'] is False


def test_zero_failures_confidence_not_zero_or_totality():
    c,rows,ids=fixtures(['SUCCESS']*8)
    r=m.audit(rows,context=c,planned_ids=ids,delta_compile='1/20',failure_policy='record_all_fixed_N',iid_assumed=True,correctness='PASS')
    assert 0<F(r['rates']['pipeline']['failure_upper'])<1
    assert not r['all_q_words_compile_success_qualified']


@pytest.mark.parametrize('bad',['INCOMPLETE','RAW_BUILD_FAILED'])
def test_unknown_or_stopped_prefix_gets_no_fixed_N_confidence(bad):
    c,rows,ids=fixtures(['SUCCESS',bad,'SUCCESS'])
    if bad=='RAW_BUILD_FAILED':rows=rows[:2]
    r=m.audit(rows,context=c,planned_ids=ids,delta_compile='1/20',failure_policy='record_all_fixed_N',iid_assumed=True,correctness='PASS')
    assert r['rates']['pipeline']['failure_upper'] is None
    if bad=='INCOMPLETE':assert m.continuation(bad,failure_policy='record_all_fixed_N').startswith('STOP_UNKNOWN')


@pytest.mark.parametrize('change',['discard','retry','resample','word'])
def test_identity_exclusion_retry_resampling_stop(change):
    c,rows,ids=fixtures(['SUCCESS','SUCCESS'])
    if change=='discard':rows=rows[1:]
    if change=='retry':rows[0]['retry']=1
    if change=='resample':rows[0]['resampled']=True
    if change=='word':rows[0]['evidence']['occurrences'][0]['raw']['word']=[]
    with pytest.raises(ValueError):m.audit(rows,context=c,planned_ids=ids)


def test_correctness_FAIL_stops_even_record_all_design():
    for s in fresh.STATUSES:
        assert m.continuation(s,failure_policy='record_all_fixed_N',correctness='FAIL').startswith('STOP_CORRECTNESS')
    assert m.continuation('RAW_BUILD_FAILED',failure_policy='stop_on_failure').startswith('STOP_FIRST')
    assert m.continuation('RAW_BUILD_FAILED',failure_policy='record_all_fixed_N').startswith('RECORD_FAILURE')
    assert m.continuation('PER_WORD_CERTIFICATE_FAILED',failure_policy='record_all_fixed_N').startswith('STOP_CERTIFICATE')
    c,rows,ids=fixtures(['RAW_BUILD_FAILED','SUCCESS'])
    for verdict in ('UNKNOWN','FAIL'):
        r=m.audit(rows,context=c,planned_ids=ids,delta_compile='1/20',failure_policy='record_all_fixed_N',iid_assumed=True,correctness=verdict)
        assert all(x['failure_upper'] is None for x in r['rates'].values())


def test_requests_inverse_cache_and_confidence_are_separate():
    r=m.operational_requests([1]*13,[2]*13,cache='none')
    assert r['word_draws']==3*8191 and r['actual_inverse_new_word_draws']==0
    with pytest.raises(ValueError):m.operational_requests([1]*13,[1]*13,cache='exact_word_source_environment')
    r=m.confidence([dict(id='shared',kind='calibration',delta='1/1000')]*13+
                   [dict(id='compile',kind='compile_confidence',delta='1/2000')],p_fail_upper='1/100000',requests=100)
    assert r['unique_confidence_events']==2 and r['delta_cal']=='1/1000'
    assert r['operational_abort_upper']=='1/1000' and r['delta_compile']=='1/2000'
    assert r['measurement_alpha']=='1/100' and r['unconditional_failure_upper'] is None
    assert r['joint_001_policy_adopted'] is False
