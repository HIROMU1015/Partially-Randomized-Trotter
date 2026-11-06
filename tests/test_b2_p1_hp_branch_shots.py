"""Artificial HP scalar records; no H4 inputs, PF construction or old replay."""
import copy
from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from trotterlib import b2_p1_hp_branch_shots as m
from trotterlib import b2_strong_pf_hp_requalification as qualifier
from test_b2_p1_strong_pf_branch_shots import bundle, identity


@pytest.fixture(autouse=True)
def no_scientific_generation(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('No private data, PF simulation or old science replay')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(m.old.branch,'git',forbidden)
    monkeypatch.setattr(m.old.branch,'fixed_json',forbidden)
    monkeypatch.setattr(qualifier.hp,'prepare_precision',forbidden)
    monkeypatch.setattr(qualifier.hp,'precise_step',forbidden)
    monkeypatch.setattr(qualifier,'evaluate_arm',forbidden)
    from qiskit import QuantumCircuit
    monkeypatch.setattr(QuantumCircuit,'__init__',forbidden)


def hp_arm(entry=m.ARMS[0],phase='.12',radius='.8'):
    ctx=m.context();rows=[]
    for index in range(18):
        q=1<<index
        z=ctx.mpf(radius)*ctx.exp(ctx.j*q*ctx.mpf(phase))
        z64=complex(float(z.real),float(z.imag))
        qualification=qualifier.qualify_hp(index,z,z,z64,z64,{'N':True,'C':True},ctx)
        check=dict(operator_max_abs_80_vs_120='1e-70',signal_difference={
            k:'1e-70' for k in ('complex_absolute','radius_absolute','phase_rad')},passed=True)
        rows.append(dict(m=index,Q=q,T_Ha_inverse=q*.02,status=qualification['status'],qualification=qualification,
            signals={'PF':m.hp.hp_signal_record(ctx,z),'independent_PF':m.hp.hp_signal_record(ctx,z)},
            precision_check={'N':copy.deepcopy(check),'C':copy.deepcopy(check)},
            normalization=1.,attenuation=1.,finite_RTE_disk=0.))
    return dict(arm_id=entry[0],formula=entry[1],substeps=entry[2],rounds=rows,
                status='P1_ALL18_HP_CONDITIONAL_SIGNAL_PASS')


def hp_bundle():
    original=bundle();original['SOURCE_SHA256']={};original['input_identity']['file_SHA256']={'toy':'public metadata'}
    meta={'blob':'synthetic_original'}
    record=dict(schema_version='b2_strong_pf_fixed_five_arm_HP_requalification_v1',
        status='FIXED_FIVE_ARM_HP_REQUALIFICATION_COMPLETE',execution_SOURCE=m.SIGNAL_SOURCE,
        precision_SOURCE=m.PRECISION_SOURCE,precision_dps=[80,120],evaluated_points=90,
        historical_results_overwritten=False,scientific_context=identity(),P1_conditions=m.old.expected_conditions(),
        original_result=meta,historical_qualified_arms=original['qualified_arms'],
        input_gate={'status':'PASS'},input_identity=copy.deepcopy(original['input_identity']),
        arm_set=[dict(formula=f,substeps=s) for _,f,s in m.ARMS],new_qualified_arms=[a[0] for a in m.ARMS],
        reused_SOURCE_SHA256={},arms=[])
    return record,original,meta


def test_shared_fixed_inputs_and_scope():
    record,original,meta=hp_bundle()
    ident,conditions=m.validate_shared(record,original,meta)
    assert ident==identity() and conditions==m.old.expected_conditions()


@pytest.mark.parametrize('field',['execution_SOURCE','precision_SOURCE','precision_dps','P1_conditions',
                                  'scientific_context','input_identity','new_qualified_arms','original_result'])
def test_shared_mismatch_stops(field):
    record,original,meta=hp_bundle()
    record[field]='wrong'
    with pytest.raises((ValueError,TypeError)):m.validate_shared(record,original,meta)


@pytest.mark.parametrize('mutation',['HP_test','precision','conversion','independent','phase','radius','nonfinite'])
def test_HP_conversion_and_saved_qualification_rejections(mutation):
    row=hp_arm()['rounds'][0]
    if mutation=='HP_test':row['qualification']['HP_tests']['precision']=False
    if mutation=='precision':row['precision_check']['N']['operator_max_abs_80_vs_120']='1.01e-9'
    if mutation=='conversion':row['qualification']['conversion_error']['PF']='1.01e-9'
    if mutation=='independent':row['qualification']['HP_independent_difference']['phase_rad']='1.01e-9'
    if mutation=='phase':row['signals']['PF']['phase_rad']='.9'
    if mutation=='radius':row['qualification']['HP_bounds']['PF_radius_lower']='-.01'
    if mutation=='nonfinite':row['signals']['PF']['real']='nan'
    with pytest.raises((ValueError,TypeError)):m.scalar_row(m.context(),row)


@pytest.mark.parametrize('phase',['.12','-.13'])
def test_existing_decoder_conversion_fixed_branch_lift_and_actual_shots(phase):
    arm=hp_arm(phase=phase)
    item=m.decode_one(arm)
    assert item['status']=='DECODED_AWAITING_POSTHOC_REFERENCE'
    choices=[r['selected_branch'] for r in item['reconstruction']['rounds']]
    m.complete_one(item,-float(phase)/.02,context_hash='toy_context')
    assert item['status']==m.old.PASS
    assert choices==[r['selected_branch'] for r in item['reconstruction']['rounds']]
    for row in item['rounds']:
        audit=row['HP_conversion']
        assert audit['branch_conversion_status']=='PASS'
        assert audit['shot_integer_conversion']['status']=='SAME_EXISTING_API_CEIL'
        assert row['complex_PF_signal_HP']==audit['HP_signal']
        assert row['signal_execution_SOURCE']==m.SIGNAL_SOURCE
        assert row['signal_result_commit']==m.SIGNAL_COMMIT
        assert row['price'] is None and row['total_circuit_cost'] is None
        assert row['RTE_budget_reserved_unreallocated']==.04
    assert item['totals']['N_all_rounds']==sum(r['N_round'] for r in item['rounds'])
    assert item['correctness']['HP_scalar_energy_target_decision_match']


def test_arm_failure_and_energy_oracle_do_not_repair_branches():
    arm=hp_arm();arm['rounds'].pop()
    assert m.decode_one(arm)['status']=='STOP_HP_SIGNAL_OR_CONVERSION'
    item=m.decode_one(hp_arm());before=[r['selected_branch'] for r in item['reconstruction']['rounds']]
    m.complete_one(item,40.,context_hash='toy')
    assert item['status']=='STOP_HP_ENERGY_SHOTS_OR_CONVERSION'
    assert before==[r['selected_branch'] for r in item['reconstruction']['rounds']]


def test_formula_mismatch_is_arm_local_stop():
    arm=hp_arm();arm['substeps']=8
    assert m.decode_one(arm)['status']=='STOP_HP_SIGNAL_OR_CONVERSION'


def test_ceil_conversion_instability_is_rejected_using_existing_API(monkeypatch):
    item=m.decode_one(hp_arm());m.complete_one(item,-6.,context_hash='toy')
    row=item['rounds'][0];audit=row['HP_conversion'];actual=m.old.shots.account_radius_round
    def unstable(**kwargs):
        result=actual(**kwargs)
        result['shots']['cosine']+=1
        return result
    monkeypatch.setattr(m.old.shots,'account_radius_round',unstable)
    with pytest.raises(ValueError,match='SHOT_INTEGER_UNSTABLE'):
        m.shot_conversion_check(m.context(),row,audit,context_hash='toy',arm_hash='toy')


def test_serialized_HP_details_and_integer_totals(tmp_path):
    item=m.decode_one(hp_arm());m.complete_one(item,-6.,context_hash='toy')
    item.update(execution_SOURCE='toy',signal_execution_SOURCE=m.SIGNAL_SOURCE)
    summary=m.old.write_arm(tmp_path/'arm',item)
    value=json.loads((tmp_path/'arm/result.json').read_text())
    assert summary['serialization_check']['JSON_CSV_integer_match']
    assert len(value['rounds'])==18
    assert isinstance(value['rounds'][0]['complex_PF_signal_HP']['real'],str)
    assert all(type(r['N_X']) is int for r in value['rounds'])


def test_old_result_comparison_never_calls_decoder_or_shots(monkeypatch):
    original=hp_bundle()[1];conditions=m.old.expected_conditions()
    old_item=m.old.decode_arms([original['arms'][0]])[0]
    m.old.complete_arms([old_item],lambda:({'validation':{'energy_Ha':-6.}},{}),context_hash='toy')
    old_item.update(execution_SOURCE=m.OLD_SOURCE,signal_result_commit=m.old.SIGNAL_COMMIT,
                    signal_execution_SOURCE=m.old.SIGNAL_SOURCE)
    summaries=[];data={}
    for entry in m.old.ARMS:
        copied=copy.deepcopy(old_item);copied.update(arm_id=entry[0],formula=entry[1],substeps=entry[2])
        item_summary={k:v for k,v in copied.items() if k not in ('selected','rounds','reconstruction')}
        item_summary['result_path']=entry[0]+'/result.json';summaries.append(item_summary)
        data[m.OLD_ROOT+'/'+item_summary['result_path']]=copied
    summary=dict(status='THREE_ARMS_CONDITIONAL_BRANCH_SHOTS_PASS',execution_SOURCE=m.OLD_SOURCE,
        scientific_context=identity(),fixed_conditions=conditions,SOURCE_SHA256={},arms=summaries,saved_inputs=[{},{}])
    data[m.OLD_ROOT+'/result.json']=summary
    def fixed(commit,path):
        assert commit==m.OLD_COMMIT
        return data[path],{'commit':commit,'path':path}
    monkeypatch.setattr(m.old.branch,'fixed_json',fixed)
    def forbidden(*a,**k):raise AssertionError('Old arm must not be recomputed')
    monkeypatch.setattr(m.old.branch,'decode_raw_means',forbidden)
    monkeypatch.setattr(m.old.shots,'account_radius_round',forbidden)
    rows,inputs,refs=m.load_old_comparison(identity(),conditions)
    assert len(rows)==3 and all(r['reused_without_new_calculation'] for r in rows)
    assert all(r['signal_SOURCE']==m.old.SIGNAL_SOURCE for r in rows)
    assert len(inputs)==4


def fake_execution(tmp_path,monkeypatch,*,stop_first=False):
    record,original,om=hp_bundle();record['SOURCE_SHA256']={};record['saved_references']=[{'branch':'B2'},{'shots':'B2'}]
    record['arms']=[hp_arm(entry) for entry in m.ARMS]
    if stop_first:record['arms'][0]['rounds'][0]['qualification']['HP_tests']['precision']=False
    source='synthetic_HP_branch_source';events=[]
    class Executor:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,fn,values):return map(fn,values)
    monkeypatch.setattr(m,'ProcessPoolExecutor',Executor)
    real=m.decode_one
    def decode(arm):
        out=real(arm);events.append(('decoder_complete',arm['arm_id']));return out
    monkeypatch.setattr(m,'decode_one',decode)
    fake_root=tmp_path/'SOURCE';fake_root.mkdir()
    paths=['src/trotterlib/b2_p1_hp_branch_shots.py','scripts/run_b2_p1_hp_branch_shots.py',
        'src/trotterlib/b2_p1_strong_pf_branch_shots.py','src/trotterlib/b2_p1_branch_validation.py',
        'src/trotterlib/b2_p1_shot_accounting.py','src/trotterlib/rpe_four_round_phase_validation.py',
        'src/trotterlib/o1_m1_resource_bridge.py','src/trotterlib/finite_rte_signal_validation.py',
        'src/trotterlib/b2_p1_strong_pf_h4_diagnosis.py']
    for name in paths:
        p=fake_root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'artificial source')
    monkeypatch.setattr(m,'ROOT',fake_root)
    def git(*args):
        if args==('rev-parse','HEAD'):return source.encode()
        if args[0]=='show' and args[1].startswith(source+':'):return b'artificial source'
        raise AssertionError(args)
    monkeypatch.setattr(m.old.branch,'git',git)
    def fixed(commit,path):
        if (commit,path)==(m.SIGNAL_COMMIT,m.SIGNAL_PATH):return record,{'synthetic':'HP saved input'}
        if (commit,path)==(m.old.SIGNAL_COMMIT,m.old.SIGNAL_PATH):return original,om
        if (commit,path)==(m.old.branch.STATE_COMMIT,m.old.branch.STATE_RESULT):
            assert len(events)==5
            events.append(('posthoc_reference',None));return {'validation':{'energy_Ha':-6.}}, {'synthetic':'reference'}
        raise AssertionError((commit,path))
    monkeypatch.setattr(m.old.branch,'fixed_json',fixed)
    def old_table(identity,conditions):
        assert len(events)==5
        rows=[]
        for name,formula,s in m.old.ARMS:
            data=dict(status=m.old.PASS,arm_id=name,formula=formula,substeps=s,
                rounds=[{'energy_estimate_Ha':-6.}],correctness={'final_energy_absolute_error_Ha':0.},
                totals=dict(N_X_total=777,N_Y_total=777,N_all_rounds=1554))
            rows.append(m.comparison_row(data,{'commit':m.OLD_COMMIT},reused=True))
        return rows,[{'synthetic':'old summary'}],record['saved_references'][-2:]
    monkeypatch.setattr(m,'load_old_comparison',old_table)
    return events


def test_full_artificial_five_arm_execution_eight_row_comparison(tmp_path,monkeypatch):
    events=fake_execution(tmp_path,monkeypatch)
    result=m.run(tmp_path/'output')
    assert result['status']=='FIVE_HP_ARMS_CONDITIONAL_BRANCH_SHOTS_PASS'
    assert len(events)==6 and events[-1][0]=='posthoc_reference'
    assert len(result['comparison'])==8
    old_rows=[r for r in result['comparison'] if r['reused_without_new_calculation']]
    new_rows=[r for r in result['comparison'] if not r['reused_without_new_calculation']]
    assert len(old_rows)==3 and len(new_rows)==5
    assert all(r['signal_SOURCE']==m.old.SIGNAL_SOURCE for r in old_rows)
    assert all(r['signal_SOURCE']==m.SIGNAL_SOURCE and r['N_all_rounds']!=1554 for r in new_rows)
    assert result['forbidden_work']['old_three_arm_recalculation']==0


def test_local_conversion_stop_preserves_numbers_and_other_arms_continue(tmp_path,monkeypatch):
    fake_execution(tmp_path,monkeypatch,stop_first=True)
    result=m.run(tmp_path/'output')
    assert result['status']=='FIVE_HP_ARMS_COMPLETE_WITH_STOPS'
    assert result['arms'][0]['status']=='STOP_HP_SIGNAL_OR_CONVERSION'
    assert result['arms'][0]['failure_saved_round_evidence']['m']==0
    assert all(r['status']==m.old.PASS for r in result['arms'][1:])
    assert result['comparison'][2]['arm_id']=='Full_S2_s4'
    assert result['comparison'][2]['N_all_rounds'] is None
