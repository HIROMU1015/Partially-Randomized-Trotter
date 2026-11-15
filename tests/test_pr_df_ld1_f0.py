"""Synthetic reader-route tests; never read or diagnose saved H4 probabilities."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_pr_df_ld1_f0 as F

@pytest.fixture
def synthetic():
    prep=dict(hash='SYNTHETIC_ONLY',component_count=2,components=[
        dict(id='a',probability=.25,probability_hex=float(.25).hex(),basis_id='toy',support=[0]),
        dict(id='b',probability=.75,probability_hex=float(.75).hex(),basis_id='toy',support=[1])])
    law=dict(m=0,candidate_id='TOY',q_PF=2,R_outer=4,event_count=8,tau=.125,tau_hex=float(.125).hex(),
             orders=[0,2],order_weights=[3.,1.],order_probabilities=[.75,.25],event_normalization=4.)
    row=dict(m=0,physical_id='TOY',q=2,R_outer=4,saved_event_positions=8)
    tables=[dict(id='components',kind='component',source={'toy':'prep'},signature=F.component_signature(prep)),
            dict(id='order_m0',kind='order',source={'toy':'law'},row=row,signature=F.law_signature(law,row))]
    values=dict(prep=prep,law=law)
    return values,dict(tables=tables)


def test_actual_entry_uses_pinned_helper_and_saved_cdf_and_logs_all_stages(synthetic,tmp_path):
    values,bundle=synthetic;helper,cdf=F.load_fixed_functions();reads=[]
    def reader(ref):reads.append(ref['toy']);return values[ref['toy']]
    result=F.run_tables(bundle,tmp_path/'run',helper,cdf,reader)
    assert reads==['prep','law'] and result['status']=='F0_DIAGNOSTIC_COMPLETE'
    assert result['counts']['new_attempted']==result['counts']['new_complete']==2
    bounds=json.loads((tmp_path/'run/ROUND_BOUNDS.json').read_text())['rows'][0]
    assert bounds['event_TV_upper']==bounds['path_TV_upper']==dict(numerator=0,denominator=1)
    assert bounds['actual_PCG64_joint_distance'] is None and bounds['price_shift'] is None
    axes=json.loads((tmp_path/'run/AXIS_MAP.json').read_text())['rows'];assert [r['axis'] for r in axes]==['X','Y']
    assert axes[0]['round_selector']==axes[1]['round_selector']==['rows',0]
    progress=[json.loads(s) for s in (tmp_path/'run/PROGRESS.jsonl').read_text().splitlines()]
    assert [(v['stage'],v['state']) for v in progress]==[('components','START'),('components','END'),('order_m0','START'),('order_m0','END')]
    assert all(v['seconds']>=0 for v in progress if v['state']=='END')
    assert result['E_C'] is None and result['G'] is None and result['counts']['compiles']==0


@pytest.mark.parametrize('change',['order','hex','missing','source'])
def test_component_order_hex_missing_and_source_mismatch_rejected_before_audit(synthetic,change):
    values,bundle=synthetic;t=deepcopy(bundle['tables'][0]);prep=deepcopy(values['prep'])
    if change=='order':prep['components'].reverse()
    elif change=='hex':prep['components'][0]['probability_hex']='0x0.0p+0'
    elif change=='missing':prep['components'].pop()
    else:prep['hash']='DIFFERENT_PREPARATION_SOURCE'
    calls=[];helper=SimpleNamespace(categorical_audit=lambda *a,**k:calls.append(True))
    with pytest.raises(ValueError):F.audit_one(t,helper,None,lambda ref:prep)
    assert calls==[]


@pytest.mark.parametrize('change',['tau_hex','physical','normalization','probabilities','orders','events'])
def test_law_identity_and_normalization_fail_before_audit(synthetic,change):
    values,bundle=synthetic;t=deepcopy(bundle['tables'][1]);law=deepcopy(values['law'])
    if change=='tau_hex':law['tau_hex']='0x0.0p+0'
    elif change=='physical':law['candidate_id']='WRONG'
    elif change=='normalization':law['event_normalization']=3.
    elif change=='probabilities':law['order_probabilities']=[.5,.5]
    elif change=='orders':law['orders']=[2,0]
    else:law['event_count']=7
    calls=[];helper=SimpleNamespace(categorical_audit=lambda *a,**k:calls.append(True))
    with pytest.raises(ValueError):F.audit_one(t,helper,None,lambda ref:law)
    assert calls==[]


def test_support_loss_is_labeled_by_stage_and_component(synthetic):
    values,bundle=synthetic;t=bundle['tables'][0];prep=values['prep']
    for row,p in zip(prep['components'],[1.,1e-25]):row.update(probability=p,probability_hex=p.hex())
    t['signature']=F.component_signature(prep);helper,cdf=F.load_fixed_functions()
    v=F.audit_one(t,helper,cdf,lambda ref:prep)
    loss=next(r for r in v['support_changes'] if r['stage']=='Q->F')
    assert loss['change']=='POSITIVE_TO_ZERO' and loss['label']['component_id']=='b' and loss['after']==0
    assert v['P'][1]>0 and v['F'][1]==v['G'][1]==0
    assert v['physical_law_distance'] is None and v['qualified_for_compiled_price'] is False


def test_failure_preserves_first_table_and_stops_without_retry(synthetic,tmp_path):
    values,bundle=synthetic;bundle['tables'].append(deepcopy(bundle['tables'][1]));bundle['tables'][-1]['id']='must_not_run'
    values['law']['tau_hex']='0x0.0p+0';helper,cdf=F.load_fixed_functions();reads=[]
    def reader(ref):reads.append(ref['toy']);return values[ref['toy']]
    v=F.run_tables(bundle,tmp_path/'run',helper,cdf,reader)
    assert reads==['prep','law'] and v['status']=='STOP_PARTIAL'
    assert (tmp_path/'run/components.json').exists() and (tmp_path/'run/STOP.json').exists()
    assert v['counts']['new_attempted']==2 and v['counts']['new_complete']==v['counts']['anomalies']==v['counts']['withheld']==1
    assert not (tmp_path/'run/must_not_run.json').exists()
    with pytest.raises(FileExistsError):F.run_tables(bundle,tmp_path/'run',helper,cdf,reader)


def test_nonmonotonic_cdf_or_mass_is_rejected(synthetic):
    values,bundle=synthetic;helper,cdf=F.load_fixed_functions();original=helper.categorical_audit
    def bad(*a,**k):
        v=original(*a,**k);v['cdf']=(Fraction(3,4),Fraction(1,2));return v
    helper.categorical_audit=bad
    with pytest.raises(ValueError,match='CDF_NONMONOTONIC_OR_MASS'):
        F.audit_one(bundle['tables'][0],helper,cdf,lambda ref:values['prep'])
