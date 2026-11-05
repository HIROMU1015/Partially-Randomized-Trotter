"""Invented metadata and mocked fixed numerical APIs; no scientific inputs."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_integrated_downstream_execution as m


class FakeCalls:
    def __init__(self,out,limits,maximum):
        self.counts=Counter();self.limits=limits;self.maximum=maximum;self.first_failure=None
    def __call__(self,kind,label,function):
        self.counts[kind]+=1
        assert self.counts[kind]<=self.limits[kind]
        assert sum(self.counts.values())<=self.maximum
        try:return function()
        except BaseException as exc:
            self.first_failure={'kind':kind,'error':str(exc)}
            raise


def emit(call,limits):
    for kind,count in limits.items():
        for i in range(count):call(kind,str(i),lambda:None)


def fake_backend(monkeypatch,*,ineligible=()):
    def save(path,value):
        with Path(path).open('x') as stream:json.dump(value,stream)
    io=NS(save=save)
    ref_calls=[]
    native_calls=[]
    def prepare(candidate_path,reference_path,physical,common,out,call):
        Path(out).mkdir(parents=True)
        emit(call,m.SETUP)
        return dict(arm=NS(model=NS(source_identity=physical['candidate_sha256']))),{'status':'PASS'}
    def reference(a,r,out,call,pf_cache=None):
        ref_calls.append((r,pf_cache))
        Path(out).mkdir(parents=True)
        emit(call,m.REFERENCE32 if pf_cache is None else m.REFERENCE_EXTRA)
        result=dict(candidate=a['arm'].model.source_identity,r=r,native_admitted=r not in ineligible,
            lanes={k:dict(energy={'same_accuracy':k!='evidence'}) for k in ('evidence','ideal_exact_native','assumed_raw_signal_budget')})
        return dict(a,config=NS(rte_steps=r)),result,{'target_id':'invented'},pf_cache or {'artificial':'PF'}
    def native(root,runtime,out,call,a,result,packet,*,split,cohort_key,prepared):
        Path(out).mkdir(parents=True)
        native_calls.append((split,cohort_key,a['config'].rte_steps))
        if not prepared:
            emit(call,m.NATIVE_COMMON);prepared.update(arm=a['arm'])
        for _ in range(1 if split=='selection' else 2):emit(call,m.NATIVE32)
        if split=='final':emit(call,{'final_calibration':1})
        emit(call,m.ACCOUNTING)
        result=deepcopy(result);packet=dict(packet)
        for key,value in result['lanes'].items():
            value['conditional_totals']={'totals':None if key=='evidence' else {'rz_count':100.+result['r']}}
        result['calibration']={'artificial':True}
        packet['kernel_samples']=[{'rz_count':100} for _ in range(32 if split=='selection' else 64)]
        return result,packet
    def strong(root,runtime,out,call,a):
        Path(out).mkdir(parents=True)
        emit(call,{'D_unit':141})
        return {'status':'D_COMPLETE','candidate':a['arm'].model.source_identity}
    def validate(rows,n):assert len(rows)==n
    bindings={k:None if k=='evidence' else {'W':1,'shell':{'rz_count':0}}
              for k in ('evidence','ideal_exact_native','assumed_raw_signal_budget')}
    def summary(rows,bindings,D,*,calibration,call,expected_n,statistics):
        assert expected_n==64 and statistics['confidence']==.95
        for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
            call('price_statistics',lane,lambda:None)
        return {'artificial_summary':True}
    stat=NS(POLICY={},shared=NS(validate_rows=validate),affine=Mock(return_value=bindings),
        arm_summary=summary,primary=Mock(return_value={'decision':'ARTIFICIAL_ONLY'}))
    adapter=NS(prepare_candidate=prepare,reference_r=reference,native_cohort=native,strong_d=strong)
    monkeypatch.setattr(m,'_backend',lambda:(adapter,stat,io,FakeCalls,{'D_unit':141}))
    return NS(adapter=adapter,statistics=stat,io=io,refs=ref_calls,native=native_calls)


def task(arm='PR'):
    return dict(geometry='H4_100',arm=arm,start='C0',strength=.1,budget=64)


def selection(tmp_path,backend,arm='PR'):
    return m.candidate_selection(tmp_path,{},'invented candidate','invented reference',
        {'candidate_sha256':arm+'A','budget_PASS':True},{'input_identity':'H'},task(arm),tmp_path/(arm+'selection'))


def test_static_registered_call_limits():
    assert sum(m.SETUP.values())==14
    assert sum(m.REFERENCE32.values())==93
    assert sum(m.REFERENCE_EXTRA.values())==80
    assert sum(m.NATIVE_COMMON.values())==35
    assert sum(m.NATIVE32.values())==289
    assert sum(m.ACCOUNTING.values())==33
    assert sum(m.SELECTION_LIMITS.values())==1268
    assert sum(m.FINAL_LIMITS.values())==618


def test_candidate_selection_all_r_and_exact_call_counts(tmp_path,monkeypatch):
    b=fake_backend(monkeypatch)
    report,state=selection(tmp_path,b)
    calls=json.loads((tmp_path/'PRselection'/'call_summary.json').read_text())
    assert calls['scientific_units']==1268
    assert [r for r,pf in b.refs]==[32,16,64]
    assert b.refs[0][1] is None and b.refs[1][1] is b.refs[2][1]
    assert [x['r'] for x in report['rows']]==[16,32,64]
    assert all(x['eligible'] and x['samples']==32 and x['cohort_role']=='selection' for x in report['rows'])
    assert state['native_prepared'] and len(state['policies'])==3


def test_scientific_ineligible_policy_stays_null_and_other_r_complete(tmp_path,monkeypatch):
    b=fake_backend(monkeypatch,ineligible=(32,))
    report,state=selection(tmp_path,b)
    rows={x['r']:x for x in report['rows']}
    assert rows[32]['eligible'] is False and rows[32]['total_RZ'] is None
    assert rows[16]['eligible'] and rows[64]['eligible']
    assert [x[2] for x in b.native]==[16,64]


def test_final64_Ewrapper_same_JU_D_and_no_repeated_work(tmp_path,monkeypatch):
    b=fake_backend(monkeypatch)
    report,state=selection(tmp_path,b)
    m.candidate_d(tmp_path,{},state,tmp_path/'D')
    final,opaque=m.final_pair(tmp_path,{},state,16,tmp_path/'final')
    d=json.loads((tmp_path/'D'/'call_summary.json').read_text())
    f=json.loads((tmp_path/'final'/'call_summary.json').read_text())
    assert d['scientific_units']==141 and f['scientific_units']==618
    assert final['samples']==64 and len(opaque['rows'])==64
    assert 'car_context' not in f['call_counts'] and 'candidate_ground' not in f['call_counts']
    assert final['candidate_sha256']==state['D']['candidate']
    with pytest.raises(ValueError,match='D_ALREADY'):m.candidate_d(tmp_path,{},state,tmp_path/'D2')
    with pytest.raises(ValueError,match='FINAL_PAIR_ALREADY'):m.final_pair(tmp_path,{},state,16,tmp_path/'again')


def test_missing_retained_native_context_fails_before_final(tmp_path,monkeypatch):
    b=fake_backend(monkeypatch)
    report,state=selection(tmp_path,b)
    state['native_prepared'].clear()
    with pytest.raises(ValueError,match='RETAIN_SELECTION'):m.final_pair(tmp_path,{},state,32,tmp_path/'badfinal')
    assert not (tmp_path/'badfinal').exists()


def test_DF_unqualified_returns_complete_null_grid_without_new_science(tmp_path,monkeypatch):
    b=fake_backend(monkeypatch)
    report,state=m.candidate_selection(tmp_path,{},'closed','closed',
        {'candidate_sha256':'badDF','budget_PASS':False},{'input_identity':'H'},task(),tmp_path/'badDF')
    assert all(x['total_RZ'] is None for x in report['rows'])
    assert state['A'] is None and not b.refs and not b.native
    assert json.loads((tmp_path/'badDF'/'call_summary.json').read_text())['scientific_units']==0


def test_correctness_failure_preserves_call_summary_and_does_not_continue(tmp_path,monkeypatch):
    b=fake_backend(monkeypatch)
    original=b.adapter.reference_r
    def failing(a,r,out,call,**kwargs):
        original(a,r,out,call,**kwargs)
        return call('law_gate','invented failure',lambda:(_ for _ in ()).throw(ValueError('bad physics binding')))
    b.adapter.reference_r=failing
    with pytest.raises(ValueError):selection(tmp_path,b)
    assert len(b.refs)==1 and not b.native
    report=json.loads((tmp_path/'PRselection'/'call_summary.json').read_text())
    assert report['status']=='CORRECTNESS_FAILURE'


def test_primary_final_comparison_9875_and_descriptive95(tmp_path,monkeypatch):
    b=fake_backend(monkeypatch)
    finals={}
    for arm in ('PR','ENTRY'):
        _,state=selection(tmp_path,b,arm)
        m.candidate_d(tmp_path,{},state,tmp_path/(arm+'D'))
        _,finals[arm]=m.final_pair(tmp_path,{},state,32,tmp_path/(arm+'final'))
    call=lambda kind,label,fn:fn()
    result=m.compare_final(finals['PR'],finals['ENTRY'],tmp_path/'primary.json',call,primary=True)
    assert result['confidence']==.9875
    assert b.statistics.primary.call_args.kwargs['expected_n']==64
    m.compare_final(finals['PR'],finals['ENTRY'],tmp_path/'description.json',call,primary=False)
    assert b.statistics.primary.call_args.kwargs['statistics']['confidence']==.95


def test_missing_final_does_not_produce_winner(tmp_path,monkeypatch):
    b=fake_backend(monkeypatch)
    result=m.compare_final(None,None,tmp_path/'missing.json',lambda k,l,f:f(),primary=True)
    assert result['comparison'] is None and not b.statistics.primary.called


def test_runtime_only_requires_source_profile_and_explicit_old_shell_paths():
    paths={x:{k:f'artifacts/shells/{x}_{k}.qpy' for k in ('raw','compiled')} for x in ('head','X','Y')}
    runtime=m.runtime_contract({'compiler':{'optimization_level':1},'target':{},'allowances':{}},'new execution source',paths)
    assert runtime['shells']==paths and runtime['source_manifest_sha256']=='new execution source'
    assert 'scope' not in runtime and 'grant' not in runtime


def test_existing_real_call_journal_with_mock_science_reaches_registered_counts(tmp_path,monkeypatch):
    from trottertracks.c1_stage3_integrated import Calls
    b=fake_backend(monkeypatch)
    backend=m._backend()
    monkeypatch.setattr(m,'_backend',lambda:(*backend[:3],Calls,backend[4]))
    _,state=selection(tmp_path,b)
    m.candidate_d(tmp_path,{},state,tmp_path/'D')
    m.final_pair(tmp_path,{},state,64,tmp_path/'final')
    for name,expected in (('PRselection',1268),('D',141),('final',618)):
        events=[json.loads(x) for x in (tmp_path/name/'science_calls.jsonl').read_text().splitlines()]
        assert sum(x['event']=='attempt' for x in events)==expected
        assert sum(x['event']=='COMPLETE' for x in events)==expected
        assert not any(x['event']=='FAIL' for x in events)
