"""Artificial references and mocked existing numerical engines only."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_integrated_retuning_adapter as m
from trottertracks import c_near_r32_statistics as stats


def invented(monkeypatch):
    counts=Counter()
    def call(k,label,fn):counts[k]+=1;return fn()
    b=NS(configure=Mock(side_effect=lambda arm,h,r,K:('tail',NS(rte_steps=r),'dist')),
         tail_gate=Mock(return_value={'status':'PASS'}),kernels=Mock(return_value={'pf':'invented PF'}),
         reference=NS(reference_round=Mock(side_effect=lambda *args,**kwargs:{'pf_operator_error':0.,'raw_signal':1+0j})))
    acc=NS(aggregate_totals=Mock(return_value={'totals':{'rz_count':100.}}))
    m4=NS(corrected_kernel=Mock(return_value={'corrected':'invented RTE'}),
          rte_round=Mock(side_effect=lambda kernels,m,pf_error,state:{'pf_operator_error':pf_error,'raw_signal_parts':[1.,0.]}),
          integrated=NS(resource_plan=Mock(return_value={'workers':2,'BLAS_threads_per_worker':1})),accuracy=acc)
    def lanes(packet,target,call):
        for lane in ('evidence','ideal_exact_native','assumed_raw_signal_budget'):
            call('signal_qualification',lane,lambda:None)
            for i in range(13):call('round_accuracy',str(i),lambda:None)
            call('energy_assessment',lane,lambda:None);call('shots_summary',lane,lambda:None)
        return {lane:dict(energy={'same_accuracy':lane!='evidence'},rounds=list(range(13)))
                for lane in ('evidence','ideal_exact_native','assumed_raw_signal_budget')}
    native=Mock(side_effect=lambda *args,**kw:([{'rz_count':1} for _ in range(32)],{'samples':32}))
    perf=NS(b=b,native=native,b2=NS(calibration=Mock(side_effect=lambda rows:{'samples':len(rows)})),
            accounting=NS(cost_calibration=Mock(return_value={}),COST_SCOPE='existing'))
    ret=NS(perf=perf,m4=m4,m3=NS(serial_cost=Mock(return_value={}),calibration_uncertainty=Mock(return_value={})),
           io=NS(save=Mock()),normalization=Mock(return_value={'status':'PASS'}),assess_lanes=lanes)
    feasibility=NS(classify_round=Mock(return_value={'status':'POTENTIALLY_FEASIBLE'}),
                   summarize=Mock(return_value={'status':'POTENTIALLY_FEASIBLE'}))
    distribution=Mock()
    monkeypatch.setattr(m,'_engines',lambda:(ret,feasibility,distribution))
    a=dict(arm=NS(model=NS(source_identity='A',input_identity='H'),plan=NS(extracted_identity=0)),ctx='ctx',
           sector={'H':np.zeros((2,2)),'generators':'artificial'},ground=np.array([1.,0.]),target='existing target',
           saved={'physical':{'sector_error':0.,'numerical_guard':0.},'shells':{}},config=NS(rte_steps=32))
    return a,ret,call,counts


def test_all_fixed_r_use_existing_reference_and_share_only_same_candidate_pf(tmp_path,monkeypatch):
    a,ret,call,c= invented(monkeypatch)
    a32,result,packet,cache=m.reference_r(a,32,tmp_path/'r32',call)
    assert c['exact_evolution']==13 and c['round_reference']==13 and c['round_accuracy']==39
    for r in (16,64):
        ar,result,packet,current=m.reference_r(a,r,tmp_path/f'r{r}',call,pf_cache=cache)
        assert ar['config'].rte_steps==r and result['r']==r and current is cache
        assert packet['target_id']==f'integrated:A:r{r}'
    assert c['exact_evolution']==13 and ret.perf.b.kernels.call_count==1
    assert ret.m4.corrected_kernel.call_count==2 and ret.m4.rte_round.call_count==26
    assert c['round_reference']==39 and c['round_accuracy']==117
    assert result['actual_CA_achieved'] is None


def setup_candidate(monkeypatch):
    a,ret,call,counts=invented(monkeypatch)
    d={'J':np.zeros((8,4,4)),'U':np.tile(np.eye(4),(8,1,1))}
    sector=dict(d,generators=np.zeros((9,36,36)),H=np.zeros((36,36)))
    ret.perf.loader=NS(load=Mock(side_effect=[d,sector]),ground=Mock(return_value=(np.zeros(36),np.eye(36))),
                       ground_gate=Mock(return_value={'status':'PASS'}))
    ret.io.sha=Mock(return_value='A')
    ret.perf.b.prepare_arm=Mock(return_value=a['arm'])
    ret.perf.b.component_check=Mock(return_value={'status':'PASS'})
    ret.perf.b.candidate_factor=Mock(return_value=np.zeros((36,36)))
    binding=Mock()
    monkeypatch.setattr(m,'_binding_engine',lambda:(binding,lambda *args:np.arange(36)))
    common=dict(p={'h':np.zeros((4,4)),'g':np.zeros((4,4,4,4)),'constant':0},input_identity='H',
                ctx='independent',basis=np.arange(36),target='existing target',shells={'mock':'shell'})
    df=dict(candidate_sha256='A',budget_PASS=True,actual_rank=8,kappa='original_g_corrected',scalar='nuclear_once',
            budget_Ha=.00019875,sector_error=0.,numerical_guard=0.,input_sha256='H',
            guard_policy='sector_operation_scale_guard8_proposal_v1')
    return ret,call,counts,common,df


def test_new_endpoint_prepare_binding_ground_and_fourteen_calls(tmp_path,monkeypatch):
    ret,call,counts,common,df=setup_candidate(monkeypatch)
    packet,status=m.prepare_candidate('candidate.npz','independent.npz',df,common,tmp_path/'A',call)
    assert sum(counts.values())==14 and counts['general_factor_gate']==9
    assert packet['arm'].model.source_identity=='A' and packet['arm'].model.input_identity=='H'
    assert packet['saved']['shells']==common['shells'] and status['status']=='PASS'


def test_wrong_candidate_reference_and_df_stop_before_science_admission(tmp_path,monkeypatch):
    for mismatch in ('bytes','DF','input','guard'):
        ret,call,counts,common,df=setup_candidate(monkeypatch)
        if mismatch=='bytes':ret.io.sha.return_value='other'
        elif mismatch=='DF':df['budget_PASS']=False
        elif mismatch=='input':df['input_sha256']='wrong original H'
        else:df['guard_policy']='wrong physical guard'
        with pytest.raises(ValueError):m.prepare_candidate('candidate.npz','reference.npz',df,common,tmp_path/mismatch,call)
        assert not ret.perf.loader.ground.called and not ret.perf.b.prepare_arm.called


def test_alias_unqualified_retained_without_downstream_packet(tmp_path,monkeypatch):
    ret,call,counts,common,df=setup_candidate(monkeypatch)
    ret.perf.loader.ground_gate.side_effect=ValueError('UNRESOLVED_GROUND_ALIAS')
    packet,status=m.prepare_candidate('candidate.npz','reference.npz',df,common,tmp_path/'alias',call)
    assert packet is None and status['status']=='UNRESOLVED_GROUND_ALIAS'


def test_wrong_or_missing_pf_cache_fails_before_work(tmp_path,monkeypatch):
    a,ret,call,c=invented(monkeypatch)
    for cache in (None,{'candidate_sha256':'other','input_identity':'H'}, {'candidate_sha256':'A','input_identity':'otherH'}):
        with pytest.raises(ValueError):m.reference_r(a,16,tmp_path/'bad',call,pf_cache=cache)
    assert not ret.perf.b.configure.called


def test_selection_final_native_batches_independent_and_selection_unchanged(tmp_path,monkeypatch):
    a,ret,call,c=invented(monkeypatch)
    ar,result,packet,cache=m.reference_r(a,32,tmp_path/'ref',call)
    selection,_=m.native_cohort(tmp_path,{'source_manifest_sha256':'S'},tmp_path/'select',call,ar,result,packet,
                               split='selection',cohort_key='H1/start0/strength1/budget16')
    final,fp=m.native_cohort(tmp_path,{'source_manifest_sha256':'S'},tmp_path/'final',call,ar,result,packet,
                            split='final',cohort_key='H1/start0/strength1/budget16')
    policies=[x.kwargs['sampling'] for x in ret.perf.native.call_args_list]
    assert len(policies)==3 and len({x['cohort'] for x in policies})==3
    assert selection['samples']==32 and final['samples']==64 and len(fp['kernel_samples'])==64
    assert 'calibration' not in result and 'kernel_samples' not in packet
    assert c['serial_cost']==52 and c['final_calibration']==1


def test_no_native_for_unqualified_or_mixed_candidate_and_no_overwrite(tmp_path,monkeypatch):
    a,ret,call,c=invented(monkeypatch)
    kwargs=dict(split='selection',cohort_key='mock')
    for result in ({'native_admitted':False}, {'native_admitted':True,'candidate':'other','r':32}):
        with pytest.raises(ValueError):m.native_cohort(tmp_path,{},tmp_path/'bad',call,a,result,{},**kwargs)
    assert not ret.perf.native.called


def test_same_ju_d_dispatch_and_fresh_output(tmp_path,monkeypatch):
    a,ret,call,c=invented(monkeypatch)
    evaluator=Mock(return_value={'candidate':'A','actual_CA_achieved':None})
    monkeypatch.setattr(m,'_strong_engine',lambda:evaluator)
    value=m.strong_d(tmp_path,{},tmp_path/'D',call,a)
    assert value['candidate']=='A' and evaluator.call_args.args[-1] is a
    with pytest.raises(FileExistsError):m.strong_d(tmp_path,{},tmp_path/'D',call,a)
    assert evaluator.call_count==1


def rows(n,center):
    return [{k:center+(i%5) for k in stats.shared.ret.m4.METRICS} for i in range(n)]


def test_existing_welch_engine_accepts_fixed64_and_family_confidence():
    policy=dict(stats.POLICY,samples=64,confidence=.9875)
    binding={'W':10,'shell':{'rz_count':3}}
    result=stats.primary(rows(64,80),rows(64,100),binding,binding,expected_n=64,statistics=policy)
    usual=stats.primary(rows(64,80),rows(64,100),binding,binding,expected_n=64,statistics=dict(policy,confidence=.95))
    assert result['samples_per_arm']==64 and result['confidence']==.9875
    assert result['halfwidth']>usual['halfwidth'] and result['difference_interval'][1]<0
    assert result['ratio_interval'] is None and result['actual_CA_achieved'] is None


def test_statistics_original_defaults_and_incomplete_final():
    binding={'W':10,'shell':{'rz_count':3}}
    old=stats.primary(rows(32,80),rows(32,100),binding,binding)
    assert old['samples_per_arm']==32 and old['confidence']==.95
    with pytest.raises(ValueError):stats.primary(rows(32,80),rows(32,100),binding,binding,expected_n=64)


def test_existing_accounting_accepts64_only_when_explicit_and_preserves_shared_se():
    old=stats.shared.ret
    sample=rows(64,100)
    calibration=old.perf.b2.calibration(sample)
    packet=dict(target_id='invented64',kernel_samples=sample,saved_calibration=calibration)
    kernel=old.accounting.cost_calibration(packet,expected_n=64)
    assert kernel['classical_cost_samples']==64
    with pytest.raises(ValueError):old.accounting.cost_calibration(packet)
    plans=[dict(m=0,shots=3),dict(m=1,shots=5)]
    uncertainty=old.m3.calibration_uncertainty(packet,plans,{'totals':{}},expected_n=64)
    assert uncertainty['samples']==64 and uncertainty['kernel_reuse_weight']==26
    assert uncertainty['weighted_SE']['rz_count']==26*calibration['sample_mean_standard_error']['rz_count']
