"""Artificial transfer and mock stage dispatch; no saved numerical decoding."""
from pathlib import Path
from copy import deepcopy
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_basis_transfer as t,c_df_basis_third as m


def rows(start=0,shift=0,constant=False,negative=False):
    out=[]
    for i in range(32):
        x=1 if constant else start+i
        out.append(dict(sample=i,features=dict(basis_switches=x,overlap_path=x*x),
            metrics={k:(100-2*x if negative else 10+3*x+shift) for k in m.old.METRICS}))
    return out


def test_global_intercept_frozen_before_heldout_prices():
    snap=t.freeze_training(dict(M4_Proposed=rows(),tail_P4=rows(32)),'rz_count')
    original=deepcopy(snap)
    out=t.evaluate(snap,rows(64),cohort=t.HOLDOUT)
    assert out['models']['A_switch']['MAE']<1e-10
    assert t.qualification(out)['status']=='SWITCH_TRANSFER_DIAGNOSTIC_PASS'
    shifted=t.evaluate(snap,rows(64,shift=100),cohort=t.HOLDOUT)
    assert shifted['models']['A_switch']['MAE']==pytest.approx(100)
    assert shifted['models']['A_switch']['signed_mean_prediction_error']==pytest.approx(-100)
    assert snap==original and not out['head_effect_separated']
    assert set(snap['models'])=={'mean_only','A_switch','B_overlap'}


def test_heldout_leakage_feature_change_rank_and_negative_predictions():
    train=dict(M4_Proposed=rows(),tail_P4=rows(32))
    with pytest.raises(ValueError,match='TRAIN_DF'):t.freeze_training(dict(train,StageA_gamma0=rows()),'rz_count')
    snap=t.freeze_training(train,'rz_count')
    with pytest.raises(ValueError,match='LEAKAGE'):t.evaluate(snap,rows(),cohort='M4_Proposed')
    bad=deepcopy(snap);bad['models']['A_switch']['features']=['third_price']
    with pytest.raises(ValueError,match='FEATURES'):t.evaluate(bad,rows(),cohort=t.HOLDOUT)
    rank=t.freeze_training(dict(M4_Proposed=rows(constant=True),tail_P4=rows(constant=True)),'rz_count')
    assert t.qualification(t.evaluate(rank,rows(),cohort=t.HOLDOUT))['status']=='UNRESOLVED'
    neg=t.freeze_training(dict(M4_Proposed=rows(negative=True),tail_P4=rows(negative=True)),'rz_count')
    out=t.evaluate(neg,rows(100),cohort=t.HOLDOUT)
    assert out['models']['A_switch']['negative_predictions']==32
    assert t.qualification(out)['status']=='SWITCH_TRANSFER_NOT_SUPPORTED'


@pytest.mark.parametrize('eligible,native_ok',[(True,True),(True,False),(False,False)])
def test_mock_end_to_end_call_graph_and_gated_native(monkeypatch,tmp_path,eligible,native_ok):
    from trottertracks import c_df_basis_third_native as native
    out=tmp_path/'fresh';out.mkdir()
    call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    def consume(limits):
        for k,n in limits.items():
            for i in range(n):call(k,str(i),lambda:None)
    def train(*args):consume(m.TRAINING_LIMITS);return {k:{} for k in m.old.METRICS}
    def ref(*args):
        consume(m.REFERENCE_LIMITS)
        return dict(result=dict(status='POTENTIALLY_FEASIBLE' if eligible else 'INFEASIBLE'))
    def acc(*args):consume(m.ACCURACY_LIMITS);return {}
    price_limits={k:v for k,v in m.PRICE_LIMITS.items() if k!='native_GO'}
    def price(*args):consume(price_limits);return {}
    def run_native(*args):consume(m.NATIVE_LIMITS);return dict(descriptor_rows=[])
    monkeypatch.setattr(m,'training',train);monkeypatch.setattr(m,'stage_reference',ref)
    monkeypatch.setattr(m,'stage_accuracy',acc);monkeypatch.setattr(m,'stage_price',price)
    monkeypatch.setattr(m,'native_go',lambda *a:dict(status='GO_FIXED32' if native_ok else 'NOT_RUN_A_GATE'))
    monkeypatch.setattr(native,'execute',run_native)
    monkeypatch.setattr(t,'evaluate',lambda *a,**kw:dict(models={}))
    monkeypatch.setattr(t,'qualification',lambda *a:dict(status='SWITCH_TRANSFER_NOT_SUPPORTED'))
    result=m.execute(tmp_path,{},out,call)
    assert not call.stop.is_set() and result['new_fit']==0 and result['mandatory_stop']
    if native_ok:
        assert dict(call.counts)==m.LIMITS and sum(call.counts.values())==493
        assert result['transfer']['status']=='SWITCH_TRANSFER_NOT_SUPPORTED'
    else:
        assert not call.counts.get('trajectory_sample',0) and result['native'] is None
        assert not call.counts.get('DF_holdout',0)
        assert sum(call.counts.values())==(129 if eligible else 73)


def test_correctness_failure_stops_dependent_stages(monkeypatch,tmp_path):
    out=tmp_path/'fresh';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    monkeypatch.setattr(m,'training',lambda *a:{})
    def fail(*args):
        return call('input_load','artificial mismatched SOURCE',lambda:(_ for _ in ()).throw(ValueError('INPUT_MISMATCH')))
    monkeypatch.setattr(m,'stage_reference',fail)
    monkeypatch.setattr(m,'stage_accuracy',lambda *a:pytest.fail('DEPENDENT SCIENCE MUST STOP'))
    with pytest.raises(ValueError,match='INPUT_MISMATCH'):m.execute(tmp_path,{},out,call)
    assert call.stop.is_set() and not call.counts.get('trajectory_sample',0)


def test_sampling_policy_is_new_source_bound_cohort_and_native_reuse():
    assert m.SAMPLING['cohort'] not in [m.p4.SAMPLING['cohort'],'M4_r128_native_fixed32_v1']
    assert m.SAMPLING['paired'] is False and 'SOURCE_digest' in m.SAMPLING['source_key']
    assert sum(m.STAGES['T'].values())==8 and sum(m.STAGES['A'].values())==64
    assert sum(m.STAGES['B'].values())==357 and m.MAX_CALLS==493
    assert m.stage_accuracy is m.p4.stage_accuracy
    from trottertracks import c_df_basis_third_native as native
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='SAMPLE_SCOPE'):
        native.sample(SimpleNamespace(role='proposed'),None,None,SimpleNamespace(rte_steps=128),0,'ARTIFICIAL')


def test_artificial_reference_native_dispatch_and_holdout493(tmp_path,monkeypatch):
    """Actual new orchestration, old CAR/PF/RPE; mocked price/operator engines."""
    import numpy as np
    from test_research_c_df_structural_pr import fixture
    from qiskit import QuantumCircuit,qpy
    from trottertracks import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b2 as b2
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
    c,saved,p,d=fixture(tmp_path)
    saved['diagnostic']=saved['P4']['final_diagnostic'];saved['sector']=saved['P4_sector']
    saved['M4_native']=saved['native']
    saved['P4']=dict(native=saved['native'])
    monkeypatch.setattr(m,'EXPORT','JU.npz');monkeypatch.setattr(m,'EXPORT_SHA','ARTIFICIAL_GAMMA0')
    monkeypatch.setattr(m,'evidence',lambda *a:saved)
    def train(root,c,out,call):
        data=call('training_rows_load','invented two cohorts',lambda:dict(M4_Proposed=rows(),tail_P4=rows(32)))
        call('training_identity','artificial',lambda:None)
        return {metric:call('transfer_training',metric,lambda metric=metric:t.freeze_training(data,metric)) for metric in m.old.METRICS}
    monkeypatch.setattr(m,'training',train)
    c['shells']={}
    for label in ('head','X','Y'):
        qc=b2.shell_build(8,label);c['shells'][label]={}
        for kind in ('raw','compiled'):
            name=label+kind+'.qpy';c['shells'][label][kind]=name
            with (tmp_path/name).open('xb') as f:qpy.dump(qc,f)
        saved['shell_reports'][label]['metrics']=circuit_metrics(qc)
    monkeypatch.setattr(b,'build_kernel',lambda *a,**kw:QuantumCircuit(9))
    monkeypatch.setattr(b2,'reference_context',lambda *a:None)
    monkeypatch.setattr(b2,'law_gate',lambda *a:dict(status='MOCK'))
    monkeypatch.setattr(b2,'trajectory_reference',lambda *a:np.eye(2,dtype=complex))
    monkeypatch.setattr(b2,'raw_gate',lambda *a:(None,None,dict(status='MOCK')))
    monkeypatch.setattr(b2,'compiled_gate',lambda *a:(None,dict(status='MOCK')))
    monkeypatch.setattr(b2,'adjoint_pair_gate',lambda *a:({'raw':QuantumCircuit(9),'compiled':QuantumCircuit(9)},dict(status='MOCK')))
    monkeypatch.setattr(b2,'XY_gate',lambda *a:dict(status='MOCK'))
    out=tmp_path/'all';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    result=m.execute(tmp_path,c,out,call)
    assert dict(call.counts)==m.LIMITS and sum(call.counts.values())==493
    assert not call.stop.is_set() and result['reference']['ground']['status']=='PASS'
    assert result['reference']['status']=='POTENTIALLY_FEASIBLE'
    assert result['native']['all_law_synthesis'] is None and result['native']['cost_support'] is None
    assert len(list((out/'native/circuits').glob('*.qpy')))==128
    records=[m.io.read(p) for p in sorted((out/'native/samples').glob('*_trajectory.json'))]
    assert len(records)==len({r['seed'] for r in records})==32
    assert all(r['role']=='baseline' and r['final_JU_sha256']=='ARTIFICIAL_GAMMA0' for r in records)
    assert len(result['native']['descriptor_rows'])==32


def test_passive_hash_mismatch_and_no_old_output_reuse(tmp_path,monkeypatch):
    from trottertracks.c_pr_aware_df_stage_b2 import profile_dict
    from trottertracks.c_pr_aware_df_stage_c import SCENARIOS
    monkeypatch.setattr(m,'EXPORT','opaque_JU');(tmp_path/m.EXPORT).write_bytes(b'ARTIFICIAL_UNDECODED')
    monkeypatch.setattr(m,'EXPORT_SHA',m.io.sha(tmp_path/m.EXPORT))
    (tmp_path/'source.json').write_text('{"files":{}}\n')
    c=dict(scope=m.SCOPE,implementation=m.VERSION,policy=m.POLICY,target=m.TARGET,
        allowances=m.ALLOWANCES,sampling=m.SAMPLING,compiler=profile_dict(),scenarios=list(SCENARIOS),
        stage_call_limits=m.STAGES,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        fresh_output=m.FRESH,export=m.EXPORT,export_sha256=m.EXPORT_SHA,new_fit=0,retry=0,resume=0,
        transfer_models={k:list(v) for k,v in t.MODELS.items()},
        transfer_policy='global_intercept_raw_features_OLS_rcond1e-12_frozen_before_third_prices',
        source_manifest='source.json',source_manifest_sha256=m.io.sha(tmp_path/'source.json'),
        input='input',records={k:k for k in m.RECORD_NAMES},
        training_records={k:k for k in ['result','STOP','SOURCE']},shells={},packages={},
        training_rows={k:[k+str(i) for i in range(32)] for k in t.TRAIN})
    paths={c['input'],m.EXPORT,*c['records'].values(),*c['training_records'].values(),
           *(p for values in c['training_rows'].values() for p in values)}
    for path in paths-{m.EXPORT}:(tmp_path/path).write_bytes(b'NO_DECODE')
    c['saved_inputs']={p:m.io.sha(tmp_path/p) for p in paths};m.io.save(tmp_path/'contract.json',c)
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/'input').write_bytes(b'MISMATCH')
    with pytest.raises(ValueError,match='SOURCE_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')
    (tmp_path/'input').write_bytes(b'NO_DECODE');(tmp_path/m.FRESH).mkdir(parents=True)
    with pytest.raises(ValueError,match='FRESH'):m.preflight(tmp_path,tmp_path/'contract.json')
