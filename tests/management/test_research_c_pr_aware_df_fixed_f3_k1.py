"""F3 binding tests: saved JSON/bytes only, invented J/U/events/circuits."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from test_research_c_pr_aware_df_fixed_car import example,build_raw
from trottertracks import c_pr_aware_df_fixed_f3_k1 as e
from trottertracks import c_pr_aware_df_stage_b as b
from trottertracks.c1_stage3_integrated import Calls
CONTRACT=ROOT/'docs/research/research_c_pr_aware_df_F3_K1_contract_draft_20261027.json'


def candidate():
    f1=e.io.read(ROOT/(e.F1+'/contract.json'))
    f2=e.io.read(ROOT/e.F2_CONTRACT)
    c=deepcopy(f2)
    for key in ('input','export','trajectory','circuits','sampling_SOURCE_sha256','original_input_sha256','B2_execution_SOURCE_commit'):
        c[key]=f1[key]
    c.update(scope=e.SCOPE,execution_adapter=e.VERSION,base_commit=e.BASE,arm='proposed',
             trajectory_index=0,seed=e.SEED,paired_trajectory=False,fresh_output=e.FRESH,
             reuse=e.expected_reuse(f1),saved_inputs=dict(f1['saved_inputs']))
    return c


def records():
    return [e.io.read(ROOT/(e.F1+'/'+n)) for n in ('contract.json','STOP.json','result.json','compiled_raw.json')]


def test_F1_saved_scope_and_correct_raw_reuse():
    c=candidate();e.f1_gate(c,*records())
    assert c['reuse']['compiled_raw']==e.F1+'/compiled_raw.json'
    assert 'kernel_compiled_raw' not in c['reuse']['compiled_raw']


@pytest.mark.parametrize('fault',['baseline_certificate','raw_hash','raw_path','compiled_path'])
def test_baseline_certificate_and_raw_mix_rejected(fault):
    c=candidate()
    if fault=='baseline_certificate':c['reuse']['compiled_raw']='artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1-run01/kernel_compiled_raw.json'
    if fault=='raw_hash':c['reuse']['raw_sha256']='wrong'
    if fault=='raw_path':c['circuits']['raw']=c['circuits']['raw'].replace('proposed','baseline')
    if fault=='compiled_path':c['circuits']['compiled']=c['circuits']['compiled'].replace('proposed','baseline')
    with pytest.raises(ValueError,match='RAW_IDENTITY'):e.f1_gate(c,*records())


@pytest.mark.parametrize('field,value',[('status','NUMERICAL_PILOT'),('norm','sector36'),('dimension',32),
    ('columns_evaluated',16),('phase_fit',True),('precision_bits',128),('allowance_exact','1/100'),
    ('upper_exact','1/10'),('gate_status','UNRESOLVED')])
def test_F1_certificate_qualification(field,value):
    r=records();r[-1][field]=value
    with pytest.raises(ValueError,match='QUALIFICATION'):e.f1_gate(candidate(),*r)


@pytest.mark.parametrize('fault',['role','STOP','result'])
def test_F1_completion_required(fault):
    r=records()
    if fault=='role':r[0]['arm']='baseline'
    if fault=='STOP':r[1]['retry']=1
    if fault=='result':r[2]['gates']['raw_actual_adjoint']='UNRESOLVED'
    with pytest.raises(ValueError):e.f1_gate(candidate(),*r)


def test_proposed_trajectory_metadata_and_event_order():
    c=candidate();p=e.io.read(ROOT/c['trajectory'])
    assert e.trajectory_gate(p,c)==p and len(p['events'])==16
    assert p['seed']==e.SEED
    assert [x['rotation_component_id'] for x in p['events']]==[x[1] for x in p['trajectory_key']]


@pytest.mark.parametrize('fault',['arm','seed','source','export','order','law'])
def test_trajectory_role_seed_source_law_order_rejection(fault):
    c=candidate();p=e.io.read(ROOT/c['trajectory'])
    if fault=='arm':p['arm']='baseline'
    if fault=='seed':p['seed']=2939838941121001441
    if fault=='source':p['SOURCE_sha256']='other'
    if fault=='export':p['final_JU_sha256']='baseline'
    if fault=='order':p['events']=list(reversed(p['events']))
    if fault=='law':p['cohort']='paired-index0-law'
    with pytest.raises(ValueError):e.trajectory_gate(p,c)


def test_common_kappa_diagnostic_reuse_without_numeric_decode(monkeypatch):
    c=candidate();c['diagnostic_exact']=e.DIAGNOSTIC+'/exact_dyadic.json'
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('no saved arrays in metadata gate'))
    e.diagnostic_gate(ROOT,c)
    c['saved_inputs'][c['input']]='wrong'
    with pytest.raises(ValueError,match='KAPPA_DIAGNOSTIC'):e.diagnostic_gate(ROOT,c)


@pytest.fixture
def proposed(example):
    old,h,_,_=example
    arm=b.prepare_arm(h,np.zeros((2,)*4),.19,old.J,old.U,role='proposed',source_identity='artificialJU',input_identity='artificialInput',ld=2)
    tail,config,dist=b.configure(arm,F(1,40),16,0)
    from trotterlib.rte import _make_event
    choices=[next(i for i,a in enumerate(arm.plan.atoms) if a.coefficient<0),
             next(i for i,a in enumerate(arm.plan.atoms) if a.frame_id.endswith('F3') and len(a.support)==2)]
    events=tuple(_make_event((choices[i%2],),tail.components,dist,0) for i in range(16))
    payload=dict(arm='proposed',sample=0,seed=e.SEED,SOURCE_sha256='sampling',final_JU_sha256='artificialJU',
        cohort='B2_fixed32_per_arm_v1',microsteps=16,quantum_shots=0,reused_as_coherent_repetitions=False,
        trajectory_key=[[0,x.rotation_component_id,[]] for x in events],events=[x.to_dict() for x in events])
    return arm,h,events,config,payload


def test_Proposed_CAR_bridges_full_Fock_and_raw_certificate(proposed):
    arm,h,events,config,_=proposed
    literal=h.astype(complex);literal[0,0]+=1j*2.**-45
    ref=e.k1.build_reference(arm,literal,.19,events,parallel=False)
    raw=e.k1.fixed.record(build_raw(arm,events,config))
    row=e.k1.compare(raw,ref,raw_identity='artificialraw',parallel=False)
    assert row['columns_evaluated']==row['dimension']==32
    assert F(row['upper_exact'])<F(1,10**8)
    assert row['target']==e.k1.TARGET
    assert F(row['upper_exact'])==sum(F(row[n]) for n in ('literal_CAR_upper_exact','kappa_bridge_upper_exact','U_bridge_upper_exact'))
    assert row['whole_law_bound'] is row['CA_achieved'] is None


def test_prepare_dispatches_Proposed_only_and_reuses_CAR(tmp_path,monkeypatch,proposed):
    arm,h,events,_,payload=proposed
    c=dict(input='input',export='export',saved_inputs={'export':'artificialJU'},
           original_input_sha256='artificialInput',trajectory='trajectory.json',sampling_SOURCE_sha256='sampling',diagnostic_exact='diagnostic.json')
    e.io.save(tmp_path/'trajectory.json',payload);e.io.save(tmp_path/'diagnostic.json',{'invented':True})
    packet=dict(h=h,g=np.zeros((2,)*4),constant=.19);data=dict(J=arm.J,U=arm.U)
    monkeypatch.setattr(e.f2,'load_arrays',lambda *a:(packet,data))
    original=b.prepare_arm;roles=[]
    def spy(*a,**kw):roles.append(kw['role']);return original(*a,**kw)
    monkeypatch.setattr(b,'prepare_arm',spy)
    def fake_CAR(actual,kappa,constant,actual_events,**kw):
        assert actual.role=='proposed' and actual_events==events
        assert kw['diagnostic_record']=={'invented':True}
        assert np.array_equal(kappa,h)
        kw['progress']('partial',{'invented':True})
        return {'invented':True}
    monkeypatch.setattr(e.k1,'build_reference',fake_CAR)
    call=Calls(tmp_path,e.LIMITS,7)
    assert e.prepare(tmp_path,c,tmp_path,call)=={'invented':True}
    assert roles==['proposed'] and sum(call.counts.values())==4
    assert (tmp_path/'partial.json').exists()


@pytest.mark.parametrize('outcome',['PASS','FAIL','WIDTH','NULL'])
def test_shared_executor_F1_triangle_and_partial_behavior(tmp_path,monkeypatch,outcome):
    c=dict(circuits={'raw':'proposed_00_raw.qpy'},saved_inputs={'proposed_00_raw.qpy':'rawhash'},
           reuse=dict(compiled_raw='F1.json',raw_sha256='rawhash'))
    prior=dict(status='VALIDATED_FIXED_CIRCUIT_PAIR',norm='full_operator_2',phase_fit=False,precision_bits=192,dimension=512,upper_exact='1/1000000000000')
    e.io.save(tmp_path/'F1.json',prior)
    def reference(root,c,out,call):
        for kind in ('input_arrays','arm_replay','trajectory_load'):call(kind,'invented',lambda:None)
        return call('CAR_reference','invented',lambda:{'invented':True})
    row=dict(status='VALIDATED_FIXED_RAW_IDEAL_CAR',target=e.k1.TARGET,norm='full_operator_2',
        dimension=512,columns_evaluated=512,phase_fit=False,precision_bits=192,raw_identity_sha256='rawhash',
        upper_exact='1/100000000000',lower_exact='0')
    if outcome=='FAIL':row.update(upper_exact='1',lower_exact='1/100')
    if outcome=='WIDTH':row.update(upper_exact='1',lower_exact='0')
    if outcome=='NULL':row.update(upper_exact=None,lower_exact=None)
    def compare(*a,**kw):assert kw['raw_identity']=='rawhash';return row
    call=Calls(tmp_path,e.LIMITS,7)
    if outcome=='FAIL':
        with pytest.raises(ValueError,match='ALLOWANCE_FAIL'):
            e.execute(tmp_path,c,tmp_path,call,reference_loader=reference,loader=lambda *a,**kw:{},compare=compare)
        assert call.stop.is_set()
    else:
        result=e.execute(tmp_path,c,tmp_path,call,reference_loader=reference,loader=lambda *a,**kw:{},compare=compare)
        assert result['status']==('COMPLETE' if outcome=='PASS' else 'UNRESOLVED')
        assert result['CA_achieved'] is result['shots'] is result['total_resources'] is None
    assert sum(call.counts.values())==(7 if outcome=='PASS' else 6)
    assert (tmp_path/'CAR_reference.json').exists() and (tmp_path/'raw_CAR.json').exists()
    assert (tmp_path/'compiled_CAR.json').exists()==(outcome=='PASS')
    journal=(tmp_path/'science_calls.jsonl').read_text()
    assert 'same proposed_00 raw' in journal and 'baseline_00' not in journal


def test_runner_failure_preserves_STOP_and_no_resume(tmp_path,monkeypatch):
    e.io.save(tmp_path/'SOURCE.json',{'files':{}});e.io.save(tmp_path/'contract.json',{'source_manifest':'SOURCE.json'})
    monkeypatch.setattr(e,'preflight',lambda *a:None);monkeypatch.setattr(e,'FRESH','run01')
    def fail(root,c,out,call):
        call('input_arrays','artificial',lambda:None);e.io.save(out/'partial.json',{'invented':True})
        call('arm_replay','fault',lambda:(_ for _ in ()).throw(ValueError('INJECTED_FAIL')))
    monkeypatch.setattr(e,'execute',fail)
    with pytest.raises(ValueError,match='INJECTED_FAIL'):e.run(tmp_path,tmp_path/'contract.json')
    stop=e.io.read(tmp_path/'run01/STOP.json')
    assert stop['scientific_units']==2 and stop['arm']=='proposed' and stop['mandatory_stop']
    assert stop['retry']==stop['resume']==0 and not stop['F1_reexecuted'] and not stop['F2_reexecuted']
    assert (tmp_path/'run01/failure.json').exists() and (tmp_path/'run01/partial.json').exists()
    with pytest.raises(FileExistsError):e.run(tmp_path,tmp_path/'contract.json')


def test_frozen_F2_math_and_call_policy_reused():
    assert e.POLICY==e.f2.POLICY and e.LIMITS==e.f2.LIMITS
    assert e.k1 is e.f2.k1 and e.FRESH!=e.f2.FRESH
    assert e.f2.SCOPE=='C_PR_AWARE_DF_FIXED_F2_K1_V1'


def test_passive_no_saved_array_or_QPY_decode(monkeypatch):
    if not CONTRACT.exists():pytest.skip('freeze contract after first artificial check')
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('passive NPZ decode'))
    monkeypatch.setattr(e.f2.old.old,'_read_qpy',lambda *a,**kw:pytest.fail('passive QPY decode'))
    assert e.preflight(ROOT,CONTRACT)['scientific_calls']==0
    assert not (ROOT/e.FRESH).exists()


@pytest.mark.parametrize('fault',['source','input','arm','paired','seed','reuse'])
def test_frozen_scope_mismatch_rejected(tmp_path,monkeypatch,fault):
    if not CONTRACT.exists():pytest.skip('freeze contract after first artificial check')
    c=e.io.read(CONTRACT);real=e.io.sha
    if fault in ('source','input'):
        name='src/trottertracks/c_pr_aware_df_fixed_f3_k1.py' if fault=='source' else c['export']
        monkeypatch.setattr(e.io,'sha',lambda p:'0'*64 if Path(p)==ROOT/name else real(p))
        path=CONTRACT
    else:
        if fault=='arm':c['arm']='baseline'
        if fault=='paired':c['paired_trajectory']=True
        if fault=='seed':c['seed']=2939838941121001441
        if fault=='reuse':c['reuse']['compiled_raw']='baseline_certificate.json'
        path=tmp_path/'contract.json';e.io.save(path,c)
    with pytest.raises(ValueError):e.preflight(ROOT,path)


def test_consumed_F2_output_still_rejected_without_decode(monkeypatch):
    # F2 was executed in this base commit. Its historical pre-run-only test
    # expects absence; preserve it and verify the correct consumed-run behavior.
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('no old science reexecution'))
    path=ROOT/'docs/research/research_c_pr_aware_df_F2_K1_contract_draft_20261027.json'
    with pytest.raises(ValueError,match='FRESH_OUTPUT_REQUIRED_NO_RETRY'):
        e.f2.preflight(ROOT,path)
