"""Invented seven-frame data only;no saved molecular arrays or QPY decoded."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json,subprocess,sys
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests/management'))
from trottertracks import c_cdf7_native32 as m
from test_research_c_cdf7_PR import fixture as original_fixture


def saved_fixture(root):
    from qiskit import qpy
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
    c,records,p,d=original_fixture(root)
    shells={}
    for label in ('head','X','Y'):
        raw=m.b2.shell_build(8,label);compiled=compile_checked(raw,CompilerProfile());shells[label]={}
        for kind,qc in [('raw',raw),('compiled',compiled)]:
            path=root/(label+'_'+kind+'.qpy');shells[label][kind]=path.name
            with path.open('xb') as stream:qpy.dump(qc,stream)
        records['shell_reports'][label]['metrics']=circuit_metrics(compiled)
    (root/c['records']['shell_reports']).write_text(json.dumps(records['shell_reports']))
    c.update(policy=m.pr.POLICY,target=m.pr.TARGET)
    old=root/'saved_PR';old.mkdir();call=m.integrated.Calls(old,m.pr.LIMITS,119)
    result=m.pr.execute(root,c,old,call)
    m.io.save(old/'contract.json',c);m.io.save(old/'SOURCE.json',dict(execution_commit=m.PR_SOURCE))
    m.io.save(old/'STOP.json',dict(failed=False,scientific_units=119,call_counts=m.pr.LIMITS,
        retry=0,resume=0,mandatory_stop=True))
    contract=dict(input=c['input'],export=c['export'],export_sha256=c['export_sha256'],input_sha256=c['original_input_sha256'],
        records={k:str(v.relative_to(root)) for k,v in dict(PR=old/'result.json',STOP=old/'STOP.json',SOURCE=old/'SOURCE.json',
          contract=old/'contract.json',tail=old/'reference/tail.json').items()},
        PR_contract_sha256=m.io.sha(old/'contract.json'),saved_reference='saved_PR/reference/sector_reference.npz',
        saved_ground='saved_PR/reference/ground_CDF7.npz',source_manifest_sha256='ARTIFICIAL_NATIVE_SOURCE',shells=shells)
    return contract,p,d,result


def test_passive_and_registered_counts():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_cdf7_native32 as m;assert sum(m.NATIVE_LIMITS.values())==323;assert sum(m.LOAD_LIMITS.values())==9;assert m.MAX_CALLS==332;assert not any(x in sys.modules for x in ['numpy','scipy','qiskit','ffsim','jax'])"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def mock_native(monkeypatch):
    from qiskit import QuantumCircuit
    monkeypatch.setattr(m.b2.b,'build_kernel',lambda *x,**k:QuantumCircuit(9))
    monkeypatch.setattr(m.b2,'reference_context',lambda *x:None)
    monkeypatch.setattr(m.b2,'law_gate',lambda *x:dict(status='MOCK_FULL_LAW'))
    monkeypatch.setattr(m.b2,'trajectory_reference',lambda *x:np.eye(2,dtype=complex))
    monkeypatch.setattr(m.b2,'raw_gate',lambda *x:(None,None,dict(status='MOCK_RAW')))
    monkeypatch.setattr(m.b2,'compiled_gate',lambda *x:(None,dict(status='MOCK_COMPILED')))
    monkeypatch.setattr(m.b2,'adjoint_pair_gate',lambda *x:({'raw':QuantumCircuit(9),'compiled':QuantumCircuit(9)},dict(status='MOCK_ADJOINT')))
    monkeypatch.setattr(m.b2,'XY_gate',lambda *x:dict(status='MOCK_XY'))


def test_artificial_saved119_to_native332_QPY128(tmp_path,monkeypatch):
    c,p,d,_=saved_fixture(tmp_path);mock_native(monkeypatch)
    out=tmp_path/'new_native';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with threadpool_limits(limits=1):r=m.execute(tmp_path,c,out,call)
    assert dict(call.counts)==m.LIMITS and sum(call.counts.values())==332 and not call.stop.is_set()
    assert len(list((out/'circuits').glob('*.qpy')))==128
    assert len(list((out/'samples').glob('*_trajectory.json')))==32
    assert len(m.io.read(out/'sector_gate.json'))==9 # eight factors plus saved sum
    assert r['completed_build_compile']==64 and r['QPY_count']==128
    assert r['cost_support'] is r['actual_CA_achieved'] is r['ideal_law_mean_advantage'] is None
    assert r['new_fit']==0 and r['PF_RTE_shots_recomputed'] is False
    from trottertracks.c_pr_aware_df_m2 import sector_match as legacy
    with pytest.raises(ValueError,match='SAVED_SECTOR_MAPPING'):
        legacy([np.eye(256)]*8,dict(generators=np.zeros((8,36,36)),full_input_indices=np.arange(36),H=np.zeros((36,36))))


def test_rank7_one_real_artificial128_full512_compile_adjoint_XY(tmp_path):
    from trottertracks.c1_h4_strong_baseline import StrongBuilder
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    c,_,p,d=original_fixture(tmp_path);b=m.b2.b;h=Fraction(1,10)
    arm=b.prepare_arm(**p,**d,role='baseline',source_identity='ARTIFICIAL_JU',input_identity='ARTIFICIAL_INPUT',ld=2)
    with threadpool_limits(limits=1):
        car=b.stage1.car_bilinears(8);G=[b.stage1.raw_base(b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car)]
        G.extend(b.stage1.raw_frame(J,U,car)[0] for J,U in zip(d['J'],d['U'],strict=True))
        tail,config,dist=b.configure(arm,h,128,0);ctx=m.b2.reference_context(arm,G,car,h)
        events,_=m.sample(arm,tail,dist,config,0,'ARTIFICIAL_NATIVE')
        ref=m.b2.trajectory_reference(arm,ctx,h,events)
        raw=b.build_kernel(arm,h,events,config,builder=StrongBuilder(relative_fusion=True))
        compiled=compile_checked(raw,CompilerProfile());R,target,_=m.b2.raw_gate(raw,ref)
        C,_=m.b2.compiled_gate(compiled,R,target);m.b2.adjoint_pair_gate(raw,compiled,R,C,target)
        shells={}
        for label in ('head','X','Y'):
            qc=m.b2.shell_build(8,label);shells[label],_=m.b2.shell_assessment(qc,compile_checked(qc,CompilerProfile()),label)
        psi=np.zeros(256,complex);psi[0]=1
        assert m.b2.XY_gate(R,C,ref,psi,shells)['status']=='PASS'
        with pytest.raises(ValueError):m.b2.raw_gate(raw,np.exp(.01j)*ref)
        with pytest.raises(ValueError):m.b2.compiled_gate(compiled,R,np.exp(.01j)*target)


def test_cohort_keys_source_JU_and_independent32(tmp_path):
    _,_,p,d=original_fixture(tmp_path);b=m.b2.b
    arm=b.prepare_arm(**p,**d,role='baseline',source_identity='ARTIFICIAL_JU',input_identity='I',ld=2)
    tail,config,dist=b.configure(arm,Fraction(1,10),128,0)
    records=[m.sample(arm,tail,dist,config,i,'SOURCE_A')[1] for i in range(32)]
    assert len({x['seed'] for x in records})==32
    assert m.sample(arm,tail,dist,config,0,'SOURCE_A')[1]==records[0]
    assert m.sample(arm,tail,dist,config,0,'SOURCE_B')[1]['seed']!=records[0]['seed']
    assert not any(x['paired'] or x['old_cohort_pooled'] or x['reused_as_coherent_repetitions'] for x in records)
    with pytest.raises(ValueError):m.sample(arm,tail,dist,config,32,'SOURCE_A')


@pytest.mark.parametrize('fault',['SOURCE','sector','law'])
def test_saved_fault_stops_before_sampling(tmp_path,monkeypatch,fault):
    c,_,_,_=saved_fixture(tmp_path);mock_native(monkeypatch)
    if fault=='SOURCE':(tmp_path/c['records']['SOURCE']).write_text(json.dumps(dict(execution_commit='WRONG')))
    elif fault=='law':
        v=m.io.read(tmp_path/c['records']['tail']);v['components'][0]['probability']+=.01
        (tmp_path/c['records']['tail']).write_text(json.dumps(v))
    else:
        path=tmp_path/c['saved_reference']
        with np.load(path) as z:values={k:z[k] for k in z.files}
        values['generators'][0,0,0]+=.01;np.savez(path,**values)
    out=tmp_path/'new_native';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with pytest.raises(ValueError),threadpool_limits(limits=1):m.execute(tmp_path,c,out,call)
    assert call.stop.is_set() and call.counts.get('trajectory_sample',0)==0


def test_shared_calibration_SE_and_shell_once(tmp_path):
    c,_,_,_=saved_fixture(tmp_path);v=m.evidence(tmp_path,c);pr=v['PR']
    cal=m.b2.calibration([{k:10+i for k in m.pr.old.old.METRICS} for i in range(32)])
    a=dict(prices=pr['price'],saved=v['saved']);r=m.calibrated_result(cal,a)
    lane='ideal_exact_native';aff=a['prices']['affine'][lane];out=r['lanes'][lane]
    assert out['conditional_totals']['rz_count']==aff['W']*cal['sample_mean']['rz_count']+aff['S']['rz_count']
    assert out['weighted_calibration_SE']['rz_count']==aff['W']*cal['sample_mean_standard_error']['rz_count']
    assert out['independent_round_calibrations']==1 and r['cost_support'] is None
    assert set(out['comparisons'])=={'CDF8_gamma0','M4_r128'}


def test_passive_byte_preflight_and_partial_STOP(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'FRESH','new_run');monkeypatch.setattr(m.pr,'EXPORT','JU.npz');monkeypatch.setattr(m.pr,'EXPORT_SHA','OPAQUE_HASH')
    m.io.save(tmp_path/'source.json',dict(files={}))
    records={k:k+'.json' for k in ('PR','STOP','SOURCE','contract','tail')}
    shells={k:{j:k+j+'.qpy' for j in ('raw','compiled')} for k in ('head','X','Y')}
    paths={'input.npz','JU.npz','reference.npz','ground.npz',*records.values(),*(p for x in shells.values() for p in x.values())}
    for p in paths:(tmp_path/p).write_bytes(b'OPAQUE')
    digest=m.io.sha(tmp_path/'JU.npz');monkeypatch.setattr(m.pr,'EXPORT_SHA',digest)
    c=dict(scope=m.SCOPE,implementation=m.VERSION,policy=m.POLICY,allowances=m.ALLOWANCES,sampling=m.SAMPLING,
        compiler=m.b2.profile_dict(),call_limits=m.LIMITS,maximum_logical_units=332,fresh_output=m.FRESH,
        input_sha256=m.pr.fit.io.P_H4_SHA256,input='input.npz',export='JU.npz',export_sha256=digest,
        PR_SOURCE=m.PR_SOURCE,PR_contract_sha256=m.PR_CONTRACT_SHA,new_fit=0,optimizer=0,retry=0,resume=0,
        PF_RTE_shots_recomputed=False,QPY_count=128,records=records,shells=shells,saved_reference='reference.npz',saved_ground='ground.npz',
        source_manifest='source.json',source_manifest_sha256=m.io.sha(tmp_path/'source.json'),packages={},saved_inputs={p:m.io.sha(tmp_path/p) for p in paths})
    m.io.save(tmp_path/'c.json',c);assert m.preflight(tmp_path,tmp_path/'c.json')==c
    (tmp_path/'ground.npz').write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='SOURCE_INPUT'):m.preflight(tmp_path,tmp_path/'c.json')
    (tmp_path/'ground.npz').write_bytes(b'OPAQUE')
    def fail(root,c,out,call):
        m.io.save(out/'partial.json',dict(preserved=True))
        def wrong():raise ValueError('ARTIFICIAL_CORRECTNESS')
        call('evidence_load','artificial',wrong)
    monkeypatch.setattr(m,'execute',fail)
    with pytest.raises(ValueError,match='ARTIFICIAL_CORRECTNESS'):m.run(tmp_path,tmp_path/'c.json')
    stop=m.io.read(tmp_path/'new_run/STOP.json');assert stop['failed'] and stop['scientific_units']==1
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,tmp_path/'c.json')
