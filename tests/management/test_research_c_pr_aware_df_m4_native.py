"""Future-native128 artificial tests only. No saved molecular inputs."""
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_m4_native as n


def test_fixed_next_scope_332_samples128_namespace():
    from test_research_c_pr_aware_df_stage_b2 import artificial
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks import c_pr_aware_df_m3 as m3
    arm,_=artificial(n=2,rank=8,role='proposed')
    tail,c,dist=b.configure(arm,Fraction(1,10),128,0)
    assert n.MAX_CALLS==332 and n.LIMITS['trajectory_sample']==n.LIMITS['compile']==32
    seeds=set()
    for i in range(32):
        events,record=n.sample(arm,tail,dist,c,i,'ARTIFICIAL_NATIVE_SOURCE')
        assert len(events)==128 and record['microsteps']==128 and not record['paired']
        assert record['seed'] not in seeds;seeds.add(record['seed'])
    _,c64,d64=b.configure(arm,Fraction(1,10),64,0)
    _,r64=m3.sample_trajectory(arm,tail,d64,c64,0,'ARTIFICIAL_NATIVE_SOURCE')
    assert n.sample(arm,tail,dist,c,0,'ARTIFICIAL_NATIVE_SOURCE')[1]['seed']!=r64['seed']
    with pytest.raises(ValueError,match='SAMPLE_SCOPE'):n.sample(arm,tail,d64,c64,0,'S')


def test_artificial128_events_full512_precision_P0_reference_adjoint_XY():
    from test_research_c_pr_aware_df_stage_b2 import artificial
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from threadpoolctl import threadpool_limits
    arm,p=artificial(n=4,rank=8,role='proposed');h=Fraction(1,10)
    with threadpool_limits(limits=1):
        car=b.stage1.car_bilinears(8);G=[b.stage1.raw_base(b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car)]
        G.extend(b.stage1.raw_frame(J,U,car)[0] for J,U in zip(arm.J,arm.U,strict=True))
        ctx=b2.reference_context(arm,G,car,h);tail,c,dist=b.configure(arm,h,128,0)
        events,_=n.sample(arm,tail,dist,c,0,'ARTIFICIAL_NATIVE_SOURCE')
        ref=b2.trajectory_reference(arm,ctx,h,events)
        np.testing.assert_allclose(ref,b.trajectory_reference(arm,G,ctx['atoms'],h,events,c),atol=1e-12,rtol=0)
        raw=b.build_kernel(arm,h,events,c);compiled=compile_checked(raw,CompilerProfile())
        R,target,_=b2.raw_gate(raw,ref);C,_=b2.compiled_gate(compiled,R,target)
        adj,checks=b2.adjoint_pair_gate(raw,compiled,R,C,target)
        shells={}
        for label in ('head','X','Y'):
            qc=b2.shell_build(8,label);shells[label],_=b2.shell_assessment(qc,compile_checked(qc,CompilerProfile()),label)
        psi=np.zeros(256,complex);psi[0]=1
        assert len(R)==512 and b2.XY_gate(R,C,ref,psi,shells)['status']=='PASS'


def test_calibrated_result_uses_saved_affine_sharedSE_without_newshots():
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    metrics=[{k:10+i for k in n.m4.METRICS} for i in range(32)];cal=b2.calibration(metrics)
    saved=dict(M4=dict(C=dict(affine={lane:dict(W=100,S={k:5 for k in n.m4.METRICS}) for lane in ('ideal_exact_native','assumed_raw_signal_budget')},
        comparisons={lane+':'+target:dict(thresholds={k:dict(mu_break_even=100.) for k in n.m4.METRICS}) for lane in ('ideal_exact_native','assumed_raw_signal_budget') for target in ('M3_r64','M2_deterministic','old_PR_s4')})))
    r=n.calibrated_result(cal,saved)
    assert r['lanes']['ideal_exact_native']['conditional_totals']['rz_count']==2555
    assert r['lanes']['ideal_exact_native']['weighted_calibration_SE']['rz_count']==100*cal['sample_mean_standard_error']['rz_count']
    assert r['cost_support'] is r['actual_CA_achieved'] is None


def test_mock_native32_dispatch_calls_QPY_and_old_helpers(tmp_path,monkeypatch):
    from test_research_c_pr_aware_df_m4 import synthetic_fixture
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    from trottertracks import c_pr_aware_df_stage_b_runner as loader
    from qiskit import QuantumCircuit,qpy
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):c,saved=synthetic_fixture(tmp_path)
    p=loader.load(tmp_path/c['input'],('h','g','constant'));d=loader.load(tmp_path/c['exports']['proposed'],('J','U'))
    arm=b.prepare_arm(**p,**d,role='proposed',source_identity='SYNTHETIC',input_identity='SYNTHETIC',ld=2)
    tail,config,dist=b.configure(arm,Fraction(1,10),128,0)
    sector=loader.load(tmp_path/c['M3_sector'],('generators','H','full_input_indices'))
    ground=loader.load(tmp_path/c['grounds']['proposed'],('energy','sector_state'))
    saved['M4']=dict(C=dict(affine={lane:dict(W=100,S={k:5 for k in n.m4.METRICS}) for lane in ('ideal_exact_native','assumed_raw_signal_budget')},
        comparisons={lane+':'+target:dict(thresholds={k:dict(mu_break_even=100.) for k in n.m4.METRICS}) for lane in ('ideal_exact_native','assumed_raw_signal_budget') for target in ('M3_r64','M2_deterministic','old_PR_s4')}))
    a=dict(p=p,arm=arm,tail=tail,config=config,distribution=dist,sector=sector,ground=ground,saved=saved)
    monkeypatch.setattr(n,'attach',lambda *args:a)
    monkeypatch.setattr(b,'build_kernel',lambda *args,**kw:QuantumCircuit(9))
    monkeypatch.setattr(b2,'reference_context',lambda *args:None)
    monkeypatch.setattr(b2,'law_gate',lambda *args:dict(status='ARTIFICIAL'))
    monkeypatch.setattr(b2,'trajectory_reference',lambda *args:np.eye(256,dtype=complex))
    c['source_manifest_sha256']='ARTIFICIAL_SOURCE';c['shells']={}
    for label in ('head','X','Y'):
        qc=b2.shell_build(8,label);c['shells'][label]={}
        for kind in ('raw','compiled'):
            name=label+kind+'.qpy';c['shells'][label][kind]=name
            with (tmp_path/name).open('xb') as f:qpy.dump(qc,f)
    out=tmp_path/'future';out.mkdir();call=n.m4.integrated.Calls(out,n.LIMITS,n.MAX_CALLS)
    with threadpool_limits(limits=1):r=n.execute(tmp_path,c,out,call)
    assert r['status']=='NATIVE128_FINITE_COHORT_COMPLETE'
    assert {k:call.counts[k] for k in n.CORE}==n.CORE and call.counts['native_result']==1
    assert sum(call.counts.values())==325 # seven attachment calls mocked above
    assert len(list((out/'circuits').glob('*.qpy')))==128


def test_native_passive_and_partial_STOP_no_resume(tmp_path,monkeypatch):
    (tmp_path/'opaque').write_bytes(b'synthetic only')
    n.io.save(tmp_path/'source.json',dict(files={}))
    c=dict(scope=n.SCOPE,implementation=n.VERSION,sampling=n.SAMPLING,call_limits=n.LIMITS,
        maximum_logical_units=n.MAX_CALLS,fresh_output=n.FRESH,retry=0,resume=0,
        source_manifest='source.json',source_manifest_sha256=n.io.sha(tmp_path/'source.json'),
        saved_inputs={'opaque':n.io.sha(tmp_path/'opaque')},packages={})
    n.io.save(tmp_path/'contract.json',c);assert n.preflight(tmp_path,tmp_path/'contract.json')==c
    monkeypatch.setattr(n.subprocess,'check_output',lambda *args,**kw:'ARTIFICIAL_NATIVE_SOURCE\n')
    def fail(root,c,call):
        n.io.save(tmp_path/n.FRESH/'partial.json',dict(preserved=True))
        def error():raise ValueError('ARTIFICIAL_NATIVE128_FAILURE')
        return call('evidence_load','synthetic',error)
    monkeypatch.setattr(n,'attach',fail)
    with pytest.raises(ValueError,match='ARTIFICIAL_NATIVE128_FAILURE'):n.run(tmp_path,tmp_path/'contract.json')
    stop=n.io.read(tmp_path/n.FRESH/'STOP.json')
    assert stop['failed'] and stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    with pytest.raises(ValueError,match='FRESH_NATIVE128'):n.run(tmp_path,tmp_path/'contract.json')
