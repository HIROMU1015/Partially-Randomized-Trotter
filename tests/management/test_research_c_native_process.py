"""Invented DF arrays only: process isolation, full gates, seeds and STOP."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_native_process as process
from trottertracks import c_pr_integrated_performance as perf
from trottertracks import c_h4_relocated_downstream as relocated
from test_research_c_pr_aware_df_stage_b import artificial


@pytest.fixture(autouse=True)
def no_saved_arrays(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('NO_SAVED_SCIENTIFIC_ARRAYS')
    monkeypatch.setattr(perf.loader,'load',forbidden)


def toy(n=2,rank=3,r=8):
    from trottertracks.c1_h4_strong_baseline import StrongBuilder
    from trottertracks.c1_h4_native_fair import basis_request,compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    _,k,_,J,U=artificial(n,rank)
    rel=relocated.prepare(k,.31,J,U,role='baseline',source_identity='invented:process',
        input_identity='invented-input',ld=1 if n==2 else 2)
    arm=rel.arm;tail,config,dist=perf.b.configure(arm,Fraction(1,10),r,0)
    car=perf.b.stage1.car_bilinears(2*n);_,G=relocated.raw_generators(rel,car=car)
    ctx=perf.b2.reference_context(arm,G,car,Fraction(1,10))
    saved=dict(lambda_R=tail.lambda_r,identity=arm.plan.extracted_identity,atoms=len(arm.plan.atoms),
               components=[dict(probability=x.probability) for x in tail.components])
    perf.b2.law_gate(arm,tail,ctx,saved)
    builder=StrongBuilder(relative_fusion=True)
    for block in arm.model.blocks:basis_request(builder,arm.model,block)
    for order in (tuple(range(arm.plan.ld+1)),tuple(reversed(range(arm.plan.ld+1)))):
        for i,j in zip(order,order[1:]):builder.transition(arm.model,arm.model.blocks[i],arm.model.blocks[j])
    shells={}
    for label in ('head','X','Y'):
        raw=perf.b2.shell_build(2*n,label);compiled=compile_checked(raw,CompilerProfile())
        shells[label],_=perf.b2.shell_assessment(raw,compiled,label)
    psi=np.zeros(1<<(2*n),complex);psi[perf.b.stage1.sector_indices(n,n//2,n//2)[0]]=1
    a=dict(arm=arm,tail=tail,config=config,dist=dist)
    prepared=dict(arm=arm,ctx=ctx,psi=psi,builder=builder,shells=shells)
    return a,prepared


def caller():
    counts=Counter();lock=threading.Lock()
    def call(k,l,fn):
        if call.stop.is_set():raise RuntimeError('DEPENDENT_WORK_STOPPED')
        value=fn()
        with lock:counts[k]+=1
        return value
    call.stop=threading.Event();call.counts=counts
    return call


def folder(path):
    path.mkdir()
    for name in ('circuits','samples','references'):(path/name).mkdir()
    return path


def policy(r=8):
    return dict(perf.SAMPLING,cohort='ARTIFICIAL_PARALLEL_ONLY',events_per_sample=r)


def old_samples(c,out,call,a,p,prepared,indices):
    from threadpoolctl import threadpool_limits
    from trottertracks.c_native_sample import run_sample
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    shared=SharedQiskitCalls(call)
    def task(i):return run_sample(i,c,out,call,a,p,prepared['ctx'],prepared['psi'],
        prepared['builder'],prepared['shells'],shared)
    with threadpool_limits(limits=1),shared.adjoint_builds():
        with ThreadPoolExecutor(max_workers=len(indices)) as pool:
            rows=dict(pool.map(task,indices))
    return [rows[i] for i in indices]


def test_same_events_metrics_references_gates_and_processes(tmp_path):
    a,prepared=toy();p=policy();c=dict(source_manifest_sha256='SYNTHETIC_FIXED_SOURCE')
    old=folder(tmp_path/'threads');new=folder(tmp_path/'process')
    baseline=old_samples(c,old,caller(),a,p,prepared,range(4))
    call=caller();rows,plan,reports=process.run_samples(c,new,call,a,p,prepared,indices=range(4))
    assert rows==baseline and len(rows)==4
    assert len({x['pid'] for x in reports})>=2 and os.getpid() not in {x['pid'] for x in reports}
    assert all(x['reference_readonly'] for x in reports) and plan['start_method']=='spawn'
    for i in range(4):
        for kind in ('events','raw','compiled','adjoint','XY','metrics'):
            name=f'sample{i:02d}_{kind}.json'
            assert json.loads((old/'samples'/name).read_text())==json.loads((new/'samples'/name).read_text())
        with np.load(old/'references'/f'sample{i:02d}.npz') as left,np.load(new/'references'/f'sample{i:02d}.npz') as right:
            np.testing.assert_array_equal(left['system_reference'],right['system_reference'])
    assert call.counts['trajectory_sample']==call.counts['kernel_metrics']==4
    assert len(list((new/'circuits').glob('*.qpy')))==16
    packed=process.pack_context(tmp_path/'unused',a,prepared)
    assert packed is prepared['process_context'] and not (tmp_path/'unused').exists()
    with pytest.raises(ValueError,match='PREPARED'):
        process.pack_context(tmp_path/'bad',dict(a,arm=toy()[0]['arm']),prepared)


def test_phase_failure_preserves_partial_STOP_and_no_retry(tmp_path):
    a,prepared=toy();prepared['ctx']['left']=prepared['ctx']['left']*np.exp(.02j)
    c=dict(source_manifest_sha256='SYNTHETIC_PHASE_FAILURE');call=caller();out=tmp_path/'run01'
    def work(task,path):
        return process.run_samples(c,folder(path/'batch'),call,a,policy(),prepared,indices=range(4))[0]
    with pytest.raises(process.NativeWorkerError):relocated.run_fresh(out,[dict(id='toy')],work)
    assert call.stop.is_set()
    stop=json.loads((out/'STOP.json').read_text())
    assert stop['failed'] and stop['mandatory_stop'] and stop['retry']==stop['resume']==0
    assert (out/'failure.json').exists() and list((out/'batch'/'samples').glob('*events.json'))
    assert not list((out/'batch'/'samples').glob('*metrics.json'))
    reports=[json.loads(x) for x in (out/'batch'/'worker_samples.jsonl').read_text().splitlines()]
    assert any(x['failure'] and x['failure']['kind']=='raw_operator_gate' for x in reports)
    with pytest.raises(FileExistsError):relocated.run_fresh(out,[dict(id='toy')],work)


def test_production_native_keeps_exact32_and_calibration(tmp_path,monkeypatch):
    a,prepared=toy();call=caller();out=tmp_path/'native';out.mkdir()
    plan=perf.m4.integrated.resource_plan
    # Four workers are sufficient for this tiny regression fixture. Production
    # uses the available-resource plan, exercised uncapped by the benchmark.
    monkeypatch.setattr(perf.m4.integrated,'resource_plan',lambda **kw:dict(plan(**kw),workers=4))
    rows,cal=perf.native(tmp_path,dict(source_manifest_sha256='SYNTHETIC32'),out,call,a,
        sampling=policy(),prepared=prepared,backend='process')
    assert len(rows)==32 and cal['cost_support'] is None
    assert call.counts['trajectory_sample']==call.counts['kernel_metrics']==32
    timing=json.loads((out/'native_timing.json').read_text())
    assert timing['samples']==32 and timing['full_input_output_dimension']==32
    assert timing['actual_inverse_circuits_independently_evaluated']
    assert len(list((out/'circuits').glob('*.qpy')))==128
    assert sorted(json.loads(f.read_text())['sample'] for f in (out/'samples').glob('*events.json'))==list(range(32))
    with pytest.raises(ValueError,match='BACKEND'):
        perf.native(tmp_path,{},tmp_path,call,a,backend='unknown')


def test_process_failure_before_metrics_and_unique_indices(tmp_path):
    a,prepared=toy();c=dict(source_manifest_sha256='SYNTHETIC_OUTPUT_COLLISION')
    out=folder(tmp_path/'duplicate');call=caller()
    with pytest.raises(ValueError,match='INDICES'):
        process.run_samples(c,out,call,a,policy(),prepared,indices=[0,0])
    assert not (out/'shared_context').exists()
    # A previously existing raw output must be preserved, never overwritten.
    protected=out/'circuits'/'sample00_raw.qpy';protected.write_bytes(b'protected')
    with pytest.raises(process.NativeWorkerError):
        process.run_samples(c,out,call,a,policy(),prepared,indices=[0])
    assert protected.read_bytes()==b'protected' and call.stop.is_set()
    assert not (out/'samples'/'sample00_metrics.json').exists()
