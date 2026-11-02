"""Invented prices/arrays and mocks; no saved H4 raw prices/arrays are opened."""
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_update064_r32_price_statistics as m


def rows(values):
    return [{k:int(v) for k in m.ret.m4.METRICS} for v in values]


def affine(D=10000,W=10,S=0):
    return dict(W=W,shell={k:S for k in m.ret.m4.METRICS},D={k:D for k in m.ret.m4.METRICS})


def test_passive_import_fixed_design_and_reuse():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_update064_r32_price_statistics as m;assert m.N==256 and m.MAX_CALLS==2374;assert not any(k in sys.modules for k in ('numpy','scipy','qiskit','ffsim'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    assert m.BATCHES==8 and m.STATISTICS['precision_halfwidth_fraction_of_D']==.01
    assert sum(m.A_LIMITS.values())==20
    assert m.shared.N==640 and m.shared.STATISTICS['precision_halfwidth_fraction_of_D']==.005
    for b in [-1,8,True]:
        with pytest.raises(ValueError):m.sampling(b)


def test_seed_no_draws_new_candidate_and_800_prior_keys(monkeypatch):
    monkeypatch.setattr(np.random,'PCG64',lambda *a:pytest.fail('no PRNG draw during seed preflight'))
    got=m.seed_manifest('invented-source',m.OLD_SEEDS)
    assert got==m.seed_manifest('invented-source',m.OLD_SEEDS)
    assert len(got['rows'])==len({r['seed'] for r in got['rows']})==256
    assert got['old_keys_checked']==800 and got['PRNG_draws']==0
    assert [x['global_id'] for x in got['rows']]==list(range(256))
    assert got['rows']!=m.seed_manifest('changed-source',m.OLD_SEEDS)['rows']
    monkeypatch.setattr(m.ret.perf.b.connection,'trajectory_seed',lambda *a:1)
    with pytest.raises(ValueError,match='COLLISION'):m.seed_manifest('invented-source',m.OLD_SEEDS)


def test_old_or_duplicate_cohort_rejected(monkeypatch):
    policy=m.sampling(0)
    monkeypatch.setattr(m,'sampling',lambda b:dict(policy,cohort=m.shared.sampling(0)['cohort']))
    with pytest.raises(ValueError,match='COHORT_ID_OVERLAP'):m.seed_manifest('new',m.OLD_SEEDS)


@pytest.mark.parametrize('center,expected',[(900,'NOMINAL_MODEL_PR_LOWER'),(1100,'NOMINAL_MODEL_PR_HIGHER'),(1000,'UNRESOLVED_SIGN_NOT_EQUIVALENCE')])
def test_256_t_interval_affine_one_sample_and_shared_SE(center,expected):
    from scipy.stats import t
    data=rows([center-10,center+10]*128)
    got=m.shared.price_statistics(data,affine(),expected_n=m.N,statistics=m.STATISTICS)
    se=np.std([r['rz_count'] for r in data],ddof=1)/math.sqrt(256)
    half=t.ppf(.975,255)*10*se
    assert got['decision']==expected and got['total_SE']==pytest.approx(10*se)
    np.testing.assert_allclose(got['difference_interval'],[10*center-half-10000,10*center+half-10000],rtol=1e-12,atol=1e-10)
    assert got['ratio_interval']==[x/10000 for x in got['total_interval']]
    assert got['equivalence'] is got['noninferiority'] is got['guaranteed_resource_superiority'] is None
    assert not got['old_pooled'] and got['samples']==256


@pytest.mark.parametrize('case',['precision','skew','influence','constant'])
def test_unqualified_screens_without_exclusions(case):
    values={'precision':[500,1500]*128,'skew':[1000]*255+[1000000],
        'influence':[1000]*254+[0,2000],'constant':[1000]*256}[case]
    got=m.shared.price_statistics(rows(values),affine(),expected_n=256,statistics=m.STATISTICS)
    expected={'precision':'UNRESOLVED_PRECISION_TARGET','skew':'UNRESOLVED_DISTRIBUTION_DIAGNOSTIC',
        'influence':'UNRESOLVED_DISTRIBUTION_DIAGNOSTIC','constant':'UNRESOLVED_ZERO_OBSERVED_VARIANCE'}[case]
    assert got['decision']==expected
    if case=='influence':
        assert abs(got['diagnostics']['moment_skewness'])<1e-12
        assert got['diagnostics']['influence_warning']
    if case!='constant': assert got['diagnostics']['observations_removed']==0


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1,True,1.25])
def test_invalid_or_missing_native_rows_STOP(bad):
    data=rows([990,1010]*128);data[0]['rz_count']=bad
    with pytest.raises(ValueError):m.shared.price_statistics(data,affine(),expected_n=256,statistics=m.STATISTICS)
    with pytest.raises(ValueError,match='INCOMPLETE'):m.shared.price_statistics(data[:-1],affine(),expected_n=256,statistics=m.STATISTICS)


def saved_fixture(root):
    metrics=m.ret.m4.METRICS
    df=dict(candidate_sha256=m.CANDIDATE,actual_rank=8,budget_PASS=True,budget_Ha=.00019875,
        kappa='original_g_corrected',scalar='nuclear_once',sector_error=1e-9,numerical_guard=1e-9)
    plans=[dict(m=i,target_id='update064:r32:'+m.CANDIDATE,shots=3) for i in range(13)]
    lane=dict(rounds=plans,energy=dict(same_accuracy=True),total_shots=15008,
        weighted_SE=dict(kernel_reuse_weight=13381920),conditional_totals=dict(totals={k:1e11 for k in metrics}))
    pr=dict(candidate=m.CANDIDATE,r=32,native_admitted=True,calibration=dict(samples=32),
        lanes={k:deepcopy(lane) for k in ('ideal_exact_native','assumed_raw_signal_budget')})
    pr['lanes']['evidence']=dict(conditional_totals=dict(totals=None))
    d=dict(candidate=m.CANDIDATE,saved_DF=df,ideal_exact_native=dict(energy=dict(same_accuracy=True),
        cost=dict(shots=13706,totals={k:105038112319 for k in metrics})),numerical={})
    source=root/'saved_SOURCE.json';m.io.save(source,dict(files={}))
    c=dict(input='input.npz',records={k:k+'.json' for k in m.RECORDS})
    c['records']['SOURCE']=source.name
    records=dict(result=dict(status='UPDATE064_PR_AND_SAME_JU_D_COMPLETE',candidate=m.CANDIDATE,PR=pr,D=d),
        STOP=dict(failed=False,scientific_units=617,retry=0,resume=0,mandatory_stop=True,call_counts=m.downstream.LIMITS),
        contract=dict(policy=m.POLICY,target=m.ret.m4.TARGET,compiler=m.ret.COMPILER,sampling=m.downstream.SAMPLING,
            source_manifest_sha256=m.io.sha(source),saved_inputs={'input.npz':m.INPUT},
            peer_paths=dict(candidate='candidate.npz'),peer_inputs={'candidate.npz':m.CANDIDATE}),
        ground_report=dict(status='PASS',alias_margin=1.6,candidate_sha256=m.CANDIDATE,energy_Ha=0.),
        law=dict(status='PASS',atoms=216),shells={})
    for k,x in records.items():m.io.save(root/c['records'][k],x)
    return c,records


@pytest.mark.parametrize('fault',['none','oldJU','Dprice','ground','W','compiler','STOP','law'])
def test_saved_064_binding_and_no_old_raw(tmp_path,monkeypatch,fault):
    c,r=saved_fixture(tmp_path)
    if fault=='oldJU':r['result']['D']['candidate']=m.ret.CANDIDATE
    if fault=='Dprice':r['result']['D']['ideal_exact_native']['cost']['totals']['rz_count']=105532782860
    if fault=='ground':r['ground_report']['candidate_sha256']=m.ret.CANDIDATE
    if fault=='W':r['result']['PR']['lanes']['ideal_exact_native']['weighted_SE']['kernel_reuse_weight']=13979232
    if fault=='compiler':r['contract']['compiler']={}
    if fault=='STOP':r['STOP']['failed']=True
    if fault=='law':r['law']['status']='FAIL'
    for k,x in r.items():(tmp_path/c['records'][k]).write_text(__import__('json').dumps(x))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('saved array/raw decode'))
    if fault=='none':assert m.saved_gate(tmp_path,c)[0]['candidate']==m.CANDIDATE
    else:
        with pytest.raises(ValueError):m.saved_gate(tmp_path,c)


def test_shell_affine_binding_and_wrong_D(tmp_path,monkeypatch):
    from test_research_c_r32_price_statistics import test_serial_binding_reuses_saved_plans_and_shell_once
    test_serial_binding_reuses_saved_plans_and_shell_once()
    bindings={lane:dict(affine(D=105038112319,W=13381920,S=52528),shots=15008)
        for lane in ('ideal_exact_native','assumed_raw_signal_budget')}
    monkeypatch.setattr(m.shared,'affine_bindings',lambda *a:bindings)
    assert m.affine_bindings({}, {})==bindings
    bindings['ideal_exact_native']['D']['rz_count']=105532782860
    with pytest.raises(ValueError,match='AFFINE_BINDING'):m.affine_bindings({}, {})


def passive_fixture(root):
    from trottertracks.c1_h4_native_fair import GAUSSIAN,PHASE
    (root/'opaque.npz').write_bytes(b'invented opaque bytes')
    entry=root/m.ENTRY;entry.parent.mkdir();entry.write_text('# invented entry\n')
    m.io.save(root/'SOURCE.json',dict(files={m.ENTRY:m.io.sha(entry)}))
    h=m.io.sha(root/'opaque.npz')
    c=dict(base_commit=m.BASE,A_technical_commit=m.A_SOURCE,candidate_commit=m.PEER,scope=m.SCOPE,
        implementation=m.VERSION,entry=m.ENTRY,policy=m.POLICY,statistics=m.STATISTICS,fixed_accounting=m.ACCOUNTING,
        sampling=[m.sampling(b) for b in range(m.BATCHES)],target=m.ret.m4.TARGET,compiler=m.ret.COMPILER,
        allowances=m.ret.m3.ALLOWANCES,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,fresh_output=m.FRESH,
        retry=0,resume=0,new_fit=0,optimizer=0,new_PF_RTE=0,new_shots=0,new_ground_eigensolve=0,new_DF_error=0,mandatory_stop=True,
        records={k:'opaque.npz' for k in m.RECORDS},peer_paths={k:'opaque.npz' for k in ('candidate','reference','basis')},
        saved_inputs={'opaque.npz':h},peer_inputs={'opaque.npz':h},input='opaque.npz',ground='opaque.npz',dependency_lock='opaque.npz',
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),packages={},shells={},
        gaussian=GAUSSIAN,compiler_phase=PHASE,old_seed_policy=m.OLD_SEEDS)
    return c


@pytest.mark.parametrize('fault',['none','N','confidence','SOURCE','input','oldseed','peer','fresh','resume'])
def test_passive_drift_fresh_and_no_numeric(tmp_path,monkeypatch,fault):
    c=passive_fixture(tmp_path);h=c['saved_inputs']['opaque.npz']
    monkeypatch.setattr(m,'CANDIDATE',h);monkeypatch.setattr(m,'INPUT',h)
    monkeypatch.setattr(m,'saved_gate',lambda *a:None)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:('OTHER' if fault=='peer' else m.PEER)+'\n')
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive decode'))
    c=deepcopy(c)
    if fault=='N':c['statistics']['samples']=640
    if fault=='confidence':c['statistics']['confidence']=.9
    if fault=='SOURCE':(tmp_path/m.ENTRY).write_text('changed')
    if fault=='input':(tmp_path/'opaque.npz').write_bytes(b'changed')
    if fault=='oldseed':c['old_seed_policy']['A_prospective']='changed'
    if fault=='fresh':(tmp_path/m.FRESH).mkdir(parents=True)
    if fault=='resume':c['resume']=1
    m.io.save(tmp_path/'c.json',c)
    if fault=='none':m.preflight(tmp_path,tmp_path/'c.json',tmp_path)
    else:
        with pytest.raises(ValueError):m.preflight(tmp_path,tmp_path/'c.json',tmp_path)


@pytest.mark.parametrize('mode',['success','precision','correctness','missing'])
def test_mock_fixed256_shared_engine_no_peeking_no_pool_partial_STOP(tmp_path,monkeypatch,mode):
    m.io.save(tmp_path/'source.json',dict(files={}))
    c=dict(source_manifest='source.json',source_manifest_sha256='invented',old_seed_policy=m.OLD_SEEDS)
    def pre(root,path,peer):
        if (root/m.FRESH).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
        return c
    monkeypatch.setattr(m,'preflight',pre);monkeypatch.setattr(m,'saved_gate',lambda *a:({},{}))
    monkeypatch.setattr(m,'affine_bindings',lambda *a:{lane:affine() for lane in ('ideal_exact_native','assumed_raw_signal_budget')})
    monkeypatch.setattr(m,'prepare_native',lambda *a:{})
    batches=[];actual=m.shared.price_statistics
    def primary(data,a,**kw):
        assert len(batches)==8 and len(data)==256
        return actual(data,a,**kw)
    monkeypatch.setattr(m.shared,'price_statistics',primary)
    def native(root,c,out,call,a,*,sampling,prepared):
        batches.append(sampling['cohort']);prepared['origin']='synthetic'
        if mode=='correctness':return call('compile','injected',lambda:(_ for _ in ()).throw(ValueError('NATIVE_FAIL')))
        values=([500,1500] if mode=='precision' else [990,1010])*16
        return rows(values[:-1] if mode=='missing' else values),None
    actual_execute=m.execute
    monkeypatch.setattr(m,'execute',lambda root,peer,c,out,call:actual_execute(root,peer,c,out,call,native_function=native))
    if mode in ('correctness','missing'):
        with pytest.raises(ValueError):m.run(tmp_path,'c',tmp_path,execution_opt_in=True)
    else:
        x=m.run(tmp_path,'c',tmp_path,execution_opt_in=True)
        assert x['candidate']==m.CANDIDATE and x['primary']['samples']==256
        assert x['primary']['decision']==('UNRESOLVED_PRECISION_TARGET' if mode=='precision' else 'UNRESOLVED_SIGN_NOT_EQUIVALENCE')
        assert not x['old_samples_pooled'] and x['quantum_shots_drawn']==0
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json')
    assert stop['failed']==(mode in ('correctness','missing')) and stop['retry']==stop['resume']==0
    if mode=='correctness':assert len(batches)==1
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,'c',tmp_path,execution_opt_in=True)


def test_no_execution_without_opt_in(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'preflight',lambda *a:pytest.fail('opt-in failure must precede all work'))
    with pytest.raises(ValueError,match='OPT_IN_REQUIRED'):m.run(tmp_path,'c',tmp_path)
    assert not (tmp_path/m.FRESH).exists()


def test_native_cohort_dispatch_uses_shared_fixture(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_research_c_pr_integrated_native import test_fixed32_native_dispatch_and_failure
    original=m.ret.perf.native
    def wrapped(root,c,out,call,a):
        occurrence=m.ret.perf.b.independent_occurrence
        monkeypatch.setattr(m.ret.perf.b,'independent_occurrence',lambda *x,**kw:occurrence(*x,**kw)[:32])
        a['config']=SimpleNamespace(rte_steps=32)
        return original(root,c,out,call,a,sampling=m.sampling(0),prepared={})
    monkeypatch.setattr(m.ret.perf,'native',wrapped)
    test_fixed32_native_dispatch_and_failure(tmp_path,monkeypatch,False)
    e=m.io.read(tmp_path/'out/samples/sample00_events.json')
    assert e['sampling']==m.sampling(0) and not e['old_cohort_pooled']


def test_artificial_prepare_reuses_new064_ground_without_eigensolve(tmp_path,monkeypatch):
    from test_research_c_update064_r32_downstream import synthetic
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        c,saved=synthetic(tmp_path,monkeypatch)
        with np.load(tmp_path/'physical.npz') as z:H=z['H']
        values,vectors=np.linalg.eigh(H)
        m.ret.perf.loader.save_arrays(tmp_path/'ground.npz',energy=values[0],sector_state=vectors[:,0])
        ground_report=dict(status='PASS',candidate_sha256=m.CANDIDATE,energy_Ha=float(values[0]),alias_margin=1.6)
        m.io.save(tmp_path/'ground.json',ground_report)
        arm=m.ret.perf.b.prepare_arm(**m.ret.perf.loader.load(tmp_path/'input.npz',('h','g','constant')),
            **m.ret.perf.loader.load(tmp_path/'new.npz',('J','U')),role='proposed',source_identity=m.CANDIDATE,input_identity=m.INPUT,ld=2)
        tail,config,_=m.ret.perf.b.configure(arm,m.ret.Fraction(1,10),32,0)
        from trottertracks.c1_molecule_extension import SectorCAR
        law=m.ret.perf.b.tail_gate(SectorCAR(4),arm,np.asarray(m.ret.perf.loader.load(tmp_path/'physical.npz',('generators',))['generators']),tail)
        m.io.save(tmp_path/'law.json',law)
        c.update(ground='ground.npz',records=dict(ground_report='ground.json',law='law.json'))
        monkeypatch.setattr(m.ret.perf.loader,'ground',lambda *a:pytest.fail('no new ground eigensolve in calibration'))
        out=tmp_path/'out';out.mkdir();call=m.ret.m4.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
        a=m.prepare_native(tmp_path,tmp_path,c,out,call)
    assert dict(call.counts)==m.A_LIMITS and a['arm'].model.source_identity==m.CANDIDATE
    assert a['config'].rte_steps==32 and m.io.read(out/'ground.json')['new_eigensolve']==0
