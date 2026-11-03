"""Invented prices, artificial matrices and mock pipelines; no H4 raw decode."""
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_r32_price_statistics as m


def rows(values):
    return [{k: int(x) for k in m.ret.m4.METRICS} for x in values]


def affine(D=10000, W=10, S=0):
    return dict(W=W, shell={k:S for k in m.ret.m4.METRICS},
        D={k:D for k in m.ret.m4.METRICS})


def test_passive_import_scope_and_batch_policy():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_r32_price_statistics as m;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'));assert m.MAX_CALLS==5845"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    assert m.N == 640 and m.BATCHES == 20
    assert len({m.sampling(b)['cohort'] for b in range(20)}) == 20
    assert sum(m.LIMITS.values()) == 5845
    for i in (-1,20,True):
        with pytest.raises(ValueError):m.sampling(i)


def test_seed_identity_reproducibility_nonoverlap_and_collision(monkeypatch):
    old=dict(source_manifest_sha256='old-r',integrated_source_manifest_sha256='old-128')
    a=m.seed_manifest('new-source',old)
    assert a == m.seed_manifest('new-source',old)
    assert len(a['rows'])==640 and len({x['seed'] for x in a['rows']})==640
    assert [x['global_id'] for x in a['rows']]==list(range(640))
    assert a['old_keys_checked']==128 and a['PRNG_draws']==0
    assert a['rows']!=m.seed_manifest('different-source',old)['rows']
    monkeypatch.setattr(m.ret.perf.b.connection,'trajectory_seed',lambda *a:1)
    with pytest.raises(ValueError,match='COLLISION'):m.seed_manifest('new',old)


@pytest.mark.parametrize('mean,D,decision',[(900,10000,'NOMINAL_MODEL_PR_LOWER'),
    (1100,10000,'NOMINAL_MODEL_PR_HIGHER'),(1000,10000,'UNRESOLVED_SIGN_NOT_EQUIVALENCE')])
def test_student_t_affine_difference_ratio_and_selection_scope(mean,D,decision):
    from scipy.stats import t
    data=rows([mean-10,mean+10]*320);a=affine(D)
    got=m.price_statistics(data,a)
    assert got['decision']==decision
    se=np.std([x['rz_count'] for x in data],ddof=1)/math.sqrt(640)
    half=t.ppf(.975,639)*10*se
    np.testing.assert_allclose(got['total_interval'],[10*mean-half,10*mean+half],rtol=1e-14)
    assert got['total_SE']==pytest.approx(10*se)
    assert got['difference_interval']==[x-D for x in got['total_interval']]
    assert got['ratio_interval']==[x/D for x in got['total_interval']]
    assert got['equivalence'] is None and got['noninferiority'] is None
    assert got['guaranteed_resource_superiority'] is None and not got['old_pooled']


@pytest.mark.parametrize('kind', ['precision','skew','constant'])
def test_unqualified_diagnostics_keep_every_observation(kind):
    values=([100,1900]*320 if kind=='precision' else
        [1000]*639+[1000000] if kind=='skew' else [1000]*640)
    got=m.price_statistics(rows(values),affine())
    assert got['decision']=={'precision':'UNRESOLVED_PRECISION_TARGET',
        'skew':'UNRESOLVED_DISTRIBUTION_DIAGNOSTIC','constant':'UNRESOLVED_ZERO_OBSERVED_VARIANCE'}[kind]
    assert got['samples']==640 and got['actual_CA_achieved'] is None
    if kind=='constant':assert got['total_interval'] is None
    else:assert got['diagnostics']['observations_removed']==0


@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),-1,1.5,True])
def test_bad_metrics_fail_without_trimming(invalid):
    data=rows([1000]*640);data[0]['rz_count']=invalid
    with pytest.raises(ValueError,match='INVALID_NATIVE_METRIC'):m.price_statistics(data,affine())
    with pytest.raises(ValueError,match='INCOMPLETE'):m.price_statistics(data[:-1],affine())


def test_serial_binding_reuses_saved_plans_and_shell_once():
    metric=m.ret.m4.METRICS;target='invented'
    plans=[dict(m=i,shots=3,target_id=target) for i in range(13)]
    W=sum(6*(1<<i) for i in range(13))
    lane=dict(rounds=plans,energy=dict(same_accuracy=True),total_shots=78,
        weighted_SE=dict(kernel_reuse_weight=W))
    packet=dict(lanes={x:deepcopy(lane) for x in ('ideal_exact_native','assumed_raw_signal_budget')})
    old=dict(shell_reports={label:dict(metrics={k:n for k in metric}) for label,n in [('head',2),('X',3),('Y',4)]},
        D={lane:dict(cost=dict(totals={k:10000 for k in metric})) for lane in ('ideal_exact_native','numerical')})
    got=m.affine_bindings(packet,old)
    assert got['ideal_exact_native']['W']==W
    # 13 rounds*3 shots*[head+X + head+Y]; no r, cohort or sqrt(13) multiplier.
    assert got['ideal_exact_native']['shell']['rz_count']==13*3*(5+6)
    cal=m.ret.perf.b2.calibration(rows([90,110]*320))
    s=m.secondary_summary(cal,got)
    assert s['ideal_exact_native']['total_SE']['rz_count']==W*cal['sample_mean_standard_error']['rz_count']
    assert 'interval' not in s['ideal_exact_native']


def fixture_contract():
    return dict(source_manifest='source.json',source_manifest_sha256='new',
        old_seed_policy=dict(source_manifest_sha256='old',integrated_source_manifest_sha256='old128'))


@pytest.mark.parametrize('mode',['success','precision','correctness'])
def test_full_holdout_mock_dispatch_no_peeking_no_old_pool_and_STOP(tmp_path,monkeypatch,mode):
    c=fixture_contract(); m.io.save(tmp_path/'source.json',dict(files={}))
    monkeypatch.setattr(m,'preflight',lambda *a:c)
    monkeypatch.setattr(m,'saved_gate',lambda *a:({},{}))
    monkeypatch.setattr(m,'affine_bindings',lambda *a:{lane:affine() for lane in ('ideal_exact_native','assumed_raw_signal_budget')})
    calls=[]
    def native(root,c,out,call,a,*,sampling,prepared):
        calls.append(sampling['cohort']);prepared['same_arm']=True
        if mode=='correctness':
            return call('compile','synthetic correctness FAIL',lambda:(_ for _ in ()).throw(ValueError('COMPILE_FAIL')))
        return rows(([900,1100] if mode=='precision' else [990,1010])*16),None
    actual=m.execute
    def execute(root,c,out,call):
        return actual(root,c,out,call,prepare_function=lambda *a:{},native_function=native)
    monkeypatch.setattr(m,'execute',execute)
    if mode=='correctness':
        with pytest.raises(ValueError,match='COMPILE_FAIL'):m.run(tmp_path,tmp_path/'contract')
        assert len(calls)==1
    else:
        result=m.run(tmp_path,tmp_path/'contract')
        assert len(calls)==20 and result['primary']['samples']==640
        assert result['primary']['decision']==('UNRESOLVED_PRECISION_TARGET' if mode=='precision' else 'UNRESOLVED_SIGN_NOT_EQUIVALENCE')
        assert not result['old_samples_pooled'] and result['quantum_shots_drawn']==0
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json')
    assert stop['failed']==(mode=='correctness') and stop['retry']==stop['resume']==0 and stop['mandatory_stop']


def test_failure_preserves_source_partial_STOP_and_fresh_rule(tmp_path,monkeypatch):
    c=fixture_contract();m.io.save(tmp_path/'source.json',dict(files={}))
    def pre(root,path):
        if (root/m.FRESH).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
        return c
    monkeypatch.setattr(m,'preflight',pre)
    monkeypatch.setattr(m,'saved_gate',lambda *a:(_ for _ in ()).throw(ValueError('INPUT_IDENTITY')))
    with pytest.raises(ValueError,match='INPUT_IDENTITY'):m.run(tmp_path,'unused')
    assert m.io.read(tmp_path/m.FRESH/'STOP.json')['scientific_units']==1
    assert (tmp_path/m.FRESH/'SOURCE.json').exists() and (tmp_path/m.FRESH/'failure.json').exists()
    with pytest.raises(ValueError,match='FRESH_NO_RETRY_RESUME'):m.run(tmp_path,'unused')


def test_real_native_dispatch_reuses_existing_artificial_fixture(monkeypatch,tmp_path):
    from test_research_c_rte_r_retuning import test_custom_native32_dispatch_and_shared_context
    # Fully mocked QPY/Operator, invented arrays. Exercises unchanged real
    # native function and same-arm cache with two distinct sampling policies.
    test_custom_native32_dispatch_and_shared_context(tmp_path,monkeypatch)


def test_real_native_uses_prospective_policy(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from test_research_c_pr_integrated_native import test_fixed32_native_dispatch_and_failure
    original=m.ret.perf.native
    def wrapped(root,c,out,call,a):
        occurrence=m.ret.perf.b.independent_occurrence
        monkeypatch.setattr(m.ret.perf.b,'independent_occurrence',lambda *x,**k:occurrence(*x,**k)[:32])
        a['config']=SimpleNamespace(rte_steps=32)
        return original(root,c,out,call,a,sampling=m.sampling(0),prepared={})
    monkeypatch.setattr(m.ret.perf,'native',wrapped)
    test_fixed32_native_dispatch_and_failure(tmp_path,monkeypatch,False)
    for i in range(32):
        v=m.io.read(tmp_path/f'out/samples/sample{i:02d}_events.json')
        assert v['sampling']==m.sampling(0) and v['old_cohort_pooled'] is False


def test_passive_preflight_never_decodes_arrays_scope_drift_and_freshness(tmp_path,monkeypatch):
    from test_research_c_rte_r_retuning import passive_fixture
    c=passive_fixture(tmp_path)
    c.update(base_commit=m.BASE,scope=m.SCOPE,implementation=m.VERSION,r=32,
        statistics=deepcopy(m.STATISTICS),call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        sampling=[m.sampling(i) for i in range(20)],fresh_output=m.FRESH,new_PF_RTE=0,new_shots=0,
        r32_result='opaque.npz',retuning_stop='opaque.npz',retuning_contract='old.json',
        old_seed_policy=dict(source_manifest_sha256='old-r',integrated_source_manifest_sha256='old128'))
    m.io.save(tmp_path/'old.json',dict(source_manifest_sha256='old-r'))
    m.io.save(tmp_path/'parent.json',dict(source_manifest_sha256='old128'))
    c['records']['parent_contract']='parent.json'
    c['saved_inputs'].update({p:m.io.sha(tmp_path/p) for p in ('old.json','parent.json')})
    digest=c['saved_inputs']['opaque.npz']
    monkeypatch.setattr(m.ret,'CANDIDATE',digest);monkeypatch.setattr(m.ret,'INPUT',digest)
    monkeypatch.setattr(m,'saved_gate',lambda *a:None)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive array decode'))
    m.io.save(tmp_path/'c.json',c);m.preflight(tmp_path,tmp_path/'c.json')
    bad=deepcopy(c);bad['statistics']['confidence']=.9;m.io.save(tmp_path/'bad.json',bad)
    with pytest.raises(ValueError,match='FIXED_PRICE_SCOPE'):m.preflight(tmp_path,tmp_path/'bad.json')
    (tmp_path/m.FRESH).mkdir(parents=True)
    with pytest.raises(ValueError,match='FRESH'):m.preflight(tmp_path,tmp_path/'c.json')
    (tmp_path/'opaque.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_INPUT'):m.preflight(tmp_path,tmp_path/'c.json')


def test_saved_scalar_gate_rejects_lost_qualification_without_raw(tmp_path,monkeypatch):
    from test_research_c_rte_r_retuning import scalar_saved
    old=scalar_saved();old['D']['ideal_exact_native']['cost']['totals']['rz_count']=105532782860
    monkeypatch.setattr(m.ret,'evidence',lambda *a:old)
    plans=[dict(m=i,target_id=f'retuning:{m.ret.CANDIDATE}:r32') for i in range(13)]
    lane=dict(energy=dict(same_accuracy=True),rounds=plans,total_shots=15202,
        weighted_SE=dict(kernel_reuse_weight=13979232),
        conditional_totals=dict(totals=dict(rz_count=106752458375)))
    v=dict(r=32,status='NATIVE_AND_CONDITIONAL_COST_COMPLETE',native_admitted=True,
        calibration=dict(samples=32,sample_mean=dict(rz_count=7636.5)),rounds=[dict(m=i) for i in range(13)],
        feasibility=dict(status='POTENTIALLY_FEASIBLE'),lanes={x:deepcopy(lane) for x in ('ideal_exact_native','assumed_raw_signal_budget')})
    v['lanes']['evidence']=dict(conditional_totals=dict(totals=None))
    m.io.save(tmp_path/'r32.json',v);m.io.save(tmp_path/'STOP.json',dict(failed=False,scientific_units=1265,retry=0,resume=0))
    c=dict(r32_result='r32.json',retuning_stop='STOP.json')
    m.saved_gate(tmp_path,c)
    v['native_admitted']=False
    (tmp_path/'r32.json').write_text(__import__('json').dumps(v))
    with pytest.raises(ValueError,match='SAVED_R32_QUALIFICATION'):m.saved_gate(tmp_path,c)
