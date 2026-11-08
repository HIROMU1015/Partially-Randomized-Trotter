"""Artificial rows/mock native only. Real saved data: byte/JSON preflight only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_cdf0_entry_independent_price as m

METRICS=('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')


@pytest.fixture(autouse=True)
def unused_output_for_metadata_tests(monkeypatch):
    """Model an unused destination without touching a completed science run.

    The freshness rejection case below explicitly models an existing directory.
    """
    fresh=ROOT/m.io.read(ROOT/m.PROPOSAL)['fresh_output'];exists=Path.exists
    monkeypatch.setattr(Path,'exists',lambda p:False if p==fresh else exists(p))


def rows(arm='CDF0',digest='artificial-source'):
    return [dict(arm=arm,batch=b,sample=i,correctness='PASS',
        metrics={k:20+i%8 for k in METRICS},event_metadata=dict(sample=i,SOURCE=digest,
        JU=m.candidate_identity(arm),sampling=m.sampling(arm,b),events=[{}]*m.ARMS[arm]['r'],
        old_cohort_pooled=False,quantum_shots=0,coherent_occurrences_independent=True))
        for b in range(4) for i in range(32)]


def test_passive_import_and_permission_before_numerical_imports():
    code="""import sys
sys.path.insert(0,'src')
from trottertracks import c_cdf0_entry_independent_price as m
try:m.execute('.')
except PermissionError:pass
else:raise AssertionError('science ran')
try:m.preflight('.')
except FileExistsError:pass  # Committed STOP makes the real destination used.
assert not any(k in sys.modules for k in ('numpy','scipy','qiskit','ffsim'))
"""
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def test_real_metadata_preflight_no_arrays_or_science():
    c,status=m.preflight(ROOT)
    assert status['scientific_calls']==status['arrays_decoded']==0
    assert status['missing_inputs']==[]
    assert status['execution_ready'] and not status['execution_permission']
    assert m.io.sha(ROOT/c['native_inputs']['ENTRY']['ground'])==m.ENTRY_GROUND_SHA
    assert status['new_sample_keys']==256 and status['old_seed_keys_checked']==1024


def test_missing_saved_ground_still_blocks_before_science(monkeypatch):
    c=m.io.read(ROOT/m.PROPOSAL);exists=Path.exists
    missing=ROOT/c['native_inputs']['ENTRY']['ground']
    monkeypatch.setattr(Path,'exists',lambda p:False if p==missing else exists(p))
    _,status=m.preflight(ROOT)
    assert status['missing_inputs']==[dict(path=c['native_inputs']['ENTRY']['ground'],expected_sha256=m.ENTRY_GROUND_SHA)]
    assert not status['execution_ready'] and not status['execution_permission']
    with pytest.raises(FileNotFoundError,match='MISSING_SAVED_INPUTS'):m.execute(ROOT,execute_science=True)


@pytest.mark.parametrize('fault',['candidate','r','relocation','N','pool','sampling','W','shell',
    'compiler','target','physical','old_seed','old_stop','scope','budget','retry','resume','permission'])
def test_preflight_fails_closed_on_design_drift(monkeypatch,fault):
    c=copy.deepcopy(m.io.read(ROOT/m.PROPOSAL));original=m.io.read
    if fault=='candidate':c['arms']['CDF0']['id']=c['arms']['ENTRY']['id']
    elif fault=='r':c['arms']['ENTRY']['r']=64
    elif fault=='relocation':c['relocation']='old'
    elif fault=='N':c['statistics']['samples']=160
    elif fault=='pool':c['old_samples_pooled']=True
    elif fault=='sampling':c['sampling']['ENTRY'][0]['cohort']='old:selection'
    elif fault=='W':c['affine']['CDF0']['W']*=64
    elif fault=='shell':c['affine']['ENTRY']['shell']['rz_count']=0
    elif fault=='compiler':c['compiler']['seed_transpiler']=18
    elif fault=='target':c['target']['alpha_total']=.05
    elif fault=='physical':c['physical_conditions']['rank']=7
    elif fault=='old_seed':c['old_seed_specs']=[]
    elif fault=='old_stop':c['old_STOP']=[]
    elif fault=='scope':c['PR_policy']['K']=1
    elif fault=='budget':c['maximum_scientific_calls']+=1
    elif fault=='retry':c['retry']=1
    elif fault=='resume':c['resume']=1
    else:c['execution_permission']=True
    monkeypatch.setattr(m.io,'read',lambda p:c if Path(p)==ROOT/m.PROPOSAL else original(p))
    with pytest.raises(ValueError):m.preflight(ROOT)


@pytest.mark.parametrize('fault',['SOURCE','input','fresh'])
def test_source_input_and_fresh_path(monkeypatch,fault):
    c=m.io.read(ROOT/m.PROPOSAL);original=m.io.sha
    if fault=='fresh':
        exists=Path.exists
        monkeypatch.setattr(Path,'exists',lambda p:True if p==ROOT/c['fresh_output'] else exists(p))
        error=FileExistsError
    else:
        target=c['SOURCE'] if fault=='SOURCE' else c['native_inputs']['CDF0']['candidate']
        monkeypatch.setattr(m.io,'sha',lambda p:'changed' if Path(p)==ROOT/target else original(p));error=ValueError
    with pytest.raises(error):m.preflight(ROOT)


def test_new_seed_keys_are_disjoint_without_draws():
    c=m.io.read(ROOT/m.PROPOSAL);manifest=m.seed_manifest(c['SOURCE_sha256'],c['old_seed_specs'])
    assert len({x['seed'] for x in manifest['rows']})==256 and manifest['PRNG_draws']==0
    collision=[dict(policy=m.sampling('CDF0',0),SOURCE=c['SOURCE_sha256'],
        candidate=m.candidate_identity('CDF0'),role='baseline')]
    with pytest.raises(ValueError,match='COLLISION'):m.seed_manifest(c['SOURCE_sha256'],collision)


@pytest.mark.parametrize('fault',['candidate','r','SOURCE','old_selection','pool','duplicate','short','long',
    'failure','zero_quantum_flag','coherent','nonfinite','negative'])
def test_row_binding_no_pool_no_exclusion(fault):
    values=rows();x=values[0];meta=x['event_metadata']
    if fault=='candidate':meta['JU']=m.candidate_identity('ENTRY')
    elif fault=='r':meta['events']=meta['events'][:16]
    elif fault=='SOURCE':meta['SOURCE']='historical-source'
    elif fault=='old_selection':meta['sampling']['cohort']='old:selection'
    elif fault=='pool':meta['old_cohort_pooled']=True
    elif fault=='duplicate':values[1]['sample']=0
    elif fault=='short':values.pop()
    elif fault=='long':values.append(copy.deepcopy(x))
    elif fault=='failure':x['correctness']='FAIL'
    elif fault=='zero_quantum_flag':meta['quantum_shots']=1
    elif fault=='coherent':meta['coherent_occurrences_independent']=False
    elif fault=='nonfinite':x['metrics']['rz_count']=float('nan')
    else:x['metrics']['rz_count']=-1
    with pytest.raises((ValueError,TypeError)):m.validate_new_rows(values,'CDF0','artificial-source')


def packet(arm):
    return dict(arm=NS(model=NS(source_identity=m.candidate_identity(arm),input_identity='artificial-input'),
        plan=NS(ld=2)),config=NS(rte_steps=m.ARMS[arm]['r']))


def fake_native(root,c,out,call,a,*,sampling,prepared,backend):
    assert 'handoff' not in c and backend=='process'
    (out/'samples').mkdir()
    arm='CDF0' if sampling['events_per_sample']==64 else 'ENTRY'
    b=int(sampling['cohort'][-2:]);values=rows(arm,c['source_manifest_sha256'])[b*32:(b+1)*32]
    for i,x in enumerate(values):
        for kind in ('trajectory_sample','kernel_build','trajectory_reference','raw_operator_gate','compile',
                     'compiled_operator_gate','adjoint_pair_gate','XY_gate','kernel_metrics'):
            call(kind,str(i),lambda:None)
        m.io.save(out/'samples'/f'sample{i:02d}_events.json',x['event_metadata'])
        m.io.save(out/'samples'/f'sample{i:02d}_XY.json',dict(status='PASS'))
    metrics=[x['metrics'] for x in values]
    from trottertracks.c_pr_aware_df_stage_b2 import calibration
    return metrics,call('calibration_summary','artificial32',lambda:calibration(metrics))


def artificial_contract():
    return dict(SOURCE_sha256='artificial-source',input_identity='artificial-input',shells={},
        affine={a:dict(W=2 if a=='CDF0' else 3,shell={k:7 if a=='CDF0' else 11 for k in METRICS}) for a in m.ARMS})


def test_four_batches_existing_affine_welch_single_weight_and_shell(tmp_path):
    from trottertracks import c_near_r32_statistics as stats
    from trottertracks.c1_stage3_integrated import Calls
    c=artificial_contract();calls=Calls(tmp_path,m.call_limits(),sum(m.call_limits().values()));collected={}
    for arm in m.ARMS:
        out=tmp_path/arm;out.mkdir()
        def call(k,l,f,arm=arm):return calls(arm+'.'+k,l,f)
        collected[arm]=m.collect_arm(tmp_path,c,out,call,arm,packet(arm),{},native_fn=fake_native)
        summary=collected[arm][2]['lanes']['ideal_exact_native']
        assert summary['totals']['rz_count']==c['affine'][arm]['W']*23.5+c['affine'][arm]['shell']['rz_count']
        assert summary['RZ_PR_D_nominal'] is None
        for k in ('compile','raw_operator_gate','compiled_operator_gate','adjoint_pair_gate','XY_gate'):
            assert calls.counts[arm+'.'+k]==128
    result=stats.primary(collected['CDF0'][0],collected['ENTRY'][0],c['affine']['CDF0'],c['affine']['ENTRY'],
        calibrations=[collected[a][1] for a in m.ARMS],expected_n=m.N,statistics=m.STATISTICS)
    assert result['samples_per_arm']==128 and result['confidence']==.95 and result['ratio_interval'] is None
    assert result['total_points']==dict(PR=54.,ENTRY=81.5)
    assert result['interval_kind']=='APPROXIMATE_FIXED_N_IID_MODEL_NOT_VALIDATED'
    assert result['decision']=='NOMINAL_MODEL_PR_LOWER'
    assert set(result['diagnostics'])=={'PR','ENTRY'}


def test_correctness_exception_preserves_partial_and_stops(tmp_path):
    from trottertracks import c_h4_relocated_downstream as bridge
    count=[]
    def fail(root,c,out,call,a,**kwargs):
        count.append(1);raise ValueError('ARTIFICIAL_RAW_REFERENCE_FAILURE')
    def work(task,out):
        m.io.save(out/'partial.json',dict(preserved=True))
        return m.collect_arm(tmp_path,artificial_contract(),out,lambda k,l,f:f(),'CDF0',packet('CDF0'),{},native_fn=fail)
    out=tmp_path/'fresh'
    with pytest.raises(ValueError,match='RAW_REFERENCE'):bridge.run_fresh(out,[dict(id='comparison')],work)
    assert len(count)==1 and (out/'partial.json').exists() and not (out/'comparison.json').exists()
    assert m.io.read(out/'STOP.json')['failed'] and m.io.read(out/'STOP.json')['mandatory_stop']
    with pytest.raises(FileExistsError):bridge.run_fresh(out,[dict(id='comparison')],work)


def test_no_ground_solve_and_native_identity_guard(tmp_path):
    c=artificial_contract();c['native_inputs']={'ENTRY':dict(ground='missing.npz')}
    with pytest.raises(FileNotFoundError,match='NO_SOLVE'):
        m._prepare_arm(tmp_path,c,tmp_path,lambda k,l,f:f(),'ENTRY',None,None,{},None)
    a=packet('ENTRY')
    with pytest.raises(ValueError,match='NATIVE_CANDIDATE'):
        m.collect_arm(tmp_path,c,tmp_path,lambda k,l,f:f(),'CDF0',a,{},native_fn=fake_native)


@pytest.mark.parametrize('arm',['CDF0','ENTRY'])
def test_artificial_rank8_saved_ground_preparation_and_callback_budget(tmp_path,monkeypatch,arm):
    import numpy as np
    from test_research_c_pr_aware_df_stage_b import artificial
    from trottertracks import c_h4_relocated_downstream as bridge
    from trottertracks import c_pr_aware_df_stage_b_runner as loader
    from trottertracks.c1_molecule_extension import SectorCAR
    from trottertracks.c1_stage1 import car_bilinears
    from trottertracks.c1_stage3_integrated import Calls
    _,kappa,_,J,U=artificial(4,8)
    model=bridge.prepare(kappa,.31,J,U,role='baseline',source_identity=m.ARMS[arm]['sha256'],input_identity='artificial-input',ld=2)
    before,_=bridge.raw_generators(model,ctx=SectorCAR(4))
    np.savez(tmp_path/'candidate.npz',J=J,U=U)
    np.savez(tmp_path/'reference.npz',J=J,U=U,generators=np.asarray(before),H=sum(before))
    np.savez(tmp_path/'ground.npz',energy=0.,sector_state=np.eye(36)[:,0])
    physical=dict(candidate_sha256=m.ARMS[arm]['sha256'],input_sha256='artificial-input',
        actual_rank=8,budget_PASS=True,budget_Ha=.00019875,kappa='original_g_corrected',scalar='nuclear_once',
        guard_policy='sector_operation_scale_guard8_proposal_v1',sector_error=0.,numerical_guard=1e-12)
    m.io.save(tmp_path/'physical.json',physical);m.io.save(tmp_path/'shell.json',{})
    # Preparation wiring only: the invented vector is not a physical ground claim.
    monkeypatch.setattr(loader,'ground_gate',lambda *a,**k:dict(status='PASS',eligible=True))
    monkeypatch.setattr(loader,'ground',lambda *a,**k:pytest.fail('fallback solve'))
    c=artificial_contract();c.update(shell_reports='shell.json',native_inputs={arm:dict(candidate='candidate.npz',reference='reference.npz',ground='ground.npz',physical_record='physical.json',physical=physical,candidate_sha256=m.ARMS[arm]['sha256'])})
    out=tmp_path/'prepared';out.mkdir();calls=Calls(out,m.call_limits(),sum(m.call_limits().values()))
    def call(k,l,f):return calls(arm+'.'+k,l,f)
    a,prepared=m._prepare_arm(tmp_path,c,out,call,arm,kappa,.31,{},car_bilinears(8))
    assert a['restored_ground_only'] and a['config'].rte_steps==m.ARMS[arm]['r']
    assert prepared['arm'] is a['arm'] and a['arm'].model.source_identity==m.candidate_identity(arm)
    expected={k:v for k,v in m.call_limits().items() if k.startswith(arm+'.') and k.split('.')[1] not in (
        'calibration_summary','final_calibration','new_rows_validation','arm_summary','trajectory_sample','kernel_build',
        'trajectory_reference','raw_operator_gate','compile','compiled_operator_gate','adjoint_pair_gate','XY_gate','kernel_metrics')}
    assert dict(calls.counts)==expected


def test_planning_is_summary_only_not_an_interval():
    invented={a:dict(origin='published_scalar_summary',W=2,kernel_variance=16,saved_total_point=100) for a in m.ARMS}
    p=m.planning_table(invented)
    assert p['actual_interval'] is p['actual_test'] is None and not p['old_verdict_changed']
    assert all(not r['precision_achievement_guaranteed'] for r in p['rows'])
    invented['CDF0']['origin']='raw_rows'
    with pytest.raises(ValueError,match='SUMMARY_ONLY'):m.planning_table(invented)
