"""Metadata/mock handoff regressions; no science, NPZ or transpiler calls."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

from trottertracks.resource_applicability import h8_fast_handoff_contract_v1 as c
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.h48_preflight_contract_v1 import task_caps


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def signals():
    return {'signals':[dict(cell=x,signals={'corrected':{'real':1.,'imag':0.}}) for x in c.plan()['cells']],
            'calls_attempted':{'primitive':1164}}


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();(root/'fixture.py').write_text('artificial source\n')
    parent=tmp_path/'parent';parent.mkdir();input_dir=tmp_path/'input';input_dir.mkdir()
    identity={'directory':str(input_dir),'fixed_input':'artificial metadata only'}
    profile={'fixed_environment':'artificial'}
    for name in c.PENDING_NAMES:write(parent/name,{})
    write(parent/'STOP_PIPELINE',{'user_intent':'retire pending continuation only'})
    term={'status':'PREFLIGHT_STOP','reason':'ValueError:USER_STOP_PIPELINE','retry':False}
    write(parent/'PIPELINE_STOP.json',term)
    write(parent/'handoff_retired_old_continuation.json',dict(pipeline_terminal=term,
        heavy_stopped=False,remaining_old_run_started=False,new_science_calls=0,
        retired_pipeline_pid=9000000,start_ticks=17))
    hashes={'fixture.py':hashlib.sha256((root/'fixture.py').read_bytes()).hexdigest()}
    common=dict(source_commit=c.OLD_SOURCE,plan=c.plan(),input_identity=identity,profile=profile,
        source_hashes=hashes,base_manifest={'plan':{'caps_proposed':{'output_bytes':2*2**30,'log_bytes':2*2**20}}},
        cumulative_carry={'H6_known_compile':1716,'H6_reserved_compile':20,'refund':0,'reset':False})
    write(parent/'signal01/manifest.json',dict(common,scope='SIGNAL'))
    write(parent/'signal01/execution_terminal.json',dict(first_stop_reason=None,completed_indices=[0],
        all_dispatched_reaped=True,cleanup_errors=[]))
    write(parent/'signal01/cost_exploration_0000/worker_terminal.json',dict(status='COMPLETE',scope='SIGNAL',source_commit=c.OLD_SOURCE))
    write(parent/'signal01/cost_exploration_0000/signal_summary.json',signals())
    task=c.old_registrations(c.OLD_SOURCE,'HEAVY',signals())[0]
    write(parent/'heavy01/manifest.json',dict(common,scope='HEAVY',registrations=[task]))
    co=dict(pid=9000001,start_ticks=23)
    write(parent/'launch-heavy01.json',dict(identity=co))
    write(parent/'heavy01/cost_exploration_0003/worker_claim.json',dict(identity={'pid':9000002,'start_ticks':24}))
    monkeypatch.setattr(c,'source_paths',lambda root:['fixture.py'])
    monkeypatch.setattr(c,'profile',lambda:profile)
    monkeypatch.setattr(c,'input_identity',lambda path:identity)
    monkeypatch.setattr(c,'inactive',lambda identity:True)
    return root,parent,tmp_path/'newoutput',task


def make_heavy_complete(parent,reg):
    d=parent/'heavy01';w=d/'cost_exploration_0003';task=reg['task']
    t=dict(first_stop_reason=None,completed_indices=[3],all_dispatched_reaped=True,
        cleanup_errors=[],retired=[dict(index=3,exit_code=0,reaped=True)])
    write(d/'execution_terminal.json',t)
    write(w/'worker_terminal.json',dict(status='COMPLETE',reason=None,source_commit=c.OLD_SOURCE,scope='HEAVY'))
    rows=[]
    for axis in ('cosine','sine'):
        row=dict(cell_id=task['cell_id'],phase=task['phase'],replica=0,seed=task['seed'],
            axis=axis,control='symmetric_directional',compiler_hash=c.COMPILER,fingerprint='a'*64,
            event_digest=digest(None),metrics={k:1 for k in ('RZ','CX','RZ_depth','CX_depth','depth','size')})
        write(w/f'wrapper_symmetric_directional_{axis}.json',row);rows.append(row)
    calls=dict(compile=2,trajectory=0,occurrence=0,control_probe=4,primitive=0,reference_matvec=0)
    write(w/'cost_result.json',dict(task=task,rows=rows,calls_attempted=calls))
    write(w/'trajectory.json',dict(task=task,events=None,event_digest=digest(None)))
    write(w/'validation_symmetric_directional.json',dict(max_error=1e-13))


def test_pending_heavy_never_admitted(fixture):
    _,parent,_,_=fixture
    assert c.admit_heavy(parent) is None


def test_original_seeds_and_exact_eighteen_calls(fixture):
    root,parent,output,_=fixture
    a=c.pending_template(root,parent,'b'*40,output)
    b=c.pending_template(root,parent,'c'*40,output)
    assert a['registrations']==b['registrations']
    assert [r['index'] for r in a['registrations']]==[0,1,2,4,5,6,7]
    caps=[task_caps(r['task'],a['plan']['caps_proposed']) for r in a['registrations']]
    for key,value in [('compile',18),('trajectory',4),('occurrence',15),('control_probe',36)]:
        assert sum(x[key] for x in caps)==value
    assert a['seed_source_commit']==c.OLD_SOURCE
    assert a['proposed_resources']['cost_workers']==3
    assert a['handoff_budget']['total_compile']==20


def test_final_requires_completed_heavy(fixture):
    root,parent,output,_=fixture;m=c.pending_template(root,parent,'b'*40,output)
    with pytest.raises(ValueError,match='HEAVY_COMPLETION_REQUIRED'):c.finalized(m,None)


def test_complete_heavy_receipt_and_native_exit_unknown_preserved(fixture):
    root,parent,output,reg=fixture;m=c.pending_template(root,parent,'b'*40,output)
    make_heavy_complete(parent,reg);h=c.admit_heavy(parent)
    assert h['calls_attempted']['compile']==2 and h['worker_exit_code']==0
    assert h['coordinator_exit_code'] is None and h['exact_exit_timestamp'] is None
    m,g=c.finalized(m,h)
    assert c.verify_binding(root,m,g,worker=True)
    assert m['technical_plan_sealed'] is True and m['H8_full_campaign_authorized'] is False


@pytest.mark.parametrize('fault',['stop','nonzero_exit','missing_reap','cleanup','worker_stop','wrong_source','wrong_seed','missing_sine','metric_negative','compiler','call_count','validation'])
def test_incomplete_or_invalid_heavy_never_authorizes_remaining(fixture,fault):
    _,parent,_,reg=fixture;make_heavy_complete(parent,reg);d=parent/'heavy01';w=d/'cost_exploration_0003'
    if fault in ('stop','nonzero_exit','missing_reap','cleanup'):
        r=c.read(d/'execution_terminal.json')
        if fault=='stop':r['first_stop_reason']='MEMORY_PRESSURE'
        elif fault=='nonzero_exit':r['retired'][0]['exit_code']=1
        elif fault=='missing_reap':r['all_dispatched_reaped']=False
        else:r['cleanup_errors']=['artificial cleanup failure']
        write(d/'execution_terminal.json',r)
    elif fault in ('worker_stop','wrong_source'):
        r=c.read(w/'worker_terminal.json')
        if fault=='worker_stop':r['status']='STOP';r['reason']='artificial'
        else:r['source_commit']='d'*40
        write(w/'worker_terminal.json',r)
    elif fault=='validation':write(w/'validation_symmetric_directional.json',{'max_error':float('nan')})
    else:
        r=c.read(w/'cost_result.json')
        if fault=='wrong_seed':r['rows'][0]['seed']+=1
        elif fault=='missing_sine':r['rows'].pop()
        elif fault=='metric_negative':r['rows'][0]['metrics']['RZ']=-1
        elif fault=='compiler':r['rows'][0]['compiler_hash']='x'*64
        else:r['calls_attempted']['compile']=1
        write(w/'cost_result.json',r)
    with pytest.raises(ValueError):c.admit_heavy(parent)


def test_parent_still_live_waits_without_dispatch(fixture,monkeypatch):
    _,parent,_,reg=fixture;make_heavy_complete(parent,reg)
    monkeypatch.setattr(c,'inactive',lambda identity:False)
    assert c.admit_heavy(parent) is None


@pytest.mark.parametrize('change',['seed','tasks','plan','workers','budget','carry','method','full_grant'])
def test_template_mutations_fail_closed(fixture,change):
    root,parent,output,_=fixture;m=c.pending_template(root,parent,'b'*40,output)
    if change=='seed':m['registrations'][0]['task']['seed']+=1
    elif change=='tasks':m['registrations'].pop()
    elif change=='plan':m['plan']['compiler_proposed']['optimization_level']=3
    elif change=='workers':m['proposed_resources']['cost_workers']=8
    elif change=='budget':m['handoff_budget']['total_compile']=21
    elif change=='carry':m['cumulative_carry']['reset']=True
    elif change=='method':m['validation_method']='old independent wrappers'
    else:m['H8_full_campaign_authorized']=True
    with pytest.raises(ValueError):c.verify_template(root,m,worker=True)


def test_changed_parent_and_current_old_source_are_rejected(fixture):
    root,parent,output,_=fixture;m=c.pending_template(root,parent,'b'*40,output)
    write(parent/'signal01/cost_exploration_0000/signal_summary.json',{'changed':'metadata'})
    with pytest.raises(ValueError,match='PENDING_PARENT_CHANGED'):c.verify_template(root,m,worker=True)


def test_old_remaining_never_duplicates(fixture):
    root,parent,output,_=fixture;(parent/'remaining01').mkdir()
    with pytest.raises(ValueError,match='OLD_REMAINING_ALREADY'):c.pending_template(root,parent,'b'*40,output)


def test_active_old_continuation_rejects(fixture,monkeypatch):
    root,parent,output,_=fixture;monkeypatch.setattr(c,'inactive',lambda identity:False)
    with pytest.raises(ValueError,match='OLD_CONTINUATION_STILL_ACTIVE'):c.pending_template(root,parent,'b'*40,output)


def test_used_output_and_lock_reject(fixture):
    root,parent,output,_=fixture;output.mkdir()
    with pytest.raises(ValueError,match='OUTPUT_ALREADY_USED'):c.pending_template(root,parent,'b'*40,output)
    output.rmdir();output.with_name(output.name+'.one_shot_lock').write_text('used')
    with pytest.raises(ValueError,match='OUTPUT_ALREADY_USED'):c.pending_template(root,parent,'b'*40,output)


def test_completed_parent_changed_after_seal_rejects(fixture):
    root,parent,output,reg=fixture;m=c.pending_template(root,parent,'b'*40,output);make_heavy_complete(parent,reg)
    m,g=c.finalized(m,c.admit_heavy(parent))
    write(parent/'heavy01/cost_exploration_0003/validation_symmetric_directional.json',{'max_error':2e-13})
    with pytest.raises(ValueError,match='HEAVY_RECEIPT_CHANGED'):c.verify_binding(root,m,g,worker=True)


@pytest.mark.parametrize('key',['approved_by_user','retry','H8_full_campaign_authorized','manifest_sha256','source_commit','output'])
def test_new_grant_is_required_and_bound(fixture,key):
    root,parent,output,reg=fixture;m=c.pending_template(root,parent,'b'*40,output);make_heavy_complete(parent,reg)
    m,g=c.finalized(m,c.admit_heavy(parent))
    g[key]=not g[key] if type(g[key]) is bool else 'changed'
    with pytest.raises(ValueError,match='NEW_AUTH_BINDING'):c.verify_authorization(m,g)


def load_script(name):
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('artificial_handoff_script',root/'scripts/resource_applicability'/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_conditional_authorization_cannot_expand_scope(fixture):
    root,parent,output,_=fixture;m=c.pending_template(root,parent,'b'*40,output)
    arm=load_script('arm_track_a_h8_fast_handoff_v1.py')
    g=dict(schema='h8_conditional_handoff_user_authorization_v1',approved_by_user=True,
        authorization_basis=c.AUTHORIZATION,template_sha256=digest(m),source_commit=m['source_commit'],
        output=m['output'],one_shot=True,retry=False,H8_full_campaign_authorized=False,
        old_heavy_must_not_be_interrupted=True,remaining_cases=7,remaining_compile=18,total_compile=20)
    arm.validate_conditional(m,g)
    for key,value in [('remaining_cases',8),('remaining_compile',20),('total_compile',22),('retry',True),('old_heavy_must_not_be_interrupted',False)]:
        bad=deepcopy(g);bad[key]=value
        with pytest.raises(ValueError):arm.validate_conditional(m,bad)


def test_combined_twenty_metrics_and_counter_without_recompile(fixture,monkeypatch):
    root,parent,output,reg=fixture;m=c.pending_template(root,parent,'b'*40,output);make_heavy_complete(parent,reg)
    m,g=c.finalized(m,c.admit_heavy(parent));run=load_script('run_track_a_h8_fast_handoff_v1.py')
    all_new={}
    for registration in m['registrations']:
        task=registration['task'];caps=task_caps(task,m['plan']['caps_proposed']);policies=['symmetric_directional','ordinary'] if task['ordinary'] else ['symmetric_directional']
        rows=[dict(cell_id=task['cell_id'],control=p,axis=a) for p in policies for a in ('cosine','sine')]
        all_new[task['cell_id']]=dict(rows=rows,calls_attempted={k:caps[k] for k in ('compile','trajectory','occurrence','control_probe','primitive','reference_matvec')})
    monkeypatch.setattr(run,'admit',lambda directory,task,m:all_new[task['cell_id']])
    class Writer:
        def write(self,name,value,**kw):self.name=name;self.value=value
    writer=Writer();run.combined_result(output,m,{'completed':'artificial'},writer)
    assert writer.value['total_wrappers']==20 and writer.value['new_wrappers']==18
    assert writer.value['calls_attempted']==dict(compile=20,trajectory=4,occurrence=15,control_probe=40,primitive=0,reference_matvec=0)
    assert writer.value['old_heavy_source']==c.OLD_SOURCE
    assert writer.value['source_and_method_provenance_distinguished'] is True


def test_instruction_failure_records_count_without_compiling():
    run=load_script('run_track_a_h8_fast_handoff_v1.py')
    ns={}
    fake="def compile_owned():\n    from types import SimpleNamespace\n    compiled=SimpleNamespace(data=[0]*7)\n    raise RuntimeError('TRANSPILED_INSTRUCTION_CAP')\n"
    exec(compile(fake,'/artificial/resource_applicability/h6_matched_memory_v3.py','exec'),ns)
    try:ns['compile_owned']()
    except RuntimeError as error:assert run.compiled_count_from_failure(error)==7


def test_instruction_failure_missing_context_stays_unknown():
    run=load_script('run_track_a_h8_fast_handoff_v1.py')
    try:raise RuntimeError('TRANSPILED_INSTRUCTION_CAP')
    except RuntimeError as error:assert run.compiled_count_from_failure(error) is None
