"""Metadata/mock retry budget/gate checks; no science or actual compilation."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import pytest
from test_h8_fast_handoff_v1 import fixture,make_heavy_complete,write
from trottertracks.resource_applicability import h8_fast_retry_contract_v1 as r
from trottertracks.resource_applicability import h8_fast_handoff_contract_v1 as c
from trottertracks.resource_applicability.h48_preflight_contract_v1 import task_caps


@pytest.fixture
def failed(fixture,monkeypatch):
    root,parent,output,reg=fixture;make_heavy_complete(parent,reg);w=parent/'heavy01/cost_exploration_0003'
    for p in w.glob('wrapper_*.json'):p.unlink()
    (w/'cost_result.json').unlink()
    t=c.read(parent/'heavy01/execution_terminal.json');t.update(first_stop_reason='WORKER_EXIT:1',completed_indices=[])
    t['retired'][0]['exit_code']=1;write(parent/'heavy01/execution_terminal.json',t)
    write(w/'worker_terminal.json',dict(status='STOP',reason='RuntimeError:TRANSPILED_INSTRUCTION_CAP',scope='HEAVY',source_commit=c.OLD_SOURCE))
    write(w/'failure_diagnostics.json',dict(message='TRANSPILED_INSTRUCTION_CAP'))
    calls=dict(compile=1,control_probe=4,trajectory=0,occurrence=0,primitive=0,reference_matvec=0)
    write(w/'compile_attempt_01.json',dict(task=reg['task'],axis='cosine',calls_attempted_before_fingerprint=calls))
    monkeypatch.setattr(r,'source_paths',lambda root:['fixture.py'])
    return root,parent,output,reg


def test_only_approved_three_caps_change():
    old=c.plan();new=r.plan();delta={k:(old['caps_proposed'][k],v) for k,v in new['caps_proposed'].items() if old['caps_proposed'][k]!=v}
    assert delta==dict(transpiled_instructions=(5000000,10000000),compile=(20,21),control_probe=(40,44))
    old['caps_proposed']=new['caps_proposed'];assert old==new


def test_failed_actual_is_preserved_without_refund(failed):
    _,parent,_,_=failed;f=r.failed_receipt(parent)
    assert f['known_calls']['compile']==1 and f['known_calls']['control_probe']==4
    assert f['actual_transpiled_instruction_count'] is None and f['refund']==0 and f['reset'] is False


def test_heavy_manifest_old_seed_new_cap_and_same_resources(failed):
    root,parent,out,reg=failed;m,g=r.make_manifest(root,parent,'b'*40,out,'HEAVY_RETRY_FAST')
    assert m['registrations']==[reg] and m['seed_source_commit']==c.OLD_SOURCE
    assert m['plan']['caps_proposed']['transpiled_instructions']==10000000
    assert m['proposed_resources']==c.resources('HEAVY',1)
    assert m['accounting']['total_compile']==21 and m['accounting']['total_control_probe']==44
    assert r.verify_binding(root,m,g,worker=True)


def test_remaining_requires_successful_retry_not_old_failure(failed):
    root,parent,out,_=failed
    with pytest.raises(ValueError,match='FAST_HEAVY_RECEIPT_REQUIRED'):
        r.make_manifest(root,parent,'b'*40,out,'REMAINING_AFTER_FAST_RETRY')
    with pytest.raises(ValueError,match='SUCCESSFUL_RETRY_SOURCE_OR_SCOPE'):
        r.make_manifest(root,parent,'b'*40,out,'REMAINING_AFTER_FAST_RETRY',parent/'heavy01')


@pytest.mark.parametrize('field',['compile','control_probe','transpiled_instructions','seed','compiler','CPU','profile','reset','automatic_retry','full_H8'])
def test_mutated_retry_binding_fails_closed(failed,field):
    root,parent,out,_=failed;m,g=r.make_manifest(root,parent,'b'*40,out,'HEAVY_RETRY_FAST')
    if field in ('compile','control_probe','transpiled_instructions'):m['plan']['caps_proposed'][field]+=1
    elif field=='seed':m['registrations'][0]['task']['seed']+=1
    elif field=='compiler':m['compiler_hash']='changed'
    elif field=='CPU':m['proposed_resources']['cost_cpu_sets']=[[0]]
    elif field=='profile':m['profile']={'changed':True}
    elif field=='reset':m['accounting']['reset']=True
    elif field=='automatic_retry':m['automatic_retry']=True
    else:m['H8_full_campaign_authorized']=True
    g['manifest_sha256']=r.digest(m)
    with pytest.raises(ValueError):r.verify_binding(root,m,g,worker=True)


@pytest.mark.parametrize('change',['counter','extra_attempt','partial_wrapper','native_exit','still_active'])
def test_failed_parent_cannot_be_silently_reset(failed,change,monkeypatch):
    _,parent,_,_=failed;w=parent/'heavy01/cost_exploration_0003'
    if change=='counter':
        d=c.read(w/'compile_attempt_01.json');d['calls_attempted_before_fingerprint']['compile']=0;write(w/'compile_attempt_01.json',d)
    elif change=='extra_attempt':write(w/'compile_attempt_02.json',{})
    elif change=='partial_wrapper':write(w/'wrapper_symmetric_directional_cosine.json',{})
    elif change=='native_exit':
        d=c.read(parent/'heavy01/execution_terminal.json');d['retired'][0]['exit_code']=0;write(parent/'heavy01/execution_terminal.json',d)
    else:monkeypatch.setattr(c,'inactive',lambda identity:False)
    with pytest.raises(ValueError):r.failed_receipt(parent)


def load_script(name):
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('artificial_retry_script',root/'scripts/resource_applicability'/name)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_combined_has_twenty_saved_wrappers_twentyone_actual(failed,monkeypatch):
    root,parent,out,_=failed;m,g=r.make_manifest(root,parent,'b'*40,out,'HEAVY_RETRY_FAST')
    signals=c.read(parent/'signal01/cost_exploration_0000/signal_summary.json')
    heavy_reg=c.old_registrations(c.OLD_SOURCE,'HEAVY',signals)[0]
    def result(task):
        caps=task_caps(task,r.plan()['caps_proposed']);policies=['symmetric_directional','ordinary'] if task['ordinary'] else ['symmetric_directional']
        rows=[dict(cell_id=task['cell_id'],axis=axis,control=policy) for policy in policies for axis in ('cosine','sine')]
        return dict(rows=rows,calls_attempted={k:caps[k] for k in ('compile','control_probe','trajectory','occurrence','primitive','reference_matvec')})
    m['accepted_retry_heavy']={'result':result(heavy_reg['task'])}
    m['registrations']=c.remaining_registrations(signals)
    module=load_script('run_track_a_h8_fast_retry_v1.py')
    class Writer:
        def write(self,name,value,**kw):self.value=value
    writer=Writer();module.combined(out,m,{},writer,lambda directory,task,m:result(task))
    assert writer.value['total_wrappers']==20
    assert writer.value['cumulative_calls']==dict(compile=21,control_probe=44,trajectory=4,occurrence=15,primitive=0,reference_matvec=0)
    assert writer.value['preserved_failed_calls']['compile']==1


def test_legacy_worker_cloning_uses_new_validator_without_global_mutation():
    module=load_script('run_track_a_h8_fast_retry_v1.py');legacy=module.legacy_metadata()
    old=legacy['worker'];validator=old.__globals__['verify_binding']
    copy=module.bound_function(old,{'verify_binding':r.verify_binding})
    assert copy.__globals__['verify_binding'] is r.verify_binding
    assert old.__globals__['verify_binding'] is validator


def test_task_caps_still_two_plus_eighteen_and_four_trajectories():
    signal={'signals':[dict(cell=x,signals={'corrected':{'real':1.,'imag':0.}}) for x in c.plan()['cells']]}
    rows=c.old_registrations(c.OLD_SOURCE,'HEAVY',signal)
    rows+=c.remaining_registrations(signal)
    caps=[task_caps(x['task'],r.plan()['caps_proposed']) for x in rows]
    assert sum(x['compile'] for x in caps)==20 and sum(x['control_probe'] for x in caps)==40
    assert sum(x['trajectory'] for x in caps)==4 and sum(x['occurrence'] for x in caps)==15


def test_pipeline_failed_explicit_retry_never_starts_remaining(tmp_path,monkeypatch):
    module=load_script('continue_track_a_h8_fast_retry_v1.py')
    manifest=tmp_path/'m.json';grant=tmp_path/'g.json';control=tmp_path/'control';out=tmp_path/'out'
    m=dict(scope='HEAVY_RETRY_FAST',source_commit='b'*40,output=str(out),old_parent={'directory':str(tmp_path/'old')},
           proposed_resources={'assigned_cpus':[30,31]})
    write(manifest,m);write(grant,{})
    monkeypatch.setattr(module,'verify_binding',lambda *a,**k:True)
    monkeypatch.setattr(module.resource,'setrlimit',lambda *a:None)
    affinity=[]
    monkeypatch.setattr(module.os,'sched_setaffinity',lambda pid,cpus:affinity.append((pid,cpus)))
    monkeypatch.setattr(module,'process_identity',lambda pid:dict(pid=pid,start_ticks=123))
    class Child:
        pid=9999999
        def wait(self):return 1
    starts=[]
    def spawn(*a,**k):
        k['preexec_fn']();starts.append(a[0]);return Child()
    monkeypatch.setattr(module.subprocess,'Popen',spawn)
    def forbidden(*a,**k):raise AssertionError('NO_REMAINING_DISPATCH_AFTER_FAILED_RETRY')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    monkeypatch.setattr(module.sys,'argv',['fixture','--heavy-manifest',str(manifest),'--heavy-grant',str(grant),
        '--manifest-sha256',module.file_hash(manifest),'--grant-sha256',module.file_hash(grant),
        '--start-heavy','--control',str(control),'--output',str(tmp_path/'remaining')])
    assert module.main()==1 and len(starts)==1
    assert affinity==[(0,{31}),(0,{30,31})]
    assert read_json(control/'heavy_native_exit.json')['exit_code']==1
    assert 'EXPLICIT_HEAVY_RETRY_EXIT' in read_json(control/'terminal.json')['reason']
    assert not (control/'remaining_auth').exists()


def read_json(path):
    return json.loads(path.read_text())
