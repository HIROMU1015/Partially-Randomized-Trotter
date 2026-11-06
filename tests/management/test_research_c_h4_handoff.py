"""Artificial-only continuation: identical cohorts, no completed sample rerun."""
from copy import deepcopy
from pathlib import Path
import json,os,signal,subprocess,sys,time
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_h4_relocated_handoff as h,c_native_process as process
from trottertracks import c_h4_relocated_downstream as bridge
from test_research_c_native_process import toy,caller,policy
from test_research_c_h4_relocated_downstream import synthetic_packet


def contract(old,new):
    return dict(source_manifest_sha256='ORIGINAL_SEED_SOURCE',
        handoff=dict(prior_output=str(old.resolve()),new_output=str(new.resolve())))


@pytest.fixture
def saved32(tmp_path,monkeypatch):
    a,prepared=toy();old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    folder=old/'batch';folder.mkdir();p=policy();c=dict(source_manifest_sha256='ORIGINAL_SEED_SOURCE')
    original=process.native(tmp_path,c,folder,caller(),a,sampling=p,prepared=prepared)
    return a,prepared,old,new,p,original


def test_all32_reused_no_numeric_tasks(saved32,tmp_path,monkeypatch):
    a,prepared,old,new,p,original=saved32
    monkeypatch.setattr(process,'run_samples',lambda *a,**k:pytest.fail('COMPLETED_SAMPLES_RERUN'))
    out=new/'batch';out.mkdir()
    rows,cal=process.native(tmp_path,contract(old,new),out,caller(),a,sampling=p,prepared=prepared)
    assert (rows,cal)==original
    assert json.loads((out/'handoff_samples.json').read_text())['computed']==0
    for src in (old/'batch'/'samples').glob('*.json'):
        assert src.read_bytes()==(out/'samples'/src.name).read_bytes()


def test_only_missing_indices_execute_and_same_seed(saved32,tmp_path):
    a,prepared,old,new,p,original=saved32
    for i in range(5,32):(old/'batch'/'samples'/f'sample{i:02d}_metrics.json').unlink()
    out=new/'batch';out.mkdir();call=caller()
    rows,cal=process.native(tmp_path,contract(old,new),out,call,a,sampling=p,prepared=prepared)
    assert (rows,cal)==original and call.counts['trajectory_sample']==27
    record=json.loads((out/'handoff_samples.json').read_text())
    assert record['reused']==5 and record['pending_indices']==list(range(5,32))
    for i in range(32):
        src=old/'batch'/'samples'/f'sample{i:02d}_events.json'
        assert src.read_bytes()==(out/'samples'/src.name).read_bytes()


def test_domain_mismatch_and_missing_verified_artifact_STOP(saved32):
    a,prepared,old,new,p,_=saved32
    bad=dict(p,cohort='DIFFERENT_COHORT')
    with pytest.raises(ValueError,match='DOMAIN'):
        h.completed_samples(old/'batch',bad,a['arm'].model.source_identity,'ORIGINAL_SEED_SOURCE')
    (old/'batch'/'circuits'/'sample00_raw_adjoint.qpy').unlink()
    with pytest.raises(ValueError,match='INCOMPLETE_VERIFIED'):
        h.completed_samples(old/'batch',p,a['arm'].model.source_identity,'ORIGINAL_SEED_SOURCE')


def test_reference13_and_ground_restore_without_eigensolve(tmp_path,monkeypatch):
    from trottertracks import c_pr_aware_df_stage_b_runner as loader
    old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    rel,a,call,calls=synthetic_packet(old)
    out=old/'r32';ar,result,packet,cache=bridge.reference_r(a,32,out/'reference',call)
    (out/'reference_result.json').write_text(json.dumps(result))
    _,G=bridge.raw_generators(rel,ctx=a['ctx'])
    def forbidden(*args,**kw):pytest.fail('SAVED_GROUND_RECOMPUTED')
    monkeypatch.setattr(loader,'ground',forbidden)
    df=a['saved']['physical'];saved=dict(J=a['d']['J'],U=a['d']['U'],generators=np.asarray(bridge.raw_generators(rel,ctx=a['ctx'])[0]),H=a['sector']['H'])
    r2,a2,report=bridge.prepare_candidate(rel.kappa,.31,a['d']['J'],a['d']['U'],saved,df,
        candidate='invented:baseline',input_identity='invented-input',role='baseline',out=new/'A',call=call,
        shell_reports={},saved_ground=old/'A'/'ground_candidate.npz')
    np.testing.assert_array_equal(a2['ground'],a['ground'])
    before=calls.count('exact_evolution')
    rr,restored,p2,c2=h.restore_reference(contract(old,new),a2,32,new/'r32'/'reference',call,None)
    assert restored==result and p2==packet
    np.testing.assert_array_equal(c2['pf'],cache['pf'])
    assert calls.count('exact_evolution')==before
    assert c2['pf_errors']==cache['pf_errors']
    target=new/'D';target.mkdir()
    with pytest.raises(ValueError,match='FULL_EIGENPAIR'):
        bridge.strong_d(tmp_path,{},target,call,a2,G)


def test_complete_cohort_bypasses_accounting_and_D_binding(saved32,tmp_path):
    a,prepared,old,new,p,(rows,cal)=saved32
    # Use the actual adapter cohort namespace for the saved sample records.
    key='invented';r=a['config'].rte_steps;sp=h.sampling_policy(key,'selection',r,0)
    cohort=old/'cohort';cohort.mkdir();(old/'batch').rename(cohort/'batch00')
    for f in (cohort/'batch00'/'samples').glob('*events.json'):
        value=json.loads(f.read_text());value['sampling']=sp;f.write_text(json.dumps(value))
    result=dict(candidate=a['arm'].model.source_identity,r=r,split='selection',samples=32,native_admitted=True,calibration=cal)
    (cohort/'result.json').write_text(json.dumps(result))
    value,packet=h.restore_cohort(contract(old,new),new/'cohort',a,result,{},split='selection',key=key)
    assert value==result and packet['kernel_samples']==rows
    assert h.restore_d(contract(old,new),new/'absent',a) is None


def test_SIGINT_drains_existing_thread_cohort_and_preserves_STOP(tmp_path):
    script=tmp_path/'drain.py'
    script.write_text('''from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,time
sys.path.insert(0,sys.argv[1])
from trottertracks.c_h4_relocated_downstream import run_fresh
root=Path(sys.argv[2])
def task(i):
 time.sleep(.3);(root/str(i)).write_text('completed')
def work(task_record,out):
 with ThreadPoolExecutor(max_workers=8) as pool:
  fs=[pool.submit(task,i) for i in range(8)]
  (root/'READY').write_text('ready')
  for f in as_completed(fs):f.result()
 return {}
run_fresh(root/'run',[dict(id='toy')],work)
''')
    child=subprocess.Popen([sys.executable,'-B',str(script),str(ROOT/'src'),str(tmp_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    deadline=time.monotonic()+5
    while not (tmp_path/'READY').exists() and time.monotonic()<deadline:time.sleep(.01)
    assert (tmp_path/'READY').exists()
    child.send_signal(signal.SIGINT);child.communicate(timeout=5)
    assert child.returncode!=0 and all((tmp_path/str(i)).exists() for i in range(8))
    stop=json.loads((tmp_path/'run'/'STOP.json').read_text())
    assert stop['failed'] and stop['reason']=='KeyboardInterrupt()' and stop['mandatory_stop']


def test_checkpoint_rejects_scientific_change_bad_STOP_and_changed_bytes(tmp_path):
    from trottertracks.c1_stage1_execution import sha
    oldroot=tmp_path/'oldroot';oldroot.mkdir();out=oldroot/'results';out.mkdir()
    (oldroot/'source.py').write_text('invented_source')
    (oldroot/'input.txt').write_text('invented_input')
    (oldroot/'SOURCE.json').write_text(json.dumps(dict(files={'source.py':sha(oldroot/'source.py')})))
    seed=sha(oldroot/'SOURCE.json')
    parent=dict(SOURCE='SOURCE.json',SOURCE_sha256=seed,inputs={'input.txt':sha(oldroot/'input.txt')},r_grid=[8,16,32,64,128])
    (oldroot/'parent.json').write_text(json.dumps(parent))
    (out/'STOP.json').write_text(json.dumps(dict(failed=True,reason='KeyboardInterrupt()',mandatory_stop=True)))
    completion=oldroot/'artifacts/research_c_h4_relocated_downstream/2026-11-06/execution/completion.json'
    completion.parent.mkdir(parents=True);completion.write_text(json.dumps(dict(failed='KeyboardInterrupt()')))
    batch=out/'batch';batch.mkdir()
    for i in range(32):
        for p in h.sample_paths(i):
            (batch/p).parent.mkdir(exist_ok=True);(batch/p).write_text('invented')
    snap=dict(version=h.VERSION,prior_output=str(out),sampling_SOURCE_sha256=seed,interrupted_batch='batch',
        completion_sha256=sha(completion),files={'STOP.json':sha(out/'STOP.json')})
    checkpoint=tmp_path/'checkpoint.json';checkpoint.write_text(json.dumps(snap))
    termination=tmp_path/'termination.json';termination.write_text(json.dumps(dict(prior_output=str(out),process_gone=True,mode='SIGINT_DRAIN')))
    c=dict(parent,sampling_SOURCE_sha256=seed,handoff=dict(termination_record='termination.json',termination_sha256=sha(termination),checkpoint='checkpoint.json',checkpoint_sha256=sha(checkpoint),
        prior_output=str(out),prior_root=str(oldroot),prior_contract='parent.json',prior_contract_sha256=sha(oldroot/'parent.json')))
    h.validate_checkpoint(tmp_path,c)
    with pytest.raises(ValueError,match='SCIENTIFIC_CONDITIONS'):
        h.validate_checkpoint(tmp_path,dict(c,r_grid=[32]))
    (out/'STOP.json').write_text(json.dumps(dict(failed=True,reason='ValueError(scientific mismatch)',mandatory_stop=True)))
    with pytest.raises(ValueError,match='PLANNED_HANDOFF'):
        h.validate_checkpoint(tmp_path,c)
    (out/'STOP.json').write_text(json.dumps(dict(failed=True,reason='KeyboardInterrupt()',mandatory_stop=True)))
    (batch/h.sample_paths(0)[-1]).unlink()
    with pytest.raises(ValueError,match='DID_NOT_DRAIN'):
        h.validate_checkpoint(tmp_path,c)
    # A forced planned interruption keeps the absent legacy STOP absent.
    (out/'STOP.json').unlink();completion.unlink()
    snap['completion_sha256']=None;snap['files']={}
    checkpoint.write_text(json.dumps(snap));c['handoff']['checkpoint_sha256']=sha(checkpoint)
    termination.write_text(json.dumps(dict(prior_output=str(out),process_gone=True,mode='SIGTERM_AFTER_SIGINT',signal=15,
        original_STOP_exists=False,original_completion_exists=False,exit_code=143)))
    c['handoff']['termination_sha256']=sha(termination)
    for p in (batch/'samples').glob('*metrics.json'):p.unlink()
    h.validate_checkpoint(tmp_path,c)
    termination.write_text(json.dumps(dict(prior_output=str(out),process_gone=False)))
    c['handoff']['termination_sha256']=sha(termination)
    with pytest.raises(ValueError,match='NOT_STOPPED'):h.validate_checkpoint(tmp_path,c)
