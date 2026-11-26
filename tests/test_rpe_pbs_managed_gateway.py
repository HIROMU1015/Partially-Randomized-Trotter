"""Ordinary managed reservations: no real PBS submission or scientific work."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rpe_pbs_gateway as G
import rpe_pbs_pilot as P
from build_rpe_pbs_independent_gateway import build


@pytest.fixture
def scheduler(monkeypatch):
    jobs, submissions, deletions = {}, [], []
    original = subprocess.run
    def run(cmd, *args, **kwargs):
        if cmd[0] == '/opt/pbs/bin/qsub':
            directory = Path(cmd[1]).parent
            job = f'{len(submissions)+1}.prism'
            submissions.append(directory)
            jobs[job] = dict(Job_Name='RPE'+directory.name[:12], job_state='Q',
                             Resource_List=G.read(directory/'SPEC.json')['resources'])
            return subprocess.CompletedProcess(cmd, 0, job, '')
        if cmd[0] == '/opt/pbs/bin/qdel':
            deletions.append(cmd[1]); jobs[cmd[1]]['job_state']='F'
            return subprocess.CompletedProcess(cmd,0,'','')
        return original(cmd,*args,**kwargs)
    monkeypatch.setattr(G,'qstat',lambda *ids:{k:v for k,v in jobs.items() if not ids or k in ids})
    monkeypatch.setattr(subprocess,'run',run)
    return SimpleNamespace(jobs=jobs,submissions=submissions,deletions=deletions)


def spec(root, token, host, *, version='v1', resources=None, reason=None):
    return dict(token=token, profile=dict(host_id=host, work_dir=str(root/'workers'/host),
                    pbs=dict(worker_source=version)), resources=resources or dict(queue='workq',ncpus=16,mem='64gb',walltime='12:00:00'),
                python=sys.executable, transport_root=str(root/version), run_id=root.name, reason=reason or {})


def ensure(root, s):
    return G.handle(root,dict(action='ensure',spec=s))


def snapshot(root):
    return G.handle(root,dict(action='snapshot'))['allocations']


def test_standard_gateway_keeps_two_nodes_and_isolates_runs(tmp_path, scheduler):
    root=tmp_path/'run1'; other=tmp_path/'run2'
    a=spec(root,'session-a','pbs-h8-0'); b=spec(root,'session-b','pbs-h8-1')
    first=ensure(root,a); second=ensure(root,b)
    assert first['pilot'] != second['pilot']
    assert ensure(root,a)['existing'] and len(scheduler.submissions)==2
    assert len(scheduler.deletions)==0
    for directory in scheduler.submissions:
        assert '--serve' in (directory/'worker.pbs').read_text()
        assert 'run_rpe_pbs_worker.py' not in (directory/'worker.pbs').read_text()
        assert G.read(directory/'SPEC.json')['run_managed'] is True
    ensure(other,spec(other,'session-c','pbs-h8-0'))
    assert len(scheduler.submissions)==3
    assert len(snapshot(root))==2 and len(snapshot(other))==1
    assert {x['host_id'] for x in snapshot(root)}=={'pbs-h8-0','pbs-h8-1'}


def test_replacement_waits_for_drain_then_reuses_same_job_and_keeps_stop_fence(tmp_path,scheduler):
    root=tmp_path/'run'; host='pbs-h6-0'; a=spec(root,'old',host); old=ensure(root,a)
    gen=scheduler.submissions[0]; scheduler.jobs['1.prism']['job_state']='R'
    d=root/'allocations'/'old'; assert P.claim(P.managed_pool(root,host),gen,'1.prism')==d
    b=spec(root,'new',host,version='v2')
    with pytest.raises(ValueError,match='still owned'):
        ensure(root,b)
    G.handle(root,dict(action='exchange',controls={'old':dict(stop=True)}))
    G.handle(root,dict(action='exchange',controls={'old':dict(stop=False,pause=False)}))
    assert G.read(d/'CONTROL.json')['stop'] is True
    with pytest.raises(ValueError,match='still owned'):
        ensure(root,b)
    G.atomic(d/'STATE.json',dict(state='EXITED',accepting=False)); P.end(d,'WORKER_EXIT',returncode=0)
    new=ensure(root,b)
    assert new['pilot']==old['pilot']; assert len(scheduler.submissions)==1 and not scheduler.deletions
    assert G.read(d/'SPEC.json')==a
    assert G.read(root/'allocations'/'new'/'SPEC.json')==b
    assert P.claim(P.managed_pool(root,host),gen,'1.prism')==root/'allocations'/'new'
    rows={x['token']:x for x in snapshot(root)}
    assert rows['old']['status']=='F' and rows['old']['reservation_status']=='R'
    assert rows['new']['status']=='R' and rows['new']['job_id']=='1.prism'
    assert rows['old']['job_id']==rows['new']['job_id']
    assert not (gen/'RELEASE.json').exists()


def test_explicit_release_does_not_cancel_live_worker_or_use_qdel_on_running_node(tmp_path,scheduler):
    root=tmp_path/'run'; host='pbs-0'; ensure(root,spec(root,'old',host)); gen=scheduler.submissions[0]
    scheduler.jobs['1.prism']['job_state']='R'; d=root/'allocations'/'old'
    P.claim(P.managed_pool(root,host),gen,'1.prism')
    with pytest.raises(ValueError,match='Drain'):
        G.handle(root,dict(action='release_managed_node',host_id=host))
    G.handle(root,dict(action='delete',token='old'))
    assert not (gen/'RELEASE.json').exists() and not scheduler.deletions
    P.end(d,'WORKER_EXIT',returncode=0)
    assert G.handle(root,dict(action='release_managed_node',host_id=host))['release_requested']
    assert (gen/'RELEASE.json').exists() and not scheduler.deletions


def test_unused_queued_session_can_be_cancelled_without_touching_other_host(tmp_path,scheduler):
    root=tmp_path/'run'; ensure(root,spec(root,'a','pbs-h8-0'));ensure(root,spec(root,'b','pbs-h8-1'))
    G.handle(root,dict(action='delete',token='a'))
    assert scheduler.deletions==['1.prism']; assert scheduler.jobs['2.prism']['job_state']=='Q'


def test_legacy_token_is_not_converted_or_resubmitted(tmp_path,scheduler):
    root=tmp_path/'run'; s=spec(root,'legacy','pbs-h8-0');d=root/'allocations'/'legacy'
    G.atomic(d/'SPEC.json',s);G.atomic(d/'QSUB_STARTED.json',dict(epoch=1));G.atomic(d/'QSUB_RETURN.json',dict(returncode=0,stdout='42.prism',stderr=''))
    assert ensure(root,s)['existing']; assert not (d/'PILOT.json').exists();assert not scheduler.submissions
    assert G.read(d/'SPEC.json')==s


def test_keeper_runs_two_artificial_workers_in_one_reservation(tmp_path,scheduler,monkeypatch):
    root=tmp_path/'run'; host='pbs-h8-0'; a=spec(root,'old',host); b=spec(root,'new',host,version='v2')
    # Actual harmless child command under each distinct version root.
    for s in (a,b):
        script=Path(s['transport_root'])/'scripts'/'run_rpe_pbs_worker.py';script.parent.mkdir(parents=True)
        script.write_text('import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument("--profile");a=p.parse_args();d=Path(a.profile).parent\nassert json.loads((d/"ALLOCATION.json").read_text())["job_id"]=="1.prism"\n(d/"ARTIFICIAL_RETURN.json").write_text(json.dumps({"job":"1.prism","version":json.loads(Path(a.profile).read_text())["pbs"]["worker_source"]}))\n(d/"STATE.json").write_text(json.dumps({"state":"EXITED","accepting":False}))\n')
    ensure(root,a);gen=scheduler.submissions[0];scheduler.jobs['1.prism']['job_state']='R'
    actual=dict(job_id='1.prism',ncpus=16,memory_bytes=64*1024**3,walltime_seconds=43200,remaining_seconds=43200)
    monkeypatch.setattr(G,'record_allocation',lambda directory:actual)
    monkeypatch.setattr(P.ctypes,'CDLL',lambda *a,**k:SimpleNamespace(prctl=lambda *a:0))
    waitpid=P.os.waitpid
    monkeypatch.setattr(P.os,'waitpid',lambda pid,*a: (_ for _ in ()).throw(ChildProcessError()) if pid == -1 else waitpid(pid,*a))
    def idle(_):
        if (root/'allocations'/'new'/'SESSION_END.json').exists():
            G.handle(root,dict(action='release_managed_node',host_id=host))
        elif (root/'allocations'/'old'/'SESSION_END.json').exists():
            ensure(root,b)
        else:raise AssertionError('Unexpected idle interval')
    monkeypatch.setattr(P.time,'sleep',idle)
    P.serve(gen)
    assert len(scheduler.submissions)==1 and not scheduler.deletions
    for sess,ver in [('old','v1'),('new','v2')]:
        d=root/'allocations'/sess
        assert G.read(d/'ARTIFICIAL_RETURN.json')==dict(job='1.prism',version=ver)
        assert G.read(d/'SESSION_END.json')['returncode']==0


def test_gateway_packaging_default_and_explicit_shared_mode(tmp_path):
    from zipfile import ZipFile
    standard=tmp_path/'common.pyz';shared=tmp_path/'shared.pyz'
    build(standard);build(shared,shared_pilot=True)
    with ZipFile(standard) as z:
        assert b'def ensure(root, spec)' in z.read('__main__.py')
        assert 'rpe_pbs_pilot.py' in z.namelist()
    with ZipFile(shared) as z:assert b'def reserve_locked' in z.read('__main__.py')
    for path in (standard,shared):
        p=subprocess.run([sys.executable,str(path),'--help'],capture_output=True,text=True)
        assert p.returncode==0 and '--serve' in p.stdout


def test_controller_checks_capability_before_any_ensure(tmp_path):
    from trottertracks.rpe_pbs import Gateway
    script=tmp_path/'fake.py'
    script.write_text('import sys,json\nlog=open(sys.argv[1],"a")\nfor l in sys.stdin:\n a=json.loads(l)["action"];log.write(a+"\\n");log.flush()\n if a=="managed_capabilities" and sys.argv[2]=="old":\n  print(json.dumps({"ok":False,"error":"Unknown action"}),flush=True)\n else:\n  print(json.dumps({"ok":True,"result":{"managed_keeper":True,"worker_session_replacement":True}}),flush=True)\n')
    for mode in ('old','new'):
        log=tmp_path/(mode+'.log')
        g=Gateway(dict(gateway_script=str(script),remote_root=str(tmp_path),
                       gateway_command=[sys.executable,str(script),str(log),mode]))
        try:
            if mode=='old':
                with pytest.raises(ValueError,match='Unknown action'):
                    g.call(dict(action='ensure',spec={}))
                assert log.read_text().splitlines()==['managed_capabilities']
            else:
                g.call(dict(action='ensure',spec={}))
                g.call(dict(action='ensure',spec={}))
                assert log.read_text().splitlines()==['managed_capabilities','ensure','ensure']
        finally:g.close()


def test_incompatible_replacement_does_not_reserve_a_second_node(tmp_path,scheduler):
    root=tmp_path/'run'; host='pbs-h8-0'; ensure(root,spec(root,'old',host)); gen=scheduler.submissions[0]
    scheduler.jobs['1.prism']['job_state']='R'; d=root/'allocations'/'old'
    P.claim(P.managed_pool(root,host),gen,'1.prism'); P.end(d,'WORKER_EXIT',returncode=0)
    more=spec(root,'more',host,resources=dict(queue='workq',ncpus=32,mem='64gb',walltime='12:00:00'))
    with pytest.raises(ValueError,match='cannot satisfy'):
        ensure(root,more)
    assert len(scheduler.submissions)==1 and not scheduler.deletions
