"""Metadata/mocks/minimal owned processes only; no real H6 science inputs."""
from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from trottertracks.resource_applicability import h6_server_contract_v1 as c
from trottertracks.resource_applicability import h6_server_resources_v1 as r
from trottertracks.resource_applicability import h6_server_execution_v1 as e
from trottertracks.resource_applicability.h6_matched_accounting_v1 import seed
from trottertracks.resource_applicability.ax2a_preparation import digest

ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def no_science_arrays(monkeypatch):
    original=Path.open
    def guarded(path,*a,**k):
        if str(path).endswith('.npz'):raise AssertionError('NO_REAL_ARRAY_INPUT')
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)


def manifest():
    task=dict(cell_id='B1_p19_q8_4th',cell=dict(method='B1'),phase='exploration',replica=0,
        seed=seed('B1_p19_q8_4th','exploration',0),ordinary=False,validate=True)
    registry=dict(missing=[dict(index=i,task=deepcopy(task)) for i in [447,450,*range(451,592)]],environment={'synthetic':True})
    return dict(schema='h6_server_phase_a_manifest_v1',kind=c.KIND,science_source=c.SCIENCE_SOURCE,
        source_commit='a'*40,plan=c.plan(),science_plan_digest=digest(c.plan()),reuse_registry=registry,phase_scope=c.phase_scope(registry),
        assigned_resources=r.resources([20,21,22,23]),observation_policy=r.observation_policy(),
        environment=registry['environment'],profile={'sha256':'b'*64},execution_plan_sealed=True,
        exclusive_output={'absolute_path':'/home/AbeHiromu/SYNTHETIC_ONLY'},**c.flags())


@pytest.mark.parametrize('tamper',['valid','draft','old_schema','phase_b','confirmation','retry','source','cpu','output'])
def test_fresh_grant_and_exact_scope(tamper):
    m=manifest();c.validate_manifest(m);g=c.grant_draft(m)
    g.update(approved_by_user=True,server_usage_approved_by_user=True)
    if tamper=='draft':g['approved_by_user']=False
    if tamper=='old_schema':g['schema']='h6_matched_authorization_v3'
    if tamper=='phase_b':g['phase_b_authorized']=True
    if tamper=='confirmation':g['confirmation_authorized']=True
    if tamper=='retry':g['retry']=True
    if tamper=='source':g['source_commit']='c'*40
    if tamper=='cpu':g['assigned_cpus']=[0,2,5,6]
    if tamper=='output':g['exclusive_output']='/home/AbeHiromu/OTHER'
    if tamper=='valid':c.validate_grant(m,g,Path(m['exclusive_output']['absolute_path']))
    else:
        with pytest.raises(ValueError):c.validate_grant(m,g,Path(m['exclusive_output']['absolute_path']))


@pytest.mark.parametrize('tamper',['resource','science','scope','authorized'])
def test_manifest_rejects_changed_contract(tamper):
    m=manifest()
    if tamper=='resource':m['assigned_resources']['aggregate_rss_bytes']=18*2**30
    if tamper=='science':m['plan']['compiler_proposed']['optimization_level']=2
    if tamper=='scope':m['phase_scope']['remaining_groups']=141
    if tamper=='authorized':m['science_authorized']=True
    with pytest.raises(ValueError):c.validate_manifest(m)


def test_dry_runner_never_imports_science():
    script=ROOT/c.RUNNER
    code=f'import sys,runpy;sys.argv=[{str(script)!r}];\ntry:runpy.run_path({str(script)!r},run_name="__main__")\nexcept SystemExit as exc:assert exc.code==0\nassert not any(x in sys.modules for x in ("numpy","qiskit","pyscf","numba","cupy"))'
    p=subprocess.run([sys.executable,'-I','-P','-B','-c',code],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr


def test_missing_grant_cannot_claim_output(tmp_path):
    out=tmp_path/'not_created'
    p=subprocess.run([sys.executable,'-I','-P','-B',str(ROOT/c.RUNNER),'--execute','--output',str(out)],
        capture_output=True,text=True,timeout=30)
    assert p.returncode!=0 and not out.exists()


def test_saved_json_roundtrip_preserves_science_digest():
    m=manifest();roundtrip=json.loads(json.dumps(m))
    c.validate_manifest(roundtrip)
    assert digest(roundtrip['plan'])==digest(c.plan())


def test_real_limit_inheritance_and_one_thread(tmp_path):
    cpus=[20,21,22,23]
    if not set(cpus)<=os.sched_getaffinity(0):pytest.skip('Proposed preparation CPUs unavailable')
    code='''import os,resource,subprocess,sys,json
sys.path.insert(0,sys.argv[1])
from trottertracks.resource_applicability.h6_matched_execution_v2 import install_limits,worker_environment
install_limits(json.loads(sys.argv[2])[2:],2*2**30,2*2**30,hard_address_bytes=32*2**30)
assert resource.getrlimit(resource.RLIMIT_AS)==(2*2**30,32*2**30)
child="import os,resource,json,sys;sys.path.insert(0,sys.argv[1]);from trottertracks.resource_applicability.h6_matched_execution_v2 import install_limits;from trottertracks.resource_applicability.h6_server_resources_v1 import THREAD_KEYS;install_limits(json.loads(sys.argv[2])[:2],32*2**30,32*2**20);assert resource.getrlimit(resource.RLIMIT_AS)==(32*2**30,32*2**30);assert sorted(os.sched_getaffinity(0))==json.loads(sys.argv[2])[:2];assert all(os.environ[k]=='1' for k in THREAD_KEYS);assert os.environ['QISKIT_PARALLEL']=='FALSE';print('CHILD_LIMIT_PASS')"
r=subprocess.run([sys.executable,'-I','-P','-B','-c',child,sys.argv[1],sys.argv[2]],env=worker_environment(1,__import__('pathlib').Path(sys.argv[3])),capture_output=True,text=True,timeout=10)
assert r.returncode==0,r.stderr
assert resource.getrlimit(resource.RLIMIT_AS)==(2*2**30,32*2**30)
assert sorted(os.sched_getaffinity(0))==json.loads(sys.argv[2])[2:]
print(r.stdout)
'''
    p=subprocess.run([sys.executable,'-I','-P','-B','-c',code,str(ROOT/'src'),json.dumps(cpus),str(tmp_path)],
        capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr


@pytest.mark.parametrize('which',['host_memory','cgroup_memory','oom','host_pressure','cgroup_pressure','nan'])
def test_memory_fail_closed(which):
    policy=r.observation_policy();base=dict(host_available_bytes=1024*2**30,effective_available_bytes=1024*2**30,
        oom={'own':{'oom':0,'oom_kill':0}},full_psi_avg10={'host':0.,'own':0.})
    row=deepcopy(base)
    if which=='host_memory':row['host_available_bytes']=0
    if which=='cgroup_memory':row['effective_available_bytes']=0
    if which=='oom':row['oom']['own']['oom_kill']=1
    if which=='host_pressure':row['full_psi_avg10']['host']=5.
    if which=='cgroup_pressure':row['full_psi_avg10']['own']=.1
    if which=='nan':row['full_psi_avg10']['host']=float('nan')
    assert r.memory_reason(row,base,policy) is not None


@pytest.mark.parametrize('error',[errno.ESRCH,errno.EPERM])
def test_cleanup_only_esrch_is_already_exited(monkeypatch,error):
    class Fake:
        pid=991;returncode=None
        def wait(self,**kw):self.returncode=0;return 0
    p=Fake();identity=dict(start_ticks=1,pgrp=p.pid,session=p.pid)
    monkeypatch.setattr(r,'process_identity',lambda pid:identity)
    def send(*args):raise OSError(error,'injected')
    monkeypatch.setattr(r.os,'killpg',send)
    if error==errno.ESRCH:assert r.stop_owned_group(p,identity)==0
    else:
        with pytest.raises(OSError):r.stop_owned_group(p,identity)
        assert p.returncode is None


def test_ownership_mismatch_and_double_cleanup(monkeypatch):
    class Fake:
        pid=992;returncode=None
    p=Fake();monkeypatch.setattr(r,'process_identity',lambda pid:dict(start_ticks=2,pgrp=992,session=992))
    with pytest.raises(ValueError,match='OWNERSHIP'):r.stop_owned_group(p,dict(start_ticks=1))
    p.returncode=0
    with pytest.raises(ValueError,match='REAPED'):r.stop_owned_group(p,dict(start_ticks=2))


@pytest.mark.parametrize('mode',['exit','abort','rss','missing','io_delay'])
def test_supervisor_minimal_process_cleanup(tmp_path,monkeypatch,mode):
    m=manifest();m['repository_root']=str(ROOT)
    m['observation_policy'].update(interval_seconds=.01,journal_emit_seconds=.01)
    host=dict(host_available_bytes=1024*2**30,effective_available_bytes=1024*2**30,
        oom={'own':{'oom':0,'oom_kill':0}},full_psi_avg10={'host':0.,'own':0.})
    monkeypatch.setattr(e,'host_memory',lambda:host)
    if mode=='rss':monkeypatch.setattr(e,'process_tree_rss',lambda pid:41*2**30)
    if mode=='missing':monkeypatch.setattr(e,'process_memory',lambda pid:(_ for _ in ()).throw(FileNotFoundError()))
    if mode=='io_delay':
        times=iter([0.,6.,6.,6.,6.,6.,6.,6.,6.]);monkeypatch.setattr(e.time,'monotonic',lambda:next(times,6.))
    code="import resource,time,os;resource.setrlimit(resource.RLIMIT_CORE,(0,0));time.sleep(.1)"
    if mode=='abort':code+=';os.abort()'
    result=e.supervise_one([sys.executable,'-I','-P','-B','-c',code],tmp_path,m)
    assert result['process_reaped'] and result['log_and_journal_closed']
    if mode=='exit':assert result['worker_exit_code']==0 and result['first_stop_reason'] is None
    if mode=='exit':
        assert result['coordinator_peak_rss_bytes']>0
        assert result['coordinator_peak_vmsize_bytes']>=result['coordinator_peak_rss_bytes']
        rows=[json.loads(line) for line in (tmp_path.parent/(tmp_path.name+'_resource_observations.jsonl')).read_text().splitlines()]
        assert rows and all({'VmRSS_bytes','VmSize_bytes','VmPeak_bytes','VmHWM_bytes'}<=set(row['coordinator']) for row in rows)
    if mode=='abort':assert result['worker_exit_code']==-6
    if mode=='rss':assert result['first_stop_reason']=='AGGREGATE_RSS_CAP'
    if mode=='missing':assert result['first_stop_reason']=='OBSERVATION_MISSING'
    if mode=='io_delay':assert result['first_stop_reason']=='OBSERVATION_STALE'
    assert not Path(f'/proc/{result["identity"]["pid"]}').exists()


def test_phase_a_function_has_no_next_phase_dispatch():
    import ast,inspect,textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(e.orchestrate)))
    names=[x.func.id for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)]
    assert names.count('supervise_one')==1
    assert not set(names)&{'tasks','confirmation_ids','resource_maps','complete_exploration'}


@pytest.mark.parametrize('quota',['unknown','bytes','inodes','disabled'])
def test_fresh_quota_gate(tmp_path,monkeypatch,quota):
    from types import SimpleNamespace
    monkeypatch.setattr(r.os,'sched_getaffinity',lambda pid:set(range(64)))
    monkeypatch.setattr(r.os,'statvfs',lambda path:SimpleNamespace(f_bavail=100*2**30,f_frsize=1,f_favail=1000000))
    row={'status':'UNKNOWN' if quota=='unknown' else 'KNOWN','items':[]}
    if quota in ('bytes','inodes'):
        row['items']=[dict(status='ACTIVE',available_bytes=0 if quota=='bytes' else 100*2**30,
            available_inodes=0 if quota=='inodes' else 1000000)]
    monkeypatch.setattr(r,'quota_readonly',lambda path:row)
    monkeypatch.setattr(r,'host_memory',lambda:dict(host_available_bytes=1024*2**30,
        effective_available_bytes=1024*2**30,oom={},full_psi_avg10={'host':0.}))
    if quota=='disabled':assert r.fresh_resources(r.resources([20,21,22,23]),r.observation_policy(),tmp_path/'fresh')['quota']==row
    else:
        with pytest.raises(ValueError,match='QUOTA'):r.fresh_resources(r.resources([20,21,22,23]),r.observation_policy(),tmp_path/'fresh')


def test_consumed_one_shot_rejected(tmp_path,monkeypatch):
    m=manifest();out=tmp_path/'output';m['exclusive_output']=dict(repository_path=c.NAMESPACE+'launch_server_v1',absolute_path=str(out))
    m['repository_root']=str(tmp_path)
    m['one_shot_lock']='only_this_lock';(tmp_path/'only_this_lock').write_text('consumed')
    g=c.grant_draft(m);g.update(approved_by_user=True,server_usage_approved_by_user=True)
    monkeypatch.setattr(c,'verify_bindings',lambda root,m:None)
    monkeypatch.setattr(c,'fresh_resources',lambda *args:{})
    monkeypatch.setattr(c,'safe_path',lambda root,p:tmp_path/'only_this_lock' if p=='only_this_lock' else out)
    with pytest.raises(ValueError,match='ONE_SHOT_CONSUMED'):c.validate_launch(tmp_path,m,g,out,requested=True)


def test_parent_death_kills_and_reaps_only_minimal_owned_child(tmp_path):
    # Test driver + harness + parent + child: at most four tiny processes.
    # The harness alone is a subreaper, so no zombie is left to init.
    code=r'''import ctypes,subprocess,sys,json,os,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from trottertracks.resource_applicability.h6_server_resources_v1 import process_identity
assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
ready=Path(sys.argv[2])
child="import sys,json,time,os;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from trottertracks.resource_applicability.h6_server_resources_v1 import install_parent_guard;install_parent_guard(json.loads(sys.argv[2]));Path(sys.argv[3]).write_text(str(os.getpid()));time.sleep(30)"
parent="import subprocess,sys,json,os,time;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from trottertracks.resource_applicability.h6_server_resources_v1 import process_identity;subprocess.Popen([sys.executable,'-I','-P','-B','-c',sys.argv[3],sys.argv[1],json.dumps(process_identity(os.getpid())),sys.argv[2]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);deadline=time.monotonic()+10;\nwhile not Path(sys.argv[2]).exists():\n assert time.monotonic()<deadline;time.sleep(.01)\n"
p=subprocess.Popen([sys.executable,'-I','-P','-B','-c',parent,sys.argv[1],str(ready),child])
assert p.wait(timeout=15)==0
pid=int(ready.read_text());deadline=time.monotonic()+10
while True:
 got,status=os.waitpid(pid,os.WNOHANG)
 if got:break
 assert time.monotonic()<deadline;time.sleep(.01)
assert os.waitstatus_to_exitcode(status)==-9
assert not Path('/proc/'+str(pid)).exists()
print('OWN_PARENT_EXIT_CHILD_REAPED')
'''
    result=subprocess.run([sys.executable,'-I','-P','-B','-c',code,str(ROOT/'src'),str(tmp_path/'ready')],
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
