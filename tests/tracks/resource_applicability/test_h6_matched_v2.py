"""Real-process limit inheritance on synthetic metadata only.

The production coordinator installs limits before the production supervisor
spawns children. Children enter verify_worker, never the numerical port.
"""
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from trottertracks.resource_applicability import h6_matched_contract_v1 as v1
from trottertracks.resource_applicability import h6_matched_contract_v2 as contract
from trottertracks.resource_applicability import h6_matched_execution_v1 as old_execution
from trottertracks.resource_applicability import h6_matched_execution_v2 as execution
from trottertracks.resource_applicability.ax2a_preparation import digest

ROOT = Path(__file__).resolve().parents[3]
GIB = 2**30


@pytest.fixture(autouse=True)
def no_real_inputs(monkeypatch):
    old = builtins.open
    oldpath = Path.open

    def check(p):
        if isinstance(p, (str, Path)) and (str(p).endswith('.npz') or '/artifacts/' in str(p)):
            raise AssertionError('REAL_SCIENCE_FORBIDDEN:' + str(p))

    def guarded(p, *a, **kw):
        check(p)
        return old(p, *a, **kw)

    def guardedpath(p, *a, **kw):
        check(p)
        return oldpath(p, *a, **kw)

    monkeypatch.setattr(builtins, 'open', guarded)
    monkeypatch.setattr(Path, 'open', guardedpath)


def run_python(code, *args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-c', code, *map(str, args)],
                            capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


CHILD = r'''
import json,sys,resource,mmap,os
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from trottertracks.resource_applicability import h6_matched_execution_v2 as e
root,directory,out=map(Path,sys.argv[2:])
m,task,writer=e.verify_worker(root,directory,out)
assert not any(x in sys.modules for x in ('numpy','scipy','numba','qiskit','openfermion'))
# Reserve virtual address space only, never touch or retain 2 GiB of RAM.
with mmap.mmap(-1,2*2**30) as probe:
    assert len(probe)==2*2**30
writer.write('synthetic_limits.json',dict(synthetic_only=True,pid=os.getpid(),
    as_limits=resource.getrlimit(resource.RLIMIT_AS),
    fsize=resource.getrlimit(resource.RLIMIT_FSIZE),
    core=resource.getrlimit(resource.RLIMIT_CORE),cpus=sorted(os.sched_getaffinity(0)),
    threads={k:os.environ[k] for k in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS')},
    numerical_port_imported=False))
writer.write('worker_terminal.json',dict(status='TASK_COMPLETE',synthetic_only=True))
'''


def synthetic_run(tmp_path, phase):
    root = tmp_path/'dummy_root'
    root.mkdir()
    for name in contract.source_paths(root):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source bytes only\n')
    for name in ('snapshot.bin', 'snapshot_receipt.json', 'df_receipt.json'):
        (root/name).write_text('{}')
    out = tmp_path/'run'
    out.mkdir()
    m = contract.preparation()
    m.update(source_commit='a'*40,
        source_hashes={p:contract.file_hash(root/p) for p in contract.source_paths(root)},
        assigned_resources=contract.resources(sorted(os.sched_getaffinity(0))[:4]),
        environment=contract.environment(), execution_plan_sealed=True,
        input_identity={key+suffix:val for key, file in (
            ('snapshot','snapshot.bin'), ('snapshot_receipt','snapshot_receipt.json'),
            ('df_receipt','df_receipt.json')) for suffix,val in (
                ('_path',file), ('_sha256',contract.file_hash(root/file)))})
    g = dict(schema='h6_matched_authorization_v2',kind=contract.KIND,approved_by_user=True,
             manifest_digest=digest(m),source_commit=m['source_commit'],
             one_shot=True,retry=False,resume=False,exclusive_output=str(out))
    for name,obj in [('frozen_manifest.json',m), ('authorization.json',g),
                     ('authorization_source.json',g)]:
        (out/name).write_text(json.dumps(obj))
    (out/'launch_binding.json').write_text(json.dumps(dict(manifest_digest=digest(m),
        authorization_digest=digest(g),authorization_source_sha256=contract.file_hash(out/'authorization_source.json'))))
    ts = ([{'phase':'signal'}] if phase=='signal' else
          [dict(phase=phase,cell_id='synthetic_cost',replica=i,seed=i) for i in (0,1)])
    if phase!='signal':
        (out/(phase+'_tasks.json')).write_text(json.dumps({'tasks':ts}))
    return root,out,ts


@pytest.mark.parametrize('phase', ['signal','exploration','confirmation'])
def test_coordinator_then_real_signal_and_cost_children(tmp_path, phase):
    root,out,tasks = synthetic_run(tmp_path,phase)
    # The entire coordinator runs in a separate process; pytest limits stay intact.
    code = r'''
import sys,json,mmap,resource,subprocess
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from trottertracks.resource_applicability import h6_matched_execution_v2 as e
root,out=map(Path,sys.argv[2:4]); phase=sys.argv[4];child=sys.argv[5]
m=json.loads((out/'frozen_manifest.json').read_text())
limits=e.install_coordinator_limits(m['assigned_resources'],m['plan']['caps_proposed']['output_bytes'])
try:
    p=mmap.mmap(-1,2*2**30)
except (OSError,MemoryError):
    allocation_denied=True
else:
    p.close();raise AssertionError('COORDINATOR_SOFT_BUDGET_NOT_ENFORCED')
real=e.subprocess.Popen
def launch(command,**kw):
    d=command[command.index('--worker')+1]
    return real([sys.executable,'-c',child,sys.argv[1],str(root),d,str(out)],**kw)
e.subprocess.Popen=launch
ts=[{'phase':'signal'}] if phase=='signal' else json.loads((out/(phase+'_tasks.json')).read_text())['tasks']
paths,peak=e.supervise_batch(root,out,m,list(enumerate(ts)),signal_stage=phase=='signal')
assert not any(x in sys.modules for x in ('numpy','scipy','numba','qiskit','openfermion'))
print(json.dumps(dict(coordinator=limits,coordinator_soft_budget_enforced=allocation_denied,
    paths=[str(p) for p in paths],peak_rss=peak,final_as=resource.getrlimit(resource.RLIMIT_AS))))
'''
    record = run_python(code, ROOT/'src', root, out, phase, CHILD)
    assert record['coordinator']['rlimit_as'] == [2*GIB,8*GIB]
    assert record['final_as'] == [2*GIB,8*GIB]
    assert record['coordinator_soft_budget_enforced']
    assert len(record['paths']) == len(tasks) and record['peak_rss'] > 0
    seen = []
    for name in record['paths']:
        path = Path(name)
        claim = json.loads((path/'worker_claim.json').read_text())
        limits = json.loads((path/'synthetic_limits.json').read_text())
        binding = json.loads((path/'task_binding.json').read_text())
        assert claim['resource_limits']['inherited_rlimit_as'] == [2*GIB,8*GIB]
        assert claim['resource_limits']['rlimit_as'] == limits['as_limits'] == [8*GIB,8*GIB]
        assert limits['fsize'] == [binding['output_bytes']]*2
        assert limits['core'] == [0,0] and limits['cpus'] == binding['cpus']
        threads = '4' if phase=='signal' else '1'
        assert limits['threads']['NUMBA_NUM_THREADS'] == limits['threads']['OMP_NUM_THREADS'] == threads
        assert all(limits['threads'][k]=='1' for k in
                   ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'))
        assert not limits['numerical_port_imported'] and limits['synthetic_only']
        seen.append(limits['cpus'])
    if phase!='signal':
        assert seen[0] != seen[1]


def test_old_hard_limit_regression_reproduces_in_real_child():
    code = r'''
import sys,json,subprocess
sys.path.insert(0,sys.argv[1])
from trottertracks.resource_applicability import h6_matched_execution_v1 as old
import os
old.install_limits(sorted(os.sched_getaffinity(0))[:4],2*2**30,2*2**30)
child='import resource;resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))'
r=subprocess.run([sys.executable,'-c',child],capture_output=True,text=True)
assert r.returncode!=0 and 'not allowed to raise maximum limit' in r.stderr
print(json.dumps({'old_worker_failed':True}))
'''
    assert run_python(code,ROOT/'src')['old_worker_failed']


def test_inherited_hard_ceiling_too_small_fails_before_launch():
    code = r'''
import sys,resource,os,json
sys.path.insert(0,sys.argv[1])
from trottertracks.resource_applicability import h6_matched_execution_v2 as e
from trottertracks.resource_applicability import h6_matched_contract_v2 as c
resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
for call in (c.check_inherited_limits,lambda:e.install_coordinator_limits(c.resources(sorted(os.sched_getaffinity(0))[:4]),2*2**30)):
    try:call()
    except ValueError as exc:assert str(exc)=='MATCHED_AS_INHERITED_HARD_TOO_SMALL'
    else:raise AssertionError('MUST_REJECT')
assert resource.getrlimit(resource.RLIMIT_AS)==(2*2**30,2*2**30)
print(json.dumps({'rejected_before_launch':True}))
'''
    assert run_python(code,ROOT/'src')['rejected_before_launch']


def sealed(tmp_path,monkeypatch):
    m=contract.preparation();out=tmp_path/'output'
    m.update(execution_plan_sealed=True,source_commit='a'*40,source_hashes={},
        environment={},input_identity={},assigned_resources=contract.resources(sorted(os.sched_getaffinity(0))[:4]),
        exclusive_output=dict(repository_path=contract.NAMESPACE+'launch_v1',absolute_path=str(out)))
    g=dict(schema='h6_matched_authorization_v2',kind=contract.KIND,approved_by_user=True,
        manifest_digest=digest(m),source_commit=m['source_commit'],one_shot=True,retry=False,resume=False,
        assigned_cpus=m['assigned_resources']['assigned_cpus'],exclusive_output=str(out))
    monkeypatch.setattr(contract,'verify_sources',lambda *a:None)
    monkeypatch.setattr(contract,'verify_parent',lambda *a:{})
    monkeypatch.setattr(contract,'environment',lambda:{})
    monkeypatch.setattr(contract,'safe_path',lambda *a:out)
    return m,g,out


@pytest.mark.parametrize('mutation',['valid','v1_grant','infrastructure','plan','digest','retry','unapproved','existing_output'])
def test_fresh_v2_grant_and_unchanged_plan_are_required(tmp_path,monkeypatch,mutation):
    m,g,out=sealed(tmp_path,monkeypatch)
    if mutation=='v1_grant':g['schema']='h6_matched_authorization_v1'
    if mutation=='infrastructure':m['execution_infrastructure']['coordinator_as_soft_bytes']=8*GIB
    if mutation=='plan':m['plan']['maximum_wrappers']=1709
    if mutation=='digest':m['science_plan_digest_v1']='changed'
    if mutation=='retry':g['retry']=True
    if mutation=='unapproved':g['approved_by_user']=False
    if mutation=='existing_output':out.mkdir()
    g['manifest_digest']=digest(m)
    if mutation=='valid':assert contract.validate_launch(tmp_path,m,g,out,requested=True)==m['assigned_resources']
    else:
        with pytest.raises(ValueError):contract.validate_launch(tmp_path,m,g,out,requested=True)


def test_science_plan_and_accounting_are_v1_reuse():
    assert contract.plan is v1.plan and contract.plan()==v1.plan()
    assert execution.tasks is old_execution.tasks
    assert execution.confirmation_ids is old_execution.confirmation_ids
    assert execution.resource_row is old_execution.resource_row
    assert execution.resource_maps.__code__.co_code==old_execution.resource_maps.__code__.co_code
    assert contract.plan()['maximum_wrappers']==1708


def test_default_v2_runner_and_denied_execute_import_no_scientific_packages(tmp_path):
    script=ROOT/contract.RUNNER
    code='import runpy,sys,json\nsys.argv='+repr([str(script),'--output',str(tmp_path/'metadata.json')])+'\ntry:runpy.run_path('+repr(str(script))+',run_name="__main__")\nexcept SystemExit as e:assert e.code==0\nassert not any(n in sys.modules for n in ("numpy","scipy","numba","qiskit","openfermion"))'
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr
    assert json.loads((tmp_path/'metadata.json').read_text())['science_authorized'] is False
    out=tmp_path/'denied'
    r=subprocess.run([sys.executable,str(script),'--execute','--output',str(out)],capture_output=True,text=True,timeout=30)
    assert r.returncode!=0 and not out.exists()


@pytest.mark.parametrize('tamper',[False,True])
def test_saved_stop_audit_checks_coordinator_limits_without_science(tmp_path,monkeypatch,tamper):
    import runpy
    root,out,_=synthetic_run(tmp_path,'signal')
    m=json.loads((out/'frozen_manifest.json').read_text())
    (out/'coordinator_limits.json').write_text(json.dumps({'rlimit_as':[2*GIB,8*GIB]}))
    files={str(p.relative_to(out)):contract.file_hash(p) for p in out.rglob('*') if p.is_file()}
    (out/'execution_inventory.json').write_text(json.dumps({'source_commit':m['source_commit'],'file_hashes':files}))
    (out/'execution_terminal.json').write_text(json.dumps(dict(status='H6_MATCHED_RESOURCE_STOP',
        source_commit=m['source_commit'],mandatory_stop=True,next_stage_authorized=False,H6_status='H6_NOT_AUTHORIZED')))
    module=runpy.run_path(str(ROOT/contract.AUDITOR))
    monkeypatch.setitem(module['audit'].__globals__,'verify_sources',lambda *a:None)
    monkeypatch.setitem(module['audit'].__globals__,'verify_parent',lambda *a:m['input_identity'])
    if tamper:
        (out/'coordinator_limits.json').write_text(json.dumps({'rlimit_as':[8*GIB,8*GIB]}))
        with pytest.raises(ValueError,match='AUDIT_COORDINATOR_LIMITS'):module['audit'](root,out)
    else:
        result=module['audit'](root,out)
        assert result['execution_status']=='H6_MATCHED_RESOURCE_STOP'
        assert not result['numerical_certificate_verified'] and not result['scientific_GO_issued']
