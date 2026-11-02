"""Bounded single-process artificial worker failure regression, no transpile."""
import builtins
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import gates,resources
from trottertracks.resource_applicability.h4_geometry.identity import require


def main():
    evidence=Path(os.environ['H4_WORKER_FAILURE_EVIDENCE'])
    plan_path=Path(os.environ.get('H4_WORKER_FAILURE_PLAN',str(evidence.parent/'ARTIFICIAL_TEST_PLAN_v1.json')))
    require(plan_path.is_absolute() and plan_path.is_relative_to('/home/AbeHiromu'),'home artificial plan')
    plan=json.loads(plan_path.read_text())
    require(str(ROOT)==plan['checkout'] and sys.executable==plan['python'] and sys.flags.safe_path and sys.dont_write_bytecode,
            'fixed checkout/Python -P -B')
    require(evidence.is_absolute() and evidence.is_relative_to('/home/AbeHiromu') and
            not any(p.is_symlink() for p in [evidence,*evidence.parents]) and evidence.stat().st_uid==os.getuid(),'private test ownership')
    require(all(os.environ.get(k)==v for k,v in gates.THREAD_ENV.items()),'test process thread1')
    tempfile.tempdir=str(evidence)
    os.environ['H4_PRELAUNCH_TEST_EVIDENCE']=str(evidence)
    resource.setrlimit(resource.RLIMIT_AS,(plan['AS_bytes'],plan['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(plan['output_bytes'],plan['output_bytes']))
    signal.alarm(plan['wall_seconds'])
    denied=[];original_import=builtins.__import__
    def importing(name,*args,**kwargs):
        if name.split('.')[0] in ('pyscf','openfermion','openfermionpyscf','trotterlib','cupy'):
            denied.append(name);raise RuntimeError('scientific/GPU import forbidden')
        return original_import(name,*args,**kwargs)
    builtins.__import__=importing
    def audit(event,args):
        if event in ('subprocess.Popen','os.system','os.posix_spawn','os.fork','os.exec','os.sched_setaffinity'):
            denied.append(event);raise RuntimeError('child/affinity forbidden')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0]))
            if str(path).startswith('/tmp/') or path.suffix in ('.npz','.npy','.sqlite','.pkl'):
                denied.append(str(path));raise RuntimeError('scientific/temp input forbidden')
        if event not in ('open','os.mkdir','os.remove','os.rmdir','os.rename','os.link'):return
        if not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        writing=event!='open'
        if event=='open':
            mode=args[1] or '';flags=args[2] or 0
            writing=bool((isinstance(mode,str) and any(c in mode for c in 'wax+')) or
                         flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if not writing:return
        for raw in args[:2] if event in ('os.link','os.rename') else args[:1]:
            path=Path(os.fsdecode(raw))
            if not path.is_absolute():
                managed=getattr(resources.MANAGED_WRITE_CONTEXT,'root',None)
                path=Path(managed or Path.cwd())/path
            require(path==Path('/dev/null') or path.is_relative_to(evidence),'artificial write escape')
    sys.addaudithook(audit)
    def module(name,file):
        spec=importlib.util.spec_from_file_location(name,ROOT/file)
        value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
        return value
    new=module('h4_worker_failure_cases','tests/tracks/resource_applicability/test_h4_worker_failures.py')
    cleanup=module('h4_worker_failure_cleanup','tests/tracks/resource_applicability/test_h4_cleanup_esrch.py')
    previous=module('h4_worker_failure_previous','tests/tracks/resource_applicability/test_h4_geometry_source.py')
    def no_transpile(*_args,**_kwargs):raise RuntimeError('additional transpile forbidden')
    import qiskit
    qiskit.transpile=no_transpile;previous.transpile=no_transpile
    suite=unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromModule(new),
        unittest.defaultTestLoader.loadTestsFromModule(cleanup),
        previous.CrossCandidateTests('test_completed_worker_releases_previous_circuit_before_next_read'),
        previous.ParallelCompileTests('test_owned_pool_uses_available_worker_and_latches_death')])
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
    peak=int(next(row.split()[1] for row in Path('/proc/self/status').read_text().splitlines() if row.startswith('VmHWM:')))*1024
    total=sum(p.stat().st_size for p in evidence.rglob('*') if p.is_file())
    require(peak<=plan['RSS_bytes'] and total<=plan['output_bytes'],'bounded artificial RSS/output')
    payload={'status':'PASS' if result.wasSuccessful() and not denied else 'FAIL','tests':result.testsRun,
             'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':peak,'output_bytes':total,
             'denied_attempts':denied,'child_processes':0,'production_workers':0,'mock_workers':12,
             'transpiles':0,'scientific_array_reads':0,'affinity_changes':0,'GPU_query_use':False,
             'scope':'LOCAL_ARTIFICIAL_FAILURE_PATH_AND_CLEANUP_ONLY','production_root_cause_proven':False}
    with (evidence/'test_result_v1.json').open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps(payload));signal.alarm(0)
    return 0 if payload['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
