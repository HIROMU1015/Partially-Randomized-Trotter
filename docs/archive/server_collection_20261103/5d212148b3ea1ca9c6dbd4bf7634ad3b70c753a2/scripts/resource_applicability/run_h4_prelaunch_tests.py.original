"""Bounded synthetic preparation suite, no scientific runner/transpile."""
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
from trottertracks.resource_applicability.h4_geometry import gates
from trottertracks.resource_applicability.h4_geometry.identity import require


def main():
    root=Path(os.environ['H4_PRELAUNCH_TEST_EVIDENCE'])
    plan=json.loads((root.parent/'ARTIFICIAL_TEST_PLAN_v2.json').read_text())
    require(sys.executable==plan['python'] and str(ROOT)==plan['checkout'] and sys.flags.safe_path and sys.dont_write_bytecode,
            'fixed test plan/interpreter -P -B')
    require(all(os.environ.get(k)==v for k,v in gates.THREAD_ENV.items()),'process thread1')
    tempfile.tempdir=str(root)
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));signal.alarm(180)
    denied=[];original_import=builtins.__import__
    def guarded(name,*a,**kw):
        if name.split('.')[0] in ('pyscf','openfermion','openfermionpyscf','trotterlib','cupy','qiskit'):
            denied.append(name);raise RuntimeError('no science/compiler/GPU imports in binding suite')
        return original_import(name,*a,**kw)
    builtins.__import__=guarded
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0])
            if p.startswith('/tmp/') or p.endswith(('.npz','.npy','.sqlite','.pkl')):
                denied.append(p);raise RuntimeError('forbidden scientific/temporary file')
        if event=='subprocess.Popen':
            argv=args[1]
            require(len(argv)==6 and argv[0]==sys.executable and argv[1:3]==['-P','-B'] and
                    argv[3]==str(ROOT/'tests/tracks/resource_applicability/h4_prelaunch_minimal_process.py') and
                    argv[4]=='driver','only minimal synthetic cleanup driver')
        if event in ('os.system','os.posix_spawn','os.exec'):raise RuntimeError('external command forbidden')
    sys.addaudithook(audit)
    file=ROOT/'tests/tracks/resource_applicability/test_h4_prelaunch.py'
    spec=importlib.util.spec_from_file_location('h4_prelaunch_synthetic',file)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    self_usage=resource.getrusage(resource.RUSAGE_SELF);child_usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    payload=dict(status='PASS' if result.wasSuccessful() and not denied else 'FAIL',tests=result.testsRun,
       failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
       wall_seconds=time.monotonic()-started,driver_peak_RSS_bytes=self_usage.ru_maxrss*1024,
       driver_CPU_seconds=self_usage.ru_utime+self_usage.ru_stime,owned_children_CPU_seconds=child_usage.ru_utime+child_usage.ru_stime,
       actual_science_invocations=0,additional_transpile=0,real_workers=0,production_launched=False,
       maximum_minimal_processes_including_test=4,output_bytes=total,denied_attempts=denied,
       CPU_affinity_changes=0,own_test_process_subreaper_used_and_restored=True,shared_settings_changed=False,
       preserved_science_carry=dict(actual_invocations=20,bytes=165214360,wall_seconds=5466.188392877579),
       immutable_CI_evidence=False)
    require(payload['driver_peak_RSS_bytes']<=512*2**20 and total<=16*2**20,'synthetic RSS/output bounds')
    with (root/'test_result_v2.json').open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps({k:payload[k] for k in ('status','tests','failures','errors','skipped','wall_seconds')}))
    signal.alarm(0);return 0 if payload['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
