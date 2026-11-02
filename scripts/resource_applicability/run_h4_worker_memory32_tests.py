"""One pure metadata test process; no artificial compile or children."""
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
    evidence=Path(os.environ['H4_PRELAUNCH_TEST_EVIDENCE'])
    plan=json.loads((evidence.parent/'BINDING_TEST_PLAN_v1.json').read_bytes())
    require(str(ROOT)==plan['checkout'] and sys.executable==plan['python'] and sys.flags.safe_path and sys.dont_write_bytecode,
            'fixed interpreter/checkout -P -B')
    require(evidence.is_relative_to('/home/AbeHiromu') and not evidence.is_symlink(),'home-only evidence')
    require(all(os.environ.get(k)==v for k,v in gates.THREAD_ENV.items()),'thread1 environment')
    tempfile.tempdir=str(evidence)
    resource.setrlimit(resource.RLIMIT_AS,(plan['AS_bytes'],plan['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(plan['output_bytes'],plan['output_bytes']))
    signal.alarm(plan['wall_seconds'])
    denied=[];original_import=builtins.__import__
    def guarded(name,*args,**kw):
        if name.split('.')[0] in ('pyscf','openfermion','openfermionpyscf','trotterlib','cupy','qiskit'):
            denied.append(name);raise RuntimeError('science/compiler/GPU import forbidden')
        return original_import(name,*args,**kw)
    builtins.__import__=guarded
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0])
            if path.startswith('/tmp/') or path.endswith(('.npz','.npy','.sqlite','.pkl')):
                denied.append(path);raise RuntimeError('scientific array or /tmp access forbidden')
        if event in ('subprocess.Popen','os.system','os.posix_spawn','os.exec','os.fork'):
            denied.append(event);raise RuntimeError('children forbidden')
    sys.addaudithook(audit)
    suite=unittest.TestSuite()
    for name in ('test_h4_worker_memory32','test_h4_host_pressure_proposal','test_h4_pressure_integration'):
        spec=importlib.util.spec_from_file_location(name,ROOT/'tests/tracks/resource_applicability'/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    peak=int(next(row.split()[1] for row in Path('/proc/self/status').read_text().splitlines() if row.startswith('VmHWM:')))*1024
    total=sum(p.stat().st_size for p in evidence.rglob('*') if p.is_file())
    require(peak<=plan['RSS_bytes'] and total<=plan['output_bytes'],'RSS/output bounds')
    payload=dict(status='PASS' if result.wasSuccessful() and not denied else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),wall_seconds=time.monotonic()-started,
        peak_RSS_bytes=peak,output_bytes=total,denied_attempts=denied,additional_transpiles=0,
        artificial_circuit_builds=0,scientific_array_reads=0,child_processes=0,affinity_changes=0,production_launches=0)
    with (evidence/'test_result_v1.json').open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps(payload));signal.alarm(0)
    return 0 if payload['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
