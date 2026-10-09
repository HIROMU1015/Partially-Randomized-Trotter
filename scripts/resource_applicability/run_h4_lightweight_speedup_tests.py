"""Single-process artificial equivalence; no production, transpile or molecule."""
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest

ROOT=Path(__file__).absolute().parents[2]
OLD_SOURCE='b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb'
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import gates
from trottertracks.resource_applicability.h4_geometry.identity import require


def main():
    evidence=Path(os.environ['H4_SPEEDUP_TEST_EVIDENCE'])
    plan=json.loads((evidence.parent/'ARTIFICIAL_TEST_PLAN_v1.json').read_text())
    require(str(ROOT)==plan['checkout'] and sys.executable==plan['python'] and sys.flags.safe_path and sys.dont_write_bytecode,
            'fixed checkout/interpreter -P -B')
    require(evidence.is_relative_to('/home/AbeHiromu') and not evidence.is_symlink(),'private home evidence')
    require(all(os.environ.get(k)==v for k,v in gates.THREAD_ENV.items()),'internal thread1')
    require(plan['old_source']==OLD_SOURCE,'old SOURCE reference')
    references={};reference_hashes={}
    for name in ('signal','ledger'):
        path='src/trottertracks/resource_applicability/h4_geometry/'+name+'.py'
        raw=subprocess.check_output(['git','-c','maintenance.auto=false','-c','gc.auto=0','-C',str(ROOT),'show',OLD_SOURCE+':'+path])
        qualified='trottertracks.resource_applicability.h4_geometry._old_speedup_'+name
        module=types.ModuleType(qualified);sys.modules[qualified]=module
        exec(compile(raw,OLD_SOURCE+':'+path,'exec'),module.__dict__)
        references[name]=module;reference_hashes[path]=hashlib.sha256(raw).hexdigest()
    tempfile.tempdir=str(evidence)
    resource.setrlimit(resource.RLIMIT_AS,(plan['AS_bytes'],plan['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(plan['output_bytes'],plan['output_bytes']))
    signal.alarm(plan['wall_seconds'])
    denied=[];original_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in ('pyscf','openfermion','openfermionpyscf','trotterlib','cupy'):
            denied.append(name);raise RuntimeError('molecular/GPU imports forbidden')
        return original_import(name,*args,**kwargs)
    builtins.__import__=guarded
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0])
            if path.startswith('/tmp/') or path.endswith(('.npz','.npy','.sqlite','.pkl')):
                denied.append(path);raise RuntimeError('scientific input or /tmp access forbidden')
        if event in ('subprocess.Popen','os.system','os.posix_spawn','os.exec','os.fork'):
            denied.append(event);raise RuntimeError('no child processes or external commands during tests')
        if event=='os.sched_setaffinity':
            denied.append(event);raise RuntimeError('no affinity changes during tests')
    sys.addaudithook(audit)
    import qiskit
    def no_transpile(*a,**kw):
        denied.append('transpile');raise RuntimeError('additional transpile forbidden')
    qiskit.transpile=no_transpile
    def load(name):
        file=ROOT/'tests/tracks/resource_applicability'/(name+'.py')
        spec=importlib.util.spec_from_file_location(name+'_limited',file)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        return module
    new=load('test_h4_lightweight_speedup');new.OLD_SIGNAL=references['signal'];new.OLD_LEDGER=references['ledger']
    previous=load('test_h4_geometry_source')
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(new)])
    for name in ('SignalTests','LedgerTests','CrossCandidateTests','ParallelCompileTests'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(previous,name)))
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
    total=sum(p.stat().st_size for p in evidence.rglob('*') if p.is_file())
    status=Path('/proc/self/status').read_text().splitlines()
    peak=int(next(row.split()[1] for row in status if row.startswith('VmHWM:')))*1024
    require(peak<=plan['RSS_bytes'] and total<=plan['output_bytes'],'bounded artificial RSS/output')
    payload={'status':'PASS' if result.wasSuccessful() and not denied else 'FAIL',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'old_source':OLD_SOURCE,'old_reference_sha256':reference_hashes,'equivalence_counts':new.COUNTS,
        'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':peak,'output_bytes':total,
        'single_test_process':True,'internal_threads':1,'real_workers':0,'mock_max_workers':12,
        'additional_transpiles':0,'scientific_array_reads':0,'GPU_query_use':False,'affinity_changes':0,
        'production_launches':0,'denied_attempts':denied,'evidence_scope':'LOCAL_ARTIFICIAL_IMPLEMENTATION_ONLY',
        'real_Gaussian_OpenFermion_and_compiler_performance_tested':False,
        'carry':{'actual_invocations':20,'charged_bytes':4428938712,'wall_seconds':5472.345380863175}}
    with (evidence/'test_result_v1.json').open('x') as f:json.dump(payload,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:payload[k] for k in ('status','tests','failures','errors','wall_seconds','peak_RSS_bytes','equivalence_counts')}))
    signal.alarm(0);return 0 if payload['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
