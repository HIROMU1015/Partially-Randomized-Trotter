"""One bounded test driver; only explicitly scoped synthetic observers spawn."""
import builtins
import hashlib
import importlib.util
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import tempfile
import unittest

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import gates
from trottertracks.resource_applicability.h4_geometry.identity import require


def main():
    evidence=Path(os.environ['H4_ARTIFICIAL_EVIDENCE'])
    require(evidence.is_absolute() and evidence.is_relative_to('/home/AbeHiromu') and
            not any(p.is_symlink() for p in [evidence,*evidence.parents]), 'synthetic evidence path')
    plan=json.loads((evidence.parent/'ARTIFICIAL_TEST_PLAN_v1.json').read_text())
    require(plan['scope']=='SYNTHETIC_ONLY' and plan['python']==sys.executable and
            plan['checkout']==str(ROOT) and all(os.environ.get(k)==v for k,v in gates.THREAD_ENV.items()),
            'pre-fixed artificial plan/interpreter/thread environment')
    require(sys.flags.safe_path and sys.dont_write_bytecode, 'Python -P -B')
    # dill queries this during Qiskit import. Avoid tempfile's /tmp write probe;
    # only this driver process sees the home-local path, without probing/writing.
    tempfile.tempdir=str(evidence)
    resource.setrlimit(resource.RLIMIT_AS,(plan['driver_AS_bytes'],plan['driver_AS_bytes']))
    signal.alarm(plan['per_attempt_wall_seconds'])
    attempts=[]
    original_import=builtins.__import__
    def guarded(name,*args,**kw):
        if name.split('.')[0] in ('pyscf','openfermion','openfermionpyscf','trotterlib','cupy'):
            attempts.append('import:'+name);raise RuntimeError('forbidden scientific/GPU import')
        return original_import(name,*args,**kw)
    builtins.__import__=guarded
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0])
            if path.startswith('/tmp/') or path.endswith(('.npz','.npy','.sqlite','.pkl')) or any(
                p in path.split('/') for p in ('.runtime','runtime','checkpoint','checkpoints','cache','caches')):
                attempts.append('open:'+path);raise RuntimeError('forbidden scientific/temporary path')
        if event=='subprocess.Popen':
            argv=args[1]
            require(len(argv)==7 and argv[0]==sys.executable and argv[1:3]==['-P','-B'] and
                    argv[3]==str(ROOT/'src/trottertracks/resource_applicability/h4_geometry/observer.py') and
                    argv[4]=='--private-observer', 'only synthetic observer process is permitted')
        if event in ('os.system','os.exec','os.posix_spawn'):raise RuntimeError('forbidden external command')
    sys.addaudithook(audit)
    import qiskit
    compiler=dict(scope='ARTIFICIAL_PREPARATION_ONLY', runtime_authorization=False,
        inherited_defaults={k:repr(v.default) for k,v in inspect.signature(qiskit.transpile).parameters.items()},
        plugins_metadata=sorted([dict(group=e.group,name=e.name,value=e.value) for e in metadata.entry_points()
            if e.group.startswith('qiskit.')],key=lambda e:(e['group'],e['name'],e['value'])),
        qiskit_version=metadata.version('qiskit'),rustworkx_version=metadata.version('rustworkx'),
        compiler_equivalence_to_old_established=False,
        explicit_options=json.loads((ROOT/'artifacts/resource_applicability/track_a_h4_new_server_preparation/2026-10-07/binding_draft_v1.json').read_text())['compiler_options_reference'])
    compiler['profile_sha256']=hashlib.sha256(json.dumps(compiler,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with (evidence/'preparation_compiler_profile_v1.json').open('x') as out:
        json.dump(compiler,out,indent=2);out.write('\n')
    def no_transpile(*a,**kw):
        attempts.append('transpile');raise RuntimeError('additional transpile budget is zero')
    qiskit.transpile=no_transpile
    # Profile defaults were captured before replacing the entry point in this process.
    test=ROOT/'tests/tracks/resource_applicability/test_h4_monitor_fix_a.py'
    spec=importlib.util.spec_from_file_location('h4_monitor_fix_a_tests',test)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    usage=resource.getrusage(resource.RUSAGE_SELF)
    logs=sum(p.stat().st_size for p in evidence.iterdir() if p.is_file())
    payload=dict(status='PASS' if result.wasSuccessful() and not attempts else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        wall_seconds=time.monotonic()-start,driver_peak_RSS_bytes=int(usage.ru_maxrss)*1024,
        driver_user_seconds=usage.ru_utime,driver_system_seconds=usage.ru_stime,
        observer_children_user_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
        observer_children_system_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        protected_attempts=attempts,measurement=module.MEASUREMENTS,observer_trace_bytes=logs,
        thread_environment=gates.THREAD_ENV,transpile_new=0,synthetic_transpile_prior=28,synthetic_cap=64,
        old_benchmarks=128,old_benchmarks_rerun=0,real_worker_launches=0,production_launches=0,
        input_receipt='NPZ6/freeze/runtime/control not received; no scientific paths accessed',
        python=sys.executable,allowed_cpus=[],runtime_authorization=False)
    require(payload['driver_peak_RSS_bytes']<=plan['driver_RSS_candidate_bytes'],'artificial driver RSS cap')
    require(logs<=plan['per_attempt_output_bytes'],'artificial output cap')
    with (evidence/'test_result_v1.json').open('x') as out:json.dump(payload,out,indent=2,ensure_ascii=False);out.write('\n')
    print(json.dumps({k:payload[k] for k in ('status','tests','failures','errors','skipped','wall_seconds','driver_peak_RSS_bytes')}))
    signal.alarm(0)
    return 0 if payload['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
