#!/usr/bin/env python3
"""Bounded mock continuation suite, with no workers or scientific imports."""
import argparse
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

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--evidence',required=True)
args=parser.parse_args();plan=json.loads(Path(args.plan).read_text());root=Path(args.evidence).resolve()
assert str(root).startswith('/home/AbeHiromu/') and not root.exists()
assert plan['checkout']==str(ROOT) and plan['python']==sys.executable
assert sys.flags.safe_path and sys.dont_write_bytecode
assert all(os.environ.get(k)==v for k,v in plan['thread_environment'].items())
assert plan['processes']==1 and plan['transpile_invocations']==0
root.mkdir(parents=True);cases=root/'cases';cases.mkdir();tempfile.tempdir=str(cases)
resource.setrlimit(resource.RLIMIT_AS,(plan['AS_bytes'],plan['AS_bytes']))
resource.setrlimit(resource.RLIMIT_FSIZE,(plan['output_bytes'],plan['output_bytes']))
signal.alarm(plan['wall_seconds'])
original_import=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('qiskit','numpy','pyscf','openfermion','trotterlib','cupy'):
        raise RuntimeError('scientific/compiler/GPU import denied: '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=guarded
def audit(event,args):
    if event in ('subprocess.Popen','os.fork','os.posix_spawn','os.exec','os.system','os.sched_setaffinity'):
        raise RuntimeError('child/affinity/external command denied')
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0]));mode,flags=args[1:3]
        writing=isinstance(mode,str) and any(k in mode for k in 'wax+') or isinstance(flags,int) and flags&(os.O_CREAT|os.O_WRONLY|os.O_RDWR|os.O_TRUNC)
        if writing and not path.is_absolute():
            from trottertracks.resource_applicability.h4_geometry.resources import MANAGED_WRITE_CONTEXT
            managed=getattr(MANAGED_WRITE_CONTEXT,'root',None)
            if managed is None or len(path.parts)!=1 or '..' in path.parts:
                raise RuntimeError('unbound relative test write')
            path=Path(managed)/path
        path=path.absolute()
        if str(path).startswith('/tmp/') or path.suffix in ('.npz','.npy','.pkl'):
            raise RuntimeError('scientific/temporary input denied')
        if writing and not path.is_relative_to(root):raise RuntimeError('test write outside evidence')
sys.addaudithook(audit);sys.path.insert(0,str(ROOT/'src'))
test=ROOT/'tests/tracks/resource_applicability/test_h4_continuation.py'
spec=importlib.util.spec_from_file_location('continuation_tests',test);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
started=time.monotonic()
with (root/'tests.log').open('w') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
summary=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,errors=len(result.errors),
    failures=len(result.failures),skipped=len(result.skipped),wall_seconds=time.monotonic()-started,
    max_RSS_bytes=int(next(l for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmHWM:')).split()[1])*1024,
    rusage_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    output_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
    real_workers=0,transpile_invocations=0,scientific_input_reads=0,production_launches=0,affinity_changes=0)
assert summary['max_RSS_bytes']<=plan['RSS_bytes'] and summary['output_bytes']<=plan['output_bytes']
(root/'result.json').write_text(json.dumps(summary,indent=2)+'\n');signal.alarm(0)
print(json.dumps(summary));sys.exit(0 if result.wasSuccessful() else 1)
