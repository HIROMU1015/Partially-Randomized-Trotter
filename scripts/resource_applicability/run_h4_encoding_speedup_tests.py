#!/usr/bin/env python3
"""Execute only the bounded encoding tests, with no scientific/compile access."""
import argparse
import builtins
import hashlib
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

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--plan', required=True)
parser.add_argument('--evidence', required=True)
args = parser.parse_args()
plan = json.loads(Path(args.plan).read_text())
evidence = Path(args.evidence).resolve()
assert str(evidence).startswith('/home/AbeHiromu/') and not evidence.exists()
assert plan['checkout'] == str(ROOT) and plan['python'] == sys.executable
assert sys.flags.safe_path and sys.flags.dont_write_bytecode
for key, value in plan['thread_environment'].items():
    assert os.environ.get(key) == value, key
assert plan['processes'] == 1 and plan['transpile_invocations'] == 0
evidence.mkdir(parents=True)
cache = evidence/'library-cache'; cache.mkdir()
os.environ['MPLCONFIGDIR'] = str(cache)
os.environ['XDG_CACHE_HOME'] = str(cache)
os.environ['TMPDIR'] = str(cache)
tempfile.tempdir = str(cache)
resource.setrlimit(resource.RLIMIT_AS, (plan['AS_bytes'], plan['AS_bytes']))
resource.setrlimit(resource.RLIMIT_FSIZE, (plan['output_bytes'], plan['output_bytes']))
signal.alarm(plan['wall_seconds'])
original_import = builtins.__import__
def guarded_import(name, *pos, **kw):
    if name.split('.')[0] in ('pyscf', 'openfermion', 'trotterlib', 'cupy'):
        raise RuntimeError('scientific/GPU import denied: '+name)
    return original_import(name, *pos, **kw)
builtins.__import__ = guarded_import
def audit(event, values):
    if event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.posix_spawn', 'os.exec', 'os.sched_setaffinity'):
        raise RuntimeError('child/affinity denied: '+event)
    if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(values[0])).absolute()
        if str(path).startswith('/tmp/') or path.suffix.lower() in ('.npz', '.npy', '.pkl'):
            raise RuntimeError('scientific/temporary input denied')
        mode, flags = values[1], values[2]
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and path != Path('/dev/null') and not path.is_relative_to(evidence):
            raise RuntimeError('test write outside evidence: '+str(path))
sys.addaudithook(audit)
sys.path.insert(0, str(ROOT/'src'))
import qiskit
def forbidden_compile(*args, **kwargs):
    raise RuntimeError('artificial/production compile denied')
qiskit.transpile = forbidden_compile
import qiskit.compiler
qiskit.compiler.transpile = forbidden_compile
test = ROOT/'tests/tracks/resource_applicability/test_h4_encoding_speedup.py'
spec = importlib.util.spec_from_file_location('encoding_tests', test)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
start = time.monotonic()
with (evidence/'tests.log').open('w') as log:
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
summary = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
    errors=len(result.errors), failures=len(result.failures), skipped=len(result.skipped),
    wall_seconds=time.monotonic()-start, max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    measurements=module.MEASUREMENTS, processes=1, transpile_invocations=0,
    real_inputs=0, production_launches=0, affinity_changes=0,
    plan_sha256=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest())
assert summary['max_RSS_bytes'] <= plan['RSS_bytes']
(evidence/'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
signal.alarm(0)
print(json.dumps(summary, ensure_ascii=False))
sys.exit(0 if result.wasSuccessful() else 1)
