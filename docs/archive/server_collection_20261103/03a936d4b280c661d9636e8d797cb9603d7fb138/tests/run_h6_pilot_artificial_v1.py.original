"""Bounded stdlib-only harness for the authorized H6 preparation tests."""
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
BLOCKED={'numpy','qiskit','cupy','pyscf','trotterlib','trottertracks'}
class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in BLOCKED:raise RuntimeError('SCIENTIFIC_IMPORT_FORBIDDEN:'+fullname)

if __name__=='__main__':
    scratch=Path(os.environ['TMPDIR']).resolve()
    if not scratch.is_relative_to('/home/AbeHiromu'):raise RuntimeError('HOME_SCRATCH_REQUIRED')
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(120,122))
    resource.setrlimit(resource.RLIMIT_FSIZE,(64*2**20,64*2**20))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(RuntimeError('ARTIFICIAL_WALL_CAP')))
    signal.alarm(180);sys.meta_path.insert(0,NoScience());started=time.monotonic()
    suite=unittest.TestSuite();names=[]
    requested=sys.argv[2:] or ['test_h6_b0i_pilot_runner_v1.py','test_h6_pilot_independent_audit_v1.py',
            'test_final_pre_review_h6_bridge_v1.py']
    allowed={'test_h6_b0i_pilot_runner_v1.py','test_h6_pilot_independent_audit_v1.py','test_final_pre_review_h6_bridge_v1.py',
        'test_h6_b0i_parallel_compile_v1.py'}
    if not set(requested)<=allowed:raise RuntimeError('ARTIFICIAL_TEST_SCOPE')
    def no_real_fork():raise RuntimeError('ARTIFICIAL_HARNESS_FORBIDS_REAL_FORK')
    os.fork=no_real_fork
    for filename in requested:
        spec=importlib.util.spec_from_file_location(filename[:-3],ROOT/'tests'/filename)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        tests=unittest.defaultTestLoader.loadTestsFromModule(module)
        names.append(dict(path='tests/'+filename,count=tests.countTestCases()));suite.addTests(tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    items=list(scratch.rglob('*'));size=sum(p.stat().st_size for p in items if p.is_file())
    if size>512*2**20 or len(items)>5000:raise RuntimeError('ARTIFICIAL_SCRATCH_CAP')
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),groups=names,
        wall_seconds=time.monotonic()-started,peak_rss_bytes=usage.ru_maxrss*1024,
        cpu_seconds=usage.ru_utime+usage.ru_stime,scratch_remaining_bytes=size,
        scratch_remaining_entries=len(items),real_scientific_imports=[k for k in sys.modules if k.split('.')[0] in BLOCKED],
        real_scientific_operations=0,real_child_processes=0,real_affinity_changes=0,additional_transpile=0,
        old_75_results_rerun=False)
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
    signal.alarm(0);sys.exit(0 if result.wasSuccessful() else 1)
