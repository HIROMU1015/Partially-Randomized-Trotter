#!/usr/bin/env python3
"""One bounded stdlib-only regression process; no live processes or science imports."""
import argparse
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

ROOT=Path(__file__).absolute().parents[2];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert args.output.is_absolute() and args.output.is_relative_to('/home/AbeHiromu')
    args.output.mkdir(exist_ok=False)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):
        assert os.environ.get(key)=='1'
    assert sys.flags.safe_path and sys.dont_write_bytecode
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(90,90));signal.alarm(120)
    blocked=dict(scientific_imports=0,scientific_array_reads=0,process_spawns=0,affinity_changes=0)
    class NoScience(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('numpy','scipy','qiskit','pyscf','cupy'):
                blocked['scientific_imports']+=1;raise AssertionError('scientific import forbidden in new contract tests')
    sys.meta_path.insert(0,NoScience())
    def audit(event,args):
        if event in ('subprocess.Popen','os.fork','os.posix_spawn','os.system'):
            blocked['process_spawns']+=1;raise AssertionError('process spawn forbidden')
        if event=='open' and isinstance(args[0],str) and args[0].endswith(('.npz','.npy')):
            blocked['scientific_array_reads']+=1;raise AssertionError('NPZ/NPY reads forbidden')
    sys.addaudithook(audit)
    def no_affinity(*args):blocked['affinity_changes']+=1;raise AssertionError('affinity forbidden in preparation tests')
    os.sched_setaffinity=no_affinity
    path=ROOT/'tests/tracks/resource_applicability/test_h4_b0i_map.py'
    spec=importlib.util.spec_from_file_location('bounded_b0i_map_contract_tests',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    started=time.monotonic()
    with (args.output/'tests.txt').open('x') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    receipt=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        passed=result.wasSuccessful(),wall_seconds=time.monotonic()-started,max_RSS_bytes=int(status['VmHWM'].split()[0])*1024,
        raw_getrusage_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        max_RSS_scope='native VmHWM of this Python process; raw getrusage may include inherited pre-exec history',
        process_count=1,threads=1,AS_cap_bytes=256*2**20,CPU_cap_seconds=90,wall_cap_seconds=120,real_compile_calls=0,artificial_compile_calls=0,**blocked)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(0 if result.wasSuccessful() and not any(blocked.values()) else 1)


if __name__=='__main__':main()
