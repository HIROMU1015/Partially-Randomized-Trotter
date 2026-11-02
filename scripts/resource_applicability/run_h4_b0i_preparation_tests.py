#!/usr/bin/env python3
"""Bounded single-process tests, artificial circuits only, actual transpile forbidden."""
import argparse
import builtins
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    scratch=args.output/'scratch';scratch.mkdir();os.environ['H4_B0I_TEST_TMP']=str(scratch)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):
        if os.environ.get(key)!='1':raise ValueError('thread1 environment required: '+key)
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(120,120));signal.alarm(180)
    start=time.monotonic();counts=dict(real_transpile=0,subprocess=0,scientific_array_access=0,GPU_or_molecular_import=0)
    original_import=builtins.__import__
    def guarded_import(name,*a,**kw):
        if name.split('.')[0] in ('cupy','pyscf','openfermion','openfermionpyscf'):
            counts['GPU_or_molecular_import']+=1;raise RuntimeError('forbidden science/GPU import')
        return original_import(name,*a,**kw)
    builtins.__import__=guarded_import
    def audit(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)) and os.fsdecode(values[0]).endswith(('.npz','.npy')):
            counts['scientific_array_access']+=1;raise RuntimeError('no NPZ/NPY reads')
        if event in ('subprocess.Popen','os.system'):
            counts['subprocess']+=1;raise RuntimeError('no real processes in mock suite')
    sys.addaudithook(audit)
    import qiskit
    def refuse_transpile(*a,**kw):counts['real_transpile']+=1;raise RuntimeError('transpile forbidden, including artificial')
    qiskit.transpile=refuse_transpile
    path=ROOT/'tests/tracks/resource_applicability/test_h4_b0i_preparation.py'
    spec=importlib.util.spec_from_file_location('h4_b0i_only_synthetic',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (args.output/'tests.log').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        scope='<=3-qubit artificial unitary/wrappers; mock launch/resource/cleanup; local, not real H4 compiler output or CI',
        counts=counts,wall_seconds=time.monotonic()-start,max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        process_count=1,internal_threads=1,AS_cap_bytes=2*2**30,CPU_cap_seconds=120,wall_cap_seconds=180)
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() and not any(counts.values()) else 1)


if __name__=='__main__':main()
