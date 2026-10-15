"""Run only V1 fabricated tests, with no real H6 inputs or execution grant."""
import importlib.util
from pathlib import Path
import resource
import sys
import unittest


def main():
    if len(sys.argv)!=1: raise SystemExit('No real-data mode')
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    path=Path(__file__).resolve().parents[1]/'tests/test_track_a_v1_readiness.py'
    spec=importlib.util.spec_from_file_location('_v1_artificial_suite',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    bad={'numpy','scipy','qiskit','pyscf','cupy','trottertracks'} & set(sys.modules)
    if bad:raise RuntimeError('Unexpected scientific import:'+str(bad))
    print('new_H6_signal=0; real_compile=0; new_grants=0; subagents=0; scientific_imports=0')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
