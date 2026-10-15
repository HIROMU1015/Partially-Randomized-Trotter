"""Artificial H6 cost-adapter tests only. No real input or native/cost run."""
import importlib.util
from pathlib import Path
import resource
import sys
import unittest


def main():
    if len(sys.argv)!=1:raise SystemExit('No real-data test mode')
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    p=Path(__file__).resolve().parents[1]/'tests/test_native_df_h6_cost_adapter.py'
    spec=importlib.util.spec_from_file_location('_v2_artificial_suite',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if {'numpy','scipy','qiskit','pyscf','cupy','trottertracks'} & set(sys.modules):
        raise RuntimeError('Unexpected scientific import')
    print('real_H6_theoretical_cost=0; native_build=0; compile=0; new_grants=0; scientific_imports=0; subagents=0')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
