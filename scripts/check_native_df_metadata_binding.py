"""Run ONLY the fabricated metadata connection tests. No real-data arguments."""
from pathlib import Path
import importlib.util
import resource
import sys
import unittest


def main():
    if len(sys.argv) != 1:
        raise SystemExit('No real-data or cost-evaluation mode is provided')
    resource.setrlimit(resource.RLIMIT_AS, (256 * 2**20, 256 * 2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('_metadata_binding_tests',root/'tests/test_native_df_metadata_binding.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    forbidden={'numpy','scipy','qiskit','qiskit_nature','pyscf','cupy','trottertracks'} & set(sys.modules)
    if forbidden:
        raise RuntimeError('Unexpected scientific imports: '+str(sorted(forbidden)))
    print('scientific_imports=0; native_build=0; compile=0; real_theory_cost=0; subagents=0')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
