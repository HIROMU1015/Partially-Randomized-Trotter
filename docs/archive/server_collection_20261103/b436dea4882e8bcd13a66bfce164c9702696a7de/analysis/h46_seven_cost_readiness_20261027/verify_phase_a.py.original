"""Record necessary synthetic/metadata checks; never run Phase B."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import prepare
import execute
import test_readiness


def audit(event,args):
    if event=='import' and str(args[0]).split('.')[0] in ('numpy','scipy','qiskit','pyscf','openfermion'):
        raise RuntimeError('Scientific runtime import forbidden in Phase A checks')


def forbidden(*args,**kwargs):
    raise RuntimeError('New molecular/sampling/build/compile call forbidden in Phase A')


def main():
    sys.addaudithook(audit)
    with patch.object(prepare.native,'compose',forbidden),patch.object(prepare.native,'sample_requests',forbidden),patch.object(prepare.native,'compile_cached',forbidden):
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_readiness)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    packages={}
    for site in prepare.PYTHON.parent.parent.glob('lib/python*/site-packages'):
        for name in ('numpy','scipy','qiskit','openfermion','pyscf','psutil','numba'):
            metadata=list(site.glob(name+'-*.dist-info/METADATA'))
            assert len(metadata)==1, 'Missing/ambiguous runtime distribution: '+name
            packages[name]=next(x.split(': ',1)[1] for x in metadata[0].read_text().splitlines() if x.startswith('Version: '))
    assert packages['numpy']=='1.26.4' and packages['qiskit']=='1.3.0'
    value=dict(status='PHASE_A_READY_MANDATORY_STOP', tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), scientific_calls=0,
        phase_B_executed=False, runtime_python=str(prepare.PYTHON), runtime_packages=packages,
        provider_SOURCE=prepare.PROVIDER, provider_copy=str(prepare.TASK/'provider'),
        H4_snapshot='SAVED_RECONSTRUCTION_SCIENTIFIC_ARRAYS_IDENTICAL; container_SHA_differs',
        H6_snapshot='ORIGINAL_BYTES_IDENTICAL',
        resources=dict(allowed_logical_CPUs=len(os.sched_getaffinity(0)), memory_observation=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:'))], reservation=False),
        source_hashes_checked=len(execute.read_plan()['source_hashes']),
        saved_data_hashes_checked=len(execute.read_plan()['saved_data_hashes']),
        scientific_output_directories_exist=Path(execute.read_plan()['private_output_root']).exists() or Path(execute.read_plan()['public_results_root']).exists(),
        new_approval_packet_or_grant_system=False, new_physical_results=False,
        validation_scope='LOCAL_SYNTHETIC_AND_SAVED_METADATA_ONLY; not molecular performance evidence')
    (HERE/'verification.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':
    main()
