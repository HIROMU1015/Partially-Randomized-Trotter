"""Run fixed Producer tests separately from independent acceptance checks."""
import scope_guard as GUARD
import json
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
ROOT=Path('/tmp/b3-independent-server-r2-20261020/runtime')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]

if __name__=='__main__':
    category=sys.argv[1]
    selections=dict(unit=['test_g6_w2c_b3_real_adapter_v1'],
        integration=['test_g6_w2c_b3_real_adapter_integration_v1.Integration'],
        regression=['test_g6_w2c_b3_port_bundle_v1.ContractAndAdmission','test_g6_w2c_b3_parallel_codec.Canonical',
                    'test_g6_w2c_b3_parallel_f01','test_g6_w2c_b3_storage_v2.Storage'])
    suffix='_'+sys.argv[2] if len(sys.argv)>2 else ''
    dest=HERE/('final_'+category+suffix);dest.mkdir(exist_ok=False)
    suite=unittest.defaultTestLoader.loadTestsFromNames(selections[category])
    with (dest/'tests.log').open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report=dict(category=category,measured_SOURCE='b7ad344401b10395c49e4cfa806909282859bab5',
        selected=selections[category],run=result.testsRun,
        PASS=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        FAIL=len(result.failures),ERROR=len(result.errors),SKIP=len(result.skipped),
        core_scientific_modules_loaded=GUARD.loaded(),core_import_attempts=GUARD.ATTEMPTS,
        failures=[dict(id=t.id(),trace=msg) for t,msg in result.failures+result.errors],
        role='Producer test rerun; independent oracle results are counted separately')
    (dest/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('failures','selected')}))
    sys.exit(not result.wasSuccessful() or bool(GUARD.loaded()) or bool(GUARD.ATTEMPTS))
