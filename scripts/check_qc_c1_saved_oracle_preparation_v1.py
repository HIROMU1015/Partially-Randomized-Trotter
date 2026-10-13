#!/usr/bin/env python3
"""Off-domain synthetic tests; never opens registered comparisons or seal."""
from pathlib import Path
import json,sys,time,resource,unittest,importlib.util
root=Path(__file__).resolve().parents[1]
p=root/'tests/test_qc_c1_saved_oracle_preparation_v1.py'
spec=importlib.util.spec_from_file_location('c1_saved_oracle_fixture_tests',p);module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tic=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromModule(module)
result=unittest.TextTestRunner(verbosity=1).run(suite)
report={'status':'OFF_DOMAIN_SYNTHETIC_PASS' if result.wasSuccessful() else 'OFF_DOMAIN_SYNTHETIC_FAIL',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'wall_seconds':time.monotonic()-tic,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'fixture_inputs':'F_COST artificial cost tables only;no Hamiltonians',
        'registered_diagnostic_evaluations':0,'registered_comparisons_or_seal_read':False,
        'new_eigensolver_PF_compile_calls':0,'workers':1,'threads':1,
        'immutable_CI_or_external_reproduction':False,'mandatory_stop':True}
folder=root/'artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14';folder.mkdir(parents=True,exist_ok=True)
(folder/'synthetic_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));sys.exit(not result.wasSuccessful())
