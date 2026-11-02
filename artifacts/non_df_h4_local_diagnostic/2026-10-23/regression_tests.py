"""Reuse frozen artificial/regression suite; no real H4 numerical calls."""
from pathlib import Path
import importlib.util, unittest, json, sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
suite=unittest.TestSuite();counts={}
for name,file in [('frozen_T','test_h4_certificate_gates.py'),('frozen_binding','test_h4_evidence_binding.py'),('frozen_L','test_h4_local_enclosure.py')]:
 spec=importlib.util.spec_from_file_location('L_execution_'+name,ROOT/'tests'/file)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 s=unittest.defaultTestLoader.loadTestsFromModule(m);counts[name]=s.countTestCases();suite.addTest(s)
with (HERE/'regression_tests.log').open('x') as log:
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),modules=counts,scope='Unmodified T67/binding33/L48 artificial numerical and mock authority/native cases; no real H4 evaluation or valid science grant',real_H4_numeric_calls=0,real_science_grants_issued=0)
with (HERE/'regression_tests.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record));sys.exit(0 if result.wasSuccessful() else 1)
