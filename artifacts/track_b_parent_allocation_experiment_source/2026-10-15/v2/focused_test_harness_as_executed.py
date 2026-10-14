"""Off-domain tests only; record source/version/runtime and actual coverage."""
import os,pathlib,sys,unittest,json,time,resource,hashlib
root=pathlib.Path.cwd();sys.path[:0]=[str(root/'src'),str(root/'tests/tracks/algorithm_codesign')]
p=root/'artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v2'
start=time.monotonic();cpu=time.process_time();suite=unittest.TestSuite();loader=unittest.defaultTestLoader
names=('test_parent_experiment_owner_repair','test_parent_experiment','test_parent_allocation_native')
for name in names:suite.addTests(loader.loadTestsFromName(name))
with (p/'focused_tests.log').open('w') as log:
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',new_owner_regression_cases=7,existing_experiment_cases=45,existing_native_regression_cases=26,total_test_cases=result.testsRun,
 failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),runtime_executable=sys.executable,
 wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 numeric_threads=1,workers_in_test_process=1,mocked_simultaneous_children=2,
 fixtures='artificial PID101/202, ratio2/11 and3/13; prior off-domain local p1/7,2/7,4/7 x1/3 and1-qubit controls',
 actual_synthesis_calls=0,actual_science_worker_executions=0,registered_operator_rechecks=0,registered_candidates=0,registered_forecasts=0,registered_matrix_evaluations=0,
 actual_acquisition_and_worker_gate_exercised_with_mock_Popen=True,real_worker_backend_not_called=True,
 scientific_execution_authorized=False,not_CI_or_external_reproduction=True,mandatory_STOP=True,
 log_sha256=hashlib.sha256((p/'focused_tests.log').read_bytes()).hexdigest(),harness_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
(p/'focused_tests_execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False));sys.exit(0 if result.wasSuccessful() else 1)
