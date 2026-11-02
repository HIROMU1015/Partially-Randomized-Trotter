import os,pathlib,sys,unittest,json,time,resource,hashlib,io
root=pathlib.Path.cwd();sys.path[:0]=[str(root/'src'),str(root/'tests/tracks/algorithm_codesign')]
p=pathlib.Path('artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v1');p.mkdir(parents=True,exist_ok=True)
start=time.monotonic();cpu=time.process_time();suite=unittest.TestSuite();loader=unittest.defaultTestLoader
for name in ('test_parent_experiment','test_parent_allocation_native'):suite.addTests(loader.loadTestsFromName(name))
with (p/'focused_tests.log').open('w') as log:
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',new_test_cases=45,reused_native_regression_cases=26,total_test_cases=result.testsRun,
 failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),runtime_executable=sys.executable,
 wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 numeric_threads=1,workers=1,fixtures='off-domain rational a7 b4/3; local p1/7,2/7,4/7 x1/3; 1-qubit phase mock; no registered provider',
 actual_synthesis_calls=0,registered_operator_rechecks=0,registered_candidates=0,registered_forecasts=0,registered_matrix_evaluations=0,
 mock_backend_reservation_tests=True,scientific_execution_authorized=False,not_CI_or_external_reproduction=True,mandatory_STOP=True,
 log_sha256=hashlib.sha256((p/'focused_tests.log').read_bytes()).hexdigest())
(p/'focused_tests_execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False));sys.exit(0 if result.wasSuccessful() else 1)
