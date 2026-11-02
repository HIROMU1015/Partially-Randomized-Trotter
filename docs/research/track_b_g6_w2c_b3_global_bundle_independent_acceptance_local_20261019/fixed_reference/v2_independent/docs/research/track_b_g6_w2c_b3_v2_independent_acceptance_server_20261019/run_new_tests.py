"""Execute new consumer tests in a process without Producer imports."""
from pathlib import Path
import importlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
m=importlib.import_module('test_g6_w2c_b3_storage_independent_v2');suite=unittest.defaultTestLoader.loadTestsFromModule(m)
def ids(s):
    for t in s:
        if isinstance(t,unittest.TestSuite):yield from ids(t)
        else:yield t.id()
collected=list(ids(suite));r=unittest.TextTestRunner(verbosity=2).run(suite)
forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1] in {'g6_w2c_b3_pilot_candidate','g6_w2c_b3_pilot_storage_v2','g6_w2c_b3_storage_verify_v2','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_admission','g6_w2c_b3_parallel_admission','g6_f3_generator','g6_f3_contract','g6_f3_precision','g7_generator','g7_provider','return_aggregation','g9_native'}]
report=dict(schema='b3_v2_new_independent_tests_v1',collected=collected,run=r.testsRun,passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),failures=[dict(test=t.id(),traceback=x) for t,x in r.failures],errors=[dict(test=t.id(),traceback=x) for t,x in r.errors],skips=len(r.skipped),forbidden_imports=forbidden,successful=r.wasSuccessful() and not forbidden)
(HERE/('new_tests_'+sys.argv[1]+'.json')).write_text(json.dumps(report,indent=2)+'\n');(HERE/('mutations_'+sys.argv[1]+'.json')).write_text(json.dumps(m.RESULTS,indent=2)+'\n')
sys.exit(0 if report['successful'] else 1)
