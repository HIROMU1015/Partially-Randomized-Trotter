"""Same-primary-agent integration tests and preserved negative overlays."""
from pathlib import Path
import importlib,json,os,sys,unittest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def main():
    label=sys.argv[1];dest=HERE/('runs_tests_'+label);dest.mkdir(exist_ok=False)
    os.environ['B3_BUNDLE_MUTATION_DIR']=str(dest/'mutations')
    os.environ.setdefault('B3_BUNDLE_MATRIX',str(HERE/'runs_final'))
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')];sys.dont_write_bytecode=True
    m=importlib.import_module('test_g6_w2c_b3_port_bundle_v1');suite=unittest.defaultTestLoader.loadTestsFromModule(m)
    def ids(s):
        for t in s:
            if isinstance(t,unittest.TestSuite):yield from ids(t)
            else:yield t.id()
    names=list(ids(suite))
    with(dest/'tests.log').open('w')as log:r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    result=dict(schema='b3_integrated_port_tests_v1',run=r.testsRun,PASS=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),
        FAIL=len(r.failures),ERROR=len(r.errors),SKIP=len(r.skipped),collected=names,source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        failures=[dict(id=t.id(),trace=msg)for t,msg in r.failures+r.errors],mutations=m.MUTATION_RESULTS,
        historical_counts_not_added=True,separate_reviewer=False)
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ('run','PASS','FAIL','ERROR','SKIP')}));print(json.dumps(result['failures'],indent=2))
    sys.exit(0 if r.wasSuccessful()else 1)
if __name__=='__main__':main()
