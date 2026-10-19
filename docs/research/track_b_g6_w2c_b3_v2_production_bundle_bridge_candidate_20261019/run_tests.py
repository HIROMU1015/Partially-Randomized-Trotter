from pathlib import Path
from hashlib import sha256
import importlib,json,os,sys,unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main():
    label=sys.argv[1];dest=HERE/('tests_'+label);dest.mkdir(exist_ok=False)
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')];sys.dont_write_bytecode=True
    m=importlib.import_module('test_g6_w2c_b3_bundle_v1');suite=unittest.defaultTestLoader.loadTestsFromModule(m)
    def ids(s):
        for t in s:
            if isinstance(t,unittest.TestSuite):yield from ids(t)
            else:yield t.id()
    collected=list(ids(suite))
    with (dest/'tests.log').open('w')as log:r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    result=dict(schema='b3_new_bundle_regressions_v1',collected=collected,run=r.testsRun,pass_count=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),fail=len(r.failures),error=len(r.errors),skip=len(r.skipped),failures=[dict(id=t.id(),trace=msg)for t,msg in r.failures+r.errors],successful=r.wasSuccessful(),source_sha256=sha256(Path(m.__file__).read_bytes()).hexdigest(),historic_counts_added=False)
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ('collected','failures')}));print(json.dumps(result['failures'],indent=2));sys.exit(0 if r.wasSuccessful()else 1)
if __name__=='__main__':main()
