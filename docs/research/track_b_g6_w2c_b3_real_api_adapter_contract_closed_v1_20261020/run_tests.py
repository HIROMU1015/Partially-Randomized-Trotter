"""New self-tests only; historical independent counts are never added."""
from pathlib import Path
import json,sys,unittest,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
sys.dont_write_bytecode=True
if __name__=='__main__':
    dest=Path(sys.argv[1]);dest.mkdir(exist_ok=False)
    suite=unittest.defaultTestLoader.loadTestsFromName('test_g6_w2c_b3_real_adapter_v1')
    with (dest/'tests.log').open('w') as log:r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    result=dict(run=r.testsRun,PASS=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),
                FAIL=len(r.failures),ERROR=len(r.errors),SKIP=len(r.skipped),separate_reviewer=False,
                measured_SOURCE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                failures=[dict(id=t.id(),trace=msg) for t,msg in r.failures+r.errors])
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    sys.exit(not r.wasSuccessful())
