"""Protected artificial v3 regressions, isolated spawn-compatible processes."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CODE='''import importlib,json,pathlib,sys,unittest,hashlib
sys.dont_write_bytecode=True
sys.path[:0]=[str(pathlib.Path('src').resolve()),str(pathlib.Path('tests/tracks/algorithm_codesign').resolve())]
m=importlib.import_module('test_'+sys.argv[1]);s=unittest.defaultTestLoader.loadTestsFromModule(m)
def ids(s):
 for t in s:
  if isinstance(t,unittest.TestSuite):yield from ids(t)
  else:yield t.id()
collected=list(ids(s));r=unittest.TextTestRunner(verbosity=2).run(s)
forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1] in ('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator','g9_native')]
data=dict(suite=sys.argv[1],collected=collected,run=r.testsRun,failures=len(r.failures),errors=len(r.errors),skips=len(r.skipped),successful=r.wasSuccessful() and not forbidden,forbidden_science_modules_imported=forbidden,source_sha256=hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest())
pathlib.Path(sys.argv[2]).write_text(json.dumps(data,indent=2)+'\\n');sys.exit(0 if data['successful'] else 1)
'''


def main():
    out=HERE/'integration_regressions';out.mkdir(exist_ok=False)
    def one(n):
        result=out/(n+'.json')
        p=subprocess.run([sys.executable,'-c',CODE,n,str(result)],cwd=ROOT,capture_output=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        (out/(n+'.log')).write_bytes(p.stdout+p.stderr)
        r=json.loads(result.read_text());r['exit_code']=p.returncode
        print(n,r['run'],r['successful'],flush=True)
        return r
    with ThreadPoolExecutor(max_workers=3)as pool:
        rows=list(pool.map(one,['g6_w2c_b3_parallel_codec','g6_w2c_b3_parallel_f01','g6_w2c_b3_parallel_integration']))
    (out/'results.json').write_text(json.dumps(dict(suites=rows),indent=2)+'\n')
    assert all(x['successful']and x['exit_code']==0 for x in rows)


if __name__=='__main__':main()
