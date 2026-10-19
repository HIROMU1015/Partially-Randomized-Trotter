"""Protected producer suites and new storage unit suite, isolated processes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CODE='''import importlib,json,pathlib,sys,unittest,hashlib
sys.dont_write_bytecode=True
sys.path[:0]=[str(pathlib.Path('src').resolve()),str(pathlib.Path('tests/tracks/algorithm_codesign').resolve())]
m=importlib.import_module('test_'+sys.argv[1]);s=unittest.defaultTestLoader.loadTestsFromModule(m)
def ids(s):
 for t in s:
  if isinstance(t,unittest.TestSuite):yield from ids(t)
  else:yield t.id()
collected=list(ids(s));r=unittest.TextTestRunner(verbosity=2).run(s)
forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1]in ('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator','g9_native')]
value=dict(suite=sys.argv[1],collected=collected,run=r.testsRun,failures=len(r.failures),errors=len(r.errors),skips=len(r.skipped),successful=r.wasSuccessful()and not forbidden,forbidden_science_imports=forbidden,source_sha256=hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest())
pathlib.Path(sys.argv[2]).write_text(json.dumps(value,indent=2)+'\\n');sys.exit(0 if value['successful']else 1)
'''

def main():
    out=HERE/'focused_tests';out.mkdir(exist_ok=False)
    def one(name):
        result=out/(name+'.json')
        p=subprocess.run([sys.executable,'-c',CODE,name,str(result)],cwd=ROOT,capture_output=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        (out/(name+'.log')).write_bytes(p.stdout+p.stderr)
        data=json.loads(result.read_text());data['exit_code']=p.returncode
        print(name,data['run'],data['successful'],flush=True);return data
    names=['g6_w2c_b3_parallel_codec','g6_w2c_b3_parallel_f01','g6_w2c_b3_parallel_integration',
           'g6_w2c_b3_pilot_candidate','g6_w2c_b3_storage_v2']
    with ThreadPoolExecutor(max_workers=len(names))as pool:rows=list(pool.map(one,names))
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');assert all(r['successful']and r['exit_code']==0 for r in rows)

if __name__=='__main__':main()
