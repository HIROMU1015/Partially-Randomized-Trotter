"""Additional existing suites; sources and evidence remain read-only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
CODE=r'''
import sys,pathlib,unittest,importlib,json
sys.dont_write_bytecode=True;root=pathlib.Path.cwd();sys.path[:0]=[str(root/'src'),str(root/'tests/tracks/algorithm_codesign')]
m=importlib.import_module(sys.argv[1]);suite=unittest.defaultTestLoader.loadTestsFromModule(m)
def ids(s):
 for x in s:
  if isinstance(x,unittest.TestSuite):yield from ids(x)
  else:yield x.id()
collected=list(ids(suite));r=unittest.TextTestRunner(verbosity=2).run(suite)
forbidden=[x for x in sys.modules if x.rsplit('.',1)[-1]in ('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator','g9_native')]
pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(collected=collected,run=r.testsRun,pass_count=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),fail=len(r.failures),error=len(r.errors),skip=len(r.skipped),forbidden_imports=forbidden),indent=2)+'\n')
'''
def main():
    dest=HERE/('fixed_regressions_'+sys.argv[1]);dest.mkdir(exist_ok=False)
    jobs=[('storage_v2',ROOT,'test_g6_w2c_b3_storage_v2'),('independent_v2',ROOT.parent/'g6-w2c-b3-v2-independent-acceptance-server-20261019','test_g6_w2c_b3_storage_independent_v2')]
    def one(job):
        label,root,module=job;env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('B3_NEGATIVE_EVIDENCE_DIR',None)
        p=subprocess.run([sys.executable,'-c',CODE,module,str(dest/(label+'.json'))],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(dest/(label+'.log')).write_bytes(p.stdout)
        r=json.loads((dest/(label+'.json')).read_text());print(label,{k:r[k]for k in ('run','pass_count','fail','error','skip')},flush=True);return dict(label=label,source_root=str(root),module=module,**r)
    with ThreadPoolExecutor(max_workers=2)as pool:rows=list(pool.map(one,jobs))
    (dest/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert all(x['fail']==x['error']==x['skip']==0 and not x['forbidden_imports']for x in rows)
if __name__=='__main__':main()
