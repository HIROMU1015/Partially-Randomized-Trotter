from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ast,json,subprocess,sys,os
HERE=Path(__file__).resolve().parent;RUNTIME=HERE/'legacy_runtime'
NAMES=['g6_w2c_b3_admission','g6_w2c_b3_readiness','g6_w2c_b3_driver','g6_w2c_cost_preparation','g6_w2c_b3_f01_provenance','g6_w2c_b3_host_integration','g6_w2c_b3_offline_verifier','g6_w2c_b3_parallel_offline_verifier','g6_w2c_b3_v3_offline_verifier','g6_w2c_b3_pilot_independent']
CODE=r'''
import importlib,pathlib,sys,unittest,json,hashlib,importlib.util
sys.dont_write_bytecode=True;root=pathlib.Path.cwd();sys.path[:0]=[str(root/'src'),str(root/'tests/tracks/algorithm_codesign')]
n=sys.argv[1];correction=sys.argv[3]=='corrected';delta=None
if n=='g6_w2c_b3_parallel_offline_verifier':
 spec=importlib.util.spec_from_file_location('versioned_legacy_test_v1',sys.argv[4]);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
 m,delta=v.load(root/'tests/tracks/algorithm_codesign/test_g6_w2c_b3_parallel_offline_verifier.py',correction)
else:m=importlib.import_module('test_'+n)
suite=unittest.defaultTestLoader.loadTestsFromModule(m)
def ids(s):
 for t in s:
  if isinstance(t,unittest.TestSuite):yield from ids(t)
  else:yield t.id()
collected=list(ids(suite));r=unittest.TextTestRunner(verbosity=2).run(suite)
forbidden=[x for x in sys.modules if x.rsplit('.',1)[-1]in ('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator','g9_native')]
result=dict(suite=n,collected=collected,run=r.testsRun,pass_count=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),fail=len(r.failures),error=len(r.errors),skip=len(r.skipped),fail_ids=[t.id()for t,_ in r.failures+r.errors],forbidden_imports=forbidden,delta=delta)
pathlib.Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
'''
def main():
    label=sys.argv[1];dest=HERE/('legacy_'+label);dest.mkdir(exist_ok=False)
    def one(pair):
        mode,n=pair;p=subprocess.run([sys.executable,'-c',CODE,n,str(dest/(mode+'_'+n+'.json')),mode,str(HERE/'versioned_legacy_test_v1.py')],cwd=RUNTIME,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        (dest/(mode+'_'+n+'.log')).write_bytes(p.stdout);row=json.loads((dest/(mode+'_'+n+'.json')).read_text());print(mode,n,row['pass_count'],row['fail'],row['error'],flush=True);return mode,row
    with ThreadPoolExecutor(max_workers=10)as pool:rows=list(pool.map(one,[(mode,n)for mode in ('original','corrected')for n in NAMES]))
    result={mode:dict(run=sum(r['run']for m,r in rows if m==mode),pass_count=sum(r['pass_count']for m,r in rows if m==mode),fail=sum(r['fail']for m,r in rows if m==mode),error=sum(r['error']for m,r in rows if m==mode),skip=sum(r['skip']for m,r in rows if m==mode),suites=[r for m,r in rows if m==mode])for mode in ('original','corrected')}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:{x:v[x]for x in ('run','pass_count','fail','error','skip')}for k,v in result.items()}))
if __name__=='__main__':main()
