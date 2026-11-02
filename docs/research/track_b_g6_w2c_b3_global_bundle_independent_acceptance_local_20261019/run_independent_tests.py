"""Run165 independent tests; public43/35/fresh8 are subsets, not extra counts."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,os,sys
import argparse
p=argparse.ArgumentParser();p.add_argument('--public-raw',required=True);p.add_argument('--fresh-raw',required=True);p.add_argument('--new-output',required=True);p.add_argument('--verify-current-fresh-runtime',action='store_true');a=p.parse_args()
HERE=Path(__file__).resolve().parent;W=HERE.parents[2];T=Path(a.public_raw).resolve().parent;public=Path(a.public_raw).resolve();fresh=Path(a.fresh_raw).resolve();out=Path(a.new_output).resolve()
assert not out.exists() and not out.is_relative_to(public) and not out.is_relative_to(fresh)
out.mkdir();mut=out/'mutation_overlays';mut.mkdir()
code=r'''from pathlib import Path
import importlib,importlib.abc,sys,unittest,json
sys.dont_write_bytecode=True;root=Path(sys.argv[1]);sys.path[:0]=[str(root/'src'),str(root/'tests/tracks/algorithm_codesign')]
forbidden={'g7_generator','g7_provider','g6_f3_generator','g6_f3_contract','g6_f3_precision','return_aggregation','g9_native','g6_w2c_b3_bundle_bridge_v1','g6_w2c_b3_bundle_offline_v1','g6_w2c_b3_bundle_contract_v1','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_pilot_storage_v2','g6_w2c_b3_admission','g6_w2c_b3_parallel_admission'}
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.rsplit('.',1)[-1]in forbidden:raise ImportError('FORBIDDEN_INDEPENDENT_IMPORT:'+fullname)
sys.meta_path.insert(0,Block());m=importlib.import_module('test_g6_w2c_b3_bundle_independent_v1');cls=getattr(m,sys.argv[2]);suite=unittest.defaultTestLoader.loadTestsFromTestCase(cls)
def ids(s):
 for x in s:
  if isinstance(x,unittest.TestSuite):yield from ids(x)
  else:yield x.id()
collected=list(ids(suite));r=unittest.TextTestRunner(verbosity=2).run(suite);bad=[n for n in sys.modules if n.rsplit('.',1)[-1]in forbidden]
report=dict(suite=sys.argv[2],collected=collected,run=r.testsRun,PASS=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),FAIL=len(r.failures),ERROR=len(r.errors),SKIP=len(r.skipped),failures=[dict(test=t.id(),traceback=x)for t,x in r.failures],errors=[dict(test=t.id(),traceback=x)for t,x in r.errors],forbidden_imports=bad,results=m.RESULTS,successful=r.wasSuccessful()and not bad)
Path(sys.argv[3]).write_text(json.dumps(report,indent=2)+'\n')
'''
def one(kind):
 p=subprocess.run(['/usr/bin/python3','-c',code,str(W),kind,str(out/(kind+'.json'))],cwd=W,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',B3_BUNDLE_RAW_ROOT=str(public),B3_BUNDLE_FRESH_ROOT=str(fresh),B3_BUNDLE_VERIFY_LIVE_RUNTIME='1'if a.verify_current_fresh_runtime else'0',B3_BUNDLE_MUTATION_EVIDENCE=str(mut),B3_BUNDLE_MEMORY_REPORT=str(out/'streaming_memory.json')),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (out/(kind+'.log')).write_bytes(p.stdout)
 if not(out/(kind+'.json')).exists():return dict(suite=kind,exit_code=p.returncode,output=p.stdout.decode()[-4500:])
 r=json.loads((out/(kind+'.json')).read_text());print(json.dumps({k:r[k]for k in ('suite','run','PASS','FAIL','ERROR','SKIP')}),flush=True);return r
classes=['PublicCases','PublicOverlays','FreshCases','Mutations','StaticContract','CLI','Memory']
with ThreadPoolExecutor(max_workers=len(classes))as pool:rows=list(pool.map(one,classes))
(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(output=str(out),totals={k:sum(r.get(k,0)for r in rows)for k in ['run','PASS','FAIL','ERROR','SKIP']},failures=[dict(suite=r['suite'],failures=r.get('failures'),errors=r.get('errors')) for r in rows if not r.get('successful',False)]),default=str))
