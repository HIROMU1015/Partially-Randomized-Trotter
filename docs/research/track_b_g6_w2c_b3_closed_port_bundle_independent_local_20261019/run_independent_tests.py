import pathlib,json,subprocess,concurrent.futures,os
import argparse,ast
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--raw-root',required=True);ap.add_argument('--new-output',required=True);a=ap.parse_args();raw=pathlib.Path(a.raw_root).resolve();dest=pathlib.Path(a.new_output).resolve();assert not dest.exists() and not dest.is_relative_to(raw) and not raw.is_relative_to(dest);dest.mkdir()
code='''import sys,json,pathlib,unittest,importlib.abc
sys.dont_write_bytecode=True;W=pathlib.Path(sys.argv[1]);sys.path[:0]=[str(W/'src'),str(W/'tests/tracks/algorithm_codesign')]
forbidden={'g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator','g6_w2c_b3_driver','g6_w2c_b3_parallel_driver','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_bundle_bridge_v1','g6_w2c_b3_bundle_store_v1'}
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.rsplit('.',1)[-1] in forbidden or '.g6_w2c_b3_api_binding_v1' in fullname or '.g6_w2c_b3_port_bundle_v1' in fullname:raise ImportError('INDEPENDENT_FORBIDDEN_IMPORT:'+fullname)
sys.meta_path.insert(0,Block())
import test_g6_w2c_b3_port_bundle_independent_v1 as m
suite=unittest.defaultTestLoader.loadTestsFromName(sys.argv[2],m);ids=[x.id() for x in suite]
r=unittest.TextTestRunner(verbosity=2).run(suite);out=dict(suite=sys.argv[2],ids=ids,run=r.testsRun,PASS=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),FAIL=len(r.failures),ERROR=len(r.errors),SKIP=len(r.skipped),failures=[dict(test=t.id(),traceback=x) for t,x in r.failures],errors=[dict(test=t.id(),traceback=x) for t,x in r.errors],results=m.RESULTS,forbidden_imports=[n for n in sys.modules if n.rsplit('.',1)[-1] in forbidden],scientific_calls=0)
pathlib.Path(sys.argv[3]).write_text(json.dumps(out,indent=2))
'''
def one(name):
 p=subprocess.run(['/usr/bin/python3','-c',code,str(W),name,str(dest/(name+'.json'))],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',B3_PORT_BUNDLE_RAW=str(raw),B3_PORT_BUNDLE_TEST_OUTPUT=str(dest/'fixtures'/name)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(dest/(name+'.log')).write_bytes(p.stdout)
 r=json.loads((dest/(name+'.json')).read_text());print({k:r[k] for k in ('suite','run','PASS','FAIL','ERROR')},flush=True);return r
source=ast.parse((W/'tests/tracks/algorithm_codesign/test_g6_w2c_b3_port_bundle_independent_v1.py').read_text())
keys=next(x.value.keys for x in source.body if isinstance(x,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='ATTACKS' for v in x.targets))
classes=['RepairedPortAttacks.test_'+k.value for k in keys]+['Contract','Components','SavedEvidence']
with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(classes),len(os.sched_getaffinity(0)))) as ex:rows=list(ex.map(one,classes))
(dest/'summary.json').write_text(json.dumps(dict(suites=rows,totals={k:sum(x[k] for x in rows) for k in ('run','PASS','FAIL','ERROR','SKIP')}),indent=2))
