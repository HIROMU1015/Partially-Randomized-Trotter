"""Reproduce the ten artificial suites only; logs in a fresh owned /tmp directory."""
import concurrent.futures,hashlib,json,os,pathlib,subprocess,sys
WT=pathlib.Path(__file__).resolve().parents[3]
import tempfile
TMP=pathlib.Path(tempfile.mkdtemp(prefix='b3-limited-independent-tests-'))
names=['g6_w2c_b3_admission','g6_w2c_b3_readiness','g6_w2c_b3_driver','g6_w2c_cost_preparation','g6_w2c_b3_f01_provenance','g6_w2c_b3_host_integration','g6_w2c_b3_offline_verifier','g6_w2c_b3_parallel_offline_verifier','g6_w2c_b3_v3_offline_verifier','g6_w2c_b3_pilot_independent']
print('Artificial test logs:',TMP,flush=True)
CODE='''import importlib,json,pathlib,sys,unittest,hashlib
sys.dont_write_bytecode=True
sys.path.insert(0,str(pathlib.Path('tests/tracks/algorithm_codesign').resolve()))
sys.path.insert(0,str(pathlib.Path('src').resolve()))
n=sys.argv[1];m=importlib.import_module('test_'+n)
suite=unittest.defaultTestLoader.loadTestsFromModule(m)
def ids(s):
 for x in s:
  if isinstance(x,unittest.TestSuite):yield from ids(x)
  else:yield x.id()
collected=list(ids(suite));r=unittest.TextTestRunner(verbosity=2).run(suite)
forbidden=[x for x in sys.modules if x.startswith('trottertracks.algorithm_codesign.') and x.rsplit('.',1)[-1] in ('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator','g9_native')]
report=dict(collected=collected,run=r.testsRun,pass_count=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),fail=len(r.failures),error=len(r.errors),skip=len(r.skipped),forbidden_science_modules_imported=forbidden,successful=r.wasSuccessful() and not forbidden)
bad={t.id() for t,_ in r.failures+r.errors+r.skipped}
report['failed_ids']=[t.id() for t,_ in r.failures+r.errors]
report['passed_ids']=[x for x in collected if x not in bad] if not forbidden else []
report['source_path']='tests/tracks/algorithm_codesign/test_'+n+'.py'
report['source_sha256']=hashlib.sha256(pathlib.Path(report['source_path']).read_bytes()).hexdigest()
pathlib.Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\\n');sys.exit(0 if report['successful'] else 1)
'''
def one(n):
 report=TMP/(n+'.json');cmd=['/usr/bin/python3','-c',CODE,n,str(report)]
 p=subprocess.run(cmd,cwd=WT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',B3_OFFLINE_MEMORY_REPORT=str(TMP/'bounded_memory.json'),B3_PARALLEL_MEMORY_REPORT=str(TMP/'parallel_bounded_memory.json'),B3_V3_MEMORY_REPORT=str(TMP/'v3_bounded_memory.json'),B3_PILOT_MEMORY_REPORT=str(TMP/'pilot_bounded_memory.json'),B3_PILOT_GUARD_REPORT=str(TMP/'pilot_guard_limit.json')))
 log=TMP/('test_'+n+'.log');log.write_bytes(p.stdout)
 data=json.loads(report.read_text());data.update(suite=n,exit_code=p.returncode,log=log.name,log_sha256=hashlib.sha256(p.stdout).hexdigest())
 print(json.dumps({k:data[k] for k in ('suite','run','fail','error','skip','exit_code')}),flush=True)
 return data
with concurrent.futures.ThreadPoolExecutor() as pool:rows=list(pool.map(one,names))
(TMP/('results_'+str(len(names))+'_suites.json')).write_text(json.dumps(rows,indent=2)+'\n')
assert all(r['successful'] and r['exit_code']==0 for r in rows)
