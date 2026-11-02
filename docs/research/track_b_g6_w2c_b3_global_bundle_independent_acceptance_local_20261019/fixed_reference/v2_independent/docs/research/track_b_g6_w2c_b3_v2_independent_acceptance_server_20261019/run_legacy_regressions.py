"""Versioned regression harness; original V1 SOURCE/tests remain untouched.

One historical assertion expected co_name while frozen observers use co_qualname.
Only that literal in the one test's code object is corrected, explicitly logged.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
old=ROOT/'docs/research/track_b_g6_w2c_b3_limited_pilot_independent_acceptance_20261019/run_independent_tests.py'
values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(old.read_text()).body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('CODE','names')}
CORRECTION='''
import types
corrected=[]
if n=='g6_w2c_b3_parallel_offline_verifier':
 def fix(code):
  return code.replace(co_consts=tuple(fix(x) if isinstance(x,types.CodeType) else '.<locals>.leaf' if x==':leaf' else x for x in code.co_consts))
 test=m.CandidateAcceptance.test_cached_observer_call_count_label_exclusive_partition
 test.__code__=fix(test.__code__)
 corrected=[test.__qualname__+': assertion suffix :leaf -> .<locals>.leaf (co_qualname)']
'''
CODE=values['CODE'].replace('suite=unittest.defaultTestLoader',CORRECTION+'\nsuite=unittest.defaultTestLoader').replace("report=dict(collected=", "report=dict(explicit_versioned_test_corrections=corrected,collected=")
def main():
 dest=HERE/('legacy_corrected_'+sys.argv[1]);dest.mkdir(exist_ok=False)
 def one(n):
  p=subprocess.run([sys.executable,'-c',CODE,n,str(dest/(n+'.json'))],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  (dest/(n+'.log')).write_bytes(p.stdout)
  row=json.loads((dest/(n+'.json')).read_text());row.update(suite=n,exit_code=p.returncode,log_sha256=hashlib.sha256(p.stdout).hexdigest());return row
 with ThreadPoolExecutor(max_workers=len(values['names'])) as pool:rows=list(pool.map(one,values['names']))
 report=dict(schema='b3_v2_versioned_legacy_regression_harness_v1',original_runner_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),original_tests_modified=False,total=sum(x['run'] for x in rows),pass_count=sum(x['pass_count'] for x in rows),suites=rows,successful=all(x['successful'] and x['exit_code']==0 for x in rows))
 (dest/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='suites'}));assert report['successful']
if __name__=='__main__':main()
