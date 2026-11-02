"""Fresh-process artificial regressions; source and original counts kept separate."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
NAMES=['g6_w2c_b3_admission','g6_w2c_b3_readiness','g6_w2c_b3_driver','g6_w2c_cost_preparation',
       'g6_w2c_b3_f01_provenance','g6_w2c_b3_host_integration','g6_w2c_b3_parallel_candidate','g6_w2c_b3_parallel_overhead']
CODE='''import importlib,json,pathlib,sys,unittest
sys.dont_write_bytecode=True
sys.path[:0]=[str(pathlib.Path('src').resolve()),str(pathlib.Path('tests/tracks/algorithm_codesign').resolve())]
m=importlib.import_module('test_'+sys.argv[1]);suite=unittest.defaultTestLoader.loadTestsFromModule(m)
def ids(s):
 for x in s:
  if isinstance(x,unittest.TestSuite):yield from ids(x)
  else:yield x.id()
collected=list(ids(suite));r=unittest.TextTestRunner(verbosity=2).run(suite)
bad=[x for x in sys.modules if x.rsplit('.',1)[-1] in ('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator','g9_native')]
value=dict(collected=collected,run=r.testsRun,fail=len(r.failures),error=len(r.errors),skip=len(r.skipped),science_imports=bad,successful=r.wasSuccessful() and not bad)
pathlib.Path(sys.argv[2]).write_text(json.dumps(value,indent=2)+'\\n');sys.exit(0 if value['successful'] else 1)
'''


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    def one(name):
        file=out/(name+'.json')
        p=subprocess.run([sys.executable,'-c',CODE,name,str(file.resolve())],cwd=ROOT,stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        (out/(name+'.log')).write_bytes(p.stdout)
        row=json.loads(file.read_text());row.update(suite=name,exit_code=p.returncode,log_sha256=sha256(p.stdout).hexdigest())
        print(json.dumps({k:row[k] for k in ('suite','run','fail','error','skip')}),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(one,NAMES))
    assert all(r['successful'] and r['exit_code']==0 for r in rows)
    assert sum(r['run'] for r in rows[:7])==311
    value=dict(existing=311,additional=sum(r['run'] for r in rows[7:]),total=sum(r['run'] for r in rows),fail=0,skip=0,suites=rows)
    (out/'summary.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
