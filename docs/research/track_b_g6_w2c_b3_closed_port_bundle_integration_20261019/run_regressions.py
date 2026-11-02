"""Read fixed worktrees; write results only into this new packet and own temp."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CODE=r'''
from pathlib import Path
import sys,unittest,importlib,json
sys.dont_write_bytecode=True;root=Path.cwd();sys.path[:0]=[str(root/'src'),str(root/'tests/tracks/algorithm_codesign')]
m=importlib.import_module(sys.argv[1]);s=unittest.defaultTestLoader.loadTestsFromModule(m)
def ids(s):
 for t in s:
  if isinstance(t,unittest.TestSuite):yield from ids(t)
  else:yield t.id()
names=list(ids(s));r=unittest.TextTestRunner(verbosity=2).run(s)
bad=[n for n in sys.modules if n.rsplit('.',1)[-1]in ('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator')]
Path(sys.argv[2]).write_text(json.dumps(dict(run=r.testsRun,PASS=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),FAIL=len(r.failures),ERROR=len(r.errors),SKIP=len(r.skipped),ids=names,forbidden_imports=bad),indent=2)+'\n')
'''
def main():
    dest=HERE/'regressions';dest.mkdir(exist_ok=False)
    jobs=[('bundle107','g6-w2c-b3-v2-production-bundle-bridge-candidate-20261019','test_g6_w2c_b3_bundle_v1'),
          ('api73','g6-w2c-b3-real-api-binding-preflight-candidate-20261019','test_g6_w2c_b3_api_binding_v1'),
          ('storage41','g6-w2c-b3-trace-reservation-accounting-repair-20261019','test_g6_w2c_b3_storage_v2'),
          ('independent57','g6-w2c-b3-v2-independent-acceptance-server-20261019','test_g6_w2c_b3_storage_independent_v2')]
    def one(job):
        label,folder,module=job;root=ROOT.parent/folder
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',B3_BUNDLE_MUTATION_DIR=str(dest/(label+'_overlays')))
        env.pop('B3_NEGATIVE_EVIDENCE_DIR',None);env.pop('B3_BUNDLE_MATRIX',None)
        if label=='bundle107':
            env['B3_BUNDLE_MATRIX']=str(root/'docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/runs_sealed')
        p=subprocess.run([sys.executable,'-c',CODE,module,str(dest/(label+'.json'))],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (dest/(label+'.log')).write_bytes(p.stdout);r=json.loads((dest/(label+'.json')).read_text())
        r.update(label=label,source_root=str(root),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
        print(label,r['PASS'],r['FAIL'],r['ERROR'],flush=True);return r
    with ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(one,jobs))
    (dest/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert all(x['run']==x['PASS']and not x['forbidden_imports']for x in rows)
if __name__=='__main__':main()
