"""Producer-free archived Bundle × Port verification; fresh output only."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,os,json,argparse,importlib.abc
P=Path(__file__).resolve().parent;W=P.parents[2];H='62967269e611b61e73b70abe20e998d9c7eaaf50';DOC='docs/research/track_b_g6_w2c_b3_closed_port_bundle_integration_20261019';Q=P/'fixed_reference'/H/DOC
sys.dont_write_bytecode=True;sys.path.insert(0,str(W/'src'))
FORBIDDEN={'g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_parallel_admission','g6_w2c_b3_admission','g6_w2c_b3_bundle_bridge_v1','g6_w2c_b3_bundle_store_v1'}
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.rsplit('.',1)[-1] in FORBIDDEN or '.g6_w2c_b3_port_bundle_v1' in fullname or '.g6_w2c_b3_api_binding_v1' in fullname:raise ImportError('INDEPENDENT_FORBIDDEN:'+fullname)
sys.meta_path.insert(0,Block())
from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_independent_v1 import adapter as A
from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_independent_v1.identity import Reference,read
C=read(Q/'technical_contract_candidate.json');REF=Reference(P/'fixed_reference'/H,C)
def one(item):
 name,path,expected=item;r=A.verify(path,REF);assert not[n for n in sys.modules if n.rsplit('.',1)[-1] in FORBIDDEN];return name,dict(expected=expected,observed=r['status'],classification_equal=r['status']==expected,result=r)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw-root',required=True);ap.add_argument('--new-output',required=True);a=ap.parse_args();raw=Path(a.raw_root).resolve();out=Path(a.new_output).resolve();assert not out.exists() and not out.is_relative_to(raw) and not raw.is_relative_to(out);out.mkdir()
 m=read(Q/'artificial_acceptance_matrix.json');items=[('cases/'+r['case'],raw/'cases'/r['case'],r['verification_status']) for r in m['cases']['observations']]
 items += [('mutations/'+r['test_id'].rsplit('.',1)[-1],raw/'restored_overlays'/r['test_id'].rsplit('.',1)[-1],r['status']) for r in m['mutations']['observations']]
 items += [('additional/'+p.name,p/'case',read(p/'observation.json')['status']) for p in (raw/'additional').iterdir() if p.is_dir()]
 jobs=min(len(items),len(os.sched_getaffinity(0)))
 with ProcessPoolExecutor(max_workers=jobs) as pool:rows=dict(pool.map(one,items))
 (out/'verification.json').write_text(json.dumps(dict(results=rows,processes=jobs,scientific_GO=False,science_imports=0,mandatory_STOP=True),indent=2,default=lambda x:{'exact_fraction':[str(x.numerator),str(x.denominator)]}))
 assert all(r['classification_equal'] for r in rows.values());print(json.dumps(dict(cases=len(rows),all_equal=True,processes=jobs,scientific_GO=False)))
if __name__=='__main__':main()
