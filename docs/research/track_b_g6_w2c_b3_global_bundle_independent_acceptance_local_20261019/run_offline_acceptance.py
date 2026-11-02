"""Independent public43 + saved35 overlays + optional local8, fresh reports only."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import argparse,json,os,sys,time,importlib.abc
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];TREE=HERE/'fixed_reference/bridge';DOC='docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019'
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
FORBIDDEN={'g7_generator','g7_provider','g6_f3_contract','g6_f3_generator','g6_f3_precision','return_aggregation','g9_native','g6_w2c_b3_bundle_bridge_v1','g6_w2c_b3_bundle_contract_v1','g6_w2c_b3_bundle_fixture_v1','g6_w2c_b3_bundle_offline_v1','g6_w2c_b3_bundle_store_v1','g6_w2c_b3_bundle_trace_v1','g6_w2c_b3_pilot_candidate','g6_w2c_b3_pilot_storage_v2','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_admission','g6_w2c_b3_parallel_admission','g6_w2c_b3_host','g6_w2c_b3_parallel_host','g6_w2c_b3_driver','g6_w2c_b3_parallel_driver'}
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.rsplit('.',1)[-1]in FORBIDDEN:raise ImportError('INDEPENDENT_CONSUMER_FORBIDDEN_IMPORT:'+fullname)
sys.meta_path.insert(0,Block())
from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_independent_adapter_v1 as A
from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_identity_v1 import Reference,read
from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_cli_v1 import write_results

def one(item):
 group,path,dest,expected,live=item;path=Path(path)
 if group=='fresh':ref=Reference(TREE,read(path/'candidate_input.json'),allow_runtime_projection=True,verify_local_runtime=live)
 else:ref=Reference(TREE)
 r=A.verify(path,ref);bad=[n for n in sys.modules if n.rsplit('.',1)[-1]in FORBIDDEN];assert not bad;r['consumer_forbidden_imports']=bad
 write_results(r,path,Path(dest)/group/path.name)
 return dict(group=group,case=path.name,expected=expected,status=r['status'],classification_equal=r['status']==expected,issues=r['issues'],evidence_gaps=r['evidence_gaps'],calls=r.get('calls'),output_budget=r.get('output_budget'),comparison_eligible=r['comparison_eligible'],evidence=group+'/'+path.name+'/verification.json')
def main():
 p=argparse.ArgumentParser();p.add_argument('--public-raw',required=True);p.add_argument('--fresh-raw');p.add_argument('--new-output',required=True);p.add_argument('--verify-current-fresh-runtime',action='store_true');a=p.parse_args()
 public=Path(a.public_raw).resolve();out=Path(a.new_output).resolve();assert not out.exists()and not out.is_relative_to(public)and not public.is_relative_to(out)
 fresh=Path(a.fresh_raw).resolve()if a.fresh_raw else None
 if fresh:assert not out.is_relative_to(fresh)and not fresh.is_relative_to(out)
 out.mkdir(parents=True);items=[]
 for r in read(TREE/DOC/'artificial_acceptance_matrix.json')['cases']:items.append(('public',str(public/'cases'/r['case']),str(out),r['verification_status'],False))
 for r in read(TREE/DOC/'negative_mutation_results.json')['observations']:
  n=r['test_id'].rsplit('.',1)[-1];items.append(('public_overlays',str(public/'restored_mutations'/n),str(out),r['status'],False))
 if fresh:
  for path in sorted(p for p in fresh.iterdir()if p.is_dir()and p.name.startswith('local_')):
   expected='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'if 'normal_bundle'in path.name else'INCOMPLETE_EVIDENCE';items.append(('fresh',str(path),str(out),expected,a.verify_current_fresh_runtime))
 start=time.monotonic();jobs=min(len(items),len(os.sched_getaffinity(0)))
 with ProcessPoolExecutor(max_workers=jobs)as pool:rows=list(pool.map(one,items))
 result=dict(schema='local_independent_global_bundle_matrix_v1',cases=rows,groups={g:dict(Counter(r['status']for r in rows if r['group']==g))for g in ('public','public_overlays','fresh')},all_classifications_equal=all(r['classification_equal']for r in rows),process_parallelism=jobs,elapsed_seconds=time.monotonic()-start,producer_self_checker_used=False,scientific_calls=0,forbidden_imports=[],mandatory_STOP=True)
 (out/'matrix.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='cases'}))
 return 0 if result['all_classifications_equal']else 2
if __name__=='__main__':raise SystemExit(main())
