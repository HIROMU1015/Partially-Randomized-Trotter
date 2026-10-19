"""Independent API static + saved port tests; frozen Bundle is explicitly separate."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,sys,os,importlib.abc
HERE=Path(__file__).resolve().parent;W=HERE.parents[2];H='82896d53fa97235fa848d6c0833eb193af8b99d6';SC='76ca3c4089b201d55da0fede08e1f77733ac38a5'
Q=HERE/'fixed_reference'/H/'docs/research/track_b_g6_w2c_b3_real_api_binding_preflight_candidate_20261019'
sys.dont_write_bytecode=True;sys.path.insert(0,str(W/'src'))
FORBIDDEN={'g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_generator','g6_f3_precision','g6_w2c_b3_driver','g6_w2c_b3_parallel_driver','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_bundle_bridge_v1','g6_w2c_b3_bundle_store_v1'}
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.rsplit('.',1)[-1] in FORBIDDEN or '.g6_w2c_b3_api_binding_v1' in fullname:raise ImportError('INDEPENDENT_CONSUMER_FORBIDDEN:'+fullname)
sys.meta_path.insert(0,Block())
from trottertracks.algorithm_codesign.g6_w2c_b3_api_independent_v1 import adapter as A,static as S,contract as K
from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_identity_v1 import Reference
from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_independent_adapter_v1 as B
C=K.read(Q/'technical_contract_candidate.json')

def one(item):
 group,path,logical=item;path=Path(path)
 if group=='port':r=A.verify(path,C,HERE/'fixed_reference'/H,trusted=C,execution_root=logical)
 elif group=='port_mutation':r=A.verify(path/'case',K.read(path/'trusted_candidate.json'),HERE/'fixed_reference'/H,trusted=C,execution_root=logical)
 else:
  tree=W/'docs/research/track_b_g6_w2c_b3_global_bundle_independent_acceptance_local_20261019/fixed_reference/bridge'
  ref=Reference(tree,logical,allow_runtime_projection=True,verify_local_runtime=False);r=B.verify(path,ref)
 assert not [n for n in sys.modules if n.rsplit('.',1)[-1] in FORBIDDEN]
 return group+'/'+path.name,r

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-root',required=True);p.add_argument('--new-output',required=True);a=p.parse_args();raw=Path(a.raw_root).resolve();out=Path(a.new_output).resolve()
 assert not out.exists() and not out.is_relative_to(raw) and not raw.is_relative_to(out);out.mkdir()
 iv=S.inventory(HERE/'fixed_reference'/SC);check=S.compare(iv,K.read(Q/'scientific_api_inventory.json'));mapping=S.verify_mapping(HERE/'fixed_reference'/SC,K.read(Q/'api_binding_manifest_candidate.json'))
 (out/'scientific_api_independent_inventory.json').write_text(json.dumps(iv,indent=2));(out/'static_mapping_verification.json').write_text(json.dumps(dict(inventory=check,mapping=mapping),indent=2))
 tasks=[]
 for name in ('sealed_small','sealed_n256','S_generator','P1_angle'):tasks.append(('port',str(raw/'port_cases'/name),K.read(Q/'acceptance'/(name+'_location.json'))['root']))
 for path in sorted((raw/'port_overlays').iterdir()):tasks.append(('port_mutation',str(path),K.read(path/'observation.json')['original_execution_root']))
 trusted=K.read(raw/'bundle_cases/normal_bundle_n256/candidate_input.json')
 for group,folder in [('bundle','bundle_cases'),('bundle_mutation','restored_bundle_overlays')]:
  tasks.extend((group,str(path),trusted) for path in sorted((raw/folder).iterdir()) if path.is_dir())
 jobs=min(len(tasks),len(os.sched_getaffinity(0)))
 with ProcessPoolExecutor(max_workers=jobs) as pool:rows=dict(pool.map(one,tasks))
 def exact_json(x):return {'exact_fraction':[str(x.numerator),str(x.denominator)]}
 (out/'verification.json').write_text(json.dumps(dict(results=rows,processes=jobs,forbidden_imports=[],science_imports=0,bundle_port_injected=False,scientific_GO=False,mandatory_STOP=True),indent=2,default=exact_json))
 expected={'sealed_small':'VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE','sealed_n256':'VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE','S_generator':'VERIFIED_ARTIFICIAL_PORT_TECHNICAL_FAILURE','P1_angle':'VERIFIED_ARTIFICIAL_PORT_TECHNICAL_FAILURE'}
 assert all(rows['port/'+n]['status']==v for n,v in expected.items())
 assert all(r['status'] in ('INTEGRITY_MISMATCH','INCOMPLETE_EVIDENCE') for n,r in rows.items() if n.startswith(('port_mutation/','bundle_mutation/')))
 print(json.dumps(dict(cases=len(rows),processes=jobs,static_entries=iv['entries'],scientific_GO=False,bundle_port_injected=False)))
if __name__=='__main__':main()
