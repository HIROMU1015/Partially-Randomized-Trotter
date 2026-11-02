"""Re-read fixed public 18 cases and two separately labeled local replays."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[3];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_independent_identity as I
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_independent_adapter as A
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_independent_cli as C
from trottertracks.algorithm_codesign.g6_w2c_b3_offline_cli import write_json
D=Path(__file__).resolve().parent;TREE=D/'fixed_reference/tree';LEGACY=ROOT/'docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/fixed_reference'

def one(case):
 v,h=C.run(case,TREE,LEGACY,D/'case_reports'/case.name)
 return dict(case=case.name,status=v['status'],issues=v['issues'],calls=v.get('calls'),trace=v.get('trace'),
             output_budget=v.get('output_budget'),admission=v.get('admission'),identity=v.get('identity'),
             cost_status=v.get('cost',{}).get('status'),cost_issues=v.get('cost',{}).get('issues'),acceptance=h)

if __name__=='__main__':
 with ThreadPoolExecutor() as pool:rows=list(pool.map(one,sorted((TREE/I.PREFIX/'forward_evidence').iterdir())))
 write_json(D/'artificial_case_verification.json',dict(schema='pilot_independent_fixed_cases_v1',cases=rows,
     counts=dict(Counter(r['status'] for r in rows)),scope='18 original published cases only',historical_pending_reclassified=False))
 fresh=[]
 for p in sorted((D/'artificial_replays').iterdir()):
  ref=I.Reference(TREE,LEGACY);c=json.loads((p/'pilot_contract.json').read_text());normalized=deepcopy(c)
  for key in ('runtime_root','python'):normalized[key]=ref.template[key]
  assert normalized==ref.template;ref.template=c
  r=A.verify(p,ref);r['scope']='SEPARATE_NEW_LOCAL_ARTIFICIAL_REPLAY';r['permitted_identity_difference']=['runtime_root','python'];fresh.append(r)
 write_json(D/'fresh_source_replay_verification.json',fresh)
 write_json(D/'pilot_scope_accounting.json',I.scope_plans())
 print(json.dumps(dict(public_cases=dict(Counter(r['status'] for r in rows)),fresh=[dict(case=r['case'],status=r['status'],issues=r['issues']) for r in fresh])))
