"""Public saved artificial raw -> independent V2 reports, no execution authority."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
import json,os,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import g6_w2c_b3_storage_independent_identity_v2 as I
from trottertracks.algorithm_codesign import g6_w2c_b3_storage_independent_adapter_v2 as A
REFERENCE=None

def reference():
    global REFERENCE
    if REFERENCE is None:
        candidate_path=HERE/'restored/fault_cases_final/S_normal/pilot_contract.json'
        public_manifest=json.loads((HERE/'public_bytes'/I.PREFIX/'fault_cases_manifest.json').read_text())
        pin=next(r for r in public_manifest if r['path']=='fault_cases_final/S_normal/pilot_contract.json')
        assert A.V.file_digest(candidate_path)==pin['sha256']
        c=json.loads(candidate_path.read_text())
        REFERENCE=I.Reference(HERE/'public_bytes',ROOT/'docs/research/track_b_g6_w2c_b3_limited_pilot_independent_acceptance_20261019/fixed_reference/tree',ROOT/'docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/fixed_reference',c)
    return REFERENCE

def one(args):
    case,dest=args;r=A.verify(case,reference())
    forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1] in {'g6_w2c_b3_pilot_candidate','g6_w2c_b3_pilot_storage_v2','g6_w2c_b3_storage_verify_v2','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_admission','g6_w2c_b3_parallel_admission','g6_f3_generator','g6_f3_contract','g6_f3_precision','g7_generator','g7_provider','return_aggregation','g9_native'}]
    assert not forbidden,forbidden
    r['forbidden_imports']=forbidden
    (dest/(case.name+'.json')).write_text(json.dumps(r,indent=2,default=lambda x:{'exact':A.V.canonical(x)})+'\n')
    obs=json.loads((case/'observations.json').read_text())
    return dict(case=case.name,status=r['status'],issues=r['issues'],producer_status=obs['verification']['status'],classification_equal=r['status']==obs['verification']['status'],physical_state_independently_verified=r.get('state') is not None,evidence_gaps=r.get('evidence_gaps',[]),cost_status=r.get('cost',{}).get('status'),forbidden_imports=forbidden)

def main():
    dest=HERE/('acceptance_'+sys.argv[1]);dest.mkdir(exist_ok=False)
    cases=sorted(p for p in (HERE/'restored/fault_cases_final').iterdir() if p.is_dir())
    started=time.monotonic();workers=min(len(cases),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(one,[(c,dest) for c in cases]))
    matrix=dict(schema='b3_v2_independent_108_case_matrix_v1',producer_public_HEAD=I.PUBLIC,measured_SOURCE=I.MEASURED,workers=workers,wall_seconds=time.monotonic()-started,counts=dict(Counter(r['status'] for r in rows)),cases=rows,physical_state_exact_cases=sum(r['physical_state_independently_verified'] for r in rows),producer_self_checker_used=False,forbidden_imports=[],real_scientific_calls=0,mandatory_STOP=True)
    (dest/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    print(json.dumps({k:v for k,v in matrix.items() if k!='cases'}))
    print(json.dumps([r for r in rows if r['issues'] or not r['classification_equal']],indent=2))
if __name__=='__main__':main()
