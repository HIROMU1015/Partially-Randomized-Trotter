"""Explicit artificial P1 manifest -> new independent verification/cost reports."""
import argparse
import csv
import json
from pathlib import Path
from . import g6_w2c_b3_parallel_offline_adapter as A
from . import g6_w2c_b3_parallel_offline_cost as C
from . import g6_w2c_b3_parallel_verifier_core as G
from . import g6_w2c_b3_offline_result as V
from .g6_w2c_b3_offline_cli import write_json


def run(manifest,output):
    manifest=Path(manifest).resolve();output=Path(output).resolve()
    V.require(not output.is_relative_to(manifest.parent)and not manifest.parent.is_relative_to(output),'OUTPUT_OUTSIDE_INPUT_REQUIRED')
    V.require(not output.exists(),'NO_OVERWRITE')
    v=A.verify(manifest);c=C.aggregate(v)
    checks=v['checks']+[G.check('P09','Separate parent/all-worker CPU and global wall, alternative partitions',c['status'],
                              'PASS'if c['status']=='VERIFIED_PARALLEL_COST_BOUNDARIES'else'PENDING_EVIDENCE',V.CLAIM,
                              'Measurement boundary comes from fixed producer SOURCE; no ranking/estimand adoption.',commit=A.PUBLISHED)]
    acceptance=dict(schema='b3_parallel_acceptance_v1',checks=checks,final_server_schema='PENDING_SERVER_SCHEMA',
                    scientific_adoption=False,totals={k:sum(x['judgment']==k for x in checks)for k in('PASS','FAIL','PENDING_EVIDENCE','REQUIRES_CONTRACT_CHANGE','NOT_APPLICABLE')})
    output.mkdir(parents=True,exist_ok=False)
    for name,obj in(('verification_report.json',v),('cost_summary.json',c),('integration_acceptance_report.json',acceptance)):write_json(output/name,obj)
    (output/'verification_report.md').write_text('# Independent artificial P1 verification\n\nStatus: **'+v['status']+'**.\n\nRun: '+str(v.get('run_id'))+'\n\n'
        'Latest state is not transition history. START_RESERVED is not actual function entry. '
        'P08 remains pending without ordered parent gate operations. Visible JSONL rows and logical COMPLETED are distinct. '
        'Compact hashes do not reconstruct full trace bytes. Future integrated schema: PENDING_SERVER_SCHEMA.\n')
    fields=['task_key','phase','pass_name','arm','layer','outer_cpu_wall_ns','setup_cpu_wall_ns','online_cpu_wall_ns','output_cpu_wall_ns','logical_completed','failed_prefix','output_bytes','output_rows']
    with(output/'comparison_table.csv').open('x',newline='')as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows([r for r in c['pairs']if c['comparison_eligible']and not r['failed_prefix']])
    (output/'cost_comparison.md').write_text('# Parallel cost boundaries\n\nStatus: **'+c['status']+'**.\n\n'
        'Parent CPU + all-worker CPU is total CPU at the same declared host boundary. '
        'Pair CPU, launch, IPC and finalization are already contained and are not added again. '
        'Global wall is the outer wall, not the sum of overlapping walls. Coarse/fine are alternative partitions. '
        'Failure prefixes are labeled and excluded from normal scientific comparisons. Unknown costs stay null. '
        'Parallel concurrent and serial execution are different conditions; no scientific ranking is adopted.\n')
    lines=['# Parallel integration acceptance\n','Future integrated schema: PENDING_SERVER_SCHEMA.\n','|ID|Judgment|Evidence|Unresolved|','|---|---|---|---|']
    for r in checks:lines.append('|'+ '|'.join(str(r[k]).replace('|','\\|').replace('\n',' ')for k in('check_id','judgment','evidence','unresolved'))+'|')
    (output/'integration_acceptance_report.md').write_text('\n'.join(lines)+'\n')
    return v,c,acceptance


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artificial-manifest',required=True,type=Path);p.add_argument('--new-output',required=True,type=Path)
    a=p.parse_args(argv);v,c,_=run(a.artificial_manifest,a.new_output)
    print(json.dumps(dict(verification=v['status'],cost=c['status'],mandatory_STOP=True)))
    return 0 if v['status']in('VERIFIED_COMPLETE','TECHNICAL_FAILURE')else 2


if __name__=='__main__':raise SystemExit(main())
