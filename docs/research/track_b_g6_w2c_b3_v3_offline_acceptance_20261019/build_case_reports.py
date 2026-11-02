"""Re-read the 47 fixed artificial archives; require fresh owned report outputs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import csv,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.g6_w2c_b3_v3_offline_cli import run
from trottertracks.algorithm_codesign.g6_w2c_b3_offline_cli import write_json
D=Path(__file__).resolve().parent

def one(path):
    name=Path(path).parts[1]
    v,c,h=run(D/path,D/'fixed_reference',D/'offline_reports'/name)
    return dict(case=name,status=v['status'],issues=v['issues'],cost_status=c['status'],cost_issues=c.get('issues'),
        calls=v.get('calls'),output_budget=v.get('output_budget'),control=v.get('control'),
        binding_evidence=v.get('binding_evidence'),acceptance=h,
        host_total_cpu_ns=c.get('total_cpu_ns'),host_wall_ns=c.get('global_wall_ns'),variant=c.get('variant'),
        comparison_eligible=c.get('comparison_eligible'),execution_condition=c.get('execution_condition'))

if __name__=='__main__':
    with ThreadPoolExecutor() as pool:rows=list(pool.map(one,json.loads((D/'published_cases.json').read_text())))
    write_json(D/'published_case_verification.json',rows)
    matrix=[]
    for i in range(1,12):
        cid='P'+str(i).zfill(2);checks=[r['acceptance']['rows'][i-1] for r in rows]
        matrix.append(dict(check_id=cid,counts=dict(Counter(c['judgment'] for c in checks)),
                           pending_cases=[r['case'] for r,c in zip(rows,checks) if c['judgment']=='PENDING_EVIDENCE'],
                           scientific_GO=False))
    write_json(D/'integration_acceptance_matrix.json',dict(schema='b3_v3_offline_matrix_summary_v1',cases=47,rows=matrix,mandatory_STOP=True))
    with (D/'artificial_process_costs.csv').open('x',newline='') as f:
        fields=['case','variant','status','cost_status','host_total_cpu_ns','host_wall_ns','comparison_eligible','execution_condition']
        w=csv.DictWriter(f,fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(cases=len(rows),status=dict(Counter(r['status'] for r in rows)),cost_status=dict(Counter(r['cost_status'] for r in rows)))))
