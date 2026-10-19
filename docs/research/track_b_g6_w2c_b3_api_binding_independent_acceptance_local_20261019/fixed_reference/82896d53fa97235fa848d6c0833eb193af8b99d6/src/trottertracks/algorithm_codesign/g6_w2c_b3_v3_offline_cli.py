"""Explicit fixed artificial v3 input -> fresh offline reports; no producer import."""
import argparse
import csv
import json
from pathlib import Path
from . import g6_w2c_b3_v3_offline_adapter as A
from . import g6_w2c_b3_v3_offline_cost as C
from . import g6_w2c_b3_v3_offline_acceptance as H
from . import g6_w2c_b3_offline_result as V
from .g6_w2c_b3_offline_cli import write_json


def run(manifest,reference,output):
    manifest=Path(manifest).resolve();reference=Path(reference).resolve();output=Path(output).resolve()
    for root in (manifest.parent,reference):
        V.require(not output.is_relative_to(root) and not root.is_relative_to(output),'OUTPUT_OUTSIDE_INPUT_REFERENCE')
    V.require(not output.exists(),'NO_OVERWRITE')
    v=A.verify(manifest,reference);c=C.aggregate(v);h=H.matrix(v,c)
    output.mkdir(parents=True,exist_ok=False)
    for name,obj in (('verification_report.json',v),('cost_summary.json',c),('integration_acceptance_report.json',h)):
        write_json(output/name,obj)
    (output/'verification_report.md').write_text('Status: **'+v['status']+'**. Artificial only.\n\n'
        'Published HEAD and measured SOURCE are distinct. Statefile is not operation history. '
        'Physical rows are not engine commits. Missing private authority and historical runtime bytes remain pending.\n')
    (output/'cost_comparison.md').write_text('Cost: **'+c['status']+'**.\n\n'
        'Total CPU = host parent + all workers. Pair CPU, IPC, launch and fine observer partitions are already contained. '
        'Global wall is outer wall. Unknown stays null. Failed prefixes are excluded; P/PF/S/SF conditions remain separate. No ranking.\n')
    (output/'integration_acceptance_report.md').write_text(
        '|ID|Judgment|Unresolved|\n|---|---|---|\n'+''.join(
        '|'+r['check_id']+'|'+r['judgment']+'|'+str(r['unresolved']).replace('|','\\|')+'|\n' for r in h['rows']))
    fields=['task_key','phase','pass_name','arm','layer','outer_cpu_wall_ns','setup_cpu_wall_ns',
            'online_cpu_wall_ns','output_cpu_wall_ns','logical_completed','failed_prefix']
    with (output/'comparison_table.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore',lineterminator='\n');w.writeheader()
        if c.get('comparison_eligible'):
            w.writerows(r for r in c.get('pairs',[]) if not r['failed_prefix'])
    return v,c,h


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artificial-manifest',required=True,type=Path)
    p.add_argument('--fixed-reference',required=True,type=Path)
    p.add_argument('--new-output',required=True,type=Path)
    a=p.parse_args(argv);v,c,h=run(a.artificial_manifest,a.fixed_reference,a.new_output)
    print(json.dumps(dict(verification=v['status'],cost=c['status'],pending=h['pending'],mandatory_STOP=True)))
    return 0 if v['status'] in ('VERIFIED_COMPLETE','TECHNICAL_FAILURE') and c['status']!='COST_INTEGRITY_MISMATCH' else 2


if __name__=='__main__':raise SystemExit(main())
