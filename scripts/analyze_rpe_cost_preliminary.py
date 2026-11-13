#!/usr/bin/env python3
"""Explicit entry for the separately authorized, fixed saved-result P0--P4 scope."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_cost_preliminary as analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    # New output only; original evidence/STOP is never edited.
    args.output.mkdir(parents=True,exist_ok=False)
    result=analysis.analyze(ROOT)
    result['analysis_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    encoded=analysis.json_values(result)
    (args.output/'RESULT.json').write_text(json.dumps(encoded,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (args.output/'ROUND_CONTRIBUTIONS.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['m','H_nominal_structural','share','share_lower','share_upper'])
        for row in result['P0_P1']['rounds']:
            writer.writerow([row['m']]+[None if v is None else float(v) for v in
                (row['H'],row['share'],*row['share_enclosure'])])
    with (args.output/'BREAK_EVEN.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['label','m','axis','PF_C','PR_S','theta','status'])
        for row in result['P2']:
            writer.writerow([row['label'],row['m'],row['axis']]+[None if row[k] is None else float(row[k]) for k in ('PF_C','PR_S','theta')]+[row['status']])
    with (args.output/'PILOT_RATIOS.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['m','axis','cohort','C_mean','S_mean','D_mean','C_over_S','D_over_S','qualification'])
        for row in result['P3']:
            writer.writerow([row['m'],row['axis'],row['cohort']]+[None if row[k] is None else float(row[k])
                for k in ('C_mean','S_mean','D_mean','C_over_S','D_over_S')]+[row['qualification']])
    with (args.output/'SIGNAL_INVENTORY.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['family','label','m','physical_candidate_id',
            'raw_real','raw_imag','corrected_real','corrected_imag','log_B','saved_raw_radius','saved_phase_to_reference','qualification'])
        for row in result['P4']:
            record=row.get('saved_record',row.get('saved_mean',{})); signal=record.get('signal',record);local=record.get('local',{})
            raw=signal.get('raw') or {};corrected=signal.get('corrected') or {}
            log_b=signal.get('log_B',(signal.get('normalization') or {}).get('log_B'))
            writer.writerow([row['family'],row.get('label'),row['m'],row['physical_candidate_id'],
                raw.get('real'),raw.get('imag'),corrected.get('real'),corrected.get('imag'),log_b,
                local.get('raw_radius'),local.get('corrected_phase_to_reference'),
                record.get('qualification','SAVED_METADATA_NOT_GLOBAL_RMSE_QUALIFIED')])
    print('Completed fixed P0-P4 once. Formal G/winner remain null. STOP.')


if __name__=='__main__':main()
