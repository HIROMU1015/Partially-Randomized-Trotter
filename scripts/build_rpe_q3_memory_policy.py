#!/usr/bin/env python3
"""Generate a separate memory policy from already accepted Q3 receipts (read only)."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.rpe_q3_memory import observation, imported_observation, envelopes, learning_policy, read


def observations(run):
    run=Path(run).resolve(); dbpath=run/'distributed/QUEUE.sqlite3'
    if dbpath.is_file():
        with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            for raw in db.execute("SELECT job_id,attempt_id,host_id,request_sha256,state,request,eligibility FROM jobs WHERE state='ACCEPTED' AND method NOT LIKE 'PREPARATION:%'"):
                row=dict(raw)
                try:
                    for key in ('request','eligibility'):row[key]=json.loads(row[key])
                    item=observation(row,run/'distributed/transfers'/row['job_id']/row['attempt_id'])
                    if item:yield item
                except (ValueError,TypeError,KeyError):continue
    pbs=run/'distributed/pbs'
    if (pbs/'STATE.json').is_file():
        from trottertracks.rpe_pbs_state import read_pbs_state
        allocations=read_pbs_state(pbs/'STATE.json').get('allocations',{})
        for document in (pbs/'offline-history/jobs').glob('*/*/IMPORTED.json'):
            try:
                host=allocations[read(document)['token']]['spec']['profile']['host_id']
                item=imported_observation(document.parent,host)
                if item:yield item
            except (OSError,ValueError,TypeError,KeyError):continue


def build(runs,settings,policy):
    rows=envelopes((row for run in runs for row in observations(run)),learning_policy(policy),settings)
    return [row for key in sorted(rows) for row in rows[key]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--round',type=Path,required=True,help='Target fixed round providing PBS host profiles')
    parser.add_argument('--policy',type=Path,help='External memory margin and sampling configuration')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    settings=read(args.round)['executor']['distributed']['pbs']
    rows=build(args.run,settings,read(args.policy) if args.policy else True)
    args.output.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
