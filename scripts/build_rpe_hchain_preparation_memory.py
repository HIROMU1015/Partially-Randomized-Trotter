#!/usr/bin/env python3
"""Refresh an existing H-chain graph memory table from accepted operational RSS only."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.rpe_q3_memory import learning_policy, read
from trottertracks.rpe_pbs import apply_hchain_preparation_memory


def class_key(stage,row):
    return stage,json.dumps(row['profile'],sort_keys=True),row.get('task_cpu_count',1)


def refresh(table, observations, policy, cutoff, oldest_owned=None):
    """Only established plan classes change; older owned attempts retain their accounting."""
    previous=table.get('preserve_adjusted_assignments_before')
    if previous is not None and oldest_owned is not None and oldest_owned<previous:
        raise ValueError('Older preparation assignments still need the preceding table')
    result=deepcopy(table);groups=defaultdict(dict)
    for sample in observations:
        key=class_key(sample['stage'],sample)
        if type(sample['peak_rss_bytes']) is not int or sample['peak_rss_bytes']<=0:continue
        jobs=groups[key];job=sample['job_id']
        jobs[job]=max(jobs.get(job,0),sample['peak_rss_bytes'])
    for stage,rows in result['stages'].items():
        for row in rows.values():
            row.pop('previous_memory_bytes',None)
            jobs=groups.get(class_key(stage,row),{})
            if not jobs:continue
            peak=max(max(jobs.values()),row.get('peak_rss_bytes',0) if row.get('origin')=='measured' else 0)
            n,d=policy['margin_numerator'],policy['margin_denominator']
            estimate=max(policy['floor_bytes'],(peak*n+d-1)//d+policy['headroom_bytes'])
            prev=row['memory_bytes']
            if len(jobs)>=policy['min_samples'] or estimate>=prev:
                row['memory_bytes']=estimate
                if estimate<prev:row['previous_memory_bytes']=prev
            row['peak_rss_bytes']=peak
            row['observations']=max(len(jobs),row.get('observations',0))
            if len(jobs)>=policy['min_samples']:row['origin']='measured'
    result['preserve_adjusted_assignments_before']=cutoff
    return result


def build(run, previous, output, policy):
    run=run.resolve();table=read(previous)
    distributed=read(run/'ROUND.requested.json')['executor']['distributed']
    apply_hchain_preparation_memory(deepcopy(distributed),run,dict(path=str(previous.resolve()),sha256=hashlib.sha256(previous.read_bytes()).hexdigest()))
    samples=[]
    with sqlite3.connect((run/'distributed/QUEUE.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        owned=db.execute("select min(assigned_epoch) from jobs where method like 'PREPARATION:%' and state in ('ASSIGNED','STARTING','RUNNING','COMPLETED_PENDING_TRANSFER','UNKNOWN')").fetchone()[0]
        for row in db.execute("select job_id,attempt_id,host_id,request_sha256,request,eligibility from jobs where state='ACCEPTED' and method in ('PREPARATION:compile','PREPARATION:audit')"):
            try:
                req=json.loads(row['request']);prep=req['preparation'];stage=prep['stage']
                if prep['provider']!='HCHAIN_SCOPED' or prep['source']['commit']!=table['source']:continue
                declared=table['stages'][stage][prep['inputs']['PLAN.json']]
                runtime=json.loads(row['eligibility'])['execution_profiles'][row['host_id']]
                cpu=runtime.get('task_cpu_count') or 1
                leaf=run/'distributed/transfers'/row['job_id']/row['attempt_id']
                receipt=read(leaf/'PROCESS_RETURN.json');outcome=read(leaf/'OUTCOME.json')
                measurement=outcome['resource_measurement']
                if (receipt['returncode']!=0 or receipt.get('signal') is not None or outcome['status']!='PREPARATION_COMPLETE'
                        or outcome['request_id']!=row['job_id'] or outcome['request_sha256']!=row['request_sha256']
                        or cpu!=declared.get('task_cpu_count',1) or measurement.get('task_cpu_count',cpu)!=cpu
                        or measurement.get('samples',0)<2):continue
                samples.append(dict(stage=stage,profile=declared['profile'],task_cpu_count=cpu,job_id=row['job_id'],
                                    peak_rss_bytes=measurement['peak_sampled_rss_bytes']))
            except (OSError,ValueError,KeyError,TypeError):continue
    result=refresh(table,samples,learning_policy(policy),time.time(),owned)
    result['previous_table_sha256']=hashlib.sha256(previous.read_bytes()).hexdigest()
    # Validate bounds before writing the requested new output. This is an
    # offline candidate, never an edit of a live table or worker selection.
    if output.resolve()==previous.resolve():raise ValueError('Choose a separate output table')
    for stage,rows in result['stages'].items():
        limit=next(row['memory_bytes'] for row in distributed['pbs']['preparation_task_seconds']
                   if (row['stage'],row['provider'],row['source'])==(stage,'HCHAIN_SCOPED',table['source']))
        if any(row['memory_bytes']>limit for row in rows.values()):raise ValueError('Measured preparation memory exceeds the fixed stage reservation')
    output.write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.run,args.previous,args.output,read(args.policy) if args.policy else True)


if __name__=='__main__':main()
