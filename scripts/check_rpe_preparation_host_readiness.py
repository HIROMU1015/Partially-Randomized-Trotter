#!/usr/bin/env python3
"""Read exact task SOURCE/input/Python/compiler metadata; never execute a leaf."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import rpe_preparation_worker as P
from trottertracks.rpe_preparation_queue import normalize_preparation_request


def check(request, profile):
    request=normalize_preparation_request(request)
    assignment=dict(job_id=request['request_id'],request=request,request_sha256=P.digest(request),
                    source_manifest=request['preparation']['source'])
    started=time.monotonic()
    row=dict(schema='RPE_PREPARATION_HOST_READINESS_V1',host_id=profile['host_id'],
        request_id=request['request_id'],request_sha256=assignment['request_sha256'],stage=request['preparation']['stage'],
        provider=request['preparation'].get('provider'),method=request.get('method'),source_manifest=assignment['source_manifest'],
        input_hashes=request['preparation'].get('input_hashes',[]),scientific_calls=0,compile_calls=0,queue_operations=0)
    try:
        result=P.dry_check(request,profile,assignment['source_manifest'])
        P.validate_preparation_fingerprint(assignment,profile,result['fingerprint'])
        row.update(status='REAL_HOST_PREPARATION_DRY_CHECK_PASS',fingerprint=result['fingerprint'],
                   fingerprint_sha256=P.digest(result['fingerprint']),execution_profile=result['fingerprint']['profile'],reasons=[])
    except Exception as error:
        row.update(status='REAL_HOST_PREPARATION_DRY_CHECK_HOLD',reasons=[type(error).__name__+': '+str(error)])
    row['wall_seconds']=time.monotonic()-started
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();tasks=P.read(args.tasks);tasks=tasks.get('tasks',tasks)if isinstance(tasks,dict)else tasks
    profile=P.read(args.profile);args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for task in tasks:
        request=task.get('request',task);row=check(request,profile);rows.append(row)
        P.put(args.output/row['request_id']/'READINESS.json',row)
        print(json.dumps({k:row.get(k)for k in ('host_id','method','stage','provider','request_id','status','reasons')},ensure_ascii=False),flush=True)
    P.put(args.output/'READINESS.json',dict(schema='RPE_PREPARATION_HOST_READINESS_SET_V1',host_id=profile['host_id'],rows=rows,
         scientific_calls=0,compile_calls=0,queue_operations=0))
    return 0


if __name__=='__main__':raise SystemExit(main())
