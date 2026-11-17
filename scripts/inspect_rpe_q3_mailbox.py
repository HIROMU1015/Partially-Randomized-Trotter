#!/usr/bin/env python3
"""Read mailbox metadata through existing SSH; never invoke a worker or write.

Safe for ASSIGNED attempts: this inspector does not use mailbox status, whose
normal recovery path may launch a previously received, unstarted assignment.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

from run_rpe_q3_relay import child_options

REMOTE = r'''
import hashlib,json,pathlib,subprocess,sys
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
try:
    profile=json.load(sys.stdin);root=pathlib.Path(profile['root']);spool=pathlib.Path(profile['work_dir'])
    revision=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if revision.returncode:raise ValueError('Cannot read worker SOURCE HEAD')
    attempts=[]
    for path in sorted(spool.glob('*/*/ASSIGNMENT.json')):
        assignment=json.loads(path.read_text(encoding='utf-8'));directory=path.parent
        state_path=directory/'WORKER_STATE.json';return_path=directory/'PROCESS_RETURN.json'
        outcome=directory/'OUTCOME.json';stages=directory/'JOB_STAGES.jsonl'
        state=json.loads(state_path.read_text(encoding='utf-8'))if state_path.exists()else{}
        kind=assignment['request']['route']['kind']
        count=None
        if kind=='artificial':
            count=0
            if stages.exists():
                with stages.open(encoding='utf-8')as lines:
                    count=sum(json.loads(line).get('stage')=='ARTIFICIAL_EVALUATION_START'for line in lines if line.strip())
        attempts.append(dict(job_id=assignment['job_id'],attempt_id=assignment['attempt_id'],
            state=state.get('state'),state_epoch=state.get('epoch'),
            process_return=json.loads(return_path.read_text(encoding='utf-8'))if return_path.exists()else None,
            outcome_sha256=hashlib.sha256(outcome.read_bytes()).hexdigest()if outcome.exists()else None,
            artificial_execute_count=count,route_kind=kind))
    print(json.dumps(dict(host_id=profile['host_id'],source_sha=revision.stdout.strip(),attempts=attempts),ensure_ascii=False))
except Exception as error:
    print(json.dumps(dict(ok=False,error=type(error).__name__+': '+str(error))));raise SystemExit(2)
'''


def inspect(config,profile):
    matches=[worker for worker in config['workers']if worker['host_id']==profile['host_id']]
    if len(matches)!=1:raise ValueError('Exactly one relay worker must match the supplied profile host_id')
    worker=matches[0]
    if worker['root']!=profile['root']:raise ValueError('Relay worker root and supplied profile root differ')
    for value in (profile['root'],profile['work_dir']):
        path=PurePosixPath(value)
        if not path.is_absolute()or'..'in path.parts:raise ValueError('Known remote SOURCE/spool must be absolute Linux paths')
    alias=worker['ssh_alias']
    if not alias or alias.startswith('-'):raise ValueError('Invalid existing SSH alias')
    command=config.get('ssh_command',['ssh'])+['-o','BatchMode=yes','-o','ConnectTimeout=10',alias,
        shlex.join([worker.get('python','python3'),'-c',REMOTE])]
    result=subprocess.run(command,input=json.dumps(profile),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        encoding='utf-8',errors='replace',timeout=config.get('request_timeout_seconds',60),**child_options())
    if result.stderr:print(result.stderr,end='',file=sys.stderr)
    try:value=json.loads(result.stdout)
    except ValueError:raise RuntimeError('SSH mailbox inspection failed (exit '+str(result.returncode)+')')
    if result.returncode or value.get('ok')is False:raise RuntimeError(value.get('error','Remote inspection failed'))
    if value.get('host_id')!=profile['host_id']:raise ValueError('Inspected mailbox host differs')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    args=parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    try:
        result=inspect(json.loads(args.config.read_text(encoding='utf-8-sig')),json.loads(args.profile.read_text(encoding='utf-8-sig')))
        print(json.dumps(result,ensure_ascii=False,separators=(',',':')));return 0
    except Exception as error:
        print(json.dumps(dict(ok=False,error=str(error)),ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
