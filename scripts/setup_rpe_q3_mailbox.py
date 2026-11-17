#!/usr/bin/env python3
"""Explicitly prepare the fixed server1 mailbox through local0's existing SSH alias.

Creates/reuses an owned detached sparse worktree and matching profile, then
checks mailbox status. Does not start a relay, controller or scientific job.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys

from run_rpe_q3_relay import child_options

DEFAULT_SOURCE='4ae47cf4844638647c83ea898ee53588675b0d75'
DEFAULT_REPOSITORY='/home/AbeHiromu/projects/partially-randomized-trotter'

REMOTE = r'''
import json, pathlib, subprocess, sys
p=json.load(sys.stdin)
def run(args):
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if result.returncode:
        raise RuntimeError(args[0]+' '+args[1]+' failed (exit '+str(result.returncode)+')')
    return result.stdout.strip()
def setup():
    repo=pathlib.Path(p['repository']);root=pathlib.Path(p['root']);profile_path=pathlib.Path(p['profile_path'])
    if not repo.is_dir():raise ValueError('Existing repository is missing: '+str(repo))
    if root.resolve()==repo.resolve():raise ValueError('Source root must be a separate worktree')
    if profile_path.resolve().is_relative_to(root.resolve()):raise ValueError('Profile must stay outside fixed SOURCE')
    for assignment in pathlib.Path(p['profile']['work_dir']).glob('*/*/ASSIGNMENT.json'):
        state=assignment.parent/'WORKER_STATE.json'
        if not state.exists() or json.loads(state.read_text()).get('state')=='ASSIGNED':
            raise ValueError('Mailbox has an unstarted assignment; setup will not launch existing jobs')
    if profile_path.exists() and json.loads(profile_path.read_text(encoding='utf-8-sig'))!=p['profile']:
        raise ValueError('Existing worker profile differs; refusing overwrite: '+str(profile_path))
    print('Fetching public coordinator branch into existing repository',file=sys.stderr,flush=True)
    run(['git','-C',str(repo),'fetch','origin','codex/rpe-distributed-q3-coordinator'])
    run(['git','-C',str(repo),'cat-file','-e',p['source']+'^{commit}'])
    if root.exists():
        if not (root/'.git').is_file():raise ValueError('Existing source root is not a linked worktree')
        if run(['git','-C',str(root),'rev-parse','HEAD'])!=p['source']:
            raise ValueError('Existing source root has a different SHA; refusing checkout')
        if run(['git','-C',str(root),'status','--porcelain']):
            raise ValueError('Existing source root has changes; refusing modification')
    else:
        print('Creating detached fixed SOURCE '+p['source'],file=sys.stderr,flush=True)
        root.parent.mkdir(parents=True,exist_ok=True)
        run(['git','-C',str(repo),'worktree','add','--no-checkout','--detach',str(root),p['source']])
        run(['git','-C',str(root),'sparse-checkout','init','--cone'])
        run(['git','-C',str(root),'sparse-checkout','set','src','scripts','tests','configs','schemas'])
        run(['git','-C',str(root),'checkout'])
    if run(['git','-C',str(root),'rev-parse','HEAD'])!=p['source']:raise ValueError('Fixed SOURCE HEAD differs')
    profile_path.parent.mkdir(parents=True,exist_ok=True)
    if not profile_path.exists():
        with profile_path.open('x',encoding='utf-8')as output:json.dump(p['profile'],output,indent=2);output.write('\n')
    elif json.loads(profile_path.read_text(encoding='utf-8-sig'))!=p['profile']:
        raise ValueError('Worker profile changed during setup; refusing overwrite')
    status=json.loads(run([p['profile']['interpreter'],'-B',str(root/'scripts/run_rpe_q3_worker.py'),
        '--profile',str(profile_path),'--mailbox','status']))
    if status['host_id']!=p['profile']['host_id'] or status['source_sha']!=p['source']:
        raise ValueError('Prepared mailbox host/SOURCE differs')
    return dict(ok=True,source=p['source'],root=str(root),profile=str(profile_path),mailbox_status=status,started_jobs=0)
try:print(json.dumps(setup(),ensure_ascii=False))
except Exception as error:
    print(json.dumps(dict(ok=False,error=type(error).__name__+': '+str(error))));raise SystemExit(2)
'''


def setup(config,profile,repository=DEFAULT_REPOSITORY,source=DEFAULT_SOURCE):
    if not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('A full fixed SOURCE SHA is required')
    matches=[worker for worker in config['workers']if worker['host_id']==profile['host_id']]
    if len(matches)!=1:raise ValueError('Exactly one relay worker must match the supplied profile host_id')
    worker=matches[0]
    if worker['root']!=profile['root']:raise ValueError('Relay worker root and supplied profile root differ')
    for value in (repository,worker['root'],worker['profile'],profile['work_dir'],profile['interpreter']):
        if not PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts:
            raise ValueError('Remote repository/profile paths must be absolute Linux paths')
    alias=worker['ssh_alias']
    if not alias or alias.startswith('-'):raise ValueError('Invalid existing SSH alias')
    payload=dict(repository=repository,source=source,root=worker['root'],profile_path=worker['profile'],profile=profile)
    command=config.get('ssh_command',['ssh'])+['-o','BatchMode=yes','-o','ConnectTimeout=10',alias,
        shlex.join([worker.get('python','python3'),'-c',REMOTE])]
    result=subprocess.run(command,input=json.dumps(payload),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        encoding='utf-8',errors='replace',timeout=config.get('setup_timeout_seconds',300),**child_options())
    if result.stderr:print(result.stderr,end='',file=sys.stderr)
    try:response=json.loads(result.stdout)
    except (ValueError,TypeError):raise RuntimeError('SSH mailbox setup failed (exit '+str(result.returncode)+')')
    if result.returncode or response.get('ok')is not True:
        raise RuntimeError(response.get('error','Remote mailbox setup failed'))
    return response


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--repository',default=DEFAULT_REPOSITORY)
    parser.add_argument('--source',default=DEFAULT_SOURCE)
    args=parser.parse_args()
    try:
        value=setup(json.loads(args.config.read_text(encoding='utf-8-sig')),
            json.loads(args.profile.read_text(encoding='utf-8-sig')),args.repository,args.source)
        print(json.dumps(value,indent=2,ensure_ascii=False));return 0
    except Exception as error:
        print(json.dumps(dict(ok=False,error=str(error)),ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
