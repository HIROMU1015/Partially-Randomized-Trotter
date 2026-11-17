#!/usr/bin/env python3
"""Local0 fixed preparation placement and metadata dry-check; never start work.

Reuses the prior deployment SSH/Windows/worktree/immutable-return helpers. The
only changed host action runs the new stage-specific metadata checker.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import types
import zipfile


def load_utility(raw):
    module=types.ModuleType('_preparation_deploy_helpers');module.__file__='legacy_auto_deployment.py'
    exec(compile(raw,'legacy_auto_deployment.py','exec'),module.__dict__)
    return module


def host_action(payload,U):
    U.SOURCE=payload['source'];host=payload['host'];repository=Path(host['repository']).resolve()
    directory=Path(host['run_directory']).resolve();directory.relative_to(repository);directory.mkdir(parents=True,exist_ok=False)
    stage=directory/'payload'
    for name,encoded in payload['files'].items():
        raw=base64.b64decode(encoded,validate=True)
        if U.sha(raw)!=payload['hashes'][name]:raise ValueError('Payload hash differs: '+name)
        U.put(stage/U.relative(name),raw)
    transport=U.ensure_transport(host)
    for source,view in host['source_views'].items():
        U.SOURCE=source
        try:U.ensure_transport(dict(repository=str(repository),transport_root=view))
        except Exception as error:
            U.put(directory/('SOURCE_'+source+'_HOLD.json'),U.encode(dict(source=source,error=str(error))))
    U.SOURCE=payload['source']
    # Fetch immutable input objects only; do not switch any existing checkout.
    for commit in host.get('input_commits',[]):
        try:U.git(repository,'cat-file','-e',commit+'^{commit}')
        except Exception:
            try:U.git(repository,'fetch','origin',commit)
            except Exception as error:U.put(directory/('INPUT_'+commit+'_HOLD.json'),U.encode(dict(commit=commit,error=str(error))))
    profile=stage/host['profile'];target=Path(host['worker_profile']);target.resolve().relative_to(repository)
    value=json.loads(profile.read_text());spool=Path(value['work_dir']);spool.resolve().relative_to(repository)
    if spool.exists():raise ValueError('New operational spool already exists; preserved')
    U.put(target,profile.read_bytes())
    command=[host['transport_python'],'-B',str(transport/'scripts/check_rpe_preparation_host_readiness.py'),
             '--tasks',str(stage/'TASKS.json'),'--profile',str(target),'--output',str(directory/'dry')]
    started=time.monotonic()
    with(directory/'CHECKER.log').open('wb')as log:
        process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:raise ValueError('Preparation metadata checker failed; see saved CHECKER.log')
    readiness=json.loads((directory/'dry/READINESS.json').read_text())
    result=dict(host_id=host['host_id'],rows=readiness['rows'],wall_seconds=time.monotonic()-started,source=payload['source'],
        transport_root=str(transport),worker_profile=str(target),spool_created=False,worker_started=False,relay_started=False,
        scientific_calls=0,compile_calls=0,queue_operations=0)
    U.put(directory/'PLACEMENT.json',U.encode({key:value for key,value in result.items()if key!='rows'}))
    U.put(directory/'PROFILE.json',profile.read_bytes());U.put(directory/'HOST_RESULT.json',U.encode(result))
    if spool.exists():raise ValueError('Operational spool changed during metadata inspection')
    returned={host['host_id']+'/'+p.relative_to(directory).as_posix():base64.b64encode(p.read_bytes()).decode()
              for p in directory.rglob('*')if p.is_file()and not p.is_relative_to(stage)}
    return dict(ok=True,result=result,files=returned)


def remote_main():
    payload=json.load(sys.stdin);U=load_utility(base64.b64decode(payload.pop('utility')))
    try:
        result=host_action(payload,U)if payload['action']=='host'else U.receive_action(payload)
    except Exception as error:result=dict(ok=False,error_type=type(error).__name__,error=str(error))
    print(json.dumps(result,ensure_ascii=False))


def remote(host,payload,U,utility):
    payload=dict(payload,utility=base64.b64encode(utility).decode())
    bootstrap="import base64,sys;exec(base64.b64decode(sys.stdin.buffer.readline()),{'__name__':'_rpe_preparation_deployment'})"
    args=[host['transport_python'],'-c',bootstrap]
    if host.get('transport')=='wsl':command=['wsl.exe','-d',host['distribution'],'--exec',*args]
    else:command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1',host['ssh_alias'],shlex.join(args)]
    data=base64.b64encode(Path(__file__).read_bytes())+b'\n'+U.encode(payload)
    return json.loads(U.run(command,input=data,capture_output=True).stdout)


def return_only(output,U,utility):
    payload=json.loads((output/'RETURN_PAYLOAD.json').read_text());host=payload.pop('return_transport')
    result=remote(host,payload,U,utility)
    if not result.get('ok'):
        U.put(output/('RETURN_ERROR-'+str(time.time_ns())+'.json'),U.encode(result))
        raise ValueError('Return incomplete; use --return-only without redry-check')
    U.put(output/'RETURN_RECEIPT.json',U.encode(result))
    return result


def finish_return(output,manifest,results,U,utility,*,recovered=False,wsl_paused_unchanged=None):
    """Return completed hosts even when a different transport is unavailable."""
    rows=[];returned={}
    for host in manifest['hosts']:
        identity=host['host_id'];result=results[identity]
        if result.get('ok'):
            facts=result['result']
            if facts.get('host_id')!=identity or facts.get('source')!=manifest['source']:
                raise ValueError('Saved host result identity/SOURCE differs: '+identity)
            for name,encoded in result['files'].items():
                U.relative(name)
                if not name.startswith(identity+'/'):raise ValueError('Saved file belongs to another host: '+name)
                base64.b64decode(encoded,validate=True)
            rows.extend(facts['rows']);returned.update(result['files'])
        else:
            rows.append(dict(host_id=identity,status='PREPARATION_OR_TRANSPORT_HOLD',reasons=[result.get('error')]))
            returned[identity+'/ERROR.json']=base64.b64encode(U.encode(result)).decode()
    flag=json.loads(Path(manifest['participation_file']).read_text(encoding='utf-8-sig'))
    summary=dict(schema='RPE_PREPARATION_DEPLOYMENT_RETURN_V1',source=manifest['source'],rows=rows,
        collection_scope='SAVED_LOCAL_RESULTS_NO_HOST_RECONNECT'if recovered else'PLACEMENT_AND_INSPECTION',
        wsl_paused_unchanged=wsl_paused_unchanged,wsl_accepting_new_jobs=flag.get('accepting_new_jobs'),
        scientific_calls=0,compile_calls=0,queue_operations=0)
    U.put(output/'SUMMARY.json',U.encode(summary));returned['SUMMARY.json']=base64.b64encode(U.encode(summary)).decode()
    payload=dict(action='receive',directory=manifest['return_directory'],files=returned,
        manifest={n:U.sha(base64.b64decode(v))for n,v in returned.items()},return_transport=manifest['return_transport'])
    U.put(output/'RETURN_PAYLOAD.json',U.encode(payload));receipt=return_only(output,U,utility)
    print(json.dumps(summary,indent=2,ensure_ascii=False));print(json.dumps(receipt,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--bundle-sha256',required=True);parser.add_argument('--output',type=Path)
    recovery=parser.add_mutually_exclusive_group()
    recovery.add_argument('--return-only',type=Path)
    recovery.add_argument('--collect-saved',type=Path,help='Collect existing host JSON files; connect only to the return host, without inspection or deployment')
    args=parser.parse_args()
    raw=args.bundle.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.bundle_sha256:raise ValueError('Published package SHA256 differs')
    with zipfile.ZipFile(io.BytesIO(raw))as archive:files={name:archive.read(name)for name in archive.namelist()}
    utility=files['legacy_auto_deployment.py'];U=load_utility(utility);manifest=json.loads(files['MANIFEST.json'])
    for name,sha in manifest['files'].items():
        U.relative(name)
        if U.sha(files[name])!=sha:raise ValueError('Package member hash differs: '+name)
    if args.return_only:
        return_only(args.return_only,U,utility);return
    if args.collect_saved:
        output=args.collect_saved
        if not output.is_dir():raise ValueError('Existing saved output directory required')
        results={}
        for host in manifest['hosts']:
            path=output/(host['host_id']+'.json')
            results[host['host_id']]=(json.loads(path.read_text())if path.is_file()else
                dict(ok=False,error_type='SAVED_RESULT_MISSING',error='No saved host result after interrupted collection; no reconnect or retry attempted'))
        finish_return(output,manifest,results,U,utility,recovered=True);return
    output=args.output
    if output is None:parser.error('--output required')
    output.mkdir(parents=True,exist_ok=False)
    participation=Path(manifest['participation_file']);before=participation.read_bytes()
    if json.loads(before.decode('utf-8-sig')).get('accepting_new_jobs')is not False:raise ValueError('WSL must remain explicitly paused; no state changed')
    def inspect(host):
        print(json.dumps(dict(event='HOST_PREPARATION_DRY_CHECK_START',host_id=host['host_id'])),flush=True)
        required=['TASKS.json',host['profile']]
        try:
            result=remote(host,dict(action='host',source=manifest['source'],host=host,files={n:base64.b64encode(files[n]).decode()for n in required},hashes={n:manifest['files'][n]for n in required}),U,utility)
        except Exception as error:
            result=dict(ok=False,error_type=type(error).__name__,error=str(error),automatic_retry=False)
        print(json.dumps(dict(event='HOST_PREPARATION_DRY_CHECK_RETURN',host_id=host['host_id'],ok=result.get('ok'))),flush=True)
        U.put(output/(host['host_id']+'.json'),U.encode(result));return host['host_id'],result
    with ThreadPoolExecutor(max_workers=3)as pool:results=dict(pool.map(inspect,manifest['hosts']))
    if participation.read_bytes()!=before:raise ValueError('WSL participation changed during inspection')
    finish_return(output,manifest,results,U,utility,wsl_paused_unchanged=True)


if __name__=='_rpe_preparation_deployment':remote_main()
elif __name__=='__main__':main()
