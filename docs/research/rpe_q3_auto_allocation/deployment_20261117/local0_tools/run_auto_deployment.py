"""Place fixed transport and dry-check real requests; stdlib, no daemon or Q3.

Local0: py -3 run_auto_deployment.py --bundle qg2:/published/delta.zip
        --bundle-sha256 SHA256 --output NEW_LOCAL_DIRECTORY
Retry a failed return without rechecking hosts: --return-only LOCAL_DIRECTORY.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time
import traceback
import zipfile

SCHEMA = 'RPE_Q3_AUTO_DEPLOYMENT_V1'
CHECKER = 'scripts/check_rpe_q3_host_readiness.py'
PROFILE_HELPER = 'src/trottertracks/rpe_host_profiles.py'
COLLECTOR = 'scripts/collect_rpe_q3_deployment_readiness.py'
SOURCE = '349558b748a82f384a4589434a63ef9b5572032c'
ALIASES = {'qg2','qg1','RomuTail'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe payload/evidence path: ' + name)
    return Path(*path.parts)


def put(path, data):
    path = Path(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError('Existing bytes differ; preserved: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def run(command, **kwargs):
    # Windows OpenSSH children must not create a terminal per invocation.
    if os.name == 'nt':
        kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
    result = subprocess.run(command, **kwargs)
    if result.returncode:
        raise RuntimeError('Command failed: ' + repr(command) + ': ' +
                           (result.stderr or b'').decode(errors='replace')[-3000:])
    return result


def git(repo, *args):
    return run(['git', '-C', str(repo), *args], capture_output=True).stdout.decode().strip()


def unpack(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP paths')
        files = {name: archive.read(name) for name in names if not name.endswith('/')}
    for name in files:relative(name)
    manifest = json.loads(files['DEPLOYMENT_MANIFEST.json'])
    if manifest.get('schema') != SCHEMA or manifest.get('mode') != 'dry-check' or manifest.get('source') != SOURCE:
        raise ValueError('Only the fixed explicit dry-check deployment is accepted')
    for name, expected in manifest['files'].items():
        relative(name)
        if name not in files or sha(files[name]) != expected:
            raise ValueError('Payload hash differs: ' + name)
    if COLLECTOR not in manifest['files']:
        raise ValueError('Metadata collector must be hash-bound')
    seen=set()
    for host in manifest['hosts']:
        identity=host['host_id']
        if identity in seen or identity not in {'server1','local','local0-wsl'}:
            raise ValueError('Duplicate or unsupported target host')
        seen.add(identity)
        if host.get('transport')=='wsl':
            if identity!='local0-wsl' or host.get('distribution')!='Ubuntu':raise ValueError('Unexpected WSL distribution')
        elif host.get('ssh_alias') not in {'qg1','RomuTail'}:
            raise ValueError('Unexpected remote SSH alias')
        if len(host['assignments'])!=4:raise ValueError('Exactly four fixed real-input assignments per target')
        for name in [host['profile'],COLLECTOR]+host['assignments']:
            if name not in manifest['files']:raise ValueError('Input is not hash-bound: '+name)
        profile=json.loads(files[host['profile']])
        if profile['host_id']!=identity:raise ValueError('Profile host identity differs')
        for selected in [profile]+[r['profile']for r in profile.get('route_profiles',[])]:
            if selected.get('adapter_root')!=host['transport_root'] or selected.get('adapter_manifest',{}).get('commit')!=SOURCE:
                raise ValueError('Profile must use the fixed common timed adapter')
        for name in host['assignments']:
            assignment=json.loads(files[name])
            if assignment['request']['route']['kind']=='artificial':raise ValueError('Real-input deployment rejects artificial routes')
            canonical=json.dumps(assignment['request'],sort_keys=True,separators=(',',':'),default=str).encode()
            if sha(canonical)!=assignment['request_sha256']:raise ValueError('Assignment canonical request hash differs')
    if seen!={'server1','local','local0-wsl'} or manifest['return']['ssh_alias']!='qg2':
        raise ValueError('Expected three target hosts and return via qg2')
    return manifest,files


def ensure_transport(host):
    """Reuse the existing sparse detached-worktree setup; never switch old worktrees."""
    repo=Path(host['repository']).resolve();root=Path(host['transport_root']).resolve()
    if not repo.is_dir():raise ValueError('Existing owned repository is missing')
    root.relative_to(repo)
    if root==repo:raise ValueError('Transport must be a separate new worktree')
    try:git(repo,'cat-file','-e',SOURCE+'^{commit}')
    except RuntimeError:git(repo,'fetch','origin',SOURCE)
    if root.exists():
        if not (root/'.git').is_file() or git(root,'rev-parse','HEAD')!=SOURCE:
            raise ValueError('Existing transport root differs; preserved')
    else:
        root.parent.mkdir(parents=True,exist_ok=True)
        git(repo,'worktree','add','--detach','--no-checkout',str(root),SOURCE)
        git(root,'sparse-checkout','set','--cone','scripts','src','configs','schemas')
        git(root,'checkout','--detach',SOURCE)
    if git(root,'rev-parse','HEAD')!=SOURCE or git(root,'diff','HEAD','--name-only','--','scripts','src','configs','schemas'):
        raise ValueError('Fixed transport source is changed')
    return root


def host_action(payload):
    started=time.monotonic();host=payload['host'];files=payload['files']
    repo=Path(host['repository']).resolve();directory=Path(host['run_directory']).resolve()
    directory.relative_to(repo)
    directory.mkdir(parents=True,exist_ok=False)
    stage=directory/'payload'
    for name,encoded in files.items():
        raw=base64.b64decode(encoded,validate=True)
        if sha(raw)!=payload['hashes'][name]:raise ValueError('Transferred payload hash differs: '+name)
        put(stage/relative(name),raw)
    profile_path=Path(host['worker_profile'])
    profile_path.resolve().relative_to(repo)
    profile=json.loads((stage/host['profile']).read_text())
    spool=Path(profile['work_dir'])
    spool.resolve().relative_to(repo)
    if spool.exists():raise ValueError('New operational spool already exists; deployment does not open it')
    transport=ensure_transport(host)
    put(profile_path,(stage/host['profile']).read_bytes())
    spec=importlib.util.spec_from_file_location('_auto_deploy_collector',stage/COLLECTOR)
    collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
    sys.path.insert(0,str(transport/'src'))
    from trottertracks.rpe_host_profiles import select_host_profile
    rows=[];combined=dict(schema='RPE_PLAN_DRY_CACHE_V1',methods={})
    for assignment_name in host['assignments']:
        assignment_path=stage/assignment_name
        assignment=json.loads(assignment_path.read_text())
        selected=select_host_profile(profile,assignment);method=assignment['method'];relative(method)
        target=directory/'dry'/relative(method)
        command=[selected['interpreter'],'-B',str(transport/CHECKER),'--assignment',str(assignment_path),
                 '--profile',str(profile_path),'--output',str(target)]
        with (directory/(method+'_CHECKER.log')).open('wb')as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        readiness=target/'READINESS.json';record=json.loads(readiness.read_text())
        proof=directory/'proof'/relative(method)
        metadata={};proof_error=None
        try:
            metadata=collector.collect(str(transport),str(assignment_path),str(profile_path),str(readiness),str(proof))
            cache=json.loads((proof/'DRY_CACHE.json').read_text())
            combined['methods'].update(cache['methods'])
        except Exception as error:
            proof_error=type(error).__name__+': '+str(error)
            put(proof/'COLLECTION_ERROR.json',encode(dict(error_type=type(error).__name__,error=str(error))))
        rows.append(dict(host_id=host['host_id'],method=method,job_id=assignment['job_id'],status=record['status'] if proof_error is None else 'READINESS_PROOF_UNCONFIRMED',
            dry_check_status=record['status'],proof_error=proof_error,
            checker_returncode=process.returncode,readiness=host['host_id']+'/dry/'+method+'/READINESS.json',
            runtime=metadata.get('actual_runtime'),selected_profile=host['host_id']+'/proof/'+method+'/SELECTED_PROFILE.json'))
    put(directory/'DRY_CACHE.json',encode(combined))
    put(directory/'PROFILE.json',(stage/host['profile']).read_bytes())
    put(directory/'PLACEMENT.json',encode(dict(host_id=host['host_id'],source=SOURCE,transport_root=str(transport),
        worker_profile=str(profile_path),worker_profile_sha256=sha(profile_path.read_bytes()),spool_created=False,
        worker_started=False,relay_started=False,controller_started=False,scientific_calls=0,queue_operations=0)))
    result=dict(host_id=host['host_id'],rows=rows,wall_seconds=time.monotonic()-started,
        scientific_calls=0,queue_operations=0,profile=str(profile_path),transport_root=str(transport))
    put(directory/'HOST_RESULT.json',encode(result))
    if spool.exists():raise ValueError('Operational spool changed during dry-check')
    returned={host['host_id']+'/'+p.relative_to(directory).as_posix():base64.b64encode(p.read_bytes()).decode()
              for p in directory.rglob('*')if p.is_file()and not p.is_relative_to(stage)}
    return dict(ok=True,result=result,files=returned)


def receive_action(payload):
    target = Path(payload['directory'])
    if not target.is_absolute():
        raise ValueError('Absolute new return directory required')
    target.mkdir(parents=True, exist_ok=True)
    manifest = payload['manifest']
    for name, expected in manifest.items():
        raw = base64.b64decode(payload['files'][name], validate=True)
        if sha(raw) != expected:
            raise ValueError('Transfer bytes differ: ' + name)
        put(target / relative(name), raw)
    manifest_raw = encode(manifest)
    put(target / 'TRANSFER_MANIFEST.json', manifest_raw)
    for name, expected in manifest.items():
        if sha((target / relative(name)).read_bytes()) != expected:
            raise ValueError('Returned file hash differs: ' + name)
    receipt = dict(transfer_manifest_sha256=sha(manifest_raw), verified_transferred_files=len(manifest),
                   destination=str(target), scientific_evaluation_calls=0, production_queue_submissions=0)
    put(target / 'RETURN_RECEIPT.json', encode(receipt))
    return dict(ok=True, receipt=receipt)


def remote_main():
    try:
        payload = json.load(sys.stdin)
        if payload['action'] == 'host':
            result = host_action(payload)
        elif payload['action'] == 'receive':
            result = receive_action(payload)
        else:
            raise ValueError('Unsupported delta action')
    except Exception as error:
        result = dict(ok=False, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    print(json.dumps(result, ensure_ascii=False))


def ssh(alias, interpreter, payload):
    if alias not in ALIASES:
        raise ValueError('Unsupported SSH alias')
    bootstrap = "import base64,sys;exec(base64.b64decode(sys.stdin.buffer.readline()),{'__name__':'_rpe_delta_remote'})"
    command = ['ssh', '-T', '-o', 'BatchMode=yes', alias, shlex.join([interpreter, '-c', bootstrap])]
    data = base64.b64encode(Path(__file__).read_bytes()) + b'\n' + encode(payload)
    response = run(command, input=data, capture_output=True)
    return json.loads(response.stdout)


def wsl(host, payload):
    bootstrap = "import base64,sys;exec(base64.b64decode(sys.stdin.buffer.readline()),{'__name__':'_rpe_delta_remote'})"
    command = ['wsl.exe', '-d', host['distribution'], '--exec', host['transport_python'], '-c', bootstrap]
    data = base64.b64encode(Path(__file__).read_bytes()) + b'\n' + encode(payload)
    response = run(command, input=data, capture_output=True)
    return json.loads(response.stdout)


def return_only(output):
    payload = json.loads((Path(output) / 'RETURN_PAYLOAD.json').read_text())
    target = payload.pop('transport')
    result = ssh(target['ssh_alias'], target['python'], payload)
    if not result.get('ok'):
        put(Path(output) / ('RETURN_ERROR-' + str(time.time_ns()) + '.json'), encode(result))
        raise ValueError('Return failed; evidence retained locally: ' + str(result.get('error')))
    put(Path(output) / 'RETURN_RECEIPT.json', encode(result))
    return result


def prepare(bundle, expected_sha, output):
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    if bundle.startswith('qg2:/'):
        run(['scp', '-q', bundle, str(output / 'delta.zip')], capture_output=True)
        raw = (output / 'delta.zip').read_bytes()
    else:
        raw = Path(bundle).read_bytes();put(output / 'delta.zip', raw)
    if sha(raw) != expected_sha:
        raise ValueError('Published bundle SHA256 differs')
    manifest, files = unpack(raw)
    participation = Path(manifest['participation_file'])
    if not participation.is_file() or json.loads(participation.read_text(encoding='utf-8-sig')).get('accepting_new_jobs') is not False:
        raise ValueError('WSL participation must already be paused; no state was changed')
    participation_bytes = participation.read_bytes()
    def changed_host(host):
        required = [COLLECTOR, host['profile']] + host['assignments']
        print(json.dumps(dict(event='HOST_DRY_CHECK_START', host_id=host['host_id'],
                              cells=len(host['assignments'])), ensure_ascii=False), flush=True)
        try:
            payload=dict(action='host',host=host,hashes={name:manifest['files'][name]for name in required},
                files={name:base64.b64encode(files[name]).decode()for name in required})
            result=wsl(host,payload) if host.get('transport')=='wsl' else ssh(host['ssh_alias'],host['transport_python'],payload)
        except Exception as error:
            result = dict(ok=False, error_type=type(error).__name__, error=str(error))
        print(json.dumps(dict(event='HOST_DRY_CHECK_RETURN', host_id=host['host_id'],
                              ok=result.get('ok'), wall_seconds=result.get('result', {}).get('wall_seconds')),
                         ensure_ascii=False), flush=True)
        return host['host_id'], result
    with ThreadPoolExecutor(max_workers=max(1, len(manifest['hosts']))) as pool:
        hosts = dict(pool.map(changed_host, manifest['hosts']))
    if participation.read_bytes() != participation_bytes:
        raise ValueError('Participation changed during dry-check; preserve the existing state')
    rows, returned = [], {}
    for host, result in hosts.items():
        put(output / (host + '.RESULT.json'), encode(result))
        if result.get('ok'):
            rows.extend(result['result']['rows']);returned.update(result['files'])
        else:
            rows.append(dict(host_id=host, status='UNCONFIRMED_PREPARATION_OR_TRANSPORT', reason=result.get('error')))
            returned[host + '/PREPARATION_ERROR.json'] = base64.b64encode(encode(result)).decode()
    summary = dict(schema=SCHEMA, scope='FIXED_TRANSPORT_PLACEMENT_AND_REAL_INPUT_DRY_CHECK_ONLY', bundle_sha256=sha(raw), rows=rows,
                   scientific_calls=0, queue_operations=0)
    put(output / 'SUMMARY.json', encode(summary))
    returned['SUMMARY.json'] = base64.b64encode(encode(summary)).decode()
    hashes = {name: sha(base64.b64decode(data)) for name, data in returned.items()}
    payload = dict(action='receive', directory=manifest['return']['directory'], files=returned, manifest=hashes,
                   transport=manifest['return'])
    put(output / 'RETURN_PAYLOAD.json', encode(payload))
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    result = return_only(output)
    return summary, result


def package(config_path, target):
    """Publish an explicit cell list; no host connection or dry-check occurs."""
    config_path = Path(config_path);config = json.loads(config_path.read_text())
    sources = config.pop('source_files')
    files = {name: (config_path.parent / path).resolve().read_bytes() for name, path in sources.items()}
    config['files'] = {name: sha(raw) for name, raw in files.items()}
    files['DEPLOYMENT_MANIFEST.json'] = encode(config)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026,11,17,0,0,0)), raw)
    raw = buffer.getvalue();unpack(raw)
    put(target, raw)
    return dict(bundle=str(target), bundle_sha256=sha(raw), hosts=len(config['hosts']),
                changed_cells=sum(len(host['assignments']) for host in config['hosts']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle');parser.add_argument('--bundle-sha256');parser.add_argument('--output', type=Path)
    parser.add_argument('--return-only', type=Path)
    parser.add_argument('--package', type=Path, help='build a ZIP from a local explicit delta config; never SSH')
    parser.add_argument('--output-zip', type=Path)
    args = parser.parse_args()
    if args.package:
        if args.output_zip is None:
            parser.error('--package requires --output-zip')
        print(json.dumps(package(args.package, args.output_zip)))
    elif args.return_only:
        return_only(args.return_only)
    elif not args.bundle or not args.bundle_sha256 or args.output is None:
        parser.error('--bundle, --bundle-sha256 and a new --output are required')
    else:
        summary, _ = prepare(args.bundle, args.bundle_sha256, args.output)
        if any(row.get('status') != 'REAL_HOST_DRY_CHECK_PASS' for row in summary['rows']):
            raise SystemExit(2)


if __name__ == '_rpe_delta_remote':
    remote_main()
elif __name__ == '__main__':
    main()
