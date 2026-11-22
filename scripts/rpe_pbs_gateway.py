#!/usr/bin/env python3
"""Python 3.6 compatible PBS/shared-home gateway. Never executes scientific tasks.

Only the PBS job starts the worker. This login-host process transfers durable
RPC files and queries the scheduler; stdout is a JSON-lines protocol over SSH.
"""
import argparse
import base64
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def safe_id(value):
    if not isinstance(value, str) or not re.match(r'^[A-Za-z0-9_-]{1,120}$', value):
        raise ValueError('Unsafe PBS submission/RPC ID')
    return value


def seconds(value):
    values = [int(part) for part in str(value).split(':')]
    if len(values) != 3:
        raise ValueError('PBS walltime must be HH:MM:SS')
    return values[0]*3600 + values[1]*60 + values[2]


def memory_bytes(value):
    match = re.match(r'^(\d+)([kmgt]?b)?$', str(value).lower())
    if not match:
        raise ValueError('Unsupported PBS memory quantity')
    return int(match.group(1))*{'b':1,'kb':1024,'mb':1024**2,'gb':1024**3,'tb':1024**4}[match.group(2) or 'b']


def qstat(*job_ids):
    command = ['/opt/pbs/bin/qstat', '-x', '-f', '-F', 'json']
    # PBS20's -u adds a text summary ahead of JSON, so filter Job_Owner here.
    command.extend(job_ids)
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if process.returncode:
        errors = process.stderr.strip().splitlines()
        if not errors or not all('Unknown Job Id' in error or 'Job has finished' in error for error in errors):
            raise RuntimeError('qstat failed: ' + process.stderr.strip())
    # PBS20 returns a blank successful response when this user has no jobs.
    if not process.stdout.strip():
        return {}
    jobs = json.loads(process.stdout).get('Jobs', {})
    return {key:value for key,value in jobs.items()
            if value.get('Job_Owner','').split('@')[0] == getpass.getuser()}


def record_allocation(directory):
    directory = Path(directory)
    job_id = os.environ['PBS_JOBID']
    job = qstat(job_id)[job_id]
    resources = job['Resource_List']
    if int(resources.get('nodect', 1)) != 1:
        raise ValueError('Initial PBS worker requires a single allocated node')
    wall = seconds(resources['walltime'])
    used = seconds(job.get('resources_used', {}).get('walltime', '00:00:00'))
    allocation = dict(job_id=job_id, ncpus=int(resources['ncpus']),
        memory_bytes=memory_bytes(resources['mem']), walltime_seconds=wall,
        remaining_seconds=max(0, wall-used), select_nodes=1,
        exec_host=job.get('exec_host'), exec_vnode=job.get('exec_vnode'),
        allocation_source='PBS_QSTAT_RESOURCE_LIST', recorded_epoch=time.time())
    atomic(directory/'ALLOCATION.json', allocation)
    return allocation


def submit_once(directory):
    directory = Path(directory)
    with (directory/'SUBMIT.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory/'QSUB_STARTED.json').exists():
            return
        # A consumed marker is not retried, even if qsub's reply is lost.
        atomic(directory/'QSUB_STARTED.json', dict(epoch=time.time(), pid=os.getpid()))
        process = subprocess.run(['/opt/pbs/bin/qsub', str(directory/'worker.pbs')],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        atomic(directory/'QSUB_RETURN.json', dict(returncode=process.returncode,
            stdout=process.stdout.strip(), stderr=process.stderr.strip(), epoch=time.time()))


def ensure(root, spec):
    token = safe_id(spec['token']); directory = root/'allocations'/token
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'INTENT.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = read(directory/'SPEC.json')
        if old is not None and old != spec:
            raise ValueError('Existing allocation specification differs')
        if old is not None:
            if not (directory/'QSUB_STARTED.json').exists():
                with (directory/'SUBMIT.log').open('ab') as log:
                    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--submit', str(directory)],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            return dict(token=token, existing=True)
        resources = spec['resources']
        if type(resources['ncpus']) is not int or resources['ncpus'] < 1:
            raise ValueError('Explicit positive PBS ncpus required')
        memory_bytes(resources['mem']); seconds(resources['walltime'])
        if resources['queue'] not in ('workq', 'large'):
            raise ValueError('Unconfigured queue')
        profile = dict(spec['profile']); profile['pbs'] = dict(profile.get('pbs') or {}, allocation_dir=str(directory))
        atomic(directory/'PROFILE.json', profile)
        command = [spec['python'], '-B', str(Path(spec['transport_root'])/'scripts/run_rpe_pbs_worker.py'),
                   '--profile', str(directory/'PROFILE.json')]
        gateway = str(Path(__file__).resolve())
        script = '\n'.join([
            '#!/bin/bash', '#PBS -N RPE' + token[:12], '#PBS -q ' + resources['queue'],
            '#PBS -l select=1:ncpus=%d:mem=%s' % (resources['ncpus'], resources['mem']),
            '#PBS -l walltime=' + resources['walltime'], '#PBS -r n',
            '#PBS -h' if resources.get('initial_hold') else '# RPE submission hold disabled',
            '#PBS -o ' + str(directory/'PBS.stdout'), '#PBS -e ' + str(directory/'PBS.stderr'),
            'set -eu', 'umask 077',
            'python3 ' + shlex.quote(gateway) + ' --record-allocation ' + shlex.quote(str(directory)),
            'exec ' + ' '.join(shlex.quote(value) for value in command), ''])
        (directory/'worker.pbs').write_text(script)
        atomic(directory/'SPEC.json', spec)
        with (directory/'SUBMIT.log').open('ab') as log:
            subprocess.Popen([sys.executable, gateway, '--submit', str(directory)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return dict(token=token, existing=False)


def snapshot(root, finished=None, directories=None):
    if directories is None:
        directories = [directory for directory in sorted((root/'allocations').glob('*'))
                       if (directory/'SPEC.json').exists()]
    if not directories:
        return []
    ids = []
    discovery = False
    for directory in directories:
        known = read(directory/'JOB_ID.json')
        receipt = read(directory/'QSUB_RETURN.json')
        if known:
            ids.append(known['job_id'])
        elif receipt and receipt['returncode'] == 0 and receipt['stdout'].strip():
            ids.append(receipt['stdout'].split()[0])
        elif (directory/'QSUB_STARTED.json').exists() and not receipt:
            discovery = True
    finished = {} if finished is None else finished
    jobs = {job_id: finished[job_id] for job_id in ids if job_id in finished}
    query = tuple(job_id for job_id in dict.fromkeys(ids) if job_id not in finished)
    if discovery or query:
        fresh = qstat(*(() if discovery else query))
        jobs.update(fresh)
        # Finished accounting records cannot make a worker eligible. Keep
        # their scheduler evidence in this gateway connection and still read
        # the worker spool each time so late completion remains recoverable.
        finished.update((job_id, job) for job_id, job in fresh.items()
                        if job.get('job_state') == 'F' and job_id in ids)
    result = []
    for directory in directories:
        spec = read(directory/'SPEC.json')
        if spec is None:
            continue
        receipt = read(directory/'QSUB_RETURN.json')
        ids = []
        if receipt and receipt['returncode'] == 0:
            candidate = receipt['stdout'].split()[0]
            if re.match(r'^\d+(?:\[.*\])?\.[A-Za-z0-9_.-]+$', candidate):
                ids = [candidate]
        matches = [key for key,job in jobs.items() if job.get('Job_Name') == 'RPE'+spec['token'][:12]]
        if len(matches) > 1:
            raise ValueError('Duplicate PBS submission token in scheduler')
        if matches:
            if ids and ids != matches:
                raise ValueError('PBS receipt/scheduler job ID mismatch')
            ids = matches
        saved_id = read(directory/'JOB_ID.json')
        if saved_id:
            if ids and ids != [saved_id['job_id']]:
                raise ValueError('Durable PBS job ID changed')
            ids = [saved_id['job_id']]
        if ids and saved_id is None:
            atomic(directory/'JOB_ID.json', dict(job_id=ids[0], recovered=receipt is None))
        job = jobs.get(ids[0], {}) if ids else {}
        state = read(directory/'STATE.json')
        if job:
            status = job['job_state']
        elif state and state.get('state') == 'EXITED':
            status = 'FINISHED_SAVED_WORKER'
        elif ids:
            status = 'HISTORY_MISSING_RECONCILE_SPOOL'
        elif receipt and receipt['returncode'] != 0:
            status = 'SUBMIT_REJECTED'
        else:
            status = 'SUBMIT_AMBIGUOUS_HOLD' if (directory/'QSUB_STARTED.json').exists() else 'SUBMIT_PENDING'
        public_job = {key:job[key] for key in ('Job_Owner','Job_Name','job_state','Resource_List','resources_used',
            'queue','ctime','qtime','stime','etime','mtime','Exit_status','exec_host','exec_vnode','run_count') if key in job}
        result.append(dict(token=spec['token'], host_id=spec['profile']['host_id'], job_id=ids[0] if ids else None,
            status=status, worker=state, receipt=receipt, allocation=read(directory/'ALLOCATION.json'),
            exit_status=job.get('Exit_status'), scheduler=public_job,
            directory=str(directory)))
    return result


def exchange(root, incoming, finished=None, snapshotter=snapshot):
    for token, responses in incoming.get('responses', {}).items():
        directory = root/'allocations'/safe_id(token)
        for request_id, response in responses.items():
            request_id = safe_id(request_id)
            # Do not resurrect responses whose request was consumed already.
            if (directory/'requests'/(request_id+'.json')).exists():
                destination = directory/'responses'/(request_id+'.json')
                old = read(destination)
                if old is not None and old != response:
                    raise ValueError('Immutable RPC response differs')
                if old is None:
                    atomic(destination, response)
    for token, control in incoming.get('controls', {}).items():
        atomic(root/'allocations'/safe_id(token)/'CONTROL.json', control)
    requests = {}
    for directory in (root/'allocations').glob('*'):
        values = [read(path) for path in sorted((directory/'requests').glob('*.json'))
                  if not (directory/'responses'/path.name).exists()]
        if values:
            requests[directory.name] = values
    return dict(requests=requests, allocations=snapshotter(root, finished))


def harvest(directory, wanted=None, owners=None):
    """Return saved bytes after PBS exits; never reconcile by launching work."""
    spec = read(directory/'SPEC.json'); profile = read(directory/'PROFILE.json')
    attempts = []; hashes = set(); incomplete = []
    work = Path(profile['work_dir'])
    helper = Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_dry_check_recovery.py'
    observe_rejection = None
    if helper.is_file():
        module_spec = importlib.util.spec_from_file_location('_pbs_dry_check_recovery', str(helper))
        recovery = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(recovery)
        observe_rejection = recovery.dry_check_rejection
    paths = work.glob('*/*/ASSIGNMENT.json') if owners is None else (
        work/safe_id(owner['job_id'])/safe_id(owner['attempt_id'])/'ASSIGNMENT.json'
        for owner in owners)
    for path in paths:
        if not path.is_file():
            continue
        assignment = read(path)
        state = read(path.parent/'WORKER_STATE.json', {})
        if not (path.parent/'PROCESS_RETURN.json').exists():
            # The same host directory can contain attempts from several
            # shared-pilot sessions. Only this session's own worker can prove
            # that an incomplete attempt lost its execution owner.
            saved_profile = read(path.parent/'WORKER_PROFILE.json', {}) if owners is not None else {}
            allocation_directory = (saved_profile.get('pbs') or {}).get('allocation_dir')
            same_session = allocation_directory and Path(allocation_directory).resolve() == directory.resolve()
            if state.get('state') != 'TRANSFERRED' and (owners is None or same_session):
                if observe_rejection and state.get('state') == 'UNKNOWN':
                    proof = observe_rejection(path.parent, assignment)
                    if proof:
                        state['detail'].update(resource_released=True, prelaunch_rejection=proof)
                incomplete.append(dict(assignment=assignment, state=state))
            continue
        names = ('OUTCOME.json','PROCESS_RETURN.json','ARTIFACT_MANIFEST.json','PREPARATION_RECEIPT.json',
                 'JOB_STAGES.jsonl','TRANSPORT_TIMING.jsonl','WORKER_COLLECT_RETURN.json')
        preparation = assignment.get('request', {}).get('route', {}).get('kind') == 'preparation'
        files = {}
        omit = {'ASSIGNMENT.json','WORKER_PROFILE.json','WORKER_STATE.json','WORKER_PROCESS.lock',
                'SUPERVISOR_STARTED.json','SCIENTIFIC_STARTING.json','WORKER_LAUNCH.json','WORKER_SUPERVISOR.log',
                'SUPERVISOR_LAUNCH.json','STARTING.json','LAUNCH.json','REQUEST.json'}
        for file in path.parent.rglob('*'):
            relative = str(file.relative_to(path.parent))
            if file.is_symlink():
                raise ValueError('Symlink in saved worker artifact')
            if file.is_file() and relative not in omit and file.relative_to(path.parent).parts[0] != '_dry_check':
                if not preparation or relative in names:
                    files[relative] = base64.b64encode(file.read_bytes()).decode('ascii')
        if 'OUTCOME.json' in files:
            hashes.update(read(path.parent/'OUTCOME.json').get('artifacts', {}).values())
            attempts.append(dict(assignment=assignment, state=state, files=files))
    cache = Path(profile.get('artifact_cache_directory') or Path(profile['work_dir'])/'artifacts')
    blobs = {}
    for sha in hashes.intersection(wanted or []):
        if not re.match('^[0-9a-f]{64}$', sha): raise ValueError('Invalid artifact hash')
        raw = (cache/sha[:2]/sha).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha: raise ValueError('Saved artifact hash differs')
        blobs[sha] = base64.b64encode(raw).decode('ascii')
    return dict(attempts=attempts, incomplete=incomplete, artifact_hashes=sorted(hashes), blobs=blobs,
                host_id=spec['profile']['host_id'])


def handle(root, value, finished=None):
    action = value['action']
    if action == 'ensure':
        return ensure(root, value['spec'])
    if action == 'exchange':
        return exchange(root, value, finished)
    if action == 'snapshot':
        return dict(allocations=snapshot(root, finished))
    if action == 'harvest':
        return harvest(root/'allocations'/safe_id(value['token']), value.get('hashes'), value.get('owners'))
    if action == 'delete':
        directory = root/'allocations'/safe_id(value['token'])
        receipt = read(directory/'JOB_ID.json')
        if not receipt:
            raise ValueError('Cannot delete an unidentified PBS job')
        # Explicit connector control only, never used to cancel a busy worker.
        process = subprocess.run(['/opt/pbs/bin/qdel', receipt['job_id']], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True)
        return dict(returncode=process.returncode, stderr=process.stderr)
    raise ValueError('Unknown PBS gateway action')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--submit', type=Path)
    parser.add_argument('--record-allocation', type=Path)
    args = parser.parse_args()
    if args.submit:
        submit_once(args.submit); return
    if args.record_allocation:
        record_allocation(args.record_allocation); return
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    finished = {}
    for line in sys.stdin:
        try:
            answer = dict(ok=True, result=handle(root, json.loads(line), finished))
        except Exception as error:
            answer = dict(ok=False, error=type(error).__name__ + ': ' + str(error))
        print(json.dumps(answer, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
