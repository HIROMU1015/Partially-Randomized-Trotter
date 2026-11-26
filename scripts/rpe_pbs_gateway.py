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


BULK_CHUNK_BYTES = 1024 * 1024
_completion_last_token = None


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


class SchedulerCache:
    def __init__(self, seconds=15):
        self.seconds = seconds
        self.observed = {}
        self.terminal_sessions = set()
        self.discoveries = {}

    def discover(self, lookup, *, force=False):
        now = time.monotonic()
        cached = self.discoveries.get(lookup)
        if force or cached is None or now - cached[0] >= self.seconds:
            cached = now, lookup()
            self.discoveries[lookup] = cached
        return cached[1]

    def query(self, job_ids, *, force=False):
        if not job_ids:
            return self.discover(qstat, force=force)
        now = time.monotonic()
        due = [job for job in job_ids if force or job not in self.observed
               or now - self.observed[job][0] >= self.seconds]
        if due:
            found = qstat(*due)
            self.observed.update((job, (now, found.get(job))) for job in due)
        return {job: self.observed[job][1] for job in job_ids if self.observed[job][1] is not None}


def snapshot(root, finished=None, directories=None, *, scheduler=None, force=False):
    if directories is None:
        directories = [directory for directory in sorted((root/'allocations').glob('*'))
                       if (directory/'SPEC.json').exists()]
    if not directories:
        return []
    ids = []
    discovery = False
    ended = set()
    for directory in directories:
        marker = directory/'STATE.json'
        worker = read(marker) if scheduler is not None else None
        if worker and worker.get('state') == 'EXITED' and marker not in scheduler.terminal_sessions:
            ended.add(marker)
            force = True
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
        ids_to_query = () if discovery else query
        fresh = (qstat(*ids_to_query) if scheduler is None else
                 scheduler.query(ids_to_query, force=force))
        jobs.update(fresh)
        # Finished accounting records cannot make a worker eligible. Keep
        # their scheduler evidence in this gateway connection and still read
        # the worker spool each time so late completion remains recoverable.
        finished.update((job_id, job) for job_id, job in fresh.items()
                        if job.get('job_state') == 'F' and job_id in ids)
        if scheduler is not None:
            scheduler.terminal_sessions.update(ended)
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


def completion_priority(path):
    if path.name.startswith('complete-q3-'):
        return 0
    if path.name.startswith('complete-evidence-'):
        return 2
    return 1


def fair_completion_requests(pending, *, inline):
    """Rotate allocations within each completion priority."""
    from collections import deque

    groups = {}
    for token, path, bulk in pending:
        priority = completion_priority(path) if inline else 0
        groups.setdefault(priority, {}).setdefault(token, deque()).append((token, path, bulk))
    for priority in sorted(groups):
        lanes = groups[priority]
        tokens = sorted(lanes)
        if _completion_last_token in tokens:
            start = tokens.index(_completion_last_token) + 1
            tokens = tokens[start:] + tokens[:start]
        ready = deque(tokens)
        while ready:
            token = ready.popleft()
            yield lanes[token].popleft()
            if lanes[token]:
                ready.append(token)


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
    if incoming.get('responses_only') is True:
        return {}
    mode = incoming.get('request_mode')
    if mode not in (None, 'control', 'completion'):
        raise ValueError('Unknown PBS exchange request mode')
    remaining = incoming.get('limit') if mode == 'completion' else None
    if mode == 'completion' and (type(remaining) is not int or remaining < 0):
        raise ValueError('Invalid PBS completion request limit')
    excluded = incoming.get('exclude', {})
    requests = {}
    pending = []
    for directory in (root/'allocations').glob('*'):
        if (directory/'SESSION_END.json').exists():
            continue  # The ended pilot's owned results use selective harvest.
        for path in sorted((directory/'requests').glob('*.json')):
            bulk = path.name.startswith(('complete-', 'artifact-put-'))
            if (mode == 'control' and bulk) or (mode == 'completion' and not bulk):
                continue
            if path.stem in excluded.get(directory.name, ()) or (directory/'responses'/path.name).exists():
                continue
            pending.append((directory.name, path, bulk))
    global _completion_last_token
    inline_budget = BULK_CHUNK_BYTES
    inline_bytes = 0
    inline_cores = 0
    inline = incoming.get('inline_q3_completions') is True and mode == 'completion'
    if not inline:
        inline_budget = 0
    if mode == 'completion':
        pending = fair_completion_requests(pending, inline=inline)
    for token, path, bulk in pending:
        if remaining == 0:
            break
        try:
            if bulk and incoming.get('bounded_results') is True:
                size = path.stat().st_size
                priority = completion_priority(path)
                if priority != 1 and 0 < size <= inline_budget:
                    value = read(path)
                    field = 'pbs_evidence' if priority == 0 else 'pbs_evidence_status_batch'
                    if value is None:
                        continue
                    if (value['request_id'] != path.stem or value['endpoint'] != '/v1/complete'
                            or field not in value['payload']):
                        raise ValueError('PBS inline completion identity changed')
                    inline_budget -= size
                    inline_bytes += size
                    inline_cores += int(priority == 0)
                else:
                    value = dict(request_id=path.stem, spool_bytes=size)
            else:
                value = read(path)
        except FileNotFoundError:
            continue
        if value is None:
            continue
        requests.setdefault(token, []).append(value)
        if remaining is not None:
            remaining -= 1
            _completion_last_token = token
    if incoming.get('inline_q3_completions') is True and mode == 'completion':
        return dict(requests=requests, allocations=[], inline_bytes=inline_bytes, inline_cores=inline_cores)
    return dict(requests=requests, allocations=[] if mode == 'completion' else snapshotter(root, finished))


def admission_exit(directory, attempt, assignment, state):
    """Prove that an ended PBS session declined this attempt before launching it."""
    detail = state.get('detail') or {}
    if (state.get('state') != 'UNKNOWN' or detail.get('reason') != 'PBS_ADMISSION_DEADLINE_BEFORE_LAUNCH'
            or detail.get('scientific_execution_started') is not False or detail.get('automatic_retry') is not False):
        return None
    try:
        worker = read(directory/'STATE.json')
        started = worker['start_clock']
        evidence = [json.loads(line) for line in (attempt/'TRANSPORT_TIMING.jsonl').read_text().splitlines()]
        records = [row for row in evidence if row['event'] == 'WORKER_ASSIGNMENT_DURABLE']
        matching, = records
        names = {'ASSIGNMENT.json', 'ORIGINAL_REQUEST.json', 'TRANSPORT_TIMING.jsonl',
                 'WORKER_PROFILE.json', 'WORKER_STATE.json'}
        if (worker['state'] != 'EXITED' or {p.name for p in attempt.iterdir()} - names
                or matching['clock_id'] != state['clock_id'] or matching['clock_id'] != worker['clock_id']
                or matching['clock_id'] != started['clock_id'] or matching['pid'] != worker['pid']
                or not started['monotonic'] <= matching['monotonic'] <= state['monotonic'] <= worker['monotonic']
                or any(matching[k] != assignment[k] for k in ('job_id', 'attempt_id'))):
            return None
        return dict(token=directory.name, **{k: assignment[k] for k in
            ('job_id', 'attempt_id', 'request_sha256')}, pid=worker['pid'], clock_id=worker['clock_id'])
    except (OSError, KeyError, TypeError, ValueError):
        return None


_PREPARATION_FILES = frozenset(('OUTCOME.json','PROCESS_RETURN.json','ARTIFACT_MANIFEST.json',
    'PREPARATION_RECEIPT.json','JOB_STAGES.jsonl','TRANSPORT_TIMING.jsonl','WORKER_COLLECT_RETURN.json'))
_OMIT_HARVEST = frozenset(('ASSIGNMENT.json','WORKER_PROFILE.json','WORKER_STATE.json','WORKER_PROCESS.lock','PBS_CORE_ACK.json',
    'SUPERVISOR_STARTED.json','SCIENTIFIC_STARTING.json','WORKER_LAUNCH.json','WORKER_SUPERVISOR.log',
    'SUPERVISOR_LAUNCH.json','STARTING.json','LAUNCH.json','REQUEST.json','PBS_ATTEMPT_OWNER.json',
    'PBS_ATTEMPT_STOP_REQUEST.json','PBS_ATTEMPT_INTERRUPTED.json'))


def harvest_path(relative, preparation):
    path = Path(relative)
    if (path.is_absolute() or not path.parts or '..' in path.parts
            or path.parts[0] == '_dry_check' or relative in _OMIT_HARVEST
            or preparation and relative not in _PREPARATION_FILES):
        raise ValueError('File is outside the saved PBS receipt')
    return path


def offline_root(directory):
    pointer = directory/'offline-search'/'INDEPENDENT.json'
    if not pointer.is_file():
        return directory/'offline-search'
    root = Path(read(pointer)['root']).resolve()
    declared = Path(read(directory/'PROFILE.json')['pbs']['independent_root']).resolve()
    if root != declared or root != (directory.parent.parent/'independent-search').resolve():
        raise ValueError('PBS independent result directory differs from allocation')
    return root


def read_bulk_chunk(directory, request):
    kind = request['kind']
    name = request['name']
    if kind == 'spool':
        identifier = safe_id(name)
        if not identifier.startswith(('complete-', 'artifact-put-')):
            raise ValueError('PBS spool payload is not a bulk request')
        path = directory/'requests'/(identifier+'.json')
    elif kind == 'offline':
        if name not in ('REQUEST.json', 'PROCESS_RETURN.json', 'OUTCOME.json'):
            raise ValueError('Unknown PBS offline receipt')
        work = offline_root(directory)/'jobs'
        path = work/safe_id(request['method'])/safe_id(request['job_id'])/name
        path.resolve().relative_to(work.resolve())
    elif kind in ('offline_report','offline_artifact'):
        work = offline_root(directory)
        part = safe_id(request['part'])
        if kind == 'offline_report':
            if name not in ('SCOPE_MAP.json','ROUND_RESULT.json'):
                raise ValueError('Unknown independent report')
            path = work/'reports'/part/name
        else:
            if not re.fullmatch(r'[0-9a-f]{64}',name):
                raise ValueError('Invalid independent artifact hash')
            item = read(work/'RESULTS.json', {}).get(safe_id(request['method'])+'/'+safe_id(request['job_id']))
            if not item or item['part'] != part or name not in item.get('artifacts',{}):
                raise ValueError('Independent artifact not in returned receipt')
            path = work/'parts'/part/'distributed'/'artifacts'/name[:2]/name
        path.resolve().relative_to(work.resolve())
    else:
        profile = read(directory/'PROFILE.json')
        work = Path(profile['work_dir'])
        if kind == 'artifact':
            if not re.fullmatch(r'[0-9a-f]{64}', name):
                raise ValueError('Invalid PBS artifact hash')
            cache = Path(profile.get('artifact_cache_directory') or work/'artifacts')
            path = cache/name[:2]/name
        elif kind == 'attempt':
            attempt = work/safe_id(request['job_id'])/safe_id(request['attempt_id'])
            assignment = read(attempt/'ASSIGNMENT.json')
            if (assignment['job_id'], assignment['attempt_id']) != (request['job_id'], request['attempt_id']):
                raise ValueError('PBS bulk receipt belongs to a different attempt')
            preparation = assignment.get('request', {}).get('route', {}).get('kind') == 'preparation'
            path = attempt/harvest_path(name, preparation)
            path.resolve().relative_to(attempt.resolve())
        else:
            raise ValueError('Unknown PBS bulk receipt kind')
    offset = request['offset']
    if type(offset) is not int or offset < 0 or path.is_symlink():
        raise ValueError('Invalid PBS bulk receipt offset or path')
    with path.open('rb') as stream:
        observed = os.fstat(stream.fileno())
        size = observed.st_size
        identity = [observed.st_dev, observed.st_ino, observed.st_mtime_ns, size]
        if offset > size or request.get('size', size) != size:
            raise ValueError('PBS bulk receipt size changed')
        if request.get('file_identity', identity) != identity:
            raise ValueError('PBS bulk receipt changed during transfer')
        stream.seek(offset)
        raw = stream.read(BULK_CHUNK_BYTES)
        if kind == 'spool':
            # The durable RPC files are ASCII JSON. Its result payload is
            # already encoded, so send this slice without another base64 layer.
            return dict(size=size, file_identity=identity, text=raw.decode('ascii'))
        return dict(size=size, file_identity=identity, data=base64.b64encode(raw).decode('ascii'))


def harvest(directory, wanted=None, owners=None, completed_only=False, manifest_only=False):
    """Read existing receipts, optionally only finished attempts on a live worker."""
    if completed_only and owners is None:
        raise ValueError('Live PBS collection requires exact attempt owners')
    spec = read(directory/'SPEC.json'); profile = read(directory/'PROFILE.json')
    attempts = []; hashes = set(); incomplete = []
    work = Path(profile['work_dir'])
    helper = Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_dry_check_recovery.py'
    if spec.get('transport_root'):
        fixed_helper = Path(spec['transport_root'])/'src/trottertracks/rpe_dry_check_recovery.py'
        if fixed_helper.is_file():
            helper = fixed_helper
    observe_rejection = None
    if helper.is_file():
        module_spec = importlib.util.spec_from_file_location('_pbs_dry_check_recovery', str(helper))
        recovery = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(recovery)
        observe_rejection = recovery.dry_check_rejection
    elif Path(__file__).parent.is_file():  # Packed PBS pilot zipapp.
        try:
            from rpe_dry_check_recovery import dry_check_rejection as observe_rejection
        except ImportError:
            pass
    paths = work.glob('*/*/ASSIGNMENT.json') if owners is None else (
        work/safe_id(owner['job_id'])/safe_id(owner['attempt_id'])/'ASSIGNMENT.json'
        for owner in owners)
    for path in paths:
        if not path.is_file():
            continue
        assignment = read(path)
        state = read(path.parent/'WORKER_STATE.json', {})
        if completed_only and state.get('state') != 'COMPLETED_PENDING_TRANSFER':
            continue
        exit_proof = state.get('detail', {}).get('pbs_attempt_exit', {})
        if state.get('state') == 'UNKNOWN' and exit_proof:
            if exit_proof.get('allocation_dir') == str(directory.resolve()):
                incomplete.append(dict(assignment=assignment, state=state))
            continue
        if not (path.parent/'PROCESS_RETURN.json').exists():
            # The same host directory can contain attempts from several
            # shared-pilot sessions. Only this session's own worker can prove
            # that an incomplete attempt lost its execution owner.
            saved_profile = read(path.parent/'WORKER_PROFILE.json', {}) if owners is not None else {}
            allocation_directory = (saved_profile.get('pbs') or {}).get('allocation_dir')
            same_session = allocation_directory and Path(allocation_directory).resolve() == directory.resolve()
            if not allocation_directory or same_session:
                proof = admission_exit(directory, path.parent, assignment, state)
                if proof:
                    state['detail'].update(resource_released=True, pbs_admission_exit=proof)
                    same_session = True  # route profiles in older workers omitted the PBS directory
            if (observe_rejection and state.get('state') == 'UNKNOWN'
                    and (owners is None or same_session or not allocation_directory)):
                # The original lock and dry-check exit receipt prove that the
                # exact attempt never started, even on older route profiles.
                proof = observe_rejection(path.parent, assignment)
                if proof:
                    state['detail'].update(resource_released=True, prelaunch_rejection=proof)
                    same_session = True
            if state.get('state') != 'TRANSFERRED' and (owners is None or same_session):
                incomplete.append(dict(assignment=assignment, state=state))
            continue
        preparation = assignment.get('request', {}).get('route', {}).get('kind') == 'preparation'
        files = {}
        candidates = (path.parent/name for name in _PREPARATION_FILES) if preparation else path.parent.rglob('*')
        for file in candidates:
            relative = str(file.relative_to(path.parent))
            if file.is_symlink():
                raise ValueError('Symlink in saved worker artifact')
            if file.is_file() and relative not in _OMIT_HARVEST and file.relative_to(path.parent).parts[0] != '_dry_check':
                files[relative] = (file.stat().st_size if manifest_only else
                                   base64.b64encode(file.read_bytes()).decode('ascii'))
        if 'OUTCOME.json' in files and (not completed_only or 'PROCESS_RETURN.json' in files):
            if not manifest_only:
                outcome = json.loads(base64.b64decode(files['OUTCOME.json']))
                hashes.update(outcome.get('artifacts', {}).values())
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


def attempt_control(root, value):
    """Address an already owned attempt on its original PBS worker only."""
    token, job, attempt_id = (safe_id(value[key]) for key in ('token', 'job_id', 'attempt_id'))
    directory = root/'allocations'/token
    allocation = read(directory/'ALLOCATION.json', {})
    if not value.get('pbs_job_id') or allocation.get('job_id') != value['pbs_job_id']:
        raise ValueError('PBS job differs from the saved allocation')
    profile = read(directory/'PROFILE.json')
    work = Path(profile['work_dir']).resolve()
    if not work.is_relative_to(root.resolve()/'workers'/safe_id(profile['host_id'])):
        raise ValueError('PBS worker spool is outside the fixed run')
    attempt = work/job/attempt_id
    if not attempt.is_dir() or attempt.resolve() != attempt:
        raise ValueError('PBS attempt spool is missing or redirected')
    identity = dict(job_id=job, attempt_id=attempt_id, request_sha256=value['request_sha256'])
    assignment = read(attempt/'ASSIGNMENT.json', {})
    saved = read(attempt/'WORKER_PROFILE.json', {})
    if (any(assignment.get(k) != v for k, v in identity.items()) or
            (saved.get('pbs') or {}).get('allocation_dir') != str(directory.resolve())):
        raise ValueError('PBS attempt does not belong to the selected request and allocation')
    owner = read(attempt/'PBS_ATTEMPT_OWNER.json', {})
    if (any(owner.get(k) != v for k, v in identity.items()) or owner.get('token') != token
            or owner.get('pbs_job_id') != value['pbs_job_id'] or owner.get('allocation_dir') != str(directory.resolve())):
        raise ValueError('This attempt has no individual stop support')
    request = attempt/'PBS_ATTEMPT_STOP_REQUEST.json'
    state = read(attempt/'WORKER_STATE.json', {})
    if value['action'] == 'attempt_stop' and not request.exists():
        if state.get('state') not in ('STARTING', 'RUNNING'):
            return dict(disposition='ALREADY_FINISHED', worker_state=state)
        worker = read(directory/'STATE.json', {})
        if (worker.get('state') == 'EXITED' or owner['clock_id'] != worker.get('clock_id') or
                qstat(value['pbs_job_id']).get(value['pbs_job_id'], {}).get('job_state') != 'R'):
            raise ValueError('Original PBS worker is not live on this allocation')
        atomic(request, owner)
    return dict(stop_requested=read(request) == owner, worker_state=state,
                interruption=read(attempt/'PBS_ATTEMPT_INTERRUPTED.json'))


def handle(root, value, finished=None, scheduler=None):
    action = value['action']
    if action in ('attempt_stop', 'attempt_status'):
        return attempt_control(root, value)
    if action == 'intent_present':
        directory = root/'allocations'/safe_id(value['token'])
        if directory.is_dir() and not directory.is_symlink():
            # Stop can write a notice before any submission exists. Keep it
            # as a fence, but it is not evidence of a submitted intent.
            return dict(present=any(path.name != 'CONTROL.json' for path in directory.iterdir()))
        return dict(present=directory.exists() or directory.is_symlink())
    if action == 'ensure':
        return ensure(root, value['spec'])
    if action == 'exchange':
        return exchange(root, value, finished, lambda root, finished: snapshot(
            root, finished, scheduler=scheduler))
    if action == 'snapshot':
        return dict(allocations=snapshot(root, finished, scheduler=scheduler, force=True))
    if action == 'harvest':
        return harvest(root/'allocations'/safe_id(value['token']), value.get('hashes'), value.get('owners'),
                       completed_only=value.get('completed_only') is True,
                       manifest_only=value.get('manifest_only') is True)
    if action == 'read_bulk_chunk':
        return read_bulk_chunk(root/'allocations'/safe_id(value['token']), value)
    if action == 'offline_results':
        directory = offline_root(root/'allocations'/safe_id(value['token']))
        return dict(jobs=read(directory/'RESULTS.json', {}), coverage=read(directory/'COVERAGE.json', {}))
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
    scheduler = SchedulerCache()
    for line in sys.stdin:
        try:
            answer = dict(ok=True, result=handle(root, json.loads(line), finished, scheduler))
        except Exception as error:
            answer = dict(ok=False, error=type(error).__name__ + ': ' + str(error))
        print(json.dumps(answer, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
