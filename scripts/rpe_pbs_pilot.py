"""Run-independent, single-node PBS reservation and run-scoped worker sessions.

Package with rpe_pbs_gateway.py as a zipapp. --root is the ordinary controller
gateway protocol. Administrative actions operate on the same shared pool.
"""
import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

import rpe_pbs_gateway as gateway
from rpe_pbs_gateway import atomic, read, safe_id


TERMINAL = {'F', 'X', 'SUBMIT_REJECTED'}
POOL_NAME = 'shared-node'
RESOURCES = dict(queue='workq', ncpus=96, mem='350gb', walltime='24:00:00')
SUCCESSOR_RESOURCES = dict(RESOURCES, walltime='72:00:00')


@contextmanager
def locked(pool):
    pool.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (pool/'POOL.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def owned_live_jobs():
    selected = subprocess.run(['/opt/pbs/bin/qselect', '-u', getpass.getuser()],
                              capture_output=True, text=True, check=True)
    ids = selected.stdout.split()
    return gateway.qstat(*ids) if ids else {}


def physical(pool, token, finished=None, *, scheduler=None, force=False):
    directory = pool/'allocations'/safe_id(token)
    row = gateway.snapshot(pool, finished, [directory], scheduler=scheduler, force=force)[0]
    if row['status'] == 'SUBMIT_REJECTED':
        # Even a nonzero qsub exit can be a lost connection after submission.
        jobs = owned_live_jobs() if scheduler is None else scheduler.discover(owned_live_jobs, force=force)
        matches = [job for job, detail in jobs.items()
                   if detail.get('Job_Name') == 'RPE'+token[:12]]
        if len(matches) > 1:
            raise ValueError('Ambiguous physical PBS reservation')
        if matches:
            atomic(directory/'JOB_ID.json', dict(job_id=matches[0], recovered=True))
            return gateway.snapshot(pool, finished, [directory], scheduler=scheduler, force=True)[0]
        row['status'] = 'SUBMIT_ERROR_RECONCILE'
    return row


def reserve_locked(pool, *, replace_job_id=None, alongside_search_job_id=None, alongside_search_job_ids=()):
    current = read(pool/'CURRENT.json')
    resources = RESOURCES
    if current:
        token = current['token']
        directory = pool/'allocations'/token
        observed = physical(pool, token)
        if observed['status'] not in TERMINAL:
            if (directory/'RELEASE.json').exists():
                raise ValueError('Explicit release in progress; reconcile this allocation first')
            gateway.submit_once(directory)
            return token
        if replace_job_id is not None:
            resources = read(directory/'SPEC.json')['resources']
            if (observed['job_id'] != replace_job_id or observed['status'] not in ('F', 'X')
                    or resources not in (RESOURCES, SUCCESSOR_RESOURCES)):
                raise ValueError('Expected the identified, ended full-node reservation')
    elif replace_job_id is not None:
        raise ValueError('No recorded reservation to replace')
    jobs = owned_live_jobs()
    alongside=set(alongside_search_job_ids)
    if alongside_search_job_id:alongside.add(alongside_search_job_id)
    for job_id, job in jobs.items():
        name = job.get('Job_Name', '')
        if not name.startswith('RPE') or job.get('job_state') in TERMINAL:
            continue
        # An operator may identify one independently managed search. Managed
        # pilot generations and unclassified scheduler records still block.
        if job_id in alongside and re.fullmatch(r'RPEsearch-[0-9a-f]{5}', name):
            continue
        raise ValueError('Another RPE PBS allocation is still present; reconcile it first')
    token = prepare_generation(pool, resources)
    atomic(pool/'CURRENT.json', dict(token=token))
    gateway.submit_once(pool/'allocations'/token)
    return token


def prepare_generation(pool, resources, predecessor=None, *, after_exit=False, when_available=False,
                       managed_host=None):
    token = uuid.uuid4().hex
    directory = pool/'allocations'/token
    directory.mkdir(parents=True)
    command = [sys.executable, '-B', str(Path(sys.argv[0]).resolve()),
               '--serve', str(directory)]
    script = ['#!/bin/bash', '#PBS -N RPE' + token[:12], '#PBS -q ' + resources['queue'],
              '#PBS -l select=1:ncpus={ncpus}:mem={mem}'.format(**resources),
              '#PBS -l walltime=' + resources['walltime'], '#PBS -r n']
    if after_exit:
        job_id = predecessor['job_id']
        if not re.fullmatch(r'[0-9]+\.[A-Za-z0-9.-]+', job_id):
            raise ValueError('A qualified PBS dependency job ID is required')
        script.append('#PBS -W depend=afterany:' + job_id)
    if resources.get('initial_hold'):
        script.append('#PBS -h')
    script += [
              '#PBS -o ' + str(directory/'PBS.stdout'), '#PBS -e ' + str(directory/'PBS.stderr'),
              'set -eu', 'umask 077', 'exec ' + shlex.join(command)]
    (directory/'worker.pbs').write_text('\n'.join(script) + '\n')
    spec = dict(token=token, resources=resources, profile=dict(host_id=managed_host or 'pbs-0'))
    if managed_host is not None:
        spec['run_managed'] = True
    if predecessor:
        spec['predecessor'] = predecessor
    if after_exit:
        spec['after_predecessor_exit'] = True
    if when_available:
        spec['retain_predecessor'] = True
    atomic(directory/'SPEC.json', spec)
    return token


def successor_mode(spec):
    if spec.get('retain_predecessor'):
        return 'available'
    return 'after-exit' if spec.get('after_predecessor_exit') else 'drain'


def cancel_dependency_hold(pool, directory, predecessor):
    """Withdraw exactly one unstarted successor held on this predecessor."""
    marker = directory/'REPLACED_DEPENDENCY_HOLD.json'
    row = physical(pool, directory.name)
    job_id = row['job_id']
    proof = dict(job_id=job_id, predecessor=predecessor)
    if read(marker) == proof and row['status'] in TERMINAL:
        return
    scheduler = gateway.qstat(job_id).get(job_id, {})
    dependency = scheduler.get('depend', '').split('@', 1)[0]
    if (row['status'] != 'H' or scheduler.get('job_state') != 'H' or
            dependency != 'afterany:'+predecessor or (directory/'ALLOCATION.json').exists()
            or any(sessions(directory))):
        raise ValueError('Only an unstarted successor held on this predecessor can be replaced')
    atomic(marker, proof)
    result = gateway.handle(pool, dict(action='delete', token=directory.name))
    if result['returncode'] or physical(pool, directory.name)['status'] not in TERMINAL:
        raise RuntimeError('The previous held successor is not confirmed cancelled')


def stage_successor(pool, job_id, *, after_exit=False, when_available=False):
    if after_exit and when_available:
        raise ValueError('Select one PBS handover mode')
    mode = 'available' if when_available else 'after-exit' if after_exit else 'drain'
    with locked(pool):
        current = read(pool/'CURRENT.json', {})['token']
        old = pool/'allocations'/safe_id(current)
        spec = read(old/'SPEC.json')
        if spec.get('predecessor', {}).get('job_id') == job_id:
            if successor_mode(spec) != mode:
                raise ValueError('The existing successor has a different handover mode')
            return physical(pool, current)
        allowed_resources = (RESOURCES, SUCCESSOR_RESOURCES) if mode != 'drain' else (RESOURCES,)
        if read(old/'JOB_ID.json', {}).get('job_id') != job_id or spec['resources'] not in allowed_resources:
            raise ValueError('The selected PBS predecessor is no longer current')
        if (old/'RELEASE.json').exists():
            raise ValueError('Current pilot already has a release request')
        successor = read(old/'SUCCESSOR.json')
        if successor is not None:
            directory = pool/'allocations'/safe_id(successor['token'])
            next_spec = read(directory/'SPEC.json')
            if successor_mode(next_spec) != mode:
                if (when_available and successor_mode(next_spec) == 'after-exit' and
                        next_spec.get('predecessor') == dict(token=current, job_id=job_id)):
                    if physical(pool, current)['status'] != 'R':
                        raise ValueError('Expected a running predecessor')
                    cancel_dependency_hold(pool, directory, job_id)
                    successor = None
                else:
                    raise ValueError('The staged successor has a different handover mode')
        if successor is None:
            observed = physical(pool, current)
            if observed['job_id'] != job_id or observed['status'] != 'R':
                raise ValueError('Expected a running predecessor')
            jobs = owned_live_jobs()
            # Separate research pools keep their own queues when this pool
            # has its predecessor ownership explicitly identified.
            pool_names = {'RPE'+p.name[:12] for p in (pool/'allocations').iterdir() if (p/'SPEC.json').is_file()}
            for key, detail in jobs.items():
                name = detail.get('Job_Name', '')
                if (key != job_id and detail.get('job_state') not in TERMINAL
                        and (name in pool_names if mode != 'drain' else name.startswith('RPE'))):
                    raise ValueError('Conflicting RPE reservation in queue')
            token = prepare_generation(pool, SUCCESSOR_RESOURCES, dict(token=current, job_id=job_id),
                                       after_exit=after_exit, when_available=when_available)
            successor = dict(token=token)
            atomic(old/'SUCCESSOR.json', successor)
        directory = pool/'allocations'/safe_id(successor['token'])
        gateway.submit_once(directory)
        return physical(pool, directory.name)


def retire_predecessor(pool, generation, predecessor, job_id, *, natural_exit=None, retain=False):
    """Called on the acquired successor, before allowing any worker to start."""
    old = pool/'allocations'/safe_id(predecessor['token'])
    with locked(pool):
        if (read(old/'SUCCESSOR.json') != dict(token=generation.name) or
                read(old/'JOB_ID.json')['job_id'] != predecessor['job_id']):
            raise ValueError('Predecessor ownership changed')
        current = read(pool/'CURRENT.json')['token']
        if current not in (old.name, generation.name) or (generation/'RELEASE.json').exists():
            raise ValueError('Successor was released or superseded')
        release = read(old/'RELEASE.json')
        if release and release.get('successor') != generation.name:
            raise ValueError('Independent release of predecessor')
        if natural_exit or retain:
            atomic(pool/'CURRENT.json', dict(token=generation.name))
            handover = dict(state='ACTIVE', old_job_id=predecessor['job_id'], new_job_id=job_id, epoch=time.time())
            if retain:
                handover['retained_predecessor'] = predecessor['token']
            else:
                handover['old_scheduler_state'] = natural_exit
            atomic(generation/'HANDOVER.json', handover)
            return
        if not release:
            atomic(old/'RELEASE.json', dict(epoch=time.time(), successor=generation.name))
        for directory in sessions(old):
            if not (directory/'SESSION_END.json').exists():
                stop_session(directory)
        atomic(pool/'CURRENT.json', dict(token=generation.name))
        atomic(generation/'HANDOVER.json', dict(state='DRAINING_PREDECESSOR',
            old_job_id=predecessor['job_id'], new_job_id=job_id, epoch=time.time()))


def predecessor_finished(pool, predecessor):
    old = pool/'allocations'/predecessor['token']
    row = physical(pool, old.name)
    if row['job_id'] != predecessor['job_id']:
        raise ValueError('Wrong predecessor job ID')
    if row['status'] in ('F', 'X'):
        return row['status']
    # PBS can purge an expired job while its successor is still queued.
    # Check both the scheduler's live set and the lock held by the old node.
    if row['status'] == 'HISTORY_MISSING_RECONCILE_SPOOL':
        if predecessor['job_id'] in owned_live_jobs():
            return None
        if read(old/'ALLOCATION.json', {}).get('job_id') != predecessor['job_id']:
            raise ValueError('Missing predecessor allocation receipt')
        with (old/'NODE.lock').open('r') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
        return 'ABSENT_FROM_LIVE_QUEUE_AND_NODE_LOCK_FREE'
    return None


def sessions(generation):
    for path in sorted((generation/'sessions').glob('*.json')):
        yield Path(read(path)['directory'])


def end(directory, reason, **detail):
    if not (directory/'SESSION_END.json').exists():
        atomic(directory/'SESSION_END.json', dict(reason=reason, epoch=time.time(), **detail))


def stop_session(directory):
    atomic(directory/'CONTROL.json', dict(stop=True, release_when_idle=True))
    if not (directory/'SESSION_START.json').exists():
        end(directory, 'DETACHED_BEFORE_START')


def update_session_control(directory, control):
    """Never turn a permanent STOP into a resume. Pause is separate."""
    prior=read(directory/'CONTROL.json',{})
    if prior.get('stop'):return
    if control.get('stop') or (control.get('release_when_idle') and not (directory/'SESSION_START.json').exists()):
        stop_session(directory)
    else:
        # A pause keeps the session and its owned attempts intact, even if idle.
        atomic(directory/'CONTROL.json',dict(control,release_when_idle=False) if control.get('pause') else control)


def successor_work_dir(root, generation):
    return root/'workers'/'pbs-0'/'.generations'/generation.name


def attach(root, pool, spec):
    token = safe_id(spec['token'])
    directory = root/'allocations'/token
    if spec['profile']['host_id'] != 'pbs-0' or spec['resources'] != RESOURCES:
        raise ValueError('Run resources differ from the shared full-node profile')
    if Path(spec['profile']['work_dir']).resolve() != root/'workers'/'pbs-0':
        raise ValueError('Run worker spool does not belong to this namespace')
    with locked(pool):
        prior = read(directory/'SPEC.json')
        if prior is not None and prior != spec:
            raise ValueError('Existing allocation specification differs')
        binding = read(directory/'PILOT.json')
        if prior is not None and binding is None:
            # This token belongs to a historical, directly submitted PBS job.
            if not (directory/'QSUB_STARTED.json').exists():
                raise ValueError('Legacy intent without a submit receipt needs reconciliation')
            return dict(token=token, existing=True)
        if binding is None:
            replay = spec.get('reason', {}).get('replay_pilot')
            replacement = spec.get('reason', {}).get('retained_replacement')
            if retained := spec.get('reason', {}).get('retained_pilot'):
                generation = safe_id(retained['token'])
                target = pool/'allocations'/generation
                observed = physical(pool, generation)
                if (observed.get('job_id') != retained['job_id'] or observed['status'] != 'R'
                        or (target/'RELEASE.json').exists()
                        or read(target/'SPEC.json')['resources'] not in (RESOURCES, SUCCESSOR_RESOURCES)):
                    raise ValueError('Selected existing PBS node is unavailable')
                binding = dict(token=generation)
            elif replacement:
                generation = safe_id(replacement['pilot_token'])
                target = pool/'allocations'/generation
                current = read(pool/'CURRENT.json', {}).get('token')
                handover = read(target/'HANDOVER.json', {})
                if (current != generation or handover.get('state') != 'ACTIVE'
                        or handover.get('retained_predecessor') != replacement['predecessor']
                        or (target/'RELEASE.json').exists() or physical(pool, generation)['status'] != 'R'):
                    raise ValueError('The selected replacement PBS node is unavailable')
                binding = dict(token=generation)
            elif replay:
                current = read(pool/'CURRENT.json', {})
                generation = safe_id(replay['token'])
                observed = physical(pool, generation)
                if (current.get('token') != generation or observed['status'] != 'R'
                        or observed['job_id'] != replay['job_id']
                        or (pool/'allocations'/generation/'RELEASE.json').exists()):
                    raise ValueError('Original PBS reservation is unavailable for attempt replay')
                binding = dict(token=generation)
            else:
                binding = dict(token=reserve_locked(pool, alongside_search_job_ids=spec.get('reason',{}).get('alongside_search_job_ids',())))
            atomic(directory/'PILOT.json', binding)
        generation = pool/'allocations'/binding['token']
        profile = dict(spec['profile'])
        profile['pbs'] = dict(profile.get('pbs') or {}, allocation_dir=str(directory))
        if read(generation/'SPEC.json').get('retain_predecessor'):
            profile['work_dir'] = str(successor_work_dir(root, generation))
        atomic(directory/'PROFILE.json', profile)
        atomic(directory/'SPEC.json', spec)
        entry = generation/'sessions'/(token+'.json')
        locator = dict(directory=str(directory))
        if entry.exists() and read(entry) != locator:
            raise ValueError('Worker session token belongs to a different run')
        atomic(entry, locator)
    return dict(token=token, existing=prior is not None, pilot=binding['token'])


def managed_pool(root, host):
    """An ordinary run owns one physical reservation pool per logical host."""
    return root/'managed-nodes'/safe_id(host)


def session_pool(root, binding):
    host = binding.get('managed_host')
    return managed_pool(root, host) if host is not None else root.parent/POOL_NAME


def compatible_resources(have, need):
    # A reservation is never resized during replacement. The worker obtains
    # its actual CPU/RAM/remaining-walltime from PBS at each session start.
    return (have['queue'] == need['queue'] and have['ncpus'] >= need['ncpus']
            and gateway.memory_bytes(have['mem']) >= gateway.memory_bytes(need['mem']))


def attach_managed(root, spec):
    """Ordinary multi-host PBS: submit a keeper, then attach a run session.

    The controller's session token, REQUEST/attempt ownership and spool are
    distinct from the physical PBS token. A later normal Resume may attach a
    new fixed worker to the same live node only after its predecessor ended.
    Legacy directly submitted tokens are never converted here.
    """
    root = Path(root).resolve()
    token = safe_id(spec['token'])
    host = safe_id(spec['profile']['host_id'])
    resources = spec['resources']
    if type(resources['ncpus']) is not int or resources['ncpus'] < 1:
        raise ValueError('Explicit positive PBS ncpus required')
    gateway.memory_bytes(resources['mem']); gateway.seconds(resources['walltime'])
    if resources['queue'] not in ('workq', 'large'):
        raise ValueError('Unconfigured queue')
    pool = managed_pool(root, host)
    directory = root/'allocations'/token
    with locked(pool):
        prior = read(directory/'SPEC.json')
        binding = read(directory/'PILOT.json')
        if prior is not None and prior != spec:
            raise ValueError('Existing allocation specification differs')
        if prior is not None and binding is None:
            raise ValueError('Existing direct allocation must not be converted')
        if binding is not None and binding.get('managed_host') != host:
            raise ValueError('Existing allocation belongs to a different managed pool')
        if binding is None:
            current = read(pool/'CURRENT.json', {}).get('token')
            reason = spec.get('reason') or {}
            replay = reason.get('replay_pilot')
            if reason.get('retained_pilot') or reason.get('retained_replacement'):
                raise ValueError('Shared-pilot continuation requires its original gateway')
            if current:
                generation = pool/'allocations'/safe_id(current)
                observed = physical(pool, current)
                if replay and (current != replay['token'] or observed.get('job_id') != replay['job_id']
                               or observed['status'] != 'R'):
                    raise ValueError('Original managed PBS reservation is unavailable for replay')
                if observed['status'] in TERMINAL:
                    current = None
                elif (generation/'RELEASE.json').exists():
                    raise ValueError('Managed PBS reservation release is pending')
                elif any(not (session/'SESSION_END.json').exists() for session in sessions(generation)):
                    raise ValueError('Previous managed worker session is still owned')
                elif not compatible_resources(read(generation/'SPEC.json')['resources'], resources):
                    raise ValueError('Existing managed PBS reservation cannot satisfy requested resources')
            elif replay:
                raise ValueError('Original managed PBS reservation is unavailable for replay')
            if current is None:
                current = prepare_generation(pool, resources, managed_host=host)
                atomic(pool/'CURRENT.json', dict(token=current))
            binding = dict(token=current, managed_host=host)
            atomic(directory/'PILOT.json', binding)
        generation = pool/'allocations'/safe_id(binding['token'])
        # A repeated ensure does not overwrite a running session's fixed files
        # or restart a completed session. Complete only an interrupted intent.
        if prior is None:
            profile = dict(spec['profile'])
            profile['pbs'] = dict(profile.get('pbs') or {}, allocation_dir=str(directory))
            atomic(directory/'PROFILE.json', profile)
            atomic(directory/'SPEC.json', spec)
        entry = generation/'sessions'/(token+'.json')
        locator = dict(directory=str(directory))
        previous = read(entry)
        if previous is not None and previous != locator:
            raise ValueError('Worker session token belongs to a different run')
        if previous is None:
            # The initial binding may have survived a lost response before
            # the session locator was written. Do not add that interrupted
            # intent behind a subsequently acquired owner of this node.
            if any(session != directory and not (session/'SESSION_END.json').exists()
                   for session in sessions(generation)):
                raise ValueError('Previous managed worker session is still owned')
            atomic(entry, locator)
        # Reconcile the same durable reservation on any lost response.
        gateway.submit_once(generation)
    return dict(token=token, existing=prior is not None, pilot=binding['token'])


def managed_exchange(root, value, finished=None, scheduler=None):
    controls = dict(value.get('controls', {}))
    for token, control in list(controls.items()):
        directory = root/'allocations'/safe_id(token)
        binding = read(directory/'PILOT.json')
        if binding is not None:
            with locked(session_pool(root, binding)):
                update_session_control(directory, control)
            del controls[token]
    return gateway.exchange(root, dict(value, controls=controls), finished,
                            lambda root, finished: session_snapshot(root, finished, scheduler=scheduler))


def release_managed_node(root, host):
    """Explicit physical release, not the worker's normal stop/detach."""
    pool = managed_pool(root, host)
    with locked(pool):
        current = read(pool/'CURRENT.json', {}).get('token')
        if not current:
            return dict(status='UNRESERVED')
        generation = pool/'allocations'/safe_id(current)
        if any(not (directory/'SESSION_END.json').exists() for directory in sessions(generation)):
            raise ValueError('Drain and reconcile managed worker sessions before releasing the PBS reservation')
        row = physical(pool, current)
        if row['status'] not in TERMINAL:
            atomic(generation/'RELEASE.json', dict(epoch=time.time()))
            if row['job_id'] and row['status'] in ('Q', 'H', 'W'):
                result = gateway.handle(pool, dict(action='delete', token=current))
                if result['returncode']:
                    raise RuntimeError('PBS physical release is pending')
        return dict(status=row['status'], job_id=row['job_id'], release_requested=True)


def detach_managed_session(root, directory):
    binding = read(directory/'PILOT.json')
    pool = session_pool(root, binding)
    with locked(pool):
        stop_session(directory)
        # Existing controller cancellation of an unstarted queued submission
        # may still withdraw its unused reservation. Never qdel a running node
        # merely to replace a worker.
        if binding.get('managed_host') is not None and not (directory/'SESSION_START.json').exists():
            generation = pool/'allocations'/safe_id(binding['token'])
            if not any(not (s/'SESSION_END.json').exists() for s in sessions(generation)):
                row = physical(pool, binding['token'])
                if row['job_id'] and row['status'] in ('Q', 'H', 'W'):
                    atomic(generation/'RELEASE.json', dict(epoch=time.time()))
                    return gateway.handle(pool, dict(action='delete', token=binding['token']))
        return dict(returncode=0, detached=True)

def session_snapshot(root, finished=None, *, scheduler=None, force=False):
    pool = root.parent/POOL_NAME
    current = read(pool/'CURRENT.json', {}).get('token')
    handover = read(pool/'allocations'/safe_id(current)/'HANDOVER.json', {}) if current else {}
    retained = handover.get('retained_predecessor') if handover.get('state') == 'ACTIVE' else None
    legacy, shared = [], []
    for directory in sorted((root/'allocations').glob('*')):
        if (directory/'SPEC.json').exists():
            (shared if (directory/'PILOT.json').exists() else legacy).append(directory)
    result = gateway.snapshot(root, finished, legacy, scheduler=scheduler, force=force)
    observations = {}
    ended_sessions = {directory: read(directory/'SESSION_END.json') for directory in shared}
    terminal_sessions = {}
    if scheduler is not None:
        for directory, ended in ended_sessions.items():
            if ended and directory not in scheduler.terminal_sessions:
                token = read(directory/'PILOT.json')['token']
                terminal_sessions.setdefault(token, []).append(directory)
    for directory in shared:
        binding = read(directory/'PILOT.json')
        token = binding['token']
        physical_pool = session_pool(root, binding)
        key = (physical_pool, token)
        if key not in observations:
            observations[key] = physical(physical_pool, token, finished, scheduler=scheduler,
                                         force=force or token in terminal_sessions)
            if token in terminal_sessions:
                scheduler.terminal_sessions.update(terminal_sessions[token])
        row = dict(observations[key])
        row.update(token=directory.name, directory=str(directory), pilot_token=token, reservation_status=row['status'],
                   allocation=read(directory/'ALLOCATION.json'), worker=read(directory/'STATE.json'))
        ended = ended_sessions[directory]
        if ended:
            row.update(status='F', exit_status=ended.get('returncode'), session_end=ended)
        elif row['status'] == 'R' and not (directory/'SESSION_START.json').exists():
            row['status'] = 'Q'  # Another run may still be draining on this node.
        control=read(directory/'CONTROL.json',{})
        if control.get('stop') or control.get('pause'):
            row['worker'] = dict(row.get('worker') or {}, accepting=False)
        if binding.get('managed_host') is None and retained == token and control.get('stop'):
            row['retained_replacement'] = current
        row['session_status'] = row['status']
        result.append(row)
    return result


def handle(root, value, finished, scheduler=None):
    pool = root.parent/POOL_NAME
    action = value['action']
    if action == 'managed_capabilities':
        return dict(managed_keeper=True, mode='shared-pool', worker_session_replacement=True)
    if action == 'ensure':
        return attach(root, pool, value['spec'])
    if action == 'snapshot':
        return dict(allocations=session_snapshot(root, finished, scheduler=scheduler, force=True))
    if action == 'exchange':
        return managed_exchange(root, value, finished, scheduler)
    if action == 'delete':
        directory = root/'allocations'/safe_id(value['token'])
        if (directory/'PILOT.json').exists():
            return detach_managed_session(root, directory)
    return gateway.handle(root, value, finished, scheduler)


def claim(pool, generation, job_id):
    with locked(pool):
        if (generation/'RELEASE.json').exists():
            return None
        for directory in sessions(generation):
            if (directory/'SESSION_END.json').exists():
                continue
            if (directory/'SESSION_START.json').exists():
                raise ValueError('Previously started worker must be reconciled before reuse')
            if read(directory/'CONTROL.json', {}).get('stop'):
                stop_session(directory)
                continue
            if read(directory/'CONTROL.json', {}).get('pause'):
                continue
            if read(generation/'SPEC.json').get('retain_predecessor'):
                profile = read(directory/'PROFILE.json', {})
                if profile.get('work_dir') != str(successor_work_dir(directory.parents[1], generation)):
                    continue  # Older gateways can bind but cannot share the predecessor's spool.
            atomic(directory/'SESSION_START.json', dict(job_id=job_id, epoch=time.time()))
            return directory
    return None


def serve(generation):
    pool = generation.parents[1]
    generation_spec = read(generation/'SPEC.json')
    predecessor = generation_spec.get('predecessor')
    after_exit = generation_spec.get('after_predecessor_exit') is True
    while True:
        try:
            actual = gateway.record_allocation(generation)
            if predecessor:
                observed = physical(pool, generation.name)
                if (observed['status'] != 'R' or observed['job_id'] != actual['job_id'] or
                        observed['scheduler'].get('Job_Name') != 'RPE'+generation.name[:12] or
                        actual['walltime_seconds'] != gateway.seconds(SUCCESSOR_RESOURCES['walltime'])):
                    raise ValueError('Successor scheduler receipt differs')
            break
        except (KeyError, OSError, RuntimeError, subprocess.SubprocessError):
            if not predecessor:
                raise
            if (generation/'RELEASE.json').exists():
                return
            time.sleep(5)
    job_id = actual['job_id']
    requested = generation_spec['resources'] if generation_spec.get('run_managed') else RESOURCES
    if (actual['ncpus'] != requested['ncpus'] or actual['memory_bytes'] != gateway.memory_bytes(requested['mem'])):
        raise ValueError('Scheduler allocation differs from requested pool')
    # PBS is Linux. Reap detached supervisors from a finished worker before
    # lending its resource pool to a different run, including UNKNOWN exits.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot track detached worker children')
    deadline = time.monotonic() + actual['remaining_seconds']
    with (generation/'NODE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if predecessor and generation_spec.get('retain_predecessor'):
            retire_predecessor(pool, generation, predecessor, job_id, retain=True)
        elif predecessor:
            if not after_exit:
                retire_predecessor(pool, generation, predecessor, job_id)
            while time.monotonic() < deadline and not (generation/'RELEASE.json').exists():
                try:
                    state = predecessor_finished(pool, predecessor)
                except (OSError, RuntimeError, subprocess.SubprocessError):
                    state = None  # A transient scheduler outage must not open admission.
                if state:
                    if after_exit:
                        retire_predecessor(pool, generation, predecessor, job_id, natural_exit=state)
                    else:
                        atomic(generation/'HANDOVER.json', dict(state='ACTIVE', old_job_id=predecessor['job_id'],
                            old_scheduler_state=state, new_job_id=job_id, epoch=time.time()))
                    break
                time.sleep(5)
        while time.monotonic() < deadline and not (generation/'RELEASE.json').exists():
            directory = claim(pool, generation, job_id)
            if directory is None:
                time.sleep(1)
                continue
            spec = read(directory/'SPEC.json')
            allocation = dict(actual, remaining_seconds=max(0, int(deadline-time.monotonic())),
                              recorded_epoch=time.time())
            atomic(directory/'ALLOCATION.json', allocation)
            command = [spec['python'], '-B', str(Path(spec['transport_root'])/'scripts/run_rpe_pbs_worker.py'),
                       '--profile', str(directory/'PROFILE.json')]
            with (directory/'PBS.stdout').open('ab') as out, (directory/'PBS.stderr').open('ab') as err:
                try:
                    returncode = subprocess.run(command, stdout=out, stderr=err).returncode
                except OSError as error:
                    err.write((str(error)+'\n').encode())
                    returncode = 127
            # Whether or not the worker exited cleanly, reap every detached
            # child before offering this node to another session.
            while True:
                try:
                    os.waitpid(-1, 0)
                except ChildProcessError:
                    break
            reason = 'WORKER_EXIT' if returncode == 0 else 'WORKER_ERROR'
            end(directory, reason, returncode=returncode)


def admin(pool, action, *, replace_job_id=None, alongside_search_job_id=None):
    with locked(pool):
        current = read(pool/'CURRENT.json')
        if action in ('reserve', 'reacquire'):
            token = reserve_locked(pool, replace_job_id=replace_job_id,
                                   alongside_search_job_id=alongside_search_job_id)
        elif current:
            token = current['token']
        else:
            return dict(status='UNRESERVED')
        row = physical(pool, token)
        generation = pool/'allocations'/token
        if action == 'release' and row['status'] not in TERMINAL:
            atomic(generation/'RELEASE.json', dict(epoch=time.time()))
            for directory in sessions(generation):
                if not (directory/'SESSION_END.json').exists():
                    stop_session(directory)
            if row['job_id'] and row['status'] in ('Q','H','W'):
                result = gateway.handle(pool, dict(action='delete', token=token))
                if result['returncode']:
                    raise RuntimeError('PBS release pending: '+result['stderr'])
        return dict(token=token, job_id=row['job_id'], status=row['status'],
                    release_requested=(generation/'RELEASE.json').exists(),
                    resources=read(generation/'SPEC.json')['resources'], scheduler=row['scheduler'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--root', type=Path)
    group.add_argument('--serve', type=Path)
    group.add_argument('--pool', type=Path)
    parser.add_argument('--action', choices=('reserve','reacquire','status','release','stage-72h',
                        'stage-after-exit','stage-when-available'), default='status')
    parser.add_argument('--replace-job-id')
    parser.add_argument('--alongside-search-job-id')
    args = parser.parse_args()
    os.umask(0o077)
    if args.alongside_search_job_id and args.action != 'reacquire':
        parser.error('--alongside-search-job-id is only valid for reacquire')
    if args.action in ('stage-72h', 'stage-after-exit', 'stage-when-available'):
        if not args.pool or not args.replace_job_id:
            parser.error(args.action + ' needs --pool and --replace-job-id')
        print(json.dumps(stage_successor(args.pool.resolve(), args.replace_job_id,
                                        after_exit=args.action == 'stage-after-exit',
                                        when_available=args.action == 'stage-when-available')))
    elif args.action == 'reacquire':
        if not args.pool or not args.replace_job_id:
            parser.error('reacquire needs --pool and --replace-job-id')
        print(json.dumps(admin(args.pool.resolve(), args.action, replace_job_id=args.replace_job_id,
                               alongside_search_job_id=args.alongside_search_job_id)))
    elif args.serve:
        serve(args.serve.resolve())
    elif args.pool:
        print(json.dumps(admin(args.pool.resolve(), args.action)))
    else:
        root = args.root.resolve()
        if not re.fullmatch(r'[0-9a-f]{20}', root.name):
            raise ValueError('Pilot requires the ordinary per-run PBS namespace')
        root.mkdir(parents=True, exist_ok=True)
        finished = {}
        scheduler = gateway.SchedulerCache()
        for line in sys.stdin:
            try:
                answer = dict(ok=True, result=handle(root, json.loads(line), finished, scheduler))
            except Exception as error:
                answer = dict(ok=False, error=type(error).__name__ + ': ' + str(error))
            print(json.dumps(answer, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
