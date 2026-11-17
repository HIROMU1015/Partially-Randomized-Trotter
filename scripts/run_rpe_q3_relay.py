#!/usr/bin/env python3
"""Run the local0 qg2-controller/qg1-mailbox relay without shared storage.

Only the server2 controller selects candidates and reserves/adopts evaluations.
Stopping this relay leaves all received server1 jobs running independently.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import threading
try:
    import fcntl
except ImportError:  # Native Windows local0; numerical workers remain Linux.
    fcntl = None
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.rpe_q3_coordinator import atomic_json
from trottertracks.rpe_q3_timing import stamp


def acquire_lock(stream):
    """One local relay process, using the platform's ordinary file lock."""
    if fcntl is not None:
        fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
        return
    import msvcrt
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write('\0'); stream.flush()
    stream.seek(0)
    try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error: raise BlockingIOError('Another relay owns this lock') from error


def background_options(platform_name=None):
    if (platform_name or os.name) == 'nt':
        return dict(creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0x8) |
                    getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x200))
    return dict(start_new_session=True)


def child_options(platform_name=None):
    """SSH must stay invisible when its parent relay has no Windows console."""
    if (platform_name or os.name) == 'nt':
        return dict(creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000))
    return {}


class Relay:
    def __init__(self, configuration):
        self.configuration = Path(configuration).resolve()
        self.config = json.loads(self.configuration.read_text(encoding='utf-8-sig'))
        self.directory = Path(self.config['state_directory']).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ssh = self.config.get('ssh_command', ['ssh'])
        self.controller = self.config['controller']
        self.poll_seconds = float(self.config.get('poll_seconds', 1))
        maximum = float(self.config.get('retry_backoff_max_seconds', 30))
        if self.poll_seconds <= 0 or maximum <= 0:
            raise ValueError('Positive relay poll and retry intervals required')
        self.retry_max_seconds = max(self.poll_seconds, maximum)
        self.url = 'http://127.0.0.1:'+str(self.controller.get('local_port', 18763))
        self.tunnel = None
        self.discovery_pool = None
        self.pending_discovery = {}
        self.pending_ticks = {}
        self.log_lock = threading.Lock()
        started = time.time()
        self.state = dict(pid=os.getpid(), configuration=str(self.configuration), workers={},
                          started_epoch=started, heartbeat_epoch=started, stopped=False,
                          transport='local0 SSH controller forwarding + worker mailbox')

    def log(self, event, **value):
        row = dict(stamp(), event=event, **value)
        with self.log_lock:
            with (self.directory/'EVENTS.jsonl').open('a') as output: output.write(json.dumps(row)+'\n')

    def post(self, path, value):
        payload = json.dumps(value).encode()
        request = urllib.request.Request(self.url+path, payload, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.config.get('request_timeout_seconds', 60)) as response:
            return json.load(response)

    def failed(self, state, error, event, **detail):
        failures = state.get('failure_count', 0)+1
        delay = self.poll_seconds
        for _ in range(failures-1):
            delay = min(self.retry_max_seconds, delay*2)
            if delay >= self.retry_max_seconds: break
        state.update(error=str(error), failure_count=failures, last_error_epoch=time.time(),
                     next_retry_epoch=time.time()+delay, retry_seconds=delay)
        self.log(event, error=str(error), failure_count=failures, next_retry_epoch=state['next_retry_epoch'],
                 retry_seconds=delay, **detail)

    def succeeded(self, state):
        state.update(failure_count=0, next_retry_epoch=0, retry_seconds=0)
        state.pop('error', None)

    def remote(self, worker, action, value=None):
        command = [worker.get('python', 'python3'), '-B', worker['root'].rstrip('/')+'/scripts/run_rpe_q3_worker.py',
                   '--profile', worker['profile'], '--mailbox', action]
        # Reuse local0's published WSL mailbox transport (6e70f5d13061).
        # Scientific execution still resolves the route-specific saved profile.
        if worker.get('transport') == 'wsl':
            command = [worker.get('wsl_executable', 'wsl.exe'), '-d', worker['distribution'], '--exec'] + command
        else:
            alias = worker['ssh_alias']
            if not alias or alias.startswith('-'): raise ValueError('Invalid existing SSH alias')
            command = self.ssh+['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', alias, shlex.join(command)]
        started = time.monotonic()
        process = subprocess.run(command,
            input=json.dumps(value or {}), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding='utf-8', errors='replace',
            timeout=self.config.get('request_timeout_seconds', 60), **child_options())
        if process.returncode:
            raise RuntimeError('Worker mailbox '+action+' failed ('+str(process.returncode)+'): '+process.stderr[-1500:])
        result = json.loads(process.stdout)
        if action == 'status' and worker.get('participation_file'):
            participation = Path(worker['participation_file'])
            enabled = participation.is_file() and json.loads(participation.read_text(encoding='utf-8-sig')).get('accepting_new_jobs') is True
            result['accepting_new_jobs'] = enabled
            if not enabled:
                # Completion transfer and reconciliation above the poll continue.
                result['slots'] = 0
        return result, time.monotonic()-started

    def ensure_tunnel(self):
        if self.config.get('external_tunnel'): return True
        state = self.state.setdefault('tunnel', dict(failure_count=0, next_retry_epoch=0))
        if self.tunnel is not None:
            code = self.tunnel.poll()
            if code is None:
                # A process that survives several normal polls is healthy;
                # an immediate exit must not reset its retry delay each time.
                if time.time()-state['started_epoch'] >= max(5, 2*self.poll_seconds):
                    self.succeeded(state)
                return True
            self.tunnel = None
            self.failed(state, 'SSH tunnel exited ('+str(code)+')', 'TUNNEL_PENDING', returncode=code)
        if time.time() < state.get('next_retry_epoch', 0): return False
        alias = self.controller.get('ssh_alias', 'qg2')
        if not alias or alias.startswith('-'): raise ValueError('Invalid controller SSH alias')
        local_port = int(self.controller.get('local_port', 18763))
        remote_port = int(self.controller.get('remote_port', 8763))
        try:
            with (self.directory/'TUNNEL.log').open('a') as log:
                self.tunnel = subprocess.Popen(self.ssh+['-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                    '-L', '127.0.0.1:%d:127.0.0.1:%d' % (local_port, remote_port), alias],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **child_options())
        except OSError as error:
            self.failed(state, error, 'TUNNEL_PENDING')
            return False
        state.update(pid=self.tunnel.pid, started_epoch=time.time())
        self.log('TUNNEL_START', pid=self.tunnel.pid, ssh_alias=alias)
        return True

    def _poll_value(self, status):
        value = dict(host_id=status['host_id'], slots=status['slots'], capacity=status.get('capacity'),
                     source_sha=status['source_sha'], cached_prefixes=status['cached_prefixes'],
                     cached_artifact_sha256=status.get('cached_artifact_sha256', []))
        if 'accepting_new_jobs' in status:
            value['accepting_new_jobs'] = status['accepting_new_jobs']
        if status.get('resource_report') is not None:
            value['resource_report'] = status['resource_report']
        return value

    def _discover(self, worker):
        # Status and current-request checks may run concurrently for independent
        # hosts; no thread claims a job or delivers an assignment here.
        started = stamp()
        status, seconds = self.remote(worker, 'status')
        if status.get('host_id') != worker['host_id']:
            raise ValueError('Mailbox host differs from relay profile')
        poll = self._poll_value(status)
        observed = self.post('/v1/poll', dict(poll, observe_only=True))
        challenges = observed.get('eligibility_challenges') or []
        if challenges and status['slots']:
            facts, _ = self.remote(worker, 'readiness', dict(challenges=challenges))
            if facts.get('host_id') != status['host_id']:
                raise ValueError('Readiness response host differs')
            poll.update(host_fingerprints=facts.get('host_fingerprints', {}),
                        readiness_errors=facts.get('readiness_errors', {}))
            observed = self.post('/v1/poll', dict(poll, observe_only=True))
        return status, seconds, started, poll, observed

    def _recover_artifacts(self, worker, attempt):
        hashes = list(dict.fromkeys(attempt.get('artifact_manifest', {}).values()))
        if not hashes:return
        missing = self.post('/v1/artifacts/missing', dict(host_id=worker['host_id'], hashes=hashes))['missing']
        for sha in missing:
            response, _ = self.remote(worker, 'artifacts-get', dict(hashes=[sha]))
            self.post('/v1/artifacts/put', dict(host_id=worker['host_id'], blobs=response['blobs']))
            self.log('ARTIFACT_RECOVERED', host_id=worker['host_id'], sha256=sha)

    def _deliver_artifacts(self, worker, assignment):
        request=assignment.get('request', {})
        if (request.get('route') or {}).get('kind') != 'preparation':return
        spec=request['preparation'];inputs=assignment.get('artifact_manifest', spec.get('inputs', {}))
        hashes=list(inputs.values()) if isinstance(inputs,dict) else [row['sha256'] for row in inputs]
        hashes += [row['sha256'] for row in spec.get('input_git_bundles', [])]
        for workspace in spec.get('workspaces', {}).values():
            hashes += [row['sha256'] for row in workspace.get('input_git_bundles', [])]
        if not hashes:return
        response, _=self.remote(worker,'artifacts-missing',dict(hashes=list(dict.fromkeys(hashes))))
        for sha in response['missing']:
            result=self.post('/v1/artifacts/get',dict(host_id=worker['host_id'],hashes=[sha]))
            self.remote(worker,'artifacts-put',dict(blobs=result['blobs']))
            self.log('ARTIFACT_DELIVERED',host_id=worker['host_id'],sha256=sha)

    def tick(self, worker, observed=None):
        info = self.state['workers'].setdefault(worker['host_id'], dict(failure_count=0, next_retry_epoch=0))
        info['phase'] = 'MAILBOX_STATUS'
        if observed is None:
            status_start=stamp()
            status, download_seconds = self.remote(worker, 'status')
            poll = self._poll_value(status)
        else:
            status, download_seconds, status_start, poll, _ = observed
        self.log('MAILBOX_STATUS_RETURN',host_id=worker['host_id'],started=status_start,seconds=download_seconds)
        host = status['host_id']
        if host != worker['host_id']: raise ValueError('Mailbox host differs from relay profile')
        for attempt in status['attempts']:
            state = attempt['state']
            if state == 'TRANSFERRED': continue
            owner = dict(host_id=host, job_id=attempt['job_id'], attempt_id=attempt['attempt_id'])
            info['phase'] = 'CONTROLLER_STATE'
            self.post('/v1/state', dict(owner, state=state, detail=attempt.get('detail', {})))
            if state == 'COMPLETED_PENDING_TRANSFER':
                started = time.monotonic()
                info['phase'] = 'CONTROLLER_ARTIFACTS'
                self._recover_artifacts(worker, attempt)
                info['phase'] = 'CONTROLLER_COMPLETE'
                response = self.post('/v1/complete', dict(owner, request_sha256=attempt['request_sha256'], files=attempt['files']))
                if response.get('ok') is not True: raise ValueError('Controller refused completion')
                upload_seconds = time.monotonic()-started
                info['phase'] = 'MAILBOX_ACK'
                acknowledgement, ack_seconds = self.remote(worker, 'ack', owner)
                if acknowledgement.get('ok') is not True: raise ValueError('Worker refused completion acknowledgement')
                timing = dict(upload_roundtrip_seconds=upload_seconds, ssh_status_download_seconds=download_seconds,
                              ssh_ack_roundtrip_seconds=ack_seconds,
                              artificial_transport_scope='SSH status includes all pending attempts for this host')
                self.log('RESULT_RECOVERED', **owner, duplicate=response.get('duplicate', False), **timing)
                try: self.post('/v1/transfer-metric', dict(owner, timing=timing))
                except Exception as error: self.log('TELEMETRY_PENDING', **owner, error=str(error))
        if (self.directory/'STOP').exists():
            return
        info['phase'] = 'CONTROLLER_POLL'
        # Recovery above always precedes new claims, including while paused.
        response = self.post('/v1/poll', poll)
        if not response['assignments'] and response.get('eligibility_challenges'):
            current, _ = self.remote(worker, 'readiness', dict(challenges=response['eligibility_challenges']))
            if current.get('host_id') != host:
                raise ValueError('Readiness response host differs')
            poll.update(host_fingerprints=current.get('host_fingerprints', {}),
                        readiness_errors=current.get('readiness_errors', {}))
            response = self.post('/v1/poll', poll)
        feedback = response.get('capacity_feedback')
        if feedback is not None and (status.get('resource_report') or {}).get('mode') == 'AUTO':
            try:
                self.remote(worker, 'resources', dict(capacity_feedback=feedback))
            except Exception as error:
                self.log('RESOURCE_FEEDBACK_PENDING', host_id=host, error=str(error))
        assignments = response['assignments']
        response_time=time.monotonic()
        for assignment in assignments:
            # Store delivery intent before SSH. A lost submit response is
            # reconciled with the same server2-owned job and attempt next tick.
            pending = self.directory/'deliveries'/assignment['job_id']
            pending.mkdir(parents=True, exist_ok=True)
            atomic_json(pending/'ASSIGNMENT.json', assignment)
            info['phase'] = 'MAILBOX_ARTIFACTS'
            self._deliver_artifacts(worker, assignment)
            info['phase'] = 'MAILBOX_SUBMIT'
            relay_wait=time.monotonic()-response_time
            self.log('REQUEST_DELIVERY_START',host_id=host,job_id=assignment['job_id'],attempt_id=assignment['attempt_id'])
            response, seconds = self.remote(worker, 'submit', assignment)
            if response.get('ok') is not True: raise ValueError('Worker refused assignment')
            self.log('REQUEST_DELIVERED', host_id=host, job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                     seconds=seconds, duplicate=response.get('duplicate', False))
            try:self.post('/v1/transfer-metric',dict(host_id=host,job_id=assignment['job_id'],attempt_id=assignment['attempt_id'],
                timing=dict(ssh_delivery_roundtrip_seconds=seconds,relay_response_to_submit_seconds=relay_wait)))
            except Exception as error:self.log('TELEMETRY_PENDING',host_id=host,job_id=assignment['job_id'],attempt_id=assignment['attempt_id'],error=str(error))
        info.update(last_success_epoch=time.time(), slots=status['slots'], phase='READY',
            states={name:sum(a['state']==name for a in status['attempts']) for name in sorted({a['state'] for a in status['attempts']})})

    def _collect_ticks(self):
        for host,(worker,future) in list(self.pending_ticks.items()):
            if not future.done(): continue
            del self.pending_ticks[host]
            info=self.state['workers'][host]
            try:future.result()
            except Exception as error:
                self.failed(info,error,'TRANSPORT_PENDING',host_id=host,phase=info.get('phase'))
            else:self.succeeded(info)

    def _state_snapshot(self):
        # Per-host tick tasks mutate only their own existing info dictionary.
        # dict.copy is atomic under Python's GIL; serialize detached snapshots.
        return dict(self.state,workers={host:info.copy() for host,info in list(self.state['workers'].items())})

    def cycle(self):
        """Observe and deliver independent hosts without serial transport waits.

        Each host has at most one discovery or delivery in flight. A blocking
        submit/ACK retains that host's future and cannot strand another host.
        The cycle waits at most one poll interval, not every SSH timeout.
        """
        self._collect_ticks()
        budget=max(0,min(self.poll_seconds,float(self.config.get('discovery_wait_seconds',self.poll_seconds))))
        deadline=time.monotonic()+budget
        if self.ensure_tunnel():
            workers=self.config['workers']
            if self.discovery_pool is None:
                self.discovery_pool=ThreadPoolExecutor(max_workers=max(1,len(workers)),thread_name_prefix='rpe-mailbox')
            for worker in workers:
                host=worker['host_id']
                info=self.state['workers'].setdefault(host,dict(failure_count=0,next_retry_epoch=0))
                if host in self.pending_ticks or host in self.pending_discovery or time.time()<info.get('next_retry_epoch',0):
                    continue
                info['phase']='MAILBOX_DISCOVERY'
                self.pending_discovery[host]=(worker,self.discovery_pool.submit(self._discover,worker))
            if self.pending_discovery:
                wait([future for _,future in self.pending_discovery.values()],timeout=max(0,deadline-time.monotonic()))
            ready=[]
            for host,(worker,future) in list(self.pending_discovery.items()):
                if not future.done():continue
                del self.pending_discovery[host]
                info=self.state['workers'][host]
                try:ready.append((worker,future.result()))
                except Exception as error:
                    self.failed(info,error,'TRANSPORT_PENDING',host_id=host,phase=info.get('phase'))
            # All ready hosts have published metadata-only offers already. The
            # central matcher enforces per-REQUEST priority and available slots.
            ready.sort(key=lambda row:row[0].get('priority_group',1 if row[0].get('transport')=='wsl' else 0))
            for worker,observed in ready:
                if (self.directory/'STOP').exists():break
                host=worker['host_id']
                self.pending_ticks[host]=(worker,self.discovery_pool.submit(self.tick,worker,observed))
            if self.pending_ticks:
                wait([future for _,future in self.pending_ticks.values()],timeout=max(0,deadline-time.monotonic()))
            self._collect_ticks()
        self.state['heartbeat_epoch']=time.time()
        atomic_json(self.directory/'STATUS.json',self._state_snapshot())

    def run(self):
        lock = (self.directory/'RELAY.lock').open('a')
        try: acquire_lock(lock)
        except BlockingIOError: raise ValueError('This relay is already running')
        self.log('RELAY_START', pid=os.getpid())
        # Reuse RomuTail's startup-state fix (4058b48d5): a previous daemon's
        # stopped=True must not satisfy this new daemon's status/stop checks.
        atomic_json(self.directory/'STATUS.json', self._state_snapshot())
        try:
            while not (self.directory/'STOP').exists():
                self.cycle()
                time.sleep(self.poll_seconds)
        finally:
            if self.tunnel is not None and self.tunnel.poll() is None:
                self.tunnel.terminate(); self.tunnel.wait(timeout=10)
            self.state.update(stopped_epoch=time.time(), stopped=True)
            atomic_json(self.directory/'STATUS.json', self._state_snapshot())
            self.log('RELAY_STOP')
            if self.discovery_pool is not None:
                self.discovery_pool.shutdown(wait=False, cancel_futures=True)
            lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument('--start', action='store_true')
    command.add_argument('--run', action='store_true', help='Run in foreground')
    command.add_argument('--status', action='store_true')
    command.add_argument('--stop', action='store_true', help='Stop only relay/tunnel; received workers continue')
    args = parser.parse_args(argv)
    relay = Relay(args.config)
    if args.status:
        path = relay.directory/'STATUS.json'
        result = json.loads(path.read_text()) if path.exists() else dict(status='NOT_STARTED')
        if 'heartbeat_epoch' in result: result['heartbeat_age_seconds'] = time.time()-result['heartbeat_epoch']
        print(json.dumps(result, indent=2)); return
    if args.stop:
        (relay.directory/'STOP').touch()
        print('Stop requested; received Q3 jobs keep running. Check --status for stopped_epoch.'); return
    if args.start:
        lock = (relay.directory/'RELAY.lock').open('a')
        try: acquire_lock(lock)
        except BlockingIOError: raise ValueError('This relay is already running')
        (relay.directory/'STOP').unlink(missing_ok=True)
        lock.close()
        with (relay.directory/'RELAY.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--config', str(relay.configuration), '--run'],
                **background_options(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(pid=process.pid, log=str(relay.directory/'RELAY.log')))); return
    relay.run()


if __name__ == '__main__': main()
