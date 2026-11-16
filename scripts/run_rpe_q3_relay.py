#!/usr/bin/env python3
"""Run the local0 qg2-controller/qg1-mailbox relay without shared storage.

Only the server2 controller selects candidates and reserves/adopts evaluations.
Stopping this relay leaves all received server1 jobs running independently.
"""
import argparse
import fcntl
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


class Relay:
    def __init__(self, configuration):
        self.configuration = Path(configuration).resolve()
        self.config = json.loads(self.configuration.read_text())
        self.directory = Path(self.config['state_directory']).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ssh = self.config.get('ssh_command', ['ssh'])
        self.controller = self.config['controller']
        self.url = 'http://127.0.0.1:'+str(self.controller.get('local_port', 18763))
        self.tunnel = None
        self.state = dict(pid=os.getpid(), configuration=str(self.configuration), workers={},
                          started_epoch=time.time(), transport='local0 SSH qg2 forwarding + qg1 mailbox')

    def log(self, event, **value):
        row = dict(epoch=time.time(), event=event, **value)
        with (self.directory/'EVENTS.jsonl').open('a') as output: output.write(json.dumps(row)+'\n')

    def post(self, path, value):
        payload = json.dumps(value).encode()
        request = urllib.request.Request(self.url+path, payload, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.config.get('request_timeout_seconds', 60)) as response:
            return json.load(response)

    def remote(self, worker, action, value=None):
        alias = worker['ssh_alias']
        if not alias or alias.startswith('-'): raise ValueError('Invalid existing SSH alias')
        command = [worker.get('python', 'python3'), '-B', worker['root'].rstrip('/')+'/scripts/run_rpe_q3_worker.py',
                   '--profile', worker['profile'], '--mailbox', action]
        started = time.monotonic()
        process = subprocess.run(self.ssh+['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', alias, shlex.join(command)],
            input=json.dumps(value or {}), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=self.config.get('request_timeout_seconds', 60))
        if process.returncode:
            raise RuntimeError('SSH mailbox '+action+' failed ('+str(process.returncode)+'): '+process.stderr[-1500:])
        result = json.loads(process.stdout)
        return result, time.monotonic()-started

    def ensure_tunnel(self):
        if self.config.get('external_tunnel'): return
        if self.tunnel is not None and self.tunnel.poll() is None: return
        alias = self.controller.get('ssh_alias', 'qg2')
        if not alias or alias.startswith('-'): raise ValueError('Invalid controller SSH alias')
        local_port = int(self.controller.get('local_port', 18763))
        remote_port = int(self.controller.get('remote_port', 8763))
        with (self.directory/'TUNNEL.log').open('a') as log:
            self.tunnel = subprocess.Popen(self.ssh+['-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                '-L', '127.0.0.1:%d:127.0.0.1:%d' % (local_port, remote_port), alias],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self.log('TUNNEL_START', pid=self.tunnel.pid, ssh_alias=alias)

    def tick(self, worker):
        status, download_seconds = self.remote(worker, 'status')
        host = status['host_id']
        if host != worker['host_id']: raise ValueError('Mailbox host differs from relay profile')
        for attempt in status['attempts']:
            state = attempt['state']
            if state == 'TRANSFERRED': continue
            owner = dict(host_id=host, job_id=attempt['job_id'], attempt_id=attempt['attempt_id'])
            self.post('/v1/state', dict(owner, state=state, detail=attempt.get('detail', {})))
            if state == 'COMPLETED_PENDING_TRANSFER':
                started = time.monotonic()
                response = self.post('/v1/complete', dict(owner, request_sha256=attempt['request_sha256'], files=attempt['files']))
                if response.get('ok') is not True: raise ValueError('Controller refused completion')
                upload_seconds = time.monotonic()-started
                self.remote(worker, 'ack', owner)
                timing = dict(upload_roundtrip_seconds=upload_seconds, ssh_status_download_seconds=download_seconds,
                              artificial_transport_scope='SSH status includes all pending attempts for this host')
                self.post('/v1/transfer-metric', dict(owner, timing=timing))
                self.log('RESULT_RECOVERED', **owner, duplicate=response.get('duplicate', False), **timing)
        assignments = self.post('/v1/poll', dict(host_id=host, slots=status['slots'],
            source_sha=status['source_sha'], cached_prefixes=status['cached_prefixes']))['assignments']
        for assignment in assignments:
            # Store delivery intent before SSH. A lost submit response is
            # reconciled with the same server2-owned job and attempt next tick.
            pending = self.directory/'deliveries'/assignment['job_id']
            pending.mkdir(parents=True, exist_ok=True)
            atomic_json(pending/'ASSIGNMENT.json', assignment)
            response, seconds = self.remote(worker, 'submit', assignment)
            if response.get('ok') is not True: raise ValueError('Worker refused assignment')
            self.log('REQUEST_DELIVERED', host_id=host, job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                     seconds=seconds, duplicate=response.get('duplicate', False))
        self.state['workers'][host] = dict(last_success_epoch=time.time(), slots=status['slots'],
            states={name:sum(a['state']==name for a in status['attempts']) for name in sorted({a['state'] for a in status['attempts']})})

    def run(self):
        lock = (self.directory/'RELAY.lock').open('a')
        try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('This relay is already running')
        self.log('RELAY_START', pid=os.getpid())
        try:
            while not (self.directory/'STOP').exists():
                self.ensure_tunnel()
                for worker in self.config['workers']:
                    if (self.directory/'STOP').exists(): break
                    try: self.tick(worker)
                    except Exception as error:
                        self.state['workers'][worker['host_id']] = dict(error=str(error), last_error_epoch=time.time())
                        self.log('TRANSPORT_PENDING', host_id=worker['host_id'], error=str(error))
                self.state['heartbeat_epoch'] = time.time()
                atomic_json(self.directory/'STATUS.json', self.state)
                time.sleep(self.config.get('poll_seconds', 1))
        finally:
            if self.tunnel is not None and self.tunnel.poll() is None:
                self.tunnel.terminate(); self.tunnel.wait(timeout=10)
            self.state.update(stopped_epoch=time.time(), stopped=True)
            atomic_json(self.directory/'STATUS.json', self.state)
            self.log('RELAY_STOP')
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
        try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('This relay is already running')
        (relay.directory/'STOP').unlink(missing_ok=True)
        lock.close()
        with (relay.directory/'RELAY.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--config', str(relay.configuration), '--run'],
                start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(pid=process.pid, log=str(relay.directory/'RELAY.log')))); return
    relay.run()


if __name__ == '__main__': main()
