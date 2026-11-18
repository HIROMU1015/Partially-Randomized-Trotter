#!/usr/bin/env python3
"""Reclaim an existing run's acknowledged worker copies without changing its release."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from run_rpe_q3_relay import Relay, acquire_lock
from trottertracks.rpe_worker_storage import report


REMOTE_REPORT = """
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1])/'src'))
from trottertracks import rpe_worker_storage as storage
payload = json.load(sys.stdin)
exec(compile(payload['source'], 'rpe_worker_storage.py', 'exec'), vars(storage))
profile = json.loads(Path(sys.argv[2]).read_bytes()) if sys.argv[2] else payload['profile']
spools = (profile['work_dir'],) if sys.argv[2] else payload['spools']
print(json.dumps(storage.report(profile.get('storage_limit'), spools=spools, pbs_runs=payload.get('pbs_runs'))))
"""


def read(path):
    return json.loads(path.read_bytes())


def wsl_running(worker, ssh):
    name = worker['distribution'].replace("'", "''")
    executable = worker.get('wsl_executable', 'wsl.exe').replace("'", "''")
    command = (f"$running = & '{executable}' --list --running --quiet | "
               "ForEach-Object { ($_ -replace [string][char]0, '').Trim() }; "
               f"if ($running -contains '{name}') {{ 'running' }}")
    encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
    raw = subprocess.check_output(
        [*worker.get('ssh_command', ssh), '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
         worker['ssh_alias'], 'powershell.exe -NoProfile -NonInteractive -EncodedCommand '+encoded],
        stderr=subprocess.PIPE, timeout=15)
    return raw.replace(b'\x00', b'').strip() == b'running'


def cleanup(run, source):
    distributed = run/'distributed'
    conf = distributed/'relay'/'CONFIG.json'
    relay = Relay(conf) if conf.exists() else None
    local = distributed/'LOCAL_WORKER.json'
    pbs = distributed/'pbs'/'SETTINGS.json'
    work = []
    if local.exists():
        profile = read(local)
        work.append(('local', lambda p=profile: report(p.get('storage_limit'), spools=(p['work_dir'],))))
    if relay is not None:
        for worker in relay.config['workers']:
            host = worker['host_id']
            if not worker.get('enabled', True):
                continue
            def remote(w=worker):
                if w.get('transport') == 'ssh-wsl' and not wsl_running(w, relay.ssh):
                    return dict(status='WSL_STOPPED')
                command = [w['python'], '-B', '-c', REMOTE_REPORT, w['root'], w['profile']]
                return relay.remote_command(w, command, 'storage-reclaim', dict(source=source))[0]
            work.append((host, remote))
    if pbs.exists():
        settings = read(pbs)
        namespace = Path(settings['remote_root'])
        runs = str(namespace.parent) if settings.get('namespace_by_run', True) else None
        for worker in settings['workers']:
            def gateway(w=worker):
                command = [settings['gateway_python'], '-B', '-c', REMOTE_REPORT,
                           settings['transport_root'], '']
                payload = dict(source=source, profile=w['profile'],
                               spools=[str(namespace/'workers'/w['host_id'])], pbs_runs=runs)
                output = subprocess.check_output(
                    ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                     settings['ssh_alias'], shlex.join(command)],
                    input=json.dumps(payload), text=True, stderr=subprocess.PIPE, timeout=45)
                return json.loads(output)
            work.append((worker['host_id'], gateway))
    with ThreadPoolExecutor(max_workers=max(1, len(work))) as pool:
        for host, future in [(host, pool.submit(check)) for host, check in work]:
            try:
                result = future.result()
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                result = dict(error_type=type(error).__name__)
            print(json.dumps(dict(epoch=time.time(), host_id=host, storage=result)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    if not (run/'CONTROLLER_CONFIG.json').is_file():
        parser.error('Expected an existing central run directory')
    source = (Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_worker_storage.py').read_text()
    if not args.watch:
        cleanup(run, source)
        return
    with closing((run/'distributed'/'STORAGE_RECLAIM.lock').open('a+b')) as lock:
        acquire_lock(lock)
        while not (run/'EXIT.json').exists():
            cleanup(run, source)
            time.sleep(60)


if __name__ == '__main__':
    main()
