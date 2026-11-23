#!/usr/bin/env python3
"""Keep a participating WSL mailbox alive across individual SSH calls."""

import argparse
import base64
import json
from pathlib import Path
import subprocess


HOLD = '''
import fcntl
import json
from pathlib import Path
import time

profile = json.loads(Path(PROFILE).read_bytes())
if profile['host_id'] != HOST:
    raise ValueError('WSL worker profile identity differs')
work = Path(profile['work_dir'])
flag = Path(PARTICIPATION)

def accepting():
    participation = json.loads(flag.read_bytes()) if flag.is_file() else {}
    return participation.get('host_id') == HOST and participation.get('accepting_new_jobs') is True

def active_attempt():
    for path in work.glob('*/*/WORKER_PROCESS.lock'):
        try:
            with path.open('rb') as lock:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except FileNotFoundError:
            continue
    return False

if not work.is_dir() or not (accepting() or active_attempt()):
    raise SystemExit(0)
with (work / '.wsl-foreground.lock').open('a') as lease:
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(0)
    paused_since = None
    print('WSL_WORKER_FOREGROUND_HELD', flush=True)
    while True:
        if accepting():
            paused_since = None
        elif paused_since is None:
            paused_since = time.monotonic()
        elif time.monotonic() - paused_since >= 30 and not active_attempt():
            break
        time.sleep(5)
'''


def powershell_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def command_for(worker):
    if worker['transport'] != 'ssh-wsl' or not worker.get('participation_wsl_file'):
        raise ValueError('A Windows SSH mailbox with a WSL participation file is required')
    declarations = dict(HOST=worker['host_id'], PROFILE=worker['profile'],
                        PARTICIPATION=worker['participation_wsl_file'])
    source = ''.join(f'{key} = {value!r}\n' for key, value in declarations.items()) + HOLD
    payload = base64.b64encode(source.encode()).decode('ascii')
    snippet = f"exec(__import__('base64').b64decode('{payload}'))"
    executable = powershell_literal(worker.get('wsl_executable', 'wsl.exe'))
    distro = powershell_literal(worker['distribution'])
    script = (
        f"$running=& {executable} --list --running --quiet | ForEach-Object {{($_ -replace [string][char]0,'').Trim()}};"
        f"if($running -notcontains {distro}){{exit 0}};"
        f"& {executable} -d {distro} --exec /usr/bin/python3 -B -c {powershell_literal(snippet)};"
        "exit $LASTEXITCODE;"
    )
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    return [*worker.get('ssh_command', ['ssh']), '-n', '-T', '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=10',
            '-o', 'ServerAliveCountMax=2', worker['ssh_alias'],
            'powershell.exe -NoProfile -NonInteractive -EncodedCommand '+encoded]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    worker = next((row for row in config['workers'] if row['host_id'] == args.host), None)
    if worker is None or worker.get('enabled') is False:
        parser.error('Enabled WSL worker not found in relay configuration')
    raise SystemExit(subprocess.run(command_for(worker)).returncode)


if __name__ == '__main__':
    main()
