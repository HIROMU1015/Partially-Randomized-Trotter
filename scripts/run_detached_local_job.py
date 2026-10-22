"""One-shot user transient service, independent of the originating chat process.

This launcher confers no scientific authorization. Commands still enforce their
own sealed grants and budgets. No settings or services of other users are edited.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


def service_args(root, request_sha):
    request = json.loads((root/'request.json').read_text())
    return ['systemd-run', '--user', '--collect', '--service-type=exec',
        '--unit='+request['unit'], '--property=WorkingDirectory='+request['cwd'],
        '--property=StandardOutput=append:'+str(root/'supervisor_stdout.log'),
        '--property=StandardError=append:'+str(root/'supervisor_stderr.log'),
        '--', sys.executable, '-B', str(Path(__file__).resolve()),
        'worker', '--root', str(root), '--request-sha', request_sha]


def launch(root, cwd, command):
    root, cwd = Path(root).resolve(), Path(cwd).resolve()
    if not command or not Path(command[0]).is_absolute() or not cwd.is_dir():
        raise ValueError('Absolute executable and existing working directory required')
    root.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:24]
    request = dict(unit='prt-local-'+digest, command=command, cwd=str(cwd), created_at_utc=now(),
        env={k:os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS') if k in os.environ},
        CPU_affinity=sorted(os.sched_getaffinity(0)), automatic_retry=False,
        grants_must_be_validated_by_command=True, scientific_authorization_added=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(root/'request.json', request)
    sha = hashlib.sha256((root/'request.json').read_bytes()).hexdigest()
    result = subprocess.run(service_args(root, sha), text=True, capture_output=True, shell=False)
    write(root/'launch_receipt.json', dict(returncode=result.returncode, stdout=result.stdout,
        stderr=result.stderr, unit=request['unit'], request_sha256=sha, automatic_retry=False))
    if result.returncode:
        raise RuntimeError('USER_SERVICE_NOT_STARTED; do not reuse this launch root')
    print(json.dumps(dict(root=str(root), unit=request['unit'], status='STARTED_NOT_COMPLETED')))


def worker(root, request_sha):
    root = Path(root).resolve()
    raw = (root/'request.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != request_sha:
        raise ValueError('Request changed')
    request = json.loads(raw)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != request['source_sha256']:
        raise ValueError('Launcher source changed')
    # Keep the original allowed CPU set; scientific worker allocation remains
    # the responsibility of the command's fresh resource checks.
    os.sched_setaffinity(0, request['CPU_affinity'])
    write(root/'worker_entered.json', dict(pid=os.getpid(), started_at_utc=now(),
        CPU_affinity=sorted(os.sched_getaffinity(0)), automatic_retry=False))
    env = dict(os.environ, **request['env'])
    with (root/'stdout.log').open('x') as stdout, (root/'stderr.log').open('x') as stderr:
        child = subprocess.Popen(request['command'], cwd=request['cwd'], env=env,
            stdout=stdout, stderr=stderr, shell=False)
        write(root/'child_started.json', dict(pid=child.pid, started_at_utc=now()))
        while child.poll() is None:
            temporary = root/'heartbeat.partial'
            write(temporary, dict(time_utc=now(), child_pid=child.pid, running=True))
            temporary.replace(root/'heartbeat.json')
            time.sleep(.25)
        code = child.wait()
    write(root/'exit_receipt.json', dict(returncode=code, child_reaped=True, ended_at_utc=now(),
        status='COMMAND_EXIT0' if code == 0 else 'COMMAND_FAILED_NO_RETRY',
        scientific_stage_success_must_be_checked_separately=True, automatic_retry=False))
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    modes = p.add_subparsers(dest='mode', required=True)
    launch_parser = modes.add_parser('launch')
    launch_parser.add_argument('--root', required=True)
    launch_parser.add_argument('--cwd', required=True)
    launch_parser.add_argument('command', nargs=argparse.REMAINDER)
    worker_parser = modes.add_parser('worker')
    worker_parser.add_argument('--root', required=True)
    worker_parser.add_argument('--request-sha', required=True)
    a = p.parse_args()
    if a.mode == 'worker':
        sys.exit(worker(a.root, a.request_sha))
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    launch(a.root, a.cwd, command)
