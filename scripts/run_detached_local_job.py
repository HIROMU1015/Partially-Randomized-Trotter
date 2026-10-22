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
import signal
import subprocess
import sys
import time


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    # Field 22 is the kernel start time; compare it as well as the PID.
    try:
        raw = Path('/proc', str(pid), 'stat').read_text()
        return dict(pid=pid, start_ticks=raw.rsplit(')', 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def heartbeat(root, value):
    temporary = root/'heartbeat.partial'
    write(temporary, value)
    temporary.replace(root/'heartbeat.json')


def service_args(root, request_sha):
    request = json.loads((root/'request.json').read_text())
    return ['systemd-run', '--user', '--collect', '--service-type=exec',
        '--unit='+request['unit'], '--property=WorkingDirectory='+request['cwd'],
        '--property=Restart=no', '--property=KillMode=control-group',
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
    # Exclusive entry precedes every child launch. A second worker invocation
    # must not modify a previous exit, STOP or heartbeat.
    write(root/'worker_entered.json', dict(pid=os.getpid(), started_at_utc=now(),
        process_identity=process_identity(os.getpid()), automatic_retry=False))
    child = None
    received_signal = None
    def interrupted(signum, frame):
        nonlocal received_signal
        received_signal = signum
        raise InterruptedError('SUPERVISOR_SIGNAL:'+str(signum))
    handlers = {sig:signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        raw = (root/'request.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != request_sha:
            raise ValueError('Request changed')
        request = json.loads(raw)
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != request['source_sha256']:
            raise ValueError('Launcher source changed')
        os.sched_setaffinity(0, request['CPU_affinity'])
        write(root/'resource_inheritance.json', dict(CPU_affinity=sorted(os.sched_getaffinity(0)),
            env=request['env'], arbitrary_CPU_RAM_worker_caps=None))
        env = dict(os.environ, **request['env'])
        with (root/'stdout.log').open('x') as stdout, (root/'stderr.log').open('x') as stderr:
            child = subprocess.Popen(request['command'], cwd=request['cwd'], env=env,
                stdout=stdout, stderr=stderr, shell=False, start_new_session=True)
            write(root/'child_started.json', dict(pid=child.pid, started_at_utc=now(),
                process_identity=process_identity(child.pid)))
            while child.poll() is None:
                heartbeat(root, dict(time_utc=now(), child_pid=child.pid, running=True))
                time.sleep(.25)
            code = child.wait()
        heartbeat(root, dict(time_utc=now(), child_pid=child.pid, running=False, returncode=code))
        receipt = dict(returncode=code, child_reaped=True, ended_at_utc=now(),
            status='COMMAND_EXIT0' if code == 0 else 'COMMAND_FAILED_NO_RETRY',
            scientific_stage_success_must_be_checked_separately=True, automatic_retry=False)
        write(root/'exit_receipt.json', receipt)
        if code:
            write(root/'STOP.json', dict(receipt, reason='CHILD_NONZERO_EXIT'))
        return code
    except BaseException as exc:
        # Stop only the new session created by this supervisor. Never search
        # for or signal another job by command name.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        code = None if child is None else child.wait()
        heartbeat(root, dict(time_utc=now(), running=False, supervisor_failed=True,
            child_pid=None if child is None else child.pid, returncode=code))
        write(root/'STOP.json', dict(status='SUPERVISOR_FAILED_NO_RETRY', reason=str(exc),
            signal=received_signal, child_launched=child is not None,
            child_reaped=child is not None, child_returncode=code,
            scientific_stage_success=False, automatic_retry=False, observed_at_utc=now()))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def status(root):
    root = Path(root).resolve()
    for name in ('STOP.json', 'exit_receipt.json'):
        if (root/name).exists():
            return dict(record=name, **json.loads((root/name).read_text()))
    if (root/'worker_entered.json').exists():
        entry = json.loads((root/'worker_entered.json').read_text())
        identity = entry.get('process_identity')
        alive = identity is not None and process_identity(entry['pid']) == identity
        return dict(status='SUPERVISOR_PRESENT' if alive else 'PROCESS_ABSENT_NO_SUCCESS_RECEIPT',
            scientific_stage_success=False, exact_failure_time=None, automatic_retry=False)
    return dict(status='WORKER_NOT_ENTERED', scientific_stage_success=False, automatic_retry=False)


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
    status_parser = modes.add_parser('status')
    status_parser.add_argument('--root', required=True)
    a = p.parse_args()
    if a.mode == 'worker':
        sys.exit(worker(a.root, a.request_sha))
    if a.mode == 'status':
        print(json.dumps(status(a.root)))
        sys.exit(0)
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    launch(a.root, a.cwd, command)
