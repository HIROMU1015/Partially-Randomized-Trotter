"""One owned numerical child, hard wall fallback and sequential wait recorder."""
import errno
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

spec=importlib.util.spec_from_file_location('h6_owned_runner',Path(__file__).with_name('h6_b0i_pilot_runner_v1.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
g=r.g


def terminate_owned(pid, expected, sig):
    current=g.identity(pid)
    if current is None: return
    if current != expected: raise g.Stop('OWN_CHILD_IDENTITY_CHANGED')
    try: fd=os.pidfd_open(pid)
    except OSError as error:
        if error.errno==errno.ESRCH:return
        raise
    try:
        current=g.identity(pid)
        if current is None:return
        if current != expected:raise g.Stop('OWN_CHILD_IDENTITY_CHANGED')
        try:signal.pidfd_send_signal(fd,sig)
        except OSError as error:
            if error.errno!=errno.ESRCH:raise
    finally:os.close(fd)


def launch(contract_path, auth_path, mapping_path, output):
    def stopped(sig, frame):raise g.Stop('LAUNCHER_SIGNAL:'+str(sig))
    handlers={sig:signal.signal(sig,stopped) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:return _launch(contract_path,auth_path,mapping_path,output)
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def _launch(contract_path, auth_path, mapping_path, output):
    raw=g.read_bytes(contract_path);araw=g.read_bytes(auth_path)
    c=g.d0.strict_json(raw);a=g.d0.strict_json(araw)
    if c.get('synthetic_only'):raise g.Stop('REAL_LAUNCHER_REJECTS_ARTIFICIAL')
    grant=g.authorization_gate(c,a,contract_sha=g.sha(raw),authorization_sha=g.sha(araw),
        output=output,scope=r.SCOPE)
    r.validate_contract(grant);g.runtime_gate(grant,c['runtime_root'])
    control=g.home_path(c['control_directory'])
    if not control.is_dir() or control.stat().st_uid!=os.getuid():raise g.Stop('OWN_CONTROL_DIRECTORY_REQUIRED')
    budget=c['launcher_budget']
    hard=resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard!=resource.RLIM_INFINITY and hard<c['budget']['address_space_bytes']:
        raise g.Stop('INHERITED_AS_HARD_LIMIT_TOO_SMALL')
    cpu_hard=resource.getrlimit(resource.RLIMIT_CPU)[1]
    if cpu_hard!=resource.RLIM_INFINITY and cpu_hard<c['budget']['cpu_seconds']+2:
        raise g.Stop('INHERITED_CPU_HARD_LIMIT_TOO_SMALL')
    # A soft limit on this metadata process; retain a hard limit that lets the
    # numerical child install its explicitly authorized 32GiB limit itself.
    resource.setrlimit(resource.RLIMIT_AS,(budget['address_space_bytes'],hard))
    command=[sys.executable,'-P','-B',str(Path(__file__).with_name('h6_b0i_pilot_runner_v1.py')),
        '--contract',str(contract_path),'--authorization',str(auth_path),'--inputs',str(mapping_path),'--output',str(output)]
    started=time.monotonic();reason=None
    log_path=control/'launch_stdout.log'
    with log_path.open('xb') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        identity=None
        try:
            identity=g.identity(process.pid)
            if identity is None:
                status=process.wait(timeout=5)
            else:
                while True:
                    try:status=process.wait(timeout=1);break
                    except subprocess.TimeoutExpired:pass
                    if time.monotonic()-started>c['budget']['wall_seconds']:reason='OWN_WORKER_WALL_CAP'
                    if log_path.stat().st_size>c['log_cap_bytes']:reason=reason or 'OWN_LOG_CAP'
                    usage=resource.getrusage(resource.RUSAGE_SELF)
                    if usage.ru_maxrss*1024>budget['rss_bytes']:reason=reason or 'LAUNCHER_RSS_CAP'
                    if usage.ru_utime+usage.ru_stime>budget['cpu_seconds']:reason=reason or 'LAUNCHER_CPU_CAP'
                    if reason:
                        terminate_owned(process.pid,identity,signal.SIGTERM)
                        try:status=process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            terminate_owned(process.pid,identity,signal.SIGKILL);status=process.wait(timeout=5)
                        break
        except BaseException as first_error:
            try:
                if identity is None:
                    try:identity=g.identity(process.pid)
                    except Exception:identity=None
                if identity is not None:
                    terminate_owned(process.pid,identity,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        terminate_owned(process.pid,identity,signal.SIGKILL);process.wait(timeout=5)
                else:process.wait(timeout=5)
            except Exception as cleanup_error:
                raise g.Stop('LAUNCHER_FIRST:'+str(first_error)+';CLEANUP_UNCONFIRMED:'+str(cleanup_error)) from first_error
            raise
    shell_status=128-status if status<0 else status
    # This second invocation shares precisely the same parent as the worker.
    recorded=subprocess.run(command+['--record-wait-status',str(shell_status)],check=False,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    receipt=dict(worker_wait_status=shell_status,recorder_status=recorded.returncode,
        original_worker_identity=identity,worker_identity_absent=g.identity(process.pid) is None,
        first_launcher_stop=reason,wall_seconds=time.monotonic()-started,
        automatic_retry=False,next_stage_authorized=False,exact_exit_timestamp=None)
    fd=os.open(control/'launcher_wait_v1.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:os.write(fd,json.dumps(receipt).encode()+b'\n');os.fsync(fd)
    finally:os.close(fd)
    return shell_status if reason is None and recorded.returncode==0 else 2


if __name__=='__main__':
    try:
        if len(sys.argv)!=5:raise g.Stop('usage: CONTRACT AUTH INPUT_MAPPING OUTPUT')
        sys.exit(launch(*(Path(x) for x in sys.argv[1:])))
    except Exception as error:
        print('H6_LAUNCH_STOP:'+type(error).__name__+':'+str(error)[:1000],file=sys.stderr);sys.exit(2)
