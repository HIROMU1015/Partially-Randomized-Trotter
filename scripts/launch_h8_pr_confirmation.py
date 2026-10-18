"""Native wait4 witness for exactly two independent H8 confirmation scopes.

No grant creation or retry. All launch paths are private metadata arguments.
A failed scope does not stop its independent sibling. Interrupted witnesses
terminate and reap only their newly created, identity-checked child sessions.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_pr_confirmation as h
c = h.c
INTERRUPTED = None


def check_interruption():
    if INTERRUPTED is not None:
        raise KeyboardInterrupt('WITNESS_SIGNAL_' + str(INTERRUPTED))


def identity(pid):
    proc = Path('/proc') / str(pid)
    tail = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttime=int(tail[19]), uid=proc.stat().st_uid, pgid=os.getpgid(pid))


def reap(process, *, block=False):
    pid, status, usage = os.wait4(process.pid, 0 if block else os.WNOHANG)
    if not pid:
        return None
    code = os.waitstatus_to_exitcode(status)
    process.returncode = code
    return dict(pid=pid, exit_code=code, reaped=True, CPU_seconds=usage.ru_utime + usage.ru_stime,
                peak_RSS_bytes=usage.ru_maxrss * 1024)


def stop(process, owned):
    if process.returncode is not None:
        return None
    result = reap(process)
    if result:
        return result
    for sig, wait in ((signal.SIGTERM, 20.), (signal.SIGKILL, 2.)):
        try:
            c.require(identity(process.pid) == owned and owned['uid'] == os.getuid()
                      and owned['pgid'] == process.pid, 'OWN_CHILD_SESSION_ONLY')
            os.killpg(process.pid, sig)
        except (FileNotFoundError, ProcessLookupError):
            return reap(process, block=True)
        until = time.monotonic() + wait
        while time.monotonic() < until:
            result = reap(process)
            if result:
                return result
            time.sleep(.05)
    return reap(process, block=True)


def launch(manifest):
    directory = c.home_path(manifest['execution_directory'])
    c.require(set(manifest['specs']) == set(manifest['grants']) == set(h.SCOPES), 'FIXED_TWO_SCOPES')
    # Validate the full batch before launching either scientific driver.
    total_memory = 0
    cpus = set()
    for scope in h.SCOPES:
        spec, grant = c.read(manifest['specs'][scope]), c.read(manifest['grants'][scope])
        h.authorize(spec, grant)
        h.verify(spec)
        h.fresh_gate(spec, grant)
        assigned = set(grant['allowed_cpus']) | {grant['driver_cpu']}
        c.require(cpus.isdisjoint(assigned), 'INDEPENDENT_SCOPE_CPU_COLLISION')
        cpus.update(assigned)
        total_memory += grant['memory_admission_estimate_bytes']
    fresh = h.effective_fresh(directory)
    c.require(fresh['memory_available_bytes'] >= total_memory
              and fresh['disk_free_bytes'] >= 2 * 512 * 2**20
              and fresh['free_inodes'] >= 20000, 'FRESH_AGGREGATE_ADMISSION')
    check_interruption()
    c.claim_attempt(directory / 'BATCH_ONE_SHOT.json', manifest['SOURCE'], 'H8_CONFIRM_TWO')
    processes, results, logs = {}, {}, []
    try:
        for scope in h.SCOPES:
            check_interruption()
            spec = c.read(manifest['specs'][scope])
            log = (directory / (scope + '.log')).open('x')
            logs.append(log)
            command = [spec['python'], '-P', '-B', str(h.ROOT / 'scripts/h8_pr_confirmation.py'),
                       '--spec', manifest['specs'][scope], '--grant', manifest['grants'][scope]]
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            # Track immediately, including a possible exit/identity-observation race.
            processes[scope] = (child, None)
            try:
                owned = identity(child.pid)
            except (FileNotFoundError, ProcessLookupError):
                results[scope] = reap(child, block=True)
                continue
            processes[scope] = (child, owned)
        c.save(directory / 'batch_started.json', dict(SOURCE=manifest['SOURCE'], witness_pid=os.getpid(),
            children={scope: owned for scope, (_, owned) in processes.items()}, fresh=fresh,
            auto_retry=False, next_stage_authorized=False))
        while len(results) < len(processes):
            check_interruption()
            for scope, (child, _) in processes.items():
                if scope not in results:
                    result = reap(child)
                    if result:
                        results[scope] = result
                        c.save(directory / (scope + '_native_completion.json'), result)
            if len(results) < len(processes):
                time.sleep(.5)
    except BaseException:
        # Signal handlers only request interruption; cleanup itself is uninterrupted.
        errors = []
        for scope, (child, owned) in processes.items():
            try:
                if scope not in results:
                    if owned is None:
                        # Only this child can be waited here; never signal an unbound PID.
                        results[scope] = reap(child, block=True)
                    else:
                        results[scope] = stop(child, owned)
            except BaseException as error:
                errors.append(dict(scope=scope, error=str(error)))
        c.save(directory / 'witness_interruption_cleanup.json', dict(results=results, errors=errors))
        raise
    finally:
        for log in logs:
            log.close()
    c.save(directory / 'batch_native_completion.json', dict(SOURCE=manifest['SOURCE'], results=results,
        mandatory_STOP=True, automatic_retry=False, next_stage_authorized=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    def interrupted(signum, frame):
        global INTERRUPTED
        INTERRUPTED = signum
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    launch(c.read(args.manifest))
