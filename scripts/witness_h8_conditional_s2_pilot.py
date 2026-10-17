"""Native parent witness: exclusive lock, wait4, owned-only cleanup."""
import argparse
import ctypes
import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_conditional_s2_contract as c
import h8_key_compile_parallel as pool
from run_h8_conditional_s2_pilot import fresh, observe


def cleanup(handle):
    errors = []
    for sig, seconds in ((signal.SIGTERM, 10.), (signal.SIGKILL, 3.)):
        try:
            pool.wait_owned(handle)
            if not handle['reaped']:
                pool.send_owned(handle, sig)
        except BaseException as exc:
            errors.append(str(exc))
        until = time.monotonic()+seconds
        while not handle['reaped'] and time.monotonic() < until:
            pool.wait_owned(handle)
            time.sleep(.02)
    if not handle['reaped']:
        pid, status, usage = os.wait4(handle['pid'], 0)
        handle.update(reaped=True, exit_code=os.waitstatus_to_exitcode(status),
            cpu_seconds=usage.ru_utime+usage.ru_stime, peak_RSS_bytes=usage.ru_maxrss*1024)
    return errors


def launch(config_path, grant_path):
    config, grant = c.read(config_path), c.read(grant_path)
    checks = fresh(config, grant)  # No scientific imports before a valid grant.
    lock, output, control = [Path(config[k]) for k in ('lock', 'output', 'control')]
    c.save(lock, dict(SOURCE=config['SOURCE'], config_sha256=c.digest(config), grant_sha256=c.digest(grant),
                      witness=pool.identity(os.getpid()), one_shot_consumed=True))
    output.mkdir()
    control.mkdir()
    c.save(control/'fresh_gate.json', checks)
    c.save(control/'witness_identity.json', dict(identity=pool.identity(os.getpid())))
    env = dict(os.environ, **{k:'1' for k in c.THREADS}, QISKIT_PARALLEL='FALSE',
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    scratch = control/'scratch'
    scratch.mkdir()
    for key, name in (('TMPDIR', 'tmp'), ('MPLCONFIGDIR', 'mpl'), ('XDG_CACHE_HOME', 'cache'), ('NUMBA_CACHE_DIR', 'numba')):
        p = scratch/name
        p.mkdir()
        env[key] = str(p)
    parent = os.getpid()
    argv = [config['python'], '-I', '-P', '-B', str(c.ROOT/'scripts/run_h8_conditional_s2_pilot.py'),
            '--driver', '--config', str(config_path), '--grant', str(grant_path)]
    first = None
    handle = None
    def interrupted(sig, frame):
        raise RuntimeError('WITNESS_INTERRUPTED:'+str(sig))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            pid = os.fork()
            if pid == 0:
                try:
                    libc = ctypes.CDLL(None, use_errno=True)
                    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0 or os.getppid() != parent:
                        os._exit(125)
                    for stream, fd in (('stdout', 1), ('stderr', 2)):
                        with (control/(stream+'.log')).open('xb', buffering=0) as f:
                            os.dup2(f.fileno(), fd)
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                    os.execve(config['python'], argv, env)
                except BaseException:
                    os._exit(127)
            handle = dict(pid=pid, identity=pool.identity(pid), reaped=False)
            c.require(handle['identity'] is not None and handle['identity'][2:] == (parent, os.getuid()), 'OWN_DRIVER_IDENTITY')
            c.save(control/'driver_identity.json', handle)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        last = 0.
        while not handle['reaped']:
            pool.wait_owned(handle)
            if time.monotonic()-last > 30:
                c.save(control/f'observation_{time.time_ns()}.json', observe())
                last = time.monotonic()
            if not handle['reaped']:
                time.sleep(.2)
    except BaseException as exc:
        first = str(exc) or type(exc).__name__
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        errors = cleanup(handle) if handle and not handle['reaped'] else []
        if handle is not None:
            c.save(control/'native_exit.json', dict(driver=handle, cleanup_errors=errors, first_STOP=first,
                all_direct_children_reaped=handle['reaped'], automatic_retry=False, SOURCE=config['SOURCE'],
                config_sha256=c.digest(config), grant_sha256=c.digest(grant)))
        terminal = c.read(output/'terminal.json') if (output/'terminal.json').exists() else None
        pools = [c.read(p) for p in output.glob('pilot_q*/parallel_receipt.json')]
        workers = [h for p in pools for h in p['workers']]
        clean = all(h['reaped'] and pool.identity(h['pid']) != tuple(h['identity']) for h in workers)
        complete = bool(handle and handle.get('exit_code') == 0 and handle['reaped'] and not errors and not first
            and terminal and terminal['status'] == 'COMPLETE_MANDATORY_STOP' and len(workers) == 16 and clean)
        c.save(control/'final_gate.json', dict(status='PASS_MANDATORY_STOP' if complete else 'STOP',
            driver_reaped=bool(handle and handle['reaped']), all_workers_reaped_and_absent=clean,
            worker_count=len(workers), first_STOP=first or (terminal or {}).get('first_STOP'),
            mandatory_STOP=True, next_stage_authorized=False, scientific_retry=False,
            SOURCE=config['SOURCE'], config_sha256=c.digest(config), grant_sha256=c.digest(grant)))
    return 0 if complete else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--grant', required=True)
    a = p.parse_args()
    raise SystemExit(launch(Path(a.config).resolve(), Path(a.grant).resolve()))
