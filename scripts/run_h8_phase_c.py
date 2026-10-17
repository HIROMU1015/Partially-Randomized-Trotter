"""Plan-only default. Future explicitly approved saved-value C, then STOP.

The launcher keeps the arithmetic driver as its direct child, waits with wait4 and
records native exit, rusage, PSI (observation only) and OOM counters. No retry.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_phase_c_runtime as r
c = r.c


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, allow_nan=False, indent=2); stream.write('\n')


def driver(config_path, grant_path, fresh_path):
    cfg, grant, fresh = map(r.read, (config_path, grant_path, fresh_path))
    cpus = r.authorize(cfg, grant); r.fresh_binding(cfg, grant, fresh)
    c.require(r.identity(os.getppid()) == fresh['launcher_identity'], 'LIVE_OWN_LAUNCHER_REQUIRED')
    c.require(Path(sys.executable).resolve() == Path(cfg['python']).resolve() and
              all(os.environ.get(k) == '1' for k in r.THREAD_KEYS), 'PINNED_PYTHON_THREAD_ONE')
    r.verify(cfg, git=False, protection=False)  # No subprocesses inside the arithmetic role.
    out, control, lock = r.path_set(cfg, supervisor_control_exists=True)
    with lock.open('x') as stream:
        stream.write(c.digest(grant)); stream.flush(); os.fsync(stream.fileno())
    out.mkdir(mode=0o700)
    calls = {k: 0 for k in r.CAPS}
    interrupted = []
    def guard():
        c.require(not interrupted, 'C_OWN_PROCESS_SIGNAL:' + (str(interrupted[0]) if interrupted else ''))
    def take(key):
        guard(); c.require(calls[key] < r.CAPS[key], 'C_CALL_LIMIT:' + key); calls[key] += 1
    def stop(signum, frame):
        interrupted.append(signum)
        raise RuntimeError('C_OWN_PROCESS_SIGNAL:' + str(signum))
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for s in old:
        signal.signal(s, stop)
    try:
        data, registry = r.load_saved(cfg)
        with (out/'accounting_rows.jsonl').open('x') as rows:
            def emit(kind, value):
                if kind == 'accounting_row':
                    rows.write(json.dumps(value, allow_nan=False)+'\n')
            result = c.evaluate(data, registry, take=take, emit=emit, guard=guard)
        c.require(calls['saved_row'] == 1900 and calls['comparison'] == 20 and calls['reaccounting_pass'] == 1,
                  'EXACT_C_ACCOUNTING_COUNTS')
        guard(); r.verify(cfg, git=False, protection=False)
        save(out/'comparisons.json', result)
        save(out/'terminal.json', dict(status='COMPLETE_MANDATORY_STOP', SOURCE=cfg['SOURCE'], cpus=cpus,
             model_sha256=c.MODEL_SHA, B2_result_commit=c.B2_RESULT, actual_calls=calls, new_compile=0, new_fit=0,
             noncancellation_shots_used=False, next_stage_authorized=False))
    except BaseException as error:
        save(out/'terminal.json', dict(status='STOP', SOURCE=cfg['SOURCE'], first_stop=str(error),
             actual_calls=calls, automatic_retry=False, next_stage_authorized=False))
        raise
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


def observe():
    """PSI and OOM counters of the host and this cgroup chain; recorded, not a STOP gate."""
    group = Path('/sys/fs/cgroup')/next(line.split(':', 2)[2].lstrip('/') for line in
        Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    pressure, events = {'host': Path('/proc/pressure/memory').read_text()}, {}
    while True:
        if (group/'memory.pressure').exists():
            pressure[str(group)] = (group/'memory.pressure').read_text()
        if (group/'memory.events').exists():
            events[str(group)] = dict(x.split() for x in (group/'memory.events').read_text().splitlines())
        if group == Path('/sys/fs/cgroup'):
            break
        group = group.parent
    memory = {l.split(':')[0]: int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()
              if l.split(':')[0] in ('MemAvailable', 'MemFree')}
    return dict(pressure=pressure, events=events, memory=memory)


B2_SCRIPTS = (b'run_h8_phase_b2', b'launch_h8_phase_b2', b'h8_phase_b2_')


def is_B2_script(argv):
    """True only for an argv entry that is a B2 Python script, not text inside a shell command."""
    return any(x.endswith(b'.py') and x.rsplit(b'/', 1)[-1].startswith(B2_SCRIPTS) for x in argv)


def protected_job_cpus():
    """CPUs used by running B2 Python scripts of this user (read-only)."""
    cpus = set()
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if p.stat().st_uid == os.getuid() and is_B2_script((p/'cmdline').read_bytes().split(b'\0')):
                    cpus |= os.sched_getaffinity(int(p.name))
            except (OSError, ProcessLookupError):
                continue
    return cpus


def fresh(config, grant):
    cpus = r.authorize(config, grant)
    c.require(not set(cpus) & protected_job_cpus(), 'C_CPUS_OVERLAP_RUNNING_PROTECTED_JOB')
    state = observe()
    policy = r.RESOURCE_POLICY['admission']
    c.require(state['memory']['MemAvailable'] >= policy['minimum_memory_available_bytes'], 'C_MEMORY_ADMISSION')
    parent = Path(config['output']).parent
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    mount = max((x for x in mounts if parent.is_relative_to(x[1])), key=lambda x: len(x[1]))
    c.require(mount[2] in ('ext4', 'xfs') and not any('quota' in x and not x.startswith('no') or
              x in ('uquota', 'gquota', 'pquota') for x in mount[3].split(',')), 'QUOTA_UNKNOWN_OR_ENABLED')
    fs = os.statvfs(parent)
    c.require(fs.f_bavail*fs.f_frsize >= policy['minimum_free_disk_bytes'] and
              fs.f_favail >= policy['minimum_free_inodes'], 'C_DISK_INODE_HEADROOM')
    return dict(SOURCE=config['SOURCE'], config_sha256=c.digest(config), grant_sha256=c.digest(grant),
        allowed_cpus=cpus, monotonic_seconds=time.monotonic(), launcher_identity=r.identity(os.getpid()),
        observation=state, source_input_verified=True, CPU_selection_verified=True,
        resources_admission_verified=True, quota_verified=True, protected_jobs_unchanged=True, launcher_wait_ready=True)


def launch(config_path, grant_path):
    cfg, grant = map(r.read, (config_path, grant_path))
    cpus = r.authorize(cfg, grant)  # Reject every invalid draft before any side effect/import/CPU change.
    r.path_set(cfg); r.verify(cfg)
    marker = Path(cfg['output']).parent/'C_LAUNCH_ATTEMPT'
    save(marker, dict(SOURCE=cfg['SOURCE'], scope=r.SCOPE, automatic_retry=False))
    try:
        value = fresh(cfg, grant)
        fpath = marker.parent/'C_fresh_gate.json'; save(fpath, value)
        Path(cfg['control']).mkdir(mode=0o700)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(Path(cfg['control'])))
        for key in r.THREAD_KEYS:
            env[key] = '1'
        command = [cfg['python'], '-P', '-B', str(Path(__file__).resolve()), '--driver-approved',
                   '--config', str(config_path), '--grant', str(grant_path), '--fresh-gate', str(fpath)]
        before, samples, t0 = observe(), 0, time.monotonic()
        with (Path(cfg['control'])/'stdout.log').open('xb') as out, (Path(cfg['control'])/'stderr.log').open('xb') as err:
            child = subprocess.Popen(command, env=env, stdout=out, stderr=err,
                                     preexec_fn=lambda: os.sched_setaffinity(0, set(cpus)))
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid == child.pid:
                    break
                samples += 1; time.sleep(.05)
        after = observe()
        code = os.waitstatus_to_exitcode(status)
        save(Path(cfg['control'])/'native_exit.json', dict(pid=child.pid, native_exit_code=code, reaped=True,
             wait_kind='os.wait4_direct_child', wall_seconds=time.monotonic()-t0, user_CPU_seconds=usage.ru_utime,
             system_CPU_seconds=usage.ru_stime, peak_RSS_bytes=usage.ru_maxrss*1024, observation_samples=samples,
             observation_before=before, observation_after=after, PSI_is_STOP_gate=False, automatic_retry=False))
        c.require(code == 0, 'C_DRIVER_NATIVE_EXIT:' + str(code))
        r.verify(cfg)
        c.require(r.read(Path(cfg['output'])/'terminal.json')['status'] == 'COMPLETE_MANDATORY_STOP', 'C_DRIVER_NOT_COMPLETE')
        save(marker.parent/'C_final_gate.json', dict(status='PASS_MANDATORY_STOP', SOURCE=cfg['SOURCE'], next_stage_authorized=False))
    except BaseException as error:
        save(marker.parent/'C_phase_STOP.json', dict(status='STOP', SOURCE=cfg['SOURCE'],
            first_stop=str(error), automatic_retry=False, next_stage_authorized=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch-approved', action='store_true'); mode.add_argument('--driver-approved', action='store_true')
    for key in ('config', 'grant', 'fresh-gate'): parser.add_argument('--'+key, type=Path)
    args = parser.parse_args()
    if not args.launch_approved and not args.driver_approved:
        if any((args.config, args.grant, args.fresh_gate)): parser.error('real paths require explicitly approved mode')
        print(json.dumps(r.plan(), indent=2)); return
    if not args.config or not args.grant: parser.error('sealed config and grant required')
    if args.launch_approved: launch(args.config, args.grant)
    else:
        if not args.fresh_gate: parser.error('fresh gate required')
        driver(args.config, args.grant, args.fresh_gate)


if __name__ == '__main__':
    main()
