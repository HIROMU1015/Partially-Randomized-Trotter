"""Plan-only default. Future explicitly approved saved-value C, then STOP."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
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
    import track_a_phase_a_parallel as owned
    c.require(list(owned.identity(os.getppid()) or ()) == fresh['observer_identity'], 'LIVE_OWN_OBSERVER_REQUIRED')
    c.require(Path(sys.executable).resolve() == Path(cfg['python']).resolve() and
              all(os.environ.get(k) == '1' for k in r.THREAD_KEYS), 'PINNED_PYTHON_THREAD_ONE')
    r.verify(cfg, git=False, protection=False)  # No subprocesses inside the arithmetic role.
    r.fresh_binding(cfg, grant, r.read(fresh_path))
    out, control, lock = r.path_set(cfg, supervisor_control_exists=True)
    with lock.open('x') as stream:
        stream.write(c.digest(grant)); stream.flush(); os.fsync(stream.fileno())
    out.mkdir(mode=0o700)
    os.sched_setaffinity(0, {cpus[0]})
    limits = r.RESOURCES
    resource.setrlimit(resource.RLIMIT_AS, (limits['AS_bytes'], limits['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU, (limits['CPU_seconds'], limits['CPU_seconds']))
    started = time.monotonic(); calls = {k: 0 for k in r.CAPS}
    def guard():
        c.require(time.monotonic()-started <= limits['wall_seconds'], 'C_WALL_LIMIT')
        c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= limits['RSS_bytes'], 'C_RSS_LIMIT')
        c.require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= limits['output_bytes'], 'C_OUTPUT_LIMIT')
    def take(key):
        guard(); c.require(calls[key] < r.CAPS[key], 'C_CALL_LIMIT:' + key); calls[key] += 1
    def stop(*args):
        raise RuntimeError('C_STOP_SIGNAL_OR_DEADLINE')
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    for s in old:
        signal.signal(s, stop)
    signal.alarm(limits['wall_seconds'])
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
        save(out/'terminal.json', dict(status='COMPLETE_MANDATORY_STOP', SOURCE=cfg['SOURCE'],
             model_sha256=c.MODEL_SHA, actual_calls=calls, new_compile=0, new_fit=0, next_stage_authorized=False))
    except BaseException as error:
        save(out/'terminal.json', dict(status='STOP', SOURCE=cfg['SOURCE'], first_stop=str(error),
             actual_calls=calls, automatic_retry=False, next_stage_authorized=False))
        raise
    finally:
        signal.alarm(0)
        for s, handler in old.items():
            signal.signal(s, handler)


def resource_probe(observer):
    value = observer.snapshot()
    c.require(value['events'] and 'host' in value['pressure'], 'MISSING_PRESSURE_OOM')
    for fields in value['events'].values():
        c.require({'oom', 'oom_kill'} <= fields.keys(), 'MISSING_OOM_COUNTERS')
    for raw in value['pressure'].values():
        full = [s for s in raw.splitlines() if s.startswith('full ')]
        c.require(len(full) == 1, 'MISSING_FULL_PSI')
        c.finite(float(dict(p.split('=') for p in full[0].split()[1:])['avg10']), 'FULL_PSI')
    return value


def fresh(config, grant, observer):
    cpus = r.authorize(config, grant)
    sibling_set = set(); cores = set()
    for cpu in cpus:
        d = Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology'
        cores.add(((d/'physical_package_id').read_text(), (d/'core_id').read_text()))
        for span in (d/'thread_siblings_list').read_text().strip().split(','):
            ends = list(map(int, span.split('-'))); sibling_set.update(range(ends[0], ends[-1]+1))
    c.require(len(cores) == 2, 'TWO_PHYSICAL_CORES')
    group = Path('/sys/fs/cgroup')/next(line.split(':', 2)[2].lstrip('/') for line in
        Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    while True:
        if (group/'cpu.max').exists():
            q, period = (group/'cpu.max').read_text().split()
            c.require(q == 'max' or int(q) >= 2*int(period), 'CPU_QUOTA_BELOW_TWO')
        else:
            c.require(group == Path('/sys/fs/cgroup'), 'CPU_QUOTA_UNKNOWN')
        if group == Path('/sys/fs/cgroup'): break
        group = group.parent
    def ticks():
        return {int(s[0][3:]): list(map(int, s[1:9])) for line in Path('/proc/stat').read_text().splitlines()
                if (s := line.split()) and s[0].startswith('cpu') and s[0][3:].isdigit()}
    a = ticks(); time.sleep(1); b = ticks()
    for cpu in sibling_set:
        delta = [y-x for x, y in zip(a[cpu], b[cpu])]; total = sum(delta)
        c.require(total > 0 and min(delta) >= 0 and 1-(delta[3]+delta[4])/total <= .05, 'PHYSICAL_CORE_BUSY')
    state = resource_probe(observer)
    c.require(observer.resource_reason(state, state, r.RESOURCES['aggregate_RSS_bytes']+
              r.RESOURCES['admission_reserve_bytes']) is None, 'C_MEMORY_PRESSURE_OOM')
    parent = Path(config['output']).parent
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    mount = max((x for x in mounts if parent.is_relative_to(x[1])), key=lambda x: len(x[1]))
    c.require(mount[2] in ('ext4', 'xfs') and not any('quota' in x and not x.startswith('no') or
              x in ('uquota', 'gquota', 'pquota') for x in mount[3].split(',')), 'QUOTA_UNKNOWN_OR_ENABLED')
    fs = os.statvfs(parent)
    c.require(fs.f_bavail*fs.f_frsize >= 2*2**30 and fs.f_favail >= 40000, 'C_DISK_INODE_HEADROOM')
    return dict(SOURCE=config['SOURCE'], config_sha256=c.digest(config), grant_sha256=c.digest(grant),
        allowed_cpus=cpus, monotonic_seconds=time.monotonic(), observer_identity=list(observer.owned.identity(os.getpid())),
        source_input_verified=True, physical_cores_verified=True, resources_idle_verified=True,
        quota_verified=True, protected_jobs_unchanged=True, external_monitor_ready=True)


def launch(config_path, grant_path):
    cfg, grant = map(r.read, (config_path, grant_path))
    cpus = r.authorize(cfg, grant)  # Reject every invalid draft before any side effect/import/CPU change.
    r.path_set(cfg); r.verify(cfg)
    marker = Path(cfg['output']).parent/'C_LAUNCH_ATTEMPT'
    save(marker, dict(SOURCE=cfg['SOURCE'], scope=r.SCOPE, automatic_retry=False))
    previous = os.sched_getaffinity(0)
    try:
        import launch_h8_strengthened_phase_a as observer
        fpath = marker.parent/'C_fresh_gate.json'
        value = fresh(cfg, grant, observer); save(fpath, value)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        for key in r.THREAD_KEYS: env[key] = '1'
        env['TMPDIR'] = str(marker.parent/'tmp')
        Path(env['TMPDIR']).mkdir(mode=0o700, exist_ok=False)
        os.sched_setaffinity(0, {cpus[1]})
        command = [cfg['python'], '-P', '-B', str(Path(__file__).resolve()), '--driver-approved',
                   '--config', str(config_path), '--grant', str(grant_path), '--fresh-gate', str(fpath)]
        observer.supervise(command, env=env, control=cfg['control'], output=cfg['output'],
            limits=r.RESOURCES, allowed_cpus=cpus, probe=lambda: resource_probe(observer))
        r.verify(cfg)
        c.require(r.read(Path(cfg['control'])/'launcher_terminal.json')['status'] == 'COMPLETE_MANDATORY_STOP',
                  'C_SUPERVISOR_NOT_COMPLETE')
        save(marker.parent/'C_final_gate.json', dict(status='PASS_MANDATORY_STOP', SOURCE=cfg['SOURCE'], next_stage_authorized=False))
    except BaseException as error:
        save(marker.parent/'C_phase_STOP.json', dict(status='STOP', SOURCE=cfg['SOURCE'],
            first_stop=str(error), automatic_retry=False, next_stage_authorized=False))
        raise
    finally:
        os.sched_setaffinity(0, previous)


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
