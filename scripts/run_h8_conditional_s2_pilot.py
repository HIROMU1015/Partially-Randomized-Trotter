"""One-shot driver; A completes before B. No implicit science retry."""
import argparse
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_conditional_s2_contract as c
import h8_key_compile_parallel as pool


def observe():
    memory = {x.split(':')[0]: int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
              if x.startswith(('MemAvailable:', 'MemFree:'))}
    group_name = next(x.split(':', 2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    root = Path('/sys/fs/cgroup')
    group = root/group_name.lstrip('/')
    groups = []
    effective = memory['MemAvailable']
    while group.is_relative_to(root):
        row = dict(group=str(group))
        for k in ('memory.events', 'memory.max', 'memory.current', 'memory.pressure'):
            if (group/k).exists():
                row[k] = (group/k).read_text()
        if row.get('memory.max', 'max').strip() != 'max':
            effective = min(effective, max(0, int(row['memory.max'])-int(row['memory.current'])))
        groups.append(row)
        if group == root:
            break
        group = group.parent
    return dict(monotonic=time.monotonic(), memory=memory, effective_available_bytes=effective,
        host_PSI=Path('/proc/pressure/memory').read_text(), cgroups=groups,
        own_affinity=sorted(os.sched_getaffinity(0)), PSI_stop=False)


def fresh(config, grant):
    c.authorize(config, grant)
    c.verify(config, runtime=True)
    for key in ('output', 'control', 'lock'):
        p = Path(config[key])
        c.require(p.is_absolute() and p.resolve() == p and p.is_relative_to('/home/AbeHiromu')
                  and not p.exists() and p.parent.is_dir(), 'EXCLUSIVE_HOME_OUTPUT_CONTROL_LOCK')
    c.require(len({config[k] for k in ('output', 'control', 'lock')}) == 3, 'DISJOINT_NAMESPACES')
    cpus = grant['worker_cpus']
    cores = []
    for cpu in cpus:
        p = Path('/sys/devices/system/cpu')/f'cpu{cpu}/topology'
        cores.append(((p/'physical_package_id').read_text().strip(), (p/'core_id').read_text().strip()))
    c.require(len(set(cores)) == len(cores), 'DISTINCT_PHYSICAL_CORES')
    pinned = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc/'cmdline').read_bytes().split(b'\0')
            if not argv or b'python' not in Path(os.fsdecode(argv[0])).name.lower().encode():
                continue
            aff = os.sched_getaffinity(int(proc.name))
            if len(aff) <= 16 and any(b'/home/AbeHiromu/projects/' in x and x.endswith(b'.py') for x in argv):
                pinned.append(dict(identity=pool.identity(int(proc.name)), cpus=sorted(aff)))
        except (OSError, ValueError):
            continue
    c.require(not any(set(cpus)&set(r['cpus']) for r in pinned), 'OWN_PINNED_JOB_OVERLAP')
    state = observe()
    policy = c.plan()['resources']
    c.require(state['effective_available_bytes'] >= len(cpus)*policy['peak_RSS_estimate_per_worker']+policy['headroom_estimate'],
              'INSUFFICIENT_EFFECTIVE_HEADROOM')
    fs = os.statvfs(Path(config['output']).parent)
    c.require(fs.f_bavail*fs.f_frsize >= policy['output_estimate_bytes'] and fs.f_favail >= policy['inodes_estimate'], 'BLOCKS_INODES')
    mount = max((line.split() for line in Path('/proc/mounts').read_text().splitlines()
                 if Path(config['output']).is_relative_to(line.split()[1])), key=lambda row: len(row[1]))
    c.require(mount[2] in ('ext4', 'xfs') and not any(('quota' in opt and not opt.startswith('no')) or
              opt in ('uquota', 'gquota', 'pquota') for opt in mount[3].split(',')), 'QUOTA_NEEDS_PROOF')
    probe = Path(config['output']).parent/('allocation_probe_'+uuid.uuid4().hex)
    try:
        with probe.open('xb') as f:
            os.posix_fallocate(f.fileno(), 0, policy['output_estimate_bytes'])
            os.fsync(f.fileno())
            allocated = probe.stat().st_blocks*512
            c.require(allocated >= policy['output_estimate_bytes'], 'REAL_ALLOCATION_PROBE')
    finally:
        if probe.exists():
            probe.unlink()
    return dict(observation=state, physical_cores=cores, own_pinned_jobs=pinned, filesystem=mount,
                free_bytes=fs.f_bavail*fs.f_frsize, free_inodes=fs.f_favail,
                quota_mount_options_present=False, numeric_quota='UNKNOWN_NOT_CLAIMED_UNLIMITED',
                own_allocation_probe_bytes=allocated, allocation_probe_removed=True,
                source_input_runtime=True, config_sha256=c.digest(config), grant_sha256=c.digest(grant))


def driver(config, grant):
    def interrupted(sig, frame):
        raise RuntimeError('DRIVER_STOP_SIGNAL:'+str(sig))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    c.authorize(config, grant)
    c.verify(config, git=False)
    c.require(Path(sys.executable).resolve() == Path(config['python']).resolve() and
        all(os.environ.get(k) == '1' for k in c.THREADS) and os.environ.get('QISKIT_PARALLEL') == 'FALSE', 'PINNED_PYTHON_THREADS')
    output = Path(config['output'])
    ledger = c.Ledger(output)
    accepted = []
    last = [0.]
    def guard():
        now = time.monotonic()
        if now-last[0] >= 30:
            c.verify(config, git=False)
            c.save(output/f'observation_{int(now*1000)}.json', observe())
            last[0] = now
    def callback(task, directory):
        # Worker affinity is one selected CPU; authorize against the inherited
        # grant in the parent, then use an exact checked task capability here.
        from h8_conditional_s2_adapter import evaluate
        return evaluate(task, directory, config, grant)
    try:
        for q in (4, 8):
            receipt = pool.map_owned(c.tasks(q), callback, cpus=grant['worker_cpus'],
                limits=dict(max_workers=len(grant['worker_cpus'])), directory=output/f'pilot_q{q}',
                deadline=math.inf, guard=guard, reserve=ledger.reserve, font_seed=config['font_seed'])
            c.require(receipt['status'] == 'PASS' and receipt['all_reaped'] and not receipt['cleanup_errors'], 'NATIVE_CLEANUP')
            values = []
            for task, worker in zip(c.tasks(q), receipt['workers']):
                p = Path(worker['directory'])/'task_result.json'
                c.require(worker['task_id'] == task['id'] and worker['exit_code'] == 0 and worker['reaped']
                    and c.sha(p) == worker['result_sha256'], 'NATIVE_EXIT_RESULT_BYTES')
                value = c.read(p)
                c.paired_result(task, value)
                value.update(native_exit=0, reaped=True, identity=worker['identity'],
                    worker_peak_RSS_bytes=worker['peak_RSS_bytes'], native_CPU_seconds=worker['cpu_seconds'],
                    result_sha256=worker['result_sha256'])
                values.append(value)
            c.save(output/f'pilot_q{q}_accepted.json', dict(q=q, results=values, summary=c.summary(values),
                all_reaped=True, SOURCE=config['SOURCE'], original_n32_mixed=False))
            accepted.extend(values)
        c.verify(config, git=False)
        c.require(ledger.used == c.CAPS and len(accepted) == 16, 'EXACT_FINAL_PILOT_COUNTS')
        c.save(output/'terminal.json', dict(status='COMPLETE_MANDATORY_STOP', SOURCE=config['SOURCE'],
            new_calls=ledger.used, prior=c.budget(), scientific_attempts=1, retry=False,
            next_stage_authorized=False, all_workers_reaped=True, first_STOP=None))
    except BaseException as error:
        c.save(output/'terminal.json', dict(status='STOP', first_STOP=str(error) or type(error).__name__,
            SOURCE=config['SOURCE'], nonrefundable_reserved=ledger.used, accepted_trajectories=len(accepted),
            prior=c.budget(), scientific_attempts=1, retry=False, next_stage_authorized=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--grant', required=True)
    p.add_argument('--driver', action='store_true')
    a = p.parse_args()
    config, grant = c.read(a.config), c.read(a.grant)
    if a.driver:
        driver(config, grant)
    else:
        c.authorize(config, grant)
        print('Authorization accepted; invoke native witness for one-shot execution.')
