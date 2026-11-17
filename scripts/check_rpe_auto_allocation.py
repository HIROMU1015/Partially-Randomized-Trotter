"""Ordinary-entry acceptance with real, labelled artificial Q3 subprocesses.

All host names are simulated on one OS using loopback HTTP and separate spools.
The normal search, candidate ordering, durable queue, adapter validation and
resume paths run unchanged. No H4 inputs, scientific DP or circuits are used.
"""
import argparse
from collections import Counter
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import check_rpe_distributed_q3 as F
from trottertracks.rpe_q3_coordinator import atomic_json
from trottertracks.rpe_q3_store import JobStore
from trottertracks.rpe_q3_timing import read_lines

HOSTS = ('server2', 'server1', 'local', 'local0-wsl')


def admission(enabled):
    return dict(schema='RPE_Q3_HOST_ADMISSION_V1',
                hosts={'local0-wsl': dict(optional=True, accepting=enabled)})


def prepare(directory, *, reference=False, join_late=False):
    directory = Path(directory).resolve()
    port = F.free_port()
    F.prepare(directory, port=port, total_slots=1, local_slots=1,
              sleep_seconds=0 if reference else .35)
    spec = json.loads((directory / 'ROUND.json').read_text())
    spec['round_id'] = 'ARTIFICIAL_AUTO_ALLOCATION_ENTRY'
    if reference:
        spec['executor'].pop('distributed')
        spec['executor']['scheduler'] = 'legacy'
    else:
        spec['executor'].pop('max_concurrent_jobs')
        distributed = spec['executor']['distributed']
        distributed.update(host_admission=str(directory / 'HOST_ADMISSION.json'),
            affinity_wait_seconds=0, auto_allocation=dict(discovery_grace_seconds=.1,
            offer_ttl_seconds=1, priority_groups=[list(HOSTS[:3]), [HOSTS[3]]]))
        # The first host starts at one checked slot. The controller's total
        # limit is omitted, so the public CLI must resolve it automatically.
        distributed['local_worker']['slot_policy'] = dict(mode='auto', initial_slots=1, max_slots=2)
        atomic_json(directory / 'HOST_ADMISSION.json', admission(not join_late))
    F.write(directory / 'ROUND.json', spec)
    for host in HOSTS:
        value = F.profile(directory, host)
        value['dp_workers'] = 1
        value['slot_policy'] = dict(mode='auto', initial_slots=1, max_slots=2)
        F.write(directory / ('PROFILE_' + host + '.json'), value)
    return port


def peak_artificial_overlap(run):
    """Reuse recorded boot-domain body boundaries; do not create new timers."""
    points = []
    intervals = []
    for path in Path(run).glob('jobs/*/*/JOB_STAGES.jsonl'):
        rows = read_lines(path)
        starts = [r for r in rows if r.get('stage') == 'ARTIFICIAL_EVALUATION_START']
        ends = [r for r in rows if r.get('stage') == 'ARTIFICIAL_EVALUATION_RETURNED']
        if len(starts) != 1 or len(ends) != 1:
            raise AssertionError('Artificial body must execute exactly once: ' + str(path))
        first, last = starts[0], ends[0]
        if first.get('clock_id') != last.get('clock_id'):
            raise AssertionError('Artificial body clocks differ')
        domain = first['clock_id']
        points.extend([(domain, first['monotonic'], 1), (domain, last['monotonic'], -1)])
        launch = json.loads((path.parent / 'LAUNCH.json').read_text())
        intervals.append(dict(job_id=path.parent.name, host_id=launch.get('host_id'), clock_id=domain,
                              start=first['monotonic'], end=last['monotonic']))
    peak = 0
    for domain in {p[0] for p in points}:
        active = 0
        for _, _, change in sorted((p for p in points if p[0] == domain), key=lambda p: (p[1], p[2])):
            active += change
            peak = max(peak, active)
    wsl_overlap = any(a['host_id'] == 'local0-wsl' and b['host_id'] in HOSTS[:3]
        and a['clock_id'] == b['clock_id'] and min(a['end'], b['end']) > max(a['start'], b['start'])
        for a in intervals for b in intervals)
    return dict(peak_concurrent_artificial_bodies=peak, wsl_overlaps_preferred_artificial_body=wsl_overlap, body_intervals=intervals,
                scope='ACTUAL_ARTIFICIAL_ADAPTER_SUBPROCESSES_ON_ONE_OS')


def _controller(directory):
    # Reuse the established artificial metadata/acquisition leaves. The public
    # schedule entry, scientific-job validation and candidate rule are unchanged.
    return F.controller(directory)


def run_one(directory, *, reference=False, join_late=False):
    directory = Path(directory).resolve()
    port = prepare(directory, reference=reference, join_late=join_late)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--controller', str(directory)]
    children, logs, workers = [], [], {}
    actions = []
    started = time.monotonic()

    def start_worker(host):
        log = (directory / (host + '.log')).open('ab');logs.append(log)
        worker = subprocess.Popen([sys.executable, '-B', str(ROOT / 'scripts/run_rpe_q3_worker.py'),
            '--profile', str(directory / ('PROFILE_' + host + '.json')),
            '--controller', 'http://127.0.0.1:' + str(port), '--poll-seconds', '.02'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        children.append(worker);workers[host] = worker
        actions.append(dict(action='START_ARTIFICIAL_WORKER', host=host, since_start=time.monotonic()-started))

    try:
        if not reference:
            # WSL may contact first. Preferred-host discovery must not turn
            # arrival order into permanent assignment priority.
            if not join_late:start_worker('local0-wsl')
            for host in ('server1', 'local'):start_worker(host)
        with (directory / 'controller.log').open('ab') as log:
            controller = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            children.append(controller)
            joined = not join_late
            paused_at = None
            resumed_after_pause = False
            while controller.poll() is None:
                if time.monotonic()-started > 100:
                    raise AssertionError('Artificial controller made no bounded completion; see controller.log')
                if join_late and not joined:
                    launches = read_lines(directory / 'run' / 'PROGRESS.jsonl')
                    if len([e for e in launches if e.get('event') == 'LAUNCH']) >= 2:
                        if any(e.get('host_id') == 'local0-wsl' for e in launches if e.get('event') == 'LAUNCH'):
                            raise AssertionError('Unconnected paused WSL received work')
                        atomic_json(directory / 'HOST_ADMISSION.json', admission(True))
                        start_worker('local0-wsl');joined = True
                        actions.append(dict(action='JOIN_AFTER_EXISTING_HOSTS_PROGRESS',
                                            since_start=time.monotonic()-started))
                if not reference and not join_late:
                    launches = read_lines(directory / 'run' / 'PROGRESS.jsonl')
                    if paused_at is None and any(e.get('event') == 'LAUNCH' and e.get('host_id') == 'local0-wsl' for e in launches):
                        atomic_json(directory / 'HOST_ADMISSION.json', admission(False))
                        paused_at = time.monotonic()
                        actions.append(dict(action='PAUSE_WITH_OWNED_ARTIFICIAL_ATTEMPT',
                                            since_start=paused_at-started))
                    elif paused_at is not None and not resumed_after_pause and time.monotonic()-paused_at > .15:
                        atomic_json(directory / 'HOST_ADMISSION.json', admission(True))
                        resumed_after_pause = True
                        actions.append(dict(action='RESUME_DRAINED_OR_RUNNING_ARTIFICIAL_ATTEMPT',
                                            since_start=time.monotonic()-started))
                time.sleep(.02)
            if controller.returncode:
                raise AssertionError('Artificial controller failed: ' + str(directory / 'controller.log'))
        summary = F.check(directory)
        before = F.launch_markers(directory)
        original_outcomes = F.outcomes(directory)
        with (directory / 'resume.log').open('ab') as log:
            resumed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        if resumed.returncode:
            raise AssertionError('Artificial resume failed: ' + str(directory / 'resume.log'))
        if before != F.launch_markers(directory) or original_outcomes != F.outcomes(directory):
            raise AssertionError('Completed job changed or relaunched on ordinary-entry resume')
        summary.update(actual_OS_host_count=1, host_names_are_simulated=True,
            transport='LOOPBACK_HTTP_WITH_SEPARATE_DURABLE_WORKER_SPOOLS',
            full_wall_seconds=time.monotonic()-started, resume_new_launches=0,
            actions=actions, artificial_overlap=peak_artificial_overlap(directory / 'run'))
        if not reference:
            store = JobStore(directory / 'run/distributed/QUEUE.sqlite3')
            events = store.events()
            for kind in ('ASSIGN', 'ACCEPT', 'ADOPT'):
                counts = Counter(e['job_id'] for e in events if e.get('event') == kind)
                if any(n != 1 for n in counts.values()):
                    raise AssertionError('Duplicate ' + kind + ' in artificial durable queue')
            summary.update(duplicate_assignments=0, duplicate_completions=0, duplicate_adoptions=0)
            setting = directory / 'run/distributed/ALLOCATION_SETTINGS.json'
            if not setting.exists():raise AssertionError('Normal entry did not preserve allocation settings')
            summary['allocation_settings'] = json.loads(setting.read_text())
            summary['worker_slot_states'] = {}
            summary['worker_slot_decisions'] = {}
            for host in HOSTS:
                folder = directory / 'workers' / host
                summary['worker_slot_states'][host] = {p.name: json.loads(p.read_text()) for p in
                    list(folder.glob('SLOT_*.json')) + list(folder.glob('*RESOURCE*.json'))}
                summary['worker_slot_decisions'][host] = [r for r in read_lines(folder / 'WORKER_EVENTS.jsonl')
                    if r.get('event') == 'SLOT_CAPACITY_DECISION']
            atomic_json(directory / 'HOST_ADMISSION.json', admission(False))
            summary['final_wsl_accepting'] = False
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/summarize_rpe_q3_timing.py'),
                str(directory / 'run'), '--output', str(directory / 'TIMING.json')],
                cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            timing = json.loads((directory / 'TIMING.json').read_text())
            if timing['missing_counts']['Q3_body_seconds']:
                raise AssertionError('New artificial adapter body timing is missing')
            summary['timing_artifact'] = str(directory / 'TIMING.json')
            summary['timing_body_rows'] = len(timing['rows'])
            summary['timing_missing_counts'] = timing['missing_counts']
        F.write(directory / 'AUTO_ACCEPTANCE.json', summary)
        return summary
    finally:
        # Only the worker/controller processes created above are signalled.
        # Every artificial job already finished on successful completion.
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=12)
            except subprocess.TimeoutExpired:
                child.kill();child.wait()
        marker = directory / 'run/distributed/LOCAL_DAEMON.json'
        if marker.exists():
            daemon = json.loads(marker.read_text())
            pid = daemon.get('pid')
            try:
                own_profile = str(directory / 'run/distributed/LOCAL_WORKER.json')
                command_bytes = Path('/proc/%s/cmdline' % pid).read_bytes().decode(errors='replace')
                if own_profile in command_bytes and 'run_rpe_q3_worker.py' in command_bytes:
                    os.kill(pid, signal.SIGTERM)
            except (OSError, TypeError):
                pass
        for log in logs:log.close()


def acceptance(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rows = {}
    for name, options in [('reference',dict(reference=True)), ('auto_four',{}), ('late_join',dict(join_late=True))]:
        print(json.dumps(dict(stage='ARTIFICIAL_NORMAL_ENTRY', case=name, event='START')), flush=True)
        rows[name] = run_one(directory / name, **options)
        print(json.dumps(dict(stage='ARTIFICIAL_NORMAL_ENTRY', case=name, event='END',
            outcomes=rows[name]['outcomes'], by_host=rows[name].get('by_host'),
            peak_bodies=rows[name]['artificial_overlap']['peak_concurrent_artificial_bodies'])), flush=True)
    reference_outcomes = F.outcomes(directory / 'reference')
    for name in ('auto_four', 'late_join'):
        if F.outcomes(directory / name) != reference_outcomes:
            raise AssertionError('Full artificial MSE bounds/decisions differ from the single-host reference: ' + name)
        if F.canonical_minima(rows[name]) != F.canonical_minima(rows['reference']):
            raise AssertionError('Artificial method minima differ: ' + name)
        if rows[name]['artificial_overlap']['peak_concurrent_artificial_bodies'] < 2:
            raise AssertionError('No actual artificial Q3 subprocess overlap: ' + name)
        if not rows[name]['artificial_overlap']['wsl_overlaps_preferred_artificial_body']:
            raise AssertionError('Optional WSL never overlapped a preferred-host artificial body: ' + name)
        if not all(rows[name].get('by_host', {}).get(host, {}).get('completed') for host in HOSTS[:3]):
            raise AssertionError('Existing three hosts did not all contribute: ' + name)
        if not rows[name]['by_host'].get('local0-wsl', {}).get('completed'):
            raise AssertionError('Participating optional WSL was not used: ' + name)
    result = dict(schema='RPE_AUTO_ALLOCATION_ARTIFICIAL_ENTRY_V1', status='PASS',
        scope='ORDINARY_ENTRY_ACTUAL_ARTIFICIAL_PROCESSES_SIMULATED_HOSTS_ONE_OS',
        cases=rows, exact_MSE_bounds_and_decisions_match_reference=True,
        method_minima_match_reference=True, scientific_H4_calls=0,
        signal_generation_calls=0, compile_calls=0, production_queue_submissions=0)
    F.write(directory / 'AUTO_ACCEPTANCE.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--controller', type=Path)
    group.add_argument('--acceptance', type=Path)
    args = parser.parse_args(argv)
    if args.controller:_controller(args.controller)
    else:acceptance(args.acceptance)


if __name__ == '__main__':main()
