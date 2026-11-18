"""Artificial acceptance for the ordinary schedule-search distributed queue.

Every configuration, price, signal and numerical result is artificial.  The
ordinary schedule CLI, candidate ordering and Q3 adapter remain production code.
Host profiles resolve paths independently; the controller does not read spools.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def profile(directory, host, slots=1):
    spool = Path(directory).resolve() / 'workers' / host
    return dict(host_id=host, root=str(ROOT), work_dir=str(spool),
                interpreter=sys.executable, slots=slots, route_views={},
                prefix_cache_directory=str(spool / 'prefix-cache'))


def prepare(directory, *, port=8765, local_slots=1, total_slots=3, sleep_seconds=.3,
            required_hosts=None):
    """Write only artificial input metadata and editable per-host profiles."""
    import check_pr_width_search as F
    directory = Path(directory).resolve()
    if (directory / 'PROBLEM.json').exists():
        raise ValueError('Use a fresh artificial fixture directory')
    F.WidthSearchTests().files(directory)
    problem = json.loads((directory / 'PROBLEM.json').read_text())
    for method in problem['methods'].values():
        method['L_D_values'] = [1, 2]
    write(directory / 'PROBLEM.json', problem)
    spec = json.loads((directory / 'ROUND.json').read_text())
    spec.update(round_id='ARTIFICIAL_DISTRIBUTED_Q3',
                max_new_q3_per_method={'PR_PAULI': 32, 'PR_DF': 32},
                artificial_fixture=dict(only=True, sleep_seconds=sleep_seconds,
                                        numerical_leaf='EXACT_RATIONAL_ARTIFICIAL_MSE'))
    spec['search_overrides'].update(lookahead=4)
    spec['executor'] = dict(scheduler='concurrent', max_concurrent_jobs=total_slots,
                            dp_workers=1, wait_for_running_jobs=True,
                            distributed=dict(bind='127.0.0.1', port=port,
                                             local_worker=profile(directory, 'server2', local_slots)))
    if required_hosts:
        spec['executor']['distributed']['required_hosts'] = list(required_hosts)
    write(directory / 'ROUND.json', spec)
    for host in ('server2', 'server1', 'local'):
        write(directory / ('PROFILE_' + host + '.json'), profile(directory, host))
    write(directory / 'FIXTURE.json', dict(artificial_only=True,
          scientific_H4_calls=0, base_commit='5829a7288d28395d2b3e89bfb676c23c690b9ddb',
          entry='scripts/run_rpe_schedule_search.py',
          fixture_leaves='tests/check_pr_width_search.py',
          controller_host='server2', shared_filesystem_required=False,
          result_scope='EXACT_RATIONAL_ARTIFICIAL_MSE; no molecular data or quantum circuit',
          controller=['--controller', str(directory), '--port', str(port), '--local-slots', str(local_slots)],
          worker=['scripts/run_rpe_q3_worker.py', '--profile', 'PROFILE_server1.json',
                  '--controller', 'http://127.0.0.1:' + str(port)]))
    return dict(status='PREPARED', artificial_only=True, directory=str(directory))


def fixture_driver():
    """Reuse the existing ordinary-entry fixture; alter only labelled leaves."""
    import check_pr_width_search as F

    class Completion(F.ArtificialCompletion):
        def ingest_price_results(self):
            # Initially only the base widths are READY.  Once those Q3 results
            # exist, publish artificial external price evidence.  The ordinary
            # width driver then refreshes and submits the waiting widths.
            if list((self.root / 'jobs').glob('*/*/OUTCOME.json')):
                write(self.root / 'ARTIFICIAL_PRICE_EVIDENCE.json', dict(ARTIFICIAL=True))
            return super().ingest_price_results()

        def install(self, context, problem, records):
            routes = super().install(context, problem, records)
            spec = json.loads((self.root.parent / 'ROUND.json').read_text())
            delay = spec['artificial_fixture']['sleep_seconds']
            for candidate in context['candidates']:
                candidate['caps'] = [1] * 12 + [2]
            for rules in routes.values():
                for rule in rules:
                    if rule['kind'] != 'artificial':
                        raise AssertionError('Fixture may only use artificial routes')
                    rule['artificial']['sleep_seconds'] = delay
            return routes

    class Driver(F.ArtificialDriver):
        def completion_api(self):
            if self.completion is None:
                self.completion = Completion(self.run)
            return self.completion

    return Driver


def controller(directory, *, port=None, local_slots=None):
    import run_rpe_schedule_search as entry
    directory = Path(directory).resolve()
    spec = json.loads((directory / 'ROUND.json').read_text())
    if spec.get('artificial_fixture', {}).get('only') is not True:
        raise ValueError('Only the labelled artificial fixture is accepted')
    distributed = spec['executor'].get('distributed')
    if distributed:
        if port is not None:
            distributed['port'] = port
        if local_slots is not None:
            distributed['local_worker']['slots'] = local_slots
        write(directory / 'ROUND.json', spec)
    started = time.monotonic()
    driver = entry.main(['--problem', str(directory / 'PROBLEM.json'),
                         '--round', str(directory / 'ROUND.json'),
                         '--run-dir', str(directory / 'run'),
                         '--max-cycles', '5', '--batch', '32'],
                        driver_factory=fixture_driver())
    write(directory / 'CONTROLLER_WALL.json', dict(artificial_only=True,
          wall_seconds=time.monotonic() - started, stop=driver.state.get('stop')))
    return check(directory)


def outcomes(directory):
    values = {}
    for file in sorted((Path(directory) / 'run').glob('jobs/*/*/OUTCOME.json')):
        request = json.loads((file.parent / 'REQUEST.json').read_text())
        result = json.loads(file.read_text())
        if request['route']['kind'] != 'artificial' or not result.get('artificial'):
            raise AssertionError('Non-artificial result in fixture')
        key = json.dumps([request['method'], request['configuration_id'],
                          request['N_X'], request['N_Y']], separators=(',', ':'))
        values[key] = [result['status'], result.get('MSE_lower'), result.get('MSE_upper')]
    return values


def check(directory):
    directory = Path(directory).resolve()
    values = outcomes(directory)
    result_path = directory / 'run' / 'ROUND_RESULT.json'
    results = json.loads(result_path.read_text())['methods'] if result_path.exists() else {}
    progress = directory / 'run' / 'PROGRESS.jsonl'
    events = [json.loads(line) for line in progress.read_text().splitlines()] if progress.exists() else []
    summary = dict(artificial_only=True, scientific_H4_calls=0,
                   outcomes=len(values), outcome_sha256=hashlib.sha256(
                       json.dumps(values, sort_keys=True).encode()).hexdigest(),
                   per_method={m: dict(best=r.get('best'), stop=r.get('stop_reason'),
                       new_calls=r.get('new_calls'), executor_metrics=r.get('executor_metrics'))
                       for m, r in results.items()},
                   launch_markers=len(list((directory / 'run').glob('jobs/*/*/LAUNCH.json'))),
                   event_counts={event: sum(row.get('event') == event for row in events)
                                 for event in sorted({row.get('event', '') for row in events})})
    timing = directory / 'CONTROLLER_WALL.json'
    if timing.exists():
        summary['controller_wall'] = json.loads(timing.read_text())
    database = directory / 'run' / 'distributed' / 'QUEUE.sqlite3'
    if database.exists():
        from trottertracks.rpe_q3_store import JobStore
        rows = JobStore(database).rows()
        hosts = sorted({row['host_id'] for row in rows if row['host_id']})
        summary['by_host'] = {host: dict(
            assigned=sum(row['host_id'] == host and row['assigned_epoch'] is not None for row in rows),
            launched=sum(row['host_id'] == host and row['started_epoch'] is not None for row in rows),
            completed=sum(row['host_id'] == host and row['state'] == 'ACCEPTED' for row in rows),
            adopted=sum(row['host_id'] == host and row['adopted_epoch'] is not None for row in rows),
            prefix_reuse_HITs=sum(json.loads(line).get('event') == 'HIT'
                for row in rows if row['host_id'] == host
                for path in [(directory / 'run' / 'jobs' / row['method'] / row['job_id'] / 'DP_PREFIX_EVENTS.jsonl')]
                if path.exists() for line in path.read_text().splitlines())) for host in hosts}
        waits = [row['assigned_epoch'] - row['enqueued_epoch'] for row in rows
                 if row['assigned_epoch'] is not None]
        summary['common_queue_wait_seconds'] = dict(count=len(waits), total=sum(waits),
            mean=sum(waits) / len(waits) if waits else None, maximum=max(waits) if waits else None)
        transfers = [row for row in events if row.get('event') == 'TRANSFER_COMPLETE']
        summary['controller_transfer_seconds'] = dict(count=len(transfers),
            total=sum(row.get('transfer_seconds', 0) for row in transfers),
            scope='CONTROLLER_RECEIVE_VALIDATION_AND_PERSISTENCE; excludes network round trip')
        summary['controller_transfer_bytes'] = sum(row.get('transfer_bytes', 0) for row in transfers)
        timings = [row for row in events if row.get('event') == 'TRANSFER_TIMING']
        summary['worker_transfer_roundtrip_seconds'] = dict(count=len(timings),
            total=sum(row.get('upload_roundtrip_seconds', 0) for row in timings),
            scope='WORKER_MEASURED_COMPLETE_UPLOAD_AND_ACK_ROUND_TRIP')
        summary['persistent_scientific_jobs'] = len([row for row in rows if row['state'] != 'CANCELLED'])
    write(directory / 'CHECKS.json', summary)
    return summary


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def launch_markers(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in (Path(directory) / 'run').glob('jobs/*/*/LAUNCH.json')}


def canonical_minima(summary):
    if any(value['best'] is None for value in summary['per_method'].values()):
        raise AssertionError('Artificial method has no accepted minimum; inspect controller log and worker SOURCE guards')
    return {method: {field: value['best'][field] for field in
            ('configuration_id', 'N_X', 'N_Y', 'status', 'MSE_lower', 'MSE_upper', 'cost')}
            for method, value in summary['per_method'].items()}


def _one_acceptance(directory, hosts):
    """Separate spools and HTTP clients on one OS, explicitly simulated hosts."""
    directory = Path(directory)
    port = free_port()
    prepare(directory, port=port, total_slots=hosts, sleep_seconds=.12)
    spec = json.loads((directory / 'ROUND.json').read_text())
    spec['executor']['distributed']['local_worker']['host_id'] = 'simulated-server2'
    write(directory / 'ROUND.json', spec)
    controller_cmd = [sys.executable, '-B', str(Path(__file__).resolve()),
                      '--controller', str(directory)]
    workers, logs = [], []
    started = time.monotonic()
    try:
        for host in ('simulated-server1', 'simulated-local')[:hosts - 1]:
            profile_path = directory / ('PROFILE_' + host + '.json')
            write(profile_path, profile(directory, host))
            log = (directory / (host + '.log')).open('ab')
            logs.append(log)
            workers.append(subprocess.Popen([sys.executable, '-B',
                str(ROOT / 'scripts/run_rpe_q3_worker.py'), '--profile', str(profile_path),
                '--controller', 'http://127.0.0.1:' + str(port), '--poll-seconds', '.02'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        with (directory / 'controller.log').open('ab') as log:
            subprocess.run(controller_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=180, check=True)
        first = check(directory)
        participating = {host for host, counts in first.get('by_host', {}).items()
                         if counts['completed'] > 0 and counts['adopted'] > 0}
        if len(participating) != hosts:
            raise AssertionError('Not every simulated host contributed accepted, adopted results: ' + str(first.get('by_host')))
        before = launch_markers(directory)
        with (directory / 'resume.log').open('ab') as log:
            subprocess.run(controller_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=180, check=True)
        if before != launch_markers(directory):
            raise AssertionError('Completed artificial jobs launched again on controller resume')
        first.update(simulated_host_count=hosts, actual_OS_host_count=1,
                     transport='LOOPBACK_HTTP_WITH_SEPARATE_WORKER_SPOOLS',
                     resume_new_launches=0, full_wall_seconds=time.monotonic() - started)
        write(directory / 'ACCEPTANCE.json', first)
        return first
    finally:
        # These processes were created above solely for this artificial test.
        # Never enumerate or signal unrelated processes or scientific workers.
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        for log in logs:
            log.close()


class FaultProxy:
    """Loopback-only test transport: drop traffic and duplicate one completion."""
    def __init__(self, backend):
        self.backend = backend.rstrip('/')
        self.offline = False
        self.duplicated = False
        self.events = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = self.rfile.read(int(self.headers['Content-Length']))
                if owner.offline:
                    owner.events.append(dict(event='ARTIFICIAL_NETWORK_OFFLINE', path=self.path))
                    self.send_response(503)
                    self.end_headers()
                    return
                request = urllib.request.Request(owner.backend + self.path, data=payload,
                          headers={'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(request, timeout=10) as response:
                        result = response.read()
                    if self.path == '/v1/complete' and not owner.duplicated:
                        owner.duplicated = True
                        with urllib.request.urlopen(request, timeout=10) as response:
                            duplicate = json.load(response)
                        owner.events.append(dict(event='DUPLICATE_COMPLETION', response=duplicate,
                                                job_id=json.loads(payload)['job_id']))
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(result)))
                    self.end_headers()
                    self.wfile.write(result)
                except (OSError, urllib.error.URLError) as error:
                    owner.events.append(dict(event='BACKEND_OFFLINE', reason=str(error), path=self.path))
                    self.send_response(503)
                    self.end_headers()

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def wait_for(predicate, *, seconds=30, reason):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError(reason)


def fault_acceptance(directory, *, live_ready_refill=False):
    """Kill only the synthetic controller; keep detached artificial Q3 running."""
    directory = Path(directory)
    port = free_port()
    prepare(directory, port=port, local_slots=0, total_slots=1, sleep_seconds=.4)
    if live_ready_refill:
        settings = json.loads((directory/'ROUND.json').read_text())
        settings.update(live_ready_refill=True, overlap_ready_batch=True)
        write(directory/'ROUND.json', settings)
    remote = profile(directory, 'simulated-recovery-worker')
    profile_path = directory / 'PROFILE_recovery.json'
    write(profile_path, remote)
    proxy = FaultProxy('http://127.0.0.1:' + str(port))
    controller_cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--controller', str(directory)]
    control = worker = None
    started = time.monotonic()
    with (directory / 'recovery-worker.log').open('ab') as worker_log, \
         (directory / 'controller-before-kill.log').open('ab') as before_log, \
         (directory / 'controller-after-restart.log').open('ab') as after_log:
        try:
            worker = subprocess.Popen([sys.executable, '-B', str(ROOT / 'scripts/run_rpe_q3_worker.py'),
                     '--profile', str(profile_path), '--controller', proxy.url, '--poll-seconds', '.02'],
                     cwd=ROOT, stdout=worker_log, stderr=subprocess.STDOUT)
            control = subprocess.Popen(controller_cmd, cwd=ROOT, stdout=before_log, stderr=subprocess.STDOUT)
            spool = Path(remote['work_dir'])

            def started_attempt():
                for path in spool.glob('*/*/JOB_STAGES.jsonl'):
                    if 'ARTIFICIAL_EVALUATION_START' in path.read_text():
                        return path.parent
                return None

            attempt = wait_for(started_attempt, reason='Artificial worker did not launch before controller kill')
            proxy.offline = True
            control.kill()
            killed_return = control.wait(timeout=10)
            wait_for(lambda: (attempt / 'ARTIFICIAL_RESULT.json').exists() and
                     json.loads((attempt / 'WORKER_STATE.json').read_text())['state'] == 'COMPLETED_PENDING_TRANSFER',
                     reason='Disconnected artificial job did not preserve completed transfer state')
            job_id = attempt.parent.name
            old_assignment = (attempt / 'ASSIGNMENT.json').read_bytes()
            preserved_result = (attempt / 'ARTIFICIAL_RESULT.json').read_bytes()
            control = subprocess.Popen(controller_cmd, cwd=ROOT, stdout=after_log, stderr=subprocess.STDOUT)
            proxy.offline = False
            code = control.wait(timeout=180)
            if code:
                raise AssertionError('Artificial restarted controller failed: ' + str(code))
            if old_assignment != (attempt / 'ASSIGNMENT.json').read_bytes():
                raise AssertionError('Recovered job changed attempt')
            if preserved_result != (attempt / 'ARTIFICIAL_RESULT.json').read_bytes():
                raise AssertionError('Recovered job numerical artifact changed')
            if (attempt / 'JOB_STAGES.jsonl').read_text().count('ARTIFICIAL_EVALUATION_START') != 1:
                raise AssertionError('Recovered completed job executed twice')
            duplicate = [event for event in proxy.events if event['event'] == 'DUPLICATE_COMPLETION']
            if not duplicate or not duplicate[0]['response'].get('duplicate'):
                raise AssertionError('Duplicate artificial completion was not idempotent')
            before = launch_markers(directory)
            with (directory / 'final-resume.log').open('ab') as log:
                subprocess.run(controller_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=180, check=True)
            after = launch_markers(directory)
            if any(after.get(name) != value for name, value in before.items()):
                raise AssertionError('Final resume changed an existing launch marker')
            # Crash recovery may continue with newly READY widths.  Those are
            # new scientific IDs, whereas rerunning an existing ID is forbidden.
            # Once those candidates finish, one further resume is fully idle.
            with (directory / 'settled-resume.log').open('ab') as log:
                subprocess.run(controller_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=180, check=True)
            if after != launch_markers(directory):
                raise AssertionError('Settled resume launched another artificial job')
            value = dict(status='PASS', artificial_only=True, actual_OS_host_count=1,
                         intentional_controller_returncode=killed_return, recovered_job_id=job_id,
                         disconnected_completed_pending_transfer=True, same_attempt_recovered=True,
                         recovered_job_execute_count=1, duplicate_completion=duplicate,
                         new_distinct_candidate_launches_on_resume=len(set(after) - set(before)),
                         existing_launch_marker_sha256={name: hashlib.sha256(data).hexdigest()
                                                       for name, data in before.items()},
                         completed_resume_new_launches=0, wall_seconds=time.monotonic() - started,
                         summary=check(directory), transport_events=proxy.events)
            write(directory / 'RECOVERY.json', value)
            return value
        finally:
            if control is not None and control.poll() is None:
                control.kill()
                control.wait()
            proxy.offline = False
            if worker is not None and worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            proxy.close()


def acceptance(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    reference = directory / 'single-reference'
    prepare(reference, total_slots=1, sleep_seconds=.12)
    spec = json.loads((reference / 'ROUND.json').read_text())
    spec['executor'].pop('distributed')
    spec['executor']['scheduler'] = 'legacy'
    write(reference / 'ROUND.json', spec)
    ref = controller(reference)
    expected = outcomes(reference)
    cases = {}
    for hosts in (1, 2, 3):
        case = directory / ('simulated-' + str(hosts) + '-hosts')
        summary = _one_acceptance(case, hosts)
        got = outcomes(case)
        common = set(expected).intersection(got)
        if not common or any(expected[key] != got[key] for key in common):
            raise AssertionError('Artificial MSE/status mismatch against single-host reference')
        if canonical_minima(summary) != canonical_minima(ref):
            raise AssertionError('Method minima changed against single-host reference')
        summary.update(reference_common_results_exact_match=len(common),
                       reference_minima_exact_match=True,
                       reference_entire_result_set_equal=set(expected) == set(got),
                       reference_result_count=len(expected), result_count=len(got),
                       artificial_wall_speedup=ref['controller_wall']['wall_seconds'] /
                           summary['controller_wall']['wall_seconds'],
                       artificial_speedup_reference='SINGLE_HOST_LEGACY_EXECUTOR')
        write(case / 'ACCEPTANCE.json', summary)
        cases[str(hosts)] = summary
    for hosts, summary in cases.items():
        summary['artificial_speedup_vs_single_distributed'] = (
            cases['1']['controller_wall']['wall_seconds'] / summary['controller_wall']['wall_seconds'])
        write(directory / ('simulated-' + hosts + '-hosts') / 'ACCEPTANCE.json', summary)
    recovery = fault_acceptance(directory / 'simulated-recovery')
    if canonical_minima(recovery['summary']) != canonical_minima(ref):
        raise AssertionError('Recovered method minima changed against reference')
    recovered_outcomes = outcomes(directory / 'simulated-recovery')
    if any(expected[key] != recovered_outcomes[key] for key in set(expected).intersection(recovered_outcomes)):
        raise AssertionError('Recovered MSE/status changed against reference')
    recovery['reference_minima_exact_match'] = True
    value = dict(schema='RPE_DISTRIBUTED_Q3_SIMULATED_ACCEPTANCE_V1', status='PASS',
                 artificial_only=True, scientific_H4_calls=0,
                 real_multi_machine_acceptance=False, actual_OS_host_count=1,
                 simulated_cases=cases,
                 recovery=recovery,
                 source_commit=subprocess.check_output(
                     ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(directory / 'SUMMARY.json', value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument('--prepare', type=Path)
    command.add_argument('--controller', type=Path)
    command.add_argument('--check', type=Path)
    command.add_argument('--acceptance', type=Path)
    parser.add_argument('--port', type=int)
    parser.add_argument('--local-slots', type=int)
    parser.add_argument('--total-slots', type=int, default=3)
    parser.add_argument('--sleep-seconds', type=float, default=.3)
    parser.add_argument('--required-hosts', nargs='+',
                        help='Wait for every named worker to register before assigning the first job')
    args = parser.parse_args(argv)
    if args.prepare:
        result = prepare(args.prepare, port=args.port or 8765,
                         local_slots=1 if args.local_slots is None else args.local_slots,
                         total_slots=args.total_slots, sleep_seconds=args.sleep_seconds,
                         required_hosts=args.required_hosts)
    elif args.controller:
        result = controller(args.controller, port=args.port, local_slots=args.local_slots)
    elif args.check:
        result = check(args.check)
    else:
        result = acceptance(args.acceptance)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == '__main__':
    main()
