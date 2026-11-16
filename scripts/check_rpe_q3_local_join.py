"""Late join/disconnect/reconnect using the fixed common artificial fixture.

All simulated hosts run on ONE Linux OS. No real SSH or H4 input is used.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    common = args.common_root.resolve()
    sys.path[:0] = [str(common / name) for name in ('scripts', 'src', 'tests')]
    import check_rpe_distributed_q3 as F
    from trottertracks.rpe_q3_store import JobStore

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output / 'late-join'
    port = F.free_port()
    F.prepare(run, port=port, total_slots=3, sleep_seconds=1., required_hosts=['server2', 'simulated-server1'])
    controller_command = [sys.executable, '-B', str(common / 'scripts/check_rpe_distributed_q3.py'), '--controller', str(run)]
    proxy = F.FaultProxy('http://127.0.0.1:' + str(port))
    processes, logs = [], []

    def launch(command, label):
        log = (output / (label + '.log')).open('ab')
        logs.append(log)
        process = subprocess.Popen(command, cwd=common, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def worker(host, url):
        profile = F.profile(run, host)
        path = run / ('PROFILE_' + host + '.json')
        F.write(path, profile)
        command = [sys.executable, '-B', str(common / 'scripts/run_rpe_q3_worker.py'),
                   '--profile', str(path), '--controller', url, '--poll-seconds', '.02']
        return profile, command

    def started(spool):
        return next((p.parent for p in Path(spool).glob('*/*/JOB_STAGES.jsonl')
                     if 'ARTIFICIAL_EVALUATION_START' in p.read_text()), None)

    try:
        server1, first_command = worker('simulated-server1', 'http://127.0.0.1:' + str(port))
        launch(first_command, 'server1')
        controller = launch(controller_command, 'controller')
        F.wait_for(lambda: started(server1['work_dir']) and started(F.profile(run, 'server2')['work_dir']),
                   reason='Both initial simulated hosts must start artificial Q3 before local joins')
        local, local_command = worker('simulated-local', proxy.url)
        local_process = launch(local_command, 'local-before-disconnect')
        attempt = F.wait_for(lambda: started(local['work_dir']), reason='Late local received no artificial Q3')
        assignment = (attempt / 'ASSIGNMENT.json').read_bytes()
        database = run / 'run/distributed/QUEUE.sqlite3'

        def other_completed():
            return sum(row['host_id'] != 'simulated-local' and row['state'] == 'ACCEPTED'
                       for row in JobStore(database).rows())

        before = other_completed()
        proxy.offline = True
        local_process.kill()  # Only this test's polling daemon; detached artificial job continues.
        local_process.wait(timeout=10)
        F.wait_for(lambda: (attempt / 'ARTIFICIAL_RESULT.json').exists() and
                   json.loads((attempt / 'WORKER_STATE.json').read_text())['state'] == 'COMPLETED_PENDING_TRANSFER',
                   reason='Disconnected local did not retain completed result')
        F.wait_for(lambda: other_completed() > before, reason='Other hosts did not continue while local was offline')
        result = (attempt / 'ARTIFICIAL_RESULT.json').read_bytes()
        proxy.offline = False
        launch(local_command, 'local-after-reconnect')
        if controller.wait(timeout=180) != 0:
            raise AssertionError('Artificial controller failed; inspect controller.log')
        summary = F.check(run)
        reference = output / 'single-reference'
        F.prepare(reference, total_slots=1, sleep_seconds=0)
        spec = json.loads((reference / 'ROUND.json').read_text())
        spec['executor'].pop('distributed')
        spec['executor']['scheduler'] = 'legacy'
        F.write(reference / 'ROUND.json', spec)
        ref = F.controller(reference)
        expected, observed = F.outcomes(reference), F.outcomes(run)
        common_keys = set(expected).intersection(observed)
        duplicates = [e for e in proxy.events if e['event'] == 'DUPLICATE_COMPLETION']
        events = JobStore(database).events()
        job_ids = {row['job_id'] for row in JobStore(database).rows()}
        checks = dict(local_joined_after_both_initial_hosts_started=True,
            local_received_and_adopted=summary['by_host']['simulated-local']['adopted'] > 0,
            other_hosts_continued_while_local_offline=True,
            same_assignment_on_reconnect=assignment == (attempt / 'ASSIGNMENT.json').read_bytes(),
            saved_result_unchanged=result == (attempt / 'ARTIFICIAL_RESULT.json').read_bytes(),
            disconnected_job_execute_count_one=(attempt / 'JOB_STAGES.jsonl').read_text().count('ARTIFICIAL_EVALUATION_START') == 1,
            duplicate_completion_idempotent=bool(duplicates) and duplicates[0]['response'].get('duplicate') is True,
            no_duplicate_acceptance_or_adoption=all(
                sum(e['job_id'] == job and e['event'] == name for e in events) <= 1
                for job in job_ids for name in ('ACCEPT', 'ADOPT')),
            reference_result_set_equal=set(expected) == set(observed),
            reference_MSE_and_decisions=bool(common_keys) and all(expected[k] == observed[k] for k in common_keys),
            reference_method_minima=F.canonical_minima(summary) == F.canonical_minima(ref))
        report = dict(status='PASS' if all(checks.values()) else 'FAIL', actual_OS_hosts=1,
            real_multi_machine_acceptance=False, scientific_H4_calls=0,
            common_source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=common, text=True).strip(),
            checks=checks, compared_reference_results=len(common_keys),
            by_host=summary['by_host'], recovered_job_id=attempt.parent.name)
        F.write(output / 'LOCAL_JOIN_CHECKS.json', report)
        print(json.dumps(report, indent=2))
        return 0 if all(checks.values()) else 1
    finally:
        proxy.offline = False
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        proxy.close()
        for log in logs:
            log.close()


if __name__ == '__main__':
    raise SystemExit(main())
