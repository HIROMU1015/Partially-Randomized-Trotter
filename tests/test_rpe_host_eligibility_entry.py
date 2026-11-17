"""Ordinary schedule entry -> distributed RPC with artificial host eligibility.

All requests, fingerprints and completed OUTCOMEs are artificial. No scientific
adapter or worker process is launched. The public entry uses its default PF
method-round branch, without a driver_factory replacement.
"""
import base64
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import run_rpe_schedule_search as Entry
from trottertracks import pf_candidate_selection as PFSelect
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_optimizer_concurrent as C
from trottertracks import rpe_q3_coordinator as Q
from trottertracks.rpe_q3_store import JobStore
from test_rpe_optimizer_concurrent import fixture


METHODS = ['FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF']
HOSTS = ['server2', 'server1', 'local']
SOURCE = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


class HostEligibilityEntryTests(unittest.TestCase):
    def test_default_method_entry_filters_hosts_holds_pauli_and_resumes_without_launch(self):
        problem, context, routes = fixture(methods=METHODS, configurations=1)
        problem['execution_routes'] = routes
        previews = O.optimize(ROOT, problem, context=context, methods=METHODS,
                              max_new={m: 0 for m in METHODS}, science_approved=False)
        self.assertTrue(all(previews[m]['pending_request'] for m in METHODS))
        cli = Entry.K.CLI
        original_factory = cli.make_executor
        created = threading.Event()
        executors, errors, outcomes, claims = [], [], [], []
        with tempfile.TemporaryDirectory(prefix='rpe-artificial-host-entry-') as temporary:
            base = Path(temporary);run = base / 'artificial-run'
            policy_path = base / 'ARTIFICIAL_HOST_POLICY.json'
            problem_path = base / 'ARTIFICIAL_PROBLEM.json';problem_path.write_text(json.dumps(problem))
            round_path = base / 'ARTIFICIAL_ROUND.json'
            round_spec = dict(round_id='ARTIFICIAL_HOST_ELIGIBILITY_ENTRY', methods=METHODS, initial='all',
                science_approved=True, max_new_q3_per_method={m: 1 for m in METHODS},
                executor=dict(scheduler='concurrent', max_concurrent_jobs=3, dp_workers=1,
                    distributed=dict(bind='127.0.0.1', port=0, affinity_wait_seconds=0,
                                     host_eligibility=str(policy_path))))
            round_path.write_text(json.dumps(round_spec))
            fingerprints, job_ids = {}, {}

            def factory(*args, **kwargs):
                executor, optimize = original_factory(*args, **kwargs)
                self.assertIsInstance(executor, C.DistributedExecutor)
                policy = dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', fixture='ARTIFICIAL_ONLY', jobs={})
                for method in METHODS:
                    view = executor.for_method(method)
                    proposal = previews[method]['pending_request']
                    job = view.job_dir(proposal)
                    request = view.request(proposal, job, view.route(proposal['configuration_id']))
                    job_ids[method] = job.name
                    allowed = [] if method == 'PR_PAULI' else ['server2'] if method == 'PR_DF' else list(HOSTS)
                    hosts = {}
                    for host in HOSTS:
                        facts = dict(fixture='ARTIFICIAL_ONLY', host_id=host, adapter_runtime='ARTIFICIAL_RUNTIME')
                        fingerprints.setdefault(host, {})[job.name] = facts
                        hosts[host] = dict(status='PASS' if host in allowed else 'MISSING',
                            current_host_fingerprint=facts, current_host_fingerprint_sha256=Q.digest(facts))
                    record = dict(method=method, request_sha256=Q.digest(request), source_manifest=executor.source_manifest,
                        source_manifest_sha256=Q.digest(executor.source_manifest), input_hashes=[],
                        eligible_host_ids=allowed, hosts=hosts)
                    record['decision_sha256'] = Q.digest(record)
                    policy['jobs'][job.name] = record
                policy_path.write_text(json.dumps(policy))
                executors.append(executor);created.set()
                return executor, optimize

            def invoke():
                try:
                    outcomes.append(Entry.main(['--problem', str(problem_path), '--round', str(round_path), '--run-dir', str(run)]))
                except BaseException as exc:
                    errors.append(exc)

            def rpc(executor, path, value):
                request = Request(executor.controller_url + path, data=json.dumps(value).encode(),
                                  headers={'Content-Type': 'application/json'}, method='POST')
                with urlopen(request, timeout=3) as response:
                    return json.load(response)

            def poll(executor, host):
                return rpc(executor, '/v1/poll', dict(host_id=host, slots=1, source_sha=SOURCE,
                    cached_prefixes=[], host_fingerprints=fingerprints[host]))

            def complete(executor, assignment, host):
                request = assignment['request']
                allocation = [dict(m=m, N_X=x, N_Y=y) for m, (x, y)
                              in enumerate(zip(request['N_X'], request['N_Y']))]
                outcome = dict(artificial=True, fixture='ARTIFICIAL_RPC_COMPLETION_NO_LEAF', status=O.PASS,
                    MSE_lower=O.enc(Fraction(3)), MSE_upper=O.enc(Fraction(3)),
                    identity=dict(request['expected'], allocation=allocation, contract_matches=True),
                    request_id=assignment['job_id'], route='artificial')
                identity = dict(host_id=host, job_id=assignment['job_id'], attempt_id=assignment['attempt_id'])
                rpc(executor, '/v1/state', dict(identity, state='RUNNING', detail={'fixture': 'ARTIFICIAL_NO_PROCESS'}))
                files = {'OUTCOME.json': outcome, 'PROCESS_RETURN.json': dict(returncode=0, signal=None, artificial=True)}
                answer = rpc(executor, '/v1/complete', dict(identity, request_sha256=assignment['request_sha256'],
                    files={name: base64.b64encode(json.dumps(value).encode()).decode() for name, value in files.items()}))
                self.assertTrue(answer['ok'])

            def pf_prepare(root, problem, current, spec, directory, effective_routes):
                return current, effective_routes, dict(status='ARTIFICIAL_EXISTING_INPUTS', artificial_only=True)

            with patch.object(O, 'prepare', return_value=context), patch.object(PFSelect, 'prepare', side_effect=pf_prepare), \
                 patch.object(cli, 'head', return_value=SOURCE), patch.object(cli, 'make_executor', side_effect=factory), \
                 patch.object(O.Executor, 'run_adapter', side_effect=AssertionError('No scientific or artificial adapter leaf is called')), \
                 redirect_stdout(io.StringIO()):
                thread = threading.Thread(target=invoke, daemon=True)
                thread.start()
                try:
                    self.assertTrue(created.wait(4), errors)
                    executor = executors[0]
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        rows = executor.store.rows()
                        if len(rows) == 3 and job_ids['PR_PAULI'] in executor.eligibility_holds:
                            break
                        if errors: raise errors[0]
                        time.sleep(.005)
                    self.assertEqual({r['method'] for r in executor.store.rows()}, {'FULL_PF', 'TRUNCATED_PF', 'PR_DF'}, executor.eligibility_holds)
                    self.assertIsNone(executor.store.row(job_ids['PR_PAULI']))
                    for host in ['server1', 'local']:
                        assigned = poll(executor, host)['assignments']
                        self.assertEqual(len(assigned), 1)
                        self.assertIn(assigned[0]['method'], ('FULL_PF', 'TRUNCATED_PF'))
                        claims.append((host, assigned[0]))
                    self.assertEqual({a['method'] for _, a in claims}, {'FULL_PF', 'TRUNCATED_PF'})
                    df = executor.store.row(job_ids['PR_DF'])
                    self.assertEqual(df['state'], 'QUEUED')
                    self.assertIsNone(df['host_id'])
                    assigned = poll(executor, 'server2')['assignments']
                    self.assertEqual(len(assigned), 1)
                    self.assertEqual(assigned[0]['method'], 'PR_DF')
                    claims.append(('server2', assigned[0]))
                    for host, assignment in claims:
                        complete(executor, assignment, host)
                    thread.join(6)
                    self.assertFalse(thread.is_alive(), 'Ordinary entry did not finish after artificial completions')
                    if errors: raise errors[0]
                    first = json.loads((run / 'ROUND_RESULT.json').read_text())
                    self.assertEqual(outcomes[0].state['stop']['reason'], 'METHOD_ROUND_COMPLETE')
                    self.assertEqual(first['methods']['PR_PAULI']['stop_reason'], 'HOST_ELIGIBILITY_HOLD')
                    self.assertEqual(first['methods']['PR_PAULI']['new_calls'], 0)
                    self.assertEqual(first['methods']['PR_PAULI']['pending_request']['hold_reason'], 'NO_MATCHING_ELIGIBLE_HOST')
                    self.assertEqual(first['methods']['PR_PAULI']['executor_metrics']['persistent_reserved']['PR_PAULI'], 0)
                    launch_bytes = {str(p.relative_to(run)): p.read_bytes() for p in run.glob('jobs/*/*/LAUNCH.json')}
                    self.assertEqual(len(launch_bytes), 3)
                    self.assertFalse(list(run.glob('jobs/PR_PAULI/*/LAUNCH.json')))
                    invoke()  # The same public entry resumes the completed scientific job IDs.
                    if errors: raise errors[0]
                    resumed = json.loads((run / 'ROUND_RESULT.json').read_text())
                    self.assertEqual(launch_bytes, {str(p.relative_to(run)): p.read_bytes() for p in run.glob('jobs/*/*/LAUNCH.json')})
                    self.assertTrue(all(r['new_calls'] == 0 for r in resumed['methods'].values()))
                    self.assertTrue(all(r['executor_metrics']['new_real_route_launches'] == 0 for r in first['methods'].values()))
                    store = JobStore(run / 'distributed/QUEUE.sqlite3')
                    self.assertEqual(store.counts('PR_PAULI'), 0)
                    self.assertEqual(len(store.rows()), 3)
                    self.assertTrue(all(r['state'] == 'ACCEPTED' for r in store.rows()))
                    acceptance = dict(schema='ARTIFICIAL_Q3_HOST_ELIGIBILITY_ENTRY_ACCEPTANCE_V1', status='PASS',
                        scope='ARTIFICIAL_LOOPBACK_RPC_COMPLETIONS_NO_ADAPTER_LEAF',
                        entry='scripts/run_rpe_schedule_search.py', driver_factory_used=False,
                        policy_loaded_from_file=True, source=SOURCE,
                        earlier_claims=[dict(host_id=h, method=a['method'], job_id=a['job_id']) for h, a in claims[:2]],
                        df_claim=dict(host_id='server2', job_id=job_ids['PR_DF']),
                        first_artificial_assignments=3, accepted_artificial_jobs=3,
                        pauli=dict(stop=first['methods']['PR_PAULI']['stop_reason'], reservations=0, launches=0),
                        completed_run_resume_new_launches=0, scientific_adapter_calls=0, new_real_route_launches=0,
                        artificial_method_budgets={m: 1 for m in METHODS})
                    output = os.environ.get('RPE_HOST_ELIGIBILITY_ENTRY_ACCEPTANCE')
                    if output:
                        path = Path(output);path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(acceptance, indent=2) + '\n')
                finally:
                    if thread.is_alive() and executors:
                        ex = executors[0]
                        with ex.state.cv:
                            for task in ex.state.tasks.values():
                                if task.status == 'RUNNING':
                                    task.status = 'DONE';task.error = 'ARTIFICIAL_TEST_CLEANUP';task.event.set()
                            ex.state.active = 0;ex.state.cv.notify_all()
                        thread.join(5)


if __name__ == '__main__':
    unittest.main()
