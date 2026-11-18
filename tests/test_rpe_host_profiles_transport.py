"""Artificial controller -> mailbox -> saved adapter-command profile binding.

No interpreter or adapter is executed: every Popen in the worker path is
replaced by a recording fake. Runtime strings and route commits are artificial.
"""
import base64
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from trottertracks import rpe_q3_worker as W
from trottertracks import rpe_q3_coordinator as Q
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
from trottertracks.rpe_host_profiles import profile_core, select_host_profile
from test_rpe_optimizer_concurrent import fixture


class HostProfileTransportTests(unittest.TestCase):
    def test_selective_configuration_is_bound_to_pr_routes_and_capable_hosts(self):
        methods = ['FULL_PF', 'PR_PAULI', 'PR_DF']
        problem, context, routes = fixture(methods=methods, configurations=1)
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            executor = DistributedExecutor(ROOT, tmp, routes, context['expected'], problem,
                optimizer_source=source, distributed=dict(port=0),
                prefix_checkpoint_rounds=[5,8,11], prefix_selective_checkpoint_rounds=[8,11],
                prefix_selective_window_seconds=600, prefix_cache_max_bytes=40*2**30,
                compact_q3_evidence=True, parallel_dp_residual=True, profile_dp_residual=True)
            try:
                adapter = {'files':{'src/trottertracks/rpe_dp_prefix_reuse.py':'fixture',
                                    'src/trottertracks/rpe_q3_evidence.py':'fixture'}}
                profile = dict(host_id='worker', reviewed_pr_q3=True,
                               prefix_cache_directory='/private/cache', adapter_manifest=adapter)
                for candidate in context['candidates']:
                    method = candidate['method']
                    kinds = {'FULL_PF':['pf_b_pf6'], 'PR_PAULI':['pauli_b_common_entry'],
                             'PR_DF':['prdf_d2_split','prdf_d2_common_entry']}[method]
                    view = executor.for_method(method)
                    proposal = dict(configuration_id=candidate['configuration_id'], vector=(1,)*26,
                                    cost=26, origin={})
                    for kind in kinds:
                        request = view.request(proposal, view.job_dir(proposal), {'kind':kind})
                        options = request['execution_profile']
                        if method == 'FULL_PF':
                            self.assertEqual(options, {'dp_workers':None})
                            continue
                        self.assertEqual(options['prefix_selective_checkpoint_rounds'], [8,11])
                        self.assertEqual(options['prefix_selective_window_seconds'], 600)
                        self.assertTrue(options['profile_dp_residual'])
                        self.assertEqual(options.get('compact_q3_evidence'), None if kind=='prdf_d2_split' else True)
                        with self.assertRaisesRegex(ValueError, 'No compatible selective PR Q3 adapter'):
                            select_host_profile(profile, {'request':request})
                        capable = dict(profile, selective_pr_q3=True)
                        self.assertEqual(profile_core(select_host_profile(capable, {'request':request})),
                                         profile_core(capable))
            finally:
                executor.finish()

    def test_assignment_mailbox_adapter_and_duplicate_keep_route_specific_interpreter(self):
        methods = ['FULL_PF', 'TRUNCATED_PF', 'PR_DF']
        problem, context, routes = fixture(methods=methods, configurations=1)
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        for method in methods:
            routes[method][0]['view_commit'] = 'ARTIFICIAL_DF_ROUTE' if method == 'PR_DF' else 'ARTIFICIAL_PF_ROUTE'
        with tempfile.TemporaryDirectory(prefix='rpe-artificial-route-profile-') as temporary:
            base = Path(temporary);run = base / 'controller';spool = base / 'server1-spool'
            common = str(base / 'ARTIFICIAL_common_python312')
            dedicated = str(base / 'ARTIFICIAL_prdf_python311rc1_launcher')
            # These names are data only; Popen is never allowed to invoke them.
            profile = dict(host_id='server1', root=str(ROOT), interpreter=common, route_views={},
                           work_dir=str(spool), slots=3, dp_workers=1)
            special = dict(profile_core(profile), interpreter=dedicated)
            profile['route_profiles'] = [dict(route_kind='artificial', view_commit='ARTIFICIAL_DF_ROUTE', profile=special)]
            policy_file = base / 'ARTIFICIAL_HOST_POLICY.json'
            executor = DistributedExecutor(ROOT, run, routes, context['expected'], problem,
                optimizer_source=source, max_concurrent=3, budgets={m: 1 for m in methods}, science_approved=True,
                distributed=dict(bind='127.0.0.1', port=0, affinity_wait_seconds=0, host_eligibility=str(policy_file)))
            def finish_artificial():
                with executor.state.cv:
                    for task in executor.state.tasks.values():
                        if task.status == 'RUNNING':
                            task.status = 'DONE';task.error = 'ARTIFICIAL_TEST_CLEANUP';task.event.set()
                    executor.state.active = 0;executor.state.cv.notify_all()
                executor.finish()
            self.addCleanup(finish_artificial)
            assignments, proposals, facts = {}, {}, {}
            policy = dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', fixture='ARTIFICIAL_ONLY', jobs={})
            for candidate in context['candidates']:
                method = candidate['method'];view = executor.for_method(method)
                proposal = dict(configuration_id=candidate['configuration_id'], vector=tuple([1] * 26),
                    cost=Fraction(sum(candidate['prices'])), origin={'kind': 'ARTIFICIAL_PROFILE_CHECK'})
                proposals[method] = proposal
                job = view.job_dir(proposal);request = view.request(proposal, job, view.route(proposal['configuration_id']))
                core = profile_core(select_host_profile(profile, dict(request=request)))
                fact = dict(fixture='ARTIFICIAL_ONLY', host_id='server1', profile=core)
                facts[job.name] = fact
                allowed = ['server2', 'server1'] if method == 'PR_DF' else ['server2', 'server1', 'local']
                hosts = {host: dict(status='PASS', current_host_fingerprint_sha256=Q.digest(fact),
                                   current_host_fingerprint=fact, execution_profile=dict(core, host_id=host)) for host in allowed}
                record = dict(method=method, request_sha256=Q.digest(request), source_manifest=executor.source_manifest,
                    source_manifest_sha256=Q.digest(executor.source_manifest), input_hashes=[],
                    eligible_host_ids=allowed, hosts=hosts)
                record['decision_sha256'] = Q.digest(record);policy['jobs'][job.name] = record
            policy_file.write_text(json.dumps(policy))
            for method in methods:
                self.assertTrue(executor.for_method(method)._enqueue(proposals[method], demand=True))
            polled = executor.rpc('/v1/poll', dict(host_id='server1', slots=3, source_sha=source,
                                 cached_prefixes=[], host_fingerprints=facts))
            self.assertEqual(len(polled['assignments']), 3)
            assignments = {a['method']: a for a in polled['assignments']}
            for method, assignment in assignments.items():
                want = dedicated if method == 'PR_DF' else common
                self.assertEqual(assignment['host_execution_profile']['interpreter'], want)
                self.assertEqual(assignment['host_execution_profile_sha256'], Q.digest(assignment['host_execution_profile']))

            # A changed policy cannot rewrite the already assigned attempt's execution profile.
            changed = deepcopy(policy)
            df_id = assignments['PR_DF']['job_id']
            changed['jobs'][df_id]['hosts']['server1']['execution_profile']['interpreter'] = common
            policy_file.write_text(json.dumps(changed))
            replay = executor.rpc('/v1/poll', dict(host_id='server1', slots=3, source_sha=source,
                                  cached_prefixes=[], host_fingerprints=facts))
            self.assertEqual({a['method']: a for a in replay['assignments']}, assignments)

            commands, saved = [], {}
            class FakeProcess:
                def __init__(self, command):
                    commands.append(list(command));self.pid = 99900000 + len(commands)
                def wait(self, *args, **kwargs): return 0
                def poll(self): return 0

            with patch.object(W, 'head', return_value=source), \
                 patch.object(W.subprocess, 'Popen', side_effect=lambda command, **kwargs: FakeProcess(command)):
                for method in methods:
                    assignment = assignments[method]
                    self.assertEqual(W.mailbox(profile, 'submit', assignment), dict(ok=True, duplicate=False))
                    attempt = spool / assignment['job_id'] / assignment['attempt_id']
                    stored = W.read(attempt / 'WORKER_PROFILE.json');saved[method] = (attempt, stored)
                    want = dedicated if method == 'PR_DF' else common
                    self.assertEqual(stored['interpreter'], want)
                    self.assertNotIn('route_profiles', stored)
                    self.assertEqual(stored['work_dir'], str(spool));self.assertEqual(stored['slots'], 3)
                    self.assertEqual(commands[-1][0], want)  # Detached supervisor uses the same selected recipe.
                    resolved = W.validate_assignment(stored, assignment)
                    self.assertEqual(resolved['route']['interpreter'], want)
                    self.assertEqual(assignment['request']['route']['interpreter'], sys.executable)
                    W.atomic(attempt / 'RESOLVED_REQUEST.json', resolved)
                    self.assertEqual(W._run_adapter(stored, attempt, 'execute', attempt), 0)
                    self.assertEqual(commands[-1][0], want)
                    self.assertEqual(commands[-1][-2:], ['--mode', 'execute'])
                    request = assignment['request']
                    allocation = [dict(m=m, N_X=x, N_Y=y) for m, (x, y) in enumerate(zip(request['N_X'], request['N_Y']))]
                    outcome = dict(artificial=True, fixture='ARTIFICIAL_NO_EXECUTION_PROFILE_CHECK',
                        status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', MSE_lower={'numerator': 1, 'denominator': 10},
                        MSE_upper={'numerator': 1, 'denominator': 10}, request_id=assignment['job_id'], route='artificial',
                        identity=dict(request['expected'], allocation=allocation, contract_matches=True))
                    W.atomic(attempt / 'OUTCOME.json', outcome)
                    W.save_state(attempt, 'COMPLETED_PENDING_TRANSFER', fixture='ARTIFICIAL_ONLY')
                    before = len(commands)
                    self.assertEqual(W.mailbox(profile, 'submit', assignment), dict(ok=True, duplicate=True))
                    self.assertEqual(len(commands), before)
                    self.assertEqual(W.read(attempt / 'WORKER_PROFILE.json'), stored)
                wrong_common = dict(profile);wrong_common.pop('route_profiles')
                before = len(commands)
                with self.assertRaisesRegex(ValueError, 'common fallback is forbidden'):
                    W.mailbox(dict(wrong_common, work_dir=str(base / 'new-wrong-common-spool')),
                              'submit', assignments['PR_DF'])
                self.assertEqual(len(commands), before)
                # Existing attempts use their stored profile even when the daemon's later base lacks the route recipe.
                self.assertEqual(W.mailbox(wrong_common, 'submit', assignments['PR_DF']), dict(ok=True, duplicate=True))
                resumed = W.Worker(wrong_common, 'http://127.0.0.1:1', transport=lambda *args: None)
                try:
                    self.assertEqual(len(resumed.records), 3)
                    for method, assignment in assignments.items():
                        resumed._acquire(assignment)
                        attempt, stored = saved[method]
                        self.assertEqual(W.read(attempt / 'WORKER_PROFILE.json'), stored)
                        self.assertEqual(W._adapter_command(stored, attempt, 'execute', attempt)[0],
                                         dedicated if method == 'PR_DF' else common)
                    self.assertEqual(len(commands), before)
                finally:
                    resumed.close()

            for method, assignment in assignments.items():
                attempt, stored = saved[method]
                files = {name: base64.b64encode((attempt / name).read_bytes()).decode()
                         for name in ('OUTCOME.json', 'PROCESS_RETURN.json')}
                response = executor.rpc('/v1/complete', dict(host_id='server1', job_id=assignment['job_id'],
                    attempt_id=assignment['attempt_id'], request_sha256=assignment['request_sha256'], files=files))
                self.assertTrue(response['ok'])
            executor.finish()
            result = dict(schema='ARTIFICIAL_ROUTE_PROFILE_TRANSPORT_ACCEPTANCE_V1', status='PASS',
                scope='ARTIFICIAL_CONTROLLER_MAILBOX_AND_COMMANDS_ALL_POPEN_MOCKED',
                official_reference='9d022a0f7', methods=methods, controller_assignments=3,
                dedicated_profile_methods=['PR_DF'], common_profile_methods=['FULL_PF', 'TRUNCATED_PF'],
                supervisor_and_execute_interpreter_match=True, wrong_common_profile_rejected=True,
                assigned_policy_change_preserves_same_attempt=True, duplicate_and_resume_profile_preserved=True,
                recorded_supervisor_commands=3, recorded_execute_commands=3, actual_Popen_calls=0,
                Popen_count_scope='Worker supervisor and adapter launch paths only; read-only Git metadata discovery is outside this count.',
                real_Q3_calls=0, real_dry_check_calls=0, scientific_adapter_executions=0)
            output = os.environ.get('RPE_Q3_PROFILE_ACCEPTANCE')
            if output:
                path = Path(output);path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    unittest.main()
