"""Synthetic queue acceptance for nonblocking, low-priority price checks.

No real signal, compiler or Q3 kernel is executed. Reuse the distributed
controller's durable claim and completion fixtures.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
import test_rpe_distributed_controller as Base
import test_rpe_preparation_queue as Prep
from trottertracks.rpe_preparation_queue import normalize_preparation_request


class OptionalPriceValidationQueue(unittest.TestCase):
    setUp = Base.ControllerFailures.setUp
    executor = Base.ControllerFailures.executor
    close_one = Base.ControllerFailures.close_one
    close_all = Base.ControllerFailures.close_all
    proposal = Base.ControllerFailures.proposal
    enqueue = Base.ControllerFailures.enqueue
    completion = Base.ControllerFailures.completion
    poll_host = Prep.PreparationQueue.poll_host
    prep_complete = Prep.PreparationQueue.prep_complete

    def request(self, name='model', optional=True):
        return dict(method='PR_PAULI', route=dict(kind='preparation'), preparation=dict(
            stage='compile', provider='PRICE_MODEL_VALIDATION', artificial=True,
            optional_price_validation=optional, command=['ARTIFICIAL_ONLY', name], outputs=['receipt.json']))

    def submit(self, executor, name='model', optional=True):
        return executor.submit_preparation(self.request(name, optional), approved=True, budget=8)

    def test_ordinary_q3_including_lookahead_runs_before_optional_validation(self):
        ex = self.executor(capacity=1)
        validation = self.submit(ex)
        normal = ex.for_method('PR_PAULI')._enqueue(self.proposal('normal-lookahead'), demand=False)
        assigned = self.poll_host(ex, 'server1')['assignments']
        self.assertEqual([a['job_id'] for a in assigned], [normal.job.name])
        self.assertEqual(ex.store.row(validation)['state'], 'QUEUED')
        ex.rpc('/v1/complete', self.completion(assigned[0]))
        self.assertEqual(self.poll_host(ex, 'server1')['assignments'][0]['job_id'], validation)

    def test_ordinary_preparation_runs_before_earlier_optional(self):
        ex = self.executor(capacity=1)
        optional = self.submit(ex)
        normal = self.submit(ex, 'required-calibration', optional=False)
        first = self.poll_host(ex, 'server1')['assignments'][0]
        self.assertEqual(first['job_id'], normal)
        self.assertEqual(ex.store.row(optional)['state'], 'QUEUED')
        ex.rpc('/v1/complete', self.prep_complete(ex, first))
        self.assertEqual(self.poll_host(ex, 'server1')['assignments'][0]['job_id'], optional)

    def test_concurrent_submit_and_host_claim_are_once(self):
        ex = self.executor(capacity=2)
        with ThreadPoolExecutor(4) as pool:
            jobs = list(pool.map(lambda _: self.submit(ex), range(8)))
        self.assertEqual(len(set(jobs)), 1)
        with ThreadPoolExecutor(2) as pool:
            replies = list(pool.map(lambda host: self.poll_host(ex, host), ['server1', 'server2']))
        assigned = [a for reply in replies for a in reply['assignments']]
        self.assertEqual(len(assigned), 1)
        self.assertEqual(sum(e['event'] == 'ASSIGN' for e in ex.store.events()), 1)

    def test_pending_validation_does_not_block_q3_or_shutdown_and_survives_resume(self):
        ex = self.executor(capacity=1)
        optional = self.submit(ex)
        normal = self.enqueue(ex, 'normal')
        assignment = self.poll_host(ex, 'server1')['assignments'][0]
        self.assertEqual(assignment['job_id'], normal.job.name)
        ex.rpc('/v1/complete', self.completion(assignment))
        self.assertIsNone(ex.preparation_result(optional, wait=False))
        ex.finish()
        self.assertEqual(ex.store.row(optional)['state'], 'QUEUED')
        resumed = self.executor(capacity=1)
        self.assertEqual(self.submit(resumed), optional)
        assignment = self.poll_host(resumed, 'server2')['assignments'][0]
        self.assertEqual(assignment['job_id'], optional)
        self.assertEqual(sum(e['event'] == 'RESERVE' for e in resumed.store.events()), 2)  # one Q3 + one validation

    def test_running_validation_detaches_without_new_attempt_or_wait(self):
        ex = self.executor(capacity=1)
        optional = self.submit(ex)
        assignment = self.poll_host(ex, 'server1')['assignments'][0]
        ex.finish()  # No synthetic completion supplied; must return without waiting.
        before = ex.store.row(optional)
        self.assertEqual(before['state'], 'ASSIGNED')
        resumed = self.executor(capacity=1)
        replay = self.poll_host(resumed, 'server1')['assignments'][0]
        self.assertEqual(replay['attempt_id'], assignment['attempt_id'])
        self.assertEqual(self.poll_host(resumed, 'server2')['assignments'], [])
        self.assertEqual(sum(e['event'] == 'ASSIGN' for e in resumed.store.events()), 1)

    def test_failed_validation_is_retained_without_poisoning_q3(self):
        ex = self.executor(capacity=1)
        optional = self.submit(ex)
        assignment = self.poll_host(ex, 'server1')['assignments'][0]
        payload = self.prep_complete(ex, assignment)
        outcome = json.loads(base64.b64decode(payload['files']['OUTCOME.json']))
        outcome.update(status='PREPARATION_FAILED', artifacts={})
        payload['files']['OUTCOME.json'] = base64.b64encode(json.dumps(outcome).encode()).decode()
        payload['files']['PROCESS_RETURN.json'] = base64.b64encode(json.dumps(dict(returncode=1, signal=None)).encode()).decode()
        ex.rpc('/v1/complete', payload)
        self.assertEqual(ex.preparation_result(optional, wait=False)['status'], 'PREPARATION_FAILED')
        self.assertIsNone(ex.state.failure)
        normal = self.enqueue(ex, 'normal-after-validation-failure')
        self.assertEqual(self.poll_host(ex, 'server1')['assignments'][0]['job_id'], normal.job.name)

    def test_optional_flag_requires_compile_and_boolean(self):
        for stage, flag in [('signal', True), ('audit', True), ('compile', 'true')]:
            request = self.request()
            request['preparation'].update(stage=stage, optional_price_validation=flag)
            with self.assertRaises(ValueError):
                normalize_preparation_request(request)


if __name__ == '__main__':
    unittest.main()
