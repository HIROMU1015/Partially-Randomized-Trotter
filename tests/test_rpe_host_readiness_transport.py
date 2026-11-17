"""Current host identities cross direct-poll and local0 mailbox transports.

All challenges are artificial metadata. No adapter, job or SSH is launched.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_q3_worker as W
from trottertracks import rpe_host_eligibility as E

spec = importlib.util.spec_from_file_location('_host_readiness_relay', ROOT / 'scripts/run_rpe_q3_relay.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class ReadinessTransportTests(unittest.TestCase):
    def test_readiness_mailbox_reads_identity_without_opening_worker_spool(self):
        profile = dict(host_id='artificial-host')
        with patch.object(W, 'Worker', side_effect=AssertionError('No worker/spool for metadata')), \
             patch.object(E, 'current_host_fingerprint', return_value={'runtime': 'now'}):
            result = W.mailbox(profile, 'readiness', {'challenges': [{'job_id': 'artificial-job'}]})
        self.assertEqual(result['host_fingerprints'], {'artificial-job': {'runtime': 'now'}})
        self.assertEqual(result['readiness_errors'], {})

    def test_worker_answers_challenge_before_any_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = dict(host_id='artificial-host', root=str(ROOT), work_dir=temporary,
                           interpreter=sys.executable, slots=1, route_views={})
            calls = []
            def transport(endpoint, payload):
                self.assertEqual(endpoint, '/v1/poll')
                calls.append(payload)
                return dict(assignments=[], stop=True,
                            eligibility_challenges=[{'job_id': 'artificial-job'}] if len(calls) == 1 else [])
            with patch.object(E, 'current_host_fingerprint', return_value={'runtime': 'now'}):
                worker = W.Worker(profile, 'http://127.0.0.1:1', transport=transport)
                try:
                    worker.tick()
                finally:
                    worker.close()
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]['host_fingerprints'], {})
            self.assertEqual(calls[1]['host_fingerprints'], {'artificial-job': {'runtime': 'now'}})
            self.assertEqual(list(Path(temporary).glob('*/*/ASSIGNMENT.json')), [])

    def test_relay_forwards_current_fingerprint_and_never_submits_challenge(self):
        with tempfile.TemporaryDirectory() as temporary:
            configuration = Path(temporary) / 'config.json'
            configuration.write_text(json.dumps(dict(state_directory=str(Path(temporary) / 'relay'),
                controller=dict(local_port=1), workers=[])))
            relay = R.Relay(configuration)
            calls, remote_calls = [], []
            def remote(worker, action, value=None):
                remote_calls.append(action)
                if action == 'status':
                    return dict(host_id='artificial-host', slots=1, source_sha='artificial',
                                cached_prefixes=[], attempts=[]), 0
                self.assertEqual(action, 'readiness')
                self.assertEqual(value, {'challenges': [{'job_id': 'artificial-job'}]})
                return dict(host_id='artificial-host', host_fingerprints={'artificial-job': {'runtime': 'now'}},
                            readiness_errors={}), 0
            def post(endpoint, payload):
                self.assertEqual(endpoint, '/v1/poll')
                calls.append(payload.copy())
                return dict(assignments=[], eligibility_challenges=[{'job_id': 'artificial-job'}] if len(calls) == 1 else [])
            with patch.object(relay, 'remote', side_effect=remote), patch.object(relay, 'post', side_effect=post):
                relay.tick({'host_id': 'artificial-host'})
            self.assertEqual(remote_calls, ['status', 'readiness'])
            self.assertEqual(calls[1]['host_fingerprints'], {'artificial-job': {'runtime': 'now'}})


if __name__ == '__main__':
    unittest.main()
