"""Additional profiles and WSL relay metadata; no adapter or job process."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_plan_prepare as P
spec = importlib.util.spec_from_file_location('_optional_relay', ROOT / 'scripts/run_rpe_q3_relay.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class OptionalPreparationTests(unittest.TestCase):
    def test_additional_profile_loaded_without_replacing_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'prepared/profiles').mkdir(parents=True)
            original = dict(host_id='existing', root='fixed-existing', slots=2)
            (base / 'prepared/profiles/existing.json').write_text(json.dumps(original))
            extra = dict(host_id='optional', root='fixed-optional', slots=1)
            (base / 'extra.json').write_text(json.dumps(extra))
            config = base / 'EVIDENCE.json'
            config.write_text(json.dumps(dict(additional_profiles={'optional': 'extra.json'})))
            _, _, profiles = P.host_profile_inputs({'output': str(base / 'prepared')}, base / 'out', ROOT, config)
            self.assertEqual(profiles, {'existing': original, 'optional': extra})
            extra['host_id'] = 'wrong'
            (base / 'extra.json').write_text(json.dumps(extra))
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                P.host_profile_inputs({'output': str(base / 'prepared')}, base / 'out', ROOT, config)
            config.write_text(json.dumps(dict(additional_profiles={'existing': original})))
            with self.assertRaisesRegex(ValueError, 'replace an existing host'):
                P.host_profile_inputs({'output': str(base / 'prepared')}, base / 'out', ROOT, config)

    def test_matrix_separates_request_pass_from_admission(self):
        policy = dict(jobs={'artificial-job': dict(method='ARTIFICIAL', request_sha256='fixed',
            hosts={'optional': dict(status='PASS'), 'existing': dict(status='UNCHECKED', reasons=['missing'])})})
        admission = dict(hosts={'optional': dict(accepting=False, optional=True, max_slots=1)})
        before = deepcopy(policy)
        cells = P.host_allocation_table(policy, admission)['rows'][0]['hosts']
        self.assertTrue(cells['optional']['can_execute_this_request'])
        self.assertFalse(cells['optional']['accepting_new_jobs'])
        self.assertFalse(cells['optional']['dispatchable_when_connected'])
        self.assertTrue(cells['existing']['accepting_new_jobs'])
        self.assertFalse(cells['existing']['can_execute_this_request'])
        self.assertEqual(policy, before)

    def test_wsl_mailbox_participation_off_keeps_recovery_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = base / 'relay.json'
            config.write_text(json.dumps(dict(state_directory=str(base / 'state'), controller={}, workers=[])))
            relay = R.Relay(config)
            flag = base / 'participation.json'
            flag.write_text(json.dumps(dict(accepting_new_jobs=False)))
            worker = dict(transport='wsl', distribution='Ubuntu', python='/transport/python3',
                root='/new/fixed/transport', profile='/checked/profile.json', participation_file=str(flag))
            calls = []
            def run(command, **kwargs):
                calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0, json.dumps(dict(slots=1, attempts=['received-job'])), '')
            with patch.object(R.subprocess, 'run', side_effect=run):
                status, _ = relay.remote(worker, 'status')
                self.assertEqual(status, dict(slots=0, attempts=['received-job']))
                # Pause only suppresses new poll slots. Readiness and result ack
                # still use the ordinary mailbox transport without launching here.
                relay.remote(worker, 'ack', {'job_id': 'received-job'})
                flag.write_text(json.dumps(dict(accepting_new_jobs=True)))
                self.assertEqual(relay.remote(worker, 'status')[0]['slots'], 1)
            self.assertEqual(calls[0][0][:6], ['wsl.exe', '-d', 'Ubuntu', '--exec', '/transport/python3', '-B'])
            self.assertIn('/new/fixed/transport/scripts/run_rpe_q3_worker.py', calls[0][0])
            self.assertIn('/checked/profile.json', calls[0][0])
            self.assertEqual(json.loads(calls[1][1]['input']), {'job_id': 'received-job'})


if __name__ == '__main__':
    unittest.main()
