"""Synthetic transport/package checks only; never invoke WSL, SSH or science."""
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

P = Path(__file__).parent
S = importlib.util.spec_from_file_location('_wsl_delta', P / 'run_wsl_delta.py')
D = importlib.util.module_from_spec(S);S.loader.exec_module(D)


class WslDeltaTests(unittest.TestCase):
    def test_fixed_three_cells_and_guard_profiles(self):
        manifest, files = D.unpack((P / 'WSL_RUNTIME_DELTA_V1.zip').read_bytes())
        self.assertEqual(len(manifest['hosts']), 3)
        self.assertEqual({h['host_id'] for h in manifest['hosts']}, {'local0-wsl'})
        profile = json.loads(files['profiles/local0-wsl.json'])
        spec = importlib.util.spec_from_file_location('_guard_profiles', P.parents[4] / 'src/trottertracks/rpe_host_profiles.py')
        helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        for host in manifest['hosts']:
            assignment = json.loads(files[host['assignments'][0]])
            selected = helper.select_host_profile(profile, assignment)
            self.assertIn('/runtime-prdf-311rc1/e80a6533ce787e22/bin/python3', selected['interpreter'])
            self.assertEqual(selected['runtime_checks'][0]['view_commit'], assignment['request']['route']['view_commit'])
            self.assertEqual(assignment['source_manifest']['commit'], '28d976b99cdf8329adcc1b79327e329e947ef774')
            self.assertEqual(len(host['assignments']), 1)
        df = profile['route_profiles'][0]['profile']
        self.assertNotIn('runtime_checks', df)  # Preserve already proved dedicated DF profile.

    def test_wsl_transport_code_is_stdin_and_has_no_shell(self):
        host = dict(distribution='Ubuntu', transport_python='python3')
        payload = dict(action='host', fixture='ARTIFICIAL_TRANSPORT_ONLY')
        with patch.object(D, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=b'{"ok":true}', stderr=b'')) as run:
            self.assertEqual(D.wsl(host, payload), {'ok': True})
        args, kwargs = run.call_args
        self.assertEqual(args[0][:7], ['wsl.exe', '-d', 'Ubuntu', '--exec', 'python3', '-c', args[0][-1]])
        source, raw = kwargs['input'].split(b'\n', 1)
        self.assertEqual(base64.b64decode(source), (P / 'run_wsl_delta.py').read_bytes())
        self.assertEqual(json.loads(raw), payload)
        self.assertNotIn('shell', kwargs)

    def test_return_is_hash_verified_and_different_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            data = b'ARTIFICIAL_RESULT_ONLY\n'
            payload = dict(directory=root, manifest={'local0-wsl/PF/READINESS.json': D.sha(data)},
                           files={'local0-wsl/PF/READINESS.json': base64.b64encode(data).decode()})
            first = D.receive_action(payload)
            self.assertTrue(first['ok'])
            self.assertEqual(first, D.receive_action(payload))
            payload['files']['local0-wsl/PF/READINESS.json'] = base64.b64encode(b'CHANGED').decode()
            with self.assertRaisesRegex(ValueError, 'Transfer bytes differ'):
                D.receive_action(payload)
            self.assertEqual((Path(root) / 'local0-wsl/PF/READINESS.json').read_bytes(), data)

    def test_modified_bundle_rejected_before_any_transport(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(D, 'wsl', side_effect=AssertionError('No WSL permitted')), patch.object(D, 'ssh', side_effect=AssertionError('No SSH permitted')):
                with self.assertRaisesRegex(ValueError, 'bundle SHA256 differs'):
                    D.prepare(str(P / 'WSL_RUNTIME_DELTA_V1.zip'), '0' * 64, Path(root) / 'new')


if __name__ == '__main__':
    unittest.main()
