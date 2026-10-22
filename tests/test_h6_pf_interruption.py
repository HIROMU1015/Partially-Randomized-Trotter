"""Artificial accounting and launch checks; no molecular calculations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import audit_h6_pf_interruption as audit
import run_detached_local_job as detached


class RecoveryBudgetTests(unittest.TestCase):
    def test_attempted_calls_are_not_refunded(self):
        rows = [dict(system='H6', formula='4th(m5_best)', method='B0', prefix=9, q=8)]
        used = dict(native_evolution=224, native_piece=3625, independent_occupation=224,
            native_low=224, native_high=224, wrapper_build=400, uncompiled_action=400,
            wrapper_compile=400, compiled_action=400, sampling=0, new_signal=0, new_full_H_reference=0)
        caps = dict(native_evolution=224, native_signed_S2_pieces=3625,
            independent_occupation_actions=224, native_low_actions=224, native_high_actions=224,
            wrapper_build=448, uncompiled_wrapper_actions=448, wrapper_compile=448,
            compiled_wrapper_actions=448)
        result = audit.standard_fresh_budget(rows, used, caps)
        self.assertEqual(result['status'], 'UNAPPROVED_NOT_A_GRANT')
        self.assertTrue(result['exceeds_fixed_internal_budget'])
        piece = next(r for r in result['cumulative_comparison'] if r['kind'] == 'native_piece')
        self.assertEqual(piece['cumulative'], 3713)
        self.assertEqual(piece['excess'], 88)
        self.assertEqual(result['fresh_caps']['wrapper_compile'], 2)
        self.assertEqual(result['fresh_caps']['new_signal'], 0)

    def test_launcher_never_uses_shell_and_preserves_cpu_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'job'
            response = type('Result', (), dict(returncode=0, stdout='started', stderr=''))()
            with patch.object(detached.subprocess, 'run', return_value=response) as run:
                detached.launch(root, directory, [sys.executable, '-c', 'print("literal $()");'])
            args = run.call_args.args[0]
            self.assertIn('--user', args)
            self.assertIn('--collect', args)
            self.assertFalse(run.call_args.kwargs['shell'])
            request = json.loads((root/'request.json').read_text())
            self.assertEqual(request['command'][-1], 'print("literal $()");')
            self.assertEqual(request['CPU_affinity'], sorted(detached.os.sched_getaffinity(0)))
            self.assertFalse(request['scientific_authorization_added'])
            with self.assertRaises(FileExistsError):
                detached.launch(root, directory, [sys.executable, '-c', 'pass'])

    def test_service_launch_failure_stays_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'job'
            response = type('Result', (), dict(returncode=1, stdout='', stderr='no bus'))()
            with patch.object(detached.subprocess, 'run', return_value=response):
                with self.assertRaises(RuntimeError):
                    detached.launch(root, directory, [sys.executable, '-c', 'pass'])
            self.assertEqual(json.loads((root/'launch_receipt.json').read_text())['returncode'], 1)
            self.assertFalse((root/'worker_entered.json').exists())


if __name__ == '__main__':
    unittest.main()
