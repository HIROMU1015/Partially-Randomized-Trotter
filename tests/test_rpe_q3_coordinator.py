"""Fixed input hashes follow the actual selected route, across repeated jobs."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.rpe_q3_coordinator import input_hashes


class FixedRouteInputs(unittest.TestCase):
    def test_unselected_route_does_not_require_foreign_input_and_cache_reuses_blob(self):
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        request = dict(configuration_id='chosen', route=dict(kind='pauli_b_common_entry',
            configurations=dict(chosen=dict(configuration=dict(commit=source, path='AGENTS.md')),
                                foreign=dict(configuration=dict(commit='MISSING_UNSELECTED_SOURCE', path='unused.json')))))
        cache = {}
        original = subprocess.check_output
        with patch('trottertracks.rpe_q3_coordinator.subprocess.check_output', wraps=original) as calls:
            first = input_hashes(ROOT, request, cache)
            first_calls = calls.call_count
            self.assertGreaterEqual(first_calls, 1)
            self.assertEqual(input_hashes(ROOT, request, cache), first)
        # Fixed Git object verification may require an initial lookup and
        # blob read; the shared input cache must avoid both on repetition.
        self.assertEqual(calls.call_count, first_calls)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]['path'], 'AGENTS.md')

    def test_selected_pauli_dictionary_and_residual_are_required_inputs(self):
        loc = lambda path: {'commit': 'fixed', 'path': path}
        config = {'spec': {'split': {'tail_dictionary': {
            'source': loc('dictionary.json'), 'P0_operator_residual_source': loc('result.json')}}}}
        files = {'config.json': json.dumps({'config': config}).encode(),
                 'dictionary.json': b'dictionary', 'result.json': b'residual'}
        configuration = dict(loc('config.json'), selector=['config'])
        request = dict(configuration_id='chosen', route=dict(kind='pauli_b_common_entry',
            configurations={'chosen': {'configuration': configuration}}))
        cache = {}
        with patch('trottertracks.rpe_q3_coordinator.subprocess.check_output',
                   side_effect=lambda cmd, cwd: files[cmd[-1].split(':', 1)[1]]) as git:
            first = input_hashes(ROOT, request, cache)
            self.assertEqual(input_hashes(ROOT, request, cache), first)
        self.assertEqual([row['path'] for row in first],
                         ['config.json', 'dictionary.json', 'result.json'])
        self.assertEqual(git.call_count, 4)


if __name__ == '__main__': unittest.main(verbosity=2)
