"""Fixed input hashes follow the actual selected route, across repeated jobs."""
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
            self.assertEqual(input_hashes(ROOT, request, cache), first)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]['path'], 'AGENTS.md')


if __name__ == '__main__': unittest.main(verbosity=2)
