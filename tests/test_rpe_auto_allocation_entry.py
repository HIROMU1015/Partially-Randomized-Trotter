"""Real artificial normal-entry process acceptance; no H4/numerical DP."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import check_rpe_auto_allocation as A


class AutoAllocationNormalEntryTests(unittest.TestCase):
    def test_new_auto_round_omits_fixed_global_limit_and_keeps_optional_wsl(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / 'fixture'
            A.prepare(base, join_late=True)
            spec = json.loads((base / 'ROUND.json').read_text())
            self.assertNotIn('max_concurrent_jobs', spec['executor'])
            distributed = spec['executor']['distributed']
            self.assertNotIn('local0-wsl', distributed.get('required_hosts', []))
            self.assertEqual(distributed['auto_allocation']['priority_groups'],
                             [['server2', 'server1', 'local'], ['local0-wsl']])
            self.assertEqual(json.loads((base / 'PROFILE_local0-wsl.json').read_text())['slots'], 1)
            self.assertFalse(json.loads((base / 'HOST_ADMISSION.json').read_text())['hosts']['local0-wsl']['accepting'])


if __name__ == '__main__':unittest.main()
