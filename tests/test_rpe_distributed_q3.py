"""Regression checks for artificial distributed schedule-entry acceptance."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import check_rpe_distributed_q3 as F


class ArtificialScheduleEntry(unittest.TestCase):
    def test_prepare_only_writes_artificial_metadata_and_host_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fixture'
            F.prepare(root, port=18765, local_slots=2, total_slots=4,
                      required_hosts=['server2', 'server1'])
            self.assertFalse((root / 'run' / 'jobs').exists())
            spec = json.loads((root / 'ROUND.json').read_text())
            self.assertTrue(spec['artificial_fixture']['only'])
            self.assertEqual(spec['executor']['distributed']['local_worker']['slots'], 2)
            self.assertEqual(spec['executor']['distributed']['required_hosts'], ['server2', 'server1'])
            profiles = [json.loads((root / ('PROFILE_' + host + '.json')).read_text())
                        for host in ('server2', 'server1', 'local')]
            self.assertEqual(len({p['work_dir'] for p in profiles}), 3)
            self.assertEqual(len({p['prefix_cache_directory'] for p in profiles}), 3)

    def test_ordinary_entry_late_ready_two_methods_exact_results_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'reference'
            F.prepare(root, total_slots=1, sleep_seconds=0)
            spec = json.loads((root / 'ROUND.json').read_text())
            spec['executor'].pop('distributed')
            spec['executor']['scheduler'] = 'legacy'
            F.write(root / 'ROUND.json', spec)
            with redirect_stdout(io.StringIO()):
                summary = F.controller(root)
            self.assertGreaterEqual(summary['outcomes'], 16)
            self.assertEqual(set(summary['per_method']), {'PR_DF', 'PR_PAULI'})
            self.assertTrue(all(item['best'] for item in summary['per_method'].values()))
            self.assertTrue((root / 'run' / 'ARTIFICIAL_PRICE_EVIDENCE.json').exists())
            before = F.launch_markers(root)
            outcomes = F.outcomes(root)
            with redirect_stdout(io.StringIO()):
                F.controller(root)
            self.assertEqual(before, F.launch_markers(root))
            self.assertEqual(outcomes, F.outcomes(root))


if __name__ == '__main__':
    unittest.main(verbosity=2)
