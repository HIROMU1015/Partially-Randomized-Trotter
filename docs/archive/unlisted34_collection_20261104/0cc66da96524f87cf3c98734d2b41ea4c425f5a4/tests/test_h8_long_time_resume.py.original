"""Artificial interrupted-cohort fixtures; no scientific runtime dispatch."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h8_long_time_adapter as a
import resume_h8_long_time_execution as resume


class RecoveryTests(unittest.TestCase):
    def test_original_seed_list_is_required_without_adding_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = a.candidate('PR', '2nd', 7, 8, 32, 4)
            config = dict(SOURCE='fixture', source_files={}, binding={})
            scope = dict(candidates=[row], n=2, cohort='EXPLORATION')
            expected = [int(a.digest(['H8_LONG_TIME_V1', 'fixture', {}, {}, 'EXPLORATION',
                row['id'], index, piece['outer'], piece['piece']])[:16], 16)
                for index in range(2) for piece in a.schedule(row)]
            a.save(root / 'seeds_PRIVATE.json', dict(cohort='EXPLORATION', values=expected))
            self.assertEqual(list(resume.restored_seeds(config, scope, root).values()), expected)
            (root / 'seeds_PRIVATE.json').write_text('{"cohort":"EXPLORATION","values":[]}')
            with self.assertRaisesRegex(ValueError, 'EXACT_ORIGINAL_SEEDS_ONLY'):
                resume.restored_seeds(config, scope, root)

    def test_completed_trials_reused_and_only_original_missing_trials_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = dict(id='fixture', method='PR')
            ep = dict(candidate=row)
            scope = dict(candidates=[row], n=2, cohort='EXPLORATION')
            for index in (0, 1):
                a.save(root / 'prices/fixture' / str(index) / 'events_PRIVATE.json', dict(events=[]))
            saved = dict(status='PASS', candidate_id='fixture', index=0, endpoint_digest=a.digest(ep),
                cohort='EXPLORATION', conditional=False, event_digest=a.digest([]),
                checks=[dict(phase_alignment=False, error=0., tolerance=1e-8) for _ in range(7)])
            a.save(root / 'prices/fixture/0/result.json', saved)
            completed, missing = resume.trial_plan(scope, {'fixture': ep}, root)
            self.assertEqual(completed, [saved])
            self.assertEqual([job[3] for job in missing], [1])
            a.save(root / 'HOLD.json', dict(reason='fixture'))
            with self.assertRaisesRegex(ValueError, 'INTERRUPTED_WITHOUT_HOLD_ONLY'):
                resume.trial_plan(scope, {'fixture': ep}, root)


if __name__ == '__main__':
    unittest.main()
