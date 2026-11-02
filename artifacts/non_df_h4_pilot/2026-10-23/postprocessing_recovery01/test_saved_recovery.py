"""Artificial recovery checks. No molecular imports, compile, sampling or jobs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('recovered_finalizer', HERE/'finalize_recovered_saved_run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class SavedRecoveryChecks(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='h4-saved-recovery-', dir='/tmp')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_wait_observes_existing_worker_without_scientific_dispatch(self):
        values = [dict(UTC='synthetic', worker=dict(active=active, reused=False),
                       cases={'df_native_PR':dict(joined=index)})
                  for index, active in enumerate([True, False])]
        with patch.object(r, 'observation', side_effect=values), \
             patch.object(r.time, 'sleep') as sleep, \
             patch.object(r.subprocess, 'run') as process, \
             contextlib.redirect_stdout(io.StringIO()):
            result = r.wait_for_worker(self.root, self.root/'observations.jsonl')
        self.assertFalse(result['worker']['active'])
        process.assert_not_called()
        sleep.assert_called_once_with(30)
        self.assertEqual(len((self.root/'observations.jsonl').read_text().splitlines()), 2)

    def test_pid_reuse_refused_without_wait_or_scientific_dispatch(self):
        value = dict(UTC='synthetic', worker=dict(active=False, reused=True),
                     cases={'df_native_PR':dict(joined=1)})
        with patch.object(r, 'observation', return_value=value), \
             patch.object(r.time, 'sleep') as sleep, \
             patch.object(r.subprocess, 'run') as process, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'WORKER_PID_REUSED'):
                r.wait_for_worker(self.root, self.root/'observations.jsonl')
        process.assert_not_called()
        sleep.assert_not_called()

    def test_missing_result_is_partial_and_never_native_completion(self):
        result = r.recovered_summary(self.root)
        self.assertEqual(result['missing_cases'], list(r.f.CASES))
        self.assertFalse(result['original_pipeline_completion_verified'])
        self.assertEqual(result['rows'], [])

    def test_complete_saved_rows_do_not_imply_native_completion(self):
        for case in r.f.CASES:
            folder = self.root/'run'/case
            folder.mkdir(parents=True)
            for name in ['result.json', 'validation.json']:
                (folder/name).write_text('{}')
            (folder/'status.json').write_text(json.dumps(dict(status='COMPLETE_FIXED_CASE')))
            (folder/'precision_rows.json').write_text(json.dumps([dict(epsilon_sig=.1)]))
        result = r.recovered_summary(self.root)
        self.assertEqual(result['status'], 'FOUR_CASES_COMPLETE')
        self.assertEqual(len(result['rows']), 4)
        self.assertFalse(result['original_pipeline_completion_verified'])
        self.assertFalse(result['scientific_superiority_claimed'])

    def test_scientific_verifier_checks_preserved_byte_for_byte(self):
        old = (HERE.parent/'verify_saved_run.py').read_text()
        new = (HERE/'verify_saved_recovery.py').read_text()
        old = old[old.index('def verify('):old.index("    for name in ['STOP.json','input_run/STOP.json','run/STOP.json']:")]
        new = new[new.index('def verify('):new.index('    # The original input stage')]
        old = old.replace("summary = read(PACK/'summary.json')",
                          "summary = read(BASE/'postprocessing_recovery01/recovered_case_summary.json')")
        self.assertEqual(old, new)

    def test_recovery_verifier_keeps_native_reaping_unverified(self):
        code = (HERE/'verify_saved_recovery.py').read_text()
        self.assertNotIn("status='PASS_SAVED_H4_FIXED_PILOT'", code)
        self.assertNotIn('all_children_reaped=True', code)
        self.assertIn("stop = read(PACK/'input_run/STOP.json')", code)
        self.assertIn('original_native_completion_verified=False', code)


if __name__ == '__main__':
    unittest.main(verbosity=2)
