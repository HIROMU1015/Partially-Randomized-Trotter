"""Transport-mocked checks only: no Git push, tag creation, or science runs."""
import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'retirement', ROOT / 'scripts/retire_archived_remote_branches.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class RetirementChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        (self.state / 'tags_verified.json').write_text('{"status": "PASS"}')
        self.branch = 'old-research'
        self.oid = '1' * 40
        self.archive = '2' * 40
        self.tag = 'refs/tags/' + driver.PREFIX + 'sample'
        self.ref = 'refs/heads/' + self.branch
        self.before = {self.ref: self.oid, self.tag: self.archive,
                       'refs/heads/main': '3' * 40}
        self.absent = {k: v for k, v in self.before.items() if k != self.ref}
        self.advanced = dict(self.before, **{self.ref: '4' * 40})
        self.plan = dict(keep_branches=['main'], archive_tags=[
            dict(tag=self.tag.removeprefix('refs/tags/'), commit=self.archive)],
            files_sha256={'branches.csv': hashlib.sha256(b'ledger').hexdigest()},
            driver_sha256=hashlib.sha256(b'driver').hexdigest())
        self.rows = [dict(old_branch=self.branch, initial_remote_sha=self.oid,
                          fixed_commit=self.oid, action='DELETE_ON_EXACT_SHA')]
        self.returncode = 0
        self.pushes = []
        self.start_patch(patch.object(driver, 'require_prepared'))
        self.start_patch(patch.object(driver, 'run', side_effect=self.git))
        self.start_patch(patch.object(driver.subprocess, 'check_output', side_effect=self.public_blob))
        self.start_patch(contextlib.redirect_stdout(io.StringIO()))

    def start_patch(self, manager):
        value = manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        return value

    def git(self, repo, *args, **kwargs):
        if args[0] == 'push':
            self.pushes.append(args)
            return subprocess.CompletedProcess(args, self.returncode, '', 'transport message')
        if args[0] == 'show':
            return subprocess.CompletedProcess(args, 0, 'branch_archive_20261102/README.md', '')
        if args[0] == 'fetch':
            return subprocess.CompletedProcess(args, 0, '', '')
        raise AssertionError(('Unexpected Git call', args))

    def public_blob(self, args):
        return b'driver' if args[-1].endswith('.py') else b'ledger'

    def retire(self, snapshots):
        with patch.object(driver, 'remote_refs', side_effect=snapshots):
            driver.retire(ROOT, self.state, self.plan, self.rows, [])
        return json.loads((self.state / 'execution.json').read_text())

    def test_success_requires_readback_and_uses_exact_lease(self):
        result = self.retire([self.before, self.before, self.absent, self.absent])
        self.assertEqual(result['deleted_count'], 1)
        self.assertEqual(result['retained_candidates'], 0)
        self.assertEqual(self.pushes, [(
            'push', '--atomic', '--porcelain',
            '--force-with-lease=' + self.ref + ':' + self.oid,
            'origin', ':' + self.ref)])

    def test_transport_error_after_server_deletion_is_confirmed_absent(self):
        self.returncode = 128
        result = self.retire([self.before, self.before, self.absent, self.absent])
        self.assertEqual(result['deleted_count'], 0)
        self.assertEqual(result['absent_after_uncertain_push'], 1)
        self.assertEqual(result['retained_candidates'], 0)
        self.assertEqual(result['results'][0]['status'], 'ABSENT_CONFIRMED_AFTER_PUSH_ERROR')

    def test_atomic_rejection_is_confirmed_present(self):
        self.returncode = 1
        result = self.retire([self.before] * 4)
        self.assertEqual(result['retained_candidates'], 1)
        self.assertEqual(result['results'][0]['status'], 'KEPT_ATOMIC_PUSH_REJECTED')

    def test_advanced_branch_is_not_pushed(self):
        result = self.retire([self.before, self.advanced, self.advanced])
        self.assertEqual(self.pushes, [])
        self.assertEqual(result['results'][0]['status'], 'KEPT_REMOTE_ADVANCED')

    def test_race_is_reported_from_actual_ref(self):
        self.returncode = 1
        result = self.retire([self.before, self.before, self.advanced, self.advanced])
        self.assertEqual(result['results'][0]['status'], 'KEPT_REMOTE_CHANGED_DURING_PUSH')
        self.assertEqual(result['results'][0]['final_sha'], '4' * 40)

    def test_failed_readback_is_durable_and_resume_does_not_repeat_push(self):
        with self.assertRaisesRegex(RuntimeError, 'offline'):
            self.retire([self.before, self.before, RuntimeError('offline')])
        progress = json.loads((self.state / 'deletion_progress.json').read_text())
        self.assertEqual(progress[0]['status'], 'UNCONFIRMED_AFTER_PUSH')
        result = self.retire([self.absent, self.absent])
        self.assertEqual(len(self.pushes), 1)
        self.assertEqual(result['absent_after_uncertain_push'], 1)

    def test_interrupted_push_is_reconciled_without_retry(self):
        with patch.object(driver, 'run', side_effect=lambda repo, *a, **kw: (
                (_ for _ in ()).throw(RuntimeError('interrupted')) if a[0] == 'push'
                else self.git(repo, *a, **kw))):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.retire([self.before, self.before])
        result = self.retire([self.before, self.before])
        self.assertEqual(self.pushes, [])
        self.assertEqual(result['results'][0]['status'], 'KEPT_AFTER_UNCONFIRMED_PUSH')

    def test_recreated_ref_is_reported_present_at_final_check(self):
        result = self.retire([self.before, self.before, self.absent, self.advanced])
        self.assertEqual(result['deleted_count'], 0)
        self.assertEqual(result['retained_candidates'], 1)
        self.assertEqual(result['results'][0]['status'], 'KEPT_REAPPEARED_AT_FINAL_CHECK')

    def test_public_driver_mismatch_blocks_push(self):
        with patch.object(driver.subprocess, 'check_output', return_value=b'ledger'):
            with self.assertRaisesRegex(ValueError, 'Public execution driver'):
                self.retire([self.before])
        self.assertEqual(self.pushes, [])

    def test_missing_archive_ref_blocks_push(self):
        with self.assertRaises(AssertionError):
            self.retire([{self.ref: self.oid, 'refs/heads/main': '3' * 40}])
        self.assertEqual(self.pushes, [])

    def test_main_is_never_a_delete_refspec(self):
        self.rows[0]['old_branch'] = 'main'
        with self.assertRaises(AssertionError):
            driver.deletion_arguments(self.rows, set())

    def test_unexpected_push_url_is_rejected_even_when_fetch_url_is_correct(self):
        allowed = 'https://github.com/' + driver.REPOSITORY + '.git\n'
        with patch.object(driver, 'run', side_effect=[
                subprocess.CompletedProcess([], 0, allowed, ''),
                subprocess.CompletedProcess([], 0, allowed + 'git@github.com:other/repo.git\n', '')]):
            with self.assertRaisesRegex(ValueError, 'fetch/push repository'):
                driver.verify_origin(ROOT)

    def test_checked_in_plan_matches_driver_and_excludes_two_review_holds(self):
        plan, rows = driver.load_plan()
        deletes = {r['old_branch'] for r in rows if r['action'].startswith('DELETE_')}
        self.assertEqual(len(deletes), plan['planned_deletions'])
        self.assertEqual(len(deletes), 352)
        for name in ['track-b-g6-return-generator-audit-20261010',
                     'codex/h46-long-time-b3-20261026']:
            self.assertIn(name, plan['keep_branches'])
            self.assertNotIn(name, deletes)

    def test_local_driver_mismatch_is_rejected(self):
        changed = self.state / 'changed-driver.py'
        changed.write_text('# changed\n')
        with patch.object(driver, 'DRIVER', changed):
            with self.assertRaisesRegex(ValueError, 'Execution driver differs'):
                driver.load_plan()


if __name__ == '__main__':
    unittest.main()
