"""Artificial temporary-repository checks; no scientific imports or execution."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('h4_saved_finalizer', Path(__file__).with_name('finalize_saved_run.py'))
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


class FinalizerGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h4-finalizer-test-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run(['git','init','-q',str(self.root)], check=True)
        (self.root/'baseline').write_text('artificial baseline only\n')
        self.g('add','baseline')
        self.g('-c','user.name=Artificial Test','-c','user.email=test@example.invalid',
            'commit','-q','-m','artificial fixture')
        self.head = self.g('rev-parse','HEAD')
        self.branch = self.g('branch','--show-current')
        self.prefixes = ['artifacts/h4', 'claims/one']

    def g(self, *args):
        return f.git(self.root, *args)

    def check_repo(self, **kwargs):
        f.check_repository(self.root,self.head,self.branch,self.prefixes,**kwargs)

    def test_expected_new_scope_allowed_without_mutating_head_or_index(self):
        path=self.root/'artifacts/h4';path.mkdir(parents=True);(path/'saved.json').write_text('{}\n')
        self.check_repo()
        self.assertEqual(self.g('rev-parse','HEAD'),self.head)
        self.assertEqual(self.g('diff','--cached','--name-only'),'')

    def test_changed_source_head_refused(self):
        with self.assertRaisesRegex(RuntimeError,'SOURCE_HEAD_CHANGED'):
            f.check_repository(self.root,'TEST_ONLY_NOT_A_SOURCE',self.branch,self.prefixes)

    def test_changed_branch_refused(self):
        with self.assertRaisesRegex(RuntimeError,'BRANCH_CHANGED'):
            f.check_repository(self.root,self.head,'another-branch',self.prefixes)

    def test_unrelated_tracked_edit_preserved_and_refused(self):
        (self.root/'baseline').write_text('other work remains\n')
        with self.assertRaisesRegex(RuntimeError,'TRACKED_WORKTREE_CHANGES'):
            self.check_repo()
        self.assertEqual((self.root/'baseline').read_text(),'other work remains\n')

    def test_unrelated_untracked_work_not_collected(self):
        (self.root/'other.txt').write_text('preserve me\n')
        with self.assertRaisesRegex(RuntimeError,'OUT_OF_SCOPE_UNTRACKED'):
            self.check_repo()
        self.assertEqual(self.g('diff','--cached','--name-only'),'')

    def test_preexisting_index_and_outside_prefix_refused_without_reset(self):
        (self.root/'other.txt').write_text('staged by another task\n');self.g('add','other.txt')
        with self.assertRaisesRegex(RuntimeError,'PREEXISTING_INDEX'):
            self.check_repo()
        with self.assertRaisesRegex(RuntimeError,'OUT_OF_SCOPE_INDEX'):
            self.check_repo(staged=True)
        self.assertEqual(self.g('diff','--cached','--name-only'),'other.txt')

    def test_waiting_never_runs_validation_or_commit_before_STOP(self):
        pack=self.root/'pack';pack.mkdir()
        stop={'status':'ARTIFICIAL_STOP_TEST_ONLY'}
        sleeps=[]
        def on_sleep(seconds):
            sleeps.append(seconds)
            (pack/'STOP.json').write_text(json.dumps(stop))
        with patch.object(f,'same_process',side_effect=[True,False]), \
             patch.object(f,'last_observation',return_value={'STOP':stop}), \
             patch.object(f.time,'sleep',side_effect=on_sleep), \
             patch.object(f.subprocess,'run') as process, contextlib.redirect_stdout(io.StringIO()):
            result=f.wait_for_existing_run(pack,123,999)
        self.assertEqual(sleeps,[15]);self.assertEqual(result,stop);process.assert_not_called()

    def test_missing_supervisor_without_STOP_is_not_scientific_retry(self):
        pack=self.root/'pack';pack.mkdir()
        with patch.object(f,'same_process',return_value=False), patch.object(f.subprocess,'run') as process:
            with self.assertRaisesRegex(RuntimeError,'SUPERVISOR_GONE_WITHOUT_STOP_NO_RETRY'):
                f.wait_for_existing_run(pack,123,999)
        process.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
