"""Scheduler rendering and mocked crontab I/O; never installs a real schedule."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h4_sync_install', ROOT/'scripts/install_h4_catalog_sync.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class Schedule(unittest.TestCase):
    def test_install_replace_remove_preserve_unrelated_lines_and_are_idempotent(self):
        old = '# owned by someone else\n17 3 * * * /existing/task\n'
        block = M.job_block('/project space/C', '/environment/python', 10)
        added = M.replace_job(old, block)
        self.assertEqual(M.replace_job(added, block), added)
        self.assertEqual(M.replace_job(added, None), old)
        self.assertEqual(M.replace_job(old, None), old)
        self.assertIn("'/project space/C/scripts/run_h4_catalog_sync.sh'", block)
        self.assertIn('*/10 * * * * /bin/bash', block)
        self.assertIn('--python /environment/python', block)
        with self.assertRaises(ValueError):
            M.replace_job(added + block, block)

    def test_paths_are_shell_quoted_and_cron_percent_escaped(self):
        block = M.job_block('/project/space and $(false)%', "/python/'interpreter", 7)
        self.assertIn("'/project/space and $(false)\\%/scripts/run_h4_catalog_sync.sh'", block)
        self.assertIn("'/python/'\"'\"'interpreter'", block)
        for args in [('relative', '/python', 10), ('/project', 'relative', 10), ('/project', '/python', 0)]:
            with self.assertRaises(ValueError):
                M.job_block(*args)
        with self.assertRaises(ValueError):
            M.job_block('/project\nBAD', '/python')

    def test_explicit_install_dry_run_and_show_use_mocked_crontab_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'scripts').mkdir(); (root/'scripts/run_h4_catalog_sync.sh').write_text('# wrapper fixture\n')
            state = ['# unrelated entry\n* 4 * * * /other\n']; calls = []
            def run(cmd, **kwargs):
                calls.append((cmd, kwargs))
                if cmd == ['crontab', '-l']:
                    return SimpleNamespace(returncode=0, stdout=state[0], stderr='')
                self.assertEqual(cmd, ['crontab', '-'])
                state[0] = kwargs['input']
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            with redirect_stdout(io.StringIO()):
                M.main(['--dry-run','--worktree',str(root)], run=run)
            self.assertEqual(len(calls), 1)
            with redirect_stdout(io.StringIO()):
                M.main(['--install','--worktree',str(root)], run=run)
            self.assertTrue(state[0].startswith('# unrelated entry\n* 4 * * * /other\n'))
            count = len(calls)
            with redirect_stdout(io.StringIO()):
                M.main(['--install','--worktree',str(root)], run=run)
            self.assertEqual(len(calls), count + 1)  # repeat only reads
            with redirect_stdout(io.StringIO()) as out:
                M.main(['--show'], run=run)
            self.assertEqual(out.getvalue(), M.job_block(root))
            with redirect_stdout(io.StringIO()):
                M.main(['--remove'], run=run)
            self.assertEqual(state[0], '# unrelated entry\n* 4 * * * /other\n')

    def test_missing_crontab_is_distinct_from_read_failure(self):
        self.assertEqual(M.current_crontab(lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='no crontab for fixture')), '')
        with self.assertRaises(RuntimeError):
            M.current_crontab(lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='permission denied'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
