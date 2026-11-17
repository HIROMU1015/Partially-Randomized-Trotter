"""Synthetic Git/caller fixtures only; never import a scientific Q3 kernel."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_prdf_portable_adapter', ROOT / 'scripts/run_rpe_optimizer_q3_job.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.outer = Path(self.temporary.name) / 'outer'
        self.outer.mkdir()
        self.git('init', '-q')
        (self.outer / 'scripts').mkdir()
        self.entry = 'scripts/run_pr_df_fullh_total_q3_fixed.py'
        (self.outer / self.entry).write_text('from pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\n')
        self.inner_sha = self.commit('synthetic inner source')
        old = """from pathlib import Path
import importlib.util,sys
VIEW=Path('/SERVER_ONLY/NEVER_CREATE_THIS')
Q3_SOURCE=%r
sys.path[:0]=[str(VIEW/'src'),str(VIEW/'scripts')]
spec=importlib.util.spec_from_file_location('_split_fixed_q3_route',VIEW/'scripts/run_pr_df_fullh_total_q3_fixed.py')
U=importlib.util.module_from_spec(spec);sys.modules[spec.name]=U;spec.loader.exec_module(U)
""" % self.inner_sha
        (self.outer / 'scripts/run_pr_df_ld_split_q3.py').write_text(old)
        old_sha = self.commit('synthetic frozen caller')
        resume = """from pathlib import Path
import subprocess,types
ROOT=Path(__file__).resolve().parents[1]
OLD=%r
def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
Q=types.ModuleType('_synthetic_saved_caller');Q.__file__=str(ROOT/'scripts/run_pr_df_ld_split_q3.py')
exec(compile(blob(OLD,'scripts/run_pr_df_ld_split_q3.py'),Q.__file__+'@'+OLD,'exec'),Q.__dict__)
U=Q.U
""" % old_sha
        (self.outer / 'scripts/run_pr_df_ld_split_q3_resume.py').write_text(resume)
        self.commit('synthetic outer caller')
        self.inner = Path(self.temporary.name) / 'inner'
        subprocess.run(['git', 'clone', '-q', '--shared', str(self.outer), str(self.inner)], check=True)
        subprocess.run(['git', 'checkout', '-q', '--detach', self.inner_sha], cwd=self.inner, check=True)
        path_before = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), path_before))

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.outer, stderr=subprocess.DEVNULL, text=True).strip()

    def commit(self, message):
        self.git('add', '.')
        self.git('-c', 'user.name=Synthetic Test', '-c', 'user.email=synthetic@example.invalid',
                 'commit', '-q', '-m', message)
        return self.git('rev-parse', 'HEAD')

    def test_relocation_keeps_frozen_caller_and_inner_bytes(self):
        before = (self.outer / 'scripts/run_pr_df_ld_split_q3.py').read_bytes()
        entry = (self.inner / self.entry).read_bytes()
        module, source = A.load_prdf_route(self.outer, {self.inner_sha: str(self.inner)})
        self.assertEqual(module.Q.VIEW, self.inner)
        self.assertEqual(module.U.ROOT, self.inner)
        self.assertEqual(source['commit'], self.inner_sha)
        self.assertEqual(before, (self.outer / 'scripts/run_pr_df_ld_split_q3.py').read_bytes())
        self.assertEqual(entry, (self.inner / self.entry).read_bytes())
        self.assertFalse(source['frozen_sources_modified'])

    def test_wrong_checkout_and_dirty_source_are_rejected_before_import(self):
        with self.assertRaisesRegex(ValueError, 'inner SOURCE HEAD differs'):
            A.load_prdf_route(self.outer, {self.inner_sha: str(self.outer)})
        with (self.inner / self.entry).open('a') as stream:
            stream.write('# changed even though executable AST is unchanged\n')
        with self.assertRaisesRegex(ValueError, 'tracked changes'):
            A.load_prdf_route(self.outer, {self.inner_sha: str(self.inner)})


if __name__ == '__main__':
    unittest.main(verbosity=2)
