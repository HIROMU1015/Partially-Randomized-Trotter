"""Worker quota and actual filesystem reserve remain separate intake checks."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_worker_storage as S


class FilesystemIntake(unittest.TestCase):
    def test_low_space_stops_new_intake_and_recovers_after_cached_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            policy=dict(limit_bytes=1000,roots=[str(root/'future-cache')],sample_path=str(root/'sample.json'))
            with patch.object(S,'_used_bytes',return_value=64), \
                 patch.object(S.shutil,'disk_usage',return_value=SimpleNamespace(free=200)) as usage, \
                 patch.object(S.time,'time',return_value=100) as clock:
                normal=S.report(policy)
                self.assertTrue(normal['accepting_new_jobs'])
                self.assertIsNone(normal['reason'])
                self.assertEqual(normal['filesystem_min_free_bytes'],100)
                usage.assert_called_with(root)
                usage.return_value=SimpleNamespace(free=99)
                clock.return_value=106
                low=S.report(policy)
                self.assertFalse(low['accepting_new_jobs'])
                self.assertEqual(low['reason'],'FILESYSTEM_SPACE_LOW')
                self.assertEqual(low['used_bytes'],64)
                self.assertEqual(low['limit_bytes'],1000)
                self.assertEqual(low['filesystem_free_bytes'],99)
                self.assertEqual(low['filesystem_low_space_roots'],policy['roots'])
                usage.return_value=SimpleNamespace(free=150)
                clock.return_value=112
                recovered=S.report(policy)
                self.assertTrue(recovered['accepting_new_jobs'])
                self.assertIsNone(recovered['reason'])
                self.assertEqual(recovered['filesystem_free_bytes'],150)

    def test_existing_quota_still_blocks_with_adequate_filesystem_space(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            policy=dict(limit_bytes=1000,roots=[str(root)],sample_path=str(root/'sample.json'))
            with patch.object(S,'_used_bytes',return_value=1000), \
                 patch.object(S.shutil,'disk_usage',return_value=SimpleNamespace(free=500)):
                result=S.report(policy)
            self.assertFalse(result['accepting_new_jobs'])
            self.assertEqual(result['reason'],'STORAGE_LIMIT_REACHED')
            self.assertEqual(result['filesystem_free_bytes'],500)


if __name__=='__main__':unittest.main()
