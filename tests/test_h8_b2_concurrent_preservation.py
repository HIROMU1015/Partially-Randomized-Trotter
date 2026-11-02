"""Concurrent metadata observation cannot waive source, evidence or frozen-tree checks."""
import hashlib,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import h8_phase_b2_recovery_runtime as r
import h8_phase_b2_recovery_contract as c

class Preservation(unittest.TestCase):
 def fixture(self):
  empty=hashlib.sha256(b'').hexdigest()
  return dict(protected_files=[],protected_worktrees=[dict(path='/frozen',HEAD='old',status_sha256=empty)],concurrent_worktree_observations=[dict(path='/other/h8-noncancellation-readiness-20261017',HEAD='old',status_sha256=empty)],baseline_worktree_count=2,provider_root='/provider')
 def git(self,path,*args):
  if args==('rev-parse','HEAD'):return b'new' if Path(path).name=='h8-noncancellation-readiness-20261017' else b'old'
  return b'?? new_file\0' if Path(path).name=='h8-noncancellation-readiness-20261017' else b''
 def test_concurrent_change_is_observed_not_reported_unchanged(self):
  cfg=self.fixture()
  with patch.object(r,'git_bytes',side_effect=self.git):v=r.protected(cfg)
  self.assertEqual(v['frozen_worktrees_unchanged'],1);row=v['concurrent_worktree_observations'][0];self.assertTrue(row['changed_since_baseline']);self.assertFalse(row['claimed_unchanged'])
 def test_unregistered_tree_cannot_be_exempted(self):
  cfg=self.fixture();cfg['concurrent_worktree_observations'][0]['path']='/other/h5'
  with self.assertRaisesRegex(ValueError,'ONLY_EXPLICIT'):r.protected(cfg)
 def test_source_or_provider_tree_can_never_be_exempted(self):
  for target in (c.ROOT,Path('/provider')):
   cfg=self.fixture();cfg['concurrent_worktree_observations'][0]['path']=str(target)
   with self.subTest(target=target),self.assertRaisesRegex(ValueError,'ONLY_EXPLICIT'):r.protected(cfg)
 def test_frozen_tree_change_still_STOPS(self):
  cfg=self.fixture()
  with patch.object(r,'git_bytes',return_value=b'changed'),self.assertRaisesRegex(ValueError,'PROTECTED_WORKTREE_CHANGED'):r.protected(cfg)
 def test_omitted_or_duplicate_baseline_rows_STOP(self):
  for duplicate in (False,True):
   cfg=self.fixture()
   if duplicate:cfg['concurrent_worktree_observations'][0]['path']='/frozen'
   else:cfg['baseline_worktree_count']=3
   with self.assertRaisesRegex(ValueError,'ALL_WORKTREE'):r.protected(cfg)
 def test_immutable_grant_or_result_change_still_STOPS(self):
  import tempfile
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'old_grant';p.write_bytes(b'old');cfg=self.fixture();cfg['protected_files']=[dict(path=str(p),bytes=3,sha256=hashlib.sha256(b'old').hexdigest())];p.write_bytes(b'new')
   with self.assertRaisesRegex(ValueError,'PROTECTED_FILE_CHANGED'):r.protected(cfg)

if __name__=='__main__':unittest.main()
