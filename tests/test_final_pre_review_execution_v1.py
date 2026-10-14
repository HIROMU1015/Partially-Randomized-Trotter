"""Resource/ownership/cleanup fixtures, never modifies system limits or jobs."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('guard_fixtures',Path(__file__).with_name('final_pre_review_fixtures_v1.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
g=f.g


class ExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='execution-guard-artificial-',dir=os.environ['TMPDIR'])
        self.root=Path(self.tmp.name);self.output=self.root/'out'
        self.contract=f.base_contract(f.driver.SCOPE,self.root/'source',self.output,{'test':'ARTIFICIAL'})
        self.auth=f.authorization(self.contract,self.output)
        self.grant=g.authorization_gate(self.contract,self.auth,contract_sha=f.d0.digest(self.contract),
            authorization_sha=f.d0.digest(self.auth),output=self.output,scope=f.driver.SCOPE,synthetic=True)
    def tearDown(self):self.tmp.cleanup()
    def memory(self):return dict(effective_available_bytes=10*2**30,host_full_psi_avg10=0.,host_oom_kill=0,cgroups=[])
    def filesystem(self,parent):return dict(available_bytes=2**30,available_inodes=100000,quota_enabled=False)

    def test_periodic_monitor_stale_and_first_stop_preserved(self):
        clock=[0.];monitor=g.Monitor(self.grant,memory=self.memory,filesystem=self.filesystem,clock=lambda:clock[0])
        monitor.phase='A_ORIGINAL_EQUALITY';clock[0]=6
        with self.assertRaisesRegex(g.Stop,'OBSERVATION_STALE'):monitor.tick(force=True)
        reason=copy.deepcopy(monitor.first_stop)
        with self.assertRaises(g.Stop):monitor.fail('later cleanup error')
        self.assertEqual(monitor.first_stop,reason)

    def test_pressure_and_oom_increment_fail_closed(self):
        monitor=g.Monitor(self.grant,memory=self.memory,filesystem=self.filesystem)
        monitor.memory=lambda:dict(self.memory(),host_full_psi_avg10=1.)
        with self.assertRaisesRegex(g.Stop,'MEMORY_PRESSURE'):monitor.tick(force=True)
        monitor=g.Monitor(self.grant,memory=self.memory,filesystem=self.filesystem)
        monitor.memory=lambda:dict(self.memory(),host_oom_kill=1)
        with self.assertRaisesRegex(g.Stop,'OOM_INCREMENT'):monitor.tick(force=True)

    def test_filesystem_block_and_inode_gate(self):
        for field,value in [('available_bytes',0),('available_inodes',0)]:
            snapshot=dict(self.filesystem(self.root));snapshot[field]=value
            with self.assertRaisesRegex(g.Stop,'HEADROOM'):g.check_filesystem(snapshot,self.contract['budget'])

    def test_enabled_quota_needs_fresh_effective_receipt(self):
        snapshot=dict(self.filesystem(self.root),quota_enabled=True,mount='ARTIFICIAL',fsid=123)
        with self.assertRaisesRegex(g.Stop,'QUOTA_RECEIPT'):g.check_filesystem(snapshot,self.contract['budget'],now=10)
        receipt=dict(uid=os.getuid(),fsid=123,mount='ARTIFICIAL',all_user_group_project_limits_checked=True,
            observed_monotonic=10,available_bytes=2**30,available_inodes=100000)
        g.check_filesystem(snapshot,self.contract['budget'],receipt,now=10)
        with self.assertRaises(g.Stop):g.check_filesystem(snapshot,self.contract['budget'],receipt,now=16)

    def test_output_cap_and_terminal_reserve_cleanup_idempotent(self):
        self.contract['budget']['output_bytes']=1;self.grant.binding_digest=f.d0.digest((self.contract,self.auth))
        artifacts=g.Artifacts(self.grant)
        with self.assertRaisesRegex(g.Stop,'OUTPUT_BYTES_CAP'):artifacts.write('too_large.json',{'large':True})
        artifacts.write('terminal.json',{'first_stop':'OUTPUT_BYTES_CAP'},terminal=True)
        artifacts.close();artifacts.close();self.assertIsNone(artifacts.dirfd)

    def test_terminal_inode_reserved_even_when_normal_cap_exhausted(self):
        self.contract['budget']['output_inode_cap']=0
        artifacts=g.Artifacts(self.grant)
        with self.assertRaises(g.Stop):artifacts.write('normal.json',{})
        artifacts.write('terminal.json',{},terminal=True);artifacts.close()

    def test_output_exclusive_existing_directory_not_modified(self):
        self.output.mkdir();(self.output/'old.txt').write_text('KEEP')
        with self.assertRaisesRegex(g.Stop,'ONE_SHOT_ALREADY_USED'):
            g.authorization_gate(self.contract,self.auth,contract_sha=f.d0.digest(self.contract),
                authorization_sha=f.d0.digest(self.auth),output=self.output,scope=f.driver.SCOPE,synthetic=True)
        self.assertEqual((self.output/'old.txt').read_text(),'KEEP')

    def test_symlink_input_rejected(self):
        target=self.root/'original.json';target.write_text('{}');link=self.root/'link.json';link.symlink_to(target)
        with self.assertRaises(g.Stop):g.read_bytes(link)

    def test_wall_monitor_and_missing_observation_io(self):
        self.contract['budget']['wall_seconds']=1;clock=[0.]
        monitor=g.Monitor(self.grant,memory=self.memory,filesystem=self.filesystem,clock=lambda:clock[0])
        clock[0]=2
        with self.assertRaisesRegex(g.Stop,'WALL_CAP'):monitor.tick(force=True)
        def missing():raise OSError('missing memory observation')
        monitor=g.Monitor(self.grant,memory=self.memory,filesystem=self.filesystem);monitor.memory=missing
        with self.assertRaisesRegex(g.Stop,'OSError'):monitor.tick(force=True)

    def test_delayed_observation_rejected(self):
        clock=[0.];monitor=g.Monitor(self.grant,memory=self.memory,filesystem=self.filesystem,clock=lambda:clock[0])
        def delayed():clock[0]+=6;return self.memory()
        monitor.memory=delayed
        with self.assertRaisesRegex(g.Stop,'IO_STALE'):monitor.tick(force=True)


if __name__=='__main__':unittest.main()
