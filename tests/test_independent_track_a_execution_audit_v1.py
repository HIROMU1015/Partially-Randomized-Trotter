"""Independent failure-path audit. Fabricated observations and owned temp files only."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('independent_execution_fixtures',
    Path(__file__).with_name('final_pre_review_fixtures_v1.py'))
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
g = f.g


class IndependentExecutionAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='independent-execution-', dir=os.environ['TMPDIR'])
        self.base = Path(self.tmp.name); self.root = self.base/'source'; self.output = self.base/'output'
        self.contract = f.base_contract(f.driver.SCOPE, self.root, self.output, {'kind':'ARTIFICIAL'})
        self.auth = f.authorization(self.contract, self.output)
        self.grant = g.authorization_gate(self.contract, self.auth,
            contract_sha=f.d0.digest(self.contract), authorization_sha=f.d0.digest(self.auth),
            output=self.output, scope=f.driver.SCOPE, synthetic=True)

    def tearDown(self):
        self.tmp.cleanup()

    def memory(self):
        return dict(effective_available_bytes=10*2**30, host_full_psi_avg10=0., host_oom_kill=0,
            cgroups=[dict(path='/ARTIFICIAL/cgroup', full_psi_avg10=0.,
                events=dict(oom=0, oom_kill=0, oom_group_kill=0))])

    def filesystem(self, parent):
        return dict(available_bytes=2**30, available_inodes=100000, quota_enabled=False)

    def test_missing_cgroup_oom_counter_cannot_admit_or_continue(self):
        for key in ('oom', 'oom_kill'):
            with self.subTest(stage='admission', key=key):
                snapshot=self.memory(); del snapshot['cgroups'][0]['events'][key]
                with self.assertRaises(g.Stop):
                    g.Monitor(self.grant, memory=lambda:snapshot, filesystem=self.filesystem)
            with self.subTest(stage='continuation', key=key):
                monitor=g.Monitor(self.grant, memory=self.memory, filesystem=self.filesystem)
                snapshot=self.memory(); del snapshot['cgroups'][0]['events'][key]
                monitor.memory=lambda:snapshot
                with self.assertRaises(g.Stop): monitor.tick(force=True)

    def test_nonfinite_pressure_is_missing_evidence_not_zero(self):
        for key, value in [('host_full_psi_avg10', math.nan), ('host_full_psi_avg10', math.inf)]:
            snapshot=self.memory(); snapshot[key]=value
            with self.subTest(value=value), self.assertRaises(g.Stop):
                g.Monitor(self.grant, memory=lambda:snapshot, filesystem=self.filesystem)
        snapshot=self.memory(); snapshot['cgroups'][0]['full_psi_avg10']=math.nan
        with self.assertRaises(g.Stop):
            g.Monitor(self.grant, memory=lambda:snapshot, filesystem=self.filesystem)

    def test_quota_nonfinite_availability_never_passes(self):
        snapshot=dict(self.filesystem(self.base),quota_enabled=True,mount='/ARTIFICIAL',fsid=123)
        quota=dict(uid=os.getuid(),fsid=123,mount='/ARTIFICIAL',all_user_group_project_limits_checked=True,
            observed_monotonic=10.,available_bytes=2**30,available_inodes=100000)
        g.check_filesystem(snapshot,self.contract['budget'],quota,now=10.)
        for field in ('available_bytes','available_inodes'):
            for value in (math.nan, math.inf):
                changed=dict(quota); changed[field]=value
                with self.subTest(field=field,value=value), self.assertRaises(g.Stop):
                    g.check_filesystem(snapshot,self.contract['budget'],changed,now=10.)

    def test_observation_interval_includes_previous_io_duration(self):
        clock=[0.]
        monitor=g.Monitor(self.grant,memory=self.memory,filesystem=self.filesystem,clock=lambda:clock[0])
        def slow(): clock[0]+=4.; return self.memory()
        monitor.memory=slow
        clock[0]=.5; monitor.tick(force=True)  # observation begins .5, ends 4.5
        clock[0]=8.5  # four seconds of unavailable Python checkpoints after slow IO
        with self.assertRaisesRegex(g.Stop,'OBSERVATION_STALE'): monitor.tick(force=True)
        self.assertGreaterEqual(monitor.max_interval,8.)

    def test_quota_missing_stops_before_mapping_and_consumes_one_shot(self):
        fs=dict(self.filesystem(self.base),quota_enabled=True,mount='/ARTIFICIAL',fsid=123)
        factory=lambda grant:g.Monitor(grant,memory=self.memory,filesystem=lambda _:fs)
        loader=Mock(side_effect=AssertionError('UNAUTHORIZED INPUT ACCESS'))
        kwargs=dict(contract_sha=f.d0.digest(self.contract),authorization_sha=f.d0.digest(self.auth),
            mapping_loader=loader,output=self.output,root=self.root,synthetic=True,monitor_factory=factory)
        self.assertEqual(f.driver.run(self.contract,self.auth,**kwargs),2)
        loader.assert_not_called()
        terminal=json.loads((self.output/'terminal_v1.json').read_text())
        self.assertEqual(terminal['status'],'FAIL_CLOSED_STOP')
        self.assertIn('QUOTA_RECEIPT',terminal['first_stop']['reason'])
        with self.assertRaisesRegex(g.Stop,'ONE_SHOT_ALREADY_USED'):
            f.driver.run(self.contract,self.auth,**kwargs)
        loader.assert_not_called()

    def test_grant_binding_mutants_never_reach_mapping(self):
        loader=Mock()
        for field in ('profile','budget','binding','input_receipts','source_hashes','allowed_cpus','carry'):
            changed=copy.deepcopy(self.auth); changed[field+'_digest']='0'*64
            with self.subTest(field=field), self.assertRaises(g.Stop):
                f.driver.run(self.contract,changed,contract_sha=f.d0.digest(self.contract),
                    authorization_sha=f.d0.digest(changed),mapping_loader=loader,
                    output=self.output,root=self.root,synthetic=True,monitor_factory=f.FakeMonitor)
        loader.assert_not_called(); self.assertFalse(self.output.exists())

    def test_native_wait_parent_and_active_identity_are_not_exit_evidence(self):
        artifacts=g.Artifacts(self.grant); artifacts.close()
        kwargs=dict(contract_sha=f.d0.digest(self.contract),authorization_sha=f.d0.digest(self.auth),
            output=self.output,wait_status=0)
        with self.assertRaisesRegex(g.Stop,'STILL_PRESENT'):
            f.driver.record_external_exit(self.contract,self.auth,**kwargs)
        with patch.object(g,'identity',return_value=None):
            with self.assertRaisesRegex(g.Stop,'PARENT_OR_RUN_BINDING'):
                f.driver.record_external_exit(self.contract,self.auth,**kwargs)
        self.assertFalse((self.output/'native_exit_v1.json').exists())

    def test_sigkill_wait_is_nonzero_and_missing_terminal_is_explicit(self):
        artifacts=g.Artifacts(self.grant); ticket=artifacts.ticket; artifacts.close()
        def exited(pid=None):
            return ticket['parent_identity'] if pid==os.getppid() else None
        with patch.object(g,'identity',side_effect=exited):
            f.driver.record_external_exit(self.contract,self.auth,contract_sha=f.d0.digest(self.contract),
                authorization_sha=f.d0.digest(self.auth),output=self.output,wait_status=137)
        receipt=json.loads((self.output/'native_exit_v1.json').read_text())
        self.assertEqual(receipt['shell_wait_status'],137)
        self.assertFalse(receipt['terminal_available']); self.assertIsNone(receipt['exact_exit_timestamp'])

    def test_io_stop_keeps_first_reason_and_partial_output_is_not_receipted(self):
        artifacts=g.Artifacts(self.grant)
        monitor=g.Monitor(self.grant,memory=self.memory,filesystem=self.filesystem)
        try:
            with self.assertRaises(g.Stop):
                artifacts.write('partial.json',{'fixture':'ARTIFICIAL'},lambda:monitor.fail('FIRST_IO_STOP'))
            self.assertEqual(artifacts.receipts,[])
            with self.assertRaisesRegex(g.Stop,'FIRST_IO_STOP'): monitor.fail('LATER_CLEANUP')
            self.assertEqual(monitor.first_stop['reason'],'FIRST_IO_STOP')
        finally: artifacts.close(); monitor.close()


if __name__=='__main__': unittest.main()
