"""Operational version checks precede the unchanged scientific mailbox."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from run_rpe_q3_relay import Relay
from test_rpe_optimizer_concurrent import fixture
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor


class ReleaseTransport(unittest.TestCase):
    def relay(self,okay=True):
        relay=Relay.__new__(Relay);relay.deployed_workers={};calls=[]
        def command(worker,argv,action,value):
            calls.append((list(argv),action,deepcopy(value)))
            return dict(status='PASS'if okay else'HOLD',ok=okay),0
        relay.remote_command=command
        worker=dict(host_id='server2',python='/PYTHON',root='/FROZEN_WORKER',profile='/PROFILE',
            deployment=dict(entry='/MANAGER',manifest='/MANIFEST',layout='/LAYOUT'))
        return relay,worker,calls

    def test_join_is_cached_but_each_assignment_is_checked_before_scientific_submit(self):
        relay,worker,calls=self.relay()
        relay.remote(worker,'status');relay.remote(worker,'status')
        self.assertEqual(sum(action=='deployment'for _,action,_ in calls),1)
        assignment={'request':{'request_id':'UNCHANGED_SCIENTIFIC_ID'},'required_release':{'id':'fixed'}}
        before=deepcopy(assignment);relay.remote(worker,'submit',assignment)
        self.assertEqual([c[1]for c in calls[-2:]],['deployment','submit'])
        self.assertIn('--assignment-stdin',calls[-2][0])
        self.assertEqual(assignment,before);self.assertEqual(calls[-1][2],before)

    def test_missing_or_wrong_release_never_reaches_worker_submit(self):
        relay,worker,calls=self.relay(False)
        with self.assertRaisesRegex(ValueError,'deployment HOLD'):relay.remote(worker,'submit',{'request':{}})
        self.assertEqual([c[1]for c in calls],['deployment'])

    def test_controller_stamps_release_outside_scientific_identity(self):
        with tempfile.TemporaryDirectory()as directory:
            problem,context,routes=fixture(['PR_PAULI'],1)
            reference={'id':'fixed-operational-release','manifest_sha256':'SYNTHETIC'}
            executor=DistributedExecutor(ROOT,directory,routes,{},problem,max_concurrent=1,
                budgets={'PR_PAULI':1},science_approved=True,distributed={'port':0,'required_release':reference})
            try:
                view=executor.for_method('PR_PAULI');candidate=context['candidates'][0]
                task=view._enqueue(dict(configuration_id=candidate['configuration_id'],vector=(1,)*26,cost=Fraction(1),origin={}),demand=True)
                row=executor.store.row(task.job.name);assignment=executor._assignment(row)
                self.assertEqual(assignment['required_release'],reference)
                self.assertEqual(assignment['request'],row['request'])
                self.assertEqual(assignment['request_sha256'],row['request_sha256'])
                self.assertNotIn('required_release',row['request'])
            finally:executor.finish()


if __name__=='__main__':unittest.main()
