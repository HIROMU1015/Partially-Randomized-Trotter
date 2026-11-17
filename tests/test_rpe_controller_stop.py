"""User stop cancels only unassigned artificial work and retains owned jobs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_rpe_optimizer_concurrent import fixture
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor


class StopTests(unittest.TestCase):
    def test_stop_keeps_owner_and_original_attempt_and_blocks_new_reservations(self):
        from fractions import Fraction
        with tempfile.TemporaryDirectory() as tmp:
            p,context,routes=fixture(['PR_DF'],1)
            run=Path(tmp)
            ex=DistributedExecutor(ROOT,run,routes,{},p,max_concurrent=2,
                budgets={'PR_DF':4},science_approved=True,distributed={'port':0})
            try:
                view=ex.for_method('PR_DF');cid=context['candidates'][0]['configuration_id']
                proposal=lambda n:dict(configuration_id=cid,vector=(n,)*26,cost=Fraction(n),origin={})
                task=view._enqueue(proposal(2),demand=True)
                assigned=ex.store.claim('server2',1)[0]
                # Restore ownership as a restarted controller would see it.
                ex.state.tasks.pop(task.job)
                ex._restore_task(assigned,proposal(2),True)
                queued=view._enqueue(proposal(1),demand=False)
                (run/'STOP_REQUESTED.json').write_text('{}')
                self.assertFalse(view._enqueue(proposal(3)))
                self.assertEqual(ex.store.row(queued.job.name)['state'],'CANCELLED')
                row=ex.store.row(assigned['job_id'])
                self.assertEqual((row['host_id'],row['attempt_id'],row['state']),('server2',assigned['attempt_id'],'ASSIGNED'))
                # Exact pending assignment is recoverable while stopping; no
                # new scientific ID can be assigned.
                reply=ex.rpc('/v1/poll',dict(host_id='server2',source_sha=ex.optimizer_source,slots=1))
                self.assertEqual([a['attempt_id']for a in reply['assignments']],[assigned['attempt_id']])
                ex.rpc('/v1/state',dict(host_id='server2',job_id=row['job_id'],attempt_id=row['attempt_id'],state='UNKNOWN',detail={}))
            finally:ex.finish()


if __name__=='__main__':unittest.main()
