"""Pure mock pool/observer regressions; no live worker or scientific compilation."""
import unittest
from trottertracks.resource_applicability.h4_b0i_map.pool import GroupQueue,FrameReader
from trottertracks.resource_applicability.h4_b0i_map import observer
from trottertracks.resource_applicability.h4_b0i_map.memory_budget import required_available
from trottertracks.resource_applicability.h4_geometry.identity import Stop


def tasks():
    return [dict(geometry=g,prefix=p,template={'q':q},axis=a,task_id=f'{g}-{p}-{q}-{a}')
        for g in ('0.70','0.80','0.90','1.10','1.40','1.60') for p in (3,4,5,6,9)
        for q in (1,2,4,8) for a in ('cosine','sine')]


class ParallelTests(unittest.TestCase):
    def test_unordered_completion_full_coverage_group_cache(self):
        queue=GroupQueue(tasks());assigned={};done=[];largest=0
        for slot in range(12):assigned[slot]=queue.next(slot)
        while assigned:
            largest=max(largest,len(assigned));slot=max(assigned);task=assigned.pop(slot)
            group=(task['geometry'],task['prefix']);queue.complete(slot,task['task_id']);done.append(task['task_id'])
            nxt=queue.next(slot)
            if nxt:assigned[slot]=nxt
        self.assertEqual(len(set(done)),240);self.assertEqual(largest,12);self.assertTrue(queue.complete_map())
    def test_heavy_first_no_double_dispatch(self):
        q=GroupQueue(tasks());t=q.next(0);self.assertEqual((t['prefix'],t['template']['q']),(9,8))
        with self.assertRaises(Stop):q.next(0)
        with self.assertRaises(Stop):q.complete(0,'wrong task')
    def test_fragmented_reply_and_oversize_stop(self):
        r=FrameReader();self.assertIsNone(r.feed((3).to_bytes(8,'big')[:4]));self.assertIsNone(r.feed((3).to_bytes(8,'big')[4:]))
        self.assertIsNone(r.feed(b'a'));self.assertEqual(r.feed(b'bc'),b'abc')
        with self.assertRaises(Stop):r.feed((8193).to_bytes(8,'big'))
    def test_parallel_admission_and_observer_32GiB(self):
        self.assertEqual(required_available(12,32*2**30),int(456.25*2**30))
        state=observer.ObservationState({'oom_events':{'fixture':0}},0.,workers=12,worker_cap=32*2**30,driver_pid=1)
        for pid in range(2,14):state.own_worker(pid)
        memory=dict(observed_at=1.,available=64*2**30,psi_full_avg10=0.,oom_events={'fixture':0})
        samples=[dict(pid=1,rss=1,address_space=1)]+[dict(pid=pid,rss=31*2**30,address_space=31*2**30) for pid in range(2,14)]
        o=dict(rss=1,address_space=1);row=state.evaluate(memory,samples,o,.9,1.)
        self.assertIsNone(row['first_failure'])
        samples[-1]['rss']=33*2**30;row=state.evaluate({**memory,'observed_at':2.},samples,o,1.9,2.)
        self.assertEqual(row['first_failure']['reason'],'owned_role_rss_as')
    def test_shared_headroom_remains_stop(self):
        state=observer.ObservationState({'oom_events':{'fixture':0}},0.,workers=12,worker_cap=32*2**30,driver_pid=1)
        memory=dict(observed_at=1.,available=63*2**30,psi_full_avg10=0.,oom_events={'fixture':0})
        row=state.evaluate(memory,[dict(pid=1,rss=1,address_space=1)],dict(rss=1,address_space=1),.9,1.)
        self.assertEqual(row['first_failure']['reason'],'memory_headroom')


if __name__=='__main__':unittest.main()
