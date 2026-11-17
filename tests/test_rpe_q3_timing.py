"""Operational clocks and joins only; no molecular data, circuits or real Q3."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_q3_timing as T


class TimingTests(unittest.TestCase):
    def fixture(self, directory, worker_clock='worker:boot', worker_offset=1000000):
        run=Path(directory);job='job';attempt='attempt'
        def event(name,t,**kw):return dict(event=name,monotonic=t,clock_id='central:boot',invocation='run',**kw)
        rows=[event('ENQUEUE',0,job=job),
              event('QUEUE_WAIT_STATE',0,reasons={job:'ALL_ELIGIBLE_SLOTS_BUSY'}),
              event('QUEUE_WAIT_STATE',2,reasons={job:'DISPATCH_READINESS_OR_POLL_WAIT'}),
              event('LAUNCH',3,job=job,attempt_id=attempt,host_id='remote',method='PR_DF',configuration='config',queue_wait_seconds=3),
              event('DEPENDENCY_WAIT_START',4,job=job,method='PR_DF'),
              event('RESULT_RECEIVED',12,job=job,attempt_id=attempt,host_id='remote',method='PR_DF'),
              event('TRANSFER_COMPLETE',12.2,job=job,method='PR_DF',transfer_seconds=.2),
              event('DEPENDENCY_WAIT_END',12.3,job=job,method='PR_DF'),
              event('ADOPT',12.4,job=job,method='PR_DF',configuration='config'),
              event('SEARCH_ADOPT',12.5,job=job,attempt_id=attempt,host_id='remote',method='PR_DF',configuration='config')]
        (run/'PROGRESS.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
        folder=run/'jobs/PR_DF/job';folder.mkdir(parents=True)
        (folder/'LAUNCH.json').write_text(json.dumps(dict(attempt_id=attempt,host_id='remote')))
        body=[dict(event='Q3_BODY_START',monotonic=worker_offset+5,clock_id=worker_clock,call_id='body'),
              dict(event='Q3_BODY_END',monotonic=worker_offset+11,clock_id=worker_clock,call_id='body',completed=True)]
        (folder/'Q3_BODY_TIMING.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in body))
        return run

    def test_cross_host_skew_never_subtracted(self):
        with tempfile.TemporaryDirectory()as d:
            result=T.summarize(self.fixture(d));r=result['rows'][0]
            self.assertEqual(r['Q3_body_seconds'],6)
            self.assertEqual(r['B_assignment_to_Q3_body']['kind'],'BOUNDED')
            self.assertEqual(r['B_assignment_to_Q3_body']['upper_seconds'],3)
            self.assertEqual(r['C_Q3_body_to_controller_receive']['upper_seconds'],3)
            self.assertAlmostEqual(r['D_receive_to_search_adopt_seconds'],.5)
            self.assertAlmostEqual(result['D_while_optimizer_waiting_union_seconds'],.3)
            self.assertEqual(result['queue_wait_observed_union_by_reason']['ALL_ELIGIBLE_SLOTS_BUSY'],2)

    def test_same_host_direct_body_times(self):
        with tempfile.TemporaryDirectory()as d:
            r=T.summarize(self.fixture(d,worker_clock='central:boot',worker_offset=0))['rows'][0]
            self.assertEqual(r['B_assignment_to_Q3_body']['seconds'],2)
            self.assertEqual(r['C_Q3_body_to_controller_receive']['seconds'],1)

    def test_reboot_domains_cannot_pair(self):
        self.assertIsNone(T.delta(dict(clock_id='host:boot1',monotonic=2),dict(clock_id='host:boot2',monotonic=5)))

    def test_summed_wait_is_not_elapsed_union(self):
        self.assertEqual(T.union_seconds([('c',1,4),('c',2,5)]),4)
        self.assertEqual(T.union_seconds([('c1',1,4),('c2',2,5)]),6)

    def test_artifact_other_attempt_is_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            run=self.fixture(d)
            (run/'jobs/PR_DF/job/LAUNCH.json').write_text(json.dumps(dict(host_id='remote',attempt_id='different')))
            r=T.summarize(run)['rows'][0]
            self.assertFalse(r['artifact_attempt_matches']);self.assertIsNone(r['Q3_body_seconds'])

    def test_duplicate_transfer_uses_first_receipt(self):
        with tempfile.TemporaryDirectory()as d:
            run=self.fixture(d)
            with(run/'PROGRESS.jsonl').open('a')as f:
                f.write(json.dumps(dict(event='RESULT_RECEIVED',job='job',attempt_id='attempt',host_id='remote',method='PR_DF',clock_id='central:boot',monotonic=100,invocation='run'))+'\n')
            self.assertAlmostEqual(T.summarize(run)['rows'][0]['D_receive_to_search_adopt_seconds'],.5)

    def test_body_observation_keeps_value_and_exception(self):
        class Accelerator:
            def __call__(self,x):
                if x<0:raise ValueError('unchanged')
                return x*7
        original=Accelerator.__call__
        with tempfile.TemporaryDirectory()as d:
            with T.body_context(SimpleNamespace(Accelerator=Accelerator),d):
                self.assertEqual(Accelerator()(4),28)
                with self.assertRaisesRegex(ValueError,'unchanged'):Accelerator()(-1)
            events=T.read_lines(Path(d)/'Q3_BODY_TIMING.jsonl')
            self.assertEqual([r['event']for r in events],['Q3_BODY_START','Q3_BODY_END']*2)
            self.assertFalse(events[-1]['completed'])
        self.assertIs(Accelerator.__call__,original)

    def test_saved_three_hosts_have_missing_body_not_fake_compute(self):
        timeline=ROOT/'docs/research/rpe_distributed_q3/checks/REAL_THREE_HOST_TIMELINE.jsonl'
        r=T.summarize(ROOT,timeline=timeline)
        self.assertEqual(r['attempts'],18)
        self.assertEqual({h:v['count']for h,v in r['by_host'].items()},{'server2':11,'server1':2,'local':5})
        self.assertEqual(r['missing_counts']['Q3_body_seconds'],18)
        self.assertEqual(r['missing_counts']['D_receive_to_search_adopt_seconds'],0)
        self.assertEqual(r['queue_wait_state_observations'],0)
        self.assertTrue(all(v['receive_boundary']=='LEGACY_VALIDATED_PERSISTED_RESULT'for v in r['rows']))


if __name__=='__main__':unittest.main()
