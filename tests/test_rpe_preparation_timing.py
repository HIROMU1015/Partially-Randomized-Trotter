"""Operational timing separates typed preparation from Q3; no numerical work."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.rpe_q3_timing import summarize


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def lines(path,values):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(value)+'\n'for value in values))


class PreparationTimingTests(unittest.TestCase):
    def test_eight_q3_and_six_preparation_are_measured_separately(self):
        with tempfile.TemporaryDirectory()as temporary:
            run=Path(temporary);events=[]
            for i in range(14):
                prep=i>=8;stage=['signal']*4+['compile','audit']
                method='PREPARATION:'+stage[i-8]if prep else'FULL_PF'
                job=('prep-'if prep else'q3-')+str(i);attempt='attempt-'+str(i);host='host-'+str(i%4)
                common=dict(method=method,job=job,attempt_id=attempt,host_id=host,clock_id='controller',invocation='one')
                launch=dict(common,event='LAUNCH',monotonic=float(i+1),queue_wait_seconds=.25)
                events.extend([launch,dict(common,event='RESULT_RECEIVED',monotonic=float(i+3)),
                    dict(common,event='TRANSFER_COMPLETE',monotonic=float(i+3.1),transfer_seconds=.1),
                    dict(common,event='ADOPT',monotonic=float(i+3.2))])
                if not prep:events.append(dict(common,event='SEARCH_ADOPT',monotonic=float(i+3.3)))
                directory=run/'jobs'/method/job
                put(directory/'LAUNCH.json',dict(host_id=host,attempt_id=attempt))
                body=[dict(event='PREPARATION_BODY_START'if prep else'Q3_BODY_START',stage=stage[i-8]if prep else None,
                           clock_id='foreign-worker',monotonic=100.,call_id='one'),
                      dict(event='PREPARATION_BODY_END'if prep else'Q3_BODY_END',stage=stage[i-8]if prep else None,
                           clock_id='foreign-worker',monotonic=100.5,call_id='one',completed=True)]
                lines(directory/('JOB_STAGES.jsonl'if prep else'Q3_BODY_TIMING.jsonl'),body)
            # An enqueued preparation has no attempt and must not turn into a
            # Q3 attempt or erase unrelated controller queue snapshots.
            events.append(dict(method='PREPARATION:signal',event='ENQUEUE',job='prep-pending',monotonic=1.))
            lines(run/'PROGRESS.jsonl',events)
            result=summarize(run)
            self.assertEqual(result['attempts'],8)
            self.assertEqual(result['missing_counts']['Q3_body_seconds'],0)
            prep=result['preparation']
            self.assertEqual(prep['attempts'],6)
            self.assertEqual({k:v['count']for k,v in prep['by_stage'].items()},{'signal':4,'compile':1,'audit':1})
            self.assertEqual(prep['missing_body_count'],0)
            self.assertTrue(all(row['preparation_body_seconds']== .5 for row in prep['rows']))
            self.assertTrue(all(row['assignment_to_body_same_clock_seconds']is None for row in prep['rows']))
            self.assertTrue(all(not row['method'].startswith('PREPARATION:')for row in result['rows']))

    def test_saved_normal_entry_pipeline_log_when_locally_available(self):
        run=Path('/home/AbeHiromu/projects/rpe-distributed-preparation-checks/entry-final-1da8462e3/first/run')
        if not(run/'PROGRESS.jsonl').exists():self.skipTest('Saved local artificial pipeline not installed')
        result=summarize(run)
        self.assertEqual(result['attempts'],8)
        self.assertEqual(result['preparation']['attempts'],6)
        self.assertEqual(result['preparation']['missing_body_count'],0)
        self.assertEqual(set(result['preparation']['by_stage']),{'signal','compile','audit'})
        self.assertEqual(sum(host['count']for host in result['preparation']['by_host'].values()),6)


if __name__=='__main__':unittest.main()
