"""Parallel candidate: fake host/admission, CPU fixtures, no science execution."""
from collections import Counter
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import b3_parallel_artificial_fixture as F
P = F.P


def raced_host(root, c, ledger, auth, settings, send):
    try:
        F.invoke(root, c, ledger, auth, settings)
        send.send(True)
    except Exception:
        send.send(False)
    finally:
        send.close()


def crash_parent(root, c, ledger, auth, settings, watch):
    F.invoke(root, c, ledger, auth, settings, watch=watch)


class ParallelCandidate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, **kwargs):
        s = P.Settings(profile_n=16, workload='light', **kwargs)
        c, ledger, auth = F.fixture(self.root, s)
        self.c, self.ledger, self.auth, self.settings = c, ledger, auth, s
        return c, ledger, auth, s

    def invoke(self, **kw):
        return F.invoke(self.root, self.c, self.ledger, self.auth, self.settings, **kw)

    def assert_spent(self):
        self.assertIsNotNone(self.ledger.row(self.c['scope']))
        with self.assertRaises(Exception):
            self.invoke()

    def test_full_plan_science_budget_without_sampling(self):
        total = 256+2*sum(F.D.SAVED_N[c['arm']] for c in F.D.schedule())
        self.assertEqual(total, 31710848)
        self.assertLess(total, 1 << 28)
        self.assertEqual(8*sum(F.D.SAVED_N.values())+256, total)
        tasks,planned=P.plan_for(F.D.SAVED_N)
        self.assertEqual(planned,total)
        self.assertEqual([sum(2*t['n'] for t in tasks if t['phase_index']==i) for i in range(3)],
                         [256,15855296,15855296])

    def test_parent_dispatch_exception_reaps_workers(self):
        self.fixture(parent_fault='dispatch')
        before={p.pid for p in mp.active_children()}
        with self.assertRaises(P.BundleFailure) as caught:self.invoke()
        self.assertEqual({p.pid for p in mp.active_children()},before)
        self.assertFalse(caught.exception.report['complete'])
        self.assertEqual(len(caught.exception.report['pairs']),2)
        self.assert_spent()

    def test_event_acquired_then_invariant_failure(self):
        self.fixture(faults=((0,0,'after_event',0),))
        with self.assertRaises(P.BundleFailure) as caught:self.invoke()
        row=next(x for x in caught.exception.report['pairs'] if x['task']['pair_index']==0)
        f=row['failure']['trial_failure_provenance']
        self.assertTrue(f['current_attempt']['result_acquired'])
        self.assertTrue(f['current_attempt']['event_present'])
        self.assertIsNotNone(f['inflight_event'])
        self.assertEqual(sum(f['completed_prefix'].values()),0)

    def test_trial_state_write_failure_reaps_without_sampling(self):
        self.fixture()
        with patch.object(P.os,'pwrite',side_effect=OSError('ARTIFICIAL_STATE_WRITE')):
            with self.assertRaises(P.BundleFailure) as caught:self.invoke()
        self.assertEqual(caught.exception.report['started_observed'],0)
        self.assertEqual(caught.exception.report['states']['COMPLETED'],0)
        self.assert_spent()

    def test_1_2_4_8_workers_full_bundle(self):
        for workers in (1, 2, 4, 8):
            with self.subTest(workers=workers):
                r = F.run(P.Settings(workers=workers, profile_n=16, workload='light'))
                self.assertTrue(r['complete'])
                self.assertEqual(r['reserved'], 768)
                self.assertEqual(r['started_observed'], 768)
                self.assertEqual(r['states']['COMPLETED'], 768)
                self.assertEqual(len(r['pairs']), 24)
                self.assertEqual(r['fake_claim_status'], 'STOPPED')
                self.assertLessEqual(r['max_inflight'], workers)
                self.assertEqual(r['output']['observed'], r['output']['reserved'])

    def test_strict_start_caps(self):
        for cap in (1, 2, 4):
            r = F.run(P.Settings(mode='P2', workers=8, inflight_cap=cap, profile_n=16, workload='light'))
            self.assertTrue(r['complete'])
            self.assertLessEqual(r['max_inflight'], cap)

    def test_serial_parallel_canonical_event_bits_file_hash(self):
        baseline = None
        for mode, workers, cap in [('S', 1, 1), ('P1', 8, 8), ('P2', 8, 2)]:
            with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as tmp:
                root = Path(tmp)
                s = P.Settings(mode=mode, workers=workers, inflight_cap=cap, profile_n=20, workload='light')
                c, ledger, auth = F.fixture(root, s)
                r = F.invoke(root, c, ledger, auth, s)
                content = {}
                for row in r['pairs']:
                    parsed = F.read_stream(root/'artificial-output'/(row['task']['key']+'.jsonl'))
                    self.assertEqual(parsed['file_sha'], row['file_sha'])
                    self.assertEqual(parsed['chain'], row['chain'])
                    self.assertEqual(parsed['partial_bytes'], 0)
                    content[row['task']['key']] = (parsed['rows'], row['outcome'])
                if baseline is None:
                    baseline = content
                else:
                    self.assertEqual(content, baseline)

    def test_phase_barriers(self):
        self.fixture()
        self.invoke()
        events = [json.loads(x) for x in (self.root/'artificial-output/control.jsonl').read_text().splitlines()]
        ends = {x['phase']: i for i, x in enumerate(events) if x['event'] == 'PHASE_END'}
        begins = {x['phase']: i for i, x in enumerate(events) if x['event'] == 'PHASE_BEGIN'}
        self.assertLess(ends[0], begins[1]);self.assertLess(ends[1], begins[2])
        self.assertEqual(events[-1]['event'], 'COMPLETION_RECEIPT')

    def test_original_serial_scheduler_matches_candidate(self):
        reference = None
        for original in (True, False):
            with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as tmp:
                root = Path(tmp)
                s = P.Settings(mode='S', workers=1, profile_n=20, workload='light')
                r = F.run(s, root=root, original=original)
                content = {p.name: F.read_stream(p)['rows'] for p in
                           (root/'artificial-output').glob('*.jsonl') if p.name not in ('summary.jsonl','control.jsonl')}
                if original:
                    reference = content
                else:
                    self.assertEqual(content, reference)
                    self.assertEqual(r['states']['COMPLETED'], 896)

    def test_inflight_completion_and_post_stop_cost(self):
        for mode, cap in [('P1',8), ('P2',2)]:
            with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as tmp:
                root = Path(tmp)
                s = P.Settings(mode=mode, workers=8, inflight_cap=cap, profile_n=16, workload='light')
                c, ledger, auth = F.fixture(root, s)
                with self.assertRaises(P.BundleFailure) as caught:
                    F.invoke(root,c,ledger,auth,s,synchronized_failure=True)
                r=caught.exception.report
                self.assertEqual(r['reserved'],cap)
                self.assertEqual(r['started_observed'],cap)
                self.assertEqual(len(r['representative_failure']['inflight_snapshot']),cap)
                self.assertEqual(r['states']['COMPLETED'],cap-1)
                self.assertGreater(r['post_stop_parent_cpu_ns'],0)
                self.assertGreater(r['stop_to_join_ns'],0)
                self.assertEqual(sum(r['states'].values()),r['planned'])

    def test_stop_save_failure_preserves_primary_and_spent(self):
        self.fixture(parent_fault='stop_save',faults=((0,0,'sample',0),))
        with self.assertRaises(P.BundleFailure) as caught:self.invoke()
        self.assertIn('STOP_SAVE_FAILURE',caught.exception.report['secondary_stop_error'])
        self.assertTrue(caught.exception.report['failures'])
        self.assert_spent()

    def test_all_entered_workers_fail_and_all_reports_survive(self):
        self.fixture()
        with self.assertRaises(P.BundleFailure) as caught:
            self.invoke(synchronized_failure='all')
        r=caught.exception.report
        self.assertEqual(r['started_observed'],8)
        self.assertEqual(r['states']['FAILED'],8)
        self.assertEqual(sum(x['failure']['reason']=='ARTIFICIAL_SYNCHRONIZED_FAILURE' for x in r['pairs']),8)
        self.assertEqual(r['representative_failure']['parent_arrival'],1)

    def test_admission_stop_save_failure_keeps_primary(self):
        self.fixture(faults=((0,0,'sample',0),))
        def failure(point):
            if point in ('failure_record_save','stop_record_save'):
                raise OSError('ARTIFICIAL_ADMISSION_STOP_SAVE_FAILURE')
        with self.assertRaises(P.BundleFailure) as caught:
            self.invoke(admission_fault=failure)
        self.assertIn('MANDATORY_STOP_RECORD_FAILURE',caught.exception.b3_admission_stop_failure['reason'])
        self.assertIn(self.ledger.row(self.c['scope'])[4],('ADMITTED','RESERVED'))
        self.assert_spent()

    def test_collision_does_not_overwrite_existing_pair(self):
        self.fixture(collision=True)
        with self.assertRaises(P.BundleFailure) as caught:self.invoke()
        self.assertTrue(any(x.get('terminal_failure',{}).get('type')=='FileExistsError'
                            for x in caught.exception.report['failures']))
        paths=list((self.root/'artificial-output').glob('COLLISION.jsonl'))
        self.assertEqual(len(paths),1)
        F.read_stream(paths[0])
        self.assert_spent()

    def test_correctness_failure_never_profiles_or_adds_trials(self):
        self.fixture(faults=((0, 0, 'zero', 0),))
        with self.assertRaises(P.BundleFailure) as ctx:
            self.invoke()
        r = ctx.exception.report
        self.assertLessEqual(r['reserved'], 256)
        self.assertTrue(all(x['task']['phase_index'] == 0 for x in r['pairs']))
        self.assertNotIn('PHASE_BEGIN\",\"phase\":1', (self.root/'artificial-output/control.jsonl').read_text())
        self.assert_spent()

    def test_baseline_failure_never_instrumented(self):
        self.fixture(faults=((0, 1, 'sample', 1),))
        with self.assertRaises(P.BundleFailure) as ctx:self.invoke()
        self.assertTrue(all(x['task']['phase_index'] < 2 for x in ctx.exception.report['pairs']))

    def test_sample_failure_f01_has_no_stale_event(self):
        self.fixture(faults=((0, 0, 'sample', 3),))
        with self.assertRaises(P.BundleFailure) as ctx:self.invoke()
        row = next(x for x in ctx.exception.report['pairs'] if x['task']['pair_index'] == 0)
        f = row['failure']['trial_failure_provenance']
        self.assertEqual(sum(f['completed_prefix'].values()), 3)
        self.assertIsNone(f['inflight_event'])
        self.assertFalse(f['current_attempt']['result_acquired'])
        self.assertEqual(f['current_attempt']['trial_index'], 3)
        self.assertEqual(f['last_completed_trial']['trial_index'], 2)

    def test_multiple_worker_failures_retained(self):
        self.fixture(faults=tuple((i, 0, 'sample', 0) for i in range(8)))
        with self.assertRaises(P.BundleFailure) as ctx:self.invoke()
        r = ctx.exception.report
        self.assertGreaterEqual(len(r['failures']), 1)
        self.assertEqual(r['representative_failure'], r['failures'][0])
        arrivals = [x['parent_arrival'] for x in r['failures']]
        self.assertEqual(arrivals, sorted(set(arrivals)))
        self.assertFalse(r['complete'])

    def test_global_call_limit(self):
        self.fixture(call_limit=7)
        with self.assertRaises(P.BundleFailure) as ctx:self.invoke()
        self.assertLessEqual(ctx.exception.report['reserved'], 7)
        self.assertLessEqual(ctx.exception.report['started_observed'], 7)

    def test_global_result_limit(self):
        self.fixture(result_bytes=1500)
        with self.assertRaises(P.BundleFailure):self.invoke()
        files=list((self.root/'artificial-output').glob('*.jsonl'))
        total=sum(p.stat().st_size for p in files if p.name!='control.jsonl')
        self.assertLessEqual(total,1500)
        self.assertFalse((self.root/'artificial-output/summary.jsonl').exists())

    def test_global_log_limit_before_first_call(self):
        self.fixture(log_bytes=20)
        with self.assertRaises(Exception):self.invoke()
        self.assert_spent()

    def test_global_log_limit_after_workers_start(self):
        self.fixture(log_bytes=12000)
        with self.assertRaises(P.BundleFailure):self.invoke()
        path=self.root/'artificial-output'
        total=(path/'control.jsonl').stat().st_size+(path/'trial_states.bin').stat().st_size
        self.assertLessEqual(total,12000)
        self.assert_spent()

    def test_partial_write_keeps_tail_not_completion(self):
        self.fixture(faults=((0,0,'write',0),))
        with self.assertRaises(P.BundleFailure) as ctx:self.invoke()
        row=next(x for x in ctx.exception.report['pairs'] if x['task']['pair_index']==0)
        f=row['failure']['trial_failure_provenance']
        self.assertEqual(sum(f['completed_prefix'].values()),0)
        self.assertGreater(F.read_stream(self.root/'artificial-output'/(row['task']['key']+'.jsonl'))['partial_bytes'],0)

    def test_worker_unknown_and_no_retry(self):
        self.fixture(faults=((0,0,'exit',0),))
        with self.assertRaises(P.BundleFailure) as ctx:self.invoke()
        self.assertGreaterEqual(ctx.exception.report['states'].get('TERMINATED_WITH_UNKNOWN_RESULT',0),1)
        self.assertTrue(any(x['status']=='LOST' for x in ctx.exception.report['pairs']))
        self.assert_spent()

    def test_reserved_before_start_worker_crash(self):
        self.fixture(faults=((0,0,'reserved_exit',0),))
        with self.assertRaises(P.BundleFailure) as ctx:self.invoke()
        self.assertGreaterEqual(ctx.exception.report['states'].get('TERMINATED_WITH_UNKNOWN_RESULT',0),1)

    def test_parent_crash_keeps_spent_and_children_exit_on_eof(self):
        self.fixture(parent_fault='after_reserved')
        ctx=mp.get_context('fork');watch=ctx.RawArray('b',24)
        child=ctx.Process(target=crash_parent,args=(self.root,self.c,self.ledger,self.auth,self.settings,watch))
        child.start();child.join(20)
        self.assertFalse(child.is_alive());self.assertEqual(child.exitcode,86)
        until=time.monotonic()+10
        while sum(watch[:8])<8 and time.monotonic()<until:time.sleep(.01)
        self.assertEqual(sum(watch[:8]),8)
        state=(self.root/'artificial-output/trial_states.bin').read_bytes()
        self.assertGreaterEqual(state.count(bytes([P.CODE['START_RESERVED']])),1)
        self.assertIn(self.ledger.row(self.c['scope'])[4],('RESERVED','ADMITTED'))
        self.assert_spent()

    def test_output_before_failure_spends_claim(self):
        self.fixture(parent_fault='before_output')
        with self.assertRaises(P.Halt):self.invoke()
        self.assertFalse((self.root/'artificial-output').exists());self.assert_spent()

    def test_atomic_admission_commit_failure_spends_claim(self):
        self.fixture()
        def fail(point):
            if point=='after_reserve_commit':raise OSError('ARTIFICIAL_COMMIT_ACK_FAILURE')
        with self.assertRaises(OSError):self.invoke(admission_fault=fail)
        self.assertFalse((self.root/'artificial-output').exists());self.assert_spent()

    def test_same_scope_two_hosts_only_one_admitted(self):
        self.fixture()
        ctx=mp.get_context('fork');children=[];reads=[]
        for i in range(2):
            recv,send=ctx.Pipe(False)
            p=ctx.Process(target=raced_host,args=(self.root,self.c,self.ledger,self.auth,self.settings,send))
            p.start();send.close();children.append(p);reads.append(recv)
        values=[r.recv() for r in reads]
        for r in reads:r.close()
        for p in children:p.join()
        self.assertEqual(sum(values),1);self.assert_spent()

    def test_policy_change_denied_before_claim(self):
        self.fixture()
        (self.root/'parallel-policy.json').write_text('{}')
        with self.assertRaises(F.A.Denied):self.invoke()
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_no_real_port_or_authorization(self):
        with self.assertRaises(F.D.Stop):
            P.execute_artificial_bundle({},root=self.root,output=self.root/'out',approval_port=object(),
                settings=P.Settings(),endpoint_factory=F.endpoint_factory,bits_factory=F.bits_factory)

    def test_completion_receipt_only_after_all_workers(self):
        self.fixture()
        r=self.invoke()
        rows=[json.loads(x) for x in (self.root/'artificial-output/control.jsonl').read_text().splitlines()]
        self.assertEqual(sum(x['event']=='WORKER_END' for x in rows),24)
        self.assertEqual(rows[-1]['event'],'COMPLETION_RECEIPT')

    def test_no_science_modules_imported(self):
        self.assertFalse([x for x in sys.modules if x.rsplit('.',1)[-1] in P.FORBIDDEN])


def failure_case(stage):
    def test(self):
        pair=1 if stage=='angle' else 0
        self.fixture(faults=((pair,0,stage,0),))
        with self.assertRaises(P.BundleFailure) as ctx:self.invoke()
        self.assertFalse(ctx.exception.report['complete'])
        self.assert_spent()
    return test


for name in ['event','angle','flush']:
    setattr(ParallelCandidate,'test_fault_'+name,failure_case(name))


if __name__=='__main__':unittest.main()
