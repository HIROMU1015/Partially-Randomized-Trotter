"""Live accounting invariants at fault time, not patched report arithmetic."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import os,tempfile,unittest
import sqlite3,threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import b3_storage_artificial_fixture as F
R=F.R


class Conn:
    def __init__(self):self.replies=[]
    def send(self,value):self.replies.append(value)


class Storage(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='b3-storage-unit-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.control=None
        self.addCleanup(self.close)
    def close(self):
        if self.control:R.close_storage(self.control)
    def create(self,fault=None,mode='P1'):
        tasks,n=R.P.plan_for(16)
        self.control=R.Coordinator(R.P.Settings(profile_n=16,codec='fused',mode=mode),self.root,tasks,n,'ARTIFICIAL_UNIT',fault or F.Fault())
        self.conn=Conn();self.task=tasks[0]
        self.control.active[self.conn]=dict(task=self.task,next=0,process=SimpleNamespace(pid=os.getpid()),report=None)
        self.control.log(dict(event='PHASE_BEGIN',phase=0))
        self.tid=f"correctness/baseline/{self.task['arm']}/{self.task['layer']}/X/0"
        return self.control
    def reserve(self):self.control.handle(self.conn,dict(op='RESERVE',trial_id=self.tid))
    def enter(self):self.control.handle(self.conn,dict(op='ENTER',trial_id=self.tid,worker_entry_ns=1))
    def finish(self):self.control.handle(self.conn,dict(op='COMPLETE',trial_id=self.tid))
    def coherent(self):
        c=self.control
        self.assertEqual(c.reservations,c.planned-c.states.counts['NOT_STARTED'])
        self.assertEqual(c.reservations,sum(c.next_indices.values()))
        self.assertEqual(c.reservations,c.active[self.conn]['next'])
        for channel in c.budget.limit:
            self.assertLessEqual(c.budget.observed[channel],c.budget.reserved[channel])
            self.assertLessEqual(c.budget.reserved[channel],c.budget.limit[channel])

    def test_fault_observes_counter_ticket_cursor_already_bound(self):
        observed=[]
        def fault(point,**ctx):
            if point=='trace.write'and ctx['op']=='RESERVE':
                c=self.control;self.coherent();self.assertIn(self.conn,c.current)
                observed.append(c.reservations);raise OSError('ARTIFICIAL_TRACE_RESERVE')
        c=self.create(fault);self.reserve();self.assertEqual(observed,[1]);self.coherent()
        self.assertEqual(c.states.counts['START_RESERVED'],1);self.assertEqual(c.started,0)
        self.assertFalse(self.conn.replies[-1]['ok']);self.enter()
        self.assertEqual(c.started,0);self.assertEqual(c.states.counts['CANCELLED_BEFORE_START'],1)

    def test_enter_record_failure_never_grants_entry(self):
        c=self.create(F.Fault('trace.write',{'op':'ENTER'}));self.reserve();self.enter()
        self.coherent();self.assertEqual(c.started,0);self.assertFalse(self.conn.replies[-1]['ok'])
        self.assertEqual(c.states.counts['CANCELLED_BEFORE_START'],1)

    def test_complete_record_failure_becomes_unknown_not_success(self):
        c=self.create(F.Fault('trace.write',{'op':'COMPLETE'}));self.reserve();self.enter();self.finish()
        self.coherent();self.assertEqual(c.states.counts['COMPLETED'],0)
        self.assertEqual(c.states.counts['TERMINATED_WITH_UNKNOWN_RESULT'],1);self.assertFalse(c.current)

    def test_state_pwrite_failure_keeps_nonrefundable_ram_intent(self):
        c=self.create(F.Fault('state.write',{'state':'START_RESERVED'}));self.reserve();self.coherent()
        self.assertEqual(c.reservations,1);self.assertEqual(c.started,0);self.assertTrue(c.states.errors)
        self.assertEqual((self.root/'trial_states.bin').read_bytes()[0],0)
        self.assertTrue(c.stopped.value);self.assertFalse(self.conn.replies[-1]['ok'])

    def test_state_after_pwrite_failure_is_explicit_even_if_file_matches(self):
        c=self.create(F.Fault('state.after_write',{'state':'START_RESERVED'}));self.reserve();self.coherent()
        self.assertEqual((self.root/'trial_states.bin').read_bytes()[0],1)
        self.assertTrue(c.states.errors);self.assertFalse(c.report()['evidence_v2']['statefile_complete'])

    def test_invalid_ack_does_not_corrupt_observed_counter(self):
        c=self.create();before=dict(c.budget.observed)
        with self.assertRaises(R.D.Stop):c.budget.wrote('result',1)
        self.assertEqual(dict(c.budget.observed),before)

    def test_trace_self_bytes_are_charged_once(self):
        c=self.create();self.reserve();self.enter();self.finish();self.coherent()
        physical=sum(x.stat().st_size for x in self.root.iterdir())
        self.assertEqual(c.budget.reserved['log'],physical)
        self.assertEqual(c.budget.observed['log'],physical)

    def test_coalesced_ack_failure_keeps_credit_and_prevents_next_reserve(self):
        fault=F.Fault('trace.write',{'op':'WRITE_ACK','channel':'result'});c=self.create(fault)
        self.reserve();self.enter();c.budget.reserve('result',17)
        next_id=self.tid[:-1]+'1'
        c.handle(self.conn,dict(op='RESERVE',trial_id=next_id,complete_before=self.tid,wrote_before=17))
        self.coherent();self.assertEqual(c.reservations,1);self.assertEqual(c.budget.observed['result'],17)
        self.assertEqual(c.states.counts['COMPLETED'],0);self.assertEqual(c.states.counts['TERMINATED_WITH_UNKNOWN_RESULT'],1)

    def test_stop_trace_and_control_failures_are_not_reported_saved(self):
        c=self.create(F.Fault('trace.write',{'op':'STOP'}));c.stop(dict(artificial=True))
        self.assertFalse(c.stop_trace_saved);self.assertTrue(c.stopped.value)
        self.reserve();self.assertEqual(c.reservations,0)

    def test_duplicate_reservation_and_completion_are_denied(self):
        c=self.create();self.reserve();self.enter();self.finish()
        self.finish();self.reserve();self.coherent()
        self.assertEqual(c.reservations,1);self.assertEqual(c.states.counts['COMPLETED'],1)
        self.assertTrue(c.stopped.value)

    def test_closed_science_entrypoint(self):
        for call in(R.production_entry,R.ProductionFactory(),R.ProductionBits()):
            with self.assertRaises(R.A.A.Denied):call(adopted=True,run_ready=True)

    def test_scientific_candidate_and_boolean_types_remain_fixed(self):
        from copy import deepcopy
        c=R.candidate(F.F.ROOT)
        for path,value in((('fixed','sigma'),True),(('proposal','profile_n'),True),(('proposal','total_calls'),1)):
            v=deepcopy(c);v[path[0]][path[1]]=value
            with self.assertRaises((R.A.A.Denied,R.D.Stop)):R.validate(v)

    def test_new_admission_spent_reuse_without_output_collision(self):
        with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')as t:
            root=Path(t);c,i,l,a=F.fixture(root,R.P.Settings(profile_n=16))
            p=R.ArtificialAdmission(c,i,root,l,a);digest=R.D.identity(i['candidate'])
            p.admit_artificial(digest);p.stop()
            self.assertFalse((root/'artificial-output').exists())
            with self.assertRaises(R.A.A.Denied):p.admit_artificial(digest)
            with self.assertRaises(sqlite3.IntegrityError):R.ArtificialAdmission(c,i,root,l,a).admit_artificial(digest)
            self.assertEqual(l.row(i['scope'])[4],'STOPPED')

    def test_new_admission_eight_contenders_same_store(self):
        with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')as t:
            root=Path(t);c,i,l,a=F.fixture(root,R.P.Settings(profile_n=16));barrier=threading.Barrier(8)
            def attempt(_):
                p=R.ArtificialAdmission(c,i,root,l,a);barrier.wait()
                try:p.admit_artificial(R.D.identity(i['candidate']));p.stop();return True
                except sqlite3.IntegrityError:return False
            with ThreadPoolExecutor(max_workers=8)as pool:outcomes=list(pool.map(attempt,range(8)))
            self.assertEqual(sum(outcomes),1);self.assertEqual(l.row(i['scope'])[4],'STOPPED')

    def test_cancel_state_persistence_failure_does_not_grant_entry(self):
        c=self.create(F.Fault('state.write',{'state':'CANCELLED_BEFORE_START'}))
        self.reserve();c.stop(dict(artificial=True));self.enter();self.coherent()
        self.assertEqual(c.started,0);self.assertEqual(c.states.counts['CANCELLED_BEFORE_START'],1)
        self.assertTrue(c.states.errors);self.assertFalse(self.conn.replies[-1]['ok'])

    def test_zero_byte_state_write_does_not_release_reservation(self):
        c=self.create()
        with patch.object(R.os,'pwrite',return_value=0):self.reserve()
        self.coherent();self.assertEqual(c.reservations,1);self.assertEqual(c.started,0)
        self.assertTrue(c.states.errors);self.assertFalse(self.conn.replies[-1]['ok'])

    def test_cleanup_does_not_attempt_stop_save_after_journal_close(self):
        c=self.create(F.Fault('trace.write',{'op':'RESERVE'}));self.reserve()
        self.assertTrue(c.stop_journal_saved);self.assertFalse(c.journal.failed)
        R.close_storage(c);self.control=None
        self.assertTrue(c.journal.file.closed);self.assertFalse(c.journal.failed)


def transaction_fault(op,point,mode):
    def test(self):
        c=self.create(F.Fault(point,{'op':op}),mode);self.reserve()
        if op in('ENTER','COMPLETE','FAIL'):self.enter()
        if op=='COMPLETE':self.finish()
        if op=='FAIL':c._terminal(self.conn,'FAIL')
        if op=='CANCEL':c._terminal(self.conn,'CANCEL')
        if op=='UNKNOWN':c._terminal(self.conn,'UNKNOWN')
        self.coherent();self.assertTrue(c.stopped.value);self.assertTrue(c.trace.failed)
        self.assertEqual(c.states.counts['COMPLETED'],0)
    return test


for mode in('S','P1'):
    for op in('RESERVE','ENTER','COMPLETE','FAIL','CANCEL','UNKNOWN'):
        for point in('trace.write','trace.flush'):
            setattr(Storage,'test_transaction_'+mode+'_'+op+'_'+point.replace('.','_'),transaction_fault(op,point,mode))


if __name__=='__main__':unittest.main()
