"""Real v3 host/ledger/worker path, fake endpoints and authorities exclusively."""
from copy import deepcopy
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import tempfile
import time
from types import ModuleType
import unittest
from unittest.mock import patch

import b3_parallel_integrated_fixture as F
P,D,V,H,A=F.P,F.D,F.V,F.H,F.A


def crash_parent(root,c,ledger,auth,settings,watch):
    F.invoke(root,c,ledger,auth,settings,watch=watch)


def race_parent(root,c,ledger,auth,send):
    try:
        F.invoke(root,c,ledger,auth);send.send(True)
    except Exception:send.send(False)
    finally:send.close()


class Integration(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='b3-r1-artificial-');self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)

    def fixture(self,**kwargs):
        self.s=P.Settings(profile_n=16,workload='light',**kwargs)
        self.c,self.l,self.a=F.fixture(self.root,self.s)

    def invoke(self,**kw):return F.invoke(self.root,self.c,self.l,self.a,self.s,**kw)

    def spent(self):
        self.assertIsNotNone(self.l.row(self.c['scope']))
        with self.assertRaises(Exception):self.invoke()

    def test_static_full_schedule_unique_offsets(self):
        plan,total=P.plan_for(D.SAVED_N)
        self.assertEqual(total,31710848);self.assertEqual(len(plan),24)
        self.assertEqual(len({t['key']for t in plan}),24)
        self.assertEqual([sum(2*t['n']for t in plan if t['phase_index']==i)for i in range(3)],
                         [256,15855296,15855296])
        self.assertEqual([t['offset']for t in plan][1:],[t['offset']+2*t['n']for t in plan][:-1])

    def test_P_PF_serial_full_trial_bytes_events_bits_trace(self):
        reference=None
        for codec,mode,workers in [('reference','S',1),('reference','P1',8),('fused','P1',8)]:
            with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')as tmp:
                root=Path(tmp);r=F.run(P.Settings(profile_n=16,workload='light',codec=codec,mode=mode,workers=workers),root)
                content={p.name:F.read_stream(p)['rows']for p in (root/'artificial-output').glob('*.jsonl')
                         if p.name not in ('control.jsonl','summary.jsonl')}
                if reference is None:reference=content
                else:self.assertEqual(content,reference)
                self.assertEqual(r['reserved'],768);self.assertEqual(r['states']['COMPLETED'],768)
                self.assertEqual(r['output']['reserved'],r['output']['observed'])
                for pair in r['pairs']:
                    parsed=F.read_stream(root/'artificial-output'/(pair['task']['key']+'.jsonl'))
                    self.assertEqual(parsed['file_sha'],pair['file_sha']);self.assertEqual(parsed['chain'],pair['chain'])
                    for axis in pair['outcome']['axes']:self.assertEqual(axis['H_bits_supplied'],160*axis['N_executed'])

    def test_host_schema_v2_cannot_enable_parallel(self):
        self.fixture();c=deepcopy(self.c);c['schema']=A.HOST_SCHEMA
        with self.assertRaisesRegex(A.Denied,'PARALLEL_V3_CONTRACT_REQUIRED'):
            H.execute_artificial_v3(c,root=self.root,ledger=self.l,authority=self.a,
                endpoint_factory=F.ArtificialFactory(c['runtime_manifest']),bits_factory=F.bits_factory)
        self.assertIsNone(self.l.row(self.c['scope']))

    def test_production_without_independent_approval_cannot_import_science(self):
        self.fixture()
        with patch.object(D,'load_modules',side_effect=AssertionError('SCIENCE REACHED')):
            with self.assertRaisesRegex(A.Denied,'INDEPENDENT_SCIENCE_APPROVAL_REQUIRED'):
                H.execute_approved_v3(self.c,root=self.root,ledger=self.l)
        self.assertIsNone(self.l.row(self.c['scope']))

    def test_even_fake_approval_cannot_promote_technical_candidate(self):
        self.fixture()
        with patch.object(D,'load_modules',side_effect=AssertionError('SCIENCE REACHED')):
            with self.assertRaisesRegex(A.Denied,'TECHNICAL_CANDIDATE_NOT_SCIENCE_RELEASE'):
                H.execute_approved_v3(self.c,root=self.root,ledger=self.l,authority=self.a)
        self.assertIsNone(self.l.row(self.c['scope']))

    def test_candidate_edited_adoption_is_rejected(self):
        self.fixture();self.c['content_adopted']=True;self.c['run_ready']=True
        with self.assertRaisesRegex(A.Denied,'UNADOPTED_TECHNICAL_CANDIDATE_ONLY'):self.invoke()
        self.assertIsNone(self.l.row(self.c['scope']))

    def test_new_driver_rejects_old_direct_execute(self):
        with self.assertRaisesRegex(D.Stop,'PARALLEL_V3_HOST_ENTRYPOINT_REQUIRED'):D.execute({})

    def test_production_factory_direct_call_is_closed(self):
        self.fixture()
        factory=H.ProductionFactory(self.root,self.c['candidate'],self.c['runtime_manifest'])
        with patch.object(D,'load_modules',side_effect=AssertionError('SCIENCE REACHED')):
            with self.assertRaisesRegex(A.Denied,'TECHNICAL_CANDIDATE_NOT_SCIENCE_RELEASE'):
                factory(P.plan_for(16)[0][0],self.s)

    def test_production_bits_direct_call_is_closed(self):
        self.fixture();factory=H.ProductionBits(self.root,self.c['candidate'])
        with patch.object(D,'load_modules',side_effect=AssertionError('SCIENCE REACHED')):
            with self.assertRaisesRegex(A.Denied,'TECHNICAL_CANDIDATE_NOT_SCIENCE_RELEASE'):factory('ordinary','X')

    def test_direct_science_module_loader_is_closed(self):
        with self.assertRaisesRegex(D.Stop,'TECHNICAL_CANDIDATE_NOT_SCIENCE_RELEASE'):D.load_modules(self.root,{})

    def test_real_module_endpoint_cannot_bypass_host(self):
        permit=D.Boundary();permit.state='ENTERED'
        with self.assertRaisesRegex(D.Stop,'TECHNICAL_CANDIDATE_NOT_SCIENCE_RELEASE'):
            D.LegacyEndpoint({'g7':ModuleType('ARTIFICIAL_MODULE_SENTINEL')},'ordinary','A',A.A6,lambda:None,permit=permit)

    def test_bound_settings_edit_rejected_before_reservation(self):
        self.fixture();self.c['settings']['workers']=1
        with self.assertRaisesRegex(A.Denied,'APPROVAL_RELEASE_OR_CONTRACT_MISMATCH'):self.invoke()
        self.assertIsNone(self.l.row(self.c['scope']))

    def test_runtime_location_mismatch(self):
        self.fixture()
        with patch.object(D,'__file__','/tmp/wrong-driver.py'):
            with self.assertRaisesRegex(A.Denied,'RUNTIME_IMPORT_LOCATION'):self.invoke()
        self.assertIsNone(self.l.row(self.c['scope']))

    def test_missing_codec_pin(self):
        self.fixture();del self.c['source_pins'][V.ROLES['serialization']]
        with self.assertRaises(A.Denied):self.invoke()
        self.assertIsNone(self.l.row(self.c['scope']))

    def test_python_runtime_mismatch(self):
        self.fixture();self.c['runtime_manifest']['python']['cache_tag']='wrong'
        with self.assertRaisesRegex(A.Denied,'RUNTIME_PYTHON_MISMATCH'):self.invoke()

    def test_unknown_source_role_rejected(self):
        self.fixture();self.c['runtime_manifest']['source_roles']['unreviewed']='other.py'
        with self.assertRaisesRegex(A.Denied,'RUNTIME_ROLE_COVERAGE'):self.invoke()

    def test_input_change_after_reservation_stays_spent(self):
        self.fixture()
        def fault(point):
            if point=='before_binding_recheck':(self.root/'artificial-fixture.py').write_bytes(b'changed')
        with self.assertRaises(A.Denied):self.invoke(admission_fault=fault)
        self.spent()

    def test_reservation_commit_ack_uncertainty_stays_spent(self):
        self.fixture()
        def fault(point):
            if point=='after_reserve_commit':raise OSError('COMMIT_ACK_UNKNOWN')
        with self.assertRaisesRegex(OSError,'COMMIT_ACK_UNKNOWN'):self.invoke(admission_fault=fault)
        self.spent()

    def test_normal_completion_scope_reuse_resume_forbidden(self):
        self.fixture();self.assertTrue(self.invoke()['complete']);self.spent()
        self.c['resume']=True
        with self.assertRaises(Exception):self.invoke()

    def test_atomic_global_scope_race(self):
        self.fixture();ctx=mp.get_context('spawn');children=[];pipes=[]
        for _ in range(2):
            recv,send=ctx.Pipe(False);p=ctx.Process(target=race_parent,args=(self.root,self.c,self.l,self.a,send))
            p.start();send.close();children.append(p);pipes.append(recv)
        results=[p.recv()for p in pipes]
        for p in children:p.join(60);self.assertFalse(p.is_alive())
        for p in pipes:p.close()
        self.assertEqual(sum(results),1);self.spent()

    def test_spawn_payload_excludes_authority_and_ledger(self):
        self.fixture();payload=pickle.dumps(F.ArtificialFactory(self.c['runtime_manifest']))
        self.assertNotIn(str(self.l.directory).encode(),payload)
        self.assertNotIn(self.a.anchor.authorization_reference.encode(),payload)
        self.assertNotIn(self.c['claim_id'].encode(),payload)
        self.assertEqual(set(vars(F.ArtificialFactory(self.c['runtime_manifest']))),{'runtime_manifest'})

    def test_spawn_does_not_inherit_parent_secret_canary(self):
        self.fixture()
        with patch.object(F,'PARENT_CANARY',object()):self.assertTrue(self.invoke()['complete'])

    def test_schema_copy_change_rejected(self):
        self.fixture();(self.root/V.SCHEMA_PATHS[0]).write_bytes(b'changed schema')
        with self.assertRaisesRegex(A.Denied,'INPUT_PINS_MISMATCH'):self.invoke()
        self.assertIsNone(self.l.row(self.c['scope']))

    def test_missing_schema_pin_rejected(self):
        self.fixture();del self.c['runtime_manifest']['schema_pins'][V.SCHEMA_PATHS[0]]
        with self.assertRaises(A.Denied):self.invoke()
        self.assertIsNone(self.l.row(self.c['scope']))

    def test_nested_parent_execution_rejected_and_codec_restored(self):
        self.fixture(codec='fused');before=D.SERIALIZATION_MODE
        with H._execution_context(self.c):
            self.assertEqual(D.SERIALIZATION_MODE,'fused')
            with self.assertRaisesRegex(A.Denied,'PARENT_EXECUTION_ALREADY_ACTIVE'):self.invoke()
        self.assertEqual(D.SERIALIZATION_MODE,before)
        self.assertIsNone(self.l.row(self.c['scope']))

    def test_initial_log_cap_failure_is_incomplete_but_spent(self):
        self.fixture(log_bytes=10)
        with self.assertRaisesRegex(D.Stop,'GLOBAL_LOG_BYTE_LIMIT'):self.invoke()
        self.assertEqual(self.l.row(self.c['scope'])[4],'STOPPED');self.spent()

    def test_outer_admission_stop_failure_preserves_failure_and_spent(self):
        self.fixture(faults=((0,0,'sample',0),))
        def fault(point):
            if point=='stop_record_save':raise OSError('STOP_RECORD_UNAVAILABLE')
        with self.assertRaises(P.BundleFailure)as caught:self.invoke(admission_fault=fault)
        self.assertIn('b3_admission_stop_failure',vars(caught.exception));self.spent()

    def test_phase_barriers_and_summary_receipt_join(self):
        self.fixture(codec='fused');r=self.invoke()
        events=[json.loads(x)for x in (self.root/'artificial-output/control.jsonl').read_text().splitlines()]
        end={x['phase']:i for i,x in enumerate(events)if x['event']=='PHASE_END'}
        begin={x['phase']:i for i,x in enumerate(events)if x['event']=='PHASE_BEGIN'}
        self.assertLess(end[0],begin[1]);self.assertLess(end[1],begin[2])
        final=events[-1];summary=F.read_stream(self.root/'artificial-output/summary.jsonl')
        self.assertEqual(final['summary_sha'],summary['file_sha']);self.assertEqual(final['summary_chain'],summary['chain'])
        self.assertEqual(final['formal_digest'],D.identity(self.c));self.assertTrue(final['all_workers_joined'])
        self.assertEqual(r['fake_claim_status'],'STOPPED')

    def test_cpu_partitions_and_codec_observer(self):
        self.fixture(codec='fused');r=self.invoke()
        self.assertEqual(r['total_cpu_ns'],r['parent_cpu_ns']+r['worker_cpu_ns'])
        for pair in r['pairs']:
            for key in ('reconciliation','fine_reconciliation'):
                rec=pair.get(key)
                if rec:
                    self.assertIsNone(rec['unknown_bit_operations'])
                    self.assertTrue(all(x>=0 for x in rec['unattributed_and_observer_residual_ns']))
        self.assertTrue(any('exact_serialization.py' in name for pair in r['pairs']for name in (pair.get('fine')or{})))
        self.assertEqual(r['measurement_condition'],'concurrent_pairs')

    def test_parent_crash_leaves_spent_unknown_and_children_exit(self):
        self.fixture(parent_fault='after_reserved');ctx=mp.get_context('spawn');watch=ctx.RawArray('b',24)
        p=ctx.Process(target=crash_parent,args=(self.root,self.c,self.l,self.a,self.s,watch));p.start();p.join(60)
        self.assertEqual(p.exitcode,86)
        until=time.monotonic()+30
        while sum(watch)<8 and time.monotonic()<until:time.sleep(.05)
        self.assertEqual(sum(watch),8)
        states=(self.root/'artificial-output/trial_states.bin').read_bytes()
        self.assertEqual(states.count(P.CODE['START_RESERVED']),1)
        self.assertEqual(self.l.row(self.c['scope'])[4],'ADMITTED');self.spent()


FAULTS=dict(sample=dict(faults=((0,0,'sample',0),)),event=dict(faults=((0,0,'event',0),)),
    after_event=dict(faults=((0,0,'after_event',0),)),angle=dict(faults=((1,0,'angle',0),)),
    partial_write=dict(faults=((0,0,'write',0),)),flush_in_emit=dict(faults=((0,0,'flush',0),),flush_rows=1),
    worker_exit=dict(faults=((0,0,'exit',0),)),reserved_exit=dict(faults=((0,0,'reserved_exit',0),)),
    unacked_exit=dict(faults=((0,0,'ack_exit',0),)),result_cap=dict(result_bytes=100),log_cap=dict(log_bytes=12000),
    parent_dispatch=dict(parent_fault='dispatch'),stop_save=dict(parent_fault='stop_save',faults=((0,0,'sample',0),)),
    P1_inflight=dict(sync_failure='one'),P2_inflight=dict(mode='P2',inflight_cap=2,sync_failure='one'),
    simultaneous=dict(sync_failure='all'),call_limit=dict(workers=1,call_limit=3))


def fault_test(codec,name,settings):
    def test(self):
        self.fixture(codec=codec,**settings);before={p.pid for p in mp.active_children()}
        with self.assertRaises(P.BundleFailure)as caught:self.invoke()
        r=caught.exception.report;self.assertFalse(r['complete']);self.assertTrue(r['mandatory_STOP'])
        self.assertEqual(before,{p.pid for p in mp.active_children()});self.spent()
        self.assertEqual(sum(r['states'].values()),r['planned'])
        if name in ('worker_exit','reserved_exit','unacked_exit'):
            self.assertGreaterEqual(r['states']['TERMINATED_WITH_UNKNOWN_RESULT'],1)
        if name=='unacked_exit':
            raw=self.root/'artificial-output/correctness_baseline_ordinary_A.jsonl'
            self.assertEqual(len(F.read_stream(raw)['rows']),1)
            self.assertGreaterEqual(r['output']['reserved']['result']-r['output']['observed'].get('result',0),raw.stat().st_size)
        if name=='flush_in_emit':
            pair=next(x for x in r['pairs']if x['task']['pair_index']==0)
            f=pair['failure']['trial_failure_provenance'];self.assertFalse(f['current_attempt']['stream_emit_returned'])
            self.assertEqual(sum(f['completed_prefix'].values()),0)
        if name in ('P1_inflight','P2_inflight','simultaneous'):
            cap=2 if name=='P2_inflight'else 8
            self.assertEqual(r['reserved'],cap)
            self.assertEqual(r['states']['FAILED'],cap if name=='simultaneous'else 1)
            self.assertEqual(r['states']['COMPLETED'],0 if name=='simultaneous'else cap-1)
        if name=='call_limit':self.assertEqual(r['states']['COMPLETED'],3)
        if name=='stop_save':self.assertIn('secondary_stop_error',r)
    return test


for codec in ('reference','fused'):
    for name,settings in FAULTS.items():setattr(Integration,'test_fault_'+codec+'_'+name,fault_test(codec,name,settings))


def mutation_test(role):
    def test(self):
        self.fixture();(self.root/V.ROLES[role]).write_bytes(b'changed')
        with self.assertRaises(A.Denied):self.invoke()
        self.assertIsNone(self.l.row(self.c['scope']))
    return test


for role in V.ROLES:setattr(Integration,'test_source_change_'+role,mutation_test(role))

if __name__=='__main__':unittest.main()
