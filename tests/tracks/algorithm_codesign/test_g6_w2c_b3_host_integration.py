"""R1/F01 host integration: private artificial authorities/endpoints only.

The test-only receipt bridge adapts ARTIFICIAL claims to the old Boundary shape.
It is never exported by product source. No real authorization/store/claim exists.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from itertools import cycle
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import sys
import threading
import unittest
from unittest.mock import patch
import tempfile

import test_g6_w2c_b3_admission as Q
import test_g6_w2c_b3_driver as T
import test_g6_w2c_b3_f01_provenance as F

sys.dont_write_bytecode = True
A, D = Q.M, Q.D
H = importlib.import_module('trottertracks.algorithm_codesign.g6_w2c_b3_host')
ROOT = Q.ROOT
REAL_PAIR = D.run_pair
REAL_STREAM = D.Stream


class ArtificialBoundaryBridge(A.ArtificialAdmission):
    """Only inside tests with mocked module loader; DB claim remains ARTIFICIAL."""
    def admit_once(self, value):
        result = self.admit_artificial(value)
        self.fault('receipt_created')
        return dict(result, claim_id='B3_NEW_ARTIFICIAL_RECEIPT_BRIDGE_ONLY')


def host_worker(args):
    c, root, store_id, anchor = args
    try:
        H.execute_approved(c, root=root, ledger=A.ClaimLedger(Path(root)/'ledger', store_id),
                           authority=A.PinnedApproval(anchor))
        return True
    except (A.Denied, D.Stop, sqlite3.Error, OSError):
        return False


class Integration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c, self.ledger, self.auth = Q.fixture(self.root)
        p = self.root/A.HOST;p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT/A.HOST).read_bytes())
        self.c['schema'] = A.HOST_SCHEMA
        self.c['source_pins'][A.HOST] = sha256(p.read_bytes()).hexdigest()
        self.c['source_roles']['host'] = [A.HOST]
        self.c['entrypoint'] = {'path': A.HOST, 'sha256': self.c['source_pins'][A.HOST],
                                'api': 'execute_approved_v1'}
        self.auth = Q.authorize(self.c, self.root, self.ledger)
        self.ports, self.phases, self.endpoints = [], [], []

    def invoke(self, c=None, authority='default'):
        return H.execute_approved(c or self.c, root=self.root, ledger=self.ledger,
                                  authority=self.auth if authority == 'default' else authority)

    @contextmanager
    def connected(self, *, fault=lambda p: None, endpoint=None, stream_mode=None,
                  stage=None, actual_child=False, operation=None):
        def factory(c, root, ledger, authority):
            p = ArtificialBoundaryBridge(c, root, ledger, authority, fault=fault)
            self.ports.append(p)
            return p
        mods = T.modules()
        mods['g7'].DeterministicBits = lambda seed:T.Bits(cycle((0, 1, 2)))
        def saved(*args):
            if stage == 'saved_budget_before':raise F.Fault(stage)
            # No parse/reanalysis of saved scientific rows. Explicit mock rows.
            result = {arm:{'artificial_only':True} for arm in D.ARMS}
            if stage == 'saved_budget_after':raise F.Fault(stage)
            return result
        def modules(*args):
            if stage == 'module_load':raise F.Fault(stage)
            return mods
        real_endpoint = D.LegacyEndpoint
        def make_endpoint(*args, **kw):
            if stage == 'correctness_before':raise F.Fault(stage)
            e = endpoint() if endpoint is not None else real_endpoint(*args, **kw)
            self.endpoints.append(e)
            return e
        def pair(e, make_bits, cells, stream, meter, **kw):
            self.phases.append((kw['phase'],kw['pass_name'],cells[0]['arm'],cells[0]['layer']))
            # Scientific candidate N stays frozen. Only injected toy execution is 16/axis.
            toy = [dict(c,N=16) for c in cells]
            return REAL_PAIR(e, make_bits, toy, stream, meter, **kw)
        real_stream = REAL_STREAM
        def stream(*args, **kw):
            if stream_mode == 'flush':kw['flush_rows'] = 1  # Injected toy I/O failure only.
            s = real_stream(*args, **kw)
            if stream_mode:s.file = F.FailingFile(s, stream_mode)
            return s
        with ExitStack() as stack:
            stack.enter_context(patch.object(H, '__file__', str(self.root/A.HOST)))
            stack.enter_context(patch.object(A, 'ProductionAdmission', factory))
            stack.enter_context(patch.object(D, 'load_saved', saved))
            stack.enter_context(patch.object(D, 'load_modules', modules))
            stack.enter_context(patch.object(D, 'clock', T.FakeClock()))
            stack.enter_context(patch.object(D, 'memory_snapshot', lambda:{'artificial_only':True}))
            stack.enter_context(patch.object(D, 'LegacyEndpoint', make_endpoint))
            stack.enter_context(patch.object(D, 'run_pair', pair))
            stack.enter_context(patch.object(D, 'Stream', stream))
            if not actual_child:stack.enter_context(patch.object(D, 'isolated_pair', lambda op:op()))
            if stage == 'source_after_receipt':
                stack.enter_context(patch.object(D, 'check_pins', side_effect=F.Fault(stage)))
            if stage == 'output_before':
                actual_mkdir = Path.mkdir
                def mkdir(p,*args,**kw):
                    if p == self.root/'artificial-output':raise OSError(stage)
                    return actual_mkdir(p,*args,**kw)
                stack.enter_context(patch.object(Path,'mkdir',mkdir))
            if stage == 'output_after':
                stack.enter_context(patch.object(D,'isolated_pair',side_effect=F.Fault(stage)))
            if operation is not None:stack.enter_context(patch.object(D,'execute',operation))
            yield

    def spent(self):
        row = self.ledger.row(self.c['scope'])
        self.assertIsNotNone(row)
        self.assertTrue(all(p.state=='STOPPED' for p in self.ports))
        with self.connected(),self.assertRaises((A.Denied,sqlite3.Error)):
            self.invoke()

    def test_full_mock_bundle_correctness_then_both_passes(self):
        with self.connected():result = self.invoke()
        self.assertEqual(len(result['pairs']),24)
        self.assertEqual(sum(sum(a['N_executed'] for a in p['axes']) for p in result['pairs']),768)
        self.assertEqual([p[:2] for p in self.phases],
            [('correctness','baseline')]*8+[('profile','baseline')]*8+[('profile','instrumented')]*8)
        self.assertEqual(len(self.endpoints),24)
        self.assertEqual(sum(e.session is not None for e in self.endpoints),3)
        self.assertEqual(sum(e.layer=='B' and e.arm!='full_return' for e in self.endpoints),9)
        self.assertTrue(result['mandatory_STOP'])
        self.assertEqual(self.ledger.row(self.c['scope'])[4],'STOPPED')
        self.spent()

    def test_host_uses_real_boundary_and_admission_once(self):
        original = D.Boundary.enter
        count=[]
        def enter(guard,c,port=None):
            count.append(port)
            return original(guard,c,port)
        with self.connected(),patch.object(D.Boundary,'enter',enter):self.invoke()
        self.assertEqual(len(count),1)
        row=self.ledger.row(self.c['scope'])
        self.assertTrue(row[0].startswith('ARTIFICIAL_'))
        binding=json.loads(row[2])
        self.assertEqual(binding['authorization']['sha256'],self.auth.anchor.authorization_sha256)
        self.spent()

    def test_no_authority_stops_before_scientific_import(self):
        with patch.object(H,'__file__',str(self.root/A.HOST)),patch.object(D,'load_modules')as modules:
            with self.assertRaisesRegex(A.Denied,'INDEPENDENT_SCIENCE_APPROVAL_REQUIRED'):
                self.invoke(authority=None)
        modules.assert_not_called();self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_approved_boolean_is_not_authority(self):
        c=deepcopy(self.c);c['approved']=True
        with patch.object(H,'__file__',str(self.root/A.HOST)),patch.object(D,'load_modules')as loader:
            with self.assertRaises(A.Denied):self.invoke(c,authority=None)
        loader.assert_not_called();self.assertIsNone(self.ledger.row(self.c['scope']))


    def test_authorization_content_hash_mismatch(self):
        Path(self.auth.anchor.authorization_reference).write_text('{"approved":true}')
        with self.connected(),self.assertRaisesRegex(A.Denied,'AUTHORIZATION_CONTENT_HASH'):
            self.invoke()
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_trusted_store_identity_mismatch(self):
        auth=A.PinnedApproval(replace(self.auth.anchor,ledger_store_id='not-the-trusted-store'))
        with self.connected(),self.assertRaisesRegex(A.Denied,'TRUSTED_LEDGER_IDENTITY'):
            self.invoke(authority=auth)
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_store_missing_is_not_reinitialized_by_host(self):
        self.ledger.path.unlink()
        with self.connected(),self.assertRaises(FileNotFoundError):self.invoke()
        self.assertFalse(self.ledger.path.exists());self.assertEqual(self.phases,[])

    def test_output_collision_is_rejected_before_reservation(self):
        (self.root/'artificial-output').mkdir()
        with self.connected(),self.assertRaisesRegex(A.Denied,'OUTPUT_COLLISION'):
            self.invoke()
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_invalid_receipt_after_reservation_is_terminal(self):
        original=ArtificialBoundaryBridge.admit_once
        def bad(p,value):return dict(original(p,value),contract_digest='0'*64)
        with self.connected(),patch.object(ArtificialBoundaryBridge,'admit_once',bad),self.assertRaisesRegex(D.Stop,'INVALID_OR_OLD'):
            self.invoke()
        self.assertEqual(self.phases,[]);self.spent()

    def test_source_change_after_reservation_is_terminal(self):
        def fault(point):
            if point=='before_binding_recheck':(self.root/A.HOST).write_bytes(b'artificial changed pinned host')
        with self.connected(fault=fault),self.assertRaisesRegex(A.Denied,'SOURCE_PINS_MISMATCH'):
            self.invoke()
        self.assertIsNotNone(self.ledger.row(self.c['scope']))
        self.assertEqual(self.phases,[])
        self.assertTrue(all(p.state=='STOPPED'for p in self.ports))

    def test_original_production_port_rejects_v1(self):
        c=deepcopy(self.c);c['schema']='b3_formal_admission_v1'
        with self.assertRaisesRegex(A.Denied,'HOST_V2_CONTRACT_REQUIRED'):
            A.validate_r1(c,self.root,False)
        with self.connected(),self.assertRaisesRegex(A.Denied,'HOST_V2_CONTRACT_REQUIRED'):
            self.invoke(c)
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_source_pins_bind_runtime_host_import_location(self):
        with self.assertRaisesRegex(A.Denied,'LOADED_HOST_ROOT_MISMATCH'):
            H.check_host(self.c,self.root)
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_actual_host_bytes_mismatch(self):
        (self.root/A.HOST).write_bytes(b'ARTIFICIAL corrupt host copy')
        with self.connected(),self.assertRaisesRegex(A.Denied,'HOST_SOURCE_SHA_MISMATCH'):
            self.invoke()
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_old_driver_forbidden_and_fixed_driver_not_blacklisted(self):
        c=deepcopy(self.c)
        c.update(scope='MOCK_SCOPE_NO_REAL_AUTHORIZATION',claim_id='B3_NEW_ARTIFICIAL_STATIC_ONLY')
        with patch.object(D,'__file__',str(self.root/A.DRIVER)),\
             patch.object(A,'__file__',str(self.root/A.ADMISSION)),\
             patch.object(H,'__file__',str(self.root/A.HOST)):
            binding=A.validate_r1(c,self.root,False)
            self.assertEqual(binding['source_pins_digest'],D.identity(c['source_pins']))
            old=(ROOT/A.DRIVER).read_bytes()
            import subprocess
            raw=subprocess.check_output(['git','-C',str(ROOT),'show',
                'e4af6c064db805cbec611bcc8f0ff75a4754d38b:'+A.DRIVER])
            self.assertEqual(sha256(raw).hexdigest(),A.OLD_DRIVER_SHA)
            (self.root/A.DRIVER).write_bytes(raw)
            c['source_pins'][A.DRIVER]=A.OLD_DRIVER_SHA
            with self.assertRaisesRegex(A.Denied,'F01_OLD_DRIVER_FORBIDDEN'):
                A.validate_r1(c,self.root,False)
            (self.root/A.DRIVER).write_bytes(old)
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_stop_secondary_exception_does_not_replace_primary(self):
        def op(c,*,approval_port,**kw):
            D.Boundary().enter(c,approval_port)
            raise F.Fault('primary driver failure')
        with self.connected(operation=op),patch.object(ArtificialBoundaryBridge,'stop',side_effect=OSError('secondary STOP')):
            with self.assertRaisesRegex(F.Fault,'primary driver failure')as caught:self.invoke()
        self.assertEqual(caught.exception.b3_admission_stop_failure['reason'],'secondary STOP')
        self.assertEqual(self.ledger.row(self.c['scope'])[4],'ADMITTED')
        # The mocked stop could not set process-local state, but DB stays spent.
        with self.connected(),self.assertRaises(sqlite3.IntegrityError):self.invoke()

    def test_stop_save_failures_preserve_primary_and_spent_claim(self):
        def fault(point):
            if point in ('failure_record_save','stop_record_save'):raise OSError('artificial stop storage')
        with self.connected(fault=fault,endpoint=lambda:F.ToyEndpoint(fail_sample=lambda t:True)):
            with self.assertRaisesRegex(F.Fault,'sample original')as caught:self.invoke()
        self.assertIn('MANDATORY_STOP_RECORD_FAILURE',caught.exception.b3_admission_stop_failure['reason'])
        self.spent()

    def test_stop_failure_on_normal_return_is_not_reported_success(self):
        def op(c,*,approval_port,**kw):return D.Boundary().enter(c,approval_port)
        def fault(point):
            if point=='stop_record_save':raise OSError('artificial STOP no space')
        with self.connected(fault=fault,operation=op),self.assertRaisesRegex(A.Denied,'MANDATORY_STOP_RECORD_FAILURE'):
            self.invoke()
        self.spent()

    def test_correctness_zero_coverage_stops_at_16_no_full_N(self):
        class Zero(F.ToyEndpoint):
            def sample(self,bits,trial_id):
                self.samples.append(trial_id)
                return D.Outcome('ZERO_R')
        e=Zero()
        with self.connected(endpoint=lambda:e),self.assertRaisesRegex(D.Stop,'NO_EXTRA_TRIALS'):
            self.invoke()
        self.assertEqual(len(e.samples),16)
        self.assertEqual(self.phases,[('correctness','baseline','ordinary','A')])
        self.spent()

    def test_successful_correctness_gate_no_premature_full_N(self):
        with self.connected():self.invoke()
        self.assertEqual(self.phases[7][:2],('correctness','baseline'))
        self.assertEqual(self.phases[8][:2],('profile','baseline'))
        self.spent()

    def test_normal_eight_frozen_pre_fix_cases_through_host(self):
        refs=json.loads((ROOT/'tests/tracks/algorithm_codesign/fixtures/g6_w2c_b3_f01_normal_reference.json').read_text())
        checked=[]
        def op(c,*,approval_port,**kw):
            D.Boundary().enter(c,approval_port)
            for i,ref in enumerate(refs['cases']):
                meter=D.Meter(T.FakeClock())
                s=D.Stream(self.root/('normal-'+str(i)+'.jsonl'),meter,byte_limit=1000000,flush_rows=2)
                e=D.LegacyEndpoint(T.modules(),ref['arm'],ref['layer'],
                    T.accepted_candidate()['decisions']['angle']if ref['layer']=='B'else None,lambda:'source')
                cells=[dict(arm=ref['arm'],layer=ref['layer'],axis=x,N=4)for x in D.AXES]
                result=REAL_PAIR(e,lambda axis:T.Bits(cycle((0,1,2))),cells,s,meter,
                    phase='profile',pass_name='baseline',correctness_trials=2)
                s.close();data=s.path.read_bytes()
                self.assertEqual(result,ref['summary']);self.assertEqual(meter.rows,ref['coarse'])
                self.assertEqual(sha256(data).hexdigest(),ref['stream_sha256'])
                self.assertEqual(s.previous,ref['chain_sha256'])
                self.assertEqual((s.rows,len(data)),(ref['stream_rows'],ref['stream_bytes']))
                self.assertEqual((e.g.packet_queries,e.g.event_queries),(ref['packet_queries'],ref['event_queries']))
                checked.append((ref['arm'],ref['layer']))
        with self.connected(operation=op):self.invoke()
        self.assertEqual(len(checked),8);self.spent()

    def test_child_original_failure_prefix_and_parent_stop(self):
        with self.connected(actual_child=True,endpoint=lambda:F.ToyEndpoint(fail_sample=lambda t:True)):
            with self.assertRaisesRegex(D.Stop,'sample original'):self.invoke()
        f=json.loads((self.root/'artificial-output/correctness_baseline_ordinary_A.failure.json').read_text())
        provenance=json.loads(f['trial_failure_provenance'])
        self.assertEqual(provenance['current_attempt']['axis'],'X')
        self.assertIsNone(provenance['inflight_event'])
        self.assertFalse(provenance['current_attempt']['result_acquired'])
        self.assertTrue((self.root/'artificial-output/failure_prefix.json').is_file())
        self.spent()

    def test_child_abnormal_exit_parent_stop_and_no_resume(self):
        class Exit(F.ToyEndpoint):
            def sample(self,*args):os._exit(88)
        with self.connected(actual_child=True,endpoint=Exit),self.assertRaisesRegex(D.Stop,'OWN_WORKER_EXITED'):
            self.invoke()
        self.assertTrue((self.root/'artificial-output/failure_prefix.json').is_file())
        self.spent()

    def test_owned_host_process_exit_after_reservation(self):
        def fault(point):
            if point=='after_reserve_commit':os._exit(88)
        with self.connected(fault=fault):
            proc=mp.get_context('fork').Process(target=lambda:self.invoke())
            proc.start();proc.join()
        self.assertEqual(proc.exitcode,88)
        self.assertEqual(self.ledger.row(self.c['scope'])[4],'RESERVED')
        self.spent()

    def test_host_thread_race_at_most_one(self):
        def op(c,*,approval_port,**kw):return D.Boundary().enter(c,approval_port)
        barrier=threading.Barrier(32)
        def worker(i):
            barrier.wait()
            return host_worker((self.c,str(self.root),self.ledger.store_id,self.auth.anchor))
        with self.connected(operation=op),ThreadPoolExecutor(max_workers=32)as pool:
            self.assertEqual(sum(pool.map(worker,range(32))),1)
        self.spent()

    def test_host_owned_process_race_at_most_one(self):
        def op(c,*,approval_port,**kw):return D.Boundary().enter(c,approval_port)
        args=(self.c,str(self.root),self.ledger.store_id,self.auth.anchor)
        with self.connected(operation=op),mp.get_context('fork').Pool(32)as pool:
            self.assertEqual(sum(pool.map(host_worker,[args]*32,chunksize=1)),1)
        self.spent()

    def test_different_digest_and_claim_do_not_unspend_scope(self):
        def op(c,*,approval_port,**kw):return D.Boundary().enter(c,approval_port)
        with self.connected(operation=op):self.invoke()
        self.c['claim_id']='ARTIFICIAL_OTHER'
        self.auth=Q.authorize(self.c,self.root,self.ledger)
        with self.connected(operation=op),self.assertRaises(sqlite3.IntegrityError):self.invoke()

    def test_same_claim_different_scope_rejected(self):
        def op(c,*,approval_port,**kw):return D.Boundary().enter(c,approval_port)
        with self.connected(operation=op):self.invoke()
        self.c['scope']='ARTIFICIAL_OTHER_SCOPE'
        self.auth=Q.authorize(self.c,self.root,self.ledger)
        with self.connected(operation=op),self.assertRaises(sqlite3.IntegrityError):self.invoke()

    def test_reused_port_and_resume_rejected(self):
        def op(c,*,approval_port,**kw):return D.Boundary().enter(c,approval_port)
        with self.connected(operation=op):self.invoke()
        p=self.ports[-1]
        with self.assertRaises(A.Denied):p.admit_once(D.identity(self.c['candidate']))
        self.spent()

    def test_no_scientific_imports(self):
        with self.connected():self.invoke()
        for n in ('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator'):
            self.assertNotIn('trottertracks.algorithm_codesign.'+n,sys.modules)


def invalid_case(edit):
    def test(self):
        c=deepcopy(self.c);edit(c)
        with self.connected(),self.assertRaises((A.Denied,KeyError,ValueError)):
            self.invoke(c)
        self.assertIsNone(self.ledger.row(self.c['scope']))
        self.assertEqual(self.phases,[])
    return test

for name,edit in {
 'missing_host_role':lambda c:c['source_roles'].pop('host'),
 'missing_host_pin':lambda c:c['source_pins'].pop(A.HOST),
 'different_host_path':lambda c:c['entrypoint'].update(path='other.py'),
 'different_host_api':lambda c:c['entrypoint'].update(api='run_anything'),
 'host_sha':lambda c:c['entrypoint'].update(sha256='0'*64),
 'formal_digest':lambda c:c['candidate'].update(status='ARTIFICIAL_EDITED'),
 'input_sha':lambda c:c['input_pins'].update({Q.BASE['saved_evidence']['path']:'0'*64}),
 'driver_sha':lambda c:c['source_pins'].update({A.DRIVER:'0'*64}),
 'science_sha':lambda c:c['source_pins'].update({next(iter(Q.BASE['source_pins'])):'0'*64}),
 'angle':lambda c:c['candidate']['decisions']['angle'].update(J=64),
 'seed':lambda c:c['candidate']['decisions']['seed'].update(namespace='OLD'),
 'cell_n':lambda c:c['candidate']['cells'][0].update(N=1),
 'retry':lambda c:c.update(retry=1),
 'resume':lambda c:c.update(resume=True),
}.items():setattr(Integration,'test_reject_'+name,invalid_case(edit))


def reserve_fault(point,committed):
    def test(self):
        def fault(p):
            if p==point:raise F.Fault('artificial '+point)
        with self.connected(fault=fault),self.assertRaisesRegex(F.Fault,point):self.invoke()
        self.assertEqual(self.phases,[])
        if committed:self.spent()
        else:self.assertIsNone(self.ledger.row(self.c['scope']))
    return test

for point,committed in {'before_reserve':False,'insert_before_commit':False,
 'after_reserve_commit':True,'before_binding_recheck':True,'partial_claim_detail_write':True,
 'detail_update_before_commit':True,'before_output_creation':True,'receipt_created':True}.items():
    setattr(Integration,'test_admission_fault_'+point,reserve_fault(point,committed))


def driver_stage(stage):
    def test(self):
        with self.connected(stage=stage),self.assertRaises((F.Fault,OSError)):self.invoke()
        self.assertEqual(self.phases,[])
        self.spent()
    return test
for stage in ('source_after_receipt','saved_budget_before','saved_budget_after','module_load',
              'output_before','output_after','correctness_before'):
    setattr(Integration,'test_driver_stage_'+stage,driver_stage(stage))


def provenance_case(mode):
    def test(self):
        e=F.ToyEndpoint()
        if mode=='previous':e.fail_sample=lambda t:t.endswith('/X/1')
        if mode=='axis':e.fail_sample=lambda t:t.endswith('/Y/0')
        if mode=='before_result':e.fail_sample=lambda t:True
        if mode=='angle':e.fail_angle=True
        if mode=='evidence':e.bad_evidence=True
        kwargs={'endpoint':lambda:e}
        if mode in ('partial','flush'):kwargs['stream_mode']=mode
        with ExitStack()as stack:
            stack.enter_context(self.connected(**kwargs))
            if mode=='failure_save':
                e.fail_sample=lambda t:True
                original=REAL_STREAM.emit
                def emit(s,payload):
                    if payload.get('kind')=='TYPED_FAILURE':raise OSError('secondary artificial failure save')
                    return original(s,payload)
                stack.enter_context(patch.object(REAL_STREAM,'emit',emit))
            with self.assertRaises((F.Fault,OSError))as caught:self.invoke()
        f=caught.exception.b3_failure_provenance
        self.assertFalse(f['is_random_zero']);self.assertTrue(f['mandatory_STOP'])
        if mode in ('previous','axis','before_result','failure_save'):
            self.assertIsNone(f['inflight_event']);self.assertFalse(f['current_attempt']['result_acquired'])
        if mode=='previous':
            self.assertTrue(f['last_completed_trial']['trial_id'].endswith('/X/0'))
            self.assertTrue(f['current_attempt']['trial_id'].endswith('/X/1'))
        if mode=='axis':
            self.assertEqual(f['last_completed_trial']['axis'],'X')
            self.assertEqual(f['current_attempt']['axis'],'Y')
        if mode in ('partial','flush'):
            self.assertFalse(f['current_attempt']['stream_emit_returned'])
            self.assertFalse(f['current_attempt']['trial_completed'])
            self.assertNotEqual(f['candidate_trace_sha256'],f['trace_sha256'])
        if mode=='failure_save':self.assertIn('failure_record_write_error',f)
        self.spent()
    return test
for mode in ('previous','axis','before_result','angle','evidence','partial','flush','failure_save'):
    setattr(Integration,'test_integrated_f01_'+mode,provenance_case(mode))

if __name__=='__main__':unittest.main()
