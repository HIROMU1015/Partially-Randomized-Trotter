"""New artificial candidate tests. No real scientific import, grant or call."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import threading
import unittest

import b3_pilot_artificial_fixture as F
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_candidate as C
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_offline as V
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_verifier_core as G


def crash_admission(root,c,inner,store,anchor):
    ledger=C.A.A.ClaimLedger(Path(root)/'ledger',store)
    auth=C.A.A.PinnedApproval(anchor)
    def crash(point):
        if point=='after_reserve_commit':os._exit(79)
    C.ArtificialAdmission(c,inner,Path(root),ledger,auth,fault=crash).admit_artificial(C.D.identity(inner['candidate']))


class Pilot(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.c,self.inner,self.ledger,self.auth=F.fixture(self.root)

    def port(self,**kwargs):
        return C.ArtificialAdmission(self.c,self.inner,self.root,self.ledger,self.auth,**kwargs)

    def digest(self):return C.D.identity(self.inner['candidate'])

    def evidence(self,name,extra):
        target=os.environ.get('B3_PILOT_EVIDENCE')
        if not target:return
        dest=Path(target)/name;dest.mkdir(parents=True,exist_ok=False)
        # Failure context contains exact Fractions; retain their canonical tags.
        (dest/'observations.json').write_text(json.dumps(json.loads(C.P.safe_json(extra)),indent=2)+'\n')
        for n,value in [('pilot_contract.json',self.c),('inner_contract.json',self.inner)]:
            (dest/n).write_text(json.dumps(value,indent=2)+'\n')
        shutil.copyfile(self.ledger.path,dest/'FAKE_ONLY_claims.sqlite3')
        output=self.root/self.inner['candidate']['output_relative']
        if output.exists():
            with tarfile.open(dest/'artificial_raw.tar.gz','w:gz') as tf:tf.add(output,arcname=output.name)

    def test_normal_live_trace_receipt_and_worker_isolation(self):
        F.F.PARENT_CANARY='FAKE_PRIVATE_PARENT_ONLY'
        try:r=F.invoke(self.root,self.c,self.inner,self.ledger,self.auth)
        finally:F.F.PARENT_CANARY=None
        self.assertTrue(r['complete']);v=V.verify(self.root,self.c,self.inner,r)
        self.assertEqual(v['history']['counts']['completed'],768)
        self.assertEqual(r['fake_claim_status'],'STOPPED')
        out=self.root/self.inner['candidate']['output_relative']
        receipt=out/'pilot_receipt.json';saved=receipt.read_bytes();obj=json.loads(saved)
        obj['summary_sha256']='0'*64;receipt.write_bytes(C.A.A.encoded(obj))
        with self.assertRaisesRegex(G.V.Invalid,'RECEIPT_SUMMARY'):V.verify(self.root,self.c,self.inner,r)
        receipt.write_bytes(saved)
        self.assertEqual(set(vars(F.F.ArtificialFactory(self.inner['runtime_manifest']))),{'runtime_manifest'})
        self.evidence('normal',dict(report=r,offline=v,worker_parent_canary_inherited=False))

    def test_production_entry_direct_helpers_and_bool_do_not_unlock(self):
        for f in (C.production_entry,C.ProductionFactory(),C.ProductionBits()):
            with self.assertRaisesRegex(C.A.A.Denied,'NO_SCIENCE_ENTRY'):f(adopted=True,run_ready=True)

    def test_atomic_same_scope_eight_contenders(self):
        barrier=threading.Barrier(8)
        def attempt(_):
            port=self.port();barrier.wait()
            try:
                port.admit_artificial(self.digest());port.stop();return 'ADMITTED'
            except (C.A.A.Denied,sqlite3.IntegrityError):return 'DENIED'
        with ThreadPoolExecutor(max_workers=8)as pool:outcomes=list(pool.map(attempt,range(8)))
        self.assertEqual(outcomes.count('ADMITTED'),1)
        self.assertEqual(self.ledger.row(self.inner['scope'])[4],'STOPPED')
        self.evidence('race',dict(contenders=8,outcomes=outcomes,store_id=self.ledger.store_id,
            scope=self.inner['scope'],same_store=True,retry=0,resume=False,
            offline_snapshot=V.admission_snapshot(self.ledger.path,self.inner,expected_status='STOPPED')))

    def test_double_claim_and_spent_after_stop(self):
        p=self.port();p.admit_artificial(self.digest());p.stop()
        with self.assertRaises(C.A.A.Denied):p.admit_artificial(self.digest())
        with self.assertRaises(sqlite3.IntegrityError):self.port().admit_artificial(self.digest())
        self.assertEqual(self.ledger.row(self.inner['scope'])[4],'STOPPED')

    def test_commit_ack_uncertain_never_refunds(self):
        def fail(point):
            if point=='after_reserve_commit':raise OSError('FAKE_COMMIT_ACK_LOST')
        with self.assertRaises(OSError):self.port(fault=fail).admit_artificial(self.digest())
        self.assertEqual(self.ledger.row(self.inner['scope'])[4],'RESERVED')
        with self.assertRaises(sqlite3.IntegrityError):self.port().admit_artificial(self.digest())
        self.evidence('commit_ack_uncertain',dict(status='RESERVED',spent=True,refunded=False))

    def test_owned_parent_process_crash_leaves_spent_reservation(self):
        p=mp.get_context('spawn').Process(target=crash_admission,args=(str(self.root),self.c,self.inner,self.ledger.store_id,self.auth.anchor))
        p.start();p.join(30)
        self.assertFalse(p.is_alive());self.assertEqual(p.exitcode,79)
        self.assertEqual(self.ledger.row(self.inner['scope'])[4],'RESERVED')
        with self.assertRaises(sqlite3.IntegrityError):self.port().admit_artificial(self.digest())
        self.evidence('parent_crash',dict(exitcode=p.exitcode,status='RESERVED',spent=True,
            host_total_cpu_ns=None,worker_cpu_ns=None,scientific_calls=0))

    def test_mutation_between_reservation_and_binding_recheck_is_spent(self):
        def mutate(point):
            if point=='before_binding_recheck':port.pilot['proposal']['profile_n']=64
        port=self.port(fault=mutate)
        with self.assertRaises(C.A.A.Denied):port.admit_artificial(self.digest())
        self.assertEqual(self.ledger.row(self.inner['scope'])[4],'STOPPED')
        self.evidence('binding_recheck',dict(status='STOPPED',spent=True,binding_mutation_denied=True))

    def test_log_budget_before_worker_entry(self):
        self.temp.cleanup();self.temp=tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.c,self.inner,self.ledger,self.auth=F.fixture(self.root,F.F.P.Settings(profile_n=16,log_bytes=100))
        with self.assertRaisesRegex(C.P.D.Stop,'GLOBAL_LOG_BYTE_LIMIT'):
            F.invoke(self.root,self.c,self.inner,self.ledger,self.auth)
        self.assertEqual(self.ledger.row(self.inner['scope'])[4],'STOPPED')
        self.evidence('log_budget',dict(status='STOPPED',spent=True,worker_entry=0,history='INCOMPLETE_EXPECTED'))

    def test_trace_partial_tail_rejected(self):
        p=self.root/'trace';p.write_bytes(b'{')
        with self.assertRaisesRegex(G.V.Invalid,'PARTIAL_LINE'):list(V.operations(p))

    def test_trace_canonical_mutation_rejected(self):
        p=self.root/'trace';raw='{ "op": "STOP" }';prev='0'*64
        p.write_text(json.dumps(dict(payload=raw,previous=prev,sha256=sha256((prev+raw).encode()).hexdigest()))+'\n')
        with self.assertRaisesRegex(G.V.Invalid,'CANONICAL'):list(V.operations(p))

    def test_inner_source_mismatch_before_reservation(self):
        p=self.root/next(iter(self.inner['source_pins']));p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaises(C.A.A.Denied):self.port().admit_artificial(self.digest())
        self.assertIsNone(self.ledger.row(self.inner['scope']))

    def test_runtime_mismatch_before_reservation(self):
        self.inner['runtime_manifest']['python']['version']='FAKE_WRONG_RUNTIME'
        with self.assertRaises(C.A.A.Denied):self.port().admit_artificial(self.digest())
        self.assertIsNone(self.ledger.row(self.inner['scope']))

    def test_schema_bytes_mismatch_before_reservation(self):
        p=self.root/next(iter(self.inner['schema_pins']));p.write_bytes(b'{}\n')
        with self.assertRaises(C.A.A.Denied):self.port().admit_artificial(self.digest())
        self.assertIsNone(self.ledger.row(self.inner['scope']))

    def test_wrong_outer_contract_digest(self):
        self.inner['pilot_candidate_binding']='0'*64
        with self.assertRaises(C.A.A.Denied):self.port().admit_artificial(self.digest())

    def test_retry_or_resume_or_adoption_or_budget_change_rejected(self):
        for group,key,value in ((None,'retry',1),(None,'resume',True),(None,'content_adopted',True),
                                ('proposal','total_calls',1),('proposal','log_bytes',1)):
            c=deepcopy(self.c);(c[group]if group else c)[key]=value
            with self.assertRaises(C.A.A.Denied):C.validate(c)

    def test_proposal_sizes_are_explicit_not_full_N(self):
        for n,total in ((64,4608),(256,16896),(1024,66048)):
            c=C.candidate(F.F.ROOT,n);C.validate(c);self.assertEqual(c['proposal']['total_calls'],total)
        self.assertEqual(self.c['scientific_reference']['call_budget']['total_sampling_calls'],31710848)

    def test_operation_mutations_reject_duplicate_and_post_stop_entry(self):
        tasks,_=G.plan(16);t=tasks[0]
        base=[dict(op='PHASE_BEGIN',phase=0),dict(op='RESERVE',task_key=t['key'],trial_id=G.V.trial(dict(t,N=16),'X',0))]
        def rows(xs):return [dict(x,seq=i,run_id='ART')for i,x in enumerate(xs)]
        for tail in ([base[1]], [dict(op='STOP'),dict(base[1],op='ENTER')],
                     [dict(base[1],op='ENTER'),dict(base[1],op='COMPLETE'),dict(base[1],op='COMPLETE')],
                     [dict(op='PHASE_BEGIN',phase=1)]):
            with self.assertRaises(G.V.Invalid):G.check_operations(rows(base+tail),tasks,'ART')

    def test_cost_no_pair_or_fine_double_add(self):
        r=dict(host_outer=dict(parent_cpu_ns=3,worker_cpu_ns=7,wall_ns=9),complete=True,total_cpu_ns=100,pair_cpu=90)
        self.assertEqual(V.cost(r)['total_cpu_ns'],10)
        r['host_outer']['total_cpu_ns']=100
        with self.assertRaises(G.V.Invalid):V.cost(r)

    def test_no_scientific_modules_loaded(self):
        self.assertFalse([n for n in sys.modules if n.rsplit('.',1)[-1]in C.P.FORBIDDEN])


def fault_test(name,settings):
    def test(self):
        # Replace unused fixture with a fresh store and its bound fault configuration.
        self.temp.cleanup();self.temp=tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.c,self.inner,self.ledger,self.auth=F.fixture(self.root,settings)
        r=F.invoke(self.root,self.c,self.inner,self.ledger,self.auth)
        self.assertFalse(r['complete']);self.assertEqual(r['fake_claim_status'],'STOPPED')
        v=V.verify(self.root,self.c,self.inner,r)
        self.assertTrue(v['history']['stopped']);self.assertFalse(v['cost']['failed_prefix_comparison_eligible'])
        if name in ('worker_crash_unknown','reserved_crash_unknown','unacked_credit_unknown'):
            self.assertGreater(v['history']['counts'].get('unknown',0),0)
        if name=='drain':
            ops=list(V.operations(self.root/self.inner['candidate']['output_relative']/'operations.jsonl'))
            stop=next(i for i,x in enumerate(ops)if x['op']=='STOP')
            self.assertTrue(any(x['op']=='COMPLETE'for x in ops[stop+1:]))
        self.evidence(name,dict(report=r,offline=v))
    return test


for name,kw in {
    'sample':dict(faults=((0,0,'sample',0),)),
    'event':dict(faults=((0,0,'event',0),)),
    'after_event':dict(faults=((0,0,'after_event',0),)),
    'angle':dict(faults=((1,0,'angle',0),)),
    'partial_write':dict(faults=((0,0,'write',0),)),
    'flush':dict(faults=((0,0,'flush',0),),flush_rows=1),
    'worker_crash_unknown':dict(faults=((0,0,'exit',0),)),
    'reserved_crash_unknown':dict(faults=((0,0,'reserved_exit',0),)),
    'unacked_credit_unknown':dict(faults=((0,0,'ack_exit',0),)),
    'call_budget':dict(call_limit=2),
    'output_budget':dict(result_bytes=100),
    'drain':dict(sync_failure='one'),
}.items():
    setattr(Pilot,'test_live_fault_'+name,fault_test(name,F.F.P.Settings(profile_n=16,codec='fused',workload='light',**kw)))


if __name__=='__main__':unittest.main()
