"""Independent pilot invariants, fixed raw, private-copy corruptions; no science."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from types import SimpleNamespace
import ast,json,os,shutil,sqlite3,tempfile,tracemalloc,unittest
from unittest.mock import patch
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_verifier_core as G
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_offline_adapter as O
from trottertracks.algorithm_codesign import g6_w2c_b3_v3_offline_adapter as B
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_independent_adapter as A
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_independent_identity as I
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_independent_trace as T
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_independent_cli as CLI
from trottertracks.algorithm_codesign import g6_w2c_b3_v3_offline_cost as C
ROOT=Path(__file__).resolve().parents[3]
PACKET=ROOT/'docs/research/track_b_g6_w2c_b3_limited_pilot_independent_acceptance_20261019'
TREE=PACKET/'fixed_reference/tree';CASES=TREE/I.PREFIX/'forward_evidence'
LEGACY=ROOT/'docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/fixed_reference'

def encode_trace(path,rows):
    prev=V.ZERO
    with Path(path).open('wb') as f:
        for r in rows:
            raw=json.dumps(r,separators=(',',':'),ensure_ascii=True);h=sha256((prev+raw).encode()).hexdigest()
            f.write((json.dumps(dict(payload=raw,previous=prev,sha256=h),separators=(',',':'))+'\n').encode());prev=h

class Published(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref=I.Reference(TREE,LEGACY)
        with ThreadPoolExecutor() as pool:cls.results={r['case']:r for r in pool.map(lambda p:A.verify(p,cls.ref),sorted(CASES.iterdir()))}
    def test_normal_768_and_trace_self_charge(self):
        r=self.results['normal'];self.assertEqual(r['calls']['completed'],768)
        self.assertEqual(r['history']['counts']['reserved'],r['history']['counts']['started_observed'])
        self.assertEqual(r['output_budget']['log']['physical_bytes'],1679509)
        self.assertGreater(r['trace']['bytes'],0);self.assertTrue(r['trace']['prefix_checked'])
    def test_actual_drain_keeps_seven_completions(self):
        r=self.results['drain'];self.assertEqual(r['calls']['failed'],1);self.assertEqual(r['calls']['completed'],7)
        self.assertEqual(r['history']['counts']['started_observed'],8);self.assertTrue(r['history']['stopped'])
    def test_snapshot_is_not_race_history(self):
        r=self.results['race'];self.assertFalse(r['admission']['snapshot_proves_atomic_history'])
        self.assertFalse(r['atomic_transition_history_verified']);self.assertEqual(r['observations']['outcomes'].count('ADMITTED'),1)
    def test_parent_crash_totals_stay_null(self):
        r=self.results['parent_crash'];self.assertIsNone(r['cost']['total_cpu_ns']);self.assertEqual(r['admission']['claim_status'],'RESERVED')
    def test_credit_not_refunded_after_worker_loss(self):
        r=self.results['unacked_credit_unknown'];self.assertGreater(r['calls']['unknown'],0)
        self.assertTrue(any(x['unknown_credit'] for x in r['output_budget'].values()))
        self.assertTrue(all(x['refunded'] is False for x in r['output_budget'].values()))
    def test_flush_visible_row_not_logical_commit(self):
        r=self.results['flush'];self.assertTrue(any(p['visible_trial_rows']>p['engine_committed_trials'] for p in r['pairs'] if p['engine_committed_trials'] is not None))
    def test_old_pending_47_unchanged(self):
        p=LEGACY.parent/'integration_acceptance_matrix.json';m=json.loads(p.read_text())
        for row in m['rows']:
            if row['check_id'] in ('P07','P08','P11'):self.assertEqual(row['counts'],{'PENDING_EVIDENCE':47})
        self.assertTrue(all(r['historical_pending_reclassified'] is False for r in self.results.values()))
    def test_static_alternative_n_and_two_conditions(self):
        rows=I.scope_plans();self.assertEqual([r['total'] for r in rows],[4608,16896,66048])
        for r in rows:self.assertEqual(r['baseline_per_condition'],r['instrumented_per_condition']);self.assertEqual(r['cells'],16)
    def test_observer_unknown_and_cost_no_double_sum(self):
        c=self.results['normal']['cost'];self.assertEqual(c['status'],'VERIFIED_PARALLEL_COST_BOUNDARIES')
        self.assertIsNone(c['observer_cost_ns']);self.assertFalse(c['pair_CPU_added_to_worker_CPU']);self.assertFalse(c['parent_worker_wall_summed'])
        h=self.results['normal']['report']['host_outer'];self.assertEqual(c['total_cpu_ns'],h['parent_cpu_ns']+h['worker_cpu_ns'])

for name in sorted(p.name for p in CASES.iterdir()):
    def test(self,n=name):
        r=self.results[n];expected='VERIFIED_COMPLETE' if n=='normal' else ('INSUFFICIENT_EVIDENCE' if n in ('race','parent_crash','binding_recheck','commit_ack_uncertain') else 'TECHNICAL_FAILURE')
        self.assertEqual(r['status'],expected,r['issues']);self.assertTrue(r['admission']['claim_status'] in ('STOPPED','RESERVED'))
        if 'history' in r:self.assertEqual(r['cost']['status'],'VERIFIED_PARALLEL_COST_BOUNDARIES',r['cost'].get('issues'))
    setattr(Published,'test_fixed_'+name,test)

class RawMutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='pilot-independent-master-')
        cls.out=B.extract_archive(CASES/'normal/artificial_raw.tar.gz',Path(cls.temp.name)/'raw')
        cls.r=O.safe_values(json.loads((CASES/'normal/observations.json').read_text()))['report']
        cls.c=json.loads((CASES/'normal/pilot_contract.json').read_text());cls.inner=json.loads((CASES/'normal/inner_contract.json').read_text());cls.ref=I.Reference(TREE,LEGACY)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='pilot-independent-mutation-');self.addCleanup(t.cleanup)
        self.root=Path(t.name);self.output=self.root/'artificial-output';shutil.copytree(self.out,self.output)
        self.report=deepcopy(self.r);self.candidate=deepcopy(self.c);self.contract=deepcopy(self.inner)
    def verify(self):return A.verify_output(self.output,self.candidate,self.contract,self.report,self.ref)
    def bad(self):
        with self.assertRaises((V.Invalid,V.Unsupported,KeyError,ValueError,TypeError)):self.verify()
    def test_pristine(self):self.assertEqual(self.verify()['status'],'VERIFIED_COMPLETE')
    def test_pilot_digest(self):self.report['pilot_binding']='0'*64;self.bad()
    def test_source_hash(self):self.contract['source_pins'][next(iter(self.contract['source_pins']))]='0'*64;self.bad()
    def test_runtime_schema_pin(self):self.contract['runtime_manifest']['schema_pins'][next(iter(self.contract['schema_pins']))]='0'*64;self.bad()
    def test_scope(self):self.contract['scope']='ARTIFICIAL_OTHER';self.bad()
    def test_claim(self):self.contract['claim_id']='ARTIFICIAL_OTHER';self.bad()
    def test_inner_candidate_binding(self):self.contract['pilot_candidate_binding']='0'*64;self.bad()
    def test_pair_duplicate(self):self.report['pairs'][1]=deepcopy(self.report['pairs'][0]);self.bad()
    def test_pair_offset(self):self.report['pairs'][0]['task']['offset']=1;self.bad()
    def test_call_cap(self):self.contract['settings']['call_limit']=767;self.report['settings']['call_limit']=767;self.bad()
    def test_statefile_changed(self):
        p=self.output/'trial_states.bin';b=bytearray(p.read_bytes());b[-1]=7;p.write_bytes(b);self.bad()
    def test_state_counter(self):self.report['states']['COMPLETED']-=1;self.bad()
    def test_summary_host_coordinator_cost(self):self.report['parent_cpu_ns']+=1;self.bad()
    def test_receipt_hash(self):
        p=self.output/'pilot_receipt.json';r=json.loads(p.read_text());r['summary_sha256']='0'*64;p.write_text(json.dumps(r,separators=(',',':')));self.bad()
    def test_receipt_missing(self):(self.output/'pilot_receipt.json').unlink();self.bad()
    def test_operation_tail(self):
        p=self.output/'operations.jsonl';p.write_bytes(p.read_bytes()[:-1]);self.bad()
    def test_operation_chain(self):
        p=self.output/'operations.jsonl';b=p.read_bytes();p.write_bytes(b[:30]+bytes([b[30]^1])+b[31:]);self.bad()
    def test_ack_overcount(self):self.report['output']['observed']['log']+=1;self.bad()
    def test_unknown_credit_refund(self):self.report['output']['reserved']['log']-=1;self.bad()
    def test_pair_partial_write(self):
        p=next(self.output.glob('correctness*.jsonl'));p.write_bytes(p.read_bytes()[:-9]);self.bad()
    def test_false_crash_success(self):self.report['states']['TERMINATED_WITH_UNKNOWN_RESULT']=1;self.bad()
    def test_cpu_double_add(self):
        v=self.verify();v['report']['total_cpu_ns']+=v['report']['pairs'][0]['cpu_ns'];self.assertEqual(C.aggregate(v)['status'],'COST_INTEGRITY_MISMATCH')
        self.report=deepcopy(self.r);self.report['host_outer']['parent_cpu_ns']=0;v=self.verify();self.assertEqual(v['status'],'INTEGRITY_MISMATCH');self.assertFalse(v['cost']['comparison_eligible'])
        row=next(r for r in CLI.acceptance(v)['checks'] if r['check_id']=='COST_BOUNDARIES');self.assertEqual(row['judgment'],'FAIL')
    def test_inputs_read_only(self):
        orig=Path.open
        def opened(p,mode='r',*args,**kwargs):
            if p.is_relative_to(self.output):self.assertIn(mode,('rb','r'))
            return orig(p,mode,*args,**kwargs)
        with patch.object(Path,'open',opened):self.verify()

for field,value in [('profile_n',64),('modes',['P1','S']),('seed_namespace','OTHER'),('total_calls',1),('codec','reference')]:
    def test(self,f=field,v=value):self.candidate['proposal'][f]=v;self.bad()
    setattr(RawMutations,'test_unadopted_proposal_change_'+field,test)

class Trace(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='pilot-independent-trace-');self.addCleanup(tmp.cleanup);self.path=Path(tmp.name)/'operations.jsonl';self.tasks,_=G.plan(16);self.t=self.tasks[0]
    def rows(self,ops):
        out=[];tid=V.trial(dict(self.t,N=16),'X',0)
        for i,op in enumerate(ops):
            r=dict(seq=i,run_id='ART',op=op)
            if op.startswith('PHASE'):r['phase']=0
            elif op.startswith('WRITE'):r.update(channel='log',amount=1)
            elif op!='STOP':r.update(task_key=self.t['key'],trial_id=tid)
            out.append(r)
        return out
    def check(self,rows):
        encode_trace(self.path,rows);d=T.Decisions(self.path,self.tasks,'ART');return G.check_operations(d,self.tasks,'ART')
    def test_enter_before_stop_can_drain(self):self.assertEqual(self.check(self.rows(['PHASE_BEGIN','RESERVE','ENTER','STOP','COMPLETE']))['counts']['completed'],1)
    def test_unknown_before_enter_keeps_reservation(self):
        r=self.check(self.rows(['PHASE_BEGIN','RESERVE','STOP','UNKNOWN']));self.assertEqual(r['counts']['reserved'],1);self.assertEqual(r['counts'].get('started_observed',0),0)
    def test_duplicate_seq(self):
        r=self.rows(['PHASE_BEGIN','RESERVE']);r[1]['seq']=0
        with self.assertRaisesRegex(V.Invalid,'SEQ'):self.check(r)
    def test_missing_seq(self):
        r=self.rows(['PHASE_BEGIN','RESERVE']);r[1]['seq']=2
        with self.assertRaisesRegex(V.Invalid,'SEQ'):self.check(r)
    def test_wrong_run_id(self):
        r=self.rows(['PHASE_BEGIN']);r[0]['run_id']='OTHER'
        with self.assertRaisesRegex(V.Invalid,'RUN_ID'):self.check(r)
    def test_wrong_trial_id(self):
        r=self.rows(['PHASE_BEGIN','RESERVE']);r[-1]['trial_id']='other'
        with self.assertRaisesRegex(V.Invalid,'TRIAL_ID'):self.check(r)
    def test_global_offset(self):
        for value in (1,False):
            r=self.rows(['PHASE_BEGIN','RESERVE']);r[-1]['global_offset']=value
            with self.assertRaisesRegex(V.Invalid,'OFFSET'):self.check(r)
    def test_phase_four_forbidden(self):
        r=self.rows(['PHASE_BEGIN']);r[0]['phase']=3
        with self.assertRaisesRegex(V.Invalid,'PHASE_RANGE'):self.check(r)
    def test_phase_barrier(self):
        r=self.rows(['PHASE_BEGIN','PHASE_END','PHASE_BEGIN']);r[-1]['phase']=1
        with self.assertRaisesRegex(V.Invalid,'PHASE_INCOMPLETE'):self.check(r)
    def test_ack_without_credit(self):
        with self.assertRaisesRegex(V.Invalid,'ACK_WITHOUT_CREDIT'):self.check(self.rows(['WRITE_ACK']))
    def test_canonical_payload_even_if_rehashed(self):
        r=self.rows(['STOP'])[0];raw=json.dumps(r);h=sha256((V.ZERO+raw).encode()).hexdigest();self.path.write_text(json.dumps(dict(payload=raw,previous=V.ZERO,sha256=h))+'\n')
        with self.assertRaisesRegex(V.Invalid,'CANONICAL'):list(T.Decisions(self.path,self.tasks,'ART'))
    def test_trace_prefix_must_end_on_row(self):
        encode_trace(self.path,self.rows(['STOP']));prefix=dict(bytes=1,rows=1,chain='0'*64,sha256='0'*64)
        with self.assertRaisesRegex(V.Invalid,'PREFIX'):list(T.Decisions(self.path,self.tasks,'ART',prefix=prefix))

for name,ops in [('stop_enter',['PHASE_BEGIN','RESERVE','STOP','ENTER']),('stop_reserve',['PHASE_BEGIN','STOP','RESERVE']),('duplicate_reserve',['PHASE_BEGIN','RESERVE','RESERVE']),('duplicate_complete',['PHASE_BEGIN','RESERVE','ENTER','COMPLETE','COMPLETE']),('cancel_entered',['PHASE_BEGIN','RESERVE','ENTER','CANCEL'])]:
    def test(self,o=ops):
        with self.assertRaises(V.Invalid):self.check(self.rows(o))
    setattr(Trace,'test_'+name,test)

class SnapshotAndAPI(unittest.TestCase):
    def test_snapshot_store_id_and_version_mutations(self):
        for statement in ("UPDATE metadata SET store_id='OTHER'","UPDATE metadata SET version=2"):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'FAKE_ONLY_claims.sqlite3';shutil.copyfile(CASES/'normal/FAKE_ONLY_claims.sqlite3',p)
                db=sqlite3.connect(p);db.execute(statement);db.commit();db.close()
                inner=json.loads((CASES/'normal/inner_contract.json').read_text())
                with self.assertRaisesRegex(V.Invalid,'BINDING'):I.admission_snapshot(p,inner)
    def test_snapshot_binding_is_not_transition_history(self):
        inner=json.loads((CASES/'normal/inner_contract.json').read_text());r=I.admission_snapshot(CASES/'normal/FAKE_ONLY_claims.sqlite3',inner)
        self.assertFalse(r['snapshot_proves_atomic_history']);self.assertFalse(r['real_authorization_verified'])
    def test_unknown_parent_cpu_zero_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'parent_crash';shutil.copytree(CASES/'parent_crash',p);o=json.loads((p/'observations.json').read_text());o['host_total_cpu_ns']=0;(p/'observations.json').write_text(json.dumps(o))
            self.assertEqual(A.verify(p,I.Reference(TREE,LEGACY))['status'],'INTEGRITY_MISMATCH')
    def test_truncated_gzip_returns_machine_integrity_status(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'normal';shutil.copytree(CASES/'normal',p);archive=p/'artificial_raw.tar.gz';archive.write_bytes(archive.read_bytes()[:16])
            self.assertEqual(A.verify(p,I.Reference(TREE,LEGACY))['status'],'INTEGRITY_MISMATCH')
    def test_cli_fresh_output_only(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'reports';v,h=CLI.run(CASES/'normal',TREE,LEGACY,out);self.assertEqual(v['status'],'VERIFIED_COMPLETE');self.assertEqual(len(list(out.iterdir())),4)
            scope=next(c for c in h['checks'] if c['check_id']=='PILOT_SCOPE_ADOPTION');self.assertTrue(scope['formal_adoption_required']);self.assertEqual(scope['judgment'],'PENDING_EVIDENCE')
            with self.assertRaises(V.Invalid):CLI.run(CASES/'normal',TREE,LEGACY,out)
    def test_bounded_trace_memory(self):
        peaks=[]
        for n in (1000,40000):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'ops';encode_trace(p,(dict(seq=i,run_id='ART',op='WRITE_RESERVE',channel='log',amount=0) for i in range(n)))
                tasks,_=G.plan(16);tracemalloc.start();reader=T.Decisions(p,tasks,'ART');G.check_operations(reader,tasks,'ART');_,peak=tracemalloc.get_traced_memory();tracemalloc.stop();peaks.append(peak)
                self.assertEqual(reader.stats['rows'],n)
        self.assertLess(peaks[1],peaks[0]+1048576)
        if os.environ.get('B3_PILOT_MEMORY_REPORT'):Path(os.environ['B3_PILOT_MEMORY_REPORT']).write_text(json.dumps(dict(rows=[1000,40000],peak_traced_bytes=peaks,bounded=True))+'\n')


class GuardSource(unittest.TestCase):
    def parsed(self):return ast.parse((TREE/I.CANDIDATE_SOURCE).read_text())
    def test_production_helpers_always_deny_without_imports(self):
        nodes=[n for n in self.parsed().body if getattr(n,'name',None) in ('production_entry','ProductionFactory','ProductionBits')]
        class Denied(Exception):pass
        namespace={'A':SimpleNamespace(A=SimpleNamespace(Denied=Denied))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'fixed-pilot-production-guard','exec'),namespace)
        for fn in (namespace['production_entry'],namespace['ProductionFactory'](),namespace['ProductionBits']()):
            for flag in (False,True):
                with self.assertRaisesRegex(Denied,'NO_SCIENCE_ENTRY'):fn(adopted=flag,run_ready=flag)
    def guard_fragment(self):
        fn=next(n for n in self.parsed().body if getattr(n,'name',None)=='execute_artificial')
        statements=[]
        for n in ast.walk(fn):
            if isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Name) and i.context_expr.id=='port' for i in n.items):
                statements=n.body[:3];break
        self.assertEqual(len(statements),3)
        namespace={}
        code=compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),'fixed-artificial-factory-guard-fragment','exec')
        return code
    def check_fragment(self,root,factory,bits):
        def require(ok,reason):
            if not ok:raise V.Invalid(reason)
        fixture=root/'fixture.py';inner=dict(runtime_manifest=dict(runtime_root=str(root),artificial_fixture=dict(path='fixture.py')),input_pins={'artificial-fixture.py':V.file_digest(fixture)})
        ns=dict(endpoint_factory=factory,bits_factory=bits,inner=inner,Path=Path,A=SimpleNamespace(require=require,A=SimpleNamespace(digest=lambda b:sha256(b).hexdigest())))
        exec(self.guard_fragment(),ns)
    def test_factory_metadata_without_matching_filename_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'fixture.py').write_text('# PINNED ARTIFICIAL FIXTURE\n')
            class Factory:
                artificial=True
                def __call__(self,*args):raise AssertionError('Must never be dispatched')
            def bits(*args):raise AssertionError('Must never be called')
            with self.assertRaisesRegex(V.Invalid,'FACTORY_SOURCE_BINDING'):self.check_fragment(root,Factory(),bits)
            Factory.artificial=False
            with self.assertRaisesRegex(V.Invalid,'ENDPOINT_REQUIRED'):self.check_fragment(root,Factory(),bits)
    def test_filename_is_not_callable_code_authentication(self):
        # Harmless reproducible limitation of the exact guard fragment; no
        # whole admission/dispatch or scientific bypass is claimed.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture=root/'fixture.py';fixture.write_text('# PINNED ARTIFICIAL FIXTURE\n')
            ns={};exec(compile('def call(self,*args): return "HARMLESS_SENTINEL"\ndef bits(*args): return "HARMLESS_BITS"\n',str(fixture),'exec'),ns)
            factory=type('ForgedFilenameFactory',(),{'artificial':True,'__call__':ns['call']})()
            self.check_fragment(root,factory,ns['bits']);self.assertEqual(factory(),'HARMLESS_SENTINEL')
            if os.environ.get('B3_PILOT_GUARD_REPORT'):
                Path(os.environ['B3_PILOT_GUARD_REPORT']).write_text(json.dumps(dict(source=I.CANDIDATE_SOURCE,case='CO_FILENAME_IS_ASSERTED_METADATA',accepted_guard_fragment=True,pinned_file_bytes_unchanged=True,whole_entrypoint_bypass_proven=False,guard_fragment_only=True,scientific_calls=0,required_boundary='Trusted caller/fixture identity must be stated; co_filename alone is not cryptographic code identity'))+'\n')


class FreshReplays(unittest.TestCase):
    def ref_for(self,name):
        p=PACKET/'artificial_replays'/name;c=json.loads((p/'pilot_contract.json').read_text());ref=I.Reference(TREE,LEGACY)
        normalized=deepcopy(c)
        for key in ('runtime_root','python'):normalized[key]=ref.template[key]
        self.assertEqual(normalized,ref.template)
        # Explicit new local runtime identity, never a replacement for the
        # historical public Python bytes; policy/source/input bytes identical.
        ref.template=c
        return p,ref
    def test_new_S_projection_is_separate_complete_artificial_evidence(self):
        p,ref=self.ref_for('fresh_S_projection');r=A.verify(p,ref)
        self.assertEqual(r['status'],'VERIFIED_COMPLETE',r['issues']);self.assertEqual(r['calls']['completed'],768)
        self.assertEqual(r['cost']['variant'],'SF');self.assertFalse(r['historical_pending_reclassified'])
    def test_producer_trace_write_failure_underreports_reserved_slot(self):
        p,ref=self.ref_for('fresh_trace_reserve_write_failure');r=A.verify(p,ref)
        self.assertEqual(r['status'],'INTEGRITY_MISMATCH');self.assertIn('GLOBAL_STATE_COUNTER_BINDING',r['issues'][0])
        report=json.loads((p/'observations.json').read_text())['report'];self.assertFalse(report['complete'])
        self.assertEqual(report['reserved'],0);self.assertEqual(report['states']['START_RESERVED'],1)
        self.assertEqual(report['started_observed'],0)

if __name__=='__main__':unittest.main()
