"""Independent fixed v3 artificial raw plus consumer-only corruptions. No science."""
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ast,io,json,os,shutil,tarfile,tempfile,tracemalloc,unittest
from unittest.mock import patch
import b3_parallel_verifier_fixtures as F
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_offline_adapter as O
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_verifier_core as G
from trottertracks.algorithm_codesign import g6_w2c_b3_v3_offline_adapter as A
from trottertracks.algorithm_codesign import g6_w2c_b3_v3_offline_identity as I
from trottertracks.algorithm_codesign import g6_w2c_b3_v3_offline_cost as C
from trottertracks.algorithm_codesign import g6_w2c_b3_v3_offline_acceptance as H
from trottertracks.algorithm_codesign import g6_w2c_b3_v3_offline_cli as CLI
ROOT=Path(__file__).resolve().parents[3]
PACKET=ROOT/'docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019'
REF=PACKET/'fixed_reference'

class Published(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths=json.loads((PACKET/'published_cases.json').read_text())
        with ThreadPoolExecutor() as pool:
            cls.results=dict(pool.map(lambda m:(m.split('/')[1],A.verify(PACKET/m,REF)),paths))
    def test_all_47_consumed(self):self.assertEqual(len(self.results),47)
    def test_failure_without_contract_is_not_identity_pass(self):
        r=self.results['fused-angle'];self.assertFalse(r['binding_evidence']['complete_contract_present'])
        self.assertIsNone(r['binding_evidence']['independently_recomputed'])
    def test_flush_row_is_not_commit(self):
        r=self.results['fused-flush_in_emit'];rows=[p for p in r['pairs'] if p.get('failure')]
        self.assertTrue(any(p['visible_trial_rows']>p['engine_committed_trials'] for p in rows))
    def test_repeated_failure_notices_are_not_calls(self):
        r=self.results['fused-sample'];self.assertTrue(any(n>1 for n in r['failure_notifications_by_trial'].values()))
        self.assertEqual(r['calls']['reserved'],r['report']['reserved'])
    def test_parent_crash_unknown_cost_is_null(self):
        r=self.results['fused-parent_crash'];c=C.aggregate(r)
        self.assertGreater(r['calls']['unknown'],0);self.assertIsNone(c['total_cpu_ns'])
        self.assertFalse(c['comparison_eligible']);self.assertFalse(r['control']['receipt_present'])
    def test_unacked_credit_is_not_refunded(self):
        r=self.results['fused-unacked_exit'];self.assertTrue(any(v['unknown_credit']>0 for v in r['output_budget'].values()))
        self.assertFalse(r['calls']['unknown_credit_refunded'])
    def test_p08_pending_in_all_public_cases(self):
        for r in self.results.values():
            h=H.matrix(r,C.aggregate(r));self.assertIn('P08',h['pending']);self.assertIn('P07',h['pending']);self.assertIn('P11',h['pending'])
    def test_p_pf_s_sf_definitions(self):
        for name in ('P','PF','S','SF'):
            c=C.aggregate(self.results['light-'+name+'-0']);self.assertEqual(c['variant'],name)
            self.assertEqual(c['measurement_condition'],'sequential_pairs' if name.startswith('S') else 'concurrent_pairs')
    def test_log_cap_subsequence_remains_pending(self):
        r=self.results['fused-log_cap'];self.assertTrue(r['control']['evidence_gaps'])
        self.assertIn('P06',H.matrix(r,C.aggregate(r))['pending'])

for name in [p.split('/')[1] for p in json.loads((PACKET/'published_cases.json').read_text())]:
    def test(self,n=name):
        r=self.results[n];expected='VERIFIED_COMPLETE' if n.endswith('-0') else 'TECHNICAL_FAILURE'
        self.assertEqual(r['status'],expected,r['issues'])
        c=C.aggregate(r)
        self.assertEqual(c['status'],'INSUFFICIENT_EVIDENCE' if n.endswith('parent_crash') else 'VERIFIED_PARALLEL_COST_BOUNDARIES',c.get('issues'))
    setattr(Published,'test_raw_'+name.replace('-','_'),test)

class Mutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='b3-v3-independent-master-')
        cls.base=PACKET/'published_cases/light-PF-0'
        cls.output=A.extract_archive(cls.base/'raw.tar.gz',Path(cls.temp.name)/'raw')
        cls.original_report=O.safe_values(json.loads((cls.base/'host_report.json').read_text()))
        cls.original_contract=json.loads((cls.base/'contract.json').read_text());cls.ref=I.Reference(REF)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):
        t=tempfile.TemporaryDirectory(prefix='b3-v3-independent-mutation-');self.addCleanup(t.cleanup)
        self.root=Path(t.name);self.out=self.root/'artificial-output';shutil.copytree(self.output,self.out)
        self.report=deepcopy(self.original_report);self.contract=deepcopy(self.original_contract)
        self.m=dict(run_id='INDEPENDENT_ARTIFICIAL_MUTATION',_input_root=str(self.root))
    def verify(self):return A._verify_directory(self.out,self.m,self.report,self.contract,self.ref)
    def bad(self):
        with self.assertRaises((V.Invalid,V.Unsupported,KeyError,ValueError)):self.verify()
    def control(self,change):
        p=self.out/'control.jsonl';rows=[r for r,_ in O.records(p)];change(rows)
        p.write_text(''.join(json.dumps(r,default=lambda f:{'exact':V.canonical(f)})+'\n' for r in rows))
    def test_pristine_copy(self):self.assertEqual(self.verify()['status'],'VERIFIED_COMPLETE')
    def test_canonical_hash_byte_flip(self):
        p=self.out/(self.report['pairs'][0]['task']['key']+'.jsonl');b=p.read_bytes();p.write_bytes(b[:30]+bytes([b[30]^1])+b[31:]);self.bad()
    def test_tail_truncation(self):
        p=self.out/(self.report['pairs'][0]['task']['key']+'.jsonl');p.write_bytes(p.read_bytes()[:-9]);self.bad()
    def test_file_sha_changed(self):self.report['pairs'][0]['file_sha']='0'*64;self.bad()
    def test_state_counter_changed(self):self.report['states']['COMPLETED']-=1;self.bad()
    def test_duplicate_pair(self):self.report['pairs'][1]=deepcopy(self.report['pairs'][0]);self.bad()
    def test_wrong_global_offset(self):self.report['pairs'][1]['task']['offset']+=1;self.bad()
    def test_bool_is_not_global_offset_zero(self):self.report['pairs'][0]['task']['offset']=False;self.bad()
    def test_wrong_global_trial_snapshot(self):
        self.report['failures']=[dict(parent_arrival=1,parent_observed_ns=1,inflight_snapshot={'wrong':{}})];self.bad()
    def test_statefile_tail(self):
        p=self.out/'trial_states.bin';p.write_bytes(p.read_bytes()[:-1]);self.bad()
    def test_statefile_completed_gap(self):
        p=self.out/'trial_states.bin';b=bytearray(p.read_bytes());b[0]=0;p.write_bytes(b);self.bad()
    def test_source_pin(self):self.contract['source_pins'][I.ROLES['host']]='0'*64;self.bad()
    def test_runtime_schema_pin(self):self.contract['runtime_manifest']['schema_pins'][next(iter(self.ref.schema_pins))]='0'*64;self.bad()
    def test_runtime_entrypoint(self):self.contract['runtime_manifest']['entrypoint']['api']='other';self.bad()
    def test_source_runtime_import_role(self):self.contract['runtime_manifest']['import_locations']['host']['sha256']='0'*64;self.bad()
    def test_schema_retry_bool_is_not_zero(self):self.contract['retry']=False;self.bad()
    def test_contract_retry(self):self.contract['retry']=1;self.bad()
    def test_contract_resume(self):self.contract['resume']=True;self.bad()
    def test_nested_r1_retry(self):self.contract['r1_reference']['retry']=1;self.bad()
    def test_real_scope_forbidden(self):self.contract['scope']='B3_REAL';self.bad()
    def test_scientific_policy_changed_in_both_copies(self):
        for c in (self.contract['candidate'],self.contract['r1_reference']['candidate']):c['input']['m']=6
        self.bad()
    def test_receipt_digest(self):
        self.control(lambda rows:rows[-1].update(summary_sha='0'*64));self.bad()
    def test_receipt_missing(self):
        p=self.out/'control.jsonl';lines=p.read_bytes().splitlines(keepends=True);p.write_bytes(b''.join(lines[:-1]))
        for k in ('reserved','observed'):self.report['output'][k]['log']=p.stat().st_size+self.report['planned']
        self.assertEqual(self.verify()['status'],'INSUFFICIENT_EVIDENCE')
    def test_missing_contract_not_complete(self):self.contract=None;self.assertEqual(self.verify()['status'],'INSUFFICIENT_EVIDENCE')
    def test_ack_over_physical(self):self.report['output']['observed']['result']+=1;self.bad()
    def test_refunded_credit(self):self.report['output']['reserved']['result']-=1;self.bad()
    def test_summary_coordinator_cost_changed(self):self.report['parent_cpu_ns']+=1;self.bad()
    def test_stop_save_failure_false_complete(self):
        self.report['secondary_stop_error']='injected';self.assertEqual(self.verify()['status'],'TECHNICAL_FAILURE')
    def test_parent_crash_fake_complete(self):
        self.report=dict(schema='b3_parallel_crash_observation_v3',complete=True);self.bad()
    def test_unknown_crash_cost_zero(self):
        self.report=dict(schema='b3_parallel_crash_observation_v3',complete=False,mandatory_STOP=True,retry=0,resume=False,
                         started_observed=None,completed=None,parent_cpu_ns=0,worker_cpu_ns=None,total_cpu_ns=None)
        self.bad()
    def test_unknown_output_collision(self):
        (self.out/'unexpected.jsonl').write_bytes(b'');self.bad()
    def test_normalized_stop_after_enter_allows_completion(self):
        tasks,_=G.plan(16);t=tasks[0];tid=V.trial(dict(t,N=t['n']),'X',0)
        rows=[dict(op='PHASE_BEGIN',phase=0),dict(op='RESERVE',task_key=t['key'],trial_id=tid),dict(op='ENTER',task_key=t['key'],trial_id=tid),dict(op='STOP'),dict(op='COMPLETE',task_key=t['key'],trial_id=tid)]
        rows=[dict(r,seq=i,run_id='ART') for i,r in enumerate(rows)]
        self.assertEqual(G.check_operations(rows,tasks,'ART')['counts']['completed'],1)
    def test_normalized_enter_after_stop_rejected(self):
        tasks,_=G.plan(16);t=tasks[0];tid=V.trial(dict(t,N=t['n']),'X',0)
        rows=[dict(op='PHASE_BEGIN',phase=0),dict(op='RESERVE',task_key=t['key'],trial_id=tid),dict(op='STOP'),dict(op='ENTER',task_key=t['key'],trial_id=tid)]
        rows=[dict(r,seq=i,run_id='ART') for i,r in enumerate(rows)]
        with self.assertRaisesRegex(V.Invalid,'ENTER_AFTER_STOP'):G.check_operations(rows,tasks,'ART')
    def test_normalized_duplicate_trial_reservation(self):
        tasks,_=G.plan(16);rows=list(F.operations(tasks));rows[2]=dict(rows[1],seq=2)
        with self.assertRaises(V.Invalid):G.check_operations(rows,tasks,'INDEPENDENT_ARTIFICIAL')
    def test_cpu_pair_added_twice(self):
        v=self.verify();v['report']['total_cpu_ns']+=1;self.assertEqual(C.aggregate(v)['status'],'COST_INTEGRITY_MISMATCH')
    def test_execution_outer_exceeds_host(self):
        v=self.verify();v['report']['execution_outer']['parent_cpu_ns']=v['report']['host_outer']['parent_cpu_ns']+1
        self.assertEqual(C.aggregate(v)['status'],'COST_INTEGRITY_MISMATCH')
    def test_unknown_observer_remains_null(self):
        c=C.aggregate(self.verify());self.assertIsNone(c['observer_cost_ns']);self.assertFalse(c['unknown_cost_is_zero'])
    def test_inputs_not_written(self):
        before={p.name:V.file_digest(p) for p in self.out.iterdir()}
        orig=Path.open
        def opened(p,mode='r',*args,**kw):
            if p.is_relative_to(self.out):self.assertIn(mode,('r','rb'))
            return orig(p,mode,*args,**kw)
        with patch.object(Path,'open',opened):self.verify()
        self.assertEqual(before,{p.name:V.file_digest(p) for p in self.out.iterdir()})

class API(unittest.TestCase):
    def test_v3_failed_reserved_slot_does_not_invent_start(self):
        counts={s:0 for s in G.STATES};counts.update(NOT_STARTED=767,FAILED=1)
        report=dict(planned=768,states=counts,reserved=1,started_observed=0,settings=dict(call_limit=31710848))
        r=A.accounting_v3(dict(counts=counts),report,768)
        self.assertEqual(r['started_observed'],0);self.assertEqual(r['failed'],1)
    def test_v3_starts_cannot_exceed_failed_unknown_bound(self):
        counts={s:0 for s in G.STATES};counts.update(NOT_STARTED=767,FAILED=1)
        report=dict(planned=768,states=counts,reserved=1,started_observed=2,settings=dict(call_limit=31710848))
        with self.assertRaises(V.Invalid):A.accounting_v3(dict(counts=counts),report,768)
    def test_cli_seven_outputs_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'reports';v,c,h=CLI.run(PACKET/'published_cases/light-PF-0/manifest.json',REF,out)
            self.assertEqual(len(list(out.iterdir())),7);self.assertEqual(v['status'],'VERIFIED_COMPLETE')
            with self.assertRaises(V.Invalid):CLI.run(PACKET/'published_cases/light-PF-0/manifest.json',REF,out)
    def test_cli_no_input_output_overlap(self):
        with self.assertRaises(V.Invalid):CLI.run(PACKET/'published_cases/light-PF-0/manifest.json',REF,REF/'new-output')
    def test_archive_traversal_link_and_duplicate(self):
        for kind in ('traversal','link','duplicate'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'bad.tar.gz'
                with tarfile.open(p,'w:gz') as f:
                    m=tarfile.TarInfo('artificial-output/../escape' if kind=='traversal' else 'artificial-output/test')
                    if kind=='link':m.type=tarfile.SYMTYPE;m.linkname='/tmp/target';f.addfile(m)
                    else:
                        m.size=1;f.addfile(m,io.BytesIO(b'x'))
                        if kind=='duplicate':f.addfile(m,io.BytesIO(b'x'))
                with self.assertRaises(V.Invalid):A.extract_archive(p,Path(d)/'out')
    def test_unsupported_manifest_and_archive_hash(self):
        for field,value,status in [('schema','other','UNSUPPORTED_SCHEMA'),('archive_sha256','0'*64,'INTEGRITY_MISMATCH')]:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);src=PACKET/'published_cases/light-PF-0';shutil.copytree(src,root/'in')
                p=root/'in/manifest.json';m=json.loads(p.read_text());m[field]=value;p.write_text(json.dumps(m))
                self.assertEqual(A.verify(p,REF)['status'],status)
    def test_fixed_schema_assertion_vocabulary_complete(self):
        r=I.Reference(REF);seen=set()
        def visit(s):
            seen.update(s)
            for k in ('properties',):
                for x in s.get(k,{}).values():visit(x)
            for k in ('items','additionalProperties'):
                if isinstance(s.get(k),dict):visit(s[k])
            for x in s.get('anyOf',[]):visit(x)
        for name in I.NAMES:visit(r.schemas[name])
        self.assertLessEqual(seen,{'$id','$schema','$ref','title','description','type','properties','required','additionalProperties','items','anyOf','const','enum','minimum','pattern'})
    def test_static_producer_one_shot_path_only(self):
        tree=REF/'tree/src/trottertracks/algorithm_codesign'
        host=ast.parse((tree/'g6_w2c_b3_parallel_host.py').read_text());admission=(tree/'g6_w2c_b3_parallel_admission.py').read_text()
        funcs={n.name for n in ast.walk(host) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        self.assertIn('execute_approved_v3',funcs)
        self.assertIn('reserve',admission);self.assertIn('resume',admission);self.assertIn('retry',admission)
        # Static SOURCE structure is not persisted v3 store history or an execution claim.
    def test_archive_streaming_bounded_memory(self):
        peaks=[]
        for size in (2*1024*1024,32*1024*1024):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);p=root/'raw.tar.gz';block=b'x'*65536
                # Build an artificial raw member without all bytes in RAM.
                raw=root/'member';
                with raw.open('wb') as f:
                    for _ in range(size//len(block)):f.write(block)
                with tarfile.open(p,'w:gz') as f:f.add(raw,arcname='artificial-output/member')
                tracemalloc.start();A.extract_archive(p,root/'out');_,peak=tracemalloc.get_traced_memory();tracemalloc.stop();peaks.append(peak)
        self.assertLess(peaks[1],peaks[0]+3*1024*1024)
        if os.environ.get('B3_V3_MEMORY_REPORT'):Path(os.environ['B3_V3_MEMORY_REPORT']).write_text(json.dumps(dict(input_member_sizes=[2097152,33554432],peak_traced_bytes=peaks,bounded_growth=True))+'\n')

if __name__=='__main__':unittest.main()
