"""Independent consumer mutations of pinned artificial bytes, never Producer import."""
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import importlib.util,json,os,shutil,sqlite3,sys,tarfile,tempfile,unittest
ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019'
spec=importlib.util.spec_from_file_location('independent_run',D/'run_acceptance.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
A,I=R.A,R.I
RESULTS=[]
class Mutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ref=R.reference()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='b3-v2-independent-mutation-');self.addCleanup(self.tmp.cleanup)
        self.base=D/'restored/fault_cases_final/S_normal';self.case=Path(self.tmp.name)/'case';shutil.copytree(self.base,self.case)
        self.out=A.B.extract_archive(self.case/'artificial_raw.tar.gz',Path(self.tmp.name)/'raw')
        self.c=json.loads((self.case/'pilot_contract.json').read_text());self.inner=json.loads((self.case/'inner_contract.json').read_text());self.report=A.A.safe_values(A.V.plain((self.case/'observations.json').read_bytes()))['report']
    def verify(self):return A.verify_output(self.out,self.c,self.inner,self.report,self.ref)
    def reject(self,expected=None):
        with self.assertRaises((A.V.Invalid,A.V.Unsupported,KeyError,ValueError,TypeError)) as ctx:self.verify()
        issue=str(ctx.exception)
        if expected:self.assertIn(expected,issue)
        if target:=os.environ.get('B3_NEGATIVE_EVIDENCE_DIR'):
            dest=Path(target)/self._testMethodName;dest.mkdir(parents=True,exist_ok=False)
            for name,value in [('pilot_contract.json',self.c),('inner_contract.json',self.inner),('observations.json',dict(report=self.report))]:
                (dest/name).write_text(json.dumps(value,default=lambda x:{'exact':A.V.canonical(x)})+'\n')
            shutil.copyfile(self.case/'FAKE_ONLY_claims.sqlite3',dest/'FAKE_ONLY_claims.sqlite3')
            with tarfile.open(dest/'artificial_raw.tar.gz','w:gz') as tf:tf.add(self.out,arcname='artificial-output')
        RESULTS.append(dict(test=self.id(),kind='independent_offline_mutation',base_case=self.base.name,base_archive_sha256=A.V.file_digest(self.base/'artificial_raw.tar.gz'),observed_rejection=issue,successful=True))
    def trace(self,change):
        path=self.out/'operations.jsonl';rows=[json.loads(json.loads(x)['payload']) for x in path.read_bytes().splitlines()];change(rows)
        previous='0'*64;lines=[]
        for n,row in enumerate(rows):
            row['seq']=n;payload=json.dumps(row,separators=(',',':'),ensure_ascii=True);digest=sha256(previous.encode()+payload.encode()).hexdigest();lines.append(json.dumps(dict(payload=payload,previous=previous,sha256=digest),separators=(',',':')).encode()+b'\n');previous=digest
        raw=b''.join(lines);path.write_bytes(raw);self.report['operation_trace_sha256']=sha256(raw).hexdigest();self.report['evidence_v2']['trace_prefix']=dict(bytes=len(raw),rows=len(lines),chain=previous,sha256=sha256(raw).hexdigest())
    def test_pristine_S(self):self.assertEqual(self.verify()['status'],'VERIFIED_ARTIFICIAL_COMPLETE')
    def test_pristine_P1(self):self.assertEqual(A.verify(D/'restored/fault_cases_final/P1_normal',self.ref)['status'],'VERIFIED_ARTIFICIAL_COMPLETE')
    def test_source_pin(self):self.c['source_pins'][I.SOURCE]='0'*64;self.reject('V2_FIXED_CANDIDATE')
    def test_candidate_schema(self):self.c['schema']='b3_pilot_candidate_v1';self.reject()
    def test_result_schema(self):self.report['schema']='b3_parallel_result_v3';self.reject('V2_RESULT_SCHEMA')
    def test_inner_binding(self):self.inner['pilot_candidate_binding']='0'*64;self.reject()
    def test_inner_source(self):self.inner['source_pins'][next(iter(self.inner['source_pins']))]='0'*64;self.reject()
    def test_scope(self):self.inner['scope']='ARTIFICIAL_OTHER';self.reject()
    def test_claim(self):self.inner['claim_id']='ARTIFICIAL_OTHER';self.reject()
    def test_runtime(self):self.inner['runtime_manifest']['python']['executable_sha256']='0'*64;self.reject()
    def test_pair_offset(self):self.report['pairs'][0]['task']['offset']+=1;self.reject()
    def test_pair_duplicate(self):self.report['pairs'][1]=deepcopy(self.report['pairs'][0]);self.reject()
    def test_counter(self):self.report['reserved']-=1;self.reject('V2_RAM_CALL_ACCOUNTING')
    def test_started_inference(self):self.report['started_observed']-=1;self.reject()
    def test_cursor(self):self.report['evidence_v2']['next_indices'][self.report['contract_order'][0]]-=1;self.reject('V2_CURSOR_RESERVED_BINDING')
    def test_ticket(self):self.report['evidence_v2']['current_tickets']=[dict(index=0)];self.reject('V2_FINAL_TICKETS')
    def test_state_reuse(self):p=self.out/'trial_states.bin';x=bytearray(p.read_bytes());x[0]=0;p.write_bytes(x);self.reject()
    def test_unknown_promoted(self):p=self.out/'trial_states.bin';x=bytearray(p.read_bytes());x[-1]=7;p.write_bytes(x);self.reject()
    def test_trace_chain(self):p=self.out/'operations.jsonl';x=bytearray(p.read_bytes());x[35]^=1;p.write_bytes(x);self.reject()
    def test_prefix_bytes(self):self.report['evidence_v2']['trace_prefix']['bytes']-=1;self.reject()
    def test_prefix_rows(self):self.report['evidence_v2']['trace_prefix']['rows']-=1;self.reject()
    def test_prefix_chain(self):self.report['evidence_v2']['trace_prefix']['chain']='0'*64;self.reject()
    def test_prefix_hash(self):self.report['evidence_v2']['trace_prefix']['sha256']='0'*64;self.reject()
    def test_partial_tail(self):p=self.out/'operations.jsonl';p.write_bytes(p.read_bytes()[:-1]);self.reject()
    def test_duplicate_reserve(self):
        def change(rows):n=next(i for i,r in enumerate(rows) if r['op']=='RESERVE');rows.insert(n+1,deepcopy(rows[n]))
        self.trace(change);self.reject()
    def test_duplicate_complete(self):
        def change(rows):n=next(i for i,r in enumerate(rows) if r['op']=='COMPLETE');rows.insert(n+1,deepcopy(rows[n]))
        self.trace(change);self.reject()
    def test_global_offset(self):
        def change(rows):next(r for r in rows if r['op']=='RESERVE')['global_offset']=999
        self.trace(change);self.reject('V2_GLOBAL_OFFSET')
    def test_trial_identity(self):
        def change(rows):next(r for r in rows if r['op']=='RESERVE')['trial_id']='wrong'
        self.trace(change);self.reject('V2_TRIAL_CURSOR_ID')
    def test_enter_after_stop(self):
        def change(rows):n=next(i for i,r in enumerate(rows) if r['op']=='ENTER');rows.insert(n,dict(rows[n],op='STOP',parent_arrival=1))
        self.trace(change);self.reject('ENTER_AFTER_STOP')
    def test_stop_before_new_reserve(self):
        def change(rows):n=next(i for i,r in enumerate(rows) if r['op']=='RESERVE');rows.insert(n,dict(rows[n],op='STOP',parent_arrival=1))
        self.trace(change);self.reject('V2_NEW_RESERVATION_AFTER_STOP')
    def test_refund_unacked(self):self.report['output']['unknown_credit']['result']=1;self.reject('V2_UNACKED_CREDIT_REFUND')
    def test_ack_exceeds_reserved(self):self.report['output']['observed']['log']+=1;self.reject()
    def test_credit_decreased(self):self.report['output']['reserved']['result']-=1;self.reject()
    def test_byte_limit_changed(self):self.report['output']['limits']['log']-=1;self.reject()
    def test_trace_self_double_charge(self):self.report['output']['reserved']['log']+=self.report['evidence_v2']['trace_prefix']['bytes'];self.reject()
    def test_diagnostic_free(self):n=(self.out/'final_diagnostic.json').stat().st_size;self.report['output']['reserved']['result']-=n;self.report['output']['observed']['result']-=n;self.reject()
    def test_receipt_missing(self):(self.out/'pilot_receipt.json').unlink();self.reject()
    def test_summary_missing(self):(self.out/'pilot_summary.jsonl').unlink();self.reject()
    def test_diagnostic_missing(self):(self.out/'final_diagnostic.json').unlink();self.reject()
    def test_diagnostic_success(self):p=self.out/'final_diagnostic.json';x=json.loads(p.read_text());x['final_acceptance']=True;p.write_text(json.dumps(x));self.reject()
    def test_marker_missing(self):p=self.out/'control.jsonl';p.write_bytes(b''.join(x for x in p.read_bytes().splitlines(keepends=True) if json.loads(x)['event']!='PILOT_COMPLETION_RECEIPT'));self.reject()
    def test_marker_moved_first(self):p=self.out/'control.jsonl';lines=p.read_bytes().splitlines(keepends=True);p.write_bytes(b''.join([lines[-1]]+lines[:-1]));self.reject()
    def test_receipt_summary_hash(self):p=self.out/'pilot_receipt.json';x=json.loads(p.read_text());x['summary_sha256']='0'*64;p.write_text(json.dumps(x));self.reject()
    def test_cpu_double_count(self):self.report['total_cpu_ns']+=1;self.reject()
    def test_parent_cost_zero(self):self.report['host_outer']['parent_cpu_ns']=0;self.reject()
    def test_wall_zero(self):self.report['host_outer']['wall_ns']=0;self.reject()
    def test_admission_save_failure_hidden(self):self.report['admission_stop']['record_errors']=['failure'];self.reject()
    def test_close_failure_hidden(self):self.report['evidence_v2']['storage_errors']=[dict(storage='state',op='CLOSE')];self.reject()
    def test_input_readonly(self):
        before={p.name:A.V.file_digest(p) for p in self.out.iterdir()};self.verify();self.assertEqual(before,{p.name:A.V.file_digest(p) for p in self.out.iterdir()})
    def test_unknown_cost_not_zero(self):
        r=A.verify(D/'restored/fault_cases_final/P1_parent_crash',self.ref);self.assertEqual(r['status'],'INCOMPLETE_EVIDENCE');self.assertIsNone(r['cost']['total_cpu_ns']);self.assertFalse(r['cost']['unknown_cost_is_zero'])
    def test_missing_physical_not_promoted(self):
        r=A.verify(D/'restored/fault_cases_final/S_state_write_STARTED',self.ref);self.assertEqual(r['status'],'INCOMPLETE_EVIDENCE');self.assertFalse(r['ram_claim']['independently_physically_proven'])
    def test_complete_record_failure_unknown(self):
        r=A.verify(D/'restored/fault_cases_final/S_trace_write_COMPLETE',self.ref);self.assertEqual(r['status'],'INCOMPLETE_EVIDENCE');self.assertEqual(r['ram_claim']['states']['COMPLETED'],0);self.assertGreater(r['ram_claim']['states']['TERMINATED_WITH_UNKNOWN_RESULT'],0)
    def test_fake_ledger_scope(self):
        p=self.case/'FAKE_ONLY_claims.sqlite3';db=sqlite3.connect(p);db.execute("UPDATE claims SET scope='ARTIFICIAL_TAMPERED'");db.commit();db.close();r=A.verify(self.case,self.ref);self.assertEqual(r['status'],'INTEGRITY_MISMATCH');RESULTS.append(dict(test=self.id(),kind='fake_snapshot_mutation',observed_rejection=r['issues'],successful=True))
    def test_fake_ledger_claim(self):
        p=self.case/'FAKE_ONLY_claims.sqlite3';db=sqlite3.connect(p);db.execute("UPDATE claims SET claim_id='ARTIFICIAL_TAMPERED'");db.commit();db.close();r=A.verify(self.case,self.ref);self.assertEqual(r['status'],'INTEGRITY_MISMATCH');RESULTS.append(dict(test=self.id(),kind='fake_snapshot_mutation',observed_rejection=r['issues'],successful=True))

    def test_diagnostic_ACK_after_marker(self):
        def change(rows):
            n=max(i for i,r in enumerate(rows) if r['op']=='WRITE_ACK' and r['channel']=='result');rows.append(rows.pop(n))
        self.trace(change);self.reject('V2_DIAGNOSTIC_ACK_BEFORE_SUCCESS_MARKER')
    def test_original_reserve_defect_reintroduced_in_report(self):
        base=D/'restored/fault_cases_final/S_original_reserve_writer';self.base=base
        shutil.copyfile(base/'FAKE_ONLY_claims.sqlite3',self.case/'FAKE_ONLY_claims.sqlite3')
        self.out=A.B.extract_archive(base/'artificial_raw.tar.gz',Path(self.tmp.name)/'fault_raw')
        self.c=json.loads((base/'pilot_contract.json').read_text());self.inner=json.loads((base/'inner_contract.json').read_text());self.report=A.A.safe_values(A.V.plain((base/'observations.json').read_bytes()))['report']
        self.report['reserved']=0;self.report['states']['NOT_STARTED']=768;self.report['states']['CANCELLED_BEFORE_START']=0;self.report['evidence_v2']['next_indices']={k:0 for k in self.report['evidence_v2']['next_indices']}
        self.reject('GLOBAL_STATE_COUNTER_BINDING')

    def test_trace_noncanonical_envelope(self):
        p=self.out/'operations.jsonl';raw=p.read_bytes();raw=b'{ '+raw[1:];p.write_bytes(raw)
        self.report['operation_trace_sha256']=sha256(raw).hexdigest();self.report['evidence_v2']['trace_prefix']['bytes']=len(raw);self.report['evidence_v2']['trace_prefix']['sha256']=sha256(raw).hexdigest()
        self.reject('V2_TRACE_CANONICAL_ENVELOPE')
