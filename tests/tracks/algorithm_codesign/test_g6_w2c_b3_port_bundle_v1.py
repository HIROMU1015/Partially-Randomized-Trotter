"""New bundle acceptance tests; historical 108/680/820 are never counted here."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import importlib.util,json,os,shutil,subprocess,sys,tempfile,unittest

from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1 import contract as K
from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1 import bridge as B
from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1 import offline as O
from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_store_v1 import FakeStore,Denied

ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/K.DOC
MUTATION_RESULTS=[]
MATRIX=Path(os.environ.get('B3_BUNDLE_MATRIX',str(DOC/'runs_final')))

class ContractAndAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=K.candidate(ROOT)
    def fresh(self):
        temp=tempfile.TemporaryDirectory(prefix='b3-bundle-artificial-');self.addCleanup(temp.cleanup);root=Path(temp.name)
        return root,FakeStore(root),K.manifest(self.c,'ARTIFICIAL_ONLY/'+root.name)
    def test_static_16896_plan(self):
        tasks,n=K.plan(256);self.assertEqual((len(tasks),n),(48,16896));self.assertEqual(O.plan(256),(tasks,n))
        for mode in ('S','P1'):
            self.assertEqual([sum(2*t['n']for t in tasks if t['condition']==mode and t['phase_index']%3==p)for p in range(3)],[256,4096,4096])
    def test_exact_one_global_atomic_claim(self):
        root,store,m=self.fresh();store.claim(m);self.assertEqual(store.snapshot(m)['claim'][3:],["SPENT",1]);store.stop(m);self.assertEqual(store.snapshot(m)['claim'][3:],["STOPPED",1])
        with self.assertRaisesRegex(Denied,'SPENT'):store.claim(m)
    def test_fake_same_scope_race_one_winner(self):
        root,store,m=self.fresh()
        def one(i):
            try:store.claim(m);return 'WIN'
            except Denied:return 'SPENT'
        with ThreadPoolExecutor(max_workers=16)as pool:result=list(pool.map(one,range(32)))
        self.assertEqual(result.count('WIN'),1);self.assertEqual(result.count('SPENT'),31);store.stop(m)
    def test_commit_ACK_unknown_spends_scope(self):
        root,store,m=self.fresh()
        def fault(p):raise OSError('ACK_UNKNOWN')
        with self.assertRaises(OSError):store.claim(m,fault)
        self.assertEqual(store.snapshot(m)['claim'][3:],["SPENT",1])
        with self.assertRaises(Denied):store.claim(m)
        store.stop(m)
    def test_retry_forbidden_before_claim(self):
        root,store,m=self.fresh();m['retry']=1
        with self.assertRaisesRegex(Denied,'RETRY'):store.claim(m)
        self.assertIsNone(store.snapshot(m)['claim'])
    def test_resume_forbidden_before_claim(self):
        root,store,m=self.fresh();m['resume']=True
        with self.assertRaisesRegex(Denied,'RESUME'):store.claim(m)
    def test_fake_store_path_cannot_redirect(self):
        root,store,m=self.fresh();store.path=root/'another.sqlite3'
        with self.assertRaisesRegex(Denied,'PATH_BINDING'):store.claim(m)
    def test_fake_store_symlink_refused(self):
        root,store,m=self.fresh();original=store.path;other=root/'fake-copy.sqlite3';shutil.copyfile(original,other);original.unlink();original.symlink_to(other)
        with self.assertRaisesRegex(Denied,'PATH_BINDING'):store.claim(m)
    def test_stop_binding_mismatch_retains_spent(self):
        root,store,m=self.fresh();store.claim(m);bad=deepcopy(m);bad['global_claim']='bad'
        with self.assertRaisesRegex(Denied,'BINDING'):store.stop(bad)
        self.assertEqual(store.snapshot(m)['claim'][3:],["SPENT",1])
    def test_production_entry_no_unlock_flag(self):
        for fn in (B.production_entry,B.ProductionFactory(),B.ProductionBits()):
            with self.assertRaisesRegex(Denied,'NO_SCIENCE_DISPATCH'):fn(run_ready=True,content_adopted=True,authority=object())
    def test_no_external_factory_parameter(self):
        root,store,m=self.fresh()
        with self.assertRaises(TypeError):B.execute_artificial(self.c,m,root=root,store=store,endpoint_factory=object())
        self.assertIsNone(store.snapshot(m)['claim'])
    def test_fake_authority_cannot_replace_store(self):
        root,store,m=self.fresh()
        with self.assertRaisesRegex(Denied,'BUILTIN_FAKE_STORE'):B.execute_artificial(self.c,m,root=root,store=object())
    def test_runtime_mutation_preclaim_rejected(self):
        root,store,m=self.fresh();c=deepcopy(self.c);c['python']['version']='mutated'
        with self.assertRaisesRegex(Denied,'BINDING'):B.execute_artificial(c,m,root=root,store=store)
        self.assertIsNone(store.snapshot(m)['claim'])
    def test_source_mutation_preclaim_rejected(self):
        root,store,m=self.fresh();c=deepcopy(self.c);c['source_pins'][K.NEW_FILES[0]]='0'*64
        with self.assertRaisesRegex(Denied,'BINDING'):B.execute_artificial(c,m,root=root,store=store)
    def test_manifest_binding_mutation_preclaim_rejected(self):
        root,store,m=self.fresh();m['candidate_binding']='0'*64
        with self.assertRaisesRegex(Denied,'BINDING'):B.execute_artificial(self.c,m,root=root,store=store)
    def test_real_namespace_not_generated(self):
        bits=B.F.bits_factory('ordinary','X')
        self.assertEqual(bits.domain,'ARTIFICIAL_API_BINDING_V1/ordinary/X')
        self.assertEqual(bits.bits(160),B.F.bits_factory('ordinary','X').bits(160))
        self.assertEqual(bits.consumed_bits,160)
    def test_fused_canonical_matches_fixed_codec(self):
        from fractions import Fraction
        x=dict(event={'raw_word':(0,1),'weight':Fraction(2,3)},evidence={'unicode':'é','negative':-2**512},angle={'partial':Fraction(3,11)})
        from trottertracks.algorithm_codesign import g6_w2c_b3_exact_serialization as E
        self.assertEqual(E.canonical(x),B.D.audit.canonical(B.D.exact(x)))

class MatrixAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows={x['case']:x for x in json.loads((MATRIX/'matrix.json').read_text())['cases']}
    def test_no_scientific_imports_in_either_process(self):
        for r in self.rows.values():self.assertFalse(r.get('forbidden_imports',[]));self.assertFalse(r.get('consumer_forbidden_imports',[]))
    def test_S_failure_prevents_P1_start(self):
        r=self.rows['S_failure_blocks_P1'];self.assertEqual([x['condition']for x in r['conditions']],['S']);self.assertFalse(r['conditions'][0]['complete'])
    def test_P1_failure_preserves_S_receipt(self):
        r=self.rows['P1_failure_preserves_S'];self.assertTrue(r['conditions'][0]['complete']);self.assertFalse(r['conditions'][1]['complete']);self.assertTrue((Path(r['root'])/'output/S_receipt.json').exists())
    def test_full_artificial_call_arithmetic(self):
        r=self.rows['normal_bundle_n256'];self.assertEqual(r['calls']['completed'],16896)
        self.assertEqual([(x['reserved'],x['completed'],x['planned'])for x in r['conditions']],[(8448,8448,8448)]*2 )
    def test_inflight_drains_before_global_finalization(self):
        r=self.rows['P1_inflight_drain'];final=O._read(Path(r['root'])/'output/final.json');s=final['states'];self.assertGreaterEqual(s['COMPLETED'],775);self.assertEqual(s['FAILED'],1);self.assertFalse(final['evidence_v2']['current_tickets']);self.assertFalse(final['evidence_v2']['pair_next'])
    def test_unknown_call_and_bytes_not_refunded(self):
        r=self.rows['missing_ACK_unknown_credit'];final=O._read(Path(r['root'])/'output/final.json');self.assertGreater(final['output']['unknown_credit']['result'],0);self.assertGreater(final['states']['TERMINATED_WITH_UNKNOWN_RESULT'],0)
    def test_reserve_fault_observed_atomic_RAM_transaction(self):
        for mode,base in (('S',0),('P1',768)):
            r=self.rows[mode+'_reserve_persistence'];obs=O._read(Path(r['root'])/'output/final.json')['fault_observation'];self.assertEqual(obs['reserved'],base+1);self.assertEqual(sum(obs['cursors'].values()),base+1);self.assertEqual(obs['started'],base);self.assertEqual(obs['states']['START_RESERVED'],1);self.assertEqual(len(obs['tickets']),1)
    def test_condition_byte_attribution_sums_once(self):
        v=json.loads((MATRIX/'normal_bundle_small_verification.json').read_text())
        for ch in ('result','log'):
            self.assertEqual(sum(x['reserved'].get(ch,0)for x in v['condition_byte_attribution'].values()),v['output_budget'][ch]['reserved'])
            self.assertEqual(sum(x['observed'].get(ch,0)for x in v['condition_byte_attribution'].values()),v['output_budget'][ch]['observed_or_self_projected'])
    def test_bridge_commit_ACK_unknown_keeps_spent(self):
        import sqlite3
        row=self.rows['admission_commit_ack_unknown'];root=Path(row['root'])
        with sqlite3.connect((root/'FAKE_ONLY.sqlite3').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)as db:r=db.execute('SELECT status,spent FROM claims').fetchone()
        self.assertEqual(r,('STOPPED',1));self.assertFalse((root/'output').exists())
    def test_closed_cost_boundaries_include_delivery(self):
        r=self.rows['normal_bundle_small'];v=json.loads((MATRIX/'normal_bundle_small_verification.json').read_text());outer=r['external_host_outer'];self.assertEqual(v['cost']['global_wall_ns'],outer['wall_ns']);self.assertEqual(v['cost']['total_cpu_ns'],outer['parent_cpu_ns']+outer['worker_cpu_ns']);self.assertFalse(v['cost']['coarse_and_fine_added'])

NORMAL={'normal_S':'VERIFIED_ARTIFICIAL_CONDITION_ONLY','normal_P1':'VERIFIED_ARTIFICIAL_CONDITION_ONLY','normal_bundle_small':'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE','normal_bundle_n256':'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'}
TECHNICAL={'S_failure_blocks_P1','P1_failure_preserves_S','F01_after_event','worker_crash','reserved_worker_crash','missing_ACK_unknown_credit','partial_worker_write','worker_flush','P1_inflight_drain','call_cap','result_cap','diagnostic_save','summary_save'}
for name in ['normal_S','normal_P1','normal_bundle_small','normal_bundle_n256','S_failure_blocks_P1','P1_failure_preserves_S','F01_after_event','worker_crash','reserved_worker_crash','missing_ACK_unknown_credit','partial_worker_write','worker_flush','P1_inflight_drain','parent_dispatch_failure','parent_crash','call_cap','result_cap','log_cap','S_reserve_persistence','S_enter_persistence','S_complete_persistence','P1_reserve_persistence','P1_enter_persistence','P1_complete_persistence','state_reserve_write','state_entry_write','trace_partial','trace_complete_flush','diagnostic_save','receipt_save','receipt_flush','receipt_close','marker_save','trace_close','state_close','journal_close','fake_STOP_save','final_close','delivery_close','trace_STOP_save','journal_STOP_save','admission_commit_ack_unknown','summary_save']:
    def test(self,name=name):
        row=self.rows[name];want=NORMAL.get(name,'TECHNICAL_FAILURE'if name in TECHNICAL else'INCOMPLETE_EVIDENCE');self.assertEqual(row['verification_status'],want,row.get('verification_issues'));self.assertFalse(row.get('verification_issues',[])if name not in ('log_cap','admission_commit_ack_unknown')else[])
        if row['spec'].get('fault'):self.assertEqual(row.get('fault_hits',1),1)
    setattr(MatrixAcceptance,'test_artificial_'+name,test)

class Mutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec=json.loads((MATRIX/'normal_bundle_small_result.json').read_text());cls.master=Path(cls.rec['root']);cls.trusted=json.loads((MATRIX/'trusted_candidate.json').read_text());cls.base_hashes={str(p.relative_to(cls.master)):O._sha(p)for p in cls.master.rglob('*')if p.is_file()}
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='b3-bundle-mutation-');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'case';shutil.copytree(self.master,self.root);self.out=self.root/'output';self.r=O._read(self.out/'final.json')
    def reseal(self):
        # Repair envelopes/digests after mutations so semantic checks must run.
        for name in self.r['physical_files_before_final']:
            p=self.out/name
            if p.exists():self.r['physical_files_before_final'][name]=dict(bytes=p.stat().st_size,sha256=O._sha(p))
        data=B._bytes(self.r);self.assertLessEqual(len(data),K.FINAL_SLOT);(self.out/'final.json').write_bytes(data+b' '*(K.FINAL_SLOT-len(data)))
        p=self.out/'delivery.json'
        if p.exists():
            d=O._read(p);d['final_sha256']=O._sha(self.out/'final.json');data=B._bytes(d);p.write_bytes(data+b' '*(K.DELIVERY_SLOT-len(data)))
        p=self.root/'external_observation.json'
        if p.exists():
            d=O._read(p);d['delivery_sha256']=O._sha(self.out/'delivery.json')if(self.out/'delivery.json').exists()else None;data=B._bytes(d);p.write_bytes(data+b' '*(K.OBSERVATION_SLOT-len(data)))
    def reject(self):
        v=O.verify(self.root,self.trusted);self.assertNotEqual(v['status'],'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE',v)
        hashes={str(p.relative_to(self.root)):O._sha(p)for p in self.root.rglob('*')if p.is_file()};changed=[name for name,h in hashes.items()if self.base_hashes.get(name)!=h];deleted=sorted(set(self.base_hashes)-set(hashes))
        evidence=dict(test_id=self.id(),status=v['status'],issues=v['issues'],evidence_gaps=v['evidence_gaps'],base_case='normal_bundle_small',changed_files=[dict(path=name,sha256=hashes[name],bytes=(self.root/name).stat().st_size)for name in sorted(changed)],deleted_files=deleted)
        if target:=os.environ.get('B3_BUNDLE_MUTATION_DIR'):
            dest=Path(target)/self._testMethodName;dest.mkdir(parents=True,exist_ok=False)
            for name in changed:
                p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(self.root/name,p)
            (dest/'overlay.json').write_text(json.dumps(evidence,indent=2)+'\n')
        MUTATION_RESULTS.append(evidence);return v
    def trace(self,change):
        path=self.out/'operations.jsonl';rows=[]
        for line in path.read_bytes().splitlines():rows.append(json.loads(json.loads(line)['payload']))
        change(rows);previous='0'*64;data=[]
        for i,r in enumerate(rows):
            r['seq']=i;raw=B.P.safe_json(r);h=sha256(previous.encode()+raw).hexdigest();data.append(B.P.safe_json(dict(payload=raw.decode(),previous=previous,sha256=h))+b'\n');previous=h
        raw=b''.join(data);path.write_bytes(raw);self.r['evidence_v2']['trace_prefix']=dict(rows=len(rows),bytes=len(raw),sha256=sha256(raw).hexdigest(),chain=previous)
    def test_fixture_normal_before_mutation(self):self.assertEqual(O.verify(self.root,self.trusted)['status'],'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE')
    def test_duplicate_RESERVE_rechained(self):
        def change(rows):i=next(i for i,r in enumerate(rows)if r['op']=='RESERVE');rows.insert(i+1,deepcopy(rows[i]))
        self.trace(change);self.reseal();self.assertIn('DUPLICATE',','.join(self.reject()['issues']))
    def test_ENTER_after_STOP_rechained(self):
        def change(rows):i=next(i for i,r in enumerate(rows)if r['op']=='ENTER');rows.insert(i,dict(rows[i],op='STOP'))
        self.trace(change);self.reseal();self.assertIn('ENTER_AFTER_STOP',','.join(self.reject()['issues']))
    def test_global_trial_mode_collision_rechained(self):
        def change(rows):r=next(r for r in rows if r['op']=='RESERVE'and r['condition']=='P1');r['global_trial_id']=r['global_trial_id'].replace('/P1/','/S/')
        self.trace(change);self.reseal();self.assertIn('GLOBAL_TRIAL',','.join(self.reject()['issues']))
    def test_semantic_CPU_double_add(self):self.r['total_cpu_ns']+=self.r['pairs'][0]['cpu_ns'];self.reseal();self.assertIn('CPU_DOUBLE_ADD',','.join(self.reject()['issues']))
    def test_semantic_observer_double_add(self):self.r['pairs'][-1]['fine_reconciliation']['exclusive_sum_ns'][0]+=1;self.reseal();self.reject()
    def test_marker_alone_not_success(self):
        for n in ('receipt.json','summary.json','delivery.json'): (self.out/n).unlink()
        self.reseal();self.reject()
    def test_caller_close_exception_not_success(self):
        p=self.root/'external_observation.json';d=O._read(p);d['bridge_returned_normally']=False;data=B._bytes(d);p.write_bytes(data+b' '*(K.OBSERVATION_SLOT-len(data)));self.assertEqual(self.reject()['status'],'INCOMPLETE_EVIDENCE')
    def test_missing_caller_observation(self):(self.root/'external_observation.json').unlink();self.assertEqual(self.reject()['status'],'INCOMPLETE_EVIDENCE')

BAD_REPORT={
 'reserved_refund':lambda r:r.update(reserved=r['reserved']-1),
 'cursor_refund':lambda r:r['evidence_v2']['next_indices'].update({next(iter(r['evidence_v2']['next_indices'])):0}),
 'RAM_state_false_complete':lambda r:r['states'].update(COMPLETED=1),
 'trial_offset':lambda r:r['pairs'][0]['task'].update(offset=3),
 'schema_mix':lambda r:r.update(schema='b3_pilot_storage_result_v2'),
 'unknown_credit_refund':lambda r:r['output']['unknown_credit'].update(result=-1),
 'byte_global_double_budget':lambda r:r['output']['limits'].update(result=2*(1<<36)),
 'call_cap':lambda r:r.update(reserved=16897),
 'duplicate_pair':lambda r:r['pairs'].append(deepcopy(r['pairs'][0])),
 'source_candidate':lambda r:r.update(source_release_candidate='0'*64),
 'runtime_identity':lambda r:r.update(candidate_binding='0'*64),
 'receipt_identity':lambda r:r.update(manifest_binding='0'*64),
 'physical_state_claim':lambda r:r['evidence_v2'].update(shadow_sha256='0'*64),
 'trusted_prefix':lambda r:r['evidence_v2']['trace_prefix'].update(bytes=1),
 'IPC_noncoalesced_ACK':lambda r:r['parent_messages'].update(WRITE_ACK=1),
 'production_permission':lambda r:r.update(production_permission=True),
}
for name,fn in BAD_REPORT.items():
    def test(self,fn=fn):fn(self.r);self.reseal();self.reject()
    setattr(Mutations,'test_mutation_'+name,test)

BAD_DOC={
 'seed_namespace':('manifest.json',lambda x:x.update(seed_namespace='WRONG_PREFIX')),
 'mode_seed_domain':('candidate.json',lambda x:x['proposal'].update(mode_is_seed_domain=True)),
 'formal_adoption_flip':('candidate.json',lambda x:x.update(run_ready=True)),
 'condition_budget_double':('manifest.json',lambda x:x['byte_limits'].update(log=2*(1<<28))),
 'summary_receipt_counter':('summary.json',lambda x:x.update(reserved=1)),
 'diagnostic_success':('diagnostic.json',lambda x:x.update(complete=True)),
 'receipt_false_completed':('receipt.json',lambda x:x.update(completed=1)),
 'receipt_preclose':('receipt.json',lambda x:x.update(storage_closed_before_receipt=False)),
 'marker_without_closed_receipt':('marker.json',lambda x:x.update(receipt_closed_before_marker=False)),
 'summary_prefix':('summary.json',lambda x:x['trace_prefix'].update(chain='0'*64)),
 'S_condition_only':('manifest.json',lambda x:x.update(modes=['S'])),
}
for name,(file,fn)in BAD_DOC.items():
    def test(self,file=file,fn=fn):
        p=self.out/file;x=O._read(p);fn(x);p.write_bytes(B._bytes(x));self.reseal();self.reject()
    setattr(Mutations,'test_mutation_document_'+name,test)

class ImportBoundary(unittest.TestCase):
    def test_consumer_never_imports_producer_or_science(self):
        code="import sys;from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1 import offline;assert not any(n.rsplit('.',1)[-1]in ('g6_w2c_b3_bundle_bridge_v1','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_admission','g7_generator','g6_f3_generator')for n in sys.modules)"
        p=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT);self.assertEqual(p.returncode,0,p.stdout.decode())


class PortIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=K.candidate(ROOT)
        cls.rows={x['case']:x for x in json.loads((MATRIX/'matrix.json').read_text())['cases']}
    def test_actual_dispatch_all_trials(self):
        for name,n in [('normal_bundle_small',1536),('normal_bundle_n256',16896),('variant_payload_propagation',1536)]:
            r=self.rows[name];p=r['port_dispatch']
            self.assertTrue(p['complete']);self.assertEqual((p['verified_trials'],p['scheduler_complete_joined']),(n,n))
            self.assertEqual(p['domains'],8);self.assertEqual(sum(p['cell_mode_counts'].values()),n)
    def test_payload_variant_reaches_raw_hash(self):
        a=self.rows['normal_bundle_small'];b=self.rows['variant_payload_propagation']
        path='output/S__correctness_baseline_ordinary_A.jsonl'
        ra=next(iter(O.V.Reader(Path(a['root'])/path))).payload
        rb=next(iter(O.V.Reader(Path(b['root'])/path))).payload
        self.assertEqual(ra['event']['artificial_port_variant'],'PORT_V1')
        self.assertEqual(rb['event']['artificial_port_variant'],'PORT_V2')
        self.assertNotEqual(O.V.event_hash(ra),O.V.event_hash(rb))
        self.assertEqual(ra['evidence']['consumption'],rb['evidence']['consumption'])
    def test_worker_factory_binding_sealed(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1.static import Rejected
        runtime=self.c['worker_runtime'];factory=B.F.ArtificialFactory(runtime,dict(artificial_only=True))
        factory.context['artificial_only']=False
        with self.assertRaisesRegex(Rejected,'CONTEXT_MUTATION'):factory(K.plan(16)[0][0],B.P.Settings())
    def test_no_worker_authority_credentials(self):
        import inspect
        source=inspect.getsource(B.execute_artificial)
        self.assertNotIn('store=store',source[source.index('factory='):source.index('conditions=[]')])
        self.assertEqual(set(self.c['worker_runtime']),{'schema','runtime_root','source_pins','python','source_roles','import_locations','schema_pins','artificial_fixture','entrypoint'})
    def test_parent_and_worker_import_attempts_block_before_load(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import ports as P
        import importlib
        P.install_guard()
        for name in P.SCIENCE:
            with self.assertRaises(P.Rejected):importlib.import_module('trottertracks.algorithm_codesign.'+name)
        self.assertFalse(P.forbidden_imports())
    def test_independent_surrogate_has_no_scientific_namespace(self):
        x=O.surrogate('ordinary','X',0,'PORT_V1');y=O.surrogate('ordinary','Y',0,'PORT_V1')
        self.assertNotEqual(x[2]['domain'],y[2]['domain']);self.assertEqual(x[3],2)
    def test_wrong_variant_rejected_preclaim(self):
        with self.assertRaises(ValueError):K.manifest(self.c,'ARTIFICIAL_ONLY/wrong',port_variant='REAL')
    def test_candidate_integer_boolean_rejected(self):
        c=deepcopy(self.c);c['scientific_reference']['scientific_reference']['policy']['sigma']=True
        with self.assertRaises(Denied):B.validate(c,K.manifest(c,'ARTIFICIAL_ONLY/bool'))

for mode,stage in [('S','audit'),('S','angle'),('S','requery'),('S','invalid_result'),('S','invalid_bits'),('P1','generator'),('P1','audit'),('P1','angle'),('P1','requery')]:
    def test(self,mode=mode,stage=stage):
        r=self.rows[mode+'_port_'+stage];self.assertEqual(r['verification_status'],'TECHNICAL_FAILURE',r.get('verification_issues'))
        self.assertNotEqual(r['verification_status'],'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE')
        if mode=='S':self.assertEqual([x['condition']for x in r['conditions']],['S'])
        else:self.assertTrue(r['conditions'][0]['complete'])
    setattr(PortIntegration,'test_port_failure_'+mode+'_'+stage,test)



def rewrite_port_payload(self,change):
    path=self.out/'S__correctness_baseline_ordinary_A.jsonl'
    old=path.read_bytes();payloads=[O.V.decode(json.loads(x)['payload'])for x in old.splitlines()]
    change(payloads[0]);previous='0'*64;lines=[]
    for payload in payloads:
        text=O.V.canonical(payload);h=sha256((previous+text).encode()).hexdigest()
        lines.append(B.P.safe_json(dict(previous=previous,sha256=h,payload=text))+b'\n');previous=h
    raw=b''.join(lines);self.assertEqual(len(raw),len(old),'Mutation must preserve global byte accounting')
    path.write_bytes(raw)
    report=next(x for x in self.r['pairs']if x['task']['key']=='S__correctness_baseline_ordinary_A')
    report['file_sha']=sha256(raw).hexdigest();report['chain']=previous
    control=self.out/'control.jsonl';oldcontrol=control.read_bytes();records=[O.B.A.safe_values(O.V.plain(x))for x in oldcontrol.splitlines()]
    for record in records:
        if record['event']=='WORKER_END'and record['report']['task']['key']==report['task']['key']:record['report']=report
    data=b''.join(B.P.safe_json(x)+b'\n'for x in records);self.assertEqual(len(data),len(oldcontrol));control.write_bytes(data)
    self.reseal()
    storage=O.B.verify_storage(self.root,self.trusted)
    self.assertEqual(storage['status'],'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE',storage)
    result=self.reject();self.assertEqual(result['status'],'INTEGRITY_MISMATCH',result)
    self.assertTrue(any('PORT_'in x for x in result['issues']),result)

PORT_MUTATIONS={
 'unconnected_factory':lambda p:p['evidence']['dispatch'].__setitem__('factory','X'*len(p['evidence']['dispatch']['factory'])),
 'wrong_port_source':lambda p:p['evidence']['dispatch'].__setitem__('endpoint_binding','0'*64),
 'wrong_candidate':lambda p:p['evidence']['dispatch'].__setitem__('candidate_binding','0'*64),
 'wrong_runtime':lambda p:p['evidence']['dispatch'].__setitem__('runtime_binding','0'*64),
 'wrong_global_trial':lambda p:p['evidence']['dispatch'].__setitem__('global_trial_id',p['evidence']['dispatch']['global_trial_id'].replace('/S/','/X/')),
 'wrong_bits_axis':lambda p:p['evidence']['consumption'].__setitem__('domain',p['evidence']['consumption']['domain'][:-1]+'Y'),
 'wrong_consume_digest':lambda p:p['evidence']['dispatch'].__setitem__('consumption_digest','0'*64),
 'wrong_event_digest':lambda p:p['evidence']['dispatch'].__setitem__('port_event_digest','0'*64),
 'wrong_worker_factory_identity':lambda p:p['evidence']['dispatch'].__setitem__('variant','PORT_V2'),
 'wrong_bits_value':lambda p:p['evidence']['consumption']['draws'][0].__setitem__('value_hex','0x'+'0'*40),
}
for name,change in PORT_MUTATIONS.items():
    def test(self,change=change):rewrite_port_payload(self,change)
    setattr(Mutations,'test_port_semantic_'+name,test)

if __name__=='__main__':unittest.main()
