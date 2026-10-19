"""Final artificial matrix + re-sealed semantic mutations; same author."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json,os,shutil,tempfile,unittest
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import offline as O,codec as C

MATRIX=Path(os.environ.get('B3_ADAPTER_MATRIX','/tmp/B3_ADAPTER_MATRIX_NOT_SET'))


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows={x['case']:x for x in json.loads((MATRIX/'matrix.json').read_text())['cases']}
    def test_normal_n256_all_evidence_reaches_raw(self):
        row=self.rows['normal_n256'];v=json.loads((MATRIX/'normal_n256_verification.json').read_text())
        self.assertEqual(v['status'],'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE',v['issues'])
        self.assertEqual(v['calls'],dict(planned=16896,reserved=16896,started=16896,completed=16896,unknown=0,cancelled=0))
        self.assertEqual((v['adapter_dispatch']['full_documents'],v['adapter_dispatch']['hashed_documents']),(1536,15360))
        self.assertEqual(v['adapter_dispatch']['scheduler_joined'],16896)
    def test_no_scientific_import(self):
        for row in self.rows.values():self.assertEqual(row['forbidden_imports'],[])
    def test_S_failures_block_P1(self):
        for name,row in self.rows.items():
            if name.startswith('S_'):
                r=O.B._read(Path(row['root'])/'output/final.json')
                self.assertEqual([c['condition'] for c in r['conditions']],['S'])
                self.assertFalse(r['conditions'][0]['complete'])
    def test_P1_failure_retains_S(self):
        row=self.rows['P1_failure_preserves_S'];root=Path(row['root'])
        self.assertTrue((root/'output/S_receipt.json').is_file())
        r=O.B._read(root/'output/final.json');self.assertTrue(r['conditions'][0]['complete'])
        self.assertFalse(r['conditions'][1]['complete'])
    def test_unknown_not_refunded(self):
        row=self.rows['missing_ACK'];r=O.B._read(Path(row['root'])/'output/final.json')
        self.assertGreater(r['states']['TERMINATED_WITH_UNKNOWN_RESULT'],0)
        self.assertGreater(r['output']['unknown_credit']['result'],0)
    def test_one_claim_stopped_all_scopes(self):
        for row in self.rows.values():
            v=json.loads((MATRIX/(row['case']+'_verification.json')).read_text())
            self.assertEqual(v['admission']['claim'][3:],['STOPPED',1])
    def test_cost_add_once(self):
        v=json.loads((MATRIX/'normal_n256_verification.json').read_text());cost=v['cost']
        self.assertEqual(cost['total_cpu_ns'],cost['parent_cpu_ns']+cost['all_worker_cpu_ns'])
        self.assertFalse(cost['coarse_and_fine_added']);self.assertFalse(cost['pair_or_IPC_added_to_outer'])
    def test_monitoring_new_stages_present(self):
        row=self.rows['normal_n256'];r=O.B._read(Path(row['root'])/'output/final.json')
        labels=' '.join(k for p in r['pairs'] if p['fine'] for k in p['fine'])
        for name in ('bits.py','codec.py','result.py','angle_binding','adapt_f3','adapt_g7'):
            self.assertIn(name,labels)


TECHNICAL=['S_generator','S_codec','S_evidence','S_bits','S_audit','S_angle','S_requery','S_worker_exit',
           'P1_failure_preserves_S','call_cap','missing_ACK','worker_write','worker_flush']
INCOMPLETE=['trace_close','trace_partial','state_close','state_write','receipt_close','marker_close','delivery_close']
for name in ['normal_n16','normal_n256']+TECHNICAL+INCOMPLETE:
    def test(self,name=name):
        row=self.rows[name];want='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE' if name.startswith('normal_') else 'TECHNICAL_FAILURE' if name in TECHNICAL else 'INCOMPLETE_EVIDENCE'
        self.assertEqual(row.get('verification_status'),want,row)
        self.assertEqual(row.get('verification_issues'),[])
        if 'fault' in row['spec']:
            if 'fault_hits' in row:self.assertEqual(row['fault_hits'],1)
            else:self.assertIn('ARTIFICIAL_STORAGE_FAULT:'+row['spec']['fault']['point'],row['exception']['reason'])
    setattr(Integration,'test_matrix_'+name,test)


class SemanticMutation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=json.loads((MATRIX/'matrix.json').read_text())['cases']
        cls.master=Path(next(x['root'] for x in rows if x['case']=='normal_n16'))
        cls.trusted=json.loads((cls.master/'candidate_input.json').read_text())
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='b3-adapter-mutation-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'case';shutil.copytree(self.master,self.root);self.out=self.root/'output'
        self.report=O.B._read(self.out/'final.json')
    def rewrite(self,change,*,angle=False):
        key='S__correctness_baseline_ordinary_'+('B' if angle else 'A')
        path=self.out/(key+'.jsonl');old=path.read_bytes()
        payloads=[O.V.decode(json.loads(x)['payload']) for x in old.splitlines()]
        change(payloads[0]);previous='0'*64;lines=[]
        for p in payloads:
            text=O.V.canonical(p);h=sha256((previous+text).encode()).hexdigest()
            lines.append(C.json_bytes(dict(previous=previous,sha256=h,payload=text))+b'\n');previous=h
        raw=b''.join(lines);self.assertEqual(len(raw),len(old));path.write_bytes(raw)
        report=next(x for x in self.report['pairs'] if x['task']['key']==key)
        report['file_sha']=sha256(raw).hexdigest();report['chain']=previous
        control=self.out/'control.jsonl';old_control=control.read_bytes()
        rows=[O.B.A.safe_values(O.V.plain(x)) for x in old_control.splitlines()]
        for row in rows:
            if row['event']=='WORKER_END' and row['report']['task']['key']==key:row['report']=report
        data=b''.join(C.json_bytes(x)+b'\n' for x in rows);self.assertEqual(len(data),len(old_control));control.write_bytes(data)
        for name in self.report['physical_files_before_final']:
            p=self.out/name;self.report['physical_files_before_final'][name]=dict(bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest())
        def slot(path,value,size):
            data=C.json_bytes(value)+b'\n';self.assertLessEqual(len(data),size);path.write_bytes(data+b' '*(size-len(data)))
        slot(self.out/'final.json',self.report,4*1024*1024)
        d=O.B._read(self.out/'delivery.json');d['final_sha256']=sha256((self.out/'final.json').read_bytes()).hexdigest()
        slot(self.out/'delivery.json',d,4096)
        obs=O.B._read(self.root/'external_observation.json');obs['delivery_sha256']=sha256((self.out/'delivery.json').read_bytes()).hexdigest()
        slot(self.root/'external_observation.json',obs,4096)
        storage=O.B.verify_storage(self.root,self.trusted)
        self.assertEqual(storage['status'],'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE',storage['issues'])
        result=O.verify(self.root,self.trusted)
        self.assertEqual(result['status'],'INTEGRITY_MISMATCH',result['issues'])
        if dest:=os.environ.get('B3_ADAPTER_MUTATIONS'):
            dest=Path(dest)/self._testMethodName;dest.mkdir(parents=True,exist_ok=False)
            for name in (key+'.jsonl','control.jsonl','final.json','delivery.json'):
                target=dest/'output'/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(self.out/name,target)
            shutil.copyfile(self.root/'external_observation.json',dest/'external_observation.json')
            (dest/'result.json').write_text(json.dumps(dict(storage_status=storage['status'],adapter_status=result['status'],issues=result['issues']),indent=2)+'\n')


for name,change in {
    'factory':lambda p:p['evidence']['dispatch'].__setitem__('factory','X'*len(p['evidence']['dispatch']['factory'])),
    'runtime':lambda p:p['evidence']['dispatch'].__setitem__('runtime_binding','0'*64),
    'source':lambda p:p['evidence']['dispatch'].__setitem__('candidate_binding','0'*64),
    'event_fingerprint':lambda p:p['evidence']['adapted'].__setitem__('event_fingerprint','0'*64),
    'f3_digest':lambda p:p['evidence']['adapted']['codec']['f3'].__setitem__('digest','0'*64),
    'driver_digest':lambda p:p['evidence']['adapted']['codec'].__setitem__('driver_digest','0'*64),
    'codec_tuple_tag':lambda p:p['evidence']['adapted']['codec']['f3'].__setitem__('document',p['evidence']['adapted']['codec']['f3']['document'].replace('$tuple','$tup1e')),
    'codec_child':lambda p:p['evidence']['adapted']['codec']['f3'].__setitem__('document',p['evidence']['adapted']['codec']['f3']['document'].replace('"child":{"$int":"0x2"}','"child":{"$int":"0x1"}')),
    'accepted_reject':lambda p:p['evidence']['adapted'].__setitem__('classification','REJECTED'),
    'seed_axis':lambda p:p['evidence']['consumption'].__setitem__('domain',p['evidence']['consumption']['domain'][:-1]+'Y'),
    'bits_value':lambda p:p['evidence']['consumption']['draws'][0].__setitem__('value_hex','0x'+'0'*40),
    'bits_width':lambda p:p['evidence']['consumption']['draws'][0].__setitem__('width',159),
    'bits_discard':lambda p:p['evidence']['consumption']['draws'][0].__setitem__('discarded_bits',95),
    'consumption_digest':lambda p:p['evidence']['dispatch'].__setitem__('consumption_digest','0'*64),
    'global_mode':lambda p:p['evidence']['dispatch'].__setitem__('global_trial_id',p['evidence']['dispatch']['global_trial_id'].replace('/S/','/Z/')),
}.items():
    def test(self,change=change):self.rewrite(change)
    setattr(SemanticMutation,'test_semantic_'+name,test)

for name,change in {
    'angle_event_id':lambda p:p['angle'].__setitem__('event_identity','0'*64),
    'angle_fingerprint':lambda p:p['angle'].__setitem__('event_fingerprint','0'*64),
    'angle_digest':lambda p:p['angle']['angle']['f3'].__setitem__('digest','0'*64),
    'requery_child':lambda p:p['angle']['requery_input'].__setitem__('child',1),
    'requery_digest':lambda p:p['angle']['requery_output'].__setitem__('digest','0'*64),
}.items():
    def test(self,change=change):self.rewrite(change,angle=True)
    setattr(SemanticMutation,'test_semantic_'+name,test)
