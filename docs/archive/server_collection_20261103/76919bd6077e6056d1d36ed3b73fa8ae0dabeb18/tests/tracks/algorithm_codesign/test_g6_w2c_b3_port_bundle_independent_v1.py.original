"""Independent semantic attacks with repaired raw/control/final/delivery hashes."""
import pathlib,os,json,copy,hashlib,shutil,unittest,sys
from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_independent_v1 import adapter as A,dispatch as D
from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_independent_v1.identity import Reference,read,equal,hashed
from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_independent_adapter_v1 as OLD
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
W=pathlib.Path(__file__).resolve().parents[3];P=W/'docs/research/track_b_g6_w2c_b3_closed_port_bundle_independent_local_20261019';H='62967269e611b61e73b70abe20e998d9c7eaaf50';Q=P/'fixed_reference'/H/'docs/research/track_b_g6_w2c_b3_closed_port_bundle_integration_20261019';C=read(Q/'technical_contract_candidate.json');REF=Reference(P/'fixed_reference'/H,C)
RAW=pathlib.Path(os.environ['B3_PORT_BUNDLE_RAW']);OUT=pathlib.Path(os.environ['B3_PORT_BUNDLE_TEST_OUTPUT']);OUT.mkdir(parents=True,exist_ok=True);RESULTS=[]

def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encoded(x):return json.dumps(x,separators=(',',':'),ensure_ascii=True).encode()+b'\n'
def put(p,b):
    if p.exists():p.unlink()
    p.write_bytes(b)
def padded(p,x):
    n=p.stat().st_size;b=encoded(x);assert len(b)<=n;put(p,b+b' '*(n-len(b)))
def copy_case(name,base='normal_bundle_small'):
    dest=OUT/name;shutil.copytree(RAW/'cases'/base,dest,copy_function=os.link);return dest

def reseal(root):
    out=root/'output';r=read(out/'final.json')
    for name in r['physical_files_before_final']:
        p=out/name
        if p.exists():r['physical_files_before_final'][name]=dict(bytes=p.stat().st_size,sha256=sh(p))
    padded(out/'final.json',r);d=read(out/'delivery.json');d['final_sha256']=sh(out/'final.json');padded(out/'delivery.json',d)
    e=read(root/'external_observation.json');e['delivery_sha256']=sh(out/'delivery.json');padded(root/'external_observation.json',e)

def change_first_evidence(root,change):
    out=root/'output';key='S__correctness_baseline_ordinary_A';p=out/(key+'.jsonl');oldlen=p.stat().st_size;envs=[json.loads(x) for x in p.read_bytes().splitlines()];prev='0'*64;output=[]
    for i,env in enumerate(envs):
        payload=json.loads(env['payload'])
        if i==0:change(payload['evidence'])
        env['payload']=json.dumps(payload,separators=(',',':'),sort_keys=True,ensure_ascii=True);env['previous']=prev;prev=hashlib.sha256(prev.encode()+env['payload'].encode()).hexdigest();env['sha256']=prev;output.append(encoded(env))
    b=b''.join(output);assert len(b)==oldlen;put(p,b)
    r=read(out/'final.json');row=next(x for x in r['pairs'] if x['task']['key']==key);row['chain']=prev;row['file_sha']=sh(p);padded(out/'final.json',r)
    control=out/'control.jsonl';lines=[]
    for line in control.read_bytes().splitlines():
        x=json.loads(line)
        if x.get('event')=='WORKER_END' and x['report']['task']['key']==key:
            x['report']['chain']=prev;x['report']['file_sha']=sh(p)
        lines.append(encoded(x))
    b=b''.join(lines);assert len(b)==control.stat().st_size;put(control,b);reseal(root)

def altered(s):return ('1' if s[0]!='1' else '2')+s[1:]
ATTACKS={
'factory':lambda e:e['dispatch'].__setitem__('factory',altered(e['dispatch']['factory'])),
'candidate':lambda e:e['dispatch'].__setitem__('candidate_binding',altered(e['dispatch']['candidate_binding'])),
'manifest':lambda e:e['dispatch'].__setitem__('manifest_binding',altered(e['dispatch']['manifest_binding'])),
'endpoint_source':lambda e:e['dispatch'].__setitem__('endpoint_binding',altered(e['dispatch']['endpoint_binding'])),
'runtime':lambda e:e['dispatch'].__setitem__('runtime_binding',altered(e['dispatch']['runtime_binding'])),
'global_trial':lambda e:e['dispatch'].__setitem__('global_trial_id',e['dispatch']['global_trial_id'].replace('/S/','/X/')),
'global_offset':lambda e:e['dispatch']['global_offset'].__setitem__('int','0x1'),
'worker_pid':lambda e:e['dispatch']['worker_pid'].__setitem__('int',e['dispatch']['worker_pid']['int'][:-1]+('1' if e['dispatch']['worker_pid']['int'][-1]!='1' else '2')),
'event_digest':lambda e:e['dispatch'].__setitem__('port_event_digest',altered(e['dispatch']['port_event_digest'])),
'consumption_digest':lambda e:e['dispatch'].__setitem__('consumption_digest',altered(e['dispatch']['consumption_digest'])),
'variant_tag_only':lambda e:e['dispatch'].__setitem__('variant','PORT_V2'),
'bits_domain':lambda e:e['consumption'].__setitem__('domain',e['consumption']['domain'][:-1]+'Y'),
'bits_value':lambda e:e['consumption']['draws'][0].__setitem__('value_hex',e['consumption']['draws'][0]['value_hex'][:-1]+'0'),
'bits_width':lambda e:e['consumption']['draws'][0]['width'].__setitem__('int','0x9f'),
}
class RepairedPortAttacks(unittest.TestCase):pass
for name,change in ATTACKS.items():
    def test(self,name=name,change=change):
        root=copy_case(name);change_first_evidence(root,change);old=OLD.verify(root,REF);self.assertEqual(old['status'],'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE',old['issues']);new=A.verify(root,REF);self.assertEqual(new['status'],'INTEGRITY_MISMATCH',new['issues']);RESULTS.append(dict(mutation=name,base_case='normal_bundle_small',storage_only=old['status'],integrated=new['status'],issues=new['issues']))
    setattr(RepairedPortAttacks,'test_'+name,test)

class Contract(unittest.TestCase):
    def test_pinned_identity(self):self.assertEqual(REF.identity(C,read(RAW/'cases/normal_bundle_small/manifest_input.json'))['planned'],1536)
    def test_science_and_producer_never_imported(self):
        bad={'g7_generator','g7_provider','return_aggregation','g6_f3_generator','g6_f3_contract','g6_f3_precision','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_bundle_bridge_v1','g6_w2c_b3_parallel_admission'}
        self.assertFalse([n for n in sys.modules if n.rsplit('.',1)[-1] in bad or '.g6_w2c_b3_port_bundle_v1' in n or '.g6_w2c_b3_api_binding_v1' in n])
    def test_variant_keeps_bits_changes_event_hash(self):
        a=read(RAW/'cases/normal_bundle_small/manifest_input.json');b=read(RAW/'cases/variant_payload_propagation/manifest_input.json');t=a['tasks'][0]
        x=D.expected(C,a,t,1234,'X',0,0);y=D.expected(C,b,t,1234,'X',0,0);self.assertEqual(x[2]['consumption'],y[2]['consumption']);self.assertNotEqual(V.identity(x[1]),V.identity(y[1]))
for field,value in [('run_ready',True),('content_adopted',True),('science_release','fake'),('scientific_adoption','fake'),('contract_grants_execution',True),('retry',False),('resume',True),('python',{}),('source_pins',{}),('worker_runtime',{}),('endpoint_binding',{}),('fixed_port_commit','0'*40),('fixed_bundle_commit','0'*40),('source_release_candidate','0'*64)]:
    def test(self,field=field,value=value):
        cc=copy.deepcopy(C);cc[field]=value
        with self.assertRaises(V.Invalid):REF.identity(cc,read(RAW/'cases/normal_bundle_small/manifest_input.json'))
    setattr(Contract,'test_reject_'+field,test)
for field,value in [('seed_namespace','ARTIFICIAL_API_BINDING_V1/S'),('port_variant','REAL'),('call_limit',True),('tasks',[]),('planned',1),('global_claim','0'*64),('endpoint_binding','0'*64),('candidate_binding','0'*64),('scientific_input','0'*64)]:
    def test(self,field=field,value=value):
        m=copy.deepcopy(read(RAW/'cases/normal_bundle_small/manifest_input.json'));m[field]=value
        with self.assertRaises(V.Invalid):REF.identity(C,m)
    setattr(Contract,'test_manifest_'+field,test)

class Components(unittest.TestCase):pass
for name,change in {
'bool_pid':lambda p:p['evidence']['dispatch'].__setitem__('worker_pid',True),
'bool_offset':lambda p:p['evidence']['dispatch'].__setitem__('global_offset',False),
'bool_width':lambda p:p['evidence']['consumption']['draws'][0].__setitem__('width',True),
'wrong_angle':lambda p:p['angle'].__setitem__('event_requery_surrogate','0'*64),
'float_weight':lambda p:p['event'].__setitem__('weight',float(p['event']['weight'])),
'missing_child':lambda p:p['event'].pop('child'),
'raw_zero_remap':lambda p:p['event'].__setitem__('raw_word',[]),
'wrong_weight':lambda p:p['event'].__setitem__('weight',p['event']['weight']*2),
'wrong_kind':lambda p:p.__setitem__('kind','ZERO_C'),
'wrong_trial':lambda p:p.__setitem__('trial_id',p['trial_id'][:-1]+'1'),
'no_dispatch':lambda p:p['evidence'].pop('dispatch'),
'bits_factory_old':lambda p:p['evidence']['consumption'].__setitem__('domain','ARTIFICIAL_R1_PARALLEL/ordinary/X'),
}.items():
    def test(self,name=name,change=change):
        m=read(RAW/'cases/normal_bundle_small/manifest_input.json');t=m['tasks'][1];kind,event,evidence,angle,end=D.expected(C,m,t,1234,'X',0,0);p=dict(trial_id=V.trial(dict(t,N=t['n']),'X',0),axis='X',kind=kind,event=event,evidence=evidence,angle=angle);change(p)
        with self.assertRaises((V.Invalid,KeyError,TypeError)):D.payload_check(p,C,m,t,1234,'X',0,0)
    setattr(Components,'test_'+name,test)

class SavedEvidence(unittest.TestCase):pass
for name,status in [('normal_bundle_small','VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'),('normal_bundle_n256','VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'),('variant_payload_propagation','VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'),('normal_S','VERIFIED_ARTIFICIAL_CONDITION_ONLY'),('normal_P1','VERIFIED_ARTIFICIAL_CONDITION_ONLY'),('S_port_audit','TECHNICAL_FAILURE'),('P1_port_angle','TECHNICAL_FAILURE')]:
    def test(self,name=name,status=status):
        r=A.verify(RAW/'cases'/name,REF);self.assertEqual(r['status'],status,r['issues']);RESULTS.append(dict(case=name,status=r['status'],dispatch=r['port_dispatch']))
    setattr(SavedEvidence,'test_'+name,test)
for name,file in [('missing_caller','external_observation.json'),('missing_receipt','output/receipt.json'),('missing_state','output/trial_states.bin'),('missing_trace','output/operations.jsonl')]:
    def test(self,name=name,file=file):
        root=copy_case(name);(root/file).unlink();r=A.verify(root,REF);self.assertEqual(r['status'],'INCOMPLETE_EVIDENCE',r['issues']);RESULTS.append(dict(mutation=name,base_case='normal_bundle_small',integrated=r['status'],issues=r['issues']))
    setattr(SavedEvidence,'test_'+name,test)
