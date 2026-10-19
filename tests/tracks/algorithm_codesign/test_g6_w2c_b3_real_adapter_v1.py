"""Artificial self-checks. No real science imports or random sample generation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import ast,json,os,subprocess,sys,tempfile,unittest
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import codec as C,bits as B,result as R,fixture as X
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import contract as K,bridge as Bridge,offline as O
from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_store_v1 import FakeStore,Denied
ROOT=Path(__file__).resolve().parents[3]


def obtain(arm='full_return',layer='B',kind='EVENT'):
    b=B.ArtificialBits(arm,'X')
    for i in range(16):
        tid=f'correctness/baseline/{arm}/{layer}/X/{i}'
        record,trace=X.build(b,trial_id=tid,arm=arm)
        if record.read().get('kind','EVENT')==kind:
            return record,trace,tid,X.adapt(record,trial_id=tid,arm=arm,layer=layer)
    raise AssertionError('ARTIFICIAL_COVERAGE')


class Codec(unittest.TestCase):
    def test_known_canonical_bytes(self):
        x={'signed':-7,'f':F(-3,5),'tuple':(1,),'list':[1]}
        expected='{"f":{"$rat":["-0x3","0x5"]},"list":[{"$int":"0x1"}],"signed":{"$int":"-0x7"},"tuple":{"$tuple":[{"$int":"0x1"}]}}'
        self.assertEqual(C.canonical(x),expected)
        self.assertEqual(C.Document.create(x).digest,sha256(expected.encode()).hexdigest())
    def test_nested_lossless_sidecar(self):
        x={1:{'nested':(F(-7,13),[True,None,2**4096])},(2,3):['é']}
        r=C.Document.create(x);s=C.convert(r,source_identity='SOURCE')
        self.assertEqual(C.restore(s,source_identity='SOURCE'),r)
        self.assertEqual(r.read(),x);self.assertNotEqual(r.document,s['driver_document'])
        self.assertIs(type(r.read()[1]['nested']),tuple)
    def test_map_order(self):
        self.assertEqual(C.canonical({2:'b',1:'a'}),C.canonical({1:'a',2:'b'}))
    def test_signed_integer_fraction(self):
        for x in (-2**2048,0,2**2048,F(-3,5),F(0)):
            self.assertEqual(C.Document.create(x).read(),x)
    def test_bool_is_distinct_from_integer(self):
        self.assertNotEqual(C.canonical(True),C.canonical(1))
        self.assertNotEqual(C.canonical(False),C.canonical(0))
    def test_driver_projection_parity(self):
        from trottertracks.algorithm_codesign import g6_w2c_b3_exact_serialization as E
        value={2:[F(-2,7),(1,True)],'x':None}
        self.assertEqual(C.driver_canonical(value),E.canonical(value))
    def test_driver_collision_rejected(self):
        with self.assertRaises(C.Rejected):C.convert(C.Document.create({1:'a','1':'b'}),source_identity='SOURCE')
    def test_wrong_digest(self):
        with self.assertRaises(C.Rejected):C.Document('{}','0'*64).read()
    def test_source_sidecar_mismatch(self):
        s=C.convert(C.Document.create((1,2)),source_identity='S')
        with self.assertRaises(C.Rejected):C.restore(s,source_identity='T')


for label,value in {
    'rational_zero_denominator':'{"$rat":["0x1","0x0"]}',
    'rational_negative_denominator':'{"$rat":["0x1","-0x2"]}',
    'rational_unreduced':'{"$rat":["0x2","0x4"]}',
    'rational_bool':'{"$rat":[true,"0x1"]}',
    'int_bool':'{"$int":true}',
    'int_decimal':'{"$int":3}',
    'unknown_tag':'{"$unknown":[]}',
    'mixed_tag':'{"$int":"0x1","other":true}',
    'bad_tuple':'{"$tuple":{}}',
    'bad_map_row':'{"$map":[[]]}',
    'duplicate_key':'{"x":true,"x":false}',
    'duplicate_map':'{"$map":[[{"$int":"0x1"},true],[{"$int":"0x1"},false]]}',
    'bool_int_map_collision':'{"$map":[[true,true],[{"$int":"0x1"},false]]}',
    'unhashable_key':'{"$map":[[[],true]]}',
    'float':'0.1','nonfinite':'NaN','noncanonical_hex':'{"$int":"0x01"}',
    'whitespace':'{ "x": true }',
}.items():
    def test(self,value=value):
        with self.assertRaises(C.Rejected):C.Document(value,C.digest_bytes(value.encode())).read()
    setattr(Codec,'test_reject_'+label,test)


class Bits(unittest.TestCase):
    def test_width_counter_discard(self):
        b=B.ArtificialBits('ordinary','X');count=0
        for n in (1,160,256,257,512,513):
            value=b.bits(n);row=b.trace[-1];blocks=(n+255)//256
            raw=b''.join(sha256(b.domain.encode()+i.to_bytes(8,'big')).digest() for i in range(count,count+blocks))
            self.assertEqual(value,int.from_bytes(raw,'big')>>(blocks*256-n));count+=blocks
            self.assertEqual(row['counter_after'],count);self.assertEqual(row['discarded_bits'],blocks*256-n)
        self.assertEqual(b.consumed_bits,1699)
    def test_artificial_prefix_and_domains(self):
        prefixes={}
        for arm in B.ARMS:
            for axis in ('X','Y'):
                values=[[B.ArtificialBits(arm,axis).bits(160)] for _ in range(8)]
                self.assertTrue(all(v==values[0] for v in values));prefixes[arm,axis]=values[0][0]
        self.assertEqual(len(set(prefixes.values())),8)
    def test_science_namespace_refused(self):
        for name in ('B3_FULL_N_NEW_20261019_V1','PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1','REAL'):
            with self.assertRaises(C.Rejected):B.ArtificialBits('ordinary','X',namespace=name)
    def test_failure_position_terminal(self):
        b=B.ArtificialBits('ordinary','X');b.bits(160)
        with self.assertRaises(C.Rejected):b.bits(True)
        self.assertEqual(b.failure_position,dict(draw_index=1,counter=1,consumed_bits=160))
        with self.assertRaises(C.Rejected):b.bits(160)
    def test_trace_roundtrip(self):
        record,tr,tid,a=obtain();self.assertEqual(B.validate_trace(tr,arm='full_return',axis='X'),C.fingerprint(tr))


class Results(unittest.TestCase):
    def test_all_cells_and_classifications(self):
        for arm in B.ARMS:
            for layer in ('A','B'):
                for axis in ('X','Y'):
                    b=B.ArtificialBits(arm,axis);seen=set()
                    for i in range(16):
                        tid=f'correctness/baseline/{arm}/{layer}/{axis}/{i}'
                        r,tr=X.build(b,trial_id=tid,arm=arm);a=X.adapt(r,trial_id=tid,arm=arm,layer=layer)
                        seen.add(a['classification']);C.restore(a['codec'],source_identity=X.SOURCE)
                        if a['event'] is not None and layer=='B':
                            q,ang=X.synthetic_angle(a);out=R.angle_binding(a,requery_input=q,requery_event=a['event'],angle_record=ang,synthetic=True)
                            self.assertFalse(out['mathematical_precision_verified'])
                    self.assertIn('ACCEPTED',seen)
                    if arm=='full_return':self.assertEqual(seen,{'RAW_NONREDUCED','COIN_REJECT','ACCEPTED'})
    def test_identity_preserved_without_zero_projection(self):
        ids=X.context('identity');r=C.Document.create(dict(schema='g6_f3_trial_v1',kind='IDENTITY',**ids,
            run_completion_or_confidence_certified=False,deterministic_operator='I',bits_consumed=0))
        a=R.adapt_f3(r,expected=ids);self.assertIsNone(a['driver_kind']);self.assertFalse(a['is_random_zero'])
    def test_failure_provenance_lossless(self):
        original=R.AdapterFailure('TECHNICAL_PRECISION_FAILURE','INSUFFICIENT','angle.enclosure',
            input_id='INPUT',details=dict(trace=[F(-3,7)],operation_context={'child':2}))
        with self.assertRaises(R.AdapterFailure) as caught:R.adapt_f3(original.record,expected=X.context('t'))
        self.assertEqual(caught.exception.record,original.record)
        self.assertTrue(caught.exception.to_dict()['mandatory_STOP'])
    def test_none_without_provenance_rejected(self):
        r,tr,tid,a=obtain('ordinary');d=r.read();d['event']=None;d['branch']='NONE'
        with self.assertRaises(C.Rejected):R.adapt_g7(None,observation=C.Document.create(d),expected=X.context(tid),arm='ordinary')
    def test_reference_reconstruction(self):
        for arm in B.ARMS:
            for layer in ('A','B'):
                b=B.ArtificialBits(arm,'X');counter=0
                for i in range(16):
                    tid=f'correctness/baseline/{arm}/{layer}/X/{i}'
                    r,tr=X.build(b,trial_id=tid,arm=arm);a=X.adapt(r,trial_id=tid,arm=arm,layer=layer)
                    ref,cons,angle,counter=O.reference(arm,'X',counter,tid,layer)
                    self.assertEqual(C.canonical(a),C.canonical(ref));self.assertEqual(tr,cons)
    def test_angle_wrong_event_object(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1.adapter import Endpoint,bits_factory
        task=K.plan(16)[0][1];c=K.candidate(ROOT);m=K.manifest(c,'ARTIFICIAL_ONLY/UNIT')
        context=dict(run_id=m['run_id'],endpoint_binding=c['endpoint_binding'],candidate_binding=K.digest(c),
            manifest_binding=K.digest(m),runtime_binding=K.digest(c['worker_runtime']),port_faults=[])
        e=Endpoint(task,None,context);r=e.sample(bits_factory(task['arm'],'X'),'correctness/baseline/ordinary/B/X/0')
        with self.assertRaises(R.AdapterFailure):e.angle_for(deepcopy(r))
        with self.assertRaises(C.Rejected):e.sample(bits_factory(task['arm'],'X'),'correctness/baseline/ordinary/B/X/1')


for name,change in {
    'missing_event':lambda d:d.pop('event'),
    'wrong_fraction':lambda d:d['event'].__setitem__('ratio',True),
    'child_bool':lambda d:d['event'].__setitem__('child',True),
    'child_mismatch':lambda d:d['event'].__setitem__('child',0),
    'word_list':lambda d:d['event'].__setitem__('word',[0,1]),
    'phase_bool':lambda d:d['event'].__setitem__('phase_i_power',True),
    'raw_nonreduced':lambda d:d['event'].__setitem__('raw_word',(0,0)),
    'signed_weight':lambda d:d['event'].__setitem__('weight',F(-1)),
    'coefficient':lambda d:d['event'].__setitem__('coefficient',F(1)),
    'unexpected_event_field':lambda d:d['event'].__setitem__('unexpected',1),
    'source':lambda d:d.__setitem__('source_id','WRONG'),
    'input':lambda d:d.__setitem__('input_id','WRONG'),
    'trial':lambda d:d.__setitem__('trial_id','WRONG'),
    'sigma_bool':lambda d:d.__setitem__('sigma',True),
    'trace_missing':lambda d:d.__setitem__('trace',None),
    'trace_count':lambda d:d['trace'].__setitem__('draw_count',99),
    'trace_value_bool':lambda d:d['trace']['draws'][0].__setitem__('value',True),
    'trace_stage':lambda d:d['trace']['draws'][0].__setitem__('stage','child'),
    'trace_raw':lambda d:d['trace'].__setitem__('raw_word',(1,2)),
    'unknown_variant':lambda d:d.__setitem__('kind','ACCEPTED'),
    'unexpected_trial_field':lambda d:d.__setitem__('unknown',True),
    'confidence':lambda d:d.__setitem__('run_completion_or_confidence_certified',True),
}.items():
    def test(self,change=change):
        r,tr,tid,a=obtain();d=r.read();change(d)
        with self.assertRaises(C.Rejected):R.adapt_f3(C.Document.create(d),expected=X.context(tid))
    setattr(Results,'test_mutation_'+name,test)

for name,change in {
    'reason':lambda d:d.__setitem__('reason','NONE'),
    'valid_zero_bool':lambda d:d.__setitem__('is_valid_zero_trial',1),
    'reason_trace':lambda d:d.__setitem__('reason','COIN_REJECT' if d['reason']=='RAW_NONREDUCED' else 'RAW_NONREDUCED'),
}.items():
    def test(self,change=change):
        r,tr,tid,a=obtain(kind='ZERO');d=r.read();change(d)
        with self.assertRaises(C.Rejected):R.adapt_f3(C.Document.create(d),expected=X.context(tid))
    setattr(Results,'test_zero_mutation_'+name,test)

for name,change in {
    'event_identity':lambda d:d['context'].__setitem__('event_id','WRONG'),
    'child':lambda d:d['context'].__setitem__('child',1),
    'ratio':lambda d:d.__setitem__('ratio',F(1)),
    'float_enclosure':lambda d:d.__setitem__('phi_interval',(1.0,2.0)),
    'reversed':lambda d:d.__setitem__('phi_interval',(F(2),F(1))),
    'bool_precision':lambda d:d.__setitem__('root_bits',True),
    'representative':lambda d:d.__setitem__('representative',F(1)),
    'signed_interval':lambda d:d.__setitem__('signed_phi_interval',(F(-1),F(1))),
    'unexpected':lambda d:d.__setitem__('extra',0),
}.items():
    def test(self,change=change):
        r,tr,tid,a=obtain();q,ang=X.synthetic_angle(a);d=ang.read();change(d)
        with self.assertRaises(C.Rejected):R.angle_binding(a,requery_input=q,requery_event=a['event'],angle_record=C.Document.create(d),synthetic=True)
    setattr(Results,'test_angle_mutation_'+name,test)


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=K.candidate(ROOT)
    def test_plan_and_closed_gate(self):
        self.assertEqual((len(K.plan(256)[0]),K.plan(256)[1]),(48,16896))
        m=K.manifest(self.c,'ARTIFICIAL_ONLY/UNIT');self.assertEqual(K.guard(self.c,m),K.digest(m))
    def test_no_dispatch_for_boolean_or_fake_authority(self):
        for fn in (K.production_entry,Bridge.production_entry,Bridge.ProductionFactory(),B.real_bits):
            with self.assertRaises(RuntimeError):fn(run_ready=True,science_release='FAKE',grant='FAKE',authority='FAKE')
    def test_no_factory_parameter(self):
        with self.assertRaises(TypeError):Bridge.execute_artificial({}, {},root=Path('/tmp'),store=None,endpoint_factory=object())
    def test_atomic_race_ack_unknown_stop(self):
        root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-adapter-race-'));store=FakeStore(root)
        m=K.manifest(self.c,'ARTIFICIAL_ONLY/'+root.name)
        def one(_):
            try:store.claim(m);return 1
            except Denied:return 0
        with ThreadPoolExecutor(max_workers=16) as pool:self.assertEqual(sum(pool.map(one,range(32))),1)
        store.stop(m);self.assertEqual(store.snapshot(m)['claim'][3:],['STOPPED',1])
        with self.assertRaises(Denied):store.claim(m)
        root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-adapter-ack-'));store=FakeStore(root)
        m=K.manifest(self.c,'ARTIFICIAL_ONLY/'+root.name)
        def fail(p):raise OSError('COMMIT_ACK_UNKNOWN')
        with self.assertRaises(OSError):store.claim(m,fail)
        self.assertEqual(store.snapshot(m)['claim'][3:],['SPENT',1])
        with self.assertRaises(Denied):store.claim(m)
        store.stop(m)
    def test_static_scientific_source_equal(self):
        for name in K.S.STATIC_MODULES:
            path=K.SOURCE+name+'.py'
            raw=subprocess.check_output(['git','show','76ca3c4089b201d55da0fede08e1f77733ac38a5:'+path],cwd=ROOT)
            self.assertEqual((ROOT/path).read_bytes(),raw)
            ast.parse(raw)
        source=(ROOT/K.SOURCE/'g6_f3_generator.py').read_text()
        self.assertIn('CANONICAL_PREFLIGHT_NOT_IMPLEMENTED',source)
    def test_offline_no_producer_import(self):
        code="import sys;from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import offline;assert not any(n.rsplit('.',1)[-1] in ('adapter','bridge','contract','fixture','ports','authority','g7_generator','g6_f3_generator','g6_f3_contract','g6_f3_precision') for n in sys.modules)"
        p=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)


for key,value in {
    'run_ready':True,'content_adopted':True,'contract_grants_execution':True,'science_release':'FAKE',
    'scientific_adoption':'FAKE','artificial_only':False,'mandatory_STOP':False,'retry':1,'resume':True,
    'python':{'version':'MUTATION'},'source_release_candidate':'0'*64,'endpoint_binding':{'real_dispatch':True},
}.items():
    def test(self,key=key,value=value):
        c=deepcopy(self.c);c[key]=value;m=K.manifest(c,'ARTIFICIAL_ONLY/NEGATIVE')
        with self.assertRaises(C.Rejected):K.guard(c,m)
    setattr(Contract,'test_preflight_mutation_'+key,test)

if __name__=='__main__':unittest.main()

class AdditionalInterfaces(unittest.TestCase):
    def test_mutated_bits_domain_rejected_before_generation(self):
        b=B.ArtificialBits('ordinary','X');b.domain='B3_FULL_N_NEW_20261019_V1/ordinary/X'
        with self.assertRaises(C.Rejected):b.bits(160)
        self.assertEqual(b.counter,0);self.assertEqual(b.consumed_bits,0)
    def test_bit_consumption_total_mutation(self):
        _,tr,_,_=obtain();tr['bits_consumed']+=1
        with self.assertRaises(C.Rejected):B.validate_trace(tr,arm='full_return',axis='X')
    def test_deterministic_event_query_is_not_sample(self):
        record,tr,tid,a=obtain();data=record.read();data['trace']=None
        data['selection_scope']='deterministic formal event query; not a sampled trial'
        out=R.adapt_f3(C.Document.create(data),expected=X.context(tid))
        self.assertEqual(out['classification'],'EVENT_QUERY');self.assertIsNone(out['driver_kind'])
    def test_bool_precision_predicate_rejected(self):
        _,_,_,a=obtain();query,record=X.synthetic_angle(a);data=record.read();data['precision_check']['passed']=1
        with self.assertRaises(C.Rejected):R.angle_binding(a,requery_input=query,requery_event=a['event'],angle_record=C.Document.create(data),synthetic=True)
    def test_wrong_requery_input(self):
        _,_,_,a=obtain();query,record=X.synthetic_angle(a);query['child']=1
        with self.assertRaises(C.Rejected):R.angle_binding(a,requery_input=query,requery_event=a['event'],angle_record=record,synthetic=True)
    def test_wrong_requery_output(self):
        _,_,_,a=obtain();query,record=X.synthetic_angle(a);e=deepcopy(a['event']);e['ratio']*=2
        with self.assertRaises(C.Rejected):R.angle_binding(a,requery_input=query,requery_event=e,angle_record=record,synthetic=True)
    def test_no_foreign_read_callable(self):
        class Foreign:
            def read(self):raise AssertionError('MUST_NOT_CALL')
        with self.assertRaises(C.Rejected):R.adapt_g7(None,observation=Foreign(),expected={},arm='ordinary')
        with self.assertRaises(C.Rejected):R.adapt_f3(Foreign(),expected={})
