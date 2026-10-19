"""Same-author artificial checks; no live scientific imports or computation."""
from copy import deepcopy
from pathlib import Path
from fractions import Fraction as F
import json
import tempfile
import unittest
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import spec as S, contract as K, interface as I
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import artificial as A
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import fixture as X, codec as C, bits as B, result as R

ROOT=Path(__file__).resolve().parents[3]


def fixture(arm='ordinary',layer='B',axis='X'):
    tid=f'correctness/baseline/{arm}/{layer}/{axis}/0'
    doc,cons=X.build(B.ArtificialBits(arm,axis),trial_id=tid,arm=arm)
    adapted=X.adapt(doc,trial_id=tid,arm=arm,layer=layer)
    angle=None
    if layer=='B' and adapted['classification']=='ACCEPTED':
        q,a=X.synthetic_angle(adapted)
        angle=(R.angle_binding(adapted,requery_input=q,requery_event=adapted['event'],angle_record=a,synthetic=True)
            if arm=='full_return' else R.canonical_angle_binding(adapted,requery_input=q,requery_event=adapted['event'],
                angle_record=a,input_payload={'id':'ARTIFICIAL_INPUT_V1'},policy_payload={'id':'ARTIFICIAL_PRECISION_V1'},synthetic=True))
    audit=None
    if layer=='B':
        ident=C.fingerprint(dict(arm=arm,source=X.SOURCE));state=C.fingerprint(dict(artificial=True,arm=arm,layer=layer))
        seal=C.DriverDocument.create(dict(arm=arm,identity=ident,state_id=state,payload=doc.document,
            completed_calls_before=0,confidence_certified=False))
        audit=R.canonical_audit(seal,arm=arm,identity=ident,state_id=state,completed_calls_before=0)
    return dict(adapted=adapted,consumption=cons,trial_id=tid,arm=arm,layer=layer,axis=axis,angle=angle,audit=audit)


class Unit(unittest.TestCase):
    def test_shape_no_profile(self):
        self.assertEqual(len(S.tasks()),16)
        self.assertEqual(sum(2*t['n'] for t in S.tasks()),512)
        self.assertEqual({t['phase'] for t in S.tasks()},{'correctness'})
    def test_distinct_routes(self):
        self.assertEqual(S.route('ordinary','B')['construction'],'G7.make_generator')
        self.assertEqual(S.route('full_return','B')['construction'],'F3.prepare')
        self.assertIn('group_index',S.route('ordinary','B')['requery'])
        self.assertIn('packet',S.route('full_return','B')['requery'])
    def test_exact_candidates(self):
        self.assertEqual(sum(map(F,S.INPUT['p'])),1)
        self.assertEqual((S.INPUT['m'],S.PRECISION['root_bits'],S.ANGLE['K_phi']),(5,256,32))
    def test_budget_separation(self):
        b=S.budgets()
        self.assertEqual((b['scheduled_sampling'],b['primary_constructions'],b['B_angle']),(512,16,256))
        self.assertLess(b['result_bytes'],64<<30)
    def test_static_signatures(self):
        a=S.signatures(ROOT)
        self.assertEqual(a['g7_generator.CanonicalGenerator.event']['arguments'],'self, group_index, word, child')
        self.assertEqual(a['g6_f3_generator.GeneratorSession.sample']['arguments'],'self, bits=None, *, trial_id')
    def test_seed_prefix_and_global_id(self):
        ident=I.ConnectionIdentity('s','i','p','r','run','claim','S','ordinary','A','X','correctness','baseline',0,'session')
        other=I.ConnectionIdentity(**{**ident.__dict__,'mode':'P1','layer':'B'})
        self.assertNotEqual(ident.global_id(),other.global_id())
        self.assertEqual(ident.proposed_seed_domain(),other.proposed_seed_domain())
    def test_artificial_connection_roundtrip(self):
        f=fixture();a=f['adapted'];d=C.restore(a['codec'],source_identity=X.SOURCE)
        self.assertEqual(I.connect_artificial(d,trial_id=f['trial_id'],arm=f['arm'],layer=f['layer']),a)
        self.assertEqual(I.validate_connection(**f)['live_compatibility'],'NOT_ESTABLISHED')
    def test_coverage_no_adaptive_sampling(self):
        rows=[dict(cell=[mo,a,l,x],completed=16,accepted=1) for mo in ('S','P1') for a in S.ARMS for l in ('A','B') for x in ('X','Y')]
        self.assertEqual(I.coverage(rows),'CANDIDATE_CORRECTNESS_PASS')
        rows[0]['accepted']=0;self.assertEqual(I.coverage(rows),'INCONCLUSIVE_COVERAGE')
        rows[0]['unknown']=True;self.assertEqual(I.coverage(rows),'TECHNICAL_STOP_UNKNOWN')
        rows[0].pop('unknown');rows[0]['scientific_mismatch']=True
        self.assertEqual(I.coverage(rows),'SCIENTIFIC_MISMATCH_STOP')
    def test_parent_core_not_loaded(self):
        self.assertEqual(G.observation()['core_loaded'],[])
        self.assertEqual(G.ATTEMPTS,[])
    def test_all_routes_positive(self):
        for arm in S.ARMS:
            for layer in ('A','B'):
                self.assertEqual(I.validate_connection(**fixture(arm,layer))['live_compatibility'],'NOT_ESTABLISHED')
    def test_real_loader_absent(self):
        for fn in (K.production_entry,I.FutureScientificResolver().resolve):
            with self.assertRaises(RuntimeError):fn(content_adopted=True,fake_approval=True)
    def test_bits_failure_position_terminal(self):
        b=B.ArtificialBits('ordinary','X')
        with self.assertRaises(Exception):b.bits(True)
        self.assertEqual(b.failure_position['counter'],0)
        with self.assertRaises(Exception):b.bits(160)
    def test_fake_one_shot(self):
        with tempfile.TemporaryDirectory(prefix='b3-bundle-artificial-preflight-unit-') as p:
            store=A.FakeStore(p);m=dict(global_scope='ARTIFICIAL_ONLY/unit',global_claim='FAKE',retry=0,resume=False)
            store.claim(m);store.stop(m)
            with self.assertRaises(Exception):store.claim(m)
            self.assertEqual(store.snapshot(m)['claim'][3:],['STOPPED',1])


class PreflightNegative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=K.candidate(ROOT)
        cls.dest='/tmp/b3-bundle-artificial-preflight-candidate/output'
        cls.m=K.manifest(cls.c,'ARTIFICIAL_ONLY/CORRECTNESS/unit',cls.dest)
    def test_positive(self):
        self.assertEqual(K.preflight(self.c,self.m,destination=self.dest),S.digest(self.m))


def candidate_change(field,value):
    def test(self):
        c=deepcopy(self.c);c[field]=value
        with self.assertRaises(Exception):K.preflight(c,self.m,destination=self.dest)
    return test


for name,field,value in [
 ('science_source','scientific_reference_commit','0'*40),('api_signature','api_signatures',{}),
 ('input','input',{}),('precision','precision',{}),('runtime','python',{}),
 ('worker_runtime','worker_runtime',{}),('approval','content_adopted',True),
 ('gate','run_ready',True),('release','science_release','fake'),('artifact','previous_artifact','0'*64)]:
    setattr(PreflightNegative,'test_wrong_'+name,candidate_change(field,value))

for field,value in [('seed',{}),('budgets',{}),('retry',1),('resume',True),('real_grant','fake-approved'),
                    ('storage_destination','/tmp/other'),('planned',16896),('tasks',[]),('fake_authority',False),
                    ('candidate_binding','0'*64),('contract_grants_execution',True)]:
    def test(self,field=field,value=value):
        m=deepcopy(self.m);m[field]=value
        with self.assertRaises(Exception):K.preflight(self.c,m,destination=self.dest)
    setattr(PreflightNegative,'test_manifest_'+field,test)


class SemanticMutation(unittest.TestCase):
    """Document mutations include recomputed codec seals, not only bad hashes."""
    def check_mutation(self,change):
        f=fixture();change(f)
        with self.assertRaises(Exception):I.validate_connection(**f)


def reseal(f,field,value):
    a=f['adapted'];doc=C.restore(a['codec'],source_identity=X.SOURCE);data=doc.read()
    data[field]=value
    a['codec']=C.convert(C.Document.create(data),source_identity=X.SOURCE)


MUTATIONS={
 'event_type':lambda f:f['adapted'].__setitem__('event',[]),
 'original_digest':lambda f:f['adapted']['codec']['f3'].__setitem__('digest','0'*64),
 'sidecar':lambda f:f['adapted']['codec'].__setitem__('source_identity','other'),
 'fingerprint':lambda f:f['adapted'].__setitem__('event_fingerprint','0'*64),
 'event_identity':lambda f:f['adapted'].__setitem__('event_identity','0'*64),
 'classification':lambda f:f['adapted'].__setitem__('classification','RAW_NONREDUCED'),
 'trial':lambda f:f.__setitem__('trial_id',f['trial_id'][:-1]+'1'),
 'seed_domain':lambda f:f['consumption'].__setitem__('domain','PROPOSED_SCIENCE/X'),
 'counter':lambda f:f['consumption']['draws'][0].__setitem__('counter_after',2),
 'width':lambda f:f['consumption']['draws'][0].__setitem__('width',159),
 'bit_value':lambda f:f['consumption']['draws'][0].__setitem__('value_hex','0x0'),
 'discard':lambda f:f['consumption']['draws'][0].__setitem__('discarded_bits',95),
 'missing_angle':lambda f:f.__setitem__('angle',None),
 'requery':lambda f:f['angle']['requery_input'].__setitem__('child',1),
 'angle_event':lambda f:f['angle'].__setitem__('event_fingerprint','0'*64),
 'angle_digest':lambda f:f['angle']['angle']['f3'].__setitem__('digest','0'*64),
 'original_trial_resealed':lambda f:reseal(f,'trial_id','wrong'),
 'original_source_resealed':lambda f:reseal(f,'source_id','wrong'),
 'original_policy_resealed':lambda f:reseal(f,'policy_id','wrong'),
 'original_branch_resealed':lambda f:reseal(f,'branch','COIN_REJECT'),
}
for name,change in MUTATIONS.items():
    def test(self,change=change):self.check_mutation(change)
    setattr(SemanticMutation,'test_reject_'+name,test)
