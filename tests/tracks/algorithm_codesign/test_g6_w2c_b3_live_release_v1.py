"""Primary-agent tests of owned artificial API objects and isolated test stores."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import tempfile,unittest,json,os,multiprocessing
import shutil
from trottertracks.algorithm_codesign.g6_w2c_b3_live_release_v1 import guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_live_release_v1 import contract as K, wrapper as W, synthetic as X
from trottertracks.algorithm_codesign.g6_w2c_b3_live_release_v1 import bits as B, authority as A
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import spec as S
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import codec as C

ROOT=Path(__file__).resolve().parents[3]


def fixture(c,arm='ordinary',layer='B',scenario='normal'):
    m=K.manifest(c,'ARTIFICIAL_ONLY/RELEASE/unit','/tmp/b3-release-test-example/output',scenario=scenario)
    task=next(t for t in m['tasks'] if t['condition']=='S' and t['arm']==arm and t['layer']==layer)
    return W.Endpoint(c,m,task,SimpleNamespace(mode='S'),symbols=X.Symbols()),m


def gate_worker(queue):
    G.install()
    try:G.science_dispatch(run_ready=True,fake_approval=True,grant='fake')
    except Exception as exc:queue.put(dict(denied=True,reason=str(exc),guard=G.observation()))
    else:queue.put(dict(denied=False))


class Unit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=K.candidate(ROOT)
    def test_all_api_routes_roundtrip(self):
        for arm in S.ARMS:
            for layer in ('A','B'):
                endpoint,m=fixture(self.c,arm,layer);bits=B.ArtificialBits(arm,'X');before=endpoint.shared_identity()
                for i in range(16):
                    tid=f'correctness/baseline/{arm}/{layer}/X/{i}';outcome=endpoint.sample(bits,tid)
                    if layer=='B' and outcome.kind=='EVENT':endpoint.angle_for(outcome)
                    a=outcome.evidence['adapted'];original=C.restore(a['codec'],source_identity=a['source_identity'])
                    self.assertIsInstance(original.read(),dict)
                    B.validate_trace(outcome.evidence['consumption'],arm,'X')
                self.assertEqual(before,endpoint.shared_identity())
    def test_runtime_document_digest(self):
        value=X.Evidence(dict(exact=1));value.digest='0'*64
        with self.assertRaises(Exception):W.from_runtime(value)
    def test_parent_gate_closed(self):
        for fn in [G.science_dispatch,A.RealAuthority().admit]:
            with self.assertRaises(Exception):fn(fake_approval=True,run_ready=True)
        self.assertEqual(G.observation()['core_loaded'],[])
    def test_spawn_gate_closed(self):
        ctx=multiprocessing.get_context('spawn');q=ctx.Queue();p=ctx.Process(target=gate_worker,args=(q,))
        p.start();p.join();self.assertEqual(p.exitcode,0)
        r=q.get(timeout=10);self.assertTrue(r['denied']);self.assertEqual(r['guard']['core_loaded'],[])
    def test_original_typed_failure_preserved(self):
        e,m=fixture(self.c)
        original=X.Evidence(dict(schema='g6_f3_failure_v1',category='ARTIFICIAL_TYPED_FAILURE',
            reason='ARTIFICIAL_SOURCE_FAILURE',stage='sample',input_id=None,details={},
            mandatory_STOP=True,run_complete=False,is_random_zero=False))
        error=RuntimeError('ARTIFICIAL_TYPED_FAILURE');error.evidence=original
        failure=e._fail(error,'release.sample')
        transport=failure.record.read()['details']['original_scientific_failure']
        self.assertEqual(transport['document'],original.document);self.assertEqual(transport['digest'],original.digest)
    def test_audit_failure_and_acquired_result_preserved(self):
        e,m=fixture(self.c,scenario='scientific_mismatch')
        with self.assertRaises(Exception) as caught:
            e.sample(B.ArtificialBits('ordinary','X'),'correctness/baseline/ordinary/B/X/0')
        details=caught.exception.record.read()['details']
        audit=details['original_driver_audit_failure']
        self.assertEqual(C.DriverDocument(audit['text'],audit['digest']).read()['completed_calls'],0)
        result=details['acquired_original_result']
        self.assertEqual(C.Document(result['document'],result['digest']).read()['schema'],'g7_observation_candidate_v1')
        self.assertEqual(len(details['observed_bit_consumption']['draws']),2)
        self.assertTrue(e.stopped)
    def test_bool_width_terminal(self):
        b=B.ArtificialBits('ordinary','X')
        with self.assertRaises(Exception):b.bits(True)
        self.assertTrue(b.stopped);self.assertEqual(b.failure_position['counter'],0)
    def test_call_plan_and_upper(self):
        self.assertEqual(sum(t['n']*2 for t in S.tasks()),512)
        self.assertEqual(384*6+128*7,K.call_budgets()['bit_draw_requests_static_upper'])
        self.assertIn('Nested',K.counter_semantics()['event_witness'])
    def test_fake_symbols_only(self):
        e,m=fixture(self.c)
        with self.assertRaises(Exception):W.Endpoint(self.c,m,e.task,e.settings,symbols=SimpleNamespace())
    def test_bad_angle_terminal(self):
        e,m=fixture(self.c,scenario='S_angle');outcome=e.sample(B.ArtificialBits('ordinary','X'),'correctness/baseline/ordinary/B/X/0')
        with self.assertRaises(Exception):e.angle_for(outcome)
        self.assertTrue(e.stopped)
        with self.assertRaises(Exception):e.sample(B.ArtificialBits('ordinary','X'),'correctness/baseline/ordinary/B/X/1')


class Preflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=K.candidate(ROOT);cls.m=K.manifest(cls.c,'ARTIFICIAL_ONLY/RELEASE/preflight','/tmp/b3-release-test-bind/output')
    def test_positive(self):self.assertEqual(K.preflight(self.c,self.m),S.digest(self.m))


for field,value in [('api_signatures',{}),('input',{}),('precision',{}),('python',{}),
 ('source_release_candidate','0'*64),('previous_artifact','0'*64),('worker_runtime',{}),
 ('science_release','fake'),('run_ready',True),('SCIENCE_RUN_AUTHORIZED',True)]:
    def test(self,field=field,value=value):
        c=deepcopy(self.c);c[field]=value
        with self.assertRaises(Exception):K.preflight(c,self.m)
    setattr(Preflight,'test_wrong_'+field,test)
for field,value in [('seed_namespace','SCIENCE'),('global_claim','fake'),('tasks',[]),('planned',16896),
 ('retry',1),('resume',True),('real_authority',True),('real_grant','fake'),('call_budgets',{}),
 ('runtime_binding','0'*64),('storage_destination','/tmp/wrong'),('authority_binding','0'*64)]:
    def test(self,field=field,value=value):
        m=deepcopy(self.m);m[field]=value
        with self.assertRaises(Exception):K.preflight(self.c,m)
    setattr(Preflight,'test_manifest_'+field,test)


def claim_worker(root,m,queue,crash='none'):
    G.install()
    try:
        store=A.TestStore(root,create=False);store.claim(m,A.test_grant(m),crash=crash)
        queue.put(dict(status='WIN',guard=G.observation()))
    except Exception as exc:queue.put(dict(status='DENIED',reason=str(exc),guard=G.observation()))


class Store(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='b3-release-test-authority-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.store=A.TestStore(self.root)
        self.m=dict(global_scope='ARTIFICIAL_ONLY/RELEASE/test',global_claim='TEST_CLAIM',retry=0,resume=False,
            SCIENCE_RUN_AUTHORIZED=False,real_authority=False,real_grant=None)
        self.raw=[]
    def tearDown(self):
        if dest:=os.environ.get('B3_RELEASE_AUTHORITY_RAW'):
            value=dict(case=self._testMethodName,manifest=self.m,snapshot=self.store.snapshot(self.m),
                observations=self.raw,guard=G.observation(),real_authority=False)
            (self.root/'test_observation.json').write_text(json.dumps(value,indent=2)+'\n')
            target=Path(dest)/self._testMethodName;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(self.root,target)
    def test_claim_stop_no_enter_no_reuse(self):
        g=A.test_grant(self.m);self.store.claim(self.m,g);self.store.enter(self.m);self.store.stop(self.m)
        with self.assertRaises(Exception):self.store.enter(self.m)
        with self.assertRaises(Exception):self.store.claim(self.m,g)
        self.assertEqual(self.store.snapshot(self.m)['claim'][3:5],['STOPPED',1])
    def test_wrong_grant(self):
        g=A.test_grant(self.m)
        with self.assertRaises(Exception):self.store.claim(self.m,A.TestGrant('wrong',g.scope,g.claim))
        self.assertFalse(self.store.snapshot(self.m)['intent_exists'])
    def test_fake_real_confusion(self):
        m=deepcopy(self.m);m['real_authority']=True
        with self.assertRaises(Exception):A.test_grant(m)
    def test_revocation_before_attempt(self):
        g=A.test_grant(self.m);self.store.revoke_test(g)
        with self.assertRaises(Exception):self.store.claim(self.m,g)
    def test_receipt_once(self):
        self.store.claim(self.m,A.test_grant(self.m));self.store.stop(self.m);self.store.link_receipt(self.m,'a'*64)
        with self.assertRaises(Exception):self.store.link_receipt(self.m,'a'*64)
        self.assertEqual(self.store.snapshot(self.m)['claim'][5],'a'*64)
    def test_atomic_one_winner(self):
        ctx=multiprocessing.get_context('spawn');q=ctx.Queue();processes=[ctx.Process(target=claim_worker,args=(str(self.root),self.m,q)) for _ in range(12)]
        for p in processes:p.start()
        for p in processes:p.join();self.assertEqual(p.exitcode,0)
        rows=[q.get(timeout=10) for _ in processes];self.assertEqual(sum(r['status']=='WIN' for r in rows),1)
        self.raw=rows
        self.assertTrue(all(not r['guard']['core_loaded'] for r in rows));self.store.stop(self.m)


for point,exitcode in [('after_intent',61),('before_commit',62),('after_commit',63)]:
    def test(self,point=point,exitcode=exitcode):
        ctx=multiprocessing.get_context('spawn');q=ctx.Queue();p=ctx.Process(target=claim_worker,args=(str(self.root),self.m,q,point))
        p.start();p.join();self.assertEqual(p.exitcode,exitcode)
        self.raw=[dict(crash_point=point,exit_code=p.exitcode)]
        reopened=A.TestStore(self.root,create=False)
        with self.assertRaises(Exception):reopened.claim(self.m,A.test_grant(self.m))
        reopened.stop(self.m)
        with self.assertRaises(Exception):reopened.enter(self.m)
        self.assertIn(reopened.snapshot(self.m)['claim'][3],('STOPPED','UNKNOWN_STOPPED'))
    setattr(Store,'test_crash_'+point,test)
