"""Primary-agent tests of owned artificial API objects and isolated test stores."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import tempfile,unittest,json,os,multiprocessing
import shutil
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import contract as K, wrapper as W, synthetic as X
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import bits as B, run_control as A
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import spec as S
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import codec as C

ROOT=Path(__file__).resolve().parents[3]


def fixture(c,arm='ordinary',layer='B',scenario='normal'):
    m=K.manifest(c,'ARTIFICIAL_ONLY/RELEASE/unit','/tmp/b3-release-test-example/output',scenario=scenario)
    task=next(t for t in m['tasks'] if t['condition']=='S' and t['arm']==arm and t['layer']==layer)
    return W.Endpoint(c,m,task,SimpleNamespace(mode='S'),symbols=X.Symbols()),m


def gate_worker(queue):
    G.install()
    from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import gate,integration
    c=K.candidate(ROOT);m=K.run_manifest_candidate(c)
    m.update(SCIENCE_RUN_AUTHORIZED=True,run_ready=True,approval_record={'fake_approval':True})
    permit=gate.LivePermit(m,S.digest(c),'FAKE',os.getpid());denials=[]
    for operation in [lambda:gate.dispatch(c,m,'/tmp/fake'),lambda:W.ScientificFactory(c,m,permit),
        lambda:B.ScientificBits('ordinary','X',c,m,permit),lambda:integration.run_live(c,m,'/tmp/fake',permit)]:
        try:operation()
        except Exception as exc:denials.append(str(exc))
    queue.put(dict(denied=len(denials)==4,reasons=denials,guard=G.observation()))


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
        from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import activation
        for fn in [G.science_dispatch,activation.admit]:
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
        store=A.SimpleStore(root,create=False);store.claim(m,A.mock_approval(m),crash=crash)
        queue.put(dict(status='WIN',guard=G.observation()))
    except Exception as exc:queue.put(dict(status='DENIED',reason=str(exc),guard=G.observation()))


class Store(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='b3-release-test-authority-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.store=A.SimpleStore(self.root)
        self.m=dict(global_scope='ARTIFICIAL_ONLY/RELEASE/test',global_claim='TEST_CLAIM',retry=0,resume=False,
            SCIENCE_RUN_AUTHORIZED=False,real_authority=False,real_grant=None,artificial_only=True,run_id='ARTIFICIAL_UNIT_RUN')
        self.raw=[]
    def tearDown(self):
        if dest:=os.environ.get('B3_FINAL_CONTROL_RAW'):
            try:snapshot=self.store.snapshot(self.m)
            except G.Closed as exc:
                if self._testMethodName!='test_state_symlink_denied':raise
                snapshot=dict(expected_snapshot_denial=str(exc),state_location_rejected=True)
            value=dict(case=self._testMethodName,manifest=self.m,snapshot=snapshot,
                observations=self.raw,guard=G.observation(),real_authority=False)
            (self.root/'test_observation.json').write_text(json.dumps(value,indent=2)+'\n')
            target=Path(dest)/self._testMethodName;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(self.root,target,symlinks=True)
    def test_claim_stop_no_enter_no_reuse(self):
        g=A.mock_approval(self.m);self.store.claim(self.m,g);self.store.enter(self.m);self.store.stop(self.m)
        with self.assertRaises(Exception):self.store.enter(self.m)
        with self.assertRaises(Exception):self.store.claim(self.m,g)
        self.assertEqual(self.store.snapshot(self.m)['claim'][3:5],['STOPPED',1])
    def test_wrong_grant(self):
        g=A.mock_approval(self.m)
        with self.assertRaises(Exception):self.store.claim(self.m,A.MockApproval('wrong',g.scope,g.claim))
        self.assertFalse(self.store.snapshot(self.m)['intent_exists'])
    def test_fake_real_confusion(self):
        m=deepcopy(self.m);m['SCIENCE_RUN_AUTHORIZED']=True
        with self.assertRaises(Exception):A.mock_approval(m)
    def test_receipt_once(self):
        self.store.claim(self.m,A.mock_approval(self.m));self.store.stop(self.m);self.store.link_receipt(self.m,'a'*64)
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
        reopened=A.SimpleStore(self.root,create=False)
        with self.assertRaises(Exception):reopened.claim(self.m,A.mock_approval(self.m))
        reopened.stop(self.m)
        with self.assertRaises(Exception):reopened.enter(self.m)
        self.assertEqual(reopened.snapshot(self.m)['execution_state'],'UNKNOWN')
        self.assertEqual(reopened.snapshot(self.m)['claim'][3],'UNKNOWN_STOPPED')
    setattr(Store,'test_crash_'+point,test)


from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import resolver as L,gate,integration
from hashlib import sha256
import ast,inspect


class Connection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=K.candidate(ROOT)
    def test_static_scientific_bindings(self):
        bound=L.static_binding(ROOT)
        self.assertEqual(bound,self.c['final_runtime_binding'])
        self.assertTrue(set(G.CORE).issubset(bound['files']))
        self.assertGreaterEqual(len(bound['symbols']),28)
        self.assertEqual(G.observation()['core_loaded'],[])
    def test_loader_source_is_after_gate(self):
        tree=ast.parse((ROOT/K.PKG/'resolver.py').read_bytes())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve')
        self.assertEqual(ast.unparse(node.body[0]),'require_live(permit, c, m)')
        self.assertTrue(any(isinstance(n,ast.Call) and ast.unparse(n.func)=='importlib.import_module' for n in ast.walk(node)))
        self.assertFalse(G.ATTEMPTS)
    def test_runtime_binding_using_artificial_module(self):
        import types
        with tempfile.TemporaryDirectory(prefix='b3-final-binding-') as tmp:
            root=Path(tmp);path=root/'fixture.py'
            path.write_text('class Fixture:\n    @classmethod\n    def create(cls, x, *, policy=None): return x\n')
            namespace={};exec(compile(path.read_text(),str(path),'exec'),namespace)
            module=types.SimpleNamespace(__file__=str(path),Fixture=namespace['Fixture'])
            node=ast.parse(path.read_text()).body[0].body[0]
            binding=dict(files=dict(fixture=dict(path='fixture.py',sha256=sha256(path.read_bytes()).hexdigest())),
                symbols={'fixture.Fixture.create':dict(path='fixture.py',qualname='Fixture.create',shape=L.shape(node))})
            self.assertTrue(L.bind_loaded(dict(fixture=module),binding,root))
            binding['symbols']['fixture.Fixture.create']['shape']=[]
            with self.assertRaises(Exception):L.bind_loaded(dict(fixture=module),binding,root)
    def test_wrong_loaded_location(self):
        fake=SimpleNamespace(__file__='/tmp/wrong')
        binding=dict(files=dict(fixture=dict(path='AGENTS.md',sha256='0'*64)),symbols={})
        with self.assertRaises(Exception):L.bind_loaded(dict(fixture=fake),binding,ROOT)
    def test_forged_permit_still_closed(self):
        m=K.run_manifest_candidate(self.c);m.update(SCIENCE_RUN_AUTHORIZED=True,run_ready=True)
        permit=gate.LivePermit(m,S.digest(self.c),'FAKE_USER_APPROVAL',os.getpid())
        for operation in [lambda:gate.require_live(permit,self.c,m),lambda:gate.dispatch(self.c,m,'/tmp/fake'),
            lambda:W.ScientificFactory(self.c,m,permit),
            lambda:B.ScientificBits('ordinary','X',self.c,m,permit),
            lambda:integration.run_live(self.c,m,'/tmp/fake',permit)]:
            with self.assertRaises(Exception):operation()
        self.assertFalse(G.ATTEMPTS)
    def test_run_manifest_closed_and_exact(self):
        m=K.run_manifest_candidate(self.c)
        self.assertFalse(m['SCIENCE_RUN_AUTHORIZED']);self.assertFalse(m['run_ready'])
        self.assertEqual(m['planned'],512);self.assertEqual(m['science_SOURCE'],S.SCIENCE)
        self.assertIsNone(m['approval_record']);self.assertTrue(m['run_id'].startswith('B3_CORRECTNESS_512/'))
        self.assertEqual(m['seed_namespace'],S.SEED);self.assertFalse(m['mode_in_seed'])
    def test_delegated_counter_artificial_only(self):
        # Exercises recorder with our ARTIFICIAL namespace, never the science one.
        p=B.ArtificialBits('ordinary','X');r=B.CounterRecorder(p,p.domain)
        values=[r.bits(160) for _ in range(3)]
        self.assertEqual(r.counter,3);self.assertEqual(r.consumed_bits,480)
        self.assertEqual(values,[int(x['value_hex'],16) for x in r.trace])
        self.assertEqual(len(p.trace),3)
    def test_delegated_counter_invalid_consumption_stops(self):
        p=SimpleNamespace(counter=0,consumed_bits=0,bits=lambda width:0)
        r=B.CounterRecorder(p,'ARTIFICIAL_ONLY/INVALID')
        with self.assertRaises(Exception):r.bits(160)
        self.assertTrue(r.stopped);self.assertEqual(r.failure_position['draw_index'],0)
    def test_delegated_counter_invalid_width(self):
        p=B.ArtificialBits('ordinary','X');r=B.CounterRecorder(p,p.domain)
        with self.assertRaises(Exception):r.bits(True)
        self.assertTrue(r.stopped);self.assertEqual(p.counter,0)
    def test_no_plain_none_classification(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import result as R
        with self.assertRaises(Exception):R.adapt_g7(None,observation=None,expected={},arm='full_return')
    def test_legacy_endpoint_versioned_subclass(self):
        from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_driver as D
        self.assertTrue(issubclass(W.Endpoint,D.LegacyEndpoint))
    def test_original_evidence_object_to_full_angle(self):
        e,m=fixture(self.c,'full_return','B');bits=B.ArtificialBits('full_return','X')
        for i in range(16):
            out=e.sample(bits,f'correctness/baseline/full_return/B/X/{i}')
            if out.kind=='EVENT':break
        else:self.fail('artificial accepted fixture absent')
        original=e.runtime_record;observed=[];call=e.session.angle_for_event
        e.session.angle_for_event=lambda record:(observed.append(record) or call(record))
        e.angle_for(out);self.assertIs(observed[0],original)
    def test_nonreduced_coin_and_event_artificial_paths(self):
        e,m=fixture(self.c,'full_return','B');bits=B.ArtificialBits('full_return','X');seen=set()
        for i in range(16):seen.add(e.sample(bits,f'correctness/baseline/full_return/B/X/{i}').kind)
        self.assertEqual(seen,{'EVENT','ZERO_R','ZERO_C'})


class SimpleState(Store):
    # Inherits focused persistence tests on the new implementation.
    def test_not_started_then_spent(self):
        self.assertEqual(self.store.execution_state(self.m),'NOT_STARTED')
        self.store.claim(self.m,A.mock_approval(self.m))
        self.assertEqual(self.store.execution_state(self.m),'START_RESERVED/SPENT')
        self.store.stop(self.m);self.assertEqual(self.store.execution_state(self.m),'STOPPED')
    def test_same_run_different_scope_denied(self):
        self.store.claim(self.m,A.mock_approval(self.m));other=deepcopy(self.m)
        other['global_scope']+='-other';other['global_claim']='OTHER'
        with self.assertRaises(Exception):self.store.claim(other,A.mock_approval(other))
    def test_same_scope_different_run_denied(self):
        self.store.claim(self.m,A.mock_approval(self.m));other=deepcopy(self.m)
        other['run_id']+='-other';other['global_claim']='OTHER'
        with self.assertRaises(Exception):self.store.claim(other,A.mock_approval(other))
        self.assertEqual(self.store.execution_state(other),'UNKNOWN')
    def test_explicit_unknown_never_enter_refund(self):
        self.store.claim(self.m,A.mock_approval(self.m));self.store.stop(self.m,unknown=True)
        self.assertEqual(self.store.execution_state(self.m),'UNKNOWN')
        self.store.stop(self.m);self.assertEqual(self.store.execution_state(self.m),'UNKNOWN')
        with self.assertRaises(Exception):self.store.enter(self.m)
        with self.assertRaises(Exception):self.store.claim(self.m,A.mock_approval(self.m))
    def test_real_store_creation_requires_gate(self):
        with tempfile.TemporaryDirectory(prefix='b3-final-live-unapproved-') as p:
            with self.assertRaises(Exception):A.SimpleStore(p)
            self.assertEqual(list(Path(p).iterdir()),[])
    def test_cannot_complete_without_caller_evidence(self):
        self.store.claim(self.m,A.mock_approval(self.m));self.store.stop(self.m)
        self.m['storage_destination']=str(self.root/'output')
        with self.assertRaises(Exception):self.store.mark_completed(self.m)
    def test_state_symlink_denied(self):
        self.store.path.rename(self.root/'original.sqlite3')
        self.store.path.symlink_to(self.root/'original.sqlite3')
        with self.assertRaises(Exception):self.store.enter(self.m)


class SetupDocuments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=K.candidate(ROOT)
    def test_artificial_setup_digest_and_change_rejected(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import offline as O
        e,m=fixture(self.c);doc,setup=O.validate_setup(e.setup_document.transport(),self.c,m,e.task,artificial=True)
        self.assertEqual(doc,e.setup_document)
        setup['precision']['root_bits']=1
        with self.assertRaises(Exception):O.validate_setup(C.Document.create(setup).transport(),self.c,m,e.task,artificial=True)
    def test_held_canonical_document_context_only(self):
        # Artificial document data exercises the live-shaped read-only branch.
        # No generator, science law, real permit or scientific bits are created.
        from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import offline as O
        from fractions import Fraction as F
        m=K.manifest(self.c,'ARTIFICIAL_ONLY/RELEASE/setup','/tmp/b3-release-test-example/output')
        task=m['tasks'][1]
        held=dict(p=tuple(map(F,S.INPUT['p'])),x=F(S.INPUT['x']),m=5,arm='ordinary',
            eta=F(S.PRECISION['eta']),rho=F(S.PRECISION['rho']),K=256,H=160,fixture_only=True)
        snap=dict(held,conditional_witnesses=())
        context=dict(input_id=S.digest(S.INPUT),policy_id=S.digest([S.PRECISION,S.ANGLE]),
            source_id=self.c['source_release_candidate'],preflight_id=O.driver_hash(snap),sigma=1)
        value=dict(state=held,context=context,preflight_record=None,canonical_preflight_snapshot=snap,
            input=S.INPUT,precision=S.PRECISION,angle=S.ANGLE,arm='ordinary',layer='B',
            source_binding=self.c['source_release_candidate'],manifest_binding=S.digest(m))
        O.validate_setup(C.Document.create(value).transport(),self.c,m,task,artificial=False)
        value['context']['preflight_id']='wrong'
        with self.assertRaises(Exception):O.validate_setup(C.Document.create(value).transport(),self.c,m,task,artificial=False)


def partial_intent_test(self):
    self.store._run_intent(self.m).write_bytes(b'{')
    self.assertEqual(self.store.execution_state(self.m),'UNKNOWN')
    with self.assertRaises(Exception):self.store.claim(self.m,A.mock_approval(self.m))
    with self.assertRaises(Exception):self.store.enter(self.m)
    self.assertEqual(self.store.execution_state(self.m),'UNKNOWN')
SimpleState.test_partial_intent_spent_unknown=partial_intent_test


def patched_anchor_denial(self):
    from unittest.mock import patch
    from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import activation
    m=K.run_manifest_candidate(self.c)
    with patch.object(activation,'APPROVED_ACTIVATION_ID','a'*64):
        with self.assertRaisesRegex(Exception,'ACTIVATION_SOURCE_CONSTANT_MISMATCH'):
            gate.dispatch(self.c,m,'/tmp/fake')
Connection.test_runtime_attribute_cannot_activate=patched_anchor_denial


def wrong_final_location(self):
    from unittest.mock import patch
    m=K.manifest(self.c,'ARTIFICIAL_ONLY/RELEASE/final-location','/tmp/b3-release-test-loc/output')
    with patch.object(W,'__file__','/tmp/unapproved-wrapper.py'):
        with self.assertRaisesRegex(Exception,'LOADED_LOCATION'):
            K.preflight(self.c,m)
Connection.test_new_namespace_import_location=wrong_final_location
