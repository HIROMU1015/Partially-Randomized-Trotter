"""Focused primary-agent artificial acceptance of the adopted pilot integration."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from hashlib import sha256
import tempfile,unittest,json,os,multiprocessing
from trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1 import guard as G
from trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1 import contract as K,spec as S,wrapper as W,synthetic as X,bits as B
from trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1 import integration as I,offline as O,run_control as A,retention as T
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1 import codec as C
ROOT=Path(__file__).resolve().parents[3]

def denied_spawn(queue):
    from trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1 import gate
    c=K.candidate(ROOT);m=K.run_manifest_candidate(c);errors=[]
    for f in [lambda:gate.dispatch(c,m,'/tmp/not-approved'),lambda:B.ScientificBits('ordinary','X',c,m,None),lambda:W.ScientificFactory(c,m,None)]:
        try:f()
        except Exception as exc:errors.append(str(exc))
    queue.put(dict(denials=errors,guard=G.observation()))


class Focused(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=K.candidate(ROOT)
    def test_contract_digest_formatter(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1.authorized_runner import contract_digest
        self.assertEqual(len(contract_digest(K.run_manifest_candidate(self.c))),64)
    def test_exact_plan(self):
        ts=S.tasks();self.assertEqual(len(ts),48);self.assertEqual(sum(2*t['n'] for t in ts),16896)
        self.assertEqual([sum(2*t['n'] for t in ts if t['phase_index']==i) for i in range(6)],[256,4096,4096]*2)
        self.assertEqual(len({t['key'] for t in ts}),48);self.assertEqual(12672*6+4224*7,105600)
    def test_routes_codec_compact_bits(self):
        for arm in S.ARMS:
            for layer in ('A','B'):
                m=K.manifest(self.c,'ARTIFICIAL_ONLY/RELEASE/unit','/tmp/b3-release-test-unit/output')
                task=next(t for t in m['tasks'] if t['phase_index']==1 and t['arm']==arm and t['layer']==layer)
                e=W.Endpoint(self.c,m,task,SimpleNamespace(mode='S'),symbols=X.Symbols());bits=B.ArtificialBits(arm,'X')
                for i in range(17):
                    tid=f'profile/baseline/{arm}/{layer}/X/{i}';o=e.sample(bits,tid)
                    angle=e.angle_for(o) if layer=='B' and o.kind=='EVENT' else None
                    a=o.evidence['adapted'];C.restore(a['codec'],source_identity=a['source_identity']).read()
                    compact=T.compact(dict(trial_id=tid,axis='X',kind=o.kind,event=o.event,evidence=o.evidence,angle=getattr(angle,'document',None),generator_exclusive_ns=0))
                    self.assertEqual(compact['original_digest'],a['codec']['original_fingerprint'])
                    self.assertFalse(compact['general_correctness_certified']);B.validate_trace(o.evidence['consumption'],arm,'X')
    def test_artificial_preflight_and_twenty_mutations(self):
        m=K.manifest(self.c,'ARTIFICIAL_ONLY/RELEASE/unit','/tmp/b3-release-test-unit/output');K.preflight(self.c,m)
        mutations=[('planned',16897),('retry',1),('resume',True),('seed_namespace',S.SEED),('tasks',[]),('workers',9),
            ('byte_limits',{}),('call_budgets',{}),('retention','ALL_DROP'),('observer','ON'),('codec','reference'),
            ('coverage','CONTINUE'),('phase_order',[]),('runtime_binding','bad'),('global_claim','bad'),
            ('storage_destination','/tmp/wrong'),('SCIENCE_RUN_AUTHORIZED',True),('artificial_only',False),('approval_record',{}),('activation_record',{})]
        for field,value in mutations:
            wrong=deepcopy(m);wrong[field]=value
            with self.subTest(field=field),self.assertRaises(Exception):K.preflight(self.c,wrong)
        wrong=deepcopy(self.c);wrong['source_pins'][next(iter(wrong['source_pins']))]='0'*64
        with self.assertRaises(Exception):K.preflight(wrong,m)
    def test_closed_dispatch(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1 import gate
        m=K.run_manifest_candidate(self.c)
        with self.assertRaises(Exception):gate.dispatch(self.c,m,'/tmp/unapproved')
        self.assertEqual(G.observation()['core_loaded'],[])
    def test_spawn_dispatch_denials(self):
        ctx=multiprocessing.get_context('spawn');q=ctx.Queue();p=ctx.Process(target=denied_spawn,args=(q,))
        p.start();p.join();self.assertEqual(p.exitcode,0);row=q.get(timeout=10)
        self.assertEqual(len(row['denials']),3);self.assertEqual(row['guard']['core_loaded'],[])
    def test_live_metadata_pure_fixture_no_permit(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1 import metadata as M
        m=K.run_manifest_candidate(self.c,source_commit='1'*40)
        m.update(SCIENCE_RUN_AUTHORIZED=True,artificial_only=False,content_adopted=True,run_ready=True,contract_grants_execution=True,
            science_release='1'*40,scientific_adoption='USER_ADOPTED_16896_PILOT',approval_status='EXPLICIT_USER_EXECUTION_APPROVED',
            activation_record=dict(id='PURE_FORMAT_FIXTURE_NOT_A_PERMIT',purpose='USER_APPROVED_ONE_SHOT_16896_PILOT'))
        text='PURE ARTIFICIAL METADATA FIXTURE: grants no execution'
        m['approval_record']=dict(approver='USER',explicit_execution_approval=True,approval_text=text,
            approval_text_sha256=sha256(text.encode()).hexdigest(),measured_SOURCE='1'*40,
            manifest_binding=S.digest({k:v for k,v in m.items() if k!='approval_record'}))
        meta=M.context(m);self.assertFalse(meta['artificial_only']);self.assertFalse(meta['external_authority_used'])
        report=M.stamp_report(dict(states=dict(COMPLETED=16),pairs=[]),m)
        self.assertEqual(report['live_API_observation']['status'],'PARTIAL_TRIAL_PREFIX');self.assertFalse(report['science_correctness_certified'])
        wrong=deepcopy(m);wrong['artificial_only']=True
        with self.assertRaises(Exception):M.context(wrong)
        self.assertEqual(G.observation()['core_loaded'],[])
    def test_no_refund(self):
        with tempfile.TemporaryDirectory(prefix='b3-release-test-claim-') as td:
            m=K.manifest(self.c,'ARTIFICIAL_ONLY/RELEASE/claim',str(Path(td)/'output'));s=A.SimpleStore(td)
            s.claim(m,A.mock_approval(m));s.enter(m);s.stop(m)
            with self.assertRaises(Exception):s.enter(m)
            with self.assertRaises(Exception):s.claim(m,A.mock_approval(m))
            self.assertEqual(s.snapshot(m)['claim'][3:5],['STOPPED',1])

class Finalization(unittest.TestCase):
    def test_receipt_close_preserves_unknown_stop(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1.bridge import Fault,_save
        from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_storage_v2 as V2
        with tempfile.TemporaryDirectory(prefix='b3-release-test-finalize-') as td:
            root=Path(td);out=root/'output';out.mkdir();c=K.candidate(ROOT)
            m=K.manifest(c,'ARTIFICIAL_ONLY/RELEASE/close',str(out));store=A.SimpleStore(root)
            store.claim(m,A.mock_approval(m));store.stop(m)
            fault=Fault(dict(point='receipt.json.close'));settings=V2.P.Settings(mode='S',profile_n=256,codec='fused',workload='light',call_limit=16896,workers=8,result_bytes=1<<36,log_bytes=1<<28)
            ctl=I.Coordinator(settings,out,m['tasks'],16896,m['run_id'],fault,manifest=m);fault.control=ctl;ctl.parent_save_cost={}
            ctl.budget.reserve('result',4096);ctl.stop(dict(artificial_control_fixture=True));V2.close_storage(ctl);ctl.closed=True;ctl.budget.notify=ctl.post_notify
            with self.assertRaises(OSError) as caught:_save(out,'receipt.json',dict(artificial_control_fixture=True,success=False),ctl,fault)
            report={};I.preserve_finalization_failure(out,m,store,ctl,report,caught.exception)
            self.assertEqual(report['classification'],'INCOMPLETE_EVIDENCE')
            self.assertTrue(O.read(out/'STOP_failure_marker.json')['mandatory_STOP'])
            self.assertEqual(store.snapshot(m)['execution_state'],'UNKNOWN')
            self.assertEqual(store.snapshot(m)['claim'][4],1)


class Integration(unittest.TestCase):
    def execute_case(self,name):
        base=Path(os.environ.get('B3_PILOT_ARTIFICIAL_ROOT','/tmp'));base.mkdir(parents=True,exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix='b3-release-test-pilot-'+name+'-',dir=base));c=K.candidate(ROOT)
        m=K.manifest(c,'ARTIFICIAL_ONLY/RELEASE/'+root.name,str(root/'output'),scenario=name)
        s=A.SimpleStore(root);result=I.run(c,m,root,s)
        (root/'test_return.json').write_text(json.dumps(O.read(root/'output/final.json'),indent=2)+'\n')
        print('ARTIFICIAL_CASE',name,root,result['classification'],flush=True)
        self.assertEqual(G.observation()['core_loaded'],[])
        self.assertTrue(s.snapshot(m)['claim'][4]);self.assertTrue((root/'output/STOP_marker.json').exists())
        return root,c,m,result
    def test_normal(self):
        root,c,m,r=self.execute_case('normal');self.assertTrue(r['complete'],r.get('failures'))
        check=O.inspect(root,c,m);(root/'test_inspection.json').write_text(json.dumps(check,indent=2)+'\n')
        self.assertEqual(check['status'],'ARTIFICIAL_TECHNICAL_COMPLETE');self.assertEqual(r['states']['COMPLETED'],16896)
        self.assertEqual([p['phase_index'] for p in r['phases']],list(range(6)))
        self.assertEqual(O.read(root/'output/pair_validation.json')['full_documents'],1536)
        self.assertEqual(O.read(root/'output/pair_validation.json')['compact_documents'],15360)
        self.assertFalse(r['parent_guard']['core_loaded'])
    def test_coverage(self):
        root,c,m,r=self.execute_case('coverage');self.assertEqual(r['classification'],'INCONCLUSIVE_COVERAGE')
        self.assertEqual(len(r['phases']),1);self.assertEqual(r['states']['COMPLETED'],256)
        self.assertFalse((root/'output/receipt.json').exists());self.assertEqual(O.inspect(root,c,m)['status'],'INCONCLUSIVE_COVERAGE')
    def test_angle_f01(self):
        root,c,m,r=self.execute_case('S_angle');self.assertFalse(r['complete']);self.assertEqual(len(r['phases']),1)
        self.assertTrue(any('failure' in str(v).lower() for v in r['pairs']));self.assertFalse((root/'output/receipt.json').exists())
    def test_unknown(self):
        root,c,m,r=self.execute_case('unknown_ack');self.assertFalse(r['complete']);self.assertEqual(len(r['phases']),1)
        self.assertTrue(any('UNKNOWN' in k and v for k,v in r['states'].items()))
        self.assertEqual(O.read(root/'output/simple_run_state.json')['execution_state'],'UNKNOWN')

if __name__=='__main__':unittest.main(verbosity=2)
