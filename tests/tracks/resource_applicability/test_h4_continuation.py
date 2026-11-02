"""Small byte/ledger fixtures and mock workers; no real run or compiler."""
import copy
from concurrent.futures import Future
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from trottertracks.resource_applicability.h4_geometry import continuation as c
from trottertracks.resource_applicability.h4_geometry import continuation_binding as cb
from trottertracks.resource_applicability.h4_geometry import execution, circuits, parallel, signal, launch_binding as lb
from trottertracks.resource_applicability.h4_geometry.ledger import Ledger, wire, json_bytes
from trottertracks.resource_applicability.h4_geometry.resources import OutputBudget
from trottertracks.resource_applicability.h4_geometry.identity import Stop, fingerprint, trajectory_seed

ROOT=Path(__file__).resolve().parents[3]
BASE=json.loads((ROOT/(c.PARENT_BUNDLE+'plan_authorized_v20.json')).read_text())
METRICS=dict(rz_count=3,rz_depth=2,cx_count=1,cx_depth=1,total_depth=4,circuit_size=5)


class Run:
    def __init__(self):
        self.workers=4;self.pool=self;self.calls=[];self.aborts=0;self.phases=[]
    def pulse(self):pass
    def phase(self,name):self.phases.append(name)
    def submit(self,function,circuit,options):
        self.calls.append(circuit);future=Future();future.set_result(METRICS);return future
    def wait_any(self,futures):return {futures[-1]}
    def abort(self):self.aborts+=1


class Fixture:
    def __init__(self, root):
        self.root=root;self.parent=copy.deepcopy(BASE)
        self.parent['output_root']=str(root/'parent-output');self.parent['control_root']=str(root/'parent-control')
        self.template=next(t for t in BASE['templates'] if t['method']=='B0')
        self.identity=c.identity_for(self.parent,'0.70',self.template)
        self.expected=[wire(self.identity,axis,None,None,str(i+1)*64) for i,axis in enumerate(('cosine','sine'))]
        self.budget=OutputBudget(Path(self.parent['output_root']))
        self.ledger=Ledger(self.budget,cap=74805)
        self.records=[]
        control=Path(self.parent['control_root']);control.mkdir()
        now=time.monotonic()
        roles=[]
        for i,role in enumerate(('driver','observer','worker','worker','worker','worker')):
            roles.append(dict(role=role,pid=990000+i,start='42',parent=1 if i==0 else 990000,uid=os.getuid()))
        self.started=dict(source=c.PARENT_SOURCE,artifact_commit=c.PARENT_REVIEW,run_id=c.PARENT_RUN,
            roles=roles,driver_pid=990000,observer_pid=990001,worker_pids=[990002,990003,990004,990005],
            observer_observation=dict(monotonic=now-9,wall_seconds=1.))
        self.clock=dict(schema_version='h4-run10-clock-v1',boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            identities=[{k:r[k] for k in c.IDENTITY_FIELDS} for r in roles],wall_origin_monotonic=now-10,
            historical_exit_code=None,exact_exit_time=None)
        self.proof=dict(schema_version='h4-run10-native-stop-v1',run_id=c.PARENT_RUN,source_commit=c.PARENT_SOURCE,
            artifact_commit=c.PARENT_REVIEW,clock=self.clock,native_audits=[dict(monotonic=now-5,identities=self.clock['identities'],same_owned_remaining=0),
            dict(monotonic=now-3,identities=self.clock['identities'],same_owned_remaining=0)],historical_exit_code=None,exact_exit_time=None,exit_data_not_inferred=True)
        self.paths=[]
        for name,value in zip(('started','clock','proof'),(self.started,self.clock,self.proof)):
            p=root/(name+'.json');p.write_bytes(json_bytes(value));self.paths.append(p)
        (control/'one-shot.json').write_bytes(json_bytes(dict(run_id=c.PARENT_RUN,source=c.PARENT_SOURCE,review=c.PARENT_REVIEW_DIGEST,driver_pid=990000,uid=os.getuid())))
        (self.budget.root/'observer.jsonl').write_bytes(json_bytes(dict(kind='observation',monotonic=now-9,wall_seconds=1.)))

    def complete(self,count=2):
        for expected in self.expected[:count]:
            key=expected['wrapper_key'];self.ledger.register(expected);self.ledger.reserve(key)
            self.records.append(self.ledger.complete(key,METRICS))
    def store(self):
        files=c.inventory(self.parent);state=c.replay(self.parent,files)
        return c.ResumeStore(self.parent,files,state,'a'*64)
    def close(self):self.ledger.close();self.budget.close()


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.f=Fixture(self.root)
        for name,path in zip(('STARTED_EXPECTED','CLOCK_EXPECTED'),self.f.paths):
            value={k:v for k,v in c.path_ref(path).items() if k!='path'}
            patcher=patch.object(c,name,value);patcher.start();self.addCleanup(patcher.stop)
    def tearDown(self):self.f.close();self.temp.cleanup()
    def test_completed_pairs_skip_build_fingerprint_and_compile(self):
        self.f.complete();store=self.f.store();run=Run()
        with patch.object(circuits,'build_evolution',side_effect=AssertionError('completed build')), \
             patch.object(circuits,'numerical_fingerprint',side_effect=AssertionError('completed encoding')):
            jobs=list(execution.candidate_wrapper_jobs(run,self.f.identity,[None],None,self.f.template,resume=store))
        self.assertTrue(all(qc is None for _,qc in jobs))
        budget=OutputBudget(self.root/'new');ledger=Ledger(budget,cap=74805,prior_invocations=2,resume=store,execution_source='b'*40)
        try:
            records=parallel.compile_wrappers(run,ledger,jobs,None,{}, {},expected_count=2)
            self.assertEqual(run.calls,[]);self.assertEqual(records,self.f.records)
            self.assertEqual(ledger.audit()['actual_invocations'],2)
            self.assertEqual(ledger.audit()['imported_wrappers'],2)
            for expected in self.f.expected:
                name='record-'+expected['wrapper_key']+'.json'
                self.assertEqual((budget.root/name).read_bytes(),(self.f.budget.root/name).read_bytes())
        finally:ledger.close();budget.close()

    def test_partial_pair_builds_and_compiles_only_missing_axis(self):
        self.f.complete(1);store=self.f.store();run=Run()
        with patch.object(circuits,'build_evolution',return_value=object()) as build, \
             patch.object(circuits,'wrapper',return_value=object()) as wrapper, \
             patch.object(circuits,'numerical_fingerprint',return_value='2'*64) as encode:
            jobs=list(execution.candidate_wrapper_jobs(run,self.f.identity,[None],{'components':[]},self.f.template,resume=store))
        self.assertEqual((build.call_count,wrapper.call_count,encode.call_count),(1,1,1))
        budget=OutputBudget(self.root/'new');ledger=Ledger(budget,cap=74805,prior_invocations=1,resume=store,execution_source='b'*40)
        try:
            records=parallel.compile_wrappers(run,ledger,jobs,None,{}, {},expected_count=2)
            self.assertEqual(len(run.calls),1)
            self.assertEqual(records[1]['execution_source_commit'],'b'*40)
            self.assertEqual(records[1]['source_commit'],c.PARENT_SOURCE)
            self.assertEqual(records[1]['actual_transpile_invocation_id'],'science-000002')
            self.assertEqual(ledger.audit()['actual_invocations'],2)
        finally:ledger.close();budget.close()

    def test_reserved_orphan_is_not_imported_and_cost_is_kept(self):
        expected=self.f.expected[0];key=expected['wrapper_key']
        self.f.ledger.register(expected);self.f.ledger.reserve(key)
        with self.assertRaises(Stop):self.f.ledger.complete(key,METRICS,interrupt_after_record=True)
        store=self.f.store();self.assertIsNone(store.expected(self.f.identity,'cosine',None,None))
        budget=OutputBudget(self.root/'new');ledger=Ledger(budget,cap=74805,prior_invocations=1,resume=store,execution_source='b'*40)
        try:
            run=Run();parallel.compile_wrappers(run,ledger,[(expected,object())],None,{}, {},expected_count=1)
            self.assertEqual(ledger.audit()['actual_invocations'],2)
        finally:ledger.close();budget.close()

    def test_cache_owner_can_be_imported_before_logical_owner_position(self):
        self.f.complete(1)
        identity=self.f.identity
        # Distinct trajectory identities in a registered random candidate.
        template=next(t for t in BASE['templates'] if t['method']=='B2')
        identity=c.identity_for(self.f.parent,'0.70',template);seeds=signal.trajectory_seeds(identity)
        owner=wire(identity,'cosine',seeds[0],0,'3'*64);follower=wire(identity,'cosine',seeds[1],1,'3'*64)
        for expected in (owner,follower):self.f.ledger.register(expected)
        self.f.ledger.reserve(owner['wrapper_key']);self.f.ledger.complete(owner['wrapper_key'],METRICS)
        self.f.ledger.complete(follower['wrapper_key'],{},owner_key=owner['wrapper_key'])
        store=self.f.store();budget=OutputBudget(self.root/'new')
        ledger=Ledger(budget,cap=74805,prior_invocations=2,resume=store,execution_source='b'*40)
        try:
            jobs=[(follower,None),(owner,None),(self.f.expected[0],None)]
            run=Run();records=parallel.compile_wrappers(run,ledger,jobs,None,{}, {},expected_count=3)
            self.assertEqual(run.calls,[]);self.assertEqual(records[0]['cache_owner_wrapper_key'],owner['wrapper_key'])
            self.assertEqual(ledger.audit()['imported_wrappers'],3)
        finally:ledger.close();budget.close()

    def test_record_mutation_missing_and_wrong_identity_fail_closed(self):
        self.f.complete(1);store=self.f.store();expected=self.f.expected[0]
        changed={**expected,'environment_fingerprint':'f'*64}
        with self.assertRaises(Stop):store.recover(changed)
        p=self.f.budget.root/('record-'+expected['wrapper_key']+'.json')
        raw=p.read_bytes();p.write_bytes(raw+b' ')
        with self.assertRaises(Stop):store.recover(expected)
        p.unlink()
        with self.assertRaises(FileNotFoundError):c.replay(self.f.parent,store.files)

    def test_ledger_gap_chain_and_refund_fail_closed(self):
        self.f.complete(1);files=c.inventory(self.f.parent)
        p=self.f.budget.root/'ledger-000001.json';raw=p.read_bytes();row=json.loads(raw);row['previous_digest']='f'*64
        p.write_bytes(json_bytes(row));files=c.inventory(self.f.parent)
        with self.assertRaises(Stop):c.replay(self.f.parent,files)
        p.write_bytes(raw);p.rename(self.f.budget.root/'ledger-000005.json')
        with self.assertRaises(Stop):c.replay(self.f.parent,c.inventory(self.f.parent))

    def test_pending_hardlink_is_evidence_only(self):
        expected=self.f.expected[0];key=expected['wrapper_key']
        self.f.ledger.register(expected);self.f.ledger.reserve(key)
        with self.assertRaises(Stop):self.f.ledger.complete(key,METRICS,interrupt_after_record=True)
        original=self.f.budget.root/('record-'+key+'.json')
        os.link(original,original.with_name(original.name+'.pending'))
        files=c.inventory(self.f.parent)
        self.assertIn('output/'+original.name+'.pending',files)
        self.assertEqual(c.replay(self.f.parent,files)['records'],{})
        os.link(original,self.root/'foreign-link')
        with self.assertRaises(Stop):
            c.file_receipt(original)

    def test_seed_identity_remains_run10_source(self):
        template=next(t for t in BASE['templates'] if t['method']=='B2')
        identity=c.identity_for(self.f.parent,'0.70',template)
        self.assertEqual(identity['source'],c.PARENT_SOURCE)
        self.assertEqual(signal.trajectory_seeds(identity)[0],trajectory_seed(identity,0,actual_inputs_frozen=True,signal_launch=True))
        different={**identity,'source':'b'*40}
        self.assertNotEqual(signal.trajectory_seeds(identity),signal.trajectory_seeds(different))

    def test_native_zero_boot_uid_and_unknown_exits(self):
        with patch.object(c,'audit_absence'):
            self.assertEqual(c.verify_stop(self.f.proof,self.f.started,self.f.clock),7.)
            bad=copy.deepcopy(self.f.proof);bad['native_audits'][1]['same_owned_remaining']=1
            with self.assertRaises(Stop):c.verify_stop(bad,self.f.started,self.f.clock)
            bad=copy.deepcopy(self.f.proof);bad['historical_exit_code']=0
            with self.assertRaises(Stop):c.verify_stop(bad,self.f.started,self.f.clock)
            bad=copy.deepcopy(self.f.clock);bad['boot_id']='different'
            with self.assertRaises(Stop):c.verify_stop(self.f.proof,self.f.started,bad)
        from trottertracks.resource_applicability.h4_geometry import observer
        with patch.object(observer,'process_sample',return_value=self.f.clock['identities'][0]):
            with self.assertRaises(Stop):c.audit_absence(self.f.clock['identities'])

    def test_receipt_roundtrip_carry_and_stopped_requirement(self):
        self.f.complete(1)
        with patch.object(c,'parent_plan',return_value=self.f.parent),patch.object(c,'audit_absence'):
            receipt=c.make_receipt(ROOT,*self.f.paths)
            p=self.root/'receipt.json';p.write_bytes(json_bytes(receipt));descriptor=c.path_ref(p)
            store,data=c.load_receipt(ROOT,descriptor)
            self.assertEqual(data['carry']['actual_invocations'],1)
            self.assertEqual(data['carry']['charged_bytes'],store.state['charged_bytes'])
            self.assertEqual(data['carry']['wall_seconds'],7.)
            self.assertEqual(data['imported_wrappers'],1)
            receipt['carry']['actual_invocations']=0;p.write_bytes(json_bytes(receipt))
            with self.assertRaises(Stop):c.load_receipt(ROOT,c.path_ref(p))
        with patch.object(c,'parent_plan',return_value=self.f.parent),patch.object(c,'audit_absence',side_effect=Stop('still live')):
            with self.assertRaises(Stop):c.make_receipt(ROOT,*self.f.paths)

    def test_saved_signal_matches_records_and_metrics(self):
        self.f.complete();store=self.f.store();metrics=[[METRICS,METRICS]]
        result=dict(geometry='0.70',template_id=self.f.template['template_id'],identity=self.f.identity,
            candidate_fingerprint=fingerprint('h4-candidate-v1',self.f.identity),paired_metrics=metrics,
            saved={'paired_costs':[[3,3]]},research_decision=None,next_stage_authorized=False)
        name='signal-0.70-'+self.f.template['template_id']+'.json'
        (self.f.budget.root/name).write_bytes(json_bytes(result));store=self.f.store()
        self.assertEqual(store.complete_signal(self.f.identity,self.f.template,[None]),result)
        result['saved']['paired_costs']=[[4,3]];(self.f.budget.root/name).write_bytes(json_bytes(result));store=self.f.store()
        with self.assertRaises(Stop):store.complete_signal(self.f.identity,self.f.template,[None])

    def test_budget_caps_and_draft_authorization_remain_closed(self):
        plan=copy.deepcopy(BASE)
        auth=json.loads((ROOT/(c.PARENT_BUNDLE+'authorization_v20.json')).read_text())
        review=json.loads((ROOT/(c.PARENT_BUNDLE+'review_v20.json')).read_text())
        plan['continuation']=dict(identity_source_commit=c.PARENT_SOURCE,receipt=dict(path=str(self.root/'not-received.json'),bytes=1,sha256='a'*64),imported_wrappers=1)
        auth['continuation_amendment']=dict(approved=False,scope='run10_encoding_only_resume',continuation=copy.deepcopy(plan['continuation']),authority=dict(path='draft.json',sha256='a'*64))
        plan['carry']=dict(actual_invocations=1,charged_bytes=128,wall_seconds=7.)
        plan['run_id']=auth['run_id']=review['run_id']=cb.RESUME_RUN_ID
        plan['source_commit']=auth['source_commit']=review['source_commit']='b'*40
        plan['sealed']=auth['approved']=auth['runtime_authorization']=review['approved']=review['runtime_authorization']=False
        auth['allowed_cpus']=[]
        lb.structural(plan,auth,review)
        with self.assertRaises(Stop):lb.authorize(plan,auth,review,explicit_launch=True)
        with self.assertRaises(Stop):cb.authorize(plan,auth)
        auth['continuation_amendment']['approved']=True;cb.authorize(plan,auth)
        plan['carry']['actual_invocations']=23
        with self.assertRaises(Stop):cb.authorize(plan,auth)
        plan['carry']['actual_invocations']=1;plan['carry']['wall_seconds']=72*3600.
        with self.assertRaises(Stop):cb.structural(plan,auth)

    def test_candidate_publication_order_with_imports_and_new_work(self):
        self.f.complete(1);store=self.f.store();budget=OutputBudget(self.root/'new')
        ledger=Ledger(budget,cap=74805,prior_invocations=1,resume=store,execution_source='b'*40)
        published=[];run=Run()
        try:
            candidates=[('first',[(self.f.expected[0],None)],1),('second',[(self.f.expected[1],object())],1)]
            parallel.compile_candidates(run,ledger,candidates,None,{}, {},expected_candidates=2,expected_wrappers=2,
                on_candidate=lambda metadata,records:published.append(metadata))
            self.assertEqual(published,['first','second']);self.assertEqual(len(run.calls),1)
        finally:ledger.close();budget.close()

    def test_forged_family_marker_and_wall_origin_rejected_on_load(self):
        self.f.complete(1)
        with patch.object(c,'parent_plan',return_value=self.f.parent),patch.object(c,'audit_absence'):
            receipt=c.make_receipt(ROOT,*self.f.paths)
            marker=Path(self.f.parent['control_root'])/'one-shot.json'
            data=c.read_json(marker);data['driver_pid']=990999;marker.write_bytes(json_bytes(data))
            receipt['files']=c.inventory(self.f.parent)
            p=self.root/'receipt.json';p.write_bytes(json_bytes(receipt))
            with self.assertRaisesRegex(Stop,'marker identity on load'):c.load_receipt(ROOT,c.path_ref(p))
            clock=copy.deepcopy(self.f.clock);clock['wall_origin_monotonic']+=5
            proof=copy.deepcopy(self.f.proof);proof['clock']=clock
            with self.assertRaisesRegex(Stop,'wall origin changed'):c.verify_stop(proof,self.f.started,clock)

    def test_observer_wall_floor_prevents_backdated_native_proof(self):
        self.f.complete(1)
        (self.f.budget.root/'observer.jsonl').write_bytes(json_bytes(dict(kind='observation',
            monotonic=self.f.proof['native_audits'][1]['monotonic']+1,wall_seconds=8.)))
        with patch.object(c,'parent_plan',return_value=self.f.parent),patch.object(c,'audit_absence'):
            with self.assertRaisesRegex(Stop,'precede retained observer'):c.make_receipt(ROOT,*self.f.paths)

    def test_full_launch_permission_gate_accepts_bound_metadata_only(self):
        plan=copy.deepcopy(BASE)
        auth=json.loads((ROOT/(c.PARENT_BUNDLE+'authorization_v20.json')).read_text())
        review=json.loads((ROOT/(c.PARENT_BUNDLE+'review_v20.json')).read_text())
        plan['continuation']=dict(identity_source_commit=c.PARENT_SOURCE,receipt=dict(path=str(self.root/'future.json'),bytes=1,sha256='a'*64),imported_wrappers=1)
        auth['continuation_amendment']=dict(approved=True,scope='run10_encoding_only_resume',continuation=copy.deepcopy(plan['continuation']),authority=dict(path='future-authority.json',sha256='a'*64))
        plan['run_id']=auth['run_id']=review['run_id']=cb.RESUME_RUN_ID
        plan['source_commit']=auth['source_commit']=review['source_commit']='b'*40
        plan['source_root']=str(ROOT)
        plan['output_root']=str(self.root/'output'/cb.RESUME_RUN_ID);plan['control_root']=str(self.root/'control'/cb.RESUME_RUN_ID)
        plan['carry']=dict(actual_invocations=1,charged_bytes=128,wall_seconds=7.)
        auth['budget_accounting']['scope']='run10_continuation'
        auth['recovery_policy']['mode']='investigate_fix_continuation'
        auth['plan_fingerprint']=review['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',plan)
        review['authorization_digest']=fingerprint('h4-newhost-authorization-v2',auth)
        with patch.object(os,'sched_setaffinity',side_effect=AssertionError('CPU action')):
            permit=lb.authorize(plan,auth,review,explicit_launch=True)
            self.assertEqual(permit.plan['carry'],plan['carry'])
        # This gate is pure metadata; absent receipt still cannot pass runtime.

    def test_first_failure_preserved_and_resumed_reservation_not_refunded(self):
        self.f.complete(1);store=self.f.store();budget=OutputBudget(self.root/'new')
        ledger=Ledger(budget,cap=74805,prior_invocations=1,resume=store,execution_source='b'*40)
        class FailedRun(Run):
            def submit(self,*args):
                future=Future();future.set_exception(RuntimeError('original worker failure'));return future
        try:
            with self.assertRaisesRegex(RuntimeError,'original worker failure'):
                parallel.compile_wrappers(FailedRun(),ledger,[(self.f.expected[0],None),(self.f.expected[1],object())],None,{}, {},expected_count=2)
            self.assertEqual(len(ledger.reservations),1)
            self.assertEqual(next(iter(ledger.reservations.values()))['status'],'RESERVED')
        finally:ledger.close();budget.close()

    def test_fresh_output_control_are_disjoint_from_all_predecessor_roots(self):
        plan=copy.deepcopy(self.f.parent)
        plan['output_root']=str(self.root/'new-output');plan['control_root']=str(self.root/'new-control')
        plan['source_root']=str(ROOT);plan['input_root']=str(self.root/'inputs');plan['stop_evidence_root']=str(self.root/'stop')
        plan['continuation']={'receipt':{'path':str(self.root/'receipt.json')}}
        cb.verify_roots(plan,self.f.parent)
        for new_field in ('output_root','control_root'):
            for old_field in ('output_root','control_root'):
                bad=copy.deepcopy(plan);bad[new_field]=str(Path(self.f.parent[old_field])/'nested')
                with self.assertRaises(Stop):cb.verify_roots(bad,self.f.parent)
        bad=copy.deepcopy(plan);bad['control_root']=str(Path(plan['output_root'])/'nested')
        with self.assertRaises(Stop):cb.verify_roots(bad,self.f.parent)

    def test_existing_worker_first_stop_text_is_receipted_but_other_text_is_rejected(self):
        path=self.f.budget.root/'worker-log-first-stop.txt'
        path.write_bytes(b'bounded synthetic worker stop evidence\n')
        files=c.inventory(self.f.parent)
        self.assertEqual(files['output/worker-log-first-stop.txt'],c.file_receipt(path))
        (self.f.budget.root/'unknown.txt').write_bytes(b'not a sealed runtime file')
        with self.assertRaises(Stop):c.inventory(self.f.parent)
