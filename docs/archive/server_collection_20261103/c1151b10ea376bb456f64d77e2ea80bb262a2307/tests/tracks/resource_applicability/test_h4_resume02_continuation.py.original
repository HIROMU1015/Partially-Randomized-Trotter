"""Two-generation byte/ledger/native fixtures; no real run, numpy or compiler."""
import copy
from contextlib import redirect_stdout
from concurrent.futures import Future
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trottertracks.resource_applicability.h4_geometry import continuation as c
from trottertracks.resource_applicability.h4_geometry import continuation_resume01 as nextgen
from trottertracks.resource_applicability.h4_geometry import continuation_binding as cb
from trottertracks.resource_applicability.h4_geometry import execution, circuits, parallel, launch_binding as lb
from trottertracks.resource_applicability.h4_geometry.identity import Stop, fingerprint, sha
from trottertracks.resource_applicability.h4_geometry.ledger import Ledger, json_bytes, wire
from trottertracks.resource_applicability.h4_geometry.resources import OutputBudget

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('legacy_fixture',Path(__file__).with_name('test_h4_continuation.py'))
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
BASE=json.loads((ROOT/(nextgen.PARENT_BUNDLE+'plan_authorized_v1.json')).read_text())
AUTH=json.loads((ROOT/(nextgen.PARENT_BUNDLE+'authorization_v1.json')).read_text())
REVIEW=json.loads((ROOT/(nextgen.PARENT_BUNDLE+'review_v1.json')).read_text())


class Family:
    def __init__(self,root,*,finish=True):
        self.root=root;(root/'run10').mkdir();self.first=legacy.Fixture(root/'run10')
        self.first.complete(1)
        key=self.first.expected[1]['wrapper_key']
        self.first.ledger.register(self.first.expected[1]);self.first.ledger.reserve(key)  # old failed call stays charged
        self.ancestor=self.first.store()
        self.ancestor_data=dict(carry=dict(actual_invocations=2,charged_bytes=self.ancestor.state['charged_bytes'],wall_seconds=8.),
                                imported_wrappers=1,unresolved_reservations=[key])
        self.ancestor_data.update({k:c.path_ref(p) for k,p in zip(('started','clock_context','native_stop_proof'),self.first.paths)})
        path=root/'ancestor.json';path.write_bytes(json_bytes(self.ancestor_data));self.ancestor_ref=c.path_ref(path)
        self.ancestor.digest=self.ancestor_ref['sha256']
        self.plan=copy.deepcopy(BASE)
        self.plan.update(output_root=str(root/'resume01-output'),control_root=str(root/'resume01-control'),
                         carry=copy.deepcopy(self.ancestor_data['carry']))
        self.plan['continuation']=dict(identity_source_commit=c.PARENT_SOURCE,receipt=self.ancestor_ref,imported_wrappers=1)
        self.budget=OutputBudget(Path(self.plan['output_root']),prior_charge=self.plan['carry']['charged_bytes'])
        self.ledger=Ledger(self.budget,cap=74805,prior_invocations=2,resume=self.ancestor,execution_source=nextgen.PARENT_SOURCE)
        self.ledger.register(self.first.expected[0]);self.ledger.import_complete(self.first.expected[0])
        self.ledger.register(self.first.expected[1]);self.ledger.reserve(key)
        if finish:self.ledger.complete(key,legacy.METRICS)
        control=Path(self.plan['control_root']);control.mkdir()
        now=time.monotonic()
        self.started=copy.deepcopy(self.first.started)
        self.started.update(source=nextgen.PARENT_SOURCE,artifact_commit=nextgen.PARENT_REVIEW,run_id=nextgen.PARENT_RUN,
                            observer_observation=dict(monotonic=now-10,wall_seconds=12.))
        self.clock=dict(schema_version='h4-resume01-clock-v1',boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            identities=c.native_identities(self.first.started),wall_origin_monotonic=now-22,captured_monotonic=now-9,
            historical_exit_code=None,exact_exit_time=None)
        self.proof=dict(schema_version='h4-resume01-native-stop-v1',run_id=nextgen.PARENT_RUN,source_commit=nextgen.PARENT_SOURCE,
            artifact_commit=nextgen.PARENT_REVIEW,clock=self.clock,native_audits=[dict(monotonic=now-5,identities=self.clock['identities'],same_owned_remaining=0),
            dict(monotonic=now-3,identities=self.clock['identities'],same_owned_remaining=0)],historical_exit_code=None,exact_exit_time=None,exit_data_not_inferred=True)
        self.paths=[]
        for name,value in zip(('started','clock','proof'),(self.started,self.clock,self.proof)):
            p=root/(name+'.json');p.write_bytes(json_bytes(value));self.paths.append(p)
        (control/'one-shot.json').write_bytes(json_bytes(dict(run_id=nextgen.PARENT_RUN,source=nextgen.PARENT_SOURCE,
            review=nextgen.PARENT_REVIEW_DIGEST,driver_pid=self.started['driver_pid'],uid=os.getuid())))
        self.budget.write('observer.jsonl',json_bytes(dict(kind='observation',monotonic=now-10,wall_seconds=12.)))

    def state(self):return nextgen.replay(self.plan,c.inventory(self.plan),self.ancestor)
    def store(self):return c.ResumeStore(self.plan,c.inventory(self.plan),self.state(),'b'*64)
    def rechain(self,edit):
        chain=None
        for p in sorted(self.budget.root.glob('ledger-*.json')):
            row=json.loads(p.read_bytes());edit(row);row['previous_digest']=chain
            p.write_bytes(json_bytes(row));chain=fingerprint('h4-ledger-delta-v1',row)
    def close(self):self.ledger.close();self.budget.close();self.first.close()


def bound_documents(family):
    plan=copy.deepcopy(family.plan);auth=copy.deepcopy(AUTH);review=copy.deepcopy(REVIEW)
    plan.update(run_id=nextgen.NEXT_RUN,source_commit='c'*40,source_root=str(ROOT),
                output_root=str(family.root/'next-output'/nextgen.NEXT_RUN),control_root=str(family.root/'next-control'/nextgen.NEXT_RUN))
    plan['continuation']=dict(identity_source_commit=c.PARENT_SOURCE,
        receipt=dict(path=str(family.root/'future-receipt.json'),bytes=1,sha256='b'*64),imported_wrappers=2)
    plan['carry']=dict(actual_invocations=3,charged_bytes=family.state()['charged_bytes'],wall_seconds=19.)
    auth['continuation_amendment']=dict(approved=True,scope=nextgen.SCOPE,continuation=copy.deepcopy(plan['continuation']),
        authority=dict(path='artificial-resume02-authority.json',sha256='a'*64))
    for doc in (auth,review):doc.update(run_id=nextgen.NEXT_RUN,source_commit=plan['source_commit'])
    plan['storage']['prior_charge_bytes']=plan['carry']['charged_bytes']
    plan['storage']['cumulative_charge_bound']=legacy.BASE['storage']['cumulative_charge_bound']+plan['carry']['charged_bytes']
    plan['storage']['required_bytes']=legacy.BASE['storage']['required_bytes']
    plan['storage']['required_inodes']=legacy.BASE['storage']['required_inodes']
    auth['plan_fingerprint']=review['plan_fingerprint']=fingerprint('h4-newhost-plan-v2',plan)
    review['authorization_digest']=fingerprint('h4-newhost-authorization-v2',auth)
    return plan,auth,review


class Resume02Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.f=Family(self.root)
        for name,path in zip(('STARTED_EXPECTED','CLOCK_EXPECTED'),self.f.paths):
            p=patch.object(nextgen,name,{k:v for k,v in c.path_ref(path).items() if k!='path'});p.start();self.addCleanup(p.stop)
        for name,value in (('parent_plan',self.f.plan),('ancestor_store',(self.f.ancestor,self.f.ancestor_data))):
            p=patch.object(nextgen,name,return_value=value);p.start();self.addCleanup(p.stop)
        p=patch.object(c,'audit_absence');p.start();self.addCleanup(p.stop)
    def tearDown(self):self.f.close();self.temp.cleanup()

    def receipt(self):
        data=nextgen.make_receipt(ROOT,*self.f.paths);p=self.root/'receipt.json';p.write_bytes(json_bytes(data))
        return p,data

    def test_two_generation_mixed_records_and_cumulative_actuals(self):
        state=self.f.state()
        self.assertEqual((state['total_actual_invocations'],len(state['reservations']),len(state['records'])),(3,1,2))
        records=state['records']
        self.assertNotIn('execution_source_commit',records[self.f.first.expected[0]['wrapper_key']])
        self.assertEqual(records[self.f.first.expected[1]['wrapper_key']]['execution_source_commit'],nextgen.PARENT_SOURCE)
        self.assertTrue(all(r['source_commit']==c.PARENT_SOURCE for r in records.values()))

    def test_next_ledger_imports_both_generations_without_recompile(self):
        store=self.f.store();budget=OutputBudget(self.root/'next');ledger=Ledger(budget,cap=74805,prior_invocations=3,resume=store,execution_source='c'*40)
        run=legacy.Run()
        try:
            with patch.object(circuits,'build_evolution',side_effect=AssertionError('completed build')),patch.object(circuits,'numerical_fingerprint',side_effect=AssertionError('completed encoding')):
                jobs=list(execution.candidate_wrapper_jobs(run,self.f.first.identity,[None],None,self.f.first.template,resume=store))
            parallel.compile_wrappers(run,ledger,jobs,None,{}, {},expected_count=2)
            self.assertEqual(run.calls,[]);self.assertEqual(ledger.audit()['actual_invocations'],3)
            for e in self.f.first.expected:
                name='record-'+e['wrapper_key']+'.json';self.assertEqual((budget.root/name).read_bytes(),(self.f.budget.root/name).read_bytes())
        finally:ledger.close();budget.close()

    def test_reserved_previous_calls_stay_charged_and_only_missing_axis_recomputes(self):
        extra=self.root/'unfinished';extra.mkdir();f=Family(extra,finish=False)
        try:
            store=f.store();self.assertEqual(store.state['total_actual_invocations'],3)
            run=legacy.Run();budget=OutputBudget(extra/'next');ledger=Ledger(budget,cap=74805,prior_invocations=3,resume=store,execution_source='c'*40)
            try:
                with patch.object(circuits,'build_evolution',return_value=object()),patch.object(circuits,'wrapper',return_value=object()),patch.object(circuits,'numerical_fingerprint',return_value='2'*64):
                    jobs=list(execution.candidate_wrapper_jobs(run,f.first.identity,[None],{'components':[]},f.first.template,resume=store))
                result=parallel.compile_wrappers(run,ledger,jobs,None,{}, {},expected_count=2)
                self.assertEqual(len(run.calls),1);self.assertEqual(result[1]['actual_transpile_invocation_id'],'science-000004')
                self.assertEqual(ledger.audit()['actual_invocations'],4)
            finally:ledger.close();budget.close()
        finally:f.close()

    def test_byte_journal_already_includes_ancestor_no_double_charge(self):
        state=self.f.state();journal=(self.f.budget.root/'byte-budget.journal').read_bytes()
        charged=sum(int(journal[i:i+128].strip()) for i in range(0,len(journal),128))
        self.assertEqual(state['charged_bytes'],charged)
        self.assertEqual(int(journal[:128].strip()),self.f.plan['carry']['charged_bytes']+128)
        self.assertNotEqual(state['charged_bytes'],charged+self.f.plan['carry']['charged_bytes'])
        p=self.f.budget.root/'byte-budget.journal';p.write_bytes(b'128\n'.ljust(128,b' ')+journal[128:])
        with self.assertRaisesRegex(Stop,'first journal carry row'):self.f.state()

    def test_receipt_roundtrip_preserves_wall_ancestor_reserved_and_record_sources(self):
        p,data=self.receipt();store,loaded=nextgen.load_receipt(ROOT,c.path_ref(p))
        self.assertEqual(data,loaded);self.assertEqual(data['carry']['actual_invocations'],3)
        self.assertEqual(data['carry']['wall_seconds'],19.)
        self.assertEqual(data['unresolved_ancestor_reservations'],self.f.ancestor_data['unresolved_reservations'])
        self.assertEqual(data['ancestor_receipt'],self.f.ancestor_ref)
        self.assertEqual(store.state['total_actual_invocations'],3)

    def test_forged_carry_receipt_or_unresolved_history_is_rejected(self):
        p,data=self.receipt()
        for field,value in [('actual_invocations',1),('charged_bytes',0),('wall_seconds',0.)]:
            bad=copy.deepcopy(data);bad['carry'][field]=value;p.write_bytes(json_bytes(bad))
            with self.assertRaisesRegex(Stop,'receipt/cumulative carry mismatch'):nextgen.load_receipt(ROOT,c.path_ref(p))
        bad=copy.deepcopy(data);bad['unresolved_ancestor_reservations']=[];p.write_bytes(json_bytes(bad))
        with self.assertRaises(Stop):nextgen.load_receipt(ROOT,c.path_ref(p))

    def test_imported_old_bytes_and_lineage_tag_cannot_be_changed(self):
        key=self.f.first.expected[0]['wrapper_key']
        self.f.rechain(lambda row:row['entries'][key].update(continuation_receipt_sha256='f'*64) if key in row['entries'] else None)
        with self.assertRaisesRegex(Stop,'imported lineage/bytes'):self.f.state()

    def test_acknowledged_ancestor_records_cannot_disappear(self):
        key=self.f.first.expected[0]['wrapper_key']
        self.f.rechain(lambda row:row['entries'].pop(key,None))
        (self.f.budget.root/('record-'+key+'.json')).unlink()
        with self.assertRaisesRegex(Stop,'ancestor COMPLETE missing'):self.f.state()

    def test_new_record_requires_fixed_execution_source_and_full_fields(self):
        key=self.f.first.expected[1]['wrapper_key'];p=self.f.budget.root/('record-'+key+'.json')
        original=json.loads(p.read_bytes())
        for change in ('wrong','missing'):
            data=copy.deepcopy(original)
            if change=='wrong':data['execution_source_commit']='f'*40
            else:data.pop('execution_source_commit')
            p.write_bytes(json_bytes(data))
            with self.assertRaisesRegex(Stop,'execution SOURCE'):self.f.state()

    def test_actual_invocation_offset_cannot_restart_or_have_gaps(self):
        key=self.f.first.expected[1]['wrapper_key']
        self.f.rechain(lambda row:row['reservations'][key].update(invocation='science-000001') if key in row['reservations'] else None)
        with self.assertRaisesRegex(Stop,'contiguous actual invocations'):self.f.state()

    def test_orphan_record_and_pending_bytes_are_retained_but_not_imported(self):
        name='record-'+'f'*64+'.json';self.f.budget.write(name,b'{}\n')
        (self.f.budget.root/(name+'.pending')).write_bytes(b'{}\n')
        _,data=self.receipt()
        self.assertIn('f'*64,data['orphan_records']);self.assertIn('output/'+name+'.pending',data['pending_files'])
        self.assertEqual(data['imported_wrappers'],2)

    def test_saved_signal_requires_both_generations_metrics_and_identity(self):
        ident=self.f.first.identity;template=self.f.first.template
        result=dict(geometry='0.70',template_id=template['template_id'],identity=ident,candidate_fingerprint=fingerprint('h4-candidate-v1',ident),
                    paired_metrics=[[legacy.METRICS,legacy.METRICS]],saved=dict(paired_costs=[[3,3]]),research_decision=None,next_stage_authorized=False)
        name='signal-0.70-'+template['template_id']+'.json';self.f.budget.write(name,json_bytes(result))
        p,_=self.receipt();store,_=nextgen.load_receipt(ROOT,c.path_ref(p))
        self.assertEqual(store.complete_signal(ident,template,[None]),result)
        result['saved']['paired_costs']=[[4,3]];(self.f.budget.root/name).write_bytes(json_bytes(result))
        with self.assertRaises(Stop):self.receipt()

    def test_live_parent_rejected_before_inventory_and_snapshot(self):
        with patch.object(c,'audit_absence',side_effect=Stop('still live')),patch.object(c,'inventory',side_effect=AssertionError('live snapshot')):
            with self.assertRaisesRegex(Stop,'still live'):self.receipt()

    def test_native_zero_clock_origin_boot_parent_graph_and_unknown_exits(self):
        self.assertEqual(nextgen.verify_stop(self.f.proof,self.f.started,self.f.clock),19.)
        for field,value in [('historical_exit_code',0),('exact_exit_time',time.time()),('exit_data_not_inferred',False)]:
            proof=copy.deepcopy(self.f.proof);proof[field]=value
            with self.assertRaises(Stop):nextgen.verify_stop(proof,self.f.started,self.f.clock)
        clock=copy.deepcopy(self.f.clock);clock['wall_origin_monotonic']+=1;proof=copy.deepcopy(self.f.proof);proof['clock']=clock
        with self.assertRaisesRegex(Stop,'wall origin'):nextgen.verify_stop(proof,self.f.started,clock)
        started=copy.deepcopy(self.f.started);started['roles'][2]['parent']=1
        with self.assertRaises(Stop):nextgen.native_identities(started)
        proof=copy.deepcopy(self.f.proof);proof['native_audits'][1]['same_owned_remaining']=1
        with self.assertRaises(Stop):nextgen.verify_stop(proof,self.f.started,self.f.clock)

    def test_marker_and_observer_floor_are_checked_on_receipt_load(self):
        p,data=self.receipt();marker=Path(self.f.plan['control_root'])/'one-shot.json'
        row=json.loads(marker.read_bytes());row['driver_pid']+=1;marker.write_bytes(json_bytes(row))
        data['files']=c.inventory(self.f.plan);p.write_bytes(json_bytes(data))
        with self.assertRaisesRegex(Stop,'marker identity'):nextgen.load_receipt(ROOT,c.path_ref(p))

    def test_closed_generation_schema_and_pure_draft_launch_gate(self):
        plan,auth,review=bound_documents(self.f)
        with patch.object(os,'sched_setaffinity',side_effect=AssertionError('CPU action')):
            permit=lb.authorize(plan,auth,review,explicit_launch=True)
            self.assertEqual(permit.plan['run_id'],nextgen.NEXT_RUN)
        auth['continuation_amendment']['approved']=False
        with self.assertRaises(Stop):cb.authorize(plan,auth)
        auth['continuation_amendment']['scope']='resume03_arbitrary'
        with self.assertRaises(Stop):cb.structural(plan,auth)
        schema=json.loads((ROOT/'schemas/h4_newhost_launch_v2.json').read_text())
        self.assertEqual(schema['properties']['authorization']['properties']['continuation_amendment']['properties']['scope']['enum'],
                         ['run10_encoding_only_resume',nextgen.SCOPE])

    def test_fresh_roots_protect_both_ancestors_and_all_proof_paths(self):
        plan,auth,_=bound_documents(self.f)
        proofs=[str(self.f.paths[0]),str(self.f.paths[1]),str(self.root/'ancestor.json')]
        for parent in (self.f.plan,self.f.first.parent):
            cb.verify_roots(plan,parent,extra_readonly=proofs)
            for field in ('output_root','control_root'):
                for protected in [parent['output_root'],parent['control_root'],*proofs]:
                    bad=copy.deepcopy(plan);bad[field]=protected+'/nested'
                    with self.assertRaises(Stop):cb.verify_roots(bad,parent,extra_readonly=proofs)

    def test_first_failure_stays_original_and_cumulative_call_is_not_refunded(self):
        extra=self.root/'unfinished';extra.mkdir();f=Family(extra,finish=False)
        class Failed(legacy.Run):
            def submit(self,*args):
                result=Future();result.set_exception(RuntimeError('original compile failure'));return result
        try:
            store=f.store();budget=OutputBudget(extra/'next');ledger=Ledger(budget,cap=74805,prior_invocations=3,resume=store,execution_source='c'*40)
            try:
                with self.assertRaisesRegex(RuntimeError,'original compile failure'):
                    parallel.compile_wrappers(Failed(),ledger,[(f.first.expected[0],None),(f.first.expected[1],object())],None,{}, {},expected_count=2)
                self.assertEqual(next(iter(ledger.reservations.values())),dict(status='RESERVED',invocation='science-000004'))
            finally:ledger.close();budget.close()
        finally:f.close()

    def test_old_run10_replay_default_and_receipt_still_work(self):
        f=self.f.first
        with patch.object(c,'parent_plan',return_value=f.parent),patch.object(c,'STARTED_EXPECTED',{k:v for k,v in c.path_ref(f.paths[0]).items() if k!='path'}), \
             patch.object(c,'CLOCK_EXPECTED',{k:v for k,v in c.path_ref(f.paths[1]).items() if k!='path'}):
            data=c.make_receipt(ROOT,*f.paths);p=self.root/'first-receipt.json';p.write_bytes(json_bytes(data))
            store,_=c.load_receipt(ROOT,c.path_ref(p));self.assertEqual(store.state['total_actual_invocations'],2)
            self.assertEqual(data['carry']['actual_invocations'],2)

    def test_numbered_worker_text_log_is_receipted_unknown_file_rejected(self):
        self.f.budget.write('worker-log-000007.txt',b'bounded artificial worker log\n')
        self.assertIn('output/worker-log-000007.txt',c.inventory(self.f.plan))
        (self.f.budget.root/'unknown.txt').write_bytes(b'unknown')
        with self.assertRaises(Stop):c.inventory(self.f.plan)

    def runtime_context(self):
        plan,auth,review=bound_documents(self.f);store=self.f.store();_,data=self.receipt()
        prefix='src/trottertracks/resource_applicability/h4_geometry/'
        def blob(root,commit,path):return (commit+':'+path).encode()
        for path in self.f.plan['source_hashes']:
            if path.startswith(prefix):self.f.plan['source_hashes'][path]=sha(blob(ROOT,nextgen.PARENT_SOURCE,path))
        plan['source_hashes']=copy.deepcopy(self.f.plan['source_hashes'])
        for path in plan['source_hashes']:
            if path.startswith(prefix) and Path(path).name in ('circuits.py','signal.py'):
                plan['source_hashes'][path]=sha(blob(ROOT,nextgen.SPEEDUP_SOURCE,path))
        plan['source_hashes'][prefix+'instruction_encoding.py']=sha(blob(ROOT,nextgen.SPEEDUP_SOURCE,prefix+'instruction_encoding.py'))
        plan['source_hashes'][prefix+'continuation_resume01.py']='d'*64
        authority=dict(schema_version='h4-user-resume01-continuation-authority-v1',resume_launch_approved=True,preparation_only=False,
            parent_source=nextgen.PARENT_SOURCE,parent_review=nextgen.PARENT_REVIEW,dataset_source=c.PARENT_SOURCE,
            validated_speedup_source=nextgen.SPEEDUP_SOURCE,preserve_dataset_seeds=True,preserve_consumed_budget=True,instruction='artificial approval only')
        return SimpleNamespace(plan=plan,authorization=auth,review=review),store,data,blob,authority

    def test_runtime_binding_source_budget_and_both_ancestors_pass_mock_only(self):
        permit,store,data,blob,authority=self.runtime_context()
        from trottertracks.resource_applicability.h4_geometry import gates
        with patch.object(lb,'reference',return_value=authority),patch.object(gates,'git_blob',side_effect=blob), \
             patch.object(nextgen,'load_receipt',return_value=(store,data)),patch.object(c,'parent_plan',return_value=self.f.first.parent):
            self.assertIs(nextgen.verify_runtime(permit),store)
            self.assertEqual(permit.plan['storage']['cumulative_charge_bound'],
                             self.f.first.parent['storage']['cumulative_charge_bound']+permit.plan['carry']['charged_bytes'])
            permit.plan['storage']['cumulative_charge_bound']=permit.plan['carry']['charged_bytes']
            with self.assertRaisesRegex(Stop,'storage includes'):nextgen.verify_runtime(permit)

    def test_preparation_authority_or_changed_science_source_cannot_launch(self):
        permit,store,data,blob,authority=self.runtime_context()
        from trottertracks.resource_applicability.h4_geometry import gates
        with patch.object(lb,'reference',return_value={**authority,'preparation_only':True}),patch.object(nextgen,'load_receipt',side_effect=AssertionError('unapproved receipt work')):
            with self.assertRaisesRegex(Stop,'explicit launch authority'):nextgen.verify_runtime(permit)
        with patch.object(lb,'reference',return_value=authority),patch.object(gates,'git_blob',side_effect=blob),patch.object(nextgen,'load_receipt',return_value=(store,data)):
            permit.plan['source_hashes']['src/trottertracks/resource_applicability/h4_geometry/signal.py']='f'*64
            with self.assertRaisesRegex(Stop,'validated scientific/monitor'):nextgen.verify_runtime(permit)

    def test_new_generation_cache_follower_imports_owner_before_logical_owner(self):
        from trottertracks.resource_applicability.h4_geometry.signal import trajectory_seeds
        template=next(t for t in self.f.plan['templates'] if t['method']=='B2')
        identity=c.identity_for(self.f.plan,'0.70',template);seeds=trajectory_seeds(identity)
        follower=wire(identity,'cosine',seeds[0],0,'a'*64);owner=wire(identity,'cosine',seeds[1],1,'a'*64)
        self.f.ledger.register(owner);self.f.ledger.reserve(owner['wrapper_key']);self.f.ledger.complete(owner['wrapper_key'],legacy.METRICS)
        self.f.ledger.register(follower);self.f.ledger.complete(follower['wrapper_key'],{},owner_key=owner['wrapper_key'])
        store=self.f.store();self.assertEqual(store.state['total_actual_invocations'],4)
        budget=OutputBudget(self.root/'cached-next');ledger=Ledger(budget,cap=74805,prior_invocations=4,resume=store,execution_source='c'*40)
        try:
            jobs=[(follower,None),(owner,None),*[(e,None) for e in self.f.first.expected]];run=legacy.Run()
            records=parallel.compile_wrappers(run,ledger,jobs,None,{}, {},expected_count=4)
            self.assertEqual(run.calls,[]);self.assertEqual(records[0]['cache_owner_wrapper_key'],owner['wrapper_key'])
            self.assertEqual(ledger.audit()['actual_invocations'],4)
        finally:ledger.close();budget.close()

    def test_receipt_cli_dispatch_and_fixed_clock_copy_mock_only(self):
        spec=importlib.util.spec_from_file_location('receipt_cli',ROOT/'scripts/resource_applicability/run_h4_resume_receipt.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        output=self.root/'cli-receipt.json'
        args=['receipt-cli','receipt','--parent','resume01','--started',str(self.f.paths[0]),
              '--clock',str(self.f.paths[1]),'--proof',str(self.f.paths[2]),'--output',str(output)]
        with patch.object(sys,'argv',args),redirect_stdout(io.StringIO()):cli.main()
        store,data=nextgen.load_receipt(ROOT,c.path_ref(output));self.assertEqual(data['carry']['actual_invocations'],3)
        clock_output=self.root/'clock-copy.json';args=['receipt-cli','capture-clock','--parent','resume01','--started',str(self.f.paths[0]),'--output',str(clock_output)]
        from trottertracks.resource_applicability.h4_geometry import observer
        rows={row['pid']:row for row in self.f.clock['identities']}
        with patch.object(sys,'argv',args),patch.object(nextgen,'CLOCK_PATH',str(self.f.paths[1])), \
             patch.object(observer,'process_sample',side_effect=lambda pid:rows[pid]),redirect_stdout(io.StringIO()):cli.main()
        self.assertEqual(clock_output.read_bytes(),self.f.paths[1].read_bytes())
