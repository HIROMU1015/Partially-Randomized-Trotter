"""Typed preparation shares durable claims/resources with Q3; no provider runs."""
import base64
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_rpe_distributed_controller as Base


class PreparationQueue(Base.ControllerFailures):
    # Reuse teardown only, not the original test methods in this class.
    def request(self, stage, name):
        return dict(method='PR_PAULI',route=dict(kind='preparation'),preparation=dict(
            stage=stage,artificial=True,command=['ARTIFICIAL_ONLY',name],outputs=['receipt.json']))

    def submit(self, executor, stage, name, **kwargs):
        return executor.submit_preparation(self.request(stage,name),approved=True,budget=kwargs.pop('budget',8),**kwargs)

    def poll_host(self,executor,host,slots=1):
        return executor.rpc('/v1/poll',dict(host_id=host,slots=slots,capacity=slots,source_sha=self.source,cached_prefixes=[]))

    def prep_complete(self,executor,assignment,body=b'ARTIFICIAL_RECEIPT'):
        sha=hashlib.sha256(body).hexdigest()
        executor.rpc('/v1/artifacts/put',dict(host_id=assignment['host_id'] if 'host_id' in assignment else 'server1',
            blobs={sha:base64.b64encode(body).decode()}))
        outcome=dict(status='PREPARATION_COMPLETE',request_id=assignment['job_id'],request_sha256=assignment['request_sha256'],stage=assignment['request']['preparation']['stage'],source_manifest=assignment['request']['preparation']['source'],artifacts={'receipt.json':sha})
        files={'OUTCOME.json':outcome,'PROCESS_RETURN.json':dict(returncode=0,signal=None)}
        row=executor.store.row(assignment['job_id'])
        return dict(host_id=row['host_id'],job_id=assignment['job_id'],attempt_id=assignment['attempt_id'],request_sha256=assignment['request_sha256'],
            files={name:base64.b64encode(json.dumps(value).encode()).decode() for name,value in files.items()})

    def test_horizon_round_shares_existing_queue_and_stage_budgets(self):
        from trottertracks.rpe_preparation_queue import submit_horizon_preparation

        executor = self.executor(capacity=3)
        tasks = {}
        for name, stage, axes in (('signal', 'signal', []), ('pair', 'compile', ['X','Y'])):
            request = self.request(stage, name)
            request['original_contract'] = dict(round=13, physical_id='one-fixed-round')
            request['preparation'].update(horizon_round={'m': 13, 'physical_id': 'one-fixed-round'}, price_axes=axes)
            tasks[name] = {'request': request}
        bindings = {cid: {'source_M': 12, 'rounds': {'13': {
            'physical_id': 'one-fixed-round', 'signal': 'signal', 'X': 'pair', 'Y': 'pair'}}}
                    for cid in ('run-candidate-a', 'run-candidate-b')}
        spec = {'horizon_extensions': {'tasks': tasks, 'bindings': bindings}, 'budgets': {'signal': 1, 'compile': 1}}
        runtime = SimpleNamespace(settings=spec, executor=executor, approved=True)
        requests = [dict(source_configuration_id=cid, configuration_id=cid+'-extended', M=13,
                         source_M=12, extension_rounds=[13]) for cid in bindings]
        reports = submit_horizon_preparation(runtime, requests)
        self.assertEqual(reports[0]['jobs'], reports[1]['jobs'])
        jobs = reports[0]['jobs']['13']
        self.assertEqual(jobs['X'], jobs['Y'])
        self.assertEqual(executor.store.preparation_reservations(), {'signal': 1, 'compile': 1})
        for assignment in self.poll_host(executor, 'server1', 2)['assignments']:
            executor.rpc('/v1/complete', self.prep_complete(executor, assignment))
        self.close_one(executor)
        runtime.executor = restarted = self.executor(capacity=3)
        self.assertEqual(submit_horizon_preparation(runtime, requests), reports)
        self.assertEqual(self.poll_host(restarted, 'server1', 2)['assignments'], [])
        self.assertEqual(sum(row['event'] == 'ASSIGN' for row in restarted.store.events()), 2)
        self.assertTrue(all(restarted.store.row(job)['state'] == 'ACCEPTED' for job in set(jobs.values())))

    def test_horizon_signal_can_run_with_missing_or_invalid_price_axis(self):
        from trottertracks.rpe_preparation_queue import submit_horizon_preparation

        executor = self.executor(capacity=1)
        request = self.request('signal', 'extension')
        request['original_contract'] = {'round': 13}
        request['preparation']['horizon_round'] = {'m': 13, 'physical_id': 'physical'}
        runtime = SimpleNamespace(executor=executor, approved=True, settings=dict(
            budgets={'signal': 1}, horizon_extensions=dict(tasks={'signal': {'request': request}},
                bindings={'origin': {'source_M': 12, 'rounds': {'13': {
                    'physical_id': 'physical', 'signal': 'signal', 'Y': 'signal'}}}})))
        extension = dict(source_configuration_id='origin', configuration_id='extended',
                         M=13, source_M=12, extension_rounds=[13])
        report, = submit_horizon_preparation(runtime, [extension])
        self.assertEqual(report['status'], 'HORIZON_PREPARATION_PARTIAL')
        self.assertEqual(set(report['incomplete']['13']), {'X', 'Y'})
        self.assertEqual(set(report['jobs']['13']), {'signal'})
        self.assertEqual(executor.store.preparation_reservations(), {'signal': 1})

    def test_shared_q3_and_preparation_slot_cannot_be_claimed_twice(self):
        ex=self.executor(capacity=1)
        prep=self.submit(ex,'compile','cell')
        q3=self.enqueue(ex,'q3')
        assigned=self.poll_host(ex,'local')['assignments']
        self.assertEqual([a['job_id'] for a in assigned],[prep])
        self.assertEqual(self.poll_host(ex,'server1')['assignments'],[])
        self.assertEqual(ex.store.counts('PR_PAULI'),1)
        self.assertEqual(ex.store.counts('PREPARATION:compile'),1)
        self.assertEqual(ex.store.row(q3.job.name)['state'],'QUEUED')
        self.assertEqual(ex.state.active,1)

    def test_retired_preparation_ownership_preserves_hold_and_releases_capacity(self):
        ex = self.executor(capacity=1)
        prior = self.submit(ex, 'signal', 'prior')
        assignment, = self.poll_host(ex, 'pbs-0')['assignments']
        owner = dict(host_id='pbs-0', job_id=prior, attempt_id=assignment['attempt_id'])
        ex.rpc('/v1/state', dict(owner, state='UNKNOWN', detail={'automatic_retry': False}))
        waiting = self.submit(ex, 'signal', 'independent')
        self.assertEqual(self.poll_host(ex, 'pbs-0')['assignments'], [])
        ex.retired_preparation_attempts.add((prior, assignment['attempt_id']))
        next_assignment, = self.poll_host(ex, 'pbs-0')['assignments']
        self.assertEqual(next_assignment['job_id'], waiting)
        self.assertEqual(self.submit(ex, 'signal', 'prior'), prior)
        self.assertEqual(ex.preparation_result(prior)['status'], 'UNKNOWN')
        self.assertEqual(ex.store.row(prior)['attempt_id'], assignment['attempt_id'])

    def test_dependency_artifacts_cross_host_and_shared_cell_is_once(self):
        ex=self.executor(capacity=2)
        cell=self.submit(ex,'compile','shared-cell')
        duplicate=self.submit(ex,'compile','shared-cell')
        self.assertEqual(cell,duplicate)
        auditreq=self.request('audit','audit')
        auditreq['preparation']['dependency_inputs']={cell:{'receipt.json':'compiled/receipt.json'}}
        auditreq['preparation']['provider'] = 'PF_AUDIT'
        self.assertFalse(ex.store.has_preparation_successor(cell, 'PF_AUDIT'))
        audit=ex.submit_preparation(auditreq,approved=True,budget=8,dependencies=[cell])
        self.assertTrue(ex.store.has_preparation_successor(cell, 'PF_AUDIT'))
        self.assertFalse(ex.store.has_preparation_successor(cell, 'PF_HOLDOUT_AUDIT'))
        assignment=self.poll_host(ex,'local')['assignments'][0]
        self.assertEqual(assignment['job_id'],cell)
        self.assertEqual(self.poll_host(ex,'server1')['assignments'],[])
        completed=self.prep_complete(ex,assignment)
        ex.rpc('/v1/complete',completed)
        ex.rpc('/v1/complete',completed)
        result=ex.preparation_result(cell)
        self.assertEqual(Path(result['artifact_directory'],'receipt.json').read_bytes(),b'ARTIFICIAL_RECEIPT')
        ex.preparation_result(cell)
        follow=self.poll_host(ex,'server1')['assignments'][0]
        self.assertEqual(follow['job_id'],audit)
        self.assertEqual(follow['artifact_manifest'],{'compiled/receipt.json':hashlib.sha256(b'ARTIFICIAL_RECEIPT').hexdigest()})
        self.assertEqual(ex.store.counts('PR_PAULI'),0)
        events=ex.store.events()
        self.assertEqual(sum(e['event']=='ACCEPT' for e in events),1)
        self.assertEqual(sum(e['event']=='ADOPT' for e in events),1)

    def test_accepted_preparation_resume_does_not_assign_or_adopt_twice(self):
        ex=self.executor(capacity=2)
        cell=self.submit(ex,'signal','saved')
        assignment=self.poll_host(ex,'server1')['assignments'][0]
        payload=self.prep_complete(ex,assignment)
        ex.rpc('/v1/complete',payload);ex.preparation_result(cell)
        self.close_one(ex)
        resumed=self.executor(capacity=2)
        self.assertEqual(self.submit(resumed,'signal','saved'),cell)
        self.assertEqual(self.poll_host(resumed,'local')['assignments'],[])
        self.assertEqual(resumed.preparation_result(cell)['status'],'PREPARATION_COMPLETE')
        self.assertEqual(sum(e['event']=='ASSIGN' for e in resumed.store.events()),1)
        self.assertEqual(sum(e['event']=='ADOPT' for e in resumed.store.events()),1)

    def test_real_stage_does_not_inherit_q3_permission_or_readiness(self):
        ex=self.executor(capacity=2)
        request=self.request('compile','real');request['preparation']['artificial']=False
        request['preparation']['source']=ex.source_manifest
        job=ex.submit_preparation(request,approved=True,budget=8)
        self.assertEqual(ex.preparation_result(job)['status'],'HOST_ELIGIBILITY_HOLD')
        self.assertIsNone(ex.store.row(job))
        self.assertEqual(ex.store.counts('PREPARATION:compile'),0)
        self.assertEqual(self.poll_host(ex,'server1')['assignments'],[])

    def test_new_real_request_gets_metadata_challenge_before_reserving(self):
        from copy import deepcopy
        from trottertracks.rpe_preparation_worker import preparation_profile_core,select_preparation_profile
        from trottertracks.rpe_q3_coordinator import digest
        ex=self.executor(capacity=2)
        request=self.request('compile','new-stage-metadata');request['preparation'].update(
            artificial=False,provider='ARTIFICIAL_METADATA_CONTRACT',source=ex.source_manifest,
            runtime_requirements={'version_info_prefix':[3],'packages':{'qiskit':'ARTIFICIAL_METADATA_ONLY'},'tools':{}})
        profile=dict(host_id='server1',root=str(ROOT),interpreter=sys.executable,route_views={},
            preparation_stage='compile',preparation_provider='ARTIFICIAL_METADATA_CONTRACT',
            preparation_adapter_manifest={'commit':self.source,'files':{'ARTIFICIAL_METADATA_ONLY':'0'*64}})
        ex.distributed['preparation_profiles']={'server1':profile}
        job=ex.submit_preparation(request,approved=True,budget=8)
        self.assertIsNone(ex.store.row(job));self.assertIsNone(ex.preparation_result(job))
        self.assertEqual(ex.store.counts('PREPARATION:compile'),0)
        first=self.poll_host(ex,'server1')
        self.assertEqual(first['assignments'],[])
        challenge=first['eligibility_challenges'][0]
        selected=select_preparation_profile(profile,challenge)
        fingerprint=dict(schema='RPE_PREPARATION_CURRENT_HOST_FINGERPRINT_V1',stage='compile',provider='ARTIFICIAL_METADATA_CONTRACT',
            request_sha256=challenge['request_sha256'],source_manifest=ex.source_manifest,input_hashes=[],working_input_hashes=[],required_views={},required_paths={},runtime_paths={},input_artifacts={},
            profile=preparation_profile_core(selected),views={self.source:str(ROOT)},
            runtime={'executable':sys.executable,'version':sys.version,'version_info':list(sys.version_info),'packages':{'qiskit':'ARTIFICIAL_METADATA_ONLY'}},tools={})
        changed=deepcopy(fingerprint);changed['source_manifest']={'commit':'OTHER','files':{}}
        report=dict(host_id='server1',slots=1,capacity=1,source_sha=self.source,host_fingerprints={job:changed})
        self.assertEqual(ex.rpc('/v1/poll',report)['assignments'],[])
        self.assertIsNone(ex.store.row(job))
        self.assertEqual(ex.preparation_result(job)['status'],'HOST_ELIGIBILITY_HOLD')
        report['host_fingerprints']={job:fingerprint}
        assigned=ex.rpc('/v1/poll',report)['assignments']
        self.assertEqual([a['job_id']for a in assigned],[job])
        self.assertEqual(ex.store.counts('PREPARATION:compile'),1)
        self.assertEqual(ex.store.counts('PR_PAULI'),0)
        self.assertEqual(ex.store.row(job)['eligibility']['host_fingerprints']['server1'],digest(fingerprint))

    def test_missing_stage_profile_is_reasoned_hold_without_attempt(self):
        ex=self.executor(capacity=2)
        ex.distributed['preparation_profiles']={'server1':dict(host_id='server1',root=str(ROOT),interpreter=sys.executable)}
        request=self.request('compile','missing-profile');request['preparation'].update(artificial=False,source=ex.source_manifest,provider='MISSING')
        job=ex.submit_preparation(request,approved=True,budget=8)
        reply=self.poll_host(ex,'server1')
        self.assertEqual(reply['assignments'],[]);self.assertEqual(reply['eligibility_challenges'],[])
        result=ex.preparation_result(job,wait=True)
        self.assertEqual(result['status'],'HOST_ELIGIBILITY_HOLD')
        self.assertIn('No verified preparation profile',result['host_reasons']['server1'])
        self.assertIsNone(ex.store.row(job))

    def test_pending_probe_follows_free_slots_and_rotates_without_reserving(self):
        ex=self.executor(capacity='auto')
        provider='ARTIFICIAL_PROBE_BATCH'
        ex.distributed['preparation_profiles']={
            'server1':dict(host_id='server1',root=str(ROOT),interpreter=sys.executable,
                           preparation_stage='signal',preparation_provider=provider)}
        jobs=[]
        for index in range(9):
            request=self.request('signal',str(index))
            request['preparation'].update(artificial=False,source=ex.source_manifest,provider=provider)
            jobs.append(ex.submit_preparation(request,approved=True,budget=8))
        for slots, expected in ((6, jobs[:6]), (2, jobs[6:8]), (0, jobs[8:])):
            reply = self.poll_host(ex, 'server1', slots)
            self.assertEqual([a['job_id'] for a in reply['eligibility_challenges']], expected)
            self.assertEqual(reply['assignments'], [])
        self.assertEqual(ex.store.counts('PREPARATION:signal'),0)

    def _pending_three_host_failures(self, ex, wsl_accepting):
        hosts=['server2','server1','local','local0-wsl']
        ex.distributed['preparation_profiles']={h:dict(host_id=h,root=str(ROOT),interpreter=sys.executable)for h in hosts}
        ex.distributed['host_admission']={'hosts':{'local0-wsl':dict(accepting=wsl_accepting,optional=True)}}
        request=self.request('compile','optional-probe-targets');request['preparation'].update(artificial=False,source=ex.source_manifest,provider='MISSING_PROFILE')
        job=ex.submit_preparation(request,approved=True,budget=8)
        for host in hosts[:3]:self.poll_host(ex,host)
        self.assertEqual(set(ex.preparation_pending[job]['failures']),set(hosts[:3]))
        return job

    def test_paused_optional_unpolled_host_does_not_delay_three_failed_probes(self):
        ex=self.executor(capacity=2)
        job=self._pending_three_host_failures(ex,wsl_accepting=False)
        self.assertNotIn('local0-wsl',ex.hosts)
        self.assertEqual(ex._preparation_probe_targets(),{'server2','server1','local'})
        result=ex.preparation_result(job)
        self.assertIsNotNone(result)
        self.assertEqual(result['status'],'HOST_ELIGIBILITY_HOLD')
        self.assertEqual(result['reason'],'ALL_PARTICIPATING_PREPARATION_HOSTS_FAIL_OR_UNCHECKED')
        self.assertEqual(ex.preparation_result(job,wait=True)['status'],'HOST_ELIGIBILITY_HOLD')
        self.assertIsNone(ex.store.row(job))
        self.assertEqual(ex.store.counts('PREPARATION:compile'),0)
        self.assertEqual(sum(e['event']=='ASSIGN'for e in ex.store.events()),0)

    def test_optional_absent_or_stale_probe_excluded_but_busy_fresh_host_retained(self):
        import time
        ex=self.executor(capacity=2)
        job=self._pending_three_host_failures(ex,wsl_accepting=True)
        self.assertEqual(ex.preparation_result(job)['status'],'HOST_ELIGIBILITY_HOLD')
        # A participating busy host may still establish readiness. It must not
        # be excluded merely because its current free slot count is zero.
        ex.hosts['local0-wsl']=dict(accepting_new_jobs=True,observed_monotonic=time.monotonic(),reported_slots=0,capacity=1)
        self.assertIn('local0-wsl',ex._preparation_probe_targets())
        self.assertIsNone(ex.preparation_result(job))
        ex.hosts['local0-wsl']['observed_monotonic']-=ex.allocation_config['offer_ttl_seconds']+1
        self.assertNotIn('local0-wsl',ex._preparation_probe_targets())
        self.assertEqual(ex.preparation_result(job,wait=True)['status'],'HOST_ELIGIBILITY_HOLD')
        self.assertIsNone(ex.store.row(job))

    def test_pbs_preparation_probe_waits_for_allocation_without_reserving(self):
        import time
        from unittest.mock import Mock, patch

        ex = self.executor(capacity=2)
        request = self.request('signal', 'pbs-readiness')
        request['preparation'].update(artificial=False, source=ex.source_manifest, provider='DF_SIGNALS')
        ex.distributed['preparation_profiles'] = {'pbs-0': {}}
        ex.distributed['host_admission'] = {'hosts': {'pbs-0': dict(accepting=True, optional=True)}}
        with patch.object(ex, 'pbs_manager', Mock(configured={'pbs-0': {}},
                settings={'task_seconds': {'signal': 120}})) as pbs, patch.object(
                ex, '_pending_preparation_assignment', return_value={}):
            pbs.preparation_readiness_wait_seconds.return_value = 80
            job = ex.submit_preparation(request, approved=True, budget=1)
            self.assertIsNone(ex.preparation_result(job))
            self.assertIsNone(ex.store.row(job))
            self.assertEqual(ex.store.counts('PREPARATION:signal'), 0)
            ex.preparation_pending[job]['probe_started_epoch'] = time.time() - 30
            self.assertIsNone(ex.preparation_result(job))
            ex.preparation_pending[job]['probe_started_epoch'] = time.time() - 90
            result = ex.preparation_result(job)
            self.assertEqual(result['host_reasons'], {'pbs-0': 'PBS_PREPARATION_READINESS_WAIT_EXPIRED'})

    def test_joining_optional_host_during_metrics_uses_consistent_snapshot(self):
        import threading
        ex=self.executor(capacity=2)
        self.poll_host(ex,'server1')
        inspecting=threading.Event();release=threading.Event();joining=threading.Event();joined=threading.Event()
        original=ex._offer_free;results=[];errors=[]
        def offer(host,policy,error=None):
            if threading.current_thread().name=='PREPARATION_METRICS_FIXTURE':
                inspecting.set()
                if not release.wait(3):raise AssertionError('Fixture did not release metrics')
            return original(host,policy,error)
        ex._offer_free=offer
        def measure():
            try:results.append(ex.metrics())
            except Exception as error:errors.append(error)
        def join():
            joining.set()
            try:self.poll_host(ex,'local0-wsl')
            except Exception as error:errors.append(error)
            finally:joined.set()
        measuring=threading.Thread(target=measure,name='PREPARATION_METRICS_FIXTURE')
        new_host=threading.Thread(target=join)
        try:
            measuring.start();self.assertTrue(inspecting.wait(2))
            new_host.start();self.assertTrue(joining.wait(2))
            # Without the controller condition lock, joining mutates the live
            # host iterator at this exact point and raises RuntimeError.
            joined.wait(.1)
        finally:
            release.set();measuring.join(3)
            if new_host.ident is not None:new_host.join(3)
        self.assertFalse(measuring.is_alive());self.assertFalse(new_host.is_alive())
        self.assertEqual(errors,[])
        self.assertEqual(set(results[0]['host_offers']),{'server1'})
        self.assertIn('local0-wsl',ex.hosts)

    def test_inspection_only_request_never_reserves_or_launches(self):
        ex=self.executor(capacity=2)
        request=self.request('compile','inspection-only')
        request['preparation']['dry_check_only']=True
        job=ex.submit_preparation(request,approved=True,budget=8)
        result=ex.preparation_result(job,wait=True)
        self.assertEqual(result['reason'],'INSPECTION_ONLY_REQUEST_NOT_EXECUTABLE')
        self.assertEqual(result['calls'],0);self.assertEqual(result['preparation_calls'],0)
        self.assertIsNone(ex.store.row(job))
        self.assertEqual(ex.store.counts('PREPARATION:compile'),0)
        self.assertEqual(self.poll_host(ex,'server1')['assignments'],[])
        self.assertEqual(sum(e['event']=='ASSIGN'for e in ex.store.events()),0)

    def test_normal_shot_search_returns_existing_hold_for_failed_preparation(self):
        from fractions import Fraction
        from trottertracks import rpe_method_optimizer as O
        ex=self.executor(capacity=2)
        blocked=self.submit(ex,'compile','outside-stage-budget',budget=0)
        ex.preparation_dependencies['not-prepared']=[blocked]
        candidate=dict(method='PR_PAULI',configuration_id='not-prepared',name='ARTIFICIAL_NOT_PREPARED',ready=True,caps=[1]*13,prices=[Fraction(1)]*26)
        knowledge={}
        result=O.run_method('PR_PAULI',[candidate],knowledge,ex.for_method('PR_PAULI'),
            dict(large_step_headroom='9/10',decrement_steps=[1],seed_rules=[],max_steps_per_method=2),
            Fraction(1),lambda *args:None,1,science_approved=True)
        self.assertEqual(result['stop_reason'],'NO_FEASIBLE_POINT_IN_RANGE')
        self.assertEqual(result['new_calls'],0);self.assertEqual(result['trace'],[])
        self.assertEqual(knowledge,{})
        self.assertEqual(result['held'][0]['reason'],'CANDIDATE_PREPARATION_NOT_COMPLETE')
        self.assertEqual(result['held'][0]['configuration_id'],'not-prepared')
        self.assertEqual(ex.store.counts('PR_PAULI'),0)
        self.assertEqual(sum(e['event']in('ASSIGN','ACCEPT','ADOPT')for e in ex.store.events()),0)

    def test_explicit_stage_budget_does_not_consume_q3_budget(self):
        ex=self.executor(capacity=2)
        one=self.submit(ex,'compile','one',budget=1)
        two=self.submit(ex,'compile','two',budget=1)
        self.assertIsNotNone(ex.store.row(one));self.assertIsNone(ex.store.row(two))
        self.assertEqual(ex.preparation_result(two)['status'],'PREPARATION_BUDGET_OR_PERMISSION_HOLD')
        self.assertEqual(ex.store.counts('PR_PAULI'),0)

    def test_failed_dependency_returns_hold_without_new_assignment(self):
        ex=self.executor(capacity=2)
        one=self.submit(ex,'compile','no-permission',budget=0)
        request=self.request('audit','requires-one')
        two=ex.submit_preparation(request,approved=True,budget=8,dependencies=[one])
        result=ex.preparation_result(two,wait=True)
        self.assertEqual(result['status'],'PREPARATION_DEPENDENCY_HOLD')
        self.assertEqual(self.poll_host(ex,'server1')['assignments'],[])


# Base fixture's existing tests already have their own coverage; inherited
# methods here would rerun them needlessly.
for _name in tuple(vars(Base.ControllerFailures)):
    if _name.startswith('test_') and _name not in vars(PreparationQueue):
        setattr(PreparationQueue,_name,None)

if __name__=='__main__':unittest.main()
