"""Typed preparation shares durable claims/resources with Q3; no provider runs."""
import base64
import hashlib
import json
from pathlib import Path
import sys
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

    def test_dependency_artifacts_cross_host_and_shared_cell_is_once(self):
        ex=self.executor(capacity=2)
        cell=self.submit(ex,'compile','shared-cell')
        duplicate=self.submit(ex,'compile','shared-cell')
        self.assertEqual(cell,duplicate)
        auditreq=self.request('audit','audit')
        auditreq['preparation']['dependency_inputs']={cell:{'receipt.json':'compiled/receipt.json'}}
        audit=ex.submit_preparation(auditreq,approved=True,budget=8,dependencies=[cell])
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
            runtime_requirements={'version_info_prefix':[3],'packages':{},'tools':{}})
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
            request_sha256=challenge['request_sha256'],source_manifest=ex.source_manifest,input_hashes=[],input_artifacts={},
            profile=preparation_profile_core(selected),views={self.source:str(ROOT)},
            runtime={'executable':sys.executable,'version':sys.version,'version_info':list(sys.version_info),'packages':{}},tools={})
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
